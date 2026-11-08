"""Shared helpers for the COMP8240 watermark replication.

Everything that touches the watermark itself goes through the authors' own
implementation (``extended_watermark_processor.py`` from
github.com/jwkirchenbauer/lm-watermarking, pinned in ``setup.sh``).  This module
only adds the glue the experiments need:

* loading models in a way that works across transformers 4.x and 5.x;
* one ``Watermarker`` object that owns a logits processor per (gamma, delta)
  and a raw + repeated-n-gram-ignored detector per gamma;
* scoring helpers that score *all* T generated tokens (the official detector
  needs one token of left context, so we prepend the last prompt token);
* cheap text attacks (deletion, random substitution, dictionary synonyms,
  whitespace) used for robustness tests.

Device rule (found during the project, see report Sec. 3.3): the green list is a
``torch.randperm`` drawn from a seeded ``torch.Generator`` on the model's
device, and CPU / MPS / CUDA generators give different permutations for the same
seed.  Generation and detection therefore both run on CPU here, so the watermark
"key" is portable to any machine.  Models that never touch the green list
(oracle, T5 attacker, LLM judge) may use the GPU/MPS.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")      # conda numpy + pip torch on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy  # noqa: F401,E402  (must be imported before torch on some macOS setups)

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = Path(os.environ.get("LMW_UPSTREAM", REPO_ROOT / "third_party" / "lm-watermarking"))
UPSTREAM_COMMIT = "82922516930c02f8aa322765defdb5863d07a00e"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---------------------------------------------------------------------------
# Upstream import
# ---------------------------------------------------------------------------

def import_upstream():
    """Import the authors' WatermarkLogitsProcessor / WatermarkDetector."""
    if not (UPSTREAM_DIR / "extended_watermark_processor.py").exists():
        raise SystemExit(
            f"Upstream code not found at {UPSTREAM_DIR}.\n"
            f"Run `bash setup.sh` (clones jwkirchenbauer/lm-watermarking @ {UPSTREAM_COMMIT[:7]}),\n"
            f"or set LMW_UPSTREAM=/path/to/lm-watermarking.")
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
    return WatermarkLogitsProcessor, WatermarkDetector


# ---------------------------------------------------------------------------
# Reproducibility / devices / model loading
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import torch
    random.seed(seed)
    numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def pick_device(choice: str = "auto"):
    """'auto' -> cuda, then mps, then cpu.  Never use this for watermark generation."""
    import torch
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device):
    import torch
    if name == "auto":
        return torch.float32 if device.type == "cpu" else torch.float16
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_hf(auto_cls, name: str, device, dtype=None):
    """from_pretrained that works on transformers 4.x (torch_dtype=) and 5.x (dtype=)."""
    kwargs = {}
    if dtype is not None:
        try:
            model = auto_cls.from_pretrained(name, dtype=dtype)
        except TypeError:
            model = auto_cls.from_pretrained(name, torch_dtype=dtype)
    else:
        model = auto_cls.from_pretrained(name, **kwargs)
    return model.to(device).eval()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Watermark processors and detectors
# ---------------------------------------------------------------------------

class Watermarker:
    """Holds one processor per (gamma, delta) and two detectors per gamma.

    Detection is done on token ids with the *last prompt token prepended*, so that
    all T continuation tokens are scored (the simple_1 / LeftHash scheme seeds the
    green list of token t from token t-1).
    """

    def __init__(self, tokenizer, gammas, deltas_by_gamma, seeding_scheme="simple_1",
                 z_threshold=4.0, device="cpu"):
        import torch
        WatermarkLogitsProcessor, WatermarkDetector = import_upstream()
        self.torch = torch
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.seeding_scheme = seeding_scheme
        self.z_threshold = z_threshold
        vocab = list(tokenizer.get_vocab().values())
        self.vocab_size = len(vocab)
        self.processors = {}
        for g in gammas:
            for d in deltas_by_gamma[g]:
                self.processors[(g, d)] = WatermarkLogitsProcessor(
                    vocab=vocab, gamma=g, delta=d, seeding_scheme=seeding_scheme)
        self.detectors = {}
        for g in gammas:
            for dedup in (False, True):
                self.detectors[(g, dedup)] = WatermarkDetector(
                    vocab=vocab, gamma=g, seeding_scheme=seeding_scheme,
                    device=self.device, tokenizer=tokenizer, z_threshold=z_threshold,
                    normalizers=[], ignore_repeated_ngrams=dedup)

    # -- scoring --------------------------------------------------------------
    def score_ids(self, context_id: int, cont_ids, gamma: float, prefix_lengths=()):
        """Score continuation token ids (list[int]).  Returns a flat dict, or {} if too short."""
        torch = self.torch
        ids = [int(context_id)] + [int(t) for t in cont_ids]
        if len(ids) < 3:
            return {}
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        out = {}
        raw = self.detectors[(gamma, False)].detect(tokenized_text=t, return_z_at_T=True)
        out["scored_tokens"] = int(raw["num_tokens_scored"])
        out["green_tokens"] = int(raw["num_green_tokens"])
        out["green_fraction"] = round(float(raw["green_fraction"]), 4)
        out["z"] = round(float(raw["z_score"]), 3)
        out["p_value"] = float(raw["p_value"])
        out["detected"] = int(bool(raw["prediction"]))
        zt = raw.get("z_score_at_T")
        for k in prefix_lengths:
            if zt is not None and len(zt) >= k:
                out[f"z_at_{k}"] = round(float(zt[k - 1]), 3)
        dd = self.detectors[(gamma, True)].detect(tokenized_text=t, return_z_at_T=False)
        out["scored_tokens_dedup"] = int(dd["num_tokens_scored"])
        out["z_dedup"] = round(float(dd["z_score"]), 3)
        out["detected_dedup"] = int(bool(dd["prediction"]))
        return out

    def score_text(self, context_id: int, text: str, gamma: float, prefix_lengths=()):
        """Score a (possibly edited) continuation string: re-tokenise, then score_ids."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.score_ids(context_id, ids, gamma, prefix_lengths)


class CPUGreenList:
    """Wrap the authors' WatermarkLogitsProcessor so the green list is always drawn on CPU.

    The model may run on MPS/CUDA for speed; at each step the context ids and the logits
    are moved to CPU, the authors' processor adds delta to the green tokens (its RNG is
    created on the device of input_ids, i.e. CPU), and the logits go back to the model's
    device.  The resulting text is therefore detectable by a CPU detector on any machine.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, input_ids, scores):
        out = self.processor(input_ids.to("cpu"), scores.to("cpu").float())
        return out.to(scores.device, dtype=scores.dtype)


def generate(model, tokenizer, prompt_ids, processor, seed, max_new_tokens, temperature,
             top_k=0, top_p=1.0, repetition_penalty=1.0, force_length=True, num_beams=1):
    """One continuation.  processor=None -> unwatermarked.  Returns list[int] of new tokens.

    Defaults are the paper's multinomial setting: softmax sampling at temperature 0.7
    with no top-k / top-p truncation (HF's default top_k=50 is switched off).
    With force_length the EOS token is suppressed until T tokens exist, so every
    sample has exactly T tokens (the paper instead keeps T = 200 +/- 5).
    """
    import torch
    from transformers import LogitsProcessorList
    dev = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    attn = torch.ones_like(input_ids)
    procs = LogitsProcessorList([CPUGreenList(processor)] if processor is not None else [])
    set_seed(seed)
    kw = dict(input_ids=input_ids, attention_mask=attn, logits_processor=procs,
              max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if force_length:
        kw["min_new_tokens"] = max_new_tokens
    if num_beams > 1:
        kw.update(do_sample=False, num_beams=num_beams)
    else:
        kw.update(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
                  repetition_penalty=repetition_penalty, num_beams=1)
    with torch.no_grad():
        out = model.generate(**kw)
    return out[0, input_ids.shape[1]:].tolist()


class RowGreenLists:
    """Batched version of CPUGreenList: row b of the batch gets processors[b] (None = no watermark).

    Lets one generate() call produce the unwatermarked text and every (gamma, delta) watermarked
    text for several prompts at once on a GPU, while every green list is still drawn on CPU,
    so the texts are detectable by the same CPU detector as the laptop run.
    """

    def __init__(self, processors):
        self.processors = processors

    def __call__(self, input_ids, scores):
        if all(p is None for p in self.processors):
            return scores
        ids, sc = input_ids.to("cpu"), scores.to("cpu").float()
        for b, proc in enumerate(self.processors):
            if proc is not None:
                sc[b:b + 1] = proc(ids[b:b + 1], sc[b:b + 1])
        return sc.to(scores.device, dtype=scores.dtype)


def generate_batch(model, tokenizer, prompts, processors, seed, max_new_tokens, temperature,
                   top_k=0, top_p=1.0, repetition_penalty=1.0, force_length=True):
    """Sample one continuation per (prompt_ids, processor) row in a single left-padded batch.

    Same sampling settings as generate(); with force_length every row has exactly
    max_new_tokens new tokens.  Returns list[list[int]] in input order.
    """
    import torch
    from transformers import LogitsProcessorList
    dev = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    L = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), L), pad, dtype=torch.long)
    attn = torch.zeros((len(prompts), L), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long)
        attn[i, L - len(p):] = 1
    set_seed(seed)
    kw = dict(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
              logits_processor=LogitsProcessorList([RowGreenLists(processors)]),
              max_new_tokens=max_new_tokens, pad_token_id=pad, do_sample=True,
              temperature=temperature, top_k=top_k, top_p=top_p,
              repetition_penalty=repetition_penalty, num_beams=1)
    if force_length:
        kw["min_new_tokens"] = max_new_tokens
    with torch.no_grad():
        out = model.generate(**kw)
    new = out[:, L:].tolist()
    if not force_length:     # trim at the first EOS
        eos = tokenizer.eos_token_id
        new = [r[:r.index(eos)] if eos in r else r for r in new]
    return new


# ---------------------------------------------------------------------------
# Cheap attacks (pure Python)
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

SYNONYMS = {
    "important": "significant", "large": "big", "big": "large", "small": "little",
    "good": "fine", "process": "procedure", "change": "shift", "team": "group",
    "skill": "ability", "night": "evening", "letter": "note", "many": "several",
    "begin": "start", "start": "begin", "help": "assist", "show": "display",
    "use": "employ", "make": "create", "need": "require", "buy": "purchase",
    "get": "obtain", "end": "finish", "quick": "fast", "hard": "difficult",
    "easy": "simple", "idea": "notion", "problem": "issue", "people": "individuals",
    "house": "home", "huge": "enormous", "happy": "glad", "sad": "unhappy",
    "smart": "clever", "strong": "powerful", "rich": "wealthy", "also": "additionally",
    "but": "however", "because": "since", "very": "extremely", "often": "frequently",
    "however": "nevertheless", "about": "regarding", "answer": "reply", "story": "tale",
    "said": "stated", "says": "states", "new": "recent", "first": "initial",
    "found": "discovered", "think": "believe", "great": "excellent", "bad": "poor",
    "different": "distinct", "several": "various", "year": "twelvemonth",
}

CHEAP_ATTACKS = [
    ("whitespace", None),
    ("delete", 0.1), ("delete", 0.3), ("delete", 0.5),
    ("random_substitute", 0.1), ("random_substitute", 0.3), ("random_substitute", 0.5),
    ("synonym", None),
]


def attack_label(name, frac):
    return name if frac is None else f"{name}_{frac:g}"


def _match_case(original, replacement):
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement.lower()


def _tidy(s):
    s = re.sub(r"[ \t]{2,}", " ", s)
    return re.sub(r"[ \t]+([,.;:!?])", r"\1", s)


def build_word_pool(texts):
    pool = set()
    for t in texts:
        for w in WORD_RE.findall(t):
            if len(w) >= 4 and w.isalpha():
                pool.add(w.lower())
    return sorted(pool)


def apply_cheap_attack(text, name, frac, rng, pool):
    """Return (new_text, realised fraction of words changed)."""
    if name == "whitespace":
        lead = " " if text[:1].isspace() else ""
        return lead + " ".join(text.split()), 0.0
    ms = list(WORD_RE.finditer(text))
    if not ms:
        return text, 0.0
    if name in ("delete", "random_substitute"):
        n = min(len(ms), max(1, round(frac * len(ms))))
        chosen = rng.sample(range(len(ms)), n)
        repl = {i: (None if name == "delete" else _match_case(ms[i].group(), rng.choice(pool)))
                for i in chosen}
    elif name == "synonym":
        repl = {i: _match_case(m.group(), SYNONYMS[m.group().lower()])
                for i, m in enumerate(ms) if m.group().lower() in SYNONYMS}
    else:
        raise ValueError(name)
    out, last = [], 0
    for i, m in enumerate(ms):
        out.append(text[last:m.start()])
        if i in repl:
            if repl[i] is not None:
                out.append(repl[i])
        else:
            out.append(m.group())
        last = m.end()
    out.append(text[last:])
    new = "".join(out)
    if name == "delete":
        new = _tidy(new)
    return new, len(repl) / len(ms)


# ---------------------------------------------------------------------------
# Small text statistics
# ---------------------------------------------------------------------------

def distinct_n(text, n=2):
    w = text.split()
    grams = list(zip(*[w[i:] for i in range(n)]))
    return round(len(set(grams)) / len(grams), 4) if grams else ""


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


# ---------------------------------------------------------------------------
# CSV / logging helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class CsvAppender:
    """Append rows to a CSV with a fixed header; creates the file if needed."""

    def __init__(self, path, fieldnames, resume=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists and resume:
            with open(self.path, encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
            if header != fieldnames:
                raise SystemExit(f"{self.path} has a different header; delete it or pass --no_resume")
            self.fh = open(self.path, "a", encoding="utf-8", newline="")
            self.w = csv.DictWriter(self.fh, fieldnames=fieldnames, restval="", extrasaction="raise")
        else:
            self.fh = open(self.path, "w", encoding="utf-8", newline="")
            self.w = csv.DictWriter(self.fh, fieldnames=fieldnames, restval="", extrasaction="raise")
            self.w.writeheader()

    def write(self, rows):
        self.w.writerows(rows)
        self.fh.flush()

    def close(self):
        self.fh.close()


def done_keys(path, key):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    with open(p, encoding="utf-8", newline="") as f:
        return {r[key] for r in csv.DictReader(f)}


class Progress:
    def __init__(self, total, label=""):
        self.total, self.label, self.t0, self.n = total, label, time.time(), 0

    def step(self, msg=""):
        if (REPO_ROOT / "ABORT").exists():
            raise SystemExit("ABORT file found - stopping (progress so far is saved).")
        deadline = os.environ.get("WM_DEADLINE")       # epoch seconds, set by the notebook
        if deadline and time.time() > float(deadline):
            raise SystemExit("Time budget used up - stopping cleanly (progress so far is saved; rerun to resume).")
        self.n += 1
        el = time.time() - self.t0
        eta = el / self.n * (self.total - self.n)
        print(f"[{self.label} {self.n}/{self.total}] {msg}  elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min",
              flush=True)


def ids_to_str(ids):
    return " ".join(str(int(i)) for i in ids)


def str_to_ids(s):
    return [int(x) for x in s.split()] if s else []


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
