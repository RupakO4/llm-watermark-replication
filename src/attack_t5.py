#!/usr/bin/env python3
"""T5 span-replacement attack (paper Sec. 7.1 / Appendix "Details of the T5 Span Attack").

Paper procedure: tokenise the watermarked text, then repeatedly (1) mask one random
word, (2) ask T5-Large for k=20 candidate fillers with a 50-way beam search, and
(3) replace the word with the first candidate that differs from the original,
until eps*T words have been replaced.  We follow that loop with two speed-ups
for a laptop, both recorded in the output: several masks (non-adjacent words, each
in its own copy of the context window) are sent to T5 as one batch, and the beam
width defaults to 20 (k=20 candidates are still returned).  Budgets are applied
cumulatively: the eps=0.3 text continues from the eps=0.1 text.

Only the attack model runs on GPU/MPS; detection stays on CPU (see wm_common).

  python src/attack_t5.py --experiment replication --prompts data/prompts_replication.csv \
      --budgets 0.1 0.3 0.5 0.7
  python src/attack_t5.py --experiment multidomain --prompts data/prompts_multidomain.csv \
      --budgets 0.1 0.3 0.5 --limit_per_source 10
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import (REPO_ROOT, CsvAppender, Progress, Watermarker, distinct_n, done_keys,  # noqa: E402
                       dump_json, load_hf, load_tokenizer, pick_device, read_csv, str_to_ids)

PIECE_RE = re.compile(r"\S+|\s+")
HAS_LETTER = re.compile(r"[A-Za-z]")


def parse_fill(decoded: str) -> str:
    s = decoded.replace("<pad>", "").replace("</s>", "")
    if "<extra_id_0>" in s:
        s = s.split("<extra_id_0>", 1)[1]
    s = re.split(r"<extra_id_\d+>", s)[0]
    return s.strip()


class SpanAttacker:
    def __init__(self, model_name, device, num_beams=5, k=5, window=15, batch=4, max_new_tokens=8):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = load_hf(AutoModelForSeq2SeqLM, model_name, device, dtype=torch.float32)
        self.device, self.num_beams, self.k = device, max(num_beams, k), k
        self.window, self.batch, self.max_new = window, batch, max_new_tokens
        self.calls = 0

    def _fill(self, contexts):
        torch = self.torch
        self._t0 = time.time()
        enc = self.tok(contexts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = self.model.generate(input_ids=enc["input_ids"].to(self.device),
                                      attention_mask=enc["attention_mask"].to(self.device), num_beams=self.num_beams, num_return_sequences=self.k,
                                      max_new_tokens=self.max_new, do_sample=False, early_stopping=True)
        self.calls += 1
        if self.calls <= 3:
            print(f"    T5 call {self.calls}: batch {len(contexts)}, {self.num_beams} beams, "
                  f"{time.time() - self._t0:.1f}s on {self.device}", flush=True)
        dec = self.tok.batch_decode(out, skip_special_tokens=False)
        return [[parse_fill(d) for d in dec[i * self.k:(i + 1) * self.k]] for i in range(len(contexts))]

    def attack(self, text, budgets, rng):
        """Return {budget: (text, realised_fraction)} applying replacements cumulatively."""
        pieces = PIECE_RE.findall(text)
        words = [i for i, s in enumerate(pieces) if not s.isspace() and HAS_LETTER.search(s)]
        n_words = max(1, sum(1 for s in pieces if not s.isspace()))
        targets = {b: math.ceil(b * n_words) for b in sorted(budgets)}
        replaced, tried = set(), set()
        snapshots = {}
        order = words[:]
        rng.shuffle(order)
        for b in sorted(budgets):
            while len(replaced) < targets[b]:
                cand = [i for i in order if i not in tried]
                if not cand:
                    break
                chosen = []
                for i in cand:
                    if all(abs(i - j) > 2 for j in chosen):
                        chosen.append(i)
                    if len(chosen) >= min(self.batch, targets[b] - len(replaced)):
                        break
                ctxs = []
                for i in chosen:
                    lo, hi = max(0, i - 2 * self.window), min(len(pieces), i + 2 * self.window + 1)
                    ctxs.append("".join(pieces[lo:i]) + "<extra_id_0>" + "".join(pieces[i + 1:hi]))
                fills = self._fill(ctxs)
                for i, cands in zip(chosen, fills):
                    tried.add(i)
                    orig = pieces[i]
                    for c in cands:
                        if c and "<" not in c and c.lower() != orig.lower():
                            pieces[i] = c
                            replaced.add(i)
                            break
            snapshots[b] = ("".join(pieces), len(replaced) / n_words)
        return snapshots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default="facebook/opt-350m", help="generator tokenizer (for detection)")
    ap.add_argument("--attack_model", default="google-t5/t5-base",
                    help="paper: T5-Large; T5-Large ran at ~1 min per call on an 8 GB M1 (swapping), so T5-Base")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--num_beams", type=int, default=5,
                    help="paper: 50 beams, k=20; 5/5 keeps T5-Large inside 8 GB of memory")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="beam search on Apple MPS stalled in our tests (torch 2.13), so CPU by default")
    ap.add_argument("--limit_per_source", type=int, default=None)
    ap.add_argument("--prefix_lengths", type=int, nargs="+", default=[25, 50, 100, 150, 200])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    out_dir = REPO_ROOT / "results" / args.experiment
    gen = read_csv(out_dir / "generations.csv")
    ctx_of = {p["prompt_id"]: str_to_ids(p["prompt_ids"])[-1] for p in read_csv(args.prompts)}
    rows = [r for r in gen if r["condition"] == "watermarked" and r["modification"] == "none"
            and float(r["gamma"]) == args.gamma and float(r["delta"]) == args.delta]
    if args.limit_per_source:
        per, kept = {}, []
        for r in rows:
            if per.get(r["source"], 0) < args.limit_per_source:
                kept.append(r)
                per[r["source"]] = per.get(r["source"], 0) + 1
        rows = kept
    T = int(rows[0]["T"]) if rows else 200
    prefix = [k for k in args.prefix_lengths if k <= T]
    fields = ["row_id", "text_id", "prompt_id", "source", "domain", "condition", "gamma", "delta",
              "modification", "edit_target", "edit_realized", "detector_gamma", "attack_model",
              "num_beams", "k", "t5_calls", "seconds", "scored_tokens", "green_tokens", "green_fraction",
              "z", "p_value", "detected", "scored_tokens_dedup", "z_dedup", "detected_dedup",
              "distinct2"] + [f"z_at_{k}" for k in prefix] + ["text"]
    out_path = out_dir / "attacks_t5.csv"
    done = {k.split("|")[0] for k in done_keys(out_path, "text_id")}
    rows = [r for r in rows if r["prompt_id"] not in done]
    dump_json(vars(args), out_dir / "attack_t5_settings.json")
    print(f"T5 span attack on {len(rows)} texts (gamma={args.gamma}, delta={args.delta}), budgets {args.budgets}")
    if not rows:
        return

    import torch
    torch.set_grad_enabled(False)
    tok = load_tokenizer(args.model)
    wm = Watermarker(tok, [args.gamma], {args.gamma: [args.delta]}, device="cpu")
    dev = pick_device(args.device)
    print(f"Attack model {args.attack_model} on {dev}")
    attacker = SpanAttacker(args.attack_model, dev, args.num_beams, args.k, batch=args.batch)
    if dev.type == "mps":      # measure one call; fall back to CPU if MPS is slower (CUDA always wins)
        probe = ["The weather was <extra_id_0> all week, so we stayed inside."] * args.batch
        t0 = time.time()
        attacker._fill(probe)
        t_acc = time.time() - t0
        attacker.model.to("cpu")
        attacker.device = torch.device("cpu")
        t0 = time.time()
        attacker._fill(probe)
        t_cpu = time.time() - t0
        print(f"probe: {dev} {t_acc:.1f}s vs cpu {t_cpu:.1f}s per call")
        if t_acc < t_cpu:
            attacker.model.to(dev)
            attacker.device = dev
        attacker.calls = 0
    print(f"T5 attacker running on {attacker.device}")
    writer = CsvAppender(out_path, fields)
    prog = Progress(len(rows), "t5")
    for r in rows:
        t0, c0 = time.time(), attacker.calls
        rng = random.Random(f"{args.seed}|{r['prompt_id']}")
        snaps = attacker.attack(r["text"], args.budgets, rng)
        secs = round(time.time() - t0, 1)
        out = []
        for b, (txt, real) in snaps.items():
            sc = wm.score_text(ctx_of[r["prompt_id"]], txt, args.gamma, prefix)
            tid = f"{r['prompt_id']}|watermarked|g{r['gamma']}|d{r['delta']}|t5_span_{b:g}"
            row = dict(row_id=f"{tid}|det{args.gamma}", text_id=tid, prompt_id=r["prompt_id"],
                       source=r["source"], domain=r["domain"], condition="watermarked", gamma=r["gamma"],
                       delta=r["delta"], modification=f"t5_span_{b:g}", edit_target=b,
                       edit_realized=round(real, 4), detector_gamma=args.gamma,
                       attack_model=args.attack_model, num_beams=args.num_beams, k=args.k,
                       t5_calls=attacker.calls - c0, seconds=secs, distinct2=distinct_n(txt, 2), text=txt)
            row.update(sc)
            out.append(row)
        writer.write(out)
        prog.step(f"{r['prompt_id']}  realised " + ", ".join(f"{b:g}->{v[1]:.2f}" for b, v in snaps.items()))
    writer.close()


if __name__ == "__main__":
    main()
