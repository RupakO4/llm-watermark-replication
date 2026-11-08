#!/usr/bin/env python3
"""Build the prompt sets (with provenance and a held-out HUMAN continuation).

Two protocols
-------------
``suffix`` (replication, the paper's protocol, Sec. 6 "Datasets and Prompts"):
    a C4 realnewslike document is tokenised and truncated to --max_context tokens;
    the last T tokens are held out as the human baseline and the rest is the prompt.

``prefix`` (new data, used for every source including C4 for a like-for-like
comparison):  after cleaning, the text starts at the first sentence-like
boundary; the first --prompt_tokens tokens are the prompt and the next
--human_tokens tokens are the human continuation.  A fixed prompt length keeps
the text *domain* as the only thing that varies between sources.

Every kept prompt records the source dataset, config, split, revision, row index,
licence, document metadata and a SHA-256 of the prompt, and every rejected
document is counted by reason, so the preprocessing is fully auditable
(``data/<name>_report.json``).

Examples
--------
  python src/build_prompts.py --set replication --per_source 100
  python src/build_prompts.py --set multidomain --per_source 30
  python src/build_prompts.py --set multidomain --per_source 2 --name smoke   # quick test
  python src/build_prompts.py --list_sources
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import random
import re
import statistics
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import REPO_ROOT, ids_to_str, load_tokenizer, dump_json  # noqa: E402

ETHICON_COMMIT = "2a4ff0df3c1bdfc04a69547d29285dde1d7cba6f"
ETHICON_URL = (f"https://raw.githubusercontent.com/coastalcph/EthiCon/{ETHICON_COMMIT}"
               f"/data/acl_22_23_data.csv")

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# kind: "hf" (streamed from the Hugging Face Hub), "hf_json" (raw jsonl on the Hub),
#       "ethicon" (CSV from GitHub), "stdlib" (local Python standard library).

SOURCES = {
    "c4": dict(
        kind="hf", hf="allenai/c4", config="realnewslike", split="train", text="text",
        meta=("url", "timestamp"), domain="news", subdomain="web news (paper corpus)",
        licence="ODC-BY 1.0 (C4); Common Crawl terms of use",
        cite="raffel2020t5", redistribute=True),
    "wikipedia": dict(
        kind="hf", hf="wikimedia/wikipedia", config="20231101.en", split="train", text="text",
        meta=("id", "title", "url"), domain="encyclopedic", subdomain="Wikipedia articles",
        licence="CC BY-SA 4.0 / GFDL", cite="wikipedia2023", redistribute=True),
    "cc_news": dict(
        kind="hf", hf="vblagoje/cc_news", config=None, split="train", text="text",
        meta=("title", "url", "date", "domain"), domain="news", subdomain="CC-News articles",
        licence="Common Crawl terms of use; article copyright with publishers",
        cite="hamborg2017newsplease", redistribute=False),
    "gutenberg": dict(
        kind="hf", hf="sedthh/gutenberg_english", config=None, split="train", text="TEXT",
        meta=("METADATA",), domain="literary", subdomain="Project Gutenberg books",
        licence="Public domain in the US (Project Gutenberg)", cite="gutenberg",
        redistribute=True, skip_chars=5000, buffer=64),     # each row is a whole book
    "imdb": dict(
        kind="hf", hf="stanfordnlp/imdb", config=None, split="train", text="text",
        meta=("label",), domain="review", subdomain="IMDb film reviews",
        licence="Research use (Maas et al. 2011); review copyright with authors",
        cite="maas2011imdb", redistribute=False, informal=True),
    "reddit": dict(
        kind="hf", hf="webis/tldr-17", config=None, split="train", text="content",
        revision="refs/convert/parquet", meta=("subreddit", "id"), domain="social",
        subdomain="Reddit posts (Webis-TLDR-17)", licence="CC BY 4.0 (dataset card); Reddit content policy",
        cite="volske2017tldr", redistribute=True, informal=True),
    "amazon": dict(
        kind="hf_json",
        data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Electronics.jsonl",
        hf="McAuley-Lab/Amazon-Reviews-2023", config="raw/review_categories/Electronics", split="train",
        text="text", title="title", meta=("parent_asin", "rating", "timestamp"), domain="review",
        subdomain="Amazon Electronics reviews", licence="Research use (McAuley Lab); no explicit licence",
        cite="hou2024amazon", redistribute=False, informal=True),
    "stackexchange": dict(
        kind="hf", hf="HuggingFaceH4/stack-exchange-preferences", config=None, split="train",
        text="answers", meta=("qid", "metadata"), domain="technical Q&A",
        subdomain="Stack Exchange answers", licence="CC BY-SA 4.0 (Stack Exchange)",
        cite="h4stackexchange", redistribute=True, buffer=500),
    "acl_ethics": dict(
        kind="ethicon", hf=f"github.com/coastalcph/EthiCon@{ETHICON_COMMIT[:7]}", config="acl_22_23_data.csv",
        split="all", text="ethical_statement", meta=("idx", "year"), domain="scientific",
        subdomain="ACL ethics statements (EthiCon)",
        licence="CC BY 4.0 (ACL Anthology papers since 2016); EthiCon repo states no licence",
        cite="karamolegkou2025ethicon", redistribute=True),
    "python_code": dict(
        kind="stdlib", hf="local Python standard library", config=None, split=None, text=None,
        meta=("path", "python_version"), domain="code", subdomain="Python stdlib source",
        licence="Python Software Foundation License", cite="python", redistribute=True),
}

SETS = {
    "replication": dict(sources=["c4"], protocol="suffix"),
    "multidomain": dict(sources=["c4", "wikipedia", "cc_news", "gutenberg", "imdb", "reddit",
                                 "amazon", "stackexchange", "acl_ethics", "python_code"],
                        protocol="prefix"),
}

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
USER_RE = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]{3,}")
TAG_RE = re.compile(r"<[^>]+>")
PRE_RE = re.compile(r"<pre[^>]*>.*?</pre>", re.S | re.I)
WS_RE = re.compile(r"[ \t]{2,}")
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
# A deliberately small list: used only to keep explicit/abusive informal posts out of a
# dataset that will be published together with model continuations of it.
UNSAFE_TERMS = re.compile(r"\b(nsfw|porn\w*|rape\w*|nudes?|fuck\w*|cunt\w*|whore\w*|slut\w*)\b", re.I)
UNSAFE_SUBREDDITS = re.compile(r"(nsfw|gonewild|sex|porn|nude)", re.I)


def clean_document(raw, key, cfg, stats):
    """Return cleaned text or (None, reason).  `stats` collects counts of edits made."""
    if not isinstance(raw, str):
        return None, "not_text"
    t = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")

    if key in ("imdb", "amazon"):
        t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
        t = html.unescape(t)
    if key == "stackexchange":
        n_pre = len(PRE_RE.findall(t))
        stats["code_blocks_removed"] += n_pre
        t = PRE_RE.sub("\n", t)
        t = re.sub(r"</p>|<br\s*/?>|</li>", "\n", t, flags=re.I)
        t = TAG_RE.sub("", t)
        t = html.unescape(t)
    if key == "acl_ethics":
        # PDF extraction artefacts: hyphenated line breaks and hard wraps.
        n_hy = len(re.findall(r"\w-\n\w", t))
        stats["hyphenation_joins"] += n_hy
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        t = re.sub(r"\s*\n\s*", " ", t)
        if re.match(r"^\s*(ethics|ethical)\s+(statement|considerations?)\s*[:.]?", t, re.I):
            t = re.sub(r"^\s*(ethics|ethical)\s+(statement|considerations?)\s*[:.]?\s*", "", t, flags=re.I)
            stats["heading_removed"] += 1
    if key == "gutenberg":
        m = GUTENBERG_START.search(t)
        if m:
            t = t[m.end():]
        skip = cfg.get("skip_chars", 0)
        if len(t) < skip + 2000:
            return None, "too_short_chars"
        t = t[skip:]          # skip title page / table of contents
        t = re.sub(r"\s*\n\s*", " ", t)   # undo hard line-wrapping of the e-texts
        stats["unwrapped"] += 1

    is_code = key == "python_code"
    if not is_code:
        t, n1 = EMAIL_RE.subn("[EMAIL]", t)
        t, n2 = URL_RE.subn("[URL]", t)
        t, n3 = PHONE_RE.subn("[PHONE]", t)
        t, n4 = USER_RE.subn("u/[USER]", t)
        stats["pii_replaced"] += n1 + n2 + n3 + n4
        if t.count("[EMAIL]") + t.count("[PHONE]") > 10:
            return None, "pii_heavy"
        if t.count("[URL]") > 5:
            return None, "link_farm"
        t = WS_RE.sub(" ", t)          # (spaces are meaningful in code, so not there)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    if len(t) < 600:
        return None, "too_short_chars"
    head = t[:2000]
    if not is_code:
        letters = [c for c in head if c.isalpha()]
        if len(letters) / max(1, len(head)) < 0.55:
            return None, "low_letter_ratio"
        if letters and sum(c.isascii() for c in letters) / len(letters) < 0.95:
            return None, "non_english_script"
        if head.count("|") > 40 or head.count("==") > 20:
            return None, "markup_heavy"
    else:
        if head.count('"""') + head.count("'''") > 6 and len(head.splitlines()) < 30:
            return None, "docstring_heavy"
    if cfg.get("informal") and UNSAFE_TERMS.search(t):
        return None, "unsafe_content"
    return t, None


def choose_span(text, tok, protocol, args, key):
    """Return dict with prompt/human ids + text, or None if too short."""
    if protocol == "suffix":
        ids = tok(text, add_special_tokens=False, truncation=True,
                  max_length=args.max_context - 1)["input_ids"]
        T = args.replication_T
        if len(ids) < args.min_prompt_tokens + T:
            return None
        p_ids, h_ids = ids[:-T], ids[-T:]
    else:
        if key != "python_code":
            m = re.search(r"[A-Z][^\n]{40,}?[.!?](\s|$)", text[:4000])
            if m:
                text = text[m.start():]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        need = args.prompt_tokens + args.human_tokens
        if len(ids) < need:
            return None
        p_ids, h_ids = ids[:args.prompt_tokens], ids[args.prompt_tokens:need]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    prompt = tok.decode(p_ids, skip_special_tokens=True)
    if len(prompt.strip()) < 80:
        return None
    return dict(prompt=prompt, human=tok.decode(h_ids, skip_special_tokens=True),
                prompt_ids=bos + p_ids, human_ids=h_ids, doc_tokens=len(ids))


# ---------------------------------------------------------------------------
# Iterators (each yields (row_index, raw_text, meta_dict))
# ---------------------------------------------------------------------------

def _meta(row, fields):
    out = {}
    for f in fields:
        v = row.get(f)
        if isinstance(v, (list, tuple)):
            v = v[:3]
        if isinstance(v, str) and len(v) > 200:
            v = v[:200]
        out[f] = v
    return out


def iter_hf(key, cfg, seed, buffer):
    from datasets import load_dataset
    kw = dict(split=cfg["split"], streaming=True)
    if cfg.get("revision"):
        kw["revision"] = cfg["revision"]
    attempts = []
    if cfg["config"]:
        attempts.append(lambda: load_dataset(cfg["hf"], cfg["config"], **kw))
    else:
        attempts.append(lambda: load_dataset(cfg["hf"], **kw))
        if cfg.get("revision"):     # parquet branch may expose the config as "default"
            attempts.append(lambda: load_dataset(cfg["hf"], "default", **kw))
            kw2 = {k: v for k, v in kw.items() if k != "revision"}
            attempts.append(lambda: load_dataset(cfg["hf"], **kw2))
    last = None
    for a in attempts:
        try:
            ds = a()
            break
        except Exception as e:      # noqa: BLE001
            last = e
    else:
        raise last
    ds = ds.shuffle(seed=seed, buffer_size=cfg.get("buffer", buffer))
    for i, row in enumerate(ds):
        if key == "stackexchange":
            answers = row.get("answers") or []
            if not answers:
                yield i, None, {}
                continue
            best = max(answers, key=lambda a: (a.get("pm_score") or 0, a.get("selected") or False))
            meta = {"qid": row.get("qid"), "answer_id": best.get("answer_id"),
                    "pm_score": best.get("pm_score"),
                    "site": (row.get("metadata") or [""])[0][:120]}
            yield i, best.get("text"), meta
        elif key == "reddit":
            sub = str(row.get("subreddit") or "")
            if UNSAFE_SUBREDDITS.search(sub):
                yield i, "__UNSAFE_SUBREDDIT__", {"subreddit": sub}
                continue
            yield i, row.get(cfg["text"]), _meta(row, cfg["meta"])
        else:
            yield i, row.get(cfg["text"]), _meta(row, cfg["meta"])


def iter_hf_json(key, cfg, seed, buffer):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=cfg["data_files"], split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer)
    for i, row in enumerate(ds):
        body = row.get(cfg["text"]) or ""
        title = row.get(cfg.get("title", ""), "") or ""
        text = (title.strip() + ". " + body) if title and not body.startswith(title) else body
        yield i, text, _meta(row, cfg["meta"])


def iter_ethicon(cache_dir, seed):
    cache = Path(cache_dir) / f"ethicon_{ETHICON_COMMIT[:7]}.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {ETHICON_URL}")
        with urllib.request.urlopen(ETHICON_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    rows = list(csv.DictReader(io.StringIO(cache.read_text(encoding="utf-8"))))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for i in order:
        r = rows[i]
        yield i, r.get("ethical_statement"), {"idx": r.get("idx"), "year": r.get("year"),
                                              "concerns_annotator1": r.get("annotation_1")}


def iter_stdlib(seed):
    import sysconfig
    root = Path(sysconfig.get_paths()["stdlib"])
    files = []
    for p in sorted(root.rglob("*.py")):
        parts = set(p.parts)
        if parts & {"test", "tests", "idlelib", "site-packages", "lib2to3", "__pycache__", "turtledemo"}:
            continue
        if p.name.startswith("test_"):
            continue
        try:
            if p.stat().st_size < 4000:
                continue
        except OSError:
            continue
        files.append(p)
    random.Random(seed).shuffle(files)
    for i, p in enumerate(files):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # skip the licence / module docstring header: start at the first def/class/import block
        m = re.search(r"^(def |class |import |from |@)", text, re.M)
        if m:
            text = text[m.start():]
        yield i, text, {"path": str(p.relative_to(root)), "python_version": sys.version.split()[0]}


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------

def harvest(key, protocol, args, tok):
    cfg = SOURCES[key]
    code = {"c4": "C4", "wikipedia": "WIK", "cc_news": "CCN", "gutenberg": "GUT", "imdb": "IMD",
            "reddit": "RED", "amazon": "AMZ", "stackexchange": "STX", "acl_ethics": "ACL",
            "python_code": "PYC"}[key]
    want = args.per_source
    rejected, stats = Counter(), Counter()
    kept, seen, scanned, example = [], set(), 0, None
    print(f"  {key}: streaming {cfg['hf']} ...", flush=True)
    try:
        if cfg["kind"] == "hf":
            it = iter_hf(key, cfg, args.seed, args.shuffle_buffer)
        elif cfg["kind"] == "hf_json":
            it = iter_hf_json(key, cfg, args.seed, args.shuffle_buffer)
        elif cfg["kind"] == "ethicon":
            it = iter_ethicon(REPO_ROOT / "data" / "cache", args.seed)
        else:
            it = iter_stdlib(args.seed)
        max_scan = args.max_scan * (5 if key in ("amazon", "reddit", "imdb") else 1)
        for idx, raw, meta in it:
            if len(kept) >= want or scanned >= max_scan:
                break
            scanned += 1
            if raw == "__UNSAFE_SUBREDDIT__":
                rejected["unsafe_subreddit"] += 1
                continue
            text, why = clean_document(raw, key, cfg, stats)
            if text is None:
                rejected[why] += 1
                continue
            span = choose_span(text, tok, protocol, args, key)
            if span is None:
                rejected["too_few_tokens"] += 1
                continue
            h = hashlib.sha256(span["prompt"][:200].encode()).hexdigest()
            if h in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(h)
            if example is None and isinstance(raw, str):
                example = {"raw": raw[:400], "clean": text[:400]}
            kept.append({
                "prompt_id": f"{code}{len(kept) + 1:03d}",
                "source": key, "domain": cfg["domain"], "subdomain": cfg["subdomain"],
                "protocol": protocol,
                "prompt": span["prompt"], "human": span["human"],
                "prompt_ids": ids_to_str(span["prompt_ids"]), "human_ids": ids_to_str(span["human_ids"]),
                "prompt_tokens": len(span["prompt_ids"]), "human_tokens": len(span["human_ids"]),
                "doc_tokens": span["doc_tokens"],
                "source_dataset": cfg["hf"], "source_config": cfg["config"] or "",
                "source_split": cfg["split"] or "", "source_revision": cfg.get("revision", ""),
                "source_row_index": idx, "source_doc_meta": json.dumps(meta, ensure_ascii=False, default=str),
                "licence": cfg["licence"], "redistributable": int(cfg["redistribute"]),
                "prompt_sha256": hashlib.sha256(span["prompt"].encode()).hexdigest()[:16],
                "tokenizer": args.tokenizer,
            })
    except Exception as e:      # noqa: BLE001  -- one broken source must not kill the run
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  !! {key}: stopped after {len(kept)} prompts ({msg})")
        rejected["source_error"] += 1
        stats["error"] = msg
    rep = {
        "source": key, "dataset": cfg["hf"], "config": cfg["config"], "licence": cfg["licence"],
        "domain": cfg["domain"], "scanned": scanned, "kept": len(kept), "rejected": dict(rejected),
        "edits": {k: v for k, v in stats.items()},
        "median_doc_tokens": statistics.median([r["doc_tokens"] for r in kept]) if kept else None,
        "example": example,
    }
    flag = "" if len(kept) >= want else f"   WARNING: wanted {want}"
    print(f"  {key:14s} kept {len(kept):3d}/{want} scanned {scanned:6d}  rejected {dict(rejected)}{flag}")
    return kept, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), default="multidomain")
    ap.add_argument("--sources", nargs="+", default=None, help="override the set's source list")
    ap.add_argument("--name", default=None, help="output stem (default: the set name)")
    ap.add_argument("--per_source", type=int, default=30)
    ap.add_argument("--prompt_tokens", type=int, default=50)
    ap.add_argument("--human_tokens", type=int, default=150)
    ap.add_argument("--replication_T", type=int, default=200)
    ap.add_argument("--min_prompt_tokens", type=int, default=50)
    ap.add_argument("--max_context", type=int, default=512,
                    help="suffix protocol: truncate documents to this many tokens (paper: model max)")
    ap.add_argument("--tokenizer", default="facebook/opt-350m")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_buffer", type=int, default=2000)
    ap.add_argument("--max_scan", type=int, default=4000)
    ap.add_argument("--list_sources", action="store_true")
    args = ap.parse_args(argv)

    if args.list_sources:
        for k, c in SOURCES.items():
            print(f"{k:14s} {c['domain']:14s} {c['hf']}  [{c['licence']}]")
        return

    set_cfg = SETS[args.set]
    sources = args.sources or set_cfg["sources"]
    protocol = set_cfg["protocol"]
    name = args.name or args.set
    tok = load_tokenizer(args.tokenizer)
    print(f"Building '{name}' ({protocol} protocol), {args.per_source} per source: {sources}")

    rows, reports = [], []
    for key in sources:
        kept, rep = harvest(key, protocol, args, tok)
        rows += kept
        reports.append(rep)
    if not rows:
        raise SystemExit("No prompts harvested - check network access to huggingface.co")

    out = REPO_ROOT / "data" / f"prompts_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    settings = {k: v for k, v in vars(args).items() if k != "list_sources"}
    dump_json({"settings": settings, "protocol": protocol, "sources": reports},
              REPO_ROOT / "data" / f"prompts_{name}_report.json")
    by = defaultdict(int)
    for r in rows:
        by[r["source"]] += 1
    print(f"Wrote {out} ({len(rows)} prompts: {dict(by)})")


if __name__ == "__main__":
    main()
