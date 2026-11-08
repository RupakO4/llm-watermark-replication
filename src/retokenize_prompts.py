#!/usr/bin/env python3
"""Re-tokenise a prompt set for a generator with a different tokenizer (e.g. Qwen2.5).

The prompt CSVs store token ids from the OPT tokenizer. For a model with another
vocabulary, the prompt and the held-out human continuation are re-encoded from
their *text* (the text itself is unchanged, so every model sees the same documents).
The human continuation is cut to --human_tokens tokens of the new tokenizer, so it
is scored over the same number of tokens as the generated text.

  python src/retokenize_prompts.py --src data/prompts_multidomain.csv \
      --dst data/prompts_multidomain_qwen.csv --tokenizer Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import ids_to_str, load_tokenizer, read_csv  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--human_tokens", type=int, default=150)
    args = ap.parse_args(argv)

    tok = load_tokenizer(args.tokenizer)
    rows = read_csv(args.src)
    out = []
    for r in rows:
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        h_ids = tok(r["human"], add_special_tokens=False)["input_ids"][: args.human_tokens]
        r = dict(r)
        r["prompt_ids"], r["human_ids"] = ids_to_str(p_ids), ids_to_str(h_ids)
        r["prompt_tokens"], r["human_tokens"] = len(p_ids), len(h_ids)
        r["human"] = tok.decode(h_ids, skip_special_tokens=True)
        r["tokenizer"] = args.tokenizer
        out.append(r)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {args.dst} ({len(out)} prompts, tokenizer {args.tokenizer})")


if __name__ == "__main__":
    main()
