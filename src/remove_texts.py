#!/usr/bin/env python3
"""Delete the text of screened-out model outputs, keeping their scores.

All generated continuations were screened for sexual content (keyword search over every
output, then manual reading of each hit in context). The outputs listed in
data/removed_texts.csv were judged inappropriate to keep; this script replaces their
text with "[removed: <reason>]" and blanks their token ids in every results file
(generations.csv and attacks_t5.csv, including all edited copies of a removed text).
Scores computed before removal (z, perplexity, judge ratings) are kept, so every number
in the report still reproduces. Run it on the raw results before src/make_release.py.

  python src/remove_texts.py              # on this repository
  python src/remove_texts.py --root DIR   # on another copy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import REPO_ROOT  # noqa: E402

csv.field_size_limit(10**9)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--list", default=None, help="default: <root>/data/removed_texts.csv")
    args = ap.parse_args(argv)
    root = Path(args.root)
    rules = list(csv.DictReader(open(args.list or root / "data" / "removed_texts.csv", encoding="utf-8")))
    for exp in sorted({r["experiment"] for r in rules}):
        mine = [r for r in rules if r["experiment"] == exp]
        for name in ("generations.csv", "attacks_t5.csv"):
            p = root / "results" / exp / name
            if not p.exists():
                continue
            with open(p, encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                fields, rows = rd.fieldnames, list(rd)
            n = 0
            for row in rows:
                hit = next((r for r in mine if row["text_id"].startswith(r["text_id_prefix"])), None)
                if hit and row.get("condition") != "human":
                    row["text"] = f"[removed: {hit['reason']}]"
                    if "token_ids" in row:
                        row["token_ids"] = ""
                    n += 1
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
                w.writeheader()
                w.writerows(rows)
            print(f"{exp}/{name}: removed the text of {n} rows")


if __name__ == "__main__":
    main()
