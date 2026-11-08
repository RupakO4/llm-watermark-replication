#!/usr/bin/env python3
"""Make the public copy of the data and results (release/), respecting source licences.

Text from openly licensed sources (C4 ODC-BY, Wikipedia and Stack Exchange CC BY-SA,
Reddit dump and ACL Anthology CC BY, Gutenberg public domain, Python PSF) is kept.
For CC-News, IMDb and Amazon the original human text is removed and only its
identifiers (dataset, split, row index, SHA-256 of the prompt) are kept;
`src/build_prompts.py` with the recorded seed rebuilds it. Our own model outputs
(the generated continuations) are kept for every source and are clearly labelled
as machine-generated.

  python src/make_release.py          # -> release/{data,results,annotation}
  python src/analysis.py --root release
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import REPO_ROOT  # noqa: E402

RESTRICTED = {"cc_news", "imdb", "amazon"}
REDACTED = "[REDACTED: source licence does not permit redistribution; rebuild with src/build_prompts.py]"


def copy_csv(src, dst, fix):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", newline="") as f, open(dst, "w", encoding="utf-8", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=r.fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in r:
            w.writerow(fix(row))


def fix_prompts(row):
    if row.get("source") in RESTRICTED:
        for k in ("prompt", "human"):
            row[k] = REDACTED
        for k in ("prompt_ids", "human_ids"):
            row[k] = ""
    return row


def fix_texts(row):
    if row.get("source") in RESTRICTED and row.get("condition") == "human":
        row["text"] = REDACTED
        if "token_ids" in row:
            row["token_ids"] = ""
    return row


def main():
    out = REPO_ROOT / "release"
    if out.exists():
        shutil.rmtree(out)
    for p in sorted((REPO_ROOT / "data").glob("prompts_*")):
        if "smoke" in p.name or p.name.startswith("prompts_t_"):
            continue
        if p.suffix == ".csv":
            copy_csv(p, out / "data" / p.name, fix_prompts)
        else:
            shutil.copy(p, out / "data" / p.name)
    for exp in ("replication", "multidomain", "multidomain_qwen", "v1_items"):
        d = REPO_ROOT / "results" / exp
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix == ".csv" and p.name in ("generations.csv", "attacks_t5.csv"):
                copy_csv(p, out / "results" / exp / p.name, fix_texts)
            elif p.is_file():
                (out / "results" / exp).mkdir(parents=True, exist_ok=True)
                shutil.copy(p, out / "results" / exp / p.name)
    env = REPO_ROOT / "results" / "environment.json"
    if env.exists():
        shutil.copy(env, out / "results" / "environment.json")
    ann = REPO_ROOT / "annotation"
    if ann.exists():
        (out / "annotation").mkdir(parents=True, exist_ok=True)
        for p in ann.glob("*.csv"):
            if p.name == "items_blind.csv":
                continue        # contains prompt beginnings; rebuilt by make_annotation.py
            shutil.copy(p, out / "annotation" / p.name)
    ann1 = REPO_ROOT / "annotation_v1"     # the 60 rated pilot texts: ids, ratings and scores only, no text
    if ann1.exists():
        (out / "annotation_v1").mkdir(parents=True, exist_ok=True)
        for p in ann1.glob("*.csv"):
            shutil.copy(p, out / "annotation_v1" / p.name)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
