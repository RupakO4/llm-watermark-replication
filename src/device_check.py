#!/usr/bin/env python3
"""Controlled test of the device bug found in the project update.

Texts watermarked on CPU are re-scored by the authors' WatermarkDetector built on
(a) CPU and (b) every accelerator present (MPS / CUDA).  If the green lists were
device-independent the z-scores would match; in practice torch.randperm with the
same seed gives a different permutation on each device, so (b) falls to chance.

  python src/device_check.py --experiment replication --prompts data/prompts_replication.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import REPO_ROOT, Watermarker, dump_json, load_tokenizer, read_csv, str_to_ids  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="replication")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default="facebook/opt-350m")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args(argv)

    import torch
    devices = ["cpu"]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    out_dir = REPO_ROOT / "results" / args.experiment
    ctx = {p["prompt_id"]: str_to_ids(p["prompt_ids"])[-1] for p in read_csv(args.prompts)}
    rows = [r for r in read_csv(out_dir / "generations.csv") if r["condition"] in ("watermarked", "unwatermarked")
            and r["modification"] == "none" and float(r["detector_gamma"]) == args.gamma
            and (r["condition"] == "unwatermarked" or (float(r["gamma"]) == args.gamma and float(r["delta"]) == args.delta))]
    tok = load_tokenizer(args.model)
    res = {"devices": devices, "gamma": args.gamma, "delta": args.delta}
    for dev in devices:
        wm = Watermarker(tok, [args.gamma], {args.gamma: [args.delta]}, device=dev)
        for cond in ("watermarked", "unwatermarked"):
            zs = []
            for r in [x for x in rows if x["condition"] == cond][: args.n]:
                sc = wm.score_ids(ctx[r["prompt_id"]], str_to_ids(r["token_ids"]), args.gamma)
                zs.append(sc["z"])
            res[f"{dev}_{cond}_mean_z"] = sum(zs) / len(zs) if zs else None
            res[f"{dev}_{cond}_detected"] = sum(z > 4 for z in zs)
            res[f"{dev}_{cond}_n"] = len(zs)
    print(res)
    dump_json(res, out_dir / "device_check.json")


if __name__ == "__main__":
    main()
