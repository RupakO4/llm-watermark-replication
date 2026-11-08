#!/usr/bin/env python3
"""Generate watermarked / unwatermarked continuations and score them.

For every prompt this produces
  * the held-out HUMAN continuation (negative control on real text),
  * one UNWATERMARKED continuation (negative control on model text),
  * one WATERMARKED continuation per (gamma, delta) in the grid - all sampled
    with the same seed as the unwatermarked one, so they are matched pairs,
  * cheap edit attacks applied to the watermarked text at (--attack_gamma,
    --attack_delta),
and scores every text with the authors' WatermarkDetector (z-test, z > 4),
both counting every token and ignoring repeated bigrams (paper Sec. 4.1).
Unwatermarked and human texts are scored under every gamma, so each
(gamma, delta) row of the paper's Table 2 has its own FPR.

Output: results/<experiment>/generations.csv, one row per (text, detector gamma).

  python src/generate.py --experiment replication --prompts data/prompts_replication.csv \
      --gammas 0.25 0.5 --deltas 1 2 5 --max_new_tokens 200
  python src/generate.py --experiment multidomain --prompts data/prompts_multidomain.csv \
      --gammas 0.5 --deltas 1 2 5 --max_new_tokens 150
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import (REPO_ROOT, CHEAP_ATTACKS, CsvAppender, Progress, Watermarker,  # noqa: E402
                       apply_cheap_attack, attack_label, build_word_pool, distinct_n, done_keys,
                       dump_json, generate, generate_batch, ids_to_str, load_hf, load_tokenizer,
                       read_csv, resolve_dtype, str_to_ids)

FIELDS_BASE = [
    "row_id", "text_id", "prompt_id", "source", "domain", "condition", "gamma", "delta",
    "modification", "edit_target", "edit_realized", "detector_gamma",
    "seed", "model", "device", "temperature", "top_k", "top_p", "repetition_penalty",
    "T", "prompt_tokens", "generated_tokens", "gen_seconds",
    "scored_tokens", "green_tokens", "green_fraction", "z", "p_value", "detected",
    "scored_tokens_dedup", "z_dedup", "detected_dedup", "distinct2", "text", "token_ids",
]


def text_id(prompt_id, condition, gamma="", delta="", mod="none"):
    return f"{prompt_id}|{condition}|g{gamma}|d{delta}|{mod}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default="facebook/opt-350m")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--deltas", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--attack_gamma", type=float, default=0.5)
    ap.add_argument("--attack_delta", type=float, default=2.0)
    ap.add_argument("--no_attacks", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_force_length", action="store_true")
    ap.add_argument("--model_device", default="cpu",
                    help="device for the LM forward pass (green lists are always drawn on CPU)")
    ap.add_argument("--dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"],
                    help="LM weights; float16 on a GPU")
    ap.add_argument("--batch_prompts", type=int, default=1,
                    help="prompts per generate() call; >1 batches every condition of several prompts "
                         "(GPU). 1 = the laptop path, one text at a time.")
    ap.add_argument("--seeding_scheme", default="simple_1")
    ap.add_argument("--z_threshold", type=float, default=4.0)
    ap.add_argument("--prefix_lengths", type=int, nargs="+", default=[25, 50, 100, 150, 200])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_per_source", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args(argv)

    prefix = [k for k in args.prefix_lengths if k <= args.max_new_tokens]
    fields = FIELDS_BASE[:-2] + [f"z_at_{k}" for k in prefix] + FIELDS_BASE[-2:]
    out_dir = REPO_ROOT / "results" / args.experiment
    out_path = out_dir / "generations.csv"
    dump_json(vars(args), out_dir / "generate_settings.json")

    prompts = read_csv(args.prompts)
    if args.limit_per_source:
        seen, kept = {}, []
        for p in prompts:
            if seen.get(p["source"], 0) < args.limit_per_source:
                kept.append(p)
                seen[p["source"]] = seen.get(p["source"], 0) + 1
        prompts = kept
    for i, p in enumerate(prompts):
        p["_seed"] = args.seed + i

    done = set() if args.no_resume else {k.split("|")[0] for k in done_keys(out_path, "text_id")}
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already done, {len(todo)} to run. "
          f"Grid gamma={args.gammas} delta={args.deltas}, T={args.max_new_tokens}, "
          f"temp={args.temperature} top_k={args.top_k} top_p={args.top_p}")
    if not todo:
        return

    from transformers import AutoModelForCausalLM
    import torch
    torch.set_grad_enabled(False)
    tok = load_tokenizer(args.model)
    from wm_common import pick_device
    dev = pick_device(args.model_device)
    model = load_hf(AutoModelForCausalLM, args.model, dev, resolve_dtype(args.dtype, dev))
    if dev.type != "cpu":      # fall back to CPU if the accelerator cannot run generate()
        try:
            generate(model, tok, str_to_ids(prompts[0]["prompt_ids"])[:20], None, 0, 4, args.temperature)
        except Exception as e:  # noqa: BLE001
            print(f"{dev} generation failed ({type(e).__name__}: {e}); using CPU")
            dev = torch.device("cpu")
            model = model.to(dev).float()
    args.model_device_used = str(dev)
    print(f"LM forward pass on {dev}; green lists and detection on CPU")
    dump_json(vars(args), out_dir / "generate_settings.json")
    wm = Watermarker(tok, args.gammas, {g: args.deltas for g in args.gammas},
                     seeding_scheme=args.seeding_scheme, z_threshold=args.z_threshold, device="cpu")
    pool = build_word_pool(p["prompt"] + " " + p["human"] for p in prompts)
    run_attacks = (not args.no_attacks and args.attack_gamma in args.gammas
                   and args.attack_delta in args.deltas)

    writer = CsvAppender(out_path, fields, resume=not args.no_resume)
    prog = Progress(len(todo), args.experiment)
    if args.batch_prompts > 1:
        B = args.batch_prompts
        for i in range(0, len(todo), B):
            chunk = todo[i:i + B]
            t0 = time.time()
            gens = generate_chunk(chunk, args, tok, model, wm)
            secs = round((time.time() - t0) / max(1, len(gens)), 2)
            for p in chunk:
                writer.write(run_prompt(p, args, tok, model, wm, pool, prefix, run_attacks,
                                        pregen=gens, pregen_secs=secs))
                prog.step(p["prompt_id"])
        writer.close()
        print(f"Wrote {out_path}")
        return
    errors = 0
    for p in todo:
        try:
            rows = run_prompt(p, args, tok, model, wm, pool, prefix, run_attacks)
        except Exception:      # noqa: BLE001
            errors += 1
            print(f"ERROR on {p['prompt_id']}:")
            traceback.print_exc()
            if errors >= 3:
                raise SystemExit("3 errors in a row - stopping (this is a bug, not bad data).")
            continue
        errors = 0
        writer.write(rows)
        prog.step(p["prompt_id"])
    writer.close()
    print(f"Wrote {out_path}")


def generate_chunk(chunk, args, tok, model, wm):
    """All conditions of several prompts in one batch -> {(prompt_id, gamma, delta): ids}.

    Unwatermarked rows have gamma = delta = None.  The batch shares one seed (that of its
    first prompt), so the unwatermarked and watermarked texts of a prompt are still drawn
    from one random stream.
    """
    conds = [(None, None)] + [(g, d) for g in args.gammas for d in args.deltas]
    keys, prompts, procs = [], [], []
    for p in chunk:
        ids = str_to_ids(p["prompt_ids"])
        for g, d in conds:
            keys.append((p["prompt_id"], g, d))
            prompts.append(ids)
            procs.append(None if g is None else wm.processors[(g, d)])
    outs = generate_batch(model, tok, prompts, procs, chunk[0]["_seed"], args.max_new_tokens,
                          args.temperature, args.top_k, args.top_p, args.repetition_penalty,
                          force_length=not args.no_force_length)
    return dict(zip(keys, outs))


def run_prompt(p, args, tok, model, wm, pool, prefix, run_attacks, pregen=None, pregen_secs=""):
    prompt_ids = str_to_ids(p["prompt_ids"])
    ctx = prompt_ids[-1]
    seed = p["_seed"]
    T = args.max_new_tokens
    common = dict(prompt_id=p["prompt_id"], source=p["source"], domain=p["domain"], seed=seed,
                  model=args.model, device=f"lm:{args.model_device_used}/greenlist:cpu", temperature=args.temperature, top_k=args.top_k,
                  top_p=args.top_p, repetition_penalty=args.repetition_penalty, T=T,
                  prompt_tokens=len(prompt_ids))
    rows = []

    def add(tid, condition, gamma, delta, mod, text, ids, det_gammas, target="", realized=0.0, secs=""):
        for g in det_gammas:
            sc = wm.score_ids(ctx, ids, g, prefix) if ids is not None else wm.score_text(ctx, text, g, prefix)
            if not sc:
                continue
            r = dict(common, row_id=f"{tid}|det{g}", text_id=tid, condition=condition, gamma=gamma,
                     delta=delta, modification=mod, edit_target=target, edit_realized=round(realized, 4),
                     detector_gamma=g, generated_tokens=len(ids) if ids is not None else "",
                     gen_seconds=secs, distinct2=distinct_n(text, 2), text=text,
                     token_ids=ids_to_str(ids) if ids is not None else "")
            r.update(sc)
            rows.append(r)

    human_ids = str_to_ids(p["human_ids"])
    add(text_id(p["prompt_id"], "human"), "human", "", "", "none", p["human"], human_ids, args.gammas)

    t0 = time.time()
    if pregen is not None:
        ids = pregen[(p["prompt_id"], None, None)]
    else:
        ids = generate(model, tok, prompt_ids, None, seed, T, args.temperature, args.top_k, args.top_p,
                       args.repetition_penalty, force_length=not args.no_force_length)
    add(text_id(p["prompt_id"], "unwatermarked"), "unwatermarked", "", "", "none",
        tok.decode(ids, skip_special_tokens=True), ids, args.gammas,
        secs=pregen_secs if pregen is not None else round(time.time() - t0, 1))

    for g in args.gammas:
        for d in args.deltas:
            t0 = time.time()
            if pregen is not None:
                ids = pregen[(p["prompt_id"], g, d)]
            else:
                ids = generate(model, tok, prompt_ids, wm.processors[(g, d)], seed, T, args.temperature,
                               args.top_k, args.top_p, args.repetition_penalty,
                               force_length=not args.no_force_length)
            text = tok.decode(ids, skip_special_tokens=True)
            add(text_id(p["prompt_id"], "watermarked", g, d), "watermarked", g, d, "none", text, ids, [g],
                secs=pregen_secs if pregen is not None else round(time.time() - t0, 1))
            if run_attacks and g == args.attack_gamma and d == args.attack_delta:
                for name, frac in CHEAP_ATTACKS:
                    lab = attack_label(name, frac)
                    rng = random.Random(f"{seed}|{p['prompt_id']}|{lab}")
                    new, realized = apply_cheap_attack(text, name, frac, rng, pool)
                    add(text_id(p["prompt_id"], "watermarked", g, d, lab), "watermarked", g, d, lab,
                        new, None, [g], target=frac if frac is not None else "", realized=realized)
    return rows


if __name__ == "__main__":
    main()
