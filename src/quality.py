#!/usr/bin/env python3
"""Text-quality and entropy measurements.

1. Oracle perplexity (paper Sec. 6 and App. A.2): a larger model from the same
   family scores each continuation *conditioned on its prompt*; PPL is
   exp(mean token loss over the continuation).  The paper used OPT-2.7B to judge
   OPT-1.3B; on an 8 GB laptop we use OPT-1.3B to judge OPT-350M (same 2-4x ratio).
2. Generator entropy: along each watermarked text (at --gamma/--delta) and each
   unwatermarked text, the generator's own next-token distribution at the
   sampling temperature is used to compute the mean Shannon entropy and the
   paper's *spike entropy* (Def. 4.1, z = (1-g)(e^d - 1) / (1 - g + e^d g)),
   which Theorem 4.2 links to how many green tokens a watermark can produce.

Neither model touches the green lists, so both may run on GPU/MPS.

  python src/quality.py --experiment multidomain --prompts data/prompts_multidomain.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import (REPO_ROOT, CsvAppender, Progress, done_keys, dump_json, load_hf,  # noqa: E402
                       load_tokenizer, pick_device, read_csv, resolve_dtype, str_to_ids)


def collect_texts(out_dir, include_attacks):
    texts = {}
    for r in read_csv(out_dir / "generations.csv"):
        if r["modification"] != "none":
            continue
        texts.setdefault(r["text_id"], r)
    t5 = out_dir / "attacks_t5.csv"
    if include_attacks and t5.exists():
        for r in read_csv(t5):
            texts.setdefault(r["text_id"], r)
    return texts


def cont_ids(r, tok):
    if r.get("token_ids"):
        return str_to_ids(r["token_ids"])
    return tok(r["text"], add_special_tokens=False)["input_ids"]


def fit_context(p_ids, c_ids, model):
    """Drop the oldest prompt tokens if prompt + continuation exceed the model's context."""
    limit = getattr(model.config, "max_position_embeddings", None) or 10**9
    keep = max(1, limit - len(c_ids) - 2)
    return p_ids[-keep:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--oracle", default="facebook/opt-1.3b")
    ap.add_argument("--generator", default="facebook/opt-350m")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--no_attacks", action="store_true")
    ap.add_argument("--skip_entropy", action="store_true")
    args = ap.parse_args(argv)

    import torch
    torch.set_grad_enabled(False)
    out_dir = REPO_ROOT / "results" / args.experiment
    dump_json(vars(args), out_dir / "quality_settings.json")
    prompts = {p["prompt_id"]: str_to_ids(p["prompt_ids"]) for p in read_csv(args.prompts)}
    texts = collect_texts(out_dir, not args.no_attacks)
    dev = pick_device(args.device)
    dtype = resolve_dtype(args.dtype, dev)

    # ---- 1. oracle perplexity --------------------------------------------------
    ppl_path = out_dir / "quality_ppl.csv"
    done = done_keys(ppl_path, "text_id")
    todo = [t for t in texts if t not in done]
    print(f"Oracle PPL: {len(todo)} texts with {args.oracle} on {dev} ({dtype})")
    if todo:
        from transformers import AutoModelForCausalLM
        tok = load_tokenizer(args.oracle)
        oracle = load_hf(AutoModelForCausalLM, args.oracle, dev, dtype)
        w = CsvAppender(ppl_path, ["text_id", "oracle", "n_tokens", "oracle_nll", "ppl"])
        prog = Progress(len(todo), "ppl")
        buf = []
        for tid in todo:
            r = texts[tid]
            p_ids, c_ids = prompts[r["prompt_id"]], cont_ids(r, tok)
            p_ids = fit_context(p_ids, c_ids, oracle)
            ids = torch.tensor([p_ids + c_ids], device=dev)
            logits = oracle(ids).logits[0, len(p_ids) - 1:-1].float()
            nll = torch.nn.functional.cross_entropy(logits, ids[0, len(p_ids):]).item()
            buf.append(dict(text_id=tid, oracle=args.oracle, n_tokens=len(c_ids),
                            oracle_nll=round(nll, 5), ppl=round(math.exp(nll), 4)))
            if len(buf) >= 20:
                w.write(buf)
                buf = []
            if prog.n % 50 == 0:
                prog.step()
            else:
                prog.n += 1
        w.write(buf)
        w.close()
        del oracle
        if dev.type == "mps":
            torch.mps.empty_cache()

    # ---- 2. generator entropy / spike entropy --------------------------------------
    if args.skip_entropy:
        return
    ent_path = out_dir / "quality_entropy.csv"
    done = done_keys(ent_path, "text_id")
    sel = [t for t, r in texts.items() if t not in done and r["modification"] == "none" and (
        r["condition"] == "unwatermarked" or (r["condition"] == "watermarked" and r["gamma"] and
                                               float(r["gamma"]) == args.gamma and
                                               float(r["delta"]) == args.delta))]
    print(f"Entropy: {len(sel)} texts with {args.generator} on {dev}")
    if not sel:
        return
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(args.generator)
    gen = load_hf(AutoModelForCausalLM, args.generator, dev, torch.float32)
    alpha = math.exp(args.delta)
    zval = (1 - args.gamma) * (alpha - 1) / (1 - args.gamma + alpha * args.gamma)
    w = CsvAppender(ent_path, ["text_id", "generator", "temperature", "spike_gamma", "spike_delta",
                               "mean_entropy", "mean_spike_entropy", "frac_low_entropy"])
    prog = Progress(len(sel), "entropy")
    buf = []
    for tid in sel:
        r = texts[tid]
        p_ids, c_ids = prompts[r["prompt_id"]], cont_ids(r, tok)
        p_ids = fit_context(p_ids, c_ids, gen)
        ids = torch.tensor([p_ids + c_ids], device=dev)
        logits = gen(ids).logits[0, len(p_ids) - 1:-1].float() / args.temperature
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
        spike = (probs / (1 + zval * probs)).sum(-1)
        buf.append(dict(text_id=tid, generator=args.generator, temperature=args.temperature,
                        spike_gamma=args.gamma, spike_delta=args.delta,
                        mean_entropy=round(ent.mean().item(), 4),
                        mean_spike_entropy=round(spike.mean().item(), 4),
                        frac_low_entropy=round((ent < 0.5).float().mean().item(), 4)))
        if len(buf) >= 20:
            w.write(buf)
            buf = []
        if prog.n % 50 == 0:
            prog.step()
        else:
            prog.n += 1
    w.write(buf)
    w.close()


if __name__ == "__main__":
    main()
