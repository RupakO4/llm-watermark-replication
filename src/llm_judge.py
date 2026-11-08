#!/usr/bin/env python3
"""LLM-as-judge: blind 1-5 quality ratings of continuations.

Perplexity (quality.py) only says how *probable* a text is to another LM; it
rewards repetitive loops, which is exactly how a strongly watermarked small model
fails.  So every continuation is also rated by an instruction-tuned LLM that never
sees the condition (human / unwatermarked / watermarked).  The judge is asked for
a single digit; instead of sampling we read its next-token probabilities over
"1".."5", and store both the most likely rating and the expected rating
(sum_k k * p(k)), which is less noisy.

The same rubric is used for the human annotation (annotate.html), so the judge
can be validated against a person with weighted Cohen's kappa (analysis.py).

  python src/llm_judge.py --experiment multidomain --prompts data/prompts_multidomain.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import (REPO_ROOT, CsvAppender, Progress, done_keys, dump_json, load_hf,  # noqa: E402
                       pick_device, read_csv, resolve_dtype)

RUBRIC = """You are grading the quality of a text continuation.

Below is the BEGINNING of a document, followed by a CONTINUATION. Rate only the CONTINUATION, on this scale:
1 = very poor: garbled, mostly repetition loops, or unrelated to the beginning
2 = poor: frequent errors or repetition, hard to follow
3 = fair: understandable but with noticeable problems (some repetition, odd or off-topic sentences)
4 = good: fluent and coherent, with minor issues
5 = excellent: fluent, coherent, consistent with the beginning, as good as skilled human writing

Consider fluency, coherence with the beginning, and repetitiveness. Do not judge factual accuracy, and do not reward or penalise style or topic. The texts may stop mid-sentence; that is expected.

BEGINNING:
\"\"\"{prompt}\"\"\"

CONTINUATION:
\"\"\"{continuation}\"\"\"

Answer with a single digit from 1 to 5."""


def shorten_prompt(text, max_words=150):
    w = text.split()
    return text if len(w) <= max_words else "... " + " ".join(w[-max_words:])


def select_texts(out_dir, deltas):
    seen, out = set(), []
    for r in read_csv(out_dir / "generations.csv"):
        if r["modification"] != "none" or r["text_id"] in seen:
            continue
        if r["condition"] == "watermarked" and deltas is not None and float(r["delta"]) not in deltas:
            continue
        seen.add(r["text_id"])
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--deltas", type=float, nargs="*", default=None,
                    help="watermark strengths to judge (default: all)")
    ap.add_argument("--gamma", type=float, default=None, help="only this gamma (default: all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    args = ap.parse_args(argv)

    import torch
    torch.set_grad_enabled(False)
    out_dir = REPO_ROOT / "results" / args.experiment
    prompts = {p["prompt_id"]: p["prompt"] for p in read_csv(args.prompts)}
    rows = select_texts(out_dir, args.deltas)
    if args.gamma is not None:
        rows = [r for r in rows if not r["gamma"] or float(r["gamma"]) == args.gamma]
    path = out_dir / "judge.csv"
    done = done_keys(path, "text_id")
    rows = [r for r in rows if r["text_id"] not in done]
    dump_json(dict(vars(args), rubric=RUBRIC), out_dir / "judge_settings.json")
    print(f"Judging {len(rows)} texts with {args.judge}")
    if not rows:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = pick_device(args.device)
    dtype = resolve_dtype(args.dtype, dev)
    tok = AutoTokenizer.from_pretrained(args.judge)
    model = load_hf(AutoModelForCausalLM, args.judge, dev, dtype)
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[-1] for k in range(1, 6)]
    print(f"Judge on {dev} ({dtype}); digit token ids {digit_ids}")

    w = CsvAppender(path, ["text_id", "judge", "p1", "p2", "p3", "p4", "p5", "judge_expected", "judge_argmax"])
    prog = Progress(len(rows), "judge")
    buf = []
    for i, r in enumerate(rows):
        msg = RUBRIC.format(prompt=shorten_prompt(prompts[r["prompt_id"]]), continuation=r["text"].strip())
        chat = [{"role": "user", "content": msg}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        logits = model(input_ids=enc["input_ids"].to(dev)).logits[0, -1].float()
        p = torch.softmax(logits[digit_ids], dim=-1).tolist()
        exp = sum((k + 1) * pk for k, pk in enumerate(p))
        buf.append(dict(text_id=r["text_id"], judge=args.judge,
                        **{f"p{k + 1}": round(pk, 4) for k, pk in enumerate(p)},
                        judge_expected=round(exp, 4), judge_argmax=1 + max(range(5), key=lambda k: p[k])))
        if len(buf) >= 20:
            w.write(buf)
            buf = []
        if i % 50 == 0:
            prog.n = i
            prog.step()
    w.write(buf)
    w.close()


if __name__ == "__main__":
    main()
