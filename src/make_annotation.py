#!/usr/bin/env python3
"""Build the blind human-annotation set and a self-contained HTML rating tool.

Sampling: --n items from the multi-domain experiment, balanced over four
conditions (human, unwatermarked, watermarked at --delta_low, watermarked at
--delta_high; all at --gamma) and spread evenly over the sources.  Each prompt is
used at most once, so an annotator never compares two continuations of the same
prompt, and the condition is never shown.

Outputs (annotation/):
  annotate.html   open in a browser, rate 1-5 with the SAME rubric the LLM judge
                  uses, then "Download CSV" -> save as annotation/ratings_<name>.csv
  items_blind.csv item_id, beginning, continuation  (what annotators see)
  key.csv         item_id -> text_id, condition, source (do not open before rating)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_judge import RUBRIC, shorten_prompt  # noqa: E402
from wm_common import REPO_ROOT, read_csv  # noqa: E402

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuation quality rating</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d2433;--muted:#5b6475;--card:#fff;--line:#d9d6cc;--acc:#b4441b}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:22px;margin:0 0 4px} .muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
.lab{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.beg{color:var(--muted);white-space:pre-wrap} .cont{white-space:pre-wrap;border-left:3px solid var(--acc);padding-left:12px}
.scale{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.scale label{border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;user-select:none}
.scale input{margin-right:6px} .scale label.on{border-color:var(--acc);background:#fff4ee}
nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
details{margin:8px 0} pre.rub{white-space:pre-wrap;font-size:14px}
input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px}
</style></head><body><main>
<h1>Rate the continuation (1&ndash;5)</h1>
<p class="muted">Each item shows the beginning of a document and a continuation. Rate only the continuation. Some continuations are written by people and some by a language model; you are not told which. Progress is saved in this browser.</p>
<details><summary>Rubric (identical to the LLM judge's)</summary><pre class="rub">__RUBRIC__</pre></details>
<p>Your name / initials: <input type="text" id="who" size="12"></p>
<nav><button id="prev">&larr; Prev</button><span id="pos"></span><button id="next">Next &rarr;</button>
<span class="muted" id="cnt"></span><button class="primary" id="dl">Download CSV</button></nav>
<div class="card"><div class="lab">Beginning</div><div class="beg" id="beg"></div></div>
<div class="card"><div class="lab">Continuation</div><div class="cont" id="cont"></div>
<div class="scale" id="scale"></div>
<p><input type="text" id="note" size="60" placeholder="optional note"></p></div>
<p class="muted">Keys: 1&ndash;5 to rate, &larr;/&rarr; to move.</p>
</main><script>
const ITEMS = __ITEMS__;
const KEY = "wm-annot-v1";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { st = {}; }
st.r = st.r || {}; st.n = st.n || {}; st.i = st.i || 0;
const $ = id => document.getElementById(id);
function save(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
const labels = ["1 very poor","2 poor","3 fair","4 good","5 excellent"];
function render(){
  const it = ITEMS[st.i]; $("beg").textContent = it.beginning; $("cont").textContent = it.continuation;
  $("pos").textContent = (st.i+1) + " / " + ITEMS.length;
  $("cnt").textContent = Object.keys(st.r).length + " rated";
  $("note").value = st.n[it.item_id] || "";
  const s = $("scale"); s.innerHTML = "";
  labels.forEach((l,k) => { const v = k+1; const lab = document.createElement("label");
    if (st.r[it.item_id] === v) lab.className = "on";
    lab.innerHTML = '<input type="radio" name="r" value="'+v+'"'+(st.r[it.item_id]===v?" checked":"")+'>'+l;
    lab.onclick = () => { st.r[it.item_id] = v; save(); render(); }; s.appendChild(lab); });
  $("who").value = st.who || "";
}
$("prev").onclick = () => { if (st.i > 0) { st.i--; save(); render(); } };
$("next").onclick = () => { if (st.i < ITEMS.length-1) { st.i++; save(); render(); } };
$("note").oninput = e => { st.n[ITEMS[st.i].item_id] = e.target.value; save(); };
$("who").oninput = e => { st.who = e.target.value; save(); };
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT" && e.target.type === "text") return;
  if (e.key >= "1" && e.key <= "5") { st.r[ITEMS[st.i].item_id] = +e.key; save(); render(); }
  if (e.key === "ArrowRight") $("next").click(); if (e.key === "ArrowLeft") $("prev").click(); });
$("dl").onclick = () => {
  const esc = s => '"' + String(s).replace(/"/g,'""') + '"';
  let out = "item_id,annotator,rating,note\n";
  ITEMS.forEach(it => { if (st.r[it.item_id]) out += [it.item_id, st.who||"", st.r[it.item_id], st.n[it.item_id]||""].map(esc).join(",") + "\n"; });
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([out], {type:"text/csv"}));
  a.download = "ratings_" + ((st.who||"annotator").replace(/\W+/g,"_")) + ".csv"; a.click();
};
render();
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="multidomain")
    ap.add_argument("--prompts", default=str(REPO_ROOT / "data" / "prompts_multidomain.csv"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta_low", type=float, default=2.0)
    ap.add_argument("--delta_high", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    out_dir = REPO_ROOT / "annotation"
    out_dir.mkdir(exist_ok=True)
    prompts = {p["prompt_id"]: p for p in read_csv(args.prompts)}
    texts = {}
    for r in read_csv(REPO_ROOT / "results" / args.experiment / "generations.csv"):
        if r["modification"] != "none":
            continue
        if r["condition"] == "human":
            c = "human"
        elif r["condition"] == "unwatermarked":
            c = "unwatermarked"
        elif float(r["gamma"]) == args.gamma and float(r["delta"]) == args.delta_low:
            c = f"watermarked_d{args.delta_low:g}"
        elif float(r["gamma"]) == args.gamma and float(r["delta"]) == args.delta_high:
            c = f"watermarked_d{args.delta_high:g}"
        else:
            continue
        texts.setdefault((r["prompt_id"], c), r)
    conds = ["human", "unwatermarked", f"watermarked_d{args.delta_low:g}", f"watermarked_d{args.delta_high:g}"]

    rng = random.Random(args.seed)
    by_src = defaultdict(list)
    for pid, p in prompts.items():
        if all((pid, c) in texts for c in conds):
            by_src[p["source"]].append(pid)
    for v in by_src.values():
        rng.shuffle(v)
    sources = sorted(by_src)
    chosen, k, used = [], 0, defaultdict(int)
    while len(chosen) < args.n and any(by_src.values()):
        si = k % len(sources)
        src = sources[si]
        if by_src[src]:
            pid = by_src[src].pop()
            # each source cycles through the four conditions from its own offset,
            # so conditions are balanced both overall and within every source
            chosen.append((pid, conds[(used[src] + si) % len(conds)]))
            used[src] += 1
        k += 1
    rng.shuffle(chosen)

    items, key = [], []
    for i, (pid, c) in enumerate(chosen, 1):
        r = texts[(pid, c)]
        iid = f"Q{i:03d}"
        items.append(dict(item_id=iid, beginning=shorten_prompt(prompts[pid]["prompt"]),
                          continuation=r["text"].strip()))
        key.append(dict(item_id=iid, text_id=r["text_id"], condition=c, source=r["source"], prompt_id=pid))
    with open(out_dir / "items_blind.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "beginning", "continuation"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(items)
    with open(out_dir / "key.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)
    rub = RUBRIC.split("BEGINNING:")[0].strip()
    page = HTML.replace("__RUBRIC__", html.escape(rub)).replace(
        "__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    (out_dir / "annotate.html").write_text(page, encoding="utf-8")
    counts = defaultdict(int)
    for k_ in key:
        counts[k_["condition"]] += 1
    print(f"Wrote {len(items)} items to annotation/ ({dict(counts)}). Open annotation/annotate.html.")


if __name__ == "__main__":
    main()
