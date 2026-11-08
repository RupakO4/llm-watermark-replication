#!/usr/bin/env python3
"""All tables, figures and numbers in the report, computed from results/.

  python src/analysis.py                 # reads results/, writes results/tables, results/figures,
                                         # and results/summary.json (every number quoted in the report)
  python src/analysis.py --root release  # same, from the redacted public copy

Detection rule everywhere: z > 4 (one-sided p < 3.2e-5), as in the paper.
TPR is measured on watermarked texts, FPR on unwatermarked model text and on the
held-out human continuations, each scored under the matching gamma.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_common import REPO_ROOT, wilson  # noqa: E402

# Kirchenbauer et al. (2023), Table 2 (multinomial sampling, T=200, ~500 samples, OPT-1.3B)
PAPER_T2 = {(0.5, 1.0): (0.767, 0.504), (0.25, 1.0): (0.729, 0.482), (0.5, 2.0): (0.984, 0.978),
            (0.25, 2.0): (0.994, 0.988), (0.5, 5.0): (0.996, 0.992), (0.25, 5.0): (1.000, 0.998)}
# Table 8 (T5 span attack, multinomial, gamma=0.5, delta=2): TPR@4, TPR@5 after attack
PAPER_T8 = {0.1: (0.819, 0.577), 0.3: (0.353, 0.127), 0.5: (0.094, 0.029), 0.7: (0.039, 0.012)}
SOURCE_LABEL = {"c4": "C4 (paper corpus)", "wikipedia": "Wikipedia", "cc_news": "CC-News",
                "gutenberg": "Gutenberg", "imdb": "IMDb", "reddit": "Reddit", "amazon": "Amazon",
                "stackexchange": "StackExchange", "acl_ethics": "ACL ethics", "python_code": "Python code"}
COLORS = {"human": "#2a78d6", "unwatermarked": "#eb6834", "watermarked": "#1baf7a"}
ATTACK_COLORS = {"delete": "#2a78d6", "random_substitute": "#eb6834", "t5_span": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"

S = {}   # summary numbers
ORACLE_NAME = "oracle"


def pretty_model(name):
    """facebook/opt-1.3b -> OPT-1.3B; Qwen/Qwen2.5-1.5B -> Qwen2.5-1.5B"""
    n = name.split("/")[-1]
    return "OPT-" + n[4:].upper() if n.lower().startswith("opt-") else n


def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load(root, exp):
    d = root / "results" / exp
    if not (d / "generations.csv").exists():
        return None
    g = pd.read_csv(d / "generations.csv", dtype={"gamma": str, "delta": str}, keep_default_na=False)
    g["gamma_f"] = pd.to_numeric(g["gamma"], errors="coerce")
    g["delta_f"] = pd.to_numeric(g["delta"], errors="coerce")
    num(g, ["z", "z_dedup", "scored_tokens_dedup", "green_fraction", "detected", "detected_dedup", "detector_gamma", "edit_realized",
            "distinct2", "gen_seconds"] + [c for c in g if c.startswith("z_at_")])
    out = {"gen": g}
    for name in ("attacks_t5", "quality_ppl", "quality_entropy", "judge"):
        p = d / f"{name}.csv"
        if p.exists():
            t = pd.read_csv(p, dtype={"gamma": str, "delta": str}, keep_default_na=False)
            num(t, [c for c in t if c not in ("text_id", "row_id", "prompt_id", "source", "domain", "condition",
                                              "modification", "text", "oracle", "judge", "generator",
                                              "attack_model", "gamma", "delta")])
            out[name] = t
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def rate(x, thr=4.0):
    x = np.asarray(x, float)
    return float((x > thr).mean()) if len(x) else float("nan")


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_tex(name, body):
    p = OUT_T / f"{name}.tex"
    p.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------

def replication(R):
    g = R["gen"]
    ppl = R.get("quality_ppl")
    pmap = dict(zip(ppl.text_id, ppl.ppl)) if ppl is not None else {}
    rows, S["rep"] = [], {}
    n_prompts = g.prompt_id.nunique()
    S["rep"]["n_prompts"] = int(n_prompts)
    S["rep"]["T"] = int(g["T"].iloc[0])
    S["rep"]["prompt_tokens_median"] = float(g.drop_duplicates("prompt_id").prompt_tokens.median())
    for gam in sorted(g.gamma_f.dropna().unique()):
        neg_u = g[(g.condition == "unwatermarked") & (g.detector_gamma == gam)]
        neg_h = g[(g.condition == "human") & (g.detector_gamma == gam)]
        for dl in sorted(g[(g.gamma_f == gam)].delta_f.dropna().unique()):
            pos = g[(g.condition == "watermarked") & (g.gamma_f == gam) & (g.delta_f == dl) & (g.modification == "none")]
            p_ppl = [pmap.get(t) for t in pos.text_id if t in pmap]
            r = dict(gamma=gam, delta=dl, n=len(pos), tpr4=rate(pos.z), tpr5=rate(pos.z, 5),
                     tpr4_dedup=rate(pos.z_dedup), fpr_unwm=rate(neg_u.z), fpr_human=rate(neg_h.z),
                     fp_unwm=int((neg_u.z > 4).sum()), fp_human=int((neg_h.z > 4).sum()),
                     auc=auc(pos.z, neg_u.z), auc_human=auc(pos.z, neg_h.z), mean_z=float(pos.z.mean()),
                     green=float(pos.green_fraction.mean()),
                     ppl_mean=float(np.mean(p_ppl)) if p_ppl else float("nan"),
                     ppl_median=float(np.median(p_ppl)) if p_ppl else float("nan"),
                     paper_tpr4=PAPER_T2.get((gam, dl), (None, None))[0],
                     paper_tpr5=PAPER_T2.get((gam, dl), (None, None))[1])
            lo, hi = wilson(int((pos.z > 4).sum()), len(pos))
            r["tpr4_lo"], r["tpr4_hi"] = lo, hi
            rows.append(r)
    T = pd.DataFrame(rows)
    S["rep"]["grid"] = T.to_dict("records")
    ent = R.get("quality_entropy")
    if ent is not None:
        w = ent[ent.text_id.str.contains("watermarked|g0.5|d2.0|none", regex=False)]
        u = ent[ent.text_id.str.contains("|unwatermarked|", regex=False)]
        S["rep"]["spike_wm"] = float(w.mean_spike_entropy.mean())
        S["rep"]["spike_unwm"] = float(u.mean_spike_entropy.mean())
    for cond in ("unwatermarked", "human"):
        ids = g[g.condition == cond].text_id.unique()
        v = [pmap[t] for t in ids if t in pmap]
        S["rep"][f"ppl_{cond}_mean"] = float(np.mean(v)) if v else None
        S["rep"][f"ppl_{cond}_median"] = float(np.median(v)) if v else None
        for gam in sorted(g.gamma_f.dropna().unique()):
            sub = g[(g.condition == cond) & (g.detector_gamma == gam)]
            S["rep"][f"mean_z_{cond}_g{gam}"] = float(sub.z.mean())
            S["rep"][f"max_z_{cond}_g{gam}"] = float(sub.z.max())

    lines = [r"\begin{tabular}{@{}cc rrr rr rr r@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{TPR at $z>4$} & \multicolumn{2}{c}{FPR at $z>4$} & \multicolumn{2}{c}{TPR at $z>5$} & \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"$\gamma$ & $\delta$ & ours & [95\% CI] & paper & unwm. & human & ours & paper & PPL \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['gamma']:g} & {r['delta']:g} & {r['tpr4']:.3f} & [{r['tpr4_lo']:.3f}, {r['tpr4_hi']:.3f}] & "
                     f"{fmt(r['paper_tpr4'])} & {r['fpr_unwm']:.3f} & {r['fpr_human']:.3f} & {r['tpr5']:.3f} & "
                     f"{fmt(r['paper_tpr5'])} & {fmt(r['ppl_mean'], 1)} \\\\")
    lines.append(r"\midrule")
    gen_name = pretty_model(R["gen"].model.iloc[0])
    lines.append(f"\\multicolumn{{9}}{{l}}{{Unwatermarked {gen_name} text}} & {fmt(S['rep'].get('ppl_unwatermarked_mean') or float('nan'), 1)} \\\\")
    lines.append(f"\\multicolumn{{9}}{{l}}{{Human continuation (C4)}} & {fmt(S['rep'].get('ppl_human_mean') or float('nan'), 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("rep_grid", "\n".join(lines))
    return T


def replication_attack(R):
    t5 = R.get("attacks_t5")
    if t5 is None:
        return None
    g = R["gen"]
    ppl = R.get("quality_ppl")
    pmap = dict(zip(ppl.text_id, ppl.ppl)) if ppl is not None else {}
    gam, dl = 0.5, 2.0
    neg = g[(g.condition == "unwatermarked") & (g.detector_gamma == gam)].z
    base = g[(g.condition == "watermarked") & (g.gamma_f == gam) & (g.delta_f == dl) & (g.modification == "none")]
    base = base[base.prompt_id.isin(t5.prompt_id.unique())]
    rows = [dict(eps=0.0, realised=0.0, n=len(base), tpr4=rate(base.z), tpr5=rate(base.z, 5), auc=auc(base.z, neg),
                 tpr4_dedup=rate(base.z_dedup),
                 mean_z=float(base.z.mean()),
                 ppl=float(np.mean([pmap[t] for t in base.text_id if t in pmap])) if pmap else float("nan"),
                 paper_tpr4=0.984, paper_tpr5=0.977)]
    for eps in sorted(t5.edit_target.unique()):
        a = t5[t5.edit_target == eps]
        v = [pmap[t] for t in a.text_id if t in pmap]
        rows.append(dict(eps=float(eps), realised=float(a.edit_realized.mean()), n=len(a), tpr4=rate(a.z),
                         tpr4_dedup=rate(a.z_dedup),
                         tpr5=rate(a.z, 5), auc=auc(a.z, neg), mean_z=float(a.z.mean()),
                         ppl=float(np.mean(v)) if v else float("nan"),
                         paper_tpr4=PAPER_T8.get(float(eps), (None, None))[0],
                         paper_tpr5=PAPER_T8.get(float(eps), (None, None))[1]))
    S["rep"]["t5"] = rows
    S["rep"]["t5_seconds_per_text"] = float(t5.drop_duplicates("prompt_id").seconds.mean())
    lines = [r"\begin{tabular}{@{}r r rrr rr r r@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{TPR, $z>4$} & \multicolumn{2}{c}{TPR, $z>5$} & & \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"$\varepsilon$ & real. & ours & dedup & paper & ours & paper & AUC & PPL \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['eps']:.1f} & {r['realised']:.2f} & {r['tpr4']:.2f} & {r['tpr4_dedup']:.2f} & {fmt(r['paper_tpr4'])} & "
                     f"{r['tpr5']:.2f} & {fmt(r['paper_tpr5'])} & {r['auc']:.3f} & {fmt(r['ppl'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("rep_t5", "\n".join(lines))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-domain
# ---------------------------------------------------------------------------

def multidomain(M, key="md", table="md_sources"):
    """Per-source detection/quality table. key: summary section; table: .tex name (None = no table)."""
    g = M["gen"]
    ppl, ent, jd = M.get("quality_ppl"), M.get("quality_entropy"), M.get("judge")
    pmap = dict(zip(ppl.text_id, ppl.ppl)) if ppl is not None else {}
    jmap = dict(zip(jd.text_id, jd.judge_expected)) if jd is not None else {}
    emap = dict(zip(ent.text_id, ent.mean_spike_entropy)) if ent is not None else {}
    hmap = dict(zip(ent.text_id, ent.mean_entropy)) if ent is not None else {}
    gam = 0.5
    S[key] = {"n_prompts": int(g.prompt_id.nunique()), "T": int(g["T"].iloc[0])}
    rows = []
    order = [s for s in SOURCE_LABEL if s in set(g.source)]

    def block(sub_g, label):
        hu = sub_g[(sub_g.condition == "human") & (sub_g.detector_gamma == gam)]
        un = sub_g[(sub_g.condition == "unwatermarked") & (sub_g.detector_gamma == gam)]
        w = {d: sub_g[(sub_g.condition == "watermarked") & (sub_g.gamma_f == gam) & (sub_g.delta_f == d) &
                      (sub_g.modification == "none")] for d in (1.0, 2.0, 5.0)}
        w2 = w[2.0]
        r = dict(source=label, n=int(sub_g.prompt_id.nunique()),
                 spike=float(np.nanmean([emap.get(t, np.nan) for t in w2.text_id])) if emap else float("nan"),
                 entropy=float(np.nanmean([hmap.get(t, np.nan) for t in w2.text_id])) if hmap else float("nan"),
                 tpr_d1=rate(w[1.0].z), tpr_d2=rate(w2.z), tpr_d5=rate(w[5.0].z), tpr_d2_dedup=rate(w2.z_dedup),
                 tpr_d2_T50=rate(w2.z_at_50) if "z_at_50" in w2 else float("nan"),
                 tpr_d2_T100=rate(w2.z_at_100) if "z_at_100" in w2 else float("nan"),
                 z_d2=float(w2.z.mean()), z_d2_median=float(w2.z.median()),
                 fp_human=int((hu.z > 4).sum()), fp_unwm=int((un.z > 4).sum()), n_neg=len(hu),
                 fp_human_dedup=int((hu.z_dedup > 4).sum()), fp_unwm_dedup=int((un.z_dedup > 4).sum()),
                 z_d2_dedup=float(w2.z_dedup.mean()), uniq_d2=float(w2.scored_tokens_dedup.mean())
                 if "scored_tokens_dedup" in w2 else float("nan"),
                 z_human_max=float(hu.z.max()), z_unwm_max=float(un.z.max()),
                 auc_d2=auc(w2.z, pd.concat([hu.z, un.z])),
                 distinct2_unwm=float(un.distinct2.mean()), distinct2_d2=float(w2.distinct2.mean()),
                 distinct2_d5=float(w[5.0].distinct2.mean()), distinct2_human=float(hu.distinct2.mean()))
        for cname, df_ in (("human", hu), ("unwm", un), ("d1", w[1.0]), ("d2", w2), ("d5", w[5.0])):
            pv = [pmap[t] for t in df_.text_id if t in pmap]
            jv = [jmap[t] for t in df_.text_id if t in jmap]
            r[f"ppl_{cname}"] = float(np.median(pv)) if pv else float("nan")
            r[f"judge_{cname}"] = float(np.mean(jv)) if jv else float("nan")
        return r

    for s in order:
        rows.append(block(g[g.source == s], s))
    rows.append(block(g[g.source != "c4"], "all_new"))
    rows.append(block(g, "all"))
    T = pd.DataFrame(rows)
    S[key]["by_source"] = T.to_dict("records")

    # spike entropy vs z correlation (per text)
    if emap:
        from scipy.stats import spearmanr
        w2 = g[(g.condition == "watermarked") & (g.gamma_f == gam) & (g.delta_f == 2.0) & (g.modification == "none")]
        x = np.array([emap.get(t, np.nan) for t in w2.text_id])
        ok = ~np.isnan(x)
        rho, p = spearmanr(x[ok], w2.z.values[ok])
        S[key]["spearman_spike_z"] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
        rho2, p2 = spearmanr(x[ok], w2.z_dedup.values[ok])
        S[key]["spearman_spike_zdedup"] = {"rho": float(rho2), "p": float(p2)}
        tm = T[~T.source.isin(["all", "all_new"])]
        rho3, p3 = spearmanr(tm.spike, tm.tpr_d1)
        S[key]["spearman_source_spike_tpr_d1"] = {"rho": float(rho3), "p": float(p3)}

    lines = [r"\begin{tabular}{@{}l r r rrr r rr rr rrr@{}}", r"\toprule",
             r" & & & \multicolumn{3}{c}{TPR ($z>4$)} & & \multicolumn{2}{c}{$\delta{=}2$ variants} & \multicolumn{2}{c}{FP ($z>4$)} &"
             r" \multicolumn{3}{c}{Judge (1--5)} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-14}",
             r"Source & $n$ & $S$ & $\delta{=}1$ & 2 & 5 & $\bar z_{\delta=2}$ & dedup & $T{=}50$ & hum. & unwm. & hum. & unwm. & $\delta{=}2$ \\",
             r"\midrule"]
    for r in rows:
        lab = {"all_new": r"\textit{All nine new}", "all": r"\textit{All ten}"}.get(r["source"], SOURCE_LABEL.get(r["source"], r["source"]))
        if r["source"] == "all_new":
            lines.append(r"\midrule")
        lines.append(f"{lab} & {r['n']} & {fmt(r['spike'], 2)} & {r['tpr_d1']:.2f} & {r['tpr_d2']:.2f} & {r['tpr_d5']:.2f} & "
                     f"{r['z_d2']:.1f} & {r['tpr_d2_dedup']:.2f} & {fmt(r['tpr_d2_T50'], 2)} & {r['fp_human']}/{r['fp_human_dedup']} & {r['fp_unwm']}/{r['fp_unwm_dedup']} & "
                     f"{fmt(r['judge_human'], 2)} & {fmt(r['judge_unwm'], 2)} & {fmt(r['judge_d2'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if table:
        write_tex(table, "\n".join(lines))
    return T



def modern_compare(M, Q, TM, TQ, names):
    """Same prompts, two generators: detection and quality per source, side by side."""
    lines = [r"\begin{tabular}{@{}l rrrrr rrrrr@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{" + names[0] + r"} & \multicolumn{5}{c}{" + names[1] + r"} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             r"Source & $S$ & TPR$_{\delta=2}$ & dedup & FP & judge & $S$ & TPR$_{\delta=2}$ & dedup & FP & judge \\",
             r"\midrule"]
    qi = {r["source"]: r for r in TQ.to_dict("records")}
    for r in TM.to_dict("records"):
        q = qi.get(r["source"])
        if q is None:
            continue
        lab = {"all_new": r"\textit{All nine new}", "all": r"\textit{All ten}"}.get(r["source"], SOURCE_LABEL.get(r["source"], r["source"]))
        if r["source"] == "all_new":
            lines.append(r"\midrule")
        cells = []
        for x in (r, q):
            fp = x["fp_human"] + x["fp_unwm"]
            fpd = x["fp_human_dedup"] + x["fp_unwm_dedup"]
            cells.append(f"{fmt(x['spike'], 2)} & {x['tpr_d2']:.2f} & {x['tpr_d2_dedup']:.2f} & {fp}/{fpd} & {fmt(x['judge_d2'], 2)}")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("modern_compare", "\n".join(lines))



def judge_check(root):
    """The 60 texts the author rated in v1: author vs the v1 judge (1.5B) and the v2 judge (3B)."""
    from scipy.stats import spearmanr, mannwhitneyu
    from sklearn.metrics import cohen_kappa_score
    a = root / "annotation_v1"
    if not (a / "key.csv").exists() or not (a / "ratings_author.csv").exists():
        return
    df = pd.read_csv(a / "key.csv").merge(pd.read_csv(a / "ratings_author.csv")[["item_id", "rating"]], on="item_id")
    if (a / "excluded_items.csv").exists():          # items removed after screening (see data/removed_texts.csv)
        df = df[~df.item_id.isin(pd.read_csv(a / "excluded_items.csv").item_id)]
    judges = {}
    if (a / "judge_v1model.csv").exists():
        judges["j15"] = pd.read_csv(a / "judge_v1model.csv")
    v2 = root / "results" / "v1_items" / "judge.csv"
    if v2.exists():
        judges["j3"] = pd.read_csv(v2)
    out = {"n": int(len(df))}
    for name, j in judges.items():
        m = df.merge(j[["text_id", "judge_expected", "judge_argmax"]].drop_duplicates("text_id"), on="text_id")
        h, ja = m.rating.astype(int), m.judge_argmax.astype(int)
        nc = m[m.source != "python_code"]
        out[name] = dict(
            n=int(len(m)),
            kq=float(cohen_kappa_score(h, ja, weights="quadratic")),
            kl=float(cohen_kappa_score(h, ja, weights="linear")),
            exact=float((h == ja).mean()), within1=float(((h - ja).abs() <= 1).mean()),
            rho=float(spearmanr(m.judge_expected, h)[0]), p=float(spearmanr(m.judge_expected, h)[1]),
            kq_nocode=float(cohen_kappa_score(nc.rating.astype(int), nc.judge_argmax.astype(int), weights="quadratic")),
            levels_used=int(ja.nunique()), mean=float(m.judge_expected.mean()), n_ones=int((ja == 1).sum()),
            by_cond={c: float(g.judge_expected.mean()) for c, g in m.groupby("condition")})
    out["human_by_cond"] = {c: float(g.rating.mean()) for c, g in df.groupby("condition")}
    out["human_mean"] = float(df.rating.mean())
    g = {c: df.loc[df.condition == c, "rating"] for c in df.condition.unique()}
    out["mw_human_d5"] = float(mannwhitneyu(g["human"], g["watermarked_d5"]).pvalue)
    out["mw_unwm_d2"] = float(mannwhitneyu(g["unwatermarked"], g["watermarked_d2"]).pvalue)
    # a second human rater (any other ratings_*.csv): human-human agreement on the same items
    others = [p for p in sorted(a.glob("ratings_*.csv")) if p.name != "ratings_author.csv"]
    if others:
        o = pd.read_csv(others[0])[["item_id", "rating"]].rename(columns={"rating": "r2"})
        m = df.merge(o, on="item_id")
        h1, h2 = m.rating.astype(int), m.r2.astype(int)
        out["hh"] = dict(n=int(len(m)), kq=float(cohen_kappa_score(h1, h2, weights="quadratic")),
                         kl=float(cohen_kappa_score(h1, h2, weights="linear")),
                         exact=float((h1 == h2).mean()), within1=float(((h1 - h2).abs() <= 1).mean()),
                         rho=float(spearmanr(h1, h2)[0]), mean2=float(h2.mean()),
                         by_cond2={c: float(g.r2.mean()) for c, g in m.groupby("condition")})
        m["avg"] = (h1 + h2) / 2
        g2 = {c: m.loc[m.condition == c, "avg"] for c in m.condition.unique()}
        out["hh"]["by_cond_avg"] = {c: float(v.mean()) for c, v in g2.items()}
        out["hh"]["mw_human_d5"] = float(mannwhitneyu(g2["human"], g2["watermarked_d5"]).pvalue)
        if (a / "ppl_v1oracle.csv").exists():
            mp = m.merge(pd.read_csv(a / "ppl_v1oracle.csv"), on="text_id")
            out["hh"]["rho_ppl_avg"], out["hh"]["p_ppl_avg"] = (float(x) for x in spearmanr(mp.ppl, mp.avg))
            out["hh"]["rho_ppl_r2"] = float(spearmanr(mp.ppl, mp.r2)[0])
        for name, j in judges.items():
            mj = m.merge(j[["text_id", "judge_expected", "judge_argmax"]].drop_duplicates("text_id"), on="text_id")
            out["hh"][f"rho_{name}_r2"] = float(spearmanr(mj.judge_expected, mj.r2)[0])
            out["hh"][f"kq_{name}_r2"] = float(cohen_kappa_score(mj.r2.astype(int), mj.judge_argmax.astype(int), weights="quadratic"))
            out["hh"][f"rho_{name}_avg"] = float(spearmanr(mj.judge_expected, mj.avg)[0])
    if (a / "ppl_v1oracle.csv").exists():
        m = df.merge(pd.read_csv(a / "ppl_v1oracle.csv"), on="text_id")
        out["rho_ppl"], out["p_ppl"] = (float(x) for x in spearmanr(m.ppl, m.rating))
    S["judge_check"] = out



def gap_analysis(R, root, rep_name):
    """Why gamma=0.5, delta=1 falls short of the paper: entropy, position in the text, prompt length."""
    from scipy.stats import spearmanr, wilcoxon, fisher_exact
    from sklearn.linear_model import LogisticRegression
    if "quality_entropy" not in R:
        return
    g = R["gen"]
    e = R["quality_entropy"]
    ent = e[e.text_id.str.contains("|unwatermarked|", regex=False)].assign(
        prompt_id=lambda d: d.text_id.str.split("|").str[0])[["prompt_id", "mean_spike_entropy"]]
    pr = root / "data" / f"prompts_{rep_name}.csv"
    plen = pd.read_csv(pr)[["prompt_id", "prompt_tokens"]] if pr.exists() else None
    out, rows = {}, []
    for gm, d in ((0.5, 1.0), (0.25, 1.0), (0.5, 2.0)):
        x = g[(g.condition == "watermarked") & (g.modification == "none") & (g.gamma_f == gm) & (g.delta_f == d)]
        x = x.drop(columns=[c for c in ("prompt_tokens",) if c in x]).merge(ent, on="prompt_id")
        det = x.z > 4
        terc = pd.qcut(x.mean_spike_entropy, 3, labels=["low", "mid", "high"])
        tpr_t = {t: float(det[terc == t].mean()) for t in ("low", "mid", "high")}
        sd = np.sqrt(gm * (1 - gm))
        g100 = gm * 100 + x.z_at_100 * 10 * sd
        g200 = gm * 200 + x.z_at_200 * np.sqrt(200) * sd
        f1, f2 = g100 / 100, (g200 - g100) / 100
        r = dict(gamma=gm, delta=d, tpr=float(det.mean()), rho=float(spearmanr(x.mean_spike_entropy, x.z)[0]),
                 rho_p=float(spearmanr(x.mean_spike_entropy, x.z)[1]), **{f"tpr_{k}": v for k, v in tpr_t.items()},
                 green1=float(f1.mean()), green2=float(f2.mean()), green_p=float(wilcoxon(f2 - f1).pvalue))
        if plen is not None:
            y = x.merge(plen, on="prompt_id")
            cap = y.prompt_tokens >= y.prompt_tokens.max()
            r.update(n_cap=int(cap.sum()), tpr_cap=float((y.z[cap] > 4).mean()), tpr_short=float((y.z[~cap] > 4).mean()),
                     cap_p=float(fisher_exact([[int((y.z[cap] > 4).sum()), int((y.z[cap] <= 4).sum())],
                                               [int((y.z[~cap] > 4).sum()), int((y.z[~cap] <= 4).sum())]])[1]),
                     rho_len=float(spearmanr(y.prompt_tokens, y.mean_spike_entropy)[0]))
        m = LogisticRegression(C=1e6).fit(x[["mean_spike_entropy"]].values, det.astype(int))
        r["pred_paper"] = float(m.predict_proba([[0.807]])[0, 1])
        r["spike_mean"] = float(x.mean_spike_entropy.mean())
        r["spike_max"] = float(x.mean_spike_entropy.max())
        rows.append(r)
    S["gap"] = rows
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{TPR by entropy third} & \multicolumn{2}{c}{green frac.} & TPR at \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"Setting & TPR & low & mid & high & 1--100 & 101--200 & $S{=}0.807$ \\", r"\midrule"]
    paper = {(0.5, 1.0): 0.767, (0.25, 1.0): 0.729, (0.5, 2.0): 0.984}
    for r in rows:
        lines.append(f"$\\gamma{{=}}{r['gamma']},\\delta{{=}}{int(r['delta'])}$ & {r['tpr']:.3f} & {r['tpr_low']:.2f} & "
                     f"{r['tpr_mid']:.2f} & {r['tpr_high']:.2f} & {r['green1']:.3f} & {r['green2']:.3f} & "
                     f"{r['pred_paper']:.3f} ({paper[(r['gamma'], r['delta'])]}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("gap_table", "\n".join(lines))


def robustness(R, M):
    out = []
    for name, D in (("C4 replication", R), ("Multi-domain", M)):
        if D is None:
            continue
        g = D["gen"]
        sub = g[(g.condition == "watermarked") & (g.gamma_f == 0.5) & (g.delta_f == 2.0)]
        for mod, grp in sub.groupby("modification"):
            kind = "none" if mod == "none" else mod.rsplit("_", 1)[0] if mod[-1].isdigit() else mod
            out.append(dict(dataset=name, attack=kind, modification=mod, realised=float(grp.edit_realized.mean()),
                            tpr4=rate(grp.z), tpr4_dedup=rate(grp.z_dedup), mean_z=float(grp.z.mean()), n=len(grp)))
        t5 = D.get("attacks_t5")
        if t5 is not None:
            base_ids = set(t5.prompt_id)
            b = sub[(sub.modification == "none") & sub.prompt_id.isin(base_ids)]
            out.append(dict(dataset=name, attack="t5_span", modification="t5_span_0", realised=0.0,
                            tpr4=rate(b.z), tpr4_dedup=rate(b.z_dedup), mean_z=float(b.z.mean()), n=len(b)))
            for eps, grp in t5.groupby("edit_target"):
                out.append(dict(dataset=name, attack="t5_span", modification=f"t5_span_{eps:g}",
                                realised=float(grp.edit_realized.mean()), tpr4=rate(grp.z),
                                tpr4_dedup=rate(grp.z_dedup), mean_z=float(grp.z.mean()), n=len(grp)))
    df = pd.DataFrame(out)
    S["robust"] = df.to_dict("records")
    order = ["none", "whitespace", "synonym", "delete_0.1", "delete_0.3", "delete_0.5", "random_substitute_0.1",
             "random_substitute_0.3", "random_substitute_0.5", "t5_span_0.1", "t5_span_0.3", "t5_span_0.5",
             "t5_span_0.7"]
    lab = {"none": "none", "whitespace": "whitespace", "synonym": "synonyms"}
    lines = [r"\begin{tabular}{@{}l rrr rrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{C4 ($T{=}200$)} & \multicolumn{3}{c}{New data ($T{=}150$)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Edit & edited & TPR & $\bar z$ & edited & TPR & $\bar z$ \\", r"\midrule"]
    for m in order:
        cells = []
        for ds in ("C4 replication", "Multi-domain"):
            r = df[(df.dataset == ds) & (df.modification == m)] if len(df) else df
            if len(r) and not (m == "none" and False):
                r = r.iloc[0]
                cells += [f"{r.realised:.2f}", f"{r.tpr4:.2f}", f"{r.mean_z:.1f}"]
            else:
                cells += ["--", "--", "--"]
        if all(c == "--" for c in cells):
            continue
        name = lab.get(m, m.replace("random_substitute", "random subst.").replace("t5_span", "T5 span").replace("_", " "))
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("robust_table", "\n".join(lines))
    return df


# ---------------------------------------------------------------------------
# LLM judge and human agreement
# ---------------------------------------------------------------------------

def judge_quality(R, M):
    from scipy.stats import spearmanr
    S["judge"] = {}
    rows = []
    for name, D in (("replication", R), ("multidomain", M)):
        if D is None or D.get("judge") is None:
            continue
        g, jd, ppl = D["gen"], D["judge"], D.get("quality_ppl")
        base = g[g.modification == "none"].drop_duplicates("text_id")
        base = base.merge(jd[["text_id", "judge_expected", "judge_argmax"]], on="text_id", how="inner")
        if ppl is not None:
            base = base.merge(ppl[["text_id", "ppl"]], on="text_id", how="left")
        base = base[(base.condition != "watermarked") | (base.gamma_f == 0.5)]
        base["cond"] = np.where(base.condition == "watermarked", "wm_d" + base.delta_f.map(lambda d: f"{d:g}"), base.condition)
        for c, grp in base.groupby("cond"):
            lo, hi = np.percentile([grp.judge_expected.sample(frac=1, replace=True, random_state=i).mean()
                                    for i in range(500)], [2.5, 97.5])
            rows.append(dict(exp=name, cond=c, n=len(grp), judge=float(grp.judge_expected.mean()), judge_lo=lo,
                             judge_hi=hi, ppl_median=float(grp.ppl.median()) if "ppl" in grp else float("nan"),
                             distinct2=float(grp.distinct2.mean())))
        if "ppl" in base:
            ok = base.ppl.notna()
            rho, p = spearmanr(base.ppl[ok], base.judge_expected[ok])
            okd = base.distinct2.notna()
            rho_d, p_d = spearmanr(base.distinct2[okd], base.judge_expected[okd])
            okb = ok & okd
            rho_pd, p_pd = spearmanr(base.ppl[okb], base.distinct2[okb])
            S["judge"][name] = {"spearman_ppl_judge": float(rho), "p": float(p),
                                "spearman_distinct2_judge": float(rho_d), "p_d": float(p_d),
                                "spearman_ppl_distinct2": float(rho_pd), "n": int(ok.sum())}
    df = pd.DataFrame(rows)
    S["judge"]["by_condition"] = df.to_dict("records")
    order = ["human", "unwatermarked", "wm_d1", "wm_d2", "wm_d5"]
    lab = {"human": "Human", "unwatermarked": "Unwatermarked", "wm_d1": r"Watermarked $\delta{=}1$",
           "wm_d2": r"Watermarked $\delta{=}2$", "wm_d5": r"Watermarked $\delta{=}5$"}
    lines = [r"\begin{tabular}{@{}l rrr rrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{C4 ($T{=}200$)} & \multicolumn{3}{c}{New data ($T{=}150$)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Condition & PPL & dist-2 & judge & PPL & dist-2 & judge \\", r"\midrule"]
    for c in order:
        cells = []
        for e in ("replication", "multidomain"):
            r = df[(df.exp == e) & (df.cond == c)] if len(df) else df
            if len(r):
                r = r.iloc[0]
                cells += [fmt(r.ppl_median, 1), fmt(r.distinct2, 2), fmt(r.judge, 2)]
            else:
                cells += ["--", "--", "--"]
        lines.append(lab[c] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("judge_quality", "\n".join(lines))
    return df


def agreement(M, root):
    """Human ratings (annotation/ratings_*.csv) vs the LLM judge, weighted Cohen's kappa."""
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score
    adir = root / "annotation"
    key_p = adir / "key.csv"
    files = sorted(adir.glob("ratings_*.csv"))
    if not key_p.exists() or not files or M is None or M.get("judge") is None:
        return None
    key = pd.read_csv(key_p)
    jd = M["judge"][["text_id", "judge_argmax", "judge_expected"]]
    raters = {}
    for f in files:
        r = pd.read_csv(f)
        name = f.stem.replace("ratings_", "")
        raters[name] = r.set_index("item_id").rating
    df = key.merge(jd, on="text_id", how="left")
    for n, s in raters.items():
        df[f"r_{n}"] = df.item_id.map(s)
    S["agreement"] = {"n_items": int(len(df)), "raters": list(raters)}
    pairs = []
    names = list(raters) + ["qwen_judge"]
    df["r_qwen_judge"] = df.judge_argmax
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = df[[f"r_{a}", f"r_{b}"]].dropna()
            if len(m) < 5:
                continue
            x, y = m.iloc[:, 0].astype(int), m.iloc[:, 1].astype(int)
            pairs.append(dict(a=a, b=b, n=len(m), kappa_quadratic=float(cohen_kappa_score(x, y, weights="quadratic")),
                              kappa_linear=float(cohen_kappa_score(x, y, weights="linear")),
                              exact=float((x == y).mean()), within1=float((abs(x - y) <= 1).mean()),
                              spearman=float(spearmanr(x, y)[0])))
    S["agreement"]["pairs"] = pairs
    by = {}
    for n in names:
        by[n] = df.groupby("condition")[f"r_{n}"].mean().to_dict()
    S["agreement"]["mean_rating_by_condition"] = by
    # How each human rater relates to the automatic quality signals on the same 60 items.
    from scipy.stats import mannwhitneyu
    qp = root / "results" / "multidomain" / "quality_ppl.csv"
    if qp.exists():
        df = df.merge(pd.read_csv(qp)[["text_id", "ppl"]].drop_duplicates("text_id"), on="text_id", how="left")
    extra = {}
    for n in raters:
        col = f"r_{n}"
        e = {}
        for sig in ("ppl", "judge_expected"):
            if sig in df:
                m = df[[sig, col]].dropna()
                rho, pv = spearmanr(m[sig], m[col])
                e[f"rho_{sig}"], e[f"p_{sig}"] = float(rho), float(pv)
        g = {c: df.loc[df.condition == c, col].dropna() for c in df.condition.unique()}
        for a, b in (("human", "watermarked_d5"), ("unwatermarked", "watermarked_d2"), ("unwatermarked", "watermarked_d5")):
            if a in g and b in g and len(g[a]) and len(g[b]):
                e[f"mw_{a}_{b}"] = float(mannwhitneyu(g[a], g[b]).pvalue)
        nc = df[df.source != "python_code"][[col, "r_qwen_judge"]].dropna().astype(int)
        e["kq_nocode"] = float(cohen_kappa_score(nc[col], nc["r_qwen_judge"], weights="quadratic"))
        e["n_nocode"] = int(len(nc))
        e["judge_never"] = [int(v) for v in range(1, 6) if v not in set(df.r_qwen_judge.dropna().astype(int))]
        e["code_mean"] = float(df.loc[df.source == "python_code", col].mean())
        extra[n] = e
    S["agreement"]["vs_signals"] = extra
    df.to_csv(OUT_T / "annotation_merged.csv", index=False)
    return pairs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def fig_tradeoff(T, S_rep):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.1, 2.15))
    for gam, mk, col in ((0.25, "s", "#2a78d6"), (0.5, "o", "#eb6834")):
        sub = T[T.gamma == gam].sort_values("delta")
        ax.plot(sub.ppl_mean, sub.mean_z, marker=mk, color=col, ms=5, label=f"$\\gamma={gam:g}$",
                markeredgecolor="white", markeredgewidth=0.8)
        for _, r in sub.iterrows():
            ax.annotate(f"$\\delta$={r.delta:g}", (r.ppl_mean, r.mean_z), textcoords="offset points",
                        xytext=(4, -9) if gam == 0.5 else (-6, 5), ha="left" if gam == 0.5 else "right",
                        fontsize=6.5, color=MUTED)
    u = S_rep.get("ppl_unwatermarked_mean")
    if u:
        ax.plot([u], [S_rep.get("mean_z_unwatermarked_g0.5", 0)], marker="D", color=INK, ms=4, ls="none",
                label="unwatermarked")
    h = S_rep.get("ppl_human_mean")
    if h:
        ax.axvline(h, color=MUTED, ls="--", lw=0.9)
        ax.text(h, ax.get_ylim()[1] * 0.45, "human\ntext PPL ", fontsize=6.5, color=MUTED, va="top", ha="right")
    ax.axhline(4, color=MUTED, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1], 4, "z = 4 ", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel(f"Oracle perplexity ({ORACLE_NAME}), lower is better")
    ax.set_ylabel("Mean z-score ($T=200$)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT_F / "rep_tradeoff.pdf")
    plt.close(fig)


def fig_robust(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.1, 2.15))
    for ds, ls in (("C4 replication", "-"), ("Multi-domain", "--")):
        d = df[df.dataset == ds]
        base = d[d.modification == "none"]
        for kind, lab in (("delete", "word deletion"), ("random_substitute", "random substitution"),
                          ("t5_span", "T5 span replacement")):
            k = d[d.attack == kind].sort_values("realised")
            if not len(k):
                continue
            if kind != "t5_span" and len(base):
                k = pd.concat([base.assign(attack=kind), k])
            ax.plot(k.realised, k.tpr4, ls=ls, marker="o", ms=3.5, color=ATTACK_COLORS[kind],
                    label=f"{lab}" if ds == "C4 replication" else None)
    ax.plot([], [], color=MUTED, ls="-", label="C4 replication")
    ax.plot([], [], color=MUTED, ls="--", label="multi-domain")
    ax.set_xlabel("Fraction of words edited")
    ax.set_ylabel("TPR at $z>4$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=6.3, loc="lower left")
    fig.savefig(OUT_F / "robustness.pdf")
    plt.close(fig)


def fig_sources(M, T):
    import matplotlib.pyplot as plt
    g = M["gen"]
    ent = M.get("quality_entropy")
    emap = dict(zip(ent.text_id, ent.mean_spike_entropy)) if ent is not None else {}
    tt = T[~T.source.isin(["all", "all_new"])].sort_values("spike" if emap else "z_d2")
    srcs = list(tt.source)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.45), gridspec_kw={"width_ratios": [1.45, 1]})
    pos = np.arange(len(srcs))
    for off, (cond, sel, lab) in zip((-0.25, 0, 0.25), (
            ("human", lambda d: d[(d.condition == "human") & (d.detector_gamma == 0.5)], "human"),
            ("unwatermarked", lambda d: d[(d.condition == "unwatermarked") & (d.detector_gamma == 0.5)], "unwatermarked"),
            ("watermarked", lambda d: d[(d.condition == "watermarked") & (d.gamma_f == 0.5) & (d.delta_f == 2.0) &
                                        (d.modification == "none")], r"watermarked $\delta=2$"))):
        data = [sel(g[g.source == s]).z.values for s in srcs]
        bp = a1.boxplot(data, positions=pos + off, widths=0.22, vert=False, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=1.8, mfc=COLORS[cond], mec="none", alpha=0.6),
                        medianprops=dict(color="white", lw=1.1), whiskerprops=dict(color=COLORS[cond], lw=0.8),
                        capprops=dict(color=COLORS[cond], lw=0.8))
        for b in bp["boxes"]:
            b.set(facecolor=COLORS[cond], edgecolor=COLORS[cond], alpha=0.95)
        a1.plot([], [], color=COLORS[cond], lw=5, label=lab)
    a1.axvline(4, color=INK, lw=0.8, ls=":")
    a1.set_yticks(pos)
    a1.set_yticklabels([SOURCE_LABEL[s] for s in srcs])
    a1.set_xlabel("z-score ($\\gamma=0.5$, $T=150$)")
    a1.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5,
              handlelength=1.2, columnspacing=1.0)
    a1.grid(axis="y", visible=False)
    a1.set_title("(a) Detection scores by source", loc="left", pad=16)
    if emap:
        w2 = g[(g.condition == "watermarked") & (g.gamma_f == 0.5) & (g.delta_f == 2.0) & (g.modification == "none")]
        x = np.array([emap.get(t, np.nan) for t in w2.text_id])
        a2.scatter(x, w2.z_dedup, s=5, color="#9b9a95", alpha=0.55, edgecolors="none")
        for _, r in tt.iterrows():
            a2.plot(r.spike, r.z_d2_dedup, "o", color="#1baf7a", ms=4.5, markeredgecolor="white", markeredgewidth=0.6)
            if r.source in ("python_code", "gutenberg", "imdb"):
                a2.annotate(SOURCE_LABEL[r.source].split(" (")[0], (r.spike, r.z_d2_dedup), fontsize=6.3,
                            color=INK, xytext=(5, -9) if r.source != "imdb" else (-4, 6),
                            ha="left" if r.source != "imdb" else "right", textcoords="offset points")
        a2.axhline(4, color=INK, lw=0.8, ls=":")
        a2.set_xlabel("Mean spike entropy of generator ($\\gamma$=0.5, $\\delta$=2)")
        a2.set_ylabel("z, repeated bigrams counted once")
        rho = S["md"].get("spearman_spike_zdedup", {}).get("rho")
        a2.set_title(f"(b) Entropy vs. detectability (Spearman $\\rho$={rho:.2f})" if rho is not None else "(b)",
                     loc="left", pad=16)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT_F / "sources.pdf")
    plt.close(fig)


def fig_length(M):
    import matplotlib.pyplot as plt
    g = M["gen"]
    w2 = g[(g.condition == "watermarked") & (g.gamma_f == 0.5) & (g.modification == "none")]
    Ts = [int(c.split("_")[-1]) for c in g if c.startswith("z_at_") and g[c].notna().any()]
    fig, ax = plt.subplots(figsize=(3.1, 1.9))
    out = {}
    for d, col in ((1.0, "#2a78d6"), (2.0, "#eb6834"), (5.0, "#1baf7a")):
        sub = w2[w2.delta_f == d]
        y = [rate(sub[f"z_at_{t}"].dropna()) for t in Ts]
        out[d] = dict(zip(Ts, y))
        ax.plot(Ts, y, marker="o", ms=3.5, color=col, label=f"$\\delta={d:g}$")
    S["md"]["tpr_by_length"] = {str(k): v for k, v in out.items()}
    ax.set_xlabel("Tokens observed $T$")
    ax.set_ylabel("TPR at $z>4$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT_F / "length.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data-source table, snippets and LaTeX number macros
# ---------------------------------------------------------------------------

TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s))
    s = "".join(c for c in s if (32 <= ord(c) < 0x250 or 0x2010 <= ord(c) <= 0x2027) and c != "\ufffd")
    out = "".join(TEX_ESC.get(c, c) for c in s)
    return " ".join(out.split())


def trunc_words(s, n):
    w = str(s).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


SHORT_ID = {"c4": "allenai/c4 (realnewslike)", "wikipedia": "wikimedia/wikipedia (20231101.en)",
            "cc_news": "vblagoje/cc\\_news", "gutenberg": "sedthh/gutenberg\\_english", "imdb": "stanfordnlp/imdb",
            "reddit": "webis/tldr-17", "amazon": "McAuley-Lab/Amazon-Reviews-2023 (Electronics)",
            "stackexchange": "HuggingFaceH4/stack-exchange-preferences", "acl_ethics": "coastalcph/EthiCon (GitHub)",
            "python_code": "CPython standard library"}
SHORT_LIC = {"c4": "ODC-BY", "wikipedia": "CC BY-SA 4.0", "cc_news": "publisher copyright", "gutenberg": "public domain",
             "imdb": "research use", "reddit": "CC BY 4.0", "amazon": "research use", "stackexchange": "CC BY-SA 4.0",
             "acl_ethics": "CC BY 4.0", "python_code": "PSF"}
REASON = {"too_short_chars": "short", "too_few_tokens": "<200 tok.", "low_letter_ratio": "non-prose",
          "non_english_script": "non-Latin", "markup_heavy": "markup", "unsafe_content": "unsafe",
          "unsafe_subreddit": "NSFW sub.", "pii_heavy": "PII", "link_farm": "links", "duplicate": "dup.",
          "not_text": "empty", "docstring_heavy": "docstrings", "source_error": "error"}


def data_tables(root, md_name):
    rp = root / "data" / f"prompts_{md_name}_report.json"
    if not rp.exists():
        return
    rep = json.loads(rp.read_text())
    S["data"] = {}
    lines = [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}l l l l r r l r c@{}}", r"\toprule",
             r"Source & Register & Dataset / origin & Licence & Scanned & Kept & Main rejections & Med.\ tok. & Released \\",
             r"\midrule"]
    tot = {"pii": 0}
    for s_ in rep["sources"]:
        k = s_["source"]
        rej = sorted(s_["rejected"].items(), key=lambda kv: -kv[1])[:3]
        rej_s = ", ".join(f"{REASON.get(a, a)} {b}" for a, b in rej) or "--"
        rel = "yes" if k not in ("cc_news", "imdb", "amazon") else "IDs only"
        lines.append(f"{SOURCE_LABEL.get(k, k)} & {tex(s_['domain'])} & \\texttt{{{SHORT_ID.get(k, k)}}} & {SHORT_LIC.get(k, '')} & "
                     f"{s_['scanned']:,} & {s_['kept']} & {tex(rej_s)} & {fmt(s_.get('median_doc_tokens') or float('nan'), 0)} & {rel} \\\\")
        S["data"][k] = {"scanned": s_["scanned"], "kept": s_["kept"], "rejected": s_["rejected"], "edits": s_["edits"],
                        "median_doc_tokens": s_.get("median_doc_tokens")}
        tot["pii"] += s_["edits"].get("pii_replaced", 0)
    lines += [r"\bottomrule", r"\end{tabular}}"]
    write_tex("data_sources", "\n".join(lines))
    S["data"]["all_pii"] = tot["pii"]
    S["data"]["examples"] = {s_["source"]: s_.get("example") for s_ in rep["sources"]}


def numbers(root, R, M, md_name, rep_name):
    N = {}

    def put(k, v, d=None):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return
        if d is not None and isinstance(v, (int, float, np.floating, np.integer)):
            v = f"{float(v):.{d}f}"
        N[k] = v

    env_p = root / "results" / "environment.json"
    if env_p.exists():
        e = json.loads(env_p.read_text())
        for k in ("cpu", "ram_gb", "macos", "python", "torch", "transformers", "datasets"):
            put(f"env.{k.replace('_gb', '')}", tex(e.get(k) or ""))
    if "rep" in S:
        put("rep.n", S["rep"]["n_prompts"])
        put("rep.promptmed", S["rep"]["prompt_tokens_median"], 0)
        put("rep.spike", S["rep"].get("spike_wm"), 3)
        put("rep.spike.unwm", S["rep"].get("spike_unwm"), 3)
        for r in S["rep"]["grid"]:
            t = f"g{r['gamma']:g}.d{r['delta']:g}"
            for m in ("tpr4", "tpr5", "tpr4_dedup", "auc", "auc_human"):
                put(f"rep.{m}.{t}", r[m], 3)
            put(f"rep.z.{t}", r["mean_z"], 1)
            put(f"rep.green.{t}", r["green"], 2)
            put(f"rep.ppl.{t}", r["ppl_mean"], 1)
            put(f"rep.fpunwm.{t}", r["fp_unwm"])
            put(f"rep.fphum.{t}", r["fp_human"])
        for c in ("unwatermarked", "human"):
            put(f"rep.ppl.{c}", S["rep"].get(f"ppl_{c}_mean"), 1)
            for gam in (0.25, 0.5):
                put(f"rep.maxz.{c}.g{gam:g}", S["rep"].get(f"max_z_{c}_g{gam}"), 2)
        for r in S["rep"].get("t5", []):
            e_ = f"e{r['eps']:g}"
            for m, d in (("tpr4", 2), ("tpr4_dedup", 2), ("tpr5", 2), ("auc", 3), ("realised", 2), ("ppl", 1), ("mean_z", 1)):
                put(f"rep.t5.{m}.{e_}", r[m], d)
    dc = root / "results" / rep_name / "device_check.json"
    if dc.exists():
        d = json.loads(dc.read_text())
        put("dev.n", d.get("cpu_watermarked_n"))
        for dev in ("cpu", "mps", "cuda"):
            put(f"dev.{dev}.z", d.get(f"{dev}_watermarked_mean_z"), 2)
            put(f"dev.{dev}.det", d.get(f"{dev}_watermarked_detected"))
            put(f"dev.{dev}.zu", d.get(f"{dev}_unwatermarked_mean_z"), 2)
    if "md" in S:
        put("md.n", S["md"]["n_prompts"])
        for r in S["md"]["by_source"]:
            k = r["source"].replace("all_new", "allnew")
            for m in ("tpr_d1", "tpr_d2", "tpr_d5", "tpr_d2_dedup", "tpr_d2_T50", "tpr_d2_T100", "auc_d2"):
                put(f"md.{k}.{m}", r[m], 2)
            for m in ("spike", "entropy", "distinct2_unwm", "distinct2_d2", "distinct2_d5", "distinct2_human",
                      "judge_human", "judge_unwm", "judge_d1", "judge_d2", "judge_d5", "z_human_max", "z_unwm_max"):
                put(f"md.{k}.{m}", r[m], 2)
            put(f"md.{k}.z_d2", r["z_d2"], 1)
            for m in ("ppl_human", "ppl_unwm", "ppl_d1", "ppl_d2", "ppl_d5"):
                put(f"md.{k}.{m}", r[m], 1)
            put(f"md.{k}.fp_human", r["fp_human"])
            put(f"md.{k}.fp_human_dedup", r["fp_human_dedup"])
            put(f"md.{k}.fp_unwm_dedup", r["fp_unwm_dedup"])
            put(f"md.{k}.z_d2_dedup", r["z_d2_dedup"], 1)
            put(f"md.{k}.uniq_d2", r["uniq_d2"], 0)
            put(f"md.{k}.fp_unwm", r["fp_unwm"])
            put(f"md.{k}.nneg", r["n_neg"])
            put(f"md.{k}.n", r["n"])
        put("md.allnew.nneg", next(r["n_neg"] for r in S["md"]["by_source"] if r["source"] == "all_new"))
        put("md.all.nneg", next(r["n_neg"] for r in S["md"]["by_source"] if r["source"] == "all"))
        sp = S["md"].get("spearman_spike_z")
        if sp:
            put("md.rho", sp["rho"], 2)
            put("md.rho.n", sp["n"])
            put("md.rho.p", f"{sp['p']:.1e}")
            put("md.rho.dedup", S["md"]["spearman_spike_zdedup"]["rho"], 2)
        for d, vals in S["md"].get("tpr_by_length", {}).items():
            for T_, v in vals.items():
                put(f"md.len.d{float(d):g}.T{T_}", v, 2)
        g = M["gen"]
        put("md.ntexts", int(g.text_id.nunique() + (M["attacks_t5"].text_id.nunique() if M.get("attacks_t5") is not None else 0)))
    for r in S.get("gap", []):
        k = f"gap.g{r['gamma']}.d{int(r['delta'])}"
        for m in ("rho", "tpr_low", "tpr_mid", "tpr_high", "rho_len"):
            if m in r:
                put(f"{k}.{m}", r[m], 2)
        for m in ("green1", "green2", "pred_paper", "spike_mean", "spike_max", "tpr_cap", "tpr_short"):
            if m in r:
                put(f"{k}.{m}", r[m], 3)
        for m in ("rho_p", "green_p", "cap_p"):
            if m in r:
                put(f"{k}.{m}", r[m], 3)
        if "n_cap" in r:
            put(f"{k}.n_cap", r["n_cap"])
    jc = S.get("judge_check")
    if jc:
        put("jc.n", jc["n"])
        for name in ("j15", "j3"):
            if name in jc:
                for k in ("kq", "kl", "exact", "within1", "rho", "kq_nocode"):
                    put(f"jc.{name}.{k}", jc[name][k], 2)
                put(f"jc.{name}.p", jc[name]["p"], 3)
                put(f"jc.{name}.levels", jc[name]["levels_used"])
                put(f"jc.{name}.mean", jc[name]["mean"], 2)
                put(f"jc.{name}.ones", jc[name]["n_ones"])
                for c, v in jc[name]["by_cond"].items():
                    put(f"jc.{name}.{c}", v, 2)
        for c, v in jc["human_by_cond"].items():
            put(f"jc.human.{c}", v, 2)
        put("jc.human.mean", jc["human_mean"], 2)
        hh = jc.get("hh")
        if hh:
            put("hh.n", hh["n"])
            for k, v in hh.items():
                if isinstance(v, float):
                    put(f"hh.{k}", v, 3 if k.startswith(("mw", "p_")) else 2)
            for c, v in hh["by_cond2"].items():
                put(f"hh.r2.{c}", v, 2)
            for c, v in hh["by_cond_avg"].items():
                put(f"hh.avg.{c}", v, 2)
        put("jc.mw.human_d5", jc["mw_human_d5"], 3)
        put("jc.mw.unwm_d2", jc["mw_unwm_d2"], 2)
        put("jc.rho_ppl", jc.get("rho_ppl"), 2)
        put("jc.p_ppl", jc.get("p_ppl"), 3)
    if "mod" in S:
        put("mod.n", S["mod"]["n_prompts"])
        for r in S["mod"]["by_source"]:
            k = r["source"].replace("all_new", "allnew")
            for m in ("tpr_d1", "tpr_d2", "tpr_d5", "tpr_d2_dedup", "tpr_d2_T50", "tpr_d2_T100", "auc_d2", "spike",
                      "entropy", "distinct2_unwm", "distinct2_d2", "distinct2_human", "judge_human", "judge_unwm",
                      "judge_d1", "judge_d2", "judge_d5", "z_human_max", "z_unwm_max"):
                put(f"mod.{k}.{m}", r[m], 2)
            for m in ("z_d2", "z_d2_dedup", "ppl_human", "ppl_unwm", "ppl_d1", "ppl_d2", "ppl_d5"):
                put(f"mod.{k}.{m}", r[m], 1)
            for m in ("fp_human", "fp_human_dedup", "fp_unwm", "fp_unwm_dedup", "n_neg", "n"):
                put(f"mod.{k}.{m}", r[m])
            put(f"mod.{k}.uniq_d2", r["uniq_d2"], 0)
        sp = S["mod"].get("spearman_spike_z")
        if sp:
            put("mod.rho", sp["rho"], 2)
            put("mod.rho.dedup", S["mod"]["spearman_spike_zdedup"]["rho"], 2)
    for r in S.get("robust", []):
        ds = "rep" if r["dataset"].startswith("C4") else "md"
        put(f"rob.{ds}.{r['modification']}", r["tpr4"], 2)
        put(f"rob.{ds}.{r['modification']}.real", r["realised"], 2)
        put(f"rob.{ds}.{r['modification']}.dedup", r.get("tpr4_dedup"), 2)
    for r in S.get("judge", {}).get("by_condition", []):
        e_ = "rep" if r["exp"] == "replication" else "md"
        put(f"judge.{e_}.{r['cond']}", r["judge"], 2)
        put(f"judge.{e_}.{r['cond']}.ppl", r["ppl_median"], 1)
        put(f"judge.{e_}.{r['cond']}.dist", r["distinct2"], 2)
        put(f"judge.{e_}.{r['cond']}.lo", r["judge_lo"], 2)
        put(f"judge.{e_}.{r['cond']}.hi", r["judge_hi"], 2)
    for e_, key in (("rep", "replication"), ("md", "multidomain")):
        j = S.get("judge", {}).get(key)
        if j:
            put(f"judge.{e_}.rho.ppl", j["spearman_ppl_judge"], 2)
            put(f"judge.{e_}.rho.dist", j["spearman_distinct2_judge"], 2)
            put(f"judge.{e_}.rho.ppldist", j["spearman_ppl_distinct2"], 2)
    for p in S.get("agreement", {}).get("pairs", []):
        put(f"agree.{p['a']}.{p['b']}.kq", p["kappa_quadratic"], 2)
        put(f"agree.{p['a']}.{p['b']}.kl", p["kappa_linear"], 2)
        put(f"agree.{p['a']}.{p['b']}.rho", p["spearman"], 2)
        put(f"agree.{p['a']}.{p['b']}.exact", p["exact"], 2)
        put(f"agree.{p['a']}.{p['b']}.within1", p["within1"], 2)
        put(f"agree.{p['a']}.{p['b']}.n", p["n"])
    for rater, e in S.get("agreement", {}).get("vs_signals", {}).items():
        for k, v in e.items():
            if isinstance(v, float):
                put(f"ann.{rater}.{k}", v, 3 if k.startswith(("p_", "mw_")) else 2)
            elif isinstance(v, int):
                put(f"ann.{rater}.{k}", v)
    for rater, by in S.get("agreement", {}).get("mean_rating_by_condition", {}).items():
        for c, v in by.items():
            put(f"ann.{rater}.{c}", v, 2)
    if "data" in S:
        put("data.all.pii", S["data"]["all_pii"])
        for k, v in S["data"].items():
            if isinstance(v, dict) and "edits" in v:
                for ek, ev in v["edits"].items():
                    if isinstance(ev, (int, float)):
                        put(f"data.{k}.{ek}", ev)
                put(f"data.{k}.scanned", v["scanned"])
        ex = S["data"].get("examples", {}).get("acl_ethics")
        if ex:
            put("snip.acl.raw", tex(trunc_words(ex["raw"], 28)))
            put("snip.acl.clean", tex(trunc_words(ex["clean"], 28)))
    # examples
    if R is not None:
        pr = pd.read_csv(root / "data" / f"prompts_{rep_name}.csv")
        r0 = pr.iloc[0]
        meta = json.loads(r0.source_doc_meta)
        u = str(meta.get("url", ""))
        put("c4.example.url", tex(u if len(u) <= 42 else u[:40] + "..."))
        put("c4.example.ts", tex(meta.get("timestamp", "")))
        put("c4.example.text", tex(trunc_words(r0.prompt, 30)))
    if M is not None:
        g = M["gen"]
        pm = pd.read_csv(root / "data" / f"prompts_{md_name}.csv")
        cand = pm[pm.source == "wikipedia"].prompt_id.tolist() or pm.prompt_id.tolist()
        best = None
        for pid in cand:
            u = g[(g.prompt_id == pid) & (g.condition == "unwatermarked") & (g.detector_gamma == 0.5)]
            w = g[(g.prompt_id == pid) & (g.condition == "watermarked") & (g.delta_f == 2.0) & (g.modification == "none")]
            if len(u) and len(w) and abs(u.z.iloc[0]) < 1.5 and 6 < w.z.iloc[0] < 12:
                best = (pid, u.iloc[0], w.iloc[0])
                break
        if best:
            pid, u, w = best
            p = pm[pm.prompt_id == pid].iloc[0]
            put("snip.row.id", tex(pid))
            put("snip.row.prompt", tex(trunc_words(p.prompt, 24)))
            put("snip.row.human", tex(trunc_words(p.human, 18)))
            put("snip.row.unwm", tex(trunc_words(u.text, 18)))
            put("snip.row.wm", tex(trunc_words(w.text, 18)))
            put("snip.row.zu", u.z, 2)
            put("snip.row.zw", w.z, 2)
    lines = ["% generated by src/analysis.py - do not edit"]
    lines += [f"\\nmdef{{{k}}}{{{v}}}" for k, v in sorted(N.items())]
    (OUT_T / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from llm_judge import RUBRIC
    jp = tex(RUBRIC).replace(tex('{prompt}'), r"\{prompt\}").replace(tex('{continuation}'), r"\{continuation\}")
    (OUT_T / "judge_prompt.tex").write_text(
        "\n".join(tex(line) + r"\\" if line.strip() else r"\smallskip" for line in RUBRIC.splitlines()), encoding="utf-8")
    return N


def main(argv=None):
    global OUT_T, OUT_F, ORACLE_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--rep", default="replication")
    ap.add_argument("--md", default="multidomain")
    ap.add_argument("--modern", default="multidomain_qwen", help="same prompts, modern generator")
    args = ap.parse_args(argv)
    root = Path(args.root)
    OUT_T = root / "results" / "tables"
    OUT_F = root / "results" / "figures"
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    style()
    R, M = load(root, args.rep), load(root, args.md)
    qs = root / "results" / args.rep / "quality_settings.json"
    ORACLE_NAME = (json.loads(qs.read_text()).get("oracle", "") if qs.exists() else "")
    ORACLE_NAME = pretty_model(ORACLE_NAME) if ORACLE_NAME else "oracle"
    if R is not None:
        T = replication(R)
        replication_attack(R)
        if T["ppl_mean"].notna().any():
            fig_tradeoff(T, S["rep"])
        gap_analysis(R, root, args.rep)
    if M is not None:
        TM = multidomain(M)
        fig_sources(M, TM)
        fig_length(M)
        agreement(M, root)
        Q = load(root, args.modern)
        if Q is not None:
            TQ = multidomain(Q, key="mod", table=None)
            gm = lambda D: pretty_model(D["gen"].model.iloc[0])
            S["mod"]["models"] = [gm(M), gm(Q)]
            modern_compare(M, Q, TM, TQ, [gm(M), gm(Q)])
    rb = robustness(R, M)
    if len(rb):
        fig_robust(rb)
    judge_quality(R, M)
    judge_check(root)
    data_tables(root, args.md)
    numbers(root, R, M, args.md, args.rep)
    (root / "results" / "summary.json").write_text(json.dumps(S, indent=2, default=float), encoding="utf-8")
    print(f"Wrote tables to {OUT_T}, figures to {OUT_F}, numbers to results/summary.json")


if __name__ == "__main__":
    main()
