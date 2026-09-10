"""Per-stratum view of the corrected deltas in first_token_gap.json.

Top row  : mean shift in delta_score per stratum, plus the ALL-items aggregate.
Bottom row: the behavioural change (R_ctx, from the judge-free `ordered`
            classifier) for the same strata.

The point of the figure is the ALL bar in the top row: it sits near zero on the
tulu decline because country_capitals and famous_biographies move in opposite
directions and cancel. Averaging across strata hides the effect entirely.
"""
import json, re, glob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

STRATA = ["country_capitals", "world_facts", "famous_biographies"]
LEGS = [("alpaca", "peak", "trough", 100, 800),
        ("tulu", "peak", "trough", 500, 3150),
        ("tulu", "trough", "recovery", 3150, 5000)]
RES = {"alpaca": "results/cpi-results/alpaca-results",
       "tulu": "results/cpi-results/tulu-results"}

rows = json.load(open("src/mechanistic-analysis/first_token_gap.json", encoding="utf-8"))
src = {it["item_id"]: it["source"]
       for it in json.load(open("dataset/conflict_eval_unified.json", encoding="utf-8"))}
for r in rows:
    r["source"] = src.get(r["item_id"], "?")


def r_ctx_by_stratum(run, step):
    """CTX rate per stratum from the deterministic `ordered` classifier (no judge)."""
    p = f"{RES[run]}/checkpoint_step{step}_metrics.json"
    per = json.load(open(p, encoding="utf-8"))["per_item"]
    c = {}
    for r in per:
        if not r.get("passed") or "ordered" not in r:
            continue
        c.setdefault(r["source"], Counter())[r["ordered"]["label"]] += 1
    return {k: v["CTX"] / max(v["CTX"] + v["PAR"], 1) for k, v in c.items()}


fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for col, (run, ep, lp, es, ls) in enumerate(LEGS):
    e = {r["item_id"]: r for r in rows if r["run"] == run and r["phase"] == ep}
    l = {r["item_id"]: r for r in rows if r["run"] == run and r["phase"] == lp}
    common = sorted(set(e) & set(l))

    labels, shifts, pcts = [], [], []
    for s in STRATA + ["ALL"]:
        ids = common if s == "ALL" else [i for i in common if e[i]["source"] == s]
        d = np.array([l[i]["delta_score"] - e[i]["delta_score"] for i in ids])
        labels.append(("ALL items" if s == "ALL" else s.replace("_", "\n")) + f"\nn={len(ids)}")
        shifts.append(d.mean())
        pcts.append((d < 0).mean())

    ax = axes[0, col]
    colors = ["#c0392b" if v < 0 else "#2471a3" for v in shifts]
    colors[-1] = "#7f8c8d"                                  # aggregate in grey
    bars = ax.bar(range(4), shifts, color=colors)
    for b, v, p in zip(bars, shifts, pcts):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.06 if v >= 0 else -0.06), f"{v:+.2f}\n{p:.0%} to PAR",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{run}: {ep} -> {lp}", fontweight="bold")
    if col == 0:
        ax.set_ylabel("mean shift in delta_score\n(neg = toward PARAMETRIC)")
    ax.margins(y=0.25)

    early, late = r_ctx_by_stratum(run, es), r_ctx_by_stratum(run, ls)
    dR = [late.get(s, 0) - early.get(s, 0) for s in STRATA] + [None]
    ax = axes[1, col]
    vals = [v for v in dR[:3]]
    bars = ax.bar(range(3), vals, color=["#c0392b" if v < 0 else "#2471a3" for v in vals])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.008 if v >= 0 else -0.008),
                f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s.replace("_", "\n") for s in STRATA], fontsize=8)
    if col == 0:
        ax.set_ylabel("change in R_ctx\n(behaviour, judge-free)")
    ax.margins(y=0.3)

fig.suptitle("The inversion is stratum-specific: capitals go parametric, biographies go contextual,\n"
             "and the ALL-items average cancels to nothing",
             fontweight="bold", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "src/mechanistic-analysis/per_stratum_shift.png"
plt.savefig(out, dpi=150)
print("wrote", out)
