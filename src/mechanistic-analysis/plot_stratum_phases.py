"""Per-stratum phase chart, corrected deltas only.

Uses ONLY first_token_gap.json (single-space scoring). No stored deltas, so
nothing here is touched by the double-space bug. Points are the phase
checkpoints: alpaca peak->trough, tulu peak->trough->recovery.
Error bars are +/- 1 SEM.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

rows = json.load(open("src/mechanistic-analysis/first_token_gap.json", encoding="utf-8"))
src = {it["item_id"]: it["source"]
       for it in json.load(open("dataset/conflict_eval_unified.json", encoding="utf-8"))}
for r in rows:
    r["source"] = src.get(r["item_id"], "?")

STRATA = ["country_capitals", "world_facts", "famous_biographies"]
COLOR = {"country_capitals": "#c0392b", "world_facts": "#e67e22",
         "famous_biographies": "#2471a3"}
PHASES = {"alpaca": ["peak", "trough"], "tulu": ["peak", "trough", "recovery"]}
METRIC = "delta_score"   # or delta_gen / first_gap_score / first_gap_gen

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                         gridspec_kw={"width_ratios": [1, 1.4]})
for ax, (run, phases) in zip(axes, PHASES.items()):
    for s in STRATA:
        mean, sem, n0 = [], [], 0
        for p in phases:
            v = np.array([r[METRIC] for r in rows
                          if r["run"] == run and r["phase"] == p and r["source"] == s])
            mean.append(v.mean())
            sem.append(v.std(ddof=1) / np.sqrt(len(v)))
            n0 = len(v)
        ax.errorbar(range(len(phases)), mean, yerr=sem, marker="o", ms=8, lw=2,
                    capsize=4, color=COLOR[s],
                    label=f"{s.replace('_', ' ')}  (n={n0})")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases)
    ax.set_xlim(-0.25, len(phases) - 0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(run, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel(f"mean {METRIC}\n(higher = favours CONTEXT)")
axes[0].legend(fontsize=8, loc="best")
axes[1].legend(fontsize=8, loc="best")
fig.suptitle("Scored preference by knowledge type, corrected deltas "
             "(capitals invert and recover; biographies move the other way)",
             fontweight="bold", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = "src/mechanistic-analysis/stratum_phases.png"
plt.savefig(out, dpi=150)
print("wrote", out)
