"""Per-stratum trajectories across every saved checkpoint (alpaca 16, tulu 100).

Row 1  R_ctx per stratum -- behaviour, from the deterministic `ordered`
       classifier on the generation. Judge-free and completely independent of
       the log-prob scorer, so unaffected by the double-space bug.

Row 2  mean delta per stratum -- the model's scored preference.
       CAVEAT: these come from the stored per-checkpoint results, which used
       the buggy double-space scoring. The bug is identical at every checkpoint
       within a run (alpaca all scored 2026-08-05, tulu all 2026-08-23), so the
       SHAPE of each curve is meaningful while the absolute level is not.
       The corrected values from first_token_gap.json are overlaid as large
       markers at the phase checkpoints for comparison.

Thin line = raw per-checkpoint value. Thick line = rolling mean (window 3).
Dashed vertical = end of epoch 1 (dataset_size / 128).
"""
import json, glob, re
from collections import Counter, defaultdict
import statistics as st
import numpy as np
import matplotlib.pyplot as plt

STRATA = ["country_capitals", "world_facts", "famous_biographies"]
COLOR = {"country_capitals": "#c0392b", "world_facts": "#e67e22",
         "famous_biographies": "#2471a3"}
RUNS = {"alpaca": ("results/cpi-results/alpaca-results", 52002 / 128),
        "tulu":   ("results/cpi-results/tulu-results",  326154 / 128)}
PHASE_STEPS = {"alpaca": {"peak": 100, "trough": 800},
               "tulu": {"peak": 500, "trough": 3150, "recovery": 5000}}

corrected = json.load(open("src/mechanistic-analysis/first_token_gap.json", encoding="utf-8"))
src = {it["item_id"]: it["source"]
       for it in json.load(open("dataset/conflict_eval_unified.json", encoding="utf-8"))}
for r in corrected:
    r["source"] = src.get(r["item_id"], "?")


def scan(results_dir):
    """step -> {stratum: (ctx_rate, mean_delta)}"""
    out = {}
    for f in glob.glob(results_dir + "/checkpoint_step*_metrics.json"):
        m = re.search(r"step(\d+)", f)
        if not m:
            continue
        lab, dl = defaultdict(Counter), defaultdict(list)
        for r in json.load(open(f, encoding="utf-8"))["per_item"]:
            if not r.get("passed"):
                continue
            if "ordered" in r:
                lab[r["source"]][r["ordered"]["label"]] += 1
            if r.get("logprob") and "delta" in r["logprob"]:
                dl[r["source"]].append(r["logprob"]["delta"])
        out[int(m.group(1))] = {
            s: (lab[s]["CTX"] / max(lab[s]["CTX"] + lab[s]["PAR"], 1),
                st.mean(dl[s]) if dl[s] else float("nan"))
            for s in STRATA}
    return dict(sorted(out.items()))


def roll(v, w=3):
    return [st.mean(v[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(v))]


fig, axes = plt.subplots(2, 2, figsize=(15, 8),
                         gridspec_kw={"width_ratios": [1, 2.4]})
for col, (run, (rdir, spe)) in enumerate(RUNS.items()):
    data = scan(rdir)
    steps = list(data)

    for row, idx, ylab in [(0, 0, "R_ctx  (behaviour, judge-free)"),
                           (1, 1, "mean delta  (scored preference)")]:
        ax = axes[row, col]
        for s in STRATA:
            v = [data[k][s][idx] for k in steps]
            ax.plot(steps, v, color=COLOR[s], lw=0.7, alpha=0.35)
            ax.plot(steps, roll(v), color=COLOR[s], lw=2,
                    label=s.replace("_", " ") if row == 0 and col == 0 else None)
        ax.axvline(spe, color="black", ls="--", lw=1, alpha=0.6)
        ax.text(spe, ax.get_ylim()[1], " epoch 1 ends", fontsize=7,
                va="top", alpha=0.7)
        for name, stp in PHASE_STEPS[run].items():
            ax.axvline(stp, color="grey", ls=":", lw=1, alpha=0.7)
            if row == 0:
                ax.text(stp, ax.get_ylim()[0], name, fontsize=7, rotation=90,
                        va="bottom", ha="right", alpha=0.8)
        if row == 1:
            ax.axhline(0, color="black", lw=0.8)
            # corrected deltas from first_token_gap.json, at the phase checkpoints
            for s in STRATA:
                xs, ys = [], []
                for name, stp in PHASE_STEPS[run].items():
                    vals = [r["delta_score"] for r in corrected
                            if r["run"] == run and r["phase"] == name and r["source"] == s]
                    if vals:
                        xs.append(stp); ys.append(np.mean(vals))
                ax.plot(xs, ys, "o", color=COLOR[s], ms=9, mec="black", mew=1.2, zorder=5)
        ax.set_xlabel("training step")
        if col == 0:
            ax.set_ylabel(ylab)
        if row == 0:
            ax.set_title(f"{run}   ({len(steps)} checkpoints, {max(steps)/spe:.2f} epochs)",
                         fontweight="bold")

axes[0, 0].legend(fontsize=9, loc="lower left")
fig.suptitle("Context reliance by knowledge type. The collapse and recovery are almost entirely country_capitals;\n"
             "famous_biographies is flat at ceiling. Circles in row 2 = corrected deltas (stored curve has a known constant offset).",
             fontweight="bold", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = "src/mechanistic-analysis/stratum_trajectory.png"
plt.savefig(out, dpi=150)
print("wrote", out)
