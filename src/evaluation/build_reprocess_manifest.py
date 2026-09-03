
import os
import csv
import json
import glob
from collections import defaultdict

RESULTS_ROOT = "results/cpi-results"
FINDINGS_CSV = os.path.join(RESULTS_ROOT, "audit_findings.csv")
OUT_PATH = os.path.join(RESULTS_ROOT, "reprocess_manifest.json")

DATASET_FIXES = [
    {
        "item_id": "cap_0181",
        "field": "parametric_answer",
        "old": "Vaiaku village, Funafuti province",
        "new": "Funafuti",
        "reason": "Gold answer is an unusually long/specific phrasing the model "
                  "never naturally produces; the model's real-world-correct "
                  "answer ('Funafuti') is a substring of the old gold text, which "
                  "confuses both the ordered/paper text-matching AND compounds "
                  "the log-prob length bias (root cause #2). Fix in "
                  "dataset/conflict_eval_unified.json before rescoring this item.",
    }
]


def checkpoint_file(run, step):
    return f"results/cpi-results/{run}-results/checkpoint_step{step}_metrics.json"


def main():
    rows = list(csv.DictReader(open(FINDINGS_CSV, encoding="utf-8")))
    print(f"Loaded {len(rows)} flagged rows from {FINDINGS_CSV}")

    judge_rerun = []
    logprob_rerun = []

    for r in rows:
        # step can be the literal string "inf" (see evaluate_checkpoint's checkpoint_id
        step_raw = r["step"]
        step = step_raw if step_raw == "inf" else int(step_raw)
        entry = {
            "run": r["run"],
            "step": step,
            "item_id": r["item_id"],
            "source": r["source"],
            "verdict": r["verdict"],
            "current_final_label": r["final_label"],
            "checkpoint_file": checkpoint_file(r["run"], r["step"]),
        }
        if r["judge_label"]:
            entry["current_judge_label"] = r["judge_label"]
            judge_rerun.append(entry)
        else:
            entry["current_logprob_label"] = r["logprob_label"]
            logprob_rerun.append(entry)

    # group counts per checkpoint file, for the downstream scripts to batch by file
    def by_file(entries):
        d = defaultdict(int)
        for e in entries:
            d[e["checkpoint_file"]] += 1
        return dict(sorted(d.items()))

    manifest = {
        "generated_from": FINDINGS_CSV,
        "counts": {
            "judge_rerun": len(judge_rerun),
            "logprob_rerun": len(logprob_rerun),
            "dataset_fixes": len(DATASET_FIXES),
            "checkpoint_files_touched_by_judge_rerun": len(by_file(judge_rerun)),
            "checkpoint_files_touched_by_logprob_rerun": len(by_file(logprob_rerun)),
        },
        "dataset_fixes": DATASET_FIXES,
        "judge_rerun": judge_rerun,
        "logprob_rerun": logprob_rerun,
    }

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\njudge_rerun    : {len(judge_rerun)} items across "
          f"{len(by_file(judge_rerun))} checkpoint files (no GPU needed)")
    print(f"logprob_rerun  : {len(logprob_rerun)} items across "
          f"{len(by_file(logprob_rerun))} checkpoint files (needs Drive checkpoints)")
    print(f"dataset_fixes  : {len(DATASET_FIXES)} item(s)")
    print(f"\nWrote manifest -> {OUT_PATH}")


if __name__ == "__main__":
    main()
