
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from eval import summarize  # noqa: E402


def load_patch(path):
    if not path or not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8")).get("patch", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-patch", default="results/cpi-results/judge_v2_patch.json")
    ap.add_argument("--logprob-patch", default="results/cpi-results/logprob_rescore_patch.json")
    ap.add_argument("--results-root", default="results/cpi-results")
    args = ap.parse_args()

    judge_patch = load_patch(args.judge_patch)
    print(f"Judge patch entries: {len(judge_patch)}"
          f"{'  (none found -- run rerun_judge.py first)' if not judge_patch else ''}")

    if os.path.exists(args.logprob_patch):
        lp = json.load(open(args.logprob_patch, encoding="utf-8"))
        would_flip = sum(1 for r in lp.get("results", [])
                          if r["new_logprob"]["label"] != r.get("old_final_label"))
        print(f"Log-prob rescore patch: {lp.get('n_processed', 0)} item(s) processed, "
              f"NOT auto-applied (see script docstring). "
              f"{would_flip} would change final_label if applied as-is.")
    else:
        print("No log-prob rescore patch found (that's fine, it's optional at this stage).")

    for run in ("alpaca", "tulu"):
        run_dir = os.path.join(args.results_root, f"{run}-results")
        files = sorted(glob.glob(os.path.join(run_dir, "checkpoint_step*_metrics.json")))
        if not files:
            continue
        print(f"\n=== {run}: {len(files)} checkpoint file(s) ===")

        traj = []
        n_patched_total = 0
        for fp in files:
            d = json.load(open(fp, encoding="utf-8"))
            step = d.get("global_step")
            per_item = d.get("per_item", [])

            n_patched = 0
            for rec in per_item:
                key = f"{run}|{step}|{rec['item_id']}"
                p = judge_patch.get(key)
                if p is None:
                    continue
                if p["new_final_label"] != rec.get("final_label"):
                    rec["final_label"] = p["new_final_label"]
                    rec["judge"] = dict(rec.get("judge") or {})
                    rec["judge"]["label"] = p["new_judge_label"]
                    rec["judge"]["other_subtype"] = p.get("new_judge_other_subtype")
                    rec["judge"]["prompt_version_used_for_correction"] = "v2"
                    n_patched += 1

            n_patched_total += n_patched
            summary = summarize(per_item, methods=("paper", "ordered", "logprob"))
            traj.append({
                "step": step,
                "n_items_relabeled_by_judge_v2": n_patched,
                **{k: v for k, v in summary["aggregate"].get("final", {}).items()},
                "n_passed": summary["aggregate"].get("n_passed"),
            })

        out_path = os.path.join(run_dir, "trajectory_summary_v2judge.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(traj, f, indent=2)
        print(f"  {n_patched_total} record(s) relabeled across the run -> {out_path}")

    print("\nDone. Original checkpoint_step*_metrics.json files were NOT modified.")


if __name__ == "__main__":
    main()
