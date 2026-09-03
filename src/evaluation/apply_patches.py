"""
Merges rerun_judge.py's patch (results/cpi-results/judge_v2_patch.json) and/or
rescore_logprob.py's patch (results/cpi-results/logprob_rescore_patch.json) on
top of the original per-checkpoint metrics files, and regenerates corrected
trajectory summaries.

Two write modes:
  default (no --write-in-place)  -- original checkpoint_step*_metrics.json
      files are left untouched; only a separate
      results/cpi-results/{run}-results/trajectory_summary_v2judge.json is
      written per run, alongside the existing trajectory_summary.json.
  --write-in-place  -- ALSO overwrites the real checkpoint_step*_metrics.json
      files with the corrected per_item records AND recomputed
      aggregate/per_stratum/filter_yield, so downstream analysis reading the
      standard filename directly sees the fix without needing to know about
      patch-merging at all. Before ever touching a given file, its pristine
      original is copied to --backup-dir (only on the FIRST write -- re-runs
      never overwrite an existing backup, so the true v1 original is never
      lost even if this script runs many times).

Two patches, two independent opt-in switches:
  --judge-patch is ALWAYS applied if present (the judge-bias fix is settled --
      see docs/label-audit-findings.md root cause #1).
  --logprob-patch is only applied to final_label if you ALSO pass
      --apply-logprob-fix. Without that flag, this script just reports how
      many items WOULD change (root cause #2's drop-first-token correction),
      so you can eyeball it before committing to it. Passing the flag applies
      rescore_logprob.py's `recommended_final_label` (the drop-first-token
      correction) and overwrites that item's `logprob` block with the
      corrected delta/lp_par_pt/lp_cf_pt, tagged with correction_method.
      NOTE: a small number of items may come out AMBIG under the corrected
      method where v1 was confident -- these were never escalated to the
      judge (they weren't AMBIG under v1, so rerun_judge.py never touched
      them), so they're applied as-is (final_label="AMBIG") rather than
      triggering a second escalation round; check
      n_dropfirst_now_ambiguous in the printed summary.

Usage:
    python apply_patches.py \
        --judge-patch results/cpi-results/judge_v2_patch.json \
        --logprob-patch results/cpi-results/logprob_rescore_patch.json \
        [--apply-logprob-fix] \
        [--write-in-place] [--backup-dir results/cpi-results/_raw_v1_backup]
"""
import os
import sys
import json
import glob
import shutil
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from eval import summarize  # noqa: E402


def load_judge_patch(path):
    if not path or not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8")).get("patch", {})


def load_logprob_patch(path):
    """logprob_rescore_patch.json stores a flat `results` list (not a dict
    keyed by run|step|item_id like the judge patch) -- index it the same way
    here so both patches can be looked up identically below."""
    if not path or not os.path.exists(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    return {f"{r['run']}|{r['step']}|{r['item_id']}": r for r in data.get("results", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-patch", default="results/cpi-results/judge_v2_patch.json")
    ap.add_argument("--logprob-patch", default="results/cpi-results/logprob_rescore_patch.json")
    ap.add_argument("--apply-logprob-fix", action="store_true",
                     help="Apply the drop-first-token log-prob correction to final_label, "
                          "not just report what it would change.")
    ap.add_argument("--results-root", default="results/cpi-results")
    ap.add_argument("--write-in-place", action="store_true",
                     help="Also overwrite the real checkpoint_step*_metrics.json files "
                          "(pristine originals are backed up first -- see script docstring).")
    ap.add_argument("--backup-dir", default="results/cpi-results/_raw_v1_backup",
                     help="Where pristine (pre-correction) originals get copied before the "
                          "first in-place write. Only used with --write-in-place.")
    args = ap.parse_args()

    judge_patch = load_judge_patch(args.judge_patch)
    print(f"Judge patch entries: {len(judge_patch)}"
          f"{'  (none found -- run rerun_judge.py first)' if not judge_patch else ''}")

    logprob_patch = load_logprob_patch(args.logprob_patch)
    if logprob_patch:
        would_flip = sum(1 for r in logprob_patch.values()
                          if r["recommended_final_label"] != r.get("old_final_label"))
        n_degenerate = sum(1 for r in logprob_patch.values()
                            if r.get("new_logprob_dropfirst", {}).get("degenerate_dropfirst"))
        mode = "WILL be applied to final_label" if args.apply_logprob_fix else \
               "NOT applied (pass --apply-logprob-fix to change that)"
        print(f"Log-prob rescore patch: {len(logprob_patch)} item(s) processed, {mode}. "
              f"{would_flip} would change final_label; {n_degenerate} degenerate "
              f"(single-token candidate, dropfirst fell back to full average).")
    else:
        print("No log-prob rescore patch found (that's fine, it's optional at this stage).")

    if args.write_in_place:
        print(f"\n--write-in-place ENABLED: real checkpoint_step*_metrics.json files will be "
              f"overwritten. Pristine originals back up to {args.backup_dir} on first write only.")

    n_dropfirst_now_ambiguous_total = 0

    for run in ("alpaca", "tulu"):
        run_dir = os.path.join(args.results_root, f"{run}-results")
        files = sorted(glob.glob(os.path.join(run_dir, "checkpoint_step*_metrics.json")))
        if not files:
            continue
        print(f"\n=== {run}: {len(files)} checkpoint file(s) ===")

        traj = []
        n_patched_total = 0
        n_files_written_in_place = 0
        for fp in files:
            d = json.load(open(fp, encoding="utf-8"))
            step = d.get("global_step")
            per_item = d.get("per_item", [])

            n_patched = 0
            n_logprob_patched = 0
            for rec in per_item:
                key = f"{run}|{step}|{rec['item_id']}"

                p = judge_patch.get(key)
                if p is not None and p["new_final_label"] != rec.get("final_label"):
                    rec["final_label"] = p["new_final_label"]
                    rec["judge"] = dict(rec.get("judge") or {})
                    rec["judge"]["label"] = p["new_judge_label"]
                    rec["judge"]["other_subtype"] = p.get("new_judge_other_subtype")
                    rec["judge"]["prompt_version_used_for_correction"] = "v2"
                    n_patched += 1

                if args.apply_logprob_fix:
                    lp = logprob_patch.get(key)
                    if lp is not None and lp["recommended_final_label"] != rec.get("final_label"):
                        rec["final_label"] = lp["recommended_final_label"]
                        rec["logprob"] = dict(rec.get("logprob") or {})
                        dropfirst = lp["new_logprob_dropfirst"]
                        rec["logprob"]["label"] = dropfirst["label"]
                        rec["logprob"]["delta"] = dropfirst.get("delta")
                        rec["logprob"]["lp_par_pt"] = dropfirst.get("lp_par_pt")
                        rec["logprob"]["lp_cf_pt"] = dropfirst.get("lp_cf_pt")
                        rec["logprob"]["correction_method"] = "drop_first_token"
                        n_patched += 1
                        n_logprob_patched += 1
                        if dropfirst["label"] == "AMBIG":
                            n_dropfirst_now_ambiguous_total += 1

            n_patched_total += n_patched
            summary = summarize(per_item, methods=("paper", "ordered", "logprob"))
            traj.append({
                "step": step,
                "n_items_relabeled_by_judge_v2": n_patched - n_logprob_patched,
                "n_items_relabeled_by_logprob_dropfirst": n_logprob_patched,
                **{k: v for k, v in summary["aggregate"].get("final", {}).items()},
                "n_passed": summary["aggregate"].get("n_passed"),
            })

            if args.write_in_place and n_patched > 0:
                # back up the PRISTINE original exactly once -- if a backup
                # already exists from a prior run, never touch it again, so
                # re-running this script can't silently back up an
                # already-corrected copy and lose the true v1 original.
                rel = os.path.relpath(fp, args.results_root)
                backup_path = os.path.join(args.backup_dir, rel)
                if not os.path.exists(backup_path):
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(fp, backup_path)

                # update the file's own aggregate/per_stratum/filter_yield so
                # a downstream tool reading this file directly (not just
                # per_item) also sees the corrected numbers.
                d["aggregate"] = summary["aggregate"]
                d["per_stratum"] = summary["per_stratum"]
                d["filter_yield"] = summary["filter_yield"]
                d["judge_v2_correction_applied"] = True
                d["logprob_dropfirst_correction_applied"] = bool(args.apply_logprob_fix)
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2)
                n_files_written_in_place += 1

        out_path = os.path.join(run_dir, "trajectory_summary_v2judge.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(traj, f, indent=2)
        print(f"  {n_patched_total} record(s) relabeled across the run -> {out_path}")
        if args.write_in_place:
            print(f"  {n_files_written_in_place} checkpoint file(s) overwritten in place "
                  f"(originals backed up under {args.backup_dir})")

    if args.apply_logprob_fix and n_dropfirst_now_ambiguous_total:
        print(f"\nNote: {n_dropfirst_now_ambiguous_total} item(s) came out AMBIG under the "
              f"drop-first-token correction (were confidently CTX/PAR under v1) and were NOT "
              f"escalated to the judge -- they were never in rerun_judge.py's scope since v1 "
              f"wasn't AMBIG for them. Their final_label is now literally 'AMBIG'; consider "
              f"escalating these specifically to the judge in a follow-up pass if you want a "
              f"non-AMBIG answer for all of them.")

    if args.write_in_place:
        print("\nDone. Original checkpoint_step*_metrics.json files were overwritten "
              f"where they had relabeled items; pristine v1 originals are under {args.backup_dir}.")
    else:
        print("\nDone. Original checkpoint_step*_metrics.json files were NOT modified "
              "(pass --write-in-place to change that).")


if __name__ == "__main__":
    main()
