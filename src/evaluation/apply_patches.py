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
      many items WOULD change, so you can eyeball it before committing to it.
      Passing the flag applies rescore_logprob.py's `recommended_final_label`
      -- the SINGLE-SPACE fix (root cause #2's double-space bug, corrected),
      NOT the drop-first-token correction. Drop-first was tried and rejected
      after a full-checkpoint test: it changed 87% of items, 76% of those
      just CTX/PAR -> AMBIG, on items that mostly didn't even have a length
      imbalance to begin with -- see rescore_logprob.py's module docstring
      for the full finding. new_logprob_dropfirst stays in the patch data for
      transparency but is never applied here.

      IMPORTANT: the corrected label is only ever applied when it's a genuine
      RESOLUTION (CTX/PAR/OTHER) -- if the corrected logprob is STILL AMBIG,
      final_label is left untouched, even if it differs from
      recommended_final_label. This matters a lot for items that were
      originally judge-escalated (old logprob was AMBIG, so old_final_label
      is the judge's actual answer): a full-checkpoint test found 96% of
      "would change" cases were exactly this -- corrected logprob still
      AMBIG, which would have overwritten a real, resolved judge answer with
      a bare "AMBIG" placeholder. The judge exists specifically to resolve
      AMBIG cases; downgrading its answer back to "unknown" because Layer 1
      is *also* still unsure is a data-quality regression, not a fix. Check
      n_skipped_still_ambiguous in the printed summary for how many were
      skipped this way.

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
                     help="Apply the single-space (double-space-bug) log-prob correction "
                          "to final_label, not just report what it would change.")
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
        mode = "WILL be applied to final_label" if args.apply_logprob_fix else \
               "NOT applied (pass --apply-logprob-fix to change that)"
        print(f"Log-prob rescore patch: {len(logprob_patch)} item(s) processed, {mode}. "
              f"{would_flip} would change final_label (single-space fix).")
    else:
        print("No log-prob rescore patch found (that's fine, it's optional at this stage).")

    if args.write_in_place:
        print(f"\n--write-in-place ENABLED: real checkpoint_step*_metrics.json files will be "
              f"overwritten. Pristine originals back up to {args.backup_dir} on first write only.")

    n_skipped_still_ambiguous_total = 0

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
                        if lp["recommended_final_label"] == "AMBIG":
                            # Never downgrade an existing resolved answer (often the
                            # judge's) to a bare "AMBIG" placeholder just because the
                            # corrected logprob is ALSO still unsure -- see docstring.
                            n_skipped_still_ambiguous_total += 1
                            continue
                        rec["final_label"] = lp["recommended_final_label"]
                        rec["logprob"] = dict(rec.get("logprob") or {})
                        fixed = lp["new_logprob"]  # single-space fix, NOT dropfirst -- see docstring
                        rec["logprob"]["label"] = fixed["label"]
                        rec["logprob"]["delta"] = fixed.get("delta")
                        rec["logprob"]["lp_par_pt"] = fixed.get("lp_par_pt")
                        rec["logprob"]["lp_cf_pt"] = fixed.get("lp_cf_pt")
                        rec["logprob"]["correction_method"] = "single_space_fix"
                        n_patched += 1
                        n_logprob_patched += 1

            n_patched_total += n_patched
            summary = summarize(per_item, methods=("paper", "ordered", "logprob"))
            traj.append({
                "step": step,
                "n_items_relabeled_by_judge_v2": n_patched - n_logprob_patched,
                "n_items_relabeled_by_logprob_singlespace_fix": n_logprob_patched,
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
                d["logprob_singlespace_correction_applied"] = bool(args.apply_logprob_fix)
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

    if args.apply_logprob_fix and n_skipped_still_ambiguous_total:
        print(f"\nn_skipped_still_ambiguous: {n_skipped_still_ambiguous_total} item(s) where "
              f"the corrected logprob is ALSO still AMBIG -- final_label was left UNCHANGED "
              f"for these (kept whatever it was: the v1 judge's answer, or the original "
              f"confident logprob call) rather than downgrading it to a bare 'AMBIG' "
              f"placeholder. If you want an actual resolution for these instead of leaving "
              f"the old answer in place, they'd need a fresh judge escalation in a follow-up "
              f"pass -- they're not currently in rerun_judge.py's scope.")

    if args.write_in_place:
        print("\nDone. Original checkpoint_step*_metrics.json files were overwritten "
              f"where they had relabeled items; pristine v1 originals are under {args.backup_dir}.")
    else:
        print("\nDone. Original checkpoint_step*_metrics.json files were NOT modified "
              "(pass --write-in-place to change that).")


if __name__ == "__main__":
    main()
