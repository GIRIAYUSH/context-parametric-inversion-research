"""
Combines rescore_logprob.py's per-checkpoint output files
(results/cpi-results/logprob_v2_full/<run>_step<step>_logprob_v2.json, one
per checkpoint, written incrementally so the long full-dataset rescore is
resumable) into the single flat patch file apply_patches.py expects:
    results/cpi-results/logprob_rescore_patch.json

Run this once rescore_logprob.py has processed all the checkpoints you want
included (it's fine to merge a partial set -- e.g. to sanity-check progress
midway through a long run; just re-run this again once more checkpoints
finish, it always rebuilds the merged file from whatever per-checkpoint files
currently exist).

Usage:
    python merge_logprob_v2.py \
        --in-dir results/cpi-results/logprob_v2_full \
        --out results/cpi-results/logprob_rescore_patch.json
"""
import os
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/cpi-results/logprob_v2_full")
    ap.add_argument("--out", default="results/cpi-results/logprob_rescore_patch.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*_logprob_v2.json")))
    print(f"Found {len(files)} per-checkpoint file(s) in {args.in_dir}")

    all_results = []
    n_repro_mismatch = 0
    n_would_change = 0
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        run, step = d["run"], d["step"]
        for r in d["results"]:
            r["run"] = run
            r["step"] = step
            all_results.append(r)
        n_repro_mismatch += d.get("n_reproduction_mismatches", 0)
        n_would_change += d.get("n_would_change_final_label", 0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "n_processed": len(all_results),
            "n_reproduction_mismatches": n_repro_mismatch,
            "n_would_change_final_label": n_would_change,
            "note": "Merged from per-checkpoint files in " + args.in_dir + ". "
                    "new_logprob is the CORRECTED single-space reproduction (not the "
                    "original double-space-bugged formula) -- see rescore_logprob.py's "
                    "module docstring. new_logprob_dropfirst / recommended_final_label "
                    "additionally applies the length-bias (root cause #2) correction.",
            "results": all_results,
        }, f, indent=2)

    print(f"Merged {len(all_results)} item(s) from {len(files)} checkpoint(s)")
    print(f"Reproduction mismatches (vs original, double-space-bugged labels): {n_repro_mismatch}")
    print(f"Would change final_label (dropfirst correction)                  : {n_would_change}")
    print(f"Patch written -> {args.out}")


if __name__ == "__main__":
    main()
