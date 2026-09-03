"""
Re-runs ONLY the Layer-2 judge, ONLY for the items flagged in
results/cpi-results/reprocess_manifest.json's `judge_rerun` list, using the
`response`/`question`/`context`/`parametric_answer`/`counterfactual_answer`
already saved in each checkpoint's metrics JSON.

No model, no GPU, no checkpoint files, no Drive needed for this one -- it's
pure API calls against text that already exists on disk. Uses the eval.py
`prompt_version="v2"` judge prompt (see docs/label-audit-findings.md, root
cause #1) which fixes the ~24:1 PAR bias found in the original v1 judge output.

This does NOT modify the original checkpoint_step*_metrics.json files -- it
writes a separate patch file mapping (run, step, item_id) -> corrected judge
verdict + final_label, so the original raw eval output stays intact and
`apply_patches.py` can merge this on top when regenerating trajectories.

Usage:
    export OPENAI_API_KEY=...
    python rerun_judge.py [--manifest results/cpi-results/reprocess_manifest.json]
                          [--out results/cpi-results/judge_v2_patch.json]
                          [--limit N]   # for a quick smoke test before the full run
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from eval import make_judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/cpi-results/reprocess_manifest.json")
    ap.add_argument("--out", default="results/cpi-results/judge_v2_patch.json")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N manifest entries (smoke test).")
    ap.add_argument("--model", default=None, help="Override judge model (default: eval.py's JUDGE_MODEL).")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    todo = manifest["judge_rerun"]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} item(s) to re-judge (v2 prompt)")

    # `checkpoint_file` entries in the manifest are repo-root-relative (e.g.
    # "results/cpi-results/alpaca-results/checkpoint_step100_metrics.json"),
    # written by build_reprocess_manifest.py which assumed it'd be run from
    # the repo root. This script is commonly run from src/evaluation/ instead
    # (that's what the notebook does), so resolve them against wherever
    # --manifest itself actually lives rather than the current directory --
    # the manifest is always at "<repo_root>/results/cpi-results/reprocess_manifest.json",
    # so walking up two levels from it gives the repo root regardless of cwd.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.manifest))))

    kwargs = {"prompt_version": "v2"}
    if args.model:
        kwargs["model"] = args.model
    judge_fn_ = make_judge(**kwargs)

    # group by checkpoint file so each metrics JSON is only opened once
    by_file = {}
    for e in todo:
        by_file.setdefault(os.path.join(repo_root, e["checkpoint_file"]), []).append(e)

    patch = {}  # f"{run}|{step}|{item_id}" -> corrected record
    n_changed = 0
    n_done = 0
    n_missing_file = 0

    for fp, entries in by_file.items():
        if not os.path.exists(fp):
            print(f"  !! missing file, skipping {len(entries)} item(s): {fp}")
            n_missing_file += len(entries)
            continue
        d = json.load(open(fp, encoding="utf-8"))
        items = {r["item_id"]: r for r in d.get("per_item", [])}

        for e in entries:
            rec = items.get(e["item_id"])
            if rec is None or "response" not in rec:
                print(f"  !! item {e['item_id']} not found / no response in {fp}")
                continue

            old_final = rec.get("final_label")
            old_judge_label = rec.get("judge", {}).get("label") if rec.get("judge") else None

            v2 = judge_fn_(
                rec["response"], rec["question"], rec["context"],
                rec["parametric_answer"], rec["counterfactual_answer"],
                item_id=e["item_id"], checkpoint_id=f"{e['run']}_{e['step']}_v2rerun",
            )

            key = f"{e['run']}|{e['step']}|{e['item_id']}"
            patch[key] = {
                "run": e["run"], "step": e["step"], "item_id": e["item_id"],
                "source": e["source"],
                "old_judge_label": old_judge_label,
                "old_final_label": old_final,
                "new_judge_label": v2["label"],
                "new_final_label": v2["label"],  # judge output IS final_label when escalated
                "new_judge_other_subtype": v2.get("other_subtype"),
                "layer2_failed": v2.get("layer2_failed", False),
            }
            n_done += 1
            if v2["label"] != old_judge_label:
                n_changed += 1

        if n_done % 200 < len(entries):  # rough periodic progress print
            print(f"  ...{n_done}/{len(todo)} done ({n_changed} changed so far)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_version": "v2",
            "model": kwargs.get("model", "eval.py default (JUDGE_MODEL)"),
            "n_processed": n_done,
            "n_label_changed": n_changed,
            "n_missing_file": n_missing_file,
            "patch": patch,
        }, f, indent=2)

    print(f"\nProcessed: {n_done}   Changed vs v1: {n_changed}   Missing files: {n_missing_file}")
    print(f"Patch written -> {args.out}")
    print("Next: run apply_patches.py to merge this into corrected trajectory summaries.")


if __name__ == "__main__":
    main()
