
import os
import json
import argparse


def load_checkpoint(results_root, run, step):
    path = os.path.join(results_root, f"{run}-results", f"checkpoint_step{step}_metrics.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_flip_set(early_rows, late_rows, direction="ctx_to_par"):
    early_map = {r["item_id"]: r for r in early_rows if r.get("passed")}
    late_map = {r["item_id"]: r for r in late_rows if r.get("passed")}
    common_ids = sorted(set(early_map) & set(late_map))

    flips = []
    for iid in common_ids:
        e, l = early_map[iid], late_map[iid]
        ef, lf = e.get("final_label"), l.get("final_label")
        is_flip = {
            "ctx_to_par": ef == "CTX" and lf == "PAR",
            "par_to_ctx": ef == "PAR" and lf == "CTX",
            "any_flip": ef != lf,
        }[direction]
        if not is_flip:
            continue
        flips.append({
            "item_id": iid,
            "source": e["source"],
            "question": e["question"],
            "context": e["context"],
            "parametric_answer": e["parametric_answer"],
            "counterfactual_answer": e["counterfactual_answer"],
            "early_final_label": ef,
            "late_final_label": lf,
            "early_response": e.get("response"),
            "late_response": l.get("response"),
            "early_logprob": e.get("logprob"),
            "late_logprob": l.get("logprob"),
        })
    return flips, len(common_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["alpaca", "tulu"])
    ap.add_argument("--early-step", required=True)
    ap.add_argument("--late-step", required=True)
    ap.add_argument("--results-root", default="results/cpi-results")
    ap.add_argument("--direction", default="ctx_to_par",
                     choices=["ctx_to_par", "par_to_ctx", "any_flip"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    early = load_checkpoint(args.results_root, args.run, args.early_step)
    late = load_checkpoint(args.results_root, args.run, args.late_step)

    flips, n_common = build_flip_set(early["per_item"], late["per_item"], args.direction)

    out = {
        "run": args.run,
        "early_step": args.early_step,
        "late_step": args.late_step,
        "direction": args.direction,
        "n_common_passed_both": n_common,
        "n_flips": len(flips),
        "flip_rate": round(len(flips) / max(n_common, 1), 4),
        "items": flips,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{args.run} step{args.early_step} -> step{args.late_step} "
          f"({args.direction}): {len(flips)}/{n_common} items flipped "
          f"({out['flip_rate']:.1%})")
    print(f"Wrote {args.output}")

    by_source = {}
    for it in flips:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
    print("By stratum:", by_source)


if __name__ == "__main__":
    main()
