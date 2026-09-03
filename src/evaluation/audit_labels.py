import os
import sys
import re
import csv
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from eval import normalize as _eval_normalize, STOP  # noqa: E402

RESULTS_GLOB = "results/cpi-results/*-results/checkpoint_step*_metrics.json"

def normalize(t):
    return re.sub(r"[^\w\s]", "", _eval_normalize(t))


def sig_words(ans):
    return [w for w in normalize(ans).split() if len(w) > 3 and w not in STOP]


def discr_words(ans, other):
    o = set(normalize(other).split())
    dw = [w for w in sig_words(ans) if w not in o]
    return dw if dw else sig_words(ans)


def _bounded(w):
    return r"(?<!\w)" + re.escape(w) + r"(?!\w)"


def contains(resp, ans, words):
    r = normalize(resp)
    if not words:
        return bool(re.search(_bounded(normalize(ans)), r))
    return any(re.search(_bounded(w), r) for w in words)


def clean_response(resp):
    """Strip special/control tokens (</s>, <s>, repeated newlines) so a
    degenerate generation doesn't accidentally 'contain' an answer by luck,
    and so we can tell a truly empty response from a real one."""
    if resp is None:
        return ""
    r = re.sub(r"</?s>", " ", resp)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def classify_item(rec):
    """Return (verdict, evidence_dict) for one per-item record, or
    (None, None) if it can't be audited (no response / missing fields)."""
    resp = rec.get("response")
    par = rec.get("parametric_answer")
    cf = rec.get("counterfactual_answer")
    final = rec.get("final_label")
    if resp is None or par is None or cf is None or final not in ("CTX", "PAR", "OTHER"):
        return None, None

    resp_clean = clean_response(resp)
    pw, cw = discr_words(par, cf), discr_words(cf, par)
    has_par = contains(resp_clean, par, pw)
    has_cf = contains(resp_clean, cf, cw)

    evidence = {
        "response_clean": resp_clean[:200],
        "has_par_evidence": has_par,
        "has_cf_evidence": has_cf,
    }

    if not resp_clean:
        if final in ("CTX", "PAR"):
            return "degenerate_response", evidence
        return None, evidence

    if final == "CTX":
        if has_par and not has_cf:
            return "label_contradicted", evidence
        if not has_cf:
            return "label_not_supported", evidence
    elif final == "PAR":
        if has_cf and not has_par:
            return "label_contradicted", evidence
        if not has_par:
            return "label_not_supported", evidence
    else:  # OTHER
        if has_par and not has_cf:
            return "other_but_answered", evidence
        if has_cf and not has_par:
            return "other_but_answered", evidence

    return None, evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cpi-results/audit_report.json")
    ap.add_argument("--csv", default="results/cpi-results/audit_findings.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(RESULTS_GLOB))
    print(f"Found {len(files)} checkpoint metrics files")

    findings = []
    totals = {"n_items_seen": 0, "n_auditable": 0}
    by_category = {}

    for fp in files:
        run = "tulu" if "tulu-results" in fp else "alpaca"
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        step = d.get("global_step")
        for rec in d.get("per_item", []):
            totals["n_items_seen"] += 1
            verdict, evidence = classify_item(rec)
            if evidence is not None:
                totals["n_auditable"] += 1
            if verdict is None:
                continue
            by_category[verdict] = by_category.get(verdict, 0) + 1
            findings.append({
                "run": run, "step": step,
                "item_id": rec.get("item_id"), "source": rec.get("source"),
                "final_label": rec.get("final_label"),
                "logprob_label": rec.get("logprob", {}).get("label"),
                "judge_label": rec.get("judge", {}).get("label") if rec.get("judge") else None,
                "verdict": verdict,
                "parametric_answer": rec.get("parametric_answer"),
                "counterfactual_answer": rec.get("counterfactual_answer"),
                "response_clean": evidence["response_clean"],
            })

    print(f"\nTotal items seen : {totals['n_items_seen']}")
    print(f"Auditable items  : {totals['n_auditable']}  (had response+par+cf+valid final_label)")
    print(f"Flagged items    : {len(findings)}")
    print("\nBy category:")
    for k, v in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {k:22s} {v}")

    # per-item_id concentration -- which specific dataset items misfire most often
    by_item = {}
    for f in findings:
        by_item.setdefault(f["item_id"], []).append(f)
    worst = sorted(by_item.items(), key=lambda x: -len(x[1]))[:20]
    print("\nTop 20 item_ids by number of flagged checkpoints:")
    for item_id, fs in worst:
        cats = {}
        for f in fs:
            cats[f["verdict"]] = cats.get(f["verdict"], 0) + 1
        print(f"  {item_id:12s} ({fs[0]['source']:20s}) n_flagged={len(fs):3d}  {cats}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"totals": totals, "by_category": by_category, "findings": findings},
                   f, indent=2)
    print(f"\nFull findings JSON -> {args.out}")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "step", "item_id", "source", "verdict",
                                          "final_label", "logprob_label", "judge_label",
                                          "parametric_answer", "counterfactual_answer",
                                          "response_clean"])
        w.writeheader()
        for row in findings:
            w.writerow(row)
    print(f"Findings CSV       -> {args.csv}")


if __name__ == "__main__":
    main()
