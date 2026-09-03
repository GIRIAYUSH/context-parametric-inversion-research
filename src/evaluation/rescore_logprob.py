"""
Re-runs ONLY the Layer-1 log-prob scorer (score_answer_logprob), ONLY for the
items flagged in results/cpi-results/reprocess_manifest.json's `logprob_rerun`
list (root cause #2 in docs/label-audit-findings.md -- the scorer's bias
toward whichever candidate answer has more tokens).

Unlike rerun_judge.py, this NEEDS the actual model weights, because the
per-token log-probs behind the original `lp_par_pt`/`lp_cf_pt` numbers were
never saved -- only the aggregates were. So this loads each checkpoint's LoRA
adapter and re-scores just the flagged items (no generation, so it's much
cheaper than the original full eval -- one forward pass per candidate answer,
not autoregressive decoding).

Checkpoints live on Google Drive, not locally, and TULU and Alpaca are two
separate training runs synced to two separate Drive paths -- both are
independently configurable via environment variables so this script runs
unchanged whether you point it at a Drive mount or pull via rclone:

    Per-run checkpoint location (REQUIRED, one of these two per run):
        CPI_CKPT_DIR_ALPACA     e.g. /content/drive/MyDrive/cpi_study/checkpoints/
                                     llama2_7b_alpaca_seed0_lr1e-04/checkpoints
                                     (used directly if CPI_CKPT_ACCESS_MODE=local)
        CPI_CKPT_DIR_TULU       same idea for the TULU run

    CPI_CKPT_ACCESS_MODE        "drive_stage" (recommended for Colab) --
                                 CPI_CKPT_DIR_* is a `drive.mount()`-ed Google
                                 Drive path. This mode COPIES just the one
                                 needed checkpoint-<step> folder from Drive
                                 into local Colab disk (CPI_LOCAL_SCRATCH_DIR)
                                 before loading it -- reading model weights
                                 repeatedly off a Drive mount is slow and can
                                 be flaky, so every checkpoint gets staged
                                 locally for the duration it's needed, then
                                 deleted before moving to the next one.
                                "local" -- CPI_CKPT_DIR_* is already local disk
                                 (no Drive involved) -- read directly, no
                                 staging/copy step.
                                "rclone" -- CPI_CKPT_DIR_* is instead an rclone
                                 remote path (e.g. "gdrive:cpi_study/checkpoints/
                                 llama2_7b_tulu_seed0_lr1e-04/checkpoints"); this
                                 script `rclone copy`s just the one needed
                                 checkpoint-<step> folder into a local scratch
                                 dir before loading it, then deletes it after,
                                 so disk usage doesn't grow across checkpoints.

    CPI_BASE_MODEL_ALPACA,      HF id of the base model for that run. Defaults
    CPI_BASE_MODEL_TULU          to meta-llama/Llama-2-7b-hf (see congif.yaml's
                                 run.model for the run these checkpoints came
                                 from -- override if you used a different one).

    CPI_LOCAL_SCRATCH_DIR       Local dir for rclone-mode temp downloads.
                                 Default: ./_rescore_scratch

    HF_TOKEN / HUGGING_FACE_HUB_TOKEN   needed for the gated Llama-2 weights.

Usage (example, Colab, rclone mode):
    import os
    os.environ["CPI_CKPT_ACCESS_MODE"] = "rclone"
    os.environ["CPI_CKPT_DIR_ALPACA"] = "gdrive:cpi_study/checkpoints/llama2_7b_alpaca_seed0_lr1e-04/checkpoints"
    os.environ["CPI_CKPT_DIR_TULU"]   = "gdrive:cpi_study/checkpoints/llama2_7b_tulu_seed0_lr1e-04/checkpoints"
    os.environ["HF_TOKEN"] = "..."
    !python rescore_logprob.py --dataset ../../dataset/conflict_eval_unified.json

Usage (Drive already mounted):
    import os
    os.environ["CPI_CKPT_DIR_ALPACA"] = "/content/drive/MyDrive/cpi_study/checkpoints/llama2_7b_alpaca_seed0_lr1e-04/checkpoints"
    os.environ["CPI_CKPT_DIR_TULU"]   = "/content/drive/MyDrive/cpi_study/checkpoints/llama2_7b_tulu_seed0_lr1e-04/checkpoints"
    !python rescore_logprob.py --dataset ../../dataset/conflict_eval_unified.json
"""
import os
import sys
import gc
import json
import shutil
import argparse
import subprocess

import torch
try:
    import torch.distributed.tensor  # noqa: F401  (peft DTensor workaround, see run_sft.py)
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from eval import build_score_prompt_ids, load_items, TAU  # noqa: E402

DEFAULT_BASE_MODEL = {
    "alpaca": os.environ.get("CPI_BASE_MODEL_ALPACA", "meta-llama/Llama-2-7b-hf"),
    "tulu": os.environ.get("CPI_BASE_MODEL_TULU", "meta-llama/Llama-2-7b-hf"),
}


def get_ckpt_root(run):
    env_key = f"CPI_CKPT_DIR_{run.upper()}"
    val = os.environ.get(env_key)
    if not val:
        raise RuntimeError(
            f"{env_key} is not set. Point it at where the {run} run's checkpoints "
            f"live (a Drive mount path, or an rclone remote path if "
            f"CPI_CKPT_ACCESS_MODE=rclone)."
        )
    return val


def ckpt_subdir_name(step):
    # run_sft.py saves numbered steps as 'checkpoint-<N>' and the very last
    # adapter as 'final' -- the manifest records that one as step "inf"
    # (see evaluate_checkpoint's checkpoint_id convention).
    return "final" if step == "inf" else f"checkpoint-{step}"


def resolve_local_checkpoint_dir(run, step, access_mode, scratch_dir):
    root = get_ckpt_root(run)
    sub = ckpt_subdir_name(step)

    if access_mode == "local":
        local_dir = os.path.join(root, sub)
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Checkpoint dir not found: {local_dir}")
        return local_dir, False  # False = don't delete after use, it's not ours

    if access_mode == "drive_stage":
        # root is a drive.mount()-ed path; copy just this one checkpoint
        # folder onto local (non-Drive) disk before loading from it, then the
        # caller deletes the local copy once this checkpoint is done.
        drive_path = os.path.join(root, sub)
        if not os.path.isdir(drive_path):
            raise FileNotFoundError(f"Checkpoint dir not found on Drive: {drive_path}")
        local_dir = os.path.join(scratch_dir, run, sub)
        if os.path.isdir(local_dir):
            shutil.rmtree(local_dir, ignore_errors=True)
        print(f"    [stage] copying {drive_path} -> {local_dir}")
        shutil.copytree(drive_path, local_dir)
        return local_dir, True  # True = ours, safe to delete after use

    # rclone mode: pull just this one checkpoint folder down
    remote_path = f"{root.rstrip('/')}/{sub}"
    local_dir = os.path.join(scratch_dir, run, sub)
    os.makedirs(local_dir, exist_ok=True)
    print(f"    [rclone] copying {remote_path} -> {local_dir}")
    subprocess.run(["rclone", "copy", remote_path, local_dir, "--fast-list"], check=True)
    return local_dir, True  # True = ours, safe to delete after use


@torch.no_grad()
def score_answer_logprob_tokens(model, tok, prompt_ids, answer_text):
    """Same computation as eval.py's score_answer_logprob (same space-prefix
    handling, same trailing-EOS strip, same indexing), but returns the FULL
    per-token log-prob tensor instead of collapsing it to a sum -- so callers
    can apply more than one normalization from a single forward pass, without
    needing eval.py itself to change (that function's callers elsewhere still
    want the original v1 behavior, unmodified, for reproducibility)."""
    answer_text = " " + answer_text.strip()
    ans = tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ans.shape[0] == 0:
        return torch.zeros(0)
    if prompt_ids.numel() > 0 and prompt_ids[-1].item() == tok.eos_token_id:
        prompt_ids = prompt_ids[:-1]
    full = torch.cat([prompt_ids, ans]).unsqueeze(0).to(model.device)
    logits = model(full).logits[0]
    n = prompt_ids.shape[0]
    lp = torch.log_softmax(logits[n - 1:-1, :].float(), dim=-1)
    tok_lp = lp[torch.arange(ans.shape[0]), ans.to(lp.device)]
    return tok_lp.cpu()


def _avg(tok_lp, drop_first):
    """Per-token average log-prob, optionally excluding the first scored
    token. Root cause #2 (docs/label-audit-findings.md): a candidate's first
    token carries the most uncertainty (the model hasn't 'committed' to that
    continuation yet), and with only 2-3 total tokens that one token
    dominates the average far more than it does for a longer candidate --
    systematically penalizing shorter (often correct) answers. Falls back to
    the full average if there's only one token to begin with (nothing left
    to average after dropping it)."""
    if tok_lp.shape[0] == 0:
        return None, 0, False
    t = tok_lp[1:] if (drop_first and tok_lp.shape[0] > 1) else tok_lp
    degenerate = drop_first and tok_lp.shape[0] <= 1
    return (t.sum().item() / t.shape[0]), t.shape[0], degenerate


def _label(delta, tau):
    return "AMBIG" if abs(delta) < tau else ("CTX" if delta > 0 else "PAR")


def load_base_model(model_id, hf_token, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", token=hf_token)
    return model, tok


def _score_pair(tok_lp_par, tok_lp_cf, tau, drop_first):
    par_pt, npar, par_degenerate = _avg(tok_lp_par, drop_first)
    cf_pt, ncf, cf_degenerate = _avg(tok_lp_cf, drop_first)
    if par_pt is None or cf_pt is None:
        return {"delta": 0.0, "label": "AMBIG", "error": "empty_answer",
                "n_par": npar, "n_cf": ncf}
    d = cf_pt - par_pt
    result = {"delta": round(d, 4), "label": _label(d, tau),
              "lp_par_pt": round(par_pt, 3), "lp_cf_pt": round(cf_pt, 3),
              "n_par": npar, "n_cf": ncf}
    if par_degenerate or cf_degenerate:
        # one candidate was a single token, so "drop first" fell back to the
        # full average for it -- the two candidates aren't on quite the same
        # footing here, flag it rather than silently presenting it as clean.
        result["degenerate_dropfirst"] = True
    return result


def rescore_one_checkpoint(base_model, tokenizer, ckpt_dir, entries, items_by_id, use_ct=True, tau=TAU):
    """entries: manifest rows for this single checkpoint. Returns list of result dicts.

    Computes TWO scores per item from the SAME forward passes (one per
    candidate answer, same as v1 -- no extra model calls):
      - new_logprob            : v1's exact formula (full per-token average),
                                  kept as a reproducibility check.
      - new_logprob_dropfirst  : the length-bias correction (root cause #2) --
                                  same per-token log-probs, but the average
                                  excludes each candidate's first (hardest,
                                  least length-comparable) token.
    `recommended_final_label` is the dropfirst label -- this is what
    apply_patches.py's --apply-logprob-fix mode writes into final_label.
    """
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()

    out = []
    try:
        with torch.no_grad():
            for e in entries:
                it = items_by_id.get(e["item_id"])
                if it is None:
                    print(f"      !! item {e['item_id']} not found in dataset, skipping")
                    continue
                q, ctx = it["question"], it["context"]
                par, cf = it["parametric_answer"], it["counterfactual_answer"]

                score_ids = build_score_prompt_ids(tokenizer, q, ctx, use_ct)
                tok_lp_par = score_answer_logprob_tokens(model, tokenizer, score_ids, par)
                tok_lp_cf = score_answer_logprob_tokens(model, tokenizer, score_ids, cf)

                new_logprob = _score_pair(tok_lp_par, tok_lp_cf, tau, drop_first=False)
                new_logprob_dropfirst = _score_pair(tok_lp_par, tok_lp_cf, tau, drop_first=True)

                out.append({
                    "run": e["run"], "step": e["step"], "item_id": e["item_id"],
                    "source": e["source"],
                    "old_logprob_label": e.get("current_logprob_label"),
                    "old_final_label": e.get("current_final_label"),
                    "new_logprob": new_logprob,
                    "new_logprob_dropfirst": new_logprob_dropfirst,
                    "recommended_final_label": new_logprob_dropfirst["label"],
                })
    finally:
        model = model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/cpi-results/reprocess_manifest.json")
    ap.add_argument("--dataset", default="dataset/conflict_eval_unified.json")
    ap.add_argument("--out", default="results/cpi-results/logprob_rescore_patch.json")
    ap.add_argument("--access-mode", default=os.environ.get("CPI_CKPT_ACCESS_MODE", "drive_stage"),
                     choices=["local", "drive_stage", "rclone"])
    ap.add_argument("--scratch-dir", default=os.environ.get("CPI_LOCAL_SCRATCH_DIR", "./_rescore_scratch"))
    ap.add_argument("--use-chat-template", action="store_true", default=True)
    ap.add_argument("--limit-checkpoints", type=int, default=None,
                     help="Only process the first N checkpoint groups (smoke test).")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    todo = manifest["logprob_rerun"]

    # apply dataset_fixes (e.g. cap_0181) before building the item lookup, so
    # rescoring uses the corrected gold answer, not the flagged bad one.
    items = load_items(args.dataset)
    items_by_id = {it["item_id"]: it for it in items}
    for fix in manifest.get("dataset_fixes", []):
        it = items_by_id.get(fix["item_id"])
        if it and it.get(fix["field"]) == fix["old"]:
            it[fix["field"]] = fix["new"]
            print(f"Applied dataset fix: {fix['item_id']}.{fix['field']} "
                  f"'{fix['old']}' -> '{fix['new']}'")

    by_ckpt = {}
    for e in todo:
        by_ckpt.setdefault((e["run"], e["step"]), []).append(e)
    # step is usually an int (numeric checkpoint) but the final adapter is
    # recorded as the literal string "inf" (see ckpt_subdir_name / the
    # manifest's step field) -- plain sorted() can't compare str to int, so
    # give "inf" a sort key that actually sorts last, same intent as the name.
    def _step_sort_key(rs):
        run, step = rs
        return (run, float("inf") if step == "inf" else float(step))

    ckpt_keys = sorted(by_ckpt.keys(), key=_step_sort_key)
    if args.limit_checkpoints:
        ckpt_keys = ckpt_keys[:args.limit_checkpoints]

    print(f"{len(todo)} item(s) across {len(ckpt_keys)} checkpoint(s) to rescore")

    loaded_bases = {}  # run -> (model, tokenizer), loaded once and reused
    all_results = []

    for i, (run, step) in enumerate(ckpt_keys, 1):
        entries = by_ckpt[(run, step)]
        print(f"[{i}/{len(ckpt_keys)}] {run} step={step}  ({len(entries)} item(s))")

        if run not in loaded_bases:
            print(f"  loading base model for '{run}': {DEFAULT_BASE_MODEL[run]}")
            loaded_bases[run] = load_base_model(DEFAULT_BASE_MODEL[run], hf_token)
        base_model, tokenizer = loaded_bases[run]

        try:
            local_dir, is_ours = resolve_local_checkpoint_dir(
                run, step, args.access_mode, args.scratch_dir)
        except Exception as ex:
            print(f"  !! could not resolve checkpoint dir for {run} step {step}: {ex}")
            continue

        try:
            results = rescore_one_checkpoint(
                base_model, tokenizer, local_dir, entries, items_by_id,
                use_ct=args.use_chat_template)
            all_results.extend(results)
        finally:
            if is_ours and os.path.isdir(local_dir):
                shutil.rmtree(local_dir, ignore_errors=True)

    n_reproduction_mismatches = sum(1 for r in all_results
                                     if r["new_logprob"]["label"] != r["old_logprob_label"])
    n_fixed = sum(1 for r in all_results
                  if r["recommended_final_label"] != r["old_final_label"])
    n_degenerate = sum(1 for r in all_results
                        if r["new_logprob_dropfirst"].get("degenerate_dropfirst"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "n_processed": len(all_results),
            "n_reproduction_mismatches": n_reproduction_mismatches,
            "n_would_change_final_label": n_fixed,
            "n_degenerate_dropfirst": n_degenerate,
            "note": "new_logprob reproduces the ORIGINAL (v1) scoring formula -- near-0 "
                    "n_reproduction_mismatches just confirms determinism. "
                    "new_logprob_dropfirst / recommended_final_label is the length-bias "
                    "correction (root cause #2): drops each candidate's first token before "
                    "averaging. n_degenerate_dropfirst counts items where one candidate was "
                    "a single token (dropfirst fell back to the full average for it -- worth "
                    "a manual glance, the two candidates aren't on quite equal footing there).",
            "results": all_results,
        }, f, indent=2)

    print(f"\nProcessed: {len(all_results)}")
    print(f"Reproduction mismatches vs originally-recorded label : {n_reproduction_mismatches} "
          f"(expected ~0 -- confirms determinism, not the fix)")
    print(f"Would change final_label (dropfirst correction)      : {n_fixed} / {len(all_results)}")
    print(f"Degenerate dropfirst (single-token candidate)         : {n_degenerate}")
    print(f"Patch written -> {args.out}")


if __name__ == "__main__":
    main()
