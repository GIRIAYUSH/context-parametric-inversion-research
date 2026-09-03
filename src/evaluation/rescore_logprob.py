"""
Re-runs the Layer-1 log-prob scorer for EVERY item in EVERY checkpoint, both
runs (~48,204 item-checkpoint pairs) -- not just the subset the audit had
flagged. Scope was widened after discovering a double-leading-space bug in
whatever version of score_answer_logprob produced the currently-stored
checkpoint_step*_metrics.json files: every candidate answer in the ENTIRE
original dataset was scored with one extra phantom space token (id 29871 for
the Llama-2 tokenizer) prepended, which dilutes a short answer's per-token
average far more than a long answer's -- see docs/label-audit-findings.md
root cause #2, and the "double-space" finding layered on top of it. Fixing
that alone requires re-scoring everything, not just the previously-flagged
items, so this no longer reads results/cpi-results/reprocess_manifest.json's
`logprob_rerun` list -- it discovers checkpoints directly from the local
checkpoint_step*_metrics.json files and rescoring every item recorded in each.

Two scores are computed per item, both already using the CORRECT (single,
not double) leading space:
  - new_logprob            : full per-token average -- the length-bias
                              mechanism is still present here in its "natural"
                              form (root cause #2's first-token dilution),
                              just without the extra bug-injected token.
                              `recommended_final_label` uses THIS one.
  - new_logprob_dropfirst   : ALSO excludes each candidate's first token
                              before averaging -- this was the originally
                              planned length-bias correction, but a
                              full-checkpoint test run rejected it: 87% of
                              items changed label (vs new_logprob's much more
                              believable 25%), and 76% of those changes were
                              just CTX/PAR -> AMBIG, on items that ALREADY had
                              similar par/cf token counts (no length-bias
                              problem to begin with). Most of these answers
                              are only 2-4 tokens; dropping the first often
                              leaves just 1 token to "average," which isn't a
                              length correction anymore, it's discarding
                              almost all the signal and averaging noise --
                              mechanically collapsing delta toward zero
                              regardless of whether length bias was present.
                              Kept in the output for transparency/diagnosis
                              only -- NOT used for recommended_final_label.

No generation, so this is much cheaper per-item than the original full eval
(2 forward passes per item, not autoregressive decoding) -- but the full
48k-item scope means this is a genuinely long job spanning many checkpoints.
Per-checkpoint resumability: each checkpoint's output is written to its own
file in --out-dir, and a checkpoint already having an output file is skipped
on a re-run (pass --no-resume to force reprocessing). If a Colab session
times out partway through, just re-run the same command.

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

Usage:
    python rescore_logprob.py --results-root ../../results/cpi-results \
        --out-dir ../../results/cpi-results/logprob_v2_full

    # smoke test on just the first checkpoint:
    python rescore_logprob.py --results-root ../../results/cpi-results \
        --out-dir ../../results/cpi-results/logprob_v2_full --limit-checkpoints 1
"""
import os
import re
import sys
import gc
import json
import glob
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
from eval import build_score_prompt_ids, TAU  # noqa: E402

DEFAULT_BASE_MODEL = {
    "alpaca": os.environ.get("CPI_BASE_MODEL_ALPACA", "meta-llama/Llama-2-7b-hf"),
    "tulu": os.environ.get("CPI_BASE_MODEL_TULU", "meta-llama/Llama-2-7b-hf"),
}

_CKPT_FILE_RE = re.compile(r"checkpoint_step(\w+)_metrics\.json$")


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
    # adapter as 'final' -- checkpoint_stepinf_metrics.json records that one
    # with the literal step value "inf".
    return "final" if step == "inf" else f"checkpoint-{step}"


def discover_checkpoints(results_root):
    """Find every (run, step, metrics_file_path) directly from the local
    checkpoint_step*_metrics.json files -- this is the full-dataset scope,
    not just the previously-flagged manifest subset."""
    found = []
    for run in ("alpaca", "tulu"):
        run_dir = os.path.join(results_root, f"{run}-results")
        for fp in sorted(glob.glob(os.path.join(run_dir, "checkpoint_step*_metrics.json"))):
            m = _CKPT_FILE_RE.search(os.path.basename(fp))
            if not m:
                continue
            step_raw = m.group(1)
            step = step_raw if step_raw == "inf" else int(step_raw)
            found.append((run, step, fp))

    def _sort_key(rsf):
        run, step, _ = rsf
        return (run, float("inf") if step == "inf" else float(step))

    return sorted(found, key=_sort_key)


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
    handling -- strip then prepend exactly ONE leading space, same
    trailing-EOS strip, same indexing), but returns the FULL per-token
    log-prob tensor instead of collapsing it to a sum, so callers can apply
    more than one normalization from a single forward pass. NOTE: this is the
    CORRECT single-space behavior -- the checkpoint files on disk were scored
    with an extra double-space bug (see module docstring); this function does
    not reproduce that bug, it fixes it."""
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


def rescore_full_checkpoint(base_model, tokenizer, ckpt_dir, per_item, dataset_fixes,
                             use_ct=True, tau=TAU):
    """Rescores EVERY item in `per_item` (a checkpoint file's own per_item
    list -- already has question/context/parametric_answer/
    counterfactual_answer/logprob/final_label recorded verbatim from the
    original run, so no separate dataset load is needed; dataset_fixes are
    applied as overrides here instead)."""
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()

    out = []
    try:
        with torch.no_grad():
            for idx, rec in enumerate(per_item):
                q, ctx = rec["question"], rec["context"]
                par = dataset_fixes.get((rec["item_id"], "parametric_answer"),
                                         rec["parametric_answer"])
                cf = dataset_fixes.get((rec["item_id"], "counterfactual_answer"),
                                        rec["counterfactual_answer"])

                score_ids = build_score_prompt_ids(tokenizer, q, ctx, use_ct)
                tok_lp_par = score_answer_logprob_tokens(model, tokenizer, score_ids, par)
                tok_lp_cf = score_answer_logprob_tokens(model, tokenizer, score_ids, cf)

                new_logprob = _score_pair(tok_lp_par, tok_lp_cf, tau, drop_first=False)
                new_logprob_dropfirst = _score_pair(tok_lp_par, tok_lp_cf, tau, drop_first=True)

                out.append({
                    "item_id": rec["item_id"], "source": rec.get("source"),
                    "old_logprob": rec.get("logprob"),
                    "old_final_label": rec.get("final_label"),
                    "new_logprob": new_logprob,
                    "new_logprob_dropfirst": new_logprob_dropfirst,
                    # recommended_final_label uses the single-space fix (new_logprob),
                    # NOT the dropfirst correction -- see module docstring: dropfirst
                    # was tested at full-checkpoint scale and rejected. new_logprob_dropfirst
                    # is kept in the output for transparency/diagnosis only.
                    "recommended_final_label": new_logprob["label"],
                })
                if (idx + 1) % 100 == 0:
                    print(f"      ...{idx + 1}/{len(per_item)} items")
    finally:
        model = model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/cpi-results")
    ap.add_argument("--out-dir", default="results/cpi-results/logprob_v2_full",
                     help="One output file per checkpoint gets written here, named "
                          "<run>_step<step>_logprob_v2.json.")
    ap.add_argument("--manifest", default="results/cpi-results/reprocess_manifest.json",
                     help="Only used for dataset_fixes (e.g. cap_0181's corrected gold "
                          "answer) -- pass --manifest '' to skip.")
    ap.add_argument("--access-mode", default=os.environ.get("CPI_CKPT_ACCESS_MODE", "drive_stage"),
                     choices=["local", "drive_stage", "rclone"])
    ap.add_argument("--scratch-dir", default=os.environ.get("CPI_LOCAL_SCRATCH_DIR", "./_rescore_scratch"))
    ap.add_argument("--use-chat-template", action="store_true", default=True)
    ap.add_argument("--limit-checkpoints", type=int, default=None,
                     help="Only process the first N checkpoints (smoke test).")
    ap.add_argument("--no-resume", action="store_true",
                     help="Re-process a checkpoint even if its output file already exists.")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    dataset_fixes = {}
    if args.manifest and os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest, encoding="utf-8"))
        for fix in manifest.get("dataset_fixes", []):
            dataset_fixes[(fix["item_id"], fix["field"])] = fix["new"]
        if dataset_fixes:
            print(f"Loaded {len(dataset_fixes)} dataset fix(es) from {args.manifest}: "
                  f"{list(dataset_fixes.keys())}")

    ckpts = discover_checkpoints(args.results_root)
    if args.limit_checkpoints:
        ckpts = ckpts[:args.limit_checkpoints]
    print(f"{len(ckpts)} checkpoint(s) discovered across both runs")

    os.makedirs(args.out_dir, exist_ok=True)
    loaded_bases = {}  # run -> (model, tokenizer), loaded once and reused
    n_total_items = 0
    n_total_reproduction_mismatches = 0
    n_total_would_change = 0

    for i, (run, step, fp) in enumerate(ckpts, 1):
        out_path = os.path.join(args.out_dir, f"{run}_step{step}_logprob_v2.json")
        if os.path.exists(out_path) and not args.no_resume:
            print(f"[{i}/{len(ckpts)}] {run} step={step}: already done, skipping "
                  f"(--no-resume to redo)")
            continue

        d = json.load(open(fp, encoding="utf-8"))
        per_item = d.get("per_item", [])
        print(f"[{i}/{len(ckpts)}] {run} step={step} ({len(per_item)} item(s))")

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
            results = rescore_full_checkpoint(
                base_model, tokenizer, local_dir, per_item, dataset_fixes,
                use_ct=args.use_chat_template)
        finally:
            if is_ours and os.path.isdir(local_dir):
                shutil.rmtree(local_dir, ignore_errors=True)

        n_repro_mismatch = sum(1 for r in results
                                if r["new_logprob"]["label"] !=
                                (r["old_logprob"] or {}).get("label"))
        n_would_change = sum(1 for r in results
                              if r["recommended_final_label"] != r["old_final_label"])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "run": run, "step": step, "n_items": len(results),
                "n_reproduction_mismatches": n_repro_mismatch,
                "n_would_change_final_label": n_would_change,
                "results": results,
            }, f, indent=2)

        n_total_items += len(results)
        n_total_reproduction_mismatches += n_repro_mismatch
        n_total_would_change += n_would_change
        print(f"  -> {out_path}  (repro_mismatch={n_repro_mismatch}, "
              f"would_change={n_would_change}/{len(results)})")

    print(f"\nDone this session. Items processed: {n_total_items}   "
          f"Reproduction mismatches: {n_total_reproduction_mismatches}   "
          f"Would change final_label: {n_total_would_change}")
    print(f"Per-checkpoint patches -> {args.out_dir}/")
    print("Next: merge_logprob_v2.py combines these into one patch for apply_patches.py.")


if __name__ == "__main__":
    main()
