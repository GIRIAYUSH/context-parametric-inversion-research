#!/usr/bin/env python
# =============================================================================
# eval_checkpoints.py
# -----------------------------------------------------------------------------
# Standalone standard-benchmark eval for LoRA checkpoints produced by
# run_sft.py, reproducing the paper's "ID accuracy" tracking
# (GSM8k / MMLU / SQuAD / ARC-Challenge via lm-eval-harness) on ALREADY-SAVED
# checkpoints instead of in-loop during training.
#
# Use this when:
#   * in-loop standard_eval was disabled or skipped for some steps,
#   * you pulled checkpoints from Google Drive into a fresh Colab runtime and
#     want to (re)run / extend the benchmark eval there,
#   * you want a different task list / limit / few-shot setting than what was
#     used during training, without re-running SFT.
#
# It reuses config.yaml's `run`, `models`, `datasets`, and `standard_eval`
# blocks so results stay consistent with run_sft.py's own in-loop eval
# (same task list, same GSM8k-drop-for-alpaca rule, same metric extraction).
#
# OUTPUT: one JSON file per checkpoint, e.g.
#   <results-dir>/id_eval_step000050.json
#   <results-dir>/id_eval_step000100.json
#   ...
# plus a rolling `id_eval_all.csv` with one row per checkpoint for easy
# plotting (step, per-task metric, id_accuracy_mean).
#
# USAGE
#   python eval_checkpoints.py --config config.yaml
#
#   # override any path without touching config.yaml:
#   python eval_checkpoints.py --config config.yaml \
#       --checkpoints-dir /content/drive/MyDrive/cpi_study/checkpoints/llama2_7b_alpaca_seed0_lr1e-04/checkpoints \
#       --results-dir /content/drive/MyDrive/cpi_study/checkpoints/llama2_7b_alpaca_seed0_lr1e-04/standard_eval \
#       --benchmark-data-dir /content/hf_datasets_cache \
#       --tasks mmlu,arc_challenge,squadv2 \
#       --limit 1000
#
# REQUIREMENTS
#   pip install "transformers>=4.44,<4.46" "peft>=0.12" "accelerate>=0.33" \
#               "lm-eval>=0.4.3" pyyaml torch
# =============================================================================

import os
import gc
import re
import csv
import glob
import json
import argparse
from datetime import datetime, timezone

import yaml
import torch
try:
    import torch.distributed.tensor  # noqa: F401  (peft DTensor workaround, see run_sft.py)
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# =============================================================================
# 0. Config + CLI
# =============================================================================
def load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                     help="config.yaml used for the SFT run (same file run_sft.py used). Optional if all paths given via CLI.")

    # ---- configurable paths (the two you asked for) ----
    ap.add_argument("--checkpoints-dir", default=None,
                     help="Directory containing checkpoint-* subfolders. "
                          "Defaults to <output_root>/<run_tag>/checkpoints from config.yaml.")
    ap.add_argument("--results-dir", default=None,
                     help="Where per-checkpoint JSON results are written. "
                          "Defaults to <output_root>/<run_tag>/standard_eval from config.yaml.")
    ap.add_argument("--benchmark-data-dir", default=None,
                     help="Local cache dir for benchmark datasets (HF datasets cache). "
                          "Sets HF_DATASETS_CACHE / HF_HOME so lm-eval-harness reuses it "
                          "across checkpoints and across Colab sessions (mount this on Drive "
                          "to avoid re-downloading MMLU/SQuAD/etc. every runtime).")

    # ---- overrides for eval behavior (fall back to config's standard_eval block) ----
    ap.add_argument("--base-model", default=None, help="Override HF id of the base model.")
    ap.add_argument("--tasks", default=None, help="Comma-separated task list, e.g. mmlu,arc_challenge,gsm8k,squadv2")
    ap.add_argument("--limit", type=int, default=None, help="Cap examples per task (subset). Omit for full sets.")
    ap.add_argument("--batch-size", default=None, help="'auto' or an int.")
    ap.add_argument("--num-fewshot-gsm8k", type=int, default=None)

    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--hf-token", default=None,
                     help="Hugging Face access token, needed for gated models like "
                          "meta-llama/Llama-2-7b-hf. Falls back to the HF_TOKEN or "
                          "HUGGING_FACE_HUB_TOKEN env var if not passed explicitly.")
    ap.add_argument("--no-resume", action="store_true",
                     help="Re-run every checkpoint even if a result JSON already exists.")
    ap.add_argument("--checkpoint-glob", default="checkpoint-*",
                     help="Glob pattern (relative to --checkpoints-dir) for checkpoint folders.")
    ap.add_argument("--include-final", action="store_true",
                     help="Also evaluate a 'final' folder (run_sft.py's trainer.save_model(final_path)) if present.")
    return ap.parse_args()


def resolve_paths(args, cfg):
    """Fill in --checkpoints-dir / --results-dir from config.yaml if not passed explicitly."""
    checkpoints_dir = args.checkpoints_dir
    results_dir = args.results_dir

    if (checkpoints_dir is None or results_dir is None) and cfg:
        seed = int(cfg["run"]["seed"])
        mname = cfg["run"]["model"]
        dname = cfg["run"]["dataset"]
        lr = cfg["training"]["learning_rate"]
        run_tag = f"{mname}_{dname}_seed{seed}_lr{float(lr):.0e}"
        run_root = os.path.join(cfg["run"]["output_root"], run_tag)
        if checkpoints_dir is None:
            checkpoints_dir = os.path.join(run_root, "checkpoints")
        if results_dir is None:
            results_dir = os.path.join(run_root, cfg["standard_eval"]["save_subdir"])

    if checkpoints_dir is None or results_dir is None:
        raise ValueError(
            "Could not resolve --checkpoints-dir / --results-dir. "
            "Pass them explicitly, or pass --config pointing at the config.yaml "
            "used for the SFT run."
        )
    return checkpoints_dir, results_dir


def resolve_eval_settings(args, cfg):
    se = (cfg or {}).get("standard_eval", {})
    tasks = (args.tasks.split(",") if args.tasks
             else list(se.get("tasks", ["mmlu", "arc_challenge", "gsm8k", "squadv2"])))

    # paper rule: drop GSM8k when the SFT dataset was Alpaca
    dname = (cfg or {}).get("run", {}).get("dataset")
    if dname == "alpaca" and "gsm8k" in tasks:
        tasks = [t for t in tasks if t != "gsm8k"]
        print("  [alpaca] dropping GSM8k from eval tasks (per paper).")

    num_fewshot = dict(se.get("num_fewshot", {"gsm8k": 5, "mmlu": 0, "arc_challenge": 0, "squadv2": 0}))
    if args.num_fewshot_gsm8k is not None:
        num_fewshot["gsm8k"] = args.num_fewshot_gsm8k

    limit = args.limit if args.limit is not None else se.get("limit")
    batch_size = args.batch_size or se.get("batch_size", "auto")

    base_model = args.base_model
    if base_model is None and cfg:
        mname = cfg["run"]["model"]
        base_model = cfg["models"][mname]["hf_id"]
    if base_model is None:
        raise ValueError("No base model resolved. Pass --base-model or --config.")

    return {
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "batch_size": batch_size,
        "base_model": base_model,
    }


# =============================================================================
# 1. Checkpoint discovery
# =============================================================================
_STEP_RE = re.compile(r"checkpoint-(\d+)$")


def discover_checkpoints(checkpoints_dir, pattern, include_final):
    found = []
    for p in glob.glob(os.path.join(checkpoints_dir, pattern)):
        if not os.path.isdir(p):
            continue
        m = _STEP_RE.search(os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])

    if include_final:
        final_path = os.path.join(checkpoints_dir, "final")
        if os.path.isdir(final_path):
            # give 'final' a step number one past the last numbered checkpoint
            # so it plots naturally to the right on a step-indexed x-axis
            next_step = (found[-1][0] + 1) if found else 0
            found.append((next_step, final_path))

    if not found:
        raise FileNotFoundError(
            f"No checkpoints matching '{pattern}' (or 'final') found under {checkpoints_dir}"
        )
    return found


def already_done(results_dir, step):
    return os.path.exists(os.path.join(results_dir, f"id_eval_step{step:06d}.json"))


# =============================================================================
# 2. Metric extraction (mirrors run_sft.py's StandardEvalCallback)
# =============================================================================
def extract_id_accuracy(results):
    """Average the primary accuracy-like metric across tasks, matching the
    paper's 'average performance across four standard benchmarks'."""
    prefer = ["acc,none", "acc_norm,none", "exact_match,none",
              "exact_match,flexible-extract", "f1,none", "acc", "exact_match"]
    vals = []
    for _, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for k in prefer:
            if k in metrics and isinstance(metrics[k], (int, float)):
                vals.append(float(metrics[k]))
                break
    return round(sum(vals) / len(vals), 5) if vals else None


def extract_per_task_primary(results):
    prefer = ["acc,none", "acc_norm,none", "exact_match,none",
              "exact_match,flexible-extract", "f1,none", "acc", "exact_match"]
    out = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for k in prefer:
            if k in metrics and isinstance(metrics[k], (int, float)):
                out[task] = round(float(metrics[k]), 5)
                break
    return out


# =============================================================================
# 3. Eval loop
# =============================================================================
def run_eval(step, ckpt_path, base_model, tokenizer, eval_cfg):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"\n=== step {step}  ({ckpt_path}) ===")
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    results = {}
    try:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer,
                    batch_size=eval_cfg["batch_size"])
        for task in eval_cfg["tasks"]:
            nfs = eval_cfg["num_fewshot"].get(task, 0)
            with torch.no_grad():
                r = lm_eval.simple_evaluate(
                    model=hflm, tasks=[task], num_fewshot=nfs,
                    limit=eval_cfg["limit"], bootstrap_iters=0,
                    verbosity="ERROR",
                )
            results[task] = r["results"].get(task, {})
            print(f"    {task}: {extract_per_task_primary({task: results[task]})}")
    finally:
        # detach adapter so the NEXT checkpoint can reuse the same base model
        # weights without a full reload from disk (big time saver over 16 ckpts)
        model = model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def write_result(results_dir, run_tag, step, ckpt_path, results):
    per_task = extract_per_task_primary(results)
    id_acc = extract_id_accuracy(results)
    out = {
        "run_tag": run_tag,
        "step": step,
        "checkpoint_path": ckpt_path,
        "id_accuracy_mean": id_acc,
        "per_task": per_task,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    json_path = os.path.join(results_dir, f"id_eval_step{step:06d}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    -> {json_path}   id_accuracy_mean={id_acc}")

    # rolling CSV for quick plotting
    csv_path = os.path.join(results_dir, "id_eval_all.csv")
    tasks_sorted = sorted(per_task.keys())
    fieldnames = ["step"] + tasks_sorted + ["id_accuracy_mean"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {"step": step, "id_accuracy_mean": id_acc}
        row.update(per_task)
        w.writerow(row)
    return json_path


# =============================================================================
# 4. Main
# =============================================================================
def main():
    args = build_args()
    cfg = load_config(args.config)

    # ---- Hugging Face auth (needed for gated models like Llama-2-7b-hf) ----
    hf_token = (args.hf_token
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        # also export so any nested HF calls (datasets, lm-eval-harness) pick it up
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("Hugging Face auth: logged in with provided token.")
    else:
        print("Hugging Face auth: no token found (HF_TOKEN/--hf-token). "
              "Gated models like meta-llama/Llama-2-7b-hf will fail to load.")

    checkpoints_dir, results_dir = resolve_paths(args, cfg)
    os.makedirs(results_dir, exist_ok=True)
    eval_cfg = resolve_eval_settings(args, cfg)

    if args.benchmark_data_dir:
        os.makedirs(args.benchmark_data_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = args.benchmark_data_dir
        os.environ["HF_HOME"] = args.benchmark_data_dir
        print(f"Benchmark dataset cache -> {args.benchmark_data_dir}")

    run_tag = (cfg or {}).get("run", {}).get("model", "run")
    if cfg:
        seed = int(cfg["run"]["seed"])
        run_tag = f"{cfg['run']['model']}_{cfg['run']['dataset']}_seed{seed}"

    print(f"Checkpoints dir : {checkpoints_dir}")
    print(f"Results dir     : {results_dir}")
    print(f"Base model      : {eval_cfg['base_model']}")
    print(f"Tasks           : {eval_cfg['tasks']}")
    print(f"Limit per task  : {eval_cfg['limit']}")
    print(f"Batch size      : {eval_cfg['batch_size']}")

    checkpoints = discover_checkpoints(checkpoints_dir, args.checkpoint_glob, args.include_final)
    print(f"\nFound {len(checkpoints)} checkpoint(s): "
          f"{[s for s, _ in checkpoints]}")

    # ---- load base model + tokenizer ONCE, reuse across all checkpoints ----
    dtype = getattr(torch, args.dtype)
    print(f"\nLoading base model in {dtype} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        eval_cfg["base_model"], use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        eval_cfg["base_model"], torch_dtype=dtype, device_map="auto", token=hf_token)

    for step, ckpt_path in checkpoints:
        if not args.no_resume and already_done(results_dir, step):
            print(f"Skipping step {step} (result already exists)")
            continue
        try:
            results = run_eval(step, ckpt_path, base_model, tokenizer, eval_cfg)
            write_result(results_dir, run_tag, step, ckpt_path, results)
        except Exception as e:
            print(f"!! Failed on step {step} ({ckpt_path}): {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print(f"\nDone. Per-checkpoint JSON + rolling CSV in: {results_dir}")


if __name__ == "__main__":
    main()