import os
import gc
import glob
import json
import shutil
import argparse
import subprocess
from datetime import datetime, timezone

import yaml
import torch
try:
    # Works around a peft bug: peft checks `torch.__version__ >= 2.5.0` to decide
    # whether DTensor support is available, then references
    # `torch.distributed.tensor.DTensor` directly -- without ever importing that
    # submodule itself. It normally gets pulled in as a side effect of some other
    # import in bigger pipelines; here we force it explicitly so `get_peft_model`
    # doesn't raise `AttributeError: module 'torch.distributed' has no attribute
    # 'tensor'`. Wrapped defensively in case a build lacks distributed entirely.
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model



def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # apply dotted --set overrides, e.g. run.dataset=alpaca
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        # best-effort typing
        for caster in (int, float):
            try:
                val = caster(val); break
            except ValueError:
                pass
        if isinstance(val, str) and val.lower() in ("true", "false"):
            val = val.lower() == "true"
        if isinstance(val, str) and val.lower() in ("null", "none"):
            val = None
        node[parts[-1]] = val
    return cfg


# LoRA target modules per architecture family. OpenInstruct applies LoRA to all
# linear layers; these are the standard names for each family.
TARGET_MODULES = {
    "llama":    ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
    "mistral":  ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
    "gpt_neox": ["query_key_value", "dense",
                 "dense_h_to_4h", "dense_4h_to_h"],
}



ALPACA_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
)


def to_messages(example, fmt, alpaca_use_template):
    """Normalize a raw dataset row into {'messages': [{'role','content'}, ...]}."""
    if fmt == "messages":
        return {"messages": example["messages"]}
    if fmt == "alpaca":
        instr = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()
        if alpaca_use_template:
            user = (ALPACA_PROMPT_INPUT.format(instruction=instr, input=inp)
                    if inp else ALPACA_PROMPT_NO_INPUT.format(instruction=instr))
        else:
            user = instr + (("\n\n" + inp) if inp else "")
        return {"messages": [{"role": "user", "content": user},
                             {"role": "assistant", "content": out}]}
    raise ValueError(f"Unknown dataset format: {fmt}")


def _concat_messages(messages, tokenizer):
    """OpenInstruct 'tulu' chat format. Base models have no chat template, so we
    impose this uniform format (exactly what OpenInstruct did for all datasets)."""
    text = ""
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "system":
            text += "<|system|>\n" + content + "\n"
        elif role == "user":
            text += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            text += "<|assistant|>\n" + content + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"Unknown role: {role}")
    return text


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """Verbatim port of OpenInstruct's masking: tokenize the full formatted
    example, then set labels=-100 for every non-assistant span (including each
    '<|assistant|>\\n' marker), so loss is computed only on assistant content."""
    messages = example["messages"]
    if not messages:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    example_text = _concat_messages(messages, tokenizer).strip()
    tok = tokenizer(example_text, return_tensors="pt",
                    max_length=max_seq_length, truncation=True)
    input_ids = tok.input_ids
    labels = input_ids.clone()

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            continue
        if i == 0:
            start = 0
        else:
            start = tokenizer(_concat_messages(messages[:i], tokenizer),
                              return_tensors="pt", max_length=max_seq_length,
                              truncation=True).input_ids.shape[1]
        if i < len(messages) - 1 and messages[i + 1]["role"] == "assistant":
            so_far = _concat_messages(messages[:i + 1], tokenizer) + "<|assistant|>\n"
        else:
            so_far = _concat_messages(messages[:i + 1], tokenizer)
        end = tokenizer(so_far, return_tensors="pt",
                        max_length=max_seq_length, truncation=True).input_ids.shape[1]
        labels[:, start:end] = -100
        if end >= max_seq_length:
            break

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": torch.ones_like(input_ids).flatten(),
    }


def build_datasets(cfg, tokenizer):
    dcfg = cfg["data"]
    dname = cfg["run"]["dataset"]
    spec = cfg["datasets"][dname]
    print(f"Loading dataset '{dname}' = {spec['hf_id']} [{spec['split']}] ...")
    raw = load_dataset(spec["hf_id"], split=spec["split"])
    if dcfg["max_train_samples"]:
        raw = raw.select(range(min(dcfg["max_train_samples"], len(raw))))
    print(f"  {len(raw):,} raw examples")

    cols = raw.column_names
    raw = raw.map(lambda ex: to_messages(ex, spec["format"], dcfg["alpaca_use_template"]),
                  remove_columns=cols, desc="to_messages")

    max_len = dcfg["max_seq_length"]
    tokenized = raw.map(
        lambda ex: encode_with_messages_format(ex, tokenizer, max_len),
        remove_columns=["messages"], desc="tokenize+mask",
        num_proc=max(1, dcfg["num_workers"]),
    )
    # drop empties and examples with no supervised (assistant) tokens
    tokenized = tokenized.filter(
        lambda ex: len(ex["input_ids"]) > 0 and any(l != -100 for l in ex["labels"]),
        desc="filter",
    )
    tokenized.set_format(type="torch",
                         columns=["input_ids", "labels", "attention_mask"])

    split = tokenized.train_test_split(test_size=dcfg["eval_split_ratio"],
                                       seed=cfg["run"]["seed"])
    print(f"  train: {len(split['train']):,} | eval: {len(split['test']):,}")
    return split["train"], split["test"]



FULL_STATE_FILES = ["optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "rng_state_0.pth", "scaler.pt", "trainer_state.json"]


class CheckpointCallback(TrainerCallback):
    def __init__(self, ckpt_cfg, gdrive_cfg, output_dir, run_tag, run_root):
        self.mode = ckpt_cfg["mode"]
        self.save_steps = int(ckpt_cfg["save_steps"])
        self.keep_n = int(ckpt_cfg["keep_full_state_n"])
        self.de = ckpt_cfg["dense_early"]
        self.output_dir = output_dir      # .../run_tag/checkpoints  (listing/trimming)
        self.run_root = run_root          # .../run_tag              (what gets synced)
        self.run_tag = run_tag
        self.gd = gdrive_cfg
        self._boundary = self._dense_int = self._sparse_int = None

    def _init_dense_schedule(self, max_steps):
        self._boundary = max(1, int(max_steps * self.de["dense_window_pct"]))
        self._dense_int = max(10, self._boundary // max(1, self.de["target_dense_checkpoints"]))
        rem = max(1, max_steps - self._boundary)
        self._sparse_int = max(50, rem // max(1, self.de["target_sparse_checkpoints"]))
        print(f"  dense-early schedule: steps 1-{self._boundary} every {self._dense_int}, "
              f"then every {self._sparse_int} (total steps {max_steps})")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0:
            return control
        if self.mode == "fixed":
            if step % self.save_steps == 0:
                control.should_save = True
        else:  # dense_early
            if self._boundary is None and state.max_steps > 0:
                self._init_dense_schedule(state.max_steps)
            if self._boundary is not None:
                dense = step <= self._boundary and step % self._dense_int == 0
                sparse = step > self._boundary and step % self._sparse_int == 0
                if dense or sparse:
                    control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        control.should_save = True  # always keep the final checkpoint
        return control

    def on_save(self, args, state, control, **kwargs):
        self._trim_old_checkpoints()
        self._sync_to_drive()
        return control

    def _trim_old_checkpoints(self):
        ckpts = sorted(glob.glob(f"{self.output_dir}/checkpoint-*"),
                       key=lambda p: int(p.split("-")[-1]))
        to_trim = ckpts[:-self.keep_n] if self.keep_n > 0 else ckpts
        removed = 0
        for ck in to_trim:
            for fn in FULL_STATE_FILES:
                fp = os.path.join(ck, fn)
                if os.path.exists(fp):
                    os.remove(fp); removed += 1
            # also drop any optimizer shards (deepspeed / sharded)
            for extra in glob.glob(f"{ck}/global_step*") + glob.glob(f"{ck}/rng_state_*.pth"):
                if os.path.isdir(extra):
                    shutil.rmtree(extra, ignore_errors=True); removed += 1
                elif os.path.exists(extra):
                    os.remove(extra); removed += 1
        if removed:
            print(f"    [trim] removed {removed} full-state file(s) from "
                  f"{len(to_trim)} older checkpoint(s); kept optimizer for last {self.keep_n}")

    def _sync_to_drive(self):
        if not (self.gd and self.gd.get("enabled") and self.gd.get("sync_after_save")):
            return
        remote = self.gd["rclone_remote"]
        dest = f"{remote}:{self.gd['remote_dir'].rstrip('/')}/{self.run_tag}"
        # Sync the whole run_root (checkpoints/ + standard_eval/) every time, so the
        # Drive layout is identical to the local layout at every point -- not just at
        # the end. `sync` mirrors deletions too, so trims propagate to Drive.
        cmd = ["rclone", "sync", self.run_root, dest, "--fast-list", "--transfers=8"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"    [drive] synced -> {dest}")
        except FileNotFoundError:
            print("    [drive] rclone not found on PATH -- skipping sync. "
                  "Install rclone and run `rclone config`.")
        except subprocess.CalledProcessError as e:
            print(f"    [drive] rclone sync failed: {e.stderr.strip()[:300]}")



class StandardEvalCallback(TrainerCallback):
    def __init__(self, se_cfg, tokenizer, results_dir, run_tag):
        self.cfg = se_cfg
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        self.run_tag = run_tag
        self.every = int(se_cfg["every_steps"])
        os.makedirs(results_dir, exist_ok=True)
        self._lm_eval = None

    def _lazy_import(self):
        if self._lm_eval is None:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            self._lm_eval = (lm_eval, HFLM)
        return self._lm_eval

    def _run(self, model, step):
        try:
            lm_eval, HFLM = self._lazy_import()
        except Exception as e:
            print(f"    [id-eval] lm-eval-harness not importable ({e}); skipping.")
            return
        tasks = list(self.cfg["tasks"])
        was_training = model.training
        model.eval()
        try:
            hflm = HFLM(pretrained=model, tokenizer=self.tokenizer,
                        batch_size=self.cfg.get("batch_size", "auto"))
            # per-task few-shot handled via task-level num_fewshot dict
            results = {}
            for task in tasks:
                nfs = self.cfg.get("num_fewshot", {}).get(task, 0)
                with torch.no_grad():
                    r = lm_eval.simple_evaluate(
                        model=hflm, tasks=[task], num_fewshot=nfs,
                        limit=self.cfg.get("limit"), bootstrap_iters=0,
                        verbosity="ERROR",
                    )
                results[task] = r["results"].get(task, {})
        except Exception as e:
            print(f"    [id-eval] evaluation error: {e}")
            results = {}
        finally:
            if was_training:
                model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # headline "ID accuracy" = mean of each task's primary accuracy metric
        primary = _extract_id_accuracy(results)
        out = {"run_tag": self.run_tag, "step": step,
               "id_accuracy_mean": primary, "results": results,
               "timestamp": datetime.now(timezone.utc).isoformat()}
        path = os.path.join(self.results_dir, f"id_eval_step{step:06d}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"    [id-eval] step {step}: ID accuracy (mean) = "
              f"{primary if primary is not None else 'n/a'}  -> {path}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every == 0:
            self._run(kwargs["model"], state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._run(kwargs["model"], state.global_step)  # final point on the curve
        return control


def _extract_id_accuracy(results):
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
                vals.append(float(metrics[k])); break
    return round(sum(vals) / len(vals), 5) if vals else None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[],
                    help="dotted overrides, e.g. run.dataset=alpaca")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    seed = int(cfg["run"]["seed"])
    set_seed(seed)

    mname = cfg["run"]["model"]
    mspec = cfg["models"][mname]
    model_id = mspec["hf_id"]
    family = mspec["family"]
    dname = cfg["run"]["dataset"]
    run_tag = f"{mname}_{dname}_seed{seed}_lr{cfg['training']['learning_rate']:.0e}"

    # Paper: "We ignore GSM8k performance when finetuning on Alpaca." Do it
    # automatically so the standard-benchmark eval matches the paper per-dataset.
    if dname == "alpaca" and "gsm8k" in cfg["standard_eval"]["tasks"]:
        cfg["standard_eval"]["tasks"] = [t for t in cfg["standard_eval"]["tasks"]
                                         if t != "gsm8k"]
        print("  [alpaca] dropping GSM8k from standard eval (per paper).")

    run_root = os.path.join(cfg["run"]["output_root"], run_tag)
    output_dir = os.path.join(run_root, "checkpoints")
    results_dir = os.path.join(run_root, cfg["standard_eval"]["save_subdir"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    # snapshot the exact resolved config used for this run
    with open(os.path.join(results_dir, "effective_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\n=== RUN {run_tag} ===")
    print(f"  model   : {model_id} ({family})")
    print(f"  dataset : {cfg['datasets'][dname]['hf_id']}")
    print(f"  output  : {output_dir}\n")

    # ---- tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True,
        trust_remote_code=cfg["model_load"]["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- data 
    train_ds, eval_ds = build_datasets(cfg, tokenizer)

    # base model 
    dtype = getattr(torch, cfg["model_load"]["torch_dtype"])
    attn = ("flash_attention_2" if cfg["model_load"]["use_flash_attention_2"]
            else cfg["model_load"]["attn_implementation_fallback"])
    print(f"Loading base model in {dtype} (attn={attn}) ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation=attn,
            trust_remote_code=cfg["model_load"]["trust_remote_code"])
    except Exception as e:
        print(f"  attn '{attn}' failed ({e}); retrying with sdpa.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa",
            trust_remote_code=cfg["model_load"]["trust_remote_code"])

    if cfg["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # LoRA 
    lcfg = cfg["lora"]
    targets = (TARGET_MODULES[family] if lcfg["target_modules"] == "auto"
               else lcfg["target_modules"])
    lora_config = LoraConfig(
        r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]), bias=lcfg["bias"],
        target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # collator (pads input_ids and pads labels with -100) 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", label_pad_token_id=-100)

    # training args 
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        max_grad_norm=t["max_grad_norm"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        group_by_length=t["group_by_length"],
        logging_steps=t["logging_steps"],
        save_strategy="no",          # our callback controls ALL saves
        eval_strategy="no",          # standard-benchmark eval is a callback
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=cfg["data"]["num_workers"],
    )
    print(f"Effective batch size: "
          f"{t['per_device_train_batch_size'] * t['gradient_accumulation_steps']}")

    # callbacks 
    callbacks = [CheckpointCallback(cfg["checkpointing"], cfg["google_drive"],
                                    output_dir, run_tag, run_root)]
    if cfg["standard_eval"]["enabled"]:
        callbacks.append(StandardEvalCallback(cfg["standard_eval"], tokenizer,
                                              results_dir, run_tag))

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tokenizer, data_collator=data_collator,
        callbacks=callbacks,
    )

    # resume detection 
    resume = cfg["checkpointing"]["resume"]
    resume_from = None
    if resume == "auto":
        cks = sorted(glob.glob(f"{output_dir}/checkpoint-*"),
                     key=lambda p: int(p.split("-")[-1]))
        resumable = [c for c in cks if os.path.exists(f"{c}/trainer_state.json")
                     and os.path.exists(f"{c}/optimizer.pt")]
        resume_from = resumable[-1] if resumable else None
        print(f"Resume: {'from ' + resume_from if resume_from else 'fresh start'}")
    elif resume not in (None, "none"):
        resume_from = resume
        print(f"Resume: from {resume_from}")

    # train 
    print("-" * 60)
    result = trainer.train(resume_from_checkpoint=resume_from)
    print("-" * 60)
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

    # save final adapter + manifest 
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    manifest = {
        "run_tag": run_tag, "model_id": model_id, "family": family,
        "dataset": cfg["datasets"][dname]["hf_id"], "seed": seed,
        "lora": {"r": lcfg["r"], "alpha": lcfg["alpha"],
                 "dropout": lcfg["dropout"], "targets": targets},
        "epochs": t["num_train_epochs"],
        "effective_batch_size": t["per_device_train_batch_size"] * t["gradient_accumulation_steps"],
        "learning_rate": t["learning_rate"],
        "lr_scheduler": t["lr_scheduler_type"], "warmup_ratio": t["warmup_ratio"],
        "max_seq_length": cfg["data"]["max_seq_length"],
        "checkpoint_dir": output_dir,
        "train_metrics": result.metrics,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(results_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # full per-step loss/lr history (for plotting the training curve later)
    with open(os.path.join(results_dir, "train_log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\nFinal adapter -> {final_path}")
    print(f"Manifest      -> {os.path.join(results_dir, 'run_manifest.json')}")

    # final Drive sync of everything (same local root / same remote dest as every
    # per-checkpoint sync above -- just a final, complete pass)
    gd = cfg["google_drive"]
    if gd.get("enabled") and gd.get("sync_after_save"):
        dest = f"{gd['rclone_remote']}:{gd['remote_dir'].rstrip('/')}/{run_tag}"
        try:
            subprocess.run(["rclone", "sync", run_root, dest, "--fast-list"],
                           check=True, capture_output=True, text=True)
            print(f"Drive         -> {dest}")
        except Exception as e:
            print(f"Final Drive sync skipped/failed: {e}")


if __name__ == "__main__":
    main()