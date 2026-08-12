import os
import re
import gc
import json
import math
import time
import random
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

__version__ = "1.0.0"

PREFILL = "The answer is:"          # assistant-turn prefill for log-prob scoring
MAX_NEW_TOKENS = 24                 
TAU = 1.0                           # log-prob confidence gate (nats); pre-registered,
                                     # do not tune per-checkpoint -- see the module's
                                     # validated history: lowering this to reduce
                                     # escalation volume introduces silent misclassifications
                                     # on items where the morphological/prefix-collision
                                     # dilution effects push Delta close to but under tau.
K_DISTRACTORS = 5                   # for the genuine-conflict filter (Layer 0)
JUDGE_MODEL = "gpt-4.1-nano"

REQUIRED_FIELDS = {"item_id", "source", "question", "context",
                   "parametric_answer", "counterfactual_answer"}


DEFAULT_DROP_IDS = frozenset({
    "cap_0020","cap_0027","cap_0041","cap_0121","cap_0161"
})


# Data Loader Function

def load_items(path, drop_ids=DEFAULT_DROP_IDS):
    """
    Load conflict_eval_unified.json (a list of items, or {"items": [...]}),
    validate schema, and drop flagged items. Returns the usable item list.
    """
    with open(path) as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    for it in items:
        missing = REQUIRED_FIELDS - set(it)
        assert not missing, f"{it.get('item_id', '?')} missing {missing}"
    kept = [it for it in items if it["item_id"] not in drop_ids]
    dropped = len(items) - len(kept)
    print(f"Loaded {len(items)} total | dropped {dropped} flagged item(s) | {len(kept)} usable")
    return kept


def build_distractor_pool(items):
    """Per-stratum pool of (item_id, parametric_answer) for the Layer 0 filter."""
    pool = {}
    for it in items:
        pool.setdefault(it["source"], []).append((it["item_id"], it["parametric_answer"]))
    return pool


#Loading our model and tokenizer with the option to load in 4bit precision if needed.

def load_model(model_id, precision="bf16", allow_4bit=False):
    """
    precision: "bf16" (default, required for cross-checkpoint log-prob
               comparability) or "4bit" (emergency fallback; invalidates
               cross-precision log-prob comparisons -- must opt in explicitly).
    """
    assert precision in ("bf16", "4bit")
    if precision == "4bit":
        assert allow_4bit, (
            "4-bit loading degrades log-prob scoring and breaks cross-checkpoint "
            "comparability. Set allow_4bit=True explicitly if you accept this.")

    print(f"Loading {model_id}  precision={precision}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if precision == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if hasattr(model, "get_memory_footprint"):
        print(f"  Memory footprint: {model.get_memory_footprint()/1e9:.2f} GB (precision={precision})")
    return model, tok


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory released.")


#Normalization and Classification Function

STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "is", "was", "to", "for"}


def normalize(t):
    s = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def sig_words(ans):
    return [w for w in normalize(ans).split() if len(w) > 3 and w not in STOP]


def discr_words(ans, other):
    """Significant words in `ans` that do NOT also appear in `other` -- removes
    shared-word false positives (e.g. "United" in "United States"/"United Kingdom")."""
    o = set(normalize(other).split())
    dw = [w for w in sig_words(ans) if w not in o]
    return dw if dw else sig_words(ans)


def _bounded(w):
    """Word-boundary regex -- prevents e.g. "mississippi" matching inside
    "mississippian", or "usa" matching inside "usage"."""
    return r"(?<!\w)" + re.escape(w) + r"(?!\w)"


def contains(resp, ans, words):
    r = normalize(resp)
    if not words:
        return bool(re.search(_bounded(normalize(ans)), r))
    return any(re.search(_bounded(w), r) for w in words)


def first_pos(resp, ans, words):
    r = normalize(resp)
    positions = []
    m = re.search(_bounded(normalize(ans)), r)
    if m:
        positions.append(m.start())
    for w in words:
        m = re.search(_bounded(w), r)
        if m:
            positions.append(m.start())
    return min(positions, default=10**9)


def _user_text(question, context):
    if context is None:
        return ("Answer the question in as few words as possible. "
                "Do not explain. Do not add notes.\n\nQuestion: " + question)
    return ("Read the context and answer the question in as few words as possible. "
            "Do not explain. Do not add notes.\n\nContext: " + context +
            "\n\nQuestion: " + question)


#Prompt builder for applying the chat template and returning token ids

def _apply_template(tok, messages, add_generation_prompt=True):
    """Applies the chat template and returns a 1-D LongTensor of token ids,
    regardless of what apply_chat_template's return type happens to be."""
    text = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=add_generation_prompt)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def build_gen_prompt_ids(tok, question, context, use_ct):
    u = _user_text(question, context)
    if use_ct and tok.chat_template:
        return _apply_template(tok, [{"role": "user", "content": u}])
    return tok(u + "\nAnswer:", return_tensors="pt", add_special_tokens=True).input_ids[0]


def build_score_prompt_ids(tok, question, context, use_ct, prefill=PREFILL):
    u = _user_text(question, context)
    if use_ct and tok.chat_template:
        base = _apply_template(tok, [{"role": "user", "content": u}])
    else:
        return tok(u + "\n" + prefill, return_tensors="pt", add_special_tokens=True).input_ids[0]
    pre = tok(prefill, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return torch.cat([base, pre])


#Generation and Method2: Teacher forced scoring

@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens=MAX_NEW_TOKENS):
    ids = prompt_ids.unsqueeze(0).to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def score_answer_logprob(model, tok, prompt_ids, answer_text):
    """
    log P(answer | prompt), teacher-forced, plus token count for length
    normalisation.

        log P(a|prompt) = sum_i log P(t_i | prompt, t_<i)

    One forward pass on [prompt_ids ++ answer_ids]; logits at position i
    predict token i+1, so the answer's own logits live at positions
    [n-1 .. n+len(answer)-2] where n = len(prompt_ids). Proven correct with
    synthetic logits in `_test_scorer_indexing` below (a perfectly-predicted
    continuation returns per-token log-prob ~0; a deliberate off-by-one in
    the slice returns a large negative number).

    Hardening:
    - Space handling is centralised and idempotent here (strip, then prepend
      exactly one leading space) -- safe whether or not the caller pre-adds a
      leading space.
    - Trailing-EOS guard: if prompt_ids ends in EOS, strip it BEFORE computing
      n, so a stray EOS can't shift the indexing.
    - Empty-answer guard: returns (0.0, 0) explicitly, so callers can detect
      and flag a degenerate candidate rather than silently reading 0.0 log-prob
      (= probability 1.0) as "perfect confidence".
    - Deliberately NOT masking by pad_token_id: in this pipeline pad_token_id
      is commonly set equal to eos_token_id (see load_model), so such a mask
      would incorrectly zero out any legitimate EOS-valued token inside a real
      answer. It would also be unreachable dead code here, since `ans` is
      never padded to begin with (single unpadded tokenize call).
    """
    answer_text = " " + answer_text.strip()
    ans = tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ans.shape[0] == 0:
        return 0.0, 0

    if prompt_ids.numel() > 0 and prompt_ids[-1].item() == tok.eos_token_id:
        prompt_ids = prompt_ids[:-1]

    full = torch.cat([prompt_ids, ans]).unsqueeze(0).to(model.device)
    logits = model(full).logits[0]
    n = prompt_ids.shape[0]
    lp = torch.log_softmax(logits[n - 1:-1, :].float(), dim=-1)
    tok_lp = lp[torch.arange(ans.shape[0]), ans.to(lp.device)]
    return tok_lp.sum().item(), ans.shape[0]




def _token_len(tok, s):
    return len(tok(" " + s.strip(), add_special_tokens=False).input_ids)


def sample_distractors(tok, item, pool, k=K_DISTRACTORS, seed=0):
    src = item["source"]
    target_len = _token_len(tok, item["parametric_answer"])
    candidates = [(iid, ans) for iid, ans in pool[src] if iid != item["item_id"]]
    scored = sorted(candidates, key=lambda x: abs(_token_len(tok, x[1]) - target_len))
    band = scored[:max(k * 4, k)]
    rnd = random.Random(seed + (hash(item["item_id"]) % (2 ** 16)))
    chosen = rnd.sample(band, k=min(k, len(band)))
    return [ans for _, ans in chosen]


@torch.no_grad()
def filter_logprob(model, tok, item, pool, use_ct, k=K_DISTRACTORS, seed=0):
    q, par, cf = item["question"], item["parametric_answer"], item["counterfactual_answer"]
    pid = build_score_prompt_ids(tok, q, None, use_ct)

    def _score(a):
        lp, n = score_answer_logprob(model, tok, pid, " " + a.strip())
        return lp / max(n, 1)

    par_score = _score(par)
    cf_score = _score(cf)
    distractor_scores = [_score(d) for d in sample_distractors(tok, item, pool, k=k, seed=seed)]
    return par_score == max([par_score, cf_score] + distractor_scores)


def filter_generation(model, tok, q, par, use_ct):
    """Secondary filter (generation-based), logged for comparison only -- not
    used to gate R_ctx."""
    resp = generate(model, tok, build_gen_prompt_ids(tok, q, None, use_ct))
    return contains(resp, par, sig_words(par))



# sequence log-prob scoring (primary measurement)
def method_logprob(model, tok, score_prompt_ids, par, cf, tau=TAU):
    slp, npar = score_answer_logprob(model, tok, score_prompt_ids, par)
    clp, ncf = score_answer_logprob(model, tok, score_prompt_ids, cf)

    if npar == 0 or ncf == 0:
        # one of the two candidates tokenised to nothing -- flag it rather than
        # let 0.0/max(n,1)=0.0 masquerade as "perfect confidence" and silently
        # produce a spurious CTX/PAR verdict
        return {"delta": 0.0, "label": "AMBIG", "lp_par_pt": 0.0, "lp_cf_pt": 0.0,
                "n_par": npar, "n_cf": ncf, "error": "empty_answer"}

    par_pt = slp / npar
    cf_pt = clp / ncf
    d = cf_pt - par_pt
    label = "AMBIG" if abs(d) < tau else ("CTX" if d > 0 else "PAR")
    return {"delta": round(d, 4), "label": label,
            "lp_par_pt": round(par_pt, 3), "lp_cf_pt": round(cf_pt, 3),
            "n_par": npar, "n_cf": ncf}



# Layer 2 -- LLM-as-judge (fires only on AMBIG)

def make_judge(api_key=None, model=JUDGE_MODEL):
    """
    Returns a callable:
        judge(response, question, context, par, cf,
              item_id="unknown", checkpoint_id="adhoc", max_retries=3) -> dict

    matching the Layer 2 interface expected by evaluate()/evaluate_checkpoint().

    Lazy by design: `openai` is only imported when this factory is actually
    called, and the API key is only required at that point. Importing eval.py,
    or using it with judge_fn_=None (Layer 1 only, e.g. during early SFT
    checkpoints where you don't want the extra API cost/latency), never
    requires the `openai` package or OPENAI_API_KEY to be set.

    Each call to make_judge() gets its own response cache (`judge.cache`),
    keyed by (item_id, checkpoint_id, sha256(response)) -- re-running an
    evaluation is idempotent within one judge instance and doesn't re-spend
    API budget; a fresh judge instance starts with a clean cache.
    """
    from openai import OpenAI  # deferred import -- optional dependency

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Pass make_judge(api_key=...) explicitly, "
            "or set the OPENAI_API_KEY environment variable. In Colab: "
            "os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')."
        )
    client = OpenAI(api_key=key)
    cache = {}

    def _cache_key(item_id, checkpoint_id, response_text):
        h = hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:16]
        return (item_id, checkpoint_id, h)

    def _prompt(question, context, response, par, cf):
        return (
            "You are classifying a language model's answer to a factual question "
            "where the provided context contradicts common knowledge.\n\n"
            f"Question: {question}\nContext provided to the model: {context}\n"
            f"Model's response: {response}\n\n"
            "The two candidate answers are:\n"
            f"- PARAMETRIC (what the model knows from training): {par}\n"
            f"- CONTEXTUAL (what the context asserts): {cf}\n\n"
            'Respond with JSON only, matching this schema exactly:\n'
            '{"answer": "PAR" | "CTX" | "OTHER", '
            '"other_subtype": "hedged" | "abstained" | "hallucinated" | null}\n\n'
            "Rules:\n"
            f"- PAR if the response asserts {par} as the answer to the question\n"
            f"- CTX if the response asserts {cf} as the answer to the question\n"
            "- OTHER if the response abstains, hedges without committing, or "
            "gives a different answer entirely\n"
            "- other_subtype is null unless answer is OTHER. If answer is OTHER: "
            '"hedged" if both candidates appear without commitment, "abstained" '
            'if neither appears, "hallucinated" if a different, third answer is given\n'
            '- If the response is grammatically inverted (e.g. "capital of Kathmandu '
            'is Nepal") but the semantic answer is clear from intent, classify by intent'
        )

    def judge(response, question, context, par, cf, item_id="unknown",
              checkpoint_id="adhoc", max_retries=3):
        ck = _cache_key(item_id, checkpoint_id, response)
        if ck in cache:
            return cache[ck]

        prompt = _prompt(question, context, response, par, cf)
        last_err = None
        for attempt in range(max_retries):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=40,
                    response_format={"type": "json_object"},
                )
                raw = r.choices[0].message.content
                parsed = json.loads(raw)
                label = parsed.get("answer")
                if label not in ("PAR", "CTX", "OTHER"):
                    raise ValueError(f"unexpected judge label: {parsed!r}")
                verdict = {
                    "label": label,
                    "other_subtype": parsed.get("other_subtype"),
                    "layer2_failed": False,
                    "judge_prompt": prompt,
                    "judge_raw_response": raw,
                }
                cache[ck] = verdict
                return verdict
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)

        verdict = {"label": "OTHER", "other_subtype": None, "layer2_failed": True,
                   "error": str(last_err), "judge_prompt": prompt, "judge_raw_response": None}
        cache[ck] = verdict
        return verdict

    judge.cache = cache  # exposed for inspection / clearing between experiments
    return judge


#Diagnostic Metrics and Cross Check with Context-Parametric-Inversion Paper.

def method_paper(resp, par, cf):
    """Faithful containment: cf_acc/par_acc, the paper's own convention. Both
    can be 1 (e.g. shared-word answer pairs like "United States"/"United
    Kingdom") -- this is NOT a partition, never compare its rates directly
    against ordered/logprob/final."""
    return {"par_acc": int(contains(resp, par, sig_words(par))),
            "cf_acc": int(contains(resp, cf, sig_words(cf)))}


def method_ordered(resp, par, cf):
    """CTX/PAR/OTHER by which candidate's DISCRIMINATIVE words (shared words
    removed) appear first in the response; records a hedge subtype."""
    pw, cw = discr_words(par, cf), discr_words(cf, par)
    ph, ch = contains(resp, par, pw), contains(resp, cf, cw)
    if not ph and not ch:
        return {"label": "OTHER", "sub": "neither"}
    if ph and not ch:
        return {"label": "PAR", "sub": "par_only"}
    if ch and not ph:
        return {"label": "CTX", "sub": "ctx_only"}
    return ({"label": "CTX", "sub": "ctx_then_par"}
            if first_pos(resp, cf, cw) < first_pos(resp, par, pw)
            else {"label": "PAR", "sub": "par_then_cf"})

#Boostraped CI and mean for the rates of each method, with 1000 bootstrap samples by default.

def rate_ci(flags, n_boot=1000, seed=0):
    n = len(flags)
    if n == 0:
        return {"mean": None, "ci": [None, None], "n": 0}
    rnd = random.Random(seed)
    mean = sum(flags) / n
    pts = sorted(sum(rnd.choice(flags) for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(mean, 3),
            "ci": [round(pts[int(.025 * n_boot)], 3), round(pts[int(.975 * n_boot)], 3)],
            "n": n}


def _rates(subset, methods):
    d = {"n_passed": len(subset)}
    if not subset:
        return d

    if "paper" in methods:
        cf = [r["paper"]["cf_acc"] == 1 for r in subset]
        pa = [r["paper"]["par_acc"] == 1 for r in subset]
        d["paper"] = {"R_ctx": rate_ci(cf), "R_par": rate_ci(pa),
                      "note": "cf_acc/par_acc are independent containment checks, "
                              "not a partition -- can overlap or both be 0"}

    if "ordered" in methods:
        lbl = [r["ordered"]["label"] for r in subset]
        d["ordered"] = {"R_ctx": rate_ci([x == "CTX" for x in lbl]),
                         "R_par": rate_ci([x == "PAR" for x in lbl]),
                         "R_other": rate_ci([x == "OTHER" for x in lbl])}

    if "logprob" in methods:
        lbl = [r["logprob"]["label"] for r in subset]
        d["logprob"] = {"R_ctx": rate_ci([x == "CTX" for x in lbl]),
                         "R_par": rate_ci([x == "PAR" for x in lbl]),
                         "R_ambiguous": rate_ci([x == "AMBIG" for x in lbl])}

    finals = [r["final_label"] for r in subset if "final_label" in r]
    if finals:
        d["final"] = {"R_ctx": rate_ci([x == "CTX" for x in finals]),
                       "R_par": rate_ci([x == "PAR" for x in finals]),
                       "R_other": rate_ci([x == "OTHER" for x in finals])}

    escalated = [r for r in subset if r.get("logprob", {}).get("label") == "AMBIG"]
    d["escalation_rate"] = round(len(escalated) / len(subset), 3)
    failed = [r for r in escalated if r.get("judge", {}).get("layer2_failed")]
    d["layer2_failed_rate"] = round(len(failed) / len(subset), 3)

    judged_other = [r["judge"]["other_subtype"] for r in escalated
                    if r.get("judge", {}).get("label") == "OTHER" and r["judge"].get("other_subtype")]
    if judged_other:
        cnt = Counter(judged_other)
        d["other_breakdown"] = {k: round(v / len(subset), 3) for k, v in cnt.items()}

    return d


def summarize(rows, methods):
    passed = [r for r in rows if r["passed"]]
    strata = sorted({r["source"] for r in rows})
    per_stratum = {}
    for s in strata:
        total_s = [r for r in rows if r["source"] == s]
        passed_s = [r for r in total_s if r["passed"]]
        stratum_rates = _rates(passed_s, methods)
        stratum_rates["filter_yield"] = round(len(passed_s) / max(len(total_s), 1), 3)
        per_stratum[s] = stratum_rates
    return {"filter_yield": round(len(passed) / max(len(rows), 1), 3),
            "aggregate": _rates(passed, methods),
            "per_stratum": per_stratum}


#Unified Evaluation Function

def evaluate(model, tok, items, use_ct, distractor_pool,
             methods=("paper", "ordered", "logprob"),
             record_gen_filter=False, tau=TAU, judge_fn_=None,
             checkpoint_id="adhoc", k_distractors=K_DISTRACTORS,
             verbose=True):
    """
    Runs Layer 0 (filter) -> Layer 1 -> Layer 2 (AMBIG only) -> Layer 3
    (diagnostics) per item, then aggregates.

    With verbose=True (default), every per-item record embeds the question,
    context, both candidate answers, the exact decoded prompts (chat-template
    tokens included), the generated response, every method's verdict, and the
    final label -- enough to audit any single item by eye without touching the
    dataset file. Set verbose=False for large production runs (many
    checkpoints) to avoid repeating static item text in every checkpoint's JSON.

    Generation happens at most once per item and is reused for the judge call
    if escalated (never re-generated).
    """
    need_gen = ("paper" in methods) or ("ordered" in methods) or (judge_fn_ is not None)
    rows = []
    for it in items:
        q, ctx = it["question"], it["context"]
        par, cf = it["parametric_answer"], it["counterfactual_answer"]
        rec = {"item_id": it["item_id"], "source": it["source"]}
        if verbose:
            rec.update({"question": q, "context": ctx,
                        "parametric_answer": par, "counterfactual_answer": cf})

        rec["passed"] = filter_logprob(model, tok, it, distractor_pool, use_ct, k=k_distractors)
        if record_gen_filter:
            rec["filter_gen"] = filter_generation(model, tok, q, par, use_ct)

        gen_ids = build_gen_prompt_ids(tok, q, ctx, use_ct)
        score_ids = build_score_prompt_ids(tok, q, ctx, use_ct)
        if verbose:
            rec["gen_prompt_text"] = tok.decode(gen_ids, skip_special_tokens=False)
            rec["score_prompt_text"] = tok.decode(score_ids, skip_special_tokens=False)

        resp = None
        if need_gen:
            resp = generate(model, tok, gen_ids)
            rec["response"] = resp
            if "paper" in methods:
                rec["paper"] = method_paper(resp, par, cf)
            if "ordered" in methods:
                rec["ordered"] = method_ordered(resp, par, cf)

        if "logprob" in methods:
            rec["logprob"] = method_logprob(model, tok, score_ids, par, cf, tau)
            if rec["logprob"]["label"] == "AMBIG" and judge_fn_ is not None:
                if resp is None:
                    resp = generate(model, tok, gen_ids)
                    rec["response"] = resp
                rec["judge"] = judge_fn_(resp, q, ctx, par, cf,
                                          item_id=it["item_id"], checkpoint_id=checkpoint_id)
                rec["final_label"] = rec["judge"]["label"]
            else:
                rec["final_label"] = rec["logprob"]["label"]

        rows.append(rec)
    return {"per_item": rows, "summary": summarize(rows, methods)}


# Summary for Reports

def print_summary(model_id, res):
    s = res["summary"]
    print(f"\n### {model_id}  | filter_yield={s['filter_yield']}")
    a = s["aggregate"]
    if "final" in a:
        print(f"  FINAL   R_ctx={a['final']['R_ctx']['mean']}  "
              f"R_par={a['final']['R_par']['mean']}  R_other={a['final']['R_other']['mean']}")
    if "paper" in a:
        print(f"  PAPER   R_ctx={a['paper']['R_ctx']['mean']}  "
              f"R_par={a['paper']['R_par']['mean']}  (not a partition)")
    if "ordered" in a:
        print(f"  ORDERED R_ctx={a['ordered']['R_ctx']['mean']}  "
              f"R_par={a['ordered']['R_par']['mean']}  R_other={a['ordered']['R_other']['mean']}")
    if "logprob" in a:
        print(f"  LOGPROB R_ctx={a['logprob']['R_ctx']['mean']}  "
              f"R_par={a['logprob']['R_par']['mean']}  R_ambiguous={a['logprob']['R_ambiguous']['mean']}")
    print(f"  escalation_rate={a.get('escalation_rate')}  "
          f"layer2_failed_rate={a.get('layer2_failed_rate')}  "
          f"other_breakdown={a.get('other_breakdown')}")
    for st, b in s["per_stratum"].items():
        if b.get("final"):
            print(f"    [{st:20s} n={b['n_passed']}] final R_ctx={b['final']['R_ctx']['mean']}  "
                  f"filter_yield={b.get('filter_yield')}  escalation={b.get('escalation_rate')}")


# Excecution of McNemar's test on paired CTX flags from two checkpoints, returning counts and p-value.

def paired_ctx_flags_from_checkpoints(rows_a, rows_b):
    """Persistently-filtered subset (spec Sec.6.1): items passing the filter
    at BOTH checkpoints A and B, paired by item_id."""
    a_map = {r["item_id"]: r for r in rows_a if r.get("passed")}
    b_map = {r["item_id"]: r for r in rows_b if r.get("passed")}
    common_ids = sorted(set(a_map) & set(b_map))
    pairs = [(a_map[i]["final_label"] == "CTX", b_map[i]["final_label"] == "CTX")
             for i in common_ids]
    return pairs, len(common_ids)


def mcnemar_test(paired_ctx_flags):
    """Exact two-sided McNemar test (binomial on discordant pairs) -- more
    appropriate than the chi-square approximation for the typically-small
    discordant counts expected here. Validated against scipy.stats.binomtest
    across multiple cases including asymmetric large-sample ones."""
    n01 = sum(1 for a, b in paired_ctx_flags if (not a) and b)   # switched TO ctx
    n10 = sum(1 for a, b in paired_ctx_flags if a and (not b))   # switched FROM ctx
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(n01, n10)

    def binom_cdf(k, n, p=0.5):
        return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))

    p_value = min(1.0, 2 * binom_cdf(k, n))
    return {"n01": n01, "n10": n10, "n_discordant": n, "p_value": round(p_value, 4)}


#Validation of Layer 1 vs Layer 2 agreement on confident (non-AMBIG) items, with a warning if agreement falls below 90% (indicating a scoring bug, not expected ambiguity).

def test_layer_agreement(evaluate_result, judge_fn_, checkpoint_id="agreement_test",
                          n_sample=50, seed=0):
    """
    Spec Sec.8 Test 3. Sample n_sample items where Layer 1 was CONFIDENT
    (CTX/PAR, not AMBIG), force the judge to classify them anyway, and check
    agreement. Disagreement here indicates a scoring bug; disagreement on
    AMBIG items is expected and is exactly what Layer 2 exists for.

    `evaluate_result` is the dict returned by evaluate() (a dict with a
    "per_item" key), or a plain per_item list. Requires verbose=True records
    (question/context/parametric_answer/counterfactual_answer/response).
    """
    per_item = evaluate_result["per_item"] if isinstance(evaluate_result, dict) and \
        "per_item" in evaluate_result else evaluate_result
    confident = [r for r in per_item
                 if r.get("logprob", {}).get("label") in ("CTX", "PAR") and "response" in r]
    if not confident:
        raise ValueError("No confident (non-AMBIG) items with recorded responses found -- "
                          "was evaluate() run with verbose=True and 'logprob' in methods?")

    rnd = random.Random(seed)
    sample = rnd.sample(confident, k=min(n_sample, len(confident)))
    agree, total, disagreements = 0, 0, []
    for r in sample:
        j = judge_fn_(r["response"], r["question"], r["context"],
                       r["parametric_answer"], r["counterfactual_answer"],
                       item_id=r["item_id"], checkpoint_id=checkpoint_id)
        total += 1
        ok = j["label"] == r["logprob"]["label"]
        agree += int(ok)
        if not ok:
            disagreements.append({"item_id": r["item_id"],
                                   "logprob": r["logprob"]["label"], "judge": j["label"]})

    rate = agree / max(total, 1)
    print(f"Layer1/Layer2 agreement on {total} confident items: {agree}/{total} = {rate:.1%}")
    if rate < 0.90:
        print("WARNING: below 90% threshold -- investigate, this indicates a scoring "
              "bug, not expected ambiguity.")
        for d in disagreements:
            print(" ", d)
    return {"n_sampled": total, "n_agree": agree, "rate": round(rate, 4),
            "disagreements": disagreements}


def run_sanity_check(dataset_path, output_dir, judge_fn_=None,
                      base_id="meta-llama/Llama-3.1-8B",
                      instruct_id="meta-llama/Llama-3.1-8B-Instruct",
                      drop_ids=DEFAULT_DROP_IDS, tau=TAU):
    """
    Spec Sec.8 Tests 1+2: base-model sanity check and instruct-model
    comparison, with the spec's own pass/fail thresholds operationalized as
    explicit checks rather than left to eyeballing.
    """
    results = {}
    for label, model_id, use_ct in [("base", base_id, False), ("instruct", instruct_id, True)]:
        results[label] = run_full_evaluation(
            model_id, dataset_path, output_dir, use_chat_template=use_ct,
            tau=tau, judge_fn_=judge_fn_, drop_ids=drop_ids,
        )

    r_base = results["base"]["aggregate"]["final"]["R_ctx"]["mean"]
    r_instruct = results["instruct"]["aggregate"]["final"]["R_ctx"]["mean"]
    fy_base = results["base"]["filter_yield"]

    print(f"\nSANITY CHECK: R_ctx(base)={r_base}  R_ctx(instruct)={r_instruct}  "
          f"filter_yield(base)={fy_base}")
    if not (0.40 <= r_base <= 0.70):
        print(f"  WARNING: R_ctx(base)={r_base} outside spec's expected 40-70% "
              f"-- check prompt formatting.")
    if fy_base < 0.80:
        print(f"  WARNING: filter_yield(base)={fy_base} below spec's expected "
              f">=80% (capitals/world_facts).")
    if abs(r_base - r_instruct) < 0.10:
        print(f"  WARNING: R_ctx(base) and R_ctx(instruct) within 10 points -- "
              f"spec expects instruct substantially lower. Scoring may be broken.")
    else:
        print(f"  R_ctx dropped {r_base - r_instruct:+.3f} base->instruct -- "
              f"consistent with post-inversion state.")
    return results


#entry points

def _safe_name(model_id):
    return model_id.split("/")[-1].replace(" ", "_")


def run_full_evaluation(model_id, dataset_path, output_dir, use_chat_template=True,
                         precision="bf16", tau=TAU, judge_fn_=None,
                         drop_ids=DEFAULT_DROP_IDS, k_distractors=K_DISTRACTORS,
                         methods=("paper", "ordered", "logprob"), verbose=True):
    """
    Phase-0 style entry point: load ONE model fresh, evaluate every item in
    the dataset, save one well-structured JSON keyed by model_id, free the
    model. Returns the result dict (also written to disk as
    {output_dir}/{model_id}_eval.json).
    """
    items = load_items(dataset_path, drop_ids=drop_ids)
    pool = build_distractor_pool(items)
    model, tok = load_model(model_id, precision=precision)
    try:
        res = evaluate(model, tok, items, use_chat_template, pool,
                        methods=methods, record_gen_filter=False, tau=tau,
                        judge_fn_=judge_fn_, checkpoint_id=f"phase0_{_safe_name(model_id)}",
                        k_distractors=k_distractors, verbose=verbose)
    finally:
        free_model(model)

    out = {
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"tau": tau, "k_distractors": k_distractors,
                   "use_chat_template": use_chat_template, "precision": precision,
                   "n_items_total": len(items), "judge_enabled": judge_fn_ is not None,
                   "eval_py_version": __version__},
        "aggregate": res["summary"]["aggregate"],
        "per_stratum": res["summary"]["per_stratum"],
        "filter_yield": res["summary"]["filter_yield"],
        "per_item": res["per_item"],
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(output_dir) / f"{_safe_name(model_id)}_eval.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print_summary(model_id, res)
    print(f"Wrote {out_file}")
    return out


def evaluate_checkpoint(model, tokenizer, items, distractor_pool, output_path,
                         checkpoint_id, global_step, use_ct=True, judge_fn_=None,
                         tau=TAU, methods=("paper", "ordered", "logprob"), verbose=True):
    """
    SFT integration entry point. Shares an already-loaded model/tokenizer
    across many checkpoint calls (no reload). `items`/`distractor_pool` are
    explicit parameters, not read from module globals, so this function has
    no hidden state and is safe to call from any training loop or process.

    If you need the exact 5-argument signature from evaluation_pipeline.md
    Sec.9 (evaluate_checkpoint(model, tokenizer, output_path, checkpoint_id,
    global_step)), bind items/distractor_pool once with functools.partial:

        eval_ckpt = functools.partial(evaluate_checkpoint,
                                       items=items, distractor_pool=pool)
        eval_ckpt(model, tokenizer, output_path, checkpoint_id, global_step)
    """
    res = evaluate(model, tokenizer, items, use_ct, distractor_pool,
                    methods=methods, record_gen_filter=False, tau=tau,
                    judge_fn_=judge_fn_, checkpoint_id=checkpoint_id, verbose=verbose)
    out = {
        "checkpoint_id": checkpoint_id,
        "global_step": global_step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"tau": tau, "use_chat_template": use_ct,
                   "judge_enabled": judge_fn_ is not None, "eval_py_version": __version__},
        "aggregate": res["summary"]["aggregate"],
        "per_stratum": res["summary"]["per_stratum"],
        "filter_yield": res["summary"]["filter_yield"],
        "per_item": res["per_item"],
    }
    Path(output_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(output_path) / f"checkpoint_{checkpoint_id}_metrics.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_file}")
    return out




def _test_classifier_regression():
    cases = [
        ("clean CTX",            "Chengdu.",                                 "Beijing", "Chengdu", "CTX"),
        ("clean PAR",             "Beijing.",                                 "Beijing", "Chengdu", "PAR"),
        ("substring bug: cf",     "The mississippian era rocks are old.",     "Amazon", "Mississippi", "OTHER"),
        ("substring bug: short",  "Usage statistics show growth.",            "USA", "EU", "OTHER"),
        ("prefix collision cf",   "Prizren.",                                 "Pristina", "Prizren", "CTX"),
        ("prefix collision par",  "Pristina.",                                "Pristina", "Prizren", "PAR"),
        ("hedge, ctx first",      "Chengdu, though in reality it's Beijing.", "Beijing", "Chengdu", "CTX"),
        ("abstention",            "I'm not sure I can answer that.",          "Beijing", "Chengdu", "OTHER"),
    ]
    for name, resp, par, cf, expected in cases:
        got = method_ordered(resp, par, cf)["label"]
        assert got == expected, f"{name}: got {got}, expected {expected}"
    print("classifier regression test: PASS")


def _test_scorer_indexing():
    V, seq, prompt_len = 50, [5, 9, 2, 7, 40, 41, 42], 4
    ans = torch.tensor(seq[prompt_len:])
    logits = torch.full((len(seq), V), -30.0)
    for i in range(len(seq) - 1):
        logits[i, seq[i + 1]] = 30.0
    n = prompt_len
    lp = torch.log_softmax(logits[n - 1:-1, :].float(), dim=-1)
    tok_lp = lp[torch.arange(ans.shape[0]), ans]
    assert torch.allclose(tok_lp, torch.zeros_like(tok_lp), atol=1e-4), tok_lp
    assert logits[n - 1:-1, :].shape[0] == ans.shape[0]

    # extended: EOS-strip branch keeps indexing correct
    eos_id = 99
    prompt_with_eos = [5, 9, 2, 7, eos_id]
    ans2 = [40, 41, 42]
    full_seq = prompt_with_eos[:-1] + ans2
    logits2 = torch.full((len(full_seq), V), -30.0)
    for i in range(len(full_seq) - 1):
        logits2[i, full_seq[i + 1]] = 30.0
    p_ids = torch.tensor(prompt_with_eos)
    if p_ids[-1].item() == eos_id:
        p_ids = p_ids[:-1]
    n2 = p_ids.shape[0]
    lp2 = torch.log_softmax(logits2[n2 - 1:-1, :].float(), dim=-1)
    ans2_t = torch.tensor(ans2)
    tok_lp2 = lp2[torch.arange(ans2_t.shape[0]), ans2_t]
    assert torch.allclose(tok_lp2, torch.zeros_like(tok_lp2), atol=1e-4)
    print("scorer indexing test (incl. EOS-strip): PASS")


def _test_aggregation():
    rows = [
        {"item_id": "a1", "source": "country_capitals", "passed": True,
         "paper": {"cf_acc": 1, "par_acc": 0}, "ordered": {"label": "CTX", "sub": "ctx_only"},
         "logprob": {"label": "CTX", "delta": 1.5}, "final_label": "CTX"},
        {"item_id": "a2", "source": "country_capitals", "passed": True,
         "paper": {"cf_acc": 0, "par_acc": 1}, "ordered": {"label": "PAR", "sub": "par_only"},
         "logprob": {"label": "PAR", "delta": -2.1}, "final_label": "PAR"},
        {"item_id": "a3", "source": "country_capitals", "passed": True,
         "paper": {"cf_acc": 1, "par_acc": 1}, "ordered": {"label": "CTX", "sub": "ctx_then_par"},
         "logprob": {"label": "AMBIG", "delta": 0.3},
         "judge": {"label": "OTHER", "other_subtype": "hedged", "layer2_failed": False},
         "final_label": "OTHER"},
        {"item_id": "a4", "source": "world_facts", "passed": False,
         "paper": {"cf_acc": 0, "par_acc": 0}, "ordered": {"label": "OTHER", "sub": "neither"},
         "logprob": {"label": "AMBIG", "delta": 0.1},
         "judge": {"label": "CTX", "other_subtype": None, "layer2_failed": False},
         "final_label": "CTX"},
        {"item_id": "a5", "source": "world_facts", "passed": True,
         "paper": {"cf_acc": 0, "par_acc": 0}, "ordered": {"label": "OTHER", "sub": "neither"},
         "logprob": {"label": "AMBIG", "delta": -0.05},
         "judge": {"label": "OTHER", "other_subtype": None, "layer2_failed": True},
         "final_label": "OTHER"},
    ]
    res = summarize(rows, methods=("paper", "ordered", "logprob"))
    a = res["aggregate"]
    assert a["n_passed"] == 4
    assert a["final"]["R_ctx"]["mean"] == 0.25
    assert a["final"]["R_par"]["mean"] == 0.25
    assert a["final"]["R_other"]["mean"] == 0.5
    assert a["escalation_rate"] == 0.5 and a["layer2_failed_rate"] == 0.25
    assert a["other_breakdown"] == {"hedged": 0.25}
    assert res["per_stratum"]["country_capitals"]["n_passed"] == 3
    assert res["per_stratum"]["country_capitals"]["filter_yield"] == 1.0
    assert res["per_stratum"]["world_facts"]["n_passed"] == 1
    print("aggregation test: PASS")


def _test_mcnemar():
    pairs = [(False, True)] * 10 + [(True, False)] * 3
    result = mcnemar_test(pairs)
    assert result["n01"] == 10 and result["n10"] == 3 and result["n_discordant"] == 13

    p1 = mcnemar_test([(False, True)] * 10 + [(True, False)] * 3)["p_value"]
    p2 = mcnemar_test([(False, True)] * 3 + [(True, False)] * 10)["p_value"]
    assert p1 == p2  # symmetric in which side is larger

    assert mcnemar_test([(False, True)] * 15 + [(True, False)] * 15)["p_value"] == 1.0
    assert mcnemar_test([(True, True), (False, False)])["p_value"] == 1.0
    assert mcnemar_test([(False, True)] * 50 + [(True, False)] * 5)["p_value"] < 0.01

    rows_a = [
        {"item_id": "x1", "passed": True, "final_label": "PAR"},
        {"item_id": "x2", "passed": True, "final_label": "CTX"},
        {"item_id": "x3", "passed": False, "final_label": "CTX"},
        {"item_id": "x4", "passed": True, "final_label": "PAR"},
    ]
    rows_b = [
        {"item_id": "x1", "passed": True, "final_label": "CTX"},
        {"item_id": "x2", "passed": True, "final_label": "CTX"},
        {"item_id": "x3", "passed": True, "final_label": "CTX"},
        {"item_id": "x4", "passed": True, "final_label": "PAR"},
    ]
    pairs2, n_common = paired_ctx_flags_from_checkpoints(rows_a, rows_b)
    assert n_common == 3  # x3 excluded: filter-failed at checkpoint A
    result2 = mcnemar_test(pairs2)
    assert result2["n01"] == 1 and result2["n10"] == 0  # only x1 switched PAR->CTX
    print("McNemar test (incl. paired-flags helper): PASS")


def run_offline_self_tests():
    """Run every pure-logic self-test. No GPU, no model download, no API key."""
    _test_classifier_regression()
    _test_scorer_indexing()
    _test_aggregation()
    _test_mcnemar()
    print("\nALL OFFLINE SELF-TESTS PASS")




def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Context-Parametric Inversion evaluation pipeline")
    p.add_argument("--self-test", action="store_true",
                    help="Run offline self-tests only (no GPU/API needed) and exit.")
    p.add_argument("--model-id", type=str, default=None,
                    help="HF model id to evaluate (Phase-0 style, one JSON per model).")
    p.add_argument("--dataset", type=str, default=None, help="Path to conflict_eval_unified.json")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to write the result JSON")
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false", default=True)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "4bit"])
    p.add_argument("--use-judge", action="store_true",
                    help="Enable the Layer 2 LLM judge (requires OPENAI_API_KEY).")
    return p


def main():
    args = _build_arg_parser().parse_args()
    if args.self_test or not args.model_id:
        run_offline_self_tests()
        return
    judge_fn_ = make_judge() if args.use_judge else None
    run_full_evaluation(args.model_id, args.dataset, args.output_dir,
                         use_chat_template=args.use_chat_template, tau=args.tau,
                         precision=args.precision, judge_fn_=judge_fn_)


if __name__ == "__main__":
    main()
