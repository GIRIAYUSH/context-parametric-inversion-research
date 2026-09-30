# Research log — Context-Parametric Inversion (CPI)

**Status as of 2026-09-29.** Findings and the claims each one supports, with the test behind it and
where to reproduce it. All mechanistic work: `meta-llama/Llama-2-7b-hf` with LoRA SFT, seed 0, two runs.

| Run | SFT data | Checkpoints | Peak | Trough | Recovery |
|---|---|---|---|---|---|
| tulu | `allenai/tulu-v2-sft-mixture` (326k) | every 50 steps, 50–5000 | 500 | 3150 | 5000 |
| alpaca | `tatsu-lab/alpaca` (52k) | every 50 steps, 50–800 | 100 | 700 | — |

Evaluation set: 412 knowledge-conflict items in three categories — country capitals, famous biographies,
world facts. Each item has a question, a passage asserting a counterfactual answer, and the model's
parametric (true) answer. **R_ctx** = share of items (passing the knowledge filter) answered with the
passage's answer.

---

## 1. Context reliance rises, drops at a trough, and recovers during SFT

| | Peak | Trough | Recovery |
|---|---|---|---|
| tulu R_ctx | 93.8% (step 500) | 77.4% (step 3150) | 91.2% (step 5000) |
| alpaca R_ctx | 88.1% (step 100) | 76.7% (step 700) | — |

- Phase picks are the local extrema of the full R_ctx trajectory (tulu 100 checkpoints, alpaca 16).
- Measurement: the log-prob scorer was corrected (single-space answer tokenisation) and the phase-window
  checkpoints rescored; corrected deltas match the reference exactly (max diff 0.0000).

*Reproduce:* `src/mechanistic-analysis/trajectory_plots.ipynb`, `score.ipynb`; figures `figures/aggregate_trajectory.png`, `figures/stratum_trajectory.png`.

## 2. The drop is carried by the country-capitals category

Change in pooled R_ctx split into each category's contribution (category change × its share of items):

| Transition | Pooled change | Capitals | Biographies | World facts |
|---|---|---|---|---|
| tulu peak → trough | −16.4 pts | **−18.7** | +2.7 | −0.4 |
| tulu trough → recovery | +13.8 pts | **+13.0** | −0.2 | +1.0 |
| alpaca peak → trough | −11.4 pts | **−10.9** | +0.2 | −0.7 |

Averaging each phase over its ±2-checkpoint window gives the same picture: capitals = 127% of the tulu drop,
87% of the tulu recovery, 118% of the alpaca drop.

Item-level switches (exact McNemar, Bonferroni ×12):

| Transition | Capitals: CTX→PAR / PAR→CTX | p |
|---|---|---|
| tulu peak → trough | **76 / 0** | 3.2 × 10⁻²² |
| tulu trough → recovery | **0 / 53** | 2.7 × 10⁻¹⁵ |
| alpaca peak → trough | **44 / 0** | 1.4 × 10⁻¹² |

Biographies and world facts show no significant switching in any transition (tulu biographies shift
slightly *toward* the context: 1 / 12, p = 0.04).

**Claim:** the peak-to-trough decline in context reliance, and tulu's recovery, are concentrated in the
country-capitals items; every capitals item that switches does so in the direction of the trend.

## 3. The same capitals flip in both runs

Of 180 capitals items classified in both runs: 36 flip in both, 37 in tulu only, 6 in alpaca only, 101 in
neither. A capital that flips under tulu is 8× more likely to flip under alpaca (49% vs 6%;
Fisher odds ratio 16.4, p = 1.3 × 10⁻¹¹). 36 of alpaca's 42 flipping capitals also flip under tulu.

**Claim:** which capitals flip is a property of the items — two SFT runs on different data flip largely the
same facts.

*Reproduce:* `flip_set.ipynb`, `flip_set.json`.

## 4. What does *not* explain which items flip

| Candidate | Test | Result |
|---|---|---|
| Answer length | Switch rate by answer length (tokens) within category | Capitals switch 28–60% at every length 1–6; biographies / world facts of the same lengths 0–9% |
| Parametric strength | No-context log-prob of the true answer (base model), and its margin over the counterfactual | Biographies are as strongly known (median −0.26 vs −0.23 per token) yet 1/93 switch; within capitals it explains 2–3% of switching (pseudo-R²) |
| Category after controls | Logistic regression: switch ~ length + strength + margin (+ capitals) | Capitals remain decisive: LR χ² = 60.6, p = 6.9 × 10⁻¹⁵, OR ≈ 26 (tulu); χ² = 25.3, p = 4.9 × 10⁻⁷, OR ≈ 12 (alpaca) |
| Frequency in SFT data | Count training examples mentioning each country / country + "capital" / country + its real capital | No flip-vs-stable difference in either run (p = 0.19–0.90) |
| "Is the answer in context? no" habit | Pattern in the generated continuations | Only at tulu's trough (49% of capitals, 8% world facts, 0% biographies; gone at recovery; never in alpaca); only weakly tied to flipping (61% vs 42%, p = 0.017) |

Parametric-strength scores were validated: adapters applied (0/412 scores identical to base), token counts
match the stored evaluations exactly, and every filter-passed item scores the true answer above the
counterfactual.

**Claim:** the capitals concentration is not explained by answer length, by how strongly the model knows
the fact, by how often the fact appears in the SFT data, or by the tulu-specific output habit.

*Reproduce:* `parametric_strength.ipynb`, `parametric-strength/`, `why_capitals.ipynb`, `why_capitals_sft_counts.csv`.

## 5. Flip set

Built from the corrected evaluations at the phase checkpoints; items must pass the knowledge filter at every
phase. Labels were checked against the model's generated text (agreement 76–78 / 80 tulu, 46–48 / 48 alpaca)
and against the neighbouring rescored checkpoints ("robust" = holds its label across the phase window).

| Run | Flips | Robust flips | Robust stable capitals |
|---|---|---|---|
| tulu | 80 (76 capitals; 55 recover by 5000) | 51 | 107 |
| alpaca | 48 (43 capitals) | 30 | 137 |

Robust flipping and robust stable capitals are matched on answer length (median 3 vs 3 tokens, tulu) and
no-context strength (−0.22 vs −0.23).

## 6. Logit lens: where the answer becomes readable

**Method.** On the generation prompt, at the answer position, project the residual stream after every layer
through the final norm and unembedding; lean `D[l] = logit(first token of context answer) − logit(first token
of own answer)` (= log-odds between the two). Compare flipping vs stable capitals with a
difference-in-differences across checkpoints, each checkpoint scale-corrected by its RMS over all items;
95% bootstrap CIs (2000 resamples).

**Validation.** The final-layer lean matches the answer the model actually generated on 98.3–99.7% of items
at every checkpoint; the lens's top token equals the generated first token on 74/75 checked items.

**Results.**
- No answer is readable before layer ~19 for any item or checkpoint (D ≈ 0).
- From layer 19, flipping and stable capitals diverge — in every comparison, every layer 19–26 has a CI
  excluding 0:

| Comparison | Peak layer | DiD at peak [95% CI] | Onset |
|---|---|---|---|
| tulu peak → trough | 20 | −0.52 [−0.67, −0.37] | 19 |
| tulu trough → recovery | 23 | +0.31 [+0.20, +0.42] | 18 |
| alpaca peak → trough | 22 | −0.44 [−0.59, −0.31] | 19 |

- 100% of flipping items lose context lean over layers 19–26 at the trough (tulu and alpaca) and 100% regain
  it at recovery; flips move far more than stable items (Mann-Whitney p = 3.8 × 10⁻¹⁰, 1.1 × 10⁻⁷, 1.3 × 10⁻¹¹).
- At the trough, the flipping items' layers 19+ lean actively to the model's own answer (about −2.5 to −3
  logits at layers 23–24, tulu), where at the peak the same layers lean to the context (+3 to +5).
- The weakening in these layers is specific to capitals: stable capitals also lose lean there (88% tulu, 91%
  alpaca) while stable biographies and world facts do not (16–48%, mean change ≈ 0).

**Claim:** the answer is first readable at layers ~19+; there, the trough model writes the model's own answer
for the flipping capitals instead of the context answer.

*Reproduce:* `logit_lens.ipynb` (Colab), `logit_lens_analysis.ipynb` (local), data `logit-lens-v2/`;
figure `figures/logit_lens.png`.

## 7. Activation patching: where the change is caused

**Method.** For each flipping capital, run the source model (peak, or recovery) and keep its residual stream
after every layer; run the trough model with its residual stream at one layer `l` (all token positions)
replaced by the source's, all later layers using trough weights. Restoration
`R(l) = (D_patched − D_trough) / (D_source − D_trough)`. Validation: clean runs reproduce the lens exactly
(max difference 0.000); a no-op patch changes nothing (0.0000); R = −0.06 at layer 1 and +1.00 at layer 31.

**Logit restoration (median R):**

| Layer | 4 | 6 | 7 | 8 | 9 | 10 | 12 | 13 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| tulu peak → trough (n = 59) | 0.10 | 0.13 | 0.36 | **0.88** | 0.78 | 0.87 | 0.90 | 1.05 | 1.20 | 1.00 |
| alpaca peak → trough (n = 36) | 0.06 | 0.07 | 0.08 | 0.19 | 0.21 | 0.21 | **0.52** | 0.67 | 0.81 | 1.04 |
| tulu recovery → trough (n = 39) | 0.11 | 0.27 | 0.38 | 0.44 | **0.56** | 0.60 | 0.75 | 0.80 | 0.83 | 0.88 |

Bold = first layer with median restoration ≥ 0.5.

The largest single-layer contributions: tulu blocks 7–8 (+0.23, **+0.52**); alpaca blocks 11–14 (+0.13 to +0.18
each); tulu recovery spread evenly over blocks 4–13.

**Generation (the patched trough model's actual answer, classified by the evaluation's own classifier):**

| Patched layer | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 19–31 |
|---|---|---|---|---|---|---|---|---|---|
| tulu peak → trough | 2% | 15% | 12% | **85%** | 81% | 83% | 98% | 93% | 97–100% |
| alpaca peak → trough | 8% | 8% | 17% | 33% | 33% | **67%** | 81% | 97% | 97–100% |
| tulu recovery → trough | 5% | 13% | 31% | **51%** | 77% | 90% | 90% | 97% | 97–100% |

Baselines: unpatched trough 0% context answers; unpatched source 95–100%; no garbled outputs (0% "other").
Generation tracks the logit sweep within ~10 points at every layer. Example (Armenia): trough says
"Yerevan."; patched at layer 6 → "Yerevan."; at layer 8 → "Gyumri."

**Claim:** restoring the source model's residual stream from layer ~8 (tulu) or ~12 (alpaca) onward is
sufficient to make the trough model generate the context answer again; the trough's changed computation
lies in layers ~5–16, and its later layers produce the context answer when given the source representation.

### 7b. Necessity: the trough's state breaks the healthy model at the same layers

**Method.** Reverse direction on the same items: the trough's residual stream at layer `l` is put into the
healthy model (peak, or recovery), which runs the remaining layers with its own weights.
Damage `N(l) = (D_patched − D_healthy) / (D_trough − D_healthy)` (0 = unchanged, 1 = fully trough-like), plus the
share of items where the corrupted healthy model *generates its own (parametric) answer*. Pre-registered:
the first layer with median damage ≥ 0.5 lies within ±2 layers of the sufficiency restoration layer, in all three.
Consistency check: unpatched leans equal the sufficiency run's for the same items (max difference 0.0000);
damage 0.00 at layer 1 and 1.00 at layer 31.

| | Damage ≥ 0.5 first at | Generates own answer ≥ 50% first at | Sufficiency layer |
|---|---|---|---|
| alpaca trough → peak (n = 36) | 12 | 14 (36% at 12, 81% at 14) | 12 |
| tulu trough → peak (n = 59) | 10 | 8 (51%) | 8 |
| tulu trough → recovery (n = 39) | 7 | 8 (62%) | 9 |

Baselines: unpatched healthy model gives its own answer on 0–3% of items, unpatched trough on 100%; no
garbled outputs (0% "other"). Pre-registered prediction: **holds** in all three.

| Patched layer | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 19–31 |
|---|---|---|---|---|---|---|---|---|---|
| tulu: peak generates own answer | 2% | 2% | 5% | 51% | 49% | 64% | 88% | 93% | 93–100% |
| alpaca: peak generates own answer | 0% | 0% | 3% | 8% | 8% | 36% | 81% | 94% | 94–100% |
| tulu: recovery generates own answer | 3% | 8% | 36% | 62% | 85% | 92% | 92% | 95% | 95–100% |

**Claim:** the trough's computation up to layers ~8–14 is sufficient to make a healthy model answer from
memory; together with 7, the same early-to-middle layer range is both sufficient to restore and sufficient to
induce the flip, in logits and in generated text, in both runs and in the recovery.

*Reproduce:* `activation_patching.ipynb` (Colab), data `activation-patching/`; figure `figures/patching.png`.

## 8. The mechanistic picture

1. SFT's context-reliance drop and recovery are carried by country-capitals items, and largely the same items
   in two independent SFT runs.
2. The change that makes these items flip is computed in the early-to-middle layers (~5–16): patching the
   healthy model's state there into the trough restores context-following, and patching the trough's state
   into the healthy model induces the flip — at the same layers, in logits and in generated text, in both runs
   and in the recovery.
3. That change becomes readable as an answer only at layers ~19+, where the trough model writes its own
   answer instead of the context answer.

---

## Figures

| File | Content |
|---|---|
| `src/mechanistic-analysis/figures/aggregate_trajectory.png` | R_ctx over SFT, both runs |
| `src/mechanistic-analysis/figures/stratum_trajectory.png` | R_ctx by category over SFT |
| `src/mechanistic-analysis/figures/logit_lens.png` | Per-layer lean, flipping vs stable capitals, per phase |
| `src/mechanistic-analysis/figures/patching.png` | Restoration and generated-answer share by patched layer |

## Next

1. Attention vs FFN in layers ~5–16: restore only attention outputs, then only FFN outputs; then per layer;
   then token positions (country token vs passage vs answer position).
2. An intervention derived from the located component, evaluated on held-out items, general benchmarks,
   and against baselines.
3. Paper draft.
