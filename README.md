# MEM-HQS · Target-routed multimodal affect estimation

Estimate **Valence / Arousal / Cognitive load** from wearable **PPG** + deployable
**operational context** text, for on-device (AR / smartwatch) use.

The model routes each target through a different path:

| Target | Population predictor | PPG | Personalization |
|---|---|---|---|
| **Valence** | frozen **Context-only** | — (no PPG) | target-wise EB intercept |
| **Arousal** | **Context + PPG** direct fusion | ✅ | target-wise EB intercept |
| **Cognitive load** | **Context + PPG** direct fusion | ✅ | target-wise EB intercept |

Text is encoded by DistilBERT; PPG is encoded by a **frozen PaPaGEI-P** (ResNet1D,
512-d) foundation model. The Valence route carries no PPG by design — this is
verified by a hard invariance check (its predictions are identical to the
context-only model).

---

## Repository layout

```
training/
  core/              model definition + training method
                     (model, train, dataset, losses, splits, alignment,
                      input_contract, papagei_embed, lodo)
  target_routed/     the target-routed experiment
                     run_*.py/.cmd, evaluate_*.py, plan/*.yaml,
                     TARGET_ROUTED_ARCHITECTURE.md
deployment/
  ppg/               Galaxy-watch PPG receiver + 6-step preprocessing
                     + PaPaGEI-P embedding (receiver.py, pipeline.py, papagei-p.py)
  runtime_pipeline/  merge context+PPG -> finalize -> V/A/C in [0,1]
                     (merge.py, context_text.py, finalize.py, tests/)
results/
  main_results.csv   headline held-out numbers (see below)
```

> **Not included (by design):** raw participant datasets (`Data_files/`), model
> weights (`*.pt`), API keys (`.env`), and the digital-behaviour **collection**
> code (ActivityWatch / Android). Only source code and one results table are here.

---

## Architecture

```
7-slot operational context ── DistilBERT ── context representation ─┐
                                                                    ├─▶ V / A / C  (each in [0,1])
10 s PPG ── frozen PaPaGEI-P (512-d) ── trainable PPG path ─────────┘
                                     (Arousal / Cognitive load only)

first K enrollment labels ── target-wise empirical-Bayes shrunk intercept  (optional personalization)
```

- Datasets used for training: **CASE + EEVR + MAUS** (participant-disjoint splits, seeds 42/43/44).
- Zero-shot evaluation datasets (never trained on): WESAD, CogWear, EmoWear, VRFS.
- PaPaGEI-P is **frozen** and used only to produce 512-d embeddings — it is never fine-tuned.
- Personalization is a leakage-free few-shot **output intercept** (no encoder/head update).

See `training/target_routed/TARGET_ROUTED_ARCHITECTURE.md` for the exact contract.

---

## Results (held-out, `results/main_results.csv`)

Trained-source **participant-holdout**, K=0, 3-seed mean. Binary Accuracy / Macro-F1
use the **1–3 vs 4–5** boundary (normalized threshold **0.625**). CCC is the primary
(continuous) metric.

| Model | Valence CCC | Arousal CCC | Cog-load CCC |
|---|---|---|---|
| PPG-only *(diagnostic, base-lineage run)* | 0.090 | 0.072 | 0.153 |
| **B0** context-only | 0.509 | 0.501 | 0.647 |
| **B1** uniform context+PPG | 0.532 | 0.440 | 0.485 |
| **B2** target-routed (proposed) | 0.509 | 0.456 | 0.492 |

(Full CCC / Accuracy / Balanced-Accuracy / Macro-F1 per target in the CSV. Each row
carries a `run` column: B0/B1/B2 are the most-recent `target_routed_title_final`
run; **PPG-only is a diagnostic baseline from the base-lineage run
`zrecording_lag4_p10s`** — a different run, so its absolute values are not
seed-matched to B0/B1/B2 and should be read only as an order-of-magnitude baseline.)

**Reading it honestly:**
- The source held-out numbers rest on a strong **title-based contextual prior**
  (participant-disjoint but not stimulus-disjoint); do not read them as general
  real-life context.
- On this in-distribution split, adding PPG does not improve CCC — the content
  prior is already strong.
- PPG's benefit appears under **domain shift** (external zero-shot): the
  context+PPG route gives a significant CCC increment for Cognitive load (CogWear)
  and Arousal (WESAD) over the matched context-only route (participant-paired
  bootstrap CI excludes zero). Those increment tables are produced by the
  `evaluate_external_zero_shot.py` script.
- **PPG-only** stays near chance (CCC ≈ 0.07–0.15) for all three targets — physiology
  alone does not decode affect for unseen participants. It is included as a diagnostic
  baseline from the base-lineage run (`run` column), not the target-routed run.

---

## Deployment (inference)

```
Galaxy Watch UDP (PPG 25 Hz + ACC)
  → deployment/ppg/receiver.py     receive
  → deployment/ppg/pipeline.py     invert → ACC-Kalman → 0.5–8 Hz bandpass
                                   → flatline gate → z-score → 25→125 Hz
                                   → 10 s / 2 s sliding windows
  → deployment/runtime_pipeline/merge.py       align context + PPG
  → deployment/runtime_pipeline/finalize.py    frozen PaPaGEI-P 512-d + model
  → V / A / Cognitive load, each in [0,1]
```

Run the plumbing test (no model needed):

```bash
python -m unittest deployment.runtime_pipeline.tests.test_smoke -v
```

Real inference needs a trained checkpoint and the PaPaGEI-P weights (both kept
private): pass them to `finalize.py` via `--ckpt` and `--papagei-ckpt`. See
`deployment/runtime_pipeline/README.md`.

> Production note: connecting a trained checkpoint to the runtime still requires
> a text-schema migration and a watch-domain (2 s overlapping window) check; this
> repository publishes the pipeline, not a production-connected build.

---

## Reference

PPG foundation model: **PaPaGEI-P** (frozen). See `deployment/ppg/README.md`.

## License

See `LICENSE`.
