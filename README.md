# MEM-HQS — Cognitive-Emotional Estimator (target-routed)

Estimates **Valence / Arousal / Cognitive load** from wearable **PPG + situational context text**.

Requirements: **PaPaGEI model · trained checkpoint · (optional) GPT API key**

Architecture: **Valence = context only**, **Arousal · Cognitive load = context + PPG** (target-routed).

---

## Folder structure

- `training/` — model and training code
  - `core/` : `model.py`, `train.py`, `papagei_embed.py`, etc. (**model definition and training method** = reusable implementation)
  - `target_routed/` : `TARGET_ROUTED_ARCHITECTURE.md` + `plan/` (configs) + `scripts/` (original run scripts, for reference)
- `deployment/` — deployment pipeline (collection → merge → inference)
  - `digital_behavior/` : PC (ActivityWatch) and Android digital-behavior collectors
  - `ppg/` : Galaxy PPG reception, preprocessing, and PaPaGEI embedding
  - `merge.py` : collected events + PPG → `datastream.jsonl`
  - `runtime_pipeline/` : `run_target_routed_deploy.py` (datastream → V/A/C)
- `results/FINAL_RESULTS.md` — final results table

---

## Setup (one time)

1. Install dependencies

```bash
pip install -r deployment/runtime_pipeline/requirements.txt
```

2. Place the PaPaGEI model at the **repository root**

```bash
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model.git
# Download the PaPaGEI-P weight → weights/papagei_p.pt
```

3. Prepare the trained checkpoint: `full.pt` (target-routed)

4. (Optional) GPT API key — only needed when generating context sentences with OpenAI. Add it to `.env`:

```text
OPENAI_API_KEY=sk-...
```

---

## Deployment: V/A/C inference

Pipeline = **collection → merge → inference**

```bash
# 1) Collect PC/Android digital behavior (deployment/digital_behavior/) + Galaxy PPG
python deployment/ppg/receiver.py --port 5005

# 2) Merge events + PPG → datastream.jsonl
python deployment/merge.py --processed processed_*.csv --digital events_*.jsonl --out datastream.jsonl

# 3) Inference: datastream → V/A/C
python deployment/runtime_pipeline/run_target_routed_deploy.py \
  --in datastream.jsonl --out final.jsonl --ckpt full.pt
```

Optional personalization using the first K self-report samples for calibration:

```bash
python deployment/runtime_pipeline/run_target_routed_deploy.py \
  --in datastream.jsonl --out final.jsonl --ckpt full.pt \
  --calibration deployment/runtime_pipeline/eb_calibration_sample.json \
  --enrollment enroll.jsonl --enroll-k 4
```

- Without personalization (default) = zero-shot. Provide `--calibration` + `--enrollment` to enable personalization.
- `enroll.jsonl` uses the same format as `datastream.jsonl`, with `label:{valence,arousal,cognitive_load}` added to each row and the same `user_id`.

---

## Training: model and method (reference)

- **Model and training method code** = `training/core/` (`model.py`, `train.py`, `dataset.py`, `losses.py`). Reusable.
- **Experiment configs** = `training/target_routed/plan/*.yaml` + `TARGET_ROUTED_ARCHITECTURE.md`.
- **Original run scripts** = `training/target_routed/scripts/` (e.g., `run_target_routed_final.py`).

```bash
python training/target_routed/scripts/run_target_routed_final.py all 0
```

- ⚠️ These scripts depend on the **original research data and paths**, so they are not directly runnable from this folder alone. Retraining requires the original CASE/EEVR/MAUS datasets (private) and path configuration at the top of the scripts. The model structure and method can still be reproduced using the code in `training/core/`.

---

## Results

`results/FINAL_RESULTS.md` — integrated table covering source (PPG-only/B0/B1/B2), external zero-shot, and personalization lift.

---

## Notes

- PaPaGEI: <https://github.com/Nokia-Bell-Labs/papagei-foundation-model> (frozen; no fine-tuning)
- To override paths, use the environment variables `PAPAGEI_ROOT` and `PAPAGEI_CKPT`.
- License: `LICENSE`.
