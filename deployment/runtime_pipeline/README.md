# MEM-HQS runtime pipeline

## Data flow

```text
Galaxy Watch UDP
  → ../ppg/receiver.py (6-step preprocessing, 125 Hz × 10 s)
  → processed_*.csv + raw_*.csv

ActivityWatch / Android AccessibilityService
  → collectors/pc_activitywatch.py / android/
  → events_*.jsonl

processed PPG + ACC + digital events
  → merge.py
  → datastream.jsonl (observed seven slots + context text + PPG)

datastream.jsonl
  → finalize.py
  → PaPaGEI-P 512 + mem_hqs_affect/model_v3
  → final_datastream.jsonl (V/A/Cog, all in [0,1])
```

## 1. Setup

Run the following from the repository root:

```powershell
python -m pip install -r runtime_pipeline\requirements.txt
```

Actual finalization requires a separate PaPaGEI repository and `papagei_p.pt`.
The PaPaGEI-S checkpoint is not used by this model.

## 2. Collection

PPG/ACC:

```powershell
Set-Location runtime_pipeline\data\watch
python ..\..\..\ppg\receiver.py --port 5005
```

PC digital behavior (ActivityWatch must already be running):

```powershell
Set-Location <repository-root>
python -m runtime_pipeline.collectors.pc_activitywatch
```

For Android collection, follow [android/README.md](android/README.md).

## 3. Merge

Slots not measured by sensors are not inferred and remain `not_recorded`. Only specify values such as `--posture sitting` when sitting is guaranteed by the experimental protocol.

```powershell
python -m runtime_pipeline.merge `
  --processed runtime_pipeline\data\watch\processed_YYYYMMDD_HHMMSS.csv `
  --raw runtime_pipeline\data\watch\raw_YYYYMMDD_HHMMSS.csv `
  --digital runtime_pipeline\data\pc\events_YYYYMMDD.jsonl `
  --posture sitting `
  --environment indoor `
  --temporal continuous `
  --out runtime_pipeline\data\datastream.jsonl
```

The default context backend generates reproducible deterministic sentences. To use GPT-4o-mini, set the `OPENAI_API_KEY` environment variable and add `--context-backend openai`. Both modes perform leakage checks for state/emotion terms.

## 4. Finalize

First, test only the pipeline wiring without external models:

```powershell
python -m runtime_pipeline.finalize `
  --in runtime_pipeline\data\datastream.jsonl `
  --out runtime_pipeline\data\final_mock.jsonl `
  --mock
```

Actual inference:

```powershell
python -m runtime_pipeline.finalize `
  --in runtime_pipeline\data\datastream.jsonl `
  --out runtime_pipeline\data\final_datastream.jsonl `
  --ckpt mem_hqs_affect\outputs\context7_gpt\model_v3\full.pt `
  --papagei-root <path-to-papagei-repository> `
  --papagei-ckpt <path-to-papagei-p-checkpoint>
```

If a record already contains a 512-dimensional `ppg_embedding`, PaPaGEI is not run again.

```powershell
python -m runtime_pipeline.finalize `
  --in runtime_pipeline\data\embedded.jsonl `
  --out runtime_pipeline\data\final_datastream.jsonl `
  --embedding-only
```

If a video/session contains multiple PPG windows, use `--pool-windows` and, when needed, `--session-id-field session_id`.

## 5. Test

```powershell
python -m unittest runtime_pipeline.tests.test_smoke -v
```

## Constraints

- All three output targets are in `[0,1]`. The legacy finalize assumption of V/A in `1–5` has been removed.
- PPG uses PaPaGEI-P. Do not mix it with PaPaGEI-S.
- Do not automatically replace `not_recorded` with `alone`, `low`, `0`, or `sitting`.
- Connect the actual checkpoint only after training and evaluation with `config_512_context7_gpt.yaml` are complete.
- Collected JSONL and PPG files contain participant data and must not be committed to Git.
