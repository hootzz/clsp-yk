# MEM-HQS runtime pipeline

## 데이터 흐름

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
  → final_datastream.jsonl (V/A/Cog, 모두 [0,1])
```

## 1. 준비

PowerShell에서 `text-clsp`를 작업 폴더로 둔다.

```powershell
& 'D:\2026\test\venv310\Scripts\python.exe' -m pip install -r runtime_pipeline\requirements.txt
```

실제 finalize에는 별도의 PaPaGEI repository와 `papagei_p.pt`가 필요하다.
PaPaGEI-S 체크포인트는 이 모델에 사용하지 않는다.

## 2. 수집

PPG/ACC:

```powershell
Set-Location runtime_pipeline\data\watch
& 'D:\2026\test\venv310\Scripts\python.exe' ..\..\..\ppg\receiver.py --port 5005
```

PC 디지털 행동(ActivityWatch가 먼저 실행되어 있어야 함):

```powershell
Set-Location D:\2026\clsp\text-clsp
& 'D:\2026\test\venv310\Scripts\python.exe' -m runtime_pipeline.collectors.pc_activitywatch
```

Android 수집은 [android/README.md](android/README.md)를 따른다.

## 3. Merge

센서로 측정하지 않은 슬롯은 추측하지 않고 `not_recorded`로 남는다. 실험
프로토콜상 착석이 확실할 때만 `--posture sitting`처럼 명시한다.

```powershell
& 'D:\2026\test\venv310\Scripts\python.exe' -m runtime_pipeline.merge `
  --processed runtime_pipeline\data\watch\processed_YYYYMMDD_HHMMSS.csv `
  --raw runtime_pipeline\data\watch\raw_YYYYMMDD_HHMMSS.csv `
  --digital runtime_pipeline\data\pc\events_YYYYMMDD.jsonl `
  --posture sitting `
  --environment indoor `
  --temporal continuous `
  --out runtime_pipeline\data\datastream.jsonl
```

기본 context backend는 재현 가능한 deterministic 문장이다. GPT-4o-mini를
쓰려면 `OPENAI_API_KEY` 환경변수와 `--context-backend openai`를 추가한다.
두 방식 모두 상태/감정 단어 누출 검사를 수행한다.

## 4. Finalize

먼저 외부 모델 없이 배관만 검사:

```powershell
& 'D:\2026\test\venv310\Scripts\python.exe' -m runtime_pipeline.finalize `
  --in runtime_pipeline\data\datastream.jsonl `
  --out runtime_pipeline\data\final_mock.jsonl `
  --mock
```

실제 추론:

```powershell
& 'D:\2026\test\venv310\Scripts\python.exe' -m runtime_pipeline.finalize `
  --in runtime_pipeline\data\datastream.jsonl `
  --out runtime_pipeline\data\final_datastream.jsonl `
  --ckpt mem_hqs_affect\outputs\context7_gpt\model_v3\full.pt `
  --papagei-root D:\2026\clsp-jw\ppg\papagei-foundation-model `
  --papagei-ckpt D:\2026\clsp-jw\ppg\weights\papagei_p.pt
```

레코드에 이미 512차원 `ppg_embedding`이 있으면 PaPaGEI를 다시 실행하지 않는다.

```powershell
& 'D:\2026\test\venv310\Scripts\python.exe' -m runtime_pipeline.finalize `
  --in runtime_pipeline\data\embedded.jsonl `
  --out runtime_pipeline\data\final_datastream.jsonl `
  --embedding-only
```

영상/세션당 PPG window가 여러 개면 `--pool-windows`와 필요시
`--session-id-field session_id`를 사용한다.

## 5. 테스트

```powershell
& 'D:\2026\test\venv310\Scripts\python.exe' -m unittest runtime_pipeline.tests.test_smoke -v
```

## 경계

- 출력 세 타깃은 모두 `[0,1]`이다. 구형 finalize의 V/A `1–5` 가정은 제거했다.
- PPG는 PaPaGEI-P를 사용한다. PaPaGEI-S와 혼용하지 않는다.
- `not_recorded`를 `alone`, `low`, `0`, `sitting`으로 자동 대체하지 않는다.
- 실제 체크포인트 연결은 `config_512_context7_gpt.yaml` 학습·평가가 끝난 후 한다.
- 수집 JSONL과 PPG는 참여자 데이터이므로 Git에 넣지 않는다.
