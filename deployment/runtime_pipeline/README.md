# MEM-HQS runtime pipeline

상태추정 연구 코드와 실제 실행 코드를 섞지 않기 위한 최소 런타임 폴더다.
`hqs-clsp`의 수집→merge→finalize 흐름에서 필요한 부분만 가져오고, 현재
`mem_hqs_affect/model_v3` 모델 형식에 맞게 정리했다. 이 폴더는
`hqs-clsp`를 import하지 않는다.

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

## 가져온 것과 제외한 것

| 구분 | 처리 |
|---|---|
| `hqs-clsp/digital_behavior/pc_collector` | 필요한 ActivityWatch 로직만 `collectors/pc_activitywatch.py`로 정리 |
| `hqs-clsp/digital_behavior/Android` | 앱 전환 수집에 필요한 최소 Android 프로젝트만 정리 |
| `hqs-clsp/context/digital.py` | 시간구간 매칭·요약을 `digital.py`로 통합; 기본 경로는 API 불필요 |
| `hqs-clsp/merge.py` | 관측 provenance와 `not_recorded`를 보존하도록 `merge.py`로 교정 |
| `hqs-clsp/finalize.py` | 새 per-target-head 체크포인트용으로 `finalize.py`를 재작성 |
| PPG receiver/pipeline | `text-clsp/ppg`와 해시가 같아 복사하지 않고 기존 파일 재사용 |
| 구형 `state_estimation_mvp` | EEVR Significance proxy·단일 3-output head라 제외 |
| 데이터/outputs/PaPaGEI 저장소 | 대용량·개인 데이터·외부 의존성이므로 복사하지 않음 |

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
