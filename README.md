# MEM-HQS — Cognitive-Emotional Estimator (target-routed)

웨어러블 **PPG + 상황 맥락 텍스트**로 **Valence / Arousal / Cognitive load**를 추정한다.
이 폴더 하나로 학습·배포가 끝난다. 사용자는 **PaPaGEI 모델 · 학습 체크포인트 · (선택) GPT 키**만 넣으면 된다.

구조: **Valence = 맥락만**, **Arousal · Cognitive load = 맥락 + PPG** (target-routed).

---

## 폴더 구성

- `training/` — 모델·학습 코드
  - `core/` : `model.py`, `train.py`, `papagei_embed.py` 등 (모델 정의·임베딩)
  - `target_routed/` : 학습·평가 실행 스크립트
- `deployment/` — 배포 (수집 → 병합 → 추론)
  - `digital_behavior/` : PC(ActivityWatch)·Android 디지털 행동 수집기
  - `ppg/` : Galaxy PPG 수신·전처리·PaPaGEI 임베딩
  - `merge.py` : 수집 이벤트 + PPG → `datastream.jsonl`
  - `runtime_pipeline/` : `run_target_routed_deploy.py` (datastream → V/A/C)
- `results/FINAL_RESULTS.md` — 최종 결과표

---

## 준비 (한 번만)

1. 의존성

```bash
pip install -r deployment/runtime_pipeline/requirements.txt
```

2. PaPaGEI 모델 — **repo 루트에** 배치

```bash
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model.git
# PaPaGEI-P weight 다운로드 →  weights/papagei_p.pt
```

3. 학습된 체크포인트 준비: `full.pt` (target-routed)

4. (선택) GPT 키 — openai로 맥락 문장 생성할 때만. `.env` 에:

```text
OPENAI_API_KEY=sk-...
```

---

## 배포: 실제 V/A/C 뽑기

흐름 = **수집 → 병합 → 추론**

```bash
# 1) 수집: PC/Android 디지털 행동 (deployment/digital_behavior/) + Galaxy PPG
python deployment/ppg/receiver.py --port 5005

# 2) 병합: 이벤트 + PPG → datastream.jsonl
python deployment/merge.py --processed processed_*.csv --digital events_*.jsonl --out datastream.jsonl

# 3) 추론: datastream → V/A/C
python deployment/runtime_pipeline/run_target_routed_deploy.py \
  --in datastream.jsonl --out final.jsonl --ckpt full.pt
```

개인화까지 (선택, 첫 K개 자기보고로 보정):

```bash
python deployment/runtime_pipeline/run_target_routed_deploy.py \
  --in datastream.jsonl --out final.jsonl --ckpt full.pt \
  --calibration deployment/runtime_pipeline/eb_calibration_sample.json \
  --enrollment enroll.jsonl --enroll-k 4
```

- 개인화 없이(기본) = zero-shot. `--calibration`+`--enrollment` 주면 개인화.
- `enroll.jsonl` = `datastream.jsonl`과 같은 포맷 + 각 줄에 `label:{valence,arousal,cognitive_load}` + 같은 `user_id`.

---

## 학습: 모델 재학습 (본인 데이터 필요)

```bash
python training/target_routed/run_target_routed_final.py all 0
```

- 학습 데이터(CASE/EEVR/MAUS)는 비공개다. 스크립트/플랜의 데이터 경로를 본인 데이터로 지정해야 실행된다.
- 코드만으로 구조·방법은 그대로 재현된다.

---

## 결과

`results/FINAL_RESULTS.md` — source(PPG-only/B0/B1/B2) + external zero-shot + 개인화 lift 통합 표.

---

## 참고

- PaPaGEI: <https://github.com/Nokia-Bell-Labs/papagei-foundation-model> (frozen, 미세조정 안 함)
- 경로 override 필요 시 환경변수: `PAPAGEI_ROOT`, `PAPAGEI_CKPT`.
- 라이선스: `LICENSE`.
