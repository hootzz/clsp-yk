# scripts/ — 원 연구 실행 스크립트 (참고용)

target-routed 모델을 학습·평가한 **원 연구 오케스트레이션** 코드다. 재현·감사용 참고.

- 실행 진입: `run_target_routed_final.py` (B0/B1/B2 학습+평가), `run_b1_uniform_followup.py`.
- 평가: `evaluate_target_routed_final.py`, `evaluate_b0_b1_b2*.py`, `evaluate_external_zero_shot.py`.
- 엔진/빌더: `train_case_window_experiments.py`, `evaluate_case_window_experiments.py`,
  `build_case_window_source.py`, `public_context_mapping.py`, `build_context7_gpt.py`,
  `case_window_common.py`, `evaluate_external_*.py`.

⚠️ **이 폴더만으론 그대로 실행되지 않는다.** 스크립트 상단이 원 연구 모노repo 경로
(`dataset_ablation_pipeline/…`, `personalization_v2/work_models/…`)와 비공개 원본 데이터
(CASE/EEVR/MAUS)를 가정한다. 재학습하려면 그 데이터·경로를 갖춰야 한다.

실제 재사용 코드(모델 정의·학습 방법)는 상위 `../../core/`에 있다.
