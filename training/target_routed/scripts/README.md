# scripts/ — Research Execution Scripts

This directory contains the orchestration scripts used to train and evaluate the target-routed model.

- `run/` — training entry points: `run_target_routed_final.py` (B0/B1/B2 training and evaluation), `run_b1_uniform_followup.py`
- `evaluate/` — evaluation scripts: `evaluate_target_routed_final.py`, `evaluate_b0_b1_b2*.py`, `evaluate_external_zero_shot.py`
- `engine/` — data builders and training/evaluation utilities: `train_case_window_experiments.py`,
  `evaluate_case_window_experiments.py`, `build_case_window_source.py`,
  `public_context_mapping.py`, `build_context7_gpt.py`, `case_window_common.py`,
  `evaluate_external_*.py`

## Notes

- These scripts depend on repository-specific paths and source datasets that are not included here.
- The files were originally organized in a flat directory. If reproducing the original execution setup, ensure that `run/`, `evaluate/`, and `engine/` are available on the same Python import path.

Reusable model definitions and training components are provided in `../../core/`.
