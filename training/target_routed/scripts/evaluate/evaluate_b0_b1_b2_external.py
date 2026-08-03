"""Run the three-condition external zero-shot follow-up."""
from __future__ import annotations

import argparse
from pathlib import Path

import evaluate_external_zero_shot as external


HERE = Path(__file__).resolve().parent
B1 = "title_context_ppg_uniform_direct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Reuse the audited external evaluator with an isolated three-condition
    # output/plan namespace.  B0/B2 checkpoints are read-only; only the B1
    # checkpoint set is newly added by the follow-up runner.
    external.MODEL_VERSION = "target_routed_title_b0_b1_b2"
    external.CONDITIONS = (
        external.BASE_CONDITION,
        B1,
        external.FINAL_CONDITION,
    )
    external.OUT = HERE / "reports" / "b0_b1_b2" / "external_zero_shot"
    external.PLAN_DIR = HERE / "plan" / "b0_b1_b2_external_zero_shot"
    external.SELECTION_PATH = (
        external.PLAN_DIR / "b0_b1_b2_selection.json"
    )
    external.WESAD_PLAN_PATH = (
        external.PLAN_DIR / "b0_b1_b2_wesad_cogwear.yaml"
    )
    external.EMOWEAR_PLAN_PATH = (
        external.PLAN_DIR / "b0_b1_b2_emowear_cogwear.yaml"
    )
    external.evaluate(force=args.force)


if __name__ == "__main__":
    main()
