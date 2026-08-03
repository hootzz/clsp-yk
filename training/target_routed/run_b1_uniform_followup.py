"""GPU follow-up that adds B1 after the existing B0/B2 command finishes.

B0 and B2 checkpoints are reused without retraining.  This script trains only
the historical uniform direct concat condition B1, then produces the complete
B0/B1/B2 source-heldout, EB, participant, and external-zero-shot comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
CASE_CODE = MODEL_DIR / "dataset_ablation_pipeline" / "case_window_level"
for path in (HERE, CASE_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_target_routed_final as base_runner  # noqa: E402
from case_window_common import json_dump, sha256_file  # noqa: E402
from evaluate_case_window_experiments import (  # noqa: E402
    evaluate_all,
    evaluate_run,
)
from train_case_window_experiments import train_one  # noqa: E402


B1_CONDITION = "title_context_ppg_uniform_direct"
ALL_CONDITIONS = (
    base_runner.BASE_CONDITION,
    B1_CONDITION,
    base_runner.FINAL_CONDITION,
)
SEEDS = base_runner.SEEDS
WORK_ROOT = base_runner.WORK_ROOT
PLAN_PATH = HERE / "plan" / "b0_b1_b2_title_nsegment.yaml"
REPORTS = HERE / "reports" / "b0_b1_b2"


def _require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("B1 follow-up is GPU-only; CPU fallback is disabled")
    index = torch.cuda.current_device()
    audit = {
        "status": "pass",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_index": int(index),
        "device_name": torch.cuda.get_device_name(index),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_dump(REPORTS / "GPU_AUDIT.json", audit)
    print(
        f"[GPU] {audit['device_name']} | torch={audit['torch']} | "
        f"CUDA={audit['cuda_runtime']}",
        flush=True,
    )
    return audit


def _materialize_plan() -> dict[str, Any]:
    plan = copy.deepcopy(base_runner._materialize_plan())
    b1 = copy.deepcopy(plan["conditions"][base_runner.FINAL_CONDITION])
    b1.update(
        {
            "fusion_mode": "concat",
            "architecture_role": "B1_uniform_direct_fusion",
            "target_routes": {
                "valence": "direct_context_ppg",
                "arousal": "direct_context_ppg",
                "cognitive_load": "direct_context_ppg",
            },
        }
    )
    plan["conditions"] = {
        base_runner.BASE_CONDITION: plan["conditions"][
            base_runner.BASE_CONDITION
        ],
        B1_CONDITION: b1,
        base_runner.FINAL_CONDITION: plan["conditions"][
            base_runner.FINAL_CONDITION
        ],
    }
    plan["b0_b1_b2_contract"] = {
        "status": "predefined_followup_control",
        "B0": "title Context-only for V/A/C",
        "B1": "historical uniform concat Context+PPG for V/A/C",
        "B2": "frozen Context-only V; target-specific direct Context+PPG A/C",
        "matched_factors": [
            "source rows",
            "participant split",
            "seeds",
            "title context",
            "normalization",
            "training budget",
        ],
        "test_metrics_used_for_architecture": False,
    }
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return plan


def _run_dir(condition: str, seed: int, *, smoke: bool = False) -> Path:
    return (
        WORK_ROOT / ("smoke" if smoke else "runs")
        / "participant_holdout" / condition / f"seed_{seed}"
    )


def _metadata(condition: str, seed: int, *, smoke: bool = False) -> Path:
    return _run_dir(condition, seed, smoke=smoke) / "run_metadata.json"


def _require_existing(conditions: tuple[str, ...]) -> None:
    missing = []
    for condition in conditions:
        for seed in SEEDS:
            path = _metadata(condition, seed)
            if not path.is_file():
                missing.append(str(path))
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                missing.append(f"{path} (not complete)")
    if missing:
        raise RuntimeError(
            "Run the current B0/B2 CMD to completion first. Missing: "
            + "; ".join(missing)
        )


def _train_b1(plan: dict[str, Any], *, force: bool, smoke: bool) -> None:
    if not smoke:
        _require_existing(
            (base_runner.BASE_CONDITION, base_runner.FINAL_CONDITION)
        )
    seeds = (42,) if smoke else SEEDS
    for seed in seeds:
        train_one(
            plan,
            "participant_holdout",
            B1_CONDITION,
            seed,
            None,
            force=force,
            smoke=smoke,
            case_loss_weight=1.0,
            run_group=None,
            ppg_backbone_checkpoint=base_runner._ppg_checkpoint(seed),
        )
        if smoke:
            _, frame = evaluate_run(
                plan,
                _metadata(B1_CONDITION, seed, smoke=True),
                force=True,
                split_name="validation",
            )
            if frame.empty:
                raise RuntimeError("B1 smoke produced no validation rows")
    if smoke:
        json_dump(
            REPORTS / "B1_SMOKE_AUDIT.json",
            {
                "status": "pass",
                "condition_id": B1_CONDITION,
                "fusion_mode": "concat",
                "seed": 42,
                "epochs": 1,
                "note": "Execution check only; not a final result.",
            },
        )
        print("[SMOKE PASS] B1 uniform direct concat", flush=True)


def _evaluate(plan: dict[str, Any], *, force: bool) -> None:
    _require_existing(ALL_CONDITIONS)
    evaluate_all(plan, protocols=["participant_holdout"], force=force)
    for condition in ALL_CONDITIONS:
        for seed in SEEDS:
            evaluate_run(
                plan,
                _metadata(condition, seed),
                force=force,
                split_name="validation",
            )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "evaluate_b0_b1_b2.py"),
            "--root",
            str(WORK_ROOT),
            "--source-root",
            str(base_runner.SOURCE_ROOT),
            "--reports",
            str(REPORTS),
        ],
        check=True,
    )


def _external(*, force: bool) -> None:
    _require_existing(ALL_CONDITIONS)
    command = [
        sys.executable,
        str(HERE / "evaluate_b0_b1_b2_external.py"),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def _write_contract(gpu: dict[str, Any]) -> None:
    files = [
        MODEL_DIR / "model.py",
        MODEL_DIR / "train.py",
        CASE_CODE / "train_case_window_experiments.py",
        CASE_CODE / "evaluate_external_emowear_cogwear.py",
        HERE / "run_b1_uniform_followup.py",
        HERE / "evaluate_b0_b1_b2.py",
        HERE / "evaluate_b0_b1_b2_external.py",
    ]
    json_dump(
        REPORTS / "B0_B1_B2_EXPERIMENT_CONTRACT.json",
        {
            "schema_version": 1,
            "status": "b1_followup_ready",
            "conditions": list(ALL_CONDITIONS),
            "seeds": list(SEEDS),
            "work_root": str(WORK_ROOT.resolve()),
            "plan": str(PLAN_PATH.resolve()),
            "gpu": gpu,
            "code_sha256": {
                str(path.resolve()): sha256_file(path)
                for path in files if path.is_file()
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "train", "evaluate", "external", "all"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gpu = _require_cuda()
    plan = _materialize_plan()
    _write_contract(gpu)
    if args.stage == "smoke":
        _train_b1(plan, force=True, smoke=True)
    if args.stage in {"train", "all"}:
        _train_b1(plan, force=args.force, smoke=False)
    if args.stage in {"evaluate", "all"}:
        _evaluate(plan, force=args.force)
    if args.stage in {"external", "all"}:
        _external(force=args.force)
    print(f"[DONE] B1 follow-up stage={args.stage}", flush=True)


if __name__ == "__main__":
    main()
