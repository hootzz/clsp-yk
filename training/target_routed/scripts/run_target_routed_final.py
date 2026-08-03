"""GPU runner for the target-routed title-context final candidate.

The population architecture is fixed by target:

* Valence: frozen title-level Context-only predictor.
* Arousal: residual-free direct Context+PPG predictor.
* Cognitive load: residual-free direct Context+PPG predictor.

A matched Context-only base is trained first for every seed.  The multimodal
candidate then imports that complete checkpoint, imports only the common PPG
projection from the seed-matched PPG-only checkpoint, and freezes the complete
text branch plus the Valence trunk/head.  This makes Valence invariant to PPG
both in the forward graph and through training updates.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
CASE_CODE = MODEL_DIR / "dataset_ablation_pipeline" / "case_window_level"
PERSONALIZATION_CODE = MODEL_DIR / "personalization_v2"
for code_dir in (CASE_CODE, PERSONALIZATION_CODE):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from case_window_common import json_dump, load_plan, sha256_file  # noqa: E402
from evaluate_case_window_experiments import (  # noqa: E402
    evaluate_all,
    evaluate_run,
)
from train_case_window_experiments import train_one  # noqa: E402


SOURCE_ROOT = (
    MODEL_DIR / "personalization_v2" / "work_models" / "nsegment_lag4_p10s"
)
SOURCE_PLAN = (
    MODEL_DIR
    / "personalization_v2"
    / "plans"
    / "case_nsegment_lag4_p10s.yaml"
)
WORK_ROOT = HERE / "work_model"
PLAN_PATH = HERE / "plan" / "target_routed_title_nsegment.yaml"
REPORTS = HERE / "reports"
SEEDS = (42, 43, 44)
BASE_CONDITION = "title_context_only_fixed_base"
FINAL_CONDITION = "title_context_ac_direct_ppg"
CONDITIONS = (BASE_CONDITION, FINAL_CONDITION)
FREEZE_PREFIXES = (
    "text_branch",
    "target_context_trunks.valence",
    "target_routed_heads.valence",
)


def _require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; CPU fallback is intentionally disabled."
        )
    index = torch.cuda.current_device()
    audit = {
        "status": "pass",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_index": int(index),
        "device_name": torch.cuda.get_device_name(index),
        "device_capability": list(torch.cuda.get_device_capability(index)),
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
    if not SOURCE_PLAN.is_file():
        raise FileNotFoundError(SOURCE_PLAN)
    source_conditions = load_plan(SOURCE_PLAN)["conditions"]
    plan = copy.deepcopy(load_plan(SOURCE_PLAN))
    plan["work_dir"] = str(WORK_ROOT.resolve())
    plan["case_source_dir"] = str((SOURCE_ROOT / "source_data").resolve())
    plan["training"]["fusion_mode"] = "target_routed_direct"
    plan["conditions"] = {
        BASE_CONDITION: {
            **copy.deepcopy(source_conditions["context_content_only"]),
            "context_contract": "title_scene_level_operational_content",
            "target_routes": {
                "valence": "context_only",
                "arousal": "context_only_matched_control",
                "cognitive_load": "context_only_matched_control",
            },
        },
        FINAL_CONDITION: {
            **copy.deepcopy(source_conditions["context_content_ppg"]),
            "context_contract": "title_scene_level_operational_content",
            "target_routes": {
                "valence": "frozen_context_only",
                "arousal": "direct_context_ppg",
                "cognitive_load": "direct_context_ppg",
            },
        },
    }
    plan["target_routed_final_contract"] = {
        "schema_version": 1,
        "status": "new_hypothesis_not_existing_result",
        "normalization": "participant_window",
        "title_level_context": {
            "case": "official content title",
            "eevr": "official scene/content identity",
            "maus": "n-back level",
        },
        "population_predictors": {
            "valence": "frozen context-only",
            "arousal": "residual-free direct context+PPG",
            "cognitive_load": "residual-free direct context+PPG",
        },
        "inference_personalization": (
            "target-wise validation-estimated empirical-Bayes random intercept"
        ),
        "freeze_parameter_prefixes": list(FREEZE_PREFIXES),
        "test_metrics_used_for_architecture": False,
    }
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return plan


def _run_dir(condition: str, seed: int, *, smoke: bool) -> Path:
    return (
        WORK_ROOT
        / ("smoke" if smoke else "runs")
        / "participant_holdout"
        / condition
        / f"seed_{seed}"
    )


def _metadata(condition: str, seed: int, *, smoke: bool) -> Path:
    return _run_dir(condition, seed, smoke=smoke) / "run_metadata.json"


def _ppg_checkpoint(seed: int) -> Path:
    path = (
        SOURCE_ROOT
        / "runs"
        / "participant_holdout"
        / "ppg_only"
        / f"seed_{seed}"
        / "model_v3"
        / "full.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _train(*, force: bool, smoke: bool) -> None:
    plan = _materialize_plan()
    seeds = (42,) if smoke else SEEDS
    for seed in seeds:
        base = train_one(
            plan,
            "participant_holdout",
            BASE_CONDITION,
            seed,
            None,
            force=force,
            smoke=smoke,
            case_loss_weight=1.0,
            run_group=None,
        )
        train_one(
            plan,
            "participant_holdout",
            FINAL_CONDITION,
            seed,
            None,
            force=force,
            smoke=smoke,
            case_loss_weight=1.0,
            run_group=None,
            ppg_backbone_checkpoint=_ppg_checkpoint(seed),
            full_init_checkpoint=Path(base["checkpoint"]),
            freeze_parameter_prefixes=FREEZE_PREFIXES,
        )
    if smoke:
        _evaluate_smoke(plan)


def _evaluate_smoke(plan: dict[str, Any]) -> None:
    predictions = {}
    for condition in CONDITIONS:
        _, frame = evaluate_run(
            plan,
            _metadata(condition, 42, smoke=True),
            force=True,
            split_name="validation",
        )
        predictions[condition] = frame
    maximum = _assert_valence_invariant(
        predictions[BASE_CONDITION], predictions[FINAL_CONDITION]
    )
    context_counts = []
    for condition in CONDITIONS:
        manifest = pd.read_csv(
            _run_dir(condition, 42, smoke=True) / "training_manifest.csv"
        )
        for dataset, group in manifest.groupby("dataset", sort=True):
            context_counts.append(
                {
                    "condition_id": condition,
                    "dataset": str(dataset).lower(),
                    "unique_contexts": int(
                        group["text"].astype(str).nunique()
                    ),
                }
            )
    json_dump(
        REPORTS / "SMOKE_AUDIT.json",
        {
            "schema_version": 1,
            "status": "pass",
            "seed": 42,
            "epochs": 1,
            "split": "validation",
            "frozen_valence_max_abs_prediction_delta": maximum,
            "frozen_valence_tolerance": 1e-7,
            "title_context_counts": context_counts,
            "note": "Smoke metrics are execution checks, not final results.",
        },
    )
    print("[SMOKE PASS] target routing and frozen Valence invariant", flush=True)


def _evaluate_runs(plan: dict[str, Any], *, force: bool) -> None:
    evaluate_all(plan, protocols=["participant_holdout"], force=force)
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            _, frame = evaluate_run(
                plan,
                _metadata(condition, seed, smoke=False),
                force=force,
                split_name="validation",
            )
            frames[(condition, seed)] = frame
    for seed in SEEDS:
        base_test = pd.read_csv(
            _run_dir(BASE_CONDITION, seed, smoke=False)
            / "evaluation"
            / "predictions.csv"
        )
        final_test = pd.read_csv(
            _run_dir(FINAL_CONDITION, seed, smoke=False)
            / "evaluation"
            / "predictions.csv"
        )
        _assert_valence_invariant(base_test, final_test)
        _assert_valence_invariant(
            frames[(BASE_CONDITION, seed)],
            frames[(FINAL_CONDITION, seed)],
        )


def _assert_valence_invariant(base: pd.DataFrame, final: pd.DataFrame) -> float:
    keys = ["sample_id", "dataset", "participant_id", "target"]
    left = base[base["target"].eq("valence")]
    right = final[final["target"].eq("valence")]
    paired = left.merge(
        right,
        on=keys,
        suffixes=("_base", "_final"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise AssertionError("Valence invariant row alignment failed")
    maximum = float(
        (paired["prediction_base"] - paired["prediction_final"])
        .abs()
        .max()
    )
    if maximum > 1e-7:
        raise AssertionError(
            f"Valence changed after A/C PPG training: max_abs={maximum}"
        )
    return maximum


def _run_analysis() -> None:
    subprocess.run(
        [
            sys.executable,
            str(HERE / "evaluate_target_routed_final.py"),
            "--root",
            str(WORK_ROOT),
            "--source-root",
            str(SOURCE_ROOT),
            "--reports",
            str(REPORTS),
        ],
        check=True,
    )


def _run_external(*, force: bool) -> None:
    command = [
        sys.executable,
        str(HERE / "evaluate_external_zero_shot.py"),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def _write_contract(gpu: dict[str, Any]) -> None:
    code_files = [
        MODEL_DIR / "model.py",
        MODEL_DIR / "train.py",
        CASE_CODE / "train_case_window_experiments.py",
        CASE_CODE / "evaluate_case_window_experiments.py",
        PERSONALIZATION_CODE / "evaluate_personalization.py",
        HERE / "run_target_routed_final.py",
        HERE / "evaluate_target_routed_final.py",
        HERE / "evaluate_external_zero_shot.py",
    ]
    json_dump(
        REPORTS / "EXPERIMENT_CONTRACT.json",
        {
            "schema_version": 1,
            "status": "target_routed_title_candidate",
            "source_root": str(SOURCE_ROOT.resolve()),
            "plan": str(PLAN_PATH.resolve()),
            "conditions": list(CONDITIONS),
            "seeds": list(SEEDS),
            "freeze_parameter_prefixes": list(FREEZE_PREFIXES),
            "gpu": gpu,
            "code_sha256": {
                str(path.resolve()): sha256_file(path)
                for path in code_files
                if path.is_file()
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
    _materialize_plan()
    _write_contract(gpu)
    if args.stage == "smoke":
        _train(force=True, smoke=True)
    if args.stage in {"train", "all"}:
        _train(force=args.force, smoke=False)
    if args.stage in {"evaluate", "all"}:
        plan = _materialize_plan()
        _evaluate_runs(plan, force=args.force)
        _run_analysis()
    if args.stage in {"external", "all"}:
        _run_external(force=args.force)
    print(f"[DONE] target-routed final stage={args.stage}", flush=True)


if __name__ == "__main__":
    main()
