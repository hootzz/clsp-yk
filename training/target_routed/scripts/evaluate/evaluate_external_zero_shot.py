"""Frozen external zero-shot evaluation for the target-routed candidate.

WESAD supplies V/A, CogWear supplies objective C, EmoWear supplies V/A, and
VRFS remains a condition-level V/A diagnostic.  No external label updates a
checkpoint, threshold, fusion weight, or architecture.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
CASE_CODE = MODEL_DIR / "dataset_ablation_pipeline" / "case_window_level"
PERSONALIZATION_DIR = MODEL_DIR / "personalization_v2"
for path in (MODEL_DIR, CASE_CODE, PERSONALIZATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_external_emowear_cogwear import (  # noqa: E402
    _audit_runs,
    _incremental,
    _load_model,
    _load_runs,
    _metrics,
    _predict,
    _summary,
    evaluate as evaluate_emowear,
    load_plan,
    resolve,
)
from evaluate_external_wesad_cogwear import (  # noqa: E402
    evaluate as evaluate_wesad,
)
from run_external_zeroshot_v2 import (  # noqa: E402
    _participant_metrics,
    _vrfs_evaluation_frame,
    _vrfs_source,
)


MODEL_VERSION = "target_routed_title_final"
BASE_CONDITION = "title_context_only_fixed_base"
FINAL_CONDITION = "title_context_ac_direct_ppg"
CONDITIONS = (BASE_CONDITION, FINAL_CONDITION)
SEEDS = (42, 43, 44)
WORK_ROOT = HERE / "work_model"
OUT = HERE / "reports" / "external_zero_shot"
PLAN_DIR = HERE / "plan" / "external_zero_shot"
SELECTION_PATH = PLAN_DIR / "target_routed_selection.json"
WESAD_PLAN_PATH = PLAN_DIR / "target_routed_wesad_cogwear.yaml"
EMOWEAR_PLAN_PATH = PLAN_DIR / "target_routed_emowear_cogwear.yaml"
BASE_WESAD_PLAN = CASE_CODE / "external_wesad_cogwear_plan.yaml"
BASE_EMOWEAR_PLAN = CASE_CODE / "external_enrollment_plan.yaml"
CURRENT_WORK = (
    MODEL_DIR / "dataset_ablation_pipeline" / "work_case_window_rescue_v3"
)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _yaml(path: Path, payload: dict[str, Any]) -> None:
    clean = {
        key: value for key, value in payload.items()
        if not str(key).startswith("_")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(clean, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _selection() -> None:
    runs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            metadata_path = (
                WORK_ROOT / "runs" / "participant_holdout" / condition
                / f"seed_{seed}" / "run_metadata.json"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Run full target-routed training first: {metadata_path}"
                )
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            expected_mode = (
                "concat"
                if condition == "title_context_ppg_uniform_direct"
                else "target_routed_direct"
            )
            if (
                metadata.get("status") != "complete"
                or metadata.get("device") != "cuda"
                or metadata.get("fusion_mode") != expected_mode
            ):
                raise RuntimeError(
                    f"Invalid target-routed checkpoint: {metadata_path}"
                )
            runs.append(
                {
                    "protocol": "participant_holdout",
                    "fold_video": None,
                    "condition_id": condition,
                    "seed": seed,
                    "checkpoint": metadata["checkpoint"],
                }
            )
    _json(
        SELECTION_PATH,
        {
            "status": "complete",
            "selected_candidate_id": MODEL_VERSION,
            "selection_data": "source validation only",
            "external_metrics_used_for_selection": False,
            "runs": runs,
        },
    )


def _resolved_wesad_plan() -> dict[str, Any]:
    base = load_plan(BASE_WESAD_PLAN)
    plan = copy.deepcopy(base)
    plan["model_dir"] = str(MODEL_DIR.resolve())
    plan["selection_json"] = str(SELECTION_PATH.resolve())
    plan["output_dir"] = str((OUT / "wesad_cogwear").resolve())
    plan["cogwear_cache_output_dir"] = str(
        (CURRENT_WORK / "external_emowear_cogwear").resolve()
    )
    plan["conditions"] = list(CONDITIONS)
    for key in ("manifest", "ppg_csv", "unmasked_manifest"):
        plan["public_source"][key] = str(
            resolve(base, base["public_source"][key])
        )
    _yaml(WESAD_PLAN_PATH, plan)
    return load_plan(WESAD_PLAN_PATH)


def _resolved_emowear_plan() -> dict[str, Any]:
    base = load_plan(BASE_EMOWEAR_PLAN)
    plan = copy.deepcopy(base)
    plan["model_dir"] = str(MODEL_DIR.resolve())
    plan["selection_json"] = str(SELECTION_PATH.resolve())
    plan["output_dir"] = str((OUT / "emowear_cogwear").resolve())
    plan["conditions"] = list(CONDITIONS)
    for key in (
        "csv_root",
        "seated_manifest",
        "seated_ppg",
        "paired_manifest",
        "walk_ppg",
    ):
        plan["emowear"][key] = str(resolve(base, base["emowear"][key]))
    plan["cogwear"]["root"] = str(
        resolve(base, base["cogwear"]["root"])
    )
    _yaml(EMOWEAR_PLAN_PATH, plan)

    # The source audit and already-computed external feature caches are
    # model-independent.  Reuse them without rebuilding PaPaGEI embeddings.
    source = CURRENT_WORK / "external_emowear_cogwear" / "source_data"
    target = OUT / "emowear_cogwear" / "source_data"
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
    return load_plan(EMOWEAR_PLAN_PATH)


def _vrfs_predict(
    plan: dict[str, Any], *, force: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = OUT / "vrfs"
    prediction_path = output / "predictions.csv"
    audit_path = output / "VRFS_ZERO_SHOT_AUDIT.json"
    if prediction_path.is_file() and audit_path.is_file() and not force:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "pass":
            return pd.read_csv(prediction_path), audit

    # VRFS features and label policies are independent of the checkpoint.
    source, source_audit = _vrfs_source(False)
    frame = _vrfs_evaluation_frame(source)
    runs = _load_runs(plan)
    run_audit = _audit_runs(plan, runs)
    if run_audit["status"] != "pass":
        raise RuntimeError(f"VRFS training audit failed: {run_audit}")
    predictions = []
    device = torch.device("cuda")
    for metadata in runs:
        print(
            f"[VRFS] {metadata['condition_id']} seed={metadata['seed']}",
            flush=True,
        )
        model = _load_model(plan, metadata, device)
        predicted, _ = _predict(
            model,
            frame,
            metadata["condition"],
            condition_id=str(metadata["condition_id"]),
            seed=int(metadata["seed"]),
            device=device,
            batch_size=256,
        )
        predicted["model_version"] = MODEL_VERSION
        predictions.append(predicted)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result = pd.concat(predictions, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(prediction_path, index=False)
    audit = {
        "status": "pass",
        "evaluation": "frozen_condition_level_diagnostic",
        "model_updates": 0,
        "prediction_rows": int(len(result)),
        "source_audit": source_audit,
        "training_audit": run_audit,
        "individual_sam_claim_allowed": False,
        "llamac_included": False,
    }
    _json(audit_path, audit)
    return result, audit


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(
            lambda value: (
                "NA" if not np.isfinite(value) else f"{value:.4f}"
            )
        )
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _valence_invariant(predictions: pd.DataFrame) -> dict[str, Any]:
    selected = predictions[predictions["target"].eq("valence")]
    keys = [
        "dataset",
        "domain",
        "seed",
        "target",
        "sample_id",
        "participant_id",
    ]
    base = selected[selected["condition_id"].eq(BASE_CONDITION)]
    final = selected[selected["condition_id"].eq(FINAL_CONDITION)]
    paired = base.merge(
        final,
        on=keys,
        suffixes=("_base", "_final"),
        validate="one_to_one",
    )
    maximum = float(
        np.abs(
            paired["prediction_base"] - paired["prediction_final"]
        ).max()
    )
    status = "pass" if maximum <= 1e-7 else "fail"
    result = {
        "status": status,
        "rows": int(len(paired)),
        "max_abs_prediction_delta": maximum,
        "tolerance": 1e-7,
    }
    if status != "pass":
        raise RuntimeError(f"External Valence invariant failed: {result}")
    return result


def _participant_bootstrap(
    participant: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 20260806,
) -> pd.DataFrame:
    keys = ["seed", "dataset", "domain", "target", "participant_id"]
    base = participant[
        participant["condition_id"].eq(BASE_CONDITION)
    ]
    rng = np.random.default_rng(seed)
    rows = []
    candidate_conditions = [
        condition for condition in CONDITIONS if condition != BASE_CONDITION
    ]
    for candidate_condition in candidate_conditions:
        candidate = participant[
            participant["condition_id"].eq(candidate_condition)
        ]
        paired = base.merge(
            candidate,
            on=keys,
            suffixes=("_base", "_candidate"),
            validate="one_to_one",
        )
        for (dataset, domain, target), group in paired.groupby(
            ["dataset", "domain", "target"], sort=True
        ):
            for metric in (
                "ccc",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ):
                user = pd.DataFrame(
                    {
                        "participant_id": group["participant_id"],
                        "delta": (
                            group[f"{metric}_candidate"]
                            - group[f"{metric}_base"]
                        ),
                    }
                ).groupby("participant_id", as_index=False)["delta"].mean()
                values = user["delta"].to_numpy(float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                boot = np.asarray(
                    [
                        rng.choice(values, len(values), replace=True).mean()
                        for _ in range(iterations)
                    ]
                )
                rows.append(
                    {
                        "candidate_condition": candidate_condition,
                        "reference_condition": BASE_CONDITION,
                        "dataset": dataset,
                        "domain": domain,
                        "target": target,
                        "metric": metric,
                        "participants": int(len(values)),
                        "mean_delta": float(values.mean()),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def _combine(vrfs: pd.DataFrame, vrfs_audit: dict[str, Any]) -> None:
    contracts = (
        (
            OUT / "wesad_cogwear" / "reports"
            / "external_zero_shot_predictions.csv",
            {"wesad", "cogwear"},
        ),
        (
            OUT / "emowear_cogwear" / "reports"
            / "external_zero_shot_predictions.csv",
            {"emowear"},
        ),
    )
    frames = []
    for path, datasets in contracts:
        frame = pd.read_csv(path)
        frames.append(frame[frame["dataset"].isin(datasets)].copy())
    frames.append(vrfs.copy())
    predictions = pd.concat(frames, ignore_index=True, sort=False)
    predictions["model_version"] = MODEL_VERSION
    predictions = predictions[
        predictions["condition_id"].isin(CONDITIONS)
    ].copy()

    duplicate_keys = [
        "dataset",
        "domain",
        "condition_id",
        "seed",
        "target",
        "sample_id",
    ]
    duplicate_rows = int(predictions.duplicated(duplicate_keys).sum())
    if duplicate_rows:
        raise RuntimeError(f"Duplicate external predictions: {duplicate_rows}")
    invariant = _valence_invariant(predictions)
    per_seed = _metrics(predictions)
    summary = _summary(per_seed)
    participant = _participant_metrics(predictions)
    incremental = _incremental(per_seed)
    bootstrap = _participant_bootstrap(participant)

    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    per_seed.to_csv(OUT / "metrics_per_seed.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    participant.to_csv(OUT / "metrics_per_participant.csv", index=False)
    incremental.to_csv(OUT / "population_incremental.csv", index=False)
    bootstrap.to_csv(OUT / "participant_bootstrap.csv", index=False)

    final_metrics = summary[
        summary["condition_id"].eq(FINAL_CONDITION)
    ][
        [
            "dataset",
            "domain",
            "target",
            "participants",
            "ccc_mean",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
        ]
    ].sort_values(["dataset", "domain", "target"])
    delta = (
        incremental.groupby(
            [
                "multimodal_condition",
                "context_condition",
                "dataset",
                "domain",
                "target",
            ],
            as_index=False,
        )[
            [
                "delta_ccc",
                "delta_accuracy",
                "delta_balanced_accuracy",
                "delta_macro_f1",
            ]
        ].mean()
    )
    bootstrap_ccc = bootstrap[bootstrap["metric"].eq("ccc")]
    lines = [
        "# Target-routed frozen external zero-shot",
        "",
        "- WESAD: individual block-level V/A; CogWear: objective rest-vs-Stroop C; EmoWear: individual seated/walk V/A.",
        "- VRFS is condition-level only because physiological sessions cannot be joined to individual SAM rows.",
        "- No external enrollment labels, model updates, threshold selection, fusion selection, or architecture selection are used.",
        "- Valence uses the frozen Context-only route. Arousal/C use the residual-free target-specific direct route.",
        "- LLaMAC is excluded.",
        "",
        "## Final-policy actual metrics",
        "",
        _markdown(final_metrics),
        "",
        "## Multimodal minus matched Context-only",
        "",
        _markdown(delta),
        "",
        "## Participant-bootstrap CCC delta",
        "",
        _markdown(bootstrap_ccc),
        "",
        "## Valence route invariant",
        "",
        _markdown(pd.DataFrame([invariant])),
        "",
        "## Claim boundary",
        "",
        "These are frozen-checkpoint external diagnostics. EmoWear was inspected during earlier model development, and VRFS has only aggregate label policies; neither is a pristine confirmatory individual-affect test.",
    ]
    report = OUT / "TARGET_ROUTED_EXTERNAL_ZERO_SHOT_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(
        OUT / "TARGET_ROUTED_EXTERNAL_ZERO_SHOT_AUDIT.json",
        {
            "schema_version": 1,
            "status": "pass",
            "evaluation": "frozen_external_zero_shot",
            "model_updates": 0,
            "personalization_labels_used": 0,
            "external_metrics_used_for_selection": False,
            "conditions": list(CONDITIONS),
            "seeds": list(SEEDS),
            "datasets": sorted(predictions["dataset"].unique().tolist()),
            "prediction_rows": int(len(predictions)),
            "duplicate_prediction_keys": duplicate_rows,
            "frozen_valence_invariant": invariant,
            "vrfs": vrfs_audit,
            "llamac_included": False,
            "report": str(report.resolve()),
        },
    )


def evaluate(*, force: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("External zero-shot is GPU-only")
    _selection()
    wesad_plan = _resolved_wesad_plan()
    emowear_plan = _resolved_emowear_plan()
    evaluate_wesad(wesad_plan, force=force)
    evaluate_emowear(emowear_plan, force=force)
    vrfs, vrfs_audit = _vrfs_predict(emowear_plan, force=force)
    _combine(vrfs, vrfs_audit)
    print(
        f"[DONE] external zero-shot -> "
        f"{OUT / 'TARGET_ROUTED_EXTERNAL_ZERO_SHOT_RESULTS.md'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate(force=args.force)


if __name__ == "__main__":
    main()
