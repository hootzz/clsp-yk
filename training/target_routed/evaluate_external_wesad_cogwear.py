"""Frozen state-head zero-shot evaluation on WESAD and CogWear.

This is the modality-comparison companion to the existing
``evaluate_wesad_prompt_zeroshot.py`` notebook-matched diagnostic.  It applies
the already-selected Context-only, PPG-only, Context+PPG, and
Generic-context+PPG checkpoints without updating a model, threshold, context
mapping, or fusion weight from WESAD/CogWear labels.

WESAD contributes normalized block-level SAM valence/arousal.  CogWear
contributes binary objective cognitive effort (rest versus Stroop).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from case_window_common import (
    configure_console,
    json_dump,
    markdown_table,
    sha256_file,
)
from evaluate_external_emowear_cogwear import (
    PPG_COLUMNS,
    _audit_runs,
    _cogwear_rows,
    _incremental,
    _load_model,
    _load_runs,
    _metrics,
    _paired_deltas,
    _predict,
    _spec_text,
    _summary,
    load_plan,
    resolve,
)
from public_context_mapping import adapt_public_manifest, audit_original7


HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = HERE / "external_wesad_cogwear_plan.yaml"
CONDITIONS = ("baseline", "stress", "amusement", "meditation")


def output_dir(plan: dict[str, Any]) -> Path:
    return resolve(plan, plan["output_dir"])


def _condition_from_sample_id(sample_id: str) -> str:
    for condition in CONDITIONS:
        if str(sample_id).endswith(f"_{condition}"):
            return condition
    raise ValueError(f"Cannot infer WESAD condition: {sample_id}")


def _wesad_rows(plan: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    active = pd.read_csv(resolve(plan, plan["public_source"]["manifest"]))
    active = active[
        active["dataset"].astype(str).str.lower().eq("wesad")
    ].copy()
    active = adapt_public_manifest(active)
    schema_audit = audit_original7(active)
    if schema_audit["status"] != "pass":
        raise ValueError(f"WESAD original-7 audit failed: {schema_audit}")

    unmasked_path = resolve(
        plan, plan["public_source"]["unmasked_manifest"]
    )
    unmasked = pd.read_csv(unmasked_path)
    labels = unmasked[
        unmasked["dataset"].astype(str).str.lower().eq("wesad")
    ][["sample_id", "valence", "arousal"]].copy()
    features = pd.read_csv(
        resolve(plan, plan["public_source"]["ppg_csv"])
    )
    features = features[
        features["sample_id"].astype(str).isin(
            set(active["sample_id"].astype(str))
        )
    ][["sample_id", *PPG_COLUMNS]].copy()

    for name, frame in (
        ("active", active),
        ("labels", labels),
        ("features", features),
    ):
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate WESAD sample_id in {name}")

    merged = (
        active.drop(columns=["valence", "arousal"], errors="ignore")
        .merge(labels, on="sample_id", how="inner", validate="one_to_one")
        .merge(features, on="sample_id", how="inner", validate="one_to_one")
    )
    if len(merged) != 60 or merged["participant_id"].nunique() != 15:
        raise ValueError(
            "Expected 60 WESAD participant-condition blocks from 15 "
            f"participants, got {len(merged)}/"
            f"{merged['participant_id'].nunique()}"
        )
    feature_values = merged[PPG_COLUMNS].to_numpy(dtype=np.float32)
    label_values = merged[["valence", "arousal"]].to_numpy(dtype=float)
    if (
        feature_values.shape != (60, 512)
        or not np.isfinite(feature_values).all()
        or not np.isfinite(label_values).all()
        or (label_values < 0.0).any()
        or (label_values > 1.0).any()
    ):
        raise ValueError("Invalid WESAD 512D features or normalized SAM")

    # One constant neutral text is used only by the pre-existing
    # Generic-context+PPG diagnostic. Operational conditions always receive
    # the protocol-derived original-7 text.
    renderer, _, validator = __import__(
        "evaluate_external_emowear_cogwear",
        fromlist=["_context_helpers"],
    )._context_helpers(plan)
    generic_text = _spec_text(
        renderer,
        validator,
        posture="not_recorded",
        energy="not_recorded",
        app_window="not_recorded",
        duration="not_recorded",
        event_type="other",
    )
    condition_order = {
        condition: index for index, condition in enumerate(CONDITIONS)
    }
    rows: list[dict[str, Any]] = []
    for source in merged.itertuples(index=False):
        condition = _condition_from_sample_id(str(source.sample_id))
        rows.append(
            {
                "sample_id": str(source.sample_id),
                "dataset": "wesad",
                "participant_id": str(source.participant_id),
                "external_unit_id": str(source.sample_id),
                "cohort": "public_external",
                "session": "participant_condition_block",
                "sequence": int(condition_order[condition]),
                "device": "e4",
                "domain": "e4",
                "phase": condition,
                "operational_text": str(source.context_text),
                "generic_text": generic_text,
                "valence": float(source.valence),
                "arousal": float(source.arousal),
                "cognitive_load": np.nan,
                "binary_threshold": float(
                    plan["evaluation"][
                        "wesad_binary_threshold_normalized"
                    ]
                ),
                "ppg_confidence": 1.0,
                **{
                    column: float(getattr(source, column))
                    for column in PPG_COLUMNS
                },
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["participant_id", "sequence"]
    ).reset_index(drop=True)
    audit = {
        "status": "pass",
        "rows": int(len(frame)),
        "participants": int(frame["participant_id"].nunique()),
        "conditions": sorted(frame["phase"].unique().tolist()),
        "operational_contexts": int(
            frame["operational_text"].nunique()
        ),
        "generic_contexts": int(frame["generic_text"].nunique()),
        "valence_range": [
            float(frame["valence"].min()),
            float(frame["valence"].max()),
        ],
        "arousal_range": [
            float(frame["arousal"].min()),
            float(frame["arousal"].max()),
        ],
        "label_source": str(unmasked_path),
        "label_source_sha256": sha256_file(unmasked_path),
        "label_use": "final scoring only",
        "schema_audit": schema_audit,
    }
    return frame, audit


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    device_deltas: pd.DataFrame,
    incremental: pd.DataFrame,
) -> None:
    display_columns = [
        "dataset",
        "domain",
        "condition_id",
        "target",
        "n_per_seed",
        "participants",
        "ccc_mean",
        "ccc_std",
        "participant_macro_ccc_mean",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
    ]
    delta_summary = (
        device_deltas.groupby(
            ["comparison", "condition_id", "target"], as_index=False
        )
        .mean(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    incremental_group_columns = [
        column
        for column in (
            "multimodal_condition",
            "context_condition",
            "dataset",
            "domain",
            "target",
        )
        if column in incremental.columns
    ]
    incremental_summary = (
        incremental.groupby(
            incremental_group_columns, as_index=False
        )
        .mean(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    lines = [
        "# WESAD + CogWear frozen external zero-shot",
        "",
        "- Selected source checkpoints: CASE + EEVR + MAUS only.",
        "- WESAD and CogWear labels are used for final scoring only.",
        "- No external fine-tuning, threshold tuning, checkpoint selection,",
        "  or fusion-weight selection is performed.",
        "- WESAD uses one E4 block mean per participant-condition and its",
        "  normalized block-level SAM V/A.",
        "- WESAD binary reporting uses the fixed normalized threshold 0.625,",
        "  corresponding to 1-3 versus 4-5 on the linearly mapped scale.",
        "- CogWear is objective cognitive effort (rest versus Stroop), not",
        "  continuous subjective cognitive-load ground truth.",
        "- Cell-level binary metrics are macro-F1, accuracy, and balanced",
        "  accuracy; CCC is retained for continuous-score diagnostics.",
        "",
        "## Seed mean and variability",
        "",
        markdown_table(summary[display_columns]),
        "",
        "## CogWear device delta",
        "",
        (
            markdown_table(delta_summary)
            if len(delta_summary)
            else "- No paired device delta."
        ),
        "",
        "## PPG incremental contribution",
        "",
        "Multimodal minus matched Context-only on identical rows:",
        "",
        markdown_table(incremental_summary),
        "",
        "## Claim boundary",
        "",
        "WESAD was inspected during historical development. This is a frozen",
        "current-checkpoint external diagnostic, not a pristine never-observed",
        "confirmatory test. The existing notebook-matched prompt-similarity",
        "evaluation remains a separate supplementary analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(plan: dict[str, Any], *, force: bool) -> None:
    reports = output_dir(plan) / "reports"
    predictions_path = reports / "external_zero_shot_predictions.csv"
    per_seed_path = reports / "external_metrics_per_seed.csv"
    summary_path = reports / "external_metrics_summary.csv"
    incremental_path = reports / "external_ppg_incremental_metrics.csv"
    device_delta_path = reports / "external_device_deltas.csv"
    report_path = reports / "EXTERNAL_WESAD_COGWEAR_RESULTS.md"
    audit_path = reports / "external_zero_shot_audit.json"
    required = [
        predictions_path,
        per_seed_path,
        summary_path,
        incremental_path,
        device_delta_path,
        report_path,
        audit_path,
    ]
    if all(path.is_file() for path in required) and not force:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "pass":
            print(f"[SKIP] passing WESAD/CogWear outputs: {report_path}")
            return

    wesad, wesad_audit = _wesad_rows(plan)
    cache_plan = dict(plan)
    cache_plan["output_dir"] = str(
        resolve(plan, plan["cogwear_cache_output_dir"])
    )
    cogwear = _cogwear_rows(cache_plan)
    evaluation_frame = pd.concat([wesad, cogwear], ignore_index=True)

    runs = _load_runs(plan)
    run_audit = _audit_runs(plan, runs)
    if run_audit["status"] != "pass":
        raise RuntimeError(
            f"External training leakage audit failed: {run_audit}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    for metadata in runs:
        condition_id = str(metadata["condition_id"])
        seed = int(metadata["seed"])
        print(f"[WESAD/COGWEAR] {condition_id} seed={seed}", flush=True)
        model = _load_model(plan, metadata, device)
        predicted, _ = _predict(
            model,
            evaluation_frame,
            metadata["condition"],
            condition_id=condition_id,
            seed=seed,
            device=device,
            batch_size=int(plan["evaluation"]["batch_size"]),
        )
        predictions.append(predicted)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_frame = pd.concat(predictions, ignore_index=True)
    per_seed = _metrics(prediction_frame)
    summary = _summary(per_seed)
    device_deltas = _paired_deltas(per_seed)
    incremental = _incremental(per_seed)
    reports.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    incremental.to_csv(incremental_path, index=False)
    device_deltas.to_csv(device_delta_path, index=False)
    _write_report(report_path, summary, device_deltas, incremental)

    expected_prediction_rows = (
        len(runs)
        * (
            len(wesad) * 2
            + len(cogwear)
        )
    )
    audit = {
        "status": (
            "pass"
            if (
                run_audit["status"] == "pass"
                and len(prediction_frame) == expected_prediction_rows
            )
            else "fail"
        ),
        "evaluation": "frozen_state_head_wesad_cogwear_zero_shot",
        "selected_candidate_id": json.loads(
            resolve(plan, plan["selection_json"]).read_text(encoding="utf-8")
        ).get("selected_candidate_id"),
        "model_updates": 0,
        "external_data_used_for_checkpoint_selection": False,
        "external_data_used_for_threshold_selection": False,
        "external_data_used_for_fusion_selection": False,
        "runs": int(len(runs)),
        "conditions": sorted(
            prediction_frame["condition_id"].unique().tolist()
        ),
        "seeds": sorted(
            map(int, prediction_frame["seed"].unique().tolist())
        ),
        "prediction_rows": int(len(prediction_frame)),
        "expected_prediction_rows": int(expected_prediction_rows),
        "source_rows": {
            "wesad": int(len(wesad)),
            "wesad_participants": int(
                wesad["participant_id"].nunique()
            ),
            "cogwear": int(len(cogwear)),
            "cogwear_participants": int(
                cogwear["participant_id"].nunique()
            ),
        },
        "thresholds": {
            "wesad": float(
                plan["evaluation"][
                    "wesad_binary_threshold_normalized"
                ]
            ),
            "cogwear": float(
                plan["evaluation"][
                    "cogwear_binary_threshold_normalized"
                ]
            ),
        },
        "wesad_source_audit": wesad_audit,
        "training_leakage_audit": run_audit,
        "outputs": {
            "predictions": str(predictions_path.resolve()),
            "per_seed": str(per_seed_path.resolve()),
            "summary": str(summary_path.resolve()),
            "incremental": str(incremental_path.resolve()),
            "device_deltas": str(device_delta_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    json_dump(audit_path, audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"WESAD/CogWear audit failed: {audit}")
    print(f"[DONE] WESAD/CogWear report -> {report_path}")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate(load_plan(args.plan), force=args.force)


if __name__ == "__main__":
    main()
