"""Frozen-checkpoint external zero-shot evaluation on EmoWear and CogWear.

The evaluator never updates a model, threshold, context mapping, or source
filter from external labels.  It evaluates the four already-trained conditions
for seeds 42/43/44 and preserves the distinction between:

* EmoWear seated V/A external transfer;
* EmoWear paired walk motion-robustness diagnostic;
* CogWear binary rest-vs-Stroop cognitive-effort transfer;
* CogWear E4-vs-Galaxy device transfer.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from case_window_common import (
    ccc,
    configure_console,
    json_dump,
    markdown_table,
    regression_metrics,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = HERE / "external_enrollment_plan.yaml"
PPG_COLUMNS = [f"ppg_f{index}" for index in range(512)]
TARGETS = ("valence", "arousal", "cognitive_load")


def load_plan(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    plan = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    plan["_plan_path"] = str(resolved)
    plan["_plan_dir"] = str(resolved.parent)
    return plan


def resolve(plan: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(plan["_plan_dir"]) / path
    return path.resolve()


def output_dir(plan: dict[str, Any]) -> Path:
    return resolve(plan, plan["output_dir"])


def _spec_text(
    renderer,
    validator,
    *,
    posture: str,
    energy: str,
    app_window: str,
    duration: str,
    event_type: str,
    detail: str = "",
) -> str:
    spec = {
        "posture": posture,
        "energy_expenditure": energy,
        "social_engagement": "not_recorded",
        "interpersonal_density": "not_recorded",
        "app_window": app_window,
        "duration": duration,
        "event_type": event_type,
        "observable_detail": detail,
    }
    validator(spec)
    return renderer(spec)


def _context_helpers(plan: dict[str, Any]):
    model_dir = resolve(plan, plan["model_dir"])
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from build_context7_gpt import (
        deterministic_text,
        duration_bucket,
        validate_spec,
    )

    return deterministic_text, duration_bucket, validate_spec


def _feature_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(["sample_id", *PPG_COLUMNS]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing feature columns: {sorted(missing)[:10]}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate feature sample_id: {path}")
    values = frame[PPG_COLUMNS].to_numpy(dtype=np.float32)
    if values.shape[1] != 512 or not np.isfinite(values).all():
        raise ValueError(f"Invalid 512D features: {path}")
    return frame.set_index("sample_id", drop=False)


def _emowear_rows(plan: dict[str, Any]) -> pd.DataFrame:
    renderer, duration_bucket, validator = _context_helpers(plan)
    paired = pd.read_csv(resolve(plan, plan["emowear"]["paired_manifest"]))
    seated_manifest = pd.read_csv(
        resolve(plan, plan["emowear"]["seated_manifest"])
    ).set_index("sample_id")
    seated_features = _feature_frame(
        resolve(plan, plan["emowear"]["seated_ppg"])
    )
    walk_features = _feature_frame(
        resolve(plan, plan["emowear"]["walk_ppg"])
    )
    rows: list[dict[str, Any]] = []
    for pair in paired.itertuples(index=False):
        seated_id = str(pair.seated_sample_id)
        walk_id = str(pair.walk_sample_id)
        if (
            seated_id not in seated_features.index
            or walk_id not in walk_features.index
            or seated_id not in seated_manifest.index
        ):
            continue
        seated_duration = float(
            seated_manifest.loc[seated_id, "ppg_duration_seconds"]
        )
        walk_duration = float(pair.walk_duration_seconds)
        seated_text = _spec_text(
            renderer,
            validator,
            posture="sitting",
            energy="sedentary",
            app_window="video_media_player",
            duration=duration_bucket(seated_duration),
            event_type="passive_video_viewing",
        )
        walk_text = _spec_text(
            renderer,
            validator,
            posture="standing",
            energy="light",
            app_window="no_active_app_window",
            duration=duration_bucket(walk_duration),
            event_type="other",
            detail="fixed indoor walking route",
        )
        walk_generic = _spec_text(
            renderer,
            validator,
            posture="standing",
            energy="light",
            app_window="no_active_app_window",
            duration=duration_bucket(walk_duration),
            event_type="other",
        )
        shared = {
            "dataset": "emowear",
            "participant_id": str(pair.participant_id),
            "external_unit_id": str(pair.pair_id),
            "cohort": "phase2",
            "session": f"seq{int(pair.sequence):02d}",
            "sequence": int(pair.sequence),
            "device": "e4",
            "valence": float(pair.valence),
            "arousal": float(pair.arousal),
            "cognitive_load": np.nan,
            "binary_threshold": float(
                plan["evaluation"][
                    "emowear_binary_threshold_normalized"
                ]
            ),
        }
        for phase, sample_id, feature_source, operational, generic in (
            (
                "seated",
                seated_id,
                seated_features,
                seated_text,
                seated_text,
            ),
            (
                "walk",
                walk_id,
                walk_features,
                walk_text,
                walk_generic,
            ),
        ):
            source = feature_source.loc[sample_id]
            rows.append(
                {
                    **shared,
                    "sample_id": sample_id,
                    "domain": phase,
                    "phase": phase,
                    "operational_text": operational,
                    "generic_text": generic,
                    "ppg_confidence": 1.0,
                    **{
                        column: float(source[column])
                        for column in PPG_COLUMNS
                    },
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No complete EmoWear seated/walk pairs")
    counts = frame.groupby(["participant_id", "external_unit_id"]).size()
    if not counts.eq(2).all():
        raise RuntimeError("EmoWear pair completeness failed")
    return frame


def _emowear_rest_rows(plan: dict[str, Any]) -> pd.DataFrame:
    renderer, duration_bucket, validator = _context_helpers(plan)
    root = output_dir(plan) / "source_data"
    manifest = pd.read_csv(root / "emowear_rest_manifest.csv")
    features = _feature_frame(root / "emowear_rest_ppg512.csv")
    rows = []
    for source in manifest.itertuples(index=False):
        sample_id = str(source.sample_id)
        feature = features.loc[sample_id]
        text = _spec_text(
            renderer,
            validator,
            posture="sitting",
            energy="sedentary",
            app_window="no_active_app_window",
            duration=duration_bucket(float(source.duration_seconds)),
            event_type="resting_baseline",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "emowear_rest",
                "participant_id": str(source.participant_id),
                "external_unit_id": sample_id,
                "cohort": "phase2",
                "session": "enrollment_rest",
                "sequence": -1,
                "device": "e4",
                "domain": "rest",
                "phase": "rest",
                "operational_text": text,
                "generic_text": text,
                "ppg_confidence": 1.0,
                **{
                    column: float(feature[column])
                    for column in PPG_COLUMNS
                },
            }
        )
    return pd.DataFrame(rows)


def _cogwear_rows(plan: dict[str, Any]) -> pd.DataFrame:
    renderer, duration_bucket, validator = _context_helpers(plan)
    root = output_dir(plan) / "source_data"
    manifest = pd.read_csv(root / "cogwear_manifest.csv")
    manifest = manifest[manifest["paired_complete"].eq(1)].copy()
    features = _feature_frame(root / "cogwear_ppg512.csv")
    rows = []
    for source in manifest.itertuples(index=False):
        sample_id = str(source.sample_id)
        feature = features.loc[sample_id]
        duration = duration_bucket(float(source.duration_seconds))
        if str(source.condition) == "baseline":
            operational = _spec_text(
                renderer,
                validator,
                posture="sitting",
                energy="sedentary",
                app_window="no_active_app_window",
                duration=duration,
                event_type="resting_baseline",
            )
            generic = operational
        else:
            operational = _spec_text(
                renderer,
                validator,
                posture="sitting",
                energy="sedentary",
                app_window="task_interface",
                duration=duration,
                event_type="other",
                detail="Stroop color-word interference task",
            )
            generic = _spec_text(
                renderer,
                validator,
                posture="sitting",
                energy="sedentary",
                app_window="task_interface",
                duration=duration,
                event_type="other",
            )
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "cogwear",
                "participant_id": str(source.participant_id),
                "external_unit_id": (
                    f"{source.participant_id}__{source.session}__"
                    f"{source.condition}"
                ),
                "cohort": str(source.cohort),
                "session": str(source.session),
                "sequence": int(source.temporal_order),
                "device": str(source.device),
                "domain": str(source.device),
                "phase": str(source.condition),
                "operational_text": operational,
                "generic_text": generic,
                "valence": np.nan,
                "arousal": np.nan,
                "cognitive_load": float(source.cognitive_load),
                "binary_threshold": float(
                    plan["evaluation"][
                        "cogwear_binary_threshold_normalized"
                    ]
                ),
                "ppg_confidence": 1.0,
                **{
                    column: float(feature[column])
                    for column in PPG_COLUMNS
                },
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No paired CogWear E4/Galaxy rows")
    required = {("baseline", "e4"), ("baseline", "galaxy"),
                ("cognitive_load", "e4"), ("cognitive_load", "galaxy")}
    for _, group in frame.groupby(["participant_id", "session"]):
        present = set(zip(group["phase"], group["device"]))
        if not required.issubset(present):
            raise RuntimeError("CogWear complete-pair filtering failed")
    return frame


def _load_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    selection_path = resolve(plan, plan["selection_json"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    seeds = set(map(int, plan["seeds"]))
    conditions = set(map(str, plan["conditions"]))
    runs = []
    for selected in selection["runs"]:
        if (
            selected.get("protocol") != "participant_holdout"
            or selected.get("fold_video") is not None
            or int(selected.get("seed", -1)) not in seeds
            or str(selected.get("condition_id")) not in conditions
        ):
            continue
        checkpoint = Path(selected["checkpoint"])
        run_dir = checkpoint.parents[1]
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_group") is not None:
            continue
        metadata["_metadata_path"] = str(metadata_path.resolve())
        runs.append(metadata)
    expected = len(seeds) * len(conditions)
    keys = {(int(row["seed"]), str(row["condition_id"])) for row in runs}
    if len(runs) != expected or len(keys) != expected:
        raise RuntimeError(
            f"Expected {expected} final participant-holdout runs, got "
            f"{len(runs)} ({len(keys)} unique)"
        )
    return sorted(runs, key=lambda row: (row["condition_id"], row["seed"]))


def _load_model(
    plan: dict[str, Any],
    metadata: dict[str, Any],
    device: torch.device,
):
    model_dir = resolve(plan, plan["model_dir"])
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model import MultimodalStateEstimator

    cfg = yaml.safe_load(Path(metadata["config"]).read_text(encoding="utf-8"))
    model = MultimodalStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        ppg_input_dim=int(cfg["model"].get("ppg_input_dim", 512)),
        projection_dim=int(cfg["model"]["projection_dim"]),
        projection_dropout=float(cfg["model"]["projection_dropout"]),
        ppg_hidden_dim=int(cfg["model"]["ppg_hidden_dim"]),
        fusion_hidden_dim=int(cfg["model"]["fusion_hidden_dim"]),
        text_init_ckpt=None,
        text_max_length=int(cfg["model"].get("text_max_length", 96)),
        posture_film_enabled=False,
        fusion_mode=str(cfg["model"].get("fusion_mode", "concat")),
    ).to(device)
    model.load_state_dict(
        torch.load(
            metadata["checkpoint"], map_location=device, weights_only=False
        ),
        strict=True,
    )
    model.eval()
    if model.fusion_mode not in {
        "concat",
        "context_prior_quality_gated_ppg_residual",
        "target_routed_direct",
    }:
        raise ValueError(
            "External evaluation supports uniform concat, the frozen "
            "residual model, or the residual-free target-routed model"
        )
    return model


@torch.inference_mode()
def _predict(
    model,
    frame: pd.DataFrame,
    condition: dict[str, Any],
    *,
    condition_id: str,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_column = (
        "generic_text"
        if condition_id == "generic_context_ppg"
        else "operational_text"
    )
    texts = frame[text_column].astype(str).tolist()
    unique_texts = list(dict.fromkeys(texts))
    if bool(condition["mute_text"]):
        text_lookup = {
            text: torch.zeros(
                model.projection_dim, dtype=torch.float32, device=device
            )
            for text in unique_texts
        }
    else:
        encoded = model.text_branch(unique_texts, device)
        text_lookup = {
            text: value for text, value in zip(unique_texts, encoded)
        }
    metadata_columns = [
        "sample_id",
        "dataset",
        "participant_id",
        "external_unit_id",
        "cohort",
        "session",
        "sequence",
        "device",
        "domain",
        "phase",
    ]
    prediction_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for begin in range(0, len(frame), batch_size):
        end = min(begin + batch_size, len(frame))
        chunk = frame.iloc[begin:end]
        text_z = torch.stack(
            [text_lookup[text] for text in texts[begin:end]], dim=0
        )
        features = torch.as_tensor(
            chunk[PPG_COLUMNS].to_numpy(dtype=np.float32), device=device
        )
        if bool(condition["mute_ppg"]):
            ppg_z = torch.zeros(
                len(chunk),
                model.projection_dim,
                dtype=features.dtype,
                device=device,
            )
            gate = torch.zeros(len(chunk), dtype=features.dtype, device=device)
        else:
            ppg_z = model.ppg_branch(features)
            gate = torch.as_tensor(
                chunk["ppg_confidence"].to_numpy(dtype=np.float32),
                device=device,
            ).clamp(0.0, 1.0)
        if model.fusion_mode == "target_routed_direct":
            # Match the main-model forward contract: quality masks the PPG
            # projection before direct A/C fusion, while Valence has no PPG
            # path at all.  The zero-PPG values below are counterfactual
            # references for auditing only; they are not a residual model.
            ppg_z = ppg_z * gate.view(-1, 1)
            valence_h = model.target_context_trunks["valence"](text_z)
            prediction = {
                "valence": model.target_routed_heads["valence"](
                    valence_h
                ).squeeze(-1)
            }
            prior = {"valence": prediction["valence"]}
            for target in ("arousal", "cognitive_load"):
                fused = torch.cat([text_z, ppg_z], dim=-1)
                zero_ppg = torch.cat(
                    [text_z, torch.zeros_like(ppg_z)], dim=-1
                )
                prediction[target] = model.target_routed_heads[target](
                    model.target_multimodal_trunks[target](fused)
                ).squeeze(-1)
                prior[target] = model.target_routed_heads[target](
                    model.target_multimodal_trunks[target](zero_ppg)
                ).squeeze(-1)
            contribution = {
                target: prediction[target] - prior[target]
                for target in TARGETS
            }
            # Preserve the historical output schema.  In this mode the field
            # is a direct-fusion counterfactual difference, not a learned
            # residual head output.
            residual = contribution
            decomposition_kind = "direct_minus_zero_ppg_counterfactual"
        elif model.fusion_mode == "concat":
            # Historical uniform direct fusion: every target consumes the
            # same concatenated Context+quality-masked PPG representation.
            ppg_z = ppg_z * gate.view(-1, 1)
            fused = torch.cat([text_z, ppg_z], dim=-1)
            zero_ppg = torch.cat(
                [text_z, torch.zeros_like(ppg_z)], dim=-1
            )
            prediction_h = model.trunk(fused)
            prior_h = model.trunk(zero_ppg)
            prediction = {
                target: model.heads[target](prediction_h).squeeze(-1)
                for target in TARGETS
            }
            prior = {
                target: model.heads[target](prior_h).squeeze(-1)
                for target in TARGETS
            }
            contribution = {
                target: prediction[target] - prior[target]
                for target in TARGETS
            }
            residual = contribution
            decomposition_kind = "concat_minus_zero_ppg_counterfactual"
        else:
            residual_h = model.ppg_residual_trunk(ppg_z)
            prior = {
                target: model.context_prior_heads[target](
                    text_z
                ).squeeze(-1)
                for target in TARGETS
            }
            residual = {
                target: model.ppg_residual_heads[target](
                    residual_h
                ).squeeze(-1)
                for target in TARGETS
            }
            contribution = {
                target: gate * residual[target] for target in TARGETS
            }
            prediction = {
                target: prior[target] + contribution[target]
                for target in TARGETS
            }
            decomposition_kind = "learned_quality_gated_residual"
        for local_index, (_, source) in enumerate(chunk.iterrows()):
            base = {
                column: source[column] for column in metadata_columns
            }
            decomposition = {
                **base,
                "condition_id": condition_id,
                "seed": int(seed),
                "fusion_mode": model.fusion_mode,
                "decomposition_kind": decomposition_kind,
            }
            for target in TARGETS:
                decomposition[f"context_prior_{target}"] = float(
                    prior[target][local_index].cpu()
                )
                decomposition[f"ppg_residual_{target}"] = float(
                    residual[target][local_index].cpu()
                )
                decomposition[f"ppg_contribution_{target}"] = float(
                    contribution[target][local_index].cpu()
                )
                decomposition[f"prediction_{target}"] = float(
                    prediction[target][local_index].cpu()
                )
                truth = source.get(target, np.nan)
                if pd.notna(truth):
                    prediction_rows.append(
                        {
                            **base,
                            "condition_id": condition_id,
                            "seed": int(seed),
                            "fusion_mode": model.fusion_mode,
                            "decomposition_kind": decomposition_kind,
                            "target": target,
                            "truth": float(truth),
                            "prediction": float(
                                prediction[target][local_index].cpu()
                            ),
                            "context_prior": float(
                                prior[target][local_index].cpu()
                            ),
                            "ppg_residual": float(
                                residual[target][local_index].cpu()
                            ),
                            "ppg_contribution": float(
                                contribution[target][local_index].cpu()
                            ),
                            "binary_threshold": float(
                                source["binary_threshold"]
                            ),
                            "text_mode": (
                                "muted"
                                if bool(condition["mute_text"])
                                else text_column
                            ),
                        }
                    )
            decomposition_rows.append(decomposition)
    return pd.DataFrame(prediction_rows), pd.DataFrame(decomposition_rows)


def _binary_metrics(
    prediction: Iterable[float],
    truth: Iterable[float],
    threshold: float,
) -> dict[str, float]:
    prediction = np.asarray(list(prediction), dtype=float) >= threshold
    truth = np.asarray(list(truth), dtype=float) >= threshold
    accuracy = float(np.mean(prediction == truth))
    recalls = []
    f1s = []
    for positive in (False, True):
        tp = int(np.sum((prediction == positive) & (truth == positive)))
        fp = int(np.sum((prediction == positive) & (truth != positive)))
        fn = int(np.sum((prediction != positive) & (truth == positive)))
        recalls.append(tp / (tp + fn) if tp + fn else float("nan"))
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def _participant_macro_ccc(frame: pd.DataFrame) -> float:
    values = []
    for _, group in frame.groupby("participant_id"):
        if len(group) >= 2:
            values.append(ccc(group["prediction"], group["truth"]))
    return float(np.nanmean(values)) if values else float("nan")


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "domain", "condition_id", "seed", "target"]
    for group_keys, group in predictions.groupby(keys, sort=True):
        threshold_values = group["binary_threshold"].unique()
        if len(threshold_values) != 1:
            raise ValueError(f"Mixed binary threshold in {group_keys}")
        regression = regression_metrics(group["prediction"], group["truth"])
        binary = _binary_metrics(
            group["prediction"],
            group["truth"],
            float(threshold_values[0]),
        )
        rows.append(
            {
                **dict(zip(keys, group_keys)),
                **regression,
                "participant_macro_ccc": _participant_macro_ccc(group),
                **binary,
                "participants": int(group["participant_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ccc",
        "participant_macro_ccc",
        "mae",
        "rmse",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    group_columns = ["dataset", "domain", "condition_id", "target"]
    rows = []
    for keys, group in per_seed.groupby(group_columns, sort=True):
        row = {
            **dict(zip(group_columns, keys)),
            "seeds": int(group["seed"].nunique()),
            "n_per_seed": int(group["n"].iloc[0]),
            "participants": int(group["participants"].iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = [
        "ccc",
        "participant_macro_ccc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    comparisons = (
        ("emowear", "walk", "seated", "walk_minus_seated"),
        ("cogwear", "galaxy", "e4", "galaxy_minus_e4"),
    )
    for dataset, candidate, reference, label in comparisons:
        selected = per_seed[per_seed["dataset"].eq(dataset)]
        index = ["condition_id", "seed", "target"]
        left = selected[selected["domain"].eq(candidate)].set_index(index)
        right = selected[selected["domain"].eq(reference)].set_index(index)
        common = left.index.intersection(right.index)
        for key in common:
            left_row = left.loc[key]
            right_row = right.loc[key]
            rows.append(
                {
                    "comparison": label,
                    "condition_id": key[0],
                    "seed": int(key[1]),
                    "target": key[2],
                    **{
                        f"delta_{metric}": float(
                            left_row[metric] - right_row[metric]
                        )
                        for metric in metric_names
                    },
                }
            )
    return pd.DataFrame(rows)


def _incremental(per_seed: pd.DataFrame) -> pd.DataFrame:
    index = ["dataset", "domain", "seed", "target"]
    rows = []
    comparison_pairs = (
        ("context_content_ppg", "context_content_only"),
        (
            "title_context_ac_direct_ppg",
            "title_context_only_fixed_base",
        ),
        (
            "title_context_ppg_uniform_direct",
            "title_context_only_fixed_base",
        ),
    )
    available = set(per_seed["condition_id"].astype(str).unique())
    for fusion_id, context_id in comparison_pairs:
        if not {fusion_id, context_id}.issubset(available):
            continue
        fusion = per_seed[
            per_seed["condition_id"].eq(fusion_id)
        ].set_index(index)
        context = per_seed[
            per_seed["condition_id"].eq(context_id)
        ].set_index(index)
        common = fusion.index.intersection(context.index)
        for key in common:
            rows.append(
                {
                    **dict(zip(index, key)),
                    "multimodal_condition": fusion_id,
                    "context_condition": context_id,
                    "delta_ccc": float(
                        fusion.loc[key, "ccc"] - context.loc[key, "ccc"]
                    ),
                    "delta_accuracy": float(
                        fusion.loc[key, "accuracy"]
                        - context.loc[key, "accuracy"]
                    ),
                    "delta_balanced_accuracy": float(
                        fusion.loc[key, "balanced_accuracy"]
                        - context.loc[key, "balanced_accuracy"]
                    ),
                    "delta_macro_f1": float(
                        fusion.loc[key, "macro_f1"]
                        - context.loc[key, "macro_f1"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _audit_runs(
    plan: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = set(map(str, plan["evaluation"]["training_datasets_must_equal"]))
    details = []
    errors = []
    for metadata in runs:
        manifest = pd.read_csv(metadata["manifest"], usecols=["dataset"])
        datasets = set(manifest["dataset"].astype(str).str.lower().unique())
        external_rows = int(
            manifest["dataset"].astype(str).str.lower().isin(
                {"emowear", "cogwear", "wesad"}
            ).sum()
        )
        row = {
            "condition_id": metadata["condition_id"],
            "seed": int(metadata["seed"]),
            "training_datasets": sorted(datasets),
            "external_training_rows": external_rows,
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": sha256_file(Path(metadata["checkpoint"])),
        }
        details.append(row)
        if datasets != expected or external_rows:
            errors.append(row)
    return {
        "status": "pass" if not errors else "fail",
        "expected_training_datasets": sorted(expected),
        "runs": details,
        "violations": errors,
    }


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
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
        deltas.groupby(["comparison", "condition_id", "target"], as_index=False)
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
        "# EmoWear + CogWear frozen external zero-shot",
        "",
        "- Frozen current checkpoints only: no external fine-tuning or threshold selection.",
        "- EmoWear seated is the V/A external-transfer result.",
        "- EmoWear walk is a paired motion-robustness diagnostic using the same trial SAM; it is not an independent walk-time affect label.",
        "- CogWear is binary objective cognitive-effort condition transfer (rest vs Stroop), not continuous subjective cognitive load.",
        "- CogWear E4 and Galaxy are evaluated on complete paired participant-session blocks.",
        "- CCC is primary for EmoWear continuous V/A. Macro-F1, accuracy, and balanced accuracy are primary for binary CogWear C; CogWear CCC is secondary.",
        "",
        "## Seed mean ± variability",
        "",
        markdown_table(summary[display_columns]),
        "",
        "## Motion/device deltas",
        "",
        markdown_table(delta_summary),
        "",
        "## PPG incremental contribution",
        "",
        "Multimodal minus matched Context-only on the identical external rows:",
        "",
        markdown_table(incremental_summary),
        "",
        "## Claim boundary",
        "",
        "These checkpoints were formally selected with CASE/EEVR/MAUS validation only. Historical development had previously inspected EmoWear with older checkpoints, so the current result should be described as a frozen-checkpoint external diagnostic rather than a pristine never-observed confirmatory test.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(plan: dict[str, Any], *, force: bool) -> None:
    out = output_dir(plan)
    reports = out / "reports"
    predictions_path = reports / "external_zero_shot_predictions.csv"
    rest_path = reports / "external_rest_decomposition.csv"
    per_seed_path = reports / "external_metrics_per_seed.csv"
    summary_path = reports / "external_metrics_summary.csv"
    report_path = reports / "EXTERNAL_ZERO_SHOT_RESULTS.md"
    audit_path = reports / "external_zero_shot_audit.json"
    required = [
        predictions_path,
        rest_path,
        per_seed_path,
        summary_path,
        report_path,
        audit_path,
    ]
    if all(path.is_file() for path in required) and not force:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "pass":
            print(f"[SKIP] external zero-shot outputs already pass: {report_path}")
            return
    source_audit = out / "source_data" / "external_source_audit.json"
    if not source_audit.is_file():
        raise FileNotFoundError(
            f"Run build_external_zeroshot_sources.py first: {source_audit}"
        )
    emowear = _emowear_rows(plan)
    cogwear = _cogwear_rows(plan)
    evaluation_frame = pd.concat([emowear, cogwear], ignore_index=True)
    rest_frame = _emowear_rest_rows(plan)
    runs = _load_runs(plan)
    run_audit = _audit_runs(plan, runs)
    if run_audit["status"] != "pass":
        raise RuntimeError(f"External training leakage audit failed: {run_audit}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(plan["evaluation"]["batch_size"])
    predictions = []
    rest_decompositions = []
    for metadata in runs:
        condition_id = str(metadata["condition_id"])
        seed = int(metadata["seed"])
        print(f"[ZERO-SHOT] {condition_id} seed={seed}")
        model = _load_model(plan, metadata, device)
        predicted, _ = _predict(
            model,
            evaluation_frame,
            metadata["condition"],
            condition_id=condition_id,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )
        _, rest = _predict(
            model,
            rest_frame,
            metadata["condition"],
            condition_id=condition_id,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )
        predictions.append(predicted)
        rest_decompositions.append(rest)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    prediction_frame = pd.concat(predictions, ignore_index=True)
    rest_decomposition = pd.concat(rest_decompositions, ignore_index=True)
    per_seed = _metrics(prediction_frame)
    summary = _summary(per_seed)
    deltas = _paired_deltas(per_seed)
    incremental = _incremental(per_seed)
    reports.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_path, index=False)
    rest_decomposition.to_csv(rest_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    deltas.to_csv(reports / "external_motion_device_deltas.csv", index=False)
    incremental.to_csv(
        reports / "external_ppg_incremental_metrics.csv", index=False
    )
    _write_report(report_path, summary, deltas, incremental)
    audit = {
        "status": "pass",
        "evaluation": "frozen_checkpoint_external_zero_shot",
        "model_updates": 0,
        "external_data_used_for_checkpoint_selection": False,
        "runs": len(runs),
        "conditions": sorted(prediction_frame["condition_id"].unique()),
        "seeds": sorted(map(int, prediction_frame["seed"].unique())),
        "source_rows": {
            "emowear_seated_walk": int(len(emowear)),
            "emowear_pairs": int(emowear["external_unit_id"].nunique()),
            "emowear_participants": int(
                emowear["participant_id"].nunique()
            ),
            "cogwear": int(len(cogwear)),
            "cogwear_participants": int(
                cogwear["participant_id"].nunique()
            ),
            "cogwear_complete_participant_sessions": int(
                cogwear[["participant_id", "session"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "training_leakage_audit": run_audit,
        "thresholds": {
            "emowear": float(
                plan["evaluation"][
                    "emowear_binary_threshold_normalized"
                ]
            ),
            "cogwear": float(
                plan["evaluation"][
                    "cogwear_binary_threshold_normalized"
                ]
            ),
        },
        "files": {
            "predictions": str(predictions_path.resolve()),
            "rest_decomposition": str(rest_path.resolve()),
            "per_seed_metrics": str(per_seed_path.resolve()),
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    json_dump(audit_path, audit)
    print(f"[DONE] external zero-shot report -> {report_path}")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate(load_plan(args.plan), force=args.force)


if __name__ == "__main__":
    main()
