"""Evaluate CASE window experiments and compare with legacy block results."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from case_window_common import (
    DEFAULT_PLAN,
    binary_metrics,
    configure_console,
    json_dump,
    load_plan,
    markdown_table,
    participant_macro_ccc,
    regression_metrics,
    resolve,
    work_dir,
)


def _load_model(
    model_dir: Path,
    cfg: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
):
    sys.path.insert(0, str(model_dir))
    from model import MultimodalStateEstimator

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
        torch.load(checkpoint, map_location=device, weights_only=False),
        strict=True,
    )
    model.eval()
    return model


def _metric_payload(
    frame: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    truth = frame["truth"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    median_threshold = float(np.median(truth))
    is_case = (
        frame["dataset"].astype(str).str.lower().eq("case").all()
        if "dataset" in frame
        else True
    )
    if is_case and frame["video_id"].notna().all():
        centered = frame.copy()
        group_columns = ["participant_id", "video_id"]
        centered["truth_centered"] = (
            centered["truth"]
            - centered.groupby(group_columns)["truth"].transform("mean")
        )
        centered["prediction_centered"] = (
            centered["prediction"]
            - centered.groupby(group_columns)["prediction"].transform("mean")
        )
        residual = regression_metrics(
            centered["prediction_centered"],
            centered["truth_centered"],
        )
    else:
        residual = {
            "n": int(len(frame)),
            "ccc": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "target": target,
        **regression_metrics(prediction, truth),
        "participant_macro_ccc": participant_macro_ccc(
            frame["participant_id"], prediction, truth
        ),
        "binary_median": binary_metrics(
            prediction, truth, median_threshold
        ),
        # CASE is [0.5,9.5], so normalized 0.5 is raw score 5.0.
        "binary_fixed_midpoint": binary_metrics(
            prediction, truth, 0.5
        ),
        "within_video_residual": residual,
    }


def evaluate_run(
    plan: dict[str, Any],
    metadata_path: Path,
    *,
    force: bool,
    split_name: str = "test",
) -> tuple[dict[str, Any], pd.DataFrame]:
    split_name = str(split_name).strip().lower()
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be validation or test")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_dir = metadata_path.parent
    evaluation_dir = (
        run_dir / "evaluation"
        if split_name == "test"
        else run_dir / f"evaluation_{split_name}"
    )
    metrics_path = evaluation_dir / "metrics.json"
    predictions_path = evaluation_dir / "predictions.csv"
    if metrics_path.is_file() and predictions_path.is_file() and not force:
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            pd.read_csv(predictions_path),
        )
    cfg = yaml.safe_load(
        Path(metadata["config"]).read_text(encoding="utf-8")
    )
    model_dir = resolve(plan, plan["model_dir"])
    sys.path.insert(0, str(model_dir))
    from dataset import MultiDatasetAffect, collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(
        model_dir,
        cfg,
        Path(metadata["checkpoint"]),
        device,
    )
    dataset = MultiDatasetAffect(
        cfg["paths"]["manifest_csv"],
        cfg["paths"]["ppg_csv"],
        ppg_dim=512,
    )
    selected = np.flatnonzero(
        dataset.df["split"]
        .astype(str)
        .str.lower()
        .eq(split_name)
        .to_numpy()
    )
    loader = DataLoader(
        Subset(dataset, selected.tolist()),
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
    )
    condition = metadata["condition"]
    targets = ("valence", "arousal", "cognitive_load")
    outputs = {target: [] for target in targets}
    truths = {target: [] for target in targets}
    masks = {target: [] for target in targets}
    sample_ids: list[str] = []
    participants: list[str] = []
    datasets: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            prediction = model(
                texts=batch["text"],
                ppg_features=batch["ppg_features"],
                ppg_mask=batch["ppg_mask"],
                postures=batch["posture"],
                mute_text=bool(condition["mute_text"]),
                mute_ppg=bool(condition["mute_ppg"]),
                device=device,
            )["preds"]
            sample_ids.extend(batch["sample_id"])
            participants.extend(batch["participant_id"])
            datasets.extend(batch["dataset"])
            for target in outputs:
                outputs[target].append(
                    prediction[target].detach().cpu().numpy()
                )
                truths[target].append(batch["targets"][target].numpy())
                masks[target].append(batch["masks"][target].numpy())
    lookup = dataset.df.set_index("sample_id")
    rows = []
    for target in outputs:
        predicted = np.concatenate(outputs[target])
        expected = np.concatenate(truths[target])
        active = np.concatenate(masks[target]).astype(bool)
        for sample_id, participant, source_dataset, truth, prediction in zip(
            np.asarray(sample_ids, dtype=object)[active],
            np.asarray(participants, dtype=object)[active],
            np.asarray(datasets, dtype=object)[active],
            expected[active],
            predicted[active],
        ):
            source = lookup.loc[sample_id]
            video_id = pd.to_numeric(
                source.get("video_id"), errors="coerce"
            )
            window_index = pd.to_numeric(
                source.get("window_index"), errors="coerce"
            )
            rows.append(
                {
                    "protocol": metadata["protocol"],
                    "evaluation_split": split_name,
                    "fold_video": metadata["fold_video"],
                    "condition_id": metadata["condition_id"],
                    "seed": metadata["seed"],
                    "case_loss_weight": metadata.get(
                        "case_loss_weight"
                    ),
                    "sample_id": sample_id,
                    "dataset": str(source_dataset).lower(),
                    "participant_id": participant,
                    "video_id": (
                        int(video_id) if pd.notna(video_id) else np.nan
                    ),
                    "window_index": (
                        int(window_index)
                        if pd.notna(window_index)
                        else np.nan
                    ),
                    "target": target,
                    "truth": float(truth),
                    "prediction": float(prediction),
                }
            )
    prediction_frame = pd.DataFrame(rows)
    task_metrics = []
    for (source_dataset, target), group in prediction_frame.groupby(
        ["dataset", "target"], sort=True
    ):
        task_metrics.append(
            {
                "dataset": str(source_dataset),
                **_metric_payload(group, str(target)),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": metadata["protocol"],
        "fold_video": metadata["fold_video"],
        "condition_id": metadata["condition_id"],
        "seed": metadata["seed"],
        "evaluation_split": split_name,
        "evaluation_rows": int(len(selected)),
        "evaluation_rows_by_dataset": {
            str(key): int(value)
            for key, value in dataset.df.iloc[selected][
                "dataset"
            ].value_counts().to_dict().items()
        },
        "binary_note": (
            "legacy-compatible binary_median uses test-truth median; "
            "binary_fixed_midpoint uses normalized 0.5 (= CASE raw 5.0). "
            "CASE does not use a 1-3 versus 4-5 label rule."
        ),
        "tasks": task_metrics,
    }
    # Preserve the historical payload keys for downstream readers of the
    # held-out test evaluation.
    if split_name == "test":
        payload["test_rows"] = payload["evaluation_rows"]
        payload["test_rows_by_dataset"] = payload[
            "evaluation_rows_by_dataset"
        ]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(metrics_path, payload)
    prediction_frame.to_csv(predictions_path, index=False)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload, prediction_frame


def _legacy_block_rows(plan: dict[str, Any]) -> pd.DataFrame:
    path = (
        resolve(plan, plan["public_source"]["manifest"]).parents[1]
        / "reports"
        / "public_dataset_task_metrics_summary.csv"
    )
    if not path.is_file():
        # public manifest is work_public_clsp/source_data/file.csv.
        path = (
            resolve(plan, plan["public_source"]["manifest"]).parents[1]
            / "reports"
            / "public_dataset_task_metrics_summary.csv"
        )
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"].astype(str).str.lower().eq("case")].copy()
    frame["evaluation"] = "legacy_block_validation"
    return frame


def _summary_from_predictions(
    predictions: pd.DataFrame,
    protocol: str,
) -> pd.DataFrame:
    rows = []
    selected = predictions[predictions["protocol"].eq(protocol)]
    if len(selected) == 0:
        return pd.DataFrame()
    group_columns = ["condition_id", "seed", "dataset", "target"]
    for keys, group in selected.groupby(group_columns, sort=True):
        condition_id, seed, source_dataset, target = keys
        metric = _metric_payload(group, target)
        fold_values = []
        if group["fold_video"].notna().any():
            for _, fold_group in group.groupby("fold_video"):
                fold_values.append(
                    regression_metrics(
                        fold_group["prediction"], fold_group["truth"]
                    )["ccc"]
                )
        rows.append(
            {
                "protocol": protocol,
                "condition_id": condition_id,
                "seed": int(seed),
                "case_loss_weight": float(
                    group["case_loss_weight"].dropna().iloc[0]
                )
                if group["case_loss_weight"].notna().any()
                else float("nan"),
                "dataset": str(source_dataset),
                "target": target,
                "n": metric["n"],
                "ccc": metric["ccc"],
                "participant_macro_ccc": metric[
                    "participant_macro_ccc"
                ],
                "mae": metric["mae"],
                "rmse": metric["rmse"],
                "median_f1": metric["binary_median"]["f1"],
                "median_accuracy": metric["binary_median"]["accuracy"],
                "midpoint_f1": metric["binary_fixed_midpoint"]["f1"],
                "midpoint_accuracy": metric[
                    "binary_fixed_midpoint"
                ]["accuracy"],
                "within_video_residual_ccc": metric[
                    "within_video_residual"
                ]["ccc"],
                "within_video_residual_mae": metric[
                    "within_video_residual"
                ]["mae"],
                "video_macro_ccc": (
                    float(np.nanmean(fold_values))
                    if fold_values
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _vac_target_summary(
    task_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate held-out dataset tasks inside V/A/C, then across seeds."""
    if len(task_summary) == 0:
        return pd.DataFrame(), pd.DataFrame()
    expected = {
        "valence": {"case", "eevr"},
        "arousal": {"case", "eevr"},
        "cognitive_load": {"maus"},
    }
    metric_columns = [
        "ccc",
        "participant_macro_ccc",
        "mae",
        "rmse",
        "median_f1",
        "median_accuracy",
        "midpoint_f1",
        "midpoint_accuracy",
    ]
    rows = []
    group_columns = ["protocol", "condition_id", "seed"]
    for keys, run in task_summary.groupby(group_columns, sort=True):
        protocol, condition_id, seed = keys
        for target, datasets in expected.items():
            selected = run[
                run["target"].eq(target)
                & run["dataset"].astype(str).str.lower().isin(datasets)
            ]
            actual = set(selected["dataset"].astype(str).str.lower())
            if actual != datasets:
                raise ValueError(
                    f"{condition_id}/seed_{seed}/{target}: held-out "
                    f"datasets={sorted(actual)}, expected={sorted(datasets)}"
                )
            row = {
                "protocol": protocol,
                "condition_id": condition_id,
                "seed": int(seed),
                "target": target,
                "datasets": "+".join(sorted(datasets)),
                "case_loss_weight": float(
                    selected["case_loss_weight"].dropna().iloc[0]
                )
                if selected["case_loss_weight"].notna().any()
                else float("nan"),
            }
            for metric in metric_columns:
                row[metric] = float(
                    np.nanmean(selected[metric].to_numpy(dtype=float))
                )
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    aggregate_rows = []
    for keys, group in per_seed.groupby(
        ["protocol", "condition_id", "target"], sort=True
    ):
        protocol, condition_id, target = keys
        row = {
            "protocol": protocol,
            "condition_id": condition_id,
            "target": target,
            "seeds": int(group["seed"].nunique()),
            "datasets": str(group["datasets"].iloc[0]),
            "case_loss_weight": float(
                group["case_loss_weight"].dropna().iloc[0]
            )
            if group["case_loss_weight"].notna().any()
            else float("nan"),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(
                np.nanstd(values, ddof=1)
                if np.isfinite(values).sum() > 1
                else 0.0
            )
        aggregate_rows.append(row)
    return per_seed, pd.DataFrame(aggregate_rows)


def _write_vac_report(
    output: Path,
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> None:
    lines = [
        "# V/A/C held-out CCC, F1, and Accuracy",
        "",
        "- V = mean(CASE V, EEVR V).",
        "- A = mean(CASE A, EEVR A).",
        "- C = MAUS C.",
        "- CCC/MAE/RMSE are primary continuous metrics.",
        "- F1/Accuracy are diagnostics; `median_*` uses each held-out",
        "  dataset×target truth median, and `midpoint_*` uses 0.5.",
        "- CASE does not use an EEVR `1–3 vs 4–5` rule.",
        "",
        "## Seed-aggregated",
        "",
    ]
    if len(aggregate):
        columns = [
            "protocol",
            "condition_id",
            "target",
            "seeds",
            "case_loss_weight",
            "ccc_mean",
            "ccc_std",
            "participant_macro_ccc_mean",
            "mae_mean",
            "rmse_mean",
            "median_f1_mean",
            "median_accuracy_mean",
            "midpoint_f1_mean",
            "midpoint_accuracy_mean",
        ]
        lines.append(markdown_table(aggregate[columns]))
    else:
        lines.append("- No completed held-out runs.")
    lines.extend(["", "## Per seed", ""])
    if len(per_seed):
        columns = [
            "protocol",
            "condition_id",
            "seed",
            "target",
            "datasets",
            "case_loss_weight",
            "ccc",
            "participant_macro_ccc",
            "mae",
            "rmse",
            "median_f1",
            "median_accuracy",
            "midpoint_f1",
            "midpoint_accuracy",
        ]
        lines.append(markdown_table(per_seed[columns]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    legacy: pd.DataFrame,
) -> None:
    lines = [
        "# Original-7 CASE/EEVR/MAUS held-out evaluation",
        "",
        "## Binary evaluation correction",
        "",
        "- The completed public evaluator did **not** use `1–3 vs 4–5`.",
        "- It binarized each held-out dataset/target at the held-out truth median.",
        "- CASE labels are `[0.5,9.5]`; fixed midpoint diagnostics therefore use",
        "  raw `5.0`, equivalently normalized `0.5`.",
        "- CCC/MAE/RMSE remain the primary metrics.",
        "",
        "## Window-level results",
        "",
    ]
    if len(summary):
        display_columns = [
            "protocol",
            "condition_id",
            "seed",
            "dataset",
            "target",
            "n",
            "ccc",
            "participant_macro_ccc",
            "mae",
            "rmse",
            "median_f1",
            "median_accuracy",
            "midpoint_f1",
            "midpoint_accuracy",
            "within_video_residual_ccc",
            "video_macro_ccc",
        ]
        lines.append(markdown_table(summary[display_columns]))
    else:
        lines.append("- No completed window-level evaluation was found.")
    lines.extend(
        [
            "",
            "## Legacy block-level reference (not a matched test)",
            "",
            "- Legacy rows used participant×video block means, title-bearing CASE",
            "  context, and a two-way train/validation split.",
            "- Window rows use a true train/validation/test split, so this table is",
            "  context only; it must not be read as a paired significance test.",
            "",
        ]
    )
    if len(legacy):
        columns = [
            "condition",
            "target",
            "ccc_mean",
            "ccc_std",
            "f1_mean",
            "accuracy_mean",
        ]
        lines.append(markdown_table(legacy[columns]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_all(
    plan: dict[str, Any],
    *,
    protocols: list[str] | None,
    force: bool,
) -> None:
    runs_root = work_dir(plan) / "runs"
    metadata_paths = sorted(runs_root.rglob("run_metadata.json"))
    allowed_conditions = set(map(str, plan["conditions"]))
    obsolete_runs = []
    selected_runs = []
    all_predictions = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # Candidate-screening and loss-weight sweep checkpoints are selected
        # on validation only. Only run_group=None checkpoints are final test
        # models.
        if metadata.get("run_group"):
            continue
        if protocols and metadata["protocol"] not in protocols:
            continue
        if str(metadata["condition_id"]) not in allowed_conditions:
            obsolete_runs.append(
                {
                    "condition_id": str(metadata["condition_id"]),
                    "protocol": str(metadata["protocol"]),
                    "metadata": str(metadata_path.resolve()),
                }
            )
            continue
        selected_runs.append(
            {
                "condition_id": str(metadata["condition_id"]),
                "protocol": str(metadata["protocol"]),
                "fold_video": metadata.get("fold_video"),
                "seed": int(metadata["seed"]),
                "metadata": str(metadata_path.resolve()),
            }
        )
        print(
            f"[EVAL] {metadata['protocol']} fold={metadata['fold_video']} "
            f"{metadata['condition_id']} seed={metadata['seed']}",
            flush=True,
        )
        _, predictions = evaluate_run(plan, metadata_path, force=force)
        all_predictions.append(predictions)
    reports = work_dir(plan) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_dump(
        reports / "EVALUATION_RUN_FILTER_AUDIT.json",
        {
            "status": "pass",
            "allowed_conditions": sorted(allowed_conditions),
            "protocol_filter": protocols,
            "selected_run_count": len(selected_runs),
            "selected_runs": selected_runs,
            "obsolete_run_count": len(obsolete_runs),
            "obsolete_condition_ids_skipped": sorted(
                {
                    row["condition_id"]
                    for row in obsolete_runs
                }
            ),
            "obsolete_runs": obsolete_runs,
        },
    )
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    if len(predictions):
        predictions.to_csv(
            reports / "public_heldout_predictions.csv", index=False
        )
    summaries = []
    if len(predictions):
        for protocol in sorted(predictions["protocol"].unique()):
            summaries.append(
                _summary_from_predictions(predictions, protocol)
            )
    summary = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame()
    )
    if len(summary):
        summary.to_csv(
            reports / "public_dataset_task_metrics_per_run.csv",
            index=False,
        )
    vac_per_seed, vac_aggregate = _vac_target_summary(summary)
    if len(vac_per_seed):
        vac_per_seed.to_csv(
            reports / "public_vac_metrics_per_seed.csv", index=False
        )
    if len(vac_aggregate):
        vac_aggregate.to_csv(
            reports / "public_vac_metrics_summary.csv", index=False
        )
    _write_vac_report(
        reports / "VAC_CCC_F1_ACCURACY.md",
        vac_per_seed,
        vac_aggregate,
    )
    legacy = _legacy_block_rows(plan)
    legacy.to_csv(reports / "legacy_case_block_reference.csv", index=False)
    _write_report(
        reports / "CASE_WINDOW_RESULTS.md",
        summary,
        legacy,
    )
    print(f"[DONE] report -> {reports/'CASE_WINDOW_RESULTS.md'}")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--protocols", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate_all(
        load_plan(args.plan),
        protocols=args.protocols,
        force=args.force,
    )


if __name__ == "__main__":
    main()
