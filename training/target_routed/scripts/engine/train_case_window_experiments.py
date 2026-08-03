"""Train CASE window-level modality/title ablations on the four-source pool."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from build_case_window_source import build as build_source
from public_context_mapping import (
    adapt_public_manifest,
    attach_public_duration_evidence,
    audit_original7,
)
from case_window_common import (
    DEFAULT_PLAN,
    configure_console,
    dataset_seed,
    json_dump,
    load_plan,
    resolve,
    sha256_file,
    work_dir,
    yaml_dump,
)

PROTOCOLS = (
    "participant_holdout",
    "leave_one_video_out",
    "participant_video_double_holdout",
)

TARGET_COLUMNS = {
    "valence": "valence",
    "arousal": "arousal",
    "cognitive_load": "cognitive_load",
}


def _source_paths(plan: dict[str, Any]) -> dict[str, Path]:
    case_root = (
        resolve(plan, plan["case_source_dir"])
        if plan.get("case_source_dir")
        else work_dir(plan) / "source_data"
    )
    generated_root = work_dir(plan) / "source_data"
    return {
        "no_title": case_root / "case_window_manifest_no_title.csv",
        "with_title": case_root / "case_window_manifest_with_title.csv",
        "ppg": case_root / "case_window_ppg512.csv",
        "splits": case_root / "participant_splits.json",
        "audit": case_root / "source_audit.json",
        "combined_ppg": (
            generated_root / "case_window_three_source_ppg512.csv"
        ),
    }


def _ensure_combined_ppg(
    plan: dict[str, Any], paths: dict[str, Path]
) -> Path:
    output = paths["combined_ppg"]
    if output.is_file():
        return output
    public_manifest = pd.read_csv(
        resolve(plan, plan["public_source"]["manifest"])
    )
    public_ppg = pd.read_csv(
        resolve(plan, plan["public_source"]["ppg_csv"])
    )
    training_datasets = set(
        map(
            str.lower,
            plan.get(
                "training_datasets",
                ["case", "eevr", "maus", "wesad"],
            ),
        )
    )
    public_dataset = public_manifest["dataset"].astype(str).str.lower()
    retained_public_ids = set(
        public_manifest.loc[
            public_dataset.isin(training_datasets - {"case"}),
            "sample_id",
        ].astype(str)
    )
    retained_eevr_ids = set(
        public_manifest.loc[
            public_dataset.eq("eevr"), "sample_id"
        ].astype(str)
    )
    public_ppg = public_ppg[
        public_ppg["sample_id"].astype(str).isin(
            retained_public_ids - retained_eevr_ids
        )
    ]
    eevr_ppg_path = plan["public_source"].get("eevr_ppg_csv")
    if retained_eevr_ids:
        if not eevr_ppg_path:
            raise ValueError(
                "public_source.eevr_ppg_csv is required for v2"
            )
        eevr_ppg = pd.read_csv(resolve(plan, eevr_ppg_path))
        eevr_ppg["sample_id"] = eevr_ppg["sample_id"].astype(str)
        eevr_ppg = eevr_ppg[
            eevr_ppg["sample_id"].isin(retained_eevr_ids)
        ].copy()
        missing_eevr = sorted(
            retained_eevr_ids - set(eevr_ppg["sample_id"])
        )
        unexpected_eevr = sorted(
            set(eevr_ppg["sample_id"]) - retained_eevr_ids
        )
        if missing_eevr or unexpected_eevr:
            raise ValueError(
                "EEVR v4 PPG join failed: "
                f"missing={missing_eevr[:10]} "
                f"unexpected={unexpected_eevr[:10]}"
            )
        feature_columns = [f"ppg_f{i}" for i in range(512)]
        eevr_ppg = eevr_ppg[["sample_id", *feature_columns]]
        public_ppg = pd.concat(
            [public_ppg, eevr_ppg], ignore_index=True
        )
    case_ppg = pd.read_csv(paths["ppg"])
    combined = pd.concat([public_ppg, case_ppg], ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id in combined PPG source")
    feature_columns = [f"ppg_f{i}" for i in range(512)]
    if any(column not in combined for column in feature_columns):
        raise ValueError("combined PPG source is not 512-dimensional")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


def _assign_case_split(
    case: pd.DataFrame,
    protocol: str,
    seed: int,
    fold_video: int | None,
) -> tuple[pd.Series, dict[str, Any]]:
    if protocol == "participant_holdout":
        column = f"split_seed_{seed}"
        if column not in case:
            raise ValueError(f"missing participant split: {column}")
        split = case[column].astype(str)
        return split, {
            "test_participants": sorted(
                case.loc[split.eq("test"), "participant_id"]
                .astype(str)
                .unique()
                .tolist()
            ),
            "validation_participants": sorted(
                case.loc[split.eq("validation"), "participant_id"]
                .astype(str)
                .unique()
                .tolist()
            ),
        }

    if fold_video is None:
        raise ValueError(f"{protocol} requires --fold-video")
    videos = sorted(map(int, case["video_id"].unique()))
    if int(fold_video) not in videos:
        raise ValueError(f"unknown CASE video fold: {fold_video}")
    validation_video = videos[
        (videos.index(int(fold_video)) + 1) % len(videos)
    ]

    if protocol == "leave_one_video_out":
        split = pd.Series("train", index=case.index, dtype=object)
        split.loc[case["video_id"].eq(validation_video)] = "validation"
        split.loc[case["video_id"].eq(int(fold_video))] = "test"
        return split, {
            "test_video": int(fold_video),
            "validation_video": int(validation_video),
            "participant_overlap_expected": True,
        }

    if protocol != "participant_video_double_holdout":
        raise ValueError(f"unknown protocol: {protocol}")
    participant_column = f"split_seed_{seed}"
    participant_role = case[participant_column].astype(str)
    is_test_participant = participant_role.eq("test")
    is_validation_participant = participant_role.eq("validation")
    is_train_participant = participant_role.eq("train")
    is_test_video = case["video_id"].eq(int(fold_video))
    split = pd.Series("excluded", index=case.index, dtype=object)
    split.loc[is_train_participant & ~is_test_video] = "train"
    split.loc[is_validation_participant & ~is_test_video] = "validation"
    split.loc[is_test_participant & is_test_video] = "test"
    return split, {
        "test_video": int(fold_video),
        "test_participants": sorted(
            case.loc[is_test_participant, "participant_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "validation_participants": sorted(
            case.loc[is_validation_participant, "participant_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "train_participants": sorted(
            case.loc[is_train_participant, "participant_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
    }


def _assign_public_validation_split(
    public: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """Create deterministic participant-holdout validation/test per dataset.

    Validation drives checkpoint selection; test participants are sealed for
    final V/A/C reporting and never affect early stopping or CASE-weight
    selection.
    """
    split = pd.Series("train", index=public.index, dtype=object)
    audit: dict[str, Any] = {}
    for dataset, frame in public.groupby(
        public["dataset"].astype(str).str.lower(), sort=True
    ):
        participants = np.asarray(
            sorted(frame["participant_id"].astype(str).unique())
        )
        if len(participants) < 2:
            raise ValueError(
                f"{dataset}: participant validation needs >=2 participants"
            )
        rng = np.random.default_rng(dataset_seed(seed, str(dataset)))
        rng.shuffle(participants)
        n_validation = max(
            1, int(round(len(participants) * validation_fraction))
        )
        n_test = max(
            1, int(round(len(participants) * test_fraction))
        )
        if n_validation + n_test >= len(participants):
            raise ValueError(
                f"{dataset}: validation/test leaves no training participant"
            )
        validation_participants = set(
            participants[:n_validation].astype(str).tolist()
        )
        test_participants = set(
            participants[
                n_validation : n_validation + n_test
            ].astype(str).tolist()
        )
        dataset_mask = (
            public["dataset"].astype(str).str.lower().eq(str(dataset))
        )
        validation_mask = dataset_mask & public["participant_id"].astype(
            str
        ).isin(validation_participants)
        test_mask = dataset_mask & public["participant_id"].astype(
            str
        ).isin(test_participants)
        split.loc[validation_mask] = "validation"
        split.loc[test_mask] = "test"
        train_participants = set(
            public.loc[
                dataset_mask & ~validation_mask & ~test_mask,
                "participant_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        overlap = sorted(
            (train_participants & validation_participants)
            | (train_participants & test_participants)
            | (validation_participants & test_participants)
        )
        if overlap:
            raise ValueError(
                f"{dataset}: public participant split leakage: {overlap}"
            )
        audit[str(dataset)] = {
            "seed": int(dataset_seed(seed, str(dataset))),
            "validation_fraction": float(validation_fraction),
            "test_fraction": float(test_fraction),
            "train_participants": sorted(train_participants),
            "validation_participants": sorted(validation_participants),
            "test_participants": sorted(test_participants),
            "train_rows": int((dataset_mask & split.eq("train")).sum()),
            "validation_rows": int(
                (dataset_mask & split.eq("validation")).sum()
            ),
            "test_rows": int(
                (dataset_mask & split.eq("test")).sum()
            ),
            "participant_overlap": overlap,
        }
    return split, audit


def _validation_task_pairs(frame: pd.DataFrame) -> list[list[str]]:
    pairs: list[list[str]] = []
    dataset_series = frame["dataset"].astype(str).str.lower()
    for dataset in sorted(dataset_series.unique()):
        subset = frame.loc[dataset_series.eq(dataset)]
        for target, column in TARGET_COLUMNS.items():
            if column in subset and pd.to_numeric(
                subset[column], errors="coerce"
            ).notna().any():
                pairs.append([str(dataset), target])
    return pairs


def _experiment_manifest(
    plan: dict[str, Any],
    condition: dict[str, Any],
    protocol: str,
    seed: int,
    fold_video: int | None,
    run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    paths = _source_paths(plan)
    case_path = paths["with_title"] if condition["title"] else paths["no_title"]
    case = pd.read_csv(case_path)
    case["context_content_identity_included"] = int(
        bool(condition["title"])
    )
    case["context_detail_policy"] = (
        "official_content_title_only_no_emotion_category"
        if condition["title"]
        else "generic_protocol_context"
    )
    case["context_detail_source"] = (
        "official_case_content_title"
        if condition["title"]
        else "none_generic_protocol_only"
    )
    case["split"], protocol_detail = _assign_case_split(
        case, protocol, seed, fold_video
    )

    public = pd.read_csv(resolve(plan, plan["public_source"]["manifest"]))
    training_datasets = set(
        map(
            str.lower,
            plan.get(
                "training_datasets",
                ["case", "eevr", "maus", "wesad"],
            ),
        )
    )
    public = public[
        public["dataset"]
        .astype(str)
        .str.lower()
        .isin(training_datasets - {"case"})
    ].copy()
    duration_contract = plan["duration_contract"]
    public, duration_audit = attach_public_duration_evidence(
        public,
        eevr_session_audit=resolve(
            plan, duration_contract["eevr_session_audit"]
        ),
        maus_raw_root=resolve(
            plan, duration_contract["maus_raw_root"]
        ),
        maus_sampling_rate_hz=float(
            duration_contract["maus_sampling_rate_hz"]
        ),
        expected_maus_duration_seconds=float(
            duration_contract["expected_maus_duration_seconds"]
        ),
    )
    context_options = plan.get("public_context", {})
    include_eevr_scene = bool(
        condition.get(
            "include_eevr_scene",
            context_options.get("include_eevr_scene", False),
        )
    )
    include_maus_level = bool(
        condition.get(
            "include_maus_level",
            context_options.get("include_maus_level", False),
        )
    )
    public = adapt_public_manifest(
        public,
        include_case_title=False,
        include_eevr_scene=include_eevr_scene,
        include_maus_level=include_maus_level,
    )
    public_validation_cfg = plan.get("public_validation", {})
    public_split_audit: dict[str, Any] = {}
    if bool(public_validation_cfg.get("enabled", False)):
        public["split"], public_split_audit = (
            _assign_public_validation_split(
                public,
                seed,
                float(
                    public_validation_cfg.get(
                        "fraction",
                        plan.get("split", {}).get(
                            "validation_fraction", 0.15
                        ),
                    )
                ),
                float(
                    public_validation_cfg.get(
                        "test_fraction",
                        plan.get("split", {}).get(
                            "legacy_holdout_fraction", 0.15
                        ),
                    )
                ),
            )
        )
    else:
        public["split"] = "train"
    public["sample_weight"] = 1.0
    public["case_context_condition"] = "not_applicable"

    manifest = pd.concat([public, case], ignore_index=True, sort=False)
    manifest["context_mode"] = str(
        condition.get("context_mode", "unspecified")
    )
    schema_audit = audit_original7(manifest)
    if schema_audit["status"] != "pass":
        raise ValueError(
            f"original-7 training schema audit failed: {schema_audit}"
        )
    manifest_path = run_dir / "training_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    train = manifest[manifest["split"].eq("train")]
    validation = manifest[manifest["split"].eq("validation")]
    test = manifest[
        manifest["split"].eq("test")
        & manifest["dataset"].astype(str).str.lower().eq("case")
    ]
    public_test = manifest[
        manifest["split"].eq("test")
        & ~manifest["dataset"].astype(str).str.lower().eq("case")
    ]
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError(
            f"empty explicit split: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}"
        )
    validation_task_pairs = _validation_task_pairs(validation)
    expected_validation_tasks = [
        [str(dataset).lower(), str(target).lower()]
        for dataset, target in public_validation_cfg.get(
            "expected_checkpoint_tasks", []
        )
    ]
    if expected_validation_tasks:
        expected_validation_tasks = sorted(expected_validation_tasks)
        if sorted(validation_task_pairs) != expected_validation_tasks:
            raise ValueError(
                "checkpoint validation tasks differ from the declared "
                f"contract: actual={sorted(validation_task_pairs)} "
                f"expected={expected_validation_tasks}"
            )
    case_train = train[
        train["dataset"].astype(str).str.lower().eq("case")
    ]
    case_train_videos = sorted(
        map(int, case_train["video_id"].dropna().unique())
    )
    case_test_videos = sorted(
        map(int, test["video_id"].dropna().unique())
    )
    eevr_rows = manifest[
        manifest["dataset"].astype(str).str.lower().eq("eevr")
    ]
    eevr_train_scenes = sorted(
        eevr_rows.loc[
            eevr_rows["split"].eq("train"),
            "source_context_sig_before_v2",
        ].astype(str).unique()
    )
    eevr_test_scenes = sorted(
        eevr_rows.loc[
            eevr_rows["split"].eq("test"),
            "source_context_sig_before_v2",
        ].astype(str).unique()
    )
    case_detail_count = len(
        {
            value
            for value in (
                case["context7_observable_detail"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            if value
        }
    )
    eevr_detail_count = len(
        {
            value
            for value in (
                eevr_rows["context7_observable_detail"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            if value
        }
    )
    maus_rows = manifest[
        manifest["dataset"].astype(str).str.lower().eq("maus")
    ]
    maus_detail_count = len(
        {
            value
            for value in (
                maus_rows["context7_observable_detail"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            if value
        }
    )
    expected_case_details = 8 if bool(condition["title"]) else 0
    expected_eevr_details = (
        int(eevr_rows["source_context_sig_before_v2"].nunique())
        if include_eevr_scene
        else 0
    )
    expected_maus_details = 3 if include_maus_level else 0
    actual_detail_counts = {
        "case": case_detail_count,
        "eevr": eevr_detail_count,
        "maus": maus_detail_count,
    }
    expected_detail_counts = {
        "case": expected_case_details,
        "eevr": expected_eevr_details,
        "maus": expected_maus_details,
    }
    if actual_detail_counts != expected_detail_counts:
        raise ValueError(
            "observable content detail contract failed: "
            f"actual={actual_detail_counts}, "
            f"expected={expected_detail_counts}"
        )
    eevr_teacher_exact_context_matches = 0
    if "teacher_text" in eevr_rows:
        teacher = (
            eevr_rows["teacher_text"].fillna("").astype(str).str.strip()
        )
        context = (
            eevr_rows["context_text"].fillna("").astype(str).str.strip()
        )
        eevr_teacher_exact_context_matches = int(
            (teacher.ne("") & teacher.eq(context)).sum()
        )
        if eevr_teacher_exact_context_matches:
            raise ValueError(
                "EEVR post-exposure teacher text leaked into state context"
            )
    if protocol == "participant_holdout":
        if case_train_videos != case_test_videos:
            raise ValueError(
                "operational participant holdout must reuse the same "
                "CASE content identities"
            )
        if not set(eevr_test_scenes).issubset(eevr_train_scenes):
            raise ValueError(
                "operational EEVR test contains a scene absent from train"
            )
        content_interpretation = "known_content_unseen_participant"
    else:
        if set(case_train_videos) & set(case_test_videos):
            raise ValueError(
                "CASE leave-one-video-out has train/test content overlap"
            )
        content_interpretation = "unseen_case_video_diagnostic"
    dataset_weight_sums = (
        train.groupby("dataset")["sample_weight"].sum().to_dict()
    )
    dataset_count = len(dataset_weight_sums)
    source_counts = (
        train["dataset"].astype(str).str.lower().value_counts().to_dict()
    )
    sampling_cfg = plan["training"].get("sampling", {})
    sampling_strategy = str(
        sampling_cfg.get("strategy", "equal_source_replacement")
    )
    if sampling_strategy == "capped_source_repeat":
        repeats = {
            str(key).lower(): int(value)
            for key, value in sampling_cfg[
                "source_repeats_per_epoch"
            ].items()
        }
        missing_repeats = sorted(set(source_counts) - set(repeats))
        invalid_repeats = {
            key: value
            for key, value in repeats.items()
            if value < 1
        }
        if missing_repeats or invalid_repeats:
            raise ValueError(
                "invalid capped-repeat contract: "
                f"missing={missing_repeats}, invalid={invalid_repeats}"
            )
        expected_draws = {
            dataset: int(count) * repeats[dataset]
            for dataset, count in source_counts.items()
        }
        total_draws = sum(expected_draws.values())
        expected_sampler_mass = {
            dataset: draws / total_draws
            for dataset, draws in expected_draws.items()
        }
    else:
        repeats = None
        expected_draws = None
        expected_sampler_mass = {
            str(dataset): 1.0 / dataset_count
            for dataset in dataset_weight_sums
        }
    audit = {
        "protocol": protocol,
        "seed": int(seed),
        "fold_video": int(fold_video) if fold_video is not None else None,
        "title": bool(condition["title"]),
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test_case": int(len(test)),
            "test_public": int(len(public_test)),
            "excluded": int(manifest["split"].eq("excluded").sum()),
        },
        "participants": {
            split: sorted(
                manifest.loc[
                    manifest["split"].eq(split)
                    & manifest["dataset"].astype(str).str.lower().eq("case"),
                    "participant_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            for split in ("train", "validation", "test")
        },
        "videos": {
            split: sorted(
                map(
                    int,
                    manifest.loc[
                        manifest["split"].eq(split)
                        & manifest["dataset"].astype(str).str.lower().eq("case"),
                        "video_id",
                    ]
                    .dropna()
                    .unique()
                    .tolist(),
                )
            )
            for split in ("train", "validation", "test")
        },
        "case_train_participant_video_weight_sums": {
            "min": float(
                case_train.groupby(["participant_id", "video_id"])[
                    "sample_weight"
                ].sum().min()
            ),
            "max": float(
                case_train.groupby(["participant_id", "video_id"])[
                    "sample_weight"
                ].sum().max()
            ),
        },
        "dataset_raw_weight_sums": {
            str(key): float(value)
            for key, value in dataset_weight_sums.items()
        },
        "expected_sampler_dataset_mass": expected_sampler_mass,
        "expected_case_batch_fraction": expected_sampler_mass.get("case"),
        "sampling_strategy": sampling_strategy,
        "source_rows": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "source_repeats_per_epoch": repeats,
        "source_draws_per_epoch": expected_draws,
        "content_evaluation": {
            "context_mode": str(
                condition.get("context_mode", "unspecified")
            ),
            "case_title_included": bool(condition["title"]),
            "eevr_scene_included": include_eevr_scene,
            "maus_level_included": include_maus_level,
            "interpretation": content_interpretation,
            "case_train_video_ids": case_train_videos,
            "case_test_video_ids": case_test_videos,
            "case_train_test_video_overlap": sorted(
                set(case_train_videos) & set(case_test_videos)
            ),
            "eevr_train_scene_count": len(eevr_train_scenes),
            "eevr_test_scene_count": len(eevr_test_scenes),
            "eevr_test_scenes_absent_from_train": sorted(
                set(eevr_test_scenes) - set(eevr_train_scenes)
            ),
            "observable_detail_counts": actual_detail_counts,
            "expected_observable_detail_counts": expected_detail_counts,
            "context_detail_sources": sorted(
                manifest["context_detail_source"]
                .fillna("")
                .astype(str)
                .unique()
                .tolist()
            ),
            "eevr_teacher_exact_context_matches": (
                eevr_teacher_exact_context_matches
            ),
        },
        "public_participant_validation": public_split_audit,
        "duration_contract": duration_audit,
        "context_schema_audit": schema_audit,
        "checkpoint_validation_tasks": validation_task_pairs,
        "expected_checkpoint_validation_tasks": (
            expected_validation_tasks or None
        ),
        "protocol_detail": protocol_detail,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    audit["maximum_allowed_source_repeats"] = (
        max(repeats.values()) if repeats else None
    )
    json_dump(run_dir / "split_and_balance_audit.json", audit)
    return manifest_path, audit


def _make_config(
    plan: dict[str, Any],
    condition_id: str,
    condition: dict[str, Any],
    seed: int,
    manifest_path: Path,
    ppg_path: Path,
    run_dir: Path,
    *,
    smoke: bool,
    case_loss_weight: float | None = None,
    ppg_backbone_checkpoint: Path | None = None,
    full_init_checkpoint: Path | None = None,
    freeze_parameter_prefixes: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    cfg = yaml.safe_load(
        resolve(plan, plan["base_config"]).read_text(encoding="utf-8")
    )
    training = plan["training"]
    cfg["seed"] = int(seed)
    cfg["paths"]["manifest_csv"] = str(manifest_path.resolve())
    cfg["paths"]["ppg_csv"] = str(ppg_path.resolve())
    cfg["paths"]["out_dir"] = str(run_dir.resolve())
    cfg["model"]["text_init_ckpt"] = str(
        resolve(plan, plan["text_stage1_checkpoint"])
    )
    cfg["model"]["fusion_mode"] = str(
        condition.get(
            "fusion_mode",
            training.get("fusion_mode", "concat"),
        )
    )
    cfg["model"]["ppg_backbone_init_checkpoint"] = (
        str(ppg_backbone_checkpoint.resolve())
        if ppg_backbone_checkpoint is not None
        else None
    )
    cfg["model"]["full_init_checkpoint"] = (
        str(full_init_checkpoint.resolve())
        if full_init_checkpoint is not None
        else None
    )
    cfg["model"]["freeze_parameter_prefixes"] = [
        str(value) for value in freeze_parameter_prefixes
    ]
    cfg["loss"]["lambda_align"] = (
        float(training["clsp_lambda"]) if condition["clsp"] else 0.0
    )
    cfg["loss"]["clsp_mode"] = "domain_safe"
    cfg["loss"]["semantic_positive_weight"] = 0.0
    cfg["loss"]["teacher_encoder"] = (
        "frozen_copy" if condition["clsp"] else "shared"
    )
    cfg["loss"]["teacher_stop_gradient"] = True
    dataset_loss_weights = {
        str(key): float(value)
        for key, value in training["dataset_loss_weights"].items()
    }
    if case_loss_weight is not None:
        dataset_loss_weights["case"] = float(case_loss_weight)
    cfg["train"].update(
        {
            "batch_size": int(training["batch_size"]),
            "epochs": 1 if smoke else int(training["epochs"]),
            "learning_rate": float(training["learning_rate"]),
            "text_encoder_learning_rate": float(
                training["text_encoder_learning_rate"]
            ),
            "ppg_backbone_learning_rate": float(
                training.get(
                    "ppg_only_learning_rate",
                    training["learning_rate"],
                )
                if condition_id == "ppg_only"
                and ppg_backbone_checkpoint is None
                else training.get(
                    "ppg_backbone_learning_rate",
                    training["learning_rate"],
                )
            ),
            "warmup_freeze_ppg_epochs": (
                int(
                    training.get(
                        "warmup_freeze_ppg_epochs", 0
                    )
                )
                if ppg_backbone_checkpoint is not None
                else 0
            ),
            "weight_decay": float(training["weight_decay"]),
            "modality_dropout": (
                float(training["modality_dropout"])
                if not (
                    condition["mute_text"] or condition["mute_ppg"]
                )
                else 0.0
            ),
            "early_stop": True,
            "early_stop_metric": "macro_ccc",
            "macro_metric_aggregation": str(
                training.get(
                    "macro_metric_aggregation", "task_balanced"
                )
            ),
            "selection_align_weight": 0.0,
            "patience": 1 if smoke else int(training["patience"]),
            "dataset_loss_weights": dataset_loss_weights,
            "samples_per_epoch": int(
                training.get("samples_per_epoch", 0)
            )
            or None,
            "split_column": "split",
            "train_split_value": "train",
            "validation_split_value": "validation",
            "save_teacher_encoder": False,
        }
    )
    sampling = training.get("sampling", {})
    if sampling:
        cfg["train"]["sampling_strategy"] = str(
            sampling["strategy"]
        )
        cfg["train"]["source_repeats_per_epoch"] = {
            str(key): int(value)
            for key, value in sampling.get(
                "source_repeats_per_epoch", {}
            ).items()
        }
    cfg["model"]["posture_film"] = {
        "enabled": False,
        "residual_scale": 0.1,
        "init_checkpoint": None,
        "freeze": True,
    }
    cfg["case_window_condition"] = {
        "condition_id": condition_id,
        **condition,
    }
    cfg["context_schema_version"] = str(plan["context_schema_version"])
    if cfg["train"]["samples_per_epoch"] is None:
        cfg["train"].pop("samples_per_epoch")
    return cfg


def _run_dir(
    plan: dict[str, Any],
    protocol: str,
    condition_id: str,
    seed: int,
    fold_video: int | None,
    smoke: bool,
    run_group: str | None = None,
) -> Path:
    root = work_dir(plan) / ("smoke" if smoke else "runs") / protocol
    if fold_video is not None:
        root = root / f"video_{int(fold_video)}"
    if run_group:
        root = root / run_group
    return root / condition_id / f"seed_{int(seed)}"


def train_one(
    plan: dict[str, Any],
    protocol: str,
    condition_id: str,
    seed: int,
    fold_video: int | None,
    *,
    force: bool,
    smoke: bool,
    case_loss_weight: float | None = None,
    run_group: str | None = None,
    ppg_backbone_checkpoint: Path | None = None,
    full_init_checkpoint: Path | None = None,
    freeze_parameter_prefixes: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    condition = plan["conditions"][condition_id]
    requested_case_weight = float(
        case_loss_weight
        if case_loss_weight is not None
        else plan["training"]["dataset_loss_weights"]["case"]
    )
    run_dir = _run_dir(
        plan,
        protocol,
        condition_id,
        seed,
        fold_video,
        smoke,
        run_group,
    )
    checkpoint = run_dir / "model_v3" / "full.pt"
    metadata_path = run_dir / "run_metadata.json"
    if checkpoint.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested_backbone = (
            str(ppg_backbone_checkpoint.resolve())
            if ppg_backbone_checkpoint is not None
            else None
        )
        requested_full_init = (
            str(full_init_checkpoint.resolve())
            if full_init_checkpoint is not None
            else None
        )
        requested_freeze_prefixes = [
            str(value) for value in freeze_parameter_prefixes
        ]
        reusable = (
            metadata.get("status") == "complete"
            and metadata.get("condition") == condition
            and metadata.get("context_schema_version")
            == plan["context_schema_version"]
            and metadata.get("public_context")
            == plan.get("public_context")
            and metadata.get("fusion_mode", "concat")
            == str(
                condition.get(
                    "fusion_mode",
                    plan["training"].get("fusion_mode", "concat"),
                )
            )
            and metadata.get("ppg_backbone_init_checkpoint")
            == requested_backbone
            and metadata.get("full_init_checkpoint")
            == requested_full_init
            and metadata.get("freeze_parameter_prefixes", [])
            == requested_freeze_prefixes
            and abs(
                float(metadata.get("case_loss_weight", float("nan")))
                - requested_case_weight
            )
            <= 1e-12
        )
        if reusable:
            print(
                f"[SKIP] {protocol}/{fold_video}/{condition_id}/seed_{seed}"
            )
            return metadata

    paths = _source_paths(plan)
    if not paths["audit"].is_file():
        build_source(plan)
    combined_ppg = _ensure_combined_ppg(plan, paths)
    manifest_path, split_audit = _experiment_manifest(
        plan,
        condition,
        protocol,
        int(seed),
        fold_video,
        run_dir,
    )
    cfg = _make_config(
        plan,
        condition_id,
        condition,
        int(seed),
        manifest_path,
        combined_ppg,
        run_dir,
        smoke=smoke,
        case_loss_weight=case_loss_weight,
        ppg_backbone_checkpoint=ppg_backbone_checkpoint,
        full_init_checkpoint=full_init_checkpoint,
        freeze_parameter_prefixes=freeze_parameter_prefixes,
    )
    yaml_dump(run_dir / "config.yaml", cfg)

    model_dir = resolve(plan, plan["model_dir"])
    sys.path.insert(0, str(model_dir))
    from train import train_model

    print(
        f"[TRAIN] protocol={protocol} fold={fold_video} "
        f"condition={condition_id} seed={seed}",
        flush=True,
    )
    model, device = train_model(
        cfg,
        verbose=True,
        mute_text=bool(condition["mute_text"]),
        mute_ppg=bool(condition["mute_ppg"]),
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(state, checkpoint)
    metrics = {
        "summary": getattr(model, "training_summary", {}),
        "epochs": getattr(model, "training_history", []),
    }
    json_dump(checkpoint.parent / "training_metrics.json", metrics)
    contract = getattr(model, "training_input_contract", None)
    if contract is not None:
        json_dump(checkpoint.parent / "training_input_contract.json", contract)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "protocol": protocol,
        "fold_video": int(fold_video) if fold_video is not None else None,
        "condition_id": condition_id,
        "condition": condition,
        "context_schema_version": plan["context_schema_version"],
        "public_context": plan.get("public_context"),
        "run_group": run_group,
        "case_loss_weight": float(
            cfg["train"]["dataset_loss_weights"]["case"]
        ),
        "fusion_mode": cfg["model"].get("fusion_mode", "concat"),
        "ppg_backbone_init_checkpoint": (
            str(ppg_backbone_checkpoint.resolve())
            if ppg_backbone_checkpoint is not None
            else None
        ),
        "full_init_checkpoint": (
            str(full_init_checkpoint.resolve())
            if full_init_checkpoint is not None
            else None
        ),
        "freeze_parameter_prefixes": [
            str(value) for value in freeze_parameter_prefixes
        ],
        "seed": int(seed),
        "device": str(device),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str((run_dir / "config.yaml").resolve()),
        "manifest": str(manifest_path.resolve()),
        "ppg_csv": str(combined_ppg.resolve()),
        "best_epoch": metrics["summary"].get("best_epoch"),
        "validation_macro_ccc": metrics["summary"].get(
            "best_selection_value"
        ),
        "split_and_balance_audit": split_audit,
    }
    json_dump(metadata_path, metadata)
    print(
        f"[DONE] {condition_id}/seed_{seed}: "
        f"best_epoch={metadata['best_epoch']} "
        f"val_CCC={metadata['validation_macro_ccc']}",
        flush=True,
    )
    del state, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def run(
    plan: dict[str, Any],
    protocols: list[str],
    conditions: list[str],
    seeds: list[int] | None,
    folds: list[int] | None,
    *,
    force: bool,
    smoke: bool,
    case_loss_weight: float | None = None,
    run_group: str | None = None,
) -> None:
    unknown_protocols = sorted(set(protocols) - set(PROTOCOLS))
    if unknown_protocols:
        raise ValueError(f"unknown protocols: {unknown_protocols}")
    unknown_conditions = sorted(set(conditions) - set(plan["conditions"]))
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {unknown_conditions}")
    videos = list(map(int, plan["source"]["content_video_ids"]))
    for protocol in protocols:
        selected_seeds = seeds or list(
            map(
                int,
                plan["execution"][
                    {
                        "participant_holdout": "participant_holdout_seeds",
                        "leave_one_video_out": "leave_one_video_out_seeds",
                        "participant_video_double_holdout": (
                            "participant_video_double_holdout_seeds"
                        ),
                    }[protocol]
                ],
            )
        )
        selected_folds: list[int | None] = (
            [None]
            if protocol == "participant_holdout"
            else list(map(int, folds or videos))
        )
        for fold_video in selected_folds:
            for condition_id in conditions:
                for seed in selected_seeds:
                    train_one(
                        plan,
                        protocol,
                        condition_id,
                        int(seed),
                        fold_video,
                        force=force,
                        smoke=smoke,
                        case_loss_weight=case_loss_weight,
                        run_group=run_group,
                    )


def _case_weight_tag(value: float) -> str:
    return f"case_weight_{float(value):.2f}".replace(".", "p")


def select_case_loss_weight(
    plan: dict[str, Any],
    seeds: list[int] | None,
    *,
    force: bool,
    smoke: bool,
) -> float:
    """Train the main fusion condition at three CASE weights and select on val."""
    sweep = plan.get("case_weight_sweep", {})
    if not bool(sweep.get("enabled", False)):
        return float(plan["training"]["dataset_loss_weights"]["case"])
    values = [float(value) for value in sweep["values"]]
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError("CASE weight sweep must contain three unique values")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("CASE weight candidates must be finite and positive")
    protocol = str(
        sweep.get("selection_protocol", "participant_holdout")
    )
    if protocol != "participant_holdout":
        raise ValueError(
            "CASE weight selection is fixed to participant_holdout"
        )
    condition_id = str(sweep["selection_condition"])
    selected_seeds = seeds or list(
        map(
            int,
            plan["execution"]["participant_holdout_seeds"],
        )
    )
    rows: list[dict[str, Any]] = []
    for value in values:
        run_group = f"case_weight_sweep/{_case_weight_tag(value)}"
        for seed in selected_seeds:
            metadata = train_one(
                plan,
                protocol,
                condition_id,
                int(seed),
                None,
                force=force,
                smoke=smoke,
                case_loss_weight=value,
                run_group=run_group,
            )
            score = metadata.get("validation_macro_ccc")
            if score is None or not np.isfinite(float(score)):
                raise ValueError(
                    f"Missing validation CCC for CASE weight {value}, "
                    f"seed {seed}"
                )
            rows.append(
                {
                    "case_loss_weight": value,
                    "seed": int(seed),
                    "target_balanced_validation_ccc": float(score),
                    "checkpoint": metadata["checkpoint"],
                }
            )
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("case_loss_weight")[
            "target_balanced_validation_ccc"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(
            ["mean", "case_loss_weight"],
            ascending=[False, True],
            kind="stable",
        )
    )
    selected = float(summary.iloc[0]["case_loss_weight"])
    output = {
        "status": "complete",
        "selection_data": "validation_only",
        "selection_metric": (
            "seed_mean_target_balanced_validation_ccc"
        ),
        "tie_break": "lower_case_weight",
        "candidates": values,
        "selected_case_loss_weight": selected,
        "runs": rows,
        "summary": summary.to_dict(orient="records"),
        "test_metrics_used": False,
    }
    destination = (
        work_dir(plan)
        / ("smoke" if smoke else "runs")
        / "case_weight_selection.json"
    )
    json_dump(destination, output)
    print(
        f"[CASE WEIGHT] selected={selected:.2f} from "
        f"{destination}",
        flush=True,
    )
    return selected


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--protocols", nargs="+", choices=PROTOCOLS,
        default=["participant_holdout"],
    )
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument(
        "--text-stage1-checkpoint",
        type=Path,
        help="Optional checkpoint override for isolated smoke validation.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--select-case-weight",
        action="store_true",
        help=(
            "Run the three-candidate validation sweep first, then use the "
            "selected CASE loss weight for the requested conditions."
        ),
    )
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.text_stage1_checkpoint:
        plan["text_stage1_checkpoint"] = str(
            args.text_stage1_checkpoint.expanduser().resolve()
        )
    selected_weight = None
    if args.select_case_weight:
        selected_weight = select_case_loss_weight(
            plan,
            args.seeds,
            force=args.force,
            smoke=args.smoke,
        )
    run(
        plan,
        args.protocols,
        args.conditions or list(plan["conditions"]),
        args.seeds,
        args.folds,
        force=args.force,
        smoke=args.smoke,
        case_loss_weight=selected_weight,
    )


if __name__ == "__main__":
    main()
