"""v3 training — partial-label masked MSE + SupCon-CLSP + modality dropout.

teacher-text vs context-text 역할 분리:
  - CLSP 정렬은 clsp_mask가 켜진 EEVR/MAUS의 teacher_text에만 적용한다.
    배치 계산을 위한 빈 자리에는 context_text가 들어가도 clsp_mask=0이라
    loss와 gradient에 기여하지 않는다.
  - 회귀 예측(fusion)은 항상 context_text 입력 (배포와 동일).
  → EEVR 자유서술/MAUS 행동 teacher는 정렬에만 기여하고 추론엔 안 들어감.

단독 실행: python train.py --config config.yaml
LODO에서 호출: train_model(cfg, exclude_datasets=[D]) → 반환된 모델을 D로 평가

Rescue opt-ins preserve legacy defaults: capped source repeats, target-balanced
CCC selection, PPG-only warm start, and staged residual-backbone fine-tuning.
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from alignment import (
    cross_modal_alignment_metrics,
    cross_modal_domain_retrieval_metrics,
)
from dataset import MultiDatasetAffect, collate
from input_contract import build_input_contract
from losses import (
    dataset_balanced_mse,
    domain_safe_clsp,
    partial_masked_mse,
    supcon_clsp,
)
from model import MultimodalStateEstimator, TARGETS
from splits import participant_validation_indices, split_summary


class CappedDatasetRepeatSampler(Sampler[int]):
    """Repeat source rows a declared, hard-bounded number of times per epoch."""

    def __init__(
        self,
        datasets,
        repeats: dict[str, int],
        seed: int,
    ):
        self.datasets = np.asarray(
            [str(value).lower() for value in datasets],
            dtype=object,
        )
        self.repeats = {
            str(key).lower(): int(value)
            for key, value in repeats.items()
        }
        self.seed = int(seed)
        self.epoch = 0
        available = sorted(set(self.datasets.tolist()))
        missing = sorted(set(available) - set(self.repeats))
        invalid = {
            key: value
            for key, value in self.repeats.items()
            if value < 1
        }
        if missing or invalid:
            raise ValueError(
                "capped repeat sampler requires a positive integer for "
                f"every source: missing={missing}, invalid={invalid}"
            )
        self.length = int(
            sum(
                int(np.sum(self.datasets == dataset))
                * self.repeats[dataset]
                for dataset in available
            )
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        indices: list[int] = []
        for dataset in sorted(set(self.datasets.tolist())):
            source = np.flatnonzero(self.datasets == dataset)
            for _ in range(self.repeats[dataset]):
                repeated = source.copy()
                rng.shuffle(repeated)
                indices.extend(repeated.tolist())
        order = np.asarray(indices, dtype=np.int64)
        rng.shuffle(order)
        return iter(order.tolist())

    def __len__(self) -> int:
        return self.length


def seed_everything(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _device(cfg):
    d = cfg.get("device", "cuda")
    return torch.device("cpu" if (d == "cuda" and not torch.cuda.is_available()) else d)


def _ccc(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if len(prediction) < 2:
        return float("nan")
    denominator = (
        prediction.var()
        + truth.var()
        + (prediction.mean() - truth.mean()) ** 2
    )
    if denominator <= 1e-12:
        return 0.0
    covariance = (
        (prediction - prediction.mean()) * (truth - truth.mean())
    ).mean()
    return float(2.0 * covariance / denominator)


def _binary_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float]:
    threshold = float(np.median(truth))
    predicted = np.asarray(prediction) >= threshold
    expected = np.asarray(truth) >= threshold
    accuracy = float(np.mean(predicted == expected))
    true_positive = int(np.sum(predicted & expected))
    false_positive = int(np.sum(predicted & ~expected))
    false_negative = int(np.sum(~predicted & expected))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return accuracy, f1


def _macro_state_metrics(
    predictions: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    datasets: list[str],
    aggregation: str = "task_balanced",
) -> dict:
    """Aggregate validation metrics without changing legacy defaults.

    ``task_balanced`` gives every dataset×target task equal weight and
    reproduces completed runs. ``target_balanced`` first averages datasets
    inside V, A, and C, then gives every available target equal weight.
    """
    aggregation = str(aggregation).strip().lower()
    if aggregation not in {"task_balanced", "target_balanced"}:
        raise ValueError(
            "macro metric aggregation must be 'task_balanced' or "
            "'target_balanced'"
        )
    dataset_array = np.asarray(datasets, dtype=object)
    rows = []
    for dataset in sorted(set(datasets)):
        in_dataset = dataset_array == dataset
        for target in TARGETS:
            valid = in_dataset & np.asarray(masks[target], dtype=bool)
            if int(valid.sum()) < 3:
                continue
            prediction = np.asarray(predictions[target])[valid]
            truth = np.asarray(truths[target])[valid]
            accuracy, f1 = _binary_metrics(prediction, truth)
            rows.append(
                {
                    "dataset": str(dataset),
                    "target": str(target),
                    "ccc": _ccc(prediction, truth),
                    "accuracy": accuracy,
                    "f1": f1,
                }
            )
    if not rows:
        return {
            "tasks": 0,
            "ccc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "aggregation": aggregation,
            "targets": {},
        }
    target_rows = {}
    for target in TARGETS:
        selected = [row for row in rows if row["target"] == target]
        if not selected:
            continue
        target_rows[target] = {
            "tasks": int(len(selected)),
            "datasets": [row["dataset"] for row in selected],
            "ccc": float(
                np.nanmean([row["ccc"] for row in selected])
            ),
            "accuracy": float(
                np.mean([row["accuracy"] for row in selected])
            ),
            "f1": float(np.mean([row["f1"] for row in selected])),
        }
    aggregate_rows = (
        list(target_rows.values())
        if aggregation == "target_balanced"
        else rows
    )
    return {
        "tasks": int(len(rows)),
        "target_count": int(len(target_rows)),
        "ccc": float(
            np.nanmean([row["ccc"] for row in aggregate_rows])
        ),
        "accuracy": float(
            np.mean([row["accuracy"] for row in aggregate_rows])
        ),
        "f1": float(
            np.mean([row["f1"] for row in aggregate_rows])
        ),
        "aggregation": aggregation,
        "targets": target_rows,
    }


def _clsp_loss(
    mode: str,
    text_z: torch.Tensor,
    ppg_z: torch.Tensor,
    batch: dict,
    valid: torch.Tensor,
    temperature: float,
    semantic_positive_weight: float,
) -> torch.Tensor:
    if mode == "legacy_supcon":
        return supcon_clsp(
            text_z,
            ppg_z,
            batch["context_sig"],
            valid,
            temperature=temperature,
        )
    if mode == "domain_safe":
        return domain_safe_clsp(
            text_z,
            ppg_z,
            batch["clsp_pair_id"],
            batch["clsp_negative_mask_sig"],
            batch["clsp_semantic_sig"],
            batch["dataset"],
            valid,
            temperature=temperature,
            semantic_positive_weight=semantic_positive_weight,
            participant_ids=batch.get("participant_id"),
            session_ids=batch.get("session_id"),
            label_group_sigs=batch.get("label_group_sig"),
            window_indices=batch.get("window_index"),
            adjacent_radii=batch.get("negative_adjacent_radius"),
            loss_groups=batch.get("clsp_loss_group"),
        )
    raise ValueError(
        "loss.clsp_mode must be 'legacy_supcon' or 'domain_safe'"
    )


def train_model(cfg, exclude_datasets=None, include_datasets=None, verbose=True,
                mute_text=False, mute_ppg=False):
    device = _device(cfg)
    seed_everything(int(cfg.get("seed", 42)))
    input_contract = build_input_contract(cfg)

    ds = MultiDatasetAffect(
        manifest_csv=cfg["paths"]["manifest_csv"],
        ppg_csv=cfg["paths"].get("ppg_csv"),
        ppg_dim=int(cfg["model"].get("ppg_input_dim", 512)),
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
    )
    # early stopping: participant 단위 val holdout (opt-in via cfg)
    # Baselines require the same participant-validation checkpoint selection as
    # multimodal models. Historically this was disabled for muted modalities.
    early = bool(cfg["train"].get("early_stop", False))
    tr_idx = np.arange(len(ds)); val_loader = None
    validation_idx = np.asarray([], dtype=int)
    validation_audit = {}
    if early:
        split_column = cfg["train"].get("split_column")
        if split_column:
            if split_column not in ds.df.columns:
                raise ValueError(
                    f"explicit split column is missing: {split_column}"
                )
            split_values = (
                ds.df[split_column].fillna("").astype(str).str.lower()
            )
            train_value = str(
                cfg["train"].get("train_split_value", "train")
            ).lower()
            validation_value = str(
                cfg["train"].get(
                    "validation_split_value", "validation"
                )
            ).lower()
            tr_idx = np.flatnonzero(
                split_values.to_numpy() == train_value
            )
            validation_idx = np.flatnonzero(
                split_values.to_numpy() == validation_value
            )
            if len(tr_idx) == 0 or len(validation_idx) == 0:
                raise ValueError(
                    "explicit split must contain non-empty train and "
                    "validation rows"
                )
            train_participants = set(
                ds.df.iloc[tr_idx]["participant_id"].astype(str)
            )
            validation_participants = set(
                ds.df.iloc[validation_idx]["participant_id"].astype(str)
            )
            validation_audit = {
                "strategy": "explicit_column",
                "column": str(split_column),
                "train_rows": int(len(tr_idx)),
                "validation_rows": int(len(validation_idx)),
                "excluded_rows": int(
                    len(ds) - len(tr_idx) - len(validation_idx)
                ),
                "train_participants": int(len(train_participants)),
                "validation_participants": int(
                    len(validation_participants)
                ),
                "participant_overlap": sorted(
                    train_participants & validation_participants
                ),
            }
        else:
            validation_idx = participant_validation_indices(ds.df, cfg)
            is_val = np.zeros(len(ds), dtype=bool)
            is_val[validation_idx] = True
            tr_idx = np.where(~is_val)[0]
            validation_audit = split_summary(ds.df, validation_idx)
        val_loader = DataLoader(torch.utils.data.Subset(ds, validation_idx.tolist()),
                                batch_size=64, shuffle=False, collate_fn=collate)

    # Dataset-balanced sampling. Each source receives equal expected sampler
    # mass, while rows are sampled uniformly inside that source. Optional
    # per-row weights are applied once in the regression loss below; using
    # them here as well would square CASE's participant×video correction.
    dtr = ds.df.iloc[tr_idx]
    row_weight = pd.to_numeric(
        dtr.get("sample_weight", 1.0), errors="coerce"
    )
    if np.isscalar(row_weight):
        row_weight = pd.Series(
            np.full(len(dtr), float(row_weight)),
            index=dtr.index,
        )
    if (
        row_weight.isna().any()
        or (row_weight < 0).any()
        or not np.isfinite(row_weight.to_numpy()).all()
    ):
        raise ValueError(
            "sample_weight must be finite and non-negative"
        )
    dataset_names = dtr["dataset"].astype(str).str.lower()
    counts = dataset_names.value_counts().to_dict()
    sampling_strategy = str(
        cfg["train"].get("sampling_strategy", "equal_source_replacement")
    ).strip().lower()
    if sampling_strategy == "equal_source_replacement":
        w = np.asarray(
            [
                1.0 / counts[str(dataset).lower()]
                for dataset in dataset_names
            ],
            dtype=float,
        )
        samples_per_epoch = int(
            cfg["train"].get("samples_per_epoch", len(tr_idx))
        )
        if samples_per_epoch <= 0:
            raise ValueError("train.samples_per_epoch must be positive")
        sampler = torch.utils.data.WeightedRandomSampler(
            w,
            num_samples=samples_per_epoch,
            replacement=True,
        )
        sampler_audit = {
            "strategy": sampling_strategy,
            "source_rows": {
                str(key): int(value) for key, value in counts.items()
            },
            "samples_per_epoch": samples_per_epoch,
            "hard_repeat_cap": False,
        }
    elif sampling_strategy == "capped_source_repeat":
        configured_repeats = {
            str(key).lower(): int(value)
            for key, value in cfg["train"].get(
                "source_repeats_per_epoch", {}
            ).items()
        }
        sampler = CappedDatasetRepeatSampler(
            dataset_names.tolist(),
            configured_repeats,
            seed=int(cfg.get("seed", 42)),
        )
        samples_per_epoch = len(sampler)
        sampler_audit = {
            "strategy": sampling_strategy,
            "source_rows": {
                str(key): int(value) for key, value in counts.items()
            },
            "source_repeats_per_epoch": {
                str(key): int(value)
                for key, value in configured_repeats.items()
            },
            "source_draws_per_epoch": {
                str(dataset): int(counts[dataset])
                * int(configured_repeats[str(dataset)])
                for dataset in sorted(counts)
            },
            "samples_per_epoch": samples_per_epoch,
            "hard_repeat_cap": True,
        }
    else:
        raise ValueError(
            "train.sampling_strategy must be "
            "'equal_source_replacement' or 'capped_source_repeat'"
        )
    train_sub = torch.utils.data.Subset(ds, tr_idx.tolist())
    loader = DataLoader(train_sub, batch_size=int(cfg["train"]["batch_size"]),
                        sampler=sampler, num_workers=int(cfg["train"].get("num_workers", 0)),
                        collate_fn=collate)

    model = MultimodalStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        ppg_input_dim=int(cfg["model"].get("ppg_input_dim", 512)),
        projection_dim=int(cfg["model"]["projection_dim"]),
        projection_dropout=float(cfg["model"]["projection_dropout"]),
        ppg_hidden_dim=int(cfg["model"]["ppg_hidden_dim"]),
        fusion_hidden_dim=int(cfg["model"]["fusion_hidden_dim"]),
        text_init_ckpt=cfg["model"].get("text_init_ckpt"),
        text_max_length=int(cfg["model"].get("text_max_length", 96)),
        posture_film_enabled=bool(
            cfg["model"].get("posture_film", {}).get("enabled", False)
        ),
        posture_film_residual_scale=float(
            cfg["model"].get("posture_film", {}).get(
                "residual_scale", 0.1
            )
        ),
        fusion_mode=str(cfg["model"].get("fusion_mode", "concat")),
    ).to(device)
    full_init = cfg["model"].get("full_init_checkpoint")
    full_init_audit = None
    if full_init:
        full_state = torch.load(
            full_init,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(full_state, strict=True)
        full_init_audit = {
            "checkpoint": str(full_init),
            "strict": True,
            "loaded_tensors": int(len(full_state)),
        }
    ppg_backbone_init = cfg["model"].get(
        "ppg_backbone_init_checkpoint"
    )
    ppg_backbone_load_audit = None
    if ppg_backbone_init:
        ppg_backbone_state = torch.load(
            ppg_backbone_init,
            map_location=device,
            weights_only=False,
        )
        ppg_backbone_load_audit = model.load_ppg_backbone_state(
            ppg_backbone_state
        )
        ppg_backbone_load_audit["checkpoint"] = str(ppg_backbone_init)
    freeze_prefixes = [
        str(value).strip().rstrip(".")
        for value in cfg["model"].get(
            "freeze_parameter_prefixes", []
        )
        if str(value).strip()
    ]
    freeze_audit = None
    if freeze_prefixes:
        matched = {prefix: [] for prefix in freeze_prefixes}
        for name, parameter in model.named_parameters():
            for prefix in freeze_prefixes:
                if name == prefix or name.startswith(prefix + "."):
                    parameter.requires_grad_(False)
                    matched[prefix].append(name)
                    break
        missing_prefixes = sorted(
            prefix for prefix, names in matched.items() if not names
        )
        if missing_prefixes:
            raise ValueError(
                "freeze_parameter_prefixes matched no parameters: "
                f"{missing_prefixes}"
            )
        freeze_audit = {
            "prefixes": freeze_prefixes,
            "matched": matched,
            "frozen_parameter_tensors": int(
                sum(len(names) for names in matched.values())
            ),
        }
    posture_film_cfg = cfg["model"].get("posture_film", {})
    posture_film_init = posture_film_cfg.get("init_checkpoint")
    if model.posture_film is not None and posture_film_init:
        posture_state = torch.load(
            posture_film_init,
            map_location=device,
            weights_only=False,
        )
        model.posture_film.load_state_dict(posture_state, strict=True)
    if (
        model.posture_film is not None
        and bool(posture_film_cfg.get("freeze", True))
    ):
        for parameter in model.posture_film.parameters():
            parameter.requires_grad_(False)

    teacher_encoder_mode = str(
        cfg["loss"].get("teacher_encoder", "shared")
    ).lower()
    teacher_encoder = None
    if teacher_encoder_mode == "frozen_copy":
        teacher_encoder = copy.deepcopy(model.text_branch).to(device)
        teacher_encoder.eval()
        for parameter in teacher_encoder.parameters():
            parameter.requires_grad_(False)
    elif teacher_encoder_mode != "shared":
        raise ValueError(
            "loss.teacher_encoder must be 'shared' or 'frozen_copy'"
        )

    base_learning_rate = float(cfg["train"]["learning_rate"])
    text_encoder_learning_rate = float(
        cfg["train"].get(
            "text_encoder_learning_rate", base_learning_rate
        )
    )
    ppg_backbone_learning_rate = float(
        cfg["train"].get(
            "ppg_backbone_learning_rate", base_learning_rate
        )
    )
    warmup_freeze_ppg_epochs = int(
        cfg["train"].get("warmup_freeze_ppg_epochs", 0)
    )
    if (
        not np.isfinite(base_learning_rate)
        or base_learning_rate <= 0.0
        or not np.isfinite(text_encoder_learning_rate)
        or text_encoder_learning_rate <= 0.0
        or not np.isfinite(ppg_backbone_learning_rate)
        or ppg_backbone_learning_rate <= 0.0
        or warmup_freeze_ppg_epochs < 0
    ):
        raise ValueError(
            "optimizer learning rates must be finite and positive; "
            "warmup_freeze_ppg_epochs must be non-negative"
        )
    text_encoder_parameter_ids = {
        id(parameter)
        for parameter in model.text_branch.encoder.parameters()
    }
    ppg_backbone_parameter_ids = {
        id(parameter)
        for _, parameter in model.ppg_backbone_named_parameters()
    }
    text_encoder_parameters = [
        parameter
        for parameter in model.text_branch.encoder.parameters()
        if parameter.requires_grad
    ]
    ppg_backbone_parameters = [
        parameter
        for _, parameter in model.ppg_backbone_named_parameters()
        if parameter.requires_grad
    ]
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in text_encoder_parameter_ids
        and id(parameter) not in ppg_backbone_parameter_ids
    ]
    optimizer_groups = []
    if text_encoder_parameters:
        optimizer_groups.append(
            {
                "params": text_encoder_parameters,
                "lr": text_encoder_learning_rate,
                "group_name": "distilbert_encoder",
            }
        )
    if ppg_backbone_parameters:
        optimizer_groups.append(
            {
                "params": ppg_backbone_parameters,
                "lr": (
                    0.0
                    if warmup_freeze_ppg_epochs > 0
                    else ppg_backbone_learning_rate
                ),
                "group_name": "ppg_backbone",
                "target_lr": ppg_backbone_learning_rate,
            }
        )
    if other_parameters:
        optimizer_groups.append(
            {
                "params": other_parameters,
                "lr": base_learning_rate,
                "group_name": "context_projection_fusion_heads",
            }
        )
    opt = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    lam = float(cfg["loss"].get("lambda_align", 0.1))
    mdrop = float(cfg["train"].get("modality_dropout", 0.0))
    temp = float(cfg["loss"].get("temperature", 0.07))
    clsp_mode = str(
        cfg["loss"].get("clsp_mode", "legacy_supcon")
    ).lower()
    semantic_positive_weight = float(
        cfg["loss"].get("semantic_positive_weight", 0.0)
    )
    teacher_stop_gradient = bool(
        cfg["loss"].get("teacher_stop_gradient", False)
    )
    dataset_loss_weights = {
        str(key).lower(): float(value)
        for key, value in cfg["train"].get(
            "dataset_loss_weights", {}
        ).items()
    }
    invalid_dataset_weights = {
        key: value
        for key, value in dataset_loss_weights.items()
        if not np.isfinite(value) or value < 0.0
    }
    if invalid_dataset_weights:
        raise ValueError(
            "train.dataset_loss_weights must contain finite, "
            f"non-negative values: {invalid_dataset_weights}"
        )
    if mute_text or mute_ppg:      # ablation: 한 모달만 → CLSP/드롭 무의미
        lam, mdrop = 0.0, 0.0

    selection_metric = str(cfg["train"].get("early_stop_metric", "joint")).lower()
    if selection_metric not in {"joint", "mse", "macro_ccc"}:
        raise ValueError(
            "train.early_stop_metric must be 'joint', 'mse', or 'macro_ccc'"
        )
    selection_align_weight = float(
        cfg["train"].get("selection_align_weight", lam)
    )
    macro_metric_aggregation = str(
        cfg["train"].get(
            "macro_metric_aggregation", "task_balanced"
        )
    ).lower()
    if macro_metric_aggregation not in {
        "task_balanced",
        "target_balanced",
    }:
        raise ValueError(
            "train.macro_metric_aggregation must be 'task_balanced' "
            "or 'target_balanced'"
        )
    best_score = float("inf")
    best_val_mse = float("inf")
    best_val_align = float("inf")
    best_selection_value = None
    best_epoch = None
    best_state = None
    patience = int(cfg["train"].get("patience", 5))
    bad = 0
    history: list[dict] = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for group in opt.param_groups:
            if group.get("group_name") == "ppg_backbone":
                group["lr"] = (
                    0.0
                    if epoch <= warmup_freeze_ppg_epochs
                    else ppg_backbone_learning_rate
                )
        model.train()
        agg = {"loss": 0.0, "reg": 0.0, "align": 0.0}
        for b in loader:
            targets = {t: b["targets"][t].to(device) for t in TARGETS}
            masks = {t: b["masks"][t].to(device) for t in TARGETS}
            out = model(texts=b["text"], ppg_features=b["ppg_features"], device=device,
                        ppg_mask=b["ppg_mask"], modality_dropout=mdrop,
                        mute_text=mute_text, mute_ppg=mute_ppg,
                        postures=b["posture"])

            reg = dataset_balanced_mse(
                out["preds"],
                targets,
                masks,
                b["dataset"],
                dataset_weights=dataset_loss_weights,
                sample_weights=b["sample_weight"].to(device),
            )

            if lam > 0.0:
                # Alignment uses only rows enabled by clsp_mask.
                if any(b["teacher_text"]):
                    align_txt = [
                        tt if tt else cx
                        for tt, cx in zip(b["teacher_text"], b["text"])
                    ]
                    if teacher_encoder is not None:
                        with torch.no_grad():
                            align_z = teacher_encoder(align_txt, device)
                    else:
                        align_z = model.text_branch(align_txt, device)
                else:
                    align_z = out["text_z_clean"]
                if teacher_stop_gradient:
                    align_z = align_z.detach()
                align = _clsp_loss(
                    clsp_mode,
                    align_z,
                    out["ppg_z_clean"],
                    b,
                    b["clsp_mask"],
                    temp,
                    semantic_positive_weight,
                )
            else:
                align = reg.new_zeros(())

            loss = reg + lam * align
            opt.zero_grad(); loss.backward(); opt.step()
            bs = len(b["sample_id"])
            agg["loss"] += float(loss) * bs; agg["reg"] += float(reg) * bs; agg["align"] += float(align) * bs
        n = samples_per_epoch
        if early:
            model.eval()
            vs, vn = 0.0, 0
            val_text_z = []
            val_ppg_z = []
            val_valid = []
            val_pair_ids: list[str] = []
            val_positive_sigs: list[str] = []
            val_negative_mask_sigs: list[str] = []
            val_semantic_sigs: list[str] = []
            val_datasets: list[str] = []
            val_participants: list[str] = []
            val_sessions: list[str] = []
            val_label_groups: list[str] = []
            val_window_indices: list[int] = []
            val_adjacent_radii: list[int] = []
            val_loss_groups: list[str] = []
            val_predictions = {target: [] for target in TARGETS}
            val_truths = {target: [] for target in TARGETS}
            val_masks = {target: [] for target in TARGETS}
            with torch.no_grad():
                for b in val_loader:
                    tg = {t: b["targets"][t].to(device) for t in TARGETS}
                    mk = {t: b["masks"][t].to(device) for t in TARGETS}
                    out = model(
                        texts=b["text"],
                        ppg_features=b["ppg_features"],
                        device=device,
                        ppg_mask=b["ppg_mask"],
                        mute_text=mute_text,
                        mute_ppg=mute_ppg,
                        postures=b["posture"],
                    )
                    vl, _ = partial_masked_mse(out["preds"], tg, mk)
                    vs += float(vl) * len(b["sample_id"]); vn += len(b["sample_id"])
                    if any(b["teacher_text"]):
                        align_txt = [
                            tt if tt else cx
                            for tt, cx in zip(b["teacher_text"], b["text"])
                        ]
                        encoder = (
                            teacher_encoder
                            if teacher_encoder is not None
                            else model.text_branch
                        )
                        align_text_z = encoder(align_txt, device)
                    else:
                        align_text_z = out["text_z_clean"]
                    val_text_z.append(align_text_z)
                    val_ppg_z.append(out["ppg_z_clean"])
                    val_valid.append(b["clsp_mask"].to(device))
                    val_pair_ids.extend(b["clsp_pair_id"])
                    val_positive_sigs.extend(b["clsp_positive_sig"])
                    val_negative_mask_sigs.extend(
                        b["clsp_negative_mask_sig"]
                    )
                    val_semantic_sigs.extend(b["clsp_semantic_sig"])
                    val_datasets.extend(b["dataset"])
                    val_participants.extend(b["participant_id"])
                    val_sessions.extend(b["session_id"])
                    val_label_groups.extend(b["label_group_sig"])
                    val_window_indices.extend(b["window_index"])
                    val_adjacent_radii.extend(
                        b["negative_adjacent_radius"]
                    )
                    val_loss_groups.extend(b["clsp_loss_group"])
                    for target in TARGETS:
                        val_predictions[target].append(
                            out["preds"][target].detach().cpu().numpy()
                        )
                        val_truths[target].append(
                            b["targets"][target].numpy()
                        )
                        val_masks[target].append(
                            b["masks"][target].numpy()
                        )
            vmse = vs / max(vn, 1)
            val_text_all = torch.cat(val_text_z, dim=0)
            val_ppg_all = torch.cat(val_ppg_z, dim=0)
            val_valid_all = torch.cat(val_valid, dim=0)
            val_clsp_batch = {
                "context_sig": val_positive_sigs,
                "clsp_pair_id": val_pair_ids,
                "clsp_positive_sig": val_positive_sigs,
                "clsp_negative_mask_sig": val_negative_mask_sigs,
                "clsp_semantic_sig": val_semantic_sigs,
                "dataset": val_datasets,
                "participant_id": val_participants,
                "session_id": val_sessions,
                "label_group_sig": val_label_groups,
                "window_index": val_window_indices,
                "negative_adjacent_radius": val_adjacent_radii,
                "clsp_loss_group": val_loss_groups,
            }
            val_align = float(
                _clsp_loss(
                    clsp_mode,
                    val_text_all,
                    val_ppg_all,
                    val_clsp_batch,
                    val_valid_all,
                    temp,
                    semantic_positive_weight,
                )
            )
            alignment = cross_modal_alignment_metrics(
                val_text_all,
                val_ppg_all,
                val_pair_ids,
                val_valid_all,
            )
            domain_retrieval = cross_modal_domain_retrieval_metrics(
                val_text_all,
                val_ppg_all,
                val_datasets,
                val_participants,
                val_valid_all,
            )
            macro = _macro_state_metrics(
                {
                    target: np.concatenate(val_predictions[target])
                    for target in TARGETS
                },
                {
                    target: np.concatenate(val_truths[target])
                    for target in TARGETS
                },
                {
                    target: np.concatenate(val_masks[target])
                    for target in TARGETS
                },
                val_datasets,
                aggregation=macro_metric_aggregation,
            )
            joint = vmse + selection_align_weight * val_align
            if selection_metric == "mse":
                selection_value = vmse
                score = selection_value
                selection_objective = "min"
            elif selection_metric == "joint":
                selection_value = joint
                score = selection_value
                selection_objective = "min"
            else:
                selection_value = float(macro["ccc"])
                score = -selection_value
                selection_objective = "max"
            record = {
                "epoch": epoch,
                "train_loss": agg["loss"] / n,
                "train_regression": agg["reg"] / n,
                "train_clsp": agg["align"] / n,
                "val_mse": vmse,
                "val_clsp": val_align,
                "val_joint": joint,
                "val_macro_tasks": macro["tasks"],
                "val_macro_ccc": macro["ccc"],
                "val_macro_accuracy": macro["accuracy"],
                "val_macro_f1": macro["f1"],
                "val_macro_aggregation": macro["aggregation"],
                "val_target_metrics": macro["targets"],
                "selection_metric": selection_metric,
                "selection_objective": selection_objective,
                "selection_value": selection_value,
                "ppg_backbone_lr": (
                    0.0
                    if epoch <= warmup_freeze_ppg_epochs
                    else ppg_backbone_learning_rate
                ),
                **{f"alignment_{key}": value for key, value in alignment.items()},
                **{
                    f"domain_retrieval_{key}": value
                    for key, value in domain_retrieval.items()
                },
            }
            history.append(record)
            if verbose:
                print(
                    f"[ep{epoch}] train={record['train_loss']:.4f} "
                    f"reg={record['train_regression']:.4f} "
                    f"clsp={record['train_clsp']:.4f} "
                    f"val_mse={vmse:.4f} val_clsp={val_align:.4f} "
                    f"CCC={macro['ccc']:+.3f} "
                    f"Acc={macro['accuracy']:.3f} F1={macro['f1']:.3f} "
                    f"gap={alignment['cos_gap']:+.4f} "
                    f"R@1={alignment['prototype_r1']:.3f} "
                    f"domain_excess="
                    f"{domain_retrieval['excess_over_chance']:+.3f}"
                )
            if score < best_score - 1e-5:
                best_score = score
                best_selection_value = selection_value
                best_val_mse = vmse
                best_val_align = val_align
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(
                            f"  early stop @ep{epoch} "
                            f"(best ep={best_epoch}, "
                            f"{selection_metric}={best_selection_value:.4f}, "
                            f"val_mse={best_val_mse:.4f}, "
                            f"val_clsp={best_val_align:.4f})"
                        )
                    break
        else:
            if verbose:
                print(f"[ep{epoch}] loss={agg['loss']/n:.4f} reg={agg['reg']/n:.4f} align={agg['align']/n:.4f}")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": agg["loss"] / n,
                    "train_regression": agg["reg"] / n,
                    "train_clsp": agg["align"] / n,
                    "ppg_backbone_lr": (
                        0.0
                        if epoch <= warmup_freeze_ppg_epochs
                        else ppg_backbone_learning_rate
                    ),
                }
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_history = history
    model.training_summary = {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric if early else None,
        "best_selection_value": (
            best_selection_value if best_epoch is not None else None
        ),
        "best_selection_score": (
            best_selection_value if best_epoch is not None else None
        ),
        "best_val_mse": best_val_mse if best_epoch is not None else None,
        "best_val_clsp": best_val_align if best_epoch is not None else None,
        "clean_clsp_embeddings": True,
        "clsp_mode": clsp_mode,
        "clsp_temporal_match_required": True,
        "clsp_primary_positive_contract": "exact_pair_only",
        "clsp_group_aware_negative_mask": [
            "same_teacher",
            "same_participant",
            "same_session",
            "same_label_group",
            "adjacent_window",
        ],
        "clsp_teacher_loss_group_balancing": True,
        "teacher_encoder": teacher_encoder_mode,
        "teacher_stop_gradient": teacher_stop_gradient,
        "semantic_positive_weight": semantic_positive_weight,
        "dataset_loss_weights": dataset_loss_weights,
        "sampling": sampler_audit,
        "optimizer_learning_rates": {
            "distilbert_encoder": text_encoder_learning_rate,
            "ppg_backbone_after_warmup": ppg_backbone_learning_rate,
            "context_projection_fusion_heads": base_learning_rate,
        },
        "warmup_freeze_ppg_epochs": warmup_freeze_ppg_epochs,
        "fusion_mode": model.fusion_mode,
        "full_model_warm_start": full_init_audit,
        "ppg_backbone_warm_start": ppg_backbone_load_audit,
        "frozen_parameter_contract": freeze_audit,
        "posture_film": {
            "enabled": model.posture_film is not None,
            "frozen": (
                all(
                    not parameter.requires_grad
                    for parameter in model.posture_film.parameters()
                )
                if model.posture_film is not None
                else None
            ),
            "init_checkpoint": posture_film_init,
        },
        "input_condition": (
            "text_only"
            if mute_ppg
            else "ppg_only"
            if mute_text
            else "text_ppg"
        ),
        "validation_split": cfg["train"].get("val_split", "global"),
        "macro_metric_aggregation": macro_metric_aggregation,
        "validation_split_audit": validation_audit,
        "input_contract": input_contract,
    }
    model.training_input_contract = input_contract
    model.fixed_teacher_encoder_state = (
        {
            key: value.detach().cpu().clone()
            for key, value in teacher_encoder.state_dict().items()
        }
        if teacher_encoder is not None
        else None
    )
    return model, device


def main(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, _ = train_model(cfg)
    out = Path(cfg["paths"].get("out_dir", "outputs")) / "model_v3"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "full.pt")
    teacher_state = getattr(model, "fixed_teacher_encoder_state", None)
    save_teacher_encoder = bool(
        cfg["train"].get("save_teacher_encoder", True)
    )
    if teacher_state is not None and save_teacher_encoder:
        torch.save(teacher_state, out / "teacher_encoder.pt")
    metrics = {
        "summary": getattr(model, "training_summary", {}),
        "epochs": getattr(model, "training_history", []),
    }
    (out / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    input_contract = getattr(model, "training_input_contract", None)
    if input_contract is not None:
        (out / "training_input_contract.json").write_text(
            json.dumps(input_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"saved -> {out/'full.pt'}")
    if teacher_state is not None and save_teacher_encoder:
        print(f"fixed teacher -> {out/'teacher_encoder.pt'}")
    elif teacher_state is not None:
        print("fixed teacher sidecar -> skipped by configuration")
    print(f"metrics -> {out/'training_metrics.json'}")
    if input_contract is not None:
        print(f"input contract -> {out/'training_input_contract.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
