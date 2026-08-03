"""Deterministic participant-level train/validation splits."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(str(dataset).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return (int(seed) + offset) % (2**32)


def participant_validation_indices(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Return deterministic validation rows without participant overlap.

    ``global`` reproduces the historical split. ``dataset_stratified`` draws
    participants independently inside every dataset so that small datasets
    cannot disappear from validation.
    """
    required = {"dataset", "participant_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"split frame is missing columns: {missing}")

    seed = int(cfg.get("seed", 42))
    fraction = float(cfg["train"].get("val_frac", 0.15))
    if not 0.0 < fraction < 1.0:
        raise ValueError("train.val_frac must be between 0 and 1")
    strategy = str(cfg["train"].get("val_split", "global")).lower()

    if strategy == "global":
        participant_ids = frame["participant_id"].astype(str).to_numpy()
        unique = np.asarray(sorted(set(participant_ids)))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        n_val = max(1, int(len(unique) * fraction))
        validation_participants = set(unique[:n_val])
        return np.flatnonzero(
            np.asarray(
                [pid in validation_participants for pid in participant_ids],
                dtype=bool,
            )
        )

    if strategy != "dataset_stratified":
        raise ValueError(
            "train.val_split must be 'global' or 'dataset_stratified'"
        )

    selected = np.zeros(len(frame), dtype=bool)
    for dataset, group in frame.groupby("dataset", sort=True):
        participant_ids = group["participant_id"].astype(str).to_numpy()
        unique = np.asarray(sorted(set(participant_ids)))
        if len(unique) < 2:
            raise ValueError(
                f"dataset {dataset!r} has fewer than two participants"
            )
        rng = np.random.default_rng(_dataset_seed(seed, str(dataset)))
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * fraction)))
        n_val = min(n_val, len(unique) - 1)
        validation_participants = set(unique[:n_val])
        selected[group.index.to_numpy()] = np.asarray(
            [pid in validation_participants for pid in participant_ids],
            dtype=bool,
        )
    return np.flatnonzero(selected)


def split_summary(frame: pd.DataFrame, validation_indices: np.ndarray) -> dict:
    """Return a serializable dataset-level participant/row audit."""
    is_validation = np.zeros(len(frame), dtype=bool)
    is_validation[np.asarray(validation_indices, dtype=int)] = True
    summary = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        indices = group.index.to_numpy()
        val_group = group.loc[indices[is_validation[indices]]]
        train_group = group.loc[indices[~is_validation[indices]]]
        train_participants = set(train_group["participant_id"].astype(str))
        validation_participants = set(
            val_group["participant_id"].astype(str)
        )
        overlap = sorted(train_participants & validation_participants)
        summary[str(dataset)] = {
            "train_rows": int(len(train_group)),
            "validation_rows": int(len(val_group)),
            "train_participants": int(len(train_participants)),
            "validation_participants": int(len(validation_participants)),
            "participant_overlap": overlap,
        }
    return summary

