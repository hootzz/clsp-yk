"""MultiDatasetAffect dataset loader.

Reads manifest.csv and ppg_features.csv and provides batches containing
partial-label target masks, ppg_mask, context_sig, and teacher_text.

- Target NaN → mask 0, excluded by partial_masked_mse.
- ppg_available=0 or failed PPG join → zero PPG features and ppg_mask 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model import TARGETS


class MultiDatasetAffect(Dataset):
    def __init__(self, manifest_csv, ppg_csv=None, ppg_dim: int = 512,
                 include_datasets=None, exclude_datasets=None):
        df = pd.read_csv(manifest_csv)
        if include_datasets:
            df = df[df["dataset"].isin(include_datasets)]
        if exclude_datasets:
            df = df[~df["dataset"].isin(exclude_datasets)]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("manifest empty after dataset filtering")

        self.ppg_dim = ppg_dim
        self.ppg_cols = [f"ppg_f{i}" for i in range(ppg_dim)]

        # PPG embedding map.
        self.ppg_map: dict[str, np.ndarray] = {}
        if ppg_csv and Path(ppg_csv).is_file():
            pdf = pd.read_csv(ppg_csv)
            cols = [c for c in self.ppg_cols if c in pdf.columns]
            for _, r in pdf.iterrows():
                self.ppg_map[str(r["sample_id"])] = r[cols].to_numpy(dtype=np.float32)

        for t in TARGETS:
            if t not in df.columns:
                df[t] = np.nan
        for c in (
            "teacher_text",
            "context_sig",
            "participant_id",
            "session_id",
            "label_group_sig",
            "clsp_pair_id",
            "clsp_positive_sig",
            "clsp_negative_mask_sig",
            "clsp_semantic_sig",
            "clsp_teacher_source",
            "clsp_teacher_tier",
            "clsp_teacher_type",
            "clsp_teacher_confidence",
            "clsp_temporal_resolution",
            "clsp_loss_group",
            "context7_posture",
            "context7_energy_expenditure",
        ):
            if c not in df.columns:
                df[c] = ""
            else:
                # pandas represents empty CSV cells as NaN; never let the
                # literal string "nan" become a teacher sentence/signature.
                df[c] = df[c].fillna("")
        empty_pair = df["clsp_pair_id"].fillna("").astype(str).str.len() == 0
        df.loc[empty_pair, "clsp_pair_id"] = df.loc[
            empty_pair, "sample_id"
        ].astype(str)
        empty_positive = (
            df["clsp_positive_sig"].fillna("").astype(str).str.len() == 0
        )
        df.loc[empty_positive, "clsp_positive_sig"] = df.loc[
            empty_positive, "context_sig"
        ].astype(str)
        empty_negative_mask = (
            df["clsp_negative_mask_sig"]
            .fillna("")
            .astype(str)
            .str.len()
            == 0
        )
        df.loc[empty_negative_mask, "clsp_negative_mask_sig"] = df.loc[
            empty_negative_mask, "clsp_positive_sig"
        ].astype(str)
        if "ppg_available" not in df.columns:
            df["ppg_available"] = df["sample_id"].astype(str).isin(self.ppg_map).astype(int)
        if "clsp_teacher_available" not in df.columns:
            # Historical manifests aligned all PPG-valid context rows.
            df["clsp_teacher_available"] = 1
        df["clsp_teacher_available"] = pd.to_numeric(
            df["clsp_teacher_available"], errors="coerce"
        ).fillna(0).astype(int)
        numeric_defaults = {
            # Historical manifests are accepted, while the generated R9
            # manifest writes these fields explicitly for every row.
            "clsp_temporal_match": 1,
            "clsp_primary_positive_eligible": 1,
            "window_index": -1,
            "negative_adjacent_radius": 1,
            "ppg_quality_reliable": 1,
            "ppg_confidence": 1.0,
            # Optional row weight.  Existing manifests therefore preserve
            # their historical row-uniform behaviour, while window-level
            # sources can equalise participant×block contributions.
            "sample_weight": 1.0,
        }
        for column, default in numeric_defaults.items():
            if column not in df.columns:
                df[column] = default
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            ).fillna(default)
        empty_session = (
            df["session_id"].fillna("").astype(str).str.strip().eq("")
        )
        df.loc[empty_session, "session_id"] = df.loc[
            empty_session, "sample_id"
        ].astype(str)
        invalid_teacher = (
            df["clsp_teacher_available"].eq(1)
            & df["teacher_text"].astype(str).str.strip().eq("")
        )
        if invalid_teacher.any():
            bad_ids = (
                df.loc[invalid_teacher, "sample_id"]
                .astype(str)
                .head(5)
                .tolist()
            )
            raise ValueError(
                "CLSP teacher was marked available but is empty for "
                f"{bad_ids}"
            )
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        sid = str(r["sample_id"])

        emb = self.ppg_map.get(sid)
        ppg_confidence = float(r.get("ppg_confidence", 1.0))
        ppg_confidence = (
            float(np.clip(ppg_confidence, 0.0, 1.0))
            if np.isfinite(ppg_confidence)
            else 0.0
        )
        has_ppg = (
            int(r.get("ppg_available", 0)) == 1
            and int(r.get("ppg_quality_reliable", 1)) == 1
            and ppg_confidence > 0.0
            and emb is not None
        )
        teacher_text = str(r.get("teacher_text", "") or "").strip()
        has_clsp_teacher = (
            has_ppg
            and int(r.get("clsp_teacher_available", 1)) == 1
            and int(r.get("clsp_temporal_match", 1)) == 1
            and int(r.get("clsp_primary_positive_eligible", 1)) == 1
            and bool(teacher_text)
        )
        ppg = torch.tensor(emb if has_ppg else np.zeros(self.ppg_dim, np.float32))

        tvals, tmask = {}, {}
        for t in TARGETS:
            v = r[t]
            ok = pd.notna(v)
            tvals[t] = float(v) if ok else 0.0
            tmask[t] = 1.0 if ok else 0.0

        return {
            "sample_id": sid,
            "dataset": str(r["dataset"]),
            "participant_id": str(r["participant_id"]),
            "session_id": str(r.get("session_id", "") or sid),
            "label_group_sig": str(
                r.get("label_group_sig", "") or ""
            ),
            "window_index": int(r.get("window_index", -1)),
            "negative_adjacent_radius": int(
                r.get("negative_adjacent_radius", 1)
            ),
            "text": str(r["context_text"]),
            "posture": str(r.get("context7_posture", "") or "not_recorded"),
            "energy_expenditure": str(
                r.get("context7_energy_expenditure", "")
                or "not_recorded"
            ),
            "teacher_text": teacher_text,
            "context_sig": str(r.get("context_sig", "") or sid),
            "clsp_pair_id": str(r.get("clsp_pair_id", "") or sid),
            "clsp_positive_sig": str(
                r.get("clsp_positive_sig", "") or r.get("context_sig", "") or sid
            ),
            "clsp_negative_mask_sig": str(
                r.get("clsp_negative_mask_sig", "")
                or r.get("clsp_positive_sig", "")
                or r.get("context_sig", "")
                or sid
            ),
            "clsp_semantic_sig": str(r.get("clsp_semantic_sig", "") or ""),
            "clsp_teacher_source": str(
                r.get("clsp_teacher_source", "") or ""
            ),
            "clsp_teacher_tier": str(r.get("clsp_teacher_tier", "") or ""),
            "clsp_teacher_type": str(
                r.get("clsp_teacher_type", "") or ""
            ),
            "clsp_teacher_confidence": str(
                r.get("clsp_teacher_confidence", "") or ""
            ),
            "clsp_temporal_resolution": str(
                r.get("clsp_temporal_resolution", "") or ""
            ),
            "clsp_loss_group": str(
                r.get("clsp_loss_group", "") or ""
            ),
            "clsp_temporal_match": int(
                r.get("clsp_temporal_match", 1)
            ),
            "clsp_primary_positive_eligible": int(
                r.get("clsp_primary_positive_eligible", 1)
            ),
            "ppg_features": ppg,
            # Confidence-gated residual: unreliable PPG is exactly zero and
            # never becomes a CLSP positive.
            "ppg_mask": torch.tensor(
                ppg_confidence if has_ppg else 0.0
            ),
            "clsp_mask": torch.tensor(1.0 if has_clsp_teacher else 0.0),
            "sample_weight": torch.tensor(
                float(r.get("sample_weight", 1.0)),
                dtype=torch.float32,
            ),
            "targets": {t: torch.tensor(tvals[t]) for t in TARGETS},
            "masks": {t: torch.tensor(tmask[t]) for t in TARGETS},
        }


def collate(batch):
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "participant_id": [b["participant_id"] for b in batch],
        "session_id": [b["session_id"] for b in batch],
        "label_group_sig": [b["label_group_sig"] for b in batch],
        "window_index": [b["window_index"] for b in batch],
        "negative_adjacent_radius": [
            b["negative_adjacent_radius"] for b in batch
        ],
        "text": [b["text"] for b in batch],
        "posture": [b["posture"] for b in batch],
        "energy_expenditure": [
            b["energy_expenditure"] for b in batch
        ],
        "teacher_text": [b["teacher_text"] for b in batch],
        "context_sig": [b["context_sig"] for b in batch],
        "clsp_pair_id": [b["clsp_pair_id"] for b in batch],
        "clsp_positive_sig": [b["clsp_positive_sig"] for b in batch],
        "clsp_negative_mask_sig": [
            b["clsp_negative_mask_sig"] for b in batch
        ],
        "clsp_semantic_sig": [b["clsp_semantic_sig"] for b in batch],
        "clsp_teacher_source": [b["clsp_teacher_source"] for b in batch],
        "clsp_teacher_tier": [b["clsp_teacher_tier"] for b in batch],
        "clsp_teacher_type": [b["clsp_teacher_type"] for b in batch],
        "clsp_teacher_confidence": [
            b["clsp_teacher_confidence"] for b in batch
        ],
        "clsp_temporal_resolution": [
            b["clsp_temporal_resolution"] for b in batch
        ],
        "clsp_loss_group": [b["clsp_loss_group"] for b in batch],
        "clsp_temporal_match": [
            b["clsp_temporal_match"] for b in batch
        ],
        "clsp_primary_positive_eligible": [
            b["clsp_primary_positive_eligible"] for b in batch
        ],
        "ppg_features": torch.stack([b["ppg_features"] for b in batch]),
        "ppg_mask": torch.stack([b["ppg_mask"] for b in batch]),
        "clsp_mask": torch.stack([b["clsp_mask"] for b in batch]),
        "sample_weight": torch.stack(
            [b["sample_weight"] for b in batch]
        ),
        "targets": {t: torch.stack([b["targets"][t] for b in batch]) for t in TARGETS},
        "masks": {t: torch.stack([b["masks"][t] for b in batch]) for t in TARGETS},
    }
