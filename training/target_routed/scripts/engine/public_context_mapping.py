"""Map public protocol metadata to the existing original-7 text builder.

This module decides slot values only.  It reuses
``model_v3/build_context7_gpt.py`` for schema validation and canonical text.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from build_context7_gpt import (  # noqa: E402
    LEGACY_SCHEMA_COLUMNS,
    SCHEMA_VERSION,
    SLOTS,
    deterministic_text,
    duration_bucket,
    make_spec,
    validate_spec,
    validate_text,
)

MAUS_SAMPLE_RE = re.compile(r"maus_(\d+)_t(\d+)$", re.IGNORECASE)
FORBIDDEN_EEVR_MAIN_SCENE_PATTERNS = (
    r"\b(?:high|low)[-\s]?(?:valence|arousal)\b",
    r"\bvalence\b",
    r"\barousal\b",
    r"\b(?:exciting|excited|relaxing|relaxed|pleasant)\b",
    r"\b(?:frightening|frightened|scary|afraid)\b",
    r"\b(?:happy|sad|calm|calming|stressful|stressed)\b",
    r"\b(?:emotion|emotional|emotionally)\b",
    r"\bI\s+(?:felt|feel|was feeling)\b",
    r"\bmade\s+me\s+feel\b",
    r"\bviewer\s+feels?\b",
)


def validate_objective_eevr_scene(detail: str) -> None:
    """Reject affect-derived or post-exposure language from main context."""
    value = str(detail or "").strip()
    hits = [
        pattern
        for pattern in FORBIDDEN_EEVR_MAIN_SCENE_PATTERNS
        if re.search(pattern, value, flags=re.IGNORECASE)
    ]
    if hits:
        raise ValueError(
            "EEVR main scene must be objective and deployment-observable; "
            f"detail={value!r}, forbidden_patterns={hits}"
        )


def _signature(spec: dict[str, str]) -> str:
    validate_spec(spec)
    return "|".join(spec[slot] for slot in SLOTS)


def attach_public_duration_evidence(
    frame: pd.DataFrame,
    *,
    eevr_session_audit: str | Path,
    maus_raw_root: str | Path,
    maus_sampling_rate_hz: float,
    expected_maus_duration_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach observed EEVR/MAUS seconds without estimating missing values."""
    result = frame.copy(deep=True)
    result["source_duration_seconds"] = np.nan
    result["duration_evidence_source"] = "not_recorded"

    eevr_mask = result["dataset"].astype(str).str.lower().eq("eevr")
    eevr_path = Path(eevr_session_audit).expanduser().resolve()
    eevr = pd.read_csv(eevr_path)
    required = {
        "sample_id",
        "n_input_rows",
        "fs_inferred_hz",
    }
    missing = sorted(required - set(eevr.columns))
    if missing:
        raise ValueError(
            f"EEVR duration audit is missing columns: {missing}"
        )
    if eevr["sample_id"].astype(str).duplicated().any():
        raise ValueError("EEVR duration audit has duplicate sample_id values")
    numerator = pd.to_numeric(eevr["n_input_rows"], errors="coerce")
    denominator = pd.to_numeric(eevr["fs_inferred_hz"], errors="coerce")
    invalid = (
        numerator.isna()
        | denominator.isna()
        | (numerator <= 0)
        | (denominator <= 0)
    )
    if invalid.any():
        raise ValueError("EEVR duration audit contains invalid rows/fs")
    eevr_seconds = pd.Series(
        (numerator / denominator).to_numpy(),
        index=eevr["sample_id"].astype(str),
    )
    requested_eevr = result.loc[eevr_mask, "sample_id"].astype(str)
    missing_eevr = sorted(set(requested_eevr) - set(eevr_seconds.index))
    if missing_eevr:
        raise ValueError(
            f"EEVR duration evidence join failed: {missing_eevr[:10]}"
        )
    result.loc[eevr_mask, "source_duration_seconds"] = (
        requested_eevr.map(eevr_seconds).to_numpy()
    )
    result.loc[eevr_mask, "duration_evidence_source"] = (
        "eevr_session_n_input_rows_div_fs_inferred_hz"
    )

    maus_mask = result["dataset"].astype(str).str.lower().eq("maus")
    maus_root = Path(maus_raw_root).expanduser().resolve()
    sampling_rate = float(maus_sampling_rate_hz)
    expected_seconds = float(expected_maus_duration_seconds)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("MAUS sampling rate must be finite and positive")
    if not np.isfinite(expected_seconds) or expected_seconds <= 0:
        raise ValueError("Expected MAUS duration must be finite and positive")
    participant_frames: dict[str, pd.DataFrame] = {}
    maus_seconds: list[float] = []
    for sample_id in result.loc[maus_mask, "sample_id"].astype(str):
        match = MAUS_SAMPLE_RE.fullmatch(sample_id)
        if not match:
            raise ValueError(f"Unexpected MAUS sample_id: {sample_id}")
        participant, trial_text = match.groups()
        trial_index = int(trial_text) - 1
        if participant not in participant_frames:
            raw_path = maus_root / participant / "inf_ppg.csv"
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"MAUS duration source not found: {raw_path}"
                )
            participant_frames[participant] = pd.read_csv(raw_path)
        raw = participant_frames[participant]
        if not 0 <= trial_index < len(raw.columns):
            raise ValueError(
                f"{sample_id}: trial index is outside inf_ppg.csv"
            )
        samples = int(
            pd.to_numeric(
                raw.iloc[:, trial_index], errors="coerce"
            ).notna().sum()
        )
        seconds = samples / sampling_rate
        if abs(seconds - expected_seconds) > 1.0 / sampling_rate:
            raise ValueError(
                f"{sample_id}: observed {seconds}s, expected "
                f"{expected_seconds}s"
            )
        maus_seconds.append(float(seconds))
    result.loc[maus_mask, "source_duration_seconds"] = maus_seconds
    result.loc[maus_mask, "duration_evidence_source"] = (
        "maus_inf_ppg_non_nan_samples_div_256hz"
    )

    duration_audit: dict[str, Any] = {
        "status": "pass",
        "bucket_boundaries_seconds": {
            "under_one_minute": "[0,60)",
            "one_to_five_minutes": "[60,300)",
            "five_to_thirty_minutes": "[300,1800)",
            "over_thirty_minutes": "[1800,inf)",
            "exact_ten_seconds": "10",
        },
        "eevr": {
            "source": str(eevr_path),
            "rows": int(eevr_mask.sum()),
            "seconds_min": float(
                result.loc[eevr_mask, "source_duration_seconds"].min()
            ),
            "seconds_max": float(
                result.loc[eevr_mask, "source_duration_seconds"].max()
            ),
            "bucket_counts": (
                result.loc[eevr_mask, "source_duration_seconds"]
                .map(duration_bucket)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        },
        "maus": {
            "source": str(maus_root),
            "sampling_rate_hz": sampling_rate,
            "rows": int(maus_mask.sum()),
            "observed_unique_seconds": sorted(
                result.loc[
                    maus_mask, "source_duration_seconds"
                ].unique().tolist()
            ),
            "boundary_rule_at_300_seconds": (
                "five_to_thirty_minutes"
            ),
            "bucket_counts": (
                result.loc[maus_mask, "source_duration_seconds"]
                .map(duration_bucket)
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        },
    }
    return result, duration_audit


def adapt_public_manifest(
    frame: pd.DataFrame,
    *,
    include_case_title: bool = False,
    include_eevr_scene: bool = False,
    include_maus_level: bool = False,
) -> pd.DataFrame:
    """Return a v2 copy; the immutable legacy source is never overwritten."""
    result = frame.copy(deep=True)
    result["source_context_sig_before_v2"] = result["context_sig"].astype(str)
    records: list[dict[str, Any]] = []
    for row in result.to_dict(orient="records"):
        spec = make_spec(
            str(row["dataset"]),
            str(row["context_sig"]),
            str(row.get("context_text_original") or row.get("context_text") or ""),
            "include" if include_maus_level else "omit",
            "include" if include_case_title else "omit",
            "include" if include_eevr_scene else "omit",
            row.get("source_duration_seconds"),
        )
        if (
            str(row["dataset"]).lower() == "eevr"
            and include_eevr_scene
        ):
            validate_objective_eevr_scene(
                spec.get("observable_detail", "")
            )
        validate_spec(spec)
        text = deterministic_text(spec)
        issues = validate_text(text, spec)
        if issues:
            raise ValueError(
                f"{row.get('sample_id')}: original-7 text failed: {issues}"
            )
        records.append(
            {
                "spec": spec,
                "text": text,
                "signature": _signature(spec),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    result = result.drop(
        columns=[
            column
            for column in LEGACY_SCHEMA_COLUMNS
            if column in result.columns
        ]
    )
    for slot in SLOTS:
        result[f"context7_{slot}"] = [
            record["spec"][slot] for record in records
        ]
    result["context7_observable_detail"] = [
        record["spec"].get("observable_detail", "")
        for record in records
    ]
    result["context7_duration_seconds"] = pd.to_numeric(
        result.get("source_duration_seconds"), errors="coerce"
    )
    result["context_text"] = [record["text"] for record in records]
    result["text"] = result["context_text"]
    result["context_sig"] = [
        record["signature"] for record in records
    ]
    result["context_input_sha256"] = [
        record["sha256"] for record in records
    ]
    result["context_schema_version"] = SCHEMA_VERSION
    result["context_input_source"] = (
        "protocol_mapping_then_existing_original7_renderer"
    )
    result["context_input_validation"] = "passed"
    dataset_names = result["dataset"].astype(str).str.lower()
    result["context_content_identity_included"] = (
        (
            dataset_names.eq("case") & bool(include_case_title)
        )
        | (
            dataset_names.eq("eevr") & bool(include_eevr_scene)
        )
        | (
            dataset_names.eq("maus") & bool(include_maus_level)
        )
    ).astype(int)
    result["context_detail_policy"] = np.select(
        [
            dataset_names.eq("eevr") & bool(include_eevr_scene),
            dataset_names.eq("case") & bool(include_case_title),
            dataset_names.eq("maus") & bool(include_maus_level),
        ],
        [
            "objective_deployment_observable_scene_only",
            "official_content_title_only_no_emotion_category",
            "observable_nback_task_demand",
        ],
        default="generic_protocol_context",
    )
    result["context_detail_source"] = np.select(
        [
            dataset_names.eq("eevr") & bool(include_eevr_scene),
            dataset_names.eq("case") & bool(include_case_title),
            dataset_names.eq("maus") & bool(include_maus_level),
        ],
        [
            "context_text_original_objective_scene_not_teacher_text",
            "official_case_content_title",
            "protocol_context_sig_nback_level",
        ],
        default="none_generic_protocol_only",
    )
    return result


def audit_original7(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "context_schema_version",
        "context_text",
        *(f"context7_{slot}" for slot in SLOTS),
    }
    missing = sorted(required - set(frame.columns))
    legacy = sorted(LEGACY_SCHEMA_COLUMNS & set(frame.columns))
    invalid: list[str] = []
    duration_mismatches: list[str] = []
    if not missing and not legacy:
        for row in frame.to_dict(orient="records"):
            spec = {
                slot: str(row[f"context7_{slot}"])
                for slot in SLOTS
            }
            try:
                validate_spec(spec)
            except ValueError:
                invalid.append(str(row.get("sample_id", "")))
                if len(invalid) >= 10:
                    break
            dataset = str(row.get("dataset", "")).lower()
            seconds = row.get("source_duration_seconds")
            if (
                dataset in {"eevr", "maus"}
                and seconds is not None
                and not pd.isna(seconds)
                and spec["duration"] != duration_bucket(seconds)
            ):
                duration_mismatches.append(
                    str(row.get("sample_id", ""))
                )
                if len(duration_mismatches) >= 10:
                    break
    versions = (
        sorted(frame["context_schema_version"].astype(str).unique())
        if "context_schema_version" in frame
        else []
    )
    passed = (
        not missing
        and not legacy
        and not invalid
        and not duration_mismatches
        and versions == [SCHEMA_VERSION]
    )
    return {
        "status": "pass" if passed else "fail",
        "schema_version": SCHEMA_VERSION,
        "rows": int(len(frame)),
        "missing_columns": missing,
        "legacy_columns": legacy,
        "invalid_sample_ids": invalid,
        "duration_mismatch_sample_ids": duration_mismatches,
        "versions": versions,
    }
