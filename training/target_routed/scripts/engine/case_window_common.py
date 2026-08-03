"""Shared paths, deterministic splits, metrics, and serialization."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = HERE / "case_window_plan.yaml"


def configure_console() -> None:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_plan(path: str | Path = DEFAULT_PLAN) -> dict[str, Any]:
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


def work_dir(plan: dict[str, Any]) -> Path:
    return resolve(plan, plan["work_dir"])


def json_dump(path: Path, payload: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return [safe(item) for item in value.tolist()]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_seed(seed: int, dataset: str = "case") -> int:
    digest = hashlib.sha256(dataset.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little", signed=False)
    return (int(seed) + offset) % (2**32)


def participant_split(
    participant_ids: Iterable[str],
    seed: int,
    holdout_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    """Keep the historical CASE holdout as test, then draw validation.

    The former pipeline used the first shuffled 15% as validation.  The
    window experiment reserves that exact set as a true test set and uses the
    next shuffled 15% for model selection.
    """
    participants = np.asarray(sorted(set(map(str, participant_ids))))
    rng = np.random.default_rng(dataset_seed(seed))
    rng.shuffle(participants)
    n_test = max(1, int(round(len(participants) * holdout_fraction)))
    n_validation = max(
        1, int(round(len(participants) * validation_fraction))
    )
    if n_test + n_validation >= len(participants):
        raise ValueError("participant split leaves no training participants")
    return {
        "train": sorted(participants[n_test + n_validation :].tolist()),
        "validation": sorted(
            participants[n_test : n_test + n_validation].tolist()
        ),
        "test": sorted(participants[:n_test].tolist()),
    }


def ccc(prediction: Iterable[float], truth: Iterable[float]) -> float:
    prediction = np.asarray(list(prediction), dtype=float)
    truth = np.asarray(list(truth), dtype=float)
    if len(prediction) < 2:
        return float("nan")
    denominator = (
        prediction.var()
        + truth.var()
        + (prediction.mean() - truth.mean()) ** 2
    )
    if denominator <= 1e-12:
        return 0.0
    covariance = np.mean(
        (prediction - prediction.mean()) * (truth - truth.mean())
    )
    return float(2.0 * covariance / denominator)


def regression_metrics(
    prediction: Iterable[float], truth: Iterable[float]
) -> dict[str, float | int]:
    prediction = np.asarray(list(prediction), dtype=float)
    truth = np.asarray(list(truth), dtype=float)
    error = prediction - truth
    return {
        "n": int(len(truth)),
        "ccc": ccc(prediction, truth),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def binary_metrics(
    prediction: Iterable[float],
    truth: Iterable[float],
    threshold: float,
) -> dict[str, float]:
    prediction = np.asarray(list(prediction), dtype=float)
    truth = np.asarray(list(truth), dtype=float)
    predicted = prediction >= threshold
    expected = truth >= threshold
    tp = int(np.sum(predicted & expected))
    fp = int(np.sum(predicted & ~expected))
    fn = int(np.sum(~predicted & expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predicted == expected)),
        "f1": float(f1),
    }


def participant_macro_ccc(
    participant_ids: Iterable[str],
    prediction: Iterable[float],
    truth: Iterable[float],
) -> float:
    participant_ids = np.asarray(list(participant_ids), dtype=object)
    prediction = np.asarray(list(prediction), dtype=float)
    truth = np.asarray(list(truth), dtype=float)
    values = []
    for participant in sorted(set(participant_ids.tolist())):
        selected = participant_ids == participant
        if int(selected.sum()) >= 2:
            values.append(ccc(prediction[selected], truth[selected]))
    return float(np.nanmean(values)) if values else float("nan")


def yaml_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate."""
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            rendered = ""
        elif isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.6g}"
        else:
            rendered = str(value)
        return (
            rendered.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    lines = [
        "| " + " | ".join(cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)
