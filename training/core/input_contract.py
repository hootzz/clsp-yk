"""Fingerprint the exact source inputs used to train and evaluate a checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_input_contract(cfg: dict) -> dict:
    """Return the data/split/model contract that C0 and E2 must share."""
    manifest = Path(cfg["paths"]["manifest_csv"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest was not found: {manifest}")
    ppg_value = cfg["paths"].get("ppg_csv")
    ppg = Path(ppg_value).resolve() if ppg_value else None
    if ppg is not None and not ppg.is_file():
        raise FileNotFoundError(f"PPG feature file was not found: {ppg}")

    comparison = {
        "manifest_sha256": _file_sha256(manifest),
        "ppg_sha256": _file_sha256(ppg) if ppg is not None else None,
        "seed": int(cfg.get("seed", 42)),
        "model": cfg.get("model", {}),
        "loss": cfg.get("loss", {}),
        "train": cfg.get("train", {}),
    }
    return {
        "format": "clsp_training_input_contract_v1",
        "manifest": str(manifest),
        "manifest_sha256": comparison["manifest_sha256"],
        "ppg_csv": str(ppg) if ppg is not None else None,
        "ppg_sha256": comparison["ppg_sha256"],
        "comparison": comparison,
        "comparison_sha256": _json_sha256(comparison),
    }


def load_contract(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_contract_match(
    recorded: dict,
    current: dict,
    *,
    checkpoint: Path,
) -> None:
    expected = str(recorded.get("comparison_sha256", ""))
    observed = str(current.get("comparison_sha256", ""))
    if not expected or expected != observed:
        raise RuntimeError(
            "checkpoint/input contract mismatch: the manifest, PPG features, "
            "split, model, or training settings changed after training. "
            f"checkpoint={checkpoint}; recorded={expected or '<missing>'}; "
            f"current={observed or '<missing>'}. Retrain the affected "
            "checkpoint before comparing it."
        )
