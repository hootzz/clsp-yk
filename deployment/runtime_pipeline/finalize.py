"""Run PaPaGEI-P and the v3 per-target estimator over merged JSONL records."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


PAPAGEI_DIM = 512
TARGET_PPG_SAMPLES = 1_250
HERE = Path(__file__).resolve().parent
TEXT_CLSP = HERE.parent
DEFAULT_MODEL_DIR = TEXT_CLSP / "mem_hqs_affect" / "model_v3"
DEFAULT_CHECKPOINT = (
    TEXT_CLSP
    / "mem_hqs_affect"
    / "outputs"
    / "context7_gpt"
    / "model_v3"
    / "full.pt"
)


def clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(1.0, float(value))), 4)


def parse_embedding(record: dict[str, Any]) -> list[float] | None:
    for key in ("ppg_embedding", "embedding", "ppg_features"):
        value = record.get(key)
        if isinstance(value, list) and len(value) == PAPAGEI_DIM:
            return [float(item) for item in value]
    columns = [f"ppg_f{index}" for index in range(PAPAGEI_DIM)]
    if all(column in record for column in columns):
        return [float(record[column]) for column in columns]
    return None


def mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot pool an empty embedding list")
    if any(len(vector) != PAPAGEI_DIM for vector in vectors):
        raise ValueError("All pooled embeddings must be 512-dimensional")
    count = float(len(vectors))
    return [
        sum(vector[index] for vector in vectors) / count
        for index in range(PAPAGEI_DIM)
    ]


def session_id(record: dict[str, Any], explicit_field: str | None = None) -> str:
    if explicit_field and record.get(explicit_field) not in (None, ""):
        return str(record[explicit_field])
    for key in ("session_id", "video_sample_id", "base_sample_id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    value = str(
        record.get("sample_id")
        or record.get("timestamp")
        or record.get("window_end_ms")
        or "sample"
    )
    value = re.sub(r"_w\d+(?:_\d+)?$", "", value)
    return re.sub(r"_window\d+$", "", value)


class PaPaGEIPEmbedder:
    """PaPaGEI-P ResNet1D encoder for already preprocessed 1250-sample windows."""

    def __init__(
        self,
        root: str | Path,
        checkpoint: str | Path,
        device: str | None = None,
    ):
        import numpy as np
        import torch

        self.np = np
        self.torch = torch
        root = Path(root).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"PaPaGEI repository not found: {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PaPaGEI-P checkpoint not found: {checkpoint}")
        sys.path.insert(0, str(root))
        from linearprobing.utils import load_model_without_module_prefix
        from models.resnet import ResNet1D

        network = ResNet1D(
            in_channels=1,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=PAPAGEI_DIM,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = load_model_without_module_prefix(network, str(checkpoint))
        self.model.to(self.device).eval()

    def embed(self, ppg: list[float]) -> list[float]:
        if len(ppg) != TARGET_PPG_SAMPLES:
            raise ValueError(
                f"Expected {TARGET_PPG_SAMPLES} preprocessed PPG samples, got {len(ppg)}"
            )
        array = self.np.asarray(ppg, dtype=self.np.float32)[None, None, :]
        tensor = self.torch.from_numpy(array).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
            embedding = output[0] if isinstance(output, (tuple, list)) else output
        vector = embedding.detach().cpu().numpy()[0]
        if len(vector) != PAPAGEI_DIM:
            raise ValueError(f"PaPaGEI-P returned {len(vector)} dimensions")
        return [float(value) for value in vector]


def _load_v3_module(model_dir: Path):
    model_file = model_dir / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"v3 model.py not found: {model_file}")
    spec = importlib.util.spec_from_file_location("mem_hqs_model_v3_runtime", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unwrap_state(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "state_dict", "model_state"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


class StateEstimatorV3:
    """Adapter for mem_hqs_affect/model_v3's per-target-head checkpoint."""

    ppg_encoder = "PaPaGEI-P"

    def __init__(
        self,
        checkpoint: str | Path,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        text_model_name: str = "distilbert-base-uncased",
        projection_dim: int = 256,
        projection_dropout: float = 0.1,
        ppg_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        text_max_length: int = 96,
        device: str | None = None,
    ):
        import torch

        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"v3 checkpoint not found: {checkpoint}\n"
                "Train config_512_context7_gpt.yaml first or pass --ckpt."
            )
        module = _load_v3_module(Path(model_dir).resolve())
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = module.MultimodalStateEstimator(
            text_model_name=text_model_name,
            ppg_input_dim=PAPAGEI_DIM,
            projection_dim=projection_dim,
            projection_dropout=projection_dropout,
            ppg_hidden_dim=ppg_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            text_init_ckpt=None,
            text_max_length=text_max_length,
        ).to(self.device)
        state = _unwrap_state(
            torch.load(checkpoint, map_location=self.device, weights_only=False)
        )
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with mem_hqs_affect/model_v3. "
                "The old hqs-clsp single 3-output-head checkpoint is intentionally unsupported."
            ) from exc
        self.model.eval()
        self.checkpoint = str(checkpoint)

    def predict(
        self,
        text_context: str,
        ppg_embedding: list[float],
    ) -> dict[str, float | None]:
        if len(ppg_embedding) != PAPAGEI_DIM:
            raise ValueError(f"Expected a 512-d PPG embedding, got {len(ppg_embedding)}")
        torch = self.torch
        features = torch.tensor(
            [ppg_embedding], dtype=torch.float32, device=self.device
        )
        mask = torch.ones(1, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            output = self.model(
                texts=[text_context],
                ppg_features=features,
                device=self.device,
                ppg_mask=mask,
            )["preds"]
        # Every v3 training target is normalized to [0,1].
        return {
            "cognitive_load": clamp01(output["cognitive_load"][0].item()),
            "valence": clamp01(output["valence"][0].item()),
            "arousal": clamp01(output["arousal"][0].item()),
        }


class MockEmbedder:
    def embed(self, ppg: list[float]) -> list[float]:
        if len(ppg) != TARGET_PPG_SAMPLES:
            raise ValueError(
                f"Expected {TARGET_PPG_SAMPLES} PPG samples, got {len(ppg)}"
            )
        mean = sum(float(value) for value in ppg) / len(ppg)
        return [mean] * PAPAGEI_DIM


class MockStateEstimator:
    checkpoint = "mock"
    ppg_encoder = "mock"

    def predict(
        self,
        text_context: str,
        ppg_embedding: list[float],
    ) -> dict[str, float | None]:
        value = clamp01(sum(ppg_embedding) / len(ppg_embedding))
        return {
            "cognitive_load": value,
            "valence": value,
            "arousal": value,
        }


def embedding_for_record(
    record: dict[str, Any],
    embedder: PaPaGEIPEmbedder | MockEmbedder | None,
) -> list[float]:
    existing = parse_embedding(record)
    if existing is not None:
        return existing
    if embedder is None:
        raise ValueError(
            f"Record {record.get('sample_id', '?')} has no 512-d embedding "
            "while --embedding-only is active"
        )
    ppg = record.get("ppg")
    if not isinstance(ppg, list):
        raise ValueError(f"Record {record.get('sample_id', '?')} has no PPG window")
    return embedder.embed([float(value) for value in ppg])


def finalize_window_records(
    records: list[dict[str, Any]],
    embedder: PaPaGEIPEmbedder | MockEmbedder | None,
    estimator: StateEstimatorV3 | MockStateEstimator,
    keep_ppg: bool,
    keep_embedding: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        record = copy.deepcopy(source)
        record.setdefault("sample_id", str(record.get("timestamp") or index))
        embedding = embedding_for_record(record, embedder)
        record["measures"] = estimator.predict(
            str(record.get("text_context") or record.get("text") or ""),
            embedding,
        )
        record["state_estimation"] = {
            "model": "mem_hqs_affect/model_v3",
            "checkpoint": estimator.checkpoint,
            "ppg_encoder": estimator.ppg_encoder,
            "target_scale": "[0,1]",
        }
        if keep_embedding:
            record["ppg_embedding"] = embedding
        if not keep_ppg:
            record.pop("ppg", None)
        output.append(record)
    return output


def finalize_pooled_records(
    records: list[dict[str, Any]],
    embedder: PaPaGEIPEmbedder | MockEmbedder | None,
    estimator: StateEstimatorV3 | MockStateEstimator,
    session_id_field: str | None,
    keep_ppg: bool,
    keep_embedding: bool,
) -> list[dict[str, Any]]:
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for source in records:
        key = session_id(source, session_id_field)
        pack = groups.setdefault(
            key,
            {"representative": copy.deepcopy(source), "embeddings": [], "ids": []},
        )
        pack["embeddings"].append(embedding_for_record(source, embedder))
        pack["ids"].append(str(source.get("sample_id") or source.get("timestamp")))

    pooled_records: list[dict[str, Any]] = []
    for key, pack in groups.items():
        record = pack["representative"]
        embedding = mean_vectors(pack["embeddings"])
        record["sample_id"] = key
        record["session_id"] = key
        record["measures"] = estimator.predict(
            str(record.get("text_context") or record.get("text") or ""),
            embedding,
        )
        record["pooling"] = {
            "method": "mean",
            "unit": "session",
            "window_count": len(pack["embeddings"]),
            "source_sample_ids": pack["ids"],
        }
        record["state_estimation"] = {
            "model": "mem_hqs_affect/model_v3",
            "checkpoint": estimator.checkpoint,
            "ppg_encoder": estimator.ppg_encoder,
            "target_scale": "[0,1]",
        }
        if keep_embedding:
            record["ppg_embedding"] = embedding
        if not keep_ppg:
            record.pop("ppg", None)
        pooled_records.append(record)
    return pooled_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--text-model-name", default="distilbert-base-uncased")
    parser.add_argument("--text-max-length", type=int, default=96)
    parser.add_argument("--device")
    parser.add_argument("--papagei-root", type=Path)
    parser.add_argument("--papagei-ckpt", type=Path)
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--pool-windows", action="store_true")
    parser.add_argument("--session-id-field")
    parser.add_argument("--keep-ppg", action="store_true")
    parser.add_argument("--keep-embedding", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    with args.input_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise SystemExit(f"[finalize] empty input: {args.input_path}")

    if args.mock:
        embedder: PaPaGEIPEmbedder | MockEmbedder | None = MockEmbedder()
        estimator: StateEstimatorV3 | MockStateEstimator = MockStateEstimator()
    else:
        estimator = StateEstimatorV3(
            checkpoint=args.ckpt,
            model_dir=args.model_dir,
            text_model_name=args.text_model_name,
            text_max_length=args.text_max_length,
            device=args.device,
        )
        if args.embedding_only:
            embedder = None
        else:
            if args.papagei_root is None or args.papagei_ckpt is None:
                raise SystemExit(
                    "[finalize] raw PPG needs --papagei-root and --papagei-ckpt; "
                    "use --embedding-only when records already contain 512-d embeddings"
                )
            embedder = PaPaGEIPEmbedder(
                args.papagei_root, args.papagei_ckpt, args.device
            )

    if args.pool_windows:
        output = finalize_pooled_records(
            records,
            embedder,
            estimator,
            args.session_id_field,
            args.keep_ppg,
            args.keep_embedding,
        )
    else:
        output = finalize_window_records(
            records,
            embedder,
            estimator,
            args.keep_ppg,
            args.keep_embedding,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[finalize] wrote {len(output)} records -> {args.out}")
    print(f"  model: {'mock' if args.mock else 'mem_hqs_affect/model_v3'}")
    print(
        "  PPG encoder: "
        + (
            "mock"
            if args.mock
            else ("existing embeddings" if args.embedding_only else "PaPaGEI-P")
        )
    )
    print(f"  pooling: {'session mean' if args.pool_windows else 'window-level'}")


if __name__ == "__main__":
    main()
