"""Target-routed inference adapter 
--------------------
This standalone adapter builds the model with the correct `fusion_mode` 
and `text_max_length=64` and loads the checkpoint.

Valence is a context-only route: its output does not depend on the PPG input
(verified at load time by `assert_valence_ppg_invariant`).

Usage (embedding already computed):
    est = StateEstimatorTR(ckpt, model_dir)
    est.predict(text_context, ppg_embedding_512)  # -> {'valence','arousal','cognitive_load'} in [0,1]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PAPAGEI_DIM = 512
# model.py that MATCHES the target-routed checkpoint = local training/core. Override with --model-dir.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "training" / "core"


def _load_model_module(model_dir: Path):
    model_dir = Path(model_dir).resolve()
    mp = model_dir / "model.py"
    if not mp.is_file():
        raise FileNotFoundError(f"model.py not found in {model_dir}")
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    spec = importlib.util.spec_from_file_location("tr_model", str(mp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unwrap_state(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model_state"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


class StateEstimatorTR:
    """target_routed_direct adapter — the one-line fix the original lacks."""

    fusion_mode = "target_routed_direct"

    def __init__(self, checkpoint: str | Path, model_dir: str | Path = DEFAULT_MODEL_DIR,
                 text_model_name: str = "distilbert-base-uncased", text_max_length: int = 64,
                 device: str | None = None):
        import torch

        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"target-routed checkpoint not found: {checkpoint}")
        module = _load_model_module(Path(model_dir))
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = module.MultimodalStateEstimator(
            text_model_name=text_model_name,
            ppg_input_dim=PAPAGEI_DIM,
            projection_dim=256,
            projection_dropout=0.1,
            ppg_hidden_dim=256,
            fusion_hidden_dim=256,
            text_init_ckpt=None,
            text_max_length=text_max_length,          # 64 (contract), not the finalize.py default 96
            fusion_mode=self.fusion_mode,             # <-- THE FIX (finalize.py omits this)
        ).to(self.device)
        state = _unwrap_state(torch.load(checkpoint, map_location=self.device, weights_only=False))
        self.model.load_state_dict(state, strict=True)  # succeeds because fusion_mode matches
        self.model.eval()
        self.checkpoint = str(checkpoint)

    def predict(self, text_context: str, ppg_embedding: list[float]) -> dict[str, float]:
        if len(ppg_embedding) != PAPAGEI_DIM:
            raise ValueError(f"Expected {PAPAGEI_DIM}-d PPG embedding, got {len(ppg_embedding)}")
        torch = self.torch
        feats = torch.tensor([ppg_embedding], dtype=torch.float32, device=self.device)
        mask = torch.ones(1, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            out = self.model(texts=[text_context], ppg_features=feats,
                             device=self.device, ppg_mask=mask)["preds"]
        return {k: _clamp01(out[k][0].item()) for k in ("valence", "arousal", "cognitive_load")}

    def assert_valence_ppg_invariant(self, text_context: str) -> bool:
        """Valence must not depend on PPG (context-only route)."""
        torch = self.torch
        vals = []
        for scale in (0.0, 1.0, 5.0):
            feats = torch.randn(1, PAPAGEI_DIM, device=self.device) * scale
            with torch.inference_mode():
                out = self.model(texts=[text_context], ppg_features=feats,
                                 device=self.device, ppg_mask=torch.ones(1, device=self.device))["preds"]
            vals.append(round(float(out["valence"][0]), 6))
        return len(set(vals)) == 1


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--text", required=True, help="rendered original-7 context text")
    ap.add_argument("--embedding", help="JSON list of 512 floats (else a smoke random embedding)")
    args = ap.parse_args()
    import torch  # noqa
    est = StateEstimatorTR(args.ckpt, args.model_dir)
    emb = json.loads(args.embedding) if args.embedding else [0.0] * PAPAGEI_DIM
    print(json.dumps(est.predict(args.text, emb), ensure_ascii=False))


if __name__ == "__main__":
    _main()
