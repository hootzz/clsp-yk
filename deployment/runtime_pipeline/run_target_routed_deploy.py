"""End-to-end deployment runner for the FINAL target-routed model.

Wires the pieces that the existing finalize.py lineages do NOT support:

    merge output (datastream.jsonl)
      -> original-7 context rendering   (context_text_original7)
      -> frozen PaPaGEI-P 512-d embedding
      -> target-routed model (fusion_mode="target_routed_direct")  (finalize_target_routed)
      -> optional target-wise EB offset (K>0)
      -> V / A / C in [0,1]
      -> final_datastream.jsonl

This does NOT edit any original file. It consumes the same `datastream.jsonl`
schema produced by the collectors->merge front-end (fields: user_state, ppg[1250]),
so it drops into the existing deployment branch by swapping only the model back-end
(the draft finalize.py uses model_multimodal + PaPaGEI-S; this uses the final
target-routed model_v3 + PaPaGEI-P).

Usage:
    python run_target_routed_deploy.py \
        --in  datastream.jsonl \
        --out final_datastream_tr.jsonl \
        --ckpt   <target_routed seed_42 full.pt> \
        --papagei-root <papagei-foundation-model> \
        --papagei-ckpt <papagei_p.pt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import context_text_original7 as ctx
from finalize_target_routed import StateEstimatorTR, DEFAULT_MODEL_DIR, PAPAGEI_DIM


def _load_papagei(model_dir: Path, root: str | None, ckpt: str | None):
    """Reuse the training-time PaPaGEI-P embedder (variant 'p')."""
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    import papagei_embed  # from model_v3
    kwargs = {}
    if root:
        kwargs["root"] = root
    if ckpt:
        kwargs["ckpt"] = ckpt
    return papagei_embed.PapageiP(device="cpu", **kwargs)


def _duration_seconds(rec: dict) -> float | None:
    dig = (rec.get("user_state") or {}).get("digital") or {}
    for key in ("duration_seconds", "duration"):
        if isinstance(dig.get(key), (int, float)):
            return float(dig[key])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="merge output datastream.jsonl")
    ap.add_argument("--out", required=True, help="output final_datastream.jsonl")
    ap.add_argument("--ckpt", required=True, help="target-routed checkpoint full.pt")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--papagei-root", default=None)
    ap.add_argument("--papagei-ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0, help="process only first N records (0=all)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    enc = _load_papagei(model_dir, args.papagei_root, args.papagei_ckpt)
    est = StateEstimatorTR(args.ckpt, model_dir=model_dir, device="cpu")

    n = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            us = rec.get("user_state") or {}
            ppg = rec.get("ppg")
            if not isinstance(ppg, list) or len(ppg) != 1250:
                continue
            detail = (us.get("digital") or {}).get("usage", "")
            text7 = ctx.build_context_text(us, observable_detail=detail,
                                           duration_seconds=_duration_seconds(rec))
            emb = np.asarray(enc.embed(np.array([ppg], dtype=np.float32))).reshape(-1)[:PAPAGEI_DIM]
            vac = est.predict(text7, emb.tolist())
            rec["measures"] = {k: round(v, 4) for k, v in vac.items()}
            rec["context_text_original7"] = text7
            rec["model_version"] = "target_routed_direct"
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"[run_target_routed_deploy] wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
