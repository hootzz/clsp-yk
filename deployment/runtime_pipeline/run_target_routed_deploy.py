"""End-to-end deployment runner for the FINAL target-routed model.

Wires the pieces that the existing finalize.py lineages do NOT support:

    merge output (datastream.jsonl)
      -> original-7 context rendering   (context_text_original7)
      -> frozen PaPaGEI-P 512-d embedding
      -> target-routed model (fusion_mode="target_routed_direct")  (finalize_target_routed)
      -> optional target-wise EB offset (K>0)                       (eb_personalization)
      -> V / A / C in [0,1]
      -> final_datastream.jsonl

It consumes the same `datastream.jsonl` schema produced by the collectors->merge front-end (fields: user_state, ppg[1250]),
so it drops into the existing deployment branch by swapping only the model back-end.

Modes:
- Default (no --calibration): K=0 zero-shot population prediction.
- With --calibration + --enrollment: K>0 personalized prediction. Enrollment records
  carry a `label` field {valence, arousal, cognitive_load}; the first K per user set a
  leakage-free EB output offset that is added to later predictions.

Usage:
    python run_target_routed_deploy.py --in datastream.jsonl --out final.jsonl \
        --ckpt <full.pt> --papagei-ckpt <papagei_p.pt> \
        [--calibration eb_calibration_sample.json --enrollment enrollment.jsonl --enroll-k 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import context_text_original7 as ctx
from finalize_target_routed import StateEstimatorTR, DEFAULT_MODEL_DIR, PAPAGEI_DIM
from eb_personalization import EBPersonalizer

TARGETS = ("valence", "arousal", "cognitive_load")


def _load_papagei(model_dir: Path, root: str | None, ckpt: str | None):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    import papagei_embed
    kwargs = {}
    if root:
        kwargs["root"] = root
    if ckpt:
        kwargs["ckpt"] = ckpt
    return papagei_embed.PapageiP(device="cpu", **kwargs)


def _duration_seconds(rec: dict):
    dig = (rec.get("user_state") or {}).get("digital") or {}
    for key in ("duration_seconds", "duration"):
        if isinstance(dig.get(key), (int, float)):
            return float(dig[key])
    return None


def _base_predict(rec, enc, est):
    """Render original-7 context, embed PPG, and return (text, base V/A/C)."""
    us = rec.get("user_state") or {}
    detail = (us.get("digital") or {}).get("usage", "")
    text7 = ctx.build_context_text(us, observable_detail=detail,
                                   duration_seconds=_duration_seconds(rec))
    emb = np.asarray(enc.embed(np.array([rec["ppg"]], dtype=np.float32))).reshape(-1)[:PAPAGEI_DIM]
    return text7, est.predict(text7, emb.tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--papagei-root", default=None)
    ap.add_argument("--papagei-ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0)
    # --- optional EB personalization (K>0). Omit for K=0 population. ---
    ap.add_argument("--calibration", default=None, help="EB variance-component bundle (JSON)")
    ap.add_argument("--enrollment", default=None, help="enrollment JSONL (records + `label`)")
    ap.add_argument("--enroll-k", type=int, default=4)
    ap.add_argument("--user-id-field", default="user_id")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    enc = _load_papagei(model_dir, args.papagei_root, args.papagei_ckpt)
    est = StateEstimatorTR(args.ckpt, model_dir=model_dir, device="cpu")

    # optional EB layer
    personalizer = None
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        cal = {k: v for k, v in cal.items() if not k.startswith("_")}
        personalizer = EBPersonalizer(cal, enroll_k=args.enroll_k)
        if args.enrollment:
            for line in open(args.enrollment, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec.get("ppg"), list) or len(rec["ppg"]) != 1250:
                    continue
                labels = rec.get("label") or {}
                if not labels:
                    continue
                user = str(rec.get(args.user_id_field, "user"))
                _, base = _base_predict(rec, enc, est)
                for t in TARGETS:
                    if labels.get(t) is not None:
                        personalizer.enroll(user, t, float(labels[t]), base[t])

    n = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec.get("ppg"), list) or len(rec["ppg"]) != 1250:
                continue
            text7, base = _base_predict(rec, enc, est)
            rec["context_text_original7"] = text7
            rec["model_version"] = "target_routed_direct"
            if personalizer is not None:
                user = str(rec.get(args.user_id_field, "user"))
                pers = personalizer.apply(user, base)
                rec["measures"] = {k: round(v, 4) for k, v in pers.items()}
                rec["measures_base"] = {k: round(v, 4) for k, v in base.items()}
                rec["personalization"] = {"mode": "EB_K>0",
                                          "support": {t: personalizer.support_count(user, t) for t in TARGETS}}
            else:
                rec["measures"] = {k: round(v, 4) for k, v in base.items()}
                rec["personalization"] = {"mode": "population_K=0"}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    mode = "EB personalized (K>0)" if personalizer else "population (K=0)"
    print(f"[run_target_routed_deploy] {mode}: wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
