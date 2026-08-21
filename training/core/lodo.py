"""LODO — Leave-One-Dataset-Out zero-shot evaluation.

For each dataset D, train on all other datasets and evaluate only the targets
available in D. This measures cross-dataset transfer to an unseen dataset.

Reports target-wise CCC for continuous prediction, Acc/F1 with a 0.5 cutoff,
and per-dataset standardized CCC as a calibration diagnostic.

Usage: python lodo.py --config config.yaml
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import MultiDatasetAffect, collate
from model import TARGETS
from train import train_model


def ccc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < 2:
        return float("nan")
    pm, ym = p.mean(), y.mean()
    d = p.var() + y.var() + (pm - ym) ** 2
    return float(2 * ((p - pm) * (y - ym)).mean() / d) if d > 1e-12 else 0.0


def bin_acc_f1(p, y, cut=None):
    # If cut is None, use the label median for high/low binarization.
    p, y = np.asarray(p), np.asarray(y)
    c = float(np.median(y)) if cut is None else cut
    pb, yb = (p >= c).astype(int), (y >= c).astype(int)
    acc = float((pb == yb).mean())
    tp = int(((pb == 1) & (yb == 1)).sum()); fp = int(((pb == 1) & (yb == 0)).sum()); fn = int(((pb == 0) & (yb == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, f1


@torch.no_grad()
def evaluate(model, device, cfg, dataset_name):
    ds = MultiDatasetAffect(cfg["paths"]["manifest_csv"], cfg["paths"].get("ppg_csv"),
                            ppg_dim=int(cfg["model"].get("ppg_input_dim", 512)),
                            include_datasets=[dataset_name])
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    model.eval()
    pred = {t: [] for t in TARGETS}; true = {t: [] for t in TARGETS}; msk = {t: [] for t in TARGETS}
    for b in loader:
        out = model(texts=b["text"], ppg_features=b["ppg_features"], device=device, ppg_mask=b["ppg_mask"])
        for t in TARGETS:
            pred[t].append(out["preds"][t].cpu().numpy())
            true[t].append(b["targets"][t].numpy()); msk[t].append(b["masks"][t].numpy())
    res = {}
    for t in TARGETS:
        m = np.concatenate(msk[t]).astype(bool)
        if m.sum() < 2:
            continue
        p = np.concatenate(pred[t])[m]; y = np.concatenate(true[t])[m]
        acc, f1 = bin_acc_f1(p, y)
        ps = (p - p.mean()) / (p.std() + 1e-8); ys = (y - y.mean()) / (y.std() + 1e-8)
        res[t] = {"n": int(m.sum()), "CCC": round(ccc(p, y), 3),
                  "CCC_std": round(ccc(ps, ys), 3), "Acc": round(acc, 3), "F1": round(f1, 3)}
    return res


def main(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    all_ds = sorted(MultiDatasetAffect(cfg["paths"]["manifest_csv"]).df["dataset"].unique().tolist())
    print(f"datasets: {all_ds}\n")
    print(f"{'held-out':10s} {'target':16s} {'n':>5s} {'CCC':>7s} {'CCCstd':>7s} {'Acc':>6s} {'F1':>6s}")
    print("-" * 62)
    for D in all_ds:
        model, device = train_model(cfg, exclude_datasets=[D], verbose=False)
        res = evaluate(model, device, cfg, D)
        for t, r in res.items():
            print(f"{D:10s} {t:16s} {r['n']:5d} {r['CCC']:7.3f} {r['CCC_std']:7.3f} {r['Acc']:6.3f} {r['F1']:6.3f}")
        print("-" * 62)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
