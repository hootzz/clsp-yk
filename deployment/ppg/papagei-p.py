import csv
import numpy as np
import torch
import sys
sys.path.append("D:/2026/test/papagei-foundation-model")

from linearprobing.utils import load_model_without_module_prefix
from models.resnet import ResNet1D

# ── 설정 ─────────────────────────────────────────────────
PROCESSED_CSV  = r"data/processed_20260414_190911.csv"
WEIGHT_PATH    = r"papagei-foundation-model/weights/papagei_p.pt"
OUT_FILE       = r"output/embeddings_p.npy"
TARGET_SAMPLES = 1250

# ── 모델 로드 ────────────────────────────────────────────
model = ResNet1D(
    in_channels=1, base_filters=32, kernel_size=3,
    stride=2, groups=1, n_block=18, n_classes=512,
)
model = load_model_without_module_prefix(model, WEIGHT_PATH)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"모델 로드 완료: {device}")

# ── CSV에서 세그먼트 로드 ────────────────────────────────
segments = []
with open(PROCESSED_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["flatline_skipped"]) == 1:
            continue
        segments.append([float(row[f"ppg_{i}"]) for i in range(TARGET_SAMPLES)])

segments = np.asarray(segments, dtype=np.float32)
print(f"세그먼트: {segments.shape}")

# ── 임베딩 추출 ──────────────────────────────────────────
signal_tensor = torch.from_numpy(segments).unsqueeze(1).to(device)  # (N, 1, 1250)

with torch.inference_mode():
    outputs = model(signal_tensor)
    embeddings = outputs[0].cpu().numpy()

print(f"임베딩: {embeddings.shape}")
print(f"평균: {embeddings.mean():.4f} | std: {embeddings.std():.4f} | NaN: {np.isnan(embeddings).any()}")

np.save(OUT_FILE, embeddings)
print(f"저장 완료: {OUT_FILE}")