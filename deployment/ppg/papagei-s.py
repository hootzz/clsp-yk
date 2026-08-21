import csv
import numpy as np
import torch
import os, sys
sys.path.append(os.environ.get("PAPAGEI_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "papagei-foundation-model")))

from linearprobing.utils import load_model_without_module_prefix
from models.resnet import ResNet1DMoE

PROCESSED_CSV  = r"data/processed_20260414_190911.csv"
WEIGHT_PATH    = r"papagei-foundation-model/weights/papagei_s.pt"
OUT_FILE       = r"output/embeddings_s.npy"
TARGET_SAMPLES = 1250

model = ResNet1DMoE(
    in_channels=1, base_filters=32, kernel_size=3,
    stride=2, groups=1, n_block=18, n_classes=512, n_experts=3,
)
model = load_model_without_module_prefix(model, WEIGHT_PATH)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded: {device}")

segments = []
with open(PROCESSED_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["flatline_skipped"]) == 1:
            continue
        segments.append([float(row[f"ppg_{i}"]) for i in range(TARGET_SAMPLES)])

segments = np.asarray(segments, dtype=np.float32)
print(f"Segments: {segments.shape}")

# PaPaGEI-S returns tuple: (embeddings, expert_outputs, gating_weights)
# outputs[0] = projection embedding Z (B, 512)
signal_tensor = torch.from_numpy(segments).unsqueeze(1).to(device)  # (N, 1, 1250)

with torch.inference_mode():
    outputs = model(signal_tensor)
    embeddings = outputs[0].cpu().numpy()

print(f"Embeddings: {embeddings.shape}")
print(f"Mean: {embeddings.mean():.4f} | std: {embeddings.std():.4f} | NaN: {np.isnan(embeddings).any()}")

np.save(OUT_FILE, embeddings)
print(f"Saved: {OUT_FILE}")
