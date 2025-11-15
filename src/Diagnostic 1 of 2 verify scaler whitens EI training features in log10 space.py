# Diagnostic 1/2: verify scaler whitens EI training features in *log10* space (should be ~0±1).
# 2025-11-12 22:13 (Europe/Helsinki) — Thomas Vikström

import numpy as np

EPS = 1e-12  # must match your generate_features
Xtr = np.load("src/training.npy").astype(np.float32)
Xsrc = np.log10(np.maximum(Xtr, EPS))

MEAN = np.load("src/ei_scaler_mean.npy").astype(np.float32)
STD  = np.load("src/ei_scaler_std.npy").astype(np.float32)
STD_SAFE = np.where(STD == 0.0, 1.0, STD)

assert Xsrc.shape[1] == MEAN.size == STD_SAFE.size, (Xsrc.shape, MEAN.shape, STD_SAFE.shape)

Z = (Xsrc - MEAN) / STD_SAFE
print("Whitened EI train (expected ~0±1):")
print(f"  global min/mean/max: {Z.min():.3f} / {Z.mean():.3f} / {Z.max():.3f}")
print(f"  per-feature mean avg: {Z.mean(axis=0).mean():.3f}")
print(f"  per-feature std  avg: {Z.std(axis=0, ddof=0).mean():.3f}")
