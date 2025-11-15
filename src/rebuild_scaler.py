# Build μ/σ from EI training features *after* log10 so they match runtime (DO_LOG=True).
# Thomas Vikström, 2025-11-12 22:03 (Europe/Helsinki) — computes and saves ei_scaler_mean.npy / ei_scaler_std.npy

import numpy as np

Xtr = np.load("src/training.npy").astype(np.float32)

# Use the SAME epsilon as your runtime DSP (keep this consistent!)
EPS = 1e-12
Xsrc = np.log10(np.maximum(Xtr, EPS))

mean = Xsrc.mean(axis=0).astype(np.float32)
std  = Xsrc.std(axis=0, ddof=0).astype(np.float32)
std_safe = np.where(std == 0.0, 1.0, std)

np.save("ei_scaler_mean.npy", mean)
np.save("ei_scaler_std.npy", std_safe)

print("Scaler saved:", mean.shape[0], "features; mean[0:3] =", mean[:3], "std[0:3] =", std_safe[:3])
