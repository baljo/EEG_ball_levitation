# Extracts mean/std vectors from Edge Impulse NPY feature exports.
# 2025-11-12 20:15 — Thomas Vikström

import numpy as np
from pathlib import Path

# point these to the files you downloaded from EI's Dashboard → Data exports
train_npy = Path("src/training.npy")
test_npy  = Path("src/testing.npy")  # not needed for scaling, but can inspect

# --- Load feature arrays
X_train = np.load(train_npy)
X_test  = np.load(test_npy)

print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# --- Compute per-feature mean and std (population std, same as scikit-learn's transform)
mean = X_train.mean(axis=0).astype(np.float32)
std  = X_train.std(axis=0, ddof=0).astype(np.float32)

# --- Safety check: zero-variance features
std_safe = np.where(std == 0.0, 1.0, std)

# --- Save & print ready-to-paste arrays
np.save("ei_scaler_mean.npy", mean)
np.save("ei_scaler_std.npy", std_safe)

print("\nPaste these arrays into your inference code:")
print("MEAN = np.array([", ", ".join([f"{v:.6g}" for v in mean]), "], dtype=np.float32)")
print("STD  = np.array([", ", ".join([f"{v:.6g}" for v in std_safe]), "], dtype=np.float32)")

