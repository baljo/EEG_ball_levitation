# Diagnose runtime vector vs EI training features (log10 space), show order/DSP issues.
# Thomas Vikström, 2025-11-12 22:30 (Europe/Helsinki)

import numpy as np

EPS = 1e-12  # must match your DSP
Xtr = np.load("src/training.npy").astype(np.float32)
Xsrc = np.log10(np.maximum(Xtr, EPS))

MEAN = np.load("src/ei_scaler_mean.npy").astype(np.float32)
STD  = np.load("src/ei_scaler_std.npy").astype(np.float32)
STD_SAFE = np.where(STD == 0.0, 1.0, STD)

def diagnose_runtime_vector(features_log10: np.ndarray):
    x = np.asarray(features_log10, dtype=np.float32).ravel()
    assert x.size == MEAN.size, f"Feature length {x.size} != scaler {MEAN.size}"

    z = (x - MEAN) / STD_SAFE
    print(f"[Z] min/mean/max: {z.min():.3f} / {z.mean():.3f} / {z.max():.3f}")

    # Correlation with EI training rows (same space, same order expected)
    for i in (0, len(Xsrc)//3, (2*len(Xsrc))//3):
        print(f"[corr vs train {i}]: {np.corrcoef(x, Xsrc[i])[0,1]:.3f}")

    # If this is high but the line above is low → ORDER mismatch (same distribution, shuffled order)
    print(f"[corr sorted↔sorted]: {np.corrcoef(np.sort(x), np.sort(Xsrc[0]))[0,1]:.3f}")

    # Show the 10 worst-matching indices to locate the offending block(s)
    bad = np.argsort(z)[:10]
    print("Worst Z indices:", list(map(int, bad)))
    print("Worst Z values :", [float(z[i]) for i in bad])

# Example call:
# diagnose_runtime_vector(features)  # 'features' must already be log10-space
