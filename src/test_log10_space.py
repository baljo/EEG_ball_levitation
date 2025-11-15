# Verifies what space the EI NPYs are in, and matches them to your runtime.
# 2025-11-12 22:20 — Thomas Vikström

import numpy as np

Xtr = np.load("src/training.npy").astype(np.float32)
print("EI train NPY (as-is):   min/mean/max =", Xtr.min(), Xtr.mean(), Xtr.max())

Xtr_log = np.log10(np.maximum(Xtr, 1e-12))
print("EI train after log10(): min/mean/max =", Xtr_log.min(), Xtr_log.mean(), Xtr_log.max())
