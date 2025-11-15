# EEG ball levitation: Muse→EI spectral→Keras/TFLite→Photon 2, with optional test mode for pasted EI feature vectors; 2025-11-15 10:15, Thomas Vikström
# Supports both live Muse streaming and offline inference from Edge Impulse feature rows to verify predictions match EI.

import os
import sys
import time
import re
import argparse
import numpy as np
import serial
from serial import SerialException
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)

# -------- Locate Edge Impulse processing-blocks --------
HERE = os.path.dirname(os.path.abspath(__file__))

candidate_roots = [
    os.path.join(HERE, "processing-blocks"),
    os.path.join(HERE, "..", "processing-blocks"),
]

pb_root = None
for root in candidate_roots:
    if (
        os.path.isdir(root)
        and os.path.isdir(os.path.join(root, "spectral_analysis"))
        and os.path.isdir(os.path.join(root, "common"))
    ):
        pb_root = root
        break

if pb_root is None:
    raise ImportError(
        "\n[EEG] Could not find Edge Impulse 'processing-blocks' with both "
        "'spectral_analysis' and 'common' as siblings.\n"
        "Expected structure:\n"
        "  src/processing-blocks/spectral_analysis/...\n"
        "  src/processing-blocks/common/spectrum.py\n"
    )

if pb_root not in sys.path:
    sys.path.insert(0, pb_root)
    print(f"[INFO] Using Edge Impulse processing-blocks from: {pb_root}")

from spectral_analysis import generate_features


# ----------------- Model selection and setup -----------------
MODEL_TFLITE = "EEG_float32.lite"
MODEL_H5 = "EEG_model_64.h5"

if os.path.exists(MODEL_H5):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_H5)
    INPUT_SHAPE = (1, model.input_shape[-1])
    EXPECTED_FEATURES = model.input_shape[-1]
    BACKEND = "keras"
    print(f"[INFO] Using Keras backend, model: {MODEL_H5}")

    def infer(features: np.ndarray):
        # features already in EI log-spectral (and optionally scaled) space
        x = features.reshape(1, -1).astype(np.float32)
        return model.predict(x, verbose=0)[0]

elif os.path.exists(MODEL_TFLITE):
    try:
        from tflite_runtime.interpreter import Interpreter
        backend = "tflite-runtime"
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        backend = "tensorflow.lite"

    interpreter = Interpreter(model_path=MODEL_TFLITE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    INPUT_SHAPE = input_details[0]["shape"]
    EXPECTED_FEATURES = int(np.prod(INPUT_SHAPE[1:]))
    BACKEND = backend
    print(f"[INFO] Using TFLite backend ({backend}), model: {MODEL_TFLITE}")

    def infer(features: np.ndarray):
        # features already in EI log-spectral (and optionally scaled) space
        x = features.reshape(INPUT_SHAPE).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"])[0]
else:
    raise FileNotFoundError(
        f"No model file found. Expected one of:\n  {MODEL_H5}\n  {MODEL_TFLITE}"
    )


# ================== CONFIG ==================

BOARD_ID = BoardIds.MUSE_2_BOARD.value
FS = 256.0                     # Hz
WINDOW_SECONDS = 2.0           # window size in seconds
STRIDE_SECONDS = 0.250         # stride between decisions
STABILITY_WINDOWS = 4          # number of last decisions to require stable target
SMOOTH_WINDOWS = 4             # number of last probabilities to average
USE_MEDIAN_SMOOTH = True       # True=median smoothing, False=mean smoothing
TARGET_THRESHOLD = 0.7

# Edge Impulse Spectral Analysis params (must match project)
IMPLEMENTATION_VERSION = 4
DRAW_GRAPHS = False
AXES = ["eeg_1", "eeg_2", "eeg_3", "eeg_4"]

SCALE_AXES = 1.0
INPUT_DECIMATION_RATIO = 1
FILTER_TYPE = "none"
FILTER_CUTOFF = 0
FILTER_ORDER = 0
ANALYSIS_TYPE = "FFT"
FFT_LENGTH = 64
SPECTRAL_PEAKS_COUNT = 0
SPECTRAL_PEAKS_THRESHOLD = 0
SPECTRAL_POWER_EDGES = "0"
DO_LOG_IN_BLOCK = True          # let EI block take log of spectrum (matches training)
DO_FFT_OVERLAP = True
WAVELET_LEVEL = 1
WAVELET = ""
EXTRA_LOW_FREQ = False

LABELS = ["calm", "non_calm", "sleep"]
TARGET_CLASS_INDEX = 1

SERIAL_ENABLED = True
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200

DEBUG_MAX_WINDOWS = 40
DEBUG_EVERY_N = 10
_feature_mismatch_warned = False


# ================== HELPERS ==================


def init_board() -> BoardShim:
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    print("[INFO] Preparing Muse / BrainFlow session...")
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream(45000)
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    print(f"[INFO] EEG channels: {eeg_channels}, sampling rate: {fs} Hz")
    time.sleep(1.0)
    return board


def init_serial():
    if not SERIAL_ENABLED:
        print("[INFO] SERIAL_ENABLED = False, running without Photon 2.")
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        print(f"[INFO] Serial connected on {SERIAL_PORT} @ {SERIAL_BAUD}.")
        return ser
    except SerialException as e:
        print(f"[WARN] Could not open serial port {SERIAL_PORT}: {e}")
        print("[WARN] Continuing without Photon 2 output.")
        return None


def get_eeg_window(board: BoardShim, window_sec: float) -> np.ndarray | None:
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    needed = int(window_sec * fs)
    data = board.get_current_board_data(needed)
    if data.shape[1] < needed:
        return None
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    window = data[eeg_channels, -needed:]
    return window


def ei_features_from_window(window: np.ndarray) -> np.ndarray:
    """Compute EI spectral features (including log power inside the block)."""
    global _feature_mismatch_warned

    ch, n = window.shape
    if ch != len(AXES):
        raise ValueError(f"Expected {len(AXES)} channels, got {ch}")

    # EI DSP expects interleaved samples per axis in row-major order
    raw_data = window.T.astype(np.float32).flatten()

    out = generate_features(
        IMPLEMENTATION_VERSION,
        DRAW_GRAPHS,
        raw_data,
        AXES,
        FS,
        SCALE_AXES,
        INPUT_DECIMATION_RATIO,
        FILTER_TYPE,
        FILTER_CUTOFF,
        FILTER_ORDER,
        ANALYSIS_TYPE,
        FFT_LENGTH,
        SPECTRAL_PEAKS_COUNT,
        SPECTRAL_PEAKS_THRESHOLD,
        SPECTRAL_POWER_EDGES,
        DO_LOG_IN_BLOCK,        # now True – same as in EI training
        DO_FFT_OVERLAP,
        WAVELET_LEVEL,
        WAVELET,
        EXTRA_LOW_FREQ,
    )

    feats = np.asarray(out["features"], dtype=np.float32)

    if feats.size != EXPECTED_FEATURES:
        if not _feature_mismatch_warned:
            print(
                f"[ERROR] Feature length mismatch: got {feats.size}, expected {EXPECTED_FEATURES}."
            )
            _feature_mismatch_warned = True
        return np.array([], dtype=np.float32)

    return feats


def run_inference(window: np.ndarray):
    feats = ei_features_from_window(window)
    if feats.size == 0:
        return None
    y = infer(feats)
    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), len(y)))}
    p_target = probs.get(LABELS[TARGET_CLASS_INDEX], float(max(y)))
    return p_target, probs


def send_decision(ser, is_target: bool):
    val = b"1\n" if is_target else b"0\n"
    if ser is not None and ser.is_open:
        try:
            ser.write(val)
        except SerialException as e:
            print(f"[WARN] Serial write failed: {e}")
    else:
        print(f"[DEBUG] Decision => {'1 (TARGET)' if is_target else '0 (NON-TARGET)'}")


# ================== EI FEATURE TEST MODE ==================


def parse_ei_feature_string(text: str) -> np.ndarray:
    """
    Parse a line copied from Edge Impulse "features" into a float32 vector.

    Accepts comma- or space-separated values and ignores other text on the line,
    so you can paste directly from EI tables.
    """
    # Find all numbers (handles integers and floats, with +/-)
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    if not nums:
        return np.array([], dtype=np.float32)
    arr = np.array([float(x) for x in nums], dtype=np.float32)
    return arr


def test_with_ei_features(feature_text: str):
    """
    Run a single inference from a pasted EI feature row and print probabilities.

    Usage from shell, for example:
      python EEG_ball_levitation.py --test-features "<paste EI features here>"

    Or:
      python EEG_ball_levitation.py --test-features "$(pbpaste)"
    """
    feats = parse_ei_feature_string(feature_text)

    print(f"[TEST] Parsed {feats.size} features from input.")
    print(f"[TEST] Model expects {EXPECTED_FEATURES} features.")

    if feats.size != EXPECTED_FEATURES:
        print("[TEST][ERROR] Feature length mismatch. "
              "Check that you're copying the correct raw feature row from EI.")
        return

    y = infer(feats)

    # Map to labels if possible
    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), len(y)))}
    # Also show raw vector for sanity
    print("[TEST] Raw model output vector:")
    for i, p in enumerate(y):
        label = LABELS[i] if i < len(LABELS) else f"class_{i}"
        print(f"  {i}: {label:9s} -> {p:.6f}")

    # Target probability (same logic as live code)
    p_target = probs.get(LABELS[TARGET_CLASS_INDEX], float(max(y)))
    print(f"[TEST] Target class index: {TARGET_CLASS_INDEX} ({LABELS[TARGET_CLASS_INDEX]})")
    print(f"[TEST] p_target = {p_target:.6f}")
    print(f"[TEST] Decision at threshold {TARGET_THRESHOLD:.3f}: "
          f"{'TARGET' if p_target >= TARGET_THRESHOLD else 'NON-TARGET'}")

    print(
        "\n[TEST] If this does not match Edge Impulse for the same feature row, "
        "there is a difference in DSP parameters or scaling (e.g., StandardScaler)."
    )


# ================== MAIN LOOP ==================


def live_loop():
    """Original live Muse→EI→decision loop."""
    board = None
    ser = None
    decisions: list[bool] = []
    prob_buffer: list[float] = []
    window_idx = 0

    try:
        board = init_board()
        ser = init_serial()
        print("[INFO] Entering main EEG → decision loop.")
        print(f"[DEBUG] Backend: {BACKEND}, Input shape: {INPUT_SHAPE}")

        while True:
            window = get_eeg_window(board, WINDOW_SECONDS)
            if window is None:
                print("[TRACE] Not enough data yet for full window.")
                time.sleep(STRIDE_SECONDS)
                window_idx += 1
                continue

            if window_idx < DEBUG_MAX_WINDOWS and (window_idx % DEBUG_EVERY_N == 0):
                ch, n = window.shape
                print(f"[DEBUG] Window {window_idx}: shape={window.shape}")
                for c in range(ch):
                    w = window[c]
                    print(
                        f"[DEBUG]  ch{c}: min={w.min():.3f}, max={w.max():.3f}, "
                        f"mean={w.mean():.3f}, std={w.std():.3f}"
                    )

            result = run_inference(window)
            if result is None:
                time.sleep(STRIDE_SECONDS)
                window_idx += 1
                continue

            p_target, probs = result
            prob_buffer.append(p_target)
            if len(prob_buffer) > SMOOTH_WINDOWS:
                prob_buffer = prob_buffer[-SMOOTH_WINDOWS:]

            # --- Smoothing ---
            if USE_MEDIAN_SMOOTH:
                p_smoothed = float(np.median(prob_buffer))
            else:
                p_smoothed = float(np.mean(prob_buffer))

            is_target = p_smoothed >= TARGET_THRESHOLD
            decisions.append(is_target)
            if len(decisions) > STABILITY_WINDOWS:
                decisions = decisions[-STABILITY_WINDOWS:]
            stable_target = all(decisions)

            calm_p = probs.get("calm", 0.0)
            non_calm_p = probs.get("non_calm", 0.0)
            sleep_p = probs.get("sleep", 0.0)

            print(
                f"p_target={p_smoothed:.3f} "
                f"(calm={calm_p:.2f}, non_calm={non_calm_p:.2f}, sleep={sleep_p:.2f}) | "
                f"last {len(decisions)}: {''.join('T' if d else 'N' for d in decisions)} | "
                f"stable_target={stable_target}"
            )

            send_decision(ser, stable_target)
            window_idx += 1
            time.sleep(STRIDE_SECONDS)

    except BrainFlowError as e:
        print(f"[ERROR] BrainFlow error: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C)...")
    finally:
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        if board:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass
        print("[INFO] Clean shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "EEG ball levitation: live Muse streaming OR test mode with "
            "Edge Impulse feature vectors."
        )
    )
    parser.add_argument(
        "--test-features",
        type=str,
        help=(
            "Run a single inference on a pasted EI feature row and exit. "
            "Paste the full 'features' line from Edge Impulse (commas/spaces OK)."
        ),
    )
    args = parser.parse_args()

    if args.test_features:
        # Pure offline test: do NOT touch BrainFlow or serial.
        test_with_ei_features(args.test_features)
    else:
        # Normal live mode
        live_loop()


if __name__ == "__main__":
    main()
