# EEG ball levitation: Muse → EI spectral features → Keras .h5 or TFLite model → Photon 2
# Thomas Vikström, 2025-11-11 23:00
# Automatically picks .h5 if available for debugging, .tflite otherwise. Adds log10 wrapper for exact EI parity.

import os
import sys
import time
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
    os.path.join(HERE, "processing-blocks"),          # if script is in src/
    os.path.join(HERE, "..", "processing-blocks"),    # if script moved into a subfolder
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
        "Make sure your tree looks like:\n"
        "  src/processing-blocks/spectral_analysis/...\n"
        "  src/processing-blocks/common/spectrum.py\n"
    )

if pb_root not in sys.path:
    sys.path.insert(0, pb_root)
    print(f"[INFO] Using Edge Impulse processing-blocks from: {pb_root}")

from spectral_analysis import generate_features


# ----------------- Model selection and setup -----------------
MODEL_TFLITE = "EEG_float32.lite"
MODEL_H5 = "EEG_model.h5"

if os.path.exists(MODEL_H5):
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_H5)
    INPUT_SHAPE = (1, model.input_shape[-1])
    EXPECTED_FEATURES = model.input_shape[-1]
    BACKEND = "keras"
    print(f"[INFO] Using Keras backend, model: {MODEL_H5}")

    def infer(features: np.ndarray):
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
FS = 256.0
WINDOW_SECONDS = 2.0
DECISION_INTERVAL = 0.2
STABILITY_WINDOWS = 3
TARGET_THRESHOLD = 0.7

# Edge Impulse Spectral Analysis params (from your Impulse)
IMPLEMENTATION_VERSION = 4
DRAW_GRAPHS = False
AXES = ["eeg_1", "eeg_2", "eeg_3", "eeg_4"]

SCALE_AXES = 1.0
INPUT_DECIMATION_RATIO = 1
FILTER_TYPE = "none"
FILTER_CUTOFF = 0
FILTER_ORDER = 0
ANALYSIS_TYPE = "FFT"
FFT_LENGTH = 256
SPECTRAL_PEAKS_COUNT = 0
SPECTRAL_PEAKS_THRESHOLD = 0
SPECTRAL_POWER_EDGES = "0"
DO_LOG = True
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


def init_board() -> BoardShim:
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    print("[INFO] Preparing Muse / BrainFlow session...")
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    print("[INFO] Starting Muse stream...")
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
    return window  # (4, N)


def ei_features_from_window(window: np.ndarray) -> np.ndarray:
    """Compute Edge Impulse spectral features with exact EI log10 scaling."""
    global _feature_mismatch_warned

    if window.ndim != 2:
        raise ValueError(f"Expected 2D EEG window, got {window.shape}")

    ch, n = window.shape
    if ch != len(AXES):
        raise ValueError(f"Expected {len(AXES)} channels, got {ch}")

    # Interleave time-major
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
        DO_LOG,
        DO_FFT_OVERLAP,
        WAVELET_LEVEL,
        WAVELET,
        EXTRA_LOW_FREQ,
    )

    feats = np.asarray(out["features"], dtype=np.float32)

    # --- Exact EI parity: EI uses log10 scaling internally ---
    if DO_LOG:
        feats = np.log10(np.maximum(feats, 1e-9))

    if feats.size != EXPECTED_FEATURES:
        if not _feature_mismatch_warned:
            print(
                f"[ERROR] Feature length mismatch: got {feats.size}, expected {EXPECTED_FEATURES}.\n"
                "Check EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE and spectral params."
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


def main():
    board = None
    ser = None
    decisions: list[bool] = []
    window_idx = 0

    try:
        board = init_board()
        ser = init_serial()

        print("[INFO] Entering main EEG → decision loop.")
        print(f"[DEBUG] Backend: {BACKEND}")
        print(f"[DEBUG] Model input shape: {INPUT_SHAPE}")

        while True:
            window = get_eeg_window(board, WINDOW_SECONDS)
            if window is None:
                print("[TRACE] Not enough data yet for full window.")
                time.sleep(DECISION_INTERVAL)
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

            # feats = ei_features_from_window(window)
            # print(
            #     f"Feature min/max: {feats.min():.6f} {feats.max():.6f} mean: {feats.mean():.6f}"
            # )

            result = run_inference(window)
            if result is None:
                time.sleep(DECISION_INTERVAL)
                window_idx += 1
                continue

            p_target, probs = result
            is_target = p_target >= TARGET_THRESHOLD

            decisions.append(is_target)
            if len(decisions) > STABILITY_WINDOWS:
                decisions = decisions[-STABILITY_WINDOWS:]
            stable_target = all(decisions)

            calm_p = probs.get("calm", 0.0)
            non_calm_p = probs.get("non_calm", 0.0)
            sleep_p = probs.get("sleep", 0.0)

            print(
                f"p_target={p_target:.3f} "
                f"(calm={calm_p:.2f}, non_calm={non_calm_p:.2f}, sleep={sleep_p:.2f}) | "
                f"last {len(decisions)}: {''.join('T' if d else 'N' for d in decisions)} | "
                f"stable_target={stable_target}"
            )

            send_decision(ser, stable_target)
            window_idx += 1
            time.sleep(DECISION_INTERVAL)

    except BrainFlowError as e:
        print(f"[ERROR] BrainFlow error: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C)...")
    finally:
        if ser is not None:
            try: ser.close()
            except Exception: pass
        if board is not None:
            try: board.stop_stream()
            except Exception: pass
            try: board.release_session()
            except Exception: pass
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    main()
