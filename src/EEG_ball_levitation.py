# EEG-controlled blower bridge: streams Muse EEG, applies EI-style spectral features, runs TFLite model, and sends stable TARGET/NOT decisions; 2025-11-11 20:40, Thomas Vikström

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

# Try tflite-runtime first (for Linux/RPi etc.), fall back to TensorFlow Lite
try:
    from tflite_runtime.interpreter import Interpreter  # usually unavailable on Windows/RPi
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow as tf
    if not hasattr(tf, "lite") or not hasattr(tf.lite, "Interpreter"):
        raise ImportError(
            "No tflite_runtime found and this TensorFlow build has no tf.lite.Interpreter.\n"
            "Install a compatible TensorFlow version, e.g.:\n"
            "  pip install \"tensorflow==2.15.0\"  or  \"tensorflow==2.16.1\""
        )
    Interpreter = tf.lite.Interpreter
    TFLITE_BACKEND = "tensorflow"

# Load your model (filename must match your exported file)
interpreter = Interpreter(model_path="EEG_float32.lite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ========= CONFIG =========

# --- Muse / BrainFlow ---
BOARD_ID = BoardIds.MUSE_2_BOARD.value
WINDOW_SECONDS = 2.0                    # EEG window length (s)
DECISION_INTERVAL = 0.2                 # How often we decide (s)
STABILITY_WINDOWS = 3                   # Require N consecutive TARGETs for stable=TRUE

# Model / decision config
LABELS = ["calm", "non_calm", "sleep"]  # assumed EI output order
TARGET_CLASS_INDEX = 1                  # treat "non_calm" as TARGET
TARGET_THRESHOLD = 0.7                  # p_TARGET >= this -> TARGET

# --- Serial / Photon 2 ---
SERIAL_ENABLED = True
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200

# Debug controls
DEBUG_FEATURES_EVERY_N = 10             # print feature stats every N windows
DEBUG_RAW_OUTPUT_EVERY_N = 10           # print raw model output vector every N windows
DEBUG_MAX_WINDOWS = 60                  # more verbose during first N windows

# ==========================


def init_board() -> BoardShim:
    """Initialize Muse 2 via BrainFlow."""
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    # If needed:
    # params.mac_address = "xx:xx:xx:xx:xx:xx"

    print("[INFO] Preparing Muse / BrainFlow session...")
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()

    print("[INFO] Starting Muse stream...")
    board.start_stream(45000)

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    print(f"[INFO] EEG channels: {eeg_channels}, sampling rate: {fs} Hz")

    time.sleep(1.0)  # Let buffer fill a bit
    return board


def init_serial():
    """Initialize serial; return serial object or None if unavailable."""
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
    """
    Grab the most recent EEG window.
    Returns array shape (n_channels, n_samples) or None if insufficient data.
    """
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    needed = int(window_sec * fs)

    data = board.get_current_board_data(needed)
    if data.shape[1] < needed:
        return None

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    window = data[eeg_channels, -needed:]

    return window


def ei_spectral_features(window: np.ndarray) -> np.ndarray:
    """
    Spectral features designed to match 1 x 532 input:

    - Muse: 4 EEG channels.
    - 2.0 s @ 256 Hz → 512 samples per channel.
    - For each of 4 channels:
        * Take last 512 samples.
        * Demean and normalize by std (if std > 0).
        * Hann window.
        * rFFT(512).
        * Keep bins where freq <= 66 Hz → 133 bins.
        * Use log10(power).
    - 4 * 133 = 532 features.
    """
    window = np.asarray(window)

    if window.ndim == 1:
        window = window.reshape(1, -1)
    elif window.ndim != 2:
        raise ValueError(f"Expected 1D or 2D EEG window, got shape {window.shape}")

    ch, n = window.shape

    if ch < 4:
        raise ValueError(f"Expected at least 4 EEG channels, got {ch}")
    if n < 512:
        raise ValueError(f"Expected at least 512 samples, got {n}")

    # Use first 4 channels, last 512 samples
    window = window[:4, -512:]
    ch, n = window.shape

    fs = 256.0
    n_fft = 512

    if n != n_fft:
        raise ValueError(f"Internal error: expected 512 samples, got {n}")

    hann = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    band_mask = freqs <= 66.0  # 133 bins from 0..66 Hz inclusive

    feats = []

    for c in range(ch):
        x = window[c, :]

        # Remove DC and normalize variance
        x = x - np.mean(x)
        std = np.std(x)
        if std > 1e-9:
            x = x / std

        x = x * hann
        spec = np.fft.rfft(x, n=n_fft)
        power = np.abs(spec) ** 2

        p_band = power[band_mask]
        p_log = np.log10(p_band + 1e-12)

        feats.extend(p_log.tolist())

    feats = np.array(feats, dtype=np.float32)

    expected_size = int(np.prod(input_details[0]["shape"]))
    if feats.size != expected_size:
        raise ValueError(
            f"Feature size {feats.size} does not match model input size {expected_size}."
        )

    return feats


def send_decision(ser, is_target: bool):
    """
    Send '1' for TARGET, '0' for NON-TARGET to Photon 2 over serial if available.
    If no serial, print for debugging.
    """
    val = b"1\n" if is_target else b"0\n"
    if ser is not None and ser.is_open:
        try:
            ser.write(val)
        except SerialException as e:
            print(f"[WARN] Serial write failed: {e}")
    else:
        print(f"[DEBUG] Decision => {'1 (TARGET)' if is_target else '0 (NON-TARGET)'}")


def debug_window_and_features(window: np.ndarray, features: np.ndarray, idx: int):
    """Print helpful debug info for early windows."""
    ch, n = window.shape
    print(f"[DEBUG] Window {idx}: shape={window.shape}")
    for c in range(min(ch, 4)):
        w = window[c, -512:]
        print(
            f"[DEBUG]  ch{c}: min={w.min():.3f}, max={w.max():.3f}, "
            f"mean={w.mean():.3f}, std={w.std():.3f}"
        )
    print(
        f"[DEBUG]  feats: shape={features.shape}, "
        f"min={features.min():.3f}, max={features.max():.3f}, "
        f"mean={features.mean():.3f}, std={features.std():.3f}"
    )


def main():
    board = None
    ser = None

    try:
        board = init_board()
        ser = init_serial()

        decisions: list[bool] = []

        print("[INFO] Entering main EEG → decision loop.")
        print("[INFO] Move/blink/relax etc. and watch probabilities.\n")
        print(f"[DEBUG] Model input shape: {input_details[0]['shape']}")
        print(f"[DEBUG] Model output shape: {output_details[0]['shape']}")

        window_idx = 0

        while True:
            window = get_eeg_window(board, WINDOW_SECONDS)

            if window is None:
                print("[TRACE] Not enough data yet for full window.")
                time.sleep(0.2)
                continue

            try:
                # 1) Features
                feats = ei_spectral_features(window)

                # Optional debug of raw signal + features
                if window_idx < DEBUG_MAX_WINDOWS and (window_idx % DEBUG_FEATURES_EVERY_N == 0):
                    debug_window_and_features(window, feats, window_idx)

                # 2) Inference
                x = feats.reshape(input_details[0]["shape"]).astype(np.float32)
                interpreter.set_tensor(input_details[0]["index"], x)
                interpreter.invoke()
                y = interpreter.get_tensor(output_details[0]["index"])[0]

                # Optional debug of raw model output
                if window_idx < DEBUG_MAX_WINDOWS and (window_idx % DEBUG_RAW_OUTPUT_EVERY_N == 0):
                    y_str = ", ".join(f"{v:.3f}" for v in y)
                    print(f"[DEBUG] raw model output: [{y_str}]")

                probs = {}
                for i, label in enumerate(LABELS):
                    if i < len(y):
                        probs[label] = float(y[i])

                if TARGET_CLASS_INDEX < len(y):
                    p_target = float(y[TARGET_CLASS_INDEX])
                else:
                    p_target = float(max(y))

            except Exception as e:
                print(f"[WARN] Inference error, skipping this window: {e}")
                time.sleep(DECISION_INTERVAL)
                window_idx += 1
                continue

            # 3) Decision logic
            is_target = p_target >= TARGET_THRESHOLD

            decisions.append(is_target)
            if len(decisions) > STABILITY_WINDOWS:
                decisions = decisions[-STABILITY_WINDOWS:]

            stable_target = all(decisions)

            calm_p = probs.get("calm", 0.0)
            non_calm_p = probs.get("non_calm", 0.0)
            sleep_p = probs.get("sleep", 0.0)

            print(
                f"p_target={p_target:0.3f} "
                f"(calm={calm_p:0.2f}, non_calm={non_calm_p:0.2f}, sleep={sleep_p:0.2f}) | "
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
            try:
                ser.close()
            except Exception:
                pass
        if board is not None:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    main()
