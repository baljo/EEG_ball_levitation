# EEG ball levitation v0.5.1: Muse→EI spectral→Keras/TFLite→blower PWM 0/128/255 via Photon 2 over serial/WiFi, with EI spectral/raw test modes; 2025-11-15 18:13, Thomas Vikström

import os
import sys
import time
import re
import argparse
from collections import Counter
import socket
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
MODEL_TFLITE = "EEG_float32_FFT8.lite" # "EEG_float32.lite"
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
FS = 256.0  # Hz
WINDOW_SECONDS = 2.0  # window size in seconds
STRIDE_SECONDS = 0.250  # stride between decisions

# These are still used in the test modes (threshold-based target/ non-target)
STABILITY_WINDOWS = 10  # (tests) number of last decisions to require stable target
SMOOTH_WINDOWS = 5  # (tests) number of last probabilities to average
USE_MEDIAN_SMOOTH = True  # (tests) True=median smoothing, False=mean smoothing
TARGET_THRESHOLD = 0.7

# For live blower mapping we use class history instead:
CLASS_HISTORY_WINDOWS = 8  # number of last predicted classes to majority-vote

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
FFT_LENGTH = 8
SPECTRAL_PEAKS_COUNT = 0
SPECTRAL_PEAKS_THRESHOLD = 0
SPECTRAL_POWER_EDGES = "0"
DO_LOG_IN_BLOCK = True  # let EI block take log of spectrum (matches training)
DO_FFT_OVERLAP = True
WAVELET_LEVEL = 1
WAVELET = ""
EXTRA_LOW_FREQ = False

LABELS = ["calm", "non_calm", "sleep"]

# Index of the "target" class for the test-modes (threshold logic)
TARGET_CLASS_INDEX = 1  # non_calm

# Class indices for blower mapping (must match LABELS)
CLASS_CALM = 0
CLASS_NON_CALM = 1
CLASS_SLEEP = 2

# Output defaults (can be overridden by CLI)
SERIAL_ENABLED = True
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200
DEFAULT_WIFI_PORT = 9000

DEBUG_MAX_WINDOWS = 40
DEBUG_EVERY_N = 10
_feature_mismatch_warned = False


# ================== OUTPUT LINK (SERIAL / WIFI) ==================


class BlowerLink:
    """
    Simple transport wrapper: either serial or WiFi TCP.
    Sends ASCII lines "<0-255>\\n" to Photon 2.
    """

    def __init__(
        self,
        serial_port: str | None = None,
        baud: int = 115200,
        wifi_host: str | None = None,
        wifi_port: int | None = None,
        timeout: float = 2.0,
    ):
        self.mode = "none"
        self.ser: serial.Serial | None = None
        self.sock: socket.socket | None = None
        self.sock_file = None

        if wifi_host is not None:
            # WiFi mode (TCP client)
            try:
                if wifi_port is None:
                    wifi_port = DEFAULT_WIFI_PORT
                self.sock = socket.create_connection((wifi_host, wifi_port), timeout=timeout)
                self.sock_file = self.sock.makefile("wb")
                self.mode = "wifi"
                print(f"[INFO] Using WiFi TCP → {wifi_host}:{wifi_port}")
            except OSError as e:
                print(f"[WARN] Could not open WiFi connection to {wifi_host}:{wifi_port}: {e}")
                print("[WARN] Continuing without Photon 2 output.")
        elif serial_port is not None:
            # Serial mode
            try:
                self.ser = serial.Serial(serial_port, baudrate=baud, timeout=1)
                self.mode = "serial"
                print(f"[INFO] Using serial → {serial_port} @ {baud} baud")
            except SerialException as e:
                print(f"[WARN] Could not open serial port {serial_port}: {e}")
                print("[WARN] Continuing without Photon 2 output.")

    def send_pwm(self, value: int):
        """Send one PWM command (0–255) followed by newline."""
        value = max(0, min(255, int(value)))
        line = f"{value}\n".encode("ascii")

        if self.mode == "wifi" and self.sock_file is not None:
            try:
                self.sock_file.write(line)
                self.sock_file.flush()
            except OSError as e:
                print(f"[WARN] WiFi write failed: {e}")
        elif self.mode == "serial" and self.ser is not None and self.ser.is_open:
            try:
                self.ser.write(line)
            except SerialException as e:
                print(f"[WARN] Serial write failed: {e}")
        else:
            print(f"[DEBUG] Blower PWM => {value} (no active link)")

    def close(self):
        if self.mode == "wifi":
            try:
                if self.sock_file is not None:
                    self.sock_file.close()
            except Exception:
                pass
            try:
                if self.sock is not None:
                    self.sock.close()
            except Exception:
                pass
        elif self.mode == "serial":
            try:
                if self.ser is not None:
                    self.ser.close()
            except Exception:
                pass


def init_output_link(args) -> BlowerLink | None:
    """
    Decide whether to use WiFi or serial based on CLI args.
    Returns BlowerLink or None if output is disabled.
    """
    if getattr(args, "no_output", False):
        print("[INFO] Blower output disabled (--no-output).")
        return None

    wifi_host = getattr(args, "wifi_host", None)
    wifi_port = getattr(args, "wifi_port", DEFAULT_WIFI_PORT)
    serial_port = getattr(args, "serial_port", SERIAL_PORT)
    serial_baud = getattr(args, "serial_baud", SERIAL_BAUD)

    if not SERIAL_ENABLED and wifi_host is None:
        print("[INFO] SERIAL_ENABLED = False and no WiFi host given; running without Photon 2.")
        return None

    link = BlowerLink(
        serial_port=None if wifi_host else serial_port,
        baud=serial_baud,
        wifi_host=wifi_host,
        wifi_port=wifi_port,
    )

    if link.mode == "none":
        return None
    return link


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
    """Compute EI spectral features (including log power inside the block) from a Muse window."""
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
        DO_LOG_IN_BLOCK,
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


def ei_features_from_raw_vector(raw: np.ndarray) -> np.ndarray:
    """
    Compute EI spectral features from a *raw* time-domain vector that has been
    copied from EI before the spectral analysis block.

    The raw array should be flattened in the same order EI uses:
    [eeg_1[0], eeg_2[0], eeg_3[0], eeg_4[0], eeg_1[1], eeg_2[1], ...].
    """
    global _feature_mismatch_warned

    if raw.ndim != 1:
        raw = raw.flatten()

    if raw.size % len(AXES) != 0:
        print(
            f"[TEST-RAW][WARN] Raw length {raw.size} is not divisible by #axes={len(AXES)}. "
            "Check that you're copying the correct raw feature row from EI."
        )

    out = generate_features(
        IMPLEMENTATION_VERSION,
        DRAW_GRAPHS,
        raw.astype(np.float32),
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
        DO_LOG_IN_BLOCK,
        DO_FFT_OVERLAP,
        WAVELET_LEVEL,
        WAVELET,
        EXTRA_LOW_FREQ,
    )

    feats = np.asarray(out["features"], dtype=np.float32)

    if feats.size != EXPECTED_FEATURES:
        if not _feature_mismatch_warned:
            print(
                f"[TEST-RAW][ERROR] Spectral feature length mismatch: got {feats.size}, "
                f"expected {EXPECTED_FEATURES}."
            )
            _feature_mismatch_warned = True
        return np.array([], dtype=np.float32)

    return feats


def run_inference(window: np.ndarray):
    """
    Live-mode inference: Muse window -> EI spectral features -> model output.

    Returns:
        y: np.ndarray, raw model output vector
        probs: dict[label -> float]
    """
    feats = ei_features_from_window(window)
    if feats.size == 0:
        return None
    y = infer(feats)
    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), len(y)))}
    return y, probs


def send_decision(ser, is_target: bool):
    """
    Legacy helper for binary target / non-target signalling (still used in tests).
    Sends '1' or '0'. Live loop now uses send_blower_pwm instead.
    """
    val = b"1\n" if is_target else b"0\n"
    if ser is not None and ser.is_open:
        try:
            ser.write(val)
        except SerialException as e:
            print(f"[WARN] Serial write failed: {e}")
    else:
        print(f"[DEBUG] Decision => {'1 (TARGET)' if is_target else '0 (NON-TARGET)'}")


def send_blower_pwm(link: BlowerLink | None, blower_pwm: int):
    """
    Send blower PWM 0–255 to Photon 2 via current link (serial or WiFi).
    """
    blower_pwm = int(max(0, min(255, blower_pwm)))
    if link is not None:
        link.send_pwm(blower_pwm)
    else:
        print(f"[DEBUG] Blower PWM command => {blower_pwm} (no link)")


def map_class_to_blower_pwm(pred_class: int) -> int:
    """
    Mapping requested (now in 0–255 PWM range as Photon 2 expects):
    - sleep     -> 0       (fan off)
    - calm      -> ~50 %   -> 128
    - non_calm  -> 100 %   -> 255
    """
    if pred_class == CLASS_SLEEP:
        return 0
    elif pred_class == CLASS_CALM:
        return 128
    elif pred_class == CLASS_NON_CALM:
        return 255
    else:
        return 0


# ================== EI FEATURE TEST MODES ==================


def parse_ei_feature_string(text: str) -> np.ndarray:
    """
    Parse a line copied from Edge Impulse into a float32 vector.

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
    Run a single inference from a pasted EI *spectral* feature row and print probabilities.

    This corresponds to the *input of the classifier* in EI (after the spectral
    analysis block and any scaling).
    """
    feats = parse_ei_feature_string(feature_text)

    print(f"[TEST] Parsed {feats.size} spectral features from input.")
    print(f"[TEST] Model expects {EXPECTED_FEATURES} features.")

    if feats.size != EXPECTED_FEATURES:
        print(
            "[TEST][ERROR] Feature length mismatch. "
            "Check that you're copying the correct spectral feature row from EI."
        )
        return

    y = infer(feats)

    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), len(y)))}
    print("[TEST] Raw model output vector:")
    for i, p in enumerate(y):
        label = LABELS[i] if i < len(LABELS) else f"class_{i}"
        print(f"  {i}: {label:9s} -> {p:.6f}")

    p_target = probs.get(LABELS[TARGET_CLASS_INDEX], float(max(y)))
    print(f"[TEST] Target class index: {TARGET_CLASS_INDEX} ({LABELS[TARGET_CLASS_INDEX]})")
    print(f"[TEST] p_target = {p_target:.6f}")
    print(
        f"[TEST] Decision at threshold {TARGET_THRESHOLD:.3f}: "
        f"{'TARGET' if p_target >= TARGET_THRESHOLD else 'NON-TARGET'}"
    )

    print(
        "\n[TEST] If this does not match Edge Impulse for the same spectral feature row, "
        "there is a difference in scaling (e.g., StandardScaler) or model export."
    )


def test_with_raw_samples(raw_text: str):
    """
    Run a single inference from *raw* time-domain samples (before spectral analysis).

    This is for data copied from EI at the stage *before* the spectral analysis
    block (i.e., raw features tab). The vector must be flattened in EI order:
    [eeg_1[0], eeg_2[0], eeg_3[0], eeg_4[0], eeg_1[1], eeg_2[1], ...].
    """
    raw = parse_ei_feature_string(raw_text)

    print(f"[TEST-RAW] Parsed {raw.size} raw values from input.")
    print(f"[TEST-RAW] Number of axes: {len(AXES)} ({AXES})")

    if raw.size == 0:
        print("[TEST-RAW][ERROR] No numeric values found in input.")
        return

    if raw.size % len(AXES) != 0:
        print(
            f"[TEST-RAW][WARN] Raw length {raw.size} is not divisible by #axes={len(AXES)} "
            f"(rem={raw.size % len(AXES)}). EI may be using a different window/axes config."
        )

    n_per_axis = raw.size // len(AXES)
    print(f"[TEST-RAW] Samples per axis (floor): {n_per_axis}")

    feats = ei_features_from_raw_vector(raw)
    if feats.size == 0:
        print("[TEST-RAW][ERROR] Could not derive spectral features from raw vector.")
        return

    print(f"[TEST-RAW] Generated {feats.size} spectral features from raw input.")
    print(f"[TEST-RAW] Model expects {EXPECTED_FEATURES} features.")

    if feats.size != EXPECTED_FEATURES:
        print(
            "[TEST-RAW][ERROR] Spectral feature length mismatch; DSP parameters likely "
            "differ between EI project and this script."
        )
        return

    y = infer(feats)

    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), len(y)))}
    print("[TEST-RAW] Raw model output vector:")
    for i, p in enumerate(y):
        label = LABELS[i] if i < len(LABELS) else f"class_{i}"
        print(f"  {i}: {label:9s} -> {p:.6f}")

    p_target = probs.get(LABELS[TARGET_CLASS_INDEX], float(max(y)))
    print(
        f"[TEST-RAW] Target class index: {TARGET_CLASS_INDEX} "
        f"({LABELS[TARGET_CLASS_INDEX]})"
    )
    print(f"[TEST-RAW] p_target = {p_target:.6f}")
    print(
        f"[TEST-RAW] Decision at threshold {TARGET_THRESHOLD:.3f}: "
        f"{'TARGET' if p_target >= TARGET_THRESHOLD else 'NON-TARGET'}"
    )

    print(
        "\n[TEST-RAW] If this does not match Edge Impulse for the same *raw* row, "
        "then the difference is in the spectral block configuration or sampling/windowing."
    )


# ================== MAIN LIVE LOOP ==================


def live_loop(args):
    """
    Live Muse → EI → blower mapping loop.

    Mapping (now in PWM 0–255):
      - sleep  (class 2) -> blower 0 (0 %)
      - calm   (class 0) -> blower 128 (~50 %)
      - non_calm (class 1) -> blower 255 (100 %)
    Uses majority vote over the last CLASS_HISTORY_WINDOWS predicted classes.
    """
    board = None
    link: BlowerLink | None = None
    class_history: list[int] = []
    window_idx = 0

    try:
        board = init_board()
        link = init_output_link(args)
        print("[INFO] Entering main EEG → blower loop.")
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

            y, probs = result

            # --- Class prediction and smoothing ---
            pred_class = int(np.argmax(y))
            class_history.append(pred_class)
            if len(class_history) > CLASS_HISTORY_WINDOWS:
                class_history = class_history[-CLASS_HISTORY_WINDOWS:]

            stable_class = Counter(class_history).most_common(1)[0][0]

            # --- Blower mapping (PWM) ---
            blower_pwm = map_class_to_blower_pwm(stable_class)
            blower_percent = int(round(blower_pwm / 255.0 * 100.0))

            calm_p = probs.get("calm", 0.0)
            non_calm_p = probs.get("non_calm", 0.0)
            sleep_p = probs.get("sleep", 0.0)

            history_str = "".join(str(c) for c in class_history)
            stable_label = LABELS[stable_class] if stable_class < len(LABELS) else "?"

            print(
                f"probs(calm={calm_p:.2f}, non_calm={non_calm_p:.2f}, sleep={sleep_p:.2f}) | "
                f"last {len(class_history)} classes: {history_str} | "
                f"stable_class={stable_class} ({stable_label}) | "
                f"blower={blower_pwm}/255 (~{blower_percent} %)"
            )

            # --- Send to Photon 2 ---
            send_blower_pwm(link, blower_pwm)

            window_idx += 1
            time.sleep(STRIDE_SECONDS)

    except BrainFlowError as e:
        print(f"[ERROR] BrainFlow error: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C)...")
    finally:
        if link:
            try:
                link.close()
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
            "EEG ball levitation: live Muse streaming OR test modes with "
            "Edge Impulse spectral or raw feature vectors."
        )
    )
    parser.add_argument(
        "--test-features",
        type=str,
        help=(
            "Run a single inference on a pasted EI *spectral* feature row "
            "(input to the classifier) and exit."
        ),
    )
    parser.add_argument(
        "--test-raw",
        type=str,
        help=(
            "Run a single inference on pasted *raw* time-domain samples "
            "copied from EI before the spectral analysis block (flattened)."
        ),
    )

    # ---- Output / transport options ----
    parser.add_argument(
        "--serial-port",
        default=SERIAL_PORT,
        help="Serial port for Photon 2 (ignored if --wifi-host is set).",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=SERIAL_BAUD,
        help="Serial baud rate for Photon 2 (default 115200).",
    )
    parser.add_argument(
        "--wifi-host",
        default=None,
        help="Photon 2 IP / hostname; if set, use WiFi TCP instead of serial.",
    )
    parser.add_argument(
        "--wifi-port",
        type=int,
        default=DEFAULT_WIFI_PORT,
        help=f"TCP port on Photon 2 (default {DEFAULT_WIFI_PORT}).",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable blower output entirely (no serial/WiFi).",
    )

    args = parser.parse_args()

    if args.test_raw:
        test_with_raw_samples(args.test_raw)
    elif args.test_features:
        test_with_ei_features(args.test_features)
    else:
        live_loop(args)


if __name__ == "__main__":
    main()
