# EEG-controlled blower bridge: reads Muse 2 via BrainFlow, applies simple TARGET/NOT logic, sends 1/0 to Photon 2 over serial; 2025-11-07 22:46, Thomas Vikström

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

# ========= CONFIG =========

# --- Muse / BrainFlow ---
BOARD_ID = BoardIds.MUSE_2_BOARD.value
WINDOW_SECONDS = 2.0           # EEG window length for decisions
DECISION_INTERVAL = 0.2        # How often we decide (s)
STABILITY_WINDOWS = 3          # Require N consecutive TARGETs
TARGET_THRESHOLD = 0.7         # p_TARGET >= this -> TARGET

# --- Serial / Photon 2 ---
SERIAL_ENABLED = True          # Set False to run without Photon 2
SERIAL_PORT = "COM5"           # On RPi5: "/dev/ttyACM0"
SERIAL_BAUD = 115200

# ==========================


def init_board() -> BoardShim:
    """Initialize Muse 2 via BrainFlow."""
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    # Optional once stable: lock to your Muse MAC to avoid scanning issues:
    # params.mac_address = "xx:xx:xx:xx:xx:xx"

    board = BoardShim(BOARD_ID, params)
    print("[INFO] Preparing Muse / BrainFlow session...")
    board.prepare_session()
    print("[INFO] Starting Muse stream...")
    # Increase buffer a bit; harmless for our use
    board.start_stream(45000)

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    print(f"[INFO] EEG channels: {eeg_channels}, sampling rate: {fs} Hz")

    # Small delay to let buffer fill
    time.sleep(1.0)

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

    # Get up to needed samples without clearing buffer
    data = board.get_current_board_data(needed)
    if data.shape[1] < needed:
        return None

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    window = data[eeg_channels, -needed:]
    return window


def compute_features(window: np.ndarray) -> np.ndarray:
    """
    Very simple placeholder features from EEG window.
    For now: mean per channel (you'll replace with real DSP/EI input later).
    """
    # window: (channels, samples)
    # Example: mean over time for each channel
    feats = window.mean(axis=1)
    return feats


def run_inference(features: np.ndarray) -> float:
    """
    Placeholder for Edge Impulse / TFLite inference.

    Should return:
        probability of TARGET state in [0, 1].

    For now: crude mapping from mean(features) just to exercise pipeline.
    Replace this with:
        - EI Linux SDK call, or
        - TFLite model invocation.

    """
    x = float(np.mean(features))
    # Arbitrary scaling: adjust/remove once real model is in place
    p = (x - (-100.0)) / (200.0)  # map approx -100..+100 to 0..1
    p = max(0.0, min(1.0, p))
    return p


def send_decision(ser, is_target: bool):
    """
    Send '1' for TARGET, '0' for NON-TARGET to Photon 2 over serial if available.
    If no serial, just print for debugging.
    """
    val = b"1\n" if is_target else b"0\n"
    if ser is not None and ser.is_open:
        try:
            ser.write(val)
        except SerialException as e:
            print(f"[WARN] Serial write failed: {e}")
    else:
        # Debug output when Photon 2 not connected:
        print(f"[DEBUG] Decision => {'1 (TARGET)' if is_target else '0 (NON-TARGET)'}")


def main():
    board = None
    ser = None

    try:
        board = init_board()
        ser = init_serial()

        decisions: list[bool] = []
        fs = BoardShim.get_sampling_rate(BOARD_ID)

        print("[INFO] Entering main EEG → decision loop.")
        print("[INFO] Move/blink/relax and watch debug output; connect Photon when ready.\n")

        while True:
            window = get_eeg_window(board, WINDOW_SECONDS)

            if window is None:
                print("[TRACE] Not enough data yet for full window.")
                time.sleep(0.2)
                continue

            feats = compute_features(window)
            p_target = run_inference(feats)
            is_target = p_target >= TARGET_THRESHOLD

            decisions.append(is_target)
            if len(decisions) > STABILITY_WINDOWS:
                decisions = decisions[-STABILITY_WINDOWS:]

            stable_target = all(decisions)

            # Log a compact line so you see it's alive
            print(
                f"p_target={p_target:0.3f} | "
                f"last {len(decisions)}: {['T' if d else 'N' for d in decisions]} | "
                f"stable_target={stable_target}"
            )

            # Send to Photon (or just debug-print)
            send_decision(ser, stable_target)

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
