# EEG → blower control bridge using BrainFlow + Edge Impulse, portable between Windows and RPi5. 2025-11-07 20:45, Thomas Vikström

import time
import numpy as np
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ==== CONFIG ====
USE_SERIAL = True
SERIAL_PORT = "COM5"      # Windows; on RPi5 use "/dev/ttyACM0"
SERIAL_BAUD = 115200

WINDOW_SECONDS = 2.0
TARGET_THRESHOLD = 0.7
STABILITY_WINDOWS = 3

# Muse via BrainFlow
params = BrainFlowInputParams()
params.serial_port = ""   # not used for Muse BLE
# For Muse 2 via BLE:
BOARD_ID = BoardIds.MUSE_2_BOARD.value

def init_board():
    BoardShim.enable_dev_board_logger()
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream()
    return board

def get_eeg_window(board, window_sec):
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    n_samples = int(window_sec * fs)
    data = board.get_board_data()  # get all buffered data
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    if data.shape[1] < n_samples:
        return None
    return data[eeg_channels, -n_samples:]

def run_inference(features: np.ndarray) -> float:
    # TODO: replace with Edge Impulse / TFLite call
    # Placeholder: fake probability based on mean as stand-in
    x = float(np.mean(features))
    return max(0.0, min(1.0, (x - 0.2) / 0.6))

def main():
    board = init_board()
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1) if USE_SERIAL else None
    decisions = []

    try:
        while True:
            window = get_eeg_window(board, WINDOW_SECONDS)
            if window is None:
                time.sleep(0.1)
                continue

            # Simple features: mean per channel
            feats = window.mean(axis=1)
            p_target = run_inference(feats)
            is_target = p_target >= TARGET_THRESHOLD

            decisions.append(is_target)
            if len(decisions) > STABILITY_WINDOWS:
                decisions = decisions[-STABILITY_WINDOWS:]

            good = all(decisions)

            if ser:
                ser.write(b"1\n" if good else b"0\n")
            else:
                # later: direct PWM on Pi here
                pass

            time.sleep(0.2)

    finally:
        if ser:
            ser.close()
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()
