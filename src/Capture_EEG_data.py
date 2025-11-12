# Collect labeled Muse 2 EEG data to CSV for Edge Impulse training.
# 2025-11-08 22:20 (Europe/Helsinki), Thomas Vikström

import os
import time
import csv
from datetime import datetime

import numpy as np
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)

# ====== CONFIG ======
LABEL = "non_calm"              # change between runs: "calm", "non_calm", etc.
DURATION_SEC = 90           # how long to record this label
OUTPUT_DIR = "data"         # folder for CSV files

BOARD_ID = BoardIds.MUSE_2_BOARD.value
USE_MAC = False             # set True and fill MAC once stable
MUSE_MAC = "xx:xx:xx:xx:xx:xx"  # optional for reliability
# ====================


def init_board() -> BoardShim:
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    if USE_MAC:
        params.mac_address = MUSE_MAC

    board = BoardShim(BOARD_ID, params)
    print("[INFO] Preparing Muse session...")
    board.prepare_session()

    print("[INFO] Starting stream...")
    # Increase internal buffer; harmless
    board.start_stream(45000)

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    fs = BoardShim.get_sampling_rate(BOARD_ID)
    print(f"[INFO] EEG channels: {eeg_channels}, sampling rate: {fs} Hz")

    # Let buffer fill a bit
    time.sleep(1.0)

    return board


def record_label(board: BoardShim, label: str, duration_sec: int) -> np.ndarray:
    """
    Record EEG for given duration.
    Returns array shape: (num_samples, num_cols)
    Columns: [t, ch1, ch2, ch3, ch4]
    """
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    fs = BoardShim.get_sampling_rate(BOARD_ID)

    print(f"[INFO] Recording label='{label}' for {duration_sec} s...")
    t_start = time.time()

    all_rows = []

    while True:
        now = time.time()
        if now - t_start >= duration_sec:
            break

        # get all new data since last call (and clear from buffer)
        data = board.get_board_data()
        if data.size == 0:
            time.sleep(0.05)
            continue

        # data shape: (num_channels, num_samples)
        eeg_data = data[eeg_channels, :]
        if eeg_data.size == 0:
            continue

        num_samples = eeg_data.shape[1]
        # Generate timestamps relative to t_start for these samples
        # Approximate: assume uniform sampling at fs
        t_end_block = now
        t_start_block = t_end_block - (num_samples / fs)
        ts = np.linspace(
            t_start_block - t_start,
            t_end_block - t_start,
            num_samples,
            endpoint=False,
        )

        # Stack: t, ch1..ch4 (Muse 2: 4 EEG channels in eeg_channels order)
        # transpose to rows
        block = np.column_stack([ts, eeg_data.T])
        all_rows.append(block)

        # tiny sleep to avoid hammering
        time.sleep(0.02)

    if not all_rows:
        print("[WARN] No data captured for this recording.")
        return np.empty((0, 5))

    full = np.vstack(all_rows)

    # Sort by time just in case (should already be ordered)
    full = full[full[:, 0].argsort()]

    print(f"[INFO] Captured {full.shape[0]} samples for '{label}'.")
    return full


def save_csv(data: np.ndarray, label: str) -> str:
    """
    Save data to CSV:
    Columns: timestamp_s, eeg_1, eeg_2, eeg_3, eeg_4
    Filename encodes label and timestamp.
    """
    if data.size == 0:
        return ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{ts}.csv"
    path = os.path.join(OUTPUT_DIR, filename)

    header = ["timestamp_s", "eeg_1", "eeg_2", "eeg_3", "eeg_4"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow([f"{row[0]:.6f}"] + [f"{v:.6f}" for v in row[1:]])

    print(f"[INFO] Saved {path}")
    return path


def main():
    board = None
    try:
        board = init_board()
        data = record_label(board, LABEL, DURATION_SEC)
        save_csv(data, LABEL)

    except BrainFlowError as e:
        print(f"[ERROR] BrainFlow error: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if board is not None:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass
        print("[INFO] Session closed.")


if __name__ == "__main__":
    main()
