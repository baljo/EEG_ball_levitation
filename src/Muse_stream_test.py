# Check continuous Muse 2 EEG streaming with BrainFlow using get_current_board_data; 2025-11-07 22:52, Thomas Vikström

import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    # Once stable, you can lock it to your Muse MAC for robustness:
    # params.mac_address = "xx:xx:xx:xx:xx:xx"

    board_id = BoardIds.MUSE_2_BOARD.value
    board = BoardShim(board_id, params)

    try:
        print("[INFO] Preparing session...")
        board.prepare_session()

        # Optional: larger internal buffer (samples), but default is usually fine
        print("[INFO] Starting stream...")
        board.start_stream(45000)

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        fs = BoardShim.get_sampling_rate(board_id)
        print(f"[INFO] EEG channels: {eeg_channels}, sampling rate: {fs} Hz")

        # Let it fill a bit
        time.sleep(1.0)

        while True:
            # Get up to 1 second of the most recent data without clearing the buffer
            data = board.get_current_board_data(fs)  # shape: (channels, num_samples)
            num_samples = data.shape[1]

            if num_samples > 0:
                # Last EEG sample across EEG channels
                eeg_last = data[eeg_channels, -1]
                # Show how many samples we saw + last values
                print(f"Recent samples: {num_samples:4d} | Last EEG: {np.round(eeg_last, 2)}")
            else:
                print("No samples in the last second (check contact / battery / distance).")

            time.sleep(0.5)

    except BrainFlowError as e:
        print(f"[ERROR] BrainFlow error: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C)...")
    finally:
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
