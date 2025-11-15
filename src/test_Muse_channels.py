# Print Muse 2 EEG channel order using BrainFlow; Thomas Vikström; 2025-11-14 23:22

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Muse 2 is BoardIds.MUSE_2_BOARD
board_id = BoardIds.MUSE_2_BOARD.value
params = BrainFlowInputParams()

# Initialize board just to access its metadata (no streaming needed)
BoardShim.enable_dev_board_logger()

# EEG channels indices (these are internal BrainFlow indices)
eeg_channels = BoardShim.get_eeg_channels(board_id)
eeg_names = BoardShim.get_eeg_names(board_id)

print("EEG channel indices:", eeg_channels)
print("EEG channel labels :", eeg_names)



# Quick BrainFlow channel name check for Muse 2; Thomas Vikström, 2025-11-14 23:30

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

board_id = BoardIds.MUSE_2_BOARD.value
print("EEG channel indices:", BoardShim.get_eeg_channels(board_id))
print("EEG channel labels :", BoardShim.get_eeg_names(board_id))
