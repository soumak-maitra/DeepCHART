# config.py

# General Configuration
DEVICE = "cuda"

# Data Configuration
TRAIN_PATHS = {
    "tau": "/user1/soumak/My_files/Tomography/Training/tau{}.npy",
    "dm": "/user1/soumak/My_files/Tomography/Training/DM{}.npy",
    "galaxy": "/user1/soumak/My_files/Tomography/Training/Galaxy_CIC{}.npy",
    "count": 108
}

TEST_PATHS = {
    "tau": "/user1/soumak/My_files/Tomography/Testing/tau{}.npy",
    "dm": "/user1/soumak/My_files/Tomography/Testing/DM{}.npy",
    "galaxy": "/user1/soumak/My_files/Tomography/Testing/Galaxy_CIC{}.npy",
    "count": 12
}

# Dataset Parameters
N = 128
N_SECTION = 96
Z_SECTION = 3 * N_SECTION
GAPS = 2.4 * N / 40
L_SMOOTH = 2.0 * N / 40

# SNR Distribution Parameters
SNR_MIN = 2.0
SNR_MAX = 10.0
ALPHA = 2.7

# Training Parameters
BATCH_SIZE = 18
LEARNING_RATE = 0.0002
NUM_EPOCHS = 500
START_EPOCH = 400

# Model Parameters
IN_CHANNELS = 2  # Ly-alpha forest + galaxy field
OUT_CHANNELS = 1
BASE_CHANNELS = 16
LATENT_DIM = 512

# Checkpoint and Output
CHECKPOINT_PATH = "/user1/soumak/My_files/Tomography/checkpoint.pth"
FINAL_MODEL_PATH = "/user1/soumak/My_files/Tomography/model_tomography_F_galaxy_to_DM.pth"
LOG_PATH = "/user1/soumak/My_files/Tomography/Output.txt"
HISTORY_PLOT = "/user1/soumak/My_files/Tomography/training_history.png"

# Dataloader
NUM_WORKERS = 4
PIN_MEMORY = True

# Plotting
SAVE_PLOT = True

