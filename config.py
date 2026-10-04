"""
NeuroStress Configuration Settings
"""

import os

# Application settings
APP_NAME = "NeuroStress"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "EEG-based Stress Detection using MSA-CBL Architecture"

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Dataset settings
DATASET_NAME = "SAM-40"
SAMPLING_RATE = 128  # Hz
NUM_CHANNELS = 32
SUBJECTS = 40
TRIALS_PER_SUBJECT = 3

# EEG Frequency bands (Hz)
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# Task labels
TASK_LABELS = {
    "Stroop": "stress",
    "Mirror_image": "stress",
    "Arithmetic": "stress",
    "Relax": "normal"
}

# Model architecture settings
MODEL_CONFIG = {
    "name": "MSA-CBL",
    "full_name": "Multi-Scale Attention CNN-BiLSTM",
    "input_shape": (32, 128, 1),  # channels x time x features
    "conv_filters": [32, 64, 128],
    "kernel_sizes": [3, 5, 7, 9],  # Multi-scale
    "lstm_units": 64,
    "attention_heads": 4,
    "dropout_rate": 0.3,
    "num_classes": 2
}

# The GAT uses the same channel order documented for SAM-40. Each four-second
# epoch is one graph in a temporal sequence (11 epochs for a 25-second trial).
GAT_CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "FC3", "FC4",
    "C3", "C4", "T3", "T4", "CP3", "CP4", "P3", "P4",
    "P7", "P8", "O1", "O2", "FC5", "FC6", "CP5", "CP6",
    "TP7", "TP8", "PO3", "PO4", "Fz", "Cz", "Pz", "Oz",
]

GAT_CONFIG = {
    "enabled": True,
    "name": "GAT + BiLSTM + Attention",
    "hidden_dim": 32,
    "heads": 4,
    "layers": 2,
    "dropout": 0.3,
    "epoch_length_seconds": 4.0,
    "epoch_overlap": 0.5,
    "sampling_rate": SAMPLING_RATE,
    "num_channels": NUM_CHANNELS,
    "num_features": 10,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "seed": 42,
    "model_path": os.path.join(BASE_DIR, "models", "gat_bilstm.pt"),
    "metrics_path": os.path.join(BASE_DIR, "models", "gat_metrics.json"),
}

# Training settings
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss": "binary_crossentropy",
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
}

# Stress detection thresholds
STRESS_THRESHOLDS = {
    "beta_alpha_ratio": {
        "stress": 2.0,
        "normal": 1.0
    },
    "frontal_asymmetry": {
        "stress": 0.2,
        "normal": 0.1
    },
    "confidence_threshold": 0.7
}

# Visualization settings
PLOT_COLORS = {
    "stress": "#ef4444",
    "normal": "#22c55e",
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "delta": "#3b82f6",
    "theta": "#06b6d4",
    "alpha": "#22c55e",
    "beta": "#f59e0b",
    "gamma": "#ef4444"
}

# Server settings
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8501,
    "headless": True
}
