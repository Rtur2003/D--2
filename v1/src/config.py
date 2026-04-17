# =========================================================================
# PROJECT CONFIGURATION
# =========================================================================
# BM 480 Derin Öğrenme - Proje 2
# Head CT Hemorrhage Classification
# =========================================================================

import os
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "head_ct" / "head_ct"
LABELS_CSV = PROJECT_ROOT / "labels.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
WEB_CRAWLED_DIR = PROJECT_ROOT / "web_crawled_test"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
WEB_CRAWLED_DIR.mkdir(exist_ok=True)

# ── Dataset Parameters ──────────────────────────────────────────────────

NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Hemorrhage"]
IMG_SIZE = 224  # Standard input size for pre-trained models

# ── Data Split Ratios ───────────────────────────────────────────────────
# Küçük veri seti (200 örnek) → Ders notuna göre:
# - Stratified split zorunlu (dengeli sınıflar)
# - 70/15/15 küçük-dengeli veri için uygun
# - Augmentation SADECE train setine uygulanmalı

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ── Training Hyperparameters (varsayılan değerler) ──────────────────────

DEFAULT_HPARAMS = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 30,
    "early_stopping_patience": 5,
    "weight_decay": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
}

# ── Hyperparameter Grid Search Space ────────────────────────────────────

HPARAM_GRID = {
    "learning_rate": [5e-4, 1e-4, 5e-5],
    "batch_size": [8, 16],
    "weight_decay": [1e-3, 1e-4],
}

# ── Device ──────────────────────────────────────────────────────────────

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Dataset path: {DATA_DIR}")
print(f"[CONFIG] Total images expected: 200")
