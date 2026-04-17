"""v2 config: hasta-bazli split + buyuk veri hiperparametreleri."""

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_raw"
LABELS_CSV = PROJECT_ROOT / "labels.csv"
TRAIN_CSV = PROJECT_ROOT / "train.csv"
VAL_CSV = PROJECT_ROOT / "val.csv"
TEST_CSV = PROJECT_ROOT / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Hemorrhage"]
IMG_SIZE = 224

# v2'de split zaten hasta-bazli yapilmis (scripts/split.py); bu oranlar
# sadece v1'den miras kalan modullerin import etmesi icin duruyor.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# 9k goruntude bs=32 + epoch=20 + patience=5 yeterli (v1'de 200 goruntu
# icin bs=16 epoch=30 idi; buyuk veri daha hizli yakinsar)
DEFAULT_HPARAMS = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 20,
    "early_stopping_patience": 5,
    "weight_decay": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
}

HPARAM_GRID = {
    "learning_rate": [5e-4, 1e-4, 5e-5],
    "batch_size": [16, 32],
    "weight_decay": [1e-3, 1e-4],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG-v2] Device: {DEVICE}")
print(f"[CONFIG-v2] Labels: {LABELS_CSV}")
