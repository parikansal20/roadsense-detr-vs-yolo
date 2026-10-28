# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Data
DATASET_DIR = ROOT / "datasets" / "potholes"   # filled by 01_download_dataset.py
DATA_YAML = DATASET_DIR / "data.yaml"

# Outputs
RFDETR_OUT = ROOT / "output" / "rfdetr"          # RF-DETR checkpoints
YOLO_PROJECT = ROOT / "runs"                     # YOLO26 runs folder
YOLO_RUN_NAME = "yolo26m_potholes"
RESULTS_DIR = ROOT / "results"

# Training
EPOCHS = 50
IMG_SIZE = 640
SEED = 42

# Inference
CONF_THRESHOLD = 0.4
