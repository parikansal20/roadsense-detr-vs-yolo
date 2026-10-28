# 01_download_dataset.py
"""Download the pothole dataset from Roboflow in YOLO format and fix data.yaml paths."""
import os
import yaml
from roboflow import Roboflow
from config import DATASET_DIR, DATA_YAML

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = os.environ.get("RF_WORKSPACE", "moksh")
PROJECT = os.environ.get("RF_PROJECT", "pothole-detection-knlu9")
VERSION = int(os.environ.get("RF_VERSION", "1"))

rf = Roboflow(api_key=API_KEY)
rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
    "yolov8", location=str(DATASET_DIR), overwrite=True
)

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)
cfg["path"] = str(DATASET_DIR)
cfg["train"] = "train/images"
cfg["val"] = "valid/images"
cfg["test"] = "test/images"
with open(DATA_YAML, "w") as f:
    yaml.safe_dump(cfg, f)

for split in ["train", "valid", "test"]:
    img_dir = DATASET_DIR / split / "images"
    n = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    print(f"{split:>5}: {n} images")
print("classes:", cfg["names"])
if not (DATASET_DIR / "test" / "images").exists():
    print("WARNING: no test split. Create one in Roboflow (Generate -> Train/Test Split).")
