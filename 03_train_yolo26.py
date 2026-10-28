# 03_train_yolo26.py
"""Fine-tune YOLO26-M on the pothole dataset, then score it on the test split."""
from ultralytics import YOLO
from config import DATA_YAML, YOLO_PROJECT, YOLO_RUN_NAME, EPOCHS, IMG_SIZE, SEED


def main():
    model = YOLO("yolo26m.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        patience=15,
        seed=SEED,
        project=str(YOLO_PROJECT),
        name=YOLO_RUN_NAME,
        exist_ok=True,
    )
    best = YOLO(YOLO_PROJECT / YOLO_RUN_NAME / "weights" / "best.pt")
    metrics = best.val(data=str(DATA_YAML), split="test")
    print(f"YOLO26 test mAP@50: {metrics.box.map50:.3f} | mAP@50:95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
