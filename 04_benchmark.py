# 04_benchmark.py
"""Compare fine-tuned RF-DETR vs YOLO26 on the test split: mAP + latency/FPS."""
import time
import cv2
import numpy as np
import pandas as pd
import torch
import supervision as sv
from PIL import Image
from supervision.metrics import MeanAveragePrecision
from rfdetr import RFDETRMedium
from ultralytics import YOLO
from config import (DATASET_DIR, DATA_YAML, RFDETR_OUT, YOLO_PROJECT,
                    YOLO_RUN_NAME, RESULTS_DIR, CONF_THRESHOLD)

EVAL_CONF = 0.01
WARMUP = 10
N_SAMPLES = 6


def find_rfdetr_checkpoint():
    ckpts = list(RFDETR_OUT.rglob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth files in {RFDETR_OUT}. Run 02_train_rfdetr.py first.")
    best = [c for c in ckpts if "best" in c.name]
    return sorted(best or ckpts, key=lambda p: p.stat().st_mtime)[-1]


def align_classes(dets, classes):
    names = dets.data.get("class_name") if dets.data else None
    if names is None or len(dets) == 0:
        return dets
    idx = {n: i for i, n in enumerate(classes)}
    dets = dets[np.array([n in idx for n in names], dtype=bool)]
    dets.class_id = np.array([idx[n] for n in dets.data["class_name"]], dtype=int)
    return dets


def make_predictors():
    ckpt = find_rfdetr_checkpoint()
    print("RF-DETR checkpoint:", ckpt)
    rfdetr = RFDETRMedium(pretrain_weights=str(ckpt))
    if hasattr(rfdetr, "optimize_for_inference"):
        rfdetr.optimize_for_inference()
    yolo = YOLO(YOLO_PROJECT / YOLO_RUN_NAME / "weights" / "best.pt")

    def rfdetr_predict(bgr, conf):
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return rfdetr.predict(rgb, threshold=conf)

    def yolo_predict(bgr, conf):
        result = yolo.predict(bgr, conf=conf, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    return {"RF-DETR-M (transformer)": rfdetr_predict, "YOLO26-M (CNN)": yolo_predict}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=str(DATASET_DIR / "test" / "images"),
        annotations_directory_path=str(DATASET_DIR / "test" / "labels"),
        data_yaml_path=str(DATA_YAML),
    )
    classes = ds.classes
    samples = list(ds)
    print(f"Test images: {len(samples)} | classes: {classes}")

    rows, sample_preds = [], {}
    for name, predict in make_predictors().items():
        for _, img, _ in samples[:WARMUP]:
            predict(img, EVAL_CONF)
        preds, targets, times = [], [], []
        for _, img, gt in samples:
            sync(); t0 = time.perf_counter()
            dets = predict(img, EVAL_CONF)
            sync(); times.append((time.perf_counter() - t0) * 1000)
            preds.append(align_classes(dets, classes))
            targets.append(gt)
        m = MeanAveragePrecision().update(preds, targets).compute()
        latency = float(np.mean(times))
        rows.append({
            "model": name,
            "mAP@50": round(m.map50, 3),
            "mAP@50:95": round(m.map50_95, 3),
            "latency_ms": round(latency, 1),
            "FPS": round(1000 / latency, 1),
        })
        sample_preds[name] = preds[:N_SAMPLES]
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    (RESULTS_DIR / "benchmark.md").write_text(
        f"Hardware: {gpu} | test images: {len(samples)}\n\n" + df.to_markdown(index=False))
    print(f"\nHardware: {gpu}\n")
    print(df.to_markdown(index=False))

    box, label = sv.BoxAnnotator(), sv.LabelAnnotator()
    names = list(sample_preds)
    for i, (_, img, _) in enumerate(samples[:N_SAMPLES]):
        panels = []
        for name in names:
            d = sample_preds[name][i]
            d = d[d.confidence >= CONF_THRESHOLD]
            labels = [f"{classes[c]} {s:.2f}" for c, s in zip(d.class_id, d.confidence)]
            panel = label.annotate(box.annotate(img.copy(), d), d, labels)
            cv2.putText(panel, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            panels.append(panel)
        cv2.imwrite(str(RESULTS_DIR / f"compare_{i}.jpg"), np.hstack(panels))
    print(f"Saved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
