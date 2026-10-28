# app.py
"""RoadSense demo: detect potholes with RF-DETR or YOLO26."""
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from PIL import Image
from config import RFDETR_OUT, YOLO_PROJECT, YOLO_RUN_NAME, CONF_THRESHOLD

st.set_page_config(page_title="RoadSense", layout="wide")
st.title("RoadSense: Transformer vs CNN Pothole Detection")


@st.cache_resource
def load_model(name):
    if name.startswith("RF-DETR"):
        from rfdetr import RFDETRMedium
        ckpts = list(RFDETR_OUT.rglob("*.pth"))
        best = [c for c in ckpts if "best" in c.name] or ckpts
        ckpt = sorted(best, key=lambda p: p.stat().st_mtime)[-1]
        model = RFDETRMedium(pretrain_weights=str(ckpt))

        def predict(bgr, conf):
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return model.predict(rgb, threshold=conf)
    else:
        from ultralytics import YOLO
        model = YOLO(YOLO_PROJECT / YOLO_RUN_NAME / "weights" / "best.pt")

        def predict(bgr, conf):
            return sv.Detections.from_ultralytics(model.predict(bgr, conf=conf, verbose=False)[0])
    return predict


def annotate(bgr, dets):
    names = dets.data.get("class_name", ["pothole"] * len(dets)) if dets.data else ["pothole"] * len(dets)
    labels = [f"{n} {c:.2f}" for n, c in zip(names, dets.confidence)]
    out = sv.BoxAnnotator(thickness=3).annotate(bgr.copy(), dets)
    return sv.LabelAnnotator().annotate(out, dets, labels)


with st.sidebar:
    model_name = st.radio("Model", ["RF-DETR-M (transformer)", "YOLO26-M (CNN)"])
    conf = st.slider("Confidence threshold", 0.1, 0.9, CONF_THRESHOLD, 0.05)
    st.caption("Benchmark results: see results/benchmark.md in the repo")

predict = load_model(model_name)
file = st.file_uploader("Upload a road image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

if file and file.type.startswith("image"):
    bgr = cv2.cvtColor(np.array(Image.open(file).convert("RGB")), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    dets = predict(bgr, conf)
    ms = (time.perf_counter() - t0) * 1000
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input")
    col2.image(cv2.cvtColor(annotate(bgr, dets), cv2.COLOR_BGR2RGB), caption=model_name)
    st.metric("Potholes found", len(dets))
    st.metric("Inference time", f"{ms:.0f} ms")
elif file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
    cap = cv2.VideoCapture(tmp.name)
    frame_box, stats = st.empty(), st.empty()
    n, total_ms = 0, 0.0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets = predict(frame, conf)
        total_ms += (time.perf_counter() - t0) * 1000
        n += 1
        frame_box.image(cv2.cvtColor(annotate(frame, dets), cv2.COLOR_BGR2RGB))
        stats.write(f"Frame {n} | potholes: {len(dets)} | avg FPS: {1000 * n / total_ms:.1f}")
    cap.release()
