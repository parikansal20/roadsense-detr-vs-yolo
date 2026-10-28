# RoadSense: Transformer vs CNN Showdown for Real-Time Pothole Detection

Fine-tuned **RF-DETR** (DINOv2 transformer, ICLR 2026) and **YOLO26** (CNN, 2026)
on a pothole dataset and benchmarked them head-to-head on accuracy and speed.

![demo](assets/demo.gif)

## Results
<!-- Paste the contents of results/benchmark.md here after running notebooks/colab_run_all.ipynb -->

| Model | mAP@50 | mAP@50:95 | Latency (ms) | FPS |
|---|---|---|---|---|
| RF-DETR-M | | | | |
| YOLO26-M  | | | | |

**Takeaway:** <!-- which model wins on accuracy, which on speed, which you would deploy where -->

![comparison](results/compare_0.jpg)

## How it works
1. `01_download_dataset.py` – pulls the dataset (YOLO format) from Roboflow
2. `02_train_rfdetr.py` – fine-tunes RF-DETR-Medium
3. `03_train_yolo26.py` – fine-tunes YOLO26-M
4. `04_benchmark.py` – shared mAP (supervision) + latency evaluation
5. `app.py` – Streamlit demo for images and video

## Run it
Easiest: open `notebooks/colab_run_all.ipynb` in Google Colab (T4 GPU) and Run all.

Locally:
```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=... RF_WORKSPACE=... RF_PROJECT=... RF_VERSION=1
python 01_download_dataset.py && python 02_train_rfdetr.py && python 03_train_yolo26.py
python 04_benchmark.py
streamlit run app.py
```

## Tech stack
PyTorch · RF-DETR · Ultralytics YOLO26 · Supervision · Roboflow · Streamlit

## Credits
RF-DETR by Roboflow (Apache 2.0) · YOLO26 by Ultralytics (AGPL-3.0) · Dataset: [Pothole Detection by moksh on Roboflow Universe](https://universe.roboflow.com/moksh/pothole-detection-knlu9)
