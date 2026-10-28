# 02_train_rfdetr.py
"""Fine-tune RF-DETR-Medium on the pothole dataset."""
from rfdetr import RFDETRMedium
from config import DATASET_DIR, RFDETR_OUT, EPOCHS


def main():
    model = RFDETRMedium()
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=str(RFDETR_OUT),
    )


if __name__ == "__main__":
    main()
