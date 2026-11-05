"""Retrain and compare YOLO models at 20 and 50 epochs."""

import csv
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# =========================================================
# 1. PATHS
# =========================================================

PROJECT_DIR = Path(r"C:\UNI\UNI_SEM_7\AMME5710\MajorProject")
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_DIR / "Ultralytics_Dataset" / "data.yaml"
OUTPUT_DIR = PROJECT_DIR / "runs" / "benchmark"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = OUTPUT_DIR / f"rerun_20_vs_50_{RUN_ID}"
COMPARISON_CSV = EXPERIMENT_DIR / "model_comparison.csv"

# =========================================================
# 2. EXPERIMENT SETTINGS
# =========================================================

MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO26n": "yolo26n.pt",
}

EPOCH_COUNTS = (20, 50)
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"
METRICS_SPLIT = "val"

COMPARISON_FIELDS = [
    "Model",
    "Epochs",
    "Split",
    "Precision",
    "Recall",
    "F1",
    "mAP50",
    "mAP50-95",
    "Preprocess_ms",
    "Inference_ms",
    "Postprocess_ms",
]


def train_and_evaluate(model_name, weights, epochs):
    """Train one model from pretrained weights and evaluate its validation split."""
    run_name = f"{model_name}_{epochs}_epochs"
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name} ({epochs} epochs)")
    print("=" * 60)

    model = YOLO(weights)
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(EXPERIMENT_DIR),
        name=run_name,
        exist_ok=False,
        seed=42,
    )

    run_dir = Path(model.trainer.save_dir)
    best_model_path = run_dir / "weights" / "best.pt"
    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Could not find trained checkpoint: {best_model_path}"
        )

    metrics = YOLO(str(best_model_path)).val(
        data=str(DATA_YAML),
        split=METRICS_SPLIT,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        project=str(EXPERIMENT_DIR),
        name=f"{run_dir.name}_{METRICS_SPLIT}",
        exist_ok=False,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    result = {
        "Model": model_name,
        "Epochs": epochs,
        "Split": METRICS_SPLIT,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "Preprocess_ms": metrics.speed["preprocess"],
        "Inference_ms": metrics.speed["inference"],
        "Postprocess_ms": metrics.speed["postprocess"],
    }

    print(f"Best checkpoint: {best_model_path}")
    print(f"Validation mAP50-95: {result['mAP50-95']:.4f}")
    return result


def main():
    """Retrain all configured models at both epoch counts and save metrics."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    dataset = check_det_dataset(str(DATA_YAML))
    validation_images = dataset["val"]
    if not Path(validation_images).exists():
        raise FileNotFoundError(
            f"The validation split configured by {DATA_YAML} does not exist: "
            f"{validation_images}"
        )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    results = []
    print(f"Dataset: {DATA_YAML}")
    print(f"Training device: {DEVICE}")
    print(f"Validation split: {validation_images}")
    print(f"Outputs: {EXPERIMENT_DIR}")

    for model_name, weights in MODELS.items():
        for epochs in EPOCH_COUNTS:
            results.append(train_and_evaluate(model_name, weights, epochs))
            with COMPARISON_CSV.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
                writer.writeheader()
                writer.writerows(results)

    print("\n" + "=" * 60)
    print("BENCHMARK RERUN COMPLETED")
    print("=" * 60)
    print(f"Results CSV: {COMPARISON_CSV}")
    print(f"Model checkpoints and validation outputs: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
