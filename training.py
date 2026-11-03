"""Train 50-epoch detector models and compare them with 20-epoch baselines."""

import csv
from pathlib import Path

import torch
from ultralytics import YOLO

# =========================================================
# 1. PATHS
# =========================================================

PROJECT_DIR = Path(r"C:\UNI\UNI_SEM_7\AMME5710\MajorProject")
DATA_YAML = PROJECT_DIR / "Ultralytics_Dataset" / "construction-ppe.yaml"
OUTPUT_DIR = PROJECT_DIR / "runs" / "benchmark"
BASELINE_CSV = Path(__file__).parent / "benchmark" / "model_comparison.csv"
COMPARISON_CSV = OUTPUT_DIR / "model_comparison_20_vs_50.csv"

# =========================================================
# 2. EXPERIMENT SETTINGS
# =========================================================

MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO26n": "yolo26n.pt",
}

BASELINE_EPOCHS = 20
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"

COMPARISON_FIELDS = [
    "Model",
    "Epochs",
    "Precision",
    "Recall",
    "F1",
    "mAP50",
    "mAP50-95",
    "Preprocess_ms",
    "Inference_ms",
    "Postprocess_ms",
]


def load_baselines():
    """Load the original 20-epoch results without modifying their CSV."""
    if not BASELINE_CSV.is_file():
        raise FileNotFoundError(f"20-epoch results not found: {BASELINE_CSV}")

    with BASELINE_CSV.open(newline="", encoding="utf-8") as file:
        baselines = {
            row["Model"]: row
            for row in csv.DictReader(file)
            if row.get("Model") in MODELS
        }

    missing = MODELS.keys() - baselines.keys()
    if missing:
        raise ValueError(
            f"Missing 20-epoch baseline rows for: {', '.join(sorted(missing))}"
        )
    return baselines


def result_from_metrics(metrics, model_name, epochs):
    """Convert Ultralytics test metrics to a CSV-ready row."""
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "Model": model_name,
        "Epochs": epochs,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "Preprocess_ms": metrics.speed["preprocess"],
        "Inference_ms": metrics.speed["inference"],
        "Postprocess_ms": metrics.speed["postprocess"],
    }


def train_and_test(model_name, weights):
    """Train and test one model in its own 50-epoch run folder."""
    print("\n" + "=" * 60)
    print(f"MODEL: {model_name} ({EPOCHS} epochs)")
    print("=" * 60)

    model = YOLO(weights)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name=f"{model_name}_{EPOCHS}_epochs",
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
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        project=str(OUTPUT_DIR),
        name=f"{run_dir.name}_test",
        exist_ok=False,
    )

    return result_from_metrics(metrics, model_name, EPOCHS), best_model_path


def main():
    """Train all configured models and save one 20-vs-50 comparison CSV."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    baselines = load_baselines()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    print(f"Training device: {DEVICE}")

    for model_name, weights in MODELS.items():
        baseline = {
            field: baselines[model_name][field]
            for field in COMPARISON_FIELDS
            if field in baselines[model_name]
        }
        baseline.update(
            {"Model": model_name, "Epochs": BASELINE_EPOCHS}
        )
        comparison_rows.append(baseline)

        result, best_model_path = train_and_test(model_name, weights)
        comparison_rows.append(result)
        print(f"Best checkpoint: {best_model_path}")

        with COMPARISON_CSV.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(comparison_rows)

        print(
            f"20 epochs mAP50-95: {baseline['mAP50-95']} | "
            f"50 epochs mAP50-95: {result['mAP50-95']:.4f}"
        )

    print("\n" + "=" * 60)
    print("50-EPOCH COMPARISON COMPLETED")
    print("=" * 60)
    print(f"Comparison CSV: {COMPARISON_CSV}")
    print(f"Model checkpoints and test outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
