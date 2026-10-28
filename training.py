from ultralytics import YOLO
import csv
from pathlib import Path

# This script benchmarks detector checkpoints; it is separate from inference.
# =========================================================
# 1. PATHS
# =========================================================

PROJECT_DIR = Path(
    r"C:\UNI\UNI_SEM_7\AMME5710\MajorProject"
)

DATA_YAML = (
    PROJECT_DIR
    / "Ultralytics_Dataset"
    / "construction-ppe.yaml"
)

# Store benchmark runs outside the current working directory's default output.
OUTPUT_DIR = PROJECT_DIR / "runs" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_YAML.is_file(), f"Dataset YAML not found: {DATA_YAML}"

# =========================================================
# 2. EXPERIMENT SETTINGS
# =========================================================

MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO26n": "yolo26n.pt",
}

# Shared settings keep each model's training/evaluation comparable.
EPOCHS = 20
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0  # Use "cpu" instead when CUDA is unavailable.

all_results = []

# =========================================================
# 3. TRAIN AND TEST EACH MODEL
# =========================================================

for model_name, weights in MODELS.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    run_dir = OUTPUT_DIR / model_name
    best_model_path = run_dir / "weights" / "best.pt"

    # Reuse an existing checkpoint to avoid repeating a completed training run.
    if best_model_path.is_file():
        print(f"Existing trained model found:\n{best_model_path}")
        trained_model = YOLO(str(best_model_path))

    else:
        print(f"Training {model_name}...")

        model = YOLO(weights)

        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=str(OUTPUT_DIR),
            name=model_name,
            exist_ok=False,
            seed=42
        )

        # Ultralytics may choose a generated run directory for this training job.
        save_dir = Path(model.trainer.save_dir)
        best_model_path = save_dir / "weights" / "best.pt"

        if not best_model_path.is_file():
            raise FileNotFoundError(
                f"Could not find trained checkpoint: {best_model_path}"
            )

        print(f"Best checkpoint:\n{best_model_path}")
        trained_model = YOLO(str(best_model_path))

    # Measure generalization on the held-out test split.

    print(f"Testing {model_name}...")

    metrics = trained_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        project=str(OUTPUT_DIR),
        name=f"{model_name}_test",
        exist_ok=True
    )

    # Convert metric objects to plain numbers for reporting and CSV output.

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    result = {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "Preprocess_ms": metrics.speed["preprocess"],
        "Inference_ms": metrics.speed["inference"],
        "Postprocess_ms": metrics.speed["postprocess"],
    }

    all_results.append(result)

    print(f"\nResults for {model_name}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-score:       {f1:.4f}")
    print(f"mAP50:          {metrics.box.map50:.4f}")
    print(f"mAP50-95:       {metrics.box.map:.4f}")
    print(f"Inference time: {metrics.speed['inference']:.2f} ms/image")

    # Persist each model's results before starting the next benchmark.
    csv_path = OUTPUT_DIR / "model_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_results)

# =========================================================
# 4. FINISHED
# =========================================================

print("\n" + "=" * 60)
print("BENCHMARK COMPLETED")
print("=" * 60)
print(f"Results CSV: {OUTPUT_DIR / 'model_comparison.csv'}")
print(f"All model outputs: {OUTPUT_DIR}")
