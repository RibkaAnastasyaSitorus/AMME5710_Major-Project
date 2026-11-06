
from pathlib import Path
import csv

from ultralytics import YOLO

# =========================================================
# 1. PATHS
# =========================================================

PROJECT_DIR = Path(
    r"C:\UNI\UNI_SEM_7\AMME5710\MajorProject"
)
REPO_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_DIR / "weights"

DATA_YAML = (
    REPO_DIR
    / "Ultralytics_Dataset"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "benchmark"
EXPERIMENT_DIR = OUTPUT_DIR / "code1_20_vs_50"

COMPARISON_CSV = EXPERIMENT_DIR / "model_comparison.csv"

assert DATA_YAML.is_file(), (
    f"Dataset YAML not found: {DATA_YAML}"
)

# =========================================================
# 2. EXPERIMENT SETTINGS
# =========================================================

MODELS = {
    "YOLOv8n": WEIGHTS_DIR / "yolov8n.pt",
    "YOLO11n": WEIGHTS_DIR / "yolo11n.pt",
    "YOLO26n": WEIGHTS_DIR / "yolo26n.pt",
}

EPOCH_COUNTS = (20, 50)

IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
SEED = 42

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
    "Total_ms",
]


# =========================================================
# 3. TRAIN AND EVALUATE ONE EXPERIMENT
# =========================================================

def train_and_evaluate(model_name, weights, epochs):

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name} ({epochs} epochs)")
    print("=" * 60)

    run_name = f"{model_name}_{epochs}_epochs"

    # -----------------------------------------------------
    # Train from pretrained weights
    # -----------------------------------------------------

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
        seed=SEED,
    )

    # -----------------------------------------------------
    # Locate best checkpoint
    # -----------------------------------------------------

    run_dir = Path(model.trainer.save_dir)
    best_model_path = run_dir / "weights" / "best.pt"

    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Could not find checkpoint: {best_model_path}"
        )

    print(f"\nBest checkpoint: {best_model_path}")

    trained_model = YOLO(str(best_model_path))

    # -----------------------------------------------------
    # Evaluate on test split
    # -----------------------------------------------------

    print(f"Evaluating {model_name} on {METRICS_SPLIT}...")

    metrics = trained_model.val(
        data=str(DATA_YAML),
        split=METRICS_SPLIT,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        project=str(EXPERIMENT_DIR),
        name=f"{run_name}_{METRICS_SPLIT}",
        exist_ok=False,
    )

    # -----------------------------------------------------
    # Extract metrics
    # -----------------------------------------------------

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    preprocess = float(metrics.speed["preprocess"])
    inference = float(metrics.speed["inference"])
    postprocess = float(metrics.speed["postprocess"])

    result = {
        "Model": model_name,
        "Epochs": epochs,
        "Split": METRICS_SPLIT,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "Preprocess_ms": preprocess,
        "Inference_ms": inference,
        "Postprocess_ms": postprocess,
        "Total_ms": preprocess + inference + postprocess,
    }

    print(f"\nResults for {model_name} ({epochs} epochs)")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-score:       {f1:.4f}")
    print(f"mAP50:          {result['mAP50']:.4f}")
    print(f"mAP50-95:       {result['mAP50-95']:.4f}")
    print(f"Total time:     {result['Total_ms']:.2f} ms/image")

    return result


# =========================================================
# 4. MAIN EXPERIMENT
# =========================================================

def main():

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)

    results = []

    print(f"Dataset: {DATA_YAML}")
    print(f"Training device: {DEVICE}")
    print(f"Epoch counts: {EPOCH_COUNTS}")
    print(f"Evaluation split: {METRICS_SPLIT}")
    print(f"Output directory: {EXPERIMENT_DIR}")

    for model_name, weights in MODELS.items():

        for epochs in EPOCH_COUNTS:

            result = train_and_evaluate(
                model_name,
                weights,
                epochs,
            )

            results.append(result)

            # Save results after each experiment
            with COMPARISON_CSV.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=COMPARISON_FIELDS,
                )

                writer.writeheader()
                writer.writerows(results)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    print(f"Results CSV: {COMPARISON_CSV}")
    print(f"All outputs: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
