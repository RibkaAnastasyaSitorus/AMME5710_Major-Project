from pathlib import Path
from ultralytics import YOLO
import torch

# Project configuration
PROJECT_DIR = Path(
    r"C:\UNI\UNI_SEM_7\AMME5710\MajorProject"
)
REPO_DIR = Path(__file__).resolve().parents[1]

# Change this to your selected experiment's best.pt
BEST_MODEL = (
    REPO_DIR
    / "benchmark"
    / "best.pt"
)

DATA_YAML = (
    REPO_DIR
    / "Ultralytics_Dataset"
    / "data.yaml"
)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

def main():
    if not BEST_MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {BEST_MODEL}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    model = YOLO(str(BEST_MODEL))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device=DEVICE,
        plots=True,
        project=str(PROJECT_DIR / "runs" / "final_test"),
        name="YOLOv8n_50_epochs",
        exist_ok=False,
    )

    print("\nFINAL TEST RESULTS")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()