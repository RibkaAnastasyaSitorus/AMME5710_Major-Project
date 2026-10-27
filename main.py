from ultralytics import YOLO

from video_processor import process_video


MODEL_PATH = "runs/benchmark/YOLOv8n/weights/best.pt"
VIDEO_PATH = "PPEVideo2.mp4"
CONFIDENCE_THRESHOLD = 0.25
PPE_RETENTION_SECONDS = 0.5

ZONE_REQUIREMENTS = {
    "general": {
        "helmet",
        "vest",
    },
    "construction": {
        "helmet",
        "vest",
        "boots",
    },
    "chemical": {
        "helmet",
        "vest",
        "gloves",
        "goggles",
    },
    "electrical": {
        "helmet",
        "gloves",
        "boots",
    },
    "lab": {
        "goggles",
        "gloves",
    },
    "testing": {
        "helmet",
    },
}

CURRENT_ZONE = "testing"


def main():
    model = YOLO(MODEL_PATH)
    print("Model classes:", model.names)

    process_video(
        model=model,
        file_path=VIDEO_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        retention_seconds=PPE_RETENTION_SECONDS,
        current_zone=CURRENT_ZONE,
        required_ppe=ZONE_REQUIREMENTS[CURRENT_ZONE],
    )


if __name__ == "__main__":
    main()
