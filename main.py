from pathlib import Path

from ultralytics import YOLO

from Zone.zone_integration import ZoneModule
from video_processor import process_media

# Resolve project assets relative to this repository's parent folder.
PROJECT_DIR = Path(__file__).resolve().parents[2]
# Select the trained checkpoint, image/video input, and matching camera setup.
MODEL_PATH = PROJECT_DIR / "runs" / "benchmark" / "YOLOv8n" / "weights" / "best.pt"
INPUT_PATH = PROJECT_DIR / "PPEVideo4.mp4"
CONFIDENCE_THRESHOLD = 0.25
# Keep a video PPE detection valid for this many seconds after it was seen.
PPE_RETENTION_SECONDS = 0.5
ZONE_CONFIG_PATH = Path(__file__).parent / "Zone" / "zone_config.json"
CAMERA_ID = "Workshop"


def main():
    """Load the detector and run zone-aware inference on the selected media."""
    model = YOLO(MODEL_PATH)
    print("Model classes:", model.names)
    # Build zone rules from the saved camera polygons and PPE requirements.
    zone_module = ZoneModule.from_json(
        ZONE_CONFIG_PATH,
        CAMERA_ID,
        retention_seconds=PPE_RETENTION_SECONDS,
    )

    process_media(
        model=model,
        file_path=INPUT_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        zone_module=zone_module,
    )


if __name__ == "__main__":
    # Keep importing this module safe for tests and other Python callers.
    main()
