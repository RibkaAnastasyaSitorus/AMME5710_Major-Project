from pathlib import Path

from ultralytics import YOLO

from Zone.zone_integration import ZoneModule
from live_camera import process_live_camera
from video_processor import process_image, process_video

# File and model locations.
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).parent / "benchmark" / "YOLO26n_50_epochs" / "weights" / "best.pt"
INPUT_PATH = PROJECT_DIR / "PPEVideo4.mp4"

# Zone configuration file saved from the zone editor.
ZONE_CONFIG_PATH = Path(__file__).parent / "Zone" / "zone_config.json"

# Detection and PPE retention settings.
CONFIDENCE_THRESHOLD = 0.25
PPE_RETENTION_SECONDS = 0.5

# Camera and zone selection.
MEDIA_CAMERA_ID = "Workshop"
LIVE_CAMERA_ID = "CameraFeed"
CAMERA_INDEX = 0

# Output settings.
DRAW_ALL_DETECTION_BOXES = False
SHOW_VIDEO_PREVIEW = True

# Run selection: 0 = image, 1 = video, 2 = live camera.
SOURCE = 2

def main():
    """Run PPE inference on an image, video, or live camera."""
    if SOURCE not in {0, 1, 2}:
        raise ValueError("SOURCE must be 0 (image), 1 (video), or 2 (camera).")

    source = {0: "image", 1: "video", 2: "camera"}[SOURCE]
    camera_id = LIVE_CAMERA_ID if source == "camera" else MEDIA_CAMERA_ID

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    print("Model classes:", model.names)
    # Build zone rules from the saved camera polygons and PPE requirements.
    zone_module = ZoneModule.from_json(
        ZONE_CONFIG_PATH,
        camera_id,
        retention_seconds=PPE_RETENTION_SECONDS,
    )

    if source == "camera":
        process_live_camera(
            model=model,
            zone_module=zone_module,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            camera_index=CAMERA_INDEX,
            draw_detections=DRAW_ALL_DETECTION_BOXES,
        )
    elif source == "image":
        process_image(
            model=model,
            file_path=INPUT_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            zone_module=zone_module,
            draw_detections=DRAW_ALL_DETECTION_BOXES,
        )
    else:
        process_video(
            model=model,
            file_path=INPUT_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            zone_module=zone_module,
            draw_detections=DRAW_ALL_DETECTION_BOXES,
            show_preview=SHOW_VIDEO_PREVIEW,
        )


if __name__ == "__main__":
    # Keep importing this module safe for tests and other Python callers.
    main()
