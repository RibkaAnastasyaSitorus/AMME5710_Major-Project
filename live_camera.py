"""Run zone-aware PPE detection on a live camera stream."""

import time

import cv2
from activity_speed import ActivityRecognizer
from ppe_detector import DEFAULT_DEVICE, detect_frame, draw_detection_boxes
from worker_tracker import update_workers

WINDOW_NAME = "Live PPE Detection"
MAX_CAMERA_WIDTH = 1280


def process_live_camera(
    model,
    zone_module,
    confidence_threshold=0.25,
    camera_index=0,
    draw_detections=False,
):
    """Capture camera frames and display live PPE / zone assessments."""
    print(f"Inference device: {DEFAULT_DEVICE}")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Cannot open camera {camera_index}. "
            "Check CAMERA_INDEX and camera permissions."
        )

    window_created = False
    try:
        camera_id = zone_module.camera_id
        reference_width, reference_height = zone_module.config["cameras"][
            camera_id
        ]["reference_size"]
        requested_width = min(reference_width, MAX_CAMERA_WIDTH)
        requested_height = round(
            requested_width * reference_height / reference_width
        )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

        success, frame = camera.read()
        if not success or frame is None:
            raise RuntimeError(f"Camera {camera_index} did not provide a frame.")

        height, width = frame.shape[:2]
        print(f"Live camera resolution: {width}x{height}")
        aspect_ratio_difference = abs(
            (width / height) / (reference_width / reference_height) - 1
        )
        if aspect_ratio_difference > 0.01:
            raise RuntimeError(
                f"Camera {camera_index} returned {width}x{height}, but zones "
                f"for {camera_id!r} require a matching aspect ratio "
                f"({reference_width}x{reference_height}). Requested "
                f"{requested_width}x{requested_height}, but the camera did "
                "not provide it. Configure the camera for a matching "
                "resolution or create zones from a reference with its actual "
                "frame shape."
            )
        zone_module.prepare((width, height))
        worker_ppe_last_seen = {}
        activity_recognizer = ActivityRecognizer()
        started_at = time.monotonic()
        previous_video_time = -1.0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True

        while True:
            video_time = time.monotonic() - started_at
            if video_time <= previous_video_time:
                video_time = previous_video_time + 1e-6
            previous_video_time = video_time

            persons, ppe_detections = detect_frame(
                frame,
                model,
                confidence_threshold,
                device=DEFAULT_DEVICE,
            )
            workers = update_workers(
                persons,
                ppe_detections,
                worker_ppe_last_seen,
                video_time,
            )
            activity_recognizer.update(workers, video_time)
            workers = zone_module.process(
                workers,
                frame_size=(width, height),
                video_time=video_time,
                ppe_last_seen=worker_ppe_last_seen,
            )

            annotated = zone_module.annotate(frame, workers)
            if draw_detections:
                annotated = draw_detection_boxes(
                    annotated,
                    persons,
                    ppe_detections,
                )

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError(f"Lost camera feed from camera {camera_index}.")
            current_height, current_width = frame.shape[:2]
            if (current_width, current_height) != (width, height):
                raise RuntimeError(
                    "Camera frame dimensions changed. Restart the stream "
                    "to rebuild the camera-specific zones."
                )
    finally:
        camera.release()
        if window_created:
            cv2.destroyWindow(WINDOW_NAME)
