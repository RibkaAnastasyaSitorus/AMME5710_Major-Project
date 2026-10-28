from pathlib import Path

import cv2

from ppe_detector import detect_frame
from worker_tracker import update_workers

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".wmv"}


def process_image(
    model,
    file_path,
    confidence_threshold,
    zone_module,
):
    """Detect PPE in one image, apply camera zones, and save an annotated PNG."""
    if Path(file_path).suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("This section expects an image file.")

    frame = cv2.imread(str(file_path))
    if frame is None:
        raise RuntimeError(f"Cannot open image: {file_path}")

    height, width = frame.shape[:2]
    zone_module.prepare((width, height))
    persons, ppe_detections = detect_frame(
        frame,
        model,
        confidence_threshold,
        tracking=False,
    )
    # One still has no temporal PPE history, so evaluate only its current detections.
    workers = update_workers(persons, ppe_detections, {}, 0.0)
    workers = zone_module.process(
        workers,
        frame_size=(width, height),
        video_time=0.0,
        ppe_last_seen=None,
    )

    output_path = str(
        Path(file_path).with_name(Path(file_path).stem + "_annotated.png")
    )
    annotated = zone_module.annotate(frame, workers)
    if not cv2.imwrite(output_path, annotated):
        raise RuntimeError(f"Cannot write annotated image: {output_path}")

    print("Image processing complete.")
    print("Output:", output_path)
    return output_path


def process_video(
    model,
    file_path,
    confidence_threshold,
    zone_module,
):
    """Process each video frame while retaining PPE evidence by person track."""
    if Path(file_path).suffix.lower() not in {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv"
    }:
        raise ValueError("This section expects a video file.")

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {file_path}")

    writer = None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise RuntimeError("Invalid video FPS.")

        zone_module.prepare((width, height))

        output_path = str(
            Path(file_path).with_name(
                Path(file_path).stem + "_temporal.mp4"
            )
        )
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError("Cannot create output video.")

        # Histories persist between frames so brief detector misses do not erase PPE.
        worker_ppe_last_seen = {}
        frame_number = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            # Convert frame index into media time for the retention window.
            video_time = (frame_number - 1) / fps

            persons, ppe_detections = detect_frame(
                frame,
                model,
                confidence_threshold,
            )
            workers = update_workers(
                persons,
                ppe_detections,
                worker_ppe_last_seen,
                video_time,
            )
            # Apply each person's current zone rule to their recent PPE evidence.
            workers = zone_module.process(
                workers,
                frame_size=(width, height),
                video_time=video_time,
                ppe_last_seen=worker_ppe_last_seen,
            )

            annotated = zone_module.annotate(frame, workers)

            writer.write(annotated)

            if frame_number % max(1, int(fps)) == 0:
                status_counts = {}
                for worker in workers.values():
                    status = worker["zone_compliance_status"]
                    status_counts[status] = status_counts.get(status, 0) + 1

                print(
                    f"{video_time:.1f}s | "
                    + " ".join(
                        f"{status}={count}"
                        for status, count in sorted(status_counts.items())
                    )
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print("Video processing complete.")
    print("Output:", output_path)
    return output_path


def process_media(
    model,
    file_path,
    confidence_threshold,
    zone_module,
):
    """Dispatch an input file to the matching still-image or video pipeline."""
    suffix = Path(file_path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return process_image(
            model,
            file_path,
            confidence_threshold,
            zone_module,
        )
    if suffix in VIDEO_EXTENSIONS:
        return process_video(
            model,
            file_path,
            confidence_threshold,
            zone_module,
        )
    raise ValueError(f"Unsupported image or video file: {file_path}")
