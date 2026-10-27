from pathlib import Path

import cv2

from compliance import evaluate_compliance
from ppe_detector import detect_frame
from worker_tracker import update_workers


def process_video(
    model,
    file_path,
    confidence_threshold,
    retention_seconds,
    current_zone,
    required_ppe,
):
    if Path(file_path).suffix.lower() not in {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv"
    }:
        raise ValueError("This section expects a video file.")

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS.")

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
        cap.release()
        raise RuntimeError("Cannot create output video.")

    worker_ppe_last_seen = {}
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
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
            evaluate_compliance(
                workers,
                worker_ppe_last_seen,
                video_time,
                required_ppe,
                retention_seconds,
            )

            annotated = frame.copy()

            for track_id, worker in workers.items():
                x1, y1, x2, y2 = worker["bbox"]

                color = (
                    (0, 255, 0)
                    if worker["status"] == "SAFE"
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                cv2.putText(
                    annotated,
                    f"Worker #{track_id}",
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    annotated,
                    worker["status"],
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                if worker["missing"]:
                    missing_text = ", ".join(
                        sorted(worker["missing"])
                    )

                    cv2.putText(
                        annotated,
                        f"Missing: {missing_text}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            cv2.rectangle(
                annotated,
                (10, 10),
                (350, 50),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                annotated,
                f"ZONE: {current_zone.upper()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            writer.write(annotated)

            if frame_number % max(1, int(fps)) == 0:
                safe_count = sum(
                    worker["status"] == "SAFE"
                    for worker in workers.values()
                )

                unsafe_count = sum(
                    worker["status"] == "UNSAFE"
                    for worker in workers.values()
                )

                print(
                    f"{video_time:.1f}s | "
                    f"SAFE={safe_count} "
                    f"UNSAFE={unsafe_count}"
                )

    finally:
        cap.release()
        writer.release()

    print("Video processing complete.")
    print("Output:", output_path)
    return output_path
