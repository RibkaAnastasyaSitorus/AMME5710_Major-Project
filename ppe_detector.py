ALL_PPE = {
    "helmet",
    "vest",
    "gloves",
    "boots",
    "goggles",
}

VIOLATION_CLASSES = {
    "no_helmet": "helmet",
    "no_goggle": "goggles",
    "no_gloves": "gloves",
    "no_boots": "boots",
    "none": "vest",
}


def normalise(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def detect_frame(frame, model, confidence_threshold):
    result = model.track(
        source=frame,
        conf=confidence_threshold,
        tracker="bytetrack.yaml",
        persist=True,
        device=0,
        verbose=False
    )[0]

    persons = []
    ppe_detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = normalise(result.names[class_id])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        track_id = None
        if box.id is not None:
            track_id = int(box.id[0])

        detection = {
            "class_name": class_name,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2),
            "track_id": track_id,
        }

        if class_name == "person":
            persons.append(detection)
        elif class_name in ALL_PPE or class_name in VIOLATION_CLASSES:
            ppe_detections.append(detection)

    return persons, ppe_detections
