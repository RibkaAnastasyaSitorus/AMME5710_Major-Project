# Canonical PPE labels used by the zone rules and the trained detector.
ALL_PPE = {
    "helmet",
    "vest",
    "gloves",
    "boots",
    "goggles",
}

# Map explicit missing-item labels back to the PPE item they contradict.
VIOLATION_CLASSES = {
    "no_helmet": "helmet",
    "no_goggle": "goggles",
    "no_gloves": "gloves",
    "no_boots": "boots",
    "none": "vest",
}


def normalise(name):
    """Standardise model labels so later class comparisons are consistent."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def detect_frame(frame, model, confidence_threshold, tracking=True):
    """Run detection and separate people from PPE-related detections.

    Video frames use persistent tracker IDs; still images use prediction only.
    """
    if tracking:
        # Persist track identities between video frames for PPE history.
        result = model.track(
            source=frame,
            conf=confidence_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            device=0,
            verbose=False
        )[0]
    else:
        # A still image has no temporal track to continue.
        result = model.predict(
            source=frame,
            conf=confidence_threshold,
            device=0,
            verbose=False
        )[0]

    # Return separate lists so the worker association step can join detections.
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
        elif not tracking and class_name == "person":
            # Temporary image-local IDs let the PPE matcher identify each person.
            track_id = len(persons)

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

    # Keep confidence and bbox data available to downstream processing.
    return persons, ppe_detections
