import math

from ppe_detector import ALL_PPE, VIOLATION_CLASSES


def update_workers(persons, ppe_detections, worker_ppe_last_seen, video_time):
    """Associate PPE boxes with people and update each tracked person's evidence."""
    workers = {}

    for index, person in enumerate(persons):
        track_id = person["track_id"]
        # Preserve people whose detector did not provide a persistent ID.
        worker_key = track_id if track_id is not None else ("untracked", index)
        workers[worker_key] = {
            "track_id": track_id,
            "bbox": person["bbox"],
            "ppe": set(),
            "status": "UNKNOWN",
            "missing": set(),
            "violations": set(),
            "ambiguous_ppe": set(),
        }

        if track_id is not None:
            worker_ppe_last_seen.setdefault(track_id, {})

    for ppe in ppe_detections:
        class_name = ppe["class_name"]
        if class_name not in ALL_PPE and class_name not in VIOLATION_CLASSES:
            continue

        px1, py1, px2, py2 = ppe["bbox"]
        # Use the PPE box centre as the point to test against person boxes.
        center_x = (px1 + px2) / 2
        center_y = (py1 + py2) / 2

        candidates = []
        for track_id, worker in workers.items():
            wx1, wy1, wx2, wy2 = worker["bbox"]

            if wx1 <= center_x <= wx2 and wy1 <= center_y <= wy2:
                width = max(wx2 - wx1, 1)
                height = max(wy2 - wy1, 1)
                # Normalize offsets so large person boxes do not dominate matches.
                relative_x = (center_x - (wx1 + wx2) / 2) / width
                relative_y = (center_y - (wy1 + wy2) / 2) / height
                distance = relative_x**2 + relative_y**2
                candidates.append((distance, track_id, worker))

        if not candidates:
            continue

        candidates.sort(key=lambda candidate: candidate[0])
        ppe_name = VIOLATION_CLASSES.get(class_name, class_name)

        # Equal best matches are ambiguous; never credit evidence to a guessed person.
        if (
            len(candidates) > 1
            and math.isclose(
                candidates[0][0],
                candidates[1][0],
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
        ):
            for distance, _, worker in candidates:
                if math.isclose(
                    distance,
                    candidates[0][0],
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                ):
                    worker["ambiguous_ppe"].add(ppe_name)
            continue

        _, track_id, worker = candidates[0]
        if class_name in VIOLATION_CLASSES:
            worker["violations"].add(ppe_name)
        else:
            worker["ppe"].add(class_name)
            if worker["track_id"] is not None:
                # Store the most recent observed time for this person's PPE item.
                worker_ppe_last_seen[track_id][class_name] = video_time

    # A missing-PPE detection overrides older positive evidence for that item.
    for track_id, worker in workers.items():
        history = worker_ppe_last_seen.get(track_id)
        if history is not None:
            for ppe_name in worker["violations"]:
                history.pop(ppe_name, None)

    # Return this frame's people with current detections and ownership metadata.
    return workers
