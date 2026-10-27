from ppe_detector import ALL_PPE, VIOLATION_CLASSES


def update_workers(persons, ppe_detections, worker_ppe_last_seen, video_time):
    workers = {}

    for person in persons:
        track_id = person["track_id"]

        if track_id is None:
            continue

        workers[track_id] = {
            "bbox": person["bbox"],
            "ppe": set(),
            "status": "UNKNOWN",
            "missing": set(),
            "violations": set(),
        }

        worker_ppe_last_seen.setdefault(track_id, {})

    for ppe in ppe_detections:
        class_name = ppe["class_name"]
        if class_name not in ALL_PPE and class_name not in VIOLATION_CLASSES:
            continue

        px1, py1, px2, py2 = ppe["bbox"]
        center_x = (px1 + px2) / 2
        center_y = py2

        for track_id, worker in workers.items():
            wx1, wy1, wx2, wy2 = worker["bbox"]

            if wx1 <= center_x <= wx2 and wy1 <= center_y <= wy2:
                if class_name in VIOLATION_CLASSES:
                    worker["violations"].add(VIOLATION_CLASSES[class_name])
                else:
                    worker["ppe"].add(class_name)
                    worker_ppe_last_seen[track_id][class_name] = video_time

                break

    for track_id, worker in workers.items():
        for ppe_name in worker["violations"]:
            worker_ppe_last_seen[track_id].pop(ppe_name, None)

    return workers
