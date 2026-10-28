def evaluate_compliance(
    workers,
    worker_ppe_last_seen,
    video_time,
    required_ppe,
    retention_seconds,
):
    """Compare each worker's recent PPE evidence with one shared PPE rule set."""
    for track_id, worker in workers.items():
        # Ignore PPE observations older than the configured temporal window.
        detected_ppe = {
            ppe_name
            for ppe_name, last_seen_time in worker_ppe_last_seen[track_id].items()
            if video_time - last_seen_time <= retention_seconds
        }

        # This legacy evaluator treats undetected requirements as non-compliance.
        missing = required_ppe - detected_ppe
        worker["missing"] = missing
        worker["status"] = "SAFE" if not missing else "UNSAFE"
