import copy
import json
import math
from pathlib import Path

if __package__:
    from .zone_core import load_config, scaled_zones
    from .zone_localisation import ZoneLocaliser
else:
    from zone_core import load_config, scaled_zones
    from zone_localisation import ZoneLocaliser


class ZoneModule:
    """Localise workers and assess zone PPE requirements for one media session."""

    def __init__(
        self,
        config,
        camera_id,
        retention_seconds=0.5,
        boundary_margin_px=6,
        track_timeout_seconds=3.0,
    ):
        if not math.isfinite(retention_seconds) or retention_seconds <= 0:
            raise ValueError(
                "retention_seconds must be finite and greater than zero."
            )
        if (
            not math.isfinite(track_timeout_seconds)
            or track_timeout_seconds <= 0
        ):
            raise ValueError(
                "track_timeout_seconds must be finite and greater than zero."
            )

        self.config = copy.deepcopy(config)
        self.camera_id = camera_id
        self.retention_seconds = float(retention_seconds)
        self.boundary_margin_px = boundary_margin_px
        self.track_timeout_seconds = track_timeout_seconds
        self.localiser = None
        # Track the current zone visit so PPE seen before entering cannot carry over.
        self.track_context = {}
        self.last_timestamp = None

    @classmethod
    def from_json(cls, config_path, camera_id, **kwargs):
        """Construct a zone module from a validated JSON configuration."""
        return cls(load_config(config_path), camera_id, **kwargs)

    def prepare(self, frame_size):
        """Validate the media dimensions and prepare the camera-specific polygons."""
        frame_size = tuple(frame_size)
        if self.localiser is None:
            self.localiser = ZoneLocaliser(
                self.config,
                self.camera_id,
                frame_size,
                self.boundary_margin_px,
            )
        elif frame_size != self.localiser.frame_size:
            raise ValueError("Frame dimensions changed. Create a new ZoneModule.")

    def process(self, workers, frame_size, video_time, ppe_last_seen=None):
        """Add location and PPE results to a copy of each worker.

        frame_size is (width, height); timestamps are in seconds.
        ppe_last_seen maps worker IDs to {PPE item: last detection time}.
        Without history, only current-frame PPE detections are used.
        """
        # A strictly increasing clock lets track history expire predictably.
        now = float(video_time)

        if not math.isfinite(now) or now < 0:
            raise ValueError("video_time must be finite and non-negative.")
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError(
                "Timestamps must increase; create a new ZoneModule for a new video."
            )
        if not isinstance(workers, dict):
            raise ValueError(
                "workers must be the dictionary returned by update_workers()."
            )

        frame_size = tuple(frame_size)
        self.prepare(frame_size)

        # Validate every box first, before mutating this module's temporal state.
        locations = {
            track_id: self.localiser.locate(
                worker["bbox"], worker.get("feet_visible")
            )
            for track_id, worker in workers.items()
        }
        self.last_timestamp = now

        # Expire tracks that have disappeared long enough to make IDs reusable.
        for track_id in list(self.track_context):
            last_seen = self.track_context[track_id]["last_seen"]
            if now - last_seen >= self.track_timeout_seconds:
                del self.track_context[track_id]

        output = copy.deepcopy(workers)

        for track_id, worker in output.items():
            location = locations[track_id]
            worker.update(location)
            worker["zone_camera_id"] = self.camera_id
            worker["zone_timestamp_s"] = now

            # Use None for uncertain location and an empty tuple for outside zones.
            zone_key = (
                None
                if location["zone_ids"] is None
                else tuple(location["zone_ids"])
            )
            context = self.track_context.get(track_id)

            # Start fresh when the worker changes zones or location is uncertain.
            if (
                context is None
                or context["zones"] != zone_key
                or zone_key is None
            ):
                context = {
                    "zones": zone_key,
                    "entered": now,
                    "last_seen": now,
                }
                self.track_context[track_id] = context

            context["last_seen"] = now
            worker["zone_retained_ppe"] = None
            worker["zone_not_detected"] = None
            worker["zone_violations"] = None
            worker["zone_ambiguous_ppe"] = None

            # Keep uncertain, unmonitored, access, and PPE cases distinct.
            if zone_key is None:
                status = "UNCERTAIN"
            elif not zone_key:
                status = "UNMONITORED"
            elif location["access_prohibited"]:
                status = "UNSAFE"
            elif not location["required_ppe"]:
                status = "NO_PPE_REQUIRED"
                worker["zone_retained_ppe"] = []
                worker["zone_not_detected"] = []
            elif worker.get("track_id", track_id) is None:
                status = "UNCERTAIN"
            else:
                if ppe_last_seen is not None:
                    history = ppe_last_seen.get(track_id)

                    # Retain detections from this zone visit until they expire.
                    evidence = (
                        None
                        if history is None
                        else {
                            item
                            for item, seen in history.items()
                            if math.isfinite(float(seen))
                            and context["entered"] <= float(seen) <= now
                            and now - float(seen) < self.retention_seconds
                        }
                    )
                else:
                    # Still images provide only current-frame PPE, with no history.
                    current = worker.get("ppe")
                    evidence = None if current is None else set(current)

                if evidence is None:
                    status = "UNCERTAIN"
                else:
                    required = set(location["required_ppe"])

                    # Missing-item or ambiguous labels cannot count as positive PPE.
                    violations = set(worker.get("violations", []))
                    ambiguous = set(worker.get("ambiguous_ppe", []))
                    conflicts = violations | ambiguous
                    evidence -= conflicts

                    worker["zone_violations"] = sorted(required & violations)
                    worker["zone_ambiguous_ppe"] = sorted(required & ambiguous)
                    missing = sorted(required - evidence - violations - ambiguous)
                    worker["zone_retained_ppe"] = sorted(evidence)
                    worker["zone_not_detected"] = missing

                    # Explicit missing-item evidence is stronger than a missed detection.
                    if worker["zone_violations"]:
                        status = "UNSAFE"
                    elif worker["zone_ambiguous_ppe"]:
                        status = "UNCERTAIN"
                    elif missing:
                        status = "REVIEW"
                    else:
                        status = "REQUIRED_PPE_DETECTED"

            worker["zone_compliance_status"] = status

        return output

    def annotate(self, frame, workers):
        """Draw zones, person locations, and per-person result summaries."""
        import cv2

        if self.localiser is None:
            raise ValueError("Call process() before annotate().")

        output = self.localiser.draw(frame)
        height, width = frame.shape[:2]

        for track_id, worker in workers.items():
            x1, y1, x2, y2 = map(int, worker["bbox"])
            display_id = worker.get("track_id", track_id)
            display_id = "untracked" if display_id is None else display_id

            # Use separate colors for confirmed compliance, unsafe, and review states.
            status = worker["zone_compliance_status"]
            if status in ("REQUIRED_PPE_DETECTED", "NO_PPE_REQUIRED"):
                colour = (0, 190, 0)
            elif status == "UNSAFE":
                colour = (0, 0, 255)
            else:
                colour = (0, 160, 255)

            cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(
                output,
                tuple(int(round(value)) for value in worker["floor_point"]),
                5,
                colour,
                -1,
            )

            # A missing zone list means uncertain; an empty list means outside zones.
            zone_label = (
                "Location uncertain"
                if worker["zone_ids"] is None
                else " / ".join(worker["zone_names"]) or "Outside marked zones"
            )

            lines = [
                f"ID {display_id}: {zone_label}",
                worker["zone_compliance_status"],
            ]

            activity = worker.get("activity")
            if activity is not None:
                lines.append(f"Activity: {activity}")

            if worker["zone_not_detected"]:
                lines.append(
                    "Not detected: " + ", ".join(worker["zone_not_detected"])
                )
            if worker["zone_violations"]:
                lines.append(
                    "Violation: " + ", ".join(worker["zone_violations"])
                )
            if worker["zone_ambiguous_ppe"]:
                lines.append(
                    "Ambiguous PPE: "
                    + ", ".join(worker["zone_ambiguous_ppe"])
                )

            base = max(15, min(height - 5 - len(lines) * 17, y1 - 8))

            for index, text in enumerate(lines):
                position = (max(0, min(width - 1, x1)), base + index * 17)
                cv2.putText(
                    output,
                    text,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1,
                )

        return output


def export_activity_zones(
    config,
    camera_id,
    frame_size,
    output_path,
    workstation_zone_ids=(),
    walkway_zone_ids=(),
):
    """Export selected scaled polygons for the separate activity program."""
    polygons = scaled_zones(config, camera_id, frame_size)

    chosen = set(workstation_zone_ids) | set(walkway_zone_ids)
    unknown = chosen - set(polygons)

    if unknown:
        raise ValueError(
            f"Zones not saved for this camera: {sorted(unknown)}"
        )

    converted = {
        "workstations": [
            polygons[zone_id] for zone_id in workstation_zone_ids
        ],
        "walkways": [
            polygons[zone_id] for zone_id in walkway_zone_ids
        ],
    }

    path = Path(output_path)

    if path.suffix.lower() != ".json":
        raise ValueError("Choose a separate .json output filename.")

    # Keep this export separate from the camera and PPE configuration.
    if path.exists():
        existing = json.loads(path.read_text())
        if "cameras" in existing or "zone_rules" in existing:
            raise ValueError(
                "Do not overwrite the main zone configuration "
                "with an activity export."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(converted, indent=2),
        encoding="utf-8",
    )

    return str(path.resolve())
