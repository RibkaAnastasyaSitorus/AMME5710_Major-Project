"""Shared zone geometry and rules; compatible with existing schema_version=1 JSON."""

import copy
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def empty_config():
    """Create the versioned config structure expected by the zone editor."""
    return {"schema_version": 1, "zone_rules": {}, "cameras": {}}


def cross(a, b, c):
    """Return the signed 2D cross product used by edge tests."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(point, a, b):
    """Check collinearity and inclusive bounds with a small numeric tolerance."""
    return (
        abs(cross(a, b, point)) < 1e-9
        and min(a[0], b[0]) - 1e-9 <= point[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= point[1] <= max(a[1], b[1]) + 1e-9
    )


def segments_intersect(a, b, c, d):
    """Detect proper crossings and endpoints that lie on another segment."""
    x = cross(a, b, c)
    y = cross(a, b, d)
    z = cross(c, d, a)
    w = cross(c, d, b)

    return (
        (x * y < 0 and z * w < 0)
        or on_segment(c, a, b)
        or on_segment(d, a, b)
        or on_segment(a, c, d)
        or on_segment(b, c, d)
    )


def validate_polygon(points, image_size):
    """Reject zone polygons that cannot be safely used as image regions."""
    width, height = image_size

    if len(points) < 3:
        raise ValueError("A zone needs at least three corners.")

    for point in points:
        if len(point) != 2:
            raise ValueError("Each corner needs x and y coordinates.")

        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            raise ValueError("Coordinates must be finite numbers.")

        if not (0 <= point[0] <= width - 1 and 0 <= point[1] <= height - 1):
            raise ValueError("A corner is outside the image.")

    if len({tuple(p) for p in points}) != len(points):
        raise ValueError(
            "Duplicate corners found. Do not click the first corner again."
        )

    n = len(points)

    # Shoelace area catches polygons whose vertices do not enclose a useful area.
    area = (
        abs(
            sum(
                points[i][0] * points[(i + 1) % n][1]
                - points[(i + 1) % n][0] * points[i][1]
                for i in range(n)
            )
        )
        / 2
    )

    if area < 1:
        raise ValueError("The polygon has no usable area.")

    for i in range(n):
        for j in range(i + 1, n):
            # Adjacent edges are allowed to meet.
            if j == i + 1 or (i == 0 and j == n - 1):
                continue

            if segments_intersect(
                points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]
            ):
                raise ValueError("Edges cross. Click corners in boundary order.")


def validate_config(config):
    """Validate schema version, shared rules, camera sizes, and polygons."""
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration version.")

    rules = config["zone_rules"]
    cameras = config["cameras"]

    if not isinstance(rules, dict) or not isinstance(cameras, dict):
        raise ValueError("Rules and cameras must be dictionaries.")

    for zone_id, rule in rules.items():
        if not zone_id or not isinstance(rule["name"], str):
            raise ValueError("Each zone needs an ID and name.")

        if not rule["name"].strip():
            raise ValueError("Zone names cannot be empty.")

        ppe = rule["required_ppe"]
        if not isinstance(ppe, list) or not all(
            isinstance(item, str) and item.strip() for item in ppe
        ):
            raise ValueError("PPE must be a list of class names.")

    for camera in cameras.values():
        size = camera["reference_size"]

        if len(size) != 2 or not all(isinstance(v, int) and v > 1 for v in size):
            raise ValueError("Invalid reference image dimensions.")

        for zone_id, points in camera["zones"].items():
            if zone_id not in rules:
                raise ValueError(f"Polygon refers to undefined zone: {zone_id}")

            validate_polygon(points, size)

    return config


def save_config(config, filename="zone_config.json"):
    """Validate then atomically replace the saved JSON configuration."""
    validate_config(config)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(temporary, path)

    return str(path.resolve())


def load_config(filename="zone_config.json"):
    """Read the JSON configuration and validate it before returning it."""
    config = json.loads(Path(filename).read_text(encoding="utf-8"))
    return validate_config(config)


# Coordinate convention: [width, height]. Polygons belong to one fixed viewpoint.
# Scaling is valid for resizing, never for camera motion, rotation or cropping.
def scaled_zones(config, camera_id, frame_size):
    """Scale polygons when the matching media has a different resolution."""
    camera = config["cameras"][camera_id]

    reference_width, reference_height = camera["reference_size"]
    frame_width, frame_height = frame_size

    if frame_width < 2 or frame_height < 2:
        raise ValueError("Invalid frame dimensions.")

    ratio_difference = abs(
        (frame_width / frame_height) / (reference_width / reference_height) - 1
    )

    if ratio_difference > 0.01:
        raise ValueError(
            "Video aspect ratio differs from the reference image. "
            "Use the matching camera view without cropping."
        )

    # Preserve the polygon shape by scaling x and y only after aspect validation.
    return {
        zone_id: [
            [
                x * (frame_width - 1) / (reference_width - 1),
                y * (frame_height - 1) / (reference_height - 1),
            ]
            for x, y in polygon
        ]
        for zone_id, polygon in camera["zones"].items()
    }


def point_in_polygon(point, polygon):
    """Return whether a point is inside a polygon; boundary points count as inside."""
    x, y = point
    inside = False

    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if on_segment(point, a, b):
            return True

        if (a[1] > y) != (b[1] > y):
            edge_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])

            if x < edge_x:
                inside = not inside

    return inside


# Baseline only: a box bottom is not evidence that feet are visible.
def locate_person(box, config, camera_id, frame_size):
    """
    box: [x1, y1, x2, y2]
    Returns all zones containing the estimated foot position.
    """
    # The bottom-centre is a floor-position estimate, not direct foot detection.
    x1, y1, x2, y2 = map(float, box)
    foot_point = ((x1 + x2) / 2, y2)

    polygons = scaled_zones(config, camera_id, frame_size)

    return [
        zone_id
        for zone_id, polygon in polygons.items()
        if point_in_polygon(foot_point, polygon)
    ]


def assess_safety(zone_ids, config, detected_ppe=None):
    """
    detected_ppe must belong to THIS person.

    None: PPE detection has not been evaluated.
    []: PPE evaluated, but no items detected.
    """
    if not zone_ids:
        return {"status": "UNMONITORED", "required": [], "not_detected": []}

    # Combine requirements if zones overlap.
    required = sorted(
        {
            item
            for zone_id in zone_ids
            for item in config["zone_rules"][zone_id]["required_ppe"]
        }
    )

    if not required:
        return {"status": "NO_PPE_REQUIRED", "required": [], "not_detected": []}

    if detected_ppe is None:
        return {"status": "PPE_NOT_EVALUATED", "required": required, "not_detected": []}

    not_detected = sorted(set(required) - set(detected_ppe))

    return {
        "status": ("REVIEW" if not_detected else "REQUIRED_PPE_DETECTED"),
        "required": required,
        "not_detected": not_detected,
    }


class ZoneSmoother:
    """Confirm a displayed zone change across consecutive frames."""

    def __init__(self, confirm_frames=5, max_gap=30):
        if confirm_frames < 1 or max_gap < 1:
            raise ValueError("Frame counts must be positive.")

        self.confirm_frames = confirm_frames
        self.max_gap = max_gap
        self.tracks = {}

    def update(self, track_id, zones, frame_index):
        """Return a zone set only after it stays stable across consecutive frames."""
        key = tuple(sorted(zones))
        state = self.tracks.get(track_id)

        # A long observation gap discards the old candidate for this person.
        if state is None or frame_index - state["last"] > self.max_gap:
            state = {
                "stable": None,
                "candidate": key,
                "count": 0,
                "last": frame_index - 1,
            }

        if state["candidate"] == key and frame_index - state["last"] == 1:
            state["count"] += 1
        else:
            state["candidate"] = key
            state["count"] = 1

        if state["count"] >= self.confirm_frames:
            state["stable"] = key

        state["last"] = frame_index
        self.tracks[track_id] = state

        # Drop stale tracks so old candidates do not accumulate indefinitely.
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if frame_index - value["last"] <= self.max_gap
        }

        return None if state["stable"] is None else list(state["stable"])
