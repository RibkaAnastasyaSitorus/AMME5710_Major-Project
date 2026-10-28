"""Camera-specific polygon localisation using the existing zone JSON schema."""
import copy
import math
import cv2
import numpy as np

if __package__:
    from .zone_core import validate_config, scaled_zones, point_in_polygon
else:
    from zone_core import validate_config, scaled_zones, point_in_polygon


class ZoneLocaliser:
    """One fixed camera view; initialise again after changing zones or camera view."""

    def __init__(self, config, camera_id, frame_size, boundary_margin_px=6):
        # Validate geometry once, then cache polygons scaled to this frame size.
        validate_config(config)
        if camera_id not in config['cameras']:
            raise ValueError(f'Camera {camera_id!r} not found. Available: {list(config["cameras"])}')
        if not config['cameras'][camera_id]['zones']:
            raise ValueError('Save at least one polygon for the selected camera.')
        if not math.isfinite(boundary_margin_px) or boundary_margin_px < 0:
            raise ValueError('Boundary margin must be finite and non-negative.')
        self.config = copy.deepcopy(config)
        self.camera_id = camera_id
        self.frame_size = tuple(frame_size)
        self.margin = boundary_margin_px
        self.polygons = scaled_zones(self.config, camera_id, frame_size)

    def locate(self, bbox, feet_visible=None):
        """Estimate a person's zone from the bottom-centre of their bounding box."""
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Person bbox must be finite xyxy pixels with positive dimensions.')
        x1, y1, x2, y2 = box
        width, height = self.frame_size
        # The box bottom approximates where the person's feet meet the floor.
        point = (float((x1 + x2) / 2), float(y2))
        candidates = [z for z, poly in self.polygons.items() if point_in_polygon(point, poly)]
        reason = None
        # These cases make the floor-point estimate unreliable for rule assignment.
        if feet_visible is False:
            reason = 'Feet reported hidden'
        elif x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
            reason = 'Person touches the frame edge'
        elif any(abs(cv2.pointPolygonTest(np.asarray(poly, np.float32), point, True)) <= self.margin
                 for poly in self.polygons.values()):
            reason = 'Estimated floor point is near a zone boundary'
        # None distinguishes uncertain location from a confident point outside zones.
        zone_ids = None if reason else sorted(candidates)
        rules = self.config['zone_rules']
        # Overlapping zones combine their PPE requirements for this person.
        return {
            'floor_point': list(point), 'location_status': 'uncertain' if reason else 'estimated',
            'location_reason': reason, 'candidate_zone_ids': candidates, 'zone_ids': zone_ids,
            'zone_names': None if reason else [rules[z]['name'] for z in zone_ids],
            'required_ppe': None if reason else sorted({p for z in zone_ids for p in rules[z]['required_ppe']}),
            # Optional metadata; the marking UI itself currently edits name and PPE.
            'access_prohibited': None if reason else any(rules[z].get('access') == 'prohibited' for z in zone_ids),
            'in_work_zone': False if reason else any(rules[z].get('zone_type') == 'workstation' for z in zone_ids),
        }

    def draw(self, frame):
        """Draw the saved zone outlines and labels on a copy of the input frame."""
        if frame.shape[1::-1] != self.frame_size:
            raise ValueError('Frame resolution changed. Recreate the localiser.')
        output = frame.copy()
        for zone_id, polygon in self.polygons.items():
            contour = np.asarray(polygon, np.int32)
            cv2.polylines(output, [contour], True, (255, 180, 0), 2)
            cv2.putText(output, self.config['zone_rules'][zone_id]['name'], tuple(contour[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 180, 0), 2)
        return output
