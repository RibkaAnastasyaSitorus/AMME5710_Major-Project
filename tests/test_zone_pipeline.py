"""Regression tests for person-to-PPE association and zone-aware inference."""

import unittest
from unittest.mock import Mock, patch

from Zone.zone_integration import ZoneModule, _draw_unsafe_worker_summary
from ppe_detector import detect_frame, draw_detection_boxes
from video_processor import process_image
from worker_tracker import update_workers


class WorkerAssociationTests(unittest.TestCase):
    """Check that PPE evidence is assigned to the correct detected person."""

    def test_ppe_is_assigned_to_the_closest_person_box(self):
        history = {}
        workers = update_workers(
            [
                {"track_id": 1, "bbox": (0, 0, 100, 200)},
                {"track_id": 2, "bbox": (100, 0, 200, 200)},
            ],
            [
                {"class_name": "helmet", "bbox": (25, 20, 45, 40)},
                {"class_name": "boots", "bbox": (155, 150, 175, 170)},
            ],
            history,
            1.0,
        )

        self.assertEqual(workers[1]["ppe"], {"helmet"})
        self.assertEqual(workers[2]["ppe"], {"boots"})
        self.assertEqual(history[1], {"helmet": 1.0})
        self.assertEqual(history[2], {"boots": 1.0})

    def test_equally_plausible_person_ownership_is_ambiguous(self):
        history = {}
        workers = update_workers(
            [
                {"track_id": 1, "bbox": (0, 0, 100, 200)},
                {"track_id": 2, "bbox": (0, 0, 100, 200)},
            ],
            [{"class_name": "helmet", "bbox": (45, 20, 55, 30)}],
            history,
            1.0,
        )

        for worker in workers.values():
            self.assertEqual(worker["ppe"], set())
            self.assertEqual(worker["ambiguous_ppe"], {"helmet"})
        self.assertEqual(history, {1: {}, 2: {}})

    def test_only_equally_best_candidates_are_marked_ambiguous(self):
        workers = update_workers(
            [
                {"track_id": 1, "bbox": (0, 0, 100, 100)},
                {"track_id": 2, "bbox": (0, 0, 100, 100)},
                {"track_id": 3, "bbox": (0, 0, 200, 200)},
            ],
            [{"class_name": "helmet", "bbox": (45, 45, 55, 55)}],
            {},
            1.0,
        )

        self.assertEqual(workers[1]["ambiguous_ppe"], {"helmet"})
        self.assertEqual(workers[2]["ambiguous_ppe"], {"helmet"})
        self.assertEqual(workers[3]["ambiguous_ppe"], set())

    def test_person_without_tracker_id_is_kept_as_untracked(self):
        workers = update_workers(
            [{"track_id": None, "bbox": (40, 20, 60, 80)}],
            [],
            {},
            0.0,
        )

        self.assertEqual(len(workers), 1)
        worker = next(iter(workers.values()))
        self.assertIsNone(worker["track_id"])


class ZoneComplianceTests(unittest.TestCase):
    """Exercise zone lookup and the statuses produced by PPE evidence."""

    def setUp(self):
        self.config = {
            "schema_version": 1,
            "zone_rules": {
                "work": {
                    "name": "Work area",
                    "required_ppe": ["helmet"],
                }
            },
            "cameras": {
                "camera": {
                    "reference_size": [100, 100],
                    "zones": {
                        "work": [
                            [10, 10],
                            [90, 10],
                            [90, 90],
                            [10, 90],
                        ]
                    },
                }
            },
        }

    def process_person(self, class_name=None):
        history = {}
        workers = update_workers(
            [{"track_id": 1, "bbox": (40, 20, 60, 80)}],
            (
                []
                if class_name is None
                else [{"class_name": class_name, "bbox": (45, 20, 55, 30)}]
            ),
            history,
            0.0,
        )
        zone_module = ZoneModule(
            self.config,
            "camera",
            retention_seconds=1.0,
            boundary_margin_px=2,
        )
        result = zone_module.process(
            workers,
            frame_size=(100, 100),
            video_time=0.0,
            ppe_last_seen=history,
        )
        return result[1]

    def test_required_ppe_detection_is_safe_for_that_zone(self):
        worker = self.process_person("helmet")

        self.assertEqual(worker["zone_ids"], ["work"])
        self.assertEqual(
            worker["zone_compliance_status"],
            "REQUIRED_PPE_DETECTED",
        )

    def test_prepare_rejects_an_incompatible_camera_aspect_ratio(self):
        zone_module = ZoneModule(self.config, "camera")

        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            zone_module.prepare((200, 100))

    def test_explicit_missing_ppe_detection_is_unsafe(self):
        worker = self.process_person("no_helmet")

        self.assertEqual(worker["zone_violations"], ["helmet"])
        self.assertEqual(worker["zone_compliance_status"], "UNSAFE")

    def test_missing_detection_requires_review_not_unsafe(self):
        worker = self.process_person()

        self.assertEqual(worker["zone_not_detected"], ["helmet"])
        self.assertEqual(worker["zone_compliance_status"], "REVIEW")

    def test_unsafe_summary_shows_worker_id_and_missing_ppe_only(self):
        frame = Mock()
        frame.shape = (480, 640, 3)
        frame.copy.return_value = frame
        workers = {
            7: {
                "track_id": 7,
                "zone_compliance_status": "UNSAFE",
                "zone_not_detected": ["vest"],
                "zone_violations": ["helmet"],
                "access_prohibited": False,
            },
            8: {
                "track_id": 8,
                "zone_compliance_status": "REVIEW",
                "zone_not_detected": ["gloves"],
                "zone_violations": [],
                "access_prohibited": False,
            },
        }

        with (
            patch("cv2.rectangle"),
            patch("cv2.addWeighted", return_value=frame),
            patch("cv2.putText") as put_text,
        ):
            result = _draw_unsafe_worker_summary(frame, workers)

        self.assertIs(result, frame)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("UNSAFE WORKERS (1)", labels)
        self.assertTrue(
            any("ID 7" in label for label in labels),
            labels,
        )
        self.assertTrue(
            any("helmet" in label for label in labels),
            labels,
        )
        self.assertTrue(
            any("vest" in label for label in labels),
            labels,
        )
        self.assertFalse(any("ID 8" in label for label in labels), labels)

    def test_untracked_person_cannot_be_reported_as_compliant(self):
        workers = update_workers(
            [{"track_id": None, "bbox": (40, 20, 60, 80)}],
            [{"class_name": "helmet", "bbox": (45, 20, 55, 30)}],
            {},
            0.0,
        )
        zone_module = ZoneModule(
            self.config,
            "camera",
            retention_seconds=1.0,
            boundary_margin_px=2,
        )

        result = zone_module.process(
            workers,
            frame_size=(100, 100),
            video_time=0.0,
            ppe_last_seen={},
        )
        worker = next(iter(result.values()))

        self.assertEqual(worker["zone_compliance_status"], "UNCERTAIN")


class ImageInferenceTests(unittest.TestCase):
    """Verify still images use prediction and the normal zone pipeline."""

    def test_detector_uses_prediction_and_assigns_still_image_person_ids(self):
        class FakeBox:
            def __init__(self, class_id, bbox):
                class Coordinates(list):
                    def tolist(self):
                        return list(self)

                self.cls = [class_id]
                self.conf = [0.9]
                self.xyxy = [Coordinates(bbox)]
                self.id = None

        class FakeResult:
            names = {0: "Person", 1: "helmet"}
            boxes = [
                FakeBox(0, [10, 10, 50, 90]),
                FakeBox(0, [60, 10, 100, 90]),
                FakeBox(1, [20, 15, 30, 25]),
            ]

        class FakeModel:
            def predict(self, **kwargs):
                self.kwargs = kwargs
                return [FakeResult()]

            def track(self, **kwargs):
                raise AssertionError("Still-image inference must not use tracking.")

        model = FakeModel()
        persons, ppe = detect_frame(
            object(),
            model,
            confidence_threshold=0.25,
            tracking=False,
        )

        self.assertEqual([person["track_id"] for person in persons], [0, 1])
        self.assertEqual(ppe[0]["class_name"], "helmet")
        self.assertEqual(model.kwargs["conf"], 0.25)

    def test_detection_overlay_draws_person_ppe_and_violation_boxes(self):
        frame = Mock()
        annotated = object()
        frame.copy.return_value = annotated
        detections = [
            {"class_name": "person", "confidence": 0.95, "bbox": (1, 2, 30, 40), "track_id": 7},
        ]
        ppe_detections = [
            {"class_name": "helmet", "confidence": 0.8, "bbox": (5, 6, 12, 14), "track_id": None},
            {"class_name": "no_helmet", "confidence": 0.7, "bbox": (15, 16, 22, 24), "track_id": None},
        ]

        with (
            patch("ppe_detector.cv2.rectangle") as rectangle,
            patch("ppe_detector.cv2.putText") as put_text,
        ):
            result = draw_detection_boxes(frame, detections, ppe_detections)

        self.assertIs(result, annotated)
        self.assertEqual(rectangle.call_count, 3)
        self.assertEqual(put_text.call_count, 3)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(
            labels,
            ["person 0.95 ID 7", "helmet 0.80", "no_helmet 0.70"],
        )

    def test_process_image_runs_zone_pipeline_and_writes_png(self):
        class FakeZoneModule:
            def __init__(self):
                self.prepared_size = None
                self.processed_workers = None

            def prepare(self, frame_size):
                self.prepared_size = frame_size

            def process(self, workers, frame_size, video_time, ppe_last_seen):
                self.processed_workers = workers
                return workers

            def annotate(self, frame, workers):
                return frame

        zone_module = FakeZoneModule()
        frame = type("Frame", (), {"shape": (40, 60, 3)})()
        with (
            patch("video_processor.cv2.imread", return_value=frame),
            patch("video_processor.cv2.imwrite", return_value=True) as imwrite,
            patch(
                "video_processor.detect_frame",
                return_value=(
                    [{"track_id": 0, "bbox": (1, 2, 20, 30)}],
                    [],
                ),
            ) as detect,
            patch(
                "video_processor.draw_detection_boxes",
                side_effect=lambda image, *_: image,
            ) as draw_boxes,
        ):
            output_path = process_image(
                object(),
                "sample.jpg",
                0.25,
                zone_module,
                draw_detections=True,
            )

        self.assertEqual(output_path, "sample_annotated.png")
        self.assertEqual(zone_module.prepared_size, (60, 40))
        detect.assert_called_once()
        draw_boxes.assert_called_once()
        imwrite.assert_called_once()
        self.assertEqual(len(zone_module.processed_workers), 1)


if __name__ == "__main__":
    unittest.main()
