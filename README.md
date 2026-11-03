# Zone-Aware PPE Detection System

## 1. Executive Summary

This project processes still images and video to detect people and personal
protective equipment (PPE), estimate each person's location in a camera-specific
zone, and assess the detected PPE against that zone's requirements. It uses an
Ultralytics YOLO model for object detection, ByteTrack for video identities,
and polygon-based zone configuration.

The output is an annotated image or video showing each detected person's
estimated zone and PPE assessment. The result is intended to assist review;
it is not a substitute for direct safety supervision.

## 2. Project Objectives

The inference system is intended to:

1. Detect people, PPE items, and explicit missing-PPE labels.
2. Associate PPE detections with individual people.
3. Estimate each person's floor position and determine which configured zone
   contains it.
4. Apply the PPE rules associated with that zone.
5. Retain recent PPE detections briefly across video frames.
6. Save annotated output for inspection.

## 3. System Overview

### 3.1 Inference workflow

The image and video inference path is:

```text
main.py
  -> video_processor.py
      -> ppe_detector.py
      -> worker_tracker.py
      -> Zone/zone_integration.py
          -> Zone/zone_localisation.py
              -> Zone/zone_core.py
```

`main.py` selects the model, input media, confidence threshold, PPE retention
period, and camera configuration. `video_processor.py` dispatches the input to
the image or video processor. The detector separates person detections from
PPE detections; the worker association step links PPE to people. The zone
module then adds each person's location, zone requirements, and compliance
status before annotation is written.

### 3.2 Zone configuration

Zone polygons and their PPE rules are stored in
[`Zone/zone_config.json`](Zone/zone_config.json). Polygons are drawn for a
specific fixed camera view and saved in that reference image's pixel
coordinates. The program can scale polygons to a different resolution when the
input has the same aspect ratio. It rejects incompatible aspect ratios rather
than risk applying a misplaced zone.

The current inference defaults in `main.py` select camera `Workshop` and video
`PPEVideo4.mp4`. The saved `Workshop` reference is 1920×1080 and currently has:

| Zone | Required PPE |
|---|---|
| desk | vest |
| conveyor | helmet, vest |

Use the zone editor to create or update camera views and PPE rules. Use the
same view, without cropping or camera movement, when recording the video that
will be analyzed.

## 4. Processing Method

### 4.1 Detection and person association

For video, YOLO tracking assigns person IDs across frames. For a still image,
the program runs prediction without video tracking and assigns temporary IDs
within that image.

PPE detections do not inherently identify their wearer. The program therefore
uses the PPE box centre and person bounding boxes to estimate ownership. It
chooses the closest plausible person; when the best ownership matches are
equally plausible, that PPE is marked ambiguous rather than credited to a
person.

### 4.2 Zone localisation

The person's lower bounding-box centre is used as an estimate of their
floor/foot position. The point is tested against that camera's saved polygons.
If the estimate is near a zone boundary, the person touches the frame edge, or
the feet are reported hidden, the location is treated as uncertain.

When zones overlap, their required PPE lists are combined for the person.

### 4.3 PPE evidence and time retention

For video, the most recent time each PPE item was detected is recorded for
each tracked person. An observation can count only during the person's current
zone visit and while it is newer than the configured retention period.
`PPE_RETENTION_SECONDS` is currently `0.5` in `main.py`.

For a still image, only detections in that image are available; no evidence is
retained over time. An explicit missing-PPE detection (for example,
`no_helmet`) overrides positive evidence for that item.

## 5. Result Interpretation

These are program-generated assessment statuses, not YOLO object classes:

| Status | Interpretation |
|---|---|
| `REQUIRED_PPE_DETECTED` | All PPE required by the person's zone was detected recently. This is evidence from the detector, not a guarantee of safety. |
| `REVIEW` | One or more required items were not detected. A missed detection alone does not prove that the person is not wearing the item. |
| `UNSAFE` | The person is in a zone configured as prohibited, or an explicit missing-PPE label contradicts a required item. |
| `UNCERTAIN` | Location, tracking identity, PPE evidence, or PPE ownership is uncertain. |
| `UNMONITORED` | The estimated floor position is outside all marked zones. |
| `NO_PPE_REQUIRED` | The person is in a marked zone with no PPE requirements. |

The annotation uses green for `REQUIRED_PPE_DETECTED` and
`NO_PPE_REQUIRED`, red for `UNSAFE`, and orange for other statuses.

## 6. Running the Program

### 6.1 Select an inference source

Set `SOURCE` near the top of `main.py`, then run `python main.py`:

| `SOURCE` | Input |
|---:|---|
| `0` | Image at `INPUT_PATH` |
| `1` | Video at `INPUT_PATH` (default) |
| `2` | Live camera at `CAMERA_INDEX` |

Set `INPUT_PATH` to the image or video file you want to analyze. The default
zone camera is `Workshop` for image/video and `CameraFeed` for live camera.
Update `MEDIA_CAMERA_ID` or `LIVE_CAMERA_ID` near the top of `main.py` to
select another saved camera view. The model path, confidence threshold, PPE
retention, and raw detection overlay are configured there too.

Run from the repository directory on Windows:

```powershell
python main.py
```

Raw detection overlays are enabled by default for visual inspection. Person
boxes are yellow, detected PPE boxes are magenta, and explicit violation boxes
are red. Each label includes its class and confidence; person labels also show
the tracker ID when available. Set `DRAW_ALL_DETECTION_BOXES = False` in
`main.py` to hide these additional boxes.

Camera mode opens a live window; press `q` or `Esc` to stop. It requests a
resolution matching the selected saved camera view and reports an error if
the camera cannot provide a matching aspect ratio. Inference uses CUDA when
available and otherwise falls back to CPU; CPU processing may not keep up
with the camera's full frame rate.

### 6.3 Supported input and output

Supported image formats: JPG, JPEG, PNG, BMP, TIFF, and WebP. The annotated
image is saved beside its input as `<input>_annotated.png`.

Supported video formats: MP4, AVI, MOV, MKV, and WMV. The annotated video is
saved beside its input with `_temporal` appended to its filename.

## 7. Limitations and Assumptions

- PPE-to-person ownership is estimated from bounding-box geometry. Overlapping
  people or PPE boxes can make the assignment ambiguous.
- A person's floor position is estimated from the bottom-centre of the person
  bounding box; the detector does not directly locate their feet.
- A missed PPE detection results in `REVIEW`, not a confirmed violation.
- Zone polygons are valid only for their saved camera view. Resolution scaling
  does not compensate for camera movement, rotation, or cropping.
- Still-image assessments have no cross-frame tracking or temporal PPE
  retention.
- The result depends on the detector's classes, confidence threshold, image
  quality, and camera view.

## 8. Source Code Map

| File | Responsibility |
|---|---|
| [`main.py`](main.py) | Configures and starts image/video inference. |
| [`video_processor.py`](video_processor.py) | Dispatches image/video inputs, runs frame processing, and writes annotated output. |
| [`ppe_detector.py`](ppe_detector.py) | Runs YOLO prediction/tracking and separates people, PPE, and violation detections. |
| [`worker_tracker.py`](worker_tracker.py) | Associates PPE with people and maintains per-person recent PPE evidence. |
| [`Zone/zone_core.py`](Zone/zone_core.py) | Validates zone JSON and polygons, scales coordinates, and provides geometry helpers. |
| [`Zone/zone_localisation.py`](Zone/zone_localisation.py) | Estimates floor points and finds zone membership and required PPE. |
| [`Zone/zone_integration.py`](Zone/zone_integration.py) | Combines location and PPE evidence, assigns statuses, annotates frames, and exports activity polygons. |
| [`Zone/zone_editor.py`](Zone/zone_editor.py) | Provides the camera and zone editing interface and persistence workflow. |
| [`Zone/zone_setup.py`](Zone/zone_setup.py) | Loads local or notebook media into the zone editor. |
| [`compliance.py`](compliance.py) | Legacy static-zone compliance helper; the active inference path uses `ZoneModule`. |
| [`training.py`](training.py) | Trains and evaluates detector models; separate from inference. |
| [`tests/test_zone_pipeline.py`](tests/test_zone_pipeline.py) | Tests PPE association, image inference, zone assignment, and compliance statuses. |

## 9. Validation

The focused test suite can be run from the repository directory with:

```powershell
python -m unittest discover -s tests -v
```
