# AMME5710_Major-Project

## Zone-aware PPE inference

The inference path is `main.py` → `video_processor.py` → `ppe_detector.py` and
`worker_tracker.py` → `Zone/zone_integration.py`. The detector returns person
tracks and PPE detections. `worker_tracker` links PPE detections to tracked
people using their image boxes and keeps short-lived, per-track PPE evidence.
`ZoneModule` uses each person's lower-centre point to find the configured zone,
combines PPE requirements for overlapping zones, and evaluates that person's
evidence against those requirements. Annotated image or video output displays
the zone, person ID, and per-person result.

The active camera and zone rules come from `Zone/zone_config.json`; `main.py`
selects the camera with `CAMERA_ID`. The current defaults select camera
`Workshop site` and image `Testing.jpg`; that camera is marked at 672×493 with
a `truck` zone requiring helmet, gloves, and boots. Set `MODEL_PATH` and
`INPUT_PATH` in `main.py` to the model checkpoint and image or video to
process. Supported images include JPG, PNG, BMP, TIFF and WebP; supported
videos include MP4, AVI, MOV, MKV and WMV.
Annotated images are saved as `<input>_annotated.png`; videos are saved next
to the input with `_temporal` appended to the name. A still image is evaluated
independently, without tracking across frames; its person detections receive
temporary IDs only to associate that image's PPE with each person.

PPE detections do not carry a reliable person identity from the model, so
ownership is estimated spatially. When a detection fits multiple overlapping
person boxes equally well, it is marked ambiguous and cannot establish
compliance for either person. The lower-centre of the person box is also only
an estimate of the floor position; it is not direct foot detection.

Results distinguish evidence levels: `REQUIRED_PPE_DETECTED` means all PPE
required by the current zone was observed recently; `REVIEW` means required PPE
was not observed (this alone does not prove it is absent); `UNSAFE` indicates a
configured prohibited zone or a detected PPE-violation class; `UNCERTAIN`
indicates uncertain location or ambiguous required PPE; `UNMONITORED` means
the person is outside marked zones; and `NO_PPE_REQUIRED` means the current
zone has no PPE requirements. Zones must be drawn for the same camera view and
aspect ratio as the input. The pipeline rejects incompatible views rather than
applying unsafe polygon scaling.

### Code map

| File | Responsibility |
| --- | --- |
| `main.py` / `run_inference.py` | Configure the model, input image/video, camera, and start inference; `run_inference.py` is a compatibility launcher. |
| `video_processor.py` | Process still images or videos, run detection and person association, apply zone rules, and write annotated output. |
| `ppe_detector.py` | Run Ultralytics prediction/tracking and split detections into people, PPE, and violation classes. |
| `worker_tracker.py` | Associate PPE to tracked people and maintain per-person recent PPE evidence. |
| `compliance.py` | Earlier static-zone compliance helper; the current inference path uses `ZoneModule` instead. |
| `Zone/zone_core.py` | Validate zone JSON and polygons, scale polygons to video resolution, and provide legacy/simple geometry helpers. |
| `Zone/zone_localisation.py` | Estimate each person's floor point from their box and determine zone membership, boundary uncertainty, and PPE requirements. |
| `Zone/zone_integration.py` | Maintain per-person zone visits, evaluate required PPE, classify results, and draw the result overlays. |
| `Zone/zone_editor.py` / `Zone/zone_setup.py` | Create, edit, validate, and save camera references and zone polygons. |
| `training.py` | Train/compare object detectors; it is separate from the inference pipeline. |

People without a tracker ID are still shown, but remain `UNCERTAIN` because PPE
history cannot safely be linked to them across frames.