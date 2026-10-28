"""Clear Add/Edit/Remove workflow for existing schema_version=1 zone JSON.
Reference images are embedded in JSON so reopening a camera restores its view.
"""

import base64
import copy
import hashlib
import html
import io
import json
import math
import uuid
from pathlib import Path


# Persistence layer: JSON keys stay compatible with your existing saved cameras.
# The editor changes copies, then validates and saves before updating its state.
class ZoneConfigStore:
    """Transactional changes: validate/write first, then update the working copy."""

    def __init__(self, filename, validate_configuration, validate_zone_polygon):
        self.path = Path(filename).resolve()
        self.validate_configuration = validate_configuration
        self.validate_zone_polygon = validate_zone_polygon
        self.reload()

    def file_fingerprint(self):
        return (
            hashlib.sha256(self.path.read_bytes()).hexdigest()
            if self.path.exists()
            else None
        )

    def reload(self):
        if self.path.exists():
            self.config = self.validate_configuration(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
        else:
            self.config = {"schema_version": 1, "zone_rules": {}, "cameras": {}}
        self.loaded_file_fingerprint = self.file_fingerprint()

    def save_config(self, config):
        self.validate_configuration(config)
        if self.file_fingerprint() != self.loaded_file_fingerprint:
            raise ValueError(
                "The saved file changed elsewhere. Cancel this edit and reload the camera before trying again."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        temp.replace(self.path)
        self.config = config
        self.loaded_file_fingerprint = self.file_fingerprint()

    def add_camera(self, camera_id, size, image_png, source):
        camera_id = camera_id.strip()
        if not camera_id:
            raise ValueError("Enter a camera name.")
        if camera_id in self.config["cameras"]:
            raise ValueError("That camera already exists. Select it from Saved camera.")
        updated = copy.deepcopy(self.config)
        updated["cameras"][camera_id] = {
            "reference_size": list(size),
            "reference_source": source,
            "reference_png": image_png,
            "zones": {},
        }
        self.save_config(updated)
        return camera_id

    def attach_reference(self, camera_id, size, image_png, source):
        updated = copy.deepcopy(self.config)
        camera_config = updated["cameras"][camera_id]
        if camera_config["reference_size"] != list(size):
            raise ValueError(
                "Reference dimensions do not match this camera. Use its original view/dimensions or create a new camera."
            )
        camera_config.update(reference_png=image_png, reference_source=source)
        self.save_config(updated)

    def save_zone(
        self,
        camera_id,
        zone_id,
        zone_name_input,
        required_ppe_input,
        polygon_points,
        update_shared=False,
    ):
        zone_name_input = zone_name_input.strip()
        if not zone_name_input:
            raise ValueError("Enter a zone name.")
        self.validate_zone_polygon(
            polygon_points, self.config["cameras"][camera_id]["reference_size"]
        )
        updated = copy.deepcopy(self.config)
        required_ppe_input = list(
            dict.fromkeys(p.strip() for p in required_ppe_input if p.strip())
        )
        rule = {"name": zone_name_input, "required_ppe": required_ppe_input}
        target = zone_id or "zone_" + uuid.uuid4().hex[:10]
        if zone_id:
            old = updated["zone_rules"][zone_id]
            others = [
                c
                for c, v in updated["cameras"].items()
                if c != camera_id and zone_id in v["zones"]
            ]
            # Default edits affect this camera. Clone the rule only if shared
            # metadata changed; geometry is always specific to this camera.
            if others and old != rule and not update_shared:
                target = "zone_" + uuid.uuid4().hex[:10]
                updated["cameras"][camera_id]["zones"].pop(zone_id, None)
        updated["zone_rules"][target] = rule
        updated["cameras"][camera_id]["zones"][target] = copy.deepcopy(polygon_points)
        self.save_config(updated)
        return target

    def remove_zone(self, camera_id, zone_id):
        updated = copy.deepcopy(self.config)
        updated["cameras"][camera_id]["zones"].pop(zone_id)
        # Retain shared rules for reuse; never delete another camera's polygon.
        self.save_config(updated)

    def remove_camera(self, camera_id):
        """Remove one saved view and its polygons; retain shared zone rules."""
        if camera_id not in self.config["cameras"]:
            raise ValueError("This camera no longer exists. Reload saved cameras.")
        updated = copy.deepcopy(self.config)
        del updated["cameras"][camera_id]
        # Original images/videos are external files: never delete them here.
        # Rules may also belong to surviving cameras, so retain the dictionary.
        self.save_config(updated)


# Interaction layer: draft corners are separate from saved camera polygons.
# Coordinates use the original reference image pixels (x rightwards, y downwards).
# editor_mode: view / add / edit / remove / remove_camera.
class ZoneEditorBase:
    """Instantiate after %matplotlib widget and the zone-core definition cell."""

    def __init__(self, config_path="zone_config.json"):
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display

        # The standalone .py can be used with an imported zone_core module.
        if "validate_config" in globals():
            validate_configuration, validate_zone_polygon = (
                validate_config,
                validate_polygon,
            )
        else:
            from zone_core import (
                validate_config as validate_configuration,
                validate_polygon as validate_zone_polygon,
            )
        self.config_store = ZoneConfigStore(
            config_path, validate_configuration, validate_zone_polygon
        )
        self.reference_image = None
        self.active_camera_id = None
        self.editor_mode = "view"
        self.polygon_points = []
        self.dragged_corner_index = None
        self.editing_zone_id = None
        self.updating_widget_values = False
        self.widgets = widgets
        self.camera_selector = widgets.Dropdown(
            description="Saved camera:",
            options=[],
            style={"description_width": "110px"},
            layout=widgets.Layout(width="460px"),
        )
        self.camera_selector.observe(self.handle_camera_selection, names="value")
        self.reload_button = self.button("Reload saved cameras", self.reload)
        self.new_camera_button = self.button("Add camera", self.open_camera_form)
        self.camera_name = widgets.Text(
            description="Camera name:",
            placeholder="camera_1",
            style={"description_width": "110px"},
        )
        self.media_path = widgets.Text(
            description="Image/video:",
            placeholder="Full local path",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="650px"),
        )
        self.upload = widgets.FileUpload(
            accept="image/*", multiple=False, description="Choose image"
        )
        self.media_mode = widgets.Dropdown(
            description="Reference from:",
            options=[("Image/video path", "path"), ("Uploaded image", "upload")],
            style={"description_width": "110px"},
        )
        self.upload.observe(
            lambda change: (
                setattr(self.media_mode, "value", "upload") if change["new"] else None
            ),
            names="value",
        )
        self.media_path.observe(
            lambda change: (
                setattr(self.media_mode, "value", "path") if change["new"] else None
            ),
            names="value",
        )
        self.create_camera_button = self.button(
            "Save camera", self.create_camera, "success"
        )
        self.attach_button = self.button(
            "Attach missing reference", self.attach_reference
        )
        self.camera_form = widgets.VBox(
            [
                widgets.HTML(
                    "Choose an image OR enter a video path. Use the same fixed viewpoint when recording."
                ),
                self.camera_name,
                self.media_mode,
                self.media_path,
                self.upload,
                widgets.HBox([self.create_camera_button, self.attach_button]),
            ]
        )
        self.camera_form.layout.display = "none"
        self.zone_selector = widgets.Select(
            description="Saved zones:",
            options=[],
            rows=5,
            layout=widgets.Layout(width="460px"),
            style={"description_width": "110px"},
        )
        self.zone_selector.observe(self.handle_zone_selection, names="value")
        self.details = widgets.HTML()
        self.add_button = self.button("Add zone", self.start_adding_zone, "success")
        self.edit_button = self.button(
            "Edit selected zone", self.start_editing_zone, "info"
        )
        self.remove_button = self.button(
            "Remove selected zone", self.start_removing_zone, "danger"
        )
        self.shared_zone = widgets.Dropdown(
            description="Reuse rule:",
            options=[],
            style={"description_width": "110px"},
            layout=widgets.Layout(width="460px"),
        )
        self.shared_zone.observe(self.handle_rule_selection, names="value")
        self.zone_name_input = widgets.Text(
            description="Zone name:",
            placeholder="Machine area",
            style={"description_width": "110px"},
        )
        self.required_ppe_input = widgets.Text(
            description="Required PPE:",
            placeholder="helmet, vest, boots (leave empty for none)",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="650px"),
        )
        self.all_views = widgets.Checkbox(
            value=False,
            description="Apply name/PPE changes to other cameras sharing this rule",
            indent=False,
        )
        self.undo_button = self.button("Undo last corner", self.undo_last_corner)
        self.redraw_button = self.button("Redraw boundary", self.clear_draft_boundary)
        self.save_button = self.button("Save zone", self.save_zone, "success")
        self.cancel_button = self.button("Cancel", self.cancel)
        self.form = widgets.VBox(
            [
                self.shared_zone,
                self.zone_name_input,
                self.required_ppe_input,
                self.all_views,
                widgets.HBox(
                    [
                        self.undo_button,
                        self.redraw_button,
                        self.save_button,
                        self.cancel_button,
                    ]
                ),
            ]
        )
        self.form.layout.display = "none"
        self.remove_confirm = self.button(
            "Yes, remove from this camera", self.confirm_zone_removal, "danger"
        )
        self.remove_cancel = self.button("Cancel removal", self.cancel)
        self.remove_form = widgets.VBox(
            [
                widgets.HTML(
                    "Remove the selected zone from this camera? Other camera views keep their zones."
                ),
                widgets.HBox([self.remove_confirm, self.remove_cancel]),
            ]
        )
        self.remove_form.layout.display = "none"
        self.status = widgets.HTML()
        self.instructions = widgets.HTML()
        self.test = widgets.Checkbox(
            description="Test a point (view mode)", value=False
        )
        with plt.ioff():
            self.figure, self.axes = plt.subplots(figsize=(12, 7))
        self.figure.canvas.mpl_connect("button_press_event", self.handle_canvas_click)
        self.figure.canvas.mpl_connect("motion_notify_event", self.handle_canvas_motion)
        self.figure.canvas.mpl_connect(
            "button_release_event", self.handle_canvas_release
        )
        display(
            widgets.VBox(
                [
                    widgets.HTML("<h3>Camera zones</h3>"),
                    widgets.HBox(
                        [
                            self.camera_selector,
                            self.reload_button,
                            self.new_camera_button,
                        ]
                    ),
                    self.camera_form,
                    self.zone_selector,
                    self.details,
                    widgets.HBox(
                        [self.add_button, self.edit_button, self.remove_button]
                    ),
                    self.form,
                    self.remove_form,
                    self.instructions,
                    self.test,
                    self.figure.canvas,
                    self.status,
                ]
            )
        )
        self.refresh_cameras()
        if self.camera_selector.value:
            self.load_camera(self.camera_selector.value)
        else:
            self.message(
                "Start with Add camera, choose your reference and Save camera."
            )
        self.update_controls()
        self.redraw_canvas()

    def button(self, label, action, style=""):
        b = self.widgets.Button(
            description=label,
            button_style=style,
            layout=self.widgets.Layout(width="215px"),
        )

        def handler(_):
            try:
                action()
            except Exception as error:
                self.message(str(error), error=True)

        b.on_click(handler)
        return b

    def message(self, text, error=False):
        colour = "#a12622" if error else "#176044"
        self.status.value = f'<div style="padding:10px;background:#f3f6f8;color:{colour}">{html.escape(text)}</div>'

    def refresh_cameras(self, select=None):
        self.updating_widget_values = True
        self.camera_selector.options = [
            (c, c) for c in self.config_store.config["cameras"]
        ]
        if select in self.config_store.config["cameras"]:
            self.camera_selector.value = select
        elif self.camera_selector.value is None and self.camera_selector.options:
            # An empty dropdown can keep None after its options are populated.
            self.camera_selector.value = self.camera_selector.options[0][1]
        self.updating_widget_values = False

    def reload(self):
        self.config_store.reload()
        self.refresh_cameras(self.active_camera_id)
        if self.camera_selector.value:
            self.load_camera(self.camera_selector.value)
        else:
            self.active_camera_id = None
            self.reference_image = None
            self.redraw_canvas()

    def handle_camera_selection(self, change):
        if self.updating_widget_values or not change["new"]:
            return
        try:
            self.config_store.reload()
            self.load_camera(change["new"])
        except Exception as error:
            self.message(str(error), True)

    @staticmethod
    def unpack_uploaded_file(value):
        """Read both ipywidgets 7 filename dictionaries and version 8 tuples."""
        if not value:
            raise ValueError("Choose an image or video first.")
        if isinstance(value, dict):
            original_name, item = next(iter(value.items()))
        else:
            item = value[0]
            original_name = "reference"
        metadata = item.get("metadata", {}) or {}
        filename = item.get("name") or metadata.get("name") or original_name
        data = bytes(item["content"])
        if not data:
            raise ValueError("The selected file is empty. Choose another file.")
        return str(filename), data

    @classmethod
    def read_uploaded_media(cls, value):
        """Decode an image, or extract a video frame from uploaded bytes."""
        import tempfile
        import numpy as np
        from PIL import Image, ImageOps, UnidentifiedImageError

        filename, data = cls.unpack_uploaded_file(value)
        suffix = Path(filename).suffix.lower()
        video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mpeg", ".mpg"}
        if suffix not in video_suffixes:
            try:
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                return np.asarray(image), filename
            except (UnidentifiedImageError, OSError):
                # Some widget versions omit file metadata. Try a video decoder too.
                pass
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            media = Path(folder) / ("reference" + (suffix if suffix in video_suffixes else ".mp4"))
            media.write_bytes(data)
            cap = cv2.VideoCapture(str(media))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
        if not ok or frame is None:
            raise ValueError(
                f"Cannot open {filename!r} as an image or video. "
                "Check that the file plays normally; for unsupported videos, export as H.264 MP4."
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), filename

    def read_media(self):
        import numpy as np
        from PIL import Image, ImageOps

        value = self.upload.value
        if self.media_mode.value == "upload":
            return self.read_uploaded_media(value)
        path = Path(self.media_path.value.strip().strip('"')).expanduser()
        if not path.is_file():
            raise ValueError("Choose an image or enter an existing video/image path.")
        return self.read_reference_media(path), str(path.resolve())

    @staticmethod
    def read_reference_media(path):
        import numpy as np
        from PIL import Image, ImageOps

        if Path(path).suffix.lower() in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".m4v",
            ".wmv", ".webm", ".mpeg", ".mpg",
        }:
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise ValueError("Cannot read the first video frame.")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))

    @staticmethod
    def encode_reference_png(reference_image):
        from PIL import Image

        stream = io.BytesIO()
        Image.fromarray(reference_image).save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def load_camera(self, camera_id):
        import numpy as np
        from PIL import Image

        self.active_camera_id = camera_id
        self.reference_image = None
        camera_config = self.config_store.config["cameras"][camera_id]
        try:
            if camera_config.get("reference_png"):
                self.reference_image = np.asarray(
                    Image.open(
                        io.BytesIO(base64.b64decode(camera_config["reference_png"]))
                    ).convert("RGB")
                )
            elif (
                camera_config.get("reference_source")
                and Path(camera_config["reference_source"]).is_file()
            ):
                reference_image = self.read_reference_media(
                    camera_config["reference_source"]
                )
                if [
                    reference_image.shape[1],
                    reference_image.shape[0],
                ] != camera_config["reference_size"]:
                    raise ValueError(
                        "Old reference orientation/size differs. Attach the exact original reference or add a new camera."
                    )
                self.reference_image = reference_image
        except Exception as error:
            self.message(
                "Zones loaded, but the reference could not be opened: " + str(error),
                True,
            )
        self.refresh_zones()
        self.redraw_canvas()
        self.update_controls()
        count = len(camera_config["zones"])
        if self.reference_image is None:
            self.message(
                f"{camera_id}: {count} saved zone(s) loaded. Attach the matching reference once to display them.",
                True,
            )
            self.open_camera_form()
            self.camera_name.value = camera_id
        else:
            self.camera_form.layout.display = "none"
            self.message(
                f"{camera_id}: reference and {count} saved zone(s) loaded automatically."
            )

    def refresh_zones(self, select=None):
        self.updating_widget_values = True
        camera_config = self.config_store.config["cameras"].get(
            self.active_camera_id, {"zones": {}}
        )
        self.zone_selector.options = [
            (self.config_store.config["zone_rules"][z]["name"], z)
            for z in camera_config["zones"]
        ]
        if select in camera_config["zones"]:
            self.zone_selector.value = select
        self.updating_widget_values = False
        self.handle_zone_selection(None)

    def handle_zone_selection(self, _):
        if self.updating_widget_values:
            return
        z = self.zone_selector.value
        if z:
            rule = self.config_store.config["zone_rules"][z]
            self.details.value = "<b>Required PPE:</b> " + html.escape(
                ", ".join(rule["required_ppe"]) or "None"
            )
        else:
            self.details.value = "No zones saved for this camera. Choose Add zone."
        self.update_controls()
        self.redraw_canvas()

    def open_camera_form(self):
        self.camera_form.layout.display = "flex"
        self.camera_name.value = ""
        self.message(
            "Enter a NEW camera name and reference, then Save camera. For a missing old reference, use Attach missing reference."
        )

    def create_camera(self):
        reference_image, source = self.read_media()
        c = self.config_store.add_camera(
            self.camera_name.value,
            [reference_image.shape[1], reference_image.shape[0]],
            self.encode_reference_png(reference_image),
            source,
        )
        self.refresh_cameras(c)
        self.load_camera(c)

    def attach_reference(self):
        if not self.active_camera_id:
            raise ValueError("Select a saved camera first.")
        if self.reference_image is not None:
            raise ValueError(
                "This camera already has a working reference. A different viewpoint needs Add camera."
            )
        reference_image, source = self.read_media()
        self.config_store.attach_reference(
            self.active_camera_id,
            [reference_image.shape[1], reference_image.shape[0]],
            self.encode_reference_png(reference_image),
            source,
        )
        self.load_camera(self.active_camera_id)

    def start_adding_zone(self):
        self.editor_mode = "add"
        self.polygon_points = []
        self.editing_zone_id = None
        self.updating_widget_values = True
        self.shared_zone.options = [("Create a new zone", None)] + [
            (r["name"] + " [" + z + "]", z)
            for z, r in self.config_store.config["zone_rules"].items()
            if z
            not in self.config_store.config["cameras"][self.active_camera_id]["zones"]
        ]
        self.shared_zone.value = None
        self.updating_widget_values = False
        self.zone_name_input.value = ""
        self.required_ppe_input.value = ""
        self.all_views.value = False
        self.form.layout.display = "flex"
        self.shared_zone.layout.display = ""
        self.update_controls()
        self.redraw_canvas()
        self.message(
            "Type a zone name and PPE, click its floor corners, then Save zone."
        )

    def handle_rule_selection(self, change):
        if self.updating_widget_values or self.editor_mode != "add":
            return
        z = change["new"]
        self.editing_zone_id = z
        if z:
            r = self.config_store.config["zone_rules"][z]
            self.zone_name_input.value = r["name"]
            self.required_ppe_input.value = ", ".join(r["required_ppe"])
        else:
            self.zone_name_input.value = ""
            self.required_ppe_input.value = ""
        self.update_controls()

    def start_editing_zone(self):
        self.editing_zone_id = self.zone_selector.value
        if not self.editing_zone_id:
            return
        self.editor_mode = "edit"
        r = self.config_store.config["zone_rules"][self.editing_zone_id]
        self.zone_name_input.value = r["name"]
        self.required_ppe_input.value = ", ".join(r["required_ppe"])
        self.polygon_points = copy.deepcopy(
            self.config_store.config["cameras"][self.active_camera_id]["zones"][
                self.editing_zone_id
            ]
        )
        self.all_views.value = False
        self.shared_zone.layout.display = "none"
        self.form.layout.display = "flex"
        self.update_controls()
        self.redraw_canvas()
        self.message(
            "Edit the name/PPE, drag corners, or Redraw boundary. Save changes or Cancel."
        )

    def start_removing_zone(self):
        self.editor_mode = "remove"
        self.editing_zone_id = self.zone_selector.value
        self.remove_form.layout.display = "flex"
        self.update_controls()

    def confirm_zone_removal(self):
        self.config_store.remove_zone(self.active_camera_id, self.editing_zone_id)
        self.finish_zone_edit()
        self.refresh_zones()
        self.message(
            "Zone removed and saved for this camera. Other camera views are unchanged."
        )

    def save_zone(self):
        z = self.config_store.save_zone(
            self.active_camera_id,
            self.editing_zone_id,
            self.zone_name_input.value,
            self.required_ppe_input.value.split(","),
            self.polygon_points,
            self.all_views.value,
        )
        self.finish_zone_edit()
        self.refresh_zones(z)
        self.message(
            "Zone saved. It will load automatically whenever you select this camera."
        )

    def finish_zone_edit(self):
        self.editor_mode = "view"
        self.polygon_points = []
        self.dragged_corner_index = None
        self.editing_zone_id = None
        self.form.layout.display = "none"
        self.remove_form.layout.display = "none"
        self.update_controls()
        self.redraw_canvas()

    def cancel(self):
        self.finish_zone_edit()
        self.message("Cancelled. Saved zones were not changed.")

    def clear_draft_boundary(self):
        self.polygon_points = []
        self.dragged_corner_index = None
        self.redraw_canvas()

    def undo_last_corner(self):
        if self.polygon_points:
            self.polygon_points.pop()
        self.dragged_corner_index = None
        self.redraw_canvas()

    # Disable competing actions while a draft or deletion confirmation is open.
    def update_controls(self):
        viewing = self.editor_mode == "view"
        ready = self.reference_image is not None
        self.camera_selector.disabled = not viewing
        self.zone_selector.disabled = not viewing
        self.reload_button.disabled = not viewing
        self.new_camera_button.disabled = not viewing
        self.create_camera_button.disabled = not viewing
        self.attach_button.disabled = not viewing
        self.add_button.disabled = not (viewing and ready)
        self.edit_button.disabled = not (viewing and ready and self.zone_selector.value)
        self.remove_button.disabled = not (viewing and self.zone_selector.value)
        self.test.disabled = not viewing
        shared = self.editing_zone_id is not None and any(
            c != self.active_camera_id and self.editing_zone_id in v["zones"]
            for c, v in self.config_store.config["cameras"].items()
        )
        self.all_views.layout.display = "" if shared else "none"
        self.save_button.description = (
            "Save changes" if self.editor_mode == "edit" else "Save zone"
        )
        self.instructions.value = (
            "<b>Draw mode:</b> click corners in order; drag a red corner to move it. Finish with Save or Cancel."
            if not viewing
            else "<b>View mode:</b> select a zone, then Add / Edit / Remove. Saved zones are shown below."
        )

    # Select saved polygons in view mode; add/drag draft corners in edit mode.
    def handle_canvas_click(self, event):
        if (
            event.inaxes != self.axes
            or event.button != 1
            or self.reference_image is None
        ):
            return
        if self.figure.canvas.toolbar and self.figure.canvas.toolbar.mode:
            return
        p = [float(event.xdata), float(event.ydata)]
        h, w = self.reference_image.shape[:2]
        p = [min(w - 1, max(0, p[0])), min(h - 1, max(0, p[1]))]
        if self.editor_mode == "view":
            camera_config = self.config_store.config["cameras"][self.active_camera_id]
            matches = [
                z
                for z, poly in camera_config["zones"].items()
                if self.contains_point(p, poly)
            ]
            if self.test.value:
                names = [
                    self.config_store.config["zone_rules"][z]["name"] for z in matches
                ]
                self.message(
                    "Point belongs to: " + (", ".join(names) or "Outside defined zones")
                )
            elif matches:
                self.zone_selector.value = matches[-1]
            return
        if self.editor_mode not in ("add", "edit"):
            return
        for i, corner in enumerate(self.polygon_points):
            xy = self.axes.transData.transform(corner)
            if math.hypot(xy[0] - event.x, xy[1] - event.y) < 10:
                self.dragged_corner_index = i
                return
        self.polygon_points.append(p)
        self.redraw_canvas()

    def handle_canvas_motion(self, event):
        if self.dragged_corner_index is None or event.inaxes != self.axes:
            return
        h, w = self.reference_image.shape[:2]
        self.polygon_points[self.dragged_corner_index] = [
            min(w - 1, max(0, float(event.xdata))),
            min(h - 1, max(0, float(event.ydata))),
        ]
        self.redraw_canvas()

    def handle_canvas_release(self, _):
        self.dragged_corner_index = None

    @staticmethod
    def contains_point(point, polygon):
        # Use the same geometry as downstream localisation.
        if "point_in_polygon" in globals():
            return point_in_polygon(point, polygon)
        from zone_core import point_in_polygon

        return point_in_polygon(point, polygon)

    def redraw_canvas(self):
        from matplotlib.patches import Polygon

        self.axes.clear()
        if self.reference_image is None:
            self.axes.text(
                0.5,
                0.5,
                "Select a camera or Add camera to load its reference",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
            )
            self.axes.set_axis_off()
        else:
            self.axes.imshow(self.reference_image)
            camera_config = self.config_store.config["cameras"][self.active_camera_id]
            for z, poly in camera_config["zones"].items():
                selected = z == self.zone_selector.value
                if self.editor_mode == "edit" and z == self.editing_zone_id:
                    continue
                colour = "#ffb000" if selected else "#168aad"
                self.axes.add_patch(
                    Polygon(
                        poly,
                        closed=True,
                        facecolor=colour,
                        edgecolor=colour,
                        alpha=0.25,
                        linewidth=3 if selected else 1.5,
                    )
                )
                self.axes.text(
                    poly[0][0],
                    poly[0][1],
                    self.config_store.config["zone_rules"][z]["name"],
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.65},
                )
            if self.polygon_points:
                x, y = zip(*self.polygon_points)
                self.axes.plot(x, y, "ro-", linewidth=2)
                if len(self.polygon_points) >= 3:
                    self.axes.add_patch(
                        Polygon(
                            self.polygon_points,
                            closed=True,
                            fill=False,
                            edgecolor="red",
                            linestyle="--",
                        )
                    )
            h, w = self.reference_image.shape[:2]
            self.axes.set_xlim(-0.5, w - 0.5)
            self.axes.set_ylim(h - 0.5, -0.5)
            self.axes.set_title(
                f'{self.active_camera_id} | {self.editor_mode.upper()} | {len(camera_config["zones"])} saved zones'
            )
            self.axes.set_xlabel("Image x (pixels)")
            self.axes.set_ylabel("Image y (pixels)")
        self.figure.canvas.draw_idle()


# Compatibility name for existing notebook calls.
ZoneEditor = ZoneEditorBase


# User-facing layout: reuse the base callbacks while showing only the current task.
class CameraZoneEditor(ZoneEditorBase):
    """Same saved schema; show only the current task using plain instructions."""

    def __init__(self, config_path="zone_config.json"):
        # Reuse the tested geometry and persistence callbacks, not the old layout.
        from IPython.utils.capture import capture_output
        from IPython.display import display

        self.layout_ready = False
        with capture_output():
            super().__init__(config_path)
        widgets = self.widgets
        self.adding_camera = False
        self.camera_pending_removal = None
        self.remove_camera_button = self.button(
            "Remove camera", self.begin_camera_removal, "danger"
        )
        self.camera_removal_message = widgets.HTML()
        self.confirm_camera_removal_button = self.button(
            "Remove camera and zones", self.confirm_camera_removal, "danger"
        )
        self.cancel_camera_removal_button = self.button(
            "Cancel", self.cancel_camera_removal
        )
        self.camera_removal_panel = widgets.VBox(
            [
                self.camera_removal_message,
                widgets.HBox(
                    [
                        self.confirm_camera_removal_button,
                        self.cancel_camera_removal_button,
                    ]
                ),
            ]
        )
        self.camera_removal_panel.layout.display = "none"
        self.reference_upload = widgets.FileUpload(
            accept="image/*,video/*",
            multiple=False,
            description="Upload image / video",
            layout=widgets.Layout(width="240px"),
        )
        self.reference_upload.observe(self.handle_reference_upload, names="value")
        self.reference_title = widgets.HTML()
        self.reference_hint = widgets.HTML()
        self.reference_cancel = self.button("Cancel", self.cancel_reference)
        self.reference_box = widgets.VBox(
            [
                self.reference_title,
                self.camera_name,
                self.reference_hint,
                widgets.HBox([self.reference_upload, self.reference_cancel]),
            ]
        )
        self.manage_box = widgets.VBox(
            [
                widgets.HTML("<h4>2. Choose what to do</h4>"),
                self.zone_selector,
                self.details,
                widgets.HBox([self.add_button, self.edit_button, self.remove_button]),
            ]
        )
        self.form.children = (
            self.zone_name_input,
            self.required_ppe_input,
            widgets.HTML(
                "PPE names: separate with commas. Leave empty if no PPE is required."
            ),
            widgets.HBox([self.undo_button, self.redraw_button]),
            widgets.HBox([self.save_button, self.cancel_button]),
        )
        self.remove_form.children = (
            widgets.HTML(
                "<b>Remove this zone from this camera?</b><br>Other cameras will keep their zones."
            ),
            widgets.HBox([self.remove_confirm, self.remove_cancel]),
        )
        self.camera_selector.description = "Camera:"
        self.camera_name.description = "New camera:"
        self.camera_name.placeholder = "Example: Lab entrance"
        self.zone_selector.description = "Zone:"
        self.zone_selector.rows = 3
        self.zone_name_input.description = "Zone name:"
        self.required_ppe_input.description = "Required PPE:"
        self.add_button.description = "Add a zone"
        self.edit_button.description = "Edit this zone"
        self.remove_button.description = "Remove this zone"
        self.new_camera_button.description = "+ New camera"
        self.undo_button.description = "Undo last point"
        self.redraw_button.description = "Draw boundary again"
        self.cancel_button.description = "Cancel changes"
        self.remove_confirm.description = "Remove zone"
        # Hide plot toolbars: pan/zoom mode previously blocked annotation clicks.
        for attribute in ("toolbar_visible", "header_visible", "footer_visible"):
            if hasattr(self.figure.canvas, attribute):
                setattr(self.figure.canvas, attribute, False)
        self.figure.set_size_inches(11, 6)
        self.layout_ready = True
        display(
            widgets.VBox(
                [
                    widgets.HTML(
                        "<h3>Mark your camera zones</h3><h4>1. Open a camera</h4>"
                    ),
                    widgets.HBox(
                        [
                            self.camera_selector,
                            self.new_camera_button,
                            self.remove_camera_button,
                        ]
                    ),
                    self.camera_removal_panel,
                    self.reference_box,
                    self.manage_box,
                    self.form,
                    self.remove_form,
                    self.instructions,
                    self.status,
                    self.figure.canvas,
                ]
            )
        )
        self.camera_form.layout.display = "none"
        self.update_layout()
        if self.reference_image is None:
            if self.active_camera_id:
                self.message(
                    "Your zones are saved. Upload the matching image/video once to show this camera."
                )
            else:
                self.message(
                    "Start here: click + New camera, enter a name, then upload an image or video."
                )
        self.redraw_canvas()

    def update_layout(self):
        if not self.layout_ready:
            return
        removing_camera = self.editor_mode == "remove_camera"
        self.camera_removal_panel.layout.display = "flex" if removing_camera else "none"
        self.remove_camera_button.disabled = (
            self.editor_mode != "view"
            or self.adding_camera
            or self.active_camera_id is None
        )
        self.reference_upload.disabled = self.editor_mode != "view"
        missing = self.reference_image is None and self.active_camera_id is not None
        show_reference = self.adding_camera or missing
        self.reference_box.layout.display = "flex" if show_reference else "none"
        self.camera_name.layout.display = "" if self.adding_camera else "none"
        self.reference_cancel.layout.display = "" if self.adding_camera else "none"
        if self.adding_camera:
            self.reference_title.value = "<b>Add a camera</b>"
            self.reference_hint.value = "Type a camera name above, then upload its reference image or a short video. It saves automatically."
        elif missing:
            self.reference_title.value = (
                "<b>Upload the reference for "
                + html.escape(self.active_camera_id)
                + "</b>"
            )
            self.reference_hint.value = "The old image/video file could not be found. Upload the same view used to draw these zones. Your saved zones will appear on it."
        self.manage_box.layout.display = (
            "flex"
            if self.reference_image is not None and not self.adding_camera
            else "none"
        )
        self.camera_selector.disabled = self.editor_mode != "view" or self.adding_camera
        self.new_camera_button.disabled = (
            self.editor_mode != "view" or self.adding_camera
        )
        self.shared_zone.layout.display = "none"
        self.all_views.layout.display = "none"
        self.test.layout.display = "none"
        if self.editor_mode == "view":
            self.instructions.value = (
                "Click <b>Add a zone</b> to draw a new floor area, or select an existing zone to edit/remove it."
                if self.reference_image is not None
                else ""
            )
        elif self.editor_mode in ("add", "edit"):
            self.instructions.value = "<b>Draw on the image:</b> click corners around the floor area. Drag a red corner to adjust it. Then click the green Save button. Cancel keeps the saved version."
        elif self.editor_mode in ("remove", "remove_camera"):
            self.instructions.value = ""
        self.form.layout.display = (
            "flex" if self.editor_mode in ("add", "edit") else "none"
        )
        self.remove_form.layout.display = (
            "flex" if self.editor_mode == "remove" else "none"
        )

    # Disable competing actions while a draft or deletion confirmation is open.
    def update_controls(self):
        super().update_controls()
        if getattr(self, "layout_ready", False):
            self.update_layout()

    def open_camera_form(self):
        if not getattr(self, "layout_ready", False):
            return super().open_camera_form()
        self.adding_camera = True
        self.camera_name.value = ""
        self.update_layout()
        self.message(
            "Enter a new camera name, then upload its image/video. Do not reuse a name for a different viewpoint."
        )

    def cancel_reference(self):
        self.adding_camera = False
        self.update_layout()
        self.message("Cancelled. Choose a saved camera or add a new one.")

    def load_camera(self, camera_id):
        # Parent restores polygons and embedded reference from the same camera.
        # Avoid its old missing-reference form opening as a NEW camera request.
        ready = getattr(self, "layout_ready", False)
        self.layout_ready = False
        try:
            super().load_camera(camera_id)
        finally:
            self.layout_ready = ready
        if ready:
            self.adding_camera = False
            self.camera_form.layout.display = "none"
            self.update_layout()
            if self.reference_image is None:
                self.message(
                    "Zones loaded. Upload the matching image/video to display this camera."
                )
            else:
                self.message("Camera opened. Its saved zones are displayed below.")

    def handle_reference_upload(self, change):
        if not change["new"]:
            return
        value = change["new"]
        try:
            if self.adding_camera and not self.camera_name.value.strip():
                raise ValueError(
                    "Enter a camera name first, then choose the file again."
                )
            reference_image, filename = self.read_uploaded_media(value)
            size = [reference_image.shape[1], reference_image.shape[0]]
            if self.adding_camera:
                camera_id = self.config_store.add_camera(
                    self.camera_name.value,
                    size,
                    self.encode_reference_png(reference_image),
                    filename,
                )
            else:
                camera_id = self.active_camera_id
                if camera_id is None:
                    raise ValueError("Click + New camera first.")
                self.config_store.attach_reference(
                    camera_id,
                    size,
                    self.encode_reference_png(reference_image),
                    filename,
                )
            self.adding_camera = False
            self.refresh_cameras(camera_id)
            self.load_camera(camera_id)
            self.message(
                "Camera saved and opened. Now click Add a zone, or select a saved zone to edit it."
            )
        except Exception as error:
            self.message(str(error), True)
        finally:
            # Let the same file be selected again after a failed attempt.
            try:
                self.reference_upload.value = () if not isinstance(value, dict) else {}
            except Exception:
                pass

    def start_adding_zone(self):
        super().start_adding_zone()
        self.message(
            "New zone: enter its name and PPE, click the floor corners, then Save zone."
        )
        self.update_layout()

    def start_editing_zone(self):
        super().start_editing_zone()
        self.message(
            "Edit this zone: update the name/PPE or drag its red corners, then Save changes. Changes apply to this camera."
        )
        self.update_layout()

    def clear_draft_boundary(self):
        super().clear_draft_boundary()
        self.message(
            "Boundary cleared in the draft. Click the new floor corners, then Save. Cancel restores the previous boundary."
        )

    def save_zone(self):
        super().save_zone()
        self.message("Saved! This zone will appear whenever you open this camera.")
        self.update_layout()

    def begin_camera_removal(self):
        """Pin the selected camera until the user confirms or cancels."""
        if (
            self.editor_mode != "view"
            or self.adding_camera
            or self.active_camera_id is None
        ):
            return
        self.camera_pending_removal = self.active_camera_id
        zone_count = len(
            self.config_store.config["cameras"][self.active_camera_id]["zones"]
        )
        camera_label = html.escape(self.active_camera_id)
        self.camera_removal_message.value = (
            f"<b>Remove {camera_label} and its {zone_count} saved zone(s)?</b><br>"
            "Other cameras, shared PPE rules and original image/video files are kept."
        )
        self.editor_mode = "remove_camera"
        self.update_controls()
        self.message("Confirm camera removal below, or Cancel to keep it.")

    def cancel_camera_removal(self):
        """Leave the saved configuration untouched."""
        self.camera_pending_removal = None
        self.editor_mode = "view"
        self.update_controls()
        self.message("Cancelled. The camera and its zones were kept.")

    def confirm_camera_removal(self):
        """Save deletion first, then select a surviving camera or clear the view."""
        if self.editor_mode != "remove_camera" or self.camera_pending_removal is None:
            return
        camera_id = self.camera_pending_removal
        # If saving fails, keep the confirmation visible and the draft unchanged.
        self.config_store.remove_camera(camera_id)
        self.camera_pending_removal = None
        self.editor_mode = "view"
        self.active_camera_id = None
        self.reference_image = None
        self.polygon_points = []
        self.dragged_corner_index = None
        self.editing_zone_id = None
        self.refresh_cameras()
        if self.camera_selector.value is not None:
            self.load_camera(self.camera_selector.value)
        else:
            # Removing the last camera must also clear stale zone/PPE details.
            self.refresh_zones()
            self.update_controls()
            self.redraw_canvas()
        self.message(
            f"Camera {camera_id} removed and saved. Use + New camera to add another view."
        )
