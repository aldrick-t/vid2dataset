"""PySide6 GUI for vid2frame.

Features
- Single and batch processing using the existing vid2frame pipeline.
- Load/save YAML configs and named profiles.
- Frame preview with scrubbing, optional downscaling to 1080p, and crop overlay.
- Interactive crop rectangle with aspect lock and numeric controls.
- Optional per-video overrides in batch mode.
- Estimated frames/duration and simple progress reporting.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from vid2frame import FrameExtractionOptions, extract_frames, get_video_properties, is_video_file


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
PRESETS_PATH = CONFIG_DIR / "presets.yaml"
PROFILES_DIR = CONFIG_DIR / "profiles"


DEFAULT_CONFIG: Dict[str, Any] = {
	"input_dir": "input",
	"output_dir": "output",
	"interval": 1,
	"prefix": "frame",
	"start_number": 0,
	"limit": None,
	"crop_area": None,
	"resize_to": None,
	"color_space": None,
	"ext": "png",
	"use_datetime_token": False,
	"datetime_format": "%Y%m%d_%H%M%S",
	"parallel_workers": 2,
	"verbose": True,
	"progress": True,
}


def ensure_files() -> None:
	CONFIG_DIR.mkdir(parents=True, exist_ok=True)
	PROFILES_DIR.mkdir(parents=True, exist_ok=True)
	if not CONFIG_PATH.exists():
		save_yaml(CONFIG_PATH, DEFAULT_CONFIG)
	if not PRESETS_PATH.exists():
		save_yaml(PRESETS_PATH, {"presets": []})


def load_yaml(path: Path, default: Any) -> Any:
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as handle:
		data = yaml.safe_load(handle)
	return default if data is None else data


def save_yaml(path: Path, data: Any) -> None:
	with path.open("w", encoding="utf-8") as handle:
		yaml.safe_dump(data, handle, sort_keys=False)


def resolve_dir(path_str: str) -> Path:
	path = Path(path_str)
	if not path.is_absolute():
		path = (BASE_DIR / path).resolve()
	return path


def build_options(config: Dict[str, Any]) -> Dict[str, Any]:
	resize_to = None
	if config.get("resize_to") and config["resize_to"].get("width") is not None and config["resize_to"].get("height") is not None:
		resize_to = (
			int(config["resize_to"]["width"]),
			int(config["resize_to"]["height"]),
		)
	crop_area = None
	if config.get("crop_area"):
		crop_area = tuple(int(x) for x in config["crop_area"])
	return {
		"interval": int(config["interval"]),
		"prefix": config["prefix"],
		"start_number": int(config["start_number"]),
		"limit": config.get("limit"),
		"crop_area": crop_area,
		"resize_to": resize_to,
		"color_space": config.get("color_space"),
		"ext": config["ext"],
		"use_datetime_token": bool(config.get("use_datetime_token", False)),
		"datetime_format": config.get("datetime_format", "%Y%m%d_%H%M%S"),
	}


def merge_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
	merged = {**base}
	for key, value in overrides.items():
		merged[key] = value
	return merged


def format_estimate(props: Dict[str, float], options: Dict[str, Any]) -> str:
	frame_count = props.get("frame_count") or 0
	fps = props.get("fps") or 0.0
	interval = max(1, options.get("interval", 1))
	estimated = frame_count // interval
	limit = options.get("limit")
	if limit is not None:
		estimated = min(estimated, int(limit))
	duration = props.get("duration")
	duration_str = f"{duration:.2f}s" if duration else "n/a"
	return f"Frames: {frame_count} | FPS: {fps:.2f} | Duration: {duration_str} | Estimated output: {estimated}"


class FrameView(QtWidgets.QLabel):
	crop_changed = QtCore.Signal(tuple)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setAlignment(QtCore.Qt.AlignCenter)
		self.setMinimumSize(320, 240)
		self._pixmap: Optional[QtGui.QPixmap] = None
		self._scale: float = 1.0
		self._display_size: Tuple[int, int] = (1, 1)
		self._source_size: Tuple[int, int] = (1, 1)
		self._crop: Optional[Tuple[int, int, int, int]] = None
		self._drag_start: Optional[QtCore.QPoint] = None
		self._drag_mode: Optional[str] = None  # "new" or "move"
		self._drag_offset: Optional[Tuple[int, int]] = None
		self._aspect_lock: bool = False
		self._lock_ratio: Optional[float] = None
		self.setMouseTracking(True)

	def set_image(self, image: QtGui.QImage, display_size: Tuple[int, int], source_size: Tuple[int, int]) -> None:
		self._display_size = display_size
		self._source_size = source_size
		self._pixmap = QtGui.QPixmap.fromImage(image)
		self._scale = min(
			self.width() / display_size[0],
			self.height() / display_size[1],
			1.0,
		)
		self.update()

	def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
		if self._display_size:
			self._scale = min(
				self.width() / self._display_size[0],
				self.height() / self._display_size[1],
				1.0,
			)
		super().resizeEvent(event)

	def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
		super().paintEvent(event)
		if not self._pixmap:
			return
		painter = QtGui.QPainter(self)
		scaled = self._pixmap.scaled(
			self._display_size[0] * self._scale,
			self._display_size[1] * self._scale,
			QtCore.Qt.KeepAspectRatio,
			QtCore.Qt.SmoothTransformation,
		)
		x = (self.width() - scaled.width()) // 2
		y = (self.height() - scaled.height()) // 2
		painter.drawPixmap(x, y, scaled)
		if self._crop:
			cx, cy, cw, ch = self._crop
			# Map source coords -> display coords -> widget coords
			sx = self._display_size[0] / self._source_size[0] if self._source_size[0] else 1.0
			sy = self._display_size[1] / self._source_size[1] if self._source_size[1] else 1.0
			pen = QtGui.QPen(QtGui.QColor(0, 200, 255), 2, QtCore.Qt.DashLine)
			painter.setPen(pen)
			painter.setBrush(QtCore.Qt.transparent)
			painter.drawRect(
				x + cx * sx * self._scale,
				y + cy * sy * self._scale,
				cw * sx * self._scale,
				ch * sy * self._scale,
			)
		painter.end()

	def set_crop(self, crop: Optional[Tuple[int, int, int, int]]) -> None:
		self._crop = crop
		self.update()

	def set_aspect_lock(self, locked: bool, ratio: Optional[float]) -> None:
		self._aspect_lock = locked
		self._lock_ratio = ratio

	def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
		if not self._pixmap:
			return
		self._drag_start = event.position().toPoint()
		self._drag_mode = None
		self._drag_offset = None
		# Determine if click is inside existing crop -> move mode
		if self._crop:
			cx, cy, cw, ch = self._crop
			if self._point_in_crop(event.position().toPoint()):
				self._drag_mode = "move"
				self._drag_offset = (event.position().toPoint().x(), event.position().toPoint().y())
		if self._drag_mode is None:
			self._drag_mode = "new"

	def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
		if not self._pixmap or not self._drag_start:
			return
		if self._drag_mode == "move" and self._crop:
			self._update_crop_move(event.position().toPoint())
		else:
			self._update_crop_from_drag(event.position().toPoint())

	def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
		if not self._pixmap or not self._drag_start:
			return
		if self._drag_mode == "move" and self._crop:
			self._update_crop_move(event.position().toPoint())
		else:
			self._update_crop_from_drag(event.position().toPoint())
		self._drag_start = None
		self._drag_mode = None
		self._drag_offset = None

	def _update_crop_from_drag(self, current: QtCore.QPoint) -> None:
		if not self._display_size:
			return
		start = self._drag_start or current
		x0 = min(start.x(), current.x())
		y0 = min(start.y(), current.y())
		x1 = max(start.x(), current.x())
		y1 = max(start.y(), current.y())
		frame_w, frame_h = self._display_size
		scaled_w = frame_w * self._scale
		scaled_h = frame_h * self._scale
		offset_x = (self.width() - scaled_w) / 2
		offset_y = (self.height() - scaled_h) / 2
		rx0 = max(0, x0 - offset_x)
		ry0 = max(0, y0 - offset_y)
		rx1 = max(0, x1 - offset_x)
		ry1 = max(0, y1 - offset_y)
		if self._aspect_lock:
			ratio = self._lock_ratio
			if ratio is None and self._crop:
				_, _, cw, ch = self._crop
				ratio = cw / ch if ch else None
			if ratio:
				width = rx1 - rx0
				height = width / ratio
				ry1 = ry0 + height
		cx = int(rx0 / self._scale)
		cy = int(ry0 / self._scale)
		cw = int((rx1 - rx0) / self._scale)
		ch = int((ry1 - ry0) / self._scale)
		cw = min(cw, frame_w - cx)
		ch = min(ch, frame_h - cy)
		# Map back to source coordinates
		sx = self._source_size[0] / self._display_size[0] if self._display_size[0] else 1.0
		sy = self._source_size[1] / self._display_size[1] if self._display_size[1] else 1.0
		cx = int(cx * sx)
		cy = int(cy * sy)
		cw = int(cw * sx)
		ch = int(ch * sy)
		if cw <= 0 or ch <= 0:
			return
		self._crop = (cx, cy, cw, ch)
		self.crop_changed.emit(self._crop)
		self.update()

	def _point_in_crop(self, point: QtCore.QPoint) -> bool:
		if not self._crop:
			return False
		mapped = self._map_point_to_source(point)
		if mapped is None:
			return False
		px, py = mapped
		cx, cy, cw, ch = self._crop
		return cx <= px <= cx + cw and cy <= py <= cy + ch

	def _update_crop_move(self, current: QtCore.QPoint) -> None:
		if not self._crop or not self._display_size:
			return
		cx, cy, cw, ch = self._crop
		source_point = self._map_point_to_source(current)
		if source_point is None:
			return
		sx, sy = source_point
		# Move preserves size, align top-left to pointer offset if available
		if self._drag_start:
			start_source = self._map_point_to_source(self._drag_start)
		else:
			start_source = None
		if start_source:
			dx = sx - start_source[0]
			dy = sy - start_source[1]
		else:
			dx = dy = 0
		nx = max(0, min(cx + dx, self._source_size[0] - cw))
		ny = max(0, min(cy + dy, self._source_size[1] - ch))
		self._crop = (nx, ny, cw, ch)
		self.crop_changed.emit(self._crop)
		self.update()

	def _map_point_to_source(self, point: QtCore.QPoint) -> Optional[Tuple[int, int]]:
		if not self._display_size:
			return None
		scaled_w = self._display_size[0] * self._scale
		scaled_h = self._display_size[1] * self._scale
		offset_x = (self.width() - scaled_w) / 2
		offset_y = (self.height() - scaled_h) / 2
		dx = (point.x() - offset_x) / self._scale
		dy = (point.y() - offset_y) / self._scale
		if dx < 0 or dy < 0 or dx > self._display_size[0] or dy > self._display_size[1]:
			return None
		sx = self._source_size[0] / self._display_size[0] if self._display_size[0] else 1.0
		sy = self._source_size[1] / self._display_size[1] if self._display_size[1] else 1.0
		return int(dx * sx), int(dy * sy)


@dataclass
class VideoItem:
	path: Path
	overrides: Dict[str, Any] = field(default_factory=dict)


class OverrideDialog(QtWidgets.QDialog):
	def __init__(self, parent=None, existing: Optional[Dict[str, Any]] = None):
		super().__init__(parent)
		self.setWindowTitle("Per-video overrides")
		self.setModal(True)
		form = QtWidgets.QFormLayout(self)

		self.interval = QtWidgets.QSpinBox()
		self.interval.setMinimum(1)
		self.interval.setMaximum(10000)
		self.limit = QtWidgets.QSpinBox()
		self.limit.setMinimum(0)
		self.limit.setMaximum(1_000_000)
		self.limit.setSpecialValueText("None")
		self.crop = QtWidgets.QLineEdit()
		self.resize_to = QtWidgets.QLineEdit()
		self.color_space = QtWidgets.QComboBox()
		self.color_space.addItems(["", "BGR", "GRAY", "HSV", "RGB"])

		form.addRow("Interval", self.interval)
		form.addRow("Limit (0=None)", self.limit)
		form.addRow("Crop x,y,w,h", self.crop)
		form.addRow("Resize w,h", self.resize_to)
		form.addRow("Color space", self.color_space)

		btns = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
		)
		btns.accepted.connect(self.accept)
		btns.rejected.connect(self.reject)
		form.addRow(btns)

		if existing:
			if "interval" in existing:
				self.interval.setValue(int(existing["interval"]))
			if "limit" in existing and existing["limit"] is not None:
				self.limit.setValue(int(existing["limit"]))
			if existing.get("crop_area"):
				self.crop.setText(
					",".join(str(x) for x in existing.get("crop_area"))
				)
			if existing.get("resize_to"):
				self.resize_to.setText(
					",".join(
						[
							str(existing["resize_to"]["width"]),
							str(existing["resize_to"]["height"]),
						]
					)
				)
			if existing.get("color_space"):
				idx = self.color_space.findText(existing["color_space"], QtCore.Qt.MatchFixedString)
				if idx >= 0:
					self.color_space.setCurrentIndex(idx)

	def result_data(self) -> Dict[str, Any]:
		data: Dict[str, Any] = {}
		data["interval"] = self.interval.value()
		data["limit"] = self.limit.value() if self.limit.value() > 0 else None
		if self.crop.text().strip():
			parts = [int(x) for x in self.crop.text().split(",")]
			if len(parts) == 4:
				data["crop_area"] = parts
		if self.resize_to.text().strip():
			parts = [int(x) for x in self.resize_to.text().split(",")]
			if len(parts) == 2:
				data["resize_to"] = {"width": parts[0], "height": parts[1]}
		cs = self.color_space.currentText().strip()
		if cs:
			data["color_space"] = cs
		return data


class ExtractWorker(QtCore.QThread):
	progress = QtCore.Signal(str)
	finished = QtCore.Signal(list)

	def __init__(self, tasks: List[VideoItem], config: Dict[str, Any], output_base: Path):
		super().__init__()
		self.tasks = tasks
		self.config = config
		self.output_base = output_base
		self._cancel = threading.Event()
		self._lock_ratio: Optional[float] = None

	def cancel(self) -> None:
		self._cancel.set()

	def run(self) -> None:  # type: ignore[override]
		reports: List[Dict[str, Any]] = []
		for item in self.tasks:
			if self._cancel.is_set():
				reports.append({"video": str(item.path), "status": "cancelled", "frames": 0})
				break
			cfg = merge_overrides(self.config, item.overrides)
			try:
				props = get_video_properties(str(item.path))
			except Exception:
				props = {}
			target_resize = None
			if props.get("width") and props.get("height"):
				target_resize = self._resolve_resize_target((int(props["width"]), int(props["height"])), cfg.get("resize_to", {}))
			if target_resize:
				cfg["resize_to"] = {"width": target_resize[0], "height": target_resize[1]}
			else:
				cfg["resize_to"] = None
			options = build_options(cfg)
			run_tag = datetime.now().strftime(cfg.get("datetime_format", "%Y%m%d_%H%M%S"))
			output_dir = self.output_base / f"{cfg.get('prefix','frame')}_{run_tag}"
			output_dir.mkdir(parents=True, exist_ok=True)
			est = format_estimate(props, options)
			self.progress.emit(f"Processing {item.path.name} -> {output_dir}\n{est}")
			try:
				frames_written = extract_frames(
					str(item.path),
					str(output_dir),
					**options,
					on_frame_saved=None,
					should_stop=self._cancel.is_set,
				)
				reports.append({
					"video": str(item.path),
					"status": "ok" if not self._cancel.is_set() else "cancelled",
					"frames": frames_written,
				})
			except Exception as exc:
				reports.append({
					"video": str(item.path),
					"status": "error",
					"error": str(exc),
					"frames": 0,
				})
		self.finished.emit(reports)

	def _resolve_resize_target(self, source_size: Tuple[int, int], resize_cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
		"""Compute resize target, filling missing dimension from source aspect."""
		if not resize_cfg:
			return None
		sw, sh = source_size
		w = resize_cfg.get("width")
		h = resize_cfg.get("height")
		if w is not None and h is not None:
			return int(w), int(h)
		if w is not None:
			return int(w), int(round(int(w) * sh / sw)) if sw else int(w)
		if h is not None:
			return int(round(int(h) * sw / sh)) if sh else int(h), int(h)
		return None


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("vid2frame GUI")
		self.resize(1200, 780)
		self._config = load_yaml(CONFIG_PATH, DEFAULT_CONFIG)
		self._profiles = self._load_profiles()
		self._current_video: Optional[Path] = None
		self._video_props: Dict[str, float] = {}
		self._preview_downscaled = False
		self._frame_cache: Dict[int, tuple] = {}
		self._source_size: Tuple[int, int] = (0, 0)
		self._display_size: Tuple[int, int] = (0, 0)
		self._capture: Optional[cv2.VideoCapture] = None
		self._play_timer = QtCore.QTimer(self)
		self._play_timer.timeout.connect(self._advance_frame)
		self._worker: Optional[ExtractWorker] = None
		self._lock_ratio: Optional[float] = None
		self._respect_resize_preview = True
		self._source_wh: Tuple[int, int] = (0, 0)

		self._build_ui()
		self._apply_config_to_form()
		self._update_profiles_ui()

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
		if self._worker and self._worker.isRunning():
			self._worker.cancel()
			self._worker.wait(1000)
		if self._capture:
			self._capture.release()
		super().closeEvent(event)

	def _build_ui(self) -> None:
		central = QtWidgets.QWidget()
		layout = QtWidgets.QHBoxLayout(central)
		splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
		layout.addWidget(splitter)
		self.setCentralWidget(central)

		controls = self._build_controls_panel()
		splitter.addWidget(controls)
		preview = self._build_preview_panel()
		splitter.addWidget(preview)
		splitter.setStretchFactor(0, 0)
		splitter.setStretchFactor(1, 1)

		self._apply_dark_theme()

		# Allow user to resize tool/preview areas comfortably
		splitter.setHandleWidth(8)

	def _build_controls_panel(self) -> QtWidgets.QWidget:
		panel = QtWidgets.QWidget()
		vbox = QtWidgets.QVBoxLayout(panel)

		file_group = QtWidgets.QGroupBox("Input / Output")
		file_form = QtWidgets.QFormLayout(file_group)
		self.input_path = QtWidgets.QLineEdit()
		btn_in = QtWidgets.QPushButton("Browse video")
		btn_in.clicked.connect(self._select_video)
		in_row = QtWidgets.QHBoxLayout()
		in_row.addWidget(self.input_path)
		in_row.addWidget(btn_in)
		file_form.addRow("Video", self._row_widget(in_row))

		self.output_dir = QtWidgets.QLineEdit()
		btn_out = QtWidgets.QPushButton("Browse output")
		btn_out.clicked.connect(self._select_output)
		out_row = QtWidgets.QHBoxLayout()
		out_row.addWidget(self.output_dir)
		out_row.addWidget(btn_out)
		file_form.addRow("Output dir", self._row_widget(out_row))

		vbox.addWidget(file_group)

		settings_group = QtWidgets.QGroupBox("Settings")
		settings_form = QtWidgets.QFormLayout(settings_group)
		self.interval = QtWidgets.QSpinBox()
		self.interval.setMinimum(1)
		self.interval.setMaximum(10000)
		self.prefix = QtWidgets.QLineEdit()
		self.start_number = QtWidgets.QSpinBox()
		self.start_number.setMinimum(0)
		self.limit = QtWidgets.QSpinBox()
		self.limit.setMinimum(0)
		self.limit.setMaximum(1_000_000)
		self.limit.setSpecialValueText("None")
		self.crop = QtWidgets.QLineEdit()
		self.resize_to = QtWidgets.QLineEdit()
		self.color_space = QtWidgets.QComboBox()
		self.color_space.addItems(["", "BGR", "GRAY", "HSV", "RGB"])
		self.ext = QtWidgets.QLineEdit()
		self.use_datetime = QtWidgets.QCheckBox("Include datetime token")
		self.datetime_format = QtWidgets.QLineEdit()
		self.parallel_workers = QtWidgets.QSpinBox()
		self.parallel_workers.setMinimum(1)
		self.parallel_workers.setMaximum(64)
		self.verbose = QtWidgets.QCheckBox("Verbose")
		self.progress = QtWidgets.QCheckBox("Show progress")
		self.crop.editingFinished.connect(self._sync_crop_to_view)
		self.resize_to.editingFinished.connect(self._refresh_preview_resize)

		settings_form.addRow(self._with_help("Interval", "Extract every Nth frame."), self.interval)
		settings_form.addRow("Prefix", self.prefix)
		settings_form.addRow(self._with_help("Start #", "Starting index for filenames."), self.start_number)
		settings_form.addRow(self._with_help("Limit", "0 = no limit."), self.limit)
		settings_form.addRow(self._with_help("Crop x,y,w,h", "Drag on preview or enter values."), self.crop)
		settings_form.addRow(self._with_help("Resize w,h", "Scaled down to preview if >1080p."), self.resize_to)
		settings_form.addRow("Color space", self.color_space)
		settings_form.addRow("Extension", self.ext)
		settings_form.addRow(self._with_help("Datetime", "Adds timestamp to filenames."), self.use_datetime)
		settings_form.addRow("Datetime format", self.datetime_format)
		settings_form.addRow(self._with_help("Parallel", "Workers for batch."), self.parallel_workers)
		settings_form.addRow("Verbose", self.verbose)
		settings_form.addRow("Progress bars", self.progress)

		vbox.addWidget(settings_group)

		profile_group = QtWidgets.QGroupBox("Profiles")
		phbox = QtWidgets.QHBoxLayout(profile_group)
		self.profile_combo = QtWidgets.QComboBox()
		btn_load = QtWidgets.QPushButton("Load")
		btn_save = QtWidgets.QPushButton("Save as")
		btn_delete = QtWidgets.QPushButton("Delete")
		btn_load.clicked.connect(self._load_profile_clicked)
		btn_save.clicked.connect(self._save_profile_clicked)
		btn_delete.clicked.connect(self._delete_profile_clicked)
		phbox.addWidget(self.profile_combo)
		phbox.addWidget(btn_load)
		phbox.addWidget(btn_save)
		phbox.addWidget(btn_delete)
		vbox.addWidget(profile_group)

		batch_group = QtWidgets.QGroupBox("Batch queue")
		b_layout = QtWidgets.QVBoxLayout(batch_group)
		self.batch_table = QtWidgets.QTableWidget(0, 3)
		self.batch_table.setHorizontalHeaderLabels(["Video", "Overrides", "Edit"])
		self.batch_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
		self.batch_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
		self.batch_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
		self.batch_table.cellClicked.connect(self._batch_row_selected)
		btns_row = QtWidgets.QHBoxLayout()
		btn_add = QtWidgets.QPushButton("Add files")
		btn_add_folder = QtWidgets.QPushButton("Add folder")
		btn_clear = QtWidgets.QPushButton("Clear")
		btn_add.clicked.connect(self._batch_add_files)
		btn_add_folder.clicked.connect(self._batch_add_folder)
		btn_clear.clicked.connect(self._batch_clear)
		btns_row.addWidget(btn_add)
		btns_row.addWidget(btn_add_folder)
		btns_row.addWidget(btn_clear)
		b_layout.addLayout(btns_row)
		b_layout.addWidget(self.batch_table)
		vbox.addWidget(batch_group)

		actions_row = QtWidgets.QHBoxLayout()
		self.btn_preview = QtWidgets.QPushButton("Load preview")
		self.btn_run_single = QtWidgets.QPushButton("Run single")
		self.btn_run_batch = QtWidgets.QPushButton("Run batch (global)")
		self.btn_run_batch_over = QtWidgets.QPushButton("Run batch (with overrides)")
		actions_row.addWidget(self.btn_preview)
		actions_row.addWidget(self.btn_run_single)
		actions_row.addWidget(self.btn_run_batch)
		actions_row.addWidget(self.btn_run_batch_over)
		self.btn_preview.clicked.connect(self._load_preview)
		self.btn_run_single.clicked.connect(self._run_single)
		self.btn_run_batch.clicked.connect(lambda: self._run_batch(False))
		self.btn_run_batch_over.clicked.connect(lambda: self._run_batch(True))
		vbox.addLayout(actions_row)

		self.log = QtWidgets.QTextEdit()
		self.log.setReadOnly(True)
		self.log.setFixedHeight(160)
		vbox.addWidget(self.log)

		vbox.addStretch()
		return panel

	def _build_preview_panel(self) -> QtWidgets.QWidget:
		panel = QtWidgets.QWidget()
		vbox = QtWidgets.QVBoxLayout(panel)

		self.frame_view = FrameView()
		self.frame_view.crop_changed.connect(self._crop_from_view)
		vbox.addWidget(self.frame_view, 1)

		slider_row = QtWidgets.QHBoxLayout()
		self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.frame_slider.setEnabled(False)
		self.frame_slider.valueChanged.connect(self._slider_changed)
		self.lbl_frame_info = QtWidgets.QLabel("No video loaded")
		self.lbl_downscale = QtWidgets.QLabel("")
		slider_row.addWidget(self.frame_slider)
		slider_row.addWidget(self.lbl_frame_info)
		vbox.addLayout(slider_row)
		vbox.addWidget(self.lbl_downscale)

		crop_row = QtWidgets.QHBoxLayout()
		self.aspect_combo = QtWidgets.QComboBox()
		self.aspect_combo.addItems([
			"Free",
			"Original",
			"16:9",
			"16:10",
			"4:3",
			"1:1",
		])
		self.aspect_combo.currentIndexChanged.connect(self._aspect_lock_changed)
		self.btn_clear_crop = QtWidgets.QPushButton("Clear crop")
		self.btn_clear_crop.clicked.connect(lambda: self._set_crop(None))
		crop_row.addWidget(QtWidgets.QLabel("Aspect"))
		crop_row.addWidget(self.aspect_combo)
		crop_row.addWidget(self.btn_clear_crop)
		vbox.addLayout(crop_row)

		keyboard_hint = QtWidgets.QLabel("Shortcuts: Space play/pause, arrows move crop (if set), S save profile")
		vbox.addWidget(keyboard_hint)

		self.estimate_label = QtWidgets.QLabel("Estimates: n/a")
		self.resize_label = QtWidgets.QLabel("")
		vbox.addWidget(self.estimate_label)
		vbox.addWidget(self.resize_label)

		self.btn_play = QtWidgets.QPushButton("Play")
		self.btn_play.setCheckable(True)
		self.btn_play.toggled.connect(self._toggle_play)
		vbox.addWidget(self.btn_play, 0, QtCore.Qt.AlignLeft)

		return panel

	def _with_help(self, label: str, help_text: str) -> QtWidgets.QWidget:
		container = QtWidgets.QWidget()
		layout = QtWidgets.QHBoxLayout(container)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(QtWidgets.QLabel(label))
		btn = QtWidgets.QToolButton()
		btn.setText("?")
		btn.setAutoRaise(True)
		btn.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, label, help_text))
		layout.addWidget(btn)
		layout.addStretch()
		return container

	def _row_widget(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
		container = QtWidgets.QWidget()
		container.setLayout(layout)
		return container

	def _apply_dark_theme(self) -> None:
		app = QtWidgets.QApplication.instance()
		if not app:
			return
		palette = QtGui.QPalette()
		base = QtGui.QColor(30, 30, 30)
		text = QtGui.QColor(230, 230, 230)
		palette.setColor(QtGui.QPalette.Window, base)
		palette.setColor(QtGui.QPalette.WindowText, text)
		palette.setColor(QtGui.QPalette.Base, QtGui.QColor(20, 20, 20))
		palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
		palette.setColor(QtGui.QPalette.ToolTipBase, text)
		palette.setColor(QtGui.QPalette.ToolTipText, text)
		palette.setColor(QtGui.QPalette.Text, text)
		palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
		palette.setColor(QtGui.QPalette.ButtonText, text)
		palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64, 128, 255))
		palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
		app.setPalette(palette)

	def _apply_config_to_form(self) -> None:
		cfg = self._config
		self.input_path.setText(str(resolve_dir(cfg.get("input_dir", "input"))))
		self.output_dir.setText(str(resolve_dir(cfg.get("output_dir", "output"))))
		self.interval.setValue(int(cfg.get("interval", 1)))
		self.prefix.setText(cfg.get("prefix", "frame"))
		self.start_number.setValue(int(cfg.get("start_number", 0)))
		self.limit.setValue(int(cfg.get("limit", 0) or 0))
		self.crop.setText(
			",".join(str(x) for x in cfg.get("crop_area", [])) if cfg.get("crop_area") else ""
		)
		if cfg.get("resize_to"):
			self.resize_to.setText(
				f"{cfg['resize_to'].get('width')},{cfg['resize_to'].get('height')}"
			)
		else:
			self.resize_to.setText("")
		cs = cfg.get("color_space") or ""
		idx = self.color_space.findText(cs, QtCore.Qt.MatchFixedString)
		if idx >= 0:
			self.color_space.setCurrentIndex(idx)
		self.ext.setText(cfg.get("ext", "png"))
		self.use_datetime.setChecked(bool(cfg.get("use_datetime_token", False)))
		self.datetime_format.setText(cfg.get("datetime_format", "%Y%m%d_%H%M%S"))
		self.parallel_workers.setValue(int(cfg.get("parallel_workers", 2)))
		self.verbose.setChecked(bool(cfg.get("verbose", True)))
		self.progress.setChecked(bool(cfg.get("progress", True)))

	def _collect_form_config(self) -> Dict[str, Any]:
		cfg = {
			"input_dir": self.input_path.text().strip(),
			"output_dir": self.output_dir.text().strip(),
			"interval": self.interval.value(),
			"prefix": self.prefix.text().strip() or "frame",
			"start_number": self.start_number.value(),
			"limit": self.limit.value() or None,
			"crop_area": None,
			"resize_to": None,
			"color_space": self.color_space.currentText().strip() or None,
			"ext": self.ext.text().strip() or "png",
			"use_datetime_token": self.use_datetime.isChecked(),
			"datetime_format": self.datetime_format.text().strip() or "%Y%m%d_%H%M%S",
			"parallel_workers": self.parallel_workers.value(),
			"verbose": self.verbose.isChecked(),
			"progress": self.progress.isChecked(),
		}
		if self.crop.text().strip():
			parts = [int(x) for x in self.crop.text().split(",")]
			if len(parts) == 4:
				cfg["crop_area"] = parts
		if "," in self.resize_to.text():
			parts_raw = self.resize_to.text().split(",")
			if len(parts_raw) == 2:
				w_raw, h_raw = parts_raw[0].strip(), parts_raw[1].strip()
				w_val = int(w_raw) if w_raw else None
				h_val = int(h_raw) if h_raw else None
				if w_val or h_val:
					cfg["resize_to"] = {"width": w_val, "height": h_val}
		return cfg

	def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
		profiles: Dict[str, Dict[str, Any]] = {}
		if PROFILES_DIR.exists():
			for path in PROFILES_DIR.glob("*.yaml"):
				profiles[path.stem] = load_yaml(path, DEFAULT_CONFIG)
		return profiles

	def _update_profiles_ui(self) -> None:
		self.profile_combo.clear()
		self.profile_combo.addItems(sorted(self._profiles.keys()))

	def _load_profile_clicked(self) -> None:
		name = self.profile_combo.currentText()
		if not name:
			return
		cfg = self._profiles.get(name)
		if cfg:
			self._config = merge_overrides(DEFAULT_CONFIG, cfg)
			self._apply_config_to_form()
			self._log(f"Loaded profile {name}")

	def _save_profile_clicked(self) -> None:
		name, ok = QtWidgets.QInputDialog.getText(self, "Save profile", "Profile name")
		if not ok or not name.strip():
			return
		cfg = self._collect_form_config()
		path = PROFILES_DIR / f"{name.strip()}.yaml"
		save_yaml(path, cfg)
		self._profiles[name.strip()] = cfg
		self._update_profiles_ui()
		self._log(f"Saved profile {name}")

	def _delete_profile_clicked(self) -> None:
		name = self.profile_combo.currentText()
		if not name:
			return
		path = PROFILES_DIR / f"{name}.yaml"
		if path.exists():
			path.unlink()
		self._profiles.pop(name, None)
		self._update_profiles_ui()
		self._log(f"Deleted profile {name}")

	def _select_video(self) -> None:
		path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", self.input_path.text())
		if path:
			self.input_path.setText(path)
			self._load_preview()

	def _select_output(self) -> None:
		path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output", self.output_dir.text())
		if path:
			self.output_dir.setText(path)

	def _load_preview(self) -> None:
		path = Path(self.input_path.text())
		if not path.exists():
			QtWidgets.QMessageBox.warning(self, "No video", "Select a valid video file")
			return
		if not is_video_file(str(path)):
			QtWidgets.QMessageBox.warning(self, "Invalid", "Cannot open this video")
			return
		self._current_video = path
		if self._capture:
			self._capture.release()
		self._capture = cv2.VideoCapture(str(path))
		frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
		width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		duration = frame_count / fps if fps > 0 else None
		self._video_props = {
			"frame_count": frame_count,
			"fps": fps,
			"width": width,
			"height": height,
			"duration": duration,
		}
		self.frame_slider.setMaximum(max(0, frame_count - 1))
		self.frame_slider.setEnabled(True)
		self.frame_slider.setValue(0)
		self._frame_cache.clear()
		self._source_size = (width, height)
		self._display_size = (width, height)
		self._source_wh = (width, height)
		self._preview_downscaled = max(width, height) > 1080
		if self._preview_downscaled:
			self.lbl_downscale.setText("Preview downscaled to 1080p for performance")
		else:
			self.lbl_downscale.setText("")
		self._show_frame(0)
		opts = build_options(self._collect_form_config())
		self.estimate_label.setText(format_estimate(self._video_props, opts))
		self._update_resize_label()

	def _read_frame(self, index: int) -> Optional[Tuple[QtGui.QImage, Tuple[int, int], Tuple[int, int]]]:
		if not self._capture:
			return None
		if index in self._frame_cache:
			return self._frame_cache[index]
		self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
		ok, frame = self._capture.read()
		if not ok or frame is None:
			return None
		h, w, _ = frame.shape
		source_size = (w, h)
		frame_for_preview = frame
		resize_cfg = self._collect_form_config().get("resize_to") if self._respect_resize_preview else None
		preview_note = ""
		if resize_cfg:
			target = self._resolve_resize_target(source_size, resize_cfg)
			if target:
				frame_for_preview = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
				preview_note = f"Preview shows resize {target[0]}x{target[1]}"
				source_size = target
		else:
			scale = min(1080 / max(w, h), 1.0)
			if scale < 1.0:
				frame_for_preview = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
				preview_note = "Preview downscaled for display"
		frame_rgb = cv2.cvtColor(frame_for_preview, cv2.COLOR_BGR2RGB)
		h, w, _ = frame_rgb.shape
		display_size = (w, h)
		# Use explicit stride to avoid skewed rendering
		image = QtGui.QImage(frame_rgb.data, w, h, frame_rgb.strides[0], QtGui.QImage.Format_RGB888)
		image = image.copy()
		self._frame_cache[index] = (image, display_size, source_size)
		self._display_size = display_size
		self._source_size = source_size
		if preview_note:
			self.resize_label.setText(preview_note)
		return image, display_size, source_size

	def _resolve_resize_target(self, source_size: Tuple[int, int], resize_cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
		if not resize_cfg:
			return None
		sw, sh = source_size
		w = resize_cfg.get("width")
		h = resize_cfg.get("height")
		if w is not None and h is not None:
			return int(w), int(h)
		if w is not None:
			return int(w), int(round(int(w) * sh / sw)) if sw else int(w)
		if h is not None:
			return int(round(int(h) * sw / sh)) if sh else int(h), int(h)
		return None

	def _show_frame(self, index: int) -> None:
		result = self._read_frame(index)
		if result is None:
			return
		img, display_size, source_size = result
		self.frame_view.set_image(img, display_size, source_size)
		self.lbl_frame_info.setText(f"Frame {index}")
		self._sync_crop_to_view()
		self._update_resize_label()

	def _update_resize_label(self) -> None:
		cfg = self._collect_form_config()
		resize_cfg = cfg.get("resize_to")
		if resize_cfg:
			w = resize_cfg.get("width")
			h = resize_cfg.get("height")
			self.resize_label.setText(
				f"Preview shows resized output: {w or 'auto'}x{h or 'auto'} (source {self._source_wh[0]}x{self._source_wh[1]})"
			)
		else:
			self.resize_label.setText("")

	def _refresh_preview_resize(self) -> None:
		self._frame_cache.clear()
		self._update_resize_label()
		if self.frame_slider.isEnabled():
			self._show_frame(self.frame_slider.value())
		else:
			self._load_preview()

	def _slider_changed(self, value: int) -> None:
		self._show_frame(value)

	def _aspect_lock_changed(self) -> None:
		text = self.aspect_combo.currentText()
		ratio_map = {
			"Original": None,
			"16:9": 16 / 9,
			"16:10": 16 / 10,
			"4:3": 4 / 3,
			"1:1": 1.0,
		}
		ratio = ratio_map.get(text)
		if text == "Original" and self.frame_view._crop:
			cx, cy, cw, ch = self.frame_view._crop
			ratio = cw / ch if ch else None
		locked = text != "Free"
		self._lock_ratio = ratio
		self.frame_view.set_aspect_lock(locked, ratio)

	def _crop_from_view(self, crop: tuple) -> None:
		self._set_crop(crop, from_view=True)

	def _set_crop(self, crop: Optional[Tuple[int, int, int, int]], from_view: bool = False) -> None:
		self.frame_view.set_crop(crop)
		self.crop.blockSignals(True)
		self.crop.setText(",".join(str(x) for x in crop) if crop else "")
		self.crop.blockSignals(False)
		self._sync_crop_to_view()

	def _sync_crop_to_view(self) -> None:
		text = self.crop.text().strip()
		if not text:
			self.frame_view.set_crop(None)
			return
		parts = text.split(",")
		if len(parts) != 4:
			return
		try:
			x, y, w, h = [int(v) for v in parts]
		except ValueError:
			return
		if self._source_size[0] and self._source_size[1]:
			x = max(0, min(x, self._source_size[0] - 1))
			y = max(0, min(y, self._source_size[1] - 1))
			w = max(1, min(w, self._source_size[0] - x))
			h = max(1, min(h, self._source_size[1] - y))
		self.frame_view.set_crop((x, y, w, h))

	def _toggle_play(self, checked: bool) -> None:
		if checked:
			self._play_timer.start(1000 // max(1, int(self._video_props.get("fps") or 24)))
			self.btn_play.setText("Pause")
		else:
			self._play_timer.stop()
			self.btn_play.setText("Play")
		self._sync_crop_to_view()

	def _advance_frame(self) -> None:
		if not self.frame_slider.isEnabled():
			return
		next_frame = self.frame_slider.value() + 1
		if next_frame > self.frame_slider.maximum():
			next_frame = 0
		self.frame_slider.blockSignals(True)
		self.frame_slider.setValue(next_frame)
		self.frame_slider.blockSignals(False)
		self._show_frame(next_frame)

	def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
		if event.key() == QtCore.Qt.Key_Space:
			self.btn_play.toggle()
		elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
			if not self.frame_view._crop:
				return
			dx = -1 if event.key() == QtCore.Qt.Key_Left else 1
			cx, cy, cw, ch = self.frame_view._crop
			self._set_crop((max(0, cx + dx), cy, cw, ch))
		elif event.key() == QtCore.Qt.Key_S:
			self._save_profile_clicked()
		else:
			super().keyPressEvent(event)

	def _batch_add_files(self) -> None:
		paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select videos", self.input_path.text())
		if not paths:
			return
		for p in paths:
			self._add_batch_item(Path(p))

	def _batch_add_folder(self) -> None:
		folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", self.input_path.text())
		if not folder:
			return
		extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
		for path in Path(folder).iterdir():
			if path.is_file() and path.suffix.lower() in extensions:
				self._add_batch_item(path)

	def _add_batch_item(self, path: Path) -> None:
		row = self.batch_table.rowCount()
		self.batch_table.insertRow(row)
		self.batch_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(path)))
		self.batch_table.setItem(row, 1, QtWidgets.QTableWidgetItem("None"))
		btn = QtWidgets.QPushButton("Edit")
		btn.clicked.connect(lambda _=False, r=row: self._edit_override(r))
		self.batch_table.setCellWidget(row, 2, btn)
		self.batch_table.cellClicked.connect(self._batch_row_selected)

	def _batch_row_selected(self, row: int, col: int) -> None:
		path_item = self.batch_table.item(row, 0)
		if not path_item:
			return
		path = Path(path_item.text())
		if not path.exists():
			self._log(f"Batch item missing: {path}")
			return
		self.input_path.setText(str(path))
		try:
			self._load_preview()
		except Exception as exc:
			self._log(f"Failed to load preview for {path.name}: {exc}")

	def _edit_override(self, row: int) -> None:
		path_item = self.batch_table.item(row, 0)
		if not path_item:
			return
		existing = path_item.data(QtCore.Qt.UserRole) or {}
		dlg = OverrideDialog(self, existing)
		if dlg.exec() == QtWidgets.QDialog.Accepted:
			data = dlg.result_data()
			path_item.setData(QtCore.Qt.UserRole, data)
			self.batch_table.item(row, 1).setText("Set" if data else "None")

	def _batch_clear(self) -> None:
		self.batch_table.setRowCount(0)

	def _collect_batch_items(self) -> List[VideoItem]:
		items: List[VideoItem] = []
		for row in range(self.batch_table.rowCount()):
			path_item = self.batch_table.item(row, 0)
			if not path_item:
				continue
			path = Path(path_item.text())
			overrides = path_item.data(QtCore.Qt.UserRole) or {}
			items.append(VideoItem(path=path, overrides=overrides))
		return items

	def _run_single(self) -> None:
		cfg = self._collect_form_config()
		self._config = cfg
		save_yaml(CONFIG_PATH, cfg)
		video = Path(self.input_path.text())
		if not video.exists():
			QtWidgets.QMessageBox.warning(self, "Missing", "Select a video file")
			return
		output_base = Path(self.output_dir.text())
		output_base.mkdir(parents=True, exist_ok=True)
		tasks = [VideoItem(path=video)]
		self._start_worker(tasks, cfg, output_base)

	def _run_batch(self, use_overrides: bool) -> None:
		cfg = self._collect_form_config()
		self._config = cfg
		save_yaml(CONFIG_PATH, cfg)
		tasks = self._collect_batch_items()
		if not tasks:
			QtWidgets.QMessageBox.information(self, "Empty", "No videos in batch")
			return
		if not use_overrides:
			for t in tasks:
				t.overrides = {}
		output_base = Path(self.output_dir.text())
		output_base.mkdir(parents=True, exist_ok=True)
		self._start_worker(tasks, cfg, output_base)

	def _start_worker(self, tasks: List[VideoItem], cfg: Dict[str, Any], output_base: Path) -> None:
		if self._worker and self._worker.isRunning():
			QtWidgets.QMessageBox.information(self, "Busy", "A run is already in progress")
			return
		self._frame_cache.clear()
		self._worker = ExtractWorker(tasks, cfg, output_base)
		self._worker.progress.connect(lambda msg: self._log(msg))
		self._worker.finished.connect(self._on_finished)
		self._worker.start()
		self._log("Started processing...")

	def _on_finished(self, reports: List[Dict[str, Any]]) -> None:
		ok = sum(1 for r in reports if r.get("status") == "ok")
		errors = [r for r in reports if r.get("status") == "error"]
		cancelled = [r for r in reports if r.get("status") == "cancelled"]
		total_frames = sum(r.get("frames", 0) for r in reports)
		msg = f"Done. Videos: {len(reports)}, OK: {ok}, Frames: {total_frames}"
		if errors:
			msg += f", Errors: {len(errors)}"
		if cancelled:
			msg += f", Cancelled: {len(cancelled)}"
		self._log(msg)
		QtWidgets.QMessageBox.information(self, "Finished", msg)
		self._worker = None

	def _log(self, text: str) -> None:
		stamp = datetime.now().strftime("%H:%M:%S")
		self.log.append(f"[{stamp}] {text}")


def main() -> None:
	ensure_files()
	app = QtWidgets.QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
