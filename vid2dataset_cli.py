"""Interactive CLI for vid2frame extraction.

Lightweight, menu-driven interface with YAML settings and presets.
Supports single or batch processing, optional parallelism, verbose logs,
and progress feedback. Settings and presets live under ./config.
"""

import shutil
import sys
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
	import yaml
except ImportError:  # pragma: no cover - dependency notice
	print("PyYAML is required: pip install pyyaml", file=sys.stderr)
	sys.exit(1)

from vid2frame import (
	FrameExtractionOptions,
	extract_frames,
	get_video_properties,
	is_video_file,
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
PRESETS_PATH = CONFIG_DIR / "presets.yaml"

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

DEFAULT_PRESETS: List[Dict[str, Any]] = [
	{
		"name": "every_30_resize_720p",
		"description": "Extract every 30th frame resized to 1280x720",
		"interval": 30,
		"resize_to": {"width": 1280, "height": 720},
	},
	{
		"name": "every_60_gray",
		"description": "Every 60th frame, grayscale",
		"interval": 60,
		"color_space": "GRAY",
	},
	{
		"name": "dense_1080p",
		"description": "Every frame resized to 1920x1080",
		"interval": 1,
		"resize_to": {"width": 1920, "height": 1080},
	},
	{
		"name": "preview_fast",
		"description": "Quick preview: every 120th frame, limit 50",
		"interval": 120,
		"limit": 50,
	},
]


def ensure_files() -> None:
	CONFIG_DIR.mkdir(parents=True, exist_ok=True)
	if not CONFIG_PATH.exists():
		save_yaml(CONFIG_PATH, DEFAULT_CONFIG)
	if not PRESETS_PATH.exists():
		save_yaml(PRESETS_PATH, {"presets": DEFAULT_PRESETS})


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


def delete_output_folder(config: Dict[str, Any]) -> None:
	output_dir = resolve_dir(config.get("output_dir", "output"))
	if not output_dir.exists():
		print(f"Output folder does not exist: {output_dir}")
		return
	if output_dir == Path("/") or len(output_dir.parts) <= 1:
		print("Refusing to delete root path.")
		return
	confirm = prompt_bool(f"Delete output folder and all contents? {output_dir}", False)
	if not confirm:
		print("Deletion cancelled.")
		return
	try:
		shutil.rmtree(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		print(f"Deleted output folder and recreated: {output_dir}")
	except Exception as exc:
		print(f"Failed to delete output folder: {exc}")


def print_menu() -> None:
	print("\nvid2frame CLI")
	print("1) Single video")
	print("2) Batch folder")
	print("3) Edit settings")
	print("4) Apply preset")
	print("5) Delete output folder")
	print("6) Exit")


def prompt(text: str, default: Optional[str] = None) -> str:
	suffix = f" [{default}]" if default is not None else ""
	return input(f"{text}{suffix}: ").strip() or (default or "")


def prompt_bool(text: str, default: bool) -> bool:
	val = prompt(text, "y" if default else "n").lower()
	return val.startswith("y")


def prompt_int(text: str, default: Optional[int], allow_none: bool = True) -> Optional[int]:
	raw = prompt(text, str(default) if default is not None else "")
	if raw == "" and allow_none:
		return None
	try:
		return int(raw)
	except ValueError:
		print("Please enter a valid integer.")
		return prompt_int(text, default, allow_none)


def edit_settings(config: Dict[str, Any]) -> Dict[str, Any]:
	print("\nEdit settings (enter to keep current)")
	config["interval"] = prompt_int("Interval (every Nth frame)", config["interval"], allow_none=False)
	config["prefix"] = prompt("Filename prefix", config["prefix"])
	config["start_number"] = prompt_int("Start number", config["start_number"], allow_none=False)
	config["limit"] = prompt_int("Limit frames (blank for none)", config.get("limit"), allow_none=True)

	crop_raw = prompt("Crop area x,y,w,h (blank to skip)", "" if config.get("crop_area") is None else ",".join(map(str, config["crop_area"])))
	config["crop_area"] = (
		tuple(int(x) for x in crop_raw.split(",")) if crop_raw else None
	)

	resize_raw = prompt(
		"Resize width,height (blank to skip)",
		"" if config.get("resize_to") is None else ",".join(
			[str(config["resize_to"].get("width")), str(config["resize_to"].get("height"))]
		),
	)
	if resize_raw:
		try:
			w, h = (int(part) for part in resize_raw.split(","))
			config["resize_to"] = {"width": w, "height": h}
		except ValueError:
			print("Invalid resize pair; keeping previous value.")
	else:
		config["resize_to"] = None

	color_space = prompt("Color space (BGR/GRAY/HSV/RGB)", config.get("color_space") or "")
	config["color_space"] = color_space.upper() if color_space else None
	config["ext"] = prompt("Image extension (png/jpg/...)", config["ext"])
	config["use_datetime_token"] = prompt_bool(
		"Include datetime token in filenames?", config.get("use_datetime_token", False)
	)
	config["datetime_format"] = prompt(
		"Datetime format (strftime)", config.get("datetime_format", "%Y%m%d_%H%M%S")
	)
	config["parallel_workers"] = prompt_int(
		"Parallel workers (>=1)", config.get("parallel_workers", 2), allow_none=False
	)
	config["verbose"] = prompt_bool("Verbose logging?", config.get("verbose", True))
	config["progress"] = prompt_bool("Show progress bars?", config.get("progress", True))

	input_dir = prompt("Default input dir", config.get("input_dir", "input"))
	output_dir = prompt("Default output dir", config.get("output_dir", "output"))
	config["input_dir"] = input_dir
	config["output_dir"] = output_dir
	return config


def select_preset(presets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
	if not presets:
		print("No presets available.")
		return None
	print("\nPresets:")
	for idx, preset in enumerate(presets, start=1):
		print(f"{idx}) {preset.get('name')} - {preset.get('description','')}")
	choice = prompt_int("Select preset number", None, allow_none=True)
	if choice is None or choice < 1 or choice > len(presets):
		return None
	return presets[choice - 1]


def apply_preset(config: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
	merged = {**config}
	for key, value in preset.items():
		if key in {"name", "description"}:
			continue
		merged[key] = value
	return merged


def collect_videos_from_folder(folder: Path) -> List[Path]:
	extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
	files: List[Path] = []
	for path in folder.iterdir():
		if path.is_file() and path.suffix.lower() in extensions:
			files.append(path)
	return files


def build_options(config: Dict[str, Any]) -> Dict[str, Any]:
	resize_to = None
	if config.get("resize_to"):
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


def make_progress_callback(total: Optional[int]):
	if not total or total <= 0:
		return None

	bar_width = 24

	def _callback(done: int) -> None:
		filled = int(bar_width * done / total)
		bar = "=" * filled + "." * (bar_width - filled)
		print(f"[{bar}] {done}/{total}", end="\r", flush=True)

	return _callback


def process_video(
	video_path: Path,
	output_dir: Path,
	config: Dict[str, Any],
	cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
	report: Dict[str, Any] = {"video": str(video_path), "status": "ok", "frames": 0}
	if cancel_event and cancel_event.is_set():
		report["status"] = "cancelled"
		return report
	if not is_video_file(str(video_path)):
		report["status"] = "skipped"
		report["error"] = "Unreadable video"
		return report

	options = build_options(config)
	try:
		props = get_video_properties(str(video_path))
		expected = props.get("frame_count", 0) // max(1, options["interval"])
		if options.get("limit"):
			expected = min(expected, options["limit"])
	except Exception:
		expected = None

	progress_cb = make_progress_callback(expected) if config.get("progress", True) else None
	should_stop = cancel_event.is_set if cancel_event else None

	if config.get("verbose", True):
		print(f"Processing {video_path.name} -> {output_dir}")

	try:
		frames_written = extract_frames(
			str(video_path),
			str(output_dir),
			**options,
			on_frame_saved=progress_cb,
			should_stop=should_stop,
		)
		if progress_cb:
			print()  # newline after progress bar
		report["frames"] = frames_written
		if cancel_event and cancel_event.is_set():
			report["status"] = "cancelled"
	except Exception as exc:
		report["status"] = "error"
		report["error"] = str(exc)
	return report


def run_single(config: Dict[str, Any]) -> None:
	video_input = prompt("Video file path", str(resolve_dir(config["input_dir"])))
	output_base = prompt("Output directory", str(resolve_dir(config["output_dir"])))
	run_tag = datetime.now().strftime(config.get("datetime_format", "%Y%m%d_%H%M%S"))
	run_output = Path(resolve_dir(output_base)) / f"{config.get('prefix', 'frame')}_{run_tag}"
	run_output.mkdir(parents=True, exist_ok=True)
	cancel_event = threading.Event()
	try:
		report = process_video(Path(video_input), run_output, config, cancel_event)
	except KeyboardInterrupt:
		cancel_event.set()
		print("\nSingle run cancelled; returning to menu.")
		return
	summarize_reports([report])


def run_batch(config: Dict[str, Any]) -> None:
	folder = prompt("Folder with videos", str(resolve_dir(config["input_dir"])))
	input_dir = Path(folder)
	if not input_dir.exists():
		print("Folder does not exist.")
		return
	videos = collect_videos_from_folder(input_dir)
	if not videos:
		print("No video files found.")
		return
	output_base = prompt("Output directory", str(resolve_dir(config["output_dir"])))
	run_tag = datetime.now().strftime(config.get("datetime_format", "%Y%m%d_%H%M%S"))
	output_dir = Path(resolve_dir(output_base)) / f"{config.get('prefix', 'frame')}_{run_tag}"
	output_dir.mkdir(parents=True, exist_ok=True)

	workers = max(1, int(config.get("parallel_workers", 1)))
	cancel_event = threading.Event()
	reports: List[Dict[str, Any]] = []
	futures: Dict[Any, Path] = {}
	done: set = set()
	executor: Optional[ThreadPoolExecutor] = None
	try:
		executor = ThreadPoolExecutor(max_workers=workers)
		futures = {executor.submit(process_video, video, output_dir, config, cancel_event): video for video in videos}
		for future in as_completed(futures):
			done.add(future)
			reports.append(future.result())
	except KeyboardInterrupt:
		cancel_event.set()
		print("\nBatch cancelled; stopping workers...")
	finally:
		if executor is not None:
			executor.shutdown(wait=False, cancel_futures=True)
		for future, video in futures.items():
			if future in done:
				continue
			if future.done():
				try:
					reports.append(future.result())
				except CancelledError:
					reports.append({"video": str(video), "status": "cancelled", "frames": 0})
				except Exception as exc:
					reports.append({"video": str(video), "status": "error", "error": str(exc), "frames": 0})
			else:
				reports.append({"video": str(video), "status": "cancelled", "frames": 0})
	summarize_reports(reports)


def summarize_reports(reports: List[Dict[str, Any]]) -> None:
	errors = [r for r in reports if r.get("status") == "error"]
	skipped = [r for r in reports if r.get("status") == "skipped"]
	cancelled = [r for r in reports if r.get("status") == "cancelled"]
	total_frames = sum(r.get("frames", 0) for r in reports)
	print("\nRun summary:")
	print(f"  Videos processed: {len(reports)}")
	print(f"  Frames written: {total_frames}")
	if skipped:
		print(f"  Skipped: {len(skipped)}")
	if cancelled:
		print(f"  Cancelled: {len(cancelled)}")
	if errors:
		print("  Errors:")
		for err in errors:
			print(f"    {err['video']}: {err.get('error')}")


def main() -> None:
	ensure_files()
	config = load_yaml(CONFIG_PATH, DEFAULT_CONFIG)
	presets_data = load_yaml(PRESETS_PATH, {"presets": DEFAULT_PRESETS})
	presets = presets_data.get("presets", [])

	while True:
		print_menu()
		choice = prompt("Choose an option", "1")
		if choice == "1":
			run_single(config)
		elif choice == "2":
			run_batch(config)
		elif choice == "3":
			config = edit_settings(config)
			save_yaml(CONFIG_PATH, config)
			print("Settings saved.")
		elif choice == "4":
			preset = select_preset(presets)
			if preset:
				config = apply_preset(config, preset)
				save_yaml(CONFIG_PATH, config)
				print(f"Applied preset: {preset.get('name')}")
		elif choice == "5":
			delete_output_folder(config)
		elif choice == "6":
			print("Goodbye.")
			break
		else:
			print("Invalid choice.")


if __name__ == "__main__":
	main()
