"""
Video to Frame Extraction Module

This module provides functionality to extract frames from video files
at specified intervals and save them as image files.

Functions:
- extract_frames(video_path, output_folder, interval): Extracts frames from the given video file at the specified interval (in frames) and saves them to the output folder.
- crop_frame(frame, crop_area): Crops the given frame to the specified area (x, y, width, height).
- resize_frame(frame, size): Resizes a frame to the given (width, height).
- color_convert_frame(frame, color_space): Converts the given frame to the specified color space (e.g., 'GRAY', 'HSV').
- save_frame(output_folder, prefix, ext, frame, sequence_number, time_tag): Saves a frame with a consistent naming scheme.

Additional Helper Functions:
- create_output_folder(folder_path): Creates the output folder if it does not exist.
- is_video_file(file_path): Checks if the given file path points to a valid video file.
- get_video_properties(video_path): Retrieves properties of the video such as frame count, frame rate, and resolution.
"""

from dataclasses import dataclass
import os
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "FrameExtractionOptions",
    "create_output_folder",
    "is_video_file",
    "get_video_properties",
    "crop_frame",
    "resize_frame",
    "color_convert_frame",
    "save_frame",
    "extract_frames",
]


@dataclass
class FrameExtractionOptions:
    """Configuration for frame extraction.

    interval: Extract every Nth frame (must be >= 1).
    prefix: Filename prefix used for saved frames.
    start_number: Starting index for sequential filenames (must be >= 0).
    limit: Optional maximum number of frames to write.
    crop_area: Optional crop rectangle as (x, y, width, height).
    resize_to: Optional target resolution as (width, height).
    color_space: Optional color conversion target (GRAY, HSV, RGB, BGR).
    ext: Image file extension to save (e.g., png, jpg).
    use_datetime_token: If True, include a timestamp token in saved names.
    datetime_format: Optional datetime format string for the token.
    """

    interval: int = 1
    prefix: str = "frame"
    start_number: int = 0
    limit: Optional[int] = None
    crop_area: Optional[Tuple[int, int, int, int]] = None
    resize_to: Optional[Tuple[int, int]] = None
    color_space: Optional[str] = None
    ext: str = "png"
    use_datetime_token: bool = False
    datetime_format: str = "%Y%m%d_%H%M%S"

    def validated(self) -> "FrameExtractionOptions":
        if self.interval < 1:
            raise ValueError("interval must be >= 1")
        if self.start_number < 0:
            raise ValueError("start_number must be >= 0")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be None or >= 1")
        if self.crop_area is not None:
            x, y, w, h = self.crop_area
            if min(w, h) <= 0:
                raise ValueError("crop_area width and height must be > 0")
            if min(x, y) < 0:
                raise ValueError("crop_area coordinates must be >= 0")
        if self.resize_to is not None:
            w, h = self.resize_to
            if min(w, h) <= 0:
                raise ValueError("resize_to width and height must be > 0")
        if self.color_space is not None:
            _ = _normalize_color_space(self.color_space)
        ext = self.ext.lstrip(".").lower()
        if not ext:
            raise ValueError("ext must be a valid image extension")
        dt_format = self.datetime_format or "%Y%m%d_%H%M%S"
        return FrameExtractionOptions(
            interval=self.interval,
            prefix=self.prefix,
            start_number=self.start_number,
            limit=self.limit,
            crop_area=self.crop_area,
            resize_to=self.resize_to,
            color_space=self.color_space,
            ext=ext,
            use_datetime_token=self.use_datetime_token,
            datetime_format=dt_format,
        )


def create_output_folder(folder_path: str) -> str:
    """Ensure output folder exists and return its path."""

    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def is_video_file(file_path: str) -> bool:
    """Return True if file exists and can be opened as a video."""

    if not os.path.isfile(file_path):
        return False
    capture = cv2.VideoCapture(file_path)
    try:
        return capture.isOpened()
    finally:
        capture.release()


def get_video_properties(video_path: str) -> Dict[str, float]:
    """Retrieve basic video properties: frame_count, fps, width, height, duration."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else None

    capture.release()
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def crop_frame(frame: np.ndarray, crop_area: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop frame using (x, y, width, height). Raises if area is out of bounds."""

    x, y, w, h = crop_area
    if w <= 0 or h <= 0:
        raise ValueError("crop_area width and height must be > 0")
    if x < 0 or y < 0:
        raise ValueError("crop_area coordinates must be >= 0")

    max_y, max_x = frame.shape[0], frame.shape[1]
    if x + w > max_x or y + h > max_y:
        raise ValueError("crop_area exceeds frame dimensions")
    return frame[y : y + h, x : x + w]


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to (width, height)."""

    width, height = size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _normalize_color_space(color_space: str) -> str:
    return color_space.strip().upper()


def color_convert_frame(frame: np.ndarray, color_space: str) -> np.ndarray:
    """Convert frame to target color space (supports GRAY, HSV, RGB, BGR)."""

    target = _normalize_color_space(color_space)
    if target == "BGR":
        return frame
    if target == "GRAY":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if target == "HSV":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if target == "RGB":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported color space: {color_space}")


def save_frame(
    output_folder: str,
    prefix: str,
    ext: str,
    frame: np.ndarray,
    sequence_number: int,
    time_tag: Optional[str] = None,
) -> str:
    """Save a single frame to disk using a consistent naming scheme.

    Returns the written filename. Raises IOError if writing fails.
    """

    parts = [prefix]
    if time_tag:
        parts.append(time_tag)
    parts.append(f"{sequence_number:06d}")
    filename = os.path.join(output_folder, f"{'_'.join(parts)}.{ext}")
    if not cv2.imwrite(filename, frame):
        raise IOError(f"Failed to write frame to {filename}")
    return filename


def extract_frames(
    video_path: str,
    output_folder: str,
    interval: int = 1,
    *,
    crop_area: Optional[Tuple[int, int, int, int]] = None,
    color_space: Optional[str] = None,
    limit: Optional[int] = None,
    prefix: str = "frame",
    start_number: int = 0,
    resize_to: Optional[Tuple[int, int]] = None,
    ext: str = "png",
    use_datetime_token: bool = False,
    datetime_format: str = "%Y%m%d_%H%M%S",
    on_frame_saved: Optional[Callable[[int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Extract frames from video and save to folder.

    Returns the number of frames written. Raises ValueError for invalid inputs.
    """

    options = FrameExtractionOptions(
        interval=interval,
        prefix=prefix,
        start_number=start_number,
        limit=limit,
        crop_area=crop_area,
        resize_to=resize_to,
        color_space=color_space,
        ext=ext,
        use_datetime_token=use_datetime_token,
        datetime_format=datetime_format,
    ).validated()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    create_output_folder(output_folder)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    time_tag = (
        datetime.now().strftime(options.datetime_format)
        if options.use_datetime_token
        else None
    )

    frame_index = 0
    saved = 0
    success, frame = capture.read()
    while success:
        if should_stop is not None and should_stop():
            break
        if frame_index % options.interval == 0:
            processed = frame
            # Resize first so crop coordinates refer to resized frame
            if options.resize_to is not None:
                processed = resize_frame(processed, options.resize_to)
            if options.crop_area is not None:
                processed = crop_frame(processed, options.crop_area)
            if options.color_space is not None:
                processed = color_convert_frame(processed, options.color_space)

            sequence_number = options.start_number + saved
            try:
                save_frame(
                    output_folder,
                    options.prefix,
                    options.ext,
                    processed,
                    sequence_number,
                    time_tag,
                )
            except Exception:
                capture.release()
                raise
            saved += 1
            if on_frame_saved is not None:
                on_frame_saved(saved)
            if options.limit is not None and saved >= options.limit:
                break

        frame_index += 1
        success, frame = capture.read()

    capture.release()
    return saved