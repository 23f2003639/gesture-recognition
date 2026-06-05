"""
camera.py — Shared camera utilities for the unified gesture recognition system.

Handles camera capture, frame preprocessing, and FPS tracking.
Used by data collection, training pipelines, and recognition loops.
"""

from collections import deque
from dataclasses import dataclass, field
import time

import cv2


_CLAHE = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))


@dataclass(frozen=True)
class CameraConfig:
    width:           int   = 640
    height:          int   = 480
    fps:             int   = 30
    # MediaPipe runs on a downscaled copy — keeps hand tracking fast
    tracking_width:  int   = 320
    tracking_height: int   = 240
    contrast_alpha:  float = 1.03
    contrast_beta:   int   = 8
    enable_clahe:    bool  = False   # Optional CLAHE for dim lighting


class FrameRateTracker:
    """Rolling-window FPS estimator."""

    def __init__(self, window_size: int = 30):
        self._timestamps: deque = deque(maxlen=window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def resize_to_fit(frame, max_width: int, max_height: int):
    """Resize frame to fit within bounds, preserving aspect ratio."""
    if max_width <= 0 or max_height <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale == 1.0:
        return frame
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    return cv2.resize(frame, (nw, nh), interpolation=interp)


def _configure_capture(cap: cv2.VideoCapture, config: CameraConfig) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS,          config.fps)
    # Buffer of 1 = always get the freshest frame, no queue buildup
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)


def open_camera(camera_index: int = 0, config: CameraConfig = CameraConfig()) -> cv2.VideoCapture:
    """Open camera with the best available backend."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            _configure_capture(cap, config)
            return cap
        cap.release()
    raise RuntimeError(f"Unable to access camera index {camera_index}.")


def prepare_frame(frame, config: CameraConfig = CameraConfig()):
    """
    Preprocess a raw camera frame:
      1. Mirror (flip) — makes the display feel like a mirror.
      2. Resize to configured display resolution.
      3. Optional contrast boost (subtle, prevents washed-out detection).
      4. Optional CLAHE for dim/uneven lighting conditions.
    """
    frame = cv2.flip(frame, 1)
    frame = resize_to_fit(frame, config.width, config.height)

    if config.contrast_alpha != 1.0 or config.contrast_beta != 0:
        frame = cv2.convertScaleAbs(frame, alpha=config.contrast_alpha, beta=config.contrast_beta)

    if config.enable_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return frame
