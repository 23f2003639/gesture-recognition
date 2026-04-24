from collections import deque
from dataclasses import dataclass
import time

import cv2

DEFAULT_CAMERA_SIZE = (640, 480)   # changed from 960x540 — matches data.py
_CLAHE = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))


@dataclass(frozen=True)
class CameraConfig:
    width: int = DEFAULT_CAMERA_SIZE[0]
    height: int = DEFAULT_CAMERA_SIZE[1]
    fps: int = 30
    tracking_width: int = 640
    tracking_height: int = 480     
    contrast_alpha: float = 1.03
    contrast_beta: int = 8
    enable_tone_mapping: bool = False


class FrameRateTracker:
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)

    def tick(self):
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / elapsed


def resize_to_fit(frame, max_width, max_height):
    if max_width <= 0 or max_height <= 0:
        return frame

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale <= 0:
        return frame

    new_width  = max(1, int(round(width  * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return frame

    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    return cv2.resize(frame, (new_width, new_height), interpolation=interpolation)


def _configure_capture(cap, config):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS,          config.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)


def open_camera(camera_index=0, config=CameraConfig()):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            _configure_capture(cap, config)
            return cap
        cap.release()

    raise RuntimeError(f"Unable to access camera index {camera_index}.")


def prepare_frame(frame, config=CameraConfig()):
    frame = cv2.flip(frame, 1)
    frame = resize_to_fit(frame, config.width, config.height)

    if config.contrast_alpha != 1.0 or config.contrast_beta != 0:
        frame = cv2.convertScaleAbs(
            frame, alpha=config.contrast_alpha, beta=config.contrast_beta
        )
    if config.enable_tone_mapping:
        frame = _apply_tone(frame)
    return frame


def _apply_tone(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _CLAHE.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
