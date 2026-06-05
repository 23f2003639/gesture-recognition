"""
hand_tracking.py — MediaPipe hand landmarker wrapper for the unified gesture system.

Provides:
  - Model auto-download
  - Synchronous detect (used during data collection)
  - AsyncHandTracker with strict backpressure (used during recognition)
  - Hand landmark drawing
"""

from pathlib import Path
import threading
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
HAND_LANDMARKER_FILE = BASE_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ─────────────────────────────────────────
#  MODEL DOWNLOAD
# ─────────────────────────────────────────

def ensure_hand_landmarker_model(model_path: Path = HAND_LANDMARKER_FILE) -> Path:
    if model_path.exists():
        return model_path
    print("Downloading hand_landmarker.task (~25 MB)…")
    tmp = model_path.with_suffix(model_path.suffix + ".download")
    try:
        urlretrieve(HAND_LANDMARKER_URL, tmp)
        tmp.replace(model_path)
        print("Download complete.")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to download MediaPipe hand landmarker model. "
            "Check your internet connection and try again."
        ) from exc
    return model_path


# ─────────────────────────────────────────
#  FRAME → MP IMAGE
# ─────────────────────────────────────────

def _frame_to_mp_image(frame, tracking_size=None):
    from camera import resize_to_fit
    if tracking_size is not None:
        frame = resize_to_fit(frame, tracking_size[0], tracking_size[1])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ─────────────────────────────────────────
#  LANDMARKER FACTORY
# ─────────────────────────────────────────

def create_hand_landmarker(
    num_hands: int = 2,
    min_detection_confidence: float = 0.6,
    min_presence_confidence: float = 0.6,
    min_tracking_confidence: float = 0.6,
    running_mode=None,
    result_callback=None,
):
    model_path = ensure_hand_landmarker_model()
    if running_mode is None:
        running_mode = mp.tasks.vision.RunningMode.VIDEO

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=result_callback,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────
#  EXTRACT HAND DATA
#  Mirrors Left/Right to match cv2.flip(frame, 1)
# ─────────────────────────────────────────

def _extract_hand_data(result) -> dict:
    """Return {"Left": [63 floats], "Right": [63 floats]} from a landmarker result."""
    entries = []
    for idx, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[idx] if idx < len(result.handedness) else []
        raw_label  = handedness[0].category_name if handedness else None
        score      = handedness[0].score         if handedness else 0.0

        # Flip because the frame is mirrored
        if raw_label == "Left":
            label = "Right"
        elif raw_label == "Right":
            label = "Left"
        else:
            label = raw_label

        values = []
        for lm in landmarks:
            values.extend([lm.x, lm.y, lm.z])

        entries.append({
            "label":   label,
            "score":   score,
            "wrist_x": landmarks[0].x if landmarks else 0.5,
            "values":  values,
        })

    if not entries:
        return {}

    hand_data: dict = {}
    remaining = list(entries)

    for preferred in ("Left", "Right"):
        matches = [e for e in remaining if e["label"] == preferred]
        if matches:
            best = max(matches, key=lambda e: e["score"])
            hand_data[preferred] = best["values"]
            remaining.remove(best)

    # Fallback: assign remaining hands by wrist x-position
    remaining.sort(key=lambda e: e["wrist_x"])
    if "Left" not in hand_data and remaining:
        hand_data["Left"] = remaining.pop(0)["values"]
    if "Right" not in hand_data and remaining:
        hand_data["Right"] = remaining.pop(0)["values"]

    return hand_data


# ─────────────────────────────────────────
#  SYNCHRONOUS DETECT  (data collection)
# ─────────────────────────────────────────

def detect_hands(landmarker, frame, timestamp_ms: int, tracking_size=None):
    mp_image = _frame_to_mp_image(frame, tracking_size=tracking_size)
    result   = landmarker.detect_for_video(mp_image, timestamp_ms)
    return result, _extract_hand_data(result)


# ─────────────────────────────────────────
#  ASYNC TRACKER  (recognition — zero latency)
#
#  Key design decisions:
#   • LIVE_STREAM mode: MediaPipe calls our callback from its own thread.
#   • Strict backpressure: only ONE inference in-flight at a time.
#     If a new frame arrives before the previous result is ready, we
#     DROP that frame rather than queue it. This keeps the latency at
#     "one-inference delay" instead of growing unboundedly.
#   • max_age_ms: stale results are discarded so we never show outdated
#     detections — the UI falls back to "no hand" gracefully.
# ─────────────────────────────────────────

class AsyncHandTracker:
    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        tracking_size: tuple = (320, 240),
    ):
        self.tracking_size = tracking_size
        self._lock                   = threading.Lock()
        self._latest_result          = None
        self._latest_hand_data: dict = {}
        self._latest_ts_ms: int      = -1
        self._submitted_ts_ms: int   = -1
        self._inference_pending      = False
        self._closed                 = False

        self._landmarker = create_hand_landmarker(
            num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
        )

    def _on_result(self, result, _output_image, timestamp_ms: int) -> None:
        hand_data = _extract_hand_data(result)
        with self._lock:
            if self._closed:
                return
            self._inference_pending  = False
            self._latest_result      = result
            self._latest_hand_data   = hand_data
            self._latest_ts_ms       = timestamp_ms

    def submit(self, frame, timestamp_ms: int) -> bool:
        """
        Submit frame for async inference.
        Returns False (drop frame) if an inference is already in-flight.
        This is the core of zero-latency operation.
        """
        with self._lock:
            if self._closed or self._inference_pending:
                return False
            timestamp_ms = max(timestamp_ms, self._submitted_ts_ms + 1)
            self._submitted_ts_ms  = timestamp_ms
            self._inference_pending = True

        mp_image = _frame_to_mp_image(frame, tracking_size=self.tracking_size)
        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception:
            with self._lock:
                self._inference_pending = False
            raise
        return True

    def latest(self, now_timestamp_ms: int = None, max_age_ms: int = 200):
        """
        Get the most recent detection.
        Returns (None, {}, -1) if no result yet or result is too stale.
        max_age_ms of 200ms is a sweet spot: tolerates ~6 skipped frames
        at 30fps without showing ghost detections.
        """
        with self._lock:
            result    = self._latest_result
            hand_data = dict(self._latest_hand_data)
            ts_ms     = self._latest_ts_ms

        if result is None:
            return None, {}, -1

        if now_timestamp_ms is not None and ts_ms >= 0:
            if now_timestamp_ms - ts_ms > max_age_ms:
                return None, {}, ts_ms

        return result, hand_data, ts_ms

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._landmarker.close()


# ─────────────────────────────────────────
#  DRAW LANDMARKS
# ─────────────────────────────────────────

def draw_hand_landmarks(frame, detection_result) -> None:
    h, w = frame.shape[:2]
    for hand_landmarks in detection_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (145, 153, 163), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 3, (225, 230, 236), -1, cv2.LINE_AA)
