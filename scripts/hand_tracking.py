from pathlib import Path
import threading
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent
HAND_LANDMARKER_MODEL_FILE = BASE_DIR / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Hand connections for manual drawing (no landmark_pb2 needed)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ─────────────────────────────────────────
#  MODEL DOWNLOAD
# ─────────────────────────────────────────
def ensure_hand_landmarker_model(model_path=HAND_LANDMARKER_MODEL_FILE):
    if model_path.exists():
        return model_path

    print("Downloading hand_landmarker.task (~25 MB) ...")
    temp_path = model_path.with_suffix(model_path.suffix + ".download")
    try:
        urlretrieve(HAND_LANDMARKER_MODEL_URL, temp_path)
        temp_path.replace(model_path)
        print("Download complete.")
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to download the MediaPipe hand landmarker model. "
            "Check your internet connection and try again."
        ) from exc

    return model_path


# ─────────────────────────────────────────
#  FRAME HELPER
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
    num_hands=2,
    min_detection_confidence=0.6,
    min_presence_confidence=0.6,
    min_tracking_confidence=0.6,
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
#  — flips Left/Right to match mirrored frame
# ─────────────────────────────────────────
def _extract_hand_data(result) -> dict:
    hand_entries = []
    handedness_items = result.handedness or []

    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = handedness_items[index] if index < len(handedness_items) else []
        raw_label  = handedness[0].category_name if handedness else None
        score      = handedness[0].score         if handedness else 0.0

        # flip label to match cv2.flip(frame, 1) mirror in prepare_frame()
        if raw_label == "Left":
            label = "Right"
        elif raw_label == "Right":
            label = "Left"
        else:
            label = raw_label

        landmark_values = []
        for landmark in landmarks:
            landmark_values.extend([landmark.x, landmark.y, landmark.z])

        hand_entries.append({
            "label":           label,
            "score":           score,
            "wrist_x":         landmarks[0].x if landmarks else 0.5,
            "landmark_values": landmark_values,
        })

    if not hand_entries:
        return {}

    hand_data = {}
    remaining = list(hand_entries)

    for preferred_label in ("Left", "Right"):
        matches = [item for item in remaining if item["label"] == preferred_label]
        if not matches:
            continue
        selected = max(matches, key=lambda item: item["score"])
        hand_data[preferred_label] = selected["landmark_values"]
        remaining.remove(selected)

    # fallback — assign by wrist position
    remaining.sort(key=lambda item: item["wrist_x"])
    if "Left" not in hand_data and remaining:
        hand_data["Left"]  = remaining.pop(0)["landmark_values"]
    if "Right" not in hand_data and remaining:
        hand_data["Right"] = remaining.pop(0)["landmark_values"]  # fixed: was pop(-1)

    return hand_data


# ─────────────────────────────────────────
#  SYNCHRONOUS DETECT  (used by data.py)
# ─────────────────────────────────────────
def detect_hands(landmarker, frame, timestamp_ms, tracking_size=None):
    mp_image = _frame_to_mp_image(frame, tracking_size=tracking_size)
    result   = landmarker.detect_for_video(mp_image, timestamp_ms)
    return result, _extract_hand_data(result)


# ─────────────────────────────────────────
#  ASYNC TRACKER  (used by model.py)
# ─────────────────────────────────────────
class AsyncHandTracker:
    def __init__(
        self,
        num_hands=2,
        min_detection_confidence=0.6,
        min_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        tracking_size=(640, 480),   # updated default to match camera.py
    ):
        self.tracking_size = tracking_size
        self._lock                    = threading.Lock()
        self._latest_result           = None
        self._latest_hand_data        = {}
        self._latest_timestamp_ms     = -1
        self._submitted_timestamp_ms  = -1
        self._inference_pending       = False
        self._closed                  = False

        self._landmarker = create_hand_landmarker(
            num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result,
        )

    def _on_result(self, result, _output_image, timestamp_ms):
        hand_data = _extract_hand_data(result)
        with self._lock:
            if self._closed:
                return
            self._inference_pending       = False
            self._latest_result           = result
            self._latest_hand_data        = hand_data
            self._latest_timestamp_ms     = timestamp_ms

    def submit(self, frame, timestamp_ms):
        with self._lock:
            if self._closed or self._inference_pending:
                return False
            timestamp_ms = max(timestamp_ms, self._submitted_timestamp_ms + 1)
            self._submitted_timestamp_ms = timestamp_ms
            self._inference_pending      = True

        mp_image = _frame_to_mp_image(frame, tracking_size=self.tracking_size)
        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception:
            with self._lock:
                self._inference_pending = False
            raise
        return True

    def latest(self, now_timestamp_ms=None, max_age_ms=300):  # tightened from 500→300ms
        with self._lock:
            result       = self._latest_result
            hand_data    = dict(self._latest_hand_data)
            timestamp_ms = self._latest_timestamp_ms

        if result is None:
            return None, {}, -1

        if now_timestamp_ms is not None and timestamp_ms >= 0:
            if now_timestamp_ms - timestamp_ms > max_age_ms:
                return None, {}, timestamp_ms

        return result, hand_data, timestamp_ms

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._landmarker.close()


# ─────────────────────────────────────────
#  DRAW LANDMARKS  (no landmark_pb2 needed)
# ─────────────────────────────────────────
def draw_hand_landmarks(frame, detection_result):
    from ui import ACCENT, TEXT_MUTED

    h, w = frame.shape[:2]

    for hand_landmarks in detection_result.hand_landmarks:
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], TEXT_MUTED, 1, cv2.LINE_AA)
        for pt in points:
            cv2.circle(frame, pt, 2, ACCENT, -1, cv2.LINE_AA)
