from collections import deque
from pathlib import Path
import pickle
import time

import cv2
import numpy as np

from camera import CameraConfig, FrameRateTracker, open_camera, prepare_frame
from hand_tracking import AsyncHandTracker, draw_hand_landmarks
from ui import OverlayState, draw_recognition_overlay

BASE_DIR  = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "gesture_model.pkl"

CONFIDENCE_THRESHOLD          = 0.70
MAX_CONSECUTIVE_CAMERA_FAILURES = 30
WINDOW_NAME   = "Gesture Recognition"
CAMERA_CONFIG = CameraConfig()

GESTURE_SPACE = "WORD_BREAK"
GESTURE_END   = "SENTENCE_BREAK"


# ─────────────────────────────────────────
#  NORMALISE  (zero-guard for absent hand)
# ─────────────────────────────────────────
def normalize(landmarks: list) -> list:
    if all(v == 0.0 for v in landmarks):
        return landmarks
    base_x, base_y = landmarks[0], landmarks[1]
    normed = landmarks.copy()
    for i in range(0, len(normed), 3):
        normed[i]     -= base_x
        normed[i + 1] -= base_y
    max_val = max(abs(x) for x in normed) + 1e-6
    return [x / max_val for x in normed]


# ─────────────────────────────────────────
#  LETTER BUFFER
# ─────────────────────────────────────────
class LetterBuffer:
    """Accumulates held gestures into words and sentences."""

    def __init__(self, hold_frames: int = 20):
        self.sentence     = ""
        self.current_word = ""
        self.last_pred    = None
        self.hold_count   = 0
        self.hold_frames  = hold_frames

    def update(self, pred: str):
        """
        Call once per frame with the current smoothed prediction.
        Returns a notice string when a gesture fires, else None.
        """
        if pred == "Unknown" or pred is None:
            self.hold_count = 0
            self.last_pred  = None
            return None

        if pred == self.last_pred:
            self.hold_count += 1
            if self.hold_count == self.hold_frames:
                # ── reset so the same gesture doesn't re-fire next frame ──
                self.hold_count = 0
                self.last_pred  = None

                if pred == GESTURE_SPACE:
                    self.add_space()
                    return "space added"
                if pred == GESTURE_END:
                    self.end_sentence()
                    return "sentence ended"
                # regular letter
                self.current_word += pred
                return f"letter: {pred}"
        else:
            self.last_pred  = pred
            self.hold_count = 0

        return None

    def add_space(self):
        if self.current_word:
            self.sentence    += self.current_word + " "
            self.current_word = ""

    def end_sentence(self):
        self.sentence    += self.current_word
        self.current_word = ""

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence[:-1]

    def clear(self):
        self.sentence     = ""
        self.current_word = ""
        self.last_pred    = None
        self.hold_count   = 0

    def display_text(self) -> str:
        return self.sentence + self.current_word


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
def load_model_bundle(model_file: Path = MODEL_FILE):
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_file}'. "
            "Run train.py first to create it."
        )
    try:
        with model_file.open("rb") as f:
            bundle = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot load '{model_file.name}' with current package versions. "
            "Re-run train.py to rebuild it."
        ) from exc

    required = {"model", "scaler", "label_encoder"}
    missing  = required.difference(bundle)
    if missing:
        raise KeyError(f"Model bundle missing keys: {', '.join(sorted(missing))}")

    return bundle["model"], bundle["scaler"], bundle["label_encoder"]


# ─────────────────────────────────────────
#  MAIN RECOGNITION LOOP
# ─────────────────────────────────────────
def run_recognition(camera_index: int = 0):
    model, scaler, label_encoder = load_model_bundle()

    cap = open_camera(camera_index=camera_index, config=CAMERA_CONFIG)

    hand_tracker         = None
    pred_history         = deque(maxlen=10)
    conf_history         = deque(maxlen=10)
    letter_buf           = LetterBuffer(hold_frames=20)
    consecutive_failures = 0
    notice_text          = None
    notice_frames        = 0
    fps_tracker          = FrameRateTracker()
    overlay_state        = OverlayState()
    window_size          = None

    try:
        hand_tracker = AsyncHandTracker(
            num_hands=2,
            min_detection_confidence=0.6,
            min_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            tracking_size=(CAMERA_CONFIG.tracking_width, CAMERA_CONFIG.tracking_height),
        )

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0),
        )

        while True:
            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_CAMERA_FAILURES:
                    raise RuntimeError("Camera stopped returning frames.")
                continue

            consecutive_failures = 0
            frame = prepare_frame(frame, CAMERA_CONFIG)

            current_window_size = (frame.shape[1], frame.shape[0])
            if window_size != current_window_size:
                cv2.resizeWindow(WINDOW_NAME, *current_window_size)
                window_size = current_window_size

            fps = fps_tracker.tick()

            # ── hand detection ───────────────────────────────────────────
            final_pred = None
            top3       = []
            avg_conf   = 0.0
            hand_data  = {}

            timestamp_ms = int(time.monotonic() * 1000)
            hand_tracker.submit(frame, timestamp_ms)
            detection_result, hand_data, _ = hand_tracker.latest(
                now_timestamp_ms=timestamp_ms
            )

            if detection_result is not None and hand_data:
                draw_hand_landmarks(frame, detection_result)

                # ── zero-guarded normalisation ───────────────────────────
                left  = normalize(hand_data.get("Left",  [0.0] * 63))
                right = normalize(hand_data.get("Right", [0.0] * 63))
                landmarks_combined = left + right

                # ── inference ────────────────────────────────────────────
                x_input    = scaler.transform([landmarks_combined])
                probs      = model.predict_proba(x_input)[0]
                top_idx    = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                pred       = label_encoder.inverse_transform(
                    [model.classes_[top_idx]]
                )[0]

                if confidence < CONFIDENCE_THRESHOLD:
                    pred = "Unknown"

                pred_history.append(pred)
                conf_history.append(confidence)

                # majority-vote smoothing over last 10 frames
                final_pred = max(set(pred_history), key=pred_history.count)
                avg_conf   = float(np.mean(conf_history))

                # ── letter buffer ─────────────────────────────────────────
                action = letter_buf.update(final_pred)
                if action:
                    notice_text   = action
                    notice_frames = 45          # show notice for ~1.5 s at 30 fps

                # ── top-3 predictions for UI ──────────────────────────────
                num_top  = min(3, len(model.classes_))
                top3_idx = np.argsort(probs)[::-1][:num_top]
                top3 = [
                    (
                        label_encoder.inverse_transform([model.classes_[i]])[0],
                        float(probs[i]),
                    )
                    for i in top3_idx
                ]
            else:
                pred_history.clear()
                conf_history.clear()

            # ── draw UI ──────────────────────────────────────────────────
            active_notice = notice_text if notice_frames > 0 else None
            draw_recognition_overlay(
                frame,
                ui_state=overlay_state,
                detection_result=detection_result,
                prediction=final_pred or "waiting",
                confidence=avg_conf if conf_history else 0.0,
                top_predictions=top3,
                display_text=letter_buf.display_text(),
                fps=fps,
                hand_count=len(hand_data),
                notice_text=active_notice,
                has_hand=bool(hand_data),
            )

            if notice_frames > 0:
                notice_frames -= 1
                if notice_frames == 0:
                    notice_text = None

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                   # ESC — quit
                break
            elif key == ord("b"):           # B — backspace
                letter_buf.backspace()
            elif key == ord("c"):           # C — clear all
                letter_buf.clear()

    finally:
        cap.release()
        if hand_tracker is not None:
            hand_tracker.close()
        cv2.destroyAllWindows()


def main():
    run_recognition()


if __name__ == "__main__":
    main()
