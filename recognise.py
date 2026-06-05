"""
recognise.py — Production-ready unified gesture recognition (static + dynamic).

HOW TO RUN:
    python recognise.py                  # unified (recommended)
    python recognise.py --mode static    # static only
    python recognise.py --mode dynamic   # dynamic only
    python recognise.py --camera 1       # different camera

CONTROLS:
    B     — backspace (1 letter / 1 word)
    C     — clear everything
    M     — toggle mode
    ESC   — quit

SENTENCE FORMAT:
    Letters spell words. WORD_BREAK = space. SENTENCE_BREAK = period + capitalise next.
    Dynamic words append directly. Output scrolls right-to-left in a single line.
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from camera import CameraConfig, FrameRateTracker, open_camera, prepare_frame
from feature_extraction import (
    canonical_label,
    extract_dynamic_sequence,
    extract_static_features,
    hand_data_to_base_features,
    sequence_motion_energy,
    sequence_presence_ratio,
)
from hand_tracking import AsyncHandTracker, draw_hand_landmarks
from personalization import choose_dynamic_tflite_path, choose_model_path, user_dynamic_tflite_path
from ui import OverlayState, draw_recognition_overlay
from utils import normalize, SEQUENCE_LENGTH, LANDMARK_DIM

BASE_DIR       = Path(__file__).resolve().parent
STATIC_MODEL   = BASE_DIR / "static_model.pkl"
DYNAMIC_MODEL  = BASE_DIR / "dynamic_model.pkl"
DYNAMIC_TFLITE = BASE_DIR / "dynamic_model.tflite"

CAMERA_CONFIG = CameraConfig()
WINDOW_NAME   = "Gesture Recognition"

# ── Confidence gates ──────────────────────────────────────────────────────────
STATIC_CONF_THRESHOLD    = 0.70
STATIC_MARGIN_THRESHOLD  = 0.05

DYNAMIC_CONF_THRESHOLD   = 0.72
DYNAMIC_MARGIN_THRESHOLD = 0.08

DYNAMIC_MIN_MOTION   = 0.035
DYNAMIC_MIN_PRESENCE = 0.50

STATIC_HOLD_FRAMES      = 20
DYNAMIC_HOLD_FRAMES     = 8
DYNAMIC_COOLDOWN_FRAMES = 20

MAX_FAILURES  = 30
GESTURE_SPACE = "WORD_BREAK"
GESTURE_END   = "SENTENCE_BREAK"
MODES         = ["unified", "static", "dynamic"]


# ─────────────────────────────────────────────────────────────────────────────
#  SENTENCE ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

class SentenceAssembler:
    """
    Single source of truth for all output text.

    Produces a single rolling string — no multi-line, no completed_sentences.
    The display() method returns everything as one string; the UI scrolls it
    right-to-left when it gets long.

    Rules:
      · Letters accumulate into current_word.
      · WORD_BREAK  → commit word, add space token.
      · SENTENCE_BREAK → commit word, add ".", capitalise next token.
      · Dynamic words → commit any partial word, then append as a token.
      · First letter / first word after SENTENCE_BREAK is auto-capitalised.
    """

    def __init__(self):
        self.tokens: list[str] = []   # committed words + punctuation
        self.current_word: str = ""   # letters accumulating right now
        self._next_cap: bool   = True # capitalise next committed token

    # ── Public API ────────────────────────────────────────────────────────

    def push_letter(self, letter: str) -> None:
        if not self.current_word and self._next_cap:
            letter = letter.upper()
        self.current_word += letter

    def push_word_break(self) -> None:
        self._commit_current_word()

    def push_sentence_end(self) -> None:
        self._commit_current_word()
        # Attach period to last real token
        for i in range(len(self.tokens) - 1, -1, -1):
            if self.tokens[i] not in ".!?,":
                self.tokens[i] = self.tokens[i] + "."
                break
        else:
            if self.tokens:
                self.tokens[-1] += "."
        self._next_cap = True

    def push_dynamic_word(self, word: str) -> None:
        self._commit_current_word()
        clean = word.replace("_", " ").replace("-", " ").strip()
        if not clean:
            return
        if self._next_cap:
            clean = clean.capitalize()
            self._next_cap = False
        self.tokens.append(clean)

    def backspace(self) -> None:
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.tokens:
            last = self.tokens[-1]
            # If it ends with a period we added, strip just the period
            if last.endswith(".") and len(last) > 1:
                self.tokens[-1] = last[:-1]
                self._next_cap = False
            else:
                self.tokens.pop()

    def clear(self) -> None:
        self.tokens.clear()
        self.current_word = ""
        self._next_cap    = True

    def display(self) -> str:
        """Return the full accumulated text as a single string."""
        parts: list[str] = []
        for token in self.tokens:
            parts.append(token)
        result = " ".join(parts)
        if self.current_word:
            result = (result + " " + self.current_word).lstrip()
        return result

    # ── Internal ──────────────────────────────────────────────────────────

    def _commit_current_word(self) -> None:
        if not self.current_word:
            return
        word = self.current_word
        self.current_word = ""
        if self._next_cap:
            word = word.capitalize()
            self._next_cap = False
        self.tokens.append(word)


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC HOLD BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class StaticHoldBuffer:
    """
    Hold a gesture for `hold_frames` consecutive frames to commit it.
    locked_pred prevents immediate re-fire while the hand is still held.
    """

    def __init__(self, hold_frames: int = STATIC_HOLD_FRAMES):
        self.hold_frames   = hold_frames
        self.last_pred     = None
        self.locked_pred   = None
        self.hold_count    = 0
        self.hold_progress = 0.0

    def update(self, pred: Optional[str]) -> Optional[str]:
        if pred in (None, "Unknown", "waiting"):
            self.hold_count    = 0
            self.last_pred     = None
            self.locked_pred   = None
            self.hold_progress = 0.0
            return None

        if pred == self.locked_pred:
            self.hold_progress = 0.0
            return None

        if pred == self.last_pred:
            self.hold_count += 1
            self.hold_progress = self.hold_count / self.hold_frames
            if self.hold_count >= self.hold_frames:
                self.hold_count    = 0
                self.locked_pred   = pred
                self.last_pred     = None
                self.hold_progress = 0.0
                return self._classify(pred)
        else:
            self.last_pred     = pred
            self.hold_count    = 1
            self.locked_pred   = None
            self.hold_progress = 1.0 / self.hold_frames
        return None

    def _classify(self, pred: str) -> str:
        if pred == GESTURE_SPACE:
            return "word_break"
        if pred == GESTURE_END:
            return "sentence_end"
        return f"letter:{pred}"

    def clear(self) -> None:
        self.last_pred     = None
        self.locked_pred   = None
        self.hold_count    = 0
        self.hold_progress = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC HOLD BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class DynamicHoldBuffer:
    def __init__(
        self,
        hold_frames: int = DYNAMIC_HOLD_FRAMES,
        cooldown_frames: int = DYNAMIC_COOLDOWN_FRAMES,
    ):
        self.hold_frames     = hold_frames
        self.cooldown_frames = cooldown_frames
        self.last_pred       = None
        self.hold_count      = 0
        self.gesture_locked  = False
        self.cooldown_count  = 0
        self.hold_progress   = 0.0

    def update(self, pred: Optional[str], confidence: float = 0.0) -> Optional[str]:
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            self.hold_progress   = 0.0
            if pred in (None, "Unknown", "waiting"):
                self.gesture_locked = False
            return None

        if pred in (None, "Unknown", "waiting"):
            self.hold_count     = 0
            self.last_pred      = None
            self.gesture_locked = False
            self.hold_progress  = 0.0
            return None

        if pred == self.last_pred:
            self.hold_count    += 1
            self.hold_progress  = self.hold_count / self.hold_frames
            if self.hold_count >= self.hold_frames and not self.gesture_locked:
                self.gesture_locked = True
                self.cooldown_count = self.cooldown_frames
                self.hold_progress  = 1.0
                return pred
        else:
            self.last_pred      = pred
            self.hold_count     = 1
            self.gesture_locked = False
            self.hold_progress  = 1.0 / self.hold_frames
        return None

    def clear(self) -> None:
        self.last_pred      = None
        self.hold_count     = 0
        self.gesture_locked = False
        self.cooldown_count = 0
        self.hold_progress  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION STABILIZER
# ─────────────────────────────────────────────────────────────────────────────

class PredictionStabilizer:
    UNKNOWN = frozenset({"Unknown", "waiting", None})

    def __init__(self, min_frames: int = 3, keep_frames: int = 10):
        self.min_frames    = min_frames
        self.keep_frames   = keep_frames
        self.candidate     = None
        self.cand_count    = 0
        self.stable        = None
        self.stable_conf   = 0.0
        self.stable_top3   = []
        self.missing_count = 0

    def update(self, pred, conf, top3, raw_pred=None):
        if pred in self.UNKNOWN:
            self.cand_count    = 0
            self.candidate     = None
            self.missing_count += 1
            if self.missing_count > self.keep_frames:
                self._clear_stable()
            return self._current()

        self.missing_count = 0
        if pred == self.candidate:
            self.cand_count += 1
        else:
            self.candidate  = pred
            self.cand_count = 1

        if self.cand_count >= self.min_frames:
            self.stable      = pred
            self.stable_conf = conf
            self.stable_top3 = top3

        if self.stable is None and raw_pred and raw_pred not in self.UNKNOWN:
            return raw_pred, conf, top3

        return self._current()

    def _current(self):
        if self.stable is None:
            return "Unknown", 0.0, []
        return self.stable, self.stable_conf, self.stable_top3

    def _clear_stable(self):
        self.stable      = None
        self.stable_conf = 0.0
        self.stable_top3 = []

    def clear(self):
        self.candidate     = None
        self.cand_count    = 0
        self.missing_count = 0
        self._clear_stable()


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class StaticEngine:
    def __init__(self, model_file: Path = STATIC_MODEL):
        if not model_file.exists():
            raise FileNotFoundError(f"Static model not found: '{model_file}'. Run train_static.py first.")
        with model_file.open("rb") as f:
            bundle = pickle.load(f)
        self.model         = bundle["model"]
        self.scaler        = bundle["scaler"]
        self.label_encoder = bundle["label_encoder"]
        self.class_labels  = np.array([
            canonical_label(label)
            for label in self.label_encoder.inverse_transform(self.model.classes_)
        ])
        self.feature_dim = int(
            bundle.get("feature_dim") or getattr(self.scaler, "n_features_in_", LANDMARK_DIM)
        )

    def predict(self, hand_data: dict) -> tuple[str, float, list]:
        base     = hand_data_to_base_features(hand_data, normalize)
        features = base if self.feature_dim == LANDMARK_DIM else extract_static_features(base)

        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Feature mismatch: model={self.feature_dim}, computed={features.shape[0]}")

        x     = self.scaler.transform([features])
        probs = self.model.predict_proba(x)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3     = [(self.class_labels[i], float(probs[i])) for i in top3_idx]
        top_idx  = int(top3_idx[0])
        conf     = float(probs[top_idx])
        margin   = float(probs[top3_idx[0]] - probs[top3_idx[1]]) if len(top3_idx) > 1 else conf

        pred = (
            self.class_labels[top_idx]
            if conf >= STATIC_CONF_THRESHOLD and margin >= STATIC_MARGIN_THRESHOLD
            else "Unknown"
        )
        return pred, conf, top3


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DynamicEngine:
    def __init__(self, pkl_file: Path = DYNAMIC_MODEL, tflite_file: Path = DYNAMIC_TFLITE):
        if not pkl_file.exists():
            raise FileNotFoundError(f"Dynamic model not found: '{pkl_file}'. Run train_dynamic.py first.")
        with pkl_file.open("rb") as f:
            bundle = pickle.load(f)

        self.scaler        = bundle["scaler"]
        self.label_encoder = bundle["label_encoder"]
        self.class_labels  = np.array([
            canonical_label(label) for label in self.label_encoder.classes_
        ])
        gate = bundle.get("gate", {})
        self.motion_threshold   = float(gate.get("motion_threshold",   DYNAMIC_MIN_MOTION))
        self.presence_threshold = float(gate.get("presence_threshold", DYNAMIC_MIN_PRESENCE))
        self.last_motion_energy  = 0.0
        self.last_presence_ratio = 0.0
        self.last_motion_active  = False

        tflite_path = Path(bundle["tflite_path"]) if bundle.get("tflite_path") else tflite_file
        self.use_tflite  = bool(tflite_path and tflite_path.exists())
        self.keras_model = None
        input_feature_dim = None

        if self.use_tflite:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            self.interpreter.allocate_tensors()
            inp_details       = self.interpreter.get_input_details()[0]
            self.in_idx       = inp_details["index"]
            self.out_idx      = self.interpreter.get_output_details()[0]["index"]
            input_feature_dim = inp_details["shape"][-1]
        else:
            keras_path = bundle.get("keras_path")
            if not keras_path or not Path(keras_path).exists():
                raise FileNotFoundError("No TFLite or Keras model found. Run train_dynamic.py.")
            import tensorflow as tf
            self.keras_model  = tf.keras.models.load_model(keras_path)
            self.interpreter  = None
            self.in_idx = self.out_idx = None
            input_feature_dim = self.keras_model.input_shape[-1]

        self.feature_dim = int(
            bundle.get("feature_dim")
            or getattr(self.scaler, "n_features_in_", LANDMARK_DIM)
            or input_feature_dim
        )
        self.sequence: deque = deque(maxlen=SEQUENCE_LENGTH)

    def push_frame(self, hand_data: dict) -> None:
        self.sequence.append(hand_data_to_base_features(hand_data, normalize))

    def predict(self) -> tuple[str, float, list]:
        if len(self.sequence) < SEQUENCE_LENGTH:
            self.last_motion_active = False
            return "waiting", 0.0, []

        arr = np.array(self.sequence, dtype=np.float32)
        self.last_presence_ratio = sequence_presence_ratio(arr)
        self.last_motion_energy  = sequence_motion_energy(arr)
        self.last_motion_active  = (
            self.last_presence_ratio >= self.presence_threshold
            and self.last_motion_energy  >= self.motion_threshold
        )
        if not self.last_motion_active:
            return "Unknown", 0.0, []

        features = arr if self.feature_dim == LANDMARK_DIM else extract_dynamic_sequence(arr)
        flat = self.scaler.transform(features)
        x    = flat.reshape(1, SEQUENCE_LENGTH, self.feature_dim).astype(np.float32)

        if self.use_tflite:
            self.interpreter.set_tensor(self.in_idx, x)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.out_idx)[0]
        else:
            probs = self.keras_model(x, training=False).numpy()[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3     = [(self.class_labels[i], float(probs[i])) for i in top3_idx]
        top_idx  = int(top3_idx[0])
        conf     = float(probs[top_idx])
        margin   = float(probs[top3_idx[0]] - probs[top3_idx[1]]) if len(top3_idx) > 1 else conf

        pred = (
            self.class_labels[top_idx]
            if conf >= DYNAMIC_CONF_THRESHOLD and margin >= DYNAMIC_MARGIN_THRESHOLD
            else "Unknown"
        )
        return pred, conf, top3

    def clear_sequence(self) -> None:
        self.sequence.clear()
        self.last_motion_energy  = 0.0
        self.last_presence_ratio = 0.0
        self.last_motion_active  = False


# ─────────────────────────────────────────────────────────────────────────────
#  SMOOTHED MAJORITY VOTE
# ─────────────────────────────────────────────────────────────────────────────

def _smoothed_prediction(labels: deque, confidences: deque, default: str = "waiting"):
    if not labels:
        return default, 0.0
    counts    = Counter(labels)
    max_count = max(counts.values())
    tied      = {lbl for lbl, cnt in counts.items() if cnt == max_count}
    best      = default
    for lbl in reversed(labels):
        if lbl in tied:
            best = lbl
            break
    confs = [c for lbl, c in zip(labels, confidences) if lbl == best]
    return best, float(np.mean(confs)) if confs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run(mode: str = "unified", camera_index: int = 0, user_id: str | None = None) -> None:
    assert mode in MODES

    static_engine:  Optional[StaticEngine]  = None
    dynamic_engine: Optional[DynamicEngine] = None

    if mode in ("static", "unified"):
        path = choose_model_path(STATIC_MODEL, user_id, "static")
        if not path.exists():
            print(f"⚠  Static model not found — falling back to dynamic-only.")
            mode = "dynamic"
        else:
            static_engine = StaticEngine(path)
            print(f"✓ Static model  — {len(static_engine.class_labels)} classes")

    if mode in ("dynamic", "unified"):
        pkl_path    = choose_model_path(DYNAMIC_MODEL, user_id, "dynamic")
        tflite_path = choose_dynamic_tflite_path(DYNAMIC_TFLITE, user_id)
        if user_id and pkl_path != DYNAMIC_MODEL:
            tflite_path = user_dynamic_tflite_path(user_id)
        if not pkl_path.exists():
            print("⚠  Dynamic model not found — falling back to static-only.")
            mode = "static" if mode == "unified" else mode
        else:
            dynamic_engine = DynamicEngine(pkl_path, tflite_path)
            print(f"✓ Dynamic model — {len(dynamic_engine.class_labels)} classes")

    text_buf     = SentenceAssembler()
    static_hold  = StaticHoldBuffer(STATIC_HOLD_FRAMES)
    dynamic_hold = DynamicHoldBuffer(DYNAMIC_HOLD_FRAMES, DYNAMIC_COOLDOWN_FRAMES)

    static_hist   = deque(maxlen=7)
    static_confs  = deque(maxlen=7)
    dynamic_hist  = deque(maxlen=5)
    dynamic_confs = deque(maxlen=5)

    static_disp  = PredictionStabilizer(min_frames=3, keep_frames=10)
    dynamic_disp = PredictionStabilizer(min_frames=2, keep_frames=6)

    cap          = open_camera(camera_index, CAMERA_CONFIG)
    hand_tracker = AsyncHandTracker(
        num_hands=2,
        tracking_size=(CAMERA_CONFIG.tracking_width, CAMERA_CONFIG.tracking_height),
    )
    fps_tracker  = FrameRateTracker()
    overlay      = OverlayState()

    notice, notice_frames = None, 0
    consecutive_fails     = 0
    window_size           = None
    skipped_frames        = 0
    mode_index            = MODES.index(mode)

    print(f"\n{'='*62}")
    print(f"  UNIFIED GESTURE RECOGNITION  |  mode: {mode.upper()}")
    print(f"  B=backspace  C=clear  M=toggle-mode  ESC=quit")
    print(f"{'='*62}\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_fails += 1
                if consecutive_fails >= MAX_FAILURES:
                    raise RuntimeError("Camera stopped responding.")
                continue
            consecutive_fails = 0
            frame = prepare_frame(frame, CAMERA_CONFIG)

            if window_size != (frame.shape[1], frame.shape[0]):
                window_size = (frame.shape[1], frame.shape[0])
                cv2.resizeWindow(WINDOW_NAME, *window_size)

            fps = fps_tracker.tick()
            ts  = int(time.monotonic() * 1000)

            submitted = hand_tracker.submit(frame, ts)
            if not submitted:
                skipped_frames += 1

            detection_result, hand_data, _ = hand_tracker.latest(ts, max_age_ms=200)
            if detection_result:
                draw_hand_landmarks(frame, detection_result)

            s_pred, s_conf, s_top3 = "waiting", 0.0, []
            d_pred, d_conf, d_top3 = "waiting", 0.0, []
            s_show_label, s_show_conf, s_show_top3 = "Unknown", 0.0, []
            d_show_label, d_show_conf, d_show_top3 = "Unknown", 0.0, []

            current_mode = MODES[mode_index]

            if hand_data:
                if static_engine and current_mode in ("static", "unified"):
                    raw_s, raw_sc, raw_s_top3 = static_engine.predict(hand_data)
                    static_hist.append(raw_s)
                    static_confs.append(raw_sc)
                    s_pred, s_conf = _smoothed_prediction(static_hist, static_confs)
                    s_top3 = raw_s_top3
                    s_show_label, s_show_conf, s_show_top3 = static_disp.update(
                        s_pred, s_conf, s_top3, raw_pred=raw_s
                    )

                if dynamic_engine and current_mode in ("dynamic", "unified"):
                    dynamic_engine.push_frame(hand_data)
                    raw_d, raw_dc, raw_d_top3 = dynamic_engine.predict()
                    if raw_d != "waiting":
                        dynamic_hist.append(raw_d)
                        dynamic_confs.append(raw_dc)
                        d_pred, d_conf = _smoothed_prediction(dynamic_hist, dynamic_confs)
                        d_top3 = raw_d_top3
                    else:
                        d_pred, d_conf = "waiting", 0.0
                    d_show_label, d_show_conf, d_show_top3 = dynamic_disp.update(
                        d_pred, d_conf, d_top3, raw_pred=raw_d
                    )

            else:
                static_hist.clear();  static_confs.clear()
                dynamic_hist.clear(); dynamic_confs.clear()
                static_disp.clear();  dynamic_disp.clear()
                if dynamic_engine:
                    dynamic_engine.clear_sequence()
                static_hold.clear()
                dynamic_hold.clear()

            # ── Commit static ──────────────────────────────────────────────
            if current_mode in ("static", "unified") and static_engine:
                dynamic_active = (
                    current_mode == "unified"
                    and dynamic_engine is not None
                    and dynamic_engine.last_motion_energy >= DYNAMIC_MIN_MOTION * 1.5
                )
                feed_pred = None if dynamic_active else s_pred
                event = static_hold.update(feed_pred)
                if event == "word_break":
                    text_buf.push_word_break()
                    notice, notice_frames = "[ space ]", 45
                elif event == "sentence_end":
                    text_buf.push_sentence_end()
                    notice, notice_frames = "[ . ]", 45
                elif event and event.startswith("letter:"):
                    letter = event.split(":", 1)[1]
                    text_buf.push_letter(letter)
                    notice, notice_frames = f"[ {letter} ]", 35

            # ── Commit dynamic ─────────────────────────────────────────────
            if current_mode in ("dynamic", "unified") and dynamic_engine:
                word_event = dynamic_hold.update(d_pred, d_conf)
                if word_event:
                    text_buf.push_dynamic_word(word_event)
                    dynamic_engine.clear_sequence()
                    dynamic_hist.clear(); dynamic_confs.clear()
                    dynamic_disp.clear()
                    notice, notice_frames = f"[ {word_event} ]", 50

            # ── Display ────────────────────────────────────────────────────
            display_text = text_buf.display()   # single rolling string

            if current_mode == "dynamic" or (
                current_mode == "unified"
                and dynamic_engine is not None
                and (
                    dynamic_engine.last_motion_active
                    or d_show_label not in ("waiting", "Unknown")
                )
            ):
                callout_pred = d_show_label
                callout_conf = d_show_conf
                callout_top3 = d_show_top3
                hold_prog    = dynamic_hold.hold_progress
                callout_mode = "dynamic"
            else:
                callout_pred = s_show_label
                callout_conf = s_show_conf
                callout_top3 = s_show_top3
                hold_prog    = static_hold.hold_progress
                callout_mode = "static"

            active_notice = notice if notice_frames > 0 else None
            draw_recognition_overlay(
                frame,
                state=overlay,
                detection_result=detection_result,
                mode=callout_mode,
                prediction=callout_pred,
                confidence=callout_conf,
                top_predictions=callout_top3,
                display_text=display_text,
                fps=fps,
                hand_count=len(hand_data) if hand_data else 0,
                notice=active_notice,
                has_hand=bool(hand_data),
                hold_progress=hold_prog,
            )

            if notice_frames > 0:
                notice_frames -= 1

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord("b"):
                text_buf.backspace()
                static_hold.clear(); dynamic_hold.clear()
            elif key == ord("c"):
                text_buf.clear()
                static_hold.clear(); dynamic_hold.clear()
            elif key == ord("m"):
                mode_index = (mode_index + 1) % len(MODES)
                notice, notice_frames = f"Mode: {MODES[mode_index]}", 60
                print(f"  → {MODES[mode_index].upper()} mode")

    finally:
        cap.release()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print(f"\n✓ Session ended  |  camera skips: {skipped_frames}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=MODES, default="unified")
    parser.add_argument("--camera",  type=int,       default=0)
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()
    run(mode=args.mode, camera_index=args.camera, user_id=args.user_id)


if __name__ == "__main__":
    main()