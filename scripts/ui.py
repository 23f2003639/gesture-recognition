from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN TOKENS 
# ─────────────────────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Neutrals
INK          = (240, 244, 248)   # primary text
INK_DIM      = (180, 188, 198)   # secondary text
INK_MUTED    = (120, 130, 142)   # tertiary / hints
INK_GHOST    = (72,  80,  92)    # very quiet

# Backgrounds
PANEL_FILL   = (12,  14,  18)    # modal/card background
TRACK_FILL   = (38,  42,  50)    # progress track

# Accents
ACCENT_CYAN  = (100, 210, 255)   # dynamic mode / word highlight
ACCENT_GREEN = (90,  210, 140)   # active / tracking indicator
ACCENT_AMBER = (255, 190,  60)   # notice / alert
ACCENT_WHITE = (220, 228, 238)   # confidence bar / landmark dots

LINE_DIM     = (55,  62,  72)    # subtle dividers
LINE_ACTIVE  = (160, 175, 195)   # active border


# ─────────────────────────────────────────────────────────────────────────────
#  LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _lerp_color(a, b, t):
    t = _clamp(t, 0.0, 1.0)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _text_size(text: str, scale: float, thickness: int = 1) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def _draw_text(
    frame,
    text: str,
    pos: Tuple[int, int],
    scale: float,
    color: tuple,
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    """Draw anti-aliased text with an optional drop-shadow."""
    if not text:
        return
    x, y = int(pos[0]), int(pos[1])
    if shadow:
        cv2.putText(frame, text, (x + 1, y + 1), FONT, scale,
                    (10, 10, 14), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _fit_text(text: str, max_w: int, base_scale: float, min_scale: float = 0.28):
    """Shrink `scale` until `text` fits within `max_w` pixels."""
    scale = base_scale
    while scale >= min_scale:
        tw, _ = _text_size(text, scale)
        if tw <= max_w:
            return text, scale
        scale -= 0.02
    # Still too wide — truncate
    t = text
    while len(t) > 3:
        t = t[:-1]
        if _text_size(t + "...", min_scale)[0] <= max_w:
            return t + "...", min_scale
    return "...", min_scale


def _panel(frame, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    """Semi-transparent dark rectangle — provides text legibility."""
    x, y = max(0, x), max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = frame[y:y+h, x:x+w]
    cv2.rectangle(roi, (0, 0), (w, h), PANEL_FILL, -1)
    cv2.addWeighted(roi, alpha, frame[y:y+h, x:x+w], 1 - alpha, 0, roi)


def _progress_bar(frame, x: int, y: int, w: int, value: float,
                  color=ACCENT_WHITE, track=TRACK_FILL) -> None:
    fill = int(w * _clamp(value, 0.0, 1.0))
    cv2.line(frame, (x, y), (x + w, y), track, 2, cv2.LINE_AA)
    if fill > 0:
        cv2.line(frame, (x, y), (x + fill, y), color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  OVERLAY STATE  (per-window, manages animations)
# ─────────────────────────────────────────────────────────────────────────────

class OverlayState:
    """
    Carries all smoothly animated values across frames.
    Create ONE instance per window; call .step() each frame.
    """

    def __init__(self):
        self._t        = time.perf_counter()
        self.fps_disp  = 0.0
        self.conf_disp = 0.0
        self.activity  = 0.0     # 0 = idle, 1 = hand detected
        self.phase     = 0.0     # for pulse/breathing animations
        # Focus rectangle (hand bounding box), smoothed
        self._focus: Optional[Tuple[float, float, float, float]] = None
        # Notice toast
        self.notice_text  = ""
        self.notice_alpha = 0.0

    def step(
        self,
        frame,
        fps: float = 0.0,
        confidence: float = 0.0,
        detection_result=None,
        notice: Optional[str] = None,
        active: bool = False,
    ) -> None:
        now = time.perf_counter()
        dt  = _clamp(now - self._t, 1 / 240, 0.15)
        self._t = now

        exp10 = 1 - math.exp(-dt * 10)
        exp6  = 1 - math.exp(-dt * 6)
        exp8  = 1 - math.exp(-dt * 8)
        exp12 = 1 - math.exp(-dt * 12)

        # FPS — smooth slowly
        self.fps_disp  = self.fps_disp + (fps - self.fps_disp) * exp6
        # Confidence — smooth quickly
        self.conf_disp = self.conf_disp + (confidence - self.conf_disp) * exp10
        # Activity
        self.activity  = self.activity + ((1.0 if active else 0.0) - self.activity) * exp10
        # Phase (drives pulse)
        self.phase    += dt * (2.0 + self.activity * 3.0)

        # Focus rect (hand bounding box)
        target = _focus_rect(frame, detection_result)
        if target is None:
            target = _default_focus(frame)
        if self._focus is None:
            self._focus = target
        else:
            self._focus = tuple(
                c + (t - c) * exp8
                for c, t in zip(self._focus, target)
            )

        # Notice toast
        if notice:
            self.notice_text = notice
        self.notice_alpha += ((1.0 if notice else 0.0) - self.notice_alpha) * exp12
        if self.notice_alpha < 0.02 and not notice:
            self.notice_text = ""

    @property
    def focus(self):
        return self._focus

    def pulse(self) -> float:
        return 0.5 + 0.5 * math.sin(self.phase * math.tau)

    def breathing(self) -> float:
        """Slow 0→1→0 breathing, independent of activity."""
        return 0.5 + 0.5 * math.sin(self.phase * math.pi * 0.4)


# ─────────────────────────────────────────────────────────────────────────────
#  FOCUS RECT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _default_focus(frame) -> Tuple[float, float, float, float]:
    fw, fh = frame.shape[1], frame.shape[0]
    w, h = int(fw * 0.18), int(fh * 0.24)
    return (fw - w) / 2, (fh - h) / 2, float(w), float(h)


def _focus_rect(frame, result) -> Optional[Tuple[float, float, float, float]]:
    if result is None or not result.hand_landmarks:
        return None
    fw, fh = frame.shape[1], frame.shape[0]
    xs, ys = [], []
    for hand in result.hand_landmarks:
        for lm in hand:
            xs.append(lm.x * fw)
            ys.append(lm.y * fh)
    if not xs:
        return None
    px = max(20, int((max(xs) - min(xs)) * 0.18))
    py = max(20, int((max(ys) - min(ys)) * 0.22))
    x  = _clamp(min(xs) - px, 0, fw - 1)
    y  = _clamp(min(ys) - py, 0, fh - 1)
    w  = _clamp(max(xs) - min(xs) + px * 2, 90, fw - x)
    h  = _clamp(max(ys) - min(ys) + py * 2, 110, fh - y)
    return x, y, w, h


# ─────────────────────────────────────────────────────────────────────────────
#  CORNER GUIDES  (hand bounding-box bracket)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_corner_guides(frame, state: OverlayState) -> None:
    if state.activity < 0.06 or state.focus is None:
        return
    x, y, w, h = (int(v) for v in state.focus)
    color = _lerp_color(LINE_DIM, LINE_ACTIVE, 0.2 + state.activity * 0.8)
    arm = max(12, int(min(w, h) * 0.12))

    for cx, cy, dx, dy in [
        (x,     y,      1,  1),
        (x + w, y,     -1,  1),
        (x,     y + h,  1, -1),
        (x + w, y + h, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  TOP-RIGHT STATUS STRIP
# ─────────────────────────────────────────────────────────────────────────────

def _draw_status_strip(frame, state: OverlayState, hand_count: int, active: bool) -> None:
    pad = 18
    y   = pad + 16

    tokens = [
        ("*", _lerp_color(INK_GHOST, ACCENT_GREEN, 0.3 + 0.7 * state.pulse() * state.activity)),
        ("tracking" if active else "idle", ACCENT_GREEN if active else INK_MUTED),
        (f"{hand_count}H", INK_DIM),
        (f"{state.fps_disp:.0f}fps" if state.fps_disp > 1 else "--", INK_MUTED),
    ]

    x = frame.shape[1] - pad
    for text, color in reversed(tokens):
        tw, _ = _text_size(text, 0.33)
        x -= tw
        _draw_text(frame, text, (x, y), 0.33, color)
        x -= 10


# ─────────────────────────────────────────────────────────────────────────────
#  NOTICE TOAST  (top-centre)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_notice(frame, state: OverlayState) -> None:
    if state.notice_alpha < 0.03 or not state.notice_text:
        return
    text  = state.notice_text
    scale = 0.40
    tw, _ = _text_size(text, scale)
    x = (frame.shape[1] - tw) // 2
    y = 32
    color = _lerp_color(INK_GHOST, ACCENT_AMBER, state.notice_alpha)
    _draw_text(frame, text, (x, y), scale, color)


# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION CALLOUT  (floating card next to hand)
# ─────────────────────────────────────────────────────────────────────────────

def _callout_pos(frame, focus, card_w: int, card_h: int):
    fx, fy, fw, fh = (int(v) for v in focus)
    pad, gap = 18, 14
    ay = int(_clamp(fy - 6, 48, frame.shape[0] - card_h - 72))

    rx = fx + int(fw) + gap
    if rx + card_w <= frame.shape[1] - pad:
        return rx, ay, "right"
    lx = fx - card_w - gap
    if lx >= pad:
        return lx, ay, "left"
    return pad, max(48, fy - card_h - 12), "top"


def _draw_prediction_callout(
    frame,
    state: OverlayState,
    mode_label: str,
    primary: str,
    secondary: str,
    confidence: float,
    alternates: str = "",
) -> None:
    """Floating prediction card anchored to the hand bounding box."""
    if state.focus is None:
        return

    # Card dimensions
    pw, _ = _text_size(primary,   1.0, 2)
    sw, _ = _text_size(secondary, 0.42)
    card_w = max(190, pw + sw + 48, _text_size(alternates, 0.28)[0] + 16)
    card_h = 60 if not alternates else 74

    cx, cy, side = _callout_pos(frame, state.focus, card_w, card_h)
    cx, cy = int(cx), int(cy)

    # Semi-transparent background
    _panel(frame, cx - 4, cy - 18, card_w + 8, card_h + 10, alpha=0.65)

    # Connector line to hand
    fx, fy, fw, fh = (int(v) for v in state.focus)
    anchor_y = fy + min(24, int(fh) // 3)
    lc = _lerp_color(INK_GHOST, LINE_ACTIVE, 0.25 + state.activity * 0.75)

    if side == "right":
        cv2.line(frame, (fx + fw, anchor_y), (cx - 10, anchor_y), lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 10, anchor_y), (cx, cy + 14), lc, 1, cv2.LINE_AA)
    elif side == "left":
        cv2.line(frame, (fx, anchor_y), (cx + card_w + 10, anchor_y), lc, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + card_w + 10, anchor_y), (cx + card_w, cy + 14), lc, 1, cv2.LINE_AA)
    else:
        cv2.line(frame, (fx + fw // 2, fy), (fx + fw // 2, cy + card_h + 8), lc, 1, cv2.LINE_AA)
        cv2.line(frame, (fx + fw // 2, cy + card_h + 8), (cx + 12, cy + card_h), lc, 1, cv2.LINE_AA)

    # Top rule
    cv2.line(frame, (cx, cy - 2), (cx + card_w, cy - 2), lc, 1, cv2.LINE_AA)

    # Mode label
    _draw_text(frame, mode_label, (cx, cy + 10), 0.28, INK_MUTED)

    # Primary prediction (big letter / word)
    primary_s, _ = _fit_text(primary, card_w - sw - 10, 1.0, 0.60)
    _draw_text(frame, primary_s, (cx, cy + 44), 1.0, INK, 2)

    # Confidence (to the right of primary)
    pw2, _ = _text_size(primary_s, 1.0, 2)
    _draw_text(frame, secondary, (cx + pw2 + 8, cy + 38), 0.42, ACCENT_WHITE)

    # Confidence bar
    _progress_bar(frame, cx, cy + 50, card_w - 4, confidence, ACCENT_WHITE)

    # Alternates hint
    if alternates:
        alt_s, alt_sc = _fit_text(alternates, card_w, 0.28, 0.24)
        _draw_text(frame, alt_s, (cx, cy + 68), alt_sc, INK_GHOST)


# ─────────────────────────────────────────────────────────────────────────────
#  BOTTOM STRIP  (accumulated text + hints)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_bottom_strip(frame, text: str, hints: str, layout_pad: int = 18) -> None:
    h, w = frame.shape[:2]
    bar_h = 76
    bar_y = h - bar_h

    _panel(frame, 0, bar_y, w, bar_h, alpha=0.72)
    cv2.line(frame, (0, bar_y), (w, bar_y), LINE_DIM, 1, cv2.LINE_AA)

    subtitle = text.strip()
    text_y   = bar_y + 38
    max_w    = w - 2 * layout_pad
    scale    = 0.72

    if subtitle:
        tw, _ = _text_size(subtitle, scale, 2)

        if tw <= max_w:
            # Fits — centre it
            sx = max(layout_pad, (w - tw) // 2)
            _draw_text(frame, subtitle, (sx, text_y), scale, INK, 2)
        else:
            # Too wide — clip from the left so the newest text stays visible.
            canvas_w = tw + layout_pad * 2
            canvas   = np.zeros((bar_h, canvas_w, 3), dtype=np.uint8)
            cv2.putText(canvas, subtitle, (layout_pad, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, INK, 2, cv2.LINE_AA)

            src_x  = canvas_w - max_w
            dst_x  = layout_pad
            src_x  = max(0, src_x)
            copy_w = min(canvas_w - src_x, max_w, frame.shape[1] - dst_x)
            if copy_w > 0:
                region = frame[bar_y:bar_y + bar_h, dst_x:dst_x + copy_w]
                region[:] = cv2.addWeighted(
                    canvas[:bar_h, src_x:src_x + copy_w], 1.0,
                    region, 0.0, 0,
                )
            # Fade-in gradient on the left edge
            fade_w = min(48, copy_w)
            for i in range(fade_w):
                alpha = i / fade_w
                frame[bar_y:bar_y + bar_h,
                      dst_x + i:dst_x + i + 1] = (
                    frame[bar_y:bar_y + bar_h,
                          dst_x + i:dst_x + i + 1].astype(float) * alpha
                ).astype(np.uint8)
    else:
        placeholder = "..."
        tw, _ = _text_size(placeholder, 0.52, 1)
        _draw_text(frame, placeholder, ((w - tw) // 2, text_y), 0.52, INK_GHOST, 1)

    hints_text, hints_scale = _fit_text(hints, w - 2 * layout_pad, 0.25, 0.20)
    hw, _ = _text_size(hints_text, hints_scale)
    _draw_text(frame, hints_text, ((w - hw) // 2, h - 14), hints_scale, INK_GHOST)

# ─────────────────────────────────────────────────────────────────────────────
#  MODE BADGE  (top-left)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_mode_badge(frame, label: str, sub: str = "") -> None:
    pad = 18
    _draw_text(frame, label, (pad, pad + 16), 0.40, INK_DIM)
    if sub:
        _draw_text(frame, sub, (pad, pad + 34), 0.28, INK_GHOST)


# ─────────────────────────────────────────────────────────────────────────────
#  HOLD PROGRESS ARC  (dynamic mode — shows hold countdown)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hold_arc(frame, state: OverlayState, hold_progress: float) -> None:
    """
    Draws a circular arc above the hand to visualise hold progress (0→1).
    Only visible when actively holding a dynamic gesture.
    """
    if state.focus is None or hold_progress <= 0.0:
        return
    fx, fy, fw, fh = (int(v) for v in state.focus)
    cx = fx + fw // 2
    cy = fy - 24
    radius = 18
    end_angle = int(360 * hold_progress) - 90
    color = _lerp_color(ACCENT_WHITE, ACCENT_CYAN, hold_progress)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, -90, end_angle,
                TRACK_FILL, 3, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, -90, end_angle,
                color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — RECOGNITION OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def draw_recognition_overlay(
    frame,
    state: OverlayState,
    detection_result,
    mode: str,                    # "static" | "dynamic"
    prediction: str,
    confidence: float,
    top_predictions: List[Tuple[str, float]],
    display_text: str,
    fps: float,
    hand_count: int,
    notice: Optional[str] = None,
    has_hand: bool = True,
    hold_progress: float = 0.0,  # dynamic only: 0→1 hold fill
) -> None:
    state.step(
        frame=frame,
        fps=fps,
        confidence=confidence,
        detection_result=detection_result,
        notice=notice,
        active=has_hand,
    )

    _draw_corner_guides(frame, state)
    _draw_status_strip(frame, state, hand_count, has_hand)
    _draw_notice(frame, state)

    mode_badge   = "STATIC - LETTER"  if mode == "static" else "DYNAMIC - WORD"
    mode_sub     = "steady hold"      if mode == "static" else "steady motion"
    _draw_mode_badge(frame, mode_badge, mode_sub)

    if has_hand and prediction not in ("waiting", "Unknown", None):
        alts = ""
        if top_predictions and len(top_predictions) > 1:
            alts = "alt: " + "  /  ".join(
                f"{n} {p * 100:.0f}%" for n, p in top_predictions[1:3]
            )
        _draw_prediction_callout(
            frame, state,
            mode_label=mode_badge,
            primary=prediction,
            secondary=f"{confidence * 100:.0f}%",
            confidence=confidence,
            alternates=alts,
        )

    if mode == "dynamic":
        _draw_hold_arc(frame, state, hold_progress)

    hints = (
        "B backspace  C clear  M mode  ESC quit"
        if mode == "static"
        else "B backspace  C clear  M mode  ESC quit"
    )
    _draw_bottom_strip(frame, display_text, hints)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — DATA COLLECTION OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def draw_collection_overlay(
    frame,
    state: OverlayState,
    detection_result,
    label: str,
    saved_count: int,
    target_samples: int,
    hand_labels: list,
    fps: float,
    notice: Optional[str] = None,
    # Dynamic collection extras
    buffer_fill: int = 0,
    sequence_length: int = 0,
) -> None:
    """Collection UI — used by both static and dynamic data.py."""
    state.step(
        frame=frame,
        fps=fps,
        confidence=saved_count / max(target_samples, 1),
        detection_result=detection_result,
        notice=notice,
        active=bool(hand_labels),
    )

    _draw_corner_guides(frame, state)
    _draw_status_strip(frame, state, len(hand_labels), bool(hand_labels))
    _draw_notice(frame, state)

    pad = 18
    # Label + count
    _draw_text(frame, f"{label}  {saved_count}/{target_samples}", (pad, pad + 16), 0.44, INK)
    state_label = "tracking" if hand_labels else "waiting for hands"
    _draw_text(frame, state_label, (pad, pad + 34), 0.28, INK_MUTED)

    # Overall progress bar
    _progress_bar(frame, pad, frame.shape[0] - pad - 8,
                  frame.shape[1] - 2 * pad,
                  saved_count / max(target_samples, 1), ACCENT_GREEN)

    # Dynamic: per-sequence capture progress
    if sequence_length > 0:
        _progress_bar(frame, pad, frame.shape[0] - pad - 20,
                      frame.shape[1] - 2 * pad,
                      buffer_fill / sequence_length, ACCENT_CYAN)
        seq_text = f"frame {buffer_fill}/{sequence_length}"
        _draw_text(frame, seq_text, (pad, frame.shape[0] - pad - 28),
                   0.28, INK_MUTED)

    hints = "SPACE start  N next  B prev  ESC quit" if sequence_length else "S save  ESC quit"
    hw, _ = _text_size(hints, 0.27)
    _draw_text(frame, hints,
               (frame.shape[1] - pad - hw, frame.shape[0] - pad - 8),
               0.27, INK_GHOST)
