from dataclasses import dataclass
import math
import time

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_PRIMARY = (244, 247, 250)
TEXT_SECONDARY = (194, 201, 210)
TEXT_MUTED = (145, 153, 163)
TEXT_SOFT = (108, 116, 127)
LINE = (132, 141, 152)
LINE_ACTIVE = (226, 231, 237)
ACCENT = (225, 230, 236)
ACCENT_SOFT = (182, 192, 205)
TRACK = (62, 68, 76)
SCRIM = (14, 16, 20)
LIVE_COLOR = (186, 208, 190)


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _mix_color(color_a, color_b, amount):
    amount = _clamp(amount, 0.0, 1.0)
    return tuple(int(color_a[i] + (color_b[i] - color_a[i]) * amount) for i in range(3))


def _measure_text(text, scale=0.4, thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return text_width, text_height


def _shadow_text(frame, text, origin, scale=0.4, color=TEXT_PRIMARY, thickness=1):
    origin = (int(origin[0]), int(origin[1]))
    shadow_color = (18, 18, 18)
    cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1), FONT, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _scrim(frame, x, y, width, height, alpha=0.18):
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(0, min(int(width), frame.shape[1] - x))
    height = max(0, min(int(height), frame.shape[0] - y))
    if width <= 0 or height <= 0:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), SCRIM, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _fit_text(text, max_width, base_scale, min_scale=0.28):
    scale = base_scale
    while scale >= min_scale:
        text_width, _ = _measure_text(text, scale=scale)
        if text_width <= max_width:
            return text, scale
        scale -= 0.02

    shortened = text
    while len(shortened) > 4:
        shortened = shortened[:-1]
        candidate = shortened + "..."
        text_width, _ = _measure_text(candidate, scale=min_scale)
        if text_width <= max_width:
            return candidate, min_scale

    return "...", min_scale


@dataclass
class OverlayLayout:
    padding: int
    bottom_y: int
    bottom_text_width: int


class OverlayState:
    def __init__(self):
        self.last_time = time.perf_counter()
        self.fps_display = 0.0
        self.confidence_display = 0.0
        self.activity = 0.0
        self.pulse_phase = 0.0
        self.focus_rect = None
        self.notice_text = ""
        self.notice_alpha = 0.0

    def step(self, frame, fps=0.0, confidence=0.0, detection_result=None, notice_text=None, active=False):
        now = time.perf_counter()
        dt = _clamp(now - self.last_time, 1 / 240, 0.2)
        self.last_time = now

        value_blend = 1 - math.exp(-dt * 10)
        rect_blend = 1 - math.exp(-dt * 8)

        if self.fps_display <= 0.01 and fps > 0:
            self.fps_display = fps
        else:
            self.fps_display += (fps - self.fps_display) * (1 - math.exp(-dt * 6))

        if self.confidence_display <= 0.01 and confidence > 0:
            self.confidence_display = confidence
        else:
            self.confidence_display += (confidence - self.confidence_display) * value_blend

        target_activity = 1.0 if active else 0.0
        self.activity += (target_activity - self.activity) * value_blend
        self.pulse_phase += dt * (2.0 + self.activity * 4.0)

        target_focus = _compute_focus_rect(frame, detection_result)
        if target_focus is None:
            target_focus = _default_focus_rect(frame)

        if self.focus_rect is None:
            self.focus_rect = target_focus
        else:
            self.focus_rect = tuple(
                current + (target - current) * rect_blend
                for current, target in zip(self.focus_rect, target_focus)
            )

        if notice_text:
            self.notice_text = notice_text
        target_notice = 1.0 if notice_text else 0.0
        self.notice_alpha += (target_notice - self.notice_alpha) * (1 - math.exp(-dt * 12))
        if self.notice_alpha < 0.02 and not notice_text:
            self.notice_text = ""

    def pulse(self):
        return 0.5 + 0.5 * math.sin(self.pulse_phase * math.tau)


def _layout_for(frame):
    padding = max(16, int(min(frame.shape[0], frame.shape[1]) * 0.02))
    return OverlayLayout(
        padding=padding,
        bottom_y=frame.shape[0] - padding - 18,
        bottom_text_width=int(frame.shape[1] * 0.58),
    )


def _default_focus_rect(frame):
    width = int(frame.shape[1] * 0.18)
    height = int(frame.shape[0] * 0.24)
    x = (frame.shape[1] - width) // 2
    y = (frame.shape[0] - height) // 2
    return x, y, width, height


def _compute_focus_rect(frame, detection_result):
    if detection_result is None or not detection_result.hand_landmarks:
        return None

    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    xs = []
    ys = []

    for hand in detection_result.hand_landmarks:
        for landmark in hand:
            xs.append(landmark.x * frame_w)
            ys.append(landmark.y * frame_h)

    if not xs or not ys:
        return None

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pad_x = max(18, int((max_x - min_x) * 0.18))
    pad_y = max(18, int((max_y - min_y) * 0.22))
    x = int(_clamp(min_x - pad_x, 0, frame_w - 1))
    y = int(_clamp(min_y - pad_y, 0, frame_h - 1))
    width = int(_clamp((max_x - min_x) + pad_x * 2, 90, frame_w - x))
    height = int(_clamp((max_y - min_y) + pad_y * 2, 110, frame_h - y))
    return x, y, width, height


def _callout_position(frame, focus_rect, width, height):
    fx, fy, fw, fh = (int(value) for value in focus_rect)
    padding = 18
    gap = 14
    target_y = int(_clamp(fy - 6, 48, frame.shape[0] - height - 72))

    right_x = fx + fw + gap
    if right_x + width <= frame.shape[1] - padding:
        return right_x, target_y, "right"

    left_x = fx - width - gap
    if left_x >= padding:
        return left_x, target_y, "left"

    return padding, max(48, fy - height - 12), "top"


def _draw_status_row(frame, layout, state, hand_count, active):
    tokens = [
        ("Tracking" if active else "Idle", LIVE_COLOR if active else TEXT_MUTED, True),
        (f"{hand_count} hand" + ("" if hand_count == 1 else "s"), TEXT_SECONDARY, False),
        (f"{state.fps_display:.0f} fps" if state.fps_display else "warming up", TEXT_MUTED, False),
    ]

    x = frame.shape[1] - layout.padding
    y = layout.padding + 14
    for index, (text, color, has_dot) in enumerate(tokens):
        text_width, _ = _measure_text(text, scale=0.35)
        x -= text_width
        if has_dot:
            dot_color = _mix_color(TEXT_MUTED, LIVE_COLOR, 0.4 + 0.6 * state.pulse() * state.activity)
            cv2.circle(frame, (x - 10, y - 4), 3, dot_color, -1, cv2.LINE_AA)
            _shadow_text(frame, text, (x, y), scale=0.35, color=color)
            x -= 18
        else:
            _shadow_text(frame, text, (x, y), scale=0.35, color=color)
            x -= 14
        if index < len(tokens) - 1:
            _shadow_text(frame, "|", (x, y), scale=0.32, color=TEXT_SOFT)
            x -= 12


def _draw_notice(frame, layout, state):
    if state.notice_alpha <= 0.02 or not state.notice_text:
        return

    color = _mix_color(TEXT_MUTED, ACCENT, state.notice_alpha)
    text_width, _ = _measure_text(state.notice_text, scale=0.35)
    x = (frame.shape[1] - text_width) // 2
    _shadow_text(frame, state.notice_text, (x, layout.padding + 14), scale=0.35, color=color)


def _draw_focus_guides(frame, state):
    if state.activity < 0.08:
        return

    x, y, width, height = (int(value) for value in state.focus_rect)
    color = _mix_color(LINE, LINE_ACTIVE, 0.25 + state.activity * 0.75)
    corner = max(12, int(min(width, height) * 0.12))

    cv2.line(frame, (x, y), (x + corner, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + corner), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + width, y), (x + width - corner, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + width, y), (x + width, y + corner), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + height), (x + corner, y + height), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + height), (x, y + height - corner), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + width, y + height), (x + width - corner, y + height), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + width, y + height), (x + width, y + height - corner), color, 1, cv2.LINE_AA)


def _draw_callout(frame, state, title, primary, secondary, meter, tertiary=""):
    focus_rect = state.focus_rect
    primary_width, _ = _measure_text(primary, scale=1.0, thickness=2)
    secondary_width, _ = _measure_text(secondary, scale=0.46)
    tertiary_width, _ = _measure_text(tertiary, scale=0.30) if tertiary else (0, 0)
    content_width = max(170, primary_width + secondary_width + 42, tertiary_width + 16)
    content_height = 54 if not tertiary else 68

    x, y, side = _callout_position(frame, focus_rect, content_width, content_height)
    x = int(x)
    y = int(y)
    content_width = int(content_width)
    content_height = int(content_height)

    fx, fy, fw, fh = (int(value) for value in focus_rect)
    anchor_y = fy + min(26, fh // 3)
    line_color = _mix_color(TEXT_SOFT, LINE_ACTIVE, 0.3 + state.activity * 0.7)

    if side == "right":
        start = (fx + fw, anchor_y)
        mid = (x - 10, anchor_y)
        end = (x, y + 16)
    elif side == "left":
        start = (fx, anchor_y)
        mid = (x + content_width + 10, anchor_y)
        end = (x + content_width, y + 16)
    else:
        start = (fx + fw // 2, fy)
        mid = (fx + fw // 2, y + content_height + 10)
        end = (x + 12, y + content_height)

    cv2.line(frame, start, mid, line_color, 1, cv2.LINE_AA)
    cv2.line(frame, mid, end, line_color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x + content_width, y), line_color, 1, cv2.LINE_AA)

    _shadow_text(frame, title, (x, y + 12), scale=0.30, color=TEXT_SECONDARY)
    _shadow_text(frame, primary, (x, y + 42), scale=1.0, color=TEXT_PRIMARY, thickness=2)
    _shadow_text(frame, secondary, (x + 56, y + 36), scale=0.46, color=ACCENT_SOFT)
    cv2.line(frame, (x, y + 48), (x + content_width - 4, y + 48), TRACK, 2, cv2.LINE_AA)
    fill_w = int((content_width - 4) * _clamp(meter, 0.0, 1.0))
    if fill_w > 0:
        cv2.line(frame, (x, y + 48), (x + fill_w, y + 48), ACCENT, 2, cv2.LINE_AA)

    if tertiary:
        tertiary, tertiary_scale = _fit_text(tertiary, content_width, 0.30, 0.27)
        _shadow_text(frame, tertiary, (x, y + content_height), scale=tertiary_scale, color=TEXT_MUTED)


def _draw_live_text(frame, layout, text, secondary_text):
    if text:
        text, text_scale = _fit_text(text, layout.bottom_text_width, 0.70, 0.40)
        _shadow_text(frame, text, (layout.padding, layout.bottom_y), scale=text_scale, color=TEXT_PRIMARY, thickness=2)

    secondary_text, secondary_scale = _fit_text(secondary_text, int(frame.shape[1] * 0.45), 0.30, 0.26)
    secondary_width, _ = _measure_text(secondary_text, scale=secondary_scale)
    _shadow_text(
        frame,
        secondary_text,
        (frame.shape[1] - layout.padding - secondary_width, layout.bottom_y),
        scale=secondary_scale,
        color=TEXT_MUTED,
    )


def draw_collection_overlay(
    frame,
    ui_state,
    detection_result,
    label,
    saved_count,
    target_samples,
    hand_labels,
    fps,
    notice_text=None,
):
    ui_state.step(
        frame=frame,
        fps=fps,
        confidence=saved_count / max(target_samples, 1),
        detection_result=detection_result,
        notice_text=notice_text,
        active=bool(hand_labels),
    )
    layout = _layout_for(frame)

    meta = f"{label}  {saved_count}/{target_samples}"
    _shadow_text(frame, meta, (layout.padding, layout.padding + 14), scale=0.44, color=TEXT_PRIMARY)
    tracked = "tracking" if hand_labels else "waiting"
    _shadow_text(frame, tracked, (layout.padding, layout.padding + 34), scale=0.30, color=TEXT_MUTED)
    _draw_status_row(frame, layout, ui_state, hand_count=len(hand_labels), active=bool(hand_labels))
    _draw_notice(frame, layout, ui_state)
    _draw_focus_guides(frame, ui_state)

    if hand_labels:
        _draw_callout(
            frame,
            state=ui_state,
            title="Label",
            primary=label,
            secondary=f"{saved_count}/{target_samples}",
            meter=ui_state.confidence_display,
            tertiary="Tracked: " + ", ".join(hand_labels).lower(),
        )

    _draw_live_text(
        frame,
        layout=layout,
        text="Ready" if hand_labels else "Place hands in frame",
        secondary_text="S save  |  ESC close",
    )


def draw_recognition_overlay(
    frame,
    ui_state,
    detection_result,
    prediction,
    confidence,
    top_predictions,
    display_text,
    fps,
    hand_count,
    notice_text=None,
    has_hand=True,
):
    ui_state.step(
        frame=frame,
        fps=fps,
        confidence=confidence,
        detection_result=detection_result,
        notice_text=notice_text,
        active=has_hand,
    )
    layout = _layout_for(frame)

    _draw_status_row(frame, layout, ui_state, hand_count=hand_count, active=has_hand)
    _draw_notice(frame, layout, ui_state)
    _draw_focus_guides(frame, ui_state)

    tertiary = ""
    if has_hand:
        alternates = top_predictions[1:3]
        if alternates:
            tertiary = "Next: " + " / ".join(f"{name} {prob * 100:.0f}%" for name, prob in alternates)

        _draw_callout(
            frame,
            state=ui_state,
            title="Prediction",
            primary=prediction,
            secondary=f"{ui_state.confidence_display * 100:.1f}% confidence",
            meter=ui_state.confidence_display,
            tertiary=tertiary,
        )

    live_text = display_text if display_text else " "
    _draw_live_text(
        frame,
        layout=layout,
        text=live_text,
        secondary_text="Swipe space  |  Both inward end  |  B delete  |  C clear  |  ESC close",
    )
