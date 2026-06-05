"""
collect_dynamic.py — Auto-loop dynamic gesture data collection.

HOW IT WORKS:
  - Opens camera, shows the current gesture label top-left.
  - Countdown 3→2→1 starts automatically when you enter CAPTURING state.
  - After each capture (good or bad), a 1-second pause then auto-starts
    the next countdown so you can keep signing without touching the keyboard.
  - When target count is reached, pauses and waits for ENTER to move on.

CONTROLS:
  SPACE       — pause / resume the auto-loop
  ENTER       — skip to next gesture  (or confirm move-on when target reached)
  ESC         — stop and save everything collected so far

ONE ROW = 30 frames × 126 values (3780 floats) + label
"""

import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from camera import CameraConfig, FrameRateTracker, open_camera, prepare_frame
from feature_extraction import sequence_motion_energy, sequence_presence_ratio
from hand_tracking import AsyncHandTracker, draw_hand_landmarks, ensure_hand_landmarker_model
from ui import OverlayState, draw_collection_overlay
from utils import normalize, SEQUENCE_LENGTH, LANDMARK_DIM

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_CSV    = Path(__file__).resolve().parent / "data_dynamic.csv"
CAMERA_INDEX  = 0
CAMERA_CONFIG = CameraConfig()
WINDOW_NAME   = "Dynamic Gesture Collection"

# Quality gates
MIN_PRESENCE = 0.75
MIN_MOTION   = 0.025

# Timing (seconds)
COUNTDOWN_SECS   = 3      # 3→2→1 before capture starts
POST_SAVE_PAUSE  = 1.0    # pause after each save before next countdown
POST_REJECT_PAUSE = 1.5   # slightly longer pause after rejection

# ── GESTURES TO COLLECT ───────────────────────────────────────────────────────
# Format: (label, target_count)
# Add or remove gestures here. Script resumes from wherever you left off.
GESTURES = [
    ("HELLO",       100),
    ("THANK-YOU",   100),
    ("YES",         100),
    ("NO",          100),
    ("PLEASE",      100),
    ("SORRY",       100),
    ("HELP",        100),
    ("GOOD",        100),
    ("BAD",         100),
    ("MORE",        100),
]
# ─────────────────────────────────────────────────────────────────────────────

# States
S_WAIT_START  = "wait_start"   # first launch — waiting for first ENTER/SPACE
S_COUNTDOWN   = "countdown"    # 3→2→1 showing
S_CAPTURING   = "capturing"    # recording frames
S_POST        = "post"         # brief pause after save/reject before next loop
S_TARGET_MET  = "target_met"   # target reached, waiting for ENTER
S_PAUSED      = "paused"       # user pressed SPACE


def _count_existing(label: str, path: Path, user_id: str | None = None) -> int:
    from personalization import count_label_samples
    return count_label_samples("dynamic", label, user_id)

def _save(label: str, frames: list, path: Path, user_id: str | None = None) -> None:
    from personalization import append_dynamic_row
    flat = [v for frame_data in frames for v in frame_data]
    append_dynamic_row(label, flat + [label], user_id)

def _draw_ui(
    frame, overlay, detection_result, label, saved, target,
    hand_labels, fps, state, countdown_remaining, buf_len, notice
):
    """Draw all overlays on the frame."""
    h, w = frame.shape[:2]
    pad = 16

    # Base overlay (landmarks, status strip, etc.)
    draw_collection_overlay(
        frame,
        state=overlay,
        detection_result=detection_result,
        label=label,
        saved_count=saved,
        target_samples=target,
        hand_labels=hand_labels,
        fps=fps,
        notice=notice,
        buffer_fill=buf_len if state == S_CAPTURING else 0,
        sequence_length=SEQUENCE_LENGTH if state == S_CAPTURING else 0,
    )

    font   = cv2.FONT_HERSHEY_SIMPLEX
    bold   = cv2.FONT_HERSHEY_DUPLEX

    # ── Gesture label — large, top-left ──────────────────────────────────
    cv2.putText(frame, label, (pad, pad + 44),
                bold, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (pad, pad + 44),
                bold, 1.4, (0, 220, 255), 1, cv2.LINE_AA)

    # ── Count below label ────────────────────────────────────────────────
    count_txt = f"{saved} / {target}"
    cv2.putText(frame, count_txt, (pad, pad + 72),
                font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Big centred countdown ─────────────────────────────────────────────
    if state == S_COUNTDOWN and countdown_remaining > 0:
        digit = str(int(countdown_remaining) + 1)
        (tw, th), _ = cv2.getTextSize(digit, bold, 5.0, 6)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.putText(frame, digit, (cx, cy), bold, 5.0, (0, 0, 0),    10, cv2.LINE_AA)
        cv2.putText(frame, digit, (cx, cy), bold, 5.0, (0, 200, 255), 6, cv2.LINE_AA)

    # ── State banner ──────────────────────────────────────────────────────
    if state == S_WAIT_START:
        msg = "Press ENTER or SPACE to begin"
        (tw, _), _ = cv2.getTextSize(msg, font, 0.7, 2)
        cv2.putText(frame, msg, ((w - tw) // 2, h // 2 + 10),
                    font, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    elif state == S_PAUSED:
        msg = "PAUSED — press SPACE to resume"
        (tw, _), _ = cv2.getTextSize(msg, font, 0.65, 2)
        cv2.rectangle(frame, (0, h//2 - 30), (w, h//2 + 20), (30, 30, 30), -1)
        cv2.putText(frame, msg, ((w - tw) // 2, h // 2 + 8),
                    font, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    elif state == S_TARGET_MET:
        msg = f"'{label}' done!  Press ENTER for next gesture"
        (tw, _), _ = cv2.getTextSize(msg, font, 0.6, 2)
        cv2.rectangle(frame, (0, h//2 - 30), (w, h//2 + 20), (0, 80, 0), -1)
        cv2.putText(frame, msg, ((w - tw) // 2, h // 2 + 8),
                    font, 0.6, (100, 255, 100), 2, cv2.LINE_AA)

    elif state == S_CAPTURING:
        cv2.putText(frame, "RECORDING", (w - 160, 36),
                    font, 0.65, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "RECORDING", (w - 160, 36),
                    font, 0.65, (0, 80, 255), 1, cv2.LINE_AA)

    # ── Controls hint — bottom right ──────────────────────────────────────
    hints = "SPACE=pause   ENTER=next   ESC=quit"
    (hw, _), _ = cv2.getTextSize(hints, font, 0.32, 1)
    cv2.putText(frame, hints, (w - hw - pad, h - pad - 20),
                font, 0.32, (120, 120, 120), 1, cv2.LINE_AA)


def collect(camera_index: int = CAMERA_INDEX, user_id: str | None = None) -> None:
    ensure_hand_landmarker_model()

    output_csv = OUTPUT_CSV
    cap          = open_camera(camera_index, CAMERA_CONFIG)
    hand_tracker = AsyncHandTracker(
        num_hands=2,
        tracking_size=(CAMERA_CONFIG.tracking_width, CAMERA_CONFIG.tracking_height),
    )
    fps_tracker  = FrameRateTracker()
    overlay      = OverlayState()

    gesture_idx  = 0
    state        = S_WAIT_START
    frame_buf    = deque(maxlen=SEQUENCE_LENGTH)
    countdown_end = 0.0
    post_end      = 0.0
    total_saved   = 0
    total_rejected = 0
    notice_text   = None
    notice_frames = 0
    paused_from   = S_WAIT_START   # state to return to after unpausing

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))

    print(f"\n{'='*60}")
    print(f"  DYNAMIC GESTURE COLLECTION")
    print(f"  Output → {output_csv}")
    print(f"  SPACE=pause  ENTER=skip/next  ESC=quit")
    print(f"{'='*60}\n")

    try:
        while gesture_idx < len(GESTURES):
            label, target = GESTURES[gesture_idx]
            already_have  = _count_existing(label, output_csv, user_id) 

            # Skip if already at target
            if already_have >= target and state not in (S_TARGET_MET,):
                state = S_TARGET_MET

            ok, frame = cap.read()
            if not ok:
                continue
            frame = prepare_frame(frame, CAMERA_CONFIG)
            fps   = fps_tracker.tick()
            ts    = int(time.monotonic() * 1000)

            hand_tracker.submit(frame, ts)
            det_result, hand_data, _ = hand_tracker.latest(ts, max_age_ms=200)
            if det_result:
                draw_hand_landmarks(frame, det_result)

            hand_labels = list(hand_data.keys()) if hand_data else []
            now = time.monotonic()

            # ── STATE MACHINE ─────────────────────────────────────────────
            countdown_remaining = 0.0

            if state == S_WAIT_START:
                pass  # waiting for keypress

            elif state == S_PAUSED:
                pass  # waiting for SPACE

            elif state == S_TARGET_MET:
                pass  # waiting for ENTER

            elif state == S_COUNTDOWN:
                countdown_remaining = countdown_end - now
                if countdown_remaining <= 0:
                    state = S_CAPTURING
                    frame_buf.clear()
                    print(f"  ▶ Recording '{label}'…")

            elif state == S_CAPTURING:
                if hand_data:
                    left  = normalize(hand_data.get("Left",  [0.0] * 63))
                    right = normalize(hand_data.get("Right", [0.0] * 63))
                    frame_buf.append(left + right)
                else:
                    frame_buf.append([0.0] * LANDMARK_DIM)

                if len(frame_buf) >= SEQUENCE_LENGTH:
                    seq    = list(frame_buf)
                    seq_np = np.array(seq, dtype=np.float32)
                    presence = sequence_presence_ratio(seq_np)
                    motion   = sequence_motion_energy(seq_np)

                    if presence >= MIN_PRESENCE and motion >= MIN_MOTION:
                        _save(label, seq, output_csv, user_id) 
                        total_saved  += 1
                        already_have += 1
                        notice_text   = f"✓  Saved  ({already_have}/{target})"
                        notice_frames = 45
                        post_end      = now + POST_SAVE_PAUSE
                        print(f"  ✓  {label}  [{already_have}/{target}]  motion={motion:.3f}")
                    else:
                        total_rejected += 1
                        reason = (
                            f"motion too low ({motion:.3f} < {MIN_MOTION})"
                            if motion < MIN_MOTION
                            else f"hand missing ({presence:.0%} present)"
                        )
                        notice_text   = f"✗  Rejected — {reason}"
                        notice_frames = 50
                        post_end      = now + POST_REJECT_PAUSE
                        print(f"  ✗  Rejected: {reason}")

                    frame_buf.clear()

                    if already_have >= target:
                        state = S_TARGET_MET
                        print(f"\n  ✓  '{label}' complete — press ENTER for next\n")
                    else:
                        state = S_POST  # brief pause then auto-countdown

            elif state == S_POST:
                if now >= post_end:
                    # Continue recording immediately
                    state = S_CAPTURING
                    frame_buf.clear()
            # ── Draw ──────────────────────────────────────────────────────
            active_notice = notice_text if notice_frames > 0 else None
            _draw_ui(
                frame, overlay, det_result,
                label, already_have, target,
                hand_labels, fps,
                state, countdown_remaining,
                len(frame_buf), active_notice,
            )
            if notice_frames > 0:
                notice_frames -= 1

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ──────────────────────────────────────────────
            if key == 27:  # ESC — quit
                print("\n  Stopping collection…")
                break

            elif key == 32:  # SPACE — pause / resume
                if state == S_PAUSED:
                    print("  ▶ Resumed")
                    state = paused_from if paused_from != S_PAUSED else S_COUNTDOWN
                    if state == S_COUNTDOWN:
                        countdown_end = time.monotonic() + COUNTDOWN_SECS
                elif state not in (S_WAIT_START, S_TARGET_MET):
                    paused_from = state
                    state       = S_PAUSED
                    frame_buf.clear()
                    print("  ⏸  Paused")
                else:
                    # SPACE on wait_start acts like begin
                    state         = S_COUNTDOWN
                    countdown_end = time.monotonic() + COUNTDOWN_SECS

            elif key in (13, 10):  # ENTER — next gesture / begin
                if state == S_WAIT_START:
                    state         = S_COUNTDOWN
                    countdown_end = time.monotonic() + COUNTDOWN_SECS
                    print(f"  Starting '{label}'…")
                elif state == S_TARGET_MET:
                    gesture_idx += 1

                    if gesture_idx < len(GESTURES):
                        state = S_COUNTDOWN
                        countdown_end = time.monotonic() + COUNTDOWN_SECS
                        print(f"  → Next: {GESTURES[gesture_idx][0]}")
                    else:
                        state = S_WAIT_START

                    frame_buf.clear()
                    if gesture_idx < len(GESTURES):
                        print(f"  → Next: {GESTURES[gesture_idx][0]}")
                else:
                    # Skip current gesture
                    print(f"  ⏭  Skipping '{label}'")
                    gesture_idx += 1
                    state        = S_WAIT_START
                    frame_buf.clear()

    finally:
        cap.release()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print(f"\n{'='*60}")
        print(f"  Session complete")
        print(f"  Saved    : {total_saved}")
        print(f"  Rejected : {total_rejected}")
        print(f"  Output   : {output_csv}")
        print(f"\n  Next step: python train_dynamic.py")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",  type=int, default=CAMERA_INDEX)
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()
    collect(camera_index=args.camera, user_id=args.user_id)
