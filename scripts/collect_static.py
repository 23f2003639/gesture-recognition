"""
collect_static.py — Auto-loop static gesture data collection.

HOW IT WORKS:
  - Opens camera, shows the current gesture label top-left.
  - Countdown 3→2→1 starts automatically.
  - During CAPTURING, every frame where a hand is detected is saved
    automatically — no button pressing needed.
  - After reaching the per-gesture target, pauses and waits for ENTER.
  - Moves through all gestures in order.

CONTROLS:
  SPACE   — pause / resume the auto-loop
  ENTER   — skip to next gesture  (or confirm move-on when target reached)
  ESC     — stop and save everything collected so far

ONE ROW = 126 landmark floats + label
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from camera import CameraConfig, FrameRateTracker, open_camera, prepare_frame
from feature_extraction import canonical_label
from hand_tracking import AsyncHandTracker, draw_hand_landmarks, ensure_hand_landmarker_model
from personalization import ensure_user_dir, user_data_path
from ui import OverlayState, draw_collection_overlay
from utils import normalize, LANDMARK_DIM

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUTPUT_CSV    = Path(__file__).resolve().parent / "data_static.csv"
CAMERA_INDEX  = 0
CAMERA_CONFIG = CameraConfig()
WINDOW_NAME   = "Static Gesture Collection"

# Frames to skip between saves — prevents saving 200 near-identical frames.
# At 30fps, SAVE_EVERY=2 saves ~15 samples/sec. Vary your hand slightly.
SAVE_EVERY = 2

# Countdown before capture starts (seconds)
COUNTDOWN_SECS = 3

# Brief pause between gestures when auto-advancing (seconds)
POST_GESTURE_PAUSE = 1.5

frame = 200;

# ── GESTURES TO COLLECT ───────────────────────────────────────────────────────
# Script resumes from wherever you left off (counts existing rows in CSV).
GESTURES = [
    ("A", frame), ("B", frame), ("C", frame), ("D", frame), ("E", frame),
    ("F", frame), ("G", frame), ("H", frame), ("I", frame), ("J", frame),
    ("K", frame), ("L", frame), ("M", frame), ("N", frame), ("O", frame),
    ("P", frame), ("Q", frame), ("R", frame), ("S", frame), ("T", frame),
    ("U", frame), ("V", frame), ("W", frame), ("X", frame), ("Y", frame),
    ("Z", frame),
    ("0", frame), ("1", frame), ("2", frame), ("3", frame), ("4", frame),
    ("5", frame), ("6", frame), ("7", frame), ("8", frame), ("9", frame),
    ("10", frame),
    ("WORD_BREAK",     frame),
    ("SENTENCE_BREAK", frame),
]
# ─────────────────────────────────────────────────────────────────────────────

# States
S_WAIT_START = "wait_start"
S_COUNTDOWN  = "countdown"
S_CAPTURING  = "capturing"
S_POST       = "post"
S_TARGET_MET = "target_met"
S_PAUSED     = "paused"


def _count_existing(label: str, path: Path, user_id: str | None = None) -> int:
    from personalization import count_label_samples
    return count_label_samples("static", label, user_id)


def _save(label: str, hand_data: dict, path: Path, user_id: str | None = None) -> None:
    from personalization import append_static_row
    left  = normalize(hand_data.get("Left",  [0.0] * 63))
    right = normalize(hand_data.get("Right", [0.0] * 63))
    row   = left + right + [label]
    append_static_row(label, row, user_id)


def _draw_ui(
    frame, overlay, detection_result, label, saved, target,
    hand_labels, fps, state, countdown_remaining, notice
):
    h, w = frame.shape[:2]
    pad  = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    bold = cv2.FONT_HERSHEY_DUPLEX

    # Base overlay from ui.py (landmarks, status strip, progress bar)
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
        buffer_fill=0,
        sequence_length=0,   # static — no sequence bar
    )

    # ── Large gesture label top-left ─────────────────────────────────────
    cv2.putText(frame, label, (pad, pad + 44),
                bold, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (pad, pad + 44),
                bold, 1.4, (0, 220, 255),   1, cv2.LINE_AA)

    # ── Count below label ────────────────────────────────────────────────
    cv2.putText(frame, f"{saved} / {target}", (pad, pad + 72),
                font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Big centred countdown ─────────────────────────────────────────────
    if state == S_COUNTDOWN and countdown_remaining > 0:
        digit = str(int(countdown_remaining) + 1)
        (tw, th), _ = cv2.getTextSize(digit, bold, 5.0, 6)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.putText(frame, digit, (cx, cy), bold, 5.0, (0,   0,   0),   10, cv2.LINE_AA)
        cv2.putText(frame, digit, (cx, cy), bold, 5.0, (0, 200, 255),    6, cv2.LINE_AA)

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
        # Pulsing green dot to show live saving
        cv2.circle(frame, (w - pad - 8, 28), 7, (0, 220, 80), -1, cv2.LINE_AA)
        cv2.putText(frame, "SAVING", (w - pad - 80, 36),
                    font, 0.55, (0, 220, 80), 1, cv2.LINE_AA)

    # ── Controls hint bottom-right ────────────────────────────────────────
    hints = "SPACE=pause   ENTER=next   ESC=quit"
    (hw, _), _ = cv2.getTextSize(hints, font, 0.32, 1)
    cv2.putText(frame, hints, (w - hw - pad, h - pad - 20),
                font, 0.32, (120, 120, 120), 1, cv2.LINE_AA)


def collect(camera_index: int = CAMERA_INDEX, user_id: str | None = None) -> None:
    ensure_hand_landmarker_model()

    output_csv = OUTPUT_CSV
    if user_id:
        ensure_user_dir(user_id)
        output_csv = user_data_path(user_id, "static")

    cap          = open_camera(camera_index, CAMERA_CONFIG)
    hand_tracker = AsyncHandTracker(
        num_hands=2,
        tracking_size=(CAMERA_CONFIG.tracking_width, CAMERA_CONFIG.tracking_height),
    )
    fps_tracker   = FrameRateTracker()
    overlay       = OverlayState()

    gesture_idx    = 0
    state          = S_WAIT_START
    countdown_end  = 0.0
    post_end       = 0.0
    frame_counter  = 0       # counts every captured frame; used for SAVE_EVERY
    total_saved    = 0
    total_skipped  = 0
    notice_text    = None
    notice_frames  = 0
    paused_from    = S_WAIT_START

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    print(f"\n{'='*60}")
    print(f"  STATIC GESTURE COLLECTION")
    print(f"  Output → {output_csv}")
    print(f"  {len(GESTURES)} gestures  |  SAVE_EVERY={SAVE_EVERY} frames")
    print(f"  SPACE=pause  ENTER=skip/next  ESC=quit")
    print(f"{'='*60}\n")

    try:
        while gesture_idx < len(GESTURES):
            label, target = GESTURES[gesture_idx]
            label         = canonical_label(label)
            already_have  = _count_existing(label, output_csv, user_id)

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
            countdown_remaining = 0.0

            # ── STATE MACHINE ──────────────────────────────────────────────
            if state == S_WAIT_START:
                pass

            elif state == S_PAUSED:
                pass

            elif state == S_TARGET_MET:
                pass

            elif state == S_COUNTDOWN:
                countdown_remaining = countdown_end - now
                if countdown_remaining <= 0:
                    state         = S_CAPTURING
                    frame_counter = 0
                    print(f"  ▶ Capturing '{label}'…")

            elif state == S_CAPTURING:
                if hand_data:
                    frame_counter += 1
                    if frame_counter % SAVE_EVERY == 0:
                        _save(label, hand_data, output_csv, user_id)
                        total_saved  += 1
                        already_have += 1
                        notice_text   = f"✓  Saved  ({already_have}/{target})"
                        notice_frames = 20

                        if already_have >= target:
                            state = S_TARGET_MET
                            print(f"\n  ✓  '{label}' complete — press ENTER for next\n")
                else:
                    # No hand — show a reminder but keep capturing state
                    notice_text   = "Show your hand!"
                    notice_frames = 15

            elif state == S_POST:
                if now >= post_end:
                    state         = S_COUNTDOWN
                    countdown_end = now + COUNTDOWN_SECS

            # ── Draw ──────────────────────────────────────────────────────
            active_notice = notice_text if notice_frames > 0 else None
            _draw_ui(
                frame, overlay, det_result,
                label, already_have, target,
                hand_labels, fps,
                state, countdown_remaining, active_notice,
            )
            if notice_frames > 0:
                notice_frames -= 1

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ──────────────────────────────────────────────
            if key == 27:   # ESC
                print("\n  Stopping collection…")
                break

            elif key == 32:   # SPACE — pause / resume
                if state == S_PAUSED:
                    print("  ▶ Resumed")
                    state = paused_from if paused_from != S_PAUSED else S_COUNTDOWN
                    if state == S_COUNTDOWN:
                        countdown_end = time.monotonic() + COUNTDOWN_SECS
                elif state not in (S_WAIT_START, S_TARGET_MET):
                    paused_from = state
                    state       = S_PAUSED
                    print("  ⏸  Paused")
                else:
                    # SPACE on wait_start = begin
                    state         = S_COUNTDOWN
                    countdown_end = time.monotonic() + COUNTDOWN_SECS

            elif key in (13, 10):   # ENTER
                if state == S_WAIT_START:
                    state         = S_COUNTDOWN
                    countdown_end = time.monotonic() + COUNTDOWN_SECS
                    print(f"  Starting '{label}'…")
                elif state == S_TARGET_MET:
                    gesture_idx += 1
                    frame_counter = 0
                    if gesture_idx < len(GESTURES):
                        state         = S_COUNTDOWN
                        countdown_end = time.monotonic() + COUNTDOWN_SECS
                        print(f"  → Next: {GESTURES[gesture_idx][0]}")
                    else:
                        print("  All gestures complete!")
                        break
                else:
                    # Skip current gesture
                    print(f"  ⏭  Skipping '{label}'")
                    gesture_idx  += 1
                    state         = S_WAIT_START
                    frame_counter = 0

    finally:
        cap.release()
        hand_tracker.close()
        cv2.destroyAllWindows()
        print(f"\n{'='*60}")
        print(f"  Session complete")
        print(f"  Saved   : {total_saved}")
        print(f"  Output  : {output_csv}")
        print(f"\n  Next step: python train_static.py")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",  type=int, default=CAMERA_INDEX)
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()
    collect(camera_index=args.camera, user_id=args.user_id)
