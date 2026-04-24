import csv
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

# ─────────────────────────────────────────
#  CONFIG — change LABEL before each session
# ─────────────────────────────────────────
LABEL          = "F"
CSV_FILE       = "gesture_data.csv"
TARGET_SAMPLES = 150
CAMERA_INDEX   = 0

FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
CONTRAST_ALPHA = 1.03
CONTRAST_BETA  = 8

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILE = Path(__file__).resolve().parent / "hand_landmarker.task"


# ─────────────────────────────────────────
#  DOWNLOAD MODEL IF MISSING
# ─────────────────────────────────────────
def download_model() -> None:
    if MODEL_FILE.exists():
        return
    print("Downloading hand_landmarker.task (~25 MB) ...")
    tmp = MODEL_FILE.with_suffix(".download")
    try:
        urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_FILE)
        print("Download complete.")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Download failed. Check your internet connection.") from exc


# ─────────────────────────────────────────
#  NORMALISE
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
#  EXTRACT LEFT / RIGHT HAND DATA
# ─────────────────────────────────────────
def extract_hand_data(result) -> dict:
    hand_entries = []
    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[index] if index < len(result.handedness) else []
        label  = handedness[0].category_name if handedness else None
        # ── flip label to match mirrored frame ──
        if label == "Left":
            label = "Right"
        elif label == "Right":
            label = "Left"
        score  = handedness[0].score         if handedness else 0.0
        values = []
        for lm in landmarks:
            values.extend([lm.x, lm.y, lm.z])
        hand_entries.append({
            "label":   label,
            "score":   score,
            "wrist_x": landmarks[0].x if landmarks else 0.5,
            "values":  values,
        })

    if not hand_entries:
        return {}

    hand_data = {}
    remaining = list(hand_entries)

    for preferred in ("Left", "Right"):
        matches = [e for e in remaining if e["label"] == preferred]
        if matches:
            best = max(matches, key=lambda e: e["score"])
            hand_data[preferred] = best["values"]
            remaining.remove(best)

    remaining.sort(key=lambda e: e["wrist_x"])
    if "Left"  not in hand_data and remaining:
        hand_data["Left"]  = remaining.pop(0)["values"]
    if "Right" not in hand_data and remaining:
        hand_data["Right"] = remaining.pop(0)["values"]

    return hand_data


# ─────────────────────────────────────────
#  DRAW LANDMARKS  (no landmark_pb2 needed)
# ─────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, result) -> None:
    h, w = frame.shape[:2]
    for hand_landmarks in result.hand_landmarks:
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main() -> None:
    download_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    saved_count        = 0
    landmarks_combined = []
    hand_labels        = []
    timestamp_ms       = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        csv_path = Path(CSV_FILE)
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)

            while True:
                success, frame = cap.read()
                if not success:
                    continue

                # identical preprocessing to what model.py will use
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = cv2.convertScaleAbs(frame, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)
                timestamp_ms += 33

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = landmarker.detect_for_video(mp_image, timestamp_ms)

                hand_data          = extract_hand_data(result)
                landmarks_combined = []
                hand_labels        = list(hand_data.keys())

                if hand_data:
                    draw_landmarks(frame, result)
                    left  = normalize(hand_data.get("Left",  [0.0] * 63))
                    right = normalize(hand_data.get("Right", [0.0] * 63))
                    landmarks_combined = left + right

                # HUD
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Label : {LABEL}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Saved : {saved_count}/{TARGET_SAMPLES}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "S = save   ESC = quit",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if hand_labels:
                    cv2.putText(frame, f"Hands : {hand_labels}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                else:
                    cv2.putText(frame, "No hands detected",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                bar_w    = w - 40
                progress = int((saved_count / TARGET_SAMPLES) * bar_w)
                cv2.rectangle(frame, (20, h - 30), (20 + progress, h - 10), (0, 255, 100), -1)
                cv2.rectangle(frame, (20, h - 30), (20 + bar_w,    h - 10), (255, 255, 255), 1)

                cv2.imshow("Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    if landmarks_combined:
                        writer.writerow(landmarks_combined + [LABEL])
                        f.flush()
                        saved_count += 1
                        print(f"Saved {saved_count}/{TARGET_SAMPLES} — {LABEL}")
                        flash = frame.copy()
                        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), -1)
                        cv2.addWeighted(flash, 0.2, frame, 0.8, 0, frame)
                        cv2.imshow("Data Collector", frame)
                        cv2.waitKey(80)
                    else:
                        print("No hands detected — nothing saved")

                elif key == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total saved: {saved_count}")


if __name__ == "__main__":
    main()