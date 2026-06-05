# 👋🏼 Real-Time Hand Gesture Recognition System

A real-time system that detects and interprets hand gestures from live video — converting them into text, letter by letter or word by word. Supports both static gestures (individual letters/signs held in place) and dynamic gestures (motion-based words like HELLO, THANK-YOU, YES, NO).

---

## Features

- **Dual-mode recognition** — static (per-frame letter classification) and dynamic (motion sequence word recognition) running simultaneously
- **Real-time inference** — asynchronous hand tracking keeps the UI responsive
- **Dual-hand tracking** — captures both hands via 21 landmarks × 2 hands = 126 features per frame
- **Sentence assembly** — letters build words, words build sentences with automatic capitalisation, spaces, and punctuation
- **Per-user personalisation** — collect and train models scoped to an individual user
- **Confidence gating** — predictions only commit when confidence ≥ 70% with a minimum margin over the second-best class

---

## Project Structure

```
gesture-recognition/
├── scripts/
│   ├── collect_static.py     # Collect static gesture training data (per-frame)
│   ├── collect_dynamic.py    # Collect dynamic gesture training data (sequences)
│   ├── train_static.py       # Train the MLP static classifier
│   ├── train_dynamic.py      # Train the temporal CNN dynamic model
│   ├── recognise.py          # Run real-time recognition (unified / static / dynamic)
│   ├── feature_extraction.py # Landmark feature engineering for both pipelines
│   ├── hand_tracking.py      # Async MediaPipe hand landmark detection
│   ├── personalization.py    # Per-user data and model path management
│   ├── camera.py             # Camera setup, frame preparation, FPS tracking
│   ├── ui.py                 # Overlay renderer (cinematic dark UI)
│   └── utils.py              # Shared constants and landmark normalisation
└── docs/
    └── demo.png
```

---

## How It Works

### Hand Landmark Detection

MediaPipe detects **21 keypoints per hand**, each with `(x, y, z)` coordinates. With two hands, each frame produces a **126-dimensional feature vector**. Landmarks are normalised — subtracted from the wrist position and divided by the max absolute value — making them position- and scale-independent.

---

### Static Gesture Recognition (Letters)

Each frame is classified independently:

1. Raw 126-dim landmark vector → feature extraction (palm normals, finger angles, tip distances)
2. `StandardScaler` normalisation
3. **MLPClassifier** `(256 → 128 → 64, ReLU)` outputs class probabilities
4. A prediction sticks only when it holds for **20 consecutive frames** at ≥ 70% confidence

**Augmentation during training:** Gaussian jitter · random scale ±12% · 2D rotation ±15° · horizontal mirror + hand-slot swap

---

### Dynamic Gesture Recognition (Words)

Sequences of 30 frames (~1 second at 30 fps) are classified as whole words:

1. Frames are buffered into a sliding window of length 30
2. A **motion gate** filters out idle frames (energy threshold + presence ratio)
3. Features are extracted per frame, scaled, then passed as `(30, feature_dim)` to a **Temporal CNN**
4. Architecture: `Conv1D → SeparableConv1D × 2 → GlobalAvg+MaxPool → Dense(96) → Softmax`
5. Converted to **TFLite** for fast CPU inference; falls back to Keras if TFLite is unavailable

**Augmentation during training:** Gaussian jitter · time-warp ±20% speed · random scale ±10% · temporal reversal · combined jitter + scale

---

### Sentence Assembly

The `SentenceAssembler` maintains a rolling output string:

| Gesture | Effect |
|---|---|
| Letter (A–Z, 0–9) | Appends to current word |
| `WORD_BREAK` | Commits word, inserts space |
| `SENTENCE_BREAK` | Commits word, appends `.`, capitalises next token |
| Dynamic word (e.g. `HELLO`) | Commits any partial word, appends as a whole token |

---

## Setup

### Install dependencies

```bash
pip install opencv-python mediapipe scikit-learn pandas numpy tensorflow
```

> **Note:** TensorFlow is required only for dynamic gesture training and inference. Static-only mode works without it.

---

## Usage

### Step 1 — Collect static gesture data

```bash
python scripts/collect_static.py
```

- Shows the current gesture label on screen with a live camera feed
- Press **S** to capture a sample frame
- Recommended: **≥ 150 samples per gesture**
- Data is saved to `data_static/`

### Step 2 — Collect dynamic gesture data

```bash
python scripts/collect_dynamic.py
```

- Auto-countdown (3→2→1) then records a 30-frame sequence
- Auto-loops after each capture — just keep signing
- Press **SPACE** to pause, **ENTER** to skip to next gesture, **ESC** to stop
- Recommended: **≥ 150 sequences per gesture** (300+ for production)
- Gestures to collect are defined in the `GESTURES` list at the top of the file
- Data is saved to `data_dynamic/`

### Step 3 — Train static model

```bash
python scripts/train_static.py
# Options:
python scripts/train_static.py --skip-cv          # skip 5-fold cross-validation (faster)
python scripts/train_static.py --user-id alice    # train a personalised model
```

Outputs `static_model.pkl` (model + scaler + label encoder).

### Step 4 — Train dynamic model

```bash
python scripts/train_dynamic.py
# Options:
python scripts/train_dynamic.py --epochs 80
python scripts/train_dynamic.py --user-id alice
python scripts/train_dynamic.py --skip-tflite     # skip TFLite conversion
```

Outputs `dynamic_model.tflite` (fast inference) + `dynamic_model.pkl` (metadata) + `dynamic_model.keras` (fallback).

### Step 5 — Run recognition

```bash
python scripts/recognise.py                   # unified (recommended)
python scripts/recognise.py --mode static     # static only
python scripts/recognise.py --mode dynamic    # dynamic only
python scripts/recognise.py --camera 1        # use a different camera index
python scripts/recognise.py --user-id alice   # load personalised model
```

**In-session controls:**

| Key | Action |
|---|---|
| `B` | Backspace (remove last letter or word) |
| `C` | Clear all text |
| `M` | Toggle mode (unified → static → dynamic → unified) |
| `ESC` | Quit |

---

## Data Format

### Static (`data_static.csv`)

Each row is one captured frame:

| x1 | y1 | z1 | … | x42 | y42 | z42 | label |
|---|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … | A |

- **126 float columns** — 21 landmarks × 2 hands × 3 coordinates
- Absent hand slots are filled with zeros
- **1 label column** — gesture class string

### Dynamic (`data_dynamic.csv`)

Each row is one 30-frame sequence:

- **3780 float columns** — 30 frames × 126 values per frame (flattened)
- **1 label column** — gesture class string (e.g. `HELLO`, `THANK-YOU`)

---

## Per-User Personalisation

Run any script with `--user-id <name>` to scope data and models to that user:

```
user_profiles/<user_id>/
├── static/
│   └── <LABEL>/data.csv
├── dynamic/
│   └── <LABEL>/data.csv
├── static_model.pkl
├── dynamic_model.pkl
├── dynamic_model.tflite
└── dynamic_model.keras
```

Training merges base data + user data so the personalised model learns from everything.

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand tracking | MediaPipe |
| Image processing | OpenCV |
| Static model | scikit-learn MLPClassifier |
| Dynamic model | TensorFlow / Keras (Temporal CNN) |
| Inference runtime | TFLite |
| Data processing | NumPy, pandas |
| UI rendering | OpenCV overlays |

---

## Model Details

### Static MLP

| Parameter | Value |
|---|---|
| Architecture | 256 → 128 → 64 (ReLU) |
| Regularisation | L2 α = 1e-4 |
| Optimiser | Adam lr = 0.001 |
| Max iterations | 1500 (early stopping) |
| Confidence threshold | ≥ 70%, margin ≥ 5% |
| Hold frames to commit | 20 consecutive frames |

### Dynamic Temporal CNN

| Parameter | Value |
|---|---|
| Input shape | (30 frames, feature_dim) |
| Layers | Conv1D(48) → SepConv1D(64) → SepConv1D(96) → Dense(96) |
| Pooling | GlobalAvg + GlobalMax concatenated |
| Regularisation | SpatialDropout1D(0.2), Dropout(0.35), L2 |
| Optimiser | Adam lr = 0.001 with ReduceLROnPlateau |
| Confidence threshold | ≥ 72%, margin ≥ 8% |
| Hold frames to commit | 8 consecutive frames (cooldown: 20 frames) |

---

<div align="center">

[![Sridevi S](https://img.shields.io/badge/Sridevi%20S-111111?style=for-the-badge&logo=github&logoColor=white)](https://github.com/23f2003639)

</div>
