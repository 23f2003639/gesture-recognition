"""
train_dynamic.py — Train the dynamic (sequence) gesture model.

Architecture: lightweight temporal CNN → Dense  (TensorFlow/Keras)
Optimized for small datasets (you currently have only 200 sequences for
2 gestures — HELLO and HEY). This version adds stronger regularisation
and data augmentation to compensate.

TARGET DATASET SIZE FOR PRODUCTION:
  ≥ 150 samples per gesture  (300+ recommended)
  ≥ 5 different gesture classes
  Collected from multiple signers and camera angles if possible.

HOW TO RUN:
    python train_dynamic.py
    python train_dynamic.py --epochs 80   # train longer
    python train_dynamic.py --user-id alice

OUTPUTS:
    dynamic_model.tflite  — fast CPU inference model (used by recognise.py)
    dynamic_model.pkl     — metadata bundle (scaler, label_encoder, gate params)
    dynamic_model.keras   — Keras fallback
"""

import argparse
import os
import pickle
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAINING_TMP_DIR = BASE_DIR / ".training_tmp"
TRAINING_TMP_DIR.mkdir(exist_ok=True)

os.environ.setdefault("TMP",  str(TRAINING_TMP_DIR))
os.environ.setdefault("TEMP", str(TRAINING_TMP_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",       "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS",      "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS",      "1")
os.environ.setdefault("OMP_NUM_THREADS",             "2")
tempfile.tempdir = str(TRAINING_TMP_DIR)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from feature_extraction import (
    DYNAMIC_FEATURE_VERSION,
    canonicalize_labels,
    dynamic_feature_dim,
    extract_dynamic_batch,
    sequence_motion_energy,
    sequence_presence_ratio,
)
from personalization import (
    ensure_user_dir,
    load_all_data,         
    user_data_path,
    user_dynamic_keras_path,
    user_dynamic_tflite_path,
    user_model_path,
)
from utils import SEQUENCE_LENGTH, LANDMARK_DIM

DATA_CSV     = BASE_DIR / "data_dynamic.csv"
MODEL_TFLITE = BASE_DIR / "dynamic_model.tflite"
MODEL_PKL    = BASE_DIR / "dynamic_model.pkl"
MODEL_KERAS  = BASE_DIR / "dynamic_model.keras"

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["TF_NUM_INTRAOP_THREADS"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ["TF_NUM_INTEROP_THREADS"])
    )
except RuntimeError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_sequence(seq: np.ndarray) -> list[np.ndarray]:
    """
    Returns 5 augmented copies of a (30, 126) sequence.
    More augmentation is critical when you have < 300 samples per class.
    """
    aug = [seq]

    # 1. Gaussian landmark jitter
    aug.append((seq + np.random.normal(0, 0.012, seq.shape)).astype(np.float32))

    # 2. Time-warp ±20% speed
    n      = len(seq)
    factor = np.random.uniform(0.8, 1.2)
    src    = np.clip(np.linspace(0, n - 1, int(n * factor)), 0, n - 1).astype(int)
    warped = seq[src]
    if len(warped) >= n:
        warped = warped[:n]
    else:
        warped = np.vstack([warped, np.tile(warped[-1], (n - len(warped), 1))])
    aug.append(warped.astype(np.float32))

    # 3. Random scale ±10%
    aug.append((seq * np.random.uniform(0.90, 1.10)).astype(np.float32))

    # 4. Reverse temporal order (sign performed "backwards") — strong regulariser
    aug.append(seq[::-1].copy().astype(np.float32))

    # 5. Combined: jitter + scale
    aug.append(
        (seq * np.random.uniform(0.93, 1.07)
         + np.random.normal(0, 0.008, seq.shape)).astype(np.float32)
    )

    return aug


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, input_shape: tuple[int, int]) -> keras.Model:
    """
    Temporal CNN with stronger regularisation for small datasets.

    Changes vs original:
      - Dropout raised to 0.35 on the dense layer
      - SpatialDropout1D raised to 0.20
      - L2 weight decay added to Conv layers
      - Global avg + max pooling concatenation kept (works well empirically)

    This architecture trains well with ≥ 150 samples/class after augmentation.
    """
    reg = keras.regularizers.l2(1e-4)

    inp = keras.Input(shape=input_shape)

    x = layers.Conv1D(48, kernel_size=3, activation="relu", padding="same",
                      kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv1D(64, kernel_size=5, activation="relu", padding="same",
                                depthwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.SpatialDropout1D(0.20)(x)

    x = layers.SeparableConv1D(96, kernel_size=3, activation="relu", padding="same",
                                depthwise_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    x   = layers.Concatenate()([avg, mx])

    x   = layers.Dense(96, activation="relu", kernel_regularizer=reg)(x)
    x   = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def convert_to_tflite(model: keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def build_gate_metadata(base_sequences: np.ndarray) -> dict:
    motions  = np.array([sequence_motion_energy(s)  for s in base_sequences], dtype=np.float32)
    presence = np.array([sequence_presence_ratio(s) for s in base_sequences], dtype=np.float32)
    usable   = motions[presence >= 0.5]
    if usable.size == 0:
        usable = motions
    motion_threshold = max(0.025, float(np.percentile(usable, 5) * 0.65))
    return {
        "motion_threshold":  motion_threshold,
        "presence_threshold": 0.50,
        "motion_p05":  float(np.percentile(motions,  5)),
        "motion_p50":  float(np.percentile(motions, 50)),
        "motion_p95":  float(np.percentile(motions, 95)),
        "presence_p50": float(np.percentile(presence, 50)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(user_id: str | None = None) -> pd.DataFrame:
    X, y = load_all_data("dynamic", user_id=user_id)
    return pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]))


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    user_id: str | None = None,
    epochs: int = 80,
    batch_size: int = 32,
    skip_tflite: bool = False,
) -> None:
    print("\n" + "=" * 62)
    print("  TRAINING DYNAMIC GESTURE MODEL  (Temporal CNN)")
    print("=" * 62 + "\n")

    df = _load_data(user_id)
    print(f"Raw sequences: {len(df)}\n")

    X_flat = df.iloc[:, :-1].values.astype(np.float32)
    y_raw  = canonicalize_labels(df.iloc[:, -1].values)

    expected = SEQUENCE_LENGTH * LANDMARK_DIM
    if X_flat.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} feature columns ({SEQUENCE_LENGTH}×{LANDMARK_DIM}), "
            f"got {X_flat.shape[1]}."
        )

    X = X_flat.reshape(-1, SEQUENCE_LENGTH, LANDMARK_DIM)
    gate_metadata = build_gate_metadata(X)

    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"Classes ({len(unique)}):")
    for cls, cnt in zip(unique, counts):
        status = "" if cnt >= 150 else "  ← collect more (target ≥ 150)"
        print(f"  {cls:<22} {cnt:>4} seqs{status}")

    if any(cnt < 30 for cnt in counts):
        print("\n⚠  Some classes have < 30 samples. Accuracy will be low.")
        print("   Run collect_dynamic.py to gather more data before deploying.\n")

    print(
        f"\nMotion gate: energy ≥ {gate_metadata['motion_threshold']:.4f}, "
        f"presence ≥ {gate_metadata['presence_threshold']:.2f}"
    )

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"Classes: {list(le.classes_)}")
    print(f"Per-frame feature dim: {dynamic_feature_dim()} ({DYNAMIC_FEATURE_VERSION})")

    # Split raw before augmentation to prevent data leakage
    X_tr_raw, X_te_raw, y_tr_ids, y_te_ids = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )

    # Augment training split only
    print("\nAugmenting training sequences…")
    aug_X, aug_y = [], []
    for seq, label in zip(X_tr_raw, y_tr_ids):
        for a in augment_sequence(seq):
            aug_X.append(a)
            aug_y.append(label)
    X_tr_base = np.array(aug_X, dtype=np.float32)
    y_tr_ids  = np.array(aug_y)
    print(f"Train after augmentation: {len(X_tr_base)} sequences")

    # Feature extraction
    print("Extracting features…")
    X_tr = extract_dynamic_batch(X_tr_base)
    X_te = extract_dynamic_batch(X_te_raw)

    # Scale — fit on train only
    print("Scaling…")
    scaler      = StandardScaler()
    feature_dim = X_tr.shape[2]
    X_tr = scaler.fit_transform(X_tr.reshape(-1, feature_dim)).reshape(X_tr.shape)
    X_te = scaler.transform(X_te.reshape(-1, feature_dim)).reshape(X_te.shape)

    y_tr = keras.utils.to_categorical(y_tr_ids, len(le.classes_))
    y_te = keras.utils.to_categorical(y_te_ids, len(le.classes_))
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # Build and train
    model = build_model(len(le.classes_), X_tr.shape[1:])
    model.summary()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(le.classes_)),
        y=y_tr_ids,
    )
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=1,
            ),
        ],
    )

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTest accuracy : {acc * 100:.2f}%  |  loss: {loss:.4f}")
    if acc < 0.80:
        print("⚠  Accuracy below 80%. Collect more diverse training data.")

    # Save paths
    output_tflite = MODEL_TFLITE
    output_pkl    = MODEL_PKL
    output_keras  = MODEL_KERAS
    if user_id:
        ensure_user_dir(user_id)
        output_tflite = user_dynamic_tflite_path(user_id)
        output_pkl    = user_model_path(user_id, "dynamic")
        output_keras  = user_dynamic_keras_path(user_id)

    model.save(output_keras)
    print(f"Saved Keras model: {output_keras}")

    tflite_available = False
    if not skip_tflite:
        print("\nConverting to TFLite…")
        try:
            tflite_bytes = convert_to_tflite(model)
            output_tflite.write_bytes(tflite_bytes)
            size_kb = output_tflite.stat().st_size // 1024
            print(f"Saved TFLite model: {output_tflite}  ({size_kb} KB)")
            tflite_available = True
        except Exception as exc:
            print(f"TFLite conversion failed: {exc}")
            print("Using Keras fallback for inference.")
    else:
        print("TFLite conversion skipped.")

    bundle = {
        "scaler":          scaler,
        "label_encoder":   le,
        "tflite_path":     str(output_tflite) if tflite_available else None,
        "keras_path":      str(output_keras),
        "feature_version": DYNAMIC_FEATURE_VERSION,
        "feature_dim":     feature_dim,
        "sequence_length": SEQUENCE_LENGTH,
        "model_type":      "temporal_cnn_v4",
        "gate":            gate_metadata,
    }
    with output_pkl.open("wb") as f:
        pickle.dump(bundle, f, protocol=5)
    print(f"Saved metadata : {output_pkl}")
    print("\nTraining complete. Run recognise.py to test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dynamic gesture model")
    parser.add_argument("--user-id",     default=None)
    parser.add_argument("--epochs",      type=int, default=80)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--skip-tflite", action="store_true")
    args = parser.parse_args()
    train(
        user_id=args.user_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_tflite=args.skip_tflite,
    )
