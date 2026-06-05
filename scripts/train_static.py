"""
HOW TO RUN:
    python train_static.py
    python train_static.py --skip-cv          # faster, skip 5-fold CV
    python train_static.py --user-id alice    # personalized model
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from feature_extraction import (
    STATIC_FEATURE_VERSION,
    SUSPECT_LABELS,
    canonicalize_labels,
    extract_static_batch,
    is_suspect_label,
    static_feature_dim,
)
from personalization import ensure_user_dir, user_data_path, user_model_path, load_all_data

BASE_DIR   = Path(__file__).resolve().parent
DATA_CSV   = BASE_DIR / "data_static.csv"
LEGACY_CSV = BASE_DIR / "gesture_data.csv"    # original well-performing dataset
MODEL_FILE = BASE_DIR / "static_model.pkl"


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION  (same 4-copy scheme as original, proven to work well)
# ─────────────────────────────────────────────────────────────────────────────

def augment(sample: np.ndarray) -> list[np.ndarray]:
    """4 augmented copies per original sample."""
    aug = [sample]

    # 1. Gaussian jitter
    aug.append(sample + np.random.normal(0, 0.012, sample.shape).astype(np.float32))

    # 2. Random scale ±12%
    aug.append((sample * random.uniform(0.88, 1.12)).astype(np.float32))

    # 3. 2D in-plane rotation ±15°
    angle = random.uniform(-15, 15)
    rad   = np.deg2rad(angle)
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
    rotated = sample.copy()
    for i in range(0, len(sample), 3):
        x, y             = sample[i], sample[i + 1]
        rotated[i]       = x * cos_a - y * sin_a
        rotated[i + 1]   = x * sin_a + y * cos_a
    aug.append(rotated)

    # 4. Horizontal mirror + hand-slot swap (simulates dominant-hand swap)
    mirrored = sample.reshape(2, 21, 3).copy()
    mirrored[:, :, 0] *= -1
    mirrored = mirrored[::-1].reshape(sample.shape)
    aug.append(mirrored.astype(np.float32))

    return aug


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING  — merge all available CSVs
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(user_id: str | None = None) -> pd.DataFrame:
    X, y = load_all_data("static", user_id=user_id)
    return pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]))

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    user_id: str | None = None,
    include_suspect_labels: bool = False,
    run_cv: bool = True,
) -> None:
    print("\n" + "=" * 62)
    print("  TRAINING STATIC GESTURE MODEL")
    print("=" * 62 + "\n")

    df = _load_data(user_id)

    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = canonicalize_labels(df.iloc[:, -1].values)

    # Remove empty/whitespace-only labels
    valid = np.array([bool(str(lbl).strip()) for lbl in y_raw])
    if not valid.all():
        n_bad = int((~valid).sum())
        print(f"  Dropping {n_bad} rows with empty labels.\n")
        X_raw = X_raw[valid]
        y_raw = y_raw[valid]

    # Suspect label filter
    suspect_mask = np.array([is_suspect_label(lbl) for lbl in y_raw])
    if suspect_mask.any() and not include_suspect_labels:
        print(f"  Skipping suspect labels: {sorted(SUSPECT_LABELS & set(y_raw))}")
        X_raw = X_raw[~suspect_mask]
        y_raw = y_raw[~suspect_mask]

    # Per-class counts
    series = pd.Series(y_raw)
    counts = series.value_counts().sort_index()
    print("Gesture counts:")
    print(counts.to_string())
    low = counts[counts < 50]
    if not low.empty:
        print(f"\n⚠  Low sample count — consider collecting more:\n{low.to_string()}\n")

    # ── Encode ────────────────────────────────────────────────────────────
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")
    print(f"Feature dim : {static_feature_dim()} ({STATIC_FEATURE_VERSION})")

    # ── Train/test split BEFORE augmentation ──────────────────────────────
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )

    # ── Augment train only ─────────────────────────────────────────────────
    aug_X, aug_y = [], []
    for x, y in zip(X_tr_raw, y_tr):
        for ax in augment(x):
            aug_X.append(ax)
            aug_y.append(y)
    X_tr_base = np.array(aug_X, dtype=np.float32)
    y_tr      = np.array(aug_y)
    print(f"\nTrain after augmentation : {len(X_tr_base)}")

    # ── Feature extraction ─────────────────────────────────────────────────
    print("Extracting features…")
    X_tr = extract_static_batch(X_tr_base)
    X_te = extract_static_batch(X_te_raw)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)   # fit on train only
    X_te   = scaler.transform(X_te)

    # ── Model ─────────────────────────────────────────────────────────────
    # (256→128→64) gives ~97-99% on well-collected data and runs in <1 ms.
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        alpha=1e-4,
        batch_size=128,
        random_state=42,
        verbose=True,
    )
    print("\nTraining…")
    model.fit(X_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    print(f"\nTest accuracy : {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # ── Cross-validation on raw (no augmented leakage) ────────────────────
    if run_cv:
        try:
            print("Running 5-fold CV on raw data (no leakage)…")
            le2  = LabelEncoder()
            y2   = le2.fit_transform(y_raw)
            sc2  = StandardScaler()
            X2   = sc2.fit_transform(extract_static_batch(X_raw.copy()))
            cvm  = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation="relu",
                max_iter=500, random_state=42,
            )
            cv_scores = cross_val_score(cvm, X2, y2, cv=5, n_jobs=1)
            print(f"CV accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
        except Exception as exc:
            print(f"CV skipped: {exc}")
    else:
        print("CV skipped.")

    # ── Save ──────────────────────────────────────────────────────────────
    output = MODEL_FILE
    if user_id:
        ensure_user_dir(user_id)
        output = user_model_path(user_id, "static")

    bundle = {
        "model":         model,
        "scaler":        scaler,
        "label_encoder": le,
        "feature_version": STATIC_FEATURE_VERSION,
        "feature_dim":   static_feature_dim(),
        "excluded_suspect_labels": [] if include_suspect_labels else sorted(SUSPECT_LABELS),
    }
    with output.open("wb") as f:
        pickle.dump(bundle, f, protocol=5)

    print(f"\n✓ Saved: {output}")
    print(f"  {len(le.classes_)} classes  |  {len(X_raw)} raw samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the static gesture classifier")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--include-suspect-labels", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    train(
        user_id=args.user_id,
        include_suspect_labels=args.include_suspect_labels,
        run_cv=not args.skip_cv,
    )
