import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
CSV_FILE   = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv(CSV_FILE, header=None, on_bad_lines="skip")
print(f"Total raw samples : {len(df)}")
print("Gesture counts:\n", df.iloc[:, -1].value_counts())

X_raw = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values

# warn if any gesture has very few samples
counts = df.iloc[:, -1].value_counts()
low    = counts[counts < 50]
if not low.empty:
    print(f"\nWARNING — low sample count (consider collecting more):\n{low}\n")

# ─────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────
def augment(sample: np.ndarray) -> list:
    augmented = [sample]

    # Gaussian noise
    augmented.append(sample + np.random.normal(0, 0.01, sample.shape))

    # Random scaling
    augmented.append(sample * random.uniform(0.9, 1.1))

    # Small 2-D rotation (x, y only — skip z)
    angle   = random.uniform(-10, 10)
    rad     = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotated = sample.copy()
    for i in range(0, len(sample), 3):
        x, y           = sample[i], sample[i + 1]
        rotated[i]     = x * cos_a - y * sin_a
        rotated[i + 1] = x * sin_a + y * cos_a
    augmented.append(rotated)

    return augmented


aug_X, aug_y = [], []
for x, y in zip(X_raw, y_raw):
    for ax in augment(x):
        aug_X.append(ax)
        aug_y.append(y)

X = np.array(aug_X, dtype=np.float32)
y = np.array(aug_y)
print(f"After augmentation : {len(X)} samples")

# ─────────────────────────────────────────
#  ENCODE LABELS
# ─────────────────────────────────────────
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# ─────────────────────────────────────────
#  TRAIN / TEST SPLIT  (before scaling)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ─────────────────────────────────────────
#  SCALE — fit ONLY on train, apply to test
# ─────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit here only
X_test  = scaler.transform(X_test)        # never fit on test

# ─────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=1000,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=True,
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────
#  EVALUATE ON TEST SET
# ─────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────
#  CROSS-VALIDATION  (on raw data only —
#  avoids augmented siblings leaking across
#  folds and inflating the CV score)
# ─────────────────────────────────────────
print("Running cross-validation on raw data (this may take a minute) ...")
le_raw        = LabelEncoder()
y_raw_encoded = le_raw.fit_transform(y_raw)
scaler_raw    = StandardScaler()
X_raw_scaled  = scaler_raw.fit_transform(X_raw.astype(np.float32))

cv_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    max_iter=500,
    random_state=42,
)
cv_scores = cross_val_score(cv_model, X_raw_scaled, y_raw_encoded, cv=5)
print(f"Cross-val accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# ─────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────
with open(MODEL_FILE, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "label_encoder": le}, f)

print(f"\nModel saved to {MODEL_FILE}")
print(f"Dataset shape  : {df.shape}")
print(f"Unique labels  : {np.unique(y_raw)}")