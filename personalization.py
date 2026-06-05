"""
Per-user data and model paths.

Folder layout
─────────────
Base data (no user):
  data_static/
    <LABEL>/
      data.csv          ← all rows for this label
  data_dynamic/
    <LABEL>/
      data.csv

Per-user data:
  user_profiles/<user_id>/
    static/
      <LABEL>/
        data.csv
    dynamic/
      <LABEL>/
        data.csv
    static_model.pkl
    dynamic_model.pkl
    dynamic_model.tflite
    dynamic_model.keras

The training scripts merge base + user folders into one DataFrame,
so the model learns from everything.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

BASE_DIR          = Path(__file__).resolve().parent
USER_PROFILES_DIR = BASE_DIR / "user_profiles"

# Base (shared) data roots — one subfolder per gesture class
BASE_STATIC_DIR  = BASE_DIR / "data_static"
BASE_DYNAMIC_DIR = BASE_DIR / "data_dynamic"


# ── Sanitisation ─────────────────────────────────────────────────────────────

def sanitize_user_id(user_id: str | None) -> str | None:
    if not user_id:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("User id must contain at least one letter or number.")
    return cleaned[:80]


def sanitize_label(label: str) -> str:
    """Make a gesture label safe to use as a folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    return cleaned.strip("_")[:60]


# ── User directory helpers ────────────────────────────────────────────────────

def user_dir(user_id: str) -> Path:
    safe_id = sanitize_user_id(user_id)
    if safe_id is None:
        raise ValueError("user_id is required")
    return USER_PROFILES_DIR / safe_id


def user_data_root(user_id: str, kind: str) -> Path:
    """Root folder that contains per-label subfolders for this user."""
    if kind not in {"static", "dynamic"}:
        raise ValueError("kind must be 'static' or 'dynamic'")
    return user_dir(user_id) / kind


def user_label_dir(user_id: str, kind: str, label: str) -> Path:
    return user_data_root(user_id, kind) / sanitize_label(label)


def user_label_csv(user_id: str, kind: str, label: str) -> Path:
    return user_label_dir(user_id, kind, label) / "data.csv"


# kept for backward-compat with train scripts that import user_data_path
def user_data_path(user_id: str, kind: str) -> Path:
    """
    Legacy flat-file path.  Still honoured if the file exists.
    New code writes to per-label CSVs instead.
    """
    if kind not in {"static", "dynamic"}:
        raise ValueError("kind must be 'static' or 'dynamic'")
    return user_dir(user_id) / f"data_{kind}.csv"


def user_model_path(user_id: str, kind: str) -> Path:
    if kind not in {"static", "dynamic"}:
        raise ValueError("kind must be 'static' or 'dynamic'")
    return user_dir(user_id) / f"{kind}_model.pkl"


def user_dynamic_tflite_path(user_id: str) -> Path:
    return user_dir(user_id) / "dynamic_model.tflite"


def user_dynamic_keras_path(user_id: str) -> Path:
    return user_dir(user_id) / "dynamic_model.keras"


# ── Base data helpers ─────────────────────────────────────────────────────────

def base_data_root(kind: str) -> Path:
    return BASE_STATIC_DIR if kind == "static" else BASE_DYNAMIC_DIR


def base_label_dir(kind: str, label: str) -> Path:
    return base_data_root(kind) / sanitize_label(label)


def base_label_csv(kind: str, label: str) -> Path:
    return base_label_dir(kind, label) / "data.csv"


# ── Ensure dirs ───────────────────────────────────────────────────────────────

def ensure_user_dir(user_id: str) -> Path:
    path = user_dir(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_user_label_dir(user_id: str, kind: str, label: str) -> Path:
    path = user_label_dir(user_id, kind, label)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_base_label_dir(kind: str, label: str) -> Path:
    path = base_label_dir(kind, label)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Count helpers ─────────────────────────────────────────────────────────────

def count_rows_in_csv(path: Path) -> int:
    """Count data rows in a CSV without loading it all into memory."""
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for row in csv.reader(f) if row)


def count_label_samples(kind: str, label: str, user_id: str | None = None) -> int:
    """Return total sample count for a label across base + user CSVs."""
    total = count_rows_in_csv(base_label_csv(kind, label))
    if user_id:
        total += count_rows_in_csv(user_label_csv(user_id, kind, label))
    return total


# ── Data writers ──────────────────────────────────────────────────────────────

def append_static_row(label: str, row: list, user_id: str | None = None) -> Path:
    """Append one static sample row to the correct per-label CSV."""
    if user_id:
        ensure_user_label_dir(user_id, "static", label)
        csv_path = user_label_csv(user_id, "static", label)
    else:
        ensure_base_label_dir("static", label)
        csv_path = base_label_csv("static", label)
    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    return csv_path


def append_dynamic_row(label: str, row: list, user_id: str | None = None) -> Path:
    """Append one dynamic sample row to the correct per-label CSV."""
    if user_id:
        ensure_user_label_dir(user_id, "dynamic", label)
        csv_path = user_label_csv(user_id, "dynamic", label)
    else:
        ensure_base_label_dir("dynamic", label)
        csv_path = base_label_csv("dynamic", label)
    with csv_path.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    return csv_path


# ── Data loader (for training scripts) ───────────────────────────────────────

def load_all_data(kind: str, user_id: str | None = None):
    """
    Load all samples for `kind` ('static' or 'dynamic') into two numpy arrays.

    Merges:
      1. Base data  (data_static/ or data_dynamic/ subfolders)
      2. Legacy flat CSV (data_static.csv / data_dynamic.csv) if it exists
      3. User per-label subfolders (if user_id given)
      4. User legacy flat CSV      (if user_id given and file exists)

    Returns (X, y) where X is float32 and y is a string label array.
    Prints a per-label count summary.
    """
    import pandas as pd

    frames = []

    # 1. Base per-label folders
    base_root = base_data_root(kind)
    if base_root.exists():
        for label_dir in sorted(base_root.iterdir()):
            if not label_dir.is_dir():
                continue
            csv_file = label_dir / "data.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file, header=None, on_bad_lines="skip")
                frames.append((label_dir.name, df))

    # 2. Legacy flat CSV in project root
    legacy_flat = BASE_DIR / f"data_{kind}.csv"
    if legacy_flat.exists():
        df = pd.read_csv(legacy_flat, header=None, on_bad_lines="skip")
        frames.append(("(legacy flat)", df))

    # 3. User per-label folders
    if user_id:
        u_root = user_data_root(user_id, kind)
        if u_root.exists():
            for label_dir in sorted(u_root.iterdir()):
                if not label_dir.is_dir():
                    continue
                csv_file = label_dir / "data.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file, header=None, on_bad_lines="skip")
                    frames.append((f"user/{user_id}/{label_dir.name}", df))

        # 4. User legacy flat CSV
        u_legacy = user_data_path(user_id, kind)
        if u_legacy.exists():
            df = pd.read_csv(u_legacy, header=None, on_bad_lines="skip")
            frames.append((f"user/{user_id}/(legacy flat)", df))

    if not frames:
        raise FileNotFoundError(
            f"No {kind} data found.\n"
            f"Expected per-label CSVs under '{base_data_root(kind)}'\n"
            f"or a flat '{legacy_flat}'.\n"
            f"Run collect_{'static' if kind == 'static' else 'dynamic'}.py first."
        )

    combined_dfs = []
    total = 0
    print(f"\n  {'Source':<40} {'rows':>6}")
    print(f"  {'-'*48}")
    for source, df in frames:
        print(f"  {source:<40} {len(df):>6}")
        total += len(df)
        combined_dfs.append(df)
    print(f"  {'─'*48}")
    print(f"  {'TOTAL':<40} {total:>6}\n")

    import pandas as pd
    combined = pd.concat(combined_dfs, ignore_index=True)
    X = combined.iloc[:, :-1].values.astype(np.float32)
    y = combined.iloc[:,  -1].values.astype(str)
    return X, y


# ── Model path helpers ────────────────────────────────────────────────────────

def choose_model_path(default_path: Path, user_id: str | None, kind: str) -> Path:
    if not user_id:
        return default_path
    path = user_model_path(user_id, kind)
    return path if path.exists() else default_path


def choose_dynamic_tflite_path(default_path: Path, user_id: str | None) -> Path:
    if not user_id:
        return default_path
    path = user_dynamic_tflite_path(user_id)
    return path if path.exists() else default_path
