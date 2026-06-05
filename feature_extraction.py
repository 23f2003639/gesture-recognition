"""
Feature extraction for the gesture recognition pipeline.

The existing CSV files store 126 normalized landmark values:
21 landmarks * 3 coordinates * 2 hand slots.

This module keeps that format intact and derives additional deterministic
features from it. That means old data remains usable, while training and
inference can share the exact same feature recipe.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from utils import LANDMARK_DIM


LABEL_ALIASES = {
    "WORD-BREAK": "WORD_BREAK",
    "SENTENCE-BREAK": "SENTENCE_BREAK",
}

SUSPECT_LABELS = {"126"}

STATIC_FEATURE_VERSION = "static_pose_v2"
DYNAMIC_FEATURE_VERSION = "dynamic_pose_velocity_v2"

HAND_DIM = 63
HAND_POINTS = 21

FINGER_TRIPLETS = (
    (1, 2, 4),    # thumb
    (5, 6, 8),    # index
    (9, 10, 12),  # middle
    (13, 14, 16), # ring
    (17, 18, 20), # pinky
)
FINGER_TIPS = (4, 8, 12, 16, 20)


def canonical_label(label: object) -> str:
    """Normalize labels without mutating the source CSV."""
    text = str(label).strip()
    return LABEL_ALIASES.get(text, text)


def canonicalize_labels(labels: Iterable[object]) -> np.ndarray:
    return np.array([canonical_label(label) for label in labels])


def is_suspect_label(label: object) -> bool:
    return canonical_label(label) in SUSPECT_LABELS


def hand_data_to_base_features(hand_data: dict, normalize_fn) -> np.ndarray:
    """Convert MediaPipe hand_data into the legacy 126-value base vector."""
    left = normalize_fn(hand_data.get("Left", [0.0] * HAND_DIM))
    right = normalize_fn(hand_data.get("Right", [0.0] * HAND_DIM))
    return np.asarray(left + right, dtype=np.float32)


def extract_static_features(base_features: np.ndarray) -> np.ndarray:
    """
    Derive production features from one 126-value landmark frame.

    Output:
      - original normalized landmarks
      - hand presence masks
      - palm normals
      - palm directions
      - finger bend angles
      - fingertip distances from wrist
      - hand span
    """
    base = _ensure_base_vector(base_features)
    left = base[:HAND_DIM].reshape(HAND_POINTS, 3)
    right = base[HAND_DIM:].reshape(HAND_POINTS, 3)

    pose_features = np.concatenate(
        [
            np.asarray([_present(left), _present(right)], dtype=np.float32),
            _single_hand_features(left),
            _single_hand_features(right),
        ]
    ).astype(np.float32)

    return np.concatenate([base, pose_features]).astype(np.float32)


def extract_static_batch(base_features: np.ndarray) -> np.ndarray:
    base = np.asarray(base_features, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != LANDMARK_DIM:
        raise ValueError(f"Expected shape (n, {LANDMARK_DIM}), got {base.shape}")
    return np.vstack([extract_static_features(row) for row in base]).astype(np.float32)


def extract_dynamic_sequence(base_sequence: np.ndarray) -> np.ndarray:
    """
    Convert a (T, 126) landmark sequence into per-frame pose + velocity features.

    Velocity is computed over the richer static feature vector, so finger motion
    and palm motion are represented directly. The first frame has zero velocity.
    """
    seq = np.asarray(base_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != LANDMARK_DIM:
        raise ValueError(f"Expected shape (T, {LANDMARK_DIM}), got {seq.shape}")

    static_seq = np.vstack([extract_static_features(frame) for frame in seq]).astype(np.float32)
    velocity = np.zeros_like(static_seq)
    velocity[1:] = static_seq[1:] - static_seq[:-1]
    return np.concatenate([static_seq, velocity], axis=1).astype(np.float32)


def extract_dynamic_batch(base_sequences: np.ndarray) -> np.ndarray:
    seqs = np.asarray(base_sequences, dtype=np.float32)
    if seqs.ndim != 3 or seqs.shape[2] != LANDMARK_DIM:
        raise ValueError(f"Expected shape (n, T, {LANDMARK_DIM}), got {seqs.shape}")
    return np.stack([extract_dynamic_sequence(seq) for seq in seqs]).astype(np.float32)


def sequence_presence_ratio(base_sequence: np.ndarray) -> float:
    """Return the fraction of frames that contain at least one detected hand."""
    seq = np.asarray(base_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != LANDMARK_DIM:
        raise ValueError(f"Expected shape (T, {LANDMARK_DIM}), got {seq.shape}")
    present = ~np.all(np.isclose(seq, 0.0, atol=1e-7), axis=1)
    return float(np.mean(present)) if len(present) else 0.0


def sequence_motion_energy(base_sequence: np.ndarray) -> float:
    """
    Estimate how much real hand motion exists in a landmark sequence.

    The dynamic classifier is a closed-set model, so a softmax confidence alone
    cannot reject static poses or unknown signs. This score gives inference a
    cheap open-set gate before a word is allowed to commit.
    """
    seq = np.asarray(base_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != LANDMARK_DIM:
        raise ValueError(f"Expected shape (T, {LANDMARK_DIM}), got {seq.shape}")

    present = ~np.all(np.isclose(seq, 0.0, atol=1e-7), axis=1)
    valid_pairs = present[1:] & present[:-1]
    if not np.any(valid_pairs):
        return 0.0

    diffs = np.diff(seq, axis=0)[valid_pairs]
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def static_feature_dim() -> int:
    return int(extract_static_features(np.zeros(LANDMARK_DIM, dtype=np.float32)).shape[0])


def dynamic_feature_dim() -> int:
    return int(extract_dynamic_sequence(np.zeros((2, LANDMARK_DIM), dtype=np.float32)).shape[1])


def _ensure_base_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != LANDMARK_DIM:
        raise ValueError(f"Expected {LANDMARK_DIM} base features, got {arr.shape[0]}")
    return arr


def _present(hand: np.ndarray) -> float:
    return 0.0 if np.allclose(hand, 0.0, atol=1e-7) else 1.0


def _single_hand_features(hand: np.ndarray) -> np.ndarray:
    if _present(hand) == 0.0:
        return np.zeros(17, dtype=np.float32)

    wrist = hand[0]
    index_mcp = hand[5]
    middle_mcp = hand[9]
    pinky_mcp = hand[17]

    palm_normal = _unit(np.cross(index_mcp - wrist, pinky_mcp - wrist))
    palm_direction = _unit(middle_mcp - wrist)

    angles = np.asarray(
        [_joint_angle(hand[a], hand[b], hand[c]) / np.pi for a, b, c in FINGER_TRIPLETS],
        dtype=np.float32,
    )

    scale = _hand_scale(hand)
    tip_distances = np.asarray(
        [np.linalg.norm(hand[idx] - wrist) / scale for idx in FINGER_TIPS],
        dtype=np.float32,
    )
    span = np.asarray([np.linalg.norm(hand[4] - hand[20]) / scale], dtype=np.float32)

    return np.concatenate(
        [palm_normal, palm_direction, angles, tip_distances, span]
    ).astype(np.float32)


def _hand_scale(hand: np.ndarray) -> float:
    wrist = hand[0]
    distances = [np.linalg.norm(hand[idx] - wrist) for idx in (5, 9, 13, 17)]
    return float(max(np.mean(distances), 1e-6))


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 0.0
    cos_angle = float(np.dot(ba, bc) / denom)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
