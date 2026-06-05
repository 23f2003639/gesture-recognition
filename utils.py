"""
utils.py — Shared constants and landmark normalization.

Both static (per-frame) and dynamic (sequence) pipelines use the
same normalization so train/inference features are identical.
"""

SEQUENCE_LENGTH = 30   # frames per dynamic gesture sample  (1 s @ 30 fps)
LANDMARK_DIM    = 126  # 21 landmarks × 2 hands × 3 coords

# Feature sizes
STATIC_FEATURE_DIM  = LANDMARK_DIM                    # 126  — one frame
DYNAMIC_FEATURE_DIM = SEQUENCE_LENGTH * LANDMARK_DIM  # 3780 — flattened sequence


def normalize(landmarks: list) -> list:
    """
    Make hand landmarks position- and scale-independent.

    Steps:
      1. Subtract wrist (landmark 0) so the hand is zero-centred.
      2. Divide by max-abs-value so hand size doesn't affect the features.

    Returns the input unchanged if it's all zeros (absent hand sentinel).
    """
    if all(v == 0.0 for v in landmarks):
        return landmarks

    base_x, base_y = landmarks[0], landmarks[1]
    normed = list(landmarks)
    for i in range(0, len(normed), 3):
        normed[i]     -= base_x
        normed[i + 1] -= base_y

    max_val = max(abs(x) for x in normed) + 1e-6
    return [x / max_val for x in normed]
