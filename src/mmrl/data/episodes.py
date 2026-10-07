from __future__ import annotations

import numpy as np


def valid_transition_mask(
    timestamps_ms,
    max_gap_ms: int,
) -> np.ndarray:
    """
    Return a boolean mask of length n-1.

    mask[i] is True iff state i -> state i+1 is a valid transition.
    """
    ts = np.asarray(timestamps_ms, dtype=np.int64)

    if ts.ndim != 1:
        raise ValueError("timestamps_ms must be one-dimensional.")

    if len(ts) < 2:
        return np.zeros(0, dtype=bool)

    dt = np.diff(ts)

    if np.any(dt <= 0):
        raise ValueError("timestamps_ms must be strictly increasing.")

    return dt <= int(max_gap_ms)


def build_episode_ids(
    timestamps_ms,
    max_gap_ms: int,
) -> np.ndarray:
    """
    Assign an episode id to every state.

    A new episode starts after every transition whose dt exceeds max_gap_ms.
    """
    ts = np.asarray(timestamps_ms, dtype=np.int64)

    if ts.ndim != 1:
        raise ValueError("timestamps_ms must be one-dimensional.")

    if len(ts) == 0:
        return np.zeros(0, dtype=np.int64)

    if len(ts) == 1:
        return np.zeros(1, dtype=np.int64)

    dt = np.diff(ts)

    if np.any(dt <= 0):
        raise ValueError("timestamps_ms must be strictly increasing.")

    boundaries = dt > int(max_gap_ms)

    episode_ids = np.zeros(len(ts), dtype=np.int64)
    episode_ids[1:] = np.cumsum(boundaries, dtype=np.int64)

    return episode_ids
