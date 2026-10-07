import numpy as np

from mmrl.data import build_episode_ids, valid_transition_mask


def test_gap_splits_episodes():
    ts = np.array(
        [0, 100, 200, 10_000, 10_100],
        dtype=np.int64,
    )

    ids = build_episode_ids(
        ts,
        max_gap_ms=5000,
    )

    assert ids.tolist() == [0, 0, 0, 1, 1]

    mask = valid_transition_mask(
        ts,
        max_gap_ms=5000,
    )

    assert mask.tolist() == [True, True, False, True]
