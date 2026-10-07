import pytest

from mmrl.state.reference_price import (
    StableReferencePriceTracker,
    build_grid_ticks,
    candidate_reference_ref2,
    grid_level_tick,
    queue_at_tick,
)


def test_candidate_odd_spread_is_mid():
    # bid=100, ask=101 => spread 1 tick, mid=100.5 => ref2=201
    assert candidate_reference_ref2(
        100, 101, previous_ref2=None
    ) == 201


def test_candidate_even_spread_uses_previous_reference():
    # bid=100, ask=102 => mid=101, candidates 100.5 and 101.5
    assert candidate_reference_ref2(
        100, 102, previous_ref2=199
    ) == 201

    assert candidate_reference_ref2(
        100, 102, previous_ref2=205
    ) == 203


def test_initial_even_spread_tie_break():
    assert candidate_reference_ref2(
        100, 102, previous_ref2=None,
        initial_even_tie_break="lower",
    ) == 201

    assert candidate_reference_ref2(
        100, 102, previous_ref2=None,
        initial_even_tie_break="upper",
    ) == 203


def test_grid_ticks():
    ref2 = 201  # p_ref = 100.5 ticks

    assert grid_level_tick(ref2, -1) == 100
    assert grid_level_tick(ref2, +1) == 101
    assert grid_level_tick(ref2, -2) == 99
    assert grid_level_tick(ref2, +2) == 102
    assert grid_level_tick(ref2, -5) == 96
    assert grid_level_tick(ref2, +5) == 105

    grid = build_grid_ticks(ref2, k=5)
    assert list(grid) == [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def test_queue_lookup_distinguishes_known_empty_and_unknown():
    bids = {
        100: 5.0,
        98: 4.0,
        97: 3.0,
        95: 2.0,
        94: 1.0,
    }
    asks = {
        102: 5.0,
        103: 4.0,
        105: 3.0,
        106: 2.0,
        108: 1.0,
    }

    x = queue_at_tick(99, bids, asks)
    assert x.known
    assert x.qty == 0.0
    assert x.status == "known_empty_bid_range"

    x = queue_at_tick(101, bids, asks)
    assert x.known
    assert x.qty == 0.0
    assert x.status == "known_empty_inside_spread"

    x = queue_at_tick(93, bids, asks)
    assert not x.known
    assert x.status == "unknown_beyond_visible_depth"

    x = queue_at_tick(109, bids, asks)
    assert not x.known
    assert x.status == "unknown_beyond_visible_depth"


def test_mid_up_updates_when_qminus1_empty():
    tracker = StableReferencePriceTracker(
        initial_even_tie_break="lower"
    )

    # reset: bid=100 ask=102 => ref=100.5 => Q-1=100
    s0 = tracker.reset(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0},
    )
    assert s0.ref2 == 201

    # mid rises. Old Q-1 tick=100 is now known empty because best bid moved to 101.
    s1 = tracker.update(
        {101: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0, 107: 1.0},
    )
    assert s1.updated
    assert s1.reason == "mid_up_qminus1_empty_update"


def test_mid_up_holds_when_qminus1_nonempty():
    tracker = StableReferencePriceTracker(
        initial_even_tie_break="lower"
    )

    tracker.reset(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0},
    )

    # Mid rises because ask rises, but old Q-1=100 still has liquidity.
    s1 = tracker.update(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0, 107: 1.0},
    )

    assert not s1.updated
    assert s1.reason == "mid_up_qminus1_nonempty_hold"


def test_mid_down_updates_when_qplus1_empty():
    tracker = StableReferencePriceTracker(
        initial_even_tie_break="upper"
    )

    # bid=100 ask=102 => upper ref=101.5 => Q+1=102
    s0 = tracker.reset(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0},
    )
    assert s0.ref2 == 203

    # mid decreases, old Q+1=102 becomes known empty.
    s1 = tracker.update(
        {99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0, 95: 1.0},
        {101: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0},
    )
    assert s1.updated
    assert s1.reason == "mid_down_qplus1_empty_update"


def test_unknown_gate_does_not_fake_zero():
    tracker = StableReferencePriceTracker(
        initial_even_tie_break="lower"
    )

    tracker.reset(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0},
    )

    # Force a synthetic old ref whose Q-1 is deeper than visible depth.
    tracker.ref2 = 189  # p_ref=94.5, Q-1=94 is below deepest visible bid 96
    tracker.prev_mid2 = 202

    s1 = tracker.update(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0, 107: 1.0},
    )

    assert not s1.updated
    assert s1.reason == "mid_up_gate_unknown"
    assert s1.gate_known is False
