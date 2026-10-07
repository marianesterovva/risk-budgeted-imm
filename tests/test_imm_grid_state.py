import numpy as np
import pytest

from mmrl.imm.grid import (
    CandidateReferencePriceTracker,
    classify_side_price,
    price_tick_to_grid_level,
)
from mmrl.imm.state_builder import IMMStateBuilder, OwnOrderView
from mmrl.imm.simulator_adapter import (
    IMMGridReplayAdapter,
    augment_current_book_for_known_empty,
)
from mmrl.simulator import MarketBookSnapshot, ReplaySimulator, TargetOrder


def _book(timestamp_us=0):
    return MarketBookSnapshot(
        timestamp_us=timestamp_us,
        bids={100: 5.0, 98: 4.0, 97: 3.0, 95: 2.0, 94: 1.0},
        asks={102: 5.0, 104: 4.0, 105: 3.0, 107: 2.0, 108: 1.0},
        mid_price=101.0,
        tick_size=1.0,
    )


def test_candidate_reference_follows_market():
    t = CandidateReferencePriceTracker()
    r0 = t.update(
        {100: 1.0, 99: 1.0, 98: 1.0, 97: 1.0, 96: 1.0},
        {102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0},
    )
    assert r0 == 201
    r1 = t.update(
        {110: 1.0, 109: 1.0, 108: 1.0, 107: 1.0, 106: 1.0},
        {112: 1.0, 113: 1.0, 114: 1.0, 115: 1.0, 116: 1.0},
    )
    assert abs(r1 - 222) == 1


def test_grid_level_roundtrip():
    ref2 = 201
    assert price_tick_to_grid_level(ref2, 100, k=5) == -1
    assert price_tick_to_grid_level(ref2, 99, k=5) == -2
    assert price_tick_to_grid_level(ref2, 101, k=5) == 1
    assert price_tick_to_grid_level(ref2, 105, k=5) == 5
    assert price_tick_to_grid_level(ref2, 95, k=5) is None


def test_known_empty_and_unknown_are_distinct():
    b = _book()
    x = classify_side_price(side="buy", price_tick=99, bids=b.bids, asks=b.asks)
    assert x.allowed and x.known and x.queue_ahead == 0.0
    assert x.status == "known_empty_bid_range"

    x = classify_side_price(side="buy", price_tick=101, bids=b.bids, asks=b.asks)
    assert x.allowed and x.status == "known_empty_inside_spread"

    x = classify_side_price(side="buy", price_tick=93, bids=b.bids, asks=b.asks)
    assert not x.allowed and not x.known

    x = classify_side_price(side="sell", price_tick=103, bids=b.bids, asks=b.asks)
    assert x.allowed and x.status == "known_empty_ask_range"


def test_market_feature_shape_and_masks():
    builder = IMMStateBuilder(quote_size=1.0, k=5, lookback=64)
    x = builder.snapshot_market_features(
        bids={100: 5.0, 98: 4.0, 97: 3.0, 95: 2.0, 94: 1.0},
        asks={102: 5.0, 104: 4.0, 105: 3.0, 107: 2.0, 108: 1.0},
        ref2=201,
    )
    assert x.shape == (20,)
    qty, known = x[:10], x[10:]
    levels = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    idx = levels.index(-2)
    assert qty[idx] == 0.0
    assert known[idx] == 1.0


def test_history_tensor_causal_and_boundary_safe():
    builder = IMMStateBuilder(quote_size=1.0, k=5, lookback=64)
    f = builder.market_feature_dim
    matrix = np.arange(100 * f, dtype=np.float32).reshape(100, f)
    episode = np.zeros(100, dtype=np.int64)
    x = builder.history_tensor(matrix, end_idx=63, episode_ids=episode)
    assert x.shape == (20, 64)
    assert np.array_equal(x[:, -1], matrix[63])

    episode[30] = 1
    with pytest.raises(ValueError):
        builder.history_tensor(matrix, end_idx=63, episode_ids=episode)


def test_private_state_is_21d():
    builder = IMMStateBuilder(quote_size=2.0, k=5, lookback=64)
    own = [
        OwnOrderView(level=-1, remaining_qty=1.0, queue_ahead=3.0),
        OwnOrderView(level=-1, remaining_qty=1.0, queue_ahead=1.0),
        OwnOrderView(level=2, remaining_qty=2.0, queue_ahead=0.0),
    ]
    s = builder.build_private_state(
        inventory=4.0,
        own_orders=own,
        market_qty_by_level={-1: 6.0, 2: 0.0},
    )
    assert s.shape == (21,)
    assert np.isfinite(s).all()
    assert s[0] == pytest.approx(2.0)
    assert np.all((s[1:11] >= 0) & (s[1:11] <= 1))


def test_adapter_does_not_change_historical_best():
    b = _book()
    target = [
        TargetOrder("buy", 99, 1.0),
        TargetOrder("sell", 103, 1.0),
        TargetOrder("buy", 101, 1.0),
    ]
    aug = augment_current_book_for_known_empty(b, target)
    assert aug.bids[99] == 0.0
    assert aug.asks[103] == 0.0
    assert 101 not in aug.bids
    assert aug.best_bid_tick == b.best_bid_tick
    assert aug.best_ask_tick == b.best_ask_tick


def test_real_replay_accepts_known_empty_outward_ticks():
    cur = _book(timestamp_us=1_000_000)
    nxt = _book(timestamp_us=1_100_000)
    sim = ReplaySimulator(
        max_gap_ms=5000,
        maker_fee_rate=0.0,
        allow_marketable_orders=False,
        allow_inside_spread=True,
        allow_beyond_depth5=False,
        trade_through_fill=False,
        allow_partial_fills=True,
    )
    sim.reset()
    adapter = IMMGridReplayAdapter(sim)
    result = adapter.step(
        current_book=cur,
        next_book=nxt,
        trades=[],
        target_book=[
            TargetOrder("buy", 99, 1.0),
            TargetOrder("sell", 103, 1.0),
        ],
    )
    assert result.placed_qty == pytest.approx(2.0)
    assert result.filled_qty == pytest.approx(0.0)


def test_adapter_rejects_unknown_target():
    cur = _book(timestamp_us=1_000_000)
    nxt = _book(timestamp_us=1_100_000)
    sim = ReplaySimulator(
        max_gap_ms=5000,
        maker_fee_rate=0.0,
        allow_marketable_orders=False,
        allow_inside_spread=True,
        allow_beyond_depth5=False,
        trade_through_fill=False,
        allow_partial_fills=True,
    )
    adapter = IMMGridReplayAdapter(sim)
    with pytest.raises(ValueError, match="not observable/passive"):
        adapter.step(
            current_book=cur,
            next_book=nxt,
            trades=[],
            target_book=[TargetOrder("buy", 93, 1.0)],
        )
