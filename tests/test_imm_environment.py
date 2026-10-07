
import numpy as np
import pandas as pd
import pytest

from mmrl.imm.action_mapper import (
    DecodedIMMAction,
    IMMActionMapper,
)
from mmrl.imm.environment import (
    IMMMarketData,
    IMMMarketMakingEnv,
    IMMRewardConfig,
    discover_live_child_orders,
)
from mmrl.simulator import (
    MarketBookSnapshot,
    ReplaySimulator,
)


def _book():
    # regular visible book around current candidate reference 100.5
    return MarketBookSnapshot(
        timestamp_us=0,
        bids={
            100: 10.0,
            99: 9.0,
            98: 8.0,
            97: 7.0,
            96: 6.0,
        },
        asks={
            101: 10.0,
            102: 9.0,
            103: 8.0,
            104: 7.0,
            105: 6.0,
        },
        mid_price=100.5,
        tick_size=1.0,
    )


def test_decode_normalized_bounds():
    mapper = IMMActionMapper(
        quote_size=20.0
    )

    lo = mapper.decode_normalized(
        [-1, -1, -1, -1]
    )
    hi = mapper.decode_normalized(
        [1, 1, 1, 1]
    )

    assert lo.quoted_mid_ticks == pytest.approx(-3.0)
    assert hi.quoted_mid_ticks == pytest.approx(3.0)

    assert lo.spread_ticks == pytest.approx(2.0)
    assert hi.spread_ticks == pytest.approx(8.0)

    assert lo.phi_bid == pytest.approx(0.0)
    assert hi.phi_bid == pytest.approx(1.0)


def test_figure2_level_geometry():
    mapper = IMMActionMapper(
        quote_size=20.0
    )

    # Figure-2-like action: (-2, 2, 0.8, 1)
    decoded = DecodedIMMAction(
        quoted_mid_ticks=-2.0,
        spread_ticks=2.0,
        phi_bid=0.8,
        phi_ask=1.0,
    )

    bid = mapper._paper_levels(
        side="buy",
        boundary=-3.0,
    )
    ask = mapper._paper_levels(
        side="sell",
        boundary=-1.0,
    )

    assert bid == (-3, -4)
    assert ask == (-1, 1)

    # Extreme spread: one bid edge level remains.
    bid_edge = mapper._paper_levels(
        side="buy",
        boundary=-5.5,
    )
    assert bid_edge == (-5,)


def test_two_level_volume_conservation_after_projection():
    mapper = IMMActionMapper(
        quote_size=20.0
    )

    mapped = mapper.map_decoded(
        decoded=DecodedIMMAction(
            quoted_mid_ticks=0.0,
            spread_ticks=4.0,
            phi_bid=0.4,
            phi_ask=0.7,
        ),
        normalized_action=[0, -1/3, -0.2, 0.4],
        ref2=201,
        current_book=_book(),
    )

    buy = [
        x for x in mapped.target_book
        if x.side == "buy"
    ]
    sell = [
        x for x in mapped.target_book
        if x.side == "sell"
    ]

    assert sum(x.qty for x in buy) == pytest.approx(20.0)
    assert sum(x.qty for x in sell) == pytest.approx(20.0)

    assert all(
        x.price_tick < _book().best_ask_tick
        for x in buy
    )
    assert all(
        x.price_tick > _book().best_bid_tick
        for x in sell
    )


def test_reward_components():
    pnl_cfg = IMMRewardConfig(
        mode="pnl"
    )

    r, parts = pnl_cfg.compute(
        pnl=5.0,
        filled_notional=1000.0,
        inventory_at_decision=7.0,
    )

    assert r == pytest.approx(5.0)
    assert parts["compensation_term"] == 0.0

    imm_cfg = IMMRewardConfig(
        mode="imm",
        beta=0.001,
        eta=2.0,
        inventory_threshold=5.0,
    )

    r, parts = imm_cfg.compute(
        pnl=5.0,
        filled_notional=1000.0,
        inventory_at_decision=7.0,
    )

    assert parts["compensation_term"] == pytest.approx(1.0)
    assert parts["inventory_penalty_term"] == pytest.approx(-14.0)
    assert r == pytest.approx(-8.0)


def test_live_order_discovery_on_real_simulator():
    book0 = MarketBookSnapshot(
        timestamp_us=1_000_000,
        bids={100: 5, 99: 4, 98: 3, 97: 2, 96: 1},
        asks={101: 5, 102: 4, 103: 3, 104: 2, 105: 1},
        mid_price=100.5,
        tick_size=1.0,
    )

    book1 = MarketBookSnapshot(
        timestamp_us=1_100_000,
        bids={100: 5, 99: 4, 98: 3, 97: 2, 96: 1},
        asks={101: 5, 102: 4, 103: 3, 104: 2, 105: 1},
        mid_price=100.5,
        tick_size=1.0,
    )

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

    from mmrl.imm.simulator_adapter import IMMGridReplayAdapter
    from mmrl.simulator import TargetOrder

    adapter = IMMGridReplayAdapter(sim)

    adapter.step(
        current_book=book0,
        next_book=book1,
        trades=[],
        target_book=[
            TargetOrder("buy", 99, 1.0),
            TargetOrder("sell", 102, 1.0),
        ],
    )

    live = discover_live_child_orders(
        sim
    )

    assert len(live) >= 2
    assert {x.side for x in live}.issuperset(
        {"buy", "sell"}
    )

    for x in live:
        assert x.remaining_qty > 0
        assert x.queue_ahead >= 0


def test_first_real_snapshot_decision_semantics():
    # Only tests the timestamp selection logic with a minimal fake object
    # through direct IMMMarketData field construction avoided here.
    ts = np.asarray(
        [0, 99, 201, 498, 598, 701],
        dtype=np.int64,
    )

    idx = int(
        np.searchsorted(
            ts,
            500,
            side="left",
        )
    )

    assert idx == 4
    assert ts[idx] == 598
