from mmrl.simulator.types import MarketBookSnapshot


def make_book(
    *,
    ts_us=1_000_000,
    bid1=100,
    ask1=102,
    bid_qty=10.0,
    ask_qty=10.0,
    tick_size=1.0,
):
    bids = {
        bid1 - i: bid_qty
        for i in range(5)
    }

    asks = {
        ask1 + i: ask_qty
        for i in range(5)
    }

    return MarketBookSnapshot(
        timestamp_us=ts_us,
        bids=bids,
        asks=asks,
        mid_price=(bid1 + ask1) * 0.5 * tick_size,
        tick_size=tick_size,
    )


import pytest

from mmrl.simulator import (
    MarketBookSnapshot,
    ReplaySimulator,
    TargetOrder,
)


def test_target_none_holds_existing_order_without_rebalance():
    current = make_book(
        ts_us=1_000_000,
        bid_qty=10.0,
    )

    next_book = MarketBookSnapshot(
        timestamp_us=1_100_000,
        bids=dict(current.bids),
        asks=dict(current.asks),
        mid_price=current.mid_price,
        tick_size=current.tick_size,
    )

    sim = ReplaySimulator(
        max_gap_ms=5000,
        maker_fee_rate=0.0,
        trade_through_fill=False,
    )

    first = sim.step(
        current_book=current,
        next_book=next_book,
        trades=[],
        target_book=[
            TargetOrder(
                side="buy",
                price_tick=current.best_bid_tick,
                qty=1.0,
            )
        ],
    )

    assert first.placed_qty == pytest.approx(1.0)
    assert len(sim.order_manager.active_orders) == 1

    current2 = next_book

    next2 = MarketBookSnapshot(
        timestamp_us=1_200_000,
        bids=dict(current.bids),
        asks=dict(current.asks),
        mid_price=current.mid_price,
        tick_size=current.tick_size,
    )

    second = sim.step(
        current_book=current2,
        next_book=next2,
        trades=[],
        target_book=None,
    )

    assert second.placed_qty == pytest.approx(0.0)
    assert second.canceled_qty == pytest.approx(0.0)
    assert len(sim.order_manager.active_orders) == 1
