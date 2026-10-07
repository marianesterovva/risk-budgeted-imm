import pytest

from conftest import make_book
from mmrl.simulator.replay import GapBoundaryError, ReplaySimulator
from mmrl.simulator.types import MarketBookSnapshot, TargetOrder, TradeEvent


def test_market_data_gap_is_episode_boundary():
    current = make_book(
        ts_us=1_000_000
    )

    next_book = make_book(
        ts_us=7_000_000
    )

    sim = ReplaySimulator(
        max_gap_ms=5000,
        maker_fee_rate=0.0,
    )

    with pytest.raises(GapBoundaryError):
        sim.step(
            current_book=current,
            next_book=next_book,
            trades=[],
            target_book=[],
        )


def test_simple_replay_step():
    current = make_book(
        ts_us=1_000_000,
        bid_qty=2.0,
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
    )

    trades = [
        TradeEvent(
            timestamp_us=1_000_001,
            price_tick=100,
            qty=3.0,
            is_buyer_maker=True,
        )
    ]

    result = sim.step(
        current_book=current,
        next_book=next_book,
        trades=trades,
        target_book=[
            TargetOrder(
                side="buy",
                price_tick=100,
                qty=2.0,
            )
        ],
    )

    assert result.placed_qty == pytest.approx(2.0)
    assert result.filled_qty == pytest.approx(1.0)
    assert result.inventory == pytest.approx(1.0)
    assert result.cash == pytest.approx(-100.0)
