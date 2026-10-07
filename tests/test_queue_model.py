import pytest

from conftest import make_book
from mmrl.simulator.order_manager import OrderManager
from mmrl.simulator.queue_model import ProportionalQueueModel
from mmrl.simulator.types import MarketBookSnapshot, TradeEvent


def test_proportional_cancellation_reduces_external_queue():
    current = make_book(
        bid_qty=10.0
    )

    next_book = MarketBookSnapshot(
        timestamp_us=current.timestamp_us + 100_000,
        bids={
            100: 5.0,
            99: 10.0,
            98: 10.0,
            97: 10.0,
            96: 10.0,
        },
        asks=dict(current.asks),
        mid_price=current.mid_price,
        tick_size=current.tick_size,
    )

    om = OrderManager()

    order = om.place(
        side="buy",
        price_tick=100,
        qty=1.0,
        book=current,
        timestamp_us=current.timestamp_us,
    )

    model = ProportionalQueueModel()

    model.reconcile(
        current_book=current,
        next_book=next_book,
        trades=[],
        orders=om.active_orders,
    )

    # net cancellation = 5 out of initial 10 => external queue halves
    assert order.queue_ahead_qty == pytest.approx(5.0)


def test_historical_addition_does_not_increase_queue():
    current = make_book(
        bid_qty=10.0
    )

    next_book = MarketBookSnapshot(
        timestamp_us=current.timestamp_us + 100_000,
        bids={
            100: 20.0,
            99: 10.0,
            98: 10.0,
            97: 10.0,
            96: 10.0,
        },
        asks=dict(current.asks),
        mid_price=current.mid_price,
        tick_size=current.tick_size,
    )

    om = OrderManager()

    order = om.place(
        side="buy",
        price_tick=100,
        qty=1.0,
        book=current,
        timestamp_us=current.timestamp_us,
    )

    ProportionalQueueModel().reconcile(
        current_book=current,
        next_book=next_book,
        trades=[],
        orders=om.active_orders,
    )

    assert order.queue_ahead_qty == pytest.approx(10.0)


def test_missing_next_depth_level_skips_reconciliation():
    current = make_book(
        bid_qty=10.0
    )

    next_book = MarketBookSnapshot(
        timestamp_us=current.timestamp_us + 100_000,
        bids={
            99: 10.0,
            98: 10.0,
            97: 10.0,
            96: 10.0,
            95: 10.0,
        },
        asks=dict(current.asks),
        mid_price=current.mid_price,
        tick_size=current.tick_size,
    )

    om = OrderManager(
        allow_beyond_depth5=True
    )

    order = om.place(
        side="buy",
        price_tick=100,
        qty=1.0,
        book=current,
        timestamp_us=current.timestamp_us,
    )

    before = order.queue_ahead_qty

    ProportionalQueueModel().reconcile(
        current_book=current,
        next_book=next_book,
        trades=[],
        orders=om.active_orders,
    )

    assert order.queue_ahead_qty == pytest.approx(before)
