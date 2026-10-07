import pytest

from conftest import make_book
from mmrl.simulator.order_manager import OrderManager
from mmrl.simulator.types import TargetOrder


def test_size_increase_preserves_old_priority():
    book = make_book()
    om = OrderManager()

    om.rebalance(
        target_book=[TargetOrder("buy", 100, 2.0)],
        current_book=book,
        timestamp_us=book.timestamp_us,
    )

    first = om.active_orders[0]
    assert first.queue_ahead_qty == pytest.approx(10.0)

    om.rebalance(
        target_book=[TargetOrder("buy", 100, 3.0)],
        current_book=book,
        timestamp_us=book.timestamp_us + 1000,
    )

    orders = om.active_orders

    assert len(orders) == 2
    assert orders[0].order_id == first.order_id
    assert orders[0].remaining_qty == pytest.approx(2.0)

    # New 1.0 child sits behind historical 10 + earlier own 2.
    assert orders[1].remaining_qty == pytest.approx(1.0)
    assert orders[1].queue_ahead_qty == pytest.approx(12.0)


def test_size_decrease_cancels_newest_first():
    book = make_book()
    om = OrderManager()

    om.rebalance(
        target_book=[TargetOrder("buy", 100, 2.0)],
        current_book=book,
        timestamp_us=book.timestamp_us,
    )

    old = om.active_orders[0]

    om.rebalance(
        target_book=[TargetOrder("buy", 100, 3.0)],
        current_book=book,
        timestamp_us=book.timestamp_us + 1000,
    )

    newest = om.active_orders[-1]

    placed, canceled = om.rebalance(
        target_book=[TargetOrder("buy", 100, 2.0)],
        current_book=book,
        timestamp_us=book.timestamp_us + 2000,
    )

    assert placed == pytest.approx(0.0)
    assert canceled == pytest.approx(1.0)

    active = om.active_orders

    assert len(active) == 1
    assert active[0].order_id == old.order_id
    assert not newest.active


def test_cancel_old_order_moves_later_order_forward():
    book = make_book()
    om = OrderManager()

    first = om.place(
        side="buy",
        price_tick=100,
        qty=2.0,
        book=book,
        timestamp_us=book.timestamp_us,
    )

    second = om.place(
        side="buy",
        price_tick=100,
        qty=1.0,
        book=book,
        timestamp_us=book.timestamp_us + 100,
    )

    assert second.queue_ahead_qty == pytest.approx(12.0)

    om.cancel_order(first.order_id)

    assert second.queue_ahead_qty == pytest.approx(10.0)


def test_marketable_order_rejected():
    book = make_book()
    om = OrderManager(
        allow_marketable_orders=False
    )

    with pytest.raises(ValueError):
        om.place(
            side="buy",
            price_tick=102,
            qty=1.0,
            book=book,
            timestamp_us=book.timestamp_us,
        )
