import pytest

from conftest import make_book
from mmrl.simulator.accounting import Portfolio
from mmrl.simulator.fill_model import FillModel
from mmrl.simulator.order_manager import OrderManager
from mmrl.simulator.types import TradeEvent


def setup_buy_order(queue_qty=10.0, order_qty=2.0):
    book = make_book(
        bid_qty=queue_qty
    )

    om = OrderManager()

    order = om.place(
        side="buy",
        price_tick=100,
        qty=order_qty,
        book=book,
        timestamp_us=book.timestamp_us,
    )

    portfolio = Portfolio(
        maker_fee_rate=0.0
    )

    model = FillModel()

    return book, om, order, portfolio, model


def test_trade_consumes_only_part_of_queue():
    book, om, order, portfolio, model = setup_buy_order()

    trade = TradeEvent(
        timestamp_us=book.timestamp_us + 1,
        price_tick=100,
        qty=6.0,
        is_buyer_maker=True,
    )

    fills = model.process_trade(
        trade=trade,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert fills == []
    assert order.queue_ahead_qty == pytest.approx(4.0)
    assert order.remaining_qty == pytest.approx(2.0)


def test_trade_reaches_agent_partial_fill():
    book, om, order, portfolio, model = setup_buy_order()

    first = TradeEvent(
        timestamp_us=book.timestamp_us + 1,
        price_tick=100,
        qty=6.0,
        is_buyer_maker=True,
    )

    model.process_trade(
        trade=first,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    second = TradeEvent(
        timestamp_us=book.timestamp_us + 2,
        price_tick=100,
        qty=5.0,
        is_buyer_maker=True,
    )

    fills = model.process_trade(
        trade=second,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert len(fills) == 1
    assert fills[0].reason == "exact_queue"
    assert fills[0].qty == pytest.approx(1.0)

    assert order.queue_ahead_qty == pytest.approx(0.0)
    assert order.remaining_qty == pytest.approx(1.0)


def test_trade_exceeds_agent_full_fill():
    book, om, order, portfolio, model = setup_buy_order()

    trade = TradeEvent(
        timestamp_us=book.timestamp_us + 1,
        price_tick=100,
        qty=15.0,
        is_buyer_maker=True,
    )

    fills = model.process_trade(
        trade=trade,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert len(fills) == 1
    assert fills[0].qty == pytest.approx(2.0)

    assert not order.active
    assert order.remaining_qty == pytest.approx(0.0)


def test_trade_through_full_fill():
    book, om, order, portfolio, model = setup_buy_order()

    trade = TradeEvent(
        timestamp_us=book.timestamp_us + 1,
        price_tick=99,
        qty=0.1,
        is_buyer_maker=True,
    )

    fills = model.process_trade(
        trade=trade,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert len(fills) == 1
    assert fills[0].reason == "trade_through"
    assert fills[0].qty == pytest.approx(2.0)


def test_opposite_side_trade_does_not_fill_bid():
    book, om, order, portfolio, model = setup_buy_order()

    trade = TradeEvent(
        timestamp_us=book.timestamp_us + 1,
        price_tick=102,
        qty=100.0,
        is_buyer_maker=False,
    )

    fills = model.process_trade(
        trade=trade,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert fills == []
    assert order.remaining_qty == pytest.approx(2.0)


def test_same_timestamp_new_order_cannot_fill():
    book, om, order, portfolio, model = setup_buy_order()

    trade = TradeEvent(
        timestamp_us=book.timestamp_us,
        price_tick=99,
        qty=100.0,
        is_buyer_maker=True,
    )

    fills = model.process_trade(
        trade=trade,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert fills == []
    assert order.remaining_qty == pytest.approx(2.0)


def test_two_own_orders_same_price_share_single_queue_axis():
    book = make_book(
        bid_qty=10.0
    )

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
        timestamp_us=book.timestamp_us + 1,
    )

    portfolio = Portfolio(
        maker_fee_rate=0.0
    )

    model = FillModel()

    trade = TradeEvent(
        timestamp_us=book.timestamp_us + 2,
        price_tick=100,
        qty=13.0,
        is_buyer_maker=True,
    )

    fills = model.process_trade(
        trade=trade,
        orders=om.active_orders,
        book=book,
        portfolio=portfolio,
    )

    assert sum(f.qty for f in fills) == pytest.approx(3.0)
    assert not first.active
    assert not second.active
