import pytest

from mmrl.simulator.accounting import Portfolio
from mmrl.simulator.types import Fill


def test_buy_and_sell_accounting_with_fee():
    portfolio = Portfolio(
        maker_fee_rate=0.001
    )

    buy = Fill(
        order_id=1,
        timestamp_us=1,
        side="buy",
        price_tick=100,
        price=100.0,
        qty=2.0,
        notional=200.0,
        fee=0.2,
        reason="exact_queue",
    )

    portfolio.apply_fill(buy)

    assert portfolio.inventory == pytest.approx(2.0)
    assert portfolio.cash == pytest.approx(-200.2)

    sell = Fill(
        order_id=2,
        timestamp_us=2,
        side="sell",
        price_tick=101,
        price=101.0,
        qty=1.0,
        notional=101.0,
        fee=0.101,
        reason="exact_queue",
    )

    portfolio.apply_fill(sell)

    assert portfolio.inventory == pytest.approx(1.0)
    assert portfolio.cash == pytest.approx(-99.301)

    assert portfolio.equity(101.0) == pytest.approx(1.699)


def test_negative_maker_fee_is_rebate():
    portfolio = Portfolio(
        maker_fee_rate=-0.0001
    )

    assert portfolio.fee_for(
        price=100.0,
        qty=2.0,
    ) == pytest.approx(-0.02)
