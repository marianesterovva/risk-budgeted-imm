
import pytest

from mmrl.imm.ltiic import (
    LTIICParams,
    LTIICPolicy,
)
from mmrl.simulator import (
    MarketBookSnapshot,
)


def _book():
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


def _policy():
    return LTIICPolicy(
        quote_size=2.0,
        params=LTIICParams(
            a_ticks=2.0,
            b_ticks_per_inventory_N=-0.5,
            c_ticks_per_trend=1.0,
            d_inventory_N=3.0,
        ),
    )


def test_eq7_raw_quote_formula():
    p = _policy()

    # inventory = +2 units = +1 N.
    d = p.decide(
        book=_book(),
        inventory=2.0,
        trend=1,
    )

    # common shift = -0.5*1 + 1*1 = +0.5
    assert (
        d.desired_ask_tick_float
        == pytest.approx(
            100.5 + 2.0 + 0.5
        )
    )

    assert (
        d.desired_bid_tick_float
        == pytest.approx(
            100.5 - 2.0 + 0.5
        )
    )


def test_trend_shifts_both_quotes_same_direction():
    p = _policy()

    up = p.decide(
        book=_book(),
        inventory=0.0,
        trend=1,
    )

    down = p.decide(
        book=_book(),
        inventory=0.0,
        trend=-1,
    )

    assert (
        up.desired_bid_tick_float
        > down.desired_bid_tick_float
    )
    assert (
        up.desired_ask_tick_float
        > down.desired_ask_tick_float
    )


def test_negative_b_reduces_long_inventory():
    p = _policy()

    flat = p.decide(
        book=_book(),
        inventory=0.0,
        trend=0,
    )

    long = p.decide(
        book=_book(),
        inventory=4.0,  # +2N
        trend=0,
    )

    assert (
        long.desired_bid_tick_float
        < flat.desired_bid_tick_float
    )

    assert (
        long.desired_ask_tick_float
        < flat.desired_ask_tick_float
    )


def test_long_limit_is_ask_only():
    p = _policy()

    d = p.decide(
        book=_book(),
        inventory=6.0,  # +3N == d
        trend=0,
    )

    assert not d.quote_bid
    assert d.quote_ask
    assert len(d.target_book) == 1
    assert d.target_book[0].side == "sell"


def test_short_limit_is_bid_only():
    p = _policy()

    d = p.decide(
        book=_book(),
        inventory=-6.0,
        trend=0,
    )

    assert d.quote_bid
    assert not d.quote_ask
    assert len(d.target_book) == 1
    assert d.target_book[0].side == "buy"


def test_target_is_passive_and_has_N_quantity():
    p = _policy()

    d = p.decide(
        book=_book(),
        inventory=0.0,
        trend=0,
    )

    assert len(d.target_book) == 2

    for order in d.target_book:
        assert order.qty == pytest.approx(2.0)

        if order.side == "buy":
            assert (
                order.price_tick
                < _book().best_ask_tick
            )
        else:
            assert (
                order.price_tick
                > _book().best_bid_tick
            )
