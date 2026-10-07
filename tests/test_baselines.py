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


from mmrl.baselines import (
    AvellanedaStoikovPolicy,
    FixedQuotePolicy,
    InventorySkewPolicy,
    NoTradePolicy,
)


def test_no_trade_returns_empty_target():
    book = make_book()
    policy = NoTradePolicy()

    assert policy.target_book(
        book=book,
        inventory=0.0,
        quote_qty=1.0,
    ) == []


def test_fixed_quote_uses_l1():
    book = make_book()
    policy = FixedQuotePolicy()

    target = policy.target_book(
        book=book,
        inventory=0.0,
        quote_qty=2.0,
    )

    assert target[0].price_tick == book.best_bid_tick
    assert target[1].price_tick == book.best_ask_tick


def test_inventory_skew_discourages_long_inventory_buying():
    book = make_book()
    policy = InventorySkewPolicy(
        skew_ticks_per_inventory_unit=1.0,
        max_skew_ticks=5,
    )

    target = policy.target_book(
        book=book,
        inventory=2.0,
        quote_qty=1.0,
    )

    assert target[0].price_tick < book.best_bid_tick
    assert target[1].price_tick == book.best_ask_tick


def test_inventory_skew_discourages_short_inventory_selling():
    book = make_book()
    policy = InventorySkewPolicy(
        skew_ticks_per_inventory_unit=1.0,
        max_skew_ticks=5,
    )

    target = policy.target_book(
        book=book,
        inventory=-2.0,
        quote_qty=1.0,
    )

    assert target[0].price_tick == book.best_bid_tick
    assert target[1].price_tick > book.best_ask_tick


def test_as_returns_passive_order_pair():
    book = make_book()
    policy = AvellanedaStoikovPolicy()

    policy.observe(book)

    target = policy.target_book(
        book=book,
        inventory=0.0,
        quote_qty=1.0,
        time_to_episode_end_s=60.0,
    )

    bid = target[0].price_tick
    ask = target[1].price_tick

    assert bid < book.best_ask_tick
    assert ask > book.best_bid_tick
    assert bid < ask
