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
