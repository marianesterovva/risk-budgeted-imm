from mmrl.baselines.base import clip_passive_tick
from mmrl.simulator.types import MarketBookSnapshot


def make_gappy_book():
    return MarketBookSnapshot(
        timestamp_us=1_000_000,
        bids={
            100: 5.0,
            98: 5.0,
            97: 5.0,
            95: 5.0,
            94: 5.0,
        },
        asks={
            102: 5.0,
            104: 5.0,
            105: 5.0,
            107: 5.0,
            110: 5.0,
        },
        mid_price=101.0,
        tick_size=1.0,
    )


def test_buy_outward_quote_snaps_to_real_visible_level():
    book = make_gappy_book()

    # 99 lies numerically inside visible bid range but is not an actual level.
    px = clip_passive_tick(
        side="buy",
        desired_tick=99,
        book=book,
    )

    assert px == 98
    assert px in book.bids


def test_sell_outward_quote_snaps_to_real_visible_level():
    book = make_gappy_book()

    # 103 is not an actual ask level.
    px = clip_passive_tick(
        side="sell",
        desired_tick=103,
        book=book,
    )

    assert px == 104
    assert px in book.asks


def test_inside_spread_quote_is_preserved():
    book = make_gappy_book()

    buy_px = clip_passive_tick(
        side="buy",
        desired_tick=101,
        book=book,
    )

    sell_px = clip_passive_tick(
        side="sell",
        desired_tick=101,
        book=book,
    )

    assert buy_px == 101
    assert sell_px == 101


def test_deeper_than_l5_clamps_to_deepest_visible():
    book = make_gappy_book()

    assert clip_passive_tick(
        side="buy",
        desired_tick=80,
        book=book,
    ) == 94

    assert clip_passive_tick(
        side="sell",
        desired_tick=120,
        book=book,
    ) == 110
