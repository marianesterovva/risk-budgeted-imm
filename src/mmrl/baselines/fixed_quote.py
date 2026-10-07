from __future__ import annotations

from mmrl.simulator.types import MarketBookSnapshot, TargetOrder

from .base import ScriptedPolicy, clip_passive_tick


class FixedQuotePolicy(ScriptedPolicy):
    name = "fixed_quote"

    def __init__(
        self,
        *,
        bid_offset_ticks: int = 0,
        ask_offset_ticks: int = 0,
    ) -> None:
        if bid_offset_ticks < 0:
            raise ValueError("bid_offset_ticks must be >= 0.")
        if ask_offset_ticks < 0:
            raise ValueError("ask_offset_ticks must be >= 0.")

        self.bid_offset_ticks = int(bid_offset_ticks)
        self.ask_offset_ticks = int(ask_offset_ticks)

    def target_book(
        self,
        *,
        book: MarketBookSnapshot,
        inventory: float,
        quote_qty: float,
        time_to_episode_end_s: float | None = None,
    ) -> list[TargetOrder]:
        if quote_qty <= 0:
            raise ValueError("quote_qty must be positive.")

        bid_tick = clip_passive_tick(
            side="buy",
            desired_tick=book.best_bid_tick - self.bid_offset_ticks,
            book=book,
        )

        ask_tick = clip_passive_tick(
            side="sell",
            desired_tick=book.best_ask_tick + self.ask_offset_ticks,
            book=book,
        )

        return [
            TargetOrder("buy", bid_tick, float(quote_qty)),
            TargetOrder("sell", ask_tick, float(quote_qty)),
        ]
