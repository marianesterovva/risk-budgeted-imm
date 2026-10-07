from __future__ import annotations

from mmrl.simulator.types import MarketBookSnapshot, TargetOrder

from .base import ScriptedPolicy


class NoTradePolicy(ScriptedPolicy):
    name = "no_trade"

    def target_book(
        self,
        *,
        book: MarketBookSnapshot,
        inventory: float,
        quote_qty: float,
        time_to_episode_end_s: float | None = None,
    ) -> list[TargetOrder]:
        return []
