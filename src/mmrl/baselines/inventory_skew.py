from __future__ import annotations

import math

from mmrl.simulator.types import MarketBookSnapshot, TargetOrder

from .base import ScriptedPolicy, clip_passive_tick


class InventorySkewPolicy(ScriptedPolicy):
    """
    Simple inventory-aware benchmark.

    q_norm = inventory / quote_qty

    Long inventory:
        keep ask at L1, move bid outward by skew ticks.

    Short inventory:
        keep bid at L1, move ask outward by skew ticks.

    This is intentionally simpler than Avellaneda-Stoikov and uses no
    predictive signal.
    """

    name = "inventory_skew"

    def __init__(
        self,
        *,
        skew_ticks_per_inventory_unit: float = 0.5,
        max_skew_ticks: int = 5,
    ) -> None:
        if skew_ticks_per_inventory_unit < 0:
            raise ValueError(
                "skew_ticks_per_inventory_unit must be non-negative."
            )

        if max_skew_ticks < 0:
            raise ValueError("max_skew_ticks must be non-negative.")

        self.skew_ticks_per_inventory_unit = float(
            skew_ticks_per_inventory_unit
        )
        self.max_skew_ticks = int(max_skew_ticks)

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

        q_norm = float(inventory) / float(quote_qty)

        skew_ticks = min(
            self.max_skew_ticks,
            int(
                math.floor(
                    abs(q_norm)
                    * self.skew_ticks_per_inventory_unit
                    + 0.5
                )
            ),
        )

        bid_tick = book.best_bid_tick
        ask_tick = book.best_ask_tick

        if q_norm > 0:
            # Long inventory: discourage additional buying.
            bid_tick -= skew_ticks

        elif q_norm < 0:
            # Short inventory: discourage additional selling.
            ask_tick += skew_ticks

        bid_tick = clip_passive_tick(
            side="buy",
            desired_tick=bid_tick,
            book=book,
        )

        ask_tick = clip_passive_tick(
            side="sell",
            desired_tick=ask_tick,
            book=book,
        )

        return [
            TargetOrder("buy", bid_tick, float(quote_qty)),
            TargetOrder("sell", ask_tick, float(quote_qty)),
        ]
