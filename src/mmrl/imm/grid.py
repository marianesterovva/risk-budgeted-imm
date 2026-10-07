from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from mmrl.state.reference_price import (
    build_grid_ticks,
    candidate_reference_ref2,
    queue_at_tick,
)


GRID_LEVELS_K5 = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class GridSlot:
    level: int
    price_tick: int
    qty: float
    known: bool
    status: str


@dataclass(frozen=True)
class TargetPriceStatus:
    side: str
    price_tick: int
    allowed: bool
    known: bool
    marketable: bool
    status: str
    queue_ahead: float


class CandidateReferencePriceTracker:
    """Current IMM candidate reference, updated on every real snapshot."""

    def __init__(self, *, initial_even_tie_break: str = "lower") -> None:
        self.initial_even_tie_break = str(initial_even_tie_break)
        self.ref2: Optional[int] = None

    def clear(self) -> None:
        self.ref2 = None

    @staticmethod
    def _best_ticks(
        bids: Mapping[int, float],
        asks: Mapping[int, float],
    ) -> tuple[int, int]:
        if not bids or not asks:
            raise ValueError("bids and asks must be non-empty")
        best_bid = max(int(x) for x in bids)
        best_ask = min(int(x) for x in asks)
        if best_bid >= best_ask:
            raise ValueError(
                f"crossed/locked book: best_bid={best_bid}, best_ask={best_ask}"
            )
        return best_bid, best_ask

    def update(
        self,
        bids: Mapping[int, float],
        asks: Mapping[int, float],
    ) -> int:
        best_bid, best_ask = self._best_ticks(bids, asks)
        candidate = candidate_reference_ref2(
            best_bid,
            best_ask,
            previous_ref2=self.ref2,
            initial_even_tie_break=self.initial_even_tie_break,
        )
        self.ref2 = int(candidate)
        return int(candidate)

    def grid_ticks(self, *, k: int = 5) -> dict[int, int]:
        if self.ref2 is None:
            raise RuntimeError("reference tracker is not initialized")
        return build_grid_ticks(self.ref2, k=int(k))


def build_grid_slots(
    bids: Mapping[int, float],
    asks: Mapping[int, float],
    ref2: int,
    *,
    k: int = 5,
) -> list[GridSlot]:
    ticks = build_grid_ticks(int(ref2), k=int(k))
    out = []
    for level in list(range(-k, 0)) + list(range(1, k + 1)):
        price_tick = int(ticks[level])
        q = queue_at_tick(price_tick, bids, asks)
        out.append(
            GridSlot(
                level=int(level),
                price_tick=price_tick,
                qty=float(q.qty),
                known=bool(q.known),
                status=str(q.status),
            )
        )
    return out


def price_tick_to_grid_level(
    ref2: int,
    price_tick: int,
    *,
    k: int = 5,
) -> Optional[int]:
    diff2 = 2 * int(price_tick) - int(ref2)
    if diff2 == 0 or diff2 % 2 == 0:
        return None
    i = (abs(diff2) + 1) // 2
    if i < 1 or i > int(k):
        return None
    return int(i if diff2 > 0 else -i)


def classify_side_price(
    *,
    side: str,
    price_tick: int,
    bids: Mapping[int, float],
    asks: Mapping[int, float],
) -> TargetPriceStatus:
    side = str(side).lower()
    price_tick = int(price_tick)
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if not bids or not asks:
        raise ValueError("bids and asks must be non-empty")

    best_bid = max(int(x) for x in bids)
    best_ask = min(int(x) for x in asks)
    deepest_bid = min(int(x) for x in bids)
    deepest_ask = max(int(x) for x in asks)

    if best_bid >= best_ask:
        raise ValueError("crossed/locked historical book")

    if side == "buy":
        if price_tick >= best_ask:
            return TargetPriceStatus(
                side, price_tick, False, True, True,
                "marketable_buy", 0.0,
            )
        if price_tick in bids:
            return TargetPriceStatus(
                side, price_tick, True, True, False,
                "occupied_bid", max(0.0, float(bids[price_tick])),
            )
        if deepest_bid <= price_tick < best_ask:
            status = (
                "known_empty_inside_spread"
                if price_tick > best_bid
                else "known_empty_bid_range"
            )
            return TargetPriceStatus(
                side, price_tick, True, True, False, status, 0.0
            )
        return TargetPriceStatus(
            side, price_tick, False, False, False,
            "unknown_below_visible_bid_depth", 0.0,
        )

    if price_tick <= best_bid:
        return TargetPriceStatus(
            side, price_tick, False, True, True,
            "marketable_sell", 0.0,
        )
    if price_tick in asks:
        return TargetPriceStatus(
            side, price_tick, True, True, False,
            "occupied_ask", max(0.0, float(asks[price_tick])),
        )
    if best_bid < price_tick <= deepest_ask:
        status = (
            "known_empty_inside_spread"
            if price_tick < best_ask
            else "known_empty_ask_range"
        )
        return TargetPriceStatus(
            side, price_tick, True, True, False, status, 0.0
        )
    return TargetPriceStatus(
        side, price_tick, False, False, False,
        "unknown_above_visible_ask_depth", 0.0,
    )
