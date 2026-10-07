
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mmrl.simulator import (
    MarketBookSnapshot,
    TargetOrder,
)

from .grid import classify_side_price


@dataclass(frozen=True)
class LTIICParams:
    a_ticks: float
    b_ticks_per_inventory_N: float
    c_ticks_per_trend: float
    d_inventory_N: float


@dataclass(frozen=True)
class LTIICDecision:
    params: LTIICParams

    inventory: float
    inventory_N: float
    trend: int

    desired_bid_tick_float: Optional[float]
    desired_ask_tick_float: Optional[float]

    actual_bid_tick: Optional[int]
    actual_ask_tick: Optional[int]

    quote_bid: bool
    quote_ask: bool

    bid_projected: bool
    ask_projected: bool

    target_book: tuple[TargetOrder, ...]

    @property
    def one_sided(self) -> bool:
        return bool(
            self.quote_bid
            != self.quote_ask
        )


def _observable_passive_ticks(
    *,
    side: str,
    book: MarketBookSnapshot,
) -> list[int]:
    side = str(side).lower()

    if side == "buy":
        lo = min(
            int(x)
            for x in book.bids
        )
        hi = int(
            book.best_ask_tick
        ) - 1
    elif side == "sell":
        lo = int(
            book.best_bid_tick
        ) + 1
        hi = max(
            int(x)
            for x in book.asks
        )
    else:
        raise ValueError(
            "side must be buy/sell"
        )

    out = []

    for tick in range(
        int(lo),
        int(hi) + 1,
    ):
        status = classify_side_price(
            side=side,
            price_tick=int(tick),
            bids=book.bids,
            asks=book.asks,
        )

        if status.allowed:
            out.append(
                int(tick)
            )

    return out


def project_ltiic_quote_tick(
    *,
    side: str,
    desired_tick_float: float,
    book: MarketBookSnapshot,
) -> tuple[int, bool]:
    """
    Quantize and project a LTIIC quote into the observable passive top-5 domain.

    BUY:
      desired continuous price is floored to avoid becoming more aggressive
      through rounding.

    SELL:
      desired continuous price is ceiled.

    If the resulting tick is outside the observable depth-limited passive domain,
    project to the nearest observable passive tick.

    This projection is a depth-5 replay adaptation, not part of Eq. (7).
    """
    side = str(side).lower()

    if side == "buy":
        desired_integer = int(
            math.floor(
                float(desired_tick_float)
                + 1e-12
            )
        )
    elif side == "sell":
        desired_integer = int(
            math.ceil(
                float(desired_tick_float)
                - 1e-12
            )
        )
    else:
        raise ValueError(
            "side must be buy/sell"
        )

    status = classify_side_price(
        side=side,
        price_tick=desired_integer,
        bids=book.bids,
        asks=book.asks,
    )

    if status.allowed:
        return (
            int(desired_integer),
            False,
        )

    valid = _observable_passive_ticks(
        side=side,
        book=book,
    )

    if not valid:
        raise RuntimeError(
            f"no observable passive ticks for side={side}"
        )

    projected = min(
        valid,
        key=lambda x: (
            abs(
                float(x)
                - float(desired_tick_float)
            ),
            -x if side == "buy" else x,
        ),
    )

    return (
        int(projected),
        True,
    )


class LTIICPolicy:
    """
    IMM Eq. (7) expert with inventory constraints.

    Cross-asset adaptation:
      inventory enters Eq. (7) as z / N_a.
    """

    def __init__(
        self,
        *,
        quote_size: float,
        params: LTIICParams,
    ) -> None:
        self.quote_size = float(
            quote_size
        )
        self.params = params

        if self.quote_size <= 0:
            raise ValueError(
                "quote_size must be positive"
            )

        if self.params.a_ticks <= 0:
            raise ValueError(
                "a_ticks must be positive"
            )

        if self.params.d_inventory_N <= 0:
            raise ValueError(
                "d_inventory_N must be positive"
            )

    def decide(
        self,
        *,
        book: MarketBookSnapshot,
        inventory: float,
        trend: int,
    ) -> LTIICDecision:
        trend = int(trend)

        if trend not in {-1, 0, 1}:
            raise ValueError(
                f"trend must be -1/0/+1, got {trend}"
            )

        inventory_N = (
            float(inventory)
            / self.quote_size
        )

        p = self.params

        common_shift = (
            float(
                p.b_ticks_per_inventory_N
            )
            * inventory_N
            + float(
                p.c_ticks_per_trend
            )
            * float(trend)
        )

        mid_tick = (
            float(book.mid_price)
            / float(book.tick_size)
        )

        desired_ask = (
            mid_tick
            + float(p.a_ticks)
            + common_shift
        )

        desired_bid = (
            mid_tick
            - float(p.a_ticks)
            + common_shift
        )

        # Paper inventory constraints.
        if inventory_N >= float(
            p.d_inventory_N
        ):
            quote_bid = False
            quote_ask = True
        elif inventory_N <= -float(
            p.d_inventory_N
        ):
            quote_bid = True
            quote_ask = False
        else:
            quote_bid = True
            quote_ask = True

        target = []

        actual_bid = None
        actual_ask = None

        bid_projected = False
        ask_projected = False

        if quote_bid:
            actual_bid, bid_projected = (
                project_ltiic_quote_tick(
                    side="buy",
                    desired_tick_float=(
                        desired_bid
                    ),
                    book=book,
                )
            )

            target.append(
                TargetOrder(
                    "buy",
                    int(actual_bid),
                    float(
                        self.quote_size
                    ),
                )
            )

        if quote_ask:
            actual_ask, ask_projected = (
                project_ltiic_quote_tick(
                    side="sell",
                    desired_tick_float=(
                        desired_ask
                    ),
                    book=book,
                )
            )

            target.append(
                TargetOrder(
                    "sell",
                    int(actual_ask),
                    float(
                        self.quote_size
                    ),
                )
            )

        return LTIICDecision(
            params=p,
            inventory=float(
                inventory
            ),
            inventory_N=float(
                inventory_N
            ),
            trend=int(
                trend
            ),
            desired_bid_tick_float=(
                float(desired_bid)
                if quote_bid
                else None
            ),
            desired_ask_tick_float=(
                float(desired_ask)
                if quote_ask
                else None
            ),
            actual_bid_tick=(
                None
                if actual_bid is None
                else int(actual_bid)
            ),
            actual_ask_tick=(
                None
                if actual_ask is None
                else int(actual_ask)
            ),
            quote_bid=bool(
                quote_bid
            ),
            quote_ask=bool(
                quote_ask
            ),
            bid_projected=bool(
                bid_projected
            ),
            ask_projected=bool(
                ask_projected
            ),
            target_book=tuple(
                target
            ),
        )
