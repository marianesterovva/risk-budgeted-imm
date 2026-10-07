
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from mmrl.simulator import MarketBookSnapshot, TargetOrder

from .grid import (
    build_grid_slots,
    classify_side_price,
)


GRID_LEVEL_ORDER = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class DecodedIMMAction:
    quoted_mid_ticks: float
    spread_ticks: float
    phi_bid: float
    phi_ask: float


@dataclass(frozen=True)
class MappedIMMAction:
    normalized_action: tuple[float, float, float, float]
    decoded: DecodedIMMAction

    desired_bid_boundary: float
    desired_ask_boundary: float

    desired_bid_levels: tuple[int, ...]
    desired_ask_levels: tuple[int, ...]

    actual_bid_levels: tuple[int, ...]
    actual_ask_levels: tuple[int, ...]

    target_book: tuple[TargetOrder, ...]

    bid_projected: bool
    ask_projected: bool
    bid_projection_grid_steps: int
    ask_projection_grid_steps: int


class IMMActionMapper:
    """
    Two-level IMM action mapping for the depth-5 crypto reference.

    Paper-specified:
      action = (m*, delta*, phi_bid, phi_ask)
      with multi-level volume allocation.

    Documented implementation choices:
      m* range   = [-3, +3] grid ticks
      spread     = [2, 8] grid ticks
      K          = 5
      side qty   = frozen train-only N_a
      passive projection required by current-reference depth-5 adaptation
    """

    def __init__(
        self,
        *,
        quote_size: float,
        k: int = 5,
        quoted_mid_min: float = -3.0,
        quoted_mid_max: float = 3.0,
        spread_min: float = 2.0,
        spread_max: float = 8.0,
        eps: float = 1e-12,
    ) -> None:
        self.quote_size = float(quote_size)
        self.k = int(k)
        self.quoted_mid_min = float(quoted_mid_min)
        self.quoted_mid_max = float(quoted_mid_max)
        self.spread_min = float(spread_min)
        self.spread_max = float(spread_max)
        self.eps = float(eps)

        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.k != 5:
            raise ValueError("reference protocol currently fixes K=5")
        if not self.quoted_mid_min < self.quoted_mid_max:
            raise ValueError("bad quoted-mid bounds")
        if not 0 < self.spread_min <= self.spread_max:
            raise ValueError("bad spread bounds")

        self.levels = tuple(GRID_LEVEL_ORDER)

    @staticmethod
    def _linear(
        u: float,
        lo: float,
        hi: float,
    ) -> float:
        u = float(np.clip(u, -1.0, 1.0))
        return float(lo + 0.5 * (u + 1.0) * (hi - lo))

    def decode_normalized(
        self,
        action: Sequence[float] | np.ndarray,
    ) -> DecodedIMMAction:
        x = np.asarray(action, dtype=np.float64).reshape(-1)
        if x.shape != (4,):
            raise ValueError(
                f"expected normalized action shape (4,), got {x.shape}"
            )

        x = np.clip(x, -1.0, 1.0)

        return DecodedIMMAction(
            quoted_mid_ticks=self._linear(
                x[0],
                self.quoted_mid_min,
                self.quoted_mid_max,
            ),
            spread_ticks=self._linear(
                x[1],
                self.spread_min,
                self.spread_max,
            ),
            phi_bid=float(0.5 * (x[2] + 1.0)),
            phi_ask=float(0.5 * (x[3] + 1.0)),
        )

    @staticmethod
    def _paper_levels(
        *,
        side: str,
        boundary: float,
    ) -> tuple[int, ...]:
        """
        Desired levels before current-book passivity projection.

        Uses the Figure-2 level coordinate:
          -5,-4,-3,-2,-1,+1,+2,+3,+4,+5
        """
        levels = list(GRID_LEVEL_ORDER)

        if side == "buy":
            eligible = [
                int(level)
                for level in levels
                if float(level) <= float(boundary)
            ]

            if not eligible:
                return (-5,)

            inner = max(eligible)
            pos = levels.index(inner)

            out = [inner]
            if pos - 1 >= 0:
                out.append(int(levels[pos - 1]))

            return tuple(out)

        if side == "sell":
            eligible = [
                int(level)
                for level in levels
                if float(level) >= float(boundary)
            ]

            if not eligible:
                return (5,)

            inner = min(eligible)
            pos = levels.index(inner)

            out = [inner]
            if pos + 1 < len(levels):
                out.append(int(levels[pos + 1]))

            return tuple(out)

        raise ValueError("side must be buy/sell")

    @staticmethod
    def _nearest_passive_pair(
        *,
        side: str,
        desired_inner_level: int,
        grid_level_to_tick: dict[int, int],
        current_book: MarketBookSnapshot,
    ) -> tuple[tuple[int, ...], bool, int]:
        """
        Project to nearest passive/observable current grid levels.

        We first find all valid price levels for the side, then choose the one
        closest in GRID ORDER to the desired inner level and its outward
        neighbor if available.
        """
        levels = list(GRID_LEVEL_ORDER)

        valid = []

        for level in levels:
            tick = int(grid_level_to_tick[level])

            status = classify_side_price(
                side=side,
                price_tick=tick,
                bids=current_book.bids,
                asks=current_book.asks,
            )

            if status.allowed:
                valid.append(int(level))

        if not valid:
            raise RuntimeError(
                f"no observable passive IMM grid level for side={side}"
            )

        desired_pos = levels.index(int(desired_inner_level))

        if side == "buy":
            # Prefer the highest price not more aggressive than desired.
            conservative = [
                x for x in valid
                if levels.index(x) <= desired_pos
            ]

            if conservative:
                inner = max(
                    conservative,
                    key=lambda x: levels.index(x),
                )
            else:
                # Current-candidate adaptation forced the desired level
                # outside the passive bid region. Choose nearest valid.
                inner = min(
                    valid,
                    key=lambda x: abs(
                        levels.index(x) - desired_pos
                    ),
                )

            inner_pos = levels.index(inner)

            outward = [
                x for x in valid
                if levels.index(x) < inner_pos
            ]

            pair = [inner]
            if outward:
                pair.append(
                    max(
                        outward,
                        key=lambda x: levels.index(x),
                    )
                )

        else:
            conservative = [
                x for x in valid
                if levels.index(x) >= desired_pos
            ]

            if conservative:
                inner = min(
                    conservative,
                    key=lambda x: levels.index(x),
                )
            else:
                inner = min(
                    valid,
                    key=lambda x: abs(
                        levels.index(x) - desired_pos
                    ),
                )

            inner_pos = levels.index(inner)

            outward = [
                x for x in valid
                if levels.index(x) > inner_pos
            ]

            pair = [inner]
            if outward:
                pair.append(
                    min(
                        outward,
                        key=lambda x: levels.index(x),
                    )
                )

        projected = int(inner) != int(desired_inner_level)

        projection_steps = abs(
            levels.index(int(inner))
            - desired_pos
        )

        return (
            tuple(int(x) for x in pair),
            bool(projected),
            int(projection_steps),
        )

    def _orders_for_side(
        self,
        *,
        side: str,
        levels: tuple[int, ...],
        phi: float,
        level_to_tick: dict[int, int],
    ) -> list[TargetOrder]:
        phi = float(np.clip(phi, 0.0, 1.0))

        if len(levels) == 1:
            allocations = [self.quote_size]
        elif len(levels) == 2:
            allocations = [
                self.quote_size * phi,
                self.quote_size * (1.0 - phi),
            ]
        else:
            raise AssertionError(levels)

        out = []

        for level, qty in zip(levels, allocations):
            if qty <= self.eps:
                continue

            out.append(
                TargetOrder(
                    side,
                    int(level_to_tick[int(level)]),
                    float(qty),
                )
            )

        return out

    def map_decoded(
        self,
        *,
        decoded: DecodedIMMAction,
        normalized_action: Sequence[float],
        ref2: int,
        current_book: MarketBookSnapshot,
    ) -> MappedIMMAction:
        bid_boundary = (
            float(decoded.quoted_mid_ticks)
            - 0.5 * float(decoded.spread_ticks)
        )
        ask_boundary = (
            float(decoded.quoted_mid_ticks)
            + 0.5 * float(decoded.spread_ticks)
        )

        desired_bid = self._paper_levels(
            side="buy",
            boundary=bid_boundary,
        )
        desired_ask = self._paper_levels(
            side="sell",
            boundary=ask_boundary,
        )

        slots = build_grid_slots(
            current_book.bids,
            current_book.asks,
            int(ref2),
            k=self.k,
        )

        level_to_tick = {
            int(slot.level): int(slot.price_tick)
            for slot in slots
        }

        actual_bid, bid_projected, bid_steps = (
            self._nearest_passive_pair(
                side="buy",
                desired_inner_level=int(desired_bid[0]),
                grid_level_to_tick=level_to_tick,
                current_book=current_book,
            )
        )

        actual_ask, ask_projected, ask_steps = (
            self._nearest_passive_pair(
                side="sell",
                desired_inner_level=int(desired_ask[0]),
                grid_level_to_tick=level_to_tick,
                current_book=current_book,
            )
        )

        target = []

        target.extend(
            self._orders_for_side(
                side="buy",
                levels=actual_bid,
                phi=decoded.phi_bid,
                level_to_tick=level_to_tick,
            )
        )

        target.extend(
            self._orders_for_side(
                side="sell",
                levels=actual_ask,
                phi=decoded.phi_ask,
                level_to_tick=level_to_tick,
            )
        )

        norm = tuple(
            float(x)
            for x in np.clip(
                np.asarray(normalized_action, dtype=float),
                -1.0,
                1.0,
            ).reshape(4)
        )

        return MappedIMMAction(
            normalized_action=norm,
            decoded=decoded,
            desired_bid_boundary=float(bid_boundary),
            desired_ask_boundary=float(ask_boundary),
            desired_bid_levels=tuple(desired_bid),
            desired_ask_levels=tuple(desired_ask),
            actual_bid_levels=tuple(actual_bid),
            actual_ask_levels=tuple(actual_ask),
            target_book=tuple(target),
            bid_projected=bool(bid_projected),
            ask_projected=bool(ask_projected),
            bid_projection_grid_steps=int(bid_steps),
            ask_projection_grid_steps=int(ask_steps),
        )

    def map_normalized(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        ref2: int,
        current_book: MarketBookSnapshot,
    ) -> MappedIMMAction:
        decoded = self.decode_normalized(action)

        return self.map_decoded(
            decoded=decoded,
            normalized_action=action,
            ref2=int(ref2),
            current_book=current_book,
        )
