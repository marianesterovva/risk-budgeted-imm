from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class QueueLookup:
    qty: float
    known: bool
    status: str


@dataclass(frozen=True)
class ReferencePriceState:
    ref2: int
    candidate_ref2: int
    mid2: int
    updated: bool
    reason: str
    gate_level: Optional[int]
    gate_known: Optional[bool]
    gate_qty: Optional[float]

    def ref_price(self, tick_size: float) -> float:
        return 0.5 * float(self.ref2) * float(tick_size)

    def mid_price_from_ticks(self, tick_size: float) -> float:
        return 0.5 * float(self.mid2) * float(tick_size)

    @property
    def ref_minus_mid_ticks(self) -> float:
        return 0.5 * float(self.ref2 - self.mid2)


def _validate_book(
    bids: Mapping[int, float],
    asks: Mapping[int, float],
) -> tuple[int, int]:
    if not bids:
        raise ValueError("bids must be non-empty")
    if not asks:
        raise ValueError("asks must be non-empty")

    best_bid = max(int(p) for p in bids)
    best_ask = min(int(p) for p in asks)

    if best_bid >= best_ask:
        raise ValueError(
            f"crossed/locked book: best_bid={best_bid}, best_ask={best_ask}"
        )

    return best_bid, best_ask


def queue_at_tick(
    price_tick: int,
    bids: Mapping[int, float],
    asks: Mapping[int, float],
    *,
    eps: float = 1e-12,
) -> QueueLookup:
    """
    L2 depth-limited lookup.

    We distinguish:
      1) occupied visible level;
      2) known-empty tick inside the observable depth range;
      3) unknown tick deeper than the available depth.

    This is crucial for depth-5 data: absence from the top-5 list does NOT
    automatically mean zero liquidity if the requested price is deeper than L5.
    """
    price_tick = int(price_tick)
    best_bid, best_ask = _validate_book(bids, asks)

    if price_tick in bids:
        qty = float(bids[price_tick])
        if qty < -eps:
            raise ValueError("negative bid quantity")
        return QueueLookup(
            qty=max(0.0, qty),
            known=True,
            status="occupied_bid",
        )

    if price_tick in asks:
        qty = float(asks[price_tick])
        if qty < -eps:
            raise ValueError("negative ask quantity")
        return QueueLookup(
            qty=max(0.0, qty),
            known=True,
            status="occupied_ask",
        )

    # Inside spread is observable and empty in the historical market LOB.
    if best_bid < price_tick < best_ask:
        return QueueLookup(
            qty=0.0,
            known=True,
            status="known_empty_inside_spread",
        )

    deepest_bid = min(int(p) for p in bids)
    deepest_ask = max(int(p) for p in asks)

    # A missing tick between best bid and the deepest visible bid is known empty:
    # if liquidity existed there, it would be among the visible top levels.
    if deepest_bid <= price_tick <= best_bid:
        return QueueLookup(
            qty=0.0,
            known=True,
            status="known_empty_bid_range",
        )

    # Symmetric ask-side case.
    if best_ask <= price_tick <= deepest_ask:
        return QueueLookup(
            qty=0.0,
            known=True,
            status="known_empty_ask_range",
        )

    return QueueLookup(
        qty=0.0,
        known=False,
        status="unknown_beyond_visible_depth",
    )


def candidate_reference_ref2(
    best_bid_tick: int,
    best_ask_tick: int,
    *,
    previous_ref2: Optional[int],
    initial_even_tie_break: str = "lower",
) -> int:
    """
    IMM candidate reference price \tilde p_ref in half-tick coordinates.

    Let spread_ticks = best_ask_tick - best_bid_tick.

    - odd spread:  candidate is the midprice;
    - even spread: candidate is mid +/- 0.5 tick, choosing the value closest
      to the prior stable reference price.

    The paper does not specify the episode-initial tie-break when an even
    spread has no prior reference value. We use a deterministic documented
    choice: "lower" by default.
    """
    best_bid_tick = int(best_bid_tick)
    best_ask_tick = int(best_ask_tick)

    spread_ticks = best_ask_tick - best_bid_tick
    if spread_ticks <= 0:
        raise ValueError(
            f"spread must be positive, got {spread_ticks}"
        )

    # Twice the midprice measured in ticks.
    mid2 = best_bid_tick + best_ask_tick

    if spread_ticks % 2 == 1:
        # Midprice is already on a half-tick coordinate.
        candidate = mid2
    else:
        lower = mid2 - 1
        upper = mid2 + 1

        if previous_ref2 is None:
            if initial_even_tie_break == "lower":
                candidate = lower
            elif initial_even_tie_break == "upper":
                candidate = upper
            else:
                raise ValueError(
                    "initial_even_tie_break must be 'lower' or 'upper'"
                )
        else:
            previous_ref2 = int(previous_ref2)
            dl = abs(lower - previous_ref2)
            du = abs(upper - previous_ref2)

            if dl < du:
                candidate = lower
            elif du < dl:
                candidate = upper
            else:
                # This tie should not normally occur because previous_ref2
                # is odd while the midpoint between lower and upper is even.
                candidate = lower if initial_even_tie_break == "lower" else upper

    if candidate % 2 == 0:
        raise AssertionError(
            f"reference candidate must be half-tick (odd ref2), got {candidate}"
        )

    return int(candidate)


def grid_level_tick(
    ref2: int,
    level: int,
) -> int:
    """
    Convert IMM grid level +/-i to an integer exchange tick.

    Q_{+i} = p_ref + (i - 1/2) tick
    Q_{-i} = p_ref - (i - 1/2) tick
    """
    ref2 = int(ref2)
    level = int(level)

    if ref2 % 2 == 0:
        raise ValueError(
            f"ref2 must be odd, got {ref2}"
        )
    if level == 0:
        raise ValueError("IMM grid has no Q_0")

    i = abs(level)
    offset2 = 2 * i - 1

    if level > 0:
        q2 = ref2 + offset2
    else:
        q2 = ref2 - offset2

    if q2 % 2 != 0:
        raise AssertionError(
            f"grid level must land on integer tick: ref2={ref2}, level={level}"
        )

    return int(q2 // 2)


def build_grid_ticks(
    ref2: int,
    *,
    k: int = 5,
) -> Dict[int, int]:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")

    levels = list(range(-k, 0)) + list(range(1, k + 1))
    return {
        level: grid_level_tick(ref2, level)
        for level in levels
    }


class StableReferencePriceTracker:
    """
    Stateful IMM-style stable reference price.

    Paper rule:
      - initialize p_ref,0 = \tilde p_ref,0;
      - if midprice increases, update p_ref to candidate only when l_{-1}=0;
      - if midprice decreases, update p_ref to candidate only when l_{+1}=0.

    Crypto/depth-5 implementation choice:
      - reference tracking uses the historical market LOB only;
      - if the gating Q_{-1}/Q_{+1} price is beyond visible depth, its
        liquidity is UNKNOWN and we conservatively do not update p_ref.
    """

    def __init__(
        self,
        *,
        k: int = 5,
        zero_eps: float = 1e-12,
        initial_even_tie_break: str = "lower",
    ) -> None:
        self.k = int(k)
        self.zero_eps = float(zero_eps)
        self.initial_even_tie_break = str(initial_even_tie_break)

        if self.k <= 0:
            raise ValueError("k must be positive")

        self.ref2: Optional[int] = None
        self.prev_mid2: Optional[int] = None

    def clear(self) -> None:
        self.ref2 = None
        self.prev_mid2 = None

    def reset(
        self,
        bids: Mapping[int, float],
        asks: Mapping[int, float],
    ) -> ReferencePriceState:
        best_bid, best_ask = _validate_book(bids, asks)
        mid2 = best_bid + best_ask

        candidate = candidate_reference_ref2(
            best_bid,
            best_ask,
            previous_ref2=None,
            initial_even_tie_break=self.initial_even_tie_break,
        )

        self.ref2 = candidate
        self.prev_mid2 = mid2

        return ReferencePriceState(
            ref2=int(self.ref2),
            candidate_ref2=int(candidate),
            mid2=int(mid2),
            updated=True,
            reason="episode_reset",
            gate_level=None,
            gate_known=None,
            gate_qty=None,
        )

    def update(
        self,
        bids: Mapping[int, float],
        asks: Mapping[int, float],
    ) -> ReferencePriceState:
        if self.ref2 is None or self.prev_mid2 is None:
            return self.reset(bids, asks)

        best_bid, best_ask = _validate_book(bids, asks)
        mid2 = best_bid + best_ask

        candidate = candidate_reference_ref2(
            best_bid,
            best_ask,
            previous_ref2=self.ref2,
            initial_even_tie_break=self.initial_even_tie_break,
        )

        updated = False
        reason = "mid_unchanged"
        gate_level: Optional[int] = None
        gate_known: Optional[bool] = None
        gate_qty: Optional[float] = None

        if mid2 > self.prev_mid2:
            gate_level = -1
            gate_tick = grid_level_tick(self.ref2, gate_level)
            gate = queue_at_tick(
                gate_tick,
                bids,
                asks,
                eps=self.zero_eps,
            )
            gate_known = bool(gate.known)
            gate_qty = float(gate.qty)

            if not gate.known:
                reason = "mid_up_gate_unknown"
            elif gate.qty <= self.zero_eps:
                if candidate != self.ref2:
                    self.ref2 = candidate
                    updated = True
                    reason = "mid_up_qminus1_empty_update"
                else:
                    reason = "mid_up_qminus1_empty_candidate_same"
            else:
                reason = "mid_up_qminus1_nonempty_hold"

        elif mid2 < self.prev_mid2:
            gate_level = +1
            gate_tick = grid_level_tick(self.ref2, gate_level)
            gate = queue_at_tick(
                gate_tick,
                bids,
                asks,
                eps=self.zero_eps,
            )
            gate_known = bool(gate.known)
            gate_qty = float(gate.qty)

            if not gate.known:
                reason = "mid_down_gate_unknown"
            elif gate.qty <= self.zero_eps:
                if candidate != self.ref2:
                    self.ref2 = candidate
                    updated = True
                    reason = "mid_down_qplus1_empty_update"
                else:
                    reason = "mid_down_qplus1_empty_candidate_same"
            else:
                reason = "mid_down_qplus1_nonempty_hold"

        self.prev_mid2 = mid2

        if self.ref2 is None:
            raise AssertionError("ref2 unexpectedly None")

        return ReferencePriceState(
            ref2=int(self.ref2),
            candidate_ref2=int(candidate),
            mid2=int(mid2),
            updated=bool(updated),
            reason=str(reason),
            gate_level=gate_level,
            gate_known=gate_known,
            gate_qty=gate_qty,
        )

    def grid_ticks(self) -> Dict[int, int]:
        if self.ref2 is None:
            raise RuntimeError("tracker is not initialized")
        return build_grid_ticks(
            self.ref2,
            k=self.k,
        )
