from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .reference_price import (
    build_grid_ticks,
    candidate_reference_ref2,
    grid_level_tick,
    queue_at_tick,
)


@dataclass(frozen=True)
class DepthAwareReferencePriceState:
    ref2: int
    candidate_ref2: int
    mid2: int
    reference_moved: bool
    paper_gate_empty: bool
    depth_truncation_reanchor: bool
    reason: str
    gate_level: Optional[int]
    gate_known: Optional[bool]
    gate_qty: Optional[float]

    def ref_price(self, tick_size: float) -> float:
        return 0.5 * float(self.ref2) * float(tick_size)

    @property
    def ref_minus_mid_ticks(self) -> float:
        return 0.5 * float(self.ref2 - self.mid2)


class DepthAwareStableReferencePriceTracker:
    """IMM stable reference + explicit top-K depth-truncation fallback."""

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
            raise ValueError(f"crossed/locked book: {best_bid} >= {best_ask}")
        return best_bid, best_ask

    def reset(self, bids, asks) -> DepthAwareReferencePriceState:
        best_bid, best_ask = self._best_ticks(bids, asks)
        mid2 = best_bid + best_ask
        candidate = candidate_reference_ref2(
            best_bid,
            best_ask,
            previous_ref2=None,
            initial_even_tie_break=self.initial_even_tie_break,
        )
        self.ref2 = int(candidate)
        self.prev_mid2 = int(mid2)
        return DepthAwareReferencePriceState(
            ref2=self.ref2,
            candidate_ref2=int(candidate),
            mid2=int(mid2),
            reference_moved=True,
            paper_gate_empty=False,
            depth_truncation_reanchor=False,
            reason="episode_reset",
            gate_level=None,
            gate_known=None,
            gate_qty=None,
        )

    def update(self, bids, asks) -> DepthAwareReferencePriceState:
        if self.ref2 is None or self.prev_mid2 is None:
            return self.reset(bids, asks)

        best_bid, best_ask = self._best_ticks(bids, asks)
        mid2 = best_bid + best_ask
        candidate = candidate_reference_ref2(
            best_bid,
            best_ask,
            previous_ref2=self.ref2,
            initial_even_tie_break=self.initial_even_tie_break,
        )
        old_ref2 = int(self.ref2)
        paper_gate_empty = False
        depth_reanchor = False
        gate_level = None
        gate_known = None
        gate_qty = None
        reason = "mid_unchanged"

        if mid2 > self.prev_mid2:
            gate_level = -1
            gate_tick = grid_level_tick(self.ref2, gate_level)
            gate = queue_at_tick(gate_tick, bids, asks, eps=self.zero_eps)
            gate_known = bool(gate.known)
            gate_qty = float(gate.qty)
            if not gate.known:
                self.ref2 = int(candidate)
                depth_reanchor = True
                reason = "mid_up_depth_truncation_reanchor"
            elif gate.qty <= self.zero_eps:
                self.ref2 = int(candidate)
                paper_gate_empty = True
                reason = "mid_up_qminus1_empty"
            else:
                reason = "mid_up_qminus1_nonempty_hold"

        elif mid2 < self.prev_mid2:
            gate_level = +1
            gate_tick = grid_level_tick(self.ref2, gate_level)
            gate = queue_at_tick(gate_tick, bids, asks, eps=self.zero_eps)
            gate_known = bool(gate.known)
            gate_qty = float(gate.qty)
            if not gate.known:
                self.ref2 = int(candidate)
                depth_reanchor = True
                reason = "mid_down_depth_truncation_reanchor"
            elif gate.qty <= self.zero_eps:
                self.ref2 = int(candidate)
                paper_gate_empty = True
                reason = "mid_down_qplus1_empty"
            else:
                reason = "mid_down_qplus1_nonempty_hold"

        self.prev_mid2 = int(mid2)
        return DepthAwareReferencePriceState(
            ref2=int(self.ref2),
            candidate_ref2=int(candidate),
            mid2=int(mid2),
            reference_moved=bool(int(self.ref2) != old_ref2),
            paper_gate_empty=paper_gate_empty,
            depth_truncation_reanchor=depth_reanchor,
            reason=reason,
            gate_level=gate_level,
            gate_known=gate_known,
            gate_qty=gate_qty,
        )

    def grid_ticks(self):
        if self.ref2 is None:
            raise RuntimeError("tracker is not initialized")
        return build_grid_ticks(self.ref2, k=self.k)
