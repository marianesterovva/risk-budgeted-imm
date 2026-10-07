from __future__ import annotations

from abc import ABC, abstractmethod

from mmrl.simulator.types import MarketBookSnapshot, TargetOrder


class ScriptedPolicy(ABC):
    """Common interface used by scripted baseline runners."""

    name: str = "scripted"

    def reset(self) -> None:
        """Reset state at the beginning of every evaluation episode."""

    def observe(self, book: MarketBookSnapshot) -> None:
        """
        Observe the current historical book.

        Called on every real snapshot, not only at decision points.
        Stateful policies may use this for causal feature updates.
        """

    @abstractmethod
    def target_book(
        self,
        *,
        book: MarketBookSnapshot,
        inventory: float,
        quote_qty: float,
        time_to_episode_end_s: float | None = None,
    ) -> list[TargetOrder]:
        """Return the desired resting book at a policy decision point."""
        raise NotImplementedError


def clip_passive_tick(
    *,
    side: str,
    desired_tick: int,
    book: MarketBookSnapshot,
) -> int:
    """
    Map a desired passive quote to a price accepted by FillSimulator v0.1.

    Key detail:
    visible depth-5 prices are NOT guaranteed to occupy every integer tick.
    Therefore a price lying numerically between L1 and L5 may still be absent
    from the historical snapshot.

    Policy:
    - if desired price is strictly inside the spread, keep it;
    - otherwise snap an outward quote to an ACTUALLY VISIBLE depth-5 level;
    - if desired price is deeper than L5, clamp to the deepest visible level;
    - never return a marketable price.

    Buy side:
        choose the most aggressive visible bid that is <= desired_tick.

    Sell side:
        choose the most aggressive visible ask that is >= desired_tick.
    """
    desired_tick = int(desired_tick)

    if side == "buy":
        best_bid = int(book.best_bid_tick)
        best_ask = int(book.best_ask_tick)

        # Passive price improvement inside the spread.
        if best_bid < desired_tick < best_ask:
            return desired_tick

        # Crossing/marketable desire: clamp to best passive price.
        if desired_tick >= best_ask:
            if best_ask - best_bid > 1:
                return best_ask - 1
            return best_bid

        visible_bids = sorted(
            (int(px) for px in book.bids.keys()),
            reverse=True,
        )

        eligible = [
            px
            for px in visible_bids
            if px <= desired_tick
        ]

        if eligible:
            # Highest visible bid not more aggressive than desired.
            return max(eligible)

        # Desired quote is deeper than visible L5.
        return min(visible_bids)

    if side == "sell":
        best_bid = int(book.best_bid_tick)
        best_ask = int(book.best_ask_tick)

        # Passive price improvement inside the spread.
        if best_bid < desired_tick < best_ask:
            return desired_tick

        # Crossing/marketable desire: clamp to best passive price.
        if desired_tick <= best_bid:
            if best_ask - best_bid > 1:
                return best_bid + 1
            return best_ask

        visible_asks = sorted(
            int(px)
            for px in book.asks.keys()
        )

        eligible = [
            px
            for px in visible_asks
            if px >= desired_tick
        ]

        if eligible:
            # Lowest visible ask not more aggressive than desired.
            return min(eligible)

        # Desired quote is deeper than visible L5.
        return max(visible_asks)

    raise ValueError(f"Unknown side: {side}")
