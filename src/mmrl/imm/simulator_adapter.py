from __future__ import annotations

from typing import Iterable, Optional

from mmrl.simulator import MarketBookSnapshot, TargetOrder

from .grid import classify_side_price


def validate_grid_target_book(
    current_book: MarketBookSnapshot,
    target_book: Iterable[TargetOrder],
) -> list:
    statuses = []
    for order in target_book:
        status = classify_side_price(
            side=order.side,
            price_tick=order.price_tick,
            bids=current_book.bids,
            asks=current_book.asks,
        )
        if not status.allowed:
            raise ValueError(
                "IMM grid target is not observable/passive: "
                f"side={order.side}, price_tick={order.price_tick}, "
                f"status={status.status}"
            )
        statuses.append(status)
    return statuses


def augment_current_book_for_known_empty(
    current_book: MarketBookSnapshot,
    target_book: Iterable[TargetOrder],
) -> MarketBookSnapshot:
    target_book = list(target_book)
    statuses = validate_grid_target_book(current_book, target_book)

    bids = dict(current_book.bids)
    asks = dict(current_book.asks)

    for order, status in zip(target_book, statuses):
        if status.status == "known_empty_bid_range":
            bids.setdefault(int(order.price_tick), 0.0)
        elif status.status == "known_empty_ask_range":
            asks.setdefault(int(order.price_tick), 0.0)

    return MarketBookSnapshot(
        timestamp_us=int(current_book.timestamp_us),
        bids=bids,
        asks=asks,
        mid_price=float(current_book.mid_price),
        tick_size=float(current_book.tick_size),
    )


class IMMGridReplayAdapter:
    """Thin compatibility wrapper over the validated ReplaySimulator."""

    def __init__(self, simulator) -> None:
        self.simulator = simulator

    def reset(self):
        return self.simulator.reset()

    def step(
        self,
        *,
        current_book,
        next_book,
        trades,
        target_book: Optional[Iterable[TargetOrder]],
    ):
        if target_book is None:
            return self.simulator.step(
                current_book=current_book,
                next_book=next_book,
                trades=trades,
                target_book=None,
            )

        target_book = list(target_book)
        current_augmented = augment_current_book_for_known_empty(
            current_book,
            target_book,
        )
        return self.simulator.step(
            current_book=current_augmented,
            next_book=next_book,
            trades=trades,
            target_book=target_book,
        )

    def __getattr__(self, name):
        return getattr(self.simulator, name)
