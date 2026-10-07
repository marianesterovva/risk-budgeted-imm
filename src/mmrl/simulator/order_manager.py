from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import MarketBookSnapshot, Order, TargetOrder


_EPS = 1e-12


class OrderManager:
    """
    Maintains the agent's resting orders.

    The policy specifies a target resting book.
    The manager makes minimal changes:
    - keep existing volume when possible;
    - add only missing volume;
    - reduce newest child orders first.
    """

    def __init__(
        self,
        *,
        allow_marketable_orders: bool = False,
        allow_inside_spread: bool = True,
        allow_beyond_depth5: bool = False,
    ) -> None:
        self.allow_marketable_orders = allow_marketable_orders
        self.allow_inside_spread = allow_inside_spread
        self.allow_beyond_depth5 = allow_beyond_depth5

        self._orders: dict[int, Order] = {}
        self._next_order_id = 1
        self._next_placement_seq = 1

    @property
    def active_orders(self) -> list[Order]:
        return sorted(
            [o for o in self._orders.values() if o.active],
            key=lambda o: o.placement_seq,
        )

    def reset(self) -> None:
        self._orders.clear()
        self._next_order_id = 1
        self._next_placement_seq = 1

    def _validate_target_price(
        self,
        target: TargetOrder,
        book: MarketBookSnapshot,
    ) -> None:
        px = int(target.price_tick)

        if not self.allow_marketable_orders:
            if target.side == "buy" and px >= book.best_ask_tick:
                raise ValueError("Marketable/crossing buy order is forbidden.")

            if target.side == "sell" and px <= book.best_bid_tick:
                raise ValueError("Marketable/crossing sell order is forbidden.")

        visible = book.visible_qty(target.side, px) is not None
        inside = book.is_inside_spread(target.side, px)

        if inside and not self.allow_inside_spread:
            raise ValueError("Inside-spread quoting is disabled.")

        if not visible and not inside and not self.allow_beyond_depth5:
            raise ValueError(
                "Target price is outside visible depth-5 and not inside the spread."
            )

    def _earlier_own_qty(
        self,
        *,
        side: str,
        price_tick: int,
        placement_seq: int | None = None,
    ) -> float:
        total = 0.0

        for order in self.active_orders:
            if order.side != side or order.price_tick != price_tick:
                continue

            if placement_seq is not None and order.placement_seq >= placement_seq:
                continue

            total += order.remaining_qty

        return total

    def place(
        self,
        *,
        side: str,
        price_tick: int,
        qty: float,
        book: MarketBookSnapshot,
        timestamp_us: int,
    ) -> Order:
        if qty <= _EPS:
            raise ValueError("Placed quantity must be positive.")

        target = TargetOrder(
            side=side,
            price_tick=int(price_tick),
            qty=float(qty),
        )
        self._validate_target_price(target, book)

        visible = book.visible_qty(side, int(price_tick))

        if visible is not None:
            external_ahead = float(visible)
        elif book.is_inside_spread(side, int(price_tick)):
            external_ahead = 0.0
        else:
            # Reached only when allow_beyond_depth5=True.
            external_ahead = 0.0

        own_ahead = self._earlier_own_qty(
            side=side,
            price_tick=int(price_tick),
        )

        order = Order(
            order_id=self._next_order_id,
            side=side,
            price_tick=int(price_tick),
            initial_qty=float(qty),
            remaining_qty=float(qty),
            queue_ahead_qty=float(external_ahead + own_ahead),
            placed_ts_us=int(timestamp_us),
            placement_seq=self._next_placement_seq,
            active=True,
        )

        self._orders[order.order_id] = order
        self._next_order_id += 1
        self._next_placement_seq += 1

        return order

    def cancel_order(self, order_id: int) -> float:
        order = self._orders[order_id]

        if not order.active:
            return 0.0

        canceled_qty = order.remaining_qty
        order.active = False
        order.remaining_qty = 0.0

        # Later own orders at the same price move forward by the canceled
        # own volume that used to stand ahead of them.
        for later in self.active_orders:
            if (
                later.side == order.side
                and later.price_tick == order.price_tick
                and later.placement_seq > order.placement_seq
            ):
                later.queue_ahead_qty = max(
                    0.0,
                    later.queue_ahead_qty - canceled_qty,
                )

        return canceled_qty

    def _group_active(self):
        grouped = defaultdict(list)

        for order in self.active_orders:
            grouped[(order.side, order.price_tick)].append(order)

        for key in grouped:
            grouped[key].sort(key=lambda o: o.placement_seq)

        return grouped

    def rebalance(
        self,
        *,
        target_book: Iterable[TargetOrder],
        current_book: MarketBookSnapshot,
        timestamp_us: int,
    ) -> tuple[float, float]:
        """
        Returns (placed_qty, canceled_qty).

        Whole child orders are canceled newest-first. If canceling a whole
        child order overshoots the target, the missing quantity is re-added
        as a fresh order at the back of the queue. This avoids assuming an
        exchange-side in-place quantity reduction that preserves priority.
        """
        aggregated: dict[tuple[str, int], float] = defaultdict(float)

        for target in target_book:
            if target.qty <= _EPS:
                continue

            self._validate_target_price(target, current_book)
            aggregated[(target.side, int(target.price_tick))] += float(target.qty)

        placed_qty = 0.0
        canceled_qty = 0.0

        grouped = self._group_active()
        all_keys = set(grouped) | set(aggregated)

        for key in sorted(all_keys):
            side, price_tick = key
            orders = grouped.get(key, [])
            current_qty = sum(o.remaining_qty for o in orders)
            target_qty = aggregated.get(key, 0.0)

            if current_qty > target_qty + _EPS:
                # Cancel newest child orders first.
                excess = current_qty - target_qty
                kept_after_cancel = current_qty

                for order in sorted(
                    orders,
                    key=lambda o: o.placement_seq,
                    reverse=True,
                ):
                    if excess <= _EPS:
                        break

                    canceled = self.cancel_order(order.order_id)
                    canceled_qty += canceled
                    kept_after_cancel -= canceled
                    excess = kept_after_cancel - target_qty

                # If whole-order cancellation overshot the target,
                # add only the missing amount as a fresh child order.
                if kept_after_cancel < target_qty - _EPS:
                    add_qty = target_qty - kept_after_cancel
                    self.place(
                        side=side,
                        price_tick=price_tick,
                        qty=add_qty,
                        book=current_book,
                        timestamp_us=timestamp_us,
                    )
                    placed_qty += add_qty

            elif current_qty < target_qty - _EPS:
                add_qty = target_qty - current_qty

                self.place(
                    side=side,
                    price_tick=price_tick,
                    qty=add_qty,
                    book=current_book,
                    timestamp_us=timestamp_us,
                )
                placed_qty += add_qty

        return placed_qty, canceled_qty
