from __future__ import annotations

from collections import defaultdict

from .types import MarketBookSnapshot, Order, TradeEvent


_EPS = 1e-12


class ProportionalQueueModel:
    """
    End-of-transition queue reconciliation.

    Known trades are assumed to have been processed first.

    For a visible price level:
        net_cancel = max(0, V_i - T_i - V_{i+1})

    A proportional fraction of the remaining *external* queue ahead is removed.

    Historical additions never increase queue-ahead for an already-resting
    agent order: they are treated as arriving behind the agent.
    """

    @staticmethod
    def _exact_trade_volume_by_side_price(
        trades: list[TradeEvent],
    ) -> dict[tuple[str, int], float]:
        out: dict[tuple[str, int], float] = defaultdict(float)

        for trade in trades:
            resting_side = (
                "buy" if trade.aggressive_side == "sell" else "sell"
            )
            out[(resting_side, trade.price_tick)] += float(trade.qty)

        return out

    @staticmethod
    def _earlier_own_qty(
        order: Order,
        active_orders: list[Order],
    ) -> float:
        total = 0.0

        for earlier in active_orders:
            if not earlier.active:
                continue

            if earlier.order_id == order.order_id:
                continue

            if (
                earlier.side == order.side
                and earlier.price_tick == order.price_tick
                and earlier.placement_seq < order.placement_seq
            ):
                total += earlier.remaining_qty

        return total

    def reconcile(
        self,
        *,
        current_book: MarketBookSnapshot,
        next_book: MarketBookSnapshot,
        trades: list[TradeEvent],
        orders: list[Order],
    ) -> None:
        trade_volume = self._exact_trade_volume_by_side_price(trades)

        active_orders = [
            o for o in orders if o.active and o.remaining_qty > _EPS
        ]

        for order in active_orders:
            current_qty = current_book.visible_qty(
                order.side,
                order.price_tick,
            )

            next_qty = next_book.visible_qty(
                order.side,
                order.price_tick,
            )

            # If the level is not visible in either snapshot, depth=5 does not
            # allow us to infer whether it vanished or merely moved beyond L5.
            if current_qty is None or next_qty is None:
                continue

            exact_trades = trade_volume.get(
                (order.side, order.price_tick),
                0.0,
            )

            net_cancel = max(
                0.0,
                float(current_qty) - float(exact_trades) - float(next_qty),
            )

            if net_cancel <= _EPS or current_qty <= _EPS:
                continue

            rho = min(
                1.0,
                net_cancel / float(current_qty),
            )

            own_ahead = self._earlier_own_qty(
                order,
                active_orders,
            )

            external_ahead = max(
                0.0,
                order.queue_ahead_qty - own_ahead,
            )

            external_after = external_ahead * (1.0 - rho)

            order.queue_ahead_qty = max(
                0.0,
                own_ahead + external_after,
            )
