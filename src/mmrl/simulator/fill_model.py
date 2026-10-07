from __future__ import annotations

from collections import defaultdict

from .accounting import Portfolio
from .types import Fill, MarketBookSnapshot, Order, TradeEvent


_EPS = 1e-12


class FillModel:
    def __init__(
        self,
        *,
        trade_through_fill: bool = True,
        allow_partial_fills: bool = True,
    ) -> None:
        self.trade_through_fill = trade_through_fill
        self.allow_partial_fills = allow_partial_fills

    @staticmethod
    def _trade_affects_order(trade: TradeEvent, order: Order) -> bool:
        if trade.aggressive_side == "sell":
            return order.side == "buy"

        return order.side == "sell"

    @staticmethod
    def _is_trade_through(trade: TradeEvent, order: Order) -> bool:
        if trade.aggressive_side == "sell":
            return order.side == "buy" and trade.price_tick < order.price_tick

        return order.side == "sell" and trade.price_tick > order.price_tick

    @staticmethod
    def _is_exact_price(trade: TradeEvent, order: Order) -> bool:
        return trade.price_tick == order.price_tick

    def _make_fill(
        self,
        *,
        order: Order,
        qty: float,
        timestamp_us: int,
        reason: str,
        book: MarketBookSnapshot,
        portfolio: Portfolio,
    ) -> Fill:
        if qty <= _EPS:
            raise ValueError("Internal error: non-positive fill.")

        qty = min(float(qty), order.remaining_qty)
        price = book.price_from_tick(order.price_tick)
        notional = price * qty
        fee = portfolio.fee_for(price, qty)

        fill = Fill(
            order_id=order.order_id,
            timestamp_us=int(timestamp_us),
            side=order.side,
            price_tick=order.price_tick,
            price=price,
            qty=qty,
            notional=notional,
            fee=fee,
            reason=reason,
        )

        order.remaining_qty -= qty

        if order.remaining_qty <= _EPS:
            order.remaining_qty = 0.0
            order.active = False

        portfolio.apply_fill(fill)
        return fill

    def process_trade(
        self,
        *,
        trade: TradeEvent,
        orders: list[Order],
        book: MarketBookSnapshot,
        portfolio: Portfolio,
    ) -> list[Fill]:
        """
        Process one historical aggressive trade.

        Same-timestamp protection:
        a newly placed order is eligible only if
            trade.timestamp_us > order.placed_ts_us

        Exact-price queue logic is evaluated from the pre-trade queue positions
        using the same historical trade volume for all same-price own orders.
        """
        eligible = [
            o
            for o in orders
            if o.active
            and self._trade_affects_order(trade, o)
            and trade.timestamp_us > o.placed_ts_us
        ]

        if not eligible:
            return []

        fills: list[Fill] = []

        # 1) Trade-through fills.
        if self.trade_through_fill:
            for order in sorted(
                eligible,
                key=lambda o: o.placement_seq,
            ):
                if not order.active:
                    continue

                if self._is_trade_through(trade, order):
                    fills.append(
                        self._make_fill(
                            order=order,
                            qty=order.remaining_qty,
                            timestamp_us=trade.timestamp_us,
                            reason="trade_through",
                            book=book,
                            portfolio=portfolio,
                        )
                    )

        # 2) Exact-price queue fills.
        exact_orders = [
            o
            for o in eligible
            if o.active and self._is_exact_price(trade, o)
        ]

        if not exact_orders:
            return fills

        # Snapshot pre-trade positions: every child order sees the same
        # historical trade volume crossing its queue position.
        pre = [
            (
                o,
                float(o.queue_ahead_qty),
                float(o.remaining_qty),
            )
            for o in exact_orders
        ]

        trade_qty = float(trade.qty)

        for order, queue_before, remaining_before in sorted(
            pre,
            key=lambda x: x[0].placement_seq,
        ):
            if not order.active:
                continue

            excess = max(0.0, trade_qty - queue_before)

            if self.allow_partial_fills:
                fill_qty = min(remaining_before, excess)
            else:
                fill_qty = remaining_before if excess >= remaining_before else 0.0

            # Queue position after the historical trade.
            order.queue_ahead_qty = max(
                0.0,
                queue_before - trade_qty,
            )

            if fill_qty > _EPS:
                fills.append(
                    self._make_fill(
                        order=order,
                        qty=fill_qty,
                        timestamp_us=trade.timestamp_us,
                        reason="exact_queue",
                        book=book,
                        portfolio=portfolio,
                    )
                )

        return fills
