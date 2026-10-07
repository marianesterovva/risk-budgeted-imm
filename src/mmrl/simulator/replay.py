from __future__ import annotations

from .accounting import Portfolio
from .fill_model import FillModel
from .order_manager import OrderManager
from .queue_model import ProportionalQueueModel
from .types import (
    MarketBookSnapshot,
    StepResult,
    TargetOrder,
    TradeEvent,
)


class GapBoundaryError(RuntimeError):
    """Raised when a caller tries to replay across a missing-data episode gap."""


class ReplaySimulator:
    def __init__(
        self,
        *,
        max_gap_ms: int,
        maker_fee_rate: float,
        allow_marketable_orders: bool = False,
        allow_inside_spread: bool = True,
        allow_beyond_depth5: bool = False,
        trade_through_fill: bool = True,
        allow_partial_fills: bool = True,
    ) -> None:
        self.max_gap_ms = int(max_gap_ms)

        if self.max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be positive.")

        self.order_manager = OrderManager(
            allow_marketable_orders=allow_marketable_orders,
            allow_inside_spread=allow_inside_spread,
            allow_beyond_depth5=allow_beyond_depth5,
        )

        self.fill_model = FillModel(
            trade_through_fill=trade_through_fill,
            allow_partial_fills=allow_partial_fills,
        )

        self.queue_model = ProportionalQueueModel()

        self.portfolio = Portfolio(
            maker_fee_rate=float(maker_fee_rate),
        )

    def reset(self) -> None:
        self.order_manager.reset()
        self.portfolio.reset()

    def step(
        self,
        *,
        current_book: MarketBookSnapshot,
        next_book: MarketBookSnapshot,
        trades: list[TradeEvent],
        target_book: list[TargetOrder] | None,
    ) -> StepResult:
        """
        Replay one real market transition.

        target_book:
          - list[TargetOrder]: a policy decision occurs now; rebalance the
            agent's resting book to this target.
          - None: no policy decision occurs; keep currently resting orders
            unchanged.

        This distinction is required when market snapshots arrive faster than
        the policy decision interval.
        """
        dt_us = next_book.timestamp_us - current_book.timestamp_us

        if dt_us <= 0:
            raise ValueError("Book timestamps must be strictly increasing.")

        if dt_us > self.max_gap_ms * 1000:
            raise GapBoundaryError(
                f"Transition gap is {dt_us / 1000:.3f} ms, "
                f"above max_gap_ms={self.max_gap_ms}. "
                "Start a new episode instead."
            )

        equity_before = self.portfolio.equity(
            current_book.mid_price
        )

        if target_book is None:
            placed_qty = 0.0
            canceled_qty = 0.0
        else:
            placed_qty, canceled_qty = self.order_manager.rebalance(
                target_book=target_book,
                current_book=current_book,
                timestamp_us=current_book.timestamp_us,
            )

        fills = []

        for trade in sorted(
            trades,
            key=lambda x: x.timestamp_us,
        ):
            if not (
                current_book.timestamp_us
                <= trade.timestamp_us
                < next_book.timestamp_us
            ):
                continue

            fills.extend(
                self.fill_model.process_trade(
                    trade=trade,
                    orders=self.order_manager.active_orders,
                    book=current_book,
                    portfolio=self.portfolio,
                )
            )

        self.queue_model.reconcile(
            current_book=current_book,
            next_book=next_book,
            trades=trades,
            orders=self.order_manager.active_orders,
        )

        equity_after = self.portfolio.equity(
            next_book.mid_price
        )

        result = StepResult(
            fills=fills,
            placed_qty=placed_qty,
            canceled_qty=canceled_qty,
            filled_qty=sum(f.qty for f in fills),
            buy_fill_qty=sum(
                f.qty for f in fills if f.side == "buy"
            ),
            sell_fill_qty=sum(
                f.qty for f in fills if f.side == "sell"
            ),
            exact_queue_fill_qty=sum(
                f.qty for f in fills if f.reason == "exact_queue"
            ),
            trade_through_fill_qty=sum(
                f.qty for f in fills if f.reason == "trade_through"
            ),
            cash=self.portfolio.cash,
            inventory=self.portfolio.inventory,
            equity_before=equity_before,
            equity_after=equity_after,
            pnl=equity_after - equity_before,
        )

        self._assert_invariants()
        return result

    def _assert_invariants(self) -> None:
        for order in self.order_manager.active_orders:
            if order.remaining_qty <= 0:
                raise AssertionError(
                    "Active order has non-positive remaining_qty."
                )

            if order.queue_ahead_qty < -1e-12:
                raise AssertionError(
                    "Negative queue_ahead_qty."
                )
