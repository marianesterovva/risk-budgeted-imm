from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Side = Literal["buy", "sell"]


@dataclass
class Order:
    order_id: int
    side: Side
    price_tick: int

    initial_qty: float
    remaining_qty: float
    queue_ahead_qty: float

    placed_ts_us: int
    placement_seq: int

    active: bool = True

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {self.side}")

        if self.initial_qty <= 0:
            raise ValueError("initial_qty must be positive.")

        if self.remaining_qty < 0:
            raise ValueError("remaining_qty must be non-negative.")

        if self.remaining_qty > self.initial_qty + 1e-12:
            raise ValueError("remaining_qty cannot exceed initial_qty.")

        if self.queue_ahead_qty < -1e-12:
            raise ValueError("queue_ahead_qty must be non-negative.")

        self.queue_ahead_qty = max(0.0, float(self.queue_ahead_qty))

    @property
    def filled_qty(self) -> float:
        return self.initial_qty - self.remaining_qty


@dataclass(frozen=True)
class TargetOrder:
    side: Side
    price_tick: int
    qty: float

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {self.side}")

        if self.qty < 0:
            raise ValueError("Target quantity cannot be negative.")


@dataclass(frozen=True)
class TradeEvent:
    timestamp_us: int
    price_tick: int
    qty: float
    is_buyer_maker: bool

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("Trade quantity must be positive.")

    @property
    def aggressive_side(self) -> Side:
        # buyer is maker => seller is taker/aggressor
        return "sell" if self.is_buyer_maker else "buy"


@dataclass(frozen=True)
class Fill:
    order_id: int
    timestamp_us: int
    side: Side

    price_tick: int
    price: float

    qty: float
    notional: float
    fee: float

    reason: Literal["exact_queue", "trade_through"]


@dataclass(frozen=True)
class MarketBookSnapshot:
    timestamp_us: int

    bids: Mapping[int, float]
    asks: Mapping[int, float]

    mid_price: float
    tick_size: float

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive.")

        if not self.bids:
            raise ValueError("bids cannot be empty.")

        if not self.asks:
            raise ValueError("asks cannot be empty.")

        if self.best_bid_tick >= self.best_ask_tick:
            raise ValueError("Crossed/locked historical book.")

        for px, qty in list(self.bids.items()) + list(self.asks.items()):
            if qty < 0:
                raise ValueError(f"Negative book quantity at tick {px}.")

    @property
    def best_bid_tick(self) -> int:
        return max(self.bids)

    @property
    def best_ask_tick(self) -> int:
        return min(self.asks)

    def visible_qty(self, side: Side, price_tick: int) -> float | None:
        book = self.bids if side == "buy" else self.asks
        qty = book.get(int(price_tick))
        return None if qty is None else float(qty)

    def is_inside_spread(self, side: Side, price_tick: int) -> bool:
        if side == "buy":
            return self.best_bid_tick < price_tick < self.best_ask_tick

        return self.best_bid_tick < price_tick < self.best_ask_tick

    def price_from_tick(self, price_tick: int) -> float:
        return float(price_tick) * self.tick_size


@dataclass
class StepResult:
    fills: list[Fill] = field(default_factory=list)

    placed_qty: float = 0.0
    canceled_qty: float = 0.0

    filled_qty: float = 0.0
    buy_fill_qty: float = 0.0
    sell_fill_qty: float = 0.0

    exact_queue_fill_qty: float = 0.0
    trade_through_fill_qty: float = 0.0

    cash: float = 0.0
    inventory: float = 0.0

    equity_before: float = 0.0
    equity_after: float = 0.0
    pnl: float = 0.0
