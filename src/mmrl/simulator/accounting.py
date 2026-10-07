from __future__ import annotations

from dataclasses import dataclass

from .types import Fill


@dataclass
class Portfolio:
    maker_fee_rate: float
    cash: float = 0.0
    inventory: float = 0.0

    def reset(self) -> None:
        self.cash = 0.0
        self.inventory = 0.0

    def apply_fill(self, fill: Fill) -> None:
        if fill.qty <= 0:
            raise ValueError("Fill qty must be positive.")

        if fill.side == "buy":
            self.inventory += fill.qty
            self.cash -= fill.notional + fill.fee

        elif fill.side == "sell":
            self.inventory -= fill.qty
            self.cash += fill.notional - fill.fee

        else:
            raise ValueError(f"Unknown fill side: {fill.side}")

    def equity(self, mid_price: float) -> float:
        return self.cash + self.inventory * float(mid_price)

    def fee_for(self, price: float, qty: float) -> float:
        return float(price) * float(qty) * self.maker_fee_rate
