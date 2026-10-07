
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class NotionalBpsReward:
    beta: float = 0.01
    eta: float = 0.05
    inventory_threshold_N: float = 3.0
    bps_scale: float = 10000.0

    def compute(
        self,
        *,
        pnl: float,
        filled_notional: float,
        inventory: float,
        quote_size: float,
        mid_price: float,
    ):
        quote_size = float(quote_size)
        mid_price = float(mid_price)

        if quote_size <= 0 or mid_price <= 0:
            raise ValueError("quote_size and mid_price must be positive")

        side_notional = quote_size * mid_price

        pnl_bps = self.bps_scale * float(pnl) / side_notional
        execution_units = float(filled_notional) / side_notional
        execution_term = float(self.beta) * execution_units

        inventory_N = float(inventory) / quote_size
        inventory_penalty = 0.0

        if abs(inventory_N) > float(self.inventory_threshold_N):
            inventory_penalty = -float(self.eta) * abs(inventory_N)

        reward = pnl_bps + execution_term + inventory_penalty

        return float(reward), {
            "pnl_bps": float(pnl_bps),
            "execution_units": float(execution_units),
            "execution_term": float(execution_term),
            "inventory_N": float(inventory_N),
            "inventory_penalty": float(inventory_penalty),
        }
