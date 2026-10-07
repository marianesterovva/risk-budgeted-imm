from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulatorConfig:
    max_gap_ms: int = 5000
    placement_latency_ms: int = 0
    cancel_latency_ms: int = 0

    queue_model: str = "proportional"
    cancellations_after_trades: bool = True

    trade_through_fill: bool = True
    allow_partial_fills: bool = True

    allow_marketable_orders: bool = False
    allow_inside_spread: bool = True
    allow_beyond_depth5: bool = False

    maker_fee_rate: float | None = None

    def validate(self, *, require_fee: bool = False) -> None:
        if self.max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be positive.")

        if self.placement_latency_ms < 0:
            raise ValueError("placement_latency_ms must be non-negative.")

        if self.cancel_latency_ms < 0:
            raise ValueError("cancel_latency_ms must be non-negative.")

        if self.queue_model != "proportional":
            raise ValueError(
                "FillSimulator v0.1 currently supports only queue_model='proportional'."
            )

        if not self.cancellations_after_trades:
            raise ValueError(
                "FillSimulator v0.1 assumes inferred cancellations are applied after trades."
            )

        if require_fee and self.maker_fee_rate is None:
            raise ValueError(
                "maker_fee_rate is unset. Fix it explicitly before PnL evaluation."
            )


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def load_simulator_config(path: str | Path) -> SimulatorConfig:
    data = load_yaml(path)
    cfg = SimulatorConfig(**data)
    cfg.validate(require_fee=False)
    return cfg
