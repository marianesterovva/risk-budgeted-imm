from __future__ import annotations

from collections import deque
import math

import numpy as np

from mmrl.simulator.types import MarketBookSnapshot, TargetOrder

from .base import ScriptedPolicy, clip_passive_tick


class AvellanedaStoikovPolicy(ScriptedPolicy):
    """
    Causal tick-space Avellaneda-Stoikov benchmark.

    We work in tick coordinates:

        r = mid_tick - q_norm * gamma * sigma2_rate * tau

        half_spread =
            0.5 * gamma * sigma2_rate * tau
            + (1/gamma) * log(1 + gamma/kappa)

    where:
      - q_norm = inventory / reference_side_quote_qty;
      - gamma and kappa have units 1/tick;
      - sigma2_rate is realized variance of mid ticks per second;
      - tau is a capped remaining horizon in seconds.

    This is a documented cross-asset implementation choice; it is not claimed
    to reproduce any exchange-specific calibration of the original paper.
    """

    name = "avellaneda_stoikov"

    def __init__(
        self,
        *,
        gamma_per_tick: float = 0.10,
        kappa_per_tick: float = 1.50,
        volatility_window_seconds: float = 60.0,
        horizon_seconds: float = 300.0,
        min_half_spread_ticks: float = 1.0,
        max_half_spread_ticks: float = 5.0,
        min_variance_rate_tick2_per_s: float = 1e-8,
    ) -> None:
        if gamma_per_tick <= 0:
            raise ValueError("gamma_per_tick must be positive.")

        if kappa_per_tick <= 0:
            raise ValueError("kappa_per_tick must be positive.")

        if volatility_window_seconds <= 0:
            raise ValueError("volatility_window_seconds must be positive.")

        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive.")

        if min_half_spread_ticks <= 0:
            raise ValueError("min_half_spread_ticks must be positive.")

        if max_half_spread_ticks < min_half_spread_ticks:
            raise ValueError(
                "max_half_spread_ticks must be >= min_half_spread_ticks."
            )

        self.gamma = float(gamma_per_tick)
        self.kappa = float(kappa_per_tick)
        self.vol_window_us = int(
            round(float(volatility_window_seconds) * 1_000_000)
        )
        self.horizon_seconds = float(horizon_seconds)
        self.min_half_spread_ticks = float(min_half_spread_ticks)
        self.max_half_spread_ticks = float(max_half_spread_ticks)
        self.min_variance_rate = float(
            min_variance_rate_tick2_per_s
        )

        self._history: deque[tuple[int, float]] = deque()

    def reset(self) -> None:
        self._history.clear()

    def observe(self, book: MarketBookSnapshot) -> None:
        mid_tick = float(book.mid_price) / float(book.tick_size)
        ts = int(book.timestamp_us)

        self._history.append((ts, mid_tick))

        cutoff = ts - self.vol_window_us

        while (
            len(self._history) >= 2
            and self._history[1][0] < cutoff
        ):
            self._history.popleft()

    def variance_rate_tick2_per_s(self) -> float:
        if len(self._history) < 2:
            return self.min_variance_rate

        ts = np.fromiter(
            (x[0] for x in self._history),
            dtype=np.int64,
        )

        mid = np.fromiter(
            (x[1] for x in self._history),
            dtype=np.float64,
        )

        elapsed_s = (ts[-1] - ts[0]) / 1_000_000.0

        if elapsed_s <= 0:
            return self.min_variance_rate

        diff = np.diff(mid)

        variance_rate = float(
            np.sum(diff * diff) / elapsed_s
        )

        return max(
            self.min_variance_rate,
            variance_rate,
        )

    def target_book(
        self,
        *,
        book: MarketBookSnapshot,
        inventory: float,
        quote_qty: float,
        time_to_episode_end_s: float | None = None,
    ) -> list[TargetOrder]:
        if quote_qty <= 0:
            raise ValueError("quote_qty must be positive.")

        q_norm = float(inventory) / float(quote_qty)
        sigma2_rate = self.variance_rate_tick2_per_s()

        if time_to_episode_end_s is None:
            tau = self.horizon_seconds
        else:
            tau = min(
                self.horizon_seconds,
                max(0.0, float(time_to_episode_end_s)),
            )

        mid_tick = float(book.mid_price) / float(book.tick_size)

        reservation_tick = (
            mid_tick
            - q_norm
            * self.gamma
            * sigma2_rate
            * tau
        )

        half_spread = (
            0.5
            * self.gamma
            * sigma2_rate
            * tau
            + (1.0 / self.gamma)
            * math.log(
                1.0 + self.gamma / self.kappa
            )
        )

        half_spread = min(
            self.max_half_spread_ticks,
            max(
                self.min_half_spread_ticks,
                half_spread,
            ),
        )

        desired_bid = int(
            math.floor(
                reservation_tick - half_spread
            )
        )

        desired_ask = int(
            math.ceil(
                reservation_tick + half_spread
            )
        )

        bid_tick = clip_passive_tick(
            side="buy",
            desired_tick=desired_bid,
            book=book,
        )

        ask_tick = clip_passive_tick(
            side="sell",
            desired_tick=desired_ask,
            book=book,
        )

        if bid_tick >= ask_tick:
            # Conservative fallback to historical L1.
            bid_tick = book.best_bid_tick
            ask_tick = book.best_ask_tick

        return [
            TargetOrder("buy", bid_tick, float(quote_qty)),
            TargetOrder("sell", ask_tick, float(quote_qty)),
        ]
