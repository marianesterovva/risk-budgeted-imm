from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .grid import build_grid_slots


@dataclass(frozen=True)
class OwnOrderView:
    level: int
    remaining_qty: float
    queue_ahead: float


class IMMStateBuilder:
    """Builds market history and the 21D IMM-style private state."""

    def __init__(
        self,
        *,
        quote_size: float,
        k: int = 5,
        lookback: int = 64,
        eps: float = 1e-12,
    ) -> None:
        self.quote_size = float(quote_size)
        self.k = int(k)
        self.lookback = int(lookback)
        self.eps = float(eps)

        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.k != 5:
            raise ValueError("reference IMM-Crypto protocol currently fixes k=5")
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

        self.levels = tuple(
            list(range(-self.k, 0)) + list(range(1, self.k + 1))
        )
        self.market_feature_names = tuple(
            [f"queue_log_q{level:+d}" for level in self.levels]
            + [f"known_mask_q{level:+d}" for level in self.levels]
        )

    @property
    def market_feature_dim(self) -> int:
        return 4 * self.k

    @property
    def private_dim(self) -> int:
        return 1 + 4 * self.k

    def snapshot_market_features(
        self,
        *,
        bids: Mapping[int, float],
        asks: Mapping[int, float],
        ref2: int,
    ) -> np.ndarray:
        slots = build_grid_slots(bids, asks, int(ref2), k=self.k)
        qty = np.asarray(
            [
                np.log1p(max(0.0, s.qty) / self.quote_size)
                for s in slots
            ],
            dtype=np.float32,
        )
        known = np.asarray(
            [1.0 if s.known else 0.0 for s in slots],
            dtype=np.float32,
        )
        out = np.concatenate([qty, known]).astype(np.float32, copy=False)
        if out.shape != (self.market_feature_dim,):
            raise AssertionError(out.shape)
        return out

    def history_tensor(
        self,
        feature_matrix: np.ndarray,
        *,
        end_idx: int,
        episode_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        episode_ids = np.asarray(episode_ids)

        if feature_matrix.ndim != 2:
            raise ValueError("feature_matrix must be 2D")
        if feature_matrix.shape[1] != self.market_feature_dim:
            raise ValueError(
                f"expected F={self.market_feature_dim}, got {feature_matrix.shape[1]}"
            )
        if len(feature_matrix) != len(episode_ids):
            raise ValueError("feature/episode length mismatch")

        end_idx = int(end_idx)
        start_idx = end_idx - self.lookback + 1
        if start_idx < 0:
            raise ValueError("lookback not ready")
        if end_idx >= len(feature_matrix):
            raise IndexError(end_idx)

        eid = episode_ids[end_idx]
        if not np.all(episode_ids[start_idx:end_idx + 1] == eid):
            raise ValueError("lookback crosses gap/split/episode boundary")

        x = feature_matrix[start_idx:end_idx + 1].T.copy()
        expected = (self.market_feature_dim, self.lookback)
        if x.shape != expected:
            raise AssertionError(f"expected {expected}, got {x.shape}")
        return x

    def build_private_state(
        self,
        *,
        inventory: float,
        own_orders: Iterable[OwnOrderView],
        market_qty_by_level: Mapping[int, float],
    ) -> np.ndarray:
        grouped = {level: [] for level in self.levels}
        for order in own_orders:
            level = int(order.level)
            if level not in grouped:
                raise ValueError(f"own order outside current Q-grid: level={level}")
            if order.remaining_qty < 0:
                raise ValueError("negative remaining_qty")
            if order.queue_ahead < 0:
                raise ValueError("negative queue_ahead")
            if order.remaining_qty > self.eps:
                grouped[level].append(order)

        q_values = []
        v_values = []

        for level in self.levels:
            children = grouped[level]
            own_total = float(sum(x.remaining_qty for x in children))
            market_qty = max(0.0, float(market_qty_by_level.get(level, 0.0)))

            if own_total <= self.eps:
                q_level = 0.0
                v_level = 0.0
            else:
                denom = max(market_qty + own_total, self.eps)
                weighted = 0.0
                for child in children:
                    q_child = float(
                        np.clip(child.queue_ahead / denom, 0.0, 1.0)
                    )
                    weighted += (child.remaining_qty / own_total) * q_child
                q_level = float(np.clip(weighted, 0.0, 1.0))
                v_level = own_total / self.quote_size

            q_values.append(q_level)
            v_values.append(v_level)

        inventory_norm = float(inventory) / self.quote_size
        out = np.asarray(
            [inventory_norm, *q_values, *v_values],
            dtype=np.float32,
        )
        if out.shape != (self.private_dim,):
            raise AssertionError(out.shape)
        return out

    @staticmethod
    def zero_signal_state() -> np.ndarray:
        return np.zeros(4, dtype=np.float32)
