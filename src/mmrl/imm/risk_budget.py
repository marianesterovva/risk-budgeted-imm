
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InventoryExposureCost:
    'Absolute inventory in train-only quote-size units.'

    reference_band_N: float = 3.0

    def compute(self, *, inventory: float, quote_size: float):
        quote_size = float(quote_size)
        if quote_size <= 0:
            raise ValueError("quote_size must be positive")

        inventory_N = float(inventory) / quote_size
        abs_inventory_N = abs(inventory_N)
        excess_over_reference_band_N = max(
            0.0,
            abs_inventory_N - float(self.reference_band_N),
        )

        return float(abs_inventory_N), {
            "inventory_N": float(inventory_N),
            "abs_inventory_N": float(abs_inventory_N),
            "reference_band_violated": bool(
                abs_inventory_N > float(self.reference_band_N)
            ),
            "excess_over_reference_band_N": float(
                excess_over_reference_band_N
            ),
        }


@dataclass
class ProjectedLagrange:
    value: float = 0.0
    lr: float = 0.02
    max_value: float = 100.0

    def __post_init__(self):
        self.value = float(np.clip(self.value, 0.0, self.max_value))
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.max_value <= 0:
            raise ValueError("max_value must be positive")

    def update(self, estimated_cost: float, budget: float) -> float:
        estimated_cost = float(estimated_cost)
        budget = float(budget)
        if not np.isfinite(estimated_cost) or not np.isfinite(budget):
            raise ValueError("estimated_cost and budget must be finite")

        self.value = float(np.clip(
            self.value + self.lr * (estimated_cost - budget),
            0.0,
            self.max_value,
        ))
        return self.value


class RiskIndexedReplayBuffer:
    'Notebook-14 indexed replay schema plus one scalar cost.'

    def __init__(
        self,
        capacity: int,
        *,
        private_dim: int = 21,
        action_dim: int = 4,
        seed: int = 42,
    ):
        self.capacity = int(capacity)
        self.private_dim = int(private_dim)
        self.action_dim = int(action_dim)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.size = 0

        self.state_idx = np.empty(self.capacity, dtype=np.int64)
        self.next_state_idx = np.empty(self.capacity, dtype=np.int64)
        self.private = np.empty(
            (self.capacity, self.private_dim), dtype=np.float32
        )
        self.next_private = np.empty(
            (self.capacity, self.private_dim), dtype=np.float32
        )
        self.action = np.empty(
            (self.capacity, self.action_dim), dtype=np.float32
        )
        self.reward = np.empty(self.capacity, dtype=np.float32)
        self.cost = np.empty(self.capacity, dtype=np.float32)
        self.done = np.empty(self.capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def add(
        self,
        *,
        state_idx,
        next_state_idx,
        private,
        next_private,
        action,
        reward,
        cost,
        done,
    ):
        i = self.pos
        self.state_idx[i] = int(state_idx)
        self.next_state_idx[i] = int(next_state_idx)
        self.private[i] = np.asarray(private, dtype=np.float32)
        self.next_private[i] = np.asarray(next_private, dtype=np.float32)
        self.action[i] = np.asarray(action, dtype=np.float32)
        self.reward[i] = float(reward)
        self.cost[i] = float(cost)
        self.done[i] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        if self.size == 0:
            raise RuntimeError("cannot sample empty replay")
        n = int(n)
        idx = self.rng.integers(0, self.size, size=n)
        return {
            "state_idx": self.state_idx[idx].copy(),
            "next_state_idx": self.next_state_idx[idx].copy(),
            "private": self.private[idx].copy(),
            "next_private": self.next_private[idx].copy(),
            "action": self.action[idx].copy(),
            "reward": self.reward[idx].copy(),
            "cost": self.cost[idx].copy(),
            "done": self.done[idx].copy(),
        }
