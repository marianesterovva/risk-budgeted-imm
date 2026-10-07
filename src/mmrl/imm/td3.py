
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

class IndexedReplayBuffer:
    def __init__(self, capacity, private_dim=21, action_dim=4, seed=42):
        self.capacity = int(capacity)
        self.state_idx = np.empty(self.capacity, np.int32)
        self.next_state_idx = np.empty(self.capacity, np.int32)
        self.private = np.empty((self.capacity, private_dim), np.float32)
        self.next_private = np.empty((self.capacity, private_dim), np.float32)
        self.action = np.empty((self.capacity, action_dim), np.float32)
        self.reward = np.empty(self.capacity, np.float32)
        self.done = np.empty(self.capacity, np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(self.size)

    def add(self, *, state_idx, next_state_idx, private, next_private, action, reward, done):
        i = self.pos
        self.state_idx[i] = int(state_idx)
        self.next_state_idx[i] = int(next_state_idx)
        self.private[i] = np.asarray(private, np.float32)
        self.next_private[i] = np.asarray(next_private, np.float32)
        self.action[i] = np.asarray(action, np.float32)
        self.reward[i] = float(reward)
        self.done[i] = float(bool(done))
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("not enough replay")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return dict(
            state_idx=self.state_idx[idx].astype(np.int64),
            next_state_idx=self.next_state_idx[idx].astype(np.int64),
            private=self.private[idx],
            next_private=self.next_private[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            done=self.done[idx],
        )

@dataclass(frozen=True)
class PaperShapedReward:
    beta: float = 0.01
    eta: float = 0.05
    inventory_threshold_N: float = 3.0

    def compute(self, *, pnl, filled_notional, inventory, quote_size, tick_size, mid_price):
        pnl_n = float(pnl) / (float(quote_size) * float(tick_size))
        exec_units = float(filled_notional) / (float(quote_size) * float(mid_price))
        exec_term = self.beta * exec_units
        inv_n = float(inventory) / float(quote_size)
        penalty = -self.eta * abs(inv_n) if abs(inv_n) > self.inventory_threshold_N else 0.0
        reward = pnl_n + exec_term + penalty
        return float(reward), dict(
            pnl_normalized=float(pnl_n),
            execution_term=float(exec_term),
            inventory_N=float(inv_n),
            inventory_penalty=float(penalty),
        )

@torch.no_grad()
def polyak_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

@torch.no_grad()
def clipped_target_action(target_actor, target_state, target_noise, noise_clip):
    a = target_actor(target_state)
    n = (torch.randn_like(a) * target_noise).clamp(-noise_clip, noise_clip)
    return (a + n).clamp(-1.0, 1.0)

def reconstruct_market_batch(market_features, state_indices, lookback=64):
    state_indices = np.asarray(state_indices, np.int64)
    offsets = np.arange(-lookback + 1, 1, dtype=np.int64)
    rows = state_indices[:, None] + offsets[None, :]
    if np.any(rows < 0):
        raise ValueError("lookback underflow")
    x = market_features[rows]              # B,L,F
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1)), dtype=np.float32)
