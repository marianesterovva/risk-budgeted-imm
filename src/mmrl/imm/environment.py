
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from mmrl.simulator import (
    MarketBookSnapshot,
    ReplaySimulator,
    TradeEvent,
)

from .action_mapper import IMMActionMapper
from .grid import (
    CandidateReferencePriceTracker,
    build_grid_slots,
    price_tick_to_grid_level,
)
from .simulator_adapter import IMMGridReplayAdapter
from .state_builder import (
    IMMStateBuilder,
    OwnOrderView,
)


_QTY_ATTRS = (
    "remaining_qty",
    "remaining",
    "open_qty",
    "leaves_qty",
    "qty_remaining",
    "qty",
    "volume",
)

_QUEUE_ATTRS = (
    "queue_ahead",
    "queue_ahead_qty",
    "ahead_qty",
    "qty_ahead",
)


@dataclass(frozen=True)
class LiveChildOrder:
    side: str
    price_tick: int
    remaining_qty: float
    queue_ahead: float
    source_path: str


@dataclass(frozen=True)
class IMMObservation:
    market: np.ndarray
    signals: np.ndarray
    private: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "market": self.market,
            "signals": self.signals,
            "private": self.private,
        }


@dataclass(frozen=True)
class IMMRewardConfig:
    mode: str = "pnl"
    beta: float = 0.0
    eta: float = 0.0
    inventory_threshold: float = float("inf")

    def compute(
        self,
        *,
        pnl: float,
        filled_notional: float,
        inventory_at_decision: float,
    ) -> tuple[float, dict[str, float]]:
        mode = str(self.mode).lower()

        if mode not in {"pnl", "imm"}:
            raise ValueError(
                "reward mode must be 'pnl' or 'imm'"
            )

        pnl_term = float(pnl)

        compensation = 0.0
        inventory_penalty = 0.0

        if mode == "imm":
            compensation = (
                float(self.beta)
                * float(filled_notional)
            )

            if (
                abs(float(inventory_at_decision))
                > float(self.inventory_threshold)
            ):
                inventory_penalty = (
                    -float(self.eta)
                    * abs(float(inventory_at_decision))
                )

        reward = (
            pnl_term
            + compensation
            + inventory_penalty
        )

        return float(reward), {
            "pnl_term": float(pnl_term),
            "compensation_term": float(compensation),
            "inventory_penalty_term": float(inventory_penalty),
        }


def _get_numeric_attr(
    obj: Any,
    names: Sequence[str],
) -> tuple[Optional[str], Optional[float]]:
    for name in names:
        if not hasattr(obj, name):
            continue

        value = getattr(obj, name)

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            return str(name), float(value)

    return None, None


def discover_live_child_orders(
    simulator: Any,
    *,
    max_depth: int = 5,
    eps: float = 1e-12,
) -> list[LiveChildOrder]:
    """
    Read-only reflection adapter over simulator internals.

    Candidate object requirements:
      - side in {buy,sell};
      - integer-like price_tick;
      - queue-ahead attribute;
      - remaining quantity attribute;
      - positive remaining quantity.

    Recursive traversal is restricted to:
      - builtin containers;
      - objects from mmrl.simulator modules;
      - simulator root object.
    """
    found: list[LiveChildOrder] = []
    visited: set[int] = set()
    accepted_ids: set[int] = set()

    def visit(
        obj: Any,
        *,
        depth: int,
        path: str,
    ) -> None:
        if obj is None:
            return

        oid = id(obj)

        if oid in visited:
            return
        visited.add(oid)

        if depth > int(max_depth):
            return

        # Candidate order object.
        side = getattr(obj, "side", None)
        price_tick = getattr(obj, "price_tick", None)

        _, qty = _get_numeric_attr(
            obj,
            _QTY_ATTRS,
        )
        _, queue = _get_numeric_attr(
            obj,
            _QUEUE_ATTRS,
        )

        if (
            isinstance(side, str)
            and side.lower() in {"buy", "sell"}
            and isinstance(
                price_tick,
                (int, np.integer),
            )
            and qty is not None
            and queue is not None
            and qty > float(eps)
        ):
            if oid not in accepted_ids:
                accepted_ids.add(oid)

                found.append(
                    LiveChildOrder(
                        side=side.lower(),
                        price_tick=int(price_tick),
                        remaining_qty=float(qty),
                        queue_ahead=max(
                            0.0,
                            float(queue),
                        ),
                        source_path=str(path),
                    )
                )

            return

        # Containers.
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(
                    value,
                    depth=depth + 1,
                    path=f"{path}[{key!r}]",
                )
            return

        if isinstance(obj, (list, tuple, set)):
            for i, value in enumerate(obj):
                visit(
                    value,
                    depth=depth + 1,
                    path=f"{path}[{i}]",
                )
            return

        # Avoid traversing pandas/numpy/torch/etc.
        module = getattr(
            type(obj),
            "__module__",
            "",
        )

        if (
            obj is not simulator
            and not str(module).startswith("mmrl.simulator")
        ):
            return

        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                if str(key).startswith("__"):
                    continue

                visit(
                    value,
                    depth=depth + 1,
                    path=f"{path}.{key}",
                )

    visit(
        simulator,
        depth=0,
        path="simulator",
    )

    found.sort(
        key=lambda x: (
            x.side,
            x.price_tick,
            x.source_path,
        )
    )

    return found


def live_orders_to_private_views(
    live_orders: Sequence[LiveChildOrder],
    *,
    ref2: int,
    k: int = 5,
) -> tuple[list[OwnOrderView], dict[str, float]]:
    views = []

    offgrid_qty = 0.0
    offgrid_count = 0

    for order in live_orders:
        level = price_tick_to_grid_level(
            int(ref2),
            int(order.price_tick),
            k=int(k),
        )

        if level is None:
            offgrid_count += 1
            offgrid_qty += float(order.remaining_qty)

            # Edge-saturating representation fallback only.
            if 2 * int(order.price_tick) < int(ref2):
                level = -int(k)
            else:
                level = int(k)

        views.append(
            OwnOrderView(
                level=int(level),
                remaining_qty=float(order.remaining_qty),
                queue_ahead=float(order.queue_ahead),
            )
        )

    return views, {
        "n_live_child_orders": int(len(live_orders)),
        "offgrid_live_child_count": int(offgrid_count),
        "offgrid_live_qty": float(offgrid_qty),
    }


class RawTradeSlicer:
    def __init__(
        self,
        trades: pd.DataFrame,
        *,
        tick_size: float,
    ) -> None:
        self.trades = trades.reset_index(drop=True)
        self.tick_size = float(tick_size)

        self.ts = (
            self.trades["effective_timestamp_us"]
            .to_numpy(np.int64)
        )

    def events(
        self,
        start_us: int,
        end_us: int,
    ) -> list[TradeEvent]:
        left = int(
            np.searchsorted(
                self.ts,
                int(start_us),
                side="left",
            )
        )
        right = int(
            np.searchsorted(
                self.ts,
                int(end_us),
                side="left",
            )
        )

        if right <= left:
            return []

        x = self.trades.iloc[left:right]

        return [
            TradeEvent(
                timestamp_us=int(r.effective_timestamp_us),
                price_tick=int(
                    round(
                        float(r.price)
                        / self.tick_size
                    )
                ),
                qty=float(r.qty),
                is_buyer_maker=bool(r.is_buyer_maker),
            )
            for r in x.itertuples(index=False)
        ]


class IMMMarketData:
    """
    One-asset in-memory replay data.

    Precomputes only the lightweight 20D market features.
    No 64-window tensors are materialized.
    """

    def __init__(
        self,
        *,
        symbol: str,
        states: pd.DataFrame,
        split: pd.DataFrame,
        trades: pd.DataFrame,
        quote_size: float,
        tick_size: float,
        lookback: int = 64,
    ) -> None:
        self.symbol = str(symbol)
        self.states = states.reset_index(drop=True)
        self.split = split.reset_index(drop=True)
        self.trades = trades.reset_index(drop=True)

        self.quote_size = float(quote_size)
        self.tick_size = float(tick_size)
        self.lookback = int(lookback)

        if len(self.states) != len(self.split):
            raise ValueError(
                "states/split length mismatch"
            )

        ts1 = (
            self.states["local_timestamp"]
            .to_numpy(np.int64)
        )
        ts2 = (
            self.split["local_timestamp"]
            .to_numpy(np.int64)
        )

        if not np.array_equal(ts1, ts2):
            raise ValueError(
                "states/split timestamp mismatch"
            )

        self.timestamp_ms = ts1

        self.episode_id = (
            self.split["experiment_episode_id"]
            .to_numpy(np.int64)
        )

        self.split_name = (
            self.split["split"]
            .astype(str)
            .to_numpy()
        )

        ready_col = (
            f"lookback_{self.lookback}_ready"
        )

        if ready_col not in self.split.columns:
            raise ValueError(
                f"missing {ready_col}"
            )

        self.lookback_ready = (
            self.split[ready_col]
            .to_numpy(bool)
        )

        self.transition_valid = (
            self.split["transition_valid"]
            .to_numpy(bool)
        )

        self.bid_ticks = np.rint(
            self.states[
                [
                    f"bid_price_{i}"
                    for i in range(1, 6)
                ]
            ].to_numpy(float)
            / self.tick_size
        ).astype(np.int64)

        self.ask_ticks = np.rint(
            self.states[
                [
                    f"ask_price_{i}"
                    for i in range(1, 6)
                ]
            ].to_numpy(float)
            / self.tick_size
        ).astype(np.int64)

        self.bid_qty = self.states[
            [
                f"bid_qty_{i}"
                for i in range(1, 6)
            ]
        ].to_numpy(np.float64)

        self.ask_qty = self.states[
            [
                f"ask_qty_{i}"
                for i in range(1, 6)
            ]
        ].to_numpy(np.float64)

        self.mid_price = (
            self.states["mid_price"]
            .to_numpy(np.float64)
        )

        self.ref2 = self._precompute_ref2()
        self.market_features = (
            self._precompute_market_features()
        )

        self.trade_slicer = RawTradeSlicer(
            self.trades,
            tick_size=self.tick_size,
        )

    def _precompute_ref2(self) -> np.ndarray:
        tracker = CandidateReferencePriceTracker()

        out = np.empty(
            len(self.states),
            dtype=np.int64,
        )

        prev_eid = None

        for i in range(len(out)):
            eid = int(self.episode_id[i])

            if prev_eid is None or eid != prev_eid:
                tracker.clear()

            bids = {
                int(self.bid_ticks[i, j]):
                    float(self.bid_qty[i, j])
                for j in range(5)
            }

            asks = {
                int(self.ask_ticks[i, j]):
                    float(self.ask_qty[i, j])
                for j in range(5)
            }

            out[i] = tracker.update(
                bids,
                asks,
            )

            prev_eid = eid

        return out

    def _precompute_market_features(
        self,
    ) -> np.ndarray:
        """
        Vectorized equivalent of IMMStateBuilder.snapshot_market_features.
        """
        n = len(self.states)

        levels = np.asarray(
            [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            dtype=np.int64,
        )

        offset2 = (
            np.sign(levels)
            * (2 * np.abs(levels) - 1)
        )

        q_ticks = (
            self.ref2[:, None]
            + offset2[None, :]
        ) // 2

        qty = np.zeros(
            (n, 10),
            dtype=np.float64,
        )

        occupied = np.zeros(
            (n, 10),
            dtype=bool,
        )

        for j in range(5):
            mb = (
                q_ticks
                == self.bid_ticks[:, j][:, None]
            )
            ma = (
                q_ticks
                == self.ask_ticks[:, j][:, None]
            )

            qty += (
                mb
                * self.bid_qty[:, j][:, None]
            )
            qty += (
                ma
                * self.ask_qty[:, j][:, None]
            )

            occupied |= mb
            occupied |= ma

        best_bid = self.bid_ticks[:, 0][:, None]
        deepest_bid = self.bid_ticks[:, 4][:, None]
        best_ask = self.ask_ticks[:, 0][:, None]
        deepest_ask = self.ask_ticks[:, 4][:, None]

        known = (
            (
                (q_ticks >= deepest_bid)
                & (q_ticks <= best_bid)
            )
            | (
                (q_ticks > best_bid)
                & (q_ticks < best_ask)
            )
            | (
                (q_ticks >= best_ask)
                & (q_ticks <= deepest_ask)
            )
        )

        # Unknown carries zero queue quantity plus mask=0.
        qty = np.where(
            known,
            qty,
            0.0,
        )

        transformed = np.log1p(
            np.maximum(qty, 0.0)
            / self.quote_size
        ).astype(np.float32)

        mask = known.astype(np.float32)

        out = np.concatenate(
            [transformed, mask],
            axis=1,
        ).astype(np.float32)

        if out.shape != (n, 20):
            raise AssertionError(out.shape)

        return out

    def book(self, idx: int) -> MarketBookSnapshot:
        idx = int(idx)

        bids = {
            int(self.bid_ticks[idx, j]):
                float(self.bid_qty[idx, j])
            for j in range(5)
        }

        asks = {
            int(self.ask_ticks[idx, j]):
                float(self.ask_qty[idx, j])
            for j in range(5)
        }

        return MarketBookSnapshot(
            timestamp_us=int(
                self.timestamp_ms[idx]
            ) * 1000,
            bids=bids,
            asks=asks,
            mid_price=float(
                self.mid_price[idx]
            ),
            tick_size=float(
                self.tick_size
            ),
        )

    def market_tensor(
        self,
        idx: int,
    ) -> np.ndarray:
        idx = int(idx)
        start = idx - self.lookback + 1

        if start < 0:
            raise ValueError(
                "lookback not ready"
            )

        eid = self.episode_id[idx]

        if not np.all(
            self.episode_id[start:idx + 1]
            == eid
        ):
            raise ValueError(
                "lookback crosses experiment episode"
            )

        x = (
            self.market_features[
                start:idx + 1
            ]
            .T
            .copy()
        )

        if x.shape != (20, self.lookback):
            raise AssertionError(x.shape)

        return x

    def episode_bounds(
        self,
        idx: int,
    ) -> tuple[int, int]:
        idx = int(idx)
        eid = self.episode_id[idx]

        left = idx
        while (
            left > 0
            and self.episode_id[left - 1] == eid
        ):
            left -= 1

        right = idx
        n = len(self.states)

        while (
            right + 1 < n
            and self.episode_id[right + 1] == eid
        ):
            right += 1

        return int(left), int(right)


class IMMMarketMakingEnv:
    """
    Gym-like environment without a hard dependency on gymnasium.

    Observation:
      market  : float32 (20,64)
      signals : float32 (4,)
      private : float32 (21,)

    One env.step(action):
      - action is reconciled once at the current decision snapshot;
      - every real snapshot/trade until the next >=500ms decision is replayed;
      - intermediate transitions use target_book=None.
    """

    def __init__(
        self,
        *,
        data: IMMMarketData,
        simulator: ReplaySimulator,
        decision_interval_ms: int = 500,
        episode_duration_minutes: float = 90.0,
        reward_config: Optional[IMMRewardConfig] = None,
        signal_matrix: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.simulator = simulator
        self.replay = IMMGridReplayAdapter(
            simulator
        )

        self.decision_interval_ms = int(
            decision_interval_ms
        )
        self.episode_duration_ms = int(
            round(
                float(episode_duration_minutes)
                * 60_000.0
            )
        )

        if self.decision_interval_ms <= 0:
            raise ValueError(
                "decision interval must be positive"
            )

        self.reward_config = (
            reward_config
            if reward_config is not None
            else IMMRewardConfig(mode="pnl")
        )

        self.state_builder = IMMStateBuilder(
            quote_size=self.data.quote_size,
            k=5,
            lookback=self.data.lookback,
        )

        self.action_mapper = IMMActionMapper(
            quote_size=self.data.quote_size,
            k=5,
        )

        if signal_matrix is None:
            self.signal_matrix = np.zeros(
                (len(self.data.states), 4),
                dtype=np.float32,
            )
        else:
            self.signal_matrix = np.asarray(
                signal_matrix,
                dtype=np.float32,
            )

            if self.signal_matrix.shape != (
                len(self.data.states),
                4,
            ):
                raise ValueError(
                    "signal matrix shape mismatch"
                )

        self.rng = np.random.default_rng(
            int(seed)
        )

        self.idx: Optional[int] = None
        self.start_idx: Optional[int] = None
        self.episode_right: Optional[int] = None
        self.episode_end_timestamp_ms: Optional[int] = None

        self.inventory = 0.0
        self.cash = 0.0

        self.cumulative_pnl = 0.0
        self.decision_count = 0

    def eligible_starts(
        self,
        *,
        split_name: str,
        require_full_duration: bool = False,
    ) -> np.ndarray:
        split_name = str(split_name)

        mask = (
            (self.data.split_name == split_name)
            & self.data.lookback_ready
        )

        idx = np.flatnonzero(mask)

        if not require_full_duration:
            return idx.astype(np.int64)

        keep = []

        for i in idx:
            _, right = self.data.episode_bounds(
                int(i)
            )

            available = (
                int(self.data.timestamp_ms[right])
                - int(self.data.timestamp_ms[i])
            )

            if available >= self.episode_duration_ms:
                keep.append(int(i))

        return np.asarray(
            keep,
            dtype=np.int64,
        )

    def reset(
        self,
        *,
        split_name: str = "train",
        start_idx: Optional[int] = None,
        require_full_duration: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict]:
        self.replay.reset()

        if start_idx is None:
            eligible = self.eligible_starts(
                split_name=split_name,
                require_full_duration=(
                    require_full_duration
                ),
            )

            if len(eligible) == 0:
                raise ValueError(
                    f"no eligible starts for {split_name}"
                )

            start_idx = int(
                self.rng.choice(eligible)
            )
        else:
            start_idx = int(start_idx)

            if (
                self.data.split_name[start_idx]
                != str(split_name)
            ):
                raise ValueError(
                    "explicit start_idx is in another split"
                )

            if not self.data.lookback_ready[start_idx]:
                raise ValueError(
                    "explicit start_idx is not lookback-ready"
                )

        _, right = self.data.episode_bounds(
            start_idx
        )

        self.idx = int(start_idx)
        self.start_idx = int(start_idx)
        self.episode_right = int(right)

        self.episode_end_timestamp_ms = min(
            int(
                self.data.timestamp_ms[
                    self.episode_right
                ]
            ),
            int(
                self.data.timestamp_ms[
                    self.start_idx
                ]
            )
            + self.episode_duration_ms,
        )

        self.inventory = 0.0
        self.cash = 0.0

        self.cumulative_pnl = 0.0
        self.decision_count = 0

        obs, private_info = self._observation()

        info = {
            "symbol": self.data.symbol,
            "state_idx": int(self.idx),
            "start_idx": int(self.start_idx),
            "split": str(split_name),
            "timestamp_ms": int(
                self.data.timestamp_ms[self.idx]
            ),
            "episode_right_idx": int(
                self.episode_right
            ),
            "episode_end_timestamp_ms": int(
                self.episode_end_timestamp_ms
            ),
            **private_info,
        }

        return obs.as_dict(), info

    def _market_qty_by_level(
        self,
        idx: int,
    ) -> dict[int, float]:
        book = self.data.book(idx)

        slots = build_grid_slots(
            book.bids,
            book.asks,
            int(self.data.ref2[idx]),
            k=5,
        )

        return {
            int(slot.level):
                float(slot.qty)
            for slot in slots
        }

    def _observation(
        self,
    ) -> tuple[IMMObservation, dict]:
        if self.idx is None:
            raise RuntimeError(
                "env not reset"
            )

        idx = int(self.idx)

        market = self.data.market_tensor(
            idx
        ).astype(
            np.float32,
            copy=False,
        )

        signals = (
            self.signal_matrix[idx]
            .astype(
                np.float32,
                copy=True,
            )
        )

        live = discover_live_child_orders(
            self.simulator
        )

        views, private_info = (
            live_orders_to_private_views(
                live,
                ref2=int(
                    self.data.ref2[idx]
                ),
                k=5,
            )
        )

        private = (
            self.state_builder
            .build_private_state(
                inventory=float(
                    self.inventory
                ),
                own_orders=views,
                market_qty_by_level=(
                    self._market_qty_by_level(
                        idx
                    )
                ),
            )
        )

        if market.shape != (20, 64):
            raise AssertionError(
                market.shape
            )
        if signals.shape != (4,):
            raise AssertionError(
                signals.shape
            )
        if private.shape != (21,):
            raise AssertionError(
                private.shape
            )

        return (
            IMMObservation(
                market=market,
                signals=signals,
                private=private,
            ),
            private_info,
        )

    def _find_next_decision_idx(
        self,
    ) -> int:
        if (
            self.idx is None
            or self.episode_right is None
            or self.episode_end_timestamp_ms is None
        ):
            raise RuntimeError(
                "env not reset"
            )

        i = int(self.idx)

        target_ts = (
            int(self.data.timestamp_ms[i])
            + self.decision_interval_ms
        )

        # First real snapshot >= decision target time.
        j = int(
            np.searchsorted(
                self.data.timestamp_ms,
                target_ts,
                side="left",
            )
        )

        j = min(
            j,
            int(self.episode_right),
        )

        # Also respect 90-minute cap.
        cap_j = int(
            np.searchsorted(
                self.data.timestamp_ms,
                int(
                    self.episode_end_timestamp_ms
                ),
                side="right",
            )
            - 1
        )

        j = min(
            j,
            cap_j,
            int(self.episode_right),
        )

        if j <= i:
            # Terminal region with less than one future snapshot.
            j = min(
                i + 1,
                int(self.episode_right),
            )

        return int(j)

    def step(
        self,
        action: Sequence[float] | np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict,
    ]:
        if (
            self.idx is None
            or self.start_idx is None
            or self.episode_right is None
            or self.episode_end_timestamp_ms is None
        ):
            raise RuntimeError(
                "env not reset"
            )

        start_idx = int(self.idx)

        if start_idx >= int(self.episode_right):
            raise RuntimeError(
                "step called after episode end"
            )

        current_book = self.data.book(
            start_idx
        )

        mapped = (
            self.action_mapper.map_normalized(
                action,
                ref2=int(
                    self.data.ref2[start_idx]
                ),
                current_book=current_book,
            )
        )

        next_decision_idx = (
            self._find_next_decision_idx()
        )

        if next_decision_idx <= start_idx:
            raise RuntimeError(
                "no valid future state"
            )

        inventory_at_decision = float(
            self.inventory
        )

        pnl = 0.0
        filled_notional = 0.0

        placed_qty = 0.0
        canceled_qty = 0.0
        filled_qty = 0.0

        buy_fill_qty = 0.0
        sell_fill_qty = 0.0

        exact_queue_fill_qty = 0.0
        trade_through_fill_qty = 0.0

        n_trade_events = 0
        n_fill_events = 0
        n_market_transitions = 0

        first = True

        for i in range(
            start_idx,
            next_decision_idx,
        ):
            if not bool(
                self.data.transition_valid[i]
            ):
                raise RuntimeError(
                    f"invalid transition {i}"
                )

            cur = self.data.book(i)
            nxt = self.data.book(i + 1)

            events = (
                self.data.trade_slicer.events(
                    cur.timestamp_us,
                    nxt.timestamp_us,
                )
            )

            result = self.replay.step(
                current_book=cur,
                next_book=nxt,
                trades=events,
                target_book=(
                    mapped.target_book
                    if first
                    else None
                ),
            )

            first = False

            pnl += float(result.pnl)

            placed_qty += float(
                result.placed_qty
            )
            canceled_qty += float(
                result.canceled_qty
            )
            filled_qty += float(
                result.filled_qty
            )

            buy_fill_qty += float(
                result.buy_fill_qty
            )
            sell_fill_qty += float(
                result.sell_fill_qty
            )

            exact_queue_fill_qty += float(
                result.exact_queue_fill_qty
            )
            trade_through_fill_qty += float(
                result.trade_through_fill_qty
            )

            n_trade_events += int(
                len(events)
            )
            n_fill_events += int(
                len(result.fills)
            )
            n_market_transitions += 1

            for fill in result.fills:
                filled_notional += float(
                    fill.notional
                )

            self.inventory = float(
                result.inventory
            )
            self.cash = float(
                result.cash
            )

        self.idx = int(
            next_decision_idx
        )

        self.cumulative_pnl += float(
            pnl
        )
        self.decision_count += 1

        reward, reward_parts = (
            self.reward_config.compute(
                pnl=float(pnl),
                filled_notional=float(
                    filled_notional
                ),
                inventory_at_decision=(
                    inventory_at_decision
                ),
            )
        )

        timestamp_now = int(
            self.data.timestamp_ms[
                self.idx
            ]
        )

        terminated = (
            self.idx >= int(
                self.episode_right
            )
        )

        truncated = (
            timestamp_now
            >= int(
                self.episode_end_timestamp_ms
            )
            and not terminated
        )

        obs, private_info = (
            self._observation()
        )

        elapsed_ms = (
            int(
                self.data.timestamp_ms[
                    self.idx
                ]
            )
            - int(
                self.data.timestamp_ms[
                    start_idx
                ]
            )
        )

        info = {
            "symbol": self.data.symbol,

            "decision_index": int(
                self.decision_count
            ),
            "state_idx_start": int(
                start_idx
            ),
            "state_idx_end": int(
                self.idx
            ),
            "timestamp_start_ms": int(
                self.data.timestamp_ms[
                    start_idx
                ]
            ),
            "timestamp_end_ms": int(
                self.data.timestamp_ms[
                    self.idx
                ]
            ),
            "elapsed_ms": int(
                elapsed_ms
            ),

            "n_market_transitions": int(
                n_market_transitions
            ),
            "n_trade_events": int(
                n_trade_events
            ),
            "n_fill_events": int(
                n_fill_events
            ),

            "pnl": float(pnl),
            "cumulative_pnl": float(
                self.cumulative_pnl
            ),
            "reward": float(reward),

            **reward_parts,

            "cash": float(self.cash),
            "inventory": float(
                self.inventory
            ),

            "filled_notional": float(
                filled_notional
            ),
            "filled_qty": float(
                filled_qty
            ),
            "buy_fill_qty": float(
                buy_fill_qty
            ),
            "sell_fill_qty": float(
                sell_fill_qty
            ),

            "placed_qty": float(
                placed_qty
            ),
            "canceled_qty": float(
                canceled_qty
            ),

            "exact_queue_fill_qty": float(
                exact_queue_fill_qty
            ),
            "trade_through_fill_qty": float(
                trade_through_fill_qty
            ),

            "quoted_mid_ticks": float(
                mapped.decoded.quoted_mid_ticks
            ),
            "spread_ticks": float(
                mapped.decoded.spread_ticks
            ),
            "phi_bid": float(
                mapped.decoded.phi_bid
            ),
            "phi_ask": float(
                mapped.decoded.phi_ask
            ),

            "desired_bid_levels": (
                mapped.desired_bid_levels
            ),
            "desired_ask_levels": (
                mapped.desired_ask_levels
            ),
            "actual_bid_levels": (
                mapped.actual_bid_levels
            ),
            "actual_ask_levels": (
                mapped.actual_ask_levels
            ),

            "bid_projected": bool(
                mapped.bid_projected
            ),
            "ask_projected": bool(
                mapped.ask_projected
            ),

            "bid_projection_grid_steps": int(
                mapped.bid_projection_grid_steps
            ),
            "ask_projection_grid_steps": int(
                mapped.ask_projection_grid_steps
            ),

            "target_book": tuple(
                mapped.target_book
            ),

            **private_info,
        }

        return (
            obs.as_dict(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )
