
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .action_mapper import (
    DecodedIMMAction,
    IMMActionMapper,
)
from .grid import price_tick_to_grid_level


@dataclass(frozen=True)
class ExpertActionEncoding:
    valid: bool
    reason: str

    bid_level: Optional[int]
    ask_level: Optional[int]

    normalized_action: Optional[np.ndarray]

    quoted_mid_ticks: Optional[float]
    spread_ticks: Optional[float]


class LTIICToIMMActionEncoder:
    """
    Strict 4D encoding for exactly representable TWO-SIDED LTIIC decisions.

    One-sided expert decisions are deliberately not assigned a surrogate action.
    """

    def __init__(
        self,
        *,
        quote_size: float,
        m_step: float = 0.25,
        spread_step: float = 0.25,
    ) -> None:
        self.quote_size = float(quote_size)

        self.mapper = IMMActionMapper(
            quote_size=self.quote_size,
            k=5,
        )

        self.m_step = float(m_step)
        self.spread_step = float(spread_step)

        self.inverse = self._build_inverse_table()

    @staticmethod
    def _values(
        lo: float,
        hi: float,
        step: float,
    ) -> np.ndarray:
        n = int(
            round(
                (float(hi) - float(lo))
                / float(step)
            )
        )

        return np.linspace(
            float(lo),
            float(hi),
            n + 1,
        )

    def _build_inverse_table(self):
        table = {}

        m_values = self._values(
            self.mapper.quoted_mid_min,
            self.mapper.quoted_mid_max,
            self.m_step,
        )

        s_values = self._values(
            self.mapper.spread_min,
            self.mapper.spread_max,
            self.spread_step,
        )

        for m in m_values:
            for spread in s_values:
                bid = self.mapper._paper_levels(
                    side="buy",
                    boundary=float(m - spread / 2.0),
                )[0]

                ask = self.mapper._paper_levels(
                    side="sell",
                    boundary=float(m + spread / 2.0),
                )[0]

                key = (
                    int(bid),
                    int(ask),
                )

                # Prefer a compact quoted-mid and smaller spread when
                # multiple continuous actions select the same inner levels.
                score = (
                    abs(float(m)),
                    float(spread),
                    abs(float(m)) + float(spread),
                )

                old = table.get(key)

                if old is None or score < old[0]:
                    table[key] = (
                        score,
                        float(m),
                        float(spread),
                    )

        return {
            key: (
                value[1],
                value[2],
            )
            for key, value in table.items()
        }

    @staticmethod
    def _normalize(
        x: float,
        lo: float,
        hi: float,
    ) -> float:
        return float(
            np.clip(
                2.0
                * (
                    (float(x) - float(lo))
                    / (float(hi) - float(lo))
                )
                - 1.0,
                -1.0,
                1.0,
            )
        )

    @staticmethod
    def _target_signature(target_book):
        return tuple(
            sorted(
                (
                    str(order.side),
                    int(order.price_tick),
                    round(float(order.qty), 12),
                )
                for order in target_book
            )
        )

    def encode(
        self,
        *,
        decision,
        ref2: int,
        current_book,
    ) -> ExpertActionEncoding:
        if not (
            bool(decision.quote_bid)
            and bool(decision.quote_ask)
        ):
            return ExpertActionEncoding(
                valid=False,
                reason="one_sided_not_representable_by_public_4d_action",
                bid_level=None,
                ask_level=None,
                normalized_action=None,
                quoted_mid_ticks=None,
                spread_ticks=None,
            )

        if (
            decision.actual_bid_tick is None
            or decision.actual_ask_tick is None
        ):
            return ExpertActionEncoding(
                valid=False,
                reason="missing_two_sided_price",
                bid_level=None,
                ask_level=None,
                normalized_action=None,
                quoted_mid_ticks=None,
                spread_ticks=None,
            )

        bid_level = price_tick_to_grid_level(
            int(ref2),
            int(decision.actual_bid_tick),
            k=5,
        )

        ask_level = price_tick_to_grid_level(
            int(ref2),
            int(decision.actual_ask_tick),
            k=5,
        )

        if bid_level is None or ask_level is None:
            return ExpertActionEncoding(
                valid=False,
                reason="expert_price_outside_current_Q5_grid",
                bid_level=bid_level,
                ask_level=ask_level,
                normalized_action=None,
                quoted_mid_ticks=None,
                spread_ticks=None,
            )

        key = (
            int(bid_level),
            int(ask_level),
        )

        if key not in self.inverse:
            return ExpertActionEncoding(
                valid=False,
                reason="inner_level_pair_not_representable_in_action_bounds",
                bid_level=int(bid_level),
                ask_level=int(ask_level),
                normalized_action=None,
                quoted_mid_ticks=None,
                spread_ticks=None,
            )

        m_star, spread = self.inverse[key]

        u = np.asarray(
            [
                self._normalize(
                    m_star,
                    self.mapper.quoted_mid_min,
                    self.mapper.quoted_mid_max,
                ),
                self._normalize(
                    spread,
                    self.mapper.spread_min,
                    self.mapper.spread_max,
                ),
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        mapped = self.mapper.map_normalized(
            u,
            ref2=int(ref2),
            current_book=current_book,
        )

        if (
            self._target_signature(mapped.target_book)
            != self._target_signature(decision.target_book)
        ):
            return ExpertActionEncoding(
                valid=False,
                reason="4d_mapper_verification_failed",
                bid_level=int(bid_level),
                ask_level=int(ask_level),
                normalized_action=None,
                quoted_mid_ticks=float(m_star),
                spread_ticks=float(spread),
            )

        return ExpertActionEncoding(
            valid=True,
            reason="exact",
            bid_level=int(bid_level),
            ask_level=int(ask_level),
            normalized_action=u,
            quoted_mid_ticks=float(m_star),
            spread_ticks=float(spread),
        )
