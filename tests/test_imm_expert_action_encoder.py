
import numpy as np

from mmrl.imm.expert_action_encoder import (
    LTIICToIMMActionEncoder,
)
from mmrl.imm.ltiic import (
    LTIICDecision,
    LTIICParams,
)
from mmrl.simulator import (
    MarketBookSnapshot,
    TargetOrder,
)


def _book():
    return MarketBookSnapshot(
        timestamp_us=0,
        bids={
            100: 5.0,
            99: 4.0,
            98: 3.0,
            97: 2.0,
            96: 1.0,
        },
        asks={
            101: 5.0,
            102: 4.0,
            103: 3.0,
            104: 2.0,
            105: 1.0,
        },
        mid_price=100.5,
        tick_size=1.0,
    )


def _params():
    return LTIICParams(
        a_ticks=2.0,
        b_ticks_per_inventory_N=-0.25,
        c_ticks_per_trend=0.5,
        d_inventory_N=3.0,
    )


def test_two_sided_exact_encoding():
    enc = LTIICToIMMActionEncoder(
        quote_size=2.0
    )

    # ref2=201 => Q_-2=99, Q_+2=102
    decision = LTIICDecision(
        params=_params(),
        inventory=0.0,
        inventory_N=0.0,
        trend=0,
        desired_bid_tick_float=99.0,
        desired_ask_tick_float=102.0,
        actual_bid_tick=99,
        actual_ask_tick=102,
        quote_bid=True,
        quote_ask=True,
        bid_projected=False,
        ask_projected=False,
        target_book=(
            TargetOrder("buy", 99, 2.0),
            TargetOrder("sell", 102, 2.0),
        ),
    )

    result = enc.encode(
        decision=decision,
        ref2=201,
        current_book=_book(),
    )

    assert result.valid
    assert result.reason == "exact"
    assert result.normalized_action.shape == (4,)
    assert np.all(
        result.normalized_action >= -1
    )
    assert np.all(
        result.normalized_action <= 1
    )
    assert result.normalized_action[2] == 1.0
    assert result.normalized_action[3] == 1.0


def test_one_sided_is_not_faked():
    enc = LTIICToIMMActionEncoder(
        quote_size=2.0
    )

    decision = LTIICDecision(
        params=_params(),
        inventory=6.0,
        inventory_N=3.0,
        trend=0,
        desired_bid_tick_float=None,
        desired_ask_tick_float=102.0,
        actual_bid_tick=None,
        actual_ask_tick=102,
        quote_bid=False,
        quote_ask=True,
        bid_projected=False,
        ask_projected=False,
        target_book=(
            TargetOrder("sell", 102, 2.0),
        ),
    )

    result = enc.encode(
        decision=decision,
        ref2=201,
        current_book=_book(),
    )

    assert not result.valid
    assert "one_sided" in result.reason
    assert result.normalized_action is None
