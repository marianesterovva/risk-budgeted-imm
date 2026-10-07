
import numpy as np

from mmrl.imm.risk_budget import (
    InventoryExposureCost,
    ProjectedLagrange,
    RiskIndexedReplayBuffer,
)


def test_inventory_exposure_cost():
    fn = InventoryExposureCost(reference_band_N=3.0)
    c, p = fn.compute(inventory=-8.0, quote_size=2.0)
    assert c == 4.0
    assert p["inventory_N"] == -4.0
    assert p["reference_band_violated"] is True
    assert p["excess_over_reference_band_N"] == 1.0


def test_projected_lagrange_moves_correct_direction():
    x = ProjectedLagrange(value=0.1, lr=0.5, max_value=10.0)
    high = x.update(estimated_cost=2.0, budget=1.0)
    assert high > 0.1
    low = x.update(estimated_cost=0.0, budget=2.0)
    assert 0.0 <= low < high


def test_risk_replay_shapes():
    b = RiskIndexedReplayBuffer(8, seed=1)
    for i in range(4):
        b.add(
            state_idx=i,
            next_state_idx=i + 1,
            private=np.zeros(21, np.float32),
            next_private=np.ones(21, np.float32),
            action=np.zeros(4, np.float32),
            reward=1.0,
            cost=0.5,
            done=False,
        )
    s = b.sample(3)
    assert s["private"].shape == (3, 21)
    assert s["action"].shape == (3, 4)
    assert s["reward"].shape == (3,)
    assert s["cost"].shape == (3,)
