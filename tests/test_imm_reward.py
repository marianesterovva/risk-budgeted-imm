
from mmrl.imm.reward import NotionalBpsReward

def test_notional_bps_reward():
    r = NotionalBpsReward(beta=0.01, eta=0.05, inventory_threshold_N=3.0)

    reward, p = r.compute(
        pnl=1.0,
        filled_notional=1000.0,
        inventory=4.0,
        quote_size=1.0,
        mid_price=100.0,
    )

    assert abs(p["pnl_bps"] - 100.0) < 1e-9
    assert abs(p["execution_term"] - 0.1) < 1e-9
    assert abs(p["inventory_penalty"] + 0.2) < 1e-9
    assert abs(reward - 99.9) < 1e-9
