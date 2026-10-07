
import numpy as np
import torch
from mmrl.imm.td3 import IndexedReplayBuffer, PaperShapedReward, polyak_update, reconstruct_market_batch

def test_replay():
    b = IndexedReplayBuffer(100)
    for i in range(20):
        b.add(
            state_idx=100+i, next_state_idx=101+i,
            private=np.zeros(21), next_private=np.zeros(21),
            action=np.zeros(4), reward=i, done=(i%5==0),
        )
    x = b.sample(8)
    assert x["private"].shape == (8,21)
    assert x["action"].shape == (8,4)

def test_reconstruct():
    f = np.arange(200*20, dtype=np.float32).reshape(200,20)
    x = reconstruct_market_batch(f, np.array([63,100]))
    assert x.shape == (2,20,64)
    assert np.array_equal(x[0,:,-1], f[63])

def test_polyak():
    a = torch.nn.Linear(3,2)
    b = torch.nn.Linear(3,2)
    with torch.no_grad():
        a.weight.fill_(1); a.bias.fill_(1)
        b.weight.fill_(0); b.bias.fill_(0)
    polyak_update(b,a,0.25)
    assert torch.allclose(b.weight, torch.full_like(b.weight,0.25))

def test_reward():
    r = PaperShapedReward()
    reward, p = r.compute(
        pnl=2, filled_notional=1000, inventory=4,
        quote_size=1, tick_size=.5, mid_price=100,
    )
    assert abs(reward - 3.9) < 1e-8
