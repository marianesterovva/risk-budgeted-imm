
import torch

from mmrl.imm.tcsa import (
    CausalConv1d,
    IMMTCSAEncoder,
    IMMStateEncoder,
    TD3Actor,
    TD3TwinCritic,
)


def test_causal_conv_does_not_see_future():
    torch.manual_seed(1)

    conv = CausalConv1d(
        3,
        4,
        kernel_size=3,
        dilation=2,
    )
    conv.eval()

    x1 = torch.randn(2, 3, 20)
    x2 = x1.clone()

    # Change only positions 15..19.
    x2[:, :, 15:] += 1000.0

    y1 = conv(x1)
    y2 = conv(x2)

    # Outputs up through t=14 must be identical.
    assert torch.allclose(
        y1[:, :, :15],
        y2[:, :, :15],
        atol=1e-6,
        rtol=1e-6,
    )


def test_tcsa_shapes_attention_rows_and_range():
    torch.manual_seed(2)

    enc = IMMTCSAEncoder(
        feature_dim=20,
        lookback=64,
        latent_dim=64,
    )

    x = torch.randn(8, 20, 64)

    z, s = enc(
        x,
        return_attention=True,
    )

    assert z.shape == (8, 64)
    assert s.shape == (8, 20, 20)

    rows = s.sum(dim=-1)
    assert torch.allclose(
        rows,
        torch.ones_like(rows),
        atol=1e-6,
    )

    # Final sigmoid from the paper expression.
    assert torch.all(z >= 0)
    assert torch.all(z <= 1)


def test_tcsa_gradients_reach_paper_parameters():
    torch.manual_seed(3)

    enc = IMMTCSAEncoder(
        feature_dim=20,
        lookback=64,
        latent_dim=64,
    )

    x = torch.randn(
        4,
        20,
        64,
        requires_grad=True,
    )

    z = enc(x)
    loss = z.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    named = dict(enc.named_parameters())

    required = [
        "spatial_attention.w1",
        "spatial_attention.w2",
        "spatial_attention.v",
        "w3",
        "w4.weight",
    ]

    for name in required:
        grad = named[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name


def test_full_imm_state_is_89d():
    market_encoder = IMMTCSAEncoder(
        feature_dim=20,
        lookback=64,
        latent_dim=64,
    )

    state_encoder = IMMStateEncoder(
        market_encoder,
        signal_dim=4,
        private_dim=21,
    )

    market = torch.randn(5, 20, 64)
    signal = torch.zeros(5, 4)
    private = torch.randn(5, 21)

    state = state_encoder(
        market,
        signal,
        private,
    )

    assert state_encoder.output_dim == 89
    assert state.shape == (5, 89)


def test_td3_actor_and_twin_critic_interfaces():
    torch.manual_seed(4)

    state_dim = 89
    action_dim = 4

    actor = TD3Actor(
        state_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
    )
    critics = TD3TwinCritic(
        state_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
    )

    state = torch.randn(7, state_dim)
    action = actor(state)

    assert action.shape == (7, 4)
    assert torch.all(action >= -1)
    assert torch.all(action <= 1)

    q1, q2 = critics(
        state,
        action,
    )

    assert q1.shape == (7, 1)
    assert q2.shape == (7, 1)

    # Independent parameter sets.
    q1_ids = {
        id(p) for p in critics.q1.parameters()
    }
    q2_ids = {
        id(p) for p in critics.q2.parameters()
    }
    assert q1_ids.isdisjoint(q2_ids)


def test_end_to_end_backward():
    torch.manual_seed(5)

    market_encoder = IMMTCSAEncoder(
        feature_dim=20,
        lookback=64,
        latent_dim=64,
    )
    state_encoder = IMMStateEncoder(
        market_encoder,
        signal_dim=4,
        private_dim=21,
    )
    actor = TD3Actor(
        89,
        action_dim=4,
    )
    critics = TD3TwinCritic(
        89,
        action_dim=4,
    )

    market = torch.randn(3, 20, 64)
    signal = torch.zeros(3, 4)
    private = torch.randn(3, 21)

    state = state_encoder(
        market,
        signal,
        private,
    )
    action = actor(state)
    q1, q2 = critics(
        state,
        action,
    )

    loss = -(q1.mean() + q2.mean())
    loss.backward()

    assert market_encoder.w3.grad is not None
    assert actor.net[0].weight.grad is not None
    assert critics.q1.net[0].weight.grad is not None
    assert critics.q2.net[0].weight.grad is not None
