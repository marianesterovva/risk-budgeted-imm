
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with left-only causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.left_padding = (
            (int(kernel_size) - 1) * int(dilation)
        )

        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.left_padding, 0),
        )
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    """
    Causal residual temporal block.

    Implementation choice for the undisclosed TCN internals.
    Input/output stay (B, F, L).
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.temporal = CausalConv1d(
            feature_dim,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.project = nn.Conv1d(
            hidden_channels,
            feature_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.temporal(x)
        y = F.relu(y)
        y = self.dropout(y)
        y = self.project(y)
        return F.relu(x + y)


class IMMTemporalConvNet(nn.Module):
    """
    Temporal convolution stage of TCSA.

    Paper-required interface:
        (B, F, L) -> (B, F, L)
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)

        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    self.feature_dim,
                    hidden_channels=int(hidden_channels),
                    kernel_size=int(kernel_size),
                    dilation=int(d),
                    dropout=float(dropout),
                )
                for d in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"expected (B,F,L), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected F={self.feature_dim}, got {x.shape[1]}"
            )

        h = x
        for block in self.blocks:
            h = block(h)

        return h


class IMMSpatialAttention(nn.Module):
    """
    Spatial attention from IMM Eq. in Section 4.1.

    Given H_hat in R^{B x F x L}:

        a = H_hat W1
        c = H_hat W2
        G = sigmoid(a c^T + b)
        S_hat = V G
        S = row_softmax(S_hat)

    Returns S in R^{B x F x F}.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        lookback: int,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.lookback = int(lookback)

        self.w1 = nn.Parameter(
            torch.empty(self.lookback)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.lookback)
        )
        self.v = nn.Parameter(
            torch.empty(
                self.feature_dim,
                self.feature_dim,
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                self.feature_dim,
                self.feature_dim,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.w1,
            mean=0.0,
            std=1.0 / max(1.0, self.lookback ** 0.5),
        )
        nn.init.normal_(
            self.w2,
            mean=0.0,
            std=1.0 / max(1.0, self.lookback ** 0.5),
        )
        nn.init.xavier_uniform_(self.v)
        nn.init.zeros_(self.b)

    def forward(
        self,
        h_hat: torch.Tensor,
    ) -> torch.Tensor:
        if h_hat.ndim != 3:
            raise ValueError(
                f"expected (B,F,L), got {tuple(h_hat.shape)}"
            )
        if h_hat.shape[1] != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        if h_hat.shape[2] != self.lookback:
            raise ValueError("lookback dimension mismatch")

        # (B,F)
        left = torch.einsum(
            "bfl,l->bf",
            h_hat,
            self.w1,
        )
        right = torch.einsum(
            "bfl,l->bf",
            h_hat,
            self.w2,
        )

        # (B,F,F)
        outer = (
            left.unsqueeze(-1)
            * right.unsqueeze(-2)
        )

        gate = torch.sigmoid(
            outer + self.b.unsqueeze(0)
        )

        # V dot gate, following the paper notation.
        s_hat = torch.einsum(
            "ij,bjk->bik",
            self.v,
            gate,
        )

        # Row normalization.
        s = torch.softmax(
            s_hat,
            dim=-1,
        )

        return s


class IMMTCSAEncoder(nn.Module):
    """
    Paper-structured TCSA encoder.

        x
        -> TCN -> H_hat
        -> SpatialAttention -> S
        -> H = S H_hat + x
        -> s_m = sigmoid(W4 ReLU(H W3 + b3) + b4)

    Input:
        x: (B, F, L)

    Output:
        s_m: (B, latent_dim)
    """

    def __init__(
        self,
        *,
        feature_dim: int = 20,
        lookback: int = 64,
        latent_dim: int = 64,
        tcn_hidden_channels: int = 64,
        tcn_kernel_size: int = 3,
        tcn_dilations: Sequence[int] = (1, 2, 4),
        tcn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.lookback = int(lookback)
        self.latent_dim = int(latent_dim)

        self.tcn = IMMTemporalConvNet(
            self.feature_dim,
            hidden_channels=int(tcn_hidden_channels),
            kernel_size=int(tcn_kernel_size),
            dilations=tuple(int(x) for x in tcn_dilations),
            dropout=float(tcn_dropout),
        )

        self.spatial_attention = IMMSpatialAttention(
            feature_dim=self.feature_dim,
            lookback=self.lookback,
        )

        # W3 in the paper: temporal projection L -> scalar for each feature.
        self.w3 = nn.Parameter(
            torch.empty(self.lookback)
        )
        self.b3 = nn.Parameter(
            torch.zeros(self.feature_dim)
        )

        # W4, b4.
        self.w4 = nn.Linear(
            self.feature_dim,
            self.latent_dim,
        )

        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        nn.init.normal_(
            self.w3,
            mean=0.0,
            std=1.0 / max(1.0, self.lookback ** 0.5),
        )
        nn.init.zeros_(self.b3)
        nn.init.xavier_uniform_(self.w4.weight)
        nn.init.zeros_(self.w4.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        if x.ndim != 3:
            raise ValueError(
                f"expected (B,F,L), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        if x.shape[2] != self.lookback:
            raise ValueError("lookback mismatch")

        h_hat = self.tcn(x)
        s = self.spatial_attention(h_hat)

        # Paper residual:
        # H = S H_hat + x
        h = torch.bmm(
            s,
            h_hat,
        ) + x

        # Paper FC expression:
        # H W3 + b3 -> feature vector.
        pooled = torch.einsum(
            "bfl,l->bf",
            h,
            self.w3,
        )
        pooled = F.relu(
            pooled + self.b3.unsqueeze(0)
        )

        s_m = torch.sigmoid(
            self.w4(pooled)
        )

        if return_attention:
            return s_m, s

        return s_m


class IMMStateEncoder(nn.Module):
    """
    Concatenate market representation, predictive signals and private state.

    market tensor : (B,20,64)
    signal state  : (B,4)
    private state : (B,21)

    latent_dim=64 -> agent state dim 89.
    """

    def __init__(
        self,
        market_encoder: IMMTCSAEncoder,
        *,
        signal_dim: int = 4,
        private_dim: int = 21,
    ) -> None:
        super().__init__()

        self.market_encoder = market_encoder
        self.signal_dim = int(signal_dim)
        self.private_dim = int(private_dim)

    @property
    def output_dim(self) -> int:
        return (
            self.market_encoder.latent_dim
            + self.signal_dim
            + self.private_dim
        )

    def forward(
        self,
        market: torch.Tensor,
        signals: torch.Tensor,
        private: torch.Tensor,
    ) -> torch.Tensor:
        if signals.ndim != 2:
            raise ValueError("signals must be 2D")
        if private.ndim != 2:
            raise ValueError("private must be 2D")
        if signals.shape[1] != self.signal_dim:
            raise ValueError("signal dim mismatch")
        if private.shape[1] != self.private_dim:
            raise ValueError("private dim mismatch")

        market_latent = self.market_encoder(
            market
        )

        return torch.cat(
            [
                market_latent,
                signals,
                private,
            ],
            dim=-1,
        )


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    final_tanh: bool = False,
) -> nn.Sequential:
    layers = []
    d = int(input_dim)

    for h in hidden_dims:
        layers.append(
            nn.Linear(d, int(h))
        )
        layers.append(nn.ReLU())
        d = int(h)

    layers.append(
        nn.Linear(d, int(output_dim))
    )

    if final_tanh:
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class TD3Actor(nn.Module):
    """
    Actor head for normalized 4D IMM action.

    The paper specifies MLP agent + TD3 but does not disclose hidden widths.
    Default [256,256] is our reference implementation choice.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        action_dim: int = 4,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.net = _mlp(
            self.state_dim,
            hidden_dims,
            self.action_dim,
            final_tanh=True,
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(state)


class TD3Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        *,
        action_dim: int = 4,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.net = _mlp(
            self.state_dim + self.action_dim,
            hidden_dims,
            1,
            final_tanh=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [state, action],
            dim=-1,
        )
        return self.net(x)


class TD3TwinCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        *,
        action_dim: int = 4,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        self.q1 = TD3Critic(
            state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )
        self.q2 = TD3Critic(
            state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        return (
            self.q1(state, action),
            self.q2(state, action),
        )
