"""Neural networks: the PINN trajectory surrogate and the shared alpha network."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ALPHA_MAX, ALPHA_MIN


def mlp(n_in, n_out, width, depth, act=nn.Tanh):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class AlphaNet(nn.Module):
    """Shared ocean-coupling network alpha(t) = g_phi(o(t)) in [ALPHA_MIN, ALPHA_MAX].

    One network for all storms, evaluated pointwise along each track, so the
    learned alpha varies with the ocean state and storm motion instead of being
    one constant per storm (Reviewer 4, comment 5).
    """

    def __init__(self, n_features, width=32, depth=2):
        super().__init__()
        self.net = mlp(n_features, 1, width, depth, act=nn.SiLU)

    def forward(self, features):              # (..., F) -> (...)
        raw = self.net(features).squeeze(-1)
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(raw)


class TrajectoryNet(nn.Module):
    """PINN surrogate of (V, m)(t) for a batch of storms.

    A shared trunk takes Fourier features of the normalised time s in [0, 1]
    and a learned per-storm embedding. Initial conditions are imposed exactly:

        V(s) = V0 + s * V* * u_V(s),     m(s) = m0 + s * u_m(s),

    so V(0) = V0 (fixed, from the observation) and m(0) = m0 (trainable,
    Reviewer 4 comment 5: m0 enters the loss only through this hard constraint).
    """

    def __init__(self, n_storms, embed_dim=16, width=64, depth=3, n_fourier=12):
        super().__init__()
        self.embed = nn.Embedding(n_storms, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.register_buffer("freqs", math.pi * torch.arange(1, n_fourier + 1, dtype=torch.float64))
        self.net = mlp(1 + 2 * n_fourier + embed_dim, 2, width, depth)

    def forward(self, s, V0, m0, v_star):
        """s: (B, N) normalised times. Returns V, m of shape (B, N)."""
        B, N = s.shape
        x = s.unsqueeze(-1)
        ang = x * self.freqs
        e = self.embed.weight[:B].unsqueeze(1).expand(B, N, -1)
        u = self.net(torch.cat([x, torch.sin(ang), torch.cos(ang), e], dim=-1))
        V = V0[:, None] + s * v_star * u[..., 0]
        m = m0[:, None] + s * u[..., 1]
        return V, m
