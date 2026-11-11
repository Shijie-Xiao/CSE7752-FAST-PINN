"""Calibration of the latent FAST parameters m0 (per storm) and alpha(t) (shared network).

Two estimators with the same parameterisation and the same observation loss:

* :class:`PINNCalibrator` -- physics-informed neural network. A trajectory
  network (V, m)(t) with hard initial conditions is fit to the observations
  while the FAST residuals are penalised at collocation points:

      L = w_obs L_obs + w_phys L_phys + w_prior L_prior + w_bound L_bound

* :class:`ShootingCalibrator` -- classical baseline: the FAST ODE is
  integrated with differentiable RK4 and only L_obs (+ prior) is minimised
  ("numerical shooting", as in FAST-ML's differentiable-solver training).

In both, m0 = sigmoid(rho) in (0, 1) and alpha = g_phi(o(t)) in [0.13, 1].
Either latent quantity can be frozen at its physics prior (FAST-ML m0,
FHLO alpha) to calibrate them separately (Reviewer 4, compensation analysis).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .config import T_STAR, V_STAR
from .data import Batch
from .networks import AlphaNet, TrajectoryNet
from .physics import at_hour, fast_rhs, hourly_forcing, obs_operator, rk4_integrate


def _logit(p):
    p = torch.as_tensor(p).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


@dataclass
class Weights:
    obs: float = 1.0
    phys: float = 1.0
    prior: float = 1e-3
    bound: float = 1.0
    m0_prior: float = 0.0


@dataclass
class History:
    it: list = field(default_factory=list)
    loss: dict = field(default_factory=dict)
    wall: float = 0.0

    def log(self, it, terms):
        self.it.append(it)
        for k, v in terms.items():
            self.loss.setdefault(k, []).append(float(v))


class LatentParams(nn.Module):
    """m0 per storm and the shared alpha network, either one optionally frozen."""

    def __init__(self, batch: Batch, n_features: int, learn_m0=True, learn_alpha=True,
                 alpha_net: AlphaNet | None = None, m0_init=None, alpha_fixed=None):
        super().__init__()
        self.learn_m0, self.learn_alpha = learn_m0, learn_alpha
        m0_init = batch.m0_prior if m0_init is None else torch.as_tensor(m0_init, dtype=batch.V0.dtype)
        self.rho = nn.Parameter(_logit(m0_init).clone(), requires_grad=learn_m0)
        self.alpha_net = alpha_net if alpha_net is not None else AlphaNet(n_features).to(batch.V0.dtype)
        for p in self.alpha_net.parameters():
            p.requires_grad_(learn_alpha)
        self.alpha_fixed = batch.alpha_prior if alpha_fixed is None else alpha_fixed

    def m0(self):
        return torch.sigmoid(self.rho)

    def alpha(self, batch: Batch):
        return self.alpha_net(batch.features) if self.learn_alpha else self.alpha_fixed


def pretrain_alpha_to_prior(alpha_net: AlphaNet, batch: Batch, iters=600, lr=1e-2):
    """Warm start: fit g_phi to the FHLO physics-prior alpha(t) along the tracks."""
    flags = [p.requires_grad for p in alpha_net.parameters()]
    for p in alpha_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(alpha_net.parameters(), lr=lr)
    mask = batch.valid
    for _ in range(iters):
        opt.zero_grad()
        loss = ((alpha_net(batch.features) - batch.alpha_prior)[mask] ** 2).mean()
        loss.backward()
        opt.step()
    for p, f in zip(alpha_net.parameters(), flags):
        p.requires_grad_(f)
    return float(loss.detach())


def obs_loss(V, batch: Batch, extra_mask=None):
    pred = obs_operator(V, batch.lat, batch.u_tran, batch.v_tran)
    mask = batch.obs_mask if extra_mask is None else batch.obs_mask & extra_mask
    return (((pred - torch.nan_to_num(batch.v_obs)) / V_STAR)[mask] ** 2).mean()


def prior_loss(params: LatentParams, alpha, batch: Batch, w: Weights):
    out = torch.zeros((), dtype=batch.V0.dtype)
    if params.learn_alpha:
        out = out + w.prior * ((alpha - batch.alpha_prior)[batch.valid] ** 2).mean()
    if params.learn_m0 and w.m0_prior > 0:
        out = out + w.m0_prior * ((params.m0() - batch.m0_prior) ** 2).mean()
    return out


# --------------------------------------------------------------------------- #
# PINN
# --------------------------------------------------------------------------- #
class PINNCalibrator(nn.Module):
    """PINN for FAST with hard initial conditions.

    ``causal_eps`` enables causal training (Wang, Sankaran & Perdikaris 2022,
    "Respecting causality is all you need for training PINNs"): the window
    of each storm is split into ``n_chunks`` consecutive chunks and chunk i's
    residual is weighted by exp(-eps * sum_{j<i} L_j) (no gradient through
    the weight), so later times are only fitted once earlier ones are. Without
    it the network can settle on FAST's trivial equilibrium V = m = 0, which
    satisfies the residual everywhere except right after t = 0.
    """

    def __init__(self, batch: Batch, params: LatentParams, width=64, depth=3, n_fourier=12,
                 n_colloc=256, causal_eps: float | None = None, n_chunks: int = 32):
        super().__init__()
        self.batch = batch
        self.params = params
        self.traj = TrajectoryNet(batch.B, width=width, depth=depth, n_fourier=n_fourier).to(batch.V0.dtype)
        self.n_colloc = n_colloc
        self.causal_eps = causal_eps
        self.n_chunks = n_chunks
        self.span = (batch.lengths - 1).to(batch.V0.dtype)       # (B,) hours
        self.last_causal_min_weight = 1.0
        #: fraction of each storm's window currently trained (curriculum)
        self.s_max = 1.0

    def trajectory_at_hours(self, t_hours):
        s = t_hours / self.span[:, None]
        return self.traj(s.clamp(max=1.0), self.batch.V0, self.params.m0(), V_STAR)

    def nodes(self):
        T = self.batch.vp.shape[1]
        t = torch.arange(T, dtype=self.batch.V0.dtype).expand(self.batch.B, T)
        return self.trajectory_at_hours(t)

    def losses(self, w: Weights, colloc=None):
        b = self.batch
        B = b.B
        dtype = b.V0.dtype
        alpha_nodes = self.params.alpha(b)
        # collocation points: stratified random in [0, s_max] of each storm's own
        # window (s_max < 1 during a time-marching curriculum)
        if colloc is None:
            base = (torch.arange(self.n_colloc, dtype=dtype) + torch.rand(B, self.n_colloc, dtype=dtype)) / self.n_colloc
            base = base * self.s_max
        else:
            base = colloc
        s = base.clone().requires_grad_(True)
        V, m = self.traj(s, b.V0, self.params.m0(), V_STAR)
        dV_ds, = torch.autograd.grad(V.sum(), s, create_graph=True)
        dm_ds, = torch.autograd.grad(m.sum(), s, create_graph=True)
        t = s.detach() * self.span[:, None]
        dV_dt = dV_ds / self.span[:, None]
        dm_dt = dm_ds / self.span[:, None]
        # FAST's forcing of hour k on [k, k+1): piecewise-constant lookup
        kappa, a_, b_, g_, vp_, xs_ = hourly_forcing(b, alpha_nodes)
        f = [at_hour(x, t) for x in (a_, b_, g_, vp_, xs_)]
        fV, fm = fast_rhs(V, m, kappa, *f)
        rV = (dV_dt - fV) * T_STAR / V_STAR
        rm = (dm_dt - fm) * T_STAR
        res = rV ** 2 + rm ** 2                                   # (B, N)
        if self.causal_eps:
            nc = self.n_chunks
            idx = (s.detach() * nc).long().clamp(max=nc - 1)
            sums = torch.zeros(B, nc, dtype=dtype).scatter_add(1, idx, res)
            cnt = torch.zeros(B, nc, dtype=dtype).scatter_add(1, idx, torch.ones_like(res)).clamp(min=1)
            Lc = sums / cnt
            weights = torch.exp(-self.causal_eps * (torch.cumsum(Lc, 1) - Lc)).detach()
            self.last_causal_min_weight = float(weights.min())
            l_phys = (weights * Lc).mean()
        else:
            l_phys = res.mean()
        l_bound = (torch.relu(-m) ** 2 + torch.relu(m - 1) ** 2 + torch.relu(-V / V_STAR) ** 2).mean()

        Vn, _ = self.nodes()
        if self.s_max < 1.0:      # curriculum: only observations inside the trained window
            hours = torch.arange(Vn.shape[1], dtype=dtype)[None, :]
            inside = hours <= self.s_max * self.span[:, None]
            l_obs = obs_loss(Vn, b, extra_mask=inside)
        else:
            l_obs = obs_loss(Vn, b)
        l_prior = prior_loss(self.params, alpha_nodes, b, w)
        total = w.obs * l_obs + w.phys * l_phys + l_prior + w.bound * l_bound
        return total, {"total": total, "obs": l_obs, "phys": l_phys, "prior": l_prior, "bound": l_bound}


def train_pinn(cal: PINNCalibrator, w: Weights, iters=4000, lr=2e-3, lr_m0=2e-2, log_every=100,
               verbose=False, callback=None, curriculum: float = 0.0):
    """Adam with cosine decay. ``callback(it)`` is called at every log step
    (its time is excluded from ``History.wall``).

    ``curriculum`` > 0 enables time marching: the trained window
    [0, s_max] of every storm grows linearly from 5 % to 100 % over the
    first ``curriculum`` fraction of the iterations.
    """
    groups = [{"params": cal.traj.parameters(), "lr": lr}]
    if cal.params.learn_alpha:
        groups.append({"params": cal.params.alpha_net.parameters(), "lr": lr})
    if cal.params.learn_m0:
        groups.append({"params": [cal.params.rho], "lr": lr_m0})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    hist = History()
    tic = time.perf_counter()
    excluded = 0.0
    for it in range(1, iters + 1):
        if curriculum > 0:
            cal.s_max = min(1.0, 0.05 + 0.95 * it / (curriculum * iters))
        opt.zero_grad()
        total, terms = cal.losses(w)
        total.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters:
            t_cb = time.perf_counter()
            hist.log(it, {k: v.detach() for k, v in terms.items()})
            if callback is not None:
                callback(it)
            if verbose:
                print(f"  it {it:5d}  " + "  ".join(f"{k}={float(v):.3e}" for k, v in terms.items()))
            excluded += time.perf_counter() - t_cb
    cal.s_max = 1.0
    hist.wall = time.perf_counter() - tic - excluded
    return hist


# --------------------------------------------------------------------------- #
# Numerical shooting (differentiable RK4)
# --------------------------------------------------------------------------- #
class ShootingCalibrator(nn.Module):
    def __init__(self, batch: Batch, params: LatentParams, substeps=1):
        super().__init__()
        self.batch = batch
        self.params = params
        self.substeps = substeps

    def losses(self, w: Weights):
        b = self.batch
        alpha = self.params.alpha(b)
        V, m = rk4_integrate(b, self.params.m0(), alpha, substeps=self.substeps)
        l_obs = obs_loss(V, b)
        l_prior = prior_loss(self.params, alpha, b, w)
        total = w.obs * l_obs + l_prior
        return total, {"total": total, "obs": l_obs, "prior": l_prior}


def train_shooting(cal: ShootingCalibrator, w: Weights, iters=300, lr=3e-3, lr_m0=5e-2,
                   log_every=25, verbose=False):
    groups = []
    if cal.params.learn_alpha:
        groups.append({"params": cal.params.alpha_net.parameters(), "lr": lr})
    if cal.params.learn_m0:
        groups.append({"params": [cal.params.rho], "lr": lr_m0})
    opt = torch.optim.Adam(groups)
    hist = History()
    tic = time.perf_counter()
    for it in range(1, iters + 1):
        opt.zero_grad()
        total, terms = cal.losses(w)
        total.backward()
        opt.step()
        if it % log_every == 0 or it == iters:
            hist.log(it, {k: v.detach() for k, v in terms.items()})
            if verbose:
                print(f"  it {it:4d}  " + "  ".join(f"{k}={float(v):.3e}" for k, v in terms.items()))
    hist.wall = time.perf_counter() - tic
    return hist


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
