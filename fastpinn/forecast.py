"""Training a calibration and using it to forecast unseen storms.

Training (``fit``): on a set of training storms, learn m0 per storm and the
shared alpha network g_phi with the PINN or with shooting.

Forecast protocol for a storm not used in training (``forecast``):

1. alpha(t) = g_phi(o(t)) along the storm's track (the network is frozen);
2. m0 is either FAST's prior or estimated from the IBTrACS observations of
   the first ASSIM_H = 24 h only (``assimilate_m0``, alpha frozen);
3. FAST is integrated (RK4, 15 min) from the first observation with this m0
   and alpha(t); everything after hour 24 is a forecast and is what is scored.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from .calibrate import (LatentParams, PINNCalibrator, ShootingCalibrator, Weights, pretrain_alpha_to_prior,
                        train_pinn, train_shooting)
from .config import ASSIM_H, MS_TO_KNOTS, PINN_CURRICULUM, PINN_ITERS, PINN_KW, SHOOT_ITERS
from .data import make_batch
from .evaluate import solve_fast


def seed_all(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(method, batch, learn_m0=True, learn_alpha=True, w_phys=1.0, seed=0, pinn_iters=PINN_ITERS,
        shoot_iters=SHOOT_ITERS):
    """Learn m0 (per training storm) and the alpha network. Returns (params, wall-clock s)."""
    seed_all(seed)
    params = LatentParams(batch, batch.features.shape[-1], learn_m0=learn_m0, learn_alpha=learn_alpha)
    pretrain_alpha_to_prior(params.alpha_net, batch)
    tic = time.perf_counter()
    if method == "pinn":
        train_pinn(PINNCalibrator(batch, params, **PINN_KW), Weights(phys=w_phys), iters=pinn_iters,
                   curriculum=PINN_CURRICULUM)
    elif method == "shooting":
        train_shooting(ShootingCalibrator(batch, params), Weights(), iters=shoot_iters)
    else:
        raise ValueError(method)
    return params, time.perf_counter() - tic


def truncate(storm, n):
    """The first n hours of a storm."""
    arrays = {f.name: getattr(storm, f.name)[:n] for f in dataclasses.fields(storm)
              if isinstance(getattr(storm, f.name), np.ndarray)}
    return dataclasses.replace(storm, T=n, **arrays)


def assimilate_m0(storms, scaler, alpha, hours=ASSIM_H, iters=150):
    """m0 from the observations of the first ``hours`` h with alpha (B, T) frozen."""
    bs = make_batch([truncate(s, hours + 1) for s in storms], scaler)
    params = LatentParams(bs, bs.features.shape[-1], learn_m0=True, learn_alpha=False,
                          alpha_fixed=torch.as_tensor(alpha)[:, : hours + 1].detach().clone())
    train_shooting(ShootingCalibrator(bs, params), Weights(), iters=iters, lr_m0=5e-2)
    with torch.no_grad():
        return params.m0().clone()


def forecast(batch, m0, alpha):
    """FAST with the given m0 (B,) and alpha (B, T). Returns V_max in kt (B, T) and (V, m)."""
    V, m, vmax = solve_fast(batch, m0, alpha)
    return vmax * MS_TO_KNOTS, V, m
