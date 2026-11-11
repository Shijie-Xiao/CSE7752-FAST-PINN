"""Tests: data re-parameterisation, torch FAST physics vs NumPy FAST, networks, losses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_default_dtype(torch.float64)

from fast.fast_equations import rhs  # noqa: E402
from fast.wind_conversion import axi_to_max_wind  # noqa: E402
from fastpinn.calibrate import LatentParams, PINNCalibrator, ShootingCalibrator, Weights  # noqa: E402
from fastpinn.config import ALPHA_MAX, ALPHA_MIN, DATA_DIR, KAPPA  # noqa: E402
from fastpinn.data import FeatureScaler, load_storms, make_batch  # noqa: E402
from fastpinn.networks import AlphaNet  # noqa: E402
from fastpinn.physics import obs_operator, rk4_integrate  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    storms = load_storms(["IRMA", "BRET", "LEE"])
    return storms, make_batch(storms, FeatureScaler(storms))


def test_gamma_reparameterisation_reproduces_data(batch):
    storms, b = batch
    for i, s in enumerate(storms):
        df = pd.read_csv(DATA_DIR / f"{s.meta['storm_id']}.csv")
        np.testing.assert_allclose(b.gamma(b.alpha_prior)[i, :s.T].numpy(), df["gamma"], atol=1e-12)
        np.testing.assert_allclose(b.beta[i, :s.T].numpy(), df["beta"], atol=1e-12)


def test_torch_rk4_matches_numpy_fast(batch):
    storms, b = batch
    with torch.no_grad():
        V, m = rk4_integrate(b, b.m0_prior, b.alpha_prior, substeps=4)
    i = 1                                                    # BRET (short)
    s = storms[i]
    y = np.array([s.V0, s.m0_prior])
    for k in range(s.T - 1):
        F = (KAPPA, s.alpha_prior[k], s.beta[k], s.eps[k] + s.k[k] * s.alpha_prior[k], s.vp[k], s.xs[k])
        for _ in range(64):                                  # fine forward Euler in NumPy as an independent check
            y = y + rhs(y, F) / 64
    assert abs(float(V[i, s.T - 1]) - y[0]) < 0.05


def test_obs_operator_matches_fast(batch):
    storms, b = batch
    V = torch.linspace(5, 80, b.vp.shape[1]).expand(b.B, -1)
    out = obs_operator(V, b.lat, b.u_tran, b.v_tran).numpy()
    for i, s in enumerate(storms):
        ref = axi_to_max_wind(V[i, :s.T].numpy(), s.lat, s.u_tran, s.v_tran)
        np.testing.assert_allclose(out[i, :s.T], ref, rtol=1e-12)


def test_alpha_net_range():
    net = AlphaNet(6)
    a = net(torch.randn(100, 6) * 10)
    assert torch.all(a >= ALPHA_MIN) and torch.all(a <= ALPHA_MAX)


def test_hard_initial_conditions(batch):
    storms, b = batch
    params = LatentParams(b, b.features.shape[-1])
    cal = PINNCalibrator(b, params, n_fourier=8, n_colloc=64)
    V, m = cal.nodes()
    torch.testing.assert_close(V[:, 0], b.V0)
    torch.testing.assert_close(m[:, 0], params.m0())


def test_losses_are_finite_and_differentiable(batch):
    storms, b = batch
    params = LatentParams(b, b.features.shape[-1])
    for cal in (PINNCalibrator(b, params, n_fourier=8, n_colloc=64, causal_eps=10.0), ShootingCalibrator(b, params)):
        total, terms = cal.losses(Weights())
        total.backward()
        assert torch.isfinite(total)
        assert params.rho.grad is not None and torch.all(torch.isfinite(params.rho.grad))
