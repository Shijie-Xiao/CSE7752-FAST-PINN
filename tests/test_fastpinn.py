"""Tests: data, torch FAST physics vs NumPy FAST, networks, losses, and the no-leakage forecast protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_default_dtype(torch.float64)

from fast.fast_equations import calibrate_chi, rhs  # noqa: E402
from fast.wind_conversion import axi_to_max_wind  # noqa: E402
from fastpinn.calibrate import LatentParams, PINNCalibrator, ShootingCalibrator, Weights  # noqa: E402
from fastpinn.config import ALPHA_MAX, ALPHA_MIN, KAPPA  # noqa: E402
from fastpinn.data import FeatureScaler, load_storms, make_batch  # noqa: E402
from fastpinn.forecast import assimilate_m0, truncate  # noqa: E402
from fastpinn.networks import AlphaNet  # noqa: E402
from fastpinn.physics import obs_operator, rk4_integrate  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    storms = load_storms(2017, names=["IRMA", "BRET", "LEE"])
    return storms, make_batch(storms, FeatureScaler(storms))


def test_gamma_reparameterisation_reproduces_data(batch):
    storms, b = batch
    for i, s in enumerate(storms):
        df = pd.read_csv(s.meta["path"])
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


@pytest.mark.parametrize("feature_set", ["climatology", "oras5", "none"])
def test_feature_sets(feature_set):
    from fastpinn.config import FEATURE_SETS
    s = load_storms(2017, names=["IRMA"], feature_set=feature_set)[0]
    assert s.features.shape == (s.T, len(FEATURE_SETS[feature_set]))
    assert np.all(np.isfinite(s.features))
    if feature_set == "oras5":                     # physical ranges of the ORAS5 inputs
        sst, t100, d26, tchp = s.features[:, 0], s.features[:, 2], s.features[:, 3], s.features[:, 4]
        assert 15 < np.median(sst) < 33 and np.all(t100 <= sst + 0.5)
        assert np.all(d26 >= 0) and np.all(tchp >= 0)


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


def test_chi_calibration():
    """FAST's tc_risk calibration: exp(log(chi + 1e-3) + 0.5) + 1.3, capped at 5."""
    chi = np.array([0.0, 0.2, 1.0, 10.0])
    np.testing.assert_allclose(calibrate_chi(chi), np.minimum((chi + 1e-3) * np.exp(0.5) + 1.3, 5.0), rtol=1e-6)
    np.testing.assert_allclose(calibrate_chi(chi, "raw"), chi)


def test_both_seasons_load():
    for season, n in [(2017, 18), (2024, 10)]:
        storms = load_storms(season)
        assert len(storms) == n
        assert all(np.all(np.isfinite(s.features)) for s in storms)


def test_constant_alpha_feature_set():
    storms = load_storms(2017, names=["IRMA", "LEE"], feature_set="none")
    b = make_batch(storms, FeatureScaler(storms))
    a = AlphaNet(b.features.shape[-1])(b.features)
    assert torch.allclose(a, a.flatten()[0].expand_as(a))


def test_physics_only_pinn_ignores_observations(batch):
    """With w_obs = 0 the PINN solver cannot see IBTrACS: replacing it by noise changes nothing."""
    import dataclasses
    from fastpinn.calibrate import train_pinn
    from fastpinn.forecast import seed_all
    storms, b = batch
    out = []
    for bb in (b, dataclasses.replace(b, v_obs=torch.rand(b.v_obs.shape) * 80)):
        seed_all(0)
        params = LatentParams(bb, bb.features.shape[-1], learn_m0=False, learn_alpha=False)
        cal = PINNCalibrator(bb, params, n_fourier=8, n_colloc=64)
        train_pinn(cal, Weights(obs=0.0, prior=0.0), iters=20, curriculum=0.5)
        out.append(cal.nodes()[0].detach())
    torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)


def test_m0_from_first_24h_uses_only_first_24h(batch):
    """The forecast protocol: changing observations after hour 24 must not change m0."""
    import dataclasses
    storms, b = batch
    altered = [dataclasses.replace(s, v_obs=np.where(np.arange(s.T) > 24, 1.0, s.v_obs)) for s in storms]
    sc = FeatureScaler(storms)
    m_a = assimilate_m0(storms, sc, b.alpha_prior, iters=10)
    m_b = assimilate_m0(altered, sc, b.alpha_prior, iters=10)
    torch.testing.assert_close(m_a, m_b, rtol=0, atol=0)
    assert truncate(storms[0], 25).T == 25


def test_splits_never_train_on_a_test_storm():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from common import SPLITS, cv_folds, load_split, season_of
    for season, n in ((2017, 18), (2024, 10)):
        names = [x for f in cv_folds(season) for x in f]
        assert len(names) == len(set(names)) == n
    for split in SPLITS:
        train, test, _ = load_split(split)
        assert not {(s.meta["season"], s.name) for s in train} & {(s.meta["season"], s.name) for s in test}
        assert {s.meta["season"] for s in test} == {season_of(split)}
