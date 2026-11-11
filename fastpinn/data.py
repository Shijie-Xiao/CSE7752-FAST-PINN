"""Storm data: FAST forcing, ocean inputs o(t) and IBTrACS observations.

``data/storms/<season>/<storm id>.csv`` holds the hourly FAST inputs (alpha,
beta, gamma, Vp, chi, S, track, IBTrACS wind; see the EAS 6130 repository)
plus the ocean inputs added by ``scripts/add_ocean_inputs.py`` (and, for
2017, ``scripts/add_oras5_inputs.py``).

Re-parameterising alpha. FAST's data give beta = 1 - eps - k and
gamma = eps + alpha k, so beta + gamma = 1 - k (1 - alpha) and
k = (1 - beta - gamma) / (1 - alpha) wherever alpha < 1. k (~0.12) is taken
from the nearest hour where it is identifiable (alpha < 0.95); over land the
data have beta = 1, gamma = 0, i.e. eps = k = 0. When the network changes
alpha, gamma(alpha) = eps + k alpha stays consistent with FAST, and
gamma(alpha_data) reproduces the data exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from fast.fast_equations import initial_moisture, ventilation
from fast.wind_conversion import invert_vmax_to_axisym, translation_velocity

from .config import (CHI_CALIBRATION, DATA_DIR, DEFAULT_FEATURES, FEATURE_SETS, KAPPA, OBS_EVERY_H,
                     TRAIN_SEASON, V_STAR)


@dataclass
class Storm:
    name: str
    T: int
    vp: np.ndarray
    beta: np.ndarray          # FAST's beta (does not depend on alpha)
    xs: np.ndarray            # chi * S (FAST floor 1e-5)
    eps: np.ndarray
    k: np.ndarray
    alpha_prior: np.ndarray   # FAST's alpha (physics prior)
    features: np.ndarray      # (T, F) raw ocean inputs o(t)
    lat: np.ndarray
    lon: np.ndarray
    u_tran: np.ndarray
    v_tran: np.ndarray
    v_obs: np.ndarray         # IBTrACS maximum wind, m/s (hourly-interpolated)
    V0: float                 # first observation inverted to the azimuthal-mean wind (fixed)
    m0_prior: float           # FAST's initial-moisture inversion (physics prior)
    meta: dict = field(default_factory=dict)

    @property
    def obs_idx(self):
        idx = np.arange(OBS_EVERY_H, self.T, OBS_EVERY_H)
        return idx[np.isfinite(self.v_obs[idx])]


def thermo_split(alpha, beta, gamma):
    """Recover (eps, k) per hour from FAST's (alpha, beta, gamma)."""
    k = np.full(len(alpha), np.nan)
    ok = alpha < 0.95
    k[ok] = (1.0 - beta[ok] - gamma[ok]) / (1.0 - alpha[ok])
    land = (beta >= 0.999) & (gamma <= 1e-3)
    if np.isfinite(k).any():
        idx = np.where(np.isfinite(k))[0]
        fill = np.searchsorted(idx, np.arange(len(k)))
        near = idx[np.clip(fill, 0, len(idx) - 1)]
        prev = idx[np.clip(fill - 1, 0, len(idx) - 1)]
        choose = np.where(np.abs(near - np.arange(len(k))) <= np.abs(prev - np.arange(len(k))), near, prev)
        k = np.where(np.isfinite(k), k, k[choose])
    else:
        k = np.full(len(alpha), 0.118)          # median over all storms
    k = np.clip(np.where(land, 0.0, k), 0.0, None)
    eps = gamma - alpha * k          # exact: gamma(alpha_data) = gamma_data (land: k = 0, eps = gamma)
    return eps, k


def _features(df, feature_set=DEFAULT_FEATURES):
    """Ocean / storm-motion inputs o(t) for the chosen feature set (see config.FEATURE_SETS)."""
    hm = np.nan_to_num(df["hm"].to_numpy(float), nan=30.0)
    st = np.nan_to_num(df["strat"].to_numpy(float), nan=0.1)
    depth = -np.nan_to_num(df["bathy"].to_numpy(float), nan=-100.0)
    shallow = ((depth <= hm) | (depth <= 0)).astype(float)
    cols = {"log_hm": lambda: np.log(np.clip(hm, 1.0, None)),
            "log_strat": lambda: np.log(np.clip(st, 1e-2, None)),
            "log_depth": lambda: np.log1p(np.clip(depth, 0.0, None)),
            "u_T": lambda: df["u_T"].to_numpy(float),
            "vp_norm": lambda: df["vp"].to_numpy(float) / V_STAR,
            "shallow": lambda: shallow,
            "o5_sst": lambda: df["o5_sst"].to_numpy(float),
            "log_o5_mld": lambda: np.log(np.clip(df["o5_mld"].to_numpy(float), 1.0, None)),
            "o5_t100": lambda: df["o5_t100"].to_numpy(float),
            "o5_d26": lambda: df["o5_d26"].to_numpy(float),
            "o5_tchp": lambda: df["o5_tchp"].to_numpy(float),
            "zero": lambda: np.zeros(len(df))}
    names = FEATURE_SETS[feature_set]
    need = {c.replace("log_", "") for c in names if "o5_" in c}
    missing = sorted(need - set(df.columns))
    if missing:
        raise KeyError(f"{missing} not in the storm file: run scripts/add_oras5_inputs.py first")
    return np.column_stack([cols[c]() for c in names])


def load_storm(path, feature_set=DEFAULT_FEATURES, chi_calibration=CHI_CALIBRATION) -> Storm:
    df = pd.read_csv(path)
    col = lambda c: df[c].to_numpy(float)  # noqa: E731
    alpha, beta, gamma, vp = col("alpha"), col("beta"), col("gamma"), col("vp")
    xs = ventilation(col("chi"), col("S"), chi_calibration)
    eps, k = thermo_split(alpha, beta, gamma)
    lat, lon = col("lat"), col("lon")
    ut, vt = translation_velocity(lat, lon)
    v_obs = col("v_obs")
    V0 = invert_vmax_to_axisym(v_obs[0], lat[0], ut[0], vt[0])
    V1 = invert_vmax_to_axisym(v_obs[1], lat[1], ut[1], vt[1])
    F0 = (KAPPA, alpha[0], beta[0], gamma[0], vp[0], xs[0])
    return Storm(name=path.stem.split("_")[-1], T=len(df), vp=vp, beta=beta, xs=xs, eps=eps, k=k,
                 alpha_prior=alpha, features=_features(df, feature_set), lat=lat, lon=lon, u_tran=ut, v_tran=vt,
                 v_obs=v_obs, V0=float(V0), m0_prior=initial_moisture(V0, V1 - V0, F0),
                 meta={"storm_id": path.stem, "season": int(path.parent.name), "path": str(path),
                       "start": str(df["time"].iloc[0])})


def season_dir(season=TRAIN_SEASON):
    return DATA_DIR / str(season)


def available_storms(season=TRAIN_SEASON):
    return sorted(p.stem.split("_")[-1] for p in season_dir(season).glob("*.csv"))


def load_storms(season=TRAIN_SEASON, names=None, feature_set=DEFAULT_FEATURES, chi_calibration=CHI_CALIBRATION):
    paths = sorted(season_dir(season).glob("*.csv"))
    if names is not None:
        paths = [p for p in paths if p.stem.split("_")[-1] in set(names)]
    return [load_storm(p, feature_set, chi_calibration) for p in paths]


class FeatureScaler:
    """Standardise ocean inputs with statistics of the training storms only."""

    def __init__(self, storms):
        X = np.concatenate([s.features for s in storms])
        self.mean = X.mean(0)
        self.std = np.where(X.std(0) > 1e-8, X.std(0), 1.0)

    def __call__(self, X):
        return (X - self.mean) / self.std


@dataclass
class Batch:
    """Several storms padded to a common length (edge values) for vectorised work."""
    names: list
    lengths: torch.Tensor
    vp: torch.Tensor          # (B, T)
    beta: torch.Tensor
    xs: torch.Tensor
    eps: torch.Tensor
    k: torch.Tensor
    alpha_prior: torch.Tensor
    features: torch.Tensor    # (B, T, F)
    lat: torch.Tensor
    u_tran: torch.Tensor
    v_tran: torch.Tensor
    v_obs: torch.Tensor       # (B, T), NaN where missing
    obs_mask: torch.Tensor    # (B, T) True at the 6-hourly IBTrACS times
    valid: torch.Tensor       # (B, T) True inside each storm
    V0: torch.Tensor          # (B,)
    m0_prior: torch.Tensor    # (B,)

    @property
    def B(self):
        return len(self.names)

    def gamma(self, alpha):
        """gamma(alpha) = eps + k alpha  (= the data's gamma at alpha = alpha_prior)."""
        return self.eps + self.k * alpha


def make_batch(storms, scaler: FeatureScaler, dtype=torch.float64) -> Batch:
    Tmax = max(s.T for s in storms)

    def pad(arrs, fill="edge"):
        out = np.empty((len(arrs), Tmax) + np.shape(arrs[0])[1:])
        for i, a in enumerate(arrs):
            out[i, : len(a)] = a
            out[i, len(a):] = a[-1] if fill == "edge" else fill
        return torch.as_tensor(out, dtype=dtype)

    obs_mask = np.zeros((len(storms), Tmax), bool)
    valid = np.zeros((len(storms), Tmax), bool)
    for i, s in enumerate(storms):
        obs_mask[i, s.obs_idx] = True
        valid[i, : s.T] = True
    return Batch(
        names=[s.name for s in storms], lengths=torch.tensor([s.T for s in storms]),
        vp=pad([s.vp for s in storms]), beta=pad([s.beta for s in storms]), xs=pad([s.xs for s in storms]),
        eps=pad([s.eps for s in storms]), k=pad([s.k for s in storms]),
        alpha_prior=pad([s.alpha_prior for s in storms]),
        features=pad([scaler(s.features) for s in storms]),
        lat=pad([s.lat for s in storms]), u_tran=pad([s.u_tran for s in storms]),
        v_tran=pad([s.v_tran for s in storms]), v_obs=pad([s.v_obs for s in storms], fill=np.nan),
        obs_mask=torch.as_tensor(obs_mask), valid=torch.as_tensor(valid),
        V0=torch.tensor([s.V0 for s in storms], dtype=dtype),
        m0_prior=torch.tensor([s.m0_prior for s in storms], dtype=dtype),
    )
