"""Evaluate calibrated parameters by running the FAST ODE solver.

After calibration, the inferred m0 and the learned alpha(t) are passed to the
FAST ODE solver (RK4, dt = 15 min, no clipping) with the remaining forcing
held fixed. The modelled maximum wind (observation operator applied) is
compared with IBTrACS at the 6-hourly observation times:

* trajectory RMSE and bias (kt)
* peak-intensity error (kt) and peak-timing error (h); peaks are identified
  identically for model and observations: the maximum over the 6-hourly
  observation times inside the storm's window
* along-track location error of the predicted peak (km), relative to the
  location of the observed peak on the same prescribed track
* physical-bound violations of the ODE solution (V < 0, m outside [0, 1])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import MS_TO_KNOTS
from .data import Batch
from .physics import obs_operator, rk4_integrate

EARTH_R_KM = 6371.0


def _gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@torch.no_grad()
def solve_fast(batch: Batch, m0, alpha, substeps=4):
    V, m = rk4_integrate(batch, torch.as_tensor(m0, dtype=batch.V0.dtype),
                         torch.as_tensor(alpha, dtype=batch.V0.dtype), substeps=substeps)
    vmax = obs_operator(V, batch.lat, batch.u_tran, batch.v_tran)
    return V.numpy(), m.numpy(), vmax.numpy()


def storm_metrics(batch: Batch, vmax, V, m, label: str, storms=None, t_from: float = 0.0):
    """Per-storm metrics; ``t_from`` restricts scoring to t >= t_from (forecast window)."""
    rows = []
    for i, name in enumerate(batch.names):
        L = int(batch.lengths[i])
        obs = batch.v_obs[i, :L].numpy()
        idx = np.where(batch.obs_mask[i, :L].numpy())[0]
        idx = idx[idx >= t_from]
        if len(idx) == 0:
            continue
        err = (vmax[i, idx] - obs[idx]) * MS_TO_KNOTS
        ip, io = idx[np.argmax(vmax[i, idx])], idx[np.argmax(obs[idx])]
        row = {
            "storm": name, "method": label, "n_obs": len(idx),
            "rmse_kt": float(np.sqrt(np.mean(err ** 2))), "bias_kt": float(np.mean(err)),
            "peak_err_kt": float((vmax[i, ip] - obs[io]) * MS_TO_KNOTS),
            "peak_timing_h": float(ip - io),
            "V_min": float(V[i, :L].min()), "m_min": float(m[i, :L].min()), "m_max": float(m[i, :L].max()),
        }
        row["bound_violation"] = bool(row["V_min"] < 0 or row["m_min"] < 0 or row["m_max"] > 1)
        if storms is not None:
            s = storms[i]
            row["peak_loc_err_km"] = float(_gc_km(s.lat[ip], s.lon[ip], s.lat[io], s.lon[io]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by="method"):
    agg = df.groupby(by).agg(
        storms=("storm", "count"), rmse_kt=("rmse_kt", "mean"), bias_kt=("bias_kt", "mean"),
        abs_peak_err_kt=("peak_err_kt", lambda x: np.mean(np.abs(x))),
        abs_peak_timing_h=("peak_timing_h", lambda x: np.mean(np.abs(x))),
        violations=("bound_violation", "sum"))
    return agg
