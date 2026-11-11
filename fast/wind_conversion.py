"""Conversion between FAST's azimuthal-mean wind and the observed maximum wind.

FAST integrates the azimuthal-mean (axisymmetric) wind V. IBTrACS reports the
maximum sustained surface wind, which also contains the asymmetry from the
storm's motion (and environmental shear). FAST therefore

1. inverts the first observation to an axisymmetric V0 (bisection), and
2. converts the integrated V back to a maximum wind at the end.

Transcribed from FAST ``axi_to_max_wind`` / ``_axi_to_max_wind_single`` /
``_invert_vmax_to_V_axisym``:

    G     = min(1, 0.8 + 0.35 (1 + tanh((|lat| - 35) / 10)))      translation factor
    U_inc = G * (u_tran, v_tran) + 0.1 * S * V / 15 * e_shear       asymmetry vector
    fac   = min(1, 0.5 V / |U_inc|)                                  asymmetry <= V/2
    V_max = | V e_theta + fac U_inc | at the azimuth theta where they align

The dataset carries no environmental winds, so the shear direction e_shear
is unknown; FAST then sets the shear term to zero (``has_env = False``) and
only the translation asymmetry remains. The storm translation velocity is
computed from the hourly track as in the parameter notes: great-circle
displacement per hour, centred 3-point rolling mean.
"""

from __future__ import annotations

import numpy as np

R_EARTH_M = 6.3781e6


def translation_velocity(lat, lon, dt_hours=1.0):
    """(u_tran, v_tran) in m/s from hourly positions (east, north components)."""
    la = np.radians(np.asarray(lat, float))
    lo = np.radians(np.asarray(lon, float))
    dla = np.gradient(la)
    dlo = np.gradient(np.unwrap(lo))
    u = R_EARTH_M * np.cos(la) * dlo / (dt_hours * 3600.0)
    v = R_EARTH_M * dla / (dt_hours * 3600.0)
    kernel = np.ones(3) / 3.0
    pad = lambda x: np.convolve(np.pad(x, 1, mode="edge"), kernel, mode="valid")  # noqa: E731
    return pad(u), pad(v)


def _asymmetry(v_axi, lat, ut, vt, s=0.0, shear_dir=(0.0, 0.0)):
    G = np.minimum(1.0, 0.8 + 0.35 * (1.0 + np.tanh((np.abs(lat) - 35.0) / 10.0)))
    sc = 0.1 * np.nan_to_num(s) * v_axi / 15.0
    U = G * ut + sc * shear_dir[0]
    W = G * vt + sc * shear_dir[1]
    return U, W


def axi_to_max_wind(v_axi, lat, ut, vt, s=0.0, shear_dir=(0.0, 0.0)):
    """V_axisym -> V_max (vectorised; FAST ``axi_to_max_wind``)."""
    v_axi = np.asarray(v_axi, float)
    U, W = _asymmetry(v_axi, lat, ut, vt, s, shear_dir)
    mag = np.sqrt(U ** 2 + W ** 2 + 1e-12)
    fac = np.minimum(1.0, (v_axi * 0.5) / mag)
    th = np.arctan2(-U, W)
    ug = v_axi * (-np.sin(th)) + U * fac
    vg = v_axi * np.cos(th) + W * fac
    return np.sqrt(ug ** 2 + vg ** 2 + 1e-12)


def invert_vmax_to_axisym(v_max_obs, lat, ut, vt, s=0.0, tol=0.01, max_iter=50):
    """Bisection inverse of :func:`axi_to_max_wind` (FAST ``_invert_vmax_to_V_axisym``)."""
    if not np.isfinite(v_max_obs) or v_max_obs <= 0:
        return np.nan
    lo, hi = 0.0, 200.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        vm = float(axi_to_max_wind(mid, lat, ut, vt, s))
        if abs(vm - v_max_obs) < tol:
            return mid
        lo, hi = (mid, hi) if vm < v_max_obs else (lo, mid)
    return 0.5 * (lo + hi)
