#!/usr/bin/env python3
"""Add the along-track ocean inputs o(t) used by the alpha network.

The hourly FAST inputs in ``data/storms/<season>/*.csv`` (extracted from the FAST
training data, identical to the EAS 6130 repository) already contain FAST's
physics-prior alpha. That alpha follows the FAST ocean-coupling formula

    z = 0.01 strat^-0.4 * h_m * u_T * Vp / V,   alpha = 1 - 0.87 exp(-z)

("ODEFAST Parameter Calculation", Sec. 3.7). This script adds the ocean and
storm-motion quantities that formula is built from, from the same monthly
climatologies FAST uses (FHLO ``precalc_data``):

    hm     mixed-layer depth (m)            mld_climatology.nc, month of the time
    strat  sub-mixed-layer stratification   strat_climatology.nc
    bathy  ocean depth (m, negative)        bathymetry.nc
    u_T    translation speed (m/s)          great-circle distance per hour,
                                            centred 3-point rolling mean

Usage::

    python scripts/add_ocean_inputs.py --fhlo <path to an FHLO clone> --data data/storms/2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
R_EARTH_M = 6.3781e6


def translation_speed(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    d = 2 * R_EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / 3600.0
    s = np.concatenate([[d[0]], d])
    return pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fhlo", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "storms" / "2017")
    args = ap.parse_args()
    pre = args.fhlo / "precalc_data"
    mld = xr.open_dataset(pre / "mld_climatology.nc")["mixed_layer"]
    strat = xr.open_dataset(pre / "strat_climatology.nc")["strat"]
    bathy = xr.open_dataset(pre / "bathymetry.nc")["bathymetry"]
    for f in sorted(args.data.glob("*.csv")):
        df = pd.read_csv(f)
        months = pd.to_datetime(df["time"]).dt.month.to_numpy()
        lat = df["lat"].to_numpy(float)
        lon = np.mod(df["lon"].to_numpy(float), 360.0)
        pts = dict(lat=xr.DataArray(lat, dims="p"), lon=xr.DataArray(lon, dims="p"))
        mon = xr.DataArray(months, dims="p")
        df["hm"] = mld.sel(month=mon, **pts, method="nearest").values
        df["strat"] = strat.sel(month=mon, **pts, method="nearest").values
        df["bathy"] = bathy.sel(**pts, method="nearest").values
        df["u_T"] = translation_speed(lat, lon)
        df.to_csv(f, index=False, float_format="%.7g")
        print(f"{f.stem:40s} hm {np.nanmedian(df.hm):5.1f} m  strat {np.nanmedian(df.strat):.3f}  "
              f"u_T {np.nanmedian(df.u_T):4.1f} m/s")


if __name__ == "__main__":
    main()
