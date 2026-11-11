#!/usr/bin/env python3
"""Add ORAS5 upper-ocean inputs along each storm track.

Source: ECMWF ORAS5 ocean reanalysis, monthly means, ensemble member opa0,
regridded to 1 deg (r1x1), served without registration by ICDC, Univ. Hamburg:
https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/EASYInit/oras5/catalog.html
(1979-2018). Only the 3-D potential temperature ``votemper`` (75 levels) is
needed; every quantity below is derived from it.

Per month, on the 1 deg grid:

    sst    temperature of the top level (0.5 m), deg C
    mld    mixed-layer depth: first depth where T < T(10 m) - 0.5 deg C, m
    t100   mean temperature of the top 100 m (Price 2009: the SST a storm
           can mix down to; the best single predictor of cold-wake cooling), deg C
    d26    depth of the 26 deg C isotherm, m (0 if SST < 26)
    tchp   tropical-cyclone heat potential  rho cp int_{-d26}^0 (T - 26) dz, kJ cm^-2

Land / missing cells (the r1x1 product has a coarse land mask) are filled
with the nearest ocean value; the storm's land/shallow-water flag is a
separate input. Monthly fields are placed at mid-month and interpolated
linearly in time to each storm hour; space is nearest-neighbour.

Writes columns o5_sst, o5_mld, o5_t100, o5_d26, o5_tchp into data/storms/*.csv.

Usage::

    python scripts/add_oras5_inputs.py            # downloads missing months into data/oras5/
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parent.parent
URL = ("https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/EASYInit/oras5/r1x1/"
       "votemper/opa0/votemper_ORAS5_1m_{ym}_r1x1.nc")
RHO_CP = 1025.0 * 4000.0          # J m^-3 K^-1
FIELDS = ("sst", "mld", "t100", "d26", "tchp")
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def month_file(oras5_dir: Path, ym: str, download: bool) -> Path:
    f = oras5_dir / f"votemper_ORAS5_1m_{ym}_r1x1.nc"
    if not f.exists():
        if not download:
            raise FileNotFoundError(f)
        print(f"  downloading {f.name}", flush=True)
        part = f.with_suffix(".nc.part")
        # the ICDC server is slow and drops long transfers: retry and resume
        cmd = ["curl", "-sS", "-f", "-L", "--retry", "10", "--retry-all-errors", "--retry-delay", "5",
               "-C", "-", "-o", str(part), URL.format(ym=ym)]
        for _ in range(10):
            if subprocess.run(cmd).returncode == 0:
                break
        else:
            raise RuntimeError(f"download failed: {f.name}")
        part.rename(f)
    return f


def crossing_depth(T, z, target):
    """Depth (per column) where T first falls below ``target`` (linear interpolation)."""
    below = T < target[None]
    first = np.argmax(below, axis=0)
    has = below.any(axis=0) & (first > 0)
    k1 = np.clip(first, 1, len(z) - 1)
    k0 = k1 - 1
    T0 = np.take_along_axis(T, k0[None], 0)[0]
    T1 = np.take_along_axis(T, k1[None], 0)[0]
    w = np.clip((T0 - target) / np.where(T0 - T1 == 0, np.nan, T0 - T1), 0, 1)
    d = z[k0] + w * (z[k1] - z[k0])
    return np.where(has, d, np.nan)


def derive(f: Path):
    ds = xr.open_dataset(f)
    T = ds["votemper"].isel(time_counter=0).values.astype(float)     # (75, lat, lon)
    z = ds["deptht"].values.astype(float)
    lat, lon = ds["lat"].values, ds["lon"].values
    ds.close()
    ocean = np.isfinite(T[0])
    sst = T[0]
    k10 = int(np.argmin(np.abs(z - 10.0)))
    mld = crossing_depth(T, z, T[k10] - 0.5)
    mld = np.where(ocean & np.isnan(mld), z[np.isfinite(T).sum(0).clip(1) - 1], mld)   # mixed to bottom
    # mean temperature of the top 100 m (trapezoid on the model levels + T at exactly 100 m)
    k = int(np.searchsorted(z, 100.0))
    T100m = T[k - 1] + (100.0 - z[k - 1]) / (z[k] - z[k - 1]) * (T[k] - T[k - 1])
    zz = np.concatenate([[0.0], z[:k], [100.0]])
    Tz = np.concatenate([T[:1], T[:k], T100m[None]])
    t100 = TRAPZ(Tz, zz, axis=0) / 100.0
    d26 = crossing_depth(T, z, np.full(sst.shape, 26.0))
    d26 = np.where(sst < 26.0, 0.0, d26)
    excess = np.clip(T - 26.0, 0.0, None)
    excess = np.where(np.isfinite(excess), excess, 0.0)
    zfull = np.concatenate([[0.0], z])
    exfull = np.concatenate([excess[:1], excess])
    tchp = RHO_CP * TRAPZ(exfull, zfull, axis=0) / 1e7                       # J m^-2 -> kJ cm^-2
    out = {}
    for name, a in zip(FIELDS, (sst, mld, t100, d26, tchp)):
        a = np.where(ocean, a, np.nan)
        idx = distance_transform_edt(~np.isfinite(a), return_distances=False, return_indices=True)
        out[name] = a[tuple(idx)]                                                    # nearest ocean value
    return lat, lon, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oras5", type=Path, default=REPO / "data" / "oras5")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "storms")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    args.oras5.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data.glob("*.csv"))
    times = pd.concat([pd.to_datetime(pd.read_csv(f, usecols=["time"])["time"]) for f in files])
    months = pd.period_range(times.min().to_period("M") - 1, times.max().to_period("M") + 1, freq="M")
    fields, mids = {}, []
    for p in months:
        ym = p.strftime("%Y%m")
        lat, lon, out = derive(month_file(args.oras5, ym, not args.no_download))
        fields[ym] = out
        mids.append(pd.Timestamp(p.start_time) + pd.Timedelta(days=15))
        print(f"  {ym}: T100 Gulf/Caribbean median {np.nanmedian(out['t100'][105:120, 270:300]):.1f} C, "
              f"TCHP {np.nanmedian(out['tchp'][105:120, 270:300]):.0f} kJ/cm2")
    keys = list(fields)
    mid_ns = np.array([m.value for m in mids], dtype=float)

    for f in files:
        df = pd.read_csv(f)
        t = pd.to_datetime(df["time"]).values.astype("datetime64[ns]").astype(float)
        ilat = np.clip(np.rint(df["lat"].to_numpy() - lat[0]).astype(int), 0, len(lat) - 1)
        ilon = np.clip(np.rint(np.mod(df["lon"].to_numpy(), 360.0) - lon[0]).astype(int), 0, len(lon) - 1)
        j = np.clip(np.searchsorted(mid_ns, t) - 1, 0, len(keys) - 2)
        w = np.clip((t - mid_ns[j]) / (mid_ns[j + 1] - mid_ns[j]), 0.0, 1.0)
        for name in FIELDS:
            a = np.array([fields[keys[jj]][name][ilat[n], ilon[n]] for n, jj in enumerate(j)])
            b = np.array([fields[keys[jj + 1]][name][ilat[n], ilon[n]] for n, jj in enumerate(j)])
            df[f"o5_{name}"] = (1 - w) * a + w * b
        df.to_csv(f, index=False, float_format="%.7g")
        print(f"{f.stem:40s} T100 {df.o5_t100.median():5.1f} C  MLD {df.o5_mld.median():5.1f} m  "
              f"D26 {df.o5_d26.median():5.0f} m  TCHP {df.o5_tchp.median():5.0f} kJ/cm2")


if __name__ == "__main__":
    main()
