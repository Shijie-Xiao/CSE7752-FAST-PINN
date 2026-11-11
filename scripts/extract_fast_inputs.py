#!/usr/bin/env python3
"""Extract the hourly FAST inputs from the ``*_dataset.pkl`` files into small CSVs.

Source: the FAST training data (``training_data/<STORM>/<STORM>_dataset.pkl``,
produced by ``prepare_complete_training_data.py``; see "ODEFAST Parameter
Calculation"). Every variable is hourly along the IBTrACS track:

    scalars[0, :, 0:4] = alpha, beta, gamma, Vp     (Vp in m/s; 0 over land)
    chi_ref, s_ref                                  (chi*S = ventilation)
    v_gt   = IBTrACS maximum sustained wind (m/s), hourly
    v_init = initial wind used by FAST (m/s)
    times, lats, lons

Only what FAST needs is kept (the 5x32x32 spatial fields are dropped). The
FAST result stored next to each pkl (``fast_reference.csv``, column
``v_fast_kts``) is copied as ``v_fast_ref_kts`` so that the re-implementation
can be checked against it.

Usage::

    python scripts/extract_fast_inputs.py --src "<...>/Project/FAST/training_data"
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def extract(storm_dir: Path, out_dir: Path):
    pkl = next(storm_dir.glob("*_dataset.pkl"))
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    sc = np.asarray(d["scalars"], dtype=np.float64)[0]
    T = sc.shape[0]
    chi = np.asarray(d["chi_ref"], dtype=np.float64)[0, :, 0]
    s = np.asarray(d["s_ref"], dtype=np.float64)[0, :, 0]
    times = pd.to_datetime(np.asarray(d["times"]).ravel()[:T])
    table = pd.DataFrame({
        "time": times.strftime("%Y-%m-%dT%H:%M"),
        "hour": np.arange(T),
        "lat": np.asarray(d["lats"], dtype=np.float64)[:T],
        "lon": np.asarray(d["lons"], dtype=np.float64)[:T],
        "alpha": sc[:, 0], "beta": sc[:, 1], "gamma": sc[:, 2], "vp": sc[:, 3],
        "chi": chi, "S": s,
        "v_obs": np.asarray(d["v_gt"], dtype=np.float64)[0, :, 0],
        "v_init": np.asarray(d["v_init"], dtype=np.float64)[0, :, 0],
    })
    ref = storm_dir / "fast_reference.csv"
    if ref.exists():
        table["v_fast_ref_kts"] = pd.read_csv(ref)["v_fast_kts"].to_numpy()[:T]
    name = str(d.get("hurricane", storm_dir.name))
    out = out_dir / f"{name}.csv"
    table.to_csv(out, index=False, float_format="%.7g")
    return name, T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="training_data directory with <STORM>/ folders")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "storms" / "2017")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in args.src.iterdir() if p.is_dir() and any(p.glob("*_dataset.pkl")))
    for p in dirs:
        name, T = extract(p, args.out)
        print(f"{name:40s} {T:4d} h")


if __name__ == "__main__":
    main()
