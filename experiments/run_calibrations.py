#!/usr/bin/env python3
"""Fit m0 and alpha on every train/test split for every training variant (the expensive step).

Splits (experiments/common.py), grouped into four evaluations:
  cv2017-1 ... cv2017-6   2017 cross-validation: train on 15 storms, forecast the other 3
  cv2024-1 ... cv2024-5   2024 cross-validation: train on 8 hurricanes, forecast the other 2
  2017to2024              train on all 18 storms of 2017, forecast the 10 hurricanes of 2024
  2024to2017              train on the 10 hurricanes of 2024, forecast the 18 storms of 2017

Variants: what is learned in training and how (VARIANTS in common.py), plus
the baselines "fast" (FAST's own alpha) and "fast_rawchi" (FAST with raw chi).

Every test storm is forecast twice per variant (fastpinn/forecast.py): with
FAST's prior m0, and with m0 estimated from its first 24 h of observations.
Only t > 24 h is scored.

Outputs (merged over all jobs):
  results/storms.csv              hourly track, Vp, FAST alpha, IBTrACS (6-hourly) of every storm
  results/forecasts.csv           hourly V_max forecast of every test storm, per split / variant / m0 source
  results/learned_m0.csv          m0 per storm: learned in training (role=train) or from the first 24 h (role=test)
  results/learned_alpha.csv       alpha(t) the network gives along every test track
  results/calibration_runs.csv    wall-clock time and in-sample RMSE of every fit
  results/calibrations/<split>__<variant>.pt   alpha-network weights, feature scaler, training m0

Runs the 53 fits in 12 parallel processes (about 2.5 h on 24 cores). A job whose
output exists is skipped; pass --force to recompute.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch

from common import (ASSIM_H, BASELINES, CALIB_DIR, RESULTS, SPLITS, VARIANTS, evaluation_of, jobs, load_split,
                    season_of, setup_torch)
from fastpinn.config import MS_TO_KNOTS
from fastpinn.data import load_storms, make_batch
from fastpinn.forecast import assimilate_m0, fit, forecast

CACHE = CALIB_DIR / "cache"
THREADS = 2


def score_rmse(batch, vmax_kt, t_from=0):
    out = []
    for i in range(batch.B):
        idx = np.where(batch.obs_mask[i].numpy())[0]
        idx = idx[idx > t_from]
        obs = batch.v_obs[i, idx].numpy() * MS_TO_KNOTS
        out.append(float(np.sqrt(np.mean((vmax_kt[i, idx] - obs) ** 2))))
    return out


def forecast_rows(split, variant, test, bs, scaler, alpha):
    """Forecasts with FAST's m0 and with m0 from the first 24 h, for the test storms."""
    m0 = {"prior": bs.m0_prior, "first24h": assimilate_m0(test, scaler, alpha)}
    rows, m0_rows = [], []
    for src, m in m0.items():
        vmax, _, _ = forecast(bs, m, alpha)
        for i, s in enumerate(test):
            rows.append(pd.DataFrame({"split": split, "storm": s.name, "variant": variant, "m0_source": src,
                                      "hour": np.arange(s.T), "vmax_kt": vmax[i, :s.T]}))
            m0_rows.append({"split": split, "variant": variant, "storm": s.name, "role": "test",
                            "m0_source": src, "m0": float(m[i]), "m0_fast": float(bs.m0_prior[i])})
    return rows, m0_rows


def run_job(job):
    split, variant = job
    out = CACHE / f"{split}__{variant}.pkl"
    warnings.filterwarnings("ignore")
    setup_torch(THREADS)
    v = VARIANTS[variant]
    train, test, scaler = load_split(split, v["features"], v["chi"])
    bt, bs = make_batch(train, scaler), make_batch(test, scaler)
    params, wall = fit(v["method"], bt, learn_m0=v["learn_m0"])
    with torch.no_grad():
        m0_tr, a_tr = params.m0().clone(), params.alpha(bt).clone()
        a_te = params.alpha_net(bs.features).clone()
    vmax_tr, _, _ = forecast(bt, m0_tr, a_tr)
    rows, m0_rows = forecast_rows(split, variant, test, bs, scaler, a_te)
    m0_rows += [{"split": split, "variant": variant, "storm": s.name, "role": "train", "m0_source": "learned",
                 "m0": float(m0_tr[i]), "m0_fast": float(bt.m0_prior[i])} for i, s in enumerate(train)]
    alpha_rows = [pd.DataFrame({"split": split, "variant": variant, "storm": s.name, "hour": np.arange(s.T),
                                "alpha": a_te[i, :s.T].numpy()}) for i, s in enumerate(test)]
    run = {"split": split, "variant": variant, "n_train": len(train), "n_test": len(test), "wall_s": wall,
           "train_rmse_kt": float(np.mean(score_rmse(bt, vmax_tr)))}
    torch.save({"alpha_net": params.alpha_net.state_dict(), "features": v["features"],
                "scaler_mean": scaler.mean, "scaler_std": scaler.std, "train_storms": [s.name for s in train],
                "m0_train": m0_tr.numpy()}, CALIB_DIR / f"{split}__{variant}.pt")
    with open(out, "wb") as f:
        pickle.dump({"forecasts": pd.concat(rows), "m0": pd.DataFrame(m0_rows),
                     "alpha": pd.concat(alpha_rows), "run": run}, f)
    return f"{split:11s} {variant:20s} {wall:5.0f} s   train RMSE {run['train_rmse_kt']:5.1f} kt"


def baseline_job(split, name):
    """FAST's own alpha (nothing learned)."""
    train, test, scaler = load_split(split, "climatology", BASELINES[name]["chi"])
    bs = make_batch(test, scaler)
    rows, m0_rows = forecast_rows(split, name, test, bs, scaler, bs.alpha_prior)
    return {"forecasts": pd.concat(rows), "m0": pd.DataFrame(m0_rows)}


def storm_table():
    rows = []
    for season in (2017, 2024):
        for s in load_storms(season):
            obs = np.full(s.T, np.nan)
            idx = np.arange(0, s.T, 6)
            obs[idx] = s.v_obs[idx] * MS_TO_KNOTS
            rows.append(pd.DataFrame({"season": season, "storm": s.name, "storm_id": s.meta["storm_id"],
                                      "hour": np.arange(s.T), "lat": s.lat, "lon": s.lon,
                                      "obs_kt": obs, "vp_kt": s.vp * MS_TO_KNOTS, "alpha_fast": s.alpha_prior,
                                      "scored": (np.arange(s.T) > ASSIM_H) & np.isfinite(obs)}))
    return pd.concat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if args.force or not (CACHE / f"{j[0]}__{j[1]}.pkl").exists()]
    print(f"{len(jobs())} fits, {len(todo)} to run on {args.workers} processes", flush=True)
    tic = time.perf_counter()
    if todo:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for msg in pool.imap_unordered(run_job, todo):
                print(f"[{time.perf_counter() - tic:5.0f} s] {msg}", flush=True)

    setup_torch(8)
    warnings.filterwarnings("ignore")
    parts = [pickle.load(open(CACHE / f"{s}__{v}.pkl", "rb")) for s, v in jobs()]
    parts += [baseline_job(split, name) for split in SPLITS for name in BASELINES]

    def tagged(df):
        df.insert(1, "evaluation", df.split.map(evaluation_of))
        df.insert(2, "season", df.split.map(season_of))
        return df

    tagged(pd.concat([p["forecasts"] for p in parts], ignore_index=True)).to_csv(
        RESULTS / "forecasts.csv", index=False, float_format="%.3f")
    tagged(pd.concat([p["m0"] for p in parts], ignore_index=True)).to_csv(
        RESULTS / "learned_m0.csv", index=False, float_format="%.4f")
    tagged(pd.concat([p["alpha"] for p in parts if "alpha" in p], ignore_index=True)).to_csv(
        RESULTS / "learned_alpha.csv", index=False, float_format="%.4f")
    tagged(pd.DataFrame([p["run"] for p in parts if "run" in p])).to_csv(
        RESULTS / "calibration_runs.csv", index=False, float_format="%.2f")
    storm_table().to_csv(RESULTS / "storms.csv", index=False, float_format="%.4f")
    print(f"merged into results/*.csv ({time.perf_counter() - tic:.0f} s)")


if __name__ == "__main__":
    main()
