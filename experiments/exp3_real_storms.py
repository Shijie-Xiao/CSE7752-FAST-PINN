#!/usr/bin/env python3
"""Experiment 3 -- calibration on real storms and hold-out prediction.

Training storms (13, 2017): estimate the per-storm initial moisture m0 and the
shared ocean-coupling network alpha(t) = g_phi(o(t)) from the 6-hourly
IBTrACS intensities, then pass them to the FAST ODE solver (RK4, 15 min) and
score against IBTrACS (in-sample):

* prior            FAST's own m0 and alpha (the physics-prior baseline)
* PINN joint       m0 and alpha together, w_phys = 1 (also w_phys = 0.1, 10)
* PINN m0 only / PINN alpha only   (compensation analysis, Reviewer 4)
* shooting joint   the same parameters fitted through differentiable RK4

Hold-out storms (5): the trained alpha network is applied to unseen tracks.
m0 is either FAST's prior or assimilated from the first 24 h of observations
(m0 only, alpha frozen). Scores use the forecast window t > 24 h.

Metrics (identical procedure for model and observations, 6-hourly times):
RMSE, bias, peak-intensity error, peak-timing error, along-track location of
the peak, bound violations.

Outputs: results/exp3_train.csv, results/exp3_test.csv, results/exp3_params.csv,
results/exp3_summary.csv, figures/fig3_real_storms.png
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
import torch

from common import C, FIGURES, INK, MUTED, PINN_KW, RESULTS, plt_style, seed_all, setup_torch, split
from fastpinn.calibrate import (LatentParams, PINNCalibrator, ShootingCalibrator, Weights,
                                pretrain_alpha_to_prior, train_pinn, train_shooting)
from fastpinn.config import MS_TO_KNOTS
from fastpinn.data import make_batch
from fastpinn.evaluate import solve_fast, storm_metrics, summarize

PINN_ITERS = 4000
SHOOT_ITERS = 300
ASSIM_H = 24
SHOW = "IRMA"


def fit(method, b, learn_m0=True, learn_alpha=True, w_phys=1.0):
    seed_all(0)
    params = LatentParams(b, b.features.shape[-1], learn_m0=learn_m0, learn_alpha=learn_alpha)
    pretrain_alpha_to_prior(params.alpha_net, b)
    w = Weights(phys=w_phys)
    if method == "pinn":
        hist = train_pinn(PINNCalibrator(b, params, **PINN_KW), w, iters=PINN_ITERS, curriculum=0.5)
    else:
        hist = train_shooting(ShootingCalibrator(b, params), w, iters=SHOOT_ITERS)
    return params, hist.wall


def truncate(storm, n):
    """First n hours of a storm (for assimilating m0 from early observations)."""
    arrays = {f.name: getattr(storm, f.name)[:n] for f in dataclasses.fields(storm)
              if isinstance(getattr(storm, f.name), np.ndarray) and f.name != "features"}
    return dataclasses.replace(storm, T=n, features=storm.features[:n], **arrays)


def assimilate_m0(storms, scaler, alpha_net=None, hours=ASSIM_H):
    """m0 from the observations of the first ``hours`` h, alpha frozen."""
    bs = make_batch([truncate(s, hours + 1) for s in storms], scaler)
    with torch.no_grad():
        a = alpha_net(bs.features) if alpha_net is not None else bs.alpha_prior
    params = LatentParams(bs, bs.features.shape[-1], learn_m0=True, learn_alpha=False, alpha_fixed=a)
    train_shooting(ShootingCalibrator(bs, params), Weights(), iters=150, lr_m0=5e-2)
    with torch.no_grad():
        return params.m0().clone()


def evaluate(b, storms, m0, alpha, label, t_from=0):
    V, m, vmax = solve_fast(b, m0, alpha)
    return storm_metrics(b, vmax, V, m, label, storms=storms, t_from=t_from), vmax


def main():
    setup_torch()
    train, test, scaler = split()
    bt = make_batch(train, scaler)
    bs = make_batch(test, scaler)

    runs = {
        "prior (FAST)": None,
        "PINN joint": ("pinn", dict()),
        "PINN joint, w_phys = 0.1": ("pinn", dict(w_phys=0.1)),
        "PINN joint, w_phys = 10": ("pinn", dict(w_phys=10.0)),
        "PINN m0 only": ("pinn", dict(learn_alpha=False)),
        "PINN alpha only": ("pinn", dict(learn_m0=False)),
        "shooting joint": ("shooting", dict()),
    }
    train_rows, param_rows, fitted, walls = [], [], {}, {}
    for label, spec in runs.items():
        tic = time.perf_counter()
        if spec is None:
            m0, alpha, net, wall = bt.m0_prior, bt.alpha_prior, None, 0.0
        else:
            params, wall = fit(spec[0], bt, **spec[1])
            with torch.no_grad():
                m0, alpha = params.m0().clone(), params.alpha(bt).clone()
            net = params.alpha_net if params.learn_alpha else None
        fitted[label] = (m0, alpha, net)
        walls[label] = wall
        met, _ = evaluate(bt, train, m0, alpha, label)
        met["calib_wall_s"] = wall
        train_rows.append(met)
        for i, s in enumerate(train):
            L = s.T
            param_rows.append({"method": label, "storm": s.name, "m0": float(m0[i]),
                               "m0_prior": float(bt.m0_prior[i]),
                               "alpha_mean": float(alpha[i, :L].mean()),
                               "alpha_prior_mean": float(bt.alpha_prior[i, :L].mean())})
        print(f"{label:28s} RMSE {met.rmse_kt.mean():6.2f} kt   ({wall:5.0f} s, {time.perf_counter() - tic:5.0f} s total)",
              flush=True)
    tr = pd.concat(train_rows, ignore_index=True)
    tr.to_csv(RESULTS / "exp3_train.csv", index=False)
    pd.DataFrame(param_rows).to_csv(RESULTS / "exp3_params.csv", index=False)

    # ---- hold-out storms --------------------------------------------------
    test_rows, series = [], {}
    m0_prior_assim = assimilate_m0(test, scaler)
    variants = [("prior (FAST)", bs.m0_prior, bs.alpha_prior),
                ("prior + m0 from first 24 h", m0_prior_assim, bs.alpha_prior)]
    for label in ("PINN joint", "shooting joint"):
        net = fitted[label][2]
        with torch.no_grad():
            a = net(bs.features)
        variants.append((f"{label} α-net, prior m0", bs.m0_prior, a))
        variants.append((f"{label} α-net + m0 from first 24 h", assimilate_m0(test, scaler, net), a))
    for label, m0, a in variants:
        met, vmax = evaluate(bs, test, m0, a, label, t_from=ASSIM_H + 1)
        test_rows.append(met)
        series[label] = vmax
    te = pd.concat(test_rows, ignore_index=True)
    te.to_csv(RESULTS / "exp3_test.csv", index=False)

    s_tr = summarize(tr).assign(split="train (in-sample)")
    s_te = summarize(te).assign(split=f"test (t > {ASSIM_H} h)")
    s_tr["calib_wall_s"] = s_tr.index.map(walls)
    summ = pd.concat([s_tr, s_te])
    summ.to_csv(RESULTS / "exp3_summary.csv")
    print(summ.to_string(float_format=lambda x: f"{x:.2f}"))

    i = bs.names.index(SHOW)
    st = test[i]
    pd.DataFrame({
        "hour": np.arange(st.T), "vp_kt": st.vp * MS_TO_KNOTS, "obs_kt": st.v_obs * MS_TO_KNOTS,
        "prior_kt": series["prior (FAST)"][i, :st.T] * MS_TO_KNOTS,
        "pinn_kt": series["PINN joint α-net + m0 from first 24 h"][i, :st.T] * MS_TO_KNOTS,
        "shooting_kt": series["shooting joint α-net + m0 from first 24 h"][i, :st.T] * MS_TO_KNOTS,
    }).to_csv(RESULTS / f"exp3_series_{SHOW}.csv", index=False)
    figure()


TRAIN_ORDER = ["prior (FAST)", "PINN m0 only", "PINN alpha only", "PINN joint", "PINN joint, w_phys = 0.1",
               "PINN joint, w_phys = 10", "shooting joint"]
TEST_ORDER = ["prior (FAST)", "prior + m0 from first 24 h", "PINN joint α-net, prior m0",
              "PINN joint α-net + m0 from first 24 h", "shooting joint α-net, prior m0",
              "shooting joint α-net + m0 from first 24 h"]


def figure():
    """Draw fig3 from the saved CSVs (no recomputation)."""
    summ = pd.read_csv(RESULTS / "exp3_summary.csv", index_col=0)
    ser = pd.read_csv(RESULTS / f"exp3_series_{SHOW}.csv")
    par = pd.read_csv(RESULTS / "exp3_params.csv")
    plt = plt_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))

    def bars(ax, s, order, title):
        s = s.reindex(order)
        y = np.arange(len(s))[::-1]
        cols = [C["prior"] if k.startswith("prior") else C["shooting"] if "shooting" in k else C["pinn"]
                for k in s.index]
        ax.barh(y, s.rmse_kt, color=cols, height=0.6)
        for yi, v in zip(y, s.rmse_kt):
            ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [k.replace(" from first 24 h", " (first 24 h)") for k in s.index], fontsize=8.5)
        ax.set(xlabel="RMSE vs IBTrACS (kt), mean over storms", title=title)
        ax.grid(axis="y", visible=False)

    bars(axes[0, 0], summ[summ.split.str.startswith("train")], TRAIN_ORDER,
         "(a) Training storms (13): in-sample calibration")
    bars(axes[0, 1], summ[summ.split.str.startswith("test")], TEST_ORDER,
         f"(b) Hold-out storms (5): forecast window t > {ASSIM_H} h")

    ax = axes[1, 0]
    ax.plot(ser.hour, ser.vp_kt, color="red", lw=1.0, label="Vp")
    ax.plot(ser.hour, ser.obs_kt, color="black", lw=1.8, label="IBTrACS")
    ax.plot(ser.hour, ser.prior_kt, color=C["prior"], lw=1.5, label="FAST prior")
    ax.plot(ser.hour, ser.pinn_kt, color=C["pinn"], lw=1.5, label="PINN α-net + m0 (first 24 h)")
    ax.plot(ser.hour, ser.shooting_kt, color=C["shooting"], lw=1.5, ls="--", label="shooting α-net + m0 (first 24 h)")
    ax.axvline(ASSIM_H, color=MUTED, lw=1, ls=":")
    ax.set(xlabel="hour", ylabel="V$_{max}$ (kt)", ylim=(0, 200),
           title=f"(c) Hold-out storm {SHOW.title()} (never seen in training)")
    ax.legend(loc="lower center", fontsize=8, ncol=2)

    ax = axes[1, 1]
    for label, mk in [("PINN joint", "o"), ("shooting joint", "s")]:
        g = par[par.method == label]
        ax.plot(g.m0_prior, g.m0, ls="none", marker=mk, ms=7,
                color=C["pinn"] if "PINN" in label else C["shooting"], label=label)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="FAST prior m0", ylabel="calibrated m0",
           title="(d) Calibrated m0 on the 13 training storms")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_real_storms.png")
    print("saved", FIGURES / "fig3_real_storms.png")


if __name__ == "__main__":
    import sys
    figure() if "--figure" in sys.argv else main()
