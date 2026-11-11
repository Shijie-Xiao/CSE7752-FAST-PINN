#!/usr/bin/env python3
"""Experiment 2 -- synthetic twin experiments: can m0 and alpha(t) be recovered?

Truth, on the real tracks and forcing of the 13 training storms:

* m0_true = m0_prior +/- 0.15 (alternating sign, clipped to [0.1, 0.95])
* alpha_true(t) = 1 - 0.87 exp(-0.4 z(t)), where z(t) is FAST's ocean-mixing
  parameter behind the prior alpha = 1 - 0.87 exp(-z): same ocean dependence,
  stronger cold-wake feedback, so the prior is deliberately wrong.

Synthetic observations: FAST (RK4, 15 min) with the true parameters, mapped by
FAST's observation operator (axi_to_max_wind), plus N(0, 1 m/s) noise, every
6 h -- the same model-observation relationship as for the real data.

Each estimator (PINN, RK4 shooting) is run in three modes: m0 only (alpha at
its prior), alpha only (m0 at its prior) and joint. This exposes
compensation between the two latent quantities (Reviewer 4, comments 4-7).
We also measure how quickly the influence of m0 on V fades along the track.

Outputs: results/exp2_twin.csv, results/exp2_twin_storms.csv,
results/exp2_m0_influence.csv, figures/fig2_twin.png
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch

from common import C, FIGURES, INK, MUTED, PINN_KW, RESULTS, plt_style, seed_all, setup_torch, split
from fastpinn.calibrate import (LatentParams, PINNCalibrator, ShootingCalibrator, Weights,
                                pretrain_alpha_to_prior, train_pinn, train_shooting)
from fastpinn.config import MS_TO_KNOTS
from fastpinn.data import make_batch
from fastpinn.evaluate import solve_fast
from fastpinn.physics import obs_operator, rk4_integrate

NOISE_MS = 1.0
C_TRUE = 0.4
PINN_ITERS = 4000
SHOOT_ITERS = 300
MODES = ("m0", "alpha", "joint")
SHOW_ALPHA = "LEE"


def alpha_true_from_prior(alpha_prior, c=C_TRUE):
    z = -torch.log(torch.clamp(1.0 - alpha_prior, min=1e-9) / 0.87)
    return 1.0 - 0.87 * torch.exp(-c * z)


def make_twin(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    sign = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(b.B)])
    m0_true = torch.clamp(b.m0_prior + 0.15 * sign, 0.1, 0.95)
    a_true = alpha_true_from_prior(b.alpha_prior)
    with torch.no_grad():
        V, _ = rk4_integrate(b, m0_true, a_true, substeps=4)
        vmax = obs_operator(V, b.lat, b.u_tran, b.v_tran)
    noise = NOISE_MS * torch.randn(vmax.shape, generator=g)
    v_obs = torch.where(b.obs_mask, vmax + noise, torch.full_like(vmax, float("nan")))
    return dataclasses.replace(b, v_obs=v_obs), m0_true, a_true, V


def m0_influence(b, alpha, dm=0.1):
    with torch.no_grad():
        V1, _ = rk4_integrate(b, b.m0_prior, alpha, substeps=2)
        V2, _ = rk4_integrate(b, torch.clamp(b.m0_prior + dm, max=0.99), alpha, substeps=2)
    return (V2 - V1).abs().numpy()


def calibrate(method, mode, bt):
    params = LatentParams(bt, bt.features.shape[-1], learn_m0=mode in ("m0", "joint"),
                          learn_alpha=mode in ("alpha", "joint"))
    pretrain_alpha_to_prior(params.alpha_net, bt)
    if method == "pinn":
        hist = train_pinn(PINNCalibrator(bt, params, **PINN_KW), Weights(), iters=PINN_ITERS, curriculum=0.5)
    else:
        hist = train_shooting(ShootingCalibrator(bt, params), Weights(), iters=SHOOT_ITERS)
    with torch.no_grad():
        return params.m0().clone(), params.alpha(bt).clone(), hist.wall


def main():
    setup_torch()
    seed_all(0)
    train, _, scaler = split()
    b = make_batch(train, scaler)
    bt, m0_true, a_true, V_true = make_twin(b)
    ocean = (a_true < 0.999) & b.valid
    Vt = V_true.numpy()

    def traj_rmse(V):
        return float(np.mean([np.sqrt(np.mean((V[i, :int(b.lengths[i])] - Vt[i, :int(b.lengths[i])]) ** 2))
                              for i in range(b.B)]) * MS_TO_KNOTS)

    rows, per_storm, alphas = [], [], {}
    for method in ("pinn", "shooting"):
        for mode in MODES:
            torch.manual_seed(1)
            m0_hat, a_hat, wall = calibrate(method, mode, bt)
            V, _, _ = solve_fast(bt, m0_hat, a_hat)
            for i in range(b.B):
                per_storm.append({"method": method, "mode": mode, "storm": b.names[i],
                                  "m0_true": float(m0_true[i]), "m0_hat": float(m0_hat[i]),
                                  "m0_prior": float(b.m0_prior[i])})
            rows.append({"method": method, "mode": mode,
                         "m0_mae": float((m0_hat - m0_true).abs().mean()),
                         "alpha_rmse_ocean": float(torch.sqrt(((a_hat - a_true)[ocean] ** 2).mean())),
                         "rmse_V_vs_truth_kt": traj_rmse(V), "wall_s": wall})
            alphas[(method, mode)] = a_hat.numpy()
            print(rows[-1], flush=True)
    Vp, _, _ = solve_fast(bt, b.m0_prior, b.alpha_prior)
    rows.append({"method": "prior", "mode": "none",
                 "m0_mae": float((b.m0_prior - m0_true).abs().mean()),
                 "alpha_rmse_ocean": float(torch.sqrt(((b.alpha_prior - a_true)[ocean] ** 2).mean())),
                 "rmse_V_vs_truth_kt": traj_rmse(Vp), "wall_s": 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp2_twin.csv", index=False)
    ps = pd.DataFrame(per_storm)
    ps.to_csv(RESULTS / "exp2_twin_storms.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    infl = m0_influence(b, a_true)
    valid = b.valid.numpy()
    lead = np.arange(infl.shape[1])
    mean_infl = np.array([np.mean(infl[valid[:, k], k]) if valid[:, k].any() else np.nan for k in lead])
    pd.DataFrame({"lead_h": lead, "mean_abs_dV_per_0p1_m0_ms": mean_infl}).to_csv(
        RESULTS / "exp2_m0_influence.csv", index=False)

    i = b.names.index(SHOW_ALPHA)
    L = int(b.lengths[i])
    pd.DataFrame({"hour": np.arange(L), "truth": a_true.numpy()[i, :L],
                  "prior": b.alpha_prior.numpy()[i, :L],
                  "pinn_joint": alphas[("pinn", "joint")][i, :L],
                  "shooting_joint": alphas[("shooting", "joint")][i, :L]}).to_csv(
        RESULTS / f"exp2_alpha_{SHOW_ALPHA}.csv", index=False)
    figure()


def figure():
    """Draw fig2 from the saved CSVs (no recomputation)."""
    df = pd.read_csv(RESULTS / "exp2_twin.csv")
    ps = pd.read_csv(RESULTS / "exp2_twin_storms.csv")
    inf = pd.read_csv(RESULTS / "exp2_m0_influence.csv")
    al = pd.read_csv(RESULTS / f"exp2_alpha_{SHOW_ALPHA}.csv")
    plt = plt_style()
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1))
    ax = axes[0]
    for (method, mode), mk in [(("pinn", "joint"), "o"), (("shooting", "joint"), "s"), (("shooting", "m0"), "^")]:
        g = ps[(ps["method"] == method) & (ps["mode"] == mode)]
        ax.plot(g.m0_true, g.m0_hat, ls="none", marker=mk, ms=6, color=C[method],
                mfc="none" if mode == "m0" else C[method],
                label=f"{'PINN' if method == 'pinn' else 'shooting'}, {mode}")
    g = ps[(ps["method"] == "pinn") & (ps["mode"] == "joint")]
    ax.plot(g.m0_true, g.m0_prior, ls="none", marker="x", color=C["prior"], label="prior (FAST)")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="true m0", ylabel="estimated m0", title="(a) m0 recovery")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    ax.plot(al.hour, al.truth, color=INK, lw=2.2, label="truth")
    ax.plot(al.hour, al.prior, color=C["prior"], lw=1.2, label="prior (FAST)")
    ax.plot(al.hour, al.pinn_joint, color=C["pinn"], lw=1.4, ls="--", label="PINN, joint")
    ax.plot(al.hour, al.shooting_joint, color=C["shooting"], lw=1.4, ls=":", label="shooting, joint")
    ax.set(xlabel="hour", ylabel="α(t)", title=f"(b) α(t) recovery, {SHOW_ALPHA.title()}")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[2]
    order = [("prior", "none")] + [(m, mo) for m in ("pinn", "shooting") for mo in MODES]
    labels, vals, cols = [], [], []
    for method, mode in order:
        r = df[(df["method"] == method) & (df["mode"] == mode)].iloc[0]
        labels.append("prior (FAST)" if method == "prior" else f"{'PINN' if method == 'pinn' else 'shooting'}: {mode}")
        vals.append(r.rmse_V_vs_truth_kt)
        cols.append(C["prior"] if method == "prior" else C[method])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set(xlabel="RMSE of V vs truth (kt)", title="(c) Trajectory error after calibration")
    ax.grid(axis="y", visible=False)

    ax = axes[3]
    ax.plot(inf.lead_h, inf.mean_abs_dV_per_0p1_m0_ms, color=C["m0"], lw=1.8)
    ax.axhline(NOISE_MS, color=MUTED, lw=1, ls=":")
    ax.text(195, NOISE_MS * 1.12, "observation noise, 1 m/s", ha="right", fontsize=8, color=INK)
    ax.set(xlabel="hour", ylabel="mean |ΔV| for Δm0 = +0.1 (m/s)", title="(d) How long m0 matters",
           xlim=(0, 200))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_twin.png")
    print("saved", FIGURES / "fig2_twin.png")


if __name__ == "__main__":
    import sys
    figure() if "--figure" in sys.argv else main()
