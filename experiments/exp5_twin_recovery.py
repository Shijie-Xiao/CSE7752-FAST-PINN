#!/usr/bin/env python3
"""Experiment 5 -- Twin experiment: can the estimators recover a known m0 and alpha?

On the real tracks and forcing of the 18 storms of 2017, a known truth is set:

* m0_true = FAST's m0 +/- 0.15 (alternating sign, clipped to [0.1, 0.95]);
* alpha_true(t) = 1 - 0.87 exp(-0.4 z(t)), where z(t) is FAST's ocean-mixing
  parameter behind its alpha = 1 - 0.87 exp(-z): same ocean dependence,
  stronger cold-wake feedback, so FAST's alpha is deliberately wrong.

Synthetic observations: FAST with the true values, converted to V_max, plus
N(0, 1 m/s) noise, every 6 h. Each estimator (PINN, shooting) is run with
m0 only, alpha only, and both learned. If one quantity is fixed at a wrong
value, the other compensates; only joint learning can find the truth.

Outputs: results/exp5_twin_summary.csv, results/exp5_twin_m0.csv,
results/exp5_twin_alpha_example.csv, figures/fig5_twin_recovery.png
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import torch

from common import C, FIGURES, INK, MS_TO_KNOTS, MUTED, RESULTS, TRAIN_SEASON, plt_style, setup_torch
from fastpinn.data import FeatureScaler, load_storms, make_batch
from fastpinn.evaluate import solve_fast
from fastpinn.forecast import fit
from fastpinn.physics import obs_operator, rk4_integrate

NOISE_MS = 1.0
C_TRUE = 0.4
MODES = {"m0": dict(learn_m0=True, learn_alpha=False), "alpha": dict(learn_m0=False, learn_alpha=True),
         "joint": dict(learn_m0=True, learn_alpha=True)}
MODE_LABEL = {"m0": "m0 only", "alpha": "α only", "joint": "m0 and α"}
EXAMPLE = "LEE"


def twin():
    storms = load_storms(TRAIN_SEASON)
    b = make_batch(storms, FeatureScaler(storms))
    g = torch.Generator().manual_seed(0)
    sign = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(b.B)])
    m0_true = torch.clamp(b.m0_prior + 0.15 * sign, 0.1, 0.95)
    z = -torch.log(torch.clamp(1.0 - b.alpha_prior, min=1e-9) / 0.87)
    a_true = 1.0 - 0.87 * torch.exp(-C_TRUE * z)
    with torch.no_grad():
        V, _ = rk4_integrate(b, m0_true, a_true, substeps=4)
        vmax = obs_operator(V, b.lat, b.u_tran, b.v_tran)
    noise = NOISE_MS * torch.randn(vmax.shape, generator=g)
    v_obs = torch.where(b.obs_mask, vmax + noise, torch.full_like(vmax, float("nan")))
    return b, dataclasses.replace(b, v_obs=v_obs), m0_true, a_true, V.numpy()


def run(job):
    method, mode = job
    warnings.filterwarnings("ignore")
    setup_torch(3)
    b, bt, m0_true, a_true, Vt = twin()
    params, wall = fit(method, bt, **MODES[mode])
    with torch.no_grad():
        m0_hat, a_hat = params.m0().clone(), params.alpha(bt).clone()
    V, _, _ = solve_fast(bt, m0_hat, a_hat)
    return method, mode, m0_hat.numpy(), a_hat.numpy(), V, wall


def main():
    setup_torch()
    b, bt, m0_true, a_true, Vt = twin()
    ocean = ((a_true < 0.999) & b.valid).numpy()
    L = b.lengths.numpy()

    def v_rmse(V):
        return float(np.mean([np.sqrt(np.mean((V[i, :L[i]] - Vt[i, :L[i]]) ** 2)) for i in range(b.B)]) * MS_TO_KNOTS)

    with mp.get_context("spawn").Pool(6) as pool:
        out = pool.map(run, [(m, mo) for m in ("pinn", "shooting") for mo in MODES])
    Vf, _, _ = solve_fast(b, b.m0_prior, b.alpha_prior)
    rows = [{"method": "FAST", "mode": "none", "m0_mae": float((b.m0_prior - m0_true).abs().mean()),
             "alpha_rmse_ocean": float(np.sqrt(np.mean((b.alpha_prior.numpy() - a_true.numpy())[ocean] ** 2))),
             "v_rmse_vs_truth_kt": v_rmse(Vf), "wall_s": 0.0}]
    per, alphas = [], {}
    for method, mode, m0_hat, a_hat, V, wall in out:
        rows.append({"method": method, "mode": mode, "m0_mae": float(np.mean(np.abs(m0_hat - m0_true.numpy()))),
                     "alpha_rmse_ocean": float(np.sqrt(np.mean((a_hat - a_true.numpy())[ocean] ** 2))),
                     "v_rmse_vs_truth_kt": v_rmse(V), "wall_s": wall})
        per += [{"method": method, "mode": mode, "storm": n, "m0_true": float(m0_true[i]), "m0_hat": float(m0_hat[i]),
                 "m0_fast": float(b.m0_prior[i])} for i, n in enumerate(b.names)]
        alphas[(method, mode)] = a_hat
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp5_twin_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(per).to_csv(RESULTS / "exp5_twin_m0.csv", index=False, float_format="%.4f")
    i = b.names.index(EXAMPLE)
    pd.DataFrame({"hour": np.arange(L[i]), "truth": a_true.numpy()[i, :L[i]], "fast": b.alpha_prior.numpy()[i, :L[i]],
                  "pinn_joint": alphas[("pinn", "joint")][i, :L[i]],
                  "shooting_joint": alphas[("shooting", "joint")][i, :L[i]]}).to_csv(
        RESULTS / "exp5_twin_alpha_example.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    figure()


def figure():
    df = pd.read_csv(RESULTS / "exp5_twin_summary.csv")
    ps = pd.read_csv(RESULTS / "exp5_twin_m0.csv")
    al = pd.read_csv(RESULTS / "exp5_twin_alpha_example.csv")
    col = {"pinn": C["pinn_joint"], "shooting": C["shooting_joint"], "FAST": C["fast"]}
    plt = plt_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1.2, 1.15]})
    ax = axes[0]
    for method, mk in [("pinn", "o"), ("shooting", "s")]:
        for mode, fill in [("joint", True), ("m0", False)]:
            g = ps[(ps["method"] == method) & (ps["mode"] == mode)]
            ax.plot(g.m0_true, g.m0_hat, ls="none", marker=mk, ms=6, color=col[method],
                    mfc=col[method] if fill else "none",
                    label=f"{'PINN' if method == 'pinn' else 'shooting'}, {MODE_LABEL[mode]} learned")
    g = ps[(ps["method"] == "pinn") & (ps["mode"] == "joint")]
    ax.plot(g.m0_true, g.m0_fast, ls="none", marker="x", color=col["FAST"], label="FAST's m0")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="true m0", ylabel="estimated m0", title="(a) Recovering m0")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    ax.plot(al.hour, al.truth, color=INK, lw=2.2, label="truth")
    ax.plot(al.hour, al.fast, color=col["FAST"], lw=1.2, label="FAST's α")
    ax.plot(al.hour, al.pinn_joint, color=col["pinn"], lw=1.4, ls="--", label="PINN, m0 and α learned")
    ax.plot(al.hour, al.shooting_joint, color=col["shooting"], lw=1.4, ls=":", label="shooting, m0 and α learned")
    ax.set(xlabel="hour", ylabel="α(t)", title=f"(b) Recovering α(t), {EXAMPLE.title()} 2017")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[2]
    order = [("FAST", "none")] + [(m, mo) for m in ("pinn", "shooting") for mo in MODES]
    labels, vals, cols = [], [], []
    for method, mode in order:
        r = df[(df["method"] == method) & (df["mode"] == mode)].iloc[0]
        labels.append("FAST (nothing learned)" if method == "FAST" else
                      f"{'PINN' if method == 'pinn' else 'shooting'}: {MODE_LABEL[mode]}")
        vals.append(r.v_rmse_vs_truth_kt)
        cols.append(col[method])
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set(xlabel="RMSE of V against the truth (kt)", title="(c) Trajectory error: learn one or both?")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_twin_recovery.png")
    print("saved", FIGURES / "fig5_twin_recovery.png")


if __name__ == "__main__":
    import sys
    figure() if "--figure" in sys.argv else main()
