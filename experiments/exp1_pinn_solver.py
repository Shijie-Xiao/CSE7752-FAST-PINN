#!/usr/bin/env python3
"""Experiment 1 -- the PINN as an alternative *solver* for the FAST ODEs.

Parameters and initial conditions are fixed at the physics prior (FAST's
alpha and m0); only the trajectory network is trained, with the physics
residual alone (w_obs = 0). Three training variants:

* plain PINN
* causal PINN (residual of later times weighted by exp(-eps * earlier loss),
  Wang, Sankaran & Perdikaris 2022), eps = 100
* time-marching curriculum (trained window grows from 5 % to 100 % of each storm)

Each is compared with a converged RK4 reference (dt = 1/16 h) in
E_inf = max_t ||((V - V_ref)/V*, m - m_ref)|| and in cost with RK4 at 1 h.
This answers Reviewer 2 (comment 7) and Reviewer 4 (comment 1): what does the
PINN offer as a solver, separately from calibration?

Outputs: results/exp1_pinn_solver.csv, figures/fig1_pinn_solver.png
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from common import FIGURES, INK, MUTED, PINN_KW, RESULTS, C, plt_style, seed_all, setup_torch, split
from fastpinn.calibrate import LatentParams, PINNCalibrator, Weights, train_pinn
from fastpinn.config import V_STAR
from fastpinn.data import make_batch
from fastpinn.physics import rk4_integrate

STORMS = ("IRMA", "MARIA", "NATE", "OPHELIA")
ITERS = 6000
VARIANTS = {
    "plain PINN": dict(causal_eps=None, curriculum=0.0),
    "causal PINN (ε = 100)": dict(causal_eps=100.0, curriculum=0.0),
    "time-marching PINN": dict(causal_eps=None, curriculum=0.7),
}
VCOL = {"plain PINN": "#eb6834", "causal PINN (ε = 100)": "#6250d6", "time-marching PINN": "#1baf7a"}


def einf(b, V, m, Vr, mr):
    out = []
    for i in range(b.B):
        L = int(b.lengths[i])
        out.append(float(np.max(np.sqrt(((V[i, :L] - Vr[i, :L]) / V_STAR) ** 2 + (m[i, :L] - mr[i, :L]) ** 2))))
    return np.array(out)


def main():
    setup_torch()
    train, test, scaler = split()
    storms = [s for s in train + test if s.name in STORMS]
    b = make_batch(storms, scaler)
    with torch.no_grad():
        Vr, mr = (x.numpy() for x in rk4_integrate(b, b.m0_prior, b.alpha_prior, substeps=16))

    rows, traj = [], {}
    with torch.no_grad():
        tic = time.perf_counter()
        V, m = rk4_integrate(b, b.m0_prior, b.alpha_prior, substeps=1)
        rk_wall = time.perf_counter() - tic
    e = einf(b, V.numpy(), m.numpy(), Vr, mr)
    rows += [{"storm": n, "solver": "RK4, dt = 1 h", "E_inf": e[i], "wall_s": rk_wall} for i, n in enumerate(b.names)]

    for name, opt in VARIANTS.items():
        seed_all(0)
        params = LatentParams(b, b.features.shape[-1], learn_m0=False, learn_alpha=False)
        cal = PINNCalibrator(b, params, causal_eps=opt["causal_eps"], **PINN_KW)
        hist = train_pinn(cal, Weights(obs=0.0, phys=1.0, prior=0.0, bound=1.0), iters=ITERS, lr=2e-3,
                          log_every=500, curriculum=opt["curriculum"])
        with torch.no_grad():
            V, m = (x.numpy() for x in cal.nodes())
        e = einf(b, V, m, Vr, mr)
        traj[name] = V
        rows += [{"storm": n, "solver": name, "E_inf": e[i], "wall_s": hist.wall} for i, n in enumerate(b.names)]
        print(f"{name:24s} E_inf {np.round(e, 3)}  ({hist.wall:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp1_pinn_solver.csv", index=False)
    print(df.pivot_table(index="storm", columns="solver", values="E_inf").to_string(float_format=lambda x: f"{x:.2e}"))

    plt = plt_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for j, name in enumerate(["IRMA", "MARIA"]):
        ax = axes[j]
        i = b.names.index(name)
        L = int(b.lengths[i])
        ax.plot(np.arange(L), Vr[i, :L], color=INK, lw=2.4, label="RK4 (reference)")
        for v, V in traj.items():
            ax.plot(np.arange(L), V[i, :L], color=VCOL[v], lw=1.4, label=v)
        ax.set(xlabel="hour", ylabel="V (m/s)", title=f"({'ab'[j]}) {name.title()}: physics-only PINN vs RK4")
    axes[0].legend(loc="upper left", fontsize=8)
    ax = axes[2]
    order = ["RK4, dt = 1 h"] + list(VARIANTS)
    for k, sol in enumerate(order):
        g = df[df.solver == sol]
        ax.semilogx(g.E_inf, np.full(len(g), k), ls="none", marker="o", ms=7,
                    color=INK if sol.startswith("RK4") else VCOL[sol], alpha=0.85)
        ax.text(g.E_inf.max() * 1.6, k, f"{g.wall_s.iloc[0] * 1e3:.0f} ms" if sol.startswith("RK4")
                else f"{g.wall_s.iloc[0]:.0f} s", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(1e-7, 30)
    ax.set(xlabel=r"$E_\infty$ vs reference (4 storms, 180–385 h)", title="(c) Accuracy and cost")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_pinn_solver.png")
    print("saved", FIGURES / "fig1_pinn_solver.png")


if __name__ == "__main__":
    main()
