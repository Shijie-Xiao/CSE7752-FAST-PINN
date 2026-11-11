#!/usr/bin/env python3
"""Experiment 4 -- which ocean information should drive alpha(t)?

Same protocol as experiment 3 (13 training storms, 5 hold-out storms, PINN
and RK4 shooting, joint calibration of m0 and the alpha network), repeated
for three input sets o(t) of g_phi:

* climatology : what FAST's own alpha formula uses (monthly-climatology
                mixed-layer depth, stratification, water depth)
* oras5       : ORAS5 ocean reanalysis for the actual month, from the
                75-level temperature: SST, mixed-layer depth, T100 (mean
                temperature of the top 100 m), D26, TCHP
* both        : the union

Storm motion, Vp and the land flag are inputs in every set. Physical check:
the cold-wake feedback should be stronger (alpha smaller) where mixing
reaches cold water, i.e. where SST - T100 is large.

Outputs: results/exp4_ocean_inputs.csv, results/exp4_alpha_vs_ocean.csv,
figures/fig4_ocean_inputs.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import C, FIGURES, INK, MUTED, RESULTS, plt_style, setup_torch, split
from exp3_real_storms import ASSIM_H, assimilate_m0, evaluate, fit
from fastpinn.data import make_batch

SETS = ("climatology", "oras5", "both")
METHODS = ("pinn", "shooting")
SET_COLOR = {"climatology": "#2a78d6", "oras5": "#eb6834", "both": "#6250d6"}


def main():
    setup_torch()
    rows, scatter = [], None
    for fs in SETS:
        train, test, scaler = split(feature_set=fs)
        bt, bs = make_batch(train, scaler), make_batch(test, scaler)
        for method in METHODS:
            params, wall = fit(method, bt)
            with torch.no_grad():
                m0, alpha = params.m0().clone(), params.alpha(bt).clone()
                a_test = params.alpha_net(bs.features)
            tr, _ = evaluate(bt, train, m0, alpha, f"{fs}/{method}")
            m0_test = assimilate_m0(test, scaler, params.alpha_net)
            te, _ = evaluate(bs, test, m0_test, a_test, f"{fs}/{method}", t_from=ASSIM_H + 1)
            rows.append({"features": fs, "method": method, "train_rmse_kt": tr.rmse_kt.mean(),
                         "train_peak_err_kt": tr.peak_err_kt.abs().mean(),
                         "test_rmse_kt": te.rmse_kt.mean(), "test_peak_err_kt": te.peak_err_kt.abs().mean(),
                         "test_peak_timing_h": te.peak_timing_h.abs().mean(),
                         "alpha_mean_train": float(alpha[bt.valid].mean()), "calib_wall_s": wall,
                         **{f"test_{s}": v for s, v in zip(te.storm, te.rmse_kt)}})
            print(rows[-1], flush=True)
            if fs == "oras5" and method == "pinn":
                pts = []
                for i, s in enumerate(train):
                    ocean = s.features[:, list(scaler_names(fs)).index("shallow")] < 0.5
                    sst = s.features[:, 0]
                    t100 = s.features[:, 2]
                    pts.append(pd.DataFrame({"storm": s.name, "alpha": alpha[i, :s.T].numpy()[ocean],
                                             "alpha_prior": s.alpha_prior[ocean],
                                             "sst_minus_t100": (sst - t100)[ocean],
                                             "tchp": s.features[ocean, 4]}))
                scatter = pd.concat(pts, ignore_index=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp4_ocean_inputs.csv", index=False)
    scatter.to_csv(RESULTS / "exp4_alpha_vs_ocean.csv", index=False)
    print(df[["features", "method", "train_rmse_kt", "test_rmse_kt", "test_peak_err_kt"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    figure()


def scaler_names(fs):
    from fastpinn.config import FEATURE_SETS
    return FEATURE_SETS[fs]


def figure():
    df = pd.read_csv(RESULTS / "exp4_ocean_inputs.csv")
    sc = pd.read_csv(RESULTS / "exp4_alpha_vs_ocean.csv")
    plt = plt_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, col, title in [(axes[0], "train_rmse_kt", "(a) Training storms, in-sample"),
                           (axes[1], "test_rmse_kt", f"(b) Hold-out storms, t > {ASSIM_H} h")]:
        x = np.arange(len(METHODS))
        for k, fs in enumerate(SETS):
            vals = [df[(df.features == fs) & (df.method == m)][col].iloc[0] for m in METHODS]
            bars = ax.bar(x + (k - 1) * 0.26, vals, width=0.24, color=SET_COLOR[fs], label=fs)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, ["PINN", "shooting"])
        ax.set(ylabel="RMSE vs IBTrACS (kt)", title=title)
        ax.grid(axis="x", visible=False)
    axes[1].legend(title="inputs of g$_\\phi$", fontsize=8.5, title_fontsize=8.5, loc="upper left", ncol=3)
    axes[1].set_ylim(0, 44)
    ax = axes[2]
    r = sc[["alpha", "sst_minus_t100"]].corr().iloc[0, 1]
    r0 = sc[["alpha_prior", "sst_minus_t100"]].corr().iloc[0, 1]
    ax.plot(sc.sst_minus_t100, sc.alpha_prior, ls="none", marker=".", ms=2.5, color=MUTED, alpha=0.5,
            label=f"FAST prior α (r = {r0:.2f})")
    ax.plot(sc.sst_minus_t100, sc.alpha, ls="none", marker=".", ms=3, color=C["pinn"], alpha=0.6,
            label=f"learned α, ORAS5 inputs (r = {r:.2f})")
    ax.set(xlabel="SST − T100 from ORAS5 (°C): cooling available to mixing", ylabel="α (1 = no cold-wake feedback)",
           title="(c) Learned α vs ORAS5 cooling potential (13 training storms)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_ocean_inputs.png")
    print("saved", FIGURES / "fig4_ocean_inputs.png")


if __name__ == "__main__":
    import sys
    figure() if "--figure" in sys.argv else main()
