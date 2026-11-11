#!/usr/bin/env python3
"""Experiment 3 -- Is the initial moisture m0 worth learning?

(a) How long m0 matters: FAST is run with m0 +/- 0.1 on all 28 storms
    (FAST's alpha); |Delta V_max| vs hours since the start.
(b) Forecast value: RMSE with FAST's m0 minus RMSE with m0 from the first
    24 h, by lead time, all 28 storms (cross-validated evaluations).
(c) Is the learned m0 a property of the storm? For each storm, the m0
    learned when it was a training storm (whole track, in the other CV folds
    of its season) against the m0 estimated from its first 24 h when it was
    held out.
(d) Does learning m0 in training help the alpha network? Out-of-sample
    error of the alpha-net trained jointly with m0 vs with m0 fixed at FAST's.

Outputs: results/exp3_m0_memory.csv, results/exp3_m0_gain_by_lead.csv,
results/exp3_m0_consistency.csv, results/exp3_m0_in_training.csv,
figures/fig3_initial_moisture.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import (ASSIM_H, C, EVAL_STYLE, EVALUATIONS, FIGURES, INK, INK2, MS_TO_KNOTS, MUTED, RESULTS,
                    TEST_SEASON, TRAIN_SEASON, color, plt_style, scored_forecasts, setup_torch)
from fastpinn.data import FeatureScaler, load_storms, make_batch
from fastpinn.evaluate import solve_fast

DM0 = 0.1
CV = ("cv2017", "cv2024")
LEAD_BINS = np.arange(0, 241, 24)
OBS_PRECISION_KT = 5.0          # IBTrACS intensities are reported in 5 kt steps


def memory():
    storms = load_storms(TRAIN_SEASON) + load_storms(TEST_SEASON)
    b = make_batch(storms, FeatureScaler(storms))
    _, _, v0 = solve_fast(b, b.m0_prior, b.alpha_prior)
    d = []
    for dm in (DM0, -DM0):
        _, _, v1 = solve_fast(b, (b.m0_prior + dm).clamp(0.01, 1.0), b.alpha_prior)
        d.append(np.abs(v1 - v0) * MS_TO_KNOTS)
    d = np.maximum(*d)
    valid = b.valid.numpy()
    hours = np.arange(d.shape[1])
    rows = []
    for k in hours[:241]:
        x = d[valid[:, k], k]
        rows.append({"hour": k, "median_kt": np.median(x), "q25_kt": np.quantile(x, 0.25),
                     "q75_kt": np.quantile(x, 0.75), "max_kt": x.max(), "storms": len(x)})
    return pd.DataFrame(rows)


def consistency():
    m0 = pd.read_csv(RESULTS / "learned_m0.csv")
    m0 = m0[(m0.variant == "pinn_joint") & m0.evaluation.isin(CV)]
    tr = m0[m0.role == "train"].groupby(["season", "storm"]).m0.agg(["mean", "min", "max", "count"])
    te = m0[(m0.role == "test") & (m0.m0_source == "first24h")].set_index(["season", "storm"])
    out = tr.join(te[["m0", "m0_fast"]]).reset_index()
    return out.rename(columns={"mean": "m0_whole_track_mean", "min": "m0_whole_track_min",
                               "max": "m0_whole_track_max", "count": "folds", "m0": "m0_first24h"})


def m0_gain_by_lead():
    """RMSE(FAST's m0) - RMSE(m0 from the first 24 h) by lead time, all storms of the CV evaluations."""
    df = scored_forecasts()
    df = df[df.evaluation.isin(CV) & df.variant.isin(["fast", "pinn_joint"])].copy()
    df["lead_bin_h"] = pd.cut(df.lead_h, LEAD_BINS, right=False, labels=LEAD_BINS[:-1] + 12).astype(float)
    r = df.groupby(["variant", "lead_bin_h", "m0_source"]).err_kt.agg(
        rmse=lambda e: np.sqrt(np.mean(e ** 2)), n="size").reset_index()
    w = r.pivot_table(index=["variant", "lead_bin_h"], columns="m0_source", values=["rmse", "n"])
    out = pd.DataFrame({"rmse_prior_kt": w[("rmse", "prior")], "rmse_first24h_kt": w[("rmse", "first24h")],
                        "n": w[("n", "prior")]}).reset_index()
    out["gain_kt"] = out.rmse_prior_kt - out.rmse_first24h_kt
    return out


def in_training():
    sm = pd.read_csv(RESULTS / "exp1_skill_summary.csv")
    sm = sm[sm.variant.isin(["pinn_joint", "pinn_alpha_only"]) & (sm.m0_source == "first24h")
            & sm.evaluation.isin(["cv2017", "2017to2024"])]
    return sm[["evaluation", "season", "variant", "label", "rmse_kt_mean", "bias_kt", "storms_better_than_fast",
               "rmse_change_vs_fast_kt", "ci95_low_kt", "ci95_high_kt"]]


def main():
    setup_torch()
    mem = memory()
    mem.to_csv(RESULTS / "exp3_m0_memory.csv", index=False, float_format="%.3f")
    con = consistency()
    con.to_csv(RESULTS / "exp3_m0_consistency.csv", index=False, float_format="%.3f")
    trn = in_training()
    trn.to_csv(RESULTS / "exp3_m0_in_training.csv", index=False, float_format="%.2f")
    gain = m0_gain_by_lead()
    gain.to_csv(RESULTS / "exp3_m0_gain_by_lead.csv", index=False, float_format="%.2f")
    above = mem.hour[mem.median_kt >= OBS_PRECISION_KT]
    print(f"m0 +/- {DM0}: median |dVmax| peaks at {mem.median_kt.max():.1f} kt (hour {mem.median_kt.idxmax()}); "
          + (f"above {OBS_PRECISION_KT:g} kt until hour {int(above.max())}" if len(above) else
             f"never above {OBS_PRECISION_KT:g} kt"))
    r = np.corrcoef(con.m0_whole_track_mean, con.m0_first24h)[0, 1]
    r0 = np.corrcoef(con.m0_fast, con.m0_first24h)[0, 1]
    print(f"m0 whole track vs first 24 h: r = {r:.2f}; FAST m0 vs first 24 h: r = {r0:.2f}  ({len(con)} storms)")
    print(gain.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(trn.to_string(index=False))
    figure(mem, con, trn, gain, r, r0)


def figure(mem, con, trn, gain, r, r0):
    plt = plt_style()
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.3))
    ax = axes[0]
    ax.fill_between(mem.hour, mem.q25_kt, mem.q75_kt, color=C["fast"], alpha=0.2, lw=0, label="25–75 % of storms")
    ax.plot(mem.hour, mem.median_kt, color=C["fast"], lw=1.8, label="median")
    ax.axhline(OBS_PRECISION_KT, color=MUTED, lw=1, ls=":")
    ax.text(238, OBS_PRECISION_KT * 1.06, "IBTrACS precision, 5 kt", ha="right", fontsize=8, color=INK2)
    ax.set(xlabel="hours since first observation", ylabel=f"|ΔV$_{{max}}$| for m0 ± {DM0} (kt)",
           title="(a) How long m0 matters (28 storms)", xlim=(0, 240))
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    for v, lab in [("fast", "FAST's α"), ("pinn_joint", "learned α (same season)")]:
        g = gain[(gain.variant == v) & (gain.n >= 8)]
        ax.plot(g.lead_bin_h, g.gain_kt, marker="o", ms=4, color=color(v, "first24h"), label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xlabel=f"lead time after hour {ASSIM_H} (h)", ylabel="RMSE(FAST's m0) − RMSE(m0 from 24 h) (kt)",
           title=f"(b) Gain from m0 of the first {ASSIM_H} h (28 storms)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    for season, ev in [(TRAIN_SEASON, "cv2017"), (TEST_SEASON, "cv2024")]:
        g = con[con.season == season]
        ax.errorbar(g.m0_whole_track_mean, g.m0_first24h,
                    xerr=[g.m0_whole_track_mean - g.m0_whole_track_min, g.m0_whole_track_max - g.m0_whole_track_mean],
                    ls="none", marker=EVAL_STYLE[ev]["marker"], ms=6, color=C["pinn_joint"], ecolor=MUTED,
                    elinewidth=1, label=f"learned, {season} storms")
    ax.plot(con.m0_fast, con.m0_first24h, ls="none", marker="x", ms=6, color=C["fast"],
            label=f"FAST's m0 on the x-axis (r = {r0:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.text(0.98, 0.03, f"learned: r = {r:.2f}", transform=ax.transAxes, ha="right", fontsize=9, color=INK)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="m0 learned from the whole track (mean, range over folds)",
           ylabel=f"m0 from the first {ASSIM_H} h (held out)", title="(c) Is the learned m0 a storm property?")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    labels = {"pinn_joint": "α-net trained\nwith m0 learned", "pinn_alpha_only": "α-net trained\nwith FAST's m0"}
    x = np.arange(2)
    for j, ev in enumerate(("cv2017", "2017to2024")):
        g = trn[trn.evaluation == ev].set_index("variant").reindex(list(labels))
        bars = ax.bar(x + (j - 0.5) * 0.36, g.rmse_kt_mean, width=0.34, color=[C[v] for v in labels],
                      alpha=1.0 if j == 0 else 0.55, label=EVALUATIONS[ev])
        for bx, v in zip(bars, g.rmse_kt_mean):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, list(labels.values()))
    ax.set(ylabel=f"out-of-sample RMSE (kt), m0 from {ASSIM_H} h at test", title="(d) Learning m0 in training")
    ax.set_ylim(0, 1.3 * float(trn.rmse_kt_mean.max()))
    ax.legend(loc="upper center", ncol=2, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_initial_moisture.png")
    print("saved", FIGURES / "fig3_initial_moisture.png")


if __name__ == "__main__":
    main()
