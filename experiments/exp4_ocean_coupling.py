#!/usr/bin/env python3
"""Experiment 4 -- Is the learned ocean coupling alpha(t) useful, and is it physical?

(a) Value of each alpha source on unseen storms, m0 from the first 24 h for
    all, as the change of per-storm RMSE against FAST's own alpha (95 %
    bootstrap interval): one constant alpha fitted to the training storms,
    the alpha-net on FAST's climatology inputs (PINN or shooting), the
    alpha-net on ORAS5 (2017 only). If the alpha-net is no better than one
    constant, it learned a bias correction, not ocean physics.
(b) alpha along the 2024 tracks (over ocean): FAST's, learned with FAST's
    calibrated chi, learned with raw chi. With raw chi FAST under-ventilates
    the storm, and the learned alpha drops to compensate.
(c) What the alpha-net responds to: change of alpha when one input goes from
    its 5th to its 95th percentile (others at the median), for the net
    trained on 2017 and the net trained on 2024, against the sign FAST's own
    formula alpha = 1 - 0.87 exp(-z), z ~ strat^-0.4 h_m u_T Vp/V implies.
(d) alpha(t) along one 2024 track.

Outputs: results/exp4_alpha_skill.csv, results/exp4_alpha_values.csv,
results/exp4_alpha_response.csv, results/exp4_alpha_example.csv,
figures/fig4_ocean_coupling.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import (ASSIM_H, C, CALIB_DIR, EVALUATIONS, FIGURES, INK, INK2, MUTED, RESULTS, TEST_SEASON,
                    TRAIN_SEASON, forest, paired_change, plt_style, setup_torch)
from fastpinn.config import FEATURE_SETS
from fastpinn.data import load_storms
from fastpinn.networks import AlphaNet

SOURCES = [("pinn_constant_alpha", "one constant α (PINN)"), ("pinn_joint", "α-net, climatology (PINN)"),
           ("shooting_joint", "α-net, climatology (shooting)"), ("pinn_joint_oras5", "α-net, ORAS5 (PINN)")]
#: input -> (readable name, sign of d alpha / d input in FAST's formula)
INPUTS = {"log_hm": ("mixed-layer depth", +1), "log_strat": ("stratification below it", -1),
          "log_depth": ("water depth", -1), "u_T": ("translation speed", +1), "vp_norm": ("Vp", +1),
          "shallow": ("shallow-water / land flag", +1)}
NETS = {"2017to2024": f"trained on {TRAIN_SEASON}", "2024to2017": f"trained on {TEST_SEASON}"}
EXAMPLE = "MILTON"
EV_ORDER = ["cv2017", "2024to2017", "cv2024", "2017to2024"]


def skill():
    """Change of RMSE against FAST's alpha, both with m0 from the first 24 h."""
    sc = pd.read_csv(RESULTS / "exp1_skill_by_storm.csv")
    sc = sc[sc.m0_source == "first24h"]
    base = sc[sc.variant == "fast"].set_index(["evaluation", "storm"]).rmse_kt
    rows = []
    for (ev, v), g in sc[sc.variant.isin([v for v, _ in SOURCES])].groupby(["evaluation", "variant"]):
        d, lo, hi = paired_change(g.rmse_kt, base.loc[ev].reindex(g.storm))
        rows.append({"evaluation": ev, "variant": v, "storms": len(g), "rmse_kt_mean": g.rmse_kt.mean(),
                     "fast_alpha_rmse_kt_mean": base.loc[ev].reindex(g.storm).mean(),
                     "change_vs_fast_alpha_kt": d, "ci95_low_kt": lo, "ci95_high_kt": hi})
    return pd.DataFrame(rows)


def values():
    al = pd.read_csv(RESULTS / "learned_alpha.csv")
    st = pd.read_csv(RESULTS / "storms.csv")
    st["ocean"] = (st.alpha_fast < 0.999) & (st.vp_kt > 0)
    al = al[al.evaluation == "2017to2024"].merge(st[["season", "storm", "hour", "alpha_fast", "ocean"]],
                                                  on=["season", "storm", "hour"])
    al = al[al.ocean]
    fast = al[al.variant == "pinn_joint"].assign(variant="fast", alpha=lambda d: d.alpha_fast)
    return pd.concat([fast, al])


def response():
    X = np.concatenate([s.features[(s.alpha_prior < 0.999) & (s.vp > 0)]
                        for season in (TRAIN_SEASON, TEST_SEASON) for s in load_storms(season)])
    base = np.median(X, axis=0)
    rows = []
    for split, net_label in NETS.items():
        ck = torch.load(CALIB_DIR / f"{split}__pinn_joint.pt", weights_only=False)
        names = FEATURE_SETS[ck["features"]]
        net = AlphaNet(len(names))
        net.load_state_dict(ck["alpha_net"])
        for j, n in enumerate(names):
            lo, hi = (0.0, 1.0) if n == "shallow" else np.quantile(X[:, j], [0.05, 0.95])
            x = np.tile(base, (2, 1))
            x[:, j] = [lo, hi]
            with torch.no_grad():
                a = net(torch.as_tensor((x - ck["scaler_mean"]) / ck["scaler_std"])).numpy()
            rows.append({"net": net_label, "input": n, "name": INPUTS[n][0], "low": lo, "high": hi,
                         "alpha_low": a[0], "alpha_high": a[1], "d_alpha": a[1] - a[0], "fast_sign": INPUTS[n][1],
                         "agrees_with_fast": bool(np.sign(a[1] - a[0]) == INPUTS[n][1])})
    return pd.DataFrame(rows)


def example():
    al = pd.read_csv(RESULTS / "learned_alpha.csv")
    st = pd.read_csv(RESULTS / "storms.csv")
    st = st[(st.season == TEST_SEASON) & (st.storm == EXAMPLE)][["hour", "alpha_fast", "vp_kt"]].set_index("hour")
    g = al[(al.storm == EXAMPLE) & (al.season == TEST_SEASON)]
    cols = {("2017to2024", "pinn_joint"): "alpha_net_trained_2017",
            ("cv2024", "pinn_joint"): "alpha_net_trained_2024",
            ("2017to2024", "pinn_constant_alpha"): "constant_alpha_trained_2017",
            ("2017to2024", "pinn_joint_rawchi"): "alpha_net_raw_chi_trained_2017"}
    for (ev, v), name in cols.items():
        st[name] = g[(g.evaluation == ev) & (g.variant == v)].set_index("hour").alpha
    return st.reset_index()


def main():
    setup_torch()
    sk = skill()
    sk.to_csv(RESULTS / "exp4_alpha_skill.csv", index=False, float_format="%.2f")
    val = values()
    stats = val.groupby("variant").alpha.describe(percentiles=[0.1, 0.5, 0.9])
    stats.to_csv(RESULTS / "exp4_alpha_values.csv", float_format="%.3f")
    res = response()
    res.to_csv(RESULTS / "exp4_alpha_response.csv", index=False, float_format="%.4f")
    ex = example()
    ex.to_csv(RESULTS / "exp4_alpha_example.csv", index=False, float_format="%.4f")
    print(sk.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(stats[["mean", "10%", "50%", "90%"]].to_string(float_format=lambda x: f"{x:.3f}"))
    print(res[["net", "name", "d_alpha", "fast_sign", "agrees_with_fast"]].to_string(index=False))
    figure(sk, val, res, ex)


def figure(sk, val, res, ex):
    plt = plt_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    axes = axes.flat
    evs = EV_ORDER

    ax = axes[0]
    s = sk.set_index(["variant", "evaluation"])
    rows = [(lab, {ev: tuple(s.loc[(v, ev), ["change_vs_fast_alpha_kt", "ci95_low_kt", "ci95_high_kt"]])
                   for ev in evs if (v, ev) in s.index}) for v, lab in SOURCES]
    colours = {lab: C[v] for v, lab in SOURCES}
    forest(ax, rows, evs, color_of=colours.get)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, title="storms forecast",
              title_fontsize=8)
    ax.set(xlabel=f"change of RMSE against FAST's α (kt), m0 from first {ASSIM_H} h for both",
           title="(a) Which α helps on unseen storms?  (< 0 = better)")

    ax = axes[1]
    groups = [("fast", "FAST's α"), ("pinn_joint", "learned,\ncalibrated χ"), ("pinn_joint_rawchi", "learned,\nraw χ")]
    data = [val[val.variant == v].alpha.to_numpy() for v, _ in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
    for patch, (v, _) in zip(bp["boxes"], groups):
        patch.set_facecolor(C[v])
        patch.set_alpha(0.75)
    for k, d in enumerate(data, start=1):
        ax.text(k, 1.03, f"mean {d.mean():.2f}", ha="center", fontsize=8.5, color=INK)
    sm = pd.read_csv(RESULTS / "exp1_skill_summary.csv").set_index(["evaluation", "variant", "m0_source"])
    r_raw = sm.loc[("2017to2024", "fast_rawchi", "prior")].rmse_kt_mean
    r_fix = sm.loc[("2017to2024", "pinn_joint_rawchi", "prior")].rmse_kt_mean
    r_fast = sm.loc[("2017to2024", "fast", "prior")].rmse_kt_mean
    ax.text(0.5, 0.03, f"RMSE on 2024 with raw χ:\nFAST {r_raw:.1f} kt → learned α {r_fix:.1f} kt\n"
            f"(FAST with calibrated χ: {r_fast:.1f} kt)", transform=ax.transAxes, ha="center", fontsize=8.5,
            color=INK2)
    ax.set_xticks(range(1, len(groups) + 1), [lab for _, lab in groups])
    ax.set(ylabel=f"α over ocean along the {TEST_SEASON} tracks (trained on {TRAIN_SEASON})", ylim=(0, 1.1),
           title="(b) Learned α absorbs model error in χ")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    names = list(dict.fromkeys(res.name))[::-1]
    y = np.arange(len(names))
    for k, (net, col) in enumerate(zip(NETS.values(), (C["pinn_joint"], "#f4b393"))):
        g = res[res.net == net].set_index("name").reindex(names)
        ax.barh(y + (0.2 if k == 0 else -0.2), g.d_alpha, height=0.38, color=col, label=f"α-net {net}")
    sign = res.drop_duplicates("name").set_index("name").fast_sign.reindex(names)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [f"{n} (FAST: {'+' if s > 0 else '−'})" for n, s in zip(names, sign)])
    ax.set(xlabel="Δα, input from 5th to 95th percentile", title="(c) What the α-net learned")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", visible=False)

    ax = axes[3]
    land = ex.vp_kt <= 0
    ax.fill_between(ex.hour, 0, 1.05, where=land, color="#ecebe7", lw=0, label="over land (Vp = 0)")
    for col, lab, colour, ls in [("alpha_fast", "FAST's α", C["fast"], "-"),
                                 ("alpha_net_trained_2017", "α-net, trained on 2017", C["pinn_joint"], "--"),
                                 ("alpha_net_trained_2024", "α-net, trained on other 2024", C["pinn_joint"], "-"),
                                 ("constant_alpha_trained_2017", "constant α, trained on 2017",
                                  C["pinn_constant_alpha"], ":"),
                                 ("alpha_net_raw_chi_trained_2017", "α-net, raw χ, trained on 2017",
                                  C["pinn_joint_rawchi"], "-.")]:
        ax.plot(ex.hour, ex[col], ls=ls, lw=1.6, color=colour, label=lab)
    ax.set(xlabel="hour since first observation", ylabel="α(t)", ylim=(0, 1.05),
           title=f"(d) α along {EXAMPLE.title()} {TEST_SEASON}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_ocean_coupling.png")
    print("saved", FIGURES / "fig4_ocean_coupling.png")


if __name__ == "__main__":
    main()
