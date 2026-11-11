#!/usr/bin/env python3
"""Experiment 2 -- the 10 North Atlantic hurricanes of 2024, storm by storm.

V_max against IBTrACS of FAST and of FAST with the learned alpha and m0 from
the first 24 h, the alpha-net (PINN) trained either
  * on the 18 storms of 2017 (a different season), or
  * on the other 8 hurricanes of 2024 (cross-validation, same season).
The first 24 h (shaded) may use observations; the rest is a forecast.

Outputs: results/exp2_hurricanes_2024.csv, figures/fig2_hurricanes_2024.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import ASSIM_H, C, FIGURES, INK, RESULTS, TEST_SEASON, plt_style

#: column -> (evaluation, variant, m0 source, label, colour, line style)
SHOWN = {
    "fast_kt": ("2017to2024", "fast", "prior", "FAST", C["fast"], "-"),
    "learned_trained_2017_kt": ("2017to2024", "pinn_joint", "first24h",
                                f"FAST + learned α + m0 ({ASSIM_H} h), trained on 2017", C["pinn_joint"], "--"),
    "learned_trained_2024_kt": ("cv2024", "pinn_joint", "first24h",
                                f"FAST + learned α + m0 ({ASSIM_H} h), trained on the other 2024 hurricanes",
                                C["pinn_joint"], "-"),
}


def main():
    fc = pd.read_csv(RESULTS / "forecasts.csv")
    st = pd.read_csv(RESULTS / "storms.csv")
    st = st[st.season == TEST_SEASON]
    sc = pd.read_csv(RESULTS / "exp1_skill_by_storm.csv")
    sc = sc[sc.season == TEST_SEASON].set_index(["evaluation", "storm", "variant", "m0_source"]).rmse_kt
    wide = st[["storm", "hour", "obs_kt", "vp_kt"]].copy()
    for col, (ev, v, s, *_) in SHOWN.items():
        g = fc[(fc.evaluation == ev) & (fc.variant == v) & (fc.m0_source == s)][["storm", "hour", "vmax_kt"]]
        wide = wide.merge(g.rename(columns={"vmax_kt": col}), on=["storm", "hour"])
    wide.to_csv(RESULTS / "exp2_hurricanes_2024.csv", index=False, float_format="%.2f")

    plt = plt_style()
    storms = list(dict.fromkeys(wide.storm))
    fig, axes = plt.subplots(3, 4, figsize=(17, 10.5), sharey=True)
    for ax, name in zip(axes.flat, storms):
        g = wide[wide.storm == name]
        ax.axvspan(0, ASSIM_H, color="#ecebe7", lw=0)
        ax.plot(g.hour, g.vp_kt, color=C["vp"], lw=0.8, alpha=0.7)
        o = g[np.isfinite(g.obs_kt)]
        ax.plot(o.hour, o.obs_kt, color=INK, lw=1.8)
        for col, (ev, v, s, lab, colour, ls) in SHOWN.items():
            ax.plot(g.hour, g[col], color=colour, lw=1.6, ls=ls)
        r = [sc[(ev, name, v, s)] for ev, v, s, *_ in SHOWN.values()]
        ax.set_title(f"{name.title()}: RMSE {r[0]:.0f} | {r[1]:.0f} | {r[2]:.0f} kt", fontsize=10)
        ax.set_ylim(0, 190)
    for ax in axes.flat[len(storms):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("V$_{max}$ (kt)")
    for ax in list(axes[-1, :2]) + list(axes[-2, 2:]):
        ax.set_xlabel("hour since first observation")
    leg = axes.flat[-2]
    for lab, col, lw, ls in [("IBTrACS", INK, 1.8, "-"), ("Vp (potential intensity)", C["vp"], 0.8, "-")] + \
            [(lab, colour, 1.6, ls) for *_, lab, colour, ls in SHOWN.values()]:
        leg.plot([], [], color=col, lw=lw, ls=ls, label=lab.replace(", trained", ",\ntrained"))
    leg.fill_between([], [], color="#ecebe7", label=f"first {ASSIM_H} h (observations may be used)")
    leg.legend(loc="center left", fontsize=9, bbox_to_anchor=(0.0, 0.5),
               title="RMSE in titles: FAST | trained on 2017 | trained on 2024", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_hurricanes_2024.png")
    print("saved", FIGURES / "fig2_hurricanes_2024.png")


if __name__ == "__main__":
    main()
