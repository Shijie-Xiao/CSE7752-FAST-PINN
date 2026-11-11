#!/usr/bin/env python3
"""Experiment 1 -- Do the learned m0 and alpha make FAST better on storms not used in training?

Reads the forecasts written by run_calibrations.py. Every number is out of
sample: a storm is scored only by a calibration that did not see it. Four
evaluations (common.py): each season cross-validated (trained on other storms
of the same season) and each season forecast by a calibration trained on the
other season. Only t > 24 h is scored; the first 24 h of observations may be
used for m0.

Compared (always the same FAST model, calibrated chi, RK4 15 min):
  FAST                                   FAST's alpha and FAST's m0
  FAST + m0 from first 24 h              m0 fitted to the first 24 h, FAST's alpha
  FAST + learned alpha                   alpha-net (trained with the PINN), FAST's m0
  FAST + learned alpha + m0 (24 h)       both
  ... the same with the alpha-net trained by shooting

The change of per-storm RMSE against FAST is reported with a 95 % bootstrap
interval over storms (paired: the same storms with and without learning).

Outputs: results/exp1_skill_by_storm.csv, results/exp1_skill_summary.csv,
results/exp1_skill_by_lead.csv, figures/fig1_forecast_skill.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (ASSIM_H, C, EVAL_STYLE, EVALUATIONS, FIGURES, INK, INK2, LABEL, MUTED, RESULTS, forest,
                    paired_change, plt_style, scored_forecasts, storm_scores)

#: (variant, m0 source) -> label, in the order of the figure
LADDER = [
    (("fast", "prior"), "FAST"),
    (("fast", "first24h"), f"FAST + m0 from first {ASSIM_H} h"),
    (("pinn_joint", "prior"), "FAST + learned α"),
    (("pinn_joint", "first24h"), f"FAST + learned α + m0 ({ASSIM_H} h)"),
    (("shooting_joint", "first24h"), "same, α trained by shooting"),
]
MAIN = ("pinn_joint", "first24h")
#: order in the figure: the 2017 storms (same / other season), then the 2024 storms
EV_ORDER = ["cv2017", "2024to2017", "cv2024", "2017to2024"]
LEAD_BINS = np.arange(0, 241, 24)


def summary(sc):
    base = sc[(sc.variant == "fast") & (sc.m0_source == "prior")].set_index(["evaluation", "storm"]).rmse_kt
    rows = []
    for (ev, variant, src), g in sc.groupby(["evaluation", "variant", "m0_source"]):
        b = base.loc[ev].reindex(g.storm).to_numpy()
        d, lo, hi = paired_change(g.rmse_kt, b)
        rows.append({"evaluation": ev, "evaluation_label": EVALUATIONS[ev], "season": g.season.iloc[0],
                     "variant": variant, "m0_source": src, "label": LABEL[variant], "storms": len(g),
                     "rmse_kt_mean": g.rmse_kt.mean(), "rmse_kt_median": g.rmse_kt.median(),
                     "bias_kt": g.bias_kt.mean(), "abs_peak_err_kt": g.peak_err_kt.abs().mean(),
                     "storms_better_than_fast": int(np.sum(g.rmse_kt.to_numpy() < b - 1e-9)),
                     "rmse_change_vs_fast_kt": d, "ci95_low_kt": lo, "ci95_high_kt": hi,
                     "rmse_change_vs_fast_pct": 100 * (g.rmse_kt.mean() / b.mean() - 1)})
    return pd.DataFrame(rows)


def by_lead(df):
    df = df.copy()
    df["lead_bin_h"] = pd.cut(df.lead_h, LEAD_BINS, right=False, labels=LEAD_BINS[:-1] + 12)
    g = df.groupby(["evaluation", "variant", "m0_source", "lead_bin_h"], observed=True)
    return g.err_kt.agg(rmse_kt=lambda e: np.sqrt(np.mean(e ** 2)), n="size").reset_index()


def main():
    df = scored_forecasts()
    sc = storm_scores(df)
    sc.to_csv(RESULTS / "exp1_skill_by_storm.csv", index=False, float_format="%.2f")
    sm = summary(sc)
    sm.to_csv(RESULTS / "exp1_skill_summary.csv", index=False, float_format="%.2f")
    lead = by_lead(df)
    lead.to_csv(RESULTS / "exp1_skill_by_lead.csv", index=False, float_format="%.2f")
    show = sm.set_index(["evaluation", "variant", "m0_source"])
    for ev in EV_ORDER:
        print(f"\n{EVALUATIONS[ev]}")
        for (v, src), lab in LADDER + [(("fast_rawchi", "prior"), "FAST, raw χ")]:
            r = show.loc[(ev, v, src)]
            print(f"  {lab:38s} RMSE {r.rmse_kt_mean:5.1f} kt  bias {r.bias_kt:+5.1f}  better on "
                  f"{r.storms_better_than_fast:2d}/{r.storms}  change {r.rmse_change_vs_fast_kt:+5.1f} kt "
                  f"[{r.ci95_low_kt:+.1f}, {r.ci95_high_kt:+.1f}]")
    figure(sc, sm)


def figure(sc, sm):
    plt = plt_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})
    show = sm.set_index(["evaluation", "variant", "m0_source"])
    evs = EV_ORDER

    ax = axes[0]
    rows = [(lab, {ev: tuple(show.loc[(ev, v, s), ["rmse_change_vs_fast_kt", "ci95_low_kt", "ci95_high_kt"]])
                   for ev in evs}) for (v, s), lab in LADDER[1:]]
    colours = {lab: (C["fast_m0"] if v == "fast" else C[v]) for (v, s), lab in LADDER[1:]}
    forest(ax, rows, evs, color_of=colours.get)
    fast = {ev: show.loc[(ev, "fast", "prior")].rmse_kt_mean for ev in evs}
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f"{lab}  (FAST: {fast[ev]:.1f} kt)" for lab, ev in zip(labels, evs)],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8,
              title="storms forecast (mean RMSE of FAST)", title_fontsize=8)
    ax.set(xlabel=f"change of RMSE against FAST after hour {ASSIM_H} (kt), mean over storms, 95 % interval",
           title="(a) Out-of-sample change of FAST's error  (< 0 = better)")
    ax.text(0.99, 0.99, "left of 0: better than FAST", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK2)

    ax = axes[1]
    piv = sc[(sc.variant == MAIN[0]) & (sc.m0_source == MAIN[1])].pivot_table(
        index=["season", "storm"], columns="evaluation", values="rmse_kt")
    base = sc[(sc.variant == "fast") & (sc.m0_source == "prior")].groupby(["season", "storm"]).rmse_kt.first()
    for season, same, other in [(2017, "cv2017", "2024to2017"), (2024, "cv2024", "2017to2024")]:
        b = base.loc[season]
        x = piv.loc[season, same] - b
        y = piv.loc[season, other] - b
        st = EVAL_STYLE[same]
        ax.plot(x, y, ls="none", marker=st["marker"], ms=7, color=C["pinn_joint"], alpha=0.9,
                label=f"{season} storms ({len(x)})")
        for name in x.index:
            if abs(x[name]) > 12 or abs(y[name]) > 12:
                ax.annotate(name.title(), (x[name], y[name]), fontsize=7, color=INK2, xytext=(4, -3),
                            textcoords="offset points")
    lim = 1.08 * float(np.nanmax(np.abs(ax.dataLim.get_points())))
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
           xlabel="ΔRMSE, trained on other storms of the same season (kt)",
           ylabel="ΔRMSE, trained on the other season (kt)",
           title=f"(b) Per storm: learned α + m0 ({ASSIM_H} h) − FAST  (< 0 = better)")
    ax.text(-lim * 0.95, lim * 0.95, "on the diagonal: the same change\nwhichever season it was trained on",
            fontsize=8, color=INK2, va="top")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_forecast_skill.png")
    print("saved", FIGURES / "fig1_forecast_skill.png")


if __name__ == "__main__":
    main()
