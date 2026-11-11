"""Shared definitions for the experiments: splits, training variants, colours, plot style."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fastpinn.config import ASSIM_H, FIGURES, MS_TO_KNOTS, RESULTS, TEST_SEASON, TRAIN_SEASON  # noqa: E402
from fastpinn.data import FeatureScaler, load_storms  # noqa: E402

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
CALIB_DIR = RESULTS / "calibrations"

# --------------------------------------------------------------------------- #
# Train / test splits
# --------------------------------------------------------------------------- #
#: Four evaluations. A storm is only ever forecast by a calibration that did
#: not see it.
#:   cv2017      2017, cross-validated: 6 folds of 3 storms, trained on the other 15
#:   cv2024      2024, cross-validated: 5 folds of 2 storms, trained on the other 8
#:   2017to2024  2024, trained on all 18 storms of 2017 (a new season)
#:   2024to2017  2017, trained on all 10 hurricanes of 2024 (a new season)
#: Folds: storms sorted by peak IBTrACS intensity and dealt round-robin, so
#: every fold has weak and strong storms.
N_FOLDS = {TRAIN_SEASON: 6, TEST_SEASON: 5}
EVALUATIONS = {
    "cv2017": f"{TRAIN_SEASON}, cross-validated",
    "cv2024": f"{TEST_SEASON}, cross-validated",
    "2017to2024": f"{TEST_SEASON}, trained on {TRAIN_SEASON}",
    "2024to2017": f"{TRAIN_SEASON}, trained on {TEST_SEASON}",
}
SPLITS = ([f"cv{TRAIN_SEASON}-{k}" for k in range(1, N_FOLDS[TRAIN_SEASON] + 1)]
          + [f"cv{TEST_SEASON}-{k}" for k in range(1, N_FOLDS[TEST_SEASON] + 1)]
          + ["2017to2024", "2024to2017"])


def cv_folds(season=TRAIN_SEASON):
    storms = load_storms(season)
    order = sorted(storms, key=lambda s: -np.nanmax(s.v_obs))
    n = N_FOLDS[season]
    return [sorted(s.name for s in order[k::n]) for k in range(n)]


def evaluation_of(split):
    """cv2017-3 -> cv2017; 2017to2024 -> 2017to2024."""
    return split.split("-")[0]


def season_of(split):
    """Season of the storms a split forecasts."""
    ev = evaluation_of(split)
    return int(ev[2:]) if ev.startswith("cv") else int(ev.split("to")[1])


def load_split(split, feature_set="climatology", chi_calibration="tc_risk"):
    """(train storms, test storms, feature scaler fitted on the training storms only)."""
    kw = dict(feature_set=feature_set, chi_calibration=chi_calibration)
    ev = evaluation_of(split)
    if ev.startswith("cv"):
        season = int(ev[2:])
        held = set(cv_folds(season)[int(split.split("-")[1]) - 1])
        storms = load_storms(season, **kw)
        train = [s for s in storms if s.name not in held]
        test = [s for s in storms if s.name in held]
    else:
        a, b = (int(x) for x in ev.split("to"))
        train, test = load_storms(a, **kw), load_storms(b, **kw)
    return train, test, FeatureScaler(train)


# --------------------------------------------------------------------------- #
# What is learned in training, and how
# --------------------------------------------------------------------------- #
#: name -> settings. "fast" (FAST's own alpha, nothing learned) is the baseline.
#: Variants without "evaluations" are run on all four evaluations.
VARIANTS = {
    "pinn_joint": dict(method="pinn", features="climatology", learn_m0=True, chi="tc_risk",
                       label="PINN: m0 + α-net"),
    "shooting_joint": dict(method="shooting", features="climatology", learn_m0=True, chi="tc_risk",
                           label="shooting: m0 + α-net"),
    "pinn_constant_alpha": dict(method="pinn", features="none", learn_m0=True, chi="tc_risk",
                                label="PINN: m0 + one constant α"),
    "pinn_alpha_only": dict(method="pinn", features="climatology", learn_m0=False, chi="tc_risk",
                            label="PINN: α-net, m0 fixed at FAST's", evaluations=("cv2017", "2017to2024")),
    "pinn_joint_oras5": dict(method="pinn", features="oras5", learn_m0=True, chi="tc_risk",
                             label="PINN: m0 + α-net on ORAS5", evaluations=("cv2017",)),
    "pinn_joint_rawchi": dict(method="pinn", features="climatology", learn_m0=True, chi="raw",
                              label="PINN: m0 + α-net, raw χ", evaluations=("2017to2024",)),
}
BASELINES = {"fast": dict(chi="tc_risk", label="FAST"), "fast_rawchi": dict(chi="raw", label="FAST, raw χ")}
LABEL = {**{k: v["label"] for k, v in VARIANTS.items()}, **{k: v["label"] for k, v in BASELINES.items()}}
M0_SOURCES = {"prior": "FAST's m0", "first24h": f"m0 from first {ASSIM_H} h"}


def color(variant, m0_source="prior"):
    """Colour of a forecast configuration (FAST with m0 from 24 h is light blue)."""
    if variant == "fast":
        return C["fast"] if m0_source == "prior" else C["fast_m0"]
    return C[variant]


def jobs():
    return [(split, name) for split in SPLITS for name, v in VARIANTS.items()
            if evaluation_of(split) in v.get("evaluations", EVALUATIONS)]


# --------------------------------------------------------------------------- #
# Scoring the forecasts written by run_calibrations.py
# --------------------------------------------------------------------------- #
def scored_forecasts():
    """forecasts.csv joined with the IBTrACS observations, only the scored times (t > 24 h)."""
    fc = pd.read_csv(RESULTS / "forecasts.csv")
    st = pd.read_csv(RESULTS / "storms.csv")
    st = st[st.scored][["season", "storm", "hour", "obs_kt"]]
    df = fc.merge(st, on=["season", "storm", "hour"])
    df["err_kt"] = df.vmax_kt - df.obs_kt
    df["lead_h"] = df.hour - ASSIM_H
    return df


def storm_scores(df):
    """Per storm: RMSE, bias and peak error over the scored times."""
    def one(g):
        ip, io = g.vmax_kt.idxmax(), g.obs_kt.idxmax()
        return pd.Series({"rmse_kt": np.sqrt(np.mean(g.err_kt ** 2)), "bias_kt": g.err_kt.mean(),
                          "peak_err_kt": g.vmax_kt[ip] - g.obs_kt[io], "n_obs": len(g)})
    keys = ["evaluation", "season", "split", "storm", "variant", "m0_source"]
    return df.groupby(keys).apply(one, include_groups=False).reset_index()


# --------------------------------------------------------------------------- #
# Colours (fixed per quantity) and plot style
# --------------------------------------------------------------------------- #
C = {"obs": "#0b0b0b", "fast": "#2a78d6", "fast_m0": "#86b6ec", "pinn_joint": "#eb6834", "shooting_joint": "#1baf7a",
     "pinn_constant_alpha": "#6250d6", "pinn_alpha_only": "#e87ba4", "pinn_joint_oras5": "#eda100",
     "pinn_joint_rawchi": "#9a9893", "fast_rawchi": "#9a9893", "vp": "#d62728"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9893"
#: evaluation -> marker: circle = 2017 storms, square = 2024 storms;
#: filled = trained on the same season, open = trained on the other season
EVAL_STYLE = {"cv2017": dict(marker="o", filled=True), "cv2024": dict(marker="s", filled=True),
              "2024to2017": dict(marker="o", filled=False), "2017to2024": dict(marker="s", filled=False)}


def paired_change(new, base, n_boot=10000, seed=0):
    """Mean of per-storm (new - base) and its 95 % bootstrap interval over storms."""
    d = np.asarray(new, float) - np.asarray(base, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return d.mean(), lo, hi


def forest(ax, rows, evaluations, color_of=lambda label: INK):
    """Horizontal interval plot. rows: [(label, {evaluation: (mean, lo, hi)})], top to bottom."""
    n = len(evaluations)
    offs = (np.arange(n)[::-1] - (n - 1) / 2) * 0.17
    y = np.arange(len(rows))[::-1]
    for yi, (label, vals) in zip(y, rows):
        col = color_of(label)
        for off, ev in zip(offs, evaluations):
            if ev not in vals:
                continue
            m, lo, hi = vals[ev]
            st = EVAL_STYLE[ev]
            ax.plot([lo, hi], [yi + off] * 2, color=col, lw=1.4, solid_capstyle="butt")
            ax.plot(m, yi + off, ls="none", marker=st["marker"], ms=7, mew=1.5, color=col,
                    mfc=col if st["filled"] else "white")
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.grid(axis="y", visible=False)
    for ev in evaluations:
        st = EVAL_STYLE[ev]
        ax.plot([], [], ls="none", marker=st["marker"], ms=7, mew=1.5, color=INK,
                mfc=INK if st["filled"] else "white", label=EVALUATIONS[ev])


def setup_torch(threads=8):
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)


def plt_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt


__all__ = ["ASSIM_H", "BASELINES", "C", "CALIB_DIR", "EVALUATIONS", "EVAL_STYLE", "FIGURES", "INK", "INK2", "LABEL",
           "M0_SOURCES", "MS_TO_KNOTS", "MUTED", "N_FOLDS", "RESULTS", "SPLITS", "TEST_SEASON", "TRAIN_SEASON",
           "VARIANTS", "color", "cv_folds", "evaluation_of", "forest", "jobs", "load_split", "paired_change",
           "plt_style", "scored_forecasts", "season_of", "setup_torch", "storm_scores"]
