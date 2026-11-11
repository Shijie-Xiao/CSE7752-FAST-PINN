"""Shared helpers for the experiments: paths, seeding, splits, plot style."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fastpinn.config import FIGURES, RESULTS, TEST_STORMS  # noqa: E402
from fastpinn.data import FeatureScaler, load_storms  # noqa: E402

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# fixed categorical colours (one per estimator)
C = {"obs": "#0b0b0b", "prior": "#2a78d6", "pinn": "#eb6834", "shooting": "#1baf7a",
     "truth": "#0b0b0b", "m0": "#e87ba4", "alpha": "#6250d6"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9893"

#: PINN architecture used everywhere
PINN_KW = dict(width=64, depth=3, n_fourier=32, n_colloc=512)


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_torch(threads=8):
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)


def split(test=TEST_STORMS):
    storms = load_storms()
    train = [s for s in storms if s.name not in test]
    test_ = [s for s in storms if s.name in test]
    return train, test_, FeatureScaler(train)


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
