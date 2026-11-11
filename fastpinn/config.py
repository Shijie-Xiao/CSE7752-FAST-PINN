"""Constants, inputs and defaults.

FAST (identical to fast/fast_equations.py and to the EAS 6130 repository)::

    dV/dt = kappa [ alpha beta Vp^2 m^3 - (1 - gamma m^3) V^2 ]
    dm/dt = kappa [ (1 - m) V - chi S m ]
    kappa = Ck / (2 h) * 3600 = 1.543e-3 (m/s)^-1 h^-1,   t in hours, V in m/s

beta = 1 - eps - k and gamma = eps + alpha k, where eps (thermodynamic
efficiency) and k (moisture term) come from the hourly data; alpha is the
ocean-coupling (cold-wake) factor, alpha = 1 - 0.87 exp(-z) in [0.13, 1].
The forcing of hour k is held constant on [k, k+1), as in FAST.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "storms"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

MS_TO_KNOTS = 1.94384
KAPPA = 0.5 * 1.2e-3 / 1400.0 * 3600.0
ALPHA_MIN, ALPHA_MAX = 0.13, 1.0

#: Wind-speed scale that normalises V in every loss term (proposal: V*).
V_STAR = 50.0
#: Natural time scale 1 / (kappa V*) ~ 13 h, normalises the ODE residuals.
T_STAR = 1.0 / (KAPPA * V_STAR)

#: IBTrACS observation cadence (the hourly series is interpolated between).
OBS_EVERY_H = 6

#: Ocean / storm-motion inputs o(t) of the alpha network g_phi: the
#: quantities FAST's alpha formula is built from (upper-ocean structure and
#: storm motion; Reviewer 4), plus Vp and a shallow-water/land indicator.
OCEAN_FEATURES = ("log_hm", "log_strat", "log_depth", "u_T", "vp_norm", "shallow")

#: Hold-out storms (2017): intense, landfalling, fast-moving, high-latitude.
TEST_STORMS = ("HARVEY", "IRMA", "MARIA", "NATE", "OPHELIA")
