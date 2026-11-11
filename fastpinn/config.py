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

#: Inputs o(t) of the alpha network g_phi, by source.
#: * climatology: the quantities FAST's own alpha formula is built from --
#:   monthly-climatology mixed-layer depth and stratification (FHLO
#:   precalc_data), water depth -- plus storm motion, Vp and a land flag.
#: * oras5: the ORAS5 ocean reanalysis for the actual month (multi-level
#:   temperature -> SST, mixed-layer depth, T100, D26, TCHP; see
#:   scripts/add_oras5_inputs.py) plus the same storm-motion / Vp / land inputs.
#: * both: union of the two.
_MOTION = ("u_T", "vp_norm", "shallow")
FEATURE_SETS = {
    "climatology": ("log_hm", "log_strat", "log_depth") + _MOTION,
    "oras5": ("o5_sst", "log_o5_mld", "o5_t100", "o5_d26", "o5_tchp") + _MOTION,
    "both": ("log_hm", "log_strat", "log_depth", "o5_sst", "log_o5_mld", "o5_t100", "o5_d26",
             "o5_tchp") + _MOTION,
}
DEFAULT_FEATURES = "climatology"
OCEAN_FEATURES = FEATURE_SETS[DEFAULT_FEATURES]

#: Hold-out storms (2017): intense, landfalling, fast-moving, high-latitude.
TEST_STORMS = ("HARVEY", "IRMA", "MARIA", "NATE", "OPHELIA")
