#!/usr/bin/env python3
"""Run everything in order (about 2.5 h on a 24-core CPU).

    python experiments/run_all.py          # all
    python experiments/run_all.py 1 2      # only the analysis of Experiments 1 and 2

Step 0 (run_calibrations) fits m0 and alpha on every split; Experiments 1-4
only read its results. Experiments 5 and 6 are independent.
"""

from __future__ import annotations

import sys
import time
import warnings

import exp1_forecast_skill
import exp2_hurricanes_2024
import exp3_initial_moisture
import exp4_ocean_coupling
import exp5_twin_recovery
import exp6_pinn_solver
import run_calibrations

STEPS = {
    "0": ("fit m0 and alpha on every split (expensive)", run_calibrations.main),
    "1": ("does learning m0 and alpha improve FAST on unseen storms?", exp1_forecast_skill.main),
    "2": ("the 2024 hurricanes, trained on 2017", exp2_hurricanes_2024.main),
    "3": ("is m0 worth learning?", exp3_initial_moisture.main),
    "4": ("is the learned alpha useful and physical?", exp4_ocean_coupling.main),
    "5": ("twin experiment: recovering a known m0 and alpha", exp5_twin_recovery.main),
    "6": ("the PINN as a FAST solver", exp6_pinn_solver.main),
}

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    keys = sys.argv[1:] or list(STEPS)
    sys.argv = sys.argv[:1]          # the steps parse their own (default) arguments
    for key in keys:
        name, fn = STEPS[key]
        print(f"\n=== {key}: {name} ===", flush=True)
        tic = time.perf_counter()
        fn()
        print(f"--- {time.perf_counter() - tic:.0f} s", flush=True)
