#!/usr/bin/env python3
"""Run all experiments in order (~25 min on a laptop CPU).

    python experiments/run_all.py          # all
    python experiments/run_all.py 2 3      # selected
"""

from __future__ import annotations

import sys
import time
import warnings

import exp1_pinn_solver
import exp2_twin
import exp3_real_storms

STEPS = {
    "1": ("PINN as a FAST solver", exp1_pinn_solver.main),
    "2": ("twin experiments (recovery and compensation)", exp2_twin.main),
    "3": ("real storms: calibration and hold-out prediction", exp3_real_storms.main),
}

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    for key in sys.argv[1:] or STEPS:
        name, fn = STEPS[key]
        print(f"\n=== Experiment {key}: {name} ===", flush=True)
        tic = time.perf_counter()
        fn()
        print(f"--- {time.perf_counter() - tic:.0f} s", flush=True)
