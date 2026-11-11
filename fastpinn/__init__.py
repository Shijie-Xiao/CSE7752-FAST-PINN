"""fastpinn: latent parameterization of the FAST hurricane intensity model.

CSE 7752 final project (Team 4). A physics-informed neural network infers the
unobserved initial inner-core moisture m0 (per storm) and a shared
ocean-coupling network alpha(t) = g_phi(o(t)); the calibrated quantities are
then passed to the FAST ODE solver. Physics follows FAST-ML
(github.com/Shijie-Xiao/FAST-ML) and FHLO (github.com/Shijie-Xiao/FHLO).
"""

from .calibrate import (LatentParams, PINNCalibrator, ShootingCalibrator, Weights,
                        pretrain_alpha_to_prior, train_pinn, train_shooting)
from .data import FeatureScaler, Storm, available_storms, load_storm, load_storms, make_batch
from .evaluate import solve_fast, storm_metrics, summarize
from .forecast import assimilate_m0, fit, forecast
from .networks import AlphaNet, TrajectoryNet
from .physics import fast_rhs, obs_operator, rk4_integrate

__all__ = [
    "LatentParams", "PINNCalibrator", "ShootingCalibrator", "Weights", "pretrain_alpha_to_prior",
    "train_pinn", "train_shooting", "FeatureScaler", "Storm", "available_storms", "load_storm",
    "load_storms", "make_batch", "solve_fast", "storm_metrics", "summarize", "AlphaNet",
    "TrajectoryNet", "fast_rhs", "obs_operator", "rk4_integrate", "fit", "assimilate_m0", "forecast",
]
