"""THE FAST EQUATIONS -- the only place where the physics of FAST is written.

FAST (Emanuel 2017, "A fast intensity simulator for tropical cyclone risk
analysis") couples the maximum azimuthal wind V(t) and the non-dimensional
inner-core moisture m(t):

    dV/dt = kappa * [ alpha * beta * Vp^2 * m^3  -  (1 - gamma * m^3) * V^2 ]    (1)
    dm/dt = kappa * [ (1 - m) * V  -  chi*S * m ]                                  (2)

    kappa = Ck / (2 h) * 3600       [ (m/s)^-1 per hour ]                         (3)

Everything below is a line-by-line transcription of FAST's
``run_fast_reference.py`` (``_physics_rhs_v``, ``_physics_rhs_m``,
``_coeff_from_cd``, ``calculate_m0_from_fast_numpy``), with t in hours,
V and Vp in m/s:

* Ck = Cd = 1.2e-3, h = 1400 m (North Atlantic boundary layer)
  -> kappa = 0.5 * 1.2e-3 / 1400 * 3600 = 1.542857e-3
* alpha, beta, gamma, Vp are the hourly ``scalars`` of the dataset;
  beta = 1 - eps - k, gamma = eps + alpha k (thermodynamic), alpha = ocean
  cold-wake factor 1 - 0.87 exp(-z) (see "ODEFAST Parameter Calculation").
* chi*S = chi_ref * s_ref, NaN -> 1e-5 and floored at 1e-5 (FAST's
  ``XS_NAN_FALLBACK``, so m can still decay over land where Vp = 0).
* Hour k's forcing is held constant on [k, k+1): FAST takes 4 sub-steps per
  hour, all with ``scalars[k]`` and ``xs[k]``.

A forcing row is ``F = (kappa, alpha, beta, gamma, vp, xs)``.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (FAST / tropical_cyclone_risk namelist)
# --------------------------------------------------------------------------- #
CD = 1.2e-3                  #: drag = enthalpy exchange coefficient (Ck = Cd)
H_BL = 1400.0                #: boundary-layer depth for the North Atlantic, m
KAPPA = 0.5 * CD / H_BL * 3600.0   #: Eq. (3), (m/s)^-1 h^-1
XS_FLOOR = 1e-5              #: floor on chi*S (FAST XS_NAN_FALLBACK)
MS_TO_KNOTS = 1.94384
V_STAR = 50.0                #: wind scale used to compare V and m in one norm

#: Column order of a forcing row / forcing table.
FORCING_FIELDS = ("kappa", "alpha", "beta", "gamma", "vp", "xs")


def coeff_from_cd(cd=CD, h_bl=H_BL):
    """kappa = 0.5 * Cd / h_bl * 3600  (FAST ``_coeff_from_cd``)."""
    return 0.5 * float(cd) / float(h_bl) * 3600.0


def ventilation(chi, s):
    """chi*S with FAST's NaN handling and floor (``process_one_pkl``)."""
    xs = np.nan_to_num(np.asarray(chi, float) * np.asarray(s, float), nan=XS_FLOOR)
    return np.maximum(xs, XS_FLOOR)


# --------------------------------------------------------------------------- #
# Right-hand side, Eq. (1)-(2)
# --------------------------------------------------------------------------- #
def rhs(y, F):
    """dy/dt for y = (V, m) and one forcing row F. Returns a length-2 array."""
    k, a, b, g, vp, xs = F
    V, m = y[0], y[1]
    m3 = m * m * m
    dV = k * (a * b * vp * vp * m3 - (1.0 - g * m3) * V * V)
    dm = k * ((1.0 - m) * V - xs * m)
    return np.array([dV, dm])


def jacobian(y, F):
    """d(rhs)/dy, 2 x 2 (proposal Eq. 6).

        J = kappa * [[ -2 (1 - gamma m^3) V ,  3 m^2 (alpha beta Vp^2 + gamma V^2) ],
                     [        1 - m         ,           -(V + chi S)             ]]

    For 0 <= m <= 1 both off-diagonal entries are >= 0 (a Metzler matrix), so
    the eigenvalues are always real: FAST has no oscillatory modes.
    """
    k, a, b, g, vp, xs = F
    V, m = y[0], y[1]
    m2 = m * m
    m3 = m2 * m
    return k * np.array([[-2.0 * (1.0 - g * m3) * V, 3.0 * m2 * (a * b * vp * vp + g * V * V)],
                         [1.0 - m, -(V + xs)]])


# --------------------------------------------------------------------------- #
# Initial moisture (FAST ``calculate_m0_from_fast_numpy``)
# --------------------------------------------------------------------------- #
def initial_moisture(V0, dVdt, F):
    """Invert Eq. (1) for m given V and dV/dt:

        m^3 = (dV/dt / kappa + V^2) / (alpha beta Vp^2 + gamma V^2),

    clipped to [0.01, 1] as in FAST.
    """
    k, a, b, g, vp, xs = F
    num = dVdt / (k + 1e-12) + V0 ** 2
    den = a * b * vp ** 2 + g * V0 ** 2
    m3 = max(num / (den + 1e-8), 0.0)
    return float(np.clip(m3 ** (1.0 / 3.0), 0.01, 1.0))


# --------------------------------------------------------------------------- #
# Equilibria and analytic special case
# --------------------------------------------------------------------------- #
def riccati_solution(t, V0, F):
    """Exact V(t) for constant forcing, m = 1 and chi*S = 0 (proposal Eq. 10).

    Then dm/dt = 0 and dV/dt = a - b V^2 with a = kappa alpha beta Vp^2,
    b = kappa (1 - gamma); V(t) = Vinf (V0 + Vinf tanh(b Vinf t)) /
    (Vinf + V0 tanh(b Vinf t)), Vinf = sqrt(a / b).
    """
    k, a_, b_, g, vp, xs = F
    a = k * a_ * b_ * vp ** 2
    b = k * (1.0 - g)
    vinf = np.sqrt(a / b)
    th = np.tanh(b * vinf * np.asarray(t, float))
    return vinf * (V0 + vinf * th) / (vinf + V0 * th)
