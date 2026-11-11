"""Differentiable FAST physics in PyTorch (same equations as fast/fast_equations.py).

* :func:`fast_rhs`       -- FAST right-hand side
* :func:`obs_operator`   -- FAST's axi_to_max_wind: azimuthal-mean V -> maximum
                            surface wind (translation asymmetry; the data have no
                            environmental winds, so the shear term is zero, as FAST
                            does in that case). Prescribed, not learned.
* :func:`rk4_integrate`  -- vectorised RK4 over a padded batch; differentiable in
                            m0 and alpha(t). The forcing of hour k is held constant
                            on [k, k+1) (FAST's convention), so RK4 with steps that
                            divide one hour keeps its 4th order. The EAS 6130 study
                            shows RK4 at dt = 1 h has error < 1e-3 kt in V_max on
                            these storms; it is the shooting baseline and the solver
                            used to evaluate every calibration.
"""

from __future__ import annotations

import torch


def fast_rhs(V, m, kappa, alpha, beta, gamma, vp, xs):
    m3 = m ** 3
    dV = kappa * (alpha * beta * vp ** 2 * m3 - (1.0 - gamma * m3) * V ** 2)
    dm = kappa * ((1.0 - m) * V - xs * m)
    return dV, dm


def obs_operator(V, lat, u_tran, v_tran):
    """Torch port of FAST ``axi_to_max_wind`` (no shear term)."""
    G = torch.clamp(0.8 + 0.35 * (1.0 + torch.tanh((lat.abs() - 35.0) / 10.0)), max=1.0)
    U, W = G * u_tran, G * v_tran
    mag = torch.sqrt(U ** 2 + W ** 2 + 1e-12)
    fac = torch.minimum(torch.ones_like(V), 0.5 * V / mag)
    th = torch.atan2(-U, W)
    ug = V * (-torch.sin(th)) + U * fac
    vg = V * torch.cos(th) + W * fac
    return torch.sqrt(ug ** 2 + vg ** 2 + 1e-12)


def hourly_forcing(batch, alpha):
    """Forcing tensors (B, T): kappa, alpha, beta, gamma, vp, xs."""
    from .config import KAPPA
    return (KAPPA, alpha, batch.beta, batch.gamma(alpha), batch.vp, batch.xs)


def rk4_integrate(batch, m0, alpha, substeps: int = 1):
    """Integrate every storm from (V0, m0); returns V, m of shape (B, T) at the hours."""
    kappa, a, b, g, vp, xs = hourly_forcing(batch, alpha)
    T = batch.vp.shape[1]
    dt = 1.0 / substeps
    V, m = batch.V0.clone(), m0
    Vs, ms = [V], [m]
    for t in range(T - 1):
        F = (kappa, a[:, t], b[:, t], g[:, t], vp[:, t], xs[:, t])
        for _ in range(substeps):
            k1V, k1m = fast_rhs(V, m, *F)
            k2V, k2m = fast_rhs(V + 0.5 * dt * k1V, m + 0.5 * dt * k1m, *F)
            k3V, k3m = fast_rhs(V + 0.5 * dt * k2V, m + 0.5 * dt * k2m, *F)
            k4V, k4m = fast_rhs(V + dt * k3V, m + dt * k3m, *F)
            V = V + dt / 6.0 * (k1V + 2 * k2V + 2 * k3V + k4V)
            m = m + dt / 6.0 * (k1m + 2 * k2m + 2 * k3m + k4m)
        Vs.append(V)
        ms.append(m)
    return torch.stack(Vs, 1), torch.stack(ms, 1)


def at_hour(x, t):
    """Piecewise-constant lookup x[:, floor(t)] for times t (B, N): FAST's hourly forcing."""
    T = x.shape[1]
    i = t.detach().floor().long().clamp(0, T - 1)
    return torch.gather(x, 1, i)
