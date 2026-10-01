"""Closed forms for EX01 (textbook E01, E12 capstones A-C): used only for comparison.

The lab computes the same quantities on another path (impedance algebra on a frequency
grid with vectorised bisection, fixed-point iteration, piecewise-linear segment sums,
energy sums); these functions are the independent side of its checks.
"""

from __future__ import annotations

import math

import numpy as np


def grid_available_power(V_LL: float, I_line: float, PF: float, eta: float) -> float:
    """Battery-side power a three-phase input can deliver: sqrt(3) V_LL I PF eta (textbook E12 A: 10.6988 kW)."""
    return math.sqrt(3.0) * V_LL * I_line * PF * eta


def cllc_gain_abcd(f: float, n: float, V_bat: float, P: float, Lr: float, Cr: float, Lm: float) -> tuple[complex, complex]:
    """FHA gain V_o'/V_1 and input impedance by a chain (ABCD) matrix product.

    Primary series Z1 = jwLr + 1/(jwCr), shunt jwLm, primary-referred secondary series with the
    referred-symmetric tank (Lr2' = Lr, Cr2' = Cr), load R_ac' = n^2 8 R_L / pi^2.  A different
    formulation from the lab's divider algebra: [V1, I1] = S1 . Sh . S2 . [Vo, Io].
    """
    w = 2 * math.pi * f
    z = 1j * w * Lr + 1 / (1j * w * Cr)
    y = 1 / (1j * w * Lm)
    rac = n * n * 8.0 * (V_bat * V_bat / P) / math.pi**2
    series = np.array([[1.0, z], [0.0, 1.0]], dtype=complex)
    shunt = np.array([[1.0, 0.0], [y, 1.0]], dtype=complex)
    abcd = series @ shunt @ series
    vo = 1.0
    io = vo / rac
    v1, i1 = abcd @ np.array([vo, io])
    return vo / v1, v1 / i1


def static_share(I_total: float, R: list[float]) -> list[float]:
    """Resistive current division between parallel branches: I_k = I G_k / sum G."""
    g = [1.0 / r for r in R]
    s = sum(g)
    return [I_total * gk / s for gk in g]


def sine_power_average(alpha: float) -> float:
    """(1/2pi) integral_0^pi sin(theta)^alpha d theta = Gamma((alpha+1)/2) / (2 sqrt(pi) Gamma(alpha/2 + 1))."""
    return math.gamma((alpha + 1.0) / 2.0) / (2.0 * math.sqrt(math.pi) * math.gamma(alpha / 2.0 + 1.0))


def mission_efficiency_harmonic(eta: list[float], E_out: list[float]) -> float:
    """Energy efficiency as the E_out-weighted harmonic mean of point efficiencies: sum E_out / sum (E_out / eta)."""
    return sum(E_out) / sum(e / h for e, h in zip(E_out, eta))


def u_difference(u1: float, u2: float, rho: float = 0.0) -> float:
    return math.sqrt(max(u1 * u1 + u2 * u2 - 2.0 * rho * u1 * u2, 0.0))
