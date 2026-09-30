"""Textbook E07 closed forms: source R-L, node capacitor C and an ideal constant-power load.

    L di/dt = Vs - R i - v,     C dv/dt = i - P/v
High-voltage equilibrium V_e = (Vs + sqrt(Vs^2 - 4 R P)) / 2 (exists only for P <= Vs^2/(4R)),
g = P/V_e^2 (negative incremental conductance of the load is -g),
A = [[-R/L, -1/L], [1/C, g/C]]: stable iff tr A < 0 and det A > 0, i.e. C > L g / R and R g < 1.
"""

from __future__ import annotations

import cmath
import math


def p_max(Vs: float, R: float) -> float:
    return Vs * Vs / (4.0 * R)


def equilibria(Vs: float, R: float, P: float) -> tuple[float, float] | None:
    """(V_high, V_low) or None when P > Vs^2/(4R)."""
    disc = Vs * Vs - 4.0 * R * P
    if disc < 0:
        return None
    r = math.sqrt(disc)
    return (Vs + r) / 2.0, (Vs - r) / 2.0


def incremental_resistance(V: float, P: float) -> float:
    return -V * V / P


def c_crit(Vs: float, R: float, L: float, P: float) -> float:
    Ve = equilibria(Vs, R, P)[0]
    g = P / Ve**2
    return L * g / R


def poles(Vs: float, R: float, L: float, C: float, P: float, branch: int = 0) -> tuple[complex, complex]:
    Ve = equilibria(Vs, R, P)[branch]
    g = P / Ve**2
    tr = -R / L + g / C
    det = (1.0 - R * g) / (L * C)
    d = cmath.sqrt(tr * tr / 4.0 - det)
    return tr / 2.0 + d, tr / 2.0 - d
