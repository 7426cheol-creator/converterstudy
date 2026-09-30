"""Textbook ch.06 / E08 closed forms (FL03, EX08): RC step, electro-thermal fixed point,
sinusoidal-PWM loss averages, 2x2 thermal matrix.  Comparison only; lab code never imports this.
"""

from __future__ import annotations

import math


def rc_step(P: float, R: float, tau: float, Tb: float, t: float) -> float:
    """Single thermal node: T(t) = T_b + P R (1 - exp(-t/tau))."""
    return Tb + P * R * (1.0 - math.exp(-t / tau))


def electrothermal_rise(P0: float, R: float, alpha: float) -> float:
    """P = P0 [1 + alpha (T - T_b)]: dT = R P0 / (1 - alpha R P0); no stable solution if the denominator <= 0."""
    den = 1.0 - alpha * R * P0
    if den <= 0:
        return math.inf
    return R * P0 / den


def pwm_conduction_forward(Ipk: float, R: float, m: float, cosphi: float) -> float:
    """Upper switch, positive half-wave (forward channel), duty 0.5(1 + m sin(wt + phi)), i = I_pk sin(wt)."""
    return Ipk**2 * R * (1.0 / 8.0 + m * cosphi / (3.0 * math.pi))


def pwm_conduction_reverse(Ipk: float, R: float, m: float, cosphi: float) -> float:
    """Upper switch, negative half-wave (synchronous reverse channel), same duty, no dead time."""
    return Ipk**2 * R * (1.0 / 8.0 - m * cosphi / (3.0 * math.pi))


def pwm_switching_linear_E(fs: float, k_E: float, Ipk: float) -> float:
    """Hard switching in the positive half-wave with E = k_E |i|: f_s k_E I_pk / pi."""
    return fs * k_E * Ipk / math.pi


def coupled_rise_2x2(Z, P0, alpha):
    """[I - Z diag(alpha P0)] dT = Z P0 solved by Cramer's rule (2 x 2)."""
    a, b = alpha * P0[0], alpha * P0[1]
    m11, m12 = 1 - Z[0][0] * a, -Z[0][1] * b
    m21, m22 = -Z[1][0] * a, 1 - Z[1][1] * b
    r1 = Z[0][0] * P0[0] + Z[0][1] * P0[1]
    r2 = Z[1][0] * P0[0] + Z[1][1] * P0[1]
    det = m11 * m22 - m12 * m21
    return ((r1 * m22 - m12 * r2) / det, (m11 * r2 - m21 * r1) / det)


def spectral_radius_2x2(M) -> float:
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    disc = tr * tr / 4 - det
    if disc >= 0:
        return max(abs(tr / 2 + math.sqrt(disc)), abs(tr / 2 - math.sqrt(disc)))
    return math.sqrt(det)
