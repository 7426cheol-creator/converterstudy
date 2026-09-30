"""Textbook ch.02 / ch.08 closed forms for OBC and PFC input sizing (comparison only).

All voltages are RMS unless stated; V_LL is line-to-line RMS, PF is the true power factor
(real power / apparent power), eta the conversion efficiency from grid to battery.
"""

from __future__ import annotations

import math


def line_current(P_bat: float, V_LL: float, eta: float, PF: float) -> float:
    """Three-phase line current RMS for a battery output power P_bat."""
    return P_bat / (math.sqrt(3.0) * V_LL * eta * PF)


def battery_power(I_line: float, V_LL: float, eta: float, PF: float) -> float:
    """Battery output power available at a line current I_line (RMS)."""
    return math.sqrt(3.0) * V_LL * I_line * eta * PF


def phase_rms(V_LL: float) -> float:
    return V_LL / math.sqrt(3.0)


def spwm_index(V_ph_rms: float, V_dc: float) -> float:
    """Sinusoidal PWM modulation index m = V_ph,peak / (V_dc/2) = 2 sqrt(2) V_ph,rms / V_dc."""
    return 2.0 * math.sqrt(2.0) * V_ph_rms / V_dc


SVPWM_LIMIT = 2.0 / math.sqrt(3.0)  # same normalisation (V_ph,peak / (V_dc/2)), ideal linear limit


def pf_from_thd(cos_phi1: float, thd: float) -> float:
    """PF = cos(phi_1) / sqrt(1 + THD^2): sinusoidal voltage, no DC component."""
    return cos_phi1 / math.sqrt(1.0 + thd * thd)


def ripple_capacitance(P: float, f_line: float, V_dc: float, dV_pp: float) -> float:
    """Single-phase 2-omega ripple, small-ripple approximation: dV_pp = P / (omega C V_dc)."""
    return P / (2.0 * math.pi * f_line * V_dc * dV_pp)


def ripple_pp(P: float, f_line: float, C: float, V_dc: float) -> float:
    return P / (2.0 * math.pi * f_line * C * V_dc)


def holdup_capacitance(P: float, dt: float, V_hi: float, V_lo: float) -> float:
    """Ideal energy bound C >= 2 P dt / (V_hi^2 - V_lo^2)."""
    return 2.0 * P * dt / (V_hi**2 - V_lo**2)


def cap_energy_delta(C: float, V_hi: float, V_lo: float) -> float:
    return 0.5 * C * (V_hi**2 - V_lo**2)
