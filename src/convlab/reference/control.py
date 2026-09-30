"""Textbook ch.09 / E07 closed forms for current-loop PI design (comparison only).

RL plant L di/dt = u - R i - e with the PI zero placed on the plant pole:
    Kp = L w_c  [V/A],  Ki = R w_c  [V/(A s)]   (controller output in volts)
    loop gain  L(s) = (Kp + Ki/s) / (L s + R) = w_c / s
A duty-output controller divides both gains by the modulator gain K_PWM [V per unit duty].
A pure delay T_d adds a phase lag of 360 f T_d degrees at frequency f.
"""

from __future__ import annotations

import math


def pi_gains_rl(L: float, R: float, fc: float) -> tuple[float, float]:
    """(Kp [V/A], Ki [V/(A s)]) for crossover fc with plant-pole cancellation."""
    wc = 2.0 * math.pi * fc
    return L * wc, R * wc


def duty_gains(Kp: float, Ki: float, K_pwm: float) -> tuple[float, float]:
    """Duty-output gains [1/A], [1/(A s)] for a modulator u = K_pwm * d."""
    return Kp / K_pwm, Ki / K_pwm


def delay_phase_deg(f: float, Td: float) -> float:
    """Phase lag of exp(-s Td) at frequency f, in degrees."""
    return 360.0 * f * Td


def pm_cancelled_pi_deg(fc: float, Td: float) -> float:
    """Phase margin of w_c/s * exp(-s Td) (continuous approximation of the delayed loop)."""
    return 90.0 - delay_phase_deg(fc, Td)


def first_order_step(t: float, fc: float) -> float:
    """Normalised closed-loop step of the ideal (delay-free) cancelled design: 1 - exp(-w_c t)."""
    return 1.0 - math.exp(-2.0 * math.pi * fc * t) if t > 0 else 0.0
