"""Closed forms for the textbook's synthetic nonlinear output capacitance (E02).

C(v) = C0 / sqrt(1 + v/V0)
Q(v) = int_0^v C du = 2 C0 V0 (sqrt(1 + v/V0) - 1)
E(v) = int_0^v u C(u) du = C0 V0^2 [ 2/3 (1+v/V0)^{3/2} - 2 (1+v/V0)^{1/2} + 4/3 ]
"""

from __future__ import annotations

import math


def C(v, C0, V0):
    return C0 / math.sqrt(1.0 + v / V0)


def Q(v, C0, V0):
    return 2.0 * C0 * V0 * (math.sqrt(1.0 + v / V0) - 1.0)


def E(v, C0, V0):
    u = 1.0 + v / V0
    return C0 * V0 * V0 * (2.0 / 3.0 * u**1.5 - 2.0 * u**0.5 + 4.0 / 3.0)


def co_tr(V, C0, V0):
    """Time/charge-equivalent capacitance Q(V)/V."""
    return Q(V, C0, V0) / V


def co_er(V, C0, V0):
    """Energy-equivalent capacitance 2 E(V) / V^2."""
    return 2.0 * E(V, C0, V0) / V**2


def hb_transition_time_constant_current(Vbus, I, C0, V0):
    """Fixed rails, two identical devices, constant injected current: t = 2 Q(Vbus) / |I|."""
    return 2.0 * Q(Vbus, C0, V0) / abs(I)


def hb_hard_turn_on_loss(Vbus, v1, C0, V0, Cpar=0.0):
    """Channel energy when the high side turns on with the node at v1 (fixed rails).

    Rail charge q = Q(Vbus) - Q(v1) + Cpar (Vbus - v1) delivered at Vbus, minus the change
    of stored energy (low side v1 -> Vbus, high side Vbus - v1 -> 0, parasitic v1 -> Vbus).
    For v1 = 0 and Cpar = 0 this is Vbus * Q(Vbus) (= C V^2 for a linear C, not 2 Eoss).
    """
    q = Q(Vbus, C0, V0) - Q(v1, C0, V0) + Cpar * (Vbus - v1)
    dE = (E(Vbus, C0, V0) - E(v1, C0, V0)) - E(Vbus - v1, C0, V0) + 0.5 * Cpar * (Vbus**2 - v1**2)
    return Vbus * q - dE
