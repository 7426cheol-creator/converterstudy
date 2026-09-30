"""Textbook ch.04-05 closed forms (FL02): loss comparison, gate/loop screens, protection timeline.

Hand-calculation formulas only; lab code never imports this module, tests and checks
compare against it.
"""

from __future__ import annotations

import math


def device_loss(Irms: float, R: float, E: float, fs: float) -> float:
    """Conduction I_rms^2 R plus one (E_on + E_off) event pair per period."""
    return Irms**2 * R + E * fs


def crossover_fs(Irms: float, RA: float, EA: float, RB: float, EB: float) -> float:
    """Switching frequency at which the two synthetic devices have equal loss."""
    return Irms**2 * (RA - RB) / (EB - EA)


def overshoot(L: float, didt: float) -> float:
    return L * didt


def displacement_current(C: float, dvdt: float) -> float:
    return C * dvdt


def gate_current(Vdrv: float, Vpl: float, Rtot: float) -> float:
    return (Vdrv - Vpl) / Rtot


def miller_time(Qgd: float, Ig: float) -> float:
    return Qgd / Ig


def rc_time(Vdrv: float, V0: float, V1: float, R: float, C: float) -> float:
    """Time for an RC gate node to go from V0 to V1 towards Vdrv."""
    return R * C * math.log((Vdrv - V0) / (Vdrv - V1))


def gate_power(Qg: float, dV: float, fs: float) -> float:
    return Qg * dV * fs


def protection_chain(*stages: float) -> float:
    return float(sum(stages))


def parallel_loss(N: int, Irms: float, R: float, E: float, k_cap: float, Qg: float, dVg: float, fs: float) -> dict:
    """N identical devices sharing I_rms: conduction ~ 1/N, capacitive part of E and gate charge ~ N."""
    cond = Irms**2 * R / N
    sw = E * fs * ((1 - k_cap) + k_cap * N)
    gate = N * Qg * dVg * fs
    return {"cond": cond, "sw": sw, "gate": gate, "device_total": cond + sw}
