"""Closed forms for EX09 (textbook E09): hand screens, trapezoid spectra and path impedances.

Only used for comparison; the lab computes the same quantities with an exact
time-domain engine, an FFT of simulated waveforms and a state-space frequency
response.  All networks are synthetic lumped relative proxies (ASSUMED values), not a
CISPR LISN or receiver.
"""

from __future__ import annotations

import math

import numpy as np


def displacement_peak(C: float, dvdt: float) -> float:
    """Peak displacement-current screen i = C dv/dt (SI: F, V/s)."""
    return C * dvdt


def ring_frequency(L: float, C: float) -> float:
    return 1.0 / (2.0 * math.pi * math.sqrt(L * C))


def damped_ring_frequency(L: float, C: float, R: float) -> float:
    """Series RLC: f_d = f_0 sqrt(1 - zeta^2), zeta = R / (2 sqrt(L/C))."""
    z = R / (2.0 * math.sqrt(L / C))
    return ring_frequency(L, C) * math.sqrt(max(0.0, 1.0 - z * z))


def line_capacitor_current(C: float, Vrms: float, f: float) -> float:
    """RMS current of one ideal capacitor at a sinusoidal line voltage: 2 pi f C V."""
    return 2.0 * math.pi * f * C * Vrms


def trapezoid_harmonics(V: float, f: float, D: float, tr: float, n: np.ndarray) -> np.ndarray:
    """One-sided amplitudes of a 0..V trapezoid, equal rise/fall tr, pulse width D/f at half height.

    A_n = 2 V D |sinc(n D)| |sinc(n f tr)|, sinc(x) = sin(pi x)/(pi x).
    """
    n = np.asarray(n, dtype=float)
    return 2.0 * V * D * np.abs(np.sinc(n * D)) * np.abs(np.sinc(n * f * tr))


def trapezoid_envelope(V: float, f: float, D: float, tr: float, freqs: np.ndarray) -> np.ndarray:
    """Bode-style bound: flat 2VD, -20 dB/dec above 1/(pi tau), -40 dB/dec above 1/(pi tr), tau = D/f."""
    tau = D / f
    f1 = 1.0 / (math.pi * tau)
    f2 = 1.0 / (math.pi * tr)
    fr = np.asarray(freqs, dtype=float)
    env = 2.0 * V * D * np.ones_like(fr)
    env = np.where(fr > f1, 2.0 * V * D * f1 / fr, env)
    env = np.where(fr > f2, 2.0 * V * D * f1 / f2 * (f2 / fr) ** 2, env)
    return env


def par(a: complex, b: complex) -> complex:
    return a * b / (a + b)


def z_meas(f: float, Rm: float, Lm: float) -> complex:
    """Per-line measurement network proxy: Rm parallel Lm (ASSUMED '50 ohm || 50 uH')."""
    w = 2 * math.pi * f
    return par(Rm, 1j * w * Lm)


def h_cm(f: float, Cpar: float, Rpar: float, Lcab: float, Rm: float, Lm: float, Cy: float) -> complex:
    """CM transfer V_meas / V_node by impedance algebra.

    Node -> (R_par + C_par) -> chassis; chassis returns to the DC bus through the Y
    capacitors (2 C_Y in parallel, local path) and through the cable (L_cab/2 for two lines
    in parallel) in series with the two measurement branches in parallel (Z_m/2).
    V_meas is the voltage across one 50-ohm port: (i_line) * Z_m with i_line = i_ext/2.
    """
    w = 2 * math.pi * f
    zc = Rpar + 1 / (1j * w * Cpar)
    zm = z_meas(f, Rm, Lm)
    zext = 1j * w * Lcab / 2 + zm / 2
    zret = par(zext, 1 / (1j * w * 2 * Cy)) if Cy > 0 else zext
    i_cm = 1.0 / (zc + zret)
    v_x = i_cm * zret
    i_ext = v_x / zext
    return i_ext / 2 * zm


def h_dm(f: float, Cdc: float, Resr: float, Lesl: float, Lcab: float, Rm: float, Lm: float) -> complex:
    """DM transfer V_meas / I_switch (ohm): the switching-cell current splits between the DC-link
    capacitor branch and the line path (L_cab on each line + both measurement branches in series)."""
    w = 2 * math.pi * f
    zdc = Resr + 1j * w * Lesl + 1 / (1j * w * Cdc)
    zm = z_meas(f, Rm, Lm)
    zline = 2 * (1j * w * Lcab) + 2 * zm
    i_line = zdc / (zdc + zline)
    return i_line * zm


def overlap_switching_energy(V: float, I: float, tr: float, tf: float) -> float:
    """Linear V-I overlap (synthetic): E = V I (tr + tf) / 2 per period (one on + one off)."""
    return 0.5 * V * I * (tr + tf)


def rc_snubber_loss(Cs: float, V: float, fsw: float) -> float:
    """RC snubber across a hard-switched node: C_s V^2 f_sw (two edges, each 1/2 C V^2 in R)."""
    return Cs * V * V * fsw
