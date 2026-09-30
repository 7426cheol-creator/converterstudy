"""Closed forms and an independent piecewise-linear path for FL11 (PSFB, DAB comparison, 12 V path).

Nothing in the switched simulation imports this module; the lab only uses it to
compare.  Conventions (textbook ch.14 / ch.11):

* PSFB: n = Np/Ns (per half winding for a centre tap), primary-referred leakage
  L_k, phase command phi = pulse width of v_AB = +-V_in per half period,
  D_cmd = phi/pi, ideal V_o = V_in D / n.
* DAB SPS: V1 = V_H, V2 = n V_L (primary-referred), phi in rad, primary lead ->
  positive HV->LV power, zero-DC half-wave antisymmetric reference solution.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from ..engine.pwl import PWL

# ----------------------------------------------------------------------------------------
# PSFB closed forms (ideal, lossless, CCM)
# ----------------------------------------------------------------------------------------


def psfb_ideal_vo(Vin: float, n: float, D: float) -> float:
    """Textbook seed: V_o = V_in D_eff / n (no leakage, ideal rectifier)."""
    return Vin * D / n


def psfb_duty_loss(Lk: float, fs: float, I: float, n: float, Vin: float) -> float:
    """Duty lost to rectifier commutation (per half period).

    The primary current has to swing from -I/n to +I/n through the leakage while
    the rectifier shorts the secondary: dt = 2 L_k I / (n V_in); per half period
    Delta D = dt / (T_s/2) = 4 L_k f_s I / (n V_in).  With I = i_Lo at the moment
    the bridge voltage reverses this is exact for the ideal periodic solution
    (L_k volt-second balance); with I = average I_o it is the usual approximation.
    """
    return 4.0 * Lk * fs * I / (n * Vin)


def psfb_equivalent_output_resistance(Lk: float, fs: float, n: float) -> float:
    """Duty loss acts like a lossless series resistance at the output: V_o = V_in D/n - R_eq I_o."""
    return 4.0 * Lk * fs / n**2


def psfb_pwl_steady(Vin: float, n: float, fs: float, phi: float, Lk: float, Lo: float, R: float) -> dict | None:
    """Independent path: ideal lossless PSFB with a constant output voltage (C_o -> infinity).

    Segment algebra per half period (all straight lines):
      commutation [0, t_c): n i_p from -I0 up with slope n V_in/L_k, i_Lo down with -V_o/L_o;
      power       [t_c, t_phi): i_Lo slope (V_in/n - V_o)/(L_k/n^2 + L_o), n i_p = i_Lo;
      freewheel   [t_phi, T/2): i_Lo slope -V_o/(L_k/n^2 + L_o), n i_p = i_Lo.
    Half-wave symmetry i_Lo(T/2) = I0 gives I0 explicitly for a given V_o; the load line
    mean(i_Lo) = V_o/R closes the problem (scalar root).  Returns None outside CCM or
    when the commutation does not finish inside the power interval.
    """
    T2 = 0.5 / fs
    tphi = phi / math.pi * T2
    Leq = Lk / n**2 + Lo

    def build(Vo):
        sP = (Vin / n - Vo) / Leq
        sF = -Vo / Leq
        den = Vo / Lo + sP
        tc = (sP * tphi + sF * (T2 - tphi)) / den
        K = n * Vin / Lk + Vo / Lo
        I0 = K * tc / 2.0
        return tc, I0, sP, sF

    def mean_minus_load(Vo):
        tc, I0, sP, sF = build(Vo)
        if tc <= 0 or tc >= tphi:
            return float("nan")
        I1 = I0 - Vo / Lo * tc
        I2 = I1 + sP * (tphi - tc)
        I3 = I2 + sF * (T2 - tphi)
        area = 0.5 * (I0 + I1) * tc + 0.5 * (I1 + I2) * (tphi - tc) + 0.5 * (I2 + I3) * (T2 - tphi)
        return area / T2 - Vo / R

    Vhi = Vin * phi / math.pi / n
    grid = np.linspace(1e-6 * Vhi, Vhi * (1 - 1e-9), 400)
    vals = [mean_minus_load(v) for v in grid]
    root = None
    for k in range(len(grid) - 1):
        a, b = vals[k], vals[k + 1]
        if math.isfinite(a) and math.isfinite(b) and a * b <= 0:
            root = brentq(mean_minus_load, grid[k], grid[k + 1], xtol=1e-14, rtol=1e-15)
            break
    if root is None:
        return None
    Vo = root
    tc, I0, sP, sF = build(Vo)
    I1 = I0 - Vo / Lo * tc
    I2 = I1 + sP * (tphi - tc)
    I3 = I2 + sF * (T2 - tphi)
    if min(I0, I1, I2, I3) <= 0:
        return None  # DCM: not covered by this path
    t = [0.0, tc, tphi, T2]
    iLo = PWL(t, [I0, I1, I2], [I1, I2, I3])
    nip = PWL(t, [-I0, I1, I2], [I1, I2, I3])  # n * i_p over the positive half period
    # full-period device waveforms (half-wave symmetry): SR pair 1 conducts (i_Lo + n i_p)/2
    tt = [0.0, tc, tphi, T2, T2 + tc, T2 + tphi, 2 * T2]
    sr1 = PWL(tt, [0.0, I1, I2, I0, 0.0, 0.0], [I1, I2, I3, 0.0, 0.0, 0.0])
    ip_full = PWL(tt, [-I0 / n, I1 / n, I2 / n, I0 / n, -I1 / n, -I2 / n], [I1 / n, I2 / n, I3 / n, -I1 / n, -I2 / n, -I3 / n])
    return {
        "Vo": Vo,
        "Io": Vo / R,
        "D_eff": n * Vo / Vin,
        "t_c": tc,
        "I0": I0,
        "I_valley": I1,
        "I_peak": I2,
        "dI_pp": I2 - I1,
        "iLo_mean": iLo.mean(),
        "iLo_rms": iLo.rms(),
        "ip_rms": ip_full.rms(),
        "ip_peak": ip_full.max_abs(),
        "sr_rms": sr1.rms(),
        "sr_peak": sr1.max(),
        "nip": nip,
        "iLo": iLo,
        "ip": ip_full,
        "sr1": sr1,
    }


# ----------------------------------------------------------------------------------------
# DAB SPS closed forms (textbook ch.11) and an independent PWL integration
# ----------------------------------------------------------------------------------------


def dab_power(V1: float, V2: float, fs: float, L: float, phi: float) -> float:
    w = 2 * math.pi * fs
    return V1 * V2 / (w * L) * phi * (1 - abs(phi) / math.pi)


def dab_phi_for_power(V1: float, V2: float, fs: float, L: float, P: float) -> float:
    """Smaller root of P = V1 V2 phi (1 - phi/pi) / (w L) (monotonic range |phi| <= pi/2)."""
    w = 2 * math.pi * fs
    x = 4 * P * w * L / (math.pi * V1 * V2)
    if x > 1:
        raise ValueError("power above the SPS maximum V1 V2 / (8 fs L)")
    return math.pi / 2 * (1 - math.sqrt(1 - x))


def dab_currents(V1: float, V2: float, fs: float, L: float, phi: float) -> dict:
    """Textbook i_0, i_phi, peak and RMS of the series-inductor current (zero-DC reference)."""
    w = 2 * math.pi * fs
    a = (V1 + V2) / (w * L)
    b = (V1 - V2) / (w * L)
    i0 = -(a * phi + b * (math.pi - phi)) / 2
    iphi = i0 + a * phi
    ipk = max(abs(i0), abs(iphi))
    # exact RMS of the two straight segments over the half period
    rms2 = (phi * (i0 * i0 + i0 * iphi + iphi * iphi) + (math.pi - phi) * (iphi * iphi + iphi * (-i0) + i0 * i0)) / (3 * math.pi)
    return {"i0": i0, "iphi": iphi, "ipk": ipk, "irms": math.sqrt(rms2)}


def dab_matched_rms(V1: float, fs: float, L: float, phi: float) -> tuple[float, float]:
    """Matched ratio (V1 = V2): I_pk = V1 phi/(w L), I_rms = I_pk sqrt(1 - 2 phi/(3 pi))."""
    w = 2 * math.pi * fs
    ipk = V1 * phi / (w * L)
    return ipk, ipk * math.sqrt(1 - 2 * phi / (3 * math.pi))


def dab_pwl(V1: float, V2: float, fs: float, L: float, phi: float) -> dict:
    """Independent path: integrate the four-interval slopes, pick the half-wave antisymmetric solution.

    The periodic condition of an ideal inductor leaves the DC offset free; the reference
    solution is fixed by i(T/2) = -i(0).  Power, RMS and peak are exact PWL integrals.
    """
    T = 1 / fs
    w = 2 * math.pi * fs
    tp = phi / w
    t = [0.0, tp, T / 2, T / 2 + tp, T]
    v1 = [V1, V1, -V1, -V1]
    v2 = [-V2, V2, V2, -V2]
    slopes = [(a - b) / L for a, b in zip(v1, v2)]
    # i(T/2) = i(0) + sum(slopes*dt over first half) ; antisymmetry: i(T/2) = -i(0)
    dt = np.diff(t)
    rise_half = slopes[0] * dt[0] + slopes[1] * dt[1]
    i_start = -rise_half / 2
    iw = PWL.from_slopes(t, slopes, i_start)
    v1w = PWL.step(t, v1)
    v2w = PWL.step(t, v2)
    P = v2w.product_integral(iw) / T
    P1 = v1w.product_integral(iw) / T
    return {"i": iw, "v1": v1w, "v2": v2w, "P": P, "P1": P1, "irms": iw.rms(), "ipk": iw.max_abs(), "i0": i_start, "iphi": iw.value_at(tp, "left"), "mean": iw.mean()}


def dab_pmax(V1: float, V2: float, fs: float, L: float) -> float:
    return V1 * V2 / (8 * fs * L)


# ----------------------------------------------------------------------------------------
# Low-voltage path
# ----------------------------------------------------------------------------------------


def dc_path_loss(P: float, V: float, R: float) -> tuple[float, float]:
    """Current and I^2 R loss of a path resistance R carrying the DC output current P/V."""
    I = P / V
    return I, I * I * R


def static_share(R: list[float], I_total: float) -> list[float]:
    """Static current divider of parallel branches with a common terminal voltage."""
    g = [1 / r for r in R]
    return [I_total * gi / sum(g) for gi in g]
