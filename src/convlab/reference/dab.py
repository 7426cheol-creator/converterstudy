"""Textbook ch.11 closed forms for single-phase-shift (SPS) DAB, primary-referred.

V1 = V_H, V2 = n V_L (n = Np/Ns), theta = w_s t, phi > 0 when the primary leads (HV -> LV power).
a = (V1 + V2)/(w_s L), b = (V1 - V2)/(w_s L);  i0 = -(a phi + b (pi - phi))/2,  i_phi = i0 + a phi,  i_pi = -i0.
P = V1 V2 phi (1 - |phi|/pi) / (w_s L).  Zero-DC (half-wave antisymmetric) reference solution.
"""

from __future__ import annotations

import math

PI = math.pi


def sps_power(V1, V2, L, fs, phi):
    return V1 * V2 / (2 * PI * fs * L) * phi * (1 - abs(phi) / PI)


def sps_pmax(V1, V2, L, fs):
    return V1 * V2 / (8 * fs * L)


def sps_phi_for_power(V1, V2, L, fs, P):
    """Smallest |phi| (monotonic branch |phi| <= pi/2) giving P; None if |P| > Pmax."""
    K = V1 * V2 / (2 * PI * fs * L)
    x = 4 * abs(P) / (PI * K)
    if x > 1 + 1e-15:
        return None
    phi = PI / 2 * (1 - math.sqrt(max(0.0, 1 - x)))
    return math.copysign(phi, P) if P != 0 else 0.0


def sps_currents(V1, V2, L, fs, phi):
    """(i0, i_phi, Irms, Ipk) for 0 <= phi <= pi."""
    w = 2 * PI * fs
    a = (V1 + V2) / (w * L)
    b = (V1 - V2) / (w * L)
    i0 = -(a * phi + b * (PI - phi)) / 2
    iphi = i0 + a * phi
    ms = (phi * (i0 * i0 + i0 * iphi + iphi * iphi) + (PI - phi) * (iphi * iphi - iphi * i0 + i0 * i0)) / (3 * PI)
    return i0, iphi, math.sqrt(ms), max(abs(i0), abs(iphi))


def sps_matched_rms(V1, L, fs, phi):
    """Matched ratio (V1 = V2): Ipk = V1 phi/(w L), Irms = Ipk sqrt(1 - 2 phi/(3 pi))."""
    ipk = V1 * phi / (2 * PI * fs * L)
    return ipk * math.sqrt(1 - 2 * phi / (3 * PI)), ipk


def sps_dP_dphi(V1, V2, L, fs, phi):
    return V1 * V2 / (2 * PI * fs * L) * (1 - 2 * abs(phi) / PI)
