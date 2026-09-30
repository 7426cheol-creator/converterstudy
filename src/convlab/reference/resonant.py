"""Textbook ch.12-13 closed forms for LLC and CLLC FHA (used only for comparison; simulations never import this).

LLC (ch.12):  R_ac = 8/pi^2 n^2 R_dc,  F = f/f_r,  k = L_m/L_r,  Q = sqrt(L_r/C_r)/R_ac,
    1/H = 1 + (1/k)(1 - F^-2) + j Q (F - 1/F);   full bridge V_o/V_i ~ |H|/n, half bridge |H|/(2n).
CLLC (ch.13): Z_p = Z_m || (Z_r2' + R_ac'),  H = Z_p/(Z_r1 + Z_p) * R_ac'/(Z_r2' + R_ac'),
    required forward gain n V_o/V_i, referred secondary L_r2' = n^2 L_r2, C_r2' = C_r2/n^2.
"""

from __future__ import annotations

import math

PI = math.pi


def fr(L: float, C: float) -> float:
    return 1.0 / (2 * PI * math.sqrt(L * C))


def rac(n: float, Rdc: float) -> float:
    return 8.0 / PI**2 * n * n * Rdc


def llc_gain(F: float, k: float, Q: float) -> float:
    """|H| from the textbook's normalised expression."""
    re = 1.0 + (1.0 - F**-2) / k
    im = Q * (F - 1.0 / F)
    return 1.0 / math.sqrt(re * re + im * im)


def llc_zin_boundary(k: float, Q: float) -> float:
    """F where Im Z_in changes sign (capacitive below, inductive above), in closed form.

    Im(Z_in)/Z0 = F - 1/F + kF/(1 + k^2 Q^2 F^2) = 0, multiplied by F(1 + k^2 Q^2 F^2), is a quadratic in y = F^2:
    k^2 Q^2 y^2 + (1 + k - k^2 Q^2) y - 1 = 0, with exactly one positive root. Q -> 0 gives F = 1/sqrt(1 + k).
    """
    a = (k * Q) ** 2
    b = 1.0 + k - a
    if a == 0.0:
        return 1.0 / math.sqrt(1.0 + k)
    y = 2.0 / (b + math.sqrt(b * b + 4.0 * a))  # the positive root, written without cancellation
    return math.sqrt(y)


def llc_zin(f: float, Lr: float, Cr: float, Lm: float, Rac: float) -> complex:
    w = 2 * PI * f
    Zm = 1j * w * Lm
    return 1j * w * Lr + 1 / (1j * w * Cr) + Zm * Rac / (Zm + Rac)


def cllc_H(f: float, L1: float, C1: float, Lm: float, L2p: float, C2p: float, Rac: float) -> complex:
    """Textbook form H = Zp/(Zr1 + Zp) * Rac/(Zr2' + Rac)."""
    w = 2 * PI * f
    Zr1 = 1j * w * L1 + 1 / (1j * w * C1)
    Zr2 = 1j * w * L2p + 1 / (1j * w * C2p)
    Zm = 1j * w * Lm
    Zb = Zr2 + Rac
    Zp = Zm * Zb / (Zm + Zb)
    return Zp / (Zr1 + Zp) * Rac / Zb


def cllc_zin(f: float, L1: float, C1: float, Lm: float, L2p: float, C2p: float, Rac: float) -> complex:
    w = 2 * PI * f
    Zr1 = 1j * w * L1 + 1 / (1j * w * C1)
    Zr2 = 1j * w * L2p + 1 / (1j * w * C2p)
    Zm = 1j * w * Lm
    Zb = Zr2 + Rac
    return Zr1 + Zm * Zb / (Zm + Zb)


def fr_tolerance(fr0: float, kL: float, kC: float) -> float:
    """Resonance with L scaled by kL and C by kC (textbook E05: 150 -> 142.857 / 157.895 / 150.188 kHz)."""
    return fr0 / math.sqrt(kL * kC)
