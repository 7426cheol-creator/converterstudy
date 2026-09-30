"""Textbook ch.10 (FL07) and E04/E05 (EX04) closed forms for magnetics.

Only used to compare with the labs' simulations and numerical paths; lab physics never imports it.

    B_pk = V / (4 f N A_e)                         (symmetric bipolar square winding voltage)
    walk per cycle = V dt / (N A_e)                (one-sided extra volt-seconds per cycle)
    delta = sqrt(rho / (pi f mu))                  (skin depth)
    L' = n^2 L,  C' = C / n^2,  R' = n^2 R,  n = N_p / N_s
    A_w >= (N_p I_p / J + N_s I_s / J) / k_u       (window screen)
    P_cu = I_dc^2 R_dc + sum_h I_h^2 R_ac(h f0)   (harmonic copper loss)
    f_r = 1 / (2 pi sqrt(L C))
DAB SPS (FL08): P = V1 V2 phi (1 - phi/pi) / (w L), I_pk = V1 phi / (w L) (matched), I_rms = I_pk sqrt(1 - 2 phi/(3 pi)).
"""

from __future__ import annotations

import math

MU0 = 4e-7 * math.pi


def bpk_square(V: float, f: float, N: float, Ae: float) -> float:
    return V / (4.0 * f * N * Ae)


def walk_per_cycle(V: float, dt: float, N: float, Ae: float) -> float:
    return V * dt / (N * Ae)


def skin_depth(rho: float, f: float, mu_r: float = 1.0) -> float:
    return math.sqrt(rho / (math.pi * f * MU0 * mu_r))


def refer(n: float, L: float | None = None, C: float | None = None, R: float | None = None) -> dict:
    return {"L": None if L is None else n * n * L, "C": None if C is None else C / (n * n), "R": None if R is None else n * n * R}


def window_area(Np: float, Ip: float, Ns: float, Is: float, J: float, ku: float) -> float:
    """Bare-copper window screen (J in A/m^2 gives m^2; J in A/mm^2 gives mm^2)."""
    return (Np * Ip / J + Ns * Is / J) / ku


def harmonic_copper(I_h, R_h, I_dc: float = 0.0, R_dc: float = 0.0) -> float:
    return I_dc * I_dc * R_dc + sum(i * i * r for i, r in zip(I_h, R_h))


def resonant_frequency(L: float, C: float) -> float:
    return 1.0 / (2.0 * math.pi * math.sqrt(L * C))


def dab_sps_matched(V1: float, L: float, fs: float, P: float) -> dict:
    """Matched-ratio SPS (V2' = V1): phase for power P and the textbook current closed forms."""
    w = 2.0 * math.pi * fs
    A = V1 * V1 / (w * L)
    phi = (math.pi - math.sqrt(math.pi**2 - 4.0 * math.pi * P / A)) / 2.0
    Ipk = V1 * phi / (w * L)
    return {"phi": phi, "I_pk": Ipk, "I_rms": Ipk * math.sqrt(1.0 - 2.0 * phi / (3.0 * math.pi))}
