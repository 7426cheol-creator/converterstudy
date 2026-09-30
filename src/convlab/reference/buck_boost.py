"""Textbook ch.02-03 closed forms for Buck and Boost (ideal CCM unless stated)."""

from __future__ import annotations

import math


def buck_ccm(Vin: float, Vo: float, L: float, fs: float, Io: float) -> dict:
    D = Vo / Vin
    dI = (Vin - Vo) * D / (L * fs)
    return {
        "D": D,
        "dI_pp": dI,
        "I_peak": Io + dI / 2,
        "I_valley": Io - dI / 2,
        "I_rms": math.sqrt(Io**2 + dI**2 / 12),
        "I_boundary": dI / 2,
    }


def buck_cap_ripple(dI_pp: float, fs: float, C: float, esr: float) -> dict:
    """Ideal triangular capacitor current: capacitive and ESR ripple components (peak-to-peak)."""
    return {"dV_C": dI_pp / (8 * fs * C), "dV_esr": dI_pp * esr, "naive_sum": dI_pp / (8 * fs * C) + dI_pp * esr}


def buck_dcm_ratio(D: float, L: float, R: float, fs: float) -> float:
    """DCM conversion ratio M = Vo/Vin = 2 / (1 + sqrt(1 + 4K/D^2)), K = 2L/(R Ts)."""
    K = 2 * L * fs / R
    return 2.0 / (1.0 + math.sqrt(1.0 + 4.0 * K / D**2))


def buck_dcm_duty_for(M: float, L: float, R: float, fs: float) -> float:
    K = 2 * L * fs / R
    return 2.0 * math.sqrt(K) / math.sqrt((2.0 / M - 1.0) ** 2 - 1.0)


def boost_ccm(Vin: float, Vo: float, L: float, fs: float, P: float, eta: float = 1.0) -> dict:
    D = 1 - Vin / Vo
    Iin = P / (eta * Vin)
    dI = Vin * D / (L * fs)
    R = Vo**2 / P
    return {"D": D, "Iin": Iin, "dI_pp": dI, "Iout": P / Vo, "R": R, "f_rhpz": R * (1 - D) ** 2 / (2 * math.pi * L)}


def boost_gvd(s: complex, Vin: float, D: float, L: float, C: float, R: float) -> complex:
    """Ideal CCM duty-to-output transfer function (state-space average, resistive load)."""
    Dp = 1 - D
    Vo = Vin / Dp
    num = Vo / Dp * (1 - s * L / (Dp**2 * R))
    den = 1 + s * L / (Dp**2 * R) + s**2 * L * C / Dp**2
    return num / den


def boost_gid(s: complex, Vin: float, D: float, L: float, C: float, R: float) -> complex:
    """Ideal CCM duty-to-inductor-current transfer function: left-half-plane zero only."""
    Dp = 1 - D
    Vo = Vin / Dp
    num = 2 * Vo / (Dp**2 * R) * (1 + s * R * C / 2)
    den = 1 + s * L / (Dp**2 * R) + s**2 * L * C / Dp**2
    return num / den
