"""Textbook ch.07 closed forms for the synthetic IPMSM (dq, amplitude-invariant, phase peak).

Used only to *compare* with the lab's own computations (switched simulation, abc-frame
reconstruction, dq transient); the lab modules never import this file.

    T_e = 3/2 p [psi_m + (L_d - L_q) i_d] i_q,     w_e = p w_m
    v_d = R_s i_d - w_e L_q i_q,  v_q = R_s i_q + w_e (L_d i_d + psi_m)
"""

from __future__ import annotations

import math


def operating_point(p: int, Rs: float, Ld: float, Lq: float, psi: float, rpm: float, P_shaft: float, i_d: float) -> dict:
    wm = rpm * 2.0 * math.pi / 60.0
    we = p * wm
    T = P_shaft / wm
    i_q = T / (1.5 * p * (psi + (Ld - Lq) * i_d))
    vd = Rs * i_d - we * Lq * i_q
    vq = Rs * i_q + we * (Ld * i_d + psi)
    I = math.hypot(i_d, i_q)
    Pcu = 1.5 * Rs * (i_d**2 + i_q**2)
    return {
        "wm": wm,
        "we": we,
        "T": T,
        "i_q": i_q,
        "vd": vd,
        "vq": vq,
        "V": math.hypot(vd, vq),
        "I": I,
        "I_rms": I / math.sqrt(2.0),
        "P_cu": Pcu,
        "P_ac": P_shaft + Pcu,
    }


def v_available(Vdc: float, margin: float) -> float:
    """SVPWM linear-range phase-peak fundamental Vdc/sqrt(3), times a regulation margin."""
    return Vdc / math.sqrt(3.0) * margin


def conduction_channel_only(I_rms: float, R_switch_position: float) -> float:
    """3 I_rms^2 R: one current path per phase at any instant (not 6x)."""
    return 3.0 * I_rms**2 * R_switch_position


def dc_link_cap_rms(I_rms: float, M: float, cos_phi: float) -> float:
    """Kolar & Round closed form for the DC-link capacitor RMS current of a 2-level VSI.

    M = V_phase_peak / (Vdc/2); infinite pulse ratio, sinusoidal phase currents.
    """
    return I_rms * math.sqrt(2.0 * M * (math.sqrt(3.0) / (4.0 * math.pi) + cos_phi**2 * (math.sqrt(3.0) / math.pi - 9.0 * M / 16.0)))


def mtpa_id(psi: float, Ld: float, Lq: float, I: float) -> float:
    """Maximum torque per ampere d-current for |i| = I (L_q > L_d)."""
    dL = Lq - Ld
    return (psi - math.sqrt(psi**2 + 8.0 * dL**2 * I**2)) / (4.0 * dL)


def asc_steady(p: int, Rs: float, Ld: float, Lq: float, psi: float, we: float) -> dict:
    """Steady active short circuit (v_d = v_q = 0) at electrical speed w_e."""
    den = Rs**2 + we**2 * Ld * Lq
    i_d = -(we**2) * Lq * psi / den
    i_q = -Rs * we * psi / den
    T = 1.5 * p * (psi + (Ld - Lq) * i_d) * i_q
    return {"i_d": i_d, "i_q": i_q, "I": math.hypot(i_d, i_q), "T": T}


def freewheel_threshold_rpm(Vdc: float, p: int, psi: float) -> float:
    """Speed at which the line-line back-EMF peak sqrt(3) w_e psi_m reaches Vdc (diode bridge starts)."""
    we = Vdc / (math.sqrt(3.0) * psi)
    return we / p * 60.0 / (2.0 * math.pi)
