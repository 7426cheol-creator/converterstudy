"""Textbook E10 closed forms: passive RLC DC-link discharge and the RC discharge resistor.

Used only to compare with the lab's exact switched simulation and its independent ODE path.

    alpha = R/(2L),  w_d = sqrt(1/(LC) - alpha^2),  i(t) = V0/(L w_d) e^{-alpha t} sin(w_d t)
    t_pk = atan(w_d/alpha)/w_d
    R_dis = t / (C ln(V0/Vf)),  P0 = V0^2/R,  E_removed = C (V0^2 - Vf^2)/2
"""

from __future__ import annotations

import math


def rlc_underdamped(V0: float, L: float, C: float, R: float) -> dict:
    alpha = R / (2.0 * L)
    w0 = 1.0 / math.sqrt(L * C)
    if alpha >= w0:
        raise ValueError("not underdamped")
    wd = math.sqrt(w0 * w0 - alpha * alpha)

    def i(t: float) -> float:
        return V0 / (L * wd) * math.exp(-alpha * t) * math.sin(wd * t)

    t_pk = math.atan(wd / alpha) / wd
    return {"alpha": alpha, "w0": w0, "wd": wd, "t_pk": t_pk, "i_pk": i(t_pk), "i": i, "E0": 0.5 * C * V0 * V0}


def rc_discharge(C: float, V0: float, Vf: float, t: float) -> dict:
    R = t / (C * math.log(V0 / Vf))
    return {"R": R, "P0": V0 * V0 / R, "E_removed": 0.5 * C * (V0 * V0 - Vf * Vf)}
