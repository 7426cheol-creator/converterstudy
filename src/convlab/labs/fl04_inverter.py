"""FL04 - Inverter: dq operating point, voltage margin and a short switching run (textbook ch.07).

Physics, kept at separate model levels:
  A  steady-state dq algebra of the textbook's synthetic IPMSM (amplitude-invariant, phase peak,
     w_e = p w_m), the shaft / AC / DC power boundaries, the voltage margin, its sensitivity to
     parameter tolerances, and alternatives when the DC bus sags.  An independent abc-frame
     reconstruction (explicit salient inductance matrix, complex-step Faraday derivative) checks
     the voltage peak and the power balance.
  A  the i_d-i_q plane: constant-torque curve, voltage limit (with R_s), current circle, MTPA
     and the torque-speed capability, each checked by a second numerical route.
  C  a three-phase two-level inverter (carrier PWM, ideal switches, no dead time) on the dq
     machine model.  The rotor angle enters as an exact LTI oscillator state (cos, sin), so each
     of the 8 switching states is linear time-invariant and the exact engine applies: no time
     step, exact edges, exact energy integrals.  Checked by a hand-written RK45 path, an energy
     ledger, the analytic operating point and the Kolar DC-link capacitor current formula.
Saturation (L_d, L_q changing with current), iron loss, dead time and device losses inside the
switching model are outside these models and are said so on every result.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize, minimize_scalar

from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import pmsm as ref
from ._common import bands_from_traj, decimate_minmax, energy_ledger, ledger_check

TB_07 = TextbookRef("인버터-알고-있는-것을-더-날카롭게-증명하기-fl04", "07. 인버터 — 알고 있는 것을 더 날카롭게 증명하기 [FL04]")
TB_19 = TextbookRef("교재실습-계약과-통과-기준", "19. 교재–실습 계약과 통과 기준")

SQ3 = math.sqrt(3.0)
PHASES = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)  # x_k = x_d cos(th - ph_k) - x_q sin(th - ph_k)
TWO_PI = 2.0 * math.pi

# ======================================================================================
# Machine model (dq, amplitude-invariant, phase peak)
# ======================================================================================


@dataclass(frozen=True)
class Machine:
    p: int
    Rs: float
    Ld: float
    Lq: float
    psi: float

    def torque(self, i_d, i_q):
        return 1.5 * self.p * (self.psi + (self.Ld - self.Lq) * i_d) * i_q

    def iq_for_torque(self, T, i_d):
        den = 1.5 * self.p * (self.psi + (self.Ld - self.Lq) * i_d)
        return T / den if den > 0 else math.nan

    def v_dq(self, we, i_d, i_q):
        return self.Rs * i_d - we * self.Lq * i_q, self.Rs * i_q + we * (self.Ld * i_d + self.psi)

    def v_mag(self, we, i_d, i_q):
        vd, vq = self.v_dq(we, i_d, i_q)
        return math.hypot(vd, vq)

    def iq_voltage_roots(self, we, i_d, V):
        """Both roots i_q of |v(i_d, i_q)| = V (including R_s), or None when the line misses the limit."""
        a = (we * self.Lq) ** 2 + self.Rs**2
        b = 2.0 * self.Rs * we * (self.Ld * i_d + self.psi - self.Lq * i_d)
        c = (self.Rs * i_d) ** 2 + (we * (self.Ld * i_d + self.psi)) ** 2 - V * V
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        r = math.sqrt(disc)
        return (-b - r) / (2.0 * a), (-b + r) / (2.0 * a)

    def iq_max_voltage(self, we, i_d, V):
        """Largest i_q >= 0 inside the voltage limit at this i_d (None if no i_q >= 0 fits)."""
        roots = self.iq_voltage_roots(we, i_d, V)
        if roots is None or roots[1] < 0:
            return None
        return roots[1]

    def mtpa_id(self, I):
        dL = self.Lq - self.Ld
        if abs(dL) < 1e-15:
            return 0.0
        return (self.psi - math.sqrt(self.psi**2 + 8.0 * dL**2 * I**2)) / (4.0 * dL)

    def abc_inductance(self, th):
        """Salient 3x3 phase inductance matrix with L_d = 3/2 (L0 + L2), L_q = 3/2 (L0 - L2)."""
        L0 = (self.Ld + self.Lq) / 3.0
        L2 = (self.Ld - self.Lq) / 3.0
        c0, cm, cp = np.cos(2 * th), np.cos(2 * th - 2 * math.pi / 3), np.cos(2 * th + 2 * math.pi / 3)
        M0 = np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]])
        M2 = np.array([[c0, cm, cp], [cm, cp, c0], [cp, c0, cm]])
        return L0 * M0 + L2 * M2


def machine_from(v: dict) -> Machine:
    return Machine(int(v["p"]), v["Rs"], v["Ld"], v["Lq"], v["psi"])


def speeds(v: dict, p: int):
    wm = v["rpm"] * TWO_PI / 60.0
    return wm, p * wm


def dq_point(m: Machine, rpm: float, P_shaft: float, i_d: float) -> dict:
    """Operating point from shaft power, speed and the chosen d-current (the lab's own algebra)."""
    wm = rpm * TWO_PI / 60.0
    we = m.p * wm
    T = P_shaft / wm
    i_q = m.iq_for_torque(T, i_d)
    vd, vq = m.v_dq(we, i_d, i_q)
    I = math.hypot(i_d, i_q)
    return {"wm": wm, "we": we, "T": T, "i_d": i_d, "i_q": i_q, "vd": vd, "vq": vq, "V": math.hypot(vd, vq), "I": I, "I_rms": I / math.sqrt(2.0), "P_cu": 1.5 * m.Rs * I * I}


def abc_check(m: Machine, we: float, i_d: float, i_q: float, n: int = 2400) -> dict:
    """Independent abc-frame path: explicit salient L(theta), flux linkage, complex-step Faraday.

    v_k = R_s i_k + w_e d(psi_k)/d(theta) over one electrical period.  Returns the phase-voltage
    peak (parabolic refinement of the grid maximum), the terminal power and copper loss averages.
    """
    th = np.linspace(0.0, TWO_PI, n, endpoint=False)
    hc = 1e-20

    def currents(t):
        return np.array([i_d * np.cos(t - ph) - i_q * np.sin(t - ph) for ph in PHASES])

    def flux(t):
        L = m.abc_inductance(t)
        return L @ currents(t) + m.psi * np.array([np.cos(t - ph) for ph in PHASES])

    va = np.empty(n)
    pw = np.empty(n)
    pc = np.empty(n)
    for k, t in enumerate(th):
        dpsi = np.imag(flux(t + 1j * hc)) / hc  # complex step: exact to round-off
        i = currents(t)
        v = m.Rs * i + we * dpsi
        va[k] = v[0]
        pw[k] = float(v @ i)
        pc[k] = m.Rs * float(i @ i)
    k = int(np.argmax(va))
    y0, y1, y2 = va[k - 1], va[k], va[(k + 1) % n]
    den = y0 - 2 * y1 + y2
    peak = y1 - 0.125 * (y2 - y0) ** 2 / den if den != 0 else y1
    return {"theta": th, "v_a": va, "V_peak": float(peak), "P_ac": float(np.mean(pw)), "P_cu": float(np.mean(pc))}


def v_avail(Vdc: float, margin: float) -> float:
    return Vdc / SQ3 * margin


def max_torque_point(m: Machine, we: float, V: float, Imax: float, n: int = 1601):
    """Maximum torque at speed w_e inside |v| <= V and |i| <= I_max (grid + bounded refinement)."""

    def tq(i_d):
        iq_c = math.sqrt(max(Imax * Imax - i_d * i_d, 0.0))
        iq_v = m.iq_max_voltage(we, i_d, V)
        if iq_v is None:
            return -1e30, 0.0
        iq = min(iq_c, iq_v)
        return m.torque(i_d, iq), iq

    grid = np.linspace(-Imax, 0.0, n)
    vals = [tq(x)[0] for x in grid]
    k = int(np.argmax(vals))
    if vals[k] <= -1e29:
        return None
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, n - 1)]
    r = minimize_scalar(lambda x: -tq(x)[0], bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
    i_d = float(r.x) if -r.fun >= vals[k] else float(grid[k])
    T, iq = tq(i_d)
    return {"T": T, "i_d": i_d, "i_q": iq, "I": math.hypot(i_d, iq), "V": m.v_mag(we, i_d, iq)}


def max_torque_slsqp(m: Machine, we: float, V: float, Imax: float, x0) -> float:
    """Independent route: SLSQP with explicit current and voltage inequality constraints."""
    cons = [
        {"type": "ineq", "fun": lambda x: (Imax * Imax - x[0] ** 2 - x[1] ** 2) / Imax**2},
        {"type": "ineq", "fun": lambda x: (V * V - m.v_mag(we, x[0], x[1]) ** 2) / V**2},
    ]
    r = minimize(lambda x: -m.torque(x[0], x[1]) / 100.0, x0, constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return -float(r.fun) * 100.0


def torque_curve_voltage_id(m: Machine, we: float, T: float, V: float, i_hi: float):
    """Most positive i_d <= i_hi on the constant-torque curve whose voltage equals V (None if none).

    |v| along the constant-torque curve first falls as i_d becomes more negative, then rises again;
    the root is searched between the curve's voltage minimum and i_hi.
    """

    def vt(i_d):
        return m.v_mag(we, i_d, m.iq_for_torque(T, i_d)) - V

    i_lim = -0.999 * m.psi / (m.Ld - m.Lq) if m.Lq > m.Ld else -5000.0  # keeps the torque denominator positive
    lo_bound = max(-5.0 * m.psi / m.Ld, i_lim if i_lim < 0 else -5000.0)
    r = minimize_scalar(lambda x: vt(x), bounds=(lo_bound, i_hi), method="bounded", options={"xatol": 1e-9})
    i_min = float(r.x)
    if vt(i_hi) <= 0:
        return i_hi, i_min
    if vt(i_min) > 0:
        return None, i_min
    return brentq(vt, i_min, i_hi, xtol=1e-12, rtol=1e-15), i_min


def voltage_ellipse_torque_id(m: Machine, we: float, T: float, V: float, i_hi: float, n: int = 4000):
    """Second route to the same intersection: walk the voltage-limit curve (upper branch) and find T."""
    ids = np.linspace(i_hi, -3.0 * m.psi / m.Ld, n)
    prev = None
    for x in ids:
        iq = m.iq_max_voltage(we, x, V)
        if iq is None:
            prev = None
            continue
        g = m.torque(x, iq) - T
        if prev is not None and prev[1] < 0 <= g:
            a, b = prev[0], x

            def f(y):
                q = m.iq_max_voltage(we, y, V)
                return (m.torque(y, q) if q is not None else -1e30) - T

            return brentq(f, b, a, xtol=1e-12, rtol=1e-15)
        prev = (x, g)
    return None


# ======================================================================================
# Two-level inverter on the dq machine: exact switched-affine model
# ======================================================================================

SVM_NAME = {(0, 0, 0): "V0", (1, 0, 0): "V1", (1, 1, 0): "V2", (0, 1, 0): "V3", (0, 1, 1): "V4", (0, 0, 1): "V5", (1, 0, 1): "V6", (1, 1, 1): "V7"}


def v_alpha_beta(S, Vdc):
    """Phase-to-neutral space vector (amplitude-invariant Clarke) of switch state S (1 = upper on)."""
    va0, vb0, vc0 = [(s - 0.5) * Vdc for s in S]
    return (2.0 / 3.0) * (va0 - 0.5 * (vb0 + vc0)), (vb0 - vc0) / SQ3


def pwm_edges(vd_cmd, vq_cmd, we, Vdc, N, cycles, method="natural", zs="minmax"):
    """Gate edges of a triangle-carrier PWM, carrier peak (+Vdc/2) at t = k T_s, T_s = T_e / N.

    method: natural (continuous reference vs carrier), regular (reference sampled at the carrier
    peak, i.e. a half-period transport delay), regular_comp (sampled at the pulse centre).
    zs: minmax zero-sequence injection (SVPWM-equivalent) or none (sinusoidal PWM).
    Returns (edges [(t, leg, value)], initial switch state, T_s).
    """
    Te = TWO_PI / we
    Ts = Te / N
    H = Vdc / 2.0

    def ref(t):
        th = we * t
        v = [vd_cmd * math.cos(th - ph) - vq_cmd * math.sin(th - ph) for ph in PHASES]
        z = -(max(v) + min(v)) / 2.0 if zs == "minmax" else 0.0
        return [x + z for x in v]

    def carrier(t, a):
        u = (t - a) / Ts
        return H * (1.0 - 4.0 * u) if u <= 0.5 else H * (-3.0 + 4.0 * u)

    edges = []
    for j in range(N * cycles):
        a = j * Ts
        mid = a + 0.5 * Ts
        b = a + Ts
        if method == "natural":
            for k in range(3):
                g = lambda t, k=k: ref(t)[k] - carrier(t, a)  # noqa: E731
                ga, gm, gb = g(a), g(mid), g(b)
                if ga < 0 < gm:
                    edges.append((brentq(g, a, mid, xtol=1e-16, rtol=1e-15), k, 1))
                if gm > 0 > gb:
                    edges.append((brentq(g, mid, b, xtol=1e-16, rtol=1e-15), k, 0))
        else:
            ts = a if method == "regular" else mid
            r = ref(ts)
            for k in range(3):
                x = r[k] / H
                if x <= -1.0 or x >= 1.0:
                    continue
                edges.append((a + Ts * (1.0 - x) / 4.0, k, 1))
                edges.append((a + Ts * (3.0 + x) / 4.0, k, 0))
    edges.sort()
    r0 = ref(0.0) if method != "regular_comp" else ref(0.5 * Ts)
    S0 = tuple(1 if r0[k] >= H else 0 for k in range(3))
    return edges, S0, Ts


def _qf(n, terms):
    """Symmetric Q on the augmented state (size n + 1) with z^T Q z = sum coef * z_i * z_j."""
    Q = np.zeros((n + 1, n + 1))
    for i, j, c in terms:
        Q[i, j] += 0.5 * c
        Q[j, i] += 0.5 * c
    return Q


def phase_current_terms(k: int, scale: float = 1.0):
    """i_k = i_d cos(th - ph) - i_q sin(th - ph) as bilinear terms in z = [i_d, i_q, c, s, 1]."""
    cp, sp = math.cos(PHASES[k]), math.sin(PHASES[k])
    # cos(th - ph) = c cp + s sp ;  sin(th - ph) = s cp - c sp
    return [(0, 2, scale * cp), (0, 3, scale * sp), (1, 3, -scale * cp), (1, 2, scale * sp)]


class InverterDQ(HybridSystem):
    """x = [i_d, i_q, cos(theta_e), sin(theta_e)]; q = (S_a, S_b, S_c), 1 = upper switch on.

    The switched phase voltages are constant within a switching state in the stationary frame;
    in the rotor frame v_d = v_alpha c + v_beta s and v_q = -v_alpha s + v_beta c, which is linear
    in the oscillator states, so each state is LTI at constant speed.
    """

    state_names = ("i_d", "i_q", "cos θ", "sin θ")
    state_units = ("A", "A", "", "")

    def __init__(self, m: Machine, we: float, Vdc: float, edges):
        self.m, self.we, self.Vdc = m, we, Vdc
        self.edges = edges
        self._times = [e[0] for e in edges]
        self._modes: dict = {}
        self._key = f"fl04inv|{m}|{we:.12g}|{Vdc:.12g}"

    def mode(self, q):
        md = self._modes.get(q)
        if md is None:
            m, we = self.m, self.we
            va, vb = v_alpha_beta(q, self.Vdc)
            A = np.zeros((4, 4))
            b = np.zeros(4)
            A[0, 0] = -m.Rs / m.Ld
            A[0, 1] = we * m.Lq / m.Ld
            A[0, 2] = va / m.Ld
            A[0, 3] = vb / m.Ld
            A[1, 0] = -we * m.Ld / m.Lq
            A[1, 1] = -m.Rs / m.Lq
            A[1, 2] = vb / m.Lq
            A[1, 3] = -va / m.Lq
            b[1] = -we * m.psi / m.Lq
            A[2, 3] = -we
            A[3, 2] = we
            md = AffineMode(f"{self._key}|{q}", A, b, label=SVM_NAME[q])
            self._modes[q] = md
        return md

    def gate_schedule(self, t0, t1):
        a = bisect.bisect_left(self._times, t0 - 1e-15)
        b = bisect.bisect_left(self._times, t1)
        out = []
        for t, k, val in self.edges[a:b]:

            def act(q, k=k, val=val):
                s = list(q)
                s[k] = val
                return tuple(s)

            out.append((t, act))
        return out

    def outputs(self, q):
        va, vb = v_alpha_beta(q, self.Vdc)
        Sa, Sb, Sc = q
        van = (2 * Sa - Sb - Sc) * self.Vdc / 3.0
        return {
            "id": np.array([1.0, 0, 0, 0, 0]),
            "iq": np.array([0, 1.0, 0, 0, 0]),
            "vd": np.array([0, 0, va, vb, 0]),
            "vq": np.array([0, 0, vb, -va, 0]),
            "c": np.array([0, 0, 1.0, 0, 0]),
            "s": np.array([0, 0, 0, 1.0, 0]),
            "van": np.array([0, 0, 0, 0, van]),
            "vab": np.array([0, 0, 0, 0, (Sa - Sb) * self.Vdc]),
        }

    def stored_energy(self):
        return np.diag([1.5 * self.m.Ld, 1.5 * self.m.Lq, 0.0, 0.0, 0.0])

    def powers(self, q):
        m, we = self.m, self.we
        idc_terms = []
        for k in range(3):
            if q[k]:
                idc_terms += phase_current_terms(k, self.Vdc)
        return {
            # DC port: V_dc x (sum of upper-switch currents), built from switch functions and phase currents
            "p_dc": _qf(4, idc_terms),
            # copper: 3/2 R_s (i_d^2 + i_q^2) = R_s (i_a^2 + i_b^2 + i_c^2)
            "p_cu": _qf(4, [(0, 0, 1.5 * m.Rs), (1, 1, 1.5 * m.Rs)]),
            # electromagnetic (shaft) power w_m T_e = 3/2 w_e [psi i_q + (L_d - L_q) i_d i_q]
            "p_em": _qf(4, [(1, 4, 1.5 * we * m.psi), (0, 1, 1.5 * we * (m.Ld - m.Lq))]),
        }

    def describe(self, q):
        return "".join(str(s) for s in q)

    # quadratic forms used for exact averages of bilinear quantities
    def idc_form(self, q):
        terms = []
        for k in range(3):
            if q[k]:
                terms += phase_current_terms(k)
        return _qf(4, terms) if terms else None

    def torque_form(self, q):
        m = self.m
        return _qf(4, [(1, 4, 1.5 * m.p * m.psi), (0, 1, 1.5 * m.p * (m.Ld - m.Lq))])


def inverter_steady(sysm: InverterDQ, S0, Te):
    """Periodic orbit: the map over one electrical period is affine in (i_d, i_q) for the fixed
    rotor trajectory (c, s) = (cos, sin), so three runs give it exactly; a fourth run checks it."""

    def end(x):
        tr = simulate(sysm, S0, [x[0], x[1], 1.0, 0.0], 0.0, Te)
        return tr.z_end[:2].copy()

    g = end([0.0, 0.0])
    e1 = end([1.0, 0.0]) - g
    e2 = end([0.0, 1.0]) - g
    M = np.column_stack([e1, e2])
    x0 = np.linalg.solve(np.eye(2) - M, g)
    return x0, M


def rk45_dq_path(m: Machine, we, Vdc, edges, S0, x0, t_end, rtol):
    """Independent path: hand-written 2-state dq ODE, rotor angle from math.cos/sin(w_e t), scipy RK45."""
    S = list(S0)
    t = 0.0
    x = np.array(x0, dtype=float)
    times = [e for e in edges if e[0] < t_end] + [(t_end, None, None)]
    for te, k, val in times:
        if te > t:
            va, vb = v_alpha_beta(tuple(S), Vdc)

            def f(tt, y, va=va, vb=vb):
                c, s = math.cos(we * tt), math.sin(we * tt)
                vd = va * c + vb * s
                vq = -va * s + vb * c
                return [(vd - m.Rs * y[0] + we * m.Lq * y[1]) / m.Ld, (vq - m.Rs * y[1] - we * (m.Ld * y[0] + m.psi)) / m.Lq]

            sol = solve_ivp(f, (t, te), x, method="RK45", rtol=rtol, atol=rtol * 1e-2)
            x = sol.y[:, -1]
            t = te
        if k is not None:
            S[k] = val
    return x


def pulse_fourier(edges, S0, Vdc, we, Te):
    """Exact fundamentals of the three phase-to-neutral voltages from the edge times (no simulation).

    v_kn is piecewise constant; its complex Fourier coefficient (2/T) int v e^{-j w t} dt is summed
    segment by segment (v(t) = |c| cos(w t + angle c)).  Returns the three phase coefficients and
    the Fortescue positive- and negative-sequence components.
    """
    S = list(S0)
    t = 0.0
    acc = np.zeros(3, dtype=complex)
    for te, k, val in [e for e in edges if e[0] < Te] + [(Te, None, None)]:
        if te > t:
            ph = (np.exp(-1j * we * te) - np.exp(-1j * we * t)) / (-1j * we)
            for j in range(3):
                acc[j] += (2 * S[j] - S[(j + 1) % 3] - S[(j + 2) % 3]) * Vdc / 3.0 * ph
            t = te
        if k is not None:
            S[k] = val
    c = 2.0 / Te * acc
    a = np.exp(2j * math.pi / 3.0)
    pos = (c[0] + a * c[1] + a * a * c[2]) / 3.0
    neg = (c[0] + a * a * c[1] + a * c[2]) / 3.0
    return c, pos, neg


def pulse_spectrum(edges, S0, Vdc, we, Te, hmax):
    """|c_h| of v_an for h = 1..hmax, exact from the edge times."""
    S = list(S0)
    t = 0.0
    hs = np.arange(1, hmax + 1)
    acc = np.zeros(hmax, dtype=complex)
    for te, k, val in [e for e in edges if e[0] < Te] + [(Te, None, None)]:
        if te > t:
            van = (2 * S[0] - S[1] - S[2]) * Vdc / 3.0
            acc += van * (np.exp(-1j * hs * we * te) - np.exp(-1j * hs * we * t)) / (-1j * hs * we)
            t = te
        if k is not None:
            S[k] = val
    return hs, np.abs(2.0 / Te * acc)


def min_pulses(edges, S0, t_end):
    """Shortest ON and OFF intervals of any leg over [0, t_end] (interior pulses only)."""
    S = list(S0)
    last = [0.0, 0.0, 0.0]
    on_min, off_min = math.inf, math.inf
    for t, k, val in edges:
        if t > t_end:
            break
        width = t - last[k]
        if last[k] > 0.0:
            if S[k] == 1:
                on_min = min(on_min, width)
            else:
                off_min = min(off_min, width)
        S[k] = val
        last[k] = t
    return on_min, off_min


# ======================================================================================
# Circuits
# ======================================================================================


def boundary_circuit(v: dict, op: dict, P_dc: float) -> Circuit:
    c = Circuit("fl04_boundary", 740, 190, title="출력 경계: DC → 인버터 → AC → 모터 → shaft")
    c.add("battery", "Vdc", 60, 105, 90, "V_dc", f"{v['Vdc']:g} V", lpos=(40, 100, "end"))
    c.add("block", "INV", 200, 105, 0, "인버터", "", w=110, h=56)
    c.add("block", "MOT", 500, 105, 0, "모터 (R_s·L_d·L_q·ψ_m)", "", w=170, h=56)
    c.wire("w_dcp", (60, 75), (60, 45), (110, 45), (110, 90), (145, 90))
    c.wire("w_dcn", (60, 135), (60, 165), (110, 165), (110, 120), (145, 120))
    c.wire("w_ac", (255, 105), (415, 105))
    c.wire("w_sh", (585, 105), (720, 105))
    c.text(335, 124, "3~ (AC)", "node")
    c.text(118, 34, f"P_DC ≥ {P_dc / 1e3:.2f} kW", "node")
    c.text(335, 96, f"P_AC ≥ {(op['P_cu'] + v['P_shaft']) / 1e3:.2f} kW", "node")
    c.text(652, 96, f"P_shaft {v['P_shaft'] / 1e3:.0f} kW", "node")
    c.text(652, 124, f"T {op['T']:.1f} Nm", "node")
    c.notes.append("경계마다 포함한 손실: AC = shaft + 모터 동손 (철손·기계손실 제외), DC = AC + 채널 전도손실 (diode·dead time·스위칭 손실 제외).")
    return c


def inverter_circuit(v: dict, title: str = "3상 2-level 인버터 + IPMSM (dq 모델)", with_svm_modes: bool = True) -> Circuit:
    """Three legs with body diodes; phase wires drop below the DC- rail to the motor (crossings carry no dot)."""
    c = Circuit("inv3", 720, 360, title=title)
    c.add("battery", "Vdc", 60, 160, 90, "V_dc", f"{v['Vdc']:g} V", lpos=(40, 156, "end"))
    xs = (190, 290, 390)
    for k, x in enumerate(xs):
        ph = "abc"[k]
        c.add("nmos", f"Q{k}H", x, 100, 90, f"S{ph}+", lpos=(x - 30, 104, "end"))
        c.add("nmos", f"Q{k}L", x, 220, 90, f"S{ph}−", lpos=(x - 30, 224, "end"))
        c.wire(f"w_up{k}", (x, 40), (x, 70))
        c.wire(f"w_mid{k}", (x, 130), (x, 160), (x, 190))
        c.wire(f"w_lo{k}", (x, 250), (x, 280))
        yb = 330 - 12 * k
        c.wire(f"w_ph{k}", (x, 160), (x + 40, 160), (x + 40, yb), (566, yb))
        c.text(x + 40, 152, ph, "node")
    c.wire("w_p0", (60, 130), (60, 40), (190, 40))
    c.wire("w_p1", (190, 40), (290, 40))
    c.wire("w_p2", (290, 40), (390, 40))
    c.wire("w_n0", (60, 190), (60, 280), (190, 280))
    c.wire("w_n1", (190, 280), (290, 280))
    c.wire("w_n2", (290, 280), (390, 280))
    c.add("block", "MOT", 628, 318, 0, "IPMSM", "R_s · L_d/L_q · ω_e ψ_m", w=150, h=58)
    c.dot((190, 40), (290, 40), (190, 280), (290, 280), (190, 160), (290, 160), (390, 160))
    c.text(410, 34, "DC+", "node")
    c.text(410, 294, "DC−", "node")
    c.probe("pA", "ia", 230, 192, "down", "i_a")
    c.probe("pB", "ib", 330, 192, "down", "i_b")
    c.probe("pC", "ic", 430, 192, "down", "i_c")
    c.probe("pDC", "idc", 118, 40, "right", "i_dc")
    c.notes.append("상 도선은 DC− 레일과 교차하지만 점이 없는 교차는 연결이 아니다.")
    if with_svm_modes:
        for q, name in SVM_NAME.items():
            key = "".join(str(s) for s in q)
            on = [f"Q{k}H" if q[k] else f"Q{k}L" for k in range(3)]
            dim = [f"Q{k}L" if q[k] else f"Q{k}H" for k in range(3)]
            act = on + ["MOT"] + [f"w_ph{k}" for k in range(3)] + [f"w_mid{k}" for k in range(3)]
            act += [f"w_up{k}" for k in range(3) if q[k]] + [f"w_lo{k}" for k in range(3) if not q[k]]
            if sum(q) in (1, 2):
                act += ["Vdc", "w_p0", "w_p1", "w_p2", "w_n0", "w_n1", "w_n2"]
                vab = (q[0] - q[1])
                text = f"{name} ({key}): 상측 {', '.join('abc'[k] for k in range(3) if q[k])}상 · 하측 {', '.join('abc'[k] for k in range(3) if not q[k])}상 — 유효 벡터, DC 전원이 전류를 공급 (v_ab = {vab:+d}·V_dc)"
            else:
                act += ["w_p1", "w_p2"] if sum(q) == 3 else ["w_n1", "w_n2"]
                dim.append("Vdc")
                text = f"{name} ({key}): 세 상이 모두 {'상측' if sum(q) == 3 else '하측'} — 영벡터, 모터 단자가 단락되어 i_dc = 0"
            c.mode(key, f"{name} ({key})", act, text, dim=dim)
    return c


# ======================================================================================
# Parameters
# ======================================================================================


def motor_params() -> list[Param]:
    return [
        Param("p", "극쌍수 p", "", 4, "", vmin=1, vmax=12, kind="int", source="TEXTBOOK", source_note="합성 IPMSM p = 4 (ω_e = p·ω_m)", group="모터"),
        Param("Rs", "고정자 상저항 R_s", "Ω", 15e-3, "mΩ", vmin=1e-4, vmax=1.0, source="TEXTBOOK", source_note="15 mΩ (합성값, 온도 미지정)", group="모터"),
        Param("Ld", "d축 인덕턴스 L_d", "H", 0.15e-3, "mH", vmin=1e-6, vmax=0.1, source="TEXTBOOK", source_note="0.15 mH (선형; 포화 미포함)", group="모터"),
        Param("Lq", "q축 인덕턴스 L_q", "H", 0.35e-3, "mH", vmin=1e-6, vmax=0.1, source="TEXTBOOK", source_note="0.35 mH (L_q > L_d: 역돌극)", group="모터"),
        Param("psi", "영구자석 쇄교자속 ψ_m", "Wb", 0.115, "Wb", vmin=1e-3, vmax=2.0, source="TEXTBOOK", source_note="0.115 Wb (phase peak, 온도 미지정)", group="모터"),
    ]


def point_params() -> list[Param]:
    return [
        Param("rpm", "기계 속도 n", "rpm", 6000.0, "rpm", vmin=10, vmax=30000, source="TEXTBOOK", source_note="6,000 rpm", group="운전점"),
        Param("P_shaft", "축 출력 P_shaft", "W", 250e3, "kW", vmin=1.0, vmax=2e6, source="TEXTBOOK", source_note="250 kW (철손·기계손실 제외)", group="운전점"),
        Param("id", "d축 전류 i_d (phase peak)", "A", -200.0, "A", vmin=-3000, vmax=1000, source="TEXTBOOK", source_note="−200 A (MTPA가 아닌 지정값)", group="운전점"),
        Param("Vdc", "DC-link 전압 V_dc", "V", 800.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="800 V (sag 760 V)", group="인버터"),
        Param("m_margin", "SVPWM 선형 전압 여유 계수", "", 0.95, "", vmin=0.5, vmax=1.0, source="TEXTBOOK", source_note="0.95 → 가용 = V_dc/√3·0.95", validity_note="1.0은 선형 SVPWM 한계(제어 여유 없음)", group="인버터"),
    ]


def _is_textbook(v: dict, keys=("p", "Rs", "Ld", "Lq", "psi", "rpm", "P_shaft", "id")) -> bool:
    tb = {"p": 4, "Rs": 15e-3, "Ld": 0.15e-3, "Lq": 0.35e-3, "psi": 0.115, "rpm": 6000.0, "P_shaft": 250e3, "id": -200.0, "Vdc": 800.0, "m_margin": 0.95, "R_sw": 3.2e-3}
    return all(abs(float(v[k]) - tb[k]) <= 1e-12 * max(1.0, abs(tb[k])) for k in keys if k in v)


def mtpa_for_torque(m: Machine, T: float):
    def f(I):
        i_d = m.mtpa_id(I)
        return m.torque(i_d, math.sqrt(max(I * I - i_d * i_d, 0.0))) - T

    I = brentq(f, 1e-6, 1e5, xtol=1e-12)
    i_d = m.mtpa_id(I)
    return i_d, math.sqrt(I * I - i_d * i_d), I


# ======================================================================================
# Experiment 1: dq operating point, boundaries, margin and its sensitivity
# ======================================================================================


def run_dq_point(v: dict) -> Result:
    res = Result("FL04", "dq_point", "A (+ 독립 abc 시간영역 재구성)")
    m = machine_from(v)
    op = dq_point(m, v["rpm"], v["P_shaft"], v["id"])
    if not math.isfinite(op["i_q"]) or op["i_q"] <= 0:
        res.verdict("NO_SOLUTION", "이 i_d에서는 ψ_m + (L_d − L_q)·i_d ≤ 0 이라 요구 토크를 내는 양의 i_q가 없다")
        return res
    we, T = op["we"], op["T"]
    Vdc = v["Vdc"]
    Va = v_avail(Vdc, v["m_margin"])
    margin = Va - op["V"]
    head = v["headroom"] / 100.0 * Va
    tb = _is_textbook(v)
    tb_bus = tb and _is_textbook(v, ("Vdc", "m_margin"))
    P_cond = 3.0 * op["I_rms"] ** 2 * v["R_sw"]
    P_ac = v["P_shaft"] + op["P_cu"]
    P_dc = P_ac + P_cond
    phi = math.atan2(op["vq"], op["vd"]) - math.atan2(op["i_q"], op["i_d"])
    res.add_metric("T", "토크 T_e = P_shaft/ω_m", T, "Nm", ref=397.887 if tb else None, ref_label="교재 397.887 Nm", tol=2e-6, basis="축 토크 (철손·기계손실 제외)")
    res.add_metric("we", "전기각속도 ω_e = p·ω_m", we, "rad/s", basis=f"f_e = {we / TWO_PI:.4g} Hz (기계 {v['rpm'] / 60:.4g} Hz × p)")
    res.add_metric("iq", "q축 전류 i_q", op["i_q"], "A", ref=427.835869 if tb else None, ref_label="지침 427.835869 A", tol=1e-8, basis="phase peak, 진폭불변 dq")
    res.add_metric("I_pk", "상전류 크기 |i| = √(i_d² + i_q²)", op["I"], "A", basis="phase peak")
    res.add_metric("I_rms", "상전류 RMS |i|/√2", op["I_rms"], "A", ref=333.949 if tb else None, ref_label="교재 333.949 A", tol=2e-6, basis="상 RMS (기본파)")
    res.add_metric("vd", "d축 전압 v_d = R_s i_d − ω_e L_q i_q", op["vd"], "V", basis="phase peak")
    res.add_metric("vq", "q축 전압 v_q = R_s i_q + ω_e(L_d i_d + ψ_m)", op["vq"], "V", basis="phase peak")
    res.add_metric("V_req", "요구 전압 |v|", op["V"], "V", ref=438.545444 if tb else None, ref_label="지침 438.545444 V", tol=1e-8, basis="상전압 peak (기본파) — V_dc와 무관")
    res.add_metric("V_avail", "가용 전압 V_dc/√3 × 여유계수", Va, "V", ref=438.786205 if tb_bus else None, ref_label="지침 438.786205 V", tol=1e-8, basis=f"SVPWM 선형 범위 × {v['m_margin']:g}")
    res.add_metric("margin", "전압 여유 = 가용 − 요구", margin, "V", ref=0.241 if tb_bus else None, ref_label="교재 0.241 V", tol=1e-3, abs_scale=1.0, basis="0 이상이면 수학적으로 가능", note="강건성은 따로 판정 (요구 여유·민감도 표)")
    res.add_metric("mi", "변조지수 |v|/(V_dc/√3)", op["V"] / (Vdc / SQ3), "", basis="1 = 선형 SVPWM 한계")
    res.add_metric("P_shaft", "① shaft 출력", v["P_shaft"], "W", basis="경계 ①: 기계 출력")
    res.add_metric("P_cu", "모터 동손 1.5·R_s(i_d² + i_q²)", op["P_cu"], "W", ref=5018.0 if tb else None, ref_label="교재 5.018 kW", tol=1e-4, basis="= 3·I_rms²·R_s (세 상 합)")
    res.add_metric("P_ac", "② 인버터 AC 출력 = shaft + 동손", P_ac, "W", ref=255.018e3 if tb else None, ref_label="교재 최소 255.018 kW", tol=1e-5, basis="경계 ②: 하한 (철손·기계손실·고조파손실 제외)")
    res.add_metric("P_cond", "채널 전도손실 3·I_rms²·R (6을 다시 곱하지 않음)", P_cond, "W", ref=1071.0 if (tb and abs(v["R_sw"] - 3.2e-3) < 1e-15) else None, ref_label="교재 1.071 kW", tol=5e-4, basis="스위치 위치 R, 상마다 한 경로; diode·dead time·스위칭 손실 제외")
    res.add_metric("P_dc", "③ DC 입력 하한 = AC + 채널 전도손실", P_dc, "W", basis="경계 ③: 반도체 손실 중 채널 전도분만 포함")
    res.add_metric("I_dc", "DC 평균 전류 하한 P_dc/V_dc", P_dc / Vdc, "A", basis="DC 평균")
    res.add_metric("pf", "기본파 역률 cos φ", math.cos(phi), "", basis="φ = ∠v − ∠i")
    # independent abc path
    ab = abc_check(m, we, v["id"], op["i_q"])
    res.add_check(check_close("상전압 peak: abc 시간영역 vs dq 폐형식", ab["V_peak"], op["V"], 1e-9, "명시적 돌극 L(θ) 행렬·ψ_abc의 복소 step 미분(Faraday) → v_a(θ) 최대값 vs √(v_d² + v_q²)", True, "V"))
    res.add_check(check_close("단자 전력: abc Σv·i 평균 vs shaft + 동손", ab["P_ac"], P_ac, 1e-9, "abc 순시전력 평균 (토크식을 쓰지 않음) vs P_shaft + 1.5R_s|i|² (토크식 → i_q)", True, "W"))
    res.add_check(check_close("dq 단자전력 1.5(v_d i_d + v_q i_q) vs shaft + 동손", 1.5 * (op["vd"] * v["id"] + op["vq"] * op["i_q"]), P_ac, 1e-12, "같은 dq 식의 에너지 항등식 (회귀)", False, "W"))
    # sensitivity to parameter tolerances at the same torque and i_d
    cases = [
        ("ψ_m −2 %", {"psi": m.psi * 0.98}),
        ("ψ_m +2 %", {"psi": m.psi * 1.02}),
        ("R_s +40 % (고온 구리)", {"Rs": m.Rs * 1.4}),
        ("L_d +5 %", {"Ld": m.Ld * 1.05}),
        ("L_d −5 %", {"Ld": m.Ld * 0.95}),
        ("L_q +5 %", {"Lq": m.Lq * 1.05}),
        ("L_q −5 %", {"Lq": m.Lq * 0.95}),
        ("V_dc −1 %", {"Vdc": Vdc * 0.99}),
    ]
    rows = []
    n_fail = 0
    for name, ch in cases:
        mm = Machine(m.p, ch.get("Rs", m.Rs), ch.get("Ld", m.Ld), ch.get("Lq", m.Lq), ch.get("psi", m.psi))
        iq = mm.iq_for_torque(T, v["id"])
        Vr = mm.v_mag(we, v["id"], iq)
        Vav = v_avail(ch.get("Vdc", Vdc), v["m_margin"])
        mg = Vav - Vr
        n_fail += mg < 0
        rows.append([name, iq, Vr, Vav, mg, "불가" if mg < 0 else ("경계" if mg < head else "가능")])
    res.tables.append(Table("t_sens", "같은 토크·같은 i_d에서 파라미터 공차가 전압 여유를 바꾸는 크기", ["변화", "i_q [A]", "요구 |v| [V]", "가용 [V]", "여유 [V]", "판정"], rows,
                            note=f"공차 {len(cases)}개 중 {n_fail}개에서 운전점이 불가능해진다. 공차 크기는 학습용 가정(ASSUMED)이며, 실제 값은 모터 공급사 자료로 바꾼다."))
    res.add_metric("n_fail", "공차 case 중 불가로 바뀌는 수", f"{n_fail} / {len(cases)}", "", basis="같은 토크·i_d")
    # sensitivity lines
    devs = np.linspace(-6.0, 6.0, 61)

    def mg_of(**kw):
        mm = Machine(m.p, kw.get("Rs", m.Rs), kw.get("Ld", m.Ld), kw.get("Lq", m.Lq), kw.get("psi", m.psi))
        iq = mm.iq_for_torque(T, v["id"])
        return v_avail(kw.get("Vdc", Vdc), v["m_margin"]) - mm.v_mag(we, v["id"], iq)

    res.add_series("s_psi", "ψ_m 변화", "V", devs.tolist(), [mg_of(psi=m.psi * (1 + d / 100)) for d in devs])
    res.add_series("s_ld", "L_d 변화", "V", devs.tolist(), [mg_of(Ld=m.Ld * (1 + d / 100)) for d in devs])
    res.add_series("s_lq", "L_q 변화", "V", devs.tolist(), [mg_of(Lq=m.Lq * (1 + d / 100)) for d in devs])
    res.add_series("s_vdc", "V_dc 변화", "V", devs.tolist(), [mg_of(Vdc=Vdc * (1 + d / 100)) for d in devs])
    res.add_series("s_rs", "R_s 변화", "V", devs.tolist(), [mg_of(Rs=m.Rs * (1 + d / 100)) for d in devs], dash=True)
    res.add_plot("p_sens", "전압 여유의 민감도 (같은 토크·같은 i_d)", ["s_psi", "s_ld", "s_lq", "s_vdc", "s_rs"], x_label="파라미터 변화", x_unit="%", y_label="전압 여유", y_unit="V", kind="xy", level="A",
                 hlines=[{"y": 0.0, "label": "경계 0 V"}, {"y": head, "label": f"요구 여유 {head:.1f} V"}],
                 proved="0.241 V 여유는 ψ_m ±2 %, 고온 R_s, L_q ±5 % 같은 흔한 공차 하나만으로도 음수가 된다는 것을 같은 토크·같은 i_d에서 계산했다. 수학적으로 가능한 점과 강건한 점은 다르다.",
                 not_yet="공차 크기·온도 상관·자석 온도계수는 합성 가정이다. L_d·L_q의 전류 의존(포화)과 전류제어 동특성·dead time 전압오차는 이 선형 정적 모델 밖이다.")
    # winding temperature (copper alpha = 0.393 %/K, R_s = 15 mOhm taken as the 20 degC value: ASSUMED)
    temps = np.linspace(20.0, 150.0, 66)
    mg_t = [mg_of(Rs=m.Rs * (1 + 0.00393 * (tc - 20.0))) for tc in temps]
    res.add_series("s_temp", "여유 vs 권선 온도 (R_s만)", "V", temps.tolist(), mg_t)
    def mg_temp(tc):
        return mg_of(Rs=m.Rs * (1 + 0.00393 * (tc - 20.0)))

    t0c = brentq(mg_temp, 20.0, 150.0, xtol=1e-9) if mg_temp(20.0) >= 0 > mg_temp(150.0) else None
    res.add_plot("p_temp", "권선 온도가 오르면 (R_s 증가만 반영)", ["s_temp"], x_label="권선 온도", x_unit="°C", y_label="전압 여유", y_unit="V", kind="xy", level="A",
                 hlines=[{"y": 0.0, "label": "0 V"}], vlines=[{"x": t0c, "label": f"{t0c:.1f} °C에서 0"}] if t0c else [],
                 proved=(f"R_s = 15 mΩ를 20 °C 값으로 보면(ASSUMED) 권선 온도가 {t0c - 20:.1f} K만 올라도 이 점의 여유가 사라진다." if t0c else "이 범위(20–150 °C)에서 여유의 부호가 바뀌지 않는다."),
                 not_yet="자석 온도에 따른 ψ_m 감소(NdFeB 약 −0.1 %/K, 이 점에서는 여유를 오히려 줄이는 방향)와 L 변화는 넣지 않았다. 교재 R_s의 온도 조건은 주어지지 않았다.")
    # abc waveforms (one electrical period)
    tt = ab["theta"] / we
    ia = v["id"] * np.cos(ab["theta"]) - op["i_q"] * np.sin(ab["theta"])
    res.add_series("va_abc", "v_a (abc 재구성)", "V", tt.tolist(), ab["v_a"].tolist())
    res.add_series("ia_abc", "i_a", "A", tt.tolist(), ia.tolist())
    res.add_plot("p_abc", "한 전기 주기의 a상 전압과 가용 전압", ["va_abc"], y_label="v_a", y_unit="V", group="abc", level="A",
                 hlines=[{"y": Va, "label": f"가용 {Va:.2f} V"}, {"y": -Va, "label": "−가용"}],
                 proved="abc 좌표에서 돌극 인덕턴스 행렬로 다시 계산한 상전압 peak가 dq 폐형식 |v|와 일치하고, 그 peak가 가용 전압에 거의 닿아 있다.",
                 not_yet="정현파 기본파만 본 정상상태다. PWM 리플·dead time·전류제어 과도는 실험 3과 FL06에서 본다.")
    res.add_plot("p_iabc1", "a상 전류 (같은 주기)", ["ia_abc"], y_label="i_a", y_unit="A", group="abc", level="A",
                 proved="상전류 peak는 |i| = 472.27 A, RMS는 333.95 A다(phase peak와 RMS를 구분).", not_yet="")
    # alternatives at this bus
    Vt = Va - head
    alt = []

    def row(name, i_d, i_q, note):
        if i_d is None or i_q is None or not math.isfinite(i_q):
            alt.append([name, "—", "—", "—", "—", "—", "—", "—", "—", "—", note])
            return
        I = math.hypot(i_d, i_q)
        Tq = m.torque(i_d, i_q)
        Vr = m.v_mag(we, i_d, i_q)
        alt.append([name, i_d, i_q, I, Tq, Tq * op["wm"] / 1e3, Vr, Va - Vr, 1.5 * m.Rs * I * I / 1e3, 3 * (I / math.sqrt(2)) ** 2 * v["R_sw"] / 1e3, note])

    row("현재 운전점", v["id"], op["i_q"], "지정 i_d")
    ida, _ = torque_curve_voltage_id(m, we, T, Vt, v["id"])
    row(f"(a) 같은 토크, 더 음의 i_d (여유 {head:.1f} V 확보)", ida, m.iq_for_torque(T, ida) if ida is not None else None,
        "전류·동손 비용 확인" if ida is not None else "이 토크로는 전압 한계 안에 들어갈 수 없다")
    ida0, _ = torque_curve_voltage_id(m, we, T, Va, v["id"])
    row("(a0) 같은 토크, 여유 0 경계", ida0, m.iq_for_torque(T, ida0) if ida0 is not None else None, "경계점 — 강건하지 않음")
    idm, iqm, _ = mtpa_for_torque(m, T)
    row("(b) 같은 토크의 MTPA", idm, iqm, "최소 전류 점 (전압은 따로 확인)")
    iqc = m.iq_max_voltage(we, v["id"], Vt)
    row("(c) i_d 고정, 토크 derating", v["id"], iqc, "토크·출력 감소를 고객과 합의")
    mt = max_torque_point(m, we, Vt, v["Imax"])
    row(f"(d) 최대 토크 (|i| ≤ {v['Imax']:g} A, 여유 포함)", mt["i_d"] if mt else None, mt["i_q"] if mt else None, "약계자 최적점: 전류 한계 근처")
    alt.append(["(e) 여유계수 1.0 (선형 한계까지)", "—", "—", "—", "—", "—", op["V"], Vdc / SQ3 - op["V"], "—", "—", "해결책 아님: 전류제어 여유를 소진"])
    alt.append(["(f) 과변조·6-step", "—", "—", "—", "—", "—", op["V"], 2 * Vdc / math.pi - op["V"], "—", "—", "기본파 최대 2V_dc/π; 고조파·리플·손실·제어 비용 (모델 밖)"])
    res.tables.append(Table("t_alt", f"V_dc = {Vdc:g} V에서 가능한 대안과 비용 (요구 여유 {v['headroom']:g} % = {head:.2f} V)", ["대안", "i_d [A]", "i_q [A]", "|i| [A]", "T [Nm]", "P_shaft [kW]", "|v| [V]", "여유 [V]", "동손 [kW]", "채널손실 [kW]", "비고"], alt,
                            note="모든 행은 같은 선형 dq 모델의 정상상태다. 약계자(더 음의 i_d)는 이 점에서는 MTPA(−219 A)를 지나기 전까지 전류가 거의 늘지 않지만, 여유를 더 요구할수록 전류·손실 비용이 커진다. 포화로 L_d·L_q가 바뀌면 표 전체가 달라진다."))
    # power boundary table
    res.tables.append(Table("t_bound", "출력 경계와 포함한 손실", ["경계", "전력 [kW]", "포함", "제외 (별도 계산 필요)"], [
        ["① shaft", v["P_shaft"] / 1e3, "기계 출력 (요구)", "—"],
        ["② AC (인버터 출력 = 모터 입력)", P_ac / 1e3, "+ 모터 동손 1.5R_s|i|²", "철손, 기계손실, PWM 고조파 손실"],
        ["③ DC (인버터 입력)", P_dc / 1e3, "+ 채널 전도손실 3·I_rms²·R", "diode 도통·dead time·스위칭(E_on/E_off)·gate·DC-link ESR 손실"],
    ], note="회생에서는 shaft → AC → DC 순서로 손실을 빼고 포트 부호를 유지한다. inverter 효율을 차량 주행거리로 바로 옮기지 않는다(모터 고조파 손실 생략)."))
    res.circuit = {"diagram": boundary_circuit(v, op, P_dc).to_json(), "intervals": [], "plot_group": ""}
    if margin < 0:
        res.verdict("FAIL_CONSTRAINT", f"요구 {op['V']:.3f} V > 가용 {Va:.3f} V ({-margin:.2f} V 부족): 같은 i_d·토크 점은 이 버스에서 운전할 수 없다. 대안 표의 비용을 보고 선택한다.")
    elif margin < head:
        res.verdict("MARGINAL", f"여유 {margin:.3f} V < 요구 여유 {head:.2f} V: 수학적 경계점이다. 공차 {len(cases)}개 중 {n_fail}개에서 불가로 바뀐다 — 강건한 점이 아니다.")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"여유 {margin:.2f} V ≥ 요구 여유 {head:.2f} V (선형 dq 정상상태 모델 범위)")
    res.assumptions += [
        "선형 L_d·L_q (포화·교차결합 없음), 정현 역기전력, 정상상태 기본파만",
        "철손·기계손실 제외 (교재 기준점), 고정자 저항은 온도 미지정 15 mΩ",
        f"가용 전압 = V_dc/√3 × {v['m_margin']:g} (SVPWM 선형 범위 + 제어 여유)",
        f"강건 판정 요구 여유 {v['headroom']:g} %와 공차 크기는 학습용 가정(ASSUMED)",
    ]
    res.not_valid_for += ["포화·온도에 따른 L_d/L_q/ψ_m 변화 (선형 모델 밖)", "전류제어 과도·dead time 전압오차", "인버터 효율·차량 주행거리 (반도체 손실 일부만 포함)"]
    res.interpretation = (
        f"요구 전압 {op['V']:.3f} V는 속도·전류·L·ψ_m이 정하는 값이라 DC 전압과 무관하고, 가용 전압만 V_dc에 비례한다. "
        f"800 V·여유계수 0.95에서 둘의 차이는 0.241 V로 수학적으로만 가능한 경계점이다. 같은 토크에서 ψ_m 2 % 감소(i_q 증가로 v_d 증가), R_s 40 % 증가, L_q 5 % 증가 중 어느 하나만으로도 여유가 음수가 된다. "
        f"출력은 shaft {v['P_shaft'] / 1e3:.0f} kW → AC {P_ac / 1e3:.3f} kW(동손 포함) → DC ≥ {P_dc / 1e3:.3f} kW(채널 전도손실 포함)로 경계를 나눠 말한다. "
        "V_dc가 760 V로 떨어지면 같은 점은 불가능하며, 더 음의 i_d·토크 derating·과변조 중 선택은 전류·손실·제어 비용과 함께 정한다."
    )
    return res


# ======================================================================================
# Experiment 2: the i_d-i_q plane
# ======================================================================================


def _iq_vlim_vec(m: Machine, we: float, ids: np.ndarray, V: float) -> np.ndarray:
    a = (we * m.Lq) ** 2 + m.Rs**2
    b = 2.0 * m.Rs * we * (m.Ld * ids + m.psi - m.Lq * ids)
    c = (m.Rs * ids) ** 2 + (we * (m.Ld * ids + m.psi)) ** 2 - V * V
    disc = b * b - 4.0 * a * c
    out = np.full(ids.shape, np.nan)
    ok = disc >= 0
    out[ok] = (-b[ok] + np.sqrt(disc[ok])) / (2.0 * a)
    out[out < 0] = np.nan
    return out


def capability(m: Machine, rpm_grid, V: float, Imax: float):
    """Maximum torque vs speed inside |i| <= I_max and |v| <= V (vectorised grid + bounded refinement)."""
    Ts = []
    ids = np.linspace(-Imax, 0.0, 1201)
    iqc = np.sqrt(np.maximum(Imax * Imax - ids * ids, 0.0))
    for n in rpm_grid:
        we = m.p * n * TWO_PI / 60.0
        iqv = _iq_vlim_vec(m, we, ids, V)
        iq = np.where(np.isnan(iqv), np.nan, np.minimum(iqc, iqv))
        tq = 1.5 * m.p * (m.psi + (m.Ld - m.Lq) * ids) * iq
        if np.all(np.isnan(tq)):
            Ts.append(math.nan)
            continue
        mt = max_torque_point(m, we, V, Imax, n=401)
        Ts.append(mt["T"] if mt else float(np.nanmax(tq)))
    return Ts


def run_dq_plane(v: dict) -> Result:
    res = Result("FL04", "dq_plane", "A")
    m = machine_from(v)
    op = dq_point(m, v["rpm"], v["P_shaft"], v["id"])
    if not math.isfinite(op["i_q"]) or op["i_q"] <= 0:
        res.verdict("NO_SOLUTION", "이 i_d에서는 요구 토크를 내는 양의 i_q가 없다")
        return res
    we, T = op["we"], op["T"]
    Imax = v["Imax"]
    V1 = v_avail(v["Vdc"], v["m_margin"])
    V2 = v_avail(v["Vdc_sag"], v["m_margin"])
    ids = np.linspace(-1.4 * Imax, 0.15 * Imax, 621)
    iqT = [m.iq_for_torque(T, x) for x in ids]
    iqT = [y if (math.isfinite(y) and 0 < y < 1.6 * Imax) else float("nan") for y in iqT]
    res.add_series("c_torque", f"토크 {T:.1f} Nm 곡선", "A", ids.tolist(), iqT)
    res.add_series("c_v1", f"전압 한계 V_dc {v['Vdc']:g} V ({V1:.1f} V)", "A", ids.tolist(), _iq_vlim_vec(m, we, ids, V1).tolist())
    res.add_series("c_v2", f"전압 한계 V_dc {v['Vdc_sag']:g} V ({V2:.1f} V)", "A", ids.tolist(), _iq_vlim_vec(m, we, ids, V2).tolist(), dash=True)
    cx = np.linspace(-Imax, min(0.15 * Imax, Imax), 301)
    res.add_series("c_circle", f"전류 한계 |i| = {Imax:g} A (ASSUMED)", "A", cx.tolist(), np.sqrt(np.maximum(Imax**2 - cx**2, 0)).tolist())
    Is = np.linspace(0.0, Imax, 121)
    mid = [m.mtpa_id(I) for I in Is]
    res.add_series("c_mtpa", "MTPA 궤적", "A", mid, [math.sqrt(max(I * I - x * x, 0.0)) for I, x in zip(Is, mid)], dash=True)
    idm, iqm, Im = mtpa_for_torque(m, T)
    ida, _ = torque_curve_voltage_id(m, we, T, V2, v["id"])
    ida2 = voltage_ellipse_torque_id(m, we, T, V2, v["id"])
    mt1 = max_torque_point(m, we, V1, Imax)
    mt2 = max_torque_point(m, we, V2, Imax)
    markers = [{"x": v["id"], "y": op["i_q"], "label": "운전점"}]
    zoom_m = [{"x": v["id"], "y": op["i_q"], "label": "운전점"}, {"x": idm, "y": iqm, "label": "MTPA"}]
    if ida is not None:
        zoom_m.append({"x": ida, "y": m.iq_for_torque(T, ida), "label": f"{v['Vdc_sag']:g} V"})
    if mt1:
        markers.append({"x": mt1["i_d"], "y": mt1["i_q"], "label": "T_max"})
    res.add_plot("p_plane", f"i_d–i_q 평면 ({v['rpm']:g} rpm, ω_e = {we:.0f} rad/s)", ["c_torque", "c_v1", "c_v2", "c_circle", "c_mtpa"], x_label="i_d", x_unit="A", y_label="i_q", y_unit="A", kind="xy", level="A", markers=markers,
                 proved="같은 토크 곡선 위에서 전압 한계(속도에 따라 줄어드는 타원, R_s 포함)와 전류 원이 만드는 가능 영역을 계산했다. V_dc가 내려가면 타원이 줄어 운전점이 밖으로 나간다.",
                 not_yet="포화로 L_d·L_q가 바뀌면 곡선 모양 자체가 달라진다(선형 모델 밖). 전류 한계 원은 합성 가정이며 소자·모터 열 한계를 계산하지 않았다.")
    # zoom around the operating point
    zl = min(v["id"], idm, ida if ida is not None else v["id"]) - 60.0
    zh = max(v["id"], idm) + 60.0
    zid = np.linspace(zl, zh, 241)
    res.add_series("z_torque", "토크 곡선", "A", zid.tolist(), [m.iq_for_torque(T, x) for x in zid])
    res.add_series("z_v1", f"전압 한계 {v['Vdc']:g} V", "A", zid.tolist(), _iq_vlim_vec(m, we, zid, V1).tolist())
    res.add_series("z_v2", f"전압 한계 {v['Vdc_sag']:g} V", "A", zid.tolist(), _iq_vlim_vec(m, we, zid, V2).tolist(), dash=True)
    Iz = np.linspace(0.8 * Im, min(1.2 * Im, Imax), 81)
    zmid = [m.mtpa_id(I) for I in Iz]
    res.add_series("z_mtpa", "MTPA 궤적", "A", zmid, [math.sqrt(max(I * I - x * x, 0.0)) for I, x in zip(Iz, zmid)], dash=True)
    res.add_plot("p_zoom", "운전점 부근 확대: 지정점·MTPA·sag 경계", ["z_torque", "z_v1", "z_v2", "z_mtpa"], x_label="i_d", x_unit="A", y_label="i_q", y_unit="A", kind="xy", level="A", markers=zoom_m,
                 proved="지정점은 800 V 전압 한계 곡선 바로 아래(0.24 V)에 있고, 같은 토크의 MTPA는 더 음의 i_d 쪽, 760 V 경계는 그보다 더 음의 i_d 쪽에 있다.",
                 not_yet="확대 구간에서도 선형 모델이다. 포화가 있으면 곡선 사이 간격 자체가 바뀐다.")
    # margin and current along the constant-torque curve
    idc = np.linspace(min(v["id"], idm) - 150.0, max(v["id"], idm) + 150.0, 301)
    vt = np.array([m.v_mag(we, x, m.iq_for_torque(T, x)) for x in idc])
    res.add_series("m_v1", f"여유 @ {v['Vdc']:g} V", "V", idc.tolist(), (V1 - vt).tolist())
    res.add_series("m_v2", f"여유 @ {v['Vdc_sag']:g} V", "V", idc.tolist(), (V2 - vt).tolist(), dash=True)
    res.add_plot("p_margin", "같은 토크 곡선을 따라가며 본 전압 여유", ["m_v1", "m_v2"], x_label="i_d", x_unit="A", y_label="여유", y_unit="V", kind="xy", level="A",
                 hlines=[{"y": 0.0, "label": "0 V"}], vlines=[{"x": v["id"], "label": "지정"}], markers=[{"x": idm, "y": V1 - m.v_mag(we, idm, iqm), "label": "MTPA"}],
                 proved="i_d를 더 음으로 하면 d축 자속(L_d i_d + ψ_m)이 줄어 같은 토크의 전압 요구가 내려간다. 지정 i_d = −200 A는 MTPA보다 오른쪽이라 전압 여유가 가장 작은 쪽에 있다.",
                 not_yet="여유가 커지는 방향의 전류 비용은 아래 그림과 대안 표에서 본다.")
    It = np.hypot(idc, [m.iq_for_torque(T, x) for x in idc])
    res.add_series("m_I", "|i| (같은 토크)", "A", idc.tolist(), It.tolist())
    res.add_plot("p_current", "같은 토크를 내는 전류 크기", ["m_I"], x_label="i_d", x_unit="A", y_label="|i|", y_unit="A", kind="xy", level="A",
                 hlines=[{"y": Imax, "label": "전류 한계"}], vlines=[{"x": v["id"], "label": "지정"}], markers=[{"x": idm, "y": Im, "label": "MTPA (최소)"}],
                 proved="MTPA에서 전류가 최소이고, 그보다 더 음의 i_d는 전류(동손·전도손실)를 늘린다. 약계자는 MTPA를 지나면서부터 공짜가 아니다.", not_yet="")
    # capability vs speed
    rpms = np.linspace(max(200.0, 0.03 * v["rpm"]), 2.0 * v["rpm"], 60)
    T1 = capability(m, rpms, V1, Imax)
    T2 = capability(m, rpms, V2, Imax)
    wms = rpms * TWO_PI / 60.0
    res.add_series("cap1", f"최대 토크 @ {v['Vdc']:g} V", "Nm", rpms.tolist(), T1)
    res.add_series("cap2", f"최대 토크 @ {v['Vdc_sag']:g} V", "Nm", rpms.tolist(), T2, dash=True)
    tcap = max([x for x in T1 if math.isfinite(x)] + [T])
    res.add_series("hyp", f"{v['P_shaft'] / 1e3:.0f} kW 일정출력 곡선", "Nm", rpms.tolist(), [v["P_shaft"] / w if v["P_shaft"] / w <= 1.3 * tcap else float("nan") for w in wms], dash=True)
    res.add_plot("p_cap", "토크–속도 능력 (전류 한계 + 전압 한계)", ["cap1", "cap2", "hyp"], x_label="n", x_unit="rpm", y_label="T", y_unit="Nm", kind="xy", level="A",
                 vlines=[{"x": v["rpm"], "label": f"{v['rpm']:g} rpm"}], markers=[{"x": v["rpm"], "y": T, "label": "요구"}],
                 proved="저속에서는 전류 한계가, 고속에서는 전압 한계가 토크를 정한다. 버스 전압이 낮아지면 전압 한계 구간의 능력이 내려간다.",
                 not_yet="요구 여유(0.95 계수 외 추가 여유)는 넣지 않은 경계 곡선이다. 열 한계·포화·철손은 없다.")
    # metrics
    res.add_metric("T", "요구 토크", T, "Nm")
    res.add_metric("I_char", "특성전류 ψ_m/L_d (전압 타원 중심)", m.psi / m.Ld, "A", basis="|i|max보다 크면 무한 속도 약계자 불가", note="이 모델: 전류 원 밖" if m.psi / m.Ld > Imax else "전류 원 안")
    res.add_metric("id_mtpa", "같은 토크의 MTPA i_d", idm, "A", basis="최소 |i| 점")
    res.add_metric("I_mtpa", "MTPA 전류 |i|", Im, "A", note=f"지정점보다 {op['I'] - Im:.2f} A 작다")
    res.add_metric("V_mtpa", "MTPA 점 요구 전압", m.v_mag(we, idm, iqm), "V", note=f"여유 @ {v['Vdc']:g} V: {V1 - m.v_mag(we, idm, iqm):.2f} V")
    if ida is not None:
        res.add_metric("id_sag", f"{v['Vdc_sag']:g} V에서 같은 토크의 경계 i_d", ida, "A", basis="여유 0 (강건하지 않음)")
        res.add_metric("I_sag", "그 점의 |i|", math.hypot(ida, m.iq_for_torque(T, ida)), "A")
    if mt1:
        res.add_metric("Tmax1", f"이 속도 최대 토크 @ {v['Vdc']:g} V", mt1["T"], "Nm", basis=f"|i| ≤ {Imax:g} A, 여유 0")
    if mt2:
        res.add_metric("Tmax2", f"이 속도 최대 토크 @ {v['Vdc_sag']:g} V", mt2["T"], "Nm", basis=f"|i| ≤ {Imax:g} A, 여유 0", note=f"출력 {mt2['T'] * op['wm'] / 1e3:.1f} kW")
    # independent checks
    lo_id = -0.99 * m.psi / m.Ld
    r_min = minimize_scalar(lambda x: math.hypot(x, m.iq_for_torque(T, x)), bounds=(lo_id, 0.0), method="bounded", options={"xatol": 1e-10})
    res.add_check(check_close("MTPA: 폐형식 vs 같은 토크 곡선 위 최소 |i|", float(r_min.x), idm, 1e-6, "∂T/∂γ = 0 해석식(|i| 고정) + 토크 역산 vs 토크 곡선을 따라 |i|를 bounded 최소화 (다른 문제 설정)", True, "A", abs_scale=1.0))
    if ida is not None and ida2 is not None:
        res.add_check(check_close("토크 곡선 ∩ 전압 한계: 두 경로", ida2, ida, 1e-9, "토크 곡선을 따라 |v| = V 풀기 vs 전압 한계 곡선(상측 근)을 따라 T = T* 풀기", True, "A", abs_scale=1.0))
    if mt1:
        s1 = max_torque_slsqp(m, we, V1, Imax, [mt1["i_d"] * 0.9, mt1["i_q"] * 0.9])
        res.add_check(check_close(f"최대 토크 @ {v['Vdc']:g} V: 격자+정제 vs SLSQP", s1, mt1["T"], 1e-6, "i_d 격자에서 가능한 최대 i_q → bounded 정제 vs 부등식 제약 SLSQP (다른 알고리즘)", True, "Nm"))
    mg1, mg2 = V1 - op["V"], V2 - op["V"]
    res.add_metric("mg1", f"지정점 전압 여유 @ {v['Vdc']:g} V", mg1, "V", basis="가용 − 요구 (0 미만 = 타원 밖)")
    res.add_metric("mg2", f"지정점 전압 여유 @ {v['Vdc_sag']:g} V", mg2, "V")
    res.verdict("PASS_WITHIN_MODEL", "선형 dq 정상상태 모델에서 토크 곡선·전압 한계·전류 원·MTPA를 두 경로로 확인했다 (포화 미포함)")
    if mg1 < 0:
        res.verdict("FAIL_CONSTRAINT", f"지정 i_d = {v['id']:g} A는 {v['Vdc']:g} V에서도 전압 한계 밖이다 ({mg1:.1f} V). 같은 토크의 경계 i_d는 대안 계산(실험 1)으로 찾는다")
    elif mg1 < 0.02 * V1:
        res.verdict("MARGINAL", f"지정점은 {v['Vdc']:g} V 타원 경계에 걸려 있다 (여유 {mg1:.3f} V < 가용의 2 %, 실험 1과 같은 학습 기준). {v['Vdc_sag']:g} V에서는 {'밖' if mg2 < 0 else '안'} ({mg2:.1f} V)")
    elif mg2 < 0:
        res.verdict("INFO", f"지정점은 {v['Vdc']:g} V 타원 안(여유 {mg1:.2f} V), {v['Vdc_sag']:g} V 타원 밖({mg2:.1f} V)")
    res.assumptions += ["선형 L_d·L_q·ψ_m, 정상상태, R_s 포함 전압식", f"전류 한계 {Imax:g} A는 합성 가정(ASSUMED)", "가용 전압 = V_dc/√3 × 여유계수 (추가 요구 여유 없음)"]
    res.not_valid_for += ["포화·교차결합이 있는 실제 모터의 MTPA/약계자 표", "열 한계·연속/순간 정격 구분", "과변조 영역"]
    res.interpretation = (
        f"같은 토크 {T:.1f} Nm는 곡선 위의 무수한 (i_d, i_q) 조합으로 낼 수 있다. 속도가 높으면 전압 한계 타원이 작아져 곡선의 일부만 쓸 수 있고, "
        f"V_dc가 {v['Vdc_sag']:g} V로 내려가면 지정점은 타원 밖이 된다. 곡선을 따라 i_d를 더 음으로 옮기면 전압이 들어오지만, MTPA({idm:.1f} A)를 지나면 같은 토크에 더 큰 전류가 든다. "
        "이 모든 곡선은 선형 L_d·L_q 가정이라 포화가 있는 실제 모터에서는 측정 기반 표로 바꿔야 한다."
    )
    return res


# ======================================================================================
# Experiment 3: short inverter switching run
# ======================================================================================


def run_switching(v: dict) -> Result:
    res = Result("FL04", "inverter_switching", "C (이상 스위치, dq 모터; 해석 운전점과 비교)")
    m = machine_from(v)
    op = dq_point(m, v["rpm"], v["P_shaft"], v["id"])
    if not math.isfinite(op["i_q"]) or op["i_q"] <= 0:
        res.verdict("NO_SOLUTION", "이 i_d에서는 요구 토크를 내는 양의 i_q가 없다")
        return res
    we, Vdc, N, cyc = op["we"], v["Vdc"], int(v["N"]), int(v["cycles"])
    Te = TWO_PI / we
    fsw = N / Te
    method, zs = v["pwm"], v["zs"]
    edges, S0, Ts = pwm_edges(op["vd"], op["vq"], we, Vdc, N, cyc, method, zs)
    sysm = InverterDQ(m, we, Vdc, edges)
    x0, M = inverter_steady(sysm, S0, Te)
    tr = simulate(sysm, S0, [x0[0], x0[1], 1.0, 0.0], 0.0, cyc * Te)
    z1, _ = tr.state_at(Te)
    Iscale = max(op["I"], 1.0)
    resid = float(np.max(np.abs(z1[:2] - x0)) / Iscale)
    w0, w1 = (cyc - 1) * Te, cyc * Te  # analysis window: last period
    mid_ = tr.mean(w0, w1, "id")
    miq = tr.mean(w0, w1, "iq")
    mvd = tr.mean(w0, w1, "vd")
    mvq = tr.mean(w0, w1, "vq")
    natural = method == "natural"
    c3, pos, neg = pulse_fourier(edges, S0, Vdc, we, Te)
    ang_cmd = math.atan2(op["vq"], op["vd"])
    dang = math.degrees((np.angle(pos) - ang_cmd + math.pi) % TWO_PI - math.pi)
    lin_ok = op["V"] <= (Vdc / SQ3 if zs == "minmax" else Vdc / 2.0)
    res.add_metric("fsw", "스위칭 주파수 f_sw = N·f_e", fsw, "Hz", basis=f"펄스비 N = {N} (동기 PWM), f_e = {1 / Te:.4g} Hz")
    res.add_metric("V1_pos", "PWM 기본파 (정상분, 펄스 Fourier)", abs(pos), "V", ref=op["V"], ref_label="지령 |v| (해석)", tol=5e-4 if (natural and lin_ok) else None, basis="상전압 peak, 세 상 Fortescue 정상분")
    res.add_metric("dang", "기본파 위상 오차 (정상분 − 지령)", dang, "deg", basis="음수 = 지연", note="regular 샘플링은 반 주기 지연 ω_e·T_s/2" if method == "regular" else "")
    res.add_metric("V1_neg", "역상분 기본파", abs(neg), "V", basis="N이 3의 배수가 아니면 세 상 펄스 배치가 달라 생긴다")
    res.add_metric("id_mean", "i_d 한 주기 평균 (스위칭 해)", mid_, "A", ref=v["id"], ref_label="해석 운전점", tol=1e-3 if (natural and lin_ok) else None, abs_scale=Iscale, basis="정상 주기해, 마지막 주기")
    res.add_metric("iq_mean", "i_q 한 주기 평균 (스위칭 해)", miq, "A", ref=op["i_q"], ref_label="해석 운전점", tol=1e-3 if (natural and lin_ok) else None, abs_scale=Iscale, basis="정상 주기해")
    res.add_metric("vd_mean", "v_d 한 주기 평균", mvd, "V", ref=op["vd"], ref_label="지령", tol=None)
    res.add_metric("vq_mean", "v_q 한 주기 평균", mvq, "V", ref=op["vq"], ref_label="지령", tol=None)
    # exact quadratic averages
    rms3 = math.sqrt(tr.integral_quadratic(w0, w1, lambda q: _qf(4, [(0, 0, 0.5), (1, 1, 0.5)])) / Te)
    idc_mean = tr.integral_quadratic(w0, w1, sysm.idc_form) / Te
    T_mean = tr.integral_quadratic(w0, w1, sysm.torque_form) / Te
    P_ac_an = 1.5 * (op["vd"] * v["id"] + op["vq"] * op["i_q"])
    res.add_metric("rms3", "상전류 RMS (세 상 평균, 정확 적분)", rms3, "A", ref=op["I_rms"], ref_label="기본파 RMS (리플 없음)", basis="√(Σ_k I_k,rms²/3), 리플 포함", note="리플만큼 기본파보다 크다")
    res.add_metric("idc_mean", "DC-link 평균 전류", idc_mean, "A", ref=P_ac_an / Vdc, ref_label="P_AC/V_dc (해석)", tol=2e-3 if (natural and lin_ok) else None, basis="Σ S_k i_k의 정확 평균 (상측 스위치 전류 합)")
    res.add_metric("T_mean", "평균 토크 (스위칭 해)", T_mean, "Nm", ref=op["T"], ref_label="요구 토크", tol=2e-3 if (natural and lin_ok) else None)
    # dense samples for bilinear quantities
    smp = tr.sample(["id", "iq", "c", "s", "vab", "van"], 0.0, cyc * Te, per_segment=10)
    t = np.array(smp["t"])
    idv, iqv, cv, sv = (np.array(smp[k]) for k in ("id", "iq", "c", "s"))
    Sab = []
    for sg in tr.window(0.0, cyc * Te):
        Sab += [sg[0].q] * 10
    Sab = np.array(Sab[: t.size])
    iph = [idv * (cv * math.cos(ph) + sv * math.sin(ph)) - iqv * (sv * math.cos(ph) - cv * math.sin(ph)) for ph in PHASES]
    idc_legs = [Sab[:, k] * iph[k] for k in range(3)]
    idc = idc_legs[0] + idc_legs[1] + idc_legs[2]
    th = we * t
    ia_f = mid_ * np.cos(th) - miq * np.sin(th)
    rip = iph[0] - ia_f
    msk = t >= w0 - 1e-15

    def tavg(y):
        return float(np.trapezoid(y[msk], t[msk]) / Te)

    rms_ph = [math.sqrt(tavg(x * x)) for x in iph]
    idc_ac = math.sqrt(max(tavg(idc * idc) - tavg(idc) ** 2, 0.0))
    Tq = 1.5 * m.p * (m.psi + (m.Ld - m.Lq) * idv) * iqv
    cs_err = float(np.max(np.abs(cv * cv + sv * sv - 1.0)))
    M_idx = op["V"] / (Vdc / 2.0)
    cosphi = math.cos(math.atan2(op["vq"], op["vd"]) - math.atan2(op["i_q"], op["i_d"]))
    kol = ref.dc_link_cap_rms(op["I_rms"], M_idx, cosphi)
    res.add_metric("I_rms_a", "a상 RMS (샘플 적분)", rms_ph[0], "A", basis="b, c상: " + ", ".join(f"{x:.2f}" for x in rms_ph[1:]) + " A")
    res.add_metric("rip_pp", "a상 전류 리플 pp (기본파 제거)", float(np.max(rip[msk]) - np.min(rip[msk])), "A", basis="i_a − 기본파(평균 dq 전류로 복원)")
    res.add_metric("rip_rms", "a상 전류 리플 RMS", math.sqrt(tavg(rip * rip)), "A")
    res.add_metric("idc_ac", "DC-link 전류 AC RMS (커패시터가 감당할 성분)", idc_ac, "A", ref=kol, ref_label="Kolar 폐형식 (A: 무한 펄스비·정현 전류)", tol=0.03 if (natural and lin_ok) else None, basis="i_dc − 평균의 RMS")
    res.add_metric("T_rip", "토크 리플 pp", float(np.max(Tq[msk]) - np.min(Tq[msk])), "Nm", basis="전류 리플이 만든 전자기 토크 리플 (기계 공진 미포함)")
    on_min, off_min = min_pulses(edges, S0, cyc * Te)
    tmin = v["t_min"]
    res.add_metric("pulse_min", "최소 펄스 폭 (ON/OFF 중 작은 값)", min(on_min, off_min), "s", basis=f"ON {on_min * 1e6:.3g} µs · OFF {off_min * 1e6:.3g} µs", note=f"허용 최소 {tmin * 1e6:g} µs (ASSUMED)")
    # checks
    res.add_check(Check("정상 주기해 잔차 (한 전기 주기)", "PASS" if resid < 1e-8 else "FAIL", resid, "rel", 1e-8, path="(i_d, i_q)에 대해 affine인 주기 map을 세 번 적분해 고정점을 풀고, 네 번째 적분으로 확인",
                        detail=f"x0 = ({x0[0]:.4f}, {x0[1]:.4f}) A, 주기 map 고유값 |λ| = {', '.join(f'{abs(z):.4f}' for z in np.linalg.eigvals(M))}"))
    errs = []
    for rt in (1e-7, 1e-9):
        xr = rk45_dq_path(m, we, Vdc, edges, S0, x0, Te, rt)
        errs.append(float(np.max(np.abs(xr - z1[:2])) / Iscale))
    res.add_check(Check("독립 solver: 손으로 쓴 dq ODE (RK45, cos/sin(ω_e t) 직접)", "PASS" if errs[-1] < 1e-6 and errs[-1] <= errs[0] + 1e-12 else "FAIL", errs[-1], "rel", 1e-6,
                        path="2상태 비자율 ODE + 스위칭 상태별 v_αβ를 RK45로 rtol 1e-7 → 1e-9 적분한 한 주기 끝 전류 vs 정확 해(4상태 자율 LTI, 행렬지수)", independent=True,
                        detail="상대오차: " + ", ".join(f"{e:.2e}" for e in errs)))
    led = energy_ledger(tr, sysm, w0, w1, ["p_dc"], ["p_em"], ["p_cu"], rated_power=max(abs(P_ac_an), 1.0))
    res.add_check(ledger_check(led, what="DC 포트(스위치 함수×상전류) − 기계 − 동손 − ΔW: "))
    res.add_check(Check("회전자각 발진기 보존 |cos²+sin²−1|", "PASS" if cs_err < 1e-10 else "FAIL", cs_err, "", 1e-10, path="발진기 상태 (cos θ, sin θ)의 행렬지수 전파 — 표본 전체 최대", detail="각도를 상태로 넣어 모든 스위칭 상태를 LTI로 만든 방식의 정확성 확인"))
    res.add_check(check_close("PWM 기본파 정상분 (펄스 Fourier) vs 스위칭 해 평균 v_d, v_q", abs(pos), math.hypot(mvd, mvq), 1e-9, "edge 시각만으로 한 해석 Fourier 적분 vs 행렬지수 1차 모멘트", False, "V"))
    # series and plots
    per = cyc * Te
    names = [("ia", "i_a", iph[0]), ("ib", "i_b", iph[1]), ("ic", "i_c", iph[2])]
    for key, lab, y in names:
        xs, ys = decimate_minmax(t, y, 3000)
        res.add_series(key, lab, "A", xs, ys)
    res.add_series("ia_f", "i_a 기본파", "A", t[::4].tolist(), ia_f[::4].tolist(), dash=True)
    xs, ys = decimate_minmax(t, rip, 3000)
    res.add_series("rip", "i_a 리플 (i_a − 기본파)", "A", xs, ys)
    for key, lab, y in (("id", "i_d", idv), ("iq", "i_q", iqv), ("Te", "T_e", Tq)):
        xs, ys = decimate_minmax(t, y, 3000)
        res.add_series(key, lab, "Nm" if key == "Te" else "A", xs, ys)
    xs, ys = decimate_minmax(t, idc, 3000)
    res.add_series("idc", "i_dc (세 상측 스위치 합)", "A", xs, ys)
    xs, ys = decimate_minmax(t, idc_legs[0], 3000)
    res.add_series("idc_a", "a상 leg의 DC 전류 S_a·i_a", "A", xs, ys)
    res.add_series("vab", "v_ab (펄스)", "V", smp["t"], smp["vab"])
    vab_cmd = SQ3 * op["V"] * np.cos(th + ang_cmd + math.pi / 6)
    res.add_series("vab_cmd", "v_ab 지령 기본파", "V", t[::4].tolist(), vab_cmd[::4].tolist(), dash=True)
    labels = {"".join(str(s) for s in q): f"{n} ({''.join(str(s) for s in q)})" for q, n in SVM_NAME.items()}

    bands = bands_from_traj(tr, 0.0, per, labels, max_bands=2 * 3 * N * cyc + 8)
    g = "sw"
    res.add_plot("p_iabc", "상전류: 스위칭 해와 기본파", ["ia", "ib", "ic", "ia_f"], y_label="전류", y_unit="A", bands=bands, group=g, level="C", window=(w0, w1),
                 proved="이상 스위치 2-level 인버터가 해석 운전점의 전압을 PWM으로 만들 때 상전류가 기본파 위에 리플을 얹고 흐른다는 것을 정확 해로 보였다.",
                 not_yet="dead time·스위칭 과도·소자 전압강하가 없는 C 수준이다. 실제 전류제어기(폐루프)가 아니라 해석 전압을 개루프로 인가했다.")
    res.add_plot("p_rip", "a상 전류 리플 (기본파 제거)", ["rip"], y_label="리플", y_unit="A", bands=bands, group=g, level="C",
                 proved="리플은 스위칭 상태마다 모터 인덕턴스에 걸리는 전압 차가 만든다. 펄스비가 작을수록 크다.", not_yet="철손·고조파 손실은 계산하지 않았다.")
    res.add_plot("p_idq", "dq 전류: 스위칭 해 vs 해석 운전점", ["id", "iq"], y_label="전류", y_unit="A", bands=bands, group=g, level="C",
                 hlines=[{"y": v["id"], "label": "i_d 해석"}, {"y": op["i_q"], "label": "i_q 해석"}],
                 proved="회전좌표계에서 PWM 리플은 평균 주위의 진동이고, 한 주기 평균은 dq 모델이 LTI라서 평균 전압에 대한 정상응답과 같다 — natural 샘플링이면 해석 운전점과 일치한다.",
                 not_yet="과변조·샘플링 지연이 있으면 평균이 달라진다(preset 참조).")
    res.add_plot("p_idc", "DC-link 전류: leg별과 합계", ["idc", "idc_a"], y_label="i_dc", y_unit="A", bands=bands, group=g, level="C", hlines=[{"y": idc_mean, "label": f"평균 {idc_mean:.1f} A"}],
                 proved="DC 전류는 스위칭 상태에 따라 상전류 조각이 이어진 펄스열이다. 평균은 P_AC/V_dc, AC 성분은 DC-link 커패시터가 감당하며 Kolar 폐형식과 비교했다.",
                 not_yet="DC 전원은 이상 전압원이다. 배터리·케이블 임피던스와 커패시터 ESR·공진은 없다.")
    res.add_plot("p_vab", "선간전압 v_ab: 펄스와 지령 기본파", ["vab", "vab_cmd"], y_label="v_ab", y_unit="V", bands=bands, group=g, level="C",
                 proved="선간전압은 ±V_dc·0의 3준위 펄스이며 그 기본파가 지령과 같다는 것을 edge 시각의 해석 Fourier로 확인했다.", not_yet="")
    res.add_plot("p_torque", "전자기 토크", ["Te"], y_label="T_e", y_unit="Nm", bands=bands, group=g, level="C", hlines=[{"y": op["T"], "label": "요구"}],
                 proved="전류 리플이 토크 리플을 만들며 평균 토크는 요구값과 같다.", not_yet="기계 관성·축 공진·NVH는 모델 밖이다.")
    hs, amp = pulse_spectrum(edges, S0, Vdc, we, Te, min(3 * N + 6, 400))
    floor = 1e-9 * max(float(amp[0]), 1.0)
    keep = amp > floor  # even and triplen harmonics are exactly zero in v_an; they are not plotted
    res.add_series("spec", "|V_h| (v_an, 0이 아닌 차수만)", "V", hs[keep].tolist(), amp[keep].tolist(), style="points")
    res.add_plot("p_spec", "상전압 고조파 스펙트럼 (edge 시각의 해석 Fourier)", ["spec"], x_label="차수 h (f/f_e)", x_unit="", y_label="진폭", y_unit="V", kind="xy", log_y=True, level="C",
                 vlines=[{"x": N, "label": "f_sw"}, {"x": 2 * N, "label": "2f_sw"}], markers=[{"x": 1, "y": float(amp[0]), "label": "기본파"}],
                 proved="기본파 외 에너지는 캐리어(N)와 그 배수 주위 측대파에 몰린다. N이 작으면 측대파가 기본파에 가까워 필터링·리플이 불리하다.", not_yet="EMI 대역·측정 조건과 무관한 이상 펄스 스펙트럼이다.")
    res.circuit = {"diagram": inverter_circuit(v).to_json(), "intervals": bands, "plot_group": g}
    ok_pulse = min(on_min, off_min) >= tmin
    if not lin_ok:
        res.verdict("FAIL_CONSTRAINT", f"요구 {op['V']:.2f} V가 이 변조의 선형 한계 {(Vdc / SQ3 if zs == 'minmax' else Vdc / 2):.2f} V를 넘는다 — 과변조로 기본파가 {abs(pos):.2f} V에 그쳐 운전점에 도달하지 못한다")
    if not ok_pulse:
        res.verdict("FAIL_CONSTRAINT", f"최소 펄스 {min(on_min, off_min) * 1e9:.3g} ns < 허용 {tmin * 1e6:g} µs: 이상 스위치 모델은 계산하지만 실제 드라이버·dead time으로는 만들 수 없는 펄스다")
    if lin_ok and ok_pulse:
        if natural:
            res.verdict("PASS_WITHIN_MODEL", "natural 샘플링 PWM의 정확 스위칭 해가 해석 운전점·에너지 수지·독립 solver·Kolar 식과 일치 (이상 스위치 C 수준)")
        else:
            res.verdict("INFO", f"{method} 샘플링: 기본파 위상 {dang:+.2f}°, 평균 전류가 해석점에서 ({mid_ - v['id']:+.1f}, {miq - op['i_q']:+.1f}) A 이동 — 폐루프 전류제어가 보상해야 할 오차")
    res.assumptions += [
        "이상 스위치 (즉시 전환, dead time·전압강하·스위칭 손실 없음), 이상 DC 전압원",
        "선형 dq 모터 (포화·철손 없음), 일정 속도 (기계 동역학 없음)",
        "해석 운전점의 dq 전압을 개루프로 인가 (전류제어기 없음), 동기 PWM (f_sw = N·f_e)",
    ]
    res.not_valid_for += ["반도체 손실·온도 (FL02/FL03)", "dead time 전압오차·저속 왜곡", "EMI·고주파 공진", "폐루프 전류제어 성능 (FL06)"]
    res.interpretation = (
        f"PWM은 한 스위칭 주기 안에서 8개 전압 벡터를 섞어 평균 전압을 만든다. natural 샘플링에서는 그 기본파(정상분)가 지령 {op['V']:.2f} V와 같고, "
        "dq 모델이 일정 속도에서 선형·시불변이므로 한 주기 평균 전류는 평균 전압에 대한 정상응답, 즉 해석 운전점과 같다. 리플은 그 위에 얹힌 스위칭 성분이다. "
        f"DC 전류는 평균 {idc_mean:.1f} A의 펄스열이고 커패시터가 AC 성분 {idc_ac:.1f} A RMS를 감당한다. "
        f"펄스비 {N}가 3의 배수가 아니면 세 상의 펄스 배치가 달라 작은 역상분({abs(neg):.2f} V)이 생긴다. 반 주기 샘플링 지연은 전압 벡터를 ω_e·T_s/2만큼 늦춘다."
    )
    return res


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q = [
    Question(
        "“feasible로 나왔으니 이 250 kW 점으로 운전 맵을 확정하자”는 말에 어떻게 답하나?",
        "요구 438.545 V, 가용 438.786 V로 여유가 0.241 V뿐인 수학적 경계점이다. 같은 토크에서 ψ_m ±2 %, R_s 고온 +40 %, L_q ±5 % 중 하나만으로 여유가 음수가 된다. "
        "전류제어 과도, dead time 전압오차, 전압 측정 오차도 여유를 쓴다. 요구 여유를 정하고(예: 가용의 2 %), 같은 토크를 MTPA(−219 A) 쪽으로 옮기거나 derating을 검토한다. 포화가 있으면 측정 L_d/L_q 맵으로 다시 확인한다.",
        "The solver says the 250 kW point is feasible. Can we freeze the operating map on it?",
        "No. The margin is only 0.24 V, a mathematical boundary. A 2 % flux change, hot winding resistance or a 5 % change in the q-axis inductance each makes it negative, and current-control transients and dead time also consume voltage. I would require an explicit voltage reserve, move the same torque towards the MTPA point, or agree on derating, and re-check with saturated inductance maps.",
        ["0.241 V", "파라미터 공차·온도", "전류제어 여유", "강건하지 않음"],
        kind="pressure",
    ),
    Question(
        "모터 동손 5.018 kW는 인버터 출력에 포함되나? shaft·AC·DC 경계를 말하라.",
        "포함된다. 인버터 AC 출력 = 모터 단자 입력 = shaft 250 kW + 동손 5.018 kW = 최소 255.018 kW (철손·기계손실은 이 기준점에서 제외라 하한). "
        "DC 입력은 여기에 반도체 손실을 더한다: 채널 전도분 3·I_rms²·R = 1.071 kW를 넣으면 ≥ 256.09 kW, diode·dead time·스위칭 손실은 별도. 회생에서는 shaft → AC → DC 순서로 손실을 빼고 부호를 유지한다.",
        "Is the motor copper loss part of the inverter output? Walk through the shaft, AC and DC boundaries.",
        "Yes. The inverter AC output equals the motor terminal input, 250 kW at the shaft plus 5.018 kW of copper loss, so at least 255.018 kW, a lower bound because iron and mechanical losses are excluded here. The DC input adds the semiconductor losses: 1.071 kW of channel conduction gives at least 256.09 kW, with diode, dead-time and switching losses still to be added.",
        ["255.018 kW", "AC = shaft + 동손", "반도체 손실은 DC 쪽", "하한"],
    ),
    Question(
        "채널 전도손실을 3·I_rms²·R로 계산했다. 스위치가 6개이니 6을 곱해야 하지 않나?",
        "아니다. 매 순간 각 상에는 상측 또는 하측 한 스위치 위치로만 전류 경로가 생긴다(R은 스위치 위치당 값). 상 RMS 전류가 그 경로를 흐르므로 합은 3·I_rms²·R = 1.071 kW다. "
        "6을 곱하면 두 배로 중복된다. diode 도통(dead time)·스위칭·gate 손실은 따로 더한다.",
        "You computed channel conduction loss as 3·I²R. With six switches, shouldn't it be six?",
        "No. At any instant each phase conducts through exactly one switch position, upper or lower, so the phase RMS current flows through one resistance per phase. The total is three times I-squared R, 1.071 kW; multiplying by six double counts. Diode, dead-time and switching losses are added separately.",
        ["상마다 한 경로", "6 곱하지 않음", "diode·dead time·switching 별도"],
        kind="calc",
    ),
    Question(
        "V_dc가 760 V로 떨어지면 무엇이 달라지나?",
        "요구 전압 438.545 V는 그대로이고 가용 전압만 416.85 V로 줄어 21.7 V가 부족하다 — 같은 점은 불가능하다. 선택지: (a) 같은 토크를 더 음의 i_d로(이 점은 MTPA 오른쪽이라 −237 A까지는 전류가 거의 같지만, 여유를 요구하면 전류·동손이 는다), "
        "(b) 토크 derating, (c) 과변조(고조파·리플·손실·제어 비용). 각 선택을 전류·자속·열 제약과 함께 평가하고, 포화로 L_d/L_q가 바뀌는 문제는 선형 모델 밖이라고 말한다.",
        "What changes if the DC bus sags to 760 V?",
        "The required 438.5 V does not change, but the available voltage drops to 416.8 V, so the same point is 21.7 V short and infeasible. The options are more negative d-current for the same torque, torque derating, or overmodulation, each with its own current, loss and control cost, and saturation effects are outside the linear model.",
        ["416.85 V", "더 음의 i_d", "derating", "선형 모델 밖(포화)"],
    ),
    Question(
        "약계자는 공짜가 아니라는데, 이 운전점에서도 그런가?",
        "이 점의 i_d = −200 A는 MTPA(−219.35 A)보다 오른쪽이라, −237 A까지는 |i|가 472.3 → 472.2 A로 거의 변하지 않는다. MTPA를 지나 더 들어가면 같은 토크에 전류가 늘어 동손·전도손실·온도가 오른다. "
        "그래서 '약계자 = 손실 증가'는 MTPA를 지난 뒤의 이야기이고, 먼저 운전점이 MTPA 대비 어디 있는지 계산한다.",
        "Field weakening is said to be never free. Is that true at this operating point?",
        "Here the given d-current of −200 A lies on the low side of MTPA at about −219 A, so moving to −237 A barely changes the current magnitude. Only beyond MTPA does the same torque need more current and therefore more loss, so I first check where the point sits relative to MTPA.",
        ["MTPA −219 A", "그 이후 전류 증가", "위치를 먼저 계산"],
    ),
    Question(
        "10 kHz 스위칭, 400 Hz 기본파에서 DC-link 커패시터 RMS 전류는 어떻게 잡나?",
        "평균 DC 전류(≈ 318.8 A)가 아니라 AC 성분이다. Kolar 식 I_C = I_rms√(2M[√3/4π + cos²φ(√3/π − 9M/16)])에 M = 1.096, cos φ = 0.821을 넣으면 ≈ 151 A RMS이고, 정확 스위칭 해도 151.5 A다. "
        "배터리·케이블 임피던스가 이 전류를 얼마나 나눠 가지는지와 ESR 발열·수명은 따로 본다.",
        "How do you size the DC-link capacitor ripple current at 10 kHz switching and a 400 Hz fundamental?",
        "Not from the 319 A average. The capacitor carries the AC part of the DC current; the Kolar formula with M of 1.10 and a power factor of 0.82 gives about 151 A RMS, which the exact switching run reproduces. How much the battery path shares, and the ESR heating and lifetime, are separate checks.",
        ["≈151 A RMS", "M·cosφ 의존", "평균 전류와 다름"],
        kind="calc",
    ),
    Question(
        "SiC와 IGBT 인버터 효율을 어떻게 공정하게 비교하나?",
        "공통 f_sw에서 비교하면 소자 기술 차이를, 각자 최적 f_sw에서 비교하면 시스템 설계 차이를 본다 — 두 비교를 섞지 않는다. 모터 고조파 손실을 빼고 인버터 효율만으로 주행거리를 말하지 않는다. "
        "운전 사이클의 저부하 점과 회생 방향도 같은 경계로 포함한다.",
        "How would you compare SiC and IGBT inverter efficiency fairly?",
        "At a common switching frequency you see the device technology; at each device's best switching frequency you see the system design. I keep the two comparisons separate, include motor harmonic losses and the drive cycle's light-load and regenerative points, and never convert inverter efficiency directly into vehicle range.",
        ["공통 fs vs 최적 fs", "모터 고조파 손실", "주행거리 직결 금지"],
    ),
]

_common_params = motor_params()

EXPERIMENTS = [
    Experiment(
        key="dq_point",
        title="250 kW 운전점: 전압여유 0.241 V와 shaft·AC·DC 경계",
        goal=(
            "합성 IPMSM(p = 4, R_s 15 mΩ, L_d 0.15 mH, L_q 0.35 mH, ψ_m 0.115 Wb)의 6,000 rpm·250 kW·i_d = −200 A 운전점에서 "
            "i_q = 427.836 A, 요구 전압 438.545 V, 가용 전압 438.786 V를 계산하고, 여유 0.241 V가 강건한 점이 아님을 공차 민감도로 보인다. "
            "모터 동손 5.018 kW를 AC 출력에 포함해 shaft·AC·DC 경계를 나누고, 760 V sag에서 같은 점을 자동 feasible로 두지 않는다."
        ),
        params=_common_params + point_params() + [
            Param("R_sw", "스위치 위치 채널 저항 R", "Ω", 3.2e-3, "mΩ", vmin=0.0, vmax=1.0, source="TEXTBOOK", source_note="합성 3.2 mΩ (온도 미지정)", group="인버터"),
            Param("headroom", "강건 판정 요구 전압 여유 (가용 대비)", "%", 2.0, "%", vmin=0.0, vmax=30.0, source="ASSUMED", source_note="전류제어·dead time·공차 여유의 학습용 값", group="판정"),
            Param("Imax", "전류 한계 |i|max (대안 d용)", "A", 550.0, "A", vmin=1.0, vmax=5000.0, source="ASSUMED", source_note="교재에 값 없음", group="판정"),
        ],
        presets=[
            Preset("nominal", "교재 800 V, i_d = −200 A", {}, "교재 07장 기준점", ("nominal", "reference")),
            Preset("sag760", "V_dc 760 V (sag)", {"Vdc": 760.0}, "같은 점을 자동 feasible로 두지 않는다", ("failure", "reference")),
            Preset("mtpa", "같은 토크를 MTPA i_d = −219.35 A로", {"id": -219.35}, "여유 11.6 V", ("variant", "reference")),
            Preset("sag_fw", "760 V에서 i_d = −252.5 A (요구 여유 2 % 확보)", {"Vdc": 760.0, "id": -252.5}, "약계자 대안의 비용", ("variant",)),
            Preset("hot", "R_s +40 % (고온 구리 21 mΩ)", {"Rs": 21e-3}, "공차 하나로 불가", ("corner",)),
        ],
        run=run_dq_point,
        model_level="A (+ 독립 abc 재구성)",
        suggested_change="V_dc를 800 → 760 V로 낮춘다 (i_d·토크는 그대로).",
        prediction=Prediction(
            "V_dc가 800 → 760 V로 떨어지면 같은 i_d = −200 A, 250 kW 운전점은?",
            ["여유가 줄지만 여전히 가능", "전압이 부족해 불가능", "요구 전압도 같이 줄어 그대로 가능", "모르겠다"],
            "전압이 부족해 불가능",
            "요구 전압 |v| = 438.545 V는 속도·전류·L·ψ_m이 정하는 값이라 V_dc와 무관하다. 가용 전압은 760/√3 × 0.95 = 416.85 V로 줄어 21.7 V가 부족하다. "
            "더 음의 i_d, derating, 과변조 같은 대안은 전류·손실·제어 비용과 함께 따로 평가한다.",
            ["V_req", "V_avail", "margin"],
            handcalc=[{"key": "iq", "label": "i_q", "unit": "A"}, {"key": "V_req", "label": "요구 전압 |v|", "unit": "V"}, {"key": "P_cu", "label": "모터 동손", "unit": "W"}],
        ),
        suggested={"Vdc": 760.0},
        student=(
            "모터가 빨리 돌수록 역기전력(ω_e ψ_m)과 인덕턴스 전압(ω_e L i)이 커진다. 인버터가 만들 수 있는 전압에는 DC 전압이 정한 한계가 있어서 고속·고출력에서는 전압이 먼저 모자란다. "
            "이 운전점은 요구 전압이 가용 전압보다 겨우 0.24 V 작아 '계산상 가능'할 뿐이고, 모터 온도나 자석 공차가 조금만 달라도 불가능해진다. "
            "출력은 축(shaft)에서 250 kW, 모터 단자(AC)에서는 구리 손실만큼 더 큰 255.018 kW, 배터리(DC)에서는 반도체 손실까지 더한 값으로 나눠 말한다."
        ),
        expert=(
            "① 판정 순서는 토크 → i_q → v_d, v_q → |v| 대 V_dc/√3·m이다. 경계점은 MARGINAL로 분리해 PASS와 섞지 않는다. "
            "② 같은 토크에서 ψ_m −2 %는 i_q를 늘려 |v_d| = ω_e L_q i_q를 키우므로 오히려 전압이 늘고, +2 %는 줄인다 — 방향이 직관과 다를 수 있으니 민감도를 계산으로 본다. R_s 고온 +40 %, L_q ±5 %도 각각 여유를 넘는다. "
            "③ 경계: shaft 250 kW → AC 255.018 kW(동손 포함, 철손 제외라 하한) → DC ≥ 256.09 kW(채널 전도분만). 채널 손실은 상마다 한 경로라 3·I²R이고 diode·dead time·스위칭 손실은 따로 더한다. "
            "④ i_d = −200 A는 이 토크의 MTPA(−219.35 A)도 아니다. MTPA로 옮기면 전류가 0.74 A 줄고 800 V 여유가 11.65 V가 된다 — 운전점 선택 자체가 설계변수다. "
            "⑤ abc 좌표에서 돌극 인덕턴스 행렬과 Faraday 미분으로 다시 계산한 전압·전력이 dq 폐형식과 일치한다(독립 경로). 포화로 L_d, L_q가 바뀌면 모든 숫자가 바뀐다."
        ),
        customer_ko=(
            "이 운전점은 계산상 전압여유가 0.24 V라서 양산 조건의 모터 공차·온도·전류제어 여유를 넣으면 유지되지 않습니다. 같은 토크를 i_d ≈ −219 A(MTPA) 부근으로 옮기면 전류는 거의 같고 여유가 11 V 이상 생기니, "
            "모터 공급사의 온도별 ψ_m·포화 L_d/L_q 표로 운전 맵을 다시 확인하시죠. 760 V sag에서는 같은 점이 불가능하므로 derating 또는 약계자 비용을 요구사항과 함께 정해야 합니다."
        ),
        customer_en=(
            "At this point the voltage margin is only 0.24 V, so it will not survive production tolerances, motor temperature or the current-control reserve. Moving the same torque to about −219 A of d-current, the MTPA point, "
            "keeps the current almost unchanged and gives more than 11 V of margin. I would re-check the operating map with the supplier's temperature-dependent flux and saturated inductance data. "
            "At a 760 V sag the same point is not feasible, so we need to agree on derating or on the cost of field weakening."
        ),
        questions=_Q[:5],
        circuit="fl04_boundary",
        textbook=[TB_07, TB_19],
        reference_presets=["nominal", "sag760", "mtpa"],
        claim_limit="선형 dq 정상상태(A). 포화·철손·전류제어 과도·반도체 손실 전체는 주장하지 않는다.",
    ),
    Experiment(
        key="dq_plane",
        title="i_d–i_q 평면: 토크 곡선·전압 한계·전류 원·MTPA",
        goal=(
            "같은 토크를 내는 (i_d, i_q) 곡선 위에 속도에 따라 줄어드는 전압 한계(R_s 포함)와 전류 한계(ASSUMED 550 A)를 그려, 800 V에서 경계에 걸린 지정점이 760 V에서 밖으로 나가는 것과 "
            "더 음의 i_d로 옮길 때의 전류 비용(MTPA 이후 증가)을 확인한다. 토크–속도 능력 곡선에서 전류 한계와 전압 한계가 각각 지배하는 구간을 본다. 포화는 선형 모델 밖이다."
        ),
        params=_common_params + point_params() + [
            Param("Vdc_sag", "비교 버스 전압 (sag)", "V", 760.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="760 V", group="인버터"),
            Param("Imax", "전류 한계 |i|max (phase peak)", "A", 550.0, "A", vmin=1.0, vmax=5000.0, source="ASSUMED", source_note="교재에 값 없음 — 모터·소자 정격에서 받아야 함", group="인버터"),
        ],
        presets=[
            Preset("nominal", "6,000 rpm, 800/760 V", {}, "교재 07장", ("nominal", "reference")),
            Preset("fast", "8,000 rpm (같은 250 kW)", {"rpm": 8000.0}, "고속: 타원 축소", ("variant", "reference")),
            Preset("imax600", "전류 한계 600 A", {"Imax": 600.0}, "", ("variant",)),
            Preset("slow", "2,000 rpm, 100 kW", {"rpm": 2000.0, "P_shaft": 100e3}, "저속: 전류 한계 지배", ("corner",)),
        ],
        run=run_dq_plane,
        model_level="A",
        suggested_change="속도를 6,000 → 8,000 rpm으로 올린다 (같은 250 kW).",
        prediction=Prediction(
            "같은 250 kW에서 속도를 6,000 → 8,000 rpm으로 올리면 요구 토크와 전압 한계 타원은?",
            ["토크 감소·타원 축소", "토크 증가·타원 확대", "토크 감소·타원 확대", "모르겠다"],
            "토크 감소·타원 축소",
            "T = P/ω_m이라 토크는 298 Nm로 줄지만, 전압 한계 타원의 크기는 V/ω_e에 비례해 줄어든다. 고속에서는 토크가 작아도 i_d를 더 음으로 가져가지 않으면(약계자) 전압이 모자란다.",
            ["T", "Tmax1", "id_sag"],
            handcalc=[{"key": "I_char", "label": "특성전류 ψ_m/L_d", "unit": "A"}, {"key": "id_mtpa", "label": "같은 토크의 MTPA i_d", "unit": "A"}],
        ),
        suggested={"rpm": 8000.0},
        student=(
            "원(전류 한계)은 모터·인버터가 견딜 전류를, 타원(전압 한계)은 인버터가 만들 수 있는 전압을 i_d–i_q 평면에 그린 것이다. 운전점은 두 영역이 겹치는 곳에만 있을 수 있다. "
            "속도가 빨라지면 타원이 작아지고, 타원의 중심(−ψ_m/L_d)을 향해 i_d를 음으로 옮겨야 전압이 들어온다. 같은 토크를 내는 곡선 위에서 전류가 가장 작은 점이 MTPA다."
        ),
        expert=(
            "① 전압 한계는 R_s 때문에 정확한 타원이 아니다(근사: (L_d i_d + ψ_m)² + (L_q i_q)² ≤ (V/ω_e)²). 여기서는 R_s를 넣은 2차식의 근으로 그린다. "
            "② 특성전류 ψ_m/L_d = 766.7 A가 전류 한계보다 크면 무한 속도까지 일정출력을 유지할 수 없다. ③ MTPA는 |i| 고정에서 토크 최대, 고속에서는 MTPV와 전류·전압 한계 교점이 궤적을 정한다. "
            "④ 교점·최대 토크를 서로 다른 두 경로(토크 곡선 따라 풀기 vs 전압 곡선 따라 풀기, 격자+정제 vs SLSQP)로 확인했다. ⑤ 실제 모터는 포화로 L_q가 전류에 따라 크게 줄어 MTPA·약계자 표를 측정/FEA로 만든다 — 선형 곡선으로 양산 맵을 확정하지 않는다."
        ),
        customer_ko=(
            "고속 영역의 가용 토크는 전압 한계가 정하고, 약계자로 i_d를 더 음으로 가져가면 전압은 들어오지만 MTPA를 지난 뒤부터는 같은 토크에 전류가 늘어 손실과 온도가 올라갑니다. "
            "포화가 있는 실제 모터에서는 L_d, L_q 맵을 받아 이 평면을 다시 그려 확인하시죠."
        ),
        customer_en=(
            "In the high-speed region the available torque is set by the voltage limit. Field weakening brings the voltage back inside the limit, but beyond the MTPA point the same torque needs more current, "
            "so losses and temperature rise. For the real motor I would redraw this plane with the saturated inductance maps."
        ),
        questions=[_Q[3], _Q[4]],
        circuit=None,
        textbook=[TB_07],
        reference_presets=["nominal", "fast"],
        claim_limit="선형 정상상태 평면(A). 포화·열 한계·과변조는 주장하지 않는다.",
    ),
    Experiment(
        key="inverter_switching",
        title="짧은 인버터 스위칭: PWM이 만든 평균 전압·리플·DC-link 전류",
        goal=(
            "3상 2-level 인버터(이상 스위치, 동기 PWM N = 25 → 10 kHz)가 해석 운전점의 dq 전압을 만들 때, 몇 전기 주기의 정확 스위칭 해에서 상전류 리플, leg별·합계 DC-link 전류, "
            "상전압 기본파 vs 지령, 스위칭 해의 평균 dq 전류 vs 해석점, 에너지 원장을 확인한다. 회전자각을 LTI 발진기 상태로 넣어 역기전력을 정확히 만든다."
        ),
        params=_common_params + point_params()[:4] + [
            Param("N", "펄스비 N = f_sw/f_e (동기 PWM)", "", 25, "", vmin=5, vmax=201, kind="int", source="ASSUMED", source_note="25 → 6,000 rpm에서 10 kHz", group="PWM"),
            Param("pwm", "PWM 샘플링", "", "natural", kind="choice", choices=[("natural", "natural (연속 지령 vs 캐리어)"), ("regular", "regular (주기 시작점 샘플, 지연 보상 없음)"), ("regular_comp", "regular (펄스 중앙 샘플, 지연 보상)")], source="ASSUMED", group="PWM"),
            Param("zs", "영상분 주입", "", "minmax", kind="choice", choices=[("minmax", "min-max 주입 (SVPWM 등가)"), ("none", "없음 (정현 PWM)")], source="ASSUMED", group="PWM"),
            Param("cycles", "표시할 전기 주기 수", "", 2, "", vmin=1, vmax=3, kind="int", source="ASSUMED", group="시뮬레이션"),
            Param("t_min", "허용 최소 펄스 폭", "s", 1e-6, "µs", vmin=0.0, vmax=20e-6, source="ASSUMED", source_note="드라이버·dead time을 고려한 학습용 값", group="PWM"),
        ],
        presets=[
            Preset("nominal", "natural SVPWM, N = 25 (10 kHz)", {}, "해석점 재현", ("nominal", "reference")),
            Preset("N27", "N = 27 (3의 배수)", {"N": 27}, "역상분 소멸", ("variant", "reference")),
            Preset("regular", "regular 샘플링 (지연 보상 없음)", {"pwm": "regular"}, "7.2° 지연 → 운전점 이동", ("variant",)),
            Preset("regular_comp", "regular 샘플링 (지연 보상)", {"pwm": "regular_comp"}, "", ("variant",)),
            Preset("spwm", "정현 PWM (영상분 주입 없음)", {"zs": "none"}, "선형 한계 V_dc/2 → 과변조", ("failure", "reference")),
            Preset("sag760", "V_dc 760 V", {"Vdc": 760.0}, "최소 펄스 28 ns", ("failure", "reference")),
            Preset("N11", "N = 11 (4.4 kHz)", {"N": 11}, "낮은 펄스비: 큰 리플", ("corner",)),
        ],
        run=run_switching,
        model_level="C (이상 스위치)",
        suggested_change="PWM 샘플링을 natural → regular(주기 시작점 샘플, 지연 보상 없음)로 바꾼다.",
        prediction=Prediction(
            "PWM 지령을 반 스위칭 주기 늦게 샘플링(보상 없음)하면 한 주기 평균 dq 전류는?",
            ["그대로 해석점", "전압 벡터가 늦어져 운전점이 크게 이동", "리플만 커진다", "모르겠다"],
            "전압 벡터가 늦어져 운전점이 크게 이동",
            "regular 샘플링은 지령을 캐리어 꼭대기에서 샘플해 펄스 중앙까지 T_s/2 늦게 낸다. 400 Hz·10 kHz에서 ω_e T_s/2 = 7.2°다. 고속에서 전압은 대부분 역기전력·인덕턴스 전압이라 "
            "작은 각 오차가 ΔI ≈ |v|·Δθ/(ω_e L) 수준의 큰 전류 오차가 되어, 평균 i_d가 약 +121 A, 토크가 약 22 % 변한다. 폐루프 전류제어가 보상해야 하지만 이 점은 전압 여유가 0.24 V뿐이다.",
            ["dang", "id_mean", "iq_mean", "T_mean"],
            handcalc=[{"key": "idc_mean", "label": "DC-link 평균 전류", "unit": "A"}, {"key": "idc_ac", "label": "DC-link AC RMS (Kolar)", "unit": "A"}],
        ),
        suggested={"pwm": "regular"},
        student=(
            "인버터는 각 상을 DC의 +쪽 또는 −쪽에 붙였다 떼는 스위치 세 쌍이다. 붙어 있는 시간 비율을 사인파 모양으로 바꾸면 모터 인덕턴스가 전류를 부드럽게 만들어, "
            "평균적으로는 원하는 사인파 전압을 받은 것처럼 전류가 흐르고 스위칭 주파수의 리플이 남는다. 배터리 쪽 전류는 켜진 상측 스위치들의 상전류를 이어 붙인 펄스열이다."
        ),
        expert=(
            "① 모델: 회전자각 θ를 (cos θ, sin θ) 발진기 상태로 넣으면 스위칭 상태마다 dq 방정식이 LTI가 되어 행렬지수로 정확히 풀린다(abc로 풀면 돌극 L(θ) 때문에 시변계). "
            "② 평균 dq 전류 = 평균 dq 전압에 대한 정상응답(LTI). natural 샘플링의 기본파 정상분은 지령과 같아(min-max 주입의 측대파 누설 ~5e-5) 해석점을 재현한다. "
            "③ DC-link 커패시터는 평균이 아니라 AC 성분을 감당하며, Kolar 식 I_C = I_rms√(2M[√3/4π + cos²φ(√3/π − 9M/16)])과 정확 해가 0.1 % 안에서 맞는다. "
            "④ 동기 PWM에서 N이 3의 배수가 아니면 세 상의 펄스 배치가 달라 역상분이 생긴다(N = 25 → 0.66 V, N = 27 → 0). "
            "⑤ 760 V에서도 같은 점은 선형 SVPWM 한계 안이지만 최소 펄스 28 ns가 필요해 실제로는 만들 수 없다. 정현 PWM(주입 없음)의 선형 한계는 V_dc/2 = 400 V라 이 점에 닿지 못한다."
        ),
        customer_ko=(
            "400 Hz 기본파에 10 kHz면 펄스비가 25로 낮아서 샘플링 지연(약 7°)과 역상분 같은 동기 PWM 효과가 무시되지 않습니다. DC-link 커패시터는 평균 319 A가 아니라 AC 성분 약 151 A RMS로 선정하시고, "
            "dead time과 최소 펄스 제한을 넣은 상태에서 전압 여유를 다시 확인하시죠."
        ),
        customer_en=(
            "With a 400 Hz fundamental and 10 kHz switching the pulse ratio is only 25, so the sampling delay of about seven degrees and synchronous-PWM effects such as a small negative-sequence voltage are not negligible. "
            "Size the DC-link capacitor for the ripple current of about 151 A RMS, not the 319 A average, and re-check the voltage margin once dead time and the minimum pulse width are included."
        ),
        questions=[_Q[5], _Q[6]],
        circuit="inv3",
        textbook=[TB_07, TB_19],
        reference_presets=["nominal", "N27", "spwm", "sag760"],
        claim_limit="이상 스위치(C) 파형·평균·리플. dead time·스위칭 손실·EMI·폐루프 성능은 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL04",
    title="인버터 — 알고 있는 것을 더 날카롭게 증명하기",
    title_en="Inverter: dq operating point, voltage margin and a short switching run",
    track="basic",
    order=4,
    path_note="14일 경로 3일차 (06 손실·열, 07 인버터)",
    textbook=[TB_07, TB_19],
    prerequisites=["FL01"],
    summary="dq 운전점 → 전압여유와 공차 민감도 → shaft/AC/DC 경계 → i_d–i_q 평면 → 정확 스위칭 해로 평균·리플·DC-link 전류 확인. 경계점과 강건한 점을 구분한다.",
    experiments=EXPERIMENTS,
    minimum_scope="dq solver·loss 경계·짧은 inverter switching; 438.545 V·전압여유·출력경계 (교재 19장 표); 760 V sag에서 같은 점 불가",
    claim_limits=[
        "선형 L_d·L_q·ψ_m (포화·교차결합·온도 의존 없음)",
        "반도체 손실은 채널 전도분만; diode·dead time·스위칭 손실은 별도",
        "스위칭 모델은 이상 스위치·개루프 전압 인가 (C 수준)",
        "모터 파라미터·공차·전류 한계는 합성 학습 값",
    ],
    test_paths=["tests/test_fl04.py"],
)
