"""DAB machinery shared by FL08 and EX06 (helper module, not a lab).

Bridge output uses the textbook E06 switching function
    b(theta; w, c) = 1/2 [ s(theta - c + w/2) - s(theta - c - w/2) ],
a pulse of width w centred at c (+1) and at c + pi (-1); w = pi is the two-level SPS square
wave.  Two leg patterns produce the same b(theta):
    'alternating'  legs A/B are 50 % square waves shifted by -/+ w/2 (zero states alternate
                   upper/lower) - the usual phase-shift implementation;
    'fixed_lower'  both legs low during every zero state (legs pulse with width w).
The patterns differ in which switches commutate and at what current, not in the port waveform.

Three independent computation paths:
  * closed form (reference/dab.py, SPS only),
  * exact piecewise-linear integration of the ideal waveforms (engine/pwl.py), zero-DC by
    half-wave antisymmetry,
  * switched-affine simulation (engine/switched.py) with optional series R and magnetizing Lm,
    periodic state from the half-wave antisymmetry condition x(T/2) = -x(0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..engine.pwl import PWL
from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ._common import sym_linear

PI = math.pi
TWO_PI = 2 * math.pi


def wrap(theta: float) -> float:
    x = math.fmod(theta, TWO_PI)
    return x + TWO_PI if x < 0 else x


def bridge_level(theta: float, w: float, c: float) -> int:
    psi = wrap(theta - c)
    if psi < w / 2 - 1e-15 or psi >= TWO_PI - w / 2 - 1e-15:
        return 1
    if PI - w / 2 - 1e-15 <= psi < PI + w / 2 - 1e-15:
        return -1
    return 0


def leg_states(theta: float, w: float, c: float, pattern: str) -> tuple[int, int]:
    """(leg A, leg B) states: +1 = upper switch on, -1 = lower switch on."""
    if pattern == "fixed_lower":
        a = 1 if wrap(theta - (c - w / 2)) < w else -1
        b_ = 1 if wrap(theta - (c + PI - w / 2)) < w else -1
        return a, b_
    a = 1 if wrap(theta - (c - w / 2)) < PI else -1
    b_ = 1 if wrap(theta - (c + w / 2)) < PI else -1
    return a, b_


def leg_edges(w: float, c: float, pattern: str) -> list[tuple[float, str, int]]:
    """Leg switching instants in [0, 2pi): (theta, 'A'|'B', +1 rising / -1 falling)."""
    if pattern == "fixed_lower":
        ev = [(c - w / 2, "A", 1), (c + w / 2, "A", -1), (c + PI - w / 2, "B", 1), (c + PI + w / 2, "B", -1)]
    else:
        ev = [(c - w / 2, "A", 1), (c - w / 2 + PI, "A", -1), (c + w / 2, "B", 1), (c + w / 2 + PI, "B", -1)]
    return sorted((wrap(th), leg, d) for th, leg, d in ev)


@dataclass
class Modulation:
    w1: float = PI
    w2: float = PI
    phi: float = 0.0
    c1: float = 0.0  # centre of the primary positive pulse (the secondary one is at c1 + phi)
    pattern1: str = "alternating"
    pattern2: str = "alternating"

    @property
    def c2(self) -> float:
        return self.c1 + self.phi

    def breaks(self) -> list[float]:
        th = {0.0, TWO_PI}
        for w, c in ((self.w1, self.c1), (self.w2, self.c2)):
            for e in (c - w / 2, c + w / 2, c + PI - w / 2, c + PI + w / 2):
                th.add(wrap(e))
        return sorted(t for t in th if 0.0 <= t <= TWO_PI)

    def levels(self, theta: float) -> tuple[int, int]:
        return bridge_level(theta, self.w1, self.c1), bridge_level(theta, self.w2, self.c2)


SPS_TEXTBOOK_C1 = PI / 2  # v1 = +V1 on [0, pi) as in the textbook's 4-interval table


@dataclass
class Waves:
    """Ideal periodic waveforms of one period from exact piecewise-linear integration."""

    T: float
    t: np.ndarray
    b1: list[int]
    b2: list[int]
    iL: PWL
    im: PWL | None
    i2: PWL
    v1: PWL
    v2: PWL
    P1: float
    P2: float
    extra: dict = field(default_factory=dict)

    @property
    def Irms(self) -> float:
        return self.iL.rms()

    @property
    def Ipk(self) -> float:
        return self.iL.max_abs()


def pwl_waves(V1: float, V2: float, L: float, fs: float, mod: Modulation, Lm: float | None = None) -> Waves:
    """Exact ideal waveforms (lossless) with the zero-mean (half-wave antisymmetric) current."""
    T = 1.0 / fs
    w = TWO_PI * fs
    th = mod.breaks()
    t = np.array(th) / w
    b1, b2 = [], []
    for a, b in zip(th[:-1], th[1:]):
        l1, l2 = mod.levels(0.5 * (a + b))
        b1.append(l1)
        b2.append(l2)
    v1s = np.array([V1 * x for x in b1], dtype=float)
    v2s = np.array([V2 * x for x in b2], dtype=float)
    iL = PWL.from_slopes(t, (v1s - v2s) / L, 0.0)
    iL = iL.shifted(-iL.mean())
    if Lm:
        im = PWL.from_slopes(t, v2s / Lm, 0.0)
        im = im.shifted(-im.mean())
        i2 = PWL(t, iL.y0 - im.y0, iL.y1 - im.y1)
    else:
        im = None
        i2 = iL
    v1 = PWL.step(t, v1s)
    v2 = PWL.step(t, v2s)
    P1 = v1.product_integral(iL) / T
    P2 = v2.product_integral(i2) / T
    return Waves(T, t, b1, b2, iL, im, i2, v1, v2, P1, P2)


def phi_for_power(V1, V2, L, fs, P, w1=PI, w2=PI, c1=0.0, lo=0.0, hi=PI / 2, pattern1="alternating", pattern2="alternating", Lm=None) -> float | None:
    """Outer phase giving power P on the monotonic branch [lo, hi] (bisection on the exact PWL power)."""

    def f(ph):
        return pwl_waves(V1, V2, L, fs, Modulation(w1, w2, ph, c1, pattern1, pattern2), Lm).P2 - P

    fa, fb = f(lo), f(hi)
    if fa * fb > 0:
        return None
    a, b = lo, hi
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# --------------------------------------------------------------------------------------
# Switched-affine simulation
# --------------------------------------------------------------------------------------


class DABSystem(HybridSystem):
    """Two 3-level bridges, series L (+R) and magnetizing Lm across the transformer (secondary side of L).

    x = [i_L] or [i_L, i_m]; q = (b1, b2).  Primary-referred; the secondary actual current is
    n (i_L - i_m) and the secondary DC-port current n b2 (i_L - i_m).
    ``mod_of_cycle(k)`` returns the Modulation of cycle k (allows mode transitions).
    """

    def __init__(self, V1, V2, L, fs, R=0.0, Lm=None, n=1.0, mod_of_cycle=None, key=""):
        self.V1, self.V2, self.L, self.fs, self.R, self.Lm, self.n = V1, V2, L, fs, R, Lm, n
        self.T = 1.0 / fs
        self.mod_of_cycle = mod_of_cycle or (lambda k: Modulation())
        self.state_names = ("i_L", "i_m") if Lm else ("i_L",)
        self._key = f"dab|{V1}|{V2}|{L}|{fs}|{R}|{Lm}|{key}"

    def mode(self, q):
        b1, b2 = q
        if self.Lm:
            A = [[-self.R / self.L, 0.0], [0.0, 0.0]]
            b = [(self.V1 * b1 - self.V2 * b2) / self.L, self.V2 * b2 / self.Lm]
        else:
            A = [[-self.R / self.L]]
            b = [(self.V1 * b1 - self.V2 * b2) / self.L]
        return AffineMode(f"{self._key}|{b1}|{b2}", A, b)

    def q_at(self, t: float):
        k = int(math.floor(t / self.T + 1e-12))
        m = self.mod_of_cycle(k)
        th = TWO_PI * self.fs * (t - k * self.T)
        return m.levels(th + 1e-12)

    def gate_schedule(self, t0, t1):
        ev = []
        k0 = int(math.floor(t0 / self.T + 1e-12))
        k1 = int(math.ceil(t1 / self.T + 1e-12))
        for k in range(k0, k1 + 1):
            m = self.mod_of_cycle(k)
            prev = self.mod_of_cycle(k - 1)
            base = k * self.T
            for th in m.breaks():
                tt = base + th / (TWO_PI * self.fs)
                if not (t0 - 1e-15 <= tt < t1):
                    continue
                if th >= TWO_PI - 1e-15:
                    continue
                lv = m.levels(th + 1e-9)
                ev.append((tt, (lambda q, lv=lv: lv)))
            if prev is not m and t0 - 1e-15 <= base < t1:
                lv = m.levels(1e-9)
                ev.append((base, (lambda q, lv=lv: lv)))
        return ev

    def outputs(self, q):
        b1, b2 = q
        m = 3 if self.Lm else 2
        e_iL = np.zeros(m)
        e_iL[0] = 1.0
        e_im = np.zeros(m)
        if self.Lm:
            e_im[1] = 1.0
        one = np.zeros(m)
        one[-1] = 1.0
        i2 = e_iL - e_im
        return {
            "iL": e_iL,
            "im": e_im,
            "i2": i2,
            "i2_act": self.n * i2,
            "v1": self.V1 * b1 * one,
            "v2": self.V2 * b2 * one,
            "vL": (self.V1 * b1 - self.V2 * b2) * one - self.R * e_iL,
            "idc1": b1 * e_iL,
            "idc2_act": self.n * b2 * i2,
        }

    def stored_energy(self):
        if self.Lm:
            return np.diag([self.L, self.Lm, 0.0])
        return np.diag([self.L, 0.0])

    def powers(self, q):
        b1, b2 = q
        m = 3 if self.Lm else 2
        e_iL = np.zeros(m)
        e_iL[0] = 1.0
        i2 = e_iL.copy()
        if self.Lm:
            i2[1] = -1.0
        return {"p1": sym_linear(e_iL, self.V1 * b1), "p2": sym_linear(i2, self.V2 * b2), "pR": self.R * np.outer(e_iL, e_iL)}

    def describe(self, q):
        return f"{q[0]:+d}|{q[1]:+d}"


def half_wave_periodic(sys: DABSystem, q0) -> np.ndarray:
    """Periodic state from x(T/2) = -x(0) for the affine map x(T/2) = M x0 + c (fixed event times)."""
    n = len(sys.state_names)
    half = sys.T / 2
    c = simulate(sys, q0, np.zeros(n), 0.0, half).z_end[:-1]
    M = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        M[:, i] = simulate(sys, q0, e, 0.0, half).z_end[:-1] - c
    return np.linalg.solve(M + np.eye(n), -c)


# --------------------------------------------------------------------------------------
# Commutation screen (per leg, per edge)
# --------------------------------------------------------------------------------------


def coss_Q(v, C0, V0):
    return 2.0 * C0 * V0 * (math.sqrt(1.0 + v / V0) - 1.0)


@dataclass
class EdgeReport:
    bridge: str
    leg: str
    theta: float
    rising: bool
    i_out: float  # current flowing OUT of the leg node into the circuit, actual units
    sign_ok: bool
    q_av: float
    q_req: float
    m_q: float
    status: str


def commutation_screen(w: Waves, mod: Modulation, fs: float, n: float, V1: float, VL: float, coss_hv, coss_lv, td_hv: float, td_lv: float, i_tiny: float = 1e-6) -> list[EdgeReport]:
    """Sign and charge screen for every leg transition (SCREEN_ONLY: current held at its edge value).

    A rising leg node needs current flowing INTO the node (out-of-node current < 0).
    Primary legs: out-of-node current A = +i_L, B = -i_L.  Secondary legs (actual units):
    C = -n i_2, D = +n i_2, with i_2 = i_L - i_m primary-referred.
    """
    out: list[EdgeReport] = []
    om = TWO_PI * fs
    q_hv = 2.0 * coss_Q(V1, *coss_hv)
    q_lv = 2.0 * coss_Q(VL, *coss_lv)
    for bridge, width, centre, pattern in (("1차", mod.w1, mod.c1, mod.pattern1), ("2차", mod.w2, mod.c2, mod.pattern2)):
        for th, leg, d in leg_edges(width, centre, pattern):
            tt = th / om
            if bridge == "1차":
                i = w.iL.value_at(tt)
                i_out = i if leg == "A" else -i
                qr, td = q_hv, td_hv
                legn = leg
            else:
                i = n * w.i2.value_at(tt)
                i_out = -i if leg == "A" else i
                qr, td = q_lv, td_lv
                legn = {"A": "C", "B": "D"}[leg]
            rising = d > 0
            if abs(i_out) <= i_tiny:
                ok = False
                st = "ZERO_CURRENT"
            else:
                ok = (i_out < 0) if rising else (i_out > 0)
                st = "SIGN_FAIL" if not ok else None
            qa = abs(i_out) * td
            mq = (qa - qr) / qr
            if st is None:
                st = "CHARGE_SHORT" if mq < 0 else "SCREEN_PASS"
            out.append(EdgeReport(bridge, legn, th, rising, i_out, ok, qa, qr, mq, st))
    return out


EDGE_KO = {"SCREEN_PASS": "부호·전하 screen 통과", "CHARGE_SHORT": "부호는 맞지만 전하 부족 (부분 전환 예상)", "SIGN_FAIL": "전류 방향 반대 → hard switching", "ZERO_CURRENT": "전류 ≈ 0 → 전하 없음"}


def min_pulse_ok(w: float, fs: float, t_min: float) -> tuple[bool, str]:
    om = TWO_PI * fs
    pulse = w / om
    zero = (PI - w) / om
    if w < PI - 1e-12 and (pulse < t_min or zero < t_min):
        return False, f"펄스 {pulse * 1e9:.0f} ns / zero {zero * 1e9:.0f} ns < 최소 {t_min * 1e9:.0f} ns"
    return True, ""


# --------------------------------------------------------------------------------------
# Circuit diagram
# --------------------------------------------------------------------------------------


def dab_circuit(V1: float, VL: float, n_label: str, L: float, Lm: float | None) -> Circuit:
    c = Circuit("dab", 760, 300, title="DAB (primary-referred L, 이상 transformer n = Np/Ns)")
    v1 = c.add("vsource", "V1", 40, 150, 90, "V_H", f"{V1:g} V")
    xa, xb = 120, 200
    qa_h = c.add("nmos", "QAH", xa, 90, 90, "A↑", lpos=(xa - 26, 84, "end"))
    qa_l = c.add("nmos", "QAL", xa, 210, 90, "A↓", lpos=(xa - 26, 204, "end"))
    qb_h = c.add("nmos", "QBH", xb, 90, 90, "B↑", lpos=(xb - 26, 84, "end"))
    qb_l = c.add("nmos", "QBL", xb, 210, 90, "B↓", lpos=(xb - 26, 204, "end"))
    lx = c.add("inductor", "L", 290, 135, 0, "L (1차 환산)", f"{L * 1e6:g} µH", lpos=(290, 106, "middle"))
    tr = c.add("transformer", "T", 390, 150, 0, f"n = {n_label}", "")
    xc, xd = 480, 560
    qc_h = c.add("nmos", "QCH", xc, 90, 90, "C↑", lpos=(xc - 26, 84, "end"))
    qc_l = c.add("nmos", "QCL", xc, 210, 90, "C↓", lpos=(xc - 26, 204, "end"))
    qd_h = c.add("nmos", "QDH", xd, 90, 90, "D↑", lpos=(xd - 26, 84, "end"))
    qd_l = c.add("nmos", "QDL", xd, 210, 90, "D↓", lpos=(xd - 26, 204, "end"))
    vl = c.add("battery", "VL", 680, 150, 90, "V_L", f"{VL:g} V")
    # primary rails
    c.wire("w_p_top", v1["a"], (40, 40), (xb, 40))
    c.wire("w_pa_top", (xa, 40), qa_h["a"])
    c.wire("w_pb_top", (xb, 40), qb_h["a"])
    c.wire("w_p_bot", v1["b"], (40, 260), (xb, 260))
    c.wire("w_pa_bot", qa_l["b"], (xa, 260))
    c.wire("w_pb_bot", qb_l["b"], (xb, 260))
    c.wire("w_pa_mid", qa_h["b"], qa_l["a"])
    c.wire("w_pb_mid", qb_h["b"], qb_l["a"])
    c.wire("w_a_L", (xa, 150), (xa + 30, 150), (xa + 30, 135), lx["a"])
    c.wire("w_L_T", lx["b"], (345, 135), (345, 120), (368, 120))
    c.wire("w_b_T", (xb, 150), (240, 150), (240, 180), (368, 180))
    # secondary
    c.wire("w_T_c", (412, 120), (xc - 30, 120), (xc - 30, 150), (xc, 150))
    c.wire("w_T_d", (412, 180), (425, 180), (425, 165), (530, 165), (530, 150), (xd, 150))
    c.wire("w_sc_mid", qc_h["b"], qc_l["a"])
    c.wire("w_sd_mid", qd_h["b"], qd_l["a"])
    c.wire("w_s_top", (xc, 40), (680, 40), vl["a"])
    c.wire("w_sc_top", (xc, 40), qc_h["a"])
    c.wire("w_sd_top", (xd, 40), qd_h["a"])
    c.wire("w_s_bot", (xc, 260), (680, 260), vl["b"])
    c.wire("w_sc_bot", qc_l["b"], (xc, 260))
    c.wire("w_sd_bot", qd_l["b"], (xd, 260))
    c.dot((xa, 150), (xb, 150), (xc, 150), (xd, 150), (xa, 40), (xa, 260), (xc, 40), (xc, 260))
    if Lm:
        c.text(390, 222, f"L_m = {Lm * 1e6:g} µH (변압기 여자, 2차측 전압이 결정)", "node")
    c.probe("pL", "iL", 228, 135, "right", "i_L")
    c.probe("p2", "i2_act", 433, 120, "right", "i_s (실제)")
    c.text(150, 290, "1차 bridge (v₁)", "node")
    c.text(520, 290, "2차 bridge (v₂)", "node")
    # modes keyed by leg states "A B | C D" with + = upper on
    for a in (1, -1):
        for b in (1, -1):
            for cc in (1, -1):
                for d in (1, -1):
                    key = f"{'+' if a > 0 else '-'}{'+' if b > 0 else '-'}|{'+' if cc > 0 else '-'}{'+' if d > 0 else '-'}"
                    act = ["L", "T", "w_a_L", "w_L_T", "w_b_T", "w_T_c", "w_T_d", "w_pa_mid", "w_pb_mid", "w_sc_mid", "w_sd_mid"]
                    act += ["QAH" if a > 0 else "QAL", "QBH" if b > 0 else "QBL", "QCH" if cc > 0 else "QCL", "QDH" if d > 0 else "QDL"]
                    dim = ["QAL" if a > 0 else "QAH", "QBL" if b > 0 else "QBH", "QCL" if cc > 0 else "QCH", "QDL" if d > 0 else "QDH"]
                    b1 = (a - b) // 2
                    b2 = (cc - d) // 2
                    if b1 != 0:
                        act += ["V1", "w_p_top", "w_p_bot", "w_pa_top" if a > 0 else "w_pa_bot", "w_pb_top" if b > 0 else "w_pb_bot"]
                    else:
                        act += ["w_p_top" if a > 0 else "w_p_bot", "w_pa_top" if a > 0 else "w_pa_bot", "w_pb_top" if b > 0 else "w_pb_bot"]
                    if b2 != 0:
                        act += ["VL", "w_s_top", "w_s_bot", "w_sc_top" if cc > 0 else "w_sc_bot", "w_sd_top" if d > 0 else "w_sd_bot"]
                    else:
                        act += ["w_s_top" if cc > 0 else "w_s_bot", "w_sc_top" if cc > 0 else "w_sc_bot", "w_sd_top" if d > 0 else "w_sd_bot"]
                    lv = {1: "+V", -1: "−V", 0: "0 (zero state)"}
                    c.mode(key, f"v₁ = {lv[b1].replace('V', 'V₁')}, v₂′ = {lv[b2].replace('V', 'V₂′')}", act, f"v_L = v₁ − v₂′ 가 인덕터 전류 기울기를 정한다 (1차 bridge {'전력 전달' if b1 else 'zero state 순환'}, 2차 bridge {'전력 전달' if b2 else 'zero state 순환'})", dim=dim)
    return c


def leg_mode_key(theta: float, mod: Modulation) -> str:
    a, b = leg_states(theta, mod.w1, mod.c1, mod.pattern1)
    cc, d = leg_states(theta, mod.w2, mod.c2, mod.pattern2)
    f = lambda x: "+" if x > 0 else "-"  # noqa: E731
    return f"{f(a)}{f(b)}|{f(cc)}{f(d)}"


def bands_for(mod: Modulation, fs: float, periods: int = 2, t_offset: float = 0.0) -> list[dict]:
    """State intervals (leg states) for plots and circuit linking, over ``periods`` periods."""
    om = TWO_PI * fs
    th = set()
    for w, c, p in ((mod.w1, mod.c1, mod.pattern1), (mod.w2, mod.c2, mod.pattern2)):
        for e in leg_edges(w, c, p):
            th.add(e[0])
    th = sorted(th | {0.0})
    out = []
    for k in range(periods):
        pts = [x + k * TWO_PI for x in th] + [TWO_PI * (k + 1)]
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a <= 1e-12:
                continue
            key = leg_mode_key(0.5 * (a + b), mod)
            b1, b2 = mod.levels(0.5 * (a + b))
            lab = f"v₁ {'+' if b1 > 0 else ('−' if b1 < 0 else '0')} / v₂′ {'+' if b2 > 0 else ('−' if b2 < 0 else '0')}"
            out.append({"x0": a / om + t_offset, "x1": b / om + t_offset, "mode": key, "label": lab})
    return out


def sample_pwl(p: PWL, periods: int = 2) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    T = p.t[-1] - p.t[0]
    for k in range(periods):
        tx, ty = p.plot_points()
        xs += [x + k * T for x in tx]
        ys += ty
    return xs, ys
