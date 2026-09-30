"""EX09 - EMI: noise source, path and victim together (textbook E09; extends FL02/FL04/EX07).

Physics (all synthetic lumped relative proxies, never an EMI compliance result):
  A  hand screens: i = C_par dv/dt (peak displacement screen), f_r = 1/(2 pi sqrt(LC)),
     2 pi f C V at the line frequency; trapezoid spectrum and envelope corners.
  C  exact switched-affine edge model: an ideal trapezoid source (gate speed = rise time)
     behind the power-loop inductance into the switch-node capacitance, optional RC
     snubber, and the CM path C_par -> chassis -> (Y capacitors || cable + a 50 ohm || 50 uH
     measurement proxy).  The node ringing, the CM current and the proxy voltage come out
     of the same state equations; an energy ledger closes over a period.
  A  DM path: switching-cell current split between the DC-link capacitor (ESR, ESL) and the
     line path through the measurement proxy.
Independent paths: closed-form trapezoid harmonics vs FFT of the simulated source,
impedance algebra (reference/emi.py) vs state-space frequency responses, Parseval
(time-domain RMS of the proxy voltage vs harmonic sum), energy ledger, zero-crossing ring
frequency vs closed form, numerical quadrature of the line-frequency current.
Optional 3-level NPC extension: two capacitor states and the neutral-point KCL.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from ..engine.periodic import shoot
from ..engine.switched import AffineMode, Guard, HybridSystem, propagator, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..model.units import fmt_si
from ..reference import emi as ref
from ._common import bands_from_traj, energy_ledger, ledger_check, sym_linear

TB_E09 = TextbookRef("expert-e09-emi-noise-source-경로-victim을-같이-본다", "E09. EMI: noise source, 경로, victim")
TB_05 = TextbookRef("gate-drivedpt보호-강점을-면접-증거로-만들기-fl02", "05. Gate drive·DPT·보호 [FL02]")
TB_E11 = TextbookRef("expert-e11-모델을-믿을-수-있는-범위-검증식별불확도", "E11. 검증·식별·불확도")

PROXY_NOTE = "relative conducted-noise proxy: 합성 50 Ω ∥ 50 µH 측정망의 고조파 peak 진폭(dBµV 환산). receiver detector·RBW·규격 LISN·limit 없음 → EMI compliance는 NOT_EVALUABLE"
BANDS = [(0.15e6, 1e6), (1e6, 10e6), (10e6, 30e6), (30e6, 100e6)]


def dbuv(v):
    v = np.maximum(np.asarray(v, dtype=float), 1e-12)
    return 20.0 * np.log10(v / 1e-6)


def lines_for_plot(f, amp, floor_rel: float = 1e-5):
    """Discrete harmonic lines for display: drop lines more than 100 dB below the largest one."""
    f = np.asarray(f)
    amp = np.asarray(amp)
    keep = amp >= floor_rel * float(np.max(amp))
    return f[keep].tolist(), dbuv(amp[keep]).tolist()


def peak_hold(f, amp, per_decade: int = 20):
    """Maximum harmonic amplitude in each 1/per_decade-decade bin (a display envelope, not a detector).

    Bins are at least 2.5 harmonic spacings wide so that every bin holds an odd harmonic.
    """
    f = np.asarray(f)
    amp = np.asarray(amp)
    df = float(f[1] - f[0]) if f.size > 1 else 1.0
    edges = [float(f[0]) * 0.999]
    step = 10 ** (1.0 / per_decade)
    while edges[-1] < f[-1]:
        edges.append(max(edges[-1] * step, edges[-1] + 2.5 * df))
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (f >= a) & (f < b)
        if sel.any():
            k = int(np.argmax(amp[sel]))
            xs.append(float(f[sel][k]))
            ys.append(float(dbuv(amp[sel][k])))
    return xs, ys


# ======================================================================================
# Hand screens: ramp through a path, LC ring
# ======================================================================================


class RampPath(HybridSystem):
    """Ideal voltage ramp (dv/dt) applied to C_par through a series R-L path. x = [v_s, i, v_c]."""

    state_names = ("v_s", "i", "v_c")

    def __init__(self, dvdt: float, V: float, C: float, R: float, L: float):
        self.k, self.V, self.C, self.R, self.L = dvdt, V, C, R, L
        self.t_r = V / dvdt
        self._key = f"ramp|{dvdt}|{V}|{C}|{R}|{L}"

    def mode(self, q):
        A = np.array([[0.0, 0.0, 0.0], [1.0 / self.L, -self.R / self.L, -1.0 / self.L], [0.0, 1.0 / self.C, 0.0]])
        b = np.array([self.k if q == "ramp" else 0.0, 0.0, 0.0])
        return AffineMode(f"{self._key}|{q}", A, b, label=q)

    def gate_schedule(self, t0, t1):
        return [(self.t_r, lambda q: "hold")] if t0 <= self.t_r < t1 else []

    def outputs(self, q):
        return {"v_s": np.array([1.0, 0, 0, 0]), "i": np.array([0, 1.0, 0, 0]), "v_c": np.array([0, 0, 1.0, 0])}


class SeriesRLC(HybridSystem):
    """Step V applied at t = 0 to a series R-L-C: x = [i, v_c]; ringing of v_c around V."""

    state_names = ("i", "v_c")

    def __init__(self, V: float, R: float, L: float, C: float):
        self.V, self.R, self.L, self.C = V, R, L, C

    def mode(self, q):
        A = np.array([[-self.R / self.L, -1.0 / self.L], [1.0 / self.C, 0.0]])
        return AffineMode(f"rlc|{self.V}|{self.R}|{self.L}|{self.C}", A, [self.V / self.L, 0.0])

    def outputs(self, q):
        return {"i": np.array([1.0, 0, 0]), "dv": np.array([0, 1.0, -self.V])}


def zero_crossing_frequency(t: np.ndarray, y: np.ndarray, skip: int = 1) -> tuple[float, int]:
    """Frequency from linearly interpolated zero crossings (same direction), skipping the first ones."""
    t = np.asarray(t)
    y = np.asarray(y)
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    tz = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    tz = tz[skip:]
    if tz.size < 2:
        return float("nan"), 0
    return float((tz.size - 1) / (tz[-1] - tz[0])), int(tz.size - 1)


# ======================================================================================
# Hard-switched cell with snubber and CM path
# ======================================================================================

CH_KO = {"ON": "하측 채널 ON (노드 0 V)", "F": "turn-off: 채널 전류 감소", "OFF": "채널 OFF", "R": "turn-on: 채널 전류 증가", "H": "turn-on: 노드 방전 (Miller)"}


class HardSwitchCell(HybridSystem):
    """Boost-type hard-switched cell (the DPT arrangement) with the CM path attached to the node.

    x = [i_ch, v_n, i_d, v_cs, v_cp, v_x, i_cab, i_lm].  The load current I_L flows into the
    switch node; the low-side channel is a gate-speed-limited current (ramp of t_fi at
    turn-off, ramp of t_ri to I_L + I_dis at turn-on, held until the node reaches 0 V, then
    the channel clamps the node); the high-side diode (ideal, no recovery) connects the node
    to V_bus through the power-loop inductance L_loop.  C_node is the total node capacitance
    (both devices), an RC snubber R_s-C_s is optional.  CM path: R_par-C_par to the chassis,
    chassis to the DC bus through 2 C_Y + stray, cable L_cab/2 and the measurement proxy
    (two R_m || L_m branches in parallel for common mode).  q = (channel phase, diode on).
    """

    state_names = ("i_ch", "v_n", "i_d", "v_cs", "v_cp", "v_x", "i_cab", "i_lm")

    def __init__(self, p: dict):
        self.p = dict(p)
        self.V = p["V_bus"]
        self.IL = p["I_L"]
        self.fs = p["f_sw"]
        self.T = 1.0 / self.fs
        self.D = p["D"]
        self.g = p.get("gate_scale", 1.0)
        self.tfi = p["t_fi"] * self.g
        self.tri = p["t_ri"] * self.g
        self.Idis = p["I_dis"] / self.g
        self.L = p["L_loop"]
        self.Rl = p["R_loop"]
        self.Cn = p["C_node"]
        self.snub = bool(p.get("snubber", False)) and p.get("C_s", 0.0) > 0
        self.Rs = p.get("R_s", 1.0)
        self.Cs = p.get("C_s", 1e-9)
        self.Cp = p["C_par"]
        self.Rp = p["R_par"]
        self.Cx = 2.0 * p["C_Y"] + p["C_stray"]
        self.Lc = 0.5 * p["L_cab"]
        self.Rm = 0.5 * p["R_m"]
        self.Lm = 0.5 * p["L_m"]
        self.t_on = self.D * self.T
        self.eps = 1e-6 * self.V  # diode forward-bias margin (round-off guard against on/off chatter)
        if self.tfi >= self.t_on or self.t_on + self.tri >= self.T:
            raise ValueError("current ramps do not fit in the switching period")
        self._key = "cell|" + "|".join(f"{k}={v}" for k, v in sorted(self.p.items()))
        self._modes: dict = {}

    # ---- continuous dynamics --------------------------------------------------------
    def _gcp(self) -> np.ndarray:
        g = np.zeros(8)
        g[1], g[4], g[5] = 1 / self.Rp, -1 / self.Rp, -1 / self.Rp
        return g

    def mode(self, q):
        m = self._modes.get(q)
        if m is not None:
            return m
        ch, dio = q
        A = np.zeros((8, 8))
        b = np.zeros(8)
        Cn = self.Cn
        gcp = self._gcp()
        b[0] = {"ON": 0.0, "F": -self.IL / self.tfi, "OFF": 0.0, "R": (self.IL + self.Idis) / self.tri, "H": 0.0}[ch]
        if ch != "ON":
            # node KCL: Cn dv_n/dt = I_L - i_ch - i_d - i_snub - i_cp
            A[1, 0] = -1 / Cn
            b[1] = self.IL / Cn
            if dio:
                A[1, 2] = -1 / Cn
            A[1, :] -= gcp / Cn
            if self.snub:
                A[1, 1] -= 1 / (self.Rs * Cn)
                A[1, 3] += 1 / (self.Rs * Cn)
        if dio:
            A[2, 1], A[2, 2] = 1 / self.L, -self.Rl / self.L
            b[2] = -self.V / self.L
        if self.snub:
            A[3, 1], A[3, 3] = 1 / (self.Rs * self.Cs), -1 / (self.Rs * self.Cs)
        A[4, :] += gcp / self.Cp
        A[5, :] += gcp / self.Cx
        A[5, 6] -= 1 / self.Cx
        A[6, 5], A[6, 6], A[6, 7] = 1 / self.Lc, -self.Rm / self.Lc, self.Rm / self.Lc
        A[7, 6], A[7, 7] = self.Rm / self.Lm, -self.Rm / self.Lm
        m = AffineMode(f"{self._key}|{ch}|{dio}", A, b, label=f"{ch}|{dio}")
        self._modes[q] = m
        return m

    def guards(self, q):
        ch, dio = q
        out = []
        if dio:

            def off(z):
                z = z.copy()
                z[2] = 0.0
                return z

            out.append(Guard("diode 전류 0 → 차단", np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0]), -1, lambda qq: (qq[0], 0), off))
        elif ch != "ON":
            out.append(Guard("노드가 V_bus 도달 → diode 도통", np.array([0, 1.0, 0, 0, 0, 0, 0, 0, -self.V - self.eps]), +1, lambda qq: (qq[0], 1)))
        if ch in ("R", "H") and not dio:

            def clamp(z, IL=self.IL):
                z = z.copy()
                z[1] = 0.0
                z[0] = IL  # the clamped channel now carries the load current (plus snubber/CM currents)
                return z

            out.append(Guard("노드가 0 V 도달 → 채널이 노드를 잡음", np.array([0, 1.0, 0, 0, 0, 0, 0, 0, 0]), -1, lambda qq: ("ON", 0), clamp))
        return out

    def gate_schedule(self, t0, t1):
        T = self.T
        ev = []
        for k in range(int(math.floor(t0 / T)) - 1, int(math.ceil(t1 / T)) + 1):
            for t, ch in ((k * T, "F"), (k * T + self.tfi, "OFF"), (k * T + self.t_on, "R"), (k * T + self.t_on + self.tri, "H")):
                if t0 - 1e-15 <= t < t1:
                    ev.append((t, lambda q, ch=ch: (ch, q[1]) if not (ch == "H" and q[0] == "ON") else q))
        return ev

    def after_event(self, q, z):
        ch, dio = q
        if not dio and ch not in ("ON",) and z[1] > self.V + self.eps:
            return (ch, 1)
        return q

    # ---- outputs, energy ------------------------------------------------------------
    def _row(self, **kw):
        r = np.zeros(9)
        for k, v in kw.items():
            r[self.state_names.index(k)] = v
        return r

    def outputs(self, q):
        ch, dio = q
        i_cp = np.zeros(9)
        i_cp[1], i_cp[4], i_cp[5] = 1 / self.Rp, -1 / self.Rp, -1 / self.Rp
        i_sn = np.zeros(9)
        if self.snub:
            i_sn[1], i_sn[3] = 1 / self.Rs, -1 / self.Rs
        v_m = np.zeros(9)
        v_m[6], v_m[7] = self.Rm, -self.Rm
        if ch == "ON":
            i_ch = self._row() + np.eye(9)[8] * self.IL - i_sn - i_cp  # channel carries what the clamped node needs
        else:
            i_ch = self._row(i_ch=1.0)
        return {
            "v_n": self._row(v_n=1.0),
            "i_ch": i_ch,
            "i_d": self._row(i_d=1.0),
            "i_cp": i_cp,
            "i_snub": i_sn,
            "v_m": v_m,
            "v_x": self._row(v_x=1.0),
        }

    def stored_energy(self):
        return np.diag([0.0, self.Cn, self.L, self.Cs if self.snub else 0.0, self.Cp, self.Cx, self.Lc, self.Lm, 0.0])

    def powers(self, q):
        ch, dio = q
        o = self.outputs(q)
        e_n = self._row(v_n=1.0)
        e_d = self._row(i_d=1.0)
        one = np.eye(9)[8]
        p_load = sym_linear(e_n, self.IL)  # load current source injects v_n * I_L
        p_bus = sym_linear(e_d, self.V)  # energy returned to the DC link through the diode path
        p_ch = 0.5 * (np.outer(e_n, o["i_ch"]) + np.outer(o["i_ch"], e_n))
        p_loop = self.Rl * np.outer(e_d, e_d)
        p_snub = self.Rs * np.outer(o["i_snub"], o["i_snub"]) if self.snub else np.zeros((9, 9))
        p_par = self.Rp * np.outer(o["i_cp"], o["i_cp"])
        p_meas = np.outer(o["v_m"], o["v_m"]) / self.Rm
        del one
        return {"p_load": p_load, "p_bus": p_bus, "p_ch": p_ch, "p_loop": p_loop, "p_snub": p_snub, "p_par": p_par, "p_meas": p_meas, "p_loss": p_ch + p_loop + p_snub + p_par + p_meas}

    def describe(self, q):
        return f"{q[0]}|{q[1]}"


def cell_steady(sys: HardSwitchCell, periods: int = 5):
    """Periodic steady state by repeating periods from the physical start state.

    The cell has no slow state: the node is clamped every cycle, the load current is a
    source and the slowest network time constant is L_m/R_m ~ 1 us << T, so the per-period
    state change drops geometrically.  Returns (last-period trajectory, per-period changes).
    """
    x = np.zeros(8)
    x[0] = sys.IL
    sc = np.array([sys.IL, sys.V, sys.IL, sys.V, sys.V, 1.0, 1e-3, 1e-3])
    changes = []
    q = ("ON", 0)
    tr = None
    for k in range(periods):
        tr = simulate(sys, q, x, k * sys.T, (k + 1) * sys.T)
        xn = tr.z_end[:8].copy()
        changes.append(float(np.max(np.abs(xn - x) / sc)))
        x, q = xn, tr.q_end
    return tr, changes


def uniform_samples(traj, names, t0: float, T: float, N: int) -> dict:
    """Exact samples on a uniform grid t0 + k T/N (k = 0..N-1) by propagating inside each segment."""
    dt = T / N
    grid = t0 + dt * np.arange(N)
    out = {n: np.empty(N) for n in names}
    k = 0
    for s, off, h in traj.window(t0, t0 + T):
        a = s.t0 + off
        b = a + h
        if k >= N:
            break
        # grid points inside [a, b)
        k_end = int(min(N, math.ceil((b - t0) / dt - 1e-9)))
        if k_end <= k:
            continue
        outs = traj.system.outputs(s.q)
        rows = np.array([outs[n] for n in names])
        z = propagator(s.mode, grid[k] - s.t0) @ s.z0
        step = propagator(s.mode, dt)
        for j in range(k, k_end):
            vals = rows @ z
            for i, n in enumerate(names):
                out[n][j] = vals[i]
            z = step @ z
        k = k_end
    return {"t": grid, **out}


def harmonics(y: np.ndarray) -> np.ndarray:
    """One-sided amplitudes A_n (n = 0..N/2) of one period sampled uniformly."""
    c = np.fft.rfft(y) / y.size
    A = 2.0 * np.abs(c)
    A[0] = abs(c[0])
    return A


def ss_freq_response(A: np.ndarray, b_in: np.ndarray, c_out: np.ndarray, f: float) -> complex:
    """C (jwI - A)^-1 B for a single input column and output row."""
    w = 2 * math.pi * f
    x = np.linalg.solve(1j * w * np.eye(A.shape[0]) - A, b_in)
    return complex(c_out @ x)


def h_dm_nodal(f: float, d: dict) -> complex:
    """DM transfer written as a 2-node admittance problem (independent of the current-divider form).

    Nodes: u (DC-link node), m (midpoint between the line inductance and the measurement
    branches, per-line symmetric).  Injection -1 A at u (the cell draws 1 A).
    """
    w = 2 * math.pi * f
    y_dc = 1.0 / (d["Resr"] + 1j * w * d["Lesl"] + 1.0 / (1j * w * d["Cdc"]))
    y_cab = 1.0 / (2j * w * d["Lcab"])
    y_m = (1.0 / d["Rm"] + 1.0 / (1j * w * d["Lm"])) / 2.0  # two R||L branches in series
    Y = np.array([[y_dc + y_cab, -y_cab], [-y_cab, y_cab + y_m]])
    u, m = np.linalg.solve(Y, np.array([1.0, 0.0]))  # 1 A injected (sign irrelevant for |H|)
    return complex(m / 2.0)  # voltage across one R||L branch = half of m


# ======================================================================================
# Experiment 1: hand screens and unit traps
# ======================================================================================


def _screen_circuit(v: dict) -> Circuit:
    c = Circuit("emi_ramp", 600, 250, title="변위전류 screen: dv/dt 전원 → 경로 R·L → C_par → chassis")
    vs = c.add("vsource", "Vs", 130, 130, 90, "dv/dt 전원", f"{v['dvdt'] / 1e3:g} kV/µs", lpos=(112, 126, "end"))
    rp = c.add("resistor", "Rp", 230, 50, 0, "R_path", fmt_si(v["R_path"], "Ω", 3), lpos=(230, 30, "middle"))
    lp = c.add("inductor", "Lp", 350, 50, 0, "L_path", fmt_si(v["L_path"], "H", 3), lpos=(350, 30, "middle"))
    cp = c.add("capacitor", "Cp", 460, 130, 90, "C_par", fmt_si(v["C_par"], "F", 3), lpos=(476, 126, "start"))
    c.add("ground", "g", 290, 210)
    c.wire("w1", vs["a"], (130, 50), rp["a"])
    c.wire("w2", rp["b"], lp["a"])
    c.wire("w3", lp["b"], (460, 50), cp["a"])
    c.wire("w4", cp["b"], (460, 210), (130, 210), vs["b"])
    c.text(520, 228, "chassis / 귀환", "small")
    c.probe("pi", "ramp_i1", 410, 38, "right", "i_CM")
    act = ["Vs", "Rp", "Lp", "Cp", "w1", "w2", "w3", "w4"]
    c.mode("ramp", "dv/dt 구간", act, "전원 전압이 일정한 기울기로 상승: C_par에 C·dv/dt 변위전류")
    c.mode("hold", "edge 이후", act, "전압이 멈추면 변위전류도 (경로 L·C 링잉을 남기고) 사라진다")
    return c


def run_hand_screens(v: dict) -> Result:
    res = Result("EX09", "hand_screens", "A (screen) + C (정확 시간영역 검산)")
    C, dvdt_us = v["C_par"], v["dvdt"]
    dvdt = dvdt_us * 1e6  # V/us -> V/s
    ipk = ref.displacement_peak(C, dvdt)
    tb1 = abs(C - 100e-12) < 1e-18 and abs(dvdt_us - 5e4) < 1e-6
    res.add_metric("i_pk", "peak 변위전류 screen i = C_par·dv/dt", ipk, "A", ref=5.0 if tb1 else None, ref_label="교재 5 A", tol=1e-12, basis="edge 동안의 peak screen — 지속 DC 전류·LISN receiver 읽음 아님")
    res.add_metric("dvdt_si", "dv/dt (SI)", dvdt, "V/s", basis=f"= {dvdt_us / 1e3:g} kV/µs = {dvdt * 1e-9:g} V/ns")
    L, Cr, Rr = v["L_r"], v["C_r"], v["R_r"]
    fr = ref.ring_frequency(L, Cr)
    tb2 = abs(L - 10e-9) < 1e-18 and abs(Cr - 1e-9) < 1e-18
    res.add_metric("f_r", "링잉 주파수 f_r = 1/(2π√(LC))", fr, "Hz", ref=50.329e6 if tb2 else None, ref_label="교재 50.329 MHz", tol=1e-5, basis="무손실 lumped L–C")
    res.add_metric("Z0", "특성 임피던스 √(L/C)", math.sqrt(L / Cr), "Ω", basis="링 전류 = 전압 변화 / Z0 규모")
    Cy, Vl, fl = v["C_y"], v["V_line"], v["f_line"]
    il = ref.line_capacitor_current(Cy, Vl, fl)
    tb3 = abs(Cy - 2.2e-9) < 1e-18 and abs(Vl - 230) < 1e-9 and abs(fl - 50) < 1e-9
    res.add_metric("i_line", "line 주파수 커패시터 전류 2πfCV", il, "A", ref=0.158965e-3 if tb3 else None, ref_label="교재 0.158965 mA", tol=1e-5, basis="RMS, 이상 단일 경로")
    res.add_metric("leak_status", "누설·접촉전류 규격 판정", "NOT_EVALUABLE", "", basis="측정망(인체 모델)·상 결선·스위칭 주파수 CM 미포함 — 단일 경로 계산")
    # --- time-domain check 1: ramp through a low-impedance path, and through a proxy-like path
    V = v["V_step"]
    t_r = V / dvdt
    tr_end = 2.5 * t_r
    curves = {}
    for tag, R, Lp in (("1", v["R_path"], v["L_path"]), ("2", v["R_path2"], v["L_path2"])):
        sysr = RampPath(dvdt, V, C, R, Lp)
        tr = simulate(sysr, "ramp", [0.0, 0.0, 0.0], 0.0, tr_end)
        smp = tr.sample(["i", "v_s"], per_segment=400)
        curves[tag] = (tr, smp)
        res.add_series(f"ramp_i{tag}", f"i_CM (R {fmt_si(R, 'Ω', 3)}, L {fmt_si(Lp, 'H', 3)})", "A", smp["t"], smp["i"])
    tr1 = curves["1"][0]
    i_end = float(tr1.state_at(t_r)[0][1])
    tau = v["R_path"] * C
    res.add_check(check_close("변위전류: 정확 시간영역 (낮은 경로 임피던스) vs C·dv/dt", i_end, ipk, 0.01, "R–L–C 경로에 dv/dt 램프를 건 행렬지수 해의 램프 끝 전류 vs 곱셈 screen", True, "A",
                              detail=f"램프 끝 전류 {i_end:.4f} A / screen {ipk:.4f} A (경로 RC = {tau * 1e9:.3g} ns ≪ t_r = {t_r * 1e9:.3g} ns일 때만 일치)"))
    lo2, hi2 = curves["2"][0].extrema(0, tr_end, "i", per_segment=400)
    i_end2 = float(curves["2"][0].state_at(t_r)[0][1])
    res.add_metric("i_end_path2", f"다른 경로({fmt_si(v['R_path2'], 'Ω', 3)}·{fmt_si(v['L_path2'], 'H', 3)})의 램프 끝 전류", i_end2, "A", basis="같은 dv/dt라도 경로 임피던스가 전류 파형을 바꾼다", note=f"screen 대비 {i_end2 / ipk:.0%}; 이후 링잉 peak {hi2:.3g} A")
    res.add_metric("t_r", "edge 시간 V_step/(dv/dt)", t_r, "s", basis=f"{V:g} V 스텝")
    # --- time-domain check 2: ring frequency from zero crossings of an RLC step response
    sysl = SeriesRLC(1.0, Rr, L, Cr)
    fd = ref.damped_ring_frequency(L, Cr, Rr)
    n_cyc = 12
    trl = simulate(sysl, None, [0.0, 0.0], 0.0, n_cyc / fd)
    sl = trl.sample(["dv"], per_segment=20000)
    f_meas, ncross = zero_crossing_frequency(np.asarray(sl["t"]), np.asarray(sl["dv"]), skip=1)
    res.add_metric("f_meas", "시간영역 링잉 주파수 (영점 교차)", f_meas, "Hz", ref=fd, ref_label="감쇠 닫힌 식 f_0√(1−ζ²)", tol=1e-4, basis=f"R = {fmt_si(Rr, 'Ω', 3)}, {ncross}주기")
    res.add_check(check_close("링잉 주파수: 정확 RLC 해의 영점 교차 vs 닫힌 식", f_meas, fd, 1e-4, "행렬지수 스텝 응답을 조밀 샘플해 선형 보간 영점 교차 → 주파수 vs f_0√(1−ζ²)", True, "Hz"))
    xs = [x for x in sl["t"]][::20]
    res.add_series("ring", "v_C − V (RLC 스텝 응답)", "V/V", xs, sl["dv"][::20])
    # --- independent check 3: line-frequency current by quadrature of i = C dv/dt
    w = 2 * math.pi * fl
    Tl = 1 / fl
    i2, _ = quad(lambda t: (Cy * math.sqrt(2) * Vl * w * math.cos(w * t)) ** 2, 0, Tl, limit=200)
    irms_q = math.sqrt(i2 / Tl)
    res.add_check(check_close("line 전류: i = C dv/dt 수치 구적 RMS vs 2πfCV", irms_q, il, 1e-9, "√2·230 V 정현파의 C dv/dt를 한 주기 수치 적분 (quad) vs phasor 식", True, "A"))
    # --- unit traps
    rows = [
        ["C_par·dv/dt (올바른 환산)", f"{C * 1e12:g} pF × {dvdt_us / 1e3:g} kV/µs = {C * 1e12:g}e-12 F × {dvdt:.3g} V/s", ipk, "×1 (기준)"],
        ["kV/µs를 V/µs로 읽음", f"{C * 1e12:g} pF × {dvdt_us / 1e3:g} V/µs", C * dvdt / 1e3, "×1/1000 (−60 dB)"],
        ["kV/µs를 kV/ms로 읽음", f"{C * 1e12:g} pF × {dvdt_us / 1e3:g} kV/ms", C * dvdt / 1e3, "×1/1000"],
        ["pF를 nF로 읽음", f"{C * 1e12:g} nF × {dvdt_us / 1e3:g} kV/µs", C * 1e3 * dvdt, "×1000 (+60 dB)"],
        ["V/ns 표기 확인", f"{dvdt_us / 1e3:g} kV/µs = {dvdt * 1e-9:g} V/ns", ipk, "같은 값"],
        ["교재 5장: C_par 1 nF × 30 kV/µs", "1e-9 F × 3e10 V/s", 1e-9 * 3e10, "순간 변위전류 (연속 30 A 아님)"],
        ["교재 5장: L_loop 15 nH × 5 kA/µs", "15e-9 H × 5e9 A/s", 15e-9 * 5e9, "V (전압 스파이크 규모)"],
        ["교재 E03: 공통 source 2 nH × 2 kA/µs", "2e-9 H × 2e9 A/s", 2e-9 * 2e9, "V (gate 참조 결합)"],
    ]
    res.tables.append(Table("t_units", "단위 함정: 같은 식, 다른 단위 해석", ["경우", "계산", "결과 [A 또는 V]", "올바른 값 대비"], rows, note="dv/dt 단위를 틀리면 세 자릿수 이상 오류가 난다(교재 E09). 5장·E03 값은 같은 screen 문법의 다른 예다."))
    frows = [[f"{Vl:g} V, {fl:g} Hz", Vl, fl, il * 1e3], [f"{Vl:g} V, 60 Hz", Vl, 60.0, ref.line_capacitor_current(Cy, Vl, 60.0) * 1e3], ["CM 잡음 10 V, 150 kHz", 10.0, 150e3, ref.line_capacitor_current(Cy, 10.0, 150e3) * 1e3], ["CM 잡음 1 V, 30 MHz", 1.0, 30e6, ref.line_capacitor_current(Cy, 1.0, 30e6) * 1e3]]
    res.tables.append(Table("t_line", f"같은 {Cy * 1e9:g} nF의 전류는 주파수에 비례한다 (2πfCV)", ["조건", "V [Vrms]", "f [Hz]", "전류 [mA rms]"], frows, note="50 Hz 값은 단일 이상 경로 계산이며 누설·접촉전류 규격 시험의 합격 판정이 아니다. 스위칭 주파수 CM 전류·측정망·상 결선은 포함하지 않았다."))
    res.add_plot("p_ramp", "dv/dt 램프가 C_par에 흘리는 변위전류", ["ramp_i1", "ramp_i2"], y_label="i_CM", y_unit="A", level="C", hlines=[{"y": ipk, "label": f"C·dv/dt = {ipk:.3g} A"}], vlines=[{"x": t_r, "label": "램프 끝"}], group="ramp",
                 bands=bands_from_traj(tr1, 0, tr_end, {"ramp": "dv/dt", "hold": "hold"}),
                 proved="경로 임피던스가 작으면 램프 동안 전류가 C·dv/dt에 붙고 램프가 끝나면 사라진다: 5 A는 edge 동안의 peak screen이다. 경로가 50 Ω·100 nH급이면 같은 dv/dt에서도 peak가 달라진다.",
                 not_yet="실제 CM 전류 spectrum은 반복률·rise time·귀환 경로·케이블·측정 대역폭이 정한다(실험 2). receiver 읽음이 아니다.")
    res.add_plot("p_ring", "L–C 링잉: 영점 교차로 잰 주파수", ["ring"], y_label="(v_C − V)/V", y_unit="", level="C",
                 proved=f"정확 RLC 스텝 응답의 영점 교차 주파수가 f_0√(1−ζ²)와 일치한다 (L = {fmt_si(L, 'H', 3)}, C = {fmt_si(Cr, 'F', 3)}).",
                 not_yet="링잉 주파수 한 점으로는 L과 C를 따로 식별할 수 없다(EX11). 실제 파형이 다르면 C의 전압 의존성·분포 layout·probe loading을 조사한다.")
    res.circuit = {"diagram": _screen_circuit(v).to_json(), "intervals": bands_from_traj(tr1, 0, tr_end, {"ramp": "dv/dt", "hold": "hold"}), "plot_group": "ramp"}
    res.verdict("SCREEN_ONLY", "C·dv/dt = 5 A는 peak 변위전류 screen, 2πfCV는 단일 이상 경로 계산 — EMI·누설전류 합격 판정이 아니다")
    res.assumptions += ["이상 램프 전원, lumped C_par·L·R (분포 layout 없음)", "링잉 검산은 직렬 RLC 스텝 응답", "line 전류는 순수 정현파 1개 경로"]
    res.not_valid_for += ["EMI 규격 판정 (NOT_EVALUABLE)", "누설·접촉전류 규격 판정", "실제 PCB·케이블의 분포 정수 효과"]
    res.interpretation = (
        f"{C * 1e12:g} pF에 {dvdt_us / 1e3:g} kV/µs(= {dvdt * 1e-9:g} V/ns)가 걸리면 edge 동안 {ipk:.3g} A가 흐른다. 이것은 램프가 지속되는 {t_r * 1e9:.3g} ns 동안의 peak 규모일 뿐, 평균 전류나 receiver 읽음이 아니다. "
        f"10 nH와 1 nF는 {fr / 1e6:.3f} MHz에서 링잉하고, 2.2 nF는 230 V·50 Hz에서 {il * 1e3:.4f} mA를 흘린다 — 둘 다 조건이 붙은 단일 경로 계산이다."
    )
    return res


# ======================================================================================
# Shared cell analysis: waveforms -> spectra -> CM/DM proxies, losses
# ======================================================================================

N_FFT = 2**16


def _cell_params(extra: list[Param] | None = None) -> list[Param]:
    ps = [
        Param("V_bus", "DC bus 전압 V_bus", "V", 800.0, "V", vmin=10, vmax=1500, source="TEXTBOOK", source_note="800 V (E02·E09 맥락)", group="source"),
        Param("I_L", "부하(인덕터) 전류 I_L", "A", 50.0, "A", vmin=0.5, vmax=1000, source="ASSUMED", source_note="I_L/C_node ≈ 50 kV/µs가 되도록 고른 합성값", group="source"),
        Param("f_sw", "스위칭 주파수 f_sw", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="ASSUMED", group="source"),
        Param("D", "노드 high 비율 D", "", 0.5, "", vmin=0.05, vmax=0.95, source="ASSUMED", group="source"),
        Param("t_fi", "turn-off 채널 전류 하강 시간 t_fi (gate 속도)", "s", 10e-9, "ns", vmin=0.5e-9, vmax=2e-6, source="ASSUMED", group="gate"),
        Param("t_ri", "turn-on 채널 전류 상승 시간 t_ri (gate 속도)", "s", 10e-9, "ns", vmin=0.5e-9, vmax=2e-6, source="ASSUMED", group="gate"),
        Param("I_dis", "turn-on 노드 방전 전류 (Miller 구간 여분 전류)", "A", 50.0, "A", vmin=0.1, vmax=2000, source="ASSUMED", source_note="turn-on dv/dt = I_dis/C_node", group="gate"),
        Param("gate_scale", "gate 속도 배율 (t_fi·t_ri ×, I_dis ÷)", "", 1.0, "", vmin=0.1, vmax=20, source="ASSUMED", source_note="R_g를 키우는 효과의 합성 표현", group="gate"),
        Param("L_loop", "power loop 인덕턴스 L_loop", "H", 10e-9, "nH", vmin=0.1e-9, vmax=1e-6, source="TEXTBOOK", source_note="교재 E09 10 nH", group="loop"),
        Param("R_loop", "loop 저항", "Ω", 0.1, "Ω", vmin=0.0, vmax=10, source="ASSUMED", group="loop"),
        Param("C_node", "스위치 노드 용량 C_node (두 소자 합)", "F", 1e-9, "nF", vmin=1e-12, vmax=1e-6, source="TEXTBOOK", source_note="교재 E09 C_eq 1 nF (선형 합성)", group="loop"),
        Param("snubber", "RC snubber 사용", "", False, kind="bool", source="ASSUMED", group="snubber"),
        Param("R_s", "snubber R_s", "Ω", 3.16, "Ω", vmin=0.01, vmax=1e3, source="ASSUMED", source_note="≈ √(L_loop/C_node)", group="snubber"),
        Param("C_s", "snubber C_s", "F", 0.47e-9, "nF", vmin=1e-12, vmax=1e-6, source="ASSUMED", group="snubber"),
        Param("C_par", "노드→chassis 기생 용량 C_par (heatsink)", "F", 100e-12, "pF", vmin=1e-13, vmax=1e-8, source="TEXTBOOK", source_note="교재 E09 100 pF", group="CM path"),
        Param("R_par", "C_par 경로 직렬 저항", "Ω", 1.0, "Ω", vmin=0.01, vmax=100, source="ASSUMED", group="CM path"),
        Param("C_Y", "Y 커패시터 (선마다 chassis로)", "F", 0.0, "nF", vmin=0.0, vmax=1e-6, source="ASSUMED", source_note="0 = 없음", group="CM path"),
        Param("C_stray", "chassis–bus 기생 용량", "F", 10e-12, "pF", vmin=1e-13, vmax=1e-8, source="ASSUMED", group="CM path"),
        Param("L_cab", "케이블 인덕턴스 (선당)", "H", 1e-6, "µH", vmin=1e-9, vmax=1e-3, source="ASSUMED", group="측정망 proxy"),
        Param("R_m", "측정 port 저항 (선당)", "Ω", 50.0, "Ω", vmin=1, vmax=1e4, source="ASSUMED", source_note="50 Ω ∥ 50 µH 형태의 합성 proxy (CISPR LISN 아님)", group="측정망 proxy"),
        Param("L_m", "측정망 인덕터 (선당)", "H", 50e-6, "µH", vmin=1e-7, vmax=1e-2, source="ASSUMED", group="측정망 proxy"),
        Param("C_dc", "DC-link C (DM 경로)", "F", 10e-6, "µF", vmin=1e-8, vmax=1e-2, source="ASSUMED", group="DM path"),
        Param("ESR", "DC-link ESR", "Ω", 5e-3, "mΩ", vmin=0.0, vmax=1.0, source="ASSUMED", group="DM path"),
        Param("ESL", "DC-link ESL", "H", 10e-9, "nH", vmin=0.0, vmax=1e-6, source="ASSUMED", group="DM path"),
    ]
    return ps + (extra or [])


def _cell_p(v: dict) -> dict:
    keys = ("V_bus", "I_L", "f_sw", "D", "t_fi", "t_ri", "I_dis", "gate_scale", "L_loop", "R_loop", "C_node", "snubber", "R_s", "C_s", "C_par", "R_par", "C_Y", "C_stray", "L_cab", "R_m", "L_m")
    return {k: v[k] for k in keys}


def analyze_cell(p: dict, v: dict, n_fft: int = N_FFT) -> dict:
    sysc = HardSwitchCell(p)
    tr, changes = cell_steady(sysc)
    T = sysc.T
    ta = tr.segments[0].t0
    smp = uniform_samples(tr, ["v_n", "i_d", "i_cp", "v_m", "i_ch"], ta, T, n_fft)
    A_v = harmonics(smp["v_n"])
    A_i = harmonics(smp["i_d"])
    A_m = harmonics(smp["v_m"])
    f = sysc.fs * np.arange(A_v.size)
    Cy_ref = sysc.Cx / 2.0
    Hc = np.array([abs(ref.h_cm(fk, p["C_par"], p["R_par"], p["L_cab"], p["R_m"], p["L_m"], Cy_ref)) if fk > 0 else 0.0 for fk in f])
    Hd = np.array([abs(ref.h_dm(fk, v["C_dc"], v["ESR"], v["ESL"], p["L_cab"], p["R_m"], p["L_m"])) if fk > 0 else 0.0 for fk in f])
    cm = A_v * Hc
    dm = A_i * Hd
    dt = T / n_fft
    dv = np.diff(smp["v_n"]) / dt
    w = (smp["t"] - ta > 0) & (smp["t"] - ta < 0.4 * T)
    v_final = p["V_bus"] + p["R_loop"] * p["I_L"]
    led = energy_ledger(tr, sysc, ta, ta + T, ["p_load"], ["p_bus"], ["p_loss"], rated_power=p["V_bus"] * p["I_L"] * p["D"])
    out = {
        "sys": sysc,
        "tr": tr,
        "ta": ta,
        "changes": changes,
        "smp": smp,
        "f": f,
        "A_v": A_v,
        "A_i": A_i,
        "A_m": A_m,
        "Hc": Hc,
        "Hd": Hd,
        "cm": cm,
        "dm": dm,
        "vmax": float(np.max(smp["v_n"])),
        "vmin": float(np.min(smp["v_n"])),
        "dvdt_up": float(np.max(dv)),
        "dvdt_dn": float(-np.min(dv)),
        "icp_pk": float(np.max(np.abs(smp["i_cp"]))),
        "P_sw": tr.energy(ta, ta + T, "p_ch") / T,
        "P_snub": tr.energy(ta, ta + T, "p_snub") / T,
        "P_loop": tr.energy(ta, ta + T, "p_loop") / T,
        "vm_rms": tr.rms(ta, ta + T, "v_m"),
        "ledger": led,
        "v_final": v_final,
    }
    fr, nz = zero_crossing_frequency(smp["t"][w], smp["v_n"][w] - v_final, skip=1)
    out["f_ring"], out["n_ring"] = fr, nz
    for tag, spec in (("cm", cm), ("dm", dm)):
        for a, b in BANDS:
            sel = (f >= a) & (f <= b)
            out[f"{tag}_{a / 1e6:g}_{b / 1e6:g}"] = float(dbuv(spec[sel]).max()) if sel.any() else float("nan")
    return out


def band_metrics(res: Result, a: dict, tag: str, label: str):
    for lo, hi in BANDS:
        k = f"{tag}_{lo / 1e6:g}_{hi / 1e6:g}"
        res.add_metric(k, f"{label} 최대 {lo / 1e6:g}–{hi / 1e6:g} MHz", a[k], "dBµV", basis="relative proxy (고조파 peak 진폭)")


def cell_circuit(v: dict) -> Circuit:
    c = Circuit("emi_cell", 965, 360, title="Source(hard-switched cell) → Path(C_par·chassis·Y·케이블) → Victim(측정망 proxy)")
    vb = c.add("vsource", "Vb", 60, 175, 90, "V_bus", f"{v['V_bus']:g} V", lpos=(46, 171, "end"))
    cdc = c.add("capacitor", "Cdc", 140, 175, 90, "C_dc", "(DM 경로)", lpos=(156, 171, "start"))
    il = c.add("isource", "IL", 220, 110, 90, "I_L", f"{v['I_L']:g} A", lpos=(234, 106, "start"))
    ll = c.add("inductor", "Lloop", 275, 50, 0, "L_loop", fmt_si(v["L_loop"], "H", 3), lpos=(275, 30, "middle"))
    dh = c.add("diode", "DH", 330, 110, 270, "D_H", lpos=(344, 116, "start"))
    ql = c.add("nmos", "QL", 330, 235, 90, "Q_L", lpos=(304, 231, "end"))
    cn = c.add("capacitor", "Cn", 400, 235, 90, "C_node", fmt_si(v["C_node"], "F", 3), lpos=(414, 222, "start"))
    rs = c.add("resistor", "Rs", 490, 205, 90, "R_s", "snubber" if v.get("snubber") else "(미사용)", lpos=(504, 201, "start"))
    cs = c.add("capacitor", "Cs", 490, 265, 90, "C_s", "", lpos=(504, 261, "start"))
    cp = c.add("capacitor", "Cp", 590, 170, 0, f"C_par {fmt_si(v['C_par'], 'F', 3)}", "", lpos=(590, 146, "middle"))
    rp = c.add("resistor", "Rp", 670, 170, 0, "R_par", "", lpos=(670, 150, "middle"))
    cy = c.add("capacitor", "Cy", 740, 235, 90, "2C_Y+C_str", "", lpos=(726, 231, "end"))
    lc = c.add("inductor", "Lcab", 800, 170, 0, "L_cab/2", "", lpos=(800, 150, "middle"))
    rm = c.add("resistor", "Rm", 850, 235, 90, "R_m/2", "", lpos=(836, 231, "end"))
    lm = c.add("inductor", "Lm", 900, 235, 90, "L_m/2", "", lpos=(914, 231, "start"))
    c.wire("w_top", vb["a"], (60, 50), (140, 50), (220, 50), ll["a"])
    c.wire("w_cdc_t", (140, 50), cdc["a"])
    c.wire("w_il_t", (220, 50), il["a"])
    c.wire("w_il_n", il["b"], (220, 170), (330, 170))
    c.wire("w_loop", ll["b"], (330, 50), dh["b"])
    c.wire("w_dh_n", dh["a"], (330, 170))
    c.wire("w_q_n", (330, 170), ql["a"])
    c.wire("w_cn", (330, 170), (400, 170), cn["a"])
    c.wire("w_rs", (400, 170), (490, 170), rs["a"])
    c.wire("w_rscs", rs["b"], cs["a"])
    c.wire("w_cp", (490, 170), cp["a"])
    c.wire("w_cprp", cp["b"], rp["a"])
    c.wire("w_x", rp["b"], (740, 170), lc["a"])
    c.wire("w_cy", (740, 170), cy["a"])
    c.wire("w_m", lc["b"], (850, 170), (900, 170))
    c.wire("w_rm", (850, 170), rm["a"])
    c.wire("w_lm", (900, 170), lm["a"])
    c.wire("w_bot", vb["b"], (60, 320), (900, 320))
    c.wire("w_cdc_b", cdc["b"], (140, 320))
    c.wire("w_q_b", ql["b"], (330, 320))
    c.wire("w_cn_b", cn["b"], (400, 320))
    c.wire("w_cs_b", cs["b"], (490, 320))
    c.wire("w_cy_b", cy["b"], (740, 320))
    c.wire("w_rm_b", rm["b"], (850, 320))
    c.wire("w_lm_b", lm["b"], (900, 320))
    c.dot((140, 50), (220, 50), (330, 170), (400, 170), (490, 170), (740, 170), (850, 170), (140, 320), (330, 320), (400, 320), (490, 320), (740, 320), (850, 320))
    c.text(352, 188, "v_n", "node")
    c.text(772, 196, "chassis", "node")
    c.text(876, 158, "v_m", "node")
    c.text(480, 346, "DC bus 귀환 (HF에서 두 선은 C_dc로 묶임 → CM은 두 측정 branch 병렬: R_m/2, L_m/2)", "small")
    c.probe("pId", "i_d", 330, 70, "up", "i_d")
    c.probe("pIch", "i_ch", 330, 290, "down", "i_ch")
    c.probe("pIcp", "i_cp", 712, 158, "right", "i_CM")
    cell_on = ["Vb", "QL", "IL", "w_top", "w_il_t", "w_il_n", "w_q_n", "w_q_b", "w_bot"]
    cm = ["Cp", "Rp", "Cy", "Lcab", "Rm", "Lm", "w_cp", "w_cprp", "w_x", "w_cy", "w_m", "w_rm", "w_lm", "w_cy_b", "w_rm_b", "w_lm_b", "w_cn", "w_rs"]
    snub = ["Rs", "Cs", "w_rscs", "w_cs_b"] if v.get("snubber") else []
    texts = {
        "ON": ("하측 채널이 노드를 0 V로 잡고 부하전류를 흘림", ["DH", "Lloop"]),
        "F": ("turn-off: 채널 전류 감소, 나머지 부하전류가 C_node를 충전 (dv/dt)", ["DH"]),
        "OFF": ("채널 OFF: 부하전류가 노드를 충전", []),
        "R": ("turn-on: 채널 전류 증가, diode 전류 감소 (L_loop di/dt dip)", []),
        "H": ("turn-on: 채널이 부하전류보다 큰 전류로 노드를 방전 (dv/dt)", ["DH"]),
    }
    for ch, (txt, dim) in texts.items():
        for dio in (0, 1):
            act = cell_on + (["Cn"] if ch != "ON" else []) + cm + snub
            if dio:
                act += ["DH", "Lloop", "w_loop", "w_dh_n"]
                t2 = " · D_H 도통: L_loop–C_node 링잉, 전류가 DC link로 돌아감"
            else:
                t2 = " · D_H 차단"
            c.mode(f"{ch}|{dio}", f"{CH_KO[ch]}" + (" + D_H 도통" if dio else ""), act, txt + t2, dim=[d for d in dim if not (dio and d in ("DH", "Lloop"))] + (["QL"] if ch == "OFF" else []))
    return c


CELL_BANDS = {f"{ch}|{d}": {"ON": "Q ON", "F": "off: i↓", "OFF": "OFF", "R": "on: i↑", "H": "on: v↓"}[ch] + (" D" if d else "") for ch in CH_KO for d in (0, 1)}


# ======================================================================================
# Experiment 2: source -> path -> victim with a relative spectrum proxy
# ======================================================================================


def run_spv(v: dict) -> Result:
    res = Result("EX09", "source_path_victim", "C (cell·CM 경로 정확 시간영역) + A (DM 경로·spectrum proxy)")
    p = _cell_p(v)
    try:
        a = analyze_cell(p, v)
    except ValueError as exc:
        res.verdict("OUT_OF_VALIDITY", str(exc))
        return res
    sysc, tr, ta = a["sys"], a["tr"], a["ta"]
    T = sysc.T
    f, A_v, A_i = a["f"], a["A_v"], a["A_i"]
    V = p["V_bus"]
    # --- source metrics
    res.add_metric("dvdt_up", "turn-off 노드 dv/dt 최대", a["dvdt_up"], "V/s", basis=f"= {a['dvdt_up'] * 1e-9:.3g} V/ns (부하전류/C_node 제한 {p['I_L'] / p['C_node'] * 1e-9:.3g} V/ns)")
    res.add_metric("dvdt_dn", "turn-on 노드 dv/dt 최대", a["dvdt_dn"], "V/s", basis=f"= {a['dvdt_dn'] * 1e-9:.3g} V/ns (gate 제한 I_dis/C_node)")
    res.add_metric("v_os", "turn-off overshoot (노드 최대 − V_bus)", a["vmax"] - V, "V", basis=f"screen I_L·√(L/C) = {p['I_L'] * math.sqrt(p['L_loop'] / p['C_node']):.4g} V (감쇠 없음)")
    fr0 = ref.ring_frequency(p["L_loop"], p["C_node"])
    res.add_metric("f_ring", "노드 링잉 주파수 (시간영역 영점 교차)", a["f_ring"], "Hz", basis=f"L_loop·C_node 닫힌 식 {fr0 / 1e6:.3f} MHz; CM 경로·감쇠가 조금 옮긴다")
    res.add_metric("icp_pk", "CM 전류 peak (C_par 경로, 시간영역)", a["icp_pk"], "A", basis=f"screen C_par·dv/dt = {p['C_par'] * a['dvdt_up']:.3g} A; 링잉이 더한다")
    res.add_metric("vm_rms", "측정 proxy 전압 RMS (시간영역)", a["vm_rms"], "V", basis="한 50 Ω port, 한 주기")
    res.add_metric("P_sw", "스위칭 손실 ∫v·i_ch (같은 모델)", a["P_sw"], "W", basis="합성 채널 전류 모델의 overlap + 용량 방전; 소자 데이터 아님")
    band_metrics(res, a, "cm", "CM proxy")
    band_metrics(res, a, "dm", "DM proxy")
    res.add_metric("emi_status", "EMI compliance", "NOT_EVALUABLE", "", basis="receiver·detector·규격 LISN·limit 없음 — relative proxy만")
    # --- checks
    res.add_check(ledger_check(a["ledger"], what="cell+CM 경로: "))
    ch = a["changes"]
    res.add_check(Check("주기별 정상상태 기준 (정규화 상태 변화 < 1e-9)", "PASS" if ch[-1] < 1e-9 else "FAIL", ch[-1], "rel", 1e-9, path="물리 초기상태에서 주기 반복 (느린 상태 없음: 노드는 매 주기 clamp)", detail="주기별 변화: " + ", ".join(f"{c:.1e}" for c in ch)))
    # FFT normalisation vs the closed-form trapezoid (analytically sampled, independent of the engine)
    tr_eq = V / a["dvdt_up"]
    tt = np.arange(N_FFT) * T / N_FFT
    tf_start = p["D"] * T
    y = np.clip(np.minimum(tt / tr_eq, (tf_start + tr_eq - tt) / tr_eq), 0.0, 1.0) * V  # half-height width D*T
    A_y = harmonics(y)
    nn = np.arange(1, 400)
    A_cf = ref.trapezoid_harmonics(V, sysc.fs, p["D"], tr_eq, nn)
    sel = A_cf > 1e-3 * A_cf.max()
    e_fft = float(np.max(np.abs(A_y[1:400][sel] - A_cf[sel]) / A_cf[sel]))
    res.add_check(Check("FFT 정규화: 해석 사다리꼴 샘플 FFT vs 닫힌 식 2VD|sinc(nD)||sinc(n f t_r)|", "PASS" if e_fft < 1e-3 else "FAIL", e_fft, "rel", 1e-3, path="numpy FFT(균일 샘플) vs 교과서 Fourier 계수", independent=True, detail=f"n = 1..399, 유효 고조파 {int(sel.sum())}개"))
    # CM transfer: state-space from the cell matrices vs impedance algebra
    m = sysc.mode(("OFF", 1))
    Acm = m.A[4:8, 4:8]
    bcm = m.A[4:8, 1]
    cvm = sysc.outputs(("OFF", 1))["v_m"][4:8]
    errs = []
    for fk in np.geomspace(1e5, 3e8, 30):
        hs = ss_freq_response(Acm, bcm, cvm, fk)
        hr = ref.h_cm(fk, p["C_par"], p["R_par"], p["L_cab"], p["R_m"], p["L_m"], sysc.Cx / 2)
        errs.append(abs(hs - hr) / abs(hr))
    res.add_check(Check("CM 경로 전달함수: 상태공간 (셀 행렬) vs 임피던스 대수", "PASS" if max(errs) < 1e-9 else "FAIL", max(errs), "rel", 1e-9, path="C(jωI−A)⁻¹B (시뮬레이션이 쓰는 행렬) vs Z 직병렬 대수 (reference/emi.py)", independent=True, detail="100 kHz–300 MHz 30점"))
    # Parseval: time-domain RMS of the proxy voltage vs harmonic sum of source x path
    rms_f = math.sqrt(float(np.sum((A_v[1:] * a["Hc"][1:]) ** 2)) / 2)
    res.add_check(check_close("Parseval: proxy 전압 RMS (시간영역) vs Σ|V_n H_cm(f_n)|²/2", a["vm_rms"], rms_f, 2e-3, "정확 모멘트 적분 RMS vs 노드 전압 FFT × 임피던스 대수 전달함수의 고조파 합", True, "V"))
    # DM transfer: nodal admittance vs current divider
    d = {"Cdc": v["C_dc"], "Resr": v["ESR"], "Lesl": v["ESL"], "Lcab": p["L_cab"], "Rm": p["R_m"], "Lm": p["L_m"]}
    ed = max(abs(abs(h_dm_nodal(fk, d)) - abs(ref.h_dm(fk, v["C_dc"], v["ESR"], v["ESL"], p["L_cab"], p["R_m"], p["L_m"]))) / abs(ref.h_dm(fk, v["C_dc"], v["ESR"], v["ESL"], p["L_cab"], p["R_m"], p["L_m"])) for fk in np.geomspace(1e5, 3e8, 30))
    res.add_check(Check("DM 경로 전달함수: 2-node admittance 해 vs 전류 분배식", "PASS" if ed < 1e-9 else "FAIL", ed, "rel", 1e-9, path="Y 행렬 선형해 vs Z 분배 대수", independent=True, detail="100 kHz–300 MHz 30점"))
    # --- series and plots
    nmax = int(min(A_v.size - 1, round(300e6 / sysc.fs)))
    fx = f[1 : nmax + 1]
    xs_, ys_ = lines_for_plot(fx, A_v[1 : nmax + 1])
    res.add_series("src_v", "스위치 노드 전압 고조파 (시간영역 FFT)", "dBµV", xs_, ys_, style="points")
    res.add_series("src_env", f"사다리꼴 envelope (t_r,eq = {tr_eq * 1e9:.3g} ns)", "dBµV", fx.tolist(), dbuv(ref.trapezoid_envelope(V, sysc.fs, p["D"], tr_eq, fx)).tolist(), dash=True)
    fp = np.geomspace(1e5, 3e8, 240)
    Hcp = [abs(ref.h_cm(fk, p["C_par"], p["R_par"], p["L_cab"], p["R_m"], p["L_m"], sysc.Cx / 2)) for fk in fp]
    Hdp = [abs(ref.h_dm(fk, v["C_dc"], v["ESR"], v["ESL"], p["L_cab"], p["R_m"], p["L_m"])) for fk in fp]
    res.add_series("h_cm", "|H_CM| = V_meas/V_node", "dB", fp.tolist(), (20 * np.log10(Hcp)).tolist())
    res.add_series("h_dm", "|H_DM| = V_meas/I_cell", "dBΩ", fp.tolist(), (20 * np.log10(Hdp)).tolist(), dash=True)
    nm = int(min(A_v.size - 1, round(108e6 / sysc.fs)))
    fm = f[1 : nm + 1]
    xs_, ys_ = peak_hold(fm, a["cm"][1 : nm + 1])
    res.add_series("cm_proxy", "CM proxy peak-hold (노드 FFT × H_CM)", "dBµV", xs_, ys_)
    xs_, ys_ = lines_for_plot(fm, a["A_m"][1 : nm + 1])
    res.add_series("cm_direct", "CM proxy 고조파: 시뮬레이션 v_m 직접 FFT", "dBµV", xs_, ys_, style="points")
    xs_, ys_ = peak_hold(fm, a["dm"][1 : nm + 1])
    res.add_series("dm_proxy", "DM proxy peak-hold (DC-link 전류 FFT × H_DM)", "dBµV", xs_, ys_, dash=True)
    sm = a["smp"]
    tt_ns = sm["t"] - ta
    w1 = tt_ns < 300e-9
    w2 = (tt_ns > sysc.t_on - 20e-9) & (tt_ns < sysc.t_on + 300e-9)
    for key, lab, unit in (("v_n", "v_n (스위치 노드)", "V"), ("i_cp", "i_CM (C_par 경로)", "A"), ("i_d", "i_d (diode → DC link)", "A"), ("i_ch", "i_ch (하측 채널)", "A")):
        res.add_series(f"e1_{key}", lab, unit, tt_ns[w1].tolist(), sm[key][w1].tolist())
        res.add_series(f"e2_{key}", lab, unit, (tt_ns[w2] - sysc.t_on).tolist(), sm[key][w2].tolist())
    b1 = bands_from_traj(tr, ta, ta + 300e-9, CELL_BANDS, t_offset=ta)
    b2 = bands_from_traj(tr, ta + sysc.t_on - 20e-9, ta + sysc.t_on + 300e-9, CELL_BANDS, t_offset=ta + sysc.t_on)
    corners = [c for c in ({"x": 1 / (math.pi * p["D"] / sysc.fs), "label": "1/(πτ)"}, {"x": 1 / (math.pi * tr_eq), "label": "1/(πt_r)"}, {"x": a["f_ring"], "label": "f_ring"}) if c["x"] >= sysc.fs]
    res.add_plot("p_src", "Source: 스위치 노드 전압 spectrum과 사다리꼴 envelope", ["src_v", "src_env"], x_label="f", x_unit="Hz", y_label="고조파 진폭", y_unit="dBµV", kind="xy", log_x=True, level="C + A", vlines=corners,
                 proved="edge 모양(rise time)이 두 번째 corner 1/(πt_r) 위의 −40 dB/dec 기울기를 정하고, loop 링잉이 f_ring 근처에 봉우리를 만든다. envelope는 같은 V·D·t_r 사다리꼴의 상한선이다.",
                 not_yet="합성 채널 전류 모델의 파형이다. 실제 소자의 Miller·gate 동특성·역회복·비선형 Coss는 없다(EX02/FL02).")
    res.add_plot("p_path", "Path: CM·DM 경로 전달함수", ["h_cm", "h_dm"], x_label="f", x_unit="Hz", y_label="|H|", y_unit="dB", kind="xy", log_x=True, level="A",
                 proved="C_par는 저주파에서 +20 dB/dec로 CM을 통과시키고, 케이블 L·chassis 용량이 공진을 만든다. DM은 DC-link C·ESL과 선로 경로의 분배가 정한다.",
                 not_yet="lumped 합성 경로다. 실제 케이블은 분포정수이고 측정망은 규격 LISN이 아니다. 필터 삽입손실은 source/load 임피던스에 따라 다르다.")
    res.add_plot("p_meas", "Victim: 측정 proxy spectrum (relative, 규격 판정 아님)", ["cm_proxy", "cm_direct", "dm_proxy"], x_label="f", x_unit="Hz", y_label="proxy", y_unit="dBµV", kind="xy", log_x=True, level="C + A",
                 vlines=[{"x": 1e6, "label": "1 MHz"}, {"x": 10e6, "label": "10 MHz"}, {"x": 30e6, "label": "30 MHz"}],
                 proved="source spectrum × 경로 전달함수 = 측정 proxy. CM은 시뮬레이션한 v_m의 FFT와 주파수영역 곱이 일치한다(같은 선형 경로).",
                 not_yet="receiver detector(peak/QP/AV)·RBW·규격 limit·측정 setup이 없어 EMI 합격/불합격을 말할 수 없다 (NOT_EVALUABLE).")
    res.add_plot("p_e1", "turn-off edge: 스위치 노드 전압 (t = 0: 채널 전류 감소 시작)", ["e1_v_n"], x_label="t", x_unit="s", y_label="v_n", y_unit="V", bands=b1, group="edge_off", level="C",
                 hlines=[{"y": V, "label": "V_bus"}],
                 proved="부하전류가 C_node를 충전해 dv/dt를 만들고, diode가 도통하면 L_loop–C_node가 링잉한다(overshoot ≈ I_L√(L/C)).", not_yet="diode 역회복·비선형 Coss 없음.")
    res.add_plot("p_e1i", "turn-off edge: C_par 경로 CM 전류", ["e1_i_cp"], x_label="t", x_unit="s", y_label="i_CM", y_unit="A", bands=b1, group="edge_off", level="C",
                 hlines=[{"y": p["C_par"] * a["dvdt_up"], "label": "C_par·dv/dt"}],
                 proved="CM 전류는 edge 동안 C_par·dv/dt 규모이고 링잉 동안 같은 주파수로 진동한다.", not_yet="실제 heatsink 용량은 분포돼 있고 전압 의존성이 있을 수 있다.")
    res.add_plot("p_e1d", "turn-off edge: diode 전류 (DC link로 돌아가는 DM source)", ["e1_i_d"], x_label="t", x_unit="s", y_label="i_d", y_unit="A", bands=b1, group="edge_off", level="C",
                 hlines=[{"y": p["I_L"], "label": "I_L"}],
                 proved="부하전류가 diode 경로로 넘어가며 L_loop 전류가 링잉한다: DC-link 전류(DM source)의 고주파 성분.", not_yet="")
    res.add_plot("p_e2", "turn-on edge: 스위치 노드 전압 (t = 0: 채널 전류 상승 시작)", ["e2_v_n"], x_label="t", x_unit="s", y_label="v_n", y_unit="V", bands=b2, group="edge_on", level="C",
                 proved="채널 전류가 부하전류를 넘어서야 diode가 꺼지고 노드가 방전된다. turn-on dv/dt는 gate(방전 전류)가 정한다.", not_yet="역회복 전류 spike는 모델에 없다.")
    res.add_plot("p_e2i", "turn-on edge: 채널 전류와 diode 전류", ["e2_i_ch", "e2_i_d"], x_label="t", x_unit="s", y_label="전류", y_unit="A", bands=b2, group="edge_on", level="C",
                 hlines=[{"y": p["I_L"], "label": "I_L"}],
                 proved="채널 전류가 오르는 동안 diode 전류가 L_loop를 통해 줄고, 채널이 I_L + I_dis를 흘리는 동안 노드가 방전된다.", not_yet="")
    res.circuit = {"diagram": cell_circuit({**v, **p}).to_json(), "intervals": b1, "plot_group": "edge_off"}
    res.tables.append(Table("t_bands", "대역별 최대 proxy (relative)", ["대역", "CM [dBµV]", "DM [dBµV]"], [[f"{lo / 1e6:g}–{hi / 1e6:g} MHz", a[f"cm_{lo / 1e6:g}_{hi / 1e6:g}"], a[f"dm_{lo / 1e6:g}_{hi / 1e6:g}"]] for lo, hi in BANDS], note=PROXY_NOTE))
    res.verdict("NOT_EVALUABLE", "EMI compliance는 판정 불가 — receiver·규격 LISN·detector·limit이 없는 relative conducted-noise proxy다. 모델 수치 검증은 통과.")
    res.assumptions += [
        "hard-switched cell: 채널 전류는 gate 속도로 정한 PWL, diode는 이상 (역회복 없음), C_node 선형",
        "CM 경로: 노드→C_par(+R_par)→chassis→(2C_Y+기생 ∥ 케이블 L/2 → 측정망 R_m/2 ∥ L_m/2)→DC bus; DC bus 두 선은 HF에서 묶임",
        "DM 경로: DC-link 전류(diode 전류)가 C_dc(ESR·ESL)와 선로(2L_cab + 두 측정 branch)로 나뉨 — 주파수영역 계산",
        "측정망은 50 Ω ∥ 50 µH 형태의 합성 proxy (규격 LISN·receiver 아님)",
    ]
    res.not_valid_for += ["EMI 규격 합격/불합격", "방사 EMI", "실제 PCB·케이블 분포정수", "소자 데이터 기반 스위칭 손실"]
    res.interpretation = (
        f"노드 전압 edge(≈{a['dvdt_up'] * 1e-9:.3g} V/ns)는 C_par를 통해 chassis로 CM 전류(peak {a['icp_pk']:.3g} A)를 흘리고, 그 전류가 케이블과 측정망을 돌아 측정 port 전압을 만든다. "
        f"spectrum으로 보면 source(사다리꼴 + {a['f_ring'] / 1e6:.3g} MHz 링잉) × 경로(C_par의 +20 dB/dec, 케이블 공진) = victim이다. 어느 요소를 바꾸느냐(속도·경로·damping)에 따라 대역별 효과와 비용이 다르다(실험 3). "
        "이 결과는 상대 비교용 proxy이고, 규격 합격 여부는 실제 측정 setup으로만 말할 수 있다."
    )
    return res


# ======================================================================================
# Experiment 3: one change at a time - noise and cost side by side
# ======================================================================================


def run_mitigation(v: dict) -> Result:
    res = Result("EX09", "mitigation", "C (cell·CM 정확 시간영역) + A (DM·열·누설 계산)")
    base = _cell_p(v)
    base["snubber"] = False
    base["C_Y"] = 0.0
    variants = [
        ("baseline", "기준", {}),
        ("gate", f"gate 느리게 (×{v['gate_factor']:g})", {"gate_scale": v["gate_factor"]}),
        ("cpar", f"C_par ×{v['cpar_factor']:g} (두꺼운 절연)", {"C_par": base["C_par"] * v["cpar_factor"]}),
        ("snub", f"RC snubber ({fmt_si(v['C_s'], 'F', 3)}, {fmt_si(v['R_s'], 'Ω', 3)})", {"snubber": True}),
        ("ycap", f"Y 커패시터 {fmt_si(v['C_Y_add'], 'F', 3)} (선당)", {"C_Y": v["C_Y_add"]}),
    ]
    out = {}
    try:
        for key, lab, over in variants:
            pp = dict(base)
            pp.update(over)
            out[key] = (lab, pp, analyze_cell(pp, v, n_fft=2**15))
    except ValueError as exc:
        res.verdict("OUT_OF_VALIDITY", str(exc))
        return res
    b = out["baseline"][2]
    V, fsw = base["V_bus"], base["f_sw"]
    rows = []
    for key, (lab, pp, a) in out.items():
        d_cm = [a[f"cm_{lo / 1e6:g}_{hi / 1e6:g}"] - b[f"cm_{lo / 1e6:g}_{hi / 1e6:g}"] for lo, hi in BANDS]
        d_dm = [a[f"dm_{lo / 1e6:g}_{hi / 1e6:g}"] - b[f"dm_{lo / 1e6:g}_{hi / 1e6:g}"] for lo, hi in BANDS]
        cost = a["P_sw"] + a["P_snub"] - (b["P_sw"] + b["P_snub"])
        other = ""
        if key == "cpar":
            dT = v["P_dev"] * v["Rth_tim"] * (1.0 / v["cpar_factor"] - 1.0)
            other = f"절연 두께 ×{1 / v['cpar_factor']:g} → R_th,TIM ×{1 / v['cpar_factor']:g} → ΔT_j ≈ +{dT:.3g} K (P_dev {v['P_dev']:g} W)"
        elif key == "ycap":
            il = ref.line_capacitor_current(v["C_Y_add"], v["V_line"], v["f_line"])
            other = f"AC 측 Y라면 {v['V_line']:g} V·{v['f_line']:g} Hz 누설 {il * 1e3:.4g} mA/개 (단일 경로), HV 측이면 저장에너지·절연 한계"
        elif key == "snub":
            other = f"R_s {a['P_snub']:.3g} W + 채널 turn-on 증가분 {a['P_sw'] - b['P_sw']:.3g} W (C_sV²f = {ref.rc_snubber_loss(v['C_s'], V, fsw):.3g} W)"
        elif key == "gate":
            other = f"turn-off dv/dt {a['dvdt_up'] * 1e-9:.3g} V/ns (부하전류 제한이면 거의 불변), turn-on {a['dvdt_dn'] * 1e-9:.3g} V/ns"
        rows.append([lab] + d_cm + d_dm + [a["vmax"] - V, cost, other])
        xs_, ys_ = peak_hold(a["f"][1 : int(108e6 / fsw) + 1], a["cm"][1 : int(108e6 / fsw) + 1])
        res.add_series(f"cm_{key}", lab, "dBµV", xs_, ys_)
        sm = a["smp"]
        tt = sm["t"] - a["ta"]
        w = tt < 250e-9
        res.add_series(f"vn_{key}", lab, "V", tt[w].tolist(), sm["v_n"][w].tolist())
    cols = ["변경"] + [f"ΔCM {lo / 1e6:g}–{hi / 1e6:g} MHz [dB]" for lo, hi in BANDS] + [f"ΔDM {lo / 1e6:g}–{hi / 1e6:g} MHz [dB]" for lo, hi in BANDS] + ["overshoot [V]", "추가 손실 [W]", "다른 비용"]
    res.tables.append(Table("t_mit", "한 번에 하나씩 바꾼 결과: 잡음 변화와 비용 (기준 대비)", cols, rows, note=PROXY_NOTE + ". 손실은 같은 합성 cell 모델의 ∫v·i_ch와 R_s 손실."))
    sn = out["snub"][2]
    g = out["gate"][2]
    cp = out["cpar"][2]
    yc = out["ycap"][2]
    res.add_metric("dcm_cpar", f"C_par ×{v['cpar_factor']:g}: CM 0.15–1 MHz 변화", cp["cm_0.15_1"] - b["cm_0.15_1"], "dB", ref=20 * math.log10(v["cpar_factor"]), ref_label="20·log10(C 비) (C_par가 경로를 지배하는 저주파)", tol=0.05, abs_scale=1.0, basis="relative proxy")
    res.add_metric("dcm_cpar_hf", f"C_par ×{v['cpar_factor']:g}: CM 30–100 MHz 변화", cp["cm_30_100"] - b["cm_30_100"], "dB", basis="경로 공진이 이동해 고주파에서는 오히려 커질 수 있다")
    res.add_metric("dcm_gate_30_100", "gate 느리게: CM 30–100 MHz 변화", g["cm_30_100"] - b["cm_30_100"], "dB", basis="relative proxy")
    res.add_metric("dcm_gate_1_10", "gate 느리게: CM 1–10 MHz 변화", g["cm_1_10"] - b["cm_1_10"], "dB", basis="저주파는 거의 그대로 (envelope 첫 corner는 D·f_sw가 정함)")
    res.add_metric("dP_gate", "gate 느리게: 스위칭 손실 증가", g["P_sw"] - b["P_sw"], "W", basis="같은 모델 ∫v·i_ch")
    res.add_metric("os_snub", "snubber: overshoot 변화", (sn["vmax"] - V) - (b["vmax"] - V), "V")
    res.add_metric("P_snub_total", "snubber 비용 (R_s + 채널 turn-on 증가분)", sn["P_snub"] + sn["P_sw"] - b["P_sw"], "W", ref=ref.rc_snubber_loss(v["C_s"], V, fsw), ref_label="C_s·V²·f_sw", tol=0.1, basis="경부하·다른 전류에서도 거의 일정")
    res.add_metric("dcm_ycap", "Y 커패시터: CM 10–30 MHz 변화", yc["cm_10_30"] - b["cm_10_30"], "dB", basis="국부 귀환 경로가 측정망을 우회 (저주파에서는 Y 임피던스가 커서 효과 작음)")
    res.add_metric("emi_status", "EMI compliance", "NOT_EVALUABLE", "", basis="relative proxy 비교만")
    res.add_check(check_close("snubber 에너지: R_s 적분 + 채널 증가분 vs C_s V² f_sw", sn["P_snub"] + sn["P_sw"] - b["P_sw"], ref.rc_snubber_loss(v["C_s"], V, fsw), 0.1, "정확 시간영역 에너지 적분 (R_s·i² + ∫v·i_ch 차이) vs 닫힌 식 (두 edge마다 ½C_sV²)", True, "W",
                              detail=f"R_s {sn['P_snub']:.4g} W + 채널 {sn['P_sw'] - b['P_sw']:.4g} W; 링잉·L_loop 에너지 교환 때문에 정확히 같지는 않다"))
    for key, (lab, pp, a) in out.items():
        res.add_check(Check(f"에너지 원장: {lab}", "PASS" if abs(a["ledger"]["normalised"]) < 1e-6 else "FAIL", a["ledger"]["normalised"], "rel", 1e-6, path="부하 source·DC link·채널·R_loop·R_s·R_par·측정 R 에너지 정확 적분", independent=True))
    res.add_plot("p_cm", "CM proxy spectrum (1/20 decade peak-hold): 한 번에 하나씩 바꿀 때", [f"cm_{k}" for k in out], x_label="f", x_unit="Hz", y_label="CM proxy", y_unit="dBµV", kind="xy", log_x=True, level="C + A",
                 vlines=[{"x": 1e6, "label": "1 MHz"}, {"x": 30e6, "label": "30 MHz"}],
                 proved="C_par 축소는 C_par가 경로를 지배하는 저주파에서 20·log10(C 비)만큼 낮추지만 경로 공진을 옮겨 고주파 일부는 오히려 커질 수 있다. Y 커패시터는 수 MHz 이상에서, snubber는 링잉 봉우리에서 크다. gate 감속은 gate가 edge를 제한할 때만 고주파를 낮춘다.",
                 not_yet="relative proxy 비교다. 실제 receiver·규격 setup·방사 결합은 없다. 한 번에 하나씩 바꾼 결과이며 상호작용은 따로 봐야 한다.")
    res.add_plot("p_vn", "turn-off edge 노드 전압: 변경별", [f"vn_{k}" for k in out], x_label="t", x_unit="s", y_label="v_n", y_unit="V", kind="time", level="C", hlines=[{"y": V, "label": "V_bus"}],
                 proved="snubber는 dv/dt와 overshoot를 낮추고 링잉을 감쇠시킨다. gate 감속은 부하전류가 dv/dt를 제한하는 turn-off edge에서는 효과가 작다.",
                 not_yet="gate 감속의 효과는 turn-on edge(gate 제한)에서 크다 — 실험 2의 turn-on 그래프.")
    res.circuit = {"diagram": cell_circuit({**v, **base, "snubber": True}).to_json(), "intervals": [], "plot_group": ""}
    res.verdict("NOT_EVALUABLE", "EMI compliance는 판정 불가 (relative proxy). 대역별 잡음 변화와 손실·열·누설 비용을 같은 모델에서 비교했다.")
    res.assumptions += ["기준 cell·경로는 실험 2와 같음 (snubber·Y 없음에서 출발)", f"C_par 축소 = 같은 면적에서 절연 두께 증가 (R_th,TIM ∝ 1/C_par; P_dev {v['P_dev']:g} W, R_th,TIM {v['Rth_tim']:g} K/W는 ASSUMED)", "Y 커패시터 누설은 AC 측 단일 이상 경로 계산"]
    res.not_valid_for += ["EMI 규격 합격/불합격", "필터 설계 확정", "실제 소자 gate 저항–손실 곡선"]
    res.interpretation = (
        "잡음을 줄이는 수단마다 효과가 나타나는 대역과 비용이 다르다. C_par를 줄이면 C_par가 지배하는 저주파에서 20·log10(C 비)만큼 줄지만 경로 공진이 옮겨 고주파 일부는 커질 수 있고, 절연이 두꺼워져 열저항이 커진다. "
        f"Y 커패시터는 수 MHz 이상의 CM을 크게 줄이지만 누설·저장에너지 한계가 있다. 이 조건에서 gate를 {v['gate_factor']:g}배 느리게 하면 30–100 MHz가 {g['cm_30_100'] - b['cm_30_100']:.2g} dB밖에 줄지 않는데 스위칭 손실은 {g['P_sw'] - b['P_sw']:.3g} W 늘었다 — "
        f"고주파를 지배하는 turn-off edge가 부하전류로 제한되기 때문이다. snubber는 링잉을 잡는 대신 약 C_sV²f = {ref.rc_snubber_loss(v['C_s'], V, fsw):.3g} W를 R_s와 채널에서 소모한다. "
        "그래서 ‘gate 저항만 키우자’가 아니라 지배 대역을 먼저 찾고 같은 loss budget·같은 측정 setup에서 비교한다."
    )
    return res


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "SiC라서 EMI가 나쁘다는 진단이 맞나?",
        "진단이 아니다. source(스위칭 edge의 dv/dt·di/dt·링잉 spectrum), path(C_par·chassis·Y·케이블·측정망 임피던스), victim(측정 port·민감도)을 분리해 본다. 같은 소자도 gate 속도·loop L·heatsink 용량·필터 배치에 따라 spectrum이 달라진다.",
        "Is 'it is SiC, so EMI is bad' a diagnosis?",
        "No. I separate the source, meaning the edge dv/dt, di/dt and ringing spectrum, the path through the parasitic capacitance, chassis, Y capacitors, cable and measurement network, and the victim port. The same device gives a different spectrum with a different gate speed, loop inductance, heatsink capacitance or filter placement.",
        ["source", "path", "victim", "spectrum"],
        kind="pressure",
    ),
    Question(
        "C_par 100 pF, dv/dt 50 kV/µs면 CM 전류는 5 A다. 이게 LISN에서 읽힐 값인가?",
        "아니다. 5 A는 edge 동안의 peak 변위전류 screen이다. 실제 측정값은 반복률, rise time, 귀환 경로와 케이블 임피던스, 측정망, receiver detector·대역폭이 정한다. 단위(50 kV/µs = 50 V/ns = 5×10¹⁰ V/s)를 틀리면 세 자릿수가 틀린다.",
        "With 100 pF and 50 kV/µs the common-mode current is 5 A. Is that what the LISN reads?",
        "No. Five amps is a peak displacement-current screen during the edge. The measured value depends on the repetition rate, rise time, return path and cable impedance, the measurement network and the receiver detector and bandwidth. And a unit slip in the dv/dt changes the answer by three orders of magnitude.",
        ["peak screen", "경로·측정망", "receiver", "단위 50 V/ns"],
        kind="calc",
    ),
    Question(
        "gate 저항을 키워 EMI를 줄이면 되지 않나?",
        "고주파(두 번째 corner 위)는 줄 수 있지만 스위칭 손실과 온도가 늘고, 부하전류가 dv/dt를 제한하는 edge에는 효과가 작다. 같은 loss budget·같은 receiver setup에서 snubber, C_par 축소, Y 커패시터, 필터와 비교하고 각 비용(손실·열·누설·저장에너지)을 함께 보고한다.",
        "Can't you just raise the gate resistance to fix EMI?",
        "It can lower the high-frequency part above the second spectral corner, but it raises switching loss and temperature, and it barely changes an edge whose dv/dt is limited by the load current. I would compare it with a snubber, a smaller parasitic capacitance, Y capacitors and a filter under the same loss budget and the same receiver setup, and report each cost.",
        ["고주파만", "손실·온도", "부하전류 제한 edge", "같은 조건 비교"],
        kind="pressure",
    ),
    Question(
        "2.2 nF Y 커패시터의 230 V·50 Hz 전류가 0.159 mA면 누설전류 규격을 만족하나?",
        "판단할 수 없다. 2πfCV는 단일 이상 경로 계산이다. 실제 누설·접촉전류 시험은 측정망(인체 모델), 상 결선, 스위칭 주파수 CM 전류, 여러 경로 합을 포함한다. 규격과 안전 등급은 실제 제품·적용 규격으로 확인한다.",
        "The 2.2 nF Y capacitor draws 0.159 mA at 230 V 50 Hz. Does it meet the leakage limit?",
        "That cannot be decided from this number. Two pi f C V is a single ideal path. A real leakage or touch-current test uses a body network, the phase connection, switching-frequency common-mode current and the sum of all paths, and the limits come from the actual product standard.",
        ["단일 경로", "측정망", "스위칭 주파수 CM", "규격 확인"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="hand_screens",
        title="손계산 screen: C·dv/dt = 5 A, 50.329 MHz, 0.158965 mA — 그리고 단위 함정",
        goal=(
            "C_par 100 pF × 50 kV/µs = 5 A(peak 변위전류 screen), 10 nH·1 nF 링잉 50.329 MHz, 2.2 nF의 230 Vrms·50 Hz 전류 0.158965 mA를 재현하고, "
            "각 숫자를 정확 시간영역 해(램프 전류, RLC 영점 교차)와 수치 구적으로 따로 확인한다. kV/µs와 V/ns의 단위 함정을 표로 본다. 모두 조건이 붙은 screen이며 EMI·누설 합격 판정이 아니다."
        ),
        params=[
            Param("C_par", "노드→chassis 기생 용량 C_par", "F", 100e-12, "pF", vmin=1e-14, vmax=1e-6, source="TEXTBOOK", source_note="교재 E09 100 pF"),
            Param("dvdt", "dv/dt", "V/us", 5e4, "kV/us", vmin=1.0, vmax=1e6, source="TEXTBOOK", source_note="교재 E09 50 kV/µs = 50 V/ns"),
            Param("V_step", "edge 전압 스텝", "V", 800.0, "V", vmin=1, vmax=2000, source="ASSUMED", source_note="800 V bus"),
            Param("R_path", "경로 저항 (낮은 임피던스 경로)", "Ω", 1.0, "Ω", vmin=1e-3, vmax=1e4, source="ASSUMED", group="시간영역 검산"),
            Param("L_path", "경로 인덕턴스 (낮은 임피던스 경로)", "H", 1e-9, "nH", vmin=1e-12, vmax=1e-3, source="ASSUMED", group="시간영역 검산"),
            Param("R_path2", "측정망 같은 경로 저항", "Ω", 50.0, "Ω", vmin=1e-3, vmax=1e4, source="ASSUMED", group="시간영역 검산"),
            Param("L_path2", "측정망 같은 경로 인덕턴스", "H", 100e-9, "nH", vmin=1e-12, vmax=1e-3, source="ASSUMED", group="시간영역 검산"),
            Param("L_r", "링잉 L", "H", 10e-9, "nH", vmin=1e-12, vmax=1e-3, source="TEXTBOOK", source_note="교재 10 nH", group="링잉"),
            Param("C_r", "링잉 C", "F", 1e-9, "nF", vmin=1e-15, vmax=1e-3, source="TEXTBOOK", source_note="교재 1 nF", group="링잉"),
            Param("R_r", "링잉 감쇠 R (직렬)", "Ω", 0.1, "Ω", vmin=0.0, vmax=100, source="ASSUMED", group="링잉"),
            Param("C_y", "chassis 용량 (Y)", "F", 2.2e-9, "nF", vmin=1e-13, vmax=1e-5, source="TEXTBOOK", source_note="교재 2.2 nF", group="line 전류"),
            Param("V_line", "line 전압 (RMS)", "V", 230.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="230 Vrms", group="line 전류"),
            Param("f_line", "line 주파수", "Hz", 50.0, "Hz", vmin=1, vmax=1e6, source="TEXTBOOK", source_note="50 Hz", group="line 전류"),
        ],
        presets=[
            Preset("textbook", "교재 세 숫자", {}, "E09", ("nominal", "reference")),
            Preset("slow_edge", "dv/dt 10 kV/µs", {"dvdt": 1e4}, "screen이 1/5", ("variant",)),
            Preset("meas_path", "경로 50 Ω·1 µH", {"L_path2": 1e-6}, "경로 임피던스가 peak를 바꾼다", ("variant",)),
        ],
        run=run_hand_screens,
        model_level="A (screen) + C (검산)",
        suggested_change="dv/dt를 50 → 10 kV/µs로 낮춘다.",
        prediction=Prediction(
            "C_par 100 pF에 50 kV/µs가 걸릴 때 edge 동안의 peak 변위전류 screen은?",
            ["5 mA", "5 A", "5 kA", "모르겠다"],
            "5 A",
            "50 kV/µs = 5×10¹⁰ V/s, 100 pF × 5×10¹⁰ V/s = 5 A. V/µs로 읽으면 5 mA, nF로 읽으면 5 kA — 단위 하나로 세 자릿수가 틀린다. 그리고 5 A는 edge 동안의 peak screen이지 LISN 읽음이 아니다.",
            ["i_pk", "i_pk_path2"],
            handcalc=[{"key": "i_pk", "label": "C·dv/dt", "unit": "A"}, {"key": "f_r", "label": "f_r", "unit": "Hz"}, {"key": "i_line", "label": "2πfCV", "unit": "A"}],
        ),
        suggested={"dvdt": 1e4},
        student="커패시터에 흐르는 전류는 전압이 얼마나 빨리 변하는지에 비례한다(i = C dv/dt). 스위치 노드가 수십 ns에 800 V를 오르내리면 작은 기생 용량으로도 수 A가 흐른다. 이 전류가 chassis와 케이블을 돌아오면 잡음이 된다.",
        expert=(
            "screen 세 개는 모두 조건부다: ① C·dv/dt는 edge 동안의 peak이고 경로 임피던스가 RC ≪ t_r일 때만 붙는다(50 Ω·100 nH 경로에서는 달라짐). ② 링잉 주파수는 L·C 곱만 알려 주므로 L과 C를 따로 식별하지 못한다(EX11). "
            "③ 2πfCV는 단일 경로의 line 주파수 전류로, 측정망·스위칭 주파수 CM·상 결선을 포함한 누설전류 시험이 아니다. 단위는 SI로 먼저 바꾼다: 50 kV/µs = 50 V/ns = 5×10¹⁰ V/s."
        ),
        customer_ko="100 pF에 50 kV/µs면 edge 동안 약 5 A의 CM 변위전류가 흐를 수 있습니다. 이 값은 LISN 측정값이 아니라 크기 screen이라, 실제 대응은 귀환 경로와 측정 setup을 확인한 뒤 정하시죠.",
        customer_en="With 100 picofarads and 50 kilovolts per microsecond, about 5 amps of common-mode displacement current can flow during each edge. That is a size screen, not the LISN reading, so let's confirm the return path and the measurement setup before choosing a countermeasure.",
        questions=[_Q[1], _Q[3]],
        circuit="emi_ramp",
        textbook=[TB_E09, TB_05],
        reference_presets=["textbook"],
        claim_limit="screen 수치와 그 시간영역 검산. EMI·누설전류 합격 판정이 아니다.",
    ),
    Experiment(
        key="source_path_victim",
        title="Source → Path → Victim: 스위칭 edge에서 측정 port까지의 CM/DM proxy spectrum",
        goal=(
            "hard-switched cell(부하전류·gate 속도·loop L·노드 C)이 만드는 스위치 노드 전압과 DC-link 전류를 정확 시간영역으로 풀고, CM 경로(C_par → chassis → Y ∥ 케이블 → 50 Ω ∥ 50 µH proxy)와 "
            "DM 경로(DC-link C·ESL ∥ 선로)를 통과한 측정 port의 relative spectrum을 만든다. 사다리꼴 envelope의 두 corner(1/(πτ), 1/(πt_r))와 링잉 봉우리를 확인한다. EMI compliance는 NOT_EVALUABLE이다."
        ),
        params=_cell_params(),
        presets=[
            Preset("nominal", "800 V·50 A·100 kHz, C_par 100 pF, loop 10 nH, C_node 1 nF", {}, "E09 숫자와 맞춘 합성 cell", ("nominal", "reference")),
            Preset("light", "부하전류 10 A", {"I_L": 10.0}, "turn-off dv/dt가 부하전류로 느려짐", ("variant",)),
            Preset("ycap", "Y 커패시터 2.2 nF", {"C_Y": 2.2e-9}, "국부 귀환 경로", ("variant",)),
        ],
        run=run_spv,
        model_level="C + A",
        suggested_change="부하전류를 50 → 10 A로 낮춘다 (turn-off dv/dt는 부하전류가 정한다).",
        prediction=Prediction(
            "부하전류를 50 → 10 A로 낮추면 turn-off 노드 dv/dt와 30–100 MHz CM proxy는?",
            ["dv/dt ↓, 고주파 CM ↓", "dv/dt 그대로 (gate가 정함)", "dv/dt ↑", "모르겠다"],
            "dv/dt ↓, 고주파 CM ↓",
            "turn-off에서 채널이 빨리 꺼지면 노드는 부하전류가 C_node를 충전하는 속도(I_L/C)로 올라간다. 전류가 1/5이면 dv/dt도 1/5이 되어 두 번째 corner가 내려가고 고주파가 준다. turn-on edge는 gate(방전 전류)가 정하므로 그대로다.",
            ["dvdt_up", "cm_30_100", "icp_pk"],
            handcalc=[{"key": "dvdt_up", "label": "turn-off dv/dt", "unit": "V/s"}],
        ),
        suggested={"I_L": 10.0},
        student=(
            "EMI는 세 부분으로 나눠 본다. 잡음을 만드는 곳(스위치 노드의 빠른 전압 변화와 링잉), 잡음이 지나가는 길(heatsink로 가는 기생 용량, chassis, 케이블), 잡음을 받는 곳(측정 장비의 50 Ω port). "
            "빠른 edge는 높은 주파수 성분을 많이 만들고, 기생 용량은 높은 주파수일수록 전류를 잘 통과시킨다."
        ),
        expert=(
            "사다리꼴 spectrum은 2VD에서 1/(πτ) 위로 −20 dB/dec, 1/(πt_r) 위로 −40 dB/dec로 떨어지고 링잉이 f_ring에 봉우리를 만든다. CM 경로 전달함수는 C_par가 지배하는 저주파에서 +20 dB/dec, 케이블 L과 chassis 용량이 공진을 만든다. "
            "이 실험의 CM proxy는 시뮬레이션한 측정 port 전압의 FFT와 주파수영역 곱(source × H)이 일치하고, Parseval로 시간영역 RMS와 대조했다. DM은 DC-link 전류(diode 전류) spectrum × 분배 전달함수다. "
            "receiver detector(peak/QP/AV)·RBW·규격 LISN이 없으므로 절대 dBµV는 규격 limit과 비교하지 않는다 — relative proxy다."
        ),
        customer_ko=(
            "현재 조건에서 스위치 노드 edge가 약 50 V/ns, 링잉이 약 50 MHz라 30 MHz 이상 CM 성분이 큽니다. heatsink 기생 용량·Y 커패시터·케이블 경로 중 무엇이 지배하는지 먼저 분리하고, "
            "같은 LISN·receiver setup에서 대책별로 비교하시죠. 지금 수치는 상대 비교용이며 규격 판정이 아닙니다."
        ),
        customer_en=(
            "Here the switch-node edge is about 50 volts per nanosecond with ringing near 50 megahertz, so the common-mode content above 30 megahertz is high. "
            "Let's first separate whether the heatsink capacitance, the Y capacitors or the cable path dominates, then compare countermeasures with the same LISN and receiver setup. These numbers are for relative comparison, not a compliance verdict."
        ),
        questions=[_Q[0]],
        circuit="emi_cell",
        textbook=[TB_E09, TB_05],
        reference_presets=["nominal"],
        runtime_hint="seconds",
        claim_limit="합성 lumped 경로의 relative conducted-noise proxy. EMI compliance PASS를 만들지 않는다.",
    ),
    Experiment(
        key="mitigation",
        title="gate 속도·경로 C·snubber·Y를 하나씩 바꾸고 잡음과 손실을 같이 본다",
        goal=(
            "기준 cell에서 gate 속도(×2 느리게), C_par(×0.5), RC snubber, Y 커패시터를 한 번에 하나씩 바꿔 대역별 CM/DM proxy 변화와 비용(스위칭 손실 ∫v·i_ch, snubber C_sV²f, 절연 두께에 따른 열저항, "
            "Y의 line 주파수 누설)을 한 표로 비교한다."
        ),
        params=_cell_params(
            [
                Param("gate_factor", "gate 감속 배율", "", 2.0, "", vmin=1.0, vmax=20, source="ASSUMED", group="변경"),
                Param("cpar_factor", "C_par 배율", "", 0.5, "", vmin=0.01, vmax=1.0, source="ASSUMED", group="변경"),
                Param("C_Y_add", "추가 Y 커패시터 (선당)", "F", 2.2e-9, "nF", vmin=1e-12, vmax=1e-6, source="TEXTBOOK", source_note="교재 2.2 nF", group="변경"),
                Param("P_dev", "소자 손실 (열 비용 계산용)", "W", 60.0, "W", vmin=0.1, vmax=5000, source="ASSUMED", group="비용"),
                Param("Rth_tim", "절연·TIM 열저항 (기준)", "K/W", 0.15, "K/W", vmin=0.001, vmax=10, source="ASSUMED", group="비용"),
                Param("V_line", "line 전압 (Y 누설 계산)", "V", 230.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", group="비용"),
                Param("f_line", "line 주파수", "Hz", 50.0, "Hz", vmin=1, vmax=1000, source="TEXTBOOK", group="비용"),
            ]
        ),
        presets=[
            Preset("nominal", "기준 + 네 가지 변경", {}, "", ("nominal", "reference")),
            Preset("big_snubber", "snubber C_s 1 nF", {"C_s": 1e-9}, "snubber 비용 2배", ("variant",)),
            Preset("slow_gate4", "gate ×4 느리게", {"gate_factor": 4.0}, "gate 제한이 turn-off까지", ("variant",)),
        ],
        run=run_mitigation,
        model_level="C + A",
        suggested_change="snubber C_s를 0.47 → 1 nF로 키운다.",
        prediction=Prediction(
            "snubber C_s를 0.47 → 1 nF로 키우면 snubber 비용(R_s 손실 + 채널 turn-on 증가분)은?",
            ["약 2배 (C_sV²f)", "그대로 (R_s만 손실)", "줄어든다", "모르겠다"],
            "약 2배 (C_sV²f)",
            "노드가 V_bus까지 오르내릴 때마다 C_s가 충·방전되므로 한 주기 비용은 약 C_s·V²·f_sw다(800 V·100 kHz에서 1 nF당 64 W). R_s에서 일부, 채널이 turn-on 때 C_s를 방전하며 나머지를 소모한다. 링잉은 더 잘 잡히지만 dv/dt 감소와 손실을 같이 본다.",
            ["P_snub_total", "os_snub"],
            handcalc=[{"key": "P_snub_total", "label": "snubber 비용", "unit": "W"}],
        ),
        suggested={"C_s": 1e-9},
        student="잡음을 줄이는 방법은 여러 가지지만 공짜는 없다. 스위치를 천천히 켜면 잡음은 줄지만 켜고 끄는 동안 손실이 늘고, snubber는 링잉을 잡는 대신 매번 커패시터를 충·방전하며 전력을 소모한다.",
        expert=(
            "대역별로 효과가 다르다: C_par 축소는 경로가 C_par로 지배되는 저주파에서 20·log10(C 비)지만 경로 공진을 옮겨 고주파 일부를 키울 수 있고, Y는 국부 귀환으로 수 MHz 이상, gate 감속은 gate가 edge를 제한할 때만 두 번째 corner 위 고주파, snubber는 링잉 봉우리. "
            "비용: gate 감속은 ∫v·i_ch 증가(부하전류 제한 edge에는 효과가 작음), snubber는 ≈ C_sV²f(R_s와 채널로 나뉨), C_par 축소는 절연 두께 → 열저항, Y는 누설·저장에너지·절연 등급. "
            "‘Cs를 두 배’로 고정하지 말고 링 주파수·감쇠·source 임피던스로 후보를 만들고, 같은 loss budget과 receiver setup에서 비교한다."
        ),
        customer_ko=(
            "이 조건에서는 gate 저항을 키워도 고주파 잡음이 거의 줄지 않는데(turn-off edge가 부하전류로 제한) 스위칭 손실은 수십 W 늘고, snubber는 약 30 W를 씁니다. heatsink 절연을 두껍게 하면 저주파 잡음은 줄지만 온도가 오르고 공진이 옮겨 가니, "
            "측정에서 지배 대역을 확인한 뒤 손실·열 예산 안에서 대책 조합을 고르시죠."
        ),
        customer_en=(
            "Here a larger gate resistance hardly lowers the high-frequency noise, because the turn-off edge is limited by the load current, yet it adds tens of watts of switching loss; the snubber costs about 30 watts. Thicker heatsink insulation lowers the low-frequency noise but raises temperature and moves the path resonance, "
            "so let's identify the dominant band in the measurement first and pick a combination within the loss and thermal budget."
        ),
        questions=[_Q[2]],
        circuit="emi_cell",
        textbook=[TB_E09],
        reference_presets=["nominal"],
        runtime_hint="seconds",
        claim_limit="relative proxy의 대역별 변화와 합성 모델의 손실·열 비용. 필터·대책의 규격 합격을 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="EX09",
    title="EMI: noise source, 경로, victim을 같이 본다",
    title_en="EMI: noise source, path and victim together",
    track="expert",
    order=9,
    path_note="E13 5회전 (E08–E11)",
    textbook=[TB_E09, TB_05, TB_E11],
    prerequisites=["FL02", "FL04", "EX07"],
    summary="손계산 screen과 단위 → 스위칭 cell의 정확 시간영역 → CM/DM 경로 → relative spectrum proxy → 대책을 하나씩 바꾸며 잡음과 비용을 같이 본다. EMI compliance는 NOT_EVALUABLE.",
    experiments=EXPERIMENTS,
    minimum_scope="CM/DM source–path–victim lumped network와 spectrum proxy; 5 A·50.329 MHz·0.158965 mA; gate 속도·경로 C·snubber 변화와 손실 (지침 10절)",
    claim_limits=[
        "relative conducted-noise proxy — receiver·규격 LISN이 없어 EMI compliance는 NOT_EVALUABLE, PASS를 만들지 않음",
        "합성 lumped 경로와 합성 채널 전류 모델 (vendor 소자·실제 PCB 아님)",
        "누설전류 계산은 단일 이상 경로 (규격 시험 아님)",
        "3-level NPC midpoint 확장은 구현하지 않음 (지침상 선택 사항)",
    ],
    test_paths=["tests/test_ex09.py"],
    extends=["FL02", "FL04", "EX07"],
)
