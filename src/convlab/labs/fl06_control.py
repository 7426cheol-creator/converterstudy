"""FL06 - Control: the plant and the sign come before the PI (textbook ch.09).

Levels, kept apart on purpose:
  A  loop-gain algebra with units: PI on the RL plant L di/dt = u - R i - e, the PI zero on the
     plant pole (Kp = L w_c [V/A], Ki = R w_c [V/(A s)] for a volt output; a duty output divides
     by the modulator gain K_PWM [V/duty]); a pure delay as exp(-s Td).  The crossover and the
     phase margin are found numerically on the full complex loop gain (independent of the gain
     formula) and compared with the exact sampled-data loop (ZOH plant, computation delay).
  B  sampled-data current loop: exact ZOH map of the RL plant between samples, sampled PI with
     computation delay, voltage saturation and back-calculation anti-windup (xI in V, Kaw in 1/s).
     The ZOH map is checked against the exact continuous simulation of the same plant with the
     switched-affine engine (piecewise-constant source, energy ledger).
  B  dq current control of a three-phase grid-connected RL filter (amplitude-invariant dq, current
     positive from grid to converter).  The dq model is checked against an independent abc
     simulation (three phase ODEs, scipy DOP853, own Park transform of the measured currents).
Controllers are sampled at t_k = k Ts; the command computed from sample k is applied (held) from
t_{k+n_d}.  Nothing here models PWM ripple, dead time or device limits other than the voltage limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq

from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import control as ref
from ._common import energy_ledger, ledger_check, sym_linear

TB_09 = TextbookRef("제어-pi보다-plant와-부호가-먼저다-fl06", "09. 제어 — PI보다 plant와 부호가 먼저다 [FL06]")
TB_08 = TextbookRef("obcpfc-전력품질을-설계-변수로-바꾸기-fl05", "08. OBC·PFC — 전력품질을 설계 변수로 바꾸기 [FL05]")
TB_E07 = TextbookRef("expert-e07-제어-단일-루프가-안정해도-시스템은-불안정할-수-있다", "E07. 제어: 단일 루프가 안정해도 시스템은 불안정할 수 있다 [EX07]")

PM_LEARNING_MIN = 45.0  # deg; learning threshold for "enough margin" (not a customer requirement)


# ======================================================================================
# Plant, discretisation and loop algebra
# ======================================================================================


def rl_zoh(L: float, R: float, Ts: float) -> tuple[float, float]:
    """Exact ZOH map of L di/dt = v - R i over one sample: i+ = a i + b v (v held)."""
    a = math.exp(-R * Ts / L)
    b = (1.0 - a) / R if R > 0 else Ts / L
    return a, b


def loop_gain_cont(f, Kp, Ki, L, R, Td=0.0, K_out=1.0):
    """Continuous loop gain K_out (Kp + Ki/s) / (L s + R) exp(-s Td) at frequencies f [Hz]."""
    s = 2j * np.pi * np.asarray(f, dtype=float)
    return K_out * (Kp + Ki / s) / (L * s + R) * np.exp(-s * Td)


def loop_gain_disc(f, Kp, Ki, L, R, Ts, n_delay=1, K_out=1.0):
    """Exact sampled-data loop gain: forward-Euler PI, ZOH plant b/(z-a), z^-n_delay computation delay."""
    z = np.exp(2j * np.pi * np.asarray(f, dtype=float) * Ts)
    a, b = rl_zoh(L, R, Ts)
    C = Kp + Ki * Ts / (z - 1.0)
    return K_out * C * b / (z - a) * z ** (-n_delay)


@dataclass
class Margins:
    fc: float | None  # crossover [Hz]
    pm: float | None  # phase margin [deg]
    f180: float | None  # phase crossover [Hz]
    gm_db: float | None  # gain margin [dB]
    note: str = ""


def margins(fun, f_lo: float, f_hi: float, n: int = 4000) -> Margins:
    """Crossover and margins of a loop gain fun(f) on [f_lo, f_hi] (log grid + bracketing root finder).

    The phase is unwrapped from the low-frequency end; the first gain crossover and the first
    -180 deg crossing are reported.
    """
    fs = np.geomspace(f_lo, f_hi, n)
    Lg = fun(fs)
    mag = np.abs(Lg)
    ph = np.unwrap(np.angle(Lg))
    fc = pm = f180 = gm = None
    lm = np.log(mag)
    idx = np.where((lm[:-1] > 0) & (lm[1:] <= 0))[0]
    if idx.size:
        k = int(idx[0])
        fc = brentq(lambda x: math.log(abs(complex(fun(np.array([x]))[0]))), fs[k], fs[k + 1], xtol=1e-12 * fs[k], rtol=1e-13)
        ph_c = np.interp(math.log(fc), np.log(fs[k : k + 2]), ph[k : k + 2])
        # exact phase at fc, re-anchored to the unwrapped branch
        pa = float(np.angle(fun(np.array([fc]))[0]))
        pa += 2 * math.pi * round((ph_c - pa) / (2 * math.pi))
        pm = 180.0 + math.degrees(pa)
    idx = np.where((ph[:-1] > -math.pi) & (ph[1:] <= -math.pi))[0]
    if idx.size:
        k = int(idx[0])
        off = 2 * math.pi * round((ph[k] - float(np.angle(fun(np.array([fs[k]]))[0]))) / (2 * math.pi))

        def phase_err(x):
            return float(np.angle(fun(np.array([x]))[0])) + off + math.pi

        try:
            f180 = brentq(phase_err, fs[k], fs[k + 1], xtol=1e-12 * fs[k], rtol=1e-13)
        except ValueError:
            f180 = float(np.interp(-math.pi, [ph[k + 1], ph[k]], [fs[k + 1], fs[k]]))
        gm = -20.0 * math.log10(abs(complex(fun(np.array([f180]))[0])))
    return Margins(fc, pm, f180, gm)


def closed_loop_poles(Kp, Ki, L, R, Ts, n_delay, K_out=1.0) -> np.ndarray:
    """Eigenvalues of the linear sampled-data loop, state [i, xI, u(k-1), ..., u(k-n_d)]."""
    a, b = rl_zoh(L, R, Ts)
    nd = max(int(n_delay), 0)
    n = 2 + nd
    M = np.zeros((n, n))
    # u_cmd[k] = K_out (-Kp i[k] + xI[k])  (reference and disturbance set to zero)
    ucmd = np.zeros(n)
    ucmd[0] = -K_out * Kp
    ucmd[1] = K_out
    if nd == 0:
        M[0, :] = a * np.eye(n)[0] + b * ucmd
    else:
        M[0, 0] = a
        M[0, 1 + nd] = b  # oldest command is applied now
    M[1, 0] = -Ts * Ki
    M[1, 1] = 1.0
    if nd >= 1:
        M[2, :] = ucmd
        for j in range(1, nd):
            M[2 + j, 1 + j] = 1.0
    return np.linalg.eigvals(M)


# ======================================================================================
# Sampled current loop (B level) with saturation and anti-windup
# ======================================================================================


@dataclass
class PIController:
    """Sampled PI in controller units y; the actuator applies u = K_out * y.

    V output: K_out = 1, gains in V/A and V/(A s), xI in V.  Duty output: K_out = K_PWM, gains in
    1/A and 1/(A s), xI dimensionless.  Back-calculation: xI+ = xI + Ts (Ki e + Kaw (y_sat - y_unsat)).
    """

    Kp: float
    Ki: float
    Ts: float
    y_max: float
    K_out: float = 1.0
    Kaw: float = 0.0
    aw: bool = True
    xI: float = 0.0

    def step(self, err: float, y_ff: float = 0.0) -> tuple[float, float]:
        y_un = self.Kp * err + self.xI + y_ff
        y_sat = min(max(y_un, -self.y_max), self.y_max)
        corr = self.Kaw * (y_sat - y_un) if self.aw else 0.0
        self.xI += self.Ts * (self.Ki * err + corr)
        return y_un, y_sat


@dataclass
class LoopRun:
    t: np.ndarray  # sample instants
    i: np.ndarray  # plant current at the sample instants
    r: np.ndarray  # reference after the limiter
    r_req: np.ndarray  # requested reference
    u_un: np.ndarray  # unsaturated command [V]
    u_cmd: np.ndarray  # saturated command [V]
    u_app: np.ndarray  # voltage actually applied during [t_k, t_k+1)
    e: np.ndarray  # back-EMF during [t_k, t_k+1)
    xI: np.ndarray  # integrator (controller units) after the update at k
    meas: np.ndarray  # measured current
    extra: dict = field(default_factory=dict)


def run_current_loop(L, R, Ts, pi: PIController, n: int, r_req, e_true, offset=0.0, e_ff=True, i_lim=math.inf, n_delay=1, i0=0.0, u_prev=0.0) -> LoopRun:
    """Sampled-data RL current loop with the exact ZOH plant map.

    ``r_req(k)`` requested current, ``e_true(k)`` back-EMF held over sample k (step changes at
    sample instants), ``offset`` added to the measured current, ``e_ff`` feedforward of the
    sampled back-EMF, ``i_lim`` reference limiter, ``n_delay`` computation delay in samples.
    """
    a, b = rl_zoh(L, R, Ts)
    t = np.arange(n) * Ts
    i = np.zeros(n + 1)
    i[0] = i0
    pipe = [u_prev] * max(n_delay, 0)
    out = {k: np.zeros(n) for k in ("r", "rq", "uu", "uc", "ua", "e", "xI", "m")}
    for k in range(n):
        rq = r_req(k)
        r = min(max(rq, -i_lim), i_lim)
        m = i[k] + offset
        ek = e_true(k)
        y_ff = (ek / pi.K_out) if e_ff else 0.0
        y_un, y_sat = pi.step(r - m, y_ff)
        u_un, u_c = pi.K_out * y_un, pi.K_out * y_sat
        if n_delay > 0:
            pipe.append(u_c)
            u_a = pipe.pop(0)
        else:
            u_a = u_c
        i[k + 1] = a * i[k] + b * (u_a - ek)
        for key, val in (("r", r), ("rq", rq), ("uu", u_un), ("uc", u_c), ("ua", u_a), ("e", ek), ("xI", pi.xI), ("m", m)):
            out[key][k] = val
    return LoopRun(t, i[:n], out["r"], out["rq"], out["uu"], out["uc"], out["ua"], out["e"], out["xI"], out["m"], {"i_end": i[n]})


class RLPlant(HybridSystem):
    """L di/dt = u_k - e_k - R i with (u_k, e_k) held over sample k: the exact continuous plant."""

    state_names = ("i",)
    state_units = ("A",)

    def __init__(self, L, R, Ts, u, e):
        self.L, self.R, self.Ts = L, R, Ts
        self.u = list(u)
        self.e = list(e)
        self._key = f"rl|{L!r}|{R!r}"

    def mode(self, q):
        v = self.u[q] - self.e[q]
        return AffineMode(f"{self._key}|{v!r}", [[-self.R / self.L]], [v / self.L], label=str(q))

    def gate_schedule(self, t0, t1):
        ev = []
        for k in range(1, len(self.u)):
            tk = k * self.Ts
            if t0 < tk < t1:
                ev.append((tk, lambda q, k=k: k))
        return ev

    def outputs(self, q):
        return {"i": np.array([1.0, 0.0]), "u": np.array([0.0, self.u[q]])}

    def stored_energy(self):
        return np.diag([self.L, 0.0])

    def powers(self, q):
        e_i = np.array([1.0, 0.0])
        return {"p_conv": sym_linear(e_i, self.u[q]), "p_emf": sym_linear(e_i, self.e[q]), "p_R": self.R * np.outer(e_i, e_i)}


def step_metrics(t, i, t_step, i_from, i_to, band=0.02):
    """Overshoot (fraction of the step), 2 % settling time after the step (None if not settled in the window)."""
    post = t >= t_step - 1e-15
    tp, ip = t[post], i[post]
    d = i_to - i_from
    if abs(d) < 1e-12:
        return {"overshoot": 0.0, "settle": 0.0, "peak": float(np.max(ip)), "final": float(ip[-1])}
    sgn = 1.0 if d > 0 else -1.0
    peak = float(np.max(sgn * ip)) * sgn
    ov = max(0.0, (sgn * (peak - i_to)) / abs(d))
    out = np.where(np.abs(ip - i_to) > band * abs(d))[0]
    if not out.size:
        settle = 0.0
    elif out[-1] + 1 < tp.size:
        settle = float(tp[out[-1] + 1] - t_step)
    else:
        settle = None
    return {"overshoot": ov, "settle": settle, "peak": peak, "final": float(ip[-1])}


def fmt_ms(x) -> str:
    return "창 안에서 정정 안 됨" if x is None else f"{x * 1e3:.3g} ms"


# ======================================================================================
# Experiment 1: PI design with units (A) + sampled-data margins
# ======================================================================================

OUT_CHOICES = [
    ("V", "전압 출력 PI (Kp [V/A], Ki [V/(A·s)])"),
    ("duty", "duty 출력 PI, K_PWM으로 환산 (Kp [1/A])"),
    ("duty_raw", "duty 출력인데 V/A 숫자를 그대로 사용 (단위 오류)"),
]


def _design_params(extra=None):
    ps = [
        Param("L", "플랜트 인덕턴스 L (설계값)", "H", 0.8e-3, "mH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="교재 09장 0.8 mH"),
        Param("R", "플랜트 저항 R", "Ω", 0.1, "Ω", vmin=1e-4, vmax=100.0, source="TEXTBOOK", source_note="0.1 Ω"),
        Param("fc", "목표 crossover f_c", "Hz", 1000.0, "Hz", vmin=1.0, vmax=1e5, source="TEXTBOOK", source_note="1 kHz (초기값이지 모든 PFC의 권장값이 아님)"),
        Param("fs", "샘플링 주파수 f_samp", "Hz", 40e3, "kHz", vmin=1e3, vmax=1e6, source="ASSUMED", source_note="40 kHz: 1 샘플 연산지연 + ZOH = 1.5·T_s = 37.5 µs (교재의 전체 지연과 같게 선택)", group="디지털"),
        Param("n_delay", "연산 지연 (샘플 수)", "", 1, "", vmin=0, vmax=4, kind="int", source="ASSUMED", source_note="k번째 샘플로 계산한 전압을 k+n_d에서 인가", group="디지털"),
        Param("K_pwm", "변조기 이득 K_PWM", "V", 800.0, "V", vmin=1.0, vmax=5000.0, source="ASSUMED", source_note="half-bridge 평균 출력 u = (d − ½)·V_dc, V_dc = 800 V → 800 V/duty", group="디지털"),
        Param("out", "제어기 출력 단위", "", "V", kind="choice", choices=OUT_CHOICES, source="TEXTBOOK", source_note="교재: 출력이 duty면 V/duty 이득을 포함해 환산", group="디지털"),
        Param("L_ratio", "실제 L / 설계 L", "", 1.0, "", vmin=0.1, vmax=5.0, source="ASSUMED", source_note="전류에 따른 L 감소 등 (E07): 1이면 정확한 소거", group="불확도"),
    ]
    return ps + (extra or [])


def _gains(v):
    """Controller gains in controller units and the actuator gain K_out."""
    Kp, Ki = v["L"] * 2 * math.pi * v["fc"], v["R"] * 2 * math.pi * v["fc"]
    if v["out"] == "V":
        return Kp, Ki, 1.0
    if v["out"] == "duty":
        return Kp / v["K_pwm"], Ki / v["K_pwm"], v["K_pwm"]
    return Kp, Ki, v["K_pwm"]  # duty output with unconverted V/A numbers (unit error)


def run_pi_design(v: dict) -> Result:
    res = Result("FL06", "pi_design", "A (+ 샘플링 루프 B)")
    L, R, fc, fs = v["L"], v["R"], v["fc"], v["fs"]
    Ts = 1.0 / fs
    nd = int(v["n_delay"])
    Lp = L * v["L_ratio"]  # actual plant inductance
    Kp, Ki, Kout = _gains(v)
    Td = (nd + 0.5) * Ts  # continuous-equivalent delay: computation + ZOH half sample
    tb = abs(L - 0.8e-3) < 1e-12 and abs(R - 0.1) < 1e-12 and abs(fc - 1000) < 1e-9
    KpV, KiV = L * 2 * math.pi * fc, R * 2 * math.pi * fc
    rp, ri = ref.pi_gains_rl(L, R, fc)
    res.add_metric("Kp_V", "Kp (전압 출력)", KpV, "V/A", ref=5.026548 if tb else rp, ref_label="교재 5.026548 V/A" if tb else "L·ω_c", tol=1e-6, basis="u [V] = Kp·e [A] + x_I [V]")
    res.add_metric("Ki_V", "Ki (전압 출력)", KiV, "V/(A·s)", ref=628.318531 if tb else ri, ref_label="교재 628.318531 V/(A·s)" if tb else "R·ω_c", tol=1e-6, basis="PI 영점 Ki/Kp = R/L = plant 극점")
    rpd, rid = ref.duty_gains(rp, ri, v["K_pwm"])
    res.add_metric("Kp_d", "Kp (duty 출력, K_PWM 환산)", KpV / v["K_pwm"], "1/A", ref=rpd, ref_label="Kp / K_PWM", tol=1e-9, basis=f"K_PWM = {v['K_pwm']:g} V/duty")
    res.add_metric("Ki_d", "Ki (duty 출력, K_PWM 환산)", KiV / v["K_pwm"], "1/(A·s)", ref=rid, ref_label="Ki / K_PWM", tol=1e-9)
    res.add_metric("Td", "전체 지연 T_d (연산 n_d·T_s + ZOH T_s/2)", Td, "s", ref=37.5e-6 if (abs(fs - 40e3) < 1e-6 and nd == 1) else None, ref_label="교재 37.5 µs", tol=1e-9, basis="연속 근사 exp(−sT_d)에 쓰는 등가 지연")
    lag = math.degrees(2 * math.pi * fc * Td)
    res.add_metric("lag_fc", "f_c에서 지연 위상", lag, "deg", ref=13.5 if (tb and abs(Td - 37.5e-6) < 1e-12) else ref.delay_phase_deg(fc, Td), ref_label="교재 13.5°" if (tb and abs(Td - 37.5e-6) < 1e-12) else "360·f_c·T_d", tol=1e-6)
    # numerical crossover and margins on the full complex loop gains
    f_hi_c = max(100 * fc, 10 * fs)
    m0 = margins(lambda f: loop_gain_cont(f, Kp, Ki, Lp, R, 0.0, Kout), fc / 1000, f_hi_c)
    md = margins(lambda f: loop_gain_cont(f, Kp, Ki, Lp, R, Td, Kout), fc / 1000, f_hi_c)
    mz = margins(lambda f: loop_gain_disc(f, Kp, Ki, Lp, R, Ts, nd, Kout), min(fc, fs) / 1000, 0.4999 * fs)
    poles = closed_loop_poles(Kp, Ki, Lp, R, Ts, nd, Kout)
    rho = float(np.max(np.abs(poles)))
    if m0.fc is not None:
        res.add_metric("fc_num", "crossover (연속, 지연 없음, 수치 탐색)", m0.fc, "Hz", ref=fc if (v["out"] != "duty_raw" and abs(v["L_ratio"] - 1) < 1e-12) else None, ref_label="설계 목표 f_c", tol=1e-6, basis="|L(j2πf)| = 1 근 찾기")
        res.add_metric("pm0", "위상여유 (연속, 지연 없음)", m0.pm, "deg", basis="PI 영점이 plant 극점을 소거하면 90°")
    if md.pm is not None:
        res.add_metric("pm_delay", "위상여유 (연속 + exp(−sT_d))", md.pm, "deg", ref=ref.pm_cancelled_pi_deg(fc, Td) if (v["out"] != "duty_raw" and abs(v["L_ratio"] - 1) < 1e-12) else None, ref_label="90° − 360·f_c·T_d", tol=1e-6)
    if mz.fc is not None:
        res.add_metric("fc_disc", "crossover (샘플링 루프, 정확)", mz.fc, "Hz", basis="ZOH plant + 순방향 Euler PI + z^−n_d")
        res.add_metric("pm_disc", "위상여유 (샘플링 루프, 정확)", mz.pm, "deg", note="연속 근사와 다르다: ZOH·적분기 이산화·지연을 모두 포함")
    else:
        res.add_metric("fc_disc", "crossover (샘플링 루프)", "나이퀴스트 주파수 아래에 없음", "", note="이득이 너무 크거나 작아 |L| = 1 교차가 f_samp/2 아래에 없다")
    if mz.gm_db is not None:
        res.add_metric("gm_disc", "이득여유 (샘플링 루프)", mz.gm_db, "dB", basis=f"위상 −180° 주파수 {mz.f180 / 1e3:.4g} kHz")
    res.add_metric("rho", "폐루프 극점 최대 |z| (샘플링 루프)", rho, "", basis="< 1 이면 선형 범위에서 안정",
                   note="≈ e^(−R·T_s/L)인 느린 극점은 PI 영점이 거의 소거한 plant 극점이다: 기준 step에는 안 보이지만 외란·포화 응답에는 L/R 꼬리로 나타난다 (실험 2).")
    # independent checks
    if v["out"] != "duty_raw" and abs(v["L_ratio"] - 1) < 1e-12 and m0.fc is not None:
        res.add_check(check_close("crossover: 복소 루프이득 근 찾기 vs 설계 목표 f_c", m0.fc, fc, 1e-8, "Kp = Lω_c, Ki = Rω_c로 만든 PI와 plant 1/(Ls+R)의 전체 복소 이득에서 |L| = 1 을 brentq로 탐색 (게인 공식을 다시 쓰지 않음)", True, "Hz"))
        res.add_check(check_close("위상여유: 수치 (지연 포함) vs 90° − 360 f_c T_d", md.pm, ref.pm_cancelled_pi_deg(fc, Td), 1e-8, "unwrap한 위상에서 f_c의 정확한 위상 vs 손계산", True, "deg", abs_scale=90.0))
    if v["out"] == "duty":
        mV = margins(lambda f: loop_gain_disc(f, KpV, KiV, Lp, R, Ts, nd, 1.0), min(fc, fs) / 1000, 0.4999 * fs)
        if mV.fc is not None and mz.fc is not None:
            res.add_check(check_close("duty 출력(환산) 루프 = 전압 출력 루프", mz.fc, mV.fc, 1e-9, "duty 출력 PI × K_PWM vs 전압 출력 PI의 샘플링 루프 crossover", True, "Hz"))
    # sampled step response (small signal, no saturation) vs ideal first-order
    n = int(min(max(12 / (2 * math.pi * fc) / Ts, 80), 2000))
    pi = PIController(Kp, Ki, Ts, 1e9, Kout, 0.0, False)
    run = run_current_loop(Lp, R, Ts, pi, n, lambda k: 1.0 if k >= 1 else 0.0, lambda k: 0.0, n_delay=nd)
    tt = run.t
    t_step = Ts
    ideal = [ref.first_order_step(t - t_step, fc) for t in tt]
    stable = rho < 1.0 - 1e-12
    sm = step_metrics(tt, run.i, t_step, 0.0, 1.0) if stable else None
    if sm:
        res.add_metric("overshoot", "소신호 step overshoot (샘플 순간)", sm["overshoot"] * 100, "%", basis="1 A step, 포화 없음")
        res.add_metric("settle", "2 % 정정시간", sm["settle"], "s")
        res.add_metric("tau_ideal", "이상 1차 응답 시정수 1/ω_c", 1 / (2 * math.pi * fc), "s", basis="지연 없는 연속 설계의 기대치")
    res.add_series("step_i", "i (샘플링 루프, 샘플 순간)", "A", (tt - t_step).tolist(), [float(x) for x in run.i], style="points")
    res.add_series("step_ideal", "이상 1차 응답 1 − e^(−ω_c t)", "A", (tt - t_step).tolist(), ideal, dash=True)
    res.add_plot("p_step", "전류 루프 소신호 step 응답", ["step_i", "step_ideal"], x_label="step 이후 시간", y_label="i", y_unit="A", level="B (샘플링)",
                 hlines=[{"y": 1.0, "label": "기준 1 A"}],
                 proved="샘플링·연산지연·ZOH를 포함한 루프의 step 응답을 샘플 순간에서 계산해, 지연 없는 연속 설계(1차 응답)와 얼마나 다른지 보였다.",
                 not_yet="PWM 리플·전류 센서 필터·dead time·포화는 이 그래프에 없다(실험 2에서 포화). 샘플 사이 전류는 실험 2에서 연속 엔진으로 확인한다.")
    # Bode data
    fgrid = np.geomspace(max(fc / 100, 1.0), 0.499 * fs, 500)
    Lc0 = loop_gain_cont(fgrid, Kp, Ki, Lp, R, 0.0, Kout)
    Lcd = loop_gain_cont(fgrid, Kp, Ki, Lp, R, Td, Kout)
    Lz = loop_gain_disc(fgrid, Kp, Ki, Lp, R, Ts, nd, Kout)
    res.add_series("mag_c", "|L| 연속 (지연 유무 동일)", "dB", fgrid.tolist(), (20 * np.log10(np.abs(Lc0))).tolist())
    res.add_series("mag_z", "|L| 샘플링 루프 (정확)", "dB", fgrid.tolist(), (20 * np.log10(np.abs(Lz))).tolist(), dash=True)
    res.add_series("ph_c0", "∠L 연속, 지연 없음", "deg", fgrid.tolist(), np.degrees(np.unwrap(np.angle(Lc0))).tolist())
    res.add_series("ph_cd", "∠L 연속 × exp(−sT_d)", "deg", fgrid.tolist(), np.degrees(np.unwrap(np.angle(Lcd))).tolist())
    res.add_series("ph_z", "∠L 샘플링 루프 (정확)", "deg", fgrid.tolist(), np.degrees(np.unwrap(np.angle(Lz))).tolist(), dash=True)
    vl = [{"x": fc, "label": f"설계 f_c {fc / 1e3:g} kHz"}]
    mk_m = [{"x": mz.fc, "y": 0.0, "label": f"샘플링 crossover {mz.fc:.4g} Hz"}] if mz.fc else []
    mk_p = [{"x": mz.fc, "y": mz.pm - 180.0, "label": f"PM {mz.pm:.3g}°"}] if mz.fc else []
    res.add_plot("p_mag", "루프이득 크기", ["mag_c", "mag_z"], x_label="f", x_unit="Hz", y_label="|L|", y_unit="dB", kind="xy", log_x=True, group="bode", hlines=[{"y": 0.0, "label": "0 dB"}], vlines=vl, markers=mk_m, level="A/B",
                 proved="PI 영점이 plant 극점을 소거하면 |L| = ω_c/ω인 −20 dB/dec 직선이 되어 설계 f_c에서 0 dB를 지난다. 단위 오류(duty에 V/A 숫자)는 이 직선을 K_PWM배 위로 올린다.",
                 not_yet="L이 전류에 따라 줄거나 센서 필터 극점이 있으면 소거가 정확하지 않다(L 비율 입력으로 일부 확인). 식별하지 않은 공진은 없다고 가정했다.")
    res.add_plot("p_phase", "루프이득 위상: 지연이 crossover에서 위상을 소비한다", ["ph_c0", "ph_cd", "ph_z"], x_label="f", x_unit="Hz", y_label="∠L", y_unit="deg", kind="xy", log_x=True, group="bode",
                 hlines=[{"y": -180.0, "label": "−180°"}], vlines=vl, markers=mk_p, level="A/B",
                 proved=f"지연이 없으면 위상은 −90°로 평평하고, T_d = {Td * 1e6:.4g} µs가 f_c에서 {lag:.3g}°를 소비한다. 샘플링 루프의 정확한 위상은 연속 근사와 따로 계산했다.",
                 not_yet="샘플링 루프 위상은 f_samp/2까지만 의미가 있다. 실측 루프이득(주입 측정)과의 비교는 하지 않았다.")
    res.tables.append(
        Table(
            "t_units",
            "같은 루프, 다른 출력 단위",
            ["출력", "Kp", "Ki", "x_I 단위", "실제 인가 전압"],
            [
                ["전압 [V]", f"{KpV:.6g} V/A", f"{KiV:.6g} V/(A·s)", "V", "u = y"],
                ["duty (환산)", f"{KpV / v['K_pwm']:.6g} 1/A", f"{KiV / v['K_pwm']:.6g} 1/(A·s)", "무차원", f"u = K_PWM·y = {v['K_pwm']:g}·y"],
                ["duty (단위 오류)", f"{KpV:.6g} 1/A (!)", f"{KiV:.6g} 1/(A·s) (!)", "무차원", f"루프이득이 {v['K_pwm']:g}배"],
            ],
            note="숫자만 같은 PI를 다른 plant·다른 출력 단위에 붙이지 않는다. 변조기 이득 K_PWM은 V_dc와 변조 방식(half-bridge, SPWM, SVPWM)에 따라 다르다.",
        )
    )
    res.circuit = {"diagram": rl_loop_circuit(v, show_sat=False).to_json(), "intervals": [], "plot_group": ""}
    if not stable:
        res.verdict("UNSTABLE", f"샘플링 폐루프 극점 |z|max = {rho:.4g} > 1: " + ("duty 출력에 V/A 숫자를 그대로 써서 루프이득이 K_PWM배가 되었다." if v["out"] == "duty_raw" else "지연·crossover·plant 오차 조합이 안정 한계를 넘었다."))
    elif mz.pm is not None and mz.pm < PM_LEARNING_MIN:
        res.verdict("MARGINAL", f"안정하지만 샘플링 루프 위상여유 {mz.pm:.3g}° < {PM_LEARNING_MIN:g}° (학습용 기준)")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"단위가 맞는 PI로 샘플링 루프가 안정 (위상여유 {mz.pm:.3g}°, 선형·이상 변조기 가정)" if mz.pm is not None else "안정")
    res.assumptions += [
        "plant: L di/dt = u − R i − e, e는 피드포워드로 상쇄된다고 보고 루프 분석에서 제외",
        "변조기는 이상 평균 전압원 u = K_out·y (PWM 리플·dead time·전압 오차 없음)",
        f"샘플링 루프: 순방향 Euler PI, 연산 지연 {nd} 샘플, ZOH (T_d 연속 근사 = {Td * 1e6:.4g} µs)",
    ]
    res.not_valid_for += ["포화·anti-windup 동작 (실험 2)", "센서 필터·PWM 리플·식별하지 않은 공진이 있는 실제 루프의 위상여유", "입력필터·CPL과의 전체 시스템 안정성 (EX07)"]
    res.interpretation = (
        f"PI 영점(Ki/Kp = R/L)이 plant 극점을 소거하면 루프이득은 ω_c/s가 되어 f_c = {fc:g} Hz에서 정확히 교차하고 위상여유는 90°다. "
        f"실제 디지털 루프에는 연산 지연과 ZOH가 있어 T_d ≈ {Td * 1e6:.4g} µs가 f_c에서 {lag:.3g}°를 먹는다. "
        "제어기 출력이 duty라면 같은 루프를 만들기 위해 이득을 K_PWM(V/duty)로 나눠야 하고, 이를 빠뜨리면 루프이득이 K_PWM배가 되어 샘플링 루프가 발산한다."
    )
    return res


# ======================================================================================
# Experiment 2: discrete current loop, saturation and anti-windup (B)
# ======================================================================================

SCENARIOS = {
    "load_step": "부하 20→80 % (전류 기준 step)",
    "input_dip": "입력(계통) 전압 −10 % step",
    "current_limit": "전류 제한 50 % (80 % 요청)",
    "sensor_offset": "전류 센서 offset",
    "extra_delay": "계산 한 주기 추가 지연",
}
ZERO_CHOICES = [("cancel", "교재 1차 설계: 영점 = plant 극점 (Ki = R·ω_c)"), ("wc5", "외란 억제형: 영점 = ω_c/5 (Ki = Kp·ω_c/5)")]
T_WIN = 10e-3  # analysed window after t = 0 (step at 1 ms)


def _loop_params():
    return [
        Param("L", "플랜트 인덕턴스 L", "H", 0.8e-3, "mH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="0.8 mH"),
        Param("R", "플랜트 저항 R", "Ω", 0.1, "Ω", vmin=1e-4, vmax=100.0, source="TEXTBOOK", source_note="0.1 Ω"),
        Param("fc", "목표 crossover f_c", "Hz", 1000.0, "Hz", vmin=10.0, vmax=1e5, source="TEXTBOOK", source_note="1 kHz"),
        Param("zero", "PI 영점 위치", "", "cancel", kind="choice", choices=ZERO_CHOICES, source="TEXTBOOK", source_note="교재 1차 설계는 극점 소거; 다른 선택은 ASSUMED 비교안", group="디지털"),
        Param("fs", "샘플링 주파수", "Hz", 40e3, "kHz", vmin=1e3, vmax=1e6, source="ASSUMED", source_note="1 샘플 지연 + ZOH = 37.5 µs", group="디지털"),
        Param("e0", "역기전력/계통 d축 전압 e", "V", 400 * math.sqrt(2) / math.sqrt(3), "V", vmin=0.0, vmax=2000.0, source="TEXTBOOK", source_note="400 V LL 계통의 상전압 peak 326.6 V (d축, 진폭불변)", group="회로"),
        Param("V_max", "전압 한계 |u| ≤ V_max", "V", 600 / math.sqrt(3), "V", vmin=1.0, vmax=5000.0, source="ASSUMED", source_note="V_dc = 600 V의 SVPWM 선형 한계 V_dc/√3 = 346.4 V (여유가 작은 합성 설계)", group="회로"),
        Param("I_rated", "정격 전류 (peak)", "A", 25.0, "A", vmin=0.1, vmax=5000.0, source="ASSUMED", source_note="20→80 % step의 기준", group="회로"),
        Param("Kaw_ratio", "anti-windup 이득 K_aw·T_i (T_i = Kp/Ki)", "", 1.0, "", vmin=0.0, vmax=200.0, source="ASSUMED", source_note="1이면 K_aw = Ki/Kp (고전적 선택), 0이면 anti-windup 없음", group="디지털"),
        Param("scenario", "시나리오", "", "load_step", kind="choice", choices=list(SCENARIOS.items()), source="TEXTBOOK", source_note="교재 09장 실험 목록", group="시나리오"),
        Param("offset", "센서 offset (시나리오 선택 시)", "A", 0.5, "A", vmin=-100.0, vmax=100.0, source="ASSUMED", source_note="정격의 2 %", group="시나리오"),
        Param("e_ff", "e 피드포워드", "", True, kind="bool", source="TEXTBOOK", source_note="교재: e feedforward가 적절하다는 전제의 1차 설계", group="디지털"),
    ]


def _pi_gains(v):
    wc = 2 * math.pi * v["fc"]
    Kp = v["L"] * wc
    Ki = v["R"] * wc if v["zero"] == "cancel" else Kp * wc / 5.0
    return Kp, Ki


def _scenario(v, name=None):
    """Signals for one scenario: (r_req(k), e(k), offset(k), n_delay, i_lim, k_step, i_from, i_to)."""
    name = name or v["scenario"]
    Ts = 1 / v["fs"]
    k0 = int(round(1e-3 / Ts))  # event at 1 ms
    Ir = v["I_rated"]
    lo, hi = 0.2 * Ir, 0.8 * Ir
    e0 = v["e0"]
    nd, lim = 1, math.inf
    r = lambda k: hi if k >= k0 else lo  # noqa: E731
    e = lambda k: e0  # noqa: E731
    off = lambda k: 0.0  # noqa: E731
    i_from, i_to = lo, hi
    if name == "input_dip":
        r = lambda k: hi  # noqa: E731
        e = lambda k: 0.9 * e0 if k >= k0 else e0  # noqa: E731
        i_from = i_to = hi
    elif name == "current_limit":
        lim = 0.5 * Ir
        i_to = lim
    elif name == "sensor_offset":
        r = lambda k: hi  # noqa: E731
        off_v = v["offset"]
        off = lambda k: off_v if k >= k0 else 0.0  # noqa: E731
        i_from, i_to = hi, hi - off_v
    elif name == "extra_delay":
        nd = 2
    return r, e, off, nd, lim, k0, i_from, i_to


def _run_scenario(v, name, aw, n):
    L, R, fs = v["L"], v["R"], v["fs"]
    Ts = 1 / fs
    Kp, Ki = _pi_gains(v)
    r, e, off, nd, lim, k0, i_from, i_to = _scenario(v, name)
    pi = PIController(Kp, Ki, Ts, v["V_max"], 1.0, v["Kaw_ratio"] * Ki / Kp, aw and v["Kaw_ratio"] > 0)
    i0 = min(r(0), lim)
    u0 = e(0) + R * i0
    pi.xI = (R * i0) if v["e_ff"] else u0
    a, b = rl_zoh(L, R, Ts)
    tt = np.arange(n) * Ts
    i = np.zeros(n + 1)
    i[0] = i0
    pipe = [u0] * nd
    rec = {k: np.zeros(n) for k in ("r", "uu", "uc", "ua", "e", "xI", "m")}
    for k in range(n):
        rk = min(r(k), lim)
        m = i[k] + off(k)
        ek = e(k)
        y_un, y_sat = pi.step(rk - m, ek if v["e_ff"] else 0.0)
        pipe.append(y_sat)
        ua = pipe.pop(0)
        i[k + 1] = a * i[k] + b * (ua - ek)
        for key, val in (("r", rk), ("uu", y_un), ("uc", y_sat), ("ua", ua), ("e", ek), ("xI", pi.xI), ("m", m)):
            rec[key][k] = val
    t_step = k0 * Ts
    sm = step_metrics(tt, i[:n], t_step, i_from, i_to)
    sat = np.abs(rec["uu"]) > v["V_max"] + 1e-9
    sat_time = float(np.sum(sat)) * Ts
    last_sat = float(tt[np.where(sat)[0][-1]] + Ts) if sat.any() else None
    rec_t = None
    band = 0.02 * abs(i_to - i_from) if abs(i_to - i_from) > 1e-12 else 0.02 * v["I_rated"]
    if last_sat is not None:
        post = (tt >= last_sat) & (np.abs(i[:n] - i_to) > band)
        idx = np.where(post)[0]
        rec_t = (float(tt[idx[-1]] + Ts - last_sat) if idx[-1] + 1 < n else None) if idx.size else 0.0
    after = tt >= t_step
    dev = float(np.max(np.abs(i[:n][after] - i_to)))
    # disturbance recovery (input dip / offset): time to return within 2 % of rated
    out = np.where(after & (np.abs(i[:n] - i_to) > 0.02 * v["I_rated"]))[0]
    dist_rec = (float(tt[out[-1]] + Ts - t_step) if out[-1] + 1 < n else None) if out.size else 0.0
    return {"t": tt, "i": i[:n], "rec": rec, "sm": sm, "sat_time": sat_time, "recover": rec_t, "dev": dev, "dist_rec": dist_rec, "i_to": i_to, "i_from": i_from, "lim": lim, "k0": k0, "nd": nd, "t_step": t_step, "had_sat": last_sat is not None}


def run_discrete_loop(v: dict) -> Result:
    res = Result("FL06", "discrete_loop", "B (정확 ZOH plant + 샘플링 PI)")
    L, R, fs = v["L"], v["R"], v["fs"]
    Ts = 1 / fs
    n = int(round(T_WIN / Ts))
    name = v["scenario"]
    A = _run_scenario(v, name, True, n)
    N = _run_scenario(v, name, False, n)
    t = A["t"]
    t_step = A["t_step"]
    Kp, Ki = _pi_gains(v)
    tb = abs(L - 0.8e-3) < 1e-12 and abs(R - 0.1) < 1e-12 and abs(v["fc"] - 1000) < 1e-9 and v["zero"] == "cancel"
    res.add_metric("Kp", "Kp (전압 출력)", Kp, "V/A", ref=5.026548 if tb else None, ref_label="교재 5.026548 V/A", tol=1e-6)
    res.add_metric("Ki", "Ki (전압 출력)", Ki, "V/(A·s)", ref=628.318531 if tb else None, ref_label="교재 628.318531 V/(A·s)", tol=1e-6, basis="PI 영점 " + ("= plant 극점 R/L" if v["zero"] == "cancel" else "= ω_c/5"))
    res.add_metric("Kaw", "anti-windup 이득 K_aw", v["Kaw_ratio"] * Ki / Kp, "1/s", basis="x_I [V]의 back-calculation 이득 (K_aw·T_i = " + f"{v['Kaw_ratio']:g})")
    step_like = name in ("load_step", "current_limit", "extra_delay")
    if step_like:
        res.add_metric("ov_aw", "overshoot (anti-windup)", A["sm"]["overshoot"] * 100, "%", basis="step 크기 대비, 샘플 순간")
        res.add_metric("ov_noaw", "overshoot (anti-windup 없음)", N["sm"]["overshoot"] * 100, "%")
        res.add_metric("ts_aw", "2 % 정정시간 (anti-windup)", A["sm"]["settle"] if A["sm"]["settle"] is not None else fmt_ms(None), "s" if A["sm"]["settle"] is not None else "", basis=f"분석 창 {T_WIN * 1e3:g} ms")
        res.add_metric("ts_noaw", "2 % 정정시간 (anti-windup 없음)", N["sm"]["settle"] if N["sm"]["settle"] is not None else fmt_ms(None), "s" if N["sm"]["settle"] is not None else "")
    res.add_metric("peak_aw", "최대 전류 (anti-windup)", float(np.max(A["i"])), "A")
    res.add_metric("peak_noaw", "최대 전류 (anti-windup 없음)", float(np.max(N["i"])), "A")
    if math.isfinite(A["lim"]):
        viol = float(np.max(A["i"])) - A["lim"]
        res.add_metric("lim_viol", "전류 제한 초과량 (anti-windup)", max(viol, 0.0), "A", basis=f"제한 {A['lim']:.4g} A는 기준값에 적용 — 루프 overshoot는 막지 못한다")
        res.add_metric("lim_viol_noaw", "전류 제한 초과량 (anti-windup 없음)", max(float(np.max(N["i"])) - N["lim"], 0.0), "A")
    if name in ("input_dip", "sensor_offset"):
        res.add_metric("dev_aw", "최대 전류 편차", A["dev"], "A", basis="사건 뒤, 목표 대비")
        res.add_metric("drec_aw", "정격 2 % 이내 복귀 시간", A["dist_rec"] if A["dist_rec"] is not None else fmt_ms(None), "s" if A["dist_rec"] is not None else "")
    if name == "sensor_offset":
        err = float(np.mean(A["i"][-40:])) - float(np.mean(A["rec"]["r"][-40:]))
        res.add_metric("ss_err", "정상상태 실제 전류 오차 (실제 − 기준)", err, "A", ref=-v["offset"], ref_label="−offset (루프가 측정값을 맞추므로)", tol=1e-3, abs_scale=v["I_rated"])
    res.add_metric("sat_time", "전압 포화 시간", A["sat_time"], "s", basis="|u_unsat| > V_max 인 샘플 × T_s")
    if A["had_sat"]:
        res.add_metric("rec_aw", "포화 해제 후 2 % 복귀 (anti-windup)", A["recover"] if A["recover"] is not None else fmt_ms(None), "s" if A["recover"] is not None else "")
    if N["had_sat"]:
        res.add_metric("rec_noaw", "포화 해제 후 2 % 복귀 (anti-windup 없음)", N["recover"] if N["recover"] is not None else fmt_ms(None), "s" if N["recover"] is not None else "", note="적분기가 포화 동안 쌓은 값을 느린 극점(L/R)으로 되돌리는 시간")
    # --- independent check: exact continuous plant with the engine, same applied voltages
    ncheck = min(n, 240)
    plant = RLPlant(L, R, Ts, A["rec"]["ua"][:ncheck], A["rec"]["e"][:ncheck])
    tr = simulate(plant, 0, [A["i"][0]], 0.0, ncheck * Ts)
    err = 0.0
    for k in range(1, ncheck):
        zk, _ = tr.state_at(k * Ts)
        err = max(err, abs(zk[0] - A["i"][k]))
    res.add_check(
        Check(
            "ZOH 이산 map vs 연속 plant 정확 적분 (엔진)",
            "PASS" if err < 1e-9 * max(1.0, v["I_rated"]) else "FAIL",
            err,
            "A",
            1e-9 * max(1.0, v["I_rated"]),
            path="i[k+1] = a·i[k] + b·(u−e) (a = e^(−RT_s/L)) vs 같은 인가전압을 샘플마다 바꾸는 RL 회로의 행렬지수 해",
            independent=True,
            detail=f"{ncheck}샘플 최대 차이 {err:.3g} A",
        )
    )
    led = energy_ledger(tr, plant, 0.0, ncheck * Ts, ["p_conv"], ["p_emf"], ["p_R"], rated_power=v["e0"] * v["I_rated"])
    res.add_check(ledger_check(led, what="컨버터 전압 u가 한 일 = e로 전달 + R 손실 + ½Li² 변화: "))
    smp = tr.sample(["i"], per_segment=6)
    res.add_series("i_cont", "i (연속 plant, 엔진)", "A", [x - t_step for x in smp["t"]], smp["i"])
    ts_ = (t - t_step).tolist()
    res.add_series("i_aw", "i (anti-windup, 샘플)", "A", ts_, A["i"].tolist(), style="points")
    res.add_series("i_noaw", "i (anti-windup 없음)", "A", ts_, N["i"].tolist())
    res.add_series("r", "기준 (제한 후)", "A", ts_, A["rec"]["r"].tolist(), dash=True)
    res.add_series("uu", "u 계산값 (포화 전)", "V", ts_, A["rec"]["uu"].tolist())
    res.add_series("ua", "u 인가 (포화·지연 후)", "V", ts_, A["rec"]["ua"].tolist())
    res.add_series("uu_n", "u 계산값 (anti-windup 없음)", "V", ts_, N["rec"]["uu"].tolist(), dash=True)
    res.add_series("xI_aw", "x_I (anti-windup)", "V", ts_, A["rec"]["xI"].tolist())
    res.add_series("xI_noaw", "x_I (anti-windup 없음)", "V", ts_, N["rec"]["xI"].tolist(), dash=True)
    bands = _sat_bands(t - t_step, np.abs(A["rec"]["uu"]) > v["V_max"] + 1e-9, Ts)
    hl = [{"y": A["lim"], "label": "전류 제한"}] if math.isfinite(A["lim"]) else []
    res.add_plot("p_i", f"전류 응답: {SCENARIOS[name]}", ["i_cont", "i_aw", "i_noaw", "r"], x_label="사건 이후 시간", y_label="i", y_unit="A", bands=bands, group="dl", level="B", hlines=hl,
                 proved="같은 사건에서 anti-windup 유무의 차이를 샘플 순간 값과 연속 plant 전류로 함께 보였다. 포화 구간(음영)에서 적분기가 계속 쌓이면 포화가 풀린 뒤 overshoot와 느린 꼬리(L/R 극점)가 생긴다.",
                 not_yet="이상 평균 변조기라 PWM 리플과 전류 센서 필터가 없다. 전류 제한은 기준값에만 걸었으므로 하드웨어 OCP의 동작은 따로 확인해야 한다.")
    res.add_plot("p_u", "전압 명령: 계산값과 실제 인가값", ["uu", "ua", "uu_n"], x_label="사건 이후 시간", y_label="u", y_unit="V", bands=bands, group="dl", level="B", hlines=[{"y": v["V_max"], "label": "V_max"}],
                 proved="포화 동안 인가 전압은 V_max에 묶이고, 계산값과의 차이(u_sat − u_unsat)가 back-calculation 입력이 된다.", not_yet="dq 벡터 포화(원형 한계)는 EX07에서 다룬다.")
    res.add_plot("p_xI", "적분기 상태 x_I [V]", ["xI_aw", "xI_noaw"], x_label="사건 이후 시간", y_label="x_I", y_unit="V", bands=bands, group="dl", level="B",
                 proved="anti-windup이 없으면 포화 동안 x_I가 필요 이상으로 커지고(windup), 극점을 소거한 설계에서는 그 초과분이 느린 극점(L/R)으로만 빠진다. K_aw가 너무 크면 반대로 x_I를 음으로 끌어내려 복귀가 늦다.",
                 not_yet="K_aw 선택 규칙(여기서는 K_aw·T_i)은 설계 선택이며, 조건부 적분(clamping) 등 다른 방식과의 비교는 하지 않았다.")
    rows = []
    for sk, sl in SCENARIOS.items():
        a_ = _run_scenario(v, sk, True, n)
        n_ = _run_scenario(v, sk, False, n)
        if sk == "sensor_offset":
            k1 = float(np.mean(a_["i"][-40:])) - float(np.mean(a_["rec"]["r"][-40:]))
            rows.append([sl, "—", "—", f"정상상태 오차 {k1:+.4g} A", f"{a_['sat_time'] * 1e6:.4g}", "—"])
            continue
        if sk == "input_dip":
            rows.append([sl, f"편차 {a_['dev']:.3g} A, 복귀 {fmt_ms(a_['dist_rec'])}", f"편차 {n_['dev']:.3g} A, 복귀 {fmt_ms(n_['dist_rec'])}", "—", f"{a_['sat_time'] * 1e6:.4g}", "—"])
            continue
        lim_txt = f"{max(float(np.max(a_['i'])) - a_['lim'], 0):.3g} / {max(float(np.max(n_['i'])) - n_['lim'], 0):.3g} A" if math.isfinite(a_["lim"]) else "—"
        rows.append([sl, f"{a_['sm']['overshoot'] * 100:.3g} % / {fmt_ms(a_['sm']['settle'])}", f"{n_['sm']['overshoot'] * 100:.3g} % / {fmt_ms(n_['sm']['settle'])}", lim_txt, f"{a_['sat_time'] * 1e6:.4g}",
                     f"{fmt_ms(a_['recover']) if a_['had_sat'] else '—'} / {fmt_ms(n_['recover']) if n_['had_sat'] else '—'}"])
    res.tables.append(
        Table(
            "t_scen",
            "교재 실험 목록 요약 (같은 PI, 샘플 순간 기준)",
            ["사건", "overshoot / 정정 (AW)", "overshoot / 정정 (AW 없음)", "제한 초과 AW / 없음, 오차", "포화 시간 [µs]", "포화 후 복귀 AW / 없음"],
            rows,
            note="성공 기준(예: overshoot ≤ 20 %, 제한 초과 0)은 학습 목표로 정한 값이지 고객 사양이 아니다. 센서 offset은 루프가 측정값을 맞추므로 실제 전류에 그대로 남는다.",
        )
    )
    res.circuit = {"diagram": rl_loop_circuit(v, show_sat=True).to_json(), "intervals": bands, "plot_group": "dl"}
    worst_ov = A["sm"]["overshoot"] if step_like else 0.0
    if math.isfinite(A["lim"]) and float(np.max(A["i"])) > A["lim"] * 1.02:
        res.verdict("FAIL_CONSTRAINT", f"전류 제한 {A['lim']:.4g} A를 루프 overshoot로 {float(np.max(A['i'])) - A['lim']:.3g} A 넘었다 (제한을 기준값에만 걸었기 때문).")
    elif name == "sensor_offset":
        res.verdict("FAIL_CONSTRAINT", f"센서 offset {v['offset']:g} A만큼 실제 전류가 기준에서 벗어난 채로 유지된다 — PI는 측정값을 맞출 뿐이다.")
    elif step_like and (worst_ov > 0.2 or A["sm"]["settle"] is None):
        res.verdict("MARGINAL", "overshoot가 학습용 기준 20 %를 넘거나 분석 창 안에서 정정되지 않았다.")
    elif name == "input_dip" and A["dist_rec"] is None:
        res.verdict("MARGINAL", f"외란이 {T_WIN * 1e3:g} ms 창 안에서 정격 2 % 이내로 제거되지 않았다: 극점을 소거한 PI의 적분 동작은 느린 극점 L/R로만 외란을 없앤다 (피드포워드가 필요한 이유).")
    else:
        res.verdict("PASS_WITHIN_MODEL", "anti-windup이 있는 샘플링 루프가 이 사건을 모델 범위에서 처리한다 (학습용 기준 overshoot ≤ 20 %, 창 안 정정).")
    res.assumptions += [
        "plant: L di/dt = u − R i − e, e는 샘플 경계에서만 바뀐다 (ZOH 정확 map)",
        f"PI: 순방향 Euler, 연산 지연 {A['nd']} 샘플, |u| ≤ V_max 포화, back-calculation K_aw = {v['Kaw_ratio']:g}·Ki/Kp",
        "이상 평균 변조기(리플·dead time 없음), 센서는 이상(필요 시 offset만)",
    ]
    res.not_valid_for += ["PWM 리플·센서 필터·양자화", "하드웨어 OCP 동작", "dq 벡터 포화·축 간 결합 (EX07)", "외부 전압 루프와의 상호작용"]
    res.interpretation = (
        "전압이 V_max에 묶이면 전류 오차가 오래 남고, anti-windup이 없는 적분기는 그 오차를 계속 적분해 필요 이상의 값을 쌓는다. 포화가 풀리면 쌓인 값이 전류를 기준 위로 밀고, "
        "극점을 소거한 설계에서는 그 초과분이 느린 극점 L/R(여기서 8 ms)으로만 빠져 회복이 늦다. back-calculation은 실제 인가값과 계산값의 차이로 적분기를 되돌리지만 K_aw가 너무 크면 적분기를 반대로 끌어내린다. "
        "전류 제한을 기준값에만 걸면 루프 overshoot만큼 제한을 넘을 수 있고, 센서 offset은 루프가 측정값을 맞추기 때문에 실제 전류 오차로 그대로 남는다."
    )
    return res


def _sat_bands(t, sat, Ts):
    bands = []
    for k in range(len(t)):
        mode = "sat" if sat[k] else "lin"
        if bands and bands[-1]["mode"] == mode:
            bands[-1]["x1"] = float(t[k] + Ts)
        else:
            bands.append({"x0": float(t[k]), "x1": float(t[k] + Ts), "mode": mode, "label": "전압 포화" if mode == "sat" else "선형"})
    return bands


def rl_loop_circuit(v: dict, show_sat: bool) -> Circuit:
    c = Circuit("rl_loop", 780, 330, title="RL plant 전류 루프 (평균 변조기)")
    u = c.add("vsource", "U", 170, 170, 90, "u = K·y", "컨버터 평균전압", lpos=(194, 166, "start"))
    rr = c.add("resistor", "R", 280, 70, 0, "R", f"{v['R']:g} Ω")
    ll = c.add("inductor", "L", 400, 70, 0, "L", f"{v['L'] * 1e3:g} mH")
    ee = c.add("vsource", "E", 510, 170, 90, "e", "역기전력/계통", lpos=(488, 166, "end"))
    c.wire("w1", u["a"], (170, 70), rr["a"])
    c.wire("w2", rr["b"], ll["a"])
    c.wire("w3", ll["b"], (510, 70), ee["a"])
    c.wire("w4", ee["b"], (510, 260), (170, 260), u["b"])
    c.dot((510, 70))
    c.probe("pi", "i_cont", 220, 58, "right", "i")
    c.add("block", "ADC", 660, 70, 0, "ADC: i[k]", w=130, h=34)
    c.add("block", "PI", 660, 150, 0, "PI + 포화 + AW", w=130, h=34)
    c.add("block", "PWM", 660, 230, 0, "n_d·T_s 지연 → u", w=130, h=34)
    c.wire("w_s", (510, 70), (595, 70))
    c.wire("w_s2", (660, 87), (660, 133))
    c.wire("w_s3", (660, 167), (660, 213))
    c.wire("w_s4", (660, 247), (660, 305), (100, 305), (100, 170), (154, 170))
    c.text(380, 292, "제어기 출력 → 변조기 → 인가 전압 (한 샘플 이상 늦게)", "note")
    active = ["U", "R", "L", "E", "w1", "w2", "w3", "w4"]
    c.mode("lin", "선형 (포화 없음)", active + ["ADC", "PI", "PWM", "w_s", "w_s2", "w_s3", "w_s4"], "u = Kp·e + x_I (+ e 피드포워드), 적분기는 오차만 적분")
    c.mode("sat", "전압 포화", active + ["PI"], "u = ±V_max에 묶임: anti-windup이 없으면 x_I가 계속 쌓인다" if show_sat else "u = ±V_max에 묶임", dim=["ADC", "PWM"])
    return c


# ======================================================================================
# Experiment 3: dq current control sign (B), dq model vs independent abc simulation
# ======================================================================================


def park(xa, xb, xc, th):
    """Amplitude-invariant Park: x_dq = (2/3)(x_a + a x_b + a^2 x_c) e^{-j th}."""
    k = 2 * math.pi / 3
    d = 2 / 3 * (xa * math.cos(th) + xb * math.cos(th - k) + xc * math.cos(th + k))
    q = -2 / 3 * (xa * math.sin(th) + xb * math.sin(th - k) + xc * math.sin(th + k))
    return d, q


def dq_plant_map(L, R, w, Ts):
    """Exact map over Ts for L di/dt = v_g - v_c - R i + w L J i (J i = [i_q, -i_d]) with inputs held."""
    A = np.array([[-R / L, w], [-w, -R / L]])
    F = np.zeros((4, 4))
    F[:2, :2] = A
    F[:2, 2:] = np.eye(2) / L  # input: (v_g - v_c) held
    E = expm(F * Ts)
    return E[:2, :2], E[:2, 2:]


@dataclass
class DQController:
    Kp: float
    Ki: float
    Ts: float
    wL: float
    V_max: float
    Kaw: float
    sign: int = +1  # +1: v_c = v_g_ff + decoupling - u (correct); -1: + u (reversed)
    decouple: bool = True
    xd: float = 0.0
    xq: float = 0.0

    def step(self, id_ref, iq_ref, id_m, iq_m, vgd, vgq):
        ed, eq = id_ref - id_m, iq_ref - iq_m
        ud = self.Kp * ed + self.xd
        uq = self.Kp * eq + self.xq
        cd = self.wL * iq_m if self.decouple else 0.0
        cq = -self.wL * id_m if self.decouple else 0.0
        vcd = vgd + cd - self.sign * ud
        vcq = vgq + cq - self.sign * uq
        mag = math.hypot(vcd, vcq)
        sc = min(1.0, self.V_max / mag) if mag > 0 else 1.0
        vcd_s, vcq_s = vcd * sc, vcq * sc
        # anti-windup on the vector actually applied: u implied by the saturated vector
        ud_s = (vgd + cd - vcd_s) * self.sign
        uq_s = (vgq + cq - vcq_s) * self.sign
        self.xd += self.Ts * (self.Ki * ed + self.Kaw * (ud_s - ud))
        self.xq += self.Ts * (self.Ki * eq + self.Kaw * (uq_s - uq))
        return vcd_s, vcq_s, sc < 1.0


def run_dq(v: dict, sign: int, n: int, trip: float):
    L, R, w = v["L"], v["R"], 2 * math.pi * v["f"]
    Ts = 1 / v["fs"]
    Vpk = v["V_LL"] * math.sqrt(2) / math.sqrt(3)
    wc = 2 * math.pi * v["fc"]
    ctl = DQController(L * wc, R * wc, Ts, w * L, v["V_dc"] / math.sqrt(3), wc, sign, bool(v["decouple"]))
    Phi, Gam = dq_plant_map(L, R, w, Ts)
    x = np.zeros(2)
    vc_pipe = [(Vpk, 0.0)]  # converter initially mirrors the grid (zero current)
    k0 = int(round(1e-3 / Ts))
    rec = {k: [] for k in ("t", "id", "iq", "vcd", "vcq", "idr", "iqr", "sat")}
    tripped = None
    for k in range(n):
        idr = v["id_step"] if k >= k0 else 0.0
        iqr = v["iq_ref"] if k >= k0 else 0.0
        vcd, vcq, sat = ctl.step(idr, iqr, x[0], x[1], Vpk, 0.0)
        vc_pipe.append((vcd, vcq))
        va = vc_pipe.pop(0)
        rec["t"].append(k * Ts)
        rec["id"].append(x[0])
        rec["iq"].append(x[1])
        rec["vcd"].append(va[0])
        rec["vcq"].append(va[1])
        rec["idr"].append(idr)
        rec["iqr"].append(iqr)
        rec["sat"].append(sat)
        x = Phi @ x + Gam @ (np.array([Vpk, 0.0]) - np.array(va))
        if tripped is None and math.hypot(x[0], x[1]) > trip:
            tripped = (k + 1) * Ts
            break
    return {k: np.array(val) for k, val in rec.items()}, tripped, ctl


def run_abc(v: dict, sign: int, n: int, trip: float, rtol=1e-10):
    """Independent path: three phase ODEs (DOP853), own Park of measured currents, same controller code."""
    L, R, w = v["L"], v["R"], 2 * math.pi * v["f"]
    Ts = 1 / v["fs"]
    Vpk = v["V_LL"] * math.sqrt(2) / math.sqrt(3)
    wc = 2 * math.pi * v["fc"]
    ctl = DQController(L * wc, R * wc, Ts, w * L, v["V_dc"] / math.sqrt(3), wc, sign, bool(v["decouple"]))
    k2 = 2 * math.pi / 3
    i = np.zeros(3)
    vc_pipe = [(Vpk, 0.0)]
    k0 = int(round(1e-3 / Ts))
    out_t, out_d, out_q, out_p, out_pdq = [], [], [], [], []
    tripped = None
    for k in range(n):
        t0 = k * Ts
        idm, iqm = park(i[0], i[1], i[2], w * t0)
        idr = v["id_step"] if k >= k0 else 0.0
        iqr = v["iq_ref"] if k >= k0 else 0.0
        vcd, vcq, _ = ctl.step(idr, iqr, idm, iqm, Vpk, 0.0)
        vc_pipe.append((vcd, vcq))
        vd, vq = vc_pipe.pop(0)
        out_t.append(t0)
        out_d.append(idm)
        out_q.append(iqm)
        vg = [Vpk * math.cos(w * t0 - j * k2) for j in range(3)]
        out_p.append(sum(vg[j] * i[j] for j in range(3)))
        out_pdq.append(1.5 * Vpk * idm)

        def rhs(t, y, vd=vd, vq=vq):
            th = w * t
            dy = np.empty(3)
            for j in range(3):
                thj = th - j * k2
                vgx = Vpk * math.cos(thj)
                vcx = vd * math.cos(thj) - vq * math.sin(thj)
                dy[j] = (vgx - vcx - R * y[j]) / L
            return dy

        sol = solve_ivp(rhs, (t0, t0 + Ts), i, method="DOP853", rtol=rtol, atol=1e-12 * max(1.0, v["id_step"]))
        i = sol.y[:, -1]
        if tripped is None and max(abs(i)) > trip:
            tripped = (k + 1) * Ts
            break
    return np.array(out_t), np.array(out_d), np.array(out_q), np.array(out_p), np.array(out_pdq), tripped


def dq_closed_loop_rho(v: dict, sign: int) -> float:
    """Spectral radius of the linear sampled dq loop (no saturation), state [i_d, i_q, x_d, x_q, v_c(k-1)]."""
    L, R, w = v["L"], v["R"], 2 * math.pi * v["f"]
    Ts = 1 / v["fs"]
    wc = 2 * math.pi * v["fc"]
    Kp, Ki, wL = L * wc, R * wc, w * L
    dec = 1.0 if v["decouple"] else 0.0
    Phi, Gam = dq_plant_map(L, R, w, Ts)
    # v_c,cmd = -sign*(Kp*(-i) + x) + dec*wL*[i_q, -i_d]   (grid and references set to zero)
    M = np.zeros((6, 6))
    M[0:2, 0:2] = Phi
    M[0:2, 4:6] = -Gam
    M[2, 0] = -Ts * Ki
    M[2, 2] = 1
    M[3, 1] = -Ts * Ki
    M[3, 3] = 1
    Kc = np.zeros((2, 4))
    Kc[0, 0], Kc[1, 1] = sign * Kp, sign * Kp
    Kc[0, 2], Kc[1, 3] = -sign, -sign
    Kc[0, 1] += dec * wL
    Kc[1, 0] += -dec * wL
    M[4:6, 0:4] = Kc
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def run_dq_sign(v: dict) -> Result:
    res = Result("FL06", "dq_sign", "B (dq 평균모델 + 독립 abc 모델)")
    Ts = 1 / v["fs"]
    n = int(round(v["t_sim"] / Ts))
    sign = +1 if v["sign"] == "correct" else -1
    Vpk = v["V_LL"] * math.sqrt(2) / math.sqrt(3)
    trip = 4.0 * max(abs(v["id_step"]), abs(v["iq_ref"]), 1.0)
    rec, tripped, _ = run_dq(v, sign, n, trip)
    ta, da, qa, pa, pdq, trip_abc = run_abc(v, sign, min(n, len(rec["t"]) + 1), trip)
    rho = dq_closed_loop_rho(v, sign)
    m = min(len(ta), len(rec["t"]))
    dev = float(np.max(np.hypot(da[:m] - rec["id"][:m], qa[:m] - rec["iq"][:m]))) if m else math.nan
    scale = max(abs(v["id_step"]), abs(v["iq_ref"]), 1.0)
    res.add_metric("Vgd", "계통 d축 전압 v_gd (상전압 peak)", Vpk, "V", ref=326.5986 if abs(v["V_LL"] - 400) < 1e-9 else None, ref_label="400·√2/√3", tol=1e-6, basis="진폭불변 dq, d축 = 계통전압")
    res.add_metric("rho", "샘플링 dq 루프 폐루프 극점 최대 |z|", rho, "", basis="선형(포화 없음) 해석: < 1 안정")
    settled = True
    if tripped is None:
        k_end = len(rec["t"]) - 1
        dec = bool(v["decouple"])
        res.add_metric("id_final", "최종 i_d", float(rec["id"][k_end]), "A", ref=v["id_step"] if dec else None, ref_label="기준", tol=0.01, abs_scale=scale)
        res.add_metric("iq_final", "최종 i_q", float(rec["iq"][k_end]), "A", ref=v["iq_ref"] if dec else None, ref_label="기준", tol=0.01, abs_scale=scale)
        post = rec["t"] >= 1e-3
        iq_dev = float(np.max(np.abs(rec["iq"][post] - rec["iqr"][post]))) if post.any() else 0.0
        res.add_metric("iq_dev", "i_d step 중 i_q 최대 편차 (교차결합)", iq_dev, "A", basis="디커플링이 있으면 작고, 없으면 ωL·i_d가 q축 외란이 된다")
        settled = abs(rec["id"][k_end] - v["id_step"]) <= 0.01 * scale and abs(rec["iq"][k_end] - v["iq_ref"]) <= 0.01 * scale
        P = 1.5 * (Vpk * rec["id"][k_end])
        Q = 1.5 * (0.0 * rec["id"][k_end] - Vpk * rec["iq"][k_end])
        res.add_metric("P", "유효전력 P = 1.5(v_d i_d + v_q i_q)", P, "W", basis="계통→컨버터 +")
        res.add_metric("Q", "무효전력 Q = 1.5(v_q i_d − v_d i_q)", Q, "var", basis="이 변환에서 +는 유도성(전류 지상) 흡수")
    else:
        res.add_metric("t_trip", "전류가 트립 수준을 넘은 시각 (step 이후)", tripped - 1e-3, "s", basis=f"|i| > {trip:.3g} A에서 시뮬레이션 중단")
    res.add_metric("frame_dev", "dq 모델 vs abc 모델(→dq) 최대 차이", dev, "A", basis="샘플 순간, 같은 제어기 코드")
    res.add_check(
        Check(
            "dq 평균모델 vs 독립 abc 시뮬레이션",
            "PASS" if dev < 1e-6 * max(scale, float(np.max(np.abs(rec["id"][:m]))) if m else scale) else "FAIL",
            dev,
            "A",
            1e-6 * scale,
            path="dq 식(±ωL 교차결합)의 정확 이산 map vs 3상 L di/dt = v_g − v_c − R i를 DOP853으로 적분하고 측정 전류를 Park 변환",
            independent=True,
            detail="교차결합 부호가 틀리면 두 경로가 즉시 어긋난다. 변조기는 샘플 동안 dq 전압을 유지(회전 각도 연속)한다고 가정해 두 모델이 같은 물리 입력을 받는다.",
        )
    )
    if m > 2:
        pm_err = float(np.max(np.abs(pa[:m] - pdq[:m])))
        res.add_check(
            Check(
                "순시전력: Σ v_x·i_x (abc) = 1.5 v_d i_d (dq)",
                "PASS" if pm_err < 1e-6 * max(1.0, Vpk * scale) else "FAIL",
                pm_err,
                "W",
                1e-6 * Vpk * scale,
                path="abc 상전압·상전류 곱의 합 vs 진폭불변 dq 전력식 (v_q = 0)",
                independent=True,
                detail="진폭불변 변환이라 1.5 계수가 필요하다 (전력불변 변환이면 1).",
            )
        )
    tt = (rec["t"] - 1e-3).tolist()
    res.add_series("id", "i_d (dq 모델)", "A", tt, rec["id"].tolist())
    res.add_series("iq", "i_q (dq 모델)", "A", tt, rec["iq"].tolist())
    res.add_series("id_abc", "i_d (abc 모델 → Park)", "A", (ta[:m] - 1e-3).tolist(), da[:m].tolist(), style="points")
    res.add_series("iq_abc", "i_q (abc 모델 → Park)", "A", (ta[:m] - 1e-3).tolist(), qa[:m].tolist(), style="points")
    res.add_series("idr", "i_d 기준", "A", tt, rec["idr"].tolist(), style="step", dash=True)
    res.add_series("vcd", "v_cd 인가", "V", tt, rec["vcd"].tolist(), style="step")
    res.add_series("vcq", "v_cq 인가", "V", tt, rec["vcq"].tolist(), style="step")
    bands = _sat_bands(np.array(tt), rec["sat"].astype(bool), Ts)
    res.add_plot("p_dq", "dq 전류: dq 모델(선)과 독립 abc 모델(점)", ["id", "iq", "id_abc", "iq_abc", "idr"], x_label="step 이후 시간", y_label="전류", y_unit="A", group="dq", level="B", bands=bands,
                 proved="dq 식의 교차결합 부호(+ωL i_q, −ωL i_d)와 제어기 부호를 3상 회로를 직접 적분한 독립 경로로 확인했다. 올바른 부호는 기준을 따라가고 뒤집힌 부호는 발산한다.",
                 not_yet="평형 계통·이상 PLL(θ = ωt)·평균 변조기 가정이다. 불평형·약계통·PLL 동특성에서 같은 주장을 자동 확장하지 않는다.")
    res.add_plot("p_vc", "컨버터 전압 명령 (dq)", ["vcd", "vcq"], x_label="step 이후 시간", y_label="v_c", y_unit="V", group="dq", level="B", bands=bands, hlines=[{"y": v["V_dc"] / math.sqrt(3), "label": "|v| 한계 V_dc/√3"}],
                 proved="전류를 늘리려면 컨버터 전압을 계통전압보다 낮춰야 한다(v_cd < v_gd). 부호를 뒤집으면 컨버터 전압이 반대로 움직여 전류가 더 커진다.", not_yet="원형 전압 한계는 SVPWM 선형영역의 이상적 근사다.")
    # abc waveforms from the independent model (reconstructed at samples)
    ia = [d * math.cos(2 * math.pi * v["f"] * t) - q * math.sin(2 * math.pi * v["f"] * t) for t, d, q in zip(ta[:m], da[:m], qa[:m])]
    res.add_series("ia", "i_a (abc 모델)", "A", (ta[:m] - 1e-3).tolist(), ia)
    res.add_series("vga", "v_ga / 10", "V", (ta[:m] - 1e-3).tolist(), [Vpk * math.cos(2 * math.pi * v["f"] * t) / 10 for t in ta[:m]], dash=True)
    res.add_plot("p_abc", "a상 전류와 계통전압 (abc 모델)", ["ia", "vga"], x_label="step 이후 시간", y_label="i_a, v_ga/10", y_unit="", group="dq", level="B",
                 proved="d축 전류만 있으면 a상 전류가 계통전압과 동상(단위 역률)이다. i_q ≠ 0이면 위상이 이동한다.", not_yet="전류 리플·고조파는 평균 모델이라 없다.")
    res.tables.append(
        Table(
            "t_sign",
            "부호 약속 (이 실험의 정의)",
            ["항목", "정의"],
            [
                ["전류 방향", "계통 → 컨버터가 +"],
                ["변환", "x_dq = (2/3)(x_a + a·x_b + a²·x_c)·e^(−jθ), θ = ωt, d축 = 계통전압 (진폭불변)"],
                ["plant", "L di_d/dt = v_gd − v_cd − R i_d + ωL i_q,  L di_q/dt = v_gq − v_cq − R i_q − ωL i_d"],
                ["전력", "P = 1.5(v_d i_d + v_q i_q),  Q = 1.5(v_q i_d − v_d i_q) (Q > 0: 유도성 흡수, 전류 지상)"],
                ["올바른 제어", "v_c = v_g + 디커플링 − PI(i* − i)  (컨버터 전압을 낮추면 유입전류 증가)"],
                ["뒤집힌 제어", "v_c = v_g + 디커플링 + PI(i* − i)  → 양의 피드백"],
            ],
            note="q축 부호와 Q의 부호는 변환 정의와 함께 적는다. 다른 교재의 변환(q가 d보다 90° 앞섬 등)을 쓰면 ωL 항과 Q 부호가 바뀐다.",
        )
    )
    res.circuit = {"diagram": dq_circuit(v).to_json(), "intervals": bands, "plot_group": "dq"}
    if tripped is not None or rho >= 1.0:
        res.verdict("UNSTABLE", f"폐루프 극점 |z|max = {rho:.4g}" + (f", step 후 {(tripped - 1e-3) * 1e3:.3g} ms에 전류가 트립 수준({trip:.3g} A)을 넘었다" if tripped else "") + ": PI 출력 부호가 plant의 −v_c와 맞지 않아 양의 피드백이 되었다.")
    elif not settled:
        res.verdict("MARGINAL", "부호는 맞지만 시뮬레이션 창 안에서 기준의 1 % 이내로 수렴하지 않았다" + (": 디커플링이 없어 ωL 교차결합이 외란으로 남고, 극점을 소거한 PI는 느린 극점(L/R)으로만 제거한다." if not v["decouple"] else "."))
    else:
        res.verdict("PASS_WITHIN_MODEL", "올바른 부호의 dq 전류 제어가 기준을 따라가고, dq 식이 독립 abc 시뮬레이션과 일치한다.")
    res.assumptions += ["평형 3상 계통, 이상 PLL (θ = ωt), 계통 전압 피드포워드", "평균 변조기가 샘플 동안 dq 전압을 유지 (연속 회전 각도)", "상별 R·L 동일, 영상분 없음"]
    res.not_valid_for += ["불평형·역상분·약계통·PLL 동특성", "PWM 리플·dead time 전압 오차", "DC-link 동특성 (FL05, EX07)"]
    res.interpretation = (
        "이 약속에서는 컨버터 전압 v_c가 plant에 음의 부호로 들어간다(L di/dt = v_g − v_c − …). 그래서 전류를 늘리려면 컨버터 전압을 계통전압보다 낮춰야 하고, "
        "인버터 전류제어처럼 PI 출력을 그대로 더하면 부호가 뒤집혀 양의 피드백이 된다. 교차결합 ±ωL 항의 부호는 변환 정의에서 나오며, 3상 회로를 직접 적분한 경로와 일치해야 믿을 수 있다."
    )
    return res


def dq_circuit(v: dict) -> Circuit:
    c = Circuit("dq_pfc", 780, 330, title="3상 계통 – RL 필터 – 컨버터 (단선도, 한 상 표시)")
    g = c.add("vsource", "G", 170, 170, 90, "v_g,abc", f"{v['V_LL']:g} V LL", lpos=(194, 166, "start"))
    rr = c.add("resistor", "R", 280, 70, 0, "R", f"{v['R']:g} Ω (상당)")
    ll = c.add("inductor", "L", 400, 70, 0, "L", f"{v['L'] * 1e3:g} mH (상당)")
    cv = c.add("vsource", "CV", 510, 170, 90, "v_c,abc", "컨버터 평균전압", lpos=(488, 166, "end"))
    c.wire("w1", g["a"], (170, 70), rr["a"])
    c.wire("w2", rr["b"], ll["a"])
    c.wire("w3", ll["b"], (510, 70), cv["a"])
    c.wire("w4", cv["b"], (510, 260), (170, 260), g["b"])
    c.dot((510, 70))
    c.probe("pi", "ia", 215, 58, "right", "i_a")
    c.add("block", "PARK", 670, 70, 0, "abc→dq (θ = ωt)", w=150, h=34)
    c.add("block", "PIDQ", 670, 150, 0, "PI_dq + ωL 디커플링", w=150, h=34)
    c.add("block", "SIGN", 670, 230, 0, "v_c = v_g − PI_dq", w=150, h=34)
    c.wire("w_m", (510, 70), (595, 70))
    c.wire("w_c1", (670, 87), (670, 133))
    c.wire("w_c2", (670, 167), (670, 213))
    c.wire("w_c3", (595, 230), (560, 230), (560, 170), (526, 170))
    c.text(340, 292, "i: 계통 → 컨버터가 +, d축 = 계통전압: P = 1.5·v_d·i_d", "note")
    act = ["G", "R", "L", "CV", "w1", "w2", "w3", "w4", "PARK", "PIDQ", "SIGN", "w_m", "w_c1", "w_c2", "w_c3"]
    c.mode("lin", "전류 제어 (선형)", act, "전류를 늘리려면 v_c를 v_g보다 낮춘다")
    c.mode("sat", "전압 벡터 포화", act, "|v_c| = V_dc/√3 (원형 한계)에 묶임")
    return c


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q = [
    Question(
        "PI 출력이 V인지 duty인지 왜 명시해야 하나?",
        "같은 루프를 만들려면 이득의 단위가 달라진다. 전압 출력이면 Kp [V/A], Ki [V/(A·s)], x_I [V]이고, duty 출력이면 변조기 이득 K_PWM [V/duty]로 나눈 Kp [1/A], Ki [1/(A·s)]다. "
        "V/A 숫자를 duty 출력에 그대로 쓰면 루프이득이 K_PWM배(여기서 800배)가 되어 샘플링 루프가 발산한다. K_PWM은 V_dc와 변조 방식에 따라 다르다.",
        "Why must you state whether the PI output is a voltage or a duty?",
        "Because the gains have different units for the same loop. A voltage-output PI has Kp in volts per ampere and an integrator in volts; a duty-output PI must divide both gains by the modulator gain in volts per unit duty. "
        "Using the volt-per-ampere numbers on a duty output multiplies the loop gain by that modulator gain, and the sampled loop becomes unstable.",
        ["V/A vs 1/A", "K_PWM = V/duty", "V_dc·변조 방식 의존", "단위 오류 = 루프이득 K배"],
    ),
    Question(
        "L = 0.8 mH, R = 0.1 Ω, f_c = 1 kHz의 전압 출력 PI를 설계하라.",
        "Kp = Lω_c = 0.8e-3 × 2π × 1000 = 5.026548 V/A, Ki = Rω_c = 628.318531 V/(A·s). PI 영점 Ki/Kp = R/L가 plant 극점을 소거해 루프이득이 ω_c/s가 된다. "
        "e 피드포워드가 적절하고 지연을 무시한 1차 설계이며, 37.5 µs 지연은 1 kHz에서 13.5°를 추가로 먹는다.",
        "Design the voltage-output current PI for 0.8 mH, 0.1 ohm and a 1 kHz crossover.",
        "Kp equals L times omega-c, 5.027 volts per ampere, and Ki equals R times omega-c, 628.3 volts per ampere-second. The PI zero cancels the plant pole so the loop gain is omega-c over s. A 37.5 microsecond total delay still costs 13.5 degrees at 1 kHz.",
        ["5.026548 V/A", "628.318531 V/(A·s)", "극점 소거", "지연 13.5°"],
        kind="calc",
    ),
    Question(
        "연속시간 설계의 위상여유가 90°인데 실제 디지털 루프는 왜 다를까?",
        "샘플링·연산·PWM 갱신이 지연을 만들고(연속 근사 T_d ≈ n_d·T_s + T_s/2), ZOH와 적분기 이산화도 위상을 바꾼다. crossover가 샘플링 주파수에 가까울수록 차이가 커진다. "
        "샘플 순서를 실제대로 구현한 이산 모델로 따로 계산해야 한다.",
        "The continuous design has 90 degrees of margin. Why is the digital loop different?",
        "Sampling, computation and the PWM update add delay, roughly the computation delay plus half a sample for the zero-order hold, and the discretised integrator adds its own phase. The closer the crossover is to the sampling frequency, the larger the difference, so I compute the margin on the exact sampled model.",
        ["연산 지연", "ZOH ½T_s", "이산 모델 따로"],
    ),
    Question(
        "anti-windup이 없으면 포화 뒤에 무엇이 보이나?",
        "포화 동안 오차가 남아 적분기가 필요 이상 쌓이고(windup), 포화가 풀린 뒤 그 값을 되돌리는 동안 전류가 기준을 넘는다. back-calculation ẋ_I = K_i e + K_aw(u_sat − u_unsat)로 "
        "실제 인가값에 맞춰 적분기를 되돌린다. x_I 단위는 V, K_aw는 1/s다.",
        "What do you see after saturation without anti-windup?",
        "During saturation the error persists and the integrator keeps accumulating, so after the limit releases the current overshoots while the integrator unwinds. Back-calculation feeds the difference between the saturated and unsaturated command back into the integrator.",
        ["windup", "overshoot·늦은 회복", "back-calculation 식", "x_I [V], K_aw [1/s]"],
    ),
    Question(
        "PFC dq 전류 제어에서 PI 부호를 거꾸로 연결하기 쉬운 이유는?",
        "전류를 계통→컨버터로 정의하면 L di/dt = v_g − v_c − …로 컨버터 전압이 음의 부호로 들어간다. 컨버터 전압을 높이면 유입전류가 줄어든다. "
        "인버터 제어처럼 v_c = PI + 피드포워드로 연결하면 양의 피드백이 되어 발산한다. 교차결합 ±ωL 항과 Q 부호도 변환 정의와 함께 적는다.",
        "Why is it easy to wire the PI sign backwards in dq PFC current control?",
        "With current defined from the grid into the converter, the converter voltage enters the plant with a minus sign, so raising it reduces the current. Copying the inverter convention, where the PI output is added to the feedforward, gives positive feedback and the current runs away.",
        ["−v_c 부호", "전압↑ → 유입전류↓", "양의 피드백", "변환 정의와 함께"],
        kind="pressure",
    ),
]

EXPERIMENTS = [
    Experiment(
        key="pi_design",
        title="단위를 가진 PI: V 출력 vs duty 출력, 그리고 지연이 먹는 위상",
        goal=(
            "L = 0.8 mH, R = 0.1 Ω, f_c = 1 kHz에서 전압 출력 PI Kp = 5.026548 V/A, Ki = 628.318531 V/(A·s)를 설계하고, duty 출력이면 K_PWM으로 나눠야 같은 루프가 된다는 것을 확인한다. "
            "37.5 µs 지연이 1 kHz에서 13.5°를 먹는 것을 연속 근사와 정확한 샘플링 루프로 따로 계산한다."
        ),
        params=_design_params(),
        presets=[
            Preset("nominal", "교재 L/R/f_c, 전압 출력", {}, "교재 09장", ("nominal", "reference")),
            Preset("duty", "duty 출력 (K_PWM 환산)", {"out": "duty"}, "같은 루프", ("variant", "reference")),
            Preset("unit_error", "duty 출력에 V/A 숫자 그대로", {"out": "duty_raw"}, "단위 오류 → 루프이득 800배", ("failure", "reference")),
            Preset("fc5k", "f_c = 5 kHz", {"fc": 5000.0}, "지연이 위상을 크게 소비", ("corner",)),
            Preset("L_low", "실제 L이 설계값의 50 %", {"L_ratio": 0.5}, "소거가 깨짐 (전류에 따른 L 감소)", ("corner",)),
        ],
        run=run_pi_design,
        model_level="A (+ 샘플링 루프 B)",
        suggested_change="제어기 출력 단위를 ‘duty 출력인데 V/A 숫자를 그대로 사용’으로 바꾼다 (다음에는 f_c를 5 kHz로).",
        prediction=Prediction(
            "duty 출력 PI에 V/A로 설계한 Kp = 5.03, Ki = 628을 그대로 넣으면 (K_PWM = 800 V/duty)?",
            ["같은 루프 — 숫자가 같으니까", "루프이득이 800배가 되어 발산", "루프이득이 1/800이 되어 느려짐", "모르겠다"],
            "루프이득이 800배가 되어 발산",
            "실제 인가 전압은 u = K_PWM·d다. duty 출력에서 Kp = 5.03 [1/A]이면 1 A 오차에 d = 5.03, 즉 4024 V를 명령하는 셈이라 crossover가 800배 위로 올라가 샘플링 주파수를 넘고 폐루프가 발산한다.",
            ["rho", "fc_disc", "pm_disc"],
            handcalc=[{"key": "Kp_V", "label": "Kp (V 출력)", "unit": "V/A"}, {"key": "Ki_V", "label": "Ki (V 출력)", "unit": "V/(A·s)"}, {"key": "lag_fc", "label": "1 kHz에서 지연 위상", "unit": "deg"}],
        ),
        suggested={"out": "duty_raw"},
        student=(
            "전류 루프가 조절할 수 있는 것은 인덕터에 걸리는 평균 전압이다. 그래서 전류 오차 1 A당 몇 V를 걸지(Kp [V/A])가 설계의 출발점이다. "
            "Kp = Lω_c로 잡으면 인덕터 전류가 원하는 속도(ω_c)로 따라오고, Ki = Rω_c로 저항 강하까지 적분기가 메운다. 제어기가 duty를 내보낸다면 1 duty가 몇 V인지(K_PWM)로 나눠야 같은 동작이 된다."
        ),
        expert=(
            "① PI 영점 Ki/Kp = R/L가 plant 극점을 소거하는 설계는 nominal 소거다. L이 전류에 따라 줄거나 센서 필터 극점이 있으면 소거가 깨지고 crossover가 이동한다(L 비율 입력). "
            "② 디지털 루프의 지연은 ‘1.5 T_s’로 외우지 말고 샘플 시점·연산 시간·갱신 시점으로 산정한다(EX07). 여기서는 1 샘플 연산 지연 + ZOH = 37.5 µs다. "
            "③ 연속 근사 exp(−sT_d)의 위상여유와 정확한 샘플링 루프(ZOH plant, 이산 적분기, z^−n_d)의 위상여유를 따로 보고한다. "
            "④ duty 출력이면 K_PWM은 V_dc와 변조 방식에 따라 바뀐다. V_dc 변동을 보상하지 않으면 루프이득도 V_dc에 비례해 움직인다."
        ),
        customer_ko=(
            "이 전류 루프는 전압 출력 기준으로 Kp 5.03 V/A, Ki 628 V/(A·s)이고 1 kHz crossover입니다. 제어기가 duty를 출력한다면 V_dc 800 V 기준 K_PWM으로 나눈 값을 쓰셔야 하며, "
            "샘플링·연산 지연 37.5 µs가 1 kHz에서 약 13.5°를 소비하니 위상여유는 이산 모델로 확인하시죠."
        ),
        customer_en=(
            "In volt-output form this current loop uses Kp of about 5.03 volts per ampere and Ki of about 628, for a 1 kHz crossover. If your controller outputs a duty, divide both by the modulator gain for your DC-link voltage. "
            "The 37.5 microsecond sampling and computation delay costs about 13.5 degrees at 1 kHz, so let's confirm the margin on the sampled model."
        ),
        questions=_Q[:3],
        circuit="rl_loop",
        textbook=[TB_09],
        reference_presets=["nominal", "duty", "unit_error"],
        claim_limit="선형 루프 해석(A)과 이상 변조기의 샘플링 루프(B). 실측 루프이득·센서 필터·공진은 주장하지 않는다.",
    ),
    Experiment(
        key="discrete_loop",
        title="샘플링 전류 루프: 포화·anti-windup·교재의 다섯 가지 사건",
        goal=(
            "RL plant를 샘플 사이에서 정확히(ZOH) 이산화하고 엔진의 연속 적분과 대조한 뒤, 샘플링 PI·1 샘플 연산 지연·전압 포화·back-calculation anti-windup으로 "
            "교재 실험(부하 20→80 %, 입력 −10 %, 전류 제한 50 %, 센서 offset, 한 주기 추가 지연)의 overshoot·정정·제한 초과·회복을 수치로 보고한다."
        ),
        params=_loop_params(),
        presets=[
            Preset("load_step", "부하 20→80 % (V_dc 600 V라 포화)", {"scenario": "load_step"}, "교재 PI, 전압 여유가 작음", ("nominal", "reference")),
            Preset("input_dip", "입력 전압 −10 % (e 피드포워드)", {"scenario": "input_dip"}, "피드포워드가 외란을 대부분 상쇄", ("variant", "reference")),
            Preset("input_dip_noff", "입력 −10 %, 피드포워드 없음", {"scenario": "input_dip", "e_ff": False}, "극점 소거 PI는 외란을 느린 극점으로만 제거", ("corner",)),
            Preset("current_limit", "전류 제한 50 %", {"scenario": "current_limit"}, "제한 초과 여부", ("variant", "reference")),
            Preset("sensor_offset", "센서 offset +0.5 A", {"scenario": "sensor_offset"}, "정상상태 오차가 남는다", ("failure", "reference")),
            Preset("extra_delay", "한 주기 추가 지연", {"scenario": "extra_delay"}, "위상여유 감소", ("corner", "reference")),
            Preset("aw_aggressive", "K_aw 과대 (K_aw·T_i = 50)", {"Kaw_ratio": 50.0}, "적분기를 음으로 끌어내림", ("failure",)),
            Preset("zero_wc5", "PI 영점 ω_c/5, 부하 step", {"zero": "wc5"}, "적분 동작이 강하면 windup이 크다", ("variant",)),
            Preset("big_vmax", "V_max 여유 큼 (V_dc 800 V)", {"V_max": 800 / math.sqrt(3)}, "포화 없음 비교", ("variant",)),
        ],
        run=run_discrete_loop,
        model_level="B (정확 ZOH plant + 샘플링 PI)",
        suggested_change="anti-windup 이득 K_aw·T_i를 1 → 0으로 바꿔 본다 (부하 step, 포화 발생 조건). 다음에는 50으로 과하게 키워 본다.",
        prediction=Prediction(
            "부하 20→80 % step에서 전압이 포화된다. anti-windup을 빼면 전류 overshoot는?",
            ["거의 같다", "커지고 회복이 늦어진다", "작아진다", "모르겠다"],
            "커지고 회복이 늦어진다",
            "포화 동안 전류 오차가 남아 적분기가 계속 쌓인다. 포화가 풀리면 쌓인 적분값이 u를 필요 이상으로 유지해 전류가 기준을 넘고, 적분기가 되돌아올 때까지 회복이 늦다.",
            ["ov_aw", "ov_noaw", "rec_noaw"],
            handcalc=[{"key": "Kp", "label": "Kp", "unit": "V/A"}, {"key": "sat_time", "label": "포화 시간 (감으로)", "unit": "s"}],
        ),
        suggested={"Kaw_ratio": 0.0},
        student=(
            "전압에는 한계가 있다. 전류를 빨리 올리라는 명령이 커도 컨버터는 V_max까지만 걸 수 있어 전류가 천천히 오른다. 그동안 PI의 적분기는 ‘아직 모자라다’며 계속 값을 쌓는데, "
            "전류가 따라잡은 뒤에도 쌓인 값 때문에 전압을 너무 오래 걸어 전류가 기준을 넘는다. anti-windup은 실제로 걸린 전압을 보고 적분기를 되돌린다."
        ),
        expert=(
            "① x_I의 단위는 V, K_aw는 1/s다(duty 출력이면 무차원/1/s). ② 전류 제한을 기준값 limiter로만 걸면 루프 overshoot만큼 제한을 넘는다 — 제한 여유 또는 빠른 하드웨어 OCP가 필요하다. "
            "③ 센서 offset은 폐루프가 측정값을 맞추므로 실제 전류 오차로 남는다. dq 시스템에서 abc 센서 offset은 기본파 주파수 리플로 나타난다. "
            "④ 한 샘플 추가 지연은 f_c에서 360·f_c·T_s만큼 위상을 더 먹는다. ⑤ 성공 기준은 학습 목표로 정한 값이며 모델의 물리 한계와 따로 표시한다."
        ),
        customer_ko=(
            "부하 step에서 전압 한계에 걸리는 조건이라 anti-windup 유무에 따라 overshoot와 회복시간이 크게 다릅니다. 전류 제한은 기준값에만 걸려 있어 루프 overshoot만큼 넘을 수 있으니, "
            "제한 여유와 하드웨어 OCP 설정을 같이 보시고 센서 offset 보정 절차도 확인하시죠."
        ),
        customer_en=(
            "At this load step the converter hits its voltage limit, so the overshoot and recovery time depend strongly on the anti-windup. The current limit acts only on the reference, so the loop overshoot can exceed it; "
            "please review the limit margin together with the hardware over-current setting, and the current-sensor offset calibration."
        ),
        questions=[_Q[3]],
        circuit="rl_loop",
        textbook=[TB_09, TB_E07],
        reference_presets=["load_step", "input_dip", "current_limit", "sensor_offset", "extra_delay"],
        claim_limit="이상 평균 변조기의 샘플링 루프(B). PWM 리플·센서 필터·하드웨어 OCP는 주장하지 않는다.",
    ),
    Experiment(
        key="dq_sign",
        title="dq PFC 전류 제어: 부호가 맞아야 PI가 의미가 있다",
        goal=(
            "계통→컨버터 전류 약속과 진폭불변 dq에서 L di_d/dt = v_gd − v_cd − R i_d + ωL i_q, L di_q/dt = v_gq − v_cq − R i_q − ωL i_d를 쓰고, "
            "올바른 부호(v_c = v_g − PI)는 수렴하고 뒤집힌 부호는 발산한다는 것을 보인다. dq 모델을 3상 회로를 직접 적분한 독립 abc 모델과 대조하고 P = 1.5(v_d i_d + v_q i_q)를 확인한다."
        ),
        params=[
            Param("V_LL", "계통 선간전압 (RMS)", "V", 400.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="400 V LL"),
            Param("f", "계통 주파수", "Hz", 50.0, "Hz", vmin=10, vmax=400, source="TEXTBOOK", source_note="50 Hz"),
            Param("L", "상당 필터 L", "H", 0.8e-3, "mH", vmin=1e-5, vmax=0.1, source="TEXTBOOK", source_note="0.8 mH"),
            Param("R", "상당 R", "Ω", 0.1, "Ω", vmin=1e-4, vmax=10, source="TEXTBOOK", source_note="0.1 Ω"),
            Param("fc", "전류 루프 f_c", "Hz", 1000.0, "Hz", vmin=10, vmax=1e4, source="TEXTBOOK", source_note="1 kHz"),
            Param("fs", "샘플링 주파수", "Hz", 40e3, "kHz", vmin=2e3, vmax=2e5, source="ASSUMED", group="디지털"),
            Param("V_dc", "DC-link 전압 (벡터 한계 V_dc/√3)", "V", 700.0, "V", vmin=100, vmax=2000, source="ASSUMED", source_note="700 V", group="회로"),
            Param("id_step", "i_d 기준 step", "A", 20.0, "A", vmin=-200, vmax=200, source="ASSUMED", group="기준"),
            Param("iq_ref", "i_q 기준", "A", 0.0, "A", vmin=-200, vmax=200, source="ASSUMED", source_note="0이면 단위 역률", group="기준"),
            Param("sign", "PI 출력 부호", "", "correct", kind="choice", choices=[("correct", "올바름: v_c = v_g − PI"), ("reversed", "뒤집힘: v_c = v_g + PI (인버터 관습 복사)")], source="TEXTBOOK", group="기준"),
            Param("decouple", "ωL 디커플링", "", True, kind="bool", source="ASSUMED", group="디지털"),
            Param("t_sim", "시뮬레이션 길이", "s", 21e-3, "ms", vmin=2e-3, vmax=40e-3, source="ASSUMED", source_note="step 뒤 한 계통주기 이상", group="시뮬레이션"),
        ],
        presets=[
            Preset("correct", "올바른 부호, i_d 0→20 A", {}, "단위 역률 PFC", ("nominal", "reference")),
            Preset("reversed", "뒤집힌 부호", {"sign": "reversed"}, "양의 피드백 → 발산", ("failure", "reference")),
            Preset("reactive", "i_q = +5 A 추가", {"iq_ref": 5.0}, "Q 부호 확인", ("variant", "reference")),
            Preset("no_decouple", "디커플링 없음", {"decouple": False}, "교차결합 외란", ("variant",)),
        ],
        run=run_dq_sign,
        model_level="B (dq 평균모델 + 독립 abc 모델)",
        suggested_change="PI 출력 부호를 ‘뒤집힘’으로 바꿔 본다.",
        prediction=Prediction(
            "PI 출력을 v_c에 더하는 인버터식 연결(v_c = v_g + PI)로 바꾸면 i_d step 응답은?",
            ["같다 — PI가 알아서 맞춘다", "반대 방향으로 가며 발산한다", "느려질 뿐 수렴한다", "모르겠다"],
            "반대 방향으로 가며 발산한다",
            "L di/dt = v_g − v_c − …라 v_c를 올리면 전류가 줄어든다. 오차가 양일 때 v_c를 올리면 전류가 더 줄고 오차가 더 커지는 양의 피드백이다. 폐루프 극점이 단위원 밖으로 나간다.",
            ["rho", "t_trip", "frame_dev"],
            handcalc=[{"key": "Vgd", "label": "v_gd (상전압 peak)", "unit": "V"}, {"key": "P", "label": "i_d = 20 A의 P", "unit": "W"}],
        ),
        suggested={"sign": "reversed"},
        student=(
            "계통에서 컨버터로 들어오는 전류는 계통전압과 컨버터 전압의 차이가 인덕터에 걸려 만들어진다. 컨버터 전압을 낮추면 차이가 커져 전류가 늘고, 높이면 줄어든다. "
            "그래서 ‘전류가 모자라면 컨버터 전압을 낮춘다’가 올바른 방향이다. 인버터에서 ‘전류가 모자라면 전압을 올린다’를 그대로 가져오면 거꾸로 된다."
        ),
        expert=(
            "① dq 식의 ±ωL 교차결합 부호는 변환 정의(θ, q축 방향)에서 나온다. 식을 외우지 말고 3상 회로 적분과 대조한다. ② 진폭불변 변환이라 P = 1.5(v_d i_d + v_q i_q)이고, Q의 부호는 변환과 함께 적는다. "
            "③ d축을 계통전압에 맞춘 평형 계통의 SISO 설계를 불평형·약계통·PLL 동특성으로 자동 확장하지 않는다. ④ 전압 벡터 포화는 원형 한계와 축별 clamp가 다르다(EX07)."
        ),
        customer_ko=(
            "전류를 계통→컨버터로 정의하셨다면 컨버터 전압이 plant에 음의 부호로 들어가므로 PI 출력은 계통전압 피드포워드에서 빼셔야 합니다. 첫 기동 전에 저전압·저전류 조건에서 "
            "i_d 기준 step의 방향을 확인하고, Q 부호는 사용하신 dq 변환 정의와 함께 문서에 남겨 주세요."
        ),
        customer_en=(
            "If current is defined from the grid into the converter, the converter voltage enters the plant with a minus sign, so the PI output must be subtracted from the grid-voltage feedforward. "
            "Before the first power-up, please verify the direction of a small d-axis current step at low voltage, and document the sign of reactive power together with your dq transform."
        ),
        questions=[_Q[4]],
        circuit="dq_pfc",
        textbook=[TB_09, TB_08],
        reference_presets=["correct", "reversed", "reactive"],
        runtime_hint="seconds",
        claim_limit="평형 계통·이상 PLL·평균 변조기의 dq 전류 루프(B). 불평형·약계통·PWM 리플은 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL06",
    title="제어 — PI보다 plant와 부호가 먼저다",
    title_en="Control: the plant and the sign come before the PI",
    track="basic",
    order=6,
    path_note="14일 경로 5일차 (09 제어, FL06)",
    textbook=[TB_09, TB_E07],
    prerequisites=["FL01", "FL05"],
    summary="단위를 가진 PI → 지연이 먹는 위상 → 샘플링 루프의 포화·anti-windup·교재 사건 → dq 부호. 연속 근사와 샘플링 루프, dq와 abc를 서로 독립 경로로 확인한다.",
    experiments=EXPERIMENTS,
    minimum_scope="RL/current loop·PI 단위(V vs duty)·delay·saturation·anti-windup·step·dq 부호 (교재 19장 표); Kp 5.026548 V/A, Ki 628.318531 V/(A·s), 37.5 µs → 13.5°",
    claim_limits=[
        "이상 평균 변조기의 선형·샘플링 루프 결과(A/B). PWM 리플·센서 필터·양자화 미포함",
        "위상여유·overshoot 기준은 학습용이며 고객 사양이 아님",
        "dq 결과는 평형 계통·이상 PLL 가정",
        "성공 기준과 모델의 물리 한계를 섞지 않음",
    ],
    test_paths=["tests/test_fl06.py"],
)
