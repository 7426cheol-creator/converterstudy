"""EX02 - Nonlinear Coss, dead time and ZVS: never judge with one energy formula (textbook E02).

Three levels, kept apart on purpose:
  A  closed-form Qoss/Eoss integrals and equivalent capacitances (vs independent quadrature);
  A  charge screen: fixed-rail half-bridge with a constant injected current, t = 2 Qoss/I;
  D  synthetic commutation event: the inductor current i(t) is a state and changes during the
     dead time; body-diode clamps, gate delays, dead-time quantisation and a hard turn-on
     residual are events.  Two independent solvers (voltage-state ODE with scipy, charge-state
     RK4) and an energy ledger check it.  It is valid only for this synthetic C(v); it is not a
     vendor device model and it does not model reverse recovery, gate dynamics or ringing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import coss as ref

TB_E02 = TextbookRef("expert-e02.-비선형-cossdead-timezvs-에너지식-하나로-판정하지-않기-ex02", "E02. 비선형 Coss·dead time·ZVS [EX02]")
TB_04 = TextbookRef("sisicgan과-데이터시트-소자를-시스템으로-읽기-fl02", "04. Si·SiC·GaN과 데이터시트 [FL02]")
TB_E13 = TextbookRef("expert-e13-빠르게-깊어지기-위한-실행-순서와-통과-기준", "E13. 실행 순서와 통과 기준")

NEAR_ZVS_FRAC = 0.05  # learning threshold (textbook ch.19): VDS <= 5 % of the bus; not a vendor ZVS criterion


# ======================================================================================
# Node capacitance model (fixed rails, low-side Coss node-gnd, high-side Coss rail-node)
# ======================================================================================


@dataclass
class Node:
    Vb: float
    C0: float
    V0: float
    Cpar: float = 0.0
    model: str = "nonlinear"  # nonlinear | fixed
    Cfix: float = 0.0  # per-device constant capacitance for model == "fixed"

    # per-device functions
    def Cd(self, v):
        if self.model == "fixed":
            return self.Cfix
        return self.C0 / math.sqrt(1.0 + max(v, -0.999 * self.V0) / self.V0)

    def Qd(self, v):
        if self.model == "fixed":
            return self.Cfix * v
        return 2.0 * self.C0 * self.V0 * (math.sqrt(1.0 + v / self.V0) - 1.0)

    def Ed(self, v):
        if self.model == "fixed":
            return 0.5 * self.Cfix * v * v
        u = 1.0 + v / self.V0
        return self.C0 * self.V0**2 * (2.0 / 3.0 * u**1.5 - 2.0 * u**0.5 + 4.0 / 3.0)

    # node functions (v = low-side VDS = node voltage; high-side VDS = Vb - v)
    def Cn(self, v):
        return self.Cd(v) + self.Cd(self.Vb - v) + self.Cpar

    def Qn(self, v):
        return self.Qd(v) + (self.Qd(self.Vb) - self.Qd(self.Vb - v)) + self.Cpar * v

    def En(self, v):
        return self.Ed(v) + self.Ed(self.Vb - v) + 0.5 * self.Cpar * v * v

    def v_of_q(self, q):
        q = min(max(q, 0.0), self.Qn(self.Vb))
        return brentq(lambda v: self.Qn(v) - q, 0.0, self.Vb, xtol=1e-12, rtol=1e-14)

    def rail_fraction(self, v):
        """Share of the node current that flows through the high-side Coss into the rail."""
        return self.Cd(self.Vb - v) / self.Cn(v)

    def hard_on_loss(self, v1):
        """Channel energy when the high side turns on with the node at v1 (fast event, fixed rails)."""
        Vb = self.Vb
        q_rail = self.Qd(Vb) - self.Qd(v1) + self.Cpar * (Vb - v1)
        dE = (self.Ed(Vb) - self.Ed(v1)) - self.Ed(Vb - v1) + 0.5 * self.Cpar * (Vb**2 - v1**2)
        return Vb * q_rail - dE


# ======================================================================================
# Synthetic commutation event with a varying inductor current
# ======================================================================================


@dataclass
class Transition:
    t: list
    v: list
    i: list
    phase: list
    events: list  # (time, name, detail)
    v_on: float
    i_on: float
    t_on: float
    t_reach: float | None
    clamp_time: float
    returned: bool
    v_max: float
    E_diode: float
    E_hard: float
    ledger: dict
    q_moved: float


def simulate_transition(node: Node, L: float, vx: float, I0: float, td: float, t_doff: float = 0.0, t_don: float = 0.0, Vf: float = 0.0, rtol: float = 1e-10, tail: float = 0.0) -> Transition:
    """Low side carries I0 (positive = into the node) and turns off; high side turns on at td + t_don.

    Phases: 'Q_L on' (channel pins v = 0) -> 'dead time' (both off, node moves with C_node(v))
    -> 'D_H clamp' (v = Vb, current into rail) or 'D_L clamp' (v = 0, current out of node)
    -> 'Q_H on'.  i is the inductor current flowing into the node; its far end sits at vx.
    """
    Vb = node.Vb
    t_on = td + t_don
    T: list = []
    Vv: list = []
    Ii: list = []
    Ph: list = []
    ev: list = [(0.0, "L측 turn-off 명령", "")]
    q_in = 0.0
    q_rail = 0.0
    E_diode = 0.0
    t = 0.0
    v = 0.0
    i = I0

    def push(tt, vv, ii, ph):
        T.append(float(tt))
        Vv.append(float(vv))
        Ii.append(float(ii))
        Ph.append(ph)

    # phase 0: low-side channel still on until the actual gate-off
    if t_doff > 0:
        push(0.0, 0.0, i, "Q_L on")
        i1 = i + vx / L * t_doff
        q_in += (i + i1) / 2 * t_doff
        i = i1
        t = t_doff
        push(t, 0.0, i, "Q_L on")
    ev.append((t, "L측 실제 gate-off", f"i = {i:.4g} A"))
    phase = "dead time"
    if i <= 0:
        # wrong direction: the low-side body diode takes the current, node stays at 0 (-Vf)
        phase = "D_L clamp"
        ev.append((t, "전류 방향이 반대: L측 body diode 도통", "노드가 V_bus로 가지 않음"))
    t_reach = None
    clamp_time = 0.0
    returned = False
    v_max = 0.0
    while t < t_on - 1e-18:
        if phase == "dead time":

            def f(tt, y):
                vv, ii = y[0], y[1]
                cn = node.Cn(vv)
                return [ii / cn, (vx - vv) / L, ii, ii * node.Cd(Vb - vv) / cn]

            def hit_top(tt, y):
                return y[0] - Vb

            hit_top.terminal = True
            hit_top.direction = 1

            def hit_bot(tt, y):
                return y[0]

            hit_bot.terminal = True
            hit_bot.direction = -1
            sol = solve_ivp(f, (t, t_on), [v, i, 0.0, 0.0], method="DOP853", rtol=rtol, atol=[1e-9, 1e-12, 1e-21, 1e-21], events=[hit_top, hit_bot], dense_output=True, max_step=(t_on - t) / 50)
            ts = np.linspace(t, sol.t[-1], 80)
            ys = sol.sol(ts)
            for k in range(ts.size):
                push(ts[k], ys[0][k], ys[1][k], "dead time")
            v_max = max(v_max, float(np.max(ys[0])))
            v, i = float(sol.y[0][-1]), float(sol.y[1][-1])
            q_in += float(sol.y[2][-1])
            q_rail += float(sol.y[3][-1])
            t = float(sol.t[-1])
            if sol.t_events[0].size:
                v = Vb
                if t_reach is None:
                    t_reach = t
                ev.append((t, "노드가 V_bus 도달 → H측 body diode clamp", f"i = {i:.4g} A"))
                phase = "D_H clamp"
            elif sol.t_events[1].size:
                v = 0.0
                ev.append((t, "노드가 0으로 되돌아옴 → L측 body diode clamp", f"i = {i:.4g} A, 최고 {v_max:.4g} V"))
                phase = "D_L clamp"
                returned = True
        elif phase == "D_H clamp":
            # v = Vb + Vf (ideal diode into the rail), di/dt constant; leaves when i reaches 0
            didt = (vx - Vb - Vf) / L
            t_zero = t + (-i / didt) if didt < 0 and i > 0 else math.inf
            t_end = min(t_on, t_zero)
            dt = t_end - t
            i1 = i + didt * dt
            qd = (i + i1) / 2 * dt
            q_in += qd
            q_rail += qd
            E_diode += Vf * qd
            clamp_time += dt
            push(t, Vb, i, "D_H clamp")
            push(t_end, Vb, i1, "D_H clamp")
            t, i = t_end, i1
            if t_zero <= t_on and t < t_on - 1e-18:
                ev.append((t, "H측 diode 전류 0 → 노드가 다시 떨어지기 시작", "dead time 과다"))
                phase = "dead time"
                returned = True
                i = 0.0
        elif phase == "D_L clamp":
            didt = (vx + Vf) / L
            t_zero = t + (-i / didt) if didt > 0 and i < 0 else math.inf
            t_end = min(t_on, t_zero)
            dt = t_end - t
            i1 = i + didt * dt
            qd = (i + i1) / 2 * dt
            q_in += qd
            E_diode += Vf * abs(qd)
            push(t, 0.0, i, "D_L clamp")
            push(t_end, 0.0, i1, "D_L clamp")
            t, i = t_end, i1
            if t_zero <= t_on and t < t_on - 1e-18:
                ev.append((t, "L측 diode 전류 0 → 노드가 다시 움직임", ""))
                phase = "dead time"
                i = 0.0
    ev.append((td, "H측 turn-on 명령", ""))
    if t_don > 0:
        ev.append((t_on, "H측 실제 turn-on", ""))
    v_on = v if phase != "D_H clamp" else Vb
    E_hard = node.hard_on_loss(v_on) if v_on < Vb - 1e-9 else 0.0
    ev.append((t_on, "turn-on 직전 V_DS(H) = V_bus − v", f"{Vb - v_on:.4g} V"))
    # energy ledger for [0, t_on]: source vx delivers vx*q_in; rail absorbs Vb*q_rail; diode dissipates E_diode
    W_L = 0.5 * L * (i * i - I0 * I0)
    W_C = node.En(v_on) - node.En(0.0)
    E_x = vx * q_in
    resid = E_x - (W_L + W_C + Vb * q_rail + E_diode)
    norm = max(abs(E_x), abs(W_L), abs(W_C), abs(Vb * q_rail), 1e-12)
    ledger = {"E_source": E_x, "dW_L": W_L, "dW_C": W_C, "E_rail": Vb * q_rail, "E_diode": E_diode, "residual": resid, "normalised": resid / norm}
    if tail > 0:
        push(t_on, Vb, i, "Q_H on")
        push(t_on + tail, Vb, i + (vx - Vb) / L * tail, "Q_H on")
    v_max = max(v_max, v_on)
    return Transition(T, Vv, Ii, Ph, ev, v_on, i, t_on, t_reach, clamp_time, returned, v_max, E_diode, E_hard, ledger, q_in)


def rk4_charge_path(node: Node, L: float, vx: float, I0: float, td: float, n: int, t_doff: float = 0.0, t_don: float = 0.0, Vf: float = 0.0):
    """Independent path: fixed-step RK4 in charge coordinates (q, i), v = Qn^-1(q).

    Phase 0 (channel still on) is integrated exactly; the free dead-time phase uses
    RK4; a top-rail clamp with an ideal diode + Vf is linear.  It does not handle a
    return from the clamp (current reversal) - callers only compare such cases.
    Returns (v, i) at the actual turn-on td + t_don.
    """
    i = I0 + vx / L * t_doff
    t_free = td + t_don - t_doff
    h = t_free / n
    q = 0.0
    Qtop = node.Qn(node.Vb)
    clamped = False
    for _ in range(n):
        if clamped:
            i += (vx - node.Vb - Vf) / L * h
            continue

        def rhs(qq, ii):
            return ii, (vx - node.v_of_q(qq)) / L

        k1 = rhs(q, i)
        k2 = rhs(q + h / 2 * k1[0], i + h / 2 * k1[1])
        k3 = rhs(q + h / 2 * k2[0], i + h / 2 * k2[1])
        k4 = rhs(q + h * k3[0], i + h * k3[1])
        qn = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        inn = i + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if qn >= Qtop:
            # linear interpolation of the crossing inside the step, then clamp
            frac = (Qtop - q) / (qn - q)
            i = i + frac * (inn - i)
            i += (vx - node.Vb - Vf) / L * (1 - frac) * h
            q = Qtop
            clamped = True
        else:
            q, i = qn, inn
    return (node.Vb if clamped else node.v_of_q(q)), i


def classify(Vb: float, v_on: float, returned: bool, I0: float, v_max: float | None = None, reached: bool | None = None) -> tuple[str, str]:
    res = Vb - v_on
    v_max = v_on if v_max is None else v_max
    if I0 <= 0 and v_max <= 1e-9:
        return "HARD", "전류 방향이 반대라 노드가 움직이지 않음 → hard turn-on (역회복 미모델)"
    if res <= 1e-9 * Vb:
        return ("ZVS_LATE_RETURN" if returned else "ZVS"), ("완전 전환 후 diode clamp 상태에서 turn-on" if not returned else "전환 후 되돌아온 이력 있음 — dead time 과다 위험")
    if res <= NEAR_ZVS_FRAC * Vb:
        return "NEAR_ZVS", f"잔류 V_DS {res:.3g} V ≤ 5 % V_bus (학습용 기준, 제조사 ZVS 보증 기준 아님)"
    if returned and reached:
        return "LOST", f"V_bus 도달 후 되돌아와 잔류 V_DS {res:.3g} V — dead time 과다"
    if returned or v_max > v_on + 1e-6 * Vb:
        return "FELL_BACK", f"최고 {v_max:.4g} V까지 올라갔다가 에너지·전하 부족으로 되돌아옴 (잔류 {res:.3g} V)"
    if res >= 0.95 * Vb:
        return "HARD", f"노드가 거의 움직이지 못함 (잔류 {res:.3g} V)"
    return "PARTIAL", f"부분 전환: 잔류 V_DS {res:.3g} V에서 hard turn-on"


CLASS_KO = {"ZVS": "완전 전환 (ZVS, 합성 모델)", "ZVS_LATE_RETURN": "완전 전환 (되돌아옴 이력)", "NEAR_ZVS": "near-ZVS (학습 기준)", "PARTIAL": "부분 전환", "LOST": "도달 후 되돌아와 ZVS 상실", "FELL_BACK": "올라가다 되돌아옴 (ZVS 실패)", "HARD": "hard switching"}


# ======================================================================================
# Experiments
# ======================================================================================


def _coss_params(extra=None):
    ps = [
        Param("C0", "C(0) 계수 C₀", "F", 2e-9, "nF", vmin=1e-12, vmax=1e-6, source="TEXTBOOK", source_note="C(v) = 2 nF/√(1+v/40 V) — 학습용 합성식, 특정 소자 fit 아님"),
        Param("V0", "전압 계수 V₀", "V", 40.0, "V", vmin=0.1, vmax=1e4, source="TEXTBOOK", source_note="40 V"),
        Param("Vb", "버스 전압 V_bus", "V", 800.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="800 V"),
    ]
    return ps + (extra or [])


def run_qe(v: dict) -> Result:
    res = Result("EX02", "qe_integrals", "A")
    C0, V0, Vb = v["C0"], v["V0"], v["Vb"]
    Qc = ref.Q(Vb, C0, V0)
    Ec = ref.E(Vb, C0, V0)
    Qn, _ = quad(lambda u: C0 / math.sqrt(1 + u / V0), 0, Vb, epsabs=0, epsrel=1e-13, limit=200)
    En, _ = quad(lambda u: u * C0 / math.sqrt(1 + u / V0), 0, Vb, epsabs=0, epsrel=1e-13, limit=200)
    tb = abs(C0 - 2e-9) < 1e-18 and abs(V0 - 40) < 1e-12 and abs(Vb - 800) < 1e-9
    res.add_metric("Qoss", "Q_oss(V_bus) = ∫C dv", Qc, "C", ref=573.212e-9 if tb else None, ref_label="교재 573.212 nC", tol=1e-6, basis="소자 1개, 0→V_bus")
    res.add_metric("Eoss", "E_oss(V_bus) = ∫v·C dv", Ec, "J", ref=180.238e-6 if tb else None, ref_label="교재 180.238 µJ", tol=1e-5, basis="소자 1개 저장에너지")
    Cs = ref.C(Vb, C0, V0)
    res.add_metric("Coss_pt", "Coss(V_bus) 한 점 (미분용량)", Cs, "F", basis="데이터시트 한 점 값에 해당")
    res.add_metric("Cotr", "C_o(tr) = Q/V (시간·전하 등가)", Qc / Vb, "F")
    res.add_metric("Coer", "C_o(er) = 2E/V² (에너지 등가)", 2 * Ec / Vb**2, "F")
    res.add_metric("Q_pt", "Coss(V_bus)·V_bus (한 점으로 계산한 전하)", Cs * Vb, "C", note=f"실제 Q_oss의 {Cs * Vb / Qc * 100:.1f} % — 전하를 과소평가")
    res.add_metric("E_pt", "½·Coss(V_bus)·V_bus² (한 점으로 계산한 에너지)", 0.5 * Cs * Vb**2, "J", note=f"실제 E_oss의 {0.5 * Cs * Vb**2 / Ec * 100:.1f} %")
    res.add_metric("VQ_vs_2E", "V_bus·Q_oss vs 2·E_oss", Vb * Qc / (2 * Ec), "", basis="비율 (선형 C면 정확히 1)", note="half-bridge hard turn-on 손실 V·Q와 흔한 근사 2E_oss가 다르다")
    res.add_check(check_close("Q_oss: 닫힌 식 vs 수치 적분(quad)", Qc, Qn, 1e-10, "해석 적분식 vs 적응 Gauss–Kronrod 구적", True, "C"))
    res.add_check(check_close("E_oss: 닫힌 식 vs 수치 적분(quad)", Ec, En, 1e-10, "해석 적분식 vs 적응 Gauss–Kronrod 구적", True, "J"))
    vs = np.linspace(0, Vb, 241)
    res.add_series("C", "C(v) 미분용량", "F", vs.tolist(), [ref.C(x, C0, V0) for x in vs])
    res.add_series("Ctr", "C_o(tr)(V) = Q/V", "F", vs[1:].tolist(), [ref.co_tr(x, C0, V0) for x in vs[1:]], dash=True)
    res.add_series("Cer", "C_o(er)(V) = 2E/V²", "F", vs[1:].tolist(), [ref.co_er(x, C0, V0) for x in vs[1:]], dash=True)
    res.add_series("Q", "Q_oss(v)", "C", vs.tolist(), [ref.Q(x, C0, V0) for x in vs])
    res.add_series("Qlin", "Coss(V_bus)·v (한 점 선형 가정)", "C", vs.tolist(), [Cs * x for x in vs], dash=True)
    res.add_series("E", "E_oss(v)", "J", vs.tolist(), [ref.E(x, C0, V0) for x in vs])
    res.add_series("Elin", "½·Coss(V_bus)·v² (한 점)", "J", vs.tolist(), [0.5 * Cs * x * x for x in vs], dash=True)
    res.add_plot("p_c", "세 가지 다른 커패시턴스", ["C", "Ctr", "Cer"], x_label="V_DS", x_unit="V", y_label="C", y_unit="F", kind="xy", log_y=True, level="A",
                 markers=[{"x": Vb, "y": Cs, "label": "Coss(V_bus)"}, {"x": Vb, "y": Qc / Vb, "label": "C_o(tr)"}, {"x": Vb, "y": 2 * Ec / Vb**2, "label": "C_o(er)"}],
                 proved="같은 소자에서도 미분용량·시간등가·에너지등가 커패시턴스가 서로 다르다는 것을 적분으로 보였다.", not_yet="합성 C(v)이며 특정 소자의 측정 곡선이 아니다. 온도·lot 편차 미포함.")
    res.add_plot("p_q", "전하: 전환시간을 정한다", ["Q", "Qlin"], x_label="V_DS", x_unit="V", y_label="Q", y_unit="C", kind="xy", level="A",
                 proved="Coss 한 점 × V는 필요한 전하를 크게 과소평가한다 → dead time 과소 설계로 이어진다.", not_yet="합성 C(v) = 2 nF/√(1 + v/40 V)의 적분이다. 실제 소자의 C_oss(v)·온도 의존·데이터시트 측정 조건은 MISSING_INPUT이며, 이 결론의 크기는 합성 소자에 한한다.")
    res.add_plot("p_e", "에너지: 저장에너지와 hard-switching 손실의 재료", ["E", "Elin"], x_label="V_DS", x_unit="V", y_label="E", y_unit="J", kind="xy", level="A",
                 proved="E_oss는 ∫v·C dv이며 한 점 ½CV²와 다르다.", not_yet="어떤 회로에서 이 에너지가 실제로 손실되는지는 회로 경로가 정한다(실험 2·3).")
    res.verdict("PASS_WITHIN_MODEL", "Q_oss·E_oss 닫힌 식이 독립 수치적분과 일치하고 교재 값을 재현")
    res.assumptions += ["합성 C(v) = C₀/√(1+v/V₀)", "소자 1개의 출력 용량(드레인-소스)만"]
    res.not_valid_for += ["실제 소자 데이터시트 곡선(MISSING_INPUT)", "온도·전류 의존성"]
    res.interpretation = "Q_oss는 전환에 필요한 전하(→ 시간), E_oss는 저장에너지다. 둘은 서로 다른 함수이고, 어느 것도 데이터시트의 Coss 한 점으로 대신할 수 없다."
    return res


def run_hb_cc(v: dict) -> Result:
    res = Result("EX02", "hb_constant_current", "A (charge screen) + D-합성 해석")
    C0, V0, Vb, I, td, Cpar = v["C0"], v["V0"], v["Vb"], v["I"], v["td"], v["Cpar"]
    node = Node(Vb, C0, V0, Cpar)
    Qreq = node.Qn(Vb)
    t_full = Qreq / abs(I) if I != 0 else math.inf
    t_ref = ref.hb_transition_time_constant_current(Vb, I, C0, V0) if Cpar == 0 else None
    tb = abs(C0 - 2e-9) < 1e-18 and abs(V0 - 40) < 1e-12 and abs(Vb - 800) < 1e-9 and Cpar == 0
    res.add_metric("t_trans", "완전 전환시간 t = Q_node(V_bus)/|I|", t_full, "s", ref=(286.606e-9 if tb and abs(abs(I) - 4) < 1e-12 else t_ref), ref_label="교재 286.606 ns (4 A)" if tb else "2Q_oss/|I|", tol=1e-5, basis="고정 rail·동일 두 소자·정전류")
    Cs = ref.C(Vb, C0, V0)
    t_pt = (2 * Cs + Cpar) * Vb / abs(I) if I != 0 else math.inf
    res.add_metric("t_pt", "Coss(V_bus) 한 점으로 계산한 전환시간", t_pt, "s", ref=174.574e-9 if tb and abs(abs(I) - 4) < 1e-12 else None, ref_label="교재 174.574 ns", tol=1e-5, note=f"{(1 - t_pt / t_full) * 100:.1f} % 과소평가" if I else "")
    res.add_metric("Qreq", "필요 전하 Q_node(V_bus) = 2Q_oss + C_par·V", Qreq, "C", basis="이 half-bridge·고정 rail 가정의 결과")
    # partial transition at the dead time (closed-form inversion vs ODE)
    if I > 0:
        q_av = I * td
        v_cf = node.v_of_q(q_av) if q_av < Qreq else Vb
        sol = solve_ivp(lambda t, y: [I / node.Cn(min(y[0], Vb))], (0, min(td, t_full * (1 - 1e-12))), [0.0], method="DOP853", rtol=1e-12, atol=1e-9)
        v_ode = float(sol.y[0][-1]) if td < t_full else Vb
        res.add_check(check_close("dead time 끝 노드 전압: 전하 역함수 vs ODE 적분", v_ode, v_cf, 1e-7, "Q_node(v) = I·t_d 역함수(해석) vs dv/dt = I/C_node(v) 수치적분", True, "V", abs_scale=Vb))
    else:
        v_cf = 0.0
    res.add_metric("v_td", "dead time 끝 노드 전압 v(t_d)", v_cf, "V", basis=f"t_d = {td * 1e9:.4g} ns")
    res.add_metric("vds_on", "H측 turn-on 직전 V_DS = V_bus − v(t_d)", Vb - v_cf, "V")
    m_q = (abs(I) * td - Qreq) / Qreq if I > 0 else -1.0
    res.add_metric("mQ", "전하 여유 m_Q = (I·t_d − Q_req)/Q_req", m_q, "", basis="charge screen")
    E_hard = node.hard_on_loss(v_cf)
    res.add_metric("E_res", "잔류 전압 hard turn-on 손실 (용량 성분)", E_hard, "J", basis="고정 rail 에너지 수지 (채널 V·I overlap 제외)")
    res.add_metric("E_hard_full", "완전 hard turn-on 시 V_bus·Q_oss", node.hard_on_loss(0.0), "J", note="선형 C면 2·E_oss와 같지만 비선형 C에서는 다르다")
    res.add_metric("E_2eoss", "흔한 근사 2·E_oss", 2 * ref.E(Vb, C0, V0), "J")
    cls, why = classify(Vb, v_cf, False, I)
    res.add_metric("class", "판정 (charge screen)", CLASS_KO[cls], "", basis="정전류 가정 — 실제 i(t) 변화 미포함")
    # dead time map vs current
    tds = [100e-9, 200e-9, 300e-9, 500e-9]
    Is = np.linspace(0.25, 10.0, 60)
    for tdk in tds:
        ys = []
        for Ik in Is:
            qk = Ik * tdk
            ys.append(Vb - (node.v_of_q(qk) if qk < Qreq else Vb))
        res.add_series(f"map_{int(tdk * 1e9)}", f"t_d = {tdk * 1e9:.0f} ns", "V", Is.tolist(), ys)
    res.add_plot("p_map", "turn-on 직전 잔류 V_DS vs 전류 (dead time별)", [f"map_{int(t * 1e9)}" for t in tds], x_label="I (정전류 가정)", x_unit="A", y_label="V_DS(H) at turn-on", y_unit="V", kind="xy",
                 hlines=[{"y": NEAR_ZVS_FRAC * Vb, "label": "5 % V_bus (학습 기준)"}], vlines=[{"x": I, "label": f"I = {I:g} A"}], level="A",
                 proved="같은 전류라도 dead time이 짧으면 부분 전환이 되고, 전하 Q_node가 전류·시간과 맞아야 완전 전환된다는 것을 보였다.",
                 not_yet="정전류 가정이다. 실제 공진 전류는 dead time 동안 변하므로 실험 3(합성 commutation event)으로 따로 확인한다.")
    # node voltage trajectory within the dead time
    ts = np.linspace(0, max(td, t_full) * 1.1, 300)
    vv = []
    for tt in ts:
        q = abs(I) * tt if I > 0 else 0.0
        vv.append(node.v_of_q(q) if q < Qreq else Vb)
    res.add_series("v_node", "v_node (비선형 C, 정전류)", "V", ts.tolist(), vv)
    vv2 = [min(abs(I) * tt / (2 * Cs + Cpar), Vb) if I > 0 else 0.0 for tt in ts]
    res.add_series("v_node_pt", "v_node (Coss(V_bus) 한 점, 선형)", "V", ts.tolist(), vv2, dash=True)
    res.add_plot("p_v", "dead time 동안의 스위치 노드 전압", ["v_node", "v_node_pt"], y_label="v_node", y_unit="V", level="A", vlines=[{"x": td, "label": "t_d (turn-on)"}, {"x": t_full, "label": "도달 (비선형)"}],
                 proved="비선형 C에서는 저전압 구간의 큰 C 때문에 초기 상승이 느리고, 한 점 선형 모델보다 늦게 도달한다.", not_yet="정전류 가정의 전하 screen이다(SCREEN_ONLY). 전환 중 전류가 변하는 경로는 resonant_transition에서 따로 풀며, gate 동역학·링잉·역회복은 없다.")
    res.tables.append(
        Table(
            "t_dt",
            "dead time별 판정 (정전류 charge screen)",
            ["t_d [ns]", "v(t_d) [V]", "잔류 V_DS [V]", "m_Q", "판정", "잔류 hard-on 용량 손실 [µJ]"],
            [[tdk * 1e9, (node.v_of_q(abs(I) * tdk) if abs(I) * tdk < Qreq else Vb) if I > 0 else 0.0,
              Vb - ((node.v_of_q(abs(I) * tdk) if abs(I) * tdk < Qreq else Vb) if I > 0 else 0.0),
              (abs(I) * tdk - Qreq) / Qreq if I > 0 else -1.0,
              CLASS_KO[classify(Vb, (node.v_of_q(abs(I) * tdk) if abs(I) * tdk < Qreq else Vb) if I > 0 else 0.0, False, I)[0]],
              node.hard_on_loss((node.v_of_q(abs(I) * tdk) if abs(I) * tdk < Qreq else Vb) if I > 0 else 0.0) * 1e6] for tdk in tds],
            note="교재 요구: dead time 100/200/300/500 ns 비교. 5 % 기준은 학습용 분류이며 제조사 ZVS 보증 기준이 아니다.",
        )
    )
    res.verdict("SCREEN_ONLY", "정전류 charge screen: 전환 가능성의 필요조건 선별. 실제 converter의 ZVS PASS가 아니다.")
    if I <= 0:
        res.verdict("FAIL_CONSTRAINT", "전류 방향이 반대 — 전하가 원하는 노드로 가지 않는다 (hard turn-on)")
    res.assumptions += ["고정 rail(이상 전압원), 두 소자 동일 C(v)", "dead time 동안 주입전류 일정 (정전류 가정)", "gate 지연·채널 전류 감소 시간 0"]
    res.not_valid_for += ["실제 공진 전류가 변하는 converter의 ZVS 판정 (실험 3 필요)", "full bridge의 여러 노드·transformer 용량이 있는 회로에 2Q 계수를 그대로 적용", "역회복 손실"]
    res.interpretation = (
        f"두 소자가 붙은 half-bridge 노드는 아래 소자를 0→V_bus로 충전하고 위 소자를 V_bus→0으로 방전해야 하므로 필요한 전하가 2Q_oss = {Qreq * 1e9:.4g} nC다. "
        f"{abs(I):g} A 정전류라면 {t_full * 1e9:.4g} ns가 걸리는데 Coss(V_bus) 한 점으로 계산하면 {t_pt * 1e9:.4g} ns로 과소평가된다. "
        "이 charge screen은 ‘가능할 수도 있다’를 거르는 필요조건일 뿐, 실제 전류가 변하는 회로의 ZVS 통과가 아니다."
    )
    res.circuit = {"diagram": hb_circuit(v, cc=True).to_json(), "intervals": [], "plot_group": ""}
    return res


def hb_circuit(v: dict, cc: bool = False) -> Circuit:
    c = Circuit("hb_coss", 620, 330, title="Half-bridge 스위치 노드 (고정 rail)")
    vb = c.add("vsource", "Vb", 60, 165, 90, "V_bus", f"{v['Vb']:g} V")
    qh = c.add("nmos", "QH", 250, 100, 90, "Q_H", lpos=(216, 92, "end"))
    ch = c.add("capacitor", "CH", 320, 100, 90, "C_oss,H", "C(V_bus−v)")
    ql = c.add("nmos", "QL", 250, 230, 90, "Q_L", lpos=(216, 222, "end"))
    cl = c.add("capacitor", "CL", 320, 230, 90, "C_oss,L", "C(v)")
    if cc:
        src = c.add("isource", "I", 470, 165, 180, "I (정전류)", f"{v['I']:g} A")
        tail = src["a"]
    else:
        lx = c.add("inductor", "L", 420, 165, 0, "L", f"{v['L'] * 1e6:g} µH", lpos=(420, 136, "middle"))
        vx = c.add("vsource", "Vx", 520, 220, 90, "v_x", f"{v['vx']:g} V")
        tail = lx["a"]
        c.wire("w_lx", lx["b"], (520, 165), vx["a"])
        c.wire("w_vx_gnd", vx["b"], (520, 300), (250, 300))
    c.wire("w_rail", vb["a"], (60, 40), (250, 40), qh["a"])
    c.wire("w_ch_top", (250, 40), (320, 40), ch["a"])
    c.wire("w_node_h", qh["b"], (250, 165))
    c.wire("w_ch_bot", ch["b"], (320, 165))
    c.wire("w_node", (250, 165), (320, 165), tail if not cc else (440, 165))
    c.wire("w_node_l", (250, 165), ql["a"])
    c.wire("w_cl_top", (320, 165), cl["a"])
    c.wire("w_ql_gnd", ql["b"], (250, 300))
    c.wire("w_cl_gnd", cl["b"], (320, 300), (250, 300))
    c.wire("w_gnd", (250, 300), (60, 300), vb["b"])
    if cc:
        c.wire("w_isrc_gnd", src["a"], (540, 165), (540, 300), (320, 300))
    c.add("ground", "g", 250, 300)
    c.dot((250, 165), (320, 165), (250, 40), (250, 300), (320, 300))
    c.text(276, 182, "v (switch node)", "node")
    c.probe("pI", "i", 380, 153, "right" if cc else "left", "i")
    c.mode("Q_L on", "Q_L ON", ["QL", "w_node_l", "w_ql_gnd", "w_node", "L", "w_lx", "Vx", "w_vx_gnd", "w_gnd"], "노드 = 0 V, 인덕터 전류는 Q_L 채널로", dim=["QH"])
    c.mode("dead time", "dead time (둘 다 OFF)", ["CH", "CL", "w_node", "w_cl_top", "w_ch_bot", "w_ch_top", "w_cl_gnd", "L", "w_lx", "Vx", "w_rail", "Vb", "w_vx_gnd", "w_gnd"], "인덕터 전류가 C_oss,L을 충전·C_oss,H를 방전 → 노드 상승", dim=["QH", "QL"])
    c.mode("D_H clamp", "Q_H body diode 도통", ["QH", "w_node_h", "w_rail", "Vb", "w_node", "L", "w_lx", "Vx", "w_vx_gnd", "w_gnd"], "노드 = V_bus에 clamp: 이 때 켜면 ZVS (diode 도통 손실 발생)", dim=["QL", "CL"])
    c.mode("D_L clamp", "Q_L body diode 도통", ["QL", "w_node_l", "w_ql_gnd", "w_node", "L", "w_lx", "Vx"], "전류가 노드 밖으로: 노드가 0에 붙어 있다", dim=["QH"])
    c.mode("Q_H on", "Q_H ON", ["QH", "w_node_h", "w_rail", "Vb", "w_node", "L", "w_lx", "Vx"], "상측 도통", dim=["QL"])
    return c


def _transition_params():
    return _coss_params(
        [
            Param("Cpar", "노드 기생 용량 C_par", "F", 0.0, "pF", vmin=0, vmax=1e-7, source="ASSUMED", source_note="보드·transformer 등 추가 노드 용량 (기본 0)", group="노드"),
            Param("L", "commutation 인덕턴스 L", "H", 20e-6, "µH", vmin=1e-8, vmax=1e-2, source="ASSUMED", source_note="교재에 값 없음 (합성)", group="회로"),
            Param("vx", "인덕터 반대편 전압 v_x", "V", 400.0, "V", vmin=-2000, vmax=2000, source="ASSUMED", source_note="V_bus/2: 대칭 half-bridge, 0: buck형, V_bus: boost형", group="회로"),
            Param("I0", "turn-off 순간 전류 I₀ (노드로 들어가는 방향 +)", "A", 4.0, "A", vmin=-100, vmax=100, source="ASSUMED", source_note="교재의 4 A와 같은 크기", group="회로"),
            Param("td", "dead time 명령 t_d", "s", 200e-9, "ns", vmin=1e-9, vmax=5e-6, source="TEXTBOOK", source_note="100/200/300/500 ns 비교", group="타이밍"),
            Param("t_doff", "L측 gate-off 지연 (명령→실제)", "s", 0.0, "ns", vmin=0, vmax=1e-6, source="ASSUMED", group="타이밍"),
            Param("t_don", "H측 turn-on 지연 (명령→실제)", "s", 0.0, "ns", vmin=0, vmax=1e-6, source="ASSUMED", group="타이밍"),
            Param("t_res", "타이머 분해능 (dead time 양자화)", "s", 0.0, "ns", vmin=0, vmax=1e-6, source="ASSUMED", source_note="0이면 양자화 없음", group="타이밍"),
            Param("Vf", "body diode 순방향전압 V_f", "V", 3.5, "V", vmin=0, vmax=10, source="ASSUMED", source_note="SiC body diode 수준의 합성값; 역회복 미모델", group="노드"),
        ]
    )


def run_resonant(v: dict) -> Result:
    res = Result("EX02", "resonant_transition", "D (합성 commutation event)")
    C0, V0, Vb = v["C0"], v["V0"], v["Vb"]
    L, vx, I0, Cpar, Vf = v["L"], v["vx"], v["I0"], v["Cpar"], v["Vf"]
    td_cmd = v["td"]
    td = math.ceil(td_cmd / v["t_res"] - 1e-9) * v["t_res"] if v["t_res"] > 0 else td_cmd
    if td != td_cmd:
        res.warnings.append(f"타이머 분해능 {v['t_res'] * 1e9:g} ns로 dead time {td_cmd * 1e9:g} → {td * 1e9:g} ns (올림)")
    node = Node(Vb, C0, V0, Cpar)
    tr = simulate_transition(node, L, vx, I0, td, v["t_doff"], v["t_don"], Vf, tail=0.25 * td)
    cls, why = classify(Vb, tr.v_on, tr.returned, I0, tr.v_max, tr.t_reach is not None)
    res.add_metric("vds_on", "H측 turn-on 직전 V_DS", Vb - tr.v_on, "V", basis=f"실제 turn-on 시각 {tr.t_on * 1e9:.4g} ns")
    res.add_metric("class", "판정 (합성 소자 모델)", CLASS_KO[cls], "", note=why)
    res.add_metric("t_reach", "노드가 V_bus에 도달한 시각", tr.t_reach if tr.t_reach is not None else float("nan"), "s", basis="L측 turn-off 명령 기준")
    res.add_metric("clamp", "H측 body diode 도통 시간", tr.clamp_time, "s", basis="dead time이 길수록 증가")
    res.add_metric("E_diode", "diode 도통 손실 V_f·∫i dt", tr.E_diode, "J")
    res.add_metric("E_hard", "잔류 전압 hard turn-on 용량 손실", tr.E_hard, "J", basis="고정 rail 에너지 수지")
    res.add_metric("i_on", "turn-on 순간 인덕터 전류", tr.i_on, "A")
    res.add_metric("q_moved", "dead time 동안 인덕터가 옮긴 전하 ∫i dt", tr.q_moved, "C")
    Qreq = node.Qn(Vb)
    res.add_metric("Qreq", "필요 전하 Q_node(V_bus)", Qreq, "C")
    t_eff = td + v["t_don"] - v["t_doff"]
    q_screen = I0 * t_eff
    res.add_metric("mQ_screen", "전하 screen 여유 (I₀·t_d,eff − Q_req)/Q_req", (q_screen - Qreq) / Qreq if I0 > 0 else -1.0, "", basis="SCREEN_ONLY: 전류를 I₀로 고정한 손계산", note="event 결과(turn-on 직전 V_DS)와 다를 수 있다 — 전류가 변하므로")
    # energy screens
    E_avail = 0.5 * L * I0 * I0
    E_need = Vb * ref.Q(Vb, C0, V0) + 0.5 * Cpar * Vb**2 - vx * Qreq
    res.add_metric("E_avail", "½·L·I₀²", E_avail, "J")
    res.add_metric("E_2eoss", "흔한 기준 2·E_oss", 2 * ref.E(Vb, C0, V0), "J", note="회로와 무관하게 고정하면 틀릴 수 있다")
    res.add_metric("E_need", "이 회로의 필요 에너지 V_bus·Q_oss + ½C_par V² − v_x·Q_node", E_need, "J", note="v_x = V_bus/2이면 0: 에너지가 아니라 시간이 제약")
    led = tr.ledger
    res.add_check(
        Check(
            "에너지 잔차 (source − ΔW_L − ΔW_C − rail − diode)",
            "PASS" if abs(led["normalised"]) < 1e-6 else "FAIL",
            led["normalised"],
            "rel",
            1e-6,
            path="v_x·∫i, V_bus·(rail 전하), ½Li², E_node(v) — 상태식과 독립적으로 정의한 에너지 항",
            independent=True,
            detail=f"source {led['E_source'] * 1e6:.5g} µJ, ΔW_L {led['dW_L'] * 1e6:.5g} µJ, ΔW_C {led['dW_C'] * 1e6:.5g} µJ, rail {led['E_rail'] * 1e6:.5g} µJ, diode {led['E_diode'] * 1e6:.4g} µJ → 잔차 {led['residual']:.3g} J",
        )
    )
    # independent solver: RK4 in charge coordinates (positive-current, no return cases)
    if I0 > 0 and not tr.returned and tr.v_max <= tr.v_on + 1e-9 * Vb:
        errs = []
        for n in (400, 800, 1600):
            vr, ir = rk4_charge_path(node, L, vx, I0, td, n, v["t_doff"], v["t_don"], Vf)
            errs.append(max(abs(vr - tr.v_on) / Vb, abs(ir - tr.i_on) / max(abs(I0), 1e-3)))
        res.add_check(
            Check(
                "독립 solver: 전하좌표 RK4 (h 반감 수렴)",
                "PASS" if errs[-1] < 2e-4 and errs[-1] <= errs[0] else "FAIL",
                errs[-1],
                "rel",
                2e-4,
                path="(q, i) 상태·고정 step RK4·v = Q⁻¹(q) vs (v, i) 상태·DOP853 적응 step",
                independent=True,
                detail="오차 (n = 400/800/1600): " + ", ".join(f"{e:.2e}" for e in errs) + " — clamp 진입은 step 내 선형보간이라 그 경우 1차 수렴",
            )
        )
    else:
        res.add_check(Check("독립 solver: 전하좌표 RK4", "NOT_RUN", path="양의 전류·되돌아옴 없는 경우만 비교", detail="노드가 되돌아오는 경우는 이 독립 경로가 다루지 않아 비교하지 않음 (에너지 잔차·해석 한계해로 검증)"))
    # limiting case: constant C -> analytic LC solution
    Cf = ref.co_tr(Vb, C0, V0)
    nf = Node(Vb, C0, V0, Cpar, model="fixed", Cfix=Cf)
    t_probe = min(td, 0.5 * (tr.t_reach or td))
    trf = simulate_transition(nf, L, vx, max(I0, 1e-3), t_probe, 0.0, 0.0, 0.0)
    Cn = 2 * Cf + Cpar
    w = 1 / math.sqrt(L * Cn)
    v_an = vx + (0 - vx) * math.cos(w * t_probe) + max(I0, 1e-3) / (Cn * w) * math.sin(w * t_probe)
    if trf.v_on < Vb - 1e-6 and trf.v_on > 1e-6:
        res.add_check(check_close("한계 경우: 고정 C에서 ODE vs 해석적 LC 해", trf.v_on, v_an, 1e-7, "DOP853 ODE vs v(t) = v_x − v_x cos ωt + I₀/(Cω) sin ωt", True, "V", abs_scale=Vb))
    # model comparison table: fixed C (single point), C_o(tr), C_o(er), nonlinear, + parasitic
    rows = []
    variants = [
        ("비선형 C(v)", Node(Vb, C0, V0, Cpar)),
        ("고정 C = Coss(V_bus) (한 점)", Node(Vb, C0, V0, Cpar, "fixed", ref.C(Vb, C0, V0))),
        ("고정 C = C_o(tr)", Node(Vb, C0, V0, Cpar, "fixed", ref.co_tr(Vb, C0, V0))),
        ("고정 C = C_o(er)", Node(Vb, C0, V0, Cpar, "fixed", ref.co_er(Vb, C0, V0))),
        ("비선형 + C_par 100 pF", Node(Vb, C0, V0, Cpar + 100e-12)),
    ]
    for name, nd in variants:
        tt = simulate_transition(nd, L, vx, I0, td, v["t_doff"], v["t_don"], Vf)
        c2, _ = classify(Vb, tt.v_on, tt.returned, I0, tt.v_max, tt.t_reach is not None)
        rows.append([name, (tt.t_reach * 1e9) if tt.t_reach else "미도달", Vb - tt.v_on, CLASS_KO[c2], tt.E_hard * 1e6])
    res.tables.append(Table("t_models", f"같은 조건(t_d = {td * 1e9:.4g} ns)에서 용량 모델별 결론", ["모델", "도달 시각 [ns]", "turn-on V_DS [V]", "판정", "잔류 손실 [µJ]"], rows, note="어떤 결론이 모델 선택에 따라 바뀌는지 본다. 한 점 Coss는 전환시간을 과소평가해 ‘ZVS 된다’는 잘못된 결론을 줄 수 있다."))
    # dead-time sweep for this operating point: residual loss vs diode loss
    tds = np.linspace(max(20e-9, 0.2 * td), max(3 * td, 1.6 * (tr.t_reach or td)), 40)
    e_hard, e_dio, vres = [], [], []
    for tdk in tds:
        tk = simulate_transition(node, L, vx, I0, float(tdk), v["t_doff"], v["t_don"], Vf)
        e_hard.append(tk.E_hard)
        e_dio.append(tk.E_diode)
        vres.append(Vb - tk.v_on)
    res.add_series("sw_hard", "잔류 hard-on 용량 손실", "J", tds.tolist(), e_hard)
    res.add_series("sw_diode", "body diode 도통 손실", "J", tds.tolist(), e_dio)
    res.add_series("sw_sum", "합 (이 두 항만)", "J", tds.tolist(), [a + b for a, b in zip(e_hard, e_dio)], dash=True)
    res.add_series("sw_vres", "turn-on 직전 V_DS", "V", tds.tolist(), vres)
    res.add_plot("p_sweep", "dead time에 따른 두 손실 항 (이 운전점)", ["sw_hard", "sw_diode", "sw_sum"], x_label="dead time t_d", x_unit="s", y_label="에너지/사건", y_unit="J", kind="xy", vlines=[{"x": td, "label": "현재 t_d"}], level="D",
                 proved="dead time이 짧으면 잔류 전압 hard-on 손실, 길면 body diode 도통 손실이 커진다. 최적은 한 점이 아니라 이 운전점·지연에 따라 움직이는 범위다.",
                 not_yet="역회복·채널 V·I overlap·gate 동역학·온도 의존 Q_oss는 모델에 없다. 다른 전류·전압 운전점에서는 범위가 달라진다.")
    res.add_plot("p_vres", "dead time에 따른 turn-on 직전 V_DS", ["sw_vres"], x_label="dead time t_d", x_unit="s", y_label="V_DS(H)", y_unit="V", kind="xy", vlines=[{"x": td, "label": "현재 t_d"}], hlines=[{"y": NEAR_ZVS_FRAC * Vb, "label": "5 % (학습 기준)"}], level="D",
                 proved="이 운전점에서 완전 전환에 필요한 최소 dead time을 파형 기반으로 찾았다.", not_yet="gate 지연 분포·타이머 양자화·온도에 대한 여유는 따로 더해야 한다.")
    # time waveforms
    res.add_series("v", "v_node", "V", tr.t, tr.v)
    res.add_series("i", "i_L (노드로 +)", "A", tr.t, tr.i)
    res.add_series("vds_h", "V_DS(H) = V_bus − v", "V", tr.t, [Vb - x for x in tr.v])
    bands = []
    for k in range(len(tr.t) - 1):
        ph = tr.phase[k]
        if bands and bands[-1]["mode"] == ph and abs(bands[-1]["x1"] - tr.t[k]) < 1e-15:
            bands[-1]["x1"] = tr.t[k + 1]
        elif tr.t[k + 1] > tr.t[k]:
            bands.append({"x0": tr.t[k], "x1": tr.t[k + 1], "mode": ph, "label": {"Q_L on": "Q_L ON", "dead time": "dead time", "D_H clamp": "D_H clamp", "D_L clamp": "D_L clamp", "Q_H on": "Q_H ON"}.get(ph, ph)})
    vl = [{"x": td, "label": "H측 on 명령"}]
    if tr.t_reach:
        vl.append({"x": tr.t_reach, "label": "V_bus 도달"})
    res.add_plot("p_tv", "스위치 노드 전압과 H측 V_DS (합성 event)", ["v", "vds_h"], y_label="전압", y_unit="V", bands=bands, vlines=vl, group="tr", level="D",
                 hlines=[{"y": NEAR_ZVS_FRAC * Vb, "label": "5 %"}],
                 proved="dead time 동안 실제로 변하는 전류로 노드를 움직여 turn-on 직전 V_DS를 사건 단위로 보고했다.",
                 not_yet="합성 C(v)·이상 diode clamp·즉시 채널 전환 가정의 D 수준 경향 결과다. 실제 소자 turn-on 파형·링잉·EMI는 아니다.")
    res.add_plot("p_ti", "인덕터 전류 (dead time 동안 변한다)", ["i"], y_label="i", y_unit="A", bands=bands, vlines=vl, group="tr", level="D", hlines=[{"y": 0.0, "label": "0 A"}],
                 proved="정전류 가정이 깨지는 정도를 보여준다: v_x에 따라 전류가 늘거나 줄어든다.", not_yet="인덕터와 합성 C(v) 노드만의 전환이다. 실제 소자의 gate turn-off 지연, 채널 전류의 tail, 배선 L의 링잉은 포함하지 않았다.")
    res.tables.append(Table("t_events", "사건 타임라인 (turn-off 명령 기준)", ["시각 [ns]", "사건", "값"], [[e[0] * 1e9, e[1], e[2]] for e in tr.events], note="명령·실제 gate-off·노드 이동·rail 도달·turn-on 명령·실제 turn-on을 구분한다."))
    res.circuit = {"diagram": hb_circuit(v).to_json(), "intervals": bands, "plot_group": "tr"}
    if cls in ("ZVS", "ZVS_LATE_RETURN", "NEAR_ZVS"):
        res.verdict("PASS_WITHIN_MODEL", f"{CLASS_KO[cls]}: 합성 소자·이 dead time·이 전류에서만. converter 전체의 ZVS PASS가 아니다 (charge/energy screen 수치는 SCREEN_ONLY 참고값).")
    else:
        res.verdict("FAIL_CONSTRAINT", f"{CLASS_KO[cls]}: {why}")
    res.assumptions += [
        "고정 rail, 두 소자 동일 합성 C(v), 선형 인덕터, 인덕터 반대편 전압 v_x 일정",
        "채널은 gate-off 순간 전류를 즉시 끊음 (turn-off 손실·gate 동역학 없음)",
        "body diode = 이상 diode + V_f (역회복 없음)",
        "hard turn-on 손실은 용량 에너지 수지만 (V·I overlap 제외)",
    ]
    res.not_valid_for += ["특정 소자의 ZVS 보증·turn-on 손실 정밀값", "링잉·dv/dt·EMI", "역회복", "온도·lot 편차"]
    res.interpretation = (
        f"인덕터 반대편 전압 v_x = {vx:g} V에서 전류는 dead time 동안 {I0:g} A → {tr.i_on:.3g} A로 변했다. "
        f"필요 전하 {Qreq * 1e9:.4g} nC를 옮기는 데 걸린 시간이 dead time보다 {'짧아' if tr.t_reach and tr.t_reach <= tr.t_on else '길어'} "
        f"turn-on 직전 V_DS는 {Vb - tr.v_on:.4g} V였다. v_x가 V_bus/2이면 인덕터가 잃는 에너지가 0이라 ‘½LI² ≥ 2E_oss’ 같은 에너지 기준은 판정 근거가 될 수 없고, "
        "시간(전하)과 전류 부호가 판정한다."
    )
    return res


def run_deadtime_map(v: dict) -> Result:
    res = Result("EX02", "deadtime_window", "D (합성 commutation event)")
    C0, V0, Vb = v["C0"], v["V0"], v["Vb"]
    node = Node(Vb, C0, V0, v["Cpar"])
    L, vx, Vf = v["L"], v["vx"], v["Vf"]
    Is = np.linspace(v["I_min"], v["I_max"], 14)
    tds = np.linspace(40e-9, 600e-9, 29)
    spread = v["spread"]
    grid = []
    for I0 in Is:
        row = []
        for tdk in tds:
            tk = simulate_transition(node, L, vx, float(I0), float(tdk), 0.0, 0.0, Vf)
            c, _ = classify(Vb, tk.v_on, tk.returned, I0, tk.v_max, tk.t_reach is not None)
            row.append((c, Vb - tk.v_on, tk.E_diode, tk.E_hard))
        grid.append(row)
    # minimum dead time for complete transition per current, with delay spread margin
    t_min, t_max = [], []
    for k, I0 in enumerate(Is):
        ok = [tds[j] for j in range(len(tds)) if grid[k][j][0] in ("ZVS", "NEAR_ZVS")]
        t_min.append(min(ok) if ok else float("nan"))
        lost = [tds[j] for j in range(len(tds)) if grid[k][j][0] in ("LOST", "ZVS_LATE_RETURN") and ok and tds[j] > min(ok)]
        t_max.append(min(lost) if lost else float("nan"))
    res.add_series("tmin", "완전 전환 최소 t_d (격자)", "s", Is.tolist(), t_min)
    res.add_series("tmin_sp", f"최소 t_d + 지연 산포 {spread * 1e9:g} ns", "s", Is.tolist(), [t + spread if t == t else float("nan") for t in t_min], dash=True)
    res.add_series("tmax", "되돌아와 ZVS를 잃기 시작하는 t_d", "s", Is.tolist(), t_max)
    lo_I = Is[0]
    need = max(t for t in t_min if t == t) if any(t == t for t in t_min) else float("nan")
    res.add_metric("td_need", f"전류 범위 {lo_I:g}–{Is[-1]:g} A 모두 완전 전환에 필요한 t_d (격자, 산포 포함)", need + spread if need == need else float("nan"), "s", basis="가장 작은 전류가 지배")
    res.add_metric("grid", "격자", f"I {len(Is)}점 × t_d {len(tds)}점 = {len(Is) * len(tds)}회 event 시뮬레이션", "")
    res.add_plot("p_win", "운전 전류별 dead time 창", ["tmin", "tmin_sp", "tmax"], x_label="turn-off 전류 I₀", x_unit="A", y_label="dead time", y_unit="s", kind="xy", level="D",
                 proved="dead time은 한 점의 최적값이 아니라 전류(운전점)와 지연 산포에 따라 움직이는 창이며, 가장 작은 전류가 최소 dead time을 지배한다.",
                 not_yet="합성 C(v) 하나와 v_x 하나에 대한 결과다. 온도별 Q_oss·gate 지연 분포의 실제 근거는 MISSING_INPUT.")
    rows = []
    for k, I0 in enumerate(Is[::3]):
        kk = k * 3
        rows.append([float(I0)] + [CLASS_KO[grid[kk][j][0]] for j in (1, 4, 8, 12, 20)])
    res.tables.append(Table("t_grid", "판정 격자 (일부)", ["I₀ [A]"] + [f"{tds[j] * 1e9:.0f} ns" for j in (1, 4, 8, 12, 20)], rows))
    res.verdict("SCREEN_ONLY", "합성 소자 event 격자 — 설계 창의 경향을 보이는 학습 결과이며 제조사 ZVS 조건이 아니다.")
    res.assumptions += ["v_x·L 고정, 합성 C(v)", "gate 지연 산포는 입력 하나로 더함 (분포 미모델)"]
    res.not_valid_for += ["실제 소자·온도에서의 dead time 확정"]
    res.interpretation = "작은 전류일수록 같은 전하를 옮기는 데 오래 걸리므로 최소 dead time은 경부하에서 정해지고, 큰 전류에서는 긴 dead time이 diode 도통을 늘린다. 그래서 dead time은 운전점별 창(또는 전류 의존 스케줄)으로 설계한다."
    return res


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "데이터시트 Coss 하나로 dead time을 정하면 왜 틀립니까?",
        "Coss는 전압 의존 미분용량이다. 시간은 전하 적분 ∫C dv, 저장에너지는 ∫v·C dv로 구한다. 동일 두 소자의 고정 rail half-bridge·정전류 가정에서는 완전 전환 전하가 2Q_oss이지만 "
        "실제 전류 변화, rail/source 에너지 교환, 다른 용량, diode 구간을 넣으면 더 상세한 모델이 필요하다. C_o(er)와 C_o(tr)를 바꾸어 쓰지 않는다.",
        "Why is one Coss value from the datasheet not enough to set the dead time?",
        "Coss is a voltage-dependent differential capacitance. Time needs the charge integral and stored energy needs the voltage-weighted integral. With fixed rails, two identical devices and a constant current the transition needs 2·Qoss, but a real commutation current changes, energy is exchanged with the rails, and other capacitances add charge.",
        ["미분용량", "전하 적분 = 시간", "C_o(tr) vs C_o(er)", "2Q는 가정의 결과"],
        kind="pressure",
    ),
    Question(
        "ZVS면 스위칭 손실이 0인가?",
        "아니다. turn-on 잔류 전압이 있으면 용량 손실이 남고, turn-off 손실·gate 손실·diode 도통·순환전류 전도손실은 그대로다. ZVS를 위해 전류를 키우면 RMS와 자성체 손실이 늘 수 있다.",
        "Does ZVS mean zero switching loss?",
        "No. Any residual voltage at turn-on still dissipates capacitive energy, and turn-off loss, gate loss, body-diode conduction and the conduction loss of the circulating current remain. Extra current bought for ZVS raises RMS and magnetic losses.",
        ["turn-on 잔류", "turn-off 손실", "gate·diode", "순환전류 비용"],
    ),
    Question(
        "Lm을 줄여 ZVS를 확보했으니 최적화 완료인가?",
        "전하 여유가 커진 이유와 함께 RMS, turn-off 손실, core/winding 손실, 전부하 온도를 다시 계산한다. 증가한 전류의 부작용을 정량화하지 못하면 ZVS만 최적화한 것이다.",
        "You reduced Lm to get ZVS. Are you done?",
        "Not yet. I would recompute the RMS current, turn-off loss, core and winding loss and the full-load temperature, because the extra magnetizing current that buys charge margin costs loss elsewhere.",
        ["RMS", "turn-off", "자성체", "온도"],
        kind="pressure",
    ),
    Question(
        "‘½LI² ≥ 2E_oss’는 항상 맞는 ZVS 조건인가?",
        "아니다. 필요한 에너지는 인덕터 반대편 전압과 rail 에너지 교환에 달려 있다. 고정 rail half-bridge에서 v_x = 0이면 V_bus·Q_oss가 필요하고, v_x = V_bus/2이면 인덕터가 잃는 에너지가 0이라 시간(전하)과 전류 부호가 판정한다.",
        "Is ‘½·L·I² ≥ 2·Eoss’ always the ZVS condition?",
        "No. The energy the inductor must supply depends on the voltage at its other end and on the energy exchanged with the rails. With the far end at mid-rail the inductor loses no net energy, so time and current direction decide instead.",
        ["회로 의존", "v_x 영향", "시간·전하가 판정"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="qe_integrals",
        title="Coss·Qoss·Eoss는 세 가지 다른 함수다",
        goal="C(v) = 2 nF/√(1+v/40 V)에서 Q_oss(800 V) = 573.212 nC, E_oss = 180.238 µJ를 적분하고, C_o(tr)·C_o(er)·Coss(800 V) 한 점이 서로 다름을 확인한다.",
        params=_coss_params(),
        presets=[Preset("textbook", "교재 C(v), 800 V", {}, "E02", ("nominal", "reference")), Preset("v400", "V_bus = 400 V", {"Vb": 400.0}, "", ("variant",))],
        run=run_qe,
        model_level="A",
        suggested_change="V_bus를 800 → 400 V로 바꿔 한 점 Coss 근사의 오차가 어떻게 변하는지 본다.",
        prediction=Prediction(
            "Coss(800 V) = 436 pF 한 점으로 ½CV²를 계산하면 실제 E_oss(800 V)보다?",
            ["크다", "작다", "같다", "모르겠다"],
            "작다",
            "저전압 구간의 C가 훨씬 크므로 ∫v·C dv가 ½·C(800)·800²보다 크다. 전하는 더 크게 과소평가된다(Q가 저전압의 큰 C를 가중치 없이 적분하므로).",
            ["Qoss", "Eoss", "E_pt"],
            handcalc=[{"key": "Qoss", "label": "Q_oss(800 V)", "unit": "C"}, {"key": "Eoss", "label": "E_oss(800 V)", "unit": "J"}],
        ),
        suggested={"Vb": 400.0},
        student="커패시터에 전하를 넣을 때 필요한 전하량과 저장되는 에너지는 다른 양이다. 전압에 따라 C가 변하면 두 양 모두 적분해야 하고, 한 점의 C로 곱하면 틀린다.",
        expert="C_o(tr)는 정해진 전압까지 같은 전하(→ 같은 시간)를 주는 등가 선형 용량이고, C_o(er)는 같은 저장에너지를 주는 등가 용량이다. dead time·ZVS 전하에는 C_o(tr), hard-switching 손실의 용량 성분에는 회로 경로에 맞는 에너지 수지를 쓴다. Eon 데이터에 Coss 성분이 포함됐는지 확인하지 않고 E_oss를 더하면 중복 계산이 된다.",
        customer_ko="데이터시트의 Coss 한 점으로 계산하시면 필요한 전하를 약 40 % 과소평가합니다. dead time은 Q_oss(또는 C_o(tr))로, 손실은 회로에 맞는 에너지 수지로 보시죠.",
        customer_en="If you use the single Coss value from the datasheet you underestimate the required charge by about forty percent. I would size the dead time from Qoss, or Co(tr), and treat the capacitive loss with an energy balance that matches your circuit.",
        questions=_Q[:1],
        circuit="hb_coss",
        textbook=[TB_E02, TB_04],
        reference_presets=["textbook"],
        claim_limit="합성 C(v)의 적분값. 실제 소자 곡선 아님.",
    ),
    Experiment(
        key="hb_constant_current",
        title="고정 rail half-bridge: 2Q_oss와 dead time (charge screen)",
        goal="두 동일 소자가 붙은 고정 rail half-bridge에서 C_node(v) = C(v) + C(800 − v), 4 A 정전류의 완전 전환시간 286.606 ns를 Coss(800 V) 한 점 계산 174.574 ns와 비교하고, dead time 100/200/300/500 ns에서 부분 전환을 확인한다.",
        params=_coss_params(
            [
                Param("I", "주입전류 I (노드로 +)", "A", 4.0, "A", vmin=-100, vmax=100, source="TEXTBOOK", source_note="4 A", group="회로"),
                Param("td", "dead time t_d", "s", 200e-9, "ns", vmin=1e-9, vmax=5e-6, source="TEXTBOOK", source_note="100/200/300/500 ns", group="타이밍"),
                Param("Cpar", "노드 기생 용량 C_par", "F", 0.0, "pF", vmin=0, vmax=1e-7, source="ASSUMED", group="노드"),
            ]
        ),
        presets=[
            Preset("textbook", "4 A, t_d 200 ns", {}, "E02", ("nominal", "reference")),
            Preset("td300", "t_d 300 ns", {"td": 300e-9}, "", ("variant",)),
            Preset("neg", "전류 반대 (−4 A)", {"I": -4.0}, "전하가 원하는 노드로 가지 않음", ("failure",)),
            Preset("low_i", "1 A, t_d 500 ns", {"I": 1.0, "td": 500e-9}, "경부하", ("corner",)),
        ],
        run=run_hb_cc,
        model_level="A (charge screen)",
        suggested_change="dead time을 200 → 300 ns로 바꾼다 (전류 4 A 유지).",
        prediction=Prediction(
            "4 A 정전류에서 dead time을 200 → 300 ns로 늘리면 H측 turn-on 직전 V_DS는?",
            ["여전히 수백 V (부분 전환)", "0 V (완전 전환 후 diode clamp)", "음수가 된다", "모르겠다"],
            "0 V (완전 전환 후 diode clamp)",
            "완전 전환에 2Q_oss/I = 286.6 ns가 필요하다. 300 ns면 13.4 ns 동안 body diode가 clamp한 뒤 turn-on한다. 200 ns에서는 전하가 부족해 부분 전환이다.",
            ["vds_on", "v_td", "mQ"],
            handcalc=[{"key": "t_trans", "label": "완전 전환시간", "unit": "s"}],
        ),
        suggested={"td": 300e-9},
        student="아래 소자는 0→800 V로 충전되고 위 소자는 800→0 V로 방전되어야 노드가 위 rail에 닿는다. 두 소자의 전하를 모두 옮겨야 하므로 한 소자 전하의 두 배가 필요하다.",
        expert="2Q_oss는 고정 rail·동일 소자·정전류라는 가정의 결과다. full bridge의 여러 노드, transformer 권선 용량, 비대칭 소자에서는 계수가 달라진다. 이 screen의 PASS는 ‘실제 전류로 따로 확인할 가치가 있다’는 뜻이지 ZVS 통과가 아니다.",
        customer_ko="4 A에서 완전 전환에 약 287 ns가 필요합니다. 현재 200 ns dead time이면 부분 전환이 되고, 한 점 Coss로 계산한 175 ns는 이를 가립니다. 실제 전류 파형으로 turn-on 직전 V_DS를 측정해 확정하시죠.",
        customer_en="At 4 A the node needs about 287 ns to swing fully, so a 200 ns dead time gives a partial transition. A single-point Coss estimate of 175 ns hides that. Let's confirm with the drain-source voltage measured just before turn-on.",
        questions=_Q[:2],
        circuit="hb_coss",
        textbook=[TB_E02],
        reference_presets=["textbook", "td300", "neg"],
        claim_limit="정전류 charge screen (SCREEN_ONLY).",
    ),
    Experiment(
        key="resonant_transition",
        title="실제로 변하는 전류로 commutation event 풀기 (합성 D 수준)",
        goal="dead time 동안 인덕터 전류가 변하는 회로(인덕터 반대편 v_x)를 state로 추가해 노드 이동·diode clamp·turn-on 직전 V_DS를 사건 단위로 풀고, 에너지 잔차·독립 solver·고정 C 한계해로 검증한다. 같은 조건에서 용량 모델(한 점·C_o(tr)·C_o(er)·비선형·+기생)에 따라 결론이 바뀌는지 본다.",
        params=_transition_params(),
        presets=[
            Preset("mid", "v_x = V_bus/2, 4 A, t_d 200 ns", {}, "대칭 half-bridge", ("nominal", "reference")),
            Preset("mid300", "v_x = V_bus/2, t_d 300 ns", {"td": 300e-9}, "", ("variant", "reference")),
            Preset("vx0", "v_x = 0 V (buck형)", {"vx": 0.0, "td": 500e-9}, "에너지가 부족하면 되돌아온다", ("failure", "reference")),
            Preset("delays", "gate 지연 20/15 ns, 타이머 10 ns", {"t_doff": 20e-9, "t_don": 15e-9, "t_res": 10e-9, "td": 285e-9}, "명령과 실제 시각 분리", ("variant",)),
            Preset("long", "t_d 1 µs (과다)", {"td": 1e-6}, "diode 도통 증가", ("corner",)),
        ],
        run=run_resonant,
        model_level="D (합성)",
        suggested_change="v_x를 400 → 0 V로 바꾼다 (같은 ½LI²에서 회로만 바꿈; t_d 500 ns).",
        prediction=Prediction(
            "½LI² = 160 µJ은 그대로 두고 v_x만 400 → 0 V로 바꾸면(t_d 500 ns) 노드는?",
            ["여전히 완전 전환", "올라가다 되돌아와 ZVS 실패", "더 빨리 전환", "모르겠다"],
            "올라가다 되돌아와 ZVS 실패",
            "v_x = 0이면 노드가 올라갈수록 인덕터 전압이 음이 되어 전류가 줄고, 인덕터는 V_bus·Q_oss = 459 µJ를 공급해야 한다. 160 µJ로는 부족해 노드가 되돌아온다. v_x = V_bus/2에서는 순에너지가 0이라 에너지가 판정 기준이 아니었다.",
            ["vds_on", "class", "E_need"],
            handcalc=[{"key": "Qreq", "label": "필요 전하 Q_node(V_bus)", "unit": "C"}],
        ),
        suggested={"vx": 0.0, "td": 500e-9},
        student="인덕터는 전류를 유지하려 하므로 스위치가 꺼지면 그 전류가 노드 커패시턴스를 충전한다. 그런데 노드 전압이 변하면 인덕터 양단 전압도 변해 전류가 늘거나 줄어든다. 그래서 ‘정전류’는 근사일 뿐이다.",
        expert="판정 절차: ① 명령·실제 gate-off·노드 이동 시작·rail 도달·turn-on 명령·실제 turn-on 시각 분리 ② 사건마다 전류 방향·turn-on 직전 V_DS·옮긴 전하·diode clamp 시간·잔류 용량 손실 기록 ③ 고정 C·비선형 C·기생 C 모델 비교 ④ gate 지연 최소/최대·dead time 양자화·온도별 Q_oss 근거 반영 ⑤ 5 % 같은 학습 기준과 제조사·고객 기준 분리. 이 모델은 채널 즉시 차단·이상 diode·합성 C(v) 가정이라 D 수준이지만 합성 경향 결과다.",
        customer_ko="현재 운전점에서 dead time 동안 전류가 줄어들어 노드가 위 rail에 닿지 못합니다. 전하·시간 기준으로 보면 dead time이나 전류 여유가 필요하고, 에너지 한 식(½LI² ≥ 2E_oss)으로는 판정하지 않습니다. turn-on 직전 V_DS와 전류를 같은 timebase로 측정해 확정하시죠.",
        customer_en="At this operating point the current decays during the dead time and the node does not reach the upper rail. The decision should rest on charge and timing, not on a single energy inequality. Let's measure the drain-source voltage just before turn-on together with the current on the same timebase.",
        questions=_Q[1:],
        circuit="hb_coss",
        textbook=[TB_E02, TB_E13],
        reference_presets=["mid", "mid300", "vx0"],
        runtime_hint="seconds",
        claim_limit="합성 C(v)·이상 diode·즉시 채널 차단의 D 수준 경향. 실제 소자 ZVS 보증 아님.",
    ),
    Experiment(
        key="deadtime_window",
        title="dead time은 한 점이 아니라 운전점별 창이다",
        goal="turn-off 전류(운전점)를 바꿔가며 완전 전환에 필요한 최소 dead time과 되돌아오기 시작하는 dead time을 합성 event 격자로 찾고, gate 지연 산포를 더한 설계 창을 본다.",
        params=_transition_params()[:4]
        + [
            Param("L", "commutation 인덕턴스 L", "H", 20e-6, "µH", vmin=1e-8, vmax=1e-2, source="ASSUMED", group="회로"),
            Param("vx", "인덕터 반대편 전압 v_x", "V", 400.0, "V", vmin=-2000, vmax=2000, source="ASSUMED", group="회로"),
            Param("Vf", "body diode V_f", "V", 3.5, "V", vmin=0, vmax=10, source="ASSUMED", group="노드"),
            Param("I_min", "전류 범위 최소", "A", 0.5, "A", vmin=0.05, vmax=100, source="ASSUMED", group="운전점"),
            Param("I_max", "전류 범위 최대", "A", 8.0, "A", vmin=0.1, vmax=200, source="ASSUMED", group="운전점"),
            Param("spread", "gate 지연 산포 (최악 합)", "s", 20e-9, "ns", vmin=0, vmax=1e-6, source="ASSUMED", source_note="분포 근거 없음 — 학습용 여유", group="타이밍"),
        ],
        presets=[Preset("mid", "v_x = V_bus/2, 0.5–8 A", {}, "", ("nominal", "reference")), Preset("vxb", "v_x = V_bus (boost형)", {"vx": 800.0}, "", ("variant",))],
        run=run_deadtime_map,
        model_level="D (합성)",
        suggested_change="전류 범위 최소를 0.5 → 2 A로 올려 (경부하 제외) 필요한 dead time이 어떻게 줄어드는지 본다.",
        prediction=Prediction(
            "경부하(작은 전류)를 운전범위에서 빼면 모든 전류에서 완전 전환에 필요한 dead time은?",
            ["늘어난다", "줄어든다", "그대로", "모르겠다"],
            "줄어든다",
            "같은 전하를 옮기는 데 전류가 작을수록 오래 걸리므로, 최소 dead time은 가장 작은 전류가 정한다.",
            ["td_need"],
        ),
        suggested={"I_min": 2.0},
        student="작은 전류로 같은 전하를 옮기려면 더 오래 걸린다. 그래서 경부하에서 dead time이 부족해지기 쉽다.",
        expert="전류 의존 dead time 스케줄, 경부하 burst, 또는 순환전류를 의도적으로 남기는 변조(EX06)가 대안이다. 각 대안의 손실·제어 복잡도를 같은 기준으로 비교한다.",
        customer_ko="dead time은 전부하 한 점에서 정하면 경부하에서 부분 전환이 됩니다. 운전 전류 범위와 gate 지연 산포를 포함한 창으로 정하거나 전류에 따라 스케줄하는 방안을 검토하시죠.",
        customer_en="A dead time chosen at full load gives partial transitions at light load. I would set it as a window over the operating-current range, including the gate-delay spread, or schedule it with the current.",
        questions=[],
        circuit="hb_coss",
        textbook=[TB_E02],
        reference_presets=["mid"],
        runtime_hint="seconds",
        claim_limit="합성 event 격자 (SCREEN_ONLY).",
    ),
]

LAB = Lab(
    id="EX02",
    title="비선형 Coss·dead time·ZVS — 에너지식 하나로 판정하지 않기",
    title_en="Nonlinear Coss, dead time and ZVS",
    track="expert",
    order=2,
    path_note="E13 1회전 (E01–E03)",
    textbook=[TB_E02, TB_04, TB_E13],
    prerequisites=["FL02", "FL08"],
    summary="Q/E 적분 → 고정 rail half-bridge charge screen → 전류가 변하는 합성 commutation event → 운전점별 dead time 창. screen과 event 결과를 섞지 않는다.",
    experiments=EXPERIMENTS,
    minimum_scope="Q/E 적분, deadtime·current sign·partial ZVS (E13 표); 286.606 ns vs 174.574 ns; 100/200/300/500 ns; 실제 i(t) 변화 경로는 회로·state를 추가해 별도 검증",
    claim_limits=[
        "charge screen은 SCREEN_ONLY — 실제 converter ZVS PASS 아님",
        "합성 C(v)의 D 수준 경향 결과이며 vendor 소자 모델이 아님",
        "2E_oss를 모든 ZVS topology의 보편 최소에너지로 쓰지 않음",
        "역회복·gate 동역학·링잉 미모델",
    ],
    test_paths=["tests/test_ex02.py"],
    extends=["FL02", "FL08"],
)
