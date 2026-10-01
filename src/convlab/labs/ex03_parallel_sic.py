"""EX03 - Parallel SiC, gate loop and DPT metrology (textbook E03; extends FL02 and FL04).

Three experiments, kept apart as the textbook asks (static sharing, common-source coupling,
timing skew and measurement deskew are different things):
  A  static four-branch sharing at a common terminal voltage (110.553/99.497/99.497/90.452 A),
     branch losses, and a positive temperature coefficient with a per-branch thermal model
     (partial self-balancing, not a cure);
  A  DPT metrology on the textbook's synthetic 40 ns linear overlap (E0 = 533.333 uJ; current
     channel -5 ns: +41.99 %, +5 ns: -33.01 %) with an explicitly stored window that includes the
     pre-transition interval, gain/offset/bandwidth sensitivity, a ranking decision under deskew
     uncertainty, and the energy reference planes of the FL02 synthetic cell (terminal vs die);
  D  dynamic sharing: four parallel branches of the FL02 commutation cell with real states -
     individual gate loops (R_g,k, L_g,k), a common gate resistor, per-branch common-source
     inductance, a shared source inductance, branch drain inductances with bounded mutual
     coupling, V_th and driver-timing spread - compared with the algebraic screens
     (L_s di/dt = 4 V, 10 ns x 2 kA/us = 20 A).  A bounded synthetic equivalent, not a vendor model.
Closed forms for comparison: reference/parallel.py (never imported here).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.optimize import root

from ..engine.pwl import PWL
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from .fl02_devices import Cell, CellSpec, DevArrays, DeviceParams, first_cross

TB_E03 = TextbookRef("expert-e03.-병렬-sicgate-loopdpt-계측-사용자-강점의-전문가-확장-ex03", "E03. 병렬 SiC·gate loop·DPT 계측 [EX03]")
TB_05 = TextbookRef("gate-drivedpt보호-강점을-면접-증거로-만들기-fl02", "05. Gate drive·DPT·보호 [FL02]")
TB_E12 = TextbookRef("expert-e12-설계-리뷰를-통과하는-답변-세-개의-통합-사례", "E12 Capstone C · 병렬 SiC 한 branch 과열")

TEXTBOOK_R = (3.6e-3, 4.0e-3, 4.0e-3, 4.4e-3)


def _is_textbook_R(R, I):
    return all(abs(a - b) < 1e-12 for a, b in zip(R, TEXTBOOK_R)) and abs(I - 400.0) < 1e-9


# ======================================================================================
# Experiment 1: static sharing and positive temperature coefficient
# ======================================================================================


def _sharing(R, I):
    G = 1.0 / np.asarray(R, dtype=float)
    return I * G / G.sum()


def _nodal(R, I):
    """Independent formulation: unknown common terminal voltage V from KCL, branch currents V/R_k."""
    R = np.asarray(R, dtype=float)
    # KCL at the top node: sum_k (V - 0)/R_k - I = 0  -> 1 x 1 linear system solved as such
    A = np.array([[np.sum(1.0 / R)]])
    V = float(np.linalg.solve(A, np.array([I]))[0])
    return V / R, V


def _electrothermal(R0, I, alpha, Rth, Tc, tol=1e-12, it_max=500):
    """Branch temperatures with R_k(T) = R_k0 [1 + alpha (T_k - T_c)] and T_k = T_c + R_th,k I_k^2 R_k(T_k).

    Returns (fixed-point iteration history, Newton solution) - two independent solution paths.
    """
    R0 = np.asarray(R0, dtype=float)
    Rth = np.asarray(Rth, dtype=float)
    T = np.full(R0.size, Tc)
    hist = []
    for _ in range(it_max):
        R = R0 * (1 + alpha * (T - Tc))
        Ik = _sharing(R, I)
        Tn = Tc + Rth * Ik**2 * R
        hist.append((Ik.copy(), Tn.copy()))
        if np.max(np.abs(Tn - T)) < tol:
            T = Tn
            break
        T = Tn
    conv = len(hist) < it_max

    def F(T):
        R = R0 * (1 + alpha * (T - Tc))
        Ik = _sharing(R, I)
        return T - Tc - Rth * Ik**2 * R

    sol = root(F, np.full(R0.size, Tc + 10.0), method="hybr", tol=1e-14)
    Tn = sol.x
    Rn = R0 * (1 + alpha * (Tn - Tc))
    ok = bool(np.max(np.abs(F(Tn))) < 1e-9)  # judged by the residual, not by the solver's progress flag
    return hist, conv, (_sharing(Rn, I), Tn, Rn, ok)


def run_static(v: dict) -> Result:
    res = Result("EX03", "static_sharing", "A (정적 저항 모델 + 전열)")
    R = [v["R1"], v["R2"], v["R3"], v["R4"]]
    I = v["I_total"]
    Ik = _sharing(R, I)
    tb = _is_textbook_R(R, I)
    refs = (110.553, 99.497, 99.497, 90.452)
    for k in range(4):
        res.add_metric(f"I{k + 1}", f"branch {k + 1} 전류 (R = {R[k] * 1e3:.4g} mΩ)", float(Ik[k]), "A", ref=refs[k] if tb else None, ref_label=f"교재 {refs[k]:g} A", tol=1e-5, basis="정적 저항 분담, 공통 단자전압")
    Pk = Ik**2 * np.asarray(R)
    res.add_metric("I_mean", "branch 평균 전류", I / 4, "A", basis="평균값으로 보면 모두 100 A")
    res.add_metric("spread", "(최대 − 최소) / 평균", float((Ik.max() - Ik.min()) / (I / 4)), "", basis="등온")
    res.add_metric("P_max", "최대 branch 손실 I²R", float(Pk.max()), "W", basis=f"branch {int(np.argmax(Pk)) + 1}; 평균 {Pk.mean():.4g} W")
    res.add_metric("P_tot", "4 branch 전도손실 합", float(Pk.sum()), "W")
    Iv, Vt = _nodal(R, I)
    res.add_check(Check("KCL 절점 풀이 vs 전류분배식", "PASS" if np.max(np.abs(Iv - Ik)) < 1e-9 * I else "FAIL", float(np.max(np.abs(Iv - Ik))), "A", 1e-9 * I, path="공통 단자전압 V를 KCL로 풀고 V/R_k (절점법) vs I·G_k/ΣG", independent=True, detail=f"단자전압 V = {Vt * 1e3:.5g} mV"))
    res.add_check(check_close("전류 합 = I_total (KCL)", float(Ik.sum()), I, 1e-12, "branch 전류 합", False, "A"))
    # electro-thermal (positive temperature coefficient)
    a = v["alpha_R"]
    Rth = [v["Rth1"], v["Rth2"], v["Rth3"], v["Rth4"]]
    hist, conv, (In, Tn, Rn, ok) = _electrothermal(R, I, a, Rth, v["Tc"])
    Ifp, Tfp = hist[-1]
    res.add_check(Check("전열 해: 고정점 반복 vs Newton (hybr)", "PASS" if conv and ok and np.max(np.abs(Ifp - In)) < 1e-6 else "FAIL", float(np.max(np.abs(Ifp - In))), "A", 1e-6, path="감쇠 없는 대입 반복 vs 다변수 Newton 계열 해법", independent=True, detail=f"반복 {len(hist)}회"))
    Pn = In**2 * Rn
    for k in range(4):
        res.add_metric(f"Iet{k + 1}", f"branch {k + 1} 전류 (전열 평형)", float(In[k]), "A", basis=f"T_j = {Tn[k]:.4g} °C")
    sp_et = float((In.max() - In.min()) / (I / 4))
    sp_iso = float((Ik.max() - Ik.min()) / (I / 4))
    res.add_metric("spread_et", "(최대 − 최소) / 평균 (전열 평형)", sp_et, "", basis=f"등온 대비 {(1 - sp_et / sp_iso) * 100 if sp_iso > 0 else 0:.3g} % 감소" if sp_iso > 0 else "")
    res.add_metric("T_hot", "가장 뜨거운 branch 온도", float(Tn.max()), "°C", basis=f"branch {int(np.argmax(Tn)) + 1}")
    res.add_metric("dT_branch", "branch 온도차 (최고 − 최저)", float(Tn.max() - Tn.min()), "K")
    x = [1, 2, 3, 4]
    res.add_series("I_iso", "등온 정적 분담", "A", x, Ik.tolist(), style="points")
    res.add_series("I_et", f"양의 온도계수 α = {a:g}/K 평형", "A", x, In.tolist(), style="points")
    res.add_series("I_avg", "평균 100 A (가정하면 안 되는 값)", "A", [1, 4], [I / 4] * 2, dash=True)
    res.add_plot("p_I", "branch별 전류: 평균이 가리는 weakest branch", ["I_iso", "I_et", "I_avg"], x_label="branch", x_unit="", y_label="전류", y_unit="A", kind="xy", level="A",
                 proved="R이 작은 branch가 더 많은 전류를 가져간다. 양의 온도계수는 차이를 일부 줄이지만 없애지 않는다.",
                 not_yet="정적 저항 모델이다. turn-on·off의 동적 분담(gate loop·common-source·skew)은 실험 3에서 푼다.")
    res.add_series("T_et", "branch 온도 (전열 평형)", "°C", x, Tn.tolist(), style="points")
    res.add_series("P_et", "branch 손실 (전열 평형)", "W", x, Pn.tolist(), style="points")
    res.add_plot("p_T", "branch 온도와 손실", ["T_et"], x_label="branch", x_unit="", y_label="T_j", y_unit="°C", kind="xy", level="A", hlines=[{"y": v["Tc"], "label": f"냉각 경계 {v['Tc']:g} °C"}],
                 proved="열 경로(R_th)가 다르면 전류가 같아도 온도가 다르다 — 뜨거운 branch 하나가 곧 불량 chip은 아니다.", not_yet="branch 사이 열 결합은 없다고 가정했다 (EX08).")
    conv_err = [float(np.max(np.abs(h[0] - In))) for h in hist]
    res.add_series("conv", "반복별 최대 전류 오차", "A", list(range(1, len(conv_err) + 1)), [max(e, 1e-16) for e in conv_err], style="points")
    res.add_plot("p_conv", "전열 반복의 수렴", ["conv"], x_label="반복", x_unit="", y_label="|ΔI|", y_unit="A", kind="xy", log_y=True, level="A",
                 proved="이 조건의 전열 되먹임은 약해서 반복이 빠르게 수렴한다.", not_yet="반복 해는 다변수 Newton(hybr) 해와 1e-6 A 이내로 비교했다. 선형 온도계수 α와 branch별 독립 R_th의 정적 모델이라 큰 α·약한 냉각에서의 수렴 실패나 branch 사이 열 결합(EX08)은 이 그림에 없다.")
    rows = [[k + 1, R[k] * 1e3, float(Ik[k]), float(Pk[k]), Rth[k], float(In[k]), float(Tn[k]), float(Pn[k])] for k in range(4)]
    res.tables.append(Table("t_branch", "branch별 결과 (branch-resolved)", ["branch", "R [mΩ]", "I 등온 [A]", "P 등온 [W]", "R_th [K/W]", "I 전열 [A]", "T_j [°C]", "P 전열 [W]"], rows,
                            note="모듈 평균(100 A·40 W)만 보면 가장 먼저 stress를 받는 branch를 놓친다."))
    res.circuit = {"diagram": _static_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    res.verdict("PASS_WITHIN_MODEL", "정적 분담이 교재 값·절점 풀이와 일치; 양의 온도계수는 분담을 일부만 개선")
    res.assumptions += ["공통 단자전압의 정적 저항 모델 (inductance·gate 동역학 없음)", f"R_k(T) = R_k[1 + α(T_j − T_c)], α = {a:g}/K (합성), branch별 독립 R_th (branch 간 열 결합 없음)"]
    res.not_valid_for += ["turn-on/off 동적 분담·peak 전류 (실험 3)", "실제 소자 분산·수율 추정", "모듈 내부 열 결합 (EX08)"]
    res.interpretation = (
        f"공통 단자전압에서 전류는 컨덕턴스에 비례해 나뉘므로 3.6 mΩ branch가 {Ik[0]:.4g} A, 4.4 mΩ branch가 {Ik[3]:.4g} A를 흘린다. 평균 100 A만 보면 이 차이가 사라진다. "
        f"양의 온도계수는 많이 흐르는 branch를 데워 R을 키우므로 차이를 {(1 - sp_et / sp_iso) * 100 if sp_iso > 0 else 0:.3g} % 줄이지만, 열 결합·package impedance·동적 gate 불일치까지 자동으로 해결하지 않는다."
    )
    return res


def _static_circuit(v: dict) -> Circuit:
    c = Circuit("parallel_static", 600, 260, title="정적 분담: 공통 단자전압의 병렬 저항")
    c.add("isource", "I", 70, 130, 270, "I_total", f"{v['I_total']:g} A", lpos=(50, 134, "end"))
    xs = [200, 300, 400, 500]
    for k, x in enumerate(xs):
        c.add("resistor", f"R{k + 1}", x, 130, 90, f"R{k + 1}", f"{v[f'R{k + 1}'] * 1e3:.3g} mΩ", lpos=(x + 14, 134, "start"))
        c.wire(f"w_t{k}", (x, 40), (x, 100))
        c.wire(f"w_b{k}", (x, 160), (x, 220))
        c.probe(f"p{k}", f"I_iso_{k}", x, 80, "down", f"I{k + 1}")
    c.wire("w_top", (70, 100), (70, 40), (500, 40))
    c.wire("w_bot", (70, 160), (70, 220), (500, 220))
    c.add("ground", "g", 300, 220)
    c.dot(*[(x, 40) for x in xs[:-1]], *[(x, 220) for x in xs[:-1]])
    c.text(290, 32, "공통 drain 단자 (V)", "")
    return c


# ======================================================================================
# Experiment 2: DPT metrology on the synthetic linear overlap + reference planes
# ======================================================================================


def _wave_knots(V, I, tr, t0, t1):
    """Continuous PWL knots of v(t) (V -> 0 over [0, t_r]) and i(t) (0 -> I) on [t0, t1]."""
    tv = [t0, 0.0, tr, t1]
    return np.array(tv), np.array([V, V, 0.0, 0.0]), np.array([0.0, 0.0, I, I])


def _pwl_on(tk, yk, a, b, extra=()):
    t = np.unique(np.concatenate([tk[(tk > a) & (tk < b)], [a, b], [x for x in extra if a < x < b]]))
    y = np.interp(t, tk, yk)
    return t, y


def overlap_energy_pwl(V, I, tr, skew, a, b, gain=0.0, off_i=0.0, off_v=0.0):
    """Exact integral of v_m(t) i_m(t) over [a, b] (both channels piecewise linear): PWL algebra."""
    tk, vk, ik = _wave_knots(V, I, tr, min(a, -abs(skew)) - 1e-9, max(b, tr + abs(skew)) + 1e-9)
    ti = tk + skew
    knots = np.concatenate([tk, ti])
    t = np.unique(np.concatenate([knots[(knots > a) & (knots < b)], [a, b]]))
    vv = np.interp(t, tk, vk) + off_v
    ii = (np.interp(t, ti, ik) + off_i) * (1.0 + gain)
    pv = PWL(t, vv[:-1], vv[1:])
    pi = PWL(t, ii[:-1], ii[1:])
    return pv.product_integral(pi)


def overlap_energy_closed(V, I, tr, tau):
    """Hand calculation (wide window): E(tau) for a current channel shifted by tau, |tau| < t_r."""
    if tau >= 0:
        return V * I * (tr - tau) ** 3 / (6 * tr * tr)
    s = -tau
    a = tr - s
    return V * I * s * s / tr + V * I / (tr * tr) * (tr * a * a / 2 + tr * s * a - a**3 / 3 - s * a * a / 2)


def lpf_pwl(tk, yk, tau, t):
    """Exact first-order low-pass response to a continuous PWL input starting flat (sum of ramp responses)."""
    if tau <= 0:
        return np.interp(t, tk, yk)
    slopes = np.diff(yk) / np.diff(tk)
    ds = np.diff(np.concatenate([[0.0], slopes]))
    y = np.full_like(t, yk[0], dtype=float)
    for k in range(ds.size):
        u = t - tk[k]
        m = u > 0
        y[m] += ds[k] * (u[m] - tau * (1.0 - np.exp(-u[m] / tau)))
    return y


def _window(kind, tr, skew, V, I, pre, post):
    if kind == "wide":
        return -pre, tr + post, f"넓은 고정창 [−{pre * 1e9:.3g} ns, t_r + {post * 1e9:.3g} ns] (전이 전 v = V 구간 포함)"
    if kind == "narrow":
        return 0.0, tr, "좁은 창 [0, t_r] (전압 전이 구간만)"
    a = skew + 0.1 * tr  # i_m reaches 10 % I
    b = 0.98 * tr  # v reaches 2 % V
    return a, b, "문턱창: i_m ≥ 10 % I → v ≤ 2 % V"


def _dpt_cell_reference_planes():
    """FL02 nominal synthetic cell (N = 1): terminal vs die energy for both events (reference planes)."""
    d = DeviceParams()
    V0 = d.Vgd0
    dev = replace(d, Cgd0=20e-9 / (2.0 * V0 * (math.sqrt(1.0 + 800.0 / V0) - 1.0)))
    spec = CellSpec(dut=[dev], hs=dev, Vbus=800.0, IL=100.0, La=8e-9, Lb=3e-9, Rp=3.0, LsH=2e-9, Ld=[[0.0]], Ls=[2e-9], Lg=[10e-9], Rg_on=[10.0], Rg_off=[10.0], topology="individual_kelvin")
    t_off, t_on = 20e-9, 1.02e-6
    cell = Cell(spec, [(t_off, "off"), (t_on, "on")], start="on")
    r = cell.simulate(t_on + 300e-9, n_samples=500, rtol=1e-7, t_fine=[(t_off - 10e-9, t_off + 280e-9, 6000), (t_on - 10e-9, t_on + 280e-9, 6000)])
    t = r.t
    vds, i = r.x[cell.i_vds][0], r.x[cell.i_i][0]
    q_ch = cell.qnames.index("E_ch0")
    out = {}
    for ev, tc in (("off", t_off), ("on", t_on)):
        a, b = tc - 5e-9, tc + 250e-9
        m = (t >= a) & (t <= b)
        e_term = float(np.trapezoid(vds[m] * i[m], t[m]))
        e_ps = float(np.trapezoid(r.out["vdsPS"][0][m] * i[m], t[m]))
        e_ch = float(np.interp(b, t, r.q[q_ch]) - np.interp(a, t, r.q[q_ch]))
        out[ev] = (e_term, e_ch, e_ps)
    D = DevArrays([dev])
    Eoss = float(D.E_ds(np.array([800.0]))[0] + D.E_gd(np.array([800.0]))[0])
    return out, Eoss, r.energy["normalised"]


def run_dpt_deskew(v: dict) -> Result:
    res = Result("EX03", "dpt_deskew", "A (합성 선형 overlap) + D (FL02 합성 셀의 참조면)")
    V, I, tr = v["V"], v["I"], v["tr"]
    pre, post = v["W_pre"], v["W_post"]
    tb = abs(V - 800) < 1e-9 and abs(I - 100) < 1e-9 and abs(tr - 40e-9) < 1e-18
    E0 = overlap_energy_pwl(V, I, tr, 0.0, -pre, tr + post)
    res.add_metric("E0", "E₀ (정렬, 넓은 고정창)", E0, "J", ref=533.333e-6 if tb else V * I * tr / 6, ref_label="교재 0.533333 mJ" if tb else "V·I·t_r/6", tol=1e-6, basis="∫v·i dt, PWL 정확 적분")
    Em5 = overlap_energy_pwl(V, I, tr, -5e-9, -pre, tr + post)
    Ep5 = overlap_energy_pwl(V, I, tr, +5e-9, -pre, tr + post)
    tbw = tb and pre >= 5e-9 and post >= 5e-9
    res.add_metric("dE_m5", "전류 채널 −5 ns (앞당김): E 변화", Em5 / E0 - 1, "", ref=0.4199 if tbw else None, ref_label="교재 +41.99 %", tol=2e-4, basis="같은 소자 파형, 측정 정렬만 변경")
    res.add_metric("dE_p5", "전류 채널 +5 ns (늦춤): E 변화", Ep5 / E0 - 1, "", ref=-0.3301 if tbw else None, ref_label="교재 −33.01 %", tol=4e-4)
    Em5n = overlap_energy_pwl(V, I, tr, -5e-9, 0.0, tr)
    res.add_metric("dE_m5_narrow", "−5 ns를 좁은 창 [0, t_r]로 적분하면", Em5n / E0 - 1, "", basis="전이 전 구간(v = V, i ≠ 0)을 빼먹어 과소평가", note="창 정의가 결과를 바꾼다 — 창을 결과와 함께 저장")
    # the user's stored settings
    sk = v["skew"]
    a, b, wtxt = _window(v["window"], tr, sk, V, I, pre, post)
    Eu = overlap_energy_pwl(V, I, tr, sk, a, b, gain=v["gain"], off_i=v["off_i"], off_v=v["off_v"]) if b > a else 0.0
    if b <= a:
        res.warnings.append("문턱창이 비어 있다 (전류 10 % 도달이 전압 2 % 도달보다 늦음) — 이 정의로는 E를 정할 수 없다.")
    res.add_metric("E_user", "저장된 설정의 E", Eu, "J", basis=f"skew {sk * 1e9:+.3g} ns, gain {v['gain'] * 100:+.3g} %, offset I {v['off_i']:+.3g} A / V {v['off_v']:+.3g} V, {wtxt}")
    res.add_metric("E_user_err", "저장된 설정 − 기준 E₀", Eu / E0 - 1, "")
    # independent paths: closed-form hand calculation and dense sampling
    for tau in (-5e-9, 0.0, 5e-9):
        e_pwl = overlap_energy_pwl(V, I, tr, tau, -max(pre, 6e-9), tr + max(post, 6e-9))
        e_cf = overlap_energy_closed(V, I, tr, tau)
        res.add_check(check_close(f"skew {tau * 1e9:+.0f} ns: PWL 정확 적분 vs 손계산 닫힌 식", e_pwl, e_cf, 1e-12, "구간별 선형 곱의 정확 적분 vs 손으로 유도한 다항식 (넓은 창)", True, "J"))
    td = np.linspace(-pre, tr + post, 20001)
    ti = np.interp(td - (-5e-9), [-1, 0, tr, 1], [0, 0, I, I])
    vd = np.interp(td, [-1, 0, tr, 1], [V, V, 0, 0])
    res.add_check(check_close("skew −5 ns: PWL 정확 적분 vs 촘촘한 표본 사다리꼴", float(np.trapezoid(vd * ti, td)), Em5, 1e-6, "20 001점 균일 표본의 사다리꼴 적분 vs PWL 정확 적분", True, "J"))
    # skew sweep
    sks = np.linspace(-10e-9, 10e-9, 81)
    res.add_series("sk_wide", "넓은 고정창", "", (sks * 1e9).tolist(), [overlap_energy_pwl(V, I, tr, s, -max(pre, 10e-9), tr + max(post, 10e-9)) / E0 for s in sks])
    res.add_series("sk_narrow", "좁은 창 [0, t_r]", "", (sks * 1e9).tolist(), [overlap_energy_pwl(V, I, tr, s, 0.0, tr) / E0 for s in sks], dash=True)
    res.add_plot("p_skew", "전류 채널 skew에 따른 E/E₀", ["sk_wide", "sk_narrow"], x_label="current skew", x_unit="ns", y_label="E/E₀", y_unit="", kind="xy", level="A",
                 markers=[{"x": -5.0, "y": Em5 / E0, "label": f"{(Em5 / E0 - 1) * 100:+.2f} %"}, {"x": 5.0, "y": Ep5 / E0, "label": f"{(Ep5 / E0 - 1) * 100:+.2f} %"}], vlines=[{"x": 0.0, "label": "정렬"}],
                 proved="소자는 그대로인데 전류 채널 ±5 ns 정렬 오차만으로 E가 −33 %~+42 % 바뀐다. 창이 전이 전 구간을 빼면 앞당긴 경우를 과소평가한다.",
                 not_yet="합성 선형 파형이다. 실제 SiC 파형은 전류 overshoot·링잉·C_oss 전류를 가진다(아래 참조면 표와 실험 3).")
    # waveforms for -5 / 0 / +5 ns
    tw = np.linspace(-pre - 5e-9, tr + post + 5e-9, 600)
    tk, vk, ik = _wave_knots(V, I, tr, tw[0] - 1e-9, tw[-1] + 1e-9)
    res.add_series("wv", "v(t)", "V", tw.tolist(), np.interp(tw, tk, vk).tolist())
    for s_, lab in ((-5e-9, "−5 ns"), (0.0, "정렬"), (5e-9, "+5 ns")):
        res.add_series(f"wi{int(s_ * 1e9)}", f"i_m(t) {lab}", "A", tw.tolist(), np.interp(tw, tk + s_, ik).tolist(), dash=s_ != 0)
        res.add_series(f"wp{int(s_ * 1e9)}", f"v·i_m {lab}", "W", tw.tolist(), (np.interp(tw, tk, vk) * np.interp(tw, tk + s_, ik)).tolist(), dash=s_ != 0)
    bands = [{"x0": float(tw[0]), "x1": 0.0, "mode": "pre", "label": "전이 전"}, {"x0": 0.0, "x1": tr, "mode": "trans", "label": "전이 t_r"}, {"x0": tr, "x1": float(tw[-1]), "mode": "post", "label": "전이 후"}]
    res.add_plot("p_w", "합성 turn-on: v 하강·i 상승 (전류 채널 이동)", ["wv"], y_label="v", y_unit="V", bands=bands, level="A", group="w", window=(-pre, tr + post),
                 proved="전압 채널은 그대로 두고 전류 채널만 옮긴다.", not_yet="전압 채널은 지연·이득 오차가 없는 기준으로 두었다(대역 제한의 영향은 probe 대역폭 그림에서 따로 본다). 선형 v 하강의 합성 파형이라 실제 turn-on의 전압 링잉·완만한 tail·C_oss 방전 구간은 없다.")
    res.add_plot("p_wi", "전류 채널 (−5 / 0 / +5 ns)", ["wi-5", "wi0", "wi5"], y_label="i", y_unit="A", bands=bands, level="A", group="w", window=(-pre, tr + post), proved="같은 전류 파형을 시간축으로만 −5/0/+5 ns 옮긴 세 채널이다. 앞당긴 채널은 v가 아직 V인 전이 전 구간에서 이미 흐르며, 세 경우의 E는 PWL 정확 적분과 손계산 닫힌 식이 1e-12로 일치한다.", not_yet="deskew 오차를 주파수와 무관한 시간 이동 하나로 단순화했다. 실제 probe의 위상 지연·gain·offset 오차는 대역폭 그림과 저장 설정(gain·offset)에서 따로 다루며, 교정 없이 실제 skew 값을 알 수는 없다.")
    res.add_plot("p_wp", "순간전력 v·i_m과 적분창", ["wp-5", "wp0", "wp5"], y_label="p", y_unit="W", bands=bands, level="A", group="w", window=(-pre, tr + post),
                 proved="−5 ns에서는 전이 전(v = V)에 이미 전류가 흘러 전력이 생긴다 — 창이 그 구간을 포함해야 +41.99 %가 된다.", not_yet="이 합성 파형은 전이 뒤 v = 0이라 창 끝(W_post)이 E에 영향을 주지 않는다. 실제 파형에서는 링잉 꼬리와 v_DS(on)·I 전도 성분 때문에 창 끝도 E를 바꾸므로 창 정의를 결과와 함께 저장해야 한다.")
    # probe bandwidth: exact first-order response to the PWL channels
    tf = np.linspace(-pre, tr + post, 6001)
    tkb, vkb, ikb = _wave_knots(V, I, tr, -pre - 1e-9, tr + post + 1e-9)
    bws = np.geomspace(10e6, 2e9, 25)
    e_bw_i, e_bw_both = [], []
    for f in bws:
        tau = 1.0 / (2 * math.pi * f)
        vf = lpf_pwl(tkb, vkb, tau, tf)
        i_f = lpf_pwl(tkb, ikb, tau, tf)
        e_bw_i.append(float(np.trapezoid(np.interp(tf, tkb, vkb) * i_f, tf)) / E0)
        e_bw_both.append(float(np.trapezoid(vf * i_f, tf)) / E0)
    res.add_series("bw_i", "전류 probe만 대역 제한", "", bws.tolist(), e_bw_i)
    res.add_series("bw_both", "v·i 두 probe 모두 같은 대역", "", bws.tolist(), e_bw_both, dash=True)
    res.add_plot("p_bw", "probe 대역폭에 따른 E/E₀ (1차 저역통과)", ["bw_i", "bw_both"], x_label="대역폭 f_3dB", x_unit="Hz", y_label="E/E₀", y_unit="", kind="xy", log_x=True, level="A", hlines=[{"y": 1.0, "label": "이상 측정"}],
                 proved="한 채널만 대역이 좁으면 그 채널이 늦어진 것(τ = 1/2πf)과 같아 E가 줄고, 두 채널이 같은 대역이면 지연은 상쇄되지만 파형이 뭉개져 E가 달라진다.",
                 not_yet="probe를 1차 저역통과로 단순화했다. 실제 probe의 지연·위상·CM 제거비는 교정 데이터가 필요하다.")
    # independent path for the bandwidth model: scipy lsim (first-order hold) vs the exact ramp superposition
    from scipy.signal import lsim

    tau20 = 1.0 / (2 * math.pi * 20e6)
    tg = np.linspace(-pre, tr + post, 3001)
    tg = np.unique(np.concatenate([tg, [0.0, tr]]))
    _, y_ls, _ = lsim(([1.0], [tau20, 1.0]), np.interp(tg, tkb, ikb), tg - tg[0], X0=[np.interp(tg[0], tkb, ikb) * tau20], interp=True)
    y_ex = lpf_pwl(tkb, ikb, tau20, tg)
    res.add_check(check_close("대역 제한 모델: 정확 ramp 중첩 vs scipy lsim (FOH)", float(y_ls[-1]), float(y_ex[-1]), 1e-6, "1차 저역통과의 해석 ramp 응답 합 vs 상태공간 이산화 시뮬레이션 (끝값)", True, "A", abs_scale=I,
                              detail=f"최대 파형 차 {float(np.max(np.abs(y_ls - y_ex))):.2e} A"))
    # uncertainty budget: one factor at a time
    def e_with(**kw):
        return overlap_energy_pwl(V, I, tr, kw.get("skew", 0.0), -max(pre, 10e-9), tr + max(post, 10e-9), gain=kw.get("gain", 0.0), off_i=kw.get("off_i", 0.0), off_v=kw.get("off_v", 0.0)) / E0 - 1

    def e_bw(f, both=False):
        tau = 1.0 / (2 * math.pi * f)
        i_f = lpf_pwl(tkb, ikb, tau, tf)
        vf = lpf_pwl(tkb, vkb, tau, tf) if both else np.interp(tf, tkb, vkb)
        return float(np.trapezoid(vf * i_f, tf)) / E0 - 1

    budget = [["current skew −1 ns / +1 ns", f"{e_with(skew=-1e-9) * 100:+.2f} / {e_with(skew=1e-9) * 100:+.2f}"],
              ["current skew −5 ns / +5 ns", f"{e_with(skew=-5e-9) * 100:+.2f} / {e_with(skew=5e-9) * 100:+.2f}"],
              ["current gain ±2 %", f"{e_with(gain=0.02) * 100:+.2f} / {e_with(gain=-0.02) * 100:+.2f}"],
              ["current offset ±1 A", f"{e_with(off_i=1.0) * 100:+.2f} / {e_with(off_i=-1.0) * 100:+.2f}"],
              ["voltage offset ±2 V", f"{e_with(off_v=2.0) * 100:+.2f} / {e_with(off_v=-2.0) * 100:+.2f}"],
              ["전류 probe 대역 100 MHz / 20 MHz", f"{e_bw(100e6) * 100:+.2f} / {e_bw(20e6) * 100:+.2f}"],
              ["두 probe 같은 대역 100 MHz / 20 MHz", f"{e_bw(100e6, True) * 100:+.2f} / {e_bw(20e6, True) * 100:+.2f}"]]
    res.tables.append(Table("t_budget", "측정 오차 요인별 E 변화 (한 번에 하나씩, 넓은 창)", ["요인", "E 변화 [%]"], budget,
                            note="offset은 창 길이에 비례해 누적된다. 대역 제한은 지연(τ = 1/2πf)과 파형 왜곡을 함께 만든다."))
    # ranking decision under deskew uncertainty
    su = v["skew_unc"]
    lo, hi = e_with(skew=su), e_with(skew=-su)
    dE = v["dE_branch"]
    res.add_metric("unc_band", f"deskew 불확도 ±{su * 1e9:.3g} ns의 E 구간", f"{lo * 100:+.3g} % ~ {hi * 100:+.3g} %", "", basis="넓은 고정창")
    unresolved = dE < (hi - lo)
    res.add_metric("dE_branch", "비교하려는 두 branch의 E 차이", dE, "", basis="합성 가정")
    # reference planes with the FL02 synthetic cell
    planes, Eoss, led = _dpt_cell_reference_planes()
    rows = []
    for ev, name in (("off", "turn-off"), ("on", "turn-on")):
        e_term, e_ch, e_ps = planes[ev]
        rows.append([name, e_term * 1e6, e_ps * 1e6, e_ch * 1e6, (e_term - e_ch) * 1e6])
    res.tables.append(Table("t_planes", "에너지 참조면 (FL02 합성 셀, 800 V·100 A, 고정창 −5/+250 ns)", ["사건", "단자 Kelvin v·i [µJ]", "power-source v·i [µJ]", "die 채널 소산 [µJ]", "단자 − die [µJ]"], rows,
                            note=f"단자 적분에는 C_oss 저장·방출이 들어간다 (DUT E_oss(800 V) = {Eoss * 1e6:.4g} µJ): turn-off 단자값은 die보다 크고 turn-on 단자값은 작다. die 값은 모델에서만 얻는다."))
    res.add_check(Check("참조면 표의 셀 에너지 원장", "PASS" if abs(led) < 1e-5 else "FAIL", led, "rel", 1e-5, path="FL02 합성 셀의 포트·소산·저장에너지 원장", independent=True))
    res.add_metric("Eoss_cell", "FL02 셀 DUT의 E_oss(800 V)", Eoss, "J", basis="turn-off 단자 적분에 들어가는 저장에너지의 크기")
    res.tables.append(Table("t_settings", "적분 설정 (결과와 함께 저장)", ["항목", "값"], [
        ["v 참조", "소자 단자 v_DS (합성 파형)"], ["i 참조", "소자 단자 전류 (합성 파형)"], ["deskew", f"{sk * 1e9:+.3g} ns (전류 채널)"], ["창", wtxt],
        ["gain / offset", f"{v['gain'] * 100:+.3g} % / I {v['off_i']:+.3g} A, V {v['off_v']:+.3g} V"], ["대역폭", "이상 (표의 대역 제한은 별도 계산)"],
        ["Coss 경계", "단자 적분 (die 채널 소산과 다름 — 참조면 표)"]]))
    res.circuit = {"diagram": _meas_circuit(v).to_json(), "intervals": bands, "plot_group": "w"}
    res.verdict("PASS_WITHIN_MODEL", "E₀·±5 ns 변화가 PWL 정확 적분·손계산·표본 적분과 일치 (교재 수치 재현)")
    if unresolved:
        res.verdict("UNRESOLVED_RANKING", f"branch 간 E 차이 {dE * 100:.3g} %가 deskew 불확도 ±{su * 1e9:.3g} ns의 E 구간 폭 {(hi - lo) * 100:.3g} %보다 작다 — die 선별·R_g 변경 결론을 서두르지 않는다")
    res.assumptions += ["교재 합성 turn-on: 40 ns 동안 v 선형 800→0 V, i 선형 0→100 A (실제 SiC 파형 아님)", f"넓은 고정창 [−{pre * 1e9:.3g} ns, t_r + {post * 1e9:.3g} ns]: 앞당긴 전류의 전이 전 구간 포함", "probe = 1차 저역통과 (지연 τ = 1/2πf)", "참조면 표는 FL02 합성 셀(D) 결과"]
    res.not_valid_for += ["특정 probe·scope 조합의 실제 오차 (교정 필요)", "실제 소자 E 순위", "링잉이 큰 파형의 창 정의 (창 길이를 따로 검토)"]
    res.interpretation = (
        f"전이 중 v·i 곱의 적분은 두 파형의 겹침에 민감하다. 전류 채널이 5 ns 앞당겨지면 전압이 아직 {V:g} V인 동안 전류가 흘러 E가 {(Em5 / E0 - 1) * 100:+.2f} %, 5 ns 늦어지면 {(Ep5 / E0 - 1) * 100:+.2f} %가 된다. "
        "이 +42 %는 적분창이 전이 전 구간(v = V, 전류 ≠ 0)을 포함해야 나오는 값이라 창 정의를 결과와 함께 저장한다. branch 사이 E 차이가 이 측정 불확도보다 작으면 순위를 확정하지 않는다."
    )
    return res


def _meas_circuit(v: dict) -> Circuit:
    c = Circuit("dpt_measure", 660, 260, title="DPT 측정 경로: v probe·i probe·scope의 지연과 대역")
    c.add("nmos", "DUT", 150, 130, 90, "DUT", lpos=(180, 126, "start"))
    c.add("vsource", "Vb", 60, 130, 90, "V_bus", "", lpos=(40, 134, "end"))
    c.wire("w_t", (60, 100), (60, 40), (150, 40), (150, 100))
    c.wire("w_b", (60, 160), (60, 220), (150, 220), (150, 160))
    c.add("block", "VP", 340, 70, 0, "v probe (대역·지연)", w=170, h=34)
    c.add("block", "IP", 340, 180, 0, "i probe (skew·대역·gain·offset)", w=230, h=34)
    c.add("block", "SC", 560, 125, 0, "scope ∫v·i dt (창)", w=150, h=40)
    c.wire("w_vp", (150, 70), (255, 70))
    c.wire("w_ip", (150, 180), (225, 180))
    c.wire("w_vs", (425, 70), (560, 70), (560, 105))
    c.wire("w_is", (455, 180), (560, 180), (560, 145))
    c.mode("pre", "전이 전", ["VP", "w_vp", "SC", "w_vs"], "v = V_bus. 전류 채널이 앞당겨지면 이 구간에도 v·i가 생긴다", dim=[])
    c.mode("trans", "전이", ["VP", "IP", "SC", "w_vp", "w_ip", "w_vs", "w_is", "DUT"], "v 하강·i 상승이 겹치는 구간", dim=[])
    c.mode("post", "전이 후", ["IP", "w_ip", "SC", "w_is"], "v ≈ 0: offset이 있으면 창 끝까지 오차가 쌓인다", dim=[])
    return c


# ======================================================================================
# Experiment 3: dynamic sharing with real states (four branches of the FL02 cell)
# ======================================================================================

T_OFF3 = 20e-9


def _dyn_spec(v: dict, over: dict | None = None, identical: bool = False) -> CellSpec:
    vv = dict(v)
    vv.update(over or {})
    base = DeviceParams().scaled(vv["scale"])
    R = [vv["R1"], vv["R2"], vv["R3"], vv["R4"]]
    dV = [vv["dVth1"], vv["dVth2"], vv["dVth3"], vv["dVth4"]]
    dl = [vv["d1"], vv["d2"], vv["d3"], vv["d4"]]
    kM, dLd4 = vv["kM"], vv["dLd4"]
    if identical:
        R = [float(np.mean(R))] * 4
        dV, dl, kM, dLd4 = [0.0] * 4, [0.0] * 4, 0.0, 0.0
    devs = [replace(base, Ron=R[k], Vth=vv["Vth"] + dV[k]) for k in range(4)]
    K = np.array([[kM ** abs(a - b) for b in range(4)] for a in range(4)])
    Ld = vv["Ld"] * K
    Ld[3, 3] += dLd4
    return CellSpec(dut=devs, hs=DeviceParams().scaled(4 * vv["scale"]), Vbus=vv["Vbus"], IL=vv["I_total"], La=vv["La"], Lb=vv["Lb"], Rp=vv["Rp"], Rloop=1e-3, LsH=0.5e-9,
                    Ld=Ld, Ls=[vv["Ls"]] * 4, Lcs=vv["Lcs"], Lg=[vv["Lg"]] * 4, Rg_on=[vv["Rg"]] * 4, Rg_off=[vv["Rg"]] * 4, Rcom=vv["Rcom"], Re=[vv["Re"]] * 4,
                    Von=vv["Von"], Voff=vv["Voff"], delay=dl, topology=vv["topology"], LgH=5e-9, RgH=1.0)


def _equiv_single_spec(v: dict) -> CellSpec:
    """N = 1 equivalent of four identical, uncoupled branches: 4x die, branch L and R divided by 4."""
    vv = dict(v)
    Rm = float(np.mean([vv["R1"], vv["R2"], vv["R3"], vv["R4"]]))
    dev = replace(DeviceParams().scaled(4 * vv["scale"]), Ron=Rm / 4, Vth=vv["Vth"])
    return CellSpec(dut=[dev], hs=DeviceParams().scaled(4 * vv["scale"]), Vbus=vv["Vbus"], IL=vv["I_total"], La=vv["La"], Lb=vv["Lb"], Rp=vv["Rp"], Rloop=1e-3, LsH=0.5e-9,
                    Ld=[[vv["Ld"] / 4]], Ls=[vv["Ls"] / 4], Lcs=vv["Lcs"], Lg=[vv["Lg"] / 4], Rg_on=[vv["Rg"] / 4], Rg_off=[vv["Rg"] / 4], Rcom=vv["Rcom"], Re=[vv["Re"] / 4],
                    Von=vv["Von"], Voff=vv["Voff"], delay=[0.0], topology=vv["topology"], LgH=5e-9, RgH=1.0)


def _turn_on_run(spec: CellSpec, win: float = 350e-9, rtol: float = 1e-6):
    t_on = 20e-9
    cell = Cell(spec, [(t_on, "on")], start="off")
    r = cell.simulate(t_on + win, n_samples=300, rtol=rtol, t_fine=[(t_on - 5e-9, t_on + win, 4000)])
    return cell, r, t_on


def _peaks(cell, r, a, b):
    m = (r.t >= a) & (r.t <= b)
    i = r.x[cell.i_i][:, m]
    return i.max(axis=1), i.min(axis=1)


def run_dynamic(v: dict) -> Result:
    res = Result("EX03", "dynamic_sharing", "D (합성 4-branch commutation cell)")
    spec = _dyn_spec(v)
    t_on = T_OFF3 + v["t_gap"]
    t_end = t_on + v["t_hold"]
    cell = Cell(spec, [(T_OFF3, "off"), (t_on, "on")], start="on")
    W = 380e-9
    r = cell.simulate(t_end, n_samples=1500, rtol=1e-6, t_fine=[(T_OFF3 - 10e-9, T_OFF3 + W, 5000), (t_on - 10e-9, t_on + W, 5000)])
    t = r.t
    i = r.x[cell.i_i]
    vds = r.x[cell.i_vds]
    o = r.out
    I = v["I_total"]
    m_on = (t >= t_on) & (t <= t_on + W)
    m_off = (t >= T_OFF3) & (t <= T_OFF3 + W)
    didt = np.gradient(i, t, axis=1)
    pk_on = i[:, m_on].max(axis=1)
    didt_on = didt[:, m_on].max(axis=1)
    i0 = r.x[cell.i_i][:, 0]
    # branch channel energies inside the event windows (solver quadrature)
    def q_win(name, a, b):
        k = cell.qnames.index(name)
        return float(np.interp(b, t, r.q[k]) - np.interp(a, t, r.q[k]))

    Eon = [q_win(f"E_ch{k}", t_on - 5e-9, t_on + W) for k in range(4)]
    Eoff = [q_win(f"E_ch{k}", T_OFF3 - 5e-9, T_OFF3 + W) for k in range(4)]
    t50 = [first_cross(t, i[k], 0.5 * I / 4, t_on, t_on + W, True) for k in range(4)]
    t50v = [x for x in t50 if x is not None]
    skew_sim = (max(t50v) - min(t50v)) if len(t50v) == 4 else float("nan")
    # measured-reference quantities
    csi = np.abs(o["vgsPS"] - o["vgsK"])[:, m_on].max(axis=1)
    dm_on = np.max(np.abs(i[:, m_on] - o["idc"][m_on] / 4.0))
    after_v = first_cross(t, vds.mean(axis=0), 0.9 * v["Vbus"], T_OFF3, T_OFF3 + W, True) or T_OFF3
    m_ring = (t >= after_v) & (t <= T_OFF3 + W)
    dm_off = float(np.max(np.abs(i[:, m_ring] - o["idc"][m_ring] / 4.0))) if np.any(m_ring) else 0.0
    for k in range(4):
        res.add_metric(f"pk{k + 1}", f"branch {k + 1} turn-on peak 전류", float(pk_on[k]), "A", basis=f"di/dt max {didt_on[k] / 1e6:.3g} A/µs, E_on(die) {Eon[k] * 1e6:.4g} µJ")
    res.add_metric("pk_spread", "branch turn-on peak 차 (최대 − 최소)", float(pk_on.max() - pk_on.min()), "A", basis="branch-resolved (모듈 전류로는 안 보임)")
    res.add_metric("didt_on", "branch 최대 di/dt (turn-on)", float(didt_on.max()) / 1e6, "A/us")
    res.add_metric("skew_sim", "branch 50 % 전류 도달 시각의 퍼짐 (시뮬레이션)", skew_sim, "s", basis="gate 타이밍·V_th·layout이 만든 실제 전류 skew")
    res.add_metric("Eon_spread", "branch E_on(die) 최대/최소", max(Eon) / min(Eon) if min(Eon) > 0 else float("nan"), "", basis="turn-on 고정창의 채널 소산")
    res.add_metric("Eoff_spread", "branch E_off(die) 최대/최소", max(Eoff) / min(Eoff) if min(Eoff) > 0 else float("nan"), "")
    # screens: textbook first-order sizes and their counterparts in the simulation
    Ls = v["Ls"]
    res.add_metric("csi_screen", "screen: L_s·di/dt (교재 2 nH × 2 kA/µs)", 2e-9 * 2e9, "V", ref=4.0, ref_label="교재 4 V", tol=1e-9, basis="branch common-source inductance의 1차 크기")
    res.add_metric("csi_sim", "시뮬레이션: 게이트 루프의 L_s,k·di_s,k/dt 최대", float(csi.max()), "V", ref=Ls * float(didt_on.max()), ref_label="L_s × (시뮬레이션 최대 di/dt)", tol=None,
                   basis=f"= power-source 기준 − Kelvin 기준 v_GS; 토폴로지 {v['topology']}", note="공통 source 노드 반환이면 이 전압이 gate 구동을 깎는다; Kelvin 반환이면 gate 루프 밖")
    dmax = max([v["d1"], v["d2"], v["d3"], v["d4"]]) - min([v["d1"], v["d2"], v["d3"], v["d4"]])
    res.add_metric("skew_screen", "screen: Δt·di/dt (교재 10 ns × 2 kA/µs)", 10e-9 * 2e9, "A", ref=20.0, ref_label="교재 20 A", tol=1e-9, basis="1차 규모 추정 — 실제 branch 전류 예측 아님")
    if dmax > 0:
        res.add_metric("skew_sim_dI", f"시뮬레이션: 드라이버 skew {dmax * 1e9:.3g} ns의 branch peak 차", float(pk_on.max() - pk_on.min()), "A", ref=dmax * float(didt_on.max()), ref_label="Δt × (시뮬레이션 di/dt)", tol=None,
                       basis="결합·commutation·gate impedance를 함께 푼 값")
    res.add_metric("dm_on", "turn-on 중 branch 차동 전류 max|i_k − Σi/4|", float(dm_on), "A")
    res.add_metric("dm_off", "turn-off 링잉 중 branch 차동 전류 max|i_k − Σi/4|", dm_off, "A", note="branch끼리 서로 반대로 흔들리는 성분 — 모듈 전류(합)에는 보이지 않는다")
    iend = i[:, -1]
    res.add_metric("i_static0", "turn-off 직전 branch 전류 (정적, 채널 I-V)", ", ".join(f"{x:.4g}" for x in i0), "A", basis="첫 펄스 끝의 DC 상태")
    res.add_metric("i_end", f"turn-on 후 {v['t_hold'] * 1e6:.3g} µs의 branch 전류", ", ".join(f"{x:.4g}" for x in iend), "A", basis="인덕턴스 분담 → 저항 분담으로 L/R 이완 중")
    # --- checks ---------------------------------------------------------------------
    led = r.energy
    res.add_check(Check("에너지 잔차 (4 branch 셀 전체)", "PASS" if abs(led["normalised"]) < 1e-5 else "FAIL", led["normalised"], "rel", 1e-5, path="V_bus·∫i_dc + 드라이버 − 부하 − (채널·diode·저항 소산) − ΔW; 상태식과 독립 정의", independent=True,
                        detail=f"E_src {led['E_src'] * 1e3:.5g} mJ, 소산 {led['E_diss'] * 1e3:.5g} mJ, 잔차 {led['residual']:.3g} J"))
    r2 = cell.simulate(t_end, n_samples=400, rtol=1e-8, t_fine=[(t_on - 10e-9, t_on + W, 3000)])
    m2 = (r2.t >= t_on) & (r2.t <= t_on + W)
    pk2 = r2.x[cell.i_i][:, m2].max(axis=1)
    e2 = [float(np.interp(t_on + W, r2.t, r2.q[cell.qnames.index(f"E_ch{k}")]) - np.interp(t_on - 5e-9, r2.t, r2.q[cell.qnames.index(f"E_ch{k}")])) for k in range(4)]
    dtol = max(float(np.max(np.abs(pk2 - pk_on) / pk_on)), max(abs(a_ - b_) / b_ for a_, b_ in zip(e2, Eon)))
    res.add_check(Check("허용오차 강화 수렴 (rtol 1e-6 → 1e-8)", "PASS" if dtol < 1e-3 else "FAIL", dtol, "rel", 1e-3, path="같은 셀 재적분: branch peak·branch E_on 변화", independent=True))
    # independent formulation: four identical uncoupled branches vs a single 4x-die equivalent
    c4, r4, ton4 = _turn_on_run(_dyn_spec(v, identical=True))
    c1, r1, ton1 = _turn_on_run(_equiv_single_spec(v))
    tt = np.linspace(ton4, ton4 + 300e-9, 400)
    i4 = np.array([np.interp(tt, r4.t, r4.x[c4.i_i][k]) for k in range(4)])
    i1 = np.interp(tt, r1.t, r1.x[c1.i_i][0])
    eq_err = float(np.max(np.abs(i4 - i1[None, :] / 4.0)) / (I / 4))
    res.add_check(Check("한계 경우: 동일·비결합 4 branch = 4배 die 단일 소자의 1/4", "PASS" if eq_err < 1e-4 else "FAIL", eq_err, "rel", 1e-4, path="4-branch 상태식(20+ 상태) vs L·R을 1/4로 환산한 단일 소자 셀(다른 행렬·차원)", independent=True,
                        detail="대칭이면 branch 전류는 모두 같고 총전류의 1/4이어야 한다"))
    # sensitivity: unidentifiable mutual coupling as a bounded range; driver skew sweep
    rowsM = []
    kms = [v["kM_min"], v["kM"], v["kM_max"]]
    spreadM = []
    for km in kms:
        cc, rr, tn = _turn_on_run(_dyn_spec(v, {"kM": km}))
        pk, _ = _peaks(cc, rr, tn, tn + 350e-9)
        csi_k = float(np.abs(rr.out["vgsPS"] - rr.out["vgsK"])[:, rr.t >= tn].max())
        spreadM.append(float(pk.max() - pk.min()))
        rowsM.append([km, *[float(x) for x in pk], float(pk.max() - pk.min()), csi_k])
    res.tables.append(Table("t_M", "식별할 수 없는 상호결합 k_M: 범위로 보고", ["k_M", "peak i1 [A]", "peak i2 [A]", "peak i3 [A]", "peak i4 [A]", "peak 차 [A]", "최대 L_s·di/dt [V]"], rowsM,
                            note="k_M은 인접 branch drain inductance 결합계수(다음 인접은 k_M², …; 양의 정부호 유지). 측정으로 식별하지 못하면 결론을 이 범위로 말한다."))
    fmtA = lambda x: "≈ 0" if x < 1e-3 else f"{x:.3g}"  # noqa: E731
    res.add_metric("M_range", "k_M 범위에서 peak 차의 범위", f"{fmtA(min(spreadM))} – {fmtA(max(spreadM))} A", "", basis=f"k_M = {kms[0]:g} … {kms[2]:g}")
    sks = [0.0, 2.5e-9, 5e-9, 10e-9]
    dI = []
    for sk in sks:
        cc, rr, tn = _turn_on_run(_dyn_spec(v, {"d1": 0.0, "d2": 0.0, "d3": 0.0, "d4": sk}))
        pk, _ = _peaks(cc, rr, tn, tn + 350e-9)
        dI.append(float(max(pk[:3]) - pk[3]))
    res.add_series("sk_sim", "시뮬레이션: branch 4 지연에 따른 peak 차", "A", [x * 1e9 for x in sks], dI, style="points")
    res.add_series("sk_scr", f"screen Δt × {didt_on.max() / 1e9:.3g} kA/µs", "A", [0.0, 10.0], [0.0, 10e-9 * float(didt_on.max())], dash=True)
    res.add_series("sk_tb", "교재 screen Δt × 2 kA/µs", "A", [0.0, 10.0], [0.0, 20.0], dash=True)
    res.add_plot("p_skew", "드라이버 timing skew → branch peak 차 (screen vs 결합 모델)", ["sk_sim", "sk_scr", "sk_tb"], x_label="branch 4 드라이버 지연", x_unit="ns", y_label="peak 차", y_unit="A", kind="xy", level="D",
                 proved="10 ns skew가 만드는 branch 전류 차는 Δt·di/dt와 같은 규모이지만, 실제 값은 결합·commutation·gate impedance가 정한다.",
                 not_yet="합성 셀의 결과다. 실제 branch 전류 예측이 아니며 layout·소자 분산 데이터가 필요하다.")
    # --- plots -------------------------------------------------------------------------
    def ser(key, lab, unit, y, a, b, dash=False):
        m = (t >= a) & (t <= b)
        res.add_series(key, lab, unit, t[m].tolist(), np.asarray(y)[m].tolist(), dash=dash)

    a_on, b_on = t_on - 10e-9, t_on + W
    a_of, b_of = T_OFF3 - 10e-9, T_OFF3 + W
    for k in range(4):
        ser(f"ion{k}", f"i{k + 1}", "A", i[k], a_on, b_on)
        ser(f"iof{k}", f"i{k + 1}", "A", i[k], a_of, b_of)
        ser(f"vgs{k}", f"v_GS{k + 1} die", "V", r.x[cell.i_vgs][k], a_on, b_on)
        ser(f"csi{k}", f"L_s·di/dt branch {k + 1}", "V", o["vgsPS"][k] - o["vgsK"][k], a_on, b_on)
        ser(f"vdo{k}", f"v_DS{k + 1}", "V", vds[k], a_of, b_of)
        ser(f"ihold{k}", f"i{k + 1}", "A", i[k], t_on, t_end)
    ser("iavg_on", "Σi/4 (모듈 평균)", "A", o["idc"] / 4, a_on, b_on, dash=True)
    ser("iavg_of", "Σi/4 (모듈 평균)", "A", o["idc"] / 4, a_of, b_of, dash=True)
    ser("gK1", "branch 1 v_GS Kelvin 핀", "V", o["vgsK"][0], a_on, b_on)
    ser("gP1", "branch 1 v_GS power-source 기준", "V", o["vgsPS"][0], a_on, b_on)
    ser("gK4", "branch 4 v_GS Kelvin 핀", "V", o["vgsK"][3], a_on, b_on, dash=True)
    ser("gP4", "branch 4 v_GS power-source 기준", "V", o["vgsPS"][3], a_on, b_on, dash=True)
    stat = _static_nonlinear(cell)
    for k in range(4):
        res.add_series(f"stat{k}", f"정적 분담 i{k + 1}", "A", [t_on, t_end], [float(stat[k])] * 2, dash=True)
    bands = _dyn_bands(t, o["idc"], vds.mean(axis=0), T_OFF3, t_on, v["Vbus"], I, t_end)
    b_on = [x for x in bands if x["x0"] >= t_on - 1e-12]
    b_off = [x for x in bands if x["x1"] <= t_on + 1e-12]
    res.add_plot("p_on_i", "turn-on: branch별 전류 (branch-resolved)", [f"ion{k}" for k in range(4)] + ["iavg_on"], y_label="i", y_unit="A", bands=b_on, group="on", level="D",
                 proved="같은 명령에도 branch 전류는 gate loop·V_th·layout·결합에 따라 다르게 오른다. 모듈 평균(점선)은 가장 먼저 stress를 받는 branch를 가린다.",
                 not_yet="합성 셀(동일 die 모델 + 지정한 분산)이다. 실제 소자 분산 분포·온도는 없다.")
    res.add_plot("p_on_g", "turn-on: branch별 local v_GS (die)", [f"vgs{k}" for k in range(4)], y_label="v_GS", y_unit="V", bands=b_on, group="on", level="D",
                 proved="각 die의 v_GS는 자기 gate loop와 common-source 전압에 따라 다르게 오른다.", not_yet="die 전압은 측정할 수 없다 — 측정은 Kelvin·power-source 기준으로 한다.")
    res.add_plot("p_on_ref", "local gate 기준: Kelvin 핀 vs power-source 기준 (branch 1·4)", ["gK1", "gP1", "gK4", "gP4"], y_label="v_GS", y_unit="V", bands=b_on, group="on", level="D",
                 proved="같은 gate라도 측정 기준점에 따라 L_s·di/dt만큼 다르게 보인다 — 측정 기준을 적지 않은 v_GS 비교는 의미가 없다.", not_yet="두 기준의 차이는 branch L_s·di/dt이며 시뮬레이션 최대값을 L_s × 최대 di/dt와 나란히 표시했고, 셀 전체 에너지 원장(1e-5)으로 모델 일관성을 확인했다. L_s를 집중 인덕턴스 하나로 두었으므로 probe의 대역·CM 제거비·ground lead와 패키지 안에서 Kelvin 핀이 die source와 분리되는 정도는 모델에 없다.")
    res.add_plot("p_on_csi", "gate 루프에 걸리는 L_s,k·di_s,k/dt", [f"csi{k}" for k in range(4)], y_label="Δv", y_unit="V", bands=b_on, group="on", level="D", hlines=[{"y": 4.0, "label": "교재 screen 4 V"}],
                 proved=f"branch common-source inductance가 turn-on 동안 최대 {csi.max():.3g} V를 만든다 (screen 4 V와 같은 규모, 실제 di/dt에 비례).", not_yet="극성·loop 위치에 따라 되먹임 부호가 달라 단순 가산하지 않는다.")
    res.add_plot("p_off_i", "turn-off: branch별 전류와 차동 링잉", [f"iof{k}" for k in range(4)] + ["iavg_of"], y_label="i", y_unit="A", bands=b_off, group="off", level="D",
                 proved="turn-off 뒤 branch 전류가 서로 반대로 흔들리는 차동 성분은 모듈 전류 합에는 거의 보이지 않는다.", not_yet="감쇠는 합성 등가가 정한다.")
    res.add_plot("p_off_v", "turn-off: branch별 v_DS", [f"vdo{k}" for k in range(4)], y_label="v_DS", y_unit="V", bands=b_off, group="off", level="D", hlines=[{"y": v["Vbus"], "label": "V_bus"}],
                 proved="branch drain inductance가 다르면 die별 overshoot도 다르다.", not_yet="branch별 v_DS는 같은 셀의 에너지 원장(1e-5)으로 확인했다. overshoot 크기는 합성 L_d·k_M과 R_p‖L_b 감쇠 등가가 정하며, 실제 die 커패시턴스 곡선·온도·역회복과 측정점(단자 vs die)에 따른 차이는 포함하지 않는다.")
    res.add_plot("p_hold", "on 상태: 인덕턴스 분담에서 저항 분담으로 (L/R 이완)", [f"ihold{k}" for k in range(4)] + [f"stat{k}" for k in range(4)], y_label="i", y_unit="A", group="hold", level="D",
                 proved="switching 직후의 분담은 인덕턴스·타이밍이 정하고, 수 µs에 걸쳐 저항(정적) 분담으로 이완한다.", not_yet="온도 상승에 따른 R 변화는 넣지 않았다 (실험 1).")
    res.add_series("M_sp", "k_M별 branch peak 차", "A", kms, spreadM, style="points")
    res.add_plot("p_M", "상호결합 k_M 범위에 대한 민감도", ["M_sp"], x_label="k_M", x_unit="", y_label="peak 차", y_unit="A", kind="xy", level="D",
                 proved="식별할 수 없는 상호결합은 한 값이 아니라 범위로 결과를 보고한다.", not_yet="결합 부호·return path가 다른 busbar는 이 구조(양의 인접 결합)와 다르다.")
    rowsB = []
    for k in range(4):
        rowsB.append([k + 1, v[f"R{k + 1}"] * 1e3, v["Vth"] + v[f"dVth{k + 1}"], v[f"d{k + 1}"] * 1e9, float(i0[k]), float(pk_on[k]), float(didt_on[k]) / 1e6, Eon[k] * 1e6, Eoff[k] * 1e6, float(iend[k]), float(stat[k])])
    res.tables.append(Table("t_branch", "branch별 결과 (branch-resolved measurement)", ["branch", "R_on [mΩ]", "V_th [V]", "드라이버 지연 [ns]", "turn-off 직전 [A]", "turn-on peak [A]", "max di/dt [A/µs]", "E_on die [µJ]", "E_off die [µJ]", f"{v['t_hold'] * 1e6:.3g} µs 후 [A]", "정적 분담 [A]"], rowsB,
                            note="branch 전류·local v_GS를 같은 timebase로 보는 것이 모듈 평균보다 먼저다. die 에너지는 모델 전용 양이다."))
    res.tables.append(Table("t_model", "동적 분담 모델의 구성 (bounded synthetic equivalent)", ["요소", "값·가정"], [
        ["branch 소자", f"FL02 합성 die ×{v['scale']:g} (C·g_m 비례), R_on은 정적 R 값, V_th 분산 지정"],
        ["gate 구동", f"{v['topology']}: R_com {v['Rcom']:g} Ω + branch R_g {v['Rg']:g} Ω + L_g {v['Lg'] * 1e9:g} nH, branch별 드라이버 지연"],
        ["power loop", f"L_a {v['La'] * 1e9:g} nH + L_b {v['Lb'] * 1e9:g} nH ‖ R_p {v['Rp']:g} Ω (감쇠 등가) + L_sH 0.5 nH"],
        ["branch inductance", f"L_d {v['Ld'] * 1e9:g} nH, 상호결합 k_M {v['kM']:g} (인접), branch 4 추가 {v['dLd4'] * 1e9:g} nH"],
        ["source inductance", f"branch L_s {v['Ls'] * 1e9:g} nH (common-source), 공유 L_cs {v['Lcs'] * 1e9:g} nH"],
        ["상측", f"4 die 등가 (×{4 * v['scale']:g}), gate −4 V 유지"],
        ["상태 수", f"{cell.n_core} (+ 에너지 적분 {cell.n_q})"],
        ["한계", "vendor 소자 모델 아님, 역회복·온도·branch 간 열 결합 없음"]]))
    order = ["p_on_i", "p_on_g", "p_on_ref", "p_on_csi", "p_off_i", "p_off_v", "p_hold", "p_skew", "p_M"]
    res.plots.sort(key=lambda pl: order.index(pl.key))
    res.circuit = {"diagram": _dyn_circuit(v).to_json(), "intervals": bands, "plot_group": ""}
    res.verdict("PASS_WITHIN_MODEL", "동적 분담을 실제 상태(branch 전류·gate loop·결합 inductance)로 풀었다: 에너지 원장·허용오차·단일 소자 등가 한계 경우 통과 (합성 D 수준, vendor fidelity 아님)")
    dm_limit = v["dm_lim"] * I / 4.0
    if dm_off > dm_limit:
        res.verdict("FAIL_CONSTRAINT", f"turn-off 뒤 branch 차동 링잉 {dm_off:.3g} A가 합성 설계 기준 {dm_limit:.3g} A(branch 평균의 {v['dm_lim'] * 100:.3g} %, ASSUMED)를 넘는다 — 모듈 전류 합에는 보이지 않는 branch 사이 발진")
        res.warnings.append("branch 사이 차동 링잉은 모듈 전류(합)·단일 전류 probe로는 보이지 않는다: branch-resolved 측정이 필요하다.")
    res.assumptions += [
        "4 branch 모두 같은 합성 die 모델(분산은 R_on·V_th·지연·layout 입력으로만)",
        "부하 인덕터 = 사건 동안 일정 전류원 I_total, 상측은 4 die 등가 하나",
        "branch drain inductance 상호결합은 양의 인접 결합 k_M^|i−j| (양의 정부호) — 식별 불가 값은 범위로",
        "power loop 감쇠는 R_p‖L_b 합성 등가, 고정 T_j, 역회복 없음",
    ]
    res.not_valid_for += ["특정 모듈·소자의 branch 전류 예측", "제조 분산으로부터의 수율 추정 (분포 입력 없음)", "기생 발진의 정확한 한계 (감쇠 등가 의존)", "EMI"]
    res.interpretation = (
        f"turn-on에서 branch 전류는 명령이 같아도 gate loop·V_th·drain/source inductance·driver 지연에 따라 다르게 오른다. 이 조건에서 branch peak 차는 {pk_on.max() - pk_on.min():.3g} A, "
        f"branch common-source 전압은 최대 {csi.max():.3g} V(screen 4 V와 같은 규모)다. 10 ns × 2 kA/µs = 20 A는 규모 추정일 뿐이고 실제 차이는 결합과 commutation을 함께 풀어야 한다. "
        "switching 직후에는 인덕턴스가, 수 µs 뒤에는 저항이 분담을 정하므로 정적·동적 분담을 따로 검증한다."
    )
    return res


def _static_nonlinear(cell: Cell):
    """On-state DC sharing with the channel I-V of each branch (algebraic, independent of the transient)."""
    from scipy.optimize import brentq

    s, D = cell.s, cell.D
    f = lambda vv: float(np.sum(D.ich(np.full(cell.n, s.Von), np.full(cell.n, vv)))) - s.IL  # noqa: E731
    vstar = brentq(f, 0.0, 200.0, xtol=1e-14)
    return D.ich(np.full(cell.n, s.Von), np.full(cell.n, vstar))


def _dyn_bands(t, idc, vmean, t_off, t_on, V, I, t_end):
    b = []

    def add(x0, x1, mode, label):
        if x0 is not None and x1 is not None and x1 > x0:
            b.append({"x0": float(x0), "x1": float(x1), "mode": mode, "label": label})

    v90 = first_cross(t, vmean, 0.9 * V, t_off, t_off + 400e-9, True)
    v10 = first_cross(t, vmean, 0.1 * V, t_off, t_off + 400e-9, True)
    i10 = first_cross(t, idc, 0.1 * I, v90 or t_off, t_off + 400e-9, False)
    add(0.5 * t_off, t_off, "on1", "ON")
    add(t_off, v10, "doff", "지연")
    add(v10, v90, "voff", "v↑")
    add(v90, i10, "ioff", "i↓")
    add(i10, (i10 or t_off) + 150e-9, "ring", "링잉")
    add((i10 or t_off) + 150e-9, t_on, "fw", "환류")
    i10n = first_cross(t, idc, 0.1 * I, t_on, t_on + 400e-9, True)
    v90n = first_cross(t, vmean, 0.9 * V, i10n or t_on, t_on + 400e-9, False)
    v10n = first_cross(t, vmean, 0.1 * V, v90n or t_on, t_on + 400e-9, False)
    add(t_on, i10n, "don", "지연")
    add(i10n, v90n, "ion", "i↑")
    add(v90n, v10n, "von", "v↓")
    add(v10n, t_end, "on2", "ON")
    return b


def _dyn_circuit(v: dict) -> Circuit:
    c = Circuit("parallel_dynamic", 920, 490, title="4 branch 병렬: 개별 gate loop, branch L_d(상호결합)·L_s, 공유 L_cs")
    c.add("vsource", "Vb", 70, 250, 90, "V_bus", f"{v['Vbus']:g} V", lpos=(88, 254, "start"))
    c.add("inductor", "Lloop", 160, 40, 0, "L_loop", "L_a + L_b‖R_p", lpos=(160, 14, "middle"))
    c.add("nmos", "HS", 260, 85, 90, "상측 (4 die 등가)", lpos=(286, 80, "start"))
    c.add("inductor", "LsH", 260, 145, 90, "L_sH", "", lpos=(274, 150, "start"))
    c.add("inductor", "Lload", 380, 110, 90, "L_load", f"I = {v['I_total']:g} A", lpos=(396, 104, "start"))
    c.wire("w_vtop", (70, 220), (70, 40), (130, 40))
    c.wire("w_l_hs", (190, 40), (260, 40), (260, 55))
    c.wire("w_top_load", (260, 40), (380, 40), (380, 80))
    c.wire("w_hs_lsh", (260, 115), (260, 115))
    c.wire("w_lsh_m", (260, 175), (260, 180))
    c.wire("w_load_m", (380, 140), (380, 180))
    xs = [470, 595, 720, 845]
    c.wire("w_mbus", (260, 180), (xs[-1], 180))
    c.dot((260, 40), (260, 180), (380, 180), *[(x, 180) for x in xs[:-1]])
    for k, x in enumerate(xs):
        c.add("inductor", f"Ld{k}", x, 225, 90, f"L_d{k + 1}", "", lpos=(x + 14, 228, "start"))
        c.add("nmos", f"Q{k}", x, 290, 90, f"Q{k + 1}", lpos=(x + 16, 286, "start"))
        c.add("inductor", f"Ls{k}", x, 360, 90, f"L_s{k + 1}", "", lpos=(x + 14, 364, "start"))
        c.add("resistor", f"Rg{k}", x - 40, 345, 90, f"R_g{k + 1}", "", lpos=(x - 50, 349, "end"))
        c.wire(f"w_m{k}", (x, 180), (x, 195))
        c.wire(f"w_d{k}", (x, 255), (x, 260))
        c.wire(f"w_s{k}", (x, 320), (x, 330))
        c.wire(f"w_cs{k}", (x, 390), (x, 405))
        c.wire(f"w_g{k}", (x - 26, 290), (x - 40, 290), (x - 40, 315))
        c.wire(f"w_gb{k}", (x - 40, 375), (x - 40, 385))
        c.text(x - 40, 398, "G", "node")
        c.probe(f"pi{k}", f"i_all{k}", x, 187, "down", f"i{k + 1}")
    c.wire("w_csbus", (xs[0], 405), (xs[-1], 405))
    c.add("inductor", "Lcs", 675, 440, 0, "L_cs (공유)", "", lpos=(675, 466, "middle"))
    c.wire("w_cs_l", (645, 405), (645, 440))
    c.wire("w_cs_r", (705, 440), (720, 440), (720, 475), (70, 475), (70, 280))
    c.add("block", "DRV", 175, 330, 0, "드라이버 → R_com → G", w=170, h=36)
    c.text(175, 372, "반환: " + {"common_source_node": "공통 source 노드", "common_kelvin_star": "각 Kelvin (R_e)", "power_ground": "DC−", "individual_kelvin": "개별 Kelvin"}.get(v["topology"], v["topology"]), "")
    c.text(175, 388, {"common_source_node": "(L_s,k가 gate loop 안)", "common_kelvin_star": "(L_s,k가 gate loop 밖)", "power_ground": "(L_s,k + L_cs가 gate loop 안)", "individual_kelvin": ""}.get(v["topology"], ""), "")
    c.text(560, 168, f"L_d 상호결합 k_M = {v['kM']:g} (인접), k_M² …", "")
    c.dot(*[(x, 405) for x in xs[1:-1]], (645, 405))
    power = ["Vb", "w_vtop", "Lloop", "w_l_hs", "w_top_load", "Lload", "w_load_m", "w_mbus", "w_csbus", "Lcs", "w_cs_l", "w_cs_r"] + [f"{p}{k}" for k in range(4) for p in ("Ld", "Q", "Ls", "w_m", "w_d", "w_s", "w_cs")]
    fw = ["Lload", "w_top_load", "w_load_m", "LsH", "w_hs_lsh", "HS", "w_lsh_m", "w_l_hs"]
    gate = ["DRV"] + [f"{p}{k}" for k in range(4) for p in ("Rg", "w_g", "w_gb")]
    c.mode("on1", "ON (1st pulse)", power + gate, "부하전류가 네 branch로 나뉘어 흐른다 (정적: 저항 분담)", dim=["HS"])
    c.mode("doff", "turn-off 지연", power + gate, "네 gate가 각자의 loop로 방전된다", dim=["HS"])
    c.mode("voff", "v 상승", power + gate + ["HS"], "Miller 구간: branch별 v_DS 상승", dim=[])
    c.mode("ioff", "i 하강", power + fw, "전류가 상측 diode로 넘어가며 branch별 L·di/dt가 다르다", dim=[])
    c.mode("ring", "링잉", power + fw, "공통 모드 링잉과 branch 사이 차동 링잉", dim=[])
    c.mode("fw", "환류", fw, "부하전류가 상측 diode로 돈다", dim=[f"Q{k}" for k in range(4)])
    c.mode("don", "turn-on 지연", fw + gate, "gate가 V_th까지 충전 (branch별 지연·V_th 차이)", dim=[f"Q{k}" for k in range(4)])
    c.mode("ion", "i 상승", power + fw + gate, "branch별 di/dt가 L_s,k·di/dt 되먹임과 skew로 달라진다", dim=[])
    c.mode("von", "v 하강", power + gate + ["HS"], "Miller 구간: 상측 C_oss 충전", dim=[])
    c.mode("on2", "ON (2nd pulse)", power + gate, "인덕턴스 분담 → 수 µs에 걸쳐 저항 분담으로 이완", dim=["HS"])
    return c


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "병렬 4개니까 모듈 정격은 소자 정격 × 4입니까?",
        "아니다. 정적 저항 분산만으로도 3.6/4.0/4.0/4.4 mΩ이면 400 A에서 110.55/99.50/99.50/90.45 A로 나뉜다. turn-on에는 gate loop·common-source·timing skew·결합 inductance가 branch 전류를 다르게 만들고, "
        "열 경로 차이는 같은 전류에서도 온도를 다르게 한다. 가장 약한 branch 기준으로 정적·동적 분담을 따로 검증한다.",
        "Four devices in parallel, so is the module rating four times the device rating?",
        "No. Static resistance spread alone gives 110.55, 99.50, 99.50 and 90.45 A at 400 A, and during switching the gate loops, common-source inductance, timing skew and coupled inductance make the branch currents differ again. The weakest branch sets the limit, and static and dynamic sharing are verified separately.",
        ["정적 분담 수치", "동적 요인", "weakest branch", "정적·동적 분리 검증"],
        kind="pressure",
    ),
    Question(
        "동일 lot만 쓰면 전류 불균형이 해결됩니까?",
        "lot 분산과 layout/gate 비대칭의 기여를 분리하기 전에는 단정하지 않는다. 부품 선별은 조립·배선 원인을 가릴 수 있다. branch별 waveform(같은 timebase·deskew), 재현실험, 공급사 분산자료로 해결책의 범위를 정한다.",
        "Will using parts from the same lot fix the current imbalance?",
        "Not before separating the lot spread from layout and gate-loop asymmetry; screening parts can hide an assembly or wiring cause. I would compare branch-resolved waveforms on a deskewed common timebase, repeat the test, and use the supplier's spread data to bound the fix.",
        ["lot vs layout 분리", "선별이 원인을 가림", "branch별 파형", "공급사 분산자료"],
        kind="pressure",
    ),
    Question(
        "DPT에서 current probe가 5 ns 어긋나면 E_on이 얼마나 틀립니까?",
        "교재의 합성 40 ns 선형 overlap(E₀ = 0.533 mJ)에서 전류 채널 −5 ns는 +41.99 %, +5 ns는 −33.01 %다. 창이 전이 전 구간(v = V, 전류 ≠ 0)을 포함해야 +42 %가 나오므로 창 정의도 함께 저장한다. "
        "branch 간 E 차이가 이 불확도보다 작으면 die 선별이나 R_g 변경 결론을 서두르지 않는다.",
        "If the current probe is 5 ns off in a double-pulse test, how wrong is E_on?",
        "On the textbook's synthetic 40 ns overlap, shifting the current 5 ns earlier gives plus 42 percent and 5 ns later minus 33 percent, with a window that includes the interval before the voltage starts to fall. So the deskew and the window definition are stored with the result, and differences smaller than that uncertainty are not used to rank parts.",
        ["+41.99 % / −33.01 %", "창 정의", "순위 보류"],
        kind="calc",
    ),
    Question(
        "공통 source inductance 2 nH, di/dt 2 kA/µs면 gate 전압이 4 V 줄어듭니까?",
        "L_s·di/dt = 4 V는 1차 크기 screen이다. 그 전압이 gate loop 안에 있는지(드라이버 반환 위치), 극성, 실제 di/dt가 얼마인지에 따라 되먹임이 달라 모든 V_GS에 단순 가산하지 않는다. "
        "측정도 Kelvin 기준인지 power-source 기준인지 먼저 적는다.",
        "With 2 nH of common-source inductance and 2 kA/µs, does the gate lose 4 V?",
        "Four volts is a first-order screen. Whether it acts on the gate depends on where the driver returns, on the polarity and on the actual di/dt, so it is not simply added to every gate voltage; and every gate measurement states its reference, Kelvin or power source.",
        ["1차 screen", "반환 위치", "측정 기준"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="static_sharing",
        title="정적 4-branch 분담: 평균 100 A가 가리는 110.55 A",
        goal=(
            "R = [3.6, 4.0, 4.0, 4.4] mΩ, 400 A에서 branch 전류 110.553/99.497/99.497/90.452 A와 branch 손실을 계산하고, 양의 온도계수와 branch별 열 경로를 넣었을 때 "
            "분담이 얼마나(일부만) 좋아지는지 본다."
        ),
        params=[
            Param("R1", "branch 1 R", "Ω", 3.6e-3, "mΩ", vmin=1e-4, vmax=1.0, source="TEXTBOOK", source_note="3.6 mΩ", group="정적 R"),
            Param("R2", "branch 2 R", "Ω", 4.0e-3, "mΩ", vmin=1e-4, vmax=1.0, source="TEXTBOOK", source_note="4.0 mΩ", group="정적 R"),
            Param("R3", "branch 3 R", "Ω", 4.0e-3, "mΩ", vmin=1e-4, vmax=1.0, source="TEXTBOOK", source_note="4.0 mΩ", group="정적 R"),
            Param("R4", "branch 4 R", "Ω", 4.4e-3, "mΩ", vmin=1e-4, vmax=1.0, source="TEXTBOOK", source_note="4.4 mΩ", group="정적 R"),
            Param("I_total", "총 전류", "A", 400.0, "A", vmin=1, vmax=5000, source="TEXTBOOK", source_note="400 A", group="정적 R"),
            Param("alpha_R", "R 온도계수 α", "1/K", 0.004, "1/K", vmin=0, vmax=0.02, source="ASSUMED", source_note="양의 온도계수 (합성)", group="전열"),
            Param("Tc", "냉각 경계 온도", "°C", 65.0, "°C", vmin=-40, vmax=150, source="ASSUMED", group="전열"),
            Param("Rth1", "branch 1 R_th", "K/W", 0.5, "K/W", vmin=0.01, vmax=10, source="ASSUMED", group="전열"),
            Param("Rth2", "branch 2 R_th", "K/W", 0.5, "K/W", vmin=0.01, vmax=10, source="ASSUMED", group="전열"),
            Param("Rth3", "branch 3 R_th", "K/W", 0.5, "K/W", vmin=0.01, vmax=10, source="ASSUMED", group="전열"),
            Param("Rth4", "branch 4 R_th", "K/W", 0.5, "K/W", vmin=0.01, vmax=10, source="ASSUMED", group="전열"),
        ],
        presets=[
            Preset("textbook", "교재 R, 400 A, α 0.004/K", {}, "E03 합성 예", ("nominal", "reference")),
            Preset("no_tempco", "온도계수 0 (등온)", {"alpha_R": 0.0}, "정적 분담만", ("variant", "reference")),
            Preset("bad_tim", "branch 1 R_th 0.8 K/W (TIM 불량)", {"Rth1": 0.8}, "열 경로 차이", ("corner",)),
        ],
        run=run_static,
        model_level="A (정적 저항 + 전열)",
        suggested_change="branch 1의 R_th를 0.5 → 0.8 K/W로 올린다 (TIM 불량 가정).",
        prediction=Prediction(
            "branch 1의 열 경로만 나빠지면(R_th 0.8 K/W) branch 1의 전류와 온도는?",
            ["전류 증가, 온도 증가", "전류 약간 감소, 온도는 가장 높음", "변화 없음", "모르겠다"],
            "전류 약간 감소, 온도는 가장 높음",
            "더 뜨거워진 branch 1의 R이 양의 온도계수로 커져 전류가 조금 줄지만, 열 경로가 나빠 온도는 여전히 가장 높다. 뜨거운 branch 하나가 곧 chip 불량은 아니다.",
            ["Iet1", "T_hot", "dT_branch"],
            handcalc=[{"key": "I1", "label": "branch 1 전류 (등온)", "unit": "A"}, {"key": "I4", "label": "branch 4 전류 (등온)", "unit": "A"}],
        ),
        suggested={"Rth1": 0.8},
        student="병렬 저항에 걸린 전압은 같으므로 저항이 작은 쪽으로 전류가 더 흐른다. 저항이 10 % 작으면 전류가 약 10 % 더 흐르고, 손실은 I²R이라 차이가 더 벌어진다.",
        expert=(
            "① 정적 분담은 공통 단자전압에서 컨덕턴스 비례다. ② 양의 온도계수는 부분적 자기 균형일 뿐 — thermal coupling·package impedance·동적 gate 불일치를 해결하지 않는다. "
            "③ 열 경로(R_th)가 다르면 같은 전류에서도 온도가 달라 ‘뜨거운 branch = 불량 chip’이 아니다. ④ 모듈 평균값으로 weakest branch를 숨기지 않는다."
        ),
        customer_ko="3.6 mΩ branch가 약 110.6 A를 흘려 평균보다 10 % 많습니다. 양의 온도계수가 일부 보정하지만 충분하지 않으니, branch별 전류·온도를 따로 측정해 가장 약한 branch 기준으로 판단하시죠.",
        customer_en="The 3.6 mΩ branch carries about 110.6 A, ten percent above the average. The positive temperature coefficient corrects only part of that, so let's measure branch currents and temperatures individually and judge by the weakest branch.",
        questions=[_Q[0], _Q[1]],
        circuit="parallel_static",
        textbook=[TB_E03, TB_E12],
        reference_presets=["textbook", "no_tempco"],
        claim_limit="정적 저항·branch별 열 경로의 분담. 동적 분담·수율 추정은 주장하지 않는다.",
    ),
    Experiment(
        key="dpt_deskew",
        title="DPT 계측: 5 ns 어긋남이 E를 −33 %~+42 % 바꾼다",
        goal=(
            "교재 합성 turn-on(40 ns, 800 V·100 A)의 E₀ = 0.533333 mJ와 전류 채널 −5 ns +41.99 %, +5 ns −33.01 %를 정확 적분으로 재현한다. 창이 전이 전 구간을 포함해야 하는 이유, "
            "gain·offset·대역폭 민감도, deskew 불확도 아래의 순위 판정, 단자 vs die 에너지 참조면을 확인한다."
        ),
        params=[
            Param("V", "V_DS (전이 전)", "V", 800.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="800 V", group="합성 파형"),
            Param("I", "I_D (전이 후)", "A", 100.0, "A", vmin=0.1, vmax=2000, source="TEXTBOOK", source_note="100 A", group="합성 파형"),
            Param("tr", "전이 시간 t_r", "s", 40e-9, "ns", vmin=2e-9, vmax=1e-6, source="TEXTBOOK", source_note="40 ns", group="합성 파형"),
            Param("skew", "전류 채널 skew (저장된 설정)", "s", 0.0, "ns", vmin=-20e-9, vmax=20e-9, source="ASSUMED", source_note="−: 전류가 앞당겨짐", group="측정 설정"),
            Param("window", "적분창", "", "wide", kind="choice", choices=[("wide", "넓은 고정창"), ("narrow", "좁은 창 [0, t_r]"), ("threshold", "문턱창 10 %/2 %")], source="ASSUMED", group="측정 설정"),
            Param("W_pre", "넓은 창: 전이 전 여유", "s", 10e-9, "ns", vmin=0, vmax=200e-9, source="ASSUMED", group="측정 설정"),
            Param("W_post", "넓은 창: 전이 후 여유", "s", 10e-9, "ns", vmin=0, vmax=200e-9, source="ASSUMED", group="측정 설정"),
            Param("gain", "전류 probe gain 오차", "", 0.0, "", vmin=-0.2, vmax=0.2, source="ASSUMED", group="측정 설정"),
            Param("off_i", "전류 offset", "A", 0.0, "A", vmin=-20, vmax=20, source="ASSUMED", group="측정 설정"),
            Param("off_v", "전압 offset", "V", 0.0, "V", vmin=-50, vmax=50, source="ASSUMED", group="측정 설정"),
            Param("skew_unc", "deskew 불확도 ±", "s", 0.5e-9, "ns", vmin=0, vmax=20e-9, source="ASSUMED", group="판정"),
            Param("dE_branch", "비교할 두 branch의 E 차이", "", 0.10, "", vmin=0, vmax=2.0, source="ASSUMED", group="판정"),
        ],
        presets=[
            Preset("textbook", "정렬, 넓은 창, deskew ±0.5 ns", {}, "E03 합성 overlap", ("nominal", "reference")),
            Preset("skew_m5", "전류 채널 −5 ns", {"skew": -5e-9}, "+41.99 %", ("variant", "reference")),
            Preset("narrow_m5", "−5 ns + 좁은 창", {"skew": -5e-9, "window": "narrow"}, "창이 전이 전 구간을 놓침", ("failure",)),
            Preset("poor_deskew", "deskew 불확도 ±5 ns", {"skew_unc": 5e-9}, "순위 확정 불가", ("failure", "reference")),
        ],
        run=run_dpt_deskew,
        model_level="A (합성 선형 overlap) + D (FL02 셀 참조면)",
        suggested_change="전류 채널 skew를 0 → −5 ns로 바꾼다 (그 다음 창을 ‘좁은 창’으로).",
        prediction=Prediction(
            "전류 채널을 5 ns 앞당기면(−5 ns) 넓은 창의 E는?",
            ["약 −33 %", "거의 그대로", "약 +42 %", "모르겠다"],
            "약 +42 %",
            "전압이 아직 800 V인 동안 전류가 흐르기 시작해 겹침이 커진다: E = VI s²/t_r + … = 757.3 µJ (+41.99 %). 좁은 창 [0, t_r]로 적분하면 전이 전 구간을 빼먹어 +37 %로 나온다.",
            ["dE_m5", "dE_p5", "dE_m5_narrow"],
            handcalc=[{"key": "E0", "label": "E₀ = V·I·t_r/6", "unit": "J"}],
        ),
        suggested={"skew": -5e-9},
        student="스위칭 손실은 전압과 전류가 겹치는 면적이다. 두 파형을 다른 probe로 재므로, 한쪽이 몇 ns만 늦게 도착해도 겹치는 면적이 크게 달라진다.",
        expert=(
            "① E = ∫v·i dt는 두 채널의 정렬에 1차로 민감하다(선형 overlap에서 −5 ns +42 %). ② 창 정의(고정·문턱)와 전이 전 구간 포함 여부를 결과와 함께 저장한다. "
            "③ gain은 비례, offset은 창 길이만큼 누적, 대역 제한은 지연(τ = 1/2πf)과 왜곡을 함께 만든다 — 하나씩 바꿔 민감도를 본다. "
            "④ 단자 적분에는 C_oss 저장·방출이 들어가 die 소산과 다르다(참조면 표). ⑤ branch E 차이가 불확도보다 작으면 UNRESOLVED_RANKING."
        ),
        customer_ko="현재 셋업의 deskew 불확도가 ±5 ns라면 E가 −33 %~+42 % 범위로 흔들립니다. branch 간 10 % 차이는 이 안에 들어가므로 die 선별이나 R_g 변경 결론은 보류하고, probe 지연을 먼저 맞춘 뒤 같은 창 정의로 다시 비교하시죠.",
        customer_en="With a deskew uncertainty of plus or minus 5 ns the energy can move between minus 33 and plus 42 percent, so a 10 percent difference between branches is inside the noise. Let's hold the die-screening or gate-resistor conclusion, align the probe delays first, and compare again with the same window definition.",
        questions=[_Q[2]],
        circuit="dpt_measure",
        textbook=[TB_E03, TB_05],
        reference_presets=["textbook", "skew_m5", "poor_deskew"],
        runtime_hint="seconds",
        claim_limit="합성 선형 파형의 측정 민감도와 합성 셀의 참조면. 실제 probe·scope 오차를 주장하지 않는다.",
    ),
    Experiment(
        key="dynamic_sharing",
        title="동적 분담: 4 branch gate loop·common-source·skew·결합 (합성 D)",
        goal=(
            "네 branch가 각자의 gate loop(R_g,k·L_g,k), branch common-source inductance, 결합된 drain inductance, V_th·타이밍 분산을 가진 합성 셀로 double-pulse를 풀어 "
            "branch별 전류·local v_GS를 보고, 대수 screen(L_s·di/dt = 4 V, 10 ns × 2 kA/µs = 20 A)과 비교한다. 식별 불가 상호결합은 범위로 보고한다."
        ),
        params=[
            Param("R1", "branch 1 R_on", "Ω", 3.6e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", source_note="정적 R 값", group="branch 분산"),
            Param("R2", "branch 2 R_on", "Ω", 4.0e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", group="branch 분산"),
            Param("R3", "branch 3 R_on", "Ω", 4.0e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", group="branch 분산"),
            Param("R4", "branch 4 R_on", "Ω", 4.4e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", group="branch 분산"),
            Param("dVth1", "branch 1 ΔV_th", "V", 0.0, "V", vmin=-1.5, vmax=1.5, source="ASSUMED", group="branch 분산"),
            Param("dVth2", "branch 2 ΔV_th", "V", 0.0, "V", vmin=-1.5, vmax=1.5, source="ASSUMED", group="branch 분산"),
            Param("dVth3", "branch 3 ΔV_th", "V", 0.0, "V", vmin=-1.5, vmax=1.5, source="ASSUMED", group="branch 분산"),
            Param("dVth4", "branch 4 ΔV_th", "V", 0.0, "V", vmin=-1.5, vmax=1.5, source="ASSUMED", group="branch 분산"),
            Param("d1", "branch 1 드라이버 지연", "s", 0.0, "ns", vmin=0, vmax=50e-9, source="ASSUMED", group="branch 분산"),
            Param("d2", "branch 2 드라이버 지연", "s", 0.0, "ns", vmin=0, vmax=50e-9, source="ASSUMED", group="branch 분산"),
            Param("d3", "branch 3 드라이버 지연", "s", 0.0, "ns", vmin=0, vmax=50e-9, source="ASSUMED", group="branch 분산"),
            Param("d4", "branch 4 드라이버 지연", "s", 0.0, "ns", vmin=0, vmax=50e-9, source="ASSUMED", source_note="교재 skew 10 ns 비교용", group="branch 분산"),
            Param("I_total", "부하전류 (4 branch 합)", "A", 400.0, "A", vmin=10, vmax=2000, source="TEXTBOOK", source_note="400 A", group="회로"),
            Param("Vbus", "V_bus", "V", 800.0, "V", vmin=50, vmax=1500, source="ASSUMED", group="회로"),
            Param("scale", "branch die 크기 (FL02 die 대비)", "", 3.0, "", vmin=1, vmax=10, source="ASSUMED", source_note="C·g_m 비례 (합성)", group="회로"),
            Param("Vth", "V_th (공칭)", "V", 4.0, "V", vmin=1, vmax=8, source="ASSUMED", group="회로"),
            Param("topology", "gate 반환 구조", "", "common_source_node", kind="choice", choices=[("common_source_node", "공통 source 노드"), ("common_kelvin_star", "Kelvin star (R_e)"), ("power_ground", "DC− (L_cs 포함)")], source="ASSUMED", source_note="공통 source 노드: branch L_s가 gate loop 안 (common-source inductance)", group="gate"),
            Param("Rg", "branch 개별 R_g", "Ω", 12.0, "Ω", vmin=0.5, vmax=100, source="ASSUMED", group="gate"),
            Param("Rcom", "공통 R_com", "Ω", 0.5, "Ω", vmin=0, vmax=20, source="ASSUMED", group="gate"),
            Param("Re", "Kelvin 반환 R_e", "Ω", 0.5, "Ω", vmin=0, vmax=10, source="ASSUMED", group="gate"),
            Param("Lg", "branch gate loop L_g", "H", 8e-9, "nH", vmin=1e-9, vmax=100e-9, source="ASSUMED", group="gate"),
            Param("Von", "gate on", "V", 18.0, "V", vmin=5, vmax=25, source="ASSUMED", group="gate"),
            Param("Voff", "gate off", "V", -4.0, "V", vmin=-10, vmax=0, source="ASSUMED", group="gate"),
            Param("La", "외부 loop L_a", "H", 2e-9, "nH", vmin=0.1e-9, vmax=100e-9, source="ASSUMED", group="power loop"),
            Param("Lb", "손실성 loop L_b", "H", 4e-9, "nH", vmin=0.1e-9, vmax=50e-9, source="ASSUMED", group="power loop"),
            Param("Rp", "감쇠 R_p (L_b와 병렬)", "Ω", 0.8, "Ω", vmin=0.01, vmax=50, source="ASSUMED", group="power loop"),
            Param("Ld", "branch drain inductance L_d", "H", 3e-9, "nH", vmin=0.1e-9, vmax=50e-9, source="ASSUMED", group="power loop"),
            Param("kM", "인접 branch 상호결합 k_M", "", 0.3, "", vmin=0.0, vmax=0.8, source="ASSUMED", source_note="식별 불가 — 범위로 보고", group="power loop"),
            Param("kM_min", "k_M 범위 하한", "", 0.0, "", vmin=0.0, vmax=0.8, source="ASSUMED", group="power loop"),
            Param("kM_max", "k_M 범위 상한", "", 0.6, "", vmin=0.0, vmax=0.8, source="ASSUMED", group="power loop"),
            Param("dLd4", "branch 4 추가 L_d (먼 경로)", "H", 0.0, "nH", vmin=0, vmax=20e-9, source="ASSUMED", group="power loop"),
            Param("Ls", "branch common-source L_s", "H", 2e-9, "nH", vmin=0.05e-9, vmax=20e-9, source="TEXTBOOK", source_note="교재 L_s = 2 nH", group="power loop"),
            Param("Lcs", "공유 source L_cs", "H", 1e-9, "nH", vmin=0, vmax=20e-9, source="ASSUMED", group="power loop"),
            Param("t_gap", "두 펄스 사이 (환류) 시간", "s", 1e-6, "µs", vmin=0.5e-6, vmax=5e-6, source="ASSUMED", group="시뮬레이션"),
            Param("t_hold", "두 번째 펄스 관찰 시간", "s", 2e-6, "µs", vmin=0.5e-6, vmax=10e-6, source="ASSUMED", group="시뮬레이션"),
            Param("dm_lim", "branch 차동 링잉 허용 (branch 평균 전류 대비)", "", 0.2, "", vmin=0.01, vmax=1.0, source="ASSUMED", source_note="합성 설계 기준 — 실제 한계는 소자·신뢰성 요구가 정한다", group="판정"),
        ],
        presets=[
            Preset("nominal", "공통 드라이버, branch L_s 2 nH, R_g 12 Ω", {}, "교재 2 kA/µs 영역", ("nominal", "reference")),
            Preset("skew10", "branch 4 드라이버 10 ns 지연", {"d4": 10e-9}, "timing skew", ("variant", "reference")),
            Preset("vth", "V_th 분산 ±0.3 V (branch 1 −, 4 +)", {"dVth1": -0.3, "dVth4": 0.3}, "V_th 분산", ("variant",)),
            Preset("layout", "branch 4 L_d +2 nH", {"dLd4": 2e-9}, "layout 비대칭", ("corner",)),
            Preset("kelvin", "Kelvin star 반환", {"topology": "common_kelvin_star"}, "L_s가 gate loop 밖", ("variant", "reference")),
            Preset("low_rg", "R_g 2 Ω (빠른 구동)", {"Rg": 2.0}, "branch 사이 차동 발진", ("failure",)),
        ],
        run=run_dynamic,
        model_level="D (합성 4-branch commutation cell)",
        suggested_change="branch 4 드라이버를 10 ns 늦춘다 (다음에 R_g를 12 → 2 Ω으로).",
        prediction=Prediction(
            "branch 4만 10 ns 늦게 켜지면(branch di/dt ≈ 2–3 kA/µs) turn-on peak 전류 차이는?",
            ["거의 0", "수~수십 A (Δt·di/dt 규모)", "수백 A", "모르겠다"],
            "수~수십 A (Δt·di/dt 규모)",
            "먼저 켜진 세 branch가 전류를 먼저 가져가므로 차이는 1차로 Δt·di/dt(10 ns × 2 kA/µs = 20 A) 규모다. 실제 값은 결합 inductance·common-source 되먹임·commutation이 함께 정한다.",
            ["pk_spread", "skew_sim_dI", "csi_sim"],
        ),
        suggested={"d4": 10e-9},
        student="네 소자에 같은 명령을 줘도 gate까지 가는 길(저항·인덕턴스)과 문턱전압이 조금씩 달라 켜지는 순간이 다르다. 먼저 켜진 소자가 전류를 먼저 가져가므로 순간 전류가 불균형해진다.",
        expert=(
            "① 동적 분담은 인덕턴스(L_d 결합·L_s)와 타이밍이 정하고, 정적 분담은 저항이 정한다 — 수 µs의 L/R 이완으로 넘어간다. ② branch common-source inductance가 gate loop 안이면 L_s·di/dt가 구동을 깎아 di/dt를 스스로 제한한다(screen 4 V). "
            "③ 10 ns × 2 kA/µs = 20 A는 규모 추정이며 결합 모델 결과와 같은 규모다. ④ 식별 불가 상호결합은 범위로 보고. ⑤ 빠른 구동(R_g 작음)에서 branch 사이 차동 링잉·발진이 생기고 모듈 전류 합에는 보이지 않는다 — 개별 R_g가 감쇠시키지만 지연·손실이 바뀐다. "
            "⑥ 합성 D 수준: vendor 소자 모델이 아니다."
        ),
        customer_ko=(
            "모듈 전류만으로는 어느 branch가 먼저 stress를 받는지 알 수 없습니다. branch 전류와 local v_GS(Kelvin 기준)를 같은 timebase로 deskew해서 보고, 정적 분산·gate timing·common-source 결합·열 경로를 하나씩 분리하겠습니다. "
            "개별 R_g 조정은 발진·분담에 도움이 되지만 지연과 스위칭 손실이 바뀌니 같은 worst-case 조건으로 비교하시죠."
        ),
        customer_en=(
            "The module current alone does not show which branch is stressed first. I would measure branch currents and local gate voltages to Kelvin source on one deskewed timebase, then separate static spread, gate timing, common-source coupling and the thermal path one at a time. Individual gate resistors help oscillation and sharing but change delay and switching loss, so compare them under the same worst case."
        ),
        questions=[_Q[3], _Q[0]],
        circuit="parallel_dynamic",
        textbook=[TB_E03, TB_E12],
        reference_presets=["nominal", "skew10", "kelvin"],
        runtime_hint="seconds",
        claim_limit="bounded synthetic equivalent (D). vendor 소자·모듈의 branch 전류 예측·수율을 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="EX03",
    title="병렬 SiC·gate loop·DPT 계측",
    title_en="Parallel SiC, gate loop and double-pulse metrology",
    track="expert",
    order=3,
    path_note="E13 1회전 (E01–E03)",
    textbook=[TB_E03, TB_E12],
    prerequisites=["FL02", "FL04"],
    summary="정적 분담(110.553 A…) → DPT deskew·창·gain·offset·대역 민감도(+41.99 %/−33.01 %) → 4 branch 동적 분담(실제 상태의 합성 D). 정적·common-source·skew·deskew를 섞지 않는다.",
    experiments=EXPERIMENTS,
    minimum_scope="static sharing, deskew sensitivity, dynamic model 범위 (E13 표); 110.553/99.497/99.497/90.452 A; 533.333 µJ, +41.99 %/−33.01 %; L_s·di/dt 4 V, 10 ns × 2 kA/µs = 20 A",
    claim_limits=[
        "정적 분담은 저항 모델 — 동적 분담과 따로 검증",
        "DPT 민감도는 교재 합성 선형 파형 — 실제 probe·scope 오차 아님",
        "동적 분담은 실제 상태·inductance·gate 결합을 가진 bounded synthetic equivalent (D) — vendor fidelity 아님",
        "식별 불가 상호결합은 범위로만 보고",
        "branch 전류 예측·제조 수율을 주장하지 않음",
    ],
    implementation="COMPLETE",
    implementation_note="동적 분담은 대수 screen이 아니라 4-branch 상태방정식(branch 전류·gate loop·결합 inductance·common-source)으로 구현되었다 (합성 D 수준).",
    test_paths=["tests/test_ex03.py"],
    extends=["FL02", "FL04"],
)
