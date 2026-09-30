"""EX07 - A stable loop can still make an unstable system (textbook E07, extends FL05/FL06).

Four experiments, each with its own model level:
  A+B  the exact source R-L + node C + ideal constant-power-load example: analytic equilibrium and
       poles, a numerical Jacobian of the nonlinear model at a numerically found equilibrium, and a
       nonlinear integration (scipy DOP853) from a perturbed equilibrium; C sweep / root locus;
       P > Vs^2/(4R) has no equilibrium.  The ideal CPL is infinite bandwidth: the integration stops
       honestly where the ideal model leaves validity.
  B    a finite-bandwidth regulated converter (averaged buck, inner current PI, outer voltage PI,
       optional input-voltage feedforward) behind the same filter: its input impedance Z_in(jw),
       the filter output impedance Z_s(jw), minor-loop gain Z_s/Z_in with a Nyquist count checked
       against the eigenvalues of the whole linearised system and a nonlinear time simulation;
       voltage/load corners; the converter's own loop margins reported separately from whole-system
       stability.
  B    digital timing: ADC sample instant, computation latency and PWM update define the actual
       delay; its phase lag 360 f T_d is checked by simulating the sample/hold chain; the exact
       sampled current loop with a fractional delay gives the current-loop margin.
  B    dq vector saturation (circle limit vs per-axis clamp) with anti-windup on the applied vector,
       and bumpless transfer from a protection mode back to the PI.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq, fsolve

from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import control as cref
from ..reference import cpl as ref
from ._common import energy_ledger, ledger_check, sym_linear
from .fl06_control import dq_plant_map, loop_gain_cont, margins, rl_zoh

TB_E07 = TextbookRef("expert-e07-제어-단일-루프가-안정해도-시스템은-불안정할-수-있다", "E07. 제어: 단일 루프가 안정해도 시스템은 불안정할 수 있다 [EX07]")
TB_09 = TextbookRef("제어-pi보다-plant와-부호가-먼저다-fl06", "09. 제어 — PI보다 plant와 부호가 먼저다 [FL06]")
TB_08 = TextbookRef("obcpfc-전력품질을-설계-변수로-바꾸기-fl05", "08. OBC·PFC — 전력품질을 설계 변수로 바꾸기 [FL05]")
TB_E13 = TextbookRef("expert-e13-빠르게-깊어지기-위한-실행-순서와-통과-기준", "E13. 실행 순서와 통과 기준")

SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)


# ======================================================================================
# Experiment 1: source R-L + C + ideal CPL (the exact textbook example)
# ======================================================================================


def cpl_rhs(Vs, R, L, C, P):
    def f(t, x):
        i, v = x
        return [(Vs - R * i - v) / L, (i - P / v) / C]

    return f


def peaks(sol, t0, t1, Ve, C, P, n_per):
    """Local maxima of v - V_e on the dense solution (derivative sign change, refined with brentq)."""
    tt = np.linspace(t0, t1, n_per)
    y = sol.sol(tt)
    dv = (y[0] - P / y[1]) / C
    out = []
    for k in range(1, tt.size):
        if dv[k - 1] > 0 >= dv[k]:
            f = lambda t: (sol.sol(t)[0] - P / sol.sol(t)[1]) / C  # noqa: E731
            try:
                tp = brentq(f, tt[k - 1], tt[k], xtol=1e-15, rtol=1e-13)
            except ValueError:
                tp = tt[k]
            out.append((tp, float(sol.sol(tp)[1] - Ve)))
    return out


def run_cpl_exact(v: dict) -> Result:
    res = Result("EX07", "cpl_exact", "A (해석 극점) + 비선형 ODE + 수치 Jacobian")
    Vs, R, L, C, P = v["Vs"], v["R"], v["L"], v["C"], v["P"]
    Pmax = Vs * Vs / (4 * R)
    tb = abs(Vs - 400) < 1e-9 and abs(R - 0.2) < 1e-12 and abs(L - 1e-3) < 1e-12 and abs(P - 10e3) < 1e-6
    res.add_metric("Pmax", "정적 공급 한계 P_max = V_s²/(4R)", Pmax, "W", ref=200e3 if (abs(Vs - 400) < 1e-9 and abs(R - 0.2) < 1e-12) else ref.p_max(Vs, R), ref_label="교재 200 kW" if abs(Vs - 400) < 1e-9 else "V_s²/(4R)", tol=1e-12)
    res.circuit = {"diagram": cpl_circuit(v).to_json(), "intervals": [], "plot_group": "cpl"}
    # P-V nose curve (both branches) - always shown
    Ps = np.linspace(0.0, Pmax, 201)
    hi = [(Vs + math.sqrt(max(Vs * Vs - 4 * R * p, 0.0))) / 2 for p in Ps]
    lo = [(Vs - math.sqrt(max(Vs * Vs - 4 * R * p, 0.0))) / 2 for p in Ps]
    res.add_series("pv_hi", "고전압 평형 (안정 가능 가지)", "V", Ps.tolist(), hi)
    res.add_series("pv_lo", "저전압 평형 (det < 0 안장점)", "V", Ps.tolist(), lo, dash=True)
    res.add_plot("p_pv", "정적 해: P–V 곡선 (평형점 존재 한계)", ["pv_hi", "pv_lo"], x_label="CPL 전력 P", x_unit="W", y_label="평형 전압 V_e", y_unit="V", kind="xy", level="A",
                 vlines=[{"x": Pmax, "label": f"P_max {Pmax / 1e3:.4g} kW"}, {"x": P, "label": f"P = {P / 1e3:.4g} kW"}] if P <= Pmax else [{"x": Pmax, "label": f"P_max {Pmax / 1e3:.4g} kW"}],
                 proved="평형점은 P ≤ V_s²/(4R)에서만 존재하고, 두 가지 중 저전압 가지는 det A < 0이라 항상 안장점이다.", not_yet="정적 공급 가능성은 동적 안정성과 다른 한계다(아래 C 조건).")
    if P > Pmax:
        # no equilibrium: show the collapse until the ideal CPL leaves validity
        f = cpl_rhs(Vs, R, L, C, P)
        ev = lambda t, x: x[1] - v["v_min_frac"] * Vs  # noqa: E731
        ev.terminal = True
        ev.direction = -1
        # load step from the 10 kW (or P_max/2) equilibrium to P at t = 0
        P0 = min(10e3, 0.5 * Pmax)
        V0 = (Vs + math.sqrt(Vs * Vs - 4 * R * P0)) / 2
        sol = solve_ivp(f, (0, v["t_end"]), [P0 / V0, V0], method="DOP853", rtol=1e-9, atol=1e-9, events=ev, dense_output=True, max_step=1e-5)
        tt = np.linspace(0, sol.t[-1], 400)
        res.add_series("v_col", f"v (t = 0에 {P0 / 1e3:g} kW 평형점에서 {P / 1e3:g} kW로 부하 step)", "V", tt.tolist(), sol.sol(tt)[1].tolist())
        res.add_plot("p_col", "평형점이 없을 때: 전압 붕괴 (이상 CPL 유효범위까지만)", ["v_col"], y_label="v", y_unit="V", level="B (비선형)", group="cpl",
                     hlines=[{"y": v["v_min_frac"] * Vs, "label": "모델 유효범위 하한"}],
                     proved="정상상태 해가 없으면 어떤 C에서도 전압이 유지되지 않는다(시뮬레이션은 유효범위 하한에서 멈춘다).", not_yet="실제 컨버터는 UVLO·전류 제한으로 CPL이 아니게 되며 그 동작은 모델 밖이다.")
        res.add_metric("t_col", "유효범위 하한 도달 시각 (부하 step 후)", float(sol.t[-1]), "s", basis="C가 전력 부족분을 공급하는 동안의 시간 — C와 전력 차이가 정한다")
        res.verdict("NO_SOLUTION", f"P = {P / 1e3:.4g} kW > P_max = {Pmax / 1e3:.4g} kW: V² − V_s·V + RP = 0의 실근이 없어 평형점이 존재하지 않는다.")
        res.assumptions += ["이상 CPL i = P/v (무한 대역), 선형 R·L·C"]
        res.not_valid_for += ["UVLO·전류 제한 이후의 실제 동작"]
        res.interpretation = "부하가 요구하는 전력이 소스가 R을 통해 줄 수 있는 최대 전력 V_s²/(4R)보다 크면 정상상태 해가 없다. 이것은 동적 안정성(C 조건)과 다른, 정적 공급 가능성의 한계다."
        return res
    Ve, Vlow = ref.equilibria(Vs, R, P) if False else ((Vs + math.sqrt(Vs * Vs - 4 * R * P)) / 2, (Vs - math.sqrt(Vs * Vs - 4 * R * P)) / 2)
    Ie = P / Ve
    g = P / Ve**2
    A = np.array([[-R / L, -1 / L], [1 / C, g / C]])
    lam = np.linalg.eigvals(A)
    lam = lam[np.argsort(-lam.imag)]
    Ccrit = L * g / R
    res.add_metric("Ve", "고전압 평형 V_e", Ve, "V", ref=394.935887 if tb else ref.equilibria(Vs, R, P)[0], ref_label="교재 394.935887 V" if tb else "(V_s + √(V_s² − 4RP))/2", tol=1e-8)
    res.add_metric("Ie", "평형 전류 I_e = P/V_e", Ie, "A")
    res.add_metric("Rinc", "증분 입력저항 dv/di = −V_e²/P", -1 / g, "Ω", ref=-15.5974 if tb else ref.incremental_resistance(Ve, P), ref_label="교재 −15.5974 Ω" if tb else "−V_e²/P", tol=5e-6)
    res.add_metric("Ccrit", "임계 C = L·g/R (tr A = 0)", Ccrit, "F", ref=320.565519e-6 if tb else ref.c_crit(Vs, R, L, P), ref_label="교재 320.565519 µF" if tb else "Lg/R", tol=1e-8)
    res.add_metric("Rg", "det 조건 R·g (< 1 필요)", R * g, "", basis="det A = (1 − Rg)/(LC) > 0")
    rl = ref.poles(Vs, R, L, C, P)
    tbc = tb and abs(C - 100e-6) < 1e-15
    tbc2 = tb and abs(C - 1e-3) < 1e-15
    r_re = 220.57 if tbc else (-67.94 if tbc2 else rl[0].real)
    r_im = 3134.19 if tbc else (991.24 if tbc2 else abs(rl[0].imag))
    lab_ = "교재" if (tbc or tbc2) else "해석식"
    # the textbook prints the poles with two decimals: tolerance = half a unit of the last printed digit
    tol_re = (0.005 / abs(r_re) + 1e-12) if (tbc or tbc2) else 1e-9
    tol_im = (0.005 / abs(r_im) + 1e-12) if (tbc or tbc2) else 1e-9
    res.add_metric("pole_re", "극점 실수부 σ", float(lam[0].real), "1/s", ref=r_re, ref_label=f"{lab_} {r_re:g} s⁻¹", tol=tol_re, basis="> 0 이면 불안정", note="교재는 소수 둘째 자리까지 표기 (허용오차 = 마지막 자리의 절반)" if (tbc or tbc2) else "")
    res.add_metric("pole_im", "극점 허수부 ω_d", float(abs(lam[0].imag)), "rad/s", ref=r_im, ref_label=f"{lab_} ±{r_im:g}", tol=tol_im)
    res.add_metric("f0", "진동 주파수 ω_d/2π", float(abs(lam[0].imag)) / (2 * math.pi), "Hz")
    # independent 1: numerical equilibrium + numerical Jacobian of the nonlinear model
    f = cpl_rhs(Vs, R, L, C, P)
    xe = fsolve(lambda x: f(0, x), [P / Vs, Vs], xtol=1e-14, full_output=False)
    J = np.zeros((2, 2))
    for j in range(2):
        h = 1e-6 * max(1.0, abs(xe[j]))
        xp, xm = xe.copy(), xe.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (np.array(f(0, xp)) - np.array(f(0, xm))) / (2 * h)
    lamJ = np.linalg.eigvals(J)
    lamJ = lamJ[np.argsort(-lamJ.imag)]
    res.add_check(check_close("평형점: fsolve(비선형 우변 = 0) vs 이차식 해", float(xe[1]), Ve, 1e-10, "Newton 계열 수치해 (고전압 쪽 초기값) vs 해석식", True, "V"))
    res.add_check(check_close("극점: 비선형 모델의 수치 Jacobian vs 해석 A", float(abs(lamJ[0] - lam[0])), 0.0, 1e-6, "중앙차분 Jacobian의 고유값 vs [[−R/L, −1/L], [1/C, g/C]]의 고유값", True, "1/s", abs_scale=abs(lam[0])))
    # independent 2: nonlinear integration from a perturbed equilibrium
    dv0 = v["dv0"]
    lim = v["dv_valid"] * Ve
    ev = lambda t, x: abs(x[1] - Ve) - lim  # noqa: E731
    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(f, (0, v["t_end"]), [Ie, Ve + dv0], method="DOP853", rtol=1e-11, atol=[1e-10, 1e-9], events=ev, dense_output=True)
    t_exit = float(sol.t_events[0][0]) if sol.t_events[0].size else None
    t_last = float(sol.t[-1])
    Td = 2 * math.pi / abs(lam[0].imag) if abs(lam[0].imag) > 0 else t_last
    pk = peaks(sol, 0.0, t_last, Ve, C, P, max(int(t_last / Td * 60), 200))
    small = [(t, a) for t, a in pk if abs(a) < 0.05 * Ve and a > 0]
    if len(small) >= 3:
        tp = np.array([t for t, _ in small])
        ap = np.array([a for _, a in small])
        sig = float(np.polyfit(tp, np.log(ap), 1)[0])
        wd = 2 * math.pi / float(np.mean(np.diff(tp)))
        res.add_metric("sig_nl", "비선형 적분에서 측정한 성장률 σ (봉우리 fit)", sig, "1/s", ref=float(lam[0].real), ref_label="선형 극점 σ", tol=0.02, abs_scale=0.05 * abs(lam[0]), basis=f"진폭 < 5 % V_e인 봉우리 {len(small)}개")
        res.add_metric("wd_nl", "비선형 적분에서 측정한 ω_d (봉우리 간격)", wd, "rad/s", ref=float(abs(lam[0].imag)), ref_label="선형 극점 ω_d", tol=0.01)
        res.add_check(check_close("성장·감쇠율: 비선형 ODE 봉우리 vs 선형 극점", sig, float(lam[0].real), 0.02, "DOP853 비선형 적분(1 V 교란)의 봉우리 로그 기울기 vs 고유값 실수부", True, "1/s", abs_scale=0.05 * abs(lam[0])))
    else:
        res.add_check(Check("성장·감쇠율: 비선형 ODE 봉우리 vs 선형 극점", "NOT_RUN", path="봉우리 3개 이상 필요", detail="진동이 과감쇠이거나 창이 짧다"))
    if t_exit is not None:
        res.add_metric("t_exit", f"|Δv|가 {v['dv_valid'] * 100:g} % V_e를 넘은 시각 (여기서 적분 중단)", t_exit, "s", note="이상 CPL 모델의 유효범위를 벗어난 뒤는 예측하지 않는다")
    # C sweep: numerical critical C (root of max Re) vs closed form
    def max_re(Cx):
        Ax = np.array([[-R / L, -1 / L], [1 / Cx, g / Cx]])
        return float(np.max(np.linalg.eigvals(Ax).real))

    Cnum = brentq(max_re, Ccrit / 20, Ccrit * 20, xtol=1e-18, rtol=1e-14)
    res.add_check(check_close("임계 C: max Re(λ(C)) = 0 수치 근 vs L·g/R", Cnum, Ccrit, 1e-9, "고유값 실수부의 부호가 바뀌는 C를 brentq로 탐색 vs trace 조건 해석식", True, "F"))
    Cs = np.geomspace(max(Ccrit / 8, 1e-6), Ccrit * 12, 90)
    reals, rex, rey = [], [], []
    for Cx in Cs:
        e2 = np.linalg.eigvals(np.array([[-R / L, -1 / L], [1 / Cx, g / Cx]]))
        reals.append(float(np.max(e2.real)))
        for z in e2:
            rex.append(float(z.real))
            rey.append(float(z.imag))
    res.add_series("re_C", "max Re(λ)", "1/s", Cs.tolist(), reals)
    res.add_plot("p_reC", "C에 따른 극점 실수부", ["re_C"], x_label="C", x_unit="F", y_label="max Re λ", y_unit="1/s", kind="xy", log_x=True, level="A",
                 hlines=[{"y": 0.0, "label": "안정 경계"}], vlines=[{"x": Ccrit, "label": f"C_crit {Ccrit * 1e6:.6g} µF"}, {"x": C, "label": "현재 C"}],
                 markers=[{"x": C, "y": float(lam[0].real), "label": f"{C * 1e6:.4g} µF"}],
                 proved="이상 CPL의 고전압 평형점은 C > L·g/R에서만 안정하고, 이 조건은 전력 공급 한계(P_max)와 별개다.",
                 not_yet="이상 무한대역 CPL의 국소(선형) 결과다. 실제 컨버터의 유한 대역·포화는 실험 2·4에서 본다.")
    res.add_series("rl", "극점 궤적 (C sweep)", "rad/s", rex, rey, style="points")
    res.add_plot("p_rl", "극점 궤적 (복소평면)", ["rl"], x_label="Re λ", x_unit="1/s", y_label="Im λ", y_unit="rad/s", kind="xy", level="A",
                 vlines=[{"x": 0.0, "label": "jω 축"}], markers=[{"x": float(lam[0].real), "y": float(lam[0].imag), "label": f"C = {C * 1e6:.4g} µF"}],
                 proved="C가 작아질수록 극점쌍이 jω 축을 넘어 우반평면으로 간다.", not_yet="같은 L·R·P에서 C만 바꾼 궤적이다. 부하 전력·소스 임피던스가 바뀌면 궤적 전체가 이동한다.")
    # time response: nonlinear vs linear
    tt = np.linspace(0.0, t_last, 1200)
    yy = sol.sol(tt)
    res.add_series("dv_nl", "Δv 비선형 (DOP853)", "V", tt.tolist(), (yy[1] - Ve).tolist())
    xl = [expm(A * t) @ np.array([0.0, dv0]) for t in tt]
    res.add_series("dv_lin", "Δv 선형 e^(At)·x₀", "V", tt.tolist(), [x[1] for x in xl], dash=True)
    res.add_series("di_nl", "Δi 비선형", "A", tt.tolist(), (yy[0] - Ie).tolist())
    vl = [{"x": t_exit, "label": "유효범위 이탈 → 중단"}] if t_exit is not None else []
    res.add_plot("p_dv", f"평형점 + {dv0:g} V 교란 응답", ["dv_nl", "dv_lin"], y_label="v − V_e", y_unit="V", level="B (비선형) + A", group="cpl", vlines=vl,
                 hlines=[{"y": lim, "label": "유효범위"}, {"y": -lim, "label": ""}],
                 proved="비선형 적분의 성장/감쇠율과 진동수가 선형 극점과 일치한다(작은 진폭). 불안정이면 교란이 지수적으로 자란다.",
                 not_yet="진폭이 커진 뒤의 실제 전압 붕괴·UVLO·전류 제한은 이 이상 CPL 모델로 예측하지 않는다(유효범위에서 중단).")
    res.add_plot("p_di", "전류 교란 Δi", ["di_nl"], y_label="i − I_e", y_unit="A", level="B", group="cpl", vlines=vl,
                 proved="전압 교란과 같은 진동수·성장률로 소스 전류가 진동한다(L–C 공진 모드).", not_yet="전류 제한·보호 동작은 모델에 없다.")
    res.tables.append(
        Table(
            "t_cond",
            "두 가지 서로 다른 한계",
            ["조건", "식", "값", "의미"],
            [
                ["정적 공급 (평형점 존재)", "P ≤ V_s²/(4R)", f"{Pmax / 1e3:.4g} kW", "넘으면 NO_SOLUTION"],
                ["동적 안정 (trace)", "C > L·g/R", f"{Ccrit * 1e6:.6g} µF", "작으면 UNSTABLE (평형점은 존재)"],
                ["동적 안정 (det)", "R·g < 1", f"R·g = {R * g:.4g}", "고전압 가지에서 만족"],
            ],
            note="P = 10 kW는 P_max 200 kW보다 훨씬 작지만 C = 100 µF에서 동적으로 불안정하다. C 증가는 이 예제의 해결책이지만 inrush·비용·수명·필터 corner도 바뀌므로 damping·제어 대안과 비교한다(실험 2).",
        )
    )
    if lam[0].real > 0:
        res.verdict("UNSTABLE", f"평형점(V_e = {Ve:.6g} V)은 존재하지만 극점 {lam[0].real:.5g} ± j{abs(lam[0].imag):.6g} s⁻¹로 불안정: C = {C * 1e6:.4g} µF < C_crit = {Ccrit * 1e6:.6g} µF.")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"극점 {lam[0].real:.5g} ± j{abs(lam[0].imag):.6g} s⁻¹로 국소 안정: C ≥ C_crit (이상 CPL·국소 선형 결과).")
    res.assumptions += ["이상 CPL i = P/v (제어 대역 무한), 선형 R·L·C, 소스 V_s 일정", f"교란 {dv0:g} V (교재 그림과 같은 크기), 비선형 적분은 |Δv| ≤ {v['dv_valid'] * 100:g} % V_e에서만"]
    res.not_valid_for += ["실제 컨버터의 유한 대역 입력 임피던스 (실험 2)", "큰 진폭의 전압 붕괴·UVLO·전류 제한", "스위칭 리플"]
    res.interpretation = (
        f"CPL은 전압이 내려가면 전류를 더 끌어가는 음의 증분저항(−V_e²/P = {-1 / g:.5g} Ω)이다. 그래서 LC 공진을 감쇠시키는 R의 역할을 CPL이 되돌려 놓고, "
        f"trace(A) = −R/L + g/C가 양이 되면(C < L·g/R = {Ccrit * 1e6:.6g} µF) 진동이 자란다. 이는 평형점의 존재(P ≤ V_s²/(4R))와는 다른 조건이며, 무한대역 이상 CPL을 가정한 국소 결과다."
    )
    return res


def cpl_circuit(v: dict) -> Circuit:
    c = Circuit("cpl", 640, 280, title="소스 R–L + 노드 C + 정전력 부하(CPL)")
    vs = c.add("vsource", "VS", 80, 150, 90, "V_s", f"{v['Vs']:g} V")
    rr = c.add("resistor", "R", 190, 60, 0, "R", f"{v['R']:g} Ω")
    ll = c.add("inductor", "L", 310, 60, 0, "L", f"{v['L'] * 1e3:g} mH")
    cc = c.add("capacitor", "C", 420, 150, 90, "C", f"{v['C'] * 1e6:.4g} µF")
    ip = c.add("isource", "CPL", 540, 150, 90, "CPL i = P/v", f"{v['P'] / 1e3:g} kW", lpos=(562, 146, "start"))
    c.wire("w1", vs["a"], (80, 60), rr["a"])
    c.wire("w2", rr["b"], ll["a"])
    c.wire("w3", ll["b"], (420, 60), cc["a"])
    c.wire("w4", (420, 60), (540, 60), ip["a"])
    c.wire("w5", cc["b"], (420, 240), (80, 240), vs["b"])
    c.wire("w6", ip["b"], (540, 240), (420, 240))
    c.dot((420, 60), (420, 240))
    c.text(420, 44, "v (node)", "node")
    c.probe("pL", "di_nl", 250, 48, "right", "i_L")
    c.text(300, 272, "증분저항 −V²/P < 0: LC 공진의 감쇠를 되돌린다", "note")
    return c


# ======================================================================================
# Experiment 2: finite-bandwidth converter vs ideal CPL (impedance, Nyquist, corners)
# ======================================================================================


class ConverterSystem:
    """Source R-L + filter C (+ optional R_d-C_d damping leg) + averaged buck with cascaded PI control.

    Converter: L_o di_Lo/dt = d v - v_o, C_o dv_o/dt = i_Lo - v_o/R_o, voltage PI -> i*, current PI with
    output-voltage feedforward -> switch-node command u [V], modulator d = u / V_ff with V_ff = v
    (input-voltage feedforward) or V_nom (fixed).  Gains: Kp_i = L_o w_ci, Ki_i = Kp_i w_ci/5;
    Kp_v = C_o w_v, Ki_v = Kp_v w_v/5.
    """

    def __init__(self, v: dict, Vs=None, P=None):
        self.Vs = v["Vs"] if Vs is None else Vs
        self.P = v["P"] if P is None else P
        self.R, self.L, self.C = v["R"], v["L"], v["C"]
        self.Rd, self.Cd = v["Rd"], v["Cd"]
        self.Vo, self.Lo, self.Co = v["Vo"], v["Lo"], v["Co"]
        self.Ro = self.Vo**2 / self.P
        self.ff = bool(v["ff"])
        self.Vnom = v["Vnom"]
        wci, wv = 2 * math.pi * v["fci"], 2 * math.pi * v["fbw"]
        self.Kpi, self.Kpv = self.Lo * wci, self.Co * wv
        self.Kii, self.Kiv = self.Kpi * wci / 5, self.Kpv * wv / 5
        self.damp = self.Cd > 0

    # ---- nonlinear model -------------------------------------------------------------
    def control(self, iLo, vo, xi, xv, v):
        istar = self.Kpv * (self.Vo - vo) + xv
        u = vo + self.Kpi * (istar - iLo) + xi
        d = u / (v if self.ff else self.Vnom)
        return d, istar, u

    def rhs(self, t, x):
        i, v, iLo, vo, xi, xv = x[:6]
        d, istar, _ = self.control(iLo, vo, xi, xv, v)
        idmp = (v - x[6]) / self.Rd if self.damp else 0.0
        out = [(self.Vs - self.R * i - v) / self.L, (i - d * iLo - idmp) / self.C, (d * v - vo) / self.Lo, (iLo - vo / self.Ro) / self.Co, self.Kii * (istar - iLo), self.Kiv * (self.Vo - vo)]
        if self.damp:
            out.append(idmp / self.Cd)
        return out

    def equilibrium(self):
        disc = self.Vs**2 - 4 * self.R * self.P
        if disc < 0:
            return None
        ve = (self.Vs + math.sqrt(disc)) / 2
        Io = self.Vo / self.Ro
        u = self.Vo * (ve if self.ff else self.Vnom) / ve
        x = [self.P / ve, ve, Io, self.Vo, u - self.Vo, Io]
        if self.damp:
            x.append(ve)
        return np.array(x)

    def jacobian(self, x):
        n = x.size
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (np.array(self.rhs(0, xp)) - np.array(self.rhs(0, xm))) / (2 * h)
        return J

    # ---- converter-only small-signal model (hand-derived partials) ----------------------
    def converter_ss(self, x):
        """(A_c, B_c, C_c, D_c) of the converter alone with the input voltage v as input, output i_in."""
        _, v, iLo, vo, xi, xv = x[:6]
        d, _, u = self.control(iLo, vo, xi, xv, v)
        du = np.array([-self.Kpi, 1 - self.Kpi * self.Kpv, 1.0, self.Kpi])  # du/d[iLo, vo, xi, xv]
        Vff = v if self.ff else self.Vnom
        dd = du / Vff
        dd_dv = -u / v**2 if self.ff else 0.0
        A = np.zeros((4, 4))
        A[0, :] = (v * dd) / self.Lo
        A[0, 1] -= 1 / self.Lo
        A[1, 0] = 1 / self.Co
        A[1, 1] = -1 / (self.Ro * self.Co)
        A[2, :] = [-self.Kii, -self.Kii * self.Kpv, 0.0, self.Kii]
        A[3, 1] = -self.Kiv
        B = np.zeros(4)
        B[0] = (d + v * dd_dv) / self.Lo
        Cc = iLo * dd + np.array([d, 0, 0, 0])
        D = iLo * dd_dv
        return A, B, Cc, D

    def Yin(self, s, x):
        """Y_in(s) = C_c (sI - A_c)^-1 B_c + D_c, vectorised by the modal (partial-fraction) form."""
        A, B, Cc, D = self.converter_ss(x)
        s = np.atleast_1d(s)
        lam, V = np.linalg.eig(A)
        r = (Cc @ V) * np.linalg.solve(V, B)  # residues
        return D + np.sum(r[None, :] / (s[:, None] - lam[None, :]), axis=1)

    def Yin_direct(self, s, x):
        A, B, Cc, D = self.converter_ss(x)
        return np.array([Cc @ np.linalg.solve(sk * np.eye(4) - A, B) + D for sk in np.atleast_1d(s)])

    def Zs(self, s):
        s = np.atleast_1d(s)
        Y = 1 / (self.R + s * self.L) + s * self.C
        if self.damp:
            Y = Y + 1 / (self.Rd + 1 / (s * self.Cd))
        return 1 / Y

    def loop_margins(self, x):
        """Converter's own loops with a stiff source (v = V_e): inner current loop and outer voltage loop."""
        v = x[1]
        k = v / (v if self.ff else self.Vnom)

        def Li(f):
            s = 2j * np.pi * np.asarray(f)
            return k * (self.Kpi + self.Kii / s) / (s * self.Lo)

        def Lv(f):
            s = 2j * np.pi * np.asarray(f)
            li = Li(f)
            Ti = li / (1 + li)
            Zo = self.Ro / (1 + s * self.Ro * self.Co)
            return (self.Kpv + self.Kiv / s) * Ti * Zo

        return margins(Li, 1.0, 1e6), margins(Lv, 0.1, 1e5), Li, Lv


def nyquist_count(Tm_fun, w_lo=1e-2, w_hi=1e8, n=60000):
    """Number of closed-loop RHP poles from the phase change of 1 + T_m(jw), w: 0 -> inf (Z = -Δarg/π)."""
    w = np.geomspace(w_lo, w_hi, n)
    F = 1 + Tm_fun(1j * w)
    ph = np.unwrap(np.angle(F))
    return int(round(-(ph[-1] - ph[0]) / math.pi)), float(np.min(np.abs(F))), w, F


def run_converter_impedance(v: dict) -> Result:
    res = Result("EX07", "converter_impedance", "B (평균 컨버터 소신호 + 비선형 시간영역)")
    sysm = ConverterSystem(v)
    xe = sysm.equilibrium()
    if xe is None:
        res.verdict("NO_SOLUTION", "P > V_s²/(4R): 평형점이 없다 (실험 1).")
        return res
    ve = xe[1]
    Rn = ve * ve / sysm.P
    J = sysm.jacobian(xe)
    lam = np.linalg.eigvals(J)
    n_unst = int(np.sum(lam.real > 0))
    dom = lam[np.argmax(lam.real)]
    # converter alone with a stiff source: own stability and loop margins
    Ac, Bc, Cc, Dc = sysm.converter_ss(xe)
    lam_c = np.linalg.eigvals(Ac)
    mi, mv, Li, Lv = sysm.loop_margins(xe)
    res.add_metric("Ve", "평형 전압 V_e (실험 1과 같은 이차식)", ve, "V", ref=394.935887 if abs(sysm.Vs - 400) < 1e-9 and abs(sysm.P - 1e4) < 1e-6 and abs(sysm.R - 0.2) < 1e-12 else ref.equilibria(sysm.Vs, sysm.R, sysm.P)[0], ref_label="394.935887 V", tol=1e-8)
    res.add_metric("Rn", "이상 CPL 증분저항 크기 V_e²/P", Rn, "Ω")
    res.add_metric("fci_m", "컨버터 전류 루프 crossover (안정한 전원)", mi.fc, "Hz")
    res.add_metric("pm_i", "컨버터 전류 루프 위상여유 (안정한 전원)", mi.pm, "deg", basis="단일 루프 판정: 입력필터 없이")
    res.add_metric("fbw_m", "컨버터 전압 루프 crossover", mv.fc, "Hz")
    res.add_metric("pm_v", "컨버터 전압 루프 위상여유", mv.pm, "deg")
    res.add_metric("conv_stable", "컨버터 단독 (이상 전원) 극점 최대 Re", float(np.max(lam_c.real)), "1/s", basis="< 0: 부하 쪽 subsystem 안정 (Nyquist 전제)")
    res.add_metric("sys_re", "전체 시스템 지배 극점 실수부", float(dom.real), "1/s", basis="필터 + 컨버터 선형화 고유값")
    res.add_metric("sys_im", "전체 시스템 지배 극점 허수부", float(abs(dom.imag)), "rad/s")
    lam_cpl = np.linalg.eigvals(np.array([[-sysm.R / sysm.L, -1 / sysm.L], [1 / sysm.C, (sysm.P / ve**2) / sysm.C]])) if not sysm.damp else None
    if lam_cpl is not None:
        res.add_metric("cpl_re", "같은 필터 + 이상 CPL 극점 실수부 (실험 1)", float(np.max(lam_cpl.real)), "1/s")
    # hand-derived small-signal model vs numerical Jacobian (independent)
    Jc = J[2:6, 2:6]
    Bn = J[2:6, 1]
    res.add_check(check_close("컨버터 소신호 A_c: 손 유도 vs 수치 Jacobian", float(np.max(np.abs(Ac - Jc))), 0.0, 1e-6, "편미분을 손으로 쓴 행렬 vs 비선형 우변의 중앙차분", True, "", abs_scale=float(np.max(np.abs(Jc)))))
    res.add_check(check_close("컨버터 입력 B_c (∂/∂v): 손 유도 vs 수치", float(np.max(np.abs(Bc - Bn))), 0.0, 1e-6, "같은 두 경로", True, "", abs_scale=max(float(np.max(np.abs(Bn))), 1.0)))
    # Nyquist count of 1 + Z_s Y_in vs eigenvalue count (independent)
    Tm = lambda s: sysm.Zs(s) * sysm.Yin(s, xe)  # noqa: E731
    Z, dmin, wN, FN = nyquist_count(Tm)
    res.add_metric("nyq", "Nyquist: 1 + Z_s/Z_in의 우반평면 영점 수", Z, "", basis="Δarg(1 + T_m), ω: 0→∞, Z = −Δarg/π")
    res.add_metric("n_unst", "고유값: 우반평면 극점 수", n_unst, "")
    res.add_check(Check("Nyquist 영점 수 = 우반평면 고유값 수", "PASS" if Z == n_unst else "FAIL", Z, "개", n_unst, path="소신호 임피던스(Z_s 해석식, Y_in 상태공간)의 위상 회전 vs 전체 시스템 Jacobian 고유값", independent=True,
                        detail=f"min |1 + T_m| = {dmin:.3g} (−1까지의 최소 거리); 컨버터 단독 안정이 Nyquist의 전제"))
    s_chk = 2j * np.pi * np.geomspace(1.0, 1e5, 25)
    y_err = float(np.max(np.abs(sysm.Yin(s_chk, xe) - sysm.Yin_direct(s_chk, xe)) / np.abs(sysm.Yin_direct(s_chk, xe))))
    res.add_check(Check("Y_in: 모드 분해(부분분수) vs 주파수별 직접 선형해", "PASS" if y_err < 1e-9 else "FAIL", y_err, "rel", 1e-9, path="고유분해 잔차 합 vs (sI − A_c)⁻¹B_c 직접 풀이 (25 주파수)", independent=False, detail="수치 구현의 회귀 확인"))
    # Middlebrook magnitude criterion (sufficient, conservative)
    f = np.geomspace(1.0, 1e5, 1200)
    s = 2j * np.pi * f
    Zs = sysm.Zs(s)
    Yi = sysm.Yin(s, xe)
    Zi = 1 / Yi
    ratio = np.abs(Zs * Yi)
    mb = float(np.max(ratio))
    res.add_metric("mb", "max |Z_s/Z_in| (Middlebrook 크기 비)", mb, "", basis="< 1이면 충분조건으로 안정 (필요조건 아님)")
    # time-domain confirmation (nonlinear averaged)
    x0 = xe.copy()
    x0[1] += v["dv0"]
    lim = 0.2 * ve
    ev = lambda t, x: abs(x[1] - ve) - lim  # noqa: E731
    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(sysm.rhs, (0, v["t_end"]), x0, method="DOP853", rtol=1e-9, atol=1e-9, events=ev, dense_output=True, max_step=2e-5)
    t_last = float(sol.t[-1])
    tt = np.linspace(0, t_last, 1500)
    dvt = sol.sol(tt)[1] - ve
    # envelope growth over the linear range: peaks of |dv|
    ab = np.abs(dvt)
    pk_i = [k for k in range(1, tt.size - 1) if ab[k] >= ab[k - 1] and ab[k] > ab[k + 1] and ab[k] < 0.05 * ve and ab[k] > 1e-6]
    if len(pk_i) >= 4 and abs(dom.imag) > 0:
        tp, ap = tt[pk_i], ab[pk_i]
        sig = float(np.polyfit(tp, np.log(ap), 1)[0])
        res.add_metric("sig_nl", "비선형 시뮬레이션의 포락선 성장률", sig, "1/s", ref=float(dom.real), ref_label="지배 고유값 실수부", tol=0.1, abs_scale=0.05 * abs(dom), basis="|Δv| 봉우리 로그 기울기 (선형 범위)")
        res.add_check(check_close("시간영역: 비선형 평균모델 포락선 vs 지배 고유값", sig, float(dom.real), 0.1, "DOP853 비선형 적분 vs 선형화 고유값 (다른 극점의 과도 성분 때문에 허용오차 10 %)", True, "1/s", abs_scale=0.05 * abs(dom)))
    res.add_series("td_v", "Δv 비선형 (컨버터 모델)", "V", tt.tolist(), dvt.tolist())
    fcpl = cpl_rhs(sysm.Vs, sysm.R, sysm.L, sysm.C, sysm.P)
    if not sysm.damp:
        evc = lambda t, x: abs(x[1] - ve) - lim  # noqa: E731
        evc.terminal = True
        evc.direction = 1
        sc = solve_ivp(fcpl, (0, v["t_end"]), [sysm.P / ve, ve + v["dv0"]], method="DOP853", rtol=1e-10, atol=1e-9, events=evc, dense_output=True)
        t2 = np.linspace(0, float(sc.t[-1]), 1200)
        res.add_series("td_cpl", "Δv 이상 CPL (같은 필터)", "V", t2.tolist(), (sc.sol(t2)[1] - ve).tolist(), dash=True)
    res.add_plot("p_td", f"시간영역 확인: 평형점 + {v['dv0']:g} V 교란", ["td_v"] + (["td_cpl"] if not sysm.damp else []), y_label="v − V_e", y_unit="V", level="B (비선형)", group="td",
                 hlines=[{"y": lim, "label": "±20 % 유효범위"}, {"y": -lim, "label": ""}],
                 proved="선형화 고유값의 판정을 비선형 평균모델 적분이 확인한다. 유한 대역 컨버터는 이상 CPL과 다른 성장·감쇠를 보일 수 있다.",
                 not_yet="스위칭 리플·duty 포화·보호 동작은 없다(평균모델 B). 큰 진폭은 유효범위에서 중단했다.")
    # impedance plots + corners
    fz = np.geomspace(1.0, 1e5, 500)
    sz = 2j * np.pi * fz
    res.add_series("Zs", "|Z_s| 입력필터 출력 임피던스", "Ω", fz.tolist(), np.abs(sysm.Zs(sz)).tolist())
    corners = [(360.0, sysm.P), (400.0, sysm.P), (440.0, sysm.P), (400.0, 0.5 * sysm.P), (400.0, 1.5 * sysm.P)]
    rows = []
    zkeys, pkeys = ["Zs"], []
    for k, (Vc, Pc) in enumerate(corners):
        sc_ = ConverterSystem(v, Vs=Vc, P=Pc)
        xc = sc_.equilibrium()
        if xc is None:
            rows.append([f"{Vc:g} V / {Pc / 1e3:g} kW", "—", "—", "—", "—", "—", "NO_SOLUTION"])
            continue
        Yc = sc_.Yin(sz, xc)
        res.add_series(f"Zin{k}", f"|Z_in| {Vc:g} V / {Pc / 1e3:g} kW", "Ω", fz.tolist(), np.abs(1 / Yc).tolist())
        res.add_series(f"Pin{k}", f"∠Z_in {Vc:g} V / {Pc / 1e3:g} kW", "deg", fz.tolist(), np.degrees(np.unwrap(np.angle(1 / Yc))).tolist())
        zkeys.append(f"Zin{k}")
        pkeys.append(f"Pin{k}")
        lamk = np.linalg.eigvals(sc_.jacobian(xc))
        Zk, dk, _, _ = nyquist_count(lambda s_, sc_=sc_, xc=xc: sc_.Zs(s_) * sc_.Yin(s_, xc))
        mbk = float(np.max(np.abs(sc_.Zs(sz) * Yc)))
        rows.append([f"{Vc:g} V / {Pc / 1e3:g} kW", xc[1], xc[1] ** 2 / Pc, float(np.max(lamk.real)), f"{Zk} / {int(np.sum(lamk.real > 0))}", mbk, "UNSTABLE" if np.max(lamk.real) > 0 else "안정"])
    res.add_series("Zcpl", f"|Z| 이상 CPL V_e²/P ({ve:.4g} V, {sysm.P / 1e3:g} kW)", "Ω", [fz[0], fz[-1]], [Rn, Rn], dash=True)
    res.add_plot("p_z", "입력필터 |Z_s|와 컨버터 |Z_in| (전압·부하 코너)", zkeys + ["Zcpl"], x_label="f", x_unit="Hz", y_label="|Z|", y_unit="Ω", kind="xy", log_x=True, log_y=True, level="B", group="zin",
                 vlines=[{"x": 1 / (2 * math.pi * math.sqrt(sysm.L * sysm.C)), "label": "필터 공진"}, {"x": v["fci"], "label": "f_ci"}],
                 proved="컨버터 입력 임피던스는 제어가 전력을 붙잡는 대역 안에서만 V_e²/P 크기의 음저항이고, 대역 밖에서는 크기·위상이 바뀐다. 저전압·고부하 코너일수록 |Z_in|이 작아 필터와의 여유가 줄어든다.",
                 not_yet="평균모델 B의 소신호 결과다. 스위칭 리플·샘플링 지연(실험 3)은 Z_in에 넣지 않았다.")
    res.add_plot("p_zph", "컨버터 ∠Z_in: 저주파 180°(음저항) → 대역 밖에서 변한다", pkeys, x_label="f", x_unit="Hz", y_label="∠Z_in", y_unit="deg", kind="xy", log_x=True, level="B", group="zin",
                 hlines=[{"y": -180.0, "label": "±180° (음저항)"}, {"y": 0.0, "label": "0° (저항)"}, {"y": 90.0, "label": "+90° (유도성)"}],
                 proved="위상이 ±180° 근처인 대역이 CPL처럼 행동하는 대역이고, 대역 밖에서는 위상이 저항·유도성 쪽으로 돌아간다.", not_yet="위상 표시는 저주파 쪽에서 unwrap한 가지(−180°에서 시작)다.")
    Tn = FN - 1
    sel = (wN > 2 * math.pi * 5) & (wN < 2 * math.pi * 5e4)
    res.add_series("nyq", "T_m = Z_s/Z_in (ω > 0)", "", Tn.real[sel].tolist(), Tn.imag[sel].tolist())
    res.add_plot("p_nyq", "minor-loop gain Nyquist (−1 점과의 관계)", ["nyq"], x_label="Re T_m", x_unit="", y_label="Im T_m", y_unit="", kind="xy", level="B",
                 markers=[{"x": -1.0, "y": 0.0, "label": "−1"}],
                 proved=f"1 + T_m의 위상 회전으로 센 우반평면 영점 {Z}개가 전체 시스템 고유값의 불안정 극점 {n_unst}개와 일치한다.",
                 not_yet="Nyquist 판정은 소스·부하 subsystem이 각각 안정하다는 전제가 필요하다. |Z_s| < |Z_in|은 보수적인 충분조건일 뿐 필요충분이 아니다.")
    res.tables.append(Table("t_corner", "전압·부하 코너별 판정 (같은 컨버터·필터)", ["코너 V_s / P", "V_e [V]", "V_e²/P [Ω]", "max Re λ [1/s]", "Nyquist / 고유값 불안정 수", "max |Z_s/Z_in|", "전체 시스템"], rows,
                            note="코너마다 평형점·이득(변조기 v/V_ff)·음저항 크기가 바뀐다. 한 운전점의 안정성으로 다른 코너를 결론내지 않는다."))
    res.tables.append(
        Table(
            "t_sep",
            "단일 루프 판정 vs 전체 시스템 판정 (따로 보고)",
            ["대상", "조건", "결과"],
            [
                ["전류 루프 (컨버터 단독, 이상 전원)", f"crossover {mi.fc:.4g} Hz", f"위상여유 {mi.pm:.3g}°"],
                ["전압 루프 (컨버터 단독, 이상 전원)", f"crossover {mv.fc:.4g} Hz", f"위상여유 {mv.pm:.3g}°"],
                ["컨버터 단독 극점", "이상 전압원 입력", "안정" if np.max(lam_c.real) < 0 else "불안정"],
                ["전체 시스템 (필터 + 컨버터)", f"지배 극점 {dom.real:.4g} ± j{abs(dom.imag):.4g}", "UNSTABLE" if n_unst else "안정"],
                ["Middlebrook 크기 조건", f"max |Z_s/Z_in| = {mb:.3g}", "충족 (충분)" if mb < 1 else "미충족 (판정 불가, 위상 필요)"],
            ],
            note="단일 루프의 위상여유가 충분해도 입력필터와 연결된 전체 시스템은 불안정할 수 있다.",
        )
    )
    res.circuit = {"diagram": converter_circuit(v).to_json(), "intervals": [], "plot_group": "td"}
    if np.max(lam_c.real) >= 0:
        res.verdict("UNSTABLE", "컨버터 단독(이상 전원)도 불안정하다: 루프 설계부터 다시 본다.")
    elif n_unst:
        res.verdict("UNSTABLE", f"컨버터 루프는 이상 전원에서 안정(전류 PM {mi.pm:.3g}°, 전압 PM {mv.pm:.3g}°)하지만 입력필터와 연결한 전체 시스템은 불안정: 지배 극점 {dom.real:.4g} ± j{abs(dom.imag):.5g} s⁻¹.")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"전체 시스템 안정 (지배 극점 실수부 {dom.real:.4g} s⁻¹)" + ("; Middlebrook 크기 조건은 충족하지 않았지만(보수적 조건) 위상 때문에 −1을 감싸지 않는다." if mb >= 1 else "."))
    res.assumptions += [
        "평균 buck(B), 무손실, 이상 변조기 d = u/V_ff, 스위칭 리플 없음",
        f"제어: 전류 PI (f_ci {v['fci']:g} Hz, 영점 f_ci/5) + 출력전압 피드포워드, 전압 PI (f_bw {v['fbw']:g} Hz, 영점 f_bw/5)",
        "입력전압 피드포워드 " + ("있음 (d = u/v)" if sysm.ff else f"없음 (d = u/{sysm.Vnom:g} V)"),
        "필터: 소스 R–L + C" + (f" + 감쇠가지 R_d {sysm.Rd:g} Ω – C_d {sysm.Cd * 1e6:g} µF" if sysm.damp else ""),
    ]
    res.not_valid_for += ["샘플링 지연·PWM의 영향 (실험 3)", "duty·전류 포화 이후의 큰 신호 동작", "스위칭 주파수 부근의 임피던스"]
    res.interpretation = (
        f"이 컨버터는 전류 루프가 입력전압 변동을 보상하는 대역(약 f_ci = {v['fci']:g} Hz 아래)에서 전력을 일정하게 붙잡아 CPL처럼 행동하고, 입력필터 공진({1 / (2 * math.pi * math.sqrt(sysm.L * sysm.C)):.4g} Hz)이 그 대역 안에 있으면 "
        "필터의 감쇠를 음저항이 상쇄한다. 입력전압 피드포워드(d = u/v)를 넣으면 모든 주파수에서 이상 CPL이 되어 실험 1의 극점과 같아진다. 반대로 대역이 공진보다 낮으면 공진 주파수에서 컨버터가 양의 임피던스처럼 보여 "
        "C < C_crit이어도 안정할 수 있다 — 이상 CPL 판정은 대역 밖에서 보수적이다. 그래서 단일 루프 여유와 전체 시스템 안정성을 따로 보고한다."
    )
    return res


def converter_circuit(v: dict) -> Circuit:
    c = Circuit("cpl_conv", 850, 330, title="입력필터 + 평균 buck 컨버터 (전류·전압 루프) — impedance port")
    vs = c.add("vsource", "VS", 60, 150, 90, "V_s", f"{v['Vs']:g} V")
    rr = c.add("resistor", "R", 150, 60, 0, "R", f"{v['R']:g} Ω")
    ll = c.add("inductor", "L", 250, 60, 0, "L", f"{v['L'] * 1e3:g} mH")
    cc = c.add("capacitor", "C", 330, 150, 90, "C", f"{v['C'] * 1e6:.4g} µF")
    q = c.add("nmos", "Q", 450, 60, 0, "Q (buck)", lpos=(450, 34, "middle"))
    dd = c.add("diode", "D", 520, 150, 270, "D", lpos=(506, 154, "end"))
    lo = c.add("inductor", "LO", 600, 60, 0, "L_o", f"{v['Lo'] * 1e6:g} µH")
    co = c.add("capacitor", "CO", 680, 150, 90, "C_o", f"{v['Co'] * 1e6:g} µF")
    ro = c.add("resistor", "RO", 760, 150, 90, "R_o", "V_o²/P")
    c.wire("w1", vs["a"], (60, 60), rr["a"])
    c.wire("w2", rr["b"], ll["a"])
    c.wire("w3", ll["b"], (330, 60), cc["a"])
    c.wire("w4", (330, 60), q["a"])
    c.wire("w5", q["b"], (520, 60), dd["b"])
    c.wire("w6", (520, 60), lo["a"])
    c.wire("w7", lo["b"], (680, 60), co["a"])
    c.wire("w8", (680, 60), (760, 60), ro["a"])
    c.wire("w9", ro["b"], (760, 240), (60, 240), vs["b"])
    c.wire("w10", cc["b"], (330, 240))
    c.wire("w11", dd["a"], (520, 240))
    c.wire("w12", co["b"], (680, 240))
    c.dot((330, 60), (330, 240), (520, 60), (520, 240), (680, 60), (680, 240))
    c.text(365, 100, "Z_s | Z_in", "node")
    c.add("block", "CTL", 600, 290, 0, "전압 PI(느림) → i* → 전류 PI(빠름) → u → d = u/V_ff", w=380, h=30)
    c.text(200, 290, "impedance port: 필터 쪽 Z_s, 컨버터 쪽 Z_in", "note")
    c.probe("pL", "none", 200, 48, "right", "i")
    return c


# ======================================================================================
# Experiment 3: digital delay, timing diagram and the current-loop margin
# ======================================================================================


def timing(v: dict) -> dict:
    """Control period, update instant and effective delay from the sample/compute/update sequence."""
    Tc = 1.0 / (v["f_pwm"] * (2 if v["update"] == "double" else 1))
    tcalc = v["t_calc"]
    if v["policy"] == "immediate":
        tau = tcalc
        note = "연산 직후 즉시 갱신 (shadow register 없이 — glitch 위험은 모델 밖)"
    else:
        m = max(1, math.ceil(tcalc / Tc - 1e-12))
        tau = m * Tc
        note = f"연산 완료 후 첫 갱신 시점 = 샘플 뒤 {m}번째 갱신 사건"
    return {"Tc": Tc, "tau": tau, "Td": tau + Tc / 2, "note": note}


def zoh_phase_sim(f: float, Tc: float, tau: float) -> float:
    """Phase lag [deg] of the fundamental of the sampled, delayed, held command (exact piecewise integrals)."""
    n_per = Tc * f  # command periods per sample
    # smallest number of samples N such that N*Tc spans a whole number of command periods
    m = 1
    while (abs(m * n_per - round(m * n_per)) > 1e-9 or round(m * n_per) == 0) and m < 100000:
        m += 1
    N = m  # samples
    W = N * Tc
    w = 2 * math.pi * f
    k = np.arange(N)
    tk = k * Tc
    a = tk + tau
    b = a + Tc
    val = np.sin(w * tk)
    c_app = np.sum(val * (np.exp(-1j * w * a) - np.exp(-1j * w * b)) / (1j * w))
    c_cmd = -0.5j * W  # integral of sin(wt) e^{-jwt} over integer periods
    return float(np.degrees(np.angle(c_cmd) - np.angle(c_app)) % 360.0)


def disc_loop_gain_frac(f, Kp, Ki, L, R, Tc, tau):
    """Exact sampled loop gain with a fractional update delay tau (input change inside the period)."""
    n = int(math.floor(tau / Tc + 1e-12))
    tf = tau - n * Tc
    a = math.exp(-R * Tc / L)
    b_old = (math.exp(-R * (Tc - tf) / L) - a) / R
    b_new = (1 - math.exp(-R * (Tc - tf) / L)) / R
    z = np.exp(2j * np.pi * np.asarray(f) * Tc)
    C = Kp + Ki * Tc / (z - 1)
    G = (b_new * z ** (-n) + b_old * z ** (-n - 1)) / (z - a)
    return C * G


def frac_loop_rho(Kp, Ki, L, R, Tc, tau) -> float:
    """Spectral radius of the sampled loop with the fractional update delay (characteristic polynomial)."""
    n = int(math.floor(tau / Tc + 1e-12))
    tf = tau - n * Tc
    a = math.exp(-R * Tc / L)
    b_old = (math.exp(-R * (Tc - tf) / L) - a) / R
    b_new = (1 - math.exp(-R * (Tc - tf) / L)) / R
    lhs = np.polymul(np.polymul([1.0, -1.0], [1.0, -a]), [1.0] + [0.0] * (n + 1))
    rhs = np.polymul([Kp, -Kp + Ki * Tc], [b_new, b_old])
    return float(np.max(np.abs(np.roots(np.polyadd(lhs, rhs)))))


class TimedRL(HybridSystem):
    """L di/dt = u(t) - R i with u changing at arbitrary instants (the actual PWM update times)."""

    state_names = ("i",)

    def __init__(self, L, R, times, levels):
        self.L, self.R = L, R
        self.times = list(times)
        self.levels = list(levels)

    def mode(self, q):
        u = self.levels[q]
        return AffineMode(f"trl|{self.L!r}|{self.R!r}|{u!r}", [[-self.R / self.L]], [u / self.L])

    def gate_schedule(self, t0, t1):
        return [(t, lambda q, k=k: k) for k, t in enumerate(self.times) if k > 0 and t0 < t < t1]

    def outputs(self, q):
        return {"i": np.array([1.0, 0.0])}

    def stored_energy(self):
        return np.diag([self.L, 0.0])

    def powers(self, q):
        e = np.array([1.0, 0.0])
        return {"p_in": sym_linear(e, self.levels[q]), "p_R": self.R * np.outer(e, e)}


def run_digital_delay(v: dict) -> Result:
    res = Result("EX07", "digital_delay", "B (샘플링 순서의 정확 이산 모델)")
    tm = timing(v)
    Tc, tau, Td = tm["Tc"], tm["tau"], tm["Td"]
    f1, f2 = v["fc1"], v["fc2"]
    is15 = abs(Td - 15e-6) < 1e-12
    lag1, lag2 = 360 * f1 * Td, 360 * f2 * Td
    res.add_metric("Tc", "제어(샘플) 주기 T_c", Tc, "s", basis=("double update: 캐리어 반주기" if v["update"] == "double" else "single update: 캐리어 주기"))
    res.add_metric("tau", "샘플 → 실제 PWM 갱신", tau, "s", basis=tm["note"])
    res.add_metric("Td", "유효 지연 T_d = (갱신 − 샘플) + T_c/2", Td, "s", ref=15e-6 if is15 else None, ref_label="교재 예 15 µs", tol=1e-9, basis="ZOH 반주기 포함, 1.5·T_s로 고정하지 않음")
    res.add_metric("lag1", f"{f1 / 1e3:g} kHz에서 지연 위상 360·f·T_d", lag1, "deg", ref=5.4 if (is15 and abs(f1 - 1e3) < 1e-9) else cref.delay_phase_deg(f1, Td), ref_label="교재 5.4°" if (is15 and abs(f1 - 1e3) < 1e-9) else "360·f·T_d", tol=1e-9)
    res.add_metric("lag2", f"{f2 / 1e3:g} kHz에서 지연 위상", lag2, "deg", ref=54.0 if (is15 and abs(f2 - 1e4) < 1e-9) else cref.delay_phase_deg(f2, Td), ref_label="교재 54°" if (is15 and abs(f2 - 1e4) < 1e-9) else "360·f·T_d", tol=1e-9)
    # independent: simulate the sample -> delay -> hold chain on a sine and read the fundamental phase
    for f_, lag_, key in ((f1, lag1, "1"), (f2, lag2, "2")):
        if f_ < 0.5 / Tc:
            ph = zoh_phase_sim(f_, Tc, tau)
            res.add_check(check_close(f"지연 위상 @ {f_ / 1e3:g} kHz: 샘플·지연·hold 시뮬레이션 vs 360·f·T_d", ph, lag_, 1e-9, "사인 명령을 T_c마다 샘플해 τ 뒤에 T_c 동안 유지한 계단 파형의 기본파 위상 (정확 구간 적분)", True, "deg"))
    # current-loop margins for the two crossovers with this exact timing
    L, R = v["L"], v["R"]
    rows = []
    pm_at = {}
    for fc in (f1, f2):
        wc = 2 * math.pi * fc
        Kp, Ki = L * wc, R * wc
        mz = margins(lambda f, Kp=Kp, Ki=Ki: disc_loop_gain_frac(f, Kp, Ki, L, R, Tc, tau), min(fc, 0.5 / Tc) / 1000, 0.4999 / Tc)
        mcd = margins(lambda f, Kp=Kp, Ki=Ki: loop_gain_cont(f, Kp, Ki, L, R, Td), fc / 1000, 100 * fc)
        pm_at[fc] = (mz, mcd)
        rows.append([f"{fc / 1e3:g} kHz", f"{360 * fc * Td:.4g}°", f"{mcd.pm:.4g}°" if mcd.pm is not None else "—", f"{mz.pm:.4g}°" if mz.pm is not None else "불안정/교차 없음", f"{mz.gm_db:.3g} dB" if mz.gm_db is not None else "—"])
    mz1, _ = pm_at[f1]
    mz2, _ = pm_at[f2]
    res.add_metric("pm1", f"전류 루프 위상여유 (f_c = {f1 / 1e3:g} kHz, 이 타이밍의 정확 이산 모델)", mz1.pm if mz1.pm is not None else "교차 없음", "deg" if mz1.pm is not None else "")
    res.add_metric("pm2", f"전류 루프 위상여유 (f_c = {f2 / 1e3:g} kHz)", mz2.pm if mz2.pm is not None else "교차 없음", "deg" if mz2.pm is not None else "", note="연속 근사 90° − 360 f_c T_d와 비교")
    if mz2.gm_db is not None:
        res.add_metric("gm2", f"이득여유 (f_c = {f2 / 1e3:g} kHz)", mz2.gm_db, "dB")
    res.tables.append(Table("t_pm", "crossover별 전류 루프 여유 (극점 소거 PI, 이 타이밍)", ["f_c", "지연 위상", "PM 연속 근사", "PM 정확 이산", "GM 정확 이산"], rows,
                            note="연속 근사는 exp(−sT_d)만 넣은 것이고, 정확 이산 모델은 샘플·분수 지연 갱신·ZOH·이산 적분기를 모두 포함한다."))
    # engine check of the fractional-delay plant map (closed-loop step at f_c1)
    wc = 2 * math.pi * f1
    Kp, Ki = L * wc, R * wc
    n = 200
    nfull = int(math.floor(tau / Tc + 1e-12))
    tf = tau - nfull * Tc
    a = math.exp(-R * Tc / L)
    b_old = (math.exp(-R * (Tc - tf) / L) - a) / R
    b_new = (1 - math.exp(-R * (Tc - tf) / L)) / R
    i = np.zeros(n + 1)
    x_int = 0.0
    ucmd = []
    for k in range(n):
        e = 1.0 - i[k]
        ucmd.append(Kp * e + x_int)
        x_int += Tc * Ki * e
        un = ucmd[k - nfull] if k - nfull >= 0 else 0.0
        uo = ucmd[k - nfull - 1] if k - nfull - 1 >= 0 else 0.0
        i[k + 1] = a * i[k] + b_old * uo + b_new * un
    times = [0.0] + [k * Tc + tau for k in range(n) if k * Tc + tau < n * Tc]
    levels = [0.0] + [ucmd[k] for k in range(n) if k * Tc + tau < n * Tc]
    plant = TimedRL(L, R, times, levels)
    tr = simulate(plant, 0, [0.0], 0.0, n * Tc)
    err = max(abs(tr.state_at(k * Tc)[0][0] - i[k]) for k in range(1, n))
    res.add_check(Check("분수 지연 이산 map vs 실제 갱신 시각의 연속 plant (엔진)", "PASS" if err < 1e-12 else "FAIL", err, "A", 1e-12, path="b_old/b_new로 나눈 정확 이산화 vs 갱신 시각마다 전압을 바꾸는 RL 회로의 행렬지수 해", independent=True, detail=f"{n}샘플 최대 차이 {err:.3g} A"))
    led = energy_ledger(tr, plant, 0.0, n * Tc, ["p_in"], [], ["p_R"], rated_power=max(abs(x) for x in levels) * 1.0 + 1e-9)
    res.add_check(ledger_check(led, what="인가 전압이 한 일 = R 손실 + ½Li²: "))
    # timing diagram: 3 control periods of a ramp command
    npd = 3
    t_end = npd * Tc
    tt = np.linspace(0, t_end, 601)
    Tcar = 1.0 / v["f_pwm"]
    car = [1 - abs(((t / Tcar) % 1.0) * 2 - 1) for t in tt]  # triangle, valley at 0
    res.add_series("car", "PWM 캐리어 (정규화)", "", tt.tolist(), car)
    cmd = lambda t: 0.2 + 0.25 * t / Tc  # noqa: E731
    res.add_series("cmd", "제어기가 보는 명령 (연속, 예: 램프)", "", tt.tolist(), [cmd(t) for t in tt], dash=True)
    ks = list(range(-3, npd + 1))
    xs, ys = [], []
    for kk in ks:
        a0, b0 = kk * Tc + tau, kk * Tc + tau + Tc
        if b0 <= 0 or a0 >= t_end:
            continue
        xs += [max(a0, 0.0), min(b0, t_end)]
        ys += [cmd(kk * Tc), cmd(kk * Tc)]
    res.add_series("held", "실제 인가값 (샘플값을 τ 뒤 T_c 동안 유지)", "", xs, ys)
    res.add_series("adc", "ADC 샘플 시각", "", [k * Tc for k in range(npd)], [cmd(k * Tc) for k in range(npd)], style="points")
    bands = []
    for k in range(npd):
        t0 = k * Tc
        bands.append({"x0": t0, "x1": min(t0 + v["t_calc"], t_end), "mode": "calc", "label": "ADC·연산"})
        if tau > v["t_calc"] and nfull == 0 or v["policy"] != "immediate":
            t_up = t0 + tau
            if t0 + v["t_calc"] < min(t_up, t0 + Tc):
                bands.append({"x0": t0 + v["t_calc"], "x1": min(t_up, t0 + Tc, t_end), "mode": "hold", "label": "갱신 대기 (이전 duty 유지)"})
    bands = [b for b in bands if b["x1"] > b["x0"]]
    res.add_plot("p_tm", "타이밍: 샘플 → 연산 → PWM 갱신 → 유지", ["car", "cmd", "held", "adc"], x_label="t", x_unit="s", y_label="정규화", y_unit="", bands=bands, group="tm", level="B",
                 vlines=[{"x": k * Tc + tau, "label": "갱신"} for k in range(npd) if k * Tc + tau <= t_end],
                 proved=f"실제 인가값은 샘플값이 τ = {tau * 1e6:.4g} µs 뒤부터 T_c = {Tc * 1e6:.4g} µs 동안 유지된 계단이다. 유효 지연 T_d = τ + T_c/2 = {Td * 1e6:.4g} µs.",
                 not_yet="ADC 변환 시간·센서 필터·PWM 갱신의 하드웨어 세부(shadow register, 캐리어 비교)는 순서만 모델링했다.")
    fs_ = np.geomspace(100.0, min(0.5 / Tc, 5e4), 300)
    res.add_series("lagf", "지연 위상 360·f·T_d", "deg", fs_.tolist(), (360 * fs_ * Td).tolist())
    res.add_plot("p_lag", "지연이 소비하는 위상", ["lagf"], x_label="f", x_unit="Hz", y_label="위상 지연", y_unit="deg", kind="xy", log_x=True, level="A",
                 markers=[{"x": f1, "y": lag1, "label": f"{lag1:.3g}°"}, {"x": f2, "y": lag2, "label": f"{lag2:.3g}°"}] if f2 < 0.5 / Tc else [{"x": f1, "y": lag1, "label": f"{lag1:.3g}°"}],
                 proved="같은 지연이라도 crossover를 10배 올리면 소비하는 위상이 10배가 된다.", not_yet="ZOH의 크기 감쇠(sinc)는 위상과 별개로 존재한다.")
    fcs = np.geomspace(200.0, 0.2 / Tc, 40)
    pm_ex, pm_ap = [], []
    for fc in fcs:
        wc_ = 2 * math.pi * fc
        m_ = margins(lambda f, Kp=L * wc_, Ki=R * wc_: disc_loop_gain_frac(f, Kp, Ki, L, R, Tc, tau), fc / 1000, 0.4999 / Tc)
        pm_ex.append(m_.pm if m_.pm is not None else float("nan"))
        pm_ap.append(90 - 360 * fc * Td)
    res.add_series("pm_ex", "PM 정확 이산", "deg", fcs.tolist(), pm_ex)
    res.add_series("pm_ap", "PM 연속 근사 90° − 360 f_c T_d", "deg", fcs.tolist(), pm_ap, dash=True)
    res.add_plot("p_pm", "crossover에 따른 전류 루프 위상여유", ["pm_ex", "pm_ap"], x_label="설계 crossover f_c", x_unit="Hz", y_label="PM", y_unit="deg", kind="xy", log_x=True, level="B",
                 hlines=[{"y": 45.0, "label": "학습 기준 45°"}], vlines=[{"x": f1, "label": f"{f1 / 1e3:g} kHz"}, {"x": f2, "label": f"{f2 / 1e3:g} kHz"}],
                 proved="지연 때문에 crossover를 올릴수록 여유가 줄고, 정확 이산 모델은 연속 근사와 다르다.", not_yet="센서 필터·계산 지연의 변동(jitter)은 넣지 않았다.")
    res.circuit = {"diagram": system_diagram(v).to_json(), "intervals": bands, "plot_group": "tm"}
    for fc, mz_ in ((f1, mz1), (f2, mz2)):
        rho = frac_loop_rho(L * 2 * math.pi * fc, R * 2 * math.pi * fc, L, R, Tc, tau)
        res.add_metric(f"rho_{int(round(fc))}", f"폐루프 극점 최대 |z| (f_c = {fc / 1e3:g} kHz)", rho, "", basis="특성다항식 (z−1)(z−a)z^(n+1) + (Kp(z−1) + K_i T_c)(b_new z + b_old)의 근")
        if rho >= 1.0:
            res.verdict("UNSTABLE", f"f_c = {fc / 1e3:g} kHz 설계: 이 타이밍(T_d = {Td * 1e6:.4g} µs)에서 폐루프 극점 |z| = {rho:.4f} ≥ 1 — 지연이 위상을 {360 * fc * Td:.3g}° 소비한다.")
        elif mz_.pm is not None and mz_.pm < 45:
            res.verdict("MARGINAL", f"f_c = {fc / 1e3:g} kHz 설계: 안정하지만 위상여유 {mz_.pm:.3g}° < 45° (학습 기준) — 지연 위상 {360 * fc * Td:.3g}°")
        else:
            res.verdict("PASS_WITHIN_MODEL", f"f_c = {fc / 1e3:g} kHz 설계: 위상여유 {mz_.pm:.3g}° (T_d = {Td * 1e6:.4g} µs)")
    res.assumptions += ["샘플은 캐리어 골(또는 정점)에서, 연산 시간 일정 (jitter 없음)", "극점 소거 PI (Kp = Lω_c, Ki = Rω_c), 이상 평균 변조기", "plant: FL06의 RL (0.8 mH, 0.1 Ω)"]
    res.not_valid_for += ["연산 시간 변동·인터럽트 지연 분포", "PWM 비교기·dead time의 전압 오차", "입력필터·CPL과 결합된 전체 시스템 (실험 2)"]
    res.interpretation = (
        f"지연은 ‘1.5 T_s’가 아니라 샘플 시각·연산 시간·갱신 시각으로 정해진다: 여기서는 {tm['note']} 뒤 T_c = {Tc * 1e6:.4g} µs 동안 유지되므로 T_d = {Td * 1e6:.4g} µs다. "
        f"crossover에서 소비하는 위상은 360·f_c·T_d라 1 kHz에서 {lag1:.3g}°, 10 kHz에서 {lag2:.3g}°다. 이것은 전류 루프의 여유이고, 입력필터와의 전체 시스템 안정성은 실험 2에서 따로 판정한다."
    )
    return res


def system_diagram(v: dict) -> Circuit:
    """Block diagram: loop nesting, sampling/actuation timing, anti-windup and the impedance port."""
    c = Circuit("sys_ex07", 840, 330, title="PFC–DC-link–DC/DC 시스템: 루프 중첩·샘플 타이밍·AW·impedance port")
    c.add("block", "GRID", 70, 70, 0, "계통", w=90, h=40)
    c.add("block", "FILT", 190, 70, 0, "입력 필터 Z_s", w=110, h=40)
    c.add("block", "PFC", 340, 70, 0, "PFC 전력단", w=120, h=40)
    c.add("block", "DCL", 480, 70, 0, "DC-link C", w=100, h=40)
    c.add("block", "DCDC", 640, 70, 0, "DC/DC (CPL 대역)", w=150, h=40)
    c.add("block", "ADC", 190, 170, 0, "ADC 샘플 t_k", w=120, h=34)
    c.add("block", "CALC", 350, 170, 0, "연산: 전류 PI + AW", w=150, h=34)
    c.add("block", "PWM", 520, 170, 0, "PWM 갱신 (ZOH)", w=130, h=34)
    c.add("block", "VPI", 350, 260, 0, "외부 루프: DC-link 전압 PI (느림)", w=240, h=34)
    c.wire("p1", (115, 70), (135, 70))
    c.wire("p2", (245, 70), (280, 70))
    c.wire("p3", (400, 70), (430, 70))
    c.wire("p4", (530, 70), (565, 70))
    c.wire("s1", (300, 90), (300, 120), (190, 120), (190, 153))
    c.wire("s2", (250, 170), (275, 170))
    c.wire("s3", (425, 170), (455, 170))
    c.wire("s4", (520, 153), (520, 135), (390, 135), (390, 90))
    c.wire("v1", (480, 90), (480, 115), (610, 115), (610, 260), (470, 260))
    c.wire("v2", (350, 243), (350, 187))
    c.text(262, 50, "impedance port", "note")
    c.text(720, 150, "전체 안정성: Z_s/Z_in (실험 2)", "note")
    c.text(720, 215, "전류 루프 여유: 이 타이밍 (실험 3)", "note")
    c.text(720, 300, "포화·AW·bumpless (실험 4)", "note")
    c.mode("calc", "ADC·연산 중", ["ADC", "CALC", "s1", "s2"], "샘플 시각의 전류로 PI를 계산한다 (아직 인가 전 — 지연의 일부)", dim=["PWM"])
    c.mode("hold", "갱신 대기", ["CALC", "s3", "PWM"], "계산은 끝났지만 PWM 갱신 사건까지 이전 duty가 유지된다", dim=["ADC"])
    return c


# ======================================================================================
# Experiment 4: dq vector saturation, anti-windup on the applied vector, bumpless transfer
# ======================================================================================

SAT_CHOICES = [("circle", "원형 한계 + 적용 벡터로 AW"), ("axis", "축별 clamp (AW는 축별 값) → 변조기가 원형으로 다시 제한"), ("circle_noaw", "원형 한계, AW 없음")]


def dq_sat_run(v: dict, strategy: str, n: int):
    """dq current loop (FL06 plant) with a V2G step that saturates the voltage vector."""
    L, R, w = v["L"], v["R"], 2 * math.pi * v["f"]
    Ts = 1 / v["fs"]
    Vpk = v["V_LL"] * SQ2 / SQ3
    Vmax = v["V_dc"] / SQ3
    wc = 2 * math.pi * v["fc"]
    Kp = L * wc
    Ki = R * wc if v["zero"] == "cancel" else Kp * wc / 5.0
    Kaw = Ki / Kp
    Phi, Gam = dq_plant_map(L, R, w, Ts)
    x = np.zeros(2)
    xd = xq = 0.0
    pipe = [(Vpk, 0.0)]
    k0 = int(round(1e-3 / Ts))
    rec = {k: np.zeros(n) for k in ("id", "iq", "vd", "vq", "vdu", "vqu", "xd", "xq", "mag_cmd", "sat")}
    for k in range(n):
        idr = v["id_step"] if k >= k0 else 0.0
        iqr = v["iq_step"] if k >= k0 else 0.0
        ed, eq = idr - x[0], iqr - x[1]
        ud, uq = Kp * ed + xd, Kp * eq + xq
        vcd = Vpk + w * L * x[1] - ud
        vcq = 0.0 - w * L * x[0] - uq
        mag = math.hypot(vcd, vcq)
        if strategy == "axis":
            cd, cq = min(max(vcd, -Vmax), Vmax), min(max(vcq, -Vmax), Vmax)  # what the controller believes
            m2 = math.hypot(cd, cq)
            sc = min(1.0, Vmax / m2) if m2 > 0 else 1.0
            ad, aq = cd * sc, cq * sc  # what the modulator can actually apply (circle)
            bd, bq = cd, cq  # AW reference = the controller's own clamp
        else:
            sc = min(1.0, Vmax / mag) if mag > 0 else 1.0
            ad, aq = vcd * sc, vcq * sc
            bd, bq = ad, aq
        # back-calculation in the u domain: u implied by the reference vector minus the computed u
        ud_b = Vpk + w * L * x[1] - bd
        uq_b = -w * L * x[0] - bq
        corr = 0.0 if strategy == "circle_noaw" else Kaw
        xd += Ts * (Ki * ed + corr * (ud_b - ud))
        xq += Ts * (Ki * eq + corr * (uq_b - uq))
        pipe.append((ad, aq))
        va = pipe.pop(0)
        rec["id"][k], rec["iq"][k] = x
        rec["vd"][k], rec["vq"][k] = va
        rec["vdu"][k], rec["vqu"][k] = vcd, vcq
        rec["xd"][k], rec["xq"][k] = xd, xq
        rec["mag_cmd"][k] = mag
        rec["sat"][k] = 1.0 if mag > Vmax + 1e-9 else 0.0
        x = Phi @ x + Gam @ (np.array([Vpk, 0.0]) - np.array(va))
    rec["t"] = np.arange(n) * Ts - 1e-3
    return rec


def dq_metrics(rec, v):
    """Overshoot beyond the target on each axis, vector-error settling and IAE after the step."""
    post = rec["t"] >= 0
    dt = rec["t"][1] - rec["t"][0]
    idf, iqf = v["id_step"], v["iq_step"]
    sd = 1.0 if idf >= 0 else -1.0
    sq = 1.0 if iqf >= 0 else -1.0
    id_ov = max(0.0, float(np.max(sd * (rec["id"][post] - idf)))) / max(abs(idf), 1e-9) * 100
    iq_ov = max(0.0, float(np.max(sq * (rec["iq"][post] - iqf))))
    err = np.hypot(rec["id"] - idf, rec["iq"] - iqf)
    band = 0.02 * max(math.hypot(idf, iqf), 1e-9)
    out = np.where(post & (err > band))[0]
    settle = (float(rec["t"][out[-1]] + dt) if out[-1] + 1 < rec["t"].size else None) if out.size else 0.0
    iae = float(np.sum(err[post]) * dt)
    return {"id_ov": id_ov, "iq_ov": iq_ov, "settle": settle, "iae": iae, "sat_time": float(np.sum(rec["sat"])) * dt, "xd_max": float(np.max(np.abs(rec["xd"]))), "xq_max": float(np.max(np.abs(rec["xq"])))}


def bumpless_run(v: dict, bumpless: bool, n: int):
    """1-axis loop without e-feedforward: normal PI -> protection mode during a sag -> back to the PI."""
    L, R = v["L"], v["R"]
    Ts = 1 / v["fs"]
    wc = 2 * math.pi * v["fc"]
    Kp, Ki = L * wc, R * wc
    a, b = rl_zoh(L, R, Ts)
    e0 = v["V_LL"] * SQ2 / SQ3
    i_ref = v["i_hold"]
    k1, k2 = int(round(1e-3 / Ts)), int(round(3e-3 / Ts))
    i = i_ref
    xI = e0 + R * i_ref  # steady state of the PI without feedforward
    u_prev = xI
    rec = {k: np.zeros(n) for k in ("i", "u", "xI", "e", "mode")}
    for k in range(n):
        e = 0.9 * e0 if k >= k1 else e0
        err = i_ref - i
        if k1 <= k < k2:  # protection mode: PI frozen, feedforward-only command
            u = e + R * i_ref
            mode = 1.0
        else:
            if k == k2 and bumpless:
                xI = u_prev - Kp * err  # initialise so that the output is continuous
            u = Kp * err + xI
            xI += Ts * Ki * err
            mode = 0.0 if k < k1 else 2.0
        rec["i"][k], rec["u"][k], rec["xI"][k], rec["e"][k], rec["mode"][k] = i, u, xI, e, mode
        i = a * i + b * (u - e)  # (one-sample delay omitted here: the transfer, not the delay, is the subject)
        u_prev = u
    rec["t"] = np.arange(n) * Ts - 3e-3
    return rec


def run_saturation_transfer(v: dict) -> Result:
    res = Result("EX07", "saturation_transfer", "B (샘플링 dq 루프 + 1축 루프)")
    Ts = 1 / v["fs"]
    n = int(round(v["t_sim"] / Ts))
    Vmax = v["V_dc"] / SQ3
    recs = {s: dq_sat_run(v, s, n) for s, _ in SAT_CHOICES}
    mets = {s: dq_metrics(recs[s], v) for s in recs}
    sel = v["sat"]
    R_ = recs[sel]
    M = mets[sel]
    res.add_metric("Vmax", "전압 벡터 한계 V_dc/√3 (SVPWM 선형, 상전압 peak)", Vmax, "V")
    res.add_metric("mag_max", "제어기가 요구한 최대 |v_c|", float(np.max(R_["mag_cmd"])), "V", basis="포화 전 계산값")
    res.add_metric("id_ov", f"i_d overshoot ({dict(SAT_CHOICES)[sel]})", M["id_ov"], "%", basis="목표를 넘어선 양 / step 크기")
    res.add_metric("iq_ov", "i_q 목표 초과량", M["iq_ov"], "A", basis="q축 목표를 넘어선 최대값")
    res.add_metric("iae", "벡터 오차 적분 IAE = ∫|i* − i| dt", M["iae"], "A·s", basis="step 이후, 두 축 합성")
    res.add_metric("settle", "2 % 정정시간 (벡터 오차)", M["settle"] if M["settle"] is not None else "창 안에서 정정 안 됨", "s" if M["settle"] is not None else "")
    res.add_metric("sat_time", "벡터 포화 시간", M["sat_time"], "s")
    base = mets["circle"]
    res.add_metric("id_ov_ref", "i_d overshoot (원형 + 적용 벡터 AW, 비교 기준)", base["id_ov"], "%")
    res.add_metric("iae_ratio", "IAE / 기준(원형 + 적용 벡터 AW)", M["iae"] / base["iae"], "", basis="1보다 크면 기준보다 나쁨")
    # bumpless transfer
    nb = int(round(7e-3 / Ts))
    B0 = bumpless_run(v, False, nb)
    B1 = bumpless_run(v, True, nb)
    k2 = int(round(3e-3 / Ts))
    jump0 = float(B0["u"][k2] - B0["u"][k2 - 1])
    jump1 = float(B1["u"][k2] - B1["u"][k2 - 1])
    dev0 = float(np.max(np.abs(B0["i"][k2:] - v["i_hold"])))
    dev1 = float(np.max(np.abs(B1["i"][k2:] - v["i_hold"])))
    res.add_metric("jump_no", "복귀 순간 전압 명령 점프 (bumpless 없음)", jump0, "V", basis="적분기가 sag 전 값을 그대로 들고 있음")
    res.add_metric("jump_bl", "복귀 순간 전압 명령 점프 (bumpless)", jump1, "V")
    res.add_metric("dev_no", "복귀 후 전류 최대 편차 (bumpless 없음)", dev0, "A")
    res.add_metric("dev_bl", "복귀 후 전류 최대 편차 (bumpless)", dev1, "A")
    res.add_check(Check("bumpless: 복귀 순간 명령 연속", "PASS" if abs(jump1) < 1e-9 * max(1.0, abs(jump0)) + 1e-9 else "FAIL", abs(jump1), "V", 1e-9, path="x_I ← u_prev − Kp·e 초기화 후 첫 출력 vs 직전 출력", independent=False, detail="회귀 확인 (정의상 0)"))
    # independent check: dq model vs abc simulation for the selected strategy (reuse FL06 path idea)
    Vpk = v["V_LL"] * SQ2 / SQ3
    w = 2 * math.pi * v["f"]
    ia = np.zeros(3)
    k2p = 2 * math.pi / 3
    dev = 0.0
    m_chk = min(n, 200)
    for k in range(m_chk):
        t0 = k * Ts
        vd, vq = R_["vd"][k], R_["vq"][k]
        idm = 2 / 3 * sum(ia[j] * math.cos(w * t0 - j * k2p) for j in range(3))
        iqm = -2 / 3 * sum(ia[j] * math.sin(w * t0 - j * k2p) for j in range(3))
        dev = max(dev, math.hypot(idm - R_["id"][k], iqm - R_["iq"][k]))

        def rhs(t, y, vd=vd, vq=vq):
            out = np.empty(3)
            for j in range(3):
                th = w * t - j * k2p
                out[j] = (Vpk * math.cos(th) - (vd * math.cos(th) - vq * math.sin(th)) - v["R"] * y[j]) / v["L"]
            return out

        ia = solve_ivp(rhs, (t0, t0 + Ts), ia, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
    res.add_check(Check("dq 포화 궤적: dq 모델 vs 같은 인가 전압의 abc 적분", "PASS" if dev < 1e-6 * max(abs(v["id_step"]), 1.0) else "FAIL", dev, "A", 1e-6 * max(abs(v["id_step"]), 1.0), path="포화된 dq 전압을 3상으로 되돌려 DOP853 적분 후 Park 변환 vs dq 정확 map", independent=True, detail=f"{m_chk}샘플"))
    # series
    t = R_["t"].tolist()
    res.add_series("id", f"i_d ({dict(SAT_CHOICES)[sel]})", "A", t, R_["id"].tolist())
    res.add_series("iq", f"i_q ({dict(SAT_CHOICES)[sel]})", "A", t, R_["iq"].tolist())
    res.add_series("id_ref", "i_d (원형 + AW, 기준)", "A", t, recs["circle"]["id"].tolist(), dash=True)
    res.add_series("iq_ref", "i_q (원형 + AW, 기준)", "A", t, recs["circle"]["iq"].tolist(), dash=True)
    bands = []
    for k in range(n):
        mode = "sat" if R_["sat"][k] > 0 else "lin"
        if bands and bands[-1]["mode"] == mode:
            bands[-1]["x1"] = t[k] + Ts
        else:
            bands.append({"x0": t[k], "x1": t[k] + Ts, "mode": mode, "label": "벡터 포화" if mode == "sat" else "선형"})
    res.add_plot("p_dq", "회생 + 무효전류 step: 전압 벡터 포화 중 dq 전류", ["id", "iq", "id_ref", "iq_ref"], x_label="step 이후 시간", y_label="전류", y_unit="A", bands=bands, group="sat", level="B",
                 proved="축별 clamp는 제어기가 믿는 벡터와 실제로 인가되는 원형 제한 벡터가 달라 적분기가 잘못 되돌려지고, q축까지 흔들린다. 적용 벡터로 anti-windup을 걸면 회복이 빠르다.",
                 not_yet="SVPWM 육각형 한계·과변조·d축 우선 같은 다른 정책은 비교하지 않았다. 평형 계통·이상 PLL 가정.")
    ang = np.linspace(0, 2 * math.pi, 181)
    res.add_series("circ", "원형 한계 |v| = V_dc/√3", "V", (Vmax * np.cos(ang)).tolist(), (Vmax * np.sin(ang)).tolist(), dash=True)
    res.add_series("vtraj_u", "요구 벡터 (포화 전)", "V", R_["vdu"].tolist(), R_["vqu"].tolist(), style="points")
    res.add_series("vtraj", "인가 벡터", "V", R_["vd"].tolist(), R_["vq"].tolist())
    res.add_plot("p_vec", "전압 벡터 궤적 (dq 평면)", ["circ", "vtraj_u", "vtraj"], x_label="v_d", x_unit="V", y_label="v_q", y_unit="V", kind="xy", level="B",
                 proved="요구 벡터가 원 밖으로 나가는 동안 인가 벡터는 원 위에 머문다. 축별 clamp는 원 밖의 정사각형 꼭짓점을 요구할 수 있다(변조기가 만들 수 없는 벡터).",
                 not_yet="그래프의 가로·세로 축척이 같지 않아 원형 한계가 타원처럼 보인다. SVPWM 육각형 한계는 그리지 않았다.")
    tb = B0["t"].tolist()
    res.add_series("bi0", "i (bumpless 없음)", "A", tb, B0["i"].tolist())
    res.add_series("bi1", "i (bumpless)", "A", tb, B1["i"].tolist(), dash=True)
    res.add_series("bu0", "u (bumpless 없음)", "V", tb, B0["u"].tolist())
    res.add_series("bu1", "u (bumpless)", "V", tb, B1["u"].tolist(), dash=True)
    bb = []
    for k in range(nb):
        md = {0.0: "normal", 1.0: "fault", 2.0: "resume"}[float(B0["mode"][k])]
        lab = {"normal": "정상 PI", "fault": "보호 모드 (sag, PI 정지)", "resume": "PI 복귀"}[md]
        if bb and bb[-1]["mode"] == md:
            bb[-1]["x1"] = tb[k] + Ts
        else:
            bb.append({"x0": tb[k], "x1": tb[k] + Ts, "mode": md, "label": lab})
    res.add_plot("p_bl_i", "보호 모드 → PI 복귀: 전류", ["bi0", "bi1"], x_label="복귀 시각 기준", y_label="i", y_unit="A", bands=bb, group="bl", level="B",
                 proved="적분기가 sag 이전 값을 들고 복귀하면 전압 명령이 점프해 전류가 튄다. 복귀 순간 적분기를 현재 출력에 맞추면(bumpless) 점프가 없다.",
                 not_yet="startup ramp만으로 급격한 부하 탈락·보호 복귀를 대신하지 않는다. 보호 모드의 실제 동작(게이트 차단 등)은 모델 밖이다.")
    res.add_plot("p_bl_u", "보호 모드 → PI 복귀: 전압 명령", ["bu0", "bu1"], x_label="복귀 시각 기준", y_label="u", y_unit="V", bands=bb, group="bl", level="B",
                 proved="bumpless가 없으면 복귀 순간 명령이 sag 크기만큼(적분기가 sag 이전 값을 기억) 점프한다.", not_yet="명령의 slew 제한·보호 모드의 실제 게이트 동작은 넣지 않았다.")
    rows = []
    for s, lab in SAT_CHOICES:
        m_ = mets[s]
        rows.append([lab, m_["id_ov"], m_["iq_ov"], (m_["settle"] * 1e3) if m_["settle"] is not None else "정정 안 됨", m_["iae"] * 1e3, m_["sat_time"] * 1e3, m_["xd_max"]])
    res.tables.append(Table("t_sat", "포화 정책 비교 (같은 회생 + 무효전류 step)", ["정책", "i_d overshoot [%]", "i_q 초과 [A]", "정정 [ms]", "IAE [mA·s]", "포화 시간 [ms]", "|x_d| 최대 [V]"], rows,
                            note="anti-windup은 ‘실제로 인가된’ 벡터로 되돌려야 한다. 축별 clamp 값으로 되돌리면 변조기의 원형 제한과 어긋난다."))
    res.circuit = {"diagram": sat_circuit(v).to_json(), "intervals": bands, "plot_group": "sat"}
    if M["settle"] is None or M["id_ov"] > 20:
        st_txt = "없음" if M["settle"] is None else f"{M['settle'] * 1e3:.3g} ms"
        res.verdict("MARGINAL", f"{dict(SAT_CHOICES)[sel]}: overshoot {M['id_ov']:.3g} %, 정정 {st_txt} — 학습용 기준(20 %, 창 안 정정)을 넘는다.")
    else:
        extra = "" if sel == "circle" else f" 다만 IAE가 기준(원형 + 적용 벡터 AW)의 {M['iae'] / base['iae']:.3g}배다."
        res.verdict("PASS_WITHIN_MODEL", f"{dict(SAT_CHOICES)[sel]}: overshoot {M['id_ov']:.3g} %, 정정 {M['settle'] * 1e3:.3g} ms (학습용 기준 안).{extra} bumpless 복귀는 명령 점프 0.")
    res.assumptions += ["FL06 dq plant (0.8 mH, 0.1 Ω), 평형 계통·이상 PLL, 1 샘플 연산 지연", f"회생 + 무효전류 step i_d {v['id_step']:g} A, i_q {v['iq_step']:g} A; V_dc {v['V_dc']:g} V; PI 영점 {'= plant 극점' if v['zero'] == 'cancel' else 'ω_c/5'}", "bumpless 실험: e 피드포워드 없는 1축 루프, sag −10 % 동안 PI 정지"]
    res.not_valid_for += ["SVPWM 육각형·과변조 영역", "보호 동작의 하드웨어 타이밍", "입력필터·CPL과의 결합 (실험 2)"]
    res.interpretation = (
        "원형 한계는 변조기가 실제로 만들 수 있는 전압의 한계라 제어기도 같은 한계로 벡터를 잘라야 한다. 축별 clamp는 원 밖의 벡터를 요구하고, 그 값으로 anti-windup을 하면 실제 인가 벡터와 어긋나 적분기가 잘못 쌓여 회복이 늦고 q축이 흔들린다. "
        "모드 전환에서는 적분기가 이전 상태의 값을 들고 돌아오면 명령이 점프하므로, 복귀 순간 적분기를 현재 출력에 맞춘다(bumpless)."
    )
    return res


def sat_circuit(v: dict) -> Circuit:
    c = Circuit("sat_dq", 780, 300, title="dq 전류 제어의 전압 벡터 포화와 anti-windup 경로")
    c.add("block", "PI", 110, 80, 0, "PI_d, PI_q", w=130, h=40)
    c.add("block", "DEC", 280, 80, 0, "v_g 피드포워드 + ωL 디커플링", w=190, h=40)
    c.add("block", "LIM", 470, 80, 0, "|v| ≤ V_dc/√3 (원형)", w=170, h=40)
    c.add("block", "MOD", 650, 80, 0, "SVPWM → 컨버터", w=150, h=40)
    c.add("block", "AW", 380, 200, 0, "AW: K_aw·(u_적용 − u_계산)", w=230, h=40)
    c.add("block", "PLANT", 650, 200, 0, "RL 필터 + 계통 (dq)", w=160, h=40)
    c.wire("a1", (175, 80), (185, 80))
    c.wire("a2", (375, 80), (385, 80))
    c.wire("a3", (555, 80), (575, 80))
    c.wire("a4", (650, 100), (650, 180))
    c.wire("aw1", (470, 100), (470, 180))
    c.wire("aw2", (265, 200), (110, 200), (110, 100))
    c.wire("fb", (570, 200), (550, 200), (550, 250), (40, 250), (40, 80), (45, 80))
    c.text(300, 280, "적용된(포화 후) 벡터로 적분기를 되돌린다 — 축별 clamp 값이 아니다", "note")
    c.mode("lin", "선형", ["PI", "DEC", "LIM", "MOD", "PLANT", "a1", "a2", "a3", "a4", "fb"], "요구 벡터가 원 안: 포화 없음", dim=["AW"])
    c.mode("sat", "벡터 포화", ["PI", "DEC", "LIM", "MOD", "PLANT", "AW", "a1", "a2", "a3", "a4", "aw1", "aw2", "fb"], "요구 벡터가 원 밖: 방향을 유지해 원 위로 줄이고 AW가 적분기를 되돌린다")
    return c


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q = [
    Question(
        "V_s = 400 V, R = 0.2 Ω, L = 1 mH, P = 10 kW CPL의 평형점과 임계 C를 유도하라.",
        "L di/dt = V_s − Ri − v, C dv/dt = i − P/v. 평형: v² − V_s v + RP = 0 → V_e = (V_s + √(V_s² − 4RP))/2 = 394.935887 V. g = P/V_e², A = [[−R/L, −1/L],[1/C, g/C]]. "
        "tr A < 0 → C > Lg/R = 320.565519 µF, det > 0 → Rg < 1. C = 100 µF는 220.57 ± j3134.19 s⁻¹로 불안정, 1 mF는 −67.94 ± j991.24 s⁻¹로 안정. 평형점 존재 한계 P ≤ V_s²/(4R) = 200 kW는 별개의 조건이다.",
        "Derive the equilibrium and the critical capacitance for the 400 V, 0.2 ohm, 1 mH, 10 kW constant-power-load example.",
        "The equilibrium solves v squared minus Vs v plus R P equals zero, giving 394.94 V on the high branch. Linearising with g equal to P over V squared, the trace condition requires C greater than L g over R, about 320.6 microfarads, and the determinant requires R g below one. So 100 microfarads is unstable and 1 millifarad is stable, while the static power limit of 200 kW is a separate condition.",
        ["394.935887 V", "trace·det", "320.565519 µF", "P_max 200 kW 별개"],
        kind="calc",
    ),
    Question(
        "전류 루프 위상여유가 60°인데 시스템이 발진한다. 무엇부터 보나?",
        "단일 루프 판정은 이상 전원을 전제로 한다. 입력필터 출력 임피던스 Z_s와 컨버터 입력 임피던스 Z_in을 같은 포트 약속으로 구해 minor-loop Z_s/Z_in의 Nyquist를 본다. "
        "컨버터가 전력을 붙잡는 대역 안에서는 −V²/P의 음저항이라 필터 공진의 감쇠를 없앤다. |Z_s| < |Z_in|은 보수적 충분조건일 뿐이다. 저전압·고부하 코너에서 다시 확인한다.",
        "The current loop has 60 degrees of margin, but the system oscillates. Where do you look first?",
        "A single-loop margin assumes a stiff source. I would compute the filter output impedance and the converter input impedance at the same port and check the minor-loop gain on a Nyquist plot. Inside its control bandwidth the converter looks like a negative resistance of V squared over P, which cancels the filter damping; the magnitude separation rule is only a conservative sufficient condition, and I would repeat the check at low line and full load.",
        ["이상 전원 전제", "Z_s/Z_in Nyquist", "대역 안 음저항", "코너 확인"],
        kind="pressure",
    ),
    Question(
        "15 µs 지연은 1 kHz와 10 kHz crossover에서 각각 얼마의 위상을 먹나? 지연은 어떻게 산정하나?",
        "360·f·T_d: 1 kHz에서 5.4°, 10 kHz에서 54°. T_d는 샘플 시각·연산 시간·PWM 갱신 시각과 ZOH 반주기로 산정하며 1.5T_s로 고정하지 않는다. 예: 50 kHz double update(T_c = 10 µs), 연산 8 µs → 다음 갱신까지 10 µs + 5 µs = 15 µs.",
        "How much phase does a 15 microsecond delay cost at 1 kHz and 10 kHz crossover, and how do you estimate the delay?",
        "Three hundred sixty times f times the delay: 5.4 degrees at 1 kHz and 54 degrees at 10 kHz. I estimate the delay from the ADC sample instant, the computation time and the PWM update instant, plus half the hold period, rather than assuming one and a half samples.",
        ["5.4°", "54°", "샘플·연산·갱신", "ZOH ½"],
        kind="calc",
    ),
    Question(
        "dq 전압 포화에서 각 축을 따로 clamp하면 무엇이 문제인가?",
        "변조기가 만들 수 있는 한계는 원(또는 육각형)이다. 축별 clamp는 원 밖의 벡터를 요구해 방향이 바뀌고, 그 값으로 anti-windup을 하면 실제 인가 벡터와 어긋나 적분기가 잘못 쌓인다. "
        "벡터를 방향을 유지해 원으로 줄이고 적용된 벡터로 back-calculation한다. 모드 전환은 bumpless로 적분기를 초기화한다.",
        "What goes wrong if you clamp the d and q voltages separately?",
        "The modulator's real limit is a circle or hexagon, so separate clamps can request a vector it cannot produce and change the vector direction; anti-windup based on those clamped values disagrees with the vector actually applied, so the integrators wind up. Scale the vector onto the circle and back-calculate from the applied vector, and initialise the integrators at mode transitions.",
        ["원형 한계", "방향 왜곡", "적용 벡터 AW", "bumpless"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="cpl_exact",
        title="소스 R–L + C + CPL: 교재 예제를 세 경로로 재현",
        goal=(
            "400 V·0.2 Ω·1 mH·10 kW에서 V_e = 394.935887 V, 증분저항 −15.5974 Ω, C_crit = 320.565519 µF를 계산하고, C = 100 µF는 불안정(220.57 ± j3134.19 s⁻¹), 1 mF는 안정(−67.94 ± j991.24 s⁻¹)임을 "
            "해석 극점·비선형 모델의 수치 Jacobian·비선형 적분 세 경로로 확인한다. P > 200 kW는 평형점이 없다(NO_SOLUTION)."
        ),
        params=[
            Param("Vs", "소스 전압 V_s", "V", 400.0, "V", vmin=1, vmax=2000, source="TEXTBOOK"),
            Param("R", "소스 저항 R", "Ω", 0.2, "Ω", vmin=1e-4, vmax=100, source="TEXTBOOK"),
            Param("L", "소스 인덕턴스 L", "H", 1e-3, "mH", vmin=1e-7, vmax=1.0, source="TEXTBOOK"),
            Param("C", "노드 커패시턴스 C", "F", 100e-6, "µF", vmin=1e-7, vmax=1.0, source="TEXTBOOK", source_note="100 µF (불안정) / 1 mF (안정)"),
            Param("P", "CPL 전력 P", "W", 10e3, "kW", vmin=1.0, vmax=1e7, source="TEXTBOOK", source_note="10 kW"),
            Param("dv0", "초기 전압 교란", "V", 1.0, "V", vmin=1e-3, vmax=50, source="TEXTBOOK", source_note="교재 그림의 1 V", group="시뮬레이션"),
            Param("t_end", "적분 길이", "s", 30e-3, "ms", vmin=1e-3, vmax=0.5, source="ASSUMED", group="시뮬레이션"),
            Param("dv_valid", "이상 CPL 유효범위 |Δv|/V_e", "", 0.2, "", vmin=0.01, vmax=0.9, source="ASSUMED", source_note="넘으면 적분 중단 (UVLO·전류 제한 전)", group="시뮬레이션"),
            Param("v_min_frac", "붕괴 판정 하한 v/V_s (해 없음일 때)", "", 0.5, "", vmin=0.05, vmax=0.95, source="ASSUMED", group="시뮬레이션"),
        ],
        presets=[
            Preset("c100u", "C = 100 µF (교재: 불안정)", {}, "E07", ("nominal", "reference", "failure")),
            Preset("c1m", "C = 1 mF (교재: 안정)", {"C": 1e-3}, "E07", ("variant", "reference")),
            Preset("c_crit", "C = 330 µF (임계 바로 위)", {"C": 330e-6}, "느린 감쇠", ("corner",)),
            Preset("p250k", "P = 250 kW (> P_max)", {"P": 250e3}, "평형점 없음", ("failure", "reference")),
        ],
        run=run_cpl_exact,
        model_level="A (해석 극점) + 비선형 ODE + 수치 Jacobian",
        suggested_change="C를 100 µF → 1 mF로 바꾼다 (그다음 P를 250 kW로).",
        prediction=Prediction(
            "C = 100 µF에서 1 mF로 늘리면 평형점 + 1 V 교란의 응답은?",
            ["둘 다 불안정", "1 mF만 안정 (감쇠 진동)", "둘 다 안정", "모르겠다"],
            "1 mF만 안정 (감쇠 진동)",
            "C_crit = L·g/R = 320.6 µF. 100 µF는 trace(A) = −R/L + g/C > 0이라 진동이 자라고, 1 mF는 trace < 0이라 감쇠한다. 둘 다 평형점(394.9 V)은 존재한다.",
            ["pole_re", "Ccrit", "sig_nl"],
            handcalc=[{"key": "Ve", "label": "V_e", "unit": "V"}, {"key": "Rinc", "label": "증분저항", "unit": "Ω"}, {"key": "Ccrit", "label": "C_crit", "unit": "F"}],
        ),
        suggested={"C": 1e-3},
        student=(
            "정전력 부하는 전압이 내려가면 같은 전력을 위해 전류를 더 끌어간다. 그러면 전압이 더 내려가는 방향이라, 저항처럼 진동을 줄이는 대신 키운다. "
            "L–C 공진이 있으면 이 ‘음의 저항’이 소스 저항의 감쇠를 이겨 진동이 자랄 수 있고, C를 키우면 음저항의 영향(g/C)이 줄어 안정해진다."
        ),
        expert=(
            "① 평형점 존재(P ≤ V_s²/4R)와 동적 안정(trace·det)은 다른 조건이다. ② 해석 극점, 비선형 모델의 수치 Jacobian, 비선형 적분의 성장률을 따로 계산해 일치를 본다. "
            "③ 불안정 경우 비선형 적분은 이상 CPL 모델의 유효범위(|Δv| 20 %)에서 멈춘다 — 그 뒤의 전압 붕괴·UVLO·전류 제한은 이 모델로 예측하지 않는다. "
            "④ 저전압 가지는 det < 0으로 항상 안장점이다. ⑤ C 증가는 해결책 중 하나일 뿐, inrush·비용·수명·필터 corner를 함께 보고 damping·제어 대안과 비교한다."
        ),
        customer_ko=(
            "현재 필터 C 100 µF는 이 전원 임피던스와 10 kW 정전력 부하에서 C_crit 약 321 µF보다 작아 입력 전압이 약 500 Hz로 발진하며 커지는 조건입니다. "
            "C를 늘리는 방법 외에 감쇠 가지나 컨버터 대역 조정도 비교해 보시고, 실제 컨버터 입력 임피던스로 다시 확인하시죠."
        ),
        customer_en=(
            "With 100 microfarads the filter is below the critical value of about 321 microfarads for this source impedance and a 10 kW constant-power load, so the input voltage oscillates at about 500 Hz and grows. "
            "Besides adding capacitance, let's compare a damping branch or a change in the converter bandwidth, and confirm with the converter's real input impedance."
        ),
        questions=[_Q[0]],
        circuit="cpl",
        textbook=[TB_E07],
        reference_presets=["c100u", "c1m", "p250k"],
        claim_limit="이상 무한대역 CPL의 국소 선형 결과와 유효범위 안의 비선형 적분. 큰 진폭 붕괴·보호 동작은 주장하지 않는다.",
    ),
    Experiment(
        key="converter_impedance",
        title="실제 converter는 유한 대역 CPL이다: Z_s/Z_in, 코너, 단일 루프 vs 전체 시스템",
        goal=(
            "같은 입력필터 뒤에 전류·전압 루프를 가진 평균 buck을 두고, 입력 임피던스 Z_in(jω)가 제어 대역 안에서만 −V²/P의 음저항임을 보인다. minor-loop Z_s/Z_in의 Nyquist 판정을 전체 시스템 고유값·비선형 시간영역과 대조하고, "
            "전압·부하 코너 5개의 주파수 응답을 비교한다. 컨버터 자신의 루프 여유와 전체 시스템 안정성을 따로 보고한다."
        ),
        params=[
            Param("Vs", "소스 전압 V_s", "V", 400.0, "V", vmin=100, vmax=1000, source="TEXTBOOK"),
            Param("R", "소스 저항 R", "Ω", 0.2, "Ω", vmin=1e-3, vmax=10, source="TEXTBOOK"),
            Param("L", "소스 L", "H", 1e-3, "mH", vmin=1e-6, vmax=0.1, source="TEXTBOOK"),
            Param("C", "필터 C", "F", 100e-6, "µF", vmin=1e-6, vmax=0.1, source="TEXTBOOK"),
            Param("Rd", "감쇠가지 R_d", "Ω", 0.0, "Ω", vmin=0, vmax=100, source="ASSUMED", source_note="C_d = 0이면 감쇠가지 없음", group="필터"),
            Param("Cd", "감쇠가지 C_d", "F", 0.0, "µF", vmin=0, vmax=0.1, source="ASSUMED", group="필터"),
            Param("P", "컨버터 출력 전력", "W", 10e3, "kW", vmin=100, vmax=1e5, source="TEXTBOOK", source_note="10 kW (저항부하 R_o = V_o²/P)"),
            Param("Vo", "출력 전압 V_o", "V", 250.0, "V", vmin=10, vmax=350, source="ASSUMED", group="컨버터"),
            Param("Lo", "출력 L_o", "H", 200e-6, "µH", vmin=1e-6, vmax=0.01, source="ASSUMED", group="컨버터"),
            Param("Co", "출력 C_o", "F", 220e-6, "µF", vmin=1e-6, vmax=0.1, source="ASSUMED", group="컨버터"),
            Param("fci", "전류 루프 crossover f_ci", "Hz", 5000.0, "Hz", vmin=50, vmax=20000, source="ASSUMED", group="제어"),
            Param("fbw", "전압 루프 crossover", "Hz", 200.0, "Hz", vmin=5, vmax=2000, source="ASSUMED", group="제어"),
            Param("ff", "입력전압 피드포워드 (d = u/v)", "", False, kind="bool", source="ASSUMED", group="제어"),
            Param("Vnom", "피드포워드 없을 때 변조기 기준 전압", "V", 400.0, "V", vmin=100, vmax=1000, source="ASSUMED", group="제어"),
            Param("dv0", "시간영역 교란", "V", 1.0, "V", vmin=0.01, vmax=20, source="TEXTBOOK", group="시뮬레이션"),
            Param("t_end", "시간영역 길이", "s", 40e-3, "ms", vmin=5e-3, vmax=0.2, source="ASSUMED", group="시뮬레이션"),
        ],
        presets=[
            Preset("nominal", "f_ci 5 kHz, C 100 µF", {}, "루프는 안정, 시스템은 불안정", ("nominal", "reference", "failure")),
            Preset("slow", "f_ci 1 kHz (공진 부근까지만 CPL)", {"fci": 1000.0, "fbw": 50.0}, "C < C_crit인데 안정", ("variant", "reference")),
            Preset("feedforward", "입력전압 피드포워드", {"ff": True}, "모든 주파수에서 이상 CPL", ("failure", "reference")),
            Preset("big_c", "C = 1 mF", {"C": 1e-3}, "용량 증가", ("variant", "reference")),
            Preset("damped", "감쇠가지 R_d 3 Ω, C_d 400 µF", {"Rd": 3.0, "Cd": 400e-6}, "감쇠 대안", ("variant",)),
        ],
        run=run_converter_impedance,
        model_level="B (평균 컨버터 소신호 + 비선형 시간영역)",
        suggested_change="전류 루프 crossover를 5 kHz → 1 kHz로 낮춘다 (C 100 µF 유지). 다음에는 입력전압 피드포워드를 켠다.",
        prediction=Prediction(
            "C = 100 µF(< C_crit)에서 컨버터 전류 루프 대역을 1 kHz로 낮추면 전체 시스템은?",
            ["여전히 불안정 — C가 C_crit보다 작으니까", "안정해질 수 있다", "대역과 무관하다", "모르겠다"],
            "안정해질 수 있다",
            "필터 공진(약 500 Hz)이 전류 루프 대역 근처나 밖이면 공진 주파수에서 컨버터가 전력을 완전히 붙잡지 못해 순수한 음저항이 아니다. 이상 CPL(무한 대역)의 C_crit 판정은 이 경우 보수적이다.",
            ["sys_re", "nyq", "mb"],
            handcalc=[{"key": "Rn", "label": "V_e²/P", "unit": "Ω"}],
        ),
        suggested={"fci": 1000.0, "fbw": 50.0},
        student=(
            "컨버터는 출력을 일정하게 지키려고 입력 전압이 내려가면 전류를 더 끌어간다. 그런데 이 반응은 제어가 따라갈 수 있는 속도(대역) 안에서만 일어난다. "
            "필터가 흔들리는 주파수가 그 대역 안이면 컨버터가 흔들림을 키우고, 대역 밖이면 컨버터는 그냥 저항처럼 보인다."
        ),
        expert=(
            "① 같은 포트 약속으로 Z_s(필터 출력)와 Z_in(컨버터 입력)을 구한다. ② minor-loop Z_s/Z_in의 Nyquist는 소스·부하 subsystem이 각각 안정하다는 전제에서 판정한다; 여기서는 1 + T_m의 위상 회전 수와 전체 Jacobian의 불안정 고유값 수를 대조한다. "
            "③ |Z_s| < |Z_in|은 보수적 충분조건 — 위상을 버리므로 ‘미충족 = 불안정’이 아니다. ④ 입력전압 피드포워드는 선 전압 외란을 잘 막지만 컨버터를 전 대역 CPL로 만든다. "
            "⑤ 코너(저전압·고부하)마다 V²/P와 변조기 이득이 바뀐다. ⑥ 단일 루프의 위상여유는 전체 시스템 안정성의 근거가 아니다."
        ),
        customer_ko=(
            "컨버터 전류 루프만 보면 위상여유가 충분하지만, 입력필터와 연결하면 약 500 Hz 공진에서 컨버터의 음의 입력저항이 감쇠를 없애 전체가 불안정합니다. "
            "Z_s와 Z_in을 같은 포트에서 측정·계산해 비교하고, 저전압·고부하 코너에서 필터 감쇠나 루프 대역 조정 중 무엇이 적절한지 같이 보시죠."
        ),
        customer_en=(
            "The converter's current loop alone has enough phase margin, but connected to the input filter its negative input resistance cancels the damping at the roughly 500 Hz filter resonance, so the whole system is unstable. "
            "Let's compare the filter output impedance and the converter input impedance at the same port, and decide between filter damping and a bandwidth change at the low-line, full-load corner."
        ),
        questions=[_Q[1]],
        circuit="cpl_conv",
        textbook=[TB_E07, TB_08],
        reference_presets=["nominal", "slow", "feedforward", "big_c"],
        runtime_hint="seconds",
        claim_limit="평균모델(B)의 소신호 임피던스와 비선형 시간영역. 스위칭·샘플링·포화를 포함한 실측 임피던스는 주장하지 않는다.",
    ),
    Experiment(
        key="digital_delay",
        title="디지털 지연: 샘플·연산·갱신 순서가 위상을 정한다",
        goal=(
            "ADC 샘플 시각, 연산 시간, PWM 갱신 시각으로 실제 지연을 산정하고(예: 50 kHz double update, 연산 8 µs → T_d = 15 µs), 360·f·T_d가 1 kHz에서 5.4°, 10 kHz에서 54°임을 "
            "샘플·hold 시뮬레이션으로 확인한다. 분수 지연을 포함한 정확 이산 모델로 전류 루프 여유를 crossover별로 보고한다."
        ),
        params=[
            Param("f_pwm", "PWM 캐리어 주파수", "Hz", 50e3, "kHz", vmin=1e3, vmax=500e3, source="ASSUMED", source_note="double update와 함께 T_d = 15 µs가 되도록"),
            Param("update", "갱신 방식", "", "double", kind="choice", choices=[("double", "double update (골·정점)"), ("single", "single update (골)")], source="ASSUMED"),
            Param("t_calc", "연산 시간 (샘플→계산 완료)", "s", 8e-6, "µs", vmin=0, vmax=1e-3, source="ASSUMED"),
            Param("policy", "갱신 정책", "", "next", kind="choice", choices=[("next", "다음 갱신 사건에 반영 (shadow register)"), ("immediate", "연산 직후 즉시 반영")], source="ASSUMED"),
            Param("fc1", "crossover 1", "Hz", 1000.0, "Hz", vmin=10, vmax=1e5, source="TEXTBOOK", source_note="1 kHz", group="루프"),
            Param("fc2", "crossover 2", "Hz", 10000.0, "Hz", vmin=10, vmax=1e5, source="TEXTBOOK", source_note="10 kHz", group="루프"),
            Param("L", "plant L", "H", 0.8e-3, "mH", vmin=1e-6, vmax=0.1, source="TEXTBOOK", source_note="FL06", group="루프"),
            Param("R", "plant R", "Ω", 0.1, "Ω", vmin=1e-4, vmax=10, source="TEXTBOOK", group="루프"),
        ],
        presets=[
            Preset("td15", "50 kHz double, 연산 8 µs → 15 µs", {}, "교재 예의 지연", ("nominal", "reference")),
            Preset("slip", "연산 12 µs (갱신 한 번 놓침)", {"t_calc": 12e-6}, "25 µs", ("failure", "reference")),
            Preset("single", "single update", {"update": "single"}, "30 µs", ("variant",)),
            Preset("immediate", "즉시 갱신", {"policy": "immediate"}, "13 µs", ("variant",)),
        ],
        run=run_digital_delay,
        model_level="B (샘플링 순서의 정확 이산 모델)",
        suggested_change="연산 시간을 8 → 12 µs로 늘려 갱신 사건을 한 번 놓치게 한다.",
        prediction=Prediction(
            "연산이 8 µs에서 12 µs로 늘어 10 µs 갱신 사건을 놓치면 10 kHz crossover의 지연 위상은?",
            ["54° 그대로 (4 µs만 늘었으니 약간)", "90°로 크게 늘어난다", "줄어든다", "모르겠다"],
            "90°로 크게 늘어난다",
            "갱신은 다음 사건(20 µs)으로 밀려 τ = 20 µs, T_d = 20 + 5 = 25 µs → 360 × 10 kHz × 25 µs = 90°. 지연은 연산 시간에 연속적으로 비례하지 않고 갱신 사건 단위로 뛴다.",
            ["Td", "lag2", "pm2"],
            handcalc=[{"key": "lag1", "label": "1 kHz 지연 위상", "unit": "deg"}, {"key": "lag2", "label": "10 kHz 지연 위상", "unit": "deg"}],
        ),
        suggested={"t_calc": 12e-6},
        student="제어기는 전류를 재고(샘플), 계산하고, 그 결과를 PWM에 넣기까지 시간이 걸린다. 그리고 넣은 값은 다음 갱신까지 그대로 유지된다. 이 ‘늦음’이 빠른 신호일수록 큰 위상 지연이 된다.",
        expert=(
            "① T_d = (갱신 − 샘플) + T_hold/2. 1.5·T_s는 한 가지 순서의 결과일 뿐이다. ② 연산 시간이 갱신 사건을 넘으면 지연이 한 주기 단위로 뛴다(slip). ③ 정확 이산 모델은 분수 지연을 b_old/b_new로 나눠 넣는다. "
            "④ 이 여유는 전류 루프의 여유다 — 입력필터·CPL과의 전체 시스템 안정성은 별도 판정이다. ⑤ ZOH는 위상뿐 아니라 크기(sinc)도 바꾼다."
        ),
        customer_ko=(
            "현재 샘플·연산·갱신 순서로 유효 지연은 15 µs이고, 10 kHz 전류 루프라면 지연만으로 54°를 씁니다. 연산이 10 µs 갱신 사건을 넘기면 25 µs로 뛰어 90°가 되니, "
            "최악 연산 시간과 갱신 정책을 먼저 확정하고 crossover를 정하시죠."
        ),
        customer_en=(
            "With the present sample, compute and update sequence the effective delay is 15 microseconds, which alone costs 54 degrees at a 10 kHz current-loop crossover. If the computation misses the 10 microsecond update event the delay jumps to 25 microseconds and 90 degrees, "
            "so let's fix the worst-case compute time and update policy before choosing the crossover."
        ),
        questions=[_Q[2]],
        circuit="sys_ex07",
        textbook=[TB_E07, TB_09],
        reference_presets=["td15", "slip"],
        claim_limit="샘플링 순서의 정확 이산 모델(B). jitter·하드웨어 PWM 세부는 주장하지 않는다.",
    ),
    Experiment(
        key="saturation_transfer",
        title="dq 벡터 포화·적용 벡터 anti-windup·bumpless 복귀",
        goal=(
            "V2G(회생) 전류 step이 전압 벡터 한계 V_dc/√3를 넘는 조건에서 원형 한계와 축별 clamp, 적용 벡터 기반 anti-windup의 차이를 dq 전류·벡터 궤적으로 비교하고, "
            "보호 모드에서 PI로 복귀할 때 적분기 초기화(bumpless)가 전압 점프와 전류 튐을 없애는 것을 본다."
        ),
        params=[
            Param("V_LL", "계통 선간전압", "V", 400.0, "V", vmin=50, vmax=1000, source="TEXTBOOK"),
            Param("f", "계통 주파수", "Hz", 50.0, "Hz", vmin=10, vmax=400, source="TEXTBOOK"),
            Param("L", "상당 L", "H", 0.8e-3, "mH", vmin=1e-5, vmax=0.1, source="TEXTBOOK", source_note="FL06"),
            Param("R", "상당 R", "Ω", 0.1, "Ω", vmin=1e-4, vmax=10, source="TEXTBOOK"),
            Param("fc", "전류 루프 f_c", "Hz", 1000.0, "Hz", vmin=10, vmax=5000, source="TEXTBOOK"),
            Param("fs", "샘플링", "Hz", 40e3, "kHz", vmin=2e3, vmax=2e5, source="ASSUMED"),
            Param("V_dc", "DC-link (V_max = V_dc/√3)", "V", 600.0, "V", vmin=100, vmax=2000, source="ASSUMED", source_note="여유가 작은 합성 값", group="포화"),
            Param("id_step", "i_d step (음수 = 회생, V2G)", "A", -30.0, "A", vmin=-200, vmax=200, source="ASSUMED", group="포화"),
            Param("iq_step", "i_q step (무효전류 지원)", "A", 40.0, "A", vmin=-200, vmax=200, source="ASSUMED", group="포화"),
            Param("zero", "PI 영점", "", "cancel", kind="choice", choices=[("cancel", "교재 1차 설계: 영점 = plant 극점"), ("wc5", "외란 억제형: 영점 = ω_c/5")], source="TEXTBOOK", source_note="FL06과 같은 선택", group="포화"),
            Param("sat", "포화 정책", "", "axis", kind="choice", choices=SAT_CHOICES, source="ASSUMED", group="포화"),
            Param("i_hold", "bumpless 실험 전류", "A", 20.0, "A", vmin=0, vmax=200, source="ASSUMED", group="bumpless"),
            Param("t_sim", "dq 시뮬레이션 길이", "s", 12e-3, "ms", vmin=3e-3, vmax=40e-3, source="ASSUMED", group="시뮬레이션"),
        ],
        presets=[
            Preset("axis", "축별 clamp (+ 변조기 원형 제한)", {}, "AW가 실제 벡터와 어긋남", ("nominal", "reference")),
            Preset("circle", "원형 한계 + 적용 벡터 AW", {"sat": "circle"}, "권장 구조", ("variant", "reference")),
            Preset("noaw", "원형 한계, AW 없음", {"sat": "circle_noaw"}, "windup", ("failure", "reference")),
            Preset("axis_wc5", "축별 clamp, 적분이 강한 PI", {"zero": "wc5"}, "차이가 더 커진다", ("corner",)),
        ],
        run=run_saturation_transfer,
        model_level="B (샘플링 dq 루프 + 1축 루프)",
        suggested_change="포화 정책을 ‘축별 clamp’에서 ‘원형 한계 + 적용 벡터 AW’로 바꾼다.",
        prediction=Prediction(
            "축별 clamp를 ‘원형 한계 + 적용 벡터 AW’로 바꾸면 i_q 목표 초과와 벡터 오차 IAE는?",
            ["차이 없다", "줄어든다 (방향 왜곡·AW 불일치가 사라짐)", "커진다", "모르겠다"],
            "줄어든다 (방향 왜곡·AW 불일치가 사라짐)",
            "축별 clamp는 변조기가 만들 수 없는 벡터를 요구하고, 그 값으로 적분기를 되돌려 실제 인가 벡터와 어긋난다. 원형 한계로 방향을 지키고 적용 벡터로 back-calculation하면 두 축 적분기가 실제와 맞는다. "
            "이 계통 필터는 ωL이 작아 q축 전압 요구가 크지 않으므로 차이는 중간 정도다 — 가장 큰 차이는 anti-windup 유무(‘AW 없음’ preset)에서 나온다.",
            ["iq_ov", "iae_ratio", "settle"],
            handcalc=[{"key": "Vmax", "label": "V_dc/√3", "unit": "V"}],
        ),
        suggested={"sat": "circle"},
        student="컨버터가 낼 수 있는 전압에는 ‘원’ 모양의 한계가 있다. d와 q를 따로 잘라내면 원 밖을 요구하게 되고, 제어기가 착각한 값으로 적분기를 고치면 두 축이 서로 방해한다.",
        expert=(
            "① 적용 벡터(포화 후)로 back-calculation: ẋ = K_i e + K_aw(u_적용 − u_계산). ② 축별 clamp는 방향을 바꾸고 원 밖(정사각형 꼭짓점)을 요구한다. ③ d축 우선 같은 정책은 목적(전력 vs 무효전력)에 따라 고른다. "
            "④ 모드 전환(보호·전류 제한·정상)에서는 적분기를 현재 출력에 맞춰 bumpless로 복귀한다. startup ramp 시험이 급격한 부하 탈락·보호 복귀를 대신하지 않는다."
        ),
        customer_ko=(
            "회생 전류 step에서 전압 벡터가 한계에 걸리는데, 현재처럼 d·q를 따로 clamp하면 q축 전류가 흔들리고 회복이 늦습니다. 벡터를 원형으로 제한하고 실제 인가 벡터로 anti-windup을 걸며, "
            "보호 모드 복귀 시 적분기 초기화 로직이 있는지도 확인해 주세요."
        ),
        customer_en=(
            "During the regenerative current step the voltage vector hits its limit, and clamping d and q separately disturbs the q-axis current and slows recovery. I suggest limiting the vector on a circle and back-calculating the integrators from the applied vector, "
            "and please check that the integrators are initialised when returning from protection mode."
        ),
        questions=[_Q[3]],
        circuit="sat_dq",
        textbook=[TB_E07, TB_09],
        reference_presets=["axis", "circle"],
        runtime_hint="seconds",
        claim_limit="평형 계통·이상 PLL의 샘플링 dq 루프(B). 과변조·보호 하드웨어 타이밍은 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="EX07",
    title="제어·입력 상호작용 — 단일 루프가 안정해도 시스템은 불안정할 수 있다",
    title_en="A stable loop can still make an unstable system",
    track="expert",
    order=7,
    path_note="E13 4회전 (E07)",
    textbook=[TB_E07, TB_09, TB_08, TB_E13],
    prerequisites=["FL05", "FL06"],
    summary="CPL 교재 예제의 세 경로 재현 → 유한 대역 컨버터의 Z_in과 minor-loop Nyquist(코너별) → 샘플·연산·갱신 지연과 전류 루프 여유 → dq 벡터 포화·적용 벡터 AW·bumpless. 루프 여유와 전체 시스템 안정성을 따로 보고한다.",
    experiments=EXPERIMENTS,
    minimum_scope=(
        "CPL exact example(394.935887 V, 320.565519 µF, 100 µF 불안정·1 mF 안정)을 비선형 적분·local Jacobian으로 독립 확인; 유한 대역 converter 입력 임피던스 비교; "
        "15 µs 지연 1 kHz 5.4°·10 kHz 54°; delay/saturation/antiwindup 실험; 최소 3개 전압/부하 코너 주파수응답; current-loop margin과 whole-system 안정성 각각 보고 (E07·E13 표)"
    ),
    claim_limits=[
        "이상 CPL 결과는 무한대역·국소(선형) 결과이며 큰 진폭 붕괴는 예측하지 않음",
        "컨버터 임피던스는 평균모델(B) 소신호 — 스위칭·샘플링 포함 실측 임피던스 아님",
        "|Z_s| < |Z_in|은 보수적 충분조건으로만 사용",
        "스위칭 파형에 임의 사인 리플을 더해 측정 THD처럼 보고하지 않음 (THD는 FL05의 시뮬레이션 전류로만)",
        "루프 여유·overshoot 기준은 학습용이며 고객 사양이 아님",
    ],
    test_paths=["tests/test_ex07.py"],
    extends=["FL05", "FL06"],
)
