"""FL09 - LLC: explain resonance in words, confirm it with equations (textbook ch.12).

Level A: FHA with the textbook definitions (R_ac = 8/pi^2 n^2 R_dc, Q = Z0/R_ac, k = Lm/Lr, full bridge
|H|/n, half bridge |H|/(2n)), computed three independent ways: the textbook normalised closed form
(reference/resonant.py), nodal analysis of the phasor network, and the time-domain periodic solution of
the FHA circuit itself (a sinusoid into R_ac - which is NOT a switching verification).
Level C: ideal bridge, full-bridge diode rectifier (off state with the i2 = 0 constraint and a floating
node) and an output C/R in the exact switched engine; periodic state by shooting; checked against the
textbook state equations integrated independently (DOP853 + events).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from ..engine.switched import simulate
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import resonant as ref
from ._common import energy_ledger, ledger_check, series_from_traj
from ._resonant import (
    PI,
    ZVS_KO,
    ResonantSystem,
    Tank,
    bands_from,
    edge_currents,
    fha,
    fha_time_domain_gain,
    ivp_period,
    periodic,
    phasor_guess,
    resonant_circuit,
    scales_for,
    summarize,
    zvs_screen,
)

TB_12 = TextbookRef("llc-공진을-말로-설명하고-식으로-확인하기-fl09", "12. LLC — 공진을 말로 설명하고 식으로 확인하기 [FL09]")
TB_19 = TextbookRef("교재실습-계약과-통과-기준", "19. 교재–실습 계약과 통과 기준 (FHA 대 switching 5 %)")
TB_E02 = TextbookRef("expert-e02.-비선형-cossdead-timezvs-에너지식-하나로-판정하지-않기-ex02", "E02. 비선형 Coss·dead time·ZVS [EX02]")

COSS_HV = (0.5e-9, 40.0)  # synthetic C(v) = C0/sqrt(1+v/V0) (ASSUMED, same form as FL08/EX02)
Q_SET = (0.2, 0.8, 1.5)


def _tank_params(Lm: float = 200e-6) -> list[Param]:
    return [
        Param("Lr", "공진 L_r", "H", 40e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", source_note="합성 LLC L_r = 40 µH", group="tank"),
        Param("Cr", "공진 C_r", "F", 28.1448e-9, "nF", vmin=1e-11, vmax=1e-4, source="TEXTBOOK", source_note="28.1448 nF → f_r = 150 kHz", group="tank"),
        Param("Lm", "여자 L_m", "H", Lm, "µH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="200 µH (k = 5)", group="tank"),
    ]


def _power_params() -> list[Param]:
    return [
        Param("Vin", "입력 DC 전압 V_in", "V", 400.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="400 → 48 V 예제"),
        Param("Vo_nom", "f_r에서 맞출 출력 V_o", "V", 48.0, "V", vmin=0.5, vmax=2000, source="TEXTBOOK", source_note="n = V_in/V_o (FB), V_in/(2V_o) (HB)"),
        Param("bridge", "1차 bridge", "", "FB", "", kind="choice", choices=[("FB", "full bridge (±V_in)"), ("HB", "half bridge (V_in/0)")], source="TEXTBOOK", source_note="FB |H|/n, HB |H|/(2n)"),
    ]


def _kb(bridge: str) -> float:
    return 1.0 if bridge == "FB" else 2.0


def _tank(v) -> Tank:
    return Tank(v["Lr"], v["Cr"], v["Lm"], R1=v.get("R1", 0.0))


# ======================================================================================
# 1. FHA gain: three forms of the same equivalent circuit
# ======================================================================================


def run_fha(v: dict) -> Result:
    res = Result("FL09", "fha_gain", "A (FHA)")
    tank = _tank(v)
    fr = tank.fr1
    Z0 = tank.Z0
    k = v["Lm"] / v["Lr"]
    textbook_tank = abs(v["Lr"] - 40e-6) < 1e-12 and abs(v["Cr"] - 28.1448e-9) < 1e-15 and abs(v["Lm"] - 200e-6) < 1e-12
    res.add_metric("fr", "공진주파수 f_r = 1/(2π√(L_rC_r))", fr, "Hz", ref=150e3 if textbook_tank else None, ref_label="교재 150 kHz", tol=1e-6)
    res.add_metric("Z0", "특성 임피던스 Z₀ = √(L_r/C_r)", Z0, "Ω")
    res.add_metric("k", "k = L_m/L_r", k, "", ref=5.0 if textbook_tank else None, ref_label="200/40", tol=1e-12)
    kb = _kb(v["bridge"])
    n = v["Vin"] / (kb * v["Vo_nom"])
    tb400 = abs(v["Vin"] - 400) < 1e-9 and abs(v["Vo_nom"] - 48) < 1e-9
    res.add_metric("n", f"f_r에서 V_o를 맞추는 n ({v['bridge']})", n, "", ref=(8.3333333 if v["bridge"] == "FB" else 4.1666667) if tb400 else None, ref_label="교재 FB 8.3333 / HB 4.1667", tol=1e-7, basis="n = N_p/N_s (full-bridge 정류 전체 권선)")
    Q = v["Q"]
    Rac = Z0 / Q
    RL = PI**2 / 8 * Rac / n**2
    res.add_metric("Rac", f"R_ac = Z₀/Q (Q = {Q:g})", Rac, "Ω", basis="1차 환산 FHA 등가저항")
    res.add_metric("RL", "해당 실제 부하 R_L = (π²/8)R_ac/n²", RL, "Ω", basis="2차측 실제값")
    res.add_metric("P_fr", "f_r에서의 출력 (V_o²/R_L)", v["Vo_nom"] ** 2 / RL, "W", basis="FHA에서 f_r이면 |H| = 1")
    Fs = np.linspace(v["F_min"], v["F_max"], 241)
    qs = sorted(set(Q_SET) | {Q})
    max_dev = 0.0
    bounds = []
    for q in qs:
        g = [fha(tank, float(F) * fr, Z0 / q).gain for F in Fs]
        gr = [ref.llc_gain(float(F), k, q) for F in Fs]
        max_dev = max(max_dev, max(abs(a - b) for a, b in zip(g, gr)))
        ph = [math.degrees(math.atan2(fha(tank, float(F) * fr, Z0 / q).Zin.imag, fha(tank, float(F) * fr, Z0 / q).Zin.real)) for F in Fs]
        lab = f"Q = {q:g}" + (" (선택)" if q == Q else "")
        res.add_series(f"g_{q:g}", lab, "", Fs.tolist(), g, dash=(q != Q and Q not in Q_SET and False))
        res.add_series(f"ph_{q:g}", lab, "deg", Fs.tolist(), ph)
        # inductive/capacitive boundary: closed form (reference) and, independently, a root of Im Z_in from the
        # phasor impedance; both are kept so the table shows the exact value even when it lies outside the range
        b_ref = ref.llc_zin_boundary(k, q)
        b_num = None
        for a0, a1, p0, p1 in zip(Fs[:-1], Fs[1:], ph[:-1], ph[1:]):
            if p0 < 0 <= p1:
                b_num = float(brentq(lambda F, q=q: fha(tank, F * fr, Z0 / q).Zin.imag, float(a0), float(a1), xtol=1e-14, rtol=1e-14))
                break
        bounds.append((q, b_ref, b_num))
    res.add_plot(
        "p_gain",
        "FHA gain |H| — Q가 크면(부하가 무거우면) peak가 낮아지고 f_r 쪽으로 온다",
        [f"g_{q:g}" for q in qs],
        x_label="F = f/f_r",
        x_unit="",
        y_label="|H|",
        y_unit="",
        kind="xy",
        level="A",
        hlines=[{"y": 1.0, "label": "|H| = 1"}],
        vlines=[{"x": 1.0, "label": "f_r"}],
        markers=[{"x": b, "y": fha(tank, b * fr, Z0 / q).gain, "label": f"Q {q:g} 경계"} for q, b, _ in bounds if v["F_min"] <= b <= v["F_max"]],
        proved="교재 정규화식·페이저 절점해석·FHA 회로의 시간영역 해가 같은 |H|를 준다 (FHA 대수 검산).",
        not_yet="실제 정류기(다이오드 도통/차단)와 사각파 고조파는 없다 → ‘switching vs FHA’ 실험에서 비교한다.",
    )
    res.add_plot(
        "p_zin",
        "입력 임피던스 위상 ∠Z_in — 0°보다 크면 inductive (ZVS의 필요조건일 뿐)",
        [f"ph_{q:g}" for q in qs],
        x_label="F = f/f_r",
        x_unit="",
        y_label="∠Z_in",
        y_unit="deg",
        kind="xy",
        level="A",
        hlines=[{"y": 0.0, "label": "inductive / capacitive"}],
        proved="Im(Z_in) 부호로 FHA상 inductive 영역을 구분했다.",
        not_yet="inductive라도 dead time 동안의 전하가 부족하면 ZVS가 아니다 (ZVS·L_m 실험).",
    )
    rows = []
    for q, b, _ in bounds:
        where = "범위 안 전부 inductive" if b < v["F_min"] else ("범위 안 전부 capacitive" if b > v["F_max"] else "")
        rows.append([f"{q:g}"] + [f"{fha(tank, F * fr, Z0 / q).gain:.4f}" for F in (0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5)] + [f"F < {b:.4f} capacitive" + (f" ({where})" if where else "")])
    res.tables.append(Table("t_gain", "|H| 표 (FHA, 합성 tank)", ["Q", "F=0.7", "0.8", "0.9", "1.0", "1.1", "1.3", "1.5", "∠Z_in 경계"], rows, note="Q = Z₀/R_ac: 이 교재 정의에서는 부하가 무거울수록 Q가 크다. 다른 문헌의 reciprocal Q나 m = (L_m+L_r)/L_r와 섞지 않는다."))
    res.add_check(Check("교재 정규화식 vs 페이저 절점해석", "PASS" if max_dev < 1e-12 else "FAIL", max_dev, "", 1e-12, path="1/H = 1 + (1−F⁻²)/k + jQ(F−1/F) (reference) vs 절점 전압 방정식 (Z_r, Z_m, R_ac)", independent=True, detail=f"{len(qs)}개 Q × {len(Fs)}점의 최대 |Δ|H||"))
    b_dev = max((abs(bn - br) for _, br, bn in bounds if bn is not None), default=0.0)
    n_num = sum(1 for *_, bn in bounds if bn is not None)
    n_in = sum(1 for _, br, _ in bounds if v["F_min"] < br < v["F_max"])
    b_ok = b_dev < 1e-9 and n_num == n_in
    res.add_check(Check("∠Z_in 경계: 닫힌 식 vs 페이저 Im Z_in의 근", "PASS" if b_ok else "FAIL", b_dev, "", 1e-9, path="k²Q²y² + (1+k−k²Q²)y − 1 = 0 (y = F²) vs brentq(Im Z_in(F)) on the nodal phasor", independent=True, detail=f"범위 안 경계 {n_in}개 중 {n_num}개를 수치로 찾음; Q마다 경계가 다르다"))
    td_dev = 0.0
    for F in (0.7, 1.0, 1.5):
        td_dev = max(td_dev, abs(fha_time_domain_gain(tank, F * fr, Rac) - fha(tank, F * fr, Rac).gain))
    res.add_check(Check("FHA 회로의 시간영역 주기해 vs 페이저", "PASS" if td_dev < 1e-9 else "FAIL", td_dev, "", 1e-9, path="사인파 전원(발진기 상태)·R_ac 부하 회로를 행렬지수로 풀어 v_load 진폭 비교", independent=True, detail="이것은 FHA 회로 자체의 검산이다 — 정류기 switching 검증이 아님"))
    g1 = max(abs(fha(tank, fr, Z0 / q).gain - 1.0) for q in qs)
    res.add_check(Check("극한: F = 1에서 모든 Q의 |H| = 1", "PASS" if g1 < 1e-12 else "FAIL", g1, "", 1e-12, path="직렬 공진에서 Z_r = 0", independent=True))
    lim = max(abs(fha(tank, F * fr, 1e15).gain - 1 / abs(1 + (1 - F**-2) / k)) for F in (0.7, 0.9, 1.3, 1.5))
    res.add_check(Check("극한: Q → 0 (무부하) |H| = 1/|1 + (1−F⁻²)/k|", "PASS" if lim < 1e-9 else "FAIL", lim, "", 1e-9, path="L_r·C_r·L_m 분압만 남는다", independent=True))
    res.verdict("PASS_WITHIN_MODEL", "FHA |H|·∠Z_in이 세 독립 형태에서 일치 (합성 tank, 이상 소자)")
    res.verdict("INFO", "FHA는 정류기를 R_ac로 둔 사인파 회로다. ‘LLC switching verified’가 아니다 — 다음 실험에서 정류기 스위칭 모델과 비교한다.")
    res.assumptions += ["기본파 근사(FHA): 정류기 입력 전압 기본파와 전류가 동상", "R_ac = 8n²R_dc/π² (full-bridge 정류, 연속 도통)", "이상 소자·무손실 tank"]
    res.not_valid_for += ["경부하·f_r 아래의 불연속 정류(off 구간)", "ZVS 판정", "동특성(G_vf)"]
    res.interpretation = (
        "f_r에서는 직렬 L_r–C_r이 기본파에 대해 0 Ω이 되어 부하와 무관하게 |H| = 1이다. f_r 아래에서는 L_m이 공진에 참여해 gain이 올라갈 수 있지만, "
        "부하가 무거우면(Q 큼) R_ac가 L_m을 눌러 그 상승이 줄어든다. 그래서 무거운 부하에서 gain을 올리려고 주파수를 내리면 오히려 capacitive 영역으로 들어가 ZVS를 잃을 수 있다."
    )
    res.circuit = {"diagram": resonant_circuit("LLC", v["Vin"], f"{n:.4g}", tank, v["bridge"]).to_json(), "intervals": [], "plot_group": ""}
    return res


# ======================================================================================
# 2. Switching model vs FHA at one operating point
# ======================================================================================


def _llc_system(v: dict, F: float | None = None, Q: float | None = None, Lm: float | None = None):
    tank = Tank(v["Lr"], v["Cr"], Lm if Lm is not None else v["Lm"], R1=v.get("R1", 0.0))
    kb = _kb(v["bridge"])
    n = v["Vin"] / (kb * v["Vo_nom"])
    Q = v["Q"] if Q is None else Q
    Rac = tank.Z0 / Q
    RL = PI**2 / 8 * Rac / n**2
    f = (v["F"] if F is None else F) * tank.fr1
    sysl = ResonantSystem(tank, v["Vin"], f, n, v["bridge"], "rc", Co=v["Co"], Rout=RL, key="fl09")
    return tank, sysl, n, kb, Rac, RL


def _solve(v, F=None, Q=None, Lm=None):
    tank, sysl, n, kb, Rac, RL = _llc_system(v, F, Q, Lm)
    per = periodic(sysl, phasor_guess(sysl, Rac))
    s = summarize(sysl, per.traj, sysl.T) if per.traj is not None else None
    return tank, sysl, n, kb, Rac, RL, per, s


def _near_fr_midload(F: float, Q: float) -> bool:
    return 0.9 - 1e-9 <= F <= 1.1 + 1e-9 and 0.5 <= Q <= 1.2


def run_switching(v: dict) -> Result:
    res = Result("FL09", "switching_vs_fha", "C (이상 스위칭·다이오드 정류) + A (FHA 비교)")
    tank, sysl, n, kb, Rac, RL, per, s = _solve(v)
    fr, F, Q = tank.fr1, v["F"], v["Q"]
    T = sysl.T
    res.add_metric("n", "권선비 n (f_r에서 V_o = V_in/(k_b n))", n, "", basis=f"{v['bridge']}: k_b = {kb:g}")
    res.add_metric("RL", "부하 R_L (Q에서 유도)", RL, "Ω", basis="R_L = (π²/8)(Z₀/Q)/n², 2차 실제값")
    res.add_metric("f", "스위칭 주파수 f = F·f_r", F * fr, "Hz")
    if not per.converged or s is None:
        res.verdict("SOLVER_FAILED", f"주기해를 찾지 못함 (잔차 {per.residual:.2e}, {per.note})")
        return res
    g_fha = fha(tank, F * fr, Rac).gain
    vo_fha = g_fha * v["Vin"] / (kb * n)
    g_td = s["vo_avg"] * kb * n / v["Vin"]
    diff = g_td / g_fha - 1
    res.add_metric("vo_td", "출력 V_o (스위칭 해, 주기 평균)", s["vo_avg"], "V", basis="C_o 평균 — 리플 포함 정확 적분")
    res.add_metric("vo_fha", "출력 V_o (FHA 예측)", vo_fha, "V", basis=f"|H|·V_in/(k_b n), |H| = {g_fha:.5f}")
    res.add_metric("g_td", "정규화 gain (스위칭) k_b n V_o/V_in", g_td, "")
    res.add_metric("g_fha", "FHA |H|", g_fha, "")
    res.add_metric("diff", "스위칭/FHA − 1", diff, "", basis="+면 FHA가 gain을 과소평가")
    res.add_metric("off_frac", "정류 off 구간 비율 (i₂ = 0)", s["off_frac"], "", basis="주기 대비; 0이면 연속 도통")
    res.add_metric("P", "출력 전력 (부하)", s["P_load"], "W")
    res.add_metric("I1_rms", "공진 전류 i_r RMS (1차)", s["I1_rms"], "A")
    res.add_metric("Im_pk", "여자전류 peak", s["im_pk"], "A")
    res.add_metric("vC1_pk", "C_r 전압 peak (DC 포함)", s["vC1_pk"], "V", basis="HB이면 V_in/2 DC가 더해진다")
    res.add_metric("vo_pp", "출력 리플 p-p", s["vo_pp"], "V", basis=f"C_o = {v['Co'] * 1e6:g} µF (ASSUMED)")
    ec = edge_currents(sysl, per.traj, T)
    res.add_metric("i_edge", "1차 bridge 상승 edge 전류 i_r(0)", ec["i_rise"], "A", basis="음수 = 노드를 올려줄 방향 (inductive)")
    res.add_metric("rho", "plant Floquet 최대 |λ|", per.rho, "", basis="event 시각 민감도 포함, 제어기 없음")
    # waveforms: two periods from the periodic state
    tr = simulate(sysl, per.q0, per.x0, 0.0, 2 * T)
    series_from_traj(res, tr, {"v1": ("v₁ (bridge)", "V"), "vm": ("v_m (L_m 전압)", "V"), "v2": ("v₂′ (정류기 입력, 1차 환산)", "V"), "i1": ("i_r (공진 전류)", "A"), "im": ("i_m (여자 전류)", "A"), "i2": ("i₂′ = i_r − i_m (정류 전류, 1차 환산)", "A"), "vC1": ("v_Cr", "V"), "vo": ("v_o (출력)", "V")}, per_segment=40)
    bands = bands_from(tr, 0.0, 2 * T)
    res.add_plot("p_i", "전류: 정류 전류 i₂′ = i_r − i_m이 0에 닿으면 정류기가 꺼진다", ["i1", "im", "i2"], y_label="전류", y_unit="A", bands=bands, group="llc", level="C", hlines=[{"y": 0.0, "label": "0"}],
                 proved="다이오드 도통/차단을 guard event로 정확히 찾고, off 구간에는 i₂ = 0 제약으로 L_r+L_m이 함께 공진한다.",
                 not_yet="dead time·C_oss·SR 타이밍 없음 (이상 스위치).")
    res.add_plot("p_v", "전압: 도통 중 v_m = ±n·V_o로 clamp, off 구간에는 떠 있다", ["v1", "vm", "v2"], y_label="전압", y_unit="V", bands=bands, group="llc", level="C",
                 proved="정류기 입력 노드가 off 구간에 clamp 사이에서 떠 있음을 보인다 (floating node).", not_yet="정류기 접합 용량이 없는 이상 다이오드라서, 실제로는 off 구간의 떠 있는 노드가 정류기 용량과 공진하며 링잉한다. 이 그래프는 그 링잉을 보이지 않는다.")
    res.add_plot("p_c", "C_r 전압과 출력 전압", ["vC1", "vo"], y_label="전압", y_unit="V", bands=bands, group="llc", level="C",
                 proved="같은 정확 주기해에서 C_r 전압의 swing과 출력 전압을 읽는다. C_r 전압 peak는 부품 전압 정격 확인의 입력이 된다.",
                 not_yet="C_r의 ESR·온도 계수·전압 의존 용량은 없고, 출력 리플은 이상 C와 저항 부하 기준이다.")
    # checks
    led = energy_ledger(tr, sysl, 0.0, T, ["p_in"], ["p_load"], ["p_R"], rated_power=max(abs(s["P_in"]), 1.0))
    res.add_check(ledger_check(led))
    xT = ivp_period(sysl, per.x0)
    dev = float(np.max(np.abs(xT - per.x0) / scales_for(sysl)))
    res.add_check(Check("독립 경로: 교재 상태식 + DOP853 한 주기", "PASS" if dev < 1e-7 else "FAIL", dev, "rel", 1e-7, path="affine 행렬·행렬지수·guard 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환", independent=True, detail="엔진 주기해 x(0)에서 출발해 x(T) = x(0)인지"))
    res.add_check(Check("주기해 잔차 (shooting)", "PASS" if per.residual < 1e-9 else "FAIL", per.residual, "rel", 1e-9, path=f"물리 {per.pre_cycles}주기 후 Newton shooting, {per.iterations}회", independent=False))
    near = _near_fr_midload(F, Q)
    if near:
        res.add_check(Check("FHA 대 switching gain (f_r 부근 중부하 5 % 초기 점검값)", "PASS" if abs(diff) <= 0.05 else "FAIL", diff, "", 0.05, path="교재 19장 기준: 0.9 ≤ F ≤ 1.1, 0.5 ≤ Q ≤ 1.2", independent=True))
    else:
        res.add_check(Check("FHA 대 switching gain (5 % 점검값 적용 범위 밖)", "INFO", diff, "", 0.05, path="f_r에서 멀거나 경부하/과부하 — 차이는 모델 적용범위의 결과로 해석한다", independent=True, detail="억지 보정으로 지우지 않는다"))
    # ZVS screen (ideal model: not evaluable; screen only)
    zs = zvs_screen(ec["i_rise"], v["Vin"], v["td"], v["C0"], v["V0"])
    res.tables.append(Table("t_zvs", "1차 leg 전환 screen (SCREEN_ONLY: 전류를 edge 값으로 고정)", ["edge 전류", "필요 전하 2Q_oss", "가용 전하 |i|·t_d", "판정"],
                            [[f"{zs['i_edge']:.3f} A", f"{zs['q_req'] * 1e9:.1f} nC", f"{zs['q_av'] * 1e9:.1f} nC", ZVS_KO[zs["status"]]]],
                            note=f"합성 C_oss (C₀ = {v['C0'] * 1e9:g} nF, V₀ = {v['V0']:g} V), t_d = {v['td'] * 1e9:g} ns (ASSUMED). 최종 판정은 turn-on 직전 V_DS (EX02)."))
    res.verdict("PASS_WITHIN_MODEL", "이상 스위칭 정류 모델의 주기해 (에너지 잔차·독립 적분 일치)")
    if not near and abs(diff) > 0.05:
        res.verdict("OUT_OF_VALIDITY", f"이 운전점(F = {F:g}, Q = {Q:g}, 정류 off {s['off_frac'] * 100:.1f} %)에서 FHA gain이 스위칭 해와 {diff * 100:+.1f} % 다르다 — FHA 수치를 설계값으로 쓰지 않는다")
    res.verdict("NOT_EVALUABLE", "ZVS: 이상 스위치에는 노드 용량·dead time이 없다")
    res.verdict("SCREEN_ONLY", f"edge 전류 부호·전하 screen: {ZVS_KO[zs['status']]}")
    res.assumptions += ["이상 bridge (dead time 0)", "이상 다이오드 full-bridge 정류, 권선·다이오드 손실 없음", f"출력 C_o = {v['Co'] * 1e6:g} µF (ASSUMED), 부하 R_L은 Q에서 유도", "주기해 = shooting (초기 과도 제외)"]
    res.not_valid_for += ["ZVS 보증", "효율·온도", "SR 타이밍", "기동 과도"]
    below = F < 1
    res.interpretation = (
        ("f_r 아래에서는 공진 전류가 반주기보다 먼저 여자전류와 같아져 정류기가 꺼지고(off 구간), 그 동안 L_r+L_m이 C_r과 천천히 공진한다. FHA는 이 구간을 모르므로 gain을 과소평가한다. " if below else "")
        + ("f_r 위에서는 정류기가 연속 도통하지만 정류기 입력이 사각파라 고조파 때문에 FHA가 gain을 다소 과대평가한다. " if F > 1 else "")
        + ("f_r에서는 반주기마다 공진이 정확히 끝나 FHA와 스위칭 해가 거의 같다. " if abs(F - 1) < 1e-9 else "")
        + "차이는 ‘오차’가 아니라 FHA의 적용범위를 알려주는 결과다."
    )
    res.circuit = {"diagram": resonant_circuit("LLC", v["Vin"], f"{n:.4g}", tank, v["bridge"]).to_json(), "intervals": bands, "plot_group": "llc"}
    return res


# ======================================================================================
# 3. Switching gain curve vs FHA over F
# ======================================================================================


def run_curve(v: dict) -> Result:
    res = Result("FL09", "gain_curve", "C (주파수 sweep) + A")
    Fs = np.round(np.linspace(v["F_min"], v["F_max"], int(v["points"])), 6)
    rows, gt, gf, off, ie = [], [], [], [], []
    failed = []
    for F in Fs:
        tank, sysl, n, kb, Rac, RL, per, s = _solve(v, F=float(F))
        g_fha = fha(tank, F * tank.fr1, Rac).gain
        gf.append(g_fha)
        if not per.converged or s is None:
            failed.append(float(F))
            gt.append(None)
            off.append(None)
            ie.append(None)
            continue
        g_td = s["vo_avg"] * kb * n / v["Vin"]
        e = edge_currents(sysl, per.traj, sysl.T)["i_rise"]
        gt.append(g_td)
        off.append(s["off_frac"])
        ie.append(e)
        rows.append([f"{F:.3f}", f"{g_fha:.4f}", f"{g_td:.4f}", f"{(g_td / g_fha - 1) * 100:+.2f} %", f"{s['off_frac'] * 100:.1f} %", f"{e:+.3f} A", "inductive" if fha(tank, F * tank.fr1, Rac).inductive else "capacitive"])
    ok = [i for i, g in enumerate(gt) if g is not None]
    res.add_series("g_td", "스위칭 해 (정류기 포함)", "", [float(Fs[i]) for i in ok], [gt[i] for i in ok], style="points")
    res.add_series("g_fha", "FHA |H|", "", Fs.tolist(), gf, dash=True)
    res.add_series("off", "정류 off 비율", "", [float(Fs[i]) for i in ok], [off[i] for i in ok], style="points")
    res.add_series("ie", "상승 edge 전류 i_r(0)", "A", [float(Fs[i]) for i in ok], [ie[i] for i in ok], style="points")
    res.add_plot("p_curve", f"gain 곡선: 스위칭 해 vs FHA (Q = {v['Q']:g})", ["g_td", "g_fha"], x_label="F = f/f_r", x_unit="", y_label="gain", y_unit="", kind="xy", level="C + A", vlines=[{"x": 1.0, "label": "f_r"}],
                 proved="각 주파수에서 정류기 포함 주기해(shooting)를 구해 FHA 곡선과 겹쳤다.", not_yet="dead time·C_oss·손실이 없는 이상 모델. 실제 gain peak 위치는 이 요소들로 더 움직인다.")
    res.add_plot("p_off", "정류 off 구간 비율: f_r 아래에서 커진다", ["off"], x_label="F", x_unit="", y_label="비율", y_unit="", kind="xy", level="C",
                 proved="주파수마다 정류기 포함 주기해에서 off 구간(i₂ = 0)의 비율을 직접 쟀다. 이 비율이 커지는 f_r 아래 영역이 FHA(연속 도통 가정)와 스위칭 gain이 갈라지는 곳이다.",
                 not_yet="off 비율은 이상 다이오드 기준이다. 실제 정류기 용량과 SR 타이밍은 off 구간의 공진과 길이를 바꾼다.")
    res.add_plot("p_edge", "1차 edge 전류: 음수여야 ZVS 방향", ["ie"], x_label="F", x_unit="", y_label="i_r(0)", y_unit="A", kind="xy", level="C", hlines=[{"y": 0.0, "label": "0"}, {"y": -2 * _qoss(v) / v["td"], "label": "전하 screen 최소 전류"}],
                 proved="edge 전류 부호와 크기를 주파수별로 보였다.", not_yet="SCREEN_ONLY — 실제 ZVS는 노드 용량 모델(EX02) 필요.")
    res.tables.append(Table("t_curve", "주파수별 비교", ["F", "FHA |H|", "스위칭 gain", "차이", "정류 off", "edge 전류", "FHA 입력"], rows))
    diffs = [(float(Fs[i]), gt[i] / gf[i] - 1) for i in ok]
    near = [abs(d) for F, d in diffs if 0.9 - 1e-9 <= F <= 1.1 + 1e-9]
    if near:
        res.add_metric("max_diff_near", "f_r 부근(0.9–1.1) 최대 |차이|", max(near), "", basis="교재 5 % 초기 점검값과 비교")
    res.add_metric("max_diff", "sweep 전체 최대 |차이|", max(abs(d) for _, d in diffs) if diffs else float("nan"), "")
    ipk = int(np.argmax([gt[i] for i in ok])) if ok else None
    if ipk is not None:
        res.add_metric("F_peak_td", "스위칭 gain 최대 F (sweep 격자)", float(Fs[ok[ipk]]), "", basis="격자 해상도 이내; 범위 끝이면 실제 peak는 더 바깥")
    res.add_metric("F_peak_fha", "FHA gain 최대 F (sweep 격자)", float(Fs[int(np.argmax(gf))]), "")
    if failed:
        res.verdict("SOLVER_FAILED", f"주기해 실패 F = {failed}")
    res.verdict("PASS_WITHIN_MODEL", "sweep의 각 점은 정류기 포함 이상 스위칭 주기해")
    if near and max(near) > 0.05:
        res.verdict("MARGINAL", "f_r 부근에서도 FHA 차이가 5 %를 넘는다")
    if diffs and max(abs(d) for _, d in diffs) > 0.05:
        res.verdict("INFO", "f_r에서 먼 점의 5 % 초과 차이는 FHA 적용범위 밖이라는 결과다 (보정하지 않음)")
    res.verdict("SCREEN_ONLY", "edge 전류는 부호·전하 screen만")
    res.assumptions += ["각 점: 물리 12주기 + Newton shooting 주기해", "이상 스위치·다이오드, 합성 tank"]
    res.not_valid_for += ["실제 gain peak 위치(기생 용량·dead time 영향)", "제어 설계(동특성 없음)"]
    res.interpretation = "FHA는 f_r 부근 중부하에서 몇 % 이내로 맞지만, f_r 아래(off 구간이 생기는 영역)에서는 스위칭 gain이 더 크고 peak도 더 낮은 주파수로 간다. 같은 gain이라도 어느 영역인지에 따라 정류 모드·ZVS 전류가 다르다."
    return res


def _qoss(v) -> float:
    C0, V0 = v["C0"], v["V0"]
    return 2.0 * C0 * V0 * (math.sqrt(1.0 + v["Vin"] / V0) - 1.0)


# ======================================================================================
# 4. ZVS charge vs Lm and load
# ======================================================================================


def run_zvs(v: dict) -> Result:
    res = Result("FL09", "zvs_lm", "C + SCREEN")
    Qs = [float(q) for q in np.geomspace(v["Q_lo"], v["Q_hi"], int(v["points"]))]
    Lms = [v["Lm_a"], v["Lm_b"], v["Lm_c"]]
    i_min = 2 * _qoss(v) / v["td"]
    rows = []
    for Lm in Lms:
        ie, ir = [], []
        for q in Qs:
            tank, sysl, n, kb, Rac, RL, per, s = _solve(v, Q=q, Lm=Lm)
            if s is None:
                ie.append(None)
                ir.append(None)
                continue
            e = edge_currents(sysl, per.traj, sysl.T)["i_rise"]
            zs = zvs_screen(e, v["Vin"], v["td"], v["C0"], v["V0"])
            ind = fha(tank, v["F"] * tank.fr1, Rac).inductive
            ie.append(abs(e) if e < 0 else -abs(e))
            ir.append(s["I1_rms"])
            rows.append([f"{Lm * 1e6:g} µH", f"{q:.3g}", f"{s['P_load']:.0f} W", f"{e:+.3f} A", "inductive" if ind else "capacitive", ZVS_KO[zs["status"]], f"{s['I1_rms']:.3f} A"])
        res.add_series(f"ie_{Lm * 1e6:g}", f"L_m = {Lm * 1e6:g} µH", "A", Qs, ie, style="line")
        res.add_series(f"ir_{Lm * 1e6:g}", f"L_m = {Lm * 1e6:g} µH", "A", Qs, ir, style="line")
    res.add_plot("p_ie", f"ZVS 방향 edge 전류 −i_r(0) vs 부하 (F = {v['F']:g})", [f"ie_{Lm * 1e6:g}" for Lm in Lms], x_label="Q (부하)", x_unit="", y_label="−i_r(0)", y_unit="A", kind="xy", log_x=True, level="C",
                 hlines=[{"y": i_min, "label": f"전하 screen 최소 {i_min:.2f} A"}, {"y": 0.0, "label": "0"}],
                 proved="경부하에서 edge 전류는 여자전류 peak(≈ nV_oT/(4L_m))로 수렴하고 부하와 거의 무관하다.", not_yet="SCREEN_ONLY: 전류를 edge 값으로 고정한 전하 비교.")
    res.add_plot("p_ir", "공진 전류 RMS vs 부하: L_m을 줄이면 순환전류 비용", [f"ir_{Lm * 1e6:g}" for Lm in Lms], x_label="Q (부하)", x_unit="", y_label="I_r,rms", y_unit="A", kind="xy", log_x=True, level="C", proved="L_m별로 같은 부하(Q) 범위의 공진 전류 RMS를 정류기 포함 주기해로 계산했다. L_m을 줄이면 경부하에서도 RMS가 크게 남는다(여자 순환전류).", not_yet="손실은 계산하지 않았다 (R_DS(on)·자성체 데이터 없음).")
    res.tables.append(Table("t_zvs", "L_m × 부하별 screen", ["L_m", "Q", "P", "edge 전류", "FHA 입력", "screen", "I_r,rms"], rows,
                            note=f"필요 전하 2Q_oss({v['Vin']:g} V) = {2 * _qoss(v) * 1e9:.1f} nC, t_d = {v['td'] * 1e9:g} ns → 최소 {i_min:.2f} A (합성값). FHA inductive와 screen 통과는 다른 판정이다."))
    res.verdict("SCREEN_ONLY", "ZVS는 edge 전류 부호·전하 screen (합성 C_oss, 일정 전류 가정)")
    res.verdict("MISSING_INPUT", "실제 소자 Q_oss(V)·dead time·SR 조건이 없어 ZVS 보증이나 손실 비교를 확정하지 않는다")
    res.assumptions += ["이상 스위칭 주기해에서 edge 전류를 읽고 전하 screen 적용", "합성 C_oss·t_d (ASSUMED)"]
    res.not_valid_for += ["ZVS 보증", "효율 최적 L_m"]
    res.interpretation = "경부하에서 ZVS 전하를 주는 것은 부하 전류가 아니라 여자전류다. L_m을 줄이면 전하 여유가 커지지만 모든 부하에서 순환전류(RMS)가 늘어 전도·자성체 손실 비용을 낸다. FHA의 inductive 판정은 부호의 필요조건일 뿐 전하 충분조건이 아니다."
    return res


# ======================================================================================
# Content
# ======================================================================================

_Q = [
    Question(
        "LLC의 R_ac와 bridge 계수는?",
        "정류기 입력을 ±n·V_o 사각파(기본파 RMS 2√2nV_o/π), 전류를 정현파(정류 평균 2I_pk/π)로 두고 전력을 맞추면 R_ac = 8n²R_dc/π²이다. full bridge 전압 기본파는 4V_in/π, half bridge는 2V_in/π라서 V_o/V_in ≈ |H|/n (FB), |H|/(2n) (HB).",
        "What are R_ac and the bridge factor of an LLC?",
        "Treat the rectifier input as a square wave of plus or minus n V_o and the current as sinusoidal, then match power: R_ac equals eight n squared R_dc over pi squared. The full-bridge fundamental is 4 V_in over pi and the half-bridge 2 V_in over pi, so the gain is |H| over n or |H| over 2n.",
        ["8n²R_dc/π²", "기본파 전력 일치", "HB 1/2"],
    ),
    Question(
        "LLC에서 Q가 크면 경부하인가?",
        "정의에 달렸다. 이 교재의 Q = Z₀/R_ac에서는 R_ac가 작을수록(무거운 부하) Q가 크다. reciprocal Q를 쓰는 문헌과 섞으면 결론이 뒤집힌다.",
        "Does a large Q mean light load?",
        "It depends on the definition. Here Q is Z0 over R_ac, so a heavier load, meaning a smaller R_ac, gives a larger Q. Mixing it with texts that use the reciprocal definition flips the conclusion.",
        ["정의 확인", "Z₀/R_ac면 heavy"],
    ),
    Question(
        "LLC에서 f_r이면 V_in = V_out인가?",
        "아니다. f_r에서 |H| = 1이라는 것은 1차 환산 기본파 비율이다. 실제 출력은 권선비 n과 bridge 계수로 정해지고(FB V_in/n, HB V_in/(2n)), 손실·dead time이 있으면 조금 달라진다.",
        "At the resonant frequency, is V_out equal to V_in?",
        "No. |H| equals one only as a primary-referred fundamental ratio. The actual output also depends on the turns ratio and the bridge factor, V_in over n for a full bridge and over 2n for a half bridge, and shifts slightly with losses and dead time.",
        ["|H| = 1은 환산비", "n·bridge 계수"],
    ),
    Question(
        "L_m을 늘리면 좋은가?",
        "여자전류·순환 RMS와 전도/자성체 손실은 줄지만, 경부하 ZVS 전하(≈ 여자전류 peak × dead time)가 줄고 f_r 아래 gain 상승 여유도 줄어든다. 전하 screen과 gain 범위를 같이 본다.",
        "Is a larger L_m always better?",
        "It lowers magnetizing and circulating current, so conduction and core losses drop. But the light-load ZVS charge, roughly the magnetizing peak times the dead time, also drops, and so does the gain boost below resonance. Check the charge screen and the gain range together.",
        ["여자전류 감소", "ZVS 전하 감소", "gain 범위"],
    ),
    Question(
        "FHA가 실패하는 조건 세 가지는?",
        "① f_r 아래 불연속 정류(off 구간) ② 경부하·기동처럼 정류 전류가 정현파가 아닐 때 ③ 출력이 강한 전압원이라 작은 위상 차가 큰 전력 차가 될 때(CLLC 배터리). 그 밖에 dead time·C_oss·SR 타이밍.",
        "Name three conditions where FHA fails.",
        "First, discontinuous rectifier conduction below resonance. Second, light load or start-up, where the rectifier current is far from sinusoidal. Third, a stiff output voltage such as a battery, where a small phase error turns into a large power error. Dead time, Coss and SR timing add more.",
        ["불연속 정류", "비정현 전류", "강한 전압원"],
    ),
]

_zvs_params = [
    Param("td", "dead time t_d (screen)", "s", 100e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", group="screen"),
    Param("C0", "합성 C_oss C₀", "F", COSS_HV[0], "nF", vmin=1e-12, vmax=1e-6, source="ASSUMED", group="screen"),
    Param("V0", "합성 C_oss V₀", "V", COSS_HV[1], "V", vmin=0.1, vmax=1e4, source="ASSUMED", group="screen"),
]

_sw_params = [
    Param("Q", "부하 Q = Z₀/R_ac", "", 0.8, "", vmin=0.01, vmax=10, source="TEXTBOOK", source_note="0.2 / 0.8 / 1.5"),
    Param("Co", "출력 C_o", "F", 200e-6, "µF", vmin=1e-7, vmax=1.0, source="ASSUMED", source_note="리플을 작게 두는 합성값", group="출력"),
    Param("R1", "직렬 저항 R₁ (결합 손실)", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="비이상"),
]

EXPERIMENTS = [
    Experiment(
        key="fha_gain",
        title="FHA gain 곡선: Q·k·bridge 계수를 식으로 확인",
        goal="L_r 40 µH, C_r 28.1448 nF, L_m 200 µH (f_r = 150 kHz, k = 5)의 |H|와 ∠Z_in을 F = 0.7–1.5, Q = 0.2/0.8/1.5에서 그리고, 교재 정규화식·절점해석·FHA 회로 시간영역 해가 같은 값을 주는지 확인한다. 400→48 V에서 n(FB 8.3333, HB 4.1667)을 구한다.",
        params=_tank_params()
        + _power_params()
        + [
            Param("Q", "강조할 Q", "", 0.8, "", vmin=0.01, vmax=10, source="TEXTBOOK"),
            Param("F_min", "F 최소", "", 0.7, "", vmin=0.2, vmax=1.0, source="TEXTBOOK", group="범위"),
            Param("F_max", "F 최대", "", 1.5, "", vmin=1.0, vmax=4.0, source="TEXTBOOK", group="범위"),
        ],
        presets=[
            Preset("textbook", "교재: F 0.7–1.5, Q 0.2/0.8/1.5", {}, "", ("nominal", "reference")),
            Preset("hb", "half bridge 400→48 V", {"bridge": "HB"}, "n = 4.1667", ("variant", "reference")),
            Preset("low_k", "L_m 100 µH (k = 2.5)", {"Lm": 100e-6}, "gain 상승 여유 ↑, 여자전류 ↑", ("variant",)),
        ],
        run=run_fha,
        model_level="A",
        suggested_change="강조할 Q를 0.8 → 1.5 (더 무거운 부하)로 바꿔 F = 0.7의 gain을 본다.",
        prediction=Prediction(
            "Q = 1.5(무거운 부하)에서 F = 0.7의 FHA gain은?",
            ["1보다 크다 (f_r 아래라서)", "1보다 작다", "정확히 1", "모르겠다"],
            "1보다 작다",
            "무거운 부하의 작은 R_ac가 L_m과 병렬로 붙어 L_m의 공진 참여를 누른다. 그래서 f_r 아래에서도 gain이 0.741로 떨어지고, 그 영역은 capacitive라 ZVS까지 잃는다.",
            ["g_fha"],
            handcalc=[{"key": "fr", "label": "f_r", "unit": "Hz"}, {"key": "Z0", "label": "Z₀", "unit": "Ω"}, {"key": "Rac", "label": "R_ac (Q = 0.8)", "unit": "Ω"}],
        ),
        suggested={"Q": 1.5},
        student="사각파가 L_r–C_r–L_m tank를 두드리면 주파수에 따라 tank의 임피던스와 전압 분배가 바뀌어 출력이 바뀐다. f_r에서는 직렬 L_r–C_r이 사라져 입력이 그대로 전달된다.",
        expert="FHA는 R_ac로 정류기를 대신한 정현파 회로다. 세 가지 계산이 일치해도 그것은 FHA 대수 검산일 뿐 스위칭 검증이 아니다. Q 정의(Z₀/R_ac)와 FB/HB 계수를 섞지 않는다. inductive(∠Z_in > 0)는 ZVS의 필요조건일 뿐이다.",
        customer_ko="이 tank는 f_r = 150 kHz에서 부하와 무관하게 gain 1이고, 무거운 부하에서는 f_r 아래로 내려도 gain이 거의 오르지 않습니다. 필요한 gain 범위를 부하별로 먼저 확인하고, 그다음 실제 정류 파형으로 검증하시죠.",
        customer_en="This tank has unity gain at 150 kHz regardless of load, and at heavy load lowering the frequency below resonance adds almost no gain. Let's first check the required gain range per load point, then verify with the actual rectifier waveforms.",
        questions=_Q[:3],
        circuit="llc",
        textbook=[TB_12],
        reference_presets=["textbook", "hb"],
        claim_limit="FHA 대수. 정류 switching·ZVS·동특성 주장 없음.",
    ),
    Experiment(
        key="switching_vs_fha",
        title="정류기가 있는 스위칭 회로 vs FHA: 어디서 맞고 어디서 틀리나",
        goal="full-bridge 다이오드 정류·출력 C/R을 가진 이상 스위칭 LLC의 주기해를 구해 FHA gain과 비교한다. f_r 부근 중부하(차이 ≤ 5 %)와 f_r 아래 무거운 부하(정류 off 구간 → 큰 차이)의 이유를 파형으로 설명한다.",
        params=_tank_params() + _power_params() + _sw_params + [Param("F", "주파수 비 F = f/f_r", "", 0.9, "", vmin=0.3, vmax=3.0, source="TEXTBOOK", source_note="F 0.7–1.5")] + _zvs_params,
        presets=[
            Preset("mid", "중부하 Q 0.8, F 0.9", {}, "f_r 부근 — 5 % 점검값 적용", ("nominal", "reference")),
            Preset("at_fr", "Q 0.8, F 1.0", {"F": 1.0}, "f_r: 거의 일치", ("variant", "reference")),
            Preset("below_heavy", "Q 0.8, F 0.7", {"F": 0.7}, "정류 off 구간 → FHA 부적합", ("failure", "reference")),
            Preset("above_light", "Q 0.2, F 1.5", {"Q": 0.2, "F": 1.5}, "경부하 f_r 위", ("corner", "reference")),
            Preset("hb", "half bridge, Q 0.8, F 0.9", {"bridge": "HB"}, "n = 4.1667", ("variant", "reference")),
        ],
        run=run_switching,
        model_level="C + A",
        suggested_change="F를 0.9 → 0.7로 낮춘다 (Q 0.8 그대로).",
        prediction=Prediction(
            "F = 0.7로 내리면 스위칭 해의 gain은 FHA보다?",
            ["FHA와 거의 같다 (±2 %)", "FHA보다 훨씬 크다", "FHA보다 작다", "모르겠다"],
            "FHA보다 훨씬 크다",
            "f_r 아래에서는 정류 전류가 반주기 전에 0이 되어 off 구간(주기의 약 33 %)이 생긴다. 그 동안의 L_r+L_m 공진이 C_r에 에너지를 더 쌓아 다음 반주기의 출력이 커지는데, FHA는 이 모드를 모른다 (약 +24 %).",
            ["g_td", "g_fha", "diff", "off_frac"],
            handcalc=[{"key": "vo_fha", "label": "FHA V_o (F 0.9, Q 0.8)", "unit": "V"}, {"key": "Im_pk", "label": "여자전류 peak ≈ nV_o/(4L_mf)", "unit": "A"}],
        ),
        suggested={"F": 0.7},
        student="정류기가 도통하는 동안 변압기 전압은 출력 전압으로 고정된다. 정류 전류(공진 전류 − 여자전류)가 0이 되면 다이오드가 꺼지고, 그때부터 회로가 달라진다.",
        expert="off 구간은 i₂ = 0 제약과 떠 있는 정류기 입력 노드로 풀어야 한다(임의 저항으로 대신하지 않는다). 5 % 점검값은 f_r 부근 중부하에만 쓰고, 멀리서의 차이는 적용범위의 결과로 기록한다. FHA 저항 회로를 ‘switching verified’로 부르지 않는다.",
        customer_ko="FHA로는 이 운전점의 출력이 맞지만, f_r 아래 무거운 부하에서는 실제 정류 파형 기준으로 gain이 20 % 이상 달라집니다. 설계 범위의 모서리 점들은 스위칭 모델이나 실측으로 확인하시죠.",
        customer_en="FHA predicts this operating point well, but below resonance at heavy load the gain from the actual rectifier waveforms differs by more than twenty percent. Let's confirm the corner points of the design range with a switching model or a measurement.",
        questions=_Q[3:],
        circuit="llc",
        textbook=[TB_12, TB_19],
        reference_presets=["mid", "at_fr", "below_heavy", "above_light", "hb"],
        claim_limit="이상 스위칭·다이오드 정류의 정상상태. ZVS·손실·기동 주장 없음.",
    ),
    Experiment(
        key="gain_curve",
        title="스위칭 gain 곡선 전체: FHA peak와 실제 peak는 다른 곳에 있다",
        goal="F 0.6–1.6에서 정류기 포함 주기해를 반복해 스위칭 gain 곡선을 그리고 FHA와 겹친다. 정류 off 비율과 edge 전류 부호가 주파수에 따라 어떻게 바뀌는지 본다.",
        params=_tank_params() + _power_params() + _sw_params + [
            Param("F_min", "F 최소", "", 0.6, "", vmin=0.3, vmax=1.0, source="ASSUMED", group="sweep"),
            Param("F_max", "F 최대", "", 1.6, "", vmin=1.0, vmax=3.0, source="ASSUMED", group="sweep"),
            Param("points", "점 개수", "", 26, "", vmin=5, vmax=81, kind="int", source="ASSUMED", group="sweep"),
        ] + _zvs_params,
        presets=[
            Preset("mid", "Q 0.8", {}, "", ("nominal", "reference")),
            Preset("light", "Q 0.2", {"Q": 0.2}, "경부하", ("variant",)),
            Preset("heavy", "Q 1.5", {"Q": 1.5}, "과부하 쪽", ("corner",)),
        ],
        run=run_curve,
        model_level="C + A",
        suggested_change="Q를 0.8 → 0.2로 낮춘다.",
        prediction=Prediction(
            "Q 0.2(경부하)에서 f_r 위(F 1.5)의 스위칭 gain은 FHA보다?",
            ["약간 크다", "약간 작다", "같다", "모르겠다"],
            "약간 작다",
            "f_r 위에서는 정류기가 연속 도통하지만 사각파 고조파 때문에 스위칭 gain이 FHA보다 몇 % 낮다 (이 합성 tank에서 약 −5 %). 기생 용량을 넣은 실제 회로의 경부하 상승과는 다른 효과이므로 이상 모델의 범위를 기억한다.",
            ["max_diff"],
        ),
        suggested={"Q": 0.2},
        student="같은 tank라도 부하와 주파수에 따라 정류 모드가 바뀌어 출력이 FHA와 다르게 움직인다.",
        expert="스위칭 gain peak는 FHA peak보다 낮은 주파수·높은 값에 있다(off 구간 때문). 제어 범위·branch를 FHA만으로 정하면 f_r 아래 운전점의 gain과 ZVS 판단이 틀어진다.",
        customer_ko="FHA gain 곡선만으로 주파수 범위를 정하기보다, 범위 양 끝과 f_r 아래 점을 스위칭 모델로 확인하겠습니다. 이상 모델 기준으로도 f_r 아래에서는 FHA와 수십 % 차이가 납니다.",
        customer_en="Rather than set the frequency range from the FHA gain curve alone, I would check both ends of the range and the points below resonance with a switching model. Even in the ideal model the gain below resonance differs from FHA by tens of percent.",
        questions=[],
        circuit="llc",
        textbook=[TB_12, TB_19],
        reference_presets=["mid"],
        runtime_hint="seconds",
        claim_limit="이상 스위칭 주기해의 gain·모드. 손실·기생 용량 없음.",
    ),
    Experiment(
        key="zvs_lm",
        title="L_m tradeoff: 경부하 ZVS 전하 vs 순환전류",
        goal="F = 1.1(f_r 위)에서 부하 Q를 0.05–1.5로 바꾸며 L_m 100/200/400 µH의 edge 전류와 공진 RMS를 비교하고, 합성 C_oss·dead time으로 전하 screen을 적용한다. FHA inductive와 전하 충분이 다른 판정임을 확인한다.",
        params=_tank_params() + _power_params() + [
            Param("F", "주파수 비 F", "", 1.1, "", vmin=0.3, vmax=3.0, source="ASSUMED"),
            Param("Co", "출력 C_o", "F", 200e-6, "µF", vmin=1e-7, vmax=1.0, source="ASSUMED", group="출력"),
            Param("Lm_a", "L_m 후보 1", "H", 100e-6, "µH", vmin=1e-6, vmax=1.0, source="ASSUMED", group="L_m 후보"),
            Param("Lm_b", "L_m 후보 2", "H", 200e-6, "µH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", group="L_m 후보"),
            Param("Lm_c", "L_m 후보 3", "H", 400e-6, "µH", vmin=1e-6, vmax=1.0, source="ASSUMED", group="L_m 후보"),
            Param("Q_lo", "Q 최소 (경부하)", "", 0.05, "", vmin=0.005, vmax=1.0, source="ASSUMED", group="sweep"),
            Param("Q_hi", "Q 최대", "", 1.5, "", vmin=0.1, vmax=10, source="ASSUMED", group="sweep"),
            Param("points", "Q 점 개수", "", 7, "", vmin=3, vmax=25, kind="int", source="ASSUMED", group="sweep"),
        ] + _zvs_params,
        presets=[
            Preset("textbook", "F 1.1, t_d 100 ns", {}, "", ("nominal", "reference")),
            Preset("short_td", "t_d 50 ns", {"td": 50e-9}, "전하 여유 감소", ("variant",)),
        ],
        run=run_zvs,
        model_level="C + SCREEN",
        suggested_change="dead time을 100 → 50 ns로 줄인다.",
        prediction=Prediction(
            "dead time을 절반으로 줄이면 경부하에서 전하 screen을 통과하는 L_m 후보는?",
            ["늘어난다", "줄어든다", "그대로", "모르겠다"],
            "줄어든다",
            "가용 전하 |i|·t_d가 절반이 된다. 경부하 edge 전류는 여자전류 peak로 정해지므로, L_m이 큰 후보부터 필요 전하 2Q_oss를 채우지 못한다.",
            [],
        ),
        suggested={"td": 50e-9},
        student="스위치가 켜지기 전에 노드 전압을 옮겨 줄 전류가 필요하다. 경부하에서는 그 전류를 여자전류가 준다.",
        expert="ZVS 판정은 부호(inductive)·전하(|i|·t_d ≥ 2Q_oss)·turn-on 직전 V_DS 순서로 닫는다. L_m을 줄이는 대가는 모든 부하에서의 순환 RMS다. SR 타이밍과 버스트 모드는 이 모델 밖이다.",
        customer_ko="경부하 ZVS를 위해 L_m을 줄이는 안은 전하 여유를 주지만 전 부하에서 순환전류가 늘어납니다. 실제 소자 Q_oss와 dead time으로 turn-on 직전 V_DS를 확인한 뒤 L_m을 정하시죠.",
        customer_en="Reducing L_m for light-load ZVS gives more charge margin but raises the circulating current at every load. Let's confirm V_DS just before turn-on with the real device Qoss and dead time before fixing L_m.",
        questions=_Q[3:4],
        circuit="llc",
        textbook=[TB_12, TB_E02],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="edge 전류 부호·전하 screen. ZVS 보증·손실 없음.",
    ),
]

LAB = Lab(
    id="FL09",
    title="LLC — 공진을 말로 설명하고 식으로 확인하기",
    title_en="LLC: resonance in words, then in equations",
    track="basic",
    order=9,
    path_note="14일 경로 9일차",
    textbook=[TB_12, TB_19],
    prerequisites=["FL01", "FL07"],
    summary="FHA(R_ac·Q·k·FB/HB) 세 경로 검산 → 정류기 포함 스위칭 해와 비교 → 전체 gain 곡선 → L_m·ZVS 전하 tradeoff.",
    experiments=EXPERIMENTS,
    minimum_scope="LLC FHA·rectifier switching; R_ac·HB/FB·gain/phase·FHA mismatch (교재 19장)",
    claim_limits=["FHA 회로는 switching 검증이 아니다", "이상 스위치: ZVS는 NOT_EVALUABLE, 전하는 SCREEN_ONLY", "손실·온도 주장 없음"],
    test_paths=["tests/test_fl09.py"],
)
