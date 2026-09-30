"""FL05 - OBC and PFC: power quality as a design variable (textbook ch.08, ch.02).

Levels, kept apart on purpose:
  A  grid input boundary: line current, power at a current limit, low line, phase vs line-to-line
     voltage, modulation index corners; checked against a time-domain three-phase power path.
  A  PF, displacement and THD definitions on synthetic waveforms (window, H, DC offset, voltage
     distortion), DFT vs the constructed amplitudes.
  C  single-phase boost PFC switching model on the exact engine: diode bridge + boost + DC link +
     resistive load, the grid as an exact LTI oscillator state (the bridge polarity changes when
     that state crosses zero), average current-mode control sampled at the carrier valley (sampled
     current PI with |v_g| feedforward, slow DC-link voltage PI, one-period computation delay).
     THD and PF come from the simulated current with the textbook definitions (H = 40, integer
     number of line cycles in steady state).  Compared with an averaged model (B) and with a
     forced ideal sine whose THD = 0 by construction (never a PFC performance claim).
  A/B instantaneous power of single-phase vs balanced three-phase, DC-link sizing by 2-omega ripple
     vs by hold-up energy (separate problems), a nonlinear energy ODE as the independent path.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..engine.switched import AffineMode, Guard, HybridSystem, Trajectory, segment_moments, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import pfc as ref
from ._common import decimate_minmax

TB_08 = TextbookRef("obcpfc-전력품질을-설계-변수로-바꾸기-fl05", "08. OBC·PFC — 전력품질을 설계 변수로 바꾸기 [FL05]")
TB_02 = TextbookRef("전력변환을-다시-배우는-네-개의-법칙", "02. 전력변환을 다시 배우는 네 개의 법칙")
TB_09 = TextbookRef("제어-pi보다-plant와-부호가-먼저다-fl06", "09. 제어 — PI보다 plant와 부호가 먼저다 [FL06]")

SQ3 = math.sqrt(3.0)
SQ2 = math.sqrt(2.0)


# ======================================================================================
# Experiment 1: grid input boundary (A)
# ======================================================================================


def three_phase_power_td(V_LL: float, I_rms: float, cos_phi1: float, thd: float = 0.0, f: float = 50.0, n: int = 3600) -> tuple[float, float, float]:
    """Grid power, phase-current RMS and PF from sampled balanced three-phase waveforms.

    Independent of the sqrt(3) closed form: v_x, i_x are built phase by phase, p = sum v_x i_x is
    averaged over one period with the uniform rule (exact for trigonometric polynomials of degree
    < n/2).  Distortion, if any, is split equally between the 5th and 7th harmonic.
    """
    Vpk = V_LL / SQ3 * SQ2
    phi = math.acos(cos_phi1)
    a1 = I_rms / math.sqrt(1.0 + thd * thd)  # fundamental RMS
    ah = a1 * thd / SQ2  # each of h = 5, 7
    t = np.arange(n) / (n * f)
    w = 2 * math.pi * f
    p = np.zeros(n)
    i2 = np.zeros(n)
    for k in range(3):
        sh = -2 * math.pi * k / 3
        v = Vpk * np.sin(w * t + sh)
        i = SQ2 * a1 * np.sin(w * t + sh - phi) + SQ2 * ah * (np.sin(5 * (w * t + sh)) + np.sin(7 * (w * t + sh)))
        p += v * i
        if k == 0:
            i2 = i * i
    P = float(np.mean(p))
    Irms = float(math.sqrt(np.mean(i2)))
    return P, Irms, P / (3 * V_LL / SQ3 * Irms)


def run_grid_boundary(v: dict) -> Result:
    res = Result("FL05", "grid_boundary", "A (+ 시간영역 3상 전력 경로)")
    P, V, eta, PF, Ilim = v["P_bat"], v["V_LL"], v["eta"], v["PF"], v["I_lim"]
    I_line = P / (SQ3 * V * eta * PF)
    P_lim = SQ3 * V * Ilim * eta * PF
    I_low = P / (SQ3 * v["V_LL_low"] * eta * PF)
    V_ph = V / SQ3
    tb = abs(P - 11e3) < 1e-6 and abs(V - 400) < 1e-9 and abs(eta - 0.97) < 1e-12 and abs(PF - 0.995) < 1e-12
    res.add_metric("I_line", "선전류 RMS (11 kW 배터리 출력)", I_line, "A", ref=16.45043 if tb else ref.line_current(P, V, eta, PF), ref_label="교재 16.45043 A" if tb else "P/(√3·V_LL·η·PF)", tol=1e-6, basis="상전류 = 선전류 (Y/Δ 무관한 3상 입력 선전류), RMS")
    res.add_metric("P_at_lim", f"{Ilim:g} A 한계에서 가능한 배터리 출력", P_lim, "W", ref=10698.81 if (tb and abs(Ilim - 16) < 1e-9) else ref.battery_power(Ilim, V, eta, PF), ref_label="교재 10.69881 kW" if (tb and abs(Ilim - 16) < 1e-9) else "√3·V·I·η·PF", tol=1e-6, basis="배터리 측 출력 (계통 입력 아님)")
    res.add_metric("I_low", f"low line {v['V_LL_low']:g} V에서 같은 출력에 필요한 선전류", I_low, "A", ref=18.278 if (tb and abs(v["V_LL_low"] - 360) < 1e-9) else ref.line_current(P, v["V_LL_low"], eta, PF), ref_label="교재 18.278 A" if (tb and abs(v["V_LL_low"] - 360) < 1e-9) else "P/(√3·V_low·η·PF)", tol=2e-5)
    V_need = P / (SQ3 * Ilim * eta * PF)
    res.add_metric("V_need", f"{P / 1e3:g} kW를 {Ilim:g} A 안에서 내는 데 필요한 최소 선간전압", V_need, "V", basis="이 전압보다 낮으면 출력과 전류 한계를 동시에 만족할 수 없다")
    res.add_metric("P_grid", "계통 입력 유효전력 P_bat/η", P / eta, "W")
    res.add_metric("V_ph", "상전압 RMS V_LL/√3", V_ph, "V", ref=230.94 if abs(V - 400) < 1e-9 else V / SQ3, ref_label="교재 230.94 V" if abs(V - 400) < 1e-9 else "V_LL/√3", tol=2e-5, basis="상전압 (Y 등가), RMS; 선간전압과 구분")
    # modulation index corners
    m_nom = 2 * SQ2 * (V / SQ3) / v["V_dc"]
    m_cor = 2 * SQ2 * (v["V_LL_hi"] / SQ3) / v["V_dc_lo"]
    tbm = abs(V - 400) < 1e-9 and abs(v["V_dc"] - 800) < 1e-9
    tbc = abs(v["V_LL_hi"] - 440) < 1e-9 and abs(v["V_dc_lo"] - 700) < 1e-9
    res.add_metric("m_nom", f"SPWM 변조지수 ({V:g} V LL / {v['V_dc']:g} V DC)", m_nom, "", ref=0.8165 if tbm else ref.spwm_index(V / SQ3, v["V_dc"]), ref_label="교재 0.8165" if tbm else "2√2·V_ph/V_dc", tol=5e-5, basis="m = V_ph,peak/(V_dc/2)")
    res.add_metric("m_cor", f"SPWM 변조지수 ({v['V_LL_hi']:g} V LL / {v['V_dc_lo']:g} V DC)", m_cor, "", ref=1.0265 if tbc else ref.spwm_index(v["V_LL_hi"] / SQ3, v["V_dc_lo"]), ref_label="교재 1.0265" if tbc else "2√2·V_ph/V_dc", tol=5e-5, basis="고전압·저 DC 코너")
    # converter voltage including the boost inductor drop at the corner current (unity PF, fundamental)
    w = 2 * math.pi * v["f"]
    I_cor = P / (SQ3 * v["V_LL_hi"] * eta * PF)
    Vc = math.hypot(v["V_LL_hi"] / SQ3 * SQ2, w * v["L_g"] * I_cor * SQ2)
    m_req = Vc / (v["V_dc_lo"] / 2)
    lim = 1.0 if v["mod"] == "spwm" else ref.SVPWM_LIMIT
    margin = (lim - m_req) / lim
    res.add_metric("m_req", "필요 변조지수 (L_g 전압강하 포함, 코너)", m_req, "", basis=f"|V_c| = √(V_ph,pk² + (ωL_g I_pk)²), I = {I_cor:.4g} A")
    res.add_metric("m_lim", "선형 변조 한계", lim, "", basis="SPWM 1, SVPWM(이상) 2/√3 = 1.1547 (같은 정규화)")
    res.add_metric("m_margin", "코너의 남은 선형 여유", margin * 100, "%", basis="동적 전류 조절·과도·dead time 전압손실은 이 여유에서 나와야 한다")
    # independent time-domain path: solve for the current with p(t) = sum v_x i_x
    Pg = P / eta

    def f_disp(I):
        return three_phase_power_td(V, I, PF)[0] - Pg

    I_td = brentq(f_disp, 1e-6, 10 * I_line + 1, xtol=1e-12, rtol=1e-14)
    thd_eq = math.sqrt(1 / PF**2 - 1)

    def f_dist(I):
        return three_phase_power_td(V, I, 1.0, thd_eq)[0] - Pg

    I_td2 = brentq(f_dist, 1e-6, 10 * I_line + 1, xtol=1e-12, rtol=1e-14)
    _, Irms2, pf2 = three_phase_power_td(V, I_td2, 1.0, thd_eq)
    res.add_check(check_close("선전류: 시간영역 Σv·i 경로 (PF를 변위각으로) vs √3 식", I_td, I_line, 1e-9, "상별 정현파 v_x·i_x를 한 주기 평균해 P_bat/η가 되는 전류를 brentq로 풀기", True, "A"))
    res.add_check(check_close("선전류: 시간영역 경로 (PF를 왜곡으로, cosφ₁ = 1, THD = √(1/PF²−1)) vs √3 식", I_td2, I_line, 1e-9, "5·7차 고조파로 PF 0.995를 만든 전류의 RMS", True, "A", detail=f"시간영역 PF = {pf2:.9f}, THD = {thd_eq * 100:.3f} %"))
    # sweeps
    Is = np.linspace(4.0, 32.0, 57)
    for VV in (v["V_LL_low"], V, v["V_LL_hi"]):
        res.add_series(f"pb_{int(VV)}", f"{VV:g} V LL", "W", Is.tolist(), [SQ3 * VV * x * eta * PF for x in Is])
    res.add_plot("p_pb", "전류 한계에 따른 가능한 배터리 출력", [f"pb_{int(VV)}" for VV in (v["V_LL_low"], V, v["V_LL_hi"])], x_label="계통 전류 한계 (RMS)", x_unit="A", y_label="배터리 출력", y_unit="W", kind="xy",
                 hlines=[{"y": P, "label": f"{P / 1e3:g} kW 목표"}], vlines=[{"x": Ilim, "label": f"{Ilim:g} A"}], level="A",
                 proved=f"{Ilim:g} A 한계에서 {V:g} V는 {P_lim / 1e3:.4g} kW만 가능하고, {P / 1e3:g} kW를 내려면 {I_line:.4g} A가 필요하다. 저전압일수록 같은 한계에서 출력이 준다.",
                 not_yet="η·PF를 운전점과 무관한 상수로 두었다. 실제 효율·PF는 전압·부하·온도에 따라 바뀐다.")
    Vs = np.linspace(330.0, 480.0, 61)
    res.add_series("i_need", f"{P / 1e3:g} kW에 필요한 선전류", "A", Vs.tolist(), [P / (SQ3 * x * eta * PF) for x in Vs])
    res.add_plot("p_iv", "계통 전압에 따른 필요 선전류", ["i_need"], x_label="선간전압 V_LL (RMS)", x_unit="V", y_label="선전류", y_unit="A", kind="xy",
                 hlines=[{"y": Ilim, "label": f"{Ilim:g} A 한계"}], vlines=[{"x": V_need, "label": f"{V_need:.4g} V"}, {"x": v["V_LL_low"], "label": "low line"}], level="A",
                 proved=f"{V_need:.4g} V 아래에서는 {P / 1e3:g} kW와 {Ilim:g} A가 동시에 성립하지 않는다 (η·PF 고정 가정).", not_yet="계통 임피던스에 의한 전압강하는 포함하지 않았다.")
    rows = []
    for VV in (v["V_LL_low"], V, v["V_LL_hi"]):
        for VD in (v["V_dc_lo"], v["V_dc"]):
            m = 2 * SQ2 * (VV / SQ3) / VD
            rows.append([f"{VV:g} V / {VD:g} V", m, "선형" if m <= 1 else "SPWM 과변조", "선형" if m <= ref.SVPWM_LIMIT else "SVPWM 과변조"])
    res.tables.append(Table("t_mod", "변조지수 코너 (m = 2√2·V_ph,rms / V_dc)", ["V_LL / V_dc", "m", "SPWM (≤ 1)", "SVPWM 이상 (≤ 1.1547)"], rows,
                            note="SVPWM의 이상 한계가 더 높아도 계통 인덕턴스 전압강하·동적 전류조절·과도·dead time 전압손실을 위한 여유가 필요하다."))
    res.tables.append(
        Table(
            "t_q",
            "먼저 물어볼 요구 (교재 FAE 답변)",
            ["질문", "왜"],
            [
                ["단상에서도 11 kW인가?", "단상 16 A·230 V는 약 3.7 kVA — 3상과 요구가 다르다"],
                ["V2G(양방향)가 필요한가?", "Vienna 같은 단방향 토폴로지는 그대로 선택 불가"],
                ["neutral 유무와 16 A 제한", "상전류 한계와 불평형 운전 가능성"],
                ["배터리 전압 범위와 DC-link 전략", "PFC 링크 800 V와 배터리 800 V급은 같은 노드가 아니다"],
            ],
        )
    )
    res.circuit = {"diagram": afe_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    if I_line > Ilim * (1 + 1e-12):
        res.verdict("CUSTOMER_DECISION_REQUIRED", f"{P / 1e3:g} kW에는 {I_line:.5g} A가 필요해 {Ilim:g} A 한계를 넘는다: 출력 보증과 전류 한계 중 무엇을 우선할지(derating 조건) 고객과 정해야 한다.")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"{P / 1e3:g} kW가 {Ilim:g} A 안에서 가능하다 (η·PF 고정 가정).")
    if m_req > lim:
        res.verdict("FAIL_CONSTRAINT", f"{v['V_LL_hi']:g} V LL / {v['V_dc_lo']:g} V DC 코너에서 필요 변조지수 {m_req:.4f} > {('SPWM 선형 한계 1' if v['mod'] == 'spwm' else 'SVPWM 이상 한계 1.1547')}: 선형 변조로 계통 전압을 만들 수 없다.")
        if v["mod"] == "spwm":
            res.verdict("OUT_OF_VALIDITY", "m > 1에서는 SPWM의 선형 관계(기본파 = m·V_dc/2)가 성립하지 않는다(과변조: 이득 비선형, 저차 고조파). 선형 모델 결과를 그 코너에 쓰지 않는다.")
    elif margin < v["headroom"]:
        res.verdict("MARGINAL", f"코너에서 선형 여유 {margin * 100:.3g} % < 학습용 동적 여유 {v['headroom'] * 100:g} %")
    res.assumptions += [
        "η와 PF는 운전점과 무관한 상수 (교재의 합성 조건)",
        "3상 평형, 선전류 = 상전류 RMS; 계통 임피던스 전압강하 없음",
        f"변조지수 코너의 L_g = {v['L_g'] * 1e3:g} mH는 ASSUMED (단위 역률 기본파 전압강하만)",
    ]
    res.not_valid_for += ["효율·PF의 부하/전압 의존성", "과변조 영역의 고조파·이득", "단상 운전의 입력 조건 (별도 계산 필요)"]
    res.interpretation = (
        f"배터리 출력 {P / 1e3:g} kW를 η = {eta:g}, PF = {PF:g}로 계통에서 가져오려면 계통 유효전력 {Pg / 1e3:.4g} kW와 피상전력이 필요하고, {V:g} V LL에서는 선전류 {I_line:.5g} A다. "
        f"그래서 ‘11 kW = 400 V × 16 A’는 정확한 출력 보증이 아니다: {Ilim:g} A 한계에서는 {P_lim / 1e3:.5g} kW, low line {v['V_LL_low']:g} V에서는 {I_low:.4g} A가 필요하다. "
        f"변조지수는 상전압 peak를 V_dc/2로 나눈 값이라 {v['V_LL_hi']:g} V / {v['V_dc_lo']:g} V 코너에서 {m_cor:.4f}가 되어 SPWM 선형영역을 벗어난다."
    )
    return res


def afe_circuit(v: dict) -> Circuit:
    c = Circuit("afe3", 700, 300, title="3상 계통 → PFC → DC-link (입력 경계)")
    g = c.add("vsource", "G", 90, 150, 90, "계통 3상", f"{v['V_LL']:g} V LL", lpos=(66, 146, "end"))
    ll = c.add("inductor", "LG", 230, 60, 0, "L_g (상당)", f"{v['L_g'] * 1e3:g} mH")
    c.add("block", "AFE", 380, 150, 0, "PFC (B6/Vienna/…)", w=150, h=150)
    cc = c.add("capacitor", "C", 540, 150, 90, "C_dc", f"{v['V_dc']:g} V")
    c.add("block", "DCDC", 640, 150, 0, "DC/DC → 배터리", w=90, h=60)
    c.wire("w1", g["a"], (90, 60), ll["a"])
    c.wire("w2", ll["b"], (305, 60))
    c.wire("w3", g["b"], (90, 240), (305, 240))
    c.wire("w4", (455, 100), (540, 100), cc["a"])
    c.wire("w5", (455, 200), (540, 200), cc["b"])
    c.wire("w6", (540, 100), (640, 100), (640, 120))
    c.wire("w7", (540, 200), (640, 200), (640, 180))
    c.probe("pI", "none", 160, 48, "right", "I_line (RMS)")
    c.text(380, 250, "m = V_ph,peak / (V_dc/2)", "note")
    return c


# ======================================================================================
# Experiment 2: PF, displacement and THD definitions (A)
# ======================================================================================


def harmonic_dft(x: np.ndarray, n_cyc_assumed: int, H: int) -> tuple[np.ndarray, float]:
    """Harmonic phasors (peak, complex) h = 1..H read at bins h * n_cyc_assumed of a uniform record, and the DC value."""
    N = x.size
    X = np.fft.rfft(x) / N
    ph = np.array([2 * X[h * n_cyc_assumed] if h * n_cyc_assumed < X.size else 0.0 for h in range(1, H + 1)])
    return ph, float(X[0].real)


def run_pf_thd(v: dict) -> Result:
    res = Result("FL05", "pf_thd", "A (합성 파형, DFT)")
    f, fa = v["f"], v["f_an"]
    w = 2 * math.pi * f
    V1, V5 = v["V1"], v["V1"] * v["v5"] / 100
    I1 = v["I1"]
    phi = math.radians(v["phi1"])
    amps = {3: v["h3"] / 100 * I1, 5: v["h5"] / 100 * I1, 7: v["h7"] / 100 * I1}
    idc = v["idc"]
    ncyc = int(v["n_cyc"])  # analyser window in cycles of its assumed frequency
    M = int(v["M"])
    Tw = ncyc / fa
    N = ncyc * M
    t = np.arange(N) * (Tw / N)
    vt = SQ2 * V1 * np.sin(w * t) + SQ2 * V5 * np.sin(5 * w * t)
    it = SQ2 * I1 * np.sin(w * t - phi) + idc
    for h, a in amps.items():
        it = it + SQ2 * a * np.sin(h * w * t - h * phi)  # harmonics phase-locked to the fundamental
    H = int(v["H"])
    Ih, I0 = harmonic_dft(it, ncyc, H)
    Vh, _ = harmonic_dft(vt, ncyc, H)
    I1m = abs(Ih[0]) / SQ2
    thd = math.sqrt(float(np.sum(np.abs(Ih[1:]) ** 2)) / 2) / I1m
    cosp = math.cos(np.angle(Vh[0]) - np.angle(Ih[0]))
    P = float(np.mean(vt * it))
    Vr = float(np.sqrt(np.mean(vt * vt)))
    Ir = float(np.sqrt(np.mean(it * it)))
    pf_true = P / (Vr * Ir)
    pf_formula = cosp / math.sqrt(1 + thd * thd)
    synced = abs(f - fa) < 1e-12
    # analytic values from the construction (valid for the synchronised integer window)
    thd_an = math.sqrt(sum(a * a for h, a in amps.items() if h <= H)) / I1
    I5 = amps[5]
    P_an = V1 * I1 * math.cos(phi) + V5 * I5 * math.cos(5 * phi)  # 5th-harmonic current lags by 5 phi
    Ir_an = math.sqrt(I1 * I1 + sum(a * a for a in amps.values()) + idc * idc)
    Vr_an = math.hypot(V1, V5)
    pf_an = P_an / (Vr_an * Ir_an)
    res.add_metric("thd", f"THD_i (H = {H}, 창 {ncyc}주기@{fa:g} Hz 가정)", thd * 100, "%", ref=thd_an * 100 if synced else None, ref_label="구성한 고조파 √Σa_h²/a_1", tol=1e-8, basis="DFT, 기본파 = 창의 1차 bin")
    res.add_metric("cos_phi1", "변위역률 cosφ₁ (기본파 위상차)", cosp, "", ref=math.cos(phi) if synced else None, ref_label="구성한 φ₁", tol=1e-9)
    res.add_metric("pf_true", "실제 PF = P/(V_rms·I_rms)", pf_true, "", ref=pf_an if synced else None, ref_label="해석 P/(V_rms I_rms)", tol=1e-9, basis="창 안 시간영역 평균")
    res.add_metric("pf_formula", "근사 PF = cosφ₁/√(1+THD²)", pf_formula, "", basis="전압 정현파·DC 성분 없음 조건의 식")
    res.add_metric("pf_err", "근사식 − 실제 PF", pf_formula - pf_true, "", note="DC offset·전압 왜곡이 있으면 0이 아니다")
    res.add_metric("I1", "기본파 RMS (DFT)", I1m, "A", ref=I1 if synced else None, ref_label="구성값", tol=1e-9)
    res.add_metric("I0", "DC 성분 (센서 offset)", I0, "A", basis="THD에는 안 들어가지만 I_rms와 PF를 바꾼다")
    res.add_metric("Irms", "전류 RMS", Ir, "A")
    res.add_metric("P", "유효전력", P, "W")
    if synced:
        res.add_check(check_close("THD: DFT (정수 창) vs 구성한 고조파", thd, thd_an, 1e-9, "rfft bin h·n vs √(Σa_h²)/a_1", True, "", abs_scale=1e-3))
        res.add_check(check_close("PF: 시간영역 평균 vs 해석식", pf_true, pf_an, 1e-9, "mean(v·i)/(√mean v²·√mean i²) vs 성분별 해석 적분", True, ""))
    # spectrum
    hh = list(range(1, H + 1))
    res.add_series("spec_i", "|I_h| / I_1 (DFT)", "%", hh, [max(abs(x) / abs(Ih[0]) * 100, 1e-6) for x in Ih], style="points")
    if synced:
        res.add_series("spec_an", "구성값", "%", [1] + sorted(k for k in amps if k <= H), [100.0] + [amps[k] / I1 * 100 for k in sorted(amps) if k <= H], style="points")
    res.add_plot("p_spec", "전류 스펙트럼 (창에서 읽은 고조파 bin)", ["spec_i"] + (["spec_an"] if synced else []), x_label="차수 h", x_unit="", y_label="I_h/I_1", y_unit="%", kind="xy", log_y=True, level="A",
                 proved="정수 주기 창에서는 DFT가 구성한 고조파를 정확히 읽는다. 창이 계통 주기와 맞지 않으면 기본파가 이웃 bin으로 새어(leakage) 고조파처럼 보인다.",
                 not_yet="합성 파형이다. 실제 측정의 센서 대역·앨리어싱 필터·PLL 동기화 오차는 모델에 없다. 스위칭 주파수 대역의 EMI를 대표하지 않는다.")
    t2 = t[t < 2 / f]
    res.add_series("v", "v (V/10)", "V", t2.tolist(), (vt[: t2.size] / 10).tolist())
    res.add_series("i", "i", "A", t2.tolist(), it[: t2.size].tolist())
    res.add_plot("p_wave", "전압(1/10)과 전류 파형", ["v", "i"], y_label="v/10, i", y_unit="", level="A",
                 proved="전류의 기본파 위상·고조파·DC offset이 파형에서 어떻게 보이는지 확인한다.", not_yet="")
    res.tables.append(
        Table(
            "t_win",
            "측정 조건 기록 (THD를 보고할 때 같이 적는 것)",
            ["항목", "값"],
            [
                ["창 길이", f"{Tw * 1e3:.6g} ms = 분석기 가정 {ncyc}주기 @ {fa:g} Hz (실제 {Tw * f:.4g}주기 @ {f:g} Hz)"],
                ["샘플", f"{N}점 ({M}/주기)"],
                ["기본파 추정", "창의 DFT 1차 bin (정수 창이면 최소자승 정현파 fit과 같음)"],
                ["고조파 상한 H", f"{H}"],
                ["DC 성분", f"{I0:.4g} A (THD 분모·분자에서 제외, I_rms에는 포함)"],
                ["정상상태", "합성 파형이라 기동 구간 없음"],
            ],
            note="교재 기준: H = 40, 정상상태 정수 10주기. 스위칭 주파수 영역의 EMI를 대표하지 않는다.",
        )
    )
    if not synced:
        res.verdict("NOT_EVALUABLE", f"분석 창이 실제 계통 {f:g} Hz의 정수 주기가 아니다({Tw * f:.4g}주기): leakage 때문에 이 DFT 값은 교재 정의의 THD로 보고할 수 없다 (동기화 필요).")
    else:
        res.verdict("PASS_WITHIN_MODEL", "정수 주기 창에서 DFT THD·PF가 구성값과 일치; 근사식은 조건(정현파 전압, DC 없음)에서만 맞는다.")
    if synced and (abs(idc) > 0 or V5 > 0):
        res.warnings.append("DC offset 또는 전압 왜곡이 있어 PF ≈ cosφ₁/√(1+THD²) 근사가 실제 PF와 다르다.")
    res.assumptions += ["합성 파형: 기본파 + 3·5·7차 전류 고조파 (+ 5차 전압, DC)", "샘플링은 이상 (양자화·앨리어싱 필터 없음)"]
    res.not_valid_for += ["IEC 61000-3-2/3-12 적합성 판정", "스위칭 주파수 대역의 전도 EMI", "실측 파형의 THD"]
    res.interpretation = (
        "PF는 실제 평균전력을 피상전력으로 나눈 값이고, 변위역률 cosφ₁은 기본파 위상차만 본다. 전압이 정현파이고 DC가 없으면 고조파 전류는 전력을 나르지 않고 RMS만 키우므로 "
        "PF = cosφ₁/√(1+THD²)가 된다. 센서 offset(DC)이나 전압 왜곡이 있으면 이 식은 근사일 뿐이다. THD 값은 창 길이·동기화·H와 함께 보고해야 하며, 창이 정수 주기가 아니면 leakage가 고조파처럼 보인다."
    )
    res.circuit = {"diagram": meter_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    return res


def meter_circuit(v: dict) -> Circuit:
    c = Circuit("meter", 620, 260, title="측정 경계: 계통 전압 v와 입력 전류 i")
    g = c.add("vsource", "G", 90, 130, 90, "v (계통)", f"{v['V1']:g} V RMS", lpos=(66, 126, "end"))
    c.add("block", "LOAD", 420, 130, 0, "비선형 입력단 (합성 전류원)", w=200, h=70)
    c.add("block", "AN", 260, 215, 0, "분석기: 창·H·동기화", w=190, h=34)
    c.wire("w1", g["a"], (90, 50), (420, 50), (420, 95))
    c.wire("w2", g["b"], (90, 190), (180, 190))
    c.wire("w3", (320, 190), (420, 190), (420, 165))
    c.probe("pi", "i", 200, 38, "right", "i")
    return c


# ======================================================================================
# Experiment 3: single-phase boost PFC on the exact engine (C)
# ======================================================================================


class BoostPFC(HybridSystem):
    """Diode bridge + boost + DC-link C + resistive load, grid as an exact oscillator state.

    x = [i_L, v_C, s, c]; v_g = V_pk s, ds/dt = w c, dc/dt = -w s.  Discrete state q = (sw, pol):
    sw in {ON, OFF, DCM}, pol = sign of v_g (which bridge diode pair conducts).  With ideal diodes
    and i_L > 0 the bridge output is |v_g| = pol V_pk s.  Guards (armed only when a crossing is
    possible, see ``arm_*``): s = 0 (bridge commutation), i_L = 0 in OFF (DCM entry), and
    pol V_pk s = v_C in DCM (natural rectifier conduction, never reached when V_pk < v_C).
    """

    state_names = ("i_L", "v_C", "s", "c")
    state_units = ("A", "V", "", "")

    def __init__(self, p: dict):
        self.p = p
        self.Vpk = p["Vpk"]
        self.w = 2 * math.pi * p["f"]
        self.L, self.C, self.R, self.RL = p["L"], p["C"], p["R"], p["RL"]
        self.T = 1.0 / p["fs"]
        self.d = 0.0
        self.arm_zc = True
        self.arm_dcm = True
        self._modes: dict = {}
        self._key = f"bpfc|{sorted((k, float(x)) for k, x in p.items() if isinstance(x, (int, float)))}"

    def mode(self, q) -> AffineMode:
        m = self._modes.get(q)
        if m is not None:
            return m
        sw, pol = q
        A = np.zeros((4, 4))
        A[2, 3] = self.w
        A[3, 2] = -self.w
        A[1, 1] = -1.0 / (self.R * self.C)
        if sw in ("ON", "OFF"):
            A[0, 2] = pol * self.Vpk / self.L
            A[0, 0] = -self.RL / self.L
        if sw == "OFF":
            A[0, 1] = -1.0 / self.L
            A[1, 0] = 1.0 / self.C
        m = AffineMode(f"{self._key}|{sw}|{pol}", A, np.zeros(4), label=self.describe(q))
        self._modes[q] = m
        return m

    def guards(self, q):
        sw, pol = q
        g = []
        if self.arm_zc:
            g.append(Guard("v_g = 0 (bridge 전환)", np.array([0.0, 0.0, 1.0, 0.0, 0.0]), -pol, lambda q: (q[0], -q[1])))
        if self.arm_dcm and sw == "OFF":

            def reset(z):
                z = z.copy()
                z[0] = 0.0
                return z

            g.append(Guard("i_L = 0 (DCM)", np.array([1.0, 0.0, 0.0, 0.0, 0.0]), -1, lambda q: ("DCM", q[1]), reset))
        if self.arm_dcm and sw == "DCM":
            g.append(Guard("|v_g| = v_C (자연 정류)", np.array([0.0, -1.0, pol * self.Vpk, 0.0, 0.0]), +1, lambda q: ("OFF", q[1])))
        return g

    def gate_schedule(self, t0, t1):
        """Centre-aligned PWM in [t0, t0 + T): OFF, ON for d*T around the centre, OFF."""
        d, T = self.d, self.T
        if d <= 0:
            return []
        ev = [(t0 + (1 - d) * T / 2, lambda q: ("ON", q[1]))]
        if d < 1:
            ev.append((t0 + (1 + d) * T / 2, lambda q: ("OFF", q[1])))
        return [e for e in ev if t0 <= e[0] < t1]

    def after_event(self, q, z):
        sw, pol = q
        s, c = z[2], z[3]
        pol = (1 if c > 0 else -1) if abs(s) <= 1e-12 else (1 if s > 0 else -1)
        if sw == "OFF" and z[0] <= 1e-12 * max(1.0, abs(z[1]) / 100.0):
            sw = "DCM"
        return (sw, pol)

    def describe(self, q):
        # band keys: Q (switch on), D (boost diode), Z (DCM, i_L = 0); + / − bridge polarity
        return f"{ {'ON': 'Q', 'OFF': 'D', 'DCM': 'Z'}[q[0]] }{'+' if q[1] > 0 else '−'}"

    def outputs(self, q):
        pol = q[1]
        e = np.eye(5)
        return {"iL": e[0], "ig": pol * e[0], "vC": e[1], "vg": self.Vpk * e[2], "s": e[2], "c": e[3], "iQ": e[0] if q[0] == "ON" else 0 * e[0], "iD": e[0] if q[0] == "OFF" else 0 * e[0]}

    def stored_energy(self):
        return np.diag([self.L, self.C, 0.0, 0.0, 0.0])

    def powers(self, q):
        pol = q[1]
        e = np.eye(5)
        Qg = 0.5 * pol * self.Vpk * (np.outer(e[0], e[2]) + np.outer(e[2], e[0]))
        return {"p_grid": Qg, "p_load": np.outer(e[1], e[1]) / self.R, "p_loss": self.RL * np.outer(e[0], e[0])}


class PFCController:
    """Sampled average current-mode control (one sample per switching period, one-period delay).

    Voltage loop (slow): P_cmd = Kpv (V_ref - v) + xv, clamp [0, P_max], back-calculation.
    Current loop (fast, volt output): v_L* = Kpi (i_ref - i) + xi, i_ref = 2 P_cmd |v_g| / V_pk^2,
    switch-node command v_sw = |v_g| - v_L*, modulator d = 1 - v_sw / v_C (divides by the measured
    DC link), clamp [0, d_max], back-calculation on the volt quantity actually applied.
    """

    def __init__(self, p: dict):
        self.T = 1.0 / p["fs"]
        self.V = p["Vref"]
        self.Vpk = p["Vpk"]
        wci = 2 * math.pi * p["fci"]
        self.Kpi = p["L"] * wci
        self.Kii = self.Kpi * wci / p["zi"]
        self.Kawi = self.Kii / self.Kpi
        wcv = 2 * math.pi * p["fcv"]
        self.Kpv = wcv * p["C"] * p["Vref"]
        self.Kiv = self.Kpv * wcv
        self.Kawv = self.Kiv / self.Kpv
        self.Pmax = 1.5 * p["P"]
        self.dmax = p["dmax"]
        self.xv = p["P"]
        self.xi = 0.0
        self.d_next = 0.0
        self.last = (0.0, 0.0)

    def state(self):
        return (self.xv, self.xi, self.d_next)

    def load(self, st):
        self.xv, self.xi, self.d_next = st

    def step(self, i_s: float, v_s: float, vg_s: float) -> float:
        ev = self.V - v_s
        Pu = self.Kpv * ev + self.xv
        Pc = min(max(Pu, 0.0), self.Pmax)
        self.xv += self.T * (self.Kiv * ev + self.Kawv * (Pc - Pu))
        avg = abs(vg_s)
        iref = 2.0 * Pc * avg / self.Vpk**2
        ei = iref - i_s
        vLu = self.Kpi * ei + self.xi
        d = 1.0 - (avg - vLu) / v_s
        dc = min(max(d, 0.0), self.dmax)
        vLa = avg - (1.0 - dc) * v_s
        self.xi += self.T * (self.Kii * ei + self.Kawi * (vLa - vLu))
        d_apply, self.d_next = self.d_next, dc
        self.last = (iref, Pc)
        return d_apply


def averaged_pfc(p: dict, ctl: PFCController, n_periods: int, x0, t0=0.0, sub: int = 4, record: bool = False):
    """B level: hand-written averaged boost ODE (CCM; i clamped at 0), RK4 with ``sub`` steps per period."""
    Vpk, L, C, R, RL = p["Vpk"], p["L"], p["C"], p["R"], p["RL"]
    w = 2 * math.pi * p["f"]
    T = 1.0 / p["fs"]
    h = T / sub
    i, v = x0
    t = t0
    ts, isg, vs, clamped = [], [], [], 0
    for _ in range(n_periods):
        dd = ctl.step(i, v, Vpk * math.sin(w * t))
        a = 1.0 - dd
        for _j in range(sub):
            if record:
                ts.append(t)
                isg.append(i if math.sin(w * t) >= 0 else -i)
                vs.append(v)

            def f(tt, ii, vv):
                return (Vpk * abs(math.sin(w * tt)) - RL * ii - a * vv) / L, (a * ii - vv / R) / C

            k1 = f(t, i, v)
            k2 = f(t + h / 2, i + h / 2 * k1[0], v + h / 2 * k1[1])
            k3 = f(t + h / 2, i + h / 2 * k2[0], v + h / 2 * k2[1])
            k4 = f(t + h, i + h * k3[0], v + h * k3[1])
            i += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            t += h
            if i < 0:
                i = 0.0
                clamped += 1
    return (i, v), t, {"t": np.array(ts), "ig": np.array(isg), "v": np.array(vs), "clamped": clamped}


def hermite_fourier(segs, t0: float, t1: float, w: float, H: int, which: str = "ig", n_gl: int = 6):
    """Fourier coefficients (complex, peak) h = 0..H of i_g or v_C over [t0, t1] from exact segment ends.

    Each segment is represented by the cubic Hermite through the exact end values and end
    derivatives (from the mode matrix), integrated against exp(-j h w t) with Gauss-Legendre.
    Also returns int x^2 dt and int x v_g dt (for RMS and power) and the Hermite error bound.
    """
    sel = [s for s in segs if s.t0 >= t0 - 1e-12 and s.t1 <= t1 + 1e-12]
    col = 0 if which == "ig" else 1
    t_a = np.array([s.t0 for s in sel])
    h_a = np.array([s.h for s in sel])
    sg = np.array([float(s.q[1]) if which == "ig" else 1.0 for s in sel])
    x0 = np.array([s.z0[col] for s in sel])
    x1 = np.array([s.z1[col] for s in sel])
    d0 = np.array([(s.mode.F @ s.z0)[col] for s in sel])
    d1 = np.array([(s.mode.F @ s.z1)[col] for s in sel])
    F4 = {}
    bound = 0.0
    for s in sel:
        k = s.mode.key
        if k not in F4:
            F4[k] = np.linalg.matrix_power(s.mode.F, 4)[col]
        bound = max(bound, (abs(F4[k] @ s.z0) + abs(F4[k] @ s.z1)) / 2 * s.h**4 / 384)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    u = (xg + 1) / 2
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    vals = (np.outer(x0, h00) + np.outer(h_a * d0, h10) + np.outer(x1, h01) + np.outer(h_a * d1, h11)) * sg[:, None]
    tq = t_a[:, None] + h_a[:, None] * u[None, :]
    wq = h_a[:, None] * wg[None, :] / 2
    Tw = t1 - t0
    coef = np.array([np.sum(vals * wq * np.exp(-1j * k * w * tq)) * 2 / Tw for k in range(H + 1)])
    sq = float(np.sum(vals * vals * wq))
    return coef, sq, (vals, tq, wq), bound


def exact_window_integrals(sys: BoostPFC, segs, t0: float, t1: float) -> dict:
    """Exact integrals over [t0, t1] with the engine's second-moment matrices (one per segment)."""
    e = np.eye(5)
    acc = {"E_grid": 0.0, "E_load": 0.0, "E_loss": 0.0, "ig2": 0.0, "ig_s": 0.0, "ig_c": 0.0, "ig": 0.0}
    for s in segs:
        if s.t0 < t0 - 1e-12 or s.t1 > t1 + 1e-12:
            continue
        S2 = segment_moments(s.mode, s.z0, s.h)
        pw = sys.powers(s.q)
        pol = s.q[1]
        acc["E_grid"] += float(np.sum(pw["p_grid"] * S2))
        acc["E_load"] += float(np.sum(pw["p_load"] * S2))
        acc["E_loss"] += float(np.sum(pw["p_loss"] * S2))
        acc["ig2"] += float(S2[0, 0])
        acc["ig_s"] += pol * float(S2[0, 2])
        acc["ig_c"] += pol * float(S2[0, 3])
        acc["ig"] += pol * float(S2[0, 4])
    del e
    return acc


def _pfc_params_from(v: dict) -> dict:
    p = {
        "Vpk": v["V_rms"] * SQ2,
        "f": v["f"],
        "L": v["L"],
        "C": v["C"],
        "R": v["Vref"] ** 2 / v["P"],
        "RL": v["RL"],
        "fs": v["fs"],
        "Vref": v["Vref"],
        "P": v["P"],
        "fci": v["fci"],
        "zi": v["zi"],
        "fcv": v["fcv"],
        "dmax": v["dmax"],
    }
    return p


def simulate_pfc(v: dict) -> dict:
    """B pre-settle -> C settle (per-line-cycle criterion) -> C analysis window of n_cyc cycles."""
    p = _pfc_params_from(v)
    per = int(round(p["fs"] / p["f"]))
    if abs(per - p["fs"] / p["f"]) > 1e-9:
        raise ValueError("f_s / f must be an integer (commensurate switching and line periods)")
    T = 1.0 / p["fs"]
    Tl = per * T
    w = 2 * math.pi * p["f"]
    # --- B pre-settle from the analytic averaged guess
    ctlB = PFCController(p)
    xB, tB, _ = averaged_pfc(p, ctlB, per * int(v["n_b"]), (0.0, p["Vref"]))
    # --- C from the B state at a zero crossing (t = 0 of the next line cycle)
    sysC = BoostPFC(p)
    ctl = PFCController(p)
    ctl.load(ctlB.state())
    z = np.array([xB[0], xB[1], 0.0, 1.0])
    q = ("OFF", 1)
    t = 0.0
    cw, sw_ = math.cos(w * T), math.sin(w * T)
    n_an = int(v["n_cyc"])
    crit_hist = []
    prev = None
    steady_at = None
    segs_an = []
    rearm = 0
    last_dcm = True
    k_cyc = 0
    max_settle = int(v["max_settle"])
    scales = np.array([max(2 * p["P"] / p["Vpk"], 1e-3), p["Vref"], max(p["P"], 1.0), 0.01 * p["Vref"], 1.0])
    t_an0 = None
    while True:
        in_an = t_an0 is not None
        for _k in range(per):
            d = ctl.step(z[0], z[1], p["Vpk"] * z[2])
            sysC.d = d
            s_end = z[2] * cw + z[3] * sw_
            sysC.arm_zc = z[2] * s_end < 0.0
            # DCM / natural-conduction guards are armed whenever a crossing is possible: always after
            # a period that touched i_L = 0, otherwise only if the unarmed exact solution shows one
            # (i_L is strictly decreasing in OFF while |v_g| < v_C, so a crossing ends the segment < 0).
            sysC.arm_dcm = last_dcm or z[0] <= 0.0
            tr = simulate(sysC, q, z, t, t + T)
            if not sysC.arm_dcm and (p["Vpk"] >= min(z[1], tr.z_end[1]) or any((s.q[0] == "OFF" and s.z1[0] < 0.0) or s.q[0] == "DCM" for s in tr.segments)):
                sysC.arm_dcm = True
                tr = simulate(sysC, q, z, t, t + T)
                rearm += 1
            last_dcm = any(s.q[0] == "DCM" for s in tr.segments) or tr.z_end[0] <= 0.0
            if in_an:
                segs_an.extend(tr.segments)
            z = tr.z_end[:-1].copy()
            q = tr.q_end
            t += T
        k_cyc += 1
        vec = np.array([z[0], z[1], ctl.xv, ctl.xi, ctl.d_next])
        if prev is not None:
            crit_hist.append(float(np.max(np.abs(vec - prev) / scales)))
        prev = vec
        if in_an:
            if k_cyc - (t_an0 / Tl) >= n_an - 1e-9:
                break
            continue
        if (len(crit_hist) >= 2 and crit_hist[-1] < v["crit"] and crit_hist[-2] < v["crit"]) or k_cyc >= max_settle:
            steady_at = k_cyc if (crit_hist and crit_hist[-1] < v["crit"]) else None
            t_an0 = t
    t_an1 = t
    # --- B continued over the same window for comparison
    ctlB2 = PFCController(p)
    ctlB2.load(ctlB.state())
    xB2, _, _ = averaged_pfc(p, ctlB2, per * (k_cyc - n_an), xB)
    _, _, recB = averaged_pfc(p, ctlB2, per * n_an, xB2, t0=0.0, record=True)
    return {"p": p, "sys": sysC, "segs": segs_an, "t0": t_an0, "t1": t_an1, "Tl": Tl, "per": per, "crit": crit_hist, "steady_at": steady_at, "settle_cycles": k_cyc - n_an, "rearm": rearm, "B": recB, "w": w}


def run_boost_pfc(v: dict) -> Result:
    res = Result("FL05", "boost_pfc", "C (정확 스위칭) + B 평균모델 + A 강제 정현파")
    ratio = v["fs"] / v["f"]
    if abs(ratio - round(ratio)) > 1e-9:
        res.verdict("OUT_OF_VALIDITY", f"f_s/f = {ratio:.6g}가 정수가 아니다: 스위칭 주기와 계통 주기가 맞지 않으면 정수 주기 창의 주기해가 존재하지 않는다 (값을 반올림하지 않고 거부).")
        return res
    if v["V_rms"] * SQ2 >= 0.97 * v["Vref"]:
        res.verdict("OUT_OF_VALIDITY", f"계통 peak {v['V_rms'] * SQ2:.4g} V가 DC-link 기준 {v['Vref']:g} V의 97 % 이상이다: boost PFC는 V_pk < V_dc에서만 전류를 정형할 수 있다.")
        return res
    sim = simulate_pfc(v)
    p, sysC, segs = sim["p"], sim["sys"], sim["segs"]
    t0, t1, Tl, w = sim["t0"], sim["t1"], sim["Tl"], sim["w"]
    H = int(v["H"])
    n_an = int(v["n_cyc"])
    Vrms = p["Vpk"] / SQ2
    # --- harmonic analysis over the whole window (textbook definition)
    coef, sq, (vals, tq, wq), bound = hermite_fourier(segs, t0, t1, w, H, "ig")
    I0 = coef[0].real / 2
    Ih = np.abs(coef[1:]) / SQ2
    I1 = float(Ih[0])
    thd = math.sqrt(float(np.sum(Ih[1:] ** 2))) / I1
    Irms_w = math.sqrt(sq / (t1 - t0))
    P_w = float(np.sum(vals * p["Vpk"] * np.sin(w * tq) * wq)) / (t1 - t0)
    pf_w = P_w / (Vrms * Irms_w)
    # v_g = V_pk sin(wt) has phase -pi/2 in the exp(-j h w t) basis; phi_1 = angle(V_1) - angle(I_1)
    cosphi = math.cos(-math.pi / 2 - float(np.angle(coef[1])))
    # --- exact moments over the last line cycle (independent path) + energy ledger
    ta, tb_ = t1 - Tl, t1
    ex = exact_window_integrals(sysC, segs, ta, tb_)
    I1c = math.hypot(ex["ig_s"], ex["ig_c"]) * 2 / Tl / SQ2
    Irms_x = math.sqrt(ex["ig2"] / Tl)
    P_x = p["Vpk"] * (ex["ig_s"]) / Tl
    coef_last, sq_last, (vl, tql, wql), _ = hermite_fourier(segs, ta, tb_, w, 1, "ig")
    I1_h_last = abs(coef_last[1]) / SQ2
    sl = [s for s in segs if s.t0 >= ta - 1e-12]
    W = sysC.stored_energy()
    dW = 0.5 * float(sl[-1].z1 @ W @ sl[-1].z1) - 0.5 * float(sl[0].z0 @ W @ sl[0].z0)
    resid = ex["E_grid"] - ex["E_load"] - ex["E_loss"] - dW
    norm = max(abs(ex["E_grid"]), abs(ex["E_load"]), 0.01 * p["P"] * Tl)
    # --- DC link: mean, 2-omega component, switching ripple (Hermite on v_C)
    cv, _, (vv, _, wv), _ = hermite_fourier(segs, t0, t1, w, 4, "vC")
    Vmean = cv[0].real / 2
    V2 = abs(cv[2])  # peak amplitude of the 2-omega component
    last2v = [s for s in segs if s.t0 >= t1 - 2 * Tl - 1e-12]
    vmin = min(min(s.z0[1], s.z1[1]) for s in last2v)
    vmax = max(max(s.z0[1], s.z1[1]) for s in last2v)
    Pin = ex["E_grid"] / Tl
    Pout = ex["E_load"] / Tl
    dV_formula = ref.ripple_pp(Pout, p["f"], p["C"], Vmean)
    # --- averaged model (B) over the same window
    B = sim["B"]
    xb = B["ig"]
    Nb = xb.size
    XB = np.fft.rfft(xb) / Nb
    IhB = np.array([2 * abs(XB[k * n_an]) / SQ2 for k in range(1, H + 1)])
    thdB = math.sqrt(float(np.sum(IhB[1:] ** 2))) / IhB[0]
    VB = np.fft.rfft(B["v"]) / Nb
    VmeanB = float(VB[0].real)
    V2B = 2 * abs(VB[2 * n_an])  # peak amplitude of the 2-omega component (B)
    # --- forced ideal sine: THD = 0 by construction, same routine on the analytic waveform
    tf = np.arange(n_an * 2000) / (2000 * p["f"])
    xf = SQ2 * I1 * np.sin(w * tf)
    Xf = np.fft.rfft(xf) / xf.size
    Ihf = np.array([2 * abs(Xf[k * n_an]) / SQ2 for k in range(1, H + 1)])
    thdf = math.sqrt(float(np.sum(Ihf[1:] ** 2))) / Ihf[0]
    # --- DCM share and duty saturation (window)
    t_dcm = sum(s.h for s in segs if s.q[0] == "DCM")
    dcm_frac = t_dcm / (t1 - t0)
    # --- metrics
    res.add_metric("thd", f"THD_i (H = {H}, 정상상태 {n_an}주기, 스위칭 모델)", thd * 100, "%", basis="계통 전류 i_g = pol·i_L, 창의 기본파 기준")
    res.add_metric("thd_B", "THD_i (평균모델 B, 같은 제어기)", thdB * 100, "%", basis="리플 없는 평균 전류 — CCM에서만 유효")
    res.add_metric("thd_forced", "THD_i (강제 이상 정현파)", thdf * 100, "%", note="구성상 0: 입력한 파형이 정현파다. closed-loop PFC 성능으로 주장하지 않는다.")
    for hh in (3, 5, 7):
        res.add_metric(f"h{hh}", f"{hh}차 고조파 / 기본파", Ih[hh - 1] / I1 * 100, "%")
    res.add_metric("pf", "PF (창 전체, 스위칭 모델)", pf_w, "", basis="P/(V_rms·I_rms), I_rms에 스위칭 리플 포함")
    res.add_metric("cos_phi1", "변위역률 cosφ₁", cosphi, "")
    res.add_metric("pf_formula", "cosφ₁/√(1+THD²) (H = 40까지)", cosphi / math.sqrt(1 + thd * thd), "", note="I_rms의 스위칭 리플 성분은 THD(H ≤ 40)에 안 들어가므로 실제 PF보다 약간 크다")
    tot = math.sqrt(max(Irms_x**2 - (ex["ig"] / Tl) ** 2 - I1c**2, 0.0)) / I1c
    res.add_metric("dist_total", "전체 왜곡 (고조파 한계 없음, 스위칭 리플 포함)", tot * 100, "%", basis="√(I_rms² − I_0² − I_1²)/I_1, 마지막 1주기 정확 적분", note="교재 THD 정의(H = 40)가 아니다: 스위칭 리플은 EMI 필터가 다룬다")
    res.add_metric("I1", "기본파 전류 RMS", I1, "A")
    res.add_metric("Irms", "계통 전류 RMS", Irms_w, "A")
    res.add_metric("Pin", "계통 입력 전력 (마지막 1주기)", Pin, "W")
    res.add_metric("eta", "효율 (DCR 손실만)", Pout / Pin, "", basis="소자 도통·스위칭 손실 미포함")
    res.add_metric("Vdc", "DC-link 평균", Vmean, "V", ref=v["Vref"], ref_label="전압 루프 기준", tol=5e-3)
    formula_ok = thd < 0.05 and dcm_frac < 5e-3
    res.add_metric("dV2w", "DC-link 2ω 성분 (peak-to-peak 환산 2·|V₂|)", 2 * V2, "V", ref=dV_formula, ref_label="P/(ωCV_dc)", tol=0.05 if formula_ok else None, basis="창의 DFT 2차 성분, 스위칭 리플 제외",
                   note="" if formula_ok else "전류가 정현 동상이 아니어서(THD·DCM) 식의 가정 밖 — 비교는 참고용")
    res.add_metric("dV_pp", "DC-link 실제 pp (스위칭 리플 포함, 마지막 2주기)", vmax - vmin, "V")
    res.add_metric("dcm", "DCM 시간 비율 (창)", dcm_frac * 100, "%", basis="영교차 부근 i_L = 0 구간")
    res.add_metric("settle", "C 모델 정상상태 판정까지 주기", sim["settle_cycles"], "주기", basis=f"B 모델 {int(v['n_b'])}주기로 초기화 후")
    # --- checks
    res.add_check(
        Check(
            "에너지 잔차 (E_grid − E_load − E_loss − ΔW), 마지막 1주기",
            "PASS" if abs(resid / norm) < 1e-6 else "FAIL",
            resid / norm,
            "rel",
            1e-6,
            path="v_g·i_g, v_C²/R, R_L·i_L², ½Li² + ½Cv²의 정확 2차 모멘트 적분 (상태식과 독립인 포트 전력 정의)",
            independent=True,
            detail=f"E_grid {ex['E_grid']:.6g} J, E_load {ex['E_load']:.6g} J, E_loss {ex['E_loss']:.4g} J, ΔW {dW:.3g} J → 잔차 {resid:.3g} J",
        )
    )
    res.add_check(check_close("기본파: Hermite–GL Fourier vs 정확 모멘트 ∫i_g·s, ∫i_g·c (마지막 1주기)", I1_h_last, I1c, 1e-7, "구간 끝 정확값·미분의 3차 Hermite + Gauss–Legendre vs 발진기 상태 s, c를 쓴 엔진 2차 모멘트", True, "A"))
    res.add_check(check_close("I_rms: Hermite–GL vs 정확 모멘트 (마지막 1주기)", math.sqrt(sq_last / Tl), Irms_x, 1e-7, "같은 두 경로", True, "A"))
    res.add_check(check_close("입력 전력: Hermite × v_g vs 정확 모멘트 (마지막 1주기)", float(np.sum(vl * p["Vpk"] * np.sin(w * tql) * wql)) / Tl, P_x, 1e-7, "같은 두 경로", True, "W"))
    if formula_ok:
        res.add_check(check_close("DC-link 2ω 리플: 스위칭 모델 vs P/(ωCV_dc)", 2 * V2, dV_formula, 0.05, "스위칭 해의 2차 Fourier 성분(C) vs 작은 리플 해석식(A)", True, "V"))
    else:
        res.add_check(Check("DC-link 2ω 리플: 스위칭 모델 vs P/(ωCV_dc)", "INFO", abs(2 * V2 - dV_formula) / dV_formula, "rel", 0.05, path="C vs A", independent=True,
                            detail=f"C {2 * V2:.4g} V, 식 {dV_formula:.4g} V — THD {thd * 100:.3g} %·DCM {dcm_frac * 100:.2g} %라 식의 가정(정현 동상 전류)이 깨져 판정하지 않음"))
    ccm = dcm_frac < 5e-3
    for name, a_, b_, tol in (("DC-link 평균", Vmean, VmeanB, 2e-3), ("기본파 전류 (B vs C)", I1, IhB[0], 0.01), ("2ω 리플 pp (B vs C)", 2 * V2, 2 * V2B, 0.03)):
        e_ = abs(a_ - b_) / abs(b_)
        res.add_check(Check(f"평균모델(B) vs 스위칭(C): {name}", ("PASS" if e_ <= tol else "FAIL") if ccm else "INFO", e_, "rel", tol, path="손으로 쓴 평균 ODE + RK4 vs 행렬지수 스위칭 해 (같은 제어기 코드)", independent=True,
                            detail=f"C {a_:.6g}, B {b_:.6g}" + ("" if ccm else " — DCM 구간이 있어 CCM 평균모델 비교는 참고용")))
    res.add_check(Check("Hermite 보간 오차 상한 (i_g)", "PASS" if bound < 1e-6 * max(I1, 1e-3) else "FAIL", bound, "A", 1e-6 * max(I1, 1e-3), path="max |F⁴z|·h⁴/384 (구간 끝의 정확한 4차 미분)", independent=False, detail="Fourier 적분의 표현 오차가 THD에 영향을 주지 않을 만큼 작다는 확인"))
    res.add_check(Check("정상상태 판정 (계통 주기별 상태 변화)", "INFO", sim["crit"][-1] if sim["crit"] else None, "rel", v["crit"], path="같은 위상(영교차)에서 [i_L, v_C, x_v, x_i, d]의 주기 간 정규화 변화",
                        detail=f"B {int(v['n_b'])}주기 사전 수렴 → C {sim['settle_cycles']}주기 후 판정" + (f" (기준 충족: {sim['steady_at']}주기)" if sim["steady_at"] else " (최대 주기 도달 — 창 안 변화량을 함께 보라)") + f"; 분석 창 {n_an}주기의 주기별 최대 변화 {max(sim['crit'][-n_an:]):.2e}"))
    # --- plots: line scale (endpoints of segments are exact samples)
    last2 = [s for s in segs if s.t0 >= t1 - 2 * Tl - 1e-12]
    tt = [s.t0 for s in last2] + [last2[-1].t1]
    ig = [s.q[1] * s.z0[0] for s in last2] + [last2[-1].q[1] * last2[-1].z1[0]]
    vc = [s.z0[1] for s in last2] + [last2[-1].z1[1]]
    tt = [x - (t1 - 2 * Tl) for x in tt]
    xs, ys = decimate_minmax(tt, ig, 2000)
    res.add_series("ig_C", "i_g 스위칭 (C, 구간 끝 정확값)", "A", xs, ys)
    tb0 = B["t"] - (B["t"][-1] + (B["t"][1] - B["t"][0]) - 2 * Tl)
    mB = tb0 >= 0
    res.add_series("ig_B", "i_g 평균모델 (B)", "A", tb0[mB].tolist(), B["ig"][mB].tolist(), dash=True)
    tf2 = np.linspace(0, 2 * Tl, 801)
    res.add_series("ig_F", "강제 이상 정현파 (THD = 0 구성)", "A", tf2.tolist(), (SQ2 * I1 * np.sin(w * tf2)).tolist(), dash=True)
    res.add_series("vg10", "v_g / 20", "V", tf2.tolist(), (p["Vpk"] * np.sin(w * tf2) / 20).tolist())
    res.add_plot("p_line", "계통 전류: 스위칭(C)·평균(B)·강제 정현파", ["ig_C", "ig_B", "ig_F", "vg10"], x_label="t (분석 창 마지막 2주기)", y_label="i_g, v_g/20", y_unit="", group="line", level="C/B/A",
                 proved="같은 제어기로 스위칭 모델과 평균모델의 계통 전류를 계산했고, 영교차 부근 왜곡과 2ω 전압 리플이 만드는 3차 고조파가 시뮬레이션 전류에서 나온다는 것을 보였다.",
                 not_yet="이상 다이오드·스위치, EMI 필터 없음. 스위칭 리플은 그대로 보이며(필터 전 전류), 전도 EMI 판정은 하지 않는다.")
    xs, ys = decimate_minmax(tt, vc, 2000)
    res.add_series("vC", "v_C (스위칭, C)", "V", xs, ys)
    vbm = B["v"][mB]
    res.add_series("vC_B", "v_C (평균모델, B)", "V", tb0[mB].tolist(), vbm.tolist(), dash=True)
    res.add_plot("p_vc", "DC-link 전압: 2ω 리플", ["vC", "vC_B"], x_label="t (분석 창 마지막 2주기)", y_label="v_C", y_unit="V", group="line", level="C/B",
                 hlines=[{"y": Vmean + dV_formula / 2, "label": "평균 + P/(2ωCV)"}, {"y": Vmean - dV_formula / 2, "label": "평균 − P/(2ωCV)"}],
                 proved="단상 순시전력 P(1−cos2ωt)의 2ω 성분을 C가 흡수해 P/(ωCV_dc)의 리플이 생긴다는 것을 스위칭 해로 확인했다.",
                 not_yet="부하는 저항(엔진이 선형이어야 함)이다. 후단 DC/DC(정전력 부하)면 증분저항이 음이 되어 전압 루프·입력필터와 상호작용한다(EX07).")
    # zero-crossing zoom with bands linked to the circuit
    tz = t1 - Tl  # zero crossing (rising) at the start of the last cycle
    zseg = [s for s in segs if tz - 0.6e-3 - 1e-12 <= s.t0 and s.t1 <= tz + 0.6e-3 + 1e-12]
    trz = Trajectory(sysC, zseg)
    smp = trz.sample(["ig", "iL", "vg"], per_segment=8)
    tzs = [x - tz for x in smp["t"]]
    res.add_series("z_ig", "i_g (C)", "A", tzs, smp["ig"])
    res.add_series("z_iL", "i_L (C, bridge 뒤)", "A", tzs, smp["iL"])
    res.add_series("z_vg", "v_g / 20", "V", tzs, [x / 20 for x in smp["vg"]])
    labels = {f"{a}{b}": f"{a_ko} ({b_ko})" for a, a_ko in (("Q", "Q ON"), ("D", "D 도통"), ("Z", "DCM")) for b, b_ko in (("+", "v_g>0"), ("−", "v_g<0"))}
    bands = [{"x0": iv["t0"] - tz, "x1": iv["t1"] - tz, "mode": iv["mode"], "label": labels.get(iv["mode"], iv["mode"])} for iv in trz.mode_intervals(tz - 0.6e-3, tz + 0.6e-3)]
    res.add_plot("p_zc", "영교차 확대: bridge 전환·DCM·duty 한계", ["z_ig", "z_iL", "z_vg"], x_label="영교차 기준 시간", y_label="전류, v_g/20", y_unit="", bands=bands, group="zc", level="C",
                 proved="영교차에서 bridge 극성이 발진기 상태 s = 0 사건으로 바뀌고, |v_g|가 작아 전류를 만들 전압이 부족하거나(duty 한계) DCM이 생겨 전류가 기준을 못 따라가는 구간이 보인다.",
                 not_yet="bridge 다이오드 역회복·접합용량, 입력 필터 커패시터의 무효전류는 모델에 없다(실제 영교차 왜곡은 더 복잡하다).")
    # spectrum
    hh = list(range(1, H + 1))
    res.add_series("sp_C", "스위칭 모델 (C)", "%", hh, [max(x / I1 * 100, 1e-4) for x in Ih], style="points")
    res.add_series("sp_B", "평균모델 (B)", "%", hh, [max(x / IhB[0] * 100, 1e-4) for x in IhB], style="points")
    res.add_series("sp_F", "강제 정현파", "%", hh, [max(x / Ihf[0] * 100, 1e-4) for x in Ihf], style="points")
    res.add_plot("p_spec", f"고조파 스펙트럼 (H = {H}, {n_an}주기 창)", ["sp_C", "sp_B", "sp_F"], x_label="차수 h", x_unit="", y_label="I_h / I_1", y_unit="%", kind="xy", log_y=True, level="C/B/A",
                 hlines=[{"y": 1e-4, "label": "표시 하한"}],
                 proved="강제 정현파의 고조파는 표시 하한(구성상 0)이고, 스위칭 모델의 3차는 전압 루프가 2ω 리플을 전류 기준에 넣은 결과, 고차 홀수는 영교차 왜곡의 결과다.",
                 not_yet="IEC 61000-3-2/3-12 한계와 비교하지 않았다(측정 조건·계통 임피던스·필터 미포함).")
    ch = sim["crit"]
    res.add_series("crit", "주기별 상태 변화 (정규화)", "", list(range(1, len(ch) + 1)), [max(x, 1e-16) for x in ch], style="points")
    res.add_plot("p_crit", "정상상태 판정: 계통 주기별 상태 변화", ["crit"], x_label="C 모델 계통 주기", x_unit="", y_label="변화", y_unit="", kind="xy", log_y=True,
                 hlines=[{"y": v["crit"], "label": "기준"}], vlines=[{"x": sim["settle_cycles"], "label": "분석 창 시작"}],
                 proved="THD 창은 주기별 상태 변화가 기준 아래로 내려간 뒤의 정수 주기만 쓴다(기동 구간 제외).", not_yet="기준값은 학습용 선택이다.")
    res.circuit = {"diagram": pfc_circuit(v).to_json(), "intervals": bands, "plot_group": "zc"}
    rows = [[h_, Ih[h_ - 1], Ih[h_ - 1] / I1 * 100, IhB[h_ - 1] / IhB[0] * 100] for h_ in range(1, min(H, 21) + 1, 2)]
    res.tables.append(Table("t_harm", "홀수 고조파 (RMS)", ["h", "I_h 스위칭 [A]", "스위칭 [%]", "평균모델 [%]"], rows, note="짝수 고조파는 반파 대칭 때문에 거의 0이다(창에서 계산해 스펙트럼에 표시)."))
    res.tables.append(
        Table(
            "t_thd",
            "THD 보고 조건 (창·기본파·H·기동 제외)",
            ["항목", "값"],
            [
                ["창", f"t = {t0 * 1e3:.4g}–{t1 * 1e3:.4g} ms (C 모델 시간), 정수 {n_an}계통주기"],
                ["기동 제외", f"B 평균모델 {int(v['n_b'])}주기 + C 모델 {sim['settle_cycles']}주기 후 (주기별 기준 {v['crit']:g})"],
                ["기본파", "창의 1차 Fourier 성분 (정수 창)"],
                ["고조파 상한", f"H = {H}"],
                ["적분", "구간 끝 정확값·미분의 3차 Hermite + Gauss–Legendre (마지막 1주기는 정확 모멘트와 대조)"],
                ["DC 성분", f"{I0:.3g} A"],
                ["전류 정의", "계통 전류 i_g = bridge 극성 × i_L (EMI 필터 전, 스위칭 리플 포함)"],
            ],
            note="강제 이상 정현파의 THD = 0은 구성의 결과이므로 PFC 성능 근거가 아니다. 이 THD도 이상 소자·합성 제어기의 모델 결과다.",
        )
    )
    res.tables.append(
        Table(
            "t_ctl",
            "제어기 설정 (단위 포함)",
            ["루프", "이득", "비고"],
            [
                ["전류 PI (전압 출력)", f"Kp = {sim_ctl_str(p)[0]}, Ki = {sim_ctl_str(p)[1]}", f"f_ci = {p['fci']:g} Hz, 영점 f_ci/{p['zi']:g}, |v_g| 피드포워드, d = 1 − v_sw/v_C"],
                ["전압 PI (전력 출력)", f"Kp = {sim_ctl_str(p)[2]}, Ki = {sim_ctl_str(p)[3]}", f"f_cv = {p['fcv']:g} Hz, i_ref = 2P·|v_g|/V_pk²"],
                ["샘플·갱신", f"f_s = {p['fs'] / 1e3:g} kHz, carrier valley 샘플", "계산한 duty는 다음 주기에 적용 (1주기 지연)"],
            ],
        )
    )
    res.verdict("PASS_WITHIN_MODEL", f"스위칭 모델(C)에서 계산한 THD {thd * 100:.3g} %, PF {pf_w:.4f} — 이상 소자·합성 제어기·이 창 조건의 모델 결과. 강제 정현파의 THD = 0은 성능 근거가 아니다.")
    if dcm_frac > 1e-3:
        res.warnings.append(f"창의 {dcm_frac * 100:.3g} %가 DCM이다: CCM 평균모델(B)과의 비교는 참고용이다.")
    res.assumptions += [
        "이상 다이오드 bridge·이상 스위치(즉시 전환), 부스트 다이오드 이상",
        f"계통은 이상 정현 전압원 (V_rms {v['V_rms']:g} V, {v['f']:g} Hz), 계통 임피던스·EMI 필터 없음",
        f"부하는 저항 R = V_ref²/P = {p['R']:.4g} Ω (엔진은 선형 회로), DCR {v['RL'] * 1e3:g} mΩ만 손실",
        "샘플링: carrier valley(OFF 구간 중앙)에서 i_L·v_C·v_g, duty는 다음 주기 적용",
    ]
    res.not_valid_for += ["IEC 고조파 규격 적합성·EMI", "소자 손실·효율 정밀값", "bridge 역회복·접합용량이 만드는 영교차 현상", "후단 DC/DC(정전력 부하)와의 상호작용 (EX07)"]
    res.interpretation = (
        f"이 모델의 THD {thd * 100:.3g} %는 대부분 3차({Ih[2] / I1 * 100:.3g} %)에서 나온다: 전압 루프가 DC-link의 2ω 리플을 보고 전력 명령을 흔들어 전류 기준에 2ω 진폭 변조를 넣기 때문이다. "
        f"고차 홀수 고조파는 영교차에서 |v_g|가 작아 전류를 만들 전압이 부족하고(duty 상한 {v['dmax']:g}) 한 주기 지연이 있어 생긴다. "
        f"강제 정현파 모델은 THD가 {thdf * 100:.1g} %로 0이지만 그것은 전류를 정현파로 ‘입력’했기 때문이지 제어·전력단이 그 파형을 만들었다는 증거가 아니다. "
        f"스위칭 리플까지 넣은 전체 왜곡 {tot * 100:.3g} %는 교재 THD(H = 40)와 다른 양이며 EMI 필터의 몫이다."
    )
    return res


def sim_ctl_str(p):
    c = PFCController(p)
    return (f"{c.Kpi:.4g} V/A", f"{c.Kii:.4g} V/(A·s)", f"{c.Kpv:.4g} W/V", f"{c.Kiv:.4g} W/(V·s)")


def pfc_circuit(v: dict) -> Circuit:
    c = Circuit("boost_pfc", 800, 330, title="단상 bridge + boost PFC (이상 소자)")
    # bridge legs at x = 160 and x = 300, rails at y = 60 (+) and y = 270 (-)
    d1 = c.add("diode", "D1", 160, 90, 270, "D1", lpos=(146, 94, "end"))
    d3 = c.add("diode", "D3", 160, 240, 270, "D3", lpos=(146, 244, "end"))
    d2 = c.add("diode", "D2", 300, 90, 270, "D2", lpos=(314, 94, "start"))
    d4 = c.add("diode", "D4", 300, 240, 270, "D4", lpos=(314, 244, "start"))
    g = c.add("vsource", "G", 230, 165, 0, "v_g", f"{v['V_rms']:g} V RMS", lpos=(230, 200, "middle"))
    ll = c.add("inductor", "L", 400, 60, 0, "L", f"{v['L'] * 1e3:g} mH")
    q = c.add("nmos", "Q", 480, 165, 90, "Q", lpos=(456, 160, "end"))
    dd = c.add("diode", "D", 550, 60, 0, "D", lpos=(550, 36, "middle"))
    cc = c.add("capacitor", "C", 620, 165, 90, "C", f"{v['C'] * 1e3:g} mF")
    rr = c.add("resistor", "RL", 710, 165, 90, "R_load", f"{v['Vref'] ** 2 / v['P']:.4g} Ω")
    c.wire("w_d1L", d1["a"], (160, 165))
    c.wire("w_d3L", d3["b"], (160, 165))
    c.wire("w_d2N", d2["a"], (300, 165))
    c.wire("w_d4N", d4["b"], (300, 165))
    c.wire("w_gL", g["a"], (160, 165))
    c.wire("w_gN", g["b"], (300, 165))
    c.wire("w_p1", d1["b"], (160, 60), (300, 60))
    c.wire("w_p2", d2["b"], (300, 60))
    c.wire("w_pL", (300, 60), ll["a"])
    c.wire("w_Lsw", ll["b"], (480, 60))
    c.wire("w_swq", (480, 60), q["a"])
    c.wire("w_swd", (480, 60), dd["a"])
    c.wire("w_dout", dd["b"], (620, 60))
    c.wire("w_oc", (620, 60), cc["a"])
    c.wire("w_or", (620, 60), (710, 60), rr["a"])
    c.wire("w_cg", cc["b"], (620, 270))
    c.wire("w_rg", rr["b"], (710, 270), (620, 270))
    c.wire("w_g1", (620, 270), (480, 270))
    c.wire("w_qg", q["b"], (480, 270))
    c.wire("w_g2", (480, 270), (300, 270))
    c.wire("w_n1", d3["a"], (160, 270), (300, 270))
    c.wire("w_n2", d4["a"], (300, 270))
    c.dot((160, 165), (300, 165), (300, 60), (480, 60), (620, 60), (620, 270), (480, 270), (300, 270))
    c.text(480, 44, "v_sw", "node")
    c.text(620, 44, "v_C", "node")
    c.probe("pL", "z_iL", 440, 48, "right", "i_L")
    c.probe("pG", "z_ig", 230, 148, "left", "i_g")
    out = ["C", "RL", "w_oc", "w_or", "w_cg", "w_rg"]
    bp = ["D1", "D4", "w_d1L", "w_d4N", "w_gL", "w_gN", "G", "w_p1", "w_pL", "w_n2"]
    bn = ["D2", "D3", "w_d2N", "w_d3L", "w_gL", "w_gN", "G", "w_p2", "w_pL", "w_n1"]
    on = ["L", "w_Lsw", "w_swq", "Q", "w_qg", "w_g2"]
    off = ["L", "w_Lsw", "w_swd", "D", "w_dout", "w_g1", "w_g2"]
    c.mode("Q+", "Q ON · v_g > 0 (D1, D4)", bp + on + out, "v_L = |v_g| − R_L i_L > 0: i_L 증가, 부하는 C가 공급", dim=["D2", "D3", "D"])
    c.mode("D+", "D 도통 · v_g > 0 (D1, D4)", bp + off + out, "v_L = |v_g| − v_C < 0: i_L이 C와 부하로", dim=["D2", "D3", "Q"])
    c.mode("Z+", "DCM · v_g > 0", out, "i_L = 0: bridge·D 모두 차단, 부하는 C가 공급", dim=["D1", "D2", "D3", "D4", "D", "Q", "L"])
    c.mode("Q−", "Q ON · v_g < 0 (D2, D3)", bn + on + out, "bridge 극성만 바뀌고 i_L은 같은 방향", dim=["D1", "D4", "D"])
    c.mode("D−", "D 도통 · v_g < 0 (D2, D3)", bn + off + out, "v_L = |v_g| − v_C < 0", dim=["D1", "D4", "Q"])
    c.mode("Z−", "DCM · v_g < 0", out, "i_L = 0", dim=["D1", "D2", "D3", "D4", "D", "Q", "L"])
    return c


# ======================================================================================
# Experiment 4: instantaneous power and DC-link sizing (A/B)
# ======================================================================================


def run_dclink(v: dict) -> Result:
    res = Result("FL05", "dclink_power", "A (+ B 에너지 ODE)")
    f = v["f"]
    w = 2 * math.pi * f
    # --- instantaneous power from waveforms
    n = 2000
    t = np.arange(2 * n) / (n * f)  # two cycles
    P1 = v["P1"]
    V1 = 230.0
    I1 = P1 / V1
    p1 = (SQ2 * V1 * np.sin(w * t)) * (SQ2 * I1 * np.sin(w * t))
    P3 = v["P3"]
    Vph = v["V_LL"] / SQ3
    I3 = P3 / (3 * Vph)
    u = v["unb"] / 100
    ph = [0.0, -2 * math.pi / 3, 2 * math.pi / 3]
    amp = [1.0, 1.0, 1.0 - u]
    pphase = []
    for k in range(3):
        pphase.append((SQ2 * Vph * np.sin(w * t + ph[k])) * (SQ2 * I3 * amp[k] * np.sin(w * t + ph[k])))
    p3 = pphase[0] + pphase[1] + pphase[2]
    # 2-omega amplitude from the exact DFT of two whole cycles (uniform rule, exact for these trig polynomials)
    rip1 = 2 * abs(np.fft.rfft(p1)[4]) / p1.size
    rip3 = 2 * abs(np.fft.rfft(p3)[4]) / p3.size
    # symmetrical components (independent path for the unbalance ripple): P_2w = 3 V+ I- (RMS)
    a = np.exp(2j * np.pi / 3)
    Iph = np.array([I3 * amp[k] * np.exp(1j * ph[k]) for k in range(3)])
    Ineg = abs((Iph[0] + a * a * Iph[1] + a * Iph[2]) / 3)
    rip3_sc = 3 * Vph * Ineg
    res.add_metric("p1_ripple", "단상 순시전력 2ω 진폭", rip1, "W", ref=P1 * 1.0, ref_label="P (p = P(1 − cos2ωt))", tol=1e-9, basis="파형 곱의 2ω Fourier 성분 (2주기 창)")
    res.add_metric("p1_mean", "단상 평균전력", float(np.mean(p1)), "W", ref=P1, ref_label="P", tol=1e-9)
    res.add_metric("p3_ripple", "3상 합 순시전력 2ω 진폭", rip3, "W", ref=rip3_sc if u > 0 else None, ref_label="3·V⁺·I⁻ (대칭좌표)", tol=1e-6, basis=f"불평형 c상 전류 −{v['unb']:g} %")
    res.add_metric("p3_mean", "3상 평균전력", float(np.mean(p3)), "W")
    if u == 0:
        res.add_check(Check("평형 3상: 순시전력 합의 리플", "PASS" if rip3 < 1e-9 * P3 else "FAIL", rip3 / P3, "rel", 1e-9, path="세 상의 v·i 곱을 더한 파형의 2ω Fourier 성분", independent=True, detail="상별 2ω 성분이 120°씩 어긋나 상쇄된다"))
    else:
        res.add_check(check_close("불평형 3상 리플: 시간영역 파형 vs 대칭좌표 3V⁺I⁻", rip3, rip3_sc, 1e-6, "상별 v·i 곱의 합 vs Fortescue 역상분 전류", True, "W"))
    # --- DC-link by 2-omega ripple (single phase)
    Pr, Vd, dV = v["P_rip"], v["V_rip"], v["dV_pp"]
    C_rip = Pr / (w * Vd * dV)
    tbr = abs(Pr - 7400) < 1e-9 and abs(Vd - 400) < 1e-9 and abs(dV - 20) < 1e-12 and abs(f - 50) < 1e-12
    res.add_metric("C_rip", "2ω 리플로 정한 C (단상)", C_rip, "F", ref=2.944e-3 if tbr else ref.ripple_capacitance(Pr, f, Vd, dV), ref_label="교재 약 2.944 mF" if tbr else "P/(ωV_dc·ΔV_pp)", tol=2e-4, basis="작은 리플 근사")
    # independent path: nonlinear energy ODE C v dv/dt = P(1 - cos 2wt) - P, periodic from v(0) = V
    sol = solve_ivp(lambda tt, y: [(Pr * (1 - math.cos(2 * w * tt)) - Pr) / (C_rip * y[0])], (0, 2 / f), [Vd], method="DOP853", rtol=1e-11, atol=1e-9, dense_output=True)
    tt = np.linspace(0, 2 / f, 1601)
    vv = sol.sol(tt)[0]
    pp_ode = float(np.max(vv) - np.min(vv))
    pp_exact = math.sqrt(Vd**2 + Pr / (w * C_rip)) - math.sqrt(Vd**2 - Pr / (w * C_rip))
    res.add_metric("pp_ode", "비선형 에너지 ODE의 리플 pp", pp_ode, "V", ref=dV, ref_label="목표 ΔV_pp", tol=0.01, basis="B: C·v·dv/dt = p_in − P")
    res.add_check(check_close("2ω 리플: 비선형 ODE vs 에너지 정확해 √(V²+P/ωC) − √(V²−P/ωC)", pp_ode, pp_exact, 1e-6, "DOP853 적분 vs ½Cv² = W₀ − (P/2ω)sin2ωt", True, "V"))
    # --- hold-up energy
    Ph, dt, Vh, Vl = v["P_hold"], v["dt_hold"], v["V_hi"], v["V_lo"]
    C_hold = 2 * Ph * dt / (Vh**2 - Vl**2)
    tbh = abs(Ph - 11e3) < 1e-9 and abs(dt - 2e-3) < 1e-15 and abs(Vh - 800) < 1e-9 and abs(Vl - 760) < 1e-9
    res.add_metric("C_hold", "hold-up으로 정한 최소 C", C_hold, "F", ref=705e-6 if tbh else ref.holdup_capacitance(Ph, dt, Vh, Vl), ref_label="교재 최소 705 µF" if tbh else "2PΔt/(V_hi² − V_lo²)", tol=2e-3 if tbh else 1e-9, basis="이상 에너지 계산 (PFC 공급 0 가정)")

    def discharge(C):
        ev = lambda tt, y: y[0] - Vl  # noqa: E731
        ev.terminal = True
        ev.direction = -1
        s = solve_ivp(lambda tt, y: [-Ph / (C * y[0])], (0, 50 * dt), [Vh], method="DOP853", rtol=1e-11, atol=1e-9, events=ev, dense_output=True)
        return (float(s.t_events[0][0]) if s.t_events[0].size else math.inf), s

    t_min, s_min = discharge(C_hold)
    res.add_check(check_close("hold-up: 정전력 방전 ODE의 V_lo 도달 시각 (C = C_min) vs Δt", t_min, dt, 1e-7, "C·v·dv/dt = −P를 event로 적분 vs 에너지 식", True, "s"))
    Cg = v["C_given"]
    dE = 0.5 * Cg * (Vh**2 - Vl**2)
    tbg = abs(Cg - 1e-3) < 1e-15 and abs(Vh - 800) < 1e-9 and abs(Vl - 760) < 1e-9
    res.add_metric("dE", f"C = {Cg * 1e6:.4g} µF를 {Vh:g}→{Vl:g} V로 쓸 때 에너지", dE, "J", ref=31.2 if tbg else ref.cap_energy_delta(Cg, Vh, Vl), ref_label="교재 31.2 J" if tbg else "½C(V_hi² − V_lo²)", tol=1e-9)
    res.add_metric("dE_frac", "저장에너지 대비 사용 비율", dE / (0.5 * Cg * Vh**2) * 100, "%", ref=9.75 if tbg else None, ref_label="교재 9.75 % (5 %가 아님)", tol=1e-9, basis="1 − (V_lo/V_hi)²")
    t_given, s_given = discharge(Cg)
    res.add_metric("t_given", f"{Ph / 1e3:g} kW를 유지하는 시간 (C = {Cg * 1e6:.4g} µF)", t_given, "s", ref=dE / Ph, ref_label="ΔE/P (교재 약 2.84 ms)", tol=1e-6)
    # --- plots
    res.add_series("p1", "단상 p(t)", "W", t.tolist(), p1.tolist())
    res.add_series("p3", f"3상 합 p(t) (불평형 {v['unb']:g} %)", "W", t.tolist(), p3.tolist())
    for k, nm in enumerate("abc"):
        res.add_series(f"pp{nm}", f"{nm}상 p(t)", "W", t.tolist(), pphase[k].tolist(), dash=True)
    res.add_plot("p_pow", "순시전력: 단상은 2ω로 출렁이고 평형 3상 합은 일정하다", ["p1", "p3", "ppa", "ppb", "ppc"], y_label="p", y_unit="W", level="A", group="pw",
                 proved="단상 p = P(1 − cos2ωt)의 2ω 성분을 파형 곱으로 계산했고, 평형 3상은 세 상의 2ω 성분이 상쇄되어 합이 일정하다. 불평형이면 역상분 전류만큼 2ω 리플이 남는다.",
                 not_yet="이상 정현파 전압·전류(단위 역률). 고조파 전류·계통 불평형 전압은 넣지 않았다.")
    res.add_series("v_rip", "v_dc (에너지 ODE, 단상)", "V", tt.tolist(), vv.tolist())
    res.add_plot("p_rip", f"단상 DC-link 2ω 리플 (C = {C_rip * 1e3:.4g} mF)", ["v_rip"], y_label="v_dc", y_unit="V", level="B", group="pw",
                 hlines=[{"y": Vd + dV / 2, "label": "+ΔV/2"}, {"y": Vd - dV / 2, "label": "−ΔV/2"}],
                 proved="작은 리플 식 P/(ωCV)으로 고른 C가 비선형 에너지 ODE에서도 목표 리플을 만든다.",
                 not_yet="후단 전력 일정·PFC 입력 전력이 정확히 P(1−cos2ωt)인 이상 경우다. ESR 발열·수명·제어가 요구하는 용량은 별도다.")
    th = np.linspace(0, min(3 * dt, 1.5 * max(t_given, t_min)), 400)
    res.add_series("vh_min", f"C_min = {C_hold * 1e6:.4g} µF", "V", th.tolist(), [float(x) if x == x else None for x in _safe_sol(s_min, th)])
    res.add_series("vh_g", f"C = {Cg * 1e6:.4g} µF", "V", th.tolist(), [float(x) if x == x else None for x in _safe_sol(s_given, th)])
    res.add_plot("p_hold", f"hold-up: {Ph / 1e3:g} kW 정전력 방전", ["vh_min", "vh_g"], y_label="v_dc", y_unit="V", level="B", kind="xy", x_label="t", x_unit="s",
                 hlines=[{"y": Vl, "label": f"V_lo {Vl:g} V"}], vlines=[{"x": dt, "label": f"Δt {dt * 1e3:g} ms"}],
                 proved="hold-up 용량은 에너지 차 ½C(V_hi² − V_lo²)로 정하며, 전압이 5 % 내려도 에너지는 약 9.75 % 쓰인다.",
                 not_yet="그동안 PFC가 공급한 전력·제어 응답·ESR 전압강하는 넣지 않았다(넣으면 필요 에너지가 달라진다).")
    res.tables.append(
        Table(
            "t_size",
            "DC-link 용량을 정하는 서로 다른 이유",
            ["기준", "값", "적용 범위"],
            [
                ["단상 2ω 리플", f"{C_rip * 1e3:.4g} mF", "단상 PFC (평형 3상에는 기본 sizing으로 쓰지 않음)"],
                ["hold-up 에너지", f"{C_hold * 1e6:.4g} µF", "부하 step·순간 정전 동안 PFC 공급 0 가정"],
                ["3상 평형 2ω", "해당 없음 (리플 0)", f"불평형 {v['unb']:g} %면 2ω 진폭 {rip3:.4g} W"],
                ["ESR 발열·수명·제어", "별도 계산 (MISSING_INPUT)", "커패시터 데이터·온도 필요"],
            ],
            note="가장 큰 값 하나로 끝내지 말고 각 요구의 근거를 따로 남긴다.",
        )
    )
    res.circuit = {"diagram": dclink_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    if t_given < dt * (1 - 1e-9):
        res.verdict("FAIL_CONSTRAINT", f"C = {Cg * 1e6:.4g} µF는 {Ph / 1e3:g} kW를 {t_given * 1e3:.3g} ms만 유지해 요구 {dt * 1e3:g} ms에 못 미친다 (최소 {C_hold * 1e6:.4g} µF).")
    else:
        res.verdict("PASS_WITHIN_MODEL", "순시전력·리플·hold-up 에너지가 파형 곱·에너지 ODE와 해석식에서 일치한다 (이상 에너지 모델 범위).")
    res.assumptions += ["단상 예: 230 V RMS 단위 역률 정현파", "후단 전력 일정 (정전력 부하), PFC 입력 전력 P(1 − cos2ωt)", "hold-up 동안 PFC 공급 0, ESR·제어 응답 없음"]
    res.not_valid_for += ["커패시터 수명·ESR 발열 (데이터 필요)", "제어 루프가 요구하는 최소 용량", "3상 고조파·불평형 전압 조건의 정밀 리플"]
    res.interpretation = (
        "단상은 전압과 전류가 동상 정현파여도 순시전력이 P(1 − cos2ωt)로 0과 2P 사이를 오가므로, 후단 전력이 일정하면 그 2ω 성분을 DC-link가 흡수해 ΔV_pp ≈ P/(ωCV)가 된다. "
        "평형 3상은 세 상의 2ω 성분이 120°씩 어긋나 합이 일정하므로 이 식을 기본 sizing에 쓰지 않는다. hold-up은 전혀 다른 요구로 에너지 차 ½C(V_hi² − V_lo²)로 정하며, "
        f"800→760 V(5 %)를 쓰면 에너지는 {1 - (Vl / Vh) ** 2:.2%}가 빠진다."
    )
    return res


def _safe_sol(s, t):
    out = []
    for x in t:
        out.append(float(s.sol(x)[0]) if x <= s.t[-1] else float("nan"))
    return out


def dclink_circuit(v: dict) -> Circuit:
    c = Circuit("dclink", 640, 260, title="에너지 모델: PFC 입력 전력 → DC-link C → 정전력 후단")
    c.add("block", "PFC", 140, 130, 0, "PFC: p_in(t)", w=150, h=80)
    cc = c.add("capacitor", "C", 330, 130, 90, "C_dc", "½Cv²")
    c.add("block", "DCDC", 510, 130, 0, "DC/DC: P 일정", w=150, h=80)
    c.wire("w1", (215, 100), (330, 100), cc["a"])
    c.wire("w2", (215, 160), (330, 160), cc["b"])
    c.wire("w3", (330, 100), (435, 100))
    c.wire("w4", (330, 160), (435, 160))
    c.probe("pi", "none", 270, 88, "right", "p_in/v")
    c.probe("po", "none", 390, 88, "right", "P/v")
    c.text(330, 215, "C·v·dv/dt = p_in(t) − P_out", "note")
    return c


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q = [
    Question(
        "11 kW, 400 V LL, η 0.97, PF 0.995의 선전류와 16 A 한계에서의 출력은?",
        "I = P/(√3·V_LL·η·PF) = 11000/(√3·400·0.97·0.995) = 16.45043 A. 16 A면 √3·400·16·0.97·0.995 = 10.69881 kW. 360 V low line에서 11 kW에는 18.278 A가 필요하다. "
        "‘11 kW = 400 V × 16 A’는 정확한 출력 보증이 아니며, 어느 요구를 우선할지 고객과 정한다.",
        "What is the line current for 11 kW at 400 V line-to-line, 97 % efficiency and 0.995 power factor, and what can you deliver at 16 A?",
        "The line current is 11 kW divided by root three times 400 V times 0.97 times 0.995, which is 16.45 A. At a 16 A limit the battery output is about 10.70 kW, and at a 360 V low line you would need 18.28 A. So 11 kW at 16 A is not a guaranteed output, and the priority must be agreed with the customer.",
        ["16.45043 A", "10.69881 kW", "18.278 A", "η·PF 포함", "고객 우선순위"],
        kind="calc",
    ),
    Question(
        "위상만 맞추면 PF가 1인가?",
        "아니다. PF = P/S이고 전압이 정현파·DC 성분 없음 조건에서 PF = cosφ₁/√(1+THD²)이다. 고조파 전류는 전력을 나르지 않고 RMS만 키운다. "
        "THD는 창 길이(정상상태 정수 주기)·기본파 추정·H·센서 offset과 함께 보고한다.",
        "If the current is in phase with the voltage, is the power factor one?",
        "No. Power factor is real power over apparent power; with a sinusoidal voltage and no DC component it equals the displacement factor divided by the square root of one plus THD squared, because harmonic currents add RMS but carry no power. THD must be reported with its window, fundamental estimate and harmonic limit.",
        ["PF = P/S", "cosφ₁/√(1+THD²)", "조건: 정현 전압·DC 없음", "창·H·offset"],
    ),
    Question(
        "강제 정현파 전류 모델의 THD = 0을 PFC 성능으로 말해도 되나?",
        "안 된다. 전류를 정현파로 ‘입력’했으니 THD = 0은 구성의 결과다. 실제 PFC 전류는 전류 루프의 추종 오차, 전압 루프가 넣는 2ω 변조(3차 고조파), 영교차의 duty 한계·DCM, 샘플링 지연이 만든다. "
        "스위칭 또는 평균 폐루프 모델의 전류로 정의된 창에서 계산해야 한다.",
        "Can you quote the zero THD of a forced-sine current model as PFC performance?",
        "No. The current was imposed as a sine, so zero THD is a property of the construction. Real PFC current distortion comes from finite current-loop tracking, the voltage loop feeding the 2-omega ripple into the reference, zero-crossing duty limits or DCM, and sampling delay, so THD must come from a closed-loop model over a defined window.",
        ["구성상 0", "전압 루프 2ω → 3차", "영교차 왜곡", "폐루프 모델·창"],
        kind="pressure",
    ),
    Question(
        "단상과 3상의 DC-link 용량을 같은 식으로 정해도 되나?",
        "안 된다. 단상 p = P(1 − cos2ωt)의 2ω 성분 때문에 ΔV_pp ≈ P/(ωCV)가 필요하지만(7.4 kW, 50 Hz, 400 V, 20 Vpp → 약 2.944 mF), 평형 3상은 합성 순시전력이 일정하다. "
        "hold-up(에너지), ESR 발열·수명, 제어 요구는 각각 별도 기준이다.",
        "Can single-phase and three-phase DC links be sized with the same formula?",
        "No. Single-phase instantaneous power pulsates at twice the line frequency, so the ripple is about P over omega C V; that gives about 2.9 mF for 7.4 kW, 50 Hz, 400 V and 20 V peak-to-peak. A balanced three-phase input has constant total power, so hold-up energy, ESR heating, lifetime and control needs size it instead.",
        ["P(1−cos2ωt)", "2.944 mF", "3상 합 일정", "hold-up 별도"],
    ),
    Question(
        "1 mF를 800 V에서 760 V까지 쓰면 에너지는 몇 % 줄어드나?",
        "ΔE = ½·1 mF·(800² − 760²) = 31.2 J로 저장에너지의 9.75 %다. 전압은 5 % 내렸지만 에너지는 V²에 비례해 약 두 배 비율로 준다. 11 kW를 약 2.84 ms 유지하는 에너지다.",
        "Using 1 mF from 800 V down to 760 V, how much energy do you get?",
        "31.2 joules, which is 9.75 % of the stored energy, not 5 %, because energy scales with voltage squared. That supports 11 kW for about 2.8 milliseconds.",
        ["31.2 J", "9.75 %", "V² 비례", "2.84 ms"],
        kind="calc",
    ),
]

_GRID_PARAMS = [
    Param("P_bat", "배터리 출력 P_bat", "W", 11e3, "kW", vmin=100, vmax=1e6, source="TEXTBOOK", source_note="11 kW (합성 조건)"),
    Param("V_LL", "계통 선간전압 (RMS)", "V", 400.0, "V", vmin=50, vmax=2000, source="TEXTBOOK", source_note="400 V LL"),
    Param("eta", "효율 η (계통→배터리)", "", 0.97, "", vmin=0.5, vmax=1.0, source="TEXTBOOK", source_note="0.97"),
    Param("PF", "역률 PF (실제 P/S)", "", 0.995, "", vmin=0.3, vmax=1.0, source="TEXTBOOK", source_note="0.995"),
    Param("I_lim", "계통 전류 한계 (RMS)", "A", 16.0, "A", vmin=1, vmax=1000, source="TEXTBOOK", source_note="16 A", group="한계"),
    Param("V_LL_low", "low line 선간전압", "V", 360.0, "V", vmin=50, vmax=2000, source="TEXTBOOK", source_note="360 V", group="한계"),
    Param("V_dc", "DC-link 공칭", "V", 800.0, "V", vmin=100, vmax=2000, source="TEXTBOOK", source_note="800 V", group="변조"),
    Param("V_LL_hi", "고전압 코너 선간전압", "V", 440.0, "V", vmin=50, vmax=2000, source="TEXTBOOK", source_note="440 V", group="변조"),
    Param("V_dc_lo", "저 DC 코너", "V", 700.0, "V", vmin=100, vmax=2000, source="TEXTBOOK", source_note="700 V", group="변조"),
    Param("mod", "변조 방식", "", "spwm", kind="choice", choices=[("spwm", "SPWM (선형 m ≤ 1)"), ("svpwm", "SVPWM (이상 선형 m ≤ 2/√3)")], source="TEXTBOOK", group="변조"),
    Param("L_g", "상당 boost 인덕턴스 L_g", "H", 1e-3, "mH", vmin=0, vmax=0.1, source="ASSUMED", source_note="코너 전압강하 계산용 합성값", group="변조"),
    Param("headroom", "학습용 동적 여유 기준", "", 0.10, "", vmin=0, vmax=0.5, source="ASSUMED", source_note="과도·전류 조절·dead time 여유 (고객 사양 아님)", group="변조"),
    Param("f", "계통 주파수", "Hz", 50.0, "Hz", vmin=10, vmax=400, source="TEXTBOOK", group="변조"),
]

EXPERIMENTS = [
    Experiment(
        key="grid_boundary",
        title="11 kW와 16 A가 동시에 안 되는 조건: 입력 경계와 변조지수",
        goal=(
            "11 kW 배터리 출력·400 V LL·η 0.97·PF 0.995에서 선전류 16.45043 A, 16 A 한계의 출력 10.69881 kW, 360 V low line의 18.278 A를 계산하고 "
            "시간영역 3상 전력 경로로 확인한다. 상전압 230.94 V와 선간전압을 구분하고, 440 V LL/700 V DC 코너의 변조지수 1.0265가 SPWM 선형영역을 벗어남을 본다."
        ),
        params=_GRID_PARAMS,
        presets=[
            Preset("nominal", "교재 조건 (SPWM)", {}, "교재 08장", ("nominal", "reference")),
            Preset("svpwm", "SVPWM으로 코너 확인", {"mod": "svpwm"}, "이상 한계 1.1547", ("variant", "reference")),
            Preset("limit20", "전류 한계 20 A", {"I_lim": 20.0}, "11 kW 가능", ("variant",)),
            Preset("low_line", "공칭을 360 V로", {"V_LL": 360.0}, "저전압 운전", ("corner",)),
        ],
        run=run_grid_boundary,
        model_level="A (+ 시간영역 3상 전력 경로)",
        suggested_change="전류 한계를 16 A → 20 A로 올리거나, 변조 방식을 SVPWM으로 바꿔 코너를 다시 본다.",
        prediction=Prediction(
            "400 V LL, η 0.97, PF 0.995에서 11 kW 배터리 출력에 필요한 선전류는 16 A 한계와 비교해?",
            ["16 A 이하", "16 A를 약간 넘는다", "20 A 이상", "모르겠다"],
            "16 A를 약간 넘는다",
            "400 V × 16 A × √3 = 11.09 kVA이지만 η·PF를 곱하면 배터리 측 10.70 kW다. 11 kW에는 16.45 A가 필요하다.",
            ["I_line", "P_at_lim", "V_need"],
            handcalc=[{"key": "I_line", "label": "선전류", "unit": "A"}, {"key": "P_at_lim", "label": "16 A 출력", "unit": "W"}, {"key": "m_cor", "label": "440/700 V 변조지수", "unit": ""}],
        ),
        suggested={"I_lim": 20.0},
        student=(
            "배터리가 받는 전력은 계통에서 가져온 전력보다 손실만큼 적고, 계통 전류는 피상전력(유효전력/PF)으로 정해진다. 그래서 11 kW를 배터리에 넣으려면 계통에서 11.34 kW, "
            "피상전력 11.40 kVA가 필요하고 400 V 3상에서 16.45 A다. 전압이 낮아지면 같은 전력에 전류가 더 필요하다."
        ),
        expert=(
            "① 선간전압/상전압, RMS/peak를 구분한다: 상전압 230.94 V RMS, peak 326.6 V. ② 변조지수 m = V_ph,peak/(V_dc/2)는 440 V/700 V 코너에서 1.0265로 SPWM 선형영역 밖이다. "
            "SVPWM 이상 한계 1.1547 안이지만 L_g 전압강하·동적 전류 조절·과도·dead time 전압손실 여유가 필요하다. ③ η·PF는 운전점에 따라 변하므로 경계 계산에 쓴 값의 조건을 적는다. "
            "④ 요구 충돌(출력 vs 전류 한계)은 계산으로 해결하는 문제가 아니라 고객 결정 사항이다."
        ),
        customer_ko=(
            "400 V·16 A 조건에서 효율 97 %·PF 0.995면 배터리 출력은 약 10.70 kW입니다. 11 kW를 보증하려면 16.45 A가 필요하고 360 V low line에서는 18.3 A가 됩니다. "
            "출력 보증과 16 A 한계 중 우선순위, 그리고 440 V·700 V 코너의 변조 방식과 DC-link 전략을 먼저 정해 주시겠어요?"
        ),
        customer_en=(
            "At 400 V and 16 A, with 97 % efficiency and a 0.995 power factor, the battery output is about 10.7 kW. Guaranteeing 11 kW needs 16.45 A, and 18.3 A at a 360 V low line. "
            "Could we first agree which has priority, the output guarantee or the 16 A limit, and the modulation and DC-link strategy for the 440 V, 700 V corner?"
        ),
        questions=[_Q[0]],
        circuit="afe3",
        textbook=[TB_08],
        reference_presets=["nominal", "svpwm"],
        claim_limit="η·PF 고정의 해석 경계(A). 실제 효율·PF 곡선과 과변조 특성은 주장하지 않는다.",
    ),
    Experiment(
        key="pf_thd",
        title="PF·변위역률·THD: 정의와 측정 창",
        goal=(
            "합성 전압·전류로 PF = P/S, 변위역률 cosφ₁, THD(H = 40, 정수 10주기 창)를 계산하고 PF ≈ cosφ₁/√(1+THD²)가 성립하는 조건(정현 전압, DC 없음)을 확인한다. "
            "센서 offset·전압 왜곡·동기화되지 않은 창이 결과를 어떻게 바꾸는지 본다."
        ),
        params=[
            Param("V1", "전압 기본파 RMS", "V", 230.0, "V", vmin=1, vmax=2000, source="ASSUMED", source_note="단상 230 V"),
            Param("f", "실제 계통 주파수", "Hz", 50.0, "Hz", vmin=10, vmax=400, source="TEXTBOOK"),
            Param("f_an", "분석기가 가정한 주파수 (창 동기)", "Hz", 50.0, "Hz", vmin=10, vmax=400, source="ASSUMED", source_note="같으면 정수 주기 창", group="측정"),
            Param("I1", "전류 기본파 RMS", "A", 16.0, "A", vmin=0.01, vmax=5000, source="ASSUMED"),
            Param("phi1", "기본파 위상차 φ₁ (전류 지상 +)", "deg", 10.0, "deg", vmin=-90, vmax=90, source="ASSUMED"),
            Param("h3", "3차 전류 (% I₁)", "%", 0.0, "%", vmin=0, vmax=100, source="ASSUMED", group="고조파"),
            Param("h5", "5차 전류 (% I₁)", "%", 4.0, "%", vmin=0, vmax=100, source="ASSUMED", group="고조파"),
            Param("h7", "7차 전류 (% I₁)", "%", 3.0, "%", vmin=0, vmax=100, source="ASSUMED", group="고조파"),
            Param("idc", "전류 DC 성분 (센서 offset)", "A", 0.0, "A", vmin=-100, vmax=100, source="ASSUMED", group="고조파"),
            Param("v5", "전압 5차 (% V₁)", "%", 0.0, "%", vmin=0, vmax=20, source="ASSUMED", group="고조파"),
            Param("n_cyc", "창 길이 (분석기 가정 주기 수)", "", 10, "", vmin=1, vmax=50, kind="int", source="TEXTBOOK", source_note="교재: 정상상태 정수 10주기", group="측정"),
            Param("H", "고조파 상한 H", "", 40, "", vmin=2, vmax=100, kind="int", source="TEXTBOOK", source_note="H = 40", group="측정"),
            Param("M", "주기당 샘플", "", 512, "", vmin=128, vmax=4096, kind="int", source="ASSUMED", group="측정"),
        ],
        presets=[
            Preset("nominal", "φ₁ 10°, 5·7차 4/3 %", {}, "정의 확인", ("nominal", "reference")),
            Preset("offset", "센서 offset 0.5 A", {"idc": 0.5}, "근사식이 틀리는 조건", ("variant", "reference")),
            Preset("vdist", "전압 5차 3 %", {"v5": 3.0, "h5": 8.0}, "고조파 전력", ("variant",)),
            Preset("unsync", "계통 49 Hz, 분석기 50 Hz", {"f": 49.0}, "비정수 창 → leakage", ("failure", "reference")),
        ],
        run=run_pf_thd,
        model_level="A (합성 파형, DFT)",
        suggested_change="전류 DC 성분(센서 offset)을 0 → 0.5 A로 넣어 근사식과 실제 PF를 비교한다.",
        prediction=Prediction(
            "전류에 0.5 A DC offset이 생기면 THD(H ≤ 40)와 실제 PF는?",
            ["둘 다 그대로", "THD는 그대로, PF는 낮아진다", "THD가 커지고 PF는 그대로", "모르겠다"],
            "THD는 그대로, PF는 낮아진다",
            "THD 정의는 h = 2..H만 쓰므로 DC가 들어가지 않는다. 하지만 DC는 I_rms를 키우고(정현 전압과의 평균전력은 0) PF를 낮춘다. 그래서 cosφ₁/√(1+THD²)가 실제 PF보다 커진다.",
            ["thd", "pf_true", "pf_formula"],
            handcalc=[{"key": "thd", "label": "THD (5·7차 4/3 %)", "unit": "%"}, {"key": "pf_true", "label": "PF", "unit": ""}],
        ),
        suggested={"idc": 0.5},
        student="PF는 ‘실제로 일한 전력 / 전압·전류 RMS의 곱’이다. 위상이 맞아도 전류 모양이 찌그러지면 RMS가 커져 PF가 1보다 작아진다.",
        expert=(
            "보고 항목: 창(정상상태 정수 주기, 동기화 방법), 기본파 추정, H, DC·센서 offset, 기동 제외 조건. 비정수 창의 leakage는 고조파처럼 보인다. "
            "전압 왜곡이 있으면 고조파 전력이 생겨 식이 근사가 된다. 규격 판정은 해당 표준의 측정 조건으로 따로 한다."
        ),
        customer_ko="THD를 비교하려면 창 길이·동기화·H와 센서 offset 처리를 같은 조건으로 맞춰야 합니다. 측정 설정을 공유해 주시면 같은 정의로 다시 계산해 보겠습니다.",
        customer_en="To compare THD numbers we need the same window, synchronisation, harmonic limit and offset handling. If you share the analyser settings, I will recompute with the same definition.",
        questions=[_Q[1]],
        circuit="meter",
        textbook=[TB_08],
        reference_presets=["nominal", "offset", "unsync"],
        claim_limit="합성 파형의 정의 계산(A). 규격 적합성·실측 THD는 주장하지 않는다.",
    ),
    Experiment(
        key="boost_pfc",
        title="단상 boost PFC 스위칭 모델: THD는 폐루프 전류에서 계산한다",
        goal=(
            "diode bridge + boost + DC-link를 정확 스위칭 엔진으로 풀고(계통 = 정확한 발진기 상태, bridge 극성은 그 상태의 영교차 사건), 샘플링 평균전류제어로 폐루프를 닫는다. "
            "THD와 PF를 교재 정의(H = 40, 정상상태 정수 10주기)로 시뮬레이션 전류에서 계산하고, 평균모델(B)과 THD = 0인 강제 정현파를 비교하며 DC-link 2ω 리플을 P/(ωCV)와 대조한다."
        ),
        params=[
            Param("V_rms", "계통 전압 RMS", "V", 230.0, "V", vmin=50, vmax=300, source="ASSUMED", source_note="단상 230 V", validity_note="boost: V_pk < V_ref"),
            Param("f", "계통 주파수", "Hz", 50.0, "Hz", vmin=40, vmax=70, source="TEXTBOOK"),
            Param("P", "출력 전력 (저항부하 정의)", "W", 3300.0, "W", vmin=100, vmax=7400, source="ASSUMED", source_note="3.3 kW 단상 OBC급"),
            Param("Vref", "DC-link 기준", "V", 400.0, "V", vmin=380, vmax=450, source="ASSUMED"),
            Param("L", "boost 인덕턴스", "H", 1e-3, "mH", vmin=100e-6, vmax=10e-3, source="ASSUMED"),
            Param("RL", "인덕터 DCR", "Ω", 0.05, "mΩ", vmin=0, vmax=1, source="ASSUMED", group="비이상"),
            Param("C", "DC-link C", "F", 1e-3, "mF", vmin=100e-6, vmax=10e-3, source="ASSUMED"),
            Param("fs", "스위칭 주파수 (f의 정수배)", "Hz", 20e3, "kHz", vmin=5e3, vmax=50e3, source="ASSUMED", source_note="20 kHz = 400 × 50 Hz"),
            Param("fci", "전류 루프 crossover", "Hz", 1000.0, "Hz", vmin=100, vmax=5000, source="ASSUMED", group="제어"),
            Param("zi", "전류 PI 영점 비 f_ci/f_z", "", 5.0, "", vmin=1, vmax=100, source="ASSUMED", source_note="정류 사인 기준 추종을 위해 저주파 이득 확보", group="제어"),
            Param("fcv", "전압 루프 crossover", "Hz", 5.0, "Hz", vmin=0.5, vmax=40, source="ASSUMED", source_note="2ω(100 Hz)보다 충분히 낮게", group="제어"),
            Param("dmax", "duty 상한", "", 0.98, "", vmin=0.5, vmax=0.999, source="ASSUMED", group="제어"),
            Param("n_cyc", "THD 창 (정수 주기)", "", 10, "", vmin=1, vmax=20, kind="int", source="TEXTBOOK", source_note="정상상태 정수 10주기", group="측정"),
            Param("H", "고조파 상한 H", "", 40, "", vmin=3, vmax=60, kind="int", source="TEXTBOOK", source_note="H = 40", group="측정"),
            Param("n_b", "평균모델 사전 수렴 주기", "", 24, "", vmin=2, vmax=200, kind="int", source="ASSUMED", group="측정"),
            Param("crit", "정상상태 기준 (주기별 변화)", "", 1e-4, "", vmin=1e-8, vmax=1e-2, source="ASSUMED", group="측정"),
            Param("max_settle", "C 모델 최대 정착 주기", "", 20, "", vmin=2, vmax=60, kind="int", source="ASSUMED", source_note="기준을 못 채우면 그대로 표시", group="측정"),
        ],
        presets=[
            Preset("nominal", "3.3 kW, 전압 루프 5 Hz", {}, "합성 설계", ("nominal", "reference")),
            Preset("fast_v", "전압 루프 20 Hz", {"fcv": 20.0}, "2ω 리플이 전류 기준에 들어감", ("variant", "reference")),
            Preset("light", "600 W (경부하)", {"P": 600.0}, "영교차 부근 DCM", ("corner", "reference")),
            Preset("small_C", "C = 470 µF", {"C": 470e-6}, "2ω 리플 증가", ("variant",)),
        ],
        run=run_boost_pfc,
        model_level="C (정확 스위칭) + B 평균모델 + A 강제 정현파",
        suggested_change="전압 루프 crossover를 5 Hz → 20 Hz로 올린다 (다음에는 출력 600 W).",
        prediction=Prediction(
            "전압 루프 대역을 5 Hz에서 20 Hz로 올리면 THD는?",
            ["줄어든다 — 더 빠른 제어", "3차 고조파가 커져 늘어난다", "변화 없다", "모르겠다"],
            "3차 고조파가 커져 늘어난다",
            "DC-link에는 P/(ωCV)의 2ω 리플이 있다. 전압 루프 이득이 크면 그 리플이 전력 명령에 들어가 전류 기준의 진폭을 2ω로 흔들고, 정현파 × 2ω 변조는 3차 고조파를 만든다.",
            ["thd", "h3", "dV2w"],
            handcalc=[{"key": "dV2w", "label": "2ω 리플 P/(ωCV)", "unit": "V"}, {"key": "I1", "label": "기본파 RMS ≈ P/V", "unit": "A"}],
        ),
        suggested={"fcv": 20.0},
        student=(
            "PFC는 전류를 전압과 같은 모양(정현파)으로 끌어오도록 duty를 매 주기 바꾼다. 그런데 DC-link 전압에는 100 Hz 리플이 있고, 전압 루프가 그 리플을 따라가면 전류 크기 명령이 100 Hz로 흔들려 전류 모양이 찌그러진다. "
            "영교차 근처에서는 입력 전압이 너무 작아 전류를 만들 수 없는 짧은 구간도 생긴다."
        ),
        expert=(
            "① THD는 폐루프 전류(스위칭 또는 평균)에서, 정상상태 정수 주기 창·H·기본파 추정을 기록하고 계산한다. 강제 정현파의 THD = 0은 구성의 결과다. "
            "② 3차 고조파는 전압 루프 대역과 2ω 리플의 곱으로 정해진다: 대역을 낮추거나 notch·리플 feedforward를 쓴다. ③ 영교차 왜곡: duty 상한·샘플링 지연·DCM, 실제로는 bridge 역회복과 입력 필터 커패시터 전류도 더해진다. "
            "④ 스위칭 리플까지 포함한 전체 왜곡은 H = 40 THD와 다른 양이며 EMI 필터가 다룬다. ⑤ 평균모델(B)은 CCM에서만 스위칭 해의 주기 평균과 맞는다."
        ),
        customer_ko=(
            "이 합성 설계의 THD는 약 3 %이고 대부분 3차 고조파입니다. 전압 루프가 DC-link 100 Hz 리플을 전류 기준에 넣기 때문이라 루프 대역·notch·리플 feedforward로 줄일 수 있습니다. "
            "비교는 같은 창(정상상태 10주기, H = 40)과 같은 부하 조건에서 해 주시고, 강제 정현파 모델 결과는 성능 근거로 쓰지 않겠습니다."
        ),
        customer_en=(
            "In this synthetic design the THD is about 3 %, mostly third harmonic, because the voltage loop feeds the 100 Hz DC-link ripple into the current reference. Lowering the loop bandwidth or adding a notch or ripple feedforward reduces it. "
            "Please compare with the same window, ten steady-state cycles and H of 40, at the same load; a forced-sine model result is not evidence of performance."
        ),
        questions=[_Q[2]],
        circuit="boost_pfc",
        textbook=[TB_08, TB_09],
        reference_presets=["nominal", "fast_v", "light"],
        runtime_hint="seconds",
        claim_limit="이상 소자·합성 제어기의 스위칭 모델(C) THD/PF. 규격 적합성·EMI·소자 손실은 주장하지 않는다.",
    ),
    Experiment(
        key="dclink_power",
        title="단상 2ω 전력과 3상 순시전력, 그리고 DC-link를 정하는 서로 다른 이유",
        goal=(
            "단상 p = P(1 − cos2ωt)와 평형 3상의 일정한 합성 순시전력을 파형 곱으로 확인하고, 7.4 kW·50 Hz·400 V·20 Vpp의 2ω 리플 용량 약 2.944 mF와 "
            "11 kW·2 ms·800→760 V hold-up 최소 705 µF를 에너지 ODE로 검증한다. 1 mF를 800→760 V로 쓰면 31.2 J, 저장에너지의 9.75 %다."
        ),
        params=[
            Param("f", "계통 주파수", "Hz", 50.0, "Hz", vmin=10, vmax=400, source="TEXTBOOK"),
            Param("P1", "단상 예 전력", "W", 7400.0, "kW", vmin=10, vmax=1e5, source="TEXTBOOK", source_note="7.4 kW", group="순시전력"),
            Param("P3", "3상 예 전력", "W", 11e3, "kW", vmin=10, vmax=1e6, source="TEXTBOOK", source_note="11 kW", group="순시전력"),
            Param("V_LL", "3상 선간전압", "V", 400.0, "V", vmin=50, vmax=2000, source="TEXTBOOK", group="순시전력"),
            Param("unb", "c상 전류 감소 (불평형)", "%", 0.0, "%", vmin=0, vmax=100, source="ASSUMED", group="순시전력"),
            Param("P_rip", "2ω 리플 sizing 전력", "W", 7400.0, "kW", vmin=10, vmax=1e5, source="TEXTBOOK", source_note="7.4 kW", group="2ω 리플"),
            Param("V_rip", "DC-link 전압", "V", 400.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="400 V", group="2ω 리플"),
            Param("dV_pp", "허용 리플 pp", "V", 20.0, "V", vmin=0.1, vmax=200, source="TEXTBOOK", source_note="20 Vpp", group="2ω 리플"),
            Param("P_hold", "hold-up 전력", "W", 11e3, "kW", vmin=10, vmax=1e6, source="TEXTBOOK", source_note="11 kW", group="hold-up"),
            Param("dt_hold", "hold-up 시간", "s", 2e-3, "ms", vmin=1e-5, vmax=1.0, source="TEXTBOOK", source_note="2 ms", group="hold-up"),
            Param("V_hi", "시작 전압", "V", 800.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="800 V", group="hold-up"),
            Param("V_lo", "허용 최저 전압", "V", 760.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="760 V", group="hold-up"),
            Param("C_given", "검토할 C", "F", 1e-3, "µF", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="1 mF (02장 확인 문제)", group="hold-up"),
        ],
        presets=[
            Preset("nominal", "교재 수치", {}, "08장·02장", ("nominal", "reference")),
            Preset("unbalance", "3상 불평형 10 %", {"unb": 10.0}, "2ω 리플이 남는다", ("variant", "reference")),
            Preset("small_c", "C = 470 µF로 hold-up", {"C_given": 470e-6}, "hold-up 부족", ("failure", "reference")),
        ],
        run=run_dclink,
        model_level="A (+ B 에너지 ODE)",
        suggested_change="3상 불평형을 0 → 10 %로 바꾼다 (다음에는 검토할 C를 470 µF로).",
        prediction=Prediction(
            "평형 3상 11 kW 입력의 합성 순시전력은?",
            ["단상처럼 0~2P로 출렁인다", "일정하다", "3ω로 출렁인다", "모르겠다"],
            "일정하다",
            "각 상의 2ω 성분이 120°씩 어긋나 합이 0이 된다. 그래서 평형 3상에는 단상의 P/(ωCV) sizing을 기본으로 쓰지 않는다. 불평형이면 역상분 전류만큼 2ω가 남는다.",
            ["p3_ripple", "p1_ripple"],
            handcalc=[{"key": "C_rip", "label": "2ω 리플 C", "unit": "F"}, {"key": "C_hold", "label": "hold-up 최소 C", "unit": "F"}, {"key": "dE", "label": "1 mF 800→760 V 에너지", "unit": "J"}],
        ),
        suggested={"unb": 10.0},
        student=(
            "단상에서는 전압과 전류가 같이 0을 지나므로 그 순간 전력도 0이고, peak에서는 평균의 두 배다. 배터리 쪽은 일정한 전력을 원하므로 그 차이를 DC-link 커패시터가 채우고 비워 100 Hz 리플이 생긴다. "
            "3상은 한 상이 0일 때 다른 상이 보충해 합이 일정하다."
        ),
        expert=(
            "① 2ω 리플 식은 단상·후단 전력 일정·작은 리플 근사다(정확해는 에너지 식). ② hold-up은 에너지 차로 정하고, 그동안 PFC 공급·제어 응답·ESR 강하를 넣으면 달라진다. "
            "③ 불평형 3상의 2ω 진폭은 3V⁺I⁻(RMS)로 대칭좌표에서 바로 나온다. ④ ESR 발열·수명·제어 최소 용량은 데이터가 필요한 별도 기준이다(MISSING_INPUT)."
        ),
        customer_ko=(
            "단상 7.4 kW에서 20 Vpp 리플을 원하시면 약 2.9 mF가 필요하지만, 3상 평형 운전에는 이 식이 기본 sizing이 아닙니다. hold-up 요구(11 kW, 2 ms, 800→760 V)는 최소 705 µF의 별도 기준이니 "
            "두 요구와 수명·발열 조건을 각각 확인해 결정하시죠."
        ),
        customer_en=(
            "For 7.4 kW single-phase with 20 V peak-to-peak ripple you need about 2.9 mF, but that formula does not size a balanced three-phase link. The hold-up requirement, 11 kW for 2 ms from 800 to 760 V, is a separate minimum of 705 µF, "
            "so let's check each requirement, plus lifetime and heating, separately."
        ),
        questions=[_Q[3], _Q[4]],
        circuit="dclink",
        textbook=[TB_08, TB_02],
        reference_presets=["nominal", "unbalance", "small_c"],
        claim_limit="이상 정현파·정전력 에너지 모델(A/B). 커패시터 수명·ESR 발열은 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL05",
    title="OBC·PFC — 전력품질을 설계 변수로 바꾸기",
    title_en="OBC and PFC: power quality as a design variable",
    track="basic",
    order=5,
    path_note="14일 경로 4일차 (08 OBC/PFC, FL05)",
    textbook=[TB_08, TB_02],
    prerequisites=["FL01"],
    summary="입력 경계(16.45 A, 16 A→10.70 kW, 변조지수) → PF·THD 정의와 창 → 폐루프 boost PFC 스위칭 모델의 THD/PF → 단상 2ω와 3상 순시전력, DC-link sizing.",
    experiments=EXPERIMENTS,
    minimum_scope="단상/3상 PFC·DC-link sizing; 16.450 A·10.699 kW·PF/THD 정의·energy (교재 19장 표); 3상 상/선간전압과 단상 2ω 커패시터 문제 분리; 강제 정현파 THD = 0을 PFC 성능으로 주장하지 않음",
    claim_limits=[
        "η·PF 고정의 해석 경계(A)",
        "THD/PF는 이상 소자·합성 제어기·정의된 창의 모델 결과 — 규격 적합성·EMI 판정 아님",
        "강제 정현파 모델의 THD = 0은 구성의 결과이며 PFC 성능이 아님",
        "스위칭 파형에 임의 사인 리플을 더한 값을 측정 THD처럼 보고하지 않음",
        "DC-link 수명·ESR 발열은 데이터 필요 (MISSING_INPUT)",
    ],
    test_paths=["tests/test_fl05.py"],
)
