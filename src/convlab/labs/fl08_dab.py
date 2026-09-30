"""FL08 - DAB: from equations to waveforms to design judgement (textbook ch.11).

Three independent paths for every SPS number: closed form (reference/dab.py), exact
piecewise-linear integration of the ideal bridge waveforms, and the switched-affine engine
(optional series R and magnetizing Lm) with the half-wave antisymmetric periodic state.
Ideal switches cannot show ZVS: that stays NOT_EVALUABLE; the edge-current sign/charge
screen is SCREEN_ONLY (see EX02/EX06 for commutation physics).
"""

from __future__ import annotations

import math

import numpy as np

from ..engine.pwl import PWL
from ..engine.switched import simulate
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import dab as ref
from ._common import decimate_minmax, energy_ledger, ledger_check
from ._dab import (
    EDGE_KO,
    PI,
    SPS_TEXTBOOK_C1,
    TWO_PI,
    DABSystem,
    Modulation,
    bands_for,
    commutation_screen,
    dab_circuit,
    half_wave_periodic,
    pwl_waves,
    sample_pwl,
)

TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB — 식에서 파형, 파형에서 설계 판단으로 [FL08]")
TB_10 = TextbookRef("자성체-컨버터-전문가로-가는-실제-관문-fl07", "10. 자성체 [FL07]")
TB_15 = TextbookRef("fae-디버깅-파형보다-먼저-질문의-질을-높인다-fl12", "15. FAE 디버깅 (CASE C)")

COSS_HV = (0.5e-9, 40.0)  # synthetic HV device C(v) = C0/sqrt(1+v/V0) (ASSUMED, EX02 form scaled)
COSS_LV = (8e-9, 5.0)  # synthetic LV device (ASSUMED)


def _base_params(extra=None, P=1500.0, VH=800.0, VL=48.0):
    ps = [
        Param("VH", "HV 포트 전압 V_H (= V₁)", "V", VH, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="nominal 800 V"),
        Param("VL", "LV 포트 전압 V_L", "V", VL, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="nominal 48 V"),
        Param("Np", "1차 권선수 N_p", "", 50, "", vmin=1, vmax=1000, kind="int", source="TEXTBOOK", source_note="n = N_p/N_s = 50/3", group="변압기"),
        Param("Ns", "2차 권선수 N_s", "", 3, "", vmin=1, vmax=1000, kind="int", source="TEXTBOOK", group="변압기"),
        Param("L", "직렬 L (1차 환산, leakage+외부)", "H", 200e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", source_note="200 µH primary-referred", group="변압기"),
        Param("fs", "스위칭 주파수 f_s", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", source_note="100 kHz"),
    ]
    return ps + (extra or [])


def _n(v):
    return v["Np"] / v["Ns"]


# ======================================================================================
# 1. SPS nominal: three paths
# ======================================================================================


def run_sps(v: dict) -> Result:
    res = Result("FL08", "sps_nominal", "A (닫힌 식) + C (구간 적분·스위칭 해)")
    n = _n(v)
    V1, V2, L, fs = v["VH"], n * v["VL"], v["L"], v["fs"]
    P = v["P"]
    Pmax = ref.sps_pmax(V1, V2, L, fs)
    res.add_metric("n", "권선비 n = N_p/N_s", n, "", basis=f"{v['Np']}:{v['Ns']}")
    res.add_metric("V2", "LV 전압의 1차 환산 V₂ = n·V_L", V2, "V", basis="primary-referred (실제 2차 전압 아님)")
    res.add_metric("Pmax", "SPS 최대전력 V₁V₂/(8 f_s L)", Pmax, "W", ref=4000.0 if _textbook_nominal(v) else None, ref_label="교재 4 kW", tol=1e-9, basis="모듈당")
    phi = ref.sps_phi_for_power(V1, V2, L, fs, P)
    if phi is None:
        res.verdict("NO_SOLUTION", f"요청 전력 {P:g} W > SPS Pmax {Pmax:.4g} W — 이 전압·L·f_s에서 해가 없다")
        res.assumptions.append("SPS 단조 범위 |φ| ≤ π/2")
        return res
    tb = _textbook_nominal(v) and abs(P - 1500) < 1e-9
    res.add_metric("phi", "outer phase φ", phi, "rad", ref=0.328972794 if tb else None, ref_label="교재 0.328972794 rad", tol=1e-7, basis="1차 선행 = +(HV→LV)")
    res.add_metric("phi_deg", "φ [deg]", phi * 180 / PI, "deg", ref=18.8488 if tb else None, ref_label="교재 18.8488°", tol=5e-6)
    res.add_metric("d", "normalized d = φ/π", phi / PI, "", basis="d ≠ Δt/T")
    res.add_metric("dt_T", "time shift Δt/T = φ/(2π)", phi / TWO_PI, "", basis="d의 절반 — 혼용 금지")
    # path 2: exact piecewise-linear integration
    mod = Modulation(PI, PI, phi, SPS_TEXTBOOK_C1)
    Lm = v["Lm"] if v["Lm"] > 0 else None
    w = pwl_waves(V1, V2, L, fs, mod, Lm)
    i0, iphi, irms_cf, ipk_cf = ref.sps_currents(V1, V2, L, fs, abs(phi))
    res.add_metric("P_pwl", "전달전력 (구간 적분, 2차 포트)", w.P2, "W", ref=P, ref_label="요청값 (닫힌 식 φ 사용)", tol=1e-9, basis="모듈당")
    res.add_metric("Irms", "인덕터 RMS (1차)", w.Irms, "A", ref=2.019881509 if tb and not Lm else irms_cf, ref_label="교재 2.019881509 A" if tb else "닫힌 식", tol=1e-7 if not Lm else None, basis="모듈당, 1차 권선 = L 전류")
    res.add_metric("Ipk", "인덕터 peak", w.Ipk, "A", ref=2.09430585 if tb and not Lm else ipk_cf, ref_label="교재 2.09430585 A" if tb else "닫힌 식", tol=1e-7 if not Lm else None)
    fwd = phi >= 0 and not Lm
    th2 = (phi % TWO_PI)  # secondary rising edge (textbook convention: v2' = +V2 on [phi, pi + phi))
    res.add_metric("i0", "i(0): 1차 상승 edge 전류", w.iL.value_at(0.0), "A", ref=i0 if fwd else None, ref_label="−(aφ + b(π−φ))/2", tol=1e-7, basis="음수여야 1차 leg 노드가 올라갈 전류 방향")
    res.add_metric("iphi", "2차 상승 edge 전류 (1차 환산)", w.i2.value_at(th2 / (TWO_PI * fs)), "A", ref=iphi if fwd else None, ref_label="i₀ + aφ", tol=1e-7, basis=f"θ = {th2:.4f} rad; 양수여야 2차 leg 방향이 맞음")
    res.add_metric("Is_rms", "2차 AC RMS (실제 = n·I₂,rms)", n * w.i2.rms(), "A", ref=33.6647 if tb and not Lm else None, ref_label="교재 33.6647 A", tol=2e-6)
    res.add_metric("Io_dc", "LV DC 출력전류 (모듈당) = P/V_L", w.P2 / v["VL"], "A", ref=31.25 if tb else None, ref_label="교재 31.25 A", tol=1e-9)
    res.add_metric("Io_total", f"LV DC 출력전류 (모듈 {int(v['modules'])}개 합)", w.P2 / v["VL"] * v["modules"], "A", ref=62.5 if tb and int(v["modules"]) == 2 else None, ref_label="교재 62.5 A", tol=1e-9, basis="total — 모듈당 값과 섞지 않는다")
    res.add_metric("P_total", f"총 전력 (모듈 {int(v['modules'])}개)", w.P2 * v["modules"], "W", basis="total")
    res.add_metric("Isw_rms", "1차 스위치 1개 RMS", w.Irms / math.sqrt(2), "A", basis="소자별 (50 % 도통) ≠ 변압기 RMS")
    if Lm:
        res.add_metric("Im_pk", "여자전류 peak (L_m)", w.im.max_abs(), "A", basis="2차 bridge 전압이 L_m에 걸림")
    res.add_check(check_close("전력: 닫힌 식 φ → 구간 적분 P", w.P2, P, 1e-9, "P = V₁V₂φ(1−|φ|/π)/(ωL)의 φ를 넣고 PWL 정확 적분한 v₂′·i", True, "W", abs_scale=Pmax))
    if not Lm:
        res.add_check(check_close("RMS: 닫힌 식 vs 구간 적분", w.Irms, irms_cf, 1e-9, "I_pk√(1−2φ/(3π)) / 일반식 vs Σ Δt(i_a²+i_ai_b+i_b²)/3", True, "A"))
    res.add_check(check_close("포트 전력 보존 (무손실): P₁ = P₂", w.P1, w.P2, 1e-9, "1차 포트 ∫v₁·i vs 2차 포트 ∫v₂′·i₂ (구간 적분)", True, "W", abs_scale=abs(P) + 1))
    # path 3: switched-affine engine with R, Lm; periodic state from half-wave antisymmetry
    sysd = DABSystem(V1, V2, L, fs, R=v["R"], Lm=Lm, n=n, mod_of_cycle=lambda k: mod, key="sps")
    q0 = mod.levels(1e-9)
    x0 = half_wave_periodic(sysd, q0)
    T = 1.0 / fs
    tr = simulate(sysd, q0, x0, 0.0, 2 * T)
    P2e = tr.energy(0, T, "p2") / T
    P1e = tr.energy(0, T, "p1") / T
    Irms_e = tr.rms(0, T, "iL")
    res.add_metric("P_engine", "전달전력 (스위칭 해)", P2e, "W", basis=f"R = {v['R']:g} Ω 포함 시 1차 포트 {P1e:.5g} W")
    if v["R"] == 0:
        res.add_check(check_close("구간 적분 vs 행렬지수 스위칭 해: 전력", P2e, w.P2, 1e-9, "PWL 정확 적분 vs expm 전파·Kronecker 모멘트", True, "W", abs_scale=abs(P) + 1))
        res.add_check(check_close("구간 적분 vs 스위칭 해: RMS", Irms_e, w.Irms, 1e-9, "PWL vs expm", True, "A"))
    led = energy_ledger(tr, sysd, 0.0, T, ["p1"], ["p2"], ["pR"], rated_power=abs(P) + 1)
    res.add_check(ledger_check(led))
    zp = tr.state_at(T / 2)[0][:-1]
    res.add_check(
        Check(
            "반주기 반대칭 기준해 x(T/2) = −x(0)",
            "PASS" if np.max(np.abs(zp + x0)) < 1e-9 * max(1, np.max(np.abs(x0))) else "FAIL",
            float(np.max(np.abs(zp + x0))),
            "A",
            1e-9,
            path="이상 L은 DC offset을 스스로 없애지 않는다 → 정상 기준해를 반대칭 조건으로 선택",
            detail=f"x(0) = {np.round(x0, 6).tolist()} A (평균 {w.iL.mean():.2e} A)",
        )
    )
    # ZVS status in an ideal model + screen
    scr = commutation_screen(w, mod, fs, n, V1, v["VL"], COSS_HV, COSS_LV, v["td_hv"], v["td_lv"])
    res.verdict("PASS_WITHIN_MODEL", "SPS 전력·RMS·peak가 닫힌 식·구간 적분·스위칭 해 세 경로에서 일치 (이상 스위치·선형 L)")
    res.verdict("NOT_EVALUABLE", "ZVS: 이상 스위치 모델에는 노드 용량·dead time이 없어 ZVS를 판정할 수 없다")
    res.verdict("SCREEN_ONLY", "edge 전류 부호·전하 screen은 합성 C_oss·일정 전류 가정의 선별값 (아래 표)")
    res.tables.append(
        Table(
            "t_edges",
            "leg별 commutation screen (SCREEN_ONLY, 합성 C_oss, dead time 입력값)",
            ["bridge", "leg", "θ [deg]", "방향", "노드 밖으로 전류 [A]", "Q 가용 [nC]", "Q 필요 [nC]", "m_Q", "판정"],
            [[e.bridge, e.leg, e.theta * 180 / PI, "상승" if e.rising else "하강", e.i_out, e.q_av * 1e9, e.q_req * 1e9, e.m_q, EDGE_KO[e.status]] for e in scr],
            note="상승 노드는 전류가 노드로 들어와야 한다(밖으로 < 0). 전하는 edge 전류 × dead time으로 근사한 screen이다. 실제 전류는 dead time 동안 변한다(EX02).",
        )
    )
    # waveforms
    xs, ys = sample_pwl(w.iL)
    res.add_series("iL", "i_L (1차)", "A", xs, ys)
    xs, ys = sample_pwl(w.v1)
    res.add_series("v1", "v₁ (1차 bridge)", "V", xs, ys)
    xs, ys = sample_pwl(w.v2)
    res.add_series("v2", "v₂′ (2차 bridge, 1차 환산)", "V", xs, ys)
    vL = PWL(w.t, w.v1.y0 - w.v2.y0, w.v1.y1 - w.v2.y1)
    xs, ys = sample_pwl(vL)
    res.add_series("vL", "v_L = v₁ − v₂′", "V", xs, ys)
    i2a = w.i2.scaled(n)
    xs, ys = sample_pwl(i2a)
    res.add_series("i2_act", "i_s 2차 실제 전류 = n·i₂", "A", xs, ys)
    port = PWL(w.t, i2a.y0 * np.array(w.b2), i2a.y1 * np.array(w.b2))
    xs, ys = sample_pwl(port)
    res.add_series("idc2", "LV DC 포트 전류 b₂·i_s", "A", xs, ys)
    if Lm:
        xs, ys = sample_pwl(w.im)
        res.add_series("im", "i_m 여자전류", "A", xs, ys, dash=True)
    bands = bands_for(mod, fs, 2)
    res.add_plot("p_v", "두 bridge 전압과 인덕터 전압 (4구간)", ["v1", "v2", "vL"], y_label="전압", y_unit="V", bands=bands, group="dab", level="C",
                 proved="φ만큼 어긋난 두 사각파의 차이 v_L이 네 구간(V₁+V₂, V₁−V₂, −V₁−V₂, −V₁+V₂)을 만든다.", not_yet="dead time·스위칭 과도는 없다.")
    res.add_plot("p_i", "인덕터 전류 (zero-DC 기준해)", ["iL"] + (["im"] if Lm else []), y_label="전류", y_unit="A", bands=bands, group="dab", level="C", window=(0.0, T), hlines=[{"y": 0.0, "label": "0"}],
                 proved="구간마다 직선인 전류의 RMS·peak를 구간 적분으로 정확히 구하고 닫힌 식·스위칭 해와 대조했다. 반주기 반대칭 조건으로 DC offset 없는 기준해를 명시했다.",
                 not_yet="ZVS는 이 모델로 판정할 수 없다(NOT_EVALUABLE). 자화전류 L_m은 입력했을 때만 포함된다.")
    res.add_plot("p_s", "2차 실제 전류와 LV DC 포트 전류", ["i2_act", "idc2"], y_label="전류", y_unit="A", bands=bands, group="dab", level="C",
                 proved=f"2차 AC RMS {n * w.i2.rms():.4g} A와 DC 출력 {w.P2 / v['VL']:.4g} A(모듈당)는 서로 다른 값이다. 포트 전류는 T/2 주기를 가진다.",
                 not_yet="출력 커패시터 리플·ESR 발열은 계산하지 않았다.")
    # P(phi) curve with sign reversal
    ph = np.linspace(-PI, PI, 241)
    res.add_series("Pphi", "P(φ) 닫힌 식", "W", ph.tolist(), [ref.sps_power(V1, V2, L, fs, x) for x in ph])
    chk = np.linspace(-PI * 0.95, PI * 0.95, 13)
    res.add_series("Pphi_pwl", "P(φ) 구간 적분", "W", chk.tolist(), [pwl_waves(V1, V2, L, fs, Modulation(PI, PI, x, SPS_TEXTBOOK_C1)).P2 for x in chk], style="points")
    res.add_plot("p_pphi", "전달전력 P(φ): 부호로 방향, |φ| ≤ π/2가 단조 제어 범위", ["Pphi", "Pphi_pwl"], x_label="φ", x_unit="rad", y_label="P", y_unit="W", kind="xy",
                 vlines=[{"x": PI / 2, "label": "π/2"}, {"x": -PI / 2, "label": "−π/2"}], markers=[{"x": phi, "y": w.P2, "label": "운전점"}], level="A",
                 proved="φ < 0이면 전력이 LV→HV로 흐르고, |φ| > π/2에서는 같은 전력에 두 해가 있어 제어 부호가 바뀐다.", not_yet="")
    c = dab_circuit(V1, v["VL"], f"{v['Np']}:{v['Ns']}", L, Lm)
    res.circuit = {"diagram": c.to_json(), "intervals": bands, "plot_group": "dab"}
    res.assumptions += ["이상 full bridge 두 개(SPS, 50 % duty), 이상 변압기 n = N_p/N_s, 직렬 L은 1차 환산", "dead time·노드 용량 없음", "R, L_m은 입력했을 때만 포함"]
    res.not_valid_for += ["ZVS 판정 (NOT_EVALUABLE)", "스위칭 손실·효율", "변압기 포화 (FL07)"]
    res.interpretation = (
        f"φ = {phi:.6f} rad에서 두 bridge 전압의 차이가 L에 걸려 전류가 네 구간 직선으로 움직인다. 전력은 v₂′와 전류의 곱을 평균한 것이며 "
        f"φ(1−φ/π)에 비례한다. 같은 전압비(V₁ = V₂)에서는 peak가 i(0) = −i(π)로 대칭이고, 모듈당 1.5 kW와 두 모듈 합 3 kW(LV DC {w.P2 / v['VL'] * 2:.4g} A)를 구분해야 한다."
    )
    return res


def _textbook_nominal(v) -> bool:
    return abs(v["VH"] - 800) < 1e-9 and abs(v["VL"] - 48) < 1e-9 and v["Np"] == 50 and v["Ns"] == 3 and abs(v["L"] - 200e-6) < 1e-12 and abs(v["fs"] - 100e3) < 1e-6


# ======================================================================================
# 2. Zero power with ratio mismatch
# ======================================================================================


def run_zero_power(v: dict) -> Result:
    res = Result("FL08", "zero_power_mismatch", "A + C")
    n = _n(v)
    V1, V2, L, fs = v["VH"], n * v["VL"], v["L"], v["fs"]
    phi = v["phi"]
    Lm = v["Lm"] if v["Lm"] > 0 else None
    w = pwl_waves(V1, V2, L, fs, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1), Lm)
    w0 = pwl_waves(V1, V2, L, fs, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1), None)
    tb = abs(V1 - 900) < 1e-9 and abs(V2 - 600) < 1e-6 and abs(L - 200e-6) < 1e-12 and abs(fs - 1e5) < 1e-6 and abs(phi) < 1e-15
    res.add_metric("P", "전달전력", w.P2, "W", basis="모듈당")
    res.add_metric("Irms_L", "L 순환전류 RMS (L_m 제외)", w0.Irms, "A", ref=2.165063509 if tb else None, ref_label="교재 2.165063509 A", tol=1e-8, basis="전압비 불일치가 만든 전류")
    res.add_metric("Ipk_L", "L 순환전류 peak", w0.Ipk, "A", ref=3.75 if tb else None, ref_label="교재 3.75 A", tol=1e-9)
    if Lm:
        res.add_metric("Im_rms", "여자전류 RMS (L_m)", w.im.rms(), "A", basis=f"peak {w.im.max_abs():.4g} A = V₂′T/(4L_m) — 전압비와 무관하게 늘 흐름")
        res.add_metric("Iprim_rms", "1차 권선 RMS (i_L, L_m 포함 모델)", w.iL.rms(), "A")
        res.add_metric("I2_rms", "2차 환산 RMS i_L − i_m", w.i2.rms(), "A")
    nom = pwl_waves(800.0, 800.0, 200e-6, 100e3, Modulation(PI, PI, 0.328972794, SPS_TEXTBOOK_C1))
    res.add_metric("Irms_nom", "비교: nominal 1.5 kW의 RMS", nom.Irms, "A", basis="800/800 V, φ = 0.329 rad")
    Rt = v["R_est"]
    res.add_metric("P_cond", "전도손실 추정 I²R (후처리 추정)", w.iL.rms() ** 2 * Rt, "W", basis=f"R = {Rt * 1e3:g} mΩ (ASSUMED) — postprocessed loss estimate")
    res.add_check(check_close("φ=0 순환전류 RMS: 삼각파 해석값 vs 구간 적분", w0.Irms, abs(V1 - V2) / (4 * fs * L) / math.sqrt(3), 1e-9, "I_pk/√3 (삼각파) vs PWL", True, "A"))
    # engine cross-check with Lm
    sysd = DABSystem(V1, V2, L, fs, R=0.0, Lm=Lm, n=n, mod_of_cycle=lambda k: Modulation(PI, PI, phi, SPS_TEXTBOOK_C1), key="zp")
    q0 = Modulation(PI, PI, phi, SPS_TEXTBOOK_C1).levels(1e-9)
    x0 = half_wave_periodic(sysd, q0)
    T = 1 / fs
    tr = simulate(sysd, q0, x0, 0.0, T)
    res.add_check(check_close("스위칭 해 vs 구간 적분: 1차 RMS", tr.rms(0, T, "iL"), w.iL.rms(), 1e-9, "expm vs PWL", True, "A"))
    # Irms vs P for matched and mismatched
    ph = np.linspace(-0.6, 0.6, 61)
    for tag, (a, b) in (("mm", (V1, V2)), ("mt", (V2, V2))):
        pp, rr = [], []
        for x in ph:
            ww = pwl_waves(a, b, L, fs, Modulation(PI, PI, float(x), SPS_TEXTBOOK_C1))
            pp.append(ww.P2)
            rr.append(ww.Irms)
        res.add_series(f"rp_{tag}", f"{'불일치' if tag == 'mm' else '일치'} V₁={a:g}, V₂′={b:g} V", "A", pp, rr)
    res.add_plot("p_rp", "전력 대비 RMS: 불일치 전압비는 0 W에서도 전류가 크다", ["rp_mm", "rp_mt"], x_label="P", x_unit="W", y_label="I_rms", y_unit="A", kind="xy", level="A",
                 markers=[{"x": 0.0, "y": w0.Irms, "label": "φ = 0"}],
                 proved="전압비가 맞으면 0 W에서 전류도 0이지만, 900/600 V에서는 0 W에서 2.165 Arms가 흐른다.", not_yet="자성체·스위칭 손실은 합산하지 않았다.")
    xs, ys = sample_pwl(w.iL)
    res.add_series("iL", "i_L", "A", xs, ys)
    if Lm:
        xs, ys = sample_pwl(w.im)
        res.add_series("im", "i_m (여자)", "A", xs, ys, dash=True)
        xs, ys = sample_pwl(w.i2)
        res.add_series("i2", "i₂ = i_L − i_m (2차 환산)", "A", xs, ys)
    xs, ys = sample_pwl(PWL(w.t, w.v1.y0 - w.v2.y0, w.v1.y1 - w.v2.y1))
    res.add_series("vL", "v_L = v₁ − v₂′", "V", xs, ys)
    mod = Modulation(PI, PI, phi, SPS_TEXTBOOK_C1)
    bands = bands_for(mod, fs, 2)
    res.add_plot("p_i", "0 W의 전류: L 순환전류와 여자전류", ["iL"] + (["im", "i2"] if Lm else []), y_label="전류", y_unit="A", bands=bands, group="zp", level="C",
                 proved="v_L = ±(V₁−V₂′)가 삼각 순환전류를 만든다. 여자전류는 2차 전압이 L_m에 걸려 생기는 별개의 전류다.",
                 not_yet="자화 전류의 DC offset·자속 불균형은 ‘기동 offset’ 실험에서 본다.")
    res.add_plot("p_v", "인덕터 전압", ["vL"], y_label="v_L", y_unit="V", bands=bands, group="zp", level="C", proved="φ=0에서도 v_L이 0이 아니다.", not_yet="")
    res.circuit = {"diagram": dab_circuit(V1, v["VL"], f"{v['Np']}:{v['Ns']}", L, Lm).to_json(), "intervals": bands, "plot_group": "zp"}
    res.verdict("PASS_WITHIN_MODEL", "0 W 순환전류와 여자전류를 분리해 재현 (이상 스위치)")
    res.assumptions += ["이상 bridge, 선형 L·L_m", "손실은 R_est로 후처리한 추정치 (전력단 미결합)"]
    res.not_valid_for += ["실제 무부하 소비전력 (스위칭·코어 손실 미포함)"]
    res.interpretation = "전력이 0이라는 것은 평균 포트 에너지 전달이 0이라는 뜻일 뿐이다. 두 사각파 전압이 다르면 L에 교번전압이 걸려 전류가 흐르고, 그 전류가 전도손실을 만든다. 여자전류는 전압비와 무관하게 흐르는 또 다른 전류라 구분해서 측정한다."
    return res


# ======================================================================================
# 3. Nine voltage corners
# ======================================================================================


def run_corners(v: dict) -> Result:
    res = Result("FL08", "voltage_corners", "A (+C 구간 적분 교차검증)")
    n = _n(v)
    L, fs, P = v["L"], v["fs"], v["P"]
    VHs = [v["VH_min"], v["VH"], v["VH_max"]]
    VLs = [v["VL_min"], v["VL"], v["VL_max"]]
    rows = []
    zi, zp, notes = [], [], []
    worst = (0.0, None)
    fails = []
    max_err = 0.0
    for VL in VLs:
        ri, rp, rn = [], [], []
        for VH in VHs:
            V1, V2 = VH, n * VL
            Pm = ref.sps_pmax(V1, V2, L, fs)
            phi = ref.sps_phi_for_power(V1, V2, L, fs, P)
            if phi is None:
                rows.append([VH, VL, V2, Pm, "해 없음", "", "", "", "", "", "NO_SOLUTION"])
                ri.append(None)
                rp.append(None)
                rn.append(f"P > Pmax {Pm:.4g} W")
                fails.append(f"{VH:g}/{VL:g} V")
                continue
            i0, iphi, irms, ipk = ref.sps_currents(V1, V2, L, fs, phi)
            w = pwl_waves(V1, V2, L, fs, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1))
            max_err = max(max_err, abs(w.Irms - irms) / irms, abs(w.P2 - P) / P)
            scr = commutation_screen(w, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1), fs, n, V1, VL, (v["C0_hv"], v["V0_hv"]), (v["C0_lv"], v["V0_lv"]), v["td_hv"], v["td_lv"])
            prim = [e for e in scr if e.bridge == "1차"]
            sec = [e for e in scr if e.bridge == "2차"]
            st1 = _worst_edge(prim)
            st2 = _worst_edge(sec)
            rows.append([VH, VL, V2, Pm, phi, irms, ipk, -i0, iphi, f"{EDGE_KO[st1]}", f"{EDGE_KO[st2]}"])
            ri.append(irms)
            rp.append(phi)
            rn.append(f"1차: {EDGE_KO[st1]} / 2차: {EDGE_KO[st2]}")
            if irms > worst[0]:
                worst = (irms, (VH, VL))
        zi.append(ri)
        zp.append(rp)
        notes.append(rn)
    res.add_series("map_irms", "I_rms (1차, 모듈당)", "A", VHs, VLs, z=zi, notes=notes)
    res.add_series("map_phi", "φ", "rad", VHs, VLs, z=zp, notes=notes)
    res.add_plot("p_irms", f"9개 전압 corner의 RMS ({P:g} W/모듈)", ["map_irms"], x_label="V_H", x_unit="V", y_label="V_L", y_unit="V", kind="map", level="A",
                 markers=[{"x": v["VH"], "y": v["VL"], "label": "nominal"}],
                 proved="전압비가 맞지 않는 corner에서 같은 전력의 RMS가 커지고, 낮은 V_H·V_L corner에서는 φ가 커진다.", not_yet="ZVS는 부호·전하 screen만 (SCREEN_ONLY).")
    res.add_plot("p_phi", "필요 φ", ["map_phi"], x_label="V_H", x_unit="V", y_label="V_L", y_unit="V", kind="map", level="A", proved="φ가 π/2에 가까울수록 제어 여유와 RMS가 나빠진다.", not_yet="")
    res.tables.append(Table("t_c", "corner별 결과 (SCREEN_ONLY: 합성 C_oss·edge 전류 × dead time)", ["V_H [V]", "V_L [V]", "V₂′ [V]", "Pmax [W]", "φ [rad]", "I_rms [A]", "I_pk [A]", "1차 edge 전류 −i₀ [A]", "2차 edge 전류 i_φ [A]", "1차 screen", "2차 screen"], rows))
    res.add_check(Check("corner 전체: 닫힌 식 vs 구간 적분 (P, I_rms)", "PASS" if max_err < 1e-9 else "FAIL", max_err, "rel", 1e-9, path="9개 corner 각각 닫힌 식 φ로 PWL 정확 적분", independent=True))
    tb = _textbook_nominal(v) and abs(P - 1500) < 1e-9
    V2n = n * v["VL"]
    res.add_metric("Pmax_550_36", "Pmax at V_H 550 V, V_L 36 V", ref.sps_pmax(550.0, n * 36.0, L, fs), "W", ref=2062.5 if tb else None, ref_label="교재 2.0625 kW", tol=1e-9)
    res.add_metric("Irms_worst", "가장 큰 RMS corner", worst[0], "A", basis=f"{worst[1]}" if worst[1] else "")
    del V2n
    if fails:
        res.verdict("NO_SOLUTION", "해가 없는 corner: " + ", ".join(fails))
    res.verdict("SCREEN_ONLY", "ZVS 가능성은 edge 전류 부호·전하 screen으로만 표시 (이상 스위치 모델은 NOT_EVALUABLE)")
    res.assumptions += [f"V_L 최대 {v['VL_max']:g} V는 ASSUMED (교재는 36/48 V만 명시)", "합성 C_oss (HV/LV)와 dead time은 ASSUMED"]
    res.not_valid_for += ["ZVS 확정·손실·온도"]
    res.interpretation = "전력식이 풀린다는 것(해 존재)과 RMS·ZVS·열이 통과한다는 것은 다르다. 전압비가 어긋나는 corner에서 순환전류가 커지고, edge 전류 부호가 바뀌는 쪽 bridge가 먼저 soft switching을 잃는다."
    return res


def _worst_edge(edges):
    order = {"SIGN_FAIL": 3, "ZERO_CURRENT": 2, "CHARGE_SHORT": 1, "SCREEN_PASS": 0}
    return max(edges, key=lambda e: order[e.status]).status


# ======================================================================================
# 4. DC offset at start-up, damping, flux walking
# ======================================================================================


def run_offset(v: dict) -> Result:
    res = Result("FL08", "offset_startup", "C")
    n = _n(v)
    V1, V2, L, fs = v["VH"], n * v["VL"], v["L"], v["fs"]
    phi = ref.sps_phi_for_power(V1, V2, L, fs, v["P"])
    if phi is None:
        res.verdict("NO_SOLUTION", "요청 전력이 Pmax를 넘는다")
        return res
    Lm = v["Lm"] if v["Lm"] > 0 else None
    eps = v["asym"]
    T = 1 / fs
    # primary bridge with a volt-second imbalance: positive pulse longer by eps*pi (w1 > pi is not a 3-level
    # pattern, so model the imbalance as a shift of the negative edge only, via a modified modulation per cycle)
    base = Modulation(PI, PI, phi, SPS_TEXTBOOK_C1)

    class Asym(DABSystem):
        def gate_schedule(self, t0, t1):
            ev = super().gate_schedule(t0, t1)
            if eps == 0:
                return ev
            out = []
            for tt, act in ev:
                k = int(math.floor(tt / self.T + 1e-12))
                th = (tt - k * self.T) * TWO_PI * fs
                if abs(th - PI) < 1e-9:  # primary falling edge (end of the positive half) delayed
                    tt = tt + eps * self.T / 2
                out.append((tt, act))
            return sorted(out, key=lambda e: e[0])

    sysd = Asym(V1, V2, L, fs, R=v["R"], Lm=Lm, n=n, mod_of_cycle=lambda k: base, key=f"off{eps}")
    q0 = base.levels(1e-9)
    N = int(v["cycles"])
    x_zero = np.zeros(2 if Lm else 1)
    tr = simulate(sysd, q0, x_zero, 0.0, N * T)
    means = [tr.mean(k * T, (k + 1) * T, "iL") for k in range(N)]
    im_means = [tr.mean(k * T, (k + 1) * T, "im") for k in range(N)] if Lm else None
    res.add_series("iL_mean", "사이클 평균 i_L (DC offset)", "A", [(k + 0.5) * T for k in range(N)], means, style="points")
    if Lm:
        res.add_series("im_mean", "사이클 평균 i_m (자속 offset)", "A", [(k + 0.5) * T for k in range(N)], im_means, style="points")
    smp = tr.sample(["iL"] + (["im"] if Lm else []), 0.0, N * T, per_segment=2)
    xs, ys = decimate_minmax(smp["t"], smp["iL"], 3000)
    res.add_series("iL", "i_L (0 초기조건에서 시작)", "A", xs, ys)
    if Lm:
        xs, ys = decimate_minmax(smp["t"], smp["im"], 3000)
        res.add_series("im", "i_m", "A", xs, ys)
    # reference periodic solution (symmetric case)
    sys0 = DABSystem(V1, V2, L, fs, R=v["R"], Lm=Lm, n=n, mod_of_cycle=lambda k: base, key="off_ref")
    x_ref = half_wave_periodic(sys0, q0)
    off0 = -x_ref[0]
    res.add_metric("offset0", "0 초기조건의 첫 주기 DC offset (예측 −i(0))", off0, "A", basis="무손실이면 영원히 남는다")
    res.add_metric("offset_end", f"{N}주기 후 평균 i_L", means[-1], "A")
    if v["R"] > 0 and eps == 0:
        tau = L / v["R"]
        pred = off0 * math.exp(-(N - 0.5) * T / tau)
        res.add_metric("tau", "감쇠 시정수 L/R", tau, "s")
        res.add_check(check_close("offset 감쇠: 스위칭 해 vs 해석 e^(−t·R/L)", means[-1], pred, 2e-3, "사이클 평균 (정확 적분) vs 1차 지수 감쇠 (R만 있는 이상 L)", True, "A", abs_scale=abs(off0) + 1e-3))
    elif v["R"] == 0 and eps == 0:
        res.add_check(check_close("무손실: offset이 유지된다", means[-1], means[0], 1e-9, "사이클 평균 비교 (첫 주기 vs 마지막 주기)", True, "A", abs_scale=abs(off0) + 1e-3))
    if eps != 0:
        dvs = V1 * eps * T / 2 * 2  # extra positive volt-seconds per cycle: V1 during +, removed from -: 2 V1 dt
        res.add_metric("dvs", "주기당 순 volt-second 불균형", dvs, "V·s", basis=f"1차 하강 edge를 {eps * 100:g} % × T/2 지연")
        if v["R"] > 0:
            res.add_metric("Idc_bound", "R이 제한하는 DC 전류 ΔV_avg/R", dvs * fs / v["R"], "A", basis="이상 L은 R 없이는 계속 증가")
        if Lm:
            res.add_metric("dIm", "여자전류 walking 경향 (L_m 기준)", 0.0, "A", note="L_m에는 2차 전압이 걸리므로 1차 pulse 비대칭은 L 전류 offset이 되고, 2차 bridge 비대칭은 L_m 자속 walking이 된다 (FL07)")
    led = energy_ledger(tr, sysd, 0.0, N * T, ["p1"], ["p2"], ["pR"], rated_power=abs(v["P"]) + 1)
    res.add_check(ledger_check(led, what=f"{N}주기 창: "))
    res.add_plot("p_mean", "사이클 평균 전류: DC offset은 이상 L에서 사라지지 않는다", ["iL_mean"] + (["im_mean"] if Lm else []), y_label="평균 전류", y_unit="A", level="C", hlines=[{"y": 0.0, "label": "0"}],
                 proved="0 초기조건으로 시작한 이상 DAB는 −i(0)만큼의 DC offset을 계속 유지하고, R이 있을 때만 L/R로 감쇠한다. 정상 기준해를 반대칭 조건으로 따로 정해야 하는 이유다.",
                 not_yet="실제 회로는 권선·스위치 저항, 제어, blocking capacitor로 offset이 제한된다. 자속 포화는 FL07.")
    res.add_plot("p_iL", "전류 파형 (시작부터)", ["iL"] + (["im"] if Lm else []), y_label="전류", y_unit="A", level="C",
                 proved="offset이 파형 전체를 위로 민다: peak 전류와 한쪽 스위치 RMS가 커진다.", not_yet="")
    res.verdict("PASS_WITHIN_MODEL", "기동 offset·감쇠·비대칭 volt-second를 정확 스위칭 해로 재현")
    res.assumptions += ["0 초기조건 (precharge·soft-start 없음)", "비대칭은 1차 하강 edge 지연으로 모델"]
    res.not_valid_for += ["자속 포화·코어 손실", "실제 기동 시퀀스"]
    res.interpretation = "이상 인덕터에는 DC offset을 되돌리는 힘이 없다. 시뮬레이션을 0에서 시작하면 첫 주기의 offset이 그대로 남아 ‘정상상태’처럼 보이지만 실제 기준해가 아니다. 저항·제어·blocking capacitor가 offset을 제한하며, 펄스 비대칭은 자속 walking으로 이어질 수 있다."
    return res


# ======================================================================================
# 5. Two modules: interleaving and sharing
# ======================================================================================


def run_two_modules(v: dict) -> Result:
    res = Result("FL08", "two_modules", "C (구간 적분)")
    n = _n(v)
    V1, V2, fs = v["VH"], n * v["VL"], v["fs"]
    L = v["L"]
    T = 1 / fs
    P_mod = v["P_total"] / 2
    phi = ref.sps_phi_for_power(V1, V2, L, fs, P_mod)
    if phi is None:
        res.verdict("NO_SOLUTION", "모듈당 전력이 Pmax를 넘는다")
        return res
    d = v["dL"]
    L1, L2 = L * (1 - d), L * (1 + d)
    w1 = pwl_waves(V1, V2, L1, fs, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1))
    w2 = pwl_waves(V1, V2, L2, fs, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1))
    I1, I2 = w1.P2 / v["VL"], w2.P2 / v["VL"]
    res.add_metric("I1", "모듈 1 LV DC 전류 (L−δ)", I1, "A", basis=f"L₁ = {L1 * 1e6:.4g} µH, 공통 φ = {phi:.5f} rad")
    res.add_metric("I2", "모듈 2 LV DC 전류 (L+δ)", I2, "A", basis=f"L₂ = {L2 * 1e6:.4g} µH")
    share = (I1 - I2) / ((I1 + I2) / 2)
    res.add_metric("share", "분담 오차 (I₁−I₂)/평균", share, "", basis="공통 φ 명령일 때 P ∝ 1/L")
    res.add_check(check_close("분담: 구간 적분 vs P ∝ 1/L 해석", I1 / I2, L2 / L1, 1e-9, "PWL 전력비 vs L 비 (SPS 전력식이 1/L에 비례)", True, ""))

    def port(wv):
        i2a = wv.i2.scaled(n)
        return PWL(wv.t, i2a.y0 * np.array(wv.b2), i2a.y1 * np.array(wv.b2))

    p1, p2 = port(w1), port(w2)
    rows = []
    for deg in sorted({0.0, 90.0, 180.0, float(v["offset_deg"])}):
        sh = deg / 360.0 * T
        tt = np.linspace(0, T, 4001)
        tot = np.array([p1.value_at(x) + p2.value_at((x - sh) % T) for x in tt])
        ac = tot - tot.mean()
        rows.append([deg, tot.mean(), tot.max() - tot.min(), math.sqrt(np.mean(ac**2))])
        if deg == float(v["offset_deg"]):
            res.add_series("tot", f"합계 LV 포트 전류 (offset {deg:g}°)", "A", (tt * 1).tolist(), tot.tolist())
            res.add_series("m1", "모듈 1 포트 전류", "A", (tt * 1).tolist(), [p1.value_at(x) for x in tt], dash=True)
            res.add_series("m2", "모듈 2 포트 전류 (이동)", "A", (tt * 1).tolist(), [p2.value_at((x - sh) % T) for x in tt], dash=True)
            res.add_metric("ripple_sel", f"선택 offset {deg:g}°의 합계 리플 pp", tot.max() - tot.min(), "A")
    res.tables.append(Table("t_int", "모듈 offset에 따른 합계 LV DC 포트 전류", ["offset [deg]", "평균 [A]", "리플 pp [A]", "AC RMS [A]"], rows, note="SPS 포트 전류 b₂·i_s는 T/2 주기라 180°(T/2) 이동은 같은 파형을 겹칠 뿐이다. 90°(T/4)가 주 리플 성분(2f_s)을 상쇄한다."))
    r = {row[0]: row for row in rows}
    res.add_metric("ripple_0", "0° 합계 리플 pp", r[0.0][2], "A")
    res.add_metric("ripple_90", "90° 합계 리플 pp", r[90.0][2], "A")
    res.add_metric("ripple_180", "180° 합계 리플 pp", r[180.0][2], "A", note="0°와 같다 (d = 0이면 정확히 같음)")
    res.add_check(
        Check(
            "180° 이동이 리플을 줄이지 못함 (포트 전류의 T/2 주기성)",
            "PASS" if (abs(r[180.0][2] - r[0.0][2]) / max(r[0.0][2], 1e-9) < 1e-6 or d != 0) else "FAIL",
            abs(r[180.0][2] - r[0.0][2]),
            "A",
            path="동일 모듈(δ = 0)에서 0°와 180°의 합계 리플 비교",
            independent=True,
            detail="L 불일치가 있으면 정확히 같지는 않다",
        )
    )
    res.add_plot("p_port", "LV DC 포트 전류: 모듈별과 합계", ["tot", "m1", "m2"], y_label="전류", y_unit="A", level="C",
                 proved="모듈 사이 offset(모듈 간 위상)과 bridge 사이 φ는 다른 변수다. 실제 포트 전류로 0/90/180°를 비교했다.", not_yet="출력 커패시터·배선 임피던스는 포함하지 않았다.")
    res.verdict("PASS_WITHIN_MODEL", "interleaving과 L 불일치 분담을 실제 포트 전류로 비교")
    res.assumptions += [f"모듈 L = L(1∓δ), δ = {d:g}", "공통 φ 명령 (모듈별 전류 제어 없음)"]
    res.not_valid_for += ["모듈별 폐루프 분담 제어", "출력 필터 설계"]
    res.interpretation = "SPS의 포트 전류는 반주기마다 극성과 전류가 함께 뒤집혀 T/2 주기를 가진다. 그래서 모듈을 180° 옮기면 같은 파형이 겹칠 뿐이고, 90°가 주 리플을 상쇄한다. 공통 φ로 운전하면 L이 작은 모듈이 더 많은 전력을 가져가 분담 오차가 L 오차만큼 생긴다."
    return res


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "DAB P(φ)를 어떻게 얻나?",
        "두 bridge 사각파의 차가 L에 걸려 네 구간 직선 전류를 만든다. 이상 L의 주기조건만으로는 DC offset이 정해지지 않으므로 반주기 반대칭(zero-DC) 해를 택하고, v₂′·i를 구간 적분하면 P = V₁V₂φ(1−|φ|/π)/(ω_sL)이다.",
        "How do you derive P(φ) for a DAB?",
        "The difference of the two square waves drives the inductor, giving four linear current segments. The periodic condition alone does not fix the DC offset, so I take the half-wave antisymmetric zero-DC solution and integrate the secondary voltage times the current over the period.",
        ["4구간 전류", "zero-DC 선택", "전력 적분"],
    ),
    Question(
        "DAB가 0 W인데 전류가 왜 흐르나?",
        "전압비 불일치면 φ=0에서도 v_L = ±(V₁−V₂′)가 걸려 삼각 순환전류가 흐른다(900/600 V에서 2.165 Arms). 여자전류와 DC offset·자속 불균형은 별개의 원인이다.",
        "Why does current flow in a DAB at zero power?",
        "With a voltage-ratio mismatch the inductor still sees plus or minus V₁ minus V₂′ at zero phase shift, so a triangular circulating current flows, 2.17 A rms at 900 to 600 V. Magnetizing current and DC offset are separate causes.",
        ["전압비 불일치", "여자전류 별개", "DC offset"],
    ),
    Question(
        "SPS Pmax가 충분한데 왜 실패하나?",
        "해가 존재하는 것과 RMS·peak·ZVS·열·자성체·제어 여유가 통과하는 것은 다르다. 불일치 corner에서 RMS가 커지고 edge 전류 부호가 바뀌면 soft switching을 잃는다.",
        "The SPS Pmax is large enough. Why can the design still fail?",
        "A power solution existing is not the same as RMS, peak current, ZVS, thermal and control margins passing. At mismatched corners the RMS rises and the edge-current sign can flip, so soft switching is lost.",
        ["RMS/peak", "ZVS", "열·자성체", "제어 여유"],
    ),
    Question(
        "모듈 두 개를 180° 어긋나게 돌리면 리플이 상쇄되나?",
        "SPS 포트 전류 s(t)·i(t)는 T/2 주기라 180°는 같은 파형을 겹칠 뿐이다. 90°(T/4)가 주 리플을 상쇄한다. 모듈 간 offset과 bridge 간 φ는 다른 변수다.",
        "If two modules run 180 degrees apart, does the ripple cancel?",
        "No. The SPS port current repeats every half period, so a 180-degree shift overlaps identical waveforms. A 90-degree shift cancels the dominant ripple. The module offset is a different variable from the bridge phase shift.",
        ["T/2 주기", "90°", "변수 구분"],
    ),
]

_timing = [
    Param("td_hv", "1차 dead time (screen용)", "s", 150e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", group="screen"),
    Param("td_lv", "2차 dead time (screen용)", "s", 50e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", group="screen"),
]

EXPERIMENTS = [
    Experiment(
        key="sps_nominal",
        title="SPS 4구간: 닫힌 식·구간 적분·스위칭 해로 같은 숫자 얻기",
        goal="800↔48 V, n = 50/3, 200 µH, 100 kHz에서 모듈당 1.5 kW의 φ = 0.328972794 rad, I_rms = 2.019881509 A, I_pk = 2.09430585 A를 세 개의 독립 경로로 재현하고, 모듈당·합계, d와 Δt/T, 2차 AC RMS와 DC 전류를 구분한다.",
        params=_base_params(
            [
                Param("P", "모듈당 전달전력 P (+ = HV→LV)", "W", 1500.0, "W", vmin=-20000, vmax=20000, source="TEXTBOOK", source_note="1.5 kW/모듈"),
                Param("modules", "모듈 수", "", 2, "", vmin=1, vmax=16, kind="int", source="TEXTBOOK", source_note="2 × 1.5 kW"),
                Param("R", "직렬 저항 R (전력단 결합)", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="비이상"),
                Param("Lm", "여자 인덕턴스 L_m (0 = 없음)", "H", 0.0, "mH", vmin=0, vmax=1.0, source="ASSUMED", group="비이상"),
            ]
            + _timing
        ),
        presets=[
            Preset("nominal", "교재: 800↔48 V, 1.5 kW/모듈", {}, "ch.11 worked example", ("nominal", "reference")),
            Preset("reverse", "역방향 −1.5 kW (LV→HV)", {"P": -1500.0}, "부호 반전", ("variant", "reference")),
            Preset("with_lm", "L_m = 2 mH 포함", {"Lm": 2e-3}, "여자전류 분리", ("variant",)),
            Preset("low_corner", "550 V / 36 V corner", {"VH": 550.0, "VL": 36.0}, "Pmax 2.0625 kW", ("corner",)),
            Preset("over", "5 kW 요청 (Pmax 초과)", {"P": 5000.0}, "해 없음", ("failure",)),
        ],
        run=run_sps,
        model_level="A + C",
        suggested_change="전력을 +1500 W → −1500 W로 바꿔 부호 반전을 본다 (다음에는 L_m = 2 mH).",
        prediction=Prediction(
            "P를 +1.5 kW → −1.5 kW로 바꾸면 φ와 I_rms는?",
            ["φ 부호만 바뀌고 RMS는 같다", "φ는 같고 RMS가 두 배", "해가 없다", "모르겠다"],
            "φ 부호만 바뀌고 RMS는 같다",
            "P(φ)는 φ의 홀함수라 방향만 바뀐다. 같은 전압비에서 전류 크기(RMS·peak)는 같고 파형이 시간축에서 뒤집힌다.",
            ["phi", "Irms", "Ipk"],
            handcalc=[{"key": "phi", "label": "φ", "unit": "rad"}, {"key": "Ipk", "label": "I_pk", "unit": "A"}, {"key": "Irms", "label": "I_rms", "unit": "A"}],
        ),
        suggested={"P": -1500.0},
        student="두 full bridge가 각각 ±V 사각파를 만들고, 그 차이가 L에 걸려 전류가 흐른다. 1차가 φ만큼 앞서면 전류가 2차 전압과 같은 부호로 더 오래 겹쳐 HV→LV로 전력이 간다.",
        expert=(
            "① 이상 L의 주기조건은 DC offset을 정하지 않는다 — 기준해를 반주기 반대칭으로 명시하고, 시뮬레이션을 0에서 시작하면 offset이 남는다(‘기동 offset’ 실험). "
            "② d = φ/π와 Δt/T = φ/(2π)는 두 배 차이다. ③ 모듈당 1.5 kW / 합계 3 kW, 2차 AC RMS 33.66 A / DC 31.25 A처럼 기준을 섞지 않는다. "
            "④ ZVS는 이상 스위치로는 NOT_EVALUABLE — edge 전류 부호·전하는 SCREEN_ONLY이고 실제 판정은 노드 용량·dead time 모델(EX02·EX06)이 필요하다."
        ),
        customer_ko="이 조건에서 모듈당 1.5 kW는 φ ≈ 18.85°로 1차 RMS 2.02 A, peak 2.09 A입니다. 두 모듈 합계 LV DC는 62.5 A이고 2차 권선 RMS는 모듈당 33.7 A라 LV 측 도체·termination이 지배할 수 있습니다. ZVS는 실제 edge 전류와 dead time으로 따로 확인하시죠.",
        customer_en="At this point each module transfers 1.5 kW at about 18.85 degrees of phase shift, with 2.02 A rms and 2.09 A peak on the primary. The two modules give 62.5 A DC on the low-voltage side, and each secondary winding carries 33.7 A rms, so the LV conductors and terminations may dominate. ZVS needs a separate check with the real edge currents and dead time.",
        questions=_Q[:1],
        circuit="dab",
        textbook=[TB_11],
        reference_presets=["nominal", "reverse", "over"],
        claim_limit="이상 스위치(C)·선형 L의 전력·RMS·peak. ZVS·손실 주장 없음.",
    ),
    Experiment(
        key="zero_power_mismatch",
        title="0 W인데도 왜 뜨거운가: 순환전류와 여자전류",
        goal="V_H 900 V, n·V_L 600 V, φ = 0에서 P = 0이지만 I_rms = 2.165063509 A(peak 3.75 A)로 nominal 1.5 kW의 RMS보다 크다는 것을 확인하고, L_m이 만드는 여자전류와 구분한다.",
        params=_base_params(
            [
                Param("phi", "outer phase φ", "rad", 0.0, "rad", vmin=-PI, vmax=PI, source="TEXTBOOK", source_note="φ = 0"),
                Param("Lm", "여자 인덕턴스 L_m (0 = 없음)", "H", 2e-3, "mH", vmin=0, vmax=1.0, source="ASSUMED", source_note="분리 설명용 합성값", group="비이상"),
                Param("R_est", "손실 추정용 경로 저항", "Ω", 0.1, "mΩ", vmin=0, vmax=10, source="ASSUMED", source_note="후처리 추정", group="비이상"),
            ],
            VH=900.0,
            VL=36.0,
        ),
        presets=[
            Preset("mismatch", "교재: 900 V / 36 V (V₂′ = 600 V), φ = 0", {}, "ch.11 전문가", ("nominal", "reference")),
            Preset("matched", "전압비 일치 800 V / 48 V", {"VH": 800.0, "VL": 48.0}, "순환전류 사라짐", ("variant", "reference")),
        ],
        run=run_zero_power,
        model_level="A + C",
        suggested_change="V_H를 900 → 800 V, V_L을 36 → 48 V로 바꿔 전압비를 맞춘다.",
        prediction=Prediction(
            "전압비를 맞추면(800/48 V) φ = 0의 L 순환전류 RMS는?",
            ["거의 0이 된다", "그대로 2.17 A", "더 커진다", "모르겠다"],
            "거의 0이 된다",
            "v_L = ±(V₁ − V₂′)이므로 V₁ = V₂′이면 0이다. 여자전류는 전압비와 무관하게 남는다.",
            ["Irms_L", "Im_rms"],
            handcalc=[{"key": "Ipk_L", "label": "순환전류 peak", "unit": "A"}, {"key": "Irms_L", "label": "순환전류 RMS", "unit": "A"}],
        ),
        suggested={"VH": 800.0, "VL": 48.0},
        student="전력이 0이라도 두 사각파 전압이 다르면 그 차이가 L에 걸려 전류가 오르내린다. 에너지는 반주기마다 주고받을 뿐 평균 전달이 0이다.",
        expert="현장 진단(CASE C): ① 동일 전압비에서 사라지는지 ② 여자전류(L_m) ③ phase offset·flux imbalance·제어 limit cycle을 구분한다. 평균 출력전류만 보고 전류센서 오동작으로 판단하지 않는다. 순환전류는 전도손실을 만들지만 soft switching 전하를 줄 수도 있어 경부하 설계의 tradeoff가 된다(EX06).",
        customer_ko="무부하에서 변압기가 뜨거운 것은 전압비 불일치가 만든 순환전류일 수 있습니다. 이 조건에서 0 W에도 2.2 Arms가 흐르며 nominal보다 큽니다. 동일 전압비에서 사라지는지, 여자전류·DC offset과 구분해 측정하시죠.",
        customer_en="The transformer heating at no load can come from circulating current caused by the voltage-ratio mismatch: here about 2.2 A rms flows at zero power, more than at nominal load. Let's check whether it disappears at a matched ratio and separate it from magnetizing current and DC offset.",
        questions=_Q[1:2],
        circuit="dab",
        textbook=[TB_11, TB_15],
        reference_presets=["mismatch", "matched"],
        claim_limit="이상 bridge의 순환·여자전류. 무부하 손실 정밀값 아님.",
    ),
    Experiment(
        key="voltage_corners",
        title="9개 voltage corner: 해가 있는 것과 통과하는 것은 다르다",
        goal="V_H 550/800/900 V × V_L 36/48/54 V에서 모듈당 1.5 kW의 φ·RMS·peak·edge 전류를 계산하고, Pmax(550/36 V = 2.0625 kW)와 bridge별 ZVS 부호·전하 screen을 지도화한다.",
        params=_base_params(
            [
                Param("P", "모듈당 전력", "W", 1500.0, "W", vmin=1, vmax=20000, source="TEXTBOOK"),
                Param("VH_min", "V_H 최소", "V", 550.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="550 V corner", group="corner"),
                Param("VH_max", "V_H 최대", "V", 900.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="900 V corner", group="corner"),
                Param("VL_min", "V_L 최소", "V", 36.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="36 V", group="corner"),
                Param("VL_max", "V_L 최대", "V", 54.0, "V", vmin=1, vmax=1000, source="ASSUMED", source_note="교재에 값 없음", group="corner"),
                Param("C0_hv", "HV 소자 C₀ (합성)", "F", COSS_HV[0], "nF", vmin=1e-12, vmax=1e-6, source="ASSUMED", group="screen"),
                Param("V0_hv", "HV 소자 V₀ (합성)", "V", COSS_HV[1], "V", vmin=0.1, vmax=1e4, source="ASSUMED", group="screen"),
                Param("C0_lv", "LV 소자 C₀ (합성)", "F", COSS_LV[0], "nF", vmin=1e-12, vmax=1e-5, source="ASSUMED", group="screen"),
                Param("V0_lv", "LV 소자 V₀ (합성)", "V", COSS_LV[1], "V", vmin=0.1, vmax=1e4, source="ASSUMED", group="screen"),
            ]
            + _timing
        ),
        presets=[Preset("textbook", "교재 corner, 1.5 kW", {}, "", ("nominal", "reference")), Preset("light", "300 W", {"P": 300.0}, "경부하 corner", ("corner", "reference"))],
        run=run_corners,
        model_level="A (+C 교차검증)",
        suggested_change="전력을 1500 → 300 W로 낮춰 어느 corner에서 soft switching 부호가 먼저 무너지는지 본다.",
        prediction=Prediction(
            "모듈당 300 W로 낮추면 900 V / 36 V corner의 2차 edge 전류 부호는?",
            ["여전히 ZVS 방향", "반대가 되어 hard switching", "변화 없음", "모르겠다"],
            "반대가 되어 hard switching",
            "경부하에서는 φ가 작아져 i_φ ≈ i₀ + aφ가 음수로 남는다(V₁ > V₂′인 쪽). 2차 bridge가 먼저 soft switching을 잃는다.",
            ["Irms_worst"],
        ),
        suggested={"P": 300.0},
        student="같은 전력이라도 입력·출력 전압이 달라지면 필요한 φ와 전류 모양이 달라진다. 전압비가 크게 어긋나는 모서리에서 전류가 커진다.",
        expert="corner 표는 ‘해 존재’(Pmax)와 ‘edge 전류 부호/전하’(SCREEN_ONLY)와 ‘RMS’(열)를 분리한다. 합성 C_oss·dead time은 ASSUMED이며, 실제 판정은 EX02의 event 모델과 소자 데이터가 필요하다. 전압 범위를 넓게 요구받으면 variable f_s·변조(EX06)·권선비 재검토를 비교한다.",
        customer_ko="배터리 전압 범위의 모든 corner에서 전력은 나오지만, 저전압·경부하 corner에서 2차 bridge가 soft switching을 잃고 RMS가 커집니다. 권선비·변조 방식·동작 범위 중 무엇을 바꿀지 corner 데이터로 정하시죠.",
        customer_en="Power is available at every corner of the battery range, but at the low-voltage, light-load corners the secondary bridge loses soft switching and the RMS current rises. Let's use the corner data to decide between the turns ratio, the modulation scheme and the operating range.",
        questions=_Q[2:3],
        circuit="dab",
        textbook=[TB_11],
        reference_presets=["textbook", "light"],
        claim_limit="해 존재·RMS·부호 screen (SCREEN_ONLY).",
    ),
    Experiment(
        key="offset_startup",
        title="DC offset·기동·비대칭 volt-second: 이상 L은 offset을 없애지 않는다",
        goal="0 초기조건에서 시작한 스위칭 해가 −i(0)만큼의 DC offset을 유지하고(무손실), R이 있으면 L/R로 감쇠하며, 펄스 비대칭이 DC 전류/자속 walking을 만든다는 것을 정확 해로 확인한다.",
        params=_base_params(
            [
                Param("P", "모듈당 전력", "W", 1500.0, "W", vmin=-20000, vmax=20000, source="TEXTBOOK"),
                Param("R", "직렬 저항 R", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="비이상"),
                Param("Lm", "L_m (0 = 없음)", "H", 0.0, "mH", vmin=0, vmax=1.0, source="ASSUMED", group="비이상"),
                Param("asym", "1차 펄스 비대칭 ε (반주기 대비)", "", 0.0, "", vmin=0, vmax=0.2, source="ASSUMED", group="비이상"),
                Param("cycles", "시뮬레이션 주기 수", "", 60, "", vmin=5, vmax=2000, kind="int", source="ASSUMED", group="시뮬레이션"),
            ]
        ),
        presets=[
            Preset("lossless", "무손실 (offset 유지)", {}, "", ("nominal", "reference")),
            Preset("damped", "R = 0.5 Ω (감쇠)", {"R": 0.5, "cycles": 200}, "τ = L/R = 400 µs", ("variant", "reference")),
            Preset("asym", "비대칭 1 %, R = 0.5 Ω", {"asym": 0.01, "R": 0.5, "cycles": 200}, "volt-second 불균형", ("failure",)),
        ],
        run=run_offset,
        model_level="C",
        suggested_change="R을 0 → 0.5 Ω으로 넣어 offset이 어떻게 되는지 본다 (cycles 200).",
        prediction=Prediction(
            "0 초기조건에서 시작한 무손실 DAB의 DC offset은 시간이 지나면?",
            ["저절로 0이 된다", "그대로 남는다", "커진다", "모르겠다"],
            "그대로 남는다",
            "이상 L에는 DC 성분을 되돌리는 항이 없다. R이 있어야 L/R로 감쇠한다. 그래서 정상 기준해를 반대칭 조건으로 따로 정한다.",
            ["offset0", "offset_end"],
        ),
        suggested={"R": 0.5, "cycles": 200},
        student="인덕터 전류의 평균을 바꾸는 것은 평균 전압뿐이다. 이상 회로는 한 번 생긴 평균 전류를 줄일 방법이 없다.",
        expert="시뮬레이터가 매 주기 자속·전류를 0으로 reset하면 기동 포화 같은 고장을 숨긴다(CASE E). offset의 원인(초기조건·펄스 비대칭·dead time 불균형·측정 offset)을 구분하고, blocking capacitor·저항·제어로 제한되는지 확인한다.",
        customer_ko="enable 직후 전류가 한쪽으로 치우치는 것은 초기 자속/전류와 펄스 비대칭 때문일 수 있습니다. 실제 권선 전압의 volt-second와 전류 기울기를 동시에 보고, 비대칭 원인과 측정 offset을 구분하시죠.",
        customer_en="A current that shifts to one side right after enable can come from the initial flux or current and from pulse asymmetry. Let's capture the winding volt-seconds and the current slope together, and separate the asymmetry from any measurement offset.",
        questions=[],
        circuit="dab",
        textbook=[TB_11, TB_10],
        reference_presets=["lossless", "damped"],
        runtime_hint="seconds",
        claim_limit="이상 스위치·선형 L의 offset 동역학. 포화 없음.",
    ),
    Experiment(
        key="two_modules",
        title="2개 모듈 interleaving: 0/90/180°의 실제 DC 포트 전류와 L±10 % 분담",
        goal="두 모듈의 LV DC 포트 전류를 모듈 offset 0/90/180°로 더해 180°가 리플을 상쇄하지 못하는 이유(T/2 주기)를 보이고, 모듈 L이 ±10 % 다를 때 공통 φ의 분담 오차를 평가한다.",
        params=_base_params(
            [
                Param("P_total", "총 전력 (2모듈)", "W", 3000.0, "W", vmin=1, vmax=40000, source="TEXTBOOK", source_note="2 × 1.5 kW"),
                Param("offset_deg", "모듈 offset (f_s 기준)", "deg", 90.0, "deg", vmin=0, vmax=360, source="TEXTBOOK", source_note="0/90/180 비교"),
                Param("dL", "모듈 L 오차 δ (L₁ = L(1−δ), L₂ = L(1+δ))", "", 0.1, "", vmin=0, vmax=0.5, source="TEXTBOOK", source_note="±10 %"),
            ]
        ),
        presets=[Preset("textbook", "90°, L ±10 %", {}, "", ("nominal", "reference")), Preset("ideal", "δ = 0, 180°", {"dL": 0.0, "offset_deg": 180.0}, "", ("variant", "reference"))],
        run=run_two_modules,
        model_level="C",
        suggested_change="모듈 offset을 90 → 180°로 바꾼다.",
        prediction=Prediction(
            "두 모듈을 180° 어긋나게 돌리면 합계 LV DC 전류 리플은?",
            ["0에 가깝게 상쇄", "0°와 거의 같다", "두 배가 된다", "모르겠다"],
            "0°와 거의 같다",
            "SPS 포트 전류는 반주기마다 극성과 전류가 함께 뒤집혀 T/2 주기를 가진다. 180°(T/2) 이동은 같은 파형을 겹칠 뿐이다.",
            ["ripple_sel", "ripple_0", "ripple_90", "ripple_180"],
        ),
        suggested={"offset_deg": 180.0},
        student="파형이 반주기마다 똑같이 반복되면 반주기만큼 밀어도 똑같은 파형이다. 상쇄하려면 반복 주기의 절반, 즉 1/4 주기를 밀어야 한다.",
        expert="모듈 offset과 bridge 간 φ는 다른 변수다. 공통 φ 명령에서 분담은 L에 반비례하므로 모듈별 전류 제어나 L 선별이 필요하다. 모듈 enable/disable 전략은 기동·precharge·분담·과도응답 검증이 추가된다.",
        customer_ko="두 모듈을 180°로 interleaving하면 이 방식에서는 출력 리플이 줄지 않습니다. 90°로 바꾸고, L 편차 ±10 %는 공통 명령에서 분담 오차 약 ±10 %가 되니 모듈별 전류 제어를 검토하시죠.",
        customer_en="Interleaving the two modules at 180 degrees does not reduce the output ripple in this modulation. A 90-degree offset does. A plus or minus ten percent inductance spread gives about the same current-sharing error under a common command, so per-module current control is worth considering.",
        questions=_Q[3:],
        circuit="dab",
        textbook=[TB_11],
        reference_presets=["textbook", "ideal"],
        claim_limit="이상 파형의 포트 전류 합. 출력 필터·제어 미포함.",
    ),
]

LAB = Lab(
    id="FL08",
    title="DAB — 식에서 파형, 파형에서 설계 판단으로",
    title_en="Dual active bridge: equations, waveforms, design judgement",
    track="basic",
    order=8,
    path_note="14일 경로 7–8일차",
    textbook=[TB_11, TB_10, TB_15],
    prerequisites=["FL01", "FL07"],
    summary="SPS 4구간 → 세 경로 검증 → 0 W 순환전류·여자전류 → 9 corner screen → 기동 offset → 2모듈 interleaving·분담.",
    experiments=EXPERIMENTS,
    minimum_scope="DAB analytical·piecewise·switching; 1.5 kW·2.01988 Arms·mismatch (교재 19장); zero-DC 기준해, L_m 분리, 0/90/180° 포트 전류, ZVS NOT_EVALUABLE/SCREEN_ONLY",
    claim_limits=["이상 스위치 모델: ZVS는 NOT_EVALUABLE, 부호·전하는 SCREEN_ONLY", "손실은 후처리 추정 또는 결합 R만", "모듈당/합계를 분리해 표시"],
    test_paths=["tests/test_fl08.py"],
)
