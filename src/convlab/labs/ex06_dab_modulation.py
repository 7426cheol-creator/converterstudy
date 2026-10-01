"""EX06 - DAB: RMS minimisation vs ZVS trade-off (textbook E06).

Extends FL08 (DAB model, three paths) and EX02 (nonlinear Coss commutation).  The bridge
switching function b(theta; w, phi) with pulse widths w1, w2 and outer phase phi is integrated
exactly (piecewise linear); SPS is checked against its closed form on an independent path.
Candidate selection goes in stages: ideal-current optimum -> per-leg commutation screen ->
loss estimate.  Without device/magnetics data the last stage is MISSING_INPUT; an optional,
clearly synthetic loss model only illustrates how the ranking *could* change.  Nothing here is
called a global optimum or an efficiency gain.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from ..engine.switched import simulate
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import dab as ref
from ._common import decimate_minmax, energy_ledger, ledger_check
from ._dab import (
    EDGE_KO,
    PI,
    TWO_PI,
    DABSystem,
    Modulation,
    bands_for,
    commutation_screen,
    coss_Q,
    dab_circuit,
    half_wave_periodic,
    leg_edges,
    leg_states,
    min_pulse_ok,
    pwl_waves,
    sample_pwl,
)
from .ex02_coss_zvs import Node

TB_E06 = TextbookRef("expert-e06-dab-rms-최소화와-zvs의-상충관계", "E06 · DAB: RMS 최소화와 ZVS의 상충관계")
TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB [FL08]")
TB_E12 = TextbookRef("expert-capstone-b-900-v저전압-dab에서-circulating-current와-zvs", "E12 Capstone B")


def first_root(V1, V2, L, fs, P, w1, w2, pattern1="alternating", pattern2="alternating", n_scan=96):
    """Smallest phi >= 0 with P(phi) = P* on the exact PWL power (scan + bisection)."""
    def f(ph):
        return pwl_waves(V1, V2, L, fs, Modulation(w1, w2, ph, 0.0, pattern1, pattern2)).P2 - P

    grid = np.linspace(0.0, PI, n_scan + 1)
    fa = f(grid[0])
    for k in range(1, grid.size):
        fb = f(grid[k])
        if fa == 0.0:
            return float(grid[k - 1])
        if fa * fb < 0:
            return float(brentq(f, grid[k - 1], grid[k], xtol=1e-14, rtol=1e-15, maxiter=100))
        fa = fb
    return None


def _params(extra=None, P=1500.0):
    return [
        Param("V1", "1차 bridge 전압 V₁ (= V_H)", "V", 900.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="900 V"),
        Param("V2", "2차 전압 1차 환산 V₂′ = n·V_L", "V", 600.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="600 V (low-battery corner)"),
        Param("n", "권선비 n = N_p/N_s", "", 50 / 3, "", vmin=0.01, vmax=1000, source="TEXTBOOK", source_note="50/3 (2차 실제 전류 환산용)"),
        Param("L", "직렬 L (1차 환산)", "H", 200e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK"),
        Param("fs", "f_s", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK"),
        Param("P", "모듈당 전력", "W", P, "W", vmin=1, vmax=20000, source="TEXTBOOK", source_note="1.5 kW"),
    ] + (extra or [])


_screen = [
    Param("C0_hv", "HV 소자 C₀ (합성)", "F", 0.5e-9, "nF", vmin=1e-12, vmax=1e-6, source="ASSUMED", group="commutation"),
    Param("V0_hv", "HV 소자 V₀ (합성)", "V", 40.0, "V", vmin=0.1, vmax=1e4, source="ASSUMED", group="commutation"),
    Param("C0_lv", "LV 소자 C₀ (합성)", "F", 8e-9, "nF", vmin=1e-12, vmax=1e-5, source="ASSUMED", group="commutation"),
    Param("V0_lv", "LV 소자 V₀ (합성)", "V", 5.0, "V", vmin=0.1, vmax=1e4, source="ASSUMED", group="commutation"),
    Param("td_hv", "1차 dead time", "s", 150e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", group="commutation"),
    Param("td_lv", "2차 dead time", "s", 50e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", group="commutation"),
    Param("t_min", "최소 펄스/zero-state 폭", "s", 200e-9, "ns", vmin=0, vmax=5e-6, source="ASSUMED", group="commutation"),
    Param("I_lim", "1차 peak 전류 한계", "A", 8.0, "A", vmin=0.01, vmax=1000, source="ASSUMED", group="commutation"),
]


def _VL(v):
    return v["V2"] / v["n"]


# ======================================================================================
# 1. General modulation and the textbook table
# ======================================================================================


def run_general(v: dict) -> Result:
    res = Result("EX06", "general_modulation", "C (정확 구간 적분) + A (SPS 닫힌 식)")
    V1, V2, L, fs, P = v["V1"], v["V2"], v["L"], v["fs"], v["P"]
    w1, w2 = v["w1"] * PI, v["w2"] * PI
    tb = abs(V1 - 900) < 1e-9 and abs(V2 - 600) < 1e-9 and abs(L - 200e-6) < 1e-12 and abs(fs - 1e5) < 1e-6 and abs(P - 1500) < 1e-9
    # SPS: closed form vs exact PWL (independent)
    phi_s = ref.sps_phi_for_power(V1, V2, L, fs, P)
    if phi_s is None:
        res.verdict("NO_SOLUTION", "SPS로도 요청 전력에 해가 없다 (P > Pmax)")
        return res
    ws = pwl_waves(V1, V2, L, fs, Modulation(PI, PI, phi_s, 0.0))
    _, _, irms_cf, ipk_cf = ref.sps_currents(V1, V2, L, fs, phi_s)
    res.add_metric("phi_sps", "SPS φ", phi_s, "rad", ref=0.399994 if tb else None, ref_label="교재 0.399994 rad", tol=2e-6)
    res.add_metric("irms_sps", "SPS I_rms", ws.Irms, "A", ref=3.113563 if tb else None, ref_label="교재 3.113563 A", tol=2e-7, basis="1차, 모듈당")
    res.add_metric("ipk_sps", "SPS I_pk", ws.Ipk, "A", ref=5.659830 if tb else None, ref_label="교재 5.659830 A", tol=2e-7)
    res.add_check(check_close("SPS: 닫힌 식 vs 구간 적분 (I_rms)", ws.Irms, irms_cf, 1e-10, "I_rms 일반식 vs b(θ;π,φ) 정확 PWL 적분", True, "A"))
    res.add_check(check_close("SPS: 닫힌 식 vs 구간 적분 (I_pk)", ws.Ipk, ipk_cf, 1e-10, "max(|i₀|,|i_φ|) vs PWL 꼭짓점", True, "A"))
    # candidate
    phi_c = first_root(V1, V2, L, fs, P, w1, w2, v["pattern"], v["pattern"])
    tbc = tb and abs(v["w1"] - 0.7) < 1e-12 and abs(v["w2"] - 1.0) < 1e-12
    if phi_c is None:
        res.verdict("NO_SOLUTION", f"w₁/π = {v['w1']:g}, w₂/π = {v['w2']:g}에서는 요청 전력을 만드는 φ가 없다")
        return res
    mod = Modulation(w1, w2, phi_c, 0.0, v["pattern"], v["pattern"])
    wc = pwl_waves(V1, V2, L, fs, mod)
    res.add_metric("phi_c", "candidate φ (가장 작은 해)", phi_c, "rad", ref=0.499016 if tbc else None, ref_label="교재 0.499016 rad", tol=2e-6)
    res.add_metric("P_c", "candidate 전달전력", wc.P2, "W", ref=P, ref_label="목표", tol=1e-9)
    res.add_metric("irms_c", "candidate I_rms", wc.Irms, "A", ref=2.893092 if tbc else None, ref_label="교재 2.893092 A", tol=2e-7)
    res.add_metric("ipk_c", "candidate I_pk", wc.Ipk, "A", ref=5.007628 if tbc else None, ref_label="교재 5.007628 A", tol=2e-7)
    res.add_metric("drms", "RMS 변화", wc.Irms / ws.Irms - 1, "", ref=-0.0708 if tbc else None, ref_label="교재 약 −7.08 %", tol=0.01, basis="상대")
    res.add_metric("dpk", "peak 변화", wc.Ipk / ws.Ipk - 1, "", ref=-0.1152 if tbc else None, ref_label="교재 약 −11.52 %", tol=0.01)
    res.add_metric("cond_ratio", "고정 등가저항 전도항 비율 (I_c/I_sps)²", (wc.Irms / ws.Irms) ** 2, "", ref=0.8634 if tbc else None, ref_label="교재 0.8634", tol=1e-3, note="전체 손실이 이만큼 준다는 뜻이 아니다 — 스위칭·zero-state 도통·자성체 손실이 다르다")
    res.add_check(check_close("candidate: 목표 전력 재현", wc.P2, P, 1e-9, "PWL 전력의 φ 근 (첫 교차 bisection)", False, "W"))
    # engine cross-check of the candidate (3-level bridges, independent integrator)
    sysd = DABSystem(V1, V2, L, fs, n=v["n"], mod_of_cycle=lambda k: mod, key="ex06c")
    q0 = mod.levels(1e-9)
    x0 = half_wave_periodic(sysd, q0)
    T = 1 / fs
    tr = simulate(sysd, q0, x0, 0.0, 2 * T)
    res.add_check(check_close("candidate: 구간 적분 vs 스위칭 해 (P)", tr.energy(0, T, "p2") / T, wc.P2, 1e-9, "PWL vs expm·3-level bridge mode", True, "W", abs_scale=abs(P)))
    res.add_check(check_close("candidate: 구간 적분 vs 스위칭 해 (I_rms)", tr.rms(0, T, "iL"), wc.Irms, 1e-9, "PWL vs expm", True, "A"))
    res.add_check(ledger_check(energy_ledger(tr, sysd, 0.0, T, ["p1"], ["p2"], ["pR"], rated_power=P)))
    # waveforms: SPS vs candidate
    xs, ys = sample_pwl(ws.iL)
    res.add_series("i_sps", "i_L SPS", "A", xs, ys, dash=True)
    xs, ys = sample_pwl(wc.iL)
    res.add_series("i_c", "i_L candidate", "A", xs, ys)
    xs, ys = sample_pwl(wc.v1)
    res.add_series("v1", "v₁ candidate (3-level)", "V", xs, ys)
    xs, ys = sample_pwl(wc.v2)
    res.add_series("v2", "v₂′ candidate", "V", xs, ys)
    # leg gate patterns
    om = TWO_PI * fs
    ts = np.linspace(0, 2 * T, 1601)
    for name, w_, c_, off in (("A", w1, 0.0, 3.0), ("B", w1, 0.0, 1.5)):
        res.add_series(f"g{name}", f"leg {name} (1차)", "", ts.tolist(), [off + 0.6 * leg_states(om * t, w_, c_, v["pattern"])[0 if name == "A" else 1] for t in ts])
    for name, w_, c_, off in (("C", w2, phi_c, -1.5), ("D", w2, phi_c, -3.0)):
        res.add_series(f"g{name}", f"leg {name} (2차)", "", ts.tolist(), [off + 0.6 * leg_states(om * t, w_, c_, v["pattern"])[0 if name == "C" else 1] for t in ts])
    bands = bands_for(mod, fs, 2)
    res.add_plot("p_i", "인덕터 전류: SPS vs width candidate", ["i_sps", "i_c"], y_label="i_L", y_unit="A", bands=bands, group="g", level="C", window=(0.0, T),
                 proved=f"같은 {P:g} W에서 candidate가 RMS를 {(1 - wc.Irms / ws.Irms) * 100:.2f} %, peak를 {(1 - wc.Ipk / ws.Ipk) * 100:.2f} % 낮춘다(정확 구간 적분, 스위칭 해로 교차검증).",
                 not_yet="global optimum이 아니다(0.1 간격 격자 + 첫 φ 근). ZVS·전체 효율 개선은 입증하지 않았다 — commutation 단계(실험 2·3)와 손실 모델이 필요하다.")
    res.add_plot("p_v", "bridge 전압: w₁ < π이면 zero interval이 생긴다", ["v1", "v2"], y_label="전압", y_unit="V", bands=bands, group="g", level="C", proved="b(θ;w,φ) 정의대로 3-level 1차 전압과 2-level 2차 전압을 만들었다.", not_yet="이상 bridge 전압이다. dead time 동안의 전압 왜곡(전류 방향에 따라 다름)과 zero-state 선택(어느 leg로 0 V를 만드는가)의 commutation 영향은 SCREEN_ONLY 범위다.")
    res.add_plot("p_g", "leg별 게이트 패턴", ["gA", "gB", "gC", "gD"], y_label="leg 상태 (위 = 상측 ON)", y_unit="", bands=bands, group="g", level="C",
                 proved=f"zero-state 패턴 ‘{v['pattern']}’: 같은 bridge 전압이라도 leg별 전환 시점과 전류가 달라진다.", not_yet="dead time은 그림에 없다(실험 3·4).")
    res.circuit = {"diagram": dab_circuit(V1, _VL(v), f"n={v['n']:.4g}", L, None).to_json(), "intervals": bands, "plot_group": "g"}
    res.tables.append(
        Table(
            "t_tab",
            "교재 E06 표 재현",
            ["항목", "SPS", "candidate"],
            [["w₁/π, w₂/π", "1, 1", f"{v['w1']:g}, {v['w2']:g}"], ["outer φ [rad]", phi_s, phi_c], ["전달전력 [W]", ws.P2, wc.P2], ["I_rms [A]", ws.Irms, wc.Irms], ["I_pk [A]", ws.Ipk, wc.Ipk]],
            note="global optimum·ZVS 달성·전체 효율 개선을 주장하지 않는다.",
        )
    )
    res.verdict("PASS_WITHIN_MODEL", "명시적 switching function으로 SPS·candidate 전력/RMS/peak를 정확 적분하고 SPS 닫힌 식·스위칭 해로 교차검증")
    res.verdict("NOT_EVALUABLE", "ZVS: 이 단계(이상 전류)에서는 판정하지 않는다 — commutation screen·event 모델 필요")
    res.assumptions += ["이상 bridge, 선형 L, zero-mean(반주기 반대칭) 전류 해", "φ는 가장 작은 양의 해 (다른 해 가지는 따로 비교해야 함)"]
    res.not_valid_for += ["ZVS 판정", "효율 개선 수치", "global optimum 주장"]
    res.interpretation = "w₁을 줄이면 1차 전압에 zero interval이 생겨 전압비 불일치(900/600 V)로 생기던 순환전류가 줄어든다. 같은 전력을 위해 φ는 커지지만 RMS와 peak가 함께 줄었다. 다만 edge 전류가 줄어 soft switching 전하도 줄 수 있어, RMS 감소가 곧 효율 이득은 아니다."
    return res


# ======================================================================================
# 2. Candidate map with staged selection
# ======================================================================================


def _cell(v, w1f, w2f, P):
    V1, V2, L, fs, n = v["V1"], v["V2"], v["L"], v["fs"], v["n"]
    w1, w2 = w1f * PI, w2f * PI
    ok1, why1 = min_pulse_ok(w1, fs, v["t_min"])
    ok2, why2 = min_pulse_ok(w2, fs, v["t_min"])
    phi = first_root(V1, V2, L, fs, P, w1, w2, v["pattern"], v["pattern"], n_scan=24)
    if phi is None:
        return {"feasible": False, "reason": "전력 해 없음", "irms": None}
    mod = Modulation(w1, w2, phi, 0.0, v["pattern"], v["pattern"])
    w = pwl_waves(V1, V2, L, fs, mod)
    reasons = []
    if not ok1:
        reasons.append("1차 " + why1)
    if not ok2:
        reasons.append("2차 " + why2)
    if w.Ipk > v["I_lim"]:
        reasons.append(f"I_pk {w.Ipk:.3g} A > 한계")
    scr = commutation_screen(w, mod, fs, n, V1, V2 / n, (v["C0_hv"], v["V0_hv"]), (v["C0_lv"], v["V0_lv"]), v["td_hv"], v["td_lv"])
    bad = [e for e in scr if e.status != "SCREEN_PASS"]
    return {"feasible": not reasons, "reason": "; ".join(reasons), "irms": w.Irms, "ipk": w.Ipk, "phi": phi, "screen_ok": not bad, "screen_bad": bad, "w": w, "mod": mod, "scr": scr}


def synthetic_loss(v, cell) -> dict:
    """ILLUSTRATIVE ONLY: synthetic device coefficients, no magnetics; not a recommendation."""
    V1, n = v["V1"], v["n"]
    w, mod = cell["w"], cell["mod"]
    # conduction: two primary switches always carry i_L; two secondary switches carry n*i2
    p_cond = 2 * v["Rds_hv"] * w.iL.integral_sq() / w.T + 2 * v["Rds_lv"] * (n * n) * w.i2.integral_sq() / w.T
    e_sw = 0.0
    hv = Node(V1, v["C0_hv"], v["V0_hv"])
    lv = Node(V1 * 0 + v["V2"] / n, v["C0_lv"], v["V0_lv"])
    for e in cell["scr"]:
        node = hv if e.bridge == "1차" else lv
        Vb = node.Vb
        t_sw = v["tsw_hv"] if e.bridge == "1차" else v["tsw_lv"]
        if e.status == "SCREEN_PASS":
            e_sw += 0.5 * Vb * abs(e.i_out) * t_sw * 0.3  # soft turn-on; residual turn-off only (synthetic factor)
        elif e.status == "CHARGE_SHORT":
            v1 = node.v_of_q(min(e.q_av, node.Qn(Vb)))
            e_sw += node.hard_on_loss(v1)
        else:
            e_sw += node.hard_on_loss(0.0) + 0.5 * Vb * abs(e.i_out) * t_sw
    return {"p_cond": p_cond, "p_sw": e_sw * v["fs"], "total": p_cond + e_sw * v["fs"]}


def run_map(v: dict) -> Result:
    res = Result("EX06", "candidate_map", "C + SCREEN")
    grid = np.round(np.arange(v["w_min"], 1.0 + 1e-9, v["w_step"]), 6)
    stages_summary = []
    for tag, P in (("nom", v["P"]), ("cor", v["P_corner"]), ("lt", v["P_light"])):
        z, notes = [], []
        cells = {}
        for w2f in grid:
            row, nrow = [], []
            for w1f in grid:
                c = _cell(v, float(w1f), float(w2f), P)
                cells[(float(w1f), float(w2f))] = c
                row.append(c["irms"] if c["feasible"] else None)
                if not c["feasible"]:
                    nrow.append(c["reason"])
                else:
                    nrow.append("screen 통과" if c["screen_ok"] else "screen 실패: " + ", ".join(f"{e.bridge}{e.leg} {EDGE_KO[e.status]}" for e in c["screen_bad"][:3]))
            z.append(row)
            notes.append(nrow)
        res.add_series(f"map_{tag}", f"I_rms ({P:g} W)", "A", grid.tolist(), grid.tolist(), z=z, notes=notes)
        feas = {k: c for k, c in cells.items() if c["feasible"]}
        st1 = min(feas.items(), key=lambda kv: kv[1]["irms"]) if feas else None
        scr_ok = {k: c for k, c in feas.items() if c["screen_ok"]}
        st2 = min(scr_ok.items(), key=lambda kv: kv[1]["irms"]) if scr_ok else None
        sps = cells.get((1.0, 1.0))
        st3 = None
        if v["synthetic_loss"] and scr_ok:
            losses = {k: synthetic_loss(v, c)["total"] for k, c in feas.items()}
            st3 = min(losses.items(), key=lambda kv: kv[1])
        mk = []
        if st1:
            mk.append({"x": st1[0][0], "y": st1[0][1], "short": "①", "label": "최소 RMS"})
        if st2 and st2[0] != (st1[0] if st1 else None):
            mk.append({"x": st2[0][0], "y": st2[0][1], "short": "②", "label": "screen 통과 최소 RMS"})
        mk.append({"x": 1.0, "y": 1.0, "short": "SPS", "label": "기준"})
        res.add_plot(f"p_{tag}", f"candidate map — {P:g} W/모듈 (빗금 = 제외, 이유는 셀 위에)", [f"map_{tag}"], x_label="w₁/π", x_unit="", y_label="w₂/π", y_unit="", kind="map", level="C",
                     markers=mk,
                     proved="각 (w₁, w₂)에서 전력 해·최소 펄스·peak 한계·leg별 부호/전하 screen을 판정해 제외 이유를 셀마다 남겼다.",
                     not_yet="screen은 edge 전류×dead time 근사(SCREEN_ONLY)다. 손실·온도는 데이터가 없어 MISSING_INPUT (합성 손실은 설명용).")
        row = [f"{P:g} W"]
        row.append(f"w₁/π={st1[0][0]:g}, w₂/π={st1[0][1]:g}, I_rms {st1[1]['irms']:.4g} A" + ("" if st1[1]["screen_ok"] else " (screen 실패)") if st1 else "없음")
        row.append(f"w₁/π={st2[0][0]:g}, w₂/π={st2[0][1]:g}, I_rms {st2[1]['irms']:.4g} A" if st2 else "없음")
        row.append((f"w₁/π={st3[0][0]:g}, w₂/π={st3[0][1]:g}, 합성 손실 {st3[1]:.3g} W" if st3 else "MISSING_INPUT (소자·자성체 손실 데이터 없음)"))
        row.append(f"I_rms {sps['irms']:.4g} A" + ("" if sps and sps.get("screen_ok") else " (screen 실패)") if sps and sps.get("irms") else "해 없음")
        stages_summary.append(row)
        changed = bool(st1 and st2 and st1[0] != st2[0])
        verdict_txt = "예 (①이 screen 실패 → ②로 이동)" if changed else ("① = ② (그대로)" if st1 and st2 else ("격자 내 screen 통과 후보 없음 → 권장 불가" if st1 else "해 없음"))
        res.add_metric(f"changed_{tag}", f"{P:g} W: ①→② 권장점 변화", verdict_txt, "", basis="최소 RMS 후보가 commutation screen을 통과하지 못하면 바뀐다")
        if st1:
            res.add_metric(f"irms1_{tag}", f"{P:g} W ① 최소 RMS", st1[1]["irms"], "A", basis=f"w₁/π={st1[0][0]:g}, w₂/π={st1[0][1]:g}")
        if st2:
            res.add_metric(f"irms2_{tag}", f"{P:g} W ② screen 통과 최소 RMS", st2[1]["irms"], "A", basis=f"w₁/π={st2[0][0]:g}, w₂/π={st2[0][1]:g}")
        if sps and sps.get("irms"):
            res.add_metric(f"irms_sps_{tag}", f"{P:g} W SPS RMS", sps["irms"], "A", basis="screen " + ("통과" if sps.get("screen_ok") else "실패"))
    res.tables.append(Table("t_stage", "단계별 권장점 (격자 탐색 — global optimum 아님)", ["운전점", "① 이상 전류 최소 RMS", "② commutation screen 통과", "③ 손실/온도", "SPS 참고"], stages_summary,
                            note="③은 실제 소자 손실(E_on/E_off·Q_oss·R_DS(on)(T))·자성체 손실·열 경계가 없으면 MISSING_INPUT이다. ‘합성 손실’을 켜면 계수가 합성값인 설명용 순위만 보여준다."))
    res.verdict("SCREEN_ONLY", "commutation 판정은 부호·전하 screen (합성 C_oss, edge 전류 × dead time)")
    res.verdict("MISSING_INPUT", "③ 손실/온도 단계: 실제 소자·자성체 손실 데이터가 없어 권장점을 확정하지 않는다")
    res.assumptions += ["격자 탐색(w 간격 입력값)과 각 셀의 가장 작은 φ 해", "합성 C_oss·dead time·최소 펄스·peak 한계 (ASSUMED)"]
    res.not_valid_for += ["global optimum", "효율·온도 개선 주장", "실제 소자 ZVS 보증"]
    res.interpretation = "RMS만 보면 zero interval을 넓힌 candidate가 좋아 보이지만, 경부하 corner에서는 edge 전류가 작아지거나 부호가 바뀌어 commutation screen을 통과하지 못한다. 그래서 단계마다 권장점이 달라질 수 있고, 마지막 손실 단계는 실제 데이터 없이는 확정하지 않는다."
    return res


# ======================================================================================
# 3. Commutation detail per switch, zero-state selection
# ======================================================================================


def run_commutation(v: dict) -> Result:
    res = Result("EX06", "commutation_detail", "C + SCREEN")
    V1, V2, L, fs, n, P = v["V1"], v["V2"], v["L"], v["fs"], v["n"], v["P"]
    rows = []
    summary = []
    for pattern in ("alternating", "fixed_lower"):
        phi = first_root(V1, V2, L, fs, P, v["w1"] * PI, v["w2"] * PI, pattern, pattern)
        if phi is None:
            summary.append([pattern, "해 없음", "", ""])
            continue
        mod = Modulation(v["w1"] * PI, v["w2"] * PI, phi, 0.0, pattern, pattern)
        w = pwl_waves(V1, V2, L, fs, mod)
        scr = commutation_screen(w, mod, fs, n, V1, V2 / n, (v["C0_hv"], v["V0_hv"]), (v["C0_lv"], v["V0_lv"]), v["td_hv"], v["td_lv"])
        for e in scr:
            rows.append([pattern, e.bridge, e.leg, e.theta * 180 / PI, "상승" if e.rising else "하강", e.i_out, e.q_av * 1e9, e.q_req * 1e9, e.m_q, EDGE_KO[e.status]])
        nbad = sum(1 for e in scr if e.status != "SCREEN_PASS")
        summary.append([pattern, f"{phi:.5f}", f"{w.Irms:.5g}", f"{len(scr) - nbad}/{len(scr)} edge 통과"])
        if pattern == v["pattern"]:
            chosen = (mod, w, scr)
    res.tables.append(Table("t_sum", "zero-state 패턴별 요약 (같은 bridge 전압, 다른 leg 전환)", ["패턴", "φ [rad]", "I_rms [A]", "screen"], summary))
    res.tables.append(Table("t_edges", "switch node별 commutation (합치지 않는다)", ["패턴", "bridge", "leg", "θ [deg]", "방향", "노드 밖 전류 [A]", "Q 가용 [nC]", "Q 필요 [nC]", "m_Q", "판정"], rows,
                            note="m_Q = (Q_available − Q_required)/Q_required. 전류가 dead time 동안 거의 일정하다는 가정이 깨지면 실제 전류 적분이 필요하다(EX02). 서로 다른 switch node의 Q budget을 합치지 않는다."))
    mod, w, scr = chosen
    xs, ys = sample_pwl(w.iL)
    res.add_series("iL", "i_L (1차)", "A", xs, ys)
    xs, ys = sample_pwl(w.i2.scaled(n))
    res.add_series("is", "i_s (2차 실제)", "A", xs, ys)
    om = TWO_PI * fs
    mk = []
    for e in scr:
        t = e.theta / om
        yv = e.i_out if e.bridge == "1차" else e.i_out
        del yv
        mk.append({"x": t, "y": w.iL.value_at(t) if e.bridge == "1차" else n * w.i2.value_at(t), "label": f"{e.leg}{'↑' if e.rising else '↓'} {'✓' if e.status == 'SCREEN_PASS' else '✗'}"})
    bands = bands_for(mod, fs, 1)
    res.add_plot("p_i", f"edge 시점의 전류 (패턴 ‘{v['pattern']}’)", ["iL", "is"], y_label="전류", y_unit="A", bands=bands, group="c", level="C", markers=mk, hlines=[{"y": 0.0, "label": "0"}],
                 proved="leg마다 전환 순간의 전류 방향과 크기를 따로 판정했다(✓ = 부호·전하 screen 통과).", not_yet="dead time 동안의 실제 전류 변화·비선형 C 궤적은 EX02 event 모델로 따로 풀어야 한다.")
    res.circuit = {"diagram": dab_circuit(V1, V2 / n, f"n={n:.4g}", L, None).to_json(), "intervals": bands, "plot_group": "c"}
    res.verdict("SCREEN_ONLY", "leg별 부호·전하 screen — 실제 ZVS 판정 아님")
    res.assumptions += ["edge 전류 × dead time 전하 근사", "고정 rail half-bridge 2Q_oss 요구전하 (EX02 가정)", "합성 C_oss (HV/LV 별도)"]
    res.not_valid_for += ["ZVS 보증", "dead time 동안 전류 변화가 큰 경우"]
    res.interpretation = "zero-state를 어떤 스위치로 만들지(패턴)에 따라 같은 bridge 전압에서도 어떤 leg가 어떤 전류에서 전환하는지가 바뀐다. 그래서 ‘이 변조는 ZVS’라는 말은 패턴·dead time·소자 전하를 붙여야 의미가 있다."
    return res


# ======================================================================================
# 4. Implementation: timer resolution, minimum pulse, mode transition offset
# ======================================================================================


def run_impl(v: dict) -> Result:
    res = Result("EX06", "implementation", "A + C")
    V1, V2, L, fs, P = v["V1"], v["V2"], v["L"], v["fs"], v["P"]
    T = 1 / fs
    dt = v["t_res"]
    dphi = TWO_PI * fs * dt
    phi = ref.sps_phi_for_power(V1, V2, L, fs, P)
    if phi is None:
        res.verdict("NO_SOLUTION", "SPS 해 없음")
        return res
    dP_lin = ref.sps_dP_dphi(V1, V2, L, fs, phi) * dphi
    dP_fin = ref.sps_power(V1, V2, L, fs, phi + dphi) - ref.sps_power(V1, V2, L, fs, phi)
    tb = abs(V1 - 900) < 1e-9 and abs(V2 - 600) < 1e-9 and abs(P - 1500) < 1e-9 and abs(dt - 10e-9) < 1e-15 and abs(L - 200e-6) < 1e-12 and abs(fs - 1e5) < 1e-6
    res.add_metric("dphi", "타이머 1 step의 Δφ = 2π f_s Δt", dphi, "rad", ref=0.006283 if tb else None, ref_label="교재 0.006283 rad", tol=1e-4)
    res.add_metric("dP_lin", "국소 ΔP = ∂P/∂φ·Δφ", dP_lin, "W", ref=20.1246 if tb else None, ref_label="정정값 20.12 W (교재 20.62 W — errata E-001)", tol=1e-4, note="교재의 20.62 W / 1.37 %는 같은 식·같은 운전점으로 재현되지 않는다")
    res.add_metric("dP_rel", "ΔP / P", dP_lin / P, "", ref=0.013416 if tb else None, ref_label="정정값 1.342 %", tol=1e-3)
    res.add_metric("dP_fin", "유한차분 P(φ+Δφ) − P(φ)", dP_fin, "W", basis="곡률 포함")
    res.add_check(check_close("국소 감도: 해석 미분 vs 유한차분 (작은 Δφ)", ref.sps_dP_dphi(V1, V2, L, fs, phi), (ref.sps_power(V1, V2, L, fs, phi + 1e-7) - ref.sps_power(V1, V2, L, fs, phi - 1e-7)) / 2e-7, 1e-6, "∂P/∂φ = V₁V₂(1−2φ/π)/(ωL) vs 중앙차분", True, "W/rad"))
    # quantised power staircase
    cmd = np.linspace(0.0, min(PI / 2, 2.5 * phi), 400)
    q = np.round(cmd / dphi) * dphi
    res.add_series("p_cmd", "연속 φ 명령의 P", "W", cmd.tolist(), [ref.sps_power(V1, V2, L, fs, x) for x in cmd], dash=True)
    res.add_series("p_q", f"Δt = {dt * 1e9:g} ns 양자화 φ의 P", "W", cmd.tolist(), [ref.sps_power(V1, V2, L, fs, x) for x in q])
    res.add_plot("p_quant", "타이머 분해능이 만드는 전력 계단", ["p_cmd", "p_q"], x_label="φ 명령", x_unit="rad", y_label="P", y_unit="W", kind="xy", level="A", vlines=[{"x": phi, "label": "운전점"}],
                 proved="한 timer step이 만드는 전력 변화를 계산했다 (이상 1-step 감도).", not_yet="dithering·폐루프 필터링·실제 timing error를 포함한 출력 리플 예측이 아니다.")
    # minimum pulse at light load for a width-modulated candidate
    rows = []
    for Pk in (1500.0, 600.0, 300.0, 150.0):
        ph = first_root(V1, V2, L, fs, Pk, v["w1"] * PI, PI)
        ok, why = min_pulse_ok(v["w1"] * PI, fs, v["t_min"])
        rows.append([Pk, ph if ph is not None else "해 없음", (v["w1"] * PI) / (TWO_PI * fs) * 1e9, ((PI - v["w1"] * PI) / (TWO_PI * fs)) * 1e9, "OK" if ok else why])
    res.tables.append(Table("t_minp", f"w₁/π = {v['w1']:g}의 펄스·zero-state 폭과 최소 펄스 {v['t_min'] * 1e9:g} ns", ["P [W]", "φ [rad]", "펄스 [ns]", "zero-state [ns]", "최소 펄스"], rows))
    # mode transition: SPS -> candidate, abrupt at a cycle boundary vs spliced at the best edge
    w1 = v["w1"] * PI
    phi_c = first_root(V1, V2, L, fs, P, w1, PI)
    m_old = Modulation(PI, PI, phi, 0.0)
    m_new = Modulation(w1, PI, phi_c, 0.0)
    w_old = pwl_waves(V1, V2, L, fs, m_old)
    w_new = pwl_waves(V1, V2, L, fs, m_new)
    k_sw = 5
    N = int(v["cycles"])
    R = v["R"]

    def run_case(splice_theta):
        class Sys(DABSystem):
            def gate_schedule(self, t0, t1):
                ev = []
                for k in range(int(math.floor(t0 / T)) - 1, int(math.ceil(t1 / T)) + 1):
                    for mm, lo, hi in ((m_old, -1e9, k_sw * T + splice_theta / (TWO_PI * fs)), (m_new, k_sw * T + splice_theta / (TWO_PI * fs), 1e9)):
                        for th in mm.breaks():
                            if th >= TWO_PI - 1e-15:
                                continue
                            tt = k * T + th / (TWO_PI * fs)
                            if lo <= tt < hi and t0 - 1e-15 <= tt < t1:
                                lv = mm.levels(th + 1e-9)
                                ev.append((tt, (lambda q, lv=lv: lv)))
                ts = k_sw * T + splice_theta / (TWO_PI * fs)
                if t0 - 1e-15 <= ts < t1:
                    lv = m_new.levels(splice_theta + 1e-9)
                    ev.append((ts, (lambda q, lv=lv: lv)))
                return sorted(ev, key=lambda e: e[0])

        s = Sys(V1, V2, L, fs, R=R, mod_of_cycle=lambda k: m_old, key=f"tr{splice_theta}")
        q0 = m_old.levels(1e-9)
        x0 = half_wave_periodic(DABSystem(V1, V2, L, fs, R=R, mod_of_cycle=lambda k: m_old, key="tr_old"), q0)
        tr = simulate(s, q0, x0, 0.0, N * T)
        means = [tr.mean(k * T, (k + 1) * T, "iL") for k in range(N)]
        return tr, means, s

    # abrupt at the cycle boundary (theta = 0)
    tr_a, mean_a, sys_a = run_case(0.0)
    # best splice: an edge instant of either modulation minimising the trajectory mismatch
    cands = sorted(set(m_old.breaks()[:-1]) | set(m_new.breaks()[:-1]))
    best = min(cands, key=lambda th: abs(w_old.iL.value_at(th / (TWO_PI * fs)) - w_new.iL.value_at(th / (TWO_PI * fs))))
    tr_b, mean_b, _ = run_case(best)
    off_pred = w_old.iL.value_at(0.0) - w_new.iL.value_at(0.0)
    res.add_metric("off_abrupt", "주기 경계에서 즉시 전환한 뒤 DC offset", mean_a[k_sw + 1], "A", ref=off_pred if R == 0 else None, ref_label="예측 i_old(0) − i_new(0)", tol=1e-6, abs_scale=abs(off_pred) + 1e-3)
    res.add_metric("off_splice", f"θ = {best * 180 / PI:.1f}°에서 이어붙인 뒤 DC offset", mean_b[k_sw + 1], "A", basis="두 정상 궤적의 전류 차가 가장 작은 edge에서 전환")
    res.add_metric("off_end", f"{N}주기 뒤 offset (즉시 전환)", mean_a[-1], "A", basis="R = 0이면 남고, R이 있으면 L/R로 감쇠")
    xs, ys = decimate_minmax(*[(tr_a.sample(['iL'], per_segment=2)[k]) for k in ('t', 'iL')], 3000)
    res.add_series("i_abrupt", "i_L: 주기 경계 즉시 전환", "A", xs, ys)
    xs, ys = decimate_minmax(*[(tr_b.sample(['iL'], per_segment=2)[k]) for k in ('t', 'iL')], 3000)
    res.add_series("i_splice", "i_L: 최적 edge에서 전환", "A", xs, ys)
    res.add_series("m_abrupt", "사이클 평균 (즉시)", "A", [(k + 0.5) * T for k in range(N)], mean_a, style="points")
    res.add_series("m_splice", "사이클 평균 (이어붙임)", "A", [(k + 0.5) * T for k in range(N)], mean_b, style="points")
    res.add_plot("p_tr", "모드 전환의 DC offset: 전환 시점이 결과를 바꾼다", ["i_abrupt", "i_splice"], y_label="i_L", y_unit="A", level="C", vlines=[{"x": k_sw * T, "label": "전환"}],
                 proved="w/φ를 순간 변경하면 전류가 새 정상 궤적에서 벗어나 DC offset이 남는다. 전환 시점을 궤적이 만나는 edge로 고르면 offset이 줄어든다.", not_yet="자속(L_m) offset과 폐루프 scheduler는 포함하지 않았다.")
    res.add_plot("p_mean", "사이클 평균 전류", ["m_abrupt", "m_splice"], y_label="평균 i_L", y_unit="A", level="C", hlines=[{"y": 0.0, "label": "0"}], proved="offset은 이상 L에서 사라지지 않는다 (R이 있으면 L/R 감쇠).", not_yet="offset의 크기는 전환 순간의 위상에 의존한다. 실제 변압기의 자화 offset(포화)과 DC 차단 C는 모델에 없으며, 전환 순서를 설계한 scheduler는 구현하지 않았다.")
    res.add_check(ledger_check(energy_ledger(tr_a, sys_a, 0.0, N * T, ["p1"], ["p2"], ["pR"], rated_power=P), what="즉시 전환 창: "))
    res.verdict("PASS_WITHIN_MODEL", "timer 감도·양자화·최소 펄스·전환 offset을 계산 (이상 bridge)")
    res.assumptions += ["이상 1-step 감도 (dithering·폐루프 없음)", "전환은 bridge 전압 패턴의 즉시 교체", f"R = {R:g} Ω"]
    res.not_valid_for += ["실제 출력 리플 예측", "scheduler 안정성 증명"]
    res.interpretation = "디지털 구현에서는 φ가 timer step으로만 움직여 전력이 계단이 되고, 경부하에서는 최소 펄스가 부드러운 제어를 끊는다. 변조를 바꿀 때 전류 궤적이 맞지 않으면 DC offset·자속 불균형이 생기므로 전환 순서와 시점을 설계해야 한다."
    return res


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "DAB TPS면 ZVS를 전영역 보장합니까?",
        "자유도가 늘어 ZVS 영역을 넓힐 수 있지만 소자 전하·edge 전류 부호·dead time·최소 펄스·전압/부하 범위라는 조건이 붙는다. zero-power/경부하의 순환전류 비용도 있다. 논문·reference의 경계를 현재 설계로 다시 검증한다.",
        "Does TPS guarantee ZVS over the whole range?",
        "It can widen the ZVS region, but only under conditions: device charge, the edge-current sign, dead time, minimum pulse, and the voltage and load range. It also costs circulating current at light load, so a published boundary has to be re-verified for this design.",
        ["조건부", "edge 전류 부호", "순환전류 비용"],
        kind="pressure",
    ),
    Question(
        "RMS가 7 % 줄었으면 효율도 좋아진 것 아닌가?",
        "고정 등가저항 기준 전도항은 13.7 % 줄지만 스위칭 edge 전류·zero-state 도통·gate 활동·자성체 손실이 달라진다. commutation screen과 손실 데이터(provenance) 없이 효율 이득으로 바꾸지 않는다.",
        "RMS dropped by seven percent. Isn't the efficiency better?",
        "Only the conduction term with a fixed resistance drops, by about fourteen percent. Edge currents, zero-state conduction, gate activity and magnetics change too, so I would not convert it into an efficiency gain without a commutation screen and loss data with known provenance.",
        ["전도항만", "edge 전류", "손실 데이터 필요"],
    ),
    Question(
        "FPGA로 φ를 10 ns 단위로 바꾸면 어떤 문제가 생기나?",
        "Δφ = 2πf_sΔt = 0.00628 rad, 이 운전점에서 ΔP ≈ 20 W(1.3 %)의 계단이 생긴다. 경부하에서 최소 펄스와 양자화가 제어를 불연속으로 만들고, 모드 전환 시 DC offset·자속 불균형이 생길 수 있다.",
        "What happens if an FPGA updates the phase in 10 ns steps?",
        "One step is 0.0063 rad, which is about 20 W or 1.3 percent at this operating point. At light load the minimum pulse and the quantization make the control discontinuous, and abrupt mode changes can leave a DC offset and flux imbalance.",
        ["Δφ 계산", "ΔP", "최소 펄스", "전환 offset"],
        kind="calc",
    ),
]

EXPERIMENTS = [
    Experiment(
        key="general_modulation",
        title="switching function을 고정하고 정확 적분: SPS vs width candidate",
        goal="b(θ; w, φ)로 w₁, w₂, φ 세 자유도를 구현하고, 900/600 V·200 µH·100 kHz·1.5 kW에서 SPS 3.113563 Arms와 candidate(w₁/π = 0.7) 2.893092 Arms를 정확 구간 적분으로 재현한다. SPS는 닫힌 식으로, candidate는 스위칭 해로 독립 검증한다.",
        params=_params([
            Param("w1", "1차 펄스 폭 w₁/π", "", 0.7, "", vmin=0.05, vmax=1.0, source="TEXTBOOK", source_note="0.7"),
            Param("w2", "2차 펄스 폭 w₂/π", "", 1.0, "", vmin=0.05, vmax=1.0, source="TEXTBOOK", source_note="1.0"),
            Param("pattern", "zero-state 패턴", "", "alternating", kind="choice", choices=[("alternating", "alternating (leg 50 %)"), ("fixed_lower", "fixed lower (leg 펄스)")], source="ASSUMED", group="구현"),
        ]),
        presets=[Preset("textbook", "교재 candidate w₁/π 0.7", {}, "E06 표", ("nominal", "reference")), Preset("sps", "SPS (w₁ = w₂ = π)", {"w1": 1.0}, "", ("variant", "reference")), Preset("w06", "w₁/π 0.6", {"w1": 0.6}, "", ("variant",))],
        run=run_general,
        model_level="C + A",
        suggested_change="w₁/π를 0.7 → 0.6으로 더 줄인다.",
        prediction=Prediction(
            "w₁/π를 0.7 → 0.6으로 줄이면 같은 1.5 kW의 RMS는?",
            ["계속 줄어든다", "다시 늘어난다", "해가 없다", "모르겠다"],
            "다시 늘어난다",
            "zero interval을 넓히면 불일치 순환전류는 줄지만, 같은 전력을 위해 φ와 전류 기울기 구간이 커져 어느 폭 이하에서는 RMS가 다시 증가한다. 최적은 전압비·전력에 따라 움직인다.",
            ["irms_c", "ipk_c", "phi_c"],
            handcalc=[{"key": "irms_sps", "label": "SPS I_rms", "unit": "A"}],
        ),
        suggested={"w1": 0.6},
        student="1차 bridge가 잠깐 0 V를 내는 구간(zero state)을 두면 두 bridge 전압 차가 줄어든 시간 동안 쓸데없는 순환전류가 덜 흐른다.",
        expert="논문마다 DPS/TPS 정의가 다르므로 구현이 어떤 gate pattern인지 보여준다. 같은 w라도 zero state를 위/아래 중 어디로 만드느냐에 따라 leg별 commutation이 달라진다. candidate는 0.1 격자 + 첫 φ 근이라 global optimum이 아니며, 다른 φ 해 가지(branch)는 제어 부호가 다를 수 있다.",
        customer_ko="low-battery corner(900/600 V)에서 width 변조로 1.5 kW의 RMS를 약 7 %, peak를 약 12 % 낮출 수 있습니다. 다만 ZVS와 전체 효율은 아직 검증하지 않았으니 edge 전류·dead time 기준의 commutation 확인과 손실 데이터로 판단하시죠.",
        customer_en="At the low-battery corner, 900 to 600 V, width modulation lowers the RMS current by about seven percent and the peak by about twelve percent at 1.5 kW. ZVS and overall efficiency are not verified yet, so the next step is a commutation check with the edge currents and dead time, and then loss data.",
        questions=_Q[:2],
        circuit="dab",
        textbook=[TB_E06, TB_11, TB_E12],
        reference_presets=["textbook", "sps"],
        claim_limit="이상 전류 파형의 RMS/peak. ZVS·효율 개선 주장 없음.",
    ),
    Experiment(
        key="candidate_map",
        title="candidate map과 단계별 권장점: 이상 전류 → commutation → 손실",
        goal="w₁, w₂ 격자(0.1 간격)에서 전력 해·최소 펄스·peak 한계·leg별 부호/전하 screen의 feasible/infeasible 이유를 표시하고, 1.5 kW·2 kW·300 W에서 ① 최소 RMS ② screen 통과 ③ 손실(MISSING_INPUT) 단계의 권장점을 비교한다. 2 kW에서 권장점이 바뀌고, 1.5 kW·300 W에서는 격자 안에 screen 통과 후보가 없다는 것까지 결과로 본다.",
        params=_params(
            [
                Param("P_corner", "비교 corner 전력 (고부하)", "W", 2000.0, "W", vmin=1, vmax=20000, source="ASSUMED", source_note="권장점이 바뀌는 corner"),
                Param("P_light", "비교 corner 전력 (경부하)", "W", 300.0, "W", vmin=1, vmax=20000, source="ASSUMED"),
                Param("w_min", "w/π 격자 최소", "", 0.3, "", vmin=0.05, vmax=0.95, source="ASSUMED", group="격자"),
                Param("w_step", "w/π 격자 간격", "", 0.1, "", vmin=0.01, vmax=0.2, source="TEXTBOOK", source_note="교재 0.1 간격", group="격자"),
                Param("pattern", "zero-state 패턴", "", "alternating", kind="choice", choices=[("alternating", "alternating"), ("fixed_lower", "fixed lower")], source="ASSUMED", group="구현"),
                Param("synthetic_loss", "③ 합성 손실 모델 사용 (설명용)", "", False, kind="bool", source="ASSUMED", group="손실"),
                Param("Rds_hv", "합성 HV R_DS(on)", "Ω", 0.08, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="손실"),
                Param("Rds_lv", "합성 LV R_DS(on)", "Ω", 0.002, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="손실"),
                Param("tsw_hv", "합성 HV 전환시간", "s", 20e-9, "ns", vmin=0, vmax=1e-6, source="ASSUMED", group="손실"),
                Param("tsw_lv", "합성 LV 전환시간", "s", 10e-9, "ns", vmin=0, vmax=1e-6, source="ASSUMED", group="손실"),
            ]
            + _screen
        ),
        presets=[
            Preset("textbook", "900/600 V: 1.5 kW · 2 kW · 300 W", {}, "", ("nominal", "reference")),
            Preset("fine", "격자 0.05 간격", {"w_step": 0.05}, "더 촘촘히 (느림)", ("variant",)),
            Preset("synthetic", "③ 합성 손실 켜기 (설명용)", {"synthetic_loss": True}, "순위가 바뀔 수 있음을 보이는 설명용", ("variant",)),
            Preset("matched", "전압비 일치 800/800 V", {"V1": 800.0, "V2": 800.0}, "", ("variant",)),
        ],
        run=run_map,
        model_level="C + SCREEN",
        suggested_change="③ 합성 손실 모델을 켜서 RMS 순위와 손실 순위가 같은지 본다 (설명용).",
        prediction=Prediction(
            "2 kW corner에서 ① 최소 RMS 후보(zero interval을 넣은 width candidate)는 commutation screen도 통과할까?",
            ["통과한다", "통과하지 못해 ② 권장점이 바뀐다", "해가 없다", "모르겠다"],
            "통과하지 못해 ② 권장점이 바뀐다",
            "zero interval을 넓혀 RMS를 줄이면 2차 edge 전류가 작아지거나 부호가 바뀌어 soft switching 조건을 잃는다. 권장점은 RMS가 조금 더 큰(약 +0.5 %) 후보로 이동한다. 1.5 kW·300 W에서는 격자 안에 통과 후보가 없어 권선비·f_s·변조 범위 재설계가 필요하다.",
            ["changed_nom", "changed_cor", "changed_lt", "irms1_cor", "irms2_cor"],
        ),
        suggested={"synthetic_loss": True},
        student="전류를 줄이는 쪽과 스위치를 부드럽게 켜는 데 필요한 전류를 남기는 쪽이 서로 부딪힌다.",
        expert="목적함수 min(P_cond + P_sw + P_mag + P_gate) s.t. P = P*, I_pk ≤ I_lim, |B| ≤ B_lim에 edge별 전류 부호·비선형 전하·최소 펄스·dead time·제어 권한을 더한다. soft switching용 전류를 의도적으로 남기면 RMS 최소점보다 총손실이 낮을 수 있고, 경부하에서 ZVS를 위해 순환전류를 과하게 강제하면 효율이 나빠질 수 있다. ‘전영역 ZVS’에는 전압·전력 범위·dead time·소자 용량·온도를 붙인다.",
        customer_ko="RMS가 가장 작은 변조점이 경부하 corner에서 soft switching 조건을 잃습니다. commutation screen을 통과하는 후보로 권장점을 옮기고, 최종 선택은 실제 소자 손실·자성체 데이터로 정하시죠. 현재 데이터로는 손실 순위를 확정하지 않습니다.",
        customer_en="The lowest-RMS modulation point loses its soft-switching conditions at the light-load corner. I would move the recommendation to a candidate that passes the commutation screen, and make the final choice with real device-loss and magnetics data. With today's data the loss ranking stays open.",
        questions=_Q[:2],
        circuit="dab",
        textbook=[TB_E06],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="격자 탐색 + SCREEN_ONLY. 손실 단계는 MISSING_INPUT (합성은 설명용).",
    ),
    Experiment(
        key="commutation_detail",
        title="switch node별 commutation: zero-state 선택이 전환 전류를 바꾼다",
        goal="candidate 한 점에서 모든 leg 전환의 전류 방향·전하 여유를 node별로 따로 판정하고, zero-state 패턴(alternating vs fixed lower)이 어떤 스위치를 어떤 전류에서 전환시키는지 비교한다.",
        params=_params([
            Param("w1", "w₁/π", "", 0.7, "", vmin=0.05, vmax=1.0, source="TEXTBOOK"),
            Param("w2", "w₂/π", "", 1.0, "", vmin=0.05, vmax=1.0, source="TEXTBOOK"),
            Param("pattern", "그래프에 표시할 패턴", "", "alternating", kind="choice", choices=[("alternating", "alternating"), ("fixed_lower", "fixed lower")], source="ASSUMED", group="구현"),
        ] + _screen),
        presets=[Preset("textbook", "교재 candidate", {}, "", ("nominal", "reference")), Preset("light", "300 W", {"P": 300.0}, "", ("corner", "reference"))],
        run=run_commutation,
        model_level="C + SCREEN",
        suggested_change="전력을 1500 → 300 W로 낮춰 어느 leg가 먼저 screen에 실패하는지 본다.",
        prediction=Prediction(
            "같은 w₁/π = 0.7에서 전력을 1.5 kW → 300 W로 낮추면 1차 A leg의 전하 screen은?",
            ["더 나빠진다", "오히려 통과한다", "변화 없다", "모르겠다"],
            "오히려 통과한다",
            "w₁을 고정한 채 φ가 작아지면 A leg 전환 시점이 전류 파형의 다른 위치로 옮겨 edge 전류가 0.51 A → 2.15 A로 커진다. ‘경부하 = ZVS에 불리’라는 직관을 leg별 계산으로 확인해야 하는 이유다. 2차 leg는 두 경우 모두 전류 방향이 반대(hard switching)다.",
            [],
        ),
        suggested={"P": 300.0},
        student="bridge 전압이 같아도 네 개의 leg는 각자 다른 순간, 다른 전류에서 켜지고 꺼진다. 각 leg의 노드를 따로 봐야 한다.",
        expert="서로 다른 switch node의 Q budget을 하나로 합쳐 모든 leg ZVS를 통과시키지 않는다. 2차(LV)는 전압이 낮아 필요 전하가 작지만 전류가 n배라 기준이 다르다. 전류가 dead time 동안 크게 변하면 I_edge·t_d 대신 EX02 event 모델로 푼다.",
        customer_ko="이 변조점에서는 leg마다 전환 조건이 다릅니다. 1.5 kW에서 1차 A leg는 전하 부족으로 부분 전환, 2차 leg는 전류 방향이 반대라 hard switching이 예상되고, 경부하에서는 오히려 1차가 나아집니다. 해당 leg의 turn-on 직전 V_DS와 전류를 우선 측정해 주시면 좋겠습니다.",
        customer_en="Each leg has its own switching condition at this modulation point. At 1.5 kW the primary A leg is short of charge and may switch partially, while the secondary legs see the wrong current direction and hard-switch. At light load the primary actually improves. I would first measure the drain-source voltage and current just before turn-on on those legs.",
        questions=_Q[:1],
        circuit="dab",
        textbook=[TB_E06],
        reference_presets=["textbook", "light"],
        claim_limit="SCREEN_ONLY",
    ),
    Experiment(
        key="implementation",
        title="FPGA/MCU 구현: timer 분해능·최소 펄스·모드 전환 offset",
        goal="Δt = 10 ns이면 Δφ = 0.006283 rad, SPS 운전점의 국소 ΔP ≈ 20.12 W(1.34 %)를 계산하고(교재 20.62 W는 errata), 경부하 최소 펄스와 SPS→candidate 모드 전환이 남기는 DC offset을 스위칭 해로 보인다.",
        params=_params([
            Param("t_res", "타이머 분해능 Δt", "s", 10e-9, "ns", vmin=1e-12, vmax=1e-6, source="TEXTBOOK", source_note="10 ns"),
            Param("w1", "전환 대상 w₁/π", "", 0.7, "", vmin=0.05, vmax=1.0, source="TEXTBOOK"),
            Param("t_min", "최소 펄스", "s", 200e-9, "ns", vmin=0, vmax=5e-6, source="ASSUMED"),
            Param("R", "직렬 저항 (offset 감쇠)", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED"),
            Param("cycles", "시뮬레이션 주기", "", 40, "", vmin=10, vmax=2000, kind="int", source="ASSUMED"),
        ]),
        presets=[Preset("textbook", "10 ns, 즉시 전환", {}, "E06", ("nominal", "reference")), Preset("damped", "R = 1 Ω", {"R": 1.0, "cycles": 200}, "", ("variant",)), Preset("t100", "Δt = 100 ns", {"t_res": 100e-9}, "", ("variant",))],
        run=run_impl,
        model_level="A + C",
        suggested_change="타이머 분해능을 10 → 100 ns로 바꾼다.",
        prediction=Prediction(
            "Δt를 10 → 100 ns로 늘리면 한 step의 ΔP는?",
            ["약 10배 (≈ 200 W)", "약 2배", "변화 없음", "모르겠다"],
            "약 10배 (≈ 200 W)",
            "Δφ = 2πf_sΔt에 비례하고 국소 감도 ∂P/∂φ는 같으므로 ΔP도 약 10배다(곡률 때문에 정확히 10배는 아니다).",
            ["dP_lin", "dP_rel"],
            handcalc=[{"key": "dphi", "label": "Δφ", "unit": "rad"}, {"key": "dP_lin", "label": "ΔP", "unit": "W"}],
        ),
        suggested={"t_res": 100e-9},
        student="디지털 타이머는 시간을 계단으로만 바꿀 수 있어 φ도 계단이 되고, 그 한 계단이 전력으로 얼마인지가 제어의 해상도다.",
        expert="교재 E06의 ‘ΔP ≈ 20.62 W(1.37 %)’는 같은 식 ∂P/∂φ = V₁V₂′(1−2φ/π)/(ω_sL)과 같은 운전점으로 20.12 W(1.342 %)가 되어 errata E-001로 기록했다. 모드 전환은 map interpolation·hysteresis·전환 순서와 cycle balance를 설계해야 하며, RMS 최소점들을 독립적으로 골라 이어 붙인 것은 안정된 scheduler가 아니다.",
        customer_ko="10 ns timer에서 φ 한 step이 약 20 W(1.3 %)에 해당합니다. 경부하 정밀 제어가 필요하면 dithering이나 f_s·변조 전략을 같이 보고, 변조 전환은 전류 궤적이 맞는 시점에 하도록 순서를 설계하시죠.",
        customer_en="With a 10 ns timer one phase step is worth about 20 W, 1.3 percent at this point. If light-load precision matters, consider dithering or a different switching-frequency or modulation strategy, and sequence mode changes at instants where the current trajectories match.",
        questions=_Q[2:],
        circuit="dab",
        textbook=[TB_E06],
        reference_presets=["textbook"],
        claim_limit="이상 1-step 감도와 이상 bridge 전환 offset.",
    ),
]

LAB = Lab(
    id="EX06",
    title="DAB: RMS 최소화와 ZVS의 상충관계",
    title_en="DAB modulation: RMS minimisation vs ZVS",
    track="expert",
    order=6,
    path_note="E13 3회전",
    textbook=[TB_E06, TB_11, TB_E12],
    prerequisites=["FL08", "EX02"],
    summary="명시적 switching function → SPS·candidate 정확 적분 → feasible map과 단계별 권장점 → node별 commutation → timer·최소 펄스·전환 offset.",
    experiments=EXPERIMENTS,
    minimum_scope="w₁, w₂, φ switching function; 정확 PWL P/Irms/Ipk; SPS 닫힌 식 독립 검증; 3.113563 → 2.893092 A; commutation sign/charge·zero-state·최소 펄스·dead time·timer; 전환 transient·offset",
    claim_limits=["global optimum 아님 (격자 + 첫 φ 해)", "ZVS는 SCREEN_ONLY, 이상 전류 단계는 NOT_EVALUABLE", "RMS 감소 ≠ 효율 이득; 손실 단계는 MISSING_INPUT (합성 손실은 설명용)"],
    test_paths=["tests/test_ex06.py"],
    extends=["FL08", "EX02"],
)
