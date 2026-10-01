"""FL10 - CLLC: defend a design failure and multiple solutions (textbook ch.13).

Level A: the textbook FHA network (Z_p = Z_m || (Z_r2' + R_ac'), H = Z_p/(Z_r1 + Z_p) R_ac'/(Z_r2' + R_ac')),
computed by the lab in admittance form and compared with the textbook closed form (reference/resonant.py).
The seed design (n = 1) must FAIL at 920 V / 850 V (max inductive gain 1.016401 < required 1.082353) and
stays FAIL.  The n = 0.93 candidate with referred symmetry has the textbook roots 164.390 / 162.811 kHz and,
at the high corner, 136.099 and 147.061 kHz.  Reverse power flow is derived again with the tanks swapped
(it is not the reciprocal of the forward gain).
Level C: the exact switched model with a stiff battery (both ports voltage-stiff) shows where 11 kW is
really delivered on each branch; FHA gain PASS, switching, ZVS, reverse and start-up are separate verdicts.
"""

from __future__ import annotations

import math

import numpy as np

from ..engine.switched import simulate
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table
from ..reference import resonant as ref
from ._common import energy_ledger, ledger_check, series_from_traj
from ._resonant import (
    PI,
    ZVS_KO,
    ResonantSystem,
    Tank,
    bands_from,
    bridge_fundamental,
    edge_currents,
    fha,
    fha_roots,
    gain_scan,
    ivp_period,
    max_inductive_gain,
    periodic,
    periodic_warm,
    phasor_guess,
    rac,
    resonant_circuit,
    scales_for,
    summarize,
    zvs_screen,
)

TB_13 = TextbookRef("cllc-설계-실패와-다중-해를-직접-방어하기-fl10", "13. CLLC — 설계 실패와 다중 해를 직접 방어하기 [FL10]")
TB_12 = TextbookRef("llc-공진을-말로-설명하고-식으로-확인하기-fl09", "12. LLC [FL09]")
TB_E05 = TextbookRef("expert-e05-cllc-gain-곡선에서-실제-동역학으로", "E05 · CLLC: gain 곡선에서 실제 동역학으로")
TB_E12 = TextbookRef("expert-e12-설계-리뷰를-통과하는-답변-세-개의-통합-사례", "E12 · 설계 리뷰 통합 사례 (CLLC 고객 상황)")

C1_TB = 28.144773e-9
COSS = (0.5e-9, 40.0)  # synthetic HV C(v) = C0/sqrt(1+v/V0) (ASSUMED)


# --------------------------------------------------------------------------------------
# Parameters and shared calculations
# --------------------------------------------------------------------------------------


def _tank_params(n: float = 1.0) -> list[Param]:
    return [
        Param("L1", "1차 직렬 L₁", "H", 40e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", source_note="L_r1 = 40 µH", group="tank"),
        Param("C1", "1차 직렬 C₁", "F", C1_TB, "nF", vmin=1e-11, vmax=1e-4, source="TEXTBOOK", source_note="28.144773 nF (f_r 150 kHz)", group="tank"),
        Param("Lm", "여자 L_m", "H", 200e-6, "µH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", group="tank"),
        Param("n", "권선비 n = N_p/N_s", "", n, "", vmin=0.2, vmax=5.0, source="TEXTBOOK", source_note="seed 1, 수정안 0.93", group="tank"),
        Param("symmetric", "환산 대칭 유지 (L₂ = L₁/n², C₂ = C₁n²)", "", True, "", kind="bool", source="TEXTBOOK", source_note="끄면 실제 L₂·C₂ = 40 µH·28.14 nF 유지 (다른 회로)", group="tank"),
    ]


def _op_params() -> list[Param]:
    return [
        Param("P", "배터리 출력", "W", 11000.0, "kW", vmin=10, vmax=1e6, source="TEXTBOOK", source_note="11 kW"),
        Param("Vbat_hi", "고전압 corner 배터리", "V", 920.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="corner"),
        Param("Vlink_hi", "고전압 corner link", "V", 850.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="corner"),
        Param("Vbat_mid", "중간 corner 배터리", "V", 800.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="corner"),
        Param("Vlink_mid", "중간 corner link", "V", 800.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="corner"),
        Param("Vbat_lo", "저전압 corner 배터리", "V", 650.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="corner"),
        Param("Vlink_lo", "저전압 corner link", "V", 700.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="corner"),
        Param("f_min", "주파수 하한", "Hz", 120e3, "kHz", vmin=1e3, vmax=1e7, source="TEXTBOOK", source_note="120–210 kHz", group="범위"),
        Param("f_max", "주파수 상한", "Hz", 210e3, "kHz", vmin=1e3, vmax=1e7, source="TEXTBOOK", group="범위"),
    ]


def tank_of(v: dict) -> tuple[Tank, float, float]:
    """Referred tank and the PHYSICAL secondary L2, C2."""
    n = v["n"]
    if v["symmetric"]:
        L2p, C2p = v["L1"], v["C1"]
        L2, C2 = v["L1"] / n**2, v["C1"] * n**2
    else:
        L2, C2 = 40e-6, C1_TB
        L2p, C2p = n**2 * L2, C2 / n**2
    return Tank(v["L1"], v["C1"], v["Lm"], L2p, C2p), L2, C2


def corners(v: dict) -> list[tuple[str, float, float]]:
    return [("저전압", v["Vbat_lo"], v["Vlink_lo"]), ("중간", v["Vbat_mid"], v["Vlink_mid"]), ("고전압", v["Vbat_hi"], v["Vlink_hi"])]


def corner_fha(tank: Tank, n: float, Vbat: float, Vlink: float, P: float, f_lo: float, f_hi: float) -> dict:
    Racv = rac(n, Vbat, P)
    g_req = n * Vbat / Vlink
    roots = fha_roots(tank, Racv, g_req, f_lo, f_hi)
    gmax, fmax = max_inductive_gain(tank, Racv, f_lo, f_hi)
    V1f_rms = bridge_fundamental(Vlink, "FB") / math.sqrt(2)
    for r in roots:
        r["I1_rms"] = V1f_rms / abs(r["point"].Zin)
        r["phase_deg"] = math.degrees(math.atan2(r["point"].Zin.imag, r["point"].Zin.real))
    ind = [r for r in roots if r["inductive"]]
    return {"Rac": Racv, "g_req": g_req, "gmax": gmax, "fmax": fmax, "roots": roots, "inductive_roots": ind, "ok": bool(ind)}


def _tb_design(v: dict) -> bool:
    return abs(v["L1"] - 40e-6) < 1e-12 and abs(v["C1"] - C1_TB) < 1e-15 and abs(v["Lm"] - 200e-6) < 1e-12 and abs(v["P"] - 11000) < 1e-9 and abs(v["f_min"] - 120e3) < 1e-6 and abs(v["f_max"] - 210e3) < 1e-6


def _tb_corners(v: dict) -> bool:
    return (v["Vbat_lo"], v["Vlink_lo"], v["Vbat_mid"], v["Vlink_mid"], v["Vbat_hi"], v["Vlink_hi"]) == (650.0, 700.0, 800.0, 800.0, 920.0, 850.0)


def _gain_plot(res: Result, tank: Tank, v: dict, per_corner: list, key: str, title: str, proved: str, not_yet: str):
    fs = None
    names = []
    hl = []
    mk = []
    for (name, Vbat, Vlink), cf in per_corner:
        fs, g, xi = gain_scan(tank, cf["Rac"], v["f_min"], v["f_max"], 361)
        sk = f"{key}_{name}"
        res.add_series(sk, f"{name} {Vbat:g}/{Vlink:g} V", "", fs.tolist(), g.tolist())
        names.append(sk)
        hl.append({"y": cf["g_req"], "label": f"{name} 필요 {cf['g_req']:.4f}"})
        tag = {"저전압": "저", "중간": "중", "고전압": "고"}.get(name, name[:1])
        for i, r in enumerate(cf["roots"]):
            mk.append({"x": r["f"], "y": cf["g_req"], "short": f"{tag}{i + 1}", "label": f"{name} 해 {r['f'] / 1e3:.3f} kHz" + (" (inductive)" if r["inductive"] else " (capacitive — 제외)")})
        if math.isfinite(cf["gmax"]):
            mk.append({"x": cf["fmax"], "y": cf["gmax"], "short": f"{tag}▲", "label": f"{name} inductive 최대 {cf['gmax']:.4f} @ {cf['fmax'] / 1e3:.2f} kHz"})
    res.add_plot(key, title, names, x_label="f_s", x_unit="Hz", y_label="|H| (n V_bat/V_link 비교)", y_unit="", kind="xy", level="A", hlines=hl, markers=mk, proved=proved, not_yet=not_yet)


def _check_textbook_form(res: Result, tank: Tank, per_corner: list, v: dict):
    dev = 0.0
    for _, cf in per_corner:
        for f in np.linspace(v["f_min"], v["f_max"], 61):
            a = fha(tank, float(f), cf["Rac"]).H
            b = ref.cllc_H(float(f), tank.L1, tank.C1, tank.Lm, tank.L2, tank.C2, cf["Rac"])
            dev = max(dev, abs(a - b))
    res.add_check(Check("교재 CLLC 식 vs 어드미턴스 절점해석", "PASS" if dev < 1e-12 else "FAIL", dev, "", 1e-12, path="H = Z_p/(Z_r1+Z_p)·R_ac′/(Z_r2′+R_ac′) (reference) vs I₁ = V₁/(Z₁ + 1/(Y_m+Y_b))", independent=True, detail="corner 3개 × 61점의 최대 |ΔH|"))


# ======================================================================================
# 1. Seed design: FAIL preserved
# ======================================================================================


def run_seed(v: dict) -> Result:
    res = Result("FL10", "seed_fail", "A (FHA)")
    tank, L2, C2 = tank_of(v)
    n = v["n"]
    res.add_metric("fr", "1차 공진 f_r", tank.fr1, "Hz", ref=150e3 if _tb_design(v) else None, ref_label="교재 150 kHz", tol=1e-6)
    per = [(c, corner_fha(tank, n, c[1], c[2], v["P"], v["f_min"], v["f_max"])) for c in corners(v)]
    hi = per[2][1]
    tb = _tb_design(v) and _tb_corners(v) and abs(n - 1.0) < 1e-12 and v["symmetric"]
    res.add_metric("g_req_hi", "고전압 corner 필요 gain n·V_bat/V_link", hi["g_req"], "", ref=1.082353 if tb else None, ref_label="교재 1.082353", tol=5e-7)
    res.add_metric("gmax_hi", "inductive 영역 최대 FHA gain", hi["gmax"], "", ref=1.016401 if tb else None, ref_label="교재 1.016401", tol=5e-7, basis=f"f = {hi['fmax'] / 1e3:.2f} kHz, R_ac = {hi['Rac']:.3f} Ω")
    res.add_metric("margin_hi", "gain 여유 (최대/필요 − 1)", hi["gmax"] / hi["g_req"] - 1, "", basis="음수 = 해 없음")
    rows = []
    for (name, Vbat, Vlink), cf in per:
        rts = ", ".join(f"{r['f'] / 1e3:.3f} kHz{'' if r['inductive'] else ' (capacitive)'}" for r in cf["roots"]) or "없음"
        rows.append([f"{name} {Vbat:g}/{Vlink:g} V", f"{cf['g_req']:.6f}", f"{cf['gmax']:.6f} @ {cf['fmax'] / 1e3:.2f} kHz", rts, "해 있음 (inductive)" if cf["ok"] else "FAIL: 범위 내 inductive 해 없음"])
    res.tables.append(Table("t_corners", f"corner별 FHA 판정 (n = {n:g}, {v['f_min'] / 1e3:g}–{v['f_max'] / 1e3:g} kHz)", ["corner", "필요 gain", "최대 inductive gain", "해", "판정"], rows,
                            note="필요 gain = n·V_bat/V_link (full bridge 양측). inductive = Im(Z_in) > 0 (FHA). capacitive 해는 ZVS를 잃으므로 운전점으로 쓰지 않는다."))
    _gain_plot(res, tank, v, per, "p_gain", f"FHA gain과 필요 gain (n = {n:g}): 고전압 corner의 선이 곡선 위에 있다", "세 corner의 R_ac로 gain 곡선을 그리고 필요 gain과 비교했다.", "FHA뿐 — 스위칭·ZVS·기동은 아직.")
    _check_textbook_form(res, tank, per, v)
    fs, g, xi = gain_scan(tank, hi["Rac"], v["f_min"], v["f_max"], 90001)
    dense = float(np.max(np.where(xi > 0, g, -np.inf)))
    res.add_check(Check("최대 inductive gain: golden refine vs 90001점 격자", "PASS" if abs(dense - hi["gmax"]) < 1e-7 else "FAIL", abs(dense - hi["gmax"]), "", 1e-7, path="두 탐색 방법의 비교 (격자 1 Hz)", independent=True))
    if not hi["ok"]:
        res.verdict("FAIL_CONSTRAINT", f"고전압 corner {v['Vbat_hi']:g}/{v['Vlink_hi']:g} V: 필요 {hi['g_req']:.6f} > inductive 최대 {hi['gmax']:.6f} — 지정 주파수 범위에 해가 없다 (seed 실패는 결과로 보존)")
    else:
        res.verdict("CANDIDATE_FHA_ONLY", "세 corner 모두 FHA inductive 해가 있다 — 스위칭·ZVS·reverse·기동은 별도 검증")
    for (name, Vbat, Vlink), cf in per[:2]:
        if not cf["ok"]:
            res.verdict("FAIL_CONSTRAINT", f"{name} corner에도 inductive 해가 없다")
    res.assumptions += ["FHA (기본파), full-bridge 양측, R_ac = 8n²V_bat²/(π²P)", "합성 학습 사양 (교재 13장)", "이상 소자·무손실"]
    res.not_valid_for += ["스위칭 파형·ZVS", "reverse", "기동"]
    res.interpretation = "대칭 CLLC(n = 1)의 FHA gain은 11 kW에서 1을 조금 넘는 정도가 최대다. 920 V 배터리를 850 V link에서 만들려면 1.082가 필요하므로 회로를 만들기 전에 이 corner가 불가능하다는 것을 식으로 잡는다. 실패는 지우지 않고 설계 변경의 출발점으로 쓴다."
    res.circuit = {"diagram": resonant_circuit("CLLC", v["Vlink_hi"], f"{n:g}", tank, "FB", out="battery", vo_label=f"{v['Vbat_hi']:g} V").to_json(), "intervals": [], "plot_group": ""}
    return res


# ======================================================================================
# 2. Candidate n = 0.93 with referred symmetry
# ======================================================================================


def run_fix(v: dict) -> Result:
    res = Result("FL10", "fix_n093", "A (FHA)")
    tank, L2, C2 = tank_of(v)
    n = v["n"]
    tb = _tb_design(v) and _tb_corners(v) and abs(n - 0.93) < 1e-12 and v["symmetric"]
    res.add_metric("L2", "2차 실제 L₂", L2, "H", ref=46.248121e-6 if tb else None, ref_label="교재 46.248121 µH", tol=1e-8, basis="L₂ = L₁/n² (환산 대칭)" if v["symmetric"] else "실제값 유지 → L₂′ = n²L₂")
    res.add_metric("C2", "2차 실제 C₂", C2, "F", ref=24.342414e-9 if tb else None, ref_label="교재 24.342414 nF", tol=1e-8, basis="C₂ = C₁n²" if v["symmetric"] else "실제값 유지 → C₂′ = C₂/n²")
    res.add_metric("L2p", "1차 환산 L₂′ = n²L₂", tank.L2, "H")
    res.add_metric("C2p", "1차 환산 C₂′ = C₂/n²", tank.C2, "F")
    per = [(c, corner_fha(tank, n, c[1], c[2], v["P"], v["f_min"], v["f_max"])) for c in corners(v)]
    refs = {"저전압": [164.390e3], "중간": [162.811e3], "고전압": [136.099e3, 147.061e3]}
    rows = []
    all_ok = True
    for (name, Vbat, Vlink), cf in per:
        ind = cf["inductive_roots"]
        all_ok &= bool(ind)
        for i, r in enumerate(ind):
            key = f"f_{name}_{i}"
            rf = refs[name][i] if tb and i < len(refs[name]) else None
            res.add_metric(key, f"{name} {Vbat:g}/{Vlink:g} V inductive 해 #{i + 1}", r["f"], "Hz", ref=rf, ref_label="교재 표" if rf else "", tol=5e-6 if rf else None, basis=f"필요 gain {cf['g_req']:.6f}")
        for r in cf["roots"]:
            rows.append([f"{name} {Vbat:g}/{Vlink:g}", f"{cf['g_req']:.6f}", f"{r['f'] / 1e3:.3f} kHz", "inductive" if r["inductive"] else "capacitive (제외)", f"{r['slope_per_Hz'] * 1e3:+.6f} /kHz", f"{r['I1_rms']:.3f} A", f"{r['phase_deg']:+.1f}°"])
        if not cf["roots"]:
            rows.append([f"{name} {Vbat:g}/{Vlink:g}", f"{cf['g_req']:.6f}", "없음", "-", "-", "-", "-"])
    hi = per[2][1]
    ind = hi["inductive_roots"]
    if len(ind) == 2:
        a, b = ind
        res.add_metric("slope_lo", "고전압 해 #1 gain 기울기 [1/kHz] (±10 Hz 중앙차분)", a["slope_per_Hz"] * 1e3, "", ref=0.001871 if tb else None, ref_label="교재 +0.001871/kHz", tol=5e-3)
        res.add_metric("slope_hi", "고전압 해 #2 gain 기울기 [1/kHz]", b["slope_per_Hz"] * 1e3, "", ref=-0.001803 if tb else None, ref_label="교재 −0.001803/kHz", tol=5e-3)
        res.add_metric("I1_lo", "해 #1 FHA 1차 직렬 RMS", a["I1_rms"], "A", ref=14.390 if tb else None, ref_label="교재 약 14.390 A", tol=5e-4, basis="V₁ 기본파 RMS/|Z_in| — 스위칭 실측값 아님")
        res.add_metric("I1_hi", "해 #2 FHA 1차 직렬 RMS", b["I1_rms"], "A", ref=14.765 if tb else None, ref_label="교재 약 14.765 A", tol=5e-4)
    res.tables.append(Table("t_roots", "corner별 FHA 해 (필요 gain을 만족하는 모든 주파수)", ["corner", "필요 gain", "해", "Z_in", "기울기", "I₁,rms (FHA)", "∠Z_in"], rows,
                            note="같은 gain의 두 해는 기울기 부호·RMS·ZVS 전류가 다르다. capacitive 해는 FHA상으로도 ZVS를 잃어 제외한다."))
    _gain_plot(res, tank, v, per, "p_gain", f"n = {n:g}{' (환산 대칭)' if v['symmetric'] else ' (실제 L₂·C₂ 유지 = 다른 회로)'}: 고전압 corner에 두 해", "세 corner의 모든 해와 inductive 여부를 찾았다 (교재 표 재현).", "해가 두 개라는 것은 제어 branch 선택이 필요하다는 뜻이다 — 선택 근거는 스위칭·동특성(EX05).")
    _check_textbook_form(res, tank, per, v)
    for (name, Vbat, Vlink), cf in per:
        for r in cf["inductive_roots"]:
            p = r["point"]
            P_fha = 0.5 * (bridge_fundamental(Vlink, "FB") ** 2) * (1 / p.Zin).real
            res.add_check(Check(f"FHA 전력 보존 {name} {r['f'] / 1e3:.3f} kHz", "PASS" if abs(P_fha - v["P"]) < 1e-6 * v["P"] else "FAIL", P_fha, "W", v["P"], path="입력 기본파 V·I·cosθ = R_ac 소비전력 (무손실 tank)", independent=True))
    if all_ok:
        res.verdict("CANDIDATE_FHA_ONLY", "세 corner 모두 FHA inductive 해가 있다. 이것은 후보가 될 이유이지 채택 근거가 아니다 — 스위칭·ZVS·reverse·기동·공차는 따로 판정한다.")
    else:
        res.verdict("FAIL_CONSTRAINT", "일부 corner에 inductive 해가 없다")
    if len(ind) == 2:
        res.verdict("UNRESOLVED_RANKING", "고전압 corner의 두 해 중 무엇을 쓸지는 FHA만으로 정하지 않는다 (기울기 부호·RMS·ZVS·동특성 비교 필요)")
    res.assumptions += ["FHA (기본파)", "n은 연속 설계변수 — 13:14 ≈ 0.9286 같은 정수 권선은 회로·공차·창 면적·손실을 다시 계산", "합성 학습 사양"]
    res.not_valid_for += ["최종 transformer 선정", "스위칭 동작점 (→ ‘시간영역 검증’ 실험)", "공차 corner"]
    res.interpretation = (
        "n을 0.93으로 줄이면 필요 gain이 1.0066으로 내려와 FHA 해가 생긴다. 2차 L·C를 L₁/n², C₁n²로 바꿔야 1차 환산 회로가 대칭으로 남는다. n만 바꾸고 실제 L·C를 그대로 두면 다른 회로다. "
        "고전압 corner에는 gain peak 양쪽에 두 해가 있어 기울기 부호가 반대다 — 같은 출력이라도 제어 방향과 전류가 다르다."
    )
    res.circuit = {"diagram": resonant_circuit("CLLC", v["Vlink_hi"], f"{n:g}", tank, "FB", out="battery", vo_label=f"{v['Vbat_hi']:g} V").to_json(), "intervals": [], "plot_group": ""}
    return res


# ======================================================================================
# 3. Branch selection: roots vs battery voltage, fold, nearest-root policy
# ======================================================================================


def run_branch(v: dict) -> Result:
    res = Result("FL10", "branch_selection", "A (FHA)")
    tank, _, _ = tank_of(v)
    n, Vlink = v["n"], v["Vlink_hi"]
    Vs = np.round(np.linspace(v["Vbat_from"], v["Vbat_to"], int(v["points"])), 6)
    lo_f, hi_f, lo_s, hi_s = [], [], [], []
    fold = None
    for Vb in Vs:
        cf = corner_fha(tank, n, float(Vb), Vlink, v["P"], v["f_min"], v["f_max"])
        ind = cf["inductive_roots"]
        # the two branches around the gain peak (roots below / above the max-gain frequency)
        below = [r for r in ind if r["f"] < cf["fmax"]]
        above = [r for r in ind if r["f"] >= cf["fmax"]]
        lo_f.append(below[-1]["f"] if below else None)
        hi_f.append(above[0]["f"] if above else None)
        lo_s.append(below[-1]["slope_per_Hz"] * 1e3 if below else None)
        hi_s.append(above[0]["slope_per_Hz"] * 1e3 if above else None)
        if fold is None and not ind:
            fold = float(Vb)
    res.add_series("f_lo", "gain peak 아래 branch", "Hz", Vs.tolist(), lo_f, style="line")
    res.add_series("f_hi", "gain peak 위 branch", "Hz", Vs.tolist(), hi_f, style="line")
    res.add_series("s_lo", "아래 branch 기울기", "", Vs.tolist(), lo_s)
    res.add_series("s_hi", "위 branch 기울기", "", Vs.tolist(), hi_s)
    # fold: required gain equals the peak gain -> solve on V_bat
    cf_hi = corner_fha(tank, n, v["Vbat_hi"], Vlink, v["P"], v["f_min"], v["f_max"])
    Vfold = _fold_voltage(tank, n, Vlink, v)
    res.add_metric("Vfold", "두 해가 만나는 배터리 전압 (fold)", Vfold, "V", basis=f"link {Vlink:g} V, {v['P'] / 1e3:g} kW에서 FHA gain peak = 필요 gain")
    res.add_metric("f_peak", f"{v['Vbat_hi']:g} V에서 gain peak 주파수", cf_hi["fmax"], "Hz")
    res.add_plot("p_roots", "배터리 전압에 따른 두 해: fold에서 만나고 그 위로는 해가 없다", ["f_lo", "f_hi"], x_label="V_bat", x_unit="V", y_label="f_s", y_unit="Hz", kind="xy", level="A",
                 vlines=[{"x": Vfold, "label": "fold"}] if Vfold else [],
                 proved="각 배터리 전압에서 필요 gain을 만족하는 inductive 해를 모두 찾았다.", not_yet="FHA 곡선의 성질 — 스위칭 모델에서 이 구조가 유지되는지는 ‘시간영역 검증’과 EX05에서 본다.")
    res.add_plot("p_slope", "gain 기울기: 부호가 반대이고 fold에서 0이 된다 (제어 이득 소멸)", ["s_lo", "s_hi"], x_label="V_bat", x_unit="V", y_label="d|H|/df (1/kHz)", y_unit="", kind="xy", level="A", hlines=[{"y": 0.0, "label": "0"}], proved="배터리 전압마다 두 FHA 해의 gain 기울기를 ±10 Hz 중앙차분으로 계산했다. 두 branch의 부호가 반대이고 fold(두 해가 만나는 V_bat)에서 0이 된다.", not_yet="정상 기울기는 G_vf(0)일 뿐 — 위상·공진극·지연은 EX05.")
    # two command policies over a battery ramp up past the fold and back down
    ramp = [float(x) for x in np.linspace(v["Vbat_from"], v["Vbat_to"], 14)] + [float(x) for x in np.linspace(v["Vbat_to"], v["Vbat_from"], 14)[1:]]
    first_cmd, lock_cmd, lock_last = [], [], None
    for Vb in ramp:
        cf = corner_fha(tank, n, Vb, Vlink, v["P"], v["f_min"], v["f_max"])
        ind = sorted(r["f"] for r in cf["inductive_roots"])
        first_cmd.append(ind[0] if ind else None)  # a solver that scans up from f_min and takes the first root
        upper = [f for f in ind if f >= cf["fmax"]]
        if upper:
            lock_last = upper[0]
        lock_cmd.append(upper[0] if upper else (cf["fmax"] if math.isfinite(cf["fmax"]) else lock_last))  # stay on the upper branch; at the fold hold the peak
    steps = list(range(len(ramp)))
    res.add_series("cmd_first", "첫 해 정책 (f_min부터 스캔해 처음 찾은 해)", "Hz", steps, first_cmd, style="points")
    res.add_series("cmd_lock", "branch 고정 정책 (위 branch, fold에서 peak로 제한)", "Hz", steps, lock_cmd)
    res.add_series("ramp_v", "배터리 전압 램프", "V", steps, ramp)
    res.add_plot("p_ramp", "명령 정책 비교: ‘처음 찾은 해’는 아래 해가 범위에 들어오는 순간 branch를 바꾼다", ["cmd_first", "cmd_lock"], x_label="램프 단계 (전압 ↑ 후 ↓)", x_unit="", y_label="f 명령", y_unit="Hz", kind="xy", level="A",
                 proved="같은 필요 gain에 대해 해를 고르는 규칙만 바꿔도 주파수 명령이 불연속이 되고 gain 기울기 부호가 뒤집힌다.",
                 not_yet="실제 제어기는 동특성·포화·hysteresis를 포함해야 한다 (EX05). fold 너머(해 없음)는 두 정책 모두 전력 제한이 필요하다.")
    jumps = [(abs(b_ - a_), k) for k, (a_, b_) in enumerate(zip(first_cmd[:-1], first_cmd[1:])) if a_ is not None and b_ is not None]
    big = max(jumps) if jumps else (0.0, None)
    res.add_metric("max_jump", "‘첫 해’ 정책의 최대 한 단계 명령 변화", big[0], "Hz", basis=(f"단계 {big[1]}→{big[1] + 1}, V_bat {ramp[big[1]]:.1f}→{ramp[big[1] + 1]:.1f} V" if big[1] is not None else ""))
    lock_j = [abs(b_ - a_) for a_, b_ in zip(lock_cmd[:-1], lock_cmd[1:]) if a_ is not None and b_ is not None]
    res.add_metric("max_jump_lock", "branch 고정 정책의 최대 한 단계 변화", max(lock_j) if lock_j else 0.0, "Hz")
    none_steps = sum(1 for x in first_cmd if x is None)
    res.add_metric("no_solution_steps", "해가 없는 램프 단계 수 (fold 너머)", none_steps, "")
    res.verdict("UNRESOLVED_RANKING", "branch 선택은 FHA만으로 결정되지 않는다: 기울기 부호·RMS·ZVS·동특성·운전 범위 제한·hysteresis 전략이 필요하다")
    res.verdict("INFO", "‘가장 높은 주파수 선택’은 시작 heuristic일 뿐 자동 최적해가 아니다")
    res.assumptions += ["FHA 해를 V_bat별로 독립 계산", f"link {Vlink:g} V, {v['P'] / 1e3:g} kW 고정"]
    res.not_valid_for += ["폐루프 안정성", "스위칭 동작점"]
    res.interpretation = "필요 gain이 올라가면 두 해가 gain peak 쪽으로 모이다가 fold에서 만나고 사라진다. fold 근처에서는 기울기가 0이라 주파수로 출력을 거의 못 움직인다. 해를 매 순간 새로 고르는 제어기는 해 집합이 바뀌는 순간 다른 branch로 뛰어 명령이 불연속이 되고 기울기 부호가 뒤집힌다."
    return res


def _fold_voltage(tank: Tank, n: float, Vlink: float, v: dict) -> float | None:
    from scipy.optimize import brentq

    def h(Vb):
        cf = corner_fha(tank, n, Vb, Vlink, v["P"], v["f_min"], v["f_max"])
        return cf["gmax"] - cf["g_req"]

    a, b = v["Vbat_from"], v["Vbat_to"]
    try:
        if h(a) * h(b) > 0:
            return None
        return float(brentq(h, a, b, xtol=1e-6))
    except ValueError:
        return None


# ======================================================================================
# 4. Reverse power flow: derive again, do not invert
# ======================================================================================


def run_reverse(v: dict) -> Result:
    res = Result("FL10", "reverse", "A (FHA)")
    tank, _, _ = tank_of(v)
    rtank = tank.swapped()
    n = v["n"]
    rows = []
    worst = "CANDIDATE_FHA_ONLY"
    per = []
    for name, Vbat, Vlink in corners(v):
        Rac_r = rac(1.0, Vlink, v["P"])  # load now on the primary (link) side: referred ratio 1
        g_req = Vlink / (n * Vbat)
        roots = fha_roots(rtank, Rac_r, g_req, v["f_min"], v["f_max"])
        gmax, fmax = max_inductive_gain(rtank, Rac_r, v["f_min"], v["f_max"])
        ind = [r for r in roots if r["inductive"]]
        fwd = corner_fha(tank, n, Vbat, Vlink, v["P"], v["f_min"], v["f_max"])
        f_fwd = fwd["inductive_roots"][-1]["f"] if fwd["inductive_roots"] else None
        recip = 1 / fwd["g_req"]
        true_rev = fha(rtank, f_fwd, Rac_r).gain if f_fwd else float("nan")
        per.append(((name, Vbat, Vlink), {"Rac": Rac_r, "g_req": g_req, "gmax": gmax, "fmax": fmax, "roots": roots, "inductive_roots": ind, "ok": bool(ind)}))
        rows.append([f"{name} {Vbat:g}→{Vlink:g} V", f"{g_req:.4f}", f"{gmax:.4f} @ {fmax / 1e3:.1f} kHz", ", ".join(f"{r['f'] / 1e3:.3f} kHz" for r in ind) or "없음", f"{recip:.4f}" if f_fwd else "-", f"{true_rev:.4f}" if f_fwd else "-", "해 있음" if ind else "FAIL"])
        res.add_metric(f"req_{name}", f"{name} reverse 필요 gain V_link/(n V_bat)", g_req, "")
        res.add_metric(f"gmax_{name}", f"{name} reverse inductive 최대 gain", gmax, "")
        if not ind:
            worst = "FAIL_CONSTRAINT"
    res.tables.append(Table("t_rev", f"reverse (배터리 → link) FHA — 탱크를 뒤집어 다시 유도 (n = {n:g})", ["방향·corner", "필요 gain", "최대 inductive gain", "해", "‘역수’ 추정 1/g_fwd", "실제 reverse gain @ forward 해", "판정"], rows,
                            note="reverse 입력은 배터리측 bridge(n·V_bat 환산), 부하는 link측 R_ac = 8V_link²/(π²P). forward 해 주파수에서 reverse gain은 1/g_fwd가 아니다."))
    _gain_plot(res, rtank, v, per, "p_gain", "reverse FHA gain과 필요 gain: 저전압 corner는 해가 없다", "tank를 뒤집고 부하 위치를 바꿔 reverse gain을 다시 계산했다.", "reverse 스위칭·precharge·sensing은 아직 (EX05).")
    dev = 0.0
    for _, cf in per:
        for f in np.linspace(v["f_min"], v["f_max"], 41):
            a = fha(rtank, float(f), cf["Rac"]).H
            b = ref.cllc_H(float(f), rtank.L1, rtank.C1, rtank.Lm, rtank.L2, rtank.C2, cf["Rac"])
            dev = max(dev, abs(a - b))
    res.add_check(Check("reverse: 교재 식(탱크 교환) vs 절점해석", "PASS" if dev < 1e-12 else "FAIL", dev, "", 1e-12, path="같은 식에 뒤집힌 회로 대입 vs 어드미턴스 해석", independent=True))
    res.verdict(worst, "reverse는 forward의 역수가 아니다: " + ("일부 corner에서 필요 gain이 inductive 최대 gain을 넘는다 (reverse FAIL 보존)" if worst == "FAIL_CONSTRAINT" else "모든 corner에 reverse FHA 해가 있다"))
    res.verdict("NOT_EVALUABLE", "reverse 스위칭 동작·precharge·sensing·제어 부호는 FHA로 판정하지 않는다 (EX05)")
    res.assumptions += ["reverse: 배터리측 bridge 구동, link측 다이오드(또는 body diode) 정류", "같은 T-모델에서 L₁↔L₂′, C₁↔C₂′ 교환", "FHA"]
    res.not_valid_for += ["reverse 스위칭 검증", "SR 동작"]
    res.interpretation = "reverse에서는 입력이 배터리측, 부하가 link측으로 바뀌어 R_ac와 필요 gain이 모두 달라진다. 대칭 tank라도 필요 gain V_link/(nV_bat)가 1.16까지 올라가는 저전압 corner에서는 해가 없다. forward gain의 역수로 추정하면 이 실패를 놓친다."
    res.circuit = {"diagram": resonant_circuit("CLLC", n * v["Vbat_lo"], f"{n:g}", rtank, "FB", out="battery", reverse=True, vo_label=f"{v['Vlink_lo']:g} V").to_json(), "intervals": [], "plot_group": ""}
    return res


# ======================================================================================
# 5. Time domain: the FHA roots in the exact switched model with a stiff battery
# ======================================================================================


class _Pt:
    """A converged switching orbit at one frequency (for continuation along a branch)."""

    def __init__(self, sysc, x, traj, M, P):
        self.sys, self.x, self.traj, self.M, self.P = sysc, x, traj, M, P


def _td_cold(tank: Tank, Vlink: float, f: float, n: float, Vbat: float, P: float):
    """Physical cycles from the FHA guess, then full Newton shooting.  Returns _Pt or None."""
    sysc = ResonantSystem(tank, Vlink, f, n, "FB", "stiff", Vo=Vbat, key="fl10")
    per = periodic(sysc, phasor_guess(sysc, rac(n, Vbat, P)))
    if not (per.converged and per.traj is not None and per.residual < 1e-8):
        return None
    return _Pt(sysc, per.x0, per.traj, per.monodromy, per.traj.energy(0.0, sysc.T, "p_rect") / sysc.T)


def _td_warm(tank: Tank, Vlink: float, f: float, n: float, Vbat: float, prev: "_Pt"):
    sysc = ResonantSystem(tank, Vlink, f, n, "FB", "stiff", Vo=Vbat, key="fl10")
    x, tr, res, ok, M = periodic_warm(sysc, prev.x, prev.M)
    if not ok:
        return None
    return _Pt(sysc, x, tr, M, tr.energy(0.0, sysc.T, "p_rect") / sysc.T)


def _td_point(tank: Tank, Vlink: float, f: float, n: float, Vbat: float, P: float, guess=None):
    """Full periodic solution with summary (used for reported operating points)."""
    sysc = ResonantSystem(tank, Vlink, f, n, "FB", "stiff", Vo=Vbat, key="fl10")
    per = periodic(sysc, phasor_guess(sysc, rac(n, Vbat, P)) if guess is None else guess, pre_cycles=12 if guess is None else 0)
    ok = per.converged and per.traj is not None and per.residual < 1e-8
    s = summarize(sysc, per.traj, sysc.T) if ok else None
    return sysc, per, s, ok


def _td_search(tank, Vlink, n, Vbat, P, f0, direction, f_lo, f_hi, max_steps=60):
    """Walk from f0 along ``direction`` (+1/-1) by continuation until the switching power crosses P, then
    bisect to 1 Hz.  Each point is warm-started (chord Newton) from the nearest converged orbit; a failed
    step is retried with a quarter of the step (the power can fall by kW per Hz close to resonance).
    Returns (f or None, reason, evaluations)."""
    start = _td_cold(tank, Vlink, f0, n, Vbat, P)
    if start is None:
        return None, "시작점 주기해 실패", []
    ev = [(f0, start.P)]

    def advance(pt: _Pt, f_to: float):
        f, cur = pt.sys.fs, pt
        h = f_to - f
        while (f_to - f) * math.copysign(1.0, h) > 1e-9:
            f_try = f + h
            if (f_try - f_to) * math.copysign(1.0, h) > 0:
                f_try = f_to
            nxt = _td_warm(tank, Vlink, f_try, n, Vbat, cur)
            if nxt is not None:
                f, cur = f_try, nxt
                h *= 1.5
            else:
                h /= 4.0
                if abs(h) < 0.25:
                    return None
        return cur

    prev = start
    step = 250.0
    for _ in range(max_steps):
        f = min(max(prev.sys.fs + direction * step, f_lo), f_hi)
        cur = advance(prev, f)
        if cur is None:
            return None, f"{f / 1e3:.3f} kHz 부근에서 주기해를 이어가지 못함 (공진점 근처 — 해 없음/발산)", ev
        ev.append((f, cur.P))
        if (prev.P - P) * (cur.P - P) <= 0:
            a, b = prev, cur
            while abs(b.sys.fs - a.sys.fs) > 1.0:
                m = advance(a, 0.5 * (a.sys.fs + b.sys.fs))
                if m is None:
                    return None, f"{0.5 * (a.sys.fs + b.sys.fs) / 1e3:.4f} kHz에서 주기해 실패", ev
                if (a.P - P) * (m.P - P) <= 0:
                    b = m
                else:
                    a = m
            return 0.5 * (a.sys.fs + b.sys.fs), "", ev
        if f in (f_lo, f_hi):
            return None, f"범위 끝 {f / 1e3:.0f} kHz까지 {P / 1e3:g} kW에 도달하지 않음 (P = {cur.P / 1e3:.2f} kW)", ev
        prev = cur
        step *= 1.5
    return None, "탐색 단계 초과", ev


def run_time_domain(v: dict) -> Result:
    res = Result("FL10", "time_domain", "C (이상 스위칭, 강한 배터리) + A")
    t0, _, _ = tank_of(v)
    tank = Tank(t0.L1, t0.C1, t0.Lm, t0.L2, t0.C2, v["R"], v["R"])
    n, P = v["n"], v["P"]
    rows = []
    op_rows = []
    verdicts = []
    for name, Vbat, Vlink in corners(v):
        cf = corner_fha(tank, n, Vbat, Vlink, P, v["f_min"], v["f_max"])
        for r in cf["inductive_roots"]:
            branch = "peak 아래 (+기울기)" if r["slope_per_Hz"] > 0 else "peak 위 (−기울기)"
            sysc, per, s, ok = _td_point(tank, Vlink, r["f"], n, Vbat, P)
            if not ok:
                rows.append([f"{name} {Vbat:g}/{Vlink:g}", f"{r['f'] / 1e3:.3f} kHz", branch, "주기해 실패", "-", "-", "-"])
                verdicts.append(("SOLVER_FAILED", f"{name} {r['f'] / 1e3:.3f} kHz 주기해 실패"))
                continue
            dP = s["P_rect"] / P - 1
            rows.append([f"{name} {Vbat:g}/{Vlink:g}", f"{r['f'] / 1e3:.3f} kHz", branch, f"{s['P_rect'] / 1e3:.2f} kW ({dP * 100:+.1f} %)", f"{s['I1_rms']:.2f} A (FHA {r['I1_rms']:.2f})", f"{s['off_frac'] * 100:.1f} %", f"{per.rho:.4f}"])
            if name == "고전압":
                res.add_metric(f"P_td_{'lo' if r['slope_per_Hz'] > 0 else 'hi'}", f"고전압 FHA 해 {r['f'] / 1e3:.3f} kHz에서 스위칭 전력", s["P_rect"], "W", basis=f"FHA는 {P / 1e3:g} kW")
            # direction: move away from the peak if the switching power is too high, towards it if too low
            away = 1 if r["slope_per_Hz"] < 0 else -1
            direction = away if s["P_rect"] > P else -away
            f_sw, why, ev = _td_search(tank, Vlink, n, Vbat, P, r["f"], direction, v["f_min"], v["f_max"])
            if f_sw is None:
                op_rows.append([f"{name} {Vbat:g}/{Vlink:g}", branch, f"{r['f'] / 1e3:.3f} kHz", "범위 내 없음", why, "-", "-"])
                verdicts.append(("FAIL_CONSTRAINT", f"{name} corner {branch}: {why} — FHA 해에 대응하는 스위칭 동작점이 {v['f_min'] / 1e3:g}–{v['f_max'] / 1e3:g} kHz에 없다"))
                if name == "고전압":
                    res.add_metric(f"f_sw_{'lo' if r['slope_per_Hz'] > 0 else 'hi'}", f"고전압 {branch} 스위칭 11 kW 주파수", "범위 내 없음", "", basis=why)
                continue
            sysc2, per2, s2, ok2 = _td_point(tank, Vlink, f_sw, n, Vbat, P)
            ec = edge_currents(sysc2, per2.traj, sysc2.T)
            zs = zvs_screen(ec["i_rise"], Vlink, v["td"], *COSS)
            # local sensitivity dP/df by central difference on the switching model (warm start, small step:
            # close to resonance the lossless power can change by kW per Hz)
            sens = float("nan")
            base = _Pt(sysc2, per2.x0, per2.traj, per2.monodromy, s2["P_rect"])
            for d in (2.0, 0.5):
                pp = _td_warm(tank, Vlink, f_sw + d, n, Vbat, base)
                pm = _td_warm(tank, Vlink, f_sw - d, n, Vbat, base)
                if pp is not None and pm is not None:
                    sens = (pp.P - pm.P) / (2 * d)
                    break
            op_rows.append([f"{name} {Vbat:g}/{Vlink:g}", branch, f"{r['f'] / 1e3:.3f} kHz", f"{f_sw / 1e3:.3f} kHz", f"{s2['I1_rms']:.2f} A", f"{sens * 1e3 / 1e3:+.1f} kW/kHz", ZVS_KO[zs["status"]]])
            if name == "고전압":
                tag = "lo" if r["slope_per_Hz"] > 0 else "hi"
                res.add_metric(f"f_sw_{tag}", f"고전압 {branch} 스위칭 11 kW 주파수", f_sw, "Hz", basis=f"FHA {r['f'] / 1e3:.3f} kHz")
                res.add_metric(f"sens_{tag}", f"고전압 {branch} dP/df (스위칭)", sens * 1e3, "W/kHz", basis="±2 Hz 중앙차분 (연속 추적)")
                if abs(sens) * 1e3 > 20 * P / 100:  # more than 20 % of P per kHz
                    verdicts.append(("MARGINAL", f"고전압 {branch} 동작점 {f_sw / 1e3:.3f} kHz: dP/df = {sens:+.0f} W/Hz — 10 Hz 오차가 {abs(sens) * 10 / P * 100:.1f} % 전력 오차"))
    res.tables.append(Table("t_td", "FHA 해 주파수에서의 스위칭 결과 (강한 link·배터리, 이상 소자)", ["corner", "FHA 해", "branch", "스위칭 전력", "I₁,rms", "정류 off", "Floquet |λ|max"], rows,
                            note="FHA는 이 주파수에서 11 kW를 예측한다. 양쪽 포트가 강한 전압원이면 전력은 작은 위상 차로 정해져 FHA의 기본파 위상 가정 오차가 큰 전력 오차가 된다."))
    res.tables.append(Table("t_op", "스위칭 모델에서 11 kW를 주는 주파수 (같은 branch를 따라 탐색)", ["corner", "branch", "FHA 해", "스위칭 동작점", "I₁,rms", "dP/df", "ZVS screen"], op_rows,
                            note=f"탐색: FHA 해에서 출발해 전력 오차를 줄이는 방향으로 걷고 교차 구간을 1 Hz까지 이분. ZVS는 합성 C_oss·t_d = {v['td'] * 1e9:g} ns screen (SCREEN_ONLY)."))
    # power curve at the high corner: FHA vs switching
    Vbat, Vlink = v["Vbat_hi"], v["Vlink_hi"]
    g_req = n * Vbat / Vlink
    fs = np.linspace(v["curve_f_lo"], v["curve_f_hi"], int(v["curve_points"]))
    p_td, p_fha, x = [], [], None
    for f in fs:
        _, per, s, ok = _td_point(tank, Vlink, float(f), n, Vbat, P, guess=x)
        if not ok and x is not None:
            _, per, s, ok = _td_point(tank, Vlink, float(f), n, Vbat, P)
        p_td.append(s["P_rect"] if ok else None)
        x = per.x0 if ok else None
        p_fha.append(_fha_power(tank, float(f), n, Vbat, g_req))
    res.add_series("P_td", "스위칭 모델 전력", "W", fs.tolist(), p_td, style="points")
    res.add_series("P_fha", "FHA 전력 (|H(R_ac)| = 필요 gain)", "W", fs.tolist(), p_fha, dash=True)
    res.add_plot("p_power", f"고전압 corner {Vbat:g}/{Vlink:g} V: 같은 주파수에서의 전력 — FHA vs 스위칭", ["P_td", "P_fha"], x_label="f_s", x_unit="Hz", y_label="P", y_unit="W", kind="xy", level="C + A",
                 hlines=[{"y": P, "label": f"{P / 1e3:g} kW"}], vlines=[{"x": tank.fr1, "label": "f_r"}],
                 proved="양쪽 포트가 강한 전압원일 때 각 주파수의 주기해 전력을 FHA와 비교했다.", not_yet="dead time·C_oss·권선 저항·배터리 내부저항이 없는 이상 모델. f_r 바로 근처는 무손실 이상 모델에 주기해가 없다.")
    # detailed waveforms at the high-corner upper-branch switching operating point (if found)
    f_show = next((m.value for m in res.metrics if m.key == "f_sw_hi" and isinstance(m.value, float)), None)
    if f_show:
        sysc, per, s, ok = _td_point(tank, Vlink, f_show, n, Vbat, P)
        T = sysc.T
        tr = simulate(sysc, per.q0, per.x0, 0.0, 2 * T)
        series_from_traj(res, tr, {"i1": ("i₁ (1차)", "A"), "i2": ("i₂′ (정류, 1차 환산)", "A"), "im": ("i_m", "A"), "v1": ("v₁", "V"), "v2": ("v₂′ (정류기 입력)", "V"), "vC1": ("v_C1", "V"), "vC2": ("v_C2′", "V")}, per_segment=40)
        bands = bands_from(tr, 0.0, 2 * T)
        res.add_plot("p_i", f"고전압 위 branch 스위칭 동작점 {f_show / 1e3:.3f} kHz의 전류", ["i1", "i2", "im"], y_label="전류", y_unit="A", bands=bands, group="cllc", level="C", hlines=[{"y": 0.0, "label": "0"}], proved="강한 배터리에서 11 kW를 주는 스위칭 주기해", not_yet="이상 스위치·이상 다이오드의 주기해이며 ZVS는 edge 전류 부호와 전하 screen으로만 본다(SCREEN_ONLY). 이 동작점 근처는 dP/df가 매우 커서(지표 sens_hi) 실제 주파수 분해능과 손실이 동작점을 바꾼다.")
        res.add_plot("p_v", "전압과 공진 커패시터", ["v1", "v2", "vC1", "vC2"], y_label="전압", y_unit="V", bands=bands, group="cllc", level="C", proved="같은 주기해에서 bridge 전압, 정류기 입력 전압, 두 공진 C의 전압을 함께 보인다. C₁·C₂′ 전압 peak는 이 정확 해에서 읽은 값이다.", not_yet="C₁·C₂′ 전압 peak는 소자 정격 확인 대상")
        led = energy_ledger(tr, sysc, 0.0, T, ["p_in"], ["p_rect"], ["p_R"], rated_power=P)
        res.add_check(ledger_check(led))
        xT = ivp_period(sysc, per.x0)
        dev = float(np.max(np.abs(xT - per.x0) / scales_for(sysc)))
        res.add_check(Check("독립 경로: 교재 E05 상태식 + DOP853 한 주기", "PASS" if dev < 1e-7 else "FAIL", dev, "rel", 1e-7, path="상태식을 직접 적어 solve_ivp 이벤트로 적분 (affine 행렬·행렬지수 미사용)", independent=True))
        res.add_metric("vC1_pk", "C₁ 전압 peak", s["vC1_pk"], "V")
        res.add_metric("vC2_pk", "C₂′ 전압 peak (1차 환산)", s["vC2_pk"], "V", basis=f"실제 2차 C₂ 전압 = v_C2′/n = {s['vC2_pk'] / n:.0f} V")
        res.circuit = {"diagram": resonant_circuit("CLLC", Vlink, f"{n:g}", tank, "FB", out="battery", vo_label=f"{Vbat:g} V").to_json(), "intervals": bands, "plot_group": "cllc"}
    # harmonic-superposition cross-check at the high-corner FHA roots (continuous conduction only)
    cf_hi = corner_fha(tank, n, Vbat, Vlink, P, v["f_min"], v["f_max"])
    for r in cf_hi["inductive_roots"]:
        sysc, per, s, ok = _td_point(tank, Vlink, r["f"], n, Vbat, P)
        if ok and s["off_frac"] == 0.0:
            ph = harmonic_power(tank, r["f"], Vlink, n * Vbat)
            if ph is not None:
                res.add_check(Check(f"독립 경로: 고조파 중첩 (정류 연속 도통) {r['f'] / 1e3:.3f} kHz", "PASS" if abs(ph / s["P_rect"] - 1) < 5e-3 else "FAIL", ph, "W", s["P_rect"], path="홀수 고조파 3999차까지 선형 회로 해 + 정류 전압 edge = i₂ 영점 조건 (주파수 영역) vs 시간영역 주기해", independent=True, detail=f"상대차 {ph / s['P_rect'] - 1:+.2e} (고조파 절단)"))
    res.verdict("CANDIDATE_FHA_ONLY", "FHA gain은 세 corner에서 통과 — 이 판정은 아래 스위칭 판정과 별개다")
    seen = set()
    for code, why in verdicts:
        if (code, why) not in seen:
            res.verdict(code, why)
            seen.add((code, why))
    res.verdict("SCREEN_ONLY", "ZVS는 edge 전류 부호·전하 screen (합성 C_oss)")
    res.verdict("NOT_EVALUABLE", "기동(빈 출력 C / 배터리 연결)과 reverse 스위칭은 이 실험에서 판정하지 않는다 (EX05)")
    res.assumptions += [f"이상 bridge·다이오드, 직렬 저항 R₁ = R₂′ = {v['R'] * 1e3:g} mΩ (그 외 손실 없음)", "link·배터리 모두 강한 전압원 (내부저항·출력 C 없음)", "주기해 = 물리 12주기 + Newton shooting (event 시각 민감도 포함)"]
    res.not_valid_for += ["실제 소자 ZVS·손실", "배터리 내부저항·케이블이 있는 실제 전력 민감도", "제어 안정성"]
    res.interpretation = (
        "양쪽이 강한 전압원이면 CLLC는 DAB처럼 두 사각파 사이의 작은 위상 차로 전력을 보낸다. f_r 근처에서 순 직렬 리액턴스가 작아 FHA의 위상 가정(정류 전압이 i₂ 기본파와 동상) 오차 몇 도가 수십 % 전력 오차가 된다. "
        "그래서 FHA 해 주파수를 그대로 쓰면 11 kW가 아니라 훨씬 큰 전력이 흐르고, 실제 11 kW 동작점은 위 branch에서는 f_r 바로 아래의 매우 가파른 곳, 아래 branch에서는 주파수 범위 밖에 있을 수 있다."
    )
    return res


def _fha_power(tank: Tank, f: float, n: float, Vbat: float, g_req: float) -> float | None:
    """Power at which the FHA gain equals the required gain at this frequency (largest such power)."""
    from scipy.optimize import brentq

    Rs = np.geomspace(1.0, 1e5, 300)
    vals = [fha(tank, f, R).gain - g_req for R in Rs]
    best = None
    for k in range(len(Rs) - 1):
        if vals[k] * vals[k + 1] < 0:
            R = brentq(lambda x: fha(tank, f, x).gain - g_req, Rs[k], Rs[k + 1], xtol=1e-12)
            p = 8 / PI**2 * n**2 * Vbat**2 / R
            best = p if best is None else max(best, p)
    return best


def harmonic_power(tank: Tank, f: float, V1: float, V2: float, H: int = 3999) -> float | None:
    """Independent frequency-domain path for continuous rectifier conduction: square v1 and v2 (odd harmonics),
    linear T-network per harmonic, v2's edge placed at i2's rising zero crossing; returns P2."""
    from scipy.optimize import brentq

    h = np.arange(1, H + 1, 2)
    w = 2 * PI * f * h
    Z1 = tank.R1 + 1j * w * tank.L1 + 1 / (1j * w * tank.C1)
    Z2 = tank.R2 + 1j * w * tank.L2 + 1 / (1j * w * tank.C2)
    Zm = 1j * w * tank.Lm
    A1 = 4 * V1 / (PI * h)
    A2 = 4 * V2 / (PI * h)

    def i2_at_edge(th):
        V2h = A2 * np.exp(-1j * h * th)
        Vm = (A1 / Z1 + V2h / Z2) / (1 / Z1 + 1 / Zm + 1 / Z2)
        I2 = (Vm - V2h) / Z2
        return float(np.sum((I2 * np.exp(1j * h * th)).imag)), V2h, I2

    ths = np.linspace(-PI, PI, 1441)
    g = [i2_at_edge(t)[0] for t in ths]
    for k in range(len(ths) - 1):
        if g[k] < 0 <= g[k + 1]:
            th = brentq(lambda t: i2_at_edge(t)[0], ths[k], ths[k + 1], xtol=1e-14)
            _, V2h, I2 = i2_at_edge(th)
            return float(0.5 * np.sum((V2h * np.conj(I2)).real))
    return None


# ======================================================================================
# Content
# ======================================================================================

_Q = [
    Question(
        "CLLC는 f_r(공칭 공진점)만 맞추면 되나?",
        "아니다. 공칭 공진점은 한 운전점일 뿐이다. 배터리·link 전압 창 전체에서 필요 gain을 먼저 확인해야 하고(seed는 920/850 V에서 1.016 < 1.082로 FAIL), 해가 여러 개면 제어 branch를 정해야 하며, 공차와 reverse는 따로 검증한다.",
        "Is it enough to match the nominal resonant point of a CLLC?",
        "No. The nominal resonant point is only one operating point. I would first check the required gain across the whole battery and link window; the seed fails at 920 over 850 V, 1.016 against 1.082. Multiple solutions need an explicit branch decision, and tolerance and reverse operation are verified separately.",
        ["gain window FAIL", "다중 해", "공차", "reverse"],
    ),
    Question(
        "gain curve 오른쪽 branch가 항상 정답인가?",
        "해당 등가모델의 기울기와 입력 임피던스를 먼저 보되, 그것이 스위칭 안정성과 ZVS를 대신하지 않는다. root별 공진 전류·SR·주파수 응답·손실·공차를 비교해야 하고, branch 이름만으로 고르지 않는다. 이 앱의 스위칭 모델처럼 FHA의 두 branch 구조 자체가 실제 회로에서 달라질 수도 있다.",
        "Is the right-hand branch of the gain curve always the answer?",
        "Check the slope and input impedance of the equivalent model first, but that does not replace switching stability and ZVS. Compare resonant current, SR behaviour, frequency response, loss and tolerance per root rather than choosing by the branch name. As the switching model here shows, the two-branch structure of FHA can itself change in the real circuit.",
        ["기울기·입력 임피던스", "root별 비교", "branch 이름으로 선택 금지"],
    ),
    Question(
        "reverse gain은 forward gain의 역수인가?",
        "아니다. reverse는 입력이 배터리측, 부하가 link측이라 R_ac와 필요 gain V_link/(nV_bat)가 다르고, 회로를 뒤집어 다시 유도해야 한다. 이 수정안은 저전압 corner(650→700 V)에서 필요 gain 1.158로 reverse FHA 해가 없다.",
        "Is the reverse gain the reciprocal of the forward gain?",
        "No. In reverse the source is on the battery side and the load on the link side, so R_ac and the required gain, V_link over n V_bat, both change and the network has to be derived again. For this candidate the low corner, 650 to 700 V, needs 1.158 and has no reverse FHA solution.",
        ["회로 재유도", "R_ac 위치", "reverse FAIL 사례"],
    ),
    Question(
        "FHA로 gain이 되는데 왜 양산 승인이 안 되나?",
        "정상 기본파 gain은 출력 가능성의 한 screen이다. 실제 정류 모드, 순환전류·열, 공차, 기동, 제어 안정성, switching boundary와 측정을 별도로 확인해야 한다. 강한 배터리 부하에서는 FHA 해 주파수에서 실제 전력이 수십 % 다를 수 있다.",
        "FHA shows enough gain. Why is the design not approved for production?",
        "The steady fundamental gain is only one screen of output capability. The real rectifier mode, circulating current and heat, tolerances, start-up, control stability and switching boundaries have to be checked separately, along with measurements. With a stiff battery load, the power at the FHA frequency can be tens of percent off.",
        ["screen일 뿐", "정류 모드·열·공차·기동·제어", "측정"],
    ),
]

_screen = [Param("td", "dead time t_d (screen)", "s", 150e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", group="screen")]

EXPERIMENTS = [
    Experiment(
        key="seed_fail",
        title="초기안 FAIL을 먼저 재현하고 저장한다",
        goal="양측 full bridge, n = 1, L_r1 = L_r2 = 40 µH, C_r1 = C_r2 = 28.144773 nF, L_m = 200 µH, 120–210 kHz, 11 kW에서 920/850 V corner의 필요 gain 1.082353이 inductive 최대 gain 1.016401보다 크다는 것을 재현하고 FAIL을 보존한다.",
        params=_tank_params(1.0) + _op_params(),
        presets=[
            Preset("seed", "교재 초기안 (n = 1)", {}, "FAIL이 정답", ("nominal", "reference", "failure")),
            Preset("fix", "n = 0.93 (환산 대칭)", {"n": 0.93}, "수정안", ("variant", "reference")),
        ],
        run=run_seed,
        model_level="A",
        suggested_change="권선비 n을 1 → 0.93으로 바꾼다 (환산 대칭 유지).",
        prediction=Prediction(
            "n을 0.93으로 줄이면 고전압 corner 필요 gain은?",
            ["1.082353 그대로", "약 1.0066으로 내려가 해가 생긴다", "올라가서 더 나빠진다", "모르겠다"],
            "약 1.0066으로 내려가 해가 생긴다",
            "필요 gain은 n·V_bat/V_link라 n에 비례한다: 0.93 × 920/850 = 1.006588. 환산 대칭을 유지하면 gain 곡선은 거의 같으므로 최대 gain(≈1.0117)보다 작아져 해가 생긴다.",
            ["g_req_hi", "gmax_hi"],
            handcalc=[{"key": "g_req_hi", "label": "필요 gain n·V_bat/V_link", "unit": ""}, {"key": "fr", "label": "f_r", "unit": "Hz"}],
        ),
        suggested={"n": 0.93},
        student="출력 전압이 높은 corner에서 tank가 만들 수 있는 gain보다 요구가 크면 어떤 주파수로도 11 kW를 못 만든다. 회로를 만들기 전에 식으로 알 수 있다.",
        expert="seed FAIL은 버그가 아니라 결과다. 필요 gain 창·inductive 영역·최대 gain을 corner별로 표로 남기고, 수정안은 무엇을 바꿨는지(n, 2차 L·C 환산)와 새로 생긴 위험(다중 해)을 같이 기록한다.",
        customer_ko="현재 초기안은 920 V 배터리·850 V link·11 kW corner에서 필요한 gain 1.08을 내지 못합니다(최대 1.016). 권선비나 link 전압 범위 조정이 필요하고, 변경안은 전 corner에서 다시 확인하겠습니다.",
        customer_en="The seed design cannot reach the gain of 1.08 needed at the 920 V battery, 850 V link, 11 kW corner; its maximum is 1.016. The turns ratio or the link voltage range has to change, and I would re-check any change at every corner.",
        questions=_Q[:1],
        circuit="cllc",
        textbook=[TB_13],
        reference_presets=["seed", "fix"],
        claim_limit="FHA gain 가능성. 스위칭·ZVS·기동·reverse 주장 없음.",
    ),
    Experiment(
        key="fix_n093",
        title="수정안 n = 0.93: 모든 해를 찾고 두 해의 차이를 표로",
        goal="n = 0.93에서 2차 L₂ = 46.248121 µH, C₂ = 24.342414 nF로 환산 대칭을 유지하고 세 corner의 해(164.390, 162.811, 136.099/147.061 kHz)와 두 해의 기울기(+0.001871/−0.001803 per kHz)·FHA RMS(14.390/14.765 A)를 재현한다. n만 바꾸고 L·C를 그대로 둔 회로와 비교한다.",
        params=_tank_params(0.93) + _op_params(),
        presets=[
            Preset("textbook", "교재 수정안 (환산 대칭)", {}, "", ("nominal", "reference")),
            Preset("kept_lc", "n만 0.93, 2차 L·C 그대로", {"symmetric": False}, "다른 회로", ("variant", "reference")),
            Preset("turns_13_14", "정수 권선 13:14", {"n": 13 / 14}, "0.9286 — 재계산 필요", ("variant",)),
        ],
        run=run_fix,
        model_level="A",
        suggested_change="환산 대칭을 끄고(2차 L·C = 40 µH·28.14 nF 유지) 해가 어떻게 바뀌는지 본다.",
        prediction=Prediction(
            "n = 0.93에서 2차 실제 L·C를 그대로 두면?",
            ["같은 회로라 해도 같다", "1차 환산 L₂′·C₂′가 바뀌어 다른 회로가 된다", "해가 무조건 없어진다", "모르겠다"],
            "1차 환산 L₂′·C₂′가 바뀌어 다른 회로가 된다",
            "L₂′ = n²L₂ = 34.6 µH, C₂′ = C₂/n² = 32.5 nF로 2차 공진이 1차와 어긋난다. gain 곡선과 해가 달라지므로 ‘n만 바꾼 시뮬레이션’을 수정안 결과로 쓰면 안 된다.",
            ["L2p", "C2p"],
        ),
        suggested={"symmetric": False},
        student="권선비를 바꾸면 2차 부품이 1차에서 보이는 크기도 바뀐다. 1차에서 본 회로를 대칭으로 유지하려면 2차 L은 키우고 C는 줄여야 한다.",
        expert="같은 gain을 주는 두 해는 입력 임피던스 위상·RMS·기울기가 다르다. 교재의 두 해는 FHA 결과이며 어느 쪽을 쓸지는 스위칭 동작·제어 부호·열·공차로 정한다. 13:14 같은 정수 권선은 연속값 0.93과 다른 회로다.",
        customer_ko="권선비 0.93과 2차 L·C 재조정으로 모든 corner에서 FHA 해가 생깁니다. 다만 고전압 corner에 136 kHz와 147 kHz 두 해가 있어 제어 branch를 정해야 하고, 실제 스위칭·열·공차 확인 전에는 후보 단계입니다.",
        customer_en="A turns ratio of 0.93 with re-scaled secondary L and C gives an FHA solution at every corner. The high corner has two solutions, 136 and 147 kHz, so a control branch has to be chosen, and the design stays a candidate until switching, thermal and tolerance checks are done.",
        questions=_Q[1:2],
        circuit="cllc",
        textbook=[TB_13],
        reference_presets=["textbook", "kept_lc"],
        claim_limit="FHA 해와 FHA 전류. 스위칭 동작점 아님.",
    ),
    Experiment(
        key="branch_selection",
        title="branch 선택: 두 해는 fold에서 만나고 기울기가 0이 된다",
        goal="link 850 V, 11 kW에서 배터리 전압을 880→935 V로 올리며 두 해의 주파수·기울기를 추적하고, fold(두 해가 만나는 전압)를 찾는다. 해를 고르는 규칙(처음 찾은 해 vs branch 고정)만 바꿔도 주파수 명령이 불연속이 되는 것을 보인다.",
        params=_tank_params(0.93)
        + _op_params()
        + [
            Param("Vbat_from", "배터리 sweep 시작", "V", 880.0, "V", vmin=1, vmax=2000, source="ASSUMED", group="sweep"),
            Param("Vbat_to", "배터리 sweep 끝", "V", 935.0, "V", vmin=1, vmax=2000, source="ASSUMED", group="sweep"),
            Param("points", "점 개수", "", 23, "", vmin=5, vmax=101, kind="int", source="ASSUMED", group="sweep"),
        ],
        presets=[Preset("textbook", "n = 0.93, 850 V link", {}, "", ("nominal", "reference"))],
        run=run_branch,
        model_level="A",
        suggested_change="sweep 끝을 935 → 920 V로 줄여 fold 전까지만 본다.",
        prediction=Prediction(
            "배터리 전압이 fold에 가까워지면 두 해의 gain 기울기는?",
            ["커진다", "0에 가까워진다", "부호가 같아진다", "모르겠다"],
            "0에 가까워진다",
            "두 해가 gain peak로 모이므로 기울기가 0으로 간다. 주파수를 많이 바꿔도 출력이 거의 안 변하고, peak를 넘으면 부호가 뒤집힌다.",
            ["Vfold"],
        ),
        suggested={"Vbat_to": 920.0},
        student="필요한 gain이 커질수록 쓸 수 있는 주파수가 gain 곡선의 꼭대기 쪽으로 몰리고, 꼭대기를 넘으면 해가 없다.",
        expert="branch selection은 운전 범위 제한·기울기 부호 확인·hysteresis/전환 전략의 설계 문제다. 매 step 해를 새로 풀어 ‘처음 찾은 해’나 ‘가장 가까운 해’를 쓰는 제어기는 해 집합이 바뀌는 순간(아래 해가 범위에 들어올 때, fold를 지날 때) 불연속이 된다. ‘가장 높은 주파수’는 시작 heuristic일 뿐이다.",
        customer_ko="고전압 corner 근처에서는 주파수를 바꿔도 출력이 거의 움직이지 않는 구간이 있고, 그 위로는 해가 없습니다. 운전 범위를 fold 아래로 제한하고 branch 전환을 명시적으로 설계하겠습니다.",
        customer_en="Near the high-voltage corner there is a region where changing the frequency barely moves the output, and above it there is no solution. I would limit the operating range below that fold and design the branch transition explicitly.",
        questions=_Q[1:2],
        circuit="cllc",
        textbook=[TB_13, TB_E05],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="FHA 해 구조. 폐루프·스위칭 주장 없음.",
    ),
    Experiment(
        key="reverse",
        title="reverse는 forward의 역수가 아니다: 회로를 뒤집어 다시 유도",
        goal="배터리 → link 방향에서 tank를 교환하고 부하를 link측 R_ac로 옮겨 reverse FHA gain을 다시 계산한다. 필요 gain V_link/(nV_bat)이 저전압 corner에서 1.158이 되어 해가 없음을 보이고, forward gain 역수 추정과 비교한다.",
        params=_tank_params(0.93) + _op_params(),
        presets=[
            Preset("textbook", "n = 0.93 수정안", {}, "저전압 corner reverse FAIL", ("nominal", "reference", "failure")),
            Preset("kept_lc", "n만 0.93, 2차 L·C 그대로", {"symmetric": False}, "비대칭 tank", ("variant",)),
        ],
        run=run_reverse,
        model_level="A",
        suggested_change="저전압 corner의 link 전압을 700 → 600 V로 낮춘다.",
        prediction=Prediction(
            "reverse 저전압 corner(650 V 배터리 → 700 V link)에 FHA 해가 있을까?",
            ["forward에 해가 있으니 있다", "필요 gain 1.158이 최대 gain보다 커서 없다", "역수 1/0.864 = 1.158이므로 있다", "모르겠다"],
            "필요 gain 1.158이 최대 gain보다 커서 없다",
            "reverse 필요 gain은 V_link/(nV_bat) = 700/(0.93·650) = 1.158이다. 대칭 tank의 최대 gain은 약 1.01이라 해가 없다. forward에 해가 있다는 사실은 reverse를 보장하지 않는다.",
            ["req_저전압", "gmax_저전압"],
        ),
        suggested={"Vlink_lo": 600.0},
        student="전력 방향을 바꾸면 입력과 부하의 위치가 바뀐다. 같은 tank라도 필요한 승압비가 달라져 한쪽 방향만 되는 corner가 생긴다.",
        expert="reverse는 실제 L/C 환산·bridge 여기·sensing/제어·precharge 조건이 달라진다. forward gain의 역수를 쓰지 않고 회로를 다시 유도한다. 저전압 corner의 reverse 실패는 V2G/V2L 요구와 link 전압 범위 협의 사항이다.",
        customer_ko="양방향 요구가 있다면 배터리 650 V에서 link 700 V로 되돌리는 방향은 현재 수정안으로 FHA상 해가 없습니다. reverse 운전 범위나 link 전압 목표를 먼저 합의하시죠.",
        customer_en="If bidirectional operation is required, returning power from a 650 V battery to a 700 V link has no FHA solution with the current candidate. Let's first agree on the reverse operating range or the link voltage target.",
        questions=_Q[2:3],
        circuit="cllc",
        textbook=[TB_13, TB_E05],
        reference_presets=["textbook"],
        claim_limit="reverse FHA 가능성. reverse 스위칭 검증 아님.",
    ),
    Experiment(
        key="time_domain",
        title="시간영역 검증: FHA 해 주파수에서 실제로 얼마의 전력이 흐르나",
        goal="n = 0.93 수정안을 강한 link·배터리 사이의 이상 스위칭 모델로 풀어 세 corner의 FHA 해 주파수에서 전력·RMS를 구하고, 같은 branch를 따라 스위칭 모델이 11 kW를 주는 주파수를 찾는다. FHA gain PASS와 스위칭·ZVS·기동 판정을 분리한다.",
        params=_tank_params(0.93)
        + _op_params()
        + _screen
        + [
            Param("R", "직렬 저항 R₁ = R₂′ (1차 환산, 결합 손실)", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", source_note="0 = 이상", group="비이상"),
            Param("curve_f_lo", "전력 곡선 시작", "Hz", 120e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="전력 곡선"),
            Param("curve_f_hi", "전력 곡선 끝", "Hz", 149e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="전력 곡선"),
            Param("curve_points", "전력 곡선 점 개수", "", 24, "", vmin=5, vmax=120, kind="int", source="ASSUMED", group="전력 곡선"),
        ],
        presets=[
            Preset("textbook", "n = 0.93, 강한 배터리, 무손실", {}, "", ("nominal", "reference")),
            Preset("lossy", "R₁ = R₂′ = 50 mΩ", {"R": 0.05}, "손실이 민감도를 줄인다", ("variant", "reference")),
        ],
        run=run_time_domain,
        model_level="C + A",
        suggested_change="(관찰 실험) 전력 곡선 범위를 120–149 → 140–149 kHz로 좁혀 f_r 근처를 확대한다.",
        prediction=Prediction(
            "고전압 corner의 FHA 해 147.061 kHz에서 스위칭 모델의 전력은?",
            ["11 kW 근처 (±5 %)", "11 kW보다 훨씬 크다", "11 kW보다 훨씬 작다", "모르겠다"],
            "11 kW보다 훨씬 크다",
            "양쪽이 강한 전압원이면 전력은 작은 위상 차로 정해진다. f_r 근처의 작은 순 리액턴스 때문에 FHA 위상 가정 오차가 커지고, 이 이상 모델에서는 약 22 kW(+100 %)가 흐른다. 11 kW 동작점은 약 148.0 kHz의 매우 가파른 곳에 있다.",
            ["P_td_hi", "f_sw_hi"],
        ),
        suggested={"curve_f_lo": 140e3},
        student="FHA는 사인파만 보고 계산한 근사다. 배터리처럼 전압이 딱 정해진 부하에서는 그 근사 오차가 전력에 크게 반영된다.",
        expert="FHA 해는 후보 주파수일 뿐이다. 스위칭 모델에서 11 kW 동작점의 위치·dP/df·ZVS 전류를 branch별로 다시 구하고, 주파수 범위 밖으로 나가는 branch는 FAIL로 남긴다. 실제 회로의 dead time·C_oss·저항·배터리 내부저항은 이 민감도를 바꾸므로 측정·비이상 모델로 이어서 확인한다.",
        customer_ko="FHA로 찾은 동작 주파수에서 실제 스위칭 모델은 11 kW가 아닌 훨씬 큰 전력을 보입니다. 고전압 corner에서 11 kW는 공진점 바로 아래의 매우 민감한 주파수에서만 나오므로, 주파수 제어 분해능·전류 제한·운전 범위를 함께 검토해야 합니다.",
        customer_en="At the frequencies found with FHA, the switching model delivers far more than 11 kW. At the high corner, 11 kW only occurs at a very sensitive frequency just below resonance, so frequency resolution, current limiting and the operating range have to be reviewed together.",
        questions=_Q[3:],
        circuit="cllc",
        textbook=[TB_13, TB_E05, TB_E12],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="이상 스위칭·강한 전압원 모델의 전력·RMS. 손실·ZVS 보증·기동 없음.",
    ),
]

LAB = Lab(
    id="FL10",
    title="CLLC — 설계 실패와 다중 해를 직접 방어하기",
    title_en="CLLC: defend the design failure and the multiple solutions",
    track="basic",
    order=10,
    path_note="14일 경로 10일차",
    textbook=[TB_13, TB_E05],
    prerequisites=["FL09"],
    summary="seed FAIL 재현·보존 → n = 0.93 환산 대칭의 모든 해 → branch·fold → reverse 재유도 → 스위칭 모델로 FHA 해 검증.",
    experiments=EXPERIMENTS,
    minimum_scope="CLLC network·full switching·branch; 초기 FAIL·수정안 2개 root·한계 (교재 19장)",
    claim_limits=["seed FAIL은 결과로 보존", "FHA gain PASS ≠ 스위칭·ZVS·reverse·기동 PASS", "이상 소자·강한 전압원 모델"],
    test_paths=["tests/test_fl10.py"],
)
