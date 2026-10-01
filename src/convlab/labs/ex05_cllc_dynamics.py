"""EX05 - CLLC: from the gain curve to real dynamics (textbook E05).

Primary-referred state model x = [i1, i2, vC1, vC2, vo] (textbook E05, plus the output capacitor and load
that a voltage loop needs).  The diode rectifier has three states; while all diodes are off, i2 = 0 is a
constraint and the rectifier input node floats (guards on the floating voltage reaching the clamps).
Independent paths: the textbook energy identity dW/dt = v1 i1 - v2 i2 - R1 i1^2 - R2 i2^2 evaluated against
the model's own derivatives in every mode; the textbook equations integrated with DOP853; the steady-state
slope from separate periodic solutions against the DC value of the sampled small-signal model; FM
injection on the nonlinear model against the linear prediction.
Floquet multipliers come from finite differences of the event-driven cycle map, so they include the
sensitivity of the rectifier event times; plant-only multipliers exclude the controller, and the
closed-loop analysis adds the controller states and the one-cycle computation delay explicitly.
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
    ResonantSystem,
    Tank,
    bands_from,
    cycle_map,
    edge_currents,
    fd_monodromy,
    fha,
    fha_roots,
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

TB_E05 = TextbookRef("expert-e05-cllc-gain-곡선에서-실제-동역학으로", "E05 · CLLC: gain 곡선에서 실제 동역학으로")
TB_13 = TextbookRef("cllc-설계-실패와-다중-해를-직접-방어하기-fl10", "13. CLLC [FL10]")
TB_E07 = TextbookRef("expert-e07-제어-단일-루프가-안정해도-시스템은-불안정할-수-있다", "E07 · 제어: 단일 루프가 안정해도 시스템은 불안정할 수 있다")
TB_E11 = TextbookRef("expert-e11-모델을-믿을-수-있는-범위-검증식별불확도", "E11 · 모델을 믿을 수 있는 범위")

C1_TB = 28.144773e-9
COSS = (0.5e-9, 40.0)


# --------------------------------------------------------------------------------------
# Parameters and the operating point
# --------------------------------------------------------------------------------------


def _params(extra=None) -> list[Param]:
    ps = [
        Param("L1", "1차 직렬 L₁", "H", 40e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", group="tank"),
        Param("C1", "1차 직렬 C₁", "F", C1_TB, "nF", vmin=1e-11, vmax=1e-4, source="TEXTBOOK", group="tank"),
        Param("Lm", "여자 L_m", "H", 200e-6, "µH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", group="tank"),
        Param("n", "권선비 n = N_p/N_s", "", 0.93, "", vmin=0.2, vmax=5.0, source="TEXTBOOK", source_note="FL10 수정안", group="tank"),
        Param("R", "직렬 저항 R₁ = R₂′ (1차 환산)", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="tank"),
        Param("Vlink", "link 전압 (구동측)", "V", 850.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="고전압 corner"),
        Param("Vref", "출력 전압 목표", "V", 920.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="920 V"),
        Param("P", "목표 출력 (R_L = V²/P)", "W", 11000.0, "kW", vmin=10, vmax=1e6, source="TEXTBOOK", source_note="11 kW"),
        Param("Co", "출력 C_o", "F", 100e-6, "µF", vmin=1e-7, vmax=1.0, source="ASSUMED", source_note="전압 루프에 필요한 출력 동특성 (합성값)", group="출력"),
        Param("f_min", "주파수 하한", "Hz", 120e3, "kHz", vmin=1e3, vmax=1e7, source="TEXTBOOK", group="범위"),
        Param("f_max", "주파수 상한", "Hz", 210e3, "kHz", vmin=1e3, vmax=1e7, source="TEXTBOOK", group="범위"),
    ]
    return ps + (extra or [])


def _tank(v) -> Tank:
    return Tank(v["L1"], v["C1"], v["Lm"], v["L1"], v["C1"], v["R"], v["R"])  # referred symmetry (FL10 candidate)


def _context_circuit(v, note: str = "") -> dict:
    """The circuit this experiment solves, drawn without waveform intervals (sweeps and maps have no single cycle)."""
    c = resonant_circuit("CLLC", v["Vlink"], f"{v['n']:g}", _tank(v), "FB")
    if note:
        c.notes.append(note)
    return {"diagram": c.to_json(), "intervals": [], "plot_group": ""}


def _RL(v) -> float:
    return v["Vref"] ** 2 / v["P"]


def _sys(v, f: float, RL: float | None = None, tank: Tank | None = None) -> ResonantSystem:
    return ResonantSystem(tank or _tank(v), v["Vlink"], f, v["n"], "FB", "rc", Co=v["Co"], Rout=RL or _RL(v), key="ex05")


def _cold(v, f, RL=None, tank=None):
    s = _sys(v, f, RL, tank)
    per = periodic(s, phasor_guess(s, rac(v["n"], v["Vref"], v["P"]), vo_guess=v["Vref"]))
    return s, per


class _Orbit:
    def __init__(self, s, x, traj, M):
        self.s, self.x, self.traj, self.M = s, x, traj, M

    @property
    def vo(self):
        return float(self.x[self.s.ix["vo"]])


def _orbit(v, f, prev: _Orbit | None = None, RL=None, tank=None) -> _Orbit | None:
    if prev is not None:
        s = _sys(v, f, RL, tank)
        x, tr, res, ok, M = periodic_warm(s, prev.x, prev.M)
        if ok:
            return _Orbit(s, x, tr, M)
    s, per = _cold(v, f, RL, tank)
    if not (per.converged and per.residual < 1e-8):
        return None
    return _Orbit(s, per.x0, per.traj, per.monodromy)


def operating_point(v, f_lo: float, f_hi: float, tank=None) -> tuple[float, _Orbit] | None:
    """Switching frequency in [f_lo, f_hi] where the cycle-average output equals Vref (upper, falling side)."""
    a = _orbit(v, f_lo, tank=tank)
    b = _orbit(v, f_hi, a, tank=tank) if a else None
    if a is None or b is None:
        return None
    va = _vo_avg(a) - v["Vref"]
    vb = _vo_avg(b) - v["Vref"]
    if va * vb > 0:
        return None
    cache = {}

    def g(f):
        o = _orbit(v, f, a, tank=tank)
        cache[f] = o
        return _vo_avg(o) - v["Vref"]

    f0 = brentq(g, f_lo, f_hi, xtol=1e-3)
    o = _orbit(v, f0, a, tank=tank)
    return f0, o


def _vo_avg(o: _Orbit) -> float:
    return o.traj.mean(0.0, o.s.T, "vo")


def linear_map(o: _Orbit):
    """Sampled small-signal model around the orbit: x_{k+1} = x* + A dx + B_f df (events included)."""
    s, x0 = o.s, o.x
    A = fd_monodromy(s, x0)
    T0 = s.T
    dT = 1e-6 * T0
    BT = (cycle_map(s, x0, T0 + dT)[0] - cycle_map(s, x0, T0 - dT)[0]) / (2 * dT)
    Bf = BT * (-1.0 / s.fs**2)
    C = np.zeros(s.ns)
    C[s.ix["vo"]] = 1.0
    return A, Bf, C


def gvf(A, Bf, C, f_mod: np.ndarray, T0: float) -> np.ndarray:
    out = []
    ns = A.shape[0]
    for fm in f_mod:
        z = np.exp(1j * 2 * PI * fm * T0)
        out.append(C @ np.linalg.solve(z * np.eye(ns) - A, Bf))
    return np.array(out)


# ======================================================================================
# 1. State equations and the energy identity
# ======================================================================================


def run_identity(v: dict) -> Result:
    res = Result("EX05", "state_identity", "C (상태식) + 항등식 검산")
    tank = _tank(v)
    n = v["n"]
    s = _sys(v, 150e3)
    rng = np.random.default_rng(int(v["seed"]))
    worst = 0.0
    worst_wrong = 0.0
    rows = []
    W_tb = lambda x: 0.5 * tank.L1 * x[0] ** 2 + 0.5 * tank.L2 * x[1] ** 2 + 0.5 * tank.Lm * (x[0] - x[1]) ** 2 + 0.5 * tank.C1 * x[2] ** 2 + 0.5 * tank.C2 * x[3] ** 2  # noqa: E731
    for b1 in (1, -1):
        for r in (1, 0, -1):
            md = s.mode((b1, r))
            devs, devs_wrong = [], []
            for _ in range(int(v["samples"])):
                x = np.array([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-900, 900), rng.uniform(-900, 900), rng.uniform(600, 1000)])
                if r == 0:
                    x[1] = 0.0
                dx = md.A @ x + md.b
                # textbook identity on the tank (the output capacitor is a separate port)
                grad = np.array([(tank.L1 + tank.Lm) * x[0] - tank.Lm * x[1], (tank.L2 + tank.Lm) * x[1] - tank.Lm * x[0], tank.C1 * x[2], tank.C2 * x[3]])
                dW = float(grad @ dx[:4])
                v1 = b1 * v["Vlink"]
                vm_ = float(s.vm_expr((b1, r)) @ np.append(x, 1.0))
                v2 = r * n * x[4] if r != 0 else vm_ - x[3]
                rhs = v1 * x[0] - v2 * x[1] - tank.R1 * x[0] ** 2 - tank.R2 * x[1] ** 2
                scale = abs(v1 * x[0]) + abs(v2 * x[1]) + 1.0
                devs.append(abs(dW - rhs) / scale)
                # a typical derivation slip: the secondary capacitor voltage entered with the wrong sign
                if r != 0:
                    c_bad = -x[3] + r * n * x[4] + tank.R2 * x[1]
                    a_ = v1 - x[2] - tank.R1 * x[0]
                    S = 1 / tank.L1 + 1 / tank.Lm + 1 / tank.L2
                    vm_bad = (a_ / tank.L1 + c_bad / tank.L2) / S
                    dx_bad = np.array([(a_ - vm_bad) / tank.L1, (vm_bad - c_bad) / tank.L2, x[0] / tank.C1, x[1] / tank.C2])
                    devs_wrong.append(abs(float(grad @ dx_bad) - rhs) / scale)
            worst = max(worst, max(devs))
            if devs_wrong:
                worst_wrong = max(worst_wrong, max(devs_wrong))
            rows.append([f"v₁ {'+' if b1 > 0 else '−'}, {({1: '정류 +', 0: '정류 off (i₂ = 0)', -1: '정류 −'})[r]}", f"{max(devs):.2e}", f"{max(devs_wrong):.2e}" if devs_wrong else "-"])
    res.tables.append(Table("t_id", "에너지 항등식 dW/dt = v₁i₁ − v₂i₂ − R₁i₁² − R₂i₂² (임의 상태, 모드별 최대 상대오차)", ["모드", "모델 (상대오차)", "C₂ 부호 실수 모델"], rows,
                            note="W = ½L₁i₁² + ½L₂i₂² + ½L_m(i₁−i₂)² + ½C₁v_C1² + ½C₂v_C2² (1차 환산). off 모드에서는 i₂ = 0이라 v₂ 항이 사라지고 노드가 떠 있다."))
    res.add_metric("id_max", "항등식 최대 상대오차 (모델)", worst, "", basis=f"모드 6개 × {int(v['samples'])}개 임의 상태")
    res.add_metric("id_wrong", "항등식 최대 상대오차 (C₂ 부호 실수)", worst_wrong, "", basis="항등식이 부호 실수를 잡는다")
    res.add_check(Check("에너지 항등식: 모델 도함수 vs 교재 포트 전력", "PASS" if worst < 1e-12 else "FAIL", worst, "rel", 1e-12, path="∇W·(A x + b) (모델 행렬) vs v₁i₁ − v₂i₂ − R₁i₁² − R₂i₂² (교재 식)", independent=True))
    res.add_check(Check("항등식이 부호 실수를 검출", "PASS" if worst_wrong > 1e-3 else "FAIL", worst_wrong, "rel", 1e-3, path="v_C2 부호를 뒤집은 도함수는 항등식을 깨야 한다", independent=True))
    # referred vs actual table
    L2, C2 = tank.L2 / n**2, tank.C2 * n**2
    res.tables.append(Table("t_ref", f"1차 환산 vs 실제 (n = N_p/N_s = {n:g})", ["양", "1차 환산 (모델 상태)", "실제 2차", "변환"],
                            [["L₂", f"{tank.L2 * 1e6:.4g} µH", f"{L2 * 1e6:.6g} µH", "L′ = n²L"], ["C₂", f"{tank.C2 * 1e9:.6g} nF", f"{C2 * 1e9:.6g} nF", "C′ = C/n²"],
                             ["i₂", "i₂′ (상태)", "i₂ = n·i₂′", "전류는 n배 (n < 1이면 작아짐)"], ["v₂", "v₂′ = ±n·V_o", "±V_o", "전압은 1/n배"], ["v_C2", "v_C2′ (상태)", "v_C2 = v_C2′/n", ""],
                             ["V_o", "(상태 v_o는 실제값)", "V_o", "출력 C 전압은 환산하지 않는다"]], note="모든 2차 양이 환산값인지 실제값인지 표시한다."))
    # trajectory: identity along a periodic orbit (with the loss term), including off intervals
    o = operating_point(v, v["f_lo"], v["f_hi"])
    if o:
        f0, orb = o
        T = orb.s.T
        tr = simulate(orb.s, orb.s.q_start(orb.x), orb.x, 0.0, T)
        led = energy_ledger(tr, orb.s, 0.0, T, ["p_in"], ["p_load"], ["p_R"], rated_power=v["P"])
        res.add_check(ledger_check(led))
        res.add_metric("f0", "운전점 (V_o = 목표)", f0, "Hz")
        series_from_traj(res, tr, {"i1": ("i₁", "A"), "i2": ("i₂′ (1차 환산)", "A"), "i2_act": ("i₂ 실제 = n·i₂′", "A"), "v2": ("v₂′", "V")}, per_segment=40)
        bands = bands_from(tr, 0.0, T)
        res.add_plot("p_ref", "환산 전류와 실제 전류: 같은 파형, n배 차이", ["i2", "i2_act"], y_label="전류", y_unit="A", bands=bands, group="id", level="C", proved="상태는 1차 환산값이고 표시할 때만 실제값으로 바꾼다.", not_yet="n배 환산은 이상 변압기 기준이다. 누설의 실제 배치(1·2차 분할), 권선 capacitance, 자화 branch 위치는 환산 T 모델의 가정이며 실측 식별(EX04 identification)이 필요하다.")
        res.add_plot("p_v2", "정류기 입력 v₂′: 도통 중 ±n·V_o, off 구간에는 떠 있다", ["v2", "i1"], y_label="값", y_unit="", bands=bands, group="id", level="C", proved="off 구간에서도 항등식이 성립한다 (i₂ = 0).", not_yet="off 구간의 떠 있는 전압은 정류기 접합 용량이 없는 이상 모델의 값이다. 실제로는 이 노드가 정류기 용량과 공진하며 링잉한다.")
        res.circuit = {"diagram": resonant_circuit("CLLC", v["Vlink"], f"{n:g}", tank, "FB").to_json(), "intervals": bands, "plot_group": "id"}
    res.verdict("PASS_WITHIN_MODEL", "상태식이 모든 모드에서 교재 에너지 항등식을 만족한다 (부호·환산 검산)")
    res.verdict("INFO", "항등식은 모델의 부호·환산을 검산할 뿐 시간영역 해의 정확도나 설계 통과를 뜻하지 않는다")
    res.assumptions += ["1차 환산 T-모델 (교재 E05) + 출력 C_o·R_L", "이상 다이오드 3-상태 정류기 (off = i₂ = 0 제약, 떠 있는 노드)"]
    res.not_valid_for += ["SR 채널/바디다이오드 구분", "권선 간 용량·누설 분포"]
    res.interpretation = "상태식을 세운 뒤 가장 먼저 할 일은 에너지 항등식 검산이다. 환산(n², 1/n²)이나 부호를 하나만 틀려도 항등식이 O(1)로 깨진다. off 구간은 저항으로 흉내 내지 않고 i₂ = 0 제약과 떠 있는 노드로 푼다."
    return res


# ======================================================================================
# 2. FHA roots vs switching operating points (R load)
# ======================================================================================


def run_points(v: dict) -> Result:
    res = Result("EX05", "operating_points", "C (스위칭 주기해) + A (FHA)")
    tank = _tank(v)
    n = v["n"]
    Racv = rac(n, v["Vref"], v["P"])
    g_req = n * v["Vref"] / v["Vlink"]
    roots = [r for r in fha_roots(tank, Racv, g_req, v["f_min"], v["f_max"]) if r["inductive"]]
    # seed must still fail (regression of the preserved failure)
    seed = Tank(v["L1"], v["C1"], v["Lm"], v["L1"], v["C1"])
    gs, _ = max_inductive_gain(seed, rac(1.0, v["Vref"], v["P"]), v["f_min"], v["f_max"])
    res.add_check(Check("seed (n = 1) 실패 보존", "PASS" if gs < 1.0 * v["Vref"] / v["Vlink"] else "FAIL", gs, "", v["Vref"] / v["Vlink"], path="FL10 seed: inductive 최대 gain < 필요 gain", independent=True, detail="수정안 결과가 초기 실패를 지우지 않았는지 회귀 확인"))
    # switching vo(f) with continuation
    fs = np.linspace(v["sweep_lo"], v["sweep_hi"], int(v["points"]))
    vo_td, vo_fha, prev = [], [], None
    for f in fs:
        o = _orbit(v, float(f), prev)
        vo_td.append(_vo_avg(o) if o else None)
        prev = o or prev
        vo_fha.append(fha(tank, float(f), Racv).gain * v["Vlink"] / n)
    res.add_series("vo_td", "스위칭 주기해 (R_L 부하)", "V", fs.tolist(), vo_td, style="points")
    res.add_series("vo_fha", "FHA (같은 R_ac)", "V", fs.tolist(), vo_fha, dash=True)
    # switching values at the FHA roots and slopes (central difference, separate periodic solutions)
    mk, rows = [], []
    for i, r in enumerate(roots):
        o = _orbit(v, r["f"])
        d = 50.0
        op = _orbit(v, r["f"] + d, o)
        om = _orbit(v, r["f"] - d, o)
        slope_td = (_vo_avg(op) - _vo_avg(om)) / (2 * d) * 1e3
        slope_fha = r["slope_per_Hz"] * v["Vlink"] / n * 1e3
        vt = _vo_avg(o)
        tag = "lo" if r["slope_per_Hz"] > 0 else "hi"
        res.add_metric(f"vo_at_{tag}", f"FHA 해 {r['f'] / 1e3:.3f} kHz에서 스위칭 V_o", vt, "V", basis=f"FHA는 {v['Vref']:g} V")
        res.add_metric(f"slope_td_{tag}", f"{r['f'] / 1e3:.3f} kHz 스위칭 기울기 dV_o/df", slope_td, "V/kHz", basis="±50 Hz 별도 주기해")
        res.add_metric(f"slope_fha_{tag}", f"{r['f'] / 1e3:.3f} kHz FHA 기울기", slope_fha, "V/kHz")
        rows.append([f"{r['f'] / 1e3:.3f} kHz", f"{vt:.2f} V", f"{slope_fha:+.3f} V/kHz", f"{slope_td:+.3f} V/kHz", "같음" if slope_fha * slope_td > 0 else "반대"])
        mk.append({"x": r["f"], "y": v["Vref"], "short": f"F{i + 1}", "label": f"FHA 해 {r['f'] / 1e3:.3f} kHz"})
    op = operating_point(v, v["f_lo"], v["f_hi"])
    if op:
        f0, o0 = op
        res.add_metric("f0", f"스위칭 모델에서 V_o = {v['Vref']:g} V인 주파수", f0, "Hz", basis=f"탐색 {v['f_lo'] / 1e3:g}–{v['f_hi'] / 1e3:g} kHz")
        mk.append({"x": f0, "y": v["Vref"], "short": "S", "label": f"스위칭 운전점 {f0 / 1e3:.4f} kHz"})
        xT = ivp_period(o0.s, o0.x)
        dev = float(np.max(np.abs(xT - o0.x) / scales_for(o0.s)))
        res.add_check(Check("독립 경로: 교재 상태식 + 출력 C/R, DOP853 한 주기", "PASS" if dev < 1e-7 else "FAIL", dev, "rel", 1e-7, path="상태식을 직접 적어 solve_ivp 이벤트로 적분", independent=True))
        tr = simulate(o0.s, o0.s.q_start(o0.x), o0.x, 0.0, o0.s.T)
        res.add_check(ledger_check(energy_ledger(tr, o0.s, 0.0, o0.s.T, ["p_in"], ["p_load"], ["p_R"], rated_power=v["P"])))
    ipk = int(np.nanargmax([x if x is not None else np.nan for x in vo_td]))
    res.add_metric("f_peak_td", "스위칭 V_o 최대 주파수 (sweep 격자)", float(fs[ipk]), "Hz", basis="FHA peak와 비교")
    gm, fm = max_inductive_gain(tank, Racv, v["f_min"], v["f_max"])
    res.add_metric("f_peak_fha", "FHA gain peak 주파수", fm, "Hz")
    res.add_plot("p_vo", f"출력 전압 vs 주파수 (R_L = {_RL(v):.1f} Ω, C_o = {v['Co'] * 1e6:g} µF): FHA와 스위칭 해", ["vo_td", "vo_fha"], x_label="f_s", x_unit="Hz", y_label="V_o", y_unit="V", kind="xy", level="C + A",
                 hlines=[{"y": v["Vref"], "label": f"목표 {v['Vref']:g} V"}], vlines=[{"x": v["f_min"], "label": "하한"}], markers=mk,
                 proved="각 주파수의 정류기 포함 주기해로 V_o를 구해 FHA 곡선과 비교했다.", not_yet="이상 소자. dead time·C_oss·SR 타이밍은 gain peak 위치를 더 옮길 수 있다.")
    res.tables.append(Table("t_roots", "FHA 두 해의 스위칭 결과", ["FHA 해", "스위칭 V_o", "FHA 기울기", "스위칭 기울기", "부호"], rows,
                            note="기울기는 FHA 식의 ±10 Hz 차분과 스위칭 주기해의 ±50 Hz 차분 (서로 다른 계산)."))
    res.verdict("PASS_WITHIN_MODEL", "각 점은 정류기 포함 이상 스위칭 주기해 (독립 적분·에너지 일치)")
    flips = [row for row in rows if row[-1] == "반대"]
    if flips:
        res.verdict("OUT_OF_VALIDITY", f"FHA의 아래 branch 해({flips[0][0]})는 스위칭 모델에서 기울기 부호가 반대다 — 이 부하에서는 FHA의 두 branch 구조를 제어 설계 근거로 쓰지 않는다")
    if op:
        res.verdict("INFO", f"스위칭 모델의 목표 전압 운전점은 {op[0] / 1e3:.4f} kHz 하나 (주파수 범위 안)")
    res.assumptions += [f"출력 C_o = {v['Co'] * 1e6:g} µF, R_L = V²/P (전압 루프용 부하)", "환산 대칭 CLLC (FL10 수정안)", "이상 소자"]
    res.not_valid_for += ["배터리(강한 전압원) 부하 — FL10 ‘시간영역 검증’", "실제 소자 ZVS·손실"]
    res.interpretation = (
        "FHA는 gain peak를 약 141.5 kHz에 두고 목표 전압을 두 주파수에서 만족한다고 본다. 정류기가 있는 스위칭 모델은 f_r 아래에서 gain이 더 높고 peak가 훨씬 낮은 주파수에 있어, "
        "주파수 범위 안에서는 V_o가 주파수에 대해 단조 감소한다. 그래서 FHA의 아래 해(136 kHz)는 스위칭 모델에서 기울기 부호가 FHA와 반대이고, 목표 전압 운전점은 하나뿐이다."
    )
    res.circuit = _context_circuit(v, "출력은 C_o ∥ R_L (전압 조절 모델). 운전점은 주파수 sweep과 연속(continuation)으로 찾는다.")
    return res


# ======================================================================================
# 3. Periodic solution, Floquet multipliers, perturbation
# ======================================================================================


def run_floquet(v: dict) -> Result:
    res = Result("EX05", "floquet", "C (주기해·Floquet, plant only)")
    op = operating_point(v, v["f_lo"], v["f_hi"])
    if not op:
        res.verdict("NO_SOLUTION", "탐색 구간에 목표 전압 운전점이 없다")
        return res
    f0, o = op
    s, x0, T = o.s, o.x, o.s.T
    M = fd_monodromy(s, x0)
    lam = np.linalg.eigvals(M)
    order = np.argsort(-np.abs(lam))
    lam = lam[order]
    rho = float(np.abs(lam[0]))
    res.add_metric("f0", "운전점", f0, "Hz")
    res.add_metric("rho", "plant Floquet 최대 |λ|", rho, "", basis="event 시각 민감도 포함 (FD), 제어기 제외")
    lam_dom = lam[0]
    f_osc = abs(np.angle(lam_dom)) / (2 * PI * T)
    zeta_like = -math.log(abs(lam_dom)) / max(abs(np.angle(lam_dom)), 1e-12)
    res.add_metric("f_osc", "지배 모드 진동 주파수", f_osc, "Hz", basis="∠λ/(2πT)")
    res.add_metric("damp", "지배 모드 감쇠비 (근사)", zeta_like, "", basis="−ln|λ|/|∠λ|")
    rows = [[f"{z.real:+.6f} {z.imag:+.6f}j", f"{abs(z):.6f}", f"{abs(np.angle(z)) / (2 * PI * T):.1f} Hz"] for z in lam]
    res.tables.append(Table("t_mult", "Floquet multipliers (plant: 전력단+출력 C, 제어기 없음)", ["λ", "|λ|", "진동 주파수"], rows,
                            note="finite-difference Poincaré Jacobian: 섭동한 상태마다 전체 event 시뮬레이션을 다시 돌리므로 다이오드 on/off 시각 변화가 포함된다. 외부 clock(게이트)은 고정."))
    # DC offsets of the periodic orbit
    tr = simulate(s, s.q_start(x0), x0, 0.0, T)
    for name, lab in (("i1", "i₁"), ("i2", "i₂′"), ("im", "i_m")):
        res.add_metric(f"avg_{name}", f"{lab} 주기 평균 (DC offset)", tr.mean(0.0, T, name), "A", basis="C₁·C₂가 직렬이라 0이어야 한다")
    # perturbation test: deviation decay vs |lambda|^k
    K = int(v["cycles"])
    sc = scales_for(s)
    dx = np.zeros(s.ns)
    dx[s.ix["vo"]] = v["dvo"]
    dx[s.ix["i1"]] = v["di1"]
    x = x0 + dx
    dev, t = [], 0.0
    for k in range(K):
        dev.append(float(np.max(np.abs(x - x0) / sc)))
        x, _ = cycle_map(s, x, T, t)
        t += T
    ks = np.arange(K)
    res.add_series("dev", "섭동 후 편차 max|Δx|/scale", "", ks.tolist(), dev)
    res.add_series("env", "Floquet 예측 포락선 |λ|^k", "", ks.tolist(), [dev[0] * rho**k for k in ks])
    res.add_plot("p_dev", f"초기상태 섭동 후 {K}주기: 편차가 |λ|^k 포락선을 따라 줄어드는가", ["dev", "env"], x_label="주기 k", x_unit="", y_label="편차", y_unit="", kind="xy", log_y=True, level="C",
                 proved="같은 branch에서 초기값을 섭동하고 주기마다 Poincaré 편차를 추적했다.", not_yet="plant만의 수렴 — 제어기 상태·sampling delay를 포함한 폐루프는 ‘폐루프’ 실험.")
    # measured decay rate from the envelope (peaks) vs rho
    tail = np.array(dev[K // 2 :])
    meas = (tail.max() / max(np.array(dev[: K // 2]).max(), 1e-300)) ** (1.0 / (K // 2)) if tail.max() > 0 else 0.0
    res.add_metric("rate_meas", "측정 감쇠율 (반구간 최대값 비의 1/k승)", meas, "", basis="Floquet |λ|max와 비교")
    res.add_check(check_close("섭동 감쇠율: 시뮬레이션 vs Floquet |λ|max", meas, rho, 2e-3, "비선형 사이클 시뮬레이션의 포락선 vs FD monodromy 고유값", True))
    ok_dc = all(abs(m.value) < 1e-6 * max(1.0, s.i_scale) for m in res.metrics if m.key.startswith("avg_"))
    res.add_check(Check("주기해의 DC offset (i₁, i₂′, i_m 평균)", "PASS" if ok_dc else "FAIL", max(abs(m.value) for m in res.metrics if m.key.startswith("avg_")), "A", 1e-6 * s.i_scale, path="shooting 주기해에서 정확 적분", independent=False))
    res.verdict("PASS_WITHIN_MODEL" if rho < 1 else "UNSTABLE", f"plant 주기해의 ρ(M) = {rho:.6f}" + (" < 1: 외부 clock 구동 sampled map의 국소 수렴" if rho < 1 else " ≥ 1"))
    res.verdict("INFO", "plant-only ρ < 1은 폐루프 안정성을 대신하지 않는다 (제어기 상태·지연 미포함)")
    if 1 - rho < 1e-3:
        res.verdict("MARGINAL", f"지배 모드 감쇠가 매우 작다 (주기당 {100 * (1 - rho):.3f} %, 약 {f_osc:.0f} Hz 진동) — 출력 C와 tank 등가 L의 공진")
    res.assumptions += ["외부 clock(고정 주파수)으로 구동되는 sampled map", "FD 섭동 1e-6 × 상태 scale, event 재탐색 포함"]
    res.not_valid_for += ["폐루프 안정성", "대신호(기동·포화)"]
    res.interpretation = "주기해를 찾았다고 끝이 아니다. 같은 branch에서 초기값을 흔들면 편차가 지배 Floquet multiplier의 거듭제곱으로 줄어든다. 이 운전점의 지배 모드는 출력 C와 tank의 등가 인덕턴스가 만드는 약 1 kHz 공진으로, 주기당 감쇠가 0.05 % 수준이라 수 ms 동안 울린다."
    res.circuit = _context_circuit(v, "주기해 주변의 초기 상태를 흔들어 한 주기 map의 고유값(Floquet 승수)을 구한다.")
    return res


# ======================================================================================
# 4. Branch-specific Gvf by the sampled small-signal model, verified by FM injection
# ======================================================================================


def run_gvf(v: dict) -> Result:
    res = Result("EX05", "gvf", "C (sampled small-signal, event 포함)")
    tank = _tank(v)
    n = v["n"]
    Racv = rac(n, v["Vref"], v["P"])
    roots = [r for r in fha_roots(tank, Racv, n * v["Vref"] / v["Vlink"], v["f_min"], v["f_max"]) if r["inductive"]]
    pts = [(f"FHA 해 {r['f'] / 1e3:.3f} kHz", r["f"]) for r in roots]
    op = operating_point(v, v["f_lo"], v["f_hi"])
    if op:
        pts.append((f"스위칭 운전점 {op[0] / 1e3:.3f} kHz", op[0]))
    fm = np.geomspace(v["fm_lo"], v["fm_hi"], int(v["points"]))
    mags, phs, rows = [], [], []
    val = None
    for i, (lab, f) in enumerate(pts):
        o = _orbit(v, f)
        A, Bf, C = linear_map(o)
        G = gvf(A, Bf, C, fm, o.s.T) * 1e3  # V/kHz
        g0 = float((C @ np.linalg.solve(np.eye(A.shape[0]) - A, Bf)).real) * 1e3
        d = 20.0
        vp, vm_ = _orbit(v, f + d, o), _orbit(v, f - d, o)
        slope = (vp.vo - vm_.vo) / (2 * d) * 1e3  # sampled vo at the section, same output as C
        k_pk = int(np.argmax(np.abs(G)))
        res.add_series(f"mag{i}", lab, "V/kHz", fm.tolist(), np.abs(G).tolist())
        res.add_series(f"ph{i}", lab, "deg", fm.tolist(), np.degrees(np.unwrap(np.angle(G))).tolist())
        mags.append(f"mag{i}")
        phs.append(f"ph{i}")
        rows.append([lab, f"{g0:+.4f}", f"{slope:+.4f}", f"{np.abs(G[k_pk]):.3g} @ {fm[k_pk]:.0f} Hz", f"{np.abs(G[k_pk]) / abs(g0):.1f}×"])
        res.add_check(check_close(f"G_vf(0) vs 정상상태 기울기 ({lab})", g0, slope, 2e-3, "선형 map의 DC 이득 C(I−A)⁻¹B vs ±20 Hz 별도 주기해의 v_o 차분", True, "V/kHz"))
        if op and abs(f - op[0]) < 1e-6:
            val = (o, A, Bf, C)
            res.add_metric("g0_op", "운전점 G_vf(0)", g0, "V/kHz")
            res.add_metric("peak_op", "운전점 |G_vf| 공진 peak", float(np.abs(G[k_pk])), "V/kHz", basis=f"{fm[k_pk]:.0f} Hz")
    res.add_plot("p_mag", "|G_vf| = |Δv_o/Δf_s| — 정상 기울기(DC)보다 공진 peak가 훨씬 크다", mags, x_label="변조 주파수", x_unit="Hz", y_label="|G_vf|", y_unit="V/kHz", kind="xy", log_x=True, log_y=True, level="C",
                 proved="Poincaré map 선형화(event 시각 포함)로 branch별 G_vf를 구했다.", not_yet="PWM/계산 지연·샘플링 방식은 제어기 쪽에서 더한다 (폐루프 실험).")
    res.add_plot("p_ph", "∠G_vf", phs, x_label="변조 주파수", x_unit="Hz", y_label="위상", y_unit="deg", kind="xy", log_x=True, level="C", proved="DC 위상 180°는 음의 기울기(주파수↑ → V_o↓)를 뜻한다.", not_yet="FM 주입으로 검증한 sampled small-signal 응답이며 운전점 한 곳의 선형 근사다. 큰 신호(포화·branch 이동)와 센서·PWM 분해능은 포함하지 않는다.")
    res.tables.append(Table("t_gvf", "branch별 G_vf 요약", ["점", "G_vf(0) [V/kHz]", "정상 기울기 [V/kHz]", "|G_vf| peak", "peak/DC"], rows,
                            note="G_vf(0)과 정상 기울기는 서로 다른 계산(선형 map vs 별도 주기해 차분). 정상 기울기만으로는 공진 peak·위상을 알 수 없다."))
    # FM injection on the nonlinear model vs the linear prediction (operating point)
    if val:
        o, A, Bf, C = val
        s, x0, T0, f0 = o.s, o.x, o.s.T, o.s.fs
        K = int(v["fm_cycles"])

        def inject(fmod: float, df: float) -> tuple[float, float]:
            """Same FM input to the nonlinear cycle map and to the linear map; max |difference| and max |response|."""
            x = x0.copy()
            xl = np.zeros(s.ns)
            t = 0.0
            err = amp = 0.0
            for _ in range(K):
                fk = f0 + df * math.sin(2 * PI * fmod * t)
                Tk = 1.0 / fk
                x, _ = cycle_map(s, x, Tk, t)
                xl = A @ xl + Bf * (fk - f0)
                t += Tk
                d_nl = x[s.ix["vo"]] - x0[s.ix["vo"]]
                err = max(err, abs(d_nl - xl[s.ix["vo"]]))
                amp = max(amp, abs(d_nl))
            return err, amp

        for fmod in (v["fm_check1"], v["fm_check2"]):
            df = v["fm_amp"] * f0
            err, amp = inject(fmod, df)
            rel = err / max(amp, 1e-30)
            path = f"f_s에 ±{v['fm_amp'] * 100:g} % 사인 변조, {K}주기 — 같은 입력열로 비선형 사이클 map과 선형 map 비교"
            detail = f"최대 |Δv_o| {amp:.4g} V, 최대 차이 {err:.3g} V"
            status = "PASS" if rel <= 0.02 else "FAIL"
            if status == "FAIL":
                # a linearisation error from nonlinearity is second order: at 1/10 of the amplitude the relative error
                # must drop about tenfold. If it does, the amplitude is outside the small-signal range (a model limit,
                # not a defect); if it does not, the linear model itself is wrong and the check stays FAIL.
                err10, amp10 = inject(fmod, df / 10)
                rel10 = err10 / max(amp10, 1e-30)
                detail += f"; 진폭 1/10에서 상대 차이 {rel10:.3g}"
                if rel10 <= 0.2 * rel and rel10 <= 0.02:
                    status = "INFO"
                    detail += " — 오차가 진폭에 비례해 줄어드는 2차 비선형 효과"
                    res.verdict("OUT_OF_VALIDITY", f"FM 진폭 ±{v['fm_amp'] * 100:g} %에서 {fmod:g} Hz 응답이 선형 G_vf와 {rel * 100:.1f} % 다르다 — 소신호 모델의 적용범위 밖이다. 진폭을 1/10로 줄이면 차이가 {rel10 * 100:.2f} %로 준다.")
            res.add_check(Check(f"FM 주입 {fmod:g} Hz: 비선형 시뮬레이션 vs 선형 예측", status, rel, "rel", 0.02, path=path, independent=True, detail=detail))
    res.verdict("PASS_WITHIN_MODEL", "branch별 G_vf를 event 포함 sampled small-signal 모델로 구하고 비선형 FM 주입으로 확인")
    res.verdict("INFO", "정상 gain 기울기는 G_vf(0)에 대한 정보일 뿐 위상·공진극·sampling delay를 알려주지 않는다")
    res.assumptions += ["cycle-to-cycle map: 주기 시작 샘플 v_o, 주기마다 한 번 바뀌는 주파수 명령", f"FM 주입 진폭 ±{v['fm_amp'] * 100:g} % (소신호)"]
    res.not_valid_for += ["f_s/2 이상의 변조", "대신호 과도"]
    res.interpretation = "주파수에서 출력까지의 전달함수는 운전점마다 다르다. DC 이득은 정상 기울기와 같지만, 출력 C와 tank 등가 L의 공진 때문에 1 kHz 부근에서 |G_vf|가 DC보다 한 자릿수 이상 커지고 위상이 급변한다. 기울기 부호만 보고 폐루프 이득을 정하면 이 공진을 놓친다."
    res.circuit = _context_circuit(v, "입력: 스위칭 주파수의 작은 FM 변조, 출력: v_o (sampled small-signal).")
    return res


# ======================================================================================
# 5. Closed loop: PI with delay and saturation, controller states in the Jacobian
# ======================================================================================


def closed_loop_matrix(A, Bf, C, Kp, Ki, sign, T0):
    """Deviation dynamics of [x; I; f_applied]: integrator on the cycle-start sample, one-cycle delay."""
    ns = A.shape[0]
    Acl = np.zeros((ns + 2, ns + 2))
    Acl[:ns, :ns] = A
    Acl[:ns, ns + 1] = Bf
    Acl[ns, :ns] = -Ki * T0 * C
    Acl[ns, ns] = 1.0
    Acl[ns + 1, :ns] = -sign * Kp * C
    Acl[ns + 1, ns] = sign
    return Acl


def run_closed_loop(v: dict) -> Result:
    res = Result("EX05", "closed_loop", "C (비선형 폐루프) + 선형 폐루프 Jacobian (제어기 상태 포함)")
    op = operating_point(v, v["f_lo"], v["f_hi"])
    if not op:
        res.verdict("NO_SOLUTION", "운전점 없음")
        return res
    f0, o = op
    s, x0, T0 = o.s, o.x, o.s.T
    A, Bf, C = linear_map(o)
    sign = -1.0 if v["branch_sign"] == "negative" else 1.0
    Kp, Ki = v["Kp"], v["Ki"]
    Acl = closed_loop_matrix(A, Bf, C, Kp, Ki, sign, T0)
    ev = np.linalg.eigvals(Acl)
    k = int(np.argmax(np.abs(ev)))
    rho = float(np.abs(ev[k]))
    f_mode = abs(np.angle(ev[k])) / (2 * PI * T0)
    res.add_metric("f0", "운전점", f0, "Hz")
    res.add_metric("rho_cl", "폐루프 최대 |λ| (plant + 적분기 + 1주기 지연)", rho, "", basis=f"지배 모드 {f_mode:.1f} Hz")
    res.add_metric("rho_plant", "plant만의 최대 |λ|", float(np.max(np.abs(np.linalg.eigvals(A)))), "")
    g0 = float((C @ np.linalg.solve(np.eye(A.shape[0]) - A, Bf))) * 1e3
    res.add_metric("g0", "plant 정상 기울기 G_vf(0)", g0, "V/kHz")
    slope_ok = (g0 < 0) == (sign < 0)
    res.add_metric("slope_ok", "제어기 부호가 정상 기울기와 맞는가", "예" if slope_ok else "아니오 (반대 branch 가정)", "")
    # nonlinear closed loop: load step
    K = int(v["cycles"])
    k_step = int(v["step_cycle"])
    s_after = _sys(v, f0, RL=_RL(v) * v["load_step"])
    x = x0.copy()
    integ = 0.0
    f_app = f0
    t = 0.0
    ts, vos, fcs = [], [], []
    sat_hits = 0
    for kk in range(K):
        vo = float(x[s.ix["vo"]])
        e = v["Vref"] - vo
        sysk = s if kk < k_step else s_after
        Tk = 1.0 / f_app
        ts.append(t)
        vos.append(vo)
        fcs.append(f_app)
        x, _ = cycle_map(sysk, x, Tk, t)
        t += Tk
        u = Kp * e + integ
        f_new = f0 + sign * u
        f_sat = min(max(f_new, v["f_min"]), v["f_max"])
        if f_sat == f_new:
            integ += Ki * T0 * e  # conditional integration (anti-windup)
        else:
            sat_hits += 1
        f_app = f_sat
    res.add_series("vo", "v_o (주기 시작 샘플)", "V", ts, vos)
    res.add_series("fc", "주파수 명령 − f₀", "Hz", ts, [f - f0 for f in fcs])
    res.add_plot("p_vo", f"부하 계단 (R_L × {v['load_step']:g}, {k_step}주기째) 후 출력 전압 — 비선형 스위칭 폐루프", ["vo"], x_label="t", x_unit="s", y_label="v_o", y_unit="V", level="C", hlines=[{"y": v["Vref"], "label": "목표"}],
                 proved="PI(1주기 계산 지연·주파수 포화·조건부 적분)를 비선형 스위칭 모델에 직접 연결해 시뮬레이션했다.", not_yet="센서 필터·PWM 분해능·SR 없음.")
    res.add_plot("p_fc", f"주파수 명령의 변화 f_s − f₀ (f₀ = {f0 / 1e3:.3f} kHz, 포화 {v['f_min'] / 1e3:g}–{v['f_max'] / 1e3:g} kHz)", ["fc"], x_label="t", x_unit="s", y_label="f_s − f₀", y_unit="Hz", level="C", hlines=[{"y": 0.0, "label": "0"}],
                 proved="같은 폐루프 시뮬레이션의 PI 출력(주파수 명령)이다. 포화에 닿는지와 적분이 멈추는지(조건부 적분)를 v_o 응답과 함께 읽을 수 있다.",
                 not_yet="주파수 명령은 연속값이다. 실제 타이머의 주파수 분해능, 변경 시점의 위상 연속성, SR 타이밍 갱신은 모델에 없다.")
    # small-signal validation of the closed-loop linear map: a separate 0.5 V reference step, short window
    Kv = int(v["val_cycles"])
    dref = -0.5
    xv = x0.copy()
    iv, fv, tv = 0.0, f0, 0.0
    zz = np.zeros(A.shape[0] + 2)
    err = amp = 0.0
    for kk in range(Kv):
        e = (v["Vref"] + dref) - float(xv[s.ix["vo"]])
        d_nl = float(xv[s.ix["vo"]]) - v["Vref"]
        err = max(err, abs(d_nl - zz[s.ix["vo"]]))
        amp = max(amp, abs(d_nl))
        xv, _ = cycle_map(s, xv, 1.0 / fv, tv)
        tv += 1.0 / fv
        u = Kp * e + iv
        iv += Ki * T0 * e
        fv = f0 + sign * u
        zn = Acl @ zz
        zn[A.shape[0]] += Ki * T0 * dref
        zn[A.shape[0] + 1] += sign * Kp * dref
        zz = zn
    res.add_check(Check("선형 폐루프(제어기 상태 포함) vs 비선형 폐루프: 0.5 V 기준 계단", "PASS" if err <= 0.05 * max(amp, 1e-9) else "FAIL", err / max(amp, 1e-9), "rel", 0.05,
                        path=f"plant FD Jacobian + 적분기 + 1주기 지연의 선형 map vs 비선형 사이클 시뮬레이션 ({Kv}주기, 소신호)", independent=True, detail=f"최대 차이 {err:.3g} V / 응답 {amp:.3g} V"))
    dev_end = abs(vos[-1] - v["Vref"])
    res.add_metric("vo_end", "창 끝 출력 전압", vos[-1], "V", basis=f"{K}주기 = {t * 1e3:.2f} ms")
    res.add_metric("f_end", "창 끝 주파수 명령", fcs[-1], "Hz")
    res.add_metric("sat", "포화 주기 수", sat_hits, "")
    res.add_metric("f_drift", "창 동안 주파수 명령 이동", fcs[-1] - f0, "Hz", basis="반대 부호면 오차를 키우는 방향으로 움직인다")
    tau_cycles = 1.0 / max(1 - rho, 1e-12) if rho < 1 else float("inf")
    res.add_metric("tau_ms", "지배 모드 시정수", tau_cycles * T0 * 1e3 if math.isfinite(tau_cycles) else float("nan"), "ms", basis="1/(1−ρ) 주기")
    if rho >= 1.0:
        res.verdict("UNSTABLE", f"폐루프 ρ = {rho:.6f} ≥ 1 (지배 모드 {f_mode:.0f} Hz)" + ("" if slope_ok else " — 제어기 부호가 실제 기울기와 반대"))
        if slope_ok:
            res.verdict("INFO", "기울기 부호는 맞지만 불안정하다: 정상 기울기만으로 안정성을 선언하지 않는다")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"폐루프 ρ = {rho:.6f} < 1 (제어기 상태·1주기 지연 포함)")
        if 1 - rho < 5e-4:
            res.verdict("MARGINAL", f"감쇠가 매우 작다 (주기당 {100 * (1 - rho):.3f} %, 시정수 {tau_cycles * T0 * 1e3:.1f} ms)")
    if sat_hits:
        res.verdict("FAIL_CONSTRAINT" if dev_end > 0.05 * v["Vref"] else "INFO", f"주파수 명령이 {sat_hits}주기 동안 포화 — 창 끝 V_o {vos[-1]:.1f} V")
    res.assumptions += ["PI: f = f₀ + s·(K_p e + ∫K_i e dt), s = 제어기가 가정한 기울기 부호", "주기 시작에서 v_o 샘플, 다음 주기에 적용(1주기 계산 지연)", "조건부 적분 anti-windup, 주파수 포화 120–210 kHz"]
    res.not_valid_for += ["센서·ADC 필터", "전류 루프·charge control", "실제 배터리 충전 모드"]
    res.interpretation = (
        "제어기 부호가 정상 기울기와 맞아도, 출력 C와 tank 등가 L의 약 1 kHz 공진이 거의 감쇠되지 않아 적분 이득을 조금만 올려도 폐루프 multiplier가 1을 넘는다. "
        "반대로 FHA 아래 branch를 믿고 부호를 뒤집으면 실수 multiplier가 1을 넘어 주파수가 한쪽으로 달아난다. 안정성은 plant의 event 포함 Jacobian에 제어기 상태·지연을 붙여 판정한다."
    )
    res.circuit = _context_circuit(v, "PI가 주기마다 v_o를 표본화해 주파수 명령을 만든다 (1주기 계산 지연, 주파수 포화, 조건부 적분).")
    return res


# ======================================================================================
# 6. Start-up (empty capacitor vs connected battery) and a reverse failure
# ======================================================================================


def run_startup(v: dict) -> Result:
    res = Result("EX05", "startup_reverse", "C (대신호 과도) + A")
    tank = _tank(v)
    n = v["n"]
    op = operating_point(v, v["f_lo"], v["f_hi"])
    f0 = op[0] if op else 148e3
    o0 = op[1] if op else None
    ss = summarize(o0.s, o0.traj, o0.s.T) if o0 else None
    if ss:
        res.add_metric("i1_pk_ss", "정상상태 i₁ peak", ss["i1_pk"], "A", basis=f"{f0 / 1e3:.3f} kHz 운전점")
        res.add_metric("vC1_pk_ss", "정상상태 v_C1 peak", ss["vC1_pk"], "V")
    # A. empty output capacitor, start at f_start and ramp to f0
    K = int(v["cycles"])
    ramp = int(v["ramp_cycles"])
    s = _sys(v, f0)
    x = np.zeros(s.ns)
    t = 0.0
    ts, i1pk, vcpk, vos, fcs = [], [], [], [], []
    E_in = E_out = E_R = 0.0
    W = s.stored_energy()
    z_start = np.append(x, 1.0)
    n_ivp = int(v["ivp_cycles"])
    x_ivp = x.copy()
    for k in range(K):
        fk = v["f_start"] + (f0 - v["f_start"]) * min(1.0, k / max(ramp, 1))
        Tk = 1.0 / fk
        x_prev = x
        x, tr = cycle_map(s, x, Tk, t)
        E_in += tr.energy(t, t + Tk, "p_in")
        E_out += tr.energy(t, t + Tk, "p_load")
        E_R += tr.energy(t, t + Tk, "p_R")
        if k < n_ivp:
            x_ivp = ivp_period(s, x_ivp, T=Tk)
            if k == n_ivp - 1:
                dev_ivp = float(np.max(np.abs(x_ivp - x) / scales_for(s)))
        lo, hi = tr.extrema(t, t + Tk, "i1")
        clo, chi = tr.extrema(t, t + Tk, "vC1")
        ts.append(t)
        i1pk.append(max(abs(lo), abs(hi)))
        vcpk.append(max(abs(clo), abs(chi)))
        vos.append(float(x_prev[s.ix["vo"]]))
        fcs.append(fk)
        t += Tk
    z_end = np.append(x, 1.0)
    dW = 0.5 * float(z_end @ W @ z_end) - 0.5 * float(z_start @ W @ z_start)
    resid = E_in - E_out - E_R - dW
    norm = max(abs(E_in), 0.01 * v["P"] * t)
    res.add_check(Check("기동 전체 에너지 잔차 (E_in − E_load − E_R − ΔW)", "PASS" if abs(resid) / norm < 1e-6 else "FAIL", resid / norm, "rel", 1e-6,
                        path="주기마다 포트·부하 에너지를 정확 적분해 합산, 시작/끝 저장에너지 차", independent=True, detail=f"E_in {E_in:.4g} J, E_load {E_out:.4g} J, ΔW {dW:.4g} J"))
    if n_ivp:
        res.add_check(Check(f"독립 경로: 교재 상태식 DOP853로 기동 첫 {n_ivp}주기 연쇄 적분", "PASS" if dev_ivp < 1e-6 else "FAIL", dev_ivp, "rel", 1e-6,
                            path="주기마다 다른 T_k로 solve_ivp(이벤트) 연쇄 vs 엔진 사이클 map", independent=True))
    res.add_series("i1pk", "주기별 i₁ peak (빈 C_o 기동)", "A", ts, i1pk)
    res.add_series("vo_up", "v_o", "V", ts, vos)
    res.add_series("f_up", "주파수 명령", "Hz", ts, fcs)
    first = int(v["first_cycles"])
    pk_first = max(i1pk[:first])
    pk_all = max(i1pk)
    k_all = int(np.argmax(i1pk))
    res.add_metric("i1_pk_start", f"기동 첫 {first}주기의 i₁ peak", pk_first, "A", basis=f"{v['f_start'] / 1e3:g} kHz에서 시작, 출력 C 0 V")
    res.add_metric("i1_pk_max", "기동 전체의 i₁ peak", pk_all, "A", basis=f"t = {ts[k_all] * 1e3:.2f} ms, 명령 {fcs[k_all] / 1e3:.1f} kHz, v_o {vos[k_all]:.0f} V — 공진 근처를 낮은 v_o로 지날 때")
    if ss:
        res.add_metric("inrush_ratio", "첫 주기 peak / 정상 peak", pk_first / ss["i1_pk"], "", basis="1보다 크면 높은 시작 주파수만으로 inrush가 제한되지 않은 것")
    res.add_metric("vo_max_start", "기동 중 v_o 최대 (overshoot)", max(vos), "V", basis=f"목표 {v['Vref']:g} V")
    res.add_metric("vo_end_start", "기동 창 끝 v_o", vos[-1], "V", basis=f"{K}주기")
    hlines = [{"y": ss["i1_pk"], "label": "정상 peak"}] if ss else []
    res.add_plot("p_start", f"빈 출력 C 기동: {v['f_start'] / 1e3:g} kHz에서 {ramp}주기 동안 {f0 / 1e3:.1f} kHz로 램프", ["i1pk"], x_label="t", x_unit="s", y_label="i₁ peak", y_unit="A", level="C", hlines=hlines,
                 proved="출력 C가 0 V면 정류기가 즉시 도통해 tank가 단락 부하를 본다 — 시작 주파수의 tank 임피던스만이 전류를 제한한다.", not_yet="precharge·burst 기동·전류 제한 없음.")
    res.add_plot("p_start_vo", "기동 중 출력 전압", ["vo_up"], x_label="t", x_unit="s", y_label="v_o", y_unit="V", level="C",
                 proved="빈 출력 C 기동에서 주기별 v_o를 보인다. 공진 근처를 낮은 v_o로 지날 때 생기는 overshoot의 시점을 i₁ peak 그래프와 맞춰 볼 수 있다.",
                 not_yet="개루프 주파수 램프이며 precharge·burst·전류 제한 기동은 없다. 출력 C와 부하는 합성 값이다.")
    # B. battery already connected (stiff): ramp down from f_start, then hold the end frequency
    sb = ResonantSystem(tank, v["Vlink"], v["f_start"], n, "FB", "stiff", Vo=v["Vref"], key="ex05b")
    xb = np.zeros(sb.ns)
    tb = 0.0
    pb, tb_s, fb_ = [], [], []
    Kb, Hb = int(v["cycles_b"]), int(v["hold_b"])
    f_end_b = v["f_end_b"]
    for k in range(Kb + Hb):
        fk = v["f_start"] + (f_end_b - v["f_start"]) * min(1.0, k / max(Kb - 1, 1))
        Tk = 1.0 / fk
        xb, trb = cycle_map(sb, xb, Tk, tb)
        pb.append(trb.energy(tb, tb + Tk, "p_rect") / Tk)
        tb_s.append(tb)
        fb_.append(fk)
        tb += Tk
    res.add_series("p_b", "배터리 연결 기동: 주기별 전력", "W", tb_s, pb)
    res.add_series("f_b", "주파수 명령", "Hz", tb_s, fb_)
    res.add_plot("p_batt", f"배터리({v['Vref']:g} V) 연결: {v['f_start'] / 1e3:g} → {f_end_b / 1e3:g} kHz 하강 ({Kb}주기) 후 유지 — 전력", ["p_b"], x_label="t", x_unit="s", y_label="주기 평균 전력", y_unit="W", level="C",
                 hlines=[{"y": v["P"], "label": f"{v['P'] / 1e3:g} kW"}], group="batt",
                 proved="배터리가 이미 있으면 높은 주파수에서는 정류기가 거의 도통하지 않다가, 공진 근처에서 tank 전류가 쌓이며 전력이 목표를 크게 넘는다.", not_yet="전류 제한·soft-start 알고리즘 없음 (개루프).")
    res.add_plot("p_batt_f", "주파수 명령", ["f_b"], x_label="t", x_unit="s", y_label="f_s", y_unit="Hz", level="C", group="batt",
                 proved="배터리 연결 기동에서 쓴 주파수 명령(하강 램프 후 유지)이다. 전력 그래프의 각 시점이 어떤 주파수에서 생겼는지 대응시킨다.",
                 not_yet="개루프 명령이다. 전력·전류 제한을 넣은 soft-start 알고리즘은 구현하지 않았다.")
    p_max_b = max(pb)
    res.add_metric("p_b_max", "배터리 연결 기동 중 최대 주기 전력", p_max_b, "W", basis=f"끝 주파수 {f_end_b / 1e3:g} kHz 유지 포함")
    # C. reverse at the low corner: battery-side bridge driven, stiff link
    Vb_lo, Vl_lo = v["Vbat_rev"], v["Vlink_rev"]
    rtank = tank.swapped()
    g_req = Vl_lo / (n * Vb_lo)
    gmax, fmax = max_inductive_gain(rtank, rac(1.0, Vl_lo, v["P"]), v["f_min"], v["f_max"])
    res.add_metric("rev_req", f"reverse {Vb_lo:g}→{Vl_lo:g} V 필요 gain", g_req, "")
    res.add_metric("rev_gmax", "reverse inductive 최대 FHA gain", gmax, "", basis=f"{fmax / 1e3:.2f} kHz")
    pmax_rev, f_best = 0.0, None
    prev = None
    for f in np.linspace(v["f_min"], v["f_max"], int(v["rev_points"])):
        sr = ResonantSystem(rtank, n * Vb_lo, float(f), 1.0, "FB", "stiff", Vo=Vl_lo, n_in=n, key="ex05r")
        if prev is not None:
            xr, trr, rr, okr, Mr = periodic_warm(sr, prev[0], prev[1])
        else:
            okr = False
        if not okr:
            per = periodic(sr, phasor_guess(sr, rac(1.0, Vl_lo, v["P"])))
            okr = per.converged and per.residual < 1e-8
            xr, trr, Mr = per.x0, per.traj, per.monodromy
        if okr:
            prev = (xr, Mr)
            p = trr.energy(0.0, sr.T, "p_rect") / sr.T
            if p > pmax_rev:
                pmax_rev, f_best = p, float(f)
    res.add_metric("rev_pmax", "reverse 스위칭 최대 전력 (120–210 kHz, 강한 link)", pmax_rev, "W", basis=f"{f_best / 1e3:.1f} kHz" if f_best else "")
    rev_fail = pmax_rev < v["P"]
    res.add_check(Check("reverse: FHA 판정과 스위칭 판정의 일치", "PASS" if (gmax < g_req) == rev_fail else "INFO", pmax_rev, "W", v["P"], path="FHA 최대 gain < 필요 gain ⇔ 스위칭 최대 전력 < 목표", independent=True))
    if ss:
        over = pk_all > v["i_limit"] or max(vos) > 1.1 * v["Vref"]
        res.verdict("FAIL_CONSTRAINT" if over else "PASS_WITHIN_MODEL", f"빈 C_o 개루프 램프 기동: i₁ peak {pk_all:.0f} A (첫 {first}주기 {pk_first:.1f} A, 정상 {ss['i1_pk']:.1f} A, 합성 한계 {v['i_limit']:g} A), v_o 최대 {max(vos):.0f} V")
    if p_max_b > 1.5 * v["P"]:
        res.verdict("FAIL_CONSTRAINT", f"배터리 연결 개루프 sweep이 공진 근처에서 {p_max_b / 1e3:.1f} kW까지 치솟는다 — 전류 제한 없는 주파수 sweep 기동은 불가")
    res.verdict("FAIL_CONSTRAINT" if rev_fail else "PASS_WITHIN_MODEL", f"reverse {Vb_lo:g}→{Vl_lo:g} V: 스위칭 최대 {pmax_rev / 1e3:.2f} kW (목표 {v['P'] / 1e3:g} kW), FHA 필요 {g_req:.3f} vs 최대 {gmax:.3f}")
    res.verdict("INFO", "ZVS/overshoot 수치는 설정한 합성·이상 모델에만 유효하다")
    res.assumptions += ["기동 A: 출력 C 0 V·tank 0 상태에서 주파수 램프 (개루프)", "기동 B: 강한 배터리 연결, 개루프 하강 sweep", "reverse: 배터리측 bridge 구동, link 강한 전압원, tank 교환"]
    res.not_valid_for += ["precharge 회로", "burst 기동", "SR 역전류"]
    res.interpretation = (
        "빈 출력 C로 기동하면 정류기가 곧바로 도통해 tank가 거의 단락 부하를 본다. 높은 시작 주파수는 tank 임피던스로 전류를 줄여 줄 뿐 자동 제한이 아니므로 첫 주기 전류가 정상 peak를 넘을 수 있다. "
        "배터리가 이미 연결돼 있으면 반대로 높은 주파수에서는 전력이 거의 흐르지 않다가 공진 근처에서 급증한다. 두 기동은 다른 문제이고, reverse 저전압 corner는 스위칭 모델에서도 목표 전력을 낼 수 없다."
    )
    res.circuit = _context_circuit(v, "A 빈 C_o 기동은 이 회로, B는 출력이 배터리(강성 전압원), C는 역방향(배터리측 bridge 구동, 1차측 정류)이다.")
    return res


# ======================================================================================
# 7. Tolerance screen
# ======================================================================================


def run_tolerance(v: dict) -> Result:
    res = Result("EX05", "tolerance", "A (FHA) + C (운전점 V_o)")
    n = v["n"]
    fr0 = 1 / (2 * PI * math.sqrt(v["L1"] * v["C1"]))
    tb = abs(v["L1"] - 40e-6) < 1e-12 and abs(v["C1"] - C1_TB) < 1e-15
    for key, kl, kc, refv in (("fr_pp", 1 + v["dLC"], 1 + v["dLC"], 142.857e3), ("fr_mm", 1 - v["dLC"], 1 - v["dLC"], 157.895e3), ("fr_pm", 1 + v["dLC"], 1 - v["dLC"], 150.188e3)):
        res.add_metric(key, f"f_r (L×{kl:g}, C×{kc:g})", ref.fr_tolerance(fr0, kl, kc), "Hz", ref=refv if tb and abs(v["dLC"] - 0.05) < 1e-12 else None, ref_label="교재 E05", tol=5e-6)
    op = operating_point(v, v["f_lo"], v["f_hi"])
    f0 = op[0] if op else None
    rows = []
    worst = None
    for kl in (1 - v["dLC"], 1 + v["dLC"]):
        for kc in (1 - v["dLC"], 1 + v["dLC"]):
            for km in (1 - v["dLm"], 1.0, 1 + v["dLm"]):
                t = Tank(v["L1"] * kl, v["C1"] * kc, v["Lm"] * km, v["L1"] * kl, v["C1"] * kc, v["R"], v["R"])
                Racv = rac(n, v["Vref"], v["P"])
                g_req = n * v["Vref"] / v["Vlink"]
                gmax, fmax = max_inductive_gain(t, Racv, v["f_min"], v["f_max"])
                roots = [r["f"] for r in fha_roots(t, Racv, g_req, v["f_min"], v["f_max"]) if r["inductive"]]
                vo_txt = "-"
                if f0:
                    o = _orbit(v, f0, tank=t)
                    if o:
                        vo = _vo_avg(o)
                        vo_txt = f"{vo:.1f} V ({(vo / v['Vref'] - 1) * 100:+.1f} %)"
                        if worst is None or abs(vo - v["Vref"]) > abs(worst[0] - v["Vref"]):
                            worst = (vo, kl, kc, km)
                rows.append([f"L×{kl:g}, C×{kc:g}, L_m×{km:g}", f"{ref.fr_tolerance(fr0, kl, kc) / 1e3:.3f} kHz", f"{gmax:.4f}", "해 있음" if roots else "FAIL", ", ".join(f"{x / 1e3:.2f}" for x in roots) or "-", vo_txt])
    names = []
    fs = np.linspace(v["f_min"], v["f_max"], 241)
    Racv = rac(n, v["Vref"], v["P"])
    for kl, kc in ((1, 1), (1 + v["dLC"], 1 + v["dLC"]), (1 - v["dLC"], 1 - v["dLC"]), (1 + v["dLC"], 1 - v["dLC"])):
        t = Tank(v["L1"] * kl, v["C1"] * kc, v["Lm"], v["L1"] * kl, v["C1"] * kc, v["R"], v["R"])
        key = f"g_{kl:g}_{kc:g}"
        res.add_series(key, "명목" if (kl, kc) == (1, 1) else f"L×{kl:g}, C×{kc:g}", "", fs.tolist(), [fha(t, float(f), Racv).gain for f in fs])
        names.append(key)
    res.add_plot("p_tol", "고전압 corner FHA gain: 같은 방향 공차는 곡선을 옮기고, 반대 방향은 거의 그대로", names, x_label="f_s", x_unit="Hz", y_label="|H|", y_unit="", kind="xy", level="A",
                 hlines=[{"y": n * v["Vref"] / v["Vlink"], "label": "필요 gain"}], proved="L·C 공차 조합별 FHA gain 곡선과 필요 gain을 겹쳤다 (L_m 명목).", not_yet="스위칭 V_o 편차는 표의 마지막 열 (명목 운전점 주파수 고정).")
    fails = sum(1 for r in rows if r[3] == "FAIL")
    res.tables.append(Table("t_tol", f"공차 corner (L·C ±{v['dLC'] * 100:g} %, L_m ±{v['dLm'] * 100:g} %) — FHA 가능성과 명목 운전점 주파수에서의 스위칭 V_o", ["corner", "f_r", "최대 inductive gain", "FHA", "FHA 해 [kHz]", f"스위칭 V_o @ {f0 / 1e3:.3f} kHz" if f0 else "스위칭 V_o"], rows,
                            note="L과 C가 같은 방향이면 f_r이 크게 움직이고, 반대 방향이면 거의 안 움직인다 — 상관관계가 중요하다. 합성 공차 screen이며 실제 부품 공차·상관관계로 바꾼다."))
    res.add_metric("n_fail", "FHA 해가 없는 공차 corner 수", fails, "", basis=f"{len(rows)}개 중")
    if worst:
        res.add_metric("vo_worst", "명목 주파수에서 가장 크게 벗어난 V_o", worst[0], "V", basis=f"L×{worst[1]:g}, C×{worst[2]:g}, L_m×{worst[3]:g}")
    res.verdict("FAIL_CONSTRAINT" if fails else "SCREEN_ONLY", f"{fails}/{len(rows)} 공차 corner에서 고전압 corner FHA 해가 없다" if fails else "모든 공차 corner에 FHA 해가 있다 (screen)")
    res.verdict("SCREEN_ONLY", "합성 공차 (L·C ±5 %, L_m ±15 %) — 실제 부품 공차·상관관계 확인 필요")
    res.assumptions += ["L₁·L₂′, C₁·C₂′가 같이 움직인다 (환산 대칭 유지)", "명목 운전점 주파수 고정 — 제어가 보정하기 전의 편차"]
    res.not_valid_for += ["통계적 수율", "온도 drift"]
    res.interpretation = "공차는 공진점만 옮기는 것이 아니라 가능한 최대 gain과 운전점 전압을 바꾼다. 고전압 corner처럼 여유가 1 % 미만인 곳에서는 공차 corner 일부에서 해가 사라진다. 제어가 주파수로 보정하더라도 범위와 기울기가 바뀐다."
    res.circuit = _context_circuit(v, "L·C 공차 corner마다 같은 회로를 다시 푼다. 그림의 값은 nominal이다.")
    return res


# ======================================================================================
# Content
# ======================================================================================

_Q = [
    Question(
        "FHA로 gain이 되는데 왜 양산 승인이 안 되나?",
        "정상 기본파 gain은 출력 가능성의 한 screen이다. 실제 정류 모드, 순환전류·열, 공차, 기동, 제어 안정성, switching boundary와 측정을 별도로 확인해야 한다. 이 모델에서는 FHA의 아래 branch가 스위칭 해에서 기울기 부호조차 다르다.",
        "FHA shows enough gain. Why is the design not approved for production?",
        "The steady fundamental gain is only one screen of output capability. The real rectifier mode, circulating current and heat, tolerances, start-up, control stability and switching boundaries need separate checks and measurements. In this model even the slope sign of the lower FHA branch differs in the switching solution.",
        ["screen일 뿐", "정류 모드", "공차·기동·제어"],
    ),
    Question(
        "gain 기울기 부호가 맞으면 루프가 안정한가?",
        "아니다. 정상 기울기는 G_vf(0)일 뿐이다. 출력 C와 tank 등가 L의 공진, sampling·계산 지연, 적분 이득이 폐루프 multiplier를 정한다. 여기서는 부호가 맞아도 K_i = 1e5 Hz/(V·s)에서 ρ > 1이다.",
        "If the sign of the gain slope is right, is the loop stable?",
        "No. The steady slope is only G_vf at DC. The resonance between the output capacitor and the tank's equivalent inductance, the sampling and computation delay, and the integral gain set the closed-loop multipliers. Here the sign is right and the loop is still unstable at Ki of 1e5.",
        ["G_vf(0)일 뿐", "공진·지연", "폐루프 multiplier"],
    ),
    Question(
        "Floquet multiplier를 finite difference로 구할 때 주의점은?",
        "상태를 섭동할 때마다 전체 event 시뮬레이션을 다시 돌려 다이오드 on/off 시각 변화가 Jacobian에 들어가야 한다. 외부 clock(게이트)은 고정이다. 제어기 상태·sampling delay를 빼면 plant-only 결과이므로 폐루프 안정성을 대신하지 않는다.",
        "What must you watch when computing Floquet multipliers by finite differences?",
        "Each perturbed state must re-run the full event-driven simulation, so the change in diode switching instants enters the Jacobian, while the external gate clock stays fixed. Without the controller states and sampling delay the result is plant-only and does not replace a closed-loop stability claim.",
        ["event 시각 포함", "외부 clock 고정", "plant-only 한계"],
    ),
    Question(
        "큰 시작 주파수로 기동하면 inrush가 자동으로 제한되나?",
        "아니다. 출력 C가 비어 있으면 정류기가 즉시 도통해 tank가 거의 단락 부하를 보고, 시작 주파수의 tank 임피던스만이 전류를 제한한다. 배터리가 이미 연결된 경우는 반대로 공진 근처에서 전력이 급증한다. precharge·burst·전류 제한·제어 포화를 같이 본다.",
        "Does starting at a high frequency automatically limit the inrush current?",
        "No. With an empty output capacitor the rectifier conducts at once and the tank sees nearly a short, so only the tank impedance at the start frequency limits the current. With the battery already connected it is the opposite: power surges near resonance. Precharge, burst start, current limiting and control saturation have to be considered together.",
        ["빈 C = 단락", "배터리 연결은 다름", "precharge·전류 제한"],
    ),
]

_op = [
    Param("f_lo", "운전점 탐색 하한", "Hz", 146e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="운전점"),
    Param("f_hi", "운전점 탐색 상한", "Hz", 149e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="운전점"),
]

EXPERIMENTS = [
    Experiment(
        key="state_identity",
        title="CLLC 상태식과 에너지 항등식: 환산과 부호를 먼저 검산",
        goal="x = [i₁, i₂, v_C1, v_C2]ᵀ(+v_o) 상태식을 세우고, 6개 모드(구동 ±, 정류 +/off/−)의 임의 상태에서 dW/dt = v₁i₁ − v₂i₂ − R₁i₁² − R₂i₂²를 확인한다. v_C2 부호 실수가 항등식을 깨는 것을 보이고, 환산값과 실제값을 표로 구분한다.",
        params=_params([Param("samples", "모드별 임의 상태 수", "", 50, "", vmin=5, vmax=500, kind="int", source="ASSUMED", group="검산"), Param("seed", "난수 seed", "", 7, "", vmin=0, vmax=10**6, kind="int", source="ASSUMED", group="검산")] + _op),
        presets=[
            Preset("textbook", "n = 0.93, 무손실", {}, "", ("nominal", "reference")),
            Preset("lossy", "R₁ = R₂′ = 50 mΩ", {"R": 0.05}, "손실 항 포함", ("variant", "reference")),
        ],
        run=run_identity,
        model_level="C",
        suggested_change="직렬 저항을 0 → 50 mΩ으로 넣어 항등식의 손실 항까지 확인한다.",
        prediction=Prediction(
            "v_C2의 부호를 한 곳 틀린 모델에서 항등식의 상대오차는?",
            ["여전히 1e-12 수준", "O(1)로 깨진다", "0이 된다", "모르겠다"],
            "O(1)로 깨진다",
            "항등식은 ∇W·ẋ와 포트 전력의 정확한 일치다. 부호 하나가 틀리면 C₂에 들어가는 에너지가 잘못 계산되어 임의 상태에서 큰 차이가 난다 — 그래서 항등식이 강력한 검산이다.",
            ["id_max", "id_wrong"],
        ),
        suggested={"R": 0.05},
        student="회로 방정식을 세웠으면 저장 에너지 변화가 들어온 전력 − 나간 전력 − 저항 손실과 같은지 확인한다. 이것이 맞지 않으면 어딘가 부호나 환산이 틀렸다.",
        expert="v₂가 외부 active bridge로 정해지는지 다이오드 도통 조건으로 정해지는지를 따로 지정한다. 다이오드가 모두 꺼지는 구간은 i₂ = 0 제약과 떠 있는 노드 방정식으로 풀고, 출력 루프에는 출력 C·부하 방정식을 더한다. 항등식 검산은 시간영역 해를 실행했다는 뜻이 아니다.",
        customer_ko="시뮬레이션 모델은 모든 도통 모드에서 에너지 항등식으로 부호와 환산을 검산했습니다. 2차 전류·전압은 1차 환산값과 실제값을 구분해 표시합니다.",
        customer_en="The simulation model passes the energy identity in every conduction mode, which checks its signs and referral. Secondary currents and voltages are shown separately as primary-referred and actual values.",
        questions=_Q[2:3],
        circuit="cllc",
        textbook=[TB_E05],
        reference_presets=["textbook", "lossy"],
        claim_limit="모델 구조 검산. 설계 통과 주장 없음.",
    ),
    Experiment(
        key="operating_points",
        title="FHA 두 해 vs 스위칭 운전점: 기울기 부호가 같은가",
        goal="출력 C_o·R_L(920 V, 11 kW) 부하에서 정류기 포함 스위칭 주기해의 V_o(f)를 FHA와 겹치고, FHA 두 해(136.099/147.061 kHz)에서의 실제 V_o와 기울기, 목표 전압을 주는 스위칭 운전점을 찾는다. seed 실패가 회귀로 남는지도 확인한다.",
        params=_params(_op + [
            Param("sweep_lo", "V_o 곡선 시작", "Hz", 100e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="sweep"),
            Param("sweep_hi", "V_o 곡선 끝", "Hz", 210e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="sweep"),
            Param("points", "점 개수", "", 34, "", vmin=5, vmax=150, kind="int", source="ASSUMED", group="sweep"),
        ]),
        presets=[Preset("textbook", "920/850 V, 11 kW R_L", {}, "", ("nominal", "reference"))],
        run=run_points,
        model_level="C + A",
        suggested_change="(관찰 실험) sweep 시작을 100 → 120 kHz로 바꿔 주파수 범위 안만 본다.",
        prediction=Prediction(
            "FHA 아래 해(136.1 kHz, FHA 기울기 +)에서 스위칭 모델의 dV_o/df 부호는?",
            ["+ (FHA와 같다)", "− (반대)", "0", "모르겠다"],
            "− (반대)",
            "스위칭 모델의 gain peak는 약 122 kHz로 FHA peak(141.5 kHz)보다 훨씬 낮다. 136 kHz는 스위칭 모델에서 이미 peak 위쪽이라 기울기가 음수(약 −3.8 V/kHz)다.",
            ["slope_td_lo", "slope_fha_lo", "vo_at_lo"],
        ),
        suggested={"sweep_lo": 120e3},
        student="FHA로 찾은 ‘오른쪽/왼쪽 해’가 실제 회로에서도 오른쪽/왼쪽인지 스위칭 모델로 확인해야 한다.",
        expert="branch는 등가모델의 성질이다. 정류 모드·고조파가 gain peak를 옮기면 branch 구조와 기울기 부호가 바뀐다. 제어 부호·운전 범위는 스위칭 모델(또는 측정)의 운전점에서 정한다.",
        customer_ko="FHA는 고전압 corner에 두 동작 주파수를 주지만, 실제 스위칭 모델에서는 주파수 범위 안에 목표 전압 운전점이 하나이고 FHA 아래 해의 제어 방향은 반대입니다. 제어 부호와 범위는 스위칭 기준으로 정하겠습니다.",
        customer_en="FHA gives two operating frequencies at the high corner, but in the switching model there is only one operating point for the target voltage inside the range, and the control direction at the lower FHA solution is reversed. I would set the control sign and range from the switching model.",
        questions=_Q[:1],
        circuit="cllc",
        textbook=[TB_E05, TB_13],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="이상 스위칭 주기해. 손실·ZVS 없음.",
    ),
    Experiment(
        key="floquet",
        title="주기해와 안정성은 따로: Floquet multiplier와 초기값 섭동",
        goal="목표 전압 운전점의 주기해를 shooting으로 구하고, event 시각을 포함한 FD Poincaré Jacobian의 multiplier로 국소 수렴을 판정한다. 초기상태를 섭동해 400주기 동안의 편차가 |λ|^k를 따르는지 비선형 시뮬레이션으로 확인하고, DC offset이 남지 않았는지 본다.",
        params=_params(_op + [
            Param("cycles", "섭동 후 주기 수", "", 400, "", vmin=20, vmax=5000, kind="int", source="ASSUMED", group="섭동"),
            Param("dvo", "v_o 섭동", "V", 2.0, "V", vmin=-100, vmax=100, source="ASSUMED", group="섭동"),
            Param("di1", "i₁ 섭동", "A", 1.0, "A", vmin=-100, vmax=100, source="ASSUMED", group="섭동"),
        ]),
        presets=[
            Preset("textbook", "920 V, 무손실", {}, "", ("nominal", "reference")),
            Preset("lossy", "R = 50 mΩ", {"R": 0.05, "f_lo": 145e3, "f_hi": 149e3}, "감쇠 증가", ("variant", "reference")),
        ],
        run=run_floquet,
        model_level="C",
        suggested_change="직렬 저항을 0 → 50 mΩ으로 넣어 지배 multiplier가 어떻게 바뀌는지 본다.",
        prediction=Prediction(
            "무손실 운전점에서 섭동 편차는 몇 주기면 1/10로 줄까?",
            ["10주기 이내", "수백~수천 주기", "줄지 않는다", "모르겠다"],
            "수백~수천 주기",
            "지배 multiplier가 |λ| ≈ 0.9995 (약 1 kHz 진동)라 주기당 0.05 %만 줄어든다: ln(10)/0.0005 ≈ 4600주기 ≈ 31 ms. 출력 C와 tank 등가 L의 공진을 부하 R_L만이 감쇠시킨다.",
            ["rho", "f_osc"],
        ),
        suggested={"R": 0.05},
        student="주기적으로 반복되는 해를 찾았더라도 조금 흔들었을 때 그 해로 돌아오는지는 따로 확인해야 한다.",
        expert="외부 clock 구동 sampled map에서 ρ(M) < 1은 국소 수렴 조건이다. FD Jacobian은 event 시각 변화를 포함해야 하고, 제어기 상태·지연을 뺀 plant-only 결과는 폐루프 안정성을 대신하지 않는다. 느린 수렴을 임의 저항으로 빠르게 만들면 손실·동특성이 바뀌므로 명시한다.",
        customer_ko="이 운전점은 plant 기준으로 수렴하지만 약 1 kHz 공진이 거의 감쇠되지 않아 수십 ms 동안 울립니다. 전압 루프 대역폭과 출력 필터 감쇠를 같이 설계해야 합니다.",
        customer_en="This operating point converges for the plant alone, but a resonance near 1 kHz is barely damped and rings for tens of milliseconds. The voltage-loop bandwidth and the damping of the output filter have to be designed together.",
        questions=_Q[2:3],
        circuit="cllc",
        textbook=[TB_E05, TB_E11],
        reference_presets=["textbook", "lossy"],
        runtime_hint="seconds",
        claim_limit="plant 국소 수렴. 폐루프·대신호 주장 없음.",
    ),
    Experiment(
        key="gvf",
        title="branch별 G_vf: 정상 기울기는 DC 한 점일 뿐",
        goal="FHA 두 해와 스위칭 운전점에서 사이클 map을 선형화해 G_vf(e^{jωT}) = Δv_o/Δf_s를 구하고, DC 값이 별도 주기해로 구한 정상 기울기와 같은지, 출력 공진 peak가 얼마나 큰지 본다. 운전점에서 비선형 FM 주입으로 선형 모델을 검증한다.",
        params=_params(_op + [
            Param("fm_lo", "변조 주파수 하한", "Hz", 10.0, "Hz", vmin=0.1, vmax=1e5, source="ASSUMED", group="Bode"),
            Param("fm_hi", "변조 주파수 상한", "Hz", 60e3, "kHz", vmin=10, vmax=1e6, source="ASSUMED", group="Bode"),
            Param("points", "점 개수", "", 120, "", vmin=10, vmax=1000, kind="int", source="ASSUMED", group="Bode"),
            Param("fm_amp", "FM 주입 진폭 (상대)", "", 2e-5, "", vmin=1e-7, vmax=1e-2, source="ASSUMED", group="검증"),
            Param("fm_check1", "FM 검증 주파수 1", "Hz", 300.0, "Hz", vmin=1, vmax=1e5, source="ASSUMED", group="검증"),
            Param("fm_check2", "FM 검증 주파수 2", "Hz", 1000.0, "Hz", vmin=1, vmax=1e5, source="ASSUMED", group="검증"),
            Param("fm_cycles", "FM 검증 주기 수", "", 400, "", vmin=50, vmax=5000, kind="int", source="ASSUMED", group="검증"),
        ]),
        presets=[Preset("textbook", "920/850 V, 11 kW", {}, "", ("nominal", "reference")), Preset("big_fm", "FM 진폭 0.2 %", {"fm_amp": 2e-3}, "비선형 효과", ("variant",))],
        run=run_gvf,
        model_level="C",
        suggested_change="FM 주입 진폭을 0.002 % → 0.2 %로 키워 선형 모델이 어디서 깨지는지 본다.",
        prediction=Prediction(
            "운전점 |G_vf|의 공진 peak는 DC 기울기의 몇 배쯤일까?",
            ["1배 (평탄)", "약 2배", "10배 이상", "모르겠다"],
            "10배 이상",
            "출력 C와 tank 등가 L의 공진이 R_L로만 약하게 감쇠되어 약 1 kHz에서 |G_vf|가 DC(약 3 V/kHz)보다 한 자릿수 이상 크다. 정상 기울기로 이득을 정하면 이 peak에서 루프가 불안정해질 수 있다.",
            ["g0_op", "peak_op"],
        ),
        suggested={"fm_amp": 2e-3},
        student="주파수를 천천히 바꾸면 출력은 정상 기울기만큼 움직이지만, 특정 속도로 흔들면 훨씬 크게 흔들릴 수 있다.",
        expert="G_vf는 branch·운전점마다 식별한다. finite-difference Poincaré 선형화는 event 시각 민감도를 포함하고, 비선형 FM 주입으로 소신호 범위를 확인한다. MathWorks의 주파수응답 추정 예제처럼 switching 모델에서 perturbation 기반 응답을 얻는 절차와 같은 생각이다.",
        customer_ko="주파수→출력 전달함수에 약 1 kHz 공진이 있어 정상 기울기만 보고 이득을 올리면 발진할 수 있습니다. 운전점별로 G_vf를 측정·식별해 보상기를 정하시죠.",
        customer_en="The frequency-to-output transfer function has a resonance near 1 kHz, so raising the gain from the steady slope alone can oscillate. Let's measure or identify G_vf at each operating point before fixing the compensator.",
        questions=_Q[1:2],
        circuit="cllc",
        textbook=[TB_E05, TB_E07],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="event 포함 소신호 모델 (이상 소자). 실제 보드 G_vf 아님.",
    ),
    Experiment(
        key="closed_loop",
        title="폐루프: 지연·포화 포함 PI, 부호가 맞아도 불안정할 수 있다",
        goal="v_o를 주기 시작에서 샘플하고 다음 주기에 주파수를 바꾸는 PI(조건부 적분, 120–210 kHz 포화)를 비선형 스위칭 모델에 연결해 부하 계단 응답을 보고, 제어기 상태·지연을 포함한 폐루프 Jacobian의 multiplier와 비교한다. 낮은 이득·높은 이득·반대 부호를 비교한다.",
        params=_params(_op + [
            Param("Kp", "비례 이득 K_p", "Hz/V", 0.0, "Hz/V", vmin=0, vmax=1e6, source="ASSUMED", group="제어기"),
            Param("Ki", "적분 이득 K_i", "Hz/(V·s)", 1e4, "Hz/(V·s)", vmin=0, vmax=1e9, source="ASSUMED", group="제어기"),
            Param("branch_sign", "제어기가 가정한 기울기", "", "negative", "", kind="choice", choices=[("negative", "음수 (f↑ → V_o↓)"), ("positive", "양수 (FHA 아래 branch 가정)")], source="ASSUMED", group="제어기"),
            Param("load_step", "부하 계단 R_L 배율", "", 1.25, "", vmin=0.1, vmax=10, source="ASSUMED", group="시나리오"),
            Param("step_cycle", "계단 시점 (주기)", "", 50, "", vmin=0, vmax=10000, kind="int", source="ASSUMED", group="시나리오"),
            Param("cycles", "시뮬레이션 주기 수", "", 1500, "", vmin=100, vmax=20000, kind="int", source="ASSUMED", group="시나리오"),
            Param("val_cycles", "소신호 검증 주기 수", "", 300, "", vmin=20, vmax=5000, kind="int", source="ASSUMED", group="시나리오"),
        ]),
        presets=[
            Preset("slow", "K_i = 1e4 (낮은 이득)", {}, "안정하지만 느림", ("nominal", "reference")),
            Preset("fast", "K_i = 1e5 (부호는 맞음)", {"Ki": 1e5}, "공진과 상호작용 → 불안정", ("failure", "reference")),
            Preset("wrong_sign", "반대 부호 (FHA 아래 branch 가정), K_i = 3e4", {"branch_sign": "positive", "Ki": 3e4, "cycles": 3000}, "주파수가 달아남", ("failure", "reference")),
        ],
        run=run_closed_loop,
        model_level="C",
        suggested_change="적분 이득 K_i를 1e4 → 1e5로 올린다 (부호는 그대로).",
        prediction=Prediction(
            "기울기 부호가 맞는 PI에서 K_i를 10배 올리면?",
            ["더 빨리 안정된다", "약 1 kHz 공진과 얽혀 불안정해진다", "변화 없다", "모르겠다"],
            "약 1 kHz 공진과 얽혀 불안정해진다",
            "폐루프 multiplier는 plant 공진·적분기·1주기 지연을 함께 본다. K_i = 1e5에서 지배 multiplier가 1을 넘는다 (약 1 kHz 모드). 정상 기울기 부호만으로 안정성을 선언할 수 없다.",
            ["rho_cl"],
        ),
        suggested={"Ki": 1e5},
        student="제어 방향이 맞아도 너무 세게 누르면 출력의 공진과 겹쳐 흔들림이 커진다.",
        expert="finite-difference Poincaré/Floquet를 쓸 때는 event 시각 민감도와 제어기 상태 포함 여부를 명시한다. 이 폐루프 Jacobian은 plant FD Jacobian에 적분기·1주기 지연을 붙인 것이고, 소신호 계단에서 비선형 폐루프와 비교해 확인한다. 포화가 걸리면 선형 판정은 참고용이다.",
        customer_ko="현재 전압 루프는 적분 이득을 조금만 올려도 약 1 kHz에서 발진할 수 있습니다. 출력 필터 감쇠나 전류 기반 제어를 검토하고, 운전점별 폐루프 margin을 측정으로 확인하시죠.",
        customer_en="With the current voltage loop, a small increase in integral gain can make it oscillate near 1 kHz. Let's consider damping in the output filter or a current-based control scheme, and confirm the closed-loop margin by measurement at each operating point.",
        questions=_Q[1:3],
        circuit="cllc",
        textbook=[TB_E05, TB_E07],
        reference_presets=["slow", "fast", "wrong_sign"],
        runtime_hint="seconds",
        claim_limit="이상 스위칭 plant + 이산 PI. 실제 제어기 구현·센서 없음.",
    ),
    Experiment(
        key="startup_reverse",
        title="기동과 reverse: 빈 출력 C, 연결된 배터리, 저전압 역방향",
        goal="출력 C가 빈 상태에서 210 kHz로 시작해 운전점으로 램프할 때 첫 주기 전류를 정상 peak와 비교하고, 배터리가 연결된 상태의 개루프 주파수 하강 sweep에서 공진 근처 전력 급증을 본다. reverse 저전압 corner(650→700 V)가 스위칭 모델에서도 목표 전력을 못 내는 실패 사례를 남긴다.",
        params=_params(_op + [
            Param("f_start", "기동 시작 주파수", "Hz", 210e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="기동"),
            Param("ramp_cycles", "램프 주기 수", "", 300, "", vmin=1, vmax=100000, kind="int", source="ASSUMED", group="기동"),
            Param("cycles", "기동 A 주기 수", "", 700, "", vmin=10, vmax=100000, kind="int", source="ASSUMED", group="기동"),
            Param("first_cycles", "초기 peak를 볼 주기 수", "", 20, "", vmin=1, vmax=1000, kind="int", source="ASSUMED", group="기동"),
            Param("i_limit", "합성 전류 한계 (screen)", "A", 40.0, "A", vmin=1, vmax=1e4, source="ASSUMED", group="기동"),
            Param("ivp_cycles", "독립 적분으로 확인할 첫 주기 수", "", 30, "", vmin=0, vmax=2000, kind="int", source="ASSUMED", group="기동"),
            Param("cycles_b", "배터리 기동 하강 주기 수", "", 500, "", vmin=10, vmax=100000, kind="int", source="ASSUMED", group="기동"),
            Param("hold_b", "끝 주파수 유지 주기 수", "", 700, "", vmin=0, vmax=100000, kind="int", source="ASSUMED", group="기동"),
            Param("f_end_b", "배터리 하강 끝 주파수", "Hz", 147.5e3, "kHz", vmin=1e3, vmax=1e7, source="ASSUMED", group="기동"),
            Param("Vbat_rev", "reverse 배터리 전압", "V", 650.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="reverse"),
            Param("Vlink_rev", "reverse link 전압", "V", 700.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", group="reverse"),
            Param("rev_points", "reverse 주파수 점 개수", "", 19, "", vmin=3, vmax=200, kind="int", source="ASSUMED", group="reverse"),
        ]),
        presets=[Preset("textbook", "210 kHz 시작, 300주기 램프", {}, "", ("nominal", "reference")), Preset("slow_ramp", "램프 600주기", {"ramp_cycles": 600, "cycles": 900}, "느린 램프", ("variant",))],
        run=run_startup,
        model_level="C",
        suggested_change="램프를 300 → 600주기로 늘린다.",
        prediction=Prediction(
            "빈 출력 C로 210 kHz에서 시작하면 첫 주기 i₁ peak는 정상 운전 peak보다?",
            ["작다 (높은 주파수라서)", "비슷하거나 크다", "0이다", "모르겠다"],
            "비슷하거나 크다",
            "출력 C가 0 V면 정류기 clamp가 0이라 tank가 단락 부하를 본다. 210 kHz의 tank 리액턴스만이 전류를 제한하므로 정상 peak의 2배 이상이 흐른다. 더 큰 문제는 램프가 v_o가 낮은 채로 공진 근처를 지날 때다 — 이 이상 모델에서는 수백 A까지 오르고 v_o가 목표를 넘어간다.",
            ["i1_pk_start", "i1_pk_ss", "inrush_ratio", "i1_pk_max", "vo_max_start"],
        ),
        suggested={"ramp_cycles": 600, "cycles": 900},
        student="출력 커패시터가 비어 있으면 처음 순간 출력이 단락처럼 보인다. 주파수를 높게 시작해도 그 단락 전류를 자동으로 막지는 못한다.",
        expert="빈 C 기동과 배터리 연결 기동을 분리한다: 전자는 단락 부하 inrush와 ‘낮은 v_o로 공진 통과’ 전류 급증, 후자는 공진 근처 전력 급증이다. 첫 몇 주기만 보고 inrush를 판정하면 램프 중의 더 큰 peak를 놓친다. precharge·burst 시작·제어 포화·transformer 자속을 함께 본다. reverse는 port를 바꾸면 L/C 환산·bridge 여기·sensing·precharge가 달라진다.",
        customer_ko="기동은 출력 C가 빈 경우와 배터리가 연결된 경우를 따로 설계해야 합니다. 높은 시작 주파수만으로는 inrush가 제한되지 않고, 배터리 연결 상태의 주파수 sweep은 공진 근처에서 전류가 급증합니다. 전류 제한 기반 soft-start를 제안드립니다.",
        customer_en="Start-up has to be designed separately for an empty output capacitor and for a connected battery. A high start frequency alone does not limit inrush, and a frequency sweep with the battery connected surges in current near resonance. I would propose a current-limited soft start.",
        questions=_Q[3:],
        circuit="cllc",
        textbook=[TB_E05, TB_13],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="개루프 기동·reverse의 이상 스위칭 과도. 보호 회로 없음.",
    ),
    Experiment(
        key="tolerance",
        title="공차: f_r만 움직이는 것이 아니다",
        goal="L·C ±5 %(같은/반대 방향)와 L_m ±15 %의 12개 corner에서 f_r(142.857/157.895/150.188 kHz), 고전압 corner FHA 가능성, 명목 운전점 주파수에서의 스위칭 V_o를 비교한다.",
        params=_params(_op + [
            Param("dLC", "L·C 공차", "", 0.05, "", vmin=0, vmax=0.5, source="TEXTBOOK", source_note="±5 %", group="공차"),
            Param("dLm", "L_m 공차", "", 0.15, "", vmin=0, vmax=0.9, source="TEXTBOOK", source_note="±15 %", group="공차"),
        ]),
        presets=[Preset("textbook", "L·C ±5 %, L_m ±15 %", {}, "", ("nominal", "reference"))],
        run=run_tolerance,
        model_level="A + C",
        suggested_change="L·C 공차를 5 → 2 %로 줄인다.",
        prediction=Prediction(
            "L +5 %, C −5 %(반대 방향)일 때 f_r은?",
            ["142.857 kHz", "150.188 kHz (거의 그대로)", "157.895 kHz", "모르겠다"],
            "150.188 kHz (거의 그대로)",
            "f_r = 1/(2π√(LC))라 반대 방향 공차는 거의 상쇄된다(1/√(1.05·0.95) = 1.00125). 같은 방향이면 ±5 %가 그대로 f_r 이동이 된다. 그래도 gain·운전점 전압은 L_m과 Z₀ 변화로 달라진다.",
            ["fr_pp", "fr_mm", "fr_pm"],
        ),
        suggested={"dLC": 0.02},
        student="부품값이 조금씩 다르면 공진 주파수와 gain 곡선이 움직여, 한 점에서 맞춘 설계가 다른 부품 조합에서는 목표를 못 낼 수 있다.",
        expert="공차 screen은 합성값으로 시작하고 실제 부품 공차·상관관계(같은 lot, 온도 계수)로 바꾼다. 여유가 1 % 미만인 corner는 공차에서 해가 사라질 수 있으므로 주파수 범위·권선비·tank 임피던스를 다시 본다.",
        customer_ko="L·C 공차 ±5 %에서 공진점은 142.9–157.9 kHz로 움직이고, 고전압 corner는 일부 공차 조합에서 해가 사라집니다. 부품 공차의 상관관계 데이터를 받아 다시 확인하겠습니다.",
        customer_en="With plus or minus five percent on L and C, the resonance moves between 142.9 and 157.9 kHz, and at the high-voltage corner some tolerance combinations lose their solution. I would like the correlation data of the component tolerances to re-check this.",
        questions=[],
        circuit="cllc",
        textbook=[TB_E05, TB_E11],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="합성 공차 screen. 수율·온도 drift 없음.",
    ),
]

LAB = Lab(
    id="EX05",
    title="CLLC — gain 곡선에서 실제 동역학으로",
    title_en="CLLC: from the gain curve to real dynamics",
    track="expert",
    order=5,
    path_note="E13 converter 주력 (E05–E07)",
    textbook=[TB_E05, TB_13],
    prerequisites=["FL09", "FL10"],
    summary="상태식·에너지 항등식 → FHA 두 해 vs 스위칭 운전점 → Floquet·섭동 → branch별 G_vf → 지연·포화 PI 폐루프 → 기동·reverse 실패 → 공차.",
    experiments=EXPERIMENTS,
    minimum_scope="E05: 상태식·항등식, 두 root의 스위칭 운전점·G_vf, 폐루프 step, startup/reverse 실패 사례, seed 실패 보존",
    claim_limits=["이상 소자·합성 출력 C", "plant-only Floquet ≠ 폐루프 안정성", "ZVS/overshoot 수치는 합성 모델에만 유효"],
    test_paths=["tests/test_ex05.py"],
)
