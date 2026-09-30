"""FL01 - Buck and Boost: the grammar of converters (textbook ch.02-03).

Physics: exact switched-affine models (level C) with ideal or resistive switches,
DCR, output capacitor ESR and resistive load; diode-emulation DCM as a state event
(i_L reaches zero), forced synchronous rectification allowing negative current;
state-space averaged models (level B); small-signal transfer functions (level A).
Every hand-calculation value is taken from reference/buck_boost.py, which the
simulation never imports - they are two independent paths.
"""

from __future__ import annotations

import math

import numpy as np

from ..engine.periodic import cycle_to_steady, shoot
from ..engine.switched import AffineMode, Guard, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table
from ..reference import buck_boost as ref
from ._common import bands_from_traj, decimate_minmax, energy_ledger, independent_ivp_check, ledger_check, sym_linear

TB_02 = TextbookRef("전력변환을-다시-배우는-네-개의-법칙", "02. 전력변환을 다시 배우는 네 개의 법칙")
TB_03 = TextbookRef("buck와-boost-컨버터의-문법-fl01", "03. Buck와 Boost — 컨버터의 문법 [FL01]")

# ======================================================================================
# Buck
# ======================================================================================


class Buck(HybridSystem):
    """Synchronous / diode-emulation / diode buck with R load.

    x = [i_L, v_C].  Output v_o = k (v_C + R_c i_L), k = R/(R + R_c) (ESR in the C branch).
    q: 'H' high side on, 'L' low side conducting, 'Z' both off with i_L = 0 (DCM).
    """

    state_names = ("i_L", "v_C")
    state_units = ("A", "V")

    def __init__(self, p: dict, duty_of_cycle=None):
        self.p = p
        self.Vin = p["Vin"]
        self.L = p["L"]
        self.C = p["C"]
        self.R = p["R"]
        self.Rc = p["esr"]
        self.RL = p["dcr"]
        self.R1 = p["rds_hi"]
        self.R2 = p["rds_lo"]
        self.Vf = p.get("vf", 0.0)
        self.rect = p["rect"]
        self.fs = p["fs"]
        self.T = 1.0 / p["fs"]
        self.D = p["D"]
        self.duty_of_cycle = duty_of_cycle or (lambda k: self.D)
        self.k = self.R / (self.R + self.Rc)
        self._key = f"buck|{sorted((k, v) for k, v in p.items() if k != 'rect')}|{self.rect}"

    # ---- continuous dynamics ------------------------------------------------------
    def _vsw(self, q) -> np.ndarray:
        if q == "H":
            return np.array([-self.R1, 0.0, self.Vin])
        if q == "L":
            if self.rect == "diode":
                return np.array([-self.R2, 0.0, -self.Vf])
            return np.array([-self.R2, 0.0, 0.0])
        return self._vo()  # Z: node floats at v_o (ideal, no ringing modelled)

    def _vo(self) -> np.ndarray:
        return np.array([self.k * self.Rc, self.k, 0.0])

    def mode(self, q) -> AffineMode:
        L, C, R, k = self.L, self.C, self.R, self.k
        A = np.zeros((2, 2))
        b = np.zeros(2)
        if q != "Z":
            vL = self._vsw(q) - self.RL * np.array([1.0, 0, 0]) - self._vo()
            A[0, :] = vL[:2] / L
            b[0] = vL[2] / L
        A[1, 0] = k / C
        A[1, 1] = -k / (R * C)
        return AffineMode(f"{self._key}|{q}", A, b, label=q)

    def guards(self, q):
        if q == "L" and self.rect in ("diode_emulation", "diode"):

            def reset(z):
                z = z.copy()
                z[0] = 0.0
                return z

            return [Guard("i_L=0 (DCM 진입)", np.array([1.0, 0.0, 0.0]), -1, lambda q: "Z", reset)]
        return []

    def gate_schedule(self, t0: float, t1: float):
        ev = []
        k0 = int(math.floor(t0 / self.T + 1e-9))
        k1 = int(math.ceil(t1 / self.T + 1e-9))
        for k in range(k0, k1 + 1):
            ta = k * self.T
            tb = ta + self.duty_of_cycle(k) * self.T
            if t0 - 1e-15 <= ta < t1:
                ev.append((ta, lambda q: "H"))
            if t0 - 1e-15 <= tb < t1:
                ev.append((tb, lambda q: "L"))
        return ev

    def after_event(self, q, z):
        if q == "L" and self.rect in ("diode_emulation", "diode") and z[0] <= 1e-12 * max(1.0, abs(z[0])):
            return "Z"
        return q

    # ---- outputs, energy ----------------------------------------------------------
    def outputs(self, q):
        e_i = np.array([1.0, 0.0, 0.0])
        vo = self._vo()
        vsw = self._vsw(q)
        k, R = self.k, self.R
        return {
            "iL": e_i,
            "vC": np.array([0.0, 1.0, 0.0]),
            "vo": vo,
            "io": vo / R,
            "iC": np.array([k, -k / R, 0.0]),
            "vesr": self.Rc * np.array([k, -k / R, 0.0]),
            "vsw": vsw,
            "vL": vsw - self.RL * e_i - vo,
            "i_hi": e_i if q == "H" else np.zeros(3),
            "i_lo": e_i if q == "L" else np.zeros(3),
        }

    def stored_energy(self):
        return np.diag([self.L, self.C, 0.0])

    def powers(self, q):
        e_i = np.array([1.0, 0.0, 0.0])
        vo = self._vo()
        iC = np.array([self.k, -self.k / self.R, 0.0])
        loss = self.RL * np.outer(e_i, e_i) + self.Rc * np.outer(iC, iC)
        if q == "H":
            loss = loss + self.R1 * np.outer(e_i, e_i)
        elif q == "L":
            loss = loss + self.R2 * np.outer(e_i, e_i)
            if self.rect == "diode":
                loss = loss + sym_linear(e_i, self.Vf)
        return {
            "p_in": sym_linear(e_i, self.Vin) if q == "H" else None,
            "p_out": np.outer(vo, vo) / self.R,
            "p_loss": loss,
        }

    def describe(self, q):
        return q

    # ---- independent formulation for the solver check -------------------------------
    def rhs_plain(self, q, x):
        """Same circuit written out by hand (no matrices) for the RK45 cross-check."""
        iL, vC = x
        vo = (vC + self.Rc * iL) * self.R / (self.R + self.Rc)
        io = vo / self.R
        if q == "H":
            vsw = self.Vin - self.R1 * iL
        elif q == "L":
            vsw = (-self.Vf if self.rect == "diode" else 0.0) - self.R2 * iL
        else:
            vsw = vo
        diL = 0.0 if q == "Z" else (vsw - self.RL * iL - vo) / self.L
        dvC = (iL - io) / self.C
        return np.array([diL, dvC])


def buck_params(defaults: dict | None = None) -> list[Param]:
    d = {"Vin": 48.0, "D": 0.25, "L": 100e-6, "fs": 100e3, "Io": 5.0, "C": 100e-6, "esr": 10e-3, "dcr": 0.0, "rds_hi": 0.0, "rds_lo": 0.0, "vf": 0.7, "rect": "sync"}
    d.update(defaults or {})
    return [
        Param("Vin", "입력전압 V_in", "V", d["Vin"], "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="교재 03장 예제 48 V"),
        Param("D", "duty D (개루프)", "", d["D"], "", vmin=0.01, vmax=0.99, source="TEXTBOOK", source_note="D = V_o/V_in = 0.25", validity_note="0 < D < 1"),
        Param("L", "인덕턴스 L", "H", d["L"], "µH", vmin=1e-7, vmax=0.1, source="TEXTBOOK", source_note="100 µH"),
        Param("fs", "스위칭 주파수 f_s", "Hz", d["fs"], "kHz", vmin=1e3, vmax=5e6, source="TEXTBOOK", source_note="100 kHz"),
        Param(
            "Io",
            "부하전류 I_o (공칭 V_o 기준)",
            "A",
            d["Io"],
            "A",
            vmin=1e-3,
            vmax=500,
            source="TEXTBOOK",
            source_note="5 A; 부하는 저항 R = D·V_in/I_o 로 모델링",
            description="저항부하 R = D·V_in / I_o. DCM에서는 V_o가 변하므로 실제 전류는 달라진다.",
        ),
        Param("C", "출력 커패시턴스 C", "F", d["C"], "µF", vmin=1e-9, vmax=1.0, source="TEXTBOOK", source_note="100 µF"),
        Param("esr", "ESR", "Ω", d["esr"], "mΩ", vmin=0.0, vmax=10.0, source="TEXTBOOK", source_note="10 mΩ"),
        Param("dcr", "인덕터 DCR", "Ω", d["dcr"], "mΩ", vmin=0.0, vmax=10.0, source="ASSUMED", source_note="이상=0. 넣으면 전력단 방정식에 손실이 결합된다", group="비이상"),
        Param("rds_hi", "상측 R_DS(on)", "Ω", d["rds_hi"], "mΩ", vmin=0.0, vmax=10.0, source="ASSUMED", source_note="이상=0 (합성)", group="비이상"),
        Param("rds_lo", "하측 R_DS(on) / 다이오드 저항", "Ω", d["rds_lo"], "mΩ", vmin=0.0, vmax=10.0, source="ASSUMED", source_note="이상=0 (합성)", group="비이상"),
        Param("vf", "다이오드 순방향전압 V_f", "V", d["vf"], "V", vmin=0.0, vmax=5.0, source="ASSUMED", source_note="정류 방식이 diode일 때만 사용", group="비이상"),
        Param(
            "rect",
            "하측 정류 방식",
            "",
            d["rect"],
            kind="choice",
            choices=[("sync", "강제 동기정류 (음전류 허용)"), ("diode_emulation", "diode-emulation (i_L=0에서 차단)"), ("diode", "다이오드 (V_f 포함)")],
            source="TEXTBOOK",
            source_note="교재: diode-emulation DCM와 강제 동기 음전류를 구분",
            group="정류",
        ),
    ]


def _buck_from_values(v: dict, D=None) -> Buck:
    D = v["D"] if D is None else D
    p = dict(v)
    p["D"] = D
    p["R"] = v["D"] * v["Vin"] / v["Io"]
    return Buck(p)


def buck_steady(sys: Buck):
    """Periodic orbit by Newton shooting (includes DCM event-time sensitivity)."""
    Vo = sys.D * sys.Vin
    guess = [Vo / sys.R, Vo]
    sol = shoot(sys, "H", guess, sys.T, scales=[max(Vo / sys.R, 0.05), max(Vo, 1.0)], tol=1e-11)
    return sol


def _buck_circuit(rect: str, v: dict) -> Circuit:
    c = Circuit("buck", 700, 300, title="Buck (동기/다이오드 정류)")
    vin = c.add("vsource", "Vin", 60, 150, 90, "V_in", f"{v['Vin']:g} V")
    q1 = c.add("nmos", "Q1", 200, 60, 0, "Q1 (상측)", lpos=(200, 34, "middle"))
    if rect == "diode":
        q2 = c.add("diode", "Q2", 300, 150, 270, "D2", lpos=(276, 154, "end"))
        lo_top, lo_bot = q2["b"], q2["a"]
    else:
        q2 = c.add("nmos", "Q2", 300, 150, 90, "Q2 (하측)", lpos=(262, 128, "end"))
        lo_top, lo_bot = q2["a"], q2["b"]
    lL = c.add("inductor", "L", 390, 60, 0, "L", f"{v['L'] * 1e6:g} µH")
    esr = c.add("resistor", "ESR", 480, 120, 90, "ESR", f"{v['esr'] * 1e3:g} mΩ")
    cc = c.add("capacitor", "C", 480, 200, 90, "C", f"{v['C'] * 1e6:g} µF")
    rl = c.add("resistor", "R", 590, 150, 90, "R_load", f"{v['D'] * v['Vin'] / v['Io']:.4g} Ω")
    c.add("ground", "gnd", 300, 250)
    c.wire("w_in_top", vin["a"], (60, 60), q1["a"])
    c.wire("w_q1_sw", q1["b"], (300, 60))
    c.wire("w_sw_lo", (300, 60), lo_top)
    c.wire("w_sw_L", (300, 60), lL["a"])
    c.wire("w_L_out", lL["b"], (480, 60))
    c.wire("w_out_c", (480, 60), esr["a"])
    c.wire("w_esr_c", esr["b"], cc["a"])
    c.wire("w_out_r", (480, 60), (590, 60), rl["a"])
    c.wire("w_c_gnd", cc["b"], (480, 250))
    c.wire("w_r_gnd", rl["b"], (590, 250), (480, 250))
    c.wire("w_gnd_mid", (480, 250), (300, 250))
    c.wire("w_lo_gnd", lo_bot, (300, 250))
    c.wire("w_gnd_in", (300, 250), (60, 250), vin["b"])
    c.dot((300, 60), (480, 60), (480, 250), (300, 250))
    c.text(300, 44, "v_sw", "node")
    c.text(480, 44, "v_o", "node")
    c.probe("pL", "iL", 442, 48, "right", "i_L")
    c.probe("pHi", "i_hi", 112, 48, "right", "i_Q1")
    c.probe("pLo", "i_lo", 300, 222, "up", "i_Q2")
    c.probe("pO", "io", 540, 48, "right", "i_o")
    ret = ["w_c_gnd", "w_r_gnd", "w_gnd_mid"]
    load = ["w_L_out", "w_out_c", "w_esr_c", "w_out_r", "ESR", "C", "R", "L"]
    c.mode("H", "Q1 ON", ["Vin", "Q1", "w_in_top", "w_q1_sw", "w_sw_L", "w_gnd_in"] + load + ret, "v_sw = V_in → v_L = V_in − v_o > 0 : i_L 증가", dim=["Q2"])
    c.mode("L", "Q1 OFF · 하측 도통", ["Q2", "w_sw_lo", "w_sw_L", "w_lo_gnd"] + load + ret, "i_L은 하측 소자로 환류, v_L = −v_o < 0 : i_L 감소", dim=["Q1", "Vin"])
    c.mode("Z", "DCM (둘 다 OFF)", ["C", "ESR", "R", "w_out_c", "w_esr_c", "w_out_r"] + ret, "i_L = 0 유지, 부하는 C가 공급. 이상모델의 v_sw = v_o (실제는 L–Coss 링잉)", dim=["Q1", "Q2", "L", "Vin"])
    return c


BUCK_LABELS = {"H": "Q1 ON", "L": "하측 도통", "Z": "DCM idle"}


def _buck_metrics_block(res: Result, sys: Buck, traj, t0, t1, prefix="", with_ref=True, ideal=True):
    T = t1 - t0
    IL = traj.mean(t0, t1, "iL")
    Vo = traj.mean(t0, t1, "vo")
    Io = traj.mean(t0, t1, "io")
    lo, hi = traj.extrema(t0, t1, "iL")
    irms = traj.rms(t0, t1, "iL")
    vlo, vhi = traj.extrema(t0, t1, "vo")
    ihi_avg = traj.mean(t0, t1, "i_hi")
    ihi_rms = traj.rms(t0, t1, "i_hi")
    ilo_avg = traj.mean(t0, t1, "i_lo")
    ilo_rms = traj.rms(t0, t1, "i_lo")
    out = {"IL": IL, "Vo": Vo, "Io": Io, "lo": lo, "hi": hi, "irms": irms, "dI": hi - lo, "dvo": vhi - vlo}
    del T
    tol_note = "" if ideal else "비이상 요소 또는 DCM이라 이상 CCM 손계산과 다를 수 있다 (판정 없이 표시)"
    r = ref.buck_ccm(sys.Vin, sys.D * sys.Vin, sys.L, sys.fs, sys.D * sys.Vin / sys.R) if with_ref else None
    chk = (lambda tol: tol) if ideal else (lambda tol: None)
    res.add_metric(prefix + "Vo", "출력전압 평균 V_o", Vo, "V", ref=sys.D * sys.Vin if with_ref else None, ref_label="손계산 D·V_in", tol=chk(0.01), basis="DC 평균, 1주기", note=tol_note)
    res.add_metric(prefix + "IL_avg", "인덕터 평균전류", IL, "A", ref=r["I_peak"] - r["dI_pp"] / 2 if r else None, ref_label="손계산 I_o", tol=chk(0.01), basis="DC 평균")
    res.add_metric(prefix + "dI_pp", "인덕터 리플 ΔI_pp", hi - lo, "A", ref=r["dI_pp"] if r else None, ref_label="(V_in−V_o)D/(L f_s)", tol=chk(0.05), basis="peak-to-peak")
    res.add_metric(prefix + "I_peak", "인덕터 peak", hi, "A", ref=r["I_peak"] if r else None, ref_label="I_o + ΔI/2", tol=chk(0.02), basis="peak (포화·OCP 기준)")
    res.add_metric(prefix + "I_valley", "인덕터 valley", lo, "A", ref=r["I_valley"] if r else None, ref_label="I_o − ΔI/2", tol=chk(0.02), basis="valley")
    res.add_metric(prefix + "IL_rms", "인덕터 RMS", irms, "A", ref=r["I_rms"] if r else None, ref_label="√(I_o² + ΔI²/12)", tol=chk(0.02), basis="RMS (발열 기준)")
    res.add_metric(prefix + "Q1_avg", "상측 스위치 평균전류 (= 입력 DC 전류)", ihi_avg, "A", basis="소자별, ON 구간만 적분")
    res.add_metric(prefix + "Q1_rms", "상측 스위치 RMS", ihi_rms, "A", basis="소자별 RMS ≠ 인덕터 RMS")
    res.add_metric(prefix + "Q2_avg", "하측 소자 평균전류", ilo_avg, "A", basis="소자별")
    res.add_metric(prefix + "Q2_rms", "하측 소자 RMS", ilo_rms, "A", basis="소자별")
    res.add_metric(prefix + "dvo_pp", "출력전압 리플 Δv_o (pp)", vhi - vlo, "V", basis="C와 ESR 성분을 더한 실제 파형의 pp")
    return out


# ------------------------------------------------------------------------------------
def run_buck_ccm(v: dict) -> Result:
    res = Result("FL01", "buck_ccm", "C (+B 평균모델 비교)")
    sys = _buck_from_values(v)
    T = sys.T
    ideal = v["dcr"] == 0 and v["rds_hi"] == 0 and v["rds_lo"] == 0 and v["rect"] != "diode"
    sol = buck_steady(sys)
    if not sol.converged:
        res.verdict("SOLVER_FAILED", f"shooting이 수렴하지 않음 (잔차 {sol.residual:.2e})")
        return res
    # steady window: two periods from the periodic state
    traj = simulate(sys, "H", sol.x0, 0.0, 2 * T)
    dcm = any(s.q == "Z" for s in traj.segments)
    m = _buck_metrics_block(res, sys, traj, 0.0, T, ideal=ideal and not dcm)
    if dcm:
        res.warnings.append("DCM 운전점: CCM 손계산(ΔI·peak·RMS 식)은 적용되지 않으므로 비교 판정을 끄고 값만 표시한다.")
    ccm_boundary = ref.buck_ccm(sys.Vin, sys.D * sys.Vin, sys.L, sys.fs, 1.0)["I_boundary"]
    res.add_metric("I_boundary", "CCM/DCM 경계 부하전류 (ΔI/2)", ccm_boundary, "A", ref=0.45 if abs(sys.Vin - 48) < 1e-9 and abs(sys.L - 100e-6) < 1e-12 and abs(sys.fs - 1e5) < 1e-6 and abs(sys.D - 0.25) < 1e-12 else None, ref_label="교재 0.45 A", tol=1e-9, basis="diode-emulation 기준")
    res.add_metric("mode", "전도 모드", "DCM" if dcm else "CCM", "", basis=f"정류: {v['rect']}")
    eff = None
    led = energy_ledger(traj, sys, 0.0, T, ["p_in"], ["p_out"], ["p_loss"], rated_power=max(m["Vo"] * m["Io"], 1e-6))
    if led["E_in"] > 0:
        eff = led["E_out"] / led["E_in"]
        res.add_metric("eff", "효율 (포트 에너지 기준)", eff, "", basis="E_out / E_in, 1주기; 손실은 전력단 방정식에 결합된 저항·V_f 손실만")
    res.add_check(ledger_check(led))
    res.add_check(
        Check(
            "shooting 주기해 잔차",
            "PASS" if sol.residual < 1e-9 else "FAIL",
            sol.residual,
            "rel",
            1e-9,
            path="Poincaré map 고정점 (Newton, 유한차분 Jacobian에 DCM 이벤트 시각 민감도 포함)",
            detail=f"반복 {sol.iterations}회, Floquet 승수 |λ|max = {sol.spectral_radius:.6f}",
        )
    )
    # independent solver path: hand-written ODE + RK45 with tightening tolerance
    sched = [(0.0, sys.D * T, "H"), (sys.D * T, T, "L")]
    ev_fn = None
    if v["rect"] in ("diode_emulation", "diode"):

        def ev_fn(mode):
            if mode == "L":
                return (lambda t, x: x[0]), "Z"
            return None

    ends = independent_ivp_check(sys.rhs_plain, sched, sol.x0, event_fn=ev_fn)
    exact_end = traj.state_at(T)[0][:2]
    errs = [float(np.max(np.abs(e - exact_end) / np.array([max(abs(m["IL"]), 0.1), max(abs(m["Vo"]), 1)]))) for e in ends]
    res.add_check(
        Check(
            "독립 solver (RK45) 허용오차 수렴",
            "PASS" if errs[-1] < 1e-6 and errs[-1] <= errs[0] + 1e-12 else "FAIL",
            errs[-1],
            "rel",
            1e-6,
            path="손으로 쓴 ODE(행렬 미사용)를 RK45로 rtol 1e-6→1e-8→1e-10 적분한 주기 끝 상태 vs 정확 해",
            independent=True,
            detail="상대오차: " + ", ".join(f"{e:.2e}" for e in errs),
        )
    )
    # averaged model (B) - state-space average of the two CCM modes
    Aav = sys.D * sys.mode("H").A + (1 - sys.D) * sys.mode("L").A
    bav = sys.D * sys.mode("H").b + (1 - sys.D) * sys.mode("L").b
    try:
        xav = np.linalg.solve(Aav, -bav)
        res.add_metric("IL_avg_model", "평균모델(B) i_L 평형점", xav[0], "A", ref=m["IL"], ref_label="스위칭 해 평균", tol=0.01 if not dcm else None, basis="평균모델은 리플·peak를 계산하지 않는다", note="DCM에서는 CCM 평균모델이 틀린다" if dcm else "")
    except np.linalg.LinAlgError:
        xav = None
    # start-up transient with per-cycle steady-state criterion
    E_ref = max(m["Vo"] * m["Io"] * T, 0.01 * m["Vo"] * m["Io"] * T, 1e-12)
    hist, kept = cycle_to_steady(sys, "H", [0.0, 0.0], T, scales=[max(abs(m["IL"]), 0.1), max(m["Vo"], 1)], energy_ref=E_ref, tol_state=1e-6, tol_energy=1e-6, max_cycles=int(v.get("max_cycles", 4000)), keep_every=0)
    ncyc = hist.cycles_run
    t_su = min(ncyc, 400) * T
    tr_su = simulate(sys, "H", [0.0, 0.0], 0.0, t_su)
    smp = tr_su.sample(["iL", "vo"], per_segment=6)
    xs, ys = decimate_minmax(smp["t"], smp["iL"], 1500)
    res.add_series("su_iL", "i_L (스위칭, C)", "A", xs, ys)
    xs, ys = decimate_minmax(smp["t"], smp["vo"], 1500)
    res.add_series("su_vo", "v_o (스위칭, C)", "V", xs, ys)
    if xav is not None:
        avg_sys = _AveragedLTI(Aav, bav, sys.key_avg if hasattr(sys, "key_avg") else "buckavg")
        tr_av = simulate(avg_sys, None, [0.0, 0.0], 0.0, t_su)
        sa = tr_av.sample(["x0", "x1"], per_segment=200)
        res.add_series("su_iL_avg", "i_L 평균모델 (B)", "A", sa["t"], sa["x0"], dash=True)
        k, Rc = sys.k, sys.Rc
        vo_av = [k * (x1 + Rc * x0) for x0, x1 in zip(sa["x0"], sa["x1"])]
        res.add_series("su_vo_avg", "v_o 평균모델 (B)", "V", sa["t"], vo_av, dash=True)
    res.add_series("crit_x", "주기별 상태 변화 (정규화)", "", list(range(1, ncyc + 1)), hist.state_change)
    res.add_series("crit_w", "주기별 저장에너지 변화 / E_ref", "", list(range(1, ncyc + 1)), [max(e, 1e-18) for e in hist.energy_change])
    steady_txt = f"{hist.steady_cycle}주기에서 기준 충족" if hist.steady_cycle else f"{ncyc}주기 안에 기준 미충족 → 정상상태 수치는 shooting 해 사용"
    res.add_check(
        Check(
            "주기별 정상상태 기준 (상태변화 < 1e-6, |ΔW|/E_ref < 1e-6, 3주기 연속)",
            "INFO",
            hist.steady_cycle or ncyc,
            "주기",
            path="0 초기조건에서 반복 주기 적분",
            detail=steady_txt + ". 마지막 몇 주기를 잘라 쓰지 않고 기준으로 판정한다.",
        )
    )
    # steady waveforms
    names = {"iL": ("i_L", "A"), "vsw": ("v_sw", "V"), "vo": ("v_o", "V"), "i_hi": ("i_Q1", "A"), "i_lo": ("i_Q2", "A"), "io": ("i_o", "A"), "vL": ("v_L", "V")}
    smp = traj.sample(list(names), 0.0, 2 * T, per_segment=24)
    for key, (lab, unit) in names.items():
        res.add_series(key, lab, unit, smp["t"], smp[key])
    bands = bands_from_traj(traj, 0.0, 2 * T, BUCK_LABELS)
    res.add_plot("p_iL", "인덕터 전류와 스위치 전류 (정상상태 2주기)", ["iL", "i_hi", "i_lo"], y_label="전류", y_unit="A", bands=bands, window=(0.0, T), group="ss", level="C",
                 hlines=[{"y": m["IL"], "label": "평균"}],
                 proved="정확 스위칭 해에서 삼각 리플·peak·valley·RMS가 손계산과 일치하고, 소자별 전류(상측/하측)가 인덕터 전류의 해당 구간이라는 것을 보였다.",
                 not_yet="스위칭 순간의 dv/dt·링잉·dead time·Coss 전하는 이상 스위치 모델 밖이다(EX02). 실제 소자 손실·온도는 계산하지 않았다.")
    res.add_plot("p_v", "스위치 노드와 인덕터 전압", ["vsw", "vL"], y_label="전압", y_unit="V", bands=bands, window=(0.0, T), group="ss", level="C",
                 proved="두 상태의 인덕터 전압(V_in−V_o, −V_o)과 그 주기 평균 0(volt-second balance)을 파형으로 확인했다.",
                 not_yet="DCM idle 구간의 스위치 노드 링잉은 계산하지 않는다(이상 모델은 v_sw = v_o).")
    res.add_plot("p_vo", "출력전압 리플", ["vo"], y_label="v_o", y_unit="V", bands=bands, window=(0.0, T), group="ss", level="C",
                 proved="C와 ESR을 포함한 출력전압 리플의 실제 파형 pp를 계산했다.", not_yet="커패시터 ESL·온도·수명은 포함하지 않았다.")
    su_series = ["su_vo", "su_vo_avg"] if xav is not None else ["su_vo"]
    res.add_plot("p_su_v", "기동 과도: 스위칭(C) vs 평균모델(B)", su_series, y_label="v_o", y_unit="V", level="B+C",
                 proved="평균모델은 기동 과도의 평균 궤적(LC 공진·감쇠)을 맞히고, 스위칭 모델은 그 위의 리플을 보여준다.",
                 not_yet="0 V 출력에 이상 전압원을 바로 인가한 수학적 기동이다. 실제 soft-start·전류제한·precharge는 모델에 없다.")
    su_i = ["su_iL", "su_iL_avg"] if xav is not None else ["su_iL"]
    res.add_plot("p_su_i", "기동 과도 인덕터 전류", su_i, y_label="i_L", y_unit="A", level="B+C",
                 proved="기동 중 인덕터 전류 overshoot(정상 peak보다 큼)를 볼 수 있다.", not_yet="포화·OCP 동작은 모델에 없다.")
    res.add_plot("p_crit", "정상상태 판정 기준의 주기별 변화", ["crit_x", "crit_w"], x_label="주기", x_unit="", y_label="정규화 변화", y_unit="", log_y=True, kind="xy",
                 hlines=[{"y": 1e-6, "label": "기준 1e-6"}],
                 proved="정상상태를 '마지막 몇 주기'가 아니라 주기별 상태·에너지 변화 기준으로 판정했고, shooting 해와 대조했다.",
                 not_yet="기준값 1e-6은 학습용 선택이다.")
    res.circuit = {"diagram": _buck_circuit(v["rect"], v).to_json(), "intervals": bands, "plot_group": "ss"}
    res.tables.append(
        Table(
            "t_loss",
            "에너지 원장 (1주기, J)",
            ["항목", "값 [J]", "설명"],
            [
                ["E_in", led["E_in"], "V_in × i_Q1 적분"],
                ["E_out", led["E_out"], "v_o²/R 적분"],
                ["E_loss", led["E_loss"], "DCR·ESR·R_DS(on)·V_f 손실 (전력단에 결합된 손실)"],
                ["ΔW", led["dW"], "½Li² + ½Cv² 변화 (주기해이면 ≈ 0)"],
                ["잔차", led["residual"], "E_in − E_out − E_loss − ΔW"],
            ],
            note="손실은 회로 방정식 안의 저항·V_f에서만 나온다. 스위칭 손실(E_on/E_off)은 이 모델에 없다.",
        )
    )
    if dcm:
        res.verdict("INFO", "이 조건은 DCM이다. CCM 손계산 기준과의 비교는 참고용이며 DCM 실험(경부하)을 보라.")
    else:
        res.verdict("PASS_WITHIN_MODEL", "CCM 정상상태 스위칭 해가 손계산·독립 solver·에너지 보존과 일치 (이상 스위치 모델 범위)")
    res.assumptions += [
        "이상 스위치(즉시 전환, dead time 없음), 선형 L·C, 저항부하 R = D·V_in/I_o",
        "입력 V_in은 이상 전압원(소스 임피던스 없음)",
        "ESR은 C 가지에 직렬; DCR·R_DS(on)는 입력값(기본 0)",
    ]
    res.not_valid_for += ["ZVS/스위칭 손실·dv/dt·링잉 (EX02, FL02)", "코어 포화·온도 (FL07, FL03)", "폐루프 응답 (FL06)"]
    res.interpretation = (
        f"ON 구간에 v_L = V_in − V_o ≈ {sys.Vin - m['Vo']:.3g} V, OFF 구간에 −V_o가 걸려 i_L은 평균 {m['IL']:.4g} A 위에서 "
        f"{m['dI']:.4g} App로 오르내린다. 평균은 부하가, 리플은 L·f_s·전압이 정한다. 스위치 전류는 인덕터 전류의 해당 구간만이므로 "
        "상측 평균전류가 입력 DC 전류(= D·I_L)이고, 소자 RMS는 인덕터 RMS와 다르다."
    )
    return res


class _AveragedLTI(HybridSystem):
    state_names = ("x0", "x1")

    def __init__(self, A, b, key):
        self._m = AffineMode(f"avg|{key}|{np.round(A, 12).tobytes().hex()[:40]}|{np.round(b, 12).tobytes().hex()[:24]}", A, b)

    def mode(self, q):
        return self._m

    def outputs(self, q):
        n = self._m.n
        out = {}
        for i in range(n):
            c = np.zeros(n + 1)
            c[i] = 1.0
            out[f"x{i}"] = c
        return out


# ------------------------------------------------------------------------------------
def run_buck_light_load(v: dict) -> Result:
    res = Result("FL01", "buck_light_load", "C (+A DCM 해석식)")
    T = 1 / v["fs"]
    out = {}
    trajs = {}
    for rect in ("diode_emulation", "sync"):
        vv = dict(v)
        vv["rect"] = rect
        sys = _buck_from_values(vv)
        sol = buck_steady(sys)
        if not sol.converged:
            res.verdict("SOLVER_FAILED", f"{rect}: shooting 미수렴")
            return res
        tr = simulate(sys, "H", sol.x0, 0.0, 2 * T)
        trajs[rect] = (sys, tr)
        lo, hi = tr.extrema(0, T, "iL")
        out[rect] = {"Vo": tr.mean(0, T, "vo"), "IL": tr.mean(0, T, "iL"), "lo": lo, "hi": hi, "rms": tr.rms(0, T, "iL"), "idle": sum(s.h for s in tr.segments if s.q == "Z" and s.t0 < T) / T}
    sysd = trajs["diode_emulation"][0]
    M_an = ref.buck_dcm_ratio(sysd.D, sysd.L, sysd.R, sysd.fs)
    dcm = out["diode_emulation"]["idle"] > 0
    res.add_metric("Vo_de", "V_o (diode-emulation)", out["diode_emulation"]["Vo"], "V", ref=(M_an * sysd.Vin) if dcm else sysd.D * sysd.Vin, ref_label="DCM 해석식 M·V_in" if dcm else "CCM D·V_in", tol=0.01, basis="DC 평균")
    res.add_metric("Vo_sync", "V_o (강제 동기)", out["sync"]["Vo"], "V", ref=sysd.D * sysd.Vin, ref_label="D·V_in (CCM 유지)", tol=0.01, basis="DC 평균")
    res.add_metric("iL_min_sync", "i_L 최소값 (강제 동기)", out["sync"]["lo"], "A", basis="음수 = 음전류(출력→입력 방향)")
    res.add_metric("iL_min_de", "i_L 최소값 (diode-emulation)", out["diode_emulation"]["lo"], "A", basis="0에서 멈춤")
    res.add_metric("idle", "DCM idle 구간 비율 (diode-emulation)", out["diode_emulation"]["idle"], "", basis="i_L = 0 구간 / T_s")
    res.add_metric("rms_de", "i_L RMS (diode-emulation)", out["diode_emulation"]["rms"], "A")
    res.add_metric("rms_sync", "i_L RMS (강제 동기)", out["sync"]["rms"], "A", note="같은 부하에서 음전류 때문에 RMS가 더 크다")
    if dcm:
        D_req = ref.buck_dcm_duty_for(sysd.D, sysd.L, sysd.R, sysd.fs)
        vv = dict(v)
        vv["rect"] = "diode_emulation"
        s2 = _buck_from_values(vv, D=D_req)
        s2.R = sysd.R
        s2.k = s2.R / (s2.R + s2.Rc)
        sol2 = buck_steady(s2)
        tr2 = simulate(s2, "H", sol2.x0, 0.0, T)
        res.add_metric("D_req", f"{sysd.D * sysd.Vin:.4g} V 유지에 필요한 duty (DCM 해석식)", D_req, "", basis="같은 R에서; 폐루프가 찾아야 할 값")
        res.add_metric("Vo_at_Dreq", "그 duty로 스위칭 시뮬레이션한 V_o", tr2.mean(0, T, "vo"), "V", ref=sysd.D * sysd.Vin, ref_label="목표", tol=0.01)
    # load sweep (switching simulation at each point) vs analytic
    Io_list = np.geomspace(0.05, 6.0, 14)
    rows = []
    vo_de, vo_sy, vo_an = [], [], []
    for Io in Io_list:
        vv = dict(v)
        vv["Io"] = float(Io)
        vv["rect"] = "diode_emulation"
        s = _buck_from_values(vv)
        so = buck_steady(s)
        tr = simulate(s, "H", so.x0, 0.0, T)
        vde = tr.mean(0, T, "vo")
        Man = ref.buck_dcm_ratio(s.D, s.L, s.R, s.fs)
        ccm = Io >= ref.buck_ccm(s.Vin, s.D * s.Vin, s.L, s.fs, 1.0)["I_boundary"]
        van = (s.D if ccm else Man) * s.Vin
        vo_de.append(vde)
        vo_an.append(van)
        vo_sy.append(s.D * s.Vin)
        rows.append([float(Io), vde, van, "CCM" if ccm else "DCM"])
    res.add_series("sw_de", "diode-emulation (스위칭 시뮬레이션)", "V", Io_list.tolist(), vo_de, style="points")
    res.add_series("sw_an", "해석식 (CCM D·V_in / DCM M·V_in)", "V", Io_list.tolist(), vo_an, dash=True)
    res.add_series("sw_sy", "강제 동기 (CCM 유지)", "V", Io_list.tolist(), vo_sy)
    err = max(abs(a - b) / b for a, b in zip(vo_de, vo_an))
    res.add_check(
        Check(
            "부하 sweep: 스위칭 V_o vs 해석식 (최대 상대오차)",
            "PASS" if err < 0.005 else "FAIL",
            err,
            "rel",
            0.005,
            path="각 부하점의 정확 주기해 vs 교과서 DCM/CCM 변환비 식",
            independent=True,
            detail="ESR·유한 C에 의한 출력 리플 때문에 해석식(평균 전압 가정)과 미세한 차이가 있다",
        )
    )
    Ib = ref.buck_ccm(sysd.Vin, sysd.D * sysd.Vin, sysd.L, sysd.fs, 1.0)["I_boundary"]
    res.add_plot("p_sweep", "부하전류에 따른 V_o (D 고정)", ["sw_de", "sw_an", "sw_sy"], x_label="I_o 설정 (저항 R = D·V_in/I_o)", x_unit="A", y_label="V_o", y_unit="V", kind="xy", log_x=True,
                 vlines=[{"x": Ib, "label": f"경계 {Ib:.3g} A"}],
                 proved="diode-emulation은 경계 아래에서 변환비가 부하에 의존해 V_o가 올라가고, 스위칭 해가 DCM 해석식과 일치한다. 강제 동기정류는 D·V_in을 유지한다.",
                 not_yet="폐루프 duty 조정, 경부하 burst, 스위칭 손실 변화는 포함하지 않았다.")
    for rect, tag in (("diode_emulation", "de"), ("sync", "sy")):
        sys, tr = trajs[rect]
        smp = tr.sample(["iL", "vsw"], 0, 2 * T, per_segment=24)
        res.add_series(f"iL_{tag}", f"i_L ({'diode-emulation' if tag == 'de' else '강제 동기'})", "A", smp["t"], smp["iL"])
        res.add_series(f"vsw_{tag}", f"v_sw ({'diode-emulation' if tag == 'de' else '강제 동기'})", "V", smp["t"], smp["vsw"])
    bands = bands_from_traj(trajs["diode_emulation"][1], 0, 2 * T, BUCK_LABELS)
    res.add_plot("p_il", "경부하 인덕터 전류: DCM vs 음전류", ["iL_de", "iL_sy"], y_label="i_L", y_unit="A", bands=bands, group="ll", level="C", hlines=[{"y": 0.0, "label": "0 A"}],
                 proved="같은 duty에서 diode-emulation은 0에서 멈추는 세 번째 구간이 생기고, 강제 동기는 음전류로 CCM을 유지한다.",
                 not_yet="DCM idle 구간의 L–Coss 링잉과 음전류가 만드는 ZVS 효과는 이 모델(C)에 없다.")
    res.add_plot("p_vsw", "스위치 노드 전압", ["vsw_de", "vsw_sy"], y_label="v_sw", y_unit="V", bands=bands, group="ll", level="C",
                 proved="DCM idle 구간에서 이상모델의 스위치 노드는 v_o에 떠 있다.", not_yet="실제로는 이 구간에 링잉이 생긴다(EX02).")
    res.circuit = {"diagram": _buck_circuit("diode_emulation", v).to_json(), "intervals": bands, "plot_group": "ll"}
    res.tables.append(Table("t_sweep", "부하 sweep (diode-emulation)", ["I_o 설정 [A]", "V_o 스위칭 [V]", "V_o 해석식 [V]", "모드"], rows))
    res.verdict("PASS_WITHIN_MODEL" if dcm else "INFO", "경부하에서 DCM 변환비를 스위칭 해와 해석식이 함께 재현" if dcm else "이 부하는 CCM이다 — I_o를 경계 아래로 줄여 보라")
    res.assumptions += ["이상 스위치; diode-emulation은 i_L = 0에서 이상적으로 차단", "개루프 duty 고정 (폐루프 duty 조정 없음)"]
    res.not_valid_for += ["DCM 링잉·EMI", "경부하 효율의 정확한 값(스위칭 손실 미포함)"]
    res.interpretation = (
        "부하가 ΔI/2 아래로 내려가면 diode-emulation의 인덕터 전류는 0에서 멈추고 스위치 노드가 v_o에 머무는 세 번째 구간이 생긴다. "
        "그 구간이 평균 스위치 노드 전압을 D·V_in보다 높여 같은 duty에서 V_o가 올라간다. 강제 동기정류는 음전류를 허용해 CCM 변환비를 유지하지만 RMS와 전도손실이 커진다."
    )
    return res


# ------------------------------------------------------------------------------------
def run_buck_ripple(v: dict) -> Result:
    res = Result("FL01", "buck_ripple", "C (+A 손계산)")
    sys = _buck_from_values(v)
    T = sys.T
    sol = buck_steady(sys)
    tr = simulate(sys, "H", sol.x0, 0.0, 2 * T)
    lo, hi = tr.extrema(0, T, "iL")
    dI = hi - lo
    r = ref.buck_cap_ripple(ref.buck_ccm(sys.Vin, sys.D * sys.Vin, sys.L, sys.fs, 1.0)["dI_pp"], sys.fs, sys.C, sys.Rc)
    vclo, vchi = tr.extrema(0, T, "vC")
    velo, vehi = tr.extrema(0, T, "vesr")
    volo, vohi = tr.extrema(0, T, "vo")
    res.add_metric("dVc", "용량성 성분 (v_C) pp", vchi - vclo, "V", ref=r["dV_C"], ref_label="ΔI/(8 f_s C)", tol=0.05, basis="pp")
    res.add_metric("dVesr", "ESR 성분 (R_c·i_C) pp", vehi - velo, "V", ref=r["dV_esr"], ref_label="ΔI·ESR", tol=0.05, basis="pp")
    res.add_metric("dVo", "총 출력 리플 v_o pp (파형 합)", vohi - volo, "V", basis="두 성분을 더한 실제 파형의 pp")
    res.add_metric("naive", "단순 합 (위상 무시)", r["naive_sum"], "V", basis="pp를 그대로 더한 값 — 과대평가")
    res.add_metric("ratio", "총 리플 / 단순 합", (vohi - volo) / r["naive_sum"], "")
    res.add_metric("dI", "인덕터 리플 ΔI_pp", dI, "A")
    Vavg = tr.mean(0, T, "vo")
    smp = tr.sample(["vC", "vesr", "vo", "iC"], 0, 2 * T, per_segment=40)
    vc_avg = tr.mean(0, T, "vC")
    res.add_series("vc_ac", "용량성 성분 v_C − 평균", "V", smp["t"], [x - vc_avg for x in smp["vC"]])
    res.add_series("vesr", "ESR 성분 R_c·i_C", "V", smp["t"], smp["vesr"])
    res.add_series("vo_ac", "총 리플 v_o − 평균", "V", smp["t"], [x - Vavg for x in smp["vo"]])
    res.add_series("iC", "커패시터 전류 i_C", "A", smp["t"], smp["iC"])
    bands = bands_from_traj(tr, 0, 2 * T, BUCK_LABELS)
    res.add_plot("p_rip", "출력 리플의 두 성분과 합", ["vc_ac", "vesr", "vo_ac"], y_label="Δv", y_unit="V", bands=bands, group="rp", level="C",
                 proved="용량성 성분(전류의 적분)과 ESR 성분(전류 자체)의 극값 시각이 달라, 파형 합의 pp가 단순 합보다 작다는 것을 정확 해로 보였다.",
                 not_yet="ESL(스위칭 edge의 spike)·커패시터 온도·측정 대역폭은 포함하지 않았다. 측정 리플은 probe 방식에 크게 의존한다.")
    res.add_plot("p_ic", "커패시터 전류", ["iC"], y_label="i_C", y_unit="A", bands=bands, group="rp", level="C",
                 proved="i_C는 인덕터 리플에서 부하전류 변동을 뺀 삼각파다.", not_yet="")
    res.circuit = {"diagram": _buck_circuit(v["rect"], v).to_json(), "intervals": bands, "plot_group": "rp"}
    res.verdict("PASS_WITHIN_MODEL", "리플 성분별 손계산과 정확 해 일치; 총 리플은 파형 합으로 판정")
    res.assumptions += ["ESR은 주파수 무관 상수, ESL 없음", "C는 이상 선형"]
    res.not_valid_for += ["ESL spike·고주파 리플 측정값", "커패시터 수명·발열"]
    res.interpretation = (
        "용량성 리플은 i_C를 적분한 것이라 i_C가 0을 지나는 순간(ON·OFF 구간 중앙)에 극값이 되고, ESR 리플은 i_C 자체라 스위칭 순간에 극값이 된다. "
        "그래서 두 pp를 그대로 더하면 과대평가다. ESR이 크면 ESR 성분이 지배해 리플이 삼각파처럼 보이고, C가 작으면 포물선 성분이 지배한다."
    )
    return res


# ======================================================================================
# Boost
# ======================================================================================


class Boost(HybridSystem):
    """Boost with diode (or ideal SR) and R load; x = [i_L, v_C]; ESR in the C branch."""

    state_names = ("i_L", "v_C")
    state_units = ("A", "V")

    def __init__(self, p: dict, duty_of_cycle):
        self.p = p
        self.Vin, self.L, self.C, self.R = p["Vin"], p["L"], p["C"], p["R"]
        self.Rc, self.RL = p["esr"], p["dcr"]
        self.fs = p["fs"]
        self.T = 1 / p["fs"]
        self.duty_of_cycle = duty_of_cycle
        self.k = self.R / (self.R + self.Rc)
        self._key = f"boost|{sorted(p.items())}"

    def _vo(self, q):
        k = self.k
        if q == "OFF":
            return np.array([k * self.Rc, k, 0.0])
        return np.array([0.0, k, 0.0])

    def mode(self, q):
        L, C, R, k = self.L, self.C, self.R, self.k
        A = np.zeros((2, 2))
        b = np.zeros(2)
        if q == "ON":
            A[0, 0] = -self.RL / L
            b[0] = self.Vin / L
            A[1, 1] = -k / (R * C)
        elif q == "OFF":
            vo = self._vo(q)
            A[0, 0] = (-self.RL - vo[0]) / L
            A[0, 1] = -vo[1] / L
            b[0] = self.Vin / L
            A[1, 0] = k / C
            A[1, 1] = -k / (R * C)
        else:
            A[1, 1] = -k / (R * C)
        return AffineMode(f"{self._key}|{q}", A, b, label=q)

    def guards(self, q):
        if q == "OFF":

            def reset(z):
                z = z.copy()
                z[0] = 0.0
                return z

            return [Guard("i_L=0", np.array([1.0, 0, 0]), -1, lambda q: "Z", reset)]
        return []

    def gate_schedule(self, t0, t1):
        ev = []
        k0 = int(math.floor(t0 / self.T + 1e-9))
        k1 = int(math.ceil(t1 / self.T + 1e-9))
        for k in range(k0, k1 + 1):
            ta = k * self.T
            tb = ta + self.duty_of_cycle(k) * self.T
            if t0 - 1e-15 <= ta < t1:
                ev.append((ta, lambda q: "ON"))
            if t0 - 1e-15 <= tb < t1:
                ev.append((tb, lambda q: "OFF"))
        return ev

    def after_event(self, q, z):
        if q == "OFF" and z[0] <= 1e-12:
            return "Z"
        return q

    def outputs(self, q):
        e_i = np.array([1.0, 0, 0])
        vo = self._vo(q)
        return {
            "iL": e_i,
            "vo": vo,
            "vC": np.array([0.0, 1.0, 0.0]),
            "i_d": e_i if q == "OFF" else np.zeros(3),
            "i_q": e_i if q == "ON" else np.zeros(3),
            "vsw": vo if q == "OFF" else (np.zeros(3) if q == "ON" else np.array([0.0, 0.0, self.Vin])),
        }

    def stored_energy(self):
        return np.diag([self.L, self.C, 0.0])

    def powers(self, q):
        e_i = np.array([1.0, 0, 0])
        vo = self._vo(q)
        iC = np.array([self.k, -self.k / self.R, 0.0]) if q == "OFF" else np.array([0.0, -self.k / self.R, 0.0])
        return {"p_in": sym_linear(e_i, self.Vin), "p_out": np.outer(vo, vo) / self.R, "p_loss": self.RL * np.outer(e_i, e_i) + self.Rc * np.outer(iC, iC)}


def run_boost_rhpz(v: dict) -> Result:
    res = Result("FL01", "boost_rhpz", "A+B+C")
    Vin, L, fs, D0, dD, C = v["Vin"], v["L"], v["fs"], v["D0"], v["dD"], v["C"]
    T = 1 / fs
    Vo_ideal = Vin / (1 - D0)
    R = Vo_ideal**2 / v["P"]
    p = {"Vin": Vin, "L": L, "C": C, "R": R, "esr": v["esr"], "dcr": v["dcr"], "fs": fs}
    k_step = 10
    D1 = D0 + dD
    if not (0.01 <= D1 <= 0.95):
        res.verdict("OUT_OF_VALIDITY", f"step 후 duty {D1:.3f}가 0.01~0.95 밖입니다")
        return res
    sys0 = Boost(p, lambda k: D0)
    guess = [Vo_ideal / ((1 - D0) * R), Vo_ideal]
    sol = shoot(sys0, "ON", guess, T, scales=[max(guess[0], 0.1), Vo_ideal], tol=1e-11)
    tr0 = simulate(sys0, "ON", sol.x0, 0.0, T)
    rb = ref.boost_ccm(Vin, Vo_ideal, L, fs, v["P"])
    lo, hi = tr0.extrema(0, T, "iL")
    ideal = v["esr"] == 0 and v["dcr"] == 0
    tl = 0.01 if ideal else None
    res.add_metric("Vo", "출력전압 평균", tr0.mean(0, T, "vo"), "V", ref=Vo_ideal, ref_label="V_in/(1−D)", tol=tl, basis="DC 평균")
    res.add_metric("Iin", "입력 평균전류 (= i_L 평균)", tr0.mean(0, T, "iL"), "A", ref=rb["Iin"], ref_label="P/(η V_in), η=1", tol=tl)
    res.add_metric("dI", "인덕터 리플 ΔI_pp", hi - lo, "A", ref=rb["dI_pp"], ref_label="V_in D/(L f_s)", tol=0.05 if ideal else None)
    res.add_metric("Iout", "출력(다이오드) 평균전류", tr0.mean(0, T, "i_d"), "A", ref=rb["Iout"], ref_label="P/V_o", tol=tl, basis="다이오드 도통 구간만")
    res.add_metric("Id_rms", "다이오드 RMS", tr0.rms(0, T, "i_d"), "A", basis="인덕터 RMS와 다름")
    res.add_metric("Iq_rms", "스위치 RMS", tr0.rms(0, T, "i_q"), "A")
    fz = ref.boost_ccm(Vin, Vo_ideal, L, fs, v["P"])["f_rhpz"]
    res.add_metric("f_rhpz", "RHP zero f_RHPZ = R(1−D)²/(2πL)", fz, "Hz", ref=12.732e3 if abs(Vin - 200) < 1e-9 and abs(L - 500e-6) < 1e-12 and abs(D0 - 0.5) < 1e-12 and abs(v["P"] - 1000) < 1e-9 else None, ref_label="교재 12.73 kHz", tol=0.001, basis="duty→v_o, 저항부하, 이상 CCM")
    f0 = (1 - D0) / (2 * math.pi * math.sqrt(L * C))
    res.add_metric("f0", "평균모델 LC 공진 (1−D)/(2π√(LC))", f0, "Hz", basis="C는 ASSUMED")
    # duty step: switching model
    n_after = int(v["cycles_after"])
    sys1 = Boost(p, lambda k: D0 if k < k_step else D1)
    tr = simulate(sys1, "ON", sol.x0, 0.0, (k_step + n_after) * T)
    t_step = k_step * T
    cyc_t, cyc_vo, cyc_il = [], [], []
    for kk in range(k_step - 5, k_step + n_after):
        a, b = kk * T, (kk + 1) * T
        cyc_t.append((a + b) / 2 - t_step)
        cyc_vo.append(tr.mean(a, b, "vo"))
        cyc_il.append(tr.mean(a, b, "iL"))
    Vb = tr.mean((k_step - 1) * T, k_step * T, "vo")
    smp = tr.sample(["vo", "iL"], (k_step - 5) * T, (k_step + n_after) * T, per_segment=10)
    tt = [t - t_step for t in smp["t"]]
    xs, ys = decimate_minmax(tt, smp["vo"], 2500)
    res.add_series("vo_sw", "v_o 스위칭 (C)", "V", xs, ys)
    xs, ys = decimate_minmax(tt, smp["iL"], 2500)
    res.add_series("il_sw", "i_L 스위칭 (C)", "A", xs, ys)
    res.add_series("vo_cyc", "v_o 주기평균 (C, 정확 적분)", "V", cyc_t, cyc_vo, style="points")
    res.add_series("il_cyc", "i_L 주기평균 (C)", "A", cyc_t, cyc_il, style="points")
    # averaged model (B): state-space average with d(t)
    sON, sOFF = sys1.mode("ON"), sys1.mode("OFF")

    def avg_mode(d):
        return AffineMode(f"bavg|{sys1._key}|{d}", d * sON.A + (1 - d) * sOFF.A, d * sON.b + (1 - d) * sOFF.b)

    x_av0 = np.linalg.solve(avg_mode(D0).A, -avg_mode(D0).b)

    class Avg(HybridSystem):
        def mode(self, q):
            return avg_mode(D0 if q == 0 else D1)

        def gate_schedule(self, a, b):
            return [(t_step, lambda q: 1)] if a <= t_step < b else []

        def outputs(self, q):
            m = avg_mode(D0 if q == 0 else D1)
            del m
            # averaged output voltage: k (v_C + R_c (1-d) i_L)
            d = D0 if q == 0 else D1
            return {"vo": np.array([sys1.k * sys1.Rc * (1 - d), sys1.k, 0.0]), "iL": np.array([1.0, 0, 0])}

    tra = simulate(Avg(), 0, x_av0, (k_step - 5) * T, (k_step + n_after) * T)
    sa = tra.sample(["vo", "iL"], per_segment=1200)
    res.add_series("vo_avg", "v_o 평균모델 (B)", "V", [t - t_step for t in sa["t"]], sa["vo"], dash=True)
    res.add_series("il_avg", "i_L 평균모델 (B)", "A", [t - t_step for t in sa["t"]], sa["iL"], dash=True)
    post = [(t, x) for t, x in zip(cyc_t, cyc_vo) if t > 0]
    vmin_cyc = min(x for _, x in post)
    t_min = [t for t, x in post if x == vmin_cyc][0]
    vmin_avg = tra.extrema(t_step, (k_step + n_after) * T, "vo", per_segment=4000)[0]
    dip_sw = vmin_cyc - Vb
    dip_av = vmin_avg - (x_av0[1] * sys1.k)
    res.add_metric("dip_sw", "step 직후 최저점 − step 전 (주기평균, C)", dip_sw, "V", basis=f"{t_min * 1e6:.1f} µs 부근 (step 기준)")
    res.add_metric("dip_avg", "step 직후 최저점 − step 전 (평균모델, B)", dip_av, "V")
    Vfin = Vin / (1 - D1)
    res.add_metric("Vo_final_ideal", "새 duty의 이상 정상값 V_in/(1−D₁)", Vfin, "V")
    res.add_check(
        Check(
            "RHP zero 부호: duty 증가 직후 출력 감소",
            "PASS" if dD > 0 and dip_sw < 0 and dip_av < 0 else ("INFO" if dD <= 0 else "FAIL"),
            dip_sw,
            "V",
            path="스위칭 모델 주기평균(C)과 평균모델(B)이 모두 초기 역응답을 보이는지",
            independent=True,
            detail="ΔD>0일 때 두 모델 모두 먼저 떨어져야 한다",
        )
    )
    res.add_plot("p_step", "duty step 응답: 스위칭(C), 주기평균, 평균모델(B)", ["vo_sw", "vo_cyc", "vo_avg"], x_label="step 이후 시간", y_label="v_o", y_unit="V", level="B+C", group="st",
                 vlines=[{"x": 0.0, "label": f"D {D0:.2f}→{D1:.2f}"}],
                 proved="duty를 올린 직후 출력(주기평균)이 먼저 떨어졌다가 오르는 역응답을 스위칭 모델과 평균모델이 함께 보였다. 이것이 duty→v_o 경로의 RHP zero의 시간영역 모습이다.",
                 not_yet="폐루프 제어·전류모드·지연은 없다. 부하가 저항이 아니거나 DCM이면 영점 위치와 응답이 달라진다.")
    res.add_plot("p_step_i", "인덕터 전류는 즉시 증가를 시작한다", ["il_sw", "il_cyc", "il_avg"], x_label="step 이후 시간", y_label="i_L", y_unit="A", level="B+C", group="st", vlines=[{"x": 0.0, "label": "step"}],
                 proved="duty→i_L 경로에는 RHP zero가 없어 전류는 처음부터 올바른 방향으로 움직인다.", not_yet="")
    # Bode (A)
    freqs = np.geomspace(10, 200e3, 300)
    Dd = D0
    gvd = [ref.boost_gvd(2j * math.pi * f, Vin, Dd, L, C, R) for f in freqs]
    gid = [ref.boost_gid(2j * math.pi * f, Vin, Dd, L, C, R) for f in freqs]
    res.add_series("gvd_mag", "|G_vd| (duty→v_o)", "dB", freqs.tolist(), [20 * math.log10(abs(g)) for g in gvd])
    res.add_series("gid_mag", "|G_id| (duty→i_L)", "dB", freqs.tolist(), [20 * math.log10(abs(g)) for g in gid], dash=True)
    ph_v = np.unwrap([math.atan2(g.imag, g.real) for g in gvd]) * 180 / math.pi
    ph_i = np.unwrap([math.atan2(g.imag, g.real) for g in gid]) * 180 / math.pi
    res.add_series("gvd_ph", "∠G_vd", "deg", freqs.tolist(), ph_v.tolist())
    res.add_series("gid_ph", "∠G_id", "deg", freqs.tolist(), ph_i.tolist(), dash=True)
    res.add_plot("p_bode_m", "소신호 전달함수 크기 (A)", ["gvd_mag", "gid_mag"], x_label="f", x_unit="Hz", y_label="크기", y_unit="dB", kind="xy", log_x=True, group="bode",
                 vlines=[{"x": fz, "label": f"f_RHPZ {fz / 1e3:.2f} kHz"}, {"x": f0, "label": f"f_0 {f0:.0f} Hz"}],
                 proved="이상 CCM 평균모델의 소신호 전달함수에서 RHP zero 위치를 계산했다.", not_yet="ESR 영점·손실·지연·샘플링은 포함하지 않았다(FL06).")
    res.add_plot("p_bode_p", "위상: RHP zero는 크기를 올리면서 위상을 더 늦춘다", ["gvd_ph", "gid_ph"], x_label="f", x_unit="Hz", y_label="위상", y_unit="deg", kind="xy", log_x=True, group="bode",
                 vlines=[{"x": fz, "label": "f_RHPZ"}], proved="G_vd는 −270°까지 내려가고(RHP zero), G_id는 LHP zero로 위상이 회복된다.", not_yet="")
    c = Circuit("boost", 640, 280, title="Boost")
    vi = c.add("vsource", "Vin", 60, 140, 90, "V_in", f"{Vin:g} V")
    li = c.add("inductor", "L", 170, 60, 0, "L", f"{L * 1e6:g} µH")
    q = c.add("nmos", "Q", 260, 140, 90, "Q", lpos=(224, 128, "end"))
    d = c.add("diode", "D", 340, 60, 0, "D", lpos=(340, 36, "middle"))
    cc = c.add("capacitor", "C", 440, 140, 90, "C", f"{C * 1e6:g} µF (가정)")
    rr = c.add("resistor", "R", 540, 140, 90, "R", f"{R:.4g} Ω")
    c.add("ground", "gnd", 260, 230)
    c.wire("w1", vi["a"], (60, 60), li["a"])
    c.wire("w2", li["b"], (260, 60))
    c.wire("w3", (260, 60), q["a"])
    c.wire("w4", (260, 60), d["a"])
    c.wire("w5", d["b"], (440, 60), cc["a"])
    c.wire("w6", (440, 60), (540, 60), rr["a"])
    c.wire("w7", q["b"], (260, 230))
    c.wire("w8", cc["b"], (440, 230), (260, 230))
    c.wire("w9", rr["b"], (540, 230), (440, 230))
    c.wire("w10", (260, 230), (60, 230), vi["b"])
    c.dot((260, 60), (440, 60), (260, 230), (440, 230))
    c.probe("pL", "iL", 104, 48, "right", "i_L")
    c.probe("pD", "i_d", 396, 48, "right", "i_D")
    c.probe("pQ", "i_q", 260, 196, "down", "i_Q")
    c.mode("ON", "Q ON", ["Vin", "L", "Q", "w1", "w2", "w3", "w7", "w10", "C", "R", "w6", "w8", "w9"], "L 충전 (v_L = V_in), 출력은 C가 공급", dim=["D"])
    c.mode("OFF", "Q OFF · D 도통", ["Vin", "L", "D", "w1", "w2", "w4", "w5", "w6", "C", "R", "w8", "w9", "w10"], "i_L이 D로 출력에 전달 (v_L = V_in − v_o < 0)", dim=["Q"])
    c.mode("Z", "DCM", ["C", "R", "w6", "w8", "w9"], "i_L = 0", dim=["Q", "D", "L"])
    bands = bands_from_traj(tr, (k_step - 5) * T, (k_step + 3) * T, {"ON": "Q ON", "OFF": "D 도통", "Z": "DCM"}, t_offset=t_step)
    res.plots[0].bands = bands
    res.circuit = {"diagram": c.to_json(), "intervals": bands, "plot_group": "st"}
    res.verdict("PASS_WITHIN_MODEL", "duty→v_o 역응답(RHP zero)을 스위칭·평균·소신호 세 수준에서 일관되게 재현")
    res.assumptions += [f"출력 C = {C * 1e6:g} µF는 교재에 없는 ASSUMED 값 (영점 위치에는 무관, 과도 모양에는 영향)", "저항부하, 이상 다이오드, 개루프 duty step"]
    res.not_valid_for += ["폐루프 안정성 판정 (FL06/EX07)", "DCM·정전력부하에서의 영점 위치"]
    res.interpretation = (
        "duty를 늘리면 다음 주기부터 OFF 구간(출력으로 에너지를 보내는 시간)이 짧아진다. 인덕터 전류가 새 평형으로 오르기 전까지는 출력으로 가는 평균전류 (1−d)·i_L이 줄어 "
        f"v_o가 먼저 떨어진다. 이것이 duty→출력 전달함수의 우반평면 영점이며 이 운전점에서 {fz / 1e3:.2f} kHz다. 전류는 처음부터 올라가므로 duty→i_L 경로에는 이 영점이 없다."
    )
    return res


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q_BUCK = [
    Question(
        "Buck 변환비를 유도하라.",
        "두 상태의 인덕터 전압을 쓴다: ON에서 v_L = V_in − V_o, OFF에서 −V_o. 정상 주기상태에서는 한 주기 적분이 0(volt-second balance)이므로 "
        "D(V_in − V_o) + (1 − D)(−V_o) = 0 → V_o = D·V_in. CCM·이상 소자 가정이다. DCM에서는 세 번째 구간이 생겨 부하가 변환비에 들어간다.",
        "Derive the buck conversion ratio.",
        "Write the inductor voltage in the two switch states, V_in − V_o and −V_o. In periodic steady state the volt-seconds over one period are zero, so D(V_in − V_o) − (1 − D)V_o = 0 and V_o = D·V_in. This assumes CCM and ideal devices.",
        ["두 상태의 v_L", "volt-second balance", "CCM 가정"],
    ),
    Question(
        "RMS·peak·average를 각각 어디에 쓰나?",
        "열(저항성 손실)은 RMS, 포화·OCP·SOA는 peak, 배터리/부하 포트 전류는 DC 평균. 소자 RMS는 해당 도통 구간만 적분한다 — 상측 스위치 RMS는 인덕터 RMS가 아니다.",
        "Where do you use RMS, peak and average current?",
        "RMS for resistive heating, peak for saturation, over-current protection and SOA, and the DC average for the port or battery current. Device RMS integrates only the device's own conduction interval.",
        ["열=RMS", "포화/OCP=peak", "포트=평균", "구간 적분"],
    ),
    Question(
        "왜 V_o = D·V_in을 DCM에도 그대로 쓰면 안 되나?",
        "인덕터 전류 0인 세 번째 구간이 생겨 그 동안 v_L = 0(스위치 노드가 v_o)이다. volt-second를 세 구간으로 다시 평균하면 도통 구간 길이와 부하가 변환비에 들어간다: M = 2/(1+√(1+4K/D²)).",
        "Why can't you use V_o = D·V_in in DCM?",
        "In DCM a third interval appears in which the inductor current is zero and the inductor voltage is zero. Averaging over three intervals puts the load and the conduction time into the conversion ratio.",
        ["세 번째 구간", "부하 의존", "식 암기 대신 재평균"],
    ),
    Question(
        "같은 조건에서 f_s를 50 kHz로 낮추면 무엇이 달라지나?",
        "리플 1.8 App, peak 5.9 A, DCM 경계 0.9 A. 평균이 유지된다는 것은 이상 duty 또는 폐루프 가정이다. 스위칭 손실은 줄 수 있지만 L/C·제어 대역·EMI 스펙트럼이 바뀐다.",
        "What changes if f_s drops to 50 kHz?",
        "Ripple doubles to 1.8 A peak-to-peak, the peak rises to 5.9 A and the DCM boundary moves to 0.9 A. Switching loss may fall, but the passive components, control bandwidth and EMI spectrum change.",
        ["1.8 App", "5.9 A", "0.9 A", "평균 유지 가정"],
        kind="calc",
    ),
    Question(
        "스위치가 꺼졌는데 인덕터 전류는 왜 계속 흐르나?",
        "v = L di/dt: 전류를 즉시 멈추려면 무한 전압이 필요하다. 상측이 꺼지면 스위치 노드 전압이 떨어져 하측 소자가 도통하고, 그 경로로 전류가 계속 흐르면서 −V_o에 의해 감소한다.",
        "Why does the inductor current continue when the switch turns off?",
        "An inductor opposes a change in current, v = L di/dt, so stopping the current instantly would need an infinite voltage. When the high-side switch turns off, the switch node falls until the low-side device conducts, and the current keeps flowing through it while the output voltage ramps it down.",
        ["v = L di/dt", "환류 경로", "−V_o로 감소"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="buck_ccm",
        title="Buck 48→12 V: 두 상태를 평균하고 스위칭 파형으로 확인",
        goal=(
            "ON/OFF 두 상태에서 인덕터 전압이 V_in−V_o와 −V_o로 바뀌어 전류가 삼각파가 되는 것을 확인하고, "
            "손계산(D = 0.25, ΔI = 0.9 App, peak 5.45 A, RMS 5.00675 A)을 정확 스위칭 해·독립 solver·에너지 보존으로 검증한다. "
            "평균모델(B)은 평균과 기동 궤적은 맞히지만 리플은 말하지 않는다는 것도 본다."
        ),
        params=buck_params(),
        presets=[
            Preset("nominal", "교재 기준점 48→12 V, 5 A", {}, "교재 03장 예제", ("nominal", "reference")),
            Preset("half_L", "L = 50 µH", {"L": 50e-6}, "리플 2배 예측 확인", ("variant",)),
            Preset("half_fs", "f_s = 50 kHz", {"fs": 50e3}, "20장 진단문제 2", ("variant",)),
            Preset("lossy", "DCR 30 mΩ, R_DS(on) 20 mΩ", {"dcr": 0.03, "rds_hi": 0.02, "rds_lo": 0.02}, "손실이 전력단에 결합된 경우", ("variant",)),
            Preset("light", "경부하 0.3 A (diode-emulation)", {"Io": 0.3, "rect": "diode_emulation"}, "DCM 진입", ("variant",)),
        ],
        run=run_buck_ccm,
        model_level="C (+B 평균모델, 기동 과도)",
        suggested_change="L을 100 µH → 50 µH로 바꾼다 (다음에는 f_s 100 → 50 kHz).",
        prediction=Prediction(
            "L을 절반으로 하면 인덕터 전류 리플 ΔI_pp와 평균전류는?",
            ["리플 2배 · 평균 유지", "리플 절반 · 평균 유지", "리플 유지 · 평균 2배", "모르겠다"],
            "리플 2배 · 평균 유지",
            "ΔI = (V_in − V_o)·D/(L·f_s)이므로 L에 반비례한다. 평균 인덕터 전류는 부하전류(5 A)가 정한다. 단 peak가 5.9 A로 올라 포화·OCP 여유가 줄고, 경계전류도 0.9 A로 올라간다.",
            ["dI_pp", "IL_avg", "I_peak"],
            handcalc=[{"key": "dI_pp", "label": "리플 ΔI_pp", "unit": "A"}, {"key": "I_peak", "label": "peak 전류", "unit": "A"}, {"key": "IL_rms", "label": "인덕터 RMS", "unit": "A"}],
        ),
        suggested={"L": 50e-6},
        student=(
            "스위치가 켜지면 인덕터 양단에 V_in − V_o = 36 V가 걸려 전류가 0.36 A/µs로 오른다. 꺼져도 인덕터 전류는 즉시 멈출 수 없어 하측 소자로 돌며 "
            "−V_o = −12 V 때문에 0.12 A/µs로 내려간다. 한 주기 뒤 제자리로 오려면 오른 만큼 내려와야 하므로 36·D = 12·(1−D), 즉 D = 0.25다. "
            "평균 인덕터 전류는 부하전류 5 A이고, 그 위에 0.9 App 삼각 리플이 얹혀 peak 5.45 A, valley 4.55 A가 된다."
        ),
        expert=(
            "① 평균모델(B)은 ON/OFF 상태행렬을 D로 가중평균한 것이라 평균 i_L·v_o와 LC 기동 과도는 맞히지만 리플·peak·소자 RMS는 원리적으로 없다. "
            "평균 곡선에 리플을 그려 넣어 스위칭 모델처럼 보이게 하지 않는다. "
            "② 스위칭 해는 각 상태의 선형 상태방정식을 행렬지수로 정확히 풀고 게이트 edge·DCM 진입을 정확한 시각에 처리한 것이라 timestep 오차가 없다. "
            "독립 검증으로 같은 회로를 행렬 없이 손으로 쓴 ODE를 RK45로 적분하고 허용오차를 조이며 수렴을 보인다. "
            "③ 정상상태는 주기별 상태·저장에너지 변화 기준과 shooting(Poincaré map 고정점)으로 판정한다. 0 V에서 시작한 기동은 LC 감쇠 시정수(2RC) 때문에 수백 주기가 걸린다. "
            "④ ESR이 있어도 C의 평균전류가 0이므로 평균 V_o는 D·V_in이다. DCR·R_DS(on)를 넣으면 손실이 전력단 방정식에 들어가 V_o가 낮아지고 에너지 원장에 손실항이 나타난다 — 후처리 손실이 아니다."
        ),
        customer_ko=(
            "이 조건(48→12 V, 5 A, 100 µH, 100 kHz)에서 인덕터는 평균 5 A에 0.9 App 리플로 peak 5.45 A를 봅니다. 포화·OCP는 peak로, 발열은 RMS 5.01 A로, "
            "부하·배터리 전류는 평균으로 보셔야 합니다. 부하가 0.45 A 아래로 내려가면 diode-emulation에서는 DCM으로 들어가니 경부하 제어 동작도 함께 확인하시죠."
        ),
        customer_en=(
            "At 48 V to 12 V, 5 A, 100 µH and 100 kHz, the inductor carries 5 A average with 0.9 A peak-to-peak ripple, so the peak is 5.45 A. "
            "Use the peak for saturation and over-current protection, the RMS of about 5.01 A for heating, and the average for the load current. "
            "Below about 0.45 A the diode-emulation mode enters DCM, so the light-load control behaviour should be checked as well."
        ),
        questions=_Q_BUCK,
        circuit="buck",
        textbook=[TB_02, TB_03],
        reference_presets=["nominal", "half_fs", "lossy"],
        claim_limit="이상 스위치 수준(C)의 파형·RMS·리플. 스위칭 손실·ZVS·링잉·포화는 주장하지 않는다.",
    ),
    Experiment(
        key="buck_light_load",
        title="경부하: diode-emulation DCM vs 강제 동기정류의 음전류",
        goal=(
            "부하가 ΔI/2 = 0.45 A 아래로 내려갈 때, diode-emulation은 전류가 0에서 멈추는 세 번째 구간(DCM)이 생겨 같은 duty에서 V_o가 오르고, "
            "강제 동기정류는 음전류가 흘러 CCM을 유지한다는 것을 비교한다. 두 동작을 같은 경계에서 섞어 말하지 않는다."
        ),
        params=buck_params(),
        presets=[
            Preset("nominal", "I_o = 5 A (CCM 기준)", {}, "두 방식이 같은 기준점", ("nominal",)),
            Preset("light", "I_o = 0.3 A", {"Io": 0.3}, "DCM 영역", ("reference",)),
            Preset("boundary", "I_o = 0.45 A (경계)", {"Io": 0.45}, "경계 부하", ("variant",)),
            Preset("very_light", "I_o = 0.05 A", {"Io": 0.05}, "깊은 DCM", ("corner",)),
        ],
        run=run_buck_light_load,
        model_level="C (+A DCM 해석식)",
        suggested_change="부하전류 I_o를 5 A → 0.3 A로 줄인다 (duty 0.25 고정).",
        prediction=Prediction(
            "I_o = 0.3 A, D = 0.25 고정이면 diode-emulation의 V_o는?",
            ["12 V 유지", "12 V보다 높아진다", "12 V보다 낮아진다", "모르겠다"],
            "12 V보다 높아진다",
            "DCM에서는 인덕터 전류 0 구간 동안 스위치 노드가 v_o에 머물러 평균 스위치 노드 전압이 D·V_in보다 커진다. 변환비가 부하에 의존한다: M = 2/(1+√(1+4K/D²)), K = 2L/(R·T_s).",
            ["Vo_de", "Vo_sync", "iL_min_sync"],
            handcalc=[{"key": "Vo_de", "label": "V_o (diode-emulation)", "unit": "V"}],
        ),
        suggested={"Io": 0.3},
        student=(
            "부하전류가 작아지면 리플의 아래쪽 끝(valley)이 0에 닿는다. 다이오드나 diode-emulation은 전류를 거꾸로 흘리지 않으므로 거기서 멈춰 버리고, 다음 스위칭까지 인덕터 전류가 0인 구간이 생긴다. "
            "동기정류를 계속 켜 두면 전류가 음수로 내려가 출력에서 입력 쪽으로 잠깐 되돌아간다."
        ),
        expert=(
            "diode-emulation DCM에서 idle 구간의 스위치 노드는 이상모델에서 v_o에 떠 있지만 실제로는 L과 노드 커패시턴스(Coss·정류소자 C)가 공진해 링잉이 생긴다 — 이 모델(C)은 그 링잉을 계산하지 않는다(EX02에서 비선형 Coss로 확장). "
            "강제 동기정류는 음전류로 RMS·전도손실이 커지지만 CCM 선형 plant를 유지하고, 음전류가 상측 turn-on 전에 스위치 노드를 끌어올려 상측 ZVS를 도울 수 있다 — ZVS 여부는 전하·dead time으로 따로 판정한다(EX02). "
            "DCM plant는 1차에 가까워져 제어 설계가 달라지므로 경계 부근 모드 전환을 폐루프에서 확인한다."
        ),
        customer_ko=(
            "경부하에서 출력이 목표보다 올라가는 것은 부품 불량이 아니라 DCM 진입으로 변환비가 바뀐 결과일 수 있습니다. 이 조건에서 12 V를 유지하려면 제어기가 duty를 약 0.204까지 줄여야 합니다. "
            "강제 동기정류로 바꾸면 전압은 유지되지만 음전류로 손실이 늘어나니 경부하 효율 요구와 함께 결정하시죠."
        ),
        customer_en=(
            "The output rising at light load is not necessarily a defect: the converter has entered DCM, so the conversion ratio now depends on the load. Here the controller must reduce the duty to about 0.204 to hold 12 V. "
            "Forced synchronous rectification keeps the ratio but circulates negative current, so the light-load efficiency target decides between them."
        ),
        questions=[_Q_BUCK[2]],
        circuit="buck",
        textbook=[TB_03],
        reference_presets=["light"],
        runtime_hint="seconds",
        claim_limit="DCM 변환비와 전류 파형(C). 링잉·경부하 효율 정밀값은 주장하지 않는다.",
    ),
    Experiment(
        key="buck_ripple",
        title="출력 리플: 커패시턴스 성분과 ESR 성분은 위상이 다르다",
        goal=(
            "C = 100 µF, ESR = 10 mΩ에서 용량성 리플 11.25 mVpp와 ESR 리플 9 mVpp를 각각 계산하고, 두 파형을 실제로 더한 총 리플이 단순 합 20.25 mV보다 작은 이유를 파형으로 확인한다."
        ),
        params=buck_params(),
        presets=[
            Preset("nominal", "교재 C 100 µF, ESR 10 mΩ", {}, "교재 03장", ("nominal", "reference")),
            Preset("esr_high", "ESR 50 mΩ", {"esr": 0.05}, "ESR 지배", ("variant",)),
            Preset("c_small", "C 22 µF", {"C": 22e-6}, "용량 지배", ("variant",)),
        ],
        run=run_buck_ripple,
        model_level="C (+A 손계산)",
        suggested_change="ESR을 10 mΩ → 50 mΩ으로 (또는 C를 100 → 22 µF로) 바꾼다.",
        prediction=Prediction(
            "C = 100 µF, ESR = 10 mΩ일 때 총 출력 리플(pp)은?",
            ["20.25 mV (단순 합)", "20.25 mV보다 작다", "20.25 mV보다 크다", "모르겠다"],
            "20.25 mV보다 작다",
            "용량성 성분은 전류의 적분이라 i_C가 0을 지날 때 극값이고, ESR 성분은 전류 자체라 스위칭 순간에 극값이다. 극값 시각이 달라 pp가 그대로 더해지지 않는다.",
            ["dVo", "naive", "ratio"],
            handcalc=[{"key": "dVc", "label": "용량성 리플 pp", "unit": "V"}, {"key": "dVesr", "label": "ESR 리플 pp", "unit": "V"}],
        ),
        suggested={"esr": 0.05},
        student="커패시터 전압 리플은 전하가 들어오고 나가는 양(전류의 면적)으로, ESR 전압은 순간 전류 크기로 생긴다. 둘은 서로 다른 순간에 최대가 된다.",
        expert=(
            "실측 리플은 probe ground lead·대역폭·측정점에 크게 좌우되고 ESL이 스위칭 edge에 spike를 만든다. 사양의 리플이 어떤 대역폭·측정법 기준인지 먼저 맞춘다. "
            "ESR은 주파수·온도 의존이 크므로 데이터시트의 100 kHz 값과 온도 조건을 확인한다(이 실험은 상수 ESR 가정)."
        ),
        customer_ko="총 리플은 두 성분의 pp를 더한 값보다 작습니다(이 조건 약 15 mV 수준). 다만 측정은 대역폭과 probe 방식에 따라 크게 달라지니 사양의 측정 조건부터 맞추시죠.",
        customer_en="The total ripple is smaller than the sum of the two peak-to-peak components because their extremes occur at different times. Measured ripple depends strongly on bandwidth and probing, so let's first align the measurement method in the specification.",
        questions=[
            Question(
                "출력 리플을 11.25 mV + 9 mV = 20.25 mV로 보고해도 되나?",
                "안 된다. 두 성분의 위상이 달라 파형을 더한 뒤 pp를 구해야 한다. 단순 합은 상한일 뿐이다.",
                "Can you report the ripple as 11.25 mV plus 9 mV?",
                "No. The two components peak at different times, so you add the waveforms first and then take the peak-to-peak value; the simple sum is only an upper bound.",
                ["위상 차이", "파형 합 후 pp"],
            )
        ],
        circuit="buck",
        textbook=[TB_03],
        reference_presets=["nominal"],
        claim_limit="상수 ESR·ESL 없음 가정의 리플 파형.",
    ),
    Experiment(
        key="boost_rhpz",
        title="Boost: duty를 늘렸는데 출력이 먼저 떨어진다 (RHP zero)",
        goal=(
            "200→400 V, 1 kW, L = 500 µH boost에서 D를 0.50→0.52로 올린 직후 출력이 잠깐 내려갔다가 오르는 것을 스위칭 모델(C)·평균모델(B)로 보고, "
            "duty→출력 전달함수의 RHP zero f_RHPZ = R(1−D)²/(2πL) = 12.73 kHz(A)와 연결한다. duty→전류 전달함수에는 이 영점이 없다는 것도 확인한다."
        ),
        params=[
            Param("Vin", "입력전압 V_in", "V", 200.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="200 V"),
            Param("P", "출력전력 (저항부하 정의)", "W", 1000.0, "W", vmin=1, vmax=1e6, source="TEXTBOOK", source_note="1 kW → R = V_o²/P = 160 Ω"),
            Param("D0", "초기 duty D", "", 0.5, "", vmin=0.05, vmax=0.9, source="TEXTBOOK", source_note="0.5"),
            Param("dD", "duty step ΔD", "", 0.02, "", vmin=-0.2, vmax=0.2, source="ASSUMED", source_note="교재에 크기 없음"),
            Param("L", "인덕턴스 L", "H", 500e-6, "µH", vmin=1e-6, vmax=0.1, source="TEXTBOOK", source_note="500 µH"),
            Param("fs", "스위칭 주파수 f_s", "Hz", 100e3, "kHz", vmin=1e3, vmax=5e6, source="TEXTBOOK", source_note="100 kHz"),
            Param("C", "출력 커패시턴스 C", "F", 100e-6, "µF", vmin=1e-7, vmax=0.1, source="ASSUMED", source_note="교재에 값 없음. 영점 위치와 무관, 과도 모양에 영향"),
            Param("esr", "ESR", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="비이상"),
            Param("dcr", "인덕터 DCR", "Ω", 0.0, "mΩ", vmin=0, vmax=10, source="ASSUMED", group="비이상"),
            Param("cycles_after", "step 이후 관찰 주기 수", "", 150, "", vmin=20, vmax=2000, kind="int", source="ASSUMED", group="시뮬레이션"),
        ],
        presets=[
            Preset("nominal", "교재 200→400 V 1 kW, ΔD +0.02", {}, "교재 03장", ("nominal", "reference")),
            Preset("light", "250 W (R 640 Ω)", {"P": 250.0}, "f_RHPZ 4배로 이동", ("variant",)),
            Preset("down", "ΔD −0.02", {"dD": -0.02}, "반대 방향 step", ("variant",)),
        ],
        run=run_boost_rhpz,
        model_level="A+B+C",
        suggested_change="duty step ΔD = +0.02로 실행한 뒤, 출력전력을 1 kW → 250 W로 낮춰 f_RHPZ가 어디로 가는지 본다.",
        prediction=Prediction(
            "D를 0.50→0.52로 올린 직후(수십 µs) 출력전압 주기평균은?",
            ["즉시 오른다", "잠깐 떨어진 뒤 오른다", "변화 없다", "모르겠다"],
            "잠깐 떨어진 뒤 오른다",
            "duty가 늘면 출력으로 에너지를 보내는 OFF 시간이 먼저 줄고, 인덕터 전류는 시간이 지나야 늘어난다. 그래서 출력으로 가는 평균전류 (1−d)·i_L이 처음에 감소한다.",
            ["dip_sw", "dip_avg", "f_rhpz"],
            handcalc=[{"key": "f_rhpz", "label": "f_RHPZ", "unit": "Hz"}, {"key": "dI", "label": "인덕터 리플 ΔI_pp", "unit": "A"}],
        ),
        suggested={"P": 250.0},
        student="duty를 늘리면 인덕터를 충전하는 시간이 늘고 출력으로 내보내는 시간은 줄어든다. 인덕터 전류가 커지기 전까지는 출력이 받는 에너지가 줄어 전압이 먼저 내려간다.",
        expert=(
            "RHP zero는 크기를 올리면서 위상을 늦추므로 전압모드 루프의 crossover를 f_RHPZ보다 충분히 낮게 제한한다. f_RHPZ ∝ R(1−D)²/L이라 최대 부하·최대 duty에서 가장 낮다. "
            "전류모드 내부 루프(duty→i_L에는 RHP zero 없음)도 외부 전압 루프에서 이 영점을 없애지는 못한다. DCM에서는 영점이 스위칭 주파수 부근 이상으로 이동한다. "
            "12.73 kHz에서 제어하라는 뜻이 아니다 — 실제 plant·지연·ESR 영점과 함께 설계한다."
        ),
        customer_ko=(
            "전압 루프 대역을 RHP zero(이 조건 12.7 kHz) 가까이 올리면 위상 여유가 급격히 줄어 불안정해질 수 있습니다. 최악 조건(최대 부하·최대 duty)에서 f_RHPZ가 가장 낮으니 "
            "그 점에서 crossover를 정하고, 빠른 응답이 필요하면 전류모드 내부 루프를 검토하시죠."
        ),
        customer_en=(
            "If the voltage-loop bandwidth approaches the right-half-plane zero, about 12.7 kHz here, the phase margin collapses. The zero is lowest at maximum load and maximum duty, so set the crossover there, and consider an inner current loop if you need faster response."
        ),
        questions=[
            Question(
                "Boost에서 duty가 늘 때 출력이 잠깐 내려가는 이유는?",
                "출력 전달 시간(OFF 구간)이 먼저 줄고 인덕터 전류는 나중에 늘기 때문이다. duty→v_o 전달함수의 RHP zero이며 저항부하·이상 CCM에서 f = R(1−D)²/(2πL).",
                "Why does the boost output dip first when the duty increases?",
                "The off-time that delivers energy to the output shrinks immediately, while the inductor current needs time to rise. That is the right-half-plane zero of the duty-to-output transfer function.",
                ["출력 전달시간 감소", "전류 증가 지연", "RHP zero 식"],
            ),
        ],
        circuit="boost",
        textbook=[TB_03],
        reference_presets=["nominal", "light"],
        runtime_hint="seconds",
        claim_limit="개루프 duty step의 역응답과 이상 CCM 소신호 영점. 폐루프 안정성은 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL01",
    title="Buck와 Boost — 컨버터의 문법",
    title_en="Buck and Boost: the grammar of converters",
    track="basic",
    order=1,
    path_note="14일 경로 1일차 (02 기초, 03 Buck/Boost)",
    textbook=[TB_02, TB_03],
    prerequisites=[],
    summary="상태별 식 → 평균 → 정확 스위칭 → DCM → 리플 → RHP zero. 손계산을 세 개의 독립 경로로 확인한다.",
    experiments=EXPERIMENTS,
    minimum_scope="Buck/Boost 상태식·평균·스위칭·DCM; 전류 slope·리플·에너지·RHP 응답 (교재 19장 표)",
    claim_limits=[
        "이상 스위치(C) 파형과 평균모델(B); 스위칭 손실·ZVS·링잉 미포함",
        "RHP 응답은 저항부하·CCM·duty 입력 조건에서만",
        "Boost 출력 C는 교재에 값이 없어 ASSUMED",
    ],
    test_paths=["tests/test_fl01.py"],
)
