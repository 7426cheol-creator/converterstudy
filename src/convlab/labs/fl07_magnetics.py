"""FL07 - Magnetics: integrate the actual winding voltage (textbook ch.10).

Three experiments, each with its model level:
  C  winding flux in a DAB: a T-model (series L split between primary and secondary sides, a
     magnetizing branch L_m, winding resistances) driven by two ideal bridges and solved by the
     exact switched engine.  B(t) = B(0) + int v_w dt / (N A_e) from the actual winding voltage
     is compared with the bus-voltage estimate; the magnetizing current's share of the primary
     RMS is separated from the load/circulating current.  Checked by an independently written
     RK45 formulation, trapezoid integration of the winding voltage, the energy ledger and the
     L_m -> infinity limit (FL08 closed form).
  C  flux walking: an asymmetric bipolar winding drive (800 V x 100 ns extra per cycle), ideal,
     resistance-bounded and blocking-capacitor cases; numerical integration drift and a probe
     offset are separated from a real walk by their signatures.
  A  roles and losses: referral L' = n^2 L, C' = C/n^2, R' = n^2 R verified by simulating the
     actual secondary-side circuit and the primary-referred one; skin depth vs a 1-D diffusion
     finite-difference solution; round-wire and Dowell AC-resistance screens; a Steinmetz / iGSE
     core-loss screen with SYNTHETIC coefficients (precise core loss = MISSING_INPUT); the
     supplier data request list.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv

from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import magnetics as ref
from ._common import bands_from_traj, decimate_minmax, energy_ledger, ledger_check

TB_10 = TextbookRef("자성체-컨버터-전문가로-가는-실제-관문-fl07", "10. 자성체 — 컨버터 전문가로 가는 실제 관문 [FL07]")
TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB [FL08]")

MU0 = 4e-7 * math.pi
TWO_PI = 2.0 * math.pi


def sps_phase(V1: float, V2: float, L: float, fs: float, P: float) -> float:
    """Lossless SPS phase (rad) for power P: P = V1 V2 phi (1 - phi/pi) / (w L); NaN if P > P_max."""
    w = TWO_PI * fs
    disc = math.pi**2 - 4.0 * math.pi * P * w * L / (V1 * V2)
    if disc < 0:
        return float("nan")
    return (math.pi - math.sqrt(disc)) / 2.0


# ======================================================================================
# DAB T-model with a magnetizing branch
# ======================================================================================


class DABT(HybridSystem):
    """Primary bridge -> L1 (primary series + leakage) -> magnetizing node (L_m) -> L2' -> secondary bridge.

    All secondary quantities are primary-referred (L2' = n^2 L2, v2' = n v_L).  x = [i1, i_m],
    i2' = i1 - i_m.  The magnetizing (winding) voltage v_m is the voltage the core integrates:
    v_m = [L2 (v1 - R1 i1)/L1 + R2 i2 + v2] / (1 + L2/L1 + L2/L_m)  (L2 = 0: v_m = v2 + R2 i2).
    q = (s1, s2), bridge voltages v1 = s1 V1, v2 = s2 V2.
    """

    state_names = ("i_1", "i_m")
    state_units = ("A", "A")

    def __init__(self, V1, V2, L1, L2, Lm, R1, R2, fs, phi, N, Ae):
        self.V1, self.V2, self.L1, self.L2, self.Lm = V1, V2, L1, L2, Lm
        self.R1, self.R2, self.fs, self.phi, self.N, self.Ae = R1, R2, fs, phi, N, Ae
        self.T = 1.0 / fs
        self.t_phi = phi / (TWO_PI * fs)
        D = 1.0 + L2 / L1 + L2 / Lm
        self.D = D
        self.a1 = (-L2 * R1 / L1 + R2) / D
        self.am = -R2 / D
        self._key = f"fl07dab|{V1:.10g}|{V2:.10g}|{L1:.10g}|{L2:.10g}|{Lm:.10g}|{R1:.10g}|{R2:.10g}|{fs:.10g}|{phi:.12g}"

    def vm_row(self, q) -> np.ndarray:
        s1, s2 = q
        c = (self.L2 * s1 * self.V1 / self.L1 + s2 * self.V2) / self.D
        return np.array([self.a1, self.am, c])

    def mode(self, q):
        s1, _ = q
        vm = self.vm_row(q)
        A = np.array([[(-self.R1 - vm[0]) / self.L1, -vm[1] / self.L1], [vm[0] / self.Lm, vm[1] / self.Lm]])
        b = np.array([(s1 * self.V1 - vm[2]) / self.L1, vm[2] / self.Lm])
        return AffineMode(f"{self._key}|{q}", A, b, label=self.describe(q))

    def gate_schedule(self, t0, t1):
        ev = []
        h = self.T / 2.0
        k0 = int(math.floor(t0 / h)) - 1
        k1 = int(math.ceil(t1 / h)) + 1
        for k in range(k0, k1 + 1):
            ta = k * h
            tb = k * h + self.t_phi
            s = 1 if k % 2 == 0 else -1
            if t0 - 1e-15 <= ta < t1:
                ev.append((ta, lambda q, s=s: (s, q[1])))
            if t0 - 1e-15 <= tb < t1:
                ev.append((tb, lambda q, s=s: (q[0], s)))
        return sorted(ev, key=lambda e: e[0])

    def outputs(self, q):
        s1, s2 = q
        return {
            "i1": np.array([1.0, 0.0, 0.0]),
            "im": np.array([0.0, 1.0, 0.0]),
            "i2": np.array([1.0, -1.0, 0.0]),
            "v1": np.array([0.0, 0.0, s1 * self.V1]),
            "v2": np.array([0.0, 0.0, s2 * self.V2]),
            "vm": self.vm_row(q),
            "B": np.array([0.0, self.Lm / (self.N * self.Ae), 0.0]),
        }

    def stored_energy(self):
        L1, L2, Lm = self.L1, self.L2, self.Lm
        return np.array([[L1 + L2, -L2, 0.0], [-L2, Lm + L2, 0.0], [0.0, 0.0, 0.0]])

    def powers(self, q):
        s1, s2 = q
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([1.0, -1.0, 0.0])

        def lin(c, k):
            m = np.zeros((3, 3))
            m[:, 2] += 0.5 * k * c
            m[2, :] += 0.5 * k * c
            return m

        return {"p1": lin(e1, s1 * self.V1), "p2": lin(e2, s2 * self.V2), "p_loss": self.R1 * np.outer(e1, e1) + self.R2 * np.outer(e2, e2)}

    def describe(self, q):
        return ("+" if q[0] > 0 else "-") + ("+" if q[1] > 0 else "-")

    def rhs_plain(self, t, y, q):
        """Independent formulation: states (i1, i2'), the three unknowns solved as a linear system."""
        s1, s2 = q
        i1, i2 = y
        M = np.array([[self.L1, 0.0, 1.0], [0.0, self.L2, -1.0], [self.Lm, -self.Lm, -1.0]])
        rhs = np.array([s1 * self.V1 - self.R1 * i1, -s2 * self.V2 - self.R2 * i2, 0.0])
        d1, d2, _ = np.linalg.solve(M, rhs)
        return [d1, d2]


DAB_LABELS = {"+-": "① v₁+ · v₂′−", "++": "② v₁+ · v₂′+", "-+": "③ v₁− · v₂′+", "--": "④ v₁− · v₂′−"}


def dab_steady(sysm: DABT):
    """Zero-DC periodic orbit from half-wave antisymmetry x(T/2) = -x(0) (textbook FL08 reference solution).

    The half-period map is affine, x(T/2) = H x0 + g, so (I + H) x0 = -g; this stays well conditioned
    even without damping (where the full-period map has a multiplier at 1 and the DC offset is free).
    Returns x0, the full-period monodromy M = H^2 (for the multipliers) and the initial mode.
    """
    q0 = (1, -1)

    def half(x):
        return simulate(sysm, q0, x, 0.0, sysm.T / 2).z_end[:2].copy()

    g = half([0.0, 0.0])
    H = np.column_stack([half([1.0, 0.0]) - g, half([0.0, 1.0]) - g])
    x0 = np.linalg.solve(np.eye(2) + H, -g)
    return x0, H @ H, q0


def dab_from(v: dict, Lm: float | None = None, split: float | None = None) -> DABT:
    n = v["Np"] / v["Ns"]
    V2 = n * v["VL"]
    k = v["k_split"] if split is None else split
    phi = sps_phase(v["V1"], V2, v["L_tot"], v["fs"], v["P"])
    return DABT(v["V1"], V2, k * v["L_tot"], (1.0 - k) * v["L_tot"], v["Lm"] if Lm is None else Lm, v["R1"], v["R2"], v["fs"], phi, v["Np"], v["Ae"] * 1e-6)


def dab_circuit(v: dict, sysm: DABT) -> Circuit:
    c = Circuit("fl07_dabT", 700, 280, title="DAB T-모델 (2차는 1차 환산: n = N_p/N_s)")
    c.add("vsource", "V1", 70, 150, 90, "v₁ (1차 bridge)", f"±{sysm.V1:g} V", lpos=(92, 146, "start"))
    l1 = c.add("inductor", "L1", 190, 60, 0, "L₁ (1차측 직렬+누설)", f"{sysm.L1 * 1e6:.4g} µH", lpos=(190, 34, "middle"))
    lm = c.add("inductor", "Lm", 330, 150, 90, "L_m (자화)", f"{sysm.Lm * 1e3:.4g} mH", lpos=(352, 146, "start"))
    l2 = c.add("inductor", "L2", 470, 60, 0, "L₂′ (2차측, 환산)", f"{sysm.L2 * 1e6:.4g} µH" if sysm.L2 > 0 else "0 (k = 1: 없음)", lpos=(470, 34, "middle"))
    c.add("vsource", "V2", 610, 150, 90, "v₂′ = n·v_L", f"±{sysm.V2:.4g} V", lpos=(588, 146, "end"))
    c.wire("w_a", (70, 120), (70, 60), l1["a"])
    c.wire("w_b", l1["b"], (330, 60))
    c.wire("w_m", (330, 60), lm["a"])
    c.wire("w_c", (330, 60), l2["a"])
    c.wire("w_d", l2["b"], (610, 60), (610, 120))
    c.wire("w_r1", (70, 180), (70, 240), (330, 240))
    c.wire("w_r2", (330, 240), (610, 240), (610, 180))
    c.wire("w_mr", lm["b"], (330, 240))
    c.dot((330, 60), (330, 240))
    c.text(330, 50, "v_m (권선 전압)", "node")
    c.probe("p1", "i1", 112, 60, "right", "i₁")
    c.probe("pm", "im", 330, 102, "down", "i_m")
    c.probe("p2", "i2", 568, 60, "right", "i₂′")
    all_ = ["V1", "L1", "Lm", "L2", "V2", "w_a", "w_b", "w_m", "w_c", "w_d", "w_r1", "w_r2", "w_mr"]
    txt = {
        "+-": "v₁ = +V₁, v₂′ = −V₂: 직렬 L에 V₁ + V₂가 걸려 i₁이 빠르게 증가",
        "++": "v₁ = +V₁, v₂′ = +V₂: 직렬 L에 V₁ − V₂ (정합이면 0) — 전력 전달 구간",
        "-+": "v₁ = −V₁, v₂′ = +V₂: 직렬 L에 −(V₁ + V₂)",
        "--": "v₁ = −V₁, v₂′ = −V₂: 직렬 L에 −V₁ + V₂",
    }
    for key, lab in DAB_LABELS.items():
        c.mode(key, lab, all_, txt[key] + ". 코어는 v_m을 적분한다.")
    c.notes.append("L₁ = k·L, L₂′ = (1 − k)·L. k = 1이면 v_m = v₂′ (2차 bridge 반사 전압), k → 0이면 v_m ≈ v₁.")
    return c


def run_winding_flux(v: dict) -> Result:
    res = Result("FL07", "winding_flux", "C (정확 스위칭 T-모델)")
    sysm = dab_from(v)
    if not math.isfinite(sysm.phi):
        res.verdict("NO_SOLUTION", "이 전압·L에서는 요구 전력을 SPS로 전달할 수 없다 (P > V₁V₂/(8 f L))")
        return res
    T = sysm.T
    N, Ae, fs = v["Np"], v["Ae"] * 1e-6, v["fs"]  # A_e parameter is in mm^2
    x0, Mmap, q0 = dab_steady(sysm)
    tr = simulate(sysm, q0, x0, 0.0, 2 * T)
    resid = float(np.max(np.abs(tr.state_at(T)[0][:2] - x0)) / max(np.max(np.abs(x0)), 1e-3))
    Blo, Bhi = tr.extrema(0.0, T, "B")
    Bpk = 0.5 * (Bhi - Blo)
    B_dc = 0.5 * (Bhi + Blo)
    Vw_eq = 4.0 * fs * N * Ae * Bpk  # the square amplitude that would give the same flux swing
    B_bus = v["V1"] / (4.0 * fs * N * Ae)
    B_sq = sysm.V2 / (4.0 * fs * N * Ae)
    tb = abs(sysm.V2 - 1000.0) < 1e-9 and abs(N - 50) < 1e-12 and abs(Ae - 250e-6) < 1e-15 and abs(fs - 100e3) < 1e-6 and v["k_split"] == 1.0
    res.add_metric("Bpk", "코어 B_pk (권선 전압 적분, 정확 해)", Bpk, "T", ref=0.2 if tb else (B_sq if v["k_split"] == 1.0 else None), ref_label="교재 ±1000 V → 0.20 T" if tb else "V₂′/(4 f N A_e)", tol=2e-3,
                   basis="(B_max − B_min)/2, 1주기", note="R₂′ 전압강하·자화전류 때문에 사각파 식과 약간 다르다")
    res.add_metric("dBpp", "ΔB_pp = 2·B_pk", 2 * Bpk, "T")
    res.add_metric("B_bus", "버스 전압만으로 계산한 B (V₁/(4 f N A_e))", B_bus, "T", note=f"권선 기준 대비 {100 * (B_bus / Bpk - 1):+.1f} %")
    res.add_metric("Vw_eq", "권선 전압의 등가 사각 진폭 4 f N A_e B_pk", Vw_eq, "V", basis="코어가 실제로 보는 volt-second를 사각파로 환산")
    res.add_metric("B_dc", "B 평균 (DC 성분)", B_dc, "T", basis="zero-DC 주기해 (R₁·R₂′가 DC offset을 감쇠)")
    Im_pk = 0.5 * (tr.extrema(0.0, T, "im")[1] - tr.extrema(0.0, T, "im")[0])
    res.add_metric("Im_pk", "자화전류 peak i_m", Im_pk, "A", basis=f"1차 환산, = N·A_e·B_pk/L_m (L_m = {sysm.Lm * 1e3:g} mH)")
    I1 = tr.rms(0.0, T, "i1")
    I2 = tr.rms(0.0, T, "i2")
    n = N / v["Ns"]
    res.add_metric("I1_rms", "1차 전류 RMS (L_m 포함)", I1, "A")
    res.add_metric("I2_rms", "2차 전류 RMS (1차 환산 i₂′)", I2, "A", basis=f"실제 2차 = n·i₂′ = {n * I2:.4g} A (n = {n:.4g})")
    P = tr.energy(0.0, T, "p1") / T
    res.add_metric("P", "전달 전력 (1차 포트 평균)", P, "W", ref=v["P"], ref_label="SPS 목표 (무손실·L_m 없음 식)", tol=0.03, basis="한 모듈")
    res.add_metric("phi", "위상 φ (rad)", sysm.phi, "", basis=f"{math.degrees(sysm.phi):.4g}° — d = φ/π = {sysm.phi / math.pi:.4g}, Δt/T = φ/2π")
    # limiting case Lm -> infinity: FL08 closed form
    lim = DABT(sysm.V1, sysm.V2, v["L_tot"], 0.0, 1e9, 0.0, 0.0, fs, sysm.phi, N, Ae)
    xl, _, _ = dab_steady(lim)
    trl = simulate(lim, q0, xl, 0.0, T)
    I_lim = trl.rms(0.0, T, "i1")
    if abs(sysm.V1 - sysm.V2) < 1e-9 * sysm.V1:
        rr = ref.dab_sps_matched(sysm.V1, v["L_tot"], fs, v["P"])
        res.add_check(check_close("L_m → ∞ 한계: 스위칭 해 I_rms vs FL08 폐형식", I_lim, rr["I_rms"], 1e-6, "L_m = 1 MH·무손실 T-모델 정확 해 vs I_pk√(1 − 2φ/3π)", True, "A"))
        res.add_metric("I_rms_noLm", "L_m 없는 1차 RMS (폐형식)", rr["I_rms"], "A", ref=2.019881509 if (abs(sysm.V1 - 800) < 1e-9 and abs(v["P"] - 1500) < 1e-9 and abs(v["L_tot"] - 200e-6) < 1e-15 and abs(fs - 1e5) < 1e-6) else None, ref_label="FL08 2.019881509 A", tol=1e-8,
                       basis="부하·순환 전류만", note=f"자화전류가 1차 RMS를 {100 * (I1 / rr['I_rms'] - 1):+.2f} % 바꾼다")
    else:
        e06 = abs(sysm.V1 - 900) < 1e-9 and abs(sysm.V2 - 600) < 1e-9 and abs(v["P"] - 1500) < 1e-9 and abs(v["L_tot"] - 200e-6) < 1e-15 and abs(fs - 1e5) < 1e-6
        res.add_metric("I_rms_noLm", "L_m 없는 1차 RMS (시뮬레이션 한계)", I_lim, "A", ref=3.113563 if e06 else None, ref_label="교재 E06 SPS 3.113563 A", tol=1e-6,
                       basis="부하·순환 전류만 (비정합: 순환전류 포함)", note=f"자화전류가 1차 RMS를 {100 * (I1 / I_lim - 1):+.2f} % 바꾼다")
    # checks
    res.add_check(Check("zero-DC 주기해 잔차 (한 주기)", "PASS" if resid < 1e-9 else "FAIL", resid, "rel", 1e-9, path="반주기 반대칭 x(T/2) = −x(0)으로 푼 초기상태를 한 주기 적분해 되돌아오는지 확인",
                        detail=f"주기 map 고유값 |λ| = {', '.join(f'{abs(z):.6f}' for z in np.linalg.eigvals(Mmap))} (R이 DC offset을 감쇠)"))
    smp = tr.sample(["vm", "B", "i1", "im", "i2", "v1", "v2"], 0.0, 2 * T, per_segment=120)
    t = np.array(smp["t"])
    vm = np.array(smp["vm"])
    B_state = np.array(smp["B"])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (vm[1:] + vm[:-1]) * np.diff(t))])
    B_int = B_state[0] + cum / (N * Ae)
    err = float(np.max(np.abs(B_int - B_state)) / max(Bpk, 1e-12))
    res.add_check(Check("B(t): 권선 전압 사다리꼴 적분 vs 자화전류 상태 L_m i_m/(N A_e)", "PASS" if err < 1e-5 else "FAIL", err, "rel", 1e-5,
                        path="출력 v_m 표본의 누적 사다리꼴 적분 (Faraday) vs 상태변수 i_m (서로 다른 양)", independent=True, detail=f"2주기 최대 편차 / B_pk (표본 {t.size}개)"))
    led = energy_ledger(tr, sysm, 0.0, T, ["p1"], ["p2"], ["p_loss"], rated_power=max(abs(P), 1.0))
    res.add_check(ledger_check(led))
    errs = []
    segs = [(sg.t0 + off, sg.t0 + off + h, sg.q) for sg, off, h in tr.window(0.0, T)]
    for rt in (1e-7, 1e-9):
        y = np.array([x0[0], x0[0] - x0[1]])
        for a, b, q in segs:
            sol = solve_ivp(lambda tt, yy, q=q: sysm.rhs_plain(tt, yy, q), (a, b), y, method="RK45", rtol=rt, atol=rt * 1e-3)
            y = sol.y[:, -1]
        zT = tr.state_at(T)[0]
        errs.append(max(abs(y[0] - zT[0]), abs(y[1] - (zT[0] - zT[1]))) / max(np.max(np.abs(x0)), 1e-3))
    res.add_check(Check("독립 solver: (i₁, i₂′) 상태·3×3 선형계 RK45", "PASS" if errs[-1] < 1e-6 and errs[-1] <= errs[0] + 1e-12 else "FAIL", errs[-1], "rel", 1e-6,
                        path="다른 상태 선택(i₁, i₂′)과 매 단계 3×3 해로 쓴 ODE를 RK45 rtol 1e-7 → 1e-9 vs 정확 해 (i₁, i_m)", independent=True,
                        detail="상대오차: " + ", ".join(f"{e:.2e}" for e in errs)))
    # plots
    for key, lab, unit in (("v1", "v₁ (1차 bridge)", "V"), ("v2", "v₂′ (2차 bridge, 1차 환산)", "V"), ("vm", "v_m (권선·자화 전압)", "V"), ("i1", "i₁ (1차)", "A"), ("i2", "i₂′ (2차 환산)", "A"), ("im", "i_m (자화)", "A")):
        res.add_series(key, lab, unit, smp["t"], smp[key])
    res.add_series("B", "B (자화전류 상태)", "T", smp["t"], smp["B"])
    res.add_series("B_int", "B (권선 전압 적분)", "T", t[::3].tolist(), B_int[::3].tolist(), dash=True)
    bands = bands_from_traj(tr, 0.0, 2 * T, DAB_LABELS)
    g = "dab"
    res.add_plot("p_v", "bridge 전압과 코어가 보는 권선 전압", ["v1", "v2", "vm"], y_label="전압", y_unit="V", bands=bands, group=g, level="C",
                 proved="직렬 L이 1차 쪽에 있으면(k = 1) 권선(자화) 전압은 1차 버스가 아니라 반사된 2차 bridge 전압을 따른다. k를 줄이면 두 전압의 가중 평균이 된다.",
                 not_yet="이상 bridge·이상 변압기 결합(누설은 L₁·L₂′로만)·선형 코어 가정이다. 권선 간 capacitance·링잉은 없다.")
    res.add_plot("p_B", "코어 자속밀도: 권선 전압 적분 vs 버스 기준", ["B", "B_int"], y_label="B", y_unit="T", bands=bands, group=g, level="C",
                 hlines=[{"y": B_bus, "label": f"버스 기준 ±{B_bus:.3f} T"}, {"y": -B_bus, "label": ""}, {"y": v["Bsat"], "label": f"B_sat {v['Bsat']:g} T (가정)"}],
                 proved="B(t)를 권선 전압의 적분과 자화전류 상태로 따로 계산해 일치시켰고, 버스 전압만 넣은 값과의 차이를 보였다.",
                 not_yet="B_sat은 재료·온도 자료가 없는 가정이다(MISSING_INPUT). 포화·히스테리시스는 모델 밖이다.")
    res.add_plot("p_i", "1차·2차·자화 전류", ["i1", "i2", "im"], y_label="전류", y_unit="A", bands=bands, group=g, level="C",
                 proved="1차 전류 = 2차(환산) 전류 + 자화전류. 자화전류는 권선 전압의 적분(삼각파)이고 부하와 무관하다 — DAB 순환전류와 다른 성분이다.",
                 not_yet="자화전류가 ZVS 전하에 주는 도움은 이상 스위치 모델로 판정하지 않는다(NOT_EVALUABLE, EX02).")
    res.circuit = {"diagram": dab_circuit(v, sysm).to_json(), "intervals": bands, "plot_group": g}
    res.tables.append(Table("t_bus", "같은 코어에서 전압을 무엇으로 잡느냐에 따른 B_pk", ["기준", "전압 [V]", "B_pk [T]", "비고"], [
        ["1차 버스 V₁", v["V1"], B_bus, "직렬 L 위치를 무시한 값"],
        ["2차 반사 n·V_L", sysm.V2, B_sq, "k = 1 (직렬 L이 1차 쪽)일 때 코어 전압"],
        ["실제 권선 전압 적분", Vw_eq, Bpk, f"k = {v['k_split']:g}, R·자화전류 포함 (정확 해)"],
    ], note="코어 자속은 코어에 감긴 권선의 전압이 정한다. DC 버스 900 V라서 V = 900을 넣는 것은 직렬 L 배치를 무시한 계산이다."))
    hi_B = max(abs(Bhi), abs(Blo))
    if hi_B > v["Bsat"]:
        res.verdict("FAIL_CONSTRAINT", f"|B|max {hi_B:.3f} T > 가정 B_sat {v['Bsat']:g} T")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"B_pk {Bpk:.4f} T (권선 전압 적분) — 버스 기준 {B_bus:.4f} T. 선형 코어·이상 bridge 범위")
    res.verdict("INFO", "포화 여유는 재료·온도별 B_sat과 DC bias 자료가 있어야 판정한다 (여기 B_sat은 가정 — MISSING_INPUT)")
    res.assumptions += ["선형 코어 (L_m 일정, 포화·히스테리시스 없음), 이상 bridge (dead time·Coss 없음)", f"직렬 L 분배 k = {v['k_split']:g}, L_m·R은 합성 가정", "SPS 위상은 무손실·L_m 없는 폐형식으로 정함"]
    res.not_valid_for += ["코어 손실 정밀값 (MISSING_INPUT, 실험 3)", "포화 판정 (재료 자료 필요)", "ZVS 판정 (NOT_EVALUABLE)"]
    res.interpretation = (
        f"코어가 적분하는 전압은 자화 가지에 걸린 권선 전압이다. 직렬 L이 1차 쪽에 있으면 그 전압은 반사된 2차 bridge 전압(±{sysm.V2:.4g} V)이라, "
        f"1차 버스 {v['V1']:g} V로 계산한 {B_bus:.3f} T가 아니라 {Bpk:.3f} T가 된다. 자화전류는 이 전압의 적분이므로 부하와 무관한 삼각파이고, "
        "1차 전류 = 2차 환산 전류 + 자화전류로 나뉜다. DC offset은 권선 저항이 감쇠시켜 zero-DC 주기해가 정상상태다."
    )
    return res


# ======================================================================================
# Experiment 2: flux walking and how to tell it from measurement artefacts
# ======================================================================================


class WalkCore(HybridSystem):
    """Bridge (+-V, asymmetric timing) -> series R -> optional blocking C_b -> winding L_m.

    x = [i_m] or [i_m, v_Cb].  The + interval lasts T/2 + dt/2 and the - interval T/2 - dt/2, so the
    bridge applies V*dt of net volt-seconds per cycle.
    """

    def __init__(self, V, fs, dt, Lm, R, Cb, N, Ae):
        self.V, self.fs, self.dt, self.Lm, self.R, self.Cb, self.N, self.Ae = V, fs, dt, Lm, R, Cb, N, Ae
        self.T = 1.0 / fs
        self.n = 2 if Cb else 1
        self.state_names = ("i_m", "v_Cb") if Cb else ("i_m",)
        self._key = f"fl07walk|{V:.10g}|{fs:.10g}|{dt:.10g}|{Lm:.10g}|{R:.10g}|{Cb}"

    def mode(self, q):
        if self.Cb:
            A = np.array([[-self.R / self.Lm, -1.0 / self.Lm], [1.0 / self.Cb, 0.0]])
            b = np.array([q * self.V / self.Lm, 0.0])
        else:
            A = np.array([[-self.R / self.Lm]])
            b = np.array([q * self.V / self.Lm])
        return AffineMode(f"{self._key}|{q}", A, b, label="+" if q > 0 else "-")

    def gate_schedule(self, t0, t1):
        ev = []
        k0 = int(math.floor(t0 / self.T)) - 1
        k1 = int(math.ceil(t1 / self.T)) + 1
        for k in range(k0, k1 + 1):
            ta = k * self.T
            tb = ta + 0.5 * self.T + 0.5 * self.dt
            if t0 - 1e-15 <= ta < t1:
                ev.append((ta, lambda q: 1))
            if t0 - 1e-15 <= tb < t1:
                ev.append((tb, lambda q: -1))
        return sorted(ev, key=lambda e: e[0])

    def outputs(self, q):
        n = self.n
        e_i = np.zeros(n + 1)
        e_i[0] = 1.0
        vw = np.zeros(n + 1)
        vw[0] = -self.R
        vw[-1] = q * self.V
        if self.Cb:
            vw[1] = -1.0
        out = {"i": e_i, "vw": vw, "vb": np.r_[np.zeros(n), q * self.V], "B": e_i * self.Lm / (self.N * self.Ae)}
        if self.Cb:
            vc = np.zeros(n + 1)
            vc[1] = 1.0
            out["vc"] = vc
        return out

    def stored_energy(self):
        W = np.zeros((self.n + 1, self.n + 1))
        W[0, 0] = self.Lm
        if self.Cb:
            W[1, 1] = self.Cb
        return W

    def powers(self, q):
        n = self.n
        e_i = np.zeros(n + 1)
        e_i[0] = 1.0
        p_in = np.zeros((n + 1, n + 1))
        p_in[0, n] = p_in[n, 0] = 0.5 * q * self.V
        return {"p_in": p_in, "p_R": self.R * np.outer(e_i, e_i)}

    def describe(self, q):
        return "+" if q > 0 else "-"


def _cum_trapz(t, y):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def walk_circuit(v: dict) -> Circuit:
    cb = v["bound"] == "Cb"
    c = Circuit("fl07_walk", 620, 250, title="비대칭 구동: bridge → R → (C_b) → 권선 L_m")
    c.add("vsource", "VB", 70, 125, 90, "bridge", f"±{v['V']:g} V", lpos=(48, 121, "end"))
    rr = c.add("resistor", "R", 190, 50, 0, "R (권선+채널)", f"{(v['R'] if v['bound'] != 'none' else 0) * 1e3:g} mΩ", lpos=(190, 24, "middle"))
    if cb:
        cc = c.add("capacitor", "Cb", 330, 50, 0, "C_b (blocking)", f"{v['Cb'] * 1e6:g} µF", lpos=(330, 24, "middle"))
        c.wire("w_rc", rr["b"], cc["a"])
        tail = cc["b"]
    else:
        tail = rr["b"]
    lm = c.add("inductor", "Lm", 470, 125, 90, "권선 L_m", f"{v['Lm'] * 1e3:g} mH · N {v['N']:g}", lpos=(492, 121, "start"))
    c.wire("w_a", (70, 95), (70, 50), rr["a"])
    c.wire("w_b", tail, (470, 50), lm["a"])
    c.wire("w_r", lm["b"], (470, 210), (70, 210), (70, 155))
    c.probe("pI", "i", 420, 38, "right", "i_m")
    c.text(505, 70, "v_w (권선 전압)", "node")
    act = ["VB", "R", "Lm", "w_a", "w_b", "w_r"] + (["Cb", "w_rc"] if cb else [])
    c.mode("+", "+V 구간 (T/2 + Δt/2)", act, "권선에 +V − R i (− v_Cb): 자속 증가")
    c.mode("-", "−V 구간 (T/2 − Δt/2)", act, "권선에 −V − R i (− v_Cb): 자속 감소 — 구간이 Δt 짧아 한 주기에 V·Δt가 남는다")
    return c


def run_flux_walk(v: dict) -> Result:
    res = Result("FL07", "flux_walk", "C (정확 스위칭) + 측정 경로 비교")
    V, fs, dt, N, Ae, Lm = v["V"], v["fs"], v["dt"], v["N"], v["Ae"] * 1e-6, v["Lm"]
    bound = v["bound"]
    R = v["R"] if bound in ("R", "Cb") else 0.0
    Cb = v["Cb"] if bound == "Cb" else None
    T = 1.0 / fs
    ncyc = int(v["cycles"])
    sysm = WalkCore(V, fs, dt, Lm, R, Cb, N, Ae)
    i0 = -V / (4.0 * fs * Lm)  # symmetric-steady start: flux at its negative peak when the + interval begins
    x0 = [i0] + ([0.0] if Cb else [])
    tr = simulate(sysm, 1, x0, 0.0, ncyc * T)
    kB = Lm / (N * Ae)
    Bpk_sym = V / (4.0 * fs * N * Ae)
    walk_an = ref.walk_per_cycle(V, dt, N, Ae)
    z1 = tr.state_at(T)[0]
    walk_sim = (z1[0] - x0[0]) * kB
    tb = abs(V - 800) < 1e-9 and abs(dt - 100e-9) < 1e-18 and abs(N - 50) < 1e-12 and abs(Ae - 250e-6) < 1e-15
    Vavg = V * dt * fs
    res.add_metric("walk", "첫 주기 자속 이동 ΔB (정확 해)", walk_sim, "T", ref=0.0064 if (tb and R == 0) else (walk_an if R == 0 else None), ref_label="교재 800 V × 100 ns/(50·250 mm²) = 0.0064 T" if tb else "V·Δt/(N A_e)", tol=1e-9,
                   basis="B(T) − B(0)", note="R이 있으면 첫 주기부터 조금 작다" if R > 0 else "")
    res.add_metric("Vavg", "bridge 전압 주기 평균 V·Δt·f", Vavg, "V", basis="비대칭이 만든 DC 성분")
    res.add_metric("Bpk_sym", "대칭 구동이었다면 B_pk", Bpk_sym, "T", basis="V/(4 f N A_e)")
    smp = tr.sample(["B", "vw", "i", "vb"] + (["vc"] if Cb else []), per_segment=16)
    t = np.array(smp["t"])
    B = np.array(smp["B"])
    k_sat = next((j for j in range(t.size) if abs(B[j]) > v["Bsat"]), None)
    t_sat = float(t[k_sat]) if k_sat is not None else None
    res.add_metric("Bmax", "창 안의 |B| 최대", float(np.max(np.abs(B))), "T", note=f"가정 B_sat {v['Bsat']:g} T")
    if t_sat is not None:
        res.add_metric("t_sat", "|B|가 B_sat을 넘는 시각", t_sat, "s", basis=f"{t_sat * fs:.1f} 주기 (선형 모델 — 실제로는 여기서 포화)")
    if bound == "R":
        tau = Lm / R
        res.add_metric("tau", "DC 전류 시정수 L_m/R", tau, "s", basis=f"{tau * fs:.0f} 주기")
        res.add_metric("Idc_inf", "R이 멈추게 하는 DC 전류 V_avg/R", Vavg / R, "A")
        res.add_metric("Bdc_inf", "그때의 DC 자속 (선형 모델)", kB * Vavg / R, "T", note="B_sat보다 훨씬 커서 물리적으로 도달 전에 포화 — R은 보호가 아니다")
    if Cb:
        f0 = 1.0 / (TWO_PI * math.sqrt(Lm * Cb))
        Q = math.sqrt(Lm / Cb) / max(R, 1e-12)
        vc = np.array(smp["vc"])
        res.add_metric("f0", "L_m–C_b 직렬 공진", f0, "Hz", basis=f"Q = {Q:.0f} (R만 감쇠)")
        res.add_metric("tau_d", "공진 과도 감쇠 시정수 2L_m/R", 2 * Lm / max(R, 1e-12), "s", basis="R만으로 감쇠하면 수십 ms — 실제 설계는 감쇠를 따로 넣는다")
        res.add_metric("vc_end", "C_b 전압 (마지막 주기 평균, 공진 과도 중)", tr.mean((ncyc - 1) * T, ncyc * T, "vc"), "V", ref=Vavg, ref_label="장기 평균 V_avg (DC를 C_b가 받음)", tol=None)
        res.add_metric("Bdc_peak", "과도 DC 자속 offset 최대", float(np.max(np.abs(B))) - Bpk_sym, "T", ref=kB * Vavg / math.sqrt(Lm / Cb), ref_label="L_m·(V_avg/Z₀)/(N A_e) 근사", tol=None, basis="C_b 충전 과도 (공진)")
        res.add_metric("Idc_end", "마지막 주기 평균 자화전류 (공진 과도 중)", tr.mean((ncyc - 1) * T, ncyc * T, "i"), "A", basis="장기적으로 → 0 (DC가 C_b에 막힌다), 지금은 L_m–C_b 진동 중")
    # checks
    res.add_check(check_close("주기당 자속 이동: 정확 해 vs volt-second 식", walk_sim, walk_an * (1.0 if R == 0 else walk_sim / walk_an), 1e-9 if R == 0 else 1.0, "행렬지수 적분의 B(T) − B(0) vs V·Δt/(N A_e)", R == 0, "T") if R == 0 else
                  Check("주기당 자속 이동 (R 있음)", "INFO", walk_sim, "T", path="R 전압강하만큼 V·Δt/(N A_e)보다 작다", detail=f"식 {walk_an:.5g} T"))
    vw = np.array(smp["vw"])
    Bi = B[0] + _cum_trapz(t, vw) / (N * Ae)
    err = float(np.max(np.abs(Bi - B)) / max(Bpk_sym, 1e-12))
    res.add_check(Check("B: 권선 전압 적분 vs 자화전류 상태", "PASS" if err < 1e-4 else "FAIL", err, "rel", 1e-4, path="v_w 표본 누적 사다리꼴 적분 vs L_m i_m/(N A_e)", independent=True, detail=f"창 전체 최대 편차 / B_pk (표본 {t.size}개)"))
    led = energy_ledger(tr, sysm, 0.0, ncyc * T, ["p_in"], [], ["p_R"], rated_power=max(V * abs(i0), 1.0))
    res.add_check(ledger_check(led))
    # measurement artefacts on a symmetric, periodic reference (true flux does not walk)
    ref_sys = WalkCore(V, fs, 0.0, Lm, 0.0, None, N, Ae)
    mcyc = min(ncyc, 40)
    trr = simulate(ref_sys, 1, [i0], 0.0, mcyc * T)
    sm = trr.sample(["B", "vw"], per_segment=40)
    tt = np.array(sm["t"])
    Bt = np.array(sm["B"])
    Bos = Bt[0] + _cum_trapz(tt, np.array(sm["vw"]) + v["V_os"]) / (N * Ae)
    fsm = v["f_samp"]
    ts = (np.arange(int(mcyc * T * fsm)) + 0.37) / fsm
    zs = [trr.state_at(x) for x in ts]
    vw_s = np.array([float(z @ ref_sys.outputs(q)["vw"]) for z, q in zs])
    Btrue_s = np.array([float(z[0]) * kB for z, _ in zs])
    Bco = Btrue_s[0] + _cum_trapz(ts, vw_s) / (N * Ae)

    def slope(x, y):
        return float(np.polyfit(x, y, 1)[0])

    drift_os = slope(tt, Bos - Bt) * T
    drift_co = slope(ts, Bco - Btrue_s) * T
    res.add_metric("drift_os", f"probe offset {v['V_os']:g} V의 적분 drift (주기당)", drift_os, "T", ref=v["V_os"] * T / (N * Ae), ref_label="V_os·T/(N A_e)", tol=1e-3, basis="실제 자속은 주기적 (대칭 구동)")
    res.add_metric("drift_co", f"{fsm / 1e6:g} MS/s 표본 적분 drift (주기당)", drift_co, "T", basis=f"T·f_s = {T * fsm:.4g}: 상승·하강 edge가 표본 사이 다른 위치에 떨어지는 동기 표본화 오차",
                   note="T·f_s가 홀수면 최대 V/(f_s N A_e), 짝수면 상쇄, 비동기면 누적되지 않고 흔들림")
    nT = T * fsm
    if abs(nT - round(nT)) < 1e-9 and int(round(nT)) % 2 == 1:
        res.add_check(check_close("동기 표본 적분 drift: 수치 vs 사다리꼴 edge 오차식", drift_co, V / (fsm * N * Ae), 2e-3, "표본 파형의 누적 사다리꼴 적분 기울기 vs 반 표본 어긋난 두 edge의 오차 합 V·Δt_s/(N A_e)", True, "T"))
    res.tables.append(Table("t_sign", "walk와 측정 오차를 구분하는 신호", ["경우", "측정 v의 주기 평균", "적분 B의 주기당 변화", "자화전류 주기 평균 변화", "판정"], [
        [f"실제 walk (Δt = {dt * 1e9:g} ns)", f"{Vavg:.3g} V", f"{walk_an:.4g} T", f"{walk_an / kB:.4g} A/주기", "실제 자속 이동 — 전류가 증거"],
        [f"probe offset {v['V_os']:g} V", f"{v['V_os']:g} V", f"{drift_os:.3g} T", "0", "측정 오차: 평균 제거로 사라짐 (단 실제 walk도 같이 지워짐)"],
        [f"표본화 {fsm / 1e6:g} MS/s 사다리꼴", "≈ 0 (실제)", f"{drift_co:.3g} T", "0", "수치 오차: 표본화를 edge에 맞추거나 구간 적분"],
    ], note="자화전류(또는 별도 sense 권선)를 같은 timebase로 보면 실제 walk와 적분 drift를 구분할 수 있다. 전압 평균을 빼는 보정은 실제 walk까지 지운다."))
    # plots
    xs, ys = decimate_minmax(t, B, 2500)
    res.add_series("B", "B (자화전류 상태)", "T", xs, ys)
    xs, ys = decimate_minmax(t, np.array(smp["i"]), 2500)
    res.add_series("i", "i_m", "A", xs, ys)
    hl = [{"y": v["Bsat"], "label": f"B_sat {v['Bsat']:g} T (가정)"}, {"y": -v["Bsat"], "label": ""}, {"y": Bpk_sym, "label": "대칭 B_pk"}]
    res.add_plot("p_B", f"자속밀도 {ncyc}주기 ({'이상: 저항 없음' if bound == 'none' else ('R만' if bound == 'R' else 'R + blocking C')})", ["B"], y_label="B", y_unit="T", level="C", hlines=hl, group="long",
                 vlines=[{"x": t_sat, "label": "B_sat 도달"}] if t_sat else [],
                 proved="한쪽으로 V·Δt가 매 주기 남으면 자속 평균이 주기마다 이동한다. R은 이동을 L_m/R 시정수로만 늦추고 도달점은 비현실적인 DC 자속이며, blocking C_b는 DC를 막아 자속을 가둔다(대신 L_m–C_b 공진 과도).",
                 not_yet="선형 코어: 실제로는 B_sat 근처에서 L_m이 무너져 전류가 급증한다(포화 모델 없음). 제어로 비대칭을 보정하는 루프는 넣지 않았다.")
    res.add_plot("p_i", "자화전류 (walk의 증거)", ["i"], y_label="i_m", y_unit="A", level="C", group="long",
                 proved="실제 walk는 자화전류의 평균이 움직이는 것으로 확인된다(측정 오차는 전류에 나타나지 않는다).", not_yet="")
    if Cb:
        xs, ys = decimate_minmax(t, np.array(smp["vc"]), 2500)
        res.add_series("vc", "v_Cb", "V", xs, ys)
        res.add_plot("p_vc", "blocking capacitor 전압", ["vc"], y_label="v_Cb", y_unit="V", level="C", group="long", hlines=[{"y": Vavg, "label": f"V_avg {Vavg:.3g} V"}],
                     proved="C_b가 비대칭의 DC 성분(V·Δt·f)을 떠안는다.", not_yet="C_b의 AC 리플 전압·손실·공진 감쇠 설계는 따로 한다.")
    tz = min(3, ncyc) * T
    smz = tr.sample(["vw", "B"], 0.0, tz, per_segment=40)
    res.add_series("vw_z", "권선 전압 v_w", "V", smz["t"], smz["vw"])
    res.add_series("B_z", "B", "T", smz["t"], smz["B"])
    bands = bands_from_traj(tr, 0.0, tz, {"+": "+V", "-": "−V"})
    res.add_plot("p_vz", "처음 3주기: 비대칭 구동의 권선 전압", ["vw_z"], y_label="v_w", y_unit="V", bands=bands, group="z", level="C",
                 proved="+V 구간이 −V 구간보다 Δt 길다 — 파형만 봐서는 100 ns 차이를 알아보기 어렵다.", not_yet="")
    res.add_plot("p_Bz", "처음 3주기: 자속", ["B_z"], y_label="B", y_unit="T", bands=bands, group="z", level="C",
                 proved="매 주기 끝의 B가 조금씩 올라간다.", not_yet="")
    res.add_series("m_true", "실제 B (대칭 구동)", "T", tt.tolist(), Bt.tolist())
    res.add_series("m_os", f"측정 v + offset {v['V_os']:g} V 적분", "T", tt.tolist(), Bos.tolist(), dash=True)
    res.add_series("m_co", f"{fsm / 1e6:g} MS/s 표본 사다리꼴 적분", "T", ts.tolist(), Bco.tolist(), dash=True)
    res.add_plot("p_meas", "측정 경로 비교: 실제 자속은 제자리인데 적분은 흐른다", ["m_true", "m_os", "m_co"], y_label="B", y_unit="T", level="C",
                 proved="대칭 구동(실제 walk 없음)에서도 probe offset이나 edge를 놓치는 표본 적분은 주기마다 일정하게 흘러 walk처럼 보인다. 자화전류는 움직이지 않는다.",
                 not_yet="실제 probe 대역폭·지연·잡음은 넣지 않았다.")
    res.circuit = {"diagram": walk_circuit(v).to_json(), "intervals": bands, "plot_group": "z"}
    if t_sat is not None:
        res.verdict("FAIL_CONSTRAINT", f"{t_sat * fs:.0f}주기({t_sat * 1e6:.0f} µs)에 |B|가 가정 B_sat {v['Bsat']:g} T를 넘는다 — {'R은 이동을 늦출 뿐 막지 못한다' if bound == 'R' else '이상 구동에서는 끝없이 이동'}")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"창 안에서 |B| ≤ {float(np.max(np.abs(B))):.3f} T (선형 코어·가정 B_sat 기준)")
    res.assumptions += ["선형 L_m (포화 없음), 이상 bridge, 비대칭은 매 주기 같은 Δt", "B_sat·R·C_b·probe offset·표본화율은 학습용 가정"]
    res.not_valid_for += ["포화 이후 전류 (포화 모델 없음)", "능동 flux balance 제어 성능", "실측 probe 오차 보정"]
    res.interpretation = (
        f"+V 구간이 {dt * 1e9:g} ns 길면 한 주기에 {V * dt * 1e6:.3g} µVs가 남아 B가 {walk_an:.4f} T씩 이동한다(평균 전압 {Vavg:.3g} V). "
        "저항은 V_avg/R의 DC 전류에서 멈추게 하지만 그 자속은 포화보다 훨씬 커서 보호가 되지 못하고, blocking C_b는 DC를 막아 자속을 가두는 대신 L_m–C_b 공진 과도를 만든다. "
        "적분한 B가 흐른다고 모두 walk는 아니다: probe offset·표본화 오차는 전압 평균에만 나타나고 자화전류에는 나타나지 않는다."
    )
    return res


# ======================================================================================
# Experiment 3: roles, referral and loss screens
# ======================================================================================


class LoopDAB(HybridSystem):
    """Single-loop DAB with series L, R and a series blocking C: L di/dt = s1 V1 - R i - v_C - s2 V2."""

    state_names = ("i", "v_C")
    state_units = ("A", "V")

    def __init__(self, V1, V2, L, R, C, fs, phi, tag):
        self.V1, self.V2, self.L, self.R, self.C, self.fs, self.phi = V1, V2, L, R, C, fs, phi
        self.T = 1.0 / fs
        self.t_phi = phi / (TWO_PI * fs)
        self._key = f"fl07loop|{tag}|{V1:.10g}|{V2:.10g}|{L:.10g}|{R:.10g}|{C:.10g}|{fs:.10g}|{phi:.12g}"

    def mode(self, q):
        s1, s2 = q
        A = np.array([[-self.R / self.L, -1.0 / self.L], [1.0 / self.C, 0.0]])
        b = np.array([(s1 * self.V1 - s2 * self.V2) / self.L, 0.0])
        return AffineMode(f"{self._key}|{q}", A, b)

    gate_schedule = DABT.gate_schedule

    def outputs(self, q):
        return {"i": np.array([1.0, 0.0, 0.0]), "vc": np.array([0.0, 1.0, 0.0])}

    def stored_energy(self):
        return np.diag([self.L, self.C, 0.0])

    def powers(self, q):
        s1, s2 = q
        e = np.array([1.0, 0.0, 0.0])
        p1 = np.zeros((3, 3))
        p1[0, 2] = p1[2, 0] = 0.5 * s1 * self.V1
        p2 = np.zeros((3, 3))
        p2[0, 2] = p2[2, 0] = 0.5 * s2 * self.V2
        return {"p1": p1, "p2": p2, "p_R": self.R * np.outer(e, e)}


def loop_steady(sysm: LoopDAB):
    q0 = (1, -1)

    def half(x):
        return simulate(sysm, q0, x, 0.0, sysm.T / 2).z_end[:2].copy()

    g = half([0.0, 0.0])
    H = np.column_stack([half([1.0, 0.0]) - g, half([0.0, 1.0]) - g])
    return np.linalg.solve(np.eye(2) + H, -g), q0


def skin_fd(rho: float, f: float, n: int = 6000, span: float = 12.0):
    """Independent path: 1-D magnetic diffusion d2H/dx2 = j w mu0 sigma H in a half-space, central differences.

    Boundary H(0) = 1, H(span*delta_guess) = 0; returns x, |H|/|H(0)| and the decay length fitted on [0.5, 5] delta_guess.
    """
    from scipy.linalg import solve_banded

    sig = 1.0 / rho
    k2 = 1j * TWO_PI * f * MU0 * sig
    dg = math.sqrt(2.0 / (TWO_PI * f * MU0 * sig))  # only sets the mesh extent
    X = span * dg
    x = np.linspace(0.0, X, n + 1)
    h = x[1] - x[0]
    m = n - 1
    ab = np.zeros((3, m), dtype=complex)
    ab[0, 1:] = 1.0
    ab[1, :] = -2.0 - k2 * h * h
    ab[2, :-1] = 1.0
    rhs = np.zeros(m, dtype=complex)
    rhs[0] = -1.0
    Hin = solve_banded((1, 1), ab, rhs)
    H = np.concatenate([[1.0], Hin, [0.0]])
    mag = np.abs(H)
    sel = (x > 0.5 * dg) & (x < 5.0 * dg)
    slope = np.polyfit(x[sel], np.log(mag[sel]), 1)[0]
    return x, mag, -1.0 / slope


def rac_round(d: float, delta: float) -> float:
    """Exact skin-effect R_ac/R_dc of an isolated round wire: Re[(k a/2) J0(k a)/J1(k a)], k = (1 - j)/delta."""
    a = d / 2.0
    ka = (1 - 1j) * a / delta
    return float(np.real(ka / 2.0 * jv(0, ka) / jv(1, ka)))


def dowell(delta_ratio, m: int):
    """Dowell AC-resistance factor of an m-layer foil/equivalent-foil winding, Delta = h/delta."""
    D = np.asarray(delta_ratio, dtype=float)
    t1 = (np.sinh(2 * D) + np.sin(2 * D)) / (np.cosh(2 * D) - np.cos(2 * D))
    t2 = (np.sinh(D) - np.sin(D)) / (np.cosh(D) + np.cos(D))
    return D * (t1 + 2.0 * (m * m - 1) / 3.0 * t2)


def igse_ki(k: float, alpha: float, beta: float) -> float:
    th = np.linspace(0.0, TWO_PI, 20001)
    integ = float(np.trapezoid(np.abs(np.cos(th)) ** alpha, th)) * 2.0 ** (beta - alpha)
    return k / ((TWO_PI) ** (alpha - 1.0) * integ)


SUPPLIER_ROWS = [
    ["권선도와 N_s 정의", "n = N_p/N_s와 환산의 기준", "권선 순서·병렬 가닥·탭"],
    ["A_e, V_e, l_e, window A_w", "B = ∫v dt/(N A_e), 손실 체적, 창 면적", "코어 형번·공차"],
    ["재료 B_sat vs 온도", "포화 여유 (DC bias 포함)", "25/100/120 °C, 측정 주파수"],
    ["재료 손실 곡선", "core loss (주파수·B·온도·waveform)", "B 정의(B_pk/ΔB), 사인/사각 여자, minor loop·DC bias 여부"],
    ["L_m 측정값", "자화전류·ZVS 전하", "측정 주파수·진폭(B 수준)·fixture·다른 권선 개방 여부"],
    ["누설 L 측정값", "DAB 전력·LLC 공진", "단락 방법(저임피던스)·주파수·fixture 보정"],
    ["DCR, R_ac(f)", "권선 손실 (고조파별)", "온도, 주파수 sweep, 측정 포트"],
    ["권선 capacitance (C_p, C_ps)", "공진·EMI·측정 오차", "측정 연결(다른 권선 접지/개방)"],
    ["절연 구조", "creepage·clearance·내전압", "working voltage, pollution degree, 재료군"],
    ["hotspot 조건", "열 설계", "냉각 조건, hotspot 위치·측정 방법, 공차"],
]


def run_loss_screen(v: dict) -> Result:
    res = Result("FL07", "loss_screen", "A (screen) + C (환산 검증용 두 회로)")
    Np, Ns = v["Np"], v["Ns"]
    n = Np / Ns
    fs = v["fs"]
    V1, VL = v["V1"], v["VL"]
    V2 = n * VL
    Lp = v["L_ref"]
    Ls = Lp / (n * n)
    phi = sps_phase(V1, V2, Lp, fs, v["P"])
    tbn = abs(Np - 50) < 1e-12 and abs(Ns - 3) < 1e-12 and abs(Lp - 200e-6) < 1e-15
    res.add_metric("n", "권선비 n = N_p/N_s", n, "", ref=50 / 3 if tbn else None, ref_label="교재 16.667", tol=1e-12)
    res.add_metric("Ls", "2차 기준 인덕턴스 L = L′/n²", Ls, "H", ref=0.72e-6 if tbn else Lp / ref.refer(n, L=1.0)["L"], ref_label="교재 0.72 µH" if tbn else "L′/n² (reference)", tol=1e-12, basis="1차 환산 200 µH ↔ 2차 실제")
    # two circuits: primary-referred vs actual secondary side
    Rs, Cs = v["R_s"], v["C_s"]
    refc = LoopDAB(V1, V2, Lp, n * n * Rs, Cs / (n * n), fs, phi, "ref")
    actc = LoopDAB(V1 / n, VL, Ls, Rs, Cs, fs, phi, "act")
    T = 1.0 / fs
    out = {}
    for tag, sysm in (("ref", refc), ("act", actc)):
        x0, q0 = loop_steady(sysm)
        tr = simulate(sysm, q0, x0, 0.0, T)
        out[tag] = {"P1": tr.energy(0, T, "p1") / T, "PR": tr.energy(0, T, "p_R") / T, "I": tr.rms(0, T, "i"), "tr": tr, "W": [0.5 * float(tr.state_at(t)[0] @ sysm.stored_energy() @ tr.state_at(t)[0]) for t in np.linspace(0, T, 9)]}
    res.add_metric("P_ref", "전달 전력: 1차 환산 회로", out["ref"]["P1"], "W", note="직렬 C가 L의 리액턴스를 일부 상쇄해 SPS 목표와 다르다 — 비교 대상은 두 회로")
    res.add_metric("P_act", "전달 전력: 2차 실제 값 회로", out["act"]["P1"], "W", ref=out["ref"]["P1"], ref_label="1차 환산 회로", tol=1e-9)
    res.add_metric("Ip", "1차 전류 RMS (환산 회로)", out["ref"]["I"], "A")
    res.add_metric("Is", "2차 전류 RMS (실제 회로)", out["act"]["I"], "A", ref=n * out["ref"]["I"], ref_label="n·I_p", tol=1e-9, basis="I_s = n·I_p")
    res.add_metric("PR", "직렬 저항 손실 (실제 R_s vs 환산 n²R_s)", out["act"]["PR"], "W", ref=out["ref"]["PR"], ref_label="환산 회로 n²R_s·I_p²", tol=1e-9)
    res.add_check(Check("환산 규칙: 두 회로의 저장에너지 파형", "PASS" if max(abs(a - b) for a, b in zip(out["ref"]["W"], out["act"]["W"])) < 1e-9 * max(out["ref"]["W"]) else "FAIL",
                        max(abs(a - b) for a, b in zip(out["ref"]["W"], out["act"]["W"])), "J", 1e-9, path="L′ = n²L, C′ = C/n², R′ = n²R로 만든 1차 환산 회로 vs 2차 실제 값 회로 (둘 다 정확 해) — 한 주기 9개 시각의 ½Li² + ½Cv²",
                        independent=True, detail="전력·RMS·손실은 위 수치 비교 (환산이 틀리면 두 회로가 다른 결과를 낸다)"))
    res.tables.append(Table("t_refer", "실제 값 ↔ 1차 환산 (n = N_p/N_s)", ["양", "2차 실제", "1차 환산", "규칙"], [
        ["직렬 L", f"{Ls * 1e6:.4g} µH", f"{Lp * 1e6:.4g} µH", "L′ = n²L"],
        ["직렬 C", f"{Cs * 1e6:.4g} µF", f"{Cs / n / n * 1e9:.4g} nF", "C′ = C/n²"],
        ["직렬 R", f"{Rs * 1e3:.4g} mΩ", f"{n * n * Rs * 1e3:.4g} mΩ", "R′ = n²R"],
        ["bridge 전압", f"{VL:g} V", f"{V2:.4g} V", "V′ = n·V"],
        ["전류 RMS", f"{out['act']['I']:.4g} A", f"{out['ref']['I']:.4g} A", "I′ = I/n"],
    ], note="회로도·결과에 실제 값과 환산 값을 섞어 표시하지 않는다. 전력과 저장에너지는 환산 전후 같다."))
    # skin depth: closed form vs finite-difference diffusion
    rho = v["rho"]
    delta = math.sqrt(rho / (math.pi * fs * MU0))
    x, mag, d_fd = skin_fd(rho, fs)
    res.add_metric("delta", f"구리 skin depth δ @ {fs / 1e3:g} kHz", delta, "m", ref=0.209e-3 if (abs(rho - 1.72e-8) < 1e-12 and abs(fs - 1e5) < 1e-6) else ref.skin_depth(rho, fs), ref_label="교재 약 0.209 mm" if abs(fs - 1e5) < 1e-6 else "√(ρ/(πfμ₀))", tol=2e-3, basis="20 °C ρ, μ ≈ μ₀")
    res.add_check(check_close("skin depth: 폐형식 vs 1-D 확산 유한차분", d_fd, delta, 1e-4, "d²H/dx² = jωμσH를 중앙차분(6000 격자)으로 풀고 ln|H| 기울기로 감쇠 길이 추정 vs √(ρ/(πfμ))", True, "m"))
    keep = x / delta <= 10.0  # the far boundary H(12 delta) = 0 bends the numerical curve only near the end
    res.add_series("sk_fd", "|H(x)|/|H(0)| 유한차분", "", (x[keep] / delta).tolist()[::20], mag[keep].tolist()[::20])
    res.add_series("sk_an", "exp(−x/δ) 폐형식", "", (x[keep] / delta).tolist()[::20], np.exp(-x[keep] / delta).tolist()[::20], dash=True)
    res.add_plot("p_skin", "도체 속 전류밀도 감쇠 (반무한 도체)", ["sk_fd", "sk_an"], x_label="x/δ", x_unit="", y_label="|J|/|J(0)|", y_unit="", kind="xy", log_y=True, level="A",
                 proved=f"δ = {delta * 1e3:.4f} mm를 폐형식과 확산 방정식의 수치해 두 경로로 확인했다.", not_yet="평판 반무한 도체의 1-D 해다. 실제 권선은 근접효과·층 배치가 지배할 수 있다.")
    # round wire and Dowell screens
    ds = np.geomspace(0.05e-3, 3e-3, 80)
    res.add_series("rw1", f"R_ac/R_dc @ {fs / 1e3:g} kHz", "", (ds * 1e3).tolist(), [rac_round(d, delta) for d in ds])
    d5 = math.sqrt(rho / (math.pi * 5 * fs * MU0))
    res.add_series("rw5", f"R_ac/R_dc @ {5 * fs / 1e3:g} kHz (5차)", "", (ds * 1e3).tolist(), [rac_round(d, d5) for d in ds], dash=True)
    small = 0.2 * delta
    res.add_check(check_close("원형선 R_ac/R_dc: Bessel 정확식 vs 저주파 전개 1 + (a/δ)⁴/48", rac_round(2 * small, delta), 1 + (small / delta) ** 4 / 48, 1e-6, "복소 Bessel J₀/J₁ vs 급수 전개 (a = 0.2δ)", True, ""))
    res.add_plot("p_round", "원형 단선의 skin 효과 (고립 도선)", ["rw1", "rw5"], x_label="선경 d", x_unit="mm", y_label="R_ac/R_dc", y_unit="", kind="xy", log_x=True, level="A",
                 vlines=[{"x": 2 * delta * 1e3, "label": "d = 2δ"}],
                 proved="선경이 2δ보다 작으면 skin 효과만으로는 R_ac가 거의 늘지 않는다 — litz 가닥 선택의 출발점이다.", not_yet="이웃 도선의 근접효과를 넣지 않았다(아래 Dowell).")
    Dg = np.geomspace(0.1, 5.0, 120)
    for m_ in (1, 2, 4, 8):
        res.add_series(f"dw{m_}", f"{m_}층", "", Dg.tolist(), dowell(Dg, m_).tolist())
    Dop = v["h_foil"] / delta
    FR_op = float(dowell(Dop, int(v["layers"])))
    res.add_metric("Delta", "Δ = h/δ (foil 두께 기준)", Dop, "", basis=f"h = {v['h_foil'] * 1e3:g} mm (ASSUMED)")
    res.add_metric("FR", f"Dowell F_R ({int(v['layers'])}층, 기본파)", FR_op, "", note=f"3차 {float(dowell(Dop * math.sqrt(3), int(v['layers']))):.3g}, 5차 {float(dowell(Dop * math.sqrt(5), int(v['layers']))):.3g}")
    lo = 0.2
    for m_ in (1, 4):
        res.add_check(check_close(f"Dowell {m_}층: 정확식 vs 저주파 전개 1 + (5m²−1)Δ⁴/45", float(dowell(lo, m_)), 1 + (5 * m_ * m_ - 1) * lo**4 / 45, 1e-4, "쌍곡·삼각 함수식 vs 급수 전개 (Δ = 0.2)", True, ""))
    res.add_plot("p_dowell", "다층 권선: 근접효과가 지배한다 (Dowell)", [f"dw{m_}" for m_ in (1, 2, 4, 8)], x_label="Δ = h/δ", x_unit="", y_label="F_R = R_ac/R_dc", y_unit="", kind="xy", log_x=True, log_y=True, level="A",
                 vlines=[{"x": Dop, "label": "이 권선"}],
                 proved="같은 Δ에서도 층 수가 늘면 F_R이 급격히 커진다. skin depth 하나로 foil·litz 최적화를 끝낼 수 없다는 교재 경고를 수치로 보였다.",
                 not_yet="1-D 평판 근사(Dowell). 실제 권선의 gap fringing·끝단·litz 가닥 구조는 측정 R_ac(f)가 필요하다(MISSING_INPUT).")
    # core loss screen (synthetic coefficients)
    kst, al, be, Ve = v["k_st"], v["alpha"], v["beta"], v["Ve"] * 1e-6  # V_e parameter in cm^3
    vv = {"V1": V1, "VL": VL, "Np": Np, "Ns": Ns, "Ae": v["Ae"], "fs": fs, "L_tot": Lp, "k_split": v["k_split"], "Lm": v["Lm"], "R1": 0.05, "R2": 0.05, "P": v["P"]}
    dab = dab_from(vv)
    x0, _, q0 = dab_steady(dab)
    trd = simulate(dab, q0, x0, 0.0, T)
    blo, bhi = trd.extrema(0.0, T, "B")
    Bpk = 0.5 * (bhi - blo)
    ki = igse_ki(kst, al, be)
    sm = trd.sample(["vm"], per_segment=400)
    dBdt = np.abs(np.array(sm["vm"])) / (Np * v["Ae"] * 1e-6)
    P_igse_num = ki * (2 * Bpk) ** (be - al) * float(np.trapezoid(dBdt**al, sm["t"])) / T * Ve
    P_sine = kst * fs**al * Bpk**be * Ve
    P_tri = ki * (4.0 * Bpk * fs) ** al * (2 * Bpk) ** (be - al) * Ve
    P_wrongB = kst * fs**al * (2 * Bpk) ** be * Ve
    res.add_metric("Bpk_core", "코어 B_pk (실험 1과 같은 T-모델)", Bpk, "T", basis=f"k = {v['k_split']:g}")
    res.add_metric("P_sine", "Steinmetz 사인 여자 P = k f^α B_pk^β V_e", P_sine, "W", basis="SYNTHETIC 계수 — 재료 값 아님")
    res.add_metric("P_igse", "iGSE (실제 권선 전압 파형, 수치)", P_igse_num, "W", basis="dB/dt = v_m/(N A_e) 표본 적분")
    res.add_metric("ratio", "iGSE / 사인 Steinmetz", P_igse_num / P_sine, "", note="같은 B_pk에서도 waveform이 손실을 바꾼다")
    res.add_metric("P_wrongB", "ΔB_pp를 B_pk 자리에 넣은 잘못된 계산", P_wrongB, "W", note=f"2^β = {2 ** be:.2f}배 과대 — 계수의 B 정의를 확인")
    if v["k_split"] == 1.0:
        res.add_check(check_close("iGSE: 삼각 자속 폐형식 vs 실제 파형 수치 적분", P_igse_num, P_tri, 2e-3, "k_i(4B_pk f)^α(2B_pk)^{β−α} vs 권선 전압 표본의 ∫|dB/dt|^α dt (R₂′ 강하만큼 차이)", True, "W"))
    res.tables.append(Table("t_supplier", "공급사에게 요청할 정보 (현재 모두 MISSING_INPUT)", ["항목", "왜 필요한가", "함께 받아야 할 조건"], SUPPLIER_ROWS,
                            note="“200 µH transformer” 한 줄은 설계 데이터가 아니다. 값마다 측정 조건이 붙어야 비교·환산이 가능하다."))
    res.verdict("PASS_WITHIN_MODEL", "환산 규칙(두 회로 일치), skin depth(두 경로), Dowell·Bessel 극한, iGSE 폐형식-수치 일치 (합성 screen 범위)")
    res.verdict("MISSING_INPUT", "정밀 core loss·R_ac는 재료 손실 곡선(주파수·B·온도·waveform)과 측정 R_ac(f)가 있어야 만든다 — 여기 Steinmetz 계수는 SYNTHETIC")
    res.assumptions += ["Steinmetz k·α·β·V_e는 SYNTHETIC (특정 재료 fit 아님), 온도 의존 없음", "Dowell 1-D 평판 근사, 원형선은 고립 도선", "환산 검증 회로의 직렬 R_s·C_s는 합성 값"]
    res.not_valid_for += ["특정 코어·재료의 손실 예측", "minor loop·DC bias·온도에 따른 손실", "litz·foil 권선 최적화 확정"]
    res.interpretation = (
        f"n = {n:.4g}에서 1차 환산 {Lp * 1e6:g} µH는 2차 기준 {Ls * 1e6:.3g} µH다. 두 회로를 따로 풀어도 전력·손실·저장에너지가 같다는 것이 L′ = n²L, C′ = C/n², R′ = n²R의 검증이다. "
        f"skin depth {delta * 1e3:.3f} mm는 도체 크기의 기준일 뿐, 층이 많으면 근접효과(Dowell F_R {FR_op:.2f})가 권선 손실을 지배한다. "
        "core loss는 B 정의·waveform·온도·재료 곡선에 따라 크게 달라지므로 합성 계수로는 비교 감각만 얻고, 정밀값은 재료 자료 없이는 만들지 않는다."
    )
    return res


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q = [
    Question(
        "DC 버스가 900 V이니 코어 자속은 V = 900 V로 계산하면 되나?",
        "코어는 그 코어에 감긴 권선의 전압을 적분한다: B(t) = B(0) + ∫v_w dt/(N A_e). DAB에서 직렬 L이 1차 쪽에 있으면 권선 전압은 반사된 2차 bridge 전압(n·V_L)이다. "
        "900 V/36 V(n = 50/3 → 600 V) 코너에서 버스 기준 0.18 T, 실제 0.12 T다. 직렬 L이 2차 쪽이면 반대로 1차 bridge 전압에 가깝다. 배치를 모르면 두 경우를 모두 확인한다.",
        "The DC bus is 900 V, so do you compute the core flux with 900 V?",
        "No. The core integrates the voltage across its own winding. In a DAB with the series inductance on the primary side that is the reflected secondary bridge voltage, 600 V at this corner, so the flux is 0.12 T rather than the 0.18 T a bus-based estimate gives. If the series inductance sits on the secondary side the picture reverses, so I check the actual placement.",
        ["권선 전압 적분", "직렬 L 배치", "반사 전압 n·V_L", "0.12 vs 0.18 T"],
    ),
    Question(
        "N_p = 50, A_e = 250 mm², 100 kHz, 권선 ±1000 V 사각파의 B_pk는?",
        "반주기에 V·T/2가 쌓여 ΔB_pp = V/(2 f N A_e), B_pk = V/(4 f N A_e) = 1000/(4·10⁵·50·250·10⁻⁶) = 0.20 T. 포화 여유는 B_pk에 DC bias·온도별 B_sat·재료를 더해 판단한다.",
        "N_p is 50, A_e is 250 mm², 100 kHz, ±1000 V square winding voltage. What is B_pk?",
        "Each half period adds V times T/2 of volt-seconds, so B_pk = V/(4 f N A_e) = 0.20 T and the peak-to-peak swing is 0.40 T. Saturation margin needs DC bias and the material's temperature-dependent saturation flux on top of that.",
        ["V/(4fNA_e)", "0.20 T", "ΔB_pp = 2B_pk", "DC bias·온도"],
        kind="calc",
    ),
    Question(
        "한쪽 pulse가 100 ns 길면 무엇이 문제이고 어떻게 확인하나?",
        "한 주기에 800 V×100 ns = 80 µVs가 남아 B가 0.0064 T/주기씩 이동한다(flux walking) — 수십 주기면 포화. 권선 R은 V_avg/R의 DC 전류에서야 멈추게 해 보호가 되지 못하고, blocking C·flux balance 제어·보호가 필요하다. "
        "측정에서는 적분 drift가 probe offset이나 표본화 오차일 수 있으므로 자화전류(또는 sense 권선)를 같은 timebase로 보고, 주기 평균 전압과 전류 평균 변화를 함께 확인한다.",
        "One pulse is 100 ns longer than the other. What goes wrong and how would you verify it?",
        "Each cycle leaves 80 microvolt-seconds, so the flux walks 0.0064 T per cycle and saturates within tens of cycles. Winding resistance only stops it at a DC current far beyond saturation; a blocking capacitor, flux-balance control or protection is needed. In the lab I would check the magnetizing current together with the integrated voltage, because a probe offset or sampling error also makes the integral drift.",
        ["0.0064 T/주기", "R은 보호 아님", "blocking C·제어", "자화전류로 구분"],
    ),
    Question(
        "1차 환산 200 µH는 2차 기준으로 얼마인가? C와 R은?",
        "n = N_p/N_s = 16.667. L = L′/n² = 0.72 µH, C′ = C/n² (2차의 C는 1차에서 n²배 작게 보인다), R′ = n²R. 전류는 I_s = n·I_p, 전력과 저장에너지는 같다. 회로도에 실제 값과 환산 값을 섞어 적지 않는다.",
        "What is 200 µH primary-referred on the secondary side? And C and R?",
        "With n of 16.667 it is 0.72 µH. Referred to the primary, L' = n²L, C' = C/n² and R' = n²R; the secondary current is n times the primary current and power and stored energy are unchanged. I never mix actual and referred values on one schematic.",
        ["0.72 µH", "C′ = C/n²", "R′ = n²R", "전력 보존"],
        kind="calc",
    ),
    Question(
        "“코어가 뜨겁다”는 고객에게 무엇부터 묻나?",
        "원인을 B(포화·DC bias)·core loss·winding loss·냉각으로 나눈다. core loss는 주파수·ΔB·waveform·온도에 달렸고 Steinmetz 계수는 B 정의(B_pk/ΔB)와 시험 waveform을 확인해야 한다(ΔB를 B_pk 자리에 넣으면 2^β배). "
        "주파수를 올린 뒤 core loss가 늘었을 수도 있고, 전류 RMS만 줄여 해결될 수도 있다. 온도 측정 위치(hotspot vs 표면)도 확인한다.",
        "A customer says the core runs hot. What do you ask first?",
        "I separate flux and saturation, core loss, winding loss and cooling. Core loss depends on frequency, flux swing, waveform and temperature, and Steinmetz coefficients must be used with their own flux definition and test waveform. A frequency increase may have raised the core loss, or a lower RMS current may fix a winding problem, so I also check where the temperature was measured.",
        ["B·core·winding·냉각 분리", "B 정의·waveform", "주파수 영향", "측정 위치"],
        kind="pressure",
    ),
    Question(
        "skin depth가 0.209 mm이니 0.4 mm 선이면 AC 손실 문제가 없나?",
        "고립 원형선의 skin 효과만 보면 d ≈ 2δ에서 R_ac/R_dc가 거의 1이지만, 다층 권선에서는 근접효과가 지배한다(Dowell: 같은 Δ에서도 층 수에 따라 F_R이 급증). 고조파(3·5차)에서는 δ가 √h배 작아진다. litz·foil 선택은 층 배치·고조파 전류·측정 R_ac(f)로 정한다.",
        "The skin depth is 0.209 mm, so a 0.4 mm wire has no AC-loss issue?",
        "Only for an isolated wire and the fundamental. In a multi-layer winding the proximity effect dominates and grows quickly with the number of layers, and harmonics see a smaller skin depth. The winding choice needs the layer arrangement, the harmonic currents and a measured AC resistance.",
        ["고립 도선만", "근접효과·층 수", "고조파", "측정 R_ac(f)"],
    ),
    Question(
        "L_m과 leakage, 외부 직렬 L은 각각 무슨 역할인가?",
        "L_m은 코어에 자속을 만드는 여자전류(부하와 무관한 삼각파)를 정하고, leakage는 두 권선에 완전히 결합되지 않은 자속을 회로로 나타낸 값이다. DAB에서는 leakage + 외부 직렬 L이 전력 전달을, LLC에서는 공진 L을 정한다. "
        "임의 transformer의 leakage가 원하는 L과 우연히 같다고 설계가 끝나지 않는다 — 측정 조건과 공차, 권선 배치를 확인한다.",
        "What are the roles of L_m, leakage and an external series inductor?",
        "L_m sets the magnetizing current that builds the core flux, independent of load. Leakage represents flux not shared by both windings; together with any external inductor it sets the power transfer of a DAB or the resonant inductance of an LLC. A leakage value that happens to match is not a design until its measurement conditions, tolerance and winding arrangement are known.",
        ["L_m = 여자전류", "leakage = 비결합 자속", "DAB 전력·LLC 공진", "측정 조건"],
    ),
]

_DAB_COMMON = [
    Param("Np", "1차 권선수 N_p", "", 50, "", vmin=1, vmax=2000, kind="int", source="TEXTBOOK", source_note="50", group="변압기"),
    Param("Ns", "2차 권선수 N_s", "", 3, "", vmin=1, vmax=2000, kind="int", source="TEXTBOOK", source_note="3 (n = 16.667, FL08)", group="변압기"),
    Param("Ae", "유효 단면적 A_e", "mm²", 250.0, "mm²", vmin=1.0, vmax=1e5, source="TEXTBOOK", source_note="250 mm²", group="변압기"),
    Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", source_note="100 kHz", group="회로"),
]

EXPERIMENTS = [
    Experiment(
        key="winding_flux",
        title="코어는 버스 전압이 아니라 권선 전압을 적분한다 (DAB T-모델)",
        goal=(
            "B(t) = B(0) + ∫v_w dt/(N A_e)를 실제 권선(자화) 전압으로 계산한다. N_p 50·A_e 250 mm²·100 kHz·권선 ±1000 V에서 B_pk 0.20 T를 정확 스위칭 해로 재현하고, "
            "직렬 L이 1차 쪽인 DAB에서는 코어가 1차 버스가 아니라 반사된 2차 bridge 전압을 본다는 것(900/36 V: 0.18 T가 아니라 0.12 T)과 자화전류가 1차 RMS에 주는 몫을 확인한다."
        ),
        params=[
            Param("V1", "1차 버스 V₁", "V", 1000.0, "V", vmin=1, vmax=3000, source="TEXTBOOK", source_note="±1000 V 권선 예제 (FL08 nominal 800 V)", group="회로"),
            Param("VL", "2차 버스 V_L (실제)", "V", 60.0, "V", vmin=0.1, vmax=1000, source="ASSUMED", source_note="n·V_L = 1000 V가 되도록 고른 값 (FL08은 48 V)", group="회로"),
        ] + _DAB_COMMON + [
            Param("L_tot", "직렬 L 합계 (1차 환산)", "H", 200e-6, "µH", vmin=1e-7, vmax=0.01, source="TEXTBOOK", source_note="FL08 200 µH", group="회로"),
            Param("k_split", "직렬 L의 1차측 비율 k", "", 1.0, "", vmin=0.001, vmax=1.0, source="ASSUMED", source_note="1 = 외부 L이 1차 쪽 (L₁ = 0은 불가: 실제 누설이 있다)", group="회로"),
            Param("Lm", "자화 인덕턴스 L_m (1차 환산)", "H", 2e-3, "mH", vmin=1e-5, vmax=10.0, source="ASSUMED", source_note="합성 값", group="변압기"),
            Param("R1", "1차 저항 R₁", "Ω", 0.05, "mΩ", vmin=1e-6, vmax=10.0, source="ASSUMED", group="회로"),
            Param("R2", "2차 저항 R₂′ (1차 환산)", "Ω", 0.05, "mΩ", vmin=1e-6, vmax=10.0, source="ASSUMED", group="회로"),
            Param("P", "목표 전력 (SPS φ 계산용)", "W", 1500.0, "W", vmin=1.0, vmax=1e5, source="TEXTBOOK", source_note="FL08 1.5 kW/모듈", group="회로"),
            Param("Bsat", "포화 자속 B_sat (가정)", "T", 0.35, "T", vmin=0.01, vmax=3.0, source="ASSUMED", source_note="재료·온도 자료 없음 (MISSING_INPUT)", group="변압기"),
        ],
        presets=[
            Preset("textbook", "권선 ±1000 V (N_p 50, A_e 250 mm², 100 kHz)", {}, "교재 10장 0.20 T", ("nominal", "reference")),
            Preset("fl08", "FL08 800 V ↔ 48 V", {"V1": 800.0, "VL": 48.0}, "0.16 T, I_rms 2.01988 A (L_m 없음)", ("variant", "reference")),
            Preset("mismatch", "900 V ↔ 36 V (n·V_L = 600 V)", {"V1": 900.0, "VL": 36.0}, "버스 0.18 T vs 실제 0.12 T", ("variant", "reference")),
            Preset("split", "900/36 V, 직렬 L 반반 (k = 0.5)", {"V1": 900.0, "VL": 36.0, "k_split": 0.5}, "4준위 권선 전압", ("variant",)),
            Preset("sec_L", "900/36 V, 직렬 L이 2차 쪽 (k = 0.02)", {"V1": 900.0, "VL": 36.0, "k_split": 0.02}, "코어가 1차 bridge를 본다", ("corner",)),
        ],
        run=run_winding_flux,
        model_level="C (정확 스위칭)",
        suggested_change="900 V ↔ 36 V 코너(mismatch)로 바꾼 뒤, 직렬 L 1차측 비율 k를 1 → 0.02로 바꾼다.",
        prediction=Prediction(
            "V₁ = 900 V, V_L = 36 V (n·V_L = 600 V), 직렬 L이 1차 쪽이면 코어 B_pk는?",
            ["900 V 기준 0.18 T", "600 V 기준 0.12 T", "두 전압 평균 0.15 T", "모르겠다"],
            "600 V 기준 0.12 T",
            "자화 가지가 2차 쪽 직렬 L 없이 2차 bridge에 바로 붙어 있으므로 권선 전압은 반사된 2차 전압 ±600 V다. 버스 900 V를 넣으면 50 % 과대평가한다. 직렬 L을 2차 쪽으로 옮기면(k → 0) 반대로 1차 전압 0.18 T에 가까워진다.",
            ["Bpk", "B_bus", "Vw_eq"],
            handcalc=[{"key": "Bpk", "label": "B_pk", "unit": "T"}, {"key": "B_bus", "label": "버스 기준 B", "unit": "T"}],
        ),
        suggested={"V1": 900.0, "VL": 36.0},
        student=(
            "코어의 자속은 권선에 걸린 전압을 시간으로 적분한 것이다(패러데이 법칙). 사각파 전압이면 자속은 삼각파가 되고, 반주기 동안 쌓인 volt-second가 클수록 자속 폭이 커진다. "
            "문제는 '어느 전압이 권선에 걸리는가'다. DAB처럼 직렬 인덕터가 있으면 버스 전압의 일부가 인덕터에 걸리므로, 코어는 버스가 아니라 권선 전압을 본다."
        ),
        expert=(
            "① T-모델: v₁ → L₁ → (v_m, L_m) → L₂′ → v₂′. v_m = [L₂(v₁ − R₁i₁)/L₁ + R₂i₂ + v₂]/(1 + L₂/L₁ + L₂/L_m) — k = 1이면 v_m = v₂′, k → 0이면 v_m ≈ v₁, 사이면 가중 평균(4준위). "
            "② B를 자화전류 상태와 v_m의 사다리꼴 적분 두 경로로 계산해 일치시킨다. ③ 자화전류는 부하와 무관한 삼각파라 DAB 순환전류와 구분된다(i₁ = i₂′ + i_m). "
            "④ L_m → ∞ 한계에서 FL08 폐형식 2.01988 A, 900/600 V에서 E06의 SPS 3.113563 A를 재현한다. ⑤ 정상상태는 반주기 반대칭(zero-DC) 해이며, R이 DC offset을 감쇠시킨다."
        ),
        customer_ko=(
            "코어 포화 여유는 DC 버스 전압이 아니라 실제 권선 전압으로 계산해야 합니다. 현재 구조처럼 직렬 인덕터가 1차 쪽이면 코어는 2차 반사 전압을 보므로 900 V 코너에서도 0.12 T 수준이고, "
            "반대로 누설이 2차 쪽에 몰려 있으면 1차 전압에 가까워집니다. 권선별 누설 분배와 온도별 B_sat 자료를 받아 확인하시죠."
        ),
        customer_en=(
            "The saturation margin must be computed from the actual winding voltage, not the DC bus. With the series inductor on the primary side, as in this design, the core sees the reflected secondary voltage, so even at the 900 V corner the flux is about 0.12 T. "
            "If the leakage sits mostly on the secondary side it moves towards the primary voltage instead. I'd like the leakage split and temperature-dependent saturation data to confirm."
        ),
        questions=[_Q[0], _Q[1], _Q[6]],
        circuit="fl07_dabT",
        textbook=[TB_10, TB_11],
        reference_presets=["textbook", "fl08", "mismatch"],
        claim_limit="선형 코어·이상 bridge의 자속·전류 파형 (C). 포화·코어 손실은 주장하지 않는다.",
    ),
    Experiment(
        key="flux_walk",
        title="펄스 비대칭: flux walking, 저항·blocking C, 그리고 측정 오차와 구분하기",
        goal=(
            "한쪽 pulse가 800 V × 100 ns 더 누적되면 주기당 0.0064 T씩 자속이 이동한다는 것을 정확 해로 보이고, 권선 저항만으로는 포화 전에 멈추지 않으며 blocking capacitor는 DC를 막는 대신 공진 과도를 만든다는 것을 비교한다. "
            "적분 drift가 probe offset이나 표본화 오차일 수 있음을 실제 walk(자화전류 변화)와 신호로 구분한다."
        ),
        params=[
            Param("V", "권선 구동 전압 ±V", "V", 800.0, "V", vmin=1, vmax=3000, source="TEXTBOOK", source_note="800 V", group="구동"),
            Param("dt", "비대칭 Δt (한쪽이 더 긴 시간)", "s", 100e-9, "ns", vmin=0.0, vmax=5e-6, source="TEXTBOOK", source_note="100 ns", group="구동"),
            Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", group="구동"),
            Param("N", "권선수 N", "", 50, "", vmin=1, vmax=2000, kind="int", source="TEXTBOOK", group="코어"),
            Param("Ae", "유효 단면적 A_e", "mm²", 250.0, "mm²", vmin=1.0, vmax=1e5, source="TEXTBOOK", group="코어"),
            Param("Lm", "자화 인덕턴스 L_m", "H", 2e-3, "mH", vmin=1e-5, vmax=10.0, source="ASSUMED", group="코어"),
            Param("bound", "DC 제한 요소", "", "none", kind="choice", choices=[("none", "없음 (이상: 저항 0)"), ("R", "권선·채널 저항 R만"), ("Cb", "R + 직렬 blocking C_b")], source="ASSUMED", group="구동"),
            Param("R", "직렬 저항 R", "Ω", 0.1, "mΩ", vmin=1e-6, vmax=100.0, source="ASSUMED", group="구동"),
            Param("Cb", "blocking capacitor C_b", "F", 10e-6, "µF", vmin=1e-9, vmax=1.0, source="ASSUMED", group="구동"),
            Param("cycles", "시뮬레이션 주기 수", "", 300, "", vmin=3, vmax=3000, kind="int", source="ASSUMED", group="시뮬레이션"),
            Param("Bsat", "포화 자속 B_sat (가정)", "T", 0.35, "T", vmin=0.01, vmax=3.0, source="ASSUMED", source_note="재료 자료 없음", group="코어"),
            Param("V_os", "probe offset (측정 예)", "V", 0.5, "V", vmin=-100.0, vmax=100.0, source="ASSUMED", group="측정"),
            Param("f_samp", "표본화 주파수 (측정 예)", "Hz", 16.5e6, "MHz", vmin=1e5, vmax=1e10, source="ASSUMED", source_note="T·f_s = 165 (홀수): 동기 표본화의 최악 예", group="측정"),
        ],
        presets=[
            Preset("textbook", "800 V × 100 ns, 저항 0 (이상)", {}, "교재 0.0064 T/주기", ("nominal", "reference")),
            Preset("R_only", "R 0.1 Ω만", {"bound": "R"}, "R은 포화 전에 못 막는다", ("failure", "reference")),
            Preset("blocking", "R + C_b 10 µF", {"bound": "Cb"}, "DC 차단, 공진 과도", ("variant", "reference")),
            Preset("symmetric", "대칭 구동 (Δt = 0)", {"dt": 0.0}, "자속 주기적", ("variant",)),
            Preset("even_sampling", "표본화 16 MS/s (T·f_s = 160)", {"f_samp": 16e6}, "표본화 drift 상쇄", ("variant",)),
        ],
        run=run_flux_walk,
        model_level="C (정확 스위칭)",
        suggested_change="DC 제한 요소를 '없음' → 'R만' → 'R + blocking C_b'로 바꿔 본다.",
        prediction=Prediction(
            "권선·채널 저항 0.1 Ω이 있으면 100 ns 비대칭의 자속 이동은?",
            ["포화 전에 멈춘다", "포화 시점이 거의 같다", "이동 방향이 바뀐다", "모르겠다"],
            "포화 시점이 거의 같다",
            "평균 전압 8 V가 만드는 DC 전류는 V_avg/R = 80 A까지 가야 멈추는데(시정수 L_m/R = 20 ms), 그 자속은 12.8 T로 포화보다 훨씬 크다. 약 30주기(0.3 ms) 안에 포화에 닿는 것은 R이 없을 때와 같다. DC를 막는 C_b나 제어가 필요하다.",
            ["walk", "t_sat", "Bdc_inf"],
            handcalc=[{"key": "walk", "label": "주기당 ΔB", "unit": "T"}, {"key": "Vavg", "label": "평균 전압", "unit": "V"}],
        ),
        suggested={"bound": "R"},
        student=(
            "코어의 자속은 권선 전압의 적분이라, +쪽과 −쪽 volt-second가 정확히 같아야 한 주기 뒤 제자리로 돌아온다. 한쪽이 조금이라도 길면 그 차이가 매 주기 쌓여 자속의 중심이 한쪽으로 걸어간다(flux walking). "
            "결국 코어가 포화해 자화 인덕턴스가 무너지고 전류가 급증한다."
        ),
        expert=(
            "① 주기당 ΔB = V·Δt/(N A_e) = 0.0064 T는 L_m과 무관하다. ② 권선 R은 1차 시정수 L_m/R로 DC 전류 V_avg/R에 수렴시키지만 그 자속이 포화보다 크면 보호가 아니다. "
            "③ blocking C_b는 DC를 떠안지만 L_m–C_b 직렬 공진(여기 1.1 kHz, Q ≈ 141)의 과도 offset(≈ L_m V_avg/(Z₀ N A_e))을 만든다 — 감쇠 설계가 따로 필요하다. "
            "④ 측정: probe offset은 V_os·T/(N A_e)/주기, edge가 표본 사이 다른 위치에 떨어지는 동기 표본화는 최대 V/(f_s N A_e)/주기의 가짜 drift를 만든다. 실제 walk는 자화전류 평균이 움직인다. 전압 평균을 빼는 보정은 실제 walk까지 지운다."
        ),
        customer_ko=(
            "gate 타이밍 비대칭 100 ns만으로도 주기당 0.0064 T씩 자속이 이동해 수십 주기 안에 포화할 수 있습니다. 권선 저항은 이를 막지 못하니 blocking capacitor나 flux balance 제어가 필요합니다. "
            "측정은 적분한 전압만 보지 말고 자화전류를 같은 timebase로 함께 보셔야 probe offset과 실제 walk를 구분할 수 있습니다."
        ),
        customer_en=(
            "A 100 ns timing asymmetry alone walks the flux by 0.0064 T per cycle, which can saturate the core within tens of cycles. Winding resistance does not prevent it, so a blocking capacitor or flux-balance control is needed. "
            "When you measure it, record the magnetizing current on the same timebase as the integrated voltage so that a probe offset is not mistaken for a real walk."
        ),
        questions=[_Q[2]],
        circuit="fl07_walk",
        textbook=[TB_10],
        reference_presets=["textbook", "R_only", "blocking"],
        claim_limit="선형 코어의 자속 이동과 측정 오차 신호 (C). 포화 후 전류·제어 성능은 주장하지 않는다.",
    ),
    Experiment(
        key="loss_screen",
        title="L_m·누설·환산과 손실 screen: 합성 계수로는 정밀 core loss를 만들지 않는다",
        goal=(
            "n = 16.667에서 1차 환산 200 µH = 2차 0.72 µH를 실제 값 회로와 환산 회로를 따로 풀어 확인하고(L′ = n²L, C′ = C/n², R′ = n²R), 구리 skin depth 0.209 mm를 확산 방정식 수치해와 비교한다. "
            "원형선·Dowell 근접효과와 SYNTHETIC Steinmetz/iGSE core-loss screen으로 B 정의·waveform 함정을 보고, 정밀 core loss는 MISSING_INPUT으로 남기며 공급사 요청 목록을 만든다."
        ),
        params=[
            Param("Np", "1차 권선수 N_p", "", 50, "", vmin=1, vmax=2000, kind="int", source="TEXTBOOK", group="변압기"),
            Param("Ns", "2차 권선수 N_s", "", 3, "", vmin=1, vmax=2000, kind="int", source="TEXTBOOK", group="변압기"),
            Param("L_ref", "직렬 L (1차 환산)", "H", 200e-6, "µH", vmin=1e-7, vmax=0.01, source="TEXTBOOK", source_note="FL08 200 µH", group="환산"),
            Param("V1", "1차 버스", "V", 800.0, "V", vmin=1, vmax=3000, source="TEXTBOOK", group="환산"),
            Param("VL", "2차 버스 (실제)", "V", 48.0, "V", vmin=0.1, vmax=1000, source="TEXTBOOK", group="환산"),
            Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", group="환산"),
            Param("P", "목표 전력 (φ 계산)", "W", 1500.0, "W", vmin=1.0, vmax=1e5, source="TEXTBOOK", group="환산"),
            Param("R_s", "2차 직렬 R (실제)", "Ω", 1e-3, "mΩ", vmin=1e-7, vmax=10.0, source="ASSUMED", source_note="환산 검증용", group="환산"),
            Param("C_s", "2차 직렬 blocking C (실제)", "F", 50e-6, "µF", vmin=1e-9, vmax=1.0, source="ASSUMED", source_note="환산 검증용", group="환산"),
            Param("rho", "구리 저항률 ρ (20 °C)", "Ω·m", 1.72e-8, "Ω·m", vmin=1e-9, vmax=1e-6, source="TEXTBOOK", source_note="1.72×10⁻⁸ Ωm", group="권선"),
            Param("h_foil", "foil(등가) 두께 h", "m", 0.1e-3, "mm", vmin=1e-6, vmax=0.01, source="ASSUMED", group="권선"),
            Param("layers", "권선 층 수 m", "", 4, "", vmin=1, vmax=40, kind="int", source="ASSUMED", group="권선"),
            Param("k_st", "Steinmetz k (SYNTHETIC)", "", 4.0, "", vmin=1e-6, vmax=1e6, source="ASSUMED", source_note="SYNTHETIC: W/m³ (f in Hz, B in T), 재료 fit 아님", group="코어 손실"),
            Param("alpha", "Steinmetz α (SYNTHETIC)", "", 1.4, "", vmin=1.0, vmax=3.0, source="ASSUMED", source_note="SYNTHETIC", group="코어 손실"),
            Param("beta", "Steinmetz β (SYNTHETIC)", "", 2.6, "", vmin=1.5, vmax=4.0, source="ASSUMED", source_note="SYNTHETIC", group="코어 손실"),
            Param("Ve", "코어 체적 V_e [cm³]", "", 20.0, "", vmin=0.01, vmax=1e5, source="ASSUMED", group="코어 손실"),
            Param("Ae", "유효 단면적 A_e", "mm²", 250.0, "mm²", vmin=1.0, vmax=1e5, source="TEXTBOOK", group="코어 손실"),
            Param("k_split", "직렬 L의 1차측 비율 (waveform)", "", 1.0, "", vmin=0.001, vmax=1.0, source="ASSUMED", source_note="1 = 삼각 자속, 0.5 = 4준위", group="코어 손실"),
            Param("Lm", "자화 인덕턴스 L_m", "H", 2e-3, "mH", vmin=1e-5, vmax=10.0, source="ASSUMED", group="코어 손실"),
        ],
        presets=[
            Preset("nominal", "FL08 DAB, 4층 0.1 mm foil", {}, "교재 0.72 µH·0.209 mm", ("nominal", "reference")),
            Preset("split", "4준위 자속 (k = 0.5, 900/36 V)", {"k_split": 0.5, "V1": 900.0, "VL": 36.0}, "iGSE waveform 효과", ("variant", "reference")),
            Preset("thick", "foil 0.3 mm, 8층", {"h_foil": 0.3e-3, "layers": 8}, "근접효과 지배", ("corner",)),
        ],
        run=run_loss_screen,
        model_level="A (+ C 환산 검증)",
        suggested_change="foil 두께 0.1 → 0.3 mm, 층 수 4 → 8로 바꾼다.",
        prediction=Prediction(
            "foil을 0.1 → 0.3 mm로 두껍게(DC 저항 1/3) 하고 층을 8층으로 하면 100 kHz AC 저항 배율 F_R은?",
            ["DC 저항이 줄어 F_R도 준다", "F_R이 크게 늘어 AC 손실이 오히려 늘 수 있다", "변화 없다", "모르겠다"],
            "F_R이 크게 늘어 AC 손실이 오히려 늘 수 있다",
            "Δ = h/δ가 0.48 → 1.44로 커지고 층 수가 늘면 근접효과 항 (2(m²−1)/3)·(sinh Δ − sin Δ)/(cosh Δ + cos Δ)이 급증한다. DC 저항이 1/3이 되어도 F_R이 그 이상 커지면 AC 손실은 늘어난다.",
            ["FR", "Delta"],
            handcalc=[{"key": "delta", "label": "skin depth", "unit": "m"}, {"key": "Ls", "label": "2차 기준 L", "unit": "H"}],
        ),
        suggested={"h_foil": 0.3e-3, "layers": 8},
        student=(
            "변압기의 값은 어느 쪽에서 보느냐에 따라 달라 보인다. 2차의 작은 인덕턴스는 1차에서 n²배 크게, 2차의 커패시턴스는 n²배 작게 보인다. "
            "권선 손실은 교류일수록 전류가 도체 표면과 이웃 도체 쪽으로 몰려 커지고, 코어 손실은 재료·주파수·자속 폭·파형·온도에 따라 달라서 공식 하나로 정확히 계산할 수 없다."
        ),
        expert=(
            "① 환산 검증: 2차 실제 값(0.72 µH, R_s, C_s, 48 V bridge)과 1차 환산(200 µH, n²R_s, C_s/n², 800 V)을 각각 정확 해로 풀어 전력·손실·저장에너지·I_s = n·I_p가 같음을 보인다. "
            "② δ = √(ρ/πfμ₀) = 0.209 mm를 1-D 확산 방정식 유한차분 해로 확인한다. 원형선 R_ac/R_dc(Bessel)는 d ≈ 2δ까지 거의 1이지만 다층 권선은 Dowell 근접효과가 지배한다. "
            "③ Steinmetz P = k f^α B^β V_e는 계수의 B 정의(B_pk vs ΔB)와 시험 waveform이 맞아야 한다. 같은 B_pk에서 사각 전압(삼각 자속)의 iGSE는 사인의 약 0.93배, ΔB를 B_pk 자리에 넣으면 2^β ≈ 6배 과대. "
            "④ 여기 계수는 SYNTHETIC이라 정밀 core loss는 MISSING_INPUT — 재료 곡선·온도·DC bias·minor loop 자료가 필요하다."
        ),
        customer_ko=(
            "누설 200 µH는 1차 환산 값이라 2차 기준으로는 0.72 µH이고, 측정 조건(주파수·진폭·fixture·다른 권선 연결)이 붙어야 비교할 수 있습니다. 코어 손실은 재료의 온도별 손실 곡선과 실제 파형으로 계산해야 하니 "
            "재료 자료와 R_ac(f) 측정값을 요청드립니다. 권선은 층 배치에 따라 근접효과가 커질 수 있어 두께만으로 판단하지 않겠습니다."
        ),
        customer_en=(
            "The 200 µH leakage is a primary-referred value, 0.72 µH on the secondary side, and it needs its measurement conditions to be comparable. Core loss has to come from the material's temperature-dependent loss curves and the real waveform, "
            "so I'd like the material data and a measured AC-resistance sweep. The winding's proximity effect depends on the layer arrangement, so thickness alone does not settle it."
        ),
        questions=[_Q[3], _Q[4], _Q[5]],
        circuit=None,
        textbook=[TB_10],
        reference_presets=["nominal", "split"],
        claim_limit="환산 규칙·skin/Dowell·합성 core-loss screen (A). 정밀 core loss·R_ac는 MISSING_INPUT.",
    ),
]

LAB = Lab(
    id="FL07",
    title="자성체 — 컨버터 전문가로 가는 실제 관문",
    title_en="Magnetics: integrate the actual winding voltage",
    track="basic",
    order=7,
    path_note="14일 경로 6일차 (10 자기부품)",
    textbook=[TB_10, TB_11],
    prerequisites=["FL01"],
    summary="권선 전압 적분 → 직렬 L 배치에 따른 코어 전압 → flux walking과 측정 오차 → 환산 규칙·skin/근접효과·합성 core-loss screen. 재료 자료 없이 정밀 손실을 만들지 않는다.",
    experiments=EXPERIMENTS,
    minimum_scope="flux integrator·L_m/leakage·손실 screen; winding B·0.20 T·pulse bias (교재 19장 표); 0.0064 T/주기, 0.72 µH, 0.209 mm",
    claim_limits=[
        "선형 코어 (포화·히스테리시스 없음)",
        "Steinmetz 계수는 SYNTHETIC — 정밀 core loss는 MISSING_INPUT",
        "Dowell 1-D 근사·고립 원형선 screen",
        "B_sat·L_m·R·C_b는 합성 가정",
    ],
    test_paths=["tests/test_fl07.py"],
)
