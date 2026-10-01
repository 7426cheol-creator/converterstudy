"""FL03 - Loss, temperature and lifetime: the numbers must agree (textbook ch.06).

Four experiments:
  A/C  single thermal node: the 100 W step (10 s = 77.642411 C, final 85 C) on the exact
       switched-affine engine, a hand-written RK4 path, and the same average loss applied as
       fast and slow pulses;
  A/B  electro-thermal feedback P = P0 [1 + alpha (T - T_ref)]: closed form, fixed-point
       iteration, exact transient with coolant 65 -> 85 C and a 30 s overload, the
       NO_STABLE_FIXED_POINT case, the loss-model validity limit, and an explicit-Euler
       step that diverges numerically while the physics is stable;
  A    per-event loss of one inverter-leg MOSFET from a synthetic E(I, T) / R(T) / V_f(I)
       map with channel, body-diode dead-time and recovery shares kept apart
       (postprocessed loss estimate), out-of-range points flagged instead of extrapolated;
  A/B  Foster vs Cauer: a physical Cauer ladder, its exact Foster form and a fitted one,
       the WRONG practice of attaching a heatsink to a Foster node, and the Foster -> Cauer
       conversion that restores a physical connection.
Thermal networks are linear RC circuits and are integrated exactly by engine/switched.py
(matrix exponential per segment).  Closed forms for comparison: reference/thermal.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, least_squares

from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ._common import energy_ledger, ledger_check, sym_linear

TB_06 = TextbookRef("손실온도수명-숫자가-서로-맞아야-한다-fl03", "06. 손실·온도·수명 [FL03]")
TB_E08 = TextbookRef("expert-e08-전열-연성-최고온도-한-점에서-mission으로", "E08 · 전열 연성")

# ======================================================================================
# Exact thermal RC network (node capacitances to thermal ground)
# ======================================================================================


class NodeNetwork(HybridSystem):
    """C dT/dt = -L T + G_b (T_b - T) + P(T), P_k = P0_k (1 + alpha_k (T_k - T_ref)).

    Node k has heat capacity C_k (J/K) to thermal ground; ``G`` holds node-to-node
    conductances (W/K, symmetric, off-diagonal) and ``Gb`` node-to-boundary conductances.
    ``segments`` is a time-ordered list of (t_start, P0 (n,), T_b (nb,)); the discrete
    state is the segment index.  Loss feedback keeps every segment affine, so the engine
    integrates it exactly, including unstable (runaway) segments.
    """

    def __init__(self, C, G, Gb, segments, alpha=None, Tref=0.0, names=None):
        self.C = np.asarray(C, dtype=float)
        n = self.C.size
        self.n = n
        self.G = np.asarray(G, dtype=float).reshape(n, n)
        self.Gb = np.asarray(Gb, dtype=float).reshape(n, -1)
        self.nb = self.Gb.shape[1]
        self.alpha = np.zeros(n) if alpha is None else np.asarray(alpha, dtype=float) * np.ones(n)
        self.Tref = float(Tref)
        self.segments = [(float(t0), np.asarray(P0, dtype=float) * np.ones(n), np.asarray(Tb, dtype=float) * np.ones(self.nb)) for t0, P0, Tb in segments]
        self.names = names or [f"T{k}" for k in range(n)]
        self.state_names = tuple(self.names)
        self.state_units = tuple("°C" for _ in range(n))
        G0 = self.G.copy()
        np.fill_diagonal(G0, 0.0)
        self.L = np.diag(G0.sum(axis=1) + self.Gb.sum(axis=1)) - G0
        self._modes: dict = {}

    def mode(self, q) -> AffineMode:
        m = self._modes.get(q)
        if m is None:
            _, P0, Tb = self.segments[q]
            A = (-self.L + np.diag(self.alpha * P0)) / self.C[:, None]
            b = (P0 * (1.0 - self.alpha * self.Tref) + self.Gb @ Tb) / self.C
            key = f"nn|{hash((A.round(15).tobytes(), b.round(12).tobytes()))}"
            m = AffineMode(key, A, b, label=str(q))
            self._modes[q] = m
        return m

    def gate_schedule(self, t0, t1):
        ev = []
        for k in range(1, len(self.segments)):
            ts = self.segments[k][0]
            if t0 - 1e-15 <= ts < t1:
                ev.append((ts, lambda q, k=k: k))
        return ev

    def _loss_row(self, q, k):
        _, P0, _ = self.segments[q]
        c = np.zeros(self.n + 1)
        c[k] = self.alpha[k] * P0[k]
        c[self.n] = P0[k] * (1.0 - self.alpha[k] * self.Tref)
        return c

    def _out_row(self, q):
        _, _, Tb = self.segments[q]
        c = np.zeros(self.n + 1)
        c[: self.n] = self.Gb.sum(axis=1)
        c[self.n] = -float(np.sum(self.Gb @ Tb))
        return c

    def outputs(self, q):
        out = {}
        for k, nm in enumerate(self.names):
            e = np.zeros(self.n + 1)
            e[k] = 1.0
            out[nm] = e
            out[f"P_{nm}"] = self._loss_row(q, k)
        out["q_out"] = self._out_row(q)
        _, _, Tb = self.segments[q]
        for j in range(self.nb):
            e = np.zeros(self.n + 1)
            e[self.n] = Tb[j]
            out[f"Tb{j}"] = e
        return out

    def stored_energy(self):
        W = np.zeros((self.n + 1, self.n + 1))
        W[: self.n, self.n] = self.C
        W[self.n, : self.n] = self.C
        return W  # 1/2 z^T W z = sum C_k T_k (heat content above 0 C)

    def powers(self, q):
        pin = sum(self._loss_row(q, k) for k in range(self.n))
        return {"p_in": sym_linear(pin, 1.0), "p_out": sym_linear(self._out_row(q), 1.0)}

    def describe(self, q):
        return str(q)

    def steady(self, q) -> np.ndarray:
        """Algebraic steady state of segment q (may be unstable or non-physical if feedback is too strong)."""
        m = self.mode(q)
        return np.linalg.solve(m.A, -m.b)


def values_at(traj, times, name) -> np.ndarray:
    """Exact output values at arbitrary times (propagated from the enclosing segment start)."""
    from ..engine.switched import propagator

    out = np.empty(len(times))
    segs = traj.segments
    j = 0
    for k, t in enumerate(times):
        while j < len(segs) - 1 and t > segs[j].t1:
            j += 1
        s = segs[j]
        z = propagator(s.mode, max(t - s.t0, 0.0)) @ s.z0
        out[k] = float(traj.system.outputs(s.q)[name] @ z)
    return out


def rk4_scalar(f, y0, t0, t1, h):
    """Hand-written classic RK4 for a scalar ODE (independent of the engine)."""
    n = max(1, int(round((t1 - t0) / h)))
    h = (t1 - t0) / n
    y, t = y0, t0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def _rc_circuit(v: dict, title: str = "단일 열노드 (열 등가회로)", controlled: bool = False) -> Circuit:
    c = Circuit("thermal_rc", 640, 250, title=title)
    ps = c.add("isource", "P", 90, 140, 270, "P(T_j)" if controlled else "P", "T_j 의존 손실" if controlled else f"{v.get('P', v.get('P0', 0)):g} W", lpos=(62, 144, "end"))
    cth = c.add("capacitor", "Cth", 220, 150, 90, "C_th = τ/R", f"{v['tau'] / v['Rth']:.4g} J/K", lpos=(236, 154, "start"))
    rth = c.add("resistor", "Rth", 330, 60, 0, "R_th", f"{v['Rth']:g} K/W", lpos=(330, 36, "middle"))
    tb = c.add("vsource", "Tb", 450, 150, 90, "T_b (냉각 경계)", "", lpos=(468, 154, "start"))
    c.add("ground", "g", 220, 230)
    c.wire("w_p_top", ps["b"], (90, 60), (220, 60))
    c.wire("w_c_top", (220, 60), cth["a"])
    c.wire("w_r_l", (220, 60), rth["a"])
    c.wire("w_r_r", rth["b"], (450, 60), tb["a"])
    c.wire("w_gnd", ps["a"], (90, 220), (450, 220), tb["b"])
    c.wire("w_c_gnd", cth["b"], (220, 220))
    c.dot((220, 60), (220, 220))
    c.text(220, 48, "T_j", "node")
    c.probe("pq", "q_R", 410, 60, "right", "열류 q")
    c.mode("on", "손실 인가", ["P", "w_p_top", "w_c_top", "Cth", "w_r_l", "Rth", "w_r_r", "Tb", "w_gnd", "w_c_gnd"], "손실이 C_th를 데우고 R_th로 냉각 경계에 흐른다", dim=[])
    c.mode("off", "손실 없음", ["Cth", "w_c_top", "w_r_l", "Rth", "w_r_r", "Tb", "w_gnd", "w_c_gnd"], "C_th에 저장된 열이 R_th로 빠져나간다", dim=["P"])
    c.notes.append("열 등가: 전류 = 열류[W], 전압 = 온도[°C], 저항 = R_th[K/W], 커패시턴스 = C_th[J/K]")
    return c


# ======================================================================================
# Experiment 1: 100 W step and pulses
# ======================================================================================


def run_rc_step(v: dict) -> Result:
    res = Result("FL03", "rc_step", "A (해석해) + C (정확 선형 적분)")
    P, R, tau, Tb, te = v["P"], v["Rth"], v["tau"], v["Tb"], v["t_eval"]
    C = tau / R
    t_end = max(6 * tau, 2 * te)
    net = NodeNetwork([C], [[0.0]], [[1.0 / R]], [(0.0, [P], [Tb])], names=["Tj"])
    tr = simulate(net, 0, [Tb], 0.0, t_end)
    T_te = float(values_at(tr, [te], "Tj")[0])
    T_cf = Tb + P * R * (1.0 - math.exp(-te / tau))
    tb_ = abs(P - 100) < 1e-12 and abs(R - 0.2) < 1e-15 and abs(tau - 10) < 1e-12 and abs(Tb - 65) < 1e-12 and abs(te - 10) < 1e-12
    res.add_metric("T_te", f"T_j({te:g} s) (정확 적분)", T_te, "°C", ref=77.642411 if tb_ else T_cf, ref_label="교재 77.642411 °C" if tb_ else "T_b + P·R(1 − e^(−t/τ))", tol=1e-8, basis="과도 해")
    T_fin = Tb + P * R
    res.add_metric("T_final", "최종 T_j (정상상태)", float(net.steady(0)[0]), "°C", ref=85.0 if tb_ else T_fin, ref_label="교재 85 °C" if tb_ else "T_b + P·R", tol=1e-12, basis="t → ∞: 상태행렬의 평형점 −A⁻¹b")
    res.add_metric("T_wrong", f"정상상태 식을 {te:g} s에 쓴 값 (오류)", T_fin, "°C", note=f"과도 해보다 {T_fin - T_te:.4g} K 높다 — 과도와 정상상태 혼동")
    res.add_metric("frac", f"{te:g} s의 정상 상승 비율 1 − e^(−t/τ)", (T_te - Tb) / (T_fin - Tb), "", basis="t = τ이면 63.2 %")
    res.add_metric("t99", "99 % 도달 시간 (4.6 τ)", tau * math.log(100.0), "s")
    res.add_metric("Cth", "C_th = τ / R_th", C, "J/K", basis="열용량")
    # independent path: hand-written RK4 with step halving
    f = lambda t, T: (P - (T - Tb) / R) / C  # noqa: E731
    errs = []
    for n in (20, 40, 80):
        errs.append(abs(rk4_scalar(f, Tb, 0.0, te, te / n) - T_te))
    order = math.log(errs[0] / errs[1], 2) if errs[1] > 0 else float("inf")
    res.add_check(Check("독립 경로: 손으로 쓴 RK4 (h 반감)", "PASS" if errs[-1] < 1e-7 and errs[-1] < errs[0] else "FAIL", errs[-1], "K", 1e-7, path="RK4 스칼라 적분(h = t/20, t/40, t/80) vs 행렬지수 정확 해", independent=True, detail="오차 " + ", ".join(f"{e:.2e}" for e in errs) + f" K — 수렴 차수 ≈ {order:.2f} (4차 기대)"))
    res.add_check(check_close("해석해 vs 정확 적분", T_te, T_cf, 1e-12, "닫힌 식 T_b + PR(1 − e^(−t/τ)) vs 엔진 expm", True, "°C", abs_scale=abs(Tb) + 1))
    led = energy_ledger(tr, net, 0.0, te, ["p_in"], ["p_out"], [], rated_power=P)
    res.add_check(ledger_check(led, 1e-6, what="열 원장: "))
    # step response and Z_th
    ts = np.linspace(0, t_end, 400)
    Tj = values_at(tr, ts, "Tj")
    res.add_series("Tj", "T_j(t) 정확 해", "°C", ts.tolist(), Tj.tolist())
    res.add_series("Tss", "정상상태 식 (시간 무관)", "°C", [0.0, t_end], [T_fin, T_fin], dash=True)
    res.add_series("q_R", "열류 (T_j−T_b)/R", "W", ts.tolist(), ((Tj - Tb) / R).tolist())
    res.add_plot("p_step", f"{P:g} W 계단: {te:g} s의 온도와 최종 온도는 다르다", ["Tj", "Tss"], y_label="T_j", y_unit="°C", level="C", group="step",
                 bands=[{"x0": 0.0, "x1": t_end, "mode": "on", "label": "손실 인가"}],
                 vlines=[{"x": te, "label": f"{te:g} s: {T_te:.6g} °C"}, {"x": tau, "label": "τ"}], markers=[{"x": te, "y": T_te, "label": f"{T_te:.6g} °C"}],
                 proved=f"{te:g} s의 T_j는 {T_te:.6g} °C이고 {T_fin:g} °C는 t → ∞의 값이다. 정상상태 식을 과도 시점에 쓰면 틀린다.",
                 not_yet="단일 RC는 한 시정수의 근사다. 실제 소자는 여러 층(die·solder·substrate·방열판)의 시정수를 가진다 (실험 4).")
    res.add_plot("p_q", "냉각 경계로 나가는 열류", ["q_R"], y_label="열류", y_unit="W", level="C", group="step", bands=[{"x0": 0.0, "x1": t_end, "mode": "on", "label": "손실 인가"}],
                 proved="초기에는 손실 대부분이 C_th를 데우는 데 쓰이고, 시간이 지나야 경계로 나가는 열이 손실과 같아진다.", not_yet="열 원장(잔차 < 1e-6)이 확인하는 것은 이 단일 노드 안의 손실 = 저장 증가 + 경계 열류뿐이다. 냉각 경계를 고정 온도 T_b로 두었으므로 방열판·냉각수의 온도 상승과 층별 열 분배는 이 그림에 없다(실험 4).")
    tz = np.geomspace(1e-3 * tau, 20 * tau, 200)
    res.add_series("Zth", "Z_th(t) = R(1 − e^(−t/τ))", "K/W", tz.tolist(), (R * (1 - np.exp(-tz / tau))).tolist())
    res.add_plot("p_zth", "열 임피던스 Z_th(t) (로그 시간)", ["Zth"], x_label="t", x_unit="s", y_label="Z_th", y_unit="K/W", kind="xy", log_x=True, level="A",
                 markers=[{"x": te, "y": R * (1 - math.exp(-te / tau)), "label": f"{te:g} s"}], proved="짧은 펄스는 R_th가 아니라 Z_th(t_pulse)만큼만 온도를 올린다.", not_yet="곡선은 닫힌 식이며 같은 식이 평가 시점에서 행렬지수 적분과 1e-12로 일치한다. 단일 시정수의 Z_th라서, 여러 층의 시정수가 섞이는 실제 소자의 짧은 펄스 영역과 반복 펄스의 중첩은 다루지 않는다(실험 4).")
    # same average loss, different pulse lengths
    D = v["duty"]
    t_p = max(12 * tau, 3 * v["T_slow"])
    rows = []
    for key, lab, per in (("const", f"일정 {D * P:g} W (같은 평균)", None), ("fast", f"{P:g} W 펄스, 주기 {v['T_fast']:g} s", v["T_fast"]), ("slow", f"{P:g} W 펄스, 주기 {v['T_slow']:g} s", v["T_slow"])):
        if per is None:
            segs = [(0.0, [D * P], [Tb])]
        else:
            segs = []
            k = 0
            while k * per < t_p:
                segs.append((k * per, [P], [Tb]))
                segs.append((k * per + D * per, [0.0], [Tb]))
                k += 1
        nk = NodeNetwork([C], [[0.0]], [[1.0 / R]], segs, names=["Tj"])
        tk = simulate(nk, 0, [Tb], 0.0, t_p)
        smp = tk.sample(["Tj"], per_segment=40 if per is None else 12)
        res.add_series(f"pl_{key}", lab, "°C", smp["t"], smp["Tj"])
        last = per if per else tau
        lo, hi = tk.extrema(t_p - 2 * last, t_p, "Tj")
        rows.append([lab, hi, lo, hi - lo])
        if key == "slow":
            slow_tr = tk
    pb = []
    for iv in slow_tr.mode_intervals(0.0, t_p):
        on = slow_tr.system.segments[int(iv["mode"])][1][0] > 0
        pb.append({"x0": iv["t0"], "x1": iv["t1"], "mode": "on" if on else "off", "label": "ON" if on else "OFF"})
    res.add_plot("p_pulse", f"같은 평균손실 {D * P:g} W, 다른 펄스 길이", ["pl_const", "pl_fast", "pl_slow"], y_label="T_j", y_unit="°C", level="C", group="pulse", bands=pb,
                 proved="평균손실이 같아도 펄스 주기가 열 시정수보다 길면 peak 온도와 온도 변동폭(ΔT_j)이 커진다 — 수명(열 사이클)에는 평균이 아니라 변동이 중요하다.",
                 not_yet="수명 모델은 계산하지 않았다 (EX08: cycle counting, 상대 proxy만).")
    res.tables.append(Table("t_pulse", "펄스 조건별 준정상 온도 (마지막 두 주기)", ["조건", "최고 T_j [°C]", "최저 T_j [°C]", "ΔT_j [K]"], rows, note="시정수보다 짧은 펄스는 평균처럼, 긴 펄스는 계단처럼 보인다."))
    res.circuit = {"diagram": _rc_circuit(v).to_json(), "intervals": pb, "plot_group": "pulse"}
    res.verdict("PASS_WITHIN_MODEL", f"T_j({te:g} s) = {T_te:.6f} °C가 해석해·독립 RK4·열 원장과 일치; 최종 {T_fin:g} °C는 과도가 끝난 뒤의 값")
    res.assumptions += ["단일 열노드 (R_th, C_th = τ/R_th), 경계 온도 T_b 일정", "손실 P는 온도와 무관한 계단 (전열 연성은 실험 2)", "펄스 비교: 같은 평균손실을 duty로 인가"]
    res.not_valid_for += ["여러 층의 열 경로 (실험 4)", "인접 소자 열 결합 (EX08)", "수명 예측"]
    res.interpretation = (
        f"C_th = τ/R_th = {C:.4g} J/K가 처음 열을 흡수하므로 온도는 지수적으로 오른다. {te:g} s = {te / tau:.3g}τ 에서는 최종 상승 {P * R:g} K의 {(T_te - Tb) / (P * R) * 100:.4g} %만 올라 "
        f"{T_te:.6g} °C다. {te:g} s 뒤에 {T_fin:g} °C라고 계산한 모델은 정상상태 식을 과도 시점에 쓴 것이다."
    )
    return res


# ======================================================================================
# Experiment 2: electro-thermal feedback
# ======================================================================================


def _et_segments(v: dict, alpha_on: bool = True):
    P0, Tb1, Tb2 = v["P0"], v["Tb"], v["Tb2"]
    t1, t2, d2 = v["t_cool"], v["t_ol"], v["d_ol"]
    k = v["k_ol"]
    segs = [(0.0, [P0], [Tb1]), (t1, [P0], [Tb2]), (t2, [k * P0], [Tb2]), (t2 + d2, [P0], [Tb2])]
    labels = [f"냉각수 {Tb1:g} °C", f"냉각수 {Tb2:g} °C", f"과부하 ×{k:g} ({d2:g} s)", "복귀"]
    return segs, labels


def run_electrothermal(v: dict) -> Result:
    res = Result("FL03", "electrothermal", "A (고정점) + C (정확 선형 과도)")
    P0, R, tau, a, Tref = v["P0"], v["Rth"], v["tau"], v["alpha"], v["Tref"]
    C = tau / R
    Tmax = v["T_valid"]
    segs, labels = _et_segments(v)
    t_end = v["t_end"]
    net = NodeNetwork([C], [[0.0]], [[1.0 / R]], segs, alpha=[a], Tref=Tref, names=["Tj"])
    net0 = NodeNetwork([C], [[0.0]], [[1.0 / R]], segs, alpha=[0.0], Tref=Tref, names=["Tj"])
    tr = simulate(net, 0, [v["Tb"]], 0.0, t_end)
    tr0 = simulate(net0, 0, [v["Tb"]], 0.0, t_end)
    # textbook fixed point at the first segment (T_b = T_ref form)
    g0 = a * R * P0
    tb_ = abs(P0 - 100) < 1e-12 and abs(R - 0.2) < 1e-15 and abs(v["Tb"] - Tref) < 1e-12
    rows = []
    no_fp = []
    for q, (lab, (t0, Pq, Tbq)) in enumerate(zip(labels, segs)):
        g = a * R * Pq[0]
        if g < 1.0:
            u = (Tbq[0] - Tref + R * Pq[0]) / (1.0 - g)
            Tfp = Tref + u
            rows.append([lab, g, 1.0 / (1.0 - g), Tfp, "안정 고정점"])
        else:
            no_fp.append(lab)
            rows.append([lab, g, "∞", "없음", "NO_STABLE_FIXED_POINT"])
    res.tables.append(Table("t_fp", "구간별 전열 고정점 (P = P₀[1 + α(T_j − T_ref)])", ["구간", "루프 이득 g = αR_thP", "민감도 1/(1 − g)", "고정점 T_j* [°C]", "판정"], rows,
                            note="g ≥ 1이면 이 단순 선형 모델에는 안정한 정상해가 없다. 실제 파손온도를 예측한 것이 아니다."))
    if g0 < 1:
        dT = R * P0 / (1 - g0) if tb_ else (v["Tb"] - Tref + R * P0) / (1 - g0) + Tref - v["Tb"]
        res.add_metric("dT_fp", "정상 온도상승 ΔT = R·P₀/(1 − αR·P₀)" if tb_ else "정상 온도상승 (T_b 기준)", dT, "K", basis=f"g = αR_thP₀ = {g0:.4g}", note="고정 손실이면 R·P₀")
        res.add_metric("Tj_fp", "첫 구간 고정점 T_j*", v["Tb"] + dT, "°C")
    else:
        res.add_metric("dT_fp", "정상 온도상승", "없음 (분모 ≤ 0)", "", basis=f"g = αR_thP₀ = {g0:.4g} ≥ 1")
    res.add_metric("g0", "루프 이득 g = α·R_th·P₀", g0, "", basis="1에 가까울수록 민감, ≥ 1이면 안정 해 없음")
    res.add_metric("dT_fixed", "고정 손실 온도상승 R·P₀", R * P0, "K", ref=20.0 if tb_ else None, ref_label="교재 20 K (85 °C)", tol=1e-12)
    # iteration history in the first segment
    its = [v["Tb"]]
    for _ in range(40):
        Tn = v["Tb"] + R * P0 * (1 + a * (its[-1] - Tref))
        its.append(Tn if math.isfinite(Tn) else float("nan"))
        if abs(Tn) > 1e6:
            break
    res.add_series("it", "고정점 반복 T_{k+1} = T_b + R·P(T_k)", "°C", list(range(len(its))), its, style="points")
    conv = g0 < 1 and abs(its[-1] - its[-2]) < 1e-9 * max(1.0, abs(its[-1]))
    if g0 < 1:
        Tfp0 = Tref + (v["Tb"] - Tref + R * P0) / (1 - g0)
        res.add_check(check_close("고정점: 반복 수렴값 vs 닫힌 식", its[-1], Tfp0, 1e-9, "단순 대입 반복(40회) vs (T_b − T_ref + R·P₀)/(1 − g) + T_ref", True, "°C"))
        # engine steady state of the first segment (long hold)
        net_s = NodeNetwork([C], [[0.0]], [[1.0 / R]], [segs[0]], alpha=[a], Tref=Tref, names=["Tj"])
        trs = simulate(net_s, 0, [v["Tb"]], 0.0, 60 * tau / max(1 - g0, 1e-3))
        res.add_check(check_close("고정점: 정확 과도의 장시간 값 vs 닫힌 식", float(trs.z_end[0]), Tfp0, 1e-8, "행렬지수 과도를 60τ_eff 적분한 끝값 vs 닫힌 식", True, "°C"))
    # transient: validity, peaks
    ts = np.linspace(0, t_end, 1200)
    Tc = values_at(tr, ts, "Tj")
    Tu = values_at(tr0, ts, "Tj")
    Pc = values_at(tr, ts, "P_Tj")
    Pu = values_at(tr0, ts, "P_Tj")
    over = np.nonzero(Tc > Tmax)[0]
    t_x = float(ts[over[0]]) if over.size else None
    if t_x is not None:
        k = over[0]
        t_x = float(brentq(lambda t: float(values_at(tr, [t], "Tj")[0]) - Tmax, ts[k - 1], ts[k])) if k > 0 else float(ts[0])
    keep = ts <= (t_x if t_x is not None else t_end)
    res.add_series("Tc", "T_j 전열 연성 (α)", "°C", ts[keep].tolist(), Tc[keep].tolist())
    res.add_series("Tu", "T_j 고정 손실 (α = 0)", "°C", ts.tolist(), Tu.tolist(), dash=True)
    res.add_series("Pc", "손실 P(T_j) 연성", "W", ts[keep].tolist(), Pc[keep].tolist())
    res.add_series("Pu", "손실 고정", "W", ts.tolist(), Pu.tolist(), dash=True)
    bands = []
    for q, (lab, (t0, _, _)) in enumerate(zip(labels, segs)):
        t1_ = segs[q + 1][0] if q + 1 < len(segs) else t_end
        bands.append({"x0": t0, "x1": min(t1_, t_end), "mode": "on", "label": lab})
    pk_c = float(np.max(Tc[keep]))
    pk_u = float(np.max(Tu))
    if t_x is None:
        res.add_metric("Tpk_c", "최고 T_j (연성)", pk_c, "°C", basis="과도 전체")
    else:
        res.add_metric("Tpk_c", "최고 T_j (연성)", f"> {Tmax:g} °C (적용범위 밖, 외삽 안 함)", "", basis=f"{t_x:.4g} s에 손실모델 적용범위 초과")
    res.add_metric("Tpk_u", "최고 T_j (고정 손실)", pk_u, "°C")
    res.add_metric("t_valid", "손실모델 적용범위 초과 시각", t_x if t_x is not None else float("nan"), "s", basis=f"T_j > {Tmax:g} °C 이후 궤적은 미지원")
    hl = [{"y": Tmax, "label": f"손실모델 적용범위 {Tmax:g} °C"}]
    res.add_plot("p_T", "전열 연성 과도: 냉각수 상승과 30 s 과부하", ["Tc", "Tu"], y_label="T_j", y_unit="°C", bands=bands, level="C", group="et", hlines=hl,
                 proved="손실이 온도와 함께 커지면 같은 조건에서 더 뜨겁고, 과부하 동안의 상승이 고정 손실보다 크다. 루프 이득이 1 이상인 구간에서는 온도가 발산한다.",
                 not_yet="α는 전체 손실에 한 선형 온도계수를 준 합성 가정이다. 실제 전도·스위칭 손실은 각자의 온도 의존성을 가진다. 적용범위 밖 궤적은 그리지 않았다.")
    res.add_plot("p_P", "손실 P(T_j)", ["Pc", "Pu"], y_label="P", y_unit="W", bands=bands, level="C", group="et",
                 proved="같은 전류라도 T_j가 오르면 R_DS(on)이 커져 손실이 늘어난다(되먹임).", not_yet=f"연성 손실 trace는 행렬지수 정확 적분의 출력이며 구간별 RK4 끝값(1e-7)과 열 원장(1e-6)으로 확인했다. 손실 곡선은 P₀[1 + α(T_j − T_ref)] 한 직선이라 R_DS(on)의 실제 온도 곡선이나 스위칭 손실의 다른 온도 의존성은 없고, 적용범위 {Tmax:g} °C 밖의 손실은 그리지 않았다.")
    res.add_plot("p_it", "고정점 반복의 수렴·발산", ["it"], x_label="반복 k", x_unit="", y_label="T_j", y_unit="°C", kind="xy", level="A",
                 proved="반복 오차는 매번 g배가 된다: g < 1이면 수렴, g ≥ 1이면 발산한다.", not_yet="선형 α 모델의 결과다. 비선형 손실 곡선에서는 국소 기울기가 g 역할을 한다.")
    gs = np.linspace(0, 0.95, 60)
    res.add_series("sens", "ΔT / (R·P₀) = 1/(1 − g)", "", gs.tolist(), (1 / (1 - gs)).tolist())
    res.add_plot("p_sens", "루프 이득에 따른 온도 민감도", ["sens"], x_label="g = α·R_th·P", x_unit="", y_label="증폭", y_unit="", kind="xy", level="A",
                 markers=[{"x": g0, "y": 1 / (1 - g0), "label": f"g = {g0:.3g}"}] if g0 < 0.95 else [], vlines=[{"x": 1.0, "label": "g = 1"}],
                 proved="분모 1 − g가 작아질수록 같은 손실 오차가 온도에 크게 증폭된다.", not_yet="곡선은 닫힌 식 1/(1 − g)이며, g < 1인 현재 조건의 값은 고정점 반복과 정확 과도의 장시간 값이 닫힌 식과 일치해 확인된다. 단일 노드·선형 α의 증폭률이라 다층 열망이나 비선형 손실 곡선에서는 해 근처의 국소 기울기로 g를 다시 구해야 한다.")
    # numerical trap: explicit Euler with a user step on the same coupled transient
    dt = v["dt_euler"]
    T = v["Tb"]
    te_, Te = [0.0], [T]
    diverged = False
    t = 0.0
    while t < t_end - 1e-12:
        q = max(k for k in range(len(segs)) if segs[k][0] <= t + 1e-12)
        Pq, Tbq = segs[q][1][0], segs[q][2][0]
        dT = (Pq * (1 + a * (T - Tref)) - (T - Tbq) / R) / C
        T = T + min(dt, t_end - t) * dT
        t += min(dt, t_end - t)
        te_.append(t)
        Te.append(T)
        if not math.isfinite(T) or abs(T) > 1e4:
            diverged = True
            break
    Tex = values_at(tr, te_, "Tj")
    err_e = float(np.max(np.abs(np.array(Te) - Tex))) if not diverged else float("inf")
    res.add_series("euler", f"명시적 Euler Δt = {dt:g} s", "°C", te_, [x if abs(x) < 1e4 else float("nan") for x in Te], style="points")
    res.add_series("Tc_full", "정확 해", "°C", ts.tolist(), Tc.tolist(), dash=True)
    lam = max(abs(1 + dt * net.mode(q).A[0, 0]) for q in range(len(segs)))
    res.add_plot("p_euler", "수치 함정: 명시적 Euler의 시간 step", ["euler", "Tc_full"], y_label="T_j", y_unit="°C", level="C", group="eu",
                 proved=f"증폭률 |1 + Δt·A| = {lam:.3g}: 1보다 크면 물리는 안정해도 수치해가 진동·발산한다. 발산한 반복이 곧 열폭주는 아니다.",
                 not_yet="명시적 Euler를 같은 연성 과도의 정확 해(행렬지수)와 겹쳐 비교했다. 단일 노드라 증폭률이 |1 + Δt·A| 하나로 정해지며, 다노드 강성 열망에서는 가장 빠른 열 시정수가 step 한계를 정한다 — 암시적·가변 step 적분기는 비교하지 않았다.")
    # RK4 path on the coupled transient (independent of the engine): segment by segment
    Tk = v["Tb"]
    knots = [sg[0] for sg in segs] + [t_end]
    for q in range(len(segs)):
        a_, b_ = knots[q], min(knots[q + 1], t_end)
        if b_ <= a_:
            continue
        Pq, Tbq = segs[q][1][0], segs[q][2][0]
        Tk = rk4_scalar(lambda t, T, Pq=Pq, Tbq=Tbq: (Pq * (1 + a * (T - Tref)) - (T - Tbq) / R) / C, Tk, a_, b_, 0.02)
    Tend = float(tr.z_end[0])
    err_rk = abs(Tk - Tend) / max(1.0, abs(Tend - v["Tb"]))
    res.add_check(Check("독립 경로: RK4(h = 0.02 s) vs 정확 과도 끝값", "PASS" if err_rk < 1e-7 else "FAIL", err_rk, "rel", 1e-7, path="구간별 RK4 스칼라 적분 vs 구간별 행렬지수", independent=True, detail=f"RK4 {Tk:.9g} °C, 정확 {Tend:.9g} °C"))
    led = energy_ledger(tr, net, 0.0, t_end, ["p_in"], ["p_out"], [], rated_power=P0)
    res.add_check(ledger_check(led, 1e-6, what="열 원장 (연성 과도): "))
    methods = [["정확 선형 적분 (행렬지수)", "수렴", "기준"], ["RK4 h = 0.02 s", "수렴", f"끝값 상대오차 {err_rk:.1e}"],
               ["고정점 단순 대입 (첫 구간)", "수렴" if conv else "발산", f"오차 비율 g = {g0:.3g}"],
               [f"명시적 Euler Δt = {dt:g} s", "발산 (SOLVER_FAILED: 수치 방법)" if diverged or lam > 1 else "수렴", f"|1 + Δt·A| = {lam:.3g}" + ("" if diverged else f", 최대 오차 {err_e:.3g} K")]]
    res.tables.append(Table("t_methods", "해법별 결과 — 반복의 실패·발산도 결과다", ["해법", "결과", "근거"], methods,
                            note="명시적 Euler의 발산은 수치 방법의 실패(SOLVER_FAILED)이고, 고정점이 없는 것(NO_STABLE_FIXED_POINT)은 모델의 결과다. 둘을 같은 빨간색으로 합치지 않는다."))
    res.circuit = {"diagram": _rc_circuit(v, "전열 연성 열노드: 손실원이 T_j에 의존", controlled=True).to_json(), "intervals": bands, "plot_group": "et"}
    if no_fp:
        res.verdict("NO_STABLE_FIXED_POINT", f"루프 이득 ≥ 1인 구간({', '.join(no_fp)})에는 이 단순 모델의 안정한 정상해가 없다 — 실제 파손온도 예측이 아니다")
    if t_x is not None:
        res.verdict("OUT_OF_VALIDITY", f"{t_x:.4g} s에 T_j가 손실모델 적용범위 {Tmax:g} °C를 넘는다 — 그 이후 값은 외삽하지 않고 미지원 처리")
    if not no_fp and t_x is None:
        res.verdict("PASS_WITHIN_MODEL", "전열 고정점·과도가 닫힌 식·반복·독립 RK4·열 원장과 일치")
    if lam > 1:
        res.warnings.append(f"명시적 Euler Δt = {dt:g} s는 수치적으로 불안정하다(|1 + Δt·A| = {lam:.3g}). 이것은 해법의 실패이며 물리적 폭주가 아니다.")
    res.assumptions += ["단일 열노드, 전체 손실에 선형 온도계수: P = P₀[1 + α(T_j − T_ref)] (합성)", f"손실모델 적용범위 T_j ≤ {Tmax:g} °C (ASSUMED) — 넘으면 미지원", "냉각수 온도는 계단 변화, 과부하는 손실 배율 k로 인가"]
    res.not_valid_for += ["실제 소자 파손온도·열폭주 판정", "적용범위 밖 온도의 손실", "다중 소자 열 결합 (EX08)"]
    res.interpretation = (
        f"손실이 T_j에 비례해 커지면 온도 상승이 다시 손실을 키운다. 한 바퀴 이득 g = α·R_th·P = {g0:.3g}이면 정상 상승은 고정 손실의 1/(1 − g) = {1 / (1 - g0) if g0 < 1 else float('inf'):.4g}배가 된다. "
        "g가 1에 가까우면 작은 손실 오차도 크게 증폭되고, g ≥ 1이면 이 모델에는 정상해가 없다. 반복 계산이 발산했을 때는 먼저 수치 방법(step·반복식)과 모델 적용범위를 확인한다."
    )
    return res


# ======================================================================================
# Experiment 3: per-event loss from a synthetic loss map (postprocessed estimate)
# ======================================================================================

MAP_I = np.array([0.0, 25.0, 50.0, 100.0, 150.0])  # A
MAP_T = np.array([25.0, 100.0, 150.0, 175.0])  # degC


def synthetic_map(scale_E: float = 1.0, R25: float = 16e-3):
    """Synthetic grid data (ASSUMED learning values, not a real part): E in J, R in ohm, V_f in V."""
    I, T = MAP_I[:, None], MAP_T[None, :]
    Eon = scale_E * (40.0 + 3.2 * I + 0.028 * I**2) * (1 + 0.0016 * (T - 25.0)) * 1e-6
    Eoff = scale_E * (10.0 + 2.0 * I + 0.006 * I**2) * (1 + 0.0008 * (T - 25.0)) * 1e-6
    Erec = scale_E * (0.25 * I) * (1 + 0.004 * (T - 25.0)) * 1e-6
    R = R25 * np.array([1.0, 21.0 / 16.0, 25.5 / 16.0, 28.0 / 16.0])
    Vf = 2.9 + 0.011 * MAP_I
    return {"Eon": Eon, "Eoff": Eoff, "Erec": Erec, "R": R, "Vf": Vf}


def bilinear(table, Iq, Tq):
    """Bilinear interpolation on (MAP_I, MAP_T); None outside the grid (no extrapolation)."""
    if Iq < MAP_I[0] - 1e-12 or Iq > MAP_I[-1] + 1e-12 or Tq < MAP_T[0] - 1e-12 or Tq > MAP_T[-1] + 1e-12:
        return None
    i = min(max(int(np.searchsorted(MAP_I, Iq, side="right") - 1), 0), MAP_I.size - 2)
    j = min(max(int(np.searchsorted(MAP_T, Tq, side="right") - 1), 0), MAP_T.size - 2)
    u = (Iq - MAP_I[i]) / (MAP_I[i + 1] - MAP_I[i])
    w = (Tq - MAP_T[j]) / (MAP_T[j + 1] - MAP_T[j])
    return float((1 - u) * (1 - w) * table[i, j] + u * (1 - w) * table[i + 1, j] + (1 - u) * w * table[i, j + 1] + u * w * table[i + 1, j + 1])


def event_losses(v: dict, Tj: float, mp: dict, lin=None):
    """Per switching period losses of the upper device over one fundamental period.

    Current i = I_pk sin(x), duty d = 0.5 (1 + m sin(x + phi)); dead time t_dt is taken from
    the turning-on edge.  i > 0: forward channel for d T_s - t_dt, hard E_on(i) + E_off(i).
    i < 0: reverse (synchronous) channel for d T_s - t_dt, body diode during both dead times,
    recovery E_rec(|i|) when the lower device turns on; the upper channel switches softly.
    ``lin`` = (k_E, R) replaces the map with E = k_E |i| and a constant R (limiting case).
    """
    N = max(8, int(round(v["fs"] / v["fout"])))
    Ts = 1.0 / v["fs"]
    phi = math.acos(v["cosphi"])
    x = 2 * math.pi * (np.arange(N) + 0.5) / N
    i = v["Ipk"] * np.sin(x)
    d = 0.5 * (1 + v["m"] * np.sin(x + phi))
    ton = np.maximum(d * Ts - v["t_dt"], 0.0)
    E = {k: np.zeros(N) for k in ("ch_f", "ch_r", "on", "off", "dio", "rec")}
    bad = np.zeros(N, dtype=bool)
    if lin is None:
        R = np.interp(Tj, MAP_T, mp["R"]) if MAP_T[0] <= Tj <= MAP_T[-1] else None
    else:
        R = lin[1]
    for k in range(N):
        ia = abs(i[k])
        if lin is None:
            if R is None or ia > MAP_I[-1]:
                bad[k] = True
                continue
            e_on, e_off, e_rec = bilinear(mp["Eon"], ia, Tj), bilinear(mp["Eoff"], ia, Tj), bilinear(mp["Erec"], ia, Tj)
            vf = float(np.interp(ia, MAP_I, mp["Vf"]))
        else:
            e_on, e_off, e_rec, vf = lin[0] * ia * 0.6, lin[0] * ia * 0.4, 0.0, 0.0
        if i[k] >= 0:
            E["ch_f"][k] = i[k] ** 2 * R * ton[k]
            E["on"][k] = e_on
            E["off"][k] = e_off
        else:
            E["ch_r"][k] = i[k] ** 2 * R * ton[k]
            E["dio"][k] = vf * ia * 2 * v["t_dt"]
            E["rec"][k] = e_rec
    P = {k: float(np.sum(val)) * v["fout"] for k, val in E.items()}
    P["total"] = sum(P[k] for k in ("ch_f", "ch_r", "on", "off", "dio", "rec"))
    return {"x": x, "i": i, "d": d, "E": E, "P": P, "bad": bad, "N": N, "R": R}


def run_loss_map(v: dict) -> Result:
    res = Result("FL03", "loss_map", "A (postprocessed loss estimate)")
    mp = synthetic_map(v["E_scale"], v["R25"])
    Tj = v["Tj"]
    ev = event_losses(v, Tj, mp)
    nbad = int(ev["bad"].sum())
    P = ev["P"]
    lab = {"ch_f": "채널 전도 (forward, i > 0)", "ch_r": "채널 전도 (reverse 동기, i < 0)", "on": "E_on (hard, i > 0)", "off": "E_off (hard, i > 0)", "dio": "body diode (dead time, i < 0)", "rec": "diode recovery E_rec (i < 0)"}
    for k in ("ch_f", "ch_r", "on", "off", "dio", "rec"):
        res.add_metric(f"P_{k}", lab[k], P[k], "W", basis="상측 소자, 기본파 주기 평균", note="범위 밖 사건 제외 — 불완전한 부분합" if nbad else "")
    if nbad:
        res.add_metric("P_total", "상측 소자 손실 합", f"불완전: 범위 밖 사건 {nbad}/{ev['N']}개 (외삽 안 함)", "", basis=f"map 범위 I ≤ {MAP_I[-1]:g} A, {MAP_T[0]:g}–{MAP_T[-1]:g} °C")
    else:
        res.add_metric("P_total", "상측 소자 손실 합 (postprocessed)", P["total"], "W", basis="이상 정현 전류에 사건 에너지를 더한 추정")
    res.add_metric("P_gate", "게이트 구동 전력 (별도, 합에 없음)", v["Qg"] * v["dVg"] * v["fs"], "W", basis="드라이버·R_g 소모")
    res.add_metric("n_events", "기본파당 스위칭 주기 수", ev["N"], "")
    res.add_metric("n_bad", "loss map 범위 밖 사건 수", nbad, "", basis="외삽하지 않고 미지원으로 표시")
    # switching-energy estimation methods (hard events only)
    hard = ev["i"] > 0
    if not nbad:
        Ptrue = P["on"] + P["off"]
        Iref = 100.0
        Eref25 = bilinear(mp["Eon"], Iref, 25.0) + bilinear(mp["Eoff"], Iref, 25.0)
        ErefT = bilinear(mp["Eon"], Iref, Tj) + bilinear(mp["Eoff"], Iref, Tj)
        EpkT = (bilinear(mp["Eon"], v["Ipk"], Tj) or float("nan")) + (bilinear(mp["Eoff"], v["Ipk"], Tj) or float("nan"))
        nh = int(hard.sum())
        P_b = nh * Eref25 * v["fout"]
        P_c = nh * EpkT * v["fout"]
        P_d = float(np.sum(ErefT * np.abs(ev["i"][hard]) / Iref)) * v["fout"]
        rows = [["사건별 E(i_k, T_j) — map 보간", Ptrue, 0.0], [f"데이터시트 한 점 E({Iref:g} A, 25 °C)를 모든 사건에", P_b, (P_b - Ptrue) / Ptrue * 100],
                [f"I_pk 점 E({v['Ipk']:g} A, T_j)를 모든 사건에", P_c, (P_c - Ptrue) / Ptrue * 100], ["한 점 E(100 A, T_j)를 전류에 비례 환산", P_d, (P_d - Ptrue) / Ptrue * 100]]
        res.tables.append(Table("t_method", "스위칭 손실 추정 방식 비교 (같은 운전점)", ["방식", "P_sw [W]", "사건별 대비 [%]"], rows,
                                note="한 데이터시트 전류점의 E를 모든 전류에 반복 적용하면 정현 전류의 대부분 사건(작은 전류)을 잘못 계산한다. 온도 조건도 맞춘다."))
        res.add_metric("sw_err_1pt", "한 점 E(100 A, 25 °C) 반복 적용 오차", (P_b - Ptrue) / Ptrue, "", basis="사건별 map 보간 대비")
        err_1pt_txt = f"한 데이터시트 점(100 A, 25 °C)을 반복하면 이 운전점에서 {(P_b - Ptrue) / Ptrue * 100:+.3g} % 틀린다. "
    else:
        err_1pt_txt = "(이번 조건은 범위 밖 사건이 있어 한 점 비교를 생략했다.) "
    # limiting case: linear E, constant R, no dead time vs closed-form sinusoidal-PWM averages
    kE, Rc = 6e-6, 20e-3
    vl = dict(v)
    vl["t_dt"] = 0.0
    evl = event_losses(vl, 25.0, mp, lin=(kE, Rc))
    phi = math.acos(v["cosphi"])
    cf_f = v["Ipk"] ** 2 * Rc * (1 / 8 + v["m"] * math.cos(phi) / (3 * math.pi))
    cf_r = v["Ipk"] ** 2 * Rc * (1 / 8 - v["m"] * math.cos(phi) / (3 * math.pi))
    cf_sw = v["fs"] * kE * v["Ipk"] / math.pi
    res.add_check(check_close("한계 경우: 사건 합 전도(forward) vs 정현 PWM 닫힌 식", evl["P"]["ch_f"], cf_f, 2e-4, "N개 스위칭 주기 합(선형 E·상수 R·dead time 0) vs I²R(1/8 + m·cosφ/3π)", True, "W"))
    res.add_check(check_close("한계 경우: 사건 합 전도(reverse) vs 닫힌 식", evl["P"]["ch_r"], cf_r, 2e-4, "같은 합 vs I²R(1/8 − m·cosφ/3π)", True, "W"))
    res.add_check(check_close("한계 경우: 사건 합 스위칭 vs f_s·k·I_pk/π", evl["P"]["on"] + evl["P"]["off"], cf_sw, 2e-4, "E = k|i| 사건 합 vs 닫힌 식", True, "W"))
    grid_ok = all(abs(bilinear(mp["Eon"], MAP_I[a], MAP_T[b]) - mp["Eon"][a, b]) < 1e-18 for a in range(MAP_I.size) for b in range(MAP_T.size))
    res.add_check(Check("map 보간이 격자점을 그대로 재현 (회귀)", "PASS" if grid_ok else "FAIL", 0.0, "", path="bilinear(격자점) = 표 값", independent=False))
    # electro-thermal operating point with the map
    Rth, Tb = v["Rth"], v["Tb"]

    def Ptot(T):
        e = event_losses(v, T, mp)
        return None if e["bad"].any() else e["P"]["total"]

    f_lo = MAP_T[0] - Tb - Rth * (Ptot(MAP_T[0]) or 0.0)
    P_hi = Ptot(MAP_T[-1])
    Tj_et = None
    if P_hi is None:
        et_status = "map 전류 범위 밖 — 전열 해를 구하지 않음"
    elif MAP_T[-1] - Tb - Rth * P_hi < 0:
        et_status = f"T_j = T_b + R_th·P(T_j)의 해가 map 온도 범위({MAP_T[-1]:g} °C) 안에 없다 — 외삽하지 않음"
    elif f_lo > 0:
        et_status = "T_b가 map 하한보다 낮은 경우 — 하한 이하는 미지원"
    else:
        Tj_et = brentq(lambda T: T - Tb - Rth * Ptot(T), MAP_T[0], MAP_T[-1], xtol=1e-9)
        et_status = f"T_j* = {Tj_et:.4g} °C (map 범위 안)"
    res.add_metric("Tj_et", "전열 동작점 T_j* = T_b + R_th·P(T_j*)", Tj_et if Tj_et is not None else et_status, "°C" if Tj_et is not None else "", basis=f"R_th {Rth:g} K/W, T_b {Tb:g} °C, map 손실로 풀이")
    Ts_ = np.linspace(MAP_T[0], MAP_T[-1], 31)
    Pc = [Ptot(T) for T in Ts_]
    res.add_series("et_P", "소자 손실 P(T_j) (map)", "W", Ts_.tolist(), [p if p is not None else float("nan") for p in Pc])
    res.add_series("et_line", "냉각 능력 (T_j − T_b)/R_th", "W", Ts_.tolist(), ((Ts_ - Tb) / Rth).tolist(), dash=True)
    res.add_plot("p_et", "전열 동작점: 손실 곡선과 냉각선의 교점", ["et_P", "et_line"], x_label="T_j", x_unit="°C", y_label="P", y_unit="W", kind="xy", level="A",
                 vlines=[{"x": Tj_et, "label": f"T_j* {Tj_et:.4g} °C"}] if Tj_et is not None else [], markers=[{"x": Tj_et, "y": (Tj_et - Tb) / Rth, "label": "동작점"}] if Tj_et is not None else [],
                 proved="손실이 온도와 함께 오르므로 동작점은 손실 곡선과 냉각선의 교점이다. 교점이 map 범위 밖이면 값을 만들지 않는다.",
                 not_yet="단일 R_th·정상상태만 본다. 과도·다중 소자 결합은 실험 2·EX08.")
    # event plot (angle axis) with bands linked to the circuit
    deg = np.degrees(ev["x"])
    for k in ("ch_f", "ch_r", "dio", "rec"):
        res.add_series(f"ev_{k}", lab[k], "J", deg.tolist(), ev["E"][k].tolist())
    res.add_series("ev_sw", "E_on + E_off (hard)", "J", deg.tolist(), (ev["E"]["on"] + ev["E"]["off"]).tolist())
    res.add_series("i_ph", "상 전류 i", "A", deg.tolist(), ev["i"].tolist())
    res.add_series("i_up", "상측 소자 전류 (on 구간)", "A", deg.tolist(), ev["i"].tolist())
    bands = [{"x0": 0.0, "x1": 180.0, "mode": "fwd", "label": "i > 0: 상측 forward"}, {"x0": 180.0, "x1": 360.0, "mode": "rev", "label": "i < 0: 상측 reverse"}]
    res.add_plot("p_ev", "스위칭 주기별 에너지 (상측 소자, 기본파 한 주기)", ["ev_ch_f", "ev_ch_r", "ev_sw", "ev_dio", "ev_rec"], x_label="전기각", x_unit="deg", y_label="에너지/주기", y_unit="J", kind="xy", bands=bands, level="A", group="ev",
                 proved="같은 소자라도 전류 방향에 따라 손실 종류가 바뀐다: i > 0에서는 forward 전도와 hard switching, i < 0에서는 reverse 전도·dead-time diode·recovery.",
                 not_yet="이상 정현 전류에 사건 에너지를 붙인 postprocessed 추정이다. 전류 리플·dead time에 의한 전압 오차·온도 변동은 결합하지 않았다.")
    hl = [{"y": MAP_I[-1], "label": f"map 상한 {MAP_I[-1]:g} A"}, {"y": -MAP_I[-1], "label": f"−{MAP_I[-1]:g} A"}]
    res.add_plot("p_i", "상 전류와 loss map 범위", ["i_ph"], x_label="전기각", x_unit="deg", y_label="i", y_unit="A", kind="xy", bands=bands, level="A", group="ev", hlines=hl,
                 proved="map 범위를 넘는 전류의 사건은 외삽하지 않고 미지원으로 센다.", not_yet="상 전류는 이상 정현파 I_pk·sin이며 map 상한은 합성 격자의 경계다(격자점 재현은 회귀 확인). 전류 리플·과도 피크 전류와 실제 소자의 SOA·단락 한계는 이 범위 판정에 들어 있지 않다.")
    Ii = np.linspace(0, MAP_I[-1], 61)
    for Tq, tag in ((25.0, "25"), (Tj, "op"), (175.0, "175")):
        if MAP_T[0] <= Tq <= MAP_T[-1]:
            res.add_series(f"mon_{tag}", f"E_on + E_off @ {Tq:g} °C", "J", Ii.tolist(), [bilinear(mp["Eon"], x, Tq) + bilinear(mp["Eoff"], x, Tq) for x in Ii])
    res.add_plot("p_map", "합성 loss map E(I, T) (격자 보간)", [k for k in ("mon_25", "mon_op", "mon_175") if any(s.key == k for s in res.series)], x_label="I", x_unit="A", y_label="E_on + E_off", y_unit="J", kind="xy", level="A",
                 vlines=[{"x": v["Ipk"], "label": f"I_pk {v['Ipk']:g} A"}, {"x": MAP_I[-1], "label": "map 상한"}],
                 proved="E는 전류·온도의 함수이며 map 밖에는 값이 없다.", not_yet="합성 map이다. 실제 데이터는 V_DC·R_g·상대 소자·적분 정의가 같아야 쓸 수 있다.")
    rows = [
        ["채널 전도 (forward·reverse)", "포함", "∫i²·R_DS(on)(T_j) — on 구간에서 dead time 제외", "reverse 동기 전도와 diode 전도를 같은 시간에 이중 계산하지 않음"],
        ["E_on, E_off", "포함 (i > 0 hard)", "map E(|i|, T_j), 사건별", "E_on에 상대 소자 C_oss 충전 에너지가 포함된 정의로 간주"],
        ["body diode (dead time)", "포함 (i < 0)", "V_f(|i|)·|i|·2t_dt", "i > 0의 dead-time 도통은 하측 소자 손실"],
        ["diode recovery E_rec", "포함 (i < 0)", "하측 hard turn-on 때 상측 diode", "E_on 측정에 recovery가 포함됐다면 하측 E_on과 중복 — 정의 확인"],
        ["C_oss 에너지", "E_on 안에 포함으로 간주", "별도 항 없음", "별도로 더하면 중복"],
        ["게이트 구동", "별도 표시", "Q_g·ΔV_g·f_s", "소자 접합 손실과 합치지 않음"],
    ]
    res.tables.append(Table("t_bound", "손실 포함 경계", ["항목", "이 계산", "계산 방법", "중복 방지"], rows, note="이상 정현 전류 파형에 손실을 후처리한 postprocessed loss estimate다. 전력단 방정식에 결합된 손실이 아니다."))
    res.tables.append(Table("t_map", "합성 loss map 격자값 (ASSUMED, 실제 부품 아님)", ["I [A]"] + [f"E_on+E_off @ {t:g} °C [µJ]" for t in MAP_T],
                            [[float(MAP_I[a])] + [float((mp["Eon"][a, b] + mp["Eoff"][a, b]) * 1e6) for b in range(MAP_T.size)] for a in range(MAP_I.size)],
                            note=f"R_DS(on)(T) = {', '.join(f'{r * 1e3:.3g}' for r in mp['R'])} mΩ @ {', '.join(f'{t:g}' for t in MAP_T)} °C; V_f(I) = 2.9 V + 11 mΩ·I. 범위 밖은 미지원."))
    res.circuit = {"diagram": _leg_circuit(v).to_json(), "intervals": bands, "plot_group": "ev"}
    if nbad:
        res.verdict("OUT_OF_VALIDITY", f"{nbad}개 사건이 loss map 범위 밖이다 — 외삽하지 않고 손실 합을 불완전으로 표시")
    elif Tj_et is None:
        res.verdict("OUT_OF_VALIDITY", et_status)
    else:
        res.verdict("PASS_WITHIN_MODEL", "사건별 map 손실(postprocessed)이 한계 경우 닫힌 식과 일치; 전열 동작점이 map 범위 안")
    res.assumptions += ["상측 소자, 이상 정현 전류 i = I_pk sin(ωt), SPWM duty 0.5(1 + m sin(ωt + φ)), 전류 리플 무시", "dead time은 켜지는 edge에서 on 시간을 줄임; i < 0 구간의 상측 channel은 soft switching", "합성 loss map (격자 bilinear 보간, 범위 밖 미지원)", "전열 동작점: 단일 R_th, 정상상태"]
    res.not_valid_for += ["실제 부품 손실·효율 보증", "map 범위 밖 전류·온도", "열 과도·mission (실험 2, EX08)"]
    res.interpretation = (
        f"i > 0 반주기에는 상측이 forward로 도통하며 hard switching하고, i < 0 반주기에는 reverse 동기 전도와 dead time의 body diode 도통, 하측 turn-on 때의 recovery가 생긴다. "
        "스위칭 에너지는 사건마다 그 순간 전류로 map에서 읽어야 한다. " + err_1pt_txt +
        "map 범위를 넘는 전류·온도는 값을 만들지 않고 미지원으로 둔다."
    )
    return res


def _leg_circuit(v: dict) -> Circuit:
    c = Circuit("inverter_leg", 560, 330, title="Half-bridge leg: 상측 소자의 손실 경계")
    vdc = c.add("vsource", "Vdc", 70, 165, 90, "V_DC", "", lpos=(52, 170, "end"))
    qu = c.add("nmos", "QU", 260, 95, 90, "상측 (관심 소자)", lpos=(290, 90, "start"))
    ql = c.add("nmos", "QL", 260, 235, 90, "하측", lpos=(290, 230, "start"))
    ld = c.add("isource", "Load", 420, 165, 90, "부하 전류 i(t)", "I_pk sin ωt", lpos=(438, 170, "start"))
    c.add("ground", "g", 260, 300)
    c.wire("w_top", vdc["a"], (70, 40), (260, 40), qu["a"])
    c.wire("w_mid_u", qu["b"], (260, 165))
    c.wire("w_mid_l", (260, 165), ql["a"])
    c.wire("w_mid_load", (260, 165), (420, 165), ld["a"])
    c.wire("w_load_ret", ld["b"], (420, 290), (260, 290))
    c.wire("w_bot", ql["b"], (260, 290), (70, 290), vdc["b"])
    c.dot((260, 165), (260, 290))
    c.text(245, 160, "SW", "node")
    c.probe("pi", "i_up", 330, 153, "right", "i")
    c.mode("fwd", "i > 0: 상측 forward", ["Vdc", "w_top", "QU", "w_mid_u", "w_mid_load", "Load", "w_load_ret", "w_bot"], "상측 채널이 전류를 부하로 보낸다 (hard switching, E_on·E_off); dead time에는 하측 diode가 도통", dim=["QL"])
    c.mode("rev", "i < 0: 상측 reverse", ["QU", "w_mid_u", "w_mid_load", "Load", "w_top", "Vdc"], "전류가 상측으로 되돌아온다: on 구간은 reverse 동기 채널, dead time은 상측 body diode, 하측 turn-on 때 상측 diode recovery", dim=[])
    return c


# ======================================================================================
# Experiment 4: Foster vs Cauer
# ======================================================================================


def ladder_GC(R, C):
    """Cauer ladder junction -> ... -> boundary: node k has C_k, R_k connects node k to k+1 (last to the boundary)."""
    n = len(R)
    G = np.zeros((n, n))
    for k in range(n):
        g = 1.0 / R[k]
        G[k, k] += g
        if k + 1 < n:
            G[k + 1, k + 1] += g
            G[k, k + 1] -= g
            G[k + 1, k] -= g
    return G, np.diag(np.asarray(C, dtype=float))


def cauer_to_foster(R, C):
    """Exact Foster form of a Cauer ladder via the generalized eigenproblem G v = lambda C v."""
    from scipy.linalg import eigh

    G, Cm = ladder_GC(R, C)
    lam, V = eigh(G, Cm)
    return V[0, :] ** 2 / lam, 1.0 / lam


def foster_to_cauer(Rf, tf):
    """Continued-fraction (Cauer I) expansion of Z(s) = sum R_i / (1 + s tau_i), in the scaled variable s*tau_max."""
    from numpy.polynomial import polynomial as Pl

    n = len(Rf)
    ts = float(max(tf))
    tau = np.asarray(tf, dtype=float) / ts
    D = np.array([1.0])
    for t in tau:
        D = Pl.polymul(D, [1.0, t])
    N = np.zeros(1)
    for i in range(n):
        term = np.array([float(Rf[i])])
        for j in range(n):
            if j != i:
                term = Pl.polymul(term, [1.0, tau[j]])
        N = Pl.polyadd(N, term)
    Rc, Cc = [], []
    num, den = D, N
    for _ in range(n):
        num = np.trim_zeros(num, "b")
        den = np.trim_zeros(den, "b")
        c = num[-1] / den[-1]
        rem = Pl.polysub(num, Pl.polymul([0.0, c], den))[: len(num) - 1]
        Cc.append(c * ts)
        rem = np.trim_zeros(rem, "b")
        r = den[-1] / rem[-1]
        Rc.append(r)
        nd = Pl.polysub(den, Pl.polymul([r], rem))[: len(den) - 1]
        num, den = rem, nd
        if len(np.trim_zeros(den, "b")) == 0:
            break
    return np.array(Rc), np.array(Cc)


def fit_foster(t, Z, n, R_total_hint):
    """Least-squares Foster fit (positive R_i, tau_i) to a sampled Z_th(t) - what a datasheet provides."""
    tau0 = np.geomspace(max(t[0] * 3, 1e-6), t[-1] / 5, n)
    x0 = np.concatenate([np.log(np.full(n, R_total_hint / n)), np.log(tau0)])

    def resid(x):
        R = np.exp(x[:n])
        tau = np.exp(x[n:])
        return (np.sum(R[None, :] * (1 - np.exp(-t[:, None] / tau[None, :])), axis=1) - Z) / np.maximum(Z, 1e-6 * R_total_hint)

    sol = least_squares(resid, x0, method="lm", xtol=1e-14, ftol=1e-14, max_nfev=20000)
    R, tau = np.exp(sol.x[:n]), np.exp(sol.x[n:])
    k = np.argsort(tau)[::-1]
    return R[k], tau[k], float(np.max(np.abs(sol.fun)))


def _cauer_plus_sink(R, C, Rtim, Rhs, Chs, P, Tb):
    """Physical ladder + TIM (no capacitance) + heatsink node -> NodeNetwork (case node eliminated)."""
    n = len(R)
    Cn = list(C) + [Chs]
    G = np.zeros((n + 1, n + 1))
    for k in range(n - 1):
        G[k, k + 1] = G[k + 1, k] = 1.0 / R[k]
    G[n - 1, n] = G[n, n - 1] = 1.0 / (R[n - 1] + Rtim)
    Gb = np.zeros((n + 1, 1))
    Gb[n, 0] = 1.0 / Rhs
    P0 = np.zeros(n + 1)
    P0[0] = P
    return NodeNetwork(Cn, G, Gb, [(0.0, P0, [Tb])], names=[f"n{k}" for k in range(n)] + ["hs"])


def run_foster_cauer(v: dict) -> Result:
    res = Result("FL03", "foster_cauer", "A (Z_th 표현) + C (정확 선형 과도)")
    R = np.array([v["R1"], v["R2"], v["R3"], v["R4"]])
    C = np.array([v["C1"], v["C2"], v["C3"], v["C4"]])
    Rtim, Rhs, Chs, P, Tb = v["Rtim"], v["Rhs"], v["Chs"], v["P"], v["Tb"]
    Rjc = float(R.sum())
    # datasheet-like Z_jc(t): case held at T_b (ideal boundary)
    Gjc = np.zeros((4, 4))
    for k in range(3):
        Gjc[k, k + 1] = Gjc[k + 1, k] = 1.0 / R[k]
    Gb = np.zeros((4, 1))
    Gb[3, 0] = 1.0 / R[3]
    Pv = np.array([P, 0, 0, 0])
    jc = NodeNetwork(C, Gjc, Gb, [(0.0, Pv, [Tb])], names=["j", "n1", "n2", "n3"])
    t_end = 200.0
    trj = simulate(jc, 0, [Tb] * 4, 0.0, t_end)
    tz = np.geomspace(1e-5, 100.0, 90)
    Zjc = (values_at(trj, tz, "j") - Tb) / P
    Rf, tf = cauer_to_foster(R, C)
    Zf = np.sum(Rf[None, :] * (1 - np.exp(-tz[:, None] / tf[None, :])), axis=1)
    err_eig = float(np.max(np.abs(Zf - Zjc)) / Rjc)
    res.add_check(Check("Foster(고유값 분해) vs 사다리 과도 (Z_jc(t))", "PASS" if err_eig < 1e-6 else "FAIL", err_eig, "rel", 1e-6, path="일반화 고유값 G·v = λ·C·v로 만든 Foster 합 vs 행렬지수로 푼 사다리 과도", independent=True, detail=f"R_jc = {Rjc:.4g} K/W"))
    Rc2, Cc2 = foster_to_cauer(Rf, tf)
    rt = float(max(np.max(np.abs(Rc2 - R) / R), np.max(np.abs(Cc2 - C) / C)))
    res.add_check(Check("Cauer → Foster → Cauer 왕복 (연분수 전개)", "PASS" if rt < 1e-8 else "FAIL", rt, "rel", 1e-8, path="고유값 분해의 Foster를 연분수로 다시 Cauer로: 원래 R·C 복원", independent=True))
    nfit = int(v["n_fit"])
    Rfit, tfit, fres = fit_foster(tz, Zjc, nfit, Rjc)
    res.add_metric("fit_res", f"Foster {nfit}항 피팅 최대 상대잔차", fres, "", basis="데이터시트가 주는 Foster 계수의 역할")
    # three connections to the same heatsink
    net_a = _cauer_plus_sink(R, C, Rtim, Rhs, Chs, P, Tb)
    tra = simulate(net_a, 0, [Tb] * 5, 0.0, t_end)
    Tj_a = values_at(tra, tz, "n0")
    Rcv, Ccv = foster_to_cauer(Rfit, tfit)
    net_c = _cauer_plus_sink(Rcv, Ccv, Rtim, Rhs, Chs, P, Tb)
    trc = simulate(net_c, 0, [Tb] * (len(Rcv) + 1), 0.0, t_end)
    Tj_c = values_at(trc, tz, "n0")
    tau_hs = Rhs * Chs
    Zfit = np.sum(Rfit[None, :] * (1 - np.exp(-tz[:, None] / tfit[None, :])), axis=1)
    T_hs_b = Tb + P * Rhs * (1 - np.exp(-tz / tau_hs))
    Tcase_b = T_hs_b + P * Rtim
    Tj_b = Tcase_b + P * Zfit
    # physical case temperature in (a): node n3 minus the drop across R4
    Tn3 = values_at(tra, tz, "n3")
    Ths = values_at(tra, tz, "hs")
    Tcase_a = Tn3 - R[3] * (Tn3 - Ths) / (R[3] + Rtim)
    dB = Tj_b - Tj_a
    dC = Tj_c - Tj_a
    kB = int(np.argmax(np.abs(dB)))
    T_ss = Tb + P * (Rjc + Rtim + Rhs)
    res.add_metric("T_ss", "정상 T_j (세 연결 모두)", float(tra.system.steady(0)[0]), "°C", ref=T_ss, ref_label="T_b + P(R_jc + R_TIM + R_hs)", tol=1e-9, basis="직렬 저항 합 — 정상상태는 같다")
    res.add_metric("err_wrong", "잘못된 연결(Foster 노드에 방열판) 최대 T_j 오차", float(dB[kB]), "K", basis=f"t ≈ {tz[kB]:.3g} s", note="정상상태는 같지만 과도가 틀린다")
    res.add_metric("t_wrong", "최대 오차 시각", float(tz[kB]), "s")
    res.add_metric("err_conv", "Foster → Cauer 변환 후 연결 최대 오차", float(np.max(np.abs(dC))), "K", basis="피팅 오차만 남는다")
    res.add_metric("case_jump", "잘못된 연결의 t = 0⁺ case 온도 계단", float(P * Rtim), "K", note="Foster 사슬은 열을 즉시 통과시켜 case가 순간적으로 뛴다 — 물리적으로 불가능")
    res.add_check(Check("변환된 Cauer 연결 ≈ 물리 사다리 연결", "PASS" if np.max(np.abs(dC)) < max(0.05, 5 * fres * P * Rjc) else "FAIL", float(np.max(np.abs(dC))), "K", max(0.05, 5 * fres * P * Rjc), path="Foster 피팅 → 연분수 Cauer → 방열판 연결 vs 원래 물리 사다리 + 방열판", independent=True))
    led = energy_ledger(tra, net_a, 0.0, t_end, ["p_in"], ["p_out"], [], rated_power=P)
    res.add_check(ledger_check(led, 1e-6, what="열 원장 (물리 사다리 + 방열판): "))
    res.add_series("Zjc", "Z_jc(t) 물리 사다리 (case 고정)", "K/W", tz.tolist(), Zjc.tolist())
    res.add_series("Zfit", f"Foster {nfit}항 피팅", "K/W", tz.tolist(), Zfit.tolist(), dash=True)
    res.add_plot("p_z", "데이터시트형 Z_jc(t): 경계가 고정된 조건의 곡선", ["Zjc", "Zfit"], x_label="t", x_unit="s", y_label="Z_th", y_unit="K/W", kind="xy", log_x=True, level="A",
                 proved="Foster 합은 case를 고정한 경계에서의 Z_jc(t) 곡선을 잘 재현한다 — 그것이 Foster의 용도다.",
                 not_yet="Foster의 내부 노드 온도는 die·solder 위치의 온도가 아니다.")
    res.add_series("Tja", "(a) 물리 사다리 + 방열판", "°C", tz.tolist(), Tj_a.tolist())
    res.add_series("Tjb", "(b) 잘못: Foster 끝 노드에 방열판", "°C", tz.tolist(), Tj_b.tolist())
    res.add_series("Tjc", "(c) Foster → Cauer 변환 후 연결", "°C", tz.tolist(), Tj_c.tolist(), dash=True)
    res.add_plot("p_tj", "같은 방열판, 세 가지 연결의 T_j(t)", ["Tja", "Tjb", "Tjc"], x_label="t", x_unit="s", y_label="T_j", y_unit="°C", kind="xy", log_x=True, level="C",
                 markers=[{"x": float(tz[kB]), "y": float(Tj_b[kB]), "label": f"오차 {dB[kB]:+.3g} K"}],
                 proved="Foster 사슬 끝에 방열판을 붙이면 열이 사슬을 즉시 통과해 방열판이 너무 일찍 데워진다. 정상상태는 같아도 과도 T_j가 틀린다.",
                 not_yet="이 예에서는 잘못된 연결이 과대평가 쪽이지만 방열판·결합 조건에 따라 과소평가도 가능하다. 실제 계면 열저항은 측정·해석으로 따로 확인한다.")
    res.add_series("dB", "(b) − (a)", "K", tz.tolist(), dB.tolist())
    res.add_series("dC", "(c) − (a)", "K", tz.tolist(), dC.tolist())
    res.add_plot("p_err", "연결 방식별 T_j 오차", ["dB", "dC"], x_label="t", x_unit="s", y_label="ΔT_j", y_unit="K", kind="xy", log_x=True, level="C",
                 proved="변환(Cauer)으로 연결하면 오차가 피팅 잔차 수준으로 줄어든다.", not_yet="(c)의 작은 오차는 ‘변환된 Cauer 연결 ≈ 물리 사다리’ 검사(피팅 잔차에 비례한 허용치)로 확인했다. 이 합성 예는 진짜 망이 사다리라 변환이 맞지만, 실제 모듈에서 피팅 Foster로 만든 Cauer는 port impedance만 재현하고 층·계면 위치는 보장하지 않는다.")
    res.add_series("Tca", "(a) 물리 case 온도", "°C", tz.tolist(), Tcase_a.tolist())
    res.add_series("Tcb", "(b) 잘못된 연결의 case 온도", "°C", tz.tolist(), Tcase_b.tolist())
    res.add_plot("p_case", "case 온도: 물리 경로 vs Foster 끝 노드", ["Tca", "Tcb"], x_label="t", x_unit="s", y_label="T_case", y_unit="°C", kind="xy", log_x=True, level="C",
                 proved="물리 경로에서는 case가 die를 통과한 열로 천천히 데워지지만, Foster 끝 노드는 t = 0⁺에 P·R_TIM만큼 뛴다.", not_yet="Foster 끝 노드의 t = 0⁺ 계단은 P·R_TIM(지표 case_jump)과 같다. 물리 case 온도는 R4와 R_TIM 사이 노드를 저항 분압으로 잡은 값이라 TIM 열용량·면내 온도 분포·열전대 위치에 따른 차이는 없다.")
    res.tables.append(Table("t_net", "열망 계수", ["표현", "R [K/W]", "C [J/K] 또는 τ [s]"], [
        ["물리 Cauer (합성, ASSUMED)", ", ".join(f"{x:.4g}" for x in R), "C: " + ", ".join(f"{x:.4g}" for x in C)],
        ["정확 Foster (고유값)", ", ".join(f"{x:.4g}" for x in Rf), "τ: " + ", ".join(f"{x:.4g}" for x in tf)],
        [f"피팅 Foster {nfit}항", ", ".join(f"{x:.4g}" for x in Rfit), "τ: " + ", ".join(f"{x:.4g}" for x in tfit)],
        ["피팅 → Cauer 변환", ", ".join(f"{x:.4g}" for x in Rcv), "C: " + ", ".join(f"{x:.4g}" for x in Ccv)],
    ], note="Foster의 R_i·τ_i는 곡선 피팅 계수이며 층의 R·C가 아니다. 변환된 Cauer도 port impedance를 재현하는 한 표현이며, 원래 층 구조와 같다는 보장은 없다(이 합성 예는 진짜 망이 사다리라 일치)."))
    res.circuit = {"diagram": _fc_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    res.verdict("PASS_WITHIN_MODEL", f"물리 사다리·변환 Cauer 연결이 일치; Foster 노드에 방열판을 붙인 잘못된 연결은 과도 T_j를 최대 {dB[kB]:+.3g} K 틀림")
    res.assumptions += ["junction→case 4층 합성 Cauer 사다리 (ASSUMED 층값)", "TIM은 열용량 없는 R, 방열판은 한 노드 R·C, 냉각 경계 T_b 일정", "데이터시트 Z_jc는 case 온도 고정 경계에서 정의"]
    res.not_valid_for += ["실제 모듈 층 구조·계면 열저항", "측정 Z_th의 경계 조건이 다른 경우", "다중 소자 cross heating (EX08)"]
    res.interpretation = (
        "Foster 합은 ‘case를 고정했을 때’의 Z_jc(t) 곡선 피팅이다. 사슬의 커패시터가 R과 병렬이라 사슬 전체가 열을 즉시 통과시키므로, 그 끝에 방열판을 붙이면 방열판이 t = 0부터 전체 손실을 받는다. "
        f"실제로는 die·solder·substrate의 열용량이 먼저 채워진 뒤에야 열이 case에 도달한다. 그래서 정상상태는 같아도 과도 T_j가 최대 {dB[kB]:+.3g} K 틀린다. "
        "외부 열망을 연결하려면 Cauer(또는 물리 기반 multiport)로 바꿔 연결한다."
    )
    return res


def _fc_circuit(v: dict) -> Circuit:
    c = Circuit("foster_cauer", 720, 330, title="(위) 물리 Cauer 사다리 + 방열판 · (아래) Foster 사슬: 내부 노드는 물리 계면이 아니다")
    xs = [120, 210, 300, 390]
    c.add("isource", "Pa", 50, 110, 270, "P", "", lpos=(30, 114, "end"))
    c.wire("w_pa", (50, 80), (50, 60), (90, 60))
    for k, x in enumerate(xs):
        c.add("capacitor", f"Ca{k}", x - 30, 100, 90, f"C{k + 1}", "", lpos=(x - 44, 104, "end"))
        c.add("resistor", f"Ra{k}", x + 5, 60, 0, f"R{k + 1}", "", lpos=(x + 5, 44, "middle"))
        c.wire(f"w_ca{k}", (x - 30, 60), (x - 30, 70))
        c.wire(f"w_cg{k}", (x - 30, 130), (x - 30, 150))
        if k:
            c.wire(f"w_r{k}", (xs[k - 1] + 35, 60), (x - 25, 60))
    c.wire("w_j", (90, 60), (xs[0] - 25, 60))
    c.add("resistor", "Rtim_a", 490, 60, 0, "R_TIM", "", lpos=(490, 44, "middle"))
    c.wire("w_tim_a", (xs[-1] + 35, 60), (460, 60))
    c.add("capacitor", "Chs_a", 560, 100, 90, "C_hs", "", lpos=(576, 104, "start"))
    c.add("resistor", "Rhs_a", 620, 60, 0, "R_hs", "", lpos=(620, 44, "middle"))
    c.wire("w_hs_a", (520, 60), (560, 60), (560, 70))
    c.wire("w_hs_a2", (560, 60), (590, 60))
    c.wire("w_cool_a", (650, 60), (690, 60), (690, 150), (50, 150), (50, 140))
    c.text(430, 52, "case", "node")
    c.text(95, 52, "T_j", "node")
    yb = 240
    c.add("isource", "Pb", 50, yb + 50, 270, "P", "", lpos=(30, yb + 54, "end"))
    c.wire("w_pb", (50, yb + 20), (50, yb), (90, yb))
    for k, x in enumerate(xs):
        c.add("resistor", f"Rb{k}", x, yb, 0, f"R_F{k + 1}", "", lpos=(x, yb - 16, "middle"))
        c.add("capacitor", f"Cb{k}", x, yb + 40, 0, f"C_F{k + 1}", "", lpos=(x, yb + 62, "middle"))
        c.wire(f"w_bl{k}", (x - 30, yb), (x - 30, yb + 40))
        c.wire(f"w_br{k}", (x + 30, yb), (x + 30, yb + 40))
        if k:
            c.wire(f"w_bs{k}", (xs[k - 1] + 30, yb), (x - 30, yb))
    c.wire("w_bj", (90, yb), (xs[0] - 30, yb))
    c.add("resistor", "Rtim_b", 490, yb, 0, "R_TIM", "", lpos=(490, yb - 16, "middle"))
    c.wire("w_tim_b", (xs[-1] + 30, yb), (460, yb))
    c.text(560, yb + 34, "✗ Foster 끝 노드 = case?", "")
    c.wire("w_hs_b", (520, yb), (600, yb))
    c.text(640, yb + 4, "→ 방열판", "")
    c.notes.append("위: 커패시터가 열적 접지에 연결된 물리 사다리. 아래: 커패시터가 저항과 병렬인 Foster 사슬 — 사슬 끝을 case로 쓰면 틀린다.")
    return c


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "P = 100 W, R_th = 0.2 K/W, τ = 10 s, T_b = 65 °C입니다. 10 s 뒤 T_j는?",
        "T_j(t) = T_b + P·R_th(1 − e^(−t/τ)) → 10 s = τ이므로 65 + 20 × 0.632 = 77.64 °C. 85 °C는 t → ∞의 정상상태다. 10 s에 85 °C라고 답하면 과도와 정상상태를 혼동한 것이다.",
        "P = 100 W, R_th = 0.2 K/W, tau = 10 s and a 65 °C boundary. What is T_j after 10 s?",
        "The single-node response is T_b + P R (1 − e^(−t/τ)); at t = τ that is 65 + 20 × 0.632 = 77.64 °C. The 85 °C figure is the steady state reached only after several time constants.",
        ["1 − e^(−t/τ)", "77.64 °C", "85 °C는 정상상태"],
        kind="calc",
    ),
    Question(
        "R_DS(on)이 온도와 함께 오르면 온도 계산은 어떻게 바뀝니까? 반복이 발산하면 열폭주입니까?",
        "온도→손실→온도를 반복한다. 선형식 P = P₀[1 + α(T_j − T_b)]면 ΔT = R·P₀/(1 − αR·P₀): 분모가 작을수록 민감하고, 분모 ≤ 0이면 이 단순모델에 안정한 정상해가 없다. "
        "하지만 반복 발산만으로 실물 열폭주를 확정하지 않는다: 수치 방법(step·반복식), 경계조건, 손실 곡선의 적용범위를 먼저 확인한다.",
        "How does a temperature-dependent R_DS(on) change the calculation, and is a diverging iteration a thermal runaway?",
        "You iterate temperature, loss and temperature. With a linear model the rise is R P0 over one minus alpha R P0, so the sensitivity grows as the denominator shrinks and there is no stable steady state when it reaches zero. A diverging iteration is not proof of runaway by itself: check the numerical method, the boundary conditions and the valid range of the loss curve first.",
        ["1/(1 − αRP₀)", "분모 ≤ 0", "수치 방법 확인", "적용범위"],
        kind="pressure",
    ),
    Question(
        "데이터시트 Foster 계수의 마지막 노드에 TIM과 방열판을 연결해도 됩니까?",
        "안 된다. Foster 계수는 지정 경계(case 고정)에서의 Z_th 곡선 피팅이며 내부 노드는 solder·case 같은 물리 위치가 아니다. 외부 열망을 연결하려면 Cauer(또는 물리 기반 multiport)로 변환·모델링한다. "
        "핀핀 직접냉각 모듈의 junction-to-fluid 정격에 포함된 경로를 TIM으로 다시 더하지도 않는다.",
        "Can you attach the TIM and heatsink to the last node of a datasheet Foster model?",
        "No. Foster coefficients are a curve fit of the thermal impedance for a stated boundary; their internal nodes are not solder or case locations. To connect an external network, convert to a Cauer form or use a physically based multiport model.",
        ["곡선 피팅", "내부 노드 ≠ 물리 위치", "Cauer 변환", "경계 중복 금지"],
    ),
    Question(
        "loss map 범위를 벗어난 운전점은 어떻게 처리합니까?",
        "외삽해서 숫자를 만들지 않는다. 경고와 미지원(OUT_OF_VALIDITY)으로 표시하고, 필요한 전류·온도 조건의 데이터를 요청한다. 한 데이터시트 전류점의 E를 모든 전류에 반복 적용하지도 않는다.",
        "How do you handle operating points outside the loss map?",
        "I do not extrapolate a number. I flag the point as outside the valid range and ask for data at the required current and temperature. I also do not reuse the energy of one datasheet current for every switching event.",
        ["외삽 금지", "미지원 표시", "데이터 요청"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="rc_step",
        title="100 W 계단: 10 s = 77.642 °C, 최종 85 °C — 과도와 정상상태",
        goal=(
            "단일 열노드(R_th 0.2 K/W, τ 10 s, T_b 65 °C)에 100 W를 인가해 10 s의 T_j = 77.642411 °C와 최종 85 °C를 정확 적분·해석해·독립 RK4로 확인하고, "
            "‘10 s 뒤 85 °C’가 왜 틀린지 본다. 같은 평균손실을 짧은/긴 펄스로 주면 peak와 ΔT_j가 달라지는 것도 확인한다."
        ),
        params=[
            Param("P", "손실 P", "W", 100.0, "W", vmin=0, vmax=1e5, source="TEXTBOOK", source_note="100 W"),
            Param("Rth", "열저항 R_th", "K/W", 0.2, "K/W", vmin=1e-4, vmax=100, source="TEXTBOOK", source_note="0.2 K/W"),
            Param("tau", "열 시정수 τ = R_th·C_th", "s", 10.0, "s", vmin=1e-4, vmax=1e5, source="TEXTBOOK", source_note="10 s"),
            Param("Tb", "경계 온도 T_b", "°C", 65.0, "°C", vmin=-40, vmax=200, source="TEXTBOOK", source_note="65 °C (경계 정의: 냉각수 입구 등)"),
            Param("t_eval", "평가 시각", "s", 10.0, "s", vmin=1e-3, vmax=1e5, source="TEXTBOOK", source_note="10 s"),
            Param("duty", "펄스 duty (평균손실 = duty·P)", "", 0.5, "", vmin=0.05, vmax=0.95, source="ASSUMED", group="펄스 비교"),
            Param("T_fast", "짧은 펄스 주기", "s", 2.0, "s", vmin=0.01, vmax=1e4, source="ASSUMED", group="펄스 비교"),
            Param("T_slow", "긴 펄스 주기", "s", 40.0, "s", vmin=0.1, vmax=1e5, source="ASSUMED", group="펄스 비교"),
        ],
        presets=[
            Preset("textbook", "교재 100 W 계단", {}, "교재 06장", ("nominal", "reference")),
            Preset("t30", "평가 시각 30 s (3τ)", {"t_eval": 30.0}, "95 % 도달", ("variant",)),
            Preset("slow_pulse", "긴 펄스 120 s", {"T_slow": 120.0}, "ΔT_j 증가", ("corner",)),
        ],
        run=run_rc_step,
        model_level="A (해석해) + C (정확 선형 적분)",
        suggested_change="평가 시각을 10 s → 30 s로 바꾼다 (그 다음 긴 펄스 주기를 40 → 120 s로).",
        prediction=Prediction(
            "같은 100 W 계단에서 10 s의 T_j는?",
            ["85 °C", "약 77.6 °C", "약 72 °C", "모르겠다"],
            "약 77.6 °C",
            "t = τ에서 정상 상승 20 K의 1 − e⁻¹ = 63.2 %만 오른다: 65 + 12.64 = 77.64 °C. 85 °C는 약 5τ = 50 s 뒤에야 가까워진다.",
            ["T_te", "T_final", "T_wrong"],
            handcalc=[{"key": "T_te", "label": "10 s의 T_j", "unit": "°C"}, {"key": "T_final", "label": "최종 T_j", "unit": "°C"}],
        ),
        suggested={"t_eval": 30.0},
        student="손실은 전력이고 온도는 시스템의 응답이다. 열용량이 먼저 열을 흡수하므로 온도는 천천히 오르고, 시정수 τ가 지나야 최종 상승의 63 %에 도달한다.",
        expert=(
            "① 단일 RC는 한 시정수 근사다: 짧은 펄스는 R_th가 아니라 Z_th(t_pulse)로 본다. ② 평균손실이 같아도 펄스 주기가 τ보다 길면 peak와 ΔT_j가 커져 열 사이클(수명)에 영향이 크다. "
            "③ 검증: 행렬지수 정확 해, 손으로 쓴 RK4(4차 수렴), 열 원장(∫P = ∫열류 + C·ΔT). ④ 경계 T_b가 냉각수 입구인지 baseplate인지 먼저 정의한다."
        ),
        customer_ko="100 W를 10초 인가하면 T_j는 약 77.6 °C이고, 85 °C는 약 1분 뒤 정상상태 값입니다. 과부하 시간이 열 시정수보다 짧은지 긴지에 따라 판단이 달라지니, 부하 시간과 경계 온도 정의를 먼저 맞추시죠.",
        customer_en="Applying 100 W for 10 s gives about 77.6 °C; 85 °C is the steady state reached after roughly a minute. Whether an overload is shorter or longer than the thermal time constant changes the conclusion, so let's first agree on the load duration and the boundary temperature definition.",
        questions=[_Q[0]],
        circuit="thermal_rc",
        textbook=[TB_06],
        reference_presets=["textbook"],
        claim_limit="단일 열노드의 과도/정상상태. 다층 열 경로·수명은 주장하지 않는다.",
    ),
    Experiment(
        key="electrothermal",
        title="R_DS(on)(T) 되먹임: 수렴·발산·과부하 (전열 반복)",
        goal=(
            "P = P₀[1 + α(T_j − T_ref)]의 전열 고정점 ΔT = R·P₀/(1 − αR·P₀)를 반복·정확 과도·독립 RK4로 확인하고, 냉각수 65→85 °C와 30 s 과부하를 연성 과도로 푼다. "
            "루프 이득 ≥ 1의 NO_STABLE_FIXED_POINT, 손실모델 적용범위 초과(OUT_OF_VALIDITY), 수치 방법의 발산(명시적 Euler)을 서로 다른 결과로 구분한다."
        ),
        params=[
            Param("P0", "기준 손실 P₀ (T_j = T_ref에서)", "W", 100.0, "W", vmin=0, vmax=1e5, source="TEXTBOOK", source_note="100 W"),
            Param("Rth", "열저항 R_th", "K/W", 0.2, "K/W", vmin=1e-4, vmax=100, source="TEXTBOOK", source_note="0.2 K/W"),
            Param("tau", "열 시정수 τ", "s", 10.0, "s", vmin=1e-3, vmax=1e5, source="TEXTBOOK", source_note="10 s"),
            Param("Tb", "냉각수 온도 (시작)", "°C", 65.0, "°C", vmin=-40, vmax=150, source="TEXTBOOK", source_note="65 °C"),
            Param("Tb2", "냉각수 온도 (상승 후)", "°C", 85.0, "°C", vmin=-40, vmax=150, source="TEXTBOOK", source_note="65→85 °C"),
            Param("Tref", "손실 기준 온도 T_ref", "°C", 65.0, "°C", vmin=-40, vmax=200, source="TEXTBOOK", source_note="교재 식은 T_b 기준", group="손실 모델"),
            Param("alpha", "손실 온도계수 α", "1/K", 0.004, "1/K", vmin=-0.02, vmax=0.2, source="ASSUMED", source_note="E08 합성 0.004/K와 같은 크기", group="손실 모델"),
            Param("T_valid", "손실모델 적용범위 상한", "°C", 175.0, "°C", vmin=50, vmax=400, source="ASSUMED", group="손실 모델"),
            Param("t_cool", "냉각수 상승 시각", "s", 60.0, "s", vmin=0, vmax=1e4, source="ASSUMED", group="시나리오"),
            Param("t_ol", "과부하 시작", "s", 120.0, "s", vmin=0, vmax=1e4, source="ASSUMED", group="시나리오"),
            Param("d_ol", "과부하 길이", "s", 30.0, "s", vmin=0, vmax=1e4, source="TEXTBOOK", source_note="30초 과부하", group="시나리오"),
            Param("k_ol", "과부하 손실 배율", "", 2.0, "", vmin=1, vmax=20, source="ASSUMED", source_note="전류 ×√2 ≈ 손실 ×2", group="시나리오"),
            Param("t_end", "모의 시간", "s", 240.0, "s", vmin=10, vmax=1e5, source="ASSUMED", group="시나리오"),
            Param("dt_euler", "명시적 Euler step (수치 함정)", "s", 1.0, "s", vmin=1e-3, vmax=1e3, source="ASSUMED", group="수치"),
        ],
        presets=[
            Preset("nominal", "α 0.004/K, 과부하 ×2", {}, "안정 (g = 0.08)", ("nominal", "reference")),
            Preset("sensitive", "α 0.02/K", {"alpha": 0.02}, "과부하 중 적용범위 초과", ("corner", "reference")),
            Preset("runaway", "α 0.06/K (g = 1.2)", {"alpha": 0.06}, "안정 고정점 없음", ("failure", "reference")),
            Preset("coarse_step", "Euler Δt 25 s", {"dt_euler": 25.0}, "수치 방법만 발산", ("failure",)),
        ],
        run=run_electrothermal,
        model_level="A (고정점) + C (정확 선형 과도)",
        suggested_change="α를 0.004 → 0.06 /K로 올린다 (다음에 Euler step을 1 → 25 s로).",
        prediction=Prediction(
            "α = 0.06/K (αR·P₀ = 1.2)이면 정상 온도는?",
            ["약간 더 높다", "두 배쯤 높다", "이 모델에는 안정한 정상해가 없다", "모르겠다"],
            "이 모델에는 안정한 정상해가 없다",
            "분모 1 − αR·P₀ = −0.2 ≤ 0: 온도가 오를수록 손실이 냉각보다 빨리 는다. 과도는 발산하고 손실모델 적용범위를 넘는다. 이것은 단순 모델의 결과이며 실제 파손온도 예측이 아니다.",
            ["g0", "dT_fp", "t_valid"],
            handcalc=[{"key": "dT_fp", "label": "정상 상승 ΔT (α = 0.004)", "unit": "K"}],
        ),
        suggested={"alpha": 0.06},
        student="R_DS(on)은 뜨거워질수록 커진다. 그래서 온도가 오르면 손실이 늘고, 늘어난 손실이 다시 온도를 올린다. 이 되먹임이 약하면 조금 더 뜨거운 곳에서 멈추고, 너무 세면 멈출 곳이 없다.",
        expert=(
            "① 루프 이득 g = αR_th·P: 정상 상승은 1/(1 − g)배로 증폭되고 g ≥ 1이면 NO_STABLE_FIXED_POINT. ② 과부하·냉각수 상승은 구간마다 g가 달라진다(과부하 중 g는 k배). "
            "③ 결과 상태를 구분한다: 모델에 정상해 없음(NO_STABLE_FIXED_POINT), 적용범위 초과(OUT_OF_VALIDITY — 외삽 안 함), 수치 방법 발산(명시적 Euler |1 + Δt·A| > 1 → SOLVER_FAILED). "
            "④ 실제 손실은 전도·스위칭이 각자의 온도 의존성을 가지므로 α 하나는 합성 가정이다."
        ),
        customer_ko=(
            "이 조건에서는 온도에 따른 손실 증가를 넣으면 정상 T_j가 약 1.7 K 더 올라가고, 30초 과부하 끝에서는 고정 손실 계산보다 10 K가량 높습니다. 온도계수가 크거나 냉각이 약해지면 이 계산에 정상해가 없어지므로, "
            "실제 손실 데이터의 온도 범위와 냉각 경계를 먼저 확인하시죠."
        ),
        customer_en=(
            "Including the temperature dependence of the loss raises the steady junction temperature by about 1.7 K here, and about 10 K at the end of the 30 s overload compared with a fixed-loss calculation. With a larger temperature coefficient or weaker cooling this simple model has no steady solution, so let's first check the temperature range of the real loss data and the cooling boundary."
        ),
        questions=[_Q[1]],
        circuit="thermal_rc",
        textbook=[TB_06, TB_E08],
        reference_presets=["nominal", "sensitive", "runaway"],
        claim_limit="단일 열노드·선형 온도계수의 전열 고정점과 과도. 실제 파손온도·열폭주 판정이 아니다.",
    ),
    Experiment(
        key="loss_map",
        title="손실 포함 경계와 loss map 범위: 채널·dead time·E_sw를 사건별로",
        goal=(
            "정현 전류 인버터 leg의 상측 MOSFET 손실을 스위칭 주기마다 계산해 forward/reverse 채널 전도, dead-time body diode, E_on/E_off, recovery를 분리하고, "
            "한 데이터시트 점의 E를 반복 적용하는 오류와 loss map 범위 밖(외삽 금지)을 확인한다. 이상 전류에 붙인 postprocessed loss estimate임을 표시한다."
        ),
        params=[
            Param("Ipk", "상 전류 peak I_pk", "A", 100.0, "A", vmin=1, vmax=1000, source="ASSUMED", group="운전점"),
            Param("fs", "스위칭 주파수 f_s", "Hz", 20e3, "kHz", vmin=1e3, vmax=500e3, source="ASSUMED", group="운전점"),
            Param("fout", "출력 주파수 f_out", "Hz", 100.0, "Hz", vmin=1, vmax=5000, source="ASSUMED", group="운전점"),
            Param("m", "변조지수 m", "", 0.9, "", vmin=0, vmax=1.15, source="ASSUMED", group="운전점"),
            Param("cosphi", "역률 cos φ", "", 0.9, "", vmin=-1, vmax=1, source="ASSUMED", group="운전점"),
            Param("t_dt", "dead time", "s", 200e-9, "ns", vmin=0, vmax=5e-6, source="ASSUMED", group="운전점"),
            Param("Tj", "loss map 평가 T_j", "°C", 125.0, "°C", vmin=-40, vmax=250, source="ASSUMED", group="온도"),
            Param("Rth", "junction→냉각수 R_th (전열 동작점)", "K/W", 0.3, "K/W", vmin=1e-3, vmax=10, source="ASSUMED", group="온도"),
            Param("Tb", "냉각수 온도", "°C", 65.0, "°C", vmin=-40, vmax=150, source="ASSUMED", group="온도"),
            Param("E_scale", "map E 배율", "", 1.0, "", vmin=0.1, vmax=10, source="ASSUMED", group="합성 map"),
            Param("R25", "R_DS(on) @ 25 °C", "Ω", 16e-3, "mΩ", vmin=1e-4, vmax=1.0, source="ASSUMED", group="합성 map"),
            Param("Qg", "총 gate 전하 (게이트 전력 별도 표시)", "C", 200e-9, "nC", vmin=0, vmax=1e-5, source="ASSUMED", group="합성 map"),
            Param("dVg", "gate swing", "V", 22.0, "V", vmin=1, vmax=40, source="ASSUMED", group="합성 map"),
        ],
        presets=[
            Preset("nominal", "100 A peak, 20 kHz, T_j 125 °C", {}, "map 안", ("nominal", "reference")),
            Preset("over_I", "I_pk 180 A (map 상한 150 A)", {"Ipk": 180.0}, "외삽 금지", ("failure", "reference")),
            Preset("hot", "R_th 1.8 K/W (냉각 약함)", {"Rth": 1.8}, "전열 동작점이 map 온도 범위 밖", ("failure", "reference")),
            Preset("pf_low", "cos φ 0.3", {"cosphi": 0.3}, "reverse 전도·diode 비중 증가", ("variant",)),
        ],
        run=run_loss_map,
        model_level="A (postprocessed loss estimate)",
        suggested_change="I_pk를 100 → 180 A로 올린다 (map 상한 150 A).",
        prediction=Prediction(
            "I_pk를 map 상한(150 A)보다 큰 180 A로 올리면 손실 결과는?",
            ["가장 가까운 map 값으로 계산된다", "선형 외삽된 값이 나온다", "범위 밖 사건을 미지원으로 표시하고 합을 불완전으로 둔다", "모르겠다"],
            "범위 밖 사건을 미지원으로 표시하고 합을 불완전으로 둔다",
            "loss map 밖의 점은 수치를 반환하기보다 경고·미지원 처리가 우선이다. 필요한 전류·온도의 데이터를 요청한다.",
            ["n_bad", "P_total"],
        ),
        suggested={"Ipk": 180.0},
        student="같은 소자라도 전류가 어느 방향으로 흐르느냐에 따라 손실의 종류가 달라진다. 전류가 순방향이면 켜고 끌 때 손실이 크고, 역방향이면 dead time 동안 body diode가 전류를 흘린다.",
        expert=(
            "① 포함 경계: 채널(forward/reverse)·dead-time diode·E_on/E_off·recovery·C_oss·게이트를 표로 구분하고 이중 계산을 막는다. ② 사건별 E(i_k, T_j) 보간 vs 한 점 E 반복의 오차. "
            "③ loss map 밖은 외삽하지 않는다(OUT_OF_VALIDITY). ④ 전열 동작점은 손실 곡선과 냉각선의 교점이며 교점이 map 밖이면 값을 만들지 않는다. "
            "⑤ 검증: 선형 E·상수 R 한계 경우의 사건 합이 정현 PWM 닫힌 식과 일치."
        ),
        customer_ko="현재 운전점의 손실은 전류 방향별로 나눠 계산해야 합니다. 150 A를 넘는 구간은 가지고 계신 loss 데이터 범위 밖이라 값을 만들지 않았습니다. 해당 전류·온도 조건의 E_on/E_off 데이터를 받으면 다시 계산하겠습니다.",
        customer_en="The loss has to be split by current direction at this operating point. Above 150 A we are outside the loss data you provided, so I did not extrapolate; with switching-energy data at that current and temperature I will recompute it.",
        questions=[_Q[3]],
        circuit="inverter_leg",
        textbook=[TB_06],
        reference_presets=["nominal", "over_I"],
        claim_limit="합성 loss map의 postprocessed 손실 추정. 실제 부품 손실·효율을 주장하지 않는다.",
    ),
    Experiment(
        key="foster_cauer",
        title="Foster vs Cauer: 내부 노드는 물리 계면이 아니다",
        goal=(
            "물리 Cauer 사다리의 Z_jc(t)를 정확 Foster(고유값)와 피팅 Foster로 표현하고, 같은 TIM·방열판을 (a) 물리 사다리, (b) Foster 끝 노드(잘못), (c) Foster→Cauer 변환 후에 연결해 "
            "T_j 과도 오차를 정량화한다. 정상상태는 같아도 과도가 틀리는 이유를 본다."
        ),
        params=[
            Param("R1", "층 1 R (die)", "K/W", 0.02, "K/W", vmin=1e-5, vmax=10, source="ASSUMED", group="물리 사다리"),
            Param("R2", "층 2 R (attach)", "K/W", 0.04, "K/W", vmin=1e-5, vmax=10, source="ASSUMED", group="물리 사다리"),
            Param("R3", "층 3 R (substrate)", "K/W", 0.06, "K/W", vmin=1e-5, vmax=10, source="ASSUMED", group="물리 사다리"),
            Param("R4", "층 4 R (baseplate)", "K/W", 0.08, "K/W", vmin=1e-5, vmax=10, source="ASSUMED", group="물리 사다리"),
            Param("C1", "층 1 C", "J/K", 0.01, "J/K", vmin=1e-6, vmax=1e4, source="ASSUMED", group="물리 사다리"),
            Param("C2", "층 2 C", "J/K", 0.06, "J/K", vmin=1e-6, vmax=1e4, source="ASSUMED", group="물리 사다리"),
            Param("C3", "층 3 C", "J/K", 0.4, "J/K", vmin=1e-6, vmax=1e4, source="ASSUMED", group="물리 사다리"),
            Param("C4", "층 4 C", "J/K", 2.5, "J/K", vmin=1e-6, vmax=1e4, source="ASSUMED", group="물리 사다리"),
            Param("Rtim", "TIM R", "K/W", 0.05, "K/W", vmin=0, vmax=10, source="ASSUMED", group="외부 열망"),
            Param("Rhs", "방열판 R (→ 냉각 경계)", "K/W", 0.15, "K/W", vmin=1e-4, vmax=10, source="ASSUMED", group="외부 열망"),
            Param("Chs", "방열판 C", "J/K", 60.0, "J/K", vmin=1e-3, vmax=1e5, source="ASSUMED", group="외부 열망"),
            Param("P", "손실 계단", "W", 100.0, "W", vmin=0.1, vmax=1e5, source="TEXTBOOK", source_note="100 W", group="외부 열망"),
            Param("Tb", "냉각 경계 온도", "°C", 65.0, "°C", vmin=-40, vmax=150, source="TEXTBOOK", source_note="65 °C", group="외부 열망"),
            Param("n_fit", "Foster 피팅 항 수", "", 4, "", vmin=2, vmax=5, kind="int", source="ASSUMED", group="피팅"),
        ],
        presets=[
            Preset("nominal", "4층 사다리 + 방열판 (τ_hs 9 s)", {}, "", ("nominal", "reference")),
            Preset("small_sink", "작은 방열판 C_hs 5 J/K", {"Chs": 5.0}, "방열판이 빨리 데워짐", ("variant", "reference")),
            Preset("fit3", "Foster 3항 피팅", {"n_fit": 3}, "피팅 오차 증가", ("variant",)),
        ],
        run=run_foster_cauer,
        model_level="A (Z_th 표현) + C (정확 선형 과도)",
        suggested_change="방열판 열용량을 60 → 5 J/K로 줄인다 (방열판이 빨리 데워지는 조건).",
        prediction=Prediction(
            "Foster 끝 노드에 방열판을 붙인 잘못된 모델의 T_j 과도는 물리 모델보다?",
            ["항상 같다", "과도에서 다르고 정상상태는 같다", "정상상태도 다르다", "모르겠다"],
            "과도에서 다르고 정상상태는 같다",
            "직렬 저항의 합은 같으므로 정상상태는 같다. 하지만 Foster 사슬은 커패시터가 저항과 병렬이라 열을 즉시 통과시켜 방열판이 너무 일찍 데워진다.",
            ["err_wrong", "T_ss", "err_conv"],
        ),
        suggested={"Chs": 5.0},
        student="Foster 모델은 온도 상승 곡선을 잘 맞추는 공식이지, 칩 안의 층을 그린 그림이 아니다. 그래서 그 공식의 중간 점에 다른 열 경로를 붙이면 열이 엉뚱한 속도로 흐른다.",
        expert=(
            "① Foster: Z(t) = Σ R_i(1 − e^(−t/τ_i)) — 지정 경계(case 고정)에서의 곡선 피팅. 내부 노드는 물리 위치가 아니다. ② Cauer: 열적 접지로 가는 커패시턴스를 가진 사다리 — 외부 열망 연결이 물리적으로 의미 있다. "
            "③ 변환: 고유값 분해(Cauer→Foster)와 연분수 전개(Foster→Cauer); 변환된 Cauer는 port impedance를 재현하는 한 표현이다. ④ 핀핀 직접냉각의 junction-to-fluid 정격에 이미 포함된 경로를 다시 더하지 않는다."
        ),
        customer_ko="데이터시트 Foster 계수는 case를 고정한 조건의 곡선 피팅이라, 그 끝에 TIM·방열판을 붙이면 과도 온도가 틀립니다. Cauer로 변환해 연결하거나 방열판을 포함한 Z_th를 측정·해석으로 확인하시죠.",
        customer_en="The datasheet Foster coefficients fit the thermal impedance with the case held fixed, so attaching the TIM and heatsink to them gives the wrong transient. I would convert to a Cauer form before connecting, or measure the thermal impedance with the heatsink in place.",
        questions=[_Q[2]],
        circuit="foster_cauer",
        textbook=[TB_06, TB_E08],
        reference_presets=["nominal", "small_sink"],
        claim_limit="합성 사다리의 표현·연결 방식 비교. 실제 모듈 층 구조·계면 값을 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL03",
    title="손실·온도·수명 — 숫자가 서로 맞아야 한다",
    title_en="Loss, temperature and lifetime: the numbers must agree",
    track="basic",
    order=3,
    path_note="14일 경로 3일차 (06 손실·열, FL03·04)",
    textbook=[TB_06],
    prerequisites=["FL02"],
    summary="100 W 계단(77.642 °C·85 °C) → 전열 반복(수렴·발산·적용범위) → 사건별 손실 경계와 loss map → Foster vs Cauer 연결.",
    experiments=EXPERIMENTS,
    minimum_scope="thermal RC·loss/T 반복 (교재 19장 표): 77.642 °C 기준·전열 수렴·경계; coolant 65→85 °C·30 s 과부하; loss map 범위; Foster/Cauer",
    claim_limits=[
        "단일 노드·합성 사다리의 열 모델 — 실제 모듈 열망·계면 값이 아님",
        "전열 발산(NO_STABLE_FIXED_POINT)은 단순 모델의 결과이며 실제 파손온도 예측이 아님",
        "Foster 내부 노드를 물리 계면으로 연결하지 않음",
        "loss map 결과는 postprocessed loss estimate, 범위 밖은 외삽하지 않음",
        "수명(년)을 계산하지 않음 (EX08: 상대 proxy만)",
    ],
    test_paths=["tests/test_fl03.py"],
)
