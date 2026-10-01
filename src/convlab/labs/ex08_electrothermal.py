"""EX08 - Electro-thermal coupling: from one peak temperature to a mission (textbook E08; extends FL02, FL03, EX01).

Three experiments:
  A  thermal matrix Z = [[.2, .05], [.05, .25]] K/W, P0 = [100, 80] W, alpha = .004/K: fixed rise [24, 25] K,
     coupled rise [26.569592, 27.751513] K, junctions 91.57/92.75 C, feedback spectral radius 0.09789.  Paths:
     linear solve, fixed-point iteration (contracts at rho), and an 8-node physical network whose port matrix is
     Z (steady state and exact transient).  rho >= 1: the iteration diverges, the network transient grows at its
     largest eigenvalue and the linear solve returns a non-physical answer - all "no stable fixed point in this
     simple model", not a proof of real runaway.  A coupled solution beyond the loss-model range is OUT_OF_VALIDITY.
  B  the dynamic self/cross model with a physical boundary: two Cauer die stacks on a shared baseplate and heatsink
     to the coolant; self and cross step responses, reciprocity, a coolant step.  Foster warnings: a delayed cross
     response cannot be followed by positive Foster terms (dense-grid NNLS bound), a signed fit needs negative R
     (not a network), and heat injected at an internal node of the self Foster chain depends on the arbitrary block
     order and gets the transient wrong - Foster internal nodes are not package layers.
  B  a synthetic mission with coolant variation and Monte Carlo parameter spread (stated seed and distributions):
     loss-only vs uncoupled vs coupled thermal, ASTM E1049 rainflow, dT_j / T_mean histograms and a RELATIVE damage
     proxy.  Traces stop where the loss model stops being valid; calibrated lifetime years are MISSING_INPUT and are
     never produced.
Networks are integrated exactly by engine/switched.py through FL03's NodeNetwork.
Closed forms for comparison: reference/thermal.py (never imported here).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from ..engine.switched import simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ._common import energy_ledger, ledger_check
from .fl03_thermal import NodeNetwork, fit_foster, values_at

TB_E08 = TextbookRef("expert-e08-전열-연성-최고온도-한-점에서-mission으로", "E08 · 전열 연성: 최고온도 한 점에서 mission으로")
TB_06 = TextbookRef("손실온도수명-숫자가-서로-맞아야-한다-fl03", "06. 손실·온도·수명 [FL03]")

K_B = 1.380649e-23  # J/K
EV = 1.602176634e-19  # J

# ======================================================================================
# Physical two-die network whose steady-state port matrix is Z
# ======================================================================================

SPLIT = (0.15, 0.30, 0.55)  # share of each die stack's self resistance per layer (die, attach, substrate)
C_DIE1 = (0.02, 0.15, 1.0)  # J/K, synthetic
C_DIE2 = (0.016, 0.12, 0.8)
NODE_NAMES = ["j1", "a1", "s1", "j2", "a2", "s2", "b", "h"]


def two_die_network(Z11, Z12, Z22, segments, Cb=30.0, Ch=200.0, f_bh=0.4, alpha=0.0, Tref=65.0, scale_die=(1.0, 1.0), scale_sh=1.0):
    """Nodes: j1-a1-s1 (die 1 Cauer stack) and j2-a2-s2 (die 2) -> shared baseplate b -> heatsink h -> coolant.

    R_self,k = Z_kk - Z12 per die stack, R_shared = Z12 (b -> h -> coolant).  Loss P_k = P0_k (1 + alpha (T_jk - T_ref))
    at the junction nodes.  ``segments``: list of (t_start, (P1, P2), T_coolant).
    """
    Rs1, Rs2, Rsh = (Z11 - Z12) * scale_die[0], (Z22 - Z12) * scale_die[1], Z12 * scale_sh
    G = np.zeros((8, 8))

    def link(a, b, R):
        G[a, b] = G[b, a] = 1.0 / R

    link(0, 1, SPLIT[0] * Rs1)
    link(1, 2, SPLIT[1] * Rs1)
    link(2, 6, SPLIT[2] * Rs1)
    link(3, 4, SPLIT[0] * Rs2)
    link(4, 5, SPLIT[1] * Rs2)
    link(5, 6, SPLIT[2] * Rs2)
    link(6, 7, f_bh * Rsh)
    Gb = np.zeros((8, 1))
    Gb[7, 0] = 1.0 / ((1 - f_bh) * Rsh)
    C = list(C_DIE1) + list(C_DIE2) + [Cb, Ch]
    segs = [(t0, [P[0], 0, 0, P[1], 0, 0, 0, 0], [Tc]) for t0, P, Tc in segments]
    al = [alpha, 0, 0, alpha, 0, 0, 0, 0]
    return NodeNetwork(C, G, Gb, segs, alpha=al, Tref=Tref, names=NODE_NAMES)


def port_matrix(net: NodeNetwork) -> np.ndarray:
    """Steady-state junction rise per watt from the network conductances (independent of the 2x2 inputs)."""
    Lm = net.L
    Z = np.zeros((2, 2))
    for col, node in enumerate((0, 3)):
        e = np.zeros(8)
        e[node] = 1.0
        T = np.linalg.solve(Lm, e)
        Z[0, col], Z[1, col] = T[0], T[3]
    return Z


# ======================================================================================
# Rainflow counting (ASTM E1049-85, three-point method with the starting-point rule)
# ======================================================================================


def reversals(y, hyst: float = 0.0) -> list:
    """Peaks and valleys of a sampled trace; excursions smaller than ``hyst`` are ignored."""
    y = [float(x) for x in y]
    out = [y[0]]
    direction = 0
    for x in y[1:]:
        if direction == 0:
            if abs(x - out[-1]) > hyst:
                direction = 1 if x > out[-1] else -1
                out.append(x)
            continue
        if (x - out[-1]) * direction >= 0:
            out[-1] = x
        elif abs(x - out[-1]) > hyst:
            direction = -direction
            out.append(x)
    return out


def rainflow(rev: list) -> list:
    """Cycles (range, mean, count) with count 1.0 for full cycles and 0.5 for half cycles."""
    stack: list = []
    cycles: list = []
    for x in rev:
        stack.append(x)
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            if len(stack) == 3:
                cycles.append((Y, 0.5 * (stack[0] + stack[1]), 0.5))
                stack.pop(0)
            else:
                cycles.append((Y, 0.5 * (stack[-2] + stack[-3]), 1.0))
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for k in range(len(stack) - 1):
        cycles.append((abs(stack[k + 1] - stack[k]), 0.5 * (stack[k + 1] + stack[k]), 0.5))
    return cycles


# ======================================================================================
# Experiment 1: thermal matrix, coupled steady state and divergence
# ======================================================================================


def _coupled(Z, P0, a):
    D = np.diag(a * P0)
    M = Z @ D
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    rise = np.linalg.solve(np.eye(2) - M, Z @ P0)
    return rise, rho, M


def run_matrix(v: dict) -> Result:
    res = Result("EX08", "thermal_matrix", "A (정상상태 열 행렬)")
    Z = np.array([[v["Z11"], v["Z12"]], [v["Z12"], v["Z22"]]])
    P0 = np.array([v["P1"], v["P2"]])
    a, Tb = v["alpha"], v["Tb"]
    tb = abs(v["Z11"] - 0.2) < 1e-15 and abs(v["Z12"] - 0.05) < 1e-15 and abs(v["Z22"] - 0.25) < 1e-15 and abs(P0[0] - 100) < 1e-12 and abs(P0[1] - 80) < 1e-12
    tba = tb and abs(a - 0.004) < 1e-15
    fixed = Z @ P0
    rise, rho, _ = _coupled(Z, P0, a)
    res.add_metric("fix1", "고정 손실 상승 ΔT₁ = (Z·P₀)₁", float(fixed[0]), "K", ref=24.0 if tb else None, ref_label="교재 24 K", tol=1e-12)
    res.add_metric("fix2", "고정 손실 상승 ΔT₂", float(fixed[1]), "K", ref=25.0 if tb else None, ref_label="교재 25 K", tol=1e-12)
    res.add_metric("rho", "되먹임 행렬 Z·diag(αP₀)의 spectral radius", rho, "", ref=0.09789 if tba else None, ref_label="교재 ≈ 0.09789", tol=1e-4, basis="< 1이면 반복 수렴·안정")
    stable = rho < 1.0
    Tval = v["T_valid"]
    beyond = stable and float(np.max(Tb + rise)) > Tval
    note_v = f"손실 모델 적용범위 {Tval:g} °C 밖 — 선형 α 외삽이며 예측이 아님" if beyond else ""
    if stable:
        res.add_metric("cp1", "전열 연성 상승 ΔT₁", float(rise[0]), "K", ref=26.569592 if tba else None, ref_label="교재 26.569592 K", tol=2e-8)
        res.add_metric("cp2", "전열 연성 상승 ΔT₂", float(rise[1]), "K", ref=27.751513 if tba else None, ref_label="교재 27.751513 K", tol=2e-8)
        res.add_metric("Tj1", "T_j1 (연성)", Tb + float(rise[0]), "°C", ref=91.57 if tba and abs(Tb - 65) < 1e-12 else None, ref_label="교재 ≈ 91.57 °C", tol=1e-4, basis=f"경계 {Tb:g} °C (고정 손실이면 {Tb + fixed[0]:.4g} °C)", note=note_v)
        res.add_metric("Tj2", "T_j2 (연성)", Tb + float(rise[1]), "°C", ref=92.75 if tba and abs(Tb - 65) < 1e-12 else None, ref_label="교재 ≈ 92.75 °C", tol=1e-4, basis=f"고정 손실이면 {Tb + fixed[1]:.4g} °C", note=note_v)
        res.add_metric("P_c", "연성 손실 P = P₀ + diag(αP₀)ΔT", ", ".join(f"{x:.4g}" for x in P0 + a * P0 * rise), "W", basis=f"온도가 오른 만큼 손실 증가 (P₀의 최대 {float(np.max(1 + a * rise)):.3g}배)")
    else:
        res.add_metric("cp1", "전열 연성 상승", "안정 해 없음 (ρ ≥ 1)", "", basis="아래 해법별 결과 참조")
    a_crit = a / rho if rho > 0 else float("inf")
    res.add_metric("a_crit", "ρ = 1이 되는 α (같은 Z·P₀)", a_crit, "1/K", basis="이 값에 가까울수록 민감")
    # fixed-point iteration
    it = [np.zeros(2)]
    for _ in range(5000):
        nxt = Z @ (P0 + a * P0 * it[-1])
        it.append(nxt)
        if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > 1e6:
            break
        if np.max(np.abs(nxt - it[-2])) < 1e-13 * max(1.0, float(np.max(np.abs(nxt)))) and len(it) > 60:
            break
    it = np.array(it)
    n_iter = len(it) - 1
    keep = np.unique(np.concatenate([np.arange(min(len(it), 60)), np.linspace(0, len(it) - 1, 40).astype(int)]))
    it_full = it
    it = it_full[keep]
    ks = keep.tolist()
    res.add_series("it1", "반복 ΔT₁", "K", ks, it[:, 0].tolist(), style="points")
    res.add_series("it2", "반복 ΔT₂", "K", ks, it[:, 1].tolist(), style="points")
    it = it_full
    if stable:
        err = np.max(np.abs(it_full[keep] - rise[None, :]), axis=1)
        res.add_series("it_err", "반복 오차 max|ΔT_k − ΔT*|", "K", ks, np.maximum(err, 1e-16).tolist(), style="points")
        kr = np.arange(0, max(ks) + 1)
        guide = err[0] * rho ** kr.astype(float)
        keep_g = guide >= 1e-17
        res.add_series("it_rho", f"기울기 ρ^k (ρ = {rho:.4g})", "K", kr[keep_g].tolist(), guide[keep_g].tolist(), dash=True)
        res.add_check(check_close("고정점 반복 수렴값 vs 선형 풀이", float(it[-1][0]), float(rise[0]), 1e-10, f"ΔT_(k+1) = Z(P₀ + diag(αP₀)ΔT_k) {n_iter}회 vs (I − Z·diag(αP₀))⁻¹Z·P₀", True, "K"))
    res.add_plot("p_it", "고정점 반복 (ΔT_{k+1} = Z·P(ΔT_k))", ["it1", "it2"], x_label="반복 k", x_unit="", y_label="ΔT", y_unit="K", kind="xy", level="A",
                 hlines=[{"y": float(rise[0]), "label": "ΔT₁*"}, {"y": float(rise[1]), "label": "ΔT₂*"}] if stable else [],
                 proved="반복은 ρ < 1일 때 기하급수로 수렴한다: 오차가 매번 약 ρ배.", not_yet="선형 α 모델이다. 비선형 손실 곡선에서는 국소 기울기로 ρ를 다시 계산한다.")
    if stable:
        res.add_plot("p_err", "반복 오차와 spectral radius", ["it_err", "it_rho"], x_label="반복 k", x_unit="", y_label="오차", y_unit="K", kind="xy", log_y=True, level="A",
                     proved=f"오차 감소율이 되먹임 행렬의 spectral radius {rho:.4g}와 같다.", not_yet="오차 기울기는 ρ^k 안내선과 겹치고 수렴값은 선형 풀이와 1e-10 이내로 일치한다. 상수 되먹임 행렬(선형 α)의 수렴률이라 비선형 손실 곡선에서는 해 근처의 국소 기울기로 ρ를 다시 계산해야 하며, 1e-16 근처의 평탄부는 부동소수점 한계다.")
    # independent formulation: steady state of the physical network with loss feedback at the junctions.
    # The passive T-network realizes Z only for 0 < Z12 < min(Z11, Z22); otherwise it is not built.
    physical = 0.0 < Z[0, 1] < min(Z[0, 0], Z[1, 1])
    net = None
    if physical:
        net = two_die_network(Z[0, 0], Z[0, 1], Z[1, 1], [(0.0, (P0[0], P0[1]), Tb)], alpha=a, Tref=Tb)
        Zn = port_matrix(net)
        res.add_check(Check("물리 망의 port 행렬 = Z (정상상태)", "PASS" if np.max(np.abs(Zn - Z)) < 1e-12 else "FAIL", float(np.max(np.abs(Zn - Z))), "K/W", 1e-12, path="8노드 Cauer 망의 컨덕턴스 행렬 풀이 vs 입력 Z", independent=True))
        if stable:
            Tss = net.steady(0)
            res.add_check(check_close("연성 상승: 2×2 행렬 풀이 vs 8노드 물리 망 정상상태", float(Tss[0] - Tb), float(rise[0]), 1e-10, "junction 손실 되먹임을 넣은 망의 평형점 −A⁻¹b vs (I − Z·D)⁻¹Z·P₀", True, "K"))
            res.add_check(check_close("연성 상승 ΔT₂: 행렬 vs 물리 망", float(Tss[3] - Tb), float(rise[1]), 1e-10, "같은 두 경로", True, "K"))
    else:
        res.add_check(Check("물리 망의 port 행렬 = Z (정상상태)", "NOT_RUN", None, "", None, path="8노드 물리 망", independent=True,
                            detail=f"Z₁₂ = {Z[0, 1]:g} K/W: 공유 경로 저항 Z₁₂가 0 < Z₁₂ < min(Z₁₁, Z₂₂)가 아니면 이 수동 T-망으로 실현할 수 없다 (두 소자가 열적으로 분리된 경우 각자 독립 경계)"))
        res.warnings.append("Z₁₂가 0 < Z₁₂ < min(Z₁₁, Z₂₂) 범위 밖이라 물리 망 확인을 생략했다 — 2×2 행렬 풀이와 반복만 비교한다.")
    # sensitivity sweep over alpha
    acr = a_crit if math.isfinite(a_crit) else 0.05
    al = np.linspace(0, 0.97 * acr, 60)
    r1, r2 = [], []
    for x in al:
        rr, _, _ = _coupled(Z, P0, x)
        r1.append(float(rr[0]))
        r2.append(float(rr[1]))
    res.add_series("sa1", "ΔT₁ 연성", "K", al.tolist(), r1)
    res.add_series("sa2", "ΔT₂ 연성", "K", al.tolist(), r2)
    res.add_series("sf1", "ΔT₁ 고정 손실", "K", [0.0, float(al[-1])], [float(fixed[0])] * 2, dash=True)
    res.add_series("sf2", "ΔT₂ 고정 손실", "K", [0.0, float(al[-1])], [float(fixed[1])] * 2, dash=True)
    res.add_plot("p_alpha", "온도계수 α에 따른 연성 상승 (ρ → 1에서 발산)", ["sa1", "sa2", "sf1", "sf2"], x_label="α", x_unit="1/K", y_label="ΔT", y_unit="K", kind="xy", level="A",
                 vlines=[{"x": a, "label": f"α = {a:g}"}] + ([{"x": a_crit, "label": "ρ = 1"}] if a_crit <= al[-1] * 1.05 else []),
                 proved="되먹임이 약하면 연성 상승은 고정 손실보다 조금 크고, ρ가 1에 가까워지면 급격히 커진다.", not_yet="α 하나가 전체 손실에 적용된 합성 가정이다. 실제 전도·스위칭 손실은 각자의 온도 곡선을 쓴다.")
    def _bars(vals, off):
        xs, ys = [], []
        for k, T in enumerate(vals):
            x0 = k + 1 + off
            xs += [x0, x0, x0 + 0.35, x0 + 0.35]
            ys += [Tb, float(T), float(T), Tb]
        return xs, ys

    xs_f, ys_f = _bars(Tb + fixed, -0.375)
    res.add_series("b_fix", "고정 손실 T_j", "°C", xs_f, ys_f)
    bar_keys = ["b_fix"]
    if stable:
        xs_c, ys_c = _bars(Tb + rise, 0.025)
        res.add_series("b_cp", "전열 연성 T_j", "°C", xs_c, ys_c)
        bar_keys.append("b_cp")
    res.add_plot("p_bar", "소자별 T_j: 고정 vs 연성 (막대)", bar_keys, x_label="소자 (1, 2)", x_unit="", y_label="T_j", y_unit="°C", kind="xy", level="A",
                 hlines=[{"y": Tb, "label": f"경계 {Tb:g} °C"}],
                 proved=f"인접 소자의 열(교차 항 {v['Z12']:g} K/W)과 온도 의존 손실이 함께 T_j를 올린다.", not_yet="정상상태만 본다. 과도·mission은 실험 2·3.")
    # method table (solve / iteration / physical transient) - matters in the divergence case
    rows = [["선형 풀이 (I − Z·D)⁻¹Z·P₀", f"{rise[0]:.6g}, {rise[1]:.6g} K" + ("" if stable else " ← 음수/비물리"), "정상해" if stable else "수학적 해는 있으나 물리적 정상상태 아님"],
            [f"고정점 반복 {n_iter}회", f"{it[-1][0]:.4g}, {it[-1][1]:.4g} K", "수렴" if stable else "발산"]]
    if net is not None:
        lam_max = float(np.max(np.real(np.linalg.eigvals(net.mode(0).A))))
        if lam_max < 0:
            res.add_metric("tau_mode", "8노드 물리 망 (손실 되먹임 포함)의 가장 느린 모드 시정수 −1/λ_max", -1.0 / lam_max, "s", basis=f"λ_max = {lam_max:.4g} 1/s < 0: 과도가 정상상태로 수렴")
        else:
            res.add_metric("tau_mode", "8노드 물리 망의 불안정 모드 e-배 성장 시간 1/λ_max", 1.0 / lam_max, "s", basis=f"λ_max = {lam_max:.4g} 1/s > 0: 온도가 지수적으로 증가")
        rows.append(["8노드 물리 망 과도 (최대 고유값)", f"λ_max = {lam_max:.3g} 1/s", "안정 (λ < 0)" if lam_max < 0 else "불안정 (λ > 0): 온도가 지수적으로 증가"])
        # the physical transient from a cold start (all nodes at the boundary temperature), exact integration
        t_sim = (8.0 if lam_max < 0 else 5.0) / abs(lam_max)
        t_long = 30.0 / abs(lam_max) if lam_max < 0 else t_sim
        tr_net = simulate(net, 0, [Tb] * 8, 0.0, t_long)
        tt = np.linspace(0.0, t_sim, 400)
        y1, y2 = values_at(tr_net, tt, "j1"), values_at(tr_net, tt, "j2")
        over = (y1 > Tval) | (y2 > Tval)
        kx = int(np.argmax(over)) + 1 if over.any() else tt.size  # stop the curves where the loss model stops being valid
        res.add_series("tr1", "T_j1 (물리 망 과도)", "°C", tt[:kx].tolist(), y1[:kx].tolist())
        res.add_series("tr2", "T_j2 (물리 망 과도)", "°C", tt[:kx].tolist(), y2[:kx].tolist())
        hl = [{"y": Tb + float(rise[0]), "label": "연성 정상해 T_j1"}, {"y": Tb + float(rise[1]), "label": "T_j2"}] if stable and not beyond else []
        res.add_plot("p_tr", "물리 망 과도: 냉간 시작에서 연성 정상해로 (또는 발산)", ["tr1", "tr2"], y_label="T_j", y_unit="°C", level="A", group="tr", hlines=hl,
                     vlines=[{"x": float(tt[kx - 1]), "label": f"{Tval:g} °C 초과 (이후 미지원)"}] if over.any() else [],
                     proved="같은 손실 되먹임을 넣은 물리 망을 정확 적분하면 ρ < 1에서는 행렬 풀이의 정상해로 가고, ρ ≥ 1에서는 지수적으로 커진다.",
                     not_yet="적용범위를 넘은 뒤의 궤적은 그리지 않는다. 실제 손실 곡선·보호 동작은 없다.")
        if lam_max < 0:
            y_end = float(values_at(tr_net, [t_long], "j1")[0])
            res.add_check(check_close("물리 망 과도의 끝값 (30/|λ_max| 뒤) vs 연성 정상해", y_end - Tb, float(rise[0]), 1e-6, "증강 행렬지수 정확 적분 (냉간 시작, 강성 망의 긴 구간) vs (I − Z·D)⁻¹Z·P₀", True, "K"))
        else:
            # late-time growth of the distance from the (non-physical) linear-solve point equals lambda_max
            ta, tb_ = 3.0 / lam_max, 5.0 / lam_max
            da, db = (values_at(tr_net, [ta, tb_], "j1") - (Tb + float(rise[0])))
            rate = math.log(abs(db) / abs(da)) / (tb_ - ta)
            res.add_check(check_close("발산 속도: 과도 적분의 ln|T − T_lin| 기울기 vs 최대 고유값", rate, lam_max, 1e-3, "정확 적분 궤적 (3/λ ~ 5/λ 구간) vs 고유값 분해", True, "1/s"))
    else:
        rows.append(["8노드 물리 망 과도", "실행 안 함", "Z₁₂가 수동 T-망 범위 밖"])
    res.tables.append(Table("t_methods", "해법별 결과 — solve()가 숫자를 준다고 정상상태가 있는 것은 아니다", ["해법", "결과", "판정"], rows,
                            note="ρ ≥ 1이면 되먹임 루프 이득이 1 이상이다. 이 단순 모델에 안정한 정상해가 없다는 뜻이며 실물 파손온도 예측은 아니다. 발산했다고 곧 열폭주를 확정하지도 않는다 — 수치 방법·경계·손실 곡선 범위를 먼저 확인한다."))
    res.circuit = {"diagram": _tnet_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    if beyond:
        res.verdict("OUT_OF_VALIDITY", f"ρ = {rho:.3g} < 1이라 수학적 정상해는 있지만 T_j {Tb + float(np.max(rise)):.4g} °C가 손실 모델 적용범위 {Tval:g} °C를 넘는다 — 선형 α 외삽이며 온도 예측이 아님")
        res.warnings.append(f"연성 해가 손실 모델 적용범위({Tval:g} °C)를 넘는다: ρ가 1에 가까우면 작은 α 오차가 큰 온도 차이가 된다.")
    elif stable:
        res.verdict("PASS_WITHIN_MODEL", f"연성 상승이 {'교재 값·' if tba else ''}반복·{'물리 망 정상상태와' if net is not None else '선형 풀이와'} 일치 (ρ = {rho:.4g} < 1)")
    else:
        res.verdict("NO_STABLE_FIXED_POINT", f"되먹임 spectral radius {rho:.3g} ≥ 1: 반복 발산, 망 과도 불안정, 선형 풀이는 비물리적 해 — 단순 모델의 결과이며 실물 열폭주 확정 아님")
    res.assumptions += ["정상상태 열 행렬 Z (같은 경계 온도 기준의 self/cross 항) — 다른 경계 기준의 R_th를 섞지 않음", "손실 P = P₀ + diag(αP₀)ΔT (전체 손실에 한 선형 온도계수, 합성)", "물리 망: 두 die Cauer stack + 공유 baseplate·방열판 (port 행렬 = Z가 되도록 구성)"]
    res.not_valid_for += ["실제 모듈의 열 행렬 (측정·해석 필요)", "비선형 손실 곡선의 큰 온도 범위", "과도·mission (실험 2·3)"]
    res.interpretation = (
        f"self 항은 자기 열, cross 항 {v['Z12']:g} K/W는 이웃 소자의 열이다. 손실이 온도와 함께 오르면 ΔT = [I − Z·diag(αP₀)]⁻¹Z·P₀가 되어 "
        + (f"고정 손실의 {fixed[0]:.4g}/{fixed[1]:.4g} K보다 높은 {rise[0]:.6g}/{rise[1]:.6g} K가 된다. 되먹임 행렬의 spectral radius {rho:.4g}가 1보다 충분히 작아 반복이 수렴한다." if stable and not beyond
           else f"spectral radius {rho:.4g}가 1에 가까워 반복은 느리게 수렴하고 상승이 {rise[0]:.4g}/{rise[1]:.4g} K로 커진다. 이 온도는 손실 모델 적용범위({Tval:g} °C) 밖의 외삽이다 — 민감도를 보여 줄 뿐 온도 예측이 아니다." if stable
           else f"되먹임 spectral radius {rho:.3g} ≥ 1이라 이 단순 모델에는 안정한 정상해가 없다. 선형 풀이가 주는 숫자는 음수/비물리적이다.")
    )
    return res


def _tnet_circuit(v: dict) -> Circuit:
    c = Circuit("thermal_matrix", 620, 260, title="정상상태 열 행렬의 물리적 T-망 (self·cross)")
    c.add("isource", "P1", 60, 140, 270, "P₁", f"{v['P1']:g} W", lpos=(40, 144, "end"))
    c.add("isource", "P2", 560, 140, 270, "P₂", f"{v['P2']:g} W", lpos=(580, 144, "start"))
    c.add("resistor", "R1", 175, 60, 0, "Z₁₁ − Z₁₂", f"{v['Z11'] - v['Z12']:.3g} K/W", lpos=(175, 36, "middle"))
    c.add("resistor", "R2", 445, 60, 0, "Z₂₂ − Z₁₂", f"{v['Z22'] - v['Z12']:.3g} K/W", lpos=(445, 36, "middle"))
    c.add("resistor", "Rs", 310, 130, 90, "Z₁₂ (공유)", f"{v['Z12']:.3g} K/W", lpos=(326, 134, "start"))
    c.add("vsource", "Tb", 310, 210, 90, "T_b", f"{v['Tb']:g} °C", lpos=(326, 214, "start"))
    c.wire("w1", (60, 110), (60, 60), (145, 60))
    c.wire("w2", (205, 60), (310, 60), (415, 60))
    c.wire("w3", (475, 60), (560, 60), (560, 110))
    c.wire("w4", (310, 60), (310, 100))
    c.wire("w5", (310, 160), (310, 180))
    c.wire("w6", (60, 170), (60, 250), (560, 250), (560, 170))
    c.wire("w7", (310, 240), (310, 250))
    c.dot((310, 60), (310, 250))
    c.text(60, 50, "T_j1", "node")
    c.text(560, 50, "T_j2", "node")
    c.text(310, 50, "공유 노드", "node")
    return c


# ======================================================================================
# Experiment 2: dynamic self/cross network with a physical boundary
# ======================================================================================


def _fit_foster_bounded(t, Z, n, scale, signed: bool, starts: int = 3):
    """Foster-form fit sum R_i (1 - exp(-t/tau_i)) with tau_i in [t0/10, 10 t_end], full-scale normalized.

    ``signed``: R_i of either sign with |R_i| <= scale (without the cap the best fit drifts toward a
    cancelling pair of ever larger opposite terms, i.e. it has no finite optimum); otherwise R_i >= 0.
    Several starting tau spreads are tried and the best kept, so a poor fit is not an optimizer artifact.
    """
    lo, hi = math.log(t[0] / 10.0), math.log(t[-1] * 10.0)
    lb = np.concatenate([np.full(n, -scale if signed else 0.0), np.full(n, lo)])
    ub = np.concatenate([np.full(n, scale if signed else np.inf), np.full(n, hi)])

    def resid(x):
        R = x[:n]
        tau = np.exp(x[n:])
        return (np.sum(R[None, :] * (1 - np.exp(-t[:, None] / tau[None, :])), axis=1) - Z) / scale

    best = None
    for k in range(starts):
        tau0 = np.geomspace(t[0] * 3 * 10**k, t[-1] / 5, n) if n > 1 else np.array([t[-1] / 5 / 10**k])
        x0 = np.concatenate([np.full(n, scale / n), np.clip(np.log(tau0), lo + 1e-9, hi - 1e-9)])
        sol = least_squares(resid, x0, bounds=(lb, ub), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        err = float(np.max(np.abs(sol.fun)))
        if best is None or err < best[2]:
            best = (sol.x[:n].copy(), np.exp(sol.x[n:]), err)
    R, tau, err = best
    k = np.argsort(tau)[::-1]
    return R[k], tau[k], err


def fit_foster_positive(t, Z, n, scale):
    return _fit_foster_bounded(t, Z, n, scale, signed=False)


def fit_foster_signed(t, Z, n, scale):
    return _fit_foster_bounded(t, Z, n, scale, signed=True)


def nnls_foster(t, Z, scale, n_grid=160):
    """Best positive Foster sum on a dense tau grid (non-negative least squares): a bound on what ANY positive Foster fit can do."""
    from scipy.optimize import nnls

    taus = np.geomspace(t[0] / 10.0, t[-1] * 10.0, n_grid)
    A = (1 - np.exp(-t[:, None] / taus[None, :])) / scale
    R, _ = nnls(A, Z / scale, maxiter=50 * n_grid)
    fit = A @ R * scale
    return fit, float(np.max(np.abs(fit - Z)) / scale), int(np.sum(R > 1e-12 * scale))


def _step_response(Z11, Z12, Z22, P, Cb, Ch, t_end, times):
    net = two_die_network(Z11, Z12, Z22, [(0.0, P, 0.0)], Cb=Cb, Ch=Ch, alpha=0.0, Tref=0.0)
    tr = simulate(net, 0, [0.0] * 8, 0.0, t_end)
    return net, tr, values_at(tr, times, "j1"), values_at(tr, times, "j2")


def eigen_step(net: NodeNetwork, P, times, node):
    """Independent path: modal (eigenvector) solution of C dT/dt = -L T + P from rest."""
    from scipy.linalg import eigh

    Cm = np.diag(net.C)
    lam, V = eigh(net.L, Cm)  # V^T C V = I, L V = C V diag(lam)
    Pv = np.zeros(8)
    Pv[0], Pv[3] = P
    coef = V.T @ Pv / lam
    return np.array([float(V[node, :] @ (coef * (1 - np.exp(-lam * t)))) for t in times])


def run_dynamic_network(v: dict) -> Result:
    res = Result("EX08", "dynamic_network", "B (물리 multiport 열망, 정확 선형 적분)")
    Z11, Z12, Z22 = v["Z11"], v["Z12"], v["Z22"]
    if not (0 < Z12 < min(Z11, Z22)):
        res.verdict("OUT_OF_VALIDITY", "Z₁₂가 0과 min(Z₁₁, Z₂₂) 사이가 아니면 이 수동(passive) T-망으로 실현할 수 없다")
        return res
    Cb, Ch = v["Cb"], v["Ch"]
    t_end = v["t_end"]
    tz = np.geomspace(1e-4, t_end, 110)
    net1, _, z11, z21 = _step_response(Z11, Z12, Z22, (1.0, 0.0), Cb, Ch, t_end, tz)
    _, _, z12, z22 = _step_response(Z11, Z12, Z22, (0.0, 1.0), Cb, Ch, t_end, tz)
    recip = float(np.max(np.abs(z12 - z21)) / Z12)
    res.add_check(Check("상반성 (reciprocity): Z₁₂(t) = Z₂₁(t)", "PASS" if recip < 1e-6 else "FAIL", recip, "rel", 1e-6, path="P₂ 계단 → T_j1과 P₁ 계단 → T_j2를 따로 적분해 비교 (수동 RC 망의 성질)", independent=True))
    Zn = port_matrix(net1)
    res.add_check(Check("정상상태 port 행렬 = Z", "PASS" if np.max(np.abs(Zn - np.array([[Z11, Z12], [Z12, Z22]]))) < 1e-12 else "FAIL", float(np.max(np.abs(Zn - np.array([[Z11, Z12], [Z12, Z22]])))), "K/W", 1e-12,
                        path="망 컨덕턴스 풀이 vs 입력 열 행렬", independent=True))
    ze = eigen_step(net1, (1.0, 0.0), tz, 0)
    res.add_check(Check("모달(고유벡터) 해석해 vs 엔진 행렬지수 (Z₁₁(t))", "PASS" if np.max(np.abs(ze - z11)) < 1e-9 * Z11 + 1e-12 else "FAIL", float(np.max(np.abs(ze - z11)) / Z11), "rel", 1e-9, path="일반화 고유값 분해의 모달 합 vs 증강 행렬지수", independent=True))
    t_c = 0.63 * Z12
    k_delay = int(np.argmax(z21 >= t_c)) if np.any(z21 >= t_c) else len(tz) - 1
    k_self = int(np.argmax(z11 >= 0.63 * Z11))
    res.add_metric("t63_self", "Z₁₁(t)이 63 %에 도달하는 시간", float(tz[k_self]), "s", basis="self heating")
    res.add_metric("t63_cross", "Z₂₁(t)이 63 %에 도달하는 시간", float(tz[k_delay]), "s", basis="cross heating은 공유 경로를 거쳐 늦게 온다")
    res.add_metric("Z21_at_1s", "Z₂₁(1 s)", float(np.interp(1.0, tz, z21)), "K/W", basis=f"정상 {Z12:g} K/W의 {np.interp(1.0, tz, z21) / Z12 * 100:.3g} %")
    # Foster fits
    n = int(v["n_fit"])
    Rf, tf, r11 = fit_foster(tz, z11, n, Z11)
    Rp, tp, rpos = fit_foster_positive(tz, z21, n, Z12)
    Rs, ts, rsig = fit_foster_signed(tz, z21, n, Z12)
    fit_nn, r_nn, n_nn = nnls_foster(tz, z21, Z12)
    res.add_metric("fit_self", f"Z₁₁ Foster {n}항 (양수) 피팅 최대 상대잔차", r11, "", basis="각 시각의 Z₁₁(t) 대비")
    res.add_metric("fit_cross_pos", f"Z₂₁ 양수 Foster {n}항 피팅 최대 오차 / Z₁₂", rpos, "", note="cross 응답의 지연을 양수 항만으로는 표현하기 어렵다")
    res.add_metric("fit_cross_nnls", "Z₂₁ 양수 Foster 조밀 격자(160 τ, NNLS) 최대 오차 / Z₁₂", r_nn, "", basis=f"양수 항 {n_nn}개 사용 — 항을 늘려도 남는 오차", note="양수 Foster 합의 기울기는 t = 0에서 최대라 지연 응답을 따라갈 수 없다")
    res.add_metric("fit_cross_sig", f"Z₂₁ 부호 자유 {n}항 피팅 최대 오차 / Z₁₂", rsig, "", basis="|R_i| ≤ Z₁₂ 제한 (제한이 없으면 크기가 커지는 상쇄 쌍으로 흘러 최적점이 없다)")
    res.add_metric("n_neg", "부호 자유 피팅의 음수 R_i 개수", int(np.sum(Rs < 0)), "", note="음수 R은 연결 가능한 물리 망이 아니다 — 곡선 표현일 뿐")
    # WRONG: cross heating injected at an internal node of the self Foster chain.  Only the port impedance of a
    # Foster network is defined; its block order is arbitrary, so both orders are evaluated.
    wrong_cases = []
    for label_o, idx in (("τ 큰 블록이 junction 쪽", np.arange(n)), ("τ 작은 블록이 junction 쪽", np.arange(n)[::-1])):
        Ro, to = Rf[idx], tf[idx]
        cum = np.cumsum(Ro[::-1])[::-1]  # heat injected at the node before block k flows through blocks k..n-1
        kk = int(np.argmin(np.abs(cum[1:] - Z12))) + 1  # an internal node (k >= 1)
        wz = np.sum(Ro[kk:][None, :] * (1 - np.exp(-tz[:, None] / to[kk:][None, :])), axis=1)
        e = wz - z21
        kw = int(np.argmax(np.abs(e)))
        wrong_cases.append({"label": label_o, "node": kk, "ss": float(cum[kk]), "z": wz, "err": float(e[kw]), "t": float(tz[kw])})
    wa, wb = wrong_cases
    res.add_metric("wrong_ss", "잘못된 모델: Foster-11 내부 노드에 P₂ 주입 시 정상 cross 저항", wa["ss"], "K/W", ref=Z12, ref_label="실제 Z₁₂", tol=None,
                   basis=f"{wa['label']}: 블록 {wa['node'] + 1}번 앞 노드 (정상값이 가장 가까운 내부 노드); 반대 순서면 {wb['ss']:.3g} K/W")
    res.add_metric("wrong_err", "잘못된 cross 모델의 최대 오차 (1 W당, 블록 순서 1)", wa["err"], "K/W", basis=f"{wa['label']}, t ≈ {wa['t']:.3g} s", note="정상값을 맞춰도 과도 cross 응답이 틀린다")
    res.add_metric("wrong_err2", "잘못된 cross 모델의 최대 오차 (1 W당, 블록 순서 2)", wb["err"], "K/W", basis=f"{wb['label']}, t ≈ {wb['t']:.3g} s", note="Foster 블록 순서는 임의 — 어느 쪽도 물리 연결이 아니다")
    res.add_series("z11", "Z₁₁(t) self", "K/W", tz.tolist(), z11.tolist())
    res.add_series("z22", "Z₂₂(t) self", "K/W", tz.tolist(), z22.tolist())
    res.add_series("z21", "Z₂₁(t) cross (P₁ → T_j2)", "K/W", tz.tolist(), z21.tolist())
    res.add_series("z12", "Z₁₂(t) cross (P₂ → T_j1)", "K/W", tz.tolist(), z12.tolist(), dash=True)
    res.add_plot("p_z", "self·cross 열 임피던스 (물리 망, 로그 시간)", ["z11", "z22", "z21", "z12"], x_label="t", x_unit="s", y_label="Z_th", y_unit="K/W", kind="xy", log_x=True, level="B",
                 proved="self 응답은 ms부터 오르지만 cross 응답은 공유 baseplate·방열판의 열용량을 거쳐 초 단위로 늦게 온다. Z₁₂(t) = Z₂₁(t) (상반성).",
                 not_yet="합성 층값이다. 실제 모듈의 cross 경로는 측정·해석 기반 multiport 모델이 필요하다.")
    fit_p = np.sum(Rp[None, :] * (1 - np.exp(-tz[:, None] / tp[None, :])), axis=1)
    fit_s = np.sum(Rs[None, :] * (1 - np.exp(-tz[:, None] / ts[None, :])), axis=1)
    res.add_series("f_pos", "Foster 양수 계수 피팅", "K/W", tz.tolist(), fit_p.tolist(), dash=True)
    res.add_series("f_sig", "Foster 부호 자유 피팅 (음수 R 포함)", "K/W", tz.tolist(), fit_s.tolist(), dash=True)
    res.add_series("f_nnls", "양수 Foster 조밀 격자 (NNLS)", "K/W", tz.tolist(), fit_nn.tolist(), dash=True)
    res.add_series("wrong", f"잘못: Foster-11 내부 노드에 P₂ 주입 ({wa['label']})", "K/W", tz.tolist(), wa["z"].tolist())
    res.add_series("wrong2", f"잘못: 같은 주입 ({wb['label']})", "K/W", tz.tolist(), wb["z"].tolist())
    res.add_series("z21b", "Z₂₁(t) 물리 망", "K/W", tz.tolist(), z21.tolist())
    res.add_plot("p_cross", "cross 응답을 Foster로 표현·연결할 때의 함정", ["z21b", "f_pos", "f_nnls", "f_sig", "wrong", "wrong2"], x_label="t", x_unit="s", y_label="cross Z", y_unit="K/W", kind="xy", log_x=True, level="B",
                 proved="cross 임피던스는 지연된 응답이라 양수 Foster로는 항을 늘려도(NNLS) 오차가 남고, 맞추려면 음수 항이 필요하다. self Foster의 내부 노드에 이웃 열을 넣으면 결과가 임의의 블록 순서에 달려 있고 과도가 틀린다.",
                 not_yet="시간별로 제각각 피팅한 cross 모델은 상반성·수동성을 보장하지 않는다. 물리 multiport(또는 제약을 둔 식별)가 필요하다.")
    # coolant step with both devices on
    P1, P2 = v["P1"], v["P2"]
    Tb, dTc, t_c = v["Tb"], v["dTc"], v["t_cool"]
    segs = [(0.0, (P1, P2), Tb), (t_c, (P1, P2), Tb + dTc)]
    net = two_die_network(Z11, Z12, Z22, segs, Cb=Cb, Ch=Ch)
    x0 = net.steady(0)
    t_c_end = t_c + v["t_after"]
    tr = simulate(net, 0, x0, 0.0, t_c_end)
    tt = np.linspace(0, t_c_end, 500)
    for nm, lab in (("j1", "T_j1"), ("j2", "T_j2"), ("b", "T_baseplate"), ("h", "T_방열판")):
        res.add_series(f"c_{nm}", lab, "°C", tt.tolist(), values_at(tr, tt, nm).tolist())
    res.add_series("c_cool", "냉각수", "°C", [0.0, t_c, t_c, t_c_end], [Tb, Tb, Tb + dTc, Tb + dTc], dash=True)
    bands = [{"x0": 0.0, "x1": t_c, "mode": "steady", "label": f"냉각수 {Tb:g} °C"}, {"x0": t_c, "x1": t_c_end, "mode": "cool", "label": f"+{dTc:g} K 계단"}]
    res.add_plot("p_cool", "냉각수 계단에 대한 응답 (두 소자 on, 정상상태에서 시작)", ["c_j1", "c_j2", "c_b", "c_h", "c_cool"], y_label="T", y_unit="°C", bands=bands, level="B", group="cool",
                 proved="냉각수 변화는 방열판·baseplate를 거쳐 junction에 도달한다 — 손실 계단과 다른 경로·시간 규모다.", not_yet="손실의 온도 의존은 이 그림에서 끄고 경로만 봤다 (실험 1·3에서 연성).")
    led = energy_ledger(tr, net, 0.0, t_c_end, ["p_in"], ["p_out"], [], rated_power=P1 + P2)
    res.add_check(ledger_check(led, 1e-6, what="열 원장 (냉각수 계단): "))
    def _terms(R, tau, scale):
        keep = np.abs(R) > 1e-9 * scale
        return ", ".join(f"{x:.4g}" for x in R[keep]) or "—", ", ".join(f"{x:.3g}" for x in tau[keep]) or "—", int(np.sum(keep))

    rows_fit = []
    for lab_f, (R_, t_, e_, sc) in (("Z₁₁ (양수, 각 시각 대비)", (Rf, tf, r11, Z11)), ("Z₂₁ (양수, / Z₁₂)", (Rp, tp, rpos, Z12)), ("Z₂₁ (부호 자유, / Z₁₂)", (Rs, ts, rsig, Z12))):
        a_, b_, k_ = _terms(R_, t_, sc)
        rows_fit.append([lab_f, a_, b_, k_, e_])
    res.tables.append(Table("t_fit", f"Foster {n}항 피팅 계수", ["응답", "R_i [K/W]", "τ_i [s]", "유효 항", "최대 잔차"], rows_fit,
        note="Foster 계수는 곡선 피팅 결과다. 내부 노드는 die·solder·baseplate 위치가 아니며, cross 피팅의 음수 R은 연결할 수 있는 열망이 아니다."))
    res.circuit = {"diagram": _net_circuit(v).to_json(), "intervals": bands, "plot_group": "cool"}
    res.verdict("PASS_WITHIN_MODEL", "물리 multiport 망: 상반성·정상 port 행렬·모달 해석해·열 원장 통과; Foster 내부 노드 연결의 오차를 정량화")
    res.assumptions += ["die별 3층 Cauer stack(합성 층값) + 공유 baseplate·방열판 → 냉각수 (port 행렬 = Z)", "손실 온도 의존 없음 (이 실험은 경로·시간 규모만)", "냉각수는 이상적인 경계 온도"]
    res.not_valid_for += ["실제 모듈의 cross 열 경로", "측정 Z_th의 경계가 다른 경우", "수명 (실험 3: 상대 proxy만)"]
    res.interpretation = (
        f"self 열은 die 층을 거쳐 ms~수백 ms에 오르지만, 이웃 소자의 열(cross)은 공유 baseplate·방열판의 열용량을 채운 뒤에야 도달해 63 % 도달이 {tz[k_delay]:.3g} s로 늦다. "
        f"양수 Foster 합은 기울기가 t = 0에서 가장 커서 이 지연 응답을 따라가지 못한다: 160개 τ의 NNLS로도 최대 오차가 Z₁₂의 {r_nn * 100:.3g} %다. 부호 자유 {n}항은 {rsig * 100:.2g} %까지 맞지만 음수 R {int(np.sum(Rs < 0))}개가 필요해 연결할 수 있는 망이 아니다. "
        f"self Foster의 내부 노드에 이웃 열을 넣으면 임의의 블록 순서에 따라 최대 오차가 Z₁₂의 {abs(wa['err']) / Z12 * 100:.3g} % 또는 {abs(wb['err']) / Z12 * 100:.3g} %가 된다 — 어느 순서도 물리 연결이 아니다. "
        "교차 결합이 필요하면 물리 기반 multiport 망을 쓴다."
    )
    return res


def _net_circuit(v: dict) -> Circuit:
    c = Circuit("thermal_multiport", 760, 350, title="물리 multiport 열망: 두 die Cauer stack + 공유 baseplate·방열판 → 냉각수")
    xs = [120, 230, 340]
    for nm, y in (("1", 60), ("2", 210)):
        c.add("isource", f"P{nm}", 50, y + 45, 270, f"P{nm}", "", lpos=(30, y + 49, "end"))
        c.wire(f"wp{nm}", (50, y + 15), (50, y), (xs[0] - 10, y))
        for k, x in enumerate(xs):
            c.add("capacitor", f"C{nm}{k}", x - 30, y + 45, 90, "", "")
            c.add("resistor", f"R{nm}{k}", x + 20, y, 0, f"R{nm}_{k + 1}", "", lpos=(x + 20, y - 16, "middle"))
            c.wire(f"wc{nm}{k}", (x - 30, y), (x - 30, y + 15))
            if k:
                c.wire(f"wr{nm}{k}", (xs[k - 1] + 50, y), (x - 10, y))
        c.wire(f"wg{nm}", (50, y + 75), (50, y + 85), (xs[-1] - 30, y + 85))
        for k, x in enumerate(xs):
            c.wire(f"wgc{nm}{k}", (x - 30, y + 75), (x - 30, y + 85))
        c.add("ground", f"g{nm}", xs[-1] - 30, y + 85)
        c.dot((xs[0] - 30, y), (xs[1] - 30, y), (xs[2] - 30, y))
        c.text(90, y - 10, f"T_j{nm}", "node")
    # shared baseplate node b, heatsink node h, coolant boundary
    xb, yb = 450, 135
    c.wire("wb1", (xs[-1] + 50, 60), (xb, 60), (xb, yb))
    c.wire("wb2", (xs[-1] + 50, 210), (xb, 210), (xb, yb))
    c.wire("wbh", (xb, yb), (530, yb))
    c.add("capacitor", "Cb", 490, yb + 45, 90, "C_b", "", lpos=(506, yb + 49, "start"))
    c.wire("wcb", (490, yb), (490, yb + 15))
    c.add("ground", "gb", 490, yb + 75)
    c.add("resistor", "Rbh", 560, yb, 0, "R_bh", "", lpos=(560, yb - 16, "middle"))
    c.wire("wh", (590, yb), (690, yb), (690, yb + 15))
    c.add("capacitor", "Chs", 630, yb + 45, 90, "C_h", "", lpos=(646, yb + 49, "start"))
    c.wire("wch", (630, yb), (630, yb + 15))
    c.add("ground", "gh", 630, yb + 75)
    c.add("resistor", "Rhc", 690, yb + 45, 90, "R_hc", "", lpos=(706, yb + 49, "start"))
    c.add("vsource", "Tc", 690, yb + 125, 90, "냉각수 T_c", "", lpos=(668, yb + 129, "end"))
    c.wire("wtc", (690, yb + 75), (690, yb + 95))
    c.add("ground", "gc", 690, yb + 155)
    c.dot((xb, yb), (490, yb), (630, yb))
    c.text(xb + 4, yb - 10, "baseplate (공유)", "node")
    c.text(630, yb - 22, "방열판", "node")
    c.notes.append("열 등가: 전류 = 열류[W], 전압 = 온도[°C], 저항 = R_th[K/W], 커패시턴스 = 열용량[J/K] (열적 기준에 연결)")
    c.notes.append("정상상태 port 행렬 = Z: die stack 저항 합 = Z_kk − Z₁₂, 공유 경로 R_bh + R_hc = Z₁₂")
    c.mode("steady", "정상", ["P1", "P2", "R10", "R11", "R12", "R20", "R21", "R22", "Rbh", "Rhc", "Tc"], "두 소자의 열이 공유 경로로 냉각수에 나간다", dim=[])
    c.mode("cool", "냉각수 계단", ["Tc", "Rhc", "Chs", "Rbh", "Cb"], "냉각수 변화가 방열판·baseplate를 거쳐 junction으로 올라온다", dim=[])
    return c


# ======================================================================================
# Experiment 3: mission, coolant variation, Monte Carlo spread, rainflow, relative damage proxy
# ======================================================================================

CITY = [(10.0, 1.8, "가속"), (30.0, 0.6, "순항"), (5.0, 0.8, "회생"), (15.0, 0.05, "정지")]


def mission_segments(v: dict, coolant_offset: float = 0.0, with_phases: bool = False):
    """Synthetic mission: warm-up city cycles (coolant ramp), highway, hill, city, park.

    Segments are (t0, load factor, coolant, tag); phases are (t0, t1, circuit mode, label).
    """
    segs, phases = [], []
    t = 0.0
    n1 = int(v["n_city1"])
    for c in range(n1):
        Tc = v["Tc_start"] + (v["Tc_nom"] - v["Tc_start"]) * (c / max(n1 - 1, 1))
        for d, f, tag in CITY:
            segs.append((t, f, Tc + coolant_offset, tag))
            t += d
    phases.append((0.0, t, "city", f"도심 ×{n1} (냉각수 warm-up)"))
    if v["t_highway"] > 0:
        segs.append((t, 1.0, v["Tc_nom"] + coolant_offset, "고속"))
        phases.append((t, t + v["t_highway"], "highway", "고속"))
        t += v["t_highway"]
    if v["t_hill"] > 0:
        segs.append((t, 1.4, v["Tc_nom"] + v["dTc_hill"] + coolant_offset, "등판"))
        phases.append((t, t + v["t_hill"], "hill", f"등판 (냉각수 +{v['dTc_hill']:g} K)"))
        t += v["t_hill"]
    n2 = int(v["n_city2"])
    if n2:
        t2 = t
        for _ in range(n2):
            for d, f, tag in CITY:
                segs.append((t, f, v["Tc_nom"] + coolant_offset, tag))
                t += d
        phases.append((t2, t, "city", f"도심 ×{n2}"))
    if v["t_park"] > 0:
        segs.append((t, 0.05, v["Tc_park"] + coolant_offset, "주차"))
        phases.append((t, t + v["t_park"], "park", "주차"))
        t += v["t_park"]
    return (segs, t, phases) if with_phases else (segs, t)


def run_mission_case(v: dict, alpha: float, scale_die=(1.0, 1.0), scale_sh=1.0, p_scale=(1.0, 1.0), coolant_offset=0.0, per_segment=10):
    segs, t_end = mission_segments(v, coolant_offset)
    P0 = (v["P1"] * p_scale[0], v["P2"] * p_scale[1])
    net = two_die_network(v["Z11"], v["Z12"], v["Z22"], [(t0, (f * P0[0], f * P0[1]), Tc) for t0, f, Tc, _ in segs], Cb=v["Cb"], Ch=v["Ch"], alpha=alpha, Tref=v["Tref"],
                          scale_die=scale_die, scale_sh=scale_sh)
    x0 = net.steady(0)
    tr = simulate(net, 0, x0, 0.0, t_end)
    smp = tr.sample(["j1", "j2", "P_j1", "P_j2", "q_out"], per_segment=per_segment)
    return net, tr, smp, segs, t_end


def damage_proxy(cycles, m, Ea, dT_ref, T_ref_C):
    """Relative Coffin-Manson-Arrhenius-shaped proxy: sum n (dT/dT_ref)^m exp(Ea/k (1/T_ref - 1/T_mean)). Not a lifetime."""
    Tr = T_ref_C + 273.15
    d = 0.0
    for rng, mean, cnt in cycles:
        if rng <= 0:
            continue
        try:
            d += cnt * (rng / dT_ref) ** m * math.exp(Ea / K_B * (1.0 / Tr - 1.0 / (mean + 273.15)))
        except OverflowError:
            return math.inf
    return d


def _hist_steps(values, weights, edges):
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    xs, ys = [edges[0]], [0.0]
    for k in range(len(counts)):
        xs += [edges[k], edges[k + 1]]
        ys += [float(counts[k]), float(counts[k])]
    xs.append(edges[-1])
    ys.append(0.0)
    return xs, ys, counts


def _first_exceed(T1, T2, Tval):
    """Index of the first sample where either junction exceeds the loss-model range (None if never)."""
    over = (T1 > Tval) | (T2 > Tval)
    return int(np.argmax(over)) if over.any() else None


def _lam_max(net, segs):
    """Largest real eigenvalue over the segments' affine modes (depends on the load factor only) and its segment tag."""
    seen: dict = {}
    for k, sg in enumerate(segs):
        if sg[1] not in seen:
            seen[sg[1]] = (float(np.max(np.real(np.linalg.eigvals(net.mode(k).A)))), sg[3])
    return max(seen.values(), key=lambda x: x[0])


def run_mission(v: dict) -> Result:
    res = Result("EX08", "mission", "B (정확 선형 열망) + 사이클 계수 (상대 proxy)")
    a = v["alpha"]
    hyst = v["hyst"]
    Tval = v["T_valid"]
    m_exp, Ea, dTr, Tr = v["m_cm"], v["Ea_eV"] * EV, v["dT_ref"], v["T_ref_d"]
    # nominal: uncoupled vs coupled
    _, tr_u, s_u, segs, t_end = run_mission_case(v, 0.0)
    net_c, tr_c, s_c, _, _ = run_mission_case(v, a)
    t, tu = np.array(s_c["t"]), np.array(s_u["t"])
    Tj1u, Tj2u = np.array(s_u["j1"]), np.array(s_u["j2"])
    Tj1c, Tj2c = np.array(s_c["j1"]), np.array(s_c["j2"])
    k_xu, k_xc = _first_exceed(Tj1u, Tj2u, Tval), _first_exceed(Tj1c, Tj2c, Tval)
    ok_u, ok_c = k_xu is None, k_xc is None
    cyc_u = rainflow(reversals(Tj1u, hyst)) if ok_u else []
    cyc_c = rainflow(reversals(Tj1c, hyst)) if ok_c else []
    D_u = damage_proxy(cyc_u, m_exp, Ea, dTr, Tr) if ok_u else None
    D_c = damage_proxy(cyc_c, m_exp, Ea, dTr, Tr) if ok_c else None
    E_u = [tr_u.integral_output(0.0, t_end, f"P_{k}") for k in ("j1", "j2")]
    E_c = [tr_c.integral_output(0.0, t_end, f"P_{k}") for k in ("j1", "j2")]
    lam, lam_tag = _lam_max(net_c, segs)
    unstable = lam > 0
    oob = f"적용범위 밖 — t = {t[k_xc]:.4g} s에 {Tval:g} °C 초과" if not ok_c else ""
    res.add_metric("t_mission", "mission 길이", t_end, "s", basis=f"{len(segs)}개 구간 (합성)")
    res.add_metric("E_loss_u", "손실 에너지 (손실만 보는 관점, 고정 손실)", sum(E_u), "J", basis=f"소자1 {E_u[0] / 1e3:.4g} kJ, 소자2 {E_u[1] / 1e3:.4g} kJ")
    if ok_c:
        res.add_metric("E_loss_c", "손실 에너지 (전열 연성)", sum(E_c), "J", basis=f"고정 손실 대비 {(sum(E_c) / sum(E_u) - 1) * 100:+.3g} %")
    else:
        res.add_metric("E_loss_c", "손실 에너지 (전열 연성)", oob, "", basis="적용범위 밖 궤적의 손실은 외삽이라 보고하지 않는다")
    res.add_metric("Tpk_u", "최고 T_j1 (비연성 열모델)", float(Tj1u.max()) if ok_u else f"적용범위 밖 — t = {tu[k_xu]:.4g} s에 {Tval:g} °C 초과", "°C" if ok_u else "")
    res.add_metric("Tpk_c", "최고 T_j1 (연성 열모델)", float(Tj1c.max()) if ok_c else oob, "°C" if ok_c else "")
    if not ok_c:
        res.add_metric("t_valid", "연성 궤적이 손실 모델 적용범위를 넘는 시각", float(t[k_xc]), "s", basis=f"T_j > {Tval:g} °C 이후 궤적은 미지원")
    cyc_ref, lab_ref = (cyc_c, "연성") if ok_c else (cyc_u, "비연성")
    trace_ref = Tj1c if ok_c else Tj1u
    if ok_c or ok_u:
        nrev = len(reversals(trace_ref, hyst))
        ncyc = sum(c[2] for c in cyc_ref)
        res.add_metric("n_cyc", f"T_j1 열 사이클 수 (rainflow, {lab_ref})", float(ncyc), "cycles", basis=f"반전점 {nrev}개, 히스테리시스 {hyst:g} K")
        res.add_metric("dT_max", f"최대 ΔT_j1 사이클 ({lab_ref})", float(max((c[0] for c in cyc_ref), default=0.0)), "K")
        res.add_metric("n_big", f"ΔT_j1 ≥ {dTr:g} K 사이클 수 ({lab_ref})", float(sum(c[2] for c in cyc_ref if c[0] >= dTr)), "cycles")
    if D_u and D_c is not None:
        res.add_metric("D_ratio", "상대 손상 proxy: 연성 / 비연성", D_c / D_u, "", basis="같은 proxy 식의 비 — 수명이 아님",
                       note=f"proxy 모양: Σn(ΔT/{dTr:g} K)^{m_exp:g}·exp(Ea/k(1/T_ref − 1/T_mean)), 계수는 ASSUMED")
    else:
        res.add_metric("D_ratio", "상대 손상 proxy: 연성 / 비연성", "적용범위 밖 — 계산 안 함", "", basis="적용범위 밖 온도의 사이클로 proxy를 만들지 않는다")
    res.add_metric("life_years", "calibrated 수명 [년]", "MISSING_INPUT", "", basis="package·접합·고장기구·시험조건별 보정 모델과 데이터가 없다 — 년 단위 값을 만들지 않는다")
    if lam < 0:
        res.add_metric("tau_mode", "연성 열망의 가장 느린 모드 시정수 −1/λ_max (가장 약한 구간)", -1.0 / lam, "s", basis=f"'{lam_tag}' 구간, λ_max = {lam:.3g} 1/s < 0: 모든 구간에 안정 평형")
    else:
        res.add_metric("tau_mode", "불안정 모드의 e-배 성장 시간 1/λ_max", 1.0 / lam, "s", basis=f"'{lam_tag}' 구간, λ_max = {lam:.3g} 1/s > 0: 그 구간에 평형 없음")
    # checks: rainflow on the ASTM E1049 example; cycle bookkeeping; heat ledger
    astm = rainflow(reversals([-2, 1, -3, 5, -1, 3, -4, 4, -2]))
    agg: dict = {}
    for rng_, _, cnt in astm:
        agg[rng_] = agg.get(rng_, 0.0) + cnt
    ok_astm = agg == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}
    res.add_check(Check("rainflow: ASTM E1049 예제 재현", "PASS" if ok_astm else "FAIL", len(astm), "사이클 항목", path="표준 예제 [−2, 1, −3, 5, −1, 3, −4, 4, −2] → 범위 3(0.5)·4(1.5)·6(0.5)·8(1.0)·9(0.5)", independent=True,
                        detail=", ".join(f"{k:g}:{agg[k]:g}" for k in sorted(agg))))
    if ok_c or ok_u:
        res.add_check(Check(f"사이클 합 = (반전점 수 − 1)/2 ({lab_ref})", "PASS" if abs(ncyc - (nrev - 1) / 2) < 1e-9 else "FAIL", ncyc, "cycles", (nrev - 1) / 2, path="rainflow 사이클 계수 보존 (half cycle 포함)", independent=False))
    t_led = t_end if ok_c else float(t[k_xc])  # the ledger covers what is reported (the valid part of the trajectory)
    led = energy_ledger(tr_c, net_c, 0.0, t_led, ["p_in"], ["p_out"], [], rated_power=v["P1"] + v["P2"])
    res.add_check(ledger_check(led, 1e-6, what="열 원장 (연성 mission" + ("): " if ok_c else f", 적용범위 안 0–{t_led:.4g} s): ")))
    # Monte Carlo spread (stated seed and distributions)
    rng = np.random.default_rng(int(v["seed"]))
    N = int(v["n_mc"])
    pk, Dm, samples = [], [], []
    n_out = n_unst = 0
    for _ in range(N):
        sd = tuple(float(x) for x in np.exp(rng.normal(0.0, v["sig_R"], 2)))
        ssh = float(np.exp(rng.normal(0.0, v["sig_R"])))
        al = float(a * (1 + rng.normal(0.0, v["sig_alpha"])))
        ps = tuple(float(x) for x in 1 + rng.normal(0.0, v["sig_P"], 2))
        co = float(rng.normal(0.0, v["sig_Tc"]))
        samples.append([sd[0], sd[1], ssh, al, ps[0], ps[1], co])
        net_s, _, sm, _, _ = run_mission_case(v, al, sd, ssh, ps, co, per_segment=8)
        T1, T2 = np.array(sm["j1"]), np.array(sm["j2"])
        n_unst += _lam_max(net_s, segs)[0] > 0
        if _first_exceed(T1, T2, Tval) is not None:
            n_out += 1
            continue
        pk.append(float(T1.max()))
        if D_c:
            Dm.append(damage_proxy(rainflow(reversals(T1, hyst)), m_exp, Ea, dTr, Tr) / D_c)
    pk, Dm = np.array(pk), np.array(Dm)
    q = lambda arr, p: float(np.percentile(arr, p))  # noqa: E731
    res.add_metric("mc_out", f"MC 표본 중 적용범위(T_j ≤ {Tval:g} °C) 밖", n_out, "samples", basis=f"N = {N}, 되먹임 불안정 구간을 가진 표본 {n_unst}개", note="적용범위 밖 표본은 분포에서 뺐다 (외삽하지 않음)" if n_out else "")
    if pk.size:
        res.add_metric("mc_pk", f"MC 최고 T_j1 P5 / P50 / P95 (적용범위 안 {pk.size}/{N})", f"{q(pk, 5):.4g} / {q(pk, 50):.4g} / {q(pk, 95):.4g} °C", "", basis=f"seed {int(v['seed'])}")
    else:
        res.add_metric("mc_pk", "MC 최고 T_j1", "모든 표본이 적용범위 밖", "", basis=f"seed {int(v['seed'])}")
    if Dm.size:
        res.add_metric("mc_D", "MC 상대 손상 proxy P5 / P50 / P95 (공칭 연성 = 1)", f"{q(Dm, 5):.3g} / {q(Dm, 50):.3g} / {q(Dm, 95):.3g}", "", basis="분포 가정 아래의 상대값 — 수명 분포가 아님")
        res.add_metric("mc_Dmax", "MC 상대 손상 proxy 최대", float(Dm.max()), "")
    else:
        res.add_metric("mc_D", "MC 상대 손상 proxy", "계산 안 함", "", basis="공칭 연성 궤적(기준)이나 표본이 적용범위 밖")
    if n_out == N:
        res.warnings.append(f"모든 MC 표본이 손실 모델 적용범위({Tval:g} °C)를 넘는다 — 분포를 만들지 않았다.")
    elif n_out:
        res.warnings.append(f"MC 표본 {n_out}/{N}개가 손실 모델 적용범위({Tval:g} °C)를 넘어 분포에서 뺐다 — 분포 꼬리가 잘린 것이므로 P95는 낙관적이다.")
    # plots: mission traces with bands (mission phases) linked to the circuit modes
    _, _, phases = mission_segments(v, 0.0, with_phases=True)
    bands = [{"x0": a0, "x1": a1, "mode": md, "label": lb} for a0, a1, md, lb in phases]
    kc = len(t) if ok_c else k_xc + 1  # the coupled traces stop where the loss model stops being valid
    ku = len(tu) if ok_u else k_xu + 1
    vl = [{"x": float(t[k_xc]), "label": f"연성 T_j > {Tval:g} °C (이후 미지원)"}] if not ok_c else []
    res.add_series("T1c", "T_j1 연성", "°C", t[:kc].tolist(), Tj1c[:kc].tolist())
    res.add_series("T1u", "T_j1 비연성", "°C", tu[:ku].tolist(), Tj1u[:ku].tolist(), dash=True)
    res.add_series("T2c", "T_j2 연성", "°C", t[:kc].tolist(), Tj2c[:kc].tolist())
    xc, yc = [], []
    for k, (t0, _, Tc, _) in enumerate(segs):
        t1 = segs[k + 1][0] if k + 1 < len(segs) else t_end
        xc += [t0, t1]
        yc += [Tc, Tc]
    res.add_series("Tcool", "냉각수", "°C", xc, yc)
    res.add_series("P1c", "P₁ 연성", "W", t[:kc].tolist(), s_c["P_j1"][:kc])
    res.add_series("P1u", "P₁ 고정 (손실만 보는 관점)", "W", tu[:ku].tolist(), s_u["P_j1"][:ku], dash=True)
    res.add_plot("p_T", "mission 온도: 비연성 vs 연성 (냉각수 변화 포함)", ["T1c", "T1u", "T2c", "Tcool"], y_label="T", y_unit="°C", bands=bands, level="B", group="ms", vlines=vl,
                 proved="같은 mission에서도 온도 의존 손실을 넣으면 고온 구간의 T_j와 사이클 진폭이 커진다. 냉각수 warm-up·주차 구간이 평균 온도를 바꾼다.",
                 not_yet="합성 mission·합성 열망이다. 실제 주행 빈도·부하 분포·수명 모델은 없다.")
    res.add_plot("p_P", "손실 trace: 손실만 보는 관점 vs 전열 연성", ["P1c", "P1u"], y_label="P", y_unit="W", bands=bands, level="B", group="ms", vlines=vl,
                 proved="손실만 보면 온도에 따른 손실 증가를 놓친다.", not_yet="연성·고정 손실 trace는 같은 물리 망의 정확 적분 출력이며 연성 mission은 열 원장(1e-6)으로 확인했다. 손실은 부하 배율 × P₀[1 + α(T_j − T_ref)]의 합성 식이라 전류·전압·스위칭 주파수에 따른 전도/스위칭 손실 분리가 없다 — 실제 mission 손실은 사건 기반 loss map(FL03)으로 다시 계산해야 한다.")
    edges = np.array([0, 2, 5, 10, 15, 20, 30, 50, 80, 120.0])
    hs = []
    for tag, cyc, lab, ok in (("c", cyc_c, "연성", ok_c), ("u", cyc_u, "비연성", ok_u)):
        if ok:
            xs, ys, _ = _hist_steps([c[0] for c in cyc], [c[2] for c in cyc], edges)
            res.add_series(f"hdT_{tag}", f"ΔT_j1 사이클 수 ({lab})", "cycles", xs, ys, dash=tag == "u")
            hs.append(f"hdT_{tag}")
    if hs:
        res.add_plot("p_hdT", "ΔT_j1 사이클 histogram (rainflow)", hs, x_label="ΔT_j", x_unit="K", y_label="사이클", y_unit="cycles", kind="xy", level="B",
                     proved="온도 trace를 rainflow로 세면 작은 ΔT 사이클이 많고 큰 ΔT 사이클은 드물다 — 손상 proxy는 큰 ΔT에 크게 좌우된다.", not_yet="히스테리시스(잡음 제거) 선택이 작은 사이클 수를 바꾼다.")
        medges = np.array([30, 50, 60, 70, 80, 90, 100, 110, 130, 160, 180.0])
        xs, ys, _ = _hist_steps([c[1] for c in cyc_ref], [c[2] for c in cyc_ref], medges)
        res.add_series("hTm", f"T_mean별 사이클 수 ({lab_ref})", "cycles", xs, ys)
        res.add_plot("p_hTm", "사이클 평균 온도 T_mean histogram", ["hTm"], x_label="T_mean", x_unit="°C", y_label="사이클", y_unit="cycles", kind="xy", level="B",
                     proved="같은 ΔT라도 평균 온도가 다르면 손상 속도가 다를 수 있어 두 축을 따로 센다.", not_yet="사이클은 ASTM E1049 예제를 재현하는 rainflow로 세었고 사이클 합 = (반전점 수 − 1)/2를 확인했다. 평균 온도만 세고 고온 유지 시간(dwell)·가열 속도는 세지 않았으며, 이 histogram을 수명으로 바꾸려면 package별로 보정된 power cycling 모델이 필요하다(calibrated 수명은 MISSING_INPUT).")
    if Dm.size:
        xs, ys, _ = _hist_steps(Dm, np.ones_like(Dm), np.linspace(min(0.5, float(Dm.min())), max(1.5, float(Dm.max()) * 1.05), 15))
        res.add_series("hD", f"상대 손상 proxy 분포 ({Dm.size}/{N})", "samples", xs, ys)
        res.add_plot("p_mc", "parameter spread에 따른 상대 손상 proxy 분포", ["hD"], x_label="상대 proxy (공칭 연성 = 1)", x_unit="", y_label="표본 수", y_unit="", kind="xy", level="B", vlines=[{"x": 1.0, "label": "공칭"}],
                     proved="R_th·α·손실·냉각수의 합성 분산만으로도 상대 손상 proxy가 크게 퍼진다 — 한 점 계산으로 수명을 말할 수 없다.", not_yet="분포는 가정이다(seed·분포 명시). calibrated 수명 년수는 만들지 않는다.")
    # tables
    row_u = (f"최고 {Tj1u.max():.4g} °C, 사이클 {sum(c[2] for c in cyc_u):.4g}, proxy 1 (기준)" if ok_u else "적용범위 밖")
    row_c = (f"최고 {Tj1c.max():.4g} °C, 사이클 {sum(c[2] for c in cyc_c):.4g}, proxy ×{D_c / D_u:.3g}, 손실 {sum(E_c) / 1e3:.4g} kJ" if ok_c and D_u
             else f"{oob}" + (f"; '{lam_tag}' 구간 되먹임 불안정 (λ = {lam:.3g} 1/s)" if unstable else ""))
    res.tables.append(Table("t_cmp", "같은 mission, 세 가지 관점", ["관점", "무엇을 보나", "결과"], [
        ["손실만 (loss-only)", "∫P dt, 평균 손실", f"{sum(E_u) / 1e3:.4g} kJ, 평균 {sum(E_u) / t_end:.4g} W (온도 없음)"],
        ["비연성 열모델", "고정 손실 → T_j trace, 사이클", row_u],
        ["연성 열모델", "P(T_j) 되먹임 → T_j trace, 사이클", row_c],
        ["calibrated 수명", "Nf 모델·보정 데이터 필요", "MISSING_INPUT — 년 단위 값 없음"]]))
    res.tables.append(Table("t_mc", "Monte Carlo 설정 (재현 가능)", ["항목", "값"], [
        ["seed", int(v["seed"])], ["표본 수", N], ["die stack R_th 배율", f"lognormal σ = {v['sig_R']:g} (die별 독립)"], ["공유 경로 R_th 배율", f"lognormal σ = {v['sig_R']:g}"],
        ["α", f"normal 평균 {a:g}/K, 상대 σ = {v['sig_alpha']:g}"], ["P₀ 배율", f"normal 1 ± {v['sig_P']:g} (소자별 독립)"], ["냉각수 offset", f"normal 0 ± {v['sig_Tc']:g} K"],
        ["상관", "모두 독립 (상관 근거 없음 — 가정)"], ["모델 오차", "포함 안 함 (합성 망·선형 α의 구조 오차는 별도)"],
        ["적용범위 밖 표본", f"{n_out}개 — 분포에서 제외 (외삽하지 않음)"]]))
    if ok_c or ok_u:
        cyc_rows = sorted(cyc_ref, key=lambda c: -c[0])[:12]
        res.tables.append(Table("t_cyc", f"큰 사이클 상위 12개 ({lab_ref}, T_j1)", ["ΔT [K]", "T_mean [°C]", "count"], [[float(a_), float(b_), float(c_)] for a_, b_, c_ in cyc_rows]))
    res.circuit = {"diagram": _mission_circuit(v).to_json(), "intervals": bands, "plot_group": "ms"}
    res.extra["mc_samples"] = {"columns": ["die1_R", "die2_R", "shared_R", "alpha", "P1_scale", "P2_scale", "coolant_offset"], "rows": samples, "seed": int(v["seed"])}
    if unstable:
        res.verdict("NO_STABLE_FIXED_POINT", f"'{lam_tag}' 구간에서 손실 되먹임이 열망보다 강하다 (λ = {lam:.3g} 1/s > 0): 그 구간에는 평형이 없고 구간 길이만큼 온도가 지수적으로 오른다 — 단순 모델의 결과이며 실물 열폭주 확정 아님")
    if not ok_c:
        res.verdict("OUT_OF_VALIDITY", f"연성 T_j가 t = {t[k_xc]:.4g} s에 손실 모델 적용범위 {Tval:g} °C를 넘는다 — 그 뒤 궤적·사이클·proxy는 만들지 않는다")
    elif not unstable:
        res.verdict("PASS_WITHIN_MODEL", "mission 열 trace·rainflow(ASTM 예제 재현)·열 원장 통과 — 합성 mission의 상대 비교")
    res.verdict("MISSING_INPUT", "calibrated lifetime 모델·데이터가 없다: 온도 사이클 histogram과 상대 손상 proxy만 보고하고 수명(년)은 만들지 않는다")
    res.assumptions += [
        "합성 mission (도심 가속/순항/회생/정지, 고속, 등판, 주차) — 실제 차량 부하 분포 아님",
        "실험 2의 물리 multiport 망, 손실 P = P₀·f(t)·[1 + α(T_j − T_ref)]",
        f"손실 모델 적용범위 T_j ≤ {Tval:g} °C (ASSUMED) — 넘으면 궤적을 끊고 proxy를 만들지 않음",
        f"rainflow: ASTM E1049 3점법, 반전점 히스테리시스 {hyst:g} K",
        f"상대 손상 proxy: Coffin–Manson–Arrhenius 모양, m = {m_exp:g}, E_a = {v['Ea_eV']:g} eV, ΔT_ref = {dTr:g} K, T_ref = {Tr:g} °C (모두 ASSUMED) — 같은 식끼리의 비만 의미",
        "MC: 독립 정규/로그정규 분산, seed 고정 (상관·모델 오차 미포함)",
    ]
    res.not_valid_for += ["수명(년)·보증 기간", "IGBT wire-bond 계수를 SiC package에 적용", "DC-link capacitor·solder 등 다른 고장 기구", "실제 mission 빈도·불확도"]
    if ok_c and D_u:
        body = (f"온도 의존 손실을 넣으면 최고 T_j가 {Tj1u.max():.4g} → {Tj1c.max():.4g} °C로 오르고 같은 proxy 식에서 상대 손상이 ×{D_c / D_u:.3g}가 된다. "
                + (f"합성 분산만으로도 proxy가 P5–P95에서 {q(Dm, 5):.3g}–{q(Dm, 95):.3g}배로 퍼진다. " if Dm.size else ""))
    else:
        body = (f"이 설정에서는 '{lam_tag}' 구간의 손실 되먹임이 열망보다 강해(λ = {lam:.3g} 1/s) 평형이 없고, " if unstable else "")
        body += f"연성 T_j가 t = {t[k_xc]:.4g} s에 적용범위 {Tval:g} °C를 넘는다. 그 뒤의 궤적·사이클·proxy는 외삽이라 만들지 않았다. " if not ok_c else ""
    res.interpretation = (
        f"손실만 보면 mission은 {sum(E_u) / 1e3:.4g} kJ의 열일 뿐이지만, 열망을 통과시키면 가속·등판마다 T_j가 오르내리는 사이클이 생긴다. " + body
        + "보정된 수명 모델이 없으므로 ‘몇 년’은 말하지 않고 사이클 histogram과 상대 비교만 보고한다."
    )
    return res


def _mission_circuit(v: dict) -> Circuit:
    c = _net_circuit(v)
    c.title = "mission: 두 소자 손실 P₁(t), P₂(t)과 냉각수 T_c(t)가 같은 물리 망에 들어간다"
    c.modes = {}
    c.mode("city", "도심", ["P1", "P2", "C10", "C20"], "가속·순항·회생·정지가 수십 초마다 바뀐다: die 층 열용량이 짧은 사이클을 만든다", dim=[])
    c.mode("highway", "고속", ["P1", "P2", "Rbh", "Rhc"], "일정 부하: baseplate·방열판까지 데워져 평균 온도가 오른다", dim=[])
    c.mode("hill", "등판", ["P1", "P2", "Tc", "Rhc"], "부하 ×1.4와 냉각수 상승이 겹친다: 최고 T_j와 큰 사이클", dim=[])
    c.mode("park", "주차", ["Tc", "Rhc", "Chs"], "손실 거의 0, 냉각수 온도가 junction을 정한다", dim=["P1", "P2"])
    return c


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "두 소자가 한 방열판에 있을 때 T_j를 R_th × P로 계산해도 됩니까?",
        "self 항만 보면 틀린다. 정상상태 ΔT = Z·P에서 off-diagonal(교차 항)이 이웃 소자의 열이다. 합성 예 Z = [[.2, .05], [.05, .25]] K/W, P = [100, 80] W면 [24, 25] K이고, "
        "α = 0.004/K의 온도 의존 손실을 넣으면 [26.57, 27.75] K로 오른다. 경계 온도(냉각수 입구·baseplate·ambient)를 같은 기준으로 정의하고 서로 다른 경계의 R_th를 섞지 않는다.",
        "Two devices share a heatsink. Can you compute T_j as R_th times P?",
        "Only the self term is not enough: the steady rise is the thermal matrix times the power vector, and the off-diagonal terms are the neighbour's heating. In the synthetic example that gives 24 and 25 K, and with a loss temperature coefficient of 0.004 per kelvin it rises to 26.57 and 27.75 K. All resistances must refer to the same boundary temperature.",
        ["교차 항", "Z·P", "연성 26.57/27.75 K", "같은 경계"],
        kind="calc",
    ),
    Question(
        "전열 반복이 발산했습니다. 열폭주입니까?",
        "되먹임 행렬 Z·diag(αP₀)의 spectral radius가 1 이상이면 이 단순 선형 모델에 안정 정상해가 없다(합성 예 0.098). 그러나 발산만으로 실물 thermal runaway를 확정하지 않는다: 수치방법, 경계조건, 비선형 손실 곡선의 적용범위를 먼저 확인한다. "
        "solve()가 숫자를 줘도(음수 온도 등) 물리적 정상상태가 아닐 수 있다.",
        "The electro-thermal iteration diverged. Is it thermal runaway?",
        "If the spectral radius of the feedback matrix is one or more, this simple linear model has no stable steady state; in the example it is about 0.098. Divergence alone does not prove runaway: check the numerical method, the boundary conditions and the valid range of the loss curve first, and note that a linear solve can return a non-physical answer.",
        ["spectral radius", "모델의 결과", "수치·경계·범위 확인"],
        kind="pressure",
    ),
    Question(
        "mission 결과로 수명 15년을 말할 수 있습니까?",
        "없다. mission에서 ΔT_j·T_mean·dwell·사이클 수를 뽑는 것과 수명을 계산하는 것은 다르다. power cycling 모델은 package·bonding·attach·고장기구·시험조건에 의존하며 IGBT wire-bond 계수를 SiC package에 그대로 쓰지 않는다. "
        "데이터가 없으면 온도 사이클 histogram·상대 손상 proxy·calibrated parameter MISSING을 보고한다.",
        "Can you state a 15-year lifetime from the mission result?",
        "No. Extracting temperature swings, mean temperatures and cycle counts is not a lifetime calculation. Power-cycling models depend on the package, bonding, die attach, failure mechanism and test conditions, so without calibrated data I report the cycle histogram, a relative damage proxy and that the calibrated parameters are missing.",
        ["사이클 추출 ≠ 수명", "package·고장기구 의존", "상대 proxy만", "MISSING_INPUT"],
        kind="pressure",
    ),
    Question(
        "cross heating을 Foster 모델로 연결하면 되지 않습니까?",
        "Foster는 지정 경계의 곡선 피팅이라 내부 노드가 실제 die·baseplate가 아니다. 이웃 열을 self Foster의 내부 노드에 넣으면 정상값을 맞춰도 과도가 틀리고, cross 임피던스를 따로 피팅하면 음수 항·상반성 위반이 생길 수 있다. "
        "물리적 연결은 Cauer 또는 측정·해석 기반 multiport 모델로 한다.",
        "Can you model cross heating by connecting Foster networks?",
        "Foster coefficients are a curve fit for a stated boundary, so their internal nodes are not the die or the baseplate. Injecting the neighbour's heat there gets the transient wrong, and fitting each cross impedance separately can need negative terms and break reciprocity. Use a Cauer or a physically based multiport model.",
        ["Foster = 피팅", "내부 노드 ≠ 물리", "상반성·수동성", "multiport"],
    ),
]

_MISSION_PARAMS = [
    Param("P1", "소자 1 기준 손실 P₀₁", "W", 100.0, "W", vmin=0, vmax=5000, source="TEXTBOOK", source_note="100 W", group="열망"),
    Param("P2", "소자 2 기준 손실 P₀₂", "W", 80.0, "W", vmin=0, vmax=5000, source="TEXTBOOK", source_note="80 W", group="열망"),
    Param("Z11", "Z₁₁", "K/W", 0.2, "K/W", vmin=1e-3, vmax=10, source="TEXTBOOK", group="열망"),
    Param("Z12", "Z₁₂ = Z₂₁", "K/W", 0.05, "K/W", vmin=1e-4, vmax=10, source="TEXTBOOK", group="열망"),
    Param("Z22", "Z₂₂", "K/W", 0.25, "K/W", vmin=1e-3, vmax=10, source="TEXTBOOK", group="열망"),
    Param("Cb", "baseplate 열용량", "J/K", 30.0, "J/K", vmin=0.1, vmax=1e5, source="ASSUMED", group="열망"),
    Param("Ch", "방열판 열용량", "J/K", 200.0, "J/K", vmin=0.1, vmax=1e6, source="ASSUMED", group="열망"),
    Param("alpha", "손실 온도계수 α", "1/K", 0.004, "1/K", vmin=0, vmax=0.03, source="TEXTBOOK", source_note="0.004/K", group="열망"),
    Param("Tref", "손실 기준 온도", "°C", 65.0, "°C", vmin=-40, vmax=200, source="TEXTBOOK", source_note="경계 65 °C", group="열망"),
    Param("T_valid", "손실 모델 적용범위 상한", "°C", 175.0, "°C", vmin=50, vmax=400, source="ASSUMED", source_note="선형 α를 믿는 온도 상한 (합성)", group="열망"),
    Param("n_city1", "warm-up 도심 사이클 수", "", 10, "", vmin=1, vmax=60, kind="int", source="ASSUMED", group="mission"),
    Param("Tc_start", "냉각수 시작 온도", "°C", 40.0, "°C", vmin=-40, vmax=120, source="ASSUMED", group="mission"),
    Param("Tc_nom", "냉각수 정상 온도", "°C", 65.0, "°C", vmin=-40, vmax=120, source="ASSUMED", group="mission"),
    Param("t_highway", "고속 구간", "s", 300.0, "s", vmin=0, vmax=1e4, source="ASSUMED", group="mission"),
    Param("t_hill", "등판 구간 (부하 ×1.4)", "s", 120.0, "s", vmin=0, vmax=1e4, source="ASSUMED", group="mission"),
    Param("dTc_hill", "등판 중 냉각수 상승", "K", 5.0, "K", vmin=0, vmax=40, source="ASSUMED", group="mission"),
    Param("n_city2", "등판 뒤 도심 사이클 수", "", 5, "", vmin=0, vmax=60, kind="int", source="ASSUMED", group="mission"),
    Param("t_park", "주차 구간", "s", 300.0, "s", vmin=0, vmax=1e4, source="ASSUMED", group="mission"),
    Param("Tc_park", "주차 중 냉각수", "°C", 60.0, "°C", vmin=-40, vmax=120, source="ASSUMED", group="mission"),
    Param("hyst", "rainflow 히스테리시스", "K", 0.2, "K", vmin=0, vmax=5, source="ASSUMED", group="사이클·proxy"),
    Param("m_cm", "proxy ΔT 지수 m", "", 5.0, "", vmin=1, vmax=10, source="ASSUMED", source_note="보정되지 않은 모양 계수", group="사이클·proxy"),
    Param("Ea_eV", "proxy 활성화 에너지", "eV", 0.8, "eV", vmin=0, vmax=2, source="ASSUMED", group="사이클·proxy"),
    Param("dT_ref", "proxy 기준 ΔT", "K", 10.0, "K", vmin=0.1, vmax=100, source="ASSUMED", group="사이클·proxy"),
    Param("T_ref_d", "proxy 기준 평균온도", "°C", 100.0, "°C", vmin=-40, vmax=250, source="ASSUMED", group="사이클·proxy"),
    Param("n_mc", "MC 표본 수", "", 30, "", vmin=5, vmax=400, kind="int", source="ASSUMED", group="Monte Carlo"),
    Param("seed", "MC seed", "", 2026, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group="Monte Carlo"),
    Param("sig_R", "R_th 로그정규 σ", "", 0.05, "", vmin=0, vmax=0.5, source="ASSUMED", group="Monte Carlo"),
    Param("sig_alpha", "α 상대 σ", "", 0.10, "", vmin=0, vmax=1, source="ASSUMED", group="Monte Carlo"),
    Param("sig_P", "손실 배율 σ", "", 0.03, "", vmin=0, vmax=0.5, source="ASSUMED", group="Monte Carlo"),
    Param("sig_Tc", "냉각수 offset σ", "K", 2.0, "K", vmin=0, vmax=20, source="ASSUMED", group="Monte Carlo"),
]

EXPERIMENTS = [
    Experiment(
        key="thermal_matrix",
        title="열 행렬: 고정 24/25 K → 연성 26.57/27.75 K, spectral radius 0.098",
        goal=(
            "Z = [[0.2, 0.05], [0.05, 0.25]] K/W, P₀ = [100, 80] W, α = 0.004/K에서 고정 상승 [24, 25] K, 연성 상승 [26.569592, 27.751513] K, junction 91.57/92.75 °C와 "
            "되먹임 spectral radius 0.09789를 재현하고, 반복·물리 망 정상상태로 확인한다. ρ ≥ 1의 발산 경우를 수치 방법과 물리를 구분해 본다."
        ),
        params=[
            Param("Z11", "Z₁₁ (self)", "K/W", 0.2, "K/W", vmin=1e-3, vmax=10, source="TEXTBOOK", source_note="0.20 K/W"),
            Param("Z12", "Z₁₂ = Z₂₁ (cross)", "K/W", 0.05, "K/W", vmin=0, vmax=10, source="TEXTBOOK", source_note="0.05 K/W"),
            Param("Z22", "Z₂₂ (self)", "K/W", 0.25, "K/W", vmin=1e-3, vmax=10, source="TEXTBOOK", source_note="0.25 K/W"),
            Param("P1", "P₀₁", "W", 100.0, "W", vmin=0, vmax=1e5, source="TEXTBOOK", source_note="100 W"),
            Param("P2", "P₀₂", "W", 80.0, "W", vmin=0, vmax=1e5, source="TEXTBOOK", source_note="80 W"),
            Param("alpha", "손실 온도계수 α", "1/K", 0.004, "1/K", vmin=0, vmax=0.2, source="TEXTBOOK", source_note="0.004/K"),
            Param("Tb", "경계 온도", "°C", 65.0, "°C", vmin=-40, vmax=200, source="TEXTBOOK", source_note="65 °C"),
            Param("T_valid", "손실 모델 적용범위 상한", "°C", 175.0, "°C", vmin=50, vmax=400, source="ASSUMED", source_note="선형 α를 믿는 온도 상한 (합성)"),
        ],
        presets=[
            Preset("textbook", "교재 Z·P₀·α", {}, "E08 합성 예", ("nominal", "reference")),
            Preset("near", "α 0.035/K (ρ ≈ 0.86)", {"alpha": 0.035}, "민감도 급증 — 적용범위 밖", ("corner", "reference")),
            Preset("diverge", "α 0.05/K (ρ ≈ 1.22)", {"alpha": 0.05}, "안정 해 없음", ("failure", "reference")),
            Preset("no_cross", "Z₁₂ = 0 (결합 없음)", {"Z12": 0.0}, "교차 항의 기여", ("variant",)),
        ],
        run=run_matrix,
        model_level="A (정상상태 열 행렬)",
        suggested_change="α를 0.004 → 0.05 /K로 올린다.",
        prediction=Prediction(
            "α = 0.05/K로 올리면 선형 풀이 (I − Z·diag(αP₀))⁻¹Z·P₀는?",
            ["조금 더 큰 양의 상승", "매우 큰 양의 상승", "숫자는 나오지만 물리적 정상상태가 아니다 (ρ ≥ 1)", "모르겠다"],
            "숫자는 나오지만 물리적 정상상태가 아니다 (ρ ≥ 1)",
            "되먹임 행렬의 spectral radius가 1.22 > 1이라 반복은 발산하고 물리 망의 과도는 불안정하다. 선형 풀이는 −100/−120 K 같은 비물리적 해를 준다.",
            ["rho", "cp1"],
            handcalc=[{"key": "fix1", "label": "고정 상승 ΔT₁", "unit": "K"}, {"key": "cp1", "label": "연성 상승 ΔT₁", "unit": "K"}],
        ),
        suggested={"alpha": 0.05},
        student="두 소자가 같은 방열판을 쓰면 서로의 열이 서로를 데운다. 거기에 뜨거워질수록 손실이 늘면 온도가 조금 더 오른다. 이 되먹임이 너무 강하면 멈출 온도가 없다.",
        expert=(
            "① ΔT = Z·P에서 diagonal은 self, off-diagonal은 이웃 heating — 경계(냉각수 입구·baseplate·ambient)를 같은 기준으로. ② 연성: [I − Z·diag(αP₀)]ΔT = Z·P₀, 반복의 수렴률 = spectral radius. "
            "③ ρ ≥ 1: 반복 발산, 물리 망 불안정, 선형 풀이는 비물리적 숫자 — 모두 ‘이 단순 모델에 안정 정상해 없음’. ④ 발산이 곧 실물 runaway는 아니다: 수치방법·경계·비선형 손실 범위를 먼저 본다. "
            "⑤ 독립 확인: 2×2 풀이 vs port 행렬이 Z인 8노드 물리 망의 정상상태."
        ),
        customer_ko="인접 소자의 열(교차 항)과 온도에 따른 손실 증가를 함께 넣으면 T_j가 91.6/92.8 °C로 고정 손실 계산보다 2.6~2.8 K 높습니다. 지금 되먹임은 약하지만(0.098), 손실 온도계수가 크거나 냉각이 약해지면 급격히 민감해지니 실제 손실의 온도 곡선과 경계 정의를 확인하시죠.",
        customer_en="With the neighbour's heating and the temperature-dependent loss included, the junctions reach about 91.6 and 92.8 °C, 2.6 to 2.8 K above the fixed-loss estimate. The feedback is weak here, 0.098, but it becomes very sensitive with a larger temperature coefficient or weaker cooling, so let's check the real loss-versus-temperature curves and the boundary definition.",
        questions=[_Q[0], _Q[1]],
        circuit="thermal_matrix",
        textbook=[TB_E08, TB_06],
        reference_presets=["textbook", "near", "diverge"],
        claim_limit="정상상태 열 행렬·선형 온도계수. 실제 모듈 열 행렬·파손온도를 주장하지 않는다.",
    ),
    Experiment(
        key="dynamic_network",
        title="동적 self/cross 열망: 물리 경계와 Foster 함정",
        goal=(
            "port 행렬이 Z와 같은 물리 multiport 열망(두 die Cauer stack + 공유 baseplate·방열판)으로 self·cross 계단 응답, 상반성, 냉각수 계단을 보고, "
            "cross 임피던스를 Foster로 표현·연결할 때의 함정(음수 항, 내부 노드 주입)을 정량화한다."
        ),
        params=[
            Param("Z11", "Z₁₁", "K/W", 0.2, "K/W", vmin=1e-3, vmax=10, source="TEXTBOOK"),
            Param("Z12", "Z₁₂", "K/W", 0.05, "K/W", vmin=1e-4, vmax=10, source="TEXTBOOK"),
            Param("Z22", "Z₂₂", "K/W", 0.25, "K/W", vmin=1e-3, vmax=10, source="TEXTBOOK"),
            Param("Cb", "baseplate 열용량", "J/K", 30.0, "J/K", vmin=0.1, vmax=1e5, source="ASSUMED"),
            Param("Ch", "방열판 열용량", "J/K", 200.0, "J/K", vmin=0.1, vmax=1e6, source="ASSUMED"),
            Param("P1", "P₁ (냉각수 계단 시험)", "W", 100.0, "W", vmin=0, vmax=1e5, source="TEXTBOOK", group="냉각수 계단"),
            Param("P2", "P₂", "W", 80.0, "W", vmin=0, vmax=1e5, source="TEXTBOOK", group="냉각수 계단"),
            Param("Tb", "냉각수 초기", "°C", 65.0, "°C", vmin=-40, vmax=150, source="TEXTBOOK", group="냉각수 계단"),
            Param("dTc", "냉각수 계단", "K", 20.0, "K", vmin=-50, vmax=80, source="TEXTBOOK", source_note="65→85 °C", group="냉각수 계단"),
            Param("t_cool", "계단 시각", "s", 5.0, "s", vmin=0, vmax=1e3, source="ASSUMED", group="냉각수 계단"),
            Param("t_after", "계단 후 관찰", "s", 60.0, "s", vmin=1, vmax=1e4, source="ASSUMED", group="냉각수 계단"),
            Param("t_end", "계단 응답 관찰 끝", "s", 200.0, "s", vmin=1, vmax=1e5, source="ASSUMED", group="해석"),
            Param("n_fit", "Foster 피팅 항 수", "", 4, "", vmin=2, vmax=6, kind="int", source="ASSUMED", group="해석"),
        ],
        presets=[
            Preset("nominal", "교재 Z, baseplate 30 J/K, 방열판 200 J/K", {}, "", ("nominal", "reference")),
            Preset("light_sink", "방열판 20 J/K", {"Ch": 20.0}, "cross 응답이 빨라짐", ("variant", "reference")),
            Preset("fit3", "Foster 3항", {"n_fit": 3}, "피팅 오차", ("variant",)),
        ],
        run=run_dynamic_network,
        model_level="B (물리 multiport 열망)",
        suggested_change="방열판 열용량을 200 → 20 J/K로 줄인다.",
        prediction=Prediction(
            "P₁ 계단에서 소자 2의 온도(cross)는 소자 1(self)과 비교해 언제 오르나?",
            ["동시에 같은 모양으로", "공유 경로의 열용량을 거쳐 늦게", "전혀 오르지 않는다", "모르겠다"],
            "공유 경로의 열용량을 거쳐 늦게",
            "이웃 열은 baseplate·방열판을 데운 뒤에야 도달한다. 방열판 열용량을 줄이면 cross 응답이 빨라진다. 이런 지연 응답은 양수 Foster 항만으로 표현하기 어렵다.",
            ["t63_self", "t63_cross", "n_neg"],
        ),
        suggested={"Ch": 20.0},
        student="한 소자의 열은 먼저 자기 칩과 접합층을 데우고, 한참 뒤에 공유 판과 방열판을 거쳐 옆 소자에 도착한다. 그래서 옆 소자의 온도는 늦게, 천천히 오른다.",
        expert=(
            "① 물리 경계: die Cauer stack(열적 접지로 가는 C)과 공유 baseplate·방열판 → 냉각수. port 행렬 = Z. ② 상반성 Z₁₂(t) = Z₂₁(t)는 수동 RC 망의 성질 — 시간별로 따로 피팅한 cross 모델은 이것을 깨뜨릴 수 있다. "
            "③ cross 응답은 지연되어 양수 Foster로 맞추기 어렵고, 부호 자유 피팅은 음수 R을 만든다 — 연결 가능한 망이 아니다. ④ self Foster의 내부 노드에 이웃 열을 넣으면 정상값을 맞춰도 과도가 틀린다. "
            "⑤ 손실 계단·cross heating·냉각수 과도를 따로 검증한다."
        ),
        customer_ko="이웃 소자의 영향은 정상상태 0.05 K/W지만 수 초에 걸쳐 늦게 나타납니다. 데이터시트 Foster 계수를 이어 붙여 교차 영향을 만들면 과도가 틀리니, 모듈 단위 multiport 열 모델이나 cross 측정 데이터로 확인하시죠.",
        customer_en="The neighbour's influence is 0.05 K/W at steady state but arrives over several seconds. Chaining datasheet Foster coefficients to model it gets the transient wrong, so let's use a module-level multiport thermal model or measured cross-heating data.",
        questions=[_Q[3]],
        circuit="thermal_multiport",
        textbook=[TB_E08, TB_06],
        reference_presets=["nominal", "light_sink"],
        runtime_hint="seconds",
        claim_limit="합성 층값의 물리 multiport 열망. 실제 모듈의 cross 경로·계면 값을 주장하지 않는다.",
    ),
    Experiment(
        key="mission",
        title="mission·냉각수·분산: 온도 사이클 histogram과 상대 손상 proxy",
        goal=(
            "합성 mission(도심·고속·등판·주차, 냉각수 warm-up·변화)을 물리 열망으로 풀어 손실만·비연성·연성 관점을 비교하고, rainflow로 ΔT_j/T_mean histogram을 만든다. "
            "seed와 분포를 명시한 parameter spread(MC)로 상대 손상 proxy의 퍼짐을 보고, calibrated 수명(년)은 MISSING_INPUT으로 둔다."
        ),
        params=_MISSION_PARAMS,
        presets=[
            Preset("nominal", "합성 mission, α 0.004/K, MC 30개 (seed 2026)", {}, "", ("nominal", "reference")),
            Preset("hot_hill", "등판 중 냉각수 +20 K", {"dTc_hill": 20.0}, "냉각수 변화", ("corner", "reference")),
            Preset("strong_alpha", "α 0.012/K", {"alpha": 0.012}, "연성 효과 증가", ("variant",)),
            Preset("wide_spread", "분산 2배 (σ_R 0.1, σ_α 0.2)", {"sig_R": 0.10, "sig_alpha": 0.20}, "proxy 퍼짐", ("variant",)),
            Preset("runaway", "α 0.03/K: 고부하 구간 되먹임 불안정", {"alpha": 0.03}, "평형 없는 구간", ("failure", "reference")),
        ],
        run=run_mission,
        model_level="B (정확 선형 열망) + 사이클 계수 (상대 proxy)",
        suggested_change="등판 중 냉각수 상승을 5 → 20 K로 올린다.",
        prediction=Prediction(
            "등판 중 냉각수가 더 오르면(5 → 20 K) 상대 손상 proxy는?",
            ["거의 그대로", "평균 온도와 큰 사이클이 함께 늘어 증가", "감소", "모르겠다"],
            "평균 온도와 큰 사이클이 함께 늘어 증가",
            "등판 구간의 T_j가 더 높아지고 뒤따르는 도심 구간과의 온도차(ΔT)도 커진다. proxy는 ΔT^m과 평균온도(Arrhenius) 모두에 민감하다. 단, 이것은 상대 비교이며 수명(년)이 아니다.",
            ["D_ratio", "Tpk_c", "dT_max"],
        ),
        suggested={"dTc_hill": 20.0},
        student="차가 가속하고 멈출 때마다 칩 온도가 오르내린다. 이 반복(열 사이클)이 접합부를 피로하게 한다. 사이클의 크기와 횟수를 세는 것은 할 수 있지만, 그것을 ‘몇 년’으로 바꾸려면 검증된 수명 모델이 필요하다.",
        expert=(
            "① 세 관점: 손실만(∫P dt), 비연성 열모델(고정 손실), 연성 열모델(P(T_j)) — 같은 mission에서 결론이 달라진다. ② rainflow(ASTM E1049)로 ΔT_j·T_mean·count를 뽑는다; 히스테리시스가 작은 사이클 수를 바꾼다. "
            "③ proxy는 Coffin–Manson–Arrhenius 모양이지만 계수는 ASSUMED — 같은 식끼리의 비만 의미가 있다. ④ MC는 seed·분포·상관(독립 가정)·모델 오차 미포함을 명시한다. "
            "⑤ calibrated Nf 모델이 없으면 수명(년) MISSING_INPUT; IGBT wire-bond 계수를 SiC package에 쓰지 않는다. DC-link capacitor·solder 등 다른 고장 기구는 별도다."
        ),
        customer_ko="이 mission에서 온도 의존 손실을 넣으면 최고 T_j와 큰 열 사이클이 늘어 상대 손상 지표가 커집니다. 다만 보정된 수명 모델과 데이터가 없어 ‘몇 년’은 말씀드릴 수 없고, 사이클 histogram과 상대 비교까지만 제시하겠습니다. package별 power cycling 데이터가 있으면 그 조건으로 다시 평가하시죠.",
        customer_en="Including the temperature-dependent loss raises the peak junction temperature and the large thermal cycles in this mission, so the relative damage indicator increases. Without a calibrated lifetime model and data I cannot state a number of years; I will give the cycle histogram and relative comparisons, and re-evaluate with package-specific power-cycling data.",
        questions=[_Q[2]],
        circuit="thermal_multiport",
        textbook=[TB_E08],
        reference_presets=["nominal", "hot_hill"],
        runtime_hint="seconds",
        claim_limit="합성 mission의 온도 사이클과 상대 손상 proxy. 수명(년)·보증을 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="EX08",
    title="전열 연성·mission — 최고온도 한 점에서 mission으로",
    title_en="Electro-thermal coupling and mission",
    track="expert",
    order=8,
    path_note="E13 5회전 (E08–E11)",
    textbook=[TB_E08],
    prerequisites=["FL02", "FL03", "EX01"],
    summary="열 행렬(24/25 K → 26.57/27.75 K, ρ 0.098) → 물리 multiport 동적 열망과 Foster 함정 → mission·냉각수·MC 분산의 사이클 histogram과 상대 proxy (수명 년수는 MISSING_INPUT).",
    experiments=EXPERIMENTS,
    minimum_scope="coupled temperature와 calibrated lifetime 한계 (E13 표): fixed 24/25 K, coupled 26.569592/27.751513 K, spectral radius 0.09789; 동적 self/cross physical boundary; mission/coolant/parameter spread; loss-only·uncoupled·coupled 비교",
    claim_limits=[
        "열 행렬·α는 합성 예 — 실제 모듈 값이 아님",
        "Foster 내부 노드를 실제 package 층으로 취급하지 않음",
        "발산(ρ ≥ 1)은 단순 모델의 결과이며 실물 열폭주 확정이 아님",
        "lifetime 데이터가 없으므로 온도 사이클 histogram과 상대 proxy만 — calibrated 수명(년) 생성하지 않음",
        "MC 분포·상관은 가정 (seed 명시)",
    ],
    test_paths=["tests/test_ex08.py"],
    extends=["FL02", "FL03", "EX01"],
)
