"""EX01 - Design space, loss maps and part selection: beyond the single best-efficiency point (textbook E01).

The question is never "which candidate is best at one point" but "which candidate satisfies every
constraint at every required corner, and is the difference between candidates larger than what the
models can resolve".  Everything here is A level (closed forms, FHA, piecewise-linear waveforms,
static electro-thermal fixed points) on synthetic learning data:

  * loss_surface   datasheet-like points -> a switching-energy surface E(V, I): fit and validation
                   points kept apart, a Coss floor at I -> 0, the characterised box, extrapolated
                   queries flagged OUT_OF_VALIDITY and never ranked;
  * obc_cllc       capstone A (11 kW OBC CLLC, 850 V link / 650-920 V battery): FHA gain window,
                   current, flux, ZVS screen and T_j as simultaneous constraints over (V_bat, P);
                   seed n = 1 FAIL at 920/850 V, n = 0.93 two FHA roots, grid limit 10.6988 kW;
  * dab_lv         capstone B (900 V <-> LV DAB, 2 x 1.5 kW): SPS power, RMS, ZVS charge screen and
                   T_j over (V_L, P); phi = 0 at 900/600 V still carries 2.16506 A;
  * sic_leg        capstone C (4-parallel SiC leg): static and electro-thermal sharing
                   (110.553/99.497/99.497/90.452 A), overshoot and common-source coupling
                   (2 nH x 2 kA/us = 4 V) against the gate resistance;
  * mission_ranking  A/B under a switching-loss model uncertainty (8 W vs +-15 W ->
                   UNRESOLVED_RANKING), mission efficiency = sum E_out / sum E_in, and a loss-accounting
                   table with double-counting checks.

Independent paths: noise-free recovery of the synthetic truth, delta-method uncertainty vs Monte Carlo
refits, impedance algebra vs a chain-matrix FHA, the network power identity, a vectorised bisection vs
brentq, segment formulas vs the PWL engine and the SPS closed form, fixed-point iteration vs Newton,
Gauss-Legendre vs the Beta-function sine average, energy sums vs the harmonic-mean identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, fsolve, least_squares

from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..model.units import fmt_si
from ..reference import design_space as ref
from ..reference import psfb as ref_dab

TB_E01 = TextbookRef("expert-e01.-설계영역손실-지도부품선정-한-점의-최고효율에서-벗어나기-ex01", "E01. 설계영역·손실 지도·부품선정 [EX01]")
TB_E12 = TextbookRef("expert-e12-설계-리뷰를-통과하는-답변-세-개의-통합-사례", "E12. 설계 리뷰: 세 개의 통합 사례")
TB_13 = TextbookRef("cllc-설계-실패와-다중-해를-직접-방어하기-fl10", "13. CLLC 설계 실패와 다중 해 [FL10]")
TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB 식에서 파형으로 [FL08]")
TB_E03 = TextbookRef("expert-e03.-병렬-sicgate-loopdpt-계측-사용자-강점의-전문가-확장-ex03", "E03. 병렬 SiC·gate loop·DPT 계측")

# category colours shared by every operating-region map (feasible is always green)
C_FEAS, C_SCREEN, C_NE, C_UNK = "var(--c3)", "var(--c6)", "var(--c8)", "var(--c7)"
C_CONS = ["var(--c5)", "var(--c2)", "var(--c4)", "var(--c1)", "var(--c5)", "var(--c2)"]


# ======================================================================================
# Synthetic switching-energy surfaces
# ======================================================================================


@dataclass(frozen=True)
class LossMap:
    """Synthetic switching energy per event E(V, I) = E_oss (V/V0)^g + E0 (I/I0)^a (V/V0)^b  [J].

    The first term is a Coss-related floor that stays at I -> 0 (textbook E01: a pure power law in I
    removes it and distorts light-load comparisons).  The surface is characterised only on the box
    [V_lo, V_hi] x [I_lo, I_hi]; outside it every value is an extrapolation.
    """

    name: str
    E0: float
    a: float
    b: float
    Eoss: float = 0.0
    g: float = 1.5
    I0: float = 100.0
    V0: float = 800.0
    V_lo: float = 0.0
    V_hi: float = math.inf
    I_lo: float = 0.0
    I_hi: float = math.inf

    def energy(self, V, I):
        V = np.asarray(V, dtype=float)
        I = np.maximum(np.asarray(I, dtype=float), 0.0)
        return self.Eoss * (V / self.V0) ** self.g + self.E0 * (I / self.I0) ** self.a * (V / self.V0) ** self.b

    def inside(self, V, I):
        V = np.asarray(V, dtype=float)
        I = np.asarray(I, dtype=float)
        e = 1e-9
        return (V >= self.V_lo * (1 - e)) & (V <= self.V_hi * (1 + e)) & (I >= self.I_lo * (1 - e)) & (I <= self.I_hi * (1 + e))

    def box_text(self) -> str:
        return f"V {self.V_lo:g}–{self.V_hi:g} V, I {self.I_lo:g}–{self.I_hi:g} A"


# Experiment 1 synthetic truth (1200 V SiC-like, hard switching E_on + E_off at the reference gate)
TRUTH_A = LossMap("A", E0=3.0e-3, a=1.15, b=1.35, Eoss=0.45e-3)
TRUTH_B = LossMap("B", E0=3.6e-3, a=1.05, b=1.25, Eoss=0.20e-3)
V_CHAR = (400.0, 600.0, 800.0)  # characterisation voltages; 400 and 800 V fit, 600 V validation
V_FIT, V_VAL = (400.0, 800.0), (600.0,)


# ======================================================================================
# Shared helpers: constraint maps, candidate tables, ranking under uncertainty
# ======================================================================================


def classify(margins: dict[str, np.ndarray], unknown: np.ndarray, extra: dict[str, np.ndarray] | None = None, loss_keys=("tj",)) -> np.ndarray:
    """Per point category, in this order of precedence:

    1. a special category from ``extra`` (e.g. burst operation outside the model);
    2. the most violated constraint that does not depend on the loss map (normalised g > 0);
    3. 'unknown' when the loss map had to be extrapolated (the loss-dependent limits cannot be judged);
    4. the most violated loss-dependent constraint (e.g. T_j);
    5. 'feasible' (every g <= 0).
    """
    names = list(margins)

    def worst_of(keys):
        if not keys:
            n = len(unknown)
            return np.full(n, -np.inf), np.full(n, "", dtype=object)
        G = np.column_stack([np.where(np.isfinite(margins[k]), margins[k], -np.inf) for k in keys])
        i = np.argmax(G, axis=1)
        return G[np.arange(G.shape[0]), i], np.array(keys, dtype=object)[i]

    g1, k1 = worst_of([k for k in names if k not in loss_keys])
    g2, k2 = worst_of([k for k in names if k in loss_keys])
    lab = np.where(g2 > 0, k2, "feasible").astype(object)
    lab = np.where(unknown, "unknown", lab)
    lab = np.where(g1 > 0, k1, lab)
    for k, mask in (extra or {}).items():
        lab = np.where(mask, k, lab)
    return lab


def tightest(margins: dict[str, float]) -> tuple[str, float]:
    """(constraint, g) with the largest normalised g (the active or nearest-active limit)."""
    finite = {k: v for k, v in margins.items() if v is not None and math.isfinite(v)}
    if not finite:
        return "—", float("nan")
    k = max(finite, key=lambda x: finite[x])
    return k, finite[k]


def add_region_plot(res: Result, key: str, title: str, x, y, cats, labels_ko: dict[str, str], colors: dict[str, str], *, x_label, x_unit, y_label, y_unit, markers, hlines=None, proved="", not_yet="", level="A"):
    """Operating-region map as one points series per category (the renderer draws no heat maps)."""
    keys = []
    for cat in labels_ko:
        m = cats == cat
        if not np.any(m):
            continue
        sk = f"{key}_{cat}"
        res.add_series(sk, f"{labels_ko[cat]} ({int(np.sum(m))}점)", "", np.asarray(x)[m].tolist(), np.asarray(y)[m].tolist(), style="points", color=colors.get(cat))
        keys.append(sk)
    res.add_plot(key, title, keys, x_label=x_label, x_unit=x_unit, y_label=y_label, y_unit=y_unit, kind="xy", level=level, markers=markers, hlines=hlines or [], proved=proved, not_yet=not_yet)


def add_loss_map(res: Result, key: str, title: str, xg, yg, cats, z, labels_ko: dict[str, str], *, z_label, z_unit, x_label, x_unit, y_label, y_unit, ok=("feasible",), cell_notes=None, markers=(), proved="", not_yet="", level="A"):
    """Heat map of z over the admissible cells (category in ``ok``); every other cell is hatched and its
    reason (the active limit) is shown on hover.  Points come from meshgrid(xg, yg, indexing='ij')."""
    nx, ny = len(xg), len(yg)
    C = np.asarray(cats, dtype=object).reshape(nx, ny)
    Z = np.asarray(z, dtype=float).reshape(nx, ny)
    N = None if cell_notes is None else np.asarray(cell_notes, dtype=object).reshape(nx, ny)
    zz, nn = [], []
    for iy in range(ny):
        zr, nr = [], []
        for ix in range(nx):
            c = C[ix, iy]
            good = c in ok and math.isfinite(Z[ix, iy])
            zr.append(float(Z[ix, iy]) if good else None)
            txt = labels_ko.get(c, str(c))
            if N is not None and N[ix, iy]:
                txt += " · " + str(N[ix, iy])
            nr.append(txt)
        zz.append(zr)
        nn.append(nr)
    sk = f"{key}_z"
    res.add_series(sk, z_label, z_unit, [float(x) for x in xg], [float(y) for y in yg], style="map", z=zz, notes=nn)
    res.add_plot(key, title, [sk], x_label=x_label, x_unit=x_unit, y_label=y_label, y_unit=y_unit, kind="map", level=level, markers=list(markers), proved=proved, not_yet=not_yet)


def rank_under_uncertainty(dP: float, u_diff: float, k: float, name_a: str = "A", name_b: str = "B") -> tuple[str, bool]:
    """dP = loss_B - loss_A (> 0: A better).  Resolved only when |dP| > k u_diff."""
    if abs(dP) > k * u_diff:
        return (name_a if dP > 0 else name_b), True
    return "UNRESOLVED_RANKING", False


# ======================================================================================
# Experiment 1: from datasheet points to a loss surface
# ======================================================================================


def _power_fit(V, I, E, I0=100.0, V0=800.0):
    """log E = c0 + a ln(I/I0) + b ln(V/V0) by linear least squares; returns (coef, cov, s)."""
    X = np.column_stack([np.ones_like(V), np.log(I / I0), np.log(V / V0)])
    y = np.log(E)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    dof = max(len(y) - 3, 1)
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef, cov, math.sqrt(s2)


def _power_eval(coef, cov, V, I, I0=100.0, V0=800.0, s_known=None, X_fit=None):
    x = np.array([1.0, math.log(I / I0), math.log(V / V0)])
    E = math.exp(float(x @ coef))
    if s_known is not None and X_fit is not None:
        cov = s_known**2 * np.linalg.inv(X_fit.T @ X_fit)
    return E, E * math.sqrt(float(x @ cov @ x))


def _floor_model(p, V, I, g=1.5, I0=100.0, V0=800.0):
    return np.exp(p[0]) * (V / V0) ** g + np.exp(p[1]) * (I / I0) ** p[2] * (V / V0) ** p[3]


def _floor_fit(V, I, E, p0=None, g=1.5):
    """E = E_oss (V/V0)^g + E0 (I/I0)^a (V/V0)^b by nonlinear least squares on relative residuals."""
    if p0 is None:
        c, _, _ = _power_fit(V, I, E)
        p0 = np.array([math.log(0.3 * float(np.min(E))), c[0], c[1], c[2]])

    def resid(p):
        return (_floor_model(p, V, I, g) - E) / E

    sol = least_squares(resid, p0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    dof = max(len(E) - 4, 1)
    s2 = float(sol.fun @ sol.fun) / dof
    JtJ = sol.jac.T @ sol.jac
    return sol.x, s2 * np.linalg.inv(JtJ), math.sqrt(s2), JtJ


def _floor_eval(p, cov, V, I, g=1.5, I0=100.0, V0=800.0):
    t_oss = math.exp(p[0]) * (V / V0) ** g
    t_sw = math.exp(p[1]) * (I / I0) ** p[2] * (V / V0) ** p[3]
    grad = np.array([t_oss, t_sw, t_sw * math.log(I / I0), t_sw * math.log(V / V0)])
    return t_oss + t_sw, math.sqrt(max(float(grad @ cov @ grad), 0.0))


def _char_data(truth: LossMap, I_levels, noise, rng):
    VV, II = np.meshgrid(np.array(V_CHAR), I_levels, indexing="ij")
    VV, II = VV.ravel(), II.ravel()
    Et = truth.energy(VV, II)
    Em = Et * (1.0 + noise * rng.standard_normal(Et.size))
    return VV, II, Et, Em


def run_loss_surface(v: dict) -> Result:
    res = Result("EX01", "loss_surface", "A (합성 loss surface fit·validation)")
    rng = np.random.default_rng(int(v["seed"]))
    I_lo, I_hi = v["I_lo_char"], 160.0
    if I_lo >= I_hi / 2:
        res.verdict("OUT_OF_VALIDITY", "특성화 전류 하한이 상한의 절반 이상이다 — 전류축 fit이 성립하지 않는다")
        return res
    I_levels = np.geomspace(I_lo, I_hi, 6)
    truths = {"A": replace(TRUTH_A, Eoss=v["Eoss_A"]), "B": replace(TRUTH_B, Eoss=v["Eoss_B"])}
    noise, k = v["noise"], v["k_cov"]
    Vq, Iq = v["query_V"], v["query_I"]
    box = LossMap("box", 0, 0, 0, V_lo=min(V_CHAR), V_hi=max(V_CHAR), I_lo=I_lo, I_hi=I_hi)
    inside = bool(box.inside(Vq, Iq))
    fits, data = {}, {}
    for nm, tr in truths.items():
        VV, II, Et, Em = _char_data(tr, I_levels, noise, rng)
        fit_m = np.isin(VV, V_FIT)
        val_m = np.isin(VV, V_VAL)
        cpow, covp, sp = _power_fit(VV[fit_m], II[fit_m], Em[fit_m])
        pfl, covf, sf, JtJ = _floor_fit(VV[fit_m], II[fit_m], Em[fit_m])
        pred_f = _floor_model(pfl, VV, II)
        pred_p = np.exp(np.column_stack([np.ones_like(VV), np.log(II / 100.0), np.log(VV / 800.0)]) @ cpow)
        ef = (pred_f - Em) / Em
        ep = (pred_p - Em) / Em
        fits[nm] = dict(cpow=cpow, covp=covp, sp=sp, pfl=pfl, covf=covf, sf=sf, JtJ=JtJ)
        data[nm] = dict(VV=VV, II=II, Et=Et, Em=Em, fit=fit_m, val=val_m, ef=ef, ep=ep)
    # query with the chosen model
    use_floor = v["rank_model"] == "floor"
    q = {}
    for nm in ("A", "B"):
        f = fits[nm]
        Ef, uf = _floor_eval(f["pfl"], f["covf"], Vq, Iq)
        Ep, up = _power_eval(f["cpow"], f["covp"], Vq, Iq)
        q[nm] = (Ef, uf, Ep, up)
    EA, uA = (q["A"][0], q["A"][1]) if use_floor else (q["A"][2], q["A"][3])
    EB, uB = (q["B"][0], q["B"][1]) if use_floor else (q["B"][2], q["B"][3])
    dE = EB - EA
    u_d = math.hypot(uA, uB)
    winner, resolved = rank_under_uncertainty(dE, u_d, k)
    val_f = max(float(np.sqrt(np.mean(data[n]["ef"][data[n]["val"]] ** 2))) for n in "AB")
    val_p = max(float(np.sqrt(np.mean(data[n]["ep"][data[n]["val"]] ** 2))) for n in "AB")
    valmax_f = max(float(np.max(np.abs(data[n]["ef"][data[n]["val"]]))) for n in "AB")
    valmax_p = max(float(np.max(np.abs(data[n]["ep"][data[n]["val"]]))) for n in "AB")
    tol_val = v["tol_val"]
    model_ko = "Coss floor + power law" if use_floor else "순수 power law"
    # --- metrics
    pA = fits["A"]["pfl"]
    covA = fits["A"]["covf"]
    res.add_metric("alpha_A", "A: 전류 지수 α (floor 모델 fit)", float(pA[2]), "", ref=TRUTH_A.a, ref_label="합성 참값", tol=max(6 * math.sqrt(covA[2, 2]), 1e-6) / TRUTH_A.a, basis=f"±{math.sqrt(covA[2, 2]):.3f} (1σ) — 관습값이 아니라 fit 결과")
    res.add_metric("beta_A", "A: 전압 지수 β (floor 모델 fit)", float(pA[3]), "", ref=TRUTH_A.b, ref_label="합성 참값", tol=max(6 * math.sqrt(covA[3, 3]), 1e-6) / TRUTH_A.b, basis=f"±{math.sqrt(covA[3, 3]):.3f} (1σ)")
    res.add_metric("Eoss_A", "A: I → 0 floor (800 V)", math.exp(pA[0]), "J", basis="Coss 관련 성분 — 순수 power law에는 없다")
    res.add_metric("val_floor", "검증점(600 V) RMS 오차: floor 모델", val_f, "", basis=f"fit에 쓰지 않은 점; 최대 {valmax_f:.1%}")
    res.add_metric("val_power", "검증점(600 V) RMS 오차: 순수 power law", val_p, "", basis=f"최대 {valmax_p:.1%} (저전류에서 커짐)")
    ratio_lo = _power_eval(fits["A"]["cpow"], fits["A"]["covp"], 800.0, I_lo)[0] / _floor_eval(pA, covA, 800.0, I_lo)[0]
    ratio_ex = _power_eval(fits["A"]["cpow"], fits["A"]["covp"], 800.0, I_lo / 4)[0] / _floor_eval(pA, covA, 800.0, I_lo / 4)[0]
    res.add_metric("light_ratio", f"A: power law / floor 모델 (800 V, I = {I_lo:g} A)", ratio_lo, "", basis=f"I = {I_lo / 4:g} A(외삽)에서는 {ratio_ex:.2f}")
    res.add_metric("E_A", f"E_A at {Vq:g} V, {Iq:g} A ({model_ko})", EA, "J", basis=f"u = {uA * 1e3:.3g} mJ (fit 공분산 전파)")
    res.add_metric("E_B", f"E_B at {Vq:g} V, {Iq:g} A", EB, "J", basis=f"u = {uB * 1e3:.3g} mJ")
    res.add_metric("dE", "E_B − E_A (양수: A가 작다)", dE, "J", basis=f"k·u_Δ = {k * u_d * 1e3:.3g} mJ")
    res.add_metric("ranking", "질의점 순위", ("UNKNOWN (외삽)" if not inside else winner), "", basis="외삽 점은 순위를 매기지 않는다" if not inside else f"|Δ| {'>' if resolved else '≤'} k·u_Δ")
    res.add_metric("query_status", "질의점 상태", "INTERPOLATION" if inside else "EXTRAPOLATION", "", basis=f"특성화 범위 {box.box_text()}")
    # crossover of the two floor fits at the query voltage
    def gap(I):
        return _floor_eval(fits["A"]["pfl"], fits["A"]["covf"], Vq, I)[0] - _floor_eval(fits["B"]["pfl"], fits["B"]["covf"], Vq, I)[0]

    I_x = None
    try:
        if gap(I_lo / 10) * gap(2 * I_hi) < 0:
            I_x = brentq(gap, I_lo / 10, 2 * I_hi, xtol=1e-9)
    except ValueError:
        I_x = None
    res.add_metric("I_cross", f"A/B 교차 전류 ({Vq:g} V, floor fit)", I_x if I_x is not None else "없음", "A" if I_x is not None else "", basis=("특성화 범위 안" if I_x is not None and I_lo <= I_x <= I_hi else "범위 밖 — 확인 필요") if I_x is not None else "")
    # --- independent checks
    VV, II, Et, _ = _char_data(truths["A"], I_levels, 0.0, np.random.default_rng(0))
    m = np.isin(VV, V_FIT)
    p0 = np.array([math.log(truths["A"].Eoss * 3), math.log(truths["A"].E0 * 0.5), 1.0, 1.0])
    pt, *_ = _floor_fit(VV[m], II[m], Et[m], p0=p0)
    e_rec = max(abs(math.exp(pt[0]) / truths["A"].Eoss - 1), abs(math.exp(pt[1]) / truths["A"].E0 - 1), abs(pt[2] - truths["A"].a), abs(pt[3] - truths["A"].b))
    res.add_check(Check("fit 구현: 잡음 없는 합성 참값 회복 (floor 모델)", "PASS" if e_rec < 1e-6 else "FAIL", e_rec, "", 1e-6, path="잡음 0 합성 데이터 → Levenberg–Marquardt fit vs 생성에 쓴 참값", independent=True, detail=f"E_oss, E0, α, β 최대 편차 {e_rec:.1e} (먼 시작점에서 출발)"))
    dA = data["A"]
    c_ls = fits["A"]["cpow"]
    X = np.column_stack([np.ones(int(dA["fit"].sum())), np.log(dA["II"][dA["fit"]] / 100.0), np.log(dA["VV"][dA["fit"]] / 800.0)])
    sol = least_squares(lambda c: X @ c - np.log(dA["Em"][dA["fit"]]), np.zeros(3), method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    res.add_check(check_close("power law: 정규방정식(lstsq) vs 반복 최소제곱(TRF)", float(np.max(np.abs(sol.x - c_ls))), 0.0, 1e-8, "같은 목적함수를 다른 알고리즘으로 풀어 계수 비교", True, "", abs_scale=1.0, detail=f"최대 계수 차 {float(np.max(np.abs(sol.x - c_ls))):.1e}"))
    # delta-method uncertainty vs Monte Carlo refits (known sigma, same design)
    K = int(v["mc_refits"])
    rng2 = np.random.default_rng(int(v["seed"]) + 1)
    Vf, If_ = dA["VV"][dA["fit"]], dA["II"][dA["fit"]]
    Etf = truths["A"].energy(Vf, If_)
    pT = np.array([math.log(truths["A"].Eoss), math.log(truths["A"].E0), truths["A"].a, truths["A"].b])
    Eqs = []
    for _ in range(K):
        Em_k = Etf * (1.0 + noise * rng2.standard_normal(Etf.size))
        pk, *_ = _floor_fit(Vf, If_, Em_k, p0=pT)
        Eqs.append(_floor_model(pk, Vq, Iq))
    u_mc = float(np.std(Eqs, ddof=1))
    # linearised prediction with the known sigma: J^T J at the truth, relative residual weights
    def resid_T(p):
        return (_floor_model(p, Vf, If_) - Etf) / Etf

    Jt = least_squares(resid_T, pT, method="lm", max_nfev=1).jac
    covT = noise**2 * np.linalg.inv(Jt.T @ Jt)
    _, u_lin = _floor_eval(pT, covT, Vq, Iq)
    se_rel = 1 / math.sqrt(2 * (K - 1))
    res.add_check(check_close(f"E_A(질의점) 불확도: delta method vs Monte Carlo 재fit ({K}회)", u_mc, u_lin, max(0.35, 4 * se_rel), "선형화 공분산 전파 vs 잡음을 새로 뽑아 다시 fit한 표준편차 (σ 알려진 조건)", True, "J", detail=f"MC {u_mc * 1e3:.4g} mJ / 선형화 {u_lin * 1e3:.4g} mJ; 허용은 MC 표본 오차 포함"))
    # sanity (not an independent comparison): positive and increasing inside the box
    Ig = np.geomspace(I_lo, I_hi, 25)
    ok_mono = True
    for nm in "AB":
        for Vc in (400.0, 600.0, 800.0):
            Ev = _floor_model(fits[nm]["pfl"], Vc, Ig)
            ok_mono &= bool(np.all(Ev > 0) and np.all(np.diff(Ev) > 0))
        Ev2 = _floor_model(fits[nm]["pfl"], np.linspace(400, 800, 25), 50.0)
        ok_mono &= bool(np.all(np.diff(Ev2) > 0))
    res.add_check(Check("fit 곡면: 특성화 범위 안에서 양수·I와 V에 단조 증가", "PASS" if ok_mono else "FAIL", 1.0 if ok_mono else 0.0, "", path="격자 점검 (물리적 타당성)", independent=False))
    # --- plots
    Ip = np.geomspace(I_lo / 5, 2 * I_hi, 160)
    inr = (Ip >= I_lo) & (Ip <= I_hi)
    for nm in "AB":
        Ef = _floor_model(fits[nm]["pfl"], Vq, Ip) * 1e3
        col = "var(--c1)" if nm == "A" else "var(--c2)"
        res.add_series(f"fit{nm}_in", f"{nm}: fit (특성화 범위)", "mJ", Ip.tolist(), np.where(inr, Ef, np.nan).tolist(), color=col)
        res.add_series(f"fit{nm}_ex", f"{nm}: 외삽 (UNKNOWN)", "mJ", Ip.tolist(), np.where(~inr, Ef, np.nan).tolist(), dash=True, color=col)
    mk = [{"x": Iq, "y": EA * 1e3, "label": f"A {EA * 1e3:.3g} mJ"}, {"x": Iq, "y": EB * 1e3, "label": f"B {EB * 1e3:.3g} mJ"}]
    if I_x is not None:
        mk.append({"x": I_x, "y": float(_floor_model(fits["A"]["pfl"], Vq, I_x)) * 1e3, "label": f"교차 {I_x:.3g} A"})
    res.add_plot("p_ab", f"A/B 스위칭 에너지 at {Vq:g} V (floor 모델 fit)", ["fitA_in", "fitA_ex", "fitB_in", "fitB_ex"], x_label="전류 I", x_unit="A", y_label="E_sw", y_unit="mJ", kind="xy", log_x=True, log_y=True, level="A",
                 vlines=[{"x": I_lo, "label": "측정 하한"}, {"x": I_hi, "label": "측정 상한"}], markers=mk,
                 proved="특성화 범위 안에서 두 후보의 에너지 곡선과 교차 전류를 fit으로 보였다. 경부하에서는 Coss floor가 큰 A가 불리해진다. 질의점 값의 불확도(delta method)는 잡음을 새로 뽑아 다시 fit한 Monte Carlo check와 맞는다.",
                 not_yet="점선(외삽) 구간은 측정이 없다 — 그 영역의 순위는 UNKNOWN이며 합성 참값과의 일치도 보장되지 않는다.")
    for nm in "AB":
        d = data[nm]
        m800 = d["VV"] == 800.0
        res.add_series(f"d{nm}", f"{nm}: 측정점 800 V (fit용)", "mJ", d["II"][m800].tolist(), (d["Em"][m800] * 1e3).tolist(), style="points", color="var(--c1)" if nm == "A" else "var(--c2)")
    If = np.geomspace(I_lo, I_hi, 80)
    for nm in "AB":
        res.add_series(f"ff{nm}", f"{nm}: floor + power law", "mJ", If.tolist(), (_floor_model(fits[nm]["pfl"], 800.0, If) * 1e3).tolist(), color="var(--c1)" if nm == "A" else "var(--c2)")
        c = fits[nm]["cpow"]
        res.add_series(f"fp{nm}", f"{nm}: 순수 power law", "mJ", If.tolist(), (np.exp(c[0] + c[1] * np.log(If / 100.0)) * 1e3).tolist(), dash=True, color="var(--c4)" if nm == "A" else "var(--c5)")  # at V = V0 = 800 V
    res.add_plot("p_form", "모델 형식: 800 V 측정점과 두 fit (저전류 영역을 보라)", ["dA", "ffA", "fpA", "dB", "ffB", "fpB"], x_label="전류 I", x_unit="A", y_label="E_sw", y_unit="mJ", kind="xy", log_x=True, log_y=True, level="A",
                 proved="순수 power law는 저전류에서 곡선이 휘는 Coss 성분을 표현하지 못해 경부하 에너지를 과소평가한다. fit 구현은 잡음 없는 합성 참값을 회복하는 check(편차 ~1e-15)로 확인했으므로 이 차이는 모델 형식 차이다.",
                 not_yet="floor의 전압 지수(1.5)는 가정이다. 실제 소자는 E_oss(V) 곡선과 측정 조건(recovery 포함 여부)을 확인해야 한다.")
    for nm in "AB":
        d = data[nm]
        res.add_series(f"vf{nm}", f"{nm}: floor 모델 오차", "", d["II"][d["val"]].tolist(), d["ef"][d["val"]].tolist(), style="points")
        res.add_series(f"vp{nm}", f"{nm}: power law 오차", "", d["II"][d["val"]].tolist(), d["ep"][d["val"]].tolist(), style="points")
    res.add_plot("p_val", "검증점(600 V, fit 미사용)의 상대 예측 오차", ["vfA", "vfB", "vpA", "vpB"], x_label="전류 I", x_unit="A", y_label="(예측 − 측정)/측정", y_unit="", kind="xy", log_x=True, level="A",
                 hlines=[{"y": tol_val, "label": f"+{tol_val:.0%}"}, {"y": -tol_val, "label": f"−{tol_val:.0%}"}],
                 proved="fit과 validation을 분리하면 모델 형식 오차(순수 power law의 저전류 편향)가 fit 잔차보다 분명히 드러난다. power law 계수는 정규방정식과 반복 최소제곱이 같은 값을 준다는 check로 확인했다.",
                 not_yet="검증점도 같은 합성 소자·같은 측정 setup이다. 온도·gate 조건·다른 lot은 검증하지 않았다.")
    fitpts = (data["A"]["II"][data["A"]["fit"]], data["A"]["VV"][data["A"]["fit"]])
    valpts = (data["A"]["II"][data["A"]["val"]], data["A"]["VV"][data["A"]["val"]])
    res.add_series("cov_fit", "fit 점", "V", fitpts[0].tolist(), fitpts[1].tolist(), style="points")
    res.add_series("cov_val", "검증 점", "V", valpts[0].tolist(), valpts[1].tolist(), style="points")
    res.add_series("cov_box", "특성화 범위 (보간 허용)", "V", [I_lo, I_hi, I_hi, I_lo, I_lo], [400, 400, 800, 800, 400], dash=True, color="var(--c8)")
    res.add_plot("p_cov", "보간 범위와 질의점", ["cov_fit", "cov_val", "cov_box"], x_label="전류 I", x_unit="A", y_label="전압 V", y_unit="V", kind="xy", log_x=True, level="A",
                 markers=[{"x": Iq, "y": Vq, "label": "질의점 (보간)" if inside else "질의점 (외삽 → UNKNOWN)"}],
                 proved="질의점이 측정 격자의 범위 안인지 밖인지를 순위 판정 전에 표시한다.",
                 not_yet="범위 안이라도 격자 사이의 곡률·온도 의존은 보간 가정이다.")
    # --- tables
    res.tables.append(Table("t_truth", "합성 참값 (학습용 가상 소자, 데이터시트 아님)", ["후보", "E0 @ 800 V·100 A [mJ]", "α", "β", "E_oss @ 800 V [mJ]", "floor 전압 지수"],
                            [[nm, t.E0 * 1e3, t.a, t.b, t.Eoss * 1e3, t.g] for nm, t in truths.items()],
                            note="실험의 ‘측정’은 이 참값에 상대 잡음을 더한 합성 데이터다. fit 결과를 참값과 비교할 수 있는 것은 합성이기 때문이다."))
    rows = []
    for nm in "AB":
        f = fits[nm]
        rows.append([nm, "floor + power law", math.exp(f["pfl"][1]) * 1e3, f["pfl"][2], f["pfl"][3], math.exp(f["pfl"][0]) * 1e3, f["sf"], float(np.sqrt(np.mean(data[nm]["ef"][data[nm]["val"]] ** 2)))])
        rows.append([nm, "순수 power law", math.exp(f["cpow"][0]) * 1e3, f["cpow"][1], f["cpow"][2], 0.0, f["sp"], float(np.sqrt(np.mean(data[nm]["ep"][data[nm]["val"]] ** 2)))])
    res.tables.append(Table("t_fit", "fit 결과 (fit 점: 400·800 V, 검증 점: 600 V)", ["후보", "모델", "E0 [mJ]", "α", "β", "E_oss [mJ]", "fit 잔차 RMS", "검증 RMS 오차"], rows))
    res.tables.append(Table("t_query", "질의점 판정", ["후보", "E (floor) [mJ]", "u [mJ]", "E (power law) [mJ]", "u [mJ]", "상태"],
                            [[nm, q[nm][0] * 1e3, q[nm][1] * 1e3, q[nm][2] * 1e3, q[nm][3] * 1e3, "보간" if inside else "외삽 → UNKNOWN"] for nm in "AB"]))
    # --- verdicts
    chosen_val = valmax_f if use_floor else valmax_p
    if not inside:
        res.verdict("OUT_OF_VALIDITY", f"질의점 ({Vq:g} V, {Iq:g} A)이 특성화 범위({box.box_text()}) 밖이다: 외삽 값은 표시만 하고 순위를 매기지 않는다 (UNKNOWN)")
    elif chosen_val > tol_val:
        res.verdict("OUT_OF_VALIDITY", f"{model_ko} fit의 검증점 최대 오차 {chosen_val:.1%} > 허용 {tol_val:.0%}: 이 fit으로 순위를 매기지 않는다")
    elif not resolved:
        res.verdict("UNRESOLVED_RANKING", f"질의점 차이 {abs(dE) * 1e3:.3g} mJ ≤ k·u_Δ = {k * u_d * 1e3:.3g} mJ: fit 불확도 안에서 순위 미확정")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"질의점에서 {winner}가 {abs(dE) * 1e3:.3g} mJ 작다 (k·u_Δ = {k * u_d * 1e3:.3g} mJ) — 합성 데이터·이 조건 한정")
    res.assumptions += [
        "합성 참값 E = E_oss(V/800)^1.5 + E0(I/100)^α(V/800)^β에 상대 잡음(1σ)을 더한 ‘측정’",
        "floor 전압 지수 1.5 고정(C ∝ V^−1/2 가정); α·β·E0·E_oss는 fit",
        "fit 점 400·800 V, 검증 점 600 V; 모든 점 같은 온도·gate 조건",
    ]
    res.not_valid_for += ["실제 소자의 스위칭 에너지(데이터시트·DPT 조건 확인 필요, MISSING_INPUT)", "특성화 범위 밖 전압·전류(외삽)", "soft-switching(ZVS) 손실 — hard-switching 데이터"]
    res.interpretation = (
        "데이터시트 몇 점을 곡면으로 바꿀 때 α·β는 관습값이 아니라 fit 결과이고, fit에 쓰지 않은 점으로 검증해야 한다. 순수 power law는 I → 0에서 0이 되어 Coss 성분을 지워 "
        "경부하 비교를 뒤집을 수 있다. 측정 범위 밖 질의점은 값이 그럴듯해도 UNKNOWN이며 순위표에 넣지 않는다."
    )
    return res


# ======================================================================================
# Experiment 2: capstone A - 11 kW OBC CLLC isolated stage (FHA operating window)
# ======================================================================================

CLLC_CATS = {
    "feasible": "모든 제약 만족",
    "gain_high": "gain 부족 (f_min~peak에서도 승압 불가)",
    "gain_low": "f_max에서도 gain 과다 (강압 불가)",
    "burst": "burst 영역 (P < P_burst, FHA 밖)",
    "current": "1차 RMS 전류 한계",
    "flux": "자속 한계",
    "zvs": "ZVS 전하 screen 실패",
    "tj": "T_j 한계",
    "unknown": "손실 지도 외삽 (UNKNOWN)",
}
CLLC_COL = {"feasible": C_FEAS, "gain_high": "var(--c5)", "gain_low": "var(--c4)", "burst": C_NE, "current": "var(--c2)", "flux": "var(--c1)", "zvs": C_SCREEN, "tj": "var(--c2)", "unknown": C_UNK}


def cllc_link(Vb, v):
    """dc-link schedule through the textbook battery/link pairs (650/700, 800/800, 920/850 V)."""
    return np.interp(np.asarray(Vb, dtype=float), [650.0, 800.0, 920.0], [v["Vlink_650"], v["Vlink_800"], v["Vlink_920"]])


def cllc_net(f, n, Vb, P, Lr, Cr, Lm):
    """FHA divider algebra, primary-referred, referred-symmetric tank (Lr2' = Lr, Cr2' = Cr)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    Rac = n * n * 8.0 * np.asarray(Vb, dtype=float) ** 2 / (np.pi**2 * np.asarray(P, dtype=float))
    Z1 = 1j * w * Lr + 1.0 / (1j * w * Cr)
    Zm = 1j * w * Lm
    Zb = Z1 + Rac
    Zp = Zm * Zb / (Zm + Zb)
    Zin = Z1 + Zp
    H = Zp / Zin * Rac / Zb
    return H, Zin, Rac


def cllc_device(v) -> LossMap:
    """Synthetic ZVS turn-off energy map of the primary SiC switches (no Coss floor: ZVS turn-off)."""
    return LossMap("OBC E_off", E0=v["Eoff_20A"], a=1.3, b=1.0, Eoss=0.0, I0=20.0, V0=800.0, V_lo=400.0, V_hi=900.0, I_lo=v["Ioff_lo_char"], I_hi=40.0)


def _linear_tj(Tc, Rth, a, tc, Pfix):
    """Steady T of P = a (1 + tc (T - 25)) + Pfix, T = Tc + Rth P (linear PTC); inf when it runs away."""
    den = 1.0 - Rth * a * tc
    T = (Tc + Rth * (a * (1.0 - 25.0 * tc) + Pfix)) / np.where(den > 0, den, np.nan)
    return np.where(den > 0, T, np.inf)


def cllc_eval(n, Vb, P, v, policy="right", nf=721):
    """Vectorised FHA operating point + constraints for points (Vb, P).  Returns a dict of arrays."""
    Vb = np.atleast_1d(np.asarray(Vb, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float))
    Vb, P = np.broadcast_arrays(Vb, P)
    Lr, Cr, Lm = v["Lr"], v["Cr"], v["Lm"]
    Vl = cllc_link(Vb, v)
    M = n * Vb / Vl
    f = np.linspace(v["f_min"], v["f_max"], nf)
    H, Zin, _ = cllc_net(f[None, :], n, Vb[:, None], P[:, None], Lr, Cr, Lm)
    g = np.abs(H)
    ind = np.angle(Zin) > 0
    gmax = np.where(ind, g, 0.0).max(axis=1)
    g_fmax = g[:, -1]
    d = g - M[:, None]
    both = ind[:, :-1] & ind[:, 1:]
    down = (d[:, :-1] > 0) & (d[:, 1:] <= 0) & both
    up = (d[:, :-1] < 0) & (d[:, 1:] >= 0) & both
    kr = np.where(down.any(axis=1), nf - 2 - np.argmax(down[:, ::-1], axis=1), -1)
    kl = np.where(up.any(axis=1), np.argmax(up, axis=1), -1)

    def refine(k):
        ok = k >= 0
        kk = np.where(ok, k, 0)
        lo, hi = f[kk].copy(), f[kk + 1].copy()
        s_lo = np.sign(d[np.arange(len(kk)), kk])
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            dm = np.abs(cllc_net(mid, n, Vb, P, Lr, Cr, Lm)[0]) - M
            same = np.sign(dm) == s_lo
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        return np.where(ok, 0.5 * (lo + hi), np.nan)

    fr, fl = refine(kr), refine(kl)
    V1 = 2.0 * math.sqrt(2.0) / math.pi * Vl

    def at(fop):
        Hh, Zi, Rac = cllc_net(np.where(np.isfinite(fop), fop, v["f_min"]), n, Vb, P, Lr, Cr, Lm)
        I1 = V1 / np.abs(Zi)
        return Hh, Zi, Rac, I1

    if policy == "min_current":
        I1r = np.where(np.isfinite(fr), at(fr)[3], np.inf)
        I1l = np.where(np.isfinite(fl), at(fl)[3], np.inf)
        fop = np.where(I1l < I1r, fl, fr)
    else:
        fop = fr
    root = np.isfinite(fop)
    Hh, Zi, Rac, I1 = at(fop)
    th = np.angle(Zi)
    Vo = np.abs(Hh) * V1
    I2 = n * Vo / Rac
    ioff = math.sqrt(2.0) * I1 * np.sin(th)
    fo = np.where(root, fop, np.nan)
    B = n * Vb / (4.0 * fo * v["Np"] * v["Ae"] * 1e-6)  # Ae in mm²
    dev = cllc_device(v)
    Eoff = dev.energy(Vl, np.maximum(ioff, 0.0))
    Psw_dev = fo * Eoff
    a1 = I1**2 / 2.0 * v["Rds"]
    Tj = _linear_tj(v["T_cool"], v["Rth"], a1, v["tc"], Psw_dev)
    a2 = I2**2 / 2.0 * v["Rds"]
    Tsr = _linear_tj(v["T_cool"], v["Rth"], a2, v["tc"], 0.0)
    P_pri_c = 4.0 * a1 * (1.0 + v["tc"] * (Tj - 25.0))
    P_off = 4.0 * Psw_dev
    P_sr = 4.0 * a2 * (1.0 + v["tc"] * (Tsr - 25.0))
    P_cu = v["R_pri"] * I1**2 + v["R_sec"] * I2**2
    P_core = v["k_core"] * (fo / 100e3) ** 1.4 * (B / 0.1) ** 2.5
    P_gate = 8.0 * v["Qg"] * v["dVg"] * fo
    P_loss = P_pri_c + P_off + P_sr + P_cu + P_core + P_gate
    I_zvs = 2.0 * v["Qoss"] / v["t_dead"]
    nan = np.full(Vb.shape, np.nan)
    # normalised constraints g (<= 0 satisfied).  Without a root the gain window itself is the
    # violated limit: above the inductive peak (gain_high) or still too high at f_max (gain_low);
    # below P_burst the second case is burst operation, outside this FHA model.
    gsafe = np.where(gmax > 0, gmax, np.nan)
    burst = (~root) & (gmax >= M) & (P < v["P_burst"])
    margins = {
        "gain_high": np.where(root | (gmax < M), M / gsafe - 1.0, nan),
        "gain_low": np.where((~root) & (gmax >= M) & ~burst, g_fmax / M - 1.0, nan),
        "current": np.where(root, I1 / v["I1_max"] - 1.0, nan),
        "flux": np.where(root, B / v["B_max"] - 1.0, nan),
        "zvs": np.where(root, 1.0 - ioff / I_zvs, nan),
        "tj": np.where(root, (Tj - v["Tj_max"]) / (v["Tj_max"] - v["T_cool"]), nan),
    }
    margins["gain_high"] = np.where(gmax > 0, margins["gain_high"], np.where(root, nan, np.inf))
    unknown = root & ~dev.inside(Vl, ioff)
    return dict(Vb=Vb, P=P, Vl=Vl, M=M, gmax=gmax, g_fmax=g_fmax, f=fo, fr=fr, fl=fl, root=root, I1=I1, I2=I2, ioff=ioff, th=th, B=B, Tj=Tj, Tsr=Tsr,
                parts=dict(pri_cond=P_pri_c, pri_off=P_off, sr_cond=P_sr, copper=P_cu, core=P_core, gate=P_gate), P_loss=P_loss, margins=margins, burst=burst, unknown=unknown, I_zvs=I_zvs, V1=V1)


def _corner_row(e, i):
    """(category, tightest constraint, its g) of point i of an evaluation dict."""
    mg = {k: np.array([e["margins"][k][i]]) for k in e["margins"]}
    extra = {"burst": np.array([bool(e["burst"][i])])} if "burst" in e else None
    cat = str(classify(mg, np.array([bool(e["unknown"][i])]), extra)[0])
    k, gm = tightest({k: float(e["margins"][k][i]) for k in e["margins"]})
    return cat, k, gm


def cllc_circuit(v: dict, n: float) -> Circuit:
    c = Circuit("ex01_cllc", 880, 330, title=f"CLLC FHA 등가회로 (1차 환산, 대칭 tank, n = {n:.3g})")
    vs = c.add("vsource", "V1", 130, 160, 90, "V_1", "기본파 0.9·V_link", lpos=(112, 156, "end"))
    lr = c.add("inductor", "Lr1", 230, 60, 0, "L_r1", fmt_si(v["Lr"], "H", 4), lpos=(230, 24, "middle"))
    cr = c.add("capacitor", "Cr1", 330, 60, 0, "C_r1", fmt_si(v["Cr"], "F", 5), lpos=(330, 24, "middle"))
    lm = c.add("inductor", "Lm", 430, 160, 90, "L_m", fmt_si(v["Lm"], "H", 4), lpos=(448, 156, "start"))
    cr2 = c.add("capacitor", "Cr2", 540, 60, 0, "C_r2′", fmt_si(v["Cr"], "F", 5), lpos=(540, 24, "middle"))
    lr2 = c.add("inductor", "Lr2", 650, 60, 0, "L_r2′", fmt_si(v["Lr"], "H", 4), lpos=(650, 24, "middle"))
    rac = c.add("resistor", "Rac", 760, 160, 90, "R_ac′", "n²·8R_L/π²", lpos=(778, 156, "start"))
    c.wire("w1", vs["a"], (130, 60), lr["a"])
    c.wire("w2", lr["b"], cr["a"])
    c.wire("w3", cr["b"], (430, 60), lm["a"])
    c.wire("w4", (430, 60), cr2["a"])
    c.wire("w5", cr2["b"], lr2["a"])
    c.wire("w6", lr2["b"], (760, 60), rac["a"])
    c.wire("w7", rac["b"], (760, 250), (430, 250), (130, 250), vs["b"])
    c.wire("w8", lm["b"], (430, 250))
    c.dot((430, 60), (430, 250))
    c.probe("pi1", "none", 170, 48, "right", "I_1")
    c.probe("pim", "none", 416, 110, "down", "I_m")
    c.probe("pi2", "none", 712, 48, "right", "I_2′")
    c.text(445, 284, f"1차 환산: C_r2′ = C_r2·n², L_r2′ = L_r2/n² — 실제 2차 L_r2 = {fmt_si(v['Lr'] / n**2, 'H', 6)}, C_r2 = {fmt_si(v['Cr'] * n**2, 'F', 6)}", "small")
    c.text(445, 302, "V_o′ = n·V_bat의 기본파, R_L = V_bat²/P", "small")
    c.mode("fha", "FHA 정상상태 (기본파 phasor)", ["V1", "Lr1", "Cr1", "Lm", "Cr2", "Lr2", "Rac", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"], "사인파 등가회로: 실제 정류·스위칭 파형이 아니다 (A 수준)")
    return c


def _cllc_is_textbook(v):
    tb = dict(Lr=40e-6, Cr=28.144773e-9, Lm=200e-6, f_min=120e3, f_max=210e3, P_rated=11000.0, Vlink_650=700.0, Vlink_800=800.0, Vlink_920=850.0)
    return all(abs(v[k] - x) <= 1e-9 * abs(x) for k, x in tb.items())


def run_obc_cllc(v: dict) -> Result:
    res = Result("EX01", "obc_cllc", "A (FHA 정적 해 + 합성 손실 지도)")
    if v["f_min"] >= v["f_max"]:
        res.verdict("OUT_OF_VALIDITY", "f_min ≥ f_max")
        return res
    n = v["n"]
    tb = _cllc_is_textbook(v)
    Pr = v["P_rated"]
    Pb = v["P_burst"]
    # --- grid availability (textbook E12 A)
    P_grid = ref.grid_available_power(v["V_LL"], v["I_line"], v["PF"], v["eta_sys"])
    P_grid_lab = 3.0 * (v["V_LL"] / math.sqrt(3.0)) * v["I_line"] * v["PF"] * v["eta_sys"]  # per-phase sum
    res.add_metric("P_grid", "입력 제한에서 가능한 배터리 전력", P_grid_lab, "W", ref=10698.8 if abs(v["V_LL"] - 400) + abs(v["I_line"] - 16) + abs(v["PF"] - 0.995) + abs(v["eta_sys"] - 0.97) < 1e-12 else None, ref_label="교재 ≈10.6988 kW", tol=2e-5, basis="3 × V_phase·I·PF·η (위상별 합)")
    res.add_check(check_close("입력 제한 전력: 위상별 합 vs √3·V_LL·I·PF·η", P_grid_lab, P_grid, 1e-12, "3·(V_LL/√3)·I·PF·η vs √3 V_LL I PF η (같은 양의 다른 표기)", False, "W"))
    short = Pr - P_grid_lab
    res.add_metric("P_short", f"{Pr / 1e3:g} kW 요구 대비 부족분", short, "W", ref=301.2 if tb else None, ref_label="교재 ≈301 W", tol=2e-3, basis="turns ratio로 해결되지 않는 입력 전류 제한")
    # --- seed diagnostics at the textbook high corner (kept whatever n is chosen)
    seed = cllc_eval(1.0, [920.0], [Pr], v)
    res.add_metric("M_req_seed", "seed n = 1: 920/850 V 요구 gain", float(seed["M"][0]), "", ref=1.082353 if tb else None, ref_label="교재 1.082353", tol=1e-6)
    res.add_metric("gmax_seed", "seed n = 1: inductive 영역 최대 FHA gain", float(seed["gmax"][0]), "", ref=1.016401 if tb else None, ref_label="교재 1.016401", tol=1e-6, basis=f"{v['f_min'] / 1e3:g}–{v['f_max'] / 1e3:g} kHz, {Pr / 1e3:g} kW")
    # --- chosen candidate: corners
    corners = [
        ("nominal 800/800 V", 800.0, Pr),
        ("고전압 920/850 V", 920.0, Pr),
        ("저전압 650/700 V", 650.0, Pr),
        ("고전압 경부하", 920.0, Pb),
        ("저전압 경부하", 650.0, Pb),
    ]
    cV = np.array([c[1] for c in corners])
    cP = np.array([c[2] for c in corners])
    e = cllc_eval(n, cV, cP, v, policy=v["branch"])
    rows = []
    bad = []
    unk = []
    for i, (nm, Vb, Pw) in enumerate(corners):
        cat, k, gm = _corner_row(e, i)
        if cat not in ("feasible",):
            (unk if cat == "unknown" else bad).append((nm, cat))
        rows.append([nm, Vb, float(e["Vl"][i]), Pw / 1e3, float(e["M"][i]), float(e["gmax"][i]), (float(e["f"][i]) / 1e3 if e["root"][i] else "해 없음"),
                     float(e["I1"][i]) if e["root"][i] else "—", float(e["B"][i]) if e["root"][i] else "—", float(e["Tj"][i]) if e["root"][i] else "—",
                     float(e["P_loss"][i]) if e["root"][i] else "—", CLLC_CATS[cat], f"{k} ({gm:+.1%})" if math.isfinite(gm) else "—"])
    res.tables.append(Table("t_corners", f"n = {n:.3g}: 요구 corner 판정 ({'오른쪽 분기' if v['branch'] == 'right' else '최소 전류 분기'})",
                            ["corner", "V_bat [V]", "V_link [V]", "P [kW]", "요구 gain", "최대 inductive gain", "f_op [kHz]", "I_1 RMS [A]", "B_pk [T]", "T_j [°C]", "손실 [W]", "판정", "가장 가까운 제약 (g)"], rows,
                            note="g = 정규화 제약값 (≤ 0 만족). gain은 FHA 정적 해이며 switching·ZVS·startup·reverse 검증이 아니다."))
    nom_loss = float(e["P_loss"][0]) if e["root"][0] else float("nan")
    res.add_metric("loss_nominal", f"n = {n:.3g}: nominal 손실 (합성 손실 지도)", nom_loss, "W", basis="1차 도통+turn-off, SR 도통, 동손, 코어, gate — postprocessed loss estimate")
    worst_i = max(range(len(corners)), key=lambda i: tightest({k: float(e["margins"][k][i]) for k in e["margins"]})[1] if not e["burst"][i] else -1)
    wk, wg = tightest({k: float(e["margins"][k][worst_i]) for k in e["margins"]})
    res.add_metric("dominant", "지배 제약 · 최악 corner", f"{wk} @ {corners[worst_i][0]}", "", basis=f"g = {wg:+.2%}")
    # textbook branch numbers for n = 0.93
    hi = cllc_eval(n, [920.0], [Pr], v)
    f_l, f_r = float(hi["fl"][0]), float(hi["fr"][0])
    is93 = tb and abs(n - 0.93) < 1e-12
    if math.isfinite(f_l) and math.isfinite(f_r):
        res.add_metric("f_left", "920/850 V: 왼쪽 분기 FHA 해", f_l, "Hz", ref=136099.47 if is93 else None, ref_label="교재 136099.47 Hz", tol=1e-7)
        res.add_metric("f_right", "920/850 V: 오른쪽 분기 FHA 해", f_r, "Hz", ref=147060.86 if is93 else None, ref_label="교재 147060.86 Hz", tol=1e-7)
        I1l = float(2 * math.sqrt(2) / math.pi * hi["Vl"][0] / abs(cllc_net(f_l, n, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[1]))
        I1r = float(2 * math.sqrt(2) / math.pi * hi["Vl"][0] / abs(cllc_net(f_r, n, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[1]))
        res.add_metric("I1_left", "왼쪽 분기 1차 RMS (FHA)", I1l, "A", ref=14.390 if is93 else None, ref_label="교재 14.390 A", tol=1e-4)
        res.add_metric("I1_right", "오른쪽 분기 1차 RMS (FHA)", I1r, "A", ref=14.765 if is93 else None, ref_label="교재 14.765 A", tol=1e-4)
        h = 10.0
        sl = [(abs(cllc_net(fx + h, n, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[0]) - abs(cllc_net(fx - h, n, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[0])) / (2 * h) * 1e3 for fx in (f_l, f_r)]
        res.add_metric("slope_left", "왼쪽 분기 gain 기울기 (±10 Hz 중앙차분)", sl[0], "1/kHz", ref=0.001871 if is93 else None, ref_label="교재 +0.001871/kHz", tol=1e-3, basis="양수: 주파수↑ → 출력↑ (제어 부호 반대)")
        res.add_metric("slope_right", "오른쪽 분기 gain 기울기", sl[1], "1/kHz", ref=-0.001803 if is93 else None, ref_label="교재 −0.001803/kHz", tol=1e-3)
    if is93:
        for (nm, Vb, fref) in (("f_650", 650.0, 164.390e3), ("f_800", 800.0, 162.811e3)):
            ee = cllc_eval(n, [Vb], [Pr], v)
            res.add_metric(nm, f"{Vb:g}/{float(ee['Vl'][0]):g} V 오른쪽 분기 해", float(ee["fr"][0]), "Hz", ref=fref, ref_label=f"교재 {fref / 1e3:.3f} kHz", tol=4e-6)
    res.add_metric("Lr2", f"실제 2차 L_r2 = L_r/n² (n = {n:.3g})", v["Lr"] / n**2, "H", ref=46.2481e-6 if is93 else None, ref_label="교재 46.2481 µH", tol=2e-6, basis="referred symmetry 유지")
    res.add_metric("Cr2", "실제 2차 C_r2 = C_r·n²", v["Cr"] * n**2, "F", ref=24.3424e-9 if is93 else None, ref_label="교재 24.3424 nF", tol=2e-6)
    # --- candidate family: nominal optimum vs required-corner feasible choice
    fam = np.round(np.arange(0.85, 1.0001, 0.01), 2)
    frows, fam_loss, fam_ok = [], [], []
    for nc in fam:
        ec = cllc_eval(float(nc), cV, cP, v, policy=v["branch"])
        cats = [_corner_row(ec, i)[0] for i in range(len(corners))]
        ok_all = all(c_ == "feasible" for c_ in cats)
        nl = float(ec["P_loss"][0]) if ec["root"][0] and cats[0] == "feasible" else float("nan")
        fam_loss.append(nl)
        fam_ok.append(ok_all)
        fails = [f"{corners[i][0]}: {CLLC_CATS[c_]}" for i, c_ in enumerate(cats) if c_ != "feasible"]
        mi = max(range(len(corners)), key=lambda i: tightest({k: float(ec["margins"][k][i]) for k in ec["margins"]})[1] if not ec["burst"][i] else -1)
        mk_, mg_ = tightest({k: float(ec["margins"][k][mi]) for k in ec["margins"]})
        frows.append([float(nc), nl, "예" if ok_all else "아니오", f"{mk_} @ {corners[mi][0]} ({mg_:+.1%})", "; ".join(fails) or "—"])
    fam_loss = np.array(fam_loss)
    i_nom = int(np.nanargmin(fam_loss)) if np.any(np.isfinite(fam_loss)) else None
    ok_idx = [i for i in range(len(fam)) if fam_ok[i] and math.isfinite(fam_loss[i])]
    i_rob = min(ok_idx, key=lambda i: fam_loss[i]) if ok_idx else None
    res.tables.append(Table("t_family", "후보군 n = 0.85–1.00: nominal 손실과 모든 요구 corner 만족 여부", ["n", "nominal 손실 [W]", "모든 corner 만족", "지배 제약 (g)", "실패 corner"], frows,
                            note="nominal 최적 = nominal 손실 최소. robust 선택 = 모든 요구 corner를 만족하는 후보 중 nominal 손실 최소. 비용 함수(cost)는 실제 자료가 없어 넣지 않았다."))
    res.add_metric("n_nominal_opt", "nominal 최적 후보 (손실 최소)", float(fam[i_nom]) if i_nom is not None else "없음", "", basis=f"{fam_loss[i_nom]:.1f} W" if i_nom is not None else "")
    res.add_metric("n_robust", "요구 corner 모두 만족하는 선택", float(fam[i_rob]) if i_rob is not None else "없음", "", basis=(f"{fam_loss[i_rob]:.1f} W (nominal 대비 +{fam_loss[i_rob] - fam_loss[i_nom]:.1f} W)" if (i_rob is not None and i_nom is not None) else "후보군 안에 없음"))
    # --- one unconfirmed datum that could change the claim (computed, not typed): a synthetic
    # tolerance screen (textbook ch.13: L/C +-5 %, Lm +-15 %) and the light-load regulation limit
    datum = "SR 정류·dead time을 포함한 실제 gain (FHA 오차) — time-domain(EX05) 확인 필요"
    tol_rows = []
    if i_rob is not None:
        nr = float(fam[i_rob])
        first = None
        for sLm in (0.85, 1.15):
            for sLr in (0.95, 1.05):
                for sCr in (0.95, 1.05):
                    et = cllc_eval(nr, cV, cP, dict(v, Lm=v["Lm"] * sLm, Lr=v["Lr"] * sLr, Cr=v["Cr"] * sCr), policy=v["branch"])
                    fails = [(corners[i][0], CLLC_CATS[_corner_row(et, i)[0]]) for i in range(len(corners)) if _corner_row(et, i)[0] != "feasible"]
                    combo = f"L_m {sLm - 1:+.0%}, L_r {sLr - 1:+.0%}, C_r {sCr - 1:+.0%}"
                    tol_rows.append([combo, "; ".join(f"{a}: {b}" for a, b in fails) or "모두 만족"])
                    if fails and first is None:
                        first = (combo, fails)
        eb = cllc_eval(nr, cV, np.where(cP == Pb, Pb / 2, cP), dict(v, P_burst=Pb / 2), policy=v["branch"])
        b_fail = [corners[i][0] for i in range(len(corners)) if _corner_row(eb, i)[0] != "feasible"]
        n_fail = sum(1 for r in tol_rows if r[1] != "모두 만족")
        if first is not None:
            datum = f"실제 L_m·L_r·C_r 공차 — 8개 공차 corner 중 {n_fail}개에서 n = {nr:.2f}가 실패 (예: {first[0]} → {first[1][0][0]}: {first[1][0][1]})"
        elif b_fail:
            datum = f"경부하 연속 regulation 하한 — burst 허용 전력이 {Pb / 2e3:.2g} kW로 낮아지면 n = {nr:.2f}가 {', '.join(b_fail)}에서 실패"
        res.tables.append(Table("t_tol", f"robust 후보 n = {nr:.2f}의 합성 공차 screen (양쪽 tank 같은 비율)", ["공차 조합", "실패 corner"], tol_rows,
                                note="공차 ±는 교재의 합성 screen 시작값이다. 실제 부품 공차·상관관계 자료로 바꿔야 한다(MISSING_INPUT)."))
    res.add_metric("unconfirmed", "주장을 바꿀 수 있는 미확인 데이터 하나", datum, "")
    # --- operating-region map for the chosen n
    Vg = np.arange(650.0, 920.0 + 1e-9, 10.0)
    Pg = np.arange(500.0, Pr + 1e-9, 500.0)
    VV, PP = np.meshgrid(Vg, Pg, indexing="ij")
    em = cllc_eval(n, VV.ravel(), PP.ravel(), v, policy=v["branch"])
    cats = classify(em["margins"], em["unknown"], {"burst": em["burst"]})
    frac = float(np.mean(cats == "feasible"))
    res.add_metric("feasible_frac", f"n = {n:.3g}: 운전영역 격자 중 모든 제약 만족 비율", frac, "", basis=f"{VV.size}점 (V_bat 10 V × P 0.5 kW); burst {float(np.mean(cats == 'burst')):.0%}")
    res.add_metric("unknown_frac", "손실 지도 외삽(UNKNOWN) 비율", float(np.mean(cats == "unknown")), "", basis=f"turn-off 지도 {cllc_device(v).box_text()}")
    mk = [{"x": c[1], "y": c[2] / 1e3, "short": sh, "label": c[0]} for c, sh in zip(corners, ("N", "H", "L", "Hℓ", "Lℓ"))]
    add_region_plot(res, "p_map", f"n = {n:.3g}: 운전영역과 active limit (V_bat × P, 점선: 입력 한계·P_burst)", VV.ravel(), PP.ravel() / 1e3, cats, CLLC_CATS, CLLC_COL, x_label="배터리 전압 V_bat", x_unit="V", y_label="배터리 전력 P", y_unit="kW",
                    markers=mk, hlines=[{"y": P_grid_lab / 1e3, "label": ""}, {"y": Pb / 1e3, "label": ""}],
                    proved="각 운전점에서 gain·전류·자속·ZVS screen·T_j를 동시에 평가해 어느 제약이 먼저 걸리는지를 영역으로 보였다. seed n = 1은 고전압 쪽에서 gain이 부족하다. FHA gain은 체인(ABCD) 행렬 check, 운전점은 입력 phasor 전력 = 부하 전력 check가 뒷받침한다.",
                    not_yet="FHA 정적 해와 합성 손실 지도다. 실제 정류 파형·dead time·SR timing·startup·reverse는 time-domain(EX05)에서 확인해야 한다.")
    notes = []
    for i in range(VV.size):
        if em["root"][i]:
            notes.append(f"f {em['f'][i] / 1e3:.1f} kHz, I₁ {em['I1'][i]:.1f} A, B {em['B'][i]:.3f} T, T_j {em['Tj'][i]:.0f} °C")
        else:
            notes.append(f"요구 gain {em['M'][i]:.4f}, 최대 {em['gmax'][i]:.4f}, f_max에서 {em['g_fmax'][i]:.4f}")
    add_loss_map(res, "p_lossmap", f"n = {n:.3g}: 가능한 영역의 합성 손실 (빗금 = 제약 위반, 이유는 셀 위에)", Vg, Pg / 1e3, cats, em["P_loss"], CLLC_CATS, z_label="손실 (postprocessed)", z_unit="W",
                 x_label="배터리 전압 V_bat", x_unit="V", y_label="배터리 전력 P", y_unit="kW", cell_notes=notes, markers=mk,
                 proved="모든 제약을 만족하는 운전점에서만 손실을 칠했다. 손실이 낮은 곳과 제약이 걸리는 곳은 다른 문제다. 소자 T_j는 선형 PTC 닫힌 해와 고정점 반복이 일치하는 check로 계산했다.",
                 not_yet="손실은 합성 손실 지도 후처리(postprocessed)이며 실제 소자·자성체 자료가 아니다.")
    xs = fam.tolist()
    res.add_series("fam_loss", "nominal 손실", "W", xs, [x if math.isfinite(x) else None for x in fam_loss])
    res.add_series("fam_ok", "모든 corner 만족", "W", [x for x, o in zip(xs, fam_ok) if o], [fam_loss[i] for i, o in enumerate(fam_ok) if o], style="points", color=C_FEAS)
    bad_i = [i for i, o in enumerate(fam_ok) if not o and math.isfinite(fam_loss[i])]
    res.add_series("fam_bad", "corner 실패", "W", [xs[i] for i in bad_i], [float(fam_loss[i]) for i in bad_i], style="points", color="var(--c5)")
    fmk = []
    if i_nom is not None:
        fmk.append({"x": float(fam[i_nom]), "y": float(fam_loss[i_nom]), "label": f"nominal 최적 n = {fam[i_nom]:.2f}"})
    if i_rob is not None:
        fmk.append({"x": float(fam[i_rob]), "y": float(fam_loss[i_rob]), "label": f"robust n = {fam[i_rob]:.2f}"})
    res.add_plot("p_family", "nominal 최적 vs 요구 corner를 모두 만족하는 선택", ["fam_loss", "fam_ok", "fam_bad"], x_label="turns ratio n", x_unit="", y_label="nominal 손실", y_unit="W", kind="xy", level="A", markers=fmk,
                 proved="nominal 손실이 가장 작은 후보와 모든 요구 corner를 만족하는 후보가 다르다. 결정은 corner feasibility가 먼저다.",
                 not_yet="손실 차이는 합성 손실 지도 기준이며 불확도 평가(mission_ranking)를 거치지 않았다.")
    fq = np.linspace(v["f_min"], v["f_max"], 400)
    Hs = np.abs(cllc_net(fq, 1.0, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[0])
    res.add_series("g_seed", "seed n = 1, 920 V·11 kW", "", (fq / 1e3).tolist(), Hs.tolist())
    keys = ["g_seed"]
    hl = [{"y": float(seed["M"][0]), "label": f"요구 {float(seed['M'][0]):.4f} (n = 1)"}]
    if abs(n - 1.0) > 1e-12:
        Hn = np.abs(cllc_net(fq, n, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[0])
        res.add_series("g_n", f"n = {n:.3g}, 920 V·11 kW", "", (fq / 1e3).tolist(), Hn.tolist())
        keys.append("g_n")
        hl.append({"y": float(hi["M"][0]), "label": f"요구 {float(hi['M'][0]):.4f} (n = {n:.3g})"})
    gmk = []
    if math.isfinite(f_l):
        gmk.append({"x": f_l / 1e3, "y": float(hi["M"][0]), "label": f"{f_l / 1e3:.3f} kHz"})
    if math.isfinite(f_r):
        gmk.append({"x": f_r / 1e3, "y": float(hi["M"][0]), "label": f"{f_r / 1e3:.3f} kHz"})
    res.add_plot("p_gain", "고전압 corner의 FHA gain: seed 실패와 수정 후보의 두 해", keys, x_label="스위칭 주파수", x_unit="kHz", y_label="|H| = nV_o/V_i", y_unit="", kind="xy", level="A", hlines=hl, markers=gmk,
                 proved="seed의 최대 gain이 요구보다 작아 해가 없다(1.016401 < 1.082353). n을 낮추면 요구 gain이 내려가 두 개의 FHA 해가 생긴다(n = 0.93: 136.099/147.061 kHz, 오른쪽 해는 격자 이분법과 brentq check가 일치).",
                 not_yet="두 해의 기울기 부호가 반대라 제어 분기 선택·전류·ZVS를 time-domain으로 비교해야 한다. FHA 해는 후보일 뿐이다.")
    res.circuit = {"diagram": cllc_circuit(v, n).to_json(), "intervals": [], "plot_group": ""}
    # --- independent checks
    i_chk = [i for i in range(len(corners)) if e["root"][i]]
    err_abcd = 0.0
    for i in i_chk:
        Ha, Za = ref.cllc_gain_abcd(float(e["f"][i]), n, float(cV[i]), float(cP[i]), v["Lr"], v["Cr"], v["Lm"])
        Hl, Zl, _ = cllc_net(float(e["f"][i]), n, float(cV[i]), float(cP[i]), v["Lr"], v["Cr"], v["Lm"])
        err_abcd = max(err_abcd, abs(Ha - Hl) / abs(Ha), abs(Za - Zl) / abs(Za))
    res.add_check(Check("FHA gain·입력 임피던스: 분압식 vs 체인(ABCD) 행렬", "PASS" if err_abcd < 1e-12 else "FAIL", err_abcd, "rel", 1e-12, path="Z_p 분압 대수 vs 직렬·병렬·직렬 2-port 행렬 곱 (reference)", independent=True, detail=f"해가 있는 corner {len(i_chk)}곳"))
    fr0 = 1.0 / (2 * math.pi * math.sqrt(v["Lr"] * v["Cr"]))
    e_fr = max(abs(abs(cllc_net(fr0, n, 800.0, Px, v["Lr"], v["Cr"], v["Lm"])[0]) - 1.0) for Px in (1000.0, 5500.0, 11000.0))
    res.add_check(Check("극한: 대칭 tank의 f_r에서 |H| = 1 (부하 무관)", "PASS" if e_fr < 1e-12 else "FAIL", e_fr, "", 1e-12, path=f"f_r = 1/(2π√(L_r C_r)) = {fr0 / 1e3:.4f} kHz, 부하 1/5.5/11 kW", independent=True))
    e_pw = 0.0
    for i in i_chk:
        Hh, Zi, Rac = cllc_net(float(e["f"][i]), n, float(cV[i]), float(cP[i]), v["Lr"], v["Cr"], v["Lm"])
        V1 = float(e["V1"][i])
        Pin = (V1 * np.conj(V1 / Zi)).real
        Pout = (abs(Hh) * V1) ** 2 / Rac
        e_pw = max(e_pw, abs(Pin - Pout) / Pout, abs(Pout - float(cP[i])) / float(cP[i]))
    res.add_check(Check("전력 항등식: Re(V_1·I_1*) = |V_o′|²/R_ac′ = P (무손실 tank)", "PASS" if e_pw < 1e-9 else "FAIL", e_pw, "rel", 1e-9, path="입력 phasor 전력 vs 부하 저항 전력 vs 지정 전력 (에너지 보존)", independent=True))
    if math.isfinite(f_r):
        Mh = float(hi["M"][0])

        def gfun(fx):
            return abs(cllc_net(fx, n, 920.0, Pr, v["Lr"], v["Cr"], v["Lm"])[0]) - Mh

        a_, b_ = max(f_r - 200.0, v["f_min"]), min(f_r + 200.0, v["f_max"])
        if gfun(a_) * gfun(b_) < 0:
            fb = brentq(gfun, a_, b_, xtol=1e-9, rtol=1e-15)
            res.add_check(check_close("오른쪽 분기 해: 격자+벡터 이분법 vs scalar brentq", f_r, fb, 1e-9, "721점 격자 교차 + 48회 이분법 vs brentq(±200 Hz 구간)", True, "Hz"))
    Ti = float(e["Tj"][i_chk[-1]]) if i_chk else float("nan")
    if i_chk:
        j = i_chk[-1]
        a1 = float(e["I1"][j]) ** 2 / 2 * v["Rds"]
        psw = float(e["f"][j]) * float(cllc_device(v).energy(float(e["Vl"][j]), max(float(e["ioff"][j]), 0.0)))
        T = v["T_cool"]
        for _ in range(200):
            T = v["T_cool"] + v["Rth"] * (a1 * (1 + v["tc"] * (T - 25)) + psw)
        res.add_check(check_close("T_j: 선형 PTC 닫힌 해 vs 고정점 반복", Ti, T, 1e-10, "T = (T_c + R_th(a(1−25α)+P_sw))/(1 − R_th a α) vs 반복 대입", True, "°C"))
    # --- verdicts
    if short > 0:
        res.verdict("CUSTOMER_DECISION_REQUIRED", f"입력 {v['V_LL']:g} V·{v['I_line']:g} A·PF {v['PF']:g}·η {v['eta_sys']:g} 가정에서 배터리 전력은 {P_grid_lab / 1e3:.4f} kW까지 — {Pr / 1e3:g} kW보다 {short:.0f} W 부족: 입력전류 요구/derating 합의 필요 (turns ratio로 해결 불가)")
    if bad:
        res.verdict("FAIL_CONSTRAINT", f"n = {n:.3g}: " + "; ".join(f"{nm} → {CLLC_CATS[c_]}" for nm, c_ in bad))
    elif unk:
        res.verdict("OUT_OF_VALIDITY", f"n = {n:.3g}: " + "; ".join(f"{nm} → 손실 지도 외삽" for nm, _ in unk) + " — 그 corner의 T_j·손실은 UNKNOWN")
    else:
        res.verdict("CANDIDATE_FHA_ONLY", f"n = {n:.3g}: 모든 요구 corner에 FHA 정적 해 존재 — switching·ZVS·SR·startup·reverse·공차 검증 전 후보")
    if i_rob is None:
        res.verdict("NO_SOLUTION", f"후보군 n 0.85–1.00, {v['f_min'] / 1e3:g}–{v['f_max'] / 1e3:g} kHz 안에 모든 요구 corner를 만족하는 후보가 없다 — 주파수 범위·burst 하한·link 전압 schedule을 다시 협의")
    res.assumptions += [
        "FHA: 기본파 phasor, full-bridge 기본파 0.9·V, 정류 부하 R_ac′ = n²·8R_L/π²; referred symmetry(L_r2′ = L_r, C_r2′ = C_r)",
        "link 전압 schedule: 배터리 650/800/920 V ↔ link 700/800/850 V 사이 선형 보간",
        f"오른쪽 분기(gain 기울기 음) 기본 선택; P < P_burst({Pb / 1e3:g} kW)은 burst 운전으로 가정해 FHA 판정에서 제외",
        "turn-off 전류 = √2·I_1·sin θ_Zin (FHA 추정), ZVS screen: i_off·t_dead ≥ 2Q_oss",
        "자속: 구형파 n·V_bat/(4 f N_p A_e) (보수적); 손실은 합성 손실 지도 후처리(postprocessed)",
    ]
    res.not_valid_for += ["실제 ZVS·SR timing·dead-time 파형 (NOT_EVALUABLE: EX05 time-domain)", "startup·reverse(역방향은 forward gain의 역수가 아님)", "실제 소자·코어 재료 손실 (MISSING_INPUT)", "경부하 burst 운전"]
    res.interpretation = (
        f"seed n = 1은 nominal에서 공진점(gain 1)에 있어 손실이 가장 작지만 920/850 V에서 필요한 gain {float(seed['M'][0]):.4f}를 낼 수 없다(최대 {float(seed['gmax'][0]):.4f}). "
        "n을 낮추면 고전압 corner에 FHA 해가 생기지만 nominal은 공진점 위로 밀려 손실이 늘고, 너무 낮추면 저전압 경부하에서 f_max로도 gain을 낮추지 못하거나 전류가 커진다. "
        "그래서 nominal 최적과 corner를 모두 만족하는 선택은 다르다. 또 입력 전류 제한은 turns ratio와 무관한 요구 조건 문제다."
    )
    return res


# ======================================================================================
# Experiment 3: capstone B - 900 V <-> LV DAB, 2 x 1.5 kW (SPS operating window)
# ======================================================================================

DAB_CATS = {
    "feasible": "모든 제약 만족 (ZVS screen 통과)",
    "screen": "제약 만족, ZVS screen 실패 → hard switching 손실 포함",
    "power": "전력 한계 (φ > φ_max 또는 P > P_max)",
    "rms": "RMS 전류 한계",
    "tj_hv": "HV 소자 T_j 한계",
    "tj_lv": "LV 소자 T_j 한계",
    "unknown": "손실 지도 외삽 (UNKNOWN)",
}
DAB_COL = {"feasible": C_FEAS, "screen": C_SCREEN, "power": "var(--c5)", "rms": "var(--c2)", "tj_hv": "var(--c4)", "tj_lv": "var(--c1)", "unknown": C_UNK}


def dab_maps(v):
    """Synthetic energy maps per switching event: ZVS turn-off only, or a hard event with a Coss floor."""
    hv_off = LossMap("HV E_off", E0=25e-6, a=1.2, b=1.0, I0=10.0, V0=800.0, V_lo=400.0, V_hi=950.0, I_lo=0.2, I_hi=20.0)
    hv_hard = LossMap("HV hard", E0=40e-6, a=1.1, b=1.3, Eoss=v["Eoss_hv"], I0=10.0, V0=800.0, V_lo=400.0, V_hi=950.0, I_lo=0.0, I_hi=20.0)
    lv_off = LossMap("LV E_off", E0=8e-6, a=1.2, b=1.0, I0=50.0, V0=48.0, V_lo=30.0, V_hi=60.0, I_lo=2.0, I_hi=150.0)
    lv_hard = LossMap("LV hard", E0=15e-6, a=1.1, b=1.3, Eoss=3e-6, I0=50.0, V0=48.0, V_lo=30.0, V_hi=60.0, I_lo=0.0, I_hi=150.0)
    return hv_off, hv_hard, lv_off, lv_hard


def dab_eval(L, VH, VL, P, v):
    """Vectorised SPS operating point per module: segment waveform, ZVS screen, losses, constraints."""
    VH, VL, P = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float)) for x in (VH, VL, P)))
    n, fs = v["n_tr"], v["fs"]
    w = 2 * math.pi * fs
    V2 = n * VL
    y = P * w * L / (VH * V2)  # = phi (1 - phi/pi)
    phimax = math.radians(v["phi_max_deg"])
    ok_p = y <= math.pi / 4
    phi = np.where(ok_p, math.pi / 2 * (1 - np.sqrt(np.clip(1 - 4 * y / math.pi, 0.0, None))), np.nan)
    T = 1 / fs
    t1 = phi / w
    t2 = T / 2 - t1
    s1 = (VH + V2) / L
    s2 = (VH - V2) / L
    i0 = -(s1 * t1 + s2 * t2) / 2
    ip = i0 + s1 * t1
    iT = ip + s2 * t2  # = -i0 (antisymmetric half-wave solution)
    ms1 = (i0 * i0 + i0 * ip + ip * ip) / 3
    ms2 = (ip * ip + ip * iT + iT * iT) / 3
    irms = np.sqrt((t1 * ms1 + t2 * ms2) / (T / 2))
    ipk = np.maximum(np.abs(i0), np.abs(ip))
    P2 = 2 / T * (-V2 * t1 * (i0 + ip) / 2 + V2 * t2 * (ip + iT) / 2)  # secondary-port power, segment means
    # ZVS screen: HV bridge switches at t = 0 and needs i(0) < -I_zvs,hv; LV bridge at t = phi/w needs i > +I_zvs,lv/n
    Iz_hv = 2 * v["Qoss_hv"] / v["t_dead"]
    Iz_lv = 2 * v["Qoss_lv"] / v["t_dead"] / n
    zvs_hv = i0 <= -Iz_hv
    zvs_lv = ip >= Iz_lv
    hv_off, hv_hard, lv_off, lv_hard = dab_maps(v)
    i_hv = np.abs(i0)
    i_lv = np.abs(ip) * n
    # one event per device per period: a ZVS commutation costs the turn-off energy; with the current in the
    # wrong direction the outgoing device hands its current to its diode (no turn-off loss) and the incoming
    # device turns on hard (Coss floor + overlap), so a hard event uses the hard map alone
    E_hv = np.where(zvs_hv, hv_off.energy(VH, i_hv), hv_hard.energy(VH, i_hv))
    E_lv = np.where(zvs_lv, lv_off.energy(VL, i_lv), lv_hard.energy(VL, i_lv))
    unk = ok_p & ~(np.where(zvs_hv, hv_off.inside(VH, i_hv), hv_hard.inside(VH, i_hv)) & np.where(zvs_lv, lv_off.inside(VL, i_lv), lv_hard.inside(VL, i_lv)))
    a_hv = irms**2 / 2 * v["R_hv"]
    a_lv = (n * irms) ** 2 / 2 * v["R_lv"]
    T_hv = _linear_tj(v["T_cool"], v["Rth_hv"], a_hv, v["tc"], fs * E_hv)
    T_lv = _linear_tj(v["T_cool"], v["Rth_lv"], a_lv, v["tc"], fs * E_lv)
    parts = dict(
        hv_cond=4 * a_hv * (1 + v["tc"] * (T_hv - 25)),
        hv_sw=4 * fs * E_hv,
        lv_cond=4 * a_lv * (1 + v["tc"] * (T_lv - 25)),
        lv_sw=4 * fs * E_lv,
        tr_cu=v["R_tr"] * irms**2,
    )
    P_loss = sum(parts.values())
    nan = np.full(P.shape, np.nan)
    ymax = phimax * (1 - phimax / math.pi)
    margins = {
        "power": y / ymax - 1.0,
        "rms": np.where(ok_p, irms / v["Irms_max"] - 1.0, nan),
        "tj_hv": np.where(ok_p, (T_hv - v["Tj_max"]) / (v["Tj_max"] - v["T_cool"]), nan),
        "tj_lv": np.where(ok_p, (T_lv - v["Tj_max"]) / (v["Tj_max"] - v["T_cool"]), nan),
    }
    return dict(VH=VH, VL=VL, P=P, V2=V2, phi=phi, i0=i0, ip=ip, irms=irms, ipk=ipk, P2=P2, zvs_hv=zvs_hv, zvs_lv=zvs_lv, T_hv=T_hv, T_lv=T_lv, parts=parts, P_loss=P_loss, margins=margins, unknown=unk, ok_p=ok_p, Iz_hv=Iz_hv, Iz_lv=Iz_lv)


def _dab_cat(e, i):
    mg = {k: np.array([e["margins"][k][i]]) for k in e["margins"]}
    cat = str(classify(mg, np.array([bool(e["unknown"][i])]), loss_keys=("tj_hv", "tj_lv"))[0])
    if cat == "feasible" and not (e["zvs_hv"][i] and e["zvs_lv"][i]):
        cat = "screen"
    return cat


def dab_circuit(v: dict, L: float) -> Circuit:
    c = Circuit("ex01_dab", 760, 250, title=f"DAB 1모듈 (SPS, L = {fmt_si(L, 'H', 3)} 1차 환산, n = {v['n_tr']:.4g}:1)")
    c.add("block", "HV", 110, 125, 0, "HV full bridge", "v_1 = ±V_H", w=120, h=70)
    ll = c.add("inductor", "L", 270, 70, 0, "L (1차 환산)", fmt_si(L, "H", 3), lpos=(270, 36, "middle"))
    tx = c.add("transformer", "TX", 390, 125, 0, f"n = {v['n_tr']:.4g} : 1", "")
    c.add("block", "LV", 590, 125, 0, "LV full bridge", "v_2 = ±V_L", w=120, h=70)
    c.wire("w1", (170, 105), (200, 105), (200, 70), ll["a"])
    c.wire("w2", ll["b"], (368, 70), tx["p1"])
    c.wire("w3", (170, 145), (200, 145), (200, 180), (368, 180), tx["p2"])
    c.wire("w4", tx["s1"], (412, 70), (500, 70), (500, 105), (530, 105))
    c.wire("w5", tx["s2"], (412, 180), (500, 180), (500, 145), (530, 145))
    c.probe("piL", "none", 330, 58, "right", "i_L")
    c.text(270, 225, "1차 bridge가 φ만큼 앞서면 HV → LV 전력 (φ = 0이어도 전압 불일치면 전류가 흐른다)", "small")
    c.mode("sps", "SPS 정상상태 (반주기 반대칭 해)", ["HV", "L", "TX", "LV", "w1", "w2", "w3", "w4", "w5"], "ideal switch 구간 선형 파형 (A 수준): ZVS는 전류 부호·크기 screen만")
    return c


def _dab_is_textbook(v):
    return abs(v["n_tr"] - 50 / 3) < 1e-9 and abs(v["fs"] - 100e3) < 1e-6


def run_dab_lv(v: dict) -> Result:
    res = Result("EX01", "dab_lv", "A (SPS 구간 선형 파형 + 합성 손실 지도)")
    L, Pm, nmod = v["L"], v["P_mod"], int(v["N_mod"])
    tb = _dab_is_textbook(v)
    tbL = tb and abs(L - 200e-6) < 1e-12
    corners = [
        ("nominal 800 V / 48 V", v["VH_nom"], v["VL_nom"], Pm),
        ("저전압 LV: 900 V / 36 V", 900.0, 36.0, Pm),
        ("900 V / 36 V 무부하 (φ = 0)", 900.0, 36.0, 0.0),
        ("550 V / 36 V", 550.0, 36.0, Pm),
        ("550 V / 54 V", 550.0, 54.0, Pm),
        ("900 V / 54 V", 900.0, 54.0, Pm),
    ]
    cVH = np.array([c[1] for c in corners])
    cVL = np.array([c[2] for c in corners])
    cP = np.array([c[3] for c in corners])
    e = dab_eval(L, cVH, cVL, cP, v)
    # --- textbook numbers (per module; the total of N modules is never mixed into one module's model)
    nom_tb = tb and abs(v["VH_nom"] - 800) < 1e-9 and abs(v["VL_nom"] - 48) < 1e-9 and abs(Pm - 1500) < 1e-9
    res.add_metric("phi_nom", "nominal φ (1모듈 1.5 kW)", float(e["phi"][0]), "rad", ref=0.328973 if (tbL and nom_tb) else None, ref_label="교재 0.328973 rad", tol=2e-6, basis=f"d = φ/π = {float(e['phi'][0]) / math.pi:.4f}")
    res.add_metric("irms_nom", "nominal I_rms (1차)", float(e["irms"][0]), "A", ref=2.01988 if (tbL and nom_tb) else None, ref_label="교재 2.01988 A", tol=2e-6)
    res.add_metric("ipk_nom", "nominal I_pk", float(e["ipk"][0]), "A", ref=2.09431 if (tbL and nom_tb) else None, ref_label="교재 2.09431 A", tol=2e-6)
    res.add_metric("irms_lowbat", "900 V / nV_L 600 V, 1.5 kW: I_rms", float(e["irms"][1]), "A", ref=3.11356 if (tbL and abs(Pm - 1500) < 1e-9) else None, ref_label="교재 E12 3.1136 A", tol=2e-5)
    res.add_metric("ipk_lowbat", "900 V / 600 V, 1.5 kW: I_pk", float(e["ipk"][1]), "A", ref=5.65983 if (tbL and abs(Pm - 1500) < 1e-9) else None, ref_label="교재 E12 5.6598 A", tol=2e-5)
    res.add_metric("irms_phi0", "900 V / 600 V, φ = 0: I_rms (전력 0)", float(e["irms"][2]), "A", ref=2.16506 if tbL else None, ref_label="교재 2.16506 A", tol=2e-6, basis=f"전달 전력 {float(e['P2'][2]):.3g} W — 전압 불일치 순환전류")
    pmax = float(cVH[0] * e["V2"][0] / (8 * v["fs"] * L))
    res.add_metric("Pmax_nom", "SPS 최대 전력 (nominal, φ = π/2)", pmax, "W", ref=4000.0 if (tbL and nom_tb) else None, ref_label="V_1V_2′/(8 f L) = 4 kW", tol=1e-9, basis="1모듈")
    res.add_metric("P_total", f"합계 {nmod}모듈", nmod * Pm, "W", basis=f"{nmod} × {Pm / 1e3:g} kW — 1모듈 모델에 합계 전력을 넣지 않는다")
    # --- corner table
    rows, bad, unk, screen = [], [], [], []
    for i, (nm, VH, VL, Pw) in enumerate(corners):
        cat = _dab_cat(e, i)
        k, gm = tightest({kk: float(e["margins"][kk][i]) for kk in e["margins"]})
        if cat in ("feasible", "screen"):
            if cat == "screen":
                screen.append(nm)
        elif cat == "unknown":
            unk.append(nm)
        else:
            bad.append((nm, cat))
        ok = bool(e["ok_p"][i])
        rows.append([nm, VH, VL, Pw / 1e3, math.degrees(float(e["phi"][i])) if ok else "해 없음", float(e["irms"][i]) if ok else "—", float(e["ipk"][i]) if ok else "—",
                     ("ZVS" if e["zvs_hv"][i] else "hard") + " / " + ("ZVS" if e["zvs_lv"][i] else "hard") if ok else "—",
                     float(e["T_hv"][i]) if ok else "—", float(e["T_lv"][i]) if ok else "—", float(e["P_loss"][i]) if ok else "—", DAB_CATS[cat], f"{k} ({gm:+.1%})" if math.isfinite(gm) else "—"])
    res.tables.append(Table("t_corners", f"L = {fmt_si(L, 'H', 3)}: 요구 corner 판정 (1모듈)", ["corner", "V_H [V]", "V_L [V]", "P [kW]", "φ [°]", "I_rms [A]", "I_pk [A]", "ZVS screen HV / LV", "T_j HV [°C]", "T_j LV [°C]", "손실 [W]", "판정", "가장 가까운 제약 (g)"], rows,
                            note="ZVS screen: HV는 i(0) ≤ −2Q_oss/t_dead, LV는 i(φ) ≥ 2Q_oss,LV/(n·t_dead). screen 실패 시 hard-switching 손실 지도로 계산 — ideal switch 모델의 ZVS 판정은 NOT_EVALUABLE."))
    nom_loss = float(e["P_loss"][0]) if e["ok_p"][0] else float("nan")
    res.add_metric("loss_nominal", f"L = {fmt_si(L, 'H', 3)}: nominal 손실 (1모듈)", nom_loss, "W", basis=f"합계 {nmod}모듈 {nmod * nom_loss:.1f} W — postprocessed loss estimate")
    # --- candidate family
    fam = np.array([100e-6, 125e-6, 150e-6, 175e-6, 200e-6, 250e-6, 300e-6])
    frows, fam_loss, fam_ok, fam_tight = [], [], [], []
    for Lc in fam:
        ec = dab_eval(float(Lc), cVH, cVL, cP, v)
        cats = [_dab_cat(ec, i) for i in range(len(corners))]
        ok_all = all(c_ in ("feasible", "screen") for c_ in cats)
        nl = float(ec["P_loss"][0]) if cats[0] in ("feasible", "screen") else float("nan")
        fam_loss.append(nl)
        fam_ok.append(ok_all)
        mi = max(range(len(corners)), key=lambda i: tightest({k: float(ec["margins"][k][i]) for k in ec["margins"]})[1])
        mk_, mg_ = tightest({k: float(ec["margins"][k][mi]) for k in ec["margins"]})
        fam_tight.append((mk_, mg_, corners[mi][0]))
        fails = [f"{corners[i][0]}: {DAB_CATS[c_]}" for i, c_ in enumerate(cats) if c_ not in ("feasible", "screen")]
        hard = sum(1 for c_ in cats if c_ == "screen")
        frows.append([Lc * 1e6, nl, float(ec["irms"][0]), max(float(x) for x in ec["irms"][np.isfinite(ec["irms"])]), "예" if ok_all else "아니오", hard, f"{mk_} @ {corners[mi][0]} ({mg_:+.1%})", "; ".join(fails) or "—"])
    fam_loss = np.array(fam_loss)
    i_nom = int(np.nanargmin(fam_loss)) if np.any(np.isfinite(fam_loss)) else None
    ok_idx = [i for i in range(len(fam)) if fam_ok[i] and math.isfinite(fam_loss[i])]
    i_rob = min(ok_idx, key=lambda i: fam_loss[i]) if ok_idx else None
    res.tables.append(Table("t_family", "후보군 L: nominal 손실과 요구 corner 만족 여부 (1모듈)", ["L [µH]", "nominal 손실 [W]", "nominal I_rms [A]", "corner 최대 I_rms [A]", "모든 corner 만족", "hard-switching corner 수", "지배 제약 (g)", "실패 corner"], frows,
                            note="L이 작으면 같은 전력에 φ가 작아 nominal RMS가 줄지만, 전압 불일치 corner의 순환전류는 1/L로 커진다. L이 크면 저전압 corner의 최대 전력이 모자란다."))
    res.add_metric("L_nominal_opt", "nominal 최적 L (손실 최소)", float(fam[i_nom]) if i_nom is not None else "없음", "H" if i_nom is not None else "", basis=f"{fam_loss[i_nom]:.2f} W" if i_nom is not None else "")
    res.add_metric("L_robust", "요구 corner 모두 만족하는 선택", float(fam[i_rob]) if i_rob is not None else "없음", "H" if i_rob is not None else "", basis=(f"{fam_loss[i_rob]:.2f} W (nominal 최적 대비 +{fam_loss[i_rob] - fam_loss[i_nom]:.2f} W)" if (i_rob is not None and i_nom is not None) else "후보군 안에 없음"))
    wk, wg, wc = fam_tight[i_rob] if i_rob is not None else ("—", float("nan"), "—")
    datum_map = {
        "power": f"φ_max 제한({v['phi_max_deg']:g}°)과 실제 L 공차 — {wc}의 전력 여유 {-wg:.1%}",
        "rms": f"transformer·소자 RMS 허용치(가정 {v['Irms_max']:g} A) — {wc}의 여유 {-wg:.1%}; 실제 열 자료로 바뀌면 선택이 바뀐다",
        "tj_hv": f"HV 소자 R_th·손실 지도 — {wc}의 T_j 여유 {-wg:.1%}",
        "tj_lv": f"LV 소자 R_th·손실 지도 — {wc}의 T_j 여유 {-wg:.1%}",
    }
    ci = max(range(len(corners)), key=lambda i: tightest({kk: float(e["margins"][kk][i]) for kk in e["margins"]})[1])
    ck, cg = tightest({kk: float(e["margins"][kk][ci]) for kk in e["margins"]})
    res.add_metric("dominant", f"L = {fmt_si(L, 'H', 3)}: 지배 제약 · 최악 corner", f"{ck} @ {corners[ci][0]}", "", basis=f"g = {cg:+.1%}")
    res.add_metric("dominant_robust", "robust 선택의 지배 제약 · 최악 corner", f"{wk} @ {wc}" if i_rob is not None else "—", "", basis=f"g = {wg:+.1%}" if math.isfinite(wg) else "")
    res.add_metric("unconfirmed", "주장을 바꿀 수 있는 미확인 데이터 하나", datum_map.get(wk, "LV 소자 Q_oss(V)·dead time (ZVS 경계)") if i_rob is not None else "요구 전압 범위·φ_max·RMS 허용치 자체", "")
    # --- map at V_H = V_H_map
    VLg = np.arange(36.0, 54.0 + 1e-9, 1.0)
    Pg = np.arange(0.05, 1.0 + 1e-9, 0.05) * Pm
    VV, PP = np.meshgrid(VLg, Pg, indexing="ij")
    em = dab_eval(L, v["VH_map"], VV.ravel(), PP.ravel(), v)
    cats = np.array([_dab_cat(em, i) for i in range(VV.size)], dtype=object)
    res.add_metric("feasible_frac", f"V_H = {v['VH_map']:g} V 운전영역 중 제약 만족 비율", float(np.mean(np.isin(cats, ("feasible", "screen")))), "", basis=f"그중 ZVS screen 통과 {float(np.mean(cats == 'feasible')):.0%}")
    res.add_metric("unknown_frac", "손실 지도 외삽(UNKNOWN) 비율", float(np.mean(cats == "unknown")), "")
    shorts = {"nominal 800 V / 48 V": "N", "저전압 LV: 900 V / 36 V": "LB", "900 V / 36 V 무부하 (φ = 0)": "φ0", "550 V / 36 V": "C1", "550 V / 54 V": "C2", "900 V / 54 V": "C3"}
    mk = [{"x": c[2], "y": c[3] / 1e3, "short": shorts.get(c[0], ""), "label": c[0]} for c in corners if abs(c[1] - v["VH_map"]) < 1e-9]
    add_region_plot(res, "p_map", f"L = {fmt_si(L, 'H', 3)}, V_H = {v['VH_map']:g} V: 운전영역과 active limit (V_L × P, 1모듈)", VV.ravel(), PP.ravel() / 1e3, cats, DAB_CATS, DAB_COL, x_label="LV 전압 V_L", x_unit="V", y_label="모듈 전력 P", y_unit="kW",
                    markers=mk, proved="V_H = 900 V에서는 전압비가 맞는(V_L ≈ 54 V) 고전력 영역만 두 브리지 모두 ZVS screen을 통과한다. V_L이 낮으면 LV 브리지가, 전압비가 맞아도 경부하이면 HV 브리지가 screen에서 떨어진다 — SPS 하나로 전 영역 ZVS는 어렵다. 작은 L(실패 preset)에서는 RMS 한계 영역이 나타난다.",
                    not_yet="ZVS는 전류 부호·크기 screen이다(SCREEN_ONLY). 실제 commutation(EX02)·dead time 파형·자화전류는 이 모델에 없다.")
    notes = []
    for i in range(VV.size):
        if em["ok_p"][i]:
            notes.append(f"φ {math.degrees(em['phi'][i]):.1f}°, I_rms {em['irms'][i]:.2f} A, ZVS HV {'O' if em['zvs_hv'][i] else 'X'} / LV {'O' if em['zvs_lv'][i] else 'X'}")
        else:
            notes.append("SPS로 이 전력을 보낼 수 없음")
    add_loss_map(res, "p_lossmap", f"L = {fmt_si(L, 'H', 3)}, V_H = {v['VH_map']:g} V: 가능한 영역의 합성 손실 (1모듈, 빗금 = 제약 위반)", VLg, Pg / 1e3, cats, em["P_loss"], DAB_CATS, z_label="손실 (반도체 + transformer 동손)", z_unit="W",
                 x_label="LV 전압 V_L", x_unit="V", y_label="모듈 전력 P", y_unit="kW", ok=("feasible", "screen"), cell_notes=notes, markers=mk,
                 proved="허용 영역 안에서도 손실이 큰 곳이 두 군데다: 전압비가 맞지 않는 쪽(V_L 낮음)은 순환전류로 도통 손실이 크고, 전압비가 맞는 쪽의 경부하는 스위칭 전류가 작아 HV 브리지 ZVS screen이 깨지면서 Coss 에너지(hard switching)가 손실을 지배한다. 순환전류는 손실이면서 ZVS를 돕기도 한다.",
                 not_yet="코어·PCB·capacitor 손실은 넣지 않았다. ZVS 실패 셀의 손실은 hard-switching 합성 지도(E_oss floor 포함) 기준이며, 부분 ZVS·dead time 최적화·변조 변경은 EX02/EX06의 범위다.")
    xs = (fam * 1e6).tolist()
    res.add_series("fam_loss", "nominal 손실 (1모듈)", "W", xs, [x if math.isfinite(x) else None for x in fam_loss])
    ok_i = [i for i, o in enumerate(fam_ok) if o and math.isfinite(fam_loss[i])]
    bad_i = [i for i, o in enumerate(fam_ok) if not o and math.isfinite(fam_loss[i])]
    res.add_series("fam_ok", "모든 corner 만족", "W", [xs[i] for i in ok_i], [float(fam_loss[i]) for i in ok_i], style="points", color=C_FEAS)
    res.add_series("fam_bad", "corner 실패", "W", [xs[i] for i in bad_i], [float(fam_loss[i]) for i in bad_i], style="points", color="var(--c5)")
    fmk = []
    if i_nom is not None:
        fmk.append({"x": xs[i_nom], "y": float(fam_loss[i_nom]), "label": f"nominal 최적 {xs[i_nom]:g} µH"})
    if i_rob is not None:
        fmk.append({"x": xs[i_rob], "y": float(fam_loss[i_rob]), "label": f"robust {xs[i_rob]:g} µH"})
    res.add_plot("p_family", "nominal 최적 L vs 요구 corner를 모두 만족하는 L", ["fam_loss", "fam_ok", "fam_bad"], x_label="L (1차 환산)", x_unit="µH", y_label="nominal 손실", y_unit="W", kind="xy", level="A", markers=fmk,
                 proved="nominal 손실은 작은 L이 유리하지만, 전압 불일치 corner의 RMS가 한계를 넘는다. corner를 모두 만족하는 L이 따로 있다.",
                 not_yet="손실 차이는 합성 손실 지도 기준이다. 변조 자유도(EPS/DPS/TPS, EX06)를 쓰면 이 경계가 바뀐다.")
    # waveform at the low-battery corner and phi = 0 (exact segments)
    for key_, i_, lab_ in (("w_lb", 1, "900/36 V, 1.5 kW"), ("w_0", 2, "900/36 V, φ = 0"), ("w_nom", 0, "nominal 800/48 V")):
        if not e["ok_p"][i_]:
            continue
        t1 = float(e["phi"][i_]) / (2 * math.pi * v["fs"])
        Th = 0.5 / v["fs"]
        i0, ip = float(e["i0"][i_]), float(e["ip"][i_])
        ts = [0.0, t1, Th, Th + t1, 2 * Th]
        ys = [i0, ip, -i0, -ip, i0]
        res.add_series(key_, lab_, "A", [t * 1e6 for t in ts], ys)
    res.add_plot("p_wave", "1차 전류 파형 (반주기 반대칭 해, 구간 선형)", [k for k in ("w_nom", "w_lb", "w_0") if any(s.key == k for s in res.series)], x_label="t", x_unit="µs", y_label="i_L", y_unit="A", kind="xy", level="A",
                 proved=f"φ = 0이면 전달 전력은 0이지만 전압 불일치 때문에 L에 교번 전압이 걸려 전류가 흐른다(I_rms {float(e['irms'][2]):.4g} A @ L = {fmt_si(L, 'H', 3)}). 구간 공식은 PWL 엔진 적분과 SPS 닫힌 식 check로 확인했다.",
                 not_yet="이상 스위치 파형이다. dead time 중 전류 경로·자화전류·ringing은 포함하지 않았다.", group="dab_wave")
    res.circuit = {"diagram": dab_circuit(v, L).to_json(), "intervals": [], "plot_group": ""}
    # --- independent checks
    err_pwl, err_p = 0.0, 0.0
    for i in range(len(corners)):
        if not e["ok_p"][i]:
            continue
        pw = ref_dab.dab_pwl(float(cVH[i]), float(e["V2"][i]), v["fs"], L, float(e["phi"][i]))
        err_pwl = max(err_pwl, abs(pw["irms"] - float(e["irms"][i])) / float(e["irms"][i]), abs(pw["i0"] - float(e["i0"][i])) / max(abs(float(e["i0"][i])), 1e-9))
        Pcf = ref_dab.dab_power(float(cVH[i]), float(e["V2"][i]), v["fs"], L, float(e["phi"][i]))
        err_p = max(err_p, abs(float(e["P2"][i]) - Pcf) / max(Pm, 1.0), abs(float(e["P2"][i]) - float(cP[i])) / max(Pm, 1.0))
    res.add_check(Check("DAB 파형: 구간 공식(벡터화) vs PWL 엔진 적분", "PASS" if err_pwl < 1e-9 else "FAIL", err_pwl, "rel", 1e-9, path="lab의 기울기·반대칭 구간 합 vs reference PWL 객체의 RMS·초기전류", independent=True, detail=f"corner {int(np.sum(e['ok_p']))}곳"))
    res.add_check(Check("전달 전력: 2차 포트 구간 적분 vs SPS 닫힌 식 vs 지정 전력", "PASS" if err_p < 1e-9 else "FAIL", err_p, "rel", 1e-9, path="(2/T)∫v_2 i dt (구간 평균) vs V_1V_2′φ(1−φ/π)/(ωL)", independent=True))
    # --- verdicts
    if bad:
        res.verdict("FAIL_CONSTRAINT", f"L = {fmt_si(L, 'H', 3)}: " + "; ".join(f"{nm} → {DAB_CATS[c_]}" for nm, c_ in bad))
    elif unk:
        res.verdict("OUT_OF_VALIDITY", f"L = {fmt_si(L, 'H', 3)}: " + ", ".join(unk) + " — 손실 지도 외삽, T_j·손실 UNKNOWN")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"L = {fmt_si(L, 'H', 3)}: 모든 요구 corner에서 전력·RMS·T_j 제약 만족 (SPS, 합성 손실 지도)")
    res.verdict("SCREEN_ONLY", "ZVS는 전류 부호·크기 screen만 했다" + (f" — screen 실패 corner: {', '.join(screen)}" if screen else "") + ". ideal switch 모델의 ZVS 판정은 NOT_EVALUABLE (EX02·EX06)")
    if i_rob is None:
        res.verdict("NO_SOLUTION", "후보군 L 100–300 µH 안에 모든 요구 corner를 만족하는 SPS 설계가 없다 — 변조 자유도(EX06)·turns ratio·모듈 운전 전략을 검토")
    res.assumptions += ["SPS(단일 위상천이), 반주기 반대칭 정상상태 해, 이상 스위치; 자화전류 무시", "per module 1.5 kW; 합계 전력은 모듈 수 × 1모듈", "ZVS screen: i·t_dead ≥ 2Q_oss (단일 Q_oss 값)", "손실은 합성 손실 지도 후처리; T_j는 선형 PTC 정상상태"]
    res.not_valid_for += ["실제 ZVS·commutation 파형 (NOT_EVALUABLE)", "EPS/DPS/TPS 변조 (EX06)", "실제 소자·transformer 손실 (MISSING_INPUT)", "모듈 on/off 전환·startup"]
    res.interpretation = (
        "SPS DAB에서 L은 두 방향으로 작용한다. L이 작으면 같은 전력을 작은 φ로 보내 nominal RMS가 줄지만, 전압비가 맞지 않는 corner에서는 순환전류가 1/L로 커진다. "
        "L이 크면 저전압 corner에서 보낼 수 있는 최대 전력(V_1V_2′/(8fL))이 모자란다. φ = 0이어도 전압 불일치면 전류가 흐른다는 것이 무부하 corner의 교훈이다."
    )
    return res


# ======================================================================================
# Experiment 4: capstone C - 4-parallel SiC inverter leg (sharing, overshoot, coupling, T_j)
# ======================================================================================

LEG_CATS = {
    "feasible": "모든 제약 만족",
    "overshoot": "turn-off 과전압 한계",
    "gate_coupling": "공통 source 결합 전압 한계 (screen)",
    "branch_current": "branch 전류 한계",
    "tj": "T_j 한계 (가장 뜨거운 branch)",
    "runaway": "전열 고정점 없음",
    "unknown": "손실 지도 외삽 (UNKNOWN)",
}
LEG_COL = {"feasible": C_FEAS, "overshoot": "var(--c5)", "gate_coupling": "var(--c4)", "branch_current": "var(--c1)", "tj": "var(--c2)", "runaway": "var(--c8)", "unknown": C_UNK}
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_TH = (_GL_X + 1.0) * math.pi / 2.0
_TW = _GL_W * math.pi / 2.0


def leg_device(v) -> LossMap:
    """Synthetic E_on + E_off per event of one branch die at the reference gate resistance."""
    return LossMap("SiC leg E_sw", E0=v["Esw_ref"], a=1.15, b=1.35, Eoss=v["Eoss_leg"], I0=100.0, V0=800.0, V_lo=300.0, V_hi=900.0, I_lo=v["I_lo_char"], I_hi=250.0)


def leg_eval(Rg, Ipk, Vbus, Tcool, v, kelvin=None, iters=400):
    """Vectorised electro-thermal fixed point of the 4 branches of one switch position.

    Per branch k: I_k = I_pk G_k / sum G (static resistive sharing with R_k(T_k)), conduction
    R_k I_k^2 / 4 (synchronous rectification, sine), switching f_sw (1/2pi) int_0^pi E(V, I_k sin th) dth
    scaled by the gate factor, T_k = T_cool + R_th P_k.  Dynamic (gate-timing) sharing is not modelled.
    """
    Ipk, Vbus, Tcool = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float)) for x in (Ipk, Vbus, Tcool)))
    R25 = np.array([v["R1"], v["R2"], v["R3"], v["R4"]])
    dev = leg_device(v)
    gfac = (Rg + v["R_int"]) / (v["R_ref"] + v["R_int"])
    N = Ipk.size
    T = np.repeat(Tcool[:, None], 4, axis=1) + 10.0
    conv = np.zeros(N, dtype=bool)
    for _ in range(iters):
        R = R25 * (1.0 + v["tc"] * (T - 25.0))
        G = 1.0 / R
        Ik = Ipk[:, None] * G / G.sum(axis=1, keepdims=True)
        Pc = R * Ik**2 / 4.0
        E = dev.energy(Vbus[:, None, None], Ik[:, :, None] * np.sin(_TH)[None, None, :]) * gfac
        Ps = v["fsw"] * (E * _TW).sum(axis=-1) / (2.0 * math.pi)
        Tn = Tcool[:, None] + v["Rth"] * (Pc + Ps)
        conv = np.max(np.abs(Tn - T), axis=1) < 1e-10
        T = np.where(np.isfinite(Tn), Tn, 1e6)
        if np.all(conv | (T.max(axis=1) > 1000.0)):
            break
    runaway = ~conv | (T.max(axis=1) > 1000.0)
    # extrapolated share of the switching energy: currents below the characterised minimum or above it
    Isin = Ik[:, :, None] * np.sin(_TH)[None, None, :]
    E = dev.energy(Vbus[:, None, None], Isin)
    outside = ~dev.inside(Vbus[:, None, None], np.maximum(Isin, dev.I_lo))  # voltage outside, or I above the box
    below = Isin < dev.I_lo
    tot = (E * _TW).sum(axis=-1)
    ex = ((E * (below | outside)) * _TW).sum(axis=-1)
    frac_ex = (ex.sum(axis=1) / np.maximum(tot.sum(axis=1), 1e-30))
    unknown = (frac_ex > v["frac_ex_max"]) | (Ik.max(axis=1) > dev.I_hi) | (Vbus < dev.V_lo) | (Vbus > dev.V_hi)
    kel = v["kelvin"] if kelvin is None else kelvin
    S = v["S_ref"] * 1e6 * (v["R_ref"] + v["R_int"]) / (Rg + v["R_int"])  # per-branch di/dt [A/s] (S_ref in A/us)
    dV = v["L_loop"] * 4.0 * S
    Lcs = v["L_cs_kelvin"] if kel else v["L_cs"]
    Vcs = Lcs * S
    P_leg = 2.0 * (Pc + Ps).sum(axis=1)  # high and low side of the leg (symmetric)
    nan = np.full(N, np.nan)
    margins = {
        "overshoot": (Vbus + dV - v["V_lim"]) / v["V_lim"],
        "gate_coupling": np.full(N, (Vcs - v["Vcs_max"]) / v["Vcs_max"]),
        "branch_current": Ik.max(axis=1) / v["I_br_max"] - 1.0,
        "tj": np.where(runaway, nan, (T.max(axis=1) - v["Tj_max"]) / (v["Tj_max"] - Tcool)),
    }
    return dict(Ipk=Ipk, Vbus=Vbus, Tcool=Tcool, Ik=Ik, T=T, Pc=Pc, Ps=Ps, P_leg=P_leg, S=S, dV=dV, Vcs=Vcs, margins=margins, unknown=unknown & ~runaway, runaway=runaway, frac_ex=frac_ex, gfac=gfac)


def _leg_cat(e, i):
    mg = {k: np.array([e["margins"][k][i]]) for k in e["margins"]}
    return str(classify(mg, np.array([bool(e["unknown"][i])]), {"runaway": np.array([bool(e["runaway"][i])])})[0])


def leg_circuit(v: dict, Rg: float) -> Circuit:
    c = Circuit("ex01_leg", 900, 330, title=f"병렬 4 branch 스위치 (한 위치), R_g = {Rg:g} Ω, {'Kelvin source' if v['kelvin'] else '전력 source 기준 gate'}")
    c.add("vsource", "Vdc", 60, 170, 90, "V_bus", "", lpos=(44, 166, "end"))
    xs = [230, 390, 550, 710]
    names = ["R1", "R2", "R3", "R4"]
    for k, x in enumerate(xs):
        c.add("nmos", f"Q{k + 1}", x, 110, 90, f"Q{k + 1}", fmt_si(v[names[k]], "Ω", 3), lpos=(x + 22, 104, "start"))
        c.add("inductor", f"Ls{k + 1}", x, 210, 90, f"L_s{k + 1}", fmt_si(v["L_cs"], "H", 2), lpos=(x + 20, 206, "start"))
        c.wire(f"wd{k}", (x, 80), (x, 50))
        c.wire(f"wm{k}", (x, 140), (x, 180))
        c.wire(f"ws{k}", (x, 240), (x, 280))
        c.probe(f"pi{k}", "none", x - 14, 160, "down", f"i_{k + 1}")
    c.wire("wtop", (60, 140), (60, 50), (710, 50))
    c.wire("wbot", (60, 200), (60, 280), (710, 280))
    c.dot((230, 50), (390, 50), (550, 50), (230, 280), (390, 280), (550, 280))
    c.text(470, 316, "gate 루프가 L_s를 공유하면 v = L_s·di/dt가 gate 전압에서 빠진다 (Kelvin source는 분리)", "small")
    c.text(830, 110, "L_loop", "small")
    c.text(830, 128, fmt_si(v["L_loop"], "H", 2), "small")
    active = ["Vdc", "wtop", "wbot"] + [f"Q{k + 1}" for k in range(4)] + [f"Ls{k + 1}" for k in range(4)] + [f"wd{k}" for k in range(4)] + [f"wm{k}" for k in range(4)] + [f"ws{k}" for k in range(4)]
    c.mode("on", "4 branch 도통 (정적 분배)", active, "정적 전류 분배는 R_DS(on)(T)만으로 정해진다 — 스위칭 순간의 동적 분배(gate timing·L_s)는 EX03")
    return c


def run_sic_leg(v: dict) -> Result:
    res = Result("EX01", "sic_leg", "A (정적·전열 분배 + 합성 손실 지도 + 결합 screen)")
    Rg = v["Rg"]
    R25 = [v["R1"], v["R2"], v["R3"], v["R4"]]
    tbR = all(abs(a - b) < 1e-15 for a, b in zip(R25, [3.6e-3, 4.0e-3, 4.0e-3, 4.4e-3]))
    # --- textbook static sharing (temperature-independent resistances)
    I_st = [v["I_nom"] * (1 / r) / sum(1 / x for x in R25) for r in R25]
    tb400 = tbR and abs(v["I_nom"] - 400.0) < 1e-9
    for k, (val, rv) in enumerate(zip(I_st, (110.553, 99.497, 99.497, 90.452))):
        res.add_metric(f"I_static_{k + 1}", f"정적 분배 branch {k + 1} (R = {R25[k] * 1e3:g} mΩ, 온도 무관)", val, "A", ref=rv if tb400 else None, ref_label=f"교재 {rv} A", tol=5e-6)
    I_ref = ref.static_share(v["I_nom"], R25)
    res.add_check(check_close("정적 분배: lab 컨덕턴스 분배 vs reference", max(abs(a - b) for a, b in zip(I_st, I_ref)), 0.0, 1e-12, "같은 식의 두 구현 (회귀)", False, "A", abs_scale=1.0))
    S0 = v["S_ref"] * 1e6 * (v["R_ref"] + v["R_int"]) / (Rg + v["R_int"])  # A/s
    Lcs_eff = v["L_cs_kelvin"] if v["kelvin"] else v["L_cs"]
    res.add_metric("didt", f"branch di/dt (R_g = {Rg:g} Ω)", S0 * 1e-6, "A/µs", basis=f"기준 {v['S_ref'] * 1e-3:g} kA/µs @ {v['R_ref']:g} Ω, 전이시간 ∝ R_g + R_int (합성)")
    tb_cs = abs(v["L_cs"] - 2e-9) < 1e-15 and abs(S0 - 2e9) < 1e-3 and not v["kelvin"]
    res.add_metric("V_cs", "공통 source 결합 L_s·di/dt", Lcs_eff * S0, "V", ref=4.0 if tb_cs else None, ref_label="교재 2 nH × 2 kA/µs = 4 V", tol=1e-12, basis=("Kelvin source: " if v["kelvin"] else "전력 source 기준: ") + f"L_s = {fmt_si(Lcs_eff, 'H', 2)} (screen)")
    res.add_metric("skew_dI", "gate timing skew의 전류차 scale", S0 * v["skew"], "A", ref=20.0 if (abs(S0 - 2e9) < 1e-3 and abs(v["skew"] - 10e-9) < 1e-18) else None, ref_label="교재 10 ns × 2 kA/µs = 20 A", tol=1e-12, basis="같은 slew 가정의 scale — branch 전류 예측이 아니다")
    # --- corners
    corners = [
        ("nominal", v["I_nom"], 800.0, v["T_cool"]),
        ("최대 bus", v["I_nom"], v["V_bus_max"], v["T_cool"]),
        ("최대 전류", v["I_max"], 700.0, v["T_cool"]),
        ("고온 냉각수", 0.75 * v["I_max"], 800.0, v["T_cool_hot"]),
    ]
    cI = np.array([c[1] for c in corners])
    cV = np.array([c[2] for c in corners])
    cT = np.array([c[3] for c in corners])
    e = leg_eval(Rg, cI, cV, cT, v)
    rows, bad, unk = [], [], []
    for i, (nm, I, V, Tc) in enumerate(corners):
        cat = _leg_cat(e, i)
        k, gm = tightest({kk: float(e["margins"][kk][i]) for kk in e["margins"]})
        if cat == "unknown":
            unk.append(nm)
        elif cat != "feasible":
            bad.append((nm, cat))
        rows.append([nm, I, V, Tc, " / ".join(f"{x:.1f}" for x in e["Ik"][i]), " / ".join(f"{x:.1f}" for x in e["T"][i]), float(e["P_leg"][i]), V + float(e["dV"]), float(e["Vcs"]), f"{float(e['frac_ex'][i]):.0%}", LEG_CATS[cat], f"{k} ({gm:+.1%})"])
    res.tables.append(Table("t_corners", f"R_g = {Rg:g} Ω: 요구 corner 판정", ["corner", "I_pk 합계 [A]", "V_bus [V]", "T_cool [°C]", "branch 전류 [A]", "branch T_j [°C]", "leg 손실 [W]", "V_peak [V]", "V_cs [V]", "외삽 에너지 비율", "판정", "가장 가까운 제약 (g)"], rows,
                            note="leg 손실 = 상·하 스위치 위치 합(대칭). 전류는 fundamental peak 기준의 정적 분배, 손실은 sine 평균. 외삽 에너지 비율 = 특성화 전류 하한 아래에서 계산된 스위칭 에너지의 몫."))
    res.add_metric("I_hot", "nominal: 가장 큰 branch 전류 (전열 연성)", float(e["Ik"][0].max()), "A", basis=f"정적(온도 무관) {max(I_st):.2f} A → PTC가 약간 균등화")
    res.add_metric("T_hot", "nominal: 가장 뜨거운 branch T_j", float(e["T"][0].max()), "°C", basis=f"branch {int(np.argmax(e['T'][0])) + 1} (R이 가장 작은 branch가 가장 많은 전류)")
    res.add_metric("loss_nominal", f"R_g = {Rg:g} Ω: nominal leg 손실", float(e["P_leg"][0]), "W", basis="postprocessed loss estimate (합성 손실 지도)")
    # --- candidate family
    fam = [1.0, 2.2, 3.3, 4.7, 6.8, 10.0]
    frows, fam_loss, fam_ok, fam_t = [], [], [], []
    for rg in fam:
        ec = leg_eval(rg, cI, cV, cT, v)
        cats = [_leg_cat(ec, i) for i in range(len(corners))]
        ok_all = all(c_ == "feasible" for c_ in cats)
        nl = float(ec["P_leg"][0]) if cats[0] in ("feasible", "gate_coupling", "overshoot") else float("nan")
        fam_loss.append(nl)
        fam_ok.append(ok_all)
        mi = max(range(len(corners)), key=lambda i: tightest({k: float(ec["margins"][k][i]) for k in ec["margins"]})[1])
        mk_, mg_ = tightest({k: float(ec["margins"][k][mi]) for k in ec["margins"]})
        fam_t.append((mk_, mg_, corners[mi][0]))
        fails = [f"{corners[i][0]}: {LEG_CATS[c_]}" for i, c_ in enumerate(cats) if c_ != "feasible"]
        frows.append([rg, nl, float(ec["S"]) * 1e-6, float(ec["Vcs"]), "예" if ok_all else "아니오", f"{mk_} @ {corners[mi][0]} ({mg_:+.1%})", "; ".join(fails) or "—"])
    fam_loss = np.array(fam_loss)
    i_nom = int(np.nanargmin(fam_loss)) if np.any(np.isfinite(fam_loss)) else None
    ok_idx = [i for i in range(len(fam)) if fam_ok[i] and math.isfinite(fam_loss[i])]
    i_rob = min(ok_idx, key=lambda i: fam_loss[i]) if ok_idx else None
    res.tables.append(Table("t_family", f"후보군 R_g ({'Kelvin source' if v['kelvin'] else '전력 source 기준 gate'}): nominal 손실과 corner 만족", ["R_g [Ω]", "nominal leg 손실 [W]", "di/dt [A/µs]", "V_cs [V]", "모든 corner 만족", "지배 제약 (g)", "실패 corner"], frows,
                            note="빠른 gate(작은 R_g)는 스위칭 손실을 줄이지만 L·di/dt 과전압과 공통 source 결합을 키운다. nominal 손실만 보면 가장 빠른 gate가 이긴다."))
    res.add_metric("Rg_nominal_opt", "nominal 최적 R_g (손실 최소)", fam[i_nom] if i_nom is not None else "없음", "Ω" if i_nom is not None else "", basis=f"{fam_loss[i_nom]:.0f} W" if i_nom is not None else "")
    res.add_metric("Rg_robust", "요구 corner 모두 만족하는 R_g", fam[i_rob] if i_rob is not None else "없음", "Ω" if i_rob is not None else "", basis=(f"{fam_loss[i_rob]:.0f} W (nominal 최적 대비 +{fam_loss[i_rob] - fam_loss[i_nom]:.0f} W)" if (i_rob is not None and i_nom is not None) else "후보군 안에 없음"))
    wk, wg, wc = fam_t[i_rob] if i_rob is not None else ("—", float("nan"), "—")
    datum_map = {
        "overshoot": f"실제 전류 루프 L_loop와 turn-off di/dt(현재 {v['L_loop'] * 1e9:g} nH·합성 R_g 관계) — {wc}의 과전압 여유 {-wg:.1%}",
        "gate_coupling": f"gate 기준점(Kelvin 여부)과 공통 source L_s — 결합 여유 {-wg:.1%}",
        "tj": f"branch별 R_th·R_DS(on)(T) 실측 — {wc}의 T_j 여유 {-wg:.1%}",
        "branch_current": f"branch 전류 정격·동적 분배 — {wc}의 여유 {-wg:.1%}",
    }
    ci = max(range(len(corners)), key=lambda i: tightest({kk: float(e["margins"][kk][i]) for kk in e["margins"]})[1])
    ck, cg = tightest({kk: float(e["margins"][kk][ci]) for kk in e["margins"]})
    res.add_metric("dominant", f"R_g = {Rg:g} Ω: 지배 제약 · 최악 corner", f"{ck} @ {corners[ci][0]}", "", basis=f"g = {cg:+.1%}")
    res.add_metric("dominant_robust", "robust 선택의 지배 제약 · 최악 corner", f"{wk} @ {wc}" if i_rob is not None else "—", "", basis=f"g = {wg:+.1%}" if math.isfinite(wg) else "")
    res.add_metric("unconfirmed", "주장을 바꿀 수 있는 미확인 데이터 하나", datum_map.get(wk, "branch별 동적 전류 분배(gate timing skew)") if i_rob is not None else "V_bus 최대값·과전압 허용치 자체", "")
    # --- map (I_pk x V_bus) at T_cool
    Ig = np.arange(50.0, 1000.0 + 1e-9, 50.0)
    Vg = np.arange(400.0, 900.0 + 1e-9, 25.0)
    II, VV = np.meshgrid(Ig, Vg, indexing="ij")
    em = leg_eval(Rg, II.ravel(), VV.ravel(), v["T_cool"], v, iters=200)
    cats = classify(em["margins"], em["unknown"], {"runaway": em["runaway"]})
    res.add_metric("feasible_frac", f"R_g = {Rg:g} Ω 운전영역 중 모든 제약 만족 비율", float(np.mean(cats == "feasible")), "", basis=f"{II.size}점, T_cool {v['T_cool']:g} °C")
    res.add_metric("unknown_frac", "손실 지도 외삽(UNKNOWN) 비율", float(np.mean(cats == "unknown")), "", basis=f"외삽 에너지 > {v['frac_ex_max']:.0%} 또는 지도 범위 밖 ({leg_device(v).box_text()})")
    shorts = {"nominal": "N", "최대 bus": "V", "최대 전류": "I", "고온 냉각수": "T"}
    mk = [{"x": c[1], "y": c[2], "short": shorts.get(c[0], ""), "label": c[0]} for c in corners if abs(c[3] - v["T_cool"]) < 1e-9]
    add_region_plot(res, "p_map", f"R_g = {Rg:g} Ω: 운전영역과 active limit (I_pk × V_bus, T_cool {v['T_cool']:g} °C)", II.ravel(), VV.ravel(), cats, LEG_CATS, LEG_COL, x_label="leg 전류 I_pk (4 branch 합)", x_unit="A", y_label="V_bus", y_unit="V",
                    markers=mk, proved="과전압은 V_bus와 di/dt가, T_j는 전류가, 외삽(UNKNOWN)은 경부하가 정한다. 같은 소자라도 gate 설정에 따라 가능한 영역이 바뀐다. branch T_j는 고정점 반복과 Newton(fsolve) check, 스위칭 평균은 Beta 함수 닫힌 식 check가 뒷받침한다.",
                    not_yet="정적 분배와 sine 평균 손실이다. 스위칭 순간의 동적 분배·gate timing skew·실제 ringing은 EX03 범위다.")
    notes = [f"T_j,max {em['T'][i].max():.0f} °C, V_peak {em['Vbus'][i] + float(em['dV']):.0f} V, 외삽 에너지 {em['frac_ex'][i]:.0%}" for i in range(II.size)]
    add_loss_map(res, "p_lossmap", f"R_g = {Rg:g} Ω: 가능한 영역의 leg 손실 (빗금 = 제약 위반·UNKNOWN)", Ig, Vg, cats, em["P_leg"], LEG_CATS, z_label="leg 손실 (상·하 합)", z_unit="W",
                 x_label="leg 전류 I_pk", x_unit="A", y_label="V_bus", y_unit="V", cell_notes=notes, markers=mk,
                 proved="허용 영역 안에서 손실은 전류와 전압에 따라 커진다. 경부하 셀은 손실 지도 하한 아래라 숫자를 칠하지 않았다(UNKNOWN).",
                 not_yet="손실은 합성 손실 지도 후처리다. 동적 분배·ringing·실제 R_th 차이는 포함하지 않았다.")
    xs = list(fam)
    res.add_series("fam_loss", "nominal leg 손실", "W", xs, [x if math.isfinite(x) else None for x in fam_loss])
    ok_i = [i for i, o in enumerate(fam_ok) if o and math.isfinite(fam_loss[i])]
    bad_i = [i for i, o in enumerate(fam_ok) if not o and math.isfinite(fam_loss[i])]
    res.add_series("fam_ok", "모든 corner 만족", "W", [xs[i] for i in ok_i], [float(fam_loss[i]) for i in ok_i], style="points", color=C_FEAS)
    res.add_series("fam_bad", "corner 실패", "W", [xs[i] for i in bad_i], [float(fam_loss[i]) for i in bad_i], style="points", color="var(--c5)")
    fmk = []
    if i_nom is not None:
        fmk.append({"x": xs[i_nom], "y": float(fam_loss[i_nom]), "label": f"nominal 최적 {xs[i_nom]:g} Ω"})
    if i_rob is not None:
        fmk.append({"x": xs[i_rob], "y": float(fam_loss[i_rob]), "label": f"robust {xs[i_rob]:g} Ω"})
    res.add_plot("p_family", "nominal 최적 R_g vs 요구 corner를 모두 만족하는 R_g", ["fam_loss", "fam_ok", "fam_bad"], x_label="외부 gate 저항 R_g", x_unit="Ω", y_label="nominal leg 손실", y_unit="W", kind="xy", level="A", markers=fmk,
                 proved="nominal 손실은 가장 빠른 gate가 가장 작지만, 최대 bus corner의 과전압·결합 전압이 그 선택을 막는다.",
                 not_yet="R_g와 di/dt·손실의 관계는 합성 비례식이다. 실제로는 DPT로 R_g별 E_on/E_off와 과전압을 측정해야 한다.")
    Iq = np.linspace(50.0, v["I_max"], 40)
    et = leg_eval(Rg, Iq, 800.0, v["T_cool"], v, iters=200)
    for k in range(4):
        res.add_series(f"Tb{k + 1}", f"branch {k + 1} ({R25[k] * 1e3:g} mΩ)", "°C", Iq.tolist(), et["T"][:, k].tolist())
    res.add_plot("p_tj", "branch별 T_j (800 V, 전열 고정점)", [f"Tb{k + 1}" for k in range(4)], x_label="leg 전류 I_pk", x_unit="A", y_label="T_j", y_unit="°C", kind="xy", level="A", hlines=[{"y": v["Tj_max"], "label": "T_j 한계"}],
                 proved="R이 가장 작은 branch가 전류를 가장 많이 받아 가장 뜨겁다. R_DS(on)의 양의 온도계수가 분배를 약간 균등하게 만든다. 온도계수 0의 극한에서 분배가 교재 정적 값(110.553 A)과 같다는 check로 연결했다.",
                 not_yet="모든 branch의 R_th가 같다고 가정했다. 열 계면 차이·센서 위치·동적 분배로 ‘뜨거운 branch’가 달라질 수 있다(원인은 판별 시험으로 분리).")
    res.circuit = {"diagram": leg_circuit(v, Rg).to_json(), "intervals": [], "plot_group": ""}
    # --- independent checks
    j = 2  # maximum-current corner
    Tfp = e["T"][j]

    def resid(Tv):
        R = np.array(R25) * (1 + v["tc"] * (Tv - 25.0))
        G = 1 / R
        Ik = cI[j] * G / G.sum()
        E = leg_device(v).energy(cV[j], Ik[:, None] * np.sin(_TH)[None, :]) * e["gfac"]
        Ps = v["fsw"] * (E * _TW).sum(axis=-1) / (2 * math.pi)
        return Tv - (cT[j] + v["Rth"] * (R * Ik**2 / 4 + Ps))

    Tn = fsolve(resid, np.full(4, cT[j] + 50.0), xtol=1e-13)
    res.add_check(check_close("전열 고정점: 반복 대입 vs Newton(fsolve)", float(np.max(np.abs(Tn - Tfp))), 0.0, 1e-7, "벡터화 고정점 반복 vs scipy fsolve (다른 알고리즘, 다른 시작점)", True, "°C", abs_scale=1.0, detail=f"최대 차 {float(np.max(np.abs(Tn - Tfp))):.1e} K @ {corners[j][0]}"))
    a_ = 1.15
    dev0 = replace(leg_device(v), Eoss=0.0)
    num = float((dev0.energy(800.0, 100.0 * np.sin(_TH)) * _TW).sum() / (2 * math.pi))
    ana = float(dev0.energy(800.0, 100.0)) * ref.sine_power_average(a_)
    res.add_check(check_close("sine 평균 스위칭 에너지: Gauss–Legendre vs Beta 함수 닫힌 식", num, ana, 1e-6, "(1/2π)∫E(I sin θ)dθ 48점 구적 vs E(I)·Γ((α+1)/2)/(2√π Γ(α/2+1)); sin^α의 끝점 특이성 때문에 구적 오차 ~1e-8", True, "J"))
    e0 = leg_eval(Rg, [v["I_nom"]], [800.0], [v["T_cool"]], dict(v, tc=0.0))
    res.add_check(check_close("극한: 온도계수 0이면 전열 분배 = 교재 정적 분배", float(e0["Ik"][0][0]), I_st[0], 1e-12, "tc = 0에서 고정점 분배 vs 컨덕턴스 분배", True, "A"))
    # --- verdicts
    if bad:
        res.verdict("FAIL_CONSTRAINT", f"R_g = {Rg:g} Ω: " + "; ".join(f"{nm} → {LEG_CATS[c_]}" for nm, c_ in bad))
    elif unk:
        res.verdict("OUT_OF_VALIDITY", f"R_g = {Rg:g} Ω: " + ", ".join(unk) + " — 손실 지도 외삽 비율이 커 T_j·손실 UNKNOWN")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"R_g = {Rg:g} Ω: 모든 요구 corner에서 과전압·T_j·branch 전류 만족 (정적 분배·합성 손실 지도)")
    res.verdict("SCREEN_ONLY", f"공통 source 결합 {Lcs_eff * S0:.2f} V는 L_s·di/dt screen이다 — 실제 gate 파형·false turn-on 여부는 측정(기준점 확인) 필요")
    if i_rob is None:
        res.verdict("NO_SOLUTION", "후보군 R_g 1–10 Ω 안에 모든 corner를 만족하는 값이 없다 — 루프 인덕턴스·Kelvin source·V_bus 요구를 다시 검토")
    res.assumptions += ["정적 분배 I_k ∝ 1/R_k(T_k), R = R25(1 + α(T − 25))", "sine 전류, 동기정류: 도통 R·I_pk²/4, 스위칭은 해당 반주기 sine 평균", "di/dt ∝ 1/(R_g + R_int), E_sw ∝ (R_g + R_int) (합성 비례식)", "과전압 = L_loop × 4 branch di/dt, 모든 branch R_th 동일"]
    res.not_valid_for += ["동적 전류 분배·gate timing skew (EX03)", "실제 ringing·false turn-on 파형", "단락·보호 동작 (EX10)", "실제 소자 손실·SOA (MISSING_INPUT)"]
    res.interpretation = (
        "nominal 손실만 보면 가장 빠른 gate(작은 R_g)가 이기지만, 최대 bus 전압에서 L_loop·di/dt 과전압과 공통 source 결합 전압이 한계를 넘는다. "
        "4개 branch의 R_DS(on) 차이로 정적 전류가 110.6/99.5/99.5/90.5 A로 나뉘어 R이 가장 작은 branch가 가장 뜨겁다. 경부하 손실은 측정 범위 아래라 UNKNOWN이다."
    )
    return res


# ======================================================================================
# Experiment 5: A/B ranking under uncertainty, mission efficiency, loss accounting (OBC CLLC)
# ======================================================================================

SPEC_KO = {"standard": "표준불확도(1σ)로 해석", "rect_limit": "±한계(직사각) → a/√3", "k2": "확장불확도 k = 2 → a/2"}
BASIS_KO = {"none": "근거 기록 없음", "same_fixture": "같은 DPT fixture·같은 probe·같은 보정으로 두 후보 측정 (공통 오차)"}
PROFILES = {
    # (V_bat [V], P_bat [W], duration [min]) - synthetic CC-CV session at the robust design
    "cc_cv": [(660, None, 10), (700, None, 10), (740, None, 10), (780, None, 10), (820, None, 10), (860, None, 10), (900, 8000, 8), (900, 5500, 10), (900, 3500, 12), (900, 2400, 15)],
    "deep_taper": [(660, None, 10), (700, None, 10), (740, None, 10), (780, None, 10), (820, None, 10), (860, None, 10), (900, 8000, 8), (900, 5500, 10), (900, 3500, 12), (900, 2400, 15), (900, 1200, 20), (900, 600, 20)],
    "cc_only": [(660, None, 10), (700, None, 10), (740, None, 10), (780, None, 10), (820, None, 10), (860, None, 10)],
    # single-phase 16 A charging: the same isolated stage runs most of its time near 3.5 kW
    "single_phase": [(700, 3500, 36), (740, 3500, 36), (780, 3500, 36), (820, 3500, 36), (860, 3500, 36)],
}


def _mission_loss(vX: dict, n: float, Vb, P, extra: dict):
    """CLLC loss breakdown of one candidate at points (Vb, P), with the accounting options applied."""
    e = cllc_eval(n, Vb, P, vX)
    parts = dict(e["parts"])
    f = e["f"]
    t_d = vX["t_dead"]
    # SR dead-time diode conduction around the current zero: i ~ sqrt2 I2 w t over each dead time
    i_dt = math.sqrt(2.0) * e["I2"] * 2 * np.pi * f * t_d / 2.0
    parts["sr_diode"] = 4.0 * extra["V_F"] * i_dt * 2.0 * t_d * f
    parts["sr_cond"] = parts["sr_cond"] * (1.0 if extra["sr_double"] else (1.0 - 2.0 * t_d * f))
    if extra["magnetics_dc_only"]:
        parts["copper"] = parts["copper"] / extra["k_ac"]
    eoss = extra["Eoss"] * (e["Vl"] / 800.0) ** 1.5
    parts["eoss_added"] = 4.0 * f * eoss if extra["add_eoss"] else np.zeros_like(f)
    hard = LossMap("hard E_on", E0=3.0 * vX["Eoff_20A"], a=1.1, b=1.3, Eoss=extra["Eoss"], I0=20.0, V0=800.0)
    parts["hard_eon"] = 4.0 * f * hard.energy(e["Vl"], np.maximum(e["ioff"], 0.0)) if extra["hard_eon"] else np.zeros_like(f)
    total = sum(parts.values())
    return e, parts, total


def run_mission(v: dict) -> Result:
    res = Result("EX01", "mission_ranking", "A (FHA 운전점 + 합성 손실 지도) + Monte Carlo")
    base = {p.key: p.default for p in P_CLLC}
    base.update(P_burst=v["P_burst"], t_dead=v["t_dead"])
    n = v["n_design"]
    k = v["k_cov"]
    extra = dict(V_F=v["V_F"], sr_double=v["sr_double"], magnetics_dc_only=v["magnetics_dc_only"], k_ac=v["k_ac"], add_eoss=v["add_eoss"], hard_eon=v["hard_eon"])
    vB = dict(base, Rds=v["Rds_B"], Eoff_20A=v["Eoff_B"])
    EossB, EossA = v["Eoss_B"], v["Eoss_B"] * v["sw_scale_A"]
    nomV, nomP = 800.0, base["P_rated"]
    # candidate A: switching energy x sw_scale_A, R_DS(on) derived from the textbook scenario
    # "A is dP_nom better at nominal" (a scenario definition, not a physics result)
    _, _, LB = _mission_loss(vB, n, [nomV], [nomP], dict(extra, Eoss=EossB, add_eoss=False, hard_eon=False, sr_double=False, magnetics_dc_only=False))
    LB = float(LB[0])

    def loss_A(R):
        vA_ = dict(base, Rds=R, Eoff_20A=v["Eoff_B"] * v["sw_scale_A"])
        return float(_mission_loss(vA_, n, [nomV], [nomP], dict(extra, Eoss=EossA, add_eoss=False, hard_eon=False, sr_double=False, magnetics_dc_only=False))[2][0])

    target = LB - v["dP_nom"]
    lo_R, hi_R = 1e-5, 3.0 * v["Rds_B"]
    if not (loss_A(lo_R) < target < loss_A(hi_R)):
        res.verdict("OUT_OF_VALIDITY", f"시나리오 불가: A의 스위칭 배율 {v['sw_scale_A']:g}로는 R_DS(on)을 어떻게 잡아도 nominal에서 {v['dP_nom']:g} W 우위를 만들 수 없다")
        return res
    R_A = brentq(lambda R: loss_A(R) - target, lo_R, hi_R, xtol=1e-12, rtol=1e-12)
    vA = dict(base, Rds=R_A, Eoff_20A=v["Eoff_B"] * v["sw_scale_A"])
    res.add_metric("Rds_A", "A의 R_DS(on) @ 25 °C (시나리오에서 도출)", R_A, "Ω", basis=f"B {v['Rds_B'] * 1e3:g} mΩ; ‘A가 nominal에서 {v['dP_nom']:g} W 좋다’를 만족하도록 계산 (DERIVED)")
    # --- nominal ranking under the switching-loss model uncertainty (textbook pressure question)
    u_nom = {"standard": v["u_sw"], "rect_limit": v["u_sw"] / math.sqrt(3.0), "k2": v["u_sw"] / 2.0}[v["u_kind"]]
    rho_stated = v["rho_AB"]
    rho = rho_stated if v["rho_basis"] != "none" else 0.0  # an unrecorded correlation is not used to shrink u
    _, pAn, LAn = _mission_loss(vA, n, [nomV], [nomP], dict(extra, Eoss=EossA))
    _, pBn, LBn = _mission_loss(vB, n, [nomV], [nomP], dict(extra, Eoss=EossB))
    dP = float(LBn[0] - LAn[0])
    u_d = ref.u_difference(u_nom, u_nom, rho)
    winner, resolved = rank_under_uncertainty(dP, u_d, k)
    clean = not (v["add_eoss"] or v["hard_eon"] or v["sr_double"] or v["magnetics_dc_only"])
    res.add_metric("dP_nom", "nominal 손실 차 B − A (양수: A가 작다)", dP, "W", ref=v["dP_nom"] if clean else None, ref_label="시나리오 (교재 8 W)", tol=1e-9, basis=f"A {float(LAn[0]):.1f} W, B {float(LBn[0]):.1f} W @ 800 V·{nomP / 1e3:g} kW")
    res.add_metric("u_sw", "스위칭 손실 모델 표준불확도 (후보당, nominal)", u_nom, "W", basis=f"±{v['u_sw']:g} W — {SPEC_KO[v['u_kind']]}")
    res.add_metric("u_diff_nom", f"차이의 표준불확도 (판정에 쓴 ρ_AB = {rho:g})", u_d, "W", basis=f"k·u = {k * u_d:.3g} W; 상관 근거: {BASIS_KO[v['rho_basis']]}")
    if rho_stated != rho:
        res.add_metric("u_diff_claimed", f"근거 없이 ρ_AB = {rho_stated:g}를 쓰면", ref.u_difference(u_nom, u_nom, rho_stated), "W", basis="주장할 수 없는 값 (MISSING_INPUT) — 판정에는 ρ = 0")
    res.add_metric("rank_nom", "nominal 순위", winner, "", basis=f"|Δ| {abs(dP):.3g} W {'>' if resolved else '≤'} k·u_Δ {k * u_d:.3g} W")
    # --- mission
    prof = PROFILES[v["profile"]]
    Vb = np.array([p[0] for p in prof], dtype=float)
    Pm = np.array([p[1] if p[1] is not None else v["P_cc"] for p in prof], dtype=float)
    dt_h = np.array([p[2] for p in prof], dtype=float) / 60.0
    eA, pA, LA = _mission_loss(vA, n, Vb, Pm, dict(extra, Eoss=EossA))
    eB, pB, LB_ = _mission_loss(vB, n, Vb, Pm, dict(extra, Eoss=EossB))
    burst = Pm < v["P_burst"]
    evaluable = eA["root"] & eB["root"] & ~burst & ~eA["unknown"] & ~eB["unknown"]
    Eout = Pm * dt_h
    ok = evaluable
    EA_loss = float(np.sum(LA[ok] * dt_h[ok]))
    EB_loss = float(np.sum(LB_[ok] * dt_h[ok]))
    Eo = float(np.sum(Eout[ok]))
    etaA_k = Pm / (Pm + LA)
    etaB_k = Pm / (Pm + LB_)
    eta_mis_A = Eo / (Eo + EA_loss)
    eta_mis_B = Eo / (Eo + EB_loss)
    eta_ar_A = float(np.mean(etaA_k[ok]))
    eta_tw_A = float(np.sum(etaA_k[ok] * dt_h[ok]) / np.sum(dt_h[ok]))
    res.add_metric("eta_mission_A", "A: mission 효율 ΣE_out/ΣE_in", eta_mis_A, "", basis=f"E_out {Eo / 1e3:.2f} kWh, 손실 {EA_loss:.1f} Wh")
    res.add_metric("eta_mission_B", "B: mission 효율", eta_mis_B, "", basis=f"손실 {EB_loss:.1f} Wh")
    res.add_metric("eta_arith_A", "A: 운전점 효율의 산술평균 (잘못된 정의)", eta_ar_A, "", basis=f"mission 효율과 {1e2 * (eta_ar_A - eta_mis_A):+.3f} %p — 손실 에너지로 {Eo * (1 / eta_ar_A - 1) - EA_loss:+.1f} Wh")
    res.add_metric("eta_time_A", "A: 시간가중 효율 평균 (역시 에너지 비가 아님)", eta_tw_A, "", basis=f"mission 효율과 {1e2 * (eta_tw_A - eta_mis_A):+.3f} %p")
    dE = EB_loss - EA_loss
    # model error is systematic per candidate: fully correlated across the mission points
    swA = float(np.sum((pA["pri_off"] * dt_h)[ok]))
    swB = float(np.sum((pB["pri_off"] * dt_h)[ok]))
    rA = u_nom / float(pAn["pri_off"][0])
    rB = u_nom / float(pBn["pri_off"][0])
    UA, UB = rA * swA, rB * swB
    uE = ref.u_difference(UA, UB, rho)
    wE, resE = rank_under_uncertainty(dE, uE, k)
    res.add_metric("dE_mission", "mission 손실 에너지 차 B − A", dE, "Wh", basis=f"중앙값으로는 {'A' if dE > 0 else 'B'}가 작다 (nominal 순위와 {'같음' if (dE > 0) == (dP > 0) else '반대'}); k·u = {k * uE:.3g} Wh — 스위칭 모델 오차는 후보마다 계통적(운전점 사이 완전 상관)")
    res.add_metric("rank_mission", "mission 순위", wE if bool(np.all(evaluable)) else "UNKNOWN (평가 불가 구간 포함)", "", basis=f"|ΔE| {abs(dE):.3g} Wh {'>' if resE else '≤'} k·u {k * uE:.3g} Wh")
    # --- loss accounting table at nominal (candidate A and B)
    names = [
        ("pri_cond", "채널 도통 (1차)", "도통 구간 i(t), R(T, V_g)", "full-bridge 소자당 I²/2·R(T)"),
        ("sr_cond", "채널 도통 (SR)", "도통 구간 i(t), R(T)", "dead time 구간은 channel에서 뺀다" if not v["sr_double"] else "중복: dead time도 channel로 계산"),
        ("sr_diode", "역방향 도통 (SR dead time)", "채널/diode 분담, dead time, V_F", "V_F·i·2t_d·f"),
        ("pri_off", "turn-off (1차, ZVS)", "event별 V/I/T/gate", "ZVS turn-off 데이터"),
        ("hard_eon", "turn-on (hard-switching E_on)", "Eon 안에 recovery·Coss 포함 여부", "ZVS에 hard 데이터 적용 = 오적용" if v["hard_eon"] else "ZVS: 0 (적용 안 함)"),
        ("eoss_added", "Coss 에너지 추가", "Q(V), E(V), 소모 여부", ("ZVS에서 Coss 에너지는 회수 — 추가하면 오적용" + (" / E_off 데이터에 이미 포함 → 중복" if v["eoff_includes_coss"] else "")) if v["add_eoss"] else "추가 안 함 (ZVS 회수)"),
        ("gate", "gate/driver", "Q_g·스윙·f", "8소자, driver 자체 손실 제외"),
        ("copper", "권선·tank 동손", "harmonic RMS, AC R", "DC R만 사용 = 누락" if v["magnetics_dc_only"] else f"AC R (k_ac = {v['k_ac']:g} 포함)"),
        ("core", "코어", "B, f, 재료 자료", "합성 Steinmetz (MISSING_INPUT: 재료)"),
    ]
    arows = []
    for key_, nm, data_, chk in names:
        arows.append([nm, data_, float(pAn[key_][0]), float(pBn[key_][0]), chk])
    arows.append(["합계", "", float(LAn[0]), float(LBn[0]), f"차이 B − A = {dP:.2f} W"])
    res.tables.append(Table("t_accounting", "손실 회계표 (nominal 800 V·11 kW, n = %.2f)" % n, ["항목", "필요한 데이터", "A [W]", "B [W]", "중복/누락 점검"], arows,
                            note="각 소자의 실제 commutation event 기준으로 한 번씩만 센다. hard-switching E_on을 ZVS에 쓰거나 E_oss를 따로 더하면 중복·오적용이다(교재 E01)."))
    issues = []
    if v["add_eoss"]:
        issues.append(f"E_oss 추가 (A {float(pAn['eoss_added'][0]):.1f} W, B {float(pBn['eoss_added'][0]):.1f} W): ZVS에서 Coss 에너지는 소모되지 않음" + (" + E_off 데이터에 이미 포함 → 중복" if v["eoff_includes_coss"] else ""))
    if v["hard_eon"]:
        issues.append(f"hard-switching E_on을 ZVS turn-on에 적용 (A {float(pAn['hard_eon'][0]):.1f} W)")
    if v["sr_double"]:
        issues.append("SR dead time 구간을 channel과 diode로 이중 계산")
    if v["magnetics_dc_only"]:
        issues.append(f"권선 AC 저항 누락 (DC R만, 동손 {float(pAn['copper'][0]):.1f} W로 과소)")
    # --- plots
    tm = np.concatenate([[0.0], np.cumsum(dt_h * 60.0)])
    def steps(y):
        xs, ys = [], []
        for i_ in range(len(y)):
            xs += [float(tm[i_]), float(tm[i_ + 1])]
            ys += [float(y[i_]), float(y[i_])]
        return xs, ys
    x1, y1 = steps(Pm / 1e3)
    res.add_series("m_P", "배터리 전력 P_out", "kW", x1, y1)
    xa, ya = steps(np.where(ok, LA, np.nan))
    xb, yb = steps(np.where(ok, LB_, np.nan))
    res.add_series("m_LA", "A 손실", "W", xa, ya)
    res.add_series("m_LB", "B 손실", "W", xb, yb)
    res.add_plot("p_prof", "합성 충전 mission (CC → CV)", ["m_P"], x_label="시간", x_unit="min", y_label="P_out", y_unit="kW", kind="time", level="A", group="mission",
                 proved="mission은 운전점과 머문 시간의 목록이다. 효율은 이 전체 에너지로 정의한다.", not_yet="합성 충전 프로파일이다. 실제 배터리 SOC·온도·충전기 제어 정책은 반영하지 않았다.")
    res.add_plot("p_mloss", "후보별 손실 전력 (같은 시간축)", ["m_LA", "m_LB"], x_label="시간", x_unit="min", y_label="손실", y_unit="W", kind="time", level="A", group="mission",
                 proved="nominal에서는 A가 작지만 경부하(CV) 구간에서는 스위칭 손실이 큰 A가 불리해진다 — mission 가중이 순위를 바꿀 수 있다.",
                 not_yet="손실은 합성 손실 지도 후처리이며 스위칭 손실 모델의 계통 오차(±)를 포함하지 않은 중앙값이다.")
    res.add_series("eff_A", "A 운전점 효율", "", (Pm / 1e3).tolist(), etaA_k.tolist(), style="points")
    res.add_series("eff_B", "B 운전점 효율", "", (Pm / 1e3).tolist(), etaB_k.tolist(), style="points")
    res.add_plot("p_eff", "운전점 효율과 mission 효율", ["eff_A", "eff_B"], x_label="배터리 전력", x_unit="kW", y_label="η", y_unit="", kind="xy", level="A",
                 hlines=[{"y": eta_mis_A, "label": f"A mission {eta_mis_A:.4%}"}, {"y": eta_ar_A, "label": f"A 산술평균 {eta_ar_A:.4%}"}],
                 vlines=[{"x": float(Pm.min()) / 1e3 - 0.6, "label": ""}, {"x": float(Pm.max()) / 1e3 + 0.6, "label": ""}],  # room for the end points
                 proved="mission 효율은 ΣE_out/ΣE_in이다(E_out 가중 조화평균과 같다는 check). 운전점 효율의 산술평균은 전력·시간 가중을 무시해 다른 값을 준다.", not_yet="회생(역방향)이 있는 mission은 에너지 경계와 부호를 따로 정의해야 한다. 운전점 효율 자체가 합성 손실 지도 기준이라, 실제 효율 곡선(측정)으로 바꾸면 값이 달라진다.")
    rr = np.linspace(-1, 1, 81)
    res.add_series("res_nom", f"nominal: k·u_Δ (k = {k:g})", "W", rr.tolist(), [k * ref.u_difference(u_nom, u_nom, r_) for r_ in rr])
    res.add_plot("p_rank", "8 W 차이를 가릴 수 있는가: 후보 간 오차 상관과 분해능", ["res_nom"], x_label="후보 간 모델 오차 상관 ρ_AB", x_unit="", y_label="분해 가능한 최소 차이", y_unit="W", kind="xy", level="A",
                 hlines=[{"y": abs(dP), "label": f"|Δ| = {abs(dP):.3g} W"}],
                 markers=[{"x": rho, "y": k * u_d, "label": f"판정에 쓴 ρ = {rho:g}"}] + ([{"x": rho_stated, "y": k * ref.u_difference(u_nom, u_nom, rho_stated), "label": f"근거 없는 ρ = {rho_stated:g}"}] if rho_stated != rho else []),
                 proved="두 후보의 스위칭 손실을 같은 방법·같은 fixture로 얻어 공통 오차가 거의 완전히 상쇄될 때만 8 W를 가릴 수 있다. mission 차이의 불확도 전파식은 seed 고정 Monte Carlo check와 일치한다.",
                 not_yet="ρ_AB의 근거(같은 DPT setup, 교대 측정)는 기록해야 한다. 근거 없는 상관으로 error bar를 줄이지 않는다.")
    res.circuit = {"diagram": cllc_circuit(base, n).to_json(), "intervals": [], "plot_group": ""}
    # --- independent checks
    eta_h = ref.mission_efficiency_harmonic(list(etaA_k[ok]), list(Eout[ok]))
    res.add_check(check_close("mission 효율: 에너지 합 vs E_out 가중 조화평균", eta_mis_A, eta_h, 1e-12, "ΣE_out/(ΣE_out + ΣE_loss) vs ΣE_out/Σ(E_out/η_k) (reference)", True, ""))
    rng = np.random.default_rng(int(v["seed"]))
    N = int(v["mc_n"])
    z = rng.standard_normal((N, 2))
    zz = z @ np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]) + 1e-15 * np.eye(2)).T
    dE_s = (EB_loss + UB * zz[:, 1]) - (EA_loss + UA * zz[:, 0])
    u_mc = float(np.std(dE_s, ddof=1))
    se = 1 / math.sqrt(2 * (N - 1))
    res.add_check(check_close(f"mission 차이 불확도: Monte Carlo (N = {N}, seed {int(v['seed'])}) vs 전파식", u_mc, uE, 5 * se + 1e-12, "상관 Gaussian 모델 오차 표본의 ΔE 표준편차 vs √(U_A² + U_B² − 2ρU_AU_B)", True, "Wh"))
    e_pw = 0.0
    for i_ in np.where(eA["root"])[0]:
        Hh, _, Rac = cllc_net(float(eA["f"][i_]), n, float(Vb[i_]), float(Pm[i_]), base["Lr"], base["Cr"], base["Lm"])
        e_pw = max(e_pw, abs((abs(Hh) * float(eA["V1"][i_])) ** 2 / Rac - Pm[i_]) / Pm[i_])
    res.add_check(Check("mission 운전점: FHA 출력 전력 = 지정 전력", "PASS" if e_pw < 1e-9 else "FAIL", e_pw, "rel", 1e-9, path="각 운전점의 |V_o′|²/R_ac′ vs P (근 찾기 결과의 에너지 확인)", independent=True))
    res.add_check(check_close("시나리오 재현: 도출한 R_DS(on)으로 nominal 차이", float(loss_A(R_A)), target, 1e-9, "brentq로 푼 R_A를 다시 넣은 손실 (회귀)", False, "W"))
    # --- verdicts
    if issues:
        res.verdict("OUT_OF_VALIDITY", "손실 회계 오류: " + "; ".join(issues) + " — 이 합계로 순위를 매기지 않는다")
    if rho_stated != 0 and v["rho_basis"] == "none":
        res.verdict("MISSING_INPUT", f"ρ_AB = {rho_stated:g}로 줄어드는 불확도({ref.u_difference(u_nom, u_nom, rho_stated):.3g} W)는 상관의 근거가 기록되지 않아 주장할 수 없다 — 판정은 독립(ρ = 0)으로 했다")
    if not resolved:
        res.verdict("UNRESOLVED_RANKING", f"nominal 차이 {abs(dP):.3g} W ≤ k·u_Δ = {k * u_d:.3g} W: 모델만으로 승자를 확정하지 않는다 — 분리 가능한 DPT/열 시험, 공통 오차 상관, mission 가중, 정격·공급 등 다른 판단축")
    if not bool(np.all(evaluable)):
        res.verdict("OUT_OF_VALIDITY", f"mission 운전점 {int(np.sum(~evaluable))}개가 FHA 연속 운전 범위 밖(P < P_burst {v['P_burst'] / 1e3:g} kW 또는 손실 지도 외삽): 그 구간 손실은 UNKNOWN — mission 순위는 평가 가능한 구간만")
    if not resE and bool(np.all(evaluable)):
        res.verdict("UNRESOLVED_RANKING", f"mission 손실 에너지 차 {abs(dE):.3g} Wh ≤ k·u = {k * uE:.3g} Wh")
    if not res.verdicts:
        res.verdict("PASS_WITHIN_MODEL", f"nominal·mission 모두 {winner}가 불확도보다 큰 차이로 앞선다 (합성 손실 지도 기준)")
    res.assumptions += [
        f"설계 n = {n:g} (obc_cllc의 robust 선택), 오른쪽 분기 FHA 운전점, 합성 손실 지도",
        f"A: turn-off 에너지 ×{v['sw_scale_A']:g}, R_DS(on)은 ‘nominal에서 {v['dP_nom']:g} W 좋다’ 시나리오로 도출",
        "스위칭 손실 모델 오차는 후보마다 계통적(운전점 간 완전 상관), 후보 간 상관 ρ_AB",
        "mission = 합성 CC-CV 충전 (P_cc, 900 V CV taper); P < P_burst 구간은 burst 운전으로 FHA 밖",
    ]
    res.not_valid_for += ["실제 소자 비교 (DATASHEET 조건·DPT 필요: MISSING_INPUT)", "burst·경부하 손실", "회생·역방향 mission", "비용 비교 (cost 자료 없음 — 만들어 넣지 않음)"]
    res.interpretation = (
        f"A는 nominal에서 {abs(dP):.3g} W 좋지만 두 후보의 스위칭 손실 모델이 각각 ±{v['u_sw']:g} W 틀릴 수 있으면 차이의 불확도는 {u_d:.3g} W라 순위를 확정할 수 없다. "
        "같은 fixture로 두 후보를 교대 측정해 공통 오차를 상쇄하거나, 분리 가능한 DPT·열 시험으로 차이를 직접 재야 한다. "
        "mission으로 보면 경부하 구간이 길수록 스위칭 손실이 작은 후보가 유리해지고, 효율은 운전점 효율의 평균이 아니라 총출력에너지/총입력에너지로 정의한다."
    )
    return res


# ======================================================================================
# Lab definition
# ======================================================================================

P_LOSS_SURFACE = [
    Param("query_V", "질의점 전압", "V", 700.0, "V", vmin=1.0, vmax=2000.0, source="ASSUMED", source_note="순위를 묻는 운전점", group="질의점"),
    Param("query_I", "질의점 전류", "A", 60.0, "A", vmin=0.01, vmax=2000.0, source="ASSUMED", group="질의점"),
    Param("rank_model", "순위에 쓸 fit 모델", "", "floor", kind="choice", choices=[("floor", "Coss floor + power law"), ("power", "순수 power law (floor 없음)")], source="ASSUMED", group="질의점"),
    Param("I_lo_char", "특성화 전류 하한", "A", 10.0, "A", vmin=0.5, vmax=79.0, source="ASSUMED", source_note="측정 격자: 하한~160 A 로그 6점 × 400/600/800 V", group="특성화"),
    Param("noise", "측정 상대 잡음 (1σ)", "", 0.03, "", vmin=0.0, vmax=0.3, source="ASSUMED", group="특성화"),
    Param("Eoss_A", "A: Coss floor E_oss @ 800 V (합성 참값)", "J", 0.45e-3, "mJ", vmin=0.0, vmax=0.1, source="ASSUMED", group="합성 참값"),
    Param("Eoss_B", "B: Coss floor E_oss @ 800 V (합성 참값)", "J", 0.20e-3, "mJ", vmin=0.0, vmax=0.1, source="ASSUMED", group="합성 참값"),
    Param("tol_val", "검증 허용 오차 (최대 상대)", "", 0.10, "", vmin=0.001, vmax=1.0, source="ASSUMED", group="판정"),
    Param("k_cov", "순위 판정 coverage factor k", "", 2.0, "", vmin=0.5, vmax=5.0, source="ASSUMED", group="판정"),
    Param("mc_refits", "불확도 검산 Monte Carlo 재fit 횟수", "", 60, "", vmin=10, vmax=2000, kind="int", source="ASSUMED", group="판정"),
    Param("seed", "난수 seed", "", 5, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group="판정"),
]


P_CLLC = [
    Param("n", "후보 turns ratio n = N_p/N_s", "", 1.0, "", vmin=0.7, vmax=1.3, source="TEXTBOOK", source_note="seed n = 1 (교재 초기안); 수정 후보 0.93", group="후보 설계"),
    Param("branch", "FHA 해 분기 선택", "", "right", kind="choice", choices=[("right", "오른쪽 분기 (gain 기울기 음, 기본)"), ("min_current", "두 해 중 1차 RMS 최소")], source="ASSUMED", group="후보 설계"),
    Param("Lr", "L_r (1차, = L_r2′)", "H", 40e-6, "uH", vmin=1e-6, vmax=1e-3, source="TEXTBOOK", group="tank"),
    Param("Cr", "C_r (1차, = C_r2′)", "F", 28.144773e-9, "nF", vmin=1e-9, vmax=1e-6, source="TEXTBOOK", group="tank"),
    Param("Lm", "L_m", "H", 200e-6, "uH", vmin=10e-6, vmax=10e-3, source="TEXTBOOK", group="tank"),
    Param("f_min", "주파수 하한", "Hz", 120e3, "kHz", vmin=10e3, vmax=1e6, source="TEXTBOOK", group="tank"),
    Param("f_max", "주파수 상한", "Hz", 210e3, "kHz", vmin=10e3, vmax=1e6, source="TEXTBOOK", group="tank"),
    Param("P_rated", "요구 배터리 전력", "W", 11000.0, "kW", vmin=100.0, vmax=50e3, source="TEXTBOOK", group="요구 corner"),
    Param("P_burst", "burst 허용 하한 (이하 FHA 판정 제외)", "W", 2200.0, "kW", vmin=50.0, vmax=11000.0, source="ASSUMED", source_note="20 %; 경부하 연속 regulation 요구는 고객 확인 필요", group="요구 corner"),
    Param("Vlink_650", "배터리 650 V일 때 link", "V", 700.0, "V", vmin=300.0, vmax=1200.0, source="TEXTBOOK", group="link schedule"),
    Param("Vlink_800", "배터리 800 V일 때 link", "V", 800.0, "V", vmin=300.0, vmax=1200.0, source="TEXTBOOK", group="link schedule"),
    Param("Vlink_920", "배터리 920 V일 때 link", "V", 850.0, "V", vmin=300.0, vmax=1200.0, source="TEXTBOOK", group="link schedule"),
    Param("V_LL", "계통 선간전압", "V", 400.0, "V", vmin=100.0, vmax=1000.0, source="TEXTBOOK", group="입력 제한"),
    Param("I_line", "입력 전류 제한", "A", 16.0, "A", vmin=1.0, vmax=200.0, source="TEXTBOOK", group="입력 제한"),
    Param("PF", "역률", "", 0.995, "", vmin=0.5, vmax=1.0, source="TEXTBOOK", group="입력 제한"),
    Param("eta_sys", "전체 효율 가정", "", 0.97, "", vmin=0.5, vmax=1.0, source="TEXTBOOK", group="입력 제한"),
    Param("I1_max", "1차 RMS 전류 한계", "A", 22.0, "A", vmin=1.0, vmax=500.0, source="ASSUMED", group="제약"),
    Param("B_max", "자속밀도 한계", "T", 0.20, "T", vmin=0.01, vmax=2.0, source="ASSUMED", group="제약"),
    Param("Np", "1차 권선 수", "", 20, "", vmin=1, vmax=500, kind="int", source="ASSUMED", group="제약"),
    Param("Ae", "코어 단면적", "mm²", 400.0, "mm²", vmin=1.0, vmax=1e5, source="ASSUMED", group="제약"),
    Param("Qoss", "스위치당 Q_oss (link 전압)", "C", 150e-9, "nC", vmin=1e-9, vmax=1e-5, source="ASSUMED", group="제약"),
    Param("t_dead", "dead time", "s", 150e-9, "ns", vmin=10e-9, vmax=2e-6, source="ASSUMED", group="제약"),
    Param("Tj_max", "T_j 설계 한계", "°C", 150.0, "°C", vmin=50.0, vmax=250.0, source="ASSUMED", group="제약"),
    Param("T_cool", "냉각수 온도", "°C", 65.0, "°C", vmin=-40.0, vmax=120.0, source="ASSUMED", group="제약"),
    Param("Rds", "R_DS(on) @ 25 °C (1차·SR 동일)", "Ω", 32e-3, "mΩ", vmin=1e-4, vmax=1.0, source="ASSUMED", group="합성 손실 지도"),
    Param("tc", "R_DS(on) 온도계수", "1/K", 0.004, "1/K", vmin=0.0, vmax=0.02, source="ASSUMED", group="합성 손실 지도"),
    Param("Rth", "R_th (접합–냉각수, 소자당)", "K/W", 1.0, "K/W", vmin=0.01, vmax=20.0, source="ASSUMED", group="합성 손실 지도"),
    Param("Eoff_20A", "turn-off 에너지 @ 800 V·20 A (ZVS)", "J", 40e-6, "uJ", vmin=1e-7, vmax=1e-2, source="ASSUMED", source_note="E ∝ I^1.3·V; 특성화 I_lo~40 A, 400~900 V", group="합성 손실 지도"),
    Param("Ioff_lo_char", "turn-off 지도 전류 하한", "A", 2.0, "A", vmin=0.1, vmax=39.0, source="ASSUMED", group="합성 손실 지도"),
    Param("R_pri", "1차 권선+tank AC 저항", "Ω", 70e-3, "mΩ", vmin=0.0, vmax=10.0, source="ASSUMED", group="합성 손실 지도"),
    Param("R_sec", "2차 권선+tank AC 저항 (실제 2차)", "Ω", 70e-3, "mΩ", vmin=0.0, vmax=10.0, source="ASSUMED", group="합성 손실 지도"),
    Param("k_core", "코어 손실 계수 (100 kHz·0.1 T 기준)", "W", 8.0, "W", vmin=0.0, vmax=1000.0, source="ASSUMED", source_note="P = k(f/100k)^1.4(B/0.1)^2.5 합성 Steinmetz", group="합성 손실 지도"),
    Param("Qg", "gate 전하", "C", 80e-9, "nC", vmin=0.0, vmax=1e-5, source="ASSUMED", group="합성 손실 지도"),
    Param("dVg", "gate 전압 스윙", "V", 20.0, "V", vmin=0.0, vmax=50.0, source="ASSUMED", group="합성 손실 지도"),
]

P_DAB = [
    Param("L", "후보 L (1차 환산 직렬 인덕턴스)", "H", 200e-6, "uH", vmin=10e-6, vmax=5e-3, source="TEXTBOOK", source_note="교재 200 µH", group="후보 설계"),
    Param("n_tr", "turns ratio n", "", 50 / 3, "", vmin=1.0, vmax=100.0, source="TEXTBOOK", source_note="50/3", group="DAB"),
    Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=1e6, source="TEXTBOOK", group="DAB"),
    Param("P_mod", "모듈 정격 전력", "W", 1500.0, "kW", vmin=10.0, vmax=50e3, source="TEXTBOOK", source_note="1.5 kW/모듈", group="DAB"),
    Param("N_mod", "모듈 수", "", 2, "", vmin=1, vmax=16, kind="int", source="TEXTBOOK", source_note="합계 3 kW", group="DAB"),
    Param("VH_nom", "nominal HV", "V", 800.0, "V", vmin=100.0, vmax=1500.0, source="TEXTBOOK", group="요구 corner"),
    Param("VL_nom", "nominal LV", "V", 48.0, "V", vmin=10.0, vmax=100.0, source="TEXTBOOK", group="요구 corner"),
    Param("VH_map", "지도에 쓸 HV 전압", "V", 900.0, "V", vmin=100.0, vmax=1500.0, source="TEXTBOOK", source_note="저전압 LV corner의 V_H", group="요구 corner"),
    Param("phi_max_deg", "허용 최대 φ", "deg", 60.0, "deg", vmin=1.0, vmax=90.0, source="ASSUMED", group="제약"),
    Param("Irms_max", "1차 RMS 허용치", "A", 4.0, "A", vmin=0.1, vmax=200.0, source="ASSUMED", group="제약"),
    Param("Qoss_hv", "HV 소자 Q_oss", "C", 100e-9, "nC", vmin=1e-9, vmax=1e-5, source="ASSUMED", group="제약"),
    Param("Qoss_lv", "LV 위치 Q_oss", "C", 300e-9, "nC", vmin=1e-9, vmax=1e-4, source="ASSUMED", group="제약"),
    Param("t_dead", "dead time", "s", 150e-9, "ns", vmin=10e-9, vmax=2e-6, source="ASSUMED", group="제약"),
    Param("Tj_max", "T_j 설계 한계", "°C", 150.0, "°C", vmin=50.0, vmax=250.0, source="ASSUMED", group="제약"),
    Param("T_cool", "냉각수 온도", "°C", 65.0, "°C", vmin=-40.0, vmax=120.0, source="ASSUMED", group="제약"),
    Param("R_hv", "HV R_DS(on) @ 25 °C", "Ω", 32e-3, "mΩ", vmin=1e-4, vmax=1.0, source="ASSUMED", group="합성 손실 지도"),
    Param("R_lv", "LV 위치 R_DS(on) @ 25 °C (병렬 포함)", "Ω", 2.5e-3, "mΩ", vmin=1e-5, vmax=0.1, source="ASSUMED", group="합성 손실 지도"),
    Param("tc", "R_DS(on) 온도계수", "1/K", 0.004, "1/K", vmin=0.0, vmax=0.02, source="ASSUMED", group="합성 손실 지도"),
    Param("Rth_hv", "HV R_th", "K/W", 1.0, "K/W", vmin=0.01, vmax=20.0, source="ASSUMED", group="합성 손실 지도"),
    Param("Rth_lv", "LV R_th", "K/W", 1.5, "K/W", vmin=0.01, vmax=20.0, source="ASSUMED", group="합성 손실 지도"),
    Param("Eoss_hv", "HV hard turn-on의 Coss floor @ 800 V", "J", 60e-6, "uJ", vmin=0.0, vmax=1e-2, source="ASSUMED", group="합성 손실 지도"),
    Param("R_tr", "transformer AC 저항 (1차 환산, 2차 동손 포함)", "Ω", 1.2, "Ω", vmin=0.0, vmax=10.0, source="ASSUMED", group="합성 손실 지도"),
]

P_LEG = [
    Param("Rg", "후보 외부 gate 저항 R_g", "Ω", 4.7, "Ω", vmin=0.1, vmax=100.0, source="ASSUMED", source_note="후보군 1–10 Ω", group="후보 설계"),
    Param("kelvin", "Kelvin source gate 연결", "", False, kind="bool", source="ASSUMED", group="후보 설계"),
    Param("R1", "branch 1 R_DS(on) @ 25 °C", "Ω", 3.6e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", source_note="교재 3.6/4/4/4.4 mΩ", group="branch"),
    Param("R2", "branch 2 R_DS(on)", "Ω", 4.0e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", group="branch"),
    Param("R3", "branch 3 R_DS(on)", "Ω", 4.0e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", group="branch"),
    Param("R4", "branch 4 R_DS(on)", "Ω", 4.4e-3, "mΩ", vmin=1e-4, vmax=0.1, source="TEXTBOOK", group="branch"),
    Param("I_nom", "nominal leg 전류 I_pk (4 branch 합)", "A", 400.0, "A", vmin=1.0, vmax=5000.0, source="TEXTBOOK", source_note="교재 total 400 A", group="요구 corner"),
    Param("I_max", "최대 leg 전류 I_pk", "A", 720.0, "A", vmin=1.0, vmax=5000.0, source="ASSUMED", group="요구 corner"),
    Param("V_bus_max", "최대 bus 전압", "V", 900.0, "V", vmin=100.0, vmax=1500.0, source="ASSUMED", group="요구 corner"),
    Param("T_cool", "냉각수 온도 (nominal)", "°C", 65.0, "°C", vmin=-40.0, vmax=120.0, source="ASSUMED", group="요구 corner"),
    Param("T_cool_hot", "고온 냉각수 corner", "°C", 85.0, "°C", vmin=-40.0, vmax=150.0, source="ASSUMED", group="요구 corner"),
    Param("S_ref", "기준 branch di/dt (R_ref에서)", "A/us", 2000.0, "kA/us", vmin=1.0, vmax=1e5, source="TEXTBOOK", source_note="교재 2 kA/µs", group="gate·루프"),
    Param("R_ref", "di/dt 기준 R_g", "Ω", 4.7, "Ω", vmin=0.1, vmax=100.0, source="ASSUMED", group="gate·루프"),
    Param("R_int", "내부 gate 저항", "Ω", 1.0, "Ω", vmin=0.0, vmax=20.0, source="ASSUMED", group="gate·루프"),
    Param("L_cs", "공통 source 인덕턴스 (전력 source 기준 gate)", "H", 2e-9, "nH", vmin=0.0, vmax=1e-7, source="TEXTBOOK", source_note="교재 2 nH", group="gate·루프"),
    Param("L_cs_kelvin", "Kelvin 연결 시 잔여 결합 L", "H", 0.2e-9, "nH", vmin=0.0, vmax=1e-7, source="ASSUMED", group="gate·루프"),
    Param("Vcs_max", "gate 결합 전압 허용치 (screen)", "V", 5.0, "V", vmin=0.1, vmax=30.0, source="ASSUMED", group="gate·루프"),
    Param("L_loop", "전력 루프 인덕턴스", "H", 12e-9, "nH", vmin=0.0, vmax=1e-6, source="ASSUMED", group="gate·루프"),
    Param("V_lim", "과전압 허용 V_DS peak (derating)", "V", 1000.0, "V", vmin=100.0, vmax=2000.0, source="ASSUMED", group="gate·루프"),
    Param("skew", "gate timing skew", "s", 10e-9, "ns", vmin=0.0, vmax=1e-6, source="TEXTBOOK", source_note="교재 10 ns", group="gate·루프"),
    Param("tc", "R_DS(on) 온도계수", "1/K", 0.005, "1/K", vmin=0.0, vmax=0.02, source="ASSUMED", group="합성 손실 지도"),
    Param("Rth", "R_th (die–냉각수, branch 동일)", "K/W", 0.8, "K/W", vmin=0.01, vmax=20.0, source="ASSUMED", group="합성 손실 지도"),
    Param("fsw", "스위칭 주파수", "Hz", 16e3, "kHz", vmin=1e3, vmax=200e3, source="ASSUMED", group="합성 손실 지도"),
    Param("Esw_ref", "E_on+E_off @ 800 V·100 A (R_ref)", "J", 4e-3, "mJ", vmin=1e-6, vmax=1.0, source="ASSUMED", source_note="E ∝ I^1.15 V^1.35 + Coss floor", group="합성 손실 지도"),
    Param("Eoss_leg", "Coss floor @ 800 V", "J", 0.4e-3, "mJ", vmin=0.0, vmax=0.1, source="ASSUMED", group="합성 손실 지도"),
    Param("I_lo_char", "손실 지도 전류 하한 (branch)", "A", 10.0, "A", vmin=0.1, vmax=200.0, source="ASSUMED", group="합성 손실 지도"),
    Param("frac_ex_max", "허용 외삽 에너지 비율", "", 0.25, "", vmin=0.0, vmax=1.0, source="ASSUMED", source_note="넘으면 UNKNOWN", group="합성 손실 지도"),
    Param("Tj_max", "T_j 설계 한계", "°C", 150.0, "°C", vmin=50.0, vmax=250.0, source="ASSUMED", group="합성 손실 지도"),
    Param("I_br_max", "branch 전류 한계 (peak)", "A", 250.0, "A", vmin=1.0, vmax=2000.0, source="ASSUMED", group="합성 손실 지도"),
]

P_MISSION = [
    Param("dP_nom", "시나리오: nominal에서 A가 작은 손실", "W", 8.0, "W", vmin=0.0, vmax=500.0, source="TEXTBOOK", source_note="교재 E01 ‘A가 nominal에서 8 W 좋다’", group="시나리오"),
    Param("u_sw", "스위칭 손실 모델 불확실성 ±", "W", 15.0, "W", vmin=0.0, vmax=500.0, source="TEXTBOOK", source_note="교재 ±15 W", group="시나리오"),
    Param("u_kind", "±값의 의미", "", "standard", kind="choice", choices=[("standard", "표준불확도 1σ"), ("rect_limit", "±한계 (직사각)"), ("k2", "확장불확도 k = 2")], source="ASSUMED", group="시나리오"),
    Param("rho_AB", "후보 간 모델 오차 상관 ρ_AB", "", 0.0, "", vmin=-1.0, vmax=1.0, source="ASSUMED", group="시나리오"),
    Param("rho_basis", "상관의 근거", "", "none", kind="choice", choices=[("none", "근거 기록 없음"), ("same_fixture", "같은 DPT fixture·probe·보정")], source="ASSUMED", group="시나리오"),
    Param("k_cov", "판정 coverage factor k", "", 2.0, "", vmin=0.5, vmax=5.0, source="ASSUMED", group="시나리오"),
    Param("sw_scale_A", "A의 turn-off 에너지 배율 (B 대비)", "", 1.35, "", vmin=0.1, vmax=10.0, source="ASSUMED", source_note="A는 낮은 R_DS(on)·큰 스위칭 에너지", group="후보"),
    Param("Rds_B", "B의 R_DS(on) @ 25 °C", "Ω", 32e-3, "mΩ", vmin=1e-4, vmax=1.0, source="ASSUMED", group="후보"),
    Param("Eoff_B", "B의 turn-off 에너지 @ 800 V·20 A", "J", 40e-6, "uJ", vmin=1e-7, vmax=1e-2, source="ASSUMED", group="후보"),
    Param("Eoss_B", "B의 E_oss @ 800 V (회계 점검용)", "J", 30e-6, "uJ", vmin=0.0, vmax=1e-2, source="ASSUMED", source_note="A = B × 배율", group="후보"),
    Param("n_design", "설계 turns ratio", "", 0.93, "", vmin=0.7, vmax=1.3, source="DERIVED", source_note="obc_cllc의 robust 선택", group="mission"),
    Param("profile", "mission", "", "cc_cv", kind="choice", choices=[("cc_cv", "CC 10.5 kW → 900 V CV (2.4 kW까지)"), ("deep_taper", "CV를 0.6 kW까지 (burst 구간 포함)"), ("cc_only", "CC 구간만")], source="ASSUMED", group="mission"),
    Param("P_cc", "CC 구간 배터리 전력", "W", 10500.0, "kW", vmin=500.0, vmax=11000.0, source="ASSUMED", source_note="입력 제한 10.7 kW 아래", group="mission"),
    Param("P_burst", "burst 허용 하한", "W", 2200.0, "kW", vmin=50.0, vmax=11000.0, source="ASSUMED", group="mission"),
    Param("add_eoss", "E_oss를 turn-on 손실로 추가", "", False, kind="bool", source="ASSUMED", group="손실 회계"),
    Param("eoff_includes_coss", "E_off 데이터에 Coss 충전 포함", "", True, kind="bool", source="ASSUMED", source_note="데이터 조건 확인 항목", group="손실 회계"),
    Param("hard_eon", "hard-switching E_on을 ZVS turn-on에 적용", "", False, kind="bool", source="ASSUMED", group="손실 회계"),
    Param("sr_double", "SR dead time을 channel과 diode로 이중 계산", "", False, kind="bool", source="ASSUMED", group="손실 회계"),
    Param("magnetics_dc_only", "권선 손실에 DC 저항만 사용", "", False, kind="bool", source="ASSUMED", group="손실 회계"),
    Param("k_ac", "권선 AC/DC 저항비", "", 2.5, "", vmin=1.0, vmax=20.0, source="ASSUMED", group="손실 회계"),
    Param("V_F", "SR body diode 전압", "V", 3.0, "V", vmin=0.1, vmax=10.0, source="ASSUMED", group="손실 회계"),
    Param("t_dead", "dead time", "s", 150e-9, "ns", vmin=10e-9, vmax=2e-6, source="ASSUMED", group="손실 회계"),
    Param("mc_n", "Monte Carlo 표본 수", "", 20000, "", vmin=1000, vmax=2_000_000, kind="int", source="ASSUMED", group="검산"),
    Param("seed", "난수 seed", "", 17, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group="검산"),
]

_Q = [
    Question(
        "후보 A가 nominal에서 8 W 좋지만 switching-loss 모델의 불확실성이 ±15 W라면?",
        "모델만으로 승자를 확정하지 않는다. 두 후보의 모델 오차가 독립이면 차이의 불확도는 √(15² + 15²) ≈ 21 W라 8 W 차이는 UNRESOLVED_RANKING이다. 같은 DPT fixture·probe·보정으로 두 후보를 교대 측정해 공통 오차를 상쇄하거나(그 상관의 근거를 기록한다), 분리 가능한 열 시험으로 차이를 직접 재고, mission 에너지 가중에서의 영향을 보며, 정격·공급·qualification 같은 다른 판단축으로 결정한다. 더 정밀한 그래프가 더 정확한 근거는 아니다.",
        "Candidate A is 8 W better at nominal, but the switching-loss model is uncertain by plus or minus 15 W. Which do you choose?",
        "I do not declare a winner from the model alone. If the two model errors are independent, the uncertainty of the difference is about 21 W, so an 8 W difference is unresolved. I would measure both devices alternately on the same double-pulse fixture to cancel common errors, or compare them in a separable thermal test, weigh the effect over the mission energy, and decide with other criteria such as ratings and supply. A finer plot is not stronger evidence.",
        ["UNRESOLVED_RANKING", "√(15² + 15²) ≈ 21 W", "같은 fixture 교대 측정", "mission 가중"],
        kind="pressure",
    ),
    Question(
        "이 소자를 추천하면 전체 사양을 만족하는가?",
        "정격 전압·전류·T_j를 하나씩 통과해도 모든 조건을 동시에 통과한 것은 아니다. 운전조건 u(V_in, V_out, P, T_cool, f, mode), 설계 d, 불확실성 θ를 나누고, 요구 corner마다 모든 제약 g_j(u, d, θ) ≤ 0을 동시에 본다: 높은 배터리전압·경부하는 gain·ZVS, 낮은 입력·전부하는 RMS·자성체·도통, 고온 냉각수는 전열. 정격 밖 후보는 효율 순위표에서 빼고, 근거가 다른 loss map끼리 직접 순위화하지 않는다.",
        "If you recommend this device, does it meet the whole specification?",
        "Passing each rating one at a time is not passing them simultaneously. I separate operating conditions, design variables and uncertainties, and check every constraint at every required corner: gain and ZVS at high battery voltage and light load, RMS current and conduction at low input and full load, and thermal limits with hot coolant. A candidate outside a rating leaves the efficiency ranking, and loss maps built on different evidence are not ranked against each other directly.",
        ["동시 제약", "요구 corner", "u·d·θ 분리", "정격 밖 후보 제외"],
    ),
    Question(
        "경쟁사는 11 kW라고 하는데요?",
        "정격의 입력전압·전류·PF·효율·출력 경계와 derating 조건을 맞춰 비교한다. 우리 가정(400 V·16 A·PF 0.995·η 0.97)에서는 배터리 쪽 10.6988 kW가 한계라 11 kW보다 약 301 W 부족하다. 상대가 틀렸다고 단정하거나 우리 가정을 숨겨 비교하지 않고, 입력전류 요구와 derating을 고객과 합의한다(CUSTOMER_DECISION_REQUIRED).",
        "A competitor claims 11 kW. Why can't you?",
        "I compare ratings under the same input voltage, current, power factor, efficiency, output boundary and derating. Under our assumptions, 400 V, 16 A, 0.995 and 97 percent, the battery side is limited to about 10.7 kW, roughly 301 W short. I would not claim the competitor is wrong or hide our assumptions; the input current requirement and derating need to be agreed with the customer.",
        ["조건을 맞춰 비교", "10.6988 kW", "derating 합의", "단정하지 않음"],
        kind="pressure",
    ),
    Question(
        "전달 전력이 0인데 왜 전류가 흐르나요?",
        "전력 0은 평균 port 에너지 전달이 0이라는 뜻이다. 900 V와 nV_L 600 V처럼 두 구형파 전압이 같지 않으면 누설 인덕턴스에 교번전압이 걸려 전류가 흐른다(L = 200 µH에서 I_rms 2.165 A). 이 전류는 동손·채널 손실을 만들고 commutation에는 도움이 될 수도 있다. ‘일을 안 하니 전류도 0’이라는 직관이 틀리는 지점이다.",
        "The transferred power is zero. Why is there current?",
        "Zero power means zero average energy transfer between the ports. When the two square-wave voltages differ, for example 900 V against 600 V referred, an alternating voltage sits across the leakage inductance and current flows, about 2.17 A RMS with 200 microhenries. It costs copper and channel loss, and it may help commutation.",
        ["평균 전력 0 ≠ 전류 0", "전압비 불일치", "2.165 A", "손실과 ZVS 양면"],
    ),
    Question(
        "4개 병렬 branch 중 하나만 뜨겁습니다. die 불량인가요?",
        "뜨거운 branch 하나만으로 불량을 확정하지 않는다. 정적 분배(3.6/4/4/4.4 mΩ → 110.55/99.50/99.50/90.45 A)만으로도 R이 작은 branch가 더 뜨겁다. gate timing skew(10 ns × 2 kA/µs ≈ 20 A scale), 공통 source 결합(2 nH × 2 kA/µs = 4 V), 열 계면, 센서·deskew 오류를 가설로 나누고, branch 전류와 local V_gs/V_ds를 같은 timebase로 측정한다. gate 측정 기준점(전력 source vs Kelvin)을 먼저 확인한다.",
        "One of four parallel branches runs hot. Is the die defective?",
        "One hot branch does not prove a defect. Static sharing alone, 3.6, 4, 4 and 4.4 milliohms, gives 110.6, 99.5, 99.5 and 90.5 A, so the lowest-resistance branch runs hottest. I separate hypotheses, gate timing skew, common-source coupling, thermal interface and sensor or deskew errors, and measure branch currents and local gate and drain voltages on one timebase, after checking whether the gate is referenced to the power source or a Kelvin source.",
        ["정적 분배 110.55 A", "skew 20 A scale", "공통 source 4 V", "가설별 판별 측정"],
    ),
    Question(
        "drive-cycle(또는 충전) 효율을 운전점 효율의 평균으로 말해도 되나?",
        "안 된다. E_loss = ΣP_loss,k·Δt_k이고 효율은 ΣE_out/ΣE_in이다(E_out 가중 조화평균과 같다). 산술평균이나 시간평균은 전력·시간 가중을 틀리게 준다. 회생이 있으면 모드별 에너지 경계와 부호를 따로 정의하고, 인버터 효율 개선을 바로 차량 에너지 이익으로 옮기지 않는다.",
        "Can you quote a drive-cycle efficiency as the average of operating-point efficiencies?",
        "No. The loss energy is the sum of loss power times duration, and the efficiency is total output energy over total input energy, which equals the output-energy-weighted harmonic mean. An arithmetic or time average weights the points wrongly. With regeneration, the energy boundary and sign of each mode must be defined separately.",
        ["ΣE_out/ΣE_in", "E_loss = ΣP·Δt", "평균 효율 금지", "회생 경계"],
    ),
    Question(
        "hard-switching E_on 데이터로 CLLC 손실을 계산해도 되나?",
        "안 된다. CLLC는 ZVS turn-on이라 hard-switching E_on을 쓰면 오적용이고, Coss 에너지(E_oss)를 따로 더하면 회수되는 에너지를 손실로 센다. E_on 데이터에 상대 소자 recovery가 들어 있는지, E_off에 Coss 충전이 들어 있는지 조건을 확인하고, 소자별 실제 commutation event로 손실 회계표를 만들어 중복·누락을 점검한다.",
        "Can you compute CLLC losses with hard-switching E_on data?",
        "No. A CLLC turns on with zero-voltage switching, so hard-switching turn-on energy does not apply, and adding the Coss energy separately counts energy that is recovered. I check whether the turn-on data include the opposite device's recovery and whether the turn-off data include Coss charging, and I build the loss table from each device's actual commutation events to catch double counting and omissions.",
        ["ZVS에 hard E_on 금지", "E_oss 중복", "데이터 포함 범위", "event 기준 회계"],
    ),
    Question(
        "데이터시트 E_on 몇 점으로 경부하 효율을 비교해도 되나?",
        "fit 곡면은 측정 범위 안에서만 보간이다. α·β는 관습값이 아니라 fit 결과로 불확도와 함께 적고, fit에 쓰지 않은 검증점으로 확인한다. I → 0의 Coss floor가 없으면 경부하 비교가 뒤집힐 수 있고, 측정 범위 밖 질의점은 UNKNOWN으로 두고 순위를 매기지 않는다.",
        "Can you compare light-load efficiency from a few datasheet E_on points?",
        "A fitted surface is interpolation inside the measured range only. The exponents are fit results reported with their uncertainty and checked on points not used in the fit. Without the Coss floor at zero current the light-load comparison can flip, and a query outside the measured range is reported as unknown and not ranked.",
        ["보간 범위", "fit/검증 분리", "Coss floor", "외삽 UNKNOWN"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="loss_surface",
        title="데이터시트 점에서 loss surface로: fit과 검증을 나누고, 외삽은 순위에서 뺀다",
        goal=(
            "합성 소자 A·B의 ‘측정점’(400/600/800 V × 10~160 A, 잡음 3 %)으로 스위칭 에너지 곡면 E(V, I)를 fit한다. α·β는 관습값이 아니라 fit 결과이고, 600 V 점은 fit에 쓰지 않고 검증에만 쓴다. "
            "I → 0에서 남는 Coss floor를 뺀 순수 power law는 저전류에서 검증에 실패한다. 특성화 범위 밖 질의점은 값만 보이고 UNKNOWN(OUT_OF_VALIDITY)으로 두며, 차이가 fit 불확도보다 작으면 UNRESOLVED_RANKING이다."
        ),
        params=P_LOSS_SURFACE,
        presets=[
            Preset("nominal", "700 V·60 A 질의 (보간)", {}, "A가 작다 (불확도 밖)", ("nominal", "reference")),
            Preset("light", "800 V·4 A 질의 (측정 하한 아래)", {"query_V": 800.0, "query_I": 4.0}, "외삽 → UNKNOWN", ("failure", "reference")),
            Preset("overvolt", "900 V·100 A 질의 (전압 외삽)", {"query_V": 900.0, "query_I": 100.0}, "외삽 → UNKNOWN", ("failure",)),
            Preset("power_law", "순수 power law로 순위 (800 V·35 A)", {"rank_model": "power", "query_V": 800.0, "query_I": 35.0}, "검증 실패 fit은 순위 금지", ("failure",)),
            Preset("close_call", "교차 전류 근처 (800 V·25 A)", {"query_V": 800.0, "query_I": 25.0}, "차이 < 불확도", ("corner",)),
        ],
        run=run_loss_surface,
        model_level="A",
        suggested_change="질의 전류를 60 A → 4 A(측정 하한 10 A 아래)로 바꾼다.",
        prediction=Prediction(
            "질의점을 측정 하한 아래(800 V, 4 A)로 옮기면 A/B 판정은?",
            ["A가 작다", "B가 작다", "값은 계산되지만 순위는 UNKNOWN (OUT_OF_VALIDITY)", "모르겠다"],
            "값은 계산되지만 순위는 UNKNOWN (OUT_OF_VALIDITY)",
            "fit 곡면은 어느 점에서나 숫자를 내지만 측정이 없는 곳의 숫자는 외삽이다. 저전류에서는 Coss floor의 모양(전압 지수)도 가정이라, 그 값으로 순위를 매기지 않는다.",
            ["ranking", "query_status", "E_A", "E_B"],
            handcalc=[{"key": "E_A", "label": "E_A(700 V, 60 A): 합성 참값식 0.45(V/800)^1.5 + 3.0(I/100)^1.15(V/800)^1.35 mJ", "unit": "J"}],
        ),
        suggested={"query_I": 4.0},
        student="스위칭 에너지는 전압과 전류가 클수록 커진다. 데이터시트 몇 점을 이어 곡면을 만들면 중간 값은 짐작할 수 있지만, 측정하지 않은 바깥(아주 작은 전류, 더 높은 전압)은 짐작일 뿐이다. 또 전류가 0이어도 소자 커패시턴스(Coss)를 충·방전하는 에너지는 남아서, 가벼운 부하에서는 이 바닥값이 비교를 좌우한다.",
        expert=(
            "① E ≈ E0(I/I0)^α(V/V0)^β의 α·β는 데이터에서 fit하고 불확도와 함께 적는다. ② fit 점과 검증 점을 분리하면 모델 형식 오차(순수 power law의 저전류 편향)가 fit 잔차보다 분명히 드러난다. "
            "③ I → 0 floor(E_oss(V))를 두면 경부하 순위가 달라진다 — floor의 전압 의존은 E_oss(V) 곡선이나 측정으로 확인한다. ④ 특성화 box 밖 질의점은 OUT_OF_VALIDITY/UNKNOWN으로 표시하고 순위표에서 뺀다. "
            "⑤ E_on에 상대 소자 recovery가, E_off에 Coss 충전이 들어 있는지 데이터 조건을 기록한다. ⑥ 불확도 전파(delta method)는 재fit Monte Carlo로 검산한다."
        ),
        customer_ko="주신 스위칭 에너지 데이터는 10~160 A, 400~800 V에서만 측정된 값입니다. 경부하 운전점(수 A)의 손실 비교는 이 범위 밖이라 지금 데이터로는 결론을 내지 않겠습니다. 저전류와 Coss 에너지 조건의 추가 측정을 받으면 비교를 다시 드리겠습니다.",
        customer_en="The switching-energy data you shared covers only 10 to 160 A and 400 to 800 V. The light-load operating points, at a few amps, are outside that range, so I will not draw a loss comparison there from the current data. With additional low-current measurements and the Coss energy conditions, I can redo the comparison.",
        questions=[_Q[7], _Q[1]],
        textbook=[TB_E01],
        reference_presets=["nominal", "light"],
        claim_limit="합성 소자·합성 측정의 fit/검증 절차. 실제 소자의 스위칭 손실 주장이 아니다.",
    ),
    Experiment(
        key="obc_cllc",
        title="11 kW OBC CLLC: seed n = 1의 실패, n = 0.93의 두 FHA 해, nominal 최적 ≠ corner 만족 선택",
        goal=(
            "배터리 650~920 V·link 700~850 V·120~210 kHz에서 FHA gain 창, 1차 RMS, 자속, ZVS 전하 screen, T_j를 동시에 평가해 운전영역과 active limit를 그린다. "
            "seed n = 1은 920/850 V에서 요구 gain 1.082353 > 최대 inductive gain 1.016401로 실패하고, n = 0.93은 136.099/147.061 kHz 두 해를 갖는 FHA 후보다. "
            "후보군 n = 0.85~1.00에서 nominal 손실 최소와 모든 요구 corner 만족 선택을 분리하고, 입력 제한(10.6988 kW)은 turns ratio로 풀리지 않는 고객 결정 사항임을 보인다."
        ),
        params=P_CLLC,
        presets=[
            Preset("seed", "seed n = 1 (교재 초기안)", {}, "920/850 V FAIL", ("nominal", "reference", "failure")),
            Preset("n093", "수정 후보 n = 0.93", {"n": 0.93}, "두 FHA 해 · CANDIDATE_FHA_ONLY", ("reference",)),
            Preset("n093_min", "n = 0.93, 최소 RMS 분기 선택", {"n": 0.93, "branch": "min_current"}, "RMS 최소 해는 ZVS screen 실패", ("failure",)),
            Preset("light_reg", "경부하 1.1 kW까지 연속 regulation 요구", {"n": 0.93, "P_burst": 1100.0}, "후보군 안에 해 없음", ("failure",)),
            Preset("n090", "n = 0.90", {"n": 0.90}, "저전압 경부하에서 강압 불가", ("corner",)),
        ],
        run=run_obc_cllc,
        model_level="A",
        suggested_change="n을 1.0 → 0.93으로 바꾼다.",
        prediction=Prediction(
            "seed n = 1을 n = 0.93으로 바꾸면?",
            ["모든 corner PASS, 설계 완료", "고전압 corner에 FHA 해가 생기지만 후보일 뿐 (CANDIDATE_FHA_ONLY)", "여전히 해 없음", "모르겠다"],
            "고전압 corner에 FHA 해가 생기지만 후보일 뿐 (CANDIDATE_FHA_ONLY)",
            "요구 gain n·V_bat/V_link가 1.0824 → 1.0066으로 내려가 inductive peak(1.0117) 아래가 되므로 두 해가 생긴다. 그러나 FHA 해는 switching·ZVS·startup·reverse 검증 전의 후보이고, nominal에서는 공진점을 벗어나 손실이 늘며, 11 kW 입력 제한은 그대로다.",
            ["M_req_seed", "gmax_seed", "f_left", "f_right", "loss_nominal"],
            handcalc=[{"key": "M_req_seed", "label": "요구 gain n·920/850 (n = 1)", "unit": ""}, {"key": "P_grid", "label": "√3·400·16·0.995·0.97", "unit": "W"}],
        ),
        suggested={"n": 0.93},
        student="배터리 전압이 높을수록 컨버터는 더 큰 승압비를 내야 한다. 공진 컨버터는 주파수로 승압비를 조절하지만 그 범위에 한계가 있다. 처음 설계(n = 1)는 가장 흔한 800 V에서 효율이 가장 좋지만 920 V 배터리를 채우지 못한다. 권선비를 바꾸면 채울 수 있는 대신 800 V 손실이 조금 늘어난다. 그리고 콘센트에서 받을 수 있는 전력(16 A)이 모자란 문제는 권선비로 해결되지 않는다.",
        expert=(
            "① 요구 corner를 먼저 정한다: 920/850 V·11 kW, 650/700 V·11 kW, 경부하 regulation 하한(burst 허용 전력). ② 각 운전점에서 gain(FHA, inductive 영역), 1차 RMS, 자속 n·V_bat/(4fN_pA_e), "
            "ZVS 전하 screen(i_off·t_d ≥ 2Q_oss), T_j를 동시에 평가해 active limit를 지도로 본다. ③ nominal 손실 최소(n = 1)와 corner 만족(n = 0.93)을 분리하고, 공차 screen(L_m ±15 %, L_r·C_r ±5 %)으로 여유를 본다. "
            "④ 두 FHA 해는 기울기 부호가 반대(+0.001871/−0.001803 per kHz)이고 RMS가 작은 왼쪽 해는 turn-off 전류가 작아 ZVS screen에서 불리하다 — 제어 분기·전류·ZVS는 time-domain(EX05)으로 확인한다. "
            "⑤ n을 바꾸면 실제 2차 L_r2 = L_r/n², C_r2 = C_r·n²로 referred symmetry를 유지한다. ⑥ 입력 전류 제한 부족분(≈301 W)은 CUSTOMER_DECISION_REQUIRED다."
        ),
        customer_ko="현재 입력 조건(400 V·16 A·PF 0.995·효율 0.97)에서는 배터리 11 kW보다 약 301 W 부족하고, 별도로 초기 CLLC(n = 1)는 920 V 배터리에서 gain이 모자랍니다. 권선비 0.93은 FHA에서 출력 가능성을 회복하지만 switching·공차·startup 검증 전 후보입니다. 입력 전류 요구와 경부하 regulation 범위를 먼저 합의하시죠.",
        customer_en="Under the current input assumptions, 400 V, 16 A, power factor 0.995 and 97 percent efficiency, the battery power is about 301 W short of 11 kW, and separately the initial CLLC with n = 1 lacks gain at a 920 V battery. A turns ratio of 0.93 restores a first-harmonic solution, but it remains a candidate until switching, tolerance and startup are verified. Let's first agree on the input current requirement and the light-load regulation range.",
        questions=[_Q[2], _Q[1]],
        textbook=[TB_E12, TB_13, TB_E01],
        reference_presets=["seed", "n093"],
        claim_limit="FHA 정적 해 + 합성 손실 지도(A 수준). switching·ZVS·startup·reverse 검증과 실제 부품 손실은 포함하지 않는다.",
    ),
    Experiment(
        key="dab_lv",
        title="900 V ↔ LV DAB (2 × 1.5 kW): φ = 0에서도 2.165 A — L의 nominal 최적과 corner 만족 선택",
        goal=(
            "SPS DAB 1모듈(1.5 kW, n = 50/3, 100 kHz)의 L 후보를 전력(φ_max)·RMS·T_j 제약과 ZVS 전하 screen으로 평가한다. nominal 800/48 V에서 φ = 0.328973 rad·I_rms 2.01988 A, "
            "저전압 LV corner 900 V/nV_L 600 V에서 1.5 kW면 I_rms 3.11356 A·I_pk 5.65983 A이고 φ = 0이어도 2.16506 A가 흐른다. 작은 L은 nominal RMS를 줄이지만 전압 불일치 corner에서 순환전류가 커지고, 큰 L은 저전압 corner의 최대 전력이 모자란다."
        ),
        params=P_DAB,
        presets=[
            Preset("nominal", "L = 200 µH (교재)", {}, "", ("nominal", "reference")),
            Preset("L100", "L = 100 µH (nominal 최적)", {"L": 100e-6}, "전압 불일치 corner RMS 초과", ("failure", "reference")),
            Preset("L300", "L = 300 µH", {"L": 300e-6}, "550 V/36 V 전력 부족", ("failure",)),
            Preset("map550", "지도를 V_H = 550 V에서", {"VH_map": 550.0}, "HV 쪽 ZVS가 먼저 깨지는 영역", ("variant",)),
        ],
        run=run_dab_lv,
        model_level="A",
        suggested_change="L을 200 → 100 µH로 줄인다.",
        prediction=Prediction(
            "L을 200 → 100 µH로 줄이면?",
            ["모든 corner에서 RMS가 줄어든다", "nominal RMS는 줄지만 전압 불일치 corner RMS가 커져 한계를 넘는다", "변화 없음", "모르겠다"],
            "nominal RMS는 줄지만 전압 불일치 corner RMS가 커져 한계를 넘는다",
            "같은 전력에서 φ가 작아져 nominal RMS는 2.020 → 1.940 A로 줄지만, 900/600 V처럼 전압이 맞지 않으면 L 양단 교번전압에 의한 순환전류가 1/L로 커진다(φ = 0 기준 2.165 → 4.330 A).",
            ["irms_nom", "irms_lowbat", "irms_phi0", "L_robust"],
            handcalc=[{"key": "irms_phi0", "label": "900/600 V, φ = 0의 I_rms (삼각파: (V_1−V_2′)/(4fL)/√3)", "unit": "A"}, {"key": "Pmax_nom", "label": "V_1V_2′/(8fL)", "unit": "W"}],
        ),
        suggested={"L": 100e-6},
        student="DAB는 두 브리지 사이 인덕터 L에 걸리는 전압 차로 전력을 보낸다. 양쪽 전압비가 맞으면 전류는 전력에 비례하지만, 맞지 않으면 전력을 보내지 않아도 인덕터에 전류가 왔다 갔다 한다. L이 작으면 이 헛도는 전류가 커지고, L이 크면 보낼 수 있는 최대 전력이 줄어든다.",
        expert=(
            "① per-module(1.5 kW)과 합계(3 kW)를 섞지 않는다. ② SPS 전류는 반주기 반대칭 해의 구간 선형 파형으로 정확히 구하고 닫힌 식·PWL 적분으로 대조한다. "
            "③ 전압 불일치(d ≠ 1)에서는 한쪽 브리지가 먼저 ZVS를 잃는다 — 부호·크기 screen(SCREEN_ONLY)으로만 판정하고 hard-switching 손실은 Coss floor가 있는 지도로 계산한다. "
            "④ L은 nominal 손실이 아니라 전압 corner 전체의 RMS·전력 여유로 고른다. ⑤ 변조 자유도(EPS/DPS/TPS, EX06)는 이 경계를 바꾸지만 ZVS 전하와 최소 펄스 조건이 붙는다."
        ),
        customer_ko="저전압 배터리(36 V)와 900 V HV가 겹치는 조건에서는 전력을 보내지 않아도 모듈당 약 2.2 A의 순환전류가 흐릅니다. 인덕턴스를 줄이면 정격점 효율은 약간 좋아지지만 이 조건의 전류가 두 배가 되어 허용 RMS를 넘습니다. 전압 범위 전체의 RMS 여유로 L을 정하고, 순환전류를 줄이는 변조는 따로 검토하겠습니다.",
        customer_en="When a 36 V low-voltage battery meets a 900 V bus, about 2.2 A of circulating current flows per module even with no power transfer. Reducing the inductance improves nominal efficiency slightly, but doubles the current at that corner and exceeds the allowed RMS. I would choose L on the RMS margin across the whole voltage range and look at modulation that reduces circulating current separately.",
        questions=[_Q[3], _Q[1]],
        textbook=[TB_E12, TB_11],
        reference_presets=["nominal", "L100"],
        claim_limit="SPS 이상 스위치 구간 선형 파형 + 합성 손실 지도(A 수준). ZVS는 screen만, 실제 commutation·자화전류·코어 손실은 포함하지 않는다.",
    ),
    Experiment(
        key="sic_leg",
        title="병렬 4 branch SiC leg: 110.6/99.5/99.5/90.5 A, 2 nH × 2 kA/µs = 4 V, 가장 빠른 gate가 답이 아닌 이유",
        goal=(
            "R_DS(on) 3.6/4/4/4.4 mΩ 네 branch의 정적 분배(400 A → 110.553/99.497/99.497/90.452 A)를 R_DS(on)(T) 전열 고정점으로 확장하고, gate 저항 후보별로 과전압(V_bus + L_loop·di/dt), "
            "공통 source 결합(L_s·di/dt, 2 nH × 2 kA/µs = 4 V), T_j, branch 전류를 동시에 본다. nominal 손실 최소(가장 빠른 gate)와 모든 corner 만족 선택을 분리한다."
        ),
        params=P_LEG,
        presets=[
            Preset("nominal", "R_g 4.7 Ω (branch 2 kA/µs)", {}, "모든 corner 만족 (과전압 여유 0.4 %)", ("nominal", "reference")),
            Preset("fast", "R_g 1 Ω (nominal 손실 최소)", {"Rg": 1.0}, "과전압·결합 한계", ("failure", "reference")),
            Preset("kelvin", "Kelvin source만 추가 (R_g 3.3 Ω)", {"Rg": 3.3, "kelvin": True}, "결합은 해결, 과전압은 남음", ("variant",)),
            Preset("kelvin_lowL", "Kelvin + 루프 8 nH (R_g 3.3 Ω)", {"Rg": 3.3, "kelvin": True, "L_loop": 8e-9}, "더 빠른 gate 가능", ("variant",)),
        ],
        run=run_sic_leg,
        model_level="A",
        suggested_change="R_g를 4.7 → 1 Ω으로 줄인다.",
        prediction=Prediction(
            "R_g를 4.7 → 1 Ω으로 줄이면?",
            ["손실이 줄어 모든 corner PASS", "nominal 손실은 줄지만 과전압·결합 전압 한계를 넘는다", "T_j가 먼저 한계를 넘는다", "모르겠다"],
            "nominal 손실은 줄지만 과전압·결합 전압 한계를 넘는다",
            "di/dt ∝ 1/(R_g + R_int)로 2 → 5.7 kA/µs가 되어 스위칭 에너지는 줄지만, 12 nH × 4 branch × di/dt 과전압이 커지고 공통 source 결합이 4 → 11.4 V가 된다.",
            ["loss_nominal", "V_cs", "didt", "Rg_robust"],
            handcalc=[{"key": "I_static_1", "label": "400 A × (1/3.6)/(1/3.6 + 2/4 + 1/4.4)", "unit": "A"}, {"key": "V_cs", "label": "2 nH × 2 kA/µs", "unit": "V"}],
        ),
        suggested={"Rg": 1.0},
        student="여러 소자를 나란히 붙이면 저항이 가장 작은 소자로 전류가 더 몰리고 그 소자가 가장 뜨거워진다. 스위칭을 빠르게 하면 손실은 줄지만, 전류가 급하게 변하면서 배선 인덕턴스에 큰 전압이 생겨 소자 전압이 튀고 gate 신호가 흔들린다.",
        expert=(
            "① 정적 분배는 R_DS(on)만의 결과다 — 동적 분배(gate timing skew, L_s)는 EX03. 10 ns skew × 2 kA/µs = 20 A는 scale이지 branch 전류 예측이 아니다. "
            "② R_DS(on)의 양의 온도계수는 분배를 약간 균등화한다(전열 고정점). ③ gate 속도는 스위칭 손실과 과전압·결합 전압의 상충이다: nominal 최적 R_g가 최대 bus corner를 통과하지 못한다. "
            "④ Kelvin source는 결합을 줄이지만 과전압은 루프 인덕턴스로만 줄어든다. ⑤ gate 파형 측정 기준점(전력 source vs Kelvin)을 먼저 확인한다. ⑥ 경부하 손실은 지도 하한 아래라 UNKNOWN이다."
        ),
        customer_ko="가장 빠른 gate 설정은 정격점 손실이 가장 작지만 900 V bus에서 과전압과 공통 source 결합 전압이 한계를 넘습니다. 현재 루프에서는 4.7 Ω이 모든 조건을 만족하고, Kelvin source와 루프 인덕턴스 감소를 함께 적용하면 더 빠른 설정도 가능해 보입니다. branch별 전류와 local gate 전압을 같은 timebase로 측정해 확인하시죠.",
        customer_en="The fastest gate setting has the lowest nominal loss, but at a 900 V bus its overshoot and common-source coupling exceed their limits. With the present loop, 4.7 ohms meets every condition, and a Kelvin source together with a lower loop inductance may allow a faster setting. Let's confirm with branch currents and local gate voltages measured on one timebase.",
        questions=[_Q[4], _Q[1]],
        textbook=[TB_E12, TB_E03],
        reference_presets=["nominal", "fast"],
        claim_limit="정적·전열 분배와 합성 손실 지도, L·di/dt 결합 screen(A 수준). 동적 분배·ringing·SOA·단락은 포함하지 않는다.",
    ),
    Experiment(
        key="mission_ranking",
        title="A가 nominal에서 8 W 좋지만 모델이 ±15 W라면? mission 효율 = ΣE_out/ΣE_in과 손실 회계",
        goal=(
            "n = 0.93 CLLC에 두 소자 후보를 넣어 비교한다. A는 nominal(800 V·11 kW)에서 8 W 작다(교재 시나리오). 스위칭 손실 모델 불확실성 ±15 W를 후보마다 적용하면 차이의 불확도는 21.2 W(ρ = 0)라 순위는 UNRESOLVED_RANKING이다. "
            "합성 충전 mission에서 효율을 ΣE_out/ΣE_in으로 계산해 운전점 효율의 평균과 비교하고, 손실 회계표로 E_oss 추가·hard E_on 적용·dead time 이중 계산·AC 저항 누락을 점검한다."
        ),
        params=P_MISSION,
        presets=[
            Preset("textbook", "A가 8 W 좋다, 모델 ±15 W", {}, "UNRESOLVED_RANKING", ("nominal", "reference")),
            Preset("same_fixture", "같은 fixture 교대 측정, ρ_AB = 0.99 (근거 기록)", {"rho_AB": 0.99, "rho_basis": "same_fixture"}, "8 W를 가릴 수 있음", ("variant",)),
            Preset("rho_nobasis", "ρ_AB = 0.99, 근거 없음", {"rho_AB": 0.99}, "MISSING_INPUT", ("failure",)),
            Preset("eoss_double", "ZVS인데 E_oss를 손실로 추가", {"add_eoss": True}, "회계 오류 → OUT_OF_VALIDITY", ("failure", "reference")),
            Preset("single_phase", "단상 3.5 kW 완속 충전 mission", {"profile": "single_phase"}, "mission 가중으로 순위가 바뀜", ("variant",)),
            Preset("deep_taper", "CV를 0.6 kW까지 (burst 구간 포함)", {"profile": "deep_taper"}, "burst 구간 UNKNOWN", ("failure",)),
        ],
        run=run_mission,
        model_level="A + MC",
        suggested_change="후보 간 오차 상관 ρ_AB를 0 → 0.99로 올린다 (근거는 아직 기록하지 않는다).",
        prediction=Prediction(
            "근거 없이 ρ_AB = 0.99로 두면 판정은?",
            ["A가 확정적으로 이긴다", "불확도는 줄지만 근거가 없어 MISSING_INPUT", "변화 없음", "모르겠다"],
            "불확도는 줄지만 근거가 없어 MISSING_INPUT",
            "공통 오차가 상쇄되면 u_Δ = 15·√(2·0.01) = 2.1 W로 줄어 8 W를 가릴 수 있다. 하지만 같은 fixture·같은 보정으로 교대 측정했다는 근거가 없으면 줄어든 error bar를 주장할 수 없다.",
            ["u_diff_nom", "rank_nom"],
            handcalc=[{"key": "u_diff_nom", "label": "√(15² + 15²) (ρ = 0)", "unit": "W"}],
        ),
        suggested={"rho_AB": 0.99},
        student="두 부품의 손실 계산이 각각 ±15 W씩 틀릴 수 있다면 8 W 차이는 계산 오차 안에 묻힌다. 누가 더 좋은지는 계산을 더 정밀하게 그려서가 아니라 두 부품을 같은 장비로 번갈아 재는 식의 분별력 있는 시험으로 가린다. 또 하루 동안의 효율은 여러 순간 효율의 평균이 아니라 ‘내보낸 에너지 합 ÷ 받은 에너지 합’이다.",
        expert=(
            "① 차이의 불확도 u_Δ = √(u_A² + u_B² − 2ρu_Au_B): 같은 방법·fixture의 공통 오차만 상쇄되며 그 근거를 기록한다. ② 모델 오차는 후보마다 계통적이라 mission 점 사이에 완전 상관으로 누적한다. "
            "③ mission 효율 = ΣE_out/ΣE_in(= E_out 가중 조화평균); 운전점 효율의 산술·시간 평균은 에너지 비가 아니다. ④ 손실 회계는 소자별 실제 commutation event로 한 번씩: ZVS에서 E_oss 추가·hard E_on 적용은 오적용, "
            "E_off 데이터에 Coss가 이미 들어 있으면 중복, 권선 DC R만 쓰면 누락. ⑤ 결정은 분리 가능한 DPT/열 시험, mission 가중, 정격·공급 등 다른 축으로 한다. 비용 자료가 없으면 만들어 넣지 않는다."
        ),
        customer_ko="두 소자의 정격점 손실 차이는 8 W로 계산되지만 스위칭 손실 모델의 불확실성이 ±15 W라 이 계산만으로 우열을 정할 수 없습니다. 같은 DPT setup에서 두 소자를 교대로 측정하거나 열 시험으로 차이를 직접 확인하고, 실제 충전 패턴의 에너지 가중으로 비교하겠습니다.",
        customer_en="The calculated nominal loss difference between the two devices is 8 W, but the switching-loss model is uncertain by plus or minus 15 W, so this calculation alone cannot rank them. I would measure both devices alternately on the same double-pulse setup or compare them in a thermal test, and weight the comparison by the energy of your actual charging profile.",
        questions=[_Q[0], _Q[5], _Q[6]],
        textbook=[TB_E01, TB_E12],
        reference_presets=["textbook", "eoss_double"],
        claim_limit="합성 손실 지도·합성 mission의 순위 판정 절차. 실제 소자 비교·수율·비용 주장이 아니다.",
    ),
]

LAB = Lab(
    id="EX01",
    title="설계영역·손실 지도·부품선정",
    title_en="Design space, loss maps and part selection",
    track="expert",
    order=1,
    path_note="E13 1회전 (E01–E03)",
    textbook=[TB_E01, TB_E12, TB_13, TB_11],
    prerequisites=["FL02", "FL03", "FL12"],
    summary="합성 loss surface의 fit·검증·외삽 → 세 flagship(11 kW OBC CLLC, 900 V↔LV DAB, 병렬 SiC leg)의 운전영역·active limit·nominal 최적 vs corner 만족 선택 → 불확도 안의 A/B 순위, mission 효율, 손실 회계.",
    experiments=EXPERIMENTS,
    minimum_scope=(
        "synthetic loss map과 동시 제약으로 운전영역·active limit·A/B ranking 시각화; nominal optimum과 required-corner feasible choice 분리; 보간 범위와 외삽 status; "
        "불확도보다 작은 차이는 UNRESOLVED_RANKING; mission efficiency = ΣE_out/ΣE_in (지침 §10, 교재 E01·E12)"
    ),
    claim_limits=[
        "손실 지도·소자·mission은 모두 합성 학습 자료 — 실제 부품 비교·추천이 아니다 (DATASHEET 자료 없음, MISSING_INPUT)",
        "FHA·SPS·정적 분배는 A 수준: ZVS·동적 분배·startup·reverse는 해당 EX(EX02/03/05/06)에서 확인",
        "비용은 자료가 없어 목적함수에 넣지 않았다",
        "모든 판정은 모델 범위 안의 verification이며 hardware validation이 아니다",
    ],
    test_paths=["tests/test_ex01.py"],
    extends=["FL02", "FL03", "FL08", "FL10"],
)
