"""EX11 - How far can the model be trusted: verification, identification, uncertainty (textbook E11).

Linked to every lab: the last experiment runs the implemented labs' reference presets and
scans their source and tests to build a data-driven test-independence map and model card.

Physics/statistics (A level, all synthetic inputs with explicit seeds):
  * GUM propagation for loss = P_in - P_out and eta = P_out/P_in with a correlation that must
    have a recorded basis; ranking under uncertainty (UNRESOLVED_RANKING);
  * binomial bounds: 0/1000 -> 1 - 0.05^(1/1000) = 0.2991 %, Clopper-Pearson for k failures;
    a deterministic grid gets no binomial bound (NOT_EVALUABLE);
  * identifiability: a ringing waveform pins only L*C and R/L; a second excitation resolves it;
    calibration/holdout at the design-deciding corner vs a random 80/20 split;
  * Monte Carlo reporting: distributions, correlation, seed, N, convergence, model-form bias
    kept apart from random spread, yield conditional on the assumptions.
Independent paths: Monte Carlo vs GUM, finite differences vs analytic sensitivities, a
binomial-sum root vs the Beta quantile, exact engine data vs closed-form model, analytic
lognormal yield vs Monte Carlo, static source scan vs executed checks.
"""

from __future__ import annotations

import ast
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset, resolve_params
from ..model.result import Check, Result, Table, check_close
from ..model.units import fmt_si
from ..reference import uncertainty as ref

TB_E11 = TextbookRef("expert-e11-모델을-믿을-수-있는-범위-검증식별불확도", "E11. 모델을 믿을 수 있는 범위: 검증·식별·불확도")
TB_E12 = TextbookRef("expert-e12-설계-리뷰를-통과하는-답변-세-개의-통합-사례", "E12. 설계 리뷰 (Q7·Q9·Q11)")
TB_19 = TextbookRef("교재실습-계약과-통과-기준", "19. 교재–실습 계약과 통과 기준")

SPEC_KO = {"standard": "표준불확도(1σ)로 주어짐", "rect_limit": "±한계(직사각 분포) → a/√3", "k2": "확장불확도 k = 2 → a/2"}
BASIS_KO = {
    "none": "근거 기록 없음",
    "same_meter": "같은 전력분석기·같은 range·동시 취득 (교정 성분 공유)",
    "cal_cert": "교정 성적서의 공분산 자료",
}


# ======================================================================================
# Experiment 1: loss from two power readings
# ======================================================================================


def run_loss_uncertainty(v: dict) -> Result:
    res = Result("EX11", "loss_uncertainty", "A (GUM 1차 전파) + Monte Carlo 검산")
    Pin, Pout = v["P_in"], v["P_out"]
    if Pout >= Pin:
        res.verdict("OUT_OF_VALIDITY", "P_out ≥ P_in: 손실이 0 이하 — 측정 경계(보조전원·냉각수 전력 등)를 먼저 맞춘다")
        return res
    kind = v["spec_type"]
    uin = ref.standard_from_spec(v["rel_u_in"], kind) * Pin
    uout = ref.standard_from_spec(v["rel_u_out"], kind) * Pout
    rho = v["rho"]
    loss = Pin - Pout
    eta = Pout / Pin
    u0 = ref.u_difference(uin, uout, 0.0)
    ur = ref.u_difference(uin, uout, rho)
    ue0 = ref.u_efficiency(Pin, Pout, uin, uout, 0.0)
    uer = ref.u_efficiency(Pin, Pout, uin, uout, rho)
    tb = abs(Pin - 11000) < 1e-9 and abs(Pout - 10780) < 1e-9 and abs(v["rel_u_in"] - 1e-3) < 1e-15 and abs(v["rel_u_out"] - 1e-3) < 1e-15 and kind == "standard"
    res.add_metric("loss", "손실 P_in − P_out", loss, "W", ref=220.0 if tb else None, ref_label="교재 220 W", tol=1e-12)
    res.add_metric("eta", "효율 η = P_out/P_in", eta, "", ref=0.98 if tb else None, ref_label="교재 0.98", tol=1e-12)
    res.add_metric("u_in", "u(P_in) (표준불확도)", uin, "W", ref=11.0 if tb else None, ref_label="교재 11 W", tol=1e-12, basis=SPEC_KO[kind])
    res.add_metric("u_out", "u(P_out)", uout, "W", ref=10.78 if tb else None, ref_label="교재 10.78 W", tol=1e-12, basis=SPEC_KO[kind])
    res.add_metric("u_loss_ind", "u(loss), 독립 (ρ = 0)", u0, "W", ref=15.40157 if tb else None, ref_label="교재 15.40157 W", tol=1e-6, basis=f"loss의 {u0 / loss:.1%}")
    res.add_metric("u_loss_rho", f"u(loss), ρ = {rho:g}", ur, "W", ref=4.87487 if (tb and abs(rho - 0.9) < 1e-12) else None, ref_label="교재 4.87487 W (ρ = 0.9)", tol=1e-5, basis=f"상관 근거: {BASIS_KO[v['rho_basis']]}")
    res.add_metric("u_eta", "u(η), 독립", ue0, "", ref=0.001386 if tb else None, ref_label="교재 ≈0.001386 (0.139 %p)", tol=5e-4, basis=f"= {ue0 * 100:.4f} %p")
    res.add_metric("u_eta_rho", f"u(η), ρ = {rho:g}", uer, "", basis=f"= {uer * 100:.4f} %p")
    res.add_metric("rel_loss_vs_eta", "상대 불확도: loss / η", (u0 / loss) / (ue0 / eta), "", basis=f"loss {u0 / loss:.2%} vs η {ue0 / eta:.3%}: 손실의 상대 불확도가 훨씬 크다")
    # --- ranking between two candidates measured separately with this setup
    dAB = v["dP_AB"]
    uA = ur if v["rho_basis"] != "none" else u0
    u_diff = ref.u_difference(uA, uA, v["rho_AB"])
    k = v["k_cov"]
    resolved = abs(dAB) > k * u_diff
    res.add_metric("u_diff", "A−B 손실 차이의 표준불확도", u_diff, "W", basis=f"u_A = u_B = {uA:.4g} W, 후보 간 오차 상관 ρ_AB = {v['rho_AB']:g}")
    res.add_metric("U_diff", f"확장 불확도 k = {k:g}", k * u_diff, "W", basis=f"|Δ| = {abs(dAB):g} W와 비교")
    res.add_metric("ranking", "순위 판정", "확정" if resolved else "UNRESOLVED_RANKING", "", basis=f"|Δ| {'>' if resolved else '≤'} k·u_Δ")
    # --- independent path 1: Monte Carlo with correlated Gaussian inputs (explicit seed)
    rng = np.random.default_rng(int(v["seed"]))
    N = int(v["mc_n"])
    z = rng.standard_normal((N, 2))
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    e = z @ L.T
    pin_s = Pin + uin * e[:, 0]
    pout_s = Pout + uout * e[:, 1]
    loss_s = pin_s - pout_s
    eta_s = pout_s / pin_s
    u_mc = float(np.std(loss_s, ddof=1))
    ue_mc = float(np.std(eta_s, ddof=1))
    se = 1.0 / math.sqrt(2 * (N - 1))  # relative standard error of a standard-deviation estimate
    res.add_check(check_close(f"u(loss): Monte Carlo (N = {N}, seed {int(v['seed'])}) vs GUM", u_mc, ur, 5 * se, "상관 Gaussian 표본(Cholesky)의 표본 표준편차 vs √(u_in² + u_out² − 2ρu_in u_out)", True, "W", detail=f"MC {u_mc:.5g} W / GUM {ur:.5g} W, 허용 = 5 × 표본 표준편차의 상대 SE ({se:.2e})"))
    res.add_check(check_close("u(η): Monte Carlo vs 1차 GUM (선형화 오차 포함)", ue_mc, uer, 5 * se + 1e-3, "P_out/P_in 비율의 표본 표준편차 vs 감도계수 식", True, "", detail=f"MC {ue_mc:.6g} / GUM {uer:.6g}"))
    # --- independent path 2: finite-difference sensitivity coefficients
    h = 1e-3
    c_out_fd = ((Pout + h) / Pin - (Pout - h) / Pin) / (2 * h)
    c_in_fd = (Pout / (Pin + h) - Pout / (Pin - h)) / (2 * h)
    ue_fd = math.sqrt((c_out_fd * uout) ** 2 + (c_in_fd * uin) ** 2 + 2 * rho * c_out_fd * c_in_fd * uout * uin)
    res.add_check(check_close("η 감도계수: 중앙차분 vs 해석 편미분", ue_fd, uer, 1e-7, "∂η/∂P 중앙차분으로 다시 조립한 u(η) vs 해석식", True, ""))
    # --- plots
    rhos = np.linspace(-1, 1, 81)
    res.add_series("u_vs_rho", "u(loss)", "W", rhos.tolist(), [ref.u_difference(uin, uout, r) for r in rhos])
    mk = [{"x": 0.0, "y": u0, "label": f"독립 {u0:.4g} W"}, {"x": 0.9, "y": ref.u_difference(uin, uout, 0.9), "label": f"ρ = 0.9: {ref.u_difference(uin, uout, 0.9):.4g} W"}]
    if rho not in (0.0, 0.9):
        mk.append({"x": rho, "y": ur, "label": f"ρ = {rho:g}: {ur:.4g} W"})
    res.add_plot("p_rho", "상관계수에 따른 loss 불확도", ["u_vs_rho"], x_label="ρ(P_in, P_out 오차)", x_unit="", y_label="u(loss)", y_unit="W", kind="xy", level="A",
                 markers=mk,
                 proved="양의 상관(같은 계측기의 공통 교정 오차)은 차이의 불확도를 줄인다. ρ = 0.9면 15.4 W가 4.87 W가 된다.",
                 not_yet="ρ 값 자체의 근거(동시 취득·같은 range·교정 공분산)는 이 계산이 만들지 않는다. 근거 없이 ρ를 넣어 error bar를 줄이면 안 된다.")
    hist, edges = np.histogram(loss_s, bins=60)
    cx = 0.5 * (edges[:-1] + edges[1:])
    res.add_series("mc_hist", f"MC 손실 분포 (N = {N})", "개", cx.tolist(), hist.tolist())
    res.add_series("gum_pdf", "GUM 정규 근사 (같은 면적)", "개", cx.tolist(), (N * (edges[1] - edges[0]) * norm.pdf(cx, loss, ur)).tolist(), dash=True)
    res.add_plot("p_hist", "손실의 Monte Carlo 분포와 GUM 근사", ["mc_hist", "gum_pdf"], x_label="loss", x_unit="W", y_label="표본 수", y_unit="", kind="xy", level="A + MC",
                 vlines=[{"x": loss - k * ur, "label": f"−{k:g}u"}, {"x": loss + k * ur, "label": f"+{k:g}u"}],
                 proved="입력 분포·상관·seed를 명시한 Monte Carlo가 1차 전파와 일치한다(손실은 입력의 선형 함수).",
                 not_yet="입력 불확도가 정규라는 것, 계측기 사양이 무엇을 뜻하는지(1σ·한계·k = 2)는 가정이다.")
    rab = np.linspace(-1, 1, 81)
    res.add_series("res_diff", f"k·u_Δ (k = {k:g})", "W", rab.tolist(), [k * ref.u_difference(uA, uA, r) for r in rab])
    res.add_plot("p_rank", "후보 A/B 차이를 가릴 수 있는가", ["res_diff"], x_label="후보 간 오차 상관 ρ_AB", x_unit="", y_label="분해 가능한 최소 차이", y_unit="W", kind="xy", level="A",
                 hlines=[{"y": abs(dAB), "label": f"|Δ| = {abs(dAB):g} W"}],
                 proved="같은 setup·같은 조건에서 두 후보를 번갈아 측정해 공통 오차가 상쇄될 때(ρ_AB → 1)만 작은 차이를 가릴 수 있다.",
                 not_yet="반복성·온도·보조전원 경계·열평형 같은 다른 오차원은 따로 예산에 넣어야 한다.")
    rows = [
        ["P_in", Pin, uin, f"{SPEC_KO[kind]}"],
        ["P_out", Pout, uout, f"{SPEC_KO[kind]}"],
        ["loss (ρ = 0)", loss, u0, "√(u_in² + u_out²)"],
        [f"loss (ρ = {rho:g})", loss, ur, f"√(u_in² + u_out² − 2ρu_in u_out); 근거: {BASIS_KO[v['rho_basis']]}"],
        ["η (ρ = 0)", eta, ue0, "감도계수 1/P_in, −P_out/P_in²"],
        ["A−B 차이", dAB, u_diff, f"k = {k:g} → {'순위 확정' if resolved else 'UNRESOLVED_RANKING'}"],
    ]
    res.tables.append(Table("t_budget", "불확도 예산 (표준불확도)", ["양", "값", "u", "근거·식"], rows, note="계측기 ‘accuracy ±0.1 %’를 그대로 1σ로 쓰지 않는다: reading/range 항, 분포, 교정, 대역폭, PF·고조파 조건을 확인한다(교재 E11)."))
    if rho != 0 and v["rho_basis"] == "none":
        res.verdict("MISSING_INPUT", f"ρ = {rho:g}로 줄어든 불확도({ur:.3g} W)는 상관의 근거가 기록되지 않아 주장할 수 없다 — 독립 값 {u0:.4g} W를 쓴다")
    if not resolved:
        res.verdict("UNRESOLVED_RANKING", f"A/B 차이 {abs(dAB):g} W ≤ k·u_Δ = {k * u_diff:.3g} W: 이 시험 하나로 순위를 확정할 수 없다 — 같은 setup 교대 측정·열 시험 등 분별력 있는 시험이 필요")
    if not res.verdicts:
        res.verdict("PASS_WITHIN_MODEL", "불확도 전파와 순위 판정이 Monte Carlo·유한차분 검산과 일치")
    res.assumptions += ["입력 오차는 정규분포(Monte Carlo), 1차 전파는 선형화", "두 후보는 같은 setup으로 따로 측정 (ρ_AB는 후보 간 공통 오차)", "측정 경계(보조전원·냉각수 전력·열평형)는 같다고 가정"]
    res.not_valid_for += ["실제 계측기 사양의 해석 (MISSING_INPUT: 사양서·교정성적서 필요)", "비정상상태·고조파가 큰 전력 측정"]
    res.interpretation = (
        f"입력·출력 전력을 각각 0.1 %로 측정해도 손실 {loss:g} W의 불확도는 독립이면 {u0:.2f} W(손실의 {u0 / loss:.1%})다. 효율로 보면 0.139 %p라 작아 보이지만, "
        f"두 후보의 손실 차이 {abs(dAB):g} W를 가리기에는 부족하다. 상관을 넣으면 불확도가 줄지만(ρ = 0.9 → 4.87 W) 같은 계측기·동시 취득 같은 근거를 기록해야 한다."
    )
    return res


# ======================================================================================
# Experiment 2: binomial bounds and why a deterministic grid gets none
# ======================================================================================


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) by direct summation of the binomial pmf (log-space terms)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    s = 0.0
    for j in range(0, k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq)
    return s


def upper_by_root(k: int, n: int, conf: float) -> float:
    """Independent path: the p at which P(X <= k) = 1 - conf, solved with brentq."""
    a = 1.0 - conf
    return brentq(lambda p: _binom_cdf(k, n, p) - a, 1e-15, 1 - 1e-15, xtol=1e-16, rtol=1e-14)


def run_binomial(v: dict) -> Result:
    res = Result("EX11", "binomial_bounds", "A (Clopper–Pearson) + 독립 근·Monte Carlo")
    n, k, conf = int(v["n_trials"]), int(v["k_fail"]), v["conf"]
    if k > n:
        res.verdict("OUT_OF_VALIDITY", "실패 수가 시행 수보다 클 수 없다")
        return res
    grid = v["sampling"] == "grid"
    # the deterministic-grid demonstration always runs (it is the counter-example)
    rng = np.random.default_rng(int(v["seed"]))
    thr = v["g_threshold"]
    p_true = float(norm.sf(thr / math.sqrt(3.0)))  # g = x1 + x2 + x3 ~ N(0, 3)
    g_side = int(round(n ** (1 / 3)))
    ax = np.linspace(-v["grid_span"], v["grid_span"], g_side)
    G = np.array(np.meshgrid(ax, ax, ax, indexing="ij")).reshape(3, -1).T
    grid_fail = int(np.sum(G.sum(axis=1) > thr))
    X = rng.standard_normal((n, 3))
    mc_fail = int(np.sum(X.sum(axis=1) > thr))
    res.add_metric("p_true", "합성 문제의 실제 실패확률 (해석)", p_true, "", basis=f"g = θ1+θ2+θ3 > {thr:g}, θ ~ N(0,1) 독립")
    res.add_metric("grid_frac", f"결정적 격자 {g_side}³ = {g_side**3}점의 실패 비율", grid_fail / g_side**3, "", basis=f"격자 범위 ±{v['grid_span']:g}σ에 따라 바뀜 — 확률이 아니다", note=f"실패 {grid_fail}점: 경계 근처 corner를 찾는 데는 유용")
    res.add_metric("mc_fail", f"IID Monte Carlo {n}회의 실패 수", mc_fail, "", basis=f"seed {int(v['seed'])}")
    if grid:
        res.add_metric("p_upper", "binomial 상한", "NOT_EVALUABLE", "", basis="결정적 격자는 IID 표본이 아니다")
        res.verdict("NOT_EVALUABLE", "결정적 grid 결과에는 binomial bound를 붙이지 않는다: 점들이 분포에서 뽑힌 IID 시행이 아니다. 격자의 실패 비율은 격자 범위·밀도가 정한 숫자이며 확률이 아니다.")
    else:
        pu = ref.clopper_pearson_upper(k, n, conf)
        pr = upper_by_root(k, n, conf)
        tb = n == 1000 and k == 0 and abs(conf - 0.95) < 1e-12
        res.add_metric("p_upper", f"one-sided {conf:.0%} 상한 (k = {k}, n = {n})", pu, "", ref=0.002991 if tb else None, ref_label="교재 0.2991 %", tol=5e-4, basis=f"= {pu * 100:.4f} %; Clopper–Pearson (Beta 분위수)")
        if k == 0:
            res.add_metric("p_upper_zero", "0 실패 닫힌 식 1 − (1 − conf)^(1/n)", ref.zero_failure_upper(n, conf), "", ref=pu, ref_label="Beta 분위수", tol=1e-12)
            res.add_metric("rule3", "rule of three 3/n (근사)", 3.0 / n, "", basis=f"정확값 대비 {3.0 / n / pu - 1:+.2%}")
        lo2, hi2 = ref.clopper_pearson_two_sided(k, n, conf)
        res.add_metric("ci2", f"양측 {conf:.0%} 구간", f"[{lo2 * 100:.4g} %, {hi2 * 100:.4g} %]", "", basis="Clopper–Pearson")
        res.add_check(check_close("상한: Beta 분위수 vs 이항 누적합의 근 (brentq)", pr, pu, 1e-9, "scipy Beta ppf vs Σ_j≤k C(n,j)p^j(1−p)^(n−j) = 1−conf를 직접 합산해 푼 p", True, ""))
        # coverage check by simulation at p = the bound: P(X <= k) should be 1 - conf
        reps = 20000
        xs = rng.binomial(n, pu, size=reps)
        cov = float(np.mean(xs <= k))
        se = math.sqrt((1 - conf) * conf / reps)
        res.add_check(Check("상한의 의미: p = 상한에서 k 이하 실패가 나올 확률 = 1 − conf (시뮬레이션)", "PASS" if abs(cov - (1 - conf)) < 5 * se else "FAIL", cov, "", 1 - conf, path=f"numpy 이항 표본 {reps}회 (seed {int(v['seed'])})", independent=True, detail=f"관측 {cov:.4f}, 기대 {1 - conf:.4f} ± {5 * se:.4f}"))
        res.verdict("PASS_WITHIN_MODEL", f"IID 가정 아래 {conf:.0%} 신뢰 상한 {pu * 100:.4g} % — ‘실패확률 0’이나 ‘ppm 수준 안전’이 아니다")
    # plots: upper bound vs n for k = 0, 1, 2
    ns = np.unique(np.geomspace(10, 1e5, 60).astype(int))
    for kk in (0, 1, 2):
        res.add_series(f"ub_k{kk}", f"k = {kk}", "", ns.tolist(), [ref.clopper_pearson_upper(kk, int(m), conf) for m in ns])
    res.add_series("ub_rule3", "3/n (k = 0 근사)", "", ns.tolist(), (3.0 / ns).tolist(), dash=True)
    res.add_plot("p_ub", f"one-sided {conf:.0%} 실패확률 상한 vs 시행 수", ["ub_k0", "ub_k1", "ub_k2", "ub_rule3"], x_label="IID 시행 수 n", x_unit="", y_label="p 상한", y_unit="", kind="xy", log_x=True, log_y=True, level="A",
                 markers=[{"x": 1000, "y": ref.zero_failure_upper(1000, 0.95), "label": "1000회 0실패: 0.2991 %"}],
                 proved="실패가 0이어도 상한은 약 3/n이다. ppm 수준을 말하려면 수백만 번의 IID 시행이 필요하다.",
                 not_yet="IID·같은 분포라는 가정이 깨지면(공정 상관·systematic bias·모델 오차) 이 상한도 의미를 잃는다.")
    ps = np.geomspace(1e-5, 2e-2, 200)
    res.add_series("lik0", f"P(0 실패 | p), n = {n}", "", ps.tolist(), [(1 - p) ** n for p in ps])
    res.add_plot("p_lik", "0 실패가 나올 확률이 5 %가 되는 p가 상한이다", ["lik0"], x_label="실패확률 p", x_unit="", y_label="P(X = 0)", y_unit="", kind="xy", log_x=True, level="A",
                 hlines=[{"y": 1 - conf, "label": f"{1 - conf:.0%}"}], vlines=[{"x": ref.zero_failure_upper(n, conf), "label": "상한"}],
                 proved="상한은 ‘그 p라면 이런 결과(0 실패)가 5 %밖에 안 나온다’는 뜻이다.", not_yet="모든 시행의 p가 같고 서로 독립이라는 가정 위의 곡선이다. 시행 사이 상관이 있으면 유효 시행 수가 줄어 상한이 커진다.")
    rows = [[kk, ref.clopper_pearson_upper(kk, n, conf) * 100, upper_by_root(kk, n, conf) * 100] for kk in range(0, 6)]
    res.tables.append(Table("t_k", f"n = {n}에서 실패 수별 one-sided {conf:.0%} 상한", ["k", "Beta 분위수 [%]", "이항합 근 [%]"], rows))
    res.tables.append(Table("t_grid", "결정적 격자 vs IID Monte Carlo (같은 합성 문제)", ["방법", "점 수", "실패 수", "해석"], [
        [f"결정적 격자 ±{v['grid_span']:g}σ", g_side**3, grid_fail, "corner가 존재하는지 찾는 데 유용; 비율은 확률이 아니며 binomial bound를 붙이지 않는다"],
        ["IID Monte Carlo", n, mc_fail, f"상한 {ref.clopper_pearson_upper(mc_fail, n, conf) * 100:.3g} % (실제 {p_true * 100:.3g} %)"],
    ], note="희귀 실패가 설계 경계 근처에 있으면 targeted corner search와 mechanism 기반 stress test가 더 직접적이다(교재 E11)."))
    res.assumptions += ["시행은 IID Bernoulli (같은 분포, 서로 독립)", "합성 실패 문제: g = θ1+θ2+θ3 > 임계, θ ~ N(0,1)"]
    res.not_valid_for += ["양산 불량률 보증", "상관된 공정·systematic bias가 있는 시험", "결정적 grid·corner sweep 결과의 확률 해석"]
    res.interpretation = (
        f"1000번의 IID 시행에서 실패가 0이어도 {conf:.0%} 신뢰로 말할 수 있는 것은 ‘실패확률이 약 0.30 % 이하’까지다. 결정적 격자는 분포에서 뽑은 표본이 아니므로 "
        "같은 계산을 붙일 수 없고, 격자의 실패 비율은 격자 범위를 바꾸면 달라지는 숫자다. 대신 격자는 실패 corner가 어디 있는지 찾는 데 쓴다."
    )
    return res


# ======================================================================================
# Experiment 3: identifiability of (L, C, R) from ringing, and calibration/holdout
# ======================================================================================


class _RLCStep(HybridSystem):
    """Data generator (exact engine): series R-L-C driven by a step V, x = [i, v_C]."""

    state_names = ("i", "v_c")

    def __init__(self, V, R, L, C):
        self.V, self.R, self.L, self.C = V, R, L, C

    def mode(self, q):
        return AffineMode(f"ex11rlc|{self.V}|{self.R}|{self.L}|{self.C}", [[-self.R / self.L, -1 / self.L], [1 / self.C, 0.0]], [self.V / self.L, 0.0])

    def outputs(self, q):
        return {"v_c": np.array([0.0, 1.0, 0.0])}


def _rlc_model(t, L, C, R, V=1.0):
    """Fit model (closed form, underdamped): capacitor voltage of a series RLC step response."""
    a = R / (2 * L)
    w0 = 1 / math.sqrt(L * C)
    wd = math.sqrt(max(w0 * w0 - a * a, 1e-30))
    return V * (1 - np.exp(-a * t) * (np.cos(wd * t) + a / wd * np.sin(wd * t)))


def _synthetic_ringing(V, R, L, C, t, sigma, rng):
    tr = simulate(_RLCStep(V, R, L, C), None, [0.0, 0.0], 0.0, float(t[-1]) * (1 + 1e-12))
    y = np.array([float(np.array([0.0, 1.0, 0.0]) @ tr.state_at(float(tk))[0]) for tk in t])
    return y + sigma * rng.standard_normal(t.size), y


def _rlc_circuit(v: dict) -> Circuit:
    c = Circuit("ex11_rlc", 700, 240, title="합성 측정 대상: step → R–L–C (ringing은 v_C에서 관측)")
    vs = c.add("vsource", "V", 90, 125, 90, "V_step", f"{v['V_step']:g} V", lpos=(72, 121, "end"))
    r = c.add("resistor", "R", 190, 50, 0, "R", fmt_si(v["R_true"], "Ω", 3), lpos=(190, 30, "middle"))
    ll = c.add("inductor", "L", 310, 50, 0, "L", fmt_si(v["L_true"], "H", 3), lpos=(310, 30, "middle"))
    cc = c.add("capacitor", "C", 420, 125, 90, "C", fmt_si(v["C_true"], "F", 3), lpos=(400, 121, "end"))
    cx = c.add("capacitor", "dC", 500, 125, 90, "ΔC (2번째 측정에 추가)", fmt_si(v["dC"], "F", 3), lpos=(530, 121, "start"))
    c.wire("w1", vs["a"], (90, 50), r["a"])
    c.wire("w2", r["b"], ll["a"])
    c.wire("w3", ll["b"], (420, 50), cc["a"])
    c.wire("w4", (420, 50), (500, 50), cx["a"])
    c.wire("w5", cc["b"], (420, 205), (90, 205), vs["b"])
    c.wire("w6", cx["b"], (500, 205), (420, 205))
    c.dot((420, 50), (420, 205))
    c.text(446, 44, "v_C (관측)", "node")
    c.probe("pi", "none", 250, 38, "right", "i")
    c.mode("on", "step 응답", ["V", "R", "L", "C", "w1", "w2", "w3", "w5"], "L·C가 ringing 주파수, R/L이 감쇠를 정한다 — 한 파형은 이 두 조합만 알려준다", dim=["dC"])
    return c


def run_identifiability(v: dict) -> Result:
    res = Result("EX11", "identifiability", "A (식별·calibration/holdout) + C (합성 측정 데이터 생성)")
    rng = np.random.default_rng(int(v["seed"]))
    L0, C0, R0, V, dC = v["L_true"], v["C_true"], v["R_true"], v["V_step"], v["dC"]
    sig = v["noise"] * V
    t = np.arange(1, int(v["n_samples"]) + 1) * v["t_window"] / int(v["n_samples"])
    y1, y1c = _synthetic_ringing(V, R0, L0, C0, t, sig, rng)
    y2, y2c = _synthetic_ringing(V, R0, L0, C0 + dC, t, sig, rng)
    # independent check of the data generator: exact engine vs closed form
    e_gen = float(np.max(np.abs(y1c - ref.rlc_step_response(t, V, L0, C0, R0)))) / V
    res.add_check(Check("합성 데이터 생성: 정확 엔진 vs 닫힌 식 step 응답", "PASS" if e_gen < 1e-9 else "FAIL", e_gen, "rel", 1e-9, path="행렬지수 RLC 해 vs V(1 − e^{−αt}(cos ω_d t + α/ω_d sin ω_d t))", independent=True))

    def fit(y_sets, x0, fixed_L=None, priorC=None):
        def resid(p):
            L = fixed_L if fixed_L is not None else float(np.exp(p[0]))
            C, R = float(np.exp(p[-2])), float(np.exp(p[-1]))
            r = [(_rlc_model(t, L, C + extra, R, V) - y) / sig for y, extra in y_sets]
            if priorC is not None:
                r.append(np.array([(C - priorC[0]) / priorC[1]]))
            return np.concatenate(r)

        # bounded trust-region fit in log-parameters: the non-identifiable direction must not run away
        centre = np.log([C0, R0]) if fixed_L is not None else np.log([L0, C0, R0])
        lo_b, hi_b = centre - math.log(1e3), centre + math.log(1e3)
        x0 = np.clip(np.asarray(x0, dtype=float), lo_b + 1e-9, hi_b - 1e-9)
        sol = least_squares(resid, x0, method="trf", bounds=(lo_b, hi_b), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        return sol

    # 1) single waveform, several starting points
    rows = []
    fits = []
    # realistic workflow: the ring frequency is read from the data first, so every start sits on
    # the measured L*C product (5 % off) and only the split between L and C differs
    for Ls in (L0 / 3.0, L0, 3.0 * L0):
        s = fit([(y1, 0.0)], np.log([Ls, 1.05 * L0 * C0 / Ls, 0.8 * R0 * Ls / L0]))
        L, C, R = np.exp(s.x)
        rms = float(np.sqrt(np.mean(s.fun**2))) * sig
        fits.append((L, C, R, rms, s))
        rows.append([Ls * 1e9, L * 1e9, C * 1e9, R, L * C * 1e18, R / L * 1e-6, rms * 1e3])
    Ls_fit = [f[0] for f in fits]
    spread = max(Ls_fit) / min(Ls_fit)
    rms_spread = (max(f[3] for f in fits) - min(f[3] for f in fits)) / min(f[3] for f in fits)
    res.tables.append(Table("t_fits", "한 waveform에 (L, C, R) 3개를 맞춘 결과: 시작점만 다르게", ["시작 L [nH]", "L [nH]", "C [nF]", "R [Ω]", "L·C [nH·nF]", "R/L [1/µs]", "잔차 RMS [mV]"], rows,
                            note="잔차가 같은데 L·C·R이 다르다. 파형이 정하는 것은 ω0 = 1/√(LC)와 α = R/(2L)뿐이다."))
    res.add_metric("L_spread", "시작점에 따른 적합 L의 최대/최소 비", spread, "", basis="같은 잔차")
    res.add_metric("rms_spread", "적합 잔차의 상대 차이", rms_spread, "", basis="≈ 0이면 구분 불가")
    LC = [f[0] * f[1] for f in fits]
    RL = [f[2] / f[0] for f in fits]
    res.add_check(Check("식별 가능한 조합: L·C와 R/L은 시작점과 무관", "PASS" if (max(LC) / min(LC) - 1) < 1e-5 and (max(RL) / min(RL) - 1) < 1e-3 else "FAIL", max(LC) / min(LC) - 1, "rel", 1e-5, path="서로 다른 시작점의 적합 결과 비교", independent=True, detail=f"L·C 편차 {max(LC) / min(LC) - 1:.1e}, R/L 편차 {max(RL) / min(RL) - 1:.1e}, L 자체는 ×{spread:.3g}"))
    # Jacobian singular values at one fit (log-parameter sensitivities)
    J = fits[1][4].jac
    sv = np.linalg.svd(J, compute_uv=False)
    res.add_metric("sv_ratio", "Jacobian 최소/최대 특이값 (log L, log C, log R)", float(sv[-1] / sv[0]), "", basis="≈ 0: 한 방향(L↑·C↓·R↑)으로 잔차가 변하지 않음")
    res.add_check(Check("Jacobian 계수 결손 (한 방향 비식별)", "PASS" if sv[-1] / sv[0] < 1e-6 else "FAIL", float(sv[-1] / sv[0]), "", 1e-6, path="적합점의 J = ∂잔차/∂log θ 특이값 분해", independent=True, detail="특이값: " + ", ".join(f"{s_:.3g}" for s_ in sv)))
    # 2) resolved: a second excitation with a known added capacitance
    sj = fit([(y1, 0.0), (y2, dC)], np.log([L0 * 2.0, C0 * 0.6, R0 * 1.5]))
    Lj, Cj, Rj = np.exp(sj.x)
    cov = np.linalg.inv(sj.jac.T @ sj.jac)
    sd = np.sqrt(np.diag(cov))  # log-parameter standard deviations (sigma known)
    res.add_metric("L_joint", "두 excitation 공동 적합 L", Lj, "H", ref=L0, ref_label="합성 참값", tol=max(0.05, 4 * sd[0]), basis=f"u(ln L) = {sd[0]:.2%}")
    res.add_metric("C_joint", "공동 적합 C", Cj, "F", ref=C0, ref_label="합성 참값", tol=max(0.05, 4 * sd[1]), basis=f"u(ln C) = {sd[1]:.2%}")
    res.add_metric("R_joint", "공동 적합 R", Rj, "Ω", ref=R0, ref_label="합성 참값", tol=max(0.1, 4 * sd[2]), basis=f"u(ln R) = {sd[2]:.2%}")
    zL = abs(math.log(Lj / L0)) / sd[0]
    res.add_check(Check("공동 적합이 참값을 불확도 안에서 회복", "PASS" if zL < 4 else "FAIL", zL, "σ", 4, path="합성 참값 vs 공동 적합 (J 공분산의 표준편차 단위)", independent=True, detail=f"L {Lj * 1e9:.4g} nH (참 {L0 * 1e9:g}), C {Cj * 1e9:.4g} nF, R {Rj:.4g} Ω"))
    # 3) resolved: an independent C measurement as a prior
    sp = fit([(y1, 0.0)], np.log([L0 * 2.0, C0 * 0.6, R0 * 1.5]), priorC=(C0 * (1 + v["C_meas_bias"]), v["C_meas_u"] * C0))
    Lp = math.exp(sp.x[0])
    res.add_metric("L_prior", "독립 C 측정(LCR)을 더한 L", Lp, "H", basis=f"C 측정 불확도 {v['C_meas_u']:.1%}, 편향 {v['C_meas_bias']:+.1%} → L도 그만큼", note="C 측정이 틀리면 L이 그 비율로 틀린다: 독립 측정의 조건(bias 전압·주파수)이 중요")
    # profile of the residual vs a fixed L
    Lgrid = np.geomspace(L0 / 5, L0 * 5, 21)
    prof1, prof2 = [], []
    for Lf in Lgrid:
        s1 = fit([(y1, 0.0)], np.log([C0 * L0 / Lf, R0 * Lf / L0]), fixed_L=Lf)
        s2 = fit([(y1, 0.0), (y2, dC)], np.log([C0 * L0 / Lf, R0 * Lf / L0]), fixed_L=Lf)
        prof1.append(float(np.sum(s1.fun**2)) / t.size)
        prof2.append(float(np.sum(s2.fun**2)) / (2 * t.size))
    res.add_series("prof1", "한 waveform: L 고정 후 (C, R) 최적 잔차", "", (Lgrid * 1e9).tolist(), prof1)
    res.add_series("prof2", f"두 excitation (C + {fmt_si(dC, 'F', 2)}): 공동 잔차", "", (Lgrid * 1e9).tolist(), prof2)
    res.add_plot("p_prof", "profile: L을 고정하고 나머지를 맞춘 잔차 (χ²/N)", ["prof1", "prof2"], x_label="L", x_unit="nH", y_label="χ²/N", y_unit="", kind="xy", log_x=True, log_y=True, level="A",
                 vlines=[{"x": L0 * 1e9, "label": "합성 참값"}],
                 proved="한 waveform의 profile은 평평하다(어떤 L이든 C·R을 바꾸면 같은 잔차). 알려진 C를 더한 두 번째 excitation은 뾰족한 최소를 만든다.",
                 not_yet="합성 데이터를 다시 맞춘 구현 검증이다. 실측에서는 probe loading·C의 전압 의존성·분포 layout이 모델 형식 오차로 들어온다.")
    res.add_series("y1", "측정(합성+잡음) waveform 1", "V", (t * 1e9).tolist(), y1.tolist(), style="points")
    res.add_series("y1fit", "적합 (어느 L이든 같은 곡선)", "V", (t * 1e9).tolist(), _rlc_model(t, fits[0][0], fits[0][1], fits[0][2], V).tolist())
    res.add_series("y2", f"waveform 2 (C + {fmt_si(dC, 'F', 2)})", "V", (t * 1e9).tolist(), y2.tolist(), style="points")
    res.add_plot("p_wave", "ringing waveform과 적합", ["y1", "y1fit", "y2"], x_label="t", x_unit="ns", y_label="v_C", y_unit="V", kind="xy", level="A",
                 proved="잡음 섞인 ringing에 (L, C, R) 모델이 잘 맞는다 — 그러나 잘 맞는다는 것이 유일한 물리 parameter를 뜻하지는 않는다.", not_yet="잘 맞는 곡선이 어느 (L, C, R) 조합인지는 이 파형만으로 정할 수 없다(위 profile). 실측 파형에는 probe 대역·deskew 오차도 섞인다.")
    # 4) calibration / holdout on a synthetic loss surface with a model-form term at the hot corner
    Ig = np.linspace(10, 100, 10)
    Tg = np.linspace(25, 150, 6)
    II, TT = np.meshgrid(Ig, Tg, indexing="ij")
    II, TT = II.ravel(), TT.ravel()
    a_, b_, c_, d_ = 4e-3, 6e-3, 0.5, v["model_form"]
    # a loss mechanism the calibration model lacks, active only at high current AND high temperature
    P_true = a_ * II**2 * (1 + b_ * (TT - 25)) + c_ * II + d_ * np.maximum(0.0, II - 50.0) ** 2 * np.maximum(0.0, TT - 110.0) / 40.0
    P_meas = P_true * (1 + v["meas_noise"] * rng.standard_normal(P_true.size))
    X = np.column_stack([II**2, II**2 * (TT - 25), II])

    def fit_eval(train, test):
        coef, *_ = np.linalg.lstsq(X[train], P_meas[train], rcond=None)
        pred = X @ coef
        e = (pred[test] - P_meas[test]) / P_meas[test]
        return float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e))), pred

    idx = rng.permutation(P_true.size)
    ntest = max(1, int(round(0.2 * P_true.size)))
    rtest, rtrain = idx[:ntest], idx[ntest:]
    corner = (TT >= v["corner_T"]) & (II >= v["corner_I"])
    ctest, ctrain = np.where(corner)[0], np.where(~corner)[0]
    r_rms, r_max, _ = fit_eval(rtrain, rtest)
    c_rms, c_max, pred_c = fit_eval(ctrain, ctest)
    res.add_metric("hold_random", "random 80/20 holdout 상대 RMS 오차", r_rms, "", basis=f"seed {int(v['seed'])}, 시험점 {ntest}개")
    res.add_metric("hold_corner", f"설계 corner holdout (T ≥ {v['corner_T']:g} °C, I ≥ {v['corner_I']:g} A) 상대 RMS 오차", c_rms, "", basis=f"시험점 {ctest.size}개 — 여기서 T_j 한계 결정")
    res.add_metric("hold_corner_max", "corner 최대 상대 오차", c_max, "", basis=f"결정에 쓰려면 허용 {v['tol_pred']:.0%} 이하여야 함 (판정은 결과 상태)")
    res.add_series("res_all", "모든 점: corner 제외 calibration의 예측 오차", "", TT.tolist(), ((pred_c - P_meas) / P_meas).tolist(), style="points")
    res.add_plot("p_hold", "holdout: 설계를 정하는 corner에서 오차가 드러난다", ["res_all"], x_label="온도", x_unit="°C", y_label="상대 예측 오차", y_unit="", kind="xy", level="A",
                 hlines=[{"y": v["tol_pred"], "label": "허용"}, {"y": -v["tol_pred"], "label": ""}], vlines=[{"x": v["corner_T"], "label": "corner"}],
                 proved="random 80/20은 같은 sweep 근처를 나눠 작은 오차를 보이지만, 설계를 결정하는 고온·대전류 corner를 떼어 두면 모델 형식 오차가 드러난다.",
                 not_yet="합성 손실면(모델 형식 항 포함)이다. 실측 calibration/validation은 다른 전류·온도·setup의 독립 데이터가 필요하다.")
    res.tables.append(Table("t_hold", "calibration / holdout 분리", ["방식", "학습 점", "시험 점", "상대 RMS 오차", "최대 오차", "해석"], [
        ["random 80/20", rtrain.size, ntest, r_rms, r_max, "같은 sweep 근처 → 외삽 위험을 평가하지 못함"],
        ["설계 corner holdout", ctrain.size, ctest.size, c_rms, c_max, "결정을 바꾸는 corner를 떼어 둠 → 모델 형식 오차 노출"],
    ], note="시뮬레이션 데이터를 다시 맞춘 것은 구현 검증에 유용하지만 실물 validation이 아니다(교재 E11)."))
    res.circuit = {"diagram": _rlc_circuit(v).to_json(), "intervals": [], "plot_group": ""}
    res.verdict("NOT_EVALUABLE", f"한 ringing waveform으로는 L·C·R을 따로 식별할 수 없다 (L 적합값 ×{spread:.3g}, 같은 잔차): L·C와 R/L만 식별된다")
    res.verdict("PASS_WITHIN_MODEL", "알려진 C를 더한 두 번째 excitation(또는 독립 C 측정)으로 L·C·R을 식별")
    if c_max > v["tol_pred"]:
        res.verdict("OUT_OF_VALIDITY", f"random holdout 오차 {r_rms:.1%}는 작지만 설계 corner 최대 오차 {c_max:.1%} > 허용 {v['tol_pred']:.0%}: corner에서는 calibration 범위 밖(모델 형식 오차)")
    res.assumptions += ["합성 ringing: 직렬 RLC step 응답 + 백색 잡음 (seed 명시)", "두 번째 excitation의 추가 C는 정확히 안다고 가정", "합성 손실면에 고온 모델 형식 항 포함; calibration 모델은 그 항이 없다"]
    res.not_valid_for += ["실측 파형의 parameter 추출 결과 (probe·layout·비선형 C 미포함)", "calibration 데이터 밖의 온도·전류 예측"]
    res.interpretation = (
        "ringing 주파수는 LC 곱을, 감쇠는 R/L을 알려줄 뿐이라 한 waveform에 L·C·R 세 개를 맞추면 서로 다른 조합이 같은 잔차를 준다. 알려진 C를 더해 주파수가 얼마나 바뀌는지 보거나 C를 독립 측정해야 L이 정해진다. "
        "마찬가지로 calibration 데이터를 random으로 나누면 잘 맞아 보여도, 설계를 결정하는 corner를 떼어 두면 모델이 그 corner를 설명하지 못한다는 것이 드러난다."
    )
    return res


# ======================================================================================
# Experiment 4: Monte Carlo reporting (distributions, correlation, seed, N, model bias)
# ======================================================================================


def run_monte_carlo(v: dict) -> Result:
    res = Result("EX11", "monte_carlo", "A (해석 lognormal 수율) + Monte Carlo")
    sL, sC, rho, tol, N, seed = v["sig_L"], v["sig_C"], v["rho_LC"], v["tol_f"], int(v["mc_n"]), int(v["seed"])
    bias = v["bias"]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N, 2))
    Lc = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]) + 1e-15 * np.eye(2))
    e = z @ Lc.T
    xL, xC = sL * e[:, 0], sC * e[:, 1]
    fr = np.exp(-0.5 * (xL + xC))  # f / f0 (lognormal L, C)

    def yield_of(fr_, b):
        return np.abs(fr_ * (1 + b) - 1) <= tol

    ok = yield_of(fr, 0.0)
    Y = float(np.mean(ok))
    npass = int(np.sum(ok))
    lo, hi = ref.clopper_pearson_two_sided(N - npass, N, 0.95)
    Y_lo, Y_hi = 1 - hi, 1 - lo
    lo3, hi3 = ref.clopper_pearson_two_sided(N - npass, N, 0.999)
    Y_lo3, Y_hi3 = 1 - hi3, 1 - lo3
    Ya = ref.lognormal_ratio_yield(sL, sC, rho, tol, 0.0)
    se_abs = math.sqrt(max(Ya * (1 - Ya), 1.0 / N) / N)
    res.add_metric("yield", "수율 (MC, 모델 편향 0)", Y, "", ref=Ya, ref_label="해석 lognormal 수율", tol=4 * se_abs, abs_scale=1.0, basis=f"95 % CI [{Y_lo:.4%}, {Y_hi:.4%}] (Clopper–Pearson); 판정 허용 4σ_MC = {4 * se_abs:.2%}")
    res.add_metric("yield_analytic", "해석 수율 Φ 식", Ya, "", basis="ln f 정규, σ_f² = (σ_L² + σ_C² + 2ρσ_Lσ_C)/4")
    res.add_metric("sigma_f", "f_r 상대 표준편차 (해석)", 0.5 * math.sqrt(sL**2 + sC**2 + 2 * rho * sL * sC), "", basis="ln 기준")
    for b in (-abs(bias), abs(bias)):
        yb = float(np.mean(yield_of(fr, b)))
        yba = ref.lognormal_ratio_yield(sL, sC, rho, tol, b)
        se_b = math.sqrt(max(yba * (1 - yba), 1.0 / N) / N)
        res.add_metric(f"yield_bias_{'m' if b < 0 else 'p'}", f"수율, 모델 형식 편향 {b:+.1%} (고정 corner, 표본 아님)", yb, "", ref=yba, ref_label="해석", tol=4 * se_b, abs_scale=1.0, basis="systematic bias는 random spread와 따로")
    res.add_metric("statement", "수율 문장", f"{Y:.2%} [{Y_lo:.2%}, {Y_hi:.2%}] — ln L, ln C ~ N(0, σ²), σ_L {sL:.1%}, σ_C {sC:.1%}, ρ {rho:g}, N {N}, seed {seed}, 편향 0 조건", "")
    res.add_check(Check("MC 수율 vs 해석 수율 (99.9 % CI 포함 여부)", "PASS" if Y_lo3 <= Ya <= Y_hi3 else "FAIL", Y - Ya, "", f"[{Y_lo3 - Y:.2e}, {Y_hi3 - Y:.2e}]", path="seed 고정 Cholesky 표본 vs ln f 정규 분포의 Φ 식", independent=True, detail=f"MC {Y:.5f} / 해석 {Ya:.5f}; 95 % 구간은 보고용(설계상 20번에 1번은 벗어난다), 검산은 99.9 % 구간"))
    # reproducibility (regression by construction)
    rng2 = np.random.default_rng(seed)
    z2 = rng2.standard_normal((N, 2))
    same = bool(np.array_equal(z, z2))
    res.add_check(Check("seed 재현성: 같은 seed → 같은 표본", "PASS" if same else "FAIL", 1.0 if same else 0.0, "", path="numpy default_rng(seed) 두 번 생성", independent=False, detail="회귀 성격 (같은 함수)"))
    # convergence of the running estimate
    Ns = np.unique(np.geomspace(50, N, 60).astype(int))
    run = np.cumsum(ok) / np.arange(1, N + 1)
    est = [float(run[k - 1]) for k in Ns]
    ci_lo, ci_hi = [], []
    for k, y in zip(Ns, est):
        lo_k, hi_k = ref.clopper_pearson_two_sided(int(round(k * (1 - y))), int(k), 0.95)
        ci_lo.append(1 - hi_k)
        ci_hi.append(1 - lo_k)
    res.add_series("conv", "누적 수율 추정", "", Ns.tolist(), est)
    res.add_series("conv_lo", "95 % CI 하한", "", Ns.tolist(), ci_lo, dash=True)
    res.add_series("conv_hi", "95 % CI 상한", "", Ns.tolist(), ci_hi, dash=True)
    res.add_plot("p_conv", "수렴: 표본 수에 따른 추정과 신뢰구간", ["conv", "conv_lo", "conv_hi"], x_label="표본 수 N", x_unit="", y_label="수율", y_unit="", kind="xy", log_x=True, level="MC",
                 hlines=[{"y": Ya, "label": "해석값"}],
                 proved="표본 수를 늘리면 추정이 해석값으로 수렴하고 신뢰구간이 1/√N로 좁아진다. N·seed·CI를 함께 보고한다.",
                 not_yet="분포 모양(lognormal)·σ·상관은 가정이다. 실제 부품 분포·lot 상관 자료가 없으면 수율은 가정 조건부 수치다.")
    hist, edges = np.histogram((fr - 1) * 100, bins=80)
    cx = 0.5 * (edges[:-1] + edges[1:])
    res.add_series("hist", "f_r 편차 분포 (MC)", "개", cx.tolist(), hist.tolist())
    res.add_plot("p_hist", "f_r 편차 분포와 사양 창", ["hist"], x_label="f_r / f_0 − 1", x_unit="%", y_label="표본 수", y_unit="", kind="xy", level="MC",
                 vlines=[{"x": -tol * 100, "label": f"−{tol:.0%}"}, {"x": tol * 100, "label": f"+{tol:.0%}"}],
                 proved="L·C의 곱이 f_r을 정하므로 양의 상관은 분포를 넓히고 음의 상관은 좁힌다.", not_yet="분포의 꼬리 모양(lognormal)은 가정이다. 사양 창 밖의 드문 부품 비율은 실제 lot 자료 없이는 말할 수 없다.")
    rows = []
    for r in (-0.9, -0.5, 0.0, 0.5, 0.9):
        Lr = np.linalg.cholesky(np.array([[1.0, r], [r, 1.0]]))
        er = z @ Lr.T
        frr = np.exp(-0.5 * (sL * er[:, 0] + sC * er[:, 1]))
        rows.append([r] + [float(np.mean(yield_of(frr, b))) for b in (-abs(bias), 0.0, abs(bias))] + [ref.lognormal_ratio_yield(sL, sC, r, tol, 0.0)])
    res.tables.append(Table("t_rho", f"상관·모델 편향에 따른 수율 (N = {N}, seed {seed}, 같은 표본)", ["ρ(L, C)", f"편향 {-abs(bias):+.0%}", "편향 0", f"편향 {abs(bias):+.0%}", "해석 (편향 0)"], rows,
                            note="모델 형식 오차(편향)는 random spread에 섞지 않고 고정 corner로 따로 본다. 공차 min/max만으로 독립 정규를 정하지 않는다(교재 E11)."))
    res.verdict("PASS_WITHIN_MODEL", f"수율 {Y:.2%}는 명시한 분포·상관·seed·N·편향 0 조건부이며 해석 값과 일치 — 양산 수율 보증이 아니다")
    res.assumptions += [f"ln L, ln C ~ 정규 (σ_L {sL:.1%}, σ_C {sC:.1%}, 상관 {rho:g})", f"사양: |f_r/f_0 − 1| ≤ {tol:.0%}", "모델 형식 편향은 ± 고정값 (표본 아님)"]
    res.not_valid_for += ["실제 부품 분포·lot 상관이 없는 양산 수율 주장", "희귀 실패(ppm)의 확률"]
    res.interpretation = (
        f"같은 공차라도 L과 C가 양의 상관이면 f_r이 더 넓게 퍼져 수율이 떨어진다. Monte Carlo 수율은 분포·상관·seed·표본 수를 함께 말해야 하는 조건부 수치다. "
        f"모델이 f_r을 {abs(bias):.0%} 틀리게 예측하는 편향은 무작위 퍼짐과 다른 성격이라 따로 corner로 본다."
    )
    return res


# ======================================================================================
# Experiment 5: test-independence map and model card, built from the repository itself
# ======================================================================================

_LABS_DIR = Path(__file__).resolve().parent
_REPO_TESTS = _LABS_DIR.parents[2] / "tests"


def _scan_checks(path: Path) -> dict:
    """Static scan: Check(...) calls with a literal independent flag, check_close(...) and ledger_check(...)."""
    out = {"indep": 0, "regr": 0, "dynamic": 0, "ledger": 0}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
        flag = None
        if name == "Check":
            kw = next((k for k in node.keywords if k.arg == "independent"), None)
            flag = False if kw is None else (kw.value.value if isinstance(kw.value, ast.Constant) else "dyn")
        elif name == "check_close":
            kw = next((k for k in node.keywords if k.arg == "independent"), None)
            val = kw.value if kw is not None else (node.args[5] if len(node.args) > 5 else None)
            flag = val.value if isinstance(val, ast.Constant) else "dyn"
        elif name == "ledger_check":
            out["ledger"] += 1
            flag = True
        else:
            continue
        if flag is True:
            out["indep"] += 1
        elif flag is False:
            out["regr"] += 1
        else:
            out["dynamic"] += 1
    return out


def _scan_tests(lab_id: str) -> dict | None:
    p = _REPO_TESTS / f"test_{lab_id.lower()}.py"
    if not p.exists():
        return None
    src = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    n_tests = sum(1 for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
    literals = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "approx" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, (int, float)):
                literals += 1
    doc = ast.get_docstring(tree) or ""
    lines = [ln.strip("* ").strip() for ln in doc.splitlines() if ln.strip().startswith("*")]
    return {"file": f"tests/{p.name}", "tests": n_tests, "literals": literals, "map": lines}


HINT_S = {"fast": 3.0, "seconds": 10.0, "slow": 60.0}  # runtime classes of the authoring contract (upper bounds)


def run_test_independence(v: dict) -> Result:
    from . import all_labs  # local import: the registry imports this module

    res = Result("EX11", "test_independence", "메타 (저장소의 검증 경로를 실행·스캔)")
    labs = all_labs()
    budget = v["budget_s"]
    t_start = time.perf_counter()
    order = []
    for lid in sorted(labs):
        for e in labs[lid].experiments:
            if lid == "EX11" and e.key == "test_independence":
                continue
            for pk in e.reference_presets or [e.presets[0].key]:
                order.append((0 if e.runtime_hint == "fast" else 1, lid, e, pk))
    order.sort(key=lambda r: (r[0], r[1], r[2].key))
    run_rows, lab_sum = [], {}
    for _, lid, e, pk in order:
        s = lab_sum.setdefault(lid, {"runs": 0, "skipped": 0, "checks": 0, "indep": 0, "regr": 0, "PASS": 0, "FAIL": 0, "NOT_RUN": 0, "INFO": 0, "mref": 0, "mref_fail": 0, "status": set()})
        # start a preset only if its runtime class still fits: a started preset may overrun the budget by
        # at most one fast preset (<= FAST_S by the authoring contract), so elapsed <= budget + FAST_S
        est = HINT_S.get(e.runtime_hint, HINT_S["slow"])
        if time.perf_counter() - t_start + est - HINT_S["fast"] > budget:
            s["skipped"] += 1
            run_rows.append([lid, e.key, pk, "—", f"예산 부족으로 미실행 (runtime_hint {e.runtime_hint} ≈ {est:g} s)", "—", "NOT_RUN", ""])
            continue
        try:
            vals, _, _ = resolve_params(e.params, e.presets, pk, {})
            r = e.run(vals)
        except Exception as exc:  # surfaced, never hidden
            s["runs"] += 1
            s["FAIL"] += 1
            run_rows.append([lid, e.key, pk, "실행 오류", str(exc)[:80], "—", "FAIL", ""])
            continue
        s["runs"] += 1
        s["status"].add(r.status)
        for c in r.checks:
            s["checks"] += 1
            s["indep" if c.independent else "regr"] += 1
            s[c.status if c.status in ("PASS", "FAIL", "NOT_RUN", "INFO") else "INFO"] += 1
            run_rows.append([lid, e.key, pk, c.name, c.path, "독립" if c.independent else "회귀", c.status, r.status])
        for m in r.metrics:
            if m.check in ("PASS", "FAIL"):
                s["mref"] += 1
                s["mref_fail"] += m.check == "FAIL"
    elapsed = time.perf_counter() - t_start
    rows = []
    fails = 0
    for lid in sorted(labs):
        s = lab_sum.get(lid)
        sc = _scan_checks(_LABS_DIR / Path(labs[lid].code_path).name)
        ts = _scan_tests(lid)
        if s is None:
            continue
        fails += s["FAIL"] + s["mref_fail"]
        rows.append([lid, s["runs"], s["skipped"], s["checks"], s["indep"], s["regr"], s["PASS"], s["FAIL"], s["NOT_RUN"], s["mref"], s["mref_fail"], f"{sc['indep']}/{sc['regr']}/{sc['dynamic']}", ts["tests"] if ts else "없음", ts["literals"] if ts else "—", ", ".join(sorted(s["status"]))])
    res.tables.append(Table("t_summary", "실습별 검증 경로 요약 (실행한 기준 preset + 소스 정적 스캔)", ["lab", "실행", "미실행", "check", "독립", "회귀", "PASS", "FAIL", "NOT_RUN", "기준값 metric", "metric FAIL", "소스 Check 독립/회귀/동적", "test 함수", "test 리터럴 approx", "결과 status"], rows,
                            note="‘독립’은 두 쪽이 공식을 공유하지 않는 check(닫힌 식 vs 적분·solver·에너지 항등식 등), ‘회귀’는 같은 함수를 두 번 부른 check다. 소스 스캔은 Check(independent=상수)·check_close·ledger_check 호출 수. test 리터럴은 pytest.approx(숫자) 개수(교과서 숫자를 test에 직접 적은 수)."))
    res.tables.append(Table("t_checks", "실행한 검증 항목 전체", ["lab", "실험", "preset", "check", "비교 경로", "독립/회귀", "판정", "결과 status"], run_rows))
    trows = []
    for lid in sorted(labs):
        ts = _scan_tests(lid)
        if ts:
            trows.append([lid, ts["file"], ts["tests"], ts["literals"], " · ".join(ts["map"][:6]) or "(독립성 표 없음)"])
    res.tables.append(Table("t_tests", "test 파일의 독립성 표 (docstring에서 추출)", ["lab", "파일", "test 수", "리터럴 기준값", "독립/회귀 선언"], trows,
                            note="tests/ 폴더가 없는 설치본에서는 비어 있다." if not trows else "각 test 파일 첫 docstring의 Independence map 줄을 그대로 옮겼다."))
    mrows = []
    for lid in sorted(labs):
        for e in labs[lid].experiments:
            mrows.append([lid, e.key, e.model_level, e.claim_limit, ", ".join(e.reference_presets)])
    res.tables.append(Table("t_card", "모델 카드: 실험별 모델 수준과 주장 한계", ["lab", "실험", "모델 수준", "주장 한계 (claim_limit)", "기준 preset"], mrows,
                            note="검증(verification) 통과는 방정식을 올바르게 풀었다는 뜻이다. 실측 validation(hardware)은 이 저장소 어디에서도 수행하지 않았다."))
    tot_runs = sum(s["runs"] for s in lab_sum.values())
    tot_skip = sum(s["skipped"] for s in lab_sum.values())
    tot_checks = sum(s["checks"] for s in lab_sum.values())
    tot_ind = sum(s["indep"] for s in lab_sum.values())
    res.add_metric("labs", "발견한 실습 수", len(labs), "", basis=", ".join(sorted(labs)))
    res.add_metric("runs", "실행한 기준 preset", tot_runs, "", basis=f"예산 {budget:g} s 중 {elapsed:.1f} s 사용; 미실행 {tot_skip}")
    res.add_metric("elapsed", "실행 시간", elapsed, "s", basis=f"상한 = 예산 {budget:g} s + fast preset 하나(≤ {HINT_S['fast']:g} s)")
    res.add_metric("checks", "실행된 check 수", tot_checks, "")
    res.add_metric("indep_frac", "독립 check 비율", tot_ind / tot_checks if tot_checks else float("nan"), "")
    res.add_metric("fails", "실패 check + 기준값 metric 실패", fails, "")
    xs = list(range(1, len(rows) + 1))
    res.add_series("n_ind", "독립 check", "개", xs, [r[4] for r in rows], style="points")
    res.add_series("n_reg", "회귀 check", "개", xs, [r[5] for r in rows], style="points")
    res.add_plot("p_map", "실습별 독립 vs 회귀 check 수 (x = 표의 lab 순서)", ["n_ind", "n_reg"], x_label="lab 순번", x_unit="", y_label="check 수", y_unit="", kind="xy", level="메타",
                 markers=[{"x": i + 1, "y": r[4], "label": r[0]} for i, r in enumerate(rows)],
                 vlines=[{"x": 0.5, "label": ""}, {"x": len(rows) + 0.7, "label": ""}],  # room for the last label
                 proved="각 실습의 핵심 수치가 최소 한 개의 독립 경로(공식을 공유하지 않는 비교)로 확인되는지를 실행 결과에서 직접 셌다.",
                 not_yet="독립 check도 같은 가정(합성 입력·이상 소자)을 공유하므로 실물 validation이 아니다. 예산 안에서 실행하지 못한 preset은 정적 스캔만 있다.")
    if tot_skip:
        res.warnings.append(f"시간 예산 {budget:g} s 안에서 {tot_skip}개 기준 preset을 실행하지 못했다 (budget_s를 늘리면 모두 실행).")
        res.verdict("NOT_EVALUABLE", f"예산 {budget:g} s 안에서 실행하지 못한 기준 preset {tot_skip}개는 판정하지 않았다 — 정적 스캔만 있다 (‘full’ preset은 전부 실행, 수 분 소요)")
    if fails:
        res.verdict("FAIL_CONSTRAINT", f"실행한 검증 중 {fails}개가 실패했다 — 해당 실습의 수치 검증을 먼저 고친다")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"실행한 {tot_runs}개 기준 preset의 {tot_checks}개 check가 모두 통과·정보 (독립 {tot_ind}개){f', 미실행 {tot_skip}개는 제외' if tot_skip else ''}. 실물 validation은 아니다.")
    res.assumptions += ["기준 preset만 실행 (fast 실험 먼저, 시간 예산 안에서)", "소스 스캔은 Check(independent=상수)만 셈 — 계산된 flag는 ‘동적’"]
    res.not_valid_for += ["hardware validation", "test가 실제로 얼마나 독립인지의 의미 판정 (선언과 구조만 셈)"]
    res.interpretation = (
        "이 표는 저장소가 스스로 무엇을 어떤 경로로 확인했는지 보여준다. 닫힌 식과 구간 적분, 행렬지수 해와 손으로 쓴 ODE, 에너지 항등식처럼 공식을 공유하지 않는 비교가 ‘독립’이고, "
        "같은 함수를 두 번 부른 비교는 ‘회귀’다. 모두 통과해도 이것은 verification이며, 실물 데이터로 모델의 목적 적합성을 보인 validation은 아니다."
    )
    return res


# ======================================================================================
# Lab definition
# ======================================================================================

_Q = [
    Question(
        "효율 98.0 %와 98.1 % 중 어떤 소자를 고르겠나?",
        "0.1 %p 차이를 측정 불확도·반복성·mission 가중과 비교한다. 입력·출력을 각각 0.1 % 표준불확도로 재면 11 kW에서 손실 불확도는 15.4 W(손실의 7 %), 효율로 0.139 %p다. 차이가 불확도보다 작으면 동률/미확정(UNRESOLVED_RANKING)으로 두고, 같은 setup 교대 측정이나 열 시험처럼 분별력 있는 시험, 또는 다른 요구사항으로 결정한다. 조건·보조전원 경계·온도가 같아야 한다.",
        "Efficiency 98.0 % versus 98.1 %: which device do you choose?",
        "I compare the 0.1 point difference with the measurement uncertainty, repeatability and mission weighting. With 0.1 percent standard uncertainty on input and output at 11 kilowatts, the loss uncertainty is about 15.4 watts, seven percent of the loss, or 0.139 points of efficiency. If the difference is smaller than that, I report the ranking as unresolved and decide with a more discriminating test or other requirements.",
        ["15.4 W", "0.139 %p", "UNRESOLVED_RANKING", "분별력 있는 시험"],
        kind="pressure",
    ),
    Question(
        "1000개 Monte Carlo가 다 통과했으니 양산에 안전한가?",
        "아니다. IID·가정 분포 아래에서도 0/1000의 one-sided 95 % 상한은 약 0.299 %다. ‘실패확률 0’이나 ppm 수준을 말할 수 없다. 분포·상관·seed·N을 명시하고, 제조 상관·systematic bias·모델 형식 오차·희귀 corner를 따로 다루며, hardware validation과 공정 자료로 연결한다. 결정적 grid에는 이 bound를 붙이지 않는다.",
        "All 1,000 Monte Carlo runs passed. Is it safe for production?",
        "No. Even under IID assumptions, zero failures in 1,000 trials only bounds the failure probability below about 0.3 percent at 95 percent confidence one-sided. Correlated manufacturing variation, systematic bias, model-form error and rare corners need separate treatment, and a deterministic grid gets no such bound at all.",
        ["0.299 %", "IID 가정", "systematic bias", "grid에 bound 없음"],
        kind="pressure",
    ),
    Question(
        "시뮬레이션과 측정이 10 % 다르다. 모델을 보정할까?",
        "먼저 단위·경계·probe deskew·시간창·온도·초기상태를 맞춘다. 그다음 민감도가 큰 parameter의 식별 가능성과 측정 불확도를 본다(ringing 주파수 하나는 L·C 곱만 알려준다). 한 곡선을 맞추는 임의 factor보다 독립 holdout, 특히 설계를 결정하는 corner에서 설명되는 물리 parameter와 모델 형식 보정을 고른다.",
        "Simulation and measurement differ by 10 %. Should you tune the model?",
        "First I align units, boundaries, probe deskew, time windows, temperature and initial state. Then I check whether the sensitive parameters are identifiable from the data and how large the measurement uncertainty is; a ringing frequency alone only gives the L times C product. I prefer physical parameters and model-form corrections that hold on an independent holdout, especially at the design-deciding corner, over an arbitrary factor that fits one curve.",
        ["경계·deskew 먼저", "식별 가능성", "holdout at corner", "임의 factor 금지"],
        kind="pressure",
    ),
    Question(
        "verification, calibration, validation, applicability는 무엇이 다른가?",
        "verification: 방정식을 코드로 올바르게 풀었는가(닫힌 식 vs 독립 적분·solver·에너지 항등식). calibration: parameter를 어떤 데이터에 맞췄는가. validation: 독립 실제 데이터에서 목적에 맞는가. applicability: 지금 질문이 모델 범위 안인가(예: FHA gain으로 startup을 설명하지 않음). 이 저장소의 check 통과는 verification이다.",
        "What is the difference between verification, calibration, validation and applicability?",
        "Verification asks whether the equations are solved correctly, for example a closed form against an independent integration. Calibration is which data the parameters were fitted to. Validation is whether the model serves its purpose on independent real data. Applicability is whether the current question is inside the model's range. Passing checks in this repository is verification only.",
        ["verification", "calibration", "validation", "applicability"],
    ),
]

EXPERIMENTS = [
    Experiment(
        key="loss_uncertainty",
        title="11 kW에서 손실 220 W의 불확도: 15.40 W, 상관 0.9면 4.87 W — 그 상관의 근거는?",
        goal=(
            "P_in 11000 W, P_out 10780 W를 각각 0.1 % 표준불확도로 재면 손실 220 W의 u = 15.40157 W(독립), ρ = 0.9 가정이면 4.87487 W다. η = 0.98의 u = 0.001386(0.139 %p). "
            "GUM 식을 seed가 명시된 Monte Carlo와 유한차분으로 따로 확인하고, 계측기 사양 해석(1σ·한계·k = 2)과 상관의 근거 기록, 후보 A/B 5 W 차이의 순위 판정(UNRESOLVED_RANKING)을 본다."
        ),
        params=[
            Param("P_in", "입력 전력 P_in", "W", 11000.0, "W", vmin=1, vmax=1e7, source="TEXTBOOK", source_note="11 kW"),
            Param("P_out", "출력 전력 P_out", "W", 10780.0, "W", vmin=1, vmax=1e7, source="TEXTBOOK", source_note="10.78 kW"),
            Param("rel_u_in", "P_in 상대 불확도 수치", "", 1e-3, "", vmin=0.0, vmax=0.2, source="TEXTBOOK", source_note="0.1 %"),
            Param("rel_u_out", "P_out 상대 불확도 수치", "", 1e-3, "", vmin=0.0, vmax=0.2, source="TEXTBOOK", source_note="0.1 %"),
            Param("spec_type", "그 수치의 의미", "", "standard", kind="choice", choices=[("standard", "표준불확도 1σ (교재 가정)"), ("rect_limit", "±한계, 직사각 분포 → a/√3"), ("k2", "확장불확도 k = 2 → a/2")], source="TEXTBOOK", source_note="‘accuracy ±0.1 %’를 그대로 1σ로 쓰지 않는다"),
            Param("rho", "P_in·P_out 오차 상관 ρ", "", 0.0, "", vmin=-1.0, vmax=1.0, source="ASSUMED", source_note="교재: ρ = 0.9 가정 예"),
            Param("rho_basis", "상관의 근거", "", "none", kind="choice", choices=[("none", "근거 기록 없음"), ("same_meter", "같은 분석기·range·동시 취득"), ("cal_cert", "교정 성적서 공분산")], source="ASSUMED"),
            Param("dP_AB", "후보 A/B 손실 차이", "W", 5.0, "W", vmin=-1e4, vmax=1e4, source="TEXTBOOK", source_note="교재 5 W 예", group="순위"),
            Param("rho_AB", "후보 간 오차 상관 ρ_AB", "", 0.0, "", vmin=-1.0, vmax=1.0, source="ASSUMED", source_note="따로 측정하면 0", group="순위"),
            Param("k_cov", "판정 coverage factor k", "", 2.0, "", vmin=0.5, vmax=5, source="ASSUMED", group="순위"),
            Param("mc_n", "Monte Carlo 표본 수", "", 200000, "", vmin=1000, vmax=5_000_000, kind="int", source="ASSUMED", group="Monte Carlo"),
            Param("seed", "난수 seed", "", 20260930, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group="Monte Carlo"),
        ],
        presets=[
            Preset("textbook", "교재: 각 0.1 % 독립", {}, "15.40157 W", ("nominal", "reference")),
            Preset("rho09_nobasis", "ρ = 0.9, 근거 없음", {"rho": 0.9}, "4.87487 W — 근거 없이 주장 불가", ("reference", "failure")),
            Preset("rho09_basis", "ρ = 0.9, 같은 분석기 동시 취득", {"rho": 0.9, "rho_basis": "same_meter"}, "근거 기록", ("variant",)),
            Preset("rect", "±0.1 %가 한계(직사각)", {"spec_type": "rect_limit"}, "u = a/√3", ("variant",)),
            Preset("paired", "같은 setup 교대 측정 (ρ_AB = 0.995)", {"rho_AB": 0.995}, "공통 오차 상쇄 → 5 W를 가릴 수 있음", ("variant",)),
        ],
        run=run_loss_uncertainty,
        model_level="A + MC",
        suggested_change="P_in·P_out 오차 상관 ρ를 0 → 0.9로 바꾼다 (근거는 아직 기록하지 않는다).",
        prediction=Prediction(
            "ρ를 0 → 0.9로 바꾸면 손실 불확도와 판정 상태는?",
            ["15.4 → 4.87 W, 그대로 주장 가능", "15.4 → 4.87 W지만 근거가 없어 MISSING_INPUT", "변화 없음", "모르겠다"],
            "15.4 → 4.87 W지만 근거가 없어 MISSING_INPUT",
            "u² = u_in² + u_out² − 2ρu_in u_out이라 양의 상관은 차이의 불확도를 줄인다. 그러나 같은 계측기·같은 range·동시 취득 같은 근거를 기록하지 않으면 줄어든 error bar를 주장할 수 없다. 두 후보의 5 W 차이는 여전히 판정 불가다.",
            ["u_loss_rho", "u_loss_ind"],
            handcalc=[{"key": "u_loss_ind", "label": "u(loss) 독립", "unit": "W"}, {"key": "u_eta", "label": "u(η)", "unit": ""}],
        ),
        suggested={"rho": 0.9},
        student="효율 98 %와 98.1 %는 작은 차이처럼 보이지만, 손실로 바꾸면 220 W와 209 W다. 두 전력계가 각각 0.1 %씩 틀릴 수 있다면 그 차이(손실)는 약 15 W만큼 틀릴 수 있어, 10 W 차이도 가릴 수 없다.",
        expert=(
            "① 차이의 불확도는 상관에 민감하다: 같은 분석기·같은 shunt range·동시 취득이면 교정 성분이 공유되어 ρ > 0이 정당화될 수 있지만, 근거를 기록하지 않으면 독립으로 둔다. "
            "② 사양의 ‘±0.1 %’는 보통 한계(reading/range 항 포함)이지 1σ가 아니다 — 분포 가정에 따라 u가 a/√3, a/2로 바뀐다. "
            "③ 후보 순위는 차이의 확장불확도와 비교해 판정하고, 부족하면 같은 setup 교대 측정(ρ_AB → 1), 열평형·보조전원 경계 통일, 손실을 직접 재는 calorimetric 방법을 제안한다."
        ),
        customer_ko="두 전력계의 불확도 0.1 %만으로도 손실 220 W의 불확도가 약 15 W라, 후보 간 5 W 차이는 이 시험으로 가릴 수 없습니다. 같은 분석기로 교대 측정하거나 열 시험처럼 분별력이 있는 방법으로 다시 비교하시죠.",
        customer_en="With 0.1 percent uncertainty on each power reading, the uncertainty of the 220 watt loss is already about 15 watts, so a 5 watt difference between the candidates cannot be resolved by this test. Let's compare again with alternating measurements on the same analyser or a more discriminating method such as a thermal test.",
        questions=[_Q[0]],
        textbook=[TB_E11, TB_E12],
        reference_presets=["textbook", "rho09_nobasis"],
        claim_limit="1차 GUM 전파와 MC의 합성 예. 실제 계측기 사양 해석·교정 자료는 MISSING_INPUT.",
    ),
    Experiment(
        key="binomial_bounds",
        title="0 fail / 1000 IID의 95 % 상한 0.2991 % — 그리고 결정적 grid에는 bound를 붙이지 않는다",
        goal=(
            "IID 1000회에서 실패 0이면 one-sided 95 % 상한은 1 − 0.05^(1/1000) = 0.2991 %다. 실패 k개일 때 Clopper–Pearson 상한을 Beta 분위수와 이항 누적합의 근으로 따로 계산한다. "
            "같은 합성 문제를 결정적 grid로 돌리면 실패 비율은 격자 범위가 정하는 숫자라 확률이 아니며, binomial bound를 붙이지 않는다(NOT_EVALUABLE)."
        ),
        params=[
            Param("n_trials", "시행 수 n", "", 1000, "", vmin=1, vmax=10_000_000, kind="int", source="TEXTBOOK", source_note="1000"),
            Param("k_fail", "관측 실패 수 k", "", 0, "", vmin=0, vmax=10_000_000, kind="int", source="TEXTBOOK", source_note="0"),
            Param("conf", "one-sided 신뢰수준", "", 0.95, "", vmin=0.5, vmax=0.999999, source="TEXTBOOK", source_note="95 %"),
            Param("sampling", "표본의 성격", "", "iid", kind="choice", choices=[("iid", "IID 무작위 시행"), ("grid", "결정적 grid (corner sweep)")], source="TEXTBOOK", source_note="grid에는 bound 금지"),
            Param("g_threshold", "합성 실패 임계 (θ1+θ2+θ3 >)", "", 6.0, "", vmin=0.1, vmax=20, source="ASSUMED", group="합성 문제"),
            Param("grid_span", "grid 범위 ±σ", "", 3.0, "", vmin=0.5, vmax=10, source="ASSUMED", group="합성 문제"),
            Param("seed", "난수 seed", "", 7, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group="합성 문제"),
        ],
        presets=[
            Preset("textbook", "0 / 1000, 95 %", {}, "0.2991 %", ("nominal", "reference")),
            Preset("k2", "2 / 1000", {"k_fail": 2}, "Clopper–Pearson", ("variant",)),
            Preset("grid", "결정적 grid 1000점", {"sampling": "grid"}, "bound 금지", ("failure", "reference")),
            Preset("million", "0 / 1 000 000", {"n_trials": 1000000}, "ppm 수준에 필요한 시행 수", ("corner",)),
        ],
        run=run_binomial,
        model_level="A",
        suggested_change="표본의 성격을 IID → 결정적 grid로 바꾼다.",
        prediction=Prediction(
            "1000번 IID 시행에서 실패가 0이면 95 % 신뢰로 말할 수 있는 실패확률은?",
            ["0 %", "0.1 % 이하", "약 0.3 % 이하", "모르겠다"],
            "약 0.3 % 이하",
            "(1 − p)^1000 = 0.05를 풀면 p = 1 − 0.05^(1/1000) = 0.2991 %(rule of three 3/n ≈ 0.3 %). 실패 0은 ‘0 %’가 아니다. 결정적 grid라면 이 계산 자체가 성립하지 않는다.",
            ["p_upper", "rule3"],
            handcalc=[{"key": "p_upper", "label": "95 % 상한", "unit": ""}],
        ),
        suggested={"sampling": "grid"},
        student="동전을 1000번 던져 앞면이 한 번도 안 나왔다고 앞면 확률이 0이라고 할 수는 없다. 확률이 0.3 % 정도만 되어도 1000번 중 한 번도 안 나올 가능성이 5 %는 되기 때문이다.",
        expert=(
            "Clopper–Pearson 상한은 P(X ≤ k | p) = 1 − conf를 푸는 p이며 Beta(k+1, n−k) 분위수와 같다. IID·동일 분포가 전제다. 결정적 grid·corner sweep은 분포에서 뽑은 표본이 아니므로 "
            "실패 비율을 확률로 읽거나 binomial bound를 붙이지 않는다 — grid는 실패 mechanism과 경계를 찾는 도구다. ppm 수준 주장은 수백만 IID 시행이나 mechanism 기반 가속시험이 필요하다."
        ),
        customer_ko="1000번 시뮬레이션에서 실패가 없었다는 것은 ‘실패확률 약 0.3 % 이하(95 %)’라는 뜻이지 0이라는 뜻이 아닙니다. corner sweep 결과는 확률로 쓰지 않고, 경계 근처는 목표 시험으로 따로 확인하시죠.",
        customer_en="No failures in 1,000 simulated runs means the failure probability is below about 0.3 percent at 95 percent confidence, not that it is zero. A corner sweep is not a probability, so let's check the region near the boundary with targeted tests.",
        questions=[_Q[1]],
        textbook=[TB_E11, TB_E12],
        reference_presets=["textbook", "grid"],
        claim_limit="IID Bernoulli 가정의 통계 상한. 양산 불량률 보증이 아니다.",
    ),
    Experiment(
        key="identifiability",
        title="ringing 하나로는 L과 C를 따로 알 수 없다 — 식별 가능성과 calibration/holdout",
        goal=(
            "정확 엔진으로 만든 합성 ringing(잡음 포함)에 (L, C, R)을 맞추면 시작점마다 L이 다르지만 잔차는 같다: 파형은 L·C와 R/L만 정한다(Jacobian 계수 결손, 평평한 profile). "
            "알려진 C를 더한 두 번째 excitation이나 독립 C 측정으로 식별을 회복한다. 이어서 합성 손실면에서 random 80/20 holdout과 설계 결정 corner holdout을 비교한다."
        ),
        params=[
            Param("L_true", "합성 참값 L", "H", 10e-9, "nH", vmin=1e-12, vmax=1e-3, source="TEXTBOOK", source_note="E09 예 10 nH", group="ringing"),
            Param("C_true", "합성 참값 C", "F", 1e-9, "nF", vmin=1e-15, vmax=1e-3, source="TEXTBOOK", source_note="E09 예 1 nF", group="ringing"),
            Param("R_true", "합성 참값 R", "Ω", 0.3, "Ω", vmin=1e-4, vmax=100, source="ASSUMED", group="ringing"),
            Param("V_step", "step 전압", "V", 1.0, "V", vmin=1e-3, vmax=1e4, source="ASSUMED", group="ringing"),
            Param("noise", "측정 잡음 σ / V", "", 0.01, "", vmin=1e-6, vmax=0.5, source="ASSUMED", group="ringing"),
            Param("n_samples", "표본 수", "", 600, "", vmin=50, vmax=20000, kind="int", source="ASSUMED", group="ringing"),
            Param("t_window", "관측 시간창", "s", 150e-9, "ns", vmin=1e-9, vmax=1e-3, source="ASSUMED", group="ringing"),
            Param("dC", "두 번째 excitation: 추가 C (정확히 앎)", "F", 1e-9, "nF", vmin=1e-15, vmax=1e-3, source="ASSUMED", group="식별 회복"),
            Param("C_meas_u", "독립 C 측정 불확도", "", 0.02, "", vmin=1e-4, vmax=0.5, source="ASSUMED", group="식별 회복"),
            Param("C_meas_bias", "독립 C 측정 편향 (bias 조건 차이)", "", 0.0, "", vmin=-0.5, vmax=0.5, source="ASSUMED", group="식별 회복"),
            Param("model_form", "합성 손실면의 고전류·고온 모델 형식 항 계수", "", 4e-3, "", vmin=0.0, vmax=1.0, source="ASSUMED", source_note="d·(I−50)²·(T−110)/40 W, 둘 다 넘을 때만", group="holdout"),
            Param("meas_noise", "손실 측정 잡음 (상대)", "", 0.005, "", vmin=0.0, vmax=0.2, source="ASSUMED", group="holdout"),
            Param("corner_T", "설계 corner 온도 ≥", "°C", 125.0, "°C", vmin=25, vmax=200, source="ASSUMED", group="holdout"),
            Param("corner_I", "설계 corner 전류 ≥", "A", 70.0, "A", vmin=1, vmax=1000, source="ASSUMED", group="holdout"),
            Param("tol_pred", "결정에 쓸 예측 허용 오차", "", 0.05, "", vmin=0.001, vmax=1.0, source="ASSUMED", group="holdout"),
            Param("seed", "난수 seed", "", 11, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED"),
        ],
        presets=[
            Preset("nominal", "10 nH·1 nF 합성 ringing, 잡음 1 %", {}, "", ("nominal", "reference")),
            Preset("low_noise", "잡음 0.1 %", {"noise": 0.001}, "잡음이 작아도 비식별은 그대로", ("variant",)),
            Preset("c_bias", "독립 C 측정이 +10 % 편향", {"C_meas_bias": 0.10}, "L이 그만큼 틀린다", ("variant",)),
        ],
        run=run_identifiability,
        model_level="A + C",
        suggested_change="측정 잡음을 1 % → 0.1 %로 줄인다.",
        prediction=Prediction(
            "잡음을 1 % → 0.1 %로 줄이면 한 waveform으로 L을 따로 알 수 있게 되나?",
            ["예, 잡음이 문제였다", "아니다, L·C 곱만 정해진다", "C만 알 수 있다", "모르겠다"],
            "아니다, L·C 곱만 정해진다",
            "파형은 ω0 = 1/√(LC)와 α = R/(2L)에만 의존한다. L을 k배, C를 1/k배, R을 k배로 바꾸면 파형이 똑같다 — 잡음과 무관한 구조적 비식별이다. 다른 excitation(알려진 C 추가)이나 독립 측정이 필요하다.",
            ["L_spread", "sv_ratio"],
        ),
        suggested={"noise": 0.001},
        student="같은 소리를 내는 기타 줄은 여러 가지다: 줄의 장력과 무게를 함께 바꾸면 같은 음이 난다. ringing 주파수도 L과 C의 곱만 알려줘서, 한 파형만으로는 둘을 따로 알 수 없다.",
        expert=(
            "비식별은 잡음의 문제가 아니라 모델 구조의 문제다: 잔차의 Jacobian이 한 방향(Δln L = −Δln C = Δln R)으로 0이다. 해결은 정보 추가 — 알려진 C를 더한 두 번째 excitation(주파수 비로 C가 정해짐), "
            "독립 C 측정(bias 조건이 다르면 그만큼 L이 틀림), 다른 측정 위치·전압·온도. calibration/validation은 분리하고, 설계를 결정하는 corner를 holdout으로 둔다. random 80/20은 같은 sweep 근처라 외삽 위험을 평가하지 못한다."
        ),
        customer_ko="측정 ringing 주파수로 loop 인덕턴스를 역산하시려면 커패시턴스를 따로 알아야 합니다. 알려진 커패시터를 추가해 주파수가 얼마나 바뀌는지 한 번 더 측정하면 L과 C를 분리할 수 있습니다.",
        customer_en="To back out the loop inductance from the ringing frequency you also need the capacitance. If you add a known capacitor and measure how much the frequency shifts, L and C can be separated.",
        questions=[_Q[2], _Q[3]],
        textbook=[TB_E11],
        reference_presets=["nominal"],
        runtime_hint="seconds",
        claim_limit="합성 데이터로 보인 식별 가능성·holdout 원리. 실측 parameter 추출이나 validation이 아니다.",
    ),
    Experiment(
        key="monte_carlo",
        title="Monte Carlo 보고: 분포·상관·seed·N·수렴, 모델 편향은 따로",
        goal=(
            "공진 tank f_r = 1/(2π√(LC))의 수율을 L·C lognormal 분포(σ, 상관 ρ), seed, 표본 수, 수렴과 함께 보고하고, 해석 수율(ln f 정규)과 신뢰구간으로 대조한다. "
            "모델 형식 편향 ±2 %는 random spread에 섞지 않고 고정 corner로 따로 본다. 수율 문장은 가정 조건부로 쓴다."
        ),
        params=[
            Param("sig_L", "ln L 표준편차 σ_L", "", 0.025, "", vmin=0.0, vmax=0.5, source="ASSUMED", source_note="±5 % 공차를 2σ로 본 합성값 — 근거 없으면 가정"),
            Param("sig_C", "ln C 표준편차 σ_C", "", 0.025, "", vmin=0.0, vmax=0.5, source="ASSUMED"),
            Param("rho_LC", "L·C 상관 ρ", "", 0.0, "", vmin=-0.99, vmax=0.99, source="ASSUMED", source_note="lot·온도 공통 요인"),
            Param("tol_f", "f_r 사양 창 ±", "", 0.04, "", vmin=0.001, vmax=0.5, source="ASSUMED"),
            Param("bias", "모델 형식 편향 크기 (±)", "", 0.02, "", vmin=0.0, vmax=0.3, source="ASSUMED"),
            Param("mc_n", "표본 수 N", "", 20000, "", vmin=100, vmax=5_000_000, kind="int", source="ASSUMED"),
            Param("seed", "난수 seed", "", 42, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED"),
        ],
        presets=[
            Preset("nominal", "σ 2.5 %, ρ 0, 사양 ±4 %", {}, "", ("nominal", "reference")),
            Preset("corr_pos", "ρ = +0.8", {"rho_LC": 0.8}, "양의 상관은 수율을 낮춘다", ("variant",)),
            Preset("corr_neg", "ρ = −0.8", {"rho_LC": -0.8}, "음의 상관은 f_r 분포를 좁힌다", ("variant",)),
            Preset("small_n", "N = 500", {"mc_n": 500}, "넓은 신뢰구간", ("corner",)),
        ],
        run=run_monte_carlo,
        model_level="A + MC",
        suggested_change="L·C 상관을 0 → +0.8로 바꾼다.",
        prediction=Prediction(
            "L과 C의 오차가 양의 상관(ρ = +0.8)이면 f_r 수율은?",
            ["올라간다", "내려간다", "그대로", "모르겠다"],
            "내려간다",
            "f_r은 L·C 곱으로 정해진다. 둘이 같은 방향으로 움직이면 곱의 퍼짐이 커져(σ_f² = (σ_L² + σ_C² + 2ρσ_Lσ_C)/4) 사양 창 밖이 늘어난다. 음의 상관이면 서로 상쇄해 좁아진다.",
            ["yield", "sigma_f"],
            handcalc=[{"key": "sigma_f", "label": "f_r 상대 σ", "unit": ""}],
        ),
        suggested={"rho_LC": 0.8},
        student="여러 부품의 오차가 동시에 한 방향으로 치우치면(같은 lot, 같은 온도) 결과가 더 크게 벗어난다. 그래서 오차가 독립인지 함께 움직이는지가 수율을 바꾼다.",
        expert=(
            "보고 항목: 분포 모양과 근거, 상관(Cholesky), seed, N, 신뢰구간(Clopper–Pearson), 수렴 곡선. 공차 min/max만으로 독립 정규를 정하지 않는다. "
            "모델 형식 오차·systematic bias·이산 corner는 random spread와 따로 관리한다 — 편향 ±2 %의 수율을 따로 적는다. 수율 문장은 항상 ‘가정 조건부’다."
        ),
        customer_ko="현재 공차 가정으로 f_r 수율은 약 90 %이지만, 이는 L·C 분포와 상관을 가정한 조건부 수치입니다. 실제 lot 분포와 온도 상관 자료를 받아 다시 계산하고, 모델 오차 ±2 %의 영향도 따로 보시죠.",
        customer_en="Under the assumed tolerances the resonant-frequency yield is about 90 percent, but that is conditional on the assumed L and C distributions and their correlation. Let's recompute with real lot data and temperature correlation, and show the effect of a plus or minus 2 percent model error separately.",
        questions=[_Q[1]],
        textbook=[TB_E11],
        reference_presets=["nominal", "corr_pos"],
        claim_limit="가정 분포 조건부 수율. 양산 수율 보증이 아니다.",
    ),
    Experiment(
        key="test_independence",
        title="이 저장소의 검증은 어떤 경로로 독립인가: test-independence 표와 모델 카드",
        goal=(
            "구현된 모든 실습의 기준 preset을 실제로 실행해 Check의 독립/회귀 flag와 판정을 모으고, 소스(Check 호출)와 tests/(리터럴 기준값, 독립성 선언)를 정적 스캔해 "
            "실습별 test-independence 표와 모델 카드(모델 수준·주장 한계)를 만든다. 통과는 verification이며 hardware validation이 아니다."
        ),
        params=[
            Param("budget_s", "실행 시간 예산", "s", 8.0, "s", vmin=0.5, vmax=3600, source="ASSUMED", source_note="fast 실험 먼저; 실행 시간 ≤ 예산 + 3 s"),
        ],
        presets=[
            Preset("nominal", "예산 8 s", {}, "", ("nominal", "reference")),
            Preset("wide", "예산 45 s (1분 안쪽)", {"budget_s": 45.0}, "대부분의 기준 preset", ("variant",)),
            Preset("full", "예산 600 s — 전부 실행, 수 분 걸림", {"budget_s": 600.0}, "모든 실습의 기준 preset (수 분)", ("variant",)),
        ],
        run=run_test_independence,
        model_level="메타",
        suggested_change="실행 시간 예산을 8 → 45 s로 늘려 더 많은 기준 preset을 실행한다 (전부는 ‘full’ preset, 수 분).",
        prediction=Prediction(
            "모든 check가 PASS라면 이 시뮬레이터가 실제 하드웨어를 맞힌다고 말할 수 있나?",
            ["예", "아니다 — verification이지 validation이 아니다", "독립 check만 있으면 예", "모르겠다"],
            "아니다 — verification이지 validation이 아니다",
            "check는 방정식을 올바르게 풀었는지(닫힌 식 vs 독립 적분·solver·에너지 항등식)를 본다. 같은 합성 입력·이상 소자 가정을 공유하므로 실물 측정과의 일치(validation)는 아니다.",
            ["checks", "indep_frac"],
        ),
        suggested={"budget_s": 45.0},
        student="시험지를 만든 사람이 답안지도 만들면 채점이 의미가 없다. 그래서 검산은 다른 방법(다른 식, 다른 적분 방법)으로 해야 하고, 그래도 그것은 ‘계산이 맞다’는 뜻이지 ‘현실과 같다’는 뜻은 아니다.",
        expert=(
            "‘독립’은 두 쪽이 공식을 공유하지 않는 비교(닫힌 식 vs 구간 적분, 행렬지수 vs 손으로 쓴 ODE, 상태식 vs 포트 에너지 항등식, GUM vs Monte Carlo)다. 같은 helper를 두 번 부른 것은 회귀다. "
            "test에는 교과서 숫자를 리터럴로 적어 reference 모듈과도 독립이게 한다. 이 표는 선언과 구조를 세는 것이므로 의미 판정은 사람이 한다."
        ),
        customer_ko="시뮬레이션의 모든 수치 검증은 서로 다른 계산 경로로 대조했지만, 실측으로 검증한 것은 아닙니다. 설계 결정에 쓰실 부분은 해당 corner의 측정으로 확인하시죠.",
        customer_en="Every numerical check in the simulation was cross-checked on an independent computational path, but nothing was validated against measurements. For the parts that drive your design decision, let's confirm at the relevant corners with measured data.",
        questions=[_Q[3]],
        textbook=[TB_E11, TB_19],
        reference_presets=["nominal"],
        runtime_hint="seconds",
        claim_limit="저장소 자체의 verification 경로 지도. 실물 validation·의미 판정은 포함하지 않는다.",
    ),
]

LAB = Lab(
    id="EX11",
    title="모델을 믿을 수 있는 범위: 검증·식별·불확도",
    title_en="How far the model can be trusted: verification, identification, uncertainty",
    track="expert",
    order=11,
    path_note="E13 5회전 (E08–E11)",
    textbook=[TB_E11, TB_E12, TB_19],
    prerequisites=["FL01", "EX02"],
    summary="손실 측정 불확도와 순위 → binomial 상한과 grid 금지 → ringing 식별 가능성과 corner holdout → Monte Carlo 보고 → 모든 실습의 test-independence 표와 모델 카드.",
    experiments=EXPERIMENTS,
    minimum_scope="uncertainty budget, calibration/holdout 구분, identifiability 예제, test independence map, model card (교재 E11); 15.40157 W·4.87487 W·0.2991 %",
    claim_limits=[
        "모든 입력은 합성·가정 (seed 명시); 실제 계측기·부품 분포 자료는 MISSING_INPUT",
        "통계 상한은 IID 가정 조건부 — 결정적 grid에는 붙이지 않음 (NOT_EVALUABLE)",
        "test-independence 표는 verification 경로의 지도이며 hardware validation이 아님",
    ],
    test_paths=["tests/test_ex11.py"],
    extends=["FL01", "EX02"],
)
