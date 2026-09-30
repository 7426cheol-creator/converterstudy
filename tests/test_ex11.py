"""EX11 verification, identification and uncertainty.

Independence map (docs/TEST_INDEPENDENCE.md):
  * textbook numbers typed literally: 15.40157 W, 4.87487 W, 0.001386, 0.2991 %
  * GUM propagation vs seeded Monte Carlo and finite differences inside the lab (independent)
  * zero-failure and Clopper-Pearson bounds: Beta quantile (reference) vs a binomial-sum root solved here by bisection (independent)
  * identifiability: the (k L, C/k, k R) waveform identity written here with numpy; exact engine vs closed-form step response (independent)
  * lognormal yield: Phi formula written here with math.erf vs the lab's Monte Carlo (independent)
  * statuses (UNRESOLVED_RANKING, MISSING_INPUT, NOT_EVALUABLE for a grid, OUT_OF_VALIDITY at the corner): regression
"""

import math

import numpy as np
import pytest

from convlab.labs import get_lab
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("EX11").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def codes(js):
    return [v["code"] for v in js["verdicts"]]


def upper_by_bisection(k, n, conf):
    """p such that sum_{j<=k} C(n,j) p^j (1-p)^(n-j) = 1 - conf, written out and bisected in log space."""

    def cdf(p):
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p) + (n - j) * math.log1p(-p)) for j in range(k + 1))

    lo, hi = 1e-12, 0.5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if cdf(mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def test_textbook_loss_uncertainty_numbers():
    # u(loss) = sqrt(11^2 + 10.78^2) = 15.40157 W ; with rho = 0.9: sqrt(11^2 + 10.78^2 - 2*0.9*11*10.78) = 4.87487 W
    assert math.sqrt(11.0**2 + 10.78**2) == pytest.approx(15.40157, abs=5e-6)
    assert math.sqrt(11.0**2 + 10.78**2 - 2 * 0.9 * 11.0 * 10.78) == pytest.approx(4.87487, abs=5e-6)
    js = run("loss_uncertainty", "textbook")
    assert metric(js, "loss") == pytest.approx(220.0)
    assert metric(js, "u_loss_ind") == pytest.approx(15.40157, abs=5e-6)
    assert metric(js, "u_eta") == pytest.approx(0.001386, rel=1e-3)  # sqrt(2) * 9.8e-4
    assert metric(js, "ranking") == "UNRESOLVED_RANKING"  # 5 W < 2 x 21.8 W
    assert js["status"]["code"] == "UNRESOLVED_RANKING"
    assert all(c["status"] == "PASS" for c in js["checks"])
    assert sum(c["independent"] for c in js["checks"]) >= 2


def test_correlation_needs_a_recorded_basis():
    js = run("loss_uncertainty", "rho09_nobasis")
    assert metric(js, "u_loss_rho") == pytest.approx(4.87487, abs=5e-6)
    assert "MISSING_INPUT" in codes(js)
    js2 = run("loss_uncertainty", "rho09_basis")
    assert metric(js2, "u_loss_rho") == pytest.approx(4.87487, abs=5e-6)
    assert "MISSING_INPUT" not in codes(js2)


def test_instrument_figure_interpretation_changes_the_standard_uncertainty():
    js = run("loss_uncertainty", "rect")
    assert metric(js, "u_in") == pytest.approx(11.0 / math.sqrt(3.0))
    js2 = run("loss_uncertainty", "textbook", spec_type="k2")
    assert metric(js2, "u_in") == pytest.approx(5.5)


def test_paired_measurement_can_resolve_the_ranking():
    js = run("loss_uncertainty", "paired")
    # u_diff = 15.40157 * sqrt(2 (1 - 0.995)) = 1.540157 W; k = 2 -> 3.08 W < 5 W
    assert metric(js, "u_diff") == pytest.approx(15.40157 * math.sqrt(2 * 0.005), rel=1e-6)
    assert metric(js, "ranking") == "확정"
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"


def test_zero_failure_bound_textbook():
    # (1 - p)^1000 = 0.05  ->  p = 1 - 0.05^(1/1000) = 0.2991 %
    assert 1 - 0.05 ** (1 / 1000) == pytest.approx(0.002991, rel=2e-4)
    assert upper_by_bisection(0, 1000, 0.95) == pytest.approx(0.002991, rel=2e-4)
    js = run("binomial_bounds", "textbook")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "p_upper") == pytest.approx(0.002991, rel=2e-4)
    assert metric(js, "p_upper_zero") == pytest.approx(metric(js, "p_upper"), rel=1e-10)
    assert metric(js, "rule3") == pytest.approx(0.003)
    assert all(c["status"] == "PASS" for c in js["checks"])


def test_clopper_pearson_for_k_failures_matches_a_sum_solved_here():
    js = run("binomial_bounds", "k2")
    assert metric(js, "p_upper") == pytest.approx(upper_by_bisection(2, 1000, 0.95), rel=1e-8)
    js2 = run("binomial_bounds", "million")
    assert metric(js2, "p_upper") == pytest.approx(-math.expm1(math.log(0.05) / 1e6), rel=1e-9)  # about 3.0 ppm


def test_deterministic_grid_gets_no_binomial_bound():
    js = run("binomial_bounds", "grid")
    assert js["status"]["code"] == "NOT_EVALUABLE"
    assert metric(js, "p_upper") == "NOT_EVALUABLE"
    # the grid failure fraction is set by the grid span, the true probability is not
    js2 = run("binomial_bounds", "grid", grid_span=2.5)
    assert metric(js2, "grid_frac") != pytest.approx(metric(js, "grid_frac"))
    assert metric(js2, "p_true") == pytest.approx(metric(js, "p_true"))
    assert metric(js, "p_true") == pytest.approx(0.5 * math.erfc(6.0 / math.sqrt(3.0) / math.sqrt(2.0)))


def test_single_ringing_waveform_does_not_identify_L_C_R():
    # written here: scaling (L, C, R) -> (k L, C / k, k R) keeps omega0 and alpha, hence the waveform
    def v_c(t, L, C, R):
        a = R / (2 * L)
        wd = math.sqrt(1 / (L * C) - a * a)
        return 1 - np.exp(-a * t) * (np.cos(wd * t) + a / wd * np.sin(wd * t))

    t = np.linspace(0, 150e-9, 400)
    assert np.max(np.abs(v_c(t, 10e-9, 1e-9, 0.3) - v_c(t, 30e-9, 1e-9 / 3, 0.9))) < 1e-12
    js = run("identifiability", "nominal")
    assert metric(js, "L_spread") > 2.0
    assert metric(js, "rms_spread") < 1e-3
    assert metric(js, "sv_ratio") < 1e-6
    assert metric(js, "L_joint") == pytest.approx(10e-9, rel=0.05)
    assert all(c["status"] == "PASS" for c in js["checks"])
    assert {"NOT_EVALUABLE", "PASS_WITHIN_MODEL", "OUT_OF_VALIDITY"} <= set(codes(js))
    js2 = run("identifiability", "low_noise")
    assert metric(js2, "L_spread") > 2.0  # less noise does not remove a structural non-identifiability


def test_corner_holdout_exposes_what_random_holdout_hides():
    js = run("identifiability", "nominal")
    assert metric(js, "hold_random") < 0.05
    assert metric(js, "hold_corner_max") > 0.05 > metric(js, "hold_random")
    js2 = run("identifiability", "c_bias")
    assert metric(js2, "L_prior") == pytest.approx(10e-9 / 1.10, rel=0.03)  # a +10 % C error moves L by 1/1.1


def test_monte_carlo_yield_matches_the_lognormal_formula_written_here():
    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def yield_formula(sL, sC, rho, tol):
        s = 0.5 * math.sqrt(sL**2 + sC**2 + 2 * rho * sL * sC)
        return phi(math.log(1 + tol) / s) - phi(math.log(1 - tol) / s)

    ys = {}
    for pk, rho in (("nominal", 0.0), ("corr_pos", 0.8), ("corr_neg", -0.8)):
        js = run("monte_carlo", pk)
        y = metric(js, "yield")
        ys[pk] = y
        n = 20000
        ya = yield_formula(0.025, 0.025, rho, 0.04)
        assert y == pytest.approx(ya, abs=4 * math.sqrt(max(ya * (1 - ya), 1 / n) / n))
        assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert ys["corr_pos"] < ys["nominal"] < ys["corr_neg"]
    assert run("monte_carlo", "nominal")["metrics"] == run("monte_carlo", "nominal")["metrics"]  # seed reproducible


def test_test_independence_map_runs_the_labs_and_scans_the_repository():
    js = run("test_independence", "nominal", budget_s="1 s")
    keys = {t["key"] for t in js["tables"]}
    assert {"t_summary", "t_checks", "t_tests", "t_card"} <= keys
    assert metric(js, "labs") >= 4
    assert metric(js, "runs") >= 1
    assert metric(js, "fails") == 0
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    tests_rows = next(t for t in js["tables"] if t["key"] == "t_tests")["rows"]
    assert any(r[0] == "EX11" and r[2] >= 10 for r in tests_rows)


def test_all_reference_presets_checks_pass():
    lab = get_lab("EX11")
    for e in lab.experiments:
        if e.key == "test_independence":
            continue  # covered above with a small budget
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("EX11").experiment("binomial_bounds")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "textbook", {"n_trials": "0", "conf": "1.5"})
    assert "n_trials" in e.value.errors and "conf" in e.value.errors
    assert run("binomial_bounds", "textbook", k_fail=5, n_trials=3)["status"]["code"] == "OUT_OF_VALIDITY"
    assert run("loss_uncertainty", "textbook", P_out="12 kW")["status"]["code"] == "OUT_OF_VALIDITY"
    exp2 = get_lab("EX11").experiment("loss_uncertainty")
    with pytest.raises(ParamError) as e2:
        resolve_params(exp2.params, exp2.presets, "textbook", {"rho": "1.5"})
    assert "rho" in e2.value.errors
