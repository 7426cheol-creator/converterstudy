"""EX05 CLLC dynamics: identity, FHA roots vs switching points, Floquet, G_vf, closed loop, start-up, tolerance."""

import functools

import pytest

from convlab.labs import get_lab
from convlab.model.params import resolve_params


@functools.lru_cache(maxsize=None)
def run(exp, preset=None):
    e = get_lab("EX05").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def codes(js):
    return {v["code"] for v in js["verdicts"]}


def test_energy_identity_holds_and_catches_a_sign_slip():
    js = run("state_identity", "textbook")
    assert metric(js, "id_max") < 1e-12
    assert metric(js, "id_wrong") > 1e-3
    assert run("state_identity", "lossy")["status"]["code"] == "PASS_WITHIN_MODEL"


def test_fha_lower_root_has_the_opposite_switching_slope():
    js = run("operating_points", "textbook")
    assert metric(js, "slope_fha_lo") > 0 and metric(js, "slope_td_lo") < 0
    assert metric(js, "vo_at_lo") == pytest.approx(961.5, abs=0.5)  # regression, ideal model
    assert metric(js, "vo_at_hi") == pytest.approx(922.9, abs=0.5)
    assert 147.9e3 < metric(js, "f0") < 148.1e3
    assert metric(js, "f_peak_td") < metric(js, "f_peak_fha") - 10e3
    assert "OUT_OF_VALIDITY" in codes(js)
    seed = next(c for c in js["checks"] if c["name"].startswith("seed"))
    assert seed["status"] == "PASS"  # the seed failure is still reproduced


def test_floquet_dominant_mode_is_a_lightly_damped_kHz_resonance():
    js = run("floquet", "textbook")
    assert metric(js, "rho") < 1.0 and 1.0 - metric(js, "rho") < 1e-3
    assert 900.0 < metric(js, "f_osc") < 1100.0
    assert metric(js, "rate_meas") == pytest.approx(metric(js, "rho"), rel=2e-3)
    assert "MARGINAL" in codes(js)
    lossy = run("floquet", "lossy")
    assert metric(lossy, "rho") < metric(js, "rho")  # series R adds damping


def test_gvf_dc_matches_steady_slope_and_resonance_dominates():
    js = run("gvf", "textbook")
    for c in js["checks"]:
        assert c["status"] == "PASS", c["name"]
    assert metric(js, "peak_op") > 10 * abs(metric(js, "g0_op"))


def test_closed_loop_low_gain_stable_high_gain_unstable_wrong_sign_runs_away():
    slow = run("closed_loop", "slow")
    fast = run("closed_loop", "fast")
    wrong = run("closed_loop", "wrong_sign")
    assert metric(slow, "rho_cl") < 1.0
    assert metric(fast, "rho_cl") > 1.0 and metric(fast, "slope_ok") == "예"
    assert fast["status"]["code"] == "UNSTABLE"
    assert metric(wrong, "rho_cl") > 1.0 and metric(wrong, "slope_ok").startswith("아니오")
    assert metric(wrong, "f_drift") < 0  # frequency moves the way that makes the error worse


def test_startup_and_reverse_failures_are_kept():
    js = run("startup_reverse", "textbook")
    assert metric(js, "i1_pk_max") > 5 * metric(js, "i1_pk_ss")
    assert metric(js, "vo_max_start") > 1.1 * 920.0
    assert metric(js, "p_b_max") > 1.5 * 11000.0
    assert metric(js, "rev_req") > metric(js, "rev_gmax")
    assert metric(js, "rev_pmax") < 11000.0
    assert js["status"]["code"] == "FAIL_CONSTRAINT"


def test_tolerance_resonance_shifts_textbook():
    js = run("tolerance", "textbook")
    assert metric(js, "fr_pp") == pytest.approx(142.857e3, abs=1.0)
    assert metric(js, "fr_mm") == pytest.approx(157.895e3, abs=1.0)
    assert metric(js, "fr_pm") == pytest.approx(150.188e3, abs=1.0)


def test_all_reference_checks_pass():
    lab = get_lab("EX05")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c["name"], c["value"])
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m["key"])
