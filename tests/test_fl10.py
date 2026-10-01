"""FL10 CLLC: seed FAIL preserved, n = 0.93 roots, branch fold, reverse, switching check."""

import pytest

from convlab.labs import get_lab
from convlab.labs._resonant import Tank, fha, rac
from convlab.labs.fl10_cllc import harmonic_power
from convlab.model.params import resolve_params
from convlab.reference import resonant as ref


def run(exp, preset=None, **over):
    e = get_lab("FL10").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


C1 = 28.144773e-9
SYM = Tank(40e-6, C1, 200e-6, 40e-6, C1)


@pytest.fixture(scope="module")
def td():
    return run("time_domain", "textbook")


def test_seed_fail_is_preserved():
    js = run("seed_fail", "seed")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "g_req_hi") == pytest.approx(1.082353, abs=5e-7)  # textbook
    assert metric(js, "gmax_hi") == pytest.approx(1.016401, abs=5e-7)  # textbook
    assert metric(js, "gmax_hi") < metric(js, "g_req_hi")


def test_candidate_referred_symmetry_and_roots():
    js = run("fix_n093", "textbook")
    assert metric(js, "L2") == pytest.approx(46.248121e-6, rel=1e-8)
    assert metric(js, "C2") == pytest.approx(24.342414e-9, rel=1e-8)
    assert metric(js, "f_저전압_0") == pytest.approx(164390.0, abs=1.0)
    assert metric(js, "f_중간_0") == pytest.approx(162811.0, abs=1.0)
    assert metric(js, "f_고전압_0") == pytest.approx(136099.47, abs=0.01)  # instructions
    assert metric(js, "f_고전압_1") == pytest.approx(147060.86, abs=0.01)
    assert metric(js, "slope_lo") == pytest.approx(0.001871, rel=5e-3)
    assert metric(js, "slope_hi") == pytest.approx(-0.001803, rel=5e-3)
    assert metric(js, "I1_lo") == pytest.approx(14.390, abs=1e-3)
    assert metric(js, "I1_hi") == pytest.approx(14.765, abs=1e-3)
    codes = {v["code"] for v in js["verdicts"]}
    assert "CANDIDATE_FHA_ONLY" in codes


def test_keeping_physical_secondary_lc_is_a_different_network():
    a = run("fix_n093", "textbook")
    b = run("fix_n093", "kept_lc")
    assert abs(metric(b, "f_고전압_0") - metric(a, "f_고전압_0")) > 1000.0
    assert metric(b, "L2p") == pytest.approx(0.93**2 * 40e-6, rel=1e-12)


@pytest.mark.parametrize("f", [120e3, 136099.47, 150e3, 175e3, 210e3])
def test_textbook_cllc_formula_matches_nodal(f):
    R = rac(0.93, 920.0, 11000.0)
    assert fha(SYM, f, R).H == pytest.approx(ref.cllc_H(f, 40e-6, C1, 200e-6, 40e-6, C1, R), abs=1e-12)


def test_reverse_low_corner_fails_and_is_not_the_reciprocal():
    js = run("reverse", "textbook")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "req_저전압") == pytest.approx(700 / (0.93 * 650), rel=1e-12)
    assert metric(js, "gmax_저전압") < metric(js, "req_저전압")
    # at the forward 800/800 root the reverse gain is not 1/g_fwd
    R_rev = rac(1.0, 800.0, 11000.0)
    g_rev = fha(SYM.swapped(), 162811.40, R_rev).gain
    assert abs(g_rev - 1 / 0.93) > 0.05


def test_branch_fold_and_first_root_policy_jump():
    js = run("branch_selection", "textbook")
    assert 920.0 < metric(js, "Vfold") < 930.0
    assert metric(js, "max_jump") > 10e3  # the first-root policy changes branch
    assert metric(js, "max_jump") > 2 * metric(js, "max_jump_lock")


def test_switching_power_at_fha_roots_far_from_11kw(td):
    assert metric(td, "P_td_lo") == pytest.approx(17341.4, rel=1e-3)  # regression (ideal model)
    assert metric(td, "P_td_hi") == pytest.approx(22242.0, rel=1e-3)
    # independent frequency-domain path
    assert harmonic_power(SYM, 136099.47, 850.0, 0.93 * 920.0) == pytest.approx(metric(td, "P_td_lo"), rel=5e-3)


def test_switching_operating_points(td):
    assert metric(td, "f_sw_lo") == "범위 내 없음"
    assert 147.9e3 < metric(td, "f_sw_hi") < 148.1e3
    codes = {v["code"] for v in td["verdicts"]}
    assert {"CANDIDATE_FHA_ONLY", "FAIL_CONSTRAINT", "SCREEN_ONLY", "NOT_EVALUABLE"} <= codes


def test_reported_11kW_point_really_gives_11kW(td):
    # On a slope of about -1.6 kW/Hz the midpoint of a 1 Hz bracket was 11.63 kW (+5.7 %), and the reported
    # currents and capacitor peaks belonged to that power (found by the Octave cross-check). The point is now
    # refined until the power is within 0.1 %; Octave's exact 11 kW point is 148004.862 Hz with C peaks
    # 817.5 V and 771.6 V (matlab/xc_cllc_*.m, ode45 + Newton shooting written from the E05 state equations).
    assert metric(td, "P_sw_hi") == pytest.approx(11000.0, rel=1e-3)
    assert metric(td, "f_sw_hi") == pytest.approx(148004.862, abs=0.5)
    assert metric(td, "vC1_pk") == pytest.approx(817.5, abs=0.1)
    assert metric(td, "vC2_pk") == pytest.approx(771.6, abs=0.1)
    # and it is almost neutrally stable in the lossless model: Floquet |lambda| about 0.99998
    assert 0.9999 < metric(td, "rho_sw_hi") < 1.0
    assert any(v["code"] == "MARGINAL" and "중립 안정" in v["why"] for v in td["verdicts"])


def test_losses_reduce_the_sensitivity():
    lossy = run("time_domain", "lossy")
    assert 147.0e3 < metric(lossy, "f_sw_hi") < 148.0e3
    assert abs(metric(lossy, "sens_hi")) < 1e5  # W/kHz, orders below the lossless value


def test_all_reference_checks_pass(td):
    lab = get_lab("FL10")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = td if (e.key, pk) == ("time_domain", "textbook") else run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c["name"], c["value"])
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m["key"])
