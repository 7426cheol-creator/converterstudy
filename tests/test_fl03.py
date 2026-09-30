"""FL03 loss, temperature and lifetime.

Independence map:
  * textbook numbers typed literally (77.642411 C at 10 s, 85 C final, 20 K fixed rise)
  * single node: matrix-exponential engine vs closed form (reference/thermal.py) vs hand-written RK4
  * electro-thermal: closed form (reference) vs fixed-point iteration vs exact transient end value
  * loss map: event summation with a linear E / constant R vs the sinusoidal-PWM closed forms in
    reference/thermal.py (independent of the lab's own inline hand calculation)
  * Foster/Cauer: generalized-eigenvalue Foster vs engine step response; continued-fraction round trip
"""

import math

import numpy as np
import pytest

from convlab.labs import get_lab
from convlab.labs.fl03_thermal import MAP_I, MAP_T, bilinear, cauer_to_foster, event_losses, foster_to_cauer, rk4_scalar, synthetic_map
from convlab.model.params import ParamError, resolve_params
from convlab.reference import thermal as ref


def run(exp, preset=None, **over):
    e = get_lab("FL03").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


# ---- experiment 1 -------------------------------------------------------------------


def test_rc_step_textbook_numbers():
    js = run("rc_step", "textbook")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "T_te") == pytest.approx(77.642411, abs=5e-7)
    assert metric(js, "T_final") == pytest.approx(85.0, abs=1e-9)
    assert metric(js, "T_wrong") == pytest.approx(85.0)  # the steady-state formula used at 10 s
    assert metric(js, "frac") == pytest.approx(1 - math.exp(-1), rel=1e-12)
    assert ref.rc_step(100, 0.2, 10, 65, 10) == pytest.approx(metric(js, "T_te"), rel=1e-13)


def test_rk4_path_is_fourth_order():
    f = lambda t, T: (100 - (T - 65) / 0.2) / 50.0  # noqa: E731
    exact = ref.rc_step(100, 0.2, 10, 65, 10)
    e1 = abs(rk4_scalar(f, 65.0, 0, 10, 0.5) - exact)
    e2 = abs(rk4_scalar(f, 65.0, 0, 10, 0.25) - exact)
    assert math.log2(e1 / e2) == pytest.approx(4.0, abs=0.1)


def test_same_average_loss_longer_pulses_swing_more():
    js = run("rc_step", "textbook")
    rows = next(t for t in js["tables"] if t["key"] == "t_pulse")["rows"]
    const, fast, slow = rows
    assert const[3] < 0.01 < fast[3] < slow[3]
    assert slow[1] > fast[1] > const[1]
    assert const[1] == pytest.approx(65 + 0.5 * 100 * 0.2, abs=1e-3)


# ---- experiment 2 -------------------------------------------------------------------


def test_electrothermal_fixed_point_and_paths():
    js = run("electrothermal", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "dT_fixed") == pytest.approx(20.0)
    assert metric(js, "dT_fp") == pytest.approx(20.0 / (1 - 0.004 * 0.2 * 100), rel=1e-12)
    assert metric(js, "dT_fp") == pytest.approx(ref.electrothermal_rise(100, 0.2, 0.004), rel=1e-12)
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    rows = next(t for t in js["tables"] if t["key"] == "t_fp")["rows"]
    assert rows[2][0].startswith("과부하") and rows[2][1] == pytest.approx(0.16)


def test_runaway_has_no_stable_fixed_point_and_leaves_the_valid_range():
    js = run("electrothermal", "runaway")
    assert js["status"]["code"] == "NO_STABLE_FIXED_POINT"
    codes = [v["code"] for v in js["verdicts"]]
    assert "OUT_OF_VALIDITY" in codes
    assert metric(js, "g0") == pytest.approx(1.2)
    assert ref.electrothermal_rise(100, 0.2, 0.06) == math.inf
    assert isinstance(metric(js, "Tpk_c"), str)  # not extrapolated beyond the loss-model range


def test_sensitive_case_exceeds_model_validity_during_overload():
    js = run("electrothermal", "sensitive")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
    t_x = metric(js, "t_valid")
    assert 120.0 < t_x < 150.0  # inside the 30 s overload window


def test_numerical_divergence_is_not_physical_runaway():
    js = run("electrothermal", "coarse_step")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"  # the physics is stable
    rows = next(t for t in js["tables"] if t["key"] == "t_methods")["rows"]
    assert "SOLVER_FAILED" in rows[3][1]
    assert any("Euler" in w for w in js["warnings"])


# ---- experiment 3 -------------------------------------------------------------------


def test_loss_map_limit_case_matches_sinusoidal_pwm_closed_forms():
    v = {"Ipk": 100.0, "fs": 20e3, "fout": 100.0, "m": 0.9, "cosphi": 0.9, "t_dt": 0.0}
    ev = event_losses(v, 25.0, synthetic_map(), lin=(6e-6, 20e-3))
    assert ev["P"]["ch_f"] == pytest.approx(ref.pwm_conduction_forward(100, 20e-3, 0.9, 0.9), rel=1e-4)
    assert ev["P"]["ch_r"] == pytest.approx(ref.pwm_conduction_reverse(100, 20e-3, 0.9, 0.9), rel=1e-4)
    assert ev["P"]["on"] + ev["P"]["off"] == pytest.approx(ref.pwm_switching_linear_E(20e3, 6e-6, 100), rel=1e-4)


def test_loss_map_nominal_breakdown_and_single_point_error():
    js = run("loss_map", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    parts = sum(metric(js, k) for k in ("P_ch_f", "P_ch_r", "P_on", "P_off", "P_dio", "P_rec"))
    assert metric(js, "P_total") == pytest.approx(parts, rel=1e-12)
    assert metric(js, "sw_err_1pt") > 0.2  # one datasheet point reused for every event overestimates here
    assert metric(js, "n_bad") == 0
    assert 65.0 < metric(js, "Tj_et") < 175.0


def test_out_of_map_points_are_flagged_not_extrapolated():
    js = run("loss_map", "over_I")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
    assert metric(js, "n_bad") > 0
    assert isinstance(metric(js, "P_total"), str)
    hot = run("loss_map", "hot")
    assert hot["status"]["code"] == "OUT_OF_VALIDITY"
    assert isinstance(metric(hot, "Tj_et"), str)
    mp = synthetic_map()
    assert bilinear(mp["Eon"], 151.0, 100.0) is None
    assert bilinear(mp["Eon"], 50.0, 180.0) is None
    for a in range(MAP_I.size):
        for b in range(MAP_T.size):
            assert bilinear(mp["Eon"], MAP_I[a], MAP_T[b]) == pytest.approx(mp["Eon"][a, b], rel=1e-15)


# ---- experiment 4 -------------------------------------------------------------------


def test_foster_cauer_round_trip_and_connection_error():
    R = np.array([0.02, 0.04, 0.06, 0.08])
    C = np.array([0.01, 0.06, 0.4, 2.5])
    Rf, tf = cauer_to_foster(R, C)
    assert Rf.sum() == pytest.approx(0.2, rel=1e-12)  # Foster resistances sum to R_jc
    Rc, Cc = foster_to_cauer(Rf, tf)
    np.testing.assert_allclose(Rc, R, rtol=1e-9)
    np.testing.assert_allclose(Cc, C, rtol=1e-9)
    js = run("foster_cauer", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "T_ss") == pytest.approx(65 + 100 * (0.2 + 0.05 + 0.15), rel=1e-12)
    assert metric(js, "err_wrong") > 1.0  # attaching the heatsink to a Foster node is wrong in the transient
    assert abs(metric(js, "err_conv")) < 0.01  # a Cauer conversion restores a physical connection
    assert metric(js, "case_jump") == pytest.approx(5.0)


# ---- general ---------------------------------------------------------------------------


def test_inputs_are_rejected_not_clamped():
    e = get_lab("FL03").experiment("electrothermal")
    with pytest.raises(ParamError) as exc:
        resolve_params(e.params, e.presets, "nominal", {"Rth": "-0.1", "alpha": "1"})
    assert "Rth" in exc.value.errors and "alpha" in exc.value.errors
    e4 = get_lab("FL03").experiment("foster_cauer")
    with pytest.raises(ParamError):
        resolve_params(e4.params, e4.presets, "nominal", {"n_fit": "2.5"})


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL03")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
