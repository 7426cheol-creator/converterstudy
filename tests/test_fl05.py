"""FL05 OBC/PFC.

Independence map:
  * textbook numbers are typed in literally (16.45043 A, 10.69881 kW, 18.278 A, 230.94 V, 0.8165,
    1.0265, 2.944 mF, 705 uF, 31.2 J, 9.75 %)
  * line current: sqrt(3) closed form vs a time-domain sum of v_x i_x solved with brentq (inside the lab)
    and vs the reference module here
  * THD / PF: DFT of the constructed waveform vs the constructed amplitudes
  * switching PFC: energy ledger with exact moments; Hermite-GL Fourier vs exact oscillator-state
    moments; averaged ODE (B) vs exact switching solution (C); 2-omega ripple vs P/(omega C V)
  * DC link: nonlinear energy ODE vs exact energy solution and vs the small-ripple formula;
    unbalance ripple vs symmetrical components
  * preset/metric plumbing: regression
"""

import math

import pytest

from convlab.labs import get_lab
from convlab.labs.fl05_pfc import three_phase_power_td
from convlab.model.params import ParamError, resolve_params
from convlab.reference import pfc as ref


def run(exp_key, preset=None, **over):
    exp = get_lab("FL05").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def check(js, prefix):
    return next(c for c in js["checks"] if c["name"].startswith(prefix))


def verdicts(js):
    return [v["code"] for v in js["verdicts"]]


def test_grid_boundary_textbook_numbers():
    js = run("grid_boundary", "nominal")
    assert metric(js, "I_line") == pytest.approx(16.45043, abs=5e-6)
    assert metric(js, "P_at_lim") == pytest.approx(10698.81, abs=0.01)
    assert metric(js, "I_low") == pytest.approx(18.278, abs=5e-4)
    assert metric(js, "V_ph") == pytest.approx(230.94, abs=5e-3)
    assert metric(js, "m_nom") == pytest.approx(0.8165, abs=5e-5)
    assert metric(js, "m_cor") == pytest.approx(1.0265, abs=5e-5)
    # 11 kW and 16 A cannot both hold at 400 V: a customer decision; SPWM fails the 440/700 corner
    v = verdicts(js)
    assert "CUSTOMER_DECISION_REQUIRED" in v and "FAIL_CONSTRAINT" in v and "OUT_OF_VALIDITY" in v
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "V_need") == pytest.approx(11000 / (math.sqrt(3) * 16 * 0.97 * 0.995), rel=1e-12)


def test_grid_current_time_domain_path():
    # p(t) = sum v_x i_x over one period at the textbook current gives P_bat / eta
    P, Irms, pf = three_phase_power_td(400.0, 16.45043, 0.995)
    assert P == pytest.approx(11000 / 0.97, rel=1e-6)
    assert pf == pytest.approx(0.995, rel=1e-12)
    # the same power factor made of distortion instead of displacement needs the same RMS current
    thd = math.sqrt(1 / 0.995**2 - 1)
    P2, Irms2, pf2 = three_phase_power_td(400.0, 16.45043, 1.0, thd)
    assert P2 == pytest.approx(11000 / 0.97, rel=1e-6) and pf2 == pytest.approx(0.995, rel=1e-9)
    js = run("grid_boundary", "nominal")
    for c in js["checks"]:
        assert c["status"] == "PASS" and c["independent"]


def test_svpwm_corner_is_inside_the_ideal_limit():
    js = run("grid_boundary", "svpwm")
    assert "FAIL_CONSTRAINT" not in verdicts(js)
    assert metric(js, "m_lim") == pytest.approx(2 / math.sqrt(3), rel=1e-12)
    assert 0.10 < metric(js, "m_margin") / 100 < 0.12


def test_pf_thd_definitions():
    js = run("pf_thd", "nominal")
    assert metric(js, "thd") == pytest.approx(5.0, rel=1e-9)  # sqrt(4^2 + 3^2) %
    assert metric(js, "cos_phi1") == pytest.approx(math.cos(math.radians(10)), rel=1e-12)
    pf = math.cos(math.radians(10)) / math.sqrt(1 + 0.05**2)
    assert metric(js, "pf_true") == pytest.approx(pf, rel=1e-9)
    assert metric(js, "pf_formula") == pytest.approx(pf, rel=1e-9)
    off = run("pf_thd", "offset")
    assert metric(off, "thd") == pytest.approx(5.0, rel=1e-9)  # DC is not in the THD
    assert metric(off, "pf_true") < metric(js, "pf_true")  # but it lowers the PF
    assert metric(off, "pf_formula") - metric(off, "pf_true") > 1e-4


def test_unsynchronised_window_is_not_evaluable():
    js = run("pf_thd", "unsync")
    assert js["status"]["code"] == "NOT_EVALUABLE"
    assert abs(metric(js, "thd") - 5.0) > 0.5  # leakage corrupts the reading


def test_boost_pfc_switching_model_nominal():
    js = run("boost_pfc", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    for c in js["checks"]:
        assert c["status"] in ("PASS", "INFO"), c
    assert abs(check(js, "에너지 잔차")["value"]) < 1e-9
    assert check(js, "기본파: Hermite")["value"] < 1e-9
    # forced ideal sine: THD = 0 by construction, the switching model is not zero
    assert metric(js, "thd_forced") < 1e-9
    assert 1.0 < metric(js, "thd") < 5.0
    assert metric(js, "pf") > 0.99
    # DC link: mean at the reference, 2-omega ripple close to P/(omega C V) (typed here)
    assert metric(js, "Vdc") == pytest.approx(400.0, rel=5e-3)
    Pout = metric(js, "Pin") * metric(js, "eta")
    assert metric(js, "dV2w") == pytest.approx(Pout / (2 * math.pi * 50 * 1e-3 * 400), rel=0.03)
    # averaged model and switching model agree in CCM
    for name in ("평균모델(B) vs 스위칭(C): DC-link", "평균모델(B) vs 스위칭(C): 기본파", "평균모델(B) vs 스위칭(C): 2ω"):
        assert check(js, name)["status"] == "PASS"
    # bridge polarity events: both half-cycles appear in the zero-crossing window
    modes = {b["mode"] for b in js["circuit"]["intervals"]}
    assert {"Q+", "Q−", "D+", "D−"} <= modes


def test_voltage_loop_bandwidth_raises_third_harmonic():
    a = run("boost_pfc", "nominal")
    b = run("boost_pfc", "fast_v")
    assert metric(b, "h3") > 3 * metric(a, "h3")
    assert metric(b, "thd") > metric(a, "thd")
    assert metric(b, "pf") < metric(a, "pf")


def test_light_load_dcm_distortion():
    js = run("boost_pfc", "light")
    assert metric(js, "dcm") > 0.1  # percent of the window in DCM
    assert metric(js, "thd") > 8.0
    assert metric(js, "pf") < 0.95
    assert abs(check(js, "에너지 잔차")["value"]) < 1e-9


def test_boost_pfc_rejects_incommensurate_or_non_boost_inputs():
    js = run("boost_pfc", "nominal", fs="20.03k")  # 400.6 switching periods per line cycle
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
    js = run("boost_pfc", "nominal", V_rms="290")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"


def test_dclink_textbook_numbers():
    js = run("dclink_power", "nominal")
    assert metric(js, "C_rip") == pytest.approx(2.944e-3, rel=2e-4)
    assert metric(js, "C_hold") == pytest.approx(705e-6, rel=2e-3)
    assert metric(js, "dE") == pytest.approx(31.2, rel=1e-12)
    assert metric(js, "dE_frac") == pytest.approx(9.75, rel=1e-12)
    assert metric(js, "t_given") == pytest.approx(2.836e-3, rel=1e-3)
    assert metric(js, "p1_ripple") == pytest.approx(7400.0, rel=1e-9)
    assert metric(js, "p3_ripple") < 1e-6
    assert ref.holdup_capacitance(11e3, 2e-3, 800, 760) == pytest.approx(705.128e-6, rel=1e-5)


def test_unbalance_and_holdup_failure():
    js = run("dclink_power", "unbalance")
    # c-phase current -10 %: I- = 0.1 I/3, ripple = 3 V+ I- = 0.1 P/3
    assert metric(js, "p3_ripple") == pytest.approx(0.1 * 11000 / 3, rel=1e-9)
    bad = run("dclink_power", "small_c")
    assert bad["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(bad, "t_given") < 2e-3


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("FL05").experiment("grid_boundary")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "nominal", {"PF": "1.2", "V_LL": "400 A"})
    assert "PF" in e.value.errors and "V_LL" in e.value.errors
    vals, _, _ = resolve_params(exp.params, exp.presets, "nominal", {"P_bat": "7.4k"})
    assert vals["P_bat"] == pytest.approx(7400.0)


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL05")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
