"""EX06 DAB modulation: textbook table, independent SPS path, staged selection, implementation."""

import math

import pytest

from convlab.labs import get_lab
from convlab.labs._dab import PI, Modulation, commutation_screen, pwl_waves
from convlab.labs.ex06_dab_modulation import first_root
from convlab.model.params import resolve_params
from convlab.reference import dab as ref


def run(exp, preset=None, **over):
    e = get_lab("EX06").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_textbook_table():
    js = run("general_modulation", "textbook")
    assert metric(js, "phi_sps") == pytest.approx(0.399994, abs=1e-6)
    assert metric(js, "irms_sps") == pytest.approx(3.113563, abs=1e-6)
    assert metric(js, "ipk_sps") == pytest.approx(5.659830, abs=1e-6)
    assert metric(js, "phi_c") == pytest.approx(0.499016, abs=1e-6)
    assert metric(js, "irms_c") == pytest.approx(2.893092, abs=1e-6)
    assert metric(js, "ipk_c") == pytest.approx(5.007628, abs=1e-6)
    assert metric(js, "drms") == pytest.approx(-0.0708, abs=5e-4)
    assert metric(js, "dpk") == pytest.approx(-0.1152, abs=5e-4)
    assert metric(js, "cond_ratio") == pytest.approx(0.8634, abs=1e-4)


def test_sps_closed_form_vs_general_switching_function():
    # independent: closed form vs b(theta; pi, phi) piecewise integration, several phases and ratios
    for V1, V2 in ((900.0, 600.0), (800.0, 800.0), (600.0, 900.0)):
        for phi in (0.05, 0.4, 0.9, 1.5):
            w = pwl_waves(V1, V2, 200e-6, 100e3, Modulation(PI, PI, phi, 0.0))
            assert w.P2 == pytest.approx(ref.sps_power(V1, V2, 200e-6, 100e3, phi), rel=1e-12)
            assert w.Irms == pytest.approx(ref.sps_currents(V1, V2, 200e-6, 100e3, phi)[2], rel=1e-12)


def test_zero_state_pattern_changes_legs_not_port_waveform():
    a = pwl_waves(900.0, 600.0, 200e-6, 100e3, Modulation(0.7 * PI, PI, 0.5, 0.0, "alternating", "alternating"))
    b = pwl_waves(900.0, 600.0, 200e-6, 100e3, Modulation(0.7 * PI, PI, 0.5, 0.0, "fixed_lower", "fixed_lower"))
    assert a.P2 == pytest.approx(b.P2, rel=1e-12) and a.Irms == pytest.approx(b.Irms, rel=1e-12)
    sa = commutation_screen(a, Modulation(0.7 * PI, PI, 0.5, 0.0, "alternating", "alternating"), 1e5, 50 / 3, 900.0, 36.0, (0.5e-9, 40.0), (8e-9, 5.0), 150e-9, 50e-9)
    sb = commutation_screen(b, Modulation(0.7 * PI, PI, 0.5, 0.0, "fixed_lower", "fixed_lower"), 1e5, 50 / 3, 900.0, 36.0, (0.5e-9, 40.0), (8e-9, 5.0), 150e-9, 50e-9)
    ia = sorted(round(e.i_out, 6) for e in sa if e.bridge == "1차")
    ib = sorted(round(e.i_out, 6) for e in sb if e.bridge == "1차")
    assert ia != ib  # same bridge voltage, different leg commutation currents


def test_secondary_zvs_sign_boundary_sps():
    # SPS secondary edge current positive iff phi > pi (1 - V2/V1)/2 (derived independently here)
    V1, V2 = 900.0, 600.0
    phi_b = PI * (1 - V2 / V1) / 2
    for phi, positive in ((phi_b * 0.9, False), (phi_b * 1.1, True)):
        _, iphi, _, _ = ref.sps_currents(V1, V2, 200e-6, 100e3, phi)
        assert (iphi > 0) == positive


def test_staged_selection_changes_at_2kw_corner():
    js = run("candidate_map", "textbook")
    assert metric(js, "changed_cor").startswith("예")
    assert metric(js, "irms2_cor") > metric(js, "irms1_cor")
    assert "권장 불가" in metric(js, "changed_nom")
    codes = {v["code"] for v in js["verdicts"]}
    assert "MISSING_INPUT" in codes and "SCREEN_ONLY" in codes


def test_timer_resolution_erratum_value():
    js = run("implementation", "textbook")
    assert metric(js, "dphi") == pytest.approx(2 * math.pi * 1e5 * 10e-9, rel=1e-12)
    assert metric(js, "dP_lin") == pytest.approx(20.1246, rel=1e-5)  # textbook states 20.62 W (erratum E-001)
    assert metric(js, "dP_rel") == pytest.approx(0.013416, rel=1e-4)


def test_abrupt_transition_offset_matches_prediction_and_splice_is_smaller():
    js = run("implementation", "textbook")
    assert abs(metric(js, "off_splice")) < 0.2 * abs(metric(js, "off_abrupt"))
    assert metric(js, "off_end") == pytest.approx(metric(js, "off_abrupt"), rel=1e-9)  # lossless: offset stays


def test_first_root_is_smallest_positive():
    phi = first_root(900.0, 600.0, 200e-6, 100e3, 1500.0, PI, PI)
    assert phi == pytest.approx(ref.sps_phi_for_power(900.0, 600.0, 200e-6, 100e3, 1500.0), rel=1e-12)


def test_reference_presets_checks_pass():
    lab = get_lab("EX06")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c["name"])
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m["key"])
