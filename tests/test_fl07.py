"""FL07 magnetics: winding-voltage flux, flux walking, referral and loss screens.

Independence map:
  * textbook numbers (0.20 T, 0.0064 T/cycle, 0.72 uH, 0.209 mm) typed literally
  * B from the magnetizing-current state vs trapezoid integration of the winding voltage (lab check)
  * T-model exact solution vs an RK45 formulation with other state variables (lab check)
  * L_m -> infinity limit vs the FL08 closed form typed here
  * referral: actual secondary-side circuit vs primary-referred circuit, both simulated
  * skin depth vs a 1-D diffusion finite-difference solution (lab check) and the formula here
  * Dowell / Bessel limiting cases typed here
"""

import math

import pytest

from convlab.labs import get_lab
from convlab.labs.fl07_magnetics import dowell, rac_round, sps_phase
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("FL07").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_textbook_winding_flux_0p2T():
    js = run("winding_flux", "textbook")
    assert 1000 / (4 * 100e3 * 50 * 250e-6) == pytest.approx(0.20, rel=1e-12)
    assert metric(js, "Bpk") == pytest.approx(0.20, rel=2e-4)  # R2' drop and magnetizing current: 0.20001 T
    assert metric(js, "dBpp") == pytest.approx(0.40, rel=2e-4)
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    for c in js["checks"]:
        assert c["status"] == "PASS", c


def test_core_sees_reflected_secondary_not_the_bus():
    js = run("winding_flux", "mismatch")
    assert metric(js, "B_bus") == pytest.approx(900 / (4 * 100e3 * 50 * 250e-6), rel=1e-12)  # 0.18 T
    assert metric(js, "Bpk") == pytest.approx(600 / (4 * 100e3 * 50 * 250e-6), rel=1e-3)  # 0.12 T
    # E06 textbook SPS RMS at 900 V / 600 V, 1.5 kW (L_m -> infinity limit)
    assert metric(js, "I_rms_noLm") == pytest.approx(3.113563, abs=1e-6)
    b_split = metric(run("winding_flux", "split"), "Bpk")
    b_sec = metric(run("winding_flux", "sec_L"), "Bpk")
    assert 0.12 < b_split < 0.18
    assert b_sec == pytest.approx(0.18, rel=0.02)


def test_fl08_limit_and_magnetizing_share():
    js = run("winding_flux", "fl08")
    phi = sps_phase(800, 800, 200e-6, 100e3, 1500)
    assert phi == pytest.approx(0.328972794, abs=1e-9)
    ipk = 800 * phi / (2 * math.pi * 100e3 * 200e-6)
    assert metric(js, "I_rms_noLm") == pytest.approx(ipk * math.sqrt(1 - 2 * phi / (3 * math.pi)), rel=1e-12)
    assert metric(js, "I_rms_noLm") == pytest.approx(2.019881509, abs=1e-9)
    assert metric(js, "Bpk") == pytest.approx(0.16, rel=2e-4)


def test_flux_walk_textbook_and_saturation():
    js = run("flux_walk", "textbook")
    assert 800 * 100e-9 / (50 * 250e-6) == pytest.approx(0.0064, rel=1e-12)
    assert metric(js, "walk") == pytest.approx(0.0064, abs=1e-12)
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    # from -0.16 T the peak rises 0.0064 T per cycle: |B| > 0.35 T after about (0.35 - 0.16)/0.0064 = 30 cycles
    assert 29 <= metric(js, "t_sat") * 100e3 <= 32


def test_resistance_does_not_stop_the_walk_before_saturation():
    js = run("flux_walk", "R_only")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "Idc_inf") == pytest.approx(8.0 / 0.1, rel=1e-9)
    assert metric(js, "Bdc_inf") == pytest.approx(2e-3 * 80 / (50 * 250e-6), rel=1e-9)  # 12.8 T: far beyond saturation
    js = run("flux_walk", "blocking")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    z0 = math.sqrt(2e-3 / 10e-6)
    assert metric(js, "Bmax") == pytest.approx(0.16 + 2e-3 * (8.0 / z0) / (50 * 250e-6), rel=5e-3)  # undamped step estimate


def test_measurement_artefacts_are_not_a_walk():
    js = run("flux_walk", "textbook")
    assert metric(js, "drift_os") == pytest.approx(0.5 * 1e-5 / (50 * 250e-6), rel=1e-6)
    # synchronous sampling with T*fs odd: the two edges sit half a sample apart -> V/(fs N Ae) per cycle
    assert metric(js, "drift_co") == pytest.approx(800 / (16.5e6 * 50 * 250e-6), rel=2e-3)
    assert abs(metric(run("flux_walk", "even_sampling"), "drift_co")) < 1e-5


def test_referral_two_circuits_and_skin_depth():
    js = run("loss_screen", "nominal")
    assert metric(js, "Ls") == pytest.approx(0.72e-6, rel=1e-12)
    assert metric(js, "P_act") == pytest.approx(metric(js, "P_ref"), rel=1e-9)
    assert metric(js, "Is") == pytest.approx(50 / 3 * metric(js, "Ip"), rel=1e-9)
    delta = math.sqrt(1.72e-8 / (math.pi * 100e3 * 4e-7 * math.pi))
    assert metric(js, "delta") == pytest.approx(delta, rel=1e-12)
    assert metric(js, "delta") == pytest.approx(0.209e-3, abs=5e-7)
    for c in js["checks"]:
        assert c["status"] == "PASS", c


def test_dowell_and_round_wire_limits():
    assert float(dowell(1e-3, 1)) == pytest.approx(1.0, abs=1e-9)
    assert float(dowell(8.0, 1)) == pytest.approx(8.0, rel=1e-3)  # skin-only foil: F_R -> Delta
    assert float(dowell(1.0, 8)) > 5 * float(dowell(1.0, 1))  # proximity dominates with layers (7.2x at Delta = 1)
    delta = 0.2e-3
    a = 10 * delta
    assert rac_round(2 * a, delta) == pytest.approx(a / (2 * delta) + 0.25, rel=5e-3)


def test_core_loss_screen_is_synthetic_and_flags_definitions():
    js = run("loss_screen", "nominal")
    # headline = worst outcome; the missing material data is a scope note next to it
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert "MISSING_INPUT" in {v["code"] for v in js["verdicts"]}
    assert metric(js, "ratio") == pytest.approx(0.9321, abs=2e-4)  # iGSE triangle vs sine, alpha 1.4, beta 2.6
    assert metric(js, "P_wrongB") / metric(js, "P_sine") == pytest.approx(2**2.6, rel=1e-9)


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("FL07").experiment("winding_flux")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "textbook", {"k_split": "0", "Lm": "2 mF", "Np": "50.5"})
    assert {"k_split", "Lm", "Np"} <= set(e.value.errors)


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL07")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
