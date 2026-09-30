"""FL06 control: PI units, delay, saturation / anti-windup, dq sign.

Independence map:
  * textbook numbers are typed in literally (Kp 5.026548 V/A, Ki 628.318531 V/(A s), 37.5 us -> 13.5 deg)
  * crossover / phase margin: numerical root finding on the complex loop gain vs the design target
  * discrete closed-loop poles: characteristic polynomial written here vs the lab's state matrix
  * ZOH map vs exact continuous engine simulation (inside the lab run) and vs the analytic step here
  * dq model vs independent abc simulation (three phase ODEs + Park); a deliberately wrong
    cross-coupling sign must be caught by the same comparison
  * preset/metric plumbing: regression
"""

import math

import numpy as np
import pytest

import convlab.labs.fl06_control as fl06
from convlab.labs import get_lab
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("FL06").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_textbook_pi_numbers_and_delay():
    js = run("pi_design", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "Kp_V") == pytest.approx(5.026548, abs=5e-7)
    assert metric(js, "Ki_V") == pytest.approx(628.318531, abs=5e-7)
    assert metric(js, "Td") == pytest.approx(37.5e-6, rel=1e-12)
    assert metric(js, "lag_fc") == pytest.approx(13.5, abs=1e-9)
    # duty-output gains are the volt gains divided by the modulator gain (800 V/duty)
    assert metric(js, "Kp_d") == pytest.approx(5.026548 / 800, rel=1e-6)
    assert metric(js, "Ki_d") == pytest.approx(628.318531 / 800, rel=1e-6)


def test_crossover_and_margin_found_numerically():
    Kp, Ki = 0.8e-3 * 2 * math.pi * 1000, 0.1 * 2 * math.pi * 1000
    m0 = fl06.margins(lambda f: fl06.loop_gain_cont(f, Kp, Ki, 0.8e-3, 0.1), 1.0, 1e6)
    assert m0.fc == pytest.approx(1000.0, rel=1e-9)
    assert m0.pm == pytest.approx(90.0, abs=1e-6)
    md = fl06.margins(lambda f: fl06.loop_gain_cont(f, Kp, Ki, 0.8e-3, 0.1, 37.5e-6), 1.0, 1e6)
    assert md.fc == pytest.approx(1000.0, rel=1e-9)
    assert md.pm == pytest.approx(90.0 - 13.5, abs=1e-6)


def test_duty_output_is_the_same_loop_and_unit_error_is_unstable():
    a = run("pi_design", "nominal")
    b = run("pi_design", "duty")
    assert metric(b, "fc_disc") == pytest.approx(metric(a, "fc_disc"), rel=1e-12)
    assert metric(b, "pm_disc") == pytest.approx(metric(a, "pm_disc"), rel=1e-12)
    c = run("pi_design", "unit_error")
    assert c["status"]["code"] == "UNSTABLE"
    assert metric(c, "rho") > 1.0
    assert metric(c, "gm_disc") < 0.0


def test_discrete_poles_match_characteristic_polynomial():
    L, R, Ts = 0.8e-3, 0.1, 25e-6
    Kp, Ki = L * 2 * math.pi * 1000, R * 2 * math.pi * 1000
    a = math.exp(-R * Ts / L)
    b = (1 - a) / R
    for nd in (0, 1, 2):
        # (z-1)(z-a) z^nd + b (Kp (z-1) + Ki Ts) = 0
        poly = np.polymul(np.polymul([1, -1], [1, -a]), [1] + [0] * nd)
        poly = np.polyadd(poly, b * np.array([Kp, -Kp + Ki * Ts]))
        ref = np.sort_complex(np.roots(poly))
        got = np.sort_complex(fl06.closed_loop_poles(Kp, Ki, L, R, Ts, nd))
        got = got[np.abs(got) > 1e-12] if nd else got
        assert np.allclose(np.sort_complex(got), ref, atol=1e-9)


def test_zoh_map_against_analytic_step():
    L, R, Ts = 0.8e-3, 0.1, 25e-6
    a, b = fl06.rl_zoh(L, R, Ts)
    # constant 10 V from i = 0 for 7 samples: i = (10/R)(1 - exp(-R t/L))
    i = 0.0
    for _ in range(7):
        i = a * i + b * 10.0
    assert i == pytest.approx(10 / R * (1 - math.exp(-R * 7 * Ts / L)), rel=1e-12)


def test_antiwindup_reduces_overshoot_and_recovery():
    js = run("discrete_loop", "load_step")
    assert metric(js, "sat_time") > 0.0  # the step really saturates (V_dc 600 V)
    assert metric(js, "ov_noaw") > metric(js, "ov_aw") + 1.0  # percent
    assert metric(js, "rec_noaw") > 3 * metric(js, "rec_aw")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    # the engine check of the ZOH map and the energy ledger are in the run
    zc = next(c for c in js["checks"] if c["name"].startswith("ZOH"))
    assert zc["status"] == "PASS" and zc["independent"] and zc["value"] < 1e-9
    led = next(c for c in js["checks"] if c["name"].startswith("에너지 잔차"))
    assert abs(led["value"]) < 1e-9


def test_too_aggressive_antiwindup_drags_the_integrator():
    js = run("discrete_loop", "aw_aggressive")
    assert js["status"]["code"] == "MARGINAL"
    assert metric(js, "peak_aw") < 20.0 * 0.95  # undershoot tail instead of overshoot


def test_sensor_offset_stays_in_the_real_current():
    js = run("discrete_loop", "sensor_offset")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "ss_err") == pytest.approx(-0.5, abs=1e-3)


def test_feedforward_matters_for_the_input_dip():
    a = run("discrete_loop", "input_dip")
    b = run("discrete_loop", "input_dip_noff")
    assert metric(b, "dev_aw") > 4 * metric(a, "dev_aw")
    assert b["status"]["code"] == "MARGINAL"


def test_extra_delay_and_limit_rows_exist():
    js = run("discrete_loop", "current_limit")
    assert metric(js, "lim_viol") >= 0.0
    t = next(t for t in js["tables"] if t["key"] == "t_scen")
    assert len(t["rows"]) == 5


def test_dq_correct_sign_tracks_and_power_factor():
    js = run("dq_sign", "correct")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "Vgd") == pytest.approx(400 * math.sqrt(2) / math.sqrt(3), rel=1e-12)
    assert metric(js, "P") == pytest.approx(1.5 * 326.5986 * 20.0, rel=1e-3)
    assert metric(js, "frame_dev") < 1e-9
    js2 = run("dq_sign", "reactive")
    # i_q > 0 with this transform: current leads the voltage, Q (absorbed, inductive +) < 0
    assert metric(js2, "Q") == pytest.approx(-1.5 * 326.5986 * 5.0, rel=2e-3)


def test_dq_reversed_sign_is_unstable():
    js = run("dq_sign", "reversed")
    assert js["status"]["code"] == "UNSTABLE"
    assert metric(js, "rho") > 1.0
    assert 0 < metric(js, "t_trip") < 3e-3


def test_abc_comparison_catches_a_wrong_cross_coupling_sign(monkeypatch):
    exp = get_lab("FL06").experiment("dq_sign")
    vals, _, _ = resolve_params(exp.params, exp.presets, "correct", {})
    good = fl06.dq_plant_map

    def wrong(L, R, w, Ts):
        return good(L, R, -w, Ts)  # flips the +wL i_q / -wL i_d terms

    monkeypatch.setattr(fl06, "dq_plant_map", wrong)
    rec, _, _ = fl06.run_dq(vals, +1, 200, 1e9)
    ta, da, qa, _, _, _ = fl06.run_abc(vals, +1, 200, 1e9)
    dev = np.max(np.hypot(da - rec["id"][: len(da)], qa - rec["iq"][: len(qa)]))
    assert dev > 0.05  # amperes: the independent path disagrees


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("FL06").experiment("pi_design")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "nominal", {"fc": "0", "out": "amps"})
    assert "fc" in e.value.errors and "out" in e.value.errors
    vals, _, _ = resolve_params(exp.params, exp.presets, "nominal", {"L": "800 uH", "fs": "20k"})
    assert vals["L"] == pytest.approx(0.8e-3) and vals["fs"] == pytest.approx(20e3)


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL06")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
