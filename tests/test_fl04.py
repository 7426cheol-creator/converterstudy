"""FL04 inverter: dq operating point, voltage margin, i_d-i_q plane, short switching run.

Independence map:
  * textbook/instruction numbers are typed in literally (independent of labs/ and reference/)
  * abc-frame reconstruction (explicit salient L(theta) + complex-step Faraday) vs dq closed form
  * Park transform of the abc inductance matrix is checked here with numpy (independent of the lab)
  * switched simulation (matrix exponential, oscillator states) vs analytic operating point, vs a
    hand-written RK45 dq ODE inside the lab, vs the Kolar DC-link formula typed in this test
  * plane intersections re-solved here by plain bisection
  * preset/metric plumbing: regression
"""

import math

import numpy as np
import pytest

from convlab.labs import get_lab
from convlab.labs.fl04_inverter import Machine, abc_check, pwm_edges, pulse_fourier
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("FL04").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


M = Machine(4, 15e-3, 0.15e-3, 0.35e-3, 0.115)
WE = 4 * 6000 * 2 * math.pi / 60


def test_textbook_operating_point_numbers():
    js = run("dq_point", "nominal")
    assert metric(js, "T") == pytest.approx(397.887, abs=5e-4)
    assert metric(js, "iq") == pytest.approx(427.835869, abs=1e-6)
    assert metric(js, "V_req") == pytest.approx(438.545444, abs=1e-6)
    assert metric(js, "V_avail") == pytest.approx(438.786205, abs=1e-6)
    assert metric(js, "margin") == pytest.approx(0.241, abs=5e-4)
    assert metric(js, "I_rms") == pytest.approx(333.949, abs=5e-4)
    assert metric(js, "P_cu") == pytest.approx(5018.0, abs=0.5)
    assert metric(js, "P_ac") == pytest.approx(255018.0, abs=0.5)
    # channel-only conduction: 3 I_rms^2 R, not 6x
    assert metric(js, "P_cond") == pytest.approx(1071.0, abs=0.5)
    assert metric(js, "P_cond") == pytest.approx(3 * 333.9487**2 * 3.2e-3, rel=1e-5)
    # the boundary point is reported as MARGINAL, never as a plain PASS
    assert js["status"]["code"] == "MARGINAL"


def test_bus_sag_760V_is_not_automatically_feasible():
    js = run("dq_point", "sag760")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "V_avail") == pytest.approx(760 / math.sqrt(3) * 0.95, rel=1e-12)
    assert metric(js, "V_avail") == pytest.approx(416.847, abs=1e-3)
    assert metric(js, "margin") == pytest.approx(416.8469 - 438.5454, abs=2e-3)
    # the required voltage does not depend on the bus
    assert metric(js, "V_req") == pytest.approx(438.545444, abs=1e-6)


def test_abc_inductance_matrix_park_transforms_to_Ld_Lq():
    for th in (0.0, 0.3, 1.7, 4.0):
        L = M.abc_inductance(th)
        ph = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        T = (2.0 / 3.0) * np.array([np.cos(th - ph), -np.sin(th - ph)])  # abc -> dq, amplitude invariant
        Tinv = np.array([np.cos(th - ph), -np.sin(th - ph)]).T
        Ldq = T @ L @ Tinv
        assert Ldq == pytest.approx(np.diag([0.15e-3, 0.35e-3]), abs=1e-15)


def test_abc_reconstruction_is_independent_and_agrees():
    iq = 397.8873577 / (1.5 * 4 * (0.115 + (0.15e-3 - 0.35e-3) * -200.0))
    ab = abc_check(M, WE, -200.0, iq)
    assert ab["V_peak"] == pytest.approx(438.545444, abs=1e-6)
    assert ab["P_ac"] == pytest.approx(250e3 + 5018.479, rel=1e-6)
    assert ab["P_cu"] == pytest.approx(5018.479, rel=1e-6)


def test_margin_is_not_robust_to_ordinary_tolerances():
    js = run("dq_point", "nominal")
    fails = int(metric(js, "n_fail").split("/")[0])
    assert fails >= 4
    # hand calculation: psi_m -2 % raises i_q and therefore |v_d|
    psi = 0.115 * 0.98
    iq = 397.8873577 / (1.5 * 4 * (psi - 0.2e-3 * -200.0))
    v = math.hypot(15e-3 * -200 - WE * 0.35e-3 * iq, 15e-3 * iq + WE * (0.15e-3 * -200 + psi))
    assert v > 438.786205


def test_mtpa_moves_the_same_torque_to_a_robust_margin():
    js = run("dq_point", "mtpa")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "margin") == pytest.approx(11.65, abs=0.05)
    assert metric(js, "I_pk") < 472.27


def test_plane_intersection_matches_plain_bisection():
    js = run("dq_plane", "nominal")
    T = 397.8873577297383
    V = 760 / math.sqrt(3) * 0.95

    def f(i_d):
        iq = T / (1.5 * 4 * (0.115 - 0.2e-3 * i_d))
        return math.hypot(15e-3 * i_d - WE * 0.35e-3 * iq, 15e-3 * iq + WE * (0.15e-3 * i_d + 0.115)) - V

    a, b = -600.0, -200.0
    for _ in range(200):
        c = 0.5 * (a + b)
        if f(c) > 0:
            b = c
        else:
            a = c
    assert metric(js, "id_sag") == pytest.approx(0.5 * (a + b), abs=1e-7)
    assert metric(js, "id_mtpa") == pytest.approx(-219.3457, abs=1e-3)
    assert metric(js, "I_char") == pytest.approx(0.115 / 0.15e-3, rel=1e-12)
    assert js["status"]["code"] == "MARGINAL"  # on the 800 V ellipse (0.24 V inside), outside the 760 V one
    assert metric(js, "mg2") < 0 < metric(js, "mg1")


def test_switching_run_reproduces_the_analytic_point():
    js = run("inverter_switching", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "fsw") == pytest.approx(10e3, rel=1e-12)
    assert metric(js, "id_mean") == pytest.approx(-200.0, abs=0.2)
    assert metric(js, "iq_mean") == pytest.approx(427.835869, abs=0.2)
    assert metric(js, "V1_pos") == pytest.approx(438.545444, rel=5e-4)
    # DC-link mean = P_AC / Vdc and ripple RMS vs Kolar & Round, both typed here
    assert metric(js, "idc_mean") == pytest.approx(255018.48 / 800, rel=2e-3)
    I, Mi, cphi = 333.9487, 438.545444 / 400.0, 0.8208636
    kolar = I * math.sqrt(2 * Mi * (math.sqrt(3) / (4 * math.pi) + cphi**2 * (math.sqrt(3) / math.pi - 9 * Mi / 16)))
    assert metric(js, "idc_ac") == pytest.approx(kolar, rel=0.01)
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    led = next(c for c in js["checks"] if c["name"].startswith("에너지 잔차"))
    assert abs(led["value"]) < 1e-9


def test_natural_sampled_spwm_fundamental_is_exact():
    # limiting case: pure sinusoidal reference inside the linear range -> fundamental equals the command
    vd, vq = -379.34408602150535, 220.04583847200496
    edges, S0, _ = pwm_edges(vd, vq, WE, 1000.0, 25, 1, "natural", "none")
    _, pos, neg = pulse_fourier(edges, S0, 1000.0, WE, 2 * math.pi / WE)
    assert abs(pos) == pytest.approx(math.hypot(vd, vq), rel=1e-12)
    assert math.atan2(pos.imag, pos.real) == pytest.approx(math.atan2(vq, vd), abs=1e-12)


def test_regular_sampling_delay_and_negative_sequence():
    js = run("inverter_switching", "regular")
    # half a carrier period: w_e * T_s / 2 = 2*pi*400 Hz * 50 us / 2 -> 7.2 degrees lag
    assert metric(js, "dang") == pytest.approx(-7.2, abs=1e-3)
    assert abs(metric(js, "id_mean") - -200.0) > 50.0
    assert metric(run("inverter_switching", "N27", cycles=1), "V1_neg") < 1e-9  # vanishes for N = 27
    assert metric(run("inverter_switching", "nominal", cycles=1), "V1_neg") > 0.1


def test_switching_failure_cases():
    js = run("inverter_switching", "spwm")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "V1_pos") < 438.545 - 5.0  # sinusoidal PWM linear limit is Vdc/2 = 400 V
    js = run("inverter_switching", "sag760")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "pulse_min") < 1e-7


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("FL04").experiment("dq_point")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "nominal", {"m_margin": "1.2", "Ld": "100 uF", "Vdc": "5 kV"})
    assert {"m_margin", "Ld", "Vdc"} <= set(e.value.errors)
    vals, _, _ = resolve_params(exp.params, exp.presets, "nominal", {"Ld": "150 uH", "P_shaft": "0.25 MW"})
    assert vals["Ld"] == pytest.approx(0.15e-3) and vals["P_shaft"] == pytest.approx(250e3)


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL04")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
