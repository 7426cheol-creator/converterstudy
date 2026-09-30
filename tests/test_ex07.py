"""EX07 system stability: CPL exact example, finite-bandwidth converter, digital delay, saturation.

Independence map:
  * textbook numbers are typed in literally (394.935887 V, -15.5974 ohm, 320.565519 uF,
    220.57 +- j3134.19 and -67.94 +- j991.24 1/s, 200 kW, 5.4 deg and 54 deg for 15 us)
  * CPL poles: analytic A vs a central-difference Jacobian written here vs a separate RK45
    integration written here (growth over a fixed time)
  * finite-bandwidth converter: Nyquist count vs Jacobian eigenvalues (inside the lab); with input
    feedforward the averaged converter must reproduce the ideal-CPL poles of experiment 1
  * delay phase: sample/hold simulation vs 360 f T_d (inside the lab) and the timing rules here
  * saturation: dq model vs abc integration (inside the lab); strategy ranking; bumpless jump = sag
  * preset/metric plumbing: regression
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from convlab.labs import get_lab
from convlab.labs.ex07_system_stability import timing, zoh_phase_sim
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("EX07").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def check(js, prefix):
    return next(c for c in js["checks"] if c["name"].startswith(prefix))


def test_cpl_textbook_numbers_100uF_unstable():
    js = run("cpl_exact", "c100u")
    assert js["status"]["code"] == "UNSTABLE"
    assert metric(js, "Ve") == pytest.approx(394.935887, abs=5e-7)
    assert metric(js, "Rinc") == pytest.approx(-15.5974, abs=5e-5)
    assert metric(js, "Ccrit") == pytest.approx(320.565519e-6, abs=5e-13)
    assert metric(js, "pole_re") == pytest.approx(220.57, abs=0.005)
    assert metric(js, "pole_im") == pytest.approx(3134.19, abs=0.005)
    assert metric(js, "Pmax") == pytest.approx(200e3, rel=1e-12)
    # the nonlinear integration grows at the linear rate and stops where the ideal CPL leaves validity
    assert metric(js, "sig_nl") == pytest.approx(220.57, rel=0.02)
    assert 0 < metric(js, "t_exit") < 30e-3


def test_cpl_1mF_stable():
    js = run("cpl_exact", "c1m")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "pole_re") == pytest.approx(-67.94, abs=0.005)
    assert metric(js, "pole_im") == pytest.approx(991.24, abs=0.005)
    assert metric(js, "sig_nl") < 0


def test_cpl_no_solution_above_pmax():
    js = run("cpl_exact", "p250k")
    assert js["status"]["code"] == "NO_SOLUTION"


def test_cpl_jacobian_and_integration_written_here():
    Vs, R, L, P = 400.0, 0.2, 1e-3, 10e3
    Ve = (Vs + math.sqrt(Vs * Vs - 4 * R * P)) / 2
    for C, sig_ref in ((100e-6, 220.5655), (1e-3, -67.9434)):
        f = lambda x: np.array([(Vs - R * x[0] - x[1]) / L, (x[0] - P / x[1]) / C])  # noqa: E731
        xe = np.array([P / Ve, Ve])
        J = np.column_stack([(f(xe + h) - f(xe - h)) / (2 * 1e-4) for h in (np.array([1e-4, 0]), np.array([0, 1e-4]))])
        lam = np.linalg.eigvals(J)
        assert max(lam.real) == pytest.approx(sig_ref, abs=1e-3)
        # independent RK45 run: the envelope over whole oscillation periods follows exp(sigma t)
        Tosc = 2 * math.pi / abs(lam[0].imag)
        sol = solve_ivp(lambda t, x: f(x), (0, 4 * Tosc), xe + [0, 0.01], method="RK45", rtol=1e-10, atol=1e-12, dense_output=True)
        amp_0 = np.max(np.abs(sol.sol(np.linspace(0, Tosc, 4001))[1] - Ve))
        amp_3 = np.max(np.abs(sol.sol(np.linspace(3 * Tosc, 4 * Tosc, 4001))[1] - Ve))
        assert math.log(amp_3 / amp_0) / (3 * Tosc) == pytest.approx(sig_ref, rel=0.05, abs=2.0)


def test_converter_loops_stable_but_system_unstable():
    js = run("converter_impedance", "nominal")
    assert js["status"]["code"] == "UNSTABLE"
    assert metric(js, "pm_i") > 45 and metric(js, "pm_v") > 45  # single loops look fine
    assert metric(js, "conv_stable") < 0  # converter alone (stiff source) is stable
    assert metric(js, "n_unst") == 2 and metric(js, "nyq") == 2
    assert check(js, "Nyquist 영점 수")["status"] == "PASS"


def test_input_feedforward_reproduces_the_ideal_cpl_poles():
    js = run("converter_impedance", "feedforward")
    assert js["status"]["code"] == "UNSTABLE"
    assert metric(js, "sys_re") == pytest.approx(220.57, abs=0.01)
    assert metric(js, "sys_im") == pytest.approx(3134.19, abs=0.01)


def test_finite_bandwidth_is_less_conservative_than_the_ideal_cpl():
    js = run("converter_impedance", "slow")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "cpl_re") > 0  # the ideal CPL with the same filter would be unstable
    assert metric(js, "mb") > 1  # Middlebrook magnitude rule violated, yet stable: not necessary
    assert metric(js, "nyq") == 0 == metric(js, "n_unst")
    big = run("converter_impedance", "big_c")
    assert big["status"]["code"] == "PASS_WITHIN_MODEL" and metric(big, "mb") < 1


def test_corner_table_has_five_corners():
    js = run("converter_impedance", "nominal")
    t = next(t for t in js["tables"] if t["key"] == "t_corner")
    assert len(t["rows"]) >= 3
    assert any("360 V" in r[0] for r in t["rows"]) and any("440 V" in r[0] for r in t["rows"])


def test_delay_phase_textbook_and_timing_rules():
    js = run("digital_delay", "td15")
    assert metric(js, "Td") == pytest.approx(15e-6, rel=1e-12)
    assert metric(js, "lag1") == pytest.approx(5.4, abs=1e-9)
    assert metric(js, "lag2") == pytest.approx(54.0, abs=1e-9)
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    base = {"f_pwm": 50e3, "update": "double", "t_calc": 8e-6, "policy": "next"}
    assert timing(base)["Td"] == pytest.approx(15e-6)
    assert timing(dict(base, t_calc=12e-6))["Td"] == pytest.approx(25e-6)  # missed update event
    assert timing(dict(base, update="single"))["Td"] == pytest.approx(30e-6)
    assert timing(dict(base, policy="immediate"))["Td"] == pytest.approx(13e-6)
    assert zoh_phase_sim(1e3, 10e-6, 10e-6) == pytest.approx(5.4, abs=1e-9)


def test_missed_update_makes_the_10kHz_loop_unstable():
    js = run("digital_delay", "slip")
    assert js["status"]["code"] == "UNSTABLE"
    assert metric(js, "lag2") == pytest.approx(90.0, abs=1e-9)
    assert metric(js, "rho_10000") > 1.0 > metric(js, "rho_1000")


def test_saturation_strategies_and_bumpless():
    ax = run("saturation_transfer", "axis")
    ci = run("saturation_transfer", "circle")
    na = run("saturation_transfer", "noaw")
    assert metric(ci, "iae") < metric(ax, "iae") < metric(na, "iae")
    assert metric(ax, "iae_ratio") > 1.05
    assert na["status"]["code"] == "MARGINAL" and metric(na, "id_ov") > 10
    assert metric(ax, "mag_max") > metric(ax, "Vmax")  # the controller asks for an infeasible vector
    # bumpless: without it the command jumps by the 10 % sag (the stale integrator), with it by 0
    assert metric(ax, "jump_no") == pytest.approx(0.1 * 400 * math.sqrt(2) / math.sqrt(3), rel=1e-6)
    assert abs(metric(ax, "jump_bl")) < 1e-9
    assert metric(ax, "dev_no") > 1.0 and metric(ax, "dev_bl") < 1e-9
    assert check(ax, "dq 포화 궤적")["status"] == "PASS"


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("EX07").experiment("cpl_exact")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "c100u", {"C": "-1", "L": "1 mF"})
    assert "C" in e.value.errors and "L" in e.value.errors
    vals, _, _ = resolve_params(exp.params, exp.presets, "c100u", {"C": "1m"})
    assert vals["C"] == pytest.approx(1e-3)


def test_all_reference_presets_checks_pass():
    lab = get_lab("EX07")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
