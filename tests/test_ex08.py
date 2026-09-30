"""EX08 electro-thermal coupling: from one peak temperature to a mission.

Independence map:
  * textbook numbers typed literally (24/25 K fixed, 26.569592/27.751513 K coupled, 91.57/92.75 C, rho 0.09789)
  * 2x2 coupled rise: reference/thermal.py (Cramer's rule, closed-form spectral radius) vs the lab's linear solve,
    fixed-point iteration, and the steady state and exact transient of the 8-node physical network
  * dynamic network: reciprocity from two separate integrations, port matrix from the conductances, modal solution
  * mission: the fixed-loss energy against a hand sum over the segment definitions, the ASTM E1049 rainflow example,
    seed reproducibility; calibrated lifetime is never produced
"""

import math

import numpy as np
import pytest

from convlab.engine.switched import simulate
from convlab.labs import get_lab
from convlab.labs.ex08_electrothermal import CITY, damage_proxy, eigen_step, port_matrix, rainflow, reversals, two_die_network
from convlab.labs.fl03_thermal import values_at
from convlab.model.params import ParamError, resolve_params
from convlab.reference import thermal as ref

Z_TB = [[0.2, 0.05], [0.05, 0.25]]
P_TB = [100.0, 80.0]


def run(exp, preset=None, **over):
    e = get_lab("EX08").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def series(js, key):
    return next(s for s in js["series"] if s["key"] == key)


def check(js, prefix):
    return next(c for c in js["checks"] if c["name"].startswith(prefix))


# ---- experiment 1: thermal matrix ------------------------------------------------------


def test_thermal_matrix_textbook_numbers():
    js = run("thermal_matrix", "textbook")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "fix1") == pytest.approx(24.0, abs=1e-12)
    assert metric(js, "fix2") == pytest.approx(25.0, abs=1e-12)
    assert metric(js, "cp1") == pytest.approx(26.569592, abs=5e-7)
    assert metric(js, "cp2") == pytest.approx(27.751513, abs=5e-7)
    assert metric(js, "Tj1") == pytest.approx(91.57, abs=5e-3)
    assert metric(js, "Tj2") == pytest.approx(92.75, abs=5e-3)
    assert metric(js, "rho") == pytest.approx(0.09789, abs=5e-6)
    # independent closed forms (Cramer's rule, 2x2 eigenvalues)
    r1, r2 = ref.coupled_rise_2x2(Z_TB, P_TB, 0.004)
    assert metric(js, "cp1") == pytest.approx(r1, rel=1e-13)
    assert metric(js, "cp2") == pytest.approx(r2, rel=1e-13)
    M = [[0.2 * 0.4, 0.05 * 0.32], [0.05 * 0.4, 0.25 * 0.32]]
    assert metric(js, "rho") == pytest.approx(ref.spectral_radius_2x2(M), rel=1e-13)
    assert metric(js, "rho") == pytest.approx(0.08 + math.sqrt(0.016 * 0.02), rel=1e-13)
    for c in js["checks"]:
        assert c["status"] == "PASS", c


def test_fixed_point_iteration_contracts_at_the_spectral_radius():
    js = run("thermal_matrix", "textbook")
    s = series(js, "it_err")
    k, e = np.array(s["x"]), np.array(s["y"])
    sel = (k >= 4) & (k <= 9)
    ratios = e[sel][1:] / e[sel][:-1]
    assert ratios == pytest.approx(0.0978885, rel=2e-2)
    it1 = series(js, "it1")["y"]
    assert it1[0] == 0.0 and it1[1] == pytest.approx(24.0)  # first iterate = fixed-loss rise


def test_divergence_has_no_stable_fixed_point():
    js = run("thermal_matrix", "diverge")
    assert js["status"]["code"] == "NO_STABLE_FIXED_POINT"
    assert metric(js, "rho") == pytest.approx(1 + math.sqrt(0.05), rel=1e-12)  # eig of [[1, .2], [.25, 1]]
    assert isinstance(metric(js, "cp1"), str)
    # the linear solve still returns numbers: -100 / -120 K (non-physical)
    r1, r2 = ref.coupled_rise_2x2(Z_TB, P_TB, 0.05)
    assert (r1, r2) == pytest.approx((-100.0, -120.0), rel=1e-12)
    rows = next(t for t in js["tables"] if t["key"] == "t_methods")["rows"]
    assert "-100" in rows[0][1] and "-120" in rows[0][1] and "비물리" in rows[0][1]
    assert rows[1][2] == "발산" and "불안정" in rows[2][2]
    assert check(js, "발산 속도")["status"] == "PASS"  # exact transient grows at the largest eigenvalue


def test_near_divergence_is_flagged_outside_the_loss_model_range():
    js = run("thermal_matrix", "near")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
    r1, _ = ref.coupled_rise_2x2(Z_TB, P_TB, 0.035)
    assert metric(js, "cp1") == pytest.approx(r1, rel=1e-12)
    assert metric(js, "Tj1") > 175.0
    for c in js["checks"]:
        assert c["status"] == "PASS", c


def test_without_cross_term_each_device_is_a_single_node():
    js = run("thermal_matrix", "no_cross")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "fix1") == pytest.approx(20.0) and metric(js, "fix2") == pytest.approx(20.0)
    assert metric(js, "cp1") == pytest.approx(20.0 / (1 - 0.004 * 0.2 * 100), rel=1e-12)
    assert metric(js, "cp2") == pytest.approx(20.0 / (1 - 0.004 * 0.25 * 80), rel=1e-12)
    assert check(js, "물리 망의 port 행렬")["status"] == "NOT_RUN"  # a passive T-network needs 0 < Z12


# ---- experiment 2: dynamic network ---------------------------------------------------


def test_physical_network_realizes_the_matrix_and_is_reciprocal():
    net = two_die_network(0.2, 0.05, 0.25, [(0.0, (1.0, 0.0), 0.0)])
    np.testing.assert_allclose(port_matrix(net), Z_TB, atol=1e-14)
    # reciprocity by two separate integrations
    t = np.geomspace(1e-3, 100, 40)
    n1 = two_die_network(0.2, 0.05, 0.25, [(0.0, (1.0, 0.0), 0.0)])
    n2 = two_die_network(0.2, 0.05, 0.25, [(0.0, (0.0, 1.0), 0.0)])
    z21 = values_at(simulate(n1, 0, [0.0] * 8, 0.0, 100.0), t, "j2")
    z12 = values_at(simulate(n2, 0, [0.0] * 8, 0.0, 100.0), t, "j1")
    np.testing.assert_allclose(z12, z21, rtol=1e-9, atol=1e-15)
    # modal solution of the same network
    z11 = values_at(simulate(n1, 0, [0.0] * 8, 0.0, 100.0), t, "j1")
    np.testing.assert_allclose(eigen_step(n1, (1.0, 0.0), t, 0), z11, rtol=1e-9, atol=1e-14)


def test_dynamic_network_cross_is_delayed_and_foster_cannot_connect_it():
    js = run("dynamic_network", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    assert metric(js, "t63_cross") > 10 * metric(js, "t63_self")
    # a positive Foster sum cannot follow a delayed response, even with a dense tau grid
    assert metric(js, "fit_cross_nnls") > 0.01
    assert metric(js, "fit_cross_pos") > 0.01
    assert metric(js, "fit_cross_sig") < 0.25 * metric(js, "fit_cross_pos")
    assert metric(js, "n_neg") >= 1
    # heat injected at an internal node of the self Foster chain: wrong either way the blocks are ordered
    assert abs(metric(js, "wrong_err")) > 0.1 * 0.05
    assert abs(metric(js, "wrong_err2")) > 0.1 * 0.05
    light = run("dynamic_network", "light_sink")
    assert metric(light, "t63_cross") < metric(js, "t63_cross")
    assert metric(light, "t63_self") == pytest.approx(metric(js, "t63_self"))


def test_dynamic_network_rejects_a_non_realizable_matrix():
    js = run("dynamic_network", "nominal", Z12=0.3)
    assert js["status"]["code"] == "OUT_OF_VALIDITY"


# ---- experiment 3: mission -------------------------------------------------------------


def test_rainflow_reproduces_the_astm_e1049_example():
    cyc = rainflow(reversals([-2, 1, -3, 5, -1, 3, -4, 4, -2]))
    agg: dict = {}
    for rng, _, cnt in cyc:
        agg[rng] = agg.get(rng, 0.0) + cnt
    assert agg == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}
    # reversal extraction: monotone runs collapse, wiggles below the hysteresis are ignored
    assert reversals([0, 1, 2, 3, 2, 1, 1.05, 0.5, 4], hyst=0.1) == [0, 3, 0.5, 4]


def test_damage_proxy_is_relative_only():
    Ea = 0.8 * 1.602176634e-19
    assert damage_proxy([(10.0, 100.0, 1.0)], 5.0, Ea, 10.0, 100.0) == pytest.approx(1.0, rel=1e-15)
    assert damage_proxy([(20.0, 100.0, 1.0)], 5.0, Ea, 10.0, 100.0) == pytest.approx(32.0, rel=1e-12)
    assert damage_proxy([(10.0, 100.0, 0.5)], 5.0, Ea, 10.0, 100.0) == pytest.approx(0.5, rel=1e-15)


def test_mission_reports_cycles_and_a_relative_proxy_but_no_lifetime():
    js = run("mission", "nominal")
    codes = [v["code"] for v in js["verdicts"]]
    assert "PASS_WITHIN_MODEL" in codes and "MISSING_INPUT" in codes
    assert js["status"]["code"] == "MISSING_INPUT"
    assert metric(js, "life_years") == "MISSING_INPUT"
    for m in js["metrics"]:
        assert m["unit"] not in ("년", "years", "h", "hours"), m
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    # the mission length and the fixed-loss energy by hand from the segment definitions
    d_city = sum(d for d, _, _ in CITY)
    e_city = sum(d * f for d, f, _ in CITY)
    assert metric(js, "t_mission") == pytest.approx(10 * d_city + 300 + 120 + 5 * d_city + 300)
    # 196 965 J; the engine integrates the stiff 8-node network with long matrix-exponential steps (~1e-8 relative)
    assert metric(js, "E_loss_u") == pytest.approx((10 * e_city + 300 * 1.0 + 120 * 1.4 + 5 * e_city + 300 * 0.05) * 180.0, rel=1e-6)
    assert metric(js, "E_loss_u") == pytest.approx(196965.0, rel=1e-6)
    # coupling raises the loss, the peak and the relative proxy
    assert metric(js, "E_loss_c") > metric(js, "E_loss_u")
    assert metric(js, "Tpk_c") > metric(js, "Tpk_u")
    assert metric(js, "D_ratio") > 1.0
    assert metric(js, "mc_out") == 0


def test_mission_monte_carlo_is_reproducible_from_its_seed():
    a = run("mission", "nominal", n_mc=6)
    b = run("mission", "nominal", n_mc=6)
    c = run("mission", "nominal", n_mc=6, seed=7)
    assert a["extra"]["mc_samples"]["rows"] == b["extra"]["mc_samples"]["rows"]
    assert metric(a, "mc_D") == metric(b, "mc_D")
    assert a["extra"]["mc_samples"]["rows"] != c["extra"]["mc_samples"]["rows"]
    assert a["extra"]["mc_samples"]["seed"] == 2026


def test_mission_hot_hill_raises_the_peak():
    nom, hot = run("mission", "nominal"), run("mission", "hot_hill")
    assert metric(hot, "Tpk_c") > metric(nom, "Tpk_c") + 10.0


def test_mission_runaway_is_not_extrapolated():
    js = run("mission", "runaway")
    codes = [v["code"] for v in js["verdicts"]]
    assert js["status"]["code"] == "NO_STABLE_FIXED_POINT"
    assert "OUT_OF_VALIDITY" in codes and "MISSING_INPUT" in codes
    assert isinstance(metric(js, "Tpk_c"), str)
    assert isinstance(metric(js, "D_ratio"), str)
    assert 0 < metric(js, "t_valid") < metric(js, "t_mission")
    t = series(js, "T1c")["x"]
    assert max(t) <= metric(js, "t_valid") + 1e-9  # the coupled trace stops where the model stops


# ---- general ---------------------------------------------------------------------------


def test_inputs_are_rejected_not_clamped():
    e = get_lab("EX08").experiment("thermal_matrix")
    with pytest.raises(ParamError) as exc:
        resolve_params(e.params, e.presets, "textbook", {"Z11": "-0.1", "alpha": "0.5"})
    assert "Z11" in exc.value.errors and "alpha" in exc.value.errors
    m = get_lab("EX08").experiment("mission")
    with pytest.raises(ParamError):
        resolve_params(m.params, m.presets, "nominal", {"n_mc": "2.5"})
    with pytest.raises(ParamError):
        resolve_params(m.params, m.presets, "nominal", {"sig_R": "-0.1"})


def test_all_reference_presets_checks_pass():
    lab = get_lab("EX08")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
