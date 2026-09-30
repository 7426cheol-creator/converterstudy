"""FL09 LLC: FHA in three forms, textbook numbers, switching vs FHA, independent integration."""

import math

import pytest

from convlab.labs import get_lab
from convlab.labs._resonant import Tank, edge_currents, fha, fha_time_domain_gain
from convlab.labs.fl09_llc import _solve
from convlab.model.params import resolve_params
from convlab.reference import resonant as ref


def run(exp, preset=None, **over):
    e = get_lab("FL09").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def vals(exp, preset=None, **over):
    e = get_lab("FL09").experiment(exp)
    v, _, _ = resolve_params(e.params, e.presets, preset, {k: str(x) for k, x in over.items()})
    return v


TANK = Tank(40e-6, 28.1448e-9, 200e-6)


@pytest.mark.parametrize("Q", [0.2, 0.8, 1.5])
@pytest.mark.parametrize("F", [0.7, 0.9, 1.0, 1.1, 1.5])
def test_fha_three_independent_forms(F, Q):
    Rac = TANK.Z0 / Q
    g_nodal = fha(TANK, F * TANK.fr1, Rac).gain
    g_textbook = ref.llc_gain(F, 5.0, Q)
    g_time = fha_time_domain_gain(TANK, F * TANK.fr1, Rac)
    assert g_nodal == pytest.approx(g_textbook, abs=1e-12)
    assert g_time == pytest.approx(g_textbook, abs=1e-9)


def test_textbook_numbers():
    js = run("fha_gain", "textbook")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "fr") == pytest.approx(150e3, rel=1e-6)  # textbook 150 kHz
    assert metric(js, "k") == pytest.approx(5.0, rel=1e-12)
    assert metric(js, "n") == pytest.approx(8.3333333, rel=1e-7)  # 400 -> 48 V full bridge
    assert metric(run("fha_gain", "hb"), "n") == pytest.approx(4.1666667, rel=1e-7)  # half bridge


def test_gain_table_regression_values():
    # computed with the textbook normalised formula (regression values, not textbook-stated)
    for F, Q, g in ((0.7, 0.2, 1.2420), (1.5, 0.2, 0.8900), (0.7, 0.8, 1.0171), (1.5, 0.8, 0.7717), (0.7, 1.5, 0.7410), (1.5, 1.5, 0.5979)):
        assert ref.llc_gain(F, 5.0, Q) == pytest.approx(g, abs=5e-5)


def test_heavy_load_below_resonance_gain_below_one_and_capacitive():
    Rac = TANK.Z0 / 1.5
    p = fha(TANK, 0.7 * TANK.fr1, Rac)
    assert p.gain < 1.0 and not p.inductive
    assert ref.llc_zin(0.7 * TANK.fr1, 40e-6, 28.1448e-9, 200e-6, Rac).imag < 0


def test_at_resonance_switching_gain_is_unity():
    js = run("switching_vs_fha", "at_fr")
    assert metric(js, "g_td") == pytest.approx(1.0, abs=1e-3)
    assert metric(js, "g_fha") == pytest.approx(1.0, abs=1e-12)


def test_near_fr_midload_within_five_percent():
    js = run("switching_vs_fha", "mid")
    assert abs(metric(js, "diff")) < 0.05
    c = next(c for c in js["checks"] if c["name"].startswith("FHA 대 switching"))
    assert c["status"] == "PASS"
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"


def test_below_resonance_heavy_load_fha_out_of_validity():
    js = run("switching_vs_fha", "below_heavy")
    assert metric(js, "diff") > 0.2  # switching gain far above FHA (off interval)
    assert metric(js, "off_frac") > 0.3
    assert js["status"]["code"] == "OUT_OF_VALIDITY"


def test_half_bridge_equivalent_to_full_bridge_normalised():
    fb = run("switching_vs_fha", "mid")
    hb = run("switching_vs_fha", "hb")
    assert metric(hb, "g_td") == pytest.approx(metric(fb, "g_td"), rel=1e-3)
    assert metric(hb, "n") == pytest.approx(metric(fb, "n") / 2, rel=1e-12)


def test_all_reference_checks_pass():
    lab = get_lab("FL09")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c["name"], c["value"])
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m["key"])


def test_light_load_edge_current_is_the_magnetizing_peak():
    # limiting case independent of the simulator: i_m,pk = n V_o / (4 L_m f)
    v = vals("switching_vs_fha", "mid")
    tank, sysl, n, kb, Rac, RL, per, s = _solve(v, F=1.1, Q=0.02)
    e = edge_currents(sysl, per.traj, sysl.T)["i_rise"]
    impk = n * s["vo_avg"] / (4 * tank.Lm * 1.1 * tank.fr1)
    assert e < 0
    assert -e == pytest.approx(impk, rel=0.05)


def test_zvs_screen_depends_on_charge_not_only_on_inductive_input():
    js = run("zvs_lm", "short_td")
    rows = next(t for t in js["tables"] if t["key"] == "t_zvs")["rows"]
    inductive_but_short = [r for r in rows if r[4] == "inductive" and "전하 부족" in r[5]]
    assert inductive_but_short, "an inductive FHA point that still fails the charge screen must exist"


def test_curve_peak_moves_below_fha_peak():
    js = run("gain_curve", "mid")
    assert metric(js, "F_peak_td") < metric(js, "F_peak_fha")
    assert metric(js, "max_diff_near") < 0.05
    assert math.isfinite(metric(js, "max_diff"))


def test_zin_boundary_is_per_q_and_matches_an_independent_root():
    # values from Octave fzero on Im Z_in (matlab/xc_llc.m) and by hand from the quadratic in F^2;
    # a table that repeats one Q's boundary on every row (the earlier bug) fails here
    js = run("fha_gain", "textbook")
    rows = {r[0]: r[-1] for r in next(t for t in js["tables"] if t["key"] == "t_gain")["rows"]}
    assert rows["0.2"] == "F < 0.4388 capacitive (범위 안 전부 inductive)"
    assert rows["0.8"] == "F < 0.8442 capacitive"
    assert rows["1.5"] == "F < 0.9554 capacitive"
    for q, f in ((0.2, 0.43884), (0.8, 0.84421), (1.5, 0.95541)):
        assert ref.llc_zin_boundary(5.0, q) == pytest.approx(f, abs=1e-5)
    chk = next(c for c in js["checks"] if c["name"].startswith("∠Z_in 경계"))
    assert chk["status"] == "PASS" and chk["independent"]
