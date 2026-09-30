"""EX03 parallel SiC, gate loop and DPT metrology.

Independence map:
  * textbook numbers typed literally (110.553/99.497/99.497/90.452 A, 533.333 uJ, +41.99 %, -33.01 %, 4 V, 20 A)
  * static sharing: reference/parallel.py current divider vs the lab's KCL nodal solve
  * overlap energy: reference/parallel.py closed forms vs the lab's exact PWL algebra; a dense trapezoid here
  * dynamic sharing: energy ledger, tolerance tightening, and the N = 4 identical-branch cell vs a separately
    built N = 1 equivalent (different matrices and state dimension); tested again here directly
"""

import numpy as np
import pytest

from convlab.labs import get_lab
from convlab.labs.ex03_parallel_sic import _dyn_spec, _equiv_single_spec, _turn_on_run, lpf_pwl, overlap_energy_pwl
from convlab.model.params import ParamError, resolve_params
from convlab.reference import parallel as ref


def run(exp, preset=None, **over):
    e = get_lab("EX03").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_static_sharing_textbook_currents():
    js = run("static_sharing", "textbook")
    for k, ref_I in enumerate((110.553, 99.497, 99.497, 90.452)):
        assert metric(js, f"I{k + 1}") == pytest.approx(ref_I, abs=5e-4)
    cur = ref.current_divider([3.6e-3, 4.0e-3, 4.0e-3, 4.4e-3], 400.0)
    assert [metric(js, f"I{k + 1}") for k in range(4)] == pytest.approx(cur, rel=1e-12)
    assert metric(js, "spread") == pytest.approx((cur[0] - cur[3]) / 100.0, rel=1e-12)


def test_positive_tempco_balances_only_partly():
    js = run("static_sharing", "textbook")
    assert 0 < metric(js, "spread_et") < metric(js, "spread")  # better, not equal
    iso = run("static_sharing", "no_tempco")
    assert metric(iso, "spread_et") == pytest.approx(metric(iso, "spread"), rel=1e-9)
    tim = run("static_sharing", "bad_tim")
    assert metric(tim, "Iet1") < metric(js, "Iet1")  # hotter branch 1 takes a little less current
    T = next(t for t in tim["tables"] if t["key"] == "t_branch")["rows"]
    assert max(T, key=lambda r: r[6])[0] == 1  # ...but it is still the hottest


def test_overlap_energy_and_deskew_textbook_numbers():
    E0 = overlap_energy_pwl(800, 100, 40e-9, 0.0, -10e-9, 50e-9)
    assert E0 == pytest.approx(533.333e-6, rel=1e-6)
    assert overlap_energy_pwl(800, 100, 40e-9, -5e-9, -10e-9, 50e-9) / E0 - 1 == pytest.approx(0.4199, abs=1e-4)
    assert overlap_energy_pwl(800, 100, 40e-9, 5e-9, -10e-9, 50e-9) / E0 - 1 == pytest.approx(-0.3301, abs=1e-4)
    for tau in (-7e-9, -5e-9, -1e-9, 0.0, 3e-9, 5e-9):
        assert overlap_energy_pwl(800, 100, 40e-9, tau, -20e-9, 60e-9) == pytest.approx(ref.shifted_overlap_energy(800, 100, 40e-9, tau), rel=1e-12)
    # the window must contain the pre-transition interval: the narrow window misses it
    assert overlap_energy_pwl(800, 100, 40e-9, -5e-9, 0.0, 40e-9) / E0 - 1 == pytest.approx(0.37305, abs=1e-4)
    # independent dense sampling in the test
    t = np.linspace(-10e-9, 50e-9, 60001)
    v = np.interp(t, [-1, 0, 40e-9, 1], [800, 800, 0, 0])
    i = np.interp(t + 5e-9, [-1, 0, 40e-9, 1], [0, 0, 100, 100])
    assert np.trapezoid(v * i, t) == pytest.approx(ref.shifted_overlap_energy(800, 100, 40e-9, -5e-9), rel=1e-6)


def test_dpt_deskew_experiment_and_ranking_status():
    js = run("dpt_deskew", "textbook")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "E0") == pytest.approx(533.333e-6, rel=1e-6)
    assert metric(js, "dE_m5") == pytest.approx(0.4199, abs=1e-4)
    assert metric(js, "dE_p5") == pytest.approx(-0.3301, abs=1e-4)
    poor = run("dpt_deskew", "poor_deskew")
    assert poor["status"]["code"] == "UNRESOLVED_RANKING"
    budget = next(t for t in js["tables"] if t["key"] == "t_budget")["rows"]
    assert any("gain" in r[0] and r[1].startswith("+2.00") for r in budget)
    planes = next(t for t in js["tables"] if t["key"] == "t_planes")["rows"]
    off, on = planes
    assert off[1] > off[3] and on[1] < on[3]  # terminal vs die: C_oss energy moves between the events


def test_band_limited_probe_is_exact_first_order_response():
    tk = np.array([-10e-9, 0.0, 40e-9, 60e-9])
    yk = np.array([0.0, 0.0, 100.0, 100.0])
    tau = 5e-9
    t = np.array([20e-9, 40e-9, 60e-9])
    y = lpf_pwl(tk, yk, tau, t)
    k = 100.0 / 40e-9
    ramp = lambda u: k * (u - tau * (1 - np.exp(-u / tau)))  # noqa: E731
    assert y[0] == pytest.approx(ramp(20e-9), rel=1e-12)
    assert y[2] == pytest.approx(ramp(60e-9) - ramp(20e-9), rel=1e-12)


def test_dynamic_sharing_nominal_and_screens():
    js = run("dynamic_sharing", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    assert metric(js, "csi_screen") == pytest.approx(4.0)
    assert metric(js, "skew_screen") == pytest.approx(20.0)
    # simulated common-source voltage = L_s x the simulated di/dt (same order as the 4 V screen)
    csi = next(m for m in js["metrics"] if m["key"] == "csi_sim")
    assert csi["value"] == pytest.approx(csi["ref"], rel=0.02)
    assert 2.0 < metric(js, "didt_on") / 1000 < 4.0  # kA/us: the textbook's regime
    assert get_lab("EX03").implementation == "COMPLETE"


def test_timing_skew_gives_first_order_screen_scale():
    js = run("dynamic_sharing", "skew10")
    dI = metric(js, "pk_spread")
    assert 15.0 < dI < 35.0  # same scale as 10 ns x 2..3 kA/us
    assert metric(js, "skew_sim") == pytest.approx(10e-9, rel=0.1)
    tab = next(t for t in js["tables"] if t["key"] == "t_M")["rows"]
    spreads = [r[5] for r in tab]
    assert max(spreads) - min(spreads) < 0.1 * dI  # the unidentifiable coupling is reported as a (narrow) range


def test_identical_branches_equal_single_equivalent_device():
    e = get_lab("EX03").experiment("dynamic_sharing")
    vals, _, _ = resolve_params(e.params, e.presets, "nominal", {})
    c4, r4, t4 = _turn_on_run(_dyn_spec(vals, identical=True))
    c1, r1, t1 = _turn_on_run(_equiv_single_spec(vals))
    tt = np.linspace(t4, t4 + 250e-9, 300)
    i4 = np.array([np.interp(tt, r4.t, r4.x[c4.i_i][k]) for k in range(4)])
    i1 = np.interp(tt, r1.t, r1.x[c1.i_i][0])
    assert np.max(np.abs(i4 - i1 / 4)) < 1e-3
    assert abs(r4.energy["normalised"]) < 1e-5 and abs(r1.energy["normalised"]) < 1e-5


def test_fast_drive_shows_differential_ringing_hidden_in_the_sum():
    nom = run("dynamic_sharing", "nominal")
    fast = run("dynamic_sharing", "low_rg")
    assert metric(fast, "dm_off") > 5 * metric(nom, "dm_off")
    assert nom["status"]["code"] == "PASS_WITHIN_MODEL" and metric(nom, "dm_off") < 0.2 * 100.0
    assert fast["status"]["code"] == "FAIL_CONSTRAINT" and metric(fast, "dm_off") > 0.2 * 100.0
    for c in fast["checks"]:
        assert c["status"] == "PASS", c  # the model is verified; the design criterion fails


def test_inputs_are_rejected_not_clamped():
    e = get_lab("EX03").experiment("dynamic_sharing")
    with pytest.raises(ParamError) as exc:
        resolve_params(e.params, e.presets, "nominal", {"kM": "0.95", "Rg": "0"})
    assert "kM" in exc.value.errors and "Rg" in exc.value.errors


def test_all_reference_presets_checks_pass():
    lab = get_lab("EX03")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
