"""FL01 Buck/Boost.

Independence map (docs/TEST_INDEPENDENCE.md):
  * textbook numbers are typed in here literally (independent of reference/ and labs/)
  * DCM ratio: reference/buck_boost.py closed form vs exact switched simulation (independent)
  * energy residual: port powers vs stored energy (independent of the state matrices)
  * RK45 path inside the lab: hand-written ODE vs matrix-exponential engine (independent)
  * preset/metric plumbing: regression
"""

import math

import pytest

from convlab.engine.switched import simulate
from convlab.labs import get_lab
from convlab.labs.fl01_buck_boost import Buck, buck_steady
from convlab.model.params import ParamError, resolve_params
from convlab.reference import buck_boost as ref


def run(exp_key, preset=None, **over):
    exp = get_lab("FL01").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) if not isinstance(v, str) else v for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_buck_textbook_numbers():
    js = run("buck_ccm", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "Vo") == pytest.approx(12.0, rel=1e-6)
    assert metric(js, "IL_avg") == pytest.approx(5.0, rel=1e-6)
    # textbook: 0.9 App, peak 5.45 A, valley 4.55 A, RMS 5.00675 A (output ripple shifts these by < 0.05 %)
    assert metric(js, "dI_pp") == pytest.approx(0.9, rel=5e-4)
    assert metric(js, "I_peak") == pytest.approx(5.45, rel=5e-4)
    assert metric(js, "I_valley") == pytest.approx(4.55, rel=5e-4)
    assert metric(js, "IL_rms") == pytest.approx(5.00675, rel=2e-5)
    assert metric(js, "I_boundary") == pytest.approx(0.45, rel=1e-12)
    # high-side average = input DC current = D * I_L
    assert metric(js, "Q1_avg") == pytest.approx(0.25 * 5.0, rel=1e-4)


def test_buck_all_checks_pass_and_energy_closes():
    js = run("buck_ccm", "nominal")
    for c in js["checks"]:
        assert c["status"] in ("PASS", "INFO"), c
    led = next(c for c in js["checks"] if c["name"].startswith("에너지 잔차"))
    assert abs(led["value"]) < 1e-9


def test_buck_half_L_doubles_ripple_keeps_average():
    a = run("buck_ccm", "nominal")
    b = run("buck_ccm", "nominal", L="50")
    assert metric(b, "dI_pp") / metric(a, "dI_pp") == pytest.approx(2.0, rel=2e-3)
    assert metric(b, "IL_avg") == pytest.approx(metric(a, "IL_avg"), rel=1e-6)
    assert metric(b, "I_peak") == pytest.approx(5.9, rel=1e-3)


def test_buck_half_fs_textbook_diagnostic():
    js = run("buck_ccm", "half_fs")
    assert metric(js, "dI_pp") == pytest.approx(1.8, rel=2e-3)
    assert metric(js, "I_peak") == pytest.approx(5.9, rel=1e-3)
    assert metric(js, "I_boundary") == pytest.approx(0.9, rel=1e-9)


def test_losses_are_coupled_not_postprocessed():
    js = run("buck_ccm", "lossy")
    # resistive drops lower the output below D*Vin, and the ledger still closes
    assert metric(js, "Vo") < 12.0 - 0.05
    led = next(c for c in js["checks"] if c["name"].startswith("에너지 잔차"))
    assert abs(led["value"]) < 1e-9
    assert 0.9 < metric(js, "eff") < 1.0


def test_dcm_ratio_matches_closed_form_and_sync_goes_negative():
    js = run("buck_light_load", "light")
    R = 0.25 * 48 / 0.3
    M = ref.buck_dcm_ratio(0.25, 100e-6, R, 100e3)
    assert metric(js, "Vo_de") == pytest.approx(M * 48, rel=2e-4)
    assert metric(js, "Vo_de") == pytest.approx(14.2337, rel=2e-4)
    assert metric(js, "Vo_sync") == pytest.approx(12.0, rel=1e-6)
    assert metric(js, "iL_min_sync") == pytest.approx(0.3 - 0.45, abs=2e-3)
    assert metric(js, "iL_min_de") == pytest.approx(0.0, abs=1e-12)
    assert metric(js, "Vo_at_Dreq") == pytest.approx(12.0, rel=1e-4)
    assert metric(js, "D_req") == pytest.approx(0.2041, abs=1e-4)


def test_dcm_idle_interval_holds_current_at_zero():
    vals = {"Vin": 48, "D": 0.25, "L": 100e-6, "fs": 100e3, "Io": 0.3, "C": 100e-6, "esr": 0.01, "dcr": 0, "rds_hi": 0, "rds_lo": 0, "vf": 0.7, "rect": "diode_emulation", "R": 40.0}
    sys = Buck(vals)
    sol = buck_steady(sys)
    tr = simulate(sys, "H", sol.x0, 0, 1e-5)
    z = [s for s in tr.segments if s.q == "Z"]
    assert z, "expected a DCM idle segment"
    for s in z:
        assert s.z0[0] == 0.0 and abs(s.z1[0]) < 1e-15


def test_periodic_state_is_periodic_over_many_cycles():
    vals = {"Vin": 48, "D": 0.25, "L": 100e-6, "fs": 100e3, "Io": 5, "C": 100e-6, "esr": 0.01, "dcr": 0, "rds_hi": 0, "rds_lo": 0, "vf": 0.7, "rect": "sync", "R": 2.4}
    sys = Buck(vals)
    sol = buck_steady(sys)
    tr = simulate(sys, "H", sol.x0, 0, 50e-5)
    assert tr.z_end[0] == pytest.approx(sol.x0[0], rel=1e-9)
    assert tr.z_end[1] == pytest.approx(sol.x0[1], rel=1e-9)
    # Floquet multipliers inside the unit circle: the orbit attracts
    assert sol.spectral_radius < 1.0


def test_ripple_components_and_phase():
    js = run("buck_ripple", "nominal")
    assert metric(js, "dVc") == pytest.approx(0.01125, rel=0.01)
    assert metric(js, "dVesr") == pytest.approx(0.009, rel=0.01)
    assert metric(js, "dVo") < metric(js, "naive")
    assert 0.5 < metric(js, "ratio") < 0.9


def test_boost_textbook_and_rhp_zero():
    js = run("boost_rhpz", "nominal")
    assert metric(js, "Vo") == pytest.approx(400.0, rel=1e-4)
    assert metric(js, "Iin") == pytest.approx(5.0, rel=1e-4)
    assert metric(js, "dI") == pytest.approx(2.0, rel=1e-6)
    assert metric(js, "Iout") == pytest.approx(2.5, rel=1e-4)
    assert metric(js, "f_rhpz") == pytest.approx(160 * 0.25 / (2 * math.pi * 500e-6), rel=1e-12)
    assert metric(js, "f_rhpz") == pytest.approx(12.73e3, rel=1e-3)
    # inverse response in both models, and the two models agree on its size
    assert metric(js, "dip_sw") < 0 and metric(js, "dip_avg") < 0
    assert metric(js, "dip_sw") == pytest.approx(metric(js, "dip_avg"), rel=0.1)


def test_boost_step_down_mirrors():
    js = run("boost_rhpz", "down")
    # a duty decrease first *raises* the output (mirror image); the 'dip' metric is the minimum after the step
    assert metric(js, "Vo_final_ideal") < 400


def test_rhp_zero_moves_with_load():
    a = run("boost_rhpz", "nominal")
    b = run("boost_rhpz", "light")
    assert metric(b, "f_rhpz") / metric(a, "f_rhpz") == pytest.approx(4.0, rel=1e-9)


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("FL01").experiment("buck_ccm")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "nominal", {"D": "1.5", "L": "100 uF"})
    assert "D" in e.value.errors and "L" in e.value.errors
    vals, _, _ = resolve_params(exp.params, exp.presets, "nominal", {"L": "0.1 mH", "fs": "50k"})
    assert vals["L"] == pytest.approx(100e-6) and vals["fs"] == pytest.approx(50e3)
