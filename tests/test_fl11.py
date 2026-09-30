"""FL11 PSFB, HV-LV and the 12 V extension.

Independence map (docs/TEST_INDEPENDENCE.md):
  * textbook numbers are typed in literally (seed 48 V, DAB 0.328973 rad / 2.09431 A / 2.01988 A /
    33.6647 A / 31.25 A / 4 kW, 62.5 A, 250 A, 3.91 W, 62.5 W)
  * duty loss: the L_k volt-second identity is written out here by hand and compared with the
    exact switched simulation (independent)
  * reference/psfb.py constant-V_o PWL steady state vs exact switched simulation (independent)
  * hand-written scalar ODE + RK45 events vs matrix-exponential engine (independent)
  * DAB closed forms vs PWL integration (independent); energy ledger (independent)
  * preset/metric plumbing and statuses: regression
"""

import math

import numpy as np
import pytest

from convlab.engine.switched import simulate
from convlab.labs import get_lab
from convlab.labs.fl11_psfb import PSFB, ivp_periods, psfb_steady, run_period
from convlab.model.params import ParamError, resolve_params
from convlab.reference import psfb as ref


def run(exp_key, preset=None, **over):
    exp = get_lab("FL11").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


NOMINAL = dict(Vin=800.0, n=12.0, fs=100e3, phi_deg=129.6, Lk=20e-6, Lo=10e-6, Co=100e-6, R=48.0**2 / 1500.0, rect="fb", rect_dev="sr")


def test_textbook_seed_48V_in_the_small_leakage_limit():
    js = run("psfb_duty_loss", "seed_ideal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    # textbook seed: 800 V * 0.72 / 12 = 48 V
    assert 800 * 0.72 / 12 == pytest.approx(48.0)
    assert metric(js, "Vo") == pytest.approx(48.0, rel=2e-4)
    assert metric(js, "D_eff") == pytest.approx(0.72, abs=1e-4)


def test_duty_loss_matches_hand_written_volt_second_identity():
    s = PSFB(NOMINAL)
    sol = psfb_steady(s)
    assert sol.converged
    tr = run_period(s, sol.x0, 1)
    T = 1e-5
    Vo = tr.mean(0, T, "vo")
    I0 = sol.x0[1]  # output-inductor current when v_AB reverses
    # half-period L_k volt-seconds: n*mean(v_rect) = V_in*D - 4 L_k f_s I0 / n  (written out here)
    dD_hand = 4 * 20e-6 * 100e3 * I0 / (12 * 800)
    assert 0.72 - 12 * Vo / 800 == pytest.approx(dD_hand, rel=1e-9)
    # the average-current textbook-style formula overestimates by the ripple
    Io = tr.mean(0, T, "iLo")
    assert 4 * 20e-6 * 100e3 * Io / (12 * 800) > dD_hand
    assert 46.0 < Vo < 47.0 and dD_hand == pytest.approx(0.02276, abs=2e-5)


def test_commutation_interval_and_sr_overlap():
    js = run("psfb_duty_loss", "nominal")
    tc = metric(js, "t_c")
    # primary current swings 2 I0/n through L_k at V_in (plus the slow L_o term): ~113 ns
    assert tc == pytest.approx(2 * metric(js, "I0") / (12 * 800 / 20e-6 + metric(js, "Vo") / 10e-6), rel=1e-3)
    assert 100e-9 < tc < 130e-9
    s = PSFB(NOMINAL)
    sol = psfb_steady(s)
    tr = run_period(s, sol.x0, 1)
    # during commutation both rectifier paths conduct and their currents add up to i_Lo
    seg = next(sg for sg in tr.segments if sg.q[1] == "C")
    z, _ = tr.state_at(seg.t0 + 0.5 * seg.h)
    out = s.outputs(seg.q)
    i1, i2, il = float(out["iSR1"] @ z), float(out["iSR2"] @ z), float(out["iLo"] @ z)
    assert i1 > 0 and i2 > 0 and i1 + i2 == pytest.approx(il, rel=1e-12)


def test_pwl_constant_vo_path_is_independent_and_agrees():
    pw = ref.psfb_pwl_steady(800, 12, 100e3, math.radians(129.6), 20e-6, 10e-6, 48**2 / 1500)
    s = PSFB(NOMINAL)
    sol = psfb_steady(s)
    tr = run_period(s, sol.x0, 1)
    assert tr.mean(0, 1e-5, "vo") == pytest.approx(pw["Vo"], rel=1e-5)
    assert tr.rms(0, 1e-5, "iP") == pytest.approx(pw["ip_rms"], rel=1e-4)
    assert tr.rms(0, 1e-5, "iSR1") == pytest.approx(pw["sr_rms"], rel=1e-4)


def test_rk45_scalar_ode_path_agrees_with_exact_engine():
    p = dict(NOMINAL, R_sr=2e-3, R_w=1e-3, R_pri=0.15, R_Lo=1e-3, Vf=0.0)
    s = PSFB(p)
    sol = psfb_steady(s)
    x0 = 0.8 * np.asarray(sol.x0)
    tr = simulate(s, ("fwt", "N"), x0, 0.0, 10 * s.T)
    xr = ivp_periods(s, x0, 1e-10, 10, rc0="N")
    assert np.max(np.abs(xr - tr.z_end[:3]) / np.array([2.6, 31.0, 48.0])) < 1e-8


def test_regulated_48V_needs_more_phase_than_the_seed():
    js = run("psfb_duty_loss", "regulate48")
    assert metric(js, "Vo") == pytest.approx(48.0, rel=1e-6)
    assert metric(js, "phi") > 129.6 + 3.0
    assert metric(js, "D_cmd") - 0.72 == pytest.approx(metric(js, "dD"), abs=1e-6)


def test_low_line_corner_cannot_make_48V_with_n12():
    # ideal duty alone would exceed 1: 48 * 12 / 550 = 1.047
    assert 48 * 12 / 550 > 1.0
    js = run("psfb_duty_loss", "low_line")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "Vo") < 48.0


def test_losses_are_coupled_and_the_ledger_closes():
    js = run("psfb_duty_loss", "lossy_sr")
    led = next(c for c in js["checks"] if c["name"].startswith("에너지 잔차"))
    assert abs(led["value"]) < 1e-9
    assert 0.99 < metric(js, "eff") < 1.0
    assert metric(js, "Vo") < 46.48


def test_center_tap_blocks_twice_the_voltage_with_one_device_conducting():
    fb = run("psfb_duty_loss", "nominal")
    ct = run("psfb_duty_loss", "nominal", rect="ct")
    assert metric(ct, "V_sr") == pytest.approx(2 * metric(fb, "V_sr"))
    assert metric(fb, "V_sr") == pytest.approx(800 / 12)
    # lossless: identical output and SR current waveforms
    assert metric(ct, "Vo") == pytest.approx(metric(fb, "Vo"), rel=1e-9)


def test_light_load_dcm_raises_the_effective_duty():
    js = run("psfb_duty_loss", "light")
    assert js["status"]["code"] == "INFO"
    assert metric(js, "D_eff") > metric(js, "D_cmd")
    assert metric(js, "t_c") == 0.0


def test_lk_tradeoff_directions():
    js = run("lk_tradeoff", "nominal")
    tab = next(t for t in js["tables"] if t["key"] == "t_lk")["rows"]
    dD = [r[2] for r in tab]
    fmin = [r[7] for r in tab]
    nmax = [r[9] for r in tab]
    assert all(b > a for a, b in zip(dD, dD[1:]))
    assert all(b < a for a, b in zip(fmin, fmin[1:]))
    assert all(b < a for a, b in zip(nmax, nmax[1:]))
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert "SCREEN_ONLY" in {v["code"] for v in js["verdicts"]}
    assert 20.0 < metric(js, "Lk_full") < 30.0  # 20 uH is marginal at full load


def test_dab_textbook_numbers_and_comparison():
    js = run("psfb_vs_dab", "nominal")
    assert metric(js, "dab_phi") == pytest.approx(0.328973, abs=1e-6)
    assert metric(js, "dab_ipk") == pytest.approx(2.09431, abs=1e-5)
    assert metric(js, "dab_irms") == pytest.approx(2.01988, abs=1e-5)
    assert metric(js, "dab_isec") == pytest.approx(33.6647, abs=1e-4)
    assert metric(js, "dab_iout") == pytest.approx(31.25)
    assert metric(js, "dab_pmax") == pytest.approx(4000.0)
    # the PSFB steps down with duty (n = 12) and circulates the reflected load current in freewheel
    assert metric(js, "psfb_ip_rms") > metric(js, "dab_irms")
    assert metric(js, "psfb_iout") == pytest.approx(31.25, rel=1e-6)


def test_dab_closed_form_vs_independent_pwl():
    phi = ref.dab_phi_for_power(800, 800, 100e3, 200e-6, 1500)
    pw = ref.dab_pwl(800, 800, 100e3, 200e-6, phi)
    assert pw["P"] == pytest.approx(1500.0, rel=1e-12)
    ipk = 800 * phi / (2 * math.pi * 100e3 * 200e-6)
    assert pw["ipk"] == pytest.approx(ipk, rel=1e-12)
    assert pw["irms"] == pytest.approx(ipk * math.sqrt(1 - 2 * phi / (3 * math.pi)), rel=1e-12)


def test_12V_extension_textbook_numbers():
    js = run("lv12_extension", "nominal")
    assert metric(js, "I_hi") == pytest.approx(62.5)
    assert metric(js, "I_lo") == pytest.approx(250.0)
    assert metric(js, "L_hi") == pytest.approx(3.90625)
    assert round(metric(js, "L_hi"), 2) == 3.91
    assert metric(js, "L_lo") == pytest.approx(62.5)
    assert metric(js, "ratio_L") == pytest.approx(16.0)
    assert metric(js, "n_lo") == pytest.approx(48.0) and metric(js, "n_hi") == pytest.approx(12.0)
    # secondary wiring inductance is seen n^2 times larger at 12 V
    assert metric(js, "dD_lo") > metric(js, "dD_hi")
    # the fixed path does not shrink with paralleling
    assert metric(js, "floor12") > 0.5 * metric(js, "loss12")


def test_12V_wiring_inductance_failure():
    js = run("lv12_extension", "stray_fail")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL11")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("FL11").experiment("psfb_duty_loss")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "nominal", {"phi_deg": "190", "Lk": "20 uF"})
    assert "phi_deg" in e.value.errors and "Lk" in e.value.errors
    vals, _, _ = resolve_params(exp.params, exp.presets, "nominal", {"Lk": "0.02 mH"})
    assert vals["Lk"] == pytest.approx(20e-6)
