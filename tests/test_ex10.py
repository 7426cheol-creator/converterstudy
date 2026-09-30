"""EX10 fault energy: passive RLC fault, protection timeline, discharge resistor, ASC vs freewheel.

Independence map:
  * textbook/instruction numbers typed literally here
  * RLC: closed form written in this test vs the exact switched engine (lab) vs DOP853 (lab check)
  * discharge: RC closed form typed here vs the exact engine's level-crossing root
  * timeline: the textbook sum 1.8 us / margin 1.2 us typed here; clearing times re-derived in the
    lab by an independent discrete-event clock (checked through the preset checks)
  * ASC steady state: formula typed here vs the lab's numerical equilibrium of its ASC model
  * freewheel threshold V_dc/(sqrt(3) p psi) typed here vs the diode-bridge hybrid simulation
  * preset/metric plumbing: regression
"""

import math

import pytest

from convlab.engine.switched import simulate
from convlab.labs import get_lab
from convlab.labs.ex10_fault_energy import Freewheel, RLCFault, asc_equilibrium
from convlab.labs.fl04_inverter import Machine
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("EX10").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


M = Machine(4, 15e-3, 0.15e-3, 0.35e-3, 0.115)


def test_rlc_textbook_numbers():
    js = run("rlc_fault", "textbook")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "E0") == pytest.approx(320.0, rel=1e-12)
    assert metric(js, "alpha") == pytest.approx(5000.0, rel=1e-12)
    assert metric(js, "wd") == pytest.approx(8660.254, abs=1e-3)
    assert metric(js, "t_pk") == pytest.approx(120.919958e-6, abs=1e-12)
    assert metric(js, "i_pk") == pytest.approx(4370.344, abs=1e-3)
    assert metric(js, "i10") == pytest.approx(760.0, abs=0.1)
    assert metric(js, "i100") == pytest.approx(4268.0, abs=0.1)
    assert metric(js, "vC_min") == pytest.approx(-800 * math.exp(-5000 * math.pi / 8660.254), rel=1e-6)  # -130.4 V


def test_rlc_engine_matches_closed_form_typed_here():
    V0, L, C, R = 800.0, 10e-6, 1e-3, 0.1
    a, wd = R / (2 * L), math.sqrt(1 / (L * C) - (R / (2 * L)) ** 2)
    tr = simulate(RLCFault(L, C, R, False, 0.0), "RLC", [0.0, V0], 0.0, 400e-6)
    for t in (5e-6, 37e-6, 120.919958e-6, 250e-6, 399e-6):
        i_an = V0 / (L * wd) * math.exp(-a * t) * math.sin(wd * t)
        assert tr.state_at(t)[0][0] == pytest.approx(i_an, abs=1e-9 * 4370)


def test_rlc_diode_clamp_changes_the_energy_path():
    js = run("rlc_fault", "clamp")
    for c in js["checks"]:
        assert c["status"] == "PASS", c
    assert metric(js, "vC_min") >= -3.0 - 1e-9  # clamped at -V_cl instead of reversing to -130 V
    # closed form typed here: v_C(t) = V0 e^{-at}(cos wt + a/w sin wt) reaches -V_cl = -3 V just after 2 t_pk
    V0, L, a, w = 800.0, 10e-6, 5000.0, math.sqrt(1e8 - 2.5e7)

    def vc(t):
        return V0 * math.exp(-a * t) * (math.cos(w * t) + a / w * math.sin(w * t))

    lo, hi = 2 * 120.919958e-6, 300e-6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if vc(mid) > -3.0 else (lo, mid)
    t_cl = 0.5 * (lo + hi)
    i_cl = V0 / (L * w) * math.exp(-a * t_cl) * math.sin(w * t_cl)
    assert metric(js, "t_clamp") == pytest.approx(t_cl, rel=1e-9)
    assert metric(js, "W_L_clamp") == pytest.approx(0.5 * L * i_cl**2, rel=1e-8)
    assert 27.5 < metric(js, "W_L_clamp") < 28.5  # vs 28.50 J at exactly 0 V (2 t_pk)
    assert metric(js, "W_end") == pytest.approx(0.5 * 1e-3 * 3.0**2, abs=1e-9)  # only the clamped charge is left


def test_timeline_textbook_sum_and_worst_case():
    js = run("protection_timeline", "textbook")
    assert js["status"]["code"] == "SCREEN_ONLY"
    assert metric(js, "t_clear_typ") == pytest.approx(1.8e-6, abs=1e-12)  # 0.5 + 0.3 + 0.2 + 0.8 us
    assert metric(js, "margin_typ") == pytest.approx(1.2e-6, abs=1e-12)
    assert metric(js, "t_clear_max") == pytest.approx(1.8e-6 * 1.4, abs=1e-12)
    assert metric(js, "t_desat") == pytest.approx(3000 * 100e-9 / 800, rel=1e-12)
    for c in js["checks"]:
        assert c["status"] == "PASS", c


def test_timeline_fault_types_and_failures():
    t1 = metric(run("protection_timeline", "textbook"), "t_clear_typ")
    t2 = metric(run("protection_timeline", "type2"), "t_clear_typ")
    assert t2 == pytest.approx(0.375e-6 + 0.3e-6 + 0.2e-6 + 0.8e-6, abs=1e-12)  # blanking already expired
    assert t2 < t1
    js = run("protection_timeline", "soft_short")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "t_clear_max") > 3e-6
    js = run("protection_timeline", "fast_off")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    # overshoot: 800 V + 15 nH * 3000 A / 0.1 us = 1250 V (typ)
    assert metric(js, "Vpk") == pytest.approx(800 + 15e-9 * 3000 / 0.1e-6, rel=1e-12)


def test_discharge_resistor_textbook_numbers_and_tolerance():
    js = run("discharge_resistor", "textbook")
    R = 2.0 / (1e-3 * math.log(800 / 60))
    assert R == pytest.approx(772.121, abs=1e-3)
    assert metric(js, "R_nom") == pytest.approx(772.121, abs=1e-3)
    assert metric(js, "P0") == pytest.approx(828.885, abs=1e-3)
    assert metric(js, "E_rem") == pytest.approx(318.2, abs=1e-9)
    assert metric(js, "t60") == pytest.approx(2.0, abs=1e-9)
    # slow corner R +5 %, C +10 % misses the (educational) 2 s requirement
    assert metric(js, "t_slow") == pytest.approx(2.0 * 1.05 * 1.10, rel=1e-12)
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    js = run("discharge_resistor", "worst_case")
    assert metric(js, "R_used") == pytest.approx(R / 1.10 / 1.05, rel=1e-12)
    assert metric(js, "t_slow") == pytest.approx(2.0, rel=1e-12)
    assert js["status"]["code"] == "MISSING_INPUT"  # numbers fine, final part selection blocked by missing data


def test_asc_steady_state_formula_and_transient():
    for rpm in (1500.0, 6000.0, 12000.0):
        we = 4 * rpm * 2 * math.pi / 60
        den = 15e-3**2 + we**2 * 0.15e-3 * 0.35e-3
        i_d, i_q = -(we**2) * 0.35e-3 * 0.115 / den, -15e-3 * we * 0.115 / den
        d, q = asc_equilibrium(M, we)
        assert (d, q) == pytest.approx((i_d, i_q), rel=1e-12)
    js = run("safe_state", "nominal")
    assert js["status"]["code"] == "CUSTOMER_DECISION_REQUIRED"
    assert metric(js, "I_asc_ss") == pytest.approx(0.115 / 0.15e-3, rel=1e-3)  # -> psi/L_d at high speed
    assert metric(js, "I_asc_pk") > metric(js, "I_asc_ss")
    assert metric(js, "T_asc_ss") < 0


def test_freewheel_threshold_and_rectification():
    n_th = 800 / (math.sqrt(3) * 4 * 0.115) * 60 / (2 * math.pi)
    assert n_th == pytest.approx(9588.33, abs=0.01)
    for fac, flows in ((0.99, False), (1.01, True)):
        we = 4 * n_th * fac * 2 * math.pi / 60
        fw = Freewheel(M, we, 800.0, True, 1e-3, 5e4)
        tr = simulate(fw, None, [0.0, 0.0, 1.0, 0.0], 0.0, 4 * 2 * math.pi / we)
        idc = tr.mean(tr.t_end - 2 * math.pi / we, tr.t_end, "idc")
        assert (idc > 1e-3) == flows, (fac, idc)
    js = run("safe_state", "nominal")
    assert metric(js, "n_th") == pytest.approx(n_th, rel=1e-12)
    assert metric(js, "fw_rect") == "아니오"
    assert metric(js, "W_mag") == pytest.approx(0.75 * (0.15e-3 * 200**2 + 0.35e-3 * 427.8359**2), rel=1e-6)
    js = run("safe_state", "overspeed")
    assert metric(js, "E_ll") == pytest.approx(math.sqrt(3) * 4 * 12000 * 2 * math.pi / 60 * 0.115, rel=1e-12)
    assert metric(js, "fw_rect") == "예" and metric(js, "fw_idc") > 50
    assert metric(js, "fw_T") < 0
    js = run("safe_state", "overspeed_open")
    assert metric(js, "vdc_max") > 900.0  # battery disconnected: the link charges towards the 1001 V EMF peak


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("EX10").experiment("discharge_resistor")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "textbook", {"tol_R": "60", "C": "1 mH", "N_ser": "2.5"})
    assert {"tol_R", "C", "N_ser"} <= set(e.value.errors)
    exp = get_lab("EX10").experiment("protection_timeline")
    with pytest.raises(ParamError):
        resolve_params(exp.params, exp.presets, "textbook", {"t_soft": "0"})


def test_all_reference_presets_checks_pass():
    lab = get_lab("EX10")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
