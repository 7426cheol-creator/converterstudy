"""EX01 design space, loss maps and part selection.

Independence map (docs/TEST_INDEPENDENCE.md):
  * textbook numbers typed literally: 1.082353, 1.016401, 136099.47 Hz, 147060.86 Hz, 14.390/14.765 A,
    +0.001871/-0.001803 per kHz, 164.390/162.811 kHz, 46.2481 uH, 24.3424 nF, 10.6988 kW (E12 A, ch.13);
    0.328973 rad, 2.01988 A, 2.09431 A, 3.11356 A, 5.65983 A, 2.16506 A (E12 B, ch.11);
    110.553/99.497/99.497/90.452 A, 4 V, 20 A (E12 C); 8 W against +-15 W -> UNRESOLVED_RANKING (E01)
  * CLLC gain: nodal analysis written here vs the lab's divider algebra (and the lab's own chain-matrix check) (independent)
  * DAB phi = 0 current: triangle-wave RMS written here vs the lab's segment sums (independent)
  * static sharing written here vs the lab; mission efficiency recomputed here as an energy-weighted harmonic mean (independent)
  * statuses (FAIL_CONSTRAINT, CANDIDATE_FHA_ONLY, CUSTOMER_DECISION_REQUIRED, NO_SOLUTION, UNRESOLVED_RANKING, OUT_OF_VALIDITY): regression
"""

import math

import numpy as np
import pytest

from convlab.labs import get_lab
from convlab.labs.ex01_design_space import PROFILES
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("EX01").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def codes(js):
    return {v["code"] for v in js["verdicts"]}


def table(js, key):
    return next(t for t in js["tables"] if t["key"] == key)


def cllc_gain(f, n, Vbat, P, Lr=40e-6, Cr=28.144773e-9, Lm=200e-6):
    """Nodal analysis written here: one unknown node voltage at the magnetizing branch."""
    w = 2 * np.pi * f
    R = n * n * 8 * (Vbat**2 / P) / np.pi**2
    Z1 = 1j * w * Lr + 1 / (1j * w * Cr)
    Zm = 1j * w * Lm
    Zb = Z1 + R
    Vm = (1 / Z1) / (1 / Z1 + 1 / Zm + 1 / Zb)  # for V1 = 1
    H = Vm * R / Zb
    Zin = Z1 + 1 / (1 / Zm + 1 / Zb)
    return H, Zin


def test_seed_cllc_fails_at_the_high_battery_corner():
    # required gain n Vo / Vi = 920/850 = 1.082353; the inductive FHA maximum is 1.016401 (textbook ch.13, E12 A)
    assert 920 / 850 == pytest.approx(1.082353, abs=5e-7)
    f = np.arange(120e3, 210e3, 1.0)
    H, Zin = cllc_gain(f, 1.0, 920.0, 11000.0)
    g_ind = np.abs(H)[np.angle(Zin) > 0].max()
    assert g_ind == pytest.approx(1.016401, abs=2e-6)
    js = run("obc_cllc", "seed")
    assert metric(js, "M_req_seed") == pytest.approx(1.082353, abs=5e-7)
    assert metric(js, "gmax_seed") == pytest.approx(1.016401, abs=2e-6)
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert "CUSTOMER_DECISION_REQUIRED" in codes(js)
    # grid limit sqrt(3) 400 16 0.995 0.97 = 10.6988 kW, about 301 W short of 11 kW
    assert math.sqrt(3) * 400 * 16 * 0.995 * 0.97 == pytest.approx(10698.8, abs=0.05)
    assert metric(js, "P_grid") == pytest.approx(10698.8, abs=0.05)
    assert metric(js, "P_short") == pytest.approx(301.2, abs=0.1)
    assert metric(js, "n_nominal_opt") == pytest.approx(1.0)
    assert metric(js, "n_robust") == pytest.approx(0.93)


def test_n093_has_two_fha_roots_at_the_high_corner():
    js = run("obc_cllc", "n093")
    assert js["status"]["code"] == "CUSTOMER_DECISION_REQUIRED"  # the input-current limit is the worst outcome
    assert "CANDIDATE_FHA_ONLY" in codes(js)
    assert metric(js, "f_left") == pytest.approx(136099.47, abs=0.02)
    assert metric(js, "f_right") == pytest.approx(147060.86, abs=0.02)
    assert metric(js, "I1_left") == pytest.approx(14.390, abs=5e-4)
    assert metric(js, "I1_right") == pytest.approx(14.765, abs=5e-4)
    assert metric(js, "slope_left") == pytest.approx(0.001871, abs=2e-6)
    assert metric(js, "slope_right") == pytest.approx(-0.001803, abs=2e-6)
    assert metric(js, "f_650") == pytest.approx(164390.0, abs=1.0)
    assert metric(js, "f_800") == pytest.approx(162811.0, abs=1.0)
    assert metric(js, "Lr2") == pytest.approx(46.2481e-6, rel=2e-6)
    assert metric(js, "Cr2") == pytest.approx(24.3424e-9, rel=2e-6)
    # independent: the nodal gain at both roots equals the required 0.93 x 920 / 850 = 1.006588
    for f in (metric(js, "f_left"), metric(js, "f_right")):
        H, Zin = cllc_gain(f, 0.93, 920.0, 11000.0)
        assert abs(H) == pytest.approx(0.93 * 920 / 850, rel=1e-9)
        assert np.angle(Zin) > 0  # inductive input impedance
    assert all(c["status"] == "PASS" for c in js["checks"])


def test_lower_rms_root_fails_the_zvs_charge_screen():
    js = run("obc_cllc", "n093_min")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    rows = table(js, "t_corners")["rows"]
    hi = next(r for r in rows if r[0].startswith("고전압 920"))
    assert hi[6] == pytest.approx(136.099, abs=1e-3)  # the minimum-RMS policy picked the left root
    assert "ZVS" in hi[11]


def test_light_load_regulation_requirement_leaves_no_candidate():
    js = run("obc_cllc", "light_reg")
    assert js["status"]["code"] == "NO_SOLUTION"
    assert metric(js, "n_robust") == "없음"


def test_dab_textbook_numbers():
    # phi = 0 at 900 V / 600 V: triangle current with peak (V1 - V2')/(4 f L) = 3.75 A, RMS = peak / sqrt(3)
    assert (900 - 600) / (4 * 100e3 * 200e-6) / math.sqrt(3) == pytest.approx(2.16506, abs=5e-6)
    js = run("dab_lv", "nominal")
    assert metric(js, "phi_nom") == pytest.approx(0.328973, abs=5e-7)
    assert metric(js, "irms_nom") == pytest.approx(2.01988, abs=5e-6)
    assert metric(js, "ipk_nom") == pytest.approx(2.09431, abs=5e-6)
    assert metric(js, "irms_lowbat") == pytest.approx(3.11356, abs=5e-5)
    assert metric(js, "ipk_lowbat") == pytest.approx(5.65983, abs=5e-5)
    assert metric(js, "irms_phi0") == pytest.approx(2.16506, abs=5e-6)
    assert metric(js, "Pmax_nom") == pytest.approx(4000.0)
    assert metric(js, "P_total") == pytest.approx(3000.0)  # 2 x 1.5 kW, never mixed into one module
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert "SCREEN_ONLY" in codes(js)  # ZVS is a sign/charge screen only
    assert all(c["status"] == "PASS" for c in js["checks"])


def test_dab_inductance_tradeoff():
    js = run("dab_lv", "L100")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "irms_phi0") == pytest.approx(2 * 2.16506, abs=1e-5)  # circulating current scales with 1/L
    assert metric(js, "L_nominal_opt") == pytest.approx(100e-6)
    assert metric(js, "L_robust") == pytest.approx(175e-6)
    assert run("dab_lv", "L300")["status"]["code"] == "FAIL_CONSTRAINT"  # 550 V / 36 V needs more than phi_max


def test_sic_leg_textbook_sharing_and_coupling():
    R = [3.6, 4.0, 4.0, 4.4]
    g = [1 / r for r in R]
    expect = [400 * x / sum(g) for x in g]
    assert expect == pytest.approx([110.553, 99.497, 99.497, 90.452], abs=5e-4)
    js = run("sic_leg", "nominal")
    for k, val in enumerate((110.553, 99.497, 99.497, 90.452)):
        assert metric(js, f"I_static_{k + 1}") == pytest.approx(val, abs=5e-4)
    assert metric(js, "V_cs") == pytest.approx(2e-9 * 2e9)  # 2 nH x 2 kA/us = 4 V
    assert metric(js, "skew_dI") == pytest.approx(2e9 * 10e-9)  # 10 ns x 2 kA/us = 20 A
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert "SCREEN_ONLY" in codes(js)
    assert metric(js, "Rg_nominal_opt") == pytest.approx(1.0)
    assert metric(js, "Rg_robust") == pytest.approx(4.7)
    assert metric(js, "I_hot") < 110.553  # the positive temperature coefficient evens the sharing slightly
    js2 = run("sic_leg", "fast")
    assert js2["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js2, "V_cs") == pytest.approx(11.4, rel=1e-9)


def test_kelvin_source_fixes_coupling_but_not_overshoot():
    js = run("sic_leg", "kelvin")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert any("과전압" in v["why"] for v in js["verdicts"])
    js2 = run("sic_leg", "kelvin_lowL")
    assert js2["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js2, "Rg_robust") == pytest.approx(3.3)


def test_mission_ranking_textbook_pressure_question():
    js = run("mission_ranking", "textbook")
    assert metric(js, "dP_nom") == pytest.approx(8.0, abs=1e-6)
    assert metric(js, "u_diff_nom") == pytest.approx(15 * math.sqrt(2), rel=1e-9)  # sqrt(15^2 + 15^2)
    assert metric(js, "rank_nom") == "UNRESOLVED_RANKING"
    assert js["status"]["code"] == "UNRESOLVED_RANKING"
    js2 = run("mission_ranking", "rho_nobasis")
    assert js2["status"]["code"] == "UNRESOLVED_RANKING"  # an unrecorded correlation does not shrink u
    assert "MISSING_INPUT" in codes(js2)
    js3 = run("mission_ranking", "same_fixture")
    assert metric(js3, "u_diff_nom") == pytest.approx(15 * math.sqrt(2 * 0.01), rel=1e-9)
    assert metric(js3, "rank_nom") == "A"


def test_mission_efficiency_is_output_energy_over_input_energy():
    js = run("mission_ranking", "textbook")
    eff = next(s for s in js["series"] if s["key"] == "eff_A")
    prof = PROFILES["cc_cv"]
    Eout = [(p if p is not None else 10500.0) * dt / 60 for _, p, dt in prof]
    eta = eff["y"]
    eta_h = sum(Eout) / sum(e / h for e, h in zip(Eout, eta))  # written here
    assert metric(js, "eta_mission_A") == pytest.approx(eta_h, rel=1e-12)
    assert metric(js, "eta_arith_A") == pytest.approx(sum(eta) / len(eta), rel=1e-12)
    assert abs(metric(js, "eta_arith_A") - metric(js, "eta_mission_A")) > 1e-3  # the average of efficiencies is not it


def test_mission_weighting_can_flip_the_central_ranking():
    js = run("mission_ranking", "textbook")
    js2 = run("mission_ranking", "single_phase")
    assert metric(js, "dE_mission") > 0  # three-phase CC-CV: A loses less energy (central value)
    assert metric(js2, "dE_mission") < 0  # 3.5 kW single-phase: B loses less
    assert js2["status"]["code"] == "UNRESOLVED_RANKING"  # neither is resolved against the model uncertainty


def test_loss_accounting_errors_are_flagged_not_ranked():
    js = run("mission_ranking", "eoss_double")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
    assert metric(js, "dP_nom") < 8.0 - 3.0  # adding recovered Coss energy nearly erases A's advantage
    rows = table(js, "t_accounting")["rows"]
    assert any("중복" in r[4] for r in rows)
    assert run("mission_ranking", "textbook", hard_eon=True)["status"]["code"] == "OUT_OF_VALIDITY"
    assert run("mission_ranking", "deep_taper")["status"]["code"] == "OUT_OF_VALIDITY"  # burst points are UNKNOWN


def test_loss_surface_interpolation_extrapolation_and_ranking():
    js = run("loss_surface", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "query_status") == "INTERPOLATION"
    assert metric(js, "ranking") == "A"
    assert metric(js, "val_power") > metric(js, "val_floor")
    assert metric(js, "light_ratio") < 1.0  # the pure power law underestimates light-load energy
    for pk in ("light", "overvolt"):
        js2 = run("loss_surface", pk)
        assert js2["status"]["code"] == "OUT_OF_VALIDITY"
        assert metric(js2, "query_status") == "EXTRAPOLATION"
        assert "UNKNOWN" in metric(js2, "ranking")
    assert run("loss_surface", "close_call")["status"]["code"] == "UNRESOLVED_RANKING"
    assert run("loss_surface", "power_law")["status"]["code"] == "OUT_OF_VALIDITY"


def test_all_reference_presets_pass_their_checks_and_explain_their_plots():
    lab = get_lab("EX01")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            assert sum(c["independent"] for c in js["checks"]) >= 1, (e.key, pk)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
            for pl in js["plots"]:
                assert pl["proved"].strip() and pl["not_yet"].strip(), (e.key, pk, pl["key"])


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("EX01").experiment("obc_cllc")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "seed", {"n": "2", "Ae": "-1"})
    assert "n" in e.value.errors and "Ae" in e.value.errors
    assert run("obc_cllc", "seed", f_min="220 kHz")["status"]["code"] == "OUT_OF_VALIDITY"  # f_min above f_max
    leg = get_lab("EX01").experiment("sic_leg")
    vals, _, _ = resolve_params(leg.params, leg.presets, "nominal", {"S_ref": "2 kA/us"})
    assert vals["S_ref"] == pytest.approx(2000.0)  # stored in A/us
