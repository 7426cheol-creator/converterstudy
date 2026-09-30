"""FL02 devices, gate drive, DPT and protection.

Independence map:
  * textbook numbers are typed literally here (14 W, 20.5 W, 6 W, 6.1 W, 18.75 kHz, 75 V, 30 A, 10 ns, 1.8 us, 1.2 us)
  * loss comparison: closed form (reference/devices.py) vs PWL waveform integration + edge counting (lab check)
  * gate-charge ODE event times vs closed forms (independent)
  * DPT cell: energy ledger (port/loss/stored terms defined apart from the state equations), tolerance
    tightening, gate-charge identity (solver-integrated i_G vs closed-form Q(v)), C_oss energy vs a
    numerical quadrature done in this test, fast scalar RHS vs vectorised RHS (regression)
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from convlab.labs import get_lab
from convlab.labs.fl02_devices import Cell, CellSpec, DevArrays, DeviceParams
from convlab.model.params import ParamError, resolve_params
from convlab.reference import devices as ref


def run(exp, preset=None, **over):
    e = get_lab("FL02").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def check(js, prefix):
    return next(c for c in js["checks"] if c["name"].startswith(prefix))


# ---- experiment 1: synthetic A/B loss comparison ------------------------------------------


def test_loss_ab_textbook_numbers():
    js = run("loss_ab", "textbook")
    assert metric(js, "PA") == pytest.approx(14.0, rel=1e-12)
    assert metric(js, "PB") == pytest.approx(20.5, rel=1e-12)
    assert metric(js, "PA_20k") == pytest.approx(6.0, rel=1e-12)
    assert metric(js, "PB_20k") == pytest.approx(6.1, rel=1e-12)
    assert metric(js, "f_cross") == pytest.approx(18.75e3, rel=1e-12)
    # closed form in reference/ agrees (independent of the lab code)
    assert ref.device_loss(10, 0.04, 100e-6, 100e3) == pytest.approx(14.0)
    assert ref.crossover_fs(10, 0.04, 100e-6, 0.025, 180e-6) == pytest.approx(18750.0)
    # the gate drive power is reported but never added to the device loss
    assert metric(js, "PgA") == pytest.approx(100e-9 * 22 * 100e3)
    assert js["status"]["code"] == "MISSING_INPUT"  # the real-part dataset is absent
    codes = [v["code"] for v in js["verdicts"]]
    assert "PASS_WITHIN_MODEL" in codes


def test_loss_ab_waveform_path_and_datasheet_template_is_empty():
    js = run("loss_ab", "textbook")
    assert check(js, "A 손실: 식 vs PWL")["status"] == "PASS"
    assert check(js, "교차 주파수")["independent"] is True
    tpl = next(t for t in js["tables"] if t["key"] == "t_datasheet")
    assert all(r[3] == "MISSING_INPUT" and r[4] == "MISSING_INPUT" for r in tpl["rows"])


def test_coss_boundary_unknown_can_leave_ranking_unresolved():
    js = run("loss_ab", "coss_unknown_20k")
    assert js["status"]["code"] == "UNRESOLVED_RANKING"
    js100 = run("loss_ab", "textbook", coss_boundary="unknown")
    assert js100["status"]["code"] != "UNRESOLVED_RANKING"  # at 100 kHz the gap exceeds the uncertainty
    ex = run("loss_ab", "coss_excluded")
    assert metric(ex, "PA") == pytest.approx(14.0 + 20e-6 * 100e3)


def test_parallel_sweep_follows_stated_assumptions():
    js = run("loss_ab", "f20k")
    s = next(x for x in js["series"] if x["key"] == "n_A")
    for N, y in zip(s["x"], s["y"]):
        assert y == pytest.approx(ref.parallel_loss(N, 10, 0.04, 100e-6, 0.3, 100e-9, 22, 20e3)["device_total"], rel=1e-12)
    # conduction ~ 1/N but the charge part ~ N: a minimum exists (not monotonic)
    k = int(np.argmin(s["y"]))
    assert 0 < k < len(s["y"]) - 1


# ---- experiment 2: gate/Miller/loop screens and protection -----------------------------------


def test_gate_protect_textbook_numbers():
    js = run("gate_protect", "textbook")
    assert metric(js, "dV") == pytest.approx(75.0, rel=1e-12)
    assert metric(js, "icm") == pytest.approx(30.0, rel=1e-12)
    assert metric(js, "Ig") == pytest.approx(2.0, rel=1e-12)
    assert metric(js, "tM") == pytest.approx(10e-9, rel=1e-12)
    assert metric(js, "t_chain") == pytest.approx(1.8e-6, rel=1e-12)
    assert metric(js, "t_left") == pytest.approx(1.2e-6, rel=1e-9)
    # independent ODE path for the plateau
    assert metric(js, "t_plateau_ode") == pytest.approx(10e-9, rel=1e-6)
    assert check(js, "plateau 시간")["status"] == "PASS"
    # never a PASS on short-circuit survival: screen + missing real data
    codes = {v["code"] for v in js["verdicts"]}
    assert codes == {"SCREEN_ONLY", "MISSING_INPUT"}


def test_long_blanking_breaks_the_assumed_withstand_time():
    js = run("gate_protect", "blank_long")
    assert metric(js, "t_chain") == pytest.approx(3.3e-6, rel=1e-12)
    assert js["status"]["code"] == "FAIL_CONSTRAINT"


# ---- experiment 3: synthetic DPT cell ---------------------------------------------------------


@pytest.fixture(scope="module")
def dpt_nominal():
    return run("dpt_cell", "nominal")


def test_dpt_cell_checks_ledger_and_convergence(dpt_nominal):
    js = dpt_nominal
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    led = check(js, "에너지 잔차")
    assert led["status"] == "PASS" and abs(led["value"]) < 1e-6
    assert check(js, "허용오차 강화")["status"] == "PASS"
    assert check(js, "gate 전하 보존")["status"] == "PASS"
    assert check(js, "표본 적분 vs solver")["status"] == "PASS"


def test_coss_energy_moves_between_turn_off_and_turn_on(dpt_nominal):
    js = dpt_nominal
    # E_oss(800 V) of the synthetic device by numerical quadrature here (independent of the lab closed form)
    d = DeviceParams()
    Cgd0 = 20e-9 / (2 * 2.0 * (math.sqrt(1 + 800 / 2.0) - 1))
    Eoss_num = quad(lambda v: v * (d.Cds0 / math.sqrt(1 + v / d.Vds0) + Cgd0 / math.sqrt(1 + v / 2.0)), 0, 800, epsrel=1e-12, limit=200)[0]
    assert metric(js, "Eoss") == pytest.approx(Eoss_num, rel=1e-9)
    # turn-off terminal energy exceeds channel dissipation by about the stored C_oss energy,
    # turn-on terminal energy falls short by the same amount; the sums agree
    Eoff, Eoff_ch = metric(js, "Eoff_fix"), metric(js, "Eoff_ch")
    Eon, Eon_ch = metric(js, "Eon_fix"), metric(js, "Eon_ch")
    assert Eoff - Eoff_ch == pytest.approx(Eoss_num, rel=0.05)
    assert Eon_ch - Eon == pytest.approx(Eoss_num, rel=0.05)
    assert metric(js, "E_sum_term") == pytest.approx(metric(js, "E_sum_ch"), rel=1e-4)


def test_overshoot_and_ringing_come_from_the_circuit_L_and_C(dpt_nominal):
    js = dpt_nominal
    ov = next(m for m in js["metrics"] if m["key"] == "dv_over")
    assert ov["value"] == pytest.approx(ov["ref"], rel=0.05)  # peak close to L_loop * |di/dt|max here
    f_nom = metric(js, "f_ring")
    hi = run("dpt_cell", "high_L")
    assert metric(hi, "vpk") > metric(js, "vpk")
    assert metric(hi, "f_ring") < 0.75 * f_nom  # more loop inductance -> lower L-C_oss ringing frequency
    fr = next(m for m in js["metrics"] if m["key"] == "f_ring")
    assert fr["check"] == "PASS"


def test_power_source_return_slows_turn_on_and_raises_Eon(dpt_nominal):
    ps = run("dpt_cell", "no_kelvin")
    assert metric(ps, "didt_on") < metric(dpt_nominal, "didt_on")
    assert metric(ps, "Eon_fix") > metric(dpt_nominal, "Eon_fix")


def test_measurement_definitions_change_E_without_changing_the_device(dpt_nominal):
    sk = run("dpt_cell", "skew5")
    assert metric(sk, "Eon_fix") == pytest.approx(metric(dpt_nominal, "Eon_fix"), rel=1e-9)  # same device waveform
    assert metric(sk, "Eon_user") > 1.1 * metric(sk, "Eon_fix")  # current shown 5 ns early -> more overlap
    ind = run("dpt_cell", "inductor_ref")
    assert metric(ind, "Eoff_user") > 5 * metric(ind, "Eoff_fix")  # I_L is not the device current after commutation
    assert ind["extra"]["dpt_settings"]["i_ref"] == "inductor"
    tab = next(t for t in dpt_nominal["tables"] if t["key"] == "t_settings")
    assert any("Coss" in r[0] for r in tab["rows"]) and any("deskew" in r[0] for r in tab["rows"])


# ---- experiment 4: blind question ---------------------------------------------------------------


def test_vgs_spike_artifact_vs_real_turn_on():
    art = run("vgs_spike", "artifact")
    assert art["status"]["code"] == "UNRESOLVED_ROOT_CAUSE"
    assert metric(art, "maxPS") > metric(art, "Vth") > metric(art, "maxdie")
    assert metric(art, "Qst_rel") < 1e-3
    real = run("vgs_spike", "real")
    assert real["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(real, "maxdie") > metric(real, "Vth")
    assert metric(real, "Qst_rel") > 0.05
    # the Kelvin pin under-reads the die during Miller injection (R_g,int drop)
    assert metric(real, "maxK") < metric(real, "maxdie")
    cl = run("vgs_spike", "clamp")
    assert cl["status"]["code"] == "MARGINAL"
    assert metric(cl, "Qst") < 0.1 * metric(real, "Qst")
    # the reference artifact follows L_sH, not the gate hold
    tab = next(t for t in art["tables"] if t["key"] == "t_disc")
    base, lay = tab["rows"][0], tab["rows"][3]
    assert lay[1] < 0.1 * base[1]


# ---- the cell itself ---------------------------------------------------------------------------


def test_cell_topologies_are_consistent_and_close_the_ledger():
    d = DeviceParams()
    out = {}
    for topo in ("individual_kelvin", "common_kelvin_star", "common_source_node", "power_ground"):
        spec = CellSpec(dut=[d], hs=d, Ld=[[0.0]], Ls=[2e-9], Lcs=0.0, topology=topo, Rg_on=[10.0], Rg_off=[10.0])
        r = Cell(spec, [(20e-9, "off"), (0.5e-6, "on")]).simulate(0.75e-6, rtol=1e-7)
        assert abs(r.energy["normalised"]) < 1e-6, topo
        out[topo] = r.energy["E_ch0"]
    # N = 1: a Kelvin star with R_e = 0 is the individual Kelvin driver; power ground with L_cs = 0 is the source node
    assert out["common_kelvin_star"] == pytest.approx(out["individual_kelvin"], rel=1e-5)
    assert out["power_ground"] == pytest.approx(out["common_source_node"], rel=1e-5)
    assert out["common_source_node"] > out["individual_kelvin"]


def test_fast_scalar_rhs_equals_vectorised_rhs():
    rng = np.random.default_rng(7)
    d = DeviceParams().scaled(3)
    devs = [d, DeviceParams(Vth=4.3).scaled(3)]
    spec = CellSpec(dut=devs, hs=d.scaled(2), Ld=np.array([[4e-9, 1e-9], [1e-9, 4e-9]]), Ls=[1e-9, 1.2e-9], Lcs=2e-9, topology="common_kelvin_star", Rcom=0.5, Re=[0.5, 0.5], IL=200.0)
    c = Cell(spec, [(10e-9, "off"), (0.3e-6, "on")])
    c._set_segment(0.3e-6, 0.3e-6 + 2e-9)
    x0 = np.concatenate([c.dc_state(), np.zeros(c.n_q)])
    for _ in range(50):
        x = x0.copy()
        x[: c.n_core] += rng.normal(size=c.n_core) * np.maximum(np.abs(x0[: c.n_core]), 1.0) * 0.2
        np.testing.assert_allclose(c.rhs(0.3e-6 + 1e-9, x), c.rhs_vec(0.3e-6 + 1e-9, x), rtol=1e-9, atol=1e-6)


def test_device_closed_forms_match_quadrature():
    D = DevArrays([DeviceParams()])
    for v in (-5.0, 0.0, 3.0, 400.0, 800.0):
        e_ds = quad(lambda u: u * float(D.Cds(np.array([u]))[0]), 0, v, limit=200, epsabs=0, epsrel=1e-13)[0]
        q_ds = quad(lambda u: float(D.Cds(np.array([u]))[0]), 0, v, limit=200, epsabs=0, epsrel=1e-13)[0]
        e_gd = quad(lambda u: u * float(D.Cgd(np.array([u]))[0]), 0, v, limit=200, epsabs=0, epsrel=1e-13)[0]
        assert float(D.E_ds(np.array([v]))[0]) == pytest.approx(e_ds, rel=1e-9, abs=1e-18)
        assert float(D.Q_ds(np.array([v]))[0]) == pytest.approx(q_ds, rel=1e-9, abs=1e-18)
        assert float(D.E_gd(np.array([v]))[0]) == pytest.approx(e_gd, rel=1e-9, abs=1e-18)


def test_inputs_are_rejected_not_clamped():
    e = get_lab("FL02").experiment("dpt_cell")
    with pytest.raises(ParamError) as exc:
        resolve_params(e.params, e.presets, "nominal", {"win_post": "400 ns", "IL": "-5"})
    assert "win_post" in exc.value.errors and "IL" in exc.value.errors
    e1 = get_lab("FL02").experiment("loss_ab")
    with pytest.raises(ParamError):
        resolve_params(e1.params, e1.presets, "textbook", {"RA": "40 uF"})


def test_all_reference_presets_checks_pass():
    lab = get_lab("FL02")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
