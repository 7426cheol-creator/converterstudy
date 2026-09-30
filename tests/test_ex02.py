"""EX02 nonlinear Coss, dead time and ZVS.

Independent paths: textbook literals; dense trapezoid integration in the test itself;
closed-form charge inversion vs ODE; voltage-state DOP853 vs charge-state RK4; energy
ledger; constant-C analytic LC solution.
"""

import math

import numpy as np
import pytest

from convlab.labs import get_lab
from convlab.labs.ex02_coss_zvs import Node, classify, rk4_charge_path, simulate_transition
from convlab.model.params import resolve_params
from convlab.reference import coss as ref

C0, V0, VB = 2e-9, 40.0, 800.0


def run(exp, preset=None, **over):
    e = get_lab("EX02").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_q_and_e_textbook_and_dense_integration():
    v = np.linspace(0, VB, 400001)
    c = C0 / np.sqrt(1 + v / V0)
    assert np.trapezoid(c, v) == pytest.approx(573.212e-9, rel=2e-6)
    assert np.trapezoid(v * c, v) == pytest.approx(180.238e-6, rel=5e-6)  # textbook value rounded to 6 digits
    assert ref.Q(VB, C0, V0) == pytest.approx(573.212e-9, rel=1e-6)
    assert ref.E(VB, C0, V0) == pytest.approx(180.238e-6, rel=3e-6)


def test_transition_times_textbook():
    js = run("hb_constant_current", "textbook")
    assert metric(js, "t_trans") == pytest.approx(286.606e-9, rel=1e-6)
    assert metric(js, "t_pt") == pytest.approx(174.574e-9, rel=1e-6)


@pytest.mark.parametrize("td,complete", [(100e-9, False), (200e-9, False), (300e-9, True), (500e-9, True)])
def test_deadtime_partial_vs_complete(td, complete):
    js = run("hb_constant_current", "textbook", td=f"{td * 1e9:g} ns")
    vds = metric(js, "vds_on")
    if complete:
        assert vds == 0.0
    else:
        assert vds > 0.05 * VB


def test_wrong_current_sign_gives_hard_switching():
    js = run("hb_constant_current", "neg")
    assert metric(js, "vds_on") == VB
    assert js["status"]["code"] == "FAIL_CONSTRAINT"


def test_hard_turn_on_loss_is_VQ_not_2E_for_nonlinear_C():
    n = Node(VB, C0, V0)
    assert n.hard_on_loss(0.0) == pytest.approx(VB * ref.Q(VB, C0, V0), rel=1e-12)
    assert n.hard_on_loss(VB) == pytest.approx(0.0, abs=1e-15)
    # linear limit: V*Q = C V^2 = 2 * (1/2 C V^2)
    f = Node(VB, C0, V0, model="fixed", Cfix=500e-12)
    assert f.hard_on_loss(0.0) == pytest.approx(500e-12 * VB**2, rel=1e-12)
    assert VB * ref.Q(VB, C0, V0) / (2 * ref.E(VB, C0, V0)) == pytest.approx(1.2721, rel=1e-4)


def test_resonant_midpoint_needs_no_net_energy_but_needs_time():
    n = Node(VB, C0, V0)
    tr = simulate_transition(n, 20e-6, VB / 2, 4.0, 400e-9)
    assert tr.t_reach is not None
    # symmetric midpoint: inductor current at arrival equals the initial current (energy neutral)
    k = next(j for j, t in enumerate(tr.t) if t >= tr.t_reach)
    assert tr.i[k] == pytest.approx(4.0, rel=1e-6)
    assert abs(tr.ledger["normalised"]) < 1e-9
    short = simulate_transition(n, 20e-6, VB / 2, 4.0, 200e-9)
    assert short.v_on < VB * 0.95  # partial: time, not energy, is the limit


def test_resonant_buck_type_falls_back_for_lack_of_energy():
    n = Node(VB, C0, V0)
    tr = simulate_transition(n, 20e-6, 0.0, 4.0, 500e-9)
    assert tr.t_reach is None and tr.v_max < VB
    # energy stored in L cannot lift the node: 1/2 L I^2 = 160 uJ < Vb*Qoss = 459 uJ
    assert 0.5 * 20e-6 * 16 < VB * ref.Q(VB, C0, V0)
    cls, _ = classify(VB, tr.v_on, tr.returned, 4.0, tr.v_max, False)
    assert cls == "FELL_BACK"
    assert abs(tr.ledger["normalised"]) < 1e-9


def test_independent_rk4_charge_path_agrees():
    n = Node(VB, C0, V0)
    tr = simulate_transition(n, 20e-6, VB / 2, 4.0, 200e-9)
    vr, ir = rk4_charge_path(n, 20e-6, VB / 2, 4.0, 200e-9, 800)
    assert vr == pytest.approx(tr.v_on, abs=1e-6)
    assert ir == pytest.approx(tr.i_on, abs=1e-9)
    tr2 = simulate_transition(n, 20e-6, VB / 2, 4.0, 300e-9, Vf=3.5)
    vr2, ir2 = rk4_charge_path(n, 20e-6, VB / 2, 4.0, 300e-9, 1600, Vf=3.5)
    assert vr2 == VB and ir2 == pytest.approx(tr2.i_on, abs=1e-5)


def test_constant_c_limit_is_analytic_lc():
    Cf = 700e-12
    n = Node(VB, C0, V0, model="fixed", Cfix=Cf)
    L, vx, I0, t = 20e-6, 400.0, 4.0, 60e-9
    tr = simulate_transition(n, L, vx, I0, t)
    w = 1 / math.sqrt(L * 2 * Cf)
    v_an = vx - vx * math.cos(w * t) + I0 / (2 * Cf * w) * math.sin(w * t)
    assert tr.v_on == pytest.approx(v_an, rel=1e-8)


def test_long_deadtime_loses_zvs():
    js = run("resonant_transition", "long")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "clamp") > 100e-9


def test_all_presets_checks_pass():
    lab = get_lab("EX02")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
