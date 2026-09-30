"""EX09 EMI source-path-victim.

Independence map (docs/TEST_INDEPENDENCE.md):
  * textbook numbers typed literally: 5 A, 50.329 MHz, 0.158965 mA
  * hand screens vs exact time-domain engine (ramp current, RLC zero crossings) and quadrature (independent)
  * trapezoid Fourier coefficients written here vs numpy FFT (independent)
  * CM path: impedance algebra (reference/emi.py) vs state-space from the simulated matrices,
    and Parseval (time-domain RMS vs harmonic sum) (independent)
  * DM path: current divider vs nodal admittance (independent); energy ledger (independent)
  * statuses (SCREEN_ONLY / NOT_EVALUABLE, never PASS for EMI compliance): regression
"""

import math

import numpy as np
import pytest

from convlab.engine.switched import simulate
from convlab.labs import get_lab
from convlab.labs.ex09_emi import HardSwitchCell, SeriesRLC, cell_steady, harmonics, zero_crossing_frequency
from convlab.model.params import ParamError, resolve_params
from convlab.reference import emi as ref


def run(exp_key, preset=None, **over):
    exp = get_lab("EX09").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


BASE = dict(V_bus=800.0, I_L=50.0, f_sw=100e3, D=0.5, t_fi=10e-9, t_ri=10e-9, I_dis=50.0, gate_scale=1.0, L_loop=10e-9, R_loop=0.1, C_node=1e-9,
            snubber=False, R_s=3.16, C_s=0.47e-9, C_par=100e-12, R_par=1.0, C_Y=0.0, C_stray=10e-12, L_cab=1e-6, R_m=50.0, L_m=50e-6)


def test_textbook_hand_screens():
    # 100 pF x 50 kV/us = 100e-12 F x 5e10 V/s = 5 A ; 10 nH / 1 nF -> 50.329 MHz ; 2.2 nF at 230 V 50 Hz -> 0.158965 mA
    assert 100e-12 * 50e3 / 1e-6 == pytest.approx(5.0)
    assert 1 / (2 * math.pi * math.sqrt(10e-9 * 1e-9)) == pytest.approx(50.329e6, rel=1e-5)
    assert 2 * math.pi * 50 * 2.2e-9 * 230 == pytest.approx(0.158965e-3, rel=1e-5)
    js = run("hand_screens", "textbook")
    assert js["status"]["code"] == "SCREEN_ONLY"
    assert metric(js, "i_pk") == pytest.approx(5.0, rel=1e-12)
    assert metric(js, "f_r") == pytest.approx(50.329e6, rel=1e-5)
    assert metric(js, "i_line") == pytest.approx(0.158965e-3, rel=1e-5)
    assert metric(js, "leak_status") == "NOT_EVALUABLE"


def test_unit_trap_is_three_orders_of_magnitude():
    js = run("hand_screens", "textbook")
    rows = next(t for t in js["tables"] if t["key"] == "t_units")["rows"]
    assert rows[0][2] == pytest.approx(5.0)
    assert rows[1][2] == pytest.approx(5e-3)  # kV/us read as V/us
    assert rows[3][2] == pytest.approx(5e3)  # pF read as nF
    assert rows[6][2] == pytest.approx(75.0)  # textbook ch.05: 15 nH x 5 kA/us


def test_ring_frequency_from_zero_crossings_of_the_exact_solution():
    s = SeriesRLC(1.0, 0.0, 10e-9, 1e-9)
    f0 = 1 / (2 * math.pi * math.sqrt(10e-12 * 1e-6))  # written out: 1/(2 pi sqrt(L C))
    tr = simulate(s, None, [0.0, 0.0], 0.0, 10 / f0)
    smp = tr.sample(["dv"], per_segment=20000)
    f, n = zero_crossing_frequency(np.asarray(smp["t"]), np.asarray(smp["dv"]), skip=1)
    assert n >= 8
    assert f == pytest.approx(f0, rel=1e-6)


def test_displacement_current_needs_a_low_impedance_path():
    js = run("hand_screens", "textbook")
    c = next(c for c in js["checks"] if c["name"].startswith("변위전류"))
    assert c["status"] == "PASS" and c["independent"]
    js2 = run("hand_screens", "meas_path")
    assert metric(js2, "i_end_path2") < 4.5  # a 50 ohm / 1 uH path has not reached C dv/dt by the end of the edge


def test_trapezoid_fft_matches_fourier_series_written_here():
    V, f, D, tr = 800.0, 100e3, 0.5, 16e-9
    N = 2**15
    t = np.arange(N) / (N * f)
    y = V * np.clip(np.minimum(t / tr, (D / f + tr - t) / tr), 0, 1)
    A = harmonics(y)
    for n in (1, 3, 5, 11, 101):
        An = 2 * V * D * abs(np.sinc(n * D)) * abs(np.sinc(n * f * tr))
        assert A[n] == pytest.approx(An, rel=2e-3)


def test_cell_ledger_and_periodicity():
    s = HardSwitchCell(BASE)
    tr, changes = cell_steady(s)
    assert changes[-1] < 1e-9
    from convlab.labs._common import energy_ledger

    ta = tr.segments[0].t0
    led = energy_ledger(tr, s, ta, ta + s.T, ["p_load"], ["p_bus"], ["p_loss"], 20e3)
    assert abs(led["normalised"]) < 1e-9


def test_source_path_victim_checks_and_status():
    js = run("source_path_victim", "nominal")
    assert js["status"]["code"] == "NOT_EVALUABLE"  # EMI compliance never PASS
    assert metric(js, "emi_status") == "NOT_EVALUABLE"
    names = {c["name"]: c for c in js["checks"]}
    assert all(c["status"] == "PASS" for c in js["checks"])
    assert any(n.startswith("Parseval") and c["independent"] for n, c in names.items())
    # load-current limited turn-off edge ~ I_L / C_node = 50 V/ns; CM peak near C_par dv/dt
    assert metric(js, "dvdt_up") == pytest.approx(50e9, rel=0.05)
    assert 4.0 < metric(js, "icp_pk") < 8.0
    # overshoot bounded by the undamped screen I_L sqrt(L/C) = 158 V
    assert 100 < metric(js, "v_os") < 50 * math.sqrt(10e-9 / 1e-9)


def test_light_load_slows_the_turn_off_edge_not_the_turn_on_edge():
    a = run("source_path_victim", "nominal")
    b = run("source_path_victim", "light")
    assert metric(b, "dvdt_up") == pytest.approx(metric(a, "dvdt_up") / 5, rel=0.05)
    assert metric(b, "dvdt_dn") == pytest.approx(metric(a, "dvdt_dn"), rel=0.02)
    assert metric(b, "cm_30_100") < metric(a, "cm_30_100") - 3.0


def test_cm_and_dm_transfer_formulations_agree():
    for f in (1e5, 1e6, 1e7, 5e7, 2e8):
        w = 2 * math.pi * f
        # CM with no Y capacitor: series C_par + R_par into (chassis stray || cable + meas/2) written here
        zc = 1.0 + 1 / (1j * w * 100e-12)
        zm = 1 / (1 / 50 + 1 / (1j * w * 50e-6))
        zext = 1j * w * 1e-6 / 2 + zm / 2
        zx = 1 / (1j * w * 10e-12)
        zret = zext * zx / (zext + zx)
        v_meas = (1 / (zc + zret)) * zret / zext / 2 * zm
        assert abs(ref.h_cm(f, 100e-12, 1.0, 1e-6, 50, 50e-6, 5e-12)) == pytest.approx(abs(v_meas), rel=1e-9)


def test_mitigation_tradeoffs():
    js = run("mitigation", "nominal")
    assert js["status"]["code"] == "NOT_EVALUABLE"
    # C_par halving: -6.02 dB where C_par dominates the path
    assert metric(js, "dcm_cpar") == pytest.approx(20 * math.log10(0.5), abs=0.1)
    # snubber cost ~ C_s V^2 f = 0.47 nF * 800^2 * 100 kHz = 30.08 W, split between R_s and the channel
    assert metric(js, "P_snub_total") == pytest.approx(0.47e-9 * 800**2 * 100e3, rel=0.1)
    assert metric(js, "os_snub") < -10.0
    # slowing the gate costs loss but barely helps the load-limited turn-off edge
    assert metric(js, "dP_gate") > 20.0
    assert abs(metric(js, "dcm_gate_30_100")) < 2.0
    assert metric(js, "dcm_ycap") < -10.0


def test_all_reference_presets_checks_pass_and_never_emi_pass():
    lab = get_lab("EX09")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            assert js["status"]["code"] != "PASS_WITHIN_MODEL"
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("EX09").experiment("hand_screens")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "textbook", {"dvdt": "50 A", "C_par": "-1 pF"})
    assert "dvdt" in e.value.errors and "C_par" in e.value.errors
    vals, _, _ = resolve_params(exp.params, exp.presets, "textbook", {"dvdt": "50 kV/us"})
    assert vals["dvdt"] == pytest.approx(5e4)  # stored in V/us


def test_edges_that_do_not_fit_are_rejected():
    # inside every parameter range, but the turn-off current ramp is longer than the on-time
    js = run("source_path_victim", "nominal", f_sw="1 MHz", D="0.05", t_fi="60 ns")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
