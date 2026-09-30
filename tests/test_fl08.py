"""FL08 DAB: textbook numbers, three independent paths, offset physics, interleaving."""

import math

import numpy as np
import pytest

from convlab.engine.switched import simulate
from convlab.labs import get_lab
from convlab.labs._dab import PI, SPS_TEXTBOOK_C1, DABSystem, Modulation, half_wave_periodic, pwl_waves
from convlab.model.params import resolve_params
from convlab.reference import dab as ref


def run(exp, preset=None, **over):
    e = get_lab("FL08").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {k: str(v) for k, v in over.items()})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def test_textbook_nominal_numbers():
    js = run("sps_nominal", "nominal")
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert metric(js, "phi") == pytest.approx(0.328972794, rel=1e-8)
    assert metric(js, "Irms") == pytest.approx(2.019881509, rel=1e-9)
    assert metric(js, "Ipk") == pytest.approx(2.09430585, rel=1e-9)
    assert metric(js, "Is_rms") == pytest.approx(33.6647, rel=5e-6)
    assert metric(js, "Io_dc") == pytest.approx(31.25, rel=1e-12)
    assert metric(js, "Io_total") == pytest.approx(62.5, rel=1e-12)
    assert metric(js, "Pmax") == pytest.approx(4000.0, rel=1e-12)
    assert metric(js, "d") == pytest.approx(2 * metric(js, "dt_T"))
    codes = {v["code"] for v in js["verdicts"]}
    assert "NOT_EVALUABLE" in codes  # ideal switches cannot judge ZVS


def test_low_corner_pmax_textbook():
    assert ref.sps_pmax(550.0, 50 / 3 * 36, 200e-6, 100e3) == pytest.approx(2062.5, rel=1e-12)


def test_mismatch_zero_power_textbook():
    js = run("zero_power_mismatch", "mismatch")
    assert metric(js, "P") == pytest.approx(0.0, abs=1e-9)
    assert metric(js, "Irms_L") == pytest.approx(2.165063509, rel=1e-9)
    assert metric(js, "Ipk_L") == pytest.approx(3.75, rel=1e-12)
    assert metric(js, "Irms_L") > metric(js, "Irms_nom")


@pytest.mark.parametrize("V1,V2", [(800.0, 800.0), (900.0, 600.0), (550.0, 600.0)])
@pytest.mark.parametrize("phi", [0.1, 0.3, 0.6, 1.0, 1.4])
def test_power_three_paths_15_points(V1, V2, phi):
    L, fs = 200e-6, 100e3
    P_cf = ref.sps_power(V1, V2, L, fs, phi)
    w = pwl_waves(V1, V2, L, fs, Modulation(PI, PI, phi, SPS_TEXTBOOK_C1))
    sysd = DABSystem(V1, V2, L, fs, mod_of_cycle=lambda k: Modulation(PI, PI, phi, SPS_TEXTBOOK_C1))
    q0 = Modulation(PI, PI, phi, SPS_TEXTBOOK_C1).levels(1e-9)
    x0 = half_wave_periodic(sysd, q0)
    tr = simulate(sysd, q0, x0, 0.0, 1 / fs)
    P_eng = tr.energy(0, 1 / fs, "p2") * fs
    scale = V1 * V2 / (8 * fs * L)
    assert w.P2 == pytest.approx(P_cf, abs=1e-9 * scale)
    assert P_eng == pytest.approx(P_cf, abs=1e-9 * scale)
    _, _, irms, _ = ref.sps_currents(V1, V2, L, fs, phi)
    assert w.Irms == pytest.approx(irms, rel=1e-9)


def test_reverse_power_same_current_magnitude():
    a = run("sps_nominal", "nominal")
    b = run("sps_nominal", "reverse")
    assert metric(b, "phi") == pytest.approx(-metric(a, "phi"))
    assert metric(b, "Irms") == pytest.approx(metric(a, "Irms"), rel=1e-12)
    assert metric(b, "P_pwl") == pytest.approx(-1500.0, rel=1e-9)


def test_over_pmax_is_no_solution():
    js = run("sps_nominal", "over")
    assert js["status"]["code"] == "NO_SOLUTION"


def test_all_reference_checks_pass():
    lab = get_lab("FL08")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c["name"], c["value"])
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m["key"])


def test_offset_persists_without_loss_and_decays_with_R():
    lossless = run("offset_startup", "lossless")
    assert metric(lossless, "offset_end") == pytest.approx(metric(lossless, "offset0"), rel=1e-9)
    assert metric(lossless, "offset0") == pytest.approx(2.09430585, rel=1e-8)  # = -i(0)
    damped = run("offset_startup", "damped")
    assert abs(metric(damped, "offset_end")) < 0.02 * abs(metric(damped, "offset0"))


def test_asymmetry_current_bounded_by_R():
    js = run("offset_startup", "asym")
    # 200 cycles = 5 tau -> within ~1 % of the dVs*fs/R bound
    assert metric(js, "offset_end") == pytest.approx(metric(js, "Idc_bound"), rel=0.02)


def test_interleaving_180_does_not_cancel_90_halves():
    js = run("two_modules", "ideal")
    r0, r90, r180 = metric(js, "ripple_0"), metric(js, "ripple_90"), metric(js, "ripple_180")
    assert r180 == pytest.approx(r0, rel=1e-9)
    assert r90 == pytest.approx(r0 / 2, rel=1e-3)


def test_sharing_ratio_follows_inverse_L():
    js = run("two_modules", "textbook")
    assert metric(js, "I1") / metric(js, "I2") == pytest.approx(1.1 / 0.9, rel=1e-9)
    assert metric(js, "share") == pytest.approx(0.2, rel=1e-9)
