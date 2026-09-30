"""EX04 transformer design closure: turns/B/window, CLLC integer turns, harmonic copper, tolerance, identification.

Independence map:
  * textbook/instruction numbers typed literally here (168.329 / 168.3235 mm2, 2.60 / 2.26 W,
    142.857 / 157.895 / 150.188 kHz, FL10 164.390 / 162.811 / 136.099 / 147.061 kHz, 14.390 / 14.765 A,
    seed gain 1.016401, 46.2481 uH / 24.3424 nF)
  * window: the E04 formula written here with the rounded and the exact FL08 currents
  * SPS current: the FL08 four-interval end points written here (general, mismatched) vs the lab's PWL;
    the lab also checks the exact switched engine (preset checks)
  * CLLC: the textbook FHA gain written here with complex numbers (not the lab's vectorized helper)
  * Fourier coefficients: scipy quad on the waveform written here vs the lab's closed-form segment integrals
  * identification: L/(1 - (f/f_SRF)^2) and the strap referral n^2 L_sh written here (approximations)
  * preset/metric plumbing: regression
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from convlab.labs import get_lab
from convlab.labs.ex04_transformer import dab_point, pwl_fourier
from convlab.model.params import ParamError, resolve_params


def run(exp_key, preset=None, **over):
    exp = get_lab("EX04").experiment(exp_key)
    vals, _, _ = resolve_params(exp.params, exp.presets, preset, {k: str(v) for k, v in over.items()})
    return exp.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def codes(js):
    return [v["code"] for v in js["verdicts"]]


def sps_rms_here(V1, V2, L, fs, P):
    """FL08 SPS written independently: phase from P = V1 V2 phi (1 - phi/pi)/(w L), RMS from the interval end points."""
    w = 2 * math.pi * fs
    phi = (math.pi - math.sqrt(math.pi**2 - 4 * math.pi * P * w * L / (V1 * V2))) / 2
    a, b = (V1 + V2) / (w * L), (V1 - V2) / (w * L)
    i0 = -(a * phi + b * (math.pi - phi)) / 2
    i1 = i0 + a * phi
    i2 = -i0
    ms = (phi * (i0 * i0 + i0 * i1 + i1 * i1) + (math.pi - phi) * (i1 * i1 + i1 * i2 + i2 * i2)) / 3 / math.pi
    return math.sqrt(ms), phi


def fha_here(f, n, Vo, P, Lr, Cr, Lm, Lr2, Cr2):
    w = 2 * math.pi * f
    Rac = 8 * n * n * Vo * Vo / P / math.pi**2
    Zr1 = complex(0, w * Lr - 1 / (w * Cr))
    Zs = complex(Rac, w * Lr2 - 1 / (w * Cr2))
    Zm = complex(0, w * Lm)
    Zp = Zm * Zs / (Zm + Zs)
    return Zp / (Zr1 + Zp) * Rac / Zs, Zr1 + Zp


def test_window_textbook_rounded_and_exact():
    js = run("turns_window", "textbook")
    assert (50 * 2.02 / 4 + 3 * 33.665 / 4) / 0.30 == pytest.approx(168.329, abs=5e-4)
    assert (50 * 2.019881509 / 4 + 3 * 33.66469 / 4) / 0.30 == pytest.approx(168.3235, abs=5e-5)
    assert metric(js, "Aw_tb") == pytest.approx(168.329, abs=5e-4)  # textbook: rounded currents
    assert metric(js, "Aw_exact") == pytest.approx(168.3235, abs=5e-5)  # erratum: exact FL08 currents
    assert metric(js, "Ip") == pytest.approx(2.019881509, abs=1e-9)
    assert metric(js, "Is") == pytest.approx(33.66469, abs=1e-5)
    assert metric(js, "B_ref") == pytest.approx(900 / (4 * 100e3 * 50 * 250e-6), rel=1e-12)  # 0.18 T at the 900 V corner
    assert metric(js, "B_nom") == pytest.approx(0.16, rel=1e-12)
    assert metric(js, "n_pass") == 1  # only 50:3 passes B, window and the n tolerance with the synthetic core
    # headline rule: no outcome verdict here (a screen is not a pass), so the worst scope note leads
    assert js["status"]["code"] == "MISSING_INPUT"
    assert {"SCREEN_ONLY", "MISSING_INPUT"} == set(codes(js))


def test_insulation_and_current_basis_change_the_screen():
    assert run("turns_window", "insulation")["status"]["code"] == "FAIL_CONSTRAINT"  # 150 mm2 < 168.3 mm2
    js = run("turns_window", "worst_current")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    assert metric(js, "Aw_ref") > 200.0  # 900/36 V corner current (circulating) needs a bigger window
    assert metric(run("turns_window", "big_window"), "n_pass") == 3  # 50:3, 66:4, 67:4


def test_integer_candidates_recompute_the_operating_current():
    for Np, Ns in ((49, 3), (50, 3), (51, 3), (67, 4)):
        n = Np / Ns
        irms, phi = sps_rms_here(800.0, n * 48.0, 200e-6, 100e3, 1500.0)
        ph, pw = dab_point(800.0, n * 48.0, 200e-6, 100e3, 1500.0)
        assert ph == pytest.approx(phi, rel=1e-13)
        assert pw.rms() == pytest.approx(irms, rel=1e-12)
    # a one-turn change moves n by 2 % and the nominal RMS by about 1 %
    r49 = sps_rms_here(800.0, 49 / 3 * 48.0, 200e-6, 100e3, 1500.0)[0]
    assert r49 / 2.019881509 - 1 == pytest.approx(0.0136, abs=5e-4)
    js = run("turns_window", "textbook")
    assert metric(js, "Is_Lm") > metric(js, "Is")  # the magnetizing current adds to the secondary winding (k = 1)


def test_cllc_textbook_fha_numbers():
    js = run("cllc_turns", "nominal")
    assert metric(js, "f650") == pytest.approx(164.390e3, abs=1.0)
    assert metric(js, "f800") == pytest.approx(162.811e3, abs=1.0)
    assert metric(js, "f920_lo") == pytest.approx(136.099e3, abs=1.0)
    assert metric(js, "f920_hi") == pytest.approx(147.061e3, abs=1.0)
    assert metric(js, "I920_lo") == pytest.approx(14.390, abs=1e-3)
    assert metric(js, "I920_hi") == pytest.approx(14.765, abs=1e-3)
    assert metric(js, "slope_lo") == pytest.approx(0.001871, abs=1e-6)
    assert metric(js, "slope_hi") == pytest.approx(-0.001803, abs=1e-6)
    assert metric(js, "seed_gain") == pytest.approx(1.016401, abs=2e-6)  # < 920/850 = 1.082353: seed FAIL kept
    assert metric(js, "Lr2") == pytest.approx(46.2481e-6, abs=1e-10)
    assert metric(js, "Cr2") == pytest.approx(24.3424e-9, abs=1e-13)
    # the roots satisfy the textbook FHA gain written here
    tank = (40e-6, 28.144773e-9, 200e-6, 40e-6, 28.144773e-9)
    for key, Vo, Vi in (("f650", 650, 700), ("f800", 800, 800), ("f920_lo", 920, 850), ("f920_hi", 920, 850)):
        H, Zin = fha_here(metric(js, key), 0.93, Vo, 11e3, *tank)
        assert abs(H) == pytest.approx(0.93 * Vo / Vi, rel=1e-9)
        assert Zin.imag > 0
    assert js["status"]["code"] == "CANDIDATE_FHA_ONLY"  # outcome; the scope notes stay beside it
    assert {"SCREEN_ONLY", "MISSING_INPUT"} <= set(codes(js))


def test_cllc_integer_turns_and_the_ratio_limit():
    js = run("cllc_turns", "nominal")
    assert 14 / 15 < metric(js, "n_max") < 15 / 16
    assert metric(js, "req920") == pytest.approx(13 / 14 * 920 / 850, rel=1e-12)
    # 15:16 at 920/850 V: the textbook gain written here never reaches the requirement in 120-210 kHz
    n = 15 / 16
    tank = (40e-6, 28.144773e-9, 200e-6, n * n * 40e-6 / 0.93**2, 28.144773e-9 * 0.93**2 / (n * n))
    gmax = max(abs(fha_here(f, n, 920, 11e3, *tank)[0]) for f in np.linspace(120e3, 210e3, 9001) if fha_here(f, n, 920, 11e3, *tank)[1].imag > 0)
    assert gmax < n * 920 / 850
    assert run("cllc_turns", "n_high")["status"]["code"] == "NO_SOLUTION"
    assert run("cllc_turns", "small_core")["status"]["code"] == "FAIL_CONSTRAINT"
    # keeping the physical secondary tank vs re-tuning it are different circuits (same n, other f_op)
    assert metric(run("cllc_turns", "retune"), "ref_f800") != pytest.approx(metric(js, "ref_f800"), abs=1.0)


def test_harmonic_copper_textbook():
    js = run("harmonic_copper", "textbook")
    assert 10**2 * 0.020 + 3**2 * 0.040 + 2**2 * 0.060 == pytest.approx(2.60, abs=1e-12)
    assert (10**2 + 3**2 + 2**2) * 0.020 == pytest.approx(2.26, abs=1e-12)
    assert metric(js, "P_harm") == pytest.approx(2.60, abs=1e-12)
    assert metric(js, "P_dc") == pytest.approx(2.26, abs=1e-12)
    assert metric(js, "dP") == pytest.approx(0.34, abs=1e-12)
    assert js["status"]["code"] == "PASS_WITHIN_MODEL"
    assert "MISSING_INPUT" in codes(js)  # core loss is not in the total


def test_harmonic_copper_dab_exact_fourier():
    _, pw = dab_point(800.0, 800.0, 200e-6, 100e3, 1500.0)
    c = pwl_fourier(pw, 7)
    T = 1e-5
    for h in (1, 3, 5, 7):
        re = quad(lambda t: pw.value_at(t) * math.cos(2 * math.pi * h * t / T), 0, T, points=list(pw.t[1:-1]), limit=200, epsabs=1e-13)[0] / T
        im = -quad(lambda t: pw.value_at(t) * math.sin(2 * math.pi * h * t / T), 0, T, points=list(pw.t[1:-1]), limit=200, epsabs=1e-13)[0] / T
        assert c[h] == pytest.approx(complex(re, im), abs=1e-10)
    assert abs(c[2]) < 1e-12 and abs(c[0]) < 1e-12  # half-wave symmetric, zero DC
    js = run("harmonic_copper", "dab")
    assert metric(js, "Irms") == pytest.approx(2.019881509, abs=1e-9)
    assert metric(js, "P_harm") > metric(js, "P_dc")
    assert metric(js, "P_dc") == pytest.approx(2.019881509**2 * 0.1 * (1 + 3.93e-3 * 80), rel=1e-9)
    # synthetic R_ac: only scope notes (SCREEN_ONLY, MISSING_INPUT), so the worst of them leads
    assert js["status"]["code"] == "MISSING_INPUT"
    assert {"SCREEN_ONLY", "MISSING_INPUT"} == set(codes(js))
    js = run("harmonic_copper", "dab_mismatch")
    assert metric(js, "Irms") == pytest.approx(3.113563, abs=1e-6)  # E06 SPS 900/600 V
    assert metric(run("harmonic_copper", "dab_thick"), "ratio") > 5 * metric(run("harmonic_copper", "dab"), "ratio")


def test_tolerance_corners_and_operating_map():
    js = run("tolerance_map", "textbook")
    f0 = 1 / (2 * math.pi * math.sqrt(40e-6 * 28.144773e-9))
    assert f0 == pytest.approx(150e3, rel=1e-6)
    assert metric(js, "fr_pp") == pytest.approx(142.857e3, abs=0.5)
    assert metric(js, "fr_mm") == pytest.approx(157.895e3, abs=0.5)
    assert metric(js, "fr_pm") == pytest.approx(150.188e3, abs=0.5)
    assert metric(js, "fr_mp") == pytest.approx(metric(js, "fr_pm"), rel=1e-12)
    assert metric(js, "fop_nom") == pytest.approx(162.811e3, abs=1.0)
    assert metric(js, "f920_lo") == pytest.approx(136.099e3, abs=1.0)
    assert metric(js, "f920_hi") == pytest.approx(147.061e3, abs=1.0)
    # same f_r, different characteristic impedance -> different operating frequency
    assert abs(metric(js, "fop_pm") - metric(js, "fop_mp")) > 1e3
    tank = (40e-6 * 1.05, 28.144773e-9 * 0.95, 200e-6, 40e-6 * 1.05, 28.144773e-9 * 0.95)
    assert abs(fha_here(metric(js, "fop_pm"), 0.93, 800, 11e3, *tank)[0]) == pytest.approx(0.93, rel=1e-9)
    assert metric(js, "p_nosol") == 0.0
    assert js["status"]["code"] == "CANDIDATE_FHA_ONLY"


def test_tolerance_correlation_monte_carlo():
    s = math.log(1.05) / 3
    js0 = run("tolerance_map", "textbook")
    js1 = run("tolerance_map", "correlated")
    jsm = run("tolerance_map", "anti")
    assert metric(js0, "sig_fr") == pytest.approx(0.5 * math.sqrt(2) * s, rel=0.07)
    assert metric(js1, "sig_fr") == pytest.approx(s, rel=0.07)
    assert metric(jsm, "sig_fr") < 1e-12  # L and C cancel exactly
    assert metric(jsm, "sig_fop") > 1e-3  # but Z0 and L_m still move the operating point
    assert metric(js1, "sig_fop") > metric(js0, "sig_fop") > metric(jsm, "sig_fop")
    assert run("tolerance_map", "wide")["status"]["code"] == "NO_SOLUTION"


def test_identification_measurement_conditions():
    js = run("identification", "nominal")
    assert abs(metric(js, "err")) < 2.0  # percent
    assert js["status"]["code"] == "PASS_WITHIN_MODEL" and "MISSING_INPUT" in codes(js)
    js = run("identification", "near_srf")
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    f_srf = 1 / (2 * math.pi * math.sqrt(2.002e-3 * (50e-12 + 100e-12 + 1e-9 / (50 / 3) ** 2)))
    assert metric(js, "f_srf") == pytest.approx(f_srf, rel=0.01)
    assert metric(js, "err") / 100 == pytest.approx(1 / (1 - (100e3 / f_srf) ** 2) - 1, rel=0.03)  # about +13.8 %
    # shorting strap on the 3-turn side seen from the 50-turn side: n^2 L_sh = 278 x 20 nH
    js = run("identification", "sc_lv_short")
    assert metric(js, "L_strap") == pytest.approx((50 / 3) ** 2 * 20e-9, rel=1e-12)
    L_exp = 2e-6 + 1 / (1 / 2e-3 + 1 / (2e-6 + (50 / 3) ** 2 * 20e-9))
    assert metric(js, "L_app") == pytest.approx(L_exp, rel=0.01)
    assert js["status"]["code"] == "FAIL_CONSTRAINT"
    js = run("identification", "sc_hv_short")
    assert abs(metric(js, "err")) < 2.0
    assert js["status"]["code"] == "PASS_WITHIN_MODEL" and "MISSING_INPUT" in codes(js)
    js = run("identification", "sc_low_f")
    assert metric(js, "Q") < 1 and js["status"]["code"] == "FAIL_CONSTRAINT"


def test_inputs_are_rejected_not_clamped():
    exp = get_lab("EX04").experiment("turns_window")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "textbook", {"ku": "0", "Np_ref": "50.5", "ins_frac": "95"})
    assert {"ku", "Np_ref", "ins_frac"} <= set(e.value.errors)
    exp = get_lab("EX04").experiment("tolerance_map")
    with pytest.raises(ParamError) as e:
        resolve_params(exp.params, exp.presets, "textbook", {"rho": "1.5", "tol_L": "-1"})
    assert {"rho", "tol_L"} <= set(e.value.errors)


def test_all_reference_presets_checks_pass():
    lab = get_lab("EX04")
    for e in lab.experiments:
        for pk in e.reference_presets:
            js = run(e.key, pk)
            for c in js["checks"]:
                assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (e.key, pk, c)
            for m in js["metrics"]:
                assert m["check"] in (None, "PASS", "INFO"), (e.key, pk, m)
