"""Engine verification against analytic circuits (independent closed forms)."""

import math

import numpy as np
import pytest

from convlab.engine.pwl import PWL
from convlab.engine.switched import AffineMode, Guard, HybridSystem, segment_moments, simulate


class RC(HybridSystem):
    state_names = ("v",)

    def __init__(self, V=10.0, R=2.0, C=1e-3, guard_at=None):
        self.V, self.R, self.C, self.guard_at = V, R, C, guard_at

    def mode(self, q):
        tau = self.R * self.C
        return AffineMode(f"rc{q}", [[-1 / tau]], [self.V / tau if q == "charge" else 0.0])

    def guards(self, q):
        if self.guard_at is None or q != "charge":
            return []
        return [Guard("half", np.array([1.0, -self.guard_at]), +1, lambda q: "hold")]

    def outputs(self, q):
        return {"v": np.array([1.0, 0.0]), "i": np.array([-1 / self.R, self.V / self.R if q == "charge" else 0.0])}

    def powers(self, q):
        # resistor loss = (V - v)^2 / R
        c = np.array([-1.0, self.V if q == "charge" else 0.0])
        return {"R": np.outer(c, c) / self.R}


def test_rc_charge_exact():
    s = RC()
    tr = simulate(s, "charge", [0.0], 0.0, 0.01)
    tau = s.R * s.C
    v_end = tr.z_end[0]
    assert v_end == pytest.approx(10 * (1 - math.exp(-0.01 / tau)), rel=1e-13)
    # integral of v and of resistor power, closed form
    T = 0.01
    iv = 10 * (T - tau * (1 - math.exp(-T / tau)))
    assert tr.integral_output(0, T, "v") == pytest.approx(iv, rel=1e-12)
    E_R = 100 / s.R * tau / 2 * (1 - math.exp(-2 * T / tau))
    assert tr.energy(0, T, "R") == pytest.approx(E_R, rel=1e-12)
    # energy balance: source energy = resistor loss + stored energy
    E_src = 10 * tr.integral_output(0, T, "i")
    assert E_src == pytest.approx(E_R + 0.5 * s.C * v_end**2, rel=1e-12)


def test_guard_located_on_exact_solution():
    s = RC(guard_at=5.0)
    tr = simulate(s, "charge", [0.0], 0.0, 0.01)
    t_half = [seg.t1 for seg in tr.segments if seg.reason.startswith("guard")][0]
    assert t_half == pytest.approx(s.R * s.C * math.log(2), rel=1e-12)
    assert tr.segments[-1].q == "hold"


class LC(HybridSystem):
    state_names = ("i", "v")

    def __init__(self, L=1e-6, C=1e-6):
        self.L, self.C = L, C

    def mode(self, q):
        return AffineMode("lc", [[0, -1 / self.L], [1 / self.C, 0]], [0, 0])

    def outputs(self, q):
        return {"i": np.array([1.0, 0, 0]), "v": np.array([0, 1.0, 0])}

    def stored_energy(self):
        return np.diag([self.L, self.C, 0.0])


def test_lossless_lc_energy_and_rms():
    s = LC()
    w = 1 / math.sqrt(s.L * s.C)
    T = 2 * math.pi / w
    tr = simulate(s, None, [0.0, 1.0], 0.0, 10 * T)
    z = tr.z_end
    W = 0.5 * z @ s.stored_energy() @ z
    assert W == pytest.approx(0.5 * s.C, rel=1e-11)
    # over whole periods the RMS of v is 1/sqrt(2) and mean is 0
    assert tr.rms(0, 10 * T, "v") == pytest.approx(1 / math.sqrt(2), rel=1e-11)
    assert abs(tr.mean(0, 10 * T, "v")) < 1e-11
    lo, hi = tr.extrema(0, T, "i")
    assert hi == pytest.approx(math.sqrt(s.C / s.L), rel=1e-9)


def test_moments_vs_gauss_legendre():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)) - 2 * np.eye(3)
    b = rng.normal(size=3)
    mode = AffineMode("rand", A, b)
    z0 = np.append(rng.normal(size=3), 1.0)
    h = 0.7
    S2 = segment_moments(mode, z0, h)
    from scipy.linalg import expm

    xg, wg = np.polynomial.legendre.leggauss(40)
    ts = 0.5 * h * (xg + 1)
    ref = sum(0.5 * h * w * np.outer(expm(mode.F * t) @ z0, expm(mode.F * t) @ z0) for t, w in zip(ts, wg))
    assert np.max(np.abs(S2 - ref)) < 1e-12 * max(1, np.max(np.abs(ref)))


def test_stiff_moments_are_finite():
    # 1 ns time constant over 10 us: exp(+1e4) must never appear
    mode = AffineMode("stiff", [[-1e9]], [1e9])
    z0 = np.array([0.0, 1.0])
    S2 = segment_moments(mode, z0, 1e-5)
    assert np.all(np.isfinite(S2))
    assert S2[0, 1] == pytest.approx(1e-5 - 1e-9, rel=1e-9)  # integral of (1 - e^{-t/tau})


def test_pwl_exact_integrals():
    # triangle 0 -> 1 -> 0 over [0, 2]: mean 0.5, rms 1/sqrt(3)
    w = PWL([0, 1, 2], [0, 1], [1, 0])
    assert w.mean() == pytest.approx(0.5)
    assert w.rms() == pytest.approx(1 / math.sqrt(3))
    sq = PWL.step([0, 1, 2], [1, -1])
    assert w.product_integral(sq) == pytest.approx(0.5 - 0.5)
    ramp = PWL([0, 2], [0], [2])
    assert ramp.product_integral(ramp) == pytest.approx(8 / 3)
    # dense check of a jumpy product
    a = PWL([0, 0.3, 1.0], [1, -2], [2, 0.5])
    b = PWL([0, 0.6, 1.0], [0.2, 1], [-1, 3])
    xs = np.linspace(0, 1, 2_000_001)
    fa = np.array([a.value_at(x) for x in xs[::1000]])
    assert fa.size > 0
    dense = np.trapezoid([a.value_at(x) * b.value_at(x) for x in xs[::10]], xs[::10])
    assert a.product_integral(b) == pytest.approx(dense, abs=2e-6)
