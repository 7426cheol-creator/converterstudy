"""Resonant-converter machinery shared by FL09 (LLC), FL10 (CLLC) and EX05 (CLLC dynamics).

Primary-referred T-network (textbook E05):
    driven bridge v1 -> R1, L1, C1 -> node m (shunt Lm) -> L2, C2, R2 (CLLC only) -> full-bridge
    diode rectifier -> output.
n = Np/Ns; referred quantities v' = n v, i' = i/n, L' = n^2 L, C' = C/n^2, R' = n^2 R.  For the
reverse direction the same network is used with the tanks swapped: the driven side is then the
battery side, so the input referral ratio ``n_in`` and the output referral ratio ``n_out`` say
how referred values map back to actual ones (actual current = n * referred current).

State x = [i1, i2, vC1, (vC2), (vo)]:
    i1  current from the driven bridge into the node (referred),
    i2  current from the node into the rectifier (referred), so im = i1 - i2 flows through Lm,
    vo  the ACTUAL output capacitor voltage (present only for an 'rc' output).
Rectifier state r: +1 (i2 > 0, v2 = +n_out vo), -1 (i2 < 0, v2 = -n_out vo) or 0 (all diodes off).
In the off state i2 = 0 is a constraint (its derivative row is zero) and the rectifier input node
floats at v2 = vm - vC2, which must stay inside the clamps; the guards are that floating voltage
reaching either clamp.  An LLC has no L2/C2: then vm = v2 + R2 i2 while a diode pair conducts.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..engine.periodic import shoot
from ..engine.switched import AffineMode, Guard, HybridSystem, simulate
from ..model.circuit import Circuit
from ._common import sym_linear

PI = math.pi
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------
# Tank and FHA (nodal analysis of the phasor network; the textbook closed forms live in
# reference/resonant.py and are only used for comparison)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tank:
    L1: float
    C1: float
    Lm: float
    L2: float = 0.0  # referred; 0 -> LLC (no secondary series element)
    C2: float | None = None  # referred; None -> no secondary series capacitor
    R1: float = 0.0
    R2: float = 0.0

    @property
    def is_llc(self) -> bool:
        return self.L2 == 0.0

    @property
    def fr1(self) -> float:
        return 1.0 / (TWO_PI * math.sqrt(self.L1 * self.C1))

    @property
    def Z0(self) -> float:
        return math.sqrt(self.L1 / self.C1)

    def swapped(self) -> "Tank":
        """The same referred network seen from the other port (reverse power flow)."""
        if self.is_llc or self.C2 is None:
            raise ValueError("an LLC tank has no secondary series branch to swap")
        return Tank(self.L2, self.C2, self.Lm, self.L1, self.C1, self.R2, self.R1)


def bridge_fundamental(Vin: float, bridge: str) -> float:
    """Peak of the fundamental of the bridge voltage: 4V/pi (full bridge), 2V/pi (half bridge)."""
    return (4.0 if bridge == "FB" else 2.0) * Vin / PI


def rac(n_out: float, Vo: float, P: float) -> float:
    """FHA equivalent of a full-bridge rectifier with a stiff output, referred: 8/pi^2 n^2 Vo^2/P."""
    return 8.0 / PI**2 * n_out**2 * Vo**2 / P


@dataclass
class FHAPoint:
    f: float
    H: complex  # vo_fund / v1_fund (referred)
    Zin: complex
    I1: complex  # peak phasors for a unit-amplitude... scaled by the caller
    I2: complex
    Vm: complex

    @property
    def gain(self) -> float:
        return abs(self.H)

    @property
    def inductive(self) -> bool:
        return self.Zin.imag > 0


def fha(tank: Tank, f: float, Rac: float, V1: complex = 1.0) -> FHAPoint:
    """Admittance form of the node equation: the shunt admittance at node m is Y_m + Y_b, so
    I1 = V1 / (Z1 + 1/(Y_m + Y_b)) and vm = V1 - Z1 I1 (no division by Z1, which is exactly zero at
    series resonance)."""
    w = TWO_PI * f
    Z1 = tank.R1 + 1j * w * tank.L1 + 1.0 / (1j * w * tank.C1)
    Z2 = tank.R2 + 1j * w * tank.L2 + (1.0 / (1j * w * tank.C2) if tank.C2 else 0.0)
    Ym = 1.0 / (1j * w * tank.Lm)
    Yb = 1.0 / (Z2 + Rac)
    I1 = V1 / (Z1 + 1.0 / (Ym + Yb))
    Vm = V1 - Z1 * I1
    I2 = Vm * Yb
    Vo = I2 * Rac
    return FHAPoint(f, Vo / V1, V1 / I1, I1, I2, Vm)


def gain_scan(tank: Tank, Rac: float, f_lo: float, f_hi: float, n: int = 1201):
    fs = np.linspace(f_lo, f_hi, n)
    pts = [fha(tank, float(f), Rac) for f in fs]
    return fs, np.array([p.gain for p in pts]), np.array([p.Zin.imag for p in pts])


def fha_roots(tank: Tank, Rac: float, g_req: float, f_lo: float, f_hi: float, n: int = 1801) -> list[dict]:
    """All frequencies in [f_lo, f_hi] where |H| = g_req, each with its input-impedance character."""
    fs, g, _ = gain_scan(tank, Rac, f_lo, f_hi, n)
    out = []
    for k in range(len(fs) - 1):
        a, b = g[k] - g_req, g[k + 1] - g_req
        if a == 0.0 or a * b < 0:
            fr = brentq(lambda f: fha(tank, f, Rac).gain - g_req, fs[k], fs[k + 1], xtol=1e-9, rtol=1e-15)
            p = fha(tank, fr, Rac)
            d = 10.0
            slope = (fha(tank, fr + d, Rac).gain - fha(tank, fr - d, Rac).gain) / (2 * d)
            out.append({"f": fr, "inductive": p.inductive, "slope_per_Hz": slope, "point": p})
    return out


def max_inductive_gain(tank: Tank, Rac: float, f_lo: float, f_hi: float, n: int = 9001) -> tuple[float, float]:
    """Largest |H| over the inductive part of [f_lo, f_hi] (refined around the best grid point)."""
    fs, g, xi = gain_scan(tank, Rac, f_lo, f_hi, n)
    mask = xi > 0
    if not mask.any():
        return float("nan"), float("nan")
    idx = int(np.argmax(np.where(mask, g, -np.inf)))
    a = fs[max(idx - 1, 0)]
    b = fs[min(idx + 1, len(fs) - 1)]
    # golden-section refine inside the bracket, staying in the inductive region
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        gc = fha(tank, c, Rac)
        gd = fha(tank, d, Rac)
        vc = gc.gain if gc.inductive else -1.0
        vd = gd.gain if gd.inductive else -1.0
        if vc > vd:
            b = d
        else:
            a = c
    fm = 0.5 * (a + b)
    pm = fha(tank, fm, Rac)
    if not pm.inductive:
        return float(g[idx]), float(fs[idx])
    return pm.gain, fm


# --------------------------------------------------------------------------------------
# Switched-affine model
# --------------------------------------------------------------------------------------

RECT_KO = {1: "정류 +", 0: "정류 off (i₂ = 0)", -1: "정류 −"}


class ResonantSystem(HybridSystem):
    """LLC/CLLC with an ideal driven bridge, full-bridge diode rectifier and a stiff or RC output.

    q = (b1, r): b1 the driven-bridge level (+1/-1 full bridge, 1/0 half bridge), r the rectifier state.
    Gate edges come from ``edges`` (list of (time, level)) when set, else from the constant fs.
    """

    def __init__(
        self,
        tank: Tank,
        Vin: float,
        fs: float,
        n_out: float,
        bridge: str = "FB",
        out: str = "stiff",
        Vo: float | None = None,
        Co: float | None = None,
        Rout: float | None = None,
        E: float = 0.0,
        n_in: float = 1.0,
        key: str = "",
    ):
        if tank.is_llc and tank.C2 is not None:
            raise ValueError("an LLC tank (L2 = 0) cannot have a secondary series capacitor")
        if out == "stiff" and Vo is None:
            raise ValueError("stiff output needs Vo")
        if out == "rc" and (Co is None or Rout is None):
            raise ValueError("rc output needs Co and Rout")
        self.tank, self.Vin, self.fs, self.n_out, self.bridge, self.out = tank, Vin, fs, n_out, bridge, out
        self.Vo, self.Co, self.Rout, self.E, self.n_in = Vo, Co, Rout, E, n_in
        self.T = 1.0 / fs
        self.edges: list[tuple[float, int]] | None = None
        names = ["i1", "i2", "vC1"]
        if tank.C2:
            names.append("vC2")
        if out == "rc":
            names.append("vo")
        self.state_names = tuple(names)
        self.ns = len(names)
        self.m = self.ns + 1
        self.ix = {k: i for i, k in enumerate(names)}
        self._key = f"res|{tank}|{Vin}|{n_out}|{bridge}|{out}|{Vo}|{Co}|{Rout}|{E}|{key}"
        self._modes: dict = {}
        self.i_scale = max(1e-9, Vin / max(tank.Z0, 1e-9))

    # ---- affine building blocks -------------------------------------------------------
    def e(self, name: str) -> np.ndarray:
        v = np.zeros(self.m)
        if name == "1":
            v[-1] = 1.0
        elif name in self.ix:
            v[self.ix[name]] = 1.0
        return v

    def levels(self) -> tuple[int, int]:
        return (1, -1) if self.bridge == "FB" else (1, 0)

    def v1_expr(self, b1: int) -> np.ndarray:
        return b1 * self.Vin * self.e("1")

    def vo_expr(self) -> np.ndarray:
        return self.e("vo") if self.out == "rc" else self.Vo * self.e("1")

    def clamp_expr(self) -> np.ndarray:
        return self.n_out * self.vo_expr()

    def _a(self, b1: int) -> np.ndarray:
        t = self.tank
        return self.v1_expr(b1) - self.e("vC1") - t.R1 * self.e("i1")

    def vm_expr(self, q) -> np.ndarray:
        b1, r = q
        t = self.tank
        a = self._a(b1)
        if r == 0:
            return a * (t.Lm / (t.L1 + t.Lm))
        if t.is_llc:
            return r * self.clamp_expr() + t.R2 * self.e("i2")
        c = (self.e("vC2") if t.C2 else 0.0) + r * self.clamp_expr() + t.R2 * self.e("i2")
        S = 1.0 / t.L1 + 1.0 / t.Lm + 1.0 / t.L2
        return (a / t.L1 + c / t.L2) / S

    def v2_float_expr(self, b1: int) -> np.ndarray:
        """Rectifier-input voltage if all diodes are off (valid with i2 = 0)."""
        vm = self.vm_expr((b1, 0))
        return vm - (self.e("vC2") if self.tank.C2 else 0.0)

    def v2_expr(self, q) -> np.ndarray:
        b1, r = q
        return r * self.clamp_expr() if r != 0 else self.v2_float_expr(b1)

    def mode(self, q) -> AffineMode:
        md = self._modes.get(q)
        if md is not None:
            return md
        b1, r = q
        t = self.tank
        a = self._a(b1)
        vm = self.vm_expr(q)
        rows = np.zeros((self.m, self.m))
        if r == 0:
            di1 = a / (t.L1 + t.Lm)
            di2 = np.zeros(self.m)
        elif t.is_llc:
            di1 = (a - vm) / t.L1
            di2 = di1 - vm / t.Lm
        else:
            c = (self.e("vC2") if t.C2 else 0.0) + r * self.clamp_expr() + t.R2 * self.e("i2")
            di1 = (a - vm) / t.L1
            di2 = (vm - c) / t.L2
        rows[self.ix["i1"]] = di1
        rows[self.ix["i2"]] = di2
        rows[self.ix["vC1"]] = self.e("i1") / t.C1
        if t.C2:
            rows[self.ix["vC2"]] = self.e("i2") / t.C2
        if self.out == "rc":
            rows[self.ix["vo"]] = (r * self.n_out * self.e("i2") - (self.e("vo") - self.E * self.e("1")) / self.Rout) / self.Co
        md = AffineMode(f"{self._key}|{b1}|{r}", rows[: self.ns, : self.ns], rows[: self.ns, self.ns], label=self.describe(q))
        self._modes[q] = md
        return md

    def _to(self, r_new):
        return lambda q: (q[0], r_new)

    def guards(self, q) -> list[Guard]:
        b1, r = q
        i2 = self.e("i2")

        def zero_i2(z):
            z = z.copy()
            z[self.ix["i2"]] = 0.0
            return z

        if r == 1:
            return [Guard("i₂ → 0 (정류 + 종료)", i2, -1, self._to(0), zero_i2)]
        if r == -1:
            return [Guard("i₂ → 0 (정류 − 종료)", i2, +1, self._to(0), zero_i2)]
        v2f = self.v2_float_expr(b1)
        cl = self.clamp_expr()
        return [Guard("v₂ → +clamp (정류 + 시작)", v2f - cl, +1, self._to(1)), Guard("v₂ → −clamp (정류 − 시작)", v2f + cl, -1, self._to(-1))]

    def after_event(self, q, z):
        b1, r = q
        i2 = float(z[self.ix["i2"]])
        v2f = float(self.v2_float_expr(b1) @ z)
        cl = float(self.clamp_expr() @ z)
        tolv = 1e-9 * max(abs(v2f), abs(cl), self.Vin, 1e-9)
        toli = 1e-9 * self.i_scale
        if r == 0:
            if abs(i2) > toli:
                # a non-zero rectifier current (e.g. a perturbed shooting state) means a diode pair conducts
                return (b1, 1 if i2 > 0 else -1)
            if v2f > cl + tolv:
                return (b1, 1)
            if v2f < -cl - tolv:
                return (b1, -1)
            return q
        if r * i2 > toli:
            return q
        if r * i2 < -toli:
            return (b1, -r)
        # current at zero: stay only if the clamp still pushes current in the conducting direction
        if r * v2f >= cl - tolv:
            return q
        if -r * v2f > cl + tolv:
            return (b1, -r)
        return (b1, 0)

    # ---- gate schedule ------------------------------------------------------------------
    def gate_schedule(self, t0, t1):
        hi, lo = self.levels()
        if self.edges is not None:
            return [(t, (lambda q, lv=lv: (lv, q[1]))) for t, lv in self.edges if t0 - 1e-15 <= t < t1]
        half = self.T / 2
        k0 = int(math.floor(t0 / half + 1e-9))
        k1 = int(math.ceil(t1 / half + 1e-9))
        ev = []
        for k in range(k0, k1 + 1):
            tt = k * half
            if t0 - 1e-15 <= tt < t1:
                lv = hi if k % 2 == 0 else lo
                ev.append((tt, (lambda q, lv=lv: (lv, q[1]))))
        return ev

    def q_start(self, x, b1=None):
        hi, _ = self.levels()
        b1 = hi if b1 is None else b1
        i2 = x[self.ix["i2"]]
        r = 0 if abs(i2) <= 1e-12 * self.i_scale else (1 if i2 > 0 else -1)
        return (b1, r)

    # ---- outputs, energy, powers -------------------------------------------------------
    def outputs(self, q) -> dict[str, np.ndarray]:
        b1, r = q
        i1, i2 = self.e("i1"), self.e("i2")
        out = {
            "i1": i1,
            "i2": i2,
            "im": i1 - i2,
            "vC1": self.e("vC1"),
            "v1": self.v1_expr(b1),
            "vm": self.vm_expr(q),
            "v2": self.v2_expr(q),
            "i1_act": self.n_in * i1,
            "i2_act": self.n_out * i2,
            "irect_act": r * self.n_out * i2,
            "vo": self.vo_expr(),
        }
        if self.tank.C2:
            out["vC2"] = self.e("vC2")
        if self.out == "rc":
            out["iload"] = (self.e("vo") - self.E * self.e("1")) / self.Rout
        return out

    def stored_energy(self) -> np.ndarray:
        t = self.tank
        W = np.zeros((self.m, self.m))
        a, b = self.ix["i1"], self.ix["i2"]
        W[a, a] += t.L1 + t.Lm
        W[b, b] += t.L2 + t.Lm
        W[a, b] -= t.Lm
        W[b, a] -= t.Lm
        W[self.ix["vC1"], self.ix["vC1"]] += t.C1
        if t.C2:
            W[self.ix["vC2"], self.ix["vC2"]] += t.C2
        if self.out == "rc":
            W[self.ix["vo"], self.ix["vo"]] += self.Co
        return W

    def tank_energy(self) -> np.ndarray:
        W = self.stored_energy()
        if self.out == "rc":
            k = self.ix["vo"]
            W[k, k] = 0.0
        return W

    def powers(self, q) -> dict[str, np.ndarray]:
        b1, r = q
        t = self.tank
        i1, i2 = self.e("i1"), self.e("i2")
        p = {
            "p_in": sym_linear(i1, b1 * self.Vin),
            "p_R": t.R1 * np.outer(i1, i1) + t.R2 * np.outer(i2, i2),
        }
        # power into the rectifier (= into the output node): v2 i2 = r n_out vo i2 while conducting, 0 when off
        if r == 0:
            p["p_rect"] = np.zeros((self.m, self.m))
        else:
            c = self.clamp_expr() * r
            p["p_rect"] = 0.5 * (np.outer(c, i2) + np.outer(i2, c))
        if self.out == "rc":
            vo = self.e("vo")
            il = (vo - self.E * self.e("1")) / self.Rout
            p["p_load"] = 0.5 * (np.outer(vo, il) + np.outer(il, vo))
        return p

    def describe(self, q) -> str:
        b1, r = q
        s1 = {1: "+", -1: "−", 0: "0"}[b1]
        s2 = {1: "P", -1: "N", 0: "off"}[r]
        return f"{s1}|{s2}"


# --------------------------------------------------------------------------------------
# Initial guesses, periodic steady state, cycle maps
# --------------------------------------------------------------------------------------


def phasor_guess(sys: ResonantSystem, Rac: float, vo_guess: float | None = None) -> np.ndarray:
    """State at t = 0 from the FHA phasors (sin reference: v1 = V1f sin(wt) for a bridge that is high on [0, T/2))."""
    t = sys.tank
    V1 = bridge_fundamental(sys.Vin, sys.bridge)
    p = fha(t, sys.fs, Rac, V1)
    w = TWO_PI * sys.fs
    x = np.zeros(sys.ns)
    x[sys.ix["i1"]] = p.I1.imag
    x[sys.ix["i2"]] = p.I2.imag
    x[sys.ix["vC1"]] = (p.I1 / (1j * w * t.C1)).imag + (sys.Vin / 2 if sys.bridge == "HB" else 0.0)
    if t.C2:
        x[sys.ix["vC2"]] = (p.I2 / (1j * w * t.C2)).imag
    if sys.out == "rc":
        x[sys.ix["vo"]] = vo_guess if vo_guess is not None else abs(p.H) * V1 * PI / 4 / sys.n_out
    return x


def scales_for(sys: ResonantSystem) -> np.ndarray:
    sc = np.ones(sys.ns)
    I = sys.i_scale
    V = max(sys.Vin, 1.0)
    sc[sys.ix["i1"]] = I
    sc[sys.ix["i2"]] = I
    sc[sys.ix["vC1"]] = V
    if sys.tank.C2:
        sc[sys.ix["vC2"]] = V
    if sys.out == "rc":
        sc[sys.ix["vo"]] = max(abs(sys.Vin / sys.n_out), 1.0)
    return sc


@dataclass
class Periodic:
    x0: np.ndarray
    q0: tuple
    traj: object
    residual: float
    iterations: int
    converged: bool
    multipliers: np.ndarray | None
    monodromy: np.ndarray | None
    pre_cycles: int
    note: str = ""

    @property
    def rho(self) -> float | None:
        return None if self.multipliers is None else float(np.max(np.abs(self.multipliers)))


def run_cycles(sys: ResonantSystem, x, n: int, t0: float = 0.0):
    x = np.asarray(x, dtype=float)
    q = sys.q_start(x)
    t = t0
    for _ in range(n):
        tr = simulate(sys, q, x, t, t + sys.T)
        x = tr.z_end[:-1].copy()
        q = tr.q_end
        t += sys.T
    return x, q


def periodic(sys: ResonantSystem, x_guess, pre_cycles: int = 12, tol: float = 1e-10, max_iter: int = 25, diverge: float = np.inf) -> Periodic:
    """Periodic orbit through t = 0: a few physical cycles from the guess, then Newton shooting.

    The shooting Jacobian is a finite difference of the full event-driven cycle map, so it
    includes the sensitivity of the rectifier turn-on/off instants (plant only: no controller).
    """
    x = np.asarray(x_guess, dtype=float)
    if pre_cycles:
        x, _ = run_cycles(sys, x, pre_cycles)
    q0 = sys.q_start(x)
    sol = shoot(sys, q0, x, sys.T, scales_for(sys), tol=tol, max_iter=max_iter, fd_eps=1e-6, diverge=diverge)
    q0 = sys.q_start(sol.x0)
    return Periodic(sol.x0, q0, sol.trajectory, sol.residual, sol.iterations, sol.converged, sol.multipliers, sol.monodromy, pre_cycles, sol.note)


def cycle_end(sys: ResonantSystem, x, t0: float = 0.0):
    """One period of the constant-frequency drive from x; the rectifier state follows the state."""
    x = np.asarray(x, dtype=float)
    tr = simulate(sys, sys.q_start(x), x, t0, t0 + sys.T)
    return tr.z_end[:-1].copy(), tr


def fd_monodromy(sys: ResonantSystem, x, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of the event-driven cycle map (includes event-time sensitivity)."""
    sc = scales_for(sys)
    x = np.asarray(x, dtype=float)
    M = np.zeros((sys.ns, sys.ns))
    for i in range(sys.ns):
        d = eps * sc[i]
        xp, xm = x.copy(), x.copy()
        xp[i] += d
        xm[i] -= d
        M[:, i] = (cycle_end(sys, xp)[0] - cycle_end(sys, xm)[0]) / (2 * d)
    return M


def periodic_warm(sys: ResonantSystem, x_guess, M: np.ndarray | None = None, tol: float = 1e-10, max_iter: int = 30, diverge: float = 5.0):
    """Chord-Newton shooting for continuation: reuse the Jacobian M of a nearby orbit and refresh it only
    when the residual stops shrinking fast.  Returns (x0, traj, residual, converged, M)."""
    sc = scales_for(sys)
    x = np.asarray(x_guess, dtype=float).copy()
    if M is None:
        M = fd_monodromy(sys, x)
    res_prev = np.inf
    fresh = True
    tr = None
    for _ in range(max_iter):
        fx, tr = cycle_end(sys, x)
        r = fx - x
        res = float(np.max(np.abs(r) / sc))
        if res < tol:
            return x, tr, res, True, M
        if res > diverge or not np.isfinite(res):
            return x, tr, res, False, M
        if res > 0.3 * res_prev and not fresh:
            M = fd_monodromy(sys, x)
            fresh = True
        else:
            fresh = False
        J = (M - np.eye(sys.ns)) * sc[None, :] / sc[:, None]
        step, *_ = np.linalg.lstsq(J, -r / sc, rcond=1e-12)
        x = x + step * sc
        res_prev = res
    return x, tr, res_prev, False, M


def cycle_map(sys: ResonantSystem, x, T: float, t0: float = 0.0):
    """One drive cycle of period T starting at t0 with the bridge high: returns (x_end, trajectory)."""
    hi, lo = sys.levels()
    sys.edges = [(t0, hi), (t0 + T / 2, lo)]
    try:
        q = sys.q_start(np.asarray(x, dtype=float))
        tr = simulate(sys, q, x, t0, t0 + T)
    finally:
        sys.edges = None
    return tr.z_end[:-1].copy(), tr


def summarize(sys: ResonantSystem, traj, T: float, t0: float = 0.0) -> dict:
    """Exact cycle averages/RMS of one periodic cycle."""
    t1 = t0 + T
    d = {
        "P_in": traj.energy(t0, t1, "p_in") / T,
        "P_rect": traj.energy(t0, t1, "p_rect") / T,
        "P_R": traj.energy(t0, t1, "p_R") / T,
        "I1_rms": traj.rms(t0, t1, "i1"),
        "I2_rms": traj.rms(t0, t1, "i2"),
        "Im_rms": traj.rms(t0, t1, "im"),
        "Irect_avg_act": traj.mean(t0, t1, "irect_act"),
        "vC1_pk": max(abs(v) for v in traj.extrema(t0, t1, "vC1")),
        "i1_pk": max(abs(v) for v in traj.extrema(t0, t1, "i1")),
        "im_pk": max(abs(v) for v in traj.extrema(t0, t1, "im")),
    }
    if sys.tank.C2:
        d["vC2_pk"] = max(abs(v) for v in traj.extrema(t0, t1, "vC2"))
    if sys.out == "rc":
        d["P_load"] = traj.energy(t0, t1, "p_load") / T
        d["vo_avg"] = traj.mean(t0, t1, "vo")
        lo, hi = traj.extrema(t0, t1, "vo")
        d["vo_pp"] = hi - lo
    # rectifier conduction share
    off = sum(s.h for s, _, _ in traj.window(t0, t1) if s.q[1] == 0)
    d["off_frac"] = off / T
    return d


def edge_currents(sys: ResonantSystem, traj, T: float, t0: float = 0.0) -> dict:
    """Driven-bridge current at the rising (t0) and falling (t0 + T/2) edges, actual units."""
    zr, _ = traj.state_at(t0)
    zf, _ = traj.state_at(t0 + T / 2)
    return {"i_rise": sys.n_in * float(zr[sys.ix["i1"]]), "i_fall": sys.n_in * float(zf[sys.ix["i1"]])}


def coss_Q(v: float, C0: float, V0: float) -> float:
    return 2.0 * C0 * V0 * (math.sqrt(1.0 + v / V0) - 1.0)


def zvs_screen(i_edge_rise: float, Vbus: float, td: float, C0: float, V0: float) -> dict:
    """Leg commutation screen at the driven bridge's rising edge (SCREEN_ONLY).

    Leg A rises when v1 goes high; its node needs current INTO the node, i.e. the bridge
    output current i1 must be negative at that instant (it lags the voltage: inductive).
    Charge screen: |i| td >= 2 Qoss(Vbus) with the current held at its edge value.
    """
    q_req = 2.0 * coss_Q(Vbus, C0, V0)
    q_av = abs(i_edge_rise) * td
    sign_ok = i_edge_rise < 0
    if abs(i_edge_rise) < 1e-9:
        st = "ZERO_CURRENT"
    elif not sign_ok:
        st = "SIGN_FAIL"
    elif q_av < q_req:
        st = "CHARGE_SHORT"
    else:
        st = "SCREEN_PASS"
    return {"i_edge": i_edge_rise, "q_av": q_av, "q_req": q_req, "margin": (q_av - q_req) / q_req, "status": st}


ZVS_KO = {"SCREEN_PASS": "부호·전하 screen 통과", "CHARGE_SHORT": "부호는 맞지만 전하 부족", "SIGN_FAIL": "전류 방향 반대 → hard switching", "ZERO_CURRENT": "전류 ≈ 0 → 전하 없음"}


# --------------------------------------------------------------------------------------
# Circuit diagram (primary-referred LLC/CLLC with a diode rectifier)
# --------------------------------------------------------------------------------------


def resonant_circuit(kind: str, Vin: float, n_label: str, tank: Tank, bridge: str = "FB", out: str = "rc", reverse: bool = False, vo_label: str = "") -> Circuit:
    """Bridge -> tank -> Lm -> (secondary tank) -> ideal transformer -> diode bridge -> C_o || R_L or battery.

    Two crossings are unavoidable when both the transformer and the load sit outside the bridges
    (the bridge + winding + load graph is K4): node A's lead crosses leg B's mid wire at y = 135 and the
    s2 lead crosses the D2 stub at y = 255.  Neither crossing is drawn with a dot.
    """
    cllc = kind == "CLLC"
    c = Circuit("cllc" if cllc else "llc", 900, 300, title=f"{kind} ({'역방향: 배터리측 bridge 구동, 1차측 다이오드 정류' if reverse else '1차 bridge 구동, 2차 다이오드 정류'}; 1차 환산 T-모델, n = Np/Ns)")
    fb = bridge == "FB"
    xa, xb, yt, yb, ytank, yret = 120, 190, 40, 260, 120, 235
    v1 = c.add("vsource", "VIN", 40, 150, 90, "n·V_bat" if reverse else "V_in", f"{Vin:g} V")
    c.add("nmos", "QAH", xa, 90, 90, "A↑", lpos=(xa - 26, 84, "end"))
    c.add("nmos", "QAL", xa, 210, 90, "A↓", lpos=(xa - 26, 204, "end"))
    c.wire("w_top", v1["a"], (40, yt), (xb if fb else xa, yt))
    c.wire("w_bot", v1["b"], (40, yb), (xb if fb else xa, yb))
    c.wire("w_a_top", (xa, yt), (xa, 60))
    c.wire("w_a_bot", (xa, 240), (xa, yb))
    c.wire("w_a_mid", (xa, 120), (xa, 180))
    if fb:
        c.add("nmos", "QBH", xb, 90, 90, "B↑", lpos=(xb + 22, 84, "start"))
        c.add("nmos", "QBL", xb, 210, 90, "B↓", lpos=(xb + 22, 204, "start"))
        c.wire("w_b_top", (xb, yt), (xb, 60))
        c.wire("w_b_bot", (xb, 240), (xb, yb))
        c.wire("w_b_mid", (xb, 120), (xb, 180))
        c.wire("w_ret_b", (xb, 150), (235, 150), (235, yret))
    else:
        c.wire("w_ret_b", (xa, yb), (xa, 275), (235, 275), (235, yret))
    xL1, xC1 = 265, 340
    L1 = c.add("inductor", "L1", xL1, ytank, 0, "L₁" if cllc else "L_r", f"{tank.L1 * 1e6:.4g} µH", lpos=(xL1, ytank - 32, "middle"))
    C1 = c.add("capacitor", "C1", xC1, ytank, 0, "C₁" if cllc else "C_r", f"{tank.C1 * 1e9:.4g} nF", lpos=(xC1, ytank - 32, "middle"))
    c.wire("w_a_L1", (xa, 150), (145, 150), (145, 135), (222, 135), (222, ytank), L1["a"])
    c.wire("w_L1_C1", L1["b"], C1["a"])
    xm = 390
    Lm = c.add("inductor", "LM", xm, 172, 90, "L_m", f"{tank.Lm * 1e6:.4g} µH", lpos=(xm - 14, 176, "end"))
    c.wire("w_C1_m", C1["b"], (xm, ytank))
    c.wire("w_m_Lm", (xm, ytank), Lm["a"])
    c.wire("w_Lm_ret", Lm["b"], (xm, yret))
    xT = 620
    tr = c.add("transformer", "T", xT, 150, 0, f"n = {n_label}", "")
    if cllc:
        L2 = c.add("inductor", "L2", 450, ytank, 0, "L₂′", f"{tank.L2 * 1e6:.4g} µH", lpos=(450, ytank - 32, "middle"))
        C2 = c.add("capacitor", "C2", 525, ytank, 0, "C₂′", f"{(tank.C2 or 0) * 1e9:.4g} nF", lpos=(525, ytank - 32, "middle"))
        c.wire("w_m_L2", (xm, ytank), L2["a"])
        c.wire("w_L2_C2", L2["b"], C2["a"])
        c.wire("w_2_T", C2["b"], tr["p1"])
        sec = ["L2", "C2", "w_m_L2", "w_L2_C2", "w_2_T"]
    else:
        c.wire("w_2_T", (xm, ytank), tr["p1"])
        sec = ["w_2_T"]
    c.wire("w_ret", (235, yret), (xm, yret), (575, yret), (575, 180), tr["p2"])
    # diode bridge: two vertical legs, AC nodes at the mid wires
    d1, d2 = 700, 760
    D1 = c.add("diode", "D1", d1, 90, 270, "D1", lpos=(d1 - 12, 84, "end"))
    D2 = c.add("diode", "D2", d1, 210, 270, "D2", lpos=(d1 - 12, 216, "end"))
    D3 = c.add("diode", "D3", d2, 90, 270, "D3", lpos=(d2 + 12, 84, "start"))
    D4 = c.add("diode", "D4", d2, 210, 270, "D4", lpos=(d2 + 12, 216, "start"))
    c.wire("w_d1_top", D1["b"], (d1, yt))
    c.wire("w_d3_top", D3["b"], (d2, yt))
    c.wire("w_d2_bot", D2["a"], (d1, 270))
    c.wire("w_d4_bot", D4["a"], (d2, 270))
    c.wire("w_mid1", D1["a"], D2["b"])
    c.wire("w_mid2", D3["a"], D4["b"])
    c.wire("w_s1", tr["s1"], (668, 120), (668, 150), (d1, 150))
    c.wire("w_s2", tr["s2"], (652, 180), (652, 255), (730, 255), (730, 150), (d2, 150))
    xo, xr = 815, 858
    c.wire("w_ptop", (d1, yt), (xr, yt))
    c.wire("w_pbot", (d1, 270), (xr, 270))
    if out == "battery":
        bt = c.add("battery", "BAT", xo, 155, 90, "V_bat" if not reverse else "V_link", vo_label, lpos=(xo - 14, 160, "end"))
        c.wire("w_o_t", (xo, yt), bt["a"])
        c.wire("w_o_b", bt["b"], (xo, 270))
        outp = ["BAT", "w_o_t", "w_o_b"]
        c.wire("w_r_t", (xr, yt), (xr, 60))
    else:
        co = c.add("capacitor", "CO", xo, 155, 90, "C_o", "", lpos=(xo - 14, 160, "end"))
        rl = c.add("resistor", "RL", xr, 155, 90, "R_L", "", lpos=(xr + 12, 160, "start"))
        c.wire("w_o_t", (xo, yt), co["a"])
        c.wire("w_o_b", co["b"], (xo, 270))
        c.wire("w_r_t", (xr, yt), rl["a"])
        c.wire("w_r_b", rl["b"], (xr, 270))
        outp = ["CO", "RL", "w_o_t", "w_o_b", "w_r_t", "w_r_b"]
    dots = [(xa, 150), (xm, ytank), (xm, yret), (d1, 150), (d2, 150), (xo, yt), (xo, 270)]
    if fb:
        dots.append((xb, 150))
    c.dot(*dots)
    c.probe("p_i1", "i1", 200, 135, "right", "i₁" if cllc else "i_r")
    c.probe("p_im", "im", xm, 131, "down", "i_m")
    c.probe("p_i2", "i2", 575 if cllc else 505, ytank, "right", "i₂′")
    hi, lo = (1, -1) if fb else (1, 0)
    base = ["L1", "C1", "LM", "w_a_L1", "w_L1_C1", "w_C1_m", "w_m_Lm", "w_Lm_ret", "w_ret", "w_ret_b", "w_a_mid"] + (["w_b_mid"] if fb else [])
    for b1 in (hi, lo):
        for r in (1, 0, -1):
            key = f"{({1: '+', -1: '−', 0: '0'})[b1]}|{({1: 'P', -1: 'N', 0: 'off'})[r]}"
            act = list(base)
            if b1 == 1:
                act += ["QAH", "w_a_top", "VIN", "w_top", "w_bot"] + (["QBL", "w_b_bot"] if fb else [])
                dim = ["QAL"] + (["QBH"] if fb else [])
            elif b1 == -1:
                act += ["QAL", "w_a_bot", "QBH", "w_b_top", "VIN", "w_top", "w_bot"]
                dim = ["QAH", "QBL"]
            else:
                act += ["QAL", "w_a_bot", "w_bot"]
                dim = ["QAH"]
            if r != 0:
                act += ["T", "w_s1", "w_s2", "w_ptop", "w_pbot"] + sec + outp
                if r == 1:
                    act += ["D1", "D4", "w_d1_top", "w_d4_bot", "w_mid1", "w_mid2"]
                    dim += ["D2", "D3"]
                else:
                    act += ["D3", "D2", "w_d3_top", "w_d2_bot", "w_mid1", "w_mid2"]
                    dim += ["D1", "D4"]
                txt = "정류기 도통: 정류기 입력이 ±n·V_o로 clamp된다 (i₂′ ≠ 0)" + ("; L_m 전압은 C₂′·L₂′를 거쳐 정해진다." if cllc else "; L_m 전압이 ±n·V_o로 고정된다.")
            else:
                act += outp
                dim += ["D1", "D2", "D3", "D4"] + (["L2", "C2"] if cllc else [])
                txt = "정류기 off: i₂′ = 0 제약, 정류기 입력 노드가 떠 있다(|v₂′| < n·V_o). L₁·C₁이 L_m과 함께 공진하고 출력은 C_o(또는 배터리)만 남는다."
            lv = {1: "+V", -1: "−V", 0: "0"}[b1]
            c.mode(key, f"v₁ = {lv}, {RECT_KO[r]}", act, txt, dim=dim)
    return c


def bands_from(traj, t0: float, t1: float, t_offset: float = 0.0) -> list[dict]:
    lab = {}
    out = []
    for iv in traj.mode_intervals(t0, t1):
        m = iv["mode"]
        b, r = m.split("|")
        lab = f"v₁ {b} / {({'P': '정류 +', 'N': '정류 −', 'off': '정류 off'})[r]}"
        out.append({"x0": iv["t0"] - t_offset, "x1": iv["t1"] - t_offset, "mode": m, "label": lab})
    return out


def fmt_f(f: float) -> str:
    return f"{f / 1e3:.3f} kHz"


def phase_deg(z: complex) -> float:
    return math.degrees(cmath.phase(z))


# --------------------------------------------------------------------------------------
# Independent paths used by the checks
# --------------------------------------------------------------------------------------


def ivp_period(sys: ResonantSystem, x0, rtol: float = 1e-11, T: float | None = None):
    """Independent path: the textbook state equations written out directly (no affine matrices)
    and integrated with DOP853; diode transitions are located as solve_ivp events.  Returns x(T)."""
    from scipy.integrate import solve_ivp

    tk = sys.tank
    L1, C1, Lm, L2, C2, R1, R2 = tk.L1, tk.C1, tk.Lm, tk.L2, tk.C2, tk.R1, tk.R2
    llc, has_c2, rc = tk.is_llc, tk.C2 is not None, sys.out == "rc"
    n_out, E = sys.n_out, sys.E
    T = sys.T if T is None else T
    hi, lo = sys.levels()

    def parts(x):
        return x[0], x[1], x[2], (x[3] if has_c2 else 0.0), (x[-1] if rc else sys.Vo)

    def rhs(t, x, v1, s):
        i1, i2, vc1, vc2, vo = parts(x)
        a = v1 - vc1 - R1 * i1
        if s == 0:
            di1, di2 = a / (L1 + Lm), 0.0
        elif llc:
            vm = s * n_out * vo + R2 * i2
            di1 = (a - vm) / L1
            di2 = di1 - vm / Lm
        else:
            c = vc2 + s * n_out * vo + R2 * i2
            vm = (a / L1 + c / L2) / (1 / L1 + 1 / Lm + 1 / L2)
            di1, di2 = (a - vm) / L1, (vm - c) / L2
        d = [di1, di2, i1 / C1]
        if has_c2:
            d.append(i2 / C2)
        if rc:
            d.append((s * n_out * i2 - (vo - E) / sys.Rout) / sys.Co)
        return d

    def v2float(x, v1):
        i1, _, vc1, vc2, _ = parts(x)
        return Lm * (v1 - vc1 - R1 * i1) / (L1 + Lm) - vc2

    def clamp(x):
        return n_out * parts(x)[4]

    sc = np.array([sys.i_scale, sys.i_scale, sys.Vin] + ([sys.Vin] if has_c2 else []) + ([max(sys.Vin / n_out, 1.0)] if rc else []))

    def run(x, ta, tb, v1, s):
        t = ta
        if s == 0:
            vf = v2float(x, v1)
            s = 1 if vf > clamp(x) else (-1 if vf < -clamp(x) else 0)
        guard = 0
        while t < tb - 1e-15 * max(1.0, tb):
            guard += 1
            if guard > 200:
                raise RuntimeError("too many diode events in one half period")
            if s != 0:
                ev = lambda tt, xx, *a: xx[1]  # noqa: E731
                ev.terminal, ev.direction = True, -s
            else:
                ev = lambda tt, xx, *a, v1=v1: abs(v2float(xx, v1)) - clamp(xx)  # noqa: E731
                ev.terminal, ev.direction = True, 1
            sol = solve_ivp(rhs, (t, tb), x, args=(v1, s), method="DOP853", rtol=rtol, atol=rtol * 1e-2 * sc, events=ev)
            x = sol.y[:, -1].copy()
            if sol.status == 1:
                t = float(sol.t_events[0][0])
                x = sol.y_events[0][0].copy()
                if s != 0:
                    x[1] = 0.0
                    vf = v2float(x, v1)
                    s = 1 if vf > clamp(x) else (-1 if vf < -clamp(x) else 0)
                else:
                    s = 1 if v2float(x, v1) > 0 else -1
            else:
                t = tb
        return x, s

    x = np.asarray(x0, dtype=float).copy()
    s0 = 0 if abs(x[1]) <= 1e-12 * sys.i_scale else (1 if x[1] > 0 else -1)
    x, s = run(x, 0.0, T / 2, hi * sys.Vin, s0)
    x, s = run(x, T / 2, T, lo * sys.Vin, s)
    return x


class FHACircuit(HybridSystem):
    """The FHA equivalent circuit itself: sinusoidal source (generated by oscillator states s, c) into the
    tank with the rectifier replaced by R_ac.  Solving it in the time domain checks the phasor algebra;
    it is NOT a switching-rectifier verification (textbook ch.12)."""

    def __init__(self, tank: Tank, f: float, Rac: float, V1f: float = 1.0):
        self.tank, self.f, self.Rac, self.V1f = tank, f, Rac, V1f
        self.has_c2 = tank.C2 is not None
        names = ["i1", "i2", "vC1"] + (["vC2"] if self.has_c2 else []) + ["s", "c"]
        self.state_names = tuple(names)
        self.ix = {k: i for i, k in enumerate(names)}
        self.ns = len(names)

    def mode(self, q):
        t, ns, ix = self.tank, self.ns, self.ix
        m = ns + 1
        e = lambda k: np.eye(m)[ix[k]]  # noqa: E731
        w = TWO_PI * self.f
        a = self.V1f * e("s") - e("vC1") - t.R1 * e("i1")
        vload = self.Rac * e("i2")  # load voltage = Rac * i2 (i2 = load-branch current)
        rows = np.zeros((m, m))
        if t.is_llc:
            vm = vload + t.R2 * e("i2")
            di1 = (a - vm) / t.L1
            di2 = di1 - vm / t.Lm
        else:
            c = (e("vC2") if self.has_c2 else 0.0) + vload + t.R2 * e("i2")
            vm = (a / t.L1 + c / t.L2) / (1 / t.L1 + 1 / t.Lm + 1 / t.L2)
            di1, di2 = (a - vm) / t.L1, (vm - c) / t.L2
        rows[ix["i1"]], rows[ix["i2"]], rows[ix["vC1"]] = di1, di2, e("i1") / t.C1
        if self.has_c2:
            rows[ix["vC2"]] = e("i2") / t.C2
        rows[ix["s"]] = w * e("c")
        rows[ix["c"]] = -w * e("s")
        return AffineMode(f"fha|{self.tank}|{self.f}|{self.Rac}", rows[:ns, :ns], rows[:ns, ns])

    def outputs(self, q):
        m = self.ns + 1
        return {"vload": self.Rac * np.eye(m)[self.ix["i2"]], "v1": self.V1f * np.eye(m)[self.ix["s"]]}


def fha_time_domain_gain(tank: Tank, f: float, Rac: float) -> float:
    """|v_load| / |v1| of the periodic solution of the FHA circuit, from exact exponentials."""
    from ..engine.switched import propagator

    sysf = FHACircuit(tank, f, Rac)
    md = sysf.mode(None)
    T = 1.0 / f
    P = propagator(md, T)
    ns = sysf.ns
    k = ns - 2  # circuit states before the oscillator
    osc0 = np.zeros(ns + 1)
    osc0[sysf.ix["c"]] = 1.0  # s(0) = 0, c(0) = 1  ->  v1 = V1f sin(w t)
    osc0[-1] = 1.0
    rhs = P[:k, :] @ osc0  # x(T) = P_xx x0 + P_x,osc osc0 ; with x0 = 0 part removed below
    Pxx = P[:k, :k]
    x0 = np.linalg.solve(np.eye(k) - Pxx, rhs)
    z0 = osc0.copy()
    z0[:k] = x0
    tr = simulate(sysf, None, z0[:-1], 0.0, T)
    lo, hi = tr.extrema(0.0, T, "vload")
    return 0.5 * (hi - lo)
