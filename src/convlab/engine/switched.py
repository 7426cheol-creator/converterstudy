"""Exact event-driven simulation of switched-affine circuits.

Inside one circuit topology state (which switches and diodes conduct) an ideal
power-electronics circuit is linear time-invariant:

    dx/dt = A x + b            (b collects the constant sources of that state)

We work with the augmented state z = [x; 1] so that dz/dt = F z with
F = [[A, b], [0, 0]].  Over an interval of length h the exact solution is
z(h) = expm(F h) z(0): no time step, no truncation error.  Gate edges are
scheduled events; diode turn-on/turn-off and DCM entry are *guards* (a linear
function of z crossing zero) that are located with a bracketing root finder on
the exact solution.  State continuity across events is therefore automatic.

Exact integrals of every quadratic form of z over a segment (port energy,
resistor loss, RMS) come from the second-moment matrix

    S2 = integral_0^h z(t) z(t)^T dt,

computed with the Kronecker-sum identity  vec(S2) = [int_0^h exp((F (+) F) t) dt] vec(z0 z0^T),
which only uses exponentials of the (stable or marginal) circuit dynamics and is
therefore safe for stiff snubber-like modes.  Because z_m = 1, the last column of
S2 is the first moment integral z dt, so averages come for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

__all__ = [
    "AffineMode",
    "Guard",
    "Segment",
    "Trajectory",
    "HybridSystem",
    "simulate",
    "propagator",
    "segment_moments",
    "SimulationError",
]


class SimulationError(RuntimeError):
    """Raised when the hybrid simulation cannot continue consistently.

    This is a *numerical/model* failure (Zeno chattering, inconsistent diode
    state, event budget exhausted), deliberately distinct from a physical
    instability, which is a valid simulation result.
    """


# --------------------------------------------------------------------------------------
# Modes and exact propagation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AffineMode:
    """One linear circuit state dx/dt = A x + b.

    ``key`` must uniquely identify (topology state, parameter set) so that
    propagators for repeated interval lengths can be cached.
    """

    key: str
    A: np.ndarray
    b: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float).reshape(-1)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
            raise ValueError(f"mode {self.key}: inconsistent A {A.shape} / b {b.shape}")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def F(self) -> np.ndarray:
        n = self.n
        F = np.zeros((n + 1, n + 1))
        F[:n, :n] = self.A
        F[:n, n] = self.b
        return F

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """dz/dt at augmented state z (last entry 0)."""
        return self.F @ z


class _PropagatorCache:
    """Caches expm(F h) and the second-moment operators for repeated (mode, h)."""

    def __init__(self, max_entries: int = 4096) -> None:
        self._prop: dict[tuple[str, float], np.ndarray] = {}
        self._mom: dict[tuple[str, float], np.ndarray] = {}
        self.max_entries = max_entries

    def propagator(self, mode: AffineMode, h: float) -> np.ndarray:
        k = (mode.key, float(h))
        P = self._prop.get(k)
        if P is None:
            P = expm(mode.F * h)
            if len(self._prop) < self.max_entries:
                self._prop[k] = P
        return P

    def moment_operator(self, mode: AffineMode, h: float) -> np.ndarray:
        k = (mode.key, float(h))
        Psi = self._mom.get(k)
        if Psi is None:
            F = mode.F
            m = F.shape[0]
            I = np.eye(m)
            K = np.kron(F, I) + np.kron(I, F)
            M = np.zeros((2 * m * m, 2 * m * m))
            M[: m * m, : m * m] = K
            M[: m * m, m * m :] = np.eye(m * m)
            Psi = expm(M * h)[: m * m, m * m :]
            if len(self._mom) < self.max_entries:
                self._mom[k] = Psi
        return Psi

    def clear(self) -> None:
        self._prop.clear()
        self._mom.clear()


_CACHE = _PropagatorCache()


def propagator(mode: AffineMode, h: float) -> np.ndarray:
    """Exact state-transition matrix expm(F h) of the augmented system."""
    if h < 0:
        raise ValueError("negative interval")
    return _CACHE.propagator(mode, h)


def segment_moments(mode: AffineMode, z0: np.ndarray, h: float) -> np.ndarray:
    """Exact S2 = integral_0^h z z^T dt for the segment starting at z0 (m x m)."""
    m = z0.shape[0]
    Psi = _CACHE.moment_operator(mode, h)
    # vec() is column-major in the Kronecker identity; for symmetric z0 z0^T it does not matter.
    S2 = (Psi @ np.outer(z0, z0).reshape(-1, order="F")).reshape(m, m, order="F")
    return 0.5 * (S2 + S2.T)


# --------------------------------------------------------------------------------------
# Guards (state-dependent events)
# --------------------------------------------------------------------------------------


@dataclass
class Guard:
    """A state event: g(z) = c . z crosses zero in ``direction``.

    direction: +1 means g goes from negative to positive, -1 the opposite,
    0 either way.  ``action`` maps the current discrete state to the new one
    (e.g. a diode stops conducting).  ``reset`` optionally returns a corrected
    continuous state (used to pin a current that must be exactly zero after a
    diode turns off, removing round-off).
    """

    name: str
    c: np.ndarray
    direction: int
    action: Callable[[object], object]
    reset: Callable[[np.ndarray], np.ndarray] | None = None

    def value(self, z: np.ndarray) -> float:
        return float(np.dot(self.c, z))


def _crossed(g0: float, g1: float, direction: int) -> bool:
    if direction > 0:
        return g0 < 0.0 <= g1
    if direction < 0:
        return g0 > 0.0 >= g1
    return (g0 < 0.0 <= g1) or (g0 > 0.0 >= g1)


_EIG_CACHE: dict[str, tuple[float, float]] = {}


def _samples_needed(mode: AffineMode, h: float) -> int:
    """Enough samples that no oscillatory guard can hide two crossings between samples."""
    info = _EIG_CACHE.get(mode.key)
    if info is None:
        ev = np.linalg.eigvals(mode.A) if mode.n else np.array([0.0])
        info = (float(np.max(np.abs(ev.imag))), float(np.max(np.abs(ev.real))))
        if len(_EIG_CACHE) < 4096:
            _EIG_CACHE[mode.key] = info
    wmax, smax = info
    n = int(np.ceil(16.0 * wmax * h / (2 * np.pi))) + int(np.ceil(2.0 * smax * h))
    return int(min(max(n, 24), 20000))


def _on_boundary(value: float, c: np.ndarray, z: np.ndarray) -> bool:
    scale = float(np.max(np.abs(c * z))) if z.size else 1.0
    return abs(value) <= 1e-11 * max(scale, 1e-30)


def first_guard_crossing(
    mode: AffineMode,
    z0: np.ndarray,
    hmax: float,
    guards: Sequence[Guard],
    xtol: float = 1e-16,
) -> tuple[float, int] | None:
    """Earliest time in [0, hmax) at which one of the guards fires, located on the exact solution.

    A guard that starts exactly on its boundary fires immediately only if its
    derivative points in the firing direction (the consistency rule for ideal
    diodes: a diode whose current is zero and falling turns off now; one whose
    current is zero and rising keeps conducting).  Otherwise it is armed and
    fires when it later crosses in the firing direction.  When the first
    derivative is itself zero to round-off (a diode that starts conducting at the
    instant its blocking voltage reaches the clamp), the second derivative
    decides; this keeps the event from ping-ponging between two states that are
    both consistent to first order.
    """
    if not guards or hmax <= 0:
        return None
    C = np.array([g.c for g in guards])
    g_start = C @ z0
    F = mode.F
    dz = F @ z0
    dg = C @ dz
    on_bnd = [_on_boundary(g_start[k], C[k], z0) for k in range(len(guards))]
    for k, g in enumerate(guards):
        if on_bnd[k]:
            s = dg[k]
            scale1 = float(np.abs(C[k]) @ (np.abs(F) @ np.abs(z0)))
            if abs(s) <= 1e-9 * scale1:
                d2 = float(C[k] @ (F @ dz))
                scale2 = float(np.abs(C[k]) @ (np.abs(F) @ (np.abs(F) @ np.abs(z0))))
                s = d2 if abs(d2) > 1e-9 * scale2 else 0.0
            if (g.direction > 0 and s > 0) or (g.direction < 0 and s < 0) or (g.direction == 0 and s != 0):
                return (0.0, k)
    N = _samples_needed(mode, hmax)
    dt = hmax / N
    step = expm(mode.F * dt)
    z = z0.copy()
    gprev = g_start.copy()
    for k, g in enumerate(guards):
        if on_bnd[k] and g.direction:
            # armed: treat as just on the non-firing side
            gprev[k] = -float(g.direction) * 1e-300

    def value_at(ck: np.ndarray, t: float) -> float:
        return float(ck @ (expm(mode.F * t) @ z0))

    for s in range(1, N + 1):
        znew = step @ z
        gnew = C @ znew
        hits = [k for k in range(len(guards)) if _crossed(gprev[k], gnew[k], guards[k].direction)]
        if hits:
            ta, tb = (s - 1) * dt, s * dt
            best: tuple[float, int] | None = None
            for k in hits:
                ck = C[k]
                d = guards[k].direction
                a = ta
                fa = value_at(ck, a)
                if s == 1 and on_bnd[k]:
                    # started on the boundary: find where the guard is strictly on the non-firing side
                    a = None
                    for frac in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 0.5):
                        v = value_at(ck, dt * frac)
                        if (d > 0 and v < 0) or (d < 0 and v > 0) or (d == 0 and v != 0):
                            a, fa = dt * frac, v
                    if a is None:
                        tk = 0.0
                        if best is None or tk < best[0]:
                            best = (tk, k)
                        continue
                fb = value_at(ck, tb)
                if fa == 0.0:
                    tk = a
                elif fb == 0.0 or fa * fb > 0:
                    tk = tb
                else:
                    tk = brentq(lambda t, ck=ck: value_at(ck, t), a, tb, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=300)
                if best is None or tk < best[0]:
                    best = (tk, k)
            return best
        z, gprev = znew, gnew
    return None


# --------------------------------------------------------------------------------------
# Hybrid system description and simulation
# --------------------------------------------------------------------------------------


class HybridSystem:
    """Interface a topology implements.

    Discrete state ``q`` is any hashable object (typically a tuple of switch/diode
    states).  Subclasses implement ``mode``, ``guards`` and ``gate_schedule``.
    ``outputs`` returns named affine output rows over z; ``stored_energy`` a
    symmetric matrix W with stored energy = 1/2 z^T W z; ``powers`` named symmetric
    matrices Q with instantaneous power = z^T Q z (ports and resistive losses).
    """

    state_names: tuple[str, ...] = ()
    state_units: tuple[str, ...] = ()

    def mode(self, q) -> AffineMode:  # pragma: no cover - interface
        raise NotImplementedError

    def guards(self, q) -> list[Guard]:  # pragma: no cover - interface
        return []

    def gate_schedule(self, t0: float, t1: float) -> list[tuple[float, Callable[[object], object]]]:
        """Scheduled gate events in [t0, t1) as (time, action(q) -> q)."""
        return []

    def after_event(self, q, z: np.ndarray):
        """Resolve the discrete state right after an event (e.g. which diode takes the current).

        Default: keep q.  Topologies with ideal diodes override this to apply the
        complementarity rules (conducting diode current >= 0, blocking diode
        voltage <= 0) and must raise SimulationError if no consistent state exists.
        """
        return q

    def outputs(self, q) -> dict[str, np.ndarray]:
        return {}

    def stored_energy(self) -> np.ndarray:
        n = len(self.state_names)
        return np.zeros((n + 1, n + 1))

    def powers(self, q) -> dict[str, np.ndarray]:
        return {}

    def describe(self, q) -> str:
        return str(q)


@dataclass
class Segment:
    t0: float
    h: float
    q: object
    mode: AffineMode
    z0: np.ndarray
    z1: np.ndarray
    reason: str = ""  # what ended the segment

    @property
    def t1(self) -> float:
        return self.t0 + self.h


@dataclass
class Trajectory:
    system: HybridSystem
    segments: list[Segment] = field(default_factory=list)

    # ---- basic accessors -----------------------------------------------------------
    @property
    def t_end(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    @property
    def z_end(self) -> np.ndarray:
        return self.segments[-1].z1

    @property
    def q_end(self):
        return self.segments[-1].q

    def window(self, t0: float, t1: float) -> list[tuple[Segment, float, float]]:
        """Segments overlapping [t0, t1] with local start offset and clipped duration."""
        out = []
        for s in self.segments:
            a = max(t0, s.t0)
            b = min(t1, s.t1)
            if b - a > 0:
                out.append((s, a - s.t0, b - a))
        return out

    # ---- exact integrals -----------------------------------------------------------
    def integral_quadratic(self, t0: float, t1: float, form: Callable[[object], np.ndarray | None]) -> float:
        """Exact integral of z^T Q(q) z over [t0, t1]; ``form(q)`` returns Q or None (=0)."""
        total = 0.0
        for s, off, h in self.window(t0, t1):
            Q = form(s.q)
            if Q is None:
                continue
            za = s.z0 if off == 0.0 else propagator(s.mode, off) @ s.z0
            S2 = segment_moments(s.mode, za, h)
            total += float(np.sum(Q * S2))
        return total

    def integral_output(self, t0: float, t1: float, name: str) -> float:
        """Exact integral of the affine output ``name``."""

        def form(q):
            c = self.system.outputs(q).get(name)
            if c is None:
                return None
            m = c.shape[0]
            Q = np.zeros((m, m))
            Q[:, m - 1] += 0.5 * c
            Q[m - 1, :] += 0.5 * c
            return Q

        return self.integral_quadratic(t0, t1, form)

    def integral_output_squared(self, t0: float, t1: float, name: str) -> float:
        def form(q):
            c = self.system.outputs(q).get(name)
            return None if c is None else np.outer(c, c)

        return self.integral_quadratic(t0, t1, form)

    def integral_output_product(self, t0: float, t1: float, a: str, b: str) -> float:
        def form(q):
            outs = self.system.outputs(q)
            ca, cb = outs.get(a), outs.get(b)
            if ca is None or cb is None:
                return None
            return 0.5 * (np.outer(ca, cb) + np.outer(cb, ca))

        return self.integral_quadratic(t0, t1, form)

    def mean(self, t0: float, t1: float, name: str) -> float:
        return self.integral_output(t0, t1, name) / (t1 - t0)

    def rms(self, t0: float, t1: float, name: str) -> float:
        return float(np.sqrt(max(self.integral_output_squared(t0, t1, name), 0.0) / (t1 - t0)))

    def energy(self, t0: float, t1: float, power_name: str) -> float:
        return self.integral_quadratic(t0, t1, lambda q: self.system.powers(q).get(power_name))

    # ---- sampling for plots and peaks ------------------------------------------------
    def state_at(self, t: float) -> tuple[np.ndarray, object]:
        for s in self.segments:
            if s.t0 <= t <= s.t1:
                return propagator(s.mode, t - s.t0) @ s.z0, s.q
        raise ValueError("time outside trajectory")

    def sample(self, names: Iterable[str], t0: float | None = None, t1: float | None = None, per_segment: int = 12) -> dict:
        """Exact samples of outputs; both sides of every event are kept so jumps render as edges."""
        names = list(names)
        t0 = self.segments[0].t0 if t0 is None else t0
        t1 = self.t_end if t1 is None else t1
        ts: list[float] = []
        cols: dict[str, list[float]] = {n: [] for n in names}
        modes: list[str] = []
        for s, off, h in self.window(t0, t1):
            outs = self.system.outputs(s.q)
            lin = np.allclose(s.mode.A, 0.0)
            k = 2 if lin else max(2, per_segment)
            za = s.z0 if off == 0.0 else propagator(s.mode, off) @ s.z0
            step = propagator(s.mode, h / (k - 1))
            z = za
            for j in range(k):
                if j:
                    z = step @ z
                ts.append(s.t0 + off + h * j / (k - 1))
                modes.append(self.system.describe(s.q))
                for n in names:
                    c = outs.get(n)
                    cols[n].append(float(c @ z) if c is not None else float("nan"))
        return {"t": ts, "mode": modes, **cols}

    def extrema(self, t0: float, t1: float, name: str, per_segment: int = 64) -> tuple[float, float]:
        """Min and max of an output over [t0, t1] (exact at segment ends, dense + refined inside)."""
        lo, hi = np.inf, -np.inf
        for s, off, h in self.window(t0, t1):
            c = self.system.outputs(s.q).get(name)
            if c is None:
                continue
            za = s.z0 if off == 0.0 else propagator(s.mode, off) @ s.z0
            lin = np.allclose(s.mode.A, 0.0)
            k = 2 if lin else per_segment
            step = propagator(s.mode, h / (k - 1))
            z = za
            vals = []
            for j in range(k):
                if j:
                    z = step @ z
                vals.append(float(c @ z))
            lo, hi = min(lo, min(vals)), max(hi, max(vals))
            if not lin:
                # refine interior extrema: derivative c . F z changes sign
                cF = c @ s.mode.F
                zz = za
                d_prev = float(cF @ zz)
                for j in range(1, k):
                    zz = step @ zz
                    d = float(cF @ zz)
                    if d_prev * d < 0:
                        ta, tb = h * (j - 1) / (k - 1), h * j / (k - 1)
                        f = lambda t: float(cF @ (propagator(s.mode, t) @ za))  # noqa: E731
                        try:
                            tm = brentq(f, ta, tb, xtol=1e-18)
                            v = float(c @ (propagator(s.mode, tm) @ za))
                            lo, hi = min(lo, v), max(hi, v)
                        except ValueError:
                            pass
                    d_prev = d
        return lo, hi

    def mode_intervals(self, t0: float, t1: float) -> list[dict]:
        """Merged intervals of identical discrete state (for circuit/waveform linking)."""
        out: list[dict] = []
        for s, off, h in self.window(t0, t1):
            label = self.system.describe(s.q)
            a = s.t0 + off
            if out and out[-1]["mode"] == label and abs(out[-1]["t1"] - a) < 1e-15 * max(1.0, abs(a)):
                out[-1]["t1"] = a + h
            else:
                out.append({"t0": a, "t1": a + h, "mode": label})
        return out


def simulate(
    system: HybridSystem,
    q0,
    x0: Sequence[float],
    t_start: float,
    t_stop: float,
    max_events: int = 2_000_000,
    min_dwell: float = 0.0,
) -> Trajectory:
    """Run the hybrid simulation from (q0, x0) over [t_start, t_stop].

    Returns the exact piecewise trajectory.  Raises SimulationError on
    inconsistent discrete states or Zeno-like event accumulation.
    """
    z = np.append(np.asarray(x0, dtype=float), 1.0)
    q = system.after_event(q0, z)
    t = t_start
    traj = Trajectory(system)
    gates = sorted(system.gate_schedule(t_start, t_stop), key=lambda e: e[0])
    gi = 0
    # apply gate events scheduled exactly at t_start
    while gi < len(gates) and gates[gi][0] <= t_start:
        q = gates[gi][1](q)
        gi += 1
    q = system.after_event(q, z)
    events = 0
    zero_len_run = 0
    while t < t_stop - 1e-18 * max(1.0, abs(t_stop)):
        mode = system.mode(q)
        t_gate = gates[gi][0] if gi < len(gates) else t_stop
        t_next = min(t_gate, t_stop)
        hmax = t_next - t
        guards = system.guards(q)
        hit = first_guard_crossing(mode, z, hmax, guards) if guards else None
        if hit is not None and hit[0] < hmax:
            h, k = hit
            z1 = propagator(mode, h) @ z if h > 0 else z.copy()
            traj.segments.append(Segment(t, h, q, mode, z, z1, reason=f"guard:{guards[k].name}"))
            t += h
            g = guards[k]
            if g.reset is not None:
                z1 = g.reset(z1)
            q = g.action(q)
            q = system.after_event(q, z1)
            z = z1
            zero_len_run = zero_len_run + 1 if h <= min_dwell else 0
            if zero_len_run > 50:
                raise SimulationError(f"Zeno-like event accumulation at t={t:.9g} s in state {system.describe(q)}")
        else:
            h = hmax
            z1 = propagator(mode, h) @ z
            traj.segments.append(Segment(t, h, q, mode, z, z1, reason="gate" if t_next == t_gate else "stop"))
            t = t_next
            z = z1
            zero_len_run = 0
            while gi < len(gates) and gates[gi][0] <= t + 1e-18 * max(1.0, abs(t)):
                q = gates[gi][1](q)
                gi += 1
            q = system.after_event(q, z)
        events += 1
        if events > max_events:
            raise SimulationError("event budget exhausted")
    # drop zero-length segments (they carry no time; keep reasons in the neighbours)
    traj.segments = [s for s in traj.segments if s.h > 0]
    return traj
