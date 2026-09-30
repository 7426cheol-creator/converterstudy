"""Periodic steady state of driven hybrid systems.

Two independent routes are provided and reported side by side:

* ``cycle_to_steady``: brute-force cycling from a physical initial condition with an
  explicit per-cycle criterion (normalised state change AND stored-energy change per
  cycle below thresholds for several consecutive cycles).  This is what a lab bench
  or a naive simulation does; it also shows *how long* the start-up transient lasts.
* ``shoot``: Newton iteration on the Poincare map F(x0) = Phi_T(x0) - x0 with a
  finite-difference Jacobian taken on the full event-driven map.  Because the map is
  evaluated by the hybrid simulator, the Jacobian includes the sensitivity of
  state-dependent event times (diode turn-off, DCM entry); gate edges are fixed by
  the external clock.  Its eigenvalues are the Floquet multipliers of the plant for
  this operating point; a controller that is not part of the simulated state is NOT
  included, so plant-only multipliers never replace a closed-loop stability claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .switched import HybridSystem, SimulationError, Trajectory, simulate

__all__ = ["CycleHistory", "PeriodicSolution", "poincare", "cycle_to_steady", "shoot"]


def poincare(system: HybridSystem, q0, x0, t0: float, T: float) -> tuple[np.ndarray, object, Trajectory]:
    traj = simulate(system, q0, x0, t0, t0 + T)
    return traj.z_end[:-1].copy(), traj.q_end, traj


def _stored(system: HybridSystem, x: np.ndarray) -> float:
    z = np.append(x, 1.0)
    return 0.5 * float(z @ system.stored_energy() @ z)


@dataclass
class CycleHistory:
    state_change: list[float] = field(default_factory=list)  # normalised max |dx|/scale per cycle
    energy_change: list[float] = field(default_factory=list)  # |dW| / E_ref per cycle
    steady_cycle: int | None = None
    cycles_run: int = 0
    criteria: dict = field(default_factory=dict)
    x_end: np.ndarray | None = None
    q_end: object = None
    t_end: float = 0.0


def cycle_to_steady(
    system: HybridSystem,
    q0,
    x0,
    T: float,
    scales,
    energy_ref: float,
    tol_state: float = 1e-6,
    tol_energy: float = 1e-6,
    consecutive: int = 3,
    max_cycles: int = 20000,
    t0: float = 0.0,
    keep_every: int = 0,
) -> tuple[CycleHistory, list[Trajectory]]:
    """Cycle the system until the per-cycle criterion holds for ``consecutive`` cycles.

    ``energy_ref`` is the normalising energy per cycle (use the larger of the port
    energy processed per cycle and 1 % of rated power x T, so that a zero-power corner
    does not divide by zero).  Returns the history and the kept trajectories (the last
    cycle is always kept).
    """
    scales = np.asarray(scales, dtype=float)
    hist = CycleHistory(criteria={"tol_state": tol_state, "tol_energy": tol_energy, "consecutive": consecutive, "energy_ref_J": energy_ref})
    x = np.asarray(x0, dtype=float)
    q = q0
    t = t0
    kept: list[Trajectory] = []
    ok_run = 0
    W_prev = _stored(system, x)
    for k in range(max_cycles):
        traj = simulate(system, q, x, t, t + T)
        x_new = traj.z_end[:-1].copy()
        q = traj.q_end
        W_new = _stored(system, x_new)
        e_x = float(np.max(np.abs(x_new - x) / scales))
        e_w = abs(W_new - W_prev) / energy_ref
        hist.state_change.append(e_x)
        hist.energy_change.append(e_w)
        if keep_every and (k % keep_every == 0):
            kept.append(traj)
        x, W_prev = x_new, W_new
        t += T
        if e_x < tol_state and e_w < tol_energy:
            ok_run += 1
            if ok_run >= consecutive and hist.steady_cycle is None:
                hist.steady_cycle = k + 1
                kept.append(traj)
                break
        else:
            ok_run = 0
        if k == max_cycles - 1:
            kept.append(traj)
    hist.cycles_run = len(hist.state_change)
    hist.x_end, hist.q_end, hist.t_end = x, q, t
    return hist, kept


@dataclass
class PeriodicSolution:
    x0: np.ndarray
    q0: object
    residual: float  # normalised max |Phi(x0) - x0| / scale
    iterations: int
    converged: bool
    monodromy: np.ndarray | None
    multipliers: np.ndarray | None
    trajectory: Trajectory | None
    note: str = ""

    @property
    def spectral_radius(self) -> float | None:
        if self.multipliers is None:
            return None
        return float(np.max(np.abs(self.multipliers)))


def _fd_jacobian(system, q0, x, t0, T, scales, eps):
    n = x.size
    J = np.zeros((n, n))
    for i in range(n):
        d = eps * scales[i]
        xp = x.copy()
        xp[i] += d
        xm = x.copy()
        xm[i] -= d
        fp, _, _ = poincare(system, q0, xp, t0, T)
        fm, _, _ = poincare(system, q0, xm, t0, T)
        J[:, i] = (fp - fm) / (2 * d)
    return J


def shoot(
    system: HybridSystem,
    q0,
    x_guess,
    T: float,
    scales,
    tol: float = 1e-10,
    max_iter: int = 30,
    fd_eps: float = 1e-6,
    t0: float = 0.0,
    rcond: float = 1e-10,
) -> PeriodicSolution:
    """Newton shooting for the periodic orbit through the section t = t0 (mod T).

    A singular (I - M), e.g. the undamped DC-offset mode of an ideal DAB inductor,
    is handled with a minimum-norm least-squares step; the returned note says so and
    the multiplier at 1 is reported rather than hidden.
    """
    scales = np.asarray(scales, dtype=float)
    x = np.asarray(x_guess, dtype=float).copy()
    note = ""
    it = 0
    res = np.inf
    for it in range(1, max_iter + 1):
        try:
            fx, qT, _ = poincare(system, q0, x, t0, T)
        except SimulationError as exc:  # pragma: no cover - reported to caller
            return PeriodicSolution(x, q0, np.inf, it, False, None, None, None, note=f"simulation failed: {exc}")
        r = fx - x
        res = float(np.max(np.abs(r) / scales))
        if res < tol:
            break
        J = _fd_jacobian(system, q0, x, t0, T, scales, fd_eps) - np.eye(x.size)
        # scale columns/rows for conditioning
        Js = J * scales[None, :] / scales[:, None]
        rs = r / scales
        step, _, rank, _ = np.linalg.lstsq(Js, -rs, rcond=rcond)
        if rank < x.size:
            note = f"(I - M) rank {rank} < {x.size}: a neutral mode exists; minimum-norm Newton step used"
        x = x + step * scales
    converged = res < tol
    M = _fd_jacobian(system, q0, x, t0, T, scales, fd_eps)
    mult = np.linalg.eigvals(M)
    _, _, traj = poincare(system, q0, x, t0, T)
    return PeriodicSolution(x, q0, res, it, converged, M, mult, traj, note=note)
