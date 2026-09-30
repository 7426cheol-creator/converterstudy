"""Helpers shared by lab modules (not a lab itself)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..engine.switched import HybridSystem, Trajectory
from ..model.result import Check, Result

__all__ = [
    "decimate_minmax",
    "energy_ledger",
    "series_from_traj",
    "bands_from_traj",
    "fmt",
    "sym_linear",
    "independent_ivp_check",
]


def fmt(v: float, d: int = 6) -> str:
    return f"{v:.{d}g}"


def sym_linear(c: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Symmetric Q such that z^T Q z = scale * (c . z) for augmented z (z[-1] = 1)."""
    m = c.shape[0]
    Q = np.zeros((m, m))
    Q[:, m - 1] += 0.5 * scale * c
    Q[m - 1, :] += 0.5 * scale * c
    return Q


def decimate_minmax(x, y, buckets: int = 2000):
    """Keep min and max of each bucket so switching envelopes survive decimation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 2 * buckets:
        return x.tolist(), y.tolist()
    edges = np.linspace(0, x.size, buckets + 1).astype(int)
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        i1, i2 = int(np.argmin(seg)), int(np.argmax(seg))
        for i in sorted({i1, i2}):
            xs.append(float(x[a + i]))
            ys.append(float(seg[i]))
    return xs, ys


def series_from_traj(res: Result, traj: Trajectory, names: dict[str, tuple[str, str]], t0=None, t1=None, per_segment=12, t_offset=0.0, max_points=6000, suffix="") -> None:
    """Add exact samples of trajectory outputs as result series. ``names`` maps output -> (label, unit)."""
    smp = traj.sample(list(names), t0, t1, per_segment=per_segment)
    t = np.asarray(smp["t"]) - t_offset
    for key, (label, unit) in names.items():
        xs, ys = decimate_minmax(t, smp[key], max_points // 2)
        res.add_series(key + suffix, label, unit, xs, ys)


def bands_from_traj(traj: Trajectory, t0: float, t1: float, labels: dict[str, str], t_offset: float = 0.0, max_bands: int = 400) -> list[dict]:
    out = []
    for iv in traj.mode_intervals(t0, t1):
        out.append({"x0": iv["t0"] - t_offset, "x1": iv["t1"] - t_offset, "mode": iv["mode"], "label": labels.get(iv["mode"], iv["mode"])})
        if len(out) >= max_bands:
            break
    return out


def energy_ledger(traj: Trajectory, system: HybridSystem, t0: float, t1: float, inputs: list[str], outputs: list[str], losses: list[str], rated_power: float) -> dict:
    """Exact energy balance over [t0, t1]: E_in - E_out - E_loss - dW (J) and its normalised residual.

    The normaliser is max(port energy processed, 1 % of rated power x window), so a
    zero-power corner does not blow the relative residual up (textbook ch.19).
    """
    E_in = sum(traj.energy(t0, t1, k) for k in inputs)
    E_out = sum(traj.energy(t0, t1, k) for k in outputs)
    E_loss = sum(traj.energy(t0, t1, k) for k in losses)
    za, _ = traj.state_at(t0)
    zb, _ = traj.state_at(t1)
    W = system.stored_energy()
    dW = 0.5 * float(zb @ W @ zb) - 0.5 * float(za @ W @ za)
    resid = E_in - E_out - E_loss - dW
    norm = max(abs(E_in), abs(E_out), 0.01 * rated_power * (t1 - t0))
    return {"E_in": E_in, "E_out": E_out, "E_loss": E_loss, "dW": dW, "residual": resid, "normalised": resid / norm, "norm": norm}


def ledger_check(led: dict, threshold: float = 0.005, what: str = "") -> Check:
    return Check(
        name="에너지 잔차 (E_in − E_out − E_loss − ΔW)",
        status="PASS" if abs(led["normalised"]) <= threshold else "FAIL",
        value=led["normalised"],
        unit="rel",
        threshold=threshold,
        path="포트 에너지·저항 손실·저장에너지를 같은 구간에서 정확 적분 (상태식과 독립적으로 정의한 포트 전력)",
        independent=True,
        detail=(
            f"{what}E_in {led['E_in']:.6g} J, E_out {led['E_out']:.6g} J, E_loss {led['E_loss']:.6g} J, ΔW {led['dW']:.3g} J → 잔차 {led['residual']:.3g} J"
        ),
    )


def independent_ivp_check(rhs_by_mode: Callable, schedule: list[tuple[float, float, str]], x0, rtols=(1e-6, 1e-8, 1e-10), event_fn=None):
    """Integrate the same circuit written independently as plain ODEs with scipy's RK45.

    ``schedule`` is a list of (t_start, t_end, mode) intervals with fixed boundaries;
    ``event_fn(mode)`` may return (event_function, next_mode) for state events such as
    DCM entry.  Returns end states for each rtol (tolerance-convergence study).
    """
    from scipy.integrate import solve_ivp

    ends = []
    for rtol in rtols:
        x = np.asarray(x0, dtype=float)
        for ta, tb, mode in schedule:
            t = ta
            m = mode
            while t < tb - 1e-18:
                ev = event_fn(m) if event_fn else None
                kw = {}
                if ev is not None:
                    f, _nxt = ev
                    f.terminal = True
                    f.direction = -1
                    kw["events"] = f
                sol = solve_ivp(lambda tt, xx, m=m: rhs_by_mode(m, xx), (t, tb), x, method="RK45", rtol=rtol, atol=rtol * 1e-3, **kw)
                x = sol.y[:, -1]
                if ev is not None and sol.t_events[0].size:
                    t = float(sol.t_events[0][0])
                    x = sol.y_events[0][0].copy()
                    m = ev[1]
                    x[0] = 0.0
                else:
                    t = tb
        ends.append(x.copy())
    return ends


def geomspace_int(a: float, b: float, n: int) -> list[float]:
    return [float(v) for v in np.geomspace(a, b, n)]


def safe(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v
