"""Exact algebra on piecewise-linear waveforms (jumps allowed between segments).

Used as an *independent* path for waveforms whose segments are straight lines
(ideal inductor currents driven by piecewise-constant voltages): averages, RMS and
product integrals are computed segment by segment in closed form,

    int i dt   = dt (ia + ib) / 2
    int i^2 dt = dt (ia^2 + ia ib + ib^2) / 3
    int a b dt = dt (2 a0 b0 + a0 b1 + a1 b0 + 2 a1 b1) / 6,

so no uniform time step is involved and edge placement errors cannot hide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PWL"]


@dataclass
class PWL:
    """Segments [t[k], t[k+1]] with linear values from y0[k] to y1[k]."""

    t: np.ndarray
    y0: np.ndarray
    y1: np.ndarray

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.y0 = np.asarray(self.y0, dtype=float)
        self.y1 = np.asarray(self.y1, dtype=float)
        if not (self.t.size == self.y0.size + 1 == self.y1.size + 1):
            raise ValueError("PWL needs len(t) = len(y0) + 1 = len(y1) + 1")
        if np.any(np.diff(self.t) < 0):
            raise ValueError("PWL breakpoints must be non-decreasing")

    # ---- constructors ---------------------------------------------------------------
    @classmethod
    def from_slopes(cls, t, slopes, y_start: float) -> "PWL":
        """Continuous waveform from piecewise-constant slopes (an ideal inductor current)."""
        t = np.asarray(t, dtype=float)
        dt = np.diff(t)
        y = np.concatenate([[y_start], y_start + np.cumsum(np.asarray(slopes) * dt)])
        return cls(t, y[:-1], y[1:])

    @classmethod
    def step(cls, t, values) -> "PWL":
        """Piecewise-constant waveform (a bridge voltage)."""
        v = np.asarray(values, dtype=float)
        return cls(t, v, v)

    # ---- basic properties -----------------------------------------------------------
    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.t)

    @property
    def span(self) -> float:
        return float(self.t[-1] - self.t[0])

    def integral(self) -> float:
        return float(np.sum(self.dt * (self.y0 + self.y1) / 2))

    def mean(self) -> float:
        return self.integral() / self.span

    def integral_sq(self) -> float:
        a, b = self.y0, self.y1
        return float(np.sum(self.dt * (a * a + a * b + b * b) / 3))

    def rms(self) -> float:
        return float(np.sqrt(self.integral_sq() / self.span))

    def max_abs(self) -> float:
        return float(max(np.max(np.abs(self.y0)), np.max(np.abs(self.y1))))

    def max(self) -> float:
        return float(max(np.max(self.y0), np.max(self.y1)))

    def min(self) -> float:
        return float(min(np.min(self.y0), np.min(self.y1)))

    def shifted(self, dy: float) -> "PWL":
        return PWL(self.t, self.y0 + dy, self.y1 + dy)

    def scaled(self, k: float) -> "PWL":
        return PWL(self.t, self.y0 * k, self.y1 * k)

    # ---- evaluation on a common refinement -------------------------------------------
    def value_at(self, tq: float, side: str = "right") -> float:
        """Value at tq; at a breakpoint ``side`` chooses the segment starting (right) or ending (left) there."""
        t = self.t
        if side == "right":
            k = int(np.searchsorted(t, tq, side="right") - 1)
        else:
            k = int(np.searchsorted(t, tq, side="left") - 1)
        k = min(max(k, 0), self.y0.size - 1)
        span = t[k + 1] - t[k]
        if span == 0:
            return float(self.y0[k])
        s = (tq - t[k]) / span
        return float(self.y0[k] + (self.y1[k] - self.y0[k]) * s)

    def refined(self, breaks) -> "PWL":
        """Same waveform with extra breakpoints (values unchanged)."""
        tb = np.unique(np.concatenate([self.t, np.asarray(breaks, dtype=float)]))
        tb = tb[(tb >= self.t[0]) & (tb <= self.t[-1])]
        y0 = np.array([self.value_at(a, "right") for a in tb[:-1]])
        y1 = np.array([self.value_at(b, "left") for b in tb[1:]])
        return PWL(tb, y0, y1)

    def product_integral(self, other: "PWL") -> float:
        """Exact integral of self(t) * other(t) over the common span."""
        if abs(self.t[0] - other.t[0]) > 1e-15 * max(1.0, abs(self.t[0])) or abs(self.t[-1] - other.t[-1]) > 1e-15 * max(
            1.0, abs(self.t[-1])
        ):
            raise ValueError("product_integral needs equal spans")
        a = self.refined(other.t)
        b = other.refined(self.t)
        if a.t.size != b.t.size or np.max(np.abs(a.t - b.t)) > 0:
            raise ValueError("refinement mismatch")
        d = a.dt
        return float(np.sum(d * (2 * a.y0 * b.y0 + a.y0 * b.y1 + a.y1 * b.y0 + 2 * a.y1 * b.y1) / 6))

    def times(self, other: "PWL") -> "PWL":
        """Pointwise product; exact only when one factor is piecewise constant (checked)."""
        a = self.refined(other.t)
        b = other.refined(self.t)
        if not (np.allclose(a.y0, a.y1) or np.allclose(b.y0, b.y1)):
            raise ValueError("product of two sloped PWL is quadratic; use product_integral")
        return PWL(a.t, a.y0 * b.y0, a.y1 * b.y1)

    def plot_points(self) -> tuple[list[float], list[float]]:
        """Vertices for plotting, keeping both sides of every jump."""
        ts: list[float] = []
        ys: list[float] = []
        for k in range(self.y0.size):
            ts += [float(self.t[k]), float(self.t[k + 1])]
            ys += [float(self.y0[k]), float(self.y1[k])]
        return ts, ys
