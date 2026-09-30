"""Closed forms for EX11 (textbook E11): GUM propagation, binomial bounds, lognormal yield.

Used only for comparison; the lab re-derives each number on a different path
(Monte Carlo with an explicit seed, a binomial-sum root, finite differences).
"""

from __future__ import annotations

import math

from scipy.stats import beta, norm


def u_difference(u1: float, u2: float, rho: float = 0.0) -> float:
    """Standard uncertainty of x1 - x2: sqrt(u1^2 + u2^2 - 2 rho u1 u2)."""
    return math.sqrt(max(u1 * u1 + u2 * u2 - 2.0 * rho * u1 * u2, 0.0))


def u_efficiency(P_in: float, P_out: float, u_in: float, u_out: float, rho: float = 0.0) -> float:
    """eta = P_out/P_in: c_out = 1/P_in, c_in = -P_out/P_in^2 (first-order GUM)."""
    c_out = 1.0 / P_in
    c_in = -P_out / P_in**2
    var = (c_out * u_out) ** 2 + (c_in * u_in) ** 2 + 2.0 * rho * c_out * c_in * u_out * u_in
    return math.sqrt(max(var, 0.0))


def standard_from_spec(a: float, kind: str) -> float:
    """Standard uncertainty from a stated figure a: 'standard' (a is 1 sigma), 'rect_limit' (+-a
    rectangular, a/sqrt 3), 'k2' (expanded uncertainty with k = 2, a/2)."""
    return {"standard": a, "rect_limit": a / math.sqrt(3.0), "k2": a / 2.0}[kind]


def zero_failure_upper(n: int, conf: float) -> float:
    """One-sided upper bound for p after 0 failures in n IID trials: 1 - (1 - conf)^(1/n)."""
    return -math.expm1(math.log(1.0 - conf) / n)  # = 1 - (1 - conf)^(1/n) without cancellation


def clopper_pearson_upper(k: int, n: int, conf: float) -> float:
    """One-sided Clopper-Pearson upper bound: Beta quantile (k + 1, n - k) at conf."""
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def clopper_pearson_two_sided(k: int, n: int, conf: float) -> tuple[float, float]:
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def lognormal_ratio_yield(sig_L: float, sig_C: float, rho: float, tol: float, bias: float = 0.0) -> float:
    """P(|f/f0 - 1| <= tol) for f = 1/(2 pi sqrt(LC)) with ln L, ln C Gaussian (sigma, rho) and a
    multiplicative model bias (1 + bias).  ln(f/f0) ~ N(ln(1 + bias), s^2), s^2 = (sL^2 + sC^2 + 2 rho sL sC)/4."""
    s = 0.5 * math.sqrt(max(sig_L**2 + sig_C**2 + 2.0 * rho * sig_L * sig_C, 0.0))
    mu = math.log(1.0 + bias)
    lo, hi = math.log(1.0 - tol), math.log(1.0 + tol)
    if s == 0:
        return 1.0 if lo <= mu <= hi else 0.0
    return float(norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s))


def rlc_step_response(t, V: float, L: float, C: float, R: float):
    """Capacitor voltage of a series RLC driven by a step V at t = 0 (underdamped closed form)."""
    import numpy as np

    t = np.asarray(t, dtype=float)
    a = R / (2.0 * L)
    w0 = 1.0 / math.sqrt(L * C)
    wd = math.sqrt(max(w0 * w0 - a * a, 1e-30))
    return V * (1.0 - np.exp(-a * t) * (np.cos(wd * t) + a / wd * np.sin(wd * t)))
