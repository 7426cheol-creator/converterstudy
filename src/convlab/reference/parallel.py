"""Textbook E03 closed forms (EX03): static current sharing, DPT overlap energy and deskew, screens.

Comparison only; lab code never imports this module.
"""

from __future__ import annotations


def current_divider(R, I_total):
    """Parallel resistive branches at a common terminal voltage: I_k = I (1/R_k) / sum(1/R_j)."""
    G = [1.0 / r for r in R]
    s = sum(G)
    return [I_total * g / s for g in G]


def overlap_energy(V, I, tr):
    """Linear V fall and I rise over the same t_r: E0 = V I t_r / 6."""
    return V * I * tr / 6.0


def shifted_overlap_energy(V, I, tr, tau):
    """Energy with the current channel delayed by tau (|tau| < t_r) over a window that covers the
    whole shifted overlap, including the pre-transition interval where v = V and the current is already non-zero.

    tau >= 0 (current late):  E = V I (t_r - tau)^3 / (6 t_r^2)
    tau < 0  (current early, s = -tau): E = V I s^2 / t_r + V I/t_r^2 [t_r a^2/2 + t_r s a - a^3/3 - s a^2/2], a = t_r - s
    """
    if tau >= 0:
        return V * I * (tr - tau) ** 3 / (6.0 * tr * tr)
    s = -tau
    a = tr - s
    mid = tr * a * a / 2 + tr * s * a - a**3 / 3 - s * a * a / 2
    return V * I * s * s / tr + V * I / (tr * tr) * mid


def csi_voltage(Ls, didt):
    return Ls * didt


def skew_current(dt, didt):
    return dt * didt
