"""EX04 - Transformer design closure: move an electrically feasible n.L onto a real part (textbook E04, extends FL07).

Five experiments, each labelled with its model level:
  A+C  DAB 50:3 turns / B / window / copper screen: integer-turns candidates around n = 50/3; each one
       recomputes the SPS operating current (exact piecewise-linear waveform, checked against the FL08
       closed form and the exact switched engine), B from the largest bridge voltage, the bare-copper
       window (textbook 168.329 mm2 with rounded currents, 168.3235 mm2 exact) and the DC copper loss.
       Insulation, thermal and core-loss inputs are MISSING_INPUT and block a final selection.
  A    CLLC n = 0.93 -> integer turns: FHA operating map at the textbook corners for each candidate
       (13:14 etc.), the physical secondary tank kept or re-tuned, B and window with a synthetic core,
       and the continuous n window the operating map allows (the n = 1 seed failure is kept).
  A    harmonic copper loss: textbook [10, 3, 2] A with R_ac [20, 40, 60] mOhm = 2.60 W vs 2.26 W with
       R_dc only; the exact Fourier series of the FL08 DAB current (closed-form segment integrals,
       Parseval and FFT checks) with a synthetic Dowell R_ac(h f0, T); core loss MISSING_INPUT.
  A    tolerance and correlation: L, C +-5 % corners (142.857 / 157.895 / 150.188 kHz) and a correlated
       log-normal Monte Carlo (stated rho, seed) of f_r, the FHA operating frequency and the high-corner
       gain margin of the n = 0.93 CLLC (162.811 kHz, 136.099 / 147.061 kHz at nominal).
  C    identification plan: open/short-circuit LCR tests on a synthetic transformer network with winding
       capacitance, solved exactly in the time domain (sinusoidal source as oscillator states) and by a
       phasor ladder; self-resonance, winding connection and shorting-strap referral (n^2) errors.
Nothing here selects a real transformer: the core, insulation, thermal and material data are synthetic
or missing, and every FHA result is a candidate before switching verification (EX05).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..engine.pwl import PWL
from ..engine.switched import AffineMode, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..reference import magnetics as ref
from ._common import bands_from_traj, energy_ledger, ledger_check
from .fl07_magnetics import DAB_LABELS, DABT, dab_steady, dowell, sps_phase

TB_E04 = TextbookRef("expert-e04.-transformer-설계-closure-전기적으로-가능한-nl을-실제-부품으로-옮기기-ex04", "E04. transformer 설계 closure — n·L을 실제 부품으로 [EX04]")
TB_E05 = TextbookRef("expert-e05-cllc-gain-곡선에서-실제-동역학으로", "E05 · CLLC: gain 곡선에서 실제 동역학으로")
TB_13 = TextbookRef("cllc-설계-실패와-다중-해를-직접-방어하기-fl10", "13. CLLC — 설계 실패와 다중 해 [FL10]")
TB_10 = TextbookRef("자성체-컨버터-전문가로-가는-실제-관문-fl07", "10. 자성체 [FL07]")
TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB [FL08]")
TB_E13 = TextbookRef("expert-e13-빠르게-깊어지기-위한-실행-순서와-통과-기준", "E13. 실행 순서와 통과 기준")

TWO_PI = 2.0 * math.pi
MU0 = 4e-7 * math.pi
RHO20 = 1.72e-8  # copper resistivity at 20 C (textbook ch.10)
ALPHA_CU = 3.93e-3  # 1/K, copper
# FL08 voltage corners the textbook names (V_H, V_L): nominal 800/48, 550/36 (P_max 2.0625 kW), 900/36 (mismatch)
DAB_CORNERS = ((800.0, 48.0, "800/48 V"), (550.0, 36.0, "550/36 V"), (900.0, 36.0, "900/36 V"))
# FL10 CLLC corners (battery V_o, link V_i) and the textbook tank (11 kW, primary-referred symmetric)
CLLC_CORNERS = ((650.0, 700.0), (800.0, 800.0), (920.0, 850.0))
CLLC_LR, CLLC_CR, CLLC_LM, CLLC_N, CLLC_P = 40e-6, 28.144773e-9, 200e-6, 0.93, 11e3
F1_RMS = 2.0 * math.sqrt(2.0) / math.pi  # fundamental RMS of a +-1 square wave


def rho_cu(T: float) -> float:
    return RHO20 * (1.0 + ALPHA_CU * (T - 20.0))


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


# ======================================================================================
# DAB SPS current as an exact piecewise-linear waveform and its exact Fourier series
# ======================================================================================


def sps_current(V1: float, V2: float, L: float, fs: float, phi: float) -> PWL:
    """Zero-DC SPS inductor current over one period (textbook FL08 four-interval form), primary-referred.

    Slopes per rad a = (V1 + V2)/(w L) on [0, phi] and b = (V1 - V2)/(w L) on [phi, pi];
    i0 = -(a phi + b (pi - phi))/2, second half antisymmetric.
    """
    w = TWO_PI * fs
    T = 1.0 / fs
    a = (V1 + V2) / (w * L)
    b = (V1 - V2) / (w * L)
    i0 = -(a * phi + b * (math.pi - phi)) / 2.0
    ip = i0 + a * phi
    tp = phi / w
    h = T / 2.0
    return PWL([0.0, tp, h, h + tp, T], [i0, ip, -i0, -ip], [ip, -i0, -ip, i0])


def dab_point(V1: float, V2: float, L: float, fs: float, P: float):
    """(phi, current PWL) for power P, or None when P > P_max = V1 V2/(8 f L)."""
    phi = sps_phase(V1, V2, L, fs, P)
    if not math.isfinite(phi):
        return None
    return phi, sps_current(V1, V2, L, fs, phi)


def pwl_fourier(p: PWL, hmax: int) -> np.ndarray:
    """Exact complex Fourier coefficients c_h = (1/T) int i(t) e^{-j h w t} dt (h = 0..hmax) of one period.

    Each straight segment i = y0 + s tau on [t_k, t_k + D] integrates in closed form:
    e^{-l t_k} [y0 (1 - e^{-l D})/l + s ((1 - e^{-l D})/l^2 - D e^{-l D}/l)],  l = j h w.
    """
    T = p.span
    w = TWO_PI / T
    d = np.diff(p.t)
    keep = d > 0
    t0 = (p.t[:-1] - p.t[0])[keep]
    d = d[keep]
    y0 = p.y0[keep]
    s = (p.y1[keep] - y0) / d
    c = np.zeros(hmax + 1, dtype=complex)
    c[0] = p.mean()
    lam = 1j * np.arange(1, hmax + 1)[:, None] * w
    E = np.exp(-lam * d[None, :])
    term = y0 * (1.0 - E) / lam + s * ((1.0 - E) / lam**2 - d * E / lam)
    c[1:] = (np.exp(-lam * t0[None, :]) * term).sum(axis=1) / T
    return c


def pwl_fourier_jumps(p: PWL, hmax: int) -> tuple[np.ndarray, float]:
    """Second path for a continuous periodic PWL (integration by parts twice):
    c_h = sum_k ds_k e^{-j h w t_k} / (T (j h w)^2), ds_k = slope jump at breakpoint t_k.

    Returns (c_0..c_hmax, sum |ds_k|); the latter bounds |c_h| <= sum|ds| / (T (h w)^2) for every h >= 1.
    """
    T = p.span
    w = TWO_PI / T
    d = np.diff(p.t)
    keep = d > 0
    t0 = (p.t[:-1] - p.t[0])[keep]
    s = (p.y1[keep] - p.y0[keep]) / d[keep]
    ds = s - np.roll(s, 1)
    lam = 1j * np.arange(1, hmax + 1) * w
    c = np.zeros(hmax + 1, dtype=complex)
    c[0] = p.mean()
    c[1:] = (np.exp(-lam[:, None] * t0[None, :]) * ds[None, :]).sum(axis=1) / (T * lam**2)
    return c, float(np.sum(np.abs(ds)))


# ======================================================================================
# Screens shared by the turns experiments
# ======================================================================================


def turn_candidates(n_t: float, Ns_max: int, extra=()) -> list[tuple[int, int]]:
    """Integer pairs (N_p, N_s) with N_p = floor/ceil(n_t N_s) for N_s = 1..Ns_max, plus ``extra``."""
    out = set()
    for Ns in range(1, Ns_max + 1):
        x = n_t * Ns
        for Np in (math.floor(x + 1e-9), math.ceil(x - 1e-9)):
            if Np >= 1:
                out.add((int(Np), Ns))
    for c in extra:
        if c[0] >= 1 and c[1] >= 1:
            out.add((int(c[0]), int(c[1])))
    return sorted(out, key=lambda c: (c[1], c[0]))


def window_req(Np, Ip, Ns, Is, J, ku) -> float:
    """Bare-copper window lower bound (J in A/mm^2 -> mm^2)."""
    return (Np * Ip + Ns * Is) / (J * ku)


def copper_dc(Np, Ip, Ns, Is, J, MLT, T) -> float:
    """DC copper loss at a fixed current density: R = rho N MLT/(I/J) so P = rho MLT J (N_p I_p + N_s I_s)."""
    return rho_cu(T) * MLT * (J * 1e6) * (Np * Ip + Ns * Is)


# ======================================================================================
# CLLC first-harmonic model (textbook ch.13, implemented here)
# ======================================================================================


def fha(f, n, Vo, P, Lr1, Cr1, Lm, Lr2p, Cr2p):
    """H = Z_p/(Z_r1 + Z_p) R_ac'/(Z_r2' + R_ac'), Z_p = Z_m || (Z_r2' + R_ac'), R_ac' = 8 n^2 R_dc/pi^2.

    Returns (H, Z_in, I2'/I1, R_ac').  All arguments broadcast (primary-referred secondary tank).
    """
    w = TWO_PI * np.asarray(f, dtype=float)
    Rac = 8.0 * np.asarray(n, dtype=float) ** 2 * (np.asarray(Vo, dtype=float) ** 2 / np.asarray(P, dtype=float)) / math.pi**2
    Zr1 = 1j * w * Lr1 + 1.0 / (1j * w * Cr1)
    Zs = 1j * w * Lr2p + 1.0 / (1j * w * Cr2p) + Rac
    Zm = 1j * w * Lm
    Zp = Zm * Zs / (Zm + Zs)
    Zin = Zr1 + Zp
    return Zp / Zin * Rac / Zs, Zin, Zm / (Zm + Zs), Rac


def cllc_tank(n: float, n0: float, Lr: float, Cr: float, Lm: float, tank2: str):
    """(Lr1, Cr1, Lm, Lr2', Cr2'): 'retune' keeps referred symmetry; 'keep' keeps the physical
    secondary L_r2 = L_r/n0^2, C_r2 = C_r n0^2 designed for n0, so the referred values move with n."""
    if tank2 == "retune":
        return Lr, Cr, Lm, Lr, Cr
    return Lr, Cr, Lm, n * n * Lr / (n0 * n0), Cr * n0 * n0 / (n * n)


def fha_solve(n, Vo, Vi, P, Lr1, Cr1, Lm, Lr2p, Cr2p, fmin, fmax, G=1801, iters=46, chunk=400):
    """Vectorized FHA solutions of |H| = n V_o/V_i on [fmin, fmax] for arrays of cases.

    Grid sign changes are refined by bisection on all cases at once; a root counts when Z_in is
    inductive there.  Returns (upper root, lower root, number of inductive roots, max inductive |H| on the
    grid).  NaN when absent.
    """
    arrs = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, dtype=float)) for a in (n, Vo, Vi, P, Lr1, Cr1, Lm, Lr2p, Cr2p)))
    n, Vo, Vi, P, Lr1, Cr1, Lm, Lr2p, Cr2p = [a.ravel().copy() for a in arrs]
    M = n.size
    req = n * Vo / Vi
    F = np.linspace(fmin, fmax, G)
    upper = np.full(M, np.nan)
    lower = np.full(M, np.nan)
    count = np.zeros(M, dtype=int)
    gmax = np.full(M, np.nan)
    for c0 in range(0, M, chunk):
        sl = slice(c0, min(M, c0 + chunk))
        pr = [x[sl, None] for x in (n, Vo, P, Lr1, Cr1, Lm, Lr2p, Cr2p)]
        H, Zin, _, _ = fha(F[None, :], *pr)
        aH = np.abs(H)
        ind = Zin.imag > 0
        gm = np.where(ind, aH, -np.inf).max(axis=1)
        gmax[sl] = np.where(np.isfinite(gm), gm, np.nan)
        g = aH - req[sl, None]
        pos = g > 0
        rs, ks = np.nonzero(pos[:, :-1] != pos[:, 1:])
        if rs.size == 0:
            continue
        rows = rs + c0
        lo, hi = F[ks].copy(), F[ks + 1].copy()
        plo = pos[rs, ks]
        par = [x[rows] for x in (n, Vo, P, Lr1, Cr1, Lm, Lr2p, Cr2p)]
        rq = req[rows]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            gmid = np.abs(fha(mid, *par)[0]) - rq
            same = (gmid > 0) == plo
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        root = 0.5 * (lo + hi)
        indr = fha(root, *par)[1].imag > 0
        for r, f, ok in zip(rows, root, indr):
            if not ok:
                continue
            count[r] += 1
            if np.isnan(upper[r]) or f > upper[r]:
                lower[r] = upper[r]
                upper[r] = f
            elif np.isnan(lower[r]) or f > lower[r]:
                lower[r] = f
    return upper, lower, count, gmax


def fha_roots_brentq(n, Vo, Vi, P, tank, fmin, fmax, G=4001):
    """Independent scalar path: dense grid + brentq on each bracket; returns [(f, inductive)]."""
    req = n * Vo / Vi
    F = np.linspace(fmin, fmax, G)
    g = np.abs(fha(F, n, Vo, P, *tank)[0]) - req
    out = []
    for k in np.nonzero(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
        r = brentq(lambda f: abs(complex(fha(f, n, Vo, P, *tank)[0])) - req, F[k], F[k + 1], xtol=1e-9, rtol=1e-14)
        out.append((r, complex(fha(r, n, Vo, P, *tank)[1]).imag > 0))
    return out


def fha_currents(f, n, Vo, Vi, P, tank):
    """FHA primary current I_1, referred secondary I_2' and actual secondary n I_2' (RMS), and R_ac', Z_in."""
    _, Zin, k2, Rac = fha(f, n, Vo, P, *tank)
    I1 = F1_RMS * Vi / abs(complex(Zin))
    I2p = I1 * abs(complex(k2))
    return I1, I2p, n * I2p, float(Rac), complex(Zin)


def max_ind_gain(n, Vo, P, tank, fmin, fmax, G=4001):
    """Largest |H| over the inductive part of [fmin, fmax] (grid, then bounded refinement or the exact edge)."""
    F = np.linspace(fmin, fmax, G)
    H, Zin, _, _ = fha(F, n, Vo, P, *tank)
    ind = Zin.imag > 0
    if not ind.any():
        return float("nan"), float("nan")
    a = np.where(ind, np.abs(H), -np.inf)
    k = int(np.argmax(a))

    def gain(f):
        return abs(complex(fha(f, n, Vo, P, *tank)[0]))

    best = (float(a[k]), float(F[k]))
    if 0 < k < G - 1 and ind[k - 1] and ind[k + 1]:
        r = minimize_scalar(lambda f: -gain(f), bounds=(F[k - 1], F[k + 1]), method="bounded", options={"xatol": 1e-4})
        if gain(r.x) > best[0]:
            best = (gain(r.x), float(r.x))
    else:
        for j in (k - 1, k + 1):
            if 0 <= j < G and not ind[j]:
                fb = brentq(lambda f: complex(fha(f, n, Vo, P, *tank)[1]).imag, min(F[j], F[k]), max(F[j], F[k]), xtol=1e-9)
                if gain(fb) > best[0]:
                    best = (gain(fb), fb)
    return best


def gain_slope(f, n, Vo, P, tank, h=10.0) -> float:
    """d|H|/df by a +-10 Hz central difference (per kHz), as in the textbook."""
    return (abs(complex(fha(f + h, n, Vo, P, *tank)[0])) - abs(complex(fha(f - h, n, Vo, P, *tank)[0]))) / (2 * h) * 1e3


# ======================================================================================
# Experiment 1: DAB 50:3 turns, B, window and copper screen
# ======================================================================================


class _Cat:
    PASS = "screen 통과"
    B = "B 초과"
    W = "window 초과"
    N = "n 허용오차 밖"
    OP = "운전점 해 없음"


def dab_equiv_circuit(V1: float, V2: float, L: float) -> Circuit:
    c = Circuit("ex04_dab", 560, 230, title="DAB 등가 (L_m → ∞, 2차는 1차 환산)")
    c.add("vsource", "V1", 80, 130, 90, "v₁ (1차 bridge)", f"±{V1:g} V", lpos=(102, 126, "start"))
    lk = c.add("inductor", "L", 280, 50, 0, "L (직렬, 1차 환산)", f"{L * 1e6:.4g} µH", lpos=(280, 24, "middle"))
    c.add("vsource", "V2", 480, 130, 90, "v₂′ = n·v_L", f"±{V2:.4g} V", lpos=(458, 126, "end"))
    c.wire("w_a", (80, 100), (80, 50), lk["a"])
    c.wire("w_b", lk["b"], (480, 50), (480, 100))
    c.wire("w_r", (80, 160), (80, 200), (480, 200), (480, 160))
    c.probe("pip", "ip", 170, 50, "right", "i_p")
    all_ = ["V1", "L", "V2", "w_a", "w_b", "w_r"]
    txt = {
        "+-": "v₁ = +V₁, v₂′ = −V₂′: L에 V₁ + V₂′",
        "++": "v₁ = +V₁, v₂′ = +V₂′: L에 V₁ − V₂′ (n이 맞으면 0 — 평평한 구간)",
        "-+": "v₁ = −V₁, v₂′ = +V₂′: L에 −(V₁ + V₂′)",
        "--": "v₁ = −V₁, v₂′ = −V₂′: L에 −V₁ + V₂′",
    }
    for key, lab in DAB_LABELS.items():
        c.mode(key, lab, all_, txt[key] + ". n이 바뀌면 V₂′ = n·V_L이 바뀌어 이 구간의 기울기가 달라진다.")
    c.notes.append("창 면적 계산의 I_p·I_s는 이 전류의 RMS(교재 방식: L_m → ∞)이다. 자화전류는 표의 L_m 열로 따로 본다.")
    return c


def run_turns_window(v: dict) -> Result:
    res = Result("EX04", "turns_window", "A (screen) + C (정확 스위칭 검산)")
    Np0, Ns0 = int(v["Np_ref"]), int(v["Ns_ref"])
    n_t = Np0 / Ns0
    V1, VL, P, fs, L = v["V1"], v["VL"], v["P"], v["fs"], v["L"]
    J, ku, Ae, MLT, Tw = v["J"], v["ku"], v["Ae"] * 1e-6, v["MLT"], v["T_w"]
    Aw_eff = v["Aw"] * (1.0 - v["ins_frac"] / 100.0)
    n_tol = v["n_tol"] / 100.0
    Lm0 = v["Lm_ref"]
    worst = v["I_basis"] == "worst"

    def vw_max(n):
        return max(v["V1_max"], n * v["VL_max"])

    cands = turn_candidates(n_t, int(v["Ns_max"]), extra=[(Np0 - 1, Ns0), (Np0, Ns0), (Np0 + 1, Ns0)])
    rows, pts = [], []
    for Np, Ns in cands:
        n = Np / Ns
        pt = dab_point(V1, n * VL, L, fs, P)
        corner_I = []
        for a, b, _ in DAB_CORNERS:
            q = dab_point(a, n * b, L, fs, P)
            corner_I.append(q[1].rms() if q else float("nan"))
        op_ok = pt is not None and all(math.isfinite(x) for x in corner_I)
        Ip = pt[1].rms() if pt else float("nan")
        Ip_w = max(corner_I) if op_ok else float("nan")
        Ib = Ip_w if worst else Ip
        Is = n * Ib
        Aw = window_req(Np, Ib, Ns, Is, J, ku)
        B = vw_max(n) / (4.0 * fs * Np * Ae)
        Pcu = copper_dc(Np, Ib, Ns, Is, J, MLT, Tw)
        Lm_c = Lm0 * (Np / Np0) ** 2
        Im_pk = n * VL / (4.0 * fs * Lm_c)
        dn = n / n_t - 1.0
        fails = []
        if not op_ok:
            fails.append(_Cat.OP)
        if B > v["B_allow"]:
            fails.append(_Cat.B)
        if not (Aw <= Aw_eff):
            fails.append(_Cat.W)
        if abs(dn) > n_tol + 1e-12:
            fails.append(_Cat.N)
        cat = _Cat.PASS if not fails else fails[0]
        pts.append({"Np": Np, "Ns": Ns, "n": n, "dn": dn, "phi": pt[0] if pt else float("nan"), "Ip": Ip, "Ip_w": Ip_w, "Ib": Ib, "Is": Is, "Aw": Aw, "B": B,
                    "Pcu": Pcu, "Lm": Lm_c, "Im": Im_pk, "fails": fails, "cat": cat})
        rows.append([f"{Np}:{Ns}", n, 100 * dn, pt[0] if pt else float("nan"), f"{Ip:.4f} / {Ip_w:.4f}", Is, B, Aw, Pcu, Lm_c * 1e3, f"{Im_pk:.3f}",
                     "screen 통과 (최종 아님)" if not fails else ", ".join(fails)])
    refp = next(p for p in pts if p["Np"] == Np0 and p["Ns"] == Ns0)
    passing = [p for p in pts if not p["fails"]]
    tb = Np0 == 50 and Ns0 == 3 and _close(V1, 800) and _close(VL, 48) and _close(P, 1500) and _close(fs, 1e5) and _close(L, 200e-6)
    tb_w = tb and _close(J, 4.0) and _close(ku, 0.30) and not worst
    # ---- metrics ------------------------------------------------------------------------
    res.add_metric("n_t", "연속 설계 n = N_p/N_s (기준 후보)", n_t, "", basis=f"{Np0}:{Ns0}")
    res.add_metric("Ip", f"기준 후보 {Np0}:{Ns0} 1차 RMS I_p (정확 PWL)", refp["Ip"], "A", ref=2.019881509 if tb else None, ref_label="FL08 2.019881509 A", tol=1e-9,
                   basis="공칭 800/48 V·1.5 kW, L_m → ∞ (부하+순환 전류)")
    res.add_metric("Is", "기준 후보 2차 RMS I_s = n·I_p", n_t * refp["Ip"], "A", ref=33.66469 if tb else None, ref_label="FL08 33.6647 A", tol=1e-6, basis="실제 2차 권선")
    Ip_r, Is_r = round(refp["Ip"], 2), round(n_t * refp["Ip"], 3)
    Aw_tb = window_req(Np0, Ip_r, Ns0, Is_r, J, ku)
    Aw_ex = window_req(Np0, refp["Ip"], Ns0, n_t * refp["Ip"], J, ku)
    res.add_metric("Aw_tb", "window 하한 (교재 방식: 반올림 전류)", Aw_tb, "mm²", ref=168.329 if tb_w else None, ref_label="교재 168.329 mm² (I_p 2.02 A, I_s 33.665 A)", tol=1e-5,
                   basis=f"I_p {Ip_r:g} A, I_s {Is_r:g} A, J {J:g} A/mm², k_u {ku:g}")
    res.add_metric("Aw_exact", "window 하한 (정확 전류)", Aw_ex, "mm²", ref=168.3235 if tb_w else None, ref_label="정확 FL08 전류: 168.3235 mm² (정오표 E-002)", tol=1e-6,
                   basis="(N_pI_p/J + N_sI_s/J)/k_u", note=f"반올림 차이 {Aw_tb - Aw_ex:+.4f} mm² ({100 * (Aw_tb / Aw_ex - 1):+.4f} %) — 공학적으로 무의미하지만 교재 수치의 출처")
    res.add_metric("Aw_ref", "기준 후보 window (선택한 전류 기준)", refp["Aw"], "mm²", basis="최악 corner 전류" if worst else "공칭 전류")
    res.add_metric("Aw_eff", "사용 가능한 window (절연 제외)", Aw_eff, "mm²", basis=f"A_w {v['Aw']:g} mm² × (1 − {v['ins_frac']:g} %)",
                   note="절연 비율 0 = 미반영 (MISSING_INPUT)" if v["ins_frac"] == 0 else "절연 비율은 가정 값")
    res.add_metric("B_ref", "기준 후보 B_pk (최대 권선 전압 기준)", refp["B"], "T", basis=f"V_w,max = max(V₁,max, n·V_L,max) = {vw_max(n_t):g} V, {fs / 1e3:g} kHz")
    res.add_metric("B_nom", "공칭 운전의 B_pk (직렬 L이 1차 쪽: 권선 전압 n·V_L)", n_t * VL / (4 * fs * Np0 * Ae), "T", basis="FL07: 코어는 권선 전압을 적분")
    Np_min = vw_max(n_t) / (4.0 * fs * Ae * v["B_allow"])
    res.add_metric("Np_min", "B 조건의 최소 N_p", Np_min, "", basis=f"V_w,max/(4 f A_e B_allow), B_allow {v['B_allow']:g} T (가정)")
    res.add_metric("Pcu_ref", "기준 후보 DC 구리 손실", refp["Pcu"], "W", basis=f"ρ({Tw:g} °C)·MLT·J·(N_pI_p + N_sI_s), MLT {MLT * 1e3:g} mm (가정)", note="AC 배율은 실험 3 (고조파별 R_ac)")
    res.add_metric("n_pass", "screen 통과 후보 수", len(passing), "", basis=f"후보 {len(pts)}개 중 (B·window·n 허용오차·운전점)")
    # ---- independent checks ---------------------------------------------------------------
    T = 1.0 / fs
    if _close(n_t * VL, V1, 1e-12):
        rr = ref.dab_sps_matched(V1, L, fs, P)
        res.add_check(check_close("I_p: 정확 PWL 적분 vs FL08 폐형식", refp["Ip"], rr["I_rms"], 1e-12, "구간 끝점 대수 ∫i²dt vs I_pk√(1 − 2φ/3π) (정합 n)", True, "A"))
    alt = max((p for p in pts if p["Ns"] == Ns0 and abs(p["Np"] - Np0) == 1 and math.isfinite(p["phi"])), key=lambda p: abs(p["dn"]), default=None)
    q0 = (1, -1)
    sims = {}
    for tag, p in (("ref", refp), ("alt", alt)):
        if p is None or not math.isfinite(p["phi"]):
            continue
        sysm = DABT(V1, p["n"] * VL, L, 0.0, 1e9, 0.0, 0.0, fs, p["phi"], p["Np"], Ae)
        x0, _, _ = dab_steady(sysm)
        tr = simulate(sysm, q0, x0, 0.0, T)
        sims[tag] = (sysm, tr)
        e = abs(tr.rms(0.0, T, "i1") - p["Ip"]) / p["Ip"]
        res.add_check(Check(f"I_p ({p['Np']}:{p['Ns']}): 정확 스위칭 해 vs PWL 대수", "PASS" if e < 1e-9 else "FAIL", e, "rel", 1e-9,
                            path="행렬지수 전파(엔진, L_m = 1 GH 한계) vs 교재 4구간 끝점 식으로 만든 PWL", independent=True,
                            detail=f"엔진 {tr.rms(0.0, T, 'i1'):.9g} A / PWL {p['Ip']:.9g} A, φ = {p['phi']:.6f} rad"))
    # B from the actual winding voltage with a finite L_m (k = 1: the core sees n V_L)
    sysB = DABT(V1, n_t * VL, L, 0.0, Lm0, 0.0, 0.0, fs, refp["phi"], Np0, Ae) if math.isfinite(refp["phi"]) else None
    if sysB is not None:
        xb, _, _ = dab_steady(sysB)
        trB = simulate(sysB, q0, xb, 0.0, T)
        blo, bhi = trB.extrema(0.0, T, "B")
        Bsim = 0.5 * (bhi - blo)
        res.add_check(check_close("B_pk: 권선 전압 적분(정확 해) vs n·V_L/(4 f N_p A_e)", Bsim, n_t * VL / (4 * fs * Np0 * Ae), 1e-9,
                                  "자화전류 상태 L_m·i_m/(N A_e)의 peak-to-peak/2 vs 사각파 폐형식 (R = 0, k = 1)", True, "T"))
        led = energy_ledger(trB, sysB, 0.0, T, ["p1"], ["p2"], ["p_loss"], rated_power=P)
        res.add_check(ledger_check(led, what="기준 후보 T-모델 (L_m 유한) "))
        Is_m = n_t * trB.rms(0.0, T, "i2")
        res.add_metric("Is_Lm", "자화전류를 넣은 2차 RMS (L_m 유한, 정확 해)", Is_m, "A", basis=f"L_m {Lm0 * 1e3:g} mH (가정), 직렬 L이 1차 쪽 (k = 1)",
                       note=f"window 계산의 I_s = n·I_p 대비 {100 * (Is_m / (n_t * refp['Ip']) - 1):+.2f} % — 1차 전류는 v₁ − v₂′로 정해져 그대로이고 자화전류는 2차 권선 전류를 바꾼다")
    # ---- plots ----------------------------------------------------------------------------
    g = "dab"
    series_keys = []
    bands: list[dict] = []
    for tag, dNp, key in (("m", -1, "ip_m"), ("0", 0, "ip"), ("p", 1, "ip_p")):
        p = next((q for q in pts if q["Np"] == Np0 + dNp and q["Ns"] == Ns0), None)
        if p is None or not math.isfinite(p["phi"]):
            continue
        sysm = DABT(V1, p["n"] * VL, L, 0.0, 1e9, 0.0, 0.0, fs, p["phi"], p["Np"], Ae)
        x0, _, _ = dab_steady(sysm)
        trp = simulate(sysm, q0, x0, 0.0, 2 * T)
        smp = trp.sample(["i1"], 0.0, 2 * T, per_segment=24)
        res.add_series(key, f"i_p {p['Np']}:{p['Ns']} (n = {p['n']:.4g})", "A", smp["t"], smp["i1"], dash=(dNp != 0))
        series_keys.append(key)
        if dNp == 0:
            bands = bands_from_traj(trp, 0.0, 2 * T, DAB_LABELS)
            smp2 = trp.sample(["v1", "v2"], 0.0, 2 * T, per_segment=4)
            res.add_series("v1", "v₁", "V", smp2["t"], smp2["v1"])
            res.add_series("v2", f"v₂′ ({Np0}:{Ns0})", "V", smp2["t"], smp2["v2"], dash=True)
    res.add_plot("p_ip", f"같은 N_s = {Ns0}에서 N_p ±1: n이 바뀌면 운전 전류가 바뀐다", series_keys, y_label="1차 전류", y_unit="A", bands=bands, group=g, level="C",
                 proved="N_p를 한 turn 바꾸면 n이 약 2 % 바뀌고 V₂′ = n·V_L이 V₁과 어긋나 전류 모양(순환 성분)과 RMS가 달라진다. 정확 스위칭 해로 그렸다.",
                 not_yet="L_m → ∞·이상 bridge·무손실. ZVS·dead time·다른 전압 corner의 전류 차이는 표의 최악 corner 열로만 본다.")
    res.add_plot("p_v", "bridge 전압 (기준 후보)", ["v1", "v2"], y_label="전압", y_unit="V", bands=bands, group=g, level="C",
                 proved="구간 ①~④는 두 bridge 전압의 부호 조합이다. 회로도의 구간 설명과 연결된다.", not_yet="이상 bridge (dead time 없음).")
    Nlo = max(1.0, min(p["Np"] for p in pts) * 0.8)
    Nhi = max(p["Np"] for p in pts) * 1.1
    Ng = np.geomspace(Nlo, Nhi, 160)
    res.add_series("B_line", "B_pk(N_p) = V_w,max/(4 f A_e N_p)", "T", Ng.tolist(), (vw_max(n_t) / (4 * fs * Ae * Ng)).tolist())
    cats = [_Cat.PASS, _Cat.B, _Cat.W, _Cat.N, _Cat.OP]
    colors = {_Cat.PASS: "#16a34a", _Cat.B: "#dc2626", _Cat.W: "#f59e0b", _Cat.N: "#6366f1", _Cat.OP: "#6b7280"}
    for i, cat in enumerate(cats):
        sel = [p for p in pts if p["cat"] == cat]
        if not sel:
            continue
        res.add_series(f"B_{i}", cat, "T", [p["Np"] for p in sel], [p["B"] for p in sel], style="points", color=colors[cat])
        res.add_series(f"W_{i}", cat, "mm²", [p["Np"] for p in sel], [p["Aw"] for p in sel], style="points", color=colors[cat])
        res.add_series(f"M_{i}", cat, "", [p["Np"] for p in sel], [p["n"] for p in sel], style="points", color=colors[cat])
    present = [i for i, cat in enumerate(cats) if any(p["cat"] == cat for p in pts)]
    res.add_plot("p_B", "B 조건: 권선이 적으면 B가 커진다", ["B_line"] + [f"B_{i}" for i in present], x_label="N_p", x_unit="", y_label="B_pk", y_unit="T", kind="xy", log_x=True, level="A",
                 hlines=[{"y": v["B_allow"], "label": f"B_allow {v['B_allow']:g} T (가정)"}], vlines=[{"x": Np_min, "label": f"N_p,min {Np_min:.1f}"}],
                 proved="B_pk ∝ 1/(N_p A_e)이므로 N_p에 하한이 있다. 색은 각 후보가 처음 걸린 조건이다.",
                 not_yet="B_allow는 재료·온도·DC bias 자료가 없는 가정이다. 포화 여유는 재료 곡선으로 확정해야 한다.")
    Aw_line = [2.0 * N * refp["Ip"] / (J * ku) for N in Ng]
    res.add_series("W_line", "연속 근사 2N_pI_p/(J k_u)", "mm²", Ng.tolist(), Aw_line, dash=True)
    wl = [{"y": Aw_eff, "label": f"사용 가능 {Aw_eff:.0f} mm²"}]
    if v["ins_frac"] > 0:
        wl.append({"y": v["Aw"], "label": f"코어 window {v['Aw']:g} mm²"})
    res.add_plot("p_W", "window 조건: 권선이 많으면 창이 모자란다", ["W_line"] + [f"W_{i}" for i in present], x_label="N_p", x_unit="", y_label="A_w,req", y_unit="mm²", kind="xy", log_x=True, level="A",
                 hlines=wl, proved="같은 J·k_u에서 필요한 창은 암페어-턴(N_pI_p + N_sI_s)에 비례한다 — N_p에 상한이 생긴다.",
                 not_yet="bare copper 하한이다. 절연·보빈·termination·litz 충진율·온도별 J는 반영되지 않았다(절연 비율만 가정으로 넣을 수 있다).")
    xr = [float(Ng[0]), float(Ng[-1])]
    res.add_series("n_line", f"연속 n {n_t:.4g}", "", xr, [n_t, n_t], dash=True, color="#6b7280")
    res.add_series("n_hi", f"n +{v['n_tol']:g} %", "", xr, [n_t * (1 + n_tol)] * 2, dash=True, color="#6366f1")
    res.add_series("n_lo", f"n −{v['n_tol']:g} %", "", xr, [n_t * (1 - n_tol)] * 2, dash=True, color="#6366f1")
    res.add_plot("p_map", "정수 후보 지도: n과 N_p를 함께 본다", [f"M_{i}" for i in present] + ["n_line", "n_hi", "n_lo"], x_label="N_p", x_unit="", y_label="n = N_p/N_s", y_unit="", kind="xy", log_x=True, level="A",
                 proved="n이 연속값에 가까운 후보도 권선수(N_p)가 B·window 창 밖이면 탈락한다. n만 맞추는 설계는 끝이 아니다.",
                 not_yet="n 허용오차는 운전점·ZVS 지도에서 받아야 할 가정 값이다.")
    res.circuit = {"diagram": dab_equiv_circuit(V1, n_t * VL, L).to_json(), "intervals": bands, "plot_group": g}
    # ---- tables ---------------------------------------------------------------------------
    res.tables.append(Table("t_cand", "정수 권선 후보 screen (거친 하한 — 최종 설계 아님)",
                            ["후보 N_p:N_s", "n", "Δn [%]", "φ [rad] (공칭)", "I_p RMS 공칭/최악 [A]", "I_s RMS [A]", "B_pk [T]", "A_w,req [mm²]", "P_cu,DC [W]", "L_m (같은 A_L) [mH]", "I_m,pk [A]", "판정"],
                            rows, note=(f"B는 V_w,max = max(V₁,max, n·V_L,max), window·구리는 {'최악 corner' if worst else '공칭'} 전류(L_m → ∞). 최악 corner는 교재 FL08의 800/48, 550/36, 900/36 V. "
                                        "L_m은 같은 코어·gap(A_L 일정)이면 N_p²에 비례한다고 본 값이고, 누설·capacitance는 권선 배치에 따라 다시 측정해야 한다.")))
    res.tables.append(Table("t_missing", "최종 선택을 막는 입력 (MISSING_INPUT / ASSUMED)", ["입력", "현재 상태", "막히는 결정", "받아야 할 것"], [
        ["절연: creepage·clearance·margin tape·층간 절연", "MISSING_INPUT" if v["ins_frac"] == 0 else f"ASSUMED (window의 {v['ins_frac']:g} %)", "실제 사용 가능한 window, 권선 폭", "working voltage, pollution degree, 재료군, 강화/기초 절연, 고도"],
        ["열: R_th·냉각·허용 온도상승", "MISSING_INPUT", "허용 손실 → J·권선 굵기", "냉각 방식, 주위/냉각수 온도, hotspot 위치, 절연 등급"],
        ["core loss (재료 곡선)", "MISSING_INPUT", "총 손실·열 판정, f 선택", "재료 손실 곡선 (f·B·온도·파형), B 정의, DC bias"],
        ["B_allow (포화·온도)", "ASSUMED", "N_p 하한", "재료 B_sat vs 온도, DC bias 여유"],
        ["J·k_u", "ASSUMED (교재 합성 screen 값)", "window 하한", "권선 방식(litz·foil), 충진율, 온도별 허용 J"],
        ["MLT·코어 치수 (A_e, A_w)", "ASSUMED" if not tb else "A_e TEXTBOOK, A_w·MLT ASSUMED", "구리 손실, window", "코어 형번·보빈 도면"],
        ["누설·권선 capacitance", "MISSING_INPUT", "직렬 L 배분, 공진·EMI", "권선 배치별 측정 (실험 5 계획)"],
    ], note="이 입력들이 없으면 최종 transformer 선택 대신 가능한 영역과 필요한 데이터만 보고한다 (교재 E04 산출물)."))
    # ---- verdicts -------------------------------------------------------------------------
    if passing:
        res.verdict("SCREEN_ONLY", f"B·window·n·운전점 screen 통과 후보 {len(passing)}개: " + ", ".join(f"{p['Np']}:{p['Ns']}" for p in passing) + " — bare copper·B 하한일 뿐 최종 설계 아님")
    else:
        res.verdict("FAIL_CONSTRAINT", "screen을 모두 통과하는 정수 권선 후보가 없다 (B 하한과 window 상한이 겹치지 않음) — 코어·J·절연 가정을 다시 봐야 한다")
    res.verdict("MISSING_INPUT", "절연(creepage·clearance)·열(R_th·냉각)·core loss(재료)·측정 누설/C 자료가 없어 최종 선택은 할 수 없다")
    res.assumptions += [
        "window = bare copper 하한 (N_pI_p/J + N_sI_s/J)/k_u — 교재 E04 식",
        "B screen은 최대 권선 전압 max(V₁,max, n·V_L,max)의 사각파: 직렬 L 배치를 확정하기 전의 보수적 값 (FL07)",
        "운전 전류는 이상 SPS (L_m → ∞, 무손실) — 교재 window 예와 같은 정의",
        f"합성 코어: A_e {v['Ae']:g} mm², A_w {v['Aw']:g} mm², MLT {MLT * 1e3:g} mm, B_allow {v['B_allow']:g} T, 권선 온도 {Tw:g} °C",
        "L_m ∝ N_p² (같은 코어·gap)",
    ]
    res.not_valid_for += ["최종 transformer 선정", "AC 구리 손실 (실험 3)", "core loss (MISSING_INPUT)", "절연 거리·내전압 판정", "열 판정"]
    op = refp
    res.interpretation = (
        f"연속 n = {n_t:.4g}을 정수 권선으로 옮기면 n·B·window·구리·운전 전류가 함께 바뀐다. N_p가 {Np_min:.1f}보다 적으면 B가 {v['B_allow']:g} T를 넘고, "
        f"N_p가 커지면 암페어-턴에 비례해 창이 모자란다. 기준 {Np0}:{Ns0}은 B {op['B']:.3f} T, window {op['Aw']:.1f} mm²(사용 가능 {Aw_eff:.0f} mm²)다. "
        f"교재 168.329 mm²는 반올림 전류(2.02/33.665 A)의 값이고 정확 전류로는 {Aw_ex:.4f} mm²다. 같은 N_s에서 N_p를 한 turn 바꾸면 n이 {100 * (1 / (Np0) ):.1f} % 가량 바뀌어 전류 모양도 바뀐다. "
        "절연·열·재료 손실 자료가 없으므로 결론은 '가능한 영역'까지다."
    )
    return res


# ======================================================================================
# Experiment 2: CLLC n = 0.93 -> integer turns (FHA operating map)
# ======================================================================================


def coordinated_vi(Vo: np.ndarray) -> np.ndarray:
    """Link voltage coordinated with the battery through the three textbook points (piecewise linear, ASSUMED between them)."""
    xs = [c[0] for c in CLLC_CORNERS]
    ys = [c[1] for c in CLLC_CORNERS]
    return np.interp(Vo, xs, ys)


def run_cllc_turns(v: dict) -> Result:
    res = Result("EX04", "cllc_turns", "A (FHA 운전점 + screen)")
    n_t = v["n_t"]
    Np0, Ns0 = int(v["Np_ref"]), int(v["Ns_ref"])
    Lr, Cr, Lm, P = v["Lr"], v["Cr"], v["Lm"], v["P"]
    fmin, fmax = v["f_min"], v["f_max"]
    tank2 = v["tank2"]
    J, ku, Ae, MLT, Tw = v["J"], v["ku"], v["Ae"] * 1e-6, v["MLT"], v["T_w"]
    Aw_eff = v["Aw"] * (1.0 - v["ins_frac"] / 100.0)
    Vo_max = max(c[0] for c in CLLC_CORNERS)
    Vi_max = max(c[1] for c in CLLC_CORNERS)
    fr = 1.0 / (TWO_PI * math.sqrt(Lr * Cr))
    tb = _close(Lr, CLLC_LR) and _close(Cr, CLLC_CR, 1e-12) and _close(Lm, CLLC_LM) and _close(P, CLLC_P) and _close(fmin, 120e3) and _close(fmax, 210e3)
    # ---- continuous design (textbook FL10 numbers) ------------------------------------------
    tank_t = cllc_tank(n_t, n_t, Lr, Cr, Lm, tank2)
    tb_n = tb and _close(n_t, CLLC_N)
    cont = {}
    for Vo, Vi in CLLC_CORNERS:
        roots = [r for r, ind in fha_roots_brentq(n_t, Vo, Vi, P, tank_t, fmin, fmax) if ind]
        cont[(Vo, Vi)] = roots
    up_t, lo_t, _, _ = fha_solve([n_t] * 3, [c[0] for c in CLLC_CORNERS], [c[1] for c in CLLC_CORNERS], P, *tank_t, fmin, fmax)
    r650, r800, r920 = cont[CLLC_CORNERS[0]], cont[CLLC_CORNERS[1]], cont[CLLC_CORNERS[2]]
    res.add_metric("fr", "공진 주파수 f_r = 1/(2π√(L_rC_r))", fr, "Hz", ref=150e3 if tb else None, ref_label="교재 150 kHz", tol=1e-6)
    if r650:
        res.add_metric("f650", f"연속 n = {n_t:g}: 650/700 V inductive 해", max(r650), "Hz", ref=164.390e3 if tb_n else None, ref_label="교재 FL10 164.390 kHz", tol=5e-6, basis="FHA (A 수준)")
    if r800:
        res.add_metric("f800", f"연속 n = {n_t:g}: 800/800 V inductive 해", max(r800), "Hz", ref=162.811e3 if tb_n else None, ref_label="교재 FL10 162.811 kHz", tol=5e-6, basis="FHA (A 수준)")
    if len(r920) >= 2:
        res.add_metric("f920_lo", "연속 n: 920/850 V 낮은 해", min(r920), "Hz", ref=136.099e3 if tb_n else None, ref_label="교재 136.099 kHz", tol=5e-6)
        res.add_metric("f920_hi", "연속 n: 920/850 V 높은 해", max(r920), "Hz", ref=147.061e3 if tb_n else None, ref_label="교재 147.061 kHz", tol=5e-6)
        I_lo = fha_currents(min(r920), n_t, 920.0, 850.0, P, tank_t)[0]
        I_hi = fha_currents(max(r920), n_t, 920.0, 850.0, P, tank_t)[0]
        res.add_metric("I920_lo", "920/850 V 낮은 해의 1차 직렬 RMS (FHA)", I_lo, "A", ref=14.390 if tb_n else None, ref_label="교재 14.390 A", tol=5e-5)
        res.add_metric("I920_hi", "920/850 V 높은 해의 1차 직렬 RMS (FHA)", I_hi, "A", ref=14.765 if tb_n else None, ref_label="교재 14.765 A", tol=5e-5)
        s_lo = gain_slope(min(r920), n_t, 920.0, P, tank_t)
        s_hi = gain_slope(max(r920), n_t, 920.0, P, tank_t)
        res.add_metric("slope_lo", "낮은 해의 gain 기울기 (±10 Hz)", s_lo, "1/kHz", ref=0.001871 if tb_n else None, ref_label="교재 +0.001871/kHz", tol=1e-3, note="같은 출력인데 주파수 변화의 부호가 반대")
        res.add_metric("slope_hi", "높은 해의 gain 기울기 (±10 Hz)", s_hi, "1/kHz", ref=-0.001803 if tb_n else None, ref_label="교재 −0.001803/kHz", tol=1e-3)
    # seed failure (n = 1, textbook initial design) kept as a result
    g_seed, _ = max_ind_gain(1.0, 920.0, P, (Lr, Cr, Lm, Lr, Cr), fmin, fmax)
    res.add_metric("seed_gain", "초기안 n = 1: 920/850 V inductive 최대 이득", g_seed, "", ref=1.016401 if tb else None, ref_label="교재 1.016401 (필요 1.082353 → 해 없음)", tol=2e-6,
                   note=f"필요 이득 {920 / 850:.6f} — 초기안 FAIL 보존")
    # ---- checks on the continuous design ------------------------------------------------------
    errs = []
    for k, (Vo, Vi) in enumerate(CLLC_CORNERS):
        rb = cont[(Vo, Vi)]
        if rb:
            errs.append(abs(up_t[k] - max(rb)) / max(rb))
        if len(rb) >= 2:
            errs.append(abs(lo_t[k] - min(rb)) / min(rb))
    if errs:
        res.add_check(Check("FHA 해: 벡터화 이분법 vs brentq", "PASS" if max(errs) < 1e-9 else "FAIL", max(errs), "rel", 1e-9,
                            path="격자 부호변화 + 전 후보 동시 이분법 vs 조밀 격자 + scipy brentq (서로 다른 root solver)", independent=True, detail=f"해 {len(errs)}개"))
    pe = []
    for (Vo, Vi), roots in cont.items():
        for r in roots:
            I1, I2p, _, Rac, Zin = fha_currents(r, n_t, Vo, Vi, P, tank_t)
            pe.append(max(abs(I1 * I1 * Zin.real - P), abs(I2p * I2p * Rac - P)) / P)
    if pe:
        res.add_check(Check("FHA 전력 항등식: |I₁|²Re Z_in = |I₂′|²R_ac′ = P", "PASS" if max(pe) < 1e-8 else "FAIL", max(pe), "rel", 1e-8,
                            path="해에서 입력 임피던스로 계산한 입력 전력과 전류분배로 계산한 R_ac′ 전력 vs 정격 P (무손실 tank)", independent=True))
    tk = (Lr, Cr, Lm, Lr, Cr)
    hr = [abs(complex(fha(fr, nn, Vo, P, *tk)[0])) for nn in (0.9, n_t, 1.0) for Vo in (650.0, 920.0)]
    e = max(abs(x - 1.0) for x in hr)
    res.add_check(Check("대칭 tank의 부하 무관점 |H(f_r)| = 1", "PASS" if e < 1e-12 else "FAIL", e, "", 1e-12,
                        path="환산 대칭(L_r2′ = L_r1, C_r2′ = C_r1)이면 f_r에서 Z_r = 0 → H = 1 (n·부하와 무관) — FHA 구현의 물리 항등식", independent=True))
    # ---- integer candidates ------------------------------------------------------------------
    cands = turn_candidates(n_t, int(v["Ns_max"]), extra=[(Np0, Ns0)])
    cands = [c for c in cands if abs(c[0] / c[1] / n_t - 1.0) <= v["n_list"] / 100.0 + 1e-12 or c == (Np0, Ns0)]
    pts, rows = [], []
    Vo_c = np.array([c[0] for c in CLLC_CORNERS])
    Vi_c = np.array([c[1] for c in CLLC_CORNERS])
    ns = np.array([Np / Ns for Np, Ns in cands])
    tanks = [cllc_tank(n, n_t, Lr, Cr, Lm, tank2) for n in ns]
    # all candidates x corners in one vectorized solve
    nn = np.repeat(ns, 3)
    VO = np.tile(Vo_c, len(ns))
    VI = np.tile(Vi_c, len(ns))
    tk = np.repeat(np.array(tanks), 3, axis=0)
    up, lo, cnt, _ = fha_solve(nn, VO, VI, P, tk[:, 0], tk[:, 1], tk[:, 2], tk[:, 3], tk[:, 4], fmin, fmax)
    for i, (Np, Ns) in enumerate(cands):
        n = ns[i]
        u, lw, ct = up[3 * i:3 * i + 3], lo[3 * i:3 * i + 3], cnt[3 * i:3 * i + 3]
        op_ok = bool(np.all(ct > 0))
        I1s, I2s = [], []
        for k, (Vo, Vi) in enumerate(CLLC_CORNERS):
            if ct[k] > 0:
                a, _, c2, _, _ = fha_currents(u[k], n, Vo, Vi, P, tanks[i])
                I1s.append(a)
                I2s.append(c2)
        I1m = max(I1s) if I1s else float("nan")
        I2m = max(I2s) if I2s else float("nan")
        Vw = max(Vi_max, n * Vo_max)
        B = Vw / (4.0 * fmin * Np * Ae)
        Aw = window_req(Np, I1m, Ns, I2m, J, ku) if op_ok else float("nan")
        Pcu = copper_dc(Np, I1m, Ns, I2m, J, MLT, Tw) if op_ok else float("nan")
        AL = Lm / Np**2
        fails = []
        if not op_ok:
            fails.append(_Cat.OP)
        if B > v["B_allow"]:
            fails.append(_Cat.B)
        if op_ok and not (Aw <= Aw_eff):
            fails.append(_Cat.W)
        cat = _Cat.PASS if not fails else fails[0]
        Lr2 = tanks[i][3] / (n * n)
        Cr2 = tanks[i][4] * n * n
        f920 = " / ".join(f"{x / 1e3:.3f}" for x in sorted([x for x in (lw[2], u[2]) if math.isfinite(x)])) or "해 없음"
        pts.append({"Np": Np, "Ns": Ns, "n": n, "up": u, "lo": lw, "cnt": ct, "B": B, "Aw": Aw, "Pcu": Pcu, "I1": I1m, "I2": I2m, "fails": fails, "cat": cat, "AL": AL})
        fop = " · ".join(f"{x / 1e3:.3f}" if math.isfinite(x) else "해 없음" for x in (u[0], u[1]))
        tank_s = f"{tanks[i][3] * 1e6:.3f} / {tanks[i][4] * 1e9:.3f}" if tank2 == "keep" else f"{Lr2 * 1e6:.3f} / {Cr2 * 1e9:.3f}"
        rows.append([f"{Np}:{Ns}", f"{n:.5f} ({100 * (n / n_t - 1):+.2f} %)", fop, f920, f"{I1m:.2f} / {I2m:.2f}" if math.isfinite(I1m) else "—", round(B, 4),
                     round(Aw, 1) if math.isfinite(Aw) else "—", round(Pcu, 2) if math.isfinite(Pcu) else "—", round(AL * 1e9, 1), tank_s, "screen 통과 (FHA 후보)" if not fails else ", ".join(fails)])
    refp = next(p for p in pts if p["Np"] == Np0 and p["Ns"] == Ns0)
    passing = [p for p in pts if not p["fails"]]
    # ---- continuous n window allowed by the operating map ----------------------------------------
    def g_hi(nv):
        tkn = cllc_tank(nv, n_t, Lr, Cr, Lm, tank2)
        gm = max_ind_gain(nv, 920.0, P, tkn, fmin, fmax)[0]
        return (gm if math.isfinite(gm) else 0.0) - nv * 920.0 / 850.0

    n_max = float("nan")
    a_, b_ = n_t * 0.97, n_t * 1.05
    if g_hi(a_) > 0 > g_hi(b_):
        n_max = brentq(g_hi, a_, b_, xtol=1e-9)

    def g_lo(nv):
        tkn = cllc_tank(nv, n_t, Lr, Cr, Lm, tank2)
        return abs(complex(fha(fmax, nv, 650.0, P, *tkn)[0])) - nv * 650.0 / 700.0

    n_min = float("nan")
    if g_lo(n_t) < 0 < g_lo(0.5 * n_t):
        n_min = brentq(g_lo, 0.5 * n_t, n_t, xtol=1e-9)
    if math.isfinite(n_max):
        res.add_metric("n_max", "n 상한 (920/850 V corner의 FHA 해가 남는 최대 n)", n_max, "", basis="inductive 최대 이득 = n·920/850",
                       note="이보다 크면 NO_SOLUTION" + ("" if math.isfinite(n_min) else f"; 하한은 [{0.5 * n_t:.3g}, {n_t:g}]에서 f_max 경계가 나타나지 않음"))
    if math.isfinite(n_min):
        res.add_metric("n_min", "n 하한 (650/700 V 해가 f_max 안에 남는 최소 n)", n_min, "", basis=f"|H(f_max)| = n·650/700, f_max {fmax / 1e3:g} kHz")
    req_ref = refp["n"] * 920.0 / 850.0
    res.add_metric("req920", f"기준 후보 {Np0}:{Ns0}: 920/850 V 필요 이득 n·V_o/V_i", req_ref, "")
    f800r = refp["up"][1]
    res.add_metric("ref_f800", f"기준 후보 {Np0}:{Ns0}: 800/800 V 운전 주파수", f800r, "Hz", note=f"연속 n 대비 {(f800r - (max(r800) if r800 else float('nan'))) / 1e3:+.3f} kHz" if math.isfinite(f800r) else "해 없음")
    tb_keep = tb and _close(n_t, CLLC_N) and tank2 == "keep"
    tk_ref = tanks[cands.index((Np0, Ns0))]
    res.add_metric("Lr2", "2차 물리 L_r2 (기준 후보)", tk_ref[3] / refp["n"] ** 2, "H", ref=46.2481e-6 if tb_keep else None, ref_label="교재 46.2481 µH (n = 0.93 설계)", tol=2e-6,
                   basis="유지(keep): n = 0.93 설계값 그대로" if tank2 == "keep" else "재선정(retune): L_r/n² (후보 n)")
    res.add_metric("Cr2", "2차 물리 C_r2 (기준 후보)", tk_ref[4] * refp["n"] ** 2, "F", ref=24.3424e-9 if tb_keep else None, ref_label="교재 24.3424 nF (n = 0.93 설계)", tol=2e-6,
                   note=f"환산 L_r2′ {tk_ref[3] * 1e6:.4f} µH, C_r2′ {tk_ref[4] * 1e9:.4f} nF (1차 tank {Lr * 1e6:g} µH, {Cr * 1e9:.4f} nF)")
    res.add_metric("Np_min", "B 조건 최소 N_p", max(Vi_max, n_t * Vo_max) / (4 * fmin * Ae * v["B_allow"]), "", basis=f"f_min {fmin / 1e3:g} kHz, A_e {v['Ae']:g} mm², B_allow {v['B_allow']:g} T (가정)")
    res.add_metric("n_pass", "screen 통과 후보 수", len(passing), "", basis=f"후보 {len(pts)}개 (|Δn| ≤ {v['n_list']:g} %)")
    # ---- plots --------------------------------------------------------------------------------
    if len(r920) >= 1:
        fa, fb = max(fmin, min(r920) - 8e3), min(fmax, max(r920) + 15e3)
    else:
        fa, fb = fmin, fmax
    F = np.linspace(fa, fb, 601)
    show = []
    for Np, Ns in ((Np0 - 1, Ns0 - 1), (Np0, Ns0), (Np0 + 1, Ns0 + 1), (Np0 + 2, Ns0 + 2)):
        if Np < 1 or Ns < 1:
            continue
        nv = Np / Ns
        tkn = cllc_tank(nv, n_t, Lr, Cr, Lm, tank2)
        H, Zin, _, _ = fha(F, nv, 920.0, P, *tkn)
        Mn = np.abs(H) * 850.0 / (nv * 920.0)
        key = f"g_{Np}_{Ns}"
        res.add_series(key, f"{Np}:{Ns} (n = {nv:.4f})", "", (F / 1e3).tolist(), np.where(Zin.imag > 0, Mn, np.nan).tolist(), dash=(Np, Ns) != (Np0, Ns0))
        show.append(key)
    H, Zin, _, _ = fha(F, n_t, 920.0, P, *tank_t)
    res.add_series("g_cont", f"연속 n = {n_t:g}", "", (F / 1e3).tolist(), np.where(Zin.imag > 0, np.abs(H) * 850.0 / (n_t * 920.0), np.nan).tolist(), dash=True, color="#6b7280")
    mk = [{"x": x / 1e3, "y": 1.0, "label": f"{x / 1e3:.2f}"} for x in r920]
    res.add_plot("p_gain", "920/850 V corner: 정규화 이득 |H|·V_i/(n·V_o) — 1과 만나는 곳이 해", show + ["g_cont"], x_label="f", x_unit="kHz", y_label="정규화 이득", y_unit="", kind="xy", level="A",
                 hlines=[{"y": 1.0, "label": "필요 이득"}], markers=mk,
                 proved="inductive 영역만 그렸다. n이 커지면 필요 이득이 올라 곡선의 봉우리가 1 아래로 내려가고 해가 사라진다. n이 작으면 한 해만 남는다(분기 구조 변화).",
                 not_yet="FHA(A 수준)의 정상 이득일 뿐이다. ZVS·SR·기동·제어 기울기·스위칭 해는 EX05에서 검증한다.")
    cats = [_Cat.PASS, _Cat.OP, _Cat.B, _Cat.W]
    colors = {_Cat.PASS: "#16a34a", _Cat.B: "#dc2626", _Cat.W: "#f59e0b", _Cat.OP: "#6b7280"}
    present = []
    for i, cat in enumerate(cats):
        sel = [p for p in pts if p["cat"] == cat]
        if not sel:
            continue
        present.append(i)
        res.add_series(f"M_{i}", cat, "", [p["Np"] for p in sel], [p["n"] for p in sel], style="points", color=colors[cat])
    Nps = [p["Np"] for p in pts]
    xr = [max(1.0, min(Nps) - 2.0), max(Nps) + 2.0]
    res.add_series("n_line", f"연속 n {n_t:g}", "", xr, [n_t, n_t], dash=True, color="#6b7280")
    keys = ["n_line"]
    if math.isfinite(n_max):
        res.add_series("nmax_line", f"n 상한 {n_max:.4f} (920/850 V 해)", "", xr, [n_max, n_max], dash=True, color="#dc2626")
        keys.append("nmax_line")
    if math.isfinite(n_min):
        res.add_series("nmin_line", f"n 하한 {n_min:.4f}", "", xr, [n_min, n_min], dash=True, color="#6366f1")
        keys.append("nmin_line")
    I1r, I2r = refp["I1"], refp["I2"]
    Np_lo = max(Vi_max, n_t * Vo_max) / (4 * fmin * Ae * v["B_allow"])
    vl = [{"x": Np_lo, "label": f"N_p,min {Np_lo:.1f} (B)"}]
    if math.isfinite(I1r):
        Np_hi = Aw_eff * J * ku / (I1r + I2r / n_t)
        vl.append({"x": Np_hi, "label": f"N_p,max {Np_hi:.1f} (window)"})
    res.add_plot("p_map", "정수 후보 지도: n 창은 운전점이, N_p 창은 B·window가 정한다", [f"M_{i}" for i in present] + keys, x_label="N_p", x_unit="", y_label="n = N_p/N_s", y_unit="", kind="xy", level="A",
                 vlines=vl, proved="n 상한은 920/850 V corner의 이득이, N_p 하한은 B가, N_p 상한은 window가 정한다(세로선은 기준 후보 전류로 본 연속 근사). 세 조건의 교집합만 후보다.",
                 not_yet="n 하한·상한은 FHA와 교재 세 corner 기준이다. 경부하·역방향·기동 corner는 포함하지 않았다.")
    ok_op = [p for p in pts if np.all(p["cnt"] > 0)]
    res.add_series("Bn", "B_pk/B_allow", "", [p["Np"] for p in pts], [p["B"] / v["B_allow"] for p in pts], style="points", color="#2563eb")
    res.add_series("Wn", "A_w,req/A_w,eff (운전점 해가 있는 후보)", "", [p["Np"] for p in ok_op], [p["Aw"] / Aw_eff for p in ok_op], style="points", color="#f59e0b")
    res.add_series("one", "한계 1", "", xr, [1.0, 1.0], dash=True, color="#6b7280")
    res.add_plot("p_norm", "정규화 screen: 1 이하가 통과", ["Bn", "Wn", "one"], x_label="N_p", x_unit="", y_label="비율", y_unit="", kind="xy", level="A",
                 proved="같은 n이라도 N_p를 두 배로 하면(13:14 → 26:28) B는 절반, window는 두 배가 된다. B는 N_p 하한을, window는 상한을 만든다.",
                 not_yet="합성 코어(A_e·A_w·B_allow·MLT 가정)의 screen이다. AC 구리·core loss·열은 빠져 있다.")
    res.tables.append(Table("t_cand", "CLLC 정수 권선 후보: FHA 운전점과 screen", ["후보", "n (Δn)", "f_op 650/700 · 800/800 V [kHz]", "920/850 V 해 [kHz]", "I₁ / I₂ 최대 [A]",
                                                                           "B_pk [T]", "A_w,req [mm²]", "P_cu,DC [W]", "A_L [nH/t²]",
                                                                           "L_r2′ / C_r2′ 환산 [µH / nF]" if tank2 == "keep" else "L_r2 / C_r2 물리 [µH / nF]", "판정"], rows,
                            note=("운전 주파수는 가장 높은 inductive 해(교재의 시작 heuristic — 자동 최적 아님). 920/850 V 열은 모든 inductive 해. I₁·I₂는 세 corner의 FHA RMS 최대(I₂는 실제 2차). "
                                  "A_L = L_m/N_p²: L_m을 유지하려면 후보마다 gap을 다시 잡아야 한다. "
                                  + (f"2차 tank: n = {n_t:g} 설계의 물리 값 L_r2 {Lr / n_t**2 * 1e6:.4f} µH, C_r2 {Cr * n_t**2 * 1e9:.4f} nF 유지 — 환산값이 후보마다 달라진다 (1차 {Lr * 1e6:g} µH / {Cr * 1e9:.4f} nF)."
                                     if tank2 == "keep" else "2차 tank: 후보마다 L_r/n², C_r n²로 다시 골라 환산 대칭 유지 — 다른 부품이다."))))
    res.tables.append(Table("t_missing", "최종 선택을 막는 입력", ["입력", "상태", "막히는 결정"], [
        ["코어 A_e·A_w·MLT, B_allow", "ASSUMED (합성 코어)", "N_p 하한·상한"],
        ["절연 (강화 절연 working voltage ~920 V)", "MISSING_INPUT" if v["ins_frac"] == 0 else "ASSUMED", "사용 가능 window"],
        ["열·냉각, core loss 재료 곡선", "MISSING_INPUT", "허용 손실, J"],
        ["측정 누설·C_ps (공진 L 일부로 쓸 때)", "MISSING_INPUT", "L_r 분배, tank 공차"],
        ["스위칭 검증 (ZVS·SR·기동·역방향)", "미실행 (EX05)", "FHA 후보 → 설계"],
    ]))
    # ---- verdicts -------------------------------------------------------------------------------
    if not np.all(refp["cnt"] > 0):
        miss = [f"{c[0]:g}/{c[1]:g} V" for c, k in zip(CLLC_CORNERS, refp["cnt"]) if k == 0]
        res.verdict("NO_SOLUTION", f"기준 후보 {Np0}:{Ns0} (n = {refp['n']:.4f}): {', '.join(miss)}에서 {fmin / 1e3:g}–{fmax / 1e3:g} kHz inductive FHA 해 없음 (필요 이득 부족)")
    if passing:
        res.verdict("CANDIDATE_FHA_ONLY", f"세 corner FHA 해 + B·window screen 통과 후보: " + ", ".join(f"{p['Np']}:{p['Ns']}" for p in passing) + " — 스위칭 검증 전 후보")
        res.verdict("SCREEN_ONLY", "B·window는 합성 코어의 거친 하한")
    else:
        res.verdict("FAIL_CONSTRAINT", "세 corner 해와 B·window를 함께 만족하는 정수 후보가 없다")
    res.verdict("MISSING_INPUT", "절연·열·core loss·측정 누설 자료 없이 최종 권선 선택 불가")
    res.assumptions += [
        "FHA (기본파 등가, R_ac′ = 8n²V_o²/(π²P)): A 수준 정상 이득",
        "link 전압은 교재 세 점(650/700, 800/800, 920/850 V)에서만 평가 (그 사이 조정은 가정)",
        f"합성 코어: A_e {v['Ae']:g} mm², A_w {v['Aw']:g} mm², MLT {MLT * 1e3:g} mm, B_allow {v['B_allow']:g} T; B는 f_min과 max(V_i, nV_o)의 사각파",
        f"2차 tank: {'n = 0.93 설계의 물리 L_r2 = 46.25 µH, C_r2 = 24.34 nF 유지' if tank2 == 'keep' else '후보마다 L_r/n², C_r n²로 재선정'}",
    ]
    res.not_valid_for += ["스위칭 운전점·ZVS·SR (EX05)", "경부하·역방향·기동", "최종 권선 선정", "core loss·열"]
    res.interpretation = (
        f"n = {n_t:g}은 연속값이다. 정수 후보마다 필요 이득 n·V_o/V_i가 바뀌어 운전 주파수가 이동하고, n이 {n_max:.4f}보다 크면 920/850 V corner의 inductive 이득이 모자라 해가 사라진다. "
        f"기준 {Np0}:{Ns0}은 800/800 V {f800r / 1e3:.3f} kHz로 연속 설계({max(r800) / 1e3 if r800 else float('nan'):.3f} kHz)와 거의 같지만, 같은 n의 26:28은 B가 절반·window가 두 배다. "
        "2차 공진 부품을 그대로 두느냐 다시 고르느냐도 서로 다른 회로다. FHA 해는 스위칭 검증 전의 후보일 뿐이다."
    )
    return res


# ======================================================================================
# Experiment 3: harmonic copper loss
# ======================================================================================


def run_harmonic_copper(v: dict) -> Result:
    res = Result("EX04", "harmonic_copper", "A (정확 Fourier + 합성 R_ac)")
    if v["wave"] == "list":
        I = np.array([v["I1"], v["I2"], v["I3"]])
        R = np.array([v["R1"], v["R2"], v["R3"]])
        Rdc = v["Rdc"]
        P_h = I * I * R
        P_harm = float(P_h.sum())
        Irms = math.sqrt(float((I * I).sum()))
        P_dc = Irms * Irms * Rdc
        tb = np.allclose(I, [10, 3, 2]) and np.allclose(R, [0.02, 0.04, 0.06]) and _close(Rdc, 0.02)
        res.add_metric("P_harm", "고조파별 R_ac 합 Σ I_h²R_ac(hf₀)", P_harm, "W", ref=2.60 if tb else ref.harmonic_copper(I, R), ref_label="교재 2.60 W" if tb else "reference 식", tol=1e-9)
        res.add_metric("P_dc", "전체 RMS × R_dc만", P_dc, "W", ref=2.26 if tb else None, ref_label="교재 2.26 W", tol=1e-9, basis=f"I_rms = √ΣI_h² = {Irms:.4f} A")
        res.add_metric("dP", "R_dc만 쓸 때 빠뜨린 손실", P_harm - P_dc, "W", ref=0.34 if tb else None, ref_label="교재 0.34 W", tol=1e-9)
        res.add_metric("ratio", "P_harm / P_dc", P_harm / P_dc, "", note=f"{100 * (P_harm / P_dc - 1):+.1f} %")
        res.add_metric("Irms", "전체 RMS", Irms, "A", basis="성분 RMS의 제곱합 (위상 무관)")
        # independent: synthesize a waveform with orders 1, 3, 5 and arbitrary phases; time-domain RMS
        t = np.linspace(0.0, 1.0, 20001)
        wave = sum(math.sqrt(2) * Ih * np.sin(TWO_PI * h * t + ph) for Ih, h, ph in zip(I, (1, 3, 5), (0.0, 1.1, -0.7)))
        rms_t = math.sqrt(float(np.trapezoid(wave * wave, t)))
        res.add_check(check_close("Parseval: 시간영역 RMS (합성 파형, 임의 위상) vs √ΣI_h²", rms_t, Irms, 1e-9, "차수 1·3·5와 임의 위상으로 만든 파형의 수치 적분 vs 성분 합", True, "A",
                                  detail="R_dc만의 손실이 성분 위상과 무관함을 확인"))
        res.model_level = "A (성분별 합)"
        xs = [0, 1, 2, 3]
        res.add_series("lp_h", "누적 Σ I_h²R_ac", "W", xs, [0.0] + np.cumsum(P_h).tolist())
        res.add_series("lp_d", "누적 Σ I_h²R_dc (R_dc만)", "W", xs, [0.0] + np.cumsum(I * I * Rdc).tolist(), dash=True)
        res.add_plot("p_list", "성분을 하나씩 더한 누적 손실: R_ac vs R_dc", ["lp_h", "lp_d"], x_label="더한 성분 수 (교재 목록 순서)", x_unit="", y_label="누적 손실", y_unit="W", kind="xy", level="A",
                     hlines=[{"y": P_harm, "label": f"{P_harm:#.3g} W"}, {"y": P_dc, "label": f"{P_dc:#.3g} W"}],
                     proved=f"첫 성분은 R_ac = R_dc라 같고, 고조파 성분을 더할 때마다 차이가 벌어진다: {P_harm:#.3g} W vs {P_dc:#.3g} W.",
                     not_yet="교재 목록은 합성 값이다. 실제 권선은 측정 R_ac(f)·온도·두 권선 전류 비가 필요하다.")
        rows = [[f"성분 {k + 1}", float(I[k]), float(R[k]) * 1e3, float(P_h[k]), float(I[k] ** 2 * Rdc)] for k in range(3)]
        res.tables.append(Table("t_h", "성분별 손실", ["성분", "I_h RMS [A]", "R_ac [mΩ]", "I_h²R_ac [W]", "I_h²R_dc [W]"], rows, note="교재는 차수를 명시하지 않는다 (합에는 필요 없음)."))
        level_verdict = ("PASS_WITHIN_MODEL", f"교재 {P_harm:.2f} W vs R_dc만 {P_dc:.2f} W 재현 (선형 R_ac 범위의 합)")
    else:
        V1, V2, L, fs, P = v["V1"], v["V2"], v["L"], v["fs"], v["P"]
        pt = dab_point(V1, V2, L, fs, P)
        if pt is None:
            res.verdict("NO_SOLUTION", "이 전압·L에서 요구 전력을 SPS로 전달할 수 없다")
            return res
        phi, pw = pt
        Tw = v["T_w"]
        Rdc = v["Rdc20"] * (1.0 + ALPHA_CU * (Tw - 20.0))
        delta1 = math.sqrt(rho_cu(Tw) / (math.pi * fs * MU0))
        Hm = int(v["H_max"])
        c = pwl_fourier(pw, Hm)
        Ih = math.sqrt(2.0) * np.abs(c[1:])
        h = np.arange(1, Hm + 1)
        Delta = v["h_cond"] / delta1
        FR = dowell(Delta * np.sqrt(h), int(v["layers"]))
        P_h = Ih * Ih * Rdc * FR
        P_harm = float(abs(c[0]) ** 2 * Rdc + P_h.sum())
        Irms = pw.rms()
        P_dc = Irms * Irms * Rdc
        tb = _close(V1, 800) and _close(V2, 800) and _close(L, 200e-6) and _close(fs, 1e5) and _close(P, 1500)
        e06 = _close(V1, 900) and _close(V2, 600) and _close(L, 200e-6) and _close(fs, 1e5) and _close(P, 1500)
        res.add_metric("Irms", "전류 RMS (정확 PWL)", Irms, "A", ref=2.019881509 if tb else (3.113563 if e06 else None), ref_label="FL08 2.019881509 A" if tb else "교재 E06 SPS 3.113563 A",
                       tol=1e-6, basis=f"φ = {phi:.6f} rad")
        res.add_metric("I1h", "기본파 RMS I₁", float(Ih[0]), "A")
        res.add_metric("THD", "고조파 RMS / 기본파", math.sqrt(max(Irms**2 - Ih[0] ** 2 - abs(c[0]) ** 2, 0.0)) / Ih[0], "", basis="전류 파형 (RMS 비)")
        res.add_metric("P_dc", "R_dc(T)만: I_rms²R_dc", P_dc, "W", basis=f"R_dc({Tw:g} °C) = {Rdc * 1e3:.4g} mΩ")
        res.add_metric("P_harm", "고조파별 합 Σ I_h²R_dc F_R(h)", P_harm, "W", basis=f"h ≤ {Hm}, 합성 Dowell {int(v['layers'])}층, h_cond {v['h_cond'] * 1e3:g} mm")
        res.add_metric("ratio", "P_harm / P_dc", P_harm / P_dc, "", note=f"{100 * (P_harm / P_dc - 1):+.1f} % — R_dc만 쓰면 이만큼 과소평가 (같은 R_dc 입력 기준: 두께를 바꾸면 R_dc 자체도 바뀐다)")
        res.add_metric("delta", f"skin depth δ ({Tw:g} °C, 기본파)", delta1, "m", basis=f"Δ₁ = h/δ = {Delta:.3f}")
        # checks: Parseval (frequency domain) vs exact time-domain RMS; FFT of samples vs closed form
        ms_f = float(abs(c[0]) ** 2 + (Ih * Ih).sum())
        e_par = abs(ms_f - Irms**2) / Irms**2
        res.add_check(Check(f"Parseval: Σ|I_h|² (h ≤ {Hm}) vs 정확 PWL ∫i²dt", "PASS" if e_par < 1e-6 else "FAIL", e_par, "rel", 1e-6,
                            path="구간별 닫힌 형태 Fourier 계수의 제곱합 vs 시간영역 구간 끝점 대수 (서로 다른 경로)", independent=True,
                            detail=f"꼬리 절단 오차 ~ Σ_{{h>{Hm}}} 1/h⁴ (기울기 불연속 파형)"))
        Nf = 1 << 14
        ts = np.arange(Nf) / Nf / fs
        ys = np.interp(ts, pw.t, np.concatenate([pw.y0, pw.y1[-1:]]))  # continuous waveform: node values
        cf = np.fft.rfft(ys) / Nf
        e_fft = float(np.max(np.abs(cf[1:16] - c[1:16]))) / float(np.abs(c[1]))
        res.add_check(Check("FFT (2¹⁴ 표본) vs 닫힌 형태 계수 (h = 1…15)", "PASS" if e_fft < 1e-6 else "FAIL", e_fft, "rel", 1e-6,
                            path="균일 표본의 이산 Fourier 변환 vs 구간 적분식", independent=True, detail="표본화 aliasing 오차가 남는다 (1/N² 차수)"))
        cj, sum_ds = pwl_fourier_jumps(pw, Hm)
        e_j = float(np.max(np.abs(cj[1:] - c[1:]))) / float(np.abs(c[1]))
        res.add_check(Check(f"기울기 점프 공식 vs 구간 적분식 (h = 1…{Hm})", "PASS" if e_j < 1e-10 else "FAIL", e_j, "rel", 1e-10,
                            path="두 번 부분적분한 c_h = ΣΔs_k e^{−jhωt_k}/(T(jhω)²) vs 구간별 ∫(y₀+sτ)e^{−jhωτ}dτ (서로 다른 유도)", independent=True,
                            detail="연속 PWL이면 두 식이 같아야 한다; 앞의 식은 |I_h| 상한 포락선도 준다"))
        if tb:
            rr = ref.dab_sps_matched(V1, L, fs, P)
            res.add_check(check_close("I_rms: PWL vs FL08 폐형식", Irms, rr["I_rms"], 1e-12, "구간 대수 vs I_pk√(1 − 2φ/3π)", True, "A"))
        res.add_check(check_close("Dowell 저주파 한계 F_R(Δ → 0) → 1", float(dowell(1e-3, int(v["layers"]))), 1.0, 1e-9, "쌍곡·삼각 함수식의 극한", True, ""))
        # plots
        T = 1.0 / fs
        tt, yy = pw.plot_points()
        tt2 = [x + T for x in tt]
        res.add_series("i", "i (정확 PWL)", "A", tt + tt2, yy + yy)
        tg = np.linspace(0.0, 2 * T, 801)
        for hh, key in ((1, "i_h1"), (5, "i_h5"), (15, "i_h15")):
            k = np.arange(1, hh + 1)
            rec = c[0].real + 2.0 * np.real(np.exp(1j * TWO_PI * fs * np.outer(tg, k)) @ c[1:hh + 1])
            res.add_series(key, f"Fourier 합 h ≤ {hh}", "A", tg.tolist(), rec.tolist(), dash=True)
        bands = []
        tp = phi / (TWO_PI * fs)
        for k2 in range(2):
            o = k2 * T
            for a, b, m in ((0.0, tp, "+-"), (tp, T / 2, "++"), (T / 2, T / 2 + tp, "-+"), (T / 2 + tp, T, "--")):
                bands.append({"x0": o + a, "x1": o + b, "mode": m, "label": DAB_LABELS[m]})
        res.add_plot("p_wave", "DAB 1차 전류와 부분 Fourier 합", ["i", "i_h1", "i_h5", "i_h15"], y_label="전류", y_unit="A", bands=bands, group="hc", level="A",
                     proved="4구간 직선 전류를 닫힌 형태 계수로 전개했다. 기울기 불연속 때문에 계수는 1/h²로 줄고, 15차까지 더하면 파형이 거의 겹친다.",
                     not_yet="이상 SPS (L_m → ∞). 실제 권선 전류는 자화전류·dead time 때문에 조금 다르다.")
        odd = h % 2 == 1
        hk = h[odd][:25]
        res.add_series("Ih", "I_h RMS (홀수)", "A", hk.tolist(), Ih[odd][:25].tolist(), style="points")
        hl_ = np.linspace(0.8, float(hk[-1]) + 3.0, 60)
        env = math.sqrt(2.0) * sum_ds * T / TWO_PI**2  # sqrt(2) sum|ds| / (T w^2) with w = 2 pi / T
        res.add_series("Ih_ref", "상한 포락선 √2·Σ|Δs|/(T(hω)²)", "A", hl_.tolist(), (env / hl_**2).tolist(), dash=True)
        res.add_plot("p_spec", "고조파 전류 (반파 대칭 → 홀수만)", ["Ih", "Ih_ref"], x_label="차수 h", x_unit="", y_label="I_h", y_unit="A", kind="xy", log_y=True, level="A",
                     proved=f"zero-DC·반파 대칭 파형이라 짝수 고조파와 DC가 0이다. 경사 구간 길이 φ/ω 때문에 h ≈ 2π/φ = {TWO_PI / phi:.1f}의 배수 근처에서 성분이 거의 0이 된다.",
                     not_yet="측정 파형이 아니다. 비대칭·offset이 있으면 짝수 성분과 DC가 생긴다.")
        cum = np.cumsum(P_h) / P_harm
        cum_dc = np.cumsum(Ih * Ih * Rdc)
        hx = [0.0] + hk.tolist()
        res.add_series("Ph", "누적 Σ I_h²R_ac(h)", "W", hx, [float(abs(c[0]) ** 2 * Rdc)] + (abs(c[0]) ** 2 * Rdc + np.cumsum(P_h))[odd][:25].tolist())
        res.add_series("Ph_dc", "누적 Σ I_h²R_dc (R_dc만)", "W", hx, [float(abs(c[0]) ** 2 * Rdc)] + (abs(c[0]) ** 2 * Rdc + cum_dc)[odd][:25].tolist(), dash=True)
        res.add_plot("p_loss", "고조파를 더해 가는 누적 손실: 합성 R_ac vs R_dc", ["Ph", "Ph_dc"], x_label="더한 최대 차수 h", x_unit="", y_label="누적 손실", y_unit="W", kind="xy", level="A",
                     hlines=[{"y": P_harm, "label": f"Σ R_ac {P_harm:.3g} W (h ≤ {Hm})"}, {"y": P_dc, "label": f"I_rms²R_dc {P_dc:.3g} W"}],
                     proved=f"R_ac 누적은 5차까지 {100 * cum[4]:.1f} %, 15차까지 {100 * cum[14]:.1f} %. 고조파 전류는 작아도 F_R이 커서 손실 비중이 전류 비중보다 크다.",
                     not_yet="F_R은 1-D Dowell 합성 값이다. 측정 R_ac(f)로 바꿔야 한다(실험 5 계획).")
        hc = np.linspace(1, float(hk[-1]), 200)
        res.add_series("FR", f"F_R(h) Dowell {int(v['layers'])}층", "", hc.tolist(), dowell(Delta * np.sqrt(hc), int(v["layers"])).tolist())
        res.add_series("FRp", "홀수 차수", "", hk.tolist(), FR[odd][:25].tolist(), style="points")
        res.add_plot("p_FR", "R_ac/R_dc = F_R(Δ√h, m)", ["FR", "FRp"], x_label="차수 h", x_unit="", y_label="F_R", y_unit="", kind="xy", level="A",
                     proved=f"δ ∝ 1/√f라 h차에서 Δ가 √h배 — 층 수 {int(v['layers'])}에서 F_R이 빠르게 커진다.",
                     not_yet="평판 1-D 근사. gap fringing·끝단·litz 가닥 구조는 없다.")
        rows = [[int(hh), hh * fs / 1e3, float(Ih[hh - 1]), float(FR[hh - 1]), float(P_h[hh - 1]), 100 * float(cum[hh - 1])] for hh in (1, 3, 5, 7, 9, 11, 13, 15)]
        res.tables.append(Table("t_h", "고조파별 손실 (홀수)", ["h", "f [kHz]", "I_h RMS [A]", "F_R", "I_h²R_ac [W]", "누적 [%]"], rows,
                                note=f"R_dc({Tw:g} °C) = {Rdc * 1e3:.4g} mΩ (합성), 온도 계수 {ALPHA_CU:g}/K. 고온에서는 R_dc가 늘고 δ도 커져 F_R은 조금 준다."))
        level_verdict = ("SCREEN_ONLY", f"고조파별 합 {P_harm:.3f} W vs R_dc만 {P_dc:.3f} W (×{P_harm / P_dc:.2f}) — R_ac는 합성 Dowell")
    res.tables.append(Table("t_break", "손실 분해 (무엇이 들어 있고 무엇이 빠졌나)", ["항목", "값", "상태"], [
        ["구리: 전체 RMS × R_dc", f"{P_dc:.4g} W", "과소평가 (고조파 R_ac 누락)"],
        ["구리: Σ I_h²R_ac(hf₀)", f"{P_harm:.4g} W", "교재 목록" if v["wave"] == "list" else "합성 R_ac (SCREEN_ONLY)"],
        ["core loss", "—", "MISSING_INPUT: 재료 손실 곡선(f·B·온도·파형)·B 정의 없음"],
        ["합계", "—", "core loss 없이 닫을 수 없음"],
    ], note="FL07의 Steinmetz/iGSE는 SYNTHETIC 계수의 비교 감각용이며, 여기 합계에 넣지 않는다."))
    res.verdict(*level_verdict)
    res.verdict("MISSING_INPUT", "정밀 core loss는 재료 곡선·파형·온도 자료가 필요하다 — 손실 합계와 열 판정은 막혀 있다")
    res.assumptions += ["권선이 선형(저항이 전류에 무관)이라 성분별 손실을 더할 수 있다", "두 권선 전류 비가 일정해 단일 R_ac(f)로 근접효과를 대표한다"]
    if v["wave"] != "list":
        res.assumptions += [f"R_ac(hf₀, T) = R_dc(T)·F_R(Δ√h, m), 합성 Dowell {int(v['layers'])}층·h {v['h_cond'] * 1e3:g} mm", f"R_dc(20 °C) {v['Rdc20'] * 1e3:g} mΩ (합성)"]
    res.not_valid_for += ["core loss", "측정 R_ac(f) 없는 정밀 권선 손실", "자화전류로 두 권선 MMF가 불균형한 경우의 근접효과"]
    res.interpretation = (
        f"전류가 비정현파이면 성분마다 R_ac(hf₀)가 다르다. 전체 RMS에 R_dc만 곱한 {P_dc:#.3g} W는 고조파별 합 {P_harm:#.3g} W보다 작다. "
        "차이는 고조파 전류가 작아도 F_R이 커서 생긴다. 이 합은 선형 권선 모델이 타당한 범위의 값이고, core loss는 재료 자료 없이 만들지 않았다."
    )
    return res


# ======================================================================================
# Experiment 4: L, C tolerance and correlation -> f_r and the operating map
# ======================================================================================


def run_tolerance_map(v: dict) -> Result:
    res = Result("EX04", "tolerance_map", "A (FHA 운전점 + Monte Carlo)")
    n, Lr, Cr, Lm, P = v["n"], v["Lr"], v["Cr"], v["Lm"], v["P"]
    fmin, fmax = v["f_min"], v["f_max"]
    tL, tC, tM = v["tol_L"] / 100.0, v["tol_C"] / 100.0, v["tol_Lm"] / 100.0
    rho = v["rho"]
    N, seed, ks = int(v["N_mc"]), int(v["seed"]), v["k_sigma"]
    fr0 = 1.0 / (TWO_PI * math.sqrt(Lr * Cr))
    tb = _close(Lr, CLLC_LR) and _close(Cr, CLLC_CR, 1e-12)
    tb5 = tb and _close(tL, 0.05) and _close(tC, 0.05)
    tb_op = tb and _close(n, CLLC_N) and _close(Lm, CLLC_LM) and _close(P, CLLC_P) and _close(fmin, 120e3) and _close(fmax, 210e3)
    corners = [("nom", "공칭", 1.0, 1.0), ("pp", "L +, C +", 1 + tL, 1 + tC), ("mm", "L −, C −", 1 - tL, 1 - tC), ("pm", "L +, C −", 1 + tL, 1 - tC), ("mp", "L −, C +", 1 - tL, 1 + tC)]
    refs_fr = {"nom": 150e3, "pp": 142.857e3, "mm": 157.895e3, "pm": 150.188e3, "mp": 150.188e3}
    lab_fr = {"nom": "교재 150 kHz", "pp": "교재 142.857 kHz", "mm": "교재 157.895 kHz", "pm": "교재 150.188 kHz", "mp": "150.188 kHz (대칭)"}
    rows = []
    cres = {}
    for key, lab, fl, fc in corners:
        tk = (Lr * fl, Cr * fc, Lm, Lr * fl, Cr * fc)
        frk = 1.0 / (TWO_PI * math.sqrt(tk[0] * tk[1]))
        sols = {}
        for Vo, Vi in CLLC_CORNERS:
            sols[(Vo, Vi)] = [r for r, ind in fha_roots_brentq(n, Vo, Vi, P, tk, fmin, fmax) if ind]
        gm, _ = max_ind_gain(n, 920.0, P, tk, fmin, fmax)
        margin = gm / (n * 920.0 / 850.0) - 1.0
        cres[key] = {"fr": frk, "sols": sols, "margin": margin, "tk": tk, "Z0": math.sqrt(tk[0] / tk[1])}
        res.add_metric(f"fr_{key}", f"f_r ({lab})", frk, "Hz", ref=refs_fr[key] if (tb5 or key == "nom" and tb) else None, ref_label=lab_fr[key], tol=5e-6)
        s800 = sols[(800.0, 800.0)]
        s920 = sols[(920.0, 850.0)]
        s650 = sols[(650.0, 700.0)]
        rows.append([lab, f"{fl:.3f} / {fc:.3f}", frk / 1e3, cres[key]["Z0"], max(s650) / 1e3 if s650 else "해 없음", max(s800) / 1e3 if s800 else "해 없음",
                     " / ".join(f"{x / 1e3:.3f}" for x in sorted(s920)) or "해 없음", 100 * margin])
    nomsol = cres["nom"]["sols"]
    if nomsol[(800.0, 800.0)]:
        res.add_metric("fop_nom", "공칭 800/800 V 운전 주파수 (FHA)", max(nomsol[(800.0, 800.0)]), "Hz", ref=162.811e3 if tb_op else None, ref_label="교재 FL10 162.811 kHz", tol=5e-6)
    s920 = sorted(nomsol[(920.0, 850.0)])
    if len(s920) >= 2:
        res.add_metric("f920_lo", "공칭 920/850 V 낮은 해", s920[0], "Hz", ref=136.099e3 if tb_op else None, ref_label="교재 136.099 kHz", tol=5e-6)
        res.add_metric("f920_hi", "공칭 920/850 V 높은 해", s920[-1], "Hz", ref=147.061e3 if tb_op else None, ref_label="교재 147.061 kHz", tol=5e-6)
    for key in ("pp", "mm", "pm", "mp"):
        s = cres[key]["sols"][(800.0, 800.0)]
        if s:
            res.add_metric(f"fop_{key}", f"800/800 V 운전 주파수 ({dict((c[0], c[1]) for c in corners)[key]})", max(s), "Hz", basis=f"Z₀ = √(L/C) = {cres[key]['Z0']:.2f} Ω")
    res.add_metric("margin_nom", "공칭 920/850 V 이득 여유 (inductive 최대/필요 − 1)", cres["nom"]["margin"], "", basis="0 이하이면 해 없음")
    # extra Lm corners at nominal L, C and the worst combination
    extra = []
    for fm in (1 - tM, 1 + tM):
        if fm <= 0 or tM == 0:
            continue
        for key, lab, fl, fc in corners:
            tk = (Lr * fl, Cr * fc, Lm * fm, Lr * fl, Cr * fc)
            gm, _ = max_ind_gain(n, 920.0, P, tk, fmin, fmax)
            extra.append((gm / (n * 920.0 / 850.0) - 1.0, lab, fm, tk))
    if extra:
        worst = min(extra, key=lambda e: e[0])
        res.add_metric("margin_worst", "L·C·L_m corner 중 최소 이득 여유 (920/850 V)", worst[0], "", basis=f"{worst[1]}, L_m ×{worst[2]:.2f}")
    # checks: brentq vs vectorized, power identity, |H(f_r)| = 1
    vec_in = [(cres[k]["tk"], Vo, Vi) for k in cres for Vo, Vi in CLLC_CORNERS]
    tk_a = np.array([x[0] for x in vec_in])
    up, lo, cnt, _ = fha_solve(n, np.array([x[1] for x in vec_in]), np.array([x[2] for x in vec_in]), P, tk_a[:, 0], tk_a[:, 1], tk_a[:, 2], tk_a[:, 3], tk_a[:, 4], fmin, fmax)
    errs = []
    for i, (tk, Vo, Vi) in enumerate(vec_in):
        key = list(cres)[i // 3]
        s = sorted(cres[key]["sols"][(Vo, Vi)])
        if s:
            errs.append(abs(up[i] - s[-1]) / s[-1])
        if len(s) >= 2:
            errs.append(abs(lo[i] - s[-2]) / s[-2])
        if cnt[i] != len(s):
            errs.append(1.0)
    res.add_check(Check("FHA 해: 벡터화 이분법 (MC 경로) vs brentq", "PASS" if errs and max(errs) < 1e-9 else "FAIL", max(errs) if errs else float("nan"), "rel", 1e-9,
                        path="Monte Carlo에 쓰는 격자·동시 이분법 vs 조밀 격자 + brentq, corner 5개 × 운전점 3개 (해 개수 포함)", independent=True))
    pe = []
    for key in cres:
        for (Vo, Vi), s in cres[key]["sols"].items():
            for r in s:
                I1, I2p, _, Rac, Zin = fha_currents(r, n, Vo, Vi, P, cres[key]["tk"])
                pe.append(max(abs(I1 * I1 * Zin.real - P), abs(I2p * I2p * Rac - P)) / P)
    res.add_check(Check("FHA 전력 항등식 (모든 corner의 해)", "PASS" if pe and max(pe) < 1e-8 else "FAIL", max(pe) if pe else float("nan"), "rel", 1e-8,
                        path="|I₁|²Re Z_in = |I₂′|²R_ac′ = P (무손실 tank의 에너지 보존)", independent=True))
    hr = [abs(complex(fha(cres[k]["fr"], n, Vo, P, *cres[k]["tk"])[0])) for k in cres for Vo in (650.0, 920.0)]
    res.add_check(Check("각 corner의 |H(f_r)| = 1 (대칭 tank, 부하 무관)", "PASS" if max(abs(x - 1) for x in hr) < 1e-12 else "FAIL", max(abs(x - 1) for x in hr), "", 1e-12,
                        path="f_r에서 Z_r = 0이면 H = Z_p/Z_p·R_ac′/R_ac′ = 1 — corner의 f_r 폐형식과 FHA 이득식이 같은 점을 가리키는지", independent=True))
    # ---- Monte Carlo ------------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((3, N))
    sL, sC, sM = math.log1p(tL) / ks, math.log1p(tC) / ks, math.log1p(tM) / ks

    def draw(r):
        zc = r * z[0] + math.sqrt(max(0.0, 1.0 - r * r)) * z[1]
        return np.exp(sL * z[0]), np.exp(sC * zc), np.exp(sM * z[2])

    fL, fC, fM = draw(rho)
    fr = fr0 / np.sqrt(fL * fC)
    s_an = 0.5 * math.sqrt(sL * sL + sC * sC + 2 * rho * sL * sC)
    s_mc = float(np.std(np.log(fr), ddof=1))
    se = s_an / math.sqrt(2 * (N - 1))
    ok_s = abs(s_mc - s_an) <= 4 * se + 1e-12
    res.add_metric("sig_fr", "σ(ln f_r) Monte Carlo", s_mc, "", ref=s_an if s_an > 0 else None, ref_label="½√(σ_L² + σ_C² + 2ρσ_Lσ_C)", tol=max(4 * se / s_an, 1e-9) if s_an > 0 else None,
                   basis=f"N = {N}, seed {seed}, ρ = {rho:g}, 공차 = {ks:g}σ (로그정규)", note=f"≈ f_r 상대 표준편차 {100 * s_mc:.3f} %")
    res.add_check(Check("MC σ(ln f_r) vs 해석식", "PASS" if ok_s else "FAIL", abs(s_mc - s_an), "", 4 * se + 1e-12,
                        path="난수 표본의 표준편차 vs ln f_r = −½(ln L + ln C) + 상수의 분산식 (4 표준오차 이내)", independent=True, detail=f"MC {s_mc:.6f}, 해석 {s_an:.6f}, 표준오차 {se:.2e}"))
    tkM = (Lr * fL, Cr * fC, Lm * fM, Lr * fL, Cr * fC)
    up8, _, c8, _ = fha_solve(n, 800.0, 800.0, P, *tkM, fmin, fmax, G=901)
    _, _, c9, g9 = fha_solve(n, 920.0, 850.0, P, *tkM, fmin, fmax, G=901)
    _, _, c6, _ = fha_solve(n, 650.0, 700.0, P, *tkM, fmin, fmax, G=901)
    nosol = (c8 == 0) | (c9 == 0) | (c6 == 0)
    p_nosol = float(np.mean(nosol))
    marg = g9 / (n * 920.0 / 850.0) - 1.0
    fop = up8[np.isfinite(up8)]
    res.add_metric("sig_fop", "σ(ln f_op) 800/800 V (MC)", float(np.std(np.log(fop), ddof=1)) if fop.size > 2 else float("nan"), "", basis="가장 높은 inductive 해")
    res.add_metric("fop_p1", "f_op 800/800 V 1 % 분위", float(np.percentile(fop, 1)) if fop.size else float("nan"), "Hz")
    res.add_metric("fop_p99", "f_op 800/800 V 99 % 분위", float(np.percentile(fop, 99)) if fop.size else float("nan"), "Hz")
    res.add_metric("p_nosol", "세 corner 중 하나라도 해 없는 표본 비율", p_nosol, "", basis=f"{int(nosol.sum())}/{N}")
    res.add_metric("marg_p1", "920/850 V 이득 여유 1 % 분위 (MC)", float(np.nanpercentile(marg, 1)), "", basis="격자 최대 이득 기준")
    # rho sweep (common random numbers)
    rhos = [-1.0, -0.5, 0.0, 0.5, 1.0]
    Ns_ = min(N, 1000)
    sfr, sfo = [], []
    for r in rhos:
        a, b, m = draw(r)
        a, b, m = a[:Ns_], b[:Ns_], m[:Ns_]
        frr = fr0 / np.sqrt(a * b)
        sfr.append(float(np.std(np.log(frr), ddof=1)))
        u, _, _, _ = fha_solve(n, 800.0, 800.0, P, Lr * a, Cr * b, Lm * m, Lr * a, Cr * b, fmin, fmax, G=601)
        u = u[np.isfinite(u)]
        sfo.append(float(np.std(np.log(u), ddof=1)) if u.size > 2 else float("nan"))
    rg = np.linspace(-1, 1, 81)
    res.add_series("s_an", "σ(ln f_r) 해석식", "%", rg.tolist(), (100 * 0.5 * np.sqrt(np.maximum(sL * sL + sC * sC + 2 * rg * sL * sC, 0.0))).tolist())
    res.add_series("s_fr", "σ(ln f_r) MC", "%", rhos, [100 * x for x in sfr], style="points", color="#16a34a")
    res.add_series("s_fo", "σ(ln f_op) MC (800/800 V)", "%", rhos, [100 * x for x in sfo], style="points", color="#dc2626")
    res.add_plot("p_rho", "상관계수 ρ가 공진점과 운전 주파수의 산포를 바꾼다", ["s_an", "s_fr", "s_fo"], x_label="ρ (L–C 상관)", x_unit="", y_label="상대 표준편차", y_unit="%", kind="xy", level="A",
                 proved="f_r의 산포는 ρ = −1에서 0(σ_L = σ_C일 때)까지 줄지만, 운전 주파수의 산포는 0이 되지 않는다 — Z₀ = √(L/C)와 L_m 공차가 이득 곡선을 바꾸기 때문.",
                 not_yet=f"로그정규·공차 = {ks:g}σ·두 tank가 같은 인자로 움직인다는 가정. 실제 분포는 부품 자료로 바꿔야 한다.")
    # histogram of f_r with the analytic density
    lo_e, hi_e = float(np.min(fr)), float(np.max(fr))
    if hi_e - lo_e > 1e-9 * fr0:
        edges = np.linspace(lo_e, hi_e, 41)
        cnts, _ = np.histogram(fr, bins=edges)
        xs, ys = [], []
        for a, b, cc in zip(edges[:-1], edges[1:], cnts):
            xs += [a / 1e3, b / 1e3]
            ys += [float(cc), float(cc)]
        res.add_series("h_fr", f"MC 표본 수 (N = {N})", "", xs, ys)
        fg = np.linspace(lo_e, hi_e, 200)
        dens = N * (edges[1] - edges[0]) / (fg * s_an * math.sqrt(TWO_PI)) * np.exp(-((np.log(fg / fr0)) ** 2) / (2 * s_an * s_an)) if s_an > 0 else np.zeros_like(fg)
        res.add_series("h_an", "해석 로그정규 밀도", "", (fg / 1e3).tolist(), dens.tolist(), dash=True)
        res.add_plot("p_hist", f"f_r 분포 (ρ = {rho:g})", ["h_fr", "h_an"], x_label="f_r", x_unit="kHz", y_label="표본 수", y_unit="", kind="xy", level="A",
                     vlines=[{"x": cres[k]["fr"] / 1e3, "label": {"pp": "L+C+", "mm": "L−C−"}[k]} for k in ("pp", "mm") if lo_e <= cres[k]["fr"] <= hi_e] + [{"x": fr0 / 1e3, "label": "공칭"}],
                     proved="표본 분포가 해석 밀도와 겹친다(σ 검산). corner는 분포의 끝이지 대표값이 아니다.",
                     not_yet="분포 모양은 가정이다. 공차 한계를 3σ로 볼지, 균일 분포로 볼지는 부품 자료가 정한다.")
    # gain curves at the high corner and the operating map
    F = np.linspace(fmin, fmax, 901)
    gk = []
    for key, lab, fl, fc in corners:
        H, Zin, _, _ = fha(F, n, 920.0, P, *cres[key]["tk"])
        res.add_series(f"g_{key}", lab, "", (F / 1e3).tolist(), np.where(Zin.imag > 0, np.abs(H), np.nan).tolist(), dash=key != "nom")
        gk.append(f"g_{key}")
    res.add_plot("p_gain", "920/850 V: corner별 이득 곡선 (inductive 영역)", gk, x_label="f", x_unit="kHz", y_label="|H|", y_unit="", kind="xy", level="A",
                 hlines=[{"y": n * 920.0 / 850.0, "label": f"필요 {n * 920 / 850:.6f}"}], markers=[{"x": x / 1e3, "y": n * 920.0 / 850.0, "label": f"{x / 1e3:.3f}"} for x in s920],
                 proved="L·C가 같은 방향으로 움직이면 곡선이 주파수 축으로 이동하고, 반대 방향이면 f_r은 같아도 봉우리 높이·모양이 바뀐다.",
                 not_yet="FHA 정상 이득. 두 해 중 어느 분기를 쓸지는 제어 기울기·ZVS·RMS로 정한다(EX05).")
    Vo_g = np.linspace(650.0, 920.0, 28)
    Vi_g = coordinated_vi(Vo_g)
    mk = []
    for key, lab, fl, fc in corners:
        tk = cres[key]["tk"]
        u, lw, _, _ = fha_solve(n, Vo_g, Vi_g, P, *tk, fmin, fmax)
        res.add_series(f"m_{key}", f"{lab} (높은 해)", "kHz", Vo_g.tolist(), (u / 1e3).tolist(), dash=key != "nom")
        mk.append(f"m_{key}")
        if key == "nom":
            res.add_series("m_nom_lo", "공칭 (낮은 inductive 해)", "kHz", Vo_g.tolist(), (lw / 1e3).tolist(), style="points", color="#6b7280")
    res.add_plot("p_map", "운전점 지도: 배터리 전압에 따른 FHA 운전 주파수", mk + ["m_nom_lo"], x_label="V_o (배터리)", x_unit="V", y_label="f_op", y_unit="kHz", kind="xy", level="A",
                 hlines=[{"y": fmin / 1e3, "label": f"f_min {fmin / 1e3:g} kHz"}],
                 proved="L·C corner가 운전 주파수 지도 전체를 옮기고, 높은 전압 쪽에서는 두 번째 inductive 해가 나타난다(분기).",
                 not_yet="link 전압은 교재 세 점 사이를 선형으로 이은 가정이다. 경부하·역방향은 포함하지 않았다.")
    res.tables.append(Table("t_corner", "L·C ±공차 corner (L_m 공칭)", ["corner", "L / C 배수", "f_r [kHz]", "Z₀ [Ω]", "650/700 V [kHz]", "800/800 V [kHz]", "920/850 V 해 [kHz]", "920/850 V 이득 여유 [%]"], rows,
                            note="f_r이 같은 두 corner(L+C−, L−C+)도 Z₀가 달라 운전 주파수가 다르다. 이득 여유 = inductive 최대 이득/필요 이득 − 1."))
    if extra:
        res.tables.append(Table("t_lm", "L_m 공차를 더한 920/850 V 이득 여유", ["corner", "L_m 배수", "여유 [%]"], [[e[1], e[2], 100 * e[0]] for e in sorted(extra)], note="여유가 0 이하이면 그 조합에서 해가 없다."))
    # verdicts
    bad_corner = [lab for key, lab, *_ in corners if any(not s for s in cres[key]["sols"].values())]
    bad_lm = [f"{e[1]}·L_m ×{e[2]:.2f}" for e in extra if e[0] <= 0]
    if bad_corner or bad_lm or p_nosol > 0:
        parts = []
        if bad_corner:
            parts.append("L·C corner " + ", ".join(bad_corner))
        if bad_lm:
            parts.append("920/850 V에서 " + ", ".join(bad_lm))
        parts.append(f"MC 표본 {100 * p_nosol:.2f} %")
        res.verdict("NO_SOLUTION", "공차 조합에서 FHA 해가 사라진다: " + "; ".join(parts))
    else:
        res.verdict("CANDIDATE_FHA_ONLY", f"모든 corner·MC 표본에서 세 운전점의 FHA 해 존재 — 최소 이득 여유 {100 * min([cres[k]['margin'] for k in cres] + [e[0] for e in extra]):.2f} % (스위칭 검증 전)")
    res.verdict("INFO", f"ρ = {rho:g}: σ(ln f_r) {100 * s_mc:.3f} % (해석 {100 * s_an:.3f} %) — 분포·상관은 가정 (부품 자료로 확인)")
    res.assumptions += [f"공차: L ±{v['tol_L']:g} %, C ±{v['tol_C']:g} %, L_m ±{v['tol_Lm']:g} % = {ks:g}σ 로그정규", "두 tank의 L은 같은 인자, C는 같은 인자로 움직인다 (교재 corner 방식)",
                        f"L_m은 L·C와 독립", "FHA (A 수준)"]
    res.not_valid_for += ["스위칭 운전점·ZVS", "온도에 따른 L·C 드리프트의 실제 상관 (자료 필요)", "양산 수율 주장"]
    res.interpretation = (
        f"f_r = 1/(2π√(LC))라 L·C가 같은 방향으로 ±{v['tol_L']:g} %면 f_r이 {cres['pp']['fr'] / 1e3:.3f}–{cres['mm']['fr'] / 1e3:.3f} kHz로 움직이고, 반대 방향이면 {cres['pm']['fr'] / 1e3:.3f} kHz로 거의 그대로다. "
        f"하지만 운전점은 f_r만으로 정해지지 않는다: 반대 방향 두 corner의 800/800 V 운전 주파수는 Z₀ 차이 때문에 서로 다르다. 920/850 V corner의 이득 여유는 공칭 {100 * cres['nom']['margin']:.2f} %라 "
        "공차에 따라 두 해의 간격이 줄어든다. 상관계수는 부품의 온도 특성과 lot 산포 자료로 정해야 한다."
    )
    return res


# ======================================================================================
# Experiment 5: identification (open/short-circuit tests with winding capacitance)
# ======================================================================================


def port_network(v: dict, port: str) -> dict:
    """Element values in the measuring port's own units (primary-referred for the primary port, actual for the secondary)."""
    n = v["Np"] / v["Ns"]
    grounded = v["conn"] == "grounded"
    Cs_ref = v["C_s"] / (n * n)
    if port == "primary":
        return {"La": v["L_lp"], "Ra": v["R_p"], "Lm": v["Lm"], "Lb": v["L_ls"], "Rb": v["R_s"], "Ca": v["C_p"] + (v["C_ps"] if grounded else 0.0),
                "Cb": Cs_ref, "Rsh": n * n * v["R_sh"], "Lsh": n * n * v["L_sh"], "scale": 1.0, "N": v["Np"]}
    k = 1.0 / (n * n)
    return {"La": v["L_ls"] * k, "Ra": v["R_s"] * k, "Lm": v["Lm"] * k, "Lb": v["L_lp"] * k, "Rb": v["R_p"] * k, "Ca": v["C_s"] + (v["C_ps"] if grounded else 0.0),
            "Cb": v["C_p"] * n * n, "Rsh": v["R_sh"] * k, "Lsh": v["L_sh"] * k, "scale": n * n, "N": v["Ns"]}


def ladder_z(f, net: dict, test: str):
    """Phasor path: Z = [jwC_a + 1/(R_a + jwL_a + Z_m || Z_b)]^-1, Z_b = R_b + jwL_b + (1/(jwC_b) || Z_term)."""
    w = TWO_PI * np.asarray(f, dtype=float)
    Yb = 1j * w * net["Cb"]
    if test == "SC":
        Yb = Yb + 1.0 / (net["Rsh"] + 1j * w * net["Lsh"])
    Zb = net["Rb"] + 1j * w * net["Lb"] + 1.0 / Yb
    Zm = 1j * w * net["Lm"]
    Zbr = net["Ra"] + 1j * w * net["La"] + Zm * Zb / (Zm + Zb)
    return 1.0 / (1j * w * net["Ca"] + 1.0 / Zbr)


def true_l(net: dict, test: str) -> float:
    """What the test is meant to identify: OC -> L_a + L_m; SC -> L_a + L_m || L_b (no capacitance, ideal short)."""
    if test == "OC":
        return net["La"] + net["Lm"]
    return net["La"] + net["Lm"] * net["Lb"] / (net["Lm"] + net["Lb"])


class LCRTest(HybridSystem):
    """Time-domain path: T-network driven by v = V_pk cos(w t) generated by oscillator states (c, s).

    x = [i_a, i_b, v_b, (i_sh), c, s]; the magnetizing node voltage is eliminated,
    v_M = [(v - R_a i_a)/L_a + (R_b i_b + v_b)/L_b] / (1/L_m + 1/L_a + 1/L_b); the port capacitor C_a sits
    across the source, so the meter current is i_a - C_a V_pk w s.
    """

    def __init__(self, net: dict, test: str, Vpk: float, f: float):
        self.net, self.test, self.Vpk, self.w = net, test, Vpk, TWO_PI * f
        self.sc = test == "SC"
        self.state_names = ("i_a", "i_b", "v_b") + (("i_sh",) if self.sc else ()) + ("c", "s")
        self.nx = len(self.state_names)
        La, Lb, Lm, Ra, Rb, Cb = net["La"], net["Lb"], net["Lm"], net["Ra"], net["Rb"], net["Cb"]
        m = self.nx
        ic, is_ = m - 2, m - 1
        D = 1.0 / Lm + 1.0 / La + 1.0 / Lb
        vM = np.zeros(m)  # v_M as a row over x
        vM[0] = -Ra / La / D
        vM[1] = Rb / Lb / D
        vM[2] = 1.0 / Lb / D
        vM[ic] = Vpk / La / D
        A = np.zeros((m, m))
        A[0] = -vM / La
        A[0, 0] += -Ra / La
        A[0, ic] += Vpk / La
        A[1] = vM / Lb
        A[1, 1] += -Rb / Lb
        A[1, 2] += -1.0 / Lb
        A[2, 1] = 1.0 / Cb
        if self.sc:
            A[2, 3] = -1.0 / Cb
            A[3, 2] = 1.0 / net["Lsh"]
            A[3, 3] = -net["Rsh"] / net["Lsh"]
        A[ic, is_] = -self.w
        A[is_, ic] = self.w
        self._A = A
        self._vM = vM
        self._key = f"ex04lcr|{test}|" + "|".join(f"{net[k]:.10g}" for k in ("La", "Lb", "Lm", "Ra", "Rb", "Ca", "Cb", "Rsh", "Lsh")) + f"|{Vpk:.10g}|{f:.10g}"

    def mode(self, q):
        return AffineMode(self._key, self._A, np.zeros(self.nx), label=q)

    def _row(self, idx, val=1.0):
        r = np.zeros(self.nx + 1)
        r[idx] = val
        return r

    def outputs(self, q):
        m = self.nx
        im = self._row(0)
        im[m - 1] = -self.net["Ca"] * self.Vpk * self.w
        return {"v": self._row(m - 2, self.Vpk), "i_meter": im, "c": self._row(m - 2), "s": self._row(m - 1), "i_a": self._row(0), "i_b": self._row(1)}

    def stored_energy(self):
        m = self.nx
        W = np.zeros((m + 1, m + 1))
        La, Lb, Lm = self.net["La"], self.net["Lb"], self.net["Lm"]
        W[0, 0] = La + Lm
        W[1, 1] = Lb + Lm
        W[0, 1] = W[1, 0] = -Lm
        W[2, 2] = self.net["Cb"]
        if self.sc:
            W[3, 3] = self.net["Lsh"]
        W[m - 2, m - 2] = self.net["Ca"] * self.Vpk**2  # 1/2 C_a v^2 with v = V_pk c
        return W

    def powers(self, q):
        m = self.nx
        Q = np.zeros((m + 1, m + 1))
        Q[0, m - 2] = Q[m - 2, 0] = 0.5 * self.Vpk
        Q[m - 2, m - 1] = Q[m - 1, m - 2] = -0.5 * self.net["Ca"] * self.Vpk**2 * self.w
        L = np.zeros((m + 1, m + 1))
        L[0, 0] = self.net["Ra"]
        L[1, 1] = self.net["Rb"]
        if self.sc:
            L[3, 3] = self.net["Rsh"]
        return {"p_port": Q, "p_R": L}

    def describe(self, q):
        return q


def lcr_periodic(sysm: LCRTest):
    """Periodic steady state of the driven network: the circuit states' one-period map is affine (oscillator fixed at c = 1, s = 0)."""
    T = TWO_PI / sysm.w
    k = sysm.nx - 2

    def per(x):
        return simulate(sysm, sysm.test, np.concatenate([x, [1.0, 0.0]]), 0.0, T).z_end[:k].copy()

    g = per(np.zeros(k))
    Phi = np.column_stack([per(np.eye(k)[j]) - g for j in range(k)])
    x0 = np.linalg.solve(np.eye(k) - Phi, g)
    return np.concatenate([x0, [1.0, 0.0]]), float(np.max(np.abs(np.linalg.eigvals(Phi))))


def id_circuit(net: dict, test: str, port: str, conn: str) -> Circuit:
    c = Circuit("ex04_id", 780, 300, title=f"{'개방' if test == 'OC' else '단락'} 시험: {'1차' if port == 'primary' else '2차'} 포트에서 측정 (값은 측정 포트 기준 환산)")
    c.add("vsource", "M", 60, 150, 90, "LCR", "V cos ωt", lpos=(82, 146, "start"))
    ca = c.add("capacitor", "Ca", 160, 150, 90, "C_a", f"{net['Ca'] * 1e12:.4g} pF" + (" (+C_ps)" if conn == "grounded" else ""), lpos=(182, 146, "start"))
    ra = c.add("resistor", "Ra", 250, 60, 0, "R_a", f"{net['Ra'] * 1e3:.4g} mΩ", lpos=(250, 34, "middle"))
    la = c.add("inductor", "La", 340, 60, 0, "L_a (누설)", f"{net['La'] * 1e6:.4g} µH", lpos=(340, 34, "middle"))
    lm = c.add("inductor", "Lm", 410, 150, 90, "L_m", f"{net['Lm'] * 1e3:.4g} mH", lpos=(432, 146, "start"))
    lb = c.add("inductor", "Lb", 490, 60, 0, "L_b (누설)", f"{net['Lb'] * 1e6:.4g} µH", lpos=(490, 34, "middle"))
    rb = c.add("resistor", "Rb", 580, 60, 0, "R_b", f"{net['Rb'] * 1e3:.4g} mΩ", lpos=(580, 34, "middle"))
    cb = c.add("capacitor", "Cb", 650, 150, 90, "C_b", f"{net['Cb'] * 1e12:.4g} pF", lpos=(628, 146, "end"))
    lsh = net["Lsh"]
    lsh_s = f"{lsh * 1e6:.4g} µH" if lsh >= 1e-6 else (f"{lsh * 1e9:.4g} nH" if lsh >= 1e-9 else f"{lsh * 1e12:.4g} pH")
    sh = c.add("inductor", "Lsh", 730, 150, 90, "단락 strap" if test == "SC" else "단락 strap (SC만)", f"{lsh_s} (포트 기준 환산)", lpos=(724, 206, "end"))
    c.wire("w1", (60, 120), (60, 60), ra["a"])
    c.wire("w1c", (160, 60), ca["a"])
    c.wire("w2", ra["b"], la["a"])
    c.wire("w3", la["b"], (410, 60), lb["a"])
    c.wire("w3m", (410, 60), lm["a"])
    c.wire("w4", lb["b"], rb["a"])
    c.wire("w5", rb["b"], (650, 60), cb["a"])
    c.wire("w5s", (650, 60), (730, 60), sh["a"])
    c.wire("g1", (60, 180), (60, 250), (160, 250), (410, 250), (650, 250))
    c.wire("g1c", ca["b"], (160, 250))
    c.wire("g2", lm["b"], (410, 250))
    c.wire("g3", cb["b"], (650, 250))
    c.wire("g4", sh["b"], (730, 250), (650, 250))
    c.dot((160, 60), (410, 60), (650, 60), (160, 250), (410, 250), (650, 250))
    c.probe("pi", "i_meter", 110, 60, "right", "i_meter")
    base = ["M", "Ca", "Ra", "La", "Lm", "Lb", "Rb", "Cb", "w1", "w1c", "w2", "w3", "w3m", "w4", "w5", "g1", "g1c", "g2", "g3"]
    c.mode("OC", "개방 시험", base, "다른 권선 개방: L_m + L_a를 읽으려 하지만 C_a·C_b와의 공진이 가까우면 크게 읽힌다.", dim=["Lsh", "w5s", "g4"])
    c.mode("SC", "단락 시험", base + ["Lsh", "w5s", "g4"], "다른 권선 단락: 누설을 읽으려 하지만 strap은 반대쪽에서 n²배(또는 1/n²배)로 보인다.")
    c.notes.append("C_ps(권선 간)는 다른 권선을 계측 접지에 묶으면 포트에 병렬로 들어간다고 본 lumped 근사다 (합성).")
    return c


def run_identification(v: dict) -> Result:
    res = Result("EX04", "identification", "C (정확 시간영역) + A (phasor)")
    test, port, conn = v["test"], v["port"], v["conn"]
    n = v["Np"] / v["Ns"]
    net = port_network(v, port)
    f = v["f_test"]
    Vpk = math.sqrt(2.0) * v["V_test"]
    Z = complex(ladder_z(f, net, test))
    w = TWO_PI * f
    L_app = Z.imag / w
    Lt = true_l(net, test)
    err = L_app / Lt - 1.0
    Q = Z.imag / Z.real if Z.real != 0 else float("inf")
    sc = net["scale"]
    res.add_metric("L_app", "LCR가 읽는 직렬 L (1차 환산)", L_app * sc, "H", basis=f"{f / 1e3:g} kHz, {'1차' if port == 'primary' else '2차'} 포트, 다른 권선 {'단락' if test == 'SC' else '개방'}·{'접지' if conn == 'grounded' else 'floating'}")
    res.add_metric("L_true", "식별하려는 값 (capacitance·strap 없는 이상 시험, 1차 환산)", Lt * sc, "H", basis="OC: L_σ,a + L_m, SC: L_σ,a + L_m∥L_σ,b")
    res.add_metric("err", "측정 오차 L_app/L_true − 1", 100.0 * err, "%", note=f"허용 ±{v['tol_id']:g} %")
    res.add_metric("Q", "Q = Im Z/Re Z", Q, "", note=f"Q_min {v['Q_min']:g}")
    # sweep, SRF, valid window
    fg = np.geomspace(100.0, 20e6, 1400)
    Zg = ladder_z(fg, net, test)
    Lg = Zg.imag / (TWO_PI * fg)
    rat = Lg / Lt
    Qg = Zg.imag / Zg.real
    k0 = np.nonzero((Zg.imag[:-1] > 0) & (Zg.imag[1:] <= 0))[0]
    f_srf = float("nan")
    if k0.size:
        j = int(k0[0])
        f_srf = brentq(lambda x: complex(ladder_z(x, net, test)).imag, fg[j], fg[j + 1], xtol=1e-6, rtol=1e-13)
    if test == "OC":
        Leff, Cpar = net["La"] + net["Lm"], net["Ca"] + net["Cb"]
    else:
        Lb_s = net["Lb"] + net["Lsh"]
        Leff, Cpar = net["La"] + net["Lm"] * Lb_s / (net["Lm"] + Lb_s), net["Ca"]
    f_srf_an = 1.0 / (TWO_PI * math.sqrt(Leff * Cpar))
    if math.isfinite(f_srf):
        res.add_metric("f_srf", "self-resonance (Im Z = 0, 첫 병렬 공진)", f_srf, "Hz")
    res.add_metric("f_srf_an", "근사 1/(2π√(L_eff·C_포트))", f_srf_an, "Hz", basis=f"L_eff {Leff * sc * 1e6:.4g} µH (1차 환산, strap 포함), C {Cpar * 1e12:.4g} pF (포트 기준)",
                   note="" if math.isfinite(f_srf) else "sweep 상한 20 MHz 밖")
    good = (np.abs(rat - 1.0) <= v["tol_id"] / 100.0) & (Qg >= v["Q_min"])
    win = (float("nan"), float("nan"))
    if good.any():
        idx = np.nonzero(good)[0]
        # the contiguous run with the largest span
        runs, s0 = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                runs.append((s0, a))
                s0 = b
        runs.append((s0, idx[-1]))
        r0 = max(runs, key=lambda r: fg[r[1]] / fg[r[0]])
        win = (float(fg[r0[0]]), float(fg[r0[1]]))
    if math.isfinite(win[0]):
        res.add_metric("f_lo", "유효 측정 창 하한 (Q ≥ Q_min, |오차| ≤ 허용)", win[0], "Hz", note="sweep 하한 100 Hz까지 유효" if win[0] <= fg[0] * 1.0001 else "Q가 하한을 정한다")
        res.add_metric("f_hi", "유효 측정 창 상한", win[1], "Hz", note="SRF가 상한을 정한다 (SRF/10 부근이 경험칙)" if math.isfinite(f_srf) else "")
    else:
        res.warnings.append("이 설정(연결·strap)에서는 허용 오차와 Q를 함께 만족하는 주파수가 sweep(100 Hz–20 MHz)에 없다")
    # the core sees the magnetizing-branch voltage, not the port voltage (a short test leaves it small)
    wz = TWO_PI * f
    Zb_ = net["Rb"] + 1j * wz * net["Lb"] + 1.0 / (1j * wz * net["Cb"] + (1.0 / (net["Rsh"] + 1j * wz * net["Lsh"]) if test == "SC" else 0.0))
    Zmb = (1j * wz * net["Lm"]) * Zb_ / (1j * wz * net["Lm"] + Zb_)
    VM = Vpk * abs(Zmb / (net["Ra"] + 1j * wz * net["La"] + Zmb))
    B_test = VM / (w * net["N"] * v["Ae"] * 1e-6)
    res.add_metric("B_test", "시험 중 코어 B_pk (자화 가지 전압 기준)", B_test, "T", basis=f"|V_M|/(ω N A_e), {v['V_test']:g} V rms, |V_M|/V_pk = {VM / Vpk:.3g}",
                   note=f"운전 B {v['B_op']:g} T의 {100 * B_test / v['B_op']:.3g} % — 저 B 투자율 ≠ 운전 B 투자율 (재료 μ(B) 자료 없음)")
    res.add_metric("L_strap", "단락 strap의 반대쪽 환산값", (n * n * v["L_sh"]) if port == "primary" else v["L_sh"] / (n * n), "H",
                   basis=f"strap {v['L_sh'] * 1e9:g} nH, n² = {n * n:.4g}: {'2차 strap이 1차에서 n²배' if port == 'primary' else '1차 strap이 2차에서 1/n²배'}")
    # exact time-domain path
    sysm = LCRTest(net, test, Vpk, f)
    z0, rho_m = lcr_periodic(sysm)
    T = 1.0 / f
    tr = simulate(sysm, test, z0, 0.0, 2 * T)
    Ic = 2.0 / T * tr.integral_output_product(0.0, T, "i_meter", "c")
    Is = 2.0 / T * tr.integral_output_product(0.0, T, "i_meter", "s")
    Z_td = Vpk / complex(Ic, -Is)
    e_td = abs(Z_td - Z) / abs(Z)
    res.add_check(Check("임피던스: 정확 시간영역 해(Fourier 사영) vs phasor ladder", "PASS" if e_td < 1e-7 else "FAIL", e_td, "rel", 1e-7,
                        path="사인 전원을 진동자 상태로 둔 상태방정식의 주기해(행렬지수)에서 전류의 cos·sin 성분을 정확 적분 vs 복소 ladder 식", independent=True,
                        detail=f"Z_td = {abs(Z_td):.6g} Ω ∠{math.degrees(math.atan2(Z_td.imag, Z_td.real)):.4f}°, 주기 map |λ|max {rho_m:.4f}"))
    rt = abs(tr.state_at(T)[0][: sysm.nx - 2] - z0[: sysm.nx - 2]).max() / max(abs(z0[: sysm.nx - 2]).max(), 1e-30)
    res.add_check(Check("주기해 잔차 (한 주기)", "PASS" if rt < 1e-8 else "FAIL", rt, "rel", 1e-8, path="affine 주기 map의 고정점을 한 주기 적분해 되돌아오는지"))
    led = energy_ledger(tr, sysm, 0.0, T, ["p_port"], [], ["p_R"], rated_power=Vpk * Vpk / abs(Z) / 2)
    res.add_check(ledger_check(led, what="LCR 시험 회로: "))
    if math.isfinite(f_srf):
        res.add_check(check_close("SRF: 수치 해 vs 근사식 1/(2π√(L_eff·C))", f_srf, f_srf_an, 0.05, "ladder의 Im Z = 0 근 vs 집중 LC 근사 (R·C 분배 무시 → 5 % 허용)", True, "Hz"))
    # plots
    def clip(r):
        return np.where((r > 0) & (r < 2.5), r, np.nan)

    fk = (fg / 1e3).tolist()  # plot axis in kHz (plain tick labels on a log axis)
    res.add_series("r_cur", "현재 설정", "", fk, clip(rat).tolist())
    alt = dict(v)
    alt["conn"] = "floating" if conn == "grounded" else "grounded"
    net_c = port_network(alt, port)
    res.add_series("r_conn", f"다른 권선 {'floating' if conn == 'grounded' else '계측 접지'}", "", fk, clip(ladder_z(fg, net_c, test).imag / (TWO_PI * fg) / true_l(net_c, test)).tolist(), dash=True)
    net_p = port_network(v, "secondary" if port == "primary" else "primary")
    res.add_series("r_port", f"{'2차' if port == 'primary' else '1차'} 포트에서 측정", "", fk, clip(ladder_z(fg, net_p, test).imag / (TWO_PI * fg) / true_l(net_p, test)).tolist(), dash=True)
    bands = [{"x0": win[0] / 1e3, "x1": win[1] / 1e3, "mode": "win", "label": "유효 창"}] if math.isfinite(win[0]) else []
    res.add_plot("p_L", f"{'개방' if test == 'OC' else '단락'} 시험: 읽히는 L / 식별하려는 L", ["r_cur", "r_conn", "r_port"], x_label="시험 주파수", x_unit="kHz", y_label="L_app/L_true", y_unit="", kind="xy", log_x=True, level="A",
                 hlines=[{"y": 1 + v["tol_id"] / 100, "label": f"+{v['tol_id']:g} %"}, {"y": 1 - v["tol_id"] / 100, "label": f"−{v['tol_id']:g} %"}],
                 vlines=[{"x": f / 1e3, "label": "시험 f"}] + ([{"x": f_srf / 1e3, "label": "SRF"}] if math.isfinite(f_srf) else []),
                 bands=bands, proved="SRF에 다가가면 L이 부풀려진다(L/(1 − (f/f_SRF)²)). 다른 권선의 접지 여부·측정 포트·단락 방법이 곡선을 바꾼다 — 측정 조건이 값의 일부다.",
                 not_yet="합성 등가회로(집중 C, lumped C_ps)다. 분포 capacitance·코어 μ(f)·fixture 잔류는 실측으로 확인한다.")
    res.add_series("q_cur", "Q (현재 설정)", "", fk, np.where(Qg > 0, Qg, np.nan).tolist())
    res.add_plot("p_Q", "Q = Im Z/Re Z: 낮으면 L 판독이 R에 묻힌다", ["q_cur"], x_label="시험 주파수", x_unit="kHz", y_label="Q", y_unit="", kind="xy", log_x=True, log_y=True, level="A",
                 hlines=[{"y": v["Q_min"], "label": f"Q_min {v['Q_min']:g}"}], vlines=[{"x": f / 1e3, "label": "시험 f"}],
                 proved=("단락 시험은 누설이 작아 Q가 낮다 — 유효 창의 하한은 Q가, 상한은 SRF가 정한다." if test == "SC"
                         else "개방 시험은 L_m이 커서 Q가 높다(코어 손실 저항은 모델에 없음) — 유효 창의 상한은 SRF가 정한다."),
                 not_yet="실제 LCR 정확도 사양(기기별)은 넣지 않았다. Q_min은 가정 값이다.")
    smp = tr.sample(["v", "i_meter"], 0.0, 2 * T, per_segment=160)
    res.add_series("v", "v (LCR 전압)", "V", smp["t"], smp["v"])
    res.add_series("i_meter", "i_meter", "A", smp["t"], smp["i_meter"])
    bt = [{"x0": 0.0, "x1": 2 * T, "mode": test, "label": "개방 시험" if test == "OC" else "단락 시험"}]
    res.add_plot("p_t", "시험 파형 (정확 시간영역 주기해)", ["v"], y_label="전압", y_unit="V", bands=bt, group="id", level="C",
                 proved="시간영역 해에서 전류의 기본파 성분을 정확 적분해 phasor 식과 같은 임피던스를 얻었다.", not_yet="LCR 미터의 자동 레벨·보정 알고리즘은 모델 밖이다.")
    res.add_plot("p_ti", "측정 전류", ["i_meter"], y_label="전류", y_unit="A", bands=bt, group="id", level="C",
                 proved=f"전압 대비 위상 {math.degrees(math.atan2(Z.imag, Z.real)):.2f}° — 순수 L이면 90°, Q가 낮을수록 작아진다.", not_yet="잡음·분해능 없음.")
    res.circuit = {"diagram": id_circuit(net, test, port, conn).to_json(), "intervals": bt, "plot_group": "id"}
    # identification plan (values from this synthetic model)
    def window_of(test_, port_, conn_):
        vv = dict(v)
        vv["conn"] = conn_
        nt = port_network(vv, port_)
        z_ = ladder_z(fg, nt, test_)
        ok = (np.abs(z_.imag / (TWO_PI * fg) / true_l(nt, test_) - 1) <= v["tol_id"] / 100) & (z_.imag / z_.real >= v["Q_min"])
        return (fg[ok].min(), fg[ok].max()) if ok.any() else (float("nan"), float("nan"))

    w_oc = window_of("OC", "primary", "floating")
    w_sc1 = window_of("SC", "primary", "floating")
    w_sc2 = window_of("SC", "secondary", "floating")

    def ff(x):
        return f"{x / 1e6:.3g} MHz" if x >= 1e6 else f"{x / 1e3:.3g} kHz"

    def fw(wd):
        return "없음 (조건 불만족)" if not math.isfinite(wd[0]) else f"{ff(wd[0])}–{ff(wd[1])}"

    res.tables.append(Table("t_plan", "식별 계획 (값은 이 합성 모델 기준)", ["항목", "시험·연결", "주파수·여자", "보정·기록", "함정", "현재 상태"], [
        ["L_m", "개방 시험: 1차 측정, 2차 개방·floating", f"창 {fw(w_oc)}; 시험 B {B_test * 1e3:.3g} mT ≪ 운전 {v['B_op']:g} T → 운전 B 수준 여자로 재확인", "open/short 보정, f·V·연결·온도 기록",
         "SRF 근처 과대, 2차 접지 시 C_ps로 SRF 하강, gap 공차", "ASSUMED → MISSING_INPUT (실측)"],
        ["누설 L_σ", "단락 시험: 고권선(1차) 단락 + 저권선(2차) 측정, 또는 Kelvin 단락", f"1차 측정·2차 strap 단락: {fw(w_sc1)}; 2차 측정·1차 단락: {fw(w_sc2)}", "측정 포트 fixture 보정, strap 치수·위치 기록",
         f"저권선 strap {v['L_sh'] * 1e9:g} nH → 1차에서 {n * n * v['L_sh'] * 1e6:.3g} µH", "ASSUMED → MISSING_INPUT"],
        ["C_p·C_s·C_ps", "SRF sweep 또는 고주파 C 측정; 권선 간은 각 권선 단락 후 3단자(guard)", "SRF 위·아래 sweep", "다른 권선 개방/단락/접지를 각각 기록",
         "측정 포트·접지 조건에 따라 다른 값", "ASSUMED → MISSING_INPUT"],
        ["R_ac(f)", "단락 시험 실수부 sweep (f₀ … 15f₀)", "Q가 너무 크면 실수부 분해능 한계", "strap·fixture R 보정(strap R도 n²배), 권선 온도",
         "두 권선 합으로만 나옴, 운전 시 전류 비(자화전류)와 다름", "MISSING_INPUT (실험 3은 합성 Dowell)"],
        ["core loss", "실제 파형 여자 (2권선법·열량법)", "운전 f·B·파형·온도·DC bias", "B 정의(B_pk/ΔB), 파형 기록", "사인 곡선을 사각파에 그대로 적용", "MISSING_INPUT"],
        ["온도", "hotspot(열전대/IR)·DCR 평균·코어 표면·냉각수 분리", "정격 운전 열평형", "측정 위치·시간", "AC 손실과 열 경계가 동시에 틀리면 hotspot 하나로 분리 불가", "MISSING_INPUT"],
    ], note="실제 값은 모두 합성 가정이다. 계획은 측정 조건을 값과 함께 받기 위한 것이다 (교재 E04 식별과 검증 계획)."))
    res.tables.append(Table("t_status", "source status", ["양", "이 실습의 값", "상태"], [
        ["L_m", f"{v['Lm'] * 1e3:g} mH", "ASSUMED"], ["L_σ (1차 환산 합)", f"{(v['L_lp'] + v['L_ls']) * 1e6:g} µH", "ASSUMED"],
        ["C_p / C_s / C_ps", f"{v['C_p'] * 1e12:g} pF / {v['C_s'] * 1e9:g} nF / {v['C_ps'] * 1e12:g} pF", "ASSUMED"], ["R_ac(f)", "—", "MISSING_INPUT"], ["core loss", "—", "MISSING_INPUT"],
    ]))
    bad = abs(err) > v["tol_id"] / 100.0 or Q < v["Q_min"]
    if bad:
        why = []
        if abs(err) > v["tol_id"] / 100.0:
            why.append(f"L 오차 {100 * err:+.1f} % (허용 ±{v['tol_id']:g} %)")
        if Q < v["Q_min"]:
            why.append(f"Q {Q:.3g} < {v['Q_min']:g} (실제 LCR 정확도 저하)")
        res.verdict("FAIL_CONSTRAINT", "이 측정 조건은 식별 정확도 요구를 만족하지 못한다: " + ", ".join(why))
    else:
        res.verdict("PASS_WITHIN_MODEL", f"이 합성 모델에서 {f / 1e3:g} kHz 측정은 유효 창 안 (오차 {100 * err:+.2f} %, Q {Q:.3g})")
    res.verdict("MISSING_INPUT", "L_m·누설·C·R_ac 실측값이 없다 — 현재 값은 합성 가정이며 계획표의 조건으로 측정해야 한다")
    res.assumptions += ["T-등가회로 + 집중 권선 capacitance (C_p는 1차 포트, C_s는 2차 포트)", "C_ps는 다른 권선을 계측 접지에 묶으면 포트에 병렬로 더해지는 lumped 근사", "선형 코어 (μ가 B·f와 무관), 코어 손실 저항 없음",
                        "LCR 미터 = 이상 사인 전압원 + 이상 전류 측정 (fixture는 보정되었다고 가정)"]
    res.not_valid_for += ["실제 transformer 값", "분포 capacitance의 고차 공진", "측정기 정확도 사양"]
    res.interpretation = (
        f"LCR가 읽는 {L_app * sc * 1e6:.4g} µH는 식별하려는 {Lt * sc * 1e6:.4g} µH와 {100 * err:+.2f} % 다르다. SRF {f_srf / 1e3:.4g} kHz에 가까울수록 capacitance 전류가 L을 부풀리고, "
        "단락 시험에서는 strap이 반대쪽으로 n²배 환산되어 보이며, 저주파에서는 Q가 낮아 L이 R에 묻힌다. 그래서 측정값에는 주파수·여자·보정·다른 권선 연결이 함께 붙어야 한다."
    )
    return res


# ======================================================================================
# Learning content
# ======================================================================================

_Q = [
    Question(
        "기초편 n = 0.93을 실제 권선 13:14로 바꾸면 무엇이 같이 바뀌나?",
        "n이 −0.15 % 바뀌어 필요 이득 nV_o/V_i와 FHA 운전 주파수가 이동한다(800/800 V 162.811 → 약 162.99 kHz). 권선수가 정해지면서 B(N_pA_e), window·구리(암페어-턴), L_m(같은 gap이면 N²), "
        "누설·capacitance(권선 배치)가 동시에 바뀐다. 2차 물리 L/C를 유지하면 환산 tank가 비대칭이 되고, 다시 고르면 다른 부품이다. 13:14는 최종 선정이 아니라 다시 계산할 후보다.",
        "What changes if the continuous ratio 0.93 becomes a real 13:14 winding?",
        "The ratio moves by 0.15 %, so the required gain and the FHA operating frequency shift slightly. Fixing the turns also fixes the flux, the window and copper through the ampere-turns, the magnetizing inductance through the gap, and leakage and capacitance through the winding layout. Keeping the physical secondary tank makes the referred tank slightly asymmetric; re-tuning it means different parts. So 13:14 is a candidate to recompute, not a final choice.",
        ["운전 주파수 이동", "B·window·구리", "L_m ∝ N²", "2차 tank 유지/재선정"],
    ),
    Question(
        "N_p 50, N_s 3, I_p 2.02 A, I_s 33.665 A, J 4 A/mm², k_u 0.3의 window 하한은?",
        "(50·2.02 + 3·33.665)/4/0.3 = 168.33 mm² (교재 168.329는 반올림 전류; 정확 전류 2.019881509/33.66469 A로는 168.3235 mm²). bare copper 하한일 뿐 절연·보빈·termination·AC 손실·온도별 허용 J는 따로 봐야 한다. "
        "1차 전류가 작아 보여도 50 turns의 누적 구리와 LV 권선이 모두 창을 차지한다.",
        "N_p is 50, N_s 3, I_p 2.02 A, I_s 33.665 A, J 4 A/mm² and k_u 0.3. What is the window lower bound?",
        "About 168.33 mm²: 168.329 with the textbook's rounded currents and 168.3235 with the exact ones. It is a bare-copper lower bound; insulation, bobbin, terminations, AC loss and the temperature-dependent current density come on top.",
        ["168.33 mm²", "반올림", "bare copper 하한", "절연·보빈"],
        kind="calc",
    ),
    Question(
        "“주파수를 두 배로 하면 transformer 크기는 반으로 줄죠?”",
        "B 제한의 N·A_e 요구는 절반이 될 수 있지만 core loss(f^α·파형), AC 구리(δ ∝ 1/√f, 근접효과), 열, capacitance·EMI는 줄지 않거나 커진다. 절연 거리(creepage·clearance)는 주파수와 무관해 "
        "작은 코어에서는 window의 더 큰 비율을 차지한다. 하나의 scaling law를 전체 설계에 쓰지 않는다.",
        "“If we double the frequency, the transformer is half the size, right?”",
        "Only the flux-limited turns-area product halves. Core loss, AC copper loss through skin and proximity effects, thermal limits, capacitance and EMI do not scale down and often get worse, and insulation distances do not shrink with frequency, so they take a larger share of a smaller window. I would not use one scaling law for the whole design.",
        ["B 제한만", "core loss·AC 구리", "절연 거리 불변", "열"],
        kind="pressure",
    ),
    Question(
        "고조파 전류 [10, 3, 2] A_rms, R_ac [20, 40, 60] mΩ의 권선 손실과 R_dc(20 mΩ)만 쓴 값은?",
        "10²·0.02 + 3²·0.04 + 2²·0.06 = 2 + 0.36 + 0.24 = 2.60 W. 전체 RMS² = 113 A²에 20 mΩ만 곱하면 2.26 W — 0.34 W(실제 손실의 13 %)를 빠뜨린다. 선형 R_ac가 타당한 범위의 합이며, "
        "두 권선의 전류 비가 자화전류 때문에 변하면 근접효과 분포가 달라져 단일 R_ac(f)로 부족할 수 있다.",
        "Harmonic currents are 10, 3 and 2 A rms with AC resistances of 20, 40 and 60 mΩ. What is the winding loss, and what do you get with the DC resistance only?",
        "2.60 W summed per harmonic, against 2.26 W from the total RMS times 20 mΩ, so 0.34 W or 13 % is missed. The sum assumes a linear AC-resistance model; if the two windings' current ratio changes, for example through magnetizing current, the proximity field changes too.",
        ["2.60 W", "2.26 W", "고조파별 R_ac", "선형 범위"],
        kind="calc",
    ),
    Question(
        "L, C가 각각 ±5 %일 때 f_r은 어디까지 움직이나? 운전점은?",
        "둘 다 +5 %: 142.857 kHz, 둘 다 −5 %: 157.895 kHz, 반대 방향: 150.188 kHz — 상관관계가 f_r 이동을 정한다. 같은 f_r이어도 Z₀ = √(L/C)가 ±5 % 달라 이득 곡선의 모양이 바뀌고 "
        "n = 0.93 CLLC의 800/800 V 운전 주파수가 162.357 vs 163.721 kHz로 다르다. 920/850 V corner의 이득 여유도 줄어든다.",
        "With ±5 % on L and C, how far can the resonance move, and what about the operating point?",
        "Both high gives 142.857 kHz, both low 157.895 kHz, opposite directions 150.188 kHz, so the correlation decides the shift. Even at the same resonance the characteristic impedance differs, which reshapes the gain curve: the 800 V operating frequency of the 0.93 design is 162.357 versus 163.721 kHz, and the high-voltage corner loses gain margin.",
        ["142.857/157.895 kHz", "150.188 kHz", "상관관계", "Z₀ 변화"],
        kind="calc",
    ),
    Question(
        "개방/단락 시험으로 L_m과 누설을 잴 때 무엇을 함께 기록하나?",
        "주파수(SRF/10 이하, 단락 시험은 Q가 충분한 주파수), 여자 전압(→ B 수준: LCR 1 V는 운전 B의 약 1 %), fixture open/short 보정, 다른 권선의 연결(개방·단락·접지 — C_ps가 달라진다), "
        "단락 방법(저권선 쪽 strap은 반대쪽에서 n²배), 온도. SRF 근처 LCR 한 점 값을 저주파 L로 쓰지 않는다.",
        "When you measure magnetizing and leakage inductance with open and short tests, what do you record with the values?",
        "The test frequency, well below self-resonance and with enough Q for the short test; the excitation level, since a 1 V LCR signal is about one percent of the operating flux; the fixture compensation; how the other winding was connected, open, shorted or grounded; how the short was made, because a strap on the low-turns side looks n² larger from the other side; and the temperature.",
        ["주파수·SRF", "여자 B 수준", "fixture 보정", "다른 권선 연결", "strap n²"],
    ),
    Question(
        "“코어 손실을 정확히 계산해 주세요.”",
        "재료의 손실 곡선(주파수·B·온도, 사인/사각 파형, B_pk/ΔB 정의), DC bias·minor loop 여부, 실제 권선 전압 파형이 있어야 한다. 없으면 정밀 core loss는 MISSING_INPUT이고 합성 계수 screen은 비교 감각용일 뿐이다. "
        "필요하면 실제 파형 여자 측정(2권선법·열량법)을 제안한다.",
        "“Please calculate the core loss exactly.”",
        "I need the material's loss curves over frequency, flux and temperature, with their waveform and flux definition, plus DC bias and minor-loop conditions and the real winding-voltage waveform. Without them the precise core loss is a missing input; a synthetic Steinmetz screen only gives a comparison. If needed I would propose a measurement with the real waveform.",
        ["재료 곡선", "B 정의·파형", "온도·DC bias", "MISSING_INPUT"],
        kind="pressure",
    ),
    Question(
        "n을 0.93에서 15:16(0.9375)으로 올리면 CLLC 운전점은?",
        "920/850 V corner의 필요 이득 n·920/850이 1.0147로 올라 inductive 최대 이득(약 1.012)을 넘는다 — FHA 해가 없다(NO_SOLUTION). n의 허용 범위는 변압기 계산이 아니라 모든 corner의 운전점 지도가 정한다(상한 약 0.935). "
        "반대로 n을 낮추면 high corner의 두 해가 하나로 바뀌어 분기 구조가 달라진다.",
        "What happens to the CLLC operating map if the ratio goes from 0.93 to 15:16?",
        "The required gain at the 920 V battery, 850 V link corner rises to about 1.015, above the maximum inductive gain of about 1.012, so there is no FHA solution there. The allowed ratio range comes from the operating map over all corners, with an upper limit near 0.935; lowering the ratio instead changes the branch structure at the high corner.",
        ["필요 이득 증가", "해 없음", "운전점 지도", "분기 구조"],
    ),
]

_CORE = [
    Param("J", "전류밀도 J (bare copper)", "A/mm²", 4.0, "A/mm²", vmin=0.5, vmax=20.0, source="TEXTBOOK", source_note="교재 합성 screening 값 4 A/mm²", group="window·구리"),
    Param("ku", "유효 충진율 k_u", "", 0.30, "", vmin=0.05, vmax=0.8, source="TEXTBOOK", source_note="교재 합성 값 0.30", group="window·구리"),
    Param("ins_frac", "절연이 차지하는 window 비율", "%", 0.0, "%", vmin=0.0, vmax=90.0, source="ASSUMED", source_note="0 = 미반영 (MISSING_INPUT): margin tape·보빈·층간 절연", group="window·구리"),
    Param("T_w", "권선 온도 (ρ 계산)", "°C", 100.0, "°C", vmin=-40.0, vmax=200.0, source="ASSUMED", group="window·구리"),
]

EXPERIMENTS = [
    Experiment(
        key="turns_window",
        title="정수 권선·B·window·구리: DAB 50:3을 부품으로 옮기기",
        goal=(
            "연속 n = 50/3을 정수 권선 후보로 바꾸면 n·B·window·구리 손실·운전 전류가 함께 바뀐다는 것을 계산한다. 교재 window 168.329 mm²(반올림 전류 2.02/33.665 A)와 정확 전류(2.019881509/33.66469 A)의 "
            "168.3235 mm²를 재현하고, 절연·열·core loss 입력이 없으면 최종 선택을 할 수 없음을 보인다."
        ),
        params=[
            Param("Np_ref", "기준 후보 N_p", "", 50, "", vmin=1, vmax=500, kind="int", source="TEXTBOOK", source_note="FL08 50:3", group="권선"),
            Param("Ns_ref", "기준 후보 N_s", "", 3, "", vmin=1, vmax=200, kind="int", source="TEXTBOOK", group="권선"),
            Param("Ns_max", "후보 탐색 N_s 최대", "", 6, "", vmin=1, vmax=40, kind="int", source="ASSUMED", group="권선"),
            Param("n_tol", "허용 n 오차 ±", "%", 1.0, "%", vmin=0.0, vmax=20.0, source="ASSUMED", source_note="운전점·ZVS 지도에서 받아야 할 값", group="권선"),
            Param("V1", "1차 버스 V₁ (공칭)", "V", 800.0, "V", vmin=10, vmax=3000, source="TEXTBOOK", group="DAB 운전점"),
            Param("VL", "2차 버스 V_L (공칭, 실제)", "V", 48.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", group="DAB 운전점"),
            Param("P", "전력 (모듈당)", "W", 1500.0, "W", vmin=1, vmax=1e5, source="TEXTBOOK", group="DAB 운전점"),
            Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", group="DAB 운전점"),
            Param("L", "직렬 L (1차 환산)", "H", 200e-6, "µH", vmin=1e-7, vmax=0.01, source="TEXTBOOK", group="DAB 운전점"),
            Param("V1_max", "최대 1차 버스 (B screen)", "V", 900.0, "V", vmin=10, vmax=3000, source="TEXTBOOK", source_note="FL08 900 V corner", group="DAB 운전점"),
            Param("VL_max", "최대 2차 버스 (B screen)", "V", 48.0, "V", vmin=1, vmax=1000, source="ASSUMED", source_note="LV 최대값 자료 없음", group="DAB 운전점"),
            Param("I_basis", "window·구리 전류 기준", "", "nominal", kind="choice", choices=[("nominal", "공칭 운전점 (교재 방식)"), ("worst", "교재 세 corner 최대")], source="ASSUMED", group="window·구리"),
            Param("Ae", "코어 A_e", "mm²", 250.0, "mm²", vmin=1.0, vmax=1e5, source="TEXTBOOK", source_note="10장 예제 250 mm²", group="코어 (합성)"),
            Param("Aw", "코어 window A_w", "mm²", 200.0, "mm²", vmin=1.0, vmax=1e5, source="ASSUMED", source_note="합성 코어", group="코어 (합성)"),
            Param("B_allow", "허용 B_pk", "T", 0.20, "T", vmin=0.01, vmax=2.0, source="ASSUMED", source_note="재료·온도 자료 없음", group="코어 (합성)"),
            Param("MLT", "평균 권선 길이 MLT", "m", 60e-3, "mm", vmin=1e-3, vmax=2.0, source="ASSUMED", group="코어 (합성)"),
            Param("Lm_ref", "기준 후보의 L_m", "H", 2e-3, "mH", vmin=1e-5, vmax=10.0, source="ASSUMED", source_note="FL07 합성 값", group="코어 (합성)"),
        ] + _CORE,
        presets=[
            Preset("textbook", "교재 50:3 (J 4 A/mm², k_u 0.3)", {}, "168.329 / 168.3235 mm²", ("nominal", "reference")),
            Preset("insulation", "window의 25 %를 절연에 사용", {"ins_frac": 25.0}, "절연 입력이 결론을 뒤집는다", ("failure", "reference")),
            Preset("worst_current", "최악 corner 전류로 window", {"I_basis": "worst"}, "900/36 V corner 전류", ("corner",)),
            Preset("big_window", "window 260 mm² (큰 합성 코어)", {"Aw": 260.0}, "66:4·67:4도 통과", ("variant",)),
        ],
        run=run_turns_window,
        model_level="A (screen) + C (검산)",
        suggested_change="절연이 차지하는 window 비율을 0 → 25 %로 바꾼다 (margin tape·보빈·층간 절연).",
        suggested={"ins_frac": 25.0},
        prediction=Prediction(
            "window의 25 %를 절연(margin tape·보빈)에 쓰면 50:3 후보는?",
            ["그대로 통과", "window 부족으로 탈락", "B가 초과한다", "모르겠다"],
            "window 부족으로 탈락",
            "유효 window가 200 → 150 mm²가 되어 필요 168.3 mm²보다 작다. 권선수를 줄이면 B가 초과하고(33:2: 0.27 T), 늘리면 window가 더 모자란다. 절연 입력 하나가 screen의 결론을 뒤집는다 — 그래서 절연 자료 없이 최종 선택을 하지 않는다.",
            ["Aw_eff", "Aw_exact", "n_pass"],
            handcalc=[{"key": "Aw_tb", "label": "window (교재 반올림 전류)", "unit": "mm²"}, {"key": "B_ref", "label": "B_pk (900 V 기준)", "unit": "T"}],
        ),
        student=(
            "변압기의 권선수는 정수다. 설계식이 n = 16.667을 요구해도 실제로는 50:3 같은 정수 쌍을 골라야 하고, 권선수를 바꾸면 코어 자속(B), 구리가 들어갈 창(window) 면적, 구리 손실이 함께 바뀐다. "
            "권선이 적으면 B가 커지고, 많으면 창이 모자란다. 그 사이에서 절연과 열까지 만족해야 부품이 된다."
        ),
        expert=(
            "① B screen: N_p ≥ V_w,max/(4 f A_e B_allow) — V_w,max는 직렬 L 배치를 확정하기 전이면 두 bridge 전압 중 큰 값(FL07). ② window: A_w ≥ (N_pI_p/J + N_sI_s/J)/k_u — bare copper 하한이며 절연·보빈·termination은 별도. "
            "교재 168.329 mm²는 반올림 전류(2.02/33.665 A)의 값, 정확 전류로는 168.3235 mm²(정오표 E-002). ③ J 고정이면 DC 구리 손실은 ρ·MLT·J·(N_pI_p + N_sI_s) — 암페어-턴에 비례. "
            "④ n이 바뀌면 V₂′ = n·V_L이 어긋나 SPS 전류의 순환 성분이 바뀐다 — 정확 PWL과 스위칭 해로 확인. ⑤ 같은 코어·gap이면 L_m ∝ N_p²: 권선을 줄이면 자화전류가 N⁻²로 는다. "
            "⑥ 최종 선택에는 절연(creepage·clearance·margin tape), 열(R_th·냉각), core loss(재료), 측정 누설/C가 필요하다 — 지금은 MISSING_INPUT."
        ),
        customer_ko=(
            "권선비는 계산상 16.667이지만 실제 권선은 50:3 같은 정수로 정해야 하고, 그 선택이 자속·창 면적·구리 손실을 동시에 바꿉니다. 현재 합성 코어 가정에서는 50:3만 자속과 창 조건을 함께 만족합니다. "
            "다만 절연 거리와 냉각 조건, 코어 재료 손실 자료가 없어 최종 선정이 아니라 가능한 영역까지만 말씀드릴 수 있습니다. 해당 자료를 주시면 다시 확인하겠습니다."
        ),
        customer_en=(
            "The ideal ratio is 16.667, but the real winding has to be an integer pair such as 50:3, and that choice changes the flux, the window area and the copper loss together. With the assumed core, only 50:3 passes both the flux and window screens. "
            "Without insulation distances, cooling conditions and core material loss data this is the feasible region, not a final selection. If you can share that data, I'll close the check."
        ),
        questions=[_Q[1], _Q[2]],
        circuit="ex04_dab",
        textbook=[TB_E04, TB_11, TB_10],
        reference_presets=["textbook", "insulation"],
        claim_limit="B·window·DC 구리 screen (A)과 이상 SPS 전류 (C). 최종 transformer 선정·절연·열·core loss는 주장하지 않는다.",
    ),
    Experiment(
        key="cllc_turns",
        title="CLLC n = 0.93을 정수 권선으로: 운전점 지도가 n의 허용 범위를 정한다",
        goal=(
            "연속 설계 n = 0.93(교재 FL10: 650/700 V 164.390 kHz, 800/800 V 162.811 kHz, 920/850 V 136.099·147.061 kHz)을 13:14 등 정수 후보로 바꾸고, 각 후보의 FHA 운전점·B·window를 다시 계산한다. "
            "n을 조금만 올려도(15:16, +0.8 %) 920/850 V corner의 해가 사라지는 것과, 2차 물리 L/C를 유지할지 다시 고를지가 다른 회로라는 것을 확인한다."
        ),
        params=[
            Param("n_t", "연속 설계 n", "", 0.93, "", vmin=0.3, vmax=3.0, source="TEXTBOOK", source_note="FL10 수정 후보 A", group="권선"),
            Param("Np_ref", "기준 후보 N_p", "", 13, "", vmin=1, vmax=300, kind="int", source="TEXTBOOK", source_note="교재 예 13:14", group="권선"),
            Param("Ns_ref", "기준 후보 N_s", "", 14, "", vmin=1, vmax=300, kind="int", source="TEXTBOOK", group="권선"),
            Param("Ns_max", "후보 탐색 N_s 최대", "", 32, "", vmin=1, vmax=80, kind="int", source="ASSUMED", group="권선"),
            Param("n_list", "표에 올릴 |Δn| 범위", "%", 1.0, "%", vmin=0.05, vmax=10.0, source="ASSUMED", group="권선"),
            Param("tank2", "2차 공진 L/C", "", "keep", kind="choice", choices=[("keep", "n = 0.93 설계의 물리 값 유지"), ("retune", "후보마다 다시 골라 환산 대칭 유지")], source="ASSUMED", group="권선"),
            Param("Lr", "L_r1 (= L_r2′ 설계)", "H", CLLC_LR, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", group="CLLC tank"),
            Param("Cr", "C_r1 (= C_r2′ 설계)", "F", CLLC_CR, "nF", vmin=1e-10, vmax=1e-5, source="TEXTBOOK", source_note="28.144773 nF", group="CLLC tank"),
            Param("Lm", "L_m (1차)", "H", CLLC_LM, "µH", vmin=1e-6, vmax=1e-1, source="TEXTBOOK", group="CLLC tank"),
            Param("P", "출력 전력", "W", CLLC_P, "kW", vmin=100.0, vmax=1e5, source="TEXTBOOK", group="CLLC tank"),
            Param("f_min", "주파수 하한", "Hz", 120e3, "kHz", vmin=1e3, vmax=5e6, source="TEXTBOOK", group="CLLC tank"),
            Param("f_max", "주파수 상한", "Hz", 210e3, "kHz", vmin=1e3, vmax=5e6, source="TEXTBOOK", group="CLLC tank"),
            Param("Ae", "코어 A_e", "mm²", 800.0, "mm²", vmin=1.0, vmax=1e5, source="ASSUMED", source_note="합성 코어", group="코어 (합성)"),
            Param("Aw", "코어 window A_w", "mm²", 600.0, "mm²", vmin=1.0, vmax=1e5, source="ASSUMED", group="코어 (합성)"),
            Param("B_allow", "허용 B_pk", "T", 0.20, "T", vmin=0.01, vmax=2.0, source="ASSUMED", group="코어 (합성)"),
            Param("MLT", "평균 권선 길이 MLT", "m", 90e-3, "mm", vmin=1e-3, vmax=2.0, source="ASSUMED", group="코어 (합성)"),
        ] + _CORE,
        presets=[
            Preset("nominal", "13:14, 2차 L/C 유지", {}, "교재 FL10 연속값 + 정수 후보", ("nominal", "reference")),
            Preset("retune", "13:14, 2차 L/C 재선정", {"tank2": "retune"}, "환산 대칭 유지", ("variant", "reference")),
            Preset("n_high", "기준 후보 15:16 (n +0.8 %)", {"Np_ref": 15, "Ns_ref": 16}, "920/850 V 해 없음", ("failure",)),
            Preset("small_core", "A_e 500 mm², B_allow 0.15 T", {"Ae": 500.0, "B_allow": 0.15}, "B 하한이 올라 13:14 탈락", ("corner",)),
        ],
        run=run_cllc_turns,
        model_level="A (FHA)",
        suggested_change="기준 후보를 13:14 → 15:16으로 바꾼다.",
        suggested={"Np_ref": 15, "Ns_ref": 16},
        prediction=Prediction(
            "15:16 (n = 0.9375, +0.8 %)이면 920/850 V corner의 FHA 해는?",
            ["두 해가 조금 이동한다", "해가 사라진다 (이득 부족)", "해가 하나 더 생긴다", "모르겠다"],
            "해가 사라진다 (이득 부족)",
            "필요 이득 n·V_o/V_i가 1.0147로 오르는데 inductive 영역의 최대 이득은 약 1.012다. n의 상한(약 0.935)은 변압기가 아니라 운전점 지도가 정한다.",
            ["n_max", "req920", "n_pass"],
            handcalc=[{"key": "req920", "label": "920/850 V 필요 이득 n·V_o/V_i", "unit": ""}, {"key": "Lr2", "label": "2차 물리 L_r2 = L_r/n²", "unit": "H"}],
        ),
        student=(
            "공진형 컨버터에서는 권선비 n이 필요한 전압이득을 정한다. n을 정수 권선에 맞추려고 조금만 바꿔도 높은 배터리 전압 corner에서 필요한 이득을 만들지 못할 수 있다. "
            "그래서 n의 허용 범위는 변압기 계산이 아니라 운전점 지도(모든 corner의 해)가 정한다."
        ),
        expert=(
            "① FHA: H = Z_p/(Z_r1 + Z_p)·R_ac′/(Z_r2′ + R_ac′), R_ac′ = 8n²R_dc/π², 필요 이득 nV_o/V_i. ② 후보마다 세 corner의 inductive 해를 찾는다. n ≳ 0.935면 920/850 V에서 해가 없고(NO_SOLUTION), "
            "n을 낮추면 두 해가 하나로(분기 구조 변화). ③ 2차 물리 L_r2·C_r2를 n = 0.93 설계값으로 유지하면 L_r2′ = n²L_r2, C_r2′ = C_r2/n²가 후보마다 달라진다 — 다시 고르면 환산 대칭. 두 회로는 다르다. "
            "④ B는 f_min 120 kHz와 max(V_i, nV_o), window는 세 corner의 최대 FHA 전류로 screen. L_m 200 µH를 유지하려면 A_L = L_m/N_p²가 후보마다 달라진다(gap 재설계). "
            "⑤ 초기안 n = 1의 920/850 V 실패(최대 1.016401 < 필요 1.082353)를 보존한다. ⑥ 결과는 FHA 후보 — 스위칭·ZVS·SR·기동은 EX05."
        ),
        customer_ko=(
            "n = 0.93은 연속값이라 실제 권선에서는 13:14(0.9286) 같은 후보를 씁니다. 이 경우 세 운전 corner에서 FHA 해는 유지되고 운전 주파수만 조금 이동하지만, n을 0.935 이상으로 올리면 "
            "920 V 배터리·850 V 링크 corner에서 필요한 이득이 나오지 않습니다. 코어·절연·열 자료가 정해지면 권선수와 2차 공진 부품을 함께 다시 확인하겠습니다."
        ),
        customer_en=(
            "0.93 is a continuous value, so the real winding uses a candidate like 13:14. It keeps an FHA solution at all three corners with a small frequency shift, but raising the ratio above about 0.935 loses the gain needed at the 920 V battery, 850 V link corner. "
            "Once the core, insulation and thermal data are fixed I will re-check the turns together with the secondary resonant parts."
        ),
        questions=[_Q[0], _Q[7]],
        circuit=None,
        textbook=[TB_E04, TB_13, TB_E05],
        reference_presets=["nominal", "retune"],
        claim_limit="FHA 정상 이득(A)과 합성 코어 screen. 스위칭 운전점·ZVS·최종 권선 선정은 주장하지 않는다.",
    ),
    Experiment(
        key="harmonic_copper",
        title="비정현파 권선 손실: 고조파별 R_ac vs R_dc만",
        goal=(
            "P_cu = I_DC²R_DC + Σ I_h²R_AC(hf₀, T)를 교재 [10, 3, 2] A·[20, 40, 60] mΩ로 계산해 2.60 W vs R_dc만 2.26 W(0.34 W 누락)를 재현하고, 실제 주기 파형(FL08 DAB 전류)의 고조파를 "
            "PWL의 정확 Fourier 계수로 구해 합성 Dowell R_ac로 손실을 비교한다. core loss는 재료 자료가 없으므로 MISSING_INPUT."
        ),
        params=[
            Param("wave", "전류 파형", "", "list", kind="choice", choices=[("list", "교재 고조파 목록"), ("dab", "FL08 DAB 1차 전류 (정확 PWL)")], source="ASSUMED", group="파형"),
            Param("I1", "성분 1 RMS", "A", 10.0, "A", vmin=0.0, vmax=1e4, source="TEXTBOOK", group="교재 목록"),
            Param("I2", "성분 2 RMS", "A", 3.0, "A", vmin=0.0, vmax=1e4, source="TEXTBOOK", group="교재 목록"),
            Param("I3", "성분 3 RMS", "A", 2.0, "A", vmin=0.0, vmax=1e4, source="TEXTBOOK", group="교재 목록"),
            Param("R1", "성분 1 R_ac", "Ω", 0.020, "mΩ", vmin=1e-6, vmax=100.0, source="TEXTBOOK", group="교재 목록"),
            Param("R2", "성분 2 R_ac", "Ω", 0.040, "mΩ", vmin=1e-6, vmax=100.0, source="TEXTBOOK", group="교재 목록"),
            Param("R3", "성분 3 R_ac", "Ω", 0.060, "mΩ", vmin=1e-6, vmax=100.0, source="TEXTBOOK", group="교재 목록"),
            Param("Rdc", "R_dc (목록 비교용)", "Ω", 0.020, "mΩ", vmin=1e-6, vmax=100.0, source="TEXTBOOK", group="교재 목록"),
            Param("V1", "V₁", "V", 800.0, "V", vmin=10, vmax=3000, source="TEXTBOOK", group="DAB 파형"),
            Param("V2", "V₂′ = n·V_L (1차 환산)", "V", 800.0, "V", vmin=10, vmax=3000, source="TEXTBOOK", group="DAB 파형"),
            Param("L", "직렬 L", "H", 200e-6, "µH", vmin=1e-7, vmax=0.01, source="TEXTBOOK", group="DAB 파형"),
            Param("fs", "스위칭 주파수 f₀", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", group="DAB 파형"),
            Param("P", "전력", "W", 1500.0, "W", vmin=1, vmax=1e5, source="TEXTBOOK", group="DAB 파형"),
            Param("Rdc20", "1차 권선 R_dc (20 °C)", "Ω", 0.10, "mΩ", vmin=1e-6, vmax=100.0, source="ASSUMED", group="권선 (합성)"),
            Param("T_w", "권선 온도", "°C", 100.0, "°C", vmin=-40.0, vmax=200.0, source="ASSUMED", group="권선 (합성)"),
            Param("h_cond", "foil(등가) 두께 h", "m", 0.1e-3, "mm", vmin=1e-6, vmax=0.01, source="ASSUMED", group="권선 (합성)"),
            Param("layers", "층 수 m", "", 4, "", vmin=1, vmax=40, kind="int", source="ASSUMED", group="권선 (합성)"),
            Param("H_max", "합에 넣는 최대 차수", "", 399, "", vmin=15, vmax=4000, kind="int", source="ASSUMED", group="권선 (합성)"),
        ],
        presets=[
            Preset("textbook", "교재 [10, 3, 2] A · [20, 40, 60] mΩ", {}, "2.60 W vs 2.26 W", ("nominal", "reference")),
            Preset("dab", "FL08 DAB 전류, 4층 0.1 mm", {"wave": "dab"}, "정확 Fourier + 합성 Dowell", ("variant", "reference")),
            Preset("dab_thick", "DAB, foil 0.3 mm·8층 (R_dc 고정 비교)", {"wave": "dab", "h_cond": 0.3e-3, "layers": 8}, "근접효과 지배", ("corner",)),
            Preset("dab_mismatch", "DAB 900 V ↔ 600 V (1차 환산)", {"wave": "dab", "V1": 900.0, "V2": 600.0}, "사다리꼴 → 삼각 성분 증가", ("variant",)),
        ],
        run=run_harmonic_copper,
        model_level="A (정확 Fourier)",
        suggested_change="파형을 교재 목록 → DAB 전류로 바꾸고, foil 두께를 0.1 → 0.3 mm로 바꾼다.",
        suggested={"wave": "dab", "h_cond": 0.3e-3},
        prediction=Prediction(
            "같은 전류에서 전체 RMS에 R_dc만 곱하면 권선 손실은?",
            ["과소평가", "과대평가", "같다", "모르겠다"],
            "과소평가",
            "R_ac(hf₀) ≥ R_dc이고 고조파일수록 커서, 전체 RMS × R_dc는 고조파 성분의 추가 손실을 빠뜨린다. 교재 예는 0.34 W(실제 손실의 13 %, R_dc 값 대비 +15 %), 두꺼운 다층 권선에서는 훨씬 커진다.",
            ["P_harm", "P_dc", "ratio"],
            handcalc=[{"key": "P_harm", "label": "고조파별 손실 합", "unit": "W"}, {"key": "P_dc", "label": "R_dc만", "unit": "W"}],
        ),
        student=(
            "전류가 사인파가 아니면 여러 주파수 성분(고조파)의 합이다. 도체의 교류 저항은 주파수가 높을수록 커지므로(skin·근접효과) 고조파 전류는 같은 크기라도 더 큰 손실을 만든다. "
            "전체 RMS에 직류 저항만 곱하면 이 추가분을 놓친다."
        ),
        expert=(
            "① 선형 권선이면 성분별로 더한다: P = I_DC²R_DC + Σ I_h²R_AC(hf₀, T). ② 교재: 10²·20 + 3²·40 + 2²·60 mΩ = 2.60 W vs 113 A²·20 mΩ = 2.26 W. "
            "③ DAB 전류는 4구간 직선 — 구간별 ∫(y₀ + sτ)e^{−jhωτ}dτ를 닫힌 형태로 적분해 정확한 계수를 얻고 Parseval(시간영역 정확 RMS)과 FFT로 검산한다. 반파 대칭이라 홀수 차수만. "
            "④ R_AC(hf₀, T) = R_DC(T)·F_R(Δ√h, m) (합성 Dowell): 온도가 오르면 R_DC는 늘고 δ도 커져 F_R은 조금 준다. ⑤ 자화전류 때문에 두 권선의 전류 비가 변하면 근접효과 자계 분포가 달라져 단일 R_ac(f)로 부족할 수 있다. "
            "⑥ core loss는 재료 곡선·파형·온도가 필요 — MISSING_INPUT."
        ),
        customer_ko=(
            "권선 손실은 전체 RMS 전류에 직류 저항을 곱하면 과소평가됩니다. 실제 전류 파형의 고조파별로 그 주파수의 AC 저항을 곱해 더해야 하고, 교재 예에서도 실제 손실의 13 %를 놓칩니다. "
            "측정된 R_ac(f)와 권선 온도를 주시면 합성 값 대신 그것으로 다시 계산하겠습니다. 코어 손실은 재료 곡선이 필요해서 지금은 합계에 넣지 않았습니다."
        ),
        customer_en=(
            "Winding loss is underestimated if the total RMS current is multiplied by the DC resistance. Each harmonic has to be multiplied by the AC resistance at its own frequency; even the textbook example misses 13 % of the real loss. "
            "With a measured AC-resistance sweep and the winding temperature I will replace the synthetic values. Core loss needs the material curves, so it is not in the total yet."
        ),
        questions=[_Q[3], _Q[6]],
        circuit=None,
        textbook=[TB_E04, TB_10],
        reference_presets=["textbook", "dab"],
        claim_limit="선형 R_ac 모델의 권선 손실 합 (A). R_ac는 합성, core loss는 MISSING_INPUT.",
    ),
    Experiment(
        key="tolerance_map",
        title="L·C ±5 %와 상관관계: f_r 이동과 운전점 지도",
        goal=(
            "L, C가 모두 +5 %면 f_r 150 → 142.857 kHz, 모두 −5 %면 157.895 kHz, L +5 %·C −5 %면 150.188 kHz(교재 E05)임을 재현하고, 상관계수 ρ를 바꾼 Monte Carlo로 f_r 분포와 "
            "n = 0.93 CLLC의 FHA 운전 주파수(공칭 162.811 kHz)·920/850 V 이득 여유가 어떻게 바뀌는지 본다. 같은 f_r이라도 특성 임피던스가 달라 운전점이 다르다."
        ),
        params=[
            Param("n", "권선비 n", "", CLLC_N, "", vmin=0.3, vmax=3.0, source="TEXTBOOK", group="CLLC"),
            Param("Lr", "L_r (두 tank, 환산)", "H", CLLC_LR, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", group="CLLC"),
            Param("Cr", "C_r (두 tank, 환산)", "F", CLLC_CR, "nF", vmin=1e-10, vmax=1e-5, source="TEXTBOOK", group="CLLC"),
            Param("Lm", "L_m", "H", CLLC_LM, "µH", vmin=1e-6, vmax=1e-1, source="TEXTBOOK", group="CLLC"),
            Param("P", "출력 전력", "W", CLLC_P, "kW", vmin=100.0, vmax=1e5, source="TEXTBOOK", group="CLLC"),
            Param("f_min", "주파수 하한", "Hz", 120e3, "kHz", vmin=1e3, vmax=5e6, source="TEXTBOOK", group="CLLC"),
            Param("f_max", "주파수 상한", "Hz", 210e3, "kHz", vmin=1e3, vmax=5e6, source="TEXTBOOK", group="CLLC"),
            Param("tol_L", "L 공차 ±", "%", 5.0, "%", vmin=0.0, vmax=30.0, source="TEXTBOOK", group="공차"),
            Param("tol_C", "C 공차 ±", "%", 5.0, "%", vmin=0.0, vmax=30.0, source="TEXTBOOK", group="공차"),
            Param("tol_Lm", "L_m 공차 ±", "%", 15.0, "%", vmin=0.0, vmax=50.0, source="TEXTBOOK", source_note="교재: 합성 공차 screen의 시작값", group="공차"),
            Param("rho", "L–C 상관계수 ρ", "", 0.0, "", vmin=-1.0, vmax=1.0, source="ASSUMED", source_note="부품 온도 특성·lot 자료로 정해야 함", group="공차"),
            Param("k_sigma", "공차 한계 = k·σ", "", 3.0, "", vmin=1.0, vmax=6.0, source="ASSUMED", group="Monte Carlo"),
            Param("N_mc", "표본 수", "", 2000, "", vmin=100, vmax=20000, kind="int", source="ASSUMED", group="Monte Carlo"),
            Param("seed", "난수 seed", "", 1, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group="Monte Carlo"),
        ],
        presets=[
            Preset("textbook", "교재 ±5 %, ρ = 0, L_m ±15 %", {}, "142.857 / 157.895 / 150.188 kHz", ("nominal", "reference")),
            Preset("correlated", "같은 방향 드리프트 (ρ = +1)", {"rho": 1.0}, "f_r 산포 최대", ("variant", "reference")),
            Preset("anti", "반대 방향 (ρ = −1)", {"rho": -1.0}, "f_r 고정, 운전점은 이동", ("variant",)),
            Preset("wide", "L·C ±10 %, L_m ±30 %", {"tol_L": 10.0, "tol_C": 10.0, "tol_Lm": 30.0}, "high corner 해 소멸 표본", ("failure",)),
        ],
        run=run_tolerance_map,
        model_level="A (FHA + MC)",
        suggested_change="L–C 상관계수 ρ를 0 → +1로, 그다음 −1로 바꾼다.",
        suggested={"rho": 1.0},
        prediction=Prediction(
            "L과 C 편차가 같은 방향(ρ = +1)이면 f_r의 표준편차는 ρ = 0 대비?",
            ["√2배 커진다", "절반이 된다", "변화 없다", "0이 된다"],
            "√2배 커진다",
            "ln f_r = −½(ln L + ln C) + 상수라 σ = ½√(σ_L² + σ_C² + 2ρσ_Lσ_C). σ_L = σ_C이면 ρ = 0에서 σ/√2, ρ = +1에서 σ, ρ = −1에서 0. 그래도 ρ = −1에서 운전 주파수는 Z₀ = √(L/C) 때문에 움직인다.",
            ["sig_fr", "sig_fop", "p_nosol"],
            handcalc=[{"key": "fr_pp", "label": "L·C 모두 +5 %의 f_r", "unit": "Hz"}, {"key": "fr_pm", "label": "L +5 %, C −5 %의 f_r", "unit": "Hz"}],
        ),
        student=(
            "공진 주파수는 L과 C의 곱으로 정해진다. 두 부품이 같은 방향으로 틀어지면 공진점이 크게 움직이고, 반대 방향이면 거의 그대로다. 그래서 부품 공차뿐 아니라 두 공차가 함께 움직이는지(상관관계)를 알아야 한다. "
            "공진점이 같아도 L/C 비가 다르면 이득 곡선의 모양이 바뀌어 실제 운전 주파수는 달라진다."
        ),
        expert=(
            "① f_r = 1/(2π√(LC)): (1.05, 1.05) → 142.857, (0.95, 0.95) → 157.895, (1.05, 0.95) → 150.188 kHz. ② 로그정규 가정에서 σ_ln f = ½√(σ_L² + σ_C² + 2ρσ_Lσ_C) — MC와 비교. "
            "③ 운전점: n = 0.93 CLLC의 800/800 V 해 162.811 kHz가 corner에 따라 154.894~171.565 kHz로 이동하고, f_r이 같은 두 corner(150.188 kHz)도 162.357 vs 163.721 kHz — Z₀가 Q = Z₀/R_ac′를 바꾼다. "
            "④ 920/850 V corner의 이득 여유는 공칭 0.50 %, L_m ±15 %까지 넣으면 0.1 %대로 줄어 두 해가 가까워진다. ⑤ FHA 결과일 뿐 — ZVS·RMS·제어 기울기·스위칭 검증은 EX05."
        ),
        customer_ko=(
            "공진 L과 C가 각각 ±5 %라도 같은 방향으로 움직이면 공진점이 ±5 % 가까이 이동하고, 반대 방향이면 거의 그대로입니다. 두 부품의 온도 특성과 lot 산포가 함께 움직이는지 자료를 주시면 "
            "운전 주파수 범위와 고전압 corner의 이득 여유를 다시 계산하겠습니다. 지금 결과는 FHA 기준이라 스위칭 검증 전 후보입니다."
        ),
        customer_en=(
            "Even with ±5 % parts, if the resonant inductor and capacitor drift in the same direction the resonance moves by almost 5 %, and if they drift oppositely it barely moves. If you can share whether their temperature coefficients and lot spread are correlated, "
            "I'll recompute the operating frequency range and the gain margin at the high-voltage corner. These are FHA results, candidates before switching verification."
        ),
        questions=[_Q[4]],
        circuit=None,
        textbook=[TB_E05, TB_13, TB_E04],
        reference_presets=["textbook", "correlated"],
        runtime_hint="seconds",
        claim_limit="FHA 운전점과 가정 분포의 Monte Carlo (A). 양산 수율·스위칭 운전점은 주장하지 않는다.",
    ),
    Experiment(
        key="identification",
        title="L_m·누설·capacitance·R_ac 식별 계획: 측정 조건이 값을 바꾼다",
        goal=(
            "합성 transformer 등가회로(누설·자화·권선 capacitance·단락 strap)를 LCR 개방/단락 시험처럼 여자해, self-resonance 근처에서 L이 과대하게 읽히는 것, 다른 권선의 접지 여부가 결과를 바꾸는 것, "
            "저권선 쪽 단락 strap이 n²배로 환산되어 누설을 크게 틀리게 하는 것을 정확 시간영역 해와 phasor 해로 보이고, 식별 계획표를 만든다."
        ),
        params=[
            Param("test", "시험", "", "OC", kind="choice", choices=[("OC", "개방 시험 (L_m)"), ("SC", "단락 시험 (누설)")], source="ASSUMED", group="시험"),
            Param("port", "측정 포트", "", "primary", kind="choice", choices=[("primary", "1차 (50 turns)"), ("secondary", "2차 (3 turns)")], source="ASSUMED", group="시험"),
            Param("conn", "다른 권선의 연결", "", "floating", kind="choice", choices=[("floating", "floating"), ("grounded", "계측 접지에 연결")], source="ASSUMED", group="시험"),
            Param("f_test", "시험 주파수", "Hz", 10e3, "kHz", vmin=10.0, vmax=1e7, source="ASSUMED", group="시험"),
            Param("V_test", "시험 전압 (RMS)", "V", 1.0, "V", vmin=1e-3, vmax=100.0, source="ASSUMED", source_note="일반 LCR 1 V", group="시험"),
            Param("tol_id", "식별 허용 오차 ±", "%", 2.0, "%", vmin=0.1, vmax=50.0, source="ASSUMED", group="시험"),
            Param("Q_min", "최소 Q", "", 10.0, "", vmin=0.1, vmax=1000.0, source="ASSUMED", group="시험"),
            Param("Np", "N_p", "", 50, "", vmin=1, vmax=500, kind="int", source="TEXTBOOK", group="변압기 (합성)"),
            Param("Ns", "N_s", "", 3, "", vmin=1, vmax=500, kind="int", source="TEXTBOOK", group="변압기 (합성)"),
            Param("Lm", "L_m (1차)", "H", 2e-3, "mH", vmin=1e-6, vmax=10.0, source="ASSUMED", group="변압기 (합성)"),
            Param("L_lp", "1차 누설", "H", 2e-6, "µH", vmin=1e-9, vmax=1e-2, source="ASSUMED", group="변압기 (합성)"),
            Param("L_ls", "2차 누설 (1차 환산)", "H", 2e-6, "µH", vmin=1e-9, vmax=1e-2, source="ASSUMED", group="변압기 (합성)"),
            Param("R_p", "1차 저항", "Ω", 0.1, "mΩ", vmin=1e-6, vmax=100.0, source="ASSUMED", group="변압기 (합성)"),
            Param("R_s", "2차 저항 (1차 환산)", "Ω", 0.1, "mΩ", vmin=1e-6, vmax=100.0, source="ASSUMED", group="변압기 (합성)"),
            Param("C_p", "1차 권선 capacitance", "F", 50e-12, "pF", vmin=1e-15, vmax=1e-6, source="ASSUMED", group="변압기 (합성)"),
            Param("C_s", "2차 권선 capacitance (실제)", "F", 1e-9, "nF", vmin=1e-15, vmax=1e-5, source="ASSUMED", group="변압기 (합성)"),
            Param("C_ps", "권선 간 capacitance", "F", 100e-12, "pF", vmin=0.0, vmax=1e-6, source="ASSUMED", group="변압기 (합성)"),
            Param("R_sh", "단락 strap 저항", "Ω", 0.2e-3, "mΩ", vmin=1e-9, vmax=1.0, source="ASSUMED", group="fixture"),
            Param("L_sh", "단락 strap 인덕턴스", "H", 20e-9, "nH", vmin=1e-12, vmax=1e-5, source="ASSUMED", group="fixture"),
            Param("Ae", "A_e", "mm²", 250.0, "mm²", vmin=1.0, vmax=1e5, source="TEXTBOOK", group="변압기 (합성)"),
            Param("B_op", "운전 B_pk (비교용)", "T", 0.16, "T", vmin=1e-3, vmax=2.0, source="TEXTBOOK", source_note="FL07/FL08 800 V·100 kHz", group="변압기 (합성)"),
        ],
        presets=[
            Preset("nominal", "개방 시험 10 kHz, 2차 floating", {}, "유효 창 안", ("nominal", "reference")),
            Preset("near_srf", "개방 100 kHz, 2차 접지", {"f_test": 100e3, "conn": "grounded"}, "SRF 근처 L 과대", ("failure",)),
            Preset("sc_lv_short", "단락: 1차 측정, 2차 strap 단락", {"test": "SC", "f_test": 100e3}, "strap이 n²배", ("failure",)),
            Preset("sc_hv_short", "단락: 2차 측정, 1차 단락", {"test": "SC", "port": "secondary", "f_test": 100e3}, "strap 영향 작음", ("variant", "reference")),
            Preset("sc_low_f", "단락 2 kHz (Q 부족)", {"test": "SC", "port": "secondary", "f_test": 2e3}, "L이 R에 묻힌다", ("corner",)),
        ],
        run=run_identification,
        model_level="C + A",
        suggested_change="시험 주파수를 10 kHz → 100 kHz로, 2차 연결을 floating → 계측 접지로 바꾼다.",
        suggested={"f_test": 100e3, "conn": "grounded"},
        prediction=Prediction(
            "100 kHz에서 2차를 계측 접지에 묶고 개방 시험을 하면 읽히는 L은?",
            ["거의 정확", "크게 읽힌다", "작게 읽힌다", "모르겠다"],
            "크게 읽힌다",
            "포트에 걸린 권선 capacitance(C_p + C_ps)가 L_m과 병렬 공진(약 290 kHz)을 만들어 L_app ≈ L/(1 − (f/f_SRF)²)가 10 % 넘게 크게 읽힌다. SRF의 1/10 이하에서 측정하고 연결 조건을 기록한다.",
            ["err", "f_srf", "L_app"],
            handcalc=[{"key": "f_srf", "label": "self-resonance", "unit": "Hz"}, {"key": "L_strap", "label": "strap의 반대쪽 환산값", "unit": "H"}],
        ),
        student=(
            "LCR 미터가 보여주는 인덕턴스는 측정 주파수·연결·fixture에 따라 달라진다. 권선에는 작은 capacitance도 있어서 높은 주파수에서는 인덕턴스와 공진해 값이 부풀려진다. "
            "단락 시험에서 단락선 자체의 인덕턴스도 권선비의 제곱만큼 커져 보일 수 있다."
        ),
        expert=(
            "① 개방 시험: Z = [jωC_a + 1/(R_a + jωL_a + Z_m∥Z_b)]⁻¹, L_app = Im Z/ω ≈ L/(1 − ω²LC) — SRF/10 이하에서 측정. ② 다른 권선을 계측 접지에 묶으면 C_ps가 포트에 병렬로 들어가 SRF가 내려간다(lumped 근사). "
            "③ 단락 시험: 저권선(2차) 쪽 strap L_sh는 1차에서 n²L_sh(50:3이면 278배: 20 nH → 5.6 µH)로 보인다. 고권선 쪽을 단락하고 저권선 쪽에서 재면 strap이 1/n²로 줄지만 그쪽 fixture 보정이 중요해진다. "
            "④ 저주파에서는 Q = ωL/R가 작아 L 정확도가 떨어진다 — 유효 창 = Q ≥ Q_min ∩ |오차| ≤ 허용. ⑤ LCR 1 V의 B는 운전 B의 약 1 %: 저 B 투자율로 잰 L_m이 운전 B에서 같다는 보장이 없다(재료 μ(B) — MISSING_INPUT). "
            "⑥ 사인 전원을 진동자 상태로 둔 정확 시간영역 해와 phasor ladder가 일치한다."
        ),
        customer_ko=(
            "L_m과 누설 측정값을 비교하려면 측정 주파수, 전압, fixture 보정, 다른 권선의 연결(개방·단락·접지)을 같이 기록해 주셔야 합니다. 100 kHz에서 2차를 접지에 묶고 잰 개방 인덕턴스는 권선 capacitance 때문에 "
            "10 % 이상 크게 나올 수 있고, 2차 단락선의 인덕턴스는 1차에서 권선비 제곱만큼 커 보입니다. 가능하면 SRF sweep과 고권선 쪽 단락 측정도 부탁드립니다."
        ),
        customer_en=(
            "To compare magnetizing and leakage values I need the test frequency, voltage, fixture compensation and how the other winding was connected: open, shorted or grounded. An open-circuit inductance measured at 100 kHz with the secondary grounded can read more than 10 % high because of winding capacitance, "
            "and a shorting strap on the low-turns side looks n² times larger from the primary. If possible, please add a resonance sweep and a short on the high-turns side."
        ),
        questions=[_Q[5]],
        circuit="ex04_id",
        textbook=[TB_E04, TB_10],
        reference_presets=["nominal", "sc_hv_short"],
        claim_limit="합성 등가회로의 측정 조건 영향 (C·A). 실제 transformer 값·측정기 정확도는 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="EX04",
    title="transformer 설계 closure — 전기적으로 가능한 n·L을 실제 부품으로",
    title_en="Transformer design closure: from an electrical n·L to a real part",
    track="expert",
    order=4,
    path_note="E13 2회전 (E04–E05)",
    textbook=[TB_E04, TB_E05, TB_13, TB_10, TB_E13],
    prerequisites=["FL07", "FL08", "FL10"],
    summary="정수 권선·B·window·구리 screen (DAB 50:3, CLLC 0.93 → 13:14) → 고조파별 권선 손실 → L·C 공차와 상관관계의 운전점 영향 → L_m·누설·C·R_ac 식별 계획. 재료·절연·열 자료 없이 최종 선택을 하지 않는다.",
    experiments=EXPERIMENTS,
    minimum_scope=(
        "n 후보 3개 이상, core/geometry 가정, B/window screening, loss breakdown, L_m/L_σ/C_ps/R_ac source status, 공차 민감도, 공급사 질문 (E04 산출물); "
        "window 168.329 mm² (정확 168.3235), 2.60 W vs 2.26 W, 142.857/157.895/150.188 kHz"
    ),
    claim_limits=[
        "코어·권선·절연 값은 합성 가정 — 최종 transformer 선정 아님",
        "정밀 core loss는 MISSING_INPUT",
        "CLLC 결과는 FHA(A) 후보 — 스위칭 검증은 EX05",
        "R_ac는 합성 Dowell, 식별 회로는 집중 capacitance 근사",
    ],
    test_paths=["tests/test_ex04.py"],
    extends=["FL07", "FL08", "FL10"],
)
