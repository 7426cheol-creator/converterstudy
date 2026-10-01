"""FL11 - PSFB, HV-LV and the 12 V extension: explain the choice (textbook ch.14, with ch.11 DAB).

Physics:
  C  exact switched-affine PSFB (engine/switched.py): primary full bridge with a
     leading and a lagging leg, primary-referred leakage L_k, ideal transformer
     n = Np/Ns, full-bridge or centre-tap rectifier (synchronous rectifiers with
     diode-emulation timing, or diodes with V_f), output L_o, C_o and a resistive
     load.  Rectifier commutation is the interval in which both rectifier paths
     conduct (secondary shorted) until the leakage current has reversed: that is
     the duty loss.  Resistances (primary path, secondary winding, SR R_DS(on),
     output path) are inside the state equations, so the energy ledger is coupled.
  A  closed forms (reference/psfb.py): ideal V_o = V_in D/n, duty loss
     4 L_k f_s I/(n V_in), a constant-V_o piecewise-linear steady state, DAB SPS
     closed forms and an independent PWL integration, path-loss budgets.
Independent paths: PWL constant-V_o solution, L_k volt-second identity, a
hand-written ODE integrated with RK45 and events, the energy ledger.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from ..engine.periodic import cycle_to_steady, shoot
from ..engine.switched import AffineMode, Guard, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close
from ..model.units import fmt_si
from ..reference import psfb as ref
from ._common import bands_from_traj, decimate_minmax, energy_ledger, ledger_check, sym_linear

TB_14 = TextbookRef("psfbhv-lv12-v-확장-선택의-이유를-설명하기-fl11", "14. PSFB·HV-LV·12 V 확장 [FL11]")
TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB — 식에서 파형으로 [FL08]")
TB_19 = TextbookRef("교재실습-계약과-통과-기준", "19. 교재–실습 계약과 통과 기준")

BRIDGE_V = {"pos": 1.0, "fwb": 0.0, "neg": -1.0, "fwt": 0.0}
BRIDGE_KO = {"pos": "+V_in (QA·QD)", "fwb": "freewheel 하단 (QB·QD)", "neg": "−V_in (QB·QC)", "fwt": "freewheel 상단 (QA·QC)"}
RECT_KO = {"P": "정류 P", "N": "정류 N", "C": "정류 commutation (양 경로 도통)", "Z": "DCM (정류 차단)"}


def _mode_label(key: str) -> str:
    br, rc = key.split("|")
    return f"{BRIDGE_KO[br]} · {RECT_KO[rc]}"


# ======================================================================================
# Switched PSFB model
# ======================================================================================


class PSFB(HybridSystem):
    """Phase-shifted full bridge with leakage, ideal transformer and a diode-like rectifier.

    x = [i_p, i_Lo, v_C]; v_o = v_C (no ESR).  q = (bridge, rect):
      bridge 'pos' (QA+QD, v_AB = +V_in), 'fwb' (QB+QD, 0), 'neg' (QB+QC, -V_in), 'fwt' (QA+QC, 0);
      rect   'P' (pair 1 conducts, n i_p = i_Lo), 'N' (pair 2, n i_p = -i_Lo),
             'C' (both paths conduct: secondary shorted, i_p free), 'Z' (all off, i = 0).
    In P/N the two inductors are in series through the transformer (L_k/n^2 + L_o); i_p is
    carried along on the invariant subspace n i_p = +-i_Lo and re-pinned by guard resets.
    Leg A (QA/QB) ends each power interval: the leading leg.  Leg B (QC/QD) ends each
    freewheel interval: the lagging leg.
    """

    state_names = ("i_p", "i_Lo", "v_C")
    state_units = ("A", "A", "V")

    def __init__(self, p: dict, phi_of_cycle=None):
        self.p = dict(p)
        self.phi_of_cycle = phi_of_cycle  # optional soft-start ramp: cycle index -> phi [rad]
        self.Vin = float(p["Vin"])
        self.n = float(p["n"])
        self.fs = float(p["fs"])
        self.T = 1.0 / self.fs
        self.phi = math.radians(float(p["phi_deg"]))
        self.tphi = self.phi / (2 * math.pi) * self.T
        self.Lk = float(p["Lk"])
        self.Lo = float(p["Lo"])
        self.C = float(p["Co"])
        self.R = float(p["R"])
        self.Rp = float(p.get("R_pri", 0.0))
        self.Rw = float(p.get("R_w", 0.0))
        self.Rsr = float(p.get("R_sr", 0.0))
        self.RLo = float(p.get("R_Lo", 0.0))
        self.rect = p.get("rect", "fb")
        dev = p.get("rect_dev", "sr")
        self.Vf = float(p.get("Vf", 0.0)) if dev == "diode" else 0.0
        fb = self.rect == "fb"
        self.nser = 2 if fb else 1  # rectifier devices in series in P/N
        self.Vfeq = self.nser * self.Vf
        self.kv = 1.0 if fb else 2.0  # guard scaling of v_s
        self.kc = 1.0 if fb else 0.5  # C-mode reflected resistance factor
        self.RoutC = self.Rsr if fb else 0.5 * (self.Rw + self.Rsr)
        self.RsP = self.Rw + self.nser * self.Rsr
        self.Leq = self.Lk / self.n**2 + self.Lo
        self.Rloop = self.Rp / self.n**2 + self.RsP + self.RLo
        self.eps = 1e-9 * max(1.0, self.Vin / self.n)  # forward-bias margin for leaving DCM (round-off guard)
        items = sorted((k, v) for k, v in self.p.items() if k != "phi_deg")  # modes do not depend on phi
        self._key = "psfb|" + "|".join(f"{k}={v}" for k, v in items)
        self._modes: dict = {}

    # ---- helpers ------------------------------------------------------------------
    def vab(self, br: str) -> float:
        return BRIDGE_V[br] * self.Vin

    def _pn_rows(self, br: str, s: float):
        """di_Lo/dt row (A1, b1) for P (s = +1) or N (s = -1)."""
        V = self.vab(br)
        a = np.array([0.0, -self.Rloop / self.Leq, -1.0 / self.Leq])
        b = (s * V / self.n - self.Vfeq) / self.Leq
        return a, b

    def vs_row(self, q) -> np.ndarray:
        """Secondary winding EMF v_s (dot positive) as an affine row over z."""
        br, rc = q
        V = self.vab(br)
        n = self.n
        if rc in ("P", "N"):
            s = 1.0 if rc == "P" else -1.0
            a, b = self._pn_rows(br, s)
            # v_s = (V - R_p i_p - L_k di_p/dt)/n with di_p/dt = (s/n) di_Lo/dt
            c = np.zeros(4)
            c[0] = -self.Rp / n
            c[1:3] = -(self.Lk * s / n**2) * a[1:3]
            c[3] = V / n - (self.Lk * s / n**2) * b
            return c
        if rc == "C":
            return np.array([self.kc * (self.Rw + self.Rsr) * n, 0.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 0.0, V / n])

    # ---- continuous dynamics --------------------------------------------------------
    def mode(self, q) -> AffineMode:
        m = self._modes.get(q)
        if m is not None:
            return m
        br, rc = q
        V = self.vab(br)
        n, C, R = self.n, self.C, self.R
        A = np.zeros((3, 3))
        b = np.zeros(3)
        A[2, 1] = 1.0 / C
        A[2, 2] = -1.0 / (R * C)
        if rc in ("P", "N"):
            s = 1.0 if rc == "P" else -1.0
            a1, b1 = self._pn_rows(br, s)
            A[1, :] = a1
            b[1] = b1
            A[0, :] = (s / n) * a1
            b[0] = (s / n) * b1
        elif rc == "C":
            A[0, 0] = -(self.Rp + n * n * self.kc * (self.Rw + self.Rsr)) / self.Lk
            b[0] = V / self.Lk
            A[1, 1] = -(self.RoutC + self.RLo) / self.Lo
            A[1, 2] = -1.0 / self.Lo
            b[1] = -self.Vfeq / self.Lo
        else:  # Z: rectifier blocks, both inductor currents are zero
            A[2, 1] = 0.0
        m = AffineMode(f"{self._key}|{br}|{rc}", A, b, label=f"{br}|{rc}")
        self._modes[q] = m
        return m

    def _other_pair_row(self, q) -> np.ndarray:
        """> 0 while the idle rectifier path stays reverse biased (P/N modes)."""
        rc = q[1]
        s = 1.0 if rc == "P" else -1.0
        c = s * self.kv * self.vs_row(q)
        c[1] -= self.Rw + self.Rsr
        return c

    def guards(self, q):
        br, rc = q
        n = self.n

        def to_zero(z):
            z = z.copy()
            z[0] = 0.0
            z[1] = 0.0
            return z

        if rc in ("P", "N"):
            return [
                Guard("idle 정류 경로 순바이어스 → commutation", self._other_pair_row(q), -1, lambda qq: (qq[0], "C")),
                Guard("i_Lo = 0 → DCM", np.array([0.0, 1.0, 0.0, 0.0]), -1, lambda qq: (qq[0], "Z"), to_zero),
            ]
        if rc == "C":

            def pin_p(z):
                z = z.copy()
                z[0] = z[1] / n
                return z

            def pin_n(z):
                z = z.copy()
                z[0] = -z[1] / n
                return z

            return [
                Guard("경로 2 전류 0 → P", np.array([-0.5 * n, 0.5, 0.0, 0.0]), -1, lambda qq: (qq[0], "P"), pin_p),
                Guard("경로 1 전류 0 → N", np.array([0.5 * n, 0.5, 0.0, 0.0]), -1, lambda qq: (qq[0], "N"), pin_n),
            ]
        V = self.vab(br)
        if V != 0.0:
            s = 1.0 if V > 0 else -1.0
            return [Guard("정류 순바이어스 (DCM 종료)", np.array([0.0, 0.0, -1.0, s * V / n - self.Vfeq - self.eps]), +1, lambda qq, s=s: (qq[0], "P" if s > 0 else "N"))]
        return []

    def gate_schedule(self, t0: float, t1: float):
        T = self.T
        ev = []
        k0 = int(math.floor(t0 / T + 1e-9)) - 1
        k1 = int(math.ceil(t1 / T + 1e-9))
        for k in range(k0, k1 + 1):
            tp = self.tphi if self.phi_of_cycle is None else self.phi_of_cycle(k) / (2 * math.pi) * T
            for t, br in ((k * T, "pos"), (k * T + tp, "fwb"), (k * T + T / 2, "neg"), (k * T + T / 2 + tp, "fwt")):
                if t0 - 1e-15 <= t < t1:
                    ev.append((t, lambda q, br=br: (br, q[1])))
        return ev

    def after_event(self, q, z):
        br, rc = q
        tol = 1e-12 * max(1.0, self.Vin / self.n)
        for _ in range(4):
            if rc == "Z":
                V = self.vab(br)
                if V != 0.0 and abs(V) / self.n - z[2] - self.Vfeq > max(tol, self.eps):
                    rc = "P" if V > 0 else "N"
                    continue
                break
            if rc in ("P", "N"):
                g = float(self._other_pair_row((br, rc)) @ z)
                if g < -tol:
                    rc = "C"
                    continue
                if z[1] <= 1e-12:
                    d = float(self.mode((br, rc)).derivative(z)[1])
                    if d <= 0:
                        rc = "Z"
                        continue
                break
            break
        return (br, rc)

    # ---- outputs, energy ------------------------------------------------------------
    def outputs(self, q):
        br, rc = q
        n = self.n
        V = self.vab(br)
        e_p = np.array([1.0, 0.0, 0.0, 0.0])
        e_l = np.array([0.0, 1.0, 0.0, 0.0])
        e_v = np.array([0.0, 0.0, 1.0, 0.0])
        one = np.array([0.0, 0.0, 0.0, 1.0])
        zero = np.zeros(4)
        vs = self.vs_row(q)
        if rc == "P":
            vr = vs - self.RsP * e_l - self.Vfeq * one
            i1, i2 = e_l, zero
        elif rc == "N":
            vr = -vs - self.RsP * e_l - self.Vfeq * one
            i1, i2 = zero, e_l
        elif rc == "C":
            vr = -self.Vfeq * one - self.RoutC * e_l
            i1 = 0.5 * e_l + 0.5 * n * e_p
            i2 = 0.5 * e_l - 0.5 * n * e_p
        else:
            vr = e_v.copy()
            i1, i2 = zero, zero
        sA = br in ("pos", "fwt")
        sB = br in ("fwb", "neg")
        sC = br in ("fwt", "neg")
        sD = br in ("pos", "fwb")
        return {
            "iP": e_p,
            "iLo": e_l,
            "vo": e_v,
            "vab": V * one,
            "vs": vs,
            "nvs": n * vs,
            "vrect": vr,
            "is": n * e_p,
            "iLo_over_n": e_l / n,
            "iSR1": i1,
            "iSR2": i2,
            "iQA": e_p if sA else zero,
            "iQB": -e_p if sB else zero,
            "iQC": -e_p if sC else zero,
            "iQD": e_p if sD else zero,
            "iin": (1.0 if br == "pos" else -1.0 if br == "neg" else 0.0) * e_p,
            "io": e_v / self.R,
        }

    def stored_energy(self):
        return np.diag([self.Lk, self.Lo, self.C, 0.0])

    def _loss_parts(self, q):
        br, rc = q
        n = self.n
        e_p = np.array([1.0, 0.0, 0.0, 0.0])
        e_l = np.array([0.0, 1.0, 0.0, 0.0])
        Z = np.zeros((4, 4))
        pri = self.Rp * np.outer(e_p, e_p)
        lo = self.RLo * np.outer(e_l, e_l)
        if rc in ("P", "N"):
            sr = self.nser * self.Rsr * np.outer(e_l, e_l) + sym_linear(e_l, self.Vfeq)
            w = self.Rw * np.outer(e_l, e_l)
        elif rc == "C":
            both = np.outer(e_l, e_l) + n * n * np.outer(e_p, e_p)
            if self.rect == "fb":
                sr = self.Rsr * both + sym_linear(e_l, self.Vfeq)
                w = self.Rw * n * n * np.outer(e_p, e_p)
            else:
                sr = 0.5 * self.Rsr * both + sym_linear(e_l, self.Vfeq)
                w = 0.5 * self.Rw * both
        else:
            sr, w, pri, lo = Z, Z, Z, Z
        return pri, sr, w, lo

    def powers(self, q):
        br, _ = q
        e_p = np.array([1.0, 0.0, 0.0, 0.0])
        e_v = np.array([0.0, 0.0, 1.0, 0.0])
        pri, sr, w, lo = self._loss_parts(q)
        return {
            "p_in": sym_linear(e_p, self.vab(br)),
            "p_out": np.outer(e_v, e_v) / self.R,
            "p_loss": pri + sr + w + lo,
            "p_pri": pri,
            "p_sr": sr,
            "p_w": w,
            "p_Lo": lo,
        }

    def describe(self, q):
        return f"{q[0]}|{q[1]}"

    # ---- independent formulation (scalar equations, no matrices) ----------------------
    def rhs_plain(self, br: str, rc: str, x) -> list[float]:
        ip, il, vc = x
        n = self.n
        V = self.vab(br)
        dvc = (il - vc / self.R) / self.C
        if rc == "Z":
            return [0.0, 0.0, -vc / (self.R * self.C)]
        if rc == "C":
            if self.rect == "fb":
                dip = (V - self.Rp * ip - n * n * (self.Rw + self.Rsr) * ip) / self.Lk
                dil = (-2 * self.Vf - self.Rsr * il - self.RLo * il - vc) / self.Lo
            else:
                dip = (V - self.Rp * ip - n * n * (self.Rw + self.Rsr) * ip / 2) / self.Lk
                dil = (-self.Vf - (self.Rw + self.Rsr) * il / 2 - self.RLo * il - vc) / self.Lo
            return [dip, dil, dvc]
        s = 1.0 if rc == "P" else -1.0
        drop = (self.Rw + self.nser * self.Rsr + self.RLo) * il + self.nser * self.Vf
        dil = (s * V / n - self.Rp * il / n**2 - drop - vc) / (self.Lk / n**2 + self.Lo)
        return [s * dil / n, dil, dvc]

    def vs_plain(self, br: str, rc: str, x) -> float:
        ip, il, vc = x
        n = self.n
        V = self.vab(br)
        if rc == "Z":
            return V / n
        if rc == "C":
            return (self.Rw + self.Rsr) * n * ip * (1.0 if self.rect == "fb" else 0.5)
        dip = self.rhs_plain(br, rc, x)[0]
        return (V - self.Rp * ip - self.Lk * dip) / n


# ======================================================================================
# Steady state, regulation and the independent solver path
# ======================================================================================


def psfb_steady(sys: PSFB, tol: float = 1e-11, guess=None):
    """Periodic orbit through the section t = 0 (the '+V_in' edge) by Newton shooting."""
    D = sys.phi / math.pi
    Vo = max(sys.Vin * D / sys.n * 0.95, 1e-3)
    Io = Vo / sys.R
    if guess is None:
        guess = [-Io / sys.n, Io, Vo]
    scales = [max(Io / sys.n, 1e-3), max(Io, 1e-2), max(Vo, 1.0)]
    return shoot(sys, ("fwt", "N"), guess, sys.T, scales=scales, tol=tol)


def run_period(sys: PSFB, x0, periods: int = 2):
    return simulate(sys, ("fwt", "N"), x0, 0.0, periods * sys.T)


def regulate_phi(p: dict, Vo_target: float, phi_max_deg: float, iters: int = 14):
    """Emulate a slow output-voltage loop: find the phase command that gives Vo_target.

    Secant iteration on the exact periodic solution (each step a shooting solve, warm
    started).  Returns (phi_deg, sys, sol, traj, ok, reason).  ``ok`` is False when the
    required phase exceeds ``phi_max_deg`` - a constraint failure, not a numerical one.
    """
    n, Vin = p["n"], p["Vin"]
    Io = Vo_target / p["R"]
    phi = min(180.0 * (Vo_target * n / Vin + ref.psfb_duty_loss(p["Lk"], p["fs"], Io, n, Vin)), phi_max_deg)
    pts = []
    guess = None
    s = sol = tr = None
    for _ in range(iters):
        q = dict(p)
        q["phi_deg"] = phi
        s = PSFB(q)
        sol = psfb_steady(s, guess=guess)
        if not sol.converged:
            return phi, s, sol, None, False, "shooting 미수렴 (SOLVER_FAILED)"
        guess = sol.x0
        tr = run_period(s, sol.x0, 1)
        Vo = tr.mean(0, s.T, "vo")
        pts.append((phi, Vo))
        err = Vo - Vo_target
        if abs(err) < 1e-7 * Vo_target:
            return phi, s, sol, tr, True, ""
        if len(pts) == 1:
            phi_new = phi - err / (Vin / n) * 180.0
        else:
            (pa, va), (pb, vb) = pts[-2], pts[-1]
            slope = (vb - va) / (pb - pa) if pb != pa else Vin / n / 180.0
            phi_new = pb - err / slope
        if phi_new > phi_max_deg:
            if phi >= phi_max_deg - 1e-9:
                return phi, s, sol, tr, False, f"필요한 phase가 상한 {phi_max_deg:g}°를 넘는다 (상한에서 V_o = {Vo:.4g} V < 목표 {Vo_target:g} V)"
            phi_new = phi_max_deg
        phi = max(phi_new, 1.0)
    return phi, s, sol, tr, False, "phase 반복 미수렴"


def _resolve_plain(sys: PSFB, br: str, rc: str, x) -> str:
    """Plain restatement of the rectifier consistency rule after a bridge edge."""
    V = sys.vab(br)
    if rc == "Z":
        if V != 0 and abs(V) / sys.n - x[2] - sys.Vfeq > sys.eps:
            return "P" if V > 0 else "N"
        return "Z"
    if rc in ("P", "N"):
        if x[1] <= 1e-12 and sys.rhs_plain(br, rc, x)[1] <= 0:
            # a conducting path with zero current that would reverse is blocked: DCM, then re-check
            return _resolve_plain(sys, br, "Z", [0.0, 0.0, x[2]])
        s = 1.0 if rc == "P" else -1.0
        if s * sys.kv * sys.vs_plain(br, rc, x) - (sys.Rw + sys.Rsr) * x[1] < 0:
            return "C"
    return rc


def ivp_periods(sys: PSFB, x0, rtol: float, periods: int, rc0: str = "Z"):
    """Hand-written scalar ODE of the same circuit, RK45 with event location, over several periods.

    Shares no matrices, propagators or guard rows with the exact engine: the right-hand
    side, the switching rule and the event functions are restated in plain scalar form.
    """
    T, n = sys.T, sys.n
    x = np.array(x0, dtype=float)
    rc = rc0
    for k in range(periods):
        tp = sys.tphi
        edges = [(k * T, k * T + tp, "pos"), (k * T + tp, k * T + T / 2, "fwb"), (k * T + T / 2, k * T + T / 2 + tp, "neg"), (k * T + T / 2 + tp, (k + 1) * T, "fwt")]
        for ta, tb, br in edges:
            rc = _resolve_plain(sys, br, rc, x)
            t = ta
            hits = 0
            while t < tb - 1e-18 * max(1.0, tb):
                evs = []
                if rc in ("P", "N"):
                    s = 1.0 if rc == "P" else -1.0

                    def g_other(tt, xx, s=s, br=br, rc=rc):
                        return s * sys.kv * sys.vs_plain(br, rc, xx) - (sys.Rw + sys.Rsr) * xx[1]

                    def g_zero(tt, xx):
                        return xx[1]

                    evs = [(g_other, "C", -1), (g_zero, "Z", -1)]
                elif rc == "C":

                    def g_p2(tt, xx):
                        return 0.5 * (xx[1] - n * xx[0])

                    def g_p1(tt, xx):
                        return 0.5 * (xx[1] + n * xx[0])

                    evs = [(g_p2, "P", -1), (g_p1, "N", -1)]
                else:
                    V = sys.vab(br)
                    if V != 0:
                        sg = 1.0 if V > 0 else -1.0

                        def g_on(tt, xx, sg=sg, V=V):
                            return sg * V / n - sys.Vfeq - sys.eps - xx[2]

                        evs = [(g_on, "P" if sg > 0 else "N", +1)]
                funcs = []
                for f, _, d in evs:
                    f.terminal = True
                    f.direction = d
                    funcs.append(f)
                sol = solve_ivp(lambda tt, xx, br=br, rc=rc: sys.rhs_plain(br, rc, xx), (t, tb), x, method="RK45", rtol=rtol, atol=rtol * 1e-3, events=funcs or None)
                hit = None
                for j, te in enumerate(sol.t_events or []):
                    if te.size and te[0] > t + 1e-15 * T and (hit is None or te[0] < hit[0]):
                        hit = (float(te[0]), j, sol.y_events[j][0].copy())
                if hit is None:
                    x = sol.y[:, -1]
                    t = tb
                else:
                    t, j, x = hit
                    rc = evs[j][1]
                    if rc == "P":
                        x[0] = x[1] / n
                    elif rc == "N":
                        x[0] = -x[1] / n
                    elif rc == "Z":
                        x[0] = 0.0
                        x[1] = 0.0
                    hits += 1
                    if hits > 30:
                        raise RuntimeError("event chattering in the RK45 path")
    return x


# ======================================================================================
# Circuit diagram
# ======================================================================================


def psfb_circuit(v: dict, rect: str = "fb") -> Circuit:
    c = Circuit("psfb", 860, 360, title="PSFB (1차 full bridge · 누설 L_k · n:1 · " + ("full-bridge 정류" if rect == "fb" else "center-tap 정류") + ")")
    vin = c.add("vsource", "Vin", 50, 180, 90, "V_in", f"{v['Vin']:g} V", lpos=(34, 176, "end"))
    qa = c.add("nmos", "QA", 140, 100, 90, "QA", lpos=(114, 96, "end"))
    qb = c.add("nmos", "QB", 140, 260, 90, "QB", lpos=(114, 256, "end"))
    qc = c.add("nmos", "QC", 250, 100, 90, "QC", lpos=(276, 96, "start"))
    qd = c.add("nmos", "QD", 250, 260, 90, "QD", lpos=(276, 256, "start"))
    lk = c.add("inductor", "Lk", 330, 150, 0, "L_k", fmt_si(v["Lk"], "H", 3), lpos=(330, 128, "middle"))
    tx = c.add("transformer", "TX", 410, 180, 0, f"n = {v['n']:.4g} : 1", "")
    c.wire("w_vin_top", vin["a"], (50, 40), (140, 40))
    c.wire("w_top_A", (140, 40), qa["a"])
    c.wire("w_top_B", (140, 40), (250, 40), qc["a"])
    c.wire("w_vin_bot", vin["b"], (50, 320), (140, 320))
    c.wire("w_bot_A", qb["b"], (140, 320))
    c.wire("w_bot_B", (140, 320), (250, 320), qd["b"])
    c.wire("w_A", qa["b"], (140, 180), qb["a"])
    c.wire("w_B", qc["b"], (250, 180), qd["a"])
    c.wire("w_A_Lk", (140, 180), (178, 180), (178, 150), lk["a"])
    c.wire("w_Lk_p1", lk["b"], tx["p1"])
    c.wire("w_B_p2", (250, 180), (276, 180), (276, 210), tx["p2"])
    c.dot((140, 40), (140, 320), (140, 180), (250, 180))
    c.text(120, 184, "A", "node")
    c.text(236, 196, "B", "node")
    c.text(118, 346, "leading leg", "small")
    c.text(262, 346, "lagging leg", "small")
    lo = c.add("inductor", "Lo", 660, 60, 0, "L_o", fmt_si(v["Lo"], "H", 3), lpos=(660, 40, "middle"))
    co = c.add("capacitor", "Co", 700, 190, 90, "C_o", fmt_si(v["Co"], "F", 3), lpos=(716, 186, "start"))
    rl = c.add("resistor", "R", 780, 190, 90, "R_load", fmt_si(v["R"], "Ω", 4), lpos=(796, 186, "start"))
    if rect == "fb":
        s1 = c.add("diode", "SR1", 500, 110, 270, "SR1", lpos=(486, 114, "end"))
        s2 = c.add("diode", "SR2", 500, 250, 270, "SR2", lpos=(486, 254, "end"))
        s3 = c.add("diode", "SR3", 580, 110, 270, "SR3", lpos=(594, 114, "start"))
        s4 = c.add("diode", "SR4", 580, 250, 270, "SR4", lpos=(594, 254, "start"))
        c.wire("w_m1", s1["a"], (500, 180), s2["b"])
        c.wire("w_m2", s3["a"], (580, 180), s4["b"])
        c.wire("w_s1", tx["s1"], (466, 150), (466, 180), (500, 180))
        c.wire("w_s2", tx["s2"], (540, 210), (540, 180), (580, 180))
        c.wire("w_op1", s1["b"], (500, 60), (580, 60))
        c.wire("w_op3", s3["b"], (580, 60))
        c.wire("w_on2", s2["a"], (500, 300), (580, 300))
        c.wire("w_on4", s4["a"], (580, 300))
        c.dot((500, 180), (580, 180), (580, 60), (580, 300))
        c.text(560, 172, "s−", "node")
        c.text(516, 172, "s+", "node")
        sec_p = ["SR1", "SR4", "w_s1", "w_s2", "w_m1", "w_m2", "w_op1", "w_on4"]
        sec_n = ["SR3", "SR2", "w_s1", "w_s2", "w_m1", "w_m2", "w_op3", "w_on2", "w_op1"]
        sec_c = ["SR1", "SR2", "SR3", "SR4", "w_s1", "w_s2", "w_m1", "w_m2", "w_op1", "w_op3", "w_on2", "w_on4"]
        c.probe("pSR1", "iSR1", 500, 76, "up", "i_SR1,4")
        c.probe("pSR2", "iSR2", 580, 76, "up", "i_SR3,2")
        c.text(540, 344, "SR1·SR4 = 경로 1 (P), SR3·SR2 = 경로 2 (N)", "small")
        c.wire("w_ret", (580, 300), (700, 300), (780, 300))
        ret = ["w_ret"]
    else:
        s1 = c.add("diode", "SR1", 500, 150, 0, "SR1", lpos=(500, 132, "middle"))
        s2 = c.add("diode", "SR2", 500, 250, 0, "SR2", lpos=(500, 276, "middle"))
        c.wire("w_s1", tx["s1"], s1["a"])
        c.wire("w_s2", tx["s2"], (450, 210), (450, 250), s2["a"])
        c.wire("w_op1", s1["b"], (580, 150), (580, 60))
        c.wire("w_op3", s2["b"], (580, 250), (580, 150))
        c.wire("w_ct", (432, 180), (458, 180))
        c.add("ground", "gnd_ct", 458, 180)
        c.dot((580, 150), (432, 180))
        c.text(446, 172, "CT", "node")
        c.text(540, 344, "CT와 부하 귀환은 같은 2차 GND (1차와 절연)", "small")
        sec_p = ["SR1", "w_s1", "w_op1", "w_ct"]
        sec_n = ["SR2", "w_s2", "w_op3", "w_op1", "w_ct"]
        sec_c = ["SR1", "SR2", "w_s1", "w_s2", "w_op1", "w_op3", "w_ct"]
        c.probe("pSR1", "iSR1", 548, 150, "right", "i_SR1")
        c.probe("pSR2", "iSR2", 548, 250, "right", "i_SR2")
        c.wire("w_ret", (700, 300), (780, 300))
        c.add("ground", "gnd_sec", 740, 300)
        ret = ["w_ret"]
    c.wire("w_o_Lo", (580, 60), lo["a"])
    c.wire("w_Lo_out", lo["b"], (700, 60), (780, 60))
    c.wire("w_co_top", (700, 60), co["a"])
    c.wire("w_r_top", (780, 60), rl["a"])
    c.wire("w_co_bot", co["b"], (700, 300))
    c.wire("w_r_bot", rl["b"], (780, 300))
    c.dot((700, 60), (700, 300))
    c.text(720, 52, "v_o", "node")
    c.probe("pLk", "iP", 300, 138, "right", "i_p")
    c.probe("pLo", "iLo", 622, 48, "right", "i_Lo")
    c.probe("pQA", "iQA", 140, 56, "down", "i_QA")
    c.probe("pQD", "iQD", 250, 300, "down", "i_QD")
    prim = ["Lk", "TX", "w_A_Lk", "w_Lk_p1", "w_B_p2"]
    load = ["Lo", "Co", "R", "w_o_Lo", "w_Lo_out", "w_co_top", "w_r_top", "w_co_bot", "w_r_bot"] + ret
    bridge = {
        "pos": (["Vin", "QA", "QD", "w_vin_top", "w_top_A", "w_A", "w_B", "w_bot_B", "w_vin_bot"], "v_AB = +V_in: QA·QD 도통, 입력이 전력을 공급", ["QB", "QC"]),
        "fwb": (["QB", "QD", "w_A", "w_B", "w_bot_A", "w_bot_B"], "v_AB = 0: 하단 freewheel (QB·QD), 1차 전류가 순환", ["QA", "QC", "Vin"]),
        "neg": (["Vin", "QB", "QC", "w_vin_top", "w_top_B", "w_A", "w_B", "w_bot_A", "w_vin_bot"], "v_AB = −V_in: QB·QC 도통", ["QA", "QD"]),
        "fwt": (["QA", "QC", "w_A", "w_B", "w_top_A", "w_top_B"], "v_AB = 0: 상단 freewheel (QA·QC)", ["QB", "QD", "Vin"]),
    }
    rect_parts = {
        "P": (sec_p, "정류 경로 1 도통: n·i_p = i_Lo, L_k와 L_o가 transformer를 통해 직렬"),
        "N": (sec_n, "정류 경로 2 도통: n·i_p = −i_Lo"),
        "C": (sec_c, "commutation: 두 정류 경로가 함께 도통해 2차가 단락 → 출력으로 전압이 가지 않는 duty loss 구간, 누설 전류가 반전 중"),
        "Z": ([], "DCM: 정류 차단, i_Lo = 0, 부하는 C_o가 공급"),
    }
    for br, (b_act, b_txt, b_dim) in bridge.items():
        for rc, (r_act, r_txt) in rect_parts.items():
            act = b_act + (prim if rc != "Z" else []) + r_act + (load if rc != "Z" else ["Co", "R", "w_co_top", "w_r_top", "w_co_bot", "w_r_bot"] + ret)
            c.mode(f"{br}|{rc}", _mode_label(f"{br}|{rc}"), act, f"{b_txt}. {r_txt}.", dim=b_dim + (["Lo", "Lk"] if rc == "Z" else []))
    return c


_BR_SHORT = {"pos": "+V", "fwb": "FW", "neg": "−V", "fwt": "FW"}
_RC_SHORT = {"P": "P", "N": "N", "C": "comm.", "Z": "DCM"}
BAND_LABELS = {f"{b}|{r}": f"{_BR_SHORT[b]} {_RC_SHORT[r]}" for b in BRIDGE_V for r in "PNCZ"}


# ======================================================================================
# Shared parameter sets and measurement helpers
# ======================================================================================


def _psfb_params(extra: list[Param] | None = None, defaults: dict | None = None) -> list[Param]:
    d = {"Vin": 800.0, "n": 12.0, "phi_deg": 129.6, "fs": 100e3, "Lk": 20e-6, "Lo": 10e-6, "Co": 100e-6, "P": 1500.0, "V_nom": 48.0}
    d.update(defaults or {})
    ps = [
        Param("Vin", "입력전압 V_in", "V", d["Vin"], "V", vmin=50, vmax=1500, source="TEXTBOOK", source_note="교재 14장 합성 seed 800 V (11장 HV 범위 550–900 V)"),
        Param("n", "권수비 n = Np/Ns", "", d["n"], "", vmin=0.5, vmax=200, source="TEXTBOOK", source_note="n = 12 합성 seed (center-tap은 반권선 기준)"),
        Param("phi_deg", "명령 phase φ (반주기 중 v_AB = ±V_in 구간)", "deg", d["phi_deg"], "deg", vmin=2, vmax=175, source="TEXTBOOK", source_note="φ = 0.72·180° = 129.6°: 교재 seed D_eff 0.72를 명령값으로 넣은 것", validity_note="D_cmd = φ/180°; 0°는 출력 0, 180°는 최대"),
        Param("fs", "스위칭 주파수 f_s", "Hz", d["fs"], "kHz", vmin=10e3, vmax=1e6, source="ASSUMED", source_note="교재 PSFB 절에 값 없음; 11장 DAB 예제와 같은 100 kHz"),
        Param("Lk", "누설 + 직렬 인덕턴스 L_k (1차 환산)", "H", d["Lk"], "µH", vmin=1e-9, vmax=2e-3, source="ASSUMED", source_note="합성값 (누설 + ZVS용 shim L)"),
        Param("Lo", "출력 인덕터 L_o", "H", d["Lo"], "µH", vmin=1e-8, vmax=1e-2, source="ASSUMED", source_note="합성값"),
        Param("Co", "출력 커패시터 C_o", "F", d["Co"], "µF", vmin=1e-7, vmax=1.0, source="ASSUMED", source_note="합성값 (ESR 없음)"),
        Param("P", "부하 P (저항 R = V_nom²/P 로 정의)", "W", d["P"], "W", vmin=1, vmax=100e3, source="TEXTBOOK", source_note="교재 비교 조건 1.5 kW (모듈당)", description="부하는 저항이다. 출력전압이 V_nom과 다르면 실제 전력도 달라진다."),
        Param("V_nom", "부하 정의 기준전압 V_nom", "V", d["V_nom"], "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="48 V"),
    ]
    return ps + (extra or [])


def _loss_params(defaults: dict | None = None) -> list[Param]:
    d = {"rect": "fb", "rect_dev": "sr", "R_sr": 0.0, "Vf": 0.7, "R_pri": 0.0, "R_w": 0.0, "R_Lo": 0.0}
    d.update(defaults or {})
    return [
        Param("rect", "정류 방식", "", d["rect"], kind="choice", choices=[("fb", "full-bridge 정류 (소자 4개, 2개 직렬 도통)"), ("ct", "center-tap 정류 (소자 2개, 반권선 2개)")], source="TEXTBOOK", source_note="교재: FB·CT·current doubler는 같은 n·같은 loss 식을 공유할 수 없다", group="정류"),
        Param("rect_dev", "정류 소자", "", d["rect_dev"], kind="choice", choices=[("sr", "동기정류 SR (diode-emulation 타이밍, R_DS(on))"), ("diode", "다이오드 (V_f + R)")], source="ASSUMED", group="정류"),
        Param("R_sr", "정류 소자 저항 (위치당, 병렬 포함)", "Ω", d["R_sr"], "mΩ", vmin=0.0, vmax=1.0, source="ASSUMED", source_note="이상 = 0", group="손실(전력단 결합)"),
        Param("Vf", "다이오드 V_f (rect_dev = diode일 때)", "V", d["Vf"], "V", vmin=0.0, vmax=3.0, source="ASSUMED", group="손실(전력단 결합)"),
        Param("R_pri", "1차 경로 저항 (스위치 2개 + 1차 권선)", "Ω", d["R_pri"], "mΩ", vmin=0.0, vmax=10.0, source="ASSUMED", source_note="이상 = 0", group="손실(전력단 결합)"),
        Param("R_w", "2차 권선 저항 (CT는 반권선당)", "Ω", d["R_w"], "mΩ", vmin=0.0, vmax=1.0, source="ASSUMED", source_note="DC 저항; AC 저항 계수는 12 V 실험에서", group="손실(전력단 결합)"),
        Param("R_Lo", "출력 경로 저항 (L_o DCR·busbar·connector·shunt 합)", "Ω", d["R_Lo"], "mΩ", vmin=0.0, vmax=1.0, source="ASSUMED", source_note="이상 = 0", group="손실(전력단 결합)"),
    ]


def _screen_params() -> list[Param]:
    return [
        Param("C_node", "스위치 노드 등가 용량 C_node (소자 2개 + 권선)", "F", 200e-12, "pF", vmin=1e-12, vmax=1e-8, source="ASSUMED", source_note="선형 등가 합성값. 비선형 Coss는 EX02", group="ZVS screen"),
        Param("t_dead", "dead time t_dead", "s", 100e-9, "ns", vmin=1e-9, vmax=2e-6, source="ASSUMED", group="ZVS screen"),
    ]


def _p_from_values(v: dict) -> dict:
    p = {k: v[k] for k in ("Vin", "n", "phi_deg", "fs", "Lk", "Lo", "Co")}
    p["R"] = v["V_nom"] ** 2 / v["P"]
    for k in ("rect", "rect_dev", "R_sr", "Vf", "R_pri", "R_w", "R_Lo"):
        p[k] = v.get(k, {"rect": "fb", "rect_dev": "sr", "Vf": 0.0}.get(k, 0.0))
    return p


def _is_lossless(p: dict) -> bool:
    return p.get("R_sr", 0) == 0 and p.get("R_pri", 0) == 0 and p.get("R_w", 0) == 0 and p.get("R_Lo", 0) == 0 and p.get("rect_dev", "sr") == "sr"


def measure(sys: PSFB, tr, x0) -> dict:
    """Exact per-period quantities of a periodic trajectory that starts at the '+V_in' edge."""
    T = sys.T
    m = {"T": T}
    for k in ("vo", "iLo", "iin"):
        m[k] = tr.mean(0, T, k)
    for k in ("iP", "iSR1", "iSR2", "iQA", "iQB", "iQC", "iQD", "is", "iLo"):
        m[k + "_rms"] = tr.rms(0, T, k)
    lo, hi = tr.extrema(0, T, "iLo")
    m["iLo_min"], m["iLo_max"] = lo, hi
    lo, hi = tr.extrema(0, T, "iP")
    m["iP_pk"] = max(abs(lo), abs(hi))
    lo, hi = tr.extrema(0, T, "iSR1")
    m["iSR_pk"] = hi
    m["tc"] = sum(h for s, off, h in tr.window(0, T / 2) if s.q[1] == "C")
    m["tz"] = sum(h for s, off, h in tr.window(0, T / 2) if s.q[1] == "Z")
    m["I0"] = float(x0[1])
    m["i_lead"] = float(tr.state_at(sys.tphi)[0][0])
    m["i_lag"] = float(tr.state_at(T / 2)[0][0])
    m["D_cmd"] = sys.phi / math.pi
    m["D_eff"] = sys.n * m["vo"] / sys.Vin
    m["dD"] = m["D_cmd"] - m["D_eff"]
    neg = sum(h for s, off, h in tr.window(0, T) if s.q[0] == "fwt")
    m["qa_rev_frac"] = neg / T
    E_in = tr.energy(0, T, "p_in")
    m["P_in"] = E_in / T
    m["P_out"] = tr.energy(0, T, "p_out") / T
    for k in ("p_pri", "p_sr", "p_w", "p_Lo", "p_loss"):
        m[k] = tr.energy(0, T, k) / T
    return m


def zvs_screen(sys: PSFB, m: dict, C_node: float, t_dead: float) -> list[dict]:
    """SCREEN_ONLY: leading leg sees the reflected output inductor (a current source),
    lagging leg sees only L_k (the rectifier shorts the secondary as soon as v_AB leaves 0)."""
    Vin, Lk = sys.Vin, sys.Lk
    out = []
    il = m["i_lead"]
    i_req_lead = C_node * Vin / t_dead
    out.append(
        {
            "leg": "leading (QA/QB)",
            "event": "전력 구간 끝 (+V_in → freewheel)",
            "i": il,
            "sign_ok": il > 0,
            "mechanism": "정류가 P를 유지 → 반사된 L_o가 전류원처럼 노드를 방전 (charge screen)",
            "need": i_req_lead,
            "need_label": "C_node·V_in/t_dead",
            "time": C_node * Vin / il if il > 0 else math.inf,
            "pass": il > 0 and il >= i_req_lead,
        }
    )
    ig = m["i_lag"]
    Z0 = math.sqrt(Lk / C_node)
    i_req_lag = Vin / Z0
    t_sw = math.sqrt(Lk * C_node) * math.asin(min(1.0, Vin / (ig * Z0))) if ig > 0 and ig * Z0 >= Vin else math.inf
    out.append(
        {
            "leg": "lagging (QC/QD)",
            "event": "freewheel 끝 (0 → −V_in)",
            "i": ig,
            "sign_ok": ig > 0,
            "mechanism": "v_AB가 0을 벗어나는 순간 정류가 commutation으로 2차를 단락 → L_k 에너지만 사용 (energy screen)",
            "need": i_req_lag,
            "need_label": "V_in·√(C_node/L_k)",
            "time": t_sw,
            "pass": ig > 0 and ig >= i_req_lag and t_sw <= t_dead,
        }
    )
    return out


def _wave_series(res: Result, sys: PSFB, tr, periods: float = 2.0, suffix: str = ""):
    T = sys.T
    names = {
        "vab": ("v_AB (1차 bridge)", "V"),
        "nvs": ("n·v_s (2차 EMF, 1차 환산)", "V"),
        "vrect": ("v_rect (정류 출력)", "V"),
        "vo": ("v_o", "V"),
        "iP": ("i_p (1차, L_k)", "A"),
        "iLo_over_n": ("i_Lo / n (1차 환산)", "A"),
        "iLo": ("i_Lo (출력 인덕터)", "A"),
        "iSR1": ("i_SR 경로 1 (SR1·SR4 / CT SR1)", "A"),
        "iSR2": ("i_SR 경로 2 (SR3·SR2 / CT SR2)", "A"),
        "iQA": ("i_QA (leading leg 상단)", "A"),
        "iQD": ("i_QD (lagging leg 하단)", "A"),
        "iQC": ("i_QC (lagging leg 상단)", "A"),
    }
    smp = tr.sample(list(names), 0.0, periods * T, per_segment=24)
    for key, (lab, unit) in names.items():
        res.add_series(key + suffix, lab, unit, smp["t"], smp[key])
    return bands_from_traj(tr, 0.0, periods * T, BAND_LABELS)


def _ledger_table(led: dict, m: dict, T: float, title: str = "에너지 원장 (1주기, J)") -> Table:
    return Table(
        "t_ledger",
        title,
        ["항목", "값 [J]", "평균 전력 [W]", "설명"],
        [
            ["E_in", led["E_in"], led["E_in"] / T, "v_AB·i_p 적분 (bridge가 입력에서 받은 에너지)"],
            ["E_out", led["E_out"], led["E_out"] / T, "v_o²/R 적분"],
            ["1차 경로 R", m["p_pri"] * T, m["p_pri"], "R_pri·i_p²"],
            ["정류 소자 (SR/다이오드)", m["p_sr"] * T, m["p_sr"], "R_sr·i² (+ V_f·i) — commutation 중 두 경로 전류 포함"],
            ["2차 권선", m["p_w"] * T, m["p_w"], "R_w·i²"],
            ["출력 경로", m["p_Lo"] * T, m["p_Lo"], "R_Lo·i_Lo²"],
            ["ΔW", led["dW"], led["dW"] / T, "½L_k i_p² + ½L_o i_Lo² + ½C_o v² 변화 (주기해이면 ≈ 0)"],
            ["잔차", led["residual"], led["residual"] / T, "E_in − E_out − E_loss − ΔW"],
        ],
        note="손실은 모두 상태방정식 안의 저항·V_f에서 나온다(전력단 결합). 스위칭 손실(E_on/E_off·Coss·gate)은 이 이상 스위치 모델에 없다.",
    )


def _screen_table(scr: list[dict]) -> Table:
    rows = []
    for s in scr:
        rows.append([s["leg"], s["event"], s["i"], "맞음" if s["sign_ok"] else "반대", s["need"], s["need_label"], (s["time"] * 1e9) if math.isfinite(s["time"]) else "도달 못함", "screen 통과" if s["pass"] else "screen 불통과", s["mechanism"]])
    return Table(
        "t_zvs",
        "ZVS screen (SCREEN_ONLY — 이상 스위치 모델의 ZVS 판정은 NOT_EVALUABLE)",
        ["leg", "전환 사건", "전환 순간 i_p [A]", "전류 방향", "필요 전류 [A]", "기준", "전환 시간 [ns]", "screen", "메커니즘"],
        rows,
        note="선형 C_node 등가·이상 전류원/LC 가정의 필요조건 선별이다. 비선형 Coss·gate 지연·transformer 용량은 EX02 수준의 사건 모델이 필요하다.",
    )


def _sequence_table(sys: PSFB, tr) -> Table:
    rows = []
    for s, off, h in tr.window(0, sys.T):
        key = sys.describe(s.q)
        meaning = {"C": "2차 단락: 누설 전류 반전 중, 출력에 전압 없음 (duty loss)", "P": "경로 1로 전력/환류", "N": "경로 2로 전력/환류", "Z": "정류 차단 (DCM)"}[s.q[1]]
        if s.q[0] in ("fwb", "fwt") and s.q[1] in ("P", "N"):
            meaning = "freewheel: 1차가 bridge에서 단락, 반사 부하전류가 1차에서 순환"
        rows.append([(s.t0 + off) * 1e6, h * 1e9, _mode_label(key), meaning])
    return Table("t_seq", "한 주기의 구간 순서 (+V_in edge 기준)", ["시작 [µs]", "길이 [ns]", "상태", "의미"], rows, note="bridge 상태(스위치 쌍)와 정류 상태(P/N/commutation/DCM)를 따로 적는다. commutation 길이가 duty loss다.")


def _add_wave_plots(res: Result, sys: PSFB, tr, bands, group: str, T: float):
    res.add_plot("p_v", "1차 bridge 전압과 2차 EMF (1차 환산): 차이가 L_k 전압", ["vab", "nvs"], y_label="전압", y_unit="V", bands=bands, window=(0.0, T), group=group, level="C",
                 proved="명령 phase 구간 안에서도 commutation 동안 n·v_s = 0(2차 단락)이라 출력으로 전압이 전달되지 않는다. v_AB − n·v_s가 누설 L_k에 걸린다.",
                 not_yet="이상 스위치라 스위치 노드 dv/dt·링잉·dead time 구간 전압은 없다. 2차 링잉(정류 소자 용량–누설 L 공진)은 모델 밖이다.")
    res.add_plot("p_i", "1차 전류와 반사 부하전류: commutation에서 −i_Lo/n → +i_Lo/n로 반전", ["iP", "iLo_over_n"], y_label="전류 (1차)", y_unit="A", bands=bands, window=(0.0, T), group=group, level="C",
                 proved="정류가 commutation인 동안 i_p는 V_in/L_k 기울기로 반전하고, 반전이 끝나야 한쪽 경로가 전류를 넘겨받는다. freewheel 동안 1차에는 반사 부하전류가 계속 순환한다.",
                 not_yet="자화전류(L_m)는 모델에 없어 1차 전류 = 반사 2차 전류다. lagging leg ZVS에 도움을 주는 자화 에너지는 포함하지 않았다.")
    res.add_plot("p_sr", "정류(SR) 경로 전류: commutation 동안 두 경로가 함께 도통", ["iSR1", "iSR2", "iLo"], y_label="전류 (2차)", y_unit="A", bands=bands, window=(0.0, T), group=group, level="C",
                 proved="commutation 동안 한 경로 전류는 (i_Lo + n i_p)/2로 오르고 다른 경로는 (i_Lo − n i_p)/2로 내려간다. SR 전류의 RMS·peak는 이 파형에서 정확 적분한다.",
                 not_yet="SR turn-off 타이밍 오차·역전류·body diode 도통·역회복은 모델에 없다(diode-emulation 이상 타이밍).")
    res.add_plot("p_rect", "정류 출력전압과 출력전압", ["vrect", "vo"], y_label="전압 (2차)", y_unit="V", bands=bands, window=(0.0, T), group=group, level="C",
                 proved="정류 출력은 commutation 동안 0(손실이 있으면 음수 소량), 전력 구간에는 V_in/n보다 약간 작고(L_k 분압), freewheel에도 0이 아니다. 평균이 V_o다.",
                 not_yet="출력 C의 ESR·ESL, 부하 동특성은 없다.")
    res.add_plot("p_q", "1차 스위치 전류: leading leg와 lagging leg", ["iQA", "iQC"], y_label="전류", y_unit="A", bands=bands, window=(0.0, T), group=group, level="C",
                 proved="freewheel 동안 스위치는 반사 부하전류를 역방향(음수)으로도 흘린다. 네 스위치의 RMS는 CCM에서 거의 같고, 차이는 전환 순간의 전류(ZVS 조건)에 있다.",
                 not_yet="역방향 도통이 채널인지 body diode인지(dead time 동안)는 이 모델이 구분하지 않는다.")


def run_duty_loss(v: dict) -> Result:
    res = Result("FL11", "psfb_duty_loss", "C (+A 손계산·PWL)")
    p = _p_from_values(v)
    lossless = _is_lossless(p)
    reg_ok, reg_why = True, ""
    if v["ctrl"] == "regulate":
        phi, sys, sol, _tr, reg_ok, reg_why = regulate_phi(p, v["Vo_target"], v["phi_max"])
        if sol is None or not sol.converged:
            res.verdict("SOLVER_FAILED", f"regulate 중 shooting 실패: {reg_why}")
            return res
        p["phi_deg"] = phi
    else:
        sys = PSFB(p)
        sol = psfb_steady(sys)
        if not sol.converged:
            res.verdict("SOLVER_FAILED", f"shooting이 수렴하지 않음 (잔차 {sol.residual:.2e})")
            return res
    T = sys.T
    tr = run_period(sys, sol.x0, 2)
    m = measure(sys, tr, sol.x0)
    ccm = m["tz"] == 0.0 and m["iLo_min"] > 0
    n, Vin, fs, Lk = sys.n, sys.Vin, sys.fs, sys.Lk
    Vo, Io = m["vo"], m["iLo"]
    dD_exact = ref.psfb_duty_loss(Lk, fs, m["I0"], n, Vin)
    dD_avg = ref.psfb_duty_loss(Lk, fs, Io, n, Vin)
    seed_limit = dD_avg < 5e-4 and lossless
    Vseed = ref.psfb_ideal_vo(Vin, n, m["D_cmd"])
    res.add_metric("phi", "명령 phase φ", math.degrees(sys.phi), "deg", basis=f"= {sys.phi:.6f} rad; " + ("출력전압 루프(regulate)가 찾은 값" if v["ctrl"] == "regulate" else "개루프 명령"))
    res.add_metric("D_cmd", "명령 duty D_cmd = φ/π", m["D_cmd"], "", basis="반주기 중 v_AB = ±V_in 구간 비율")
    res.add_metric("Vo", "출력전압 V_o (정확 스위칭 해)", Vo, "V", ref=Vseed, ref_label="교재 seed V_in·D/n (L_k → 0 한계)", tol=0.002 if seed_limit else None, basis="DC 평균, 1주기",
                   note="" if seed_limit else f"seed 식과의 차이 {Vseed - Vo:.4g} V = duty loss" + (" + 저항 강하" if not lossless else ""))
    res.add_metric("D_eff", "유효 duty D_eff = n·V_o/V_in", m["D_eff"], "", ref=0.72 if (seed_limit and abs(m["D_cmd"] - 0.72) < 1e-9) else None, ref_label="교재 seed 0.72", tol=1e-3 if seed_limit else None, basis="출력이 실제로 받은 volt-second 비율")
    res.add_metric("dD", "duty loss ΔD = D_cmd − D_eff", m["dD"], "", ref=dD_exact if (lossless and ccm) else None, ref_label="L_k volt-second 항등식 4L_k f_s I_0/(n V_in)", tol=1e-6 if (lossless and ccm) else None, basis="반주기당",
                   note="" if lossless else "저항 강하가 섞여 항등식 비교는 끔 (값만 표시)")
    res.add_metric("dD_avg", "근사식 4L_k f_s I_o/(n V_in) (평균전류 사용)", dD_avg, "", basis="교과서형 평균 근사", note=f"정확값 대비 {dD_avg / max(m['dD'], 1e-15) - 1:+.1%}: commutation 시작 전류 I_0는 평균 I_o보다 리플만큼 작다" if ccm and m["dD"] > 0 else "")
    res.add_metric("t_c", "commutation 구간 t_c (정류 양 경로 도통)", m["tc"], "s", basis="반주기당; 이 동안 2차가 단락")
    res.add_metric("tc_frac", "t_c / (T_s/2)", m["tc"] / (T / 2), "", basis="시간 비율 (ΔD와 거의 같지만 같지 않다: L_k 분압 효과)")
    res.add_metric("Io", "출력 DC 전류 I_o", Io, "A", basis="i_Lo 평균 = 부하전류")
    res.add_metric("I0", "commutation 시작 전류 I_0 = i_Lo(0)", m["I0"], "A", basis="v_AB가 반전하는 순간의 출력 인덕터 전류")
    res.add_metric("dILo", "출력 인덕터 리플 ΔI_pp", m["iLo_max"] - m["iLo_min"], "A", basis="peak-to-peak, 2f_s 리플")
    res.add_metric("R_eq", "duty loss 등가 출력저항 4L_k f_s/n²", ref.psfb_equivalent_output_resistance(Lk, fs, n), "Ω", basis="V_o ≈ V_in D/n − R_eq·I_o (무손실 저항처럼 보임, 열은 없음)")
    res.add_metric("Ip_rms", "1차 전류 RMS", m["iP_rms"], "A", basis="L_k·transformer 1차")
    res.add_metric("Ip_pk", "1차 전류 peak", m["iP_pk"], "A")
    res.add_metric("Isw_rms", "1차 스위치 RMS (QA; 네 스위치 거의 같음)", m["iQA_rms"], "A", basis=f"소자별; QC {m['iQC_rms']:.4g} A")
    res.add_metric("Isr_rms", "SR(정류 소자) RMS, 소자당", m["iSR1_rms"], "A", basis="FB: SR1·SR4가 같은 전류 / CT: 반권선 전류와 같음")
    res.add_metric("Isr_pk", "SR peak", m["iSR_pk"], "A")
    res.add_metric("Is_rms", "2차 권선 RMS" + (" (반권선당)" if sys.rect == "ct" else ""), m["iSR1_rms"] if sys.rect == "ct" else m["is_rms"], "A", basis="FB 권선은 ±i_Lo 교번, CT 반권선은 반주기씩만 도통")
    res.add_metric("V_sr", "정류 소자 blocking 전압 (이상)", (1 if sys.rect == "fb" else 2) * Vin / n, "V", basis="FB: V_in/n, CT: 2V_in/n (링잉·overshoot 제외)")
    res.add_metric("Iin", "입력 DC 전류", m["iin"], "A", basis="bridge 입력 평균")
    res.add_metric("Pout", "출력 전력", m["P_out"], "W")
    if not lossless:
        res.add_metric("eff", "효율 (전력단 결합 손실만)", m["P_out"] / m["P_in"], "", basis="E_out/E_in; 스위칭 손실 미포함")
    res.add_metric("zvs", "ZVS 판정", "NOT_EVALUABLE (이상 스위치 모델)", "", basis="전환 순간 전류 screen은 표 참조 (SCREEN_ONLY)")
    led = energy_ledger(tr, sys, 0.0, T, ["p_in"], ["p_out"], ["p_loss"], rated_power=max(v["P"], 1.0))
    res.add_check(ledger_check(led))
    res.add_check(Check("shooting 주기해 잔차", "PASS" if sol.residual < 1e-9 else "FAIL", sol.residual, "rel", 1e-9, path="+V_in edge 단면의 Poincaré map 고정점 (Newton, 유한차분 Jacobian; 정류 사건 시각 민감도 포함)", detail=f"반복 {sol.iterations}회, Floquet |λ|max = {sol.spectral_radius:.6f}"))
    # independent PWL path (C_o -> infinity), lossless CCM only
    pw = ref.psfb_pwl_steady(Vin, n, fs, sys.phi, Lk, sys.Lo, sys.R) if (lossless and ccm) else None
    if pw is not None:
        res.add_check(check_close("PWL 독립 경로: V_o (C_o→∞ 상수 출력 가정)", Vo, pw["Vo"], 5e-4, "구간별 직선 대수 + load line (reference/psfb.py) vs 정확 스위칭 해", True, "V", detail=f"스위칭 {Vo:.7g} V / PWL {pw['Vo']:.7g} V (차이는 유한 C_o의 출력 리플)"))
        res.add_check(check_close("PWL 독립 경로: commutation 시간 t_c", m["tc"], pw["t_c"], 5e-3, "t_c = 2I_0/(nV_in/L_k + V_o/L_o) vs 사건 위치", True, "s"))
        res.add_check(check_close("PWL 독립 경로: 1차 RMS", m["iP_rms"], pw["ip_rms"], 2e-3, "PWL 정확 적분 vs Kronecker 모멘트 적분", True, "A"))
        res.add_check(check_close("PWL 독립 경로: SR RMS", m["iSR1_rms"], pw["sr_rms"], 2e-3, "PWL 정확 적분 vs Kronecker 모멘트 적분", True, "A"))
        res.add_check(check_close("L_k volt-second 항등식: ΔD = 4L_k f_s I_0/(n V_in)", m["dD"], dD_exact, 1e-6, "반주기 L_k 전압 적분(손유도) vs 시뮬레이션 평균 V_o", True, "", abs_scale=1e-3))
    else:
        res.add_check(Check("PWL 독립 경로 / L_k 항등식", "NOT_RUN", path="무손실·CCM에서만 성립", detail="저항·V_f가 있거나 DCM이면 이 손계산 경로가 적용되지 않는다. 대신 에너지 원장과 RK45 독립 solver로 검증한다."))
    # independent solver path: plain ODE + RK45 with events over a multi-period transient that starts
    # from 0.8 x the periodic state (LC ringing exercises the integrator; no grazing diode events)
    Nck = 20
    x_start = 0.8 * np.asarray(sol.x0)
    tr_ck = simulate(sys, ("fwt", "N"), x_start, 0.0, Nck * T)
    x_ex = tr_ck.z_end[:3]
    sc = np.array([max(abs(Io) / n, 1e-3), max(abs(Io), 1e-2), max(abs(Vo), 1.0)])
    errs = []
    for rt in (1e-6, 1e-8, 1e-10):
        try:
            xr = ivp_periods(sys, x_start, rt, Nck, rc0="N")
            errs.append(float(np.max(np.abs(xr - x_ex) / sc)))
        except Exception as exc:  # reported, not hidden
            errs.append(float("nan"))
            res.warnings.append(f"RK45 경로 실패 (rtol {rt:g}): {exc}")
    okc = all(math.isfinite(e) for e in errs) and errs[-1] < 1e-6 and errs[-1] <= max(errs[0], 1e-9)
    res.add_check(Check(f"독립 solver: 손으로 쓴 ODE + RK45 사건 ({Nck}주기 과도)", "PASS" if okc else "FAIL", errs[-1], "rel", 1e-6, path="스칼라 식·별도 전환 규칙·solve_ivp 사건 vs 행렬지수 정확 해 (주기해의 0.8배에서 시작한 20주기 과도)", independent=True, detail="rtol 1e-6/1e-8/1e-10 오차: " + ", ".join(f"{e:.2e}" for e in errs) + " (0 초기조건 기동에는 SR 전류가 0을 스치는 사건이 있어 허용오차에 민감하므로 쓰지 않음)"))
    # per-cycle steady-state criterion from zero initial state (mathematical start-up, no soft start)
    E_ref = max(abs(m["P_out"]) * T, 0.01 * v["P"] * T, 1e-12)
    hist, _ = cycle_to_steady(sys, ("fwt", "Z"), [0.0, 0.0, 0.0], T, scales=sc, energy_ref=E_ref, tol_state=1e-6, tol_energy=1e-6, max_cycles=3000)
    ncyc = hist.cycles_run
    res.add_check(Check("주기별 정상상태 기준 (상태변화 < 1e-6, |ΔW|/E_ref < 1e-6, 3주기 연속)", "INFO", hist.steady_cycle or ncyc, "주기", path="0 초기조건(soft-start 없음)에서 반복 주기 적분", detail=(f"{hist.steady_cycle}주기에서 기준 충족" if hist.steady_cycle else f"{ncyc}주기 안에 미충족 → 수치는 shooting 해 사용") + f"; 끝 상태와 shooting 해 차이 {float(np.max(np.abs(hist.x_end - sol.x0) / sc)):.1e}"))
    su_n = min(ncyc, 300)
    tr_st = simulate(sys, ("fwt", "Z"), [0.0, 0.0, 0.0], 0.0, su_n * T)
    smp = tr_st.sample(["vo", "iLo"], per_segment=4)
    xs, ys = decimate_minmax(smp["t"], smp["vo"], 1200)
    res.add_series("su_vo", "v_o 기동 (soft-start 없음)", "V", xs, ys)
    xs, ys = decimate_minmax(smp["t"], smp["iLo"], 1200)
    res.add_series("su_iLo", "i_Lo 기동", "A", xs, ys)
    res.add_series("crit_x", "주기별 상태 변화 (정규화)", "", list(range(1, ncyc + 1)), hist.state_change)
    res.add_series("crit_w", "주기별 저장에너지 변화 / E_ref", "", list(range(1, ncyc + 1)), [max(e, 1e-18) for e in hist.energy_change])
    # load sweep at the same phase: duty loss grows with current
    Ps = np.geomspace(0.04, 1.3, 11) * v["P"]
    sw_I, sw_D, sw_Dx = [], [], []
    for Pk in Ps:
        q = dict(p)
        q["R"] = v["V_nom"] ** 2 / Pk
        sk = PSFB(q)
        so = psfb_steady(sk)
        if not so.converged:
            continue
        tk = run_period(sk, so.x0, 1)
        vo_k = tk.mean(0, T, "vo")
        io_k = tk.mean(0, T, "iLo")
        cc = sum(h for s, off, h in tk.window(0, T) if s.q[1] == "Z") == 0.0
        sw_I.append(io_k)
        sw_D.append(n * vo_k / Vin)
        sw_Dx.append(m["D_cmd"] - ref.psfb_duty_loss(Lk, fs, so.x0[1], n, Vin) if (cc and lossless) else float("nan"))
    Ig = np.linspace(0, max(sw_I) * 1.05, 50)
    res.add_series("ld_sim", "D_eff 정확 스위칭 해", "", sw_I, sw_D, style="points")
    res.add_series("ld_avg", "D_cmd − 4L_k f_s I_o/(n V_in) (평균 근사)", "", Ig.tolist(), [m["D_cmd"] - ref.psfb_duty_loss(Lk, fs, i, n, Vin) for i in Ig], dash=True)
    if lossless:
        res.add_series("ld_id", "L_k 항등식 (I_0 사용, CCM)", "", sw_I, sw_Dx, style="points")
    bands = _wave_series(res, sys, tr)
    _add_wave_plots(res, sys, tr, bands, "ss", T)
    res.add_plot("p_load", "부하전류에 따른 유효 duty (같은 명령 phase)", ["ld_sim", "ld_avg"] + (["ld_id"] if lossless else []), x_label="출력 전류 I_o", x_unit="A", y_label="D_eff", y_unit="", kind="xy", level="C+A",
                 hlines=[{"y": m["D_cmd"], "label": f"D_cmd {m['D_cmd']:.3f}"}], vlines=[{"x": Io, "label": "현재 운전점"}],
                 proved="CCM에서 D_eff는 전류에 비례해 줄어든다(duty loss ∝ I). 평균 근사선은 리플만큼 과대평가하고, commutation 시작 전류 I_0를 쓴 항등식은 정확 해와 일치한다. 경부하 DCM에서는 D_eff가 D_cmd보다 커질 수 있다.",
                 not_yet="폐루프가 없으므로 실제 제품에서는 phase가 V_o를 맞추도록 움직인다. 자화전류·2차 링잉은 포함하지 않았다.")
    res.add_plot("p_su", "기동 과도 v_o (0 초기조건, soft-start 없음)", ["su_vo"], y_label="v_o", y_unit="V", level="C",
                 proved="0 초기조건에서 출력 LC가 감쇠 진동하며 주기해로 수렴하는 것을 정확 스위칭 해로 보였다. 정상상태는 아래 주기별 기준으로 판정했다.",
                 not_yet="수학적 기동이다. 실제 soft-start·전류제한·precharge·SR 기동 시퀀스는 없다.")
    res.add_plot("p_crit", "정상상태 판정 기준의 주기별 변화", ["crit_x", "crit_w"], x_label="주기", x_unit="", y_label="정규화 변화", y_unit="", log_y=True, kind="xy", hlines=[{"y": 1e-6, "label": "기준 1e-6"}],
                 proved="정상상태를 마지막 몇 주기가 아니라 주기별 상태·저장에너지 변화 기준으로 판정하고 shooting 해와 대조했다.", not_yet="기준값 1e-6은 학습용 선택이다.")
    res.circuit = {"diagram": psfb_circuit({**v, **p}, sys.rect).to_json(), "intervals": bands, "plot_group": "ss"}
    scr = zvs_screen(sys, m, v["C_node"], v["t_dead"])
    res.tables.append(_ledger_table(led, m, T))
    res.tables.append(_screen_table(scr))
    res.tables.append(_sequence_table(sys, tr))
    if v["ctrl"] == "regulate" and not reg_ok:
        res.verdict("FAIL_CONSTRAINT", f"목표 {v['Vo_target']:g} V를 만들 수 없다: {reg_why}. 이상 seed만으로도 D = {v['Vo_target'] * n / Vin:.3f}가 필요하고, duty loss와 저항 강하가 더해진다.")
    elif v["ctrl"] == "regulate":
        res.verdict("PASS_WITHIN_MODEL", f"V_o = {v['Vo_target']:g} V를 위해 φ = {math.degrees(sys.phi):.3f}° (D_cmd {m['D_cmd']:.4f})가 필요: 이상 seed 0.72보다 duty loss만큼 크다 (이상 스위치 모델)")
    elif not ccm:
        res.verdict("INFO", "이 부하는 DCM이다: 정류가 i_Lo = 0에서 멈추고 commutation(duty loss)이 없어 D_eff가 D_cmd보다 커질 수 있다.")
    else:
        res.verdict("PASS_WITHIN_MODEL", "명령 phase → 유효 duty 손실을 정확 스위칭 해·PWL 독립 경로·L_k 항등식·RK45·에너지 원장으로 확인 (이상 스위치 모델)")
    res.assumptions += [
        "이상 스위치(즉시 전환, dead time 없음), 이상 transformer(자화전류 없음, L_m → ∞), 누설은 1차 환산 L_k 하나",
        "정류: SR은 diode-emulation 이상 타이밍(전류 0에서 차단) + R_DS(on); 다이오드는 V_f + R",
        "부하는 저항 R = V_nom²/P, 출력 C는 ESR 없음; 입력은 이상 전압원",
        "손실은 상태방정식 안의 저항·V_f만 (스위칭·Coss·gate·core 손실 없음)",
    ]
    res.not_valid_for += ["ZVS 판정 (NOT_EVALUABLE; 표는 SCREEN_ONLY)", "2차 정류 소자 링잉·overshoot·역회복", "SR 타이밍 오차·역전류", "자화전류가 있는 1차 전류 파형", "폐루프 동특성 (regulate는 느린 루프의 정상점만)"]
    res.interpretation = (
        f"명령 phase {math.degrees(sys.phi):.1f}°(D_cmd {m['D_cmd']:.3f})가 만들어도, v_AB가 반전하면 누설 L_k의 전류가 −I/n에서 +I/n로 바뀌는 동안(t_c = {m['tc'] * 1e9:.1f} ns) "
        "두 정류 경로가 함께 도통해 2차가 단락되고 출력은 전압을 받지 못한다. 그래서 유효 duty는 "
        f"{m['D_eff']:.4f}, V_o는 {Vo:.3f} V로 seed 식 {Vseed:.3f} V보다 낮다. 손실은 전류에 비례하므로(ΔD = 4L_k f_s I_0/(nV_in)) 무손실인데도 출력저항처럼 보이고, "
        "L_k를 키우면 lagging leg ZVS에는 유리하지만 duty loss와 순환전류가 커진다."
    )
    return res


# ======================================================================================
# Experiment 2: L_k trade-off (duty loss vs lagging-leg ZVS screen vs turns-ratio headroom)
# ======================================================================================


def n_max_for_headroom(Vin_min: float, Vo: float, Io: float, Lk: float, fs: float, D_max: float) -> float:
    """Largest n that still reaches Vo at the low-line corner with D_max (averaged duty-loss relation).

    Vo = (Vin/n) (D_max - 4 Lk fs Io/(n Vin))  ->  Vo n^2 - Vin D_max n + 4 Lk fs Io = 0 (larger root).
    """
    a, b, c = Vo, -Vin_min * D_max, 4 * Lk * fs * Io
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("nan")
    return (-b + math.sqrt(disc)) / (2 * a)


def run_lk_tradeoff(v: dict) -> Result:
    res = Result("FL11", "lk_tradeoff", "C (duty loss·전류) + A (screen·n 여유)")
    p0 = _p_from_values(v)
    Vt = v["V_nom"]
    Lks = np.geomspace(v["Lk_min"], v["Lk_max"], 9)
    rows = []
    L_us, dD, dD_avg, dcmd, ilag, ireq, fmin, nmax, iprms_n, tsw = [], [], [], [], [], [], [], [], [], []
    fails = []
    for Lk in Lks:
        p = dict(p0)
        p["Lk"] = float(Lk)
        phi, sys, sol, tr, ok, why = regulate_phi(p, Vt, v["phi_max"])
        if sol is None or not sol.converged:
            res.verdict("SOLVER_FAILED", f"L_k = {Lk * 1e6:.3g} µH: {why}")
            return res
        if not ok:
            fails.append((Lk, why))
        tr = run_period(sys, sol.x0, 1)
        m = measure(sys, tr, sol.x0)
        scr = zvs_screen(sys, m, v["C_node"], v["t_dead"])
        lag = scr[1]
        # lagging-leg current = I_0/n; the ripple offset (I_o - I_0) is nearly load independent in CCM
        offset = m["iLo"] - m["I0"]
        I_min = sys.n * lag["need"] + offset
        frac = I_min / m["iLo"]
        nm = n_max_for_headroom(v["Vin_min"], Vt, Vt / p["R"], Lk, sys.fs, v["phi_max"] / 180.0)
        pw = ref.psfb_pwl_steady(sys.Vin, nm, sys.fs, math.pi * min(0.999, (Vt * nm / sys.Vin + ref.psfb_duty_loss(Lk, sys.fs, Vt / p["R"], nm, sys.Vin))), Lk, sys.Lo, sys.R) if math.isfinite(nm) else None
        L_us.append(Lk * 1e6)
        dD.append(m["dD"])
        dD_avg.append(ref.psfb_duty_loss(Lk, sys.fs, m["iLo"], sys.n, sys.Vin))
        dcmd.append(m["D_cmd"])
        ilag.append(lag["i"])
        ireq.append(lag["need"])
        fmin.append(frac)
        nmax.append(nm)
        iprms_n.append(pw["ip_rms"] if pw else float("nan"))
        tsw.append(lag["time"])
        rows.append([Lk * 1e6, m["D_cmd"], m["dD"], m["tc"] * 1e9, m["iP_rms"], lag["i"], lag["need"], frac * 100, (lag["time"] * 1e9) if math.isfinite(lag["time"]) else "도달 못함", nm, pw["ip_rms"] if pw else "—"])
    res.add_series("dd_sim", "ΔD 정확 스위칭 해 (48 V 유지)", "", L_us, dD, style="points")
    res.add_series("dd_avg", "4L_k f_s I_o/(n V_in) 근사", "", L_us, dD_avg, dash=True)
    res.add_series("dcmd", "필요한 D_cmd", "", L_us, dcmd)
    res.add_series("ilag", "lagging leg 전환 전류 (정격 부하)", "A", L_us, ilag, style="points")
    res.add_series("ireq", "screen 필요 전류 V_in√(C_node/L_k)", "A", L_us, ireq, dash=True)
    res.add_series("fmin", "lagging leg screen 최소 부하 비율", "", L_us, fmin)
    res.add_series("nmax", f"최대 n (L_k {L_us[0]:.3g} µH의 {nmax[0]:.3g} 대비)", "", L_us, [x / nmax[0] for x in nmax])
    res.add_series("iprms", f"그 n에서 공칭 1차 RMS (PWL; {iprms_n[0]:.3g} A 대비)", "", L_us, [x / iprms_n[0] for x in iprms_n])
    Lk_sel = p0["Lk"] * 1e6
    res.add_plot("p_dd", "L_k에 따른 duty loss (48 V 유지; 필요한 명령 duty는 표)", ["dd_sim", "dd_avg"], x_label="L_k", x_unit="µH", y_label="duty", y_unit="", kind="xy", log_x=True, level="C+A", vlines=[{"x": Lk_sel, "label": "현재 L_k"}],
                 proved="L_k가 커지면 commutation이 길어져 duty loss가 비례해 늘고, 같은 48 V를 위해 명령 duty가 커진다(정확 스위칭 해).",
                 not_yet="자화전류·2차 링잉·스위칭 손실은 없다.")
    res.add_plot("p_zvs", "lagging leg ZVS screen: 가진 전류 vs 필요한 전류", ["ilag", "ireq"], x_label="L_k", x_unit="µH", y_label="전류 (1차)", y_unit="A", kind="xy", log_x=True, level="A (SCREEN_ONLY)", vlines=[{"x": Lk_sel, "label": "현재 L_k"}],
                 proved="lagging leg는 L_k 에너지만으로 노드를 옮겨야 하므로 필요한 전류가 1/√L_k로 줄어든다. 전환 순간의 전류는 부하가 정하고 L_k에는 거의 무관하다.",
                 not_yet="선형 C_node 에너지 screen이다. 비선형 Coss, 자화전류, dead time 중 전류 변화(EX02)는 포함하지 않았다. ZVS PASS가 아니다.")
    res.add_plot("p_fmin", "lagging leg screen을 통과하는 최소 부하 (정격 대비)", ["fmin"], x_label="L_k", x_unit="µH", y_label="최소 부하 / 정격", y_unit="", kind="xy", log_x=True, level="A (SCREEN_ONLY)", hlines=[{"y": 1.0, "label": "정격"}], vlines=[{"x": Lk_sel, "label": "현재 L_k"}],
                 proved="경부하 ZVS 범위를 넓히려면 L_k가 커야 한다. 1보다 크면 정격 부하에서도 screen을 통과하지 못한다.",
                 not_yet="CCM 선형 외삽(I_0 = I_o − 리플 오프셋)이다. DCM 경부하에서는 전환 전류가 0이 된다.")
    res.add_plot("p_n", "대가: 저전압 corner 여유를 지키려면 n을 낮춰야 하고 1차 전류가 는다", ["nmax", "iprms"], x_label="L_k", x_unit="µH", y_label="가장 작은 L_k 대비 비율", y_unit="", kind="xy", log_x=True, level="A (평균 duty loss + PWL)", vlines=[{"x": Lk_sel, "label": "현재 L_k"}], hlines=[{"y": 1.0, "label": "기준"}],
                 proved=f"V_in {v['Vin_min']:g} V corner에서 φ 상한으로 48 V를 만들려면 duty loss만큼 n을 낮춰야 하고, 그만큼 반사 전류(1차 RMS·순환전류)가 커진다.",
                 not_yet="평균 duty loss 식과 무손실 PWL 경로의 A 수준 계산이다. 실제 권선은 정수 권수이고 window·손실을 다시 봐야 한다.")
    res.tables.append(Table("t_lk", "L_k sweep (정격 부하, 48 V 유지)", ["L_k [µH]", "D_cmd", "ΔD", "t_c [ns]", "1차 RMS [A]", "lagging 전환 전류 [A]", "screen 필요 [A]", "screen 최소 부하 [%]", "lagging 전환 시간 [ns]", f"n 최대 ({v['Vin_min']:g} V corner)", "그 n의 1차 RMS [A]"], rows,
                            note="ΔD·전류는 정확 스위칭 해(C), screen·n 여유는 A 수준. screen은 필요조건 선별(SCREEN_ONLY)이며 ZVS 판정은 이상 스위치 모델에서 NOT_EVALUABLE."))
    # verification of the linear extrapolation at the selected L_k: direct switched run at the predicted minimum load
    p = dict(p0)
    phi, sys, sol, tr, ok, _ = regulate_phi(p, Vt, v["phi_max"])
    m = measure(sys, run_period(sys, sol.x0, 1), sol.x0)
    need = zvs_screen(sys, m, v["C_node"], v["t_dead"])[1]["need"]
    I_min = sys.n * need + (m["iLo"] - m["I0"])
    if 0 < I_min < 1.5 * m["iLo"]:
        q = dict(p)
        q["R"] = Vt / I_min
        ph2, s2, so2, _t2, ok2, _ = regulate_phi(q, Vt, v["phi_max"])
        if so2 is not None and so2.converged:
            m2 = measure(s2, run_period(s2, so2.x0, 1), so2.x0)
            e = (m2["i_lag"] - need) / need
            res.add_check(Check("선형 외삽 검증: 예측 최소 부하에서 직접 스위칭 해", "PASS" if abs(e) < 0.05 else "FAIL", e, "rel", 0.05, path="I_min = n·i_req + (I_o − I_0) 외삽 vs 그 부하로 다시 푼 정확 스위칭 해의 lagging 전환 전류", independent=True,
                                detail=f"예측 {I_min:.3f} A 부하에서 전환 전류 {m2['i_lag']:.4f} A vs 필요 {need:.4f} A"))
    res.add_metric("dD_sel", "현재 L_k의 duty loss (48 V 유지)", m["dD"], "", basis=f"L_k = {p0['Lk'] * 1e6:.3g} µH")
    res.add_metric("fmin_sel", "현재 L_k의 lagging leg screen 최소 부하 비율", (sys.n * need + (m["iLo"] - m["I0"])) / m["iLo"], "", basis="정격 대비; SCREEN_ONLY")
    res.add_metric("Lk_full", "정격 부하에서 lagging screen을 통과하는 최소 L_k", _crossing(L_us, [a - b for a, b in zip(ilag, ireq)]), "µH", basis="가진 전류 = 필요 전류 교점 (로그 보간)")
    res.add_metric("zvs", "ZVS 판정", "NOT_EVALUABLE (이상 스위치) · 표는 SCREEN_ONLY", "")
    # ratio of duty loss between extremes is linear in L_k (sanity: same current)
    slope = (dD[-1] - dD[0]) / (Lks[-1] - Lks[0])
    res.add_check(Check("duty loss 선형성: ΔD/L_k 기울기 vs 4 f_s I_0/(n V_in)", "INFO", slope, "1/H", path="sweep 양 끝 기울기 vs 손계산 계수", detail=f"손계산 계수 ≈ {4 * p0['fs'] * m['I0'] / (p0['n'] * p0['Vin']):.4g} 1/H (I_0가 L_k에 따라 약간 변해 완전히 같지는 않음)"))
    res.circuit = {"diagram": psfb_circuit({**v, **p0}, p0["rect"]).to_json(), "intervals": [], "plot_group": ""}
    res.verdict("PASS_WITHIN_MODEL", "duty loss와 전류는 정확 스위칭 해(C)로 계산")
    res.verdict("SCREEN_ONLY", "lagging leg ZVS 범위는 선형 C_node 에너지 screen — 실제 ZVS PASS가 아니다")
    if fails:
        res.warnings.append("일부 L_k에서 48 V 유지 불가: " + "; ".join(f"{a * 1e6:.3g} µH: {b}" for a, b in fails))
    res.assumptions += ["정격 부하 저항 고정, V_o는 느린 루프가 48 V로 유지(regulate)", "lagging leg screen: ½L_k i² ≥ ½C_node V_in² (선형 C, 자화전류 없음)", f"저전압 corner V_in = {v['Vin_min']:g} V, φ 상한 {v['phi_max']:g}° (ASSUMED)"]
    res.not_valid_for += ["ZVS 보증·turn-on 손실", "정수 권수 transformer 설계", "경부하 DCM에서의 ZVS"]
    res.interpretation = (
        "L_k를 키우면 lagging leg가 쓸 수 있는 에너지(½L_k i²)가 늘어 경부하까지 ZVS screen을 통과한다. 그러나 같은 전류를 반전시키는 데 시간이 더 걸려 duty loss가 L_k에 비례해 늘고, "
        "저전압 corner에서 출력을 유지하려면 권수비를 낮춰야 해 1차·순환 전류가 커진다. ‘PSFB는 ZVS’가 아니라 ‘어느 부하까지, 무엇을 대가로’가 답이다."
    )
    return res


def _crossing(xs, ys) -> float:
    for k in range(len(xs) - 1):
        a, b = ys[k], ys[k + 1]
        if a == 0:
            return float(xs[k])
        if a * b < 0:
            la, lb = math.log(xs[k]), math.log(xs[k + 1])
            return float(math.exp(la + (lb - la) * (0 - a) / (b - a)))
    return float("nan")


# ======================================================================================
# Experiment 3: PSFB vs DAB at the same 800 -> 48 V, 1.5 kW
# ======================================================================================


def run_psfb_vs_dab(v: dict) -> Result:
    res = Result("FL11", "psfb_vs_dab", "C (PSFB 정확 스위칭) + A/PWL (DAB 닫힌 식·구간 적분)")
    Vin, VL, P, fs = v["Vin"], v["V_L"], v["P"], v["fs"]
    T = 1 / fs
    # --- PSFB: regulated to V_L at P ---------------------------------------------------
    p = {"Vin": Vin, "n": v["n"], "phi_deg": 129.6, "fs": fs, "Lk": v["Lk"], "Lo": v["Lo"], "Co": v["Co"], "R": VL**2 / P, "rect": v["rect"], "rect_dev": "sr", "R_sr": 0.0, "Vf": 0.0, "R_pri": 0.0, "R_w": 0.0, "R_Lo": 0.0}
    phi_p, sys, sol, _tr, ok, why = regulate_phi(p, VL, v["phi_max"])
    if sol is None or not sol.converged:
        res.verdict("SOLVER_FAILED", f"PSFB: {why}")
        return res
    if not ok:
        res.verdict("FAIL_CONSTRAINT", f"PSFB가 {VL:g} V·{P:g} W를 만들지 못함: {why}")
    tr = run_period(sys, sol.x0, 2)
    m = measure(sys, tr, sol.x0)
    # --- DAB: textbook closed forms and an independent PWL integration -------------------
    nd, L = v["n_dab"], v["L_dab"]
    V1, V2 = Vin, nd * VL
    try:
        phi_d = ref.dab_phi_for_power(V1, V2, fs, L, P)
    except ValueError as exc:
        res.verdict("NO_SOLUTION", f"DAB: {exc}")
        return res
    cf = ref.dab_currents(V1, V2, fs, L, phi_d)
    ipk_m, irms_m = ref.dab_matched_rms(V1, fs, L, phi_d)
    # independent path: solve P(phi) = P on the PWL power integral
    from scipy.optimize import brentq

    phi_pwl = brentq(lambda ph: ref.dab_pwl(V1, V2, fs, L, ph)["P"] - P, 1e-6, math.pi / 2, xtol=1e-15, rtol=1e-15)
    dw = ref.dab_pwl(V1, V2, fs, L, phi_pwl)
    tb = abs(Vin - 800) < 1e-9 and abs(VL - 48) < 1e-9 and abs(P - 1500) < 1e-9 and abs(fs - 1e5) < 1e-6 and abs(nd - 50 / 3) < 1e-9 and abs(L - 200e-6) < 1e-15
    res.add_metric("dab_phi", "DAB 위상 φ (1.5 kW, SPS)", phi_d, "rad", ref=0.328973 if tb else None, ref_label="교재 0.328973 rad", tol=2e-6, basis=f"= {math.degrees(phi_d):.4f}°; d = φ/π = {phi_d / math.pi:.4f} (Δt/T = φ/2π = {phi_d / (2 * math.pi):.4f}와 다름)")
    res.add_metric("dab_ipk", "DAB 인덕터 전류 peak", dw["ipk"], "A", ref=2.09431 if tb else ipk_m, ref_label="교재 2.09431 A" if tb else "V1φ/(ωL)", tol=5e-6, basis="1차 환산, matched ratio")
    res.add_metric("dab_irms", "DAB 인덕터 전류 RMS", dw["irms"], "A", ref=2.01988 if tb else irms_m, ref_label="교재 2.01988 A" if tb else "I_pk√(1−2φ/3π)", tol=5e-6, basis="1차 (= transformer 1차 RMS)")
    res.add_metric("dab_isec", "DAB 2차 AC RMS n·I_rms", nd * dw["irms"], "A", ref=33.6647 if tb else None, ref_label="교재 33.6647 A", tol=5e-6, basis="2차 권선 실제값")
    res.add_metric("dab_iout", "DAB 출력 DC 전류 P/V_L", P / VL, "A", ref=31.25 if tb else None, ref_label="교재 31.25 A", tol=1e-12, basis="모듈당")
    res.add_metric("dab_pmax", "DAB 최대전력 V1V2/(8 f_s L)", ref.dab_pmax(V1, V2, fs, L), "W", ref=4000.0 if tb else None, ref_label="교재 4 kW", tol=1e-12, basis="SPS, φ = π/2")
    res.add_metric("psfb_phi", "PSFB 명령 phase (48 V 유지)", math.degrees(sys.phi), "deg", basis=f"D_cmd = {m['D_cmd']:.4f}, 이상 seed 0.72 + duty loss")
    res.add_metric("psfb_ip_rms", "PSFB 1차 RMS", m["iP_rms"], "A", basis="L_k·transformer 1차")
    res.add_metric("psfb_ip_pk", "PSFB 1차 peak", m["iP_pk"], "A")
    res.add_metric("psfb_isr_rms", "PSFB SR RMS (소자당)", m["iSR1_rms"], "A")
    res.add_metric("psfb_iout", "PSFB 출력 DC 전류", m["iLo"], "A", ref=P / VL, ref_label="P/V_L", tol=1e-6, basis="regulate 결과")
    res.add_check(check_close("DAB φ: 닫힌 식 vs PWL 전력 적분 근", phi_pwl, phi_d, 1e-9, "P = V1V2φ(1−φ/π)/(ωL) 역산 vs 4구간 PWL의 ∫v2'·i 적분을 brentq로 푼 φ", True, "rad"))
    res.add_check(check_close("DAB RMS: 닫힌 식 vs PWL", dw["irms"], cf["irms"], 1e-9, "i_0·i_φ 식의 두 직선 RMS vs PWL.rms()", True, "A"))
    res.add_check(check_close("DAB 입력·출력 전력 일치 (무손실)", dw["P1"], dw["P"], 1e-9, "∫v1·i vs ∫v2'·i (PWL)", True, "W"))
    res.add_check(check_close("DAB zero-DC 기준해: 전류 평균 0", dw["mean"], 0.0, 1e-9, "반주기 반대칭 초기조건 → 주기 평균", True, "A", abs_scale=max(dw["irms"], 1e-3)))
    pw = ref.psfb_pwl_steady(Vin, sys.n, fs, sys.phi, sys.Lk, sys.Lo, sys.R)
    if pw is not None:
        res.add_check(check_close("PSFB: PWL 독립 경로 1차 RMS", m["iP_rms"], pw["ip_rms"], 2e-3, "PWL 정확 적분 vs 정확 스위칭 해", True, "A"))
    led = energy_ledger(tr, sys, 0.0, T, ["p_in"], ["p_out"], ["p_loss"], rated_power=P)
    res.add_check(ledger_check(led, what="PSFB: "))
    # --- zero-power / backflow and stored energy -----------------------------------------
    smp = tr.sample(["vab", "iP"], 0.0, T, per_segment=60)
    t_arr = np.asarray(smp["t"])
    pin = np.asarray(smp["vab"]) * np.asarray(smp["iP"])
    fw = sum(h for s, off, h in tr.window(0, T) if s.q[0] in ("fwb", "fwt")) / T
    E_back_psfb, t_back = 0.0, 0.0
    for sg, off, h in tr.window(0, T):
        if sg.q[1] != "C":
            continue
        ss = tr.sample(["vab", "iP"], sg.t0 + off, sg.t0 + off + h, per_segment=801)
        pp = np.asarray(ss["vab"]) * np.asarray(ss["iP"])
        E_back_psfb += -float(np.trapezoid(np.minimum(pp, 0.0), ss["t"]))
        t_back += h * float(np.mean(pp < 0))
    back_psfb = t_back / T
    iw = dw["i"]
    t_zero = -dw["i0"] / ((V1 + V2) / L) if dw["i0"] < 0 else 0.0
    back_dab = 2 * t_zero / T
    E_back_dab = 2 * V1 * 0.5 * (-dw["i0"]) * t_zero
    lk_e = 0.5 * sys.Lk * m["iP_pk"] ** 2
    lo_e = 0.5 * sys.Lo * m["iLo_max"] ** 2
    dab_e = 0.5 * L * dw["ipk"] ** 2
    # --- soft-switching screens ---------------------------------------------------------------
    scr = zvs_screen(sys, m, v["C_node"], v["t_dead"])
    i_edge1 = dw["i0"]  # primary edge at theta = 0: needs i < 0 (current out of the rising node)
    i_edge2 = dw["iphi"]  # secondary edge at theta = phi, primary-referred: needs i > 0
    dab_req1 = v["C_node"] * V1 / v["t_dead"]
    dab_req2 = v["C_node_sec"] * VL / v["t_dead"] / nd
    rows = [
        ["변환비 n = Np/Ns", sys.n, nd, "PSFB는 duty로 강압하므로 n이 작다; DAB는 V1 = nV_L 정합"],
        ["제어 변수", f"leg 사이 φ = {math.degrees(sys.phi):.2f}° (D_cmd {m['D_cmd']:.4f})", f"bridge 사이 φ = {phi_d:.6f} rad ({math.degrees(phi_d):.3f}°)", "같은 기호 φ지만 다른 물리량"],
        ["전달 전력 [W]", m["P_out"], dw["P"], "둘 다 1.5 kW (모듈당)"],
        ["1차 RMS [A]", m["iP_rms"], dw["irms"], "transformer 1차 = bridge 전류"],
        ["1차 peak [A]", m["iP_pk"], dw["ipk"], "스위치 turn-off 전류의 상한 성격"],
        ["1차 스위치 RMS (소자당) [A]", m["iQA_rms"], dw["irms"] / math.sqrt(2), "반주기 도통"],
        ["2차 권선 RMS [A]", m["is_rms"] if sys.rect == "fb" else m["iSR1_rms"], nd * dw["irms"], "PSFB는 n·i_p, DAB는 n·i"],
        ["2차 소자 RMS (소자당) [A]", m["iSR1_rms"], nd * dw["irms"] / math.sqrt(2), "PSFB SR vs DAB 2차 active switch"],
        ["출력 DC 전류 [A]", m["iLo"], P / VL, "같다"],
        ["zero-power 구간 비율", fw, 0.0, "PSFB freewheel: 입력전력 0인데 반사 부하전류가 1차에서 순환"],
        ["역방향(backflow) 구간 비율", back_psfb, back_dab, "입력 순간전력 < 0: PSFB는 commutation 전반, DAB는 전류 부호가 늦게 바뀌는 구간"],
        ["backflow 에너지/주기 [µJ]", E_back_psfb * 1e6, E_back_dab * 1e6, "입력으로 되돌아간 에너지"],
        ["직렬/누설 L 최대 저장에너지 [µJ]", lk_e * 1e6, dab_e * 1e6, "PSFB ½L_k i_pk², DAB ½L i_pk²"],
        ["출력 필터 L 저장에너지 [µJ]", lo_e * 1e6, 0.0, "PSFB는 출력 L이 필요 (DAB는 C만)"],
        ["소프트 스위칭 증거 수준", "NOT_EVALUABLE · leg별 전류 screen만 (SCREEN_ONLY)", "NOT_EVALUABLE · edge 전류 부호 screen만 (SCREEN_ONLY)", "이상 스위치 모델은 ZVS를 입증하지 못한다"],
        ["1차 edge 전류 screen", f"leading {scr[0]['i']:.3g} A ({'통과' if scr[0]['pass'] else '불통과'}), lagging {scr[1]['i']:.3g} A ({'통과' if scr[1]['pass'] else '불통과'})", f"i(0) = {i_edge1:.3g} A (부호 {'맞음' if i_edge1 < 0 else '반대'}, 필요 {dab_req1:.3g} A)", "SCREEN_ONLY"],
        ["2차 edge 전류 screen", "SR은 전류 0 근처에서 전환 (diode-emulation)", f"n·i(φ) = {nd * i_edge2:.3g} A (부호 {'맞음' if i_edge2 > 0 else '반대'}, 필요 {dab_req2 * nd:.3g} A)", "SCREEN_ONLY"],
        ["양방향", "불가 (다이오드형 정류)", "가능 (φ 부호)", "요구사항이 결정"],
    ]
    res.tables.append(Table("t_cmp", f"같은 {Vin:g}→{VL:g} V·{P:g} W (모듈당)에서의 비교", ["항목", "PSFB (C, 정확 스위칭)", "DAB SPS (A/PWL)", "설명"], rows,
                            note="전달전력만 같게 둔 비교다. 손실·효율 순위는 소자·자성체 손실 모델 없이는 말하지 않는다. DAB 전체 실험은 FL08."))
    # --- waveforms on one time axis --------------------------------------------------------------
    bands = _wave_series(res, sys, tr, periods=1.0)
    dab_t, dab_i = iw.plot_points()
    res.add_series("dab_i", "DAB i_L (1차)", "A", dab_t, dab_i)
    t1, v1 = dw["v1"].plot_points()
    res.add_series("dab_v1", "DAB v1 (1차 bridge)", "V", t1, v1, dash=True)
    t2, v2 = dw["v2"].plot_points()
    res.add_series("dab_v2", "DAB v2' = n·v_L bridge (1차 환산)", "V", t2, v2, dash=True)
    sec = iw.scaled(nd)
    s2 = dw["v2"]
    ts, ys = [], []
    for k in range(sec.y0.size):
        on = s2.value_at(0.5 * (sec.t[k] + sec.t[k + 1])) > 0
        ts += [float(sec.t[k]), float(sec.t[k + 1])]
        ys += [float(sec.y0[k]) if on else 0.0, float(sec.y1[k]) if on else 0.0]
    res.add_series("dab_isw2", "DAB 2차 switch 전류 (v2' > 0 구간 n·i)", "A", ts, ys)
    res.add_series("psfb_pin", "PSFB 입력 순간전력 v_AB·i_p", "W", t_arr.tolist(), pin.tolist())
    tt = np.linspace(0, T, 801)
    res.add_series("dab_pin", "DAB 입력 순간전력 v1·i", "W", tt.tolist(), [dw["v1"].value_at(x) * iw.value_at(x) for x in tt])
    res.add_plot("c_i", "1차 전류: PSFB i_p vs DAB i_L (같은 1.5 kW, 같은 100 kHz)", ["iP", "dab_i"], y_label="전류 (1차)", y_unit="A", bands=bands, group="cmp", level="C + PWL",
                 proved="PSFB 1차 전류는 부하전류의 반사(사각파 + commutation 경사)이고, DAB는 두 사각파 전압 차이가 만든 사다리꼴/삼각 전류다. 같은 전력에서 RMS·peak·edge 전류가 다르다.",
                 not_yet="자화전류, dead time, 스위칭 전환 파형은 둘 다 없다. 손실·효율 비교가 아니다.")
    res.add_plot("c_v", "bridge 전압: PSFB v_AB (leg 사이 phase) vs DAB v1·v2' (bridge 사이 phase)", ["vab", "dab_v1", "dab_v2"], y_label="전압 (1차)", y_unit="V", bands=bands, group="cmp", level="C + PWL",
                 proved="PSFB는 한 bridge 안의 두 leg 위상으로 ±V_in/0 세 레벨을 만들고, DAB는 두 bridge가 각각 50 % 사각파이며 그 사이 위상이 전력을 정한다. DAB 쪽 위상은 닫힌 식 역산과 PWL 전력 적분의 근이 일치하는 check로 확인했다.",
                 not_yet="두 파형 모두 이상 스위치의 계단 전압이다. dead time 동안 노드 전압이 Coss를 따라 움직이는 구간·ringing·ZVS 여부는 이 그림이 보이지 않는다(PSFB lagging leg ZVS는 zvs_screen의 에너지 screen만, DAB는 EX02/EX06). 같은 위상이라도 두 converter의 제어 변수 의미가 다르다는 것까지만 보였다.")
    res.add_plot("c_sec", "2차 소자 전류: PSFB SR vs DAB 2차 switch", ["iSR1", "dab_isw2"], y_label="전류 (2차)", y_unit="A", bands=bands, group="cmp", level="C + PWL",
                 proved="PSFB SR은 출력 인덕터 전류(거의 DC)를 반주기씩 나르고, DAB 2차 switch는 n배로 커진 교류 인덕터 전류를 나른다.", not_yet="SR 타이밍·역전류, DAB 2차 ZVS 전하는 포함하지 않았다.")
    res.add_plot("c_p", "입력 순간전력: zero-power 구간과 backflow", ["psfb_pin", "dab_pin"], y_label="p_in", y_unit="W", bands=bands, group="cmp", level="C + PWL", hlines=[{"y": 0.0, "label": "0 W"}],
                 proved="PSFB는 freewheel 동안 입력전력이 0이지만 1차 전류가 순환하고, commutation 전반에는 에너지가 입력으로 되돌아간다. DAB는 전류 부호가 늦게 바뀌는 구간에 backflow가 있다. 두 순간전력의 주기 평균이 출력전력과 같다는 것은 DAB ∫v1·i = ∫v2′·i check와 PSFB 에너지 장부 check가 뒷받침한다.",
                 not_yet="backflow 에너지는 무손실 파형에서 잰 순환 에너지일 뿐 손실이 아니다. 그 전류가 만드는 도통·스위칭 손실과 효율 차이는 소자·자성체 자료가 없어 계산하지 않았다(MISSING_INPUT). 입력 capacitor ripple·EMI 영향도 이 그림의 범위 밖이다.")
    res.circuit = {"diagram": psfb_circuit({**v, **p}, sys.rect).to_json(), "intervals": bands, "plot_group": "cmp"}
    if not res.verdicts:
        res.verdict("PASS_WITHIN_MODEL", "같은 800→48 V·1.5 kW에서 PSFB(C)와 DAB(A/PWL)의 전류·전압·zero-power 구간을 비교 (손실·ZVS 판정 아님)")
    res.assumptions += ["PSFB: 무손실 이상 스위치, 느린 루프가 48 V 유지", "DAB: 교재 11장 모듈 (n = 50/3, L = 200 µH 1차 환산), SPS, zero-DC 반대칭 기준해", "두 converter 모두 자화전류 없음"]
    res.not_valid_for += ["효율·손실 순위 (소자·자성체 손실 모델 없음)", "ZVS 판정 (둘 다 NOT_EVALUABLE; screen만)", "DAB의 다른 전압 corner (FL08)"]
    res.interpretation = (
        f"같은 1.5 kW에서 PSFB는 1차 RMS {m['iP_rms']:.3f} A, DAB는 {dw['irms']:.3f} A다. PSFB는 duty로 강압해 n = {sys.n:g}이고 출력 인덕터가 전류를 평탄하게 만들지만, "
        f"freewheel 동안({fw:.1%}) 반사 부하전류가 1차에서 순환하고 commutation 동안 duty를 잃는다. DAB는 n = {nd:.4g}로 전압을 정합하고 직렬 L의 전류로 전력을 보내므로 "
        "출력 L이 없지만 2차 스위치가 교류 전류를 나르고 양방향이 된다. 무엇이 더 좋은지는 요구(양방향·범위·비용)와 손실 모델이 정한다."
    )
    return res


# ======================================================================================
# Experiment 4: 12 V extension - high current, path loss and why paralleling has limits
# ======================================================================================


def lv_budget(Io: float, rect: str, v: dict, N: int) -> dict:
    """A-level path-loss budget (DC + half-period RMS, ripple and commutation overlap neglected)."""
    fs = v["fs"]
    pos = 2 if rect == "ct" else 4  # rectifier positions
    ser = 1 if rect == "ct" else 2  # positions in series during conduction
    Rds = v["R_ds"]
    P_cond = Io**2 * ser * Rds / N  # each position carries Io for half the period
    P_w = Io**2 * v["R_w"] * v["F_R"]  # FB: one winding RMS Io; CT: two halves each Io/sqrt(2)
    R_out = v["R_bus"] + v["R_conn"] + v["R_pcb"] + v["R_shunt"] + v["R_dcr"]
    P_out = Io**2 * R_out
    P_gate = pos * N * v["Q_g"] * v["V_drv"] * fs
    P_oss = pos * N * v["E_oss"] * fs
    d = v["spread"]
    g_hot = 1 / (1 - d)
    share = g_hot / (g_hot + (N - 1)) if N > 1 else 1.0
    I_pos_rms = Io / math.sqrt(2)
    P_hot = (share * I_pos_rms) ** 2 * Rds * (1 - d) + v["Q_g"] * v["V_drv"] * fs + v["E_oss"] * fs
    P_avg_dev = (I_pos_rms / N) ** 2 * Rds + v["Q_g"] * v["V_drv"] * fs + v["E_oss"] * fs
    return {"P_cond": P_cond, "P_w": P_w, "P_out": P_out, "P_gate": P_gate, "P_oss": P_oss, "R_out": R_out, "total": P_cond + P_w + P_out + P_gate + P_oss, "share_hot": share, "P_hot": P_hot, "P_avg_dev": P_avg_dev, "pos": pos, "ser": ser}


def run_lv12(v: dict) -> Result:
    res = Result("FL11", "lv12_extension", "A (경로 손실 budget) + C (결합 손실 스위칭 검산)")
    P, Vh, Vl, Rp = v["P"], v["V_hi"], v["V_lo"], v["R_path"]
    Ih, Lh = ref.dc_path_loss(P, Vh, Rp)
    Il, Ll = ref.dc_path_loss(P, Vl, Rp)
    tb = abs(P - 3000) < 1e-9 and abs(Vh - 48) < 1e-9 and abs(Vl - 12) < 1e-9 and abs(Rp - 1e-3) < 1e-15
    res.add_metric("I_hi", f"{Vh:g} V 출력 전류", Ih, "A", ref=62.5 if tb else None, ref_label="교재 62.5 A", tol=1e-12, basis=f"{P:g} W 총합 DC")
    res.add_metric("I_lo", f"{Vl:g} V 출력 전류", Il, "A", ref=250.0 if tb else None, ref_label="교재 250 A", tol=1e-12, basis=f"{P:g} W 총합 DC")
    res.add_metric("L_hi", f"같은 {Rp * 1e3:g} mΩ 경로 손실 @ {Vh:g} V", Lh, "W", ref=3.90625 if tb else None, ref_label="교재 3.91 W", tol=1e-12, basis="I²R, DC")
    res.add_metric("L_lo", f"같은 {Rp * 1e3:g} mΩ 경로 손실 @ {Vl:g} V", Ll, "W", ref=62.5 if tb else None, ref_label="교재 62.5 W", tol=1e-12, basis="I²R, DC")
    res.add_metric("ratio_I", "전류 비", Il / Ih, "", ref=Vh / Vl, ref_label="V_hi/V_lo", tol=1e-12)
    res.add_metric("ratio_L", "같은 저항의 손실 비", Ll / Lh, "", ref=(Vh / Vl) ** 2, ref_label="(V_hi/V_lo)²", tol=1e-12, basis="전압 1/4 → 전류 4배 → 손실 16배")
    # turns ratio and the secondary stray inductance seen from the primary
    Vin, D, fs, Lk, Ls = v["Vin"], v["D_design"], v["fs"], v["Lk"], v["L_sig"]
    n_h, n_l = Vin * D / Vh, Vin * D / Vl
    Lt_h, Lt_l = Lk + n_h**2 * Ls, Lk + n_l**2 * Ls
    dD_h0, dD_l0 = ref.psfb_duty_loss(Lk, fs, Ih, n_h, Vin), ref.psfb_duty_loss(Lk, fs, Il, n_l, Vin)
    dD_h, dD_l = ref.psfb_duty_loss(Lt_h, fs, Ih, n_h, Vin), ref.psfb_duty_loss(Lt_l, fs, Il, n_l, Vin)
    res.add_metric("n_hi", f"권수비 n @ {Vh:g} V (D_eff {D:g})", n_h, "", basis="n = V_in·D_eff/V_o")
    res.add_metric("n_lo", f"권수비 n @ {Vl:g} V", n_l, "")
    res.add_metric("Lsig_ref_lo", f"2차 배선 L_σ {Ls * 1e9:g} nH의 1차 환산 @ {Vl:g} V", n_l**2 * Ls, "H", basis=f"n²·L_σ (48 V에서는 {n_h**2 * Ls * 1e6:.3g} µH)")
    res.add_metric("dD_hi", f"duty loss @ {Vh:g} V (L_k + n²L_σ)", dD_h, "", basis=f"L_σ 없으면 {dD_h0:.4f}; 같은 전력이면 I/n이 같아 L_k 성분은 같다")
    res.add_metric("dD_lo", f"duty loss @ {Vl:g} V (L_k + n²L_σ)", dD_l, "", basis=f"L_σ 없으면 {dD_l0:.4f}")
    # budgets
    N = int(v["N_par"])
    b12 = lv_budget(Il, v["rect"], v, N)
    b48 = lv_budget(Ih, v["rect"], v, N)
    rows = []
    for key, lab, how in (
        ("P_cond", "SR 채널 전도", f"I_o²·{b12['ser']}·R_ds/N (위치당 반주기 도통)"),
        ("P_w", "2차 권선 (AC 계수 포함)", "I_o²·R_w·F_R"),
        ("P_out", "출력 경로 (busbar·connector·PCB·shunt·L_o DCR)", "I_o²·ΣR"),
        ("P_gate", "SR gate 구동", f"{b12['pos']}·N·Q_g·V_drv·f_s"),
        ("P_oss", "SR Coss (합성 E_oss·f_s)", f"{b12['pos']}·N·E_oss·f_s"),
    ):
        rows.append([lab, b48[key], b12[key], b12[key] / b12["total"] * 100, how])
    rows.append(["합계", b48["total"], b12["total"], 100.0, f"{P:g} W 대비 {Vh:g} V {b48['total'] / P:.2%} / {Vl:g} V {b12['total'] / P:.2%}"])
    res.tables.append(Table("t_budget", f"2차 경로 손실 budget (같은 부품값, {v['rect'].upper()} 정류, N = {N})", ["항목", f"{Vh:g} V [W]", f"{Vl:g} V [W]", f"{Vl:g} V 비중 [%]", "계산"], rows,
                            note="모든 저항·전하·에너지는 ASSUMED 합성값. 리플·commutation 중복 도통은 무시한 A 수준(아래 스위칭 검산과 비교). 1차 측·transformer core 손실은 포함하지 않았다."))
    res.add_metric("loss12", f"{Vl:g} V 2차 경로 손실 합 (budget)", b12["total"], "W", basis=f"{P:g} W의 {b12['total'] / P:.2%}")
    res.add_metric("loss48", f"{Vh:g} V 2차 경로 손실 합 (같은 부품)", b48["total"], "W")
    res.add_metric("floor12", f"{Vl:g} V 고정 경로 손실 (N과 무관)", b12["P_w"] + b12["P_out"], "W", basis="권선 + 출력 경로: 소자를 병렬로 늘려도 줄지 않는다")
    # paralleling sweep
    Ns = list(range(1, int(v["N_max"]) + 1))
    cond = [lv_budget(Il, v["rect"], v, k)["P_cond"] for k in Ns]
    sw = [lv_budget(Il, v["rect"], v, k)["P_gate"] + lv_budget(Il, v["rect"], v, k)["P_oss"] for k in Ns]
    tot = [lv_budget(Il, v["rect"], v, k)["total"] for k in Ns]
    hot = [lv_budget(Il, v["rect"], v, k)["P_hot"] for k in Ns]
    avg = [lv_budget(Il, v["rect"], v, k)["P_avg_dev"] for k in Ns]
    k_opt = int(np.argmin(tot))
    coef = Il**2 * b12["ser"] * v["R_ds"]
    per = b12["pos"] * (v["Q_g"] * v["V_drv"] + v["E_oss"]) * fs
    N_star = math.sqrt(coef / per)
    res.add_series("n_cond", "SR 전도 (∝ 1/N)", "W", Ns, cond)
    res.add_series("n_sw", "gate + Coss (∝ N)", "W", Ns, sw)
    res.add_series("n_tot", "2차 경로 합계", "W", Ns, tot)
    res.add_series("n_hot", "가장 뜨거운 소자 손실 (분산 포함)", "W", Ns, hot)
    res.add_series("n_avg", "평균 소자 손실", "W", Ns, avg, dash=True)
    res.add_metric("N_opt", "합계 최소 병렬 수 (정수 sweep)", Ns[k_opt], "", basis=f"연속 최적 √(전도계수/소자당 고정손실) = {N_star:.2f}")
    res.add_metric("hot_ratio", f"N = {N}에서 가장 뜨거운 소자 / 평균 소자 손실", b12["P_hot"] / b12["P_avg_dev"], "", basis=f"R_DS(on) 분산 {v['spread']:.0%} 한 개 가정 (정적 분담)")
    res.add_check(Check("병렬 최적 N: 연속 해 vs 정수 sweep", "PASS" if abs(Ns[k_opt] - N_star) <= 1.0 else "FAIL", Ns[k_opt] - N_star, "", 1.0, path="dP/dN = 0 해석해 vs 정수 N 전수 계산", independent=True, detail=f"정수 최적 {Ns[k_opt]}, 연속 {N_star:.3f}"))
    # output voltage sweep for the same P and R
    Vs = np.geomspace(6, 60, 60)
    res.add_series("v_loss", f"같은 {Rp * 1e3:g} mΩ 경로 손실 (P = {P:g} W)", "W", Vs.tolist(), [(P / x) ** 2 * Rp for x in Vs])
    # duty loss vs secondary stray inductance
    Lsg = np.linspace(0, 20e-9, 41)
    res.add_series("dd_hi", f"{Vh:g} V (n = {n_h:.3g})", "", (Lsg * 1e9).tolist(), [ref.psfb_duty_loss(Lk + n_h**2 * x, fs, Ih, n_h, Vin) for x in Lsg])
    res.add_series("dd_lo", f"{Vl:g} V (n = {n_l:.3g})", "", (Lsg * 1e9).tolist(), [ref.psfb_duty_loss(Lk + n_l**2 * x, fs, Il, n_l, Vin) for x in Lsg])
    # switched check at the low voltage with the budget resistances coupled into the state equations
    ok_sim = False
    if v["sim_check"]:
        R_load = Vl**2 / P
        p = {"Vin": Vin, "n": n_l, "phi_deg": 180 * D, "fs": fs, "Lk": Lt_l, "Lo": v["Lo_lv"], "Co": v["Co_lv"], "R": R_load, "rect": v["rect"], "rect_dev": "sr", "R_sr": v["R_ds"] / N, "Vf": 0.0, "R_pri": 0.0, "R_w": v["R_w"] * v["F_R"], "R_Lo": b12["R_out"]}
        phi, sys, sol, _t, ok, why = regulate_phi(p, Vl, v["phi_max"])
        if sol is None or not sol.converged:
            res.add_check(Check("스위칭 검산", "FAIL", path="regulate", detail=why))
        else:
            tr = run_period(sys, sol.x0, 2)
            m = measure(sys, tr, sol.x0)
            led = energy_ledger(tr, sys, 0.0, sys.T, ["p_in"], ["p_out"], ["p_loss"], rated_power=P)
            res.add_check(ledger_check(led, what=f"{Vl:g} V 스위칭: "))
            res.add_metric("phi_lo", f"{Vl:g} V 유지에 필요한 명령 phase (스위칭)", math.degrees(sys.phi), "deg", basis=f"D_cmd {m['D_cmd']:.4f} (seed {D:g} + duty loss + 저항 강하)")
            nt = "" if ok else "12 V 유지 실패 → budget 운전점(250 A)이 아니므로 판정하지 않음"
            res.add_metric("sim_sr", "SR 전도 손실 (결합 스위칭 모델)", m["p_sr"], "W", ref=b12["P_cond"], ref_label="budget I_o²·R/N", tol=0.08 if ok else None, basis="리플·commutation 중복 도통 포함", note=nt)
            res.add_metric("sim_w", "2차 권선 손실 (결합 스위칭 모델)", m["p_w"], "W", ref=b12["P_w"], ref_label="budget I_o²·R_w·F_R", tol=0.08 if ok else None, note=nt)
            res.add_metric("sim_out", "출력 경로 손실 (결합 스위칭 모델)", m["p_Lo"], "W", ref=b12["P_out"], ref_label="budget I_o²·ΣR", tol=0.03 if ok else None, note=nt)
            res.add_metric("sim_vo", "스위칭 모델 V_o", m["vo"], "V", ref=Vl, ref_label="목표", tol=1e-5 if ok else None, basis="regulate 결과")
            res.add_metric("sim_eff", "2차 경로만의 효율 (스위칭)", m["P_out"] / m["P_in"], "", basis="1차·스위칭·core 손실 제외")
            ok_sim = ok
            if not ok:
                res.verdict("FAIL_CONSTRAINT", f"{Vl:g} V를 유지할 phase 여유가 없다: {why}")
            bands = _wave_series(res, sys, tr, periods=2.0)
            res.add_plot("p_lv_i", f"{Vl:g} V 스위칭 검산: SR 경로 전류와 출력 인덕터 전류", ["iSR1", "iSR2", "iLo"], y_label="전류 (2차)", y_unit="A", bands=bands, group="lv", level="C",
                         proved=f"{Il:.0f} A급 출력에서 SR이 반주기씩 전류를 나르고 commutation 동안 두 경로가 겹친다. 결합 모델의 손실이 budget과 비교된다.",
                         not_yet="SR 병렬 소자 사이의 동적 분담·배선 L의 분배는 모델에 없다(위치당 등가 R 하나).")
            res.add_plot("p_lv_v", f"{Vl:g} V 스위칭 검산: 1차 bridge 전압과 2차 EMF (1차 환산)", ["vab", "nvs"], y_label="전압 (1차 환산)", y_unit="V", bands=bands, group="lv", level="C",
                         proved="n = 48에서는 2차 배선 L_σ가 n²배로 1차에 보여 commutation(duty loss)이 길어진다.", not_yet="2차 링잉은 모델 밖이다.")
            res.circuit = {"diagram": psfb_circuit({**v, **p}, v["rect"]).to_json(), "intervals": bands, "plot_group": "lv"}
    res.add_plot("p_vloss", "같은 전력·같은 저항에서 출력전압에 따른 경로 손실", ["v_loss"], x_label="출력전압", x_unit="V", y_label="I²R", y_unit="W", kind="xy", log_x=True, log_y=True, level="A",
                 vlines=[{"x": Vh, "label": f"{Vh:g} V"}, {"x": Vl, "label": f"{Vl:g} V"}],
                 proved="같은 전력에서 전류는 1/V, 같은 저항의 손실은 1/V²로 는다(로그-로그 기울기 −2): 48 → 12 V는 16배.", not_yet="저항 자체는 설계에 따라 바뀐다 (12 V는 더 굵은 도체가 필요).")
    res.add_plot("p_npar", "SR 병렬 수에 따른 손실: 병렬로만은 한계가 있다", ["n_cond", "n_sw", "n_tot"], x_label="위치당 병렬 수 N", x_unit="", y_label="손실", y_unit="W", kind="xy", level="A",
                 hlines=[{"y": b12["P_w"] + b12["P_out"], "label": "고정 경로 손실 (N 무관)"}], vlines=[{"x": N, "label": f"N = {N}"}],
                 proved="전도 손실은 1/N로 줄지만 gate·Coss 손실은 N에 비례해 늘어 합계 최소점이 있고, 권선·busbar·connector·shunt의 고정 손실은 N으로 줄지 않는다.",
                 not_yet="gate·Coss 값은 합성이다. 병렬 소자 사이 동적 분담·layout L 비대칭(EX03)은 포함하지 않았다.")
    res.add_plot("p_hot", "가장 뜨거운 소자 vs 평균 소자 (정적 분담)", ["n_hot", "n_avg"], x_label="위치당 병렬 수 N", x_unit="", y_label="소자당 손실", y_unit="W", kind="xy", level="A",
                 proved="R_DS(on)이 작은 소자 하나가 더 많은 전류를 가져가 가장 뜨거운 소자의 손실이 평균보다 크다. N이 커져도 그 비율은 줄지 않는다.",
                 not_yet="온도계수에 의한 자기 분담 보정·열 결합(EX08)·동적 분담은 없다.")
    res.add_plot("p_lsig", "2차 배선 L_σ가 duty loss를 키운다 (n²배로 1차에 보임)", ["dd_hi", "dd_lo"], x_label="2차 루프 인덕턴스 L_σ", x_unit="nH", y_label="ΔD", y_unit="", kind="xy", level="A",
                 vlines=[{"x": Ls * 1e9, "label": f"L_σ = {Ls * 1e9:g} nH"}],
                 proved=f"같은 3 kW에서 반사 전류 I/n은 같지만, n = {n_l:.0f}에서는 2차 배선 수 nH가 n²배로 보여 duty loss가 크게 는다. 12 V는 배선 L이 제어 범위를 먹는다.",
                 not_yet="평균 duty loss 식(A)이다. 배선 L은 SR 루프·busbar·transformer 단자 합의 합성값.")
    if not any(c == "FAIL_CONSTRAINT" for c, _ in res.verdicts):
        res.verdict("PASS_WITHIN_MODEL", "12 V 확장의 전류·경로 손실·병렬 한계를 A 수준 budget으로 계산하고 결합 스위칭 모델로 검산" if ok_sim or not v["sim_check"] else "budget 계산 완료")
    res.assumptions += ["모든 경로 저항·Q_g·E_oss·분산은 ASSUMED 합성값 (특정 부품 아님)", "budget: 위치당 전류는 반주기 도통 I_o, 리플·commutation 중복 무시", "권선 AC 저항은 상수 계수 F_R (정밀값은 MISSING_INPUT: 권선 구조·주파수 필요)", f"{P:g} W는 총합 (모듈 분할 없음)"]
    res.not_valid_for += ["특정 MOSFET·busbar 설계의 손실 정밀값", "병렬 소자 동적 분담·layout (EX03)", "SR 소자 온도·수명 (FL03/EX08)", "current doubler 정류 (모델 없음)"]
    res.interpretation = (
        f"{P:g} W를 {Vl:g} V로 내보내면 전류가 {Il:.0f} A라 같은 1 mΩ가 {Ll:.1f} W를 먹는다. SR은 병렬로 전도손실을 줄일 수 있지만 gate·Coss 손실이 N에 비례해 늘고, "
        "권선·busbar·connector·PCB·shunt 같은 고정 경로는 소자 수와 무관하다. 게다가 n이 커져 2차 배선 인덕턴스가 n²배로 1차에 보여 duty loss까지 커진다. "
        "그래서 12 V 확장은 ‘소자를 더 붙이면 된다’가 아니라 정류 방식·권선·termination·배선 설계 전체의 문제다."
    )
    return res


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_CTRL = [
    Param("ctrl", "제어", "", "open", kind="choice", choices=[("open", "개루프 (명령 phase 고정)"), ("regulate", "느린 전압 루프 (V_o 목표를 맞추는 phase를 찾음)")], source="ASSUMED", group="제어"),
    Param("Vo_target", "regulate 목표 V_o", "V", 48.0, "V", vmin=0.5, vmax=1000, source="TEXTBOOK", source_note="48 V", group="제어"),
    Param("phi_max", "phase 상한 φ_max", "deg", 170.0, "deg", vmin=10, vmax=178, source="ASSUMED", source_note="dead time·최소 freewheel 여유 (합성)", group="제어"),
]

_Q_PSFB = [
    Question(
        "PSFB의 phase shift와 DAB의 phase shift는 같은 것인가?",
        "아니다. PSFB의 φ는 한 full bridge 안의 두 leg 사이 위상으로 transformer에 ±V_in/0의 세 레벨을 만들고, 출력은 2차 정류와 출력 L이 평균낸다(V_o ≈ V_in D_eff/n). DAB의 φ는 두 active bridge 사이 위상으로 직렬 L 전류와 전력을 정하고(P = V1V2φ(1−|φ|/π)/(ωL)), 양쪽 bridge는 50 % 사각파다. 같은 기호로 같은 식을 쓰면 안 된다.",
        "Is the phase shift of a PSFB the same as that of a DAB?",
        "No. In a PSFB the phase is between the two legs of one bridge; it sets the width of the plus-minus V_in pulses, and the rectifier and output inductor average them, so V_o is about V_in times D_eff over n. In a DAB the phase is between two active bridges, each a 50 percent square wave, and it sets the inductor current and the power.",
        ["leg 사이 vs bridge 사이", "세 레벨 vs 두 사각파", "V_in·D_eff/n", "전력식 다름"],
    ),
    Question(
        "명령 duty가 0.72인데 출력은 0.72·V_in/n보다 낮다. 왜인가?",
        "v_AB가 반전하면 누설 L_k 전류가 −I/n에서 +I/n로 바뀌어야 한다. 그동안 두 정류 경로가 함께 도통해 2차가 단락되고 출력은 전압을 받지 못한다(duty loss). 이상 주기해에서 ΔD = 4L_k f_s I_0/(n V_in)이고, I_0는 반전 순간의 출력 인덕터 전류다. 전류에 비례하므로 무손실 출력저항 4L_k f_s/n²처럼 보인다.",
        "The commanded duty is 0.72 but the output is below 0.72 V_in over n. Why?",
        "When the bridge voltage reverses, the leakage current has to swing from minus I over n to plus I over n. Meanwhile both rectifier paths conduct, the secondary is shorted and the output receives no volt-seconds. For the ideal periodic solution the lost duty is 4 L_k f_s I_0 over n V_in, so it grows with load current and looks like a lossless output resistance.",
        ["commutation 동안 2차 단락", "L_k 전류 반전", "ΔD ∝ I", "등가 출력저항"],
        kind="calc",
    ),
    Question(
        "PSFB는 ZVS니까 Eon = 0이라고 해도 되나?",
        "조건 없이 말하면 안 된다. leading leg는 반사된 출력 인덕터 전류로 노드를 옮기지만, lagging leg는 v_AB가 0을 벗어나는 순간 정류가 2차를 단락해 L_k(와 자화) 에너지만 쓴다. 그래서 경부하에서 lagging leg부터 ZVS를 잃는다. L_k를 키우면 범위가 넓어지지만 duty loss와 순환전류가 는다. 이상 스위치 모델에서 ZVS는 NOT_EVALUABLE이고 전류 screen은 SCREEN_ONLY다.",
        "Can you say a PSFB has zero turn-on loss because it is ZVS?",
        "Not without conditions. The leading leg is commutated by the reflected output-inductor current, but the lagging leg only has the leakage and magnetising energy because the rectifier shorts the secondary as soon as the bridge voltage leaves zero. So the lagging leg loses ZVS first at light load, and a larger leakage inductance buys range at the cost of duty loss and circulating current.",
        ["leading vs lagging", "L_k 에너지만", "경부하", "대가: duty loss"],
        kind="pressure",
    ),
]

EXPERIMENTS = [
    Experiment(
        key="psfb_duty_loss",
        title="PSFB: 명령 phase가 그대로 출력 duty가 되지 않는다 (duty loss와 SR 전류)",
        goal=(
            "800 V·n = 12·D_eff 0.72가 주는 이상 48 V는 합성 seed다. 누설 L_k가 있으면 v_AB가 반전할 때 두 정류 경로가 함께 도통해 2차가 단락되는 commutation 구간이 생겨 "
            "유효 duty가 줄어든다. 정확 스위칭 해로 명령 phase → D_eff·V_o·SR 전류와 RMS를 계산하고, PWL 독립 경로·L_k volt-second 항등식·RK45 독립 solver·에너지 원장으로 검증한다."
        ),
        params=_psfb_params(_loss_params() + _CTRL + _screen_params()),
        presets=[
            Preset("nominal", "800 V·n 12·φ 129.6°·L_k 20 µH·1.5 kW", {}, "seed phase를 그대로 명령", ("nominal", "reference")),
            Preset("seed_ideal", "L_k → 10 nH (이상 seed 한계)", {"Lk": 1e-8}, "V_o = 48 V·D_eff 0.72 재현", ("reference", "variant")),
            Preset("regulate48", "48 V 유지 (느린 전압 루프)", {"ctrl": "regulate"}, "duty loss만큼 phase가 커진다", ("reference", "variant")),
            Preset("lossy_sr", "SR 2 mΩ·2차 권선 1 mΩ·1차 150 mΩ·출력 경로 1 mΩ", {"R_sr": 2e-3, "R_w": 1e-3, "R_pri": 0.15, "R_Lo": 1e-3}, "손실이 전력단에 결합된 경우", ("reference", "variant")),
            Preset("ct", "center-tap 정류 (같은 손실값)", {"rect": "ct", "R_sr": 2e-3, "R_w": 1e-3, "R_pri": 0.15, "R_Lo": 1e-3}, "소자 수·전압 stress·권선 이용률 비교", ("variant",)),
            Preset("diode", "다이오드 정류 (V_f 0.7 V, 5 mΩ)", {"rect_dev": "diode", "R_sr": 5e-3}, "SR 대신 다이오드", ("variant",)),
            Preset("light", "경부하 100 W", {"P": 100.0}, "DCM: commutation 없음, D_eff > D_cmd", ("corner",)),
            Preset("low_line", "V_in 550 V corner에서 48 V 유지", {"Vin": 550.0, "ctrl": "regulate"}, "n = 12로는 불가", ("failure", "reference")),
        ],
        run=run_duty_loss,
        model_level="C (+A 손계산·PWL)",
        suggested_change="누설 L_k를 20 → 40 µH로 두 배로 한다 (명령 phase 129.6° 고정).",
        prediction=Prediction(
            "L_k를 20 → 40 µH로 두 배로 하면 (φ 고정) duty loss ΔD와 V_o는?",
            ["ΔD 약 2배, V_o 감소", "ΔD 그대로 (L_k는 손실이 없으니까)", "ΔD 절반, V_o 증가", "모르겠다"],
            "ΔD 약 2배, V_o 감소",
            "ΔD = 4L_k f_s I_0/(n V_in): 누설 전류를 −I/n에서 +I/n로 반전시키는 시간이 L_k에 비례한다. L_k는 무손실이지만 출력 쪽에서는 등가 저항 4L_k f_s/n²처럼 보인다. V_o가 떨어지면 부하전류도 조금 줄어 정확히 2배는 아니다.",
            ["dD", "Vo", "t_c"],
            handcalc=[{"key": "Vo", "label": "V_o (이상 seed와 비교)", "unit": "V"}, {"key": "dD", "label": "duty loss ΔD", "unit": ""}, {"key": "t_c", "label": "commutation 시간", "unit": "s"}],
        ),
        suggested={"Lk": 40e-6},
        student=(
            "PSFB의 1차 bridge는 두 leg를 서로 어긋나게 켜서 transformer에 +V_in, 0, −V_in을 만든다. 2차에서 정류하면 +V_in/n 펄스가 반주기마다 나오고, 출력 인덕터가 그 평균을 V_o로 만든다. "
            "그런데 전압이 반전되는 순간, 누설 인덕턴스에 흐르던 전류는 즉시 방향을 바꿀 수 없다. 전류가 바뀌는 동안 2차 정류 소자 네 개가 모두 도통해 2차가 단락되고, 이 시간만큼 출력은 전압을 받지 못한다. "
            "그래서 명령 duty 0.72를 줘도 실제로 출력에 전달된 duty는 조금 작다."
        ),
        expert=(
            "① 이상 주기해에서 L_k의 반주기 volt-second를 쓰면 ΔD = 4L_k f_s I_0/(n V_in)이 정확하다(I_0 = v_AB 반전 순간 i_Lo). 평균전류 I_o를 넣은 교과서형 근사는 리플만큼 과대평가한다. "
            "② commutation 시간 t_c와 ΔD는 같지 않다: 전력 구간에는 L_k/n²와 L_o가 전압을 나눠 정류 출력이 V_in/n보다 조금 작고, freewheel에도 정류 출력이 0이 아니다. 둘을 섞지 않는다. "
            "③ duty loss는 전류에 비례하므로 무손실 출력저항 4L_k f_s/n²로 보이고, 저전압 corner의 phase 여유를 먹는다: V_in 550 V에서 n = 12는 이상 D부터 1을 넘는다. "
            "④ SR 전류는 commutation 동안 (i_Lo ± n i_p)/2로 두 경로가 나눠 가지며, RMS는 이 파형을 정확 적분한다. FB 정류는 소자 2개가 직렬로 도통하고 blocking은 V_in/n, CT는 1개 도통·2V_in/n blocking이다. "
            "⑤ 이 모델은 자화전류·2차 링잉·SR 타이밍 오차를 계산하지 않는다. ZVS 판정은 NOT_EVALUABLE이고 전환 순간 전류 표는 SCREEN_ONLY다."
        ),
        customer_ko=(
            "명령 duty 0.72로는 48 V가 아니라 약 46.5 V가 나옵니다. 누설 인덕턴스 20 µH가 전압 반전 때 약 113 ns 동안 2차를 단락시키기 때문이고, 이 duty loss는 부하전류에 비례합니다. "
            "48 V를 유지하려면 phase가 약 134°까지 필요하니, 저전압 입력 corner에서 phase 여유가 남는지 권수비와 함께 확인하시죠."
        ),
        customer_en=(
            "With a commanded duty of 0.72 the output is about 46.5 V rather than 48 V, because the 20 microhenry leakage shorts the secondary for roughly 113 nanoseconds at each voltage reversal, and that duty loss grows with load current. "
            "Holding 48 V needs about 134 degrees of phase, so let's check that the low-input-voltage corner still has phase margin with this turns ratio."
        ),
        questions=_Q_PSFB[:2],
        circuit="psfb",
        textbook=[TB_14, TB_19],
        reference_presets=["nominal", "seed_ideal", "regulate48", "lossy_sr", "low_line"],
        runtime_hint="seconds",
        claim_limit="이상 스위치 수준(C)의 duty loss·SR 전류·RMS. ZVS·링잉·스위칭 손실·자화전류는 주장하지 않는다.",
    ),
    Experiment(
        key="lk_tradeoff",
        title="L_k를 키우면 lagging leg ZVS 범위가 넓어지지만 무엇을 내주나",
        goal=(
            "PSFB의 leading leg는 반사된 출력 인덕터 전류로, lagging leg는 L_k 에너지만으로 스위치 노드를 옮긴다. L_k를 5–80 µH로 바꾸며 48 V를 유지할 때 duty loss(정확 스위칭 해), "
            "lagging leg ZVS screen이 통과하는 최소 부하(SCREEN_ONLY), 저전압 corner phase 여유를 지키는 최대 권수비와 그에 따른 1차 전류를 함께 본다."
        ),
        params=_psfb_params(
            _loss_params()
            + _screen_params()
            + [
                Param("Lk_min", "sweep L_k 최소", "H", 5e-6, "µH", vmin=1e-7, vmax=1e-3, source="ASSUMED", group="sweep"),
                Param("Lk_max", "sweep L_k 최대", "H", 80e-6, "µH", vmin=1e-7, vmax=2e-3, source="ASSUMED", group="sweep"),
                Param("Vin_min", "저전압 입력 corner V_in,min", "V", 700.0, "V", vmin=100, vmax=1500, source="ASSUMED", source_note="n 여유 계산용 합성 corner", group="제어"),
                Param("phi_max", "phase 상한 φ_max", "deg", 170.0, "deg", vmin=10, vmax=178, source="ASSUMED", group="제어"),
            ]
        ),
        presets=[
            Preset("nominal", "C_node 200 pF, dead time 100 ns, 1.5 kW", {}, "", ("nominal", "reference")),
            Preset("cnode100", "C_node 100 pF", {"C_node": 100e-12}, "작은 노드 용량", ("variant",)),
            Preset("vin650", "저전압 corner 650 V", {"Vin_min": 650.0}, "n 여유가 더 빠듯", ("corner",)),
        ],
        run=run_lk_tradeoff,
        model_level="C + A (screen)",
        suggested_change="현재 L_k를 20 → 40 µH로 바꿔 duty loss와 screen 최소 부하가 어떻게 움직이는지 본다.",
        prediction=Prediction(
            "L_k를 20 → 40 µH로 키우면 (48 V 유지) lagging leg ZVS screen이 통과하는 최소 부하와 duty loss는?",
            ["최소 부하 ↓, duty loss ↑", "최소 부하 ↑, duty loss ↓", "둘 다 ↓", "모르겠다"],
            "최소 부하 ↓, duty loss ↑",
            "lagging leg에 필요한 전류는 V_in√(C_node/L_k)로 L_k가 클수록 작아진다(더 가벼운 부하까지 통과). 대신 누설 전류 반전 시간이 L_k에 비례해 duty loss가 늘고, 전환 시간 자체도 √(L_k C)로 길어진다.",
            ["dD_sel", "fmin_sel"],
            handcalc=[{"key": "fmin_sel", "label": "screen 최소 부하 비율", "unit": ""}],
        ),
        suggested={"Lk": 40e-6},
        student=(
            "스위치가 꺼지고 다음 스위치가 켜지기 전 dead time 동안 누군가 스위치 노드의 커패시턴스를 반대쪽 전압까지 옮겨줘야 한다. leading leg 쪽은 출력 인덕터의 큰 전류가 이 일을 해 주지만, "
            "lagging leg 쪽은 그 순간 2차가 단락되어 누설 인덕턴스에 저장된 작은 에너지만 쓸 수 있다. 그래서 부하가 가벼워지면 lagging leg부터 ZVS가 어려워진다."
        ),
        expert=(
            "screen 식: leading leg는 전류원 가정 charge screen i ≥ C_node V_in/t_dead, lagging leg는 LC 에너지 screen i ≥ V_in√(C_node/L_k)와 전환 시간 √(L_k C)·asin(…) ≤ t_dead. "
            "L_k를 키우면 필요한 전류는 1/√L_k로 줄지만 duty loss는 L_k에 비례하고, 저전압 corner에서 48 V를 지키려면 n을 낮춰야 해 1차 RMS·순환전류가 커진다. "
            "자화전류를 의도적으로 키우는 방법, 외부 shim L과 clamp diode, 경부하 burst는 각각 손실·링잉·제어 복잡도의 비용이 있다. 모든 판단은 비선형 Coss 사건 모델(EX02)과 측정으로 확정한다."
        ),
        customer_ko=(
            "현재 20 µH에서는 정격 부하에서도 lagging leg의 ZVS screen이 빠듯합니다. 40–80 µH로 키우면 경부하 범위가 넓어지지만 duty loss가 두세 배가 되어 저전압 입력에서 권수비를 낮춰야 하고, "
            "그만큼 1차 전류가 늘어납니다. 경부하 효율 요구와 입력전압 범위를 먼저 확정한 뒤 L_k를 정하시죠."
        ),
        customer_en=(
            "At 20 microhenries the lagging-leg ZVS screen is marginal even at full load. Raising it to 40 to 80 microhenries extends the light-load range, but the duty loss grows two to three times, "
            "which forces a lower turns ratio for the low-input corner and raises the primary current. Let's fix the light-load efficiency target and the input range first, then choose the leakage."
        ),
        questions=[_Q_PSFB[2]],
        circuit="psfb",
        textbook=[TB_14],
        reference_presets=["nominal"],
        runtime_hint="seconds",
        claim_limit="duty loss·전류는 C, ZVS는 선형 C_node screen (SCREEN_ONLY). ZVS 보증 아님.",
    ),
    Experiment(
        key="psfb_vs_dab",
        title="같은 800→48 V·1.5 kW: PSFB와 DAB는 무엇이 다른가",
        goal=(
            "교재 실습: 동일 800→48 V·1.5 kW(모듈당)의 이상 PSFB와 DAB를 만든다. 전달전력만 같게 두지 말고 bridge·2차 RMS, peak, zero-power 구간, 인덕터 에너지, "
            "소프트 스위칭을 어느 수준까지 말할 수 있는지를 비교한다. DAB는 교재 11장 모듈(n = 50/3, L = 200 µH, φ = 0.328973 rad, I_rms 2.01988 A)을 닫힌 식과 독립 PWL 적분으로 재현한다."
        ),
        params=[
            Param("Vin", "입력전압 V_in (= DAB V_H)", "V", 800.0, "V", vmin=100, vmax=1500, source="TEXTBOOK", source_note="800 V"),
            Param("V_L", "출력전압 V_L", "V", 48.0, "V", vmin=1, vmax=500, source="TEXTBOOK", source_note="48 V"),
            Param("P", "전달 전력 (모듈당)", "W", 1500.0, "W", vmin=10, vmax=20e3, source="TEXTBOOK", source_note="1.5 kW/모듈 (총 3 kW는 2모듈)"),
            Param("fs", "스위칭 주파수 f_s (둘 다)", "Hz", 100e3, "kHz", vmin=10e3, vmax=1e6, source="TEXTBOOK", source_note="11장 DAB 100 kHz"),
            Param("n", "PSFB 권수비 n", "", 12.0, "", vmin=1, vmax=100, source="TEXTBOOK", source_note="14장 seed n = 12", group="PSFB"),
            Param("Lk", "PSFB 누설 L_k (1차 환산)", "H", 20e-6, "µH", vmin=1e-9, vmax=1e-3, source="ASSUMED", group="PSFB"),
            Param("Lo", "PSFB 출력 L_o", "H", 10e-6, "µH", vmin=1e-8, vmax=1e-2, source="ASSUMED", group="PSFB"),
            Param("Co", "PSFB 출력 C_o", "F", 100e-6, "µF", vmin=1e-7, vmax=1.0, source="ASSUMED", group="PSFB"),
            Param("rect", "PSFB 정류", "", "fb", kind="choice", choices=[("fb", "full-bridge"), ("ct", "center-tap")], source="ASSUMED", group="PSFB"),
            Param("phi_max", "PSFB phase 상한", "deg", 170.0, "deg", vmin=10, vmax=178, source="ASSUMED", group="PSFB"),
            Param("n_dab", "DAB 권수비 n", "", 50 / 3, "", vmin=1, vmax=100, source="TEXTBOOK", source_note="11장 n = 50/3", group="DAB"),
            Param("L_dab", "DAB 직렬 L (1차 환산)", "H", 200e-6, "µH", vmin=1e-6, vmax=5e-3, source="TEXTBOOK", source_note="11장 200 µH", group="DAB"),
            Param("C_node", "1차 노드 등가 C (screen)", "F", 200e-12, "pF", vmin=1e-12, vmax=1e-8, source="ASSUMED", group="screen"),
            Param("C_node_sec", "DAB 2차 노드 등가 C (screen)", "F", 5e-9, "nF", vmin=1e-11, vmax=1e-6, source="ASSUMED", source_note="저전압 소자 병렬 합성값", group="screen"),
            Param("t_dead", "dead time (screen)", "s", 100e-9, "ns", vmin=1e-9, vmax=2e-6, source="ASSUMED", group="screen"),
        ],
        presets=[
            Preset("nominal", "교재 조건 800→48 V·1.5 kW", {}, "14장 실습 + 11장 DAB 모듈", ("nominal", "reference")),
            Preset("dab_L150", "DAB L = 150 µH", {"L_dab": 150e-6}, "DAB 전류 모양 변화", ("variant",)),
            Preset("psfb_ct", "PSFB center-tap 정류", {"rect": "ct"}, "2차 소자 수·전압 stress", ("variant",)),
            Preset("psfb_lk40", "PSFB L_k 40 µH", {"Lk": 40e-6}, "duty loss·1차 peak", ("variant",)),
        ],
        run=run_psfb_vs_dab,
        model_level="C (PSFB) + A/PWL (DAB)",
        suggested_change="DAB 직렬 L을 200 → 150 µH로 바꿔 DAB 전류가 어떻게 달라지는지 PSFB와 비교한다.",
        prediction=Prediction(
            "같은 800→48 V·1.5 kW에서 transformer 1차 RMS가 더 큰 쪽은?",
            ["PSFB", "DAB", "같다", "모르겠다"],
            "PSFB",
            "PSFB는 duty로 강압하므로 n = 12로 작고, 1차에 반사 부하전류 I_o/n ≈ 2.6 A가 전력 구간뿐 아니라 freewheel에도 흐른다. DAB는 n = 50/3으로 전압을 정합해 반사 전류가 작고 전류가 사다리꼴이다.",
            ["psfb_ip_rms", "dab_irms"],
            handcalc=[{"key": "dab_phi", "label": "DAB φ", "unit": "rad"}, {"key": "dab_irms", "label": "DAB I_rms", "unit": "A"}],
        ),
        suggested={"L_dab": 150e-6},
        student=(
            "두 컨버터 모두 ‘phase’로 전력을 조절하지만 하는 일이 다르다. PSFB는 한 bridge의 두 다리를 어긋나게 켜서 전압 펄스의 폭을 바꾸고, 2차 정류와 출력 인덕터가 평균을 만든다. "
            "DAB는 양쪽 bridge가 모두 사각파를 내고, 두 사각파 사이의 시간차가 직렬 인덕터 전류를 만들어 전력을 보낸다. 같은 1.5 kW라도 전류의 모양·크기·흐르는 시간이 다르다."
        ),
        expert=(
            "비교 항목: 1차 RMS(PSFB가 큼: 작은 n + freewheel 순환), 1차 peak·turn-off 전류, 2차 소자 RMS(PSFB SR은 거의 DC를 반주기씩, DAB 2차 switch는 n배 교류), zero-power 구간(PSFB freewheel 26 %), "
            "backflow(PSFB commutation 전반 vs DAB 전류 부호 지연), 인덕터(PSFB는 누설 L_k + 출력 L_o, DAB는 직렬 L만). 소프트 스위칭은 둘 다 이상 모델로는 NOT_EVALUABLE이고, "
            "PSFB는 lagging leg가, DAB는 ratio mismatch·경부하 edge 전류 부호가 먼저 위험하다. 양방향 요구·입력 범위·경부하 효율·비용이 선택을 정한다(교재 표)."
        ),
        customer_ko=(
            "같은 1.5 kW에서 PSFB는 1차 RMS 약 2.6 A, DAB는 약 2.0 A입니다. PSFB는 단방향이지만 PWM 경험을 살릴 수 있고 출력 인덕터가 필요하며, DAB는 양방향과 명확한 phase 제어가 장점이지만 "
            "전압비 mismatch와 경부하 순환전류를 봐야 합니다. 양방향이 roadmap에 있는지와 입력 전압 범위를 먼저 확인하고 손실 모델로 비교하시죠."
        ),
        customer_en=(
            "At the same 1.5 kilowatts the PSFB carries about 2.6 amps RMS on the primary and the DAB about 2.0 amps. The PSFB is unidirectional, reuses PWM experience and needs an output inductor; "
            "the DAB is bidirectional with a clean phase control, but you must watch ratio mismatch and light-load circulating current. Let's confirm whether bidirectional operation is on the roadmap and the input range, then compare with a loss model."
        ),
        questions=[
            _Q_PSFB[0],
            Question(
                "양방향은 당장 필요 없지만 나중에 생길 수 있다면 무엇을 고르겠나?",
                "답은 하나가 아니다. 실제 roadmap과 하드웨어 upgrade 비용(2차 active switch, 제어, 검증)을 확인하고, 현재 비용·효율·검증 일정과 미래 전환 가능성을 분리해 제안한다. 단방향이 확정이면 PSFB나 LLC+SR, 양방향 가능성이 크면 DAB/CLLC를 같은 조건·같은 손실 경계로 비교한다.",
                "Bidirectional operation is not needed now but may come later. What would you choose?",
                "There is no single answer. I would confirm the real roadmap and the upgrade cost, such as secondary active switches, control and validation, and separate today's cost, efficiency and schedule from the future option. If unidirectional is firm, I compare PSFB and LLC with synchronous rectification; if bidirectional is likely, DAB or CLLC under the same conditions and loss boundaries.",
                ["roadmap 확인", "upgrade 비용", "현재 vs 미래 분리", "같은 조건 비교"],
                kind="pressure",
            ),
        ],
        circuit="psfb",
        textbook=[TB_14, TB_11],
        reference_presets=["nominal"],
        claim_limit="같은 전력의 전류·전압 파형 비교. 효율 순위·ZVS 판정은 주장하지 않는다.",
    ),
    Experiment(
        key="lv12_extension",
        title="12 V 확장: 전류 4배, 같은 저항 손실 16배 — 소자만 병렬로 늘리면 되나",
        goal=(
            "3 kW를 48 V에서 12 V로 바꾸면 전류가 62.5 A → 250 A, 같은 1 mΩ 경로 손실이 3.91 W → 62.5 W가 된다(교재). SR R_DS(on)·병렬 수·권선·busbar·connector·PCB·current sense를 "
            "모두 ASSUMED 저항으로 적은 경로 손실 budget을 만들고, 병렬만으로는 줄지 않는 고정 손실·gate/Coss 손실·분담 불균형, 그리고 n² 배로 1차에 보이는 2차 배선 L의 duty loss를 본다. "
            "budget은 저항을 상태방정식에 넣은 12 V 스위칭 모델로 검산한다."
        ),
        params=[
            Param("P", "전력 (총합)", "W", 3000.0, "W", vmin=10, vmax=50e3, source="TEXTBOOK", source_note="3 kW"),
            Param("V_hi", "기준 출력전압", "V", 48.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="48 V"),
            Param("V_lo", "확장 출력전압", "V", 12.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="12 V"),
            Param("R_path", "비교용 총 경로 저항", "Ω", 1e-3, "mΩ", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="1 mΩ"),
            Param("Vin", "입력전압", "V", 800.0, "V", vmin=100, vmax=1500, source="TEXTBOOK", group="변환기"),
            Param("D_design", "설계 유효 duty (n 결정)", "", 0.72, "", vmin=0.1, vmax=0.95, source="TEXTBOOK", source_note="seed 0.72", group="변환기"),
            Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=10e3, vmax=1e6, source="ASSUMED", group="변환기"),
            Param("Lk", "1차 누설 L_k", "H", 20e-6, "µH", vmin=1e-9, vmax=1e-3, source="ASSUMED", group="변환기"),
            Param("L_sig", "2차 루프 배선 L_σ (SR 루프·busbar·단자)", "H", 5e-9, "nH", vmin=0.0, vmax=1e-6, source="ASSUMED", source_note="합성 5 nH", group="배선"),
            Param("rect", "12 V 정류 방식", "", "ct", kind="choice", choices=[("ct", "center-tap (도통 소자 1개)"), ("fb", "full-bridge (도통 소자 2개 직렬)")], source="ASSUMED", group="정류"),
            Param("R_ds", "SR R_DS(on) (소자당, 고온)", "Ω", 1.0e-3, "mΩ", vmin=1e-6, vmax=0.1, source="ASSUMED", source_note="합성 1 mΩ", group="정류"),
            Param("N_par", "위치당 병렬 수 N", "", 4, "", vmin=1, vmax=64, kind="int", source="ASSUMED", group="정류"),
            Param("N_max", "병렬 sweep 상한", "", 16, "", vmin=2, vmax=64, kind="int", source="ASSUMED", group="정류"),
            Param("Q_g", "SR gate 전하 Q_g (소자당)", "C", 80e-9, "nC", vmin=0.0, vmax=1e-6, source="ASSUMED", group="정류"),
            Param("V_drv", "gate 구동 전압 스윙", "V", 10.0, "V", vmin=1, vmax=30, source="ASSUMED", group="정류"),
            Param("E_oss", "SR Coss 관련 에너지/주기 (소자당)", "J", 1.0e-6, "µJ", vmin=0.0, vmax=1e-3, source="ASSUMED", source_note="합성; Coss 손실 경계는 EX02", group="정류"),
            Param("spread", "R_DS(on) 분산 (한 소자가 이만큼 낮음)", "", 0.10, "", vmin=0.0, vmax=0.5, source="ASSUMED", group="정류"),
            Param("R_w", "2차 권선 DC 저항 (CT는 반권선당)", "Ω", 0.15e-3, "mΩ", vmin=0.0, vmax=0.1, source="ASSUMED", group="배선"),
            Param("F_R", "권선 AC 저항 계수 R_ac/R_dc", "", 1.6, "", vmin=1.0, vmax=10.0, source="ASSUMED", source_note="정밀값은 권선 구조·주파수 필요 (MISSING_INPUT)", group="배선"),
            Param("R_bus", "busbar 저항", "Ω", 50e-6, "mΩ", vmin=0.0, vmax=0.1, source="ASSUMED", group="배선"),
            Param("R_conn", "connector 접촉 저항 (합)", "Ω", 100e-6, "mΩ", vmin=0.0, vmax=0.1, source="ASSUMED", group="배선"),
            Param("R_pcb", "PCB 구리 저항", "Ω", 100e-6, "mΩ", vmin=0.0, vmax=0.1, source="ASSUMED", group="배선"),
            Param("R_shunt", "current sense shunt", "Ω", 100e-6, "mΩ", vmin=0.0, vmax=0.1, source="ASSUMED", group="배선"),
            Param("R_dcr", "출력 인덕터 DCR", "Ω", 100e-6, "mΩ", vmin=0.0, vmax=0.1, source="ASSUMED", group="배선"),
            Param("sim_check", "12 V 스위칭 검산 실행", "", True, kind="bool", source="ASSUMED", group="검산"),
            Param("Lo_lv", "12 V 출력 L_o", "H", 0.625e-6, "µH", vmin=1e-9, vmax=1e-3, source="ASSUMED", source_note="48 V의 10 µH를 전압² 비로 축소 (같은 상대 리플)", group="검산"),
            Param("Co_lv", "12 V 출력 C_o", "F", 1.6e-3, "mF", vmin=1e-6, vmax=1.0, source="ASSUMED", group="검산"),
            Param("phi_max", "phase 상한", "deg", 170.0, "deg", vmin=10, vmax=178, source="ASSUMED", group="검산"),
        ],
        presets=[
            Preset("nominal", "교재 3 kW·48/12 V·1 mΩ + 합성 12 V budget", {}, "", ("nominal", "reference")),
            Preset("fb_rect", "12 V를 full-bridge 정류로", {"rect": "fb"}, "직렬 도통 소자 2개", ("variant",)),
            Preset("more_par", "SR 병렬 12개", {"N_par": 12}, "병렬만 늘리면?", ("variant",)),
            Preset("stray_fail", "2차 배선 L_σ 30 nH", {"L_sig": 30e-9}, "duty loss로 12 V 유지 불가", ("failure", "reference")),
        ],
        run=run_lv12,
        model_level="A (budget) + C (검산)",
        suggested_change="SR 병렬 수를 4 → 12로 늘린다.",
        prediction=Prediction(
            "3 kW를 48 V → 12 V로 바꾸면 같은 1 mΩ 경로의 손실은? (그다음: SR을 4 → 12개로 병렬하면 2차 경로 합계 손실은 1/3이 되나?)",
            ["4배", "16배", "같다", "모르겠다"],
            "16배",
            "같은 전력에서 전류는 1/V로 4배, I²R은 16배(3.91 W → 62.5 W). 병렬 수를 늘리면 SR 전도손실은 1/N로 줄지만 gate·Coss 손실은 N배로 늘고 권선·busbar·connector·shunt는 그대로라 합계는 1/3이 되지 않는다.",
            ["L_lo", "ratio_L", "loss12"],
            handcalc=[{"key": "I_lo", "label": "12 V 전류", "unit": "A"}, {"key": "L_lo", "label": "12 V에서 1 mΩ 손실", "unit": "W"}],
        ),
        suggested={"N_par": 12},
        student=(
            "같은 전력을 더 낮은 전압으로 보내려면 전류가 그만큼 커져야 한다. 저항에서의 손실은 전류의 제곱에 비례하므로, 전압을 1/4로 낮추면 같은 저항에서 손실은 16배가 된다. "
            "그래서 12 V 출력에서는 MOSFET뿐 아니라 권선·busbar·connector·기판 구리·전류 센서 같은 작은 저항이 모두 중요해진다."
        ),
        expert=(
            "① budget은 경로 요소별로 I_rms²R을 적고(위치당 반주기 도통, 권선 AC 계수, DC 경로), 결합 스위칭 모델의 원장과 비교해 리플·commutation 중복 도통 효과를 확인한다. "
            "② 병렬 N: 전도 ∝ 1/N, gate(Q_g V f_s)·Coss(E_oss f_s) ∝ N → 최적 N이 있고, 권선·busbar·connector·shunt의 고정 손실은 줄지 않는다. 가장 낮은 R_DS(on) 소자가 전류를 더 가져가 hottest device가 평균보다 뜨겁다(동적 분담은 EX03). "
            "③ n = 48이면 2차 배선 수 nH가 n²배로 1차에 보여 duty loss가 커지고, 저전압 corner phase 여유를 먹는다 — 배선이 제어 범위의 문제다. "
            "④ CT는 도통 소자 1개·blocking 2V_in/n, FB는 직렬 2개·V_in/n; current doubler는 여기서 모델하지 않았다. 정류 방식마다 같은 n·같은 loss 식을 공유하지 않는다."
        ),
        customer_ko=(
            "12 V·3 kW는 250 A라 같은 1 mΩ가 62.5 W를 먹습니다. SR을 병렬로 늘리면 전도손실은 줄지만 gate·Coss 손실이 늘고 권선·busbar·connector·shunt 손실은 그대로입니다. "
            "게다가 2차 배선 인덕턴스가 1차에서 권수비 제곱으로 보여 duty가 부족해질 수 있으니, 정류 방식과 termination·배선 설계를 소자 선정과 같이 검토하시죠."
        ),
        customer_en=(
            "At 12 volts and 3 kilowatts the current is 250 amps, so the same 1 milliohm path dissipates 62.5 watts. Paralleling more synchronous rectifiers cuts conduction loss but adds gate and output-capacitance loss, "
            "and the winding, busbar, connector and shunt losses stay. The secondary wiring inductance also appears on the primary multiplied by the turns ratio squared and eats the duty range, so the rectifier topology, termination and layout should be reviewed together with the device choice."
        ),
        questions=[
            Question(
                "12 V 출력에서 MOSFET을 병렬로 더 붙이면 해결되지 않나?",
                "일부만 해결된다. SR 전도손실은 1/N로 줄지만 gate 구동·Coss 손실은 N배로 늘고, 권선·busbar·connector·PCB·shunt의 고정 손실은 소자 수와 무관하다. 병렬 소자 사이 정적·동적 분담이 달라 가장 뜨거운 소자가 평균보다 뜨겁다. 또 n이 커져 2차 배선 L이 n²배로 1차에 보여 duty loss가 는다.",
                "At 12 V output, can you just parallel more MOSFETs?",
                "Only partly. Conduction loss falls as one over N, but gate-drive and output-capacitance losses grow with N, and the winding, busbar, connector, PCB and shunt losses do not change. Static and dynamic sharing make the hottest device hotter than the average, and with a high turns ratio the secondary wiring inductance appears on the primary scaled by n squared and increases the duty loss.",
                ["고정 경로 손실", "gate·Coss ∝ N", "분담 불균형", "배선 L × n²"],
                kind="pressure",
            ),
            Question(
                "48 V → 12 V로 같은 3 kW면 1 mΩ 경로 손실은 얼마인가?",
                "전류 62.5 A → 250 A(4배), 손실 3.91 W → 62.5 W(16배). 같은 전력에서 I ∝ 1/V, I²R ∝ 1/V².",
                "With the same 3 kW, what is the loss of a 1 milliohm path at 48 V and at 12 V?",
                "The current goes from 62.5 to 250 amps, four times, and the loss from 3.91 to 62.5 watts, sixteen times, because at constant power the current scales as one over V and the resistive loss as one over V squared.",
                ["62.5 A", "250 A", "3.91 W", "62.5 W"],
                kind="calc",
            ),
        ],
        circuit="psfb",
        textbook=[TB_14],
        reference_presets=["nominal", "stray_fail"],
        runtime_hint="seconds",
        claim_limit="합성 저항·전하값의 A 수준 budget + 결합 스위칭 검산. 특정 부품·layout의 손실·온도는 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL11",
    title="PSFB·HV-LV·12 V 확장 — 선택의 이유를 설명하기",
    title_en="PSFB, HV-LV conversion and the 12 V extension",
    track="basic",
    order=11,
    path_note="14일 경로 11일차 (14 PSFB·48/12 V, FL11)",
    textbook=[TB_14, TB_11, TB_19],
    prerequisites=["FL01", "FL08"],
    summary="명령 phase → commutation duty loss → SR 전류 (정확 스위칭) → L_k와 ZVS의 대가 → 같은 1.5 kW의 DAB와 비교 → 12 V 대전류 경로 손실과 배선.",
    experiments=EXPERIMENTS,
    minimum_scope="PSFB waveforms·SR·duty loss; D_eff·LV RMS·DAB 비교 (교재 19장 표); 12 V 대전류 손실·배선 (지침 5절)",
    claim_limits=[
        "PSFB는 이상 스위치(C) 모델: ZVS는 NOT_EVALUABLE, 전환 전류 표는 SCREEN_ONLY",
        "자화전류·2차 링잉·SR 타이밍 오차·스위칭 손실은 계산하지 않음",
        "n = 12·V_in 800 V·D_eff 0.72의 48 V는 합성 seed; L_k·L_o·C_o·경로 저항은 ASSUMED",
        "DAB 비교는 교재 11장 모듈의 닫힌 식·PWL (FL08의 전체 실험 대체 아님)",
        "12 V budget은 합성 부품값의 A 수준; 특정 소자·layout 손실 아님",
    ],
    test_paths=["tests/test_fl11.py"],
)
