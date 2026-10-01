"""FL02 - Si/SiC/GaN devices, gate drive, DPT and protection (textbook ch.04-05).

Four experiments, kept at their own model level on purpose:
  A  synthetic A/B loss comparison (I_rms^2 R + E f_s) with an f_s sweep, the crossover
     frequency, an N-parallel sweep with explicit scaling assumptions and a datasheet
     extraction template whose values are all MISSING_INPUT (no real part data);
  A/B first-order gate/Miller/L di/dt/C dv/dt screens, a gate-charge ODE, and the
     short-circuit protection timeline that keeps device survival time and system
     reaction time apart;
  D  a bounded synthetic commutation cell (DPT): gate driver -> MOSFET channel with
     V_th and transconductance, C_gs, nonlinear C_gd(v) and C_ds(v), power-loop and
     common-source inductance, a lossy loop element, the high-side device with its own
     gate loop and body diode, and a load current held constant during the event.  It is
     integrated with a stiff solver; ringing comes only from these L and C.  DPT energy
     integration stores the v/i referents, deskew, window and Coss inclusion boundary.
  D  the blind question: a V_GS spike on the power-source reference vs the Kelvin
     reference (common-source inductance artifact vs real Miller turn-on).
The cell is a synthetic trend model, not a vendor device model, and it says nothing
about short-circuit survival.  Closed forms for comparison live in reference/devices.py,
which this module never imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..engine.pwl import PWL
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close

TB_04 = TextbookRef("sisicgan과-데이터시트-소자를-시스템으로-읽기-fl02", "04. Si·SiC·GaN과 데이터시트 [FL02]")
TB_05 = TextbookRef("gate-drivedpt보호-강점을-면접-증거로-만들기-fl02", "05. Gate drive·DPT·보호 [FL02]")
TB_18 = TextbookRef("시뮬레이션-최적안-python-주력-simulink-교차검증", "18. 시뮬레이션 최적안 (모델 수준)")

# ======================================================================================
# Synthetic device model (one die, or s identical dies lumped)
# ======================================================================================


@dataclass(frozen=True)
class DeviceParams:
    """Synthetic SiC-like MOSFET at a fixed junction temperature (not a vendor model).

    Channel: i_ch = g_m * V_s*softplus((v_gs - V_th)/V_s) * tanh(v_ds / V_k), with
    V_k = R_on g_m (V_on,ref - V_th) so that the on-resistance at V_on,ref is R_on.
    Capacitances: C_gs constant; C_gd(v_dg) = C_gd0/sqrt(1 + v_dg/V_gd0) for v_dg >= 0
    and C_gd0 below (gate above drain); C_ds(v) = C_ds0/sqrt(1 + v/V_ds0) (floored at
    2 C_ds0 for v < -0.75 V_ds0).  Body diode: smooth ideal diode V_f + R_d i.  No reverse
    recovery charge (unipolar body-diode assumption), no temperature dependence.
    """

    Vth: float = 4.0
    gm: float = 20.0
    Ron: float = 20e-3
    Von_ref: float = 18.0
    Cgs: float = 4e-9
    Cgd0: float = 263e-12
    Vgd0: float = 2.0
    Cds0: float = 1.2e-9
    Vds0: float = 10.0
    Rgint: float = 1.0
    Vf: float = 2.8
    Rd: float = 20e-3
    Vs_ch: float = 0.1
    Vs_d: float = 0.05

    @property
    def Vk(self) -> float:
        return self.Ron * self.gm * (self.Von_ref - self.Vth)

    def scaled(self, s: float) -> "DeviceParams":
        """s identical dies in parallel lumped into one (same voltages, s x currents)."""
        return replace(self, gm=self.gm * s, Ron=self.Ron / s, Cgs=self.Cgs * s, Cgd0=self.Cgd0 * s, Cds0=self.Cds0 * s, Rgint=self.Rgint / s, Rd=self.Rd / s)


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 35.0, x, np.log1p(np.exp(np.minimum(x, 35.0))))


class DevArrays:
    """Vectorised device functions for N branches (arrays) - used by the cell."""

    def __init__(self, devs: list[DeviceParams]):
        f = lambda name: np.array([getattr(d, name) for d in devs], dtype=float)  # noqa: E731
        self.n = len(devs)
        self.Vth, self.gm, self.Cgs = f("Vth"), f("gm"), f("Cgs")
        self.Cgd0, self.Vgd0, self.Cds0, self.Vds0 = f("Cgd0"), f("Vgd0"), f("Cds0"), f("Vds0")
        self.Rgint, self.Vf, self.Rd = f("Rgint"), f("Vf"), f("Rd")
        self.Vs_ch, self.Vs_d = f("Vs_ch"), f("Vs_d")
        self.Vk = np.array([d.Vk for d in devs])

    # capacitances --------------------------------------------------------------------
    def Cgd(self, vdg):
        u = 1.0 + np.maximum(vdg, 0.0) / self.Vgd0
        return self.Cgd0 / np.sqrt(u)

    def Cds(self, v):
        u = np.maximum(1.0 + v / self.Vds0, 0.25)
        return self.Cds0 / np.sqrt(u)

    # conduction ----------------------------------------------------------------------
    def ich(self, vgs, vds):
        ov = self.Vs_ch * _softplus((vgs - self.Vth) / self.Vs_ch)
        return self.gm * ov * np.tanh(vds / self.Vk)

    def ibody(self, vds):
        """Body-diode current, source -> drain (positive when v_ds < -V_f)."""
        return self.Vs_d / self.Rd * _softplus((-vds - self.Vf) / self.Vs_d)

    # closed-form stored energy and gate charge (independent of the ODE's C(v) use) ------
    def E_gd(self, vdg):
        vdg = np.asarray(vdg, dtype=float)
        u = 1.0 + np.maximum(vdg, 0.0) / self.Vgd0
        pos = self.Cgd0 * self.Vgd0**2 * (2.0 / 3.0 * u**1.5 - 2.0 * u**0.5 + 4.0 / 3.0)
        neg = 0.5 * self.Cgd0 * vdg**2
        return np.where(vdg >= 0.0, pos, neg)

    def Q_gd(self, vdg):
        vdg = np.asarray(vdg, dtype=float)
        u = 1.0 + np.maximum(vdg, 0.0) / self.Vgd0
        return np.where(vdg >= 0.0, 2.0 * self.Cgd0 * self.Vgd0 * (np.sqrt(u) - 1.0), self.Cgd0 * vdg)

    def E_ds(self, v):
        v = np.asarray(v, dtype=float)
        u = 1.0 + v / self.Vds0
        ua = np.maximum(u, 0.25)
        e = self.Cds0 * self.Vds0**2 * (2.0 / 3.0 * ua**1.5 - 2.0 * ua**0.5 + 4.0 / 3.0)
        va = -0.75 * self.Vds0
        return np.where(u >= 0.25, e, e + self.Cds0 * (v**2 - va**2))

    def Q_ds(self, v):
        v = np.asarray(v, dtype=float)
        u = np.maximum(1.0 + v / self.Vds0, 0.25)
        q = 2.0 * self.Cds0 * self.Vds0 * (np.sqrt(u) - 1.0)
        va = -0.75 * self.Vds0
        return np.where(1.0 + v / self.Vds0 >= 0.25, q, q + 2.0 * self.Cds0 * (v - va))

    def Q_gate(self, vgs, vds):
        """Charge on the gate node: C_gs v_gs - Q_gd(v_ds - v_gs)."""
        return self.Cgs * vgs - self.Q_gd(vds - vgs)

    def E_cap(self, vgs, vds):
        return 0.5 * self.Cgs * vgs**2 + self.E_gd(vds - vgs) + self.E_ds(vds)


# ======================================================================================
# Commutation cell: N parallel low-side branches (DUT) + one high-side device (HS, off)
# ======================================================================================

TOPOLOGIES = {
    "individual_kelvin": "개별 드라이버 · 각 Kelvin source 기준 (게이트 루프에 L_s 없음)",
    "common_kelvin_star": "공통 드라이버 · 반환선을 각 Kelvin에 R_e로 연결 (Kelvin 순환전류 포함)",
    "common_source_node": "드라이버 반환 = 공통 source 노드 (개별 L_s,k가 게이트 루프에 = common-source inductance)",
    "power_ground": "드라이버 반환 = DC− (L_s,k + 공통 L_cs가 게이트 루프에)",
}


@dataclass
class CellSpec:
    """Circuit of the synthetic commutation cell.  SI units; arrays have one entry per branch.

    Power path: V_bus -> R_loop -> L_a -> (L_b || R_p) -> P -> HS drain ... HS source -> L_sH -> M
    -> branch k: L_d (mutual matrix) -> DUT_k die -> L_s,k -> CS -> L_cs -> DC-.
    The load inductor between P and M is an ideal current source I_L during the event.
    """

    dut: list
    hs: DeviceParams
    Vbus: float = 800.0
    IL: float = 100.0
    La: float = 9e-9
    Lb: float = 3e-9
    Rp: float = 3.0
    Rloop: float = 2e-3
    LsH: float = 1e-9
    Ld: np.ndarray = None
    Ls: np.ndarray = None
    Lcs: float = 0.0
    Lg: np.ndarray = None
    Rg_on: np.ndarray = None
    Rg_off: np.ndarray = None
    Rcom: float = 0.0
    Re: np.ndarray = None
    Von: float = 18.0
    Voff: float = -4.0
    delay: np.ndarray = None
    t_slew: float = 2e-9
    topology: str = "individual_kelvin"
    LgH: float = 10e-9
    RgH: float = 3.5
    VoffH: float = -4.0

    def __post_init__(self):
        n = len(self.dut)
        arr = lambda v, d: np.full(n, float(d)) if v is None else np.asarray(v, dtype=float).reshape(-1) * np.ones(n)  # noqa: E731
        self.Ld = np.zeros((n, n)) if self.Ld is None else np.asarray(self.Ld, dtype=float).reshape(n, n)
        self.Ls = arr(self.Ls, 0.0)
        self.Lg = arr(self.Lg, 10e-9)
        self.Rg_on = arr(self.Rg_on, 3.5)
        self.Rg_off = arr(self.Rg_off, 3.5)
        self.Re = arr(self.Re, 0.0)
        self.delay = arr(self.delay, 0.0)
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {self.topology}")
        if np.any(np.linalg.eigvalsh(0.5 * (self.Ld + self.Ld.T)) < -1e-18):
            raise ValueError("branch inductance matrix is not positive semidefinite (|M| too large)")

    @property
    def n(self) -> int:
        return len(self.dut)

    @property
    def L_total_branch(self) -> float:
        """Commutation-loop inductance seen by branch 1 (for the L di/dt screen)."""
        return self.La + self.Lb + self.LsH + self.Ld[0, 0] + self.Ls[0] + self.Lcs


@dataclass
class CellRun:
    """Result of one cell simulation (dense samples + exact end-point energy integrals)."""

    t: np.ndarray
    x: np.ndarray  # core states at samples (n_core, n_t)
    q: np.ndarray  # quadrature states at samples
    out: dict  # measured / derived signals at samples
    energy: dict  # integrals over the whole run (J) and the ledger
    nfev: int
    runtime: float
    rtol: float
    edges: list = field(default_factory=list)


class Cell:
    """State equations of the commutation cell (see CellSpec for the circuit).

    Core states: v_gs,k, v_ds,k (die), i_k (branch drain current), i_g,k (gate loop),
    [i_s,k source-inductor current for the Kelvin-star topology], v_gs,H, v_ds,H, i_g,H,
    i_b (current in the lossy loop inductance L_b).  The inductor derivatives come from one
    constant linear system per topology (KVL of every loop, mutual terms included); the
    capacitor derivatives from each die's 2x2 capacitance matrix.  Quadrature states
    integrate every port and loss power so the energy ledger is closed on the solver's own
    accuracy, independently of the state equations.
    """

    def __init__(self, spec: CellSpec, edges: list[tuple[float, str]], start: str = "on"):
        self.s = spec
        n = spec.n
        self.n = n
        self.D = DevArrays(spec.dut)
        self.H = DevArrays([spec.hs])
        self.edges = sorted(edges)
        self.start = start
        self.kstar = spec.topology == "common_kelvin_star"
        k = 0
        self.i_vgs = slice(k, k + n)
        k += n
        self.i_vds = slice(k, k + n)
        k += n
        self.i_i = slice(k, k + n)
        k += n
        self.i_ig = slice(k, k + n)
        k += n
        if self.kstar:
            self.i_is = slice(k, k + n)
            k += n
        self.i_vgsH, self.i_vdsH, self.i_igH, self.i_ib = k, k + 1, k + 2, k + 3
        k += 4
        self.n_core = k
        self.qnames = ["E_src", "E_drv", "E_drvH", "E_load", "E_Rloop", "E_Rgate", "E_chH", "E_body"] + [f"E_ch{j}" for j in range(n)] + [f"Qg{j}" for j in range(n)] + [f"E_term{j}" for j in range(n)]
        self.n_q = len(self.qnames)
        self._build_matrix()
        self._build_drive()
        # column-shaped device constants: rows 0..n-1 = low-side branches, row n = high side
        c = lambda name: np.concatenate([getattr(self.D, name), getattr(self.H, name)]).reshape(-1, 1)  # noqa: E731
        self._c = {k: c(k) for k in ("Vth", "gm", "Vk", "Cgs", "Cgd0", "Vgd0", "Cds0", "Vds0", "Vf", "Rd", "Vs_ch", "Vs_d", "Rgint")}
        self._gmVs = self._c["gm"] * self._c["Vs_ch"]
        self._dVs = self._c["Vs_d"] / self._c["Rd"]
        # scalar copies for the fast right-hand side (pure Python floats: small N is faster than numpy)
        self._sc = [tuple(float(self._c[k][j, 0]) for k in ("Vth", "gm", "Vk", "Cgs", "Cgd0", "Vgd0", "Cds0", "Vds0", "Vf", "Rd", "Vs_ch", "Vs_d")) for j in range(n + 1)]
        self._Minv_np = self.Minv
        self._Re = [float(v) for v in spec.Re]
        self._RgH = float(spec.RgH + self.H.Rgint[0])
        self._seg = None

    # ---- inductance system ------------------------------------------------------------
    def _build_matrix(self):
        s, n = self.s, self.n
        Ltop = s.La + s.LsH
        Ld = s.Ld
        topo = s.topology
        if topo == "individual_kelvin":
            M = np.zeros((2 * n, 2 * n))
            M[:n, :n] = Ltop + s.Lcs + Ld + np.diag(s.Ls)
            M[n:, n:] = np.diag(s.Lg)
        elif topo == "common_kelvin_star":
            M = np.zeros((3 * n + 1, 3 * n + 1))
            M[:n, :n] = Ltop + Ld
            M[:n, 2 * n : 3 * n] = s.Lcs + np.diag(s.Ls)
            M[n : 2 * n, :n] = Ltop + Ld
            M[n : 2 * n, 3 * n] = 1.0
            M[2 * n : 3 * n, n : 2 * n] = np.diag(s.Lg)
            M[2 * n : 3 * n, 3 * n] = -1.0
            M[2 * n : 3 * n, :n] = -(Ltop + Ld)
            M[3 * n, 2 * n : 3 * n] = 1.0
            M[3 * n, :n] = -1.0
        elif topo == "common_source_node":
            M = np.zeros((2 * n, 2 * n))
            M[:n, :n] = Ltop + s.Lcs + Ld + np.diag(s.Ls)
            M[:n, n:] = np.diag(s.Ls)
            M[n:, :n] = np.diag(s.Ls)
            M[n:, n:] = np.diag(s.Lg + s.Ls)
        else:  # power_ground
            M = np.zeros((2 * n, 2 * n))
            M[:n, :n] = Ltop + s.Lcs + Ld + np.diag(s.Ls)
            M[:n, n:] = s.Lcs + np.diag(s.Ls)
            M[n:, :n] = s.Lcs + np.diag(s.Ls)
            M[n:, n:] = s.Lcs + np.diag(s.Lg + s.Ls)
        if np.linalg.cond(M) > 1e14:
            raise ValueError("singular inductance system (check L values)")
        self.M = M
        self.Minv = np.linalg.inv(M)

    # ---- gate command: piecewise-linear driver voltage per branch ----------------------
    def _build_drive(self):
        s, n = self.s, self.n
        lvl = {"on": s.Von, "off": s.Voff}
        tau = s.delay if s.topology != "common_kelvin_star" else np.zeros(n)
        self._tk, self._vk, self._on_t, self._on_v = [], [], [], []
        for k in range(n):
            tk, vk = [-1.0], [lvl[self.start]]
            on_t, on_v = [-1.0], [self.start == "on"]
            prev = lvl[self.start]
            for te, st in self.edges:
                ta = te + tau[k]
                tk += [ta, ta + s.t_slew]
                vk += [prev, lvl[st]]
                prev = lvl[st]
                on_t.append(ta)
                on_v.append(st == "on")
            self._tk.append(np.array(tk))
            self._vk.append(np.array(vk))
            self._on_t.append(np.array(on_t))
            self._on_v.append(np.array(on_v))

    def drive_at(self, t):
        """(v_drv (n, T), on (n, T)) for an array of times."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        v = np.array([np.interp(t, self._tk[k], self._vk[k]) for k in range(self.n)])
        on = np.array([self._on_v[k][np.searchsorted(self._on_t[k], t, side="right") - 1] for k in range(self.n)])
        return v, on

    def breakpoints(self) -> list[float]:
        s = self.s
        tau = s.delay if s.topology != "common_kelvin_star" else np.zeros(self.n)
        bp = set()
        for te, _ in self.edges:
            for tk in tau:
                bp.add(float(te + tk))
                bp.add(float(te + tk + s.t_slew))
        return sorted(bp)

    def _set_segment(self, ta: float, tb: float):
        tm = 0.5 * (ta + tb)
        va, _ = self.drive_at(ta)
        vb, _ = self.drive_at(tb)
        _, on = self.drive_at(tm)
        s = self.s
        Rgp = np.where(on[:, 0], s.Rg_on, s.Rg_off) + self.D.Rgint
        self._seg = (ta, va[:, 0:1], ((vb - va) / max(tb - ta, 1e-30))[:, 0:1], Rgp.reshape(-1, 1))
        self._segs = (ta, [float(v) for v in va[:, 0]], [float(v) for v in ((vb - va) / max(tb - ta, 1e-30))[:, 0]], [float(v) for v in Rgp])

    # ---- evaluation core (vectorised over columns) ----------------------------------------
    def _eval(self, X, vdrv, Rgp):
        """All algebraic quantities for core states X (n_core, T)."""
        s, n = self.s, self.n
        vgs, vds, i, ig = X[self.i_vgs], X[self.i_vds], X[self.i_i], X[self.i_ig]
        vgsH, vdsH, igH, ib = X[self.i_vgsH], X[self.i_vdsH], X[self.i_igH], X[self.i_ib]
        idc = i.sum(axis=0)
        W = s.Vbus - s.Rloop * idc - s.Rp * (idc - ib) - vdsH
        sig = ig.sum(axis=0)
        topo = s.topology
        ie = None
        if topo == "individual_kelvin":
            r = np.vstack([W - vds, vdrv - Rgp * ig - vgs])
        elif topo == "common_kelvin_star":
            ie = i + ig - X[self.i_is]
            vd = vdrv[0]
            r = np.vstack([W - vds, W - vds - s.Re.reshape(-1, 1) * ie, vd - s.Rcom * sig - Rgp * ig - vgs - (W - vds), np.zeros((1, X.shape[1]))])
        else:
            r = np.vstack([W - vds, vdrv - s.Rcom * sig - Rgp * ig - vgs])
        y = self.Minv @ r
        a = y[:n]
        g = y[n : 2 * n]
        A = a.sum(axis=0)
        # all dies at once: rows 0..n-1 low side, row n high side
        c = self._c
        VGS = np.vstack([vgs, vgsH[None, :] if vgsH.ndim == 1 else vgsH])
        VDS = np.vstack([vds, vdsH[None, :] if vdsH.ndim == 1 else vdsH])
        IG = np.vstack([ig, igH[None, :] if igH.ndim == 1 else igH])
        idH = idc - s.IL
        ID = np.vstack([i, idH[None, :]])
        cgd = c["Cgd0"] / np.sqrt(1.0 + np.maximum(VDS - VGS, 0.0) / c["Vgd0"])
        cds = c["Cds0"] / np.sqrt(np.maximum(1.0 + VDS / c["Vds0"], 0.25))
        cgs = c["Cgs"]
        ICH = self._gmVs * np.logaddexp(0.0, (VGS - c["Vth"]) / c["Vs_ch"]) * np.tanh(VDS / c["Vk"])
        IBD = self._dVs * np.logaddexp(0.0, (-VDS - c["Vf"]) / c["Vs_d"])
        bb = ID - ICH + IBD
        det = cgs * cgd + cgs * cds + cgd * cds
        DVGS = ((cgd + cds) * IG + cgd * bb) / det
        DVDS = (cgd * IG + (cgs + cgd) * bb) / det
        dvgs, dvds, ich, ibd = DVGS[:n], DVDS[:n], ICH[:n], IBD[:n]
        dvgsH, dvdsH, ichH, ibH = DVGS[n], DVDS[n], ICH[n], IBD[n]
        RgH = s.RgH + self.H.Rgint[0]
        digH = (s.VoffH - RgH * igH - vgsH) / s.LgH
        dib = s.Rp * (idc - ib) / s.Lb
        return dict(vgs=vgs, vds=vds, i=i, ig=ig, vgsH=vgsH, vdsH=vdsH, igH=igH, ib=ib, idc=idc, W=W, sig=sig, ie=ie, y=y, a=a, g=g, A=A,
                    dvgs=dvgs, dvds=dvds, ich=ich, ibd=ibd, dvgsH=dvgsH, dvdsH=dvdsH, ichH=ichH, ibH=ibH, digH=digH, dib=dib, RgH=RgH, vdrv=vdrv, Rgp=Rgp)

    def rhs_vec(self, t, xx):
        """Vectorised right-hand side (same equations as rhs; kept for cross-checking)."""
        ta, va, sl, Rgp = self._seg
        n = self.n
        X = xx[: self.n_core].reshape(-1, 1)
        z = self._eval(X, va + sl * (t - ta), Rgp)
        s = self.s
        dx = np.empty(self.n_core + self.n_q)
        dx[self.i_vgs] = z["dvgs"][:, 0]
        dx[self.i_vds] = z["dvds"][:, 0]
        dx[self.i_i] = z["a"][:, 0]
        dx[self.i_ig] = z["g"][:, 0]
        if self.kstar:
            dx[self.i_is] = z["y"][2 * n : 3 * n, 0]
        dx[self.i_vgsH] = z["dvgsH"][0]
        dx[self.i_vdsH] = z["dvdsH"][0]
        dx[self.i_igH] = z["digH"][0]
        dx[self.i_ib] = z["dib"][0]
        self._power(z, dx[self.n_core :].reshape(-1, 1))
        del s
        return dx

    def rhs(self, t, xx):
        """Right-hand side in plain Python floats (fast for the few branches used here)."""
        s, n = self.s, self.n
        ta, va, sl, Rgp = self._segs
        x = xx.tolist()
        vgs = x[0:n]
        vds = x[n : 2 * n]
        i = x[2 * n : 3 * n]
        ig = x[3 * n : 4 * n]
        k0 = self.i_vgsH
        vgsH, vdsH, igH, ib = x[k0], x[k0 + 1], x[k0 + 2], x[k0 + 3]
        idc = sum(i)
        sig = sum(ig)
        W = s.Vbus - s.Rloop * idc - s.Rp * (idc - ib) - vdsH
        dt = t - ta
        vdrv = [va[k] + sl[k] * dt for k in range(n)]
        topo = s.topology
        ie = None
        if topo == "individual_kelvin":
            r = [W - vds[k] for k in range(n)] + [vdrv[k] - Rgp[k] * ig[k] - vgs[k] for k in range(n)]
        elif topo == "common_kelvin_star":
            isk = x[4 * n : 5 * n]
            ie = [i[k] + ig[k] - isk[k] for k in range(n)]
            Re = self._Re
            vd = vdrv[0]
            r = [W - vds[k] for k in range(n)] + [W - vds[k] - Re[k] * ie[k] for k in range(n)] + [vd - s.Rcom * sig - Rgp[k] * ig[k] - vgs[k] - (W - vds[k]) for k in range(n)] + [0.0]
        else:
            r = [W - vds[k] for k in range(n)] + [vdrv[k] - s.Rcom * sig - Rgp[k] * ig[k] - vgs[k] for k in range(n)]
        y = (self._Minv_np @ np.array(r)).tolist()
        A = sum(y[:n])
        dx = [0.0] * (self.n_core + self.n_q)
        # dies (k < n low side, k == n high side)
        idH = idc - s.IL
        p_ch = [0.0] * (n + 1)
        p_body = 0.0
        sqrt, exp, log1p, tanh = math.sqrt, math.exp, math.log1p, math.tanh
        for k in range(n + 1):
            Vth, gm, Vk, Cgs, Cgd0, Vgd0, Cds0, Vds0, Vf, Rd, Vs_ch, Vs_d = self._sc[k]
            if k < n:
                vg, vd_, iG, iD = vgs[k], vds[k], ig[k], i[k]
            else:
                vg, vd_, iG, iD = vgsH, vdsH, igH, idH
            vdg = vd_ - vg
            cgd = Cgd0 / sqrt(1.0 + (vdg if vdg > 0.0 else 0.0) / Vgd0)
            u = 1.0 + vd_ / Vds0
            cds = Cds0 / sqrt(u if u > 0.25 else 0.25)
            a1 = (vg - Vth) / Vs_ch
            ov = Vs_ch * (a1 if a1 > 35.0 else log1p(exp(a1)))
            ich = gm * ov * tanh(vd_ / Vk)
            a2 = (-vd_ - Vf) / Vs_d
            ibd = Vs_d / Rd * (a2 if a2 > 35.0 else log1p(exp(a2)))
            b = iD - ich + ibd
            det = Cgs * cgd + Cgs * cds + cgd * cds
            if k < n:
                dx[k] = ((cgd + cds) * iG + cgd * b) / det
                dx[n + k] = (cgd * iG + (Cgs + cgd) * b) / det
            else:
                dx[k0] = ((cgd + cds) * iG + cgd * b) / det
                dx[k0 + 1] = (cgd * iG + (Cgs + cgd) * b) / det
            p_ch[k] = vd_ * ich
            p_body += -vd_ * ibd
        for k in range(n):
            dx[2 * n + k] = y[k]
            dx[3 * n + k] = y[n + k]
        if self.kstar:
            for k in range(n):
                dx[4 * n + k] = y[2 * n + k]
        RgH = self._RgH
        dx[k0 + 2] = (s.VoffH - RgH * igH - vgsH) / s.LgH
        dx[k0 + 3] = s.Rp * (idc - ib) / s.Lb
        q0 = self.n_core
        if topo != "common_kelvin_star":
            p_drv = sum(vdrv[k] * ig[k] for k in range(n))
        else:
            p_drv = vdrv[0] * sig
        pRg = sum(Rgp[k] * ig[k] * ig[k] for k in range(n)) + s.Rcom * sig * sig + RgH * igH * igH
        if ie is not None:
            pRg += sum(self._Re[k] * ie[k] * ie[k] for k in range(n))
        dx[q0] = s.Vbus * idc
        dx[q0 + 1] = p_drv
        dx[q0 + 2] = s.VoffH * igH
        dx[q0 + 3] = (vdsH + s.LsH * A) * s.IL
        dx[q0 + 4] = s.Rloop * idc * idc + s.Rp * (idc - ib) ** 2
        dx[q0 + 5] = pRg
        dx[q0 + 6] = p_ch[n]
        dx[q0 + 7] = p_body
        for k in range(n):
            dx[q0 + 8 + k] = p_ch[k]
            dx[q0 + 8 + n + k] = ig[k]
            dx[q0 + 8 + 2 * n + k] = vds[k] * i[k]
        return np.array(dx)

    def _power(self, z, q):
        """Port and loss powers (rows of the quadrature block), vectorised over columns."""
        s, n = self.s, self.n
        i, ig, vds = z["i"], z["ig"], z["vds"]
        idc, sig = z["idc"], z["sig"]
        if s.topology != "common_kelvin_star":
            p_drv = (z["vdrv"] * ig).sum(axis=0)
        else:
            p_drv = z["vdrv"][0] * sig
        pRg = (z["Rgp"] * ig * ig).sum(axis=0) + s.Rcom * sig**2 + z["RgH"] * z["igH"] ** 2
        if z["ie"] is not None:
            pRg = pRg + (s.Re.reshape(-1, 1) * z["ie"] ** 2).sum(axis=0)
        q[0] = s.Vbus * idc
        q[1] = p_drv
        q[2] = s.VoffH * z["igH"]
        q[3] = (z["vdsH"] + s.LsH * z["A"]) * s.IL
        q[4] = s.Rloop * idc * idc + s.Rp * (idc - z["ib"]) ** 2
        q[5] = pRg
        q[6] = z["vdsH"] * z["ichH"]
        q[7] = (-vds * z["ibd"]).sum(axis=0) + (-z["vdsH"]) * z["ibH"]
        q[8 : 8 + n] = vds * z["ich"]
        q[8 + n : 8 + 2 * n] = ig
        q[8 + 2 * n : 8 + 3 * n] = vds * i
        return q

    def jac(self, t, xx):
        """Finite-difference Jacobian over the core states only (quadrature states do not feed back)."""
        f0 = self.rhs(t, xx)
        m = xx.size
        J = np.zeros((m, m))
        for j in range(self.n_core):
            hh = 1e-7 * max(abs(xx[j]), 1e-3)
            xp = xx.copy()
            xp[j] += hh
            J[:, j] = (self.rhs(t, xp) - f0) / hh
        return J

    # ---- stored energy ----------------------------------------------------------------------
    def stored(self, x):
        s, n, D, H = self.s, self.n, self.D, self.H
        vgs, vds, i, ig = x[self.i_vgs], x[self.i_vds], x[self.i_i], x[self.i_ig]
        idc = float(np.sum(i))
        if s.topology == "individual_kelvin":
            isk, ics = i, idc
        elif self.kstar:
            isk, ics = x[self.i_is], idc
        elif s.topology == "common_source_node":
            isk, ics = i + ig, idc
        else:
            isk, ics = i + ig, idc + float(np.sum(ig))
        Wm = 0.5 * s.La * idc**2 + 0.5 * s.Lb * x[self.i_ib] ** 2 + 0.5 * s.LsH * (idc - s.IL) ** 2 + 0.5 * float(i @ s.Ld @ i) + 0.5 * float(np.sum(s.Ls * isk**2)) + 0.5 * s.Lcs * ics**2
        Wm += 0.5 * float(np.sum(s.Lg * ig**2)) + 0.5 * s.LgH * x[self.i_igH] ** 2
        Wc = float(np.sum(D.E_cap(vgs, vds))) + float(H.E_cap(np.array([x[self.i_vgsH]]), np.array([x[self.i_vdsH]]))[0])
        del n
        return Wm + Wc

    # ---- initial DC states ----------------------------------------------------------------------
    def dc_state(self) -> np.ndarray:
        s, n, D, H = self.s, self.n, self.D, self.H
        x = np.zeros(self.n_core)
        if self.start == "on":
            f = lambda v: float(np.sum(D.ich(np.full(n, s.Von), np.full(n, v)))) - s.IL  # noqa: E731
            vstar = brentq(f, 0.0, 200.0, xtol=1e-14)
            i = D.ich(np.full(n, s.Von), np.full(n, vstar))
            x[self.i_vgs] = s.Von
            x[self.i_vds] = vstar
            x[self.i_i] = i
            if self.kstar:
                x[self.i_is] = i
            idc = float(np.sum(i))
            x[self.i_vdsH] = s.Vbus - s.Rloop * idc - vstar
            x[self.i_ib] = idc
        else:
            vf = brentq(lambda v: float(H.ibody(np.array([-v]))[0]) - s.IL, 0.0, 50.0, xtol=1e-14)
            x[self.i_vgs] = s.Voff
            x[self.i_vdsH] = -vf
            x[self.i_vds] = s.Vbus + vf
        x[self.i_vgsH] = s.VoffH
        return x

    # ---- measured / derived outputs at sample times -------------------------------------------
    def outputs(self, t_arr, X):
        """Signals a probe would see (and model-only die quantities), vectorised over samples."""
        s, n = self.s, self.n
        vdrv, on = self.drive_at(t_arr)
        Rgp = np.where(on, s.Rg_on.reshape(-1, 1), s.Rg_off.reshape(-1, 1)) + self.D.Rgint.reshape(-1, 1)
        z = self._eval(X, vdrv, Rgp)
        a, g = z["a"], z["g"]
        if s.topology == "individual_kelvin":
            dis = a
            dics = z["A"]
        elif self.kstar:
            dis = z["y"][2 * n : 3 * n]
            dics = dis.sum(axis=0)
        elif s.topology == "common_source_node":
            dis = a + g
            dics = z["A"]
        else:
            dis = a + g
            dics = (a + g).sum(axis=0)
        vgsK = z["vgs"] + self.D.Rgint.reshape(-1, 1) * z["ig"]
        vgsHK = z["vgsH"] + self.H.Rgint[0] * z["igH"]
        Ls = s.Ls.reshape(-1, 1)
        q = np.empty((self.n_q, X.shape[1]))
        self._power(z, q)
        return {
            "idc": z["idc"],
            "vM": z["W"] - (s.La + s.LsH) * z["A"],
            "vgsK": vgsK,
            "vgsPS": vgsK + Ls * dis,
            "vgsDC": vgsK + Ls * dis + s.Lcs * dics,
            "vdsPS": z["vds"] + Ls * dis,
            "vgsHK": vgsHK,
            "vgsHPS": vgsHK + s.LsH * z["A"],
            "ich": z["ich"],
            "ichH": z["ichH"],
            "ibH": z["ibH"],
            "ibd": z["ibd"],
            "vdrv": vdrv,
            "didt": a,
            "dics": dics,
            "ig": z["ig"],
            "p": q,
        }

    # ---- integration -----------------------------------------------------------------------------
    def simulate(self, t_end: float, n_samples: int = 4000, rtol: float = 1e-8, method: str = "LSODA", x0=None, t_fine: list | None = None) -> CellRun:
        import time as _time

        t_start = _time.perf_counter()
        n = self.n
        x0 = self.dc_state() if x0 is None else np.asarray(x0, dtype=float)
        xx = np.concatenate([x0, np.zeros(self.n_q)])
        W0 = self.stored(x0)
        atol = np.empty(xx.size)
        atol[self.i_vgs] = 1e-6
        atol[self.i_vds] = 1e-5
        atol[self.i_i] = 1e-6
        atol[self.i_ig] = 1e-8
        if self.kstar:
            atol[self.i_is] = 1e-6
        atol[self.i_vgsH], atol[self.i_vdsH], atol[self.i_igH], atol[self.i_ib] = 1e-6, 1e-5, 1e-8, 1e-6
        atol[self.n_core :] = 1e-13
        bps = [b for b in self.breakpoints() if 0.0 < b < t_end]
        knots = [0.0] + bps + [t_end]
        sols = []
        nfev = 0
        for ta, tb in zip(knots[:-1], knots[1:]):
            if tb - ta <= 0:
                continue
            self._set_segment(ta, tb)
            kw = {"jac": self.jac} if method in ("LSODA", "Radau", "BDF") else {}
            sol = solve_ivp(self.rhs, (ta, tb), xx, method=method, rtol=rtol, atol=atol, dense_output=True, max_step=max((tb - ta) / 20, 1e-12), **kw)
            if sol.status != 0:
                raise RuntimeError(f"cell integration failed at t={sol.t[-1]:.4g}: {sol.message}")
            nfev += sol.nfev
            sols.append((ta, tb, sol))
            xx = sol.y[:, -1].copy()
        grid = np.linspace(0.0, t_end, n_samples)
        for a, b, m in t_fine or []:
            grid = np.concatenate([grid, np.linspace(a, b, m)])
        grid = np.unique(np.concatenate([grid, np.array(knots)]))
        Y = np.empty((xx.size, grid.size))
        for ta, tb, sol in sols:
            m = (grid >= ta) & (grid <= tb)
            if np.any(m):
                Y[:, m] = sol.sol(grid[m])
        X = Y[: self.n_core]
        Qd = Y[self.n_core :]
        out = self.outputs(grid, X)
        endq = xx[self.n_core :]
        E = {name: float(endq[k]) for k, name in enumerate(self.qnames)}
        W1 = self.stored(xx[: self.n_core])
        E["dW"] = W1 - W0
        E_in = E["E_src"] + E["E_drv"] + E["E_drvH"]
        E_diss = E["E_Rloop"] + E["E_Rgate"] + E["E_chH"] + E["E_body"] + sum(E[f"E_ch{j}"] for j in range(n))
        resid = E_in - E["E_load"] - E_diss - E["dW"]
        norm = max(abs(E["E_src"]), abs(E["E_load"]), E_diss, 1e-12)
        E.update({"E_in": E_in, "E_diss": E_diss, "residual": resid, "normalised": resid / norm, "norm": norm})
        return CellRun(grid, X, Qd, out, E, nfev, _time.perf_counter() - t_start, rtol, list(self.edges))


# ======================================================================================
# Experiment 1: synthetic A/B loss comparison (level A, postprocessed estimate)
# ======================================================================================


def _loss(Irms, R, E, fs, E_add=0.0):
    """Device loss of the synthetic comparison: I_rms^2 R + (E + E_add) f_s (one on/off pair per period)."""
    return Irms * Irms * R + (E + E_add) * fs


def _pwl_loss(Irms: float, R: float, E: float, fs: float, D: float = 0.5, cycles: int = 7) -> float:
    """Independent path: rectangular switch-current waveform with the given RMS over whole cycles.

    Conduction energy from the exact PWL integral of i^2 R; switching energy by counting
    on/off edge pairs in the waveform; power = energy / time.  No use of I_rms^2 R + E f.
    """
    T = 1.0 / fs
    Ion = Irms / math.sqrt(D)
    t, y = [0.0], []
    for k in range(cycles):
        t += [k * T + D * T, (k + 1) * T]
        y += [Ion, 0.0]
    w = PWL(np.array(t), np.array(y), np.array(y))
    E_cond = R * w.integral_sq()
    rising = sum(1 for k in range(1, len(y)) if y[k] > y[k - 1]) + (1 if y[0] > 0 else 0)
    return (E_cond + rising * E) / w.span


_DATASHEET_ROWS = [
    ("R_DS(on)", "typ/max, T_j, V_GS, I_D, package 포함 범위", "고온 전도손실·병렬 수"),
    ("E_on / E_off", "V_DC, I_D, T_j, R_g, V_GS, 상대 diode/스위치, 적분 정의", "실제 사건당 손실"),
    ("Q_oss / E_oss", "전압에 따른 비선형 곡선", "ZVS 전하·hard turn-on 에너지"),
    ("Q_g / Q_gd", "V_DS·I_D·gate swing", "구동 전류·Miller 구간"),
    ("V_GS abs max / 권장", "transient 허용조건 포함", "gate overshoot·구동 여유"),
    ("SC withstand / SOA", "시험전압·온도·gate·보증 유무", "보호시간·고장 생존성"),
    ("R_th / Z_th", "junction→case/fluid 등 참조 경계", "사용할 냉각망"),
    ("Qualification / status", "정확한 부품 번호·문서 revision", "자동차 적용·변경 리스크"),
]


def run_loss_ab(v: dict) -> Result:
    res = Result("FL02", "loss_ab", "A (postprocessed loss estimate)")
    RA, EA, RB, EB, I, fs = v["RA"], v["EA"], v["RB"], v["EB"], v["Irms"], v["fs"]
    bnd = v["coss_boundary"]
    addA = v["EossA"] if bnd == "excluded" else 0.0
    addB = v["EossB"] if bnd == "excluded" else 0.0
    PA = _loss(I, RA, EA, fs, addA)
    PB = _loss(I, RB, EB, fs, addB)
    tb = all(abs(v[k] - d) <= 1e-12 * max(1.0, abs(d)) for k, d in (("RA", 0.04), ("EA", 100e-6), ("RB", 0.025), ("EB", 180e-6), ("Irms", 10.0))) and bnd == "included"
    tb100 = tb and abs(fs - 100e3) < 1e-6
    res.add_metric("PA", "소자 A 손실 (I²R + E·f_s)", PA, "W", ref=14.0 if tb100 else None, ref_label="교재 14 W", tol=1e-9, basis="소자 1개, 주기 평균, postprocessed")
    res.add_metric("PB", "소자 B 손실", PB, "W", ref=20.5 if tb100 else None, ref_label="교재 20.5 W", tol=1e-9, basis="소자 1개, 주기 평균, postprocessed")
    res.add_metric("PA_cond", "A 전도손실 I_rms²·R", I * I * RA, "W", basis="RMS 전류 기준")
    res.add_metric("PA_sw", "A 스위칭손실 E·f_s", (EA + addA) * fs, "W", basis="사건당 E × 주기당 1쌍")
    res.add_metric("PB_cond", "B 전도손실", I * I * RB, "W")
    res.add_metric("PB_sw", "B 스위칭손실", (EB + addB) * fs, "W")
    f_x = None
    dE = (EB + addB) - (EA + addA)
    if abs(dE) > 0 and (RA - RB) / dE > 0:
        f_x = I * I * (RA - RB) / dE
    res.add_metric("f_cross", "두 소자 손실이 같아지는 f_s", f_x if f_x is not None else float("nan"), "Hz", ref=18.75e3 if tb else None, ref_label="교재 수치로 계산 18.75 kHz", tol=1e-9, basis="이보다 낮은 f_s에서는 낮은 R_DS(on)이 유리")
    PA20, PB20 = _loss(I, RA, EA, 20e3, addA), _loss(I, RB, EB, 20e3, addB)
    res.add_metric("PA_20k", "A 손실 @ 20 kHz", PA20, "W", ref=6.0 if tb else None, ref_label="교재 6 W", tol=1e-9)
    res.add_metric("PB_20k", "B 손실 @ 20 kHz", PB20, "W", ref=6.1 if tb else None, ref_label="교재 6.1 W", tol=1e-9)
    rank = "A가 낮음" if PA < PB else ("B가 낮음" if PB < PA else "같음")
    res.add_metric("rank", "이 f_s에서의 순위 (합성 데이터)", rank, "", basis="제품 순위가 아니라 비교 논리 훈련")
    # gate drive power: reported, never added to the junction loss
    PgA = v["QgA"] * v["dVg"] * fs
    PgB = v["QgB"] * v["dVg"] * fs
    res.add_metric("PgA", "A 게이트 구동 전력 Q_g·ΔV_g·f_s (별도)", PgA, "W", basis="드라이버·게이트 저항에서 소모 — 소자 손실에 더하지 않음")
    res.add_metric("PgB", "B 게이트 구동 전력 (별도)", PgB, "W", basis="대부분 드라이버 쪽 발열")
    # Coss boundary unknown: rank under both definitions
    unresolved = False
    if bnd == "unknown":
        lo_A, hi_A = _loss(I, RA, EA, fs), _loss(I, RA, EA, fs, v["EossA"])
        lo_B, hi_B = _loss(I, RB, EB, fs), _loss(I, RB, EB, fs, v["EossB"])
        res.add_metric("PA_range", "A 손실 범위 (E에 Coss 포함 ↔ 미포함)", f"{lo_A:.4g} – {hi_A:.4g} W", "", basis="적분 정의 미확인")
        res.add_metric("PB_range", "B 손실 범위", f"{lo_B:.4g} – {hi_B:.4g} W", "")
        rank_lo = lo_A < lo_B
        rank_hi = hi_A < hi_B
        worst_A_best_B = hi_A < lo_B
        worst_B_best_A = hi_B < lo_A
        unresolved = not (worst_A_best_B or worst_B_best_A)
        res.add_metric("rank_robust", "두 정의 모두에서 순위가 같은가", "예" if rank_lo == rank_hi and not unresolved else "아니오 (구간이 겹침)", "")
    # frequency sweep
    fsw = np.geomspace(2e3, 300e3, 121)
    res.add_series("sw_A", "소자 A (R 40 mΩ, E 100 µJ)" if tb else "소자 A", "W", fsw.tolist(), [_loss(I, RA, EA, f, addA) for f in fsw])
    res.add_series("sw_B", "소자 B (R 25 mΩ, E 180 µJ)" if tb else "소자 B", "W", fsw.tolist(), [_loss(I, RB, EB, f, addB) for f in fsw])
    res.add_series("sw_Ac", "A 전도 성분", "W", [fsw[0], fsw[-1]], [I * I * RA] * 2, dash=True)
    res.add_series("sw_Bc", "B 전도 성분", "W", [fsw[0], fsw[-1]], [I * I * RB] * 2, dash=True)
    vl = [{"x": fs, "label": f"f_s {fs / 1e3:.4g} kHz"}]
    if f_x is not None and 2e3 <= f_x <= 300e3:
        vl.append({"x": f_x, "label": f"교차 {f_x / 1e3:.4g} kHz"})
    res.add_plot("p_fs", "f_s에 따른 소자 손실: 낮은 R_DS(on)이 항상 유리하지 않다", ["sw_A", "sw_B", "sw_Ac", "sw_Bc"], x_label="f_s", x_unit="Hz", y_label="손실", y_unit="W", kind="xy", log_x=True, log_y=True, vlines=vl, level="A",
                 markers=[{"x": fs, "y": PA, "label": "A"}, {"x": fs, "y": PB, "label": "B"}],
                 proved="전도손실은 f_s와 무관하고 스위칭손실은 f_s에 비례하므로, 두 합성 소자의 순위가 교차 주파수에서 뒤바뀐다는 것을 보였다.",
                 not_yet="E는 한 운전점(전류·온도·R_g·V_DC)의 값이다. 실제 E(I, T)와 R_DS(on)(T)는 운전점마다 바뀌므로 제품 순위가 아니다(FL03·EX01).")
    # N-parallel sweep with explicit scaling assumptions
    Ns = list(range(1, 9))
    kc = v["k_cap"]
    for tag, R, E, Qg, add in (("A", RA, EA, v["QgA"], addA), ("B", RB, EB, v["QgB"], addB)):
        cond = [I * I * R / N for N in Ns]
        sw = [(E + add) * fs * ((1 - kc) + kc * N) for N in Ns]
        res.add_series(f"n_{tag}", f"{tag}: 소자 손실 합 (전도+스위칭)", "W", Ns, [a + b for a, b in zip(cond, sw)], style="points" if False else "line")
        res.add_series(f"n_{tag}_g", f"{tag}: 게이트 구동 전력 N·Q_g·ΔV·f_s (별도)", "W", Ns, [N * Qg * v["dVg"] * fs for N in Ns], dash=True)
    NA = math.sqrt(I * I * RA / max((EA + addA) * fs * kc, 1e-30)) if kc > 0 else float("inf")
    NB = math.sqrt(I * I * RB / max((EB + addB) * fs * kc, 1e-30)) if kc > 0 else float("inf")
    res.add_metric("N_opt_A", "A: 손실 최소 병렬 수 √(I²R / (k_cap·E·f_s)) (연속 근사)", NA, "", basis="가정: 전도 ∝ 1/N, E의 k_cap 비율 ∝ N")
    res.add_metric("N_opt_B", "B: 손실 최소 병렬 수 (연속 근사)", NB, "")
    res.add_plot("p_n", "병렬 수 N에 따른 손실 (명시적 스케일링 가정)", ["n_A", "n_B", "n_A_g", "n_B_g"], x_label="병렬 소자 수 N", x_unit="", y_label="전력", y_unit="W", kind="xy", level="A",
                 proved="전도손실이 1/N로 줄어도 전하·용량 관련 스위칭 성분과 게이트 구동 전력은 N에 비례해 늘어, N을 늘리는 것만으로 손실이 계속 줄지 않는다는 것을 가정과 함께 보였다.",
                 not_yet="병렬 소자 간 정적·동적 전류분담, gate loop 비대칭, 기생 발진은 이 식에 없다 — EX03에서 branch별 회로로 따로 푼다. k_cap은 합성 가정이다.")
    # independent path: PWL waveform integration + edge counting
    pA_pwl = _pwl_loss(I, RA, EA + addA, fs)
    pB_pwl = _pwl_loss(I, RB, EB + addB, fs)
    res.add_check(check_close("A 손실: 식 vs PWL 파형 적분 + edge 계수", pA_pwl, PA, 1e-12, "사각 스위치 전류(같은 RMS)의 ∫i²R dt 정확 적분과 on/off 쌍 계수 vs I²R + E·f_s", True, "W"))
    res.add_check(check_close("B 손실: 식 vs PWL 파형 적분 + edge 계수", pB_pwl, PB, 1e-12, "같은 독립 경로", True, "W"))
    if f_x is not None:
        g = lambda f: _pwl_loss(I, RA, EA + addA, f) - _pwl_loss(I, RB, EB + addB, f)  # noqa: E731
        try:
            f_num = brentq(g, 100.0, 1e8, xtol=1e-9, rtol=1e-14)
            res.add_check(check_close("교차 주파수: 닫힌 식 vs 파형 경로의 근 찾기", f_num, f_x, 1e-9, "PWL 파형 손실 차이의 brentq 근 vs I²ΔR/ΔE", True, "Hz"))
        except ValueError:
            pass
    # tables
    rows = [
        ["전도 (채널 I²R)", "포함", "소자 채널", "R_DS(on)의 온도·V_GS 조건을 E와 같은 T_j로 맞춘다"],
        ["스위칭 E_on + E_off", "포함 (사건당 1쌍)", "소자 (+ 상대 소자 일부)", "데이터의 적분창·참조면·R_g가 다르면 비교 불가"],
        ["Coss 에너지 (E_oss)", {"included": "E에 이미 포함으로 간주 (교재 예)", "excluded": "E에 없음 → E_oss·f_s를 더함", "unknown": "정의 미확인 → 두 경계 모두 계산"}[bnd], "hard turn-on 시 채널", "E_on에 포함됐는데 또 더하면 중복 계산"],
        ["역회복 / diode 전하 (Q_rr, Q_c)", "이 비교에서 별도 항 없음", "상대 diode와 스위칭 소자", "E_on 측정에 이미 들어간 recovery를 E_rec로 또 더하지 않는다"],
        ["dead-time diode 도통", "미포함 (상대 소자 손실)", "상대 소자 body diode", "half-bridge leg 전체로 볼 때 따로 더한다 (FL03)"],
        ["게이트 구동 Q_g·ΔV·f_s", "미포함 (별도 표시)", "드라이버·외부/내부 R_g", "소자 접합 손실과 섞지 않는다"],
    ]
    res.tables.append(Table("t_bound", "손실 포함 경계 (이 비교가 무엇을 더했고 무엇을 뺐나)", ["손실 항목", "이 계산에서", "어디서 소모", "중복 방지 규칙"], rows,
                            note="이 결과는 이상 전류(I_rms)에 사건 에너지를 곱해 더한 postprocessed loss estimate다. 전력단 방정식에 결합된 손실이 아니다."))
    res.tables.append(Table("t_breakdown", f"손실 분해 @ f_s = {fs / 1e3:.4g} kHz", ["소자", "전도 [W]", "스위칭 [W]", "합계 [W]", "게이트 구동 (별도) [W]"],
                            [["A", I * I * RA, (EA + addA) * fs, PA, PgA], ["B", I * I * RB, (EB + addB) * fs, PB, PgB]]))
    res.tables.append(Table("t_datasheet", "실제 부품 데이터 추출 양식 (별도 데이터셋 — 값 없음)", ["항목", "읽어야 할 조건", "설계 질문", "부품 A", "부품 B", "출처 (part/rev/URL/조건)"],
                            [[a, b, c, "MISSING_INPUT", "MISSING_INPUT", "MISSING_INPUT"] for a, b, c in _DATASHEET_ROWS],
                            note="합성 A/B는 교재의 비교 논리용 수치다. 실제 부품 값은 exact part·revision·시험조건·URL과 함께 이 양식에 따로 채워야 DATASHEET 출처가 된다. 이 앱은 실제 부품 수치를 싣지 않는다."))
    if unresolved:
        res.verdict("UNRESOLVED_RANKING", "E 데이터의 Coss 포함 여부를 모르면 두 소자의 손실 구간이 겹쳐 순위를 확정할 수 없다 (적분 정의 확인 필요)")
    else:
        res.verdict("PASS_WITHIN_MODEL", f"합성 A/B 비교(A 수준 postprocessed 추정)가 손계산·독립 파형 경로와 일치 — 이 f_s에서 {rank}")
    res.verdict("MISSING_INPUT", "실제 부품 데이터셋 없음: 추출 양식의 값이 모두 MISSING_INPUT이며 실제 부품·제품 순위는 주장하지 않는다")
    res.assumptions += [
        "R은 실제 운전온도의 값, E는 한 운전점(전류·전압·온도·R_g)의 사건당 E_on+E_off (교재 합성 소자)",
        "주기마다 on/off 1쌍, 스위치 RMS 10 A — 파형 모양과 무관하게 RMS로 전도손실 계산",
        f"병렬 sweep: 전도 ∝ 1/N, E의 k_cap = {kc:g} 비율(전하·용량 성분) ∝ N, 나머지는 N과 무관, 게이트 전하 ∝ N (합성 가정)",
        "게이트 구동 전력은 소자 손실에 더하지 않고 별도 표시",
    ]
    res.not_valid_for += ["실제 부품의 손실 순위·선정", "E(I, T)·R_DS(on)(T)가 변하는 운전영역 전체 (FL03, EX01)", "병렬 소자의 전류 불균형·발진 (EX03)"]
    res.interpretation = (
        f"전도손실은 I_rms²R이라 f_s와 무관하고, 스위칭손실은 사건 에너지 × 사건 수라 f_s에 비례한다. 그래서 R이 작고 E가 큰 B는 {fs / 1e3:.4g} kHz에서 "
        f"{PB:.4g} W로 A의 {PA:.4g} W보다 뜨겁고, 교차 주파수 {f_x / 1e3 if f_x else float('nan'):.4g} kHz 아래에서만 B가 유리하다. "
        "‘낮은 R_DS(on)으로 drop-in 교체했더니 더 뜨겁다’는 현상의 첫 번째 설명이다 — 그 다음은 gate 파형·dead time·열 경계·측정 조건 비교다."
    )
    return res


# ======================================================================================
# Experiment 2: gate / Miller / loop screens and the protection timeline (A + gate-charge ODE)
# ======================================================================================


def gate_charge_ode(Vdrv, Voff, Vpl, Rtot, Q1, Qgd, C2, rtol=1e-11):
    """Gate-charge ODE: dQ/dt = (V_drv - v_gs(Q))/R with v_gs(Q) piecewise (C1 ramp, plateau, C2 ramp).

    Returns the event times (plateau entry and exit), a sampled waveform and the total charge.
    """
    C1 = Q1 / (Vpl - Voff)

    def vgs_of_Q(Q):
        if Q < Q1:
            return Voff + Q / C1
        if Q <= Q1 + Qgd:
            return Vpl
        return Vpl + (Q - Q1 - Qgd) / C2

    def f(t, y):
        return [(Vdrv - vgs_of_Q(y[0])) / Rtot]

    def ev1(t, y):
        return y[0] - Q1

    ev1.terminal = True

    def ev2(t, y):
        return y[0] - Q1 - Qgd

    ev2.terminal = True
    tau1 = Rtot * C1
    tau2 = Rtot * C2
    t_end = 10 * (tau1 + tau2) + 10 * Qgd * Rtot / max(Vdrv - Vpl, 1e-9)
    s1 = solve_ivp(f, (0, t_end), [0.0], events=ev1, rtol=rtol, atol=1e-18, dense_output=True, method="DOP853")
    t1 = float(s1.t_events[0][0])
    s2 = solve_ivp(f, (t1, t_end), [Q1], events=ev2, rtol=rtol, atol=1e-18, dense_output=True, method="DOP853")
    t2 = float(s2.t_events[0][0])
    t3 = t2 + 6 * tau2
    s3 = solve_ivp(f, (t2, t3), [Q1 + Qgd], rtol=rtol, atol=1e-18, dense_output=True, method="DOP853")
    ts = np.concatenate([np.linspace(0, t1, 120), np.linspace(t1, t2, 60), np.linspace(t2, t3, 160)])
    Qs = np.concatenate([s1.sol(np.linspace(0, t1, 120))[0], s2.sol(np.linspace(t1, t2, 60))[0], s3.sol(np.linspace(t2, t3, 160))[0]])
    vs = np.array([vgs_of_Q(q) for q in Qs])
    ig = (Vdrv - vs) / Rtot
    Qtot = Q1 + Qgd + C2 * (Vdrv - Vpl)
    return {"t1": t1, "t2": t2, "t": ts, "Q": Qs, "v": vs, "ig": ig, "Qtot": Qtot, "C1": C1}


SC_STAGES = [("t_blank", "blanking", "blanking"), ("t_filter", "검출·필터", "검출"), ("t_prop", "전파·로직", "전파"), ("t_soft", "gate 방전·전류 소거", "소거")]


def run_gate_protect(v: dict) -> Result:
    res = Result("FL02", "gate_protect", "A (1차 screen) + B (gate-charge ODE)")
    L, didt, C, dvdt = v["L_loop"], v["didt"] * 1e6, v["C_par"], v["dvdt"] * 1e6
    dV = L * didt
    icm = C * dvdt
    Rtot = v["R_drv"] + v["Rg_ext"] + v["Rg_int"]
    Ig = (v["Vdrv"] - v["Vpl"]) / Rtot
    tM = v["Qgd"] / Ig
    tb_l = abs(L - 15e-9) < 1e-18 and abs(v["didt"] - 5000) < 1e-9
    tb_c = abs(C - 1e-9) < 1e-18 and abs(v["dvdt"] - 30000) < 1e-9
    tb_m = abs(v["Qgd"] - 20e-9) < 1e-18 and abs(Ig - 2.0) < 1e-12
    res.add_metric("dV", "loop 전압 ΔV ≈ L_loop·di/dt", dV, "V", ref=75.0 if tb_l else None, ref_label="교재 75 V", tol=1e-9, basis="1차 크기 screen (overshoot 상한 아님)")
    res.add_metric("icm", "변위전류 i ≈ C_par·dv/dt", icm, "A", ref=30.0 if tb_c else None, ref_label="교재 30 A", tol=1e-9, basis="edge 동안의 순간값 — LISN 연속 30 A·EMC 불합격 판정 아님")
    res.add_metric("Ig", "plateau gate 전류 (V_drv − V_pl)/(R_drv + R_g + R_g,int)", Ig, "A", basis="드라이버 광고 peak 전류와 다르다")
    res.add_metric("Ipk_drv", "드라이버 광고 peak 전류 (비교용)", v["Ipk_drv"], "A", note=f"실제 plateau 전류는 그 {Ig / v['Ipk_drv'] * 100:.0f} %")
    res.add_metric("tM", "Miller 시간 t ≈ Q_gd / I_g", tM, "s", ref=10e-9 if tb_m else None, ref_label="교재 10 ns", tol=1e-9, basis="1차 근사 (Q_gd의 V_DS 조건 의존)")
    # gate-charge ODE (independent path for the plateau time)
    gc = gate_charge_ode(v["Vdrv"], v["Voff"], v["Vpl"], Rtot, v["Qgs"], v["Qgd"], v["C2"])
    t_pl = gc["t2"] - gc["t1"]
    res.add_metric("t_plateau_ode", "gate-charge ODE의 plateau 지속시간", t_pl, "s", ref=tM, ref_label="Q_gd/I_g", tol=1e-7)
    t1_cf = Rtot * gc["C1"] * math.log((v["Vdrv"] - v["Voff"]) / (v["Vdrv"] - v["Vpl"]))
    res.add_metric("t_delay", "plateau 도달까지 (R·C₁ 충전)", gc["t1"], "s", basis="turn-on 지연의 1차 모델")
    res.add_check(check_close("plateau 시간: event ODE vs Q_gd/I_g", t_pl, tM, 1e-7, "piecewise Q–v 모델 ODE(DOP853)의 event 시각 차 vs 닫힌 식", True, "s"))
    res.add_check(check_close("plateau 도달: event ODE vs R·C₁·ln(ΔV₀/ΔV₁)", gc["t1"], t1_cf, 1e-7, "ODE event vs RC 충전 닫힌 식", True, "s"))
    Pg = gc["Qtot"] * (v["Vdrv"] - v["Voff"]) * v["fs"]
    res.add_metric("Qg_tot", "총 gate 전하 Q_g (V_off → V_drv)", gc["Qtot"], "C", basis="합성 Q–v 곡선")
    res.add_metric("P_gate", "게이트 구동 전력 Q_g·(V_drv − V_off)·f_s", Pg, "W", basis="드라이버·R_g에서 소모 (FL02 실험 1의 별도 항)")
    # protection timeline
    stages = [(k, lab, v[k]) for k, lab, _ in SC_STAGES]
    t_chain = sum(x for _, _, x in stages)
    t_sc = v["t_sc"]
    tb_sc = all(abs(v[k] - d) < 1e-15 for k, d in (("t_blank", 0.5e-6), ("t_filter", 0.3e-6), ("t_prop", 0.2e-6), ("t_soft", 0.8e-6), ("t_sc", 3e-6)))
    res.add_metric("t_chain", "보호 체인 합 (blanking + 검출 + 전파 + 소거)", t_chain, "s", ref=1.8e-6 if tb_sc else None, ref_label="교재 1.8 µs", tol=1e-9, basis="합성 보호 예")
    res.add_metric("t_left", "가정한 SC 생존시간 − 체인", t_sc - t_chain, "s", ref=1.2e-6 if tb_sc else None, ref_label="교재 1.2 µs", tol=1e-9, note="확정 마진이 아니다: 온도·분산·최악조건 검토 전")
    res.add_metric("t_sys", "시스템 안전반응 시간 (별도 계층)", v["t_sys"], "s", basis="토크·ASC/FW 등 시스템 반응 — 소자 보호를 대신하지 못함")
    res.add_check(Check("보호 체인 산술 (회귀)", "PASS" if abs(t_chain - sum(v[k] for k, _, _ in SC_STAGES)) < 1e-18 else "FAIL", t_chain, "s", path="단계 합", independent=False, detail=" + ".join(f"{lab} {x * 1e6:.3g} µs" for _, lab, x in stages)))
    # plots: sensitivity of the first-order screens
    ds = np.linspace(0.5e9, 20e9, 60)
    for Lk in (5e-9, 10e-9, 15e-9, 20e-9):
        res.add_series(f"ov_{int(Lk * 1e9)}", f"L = {Lk * 1e9:.0f} nH", "V", (ds / 1e6).tolist(), (Lk * ds).tolist())
    res.add_plot("p_ov", "loop 전압 L·di/dt (1차 크기)", [f"ov_{k}" for k in (5, 10, 15, 20)], x_label="di/dt", x_unit="A/us", y_label="ΔV", y_unit="V", kind="xy", level="A",
                 markers=[{"x": didt / 1e6, "y": dV, "label": f"{dV:.4g} V"}],
                 proved="같은 di/dt에서 loop inductance가 overshoot 크기를 비례로 정한다는 1차 관계를 크기로 익혔다.",
                 not_yet="실제 peak은 di/dt가 일정하지 않고 C_oss와 공진하므로 이 직선과 다르다 — 실험 3의 합성 셀(D)에서 비교한다.")
    dvs = np.linspace(1e9, 100e9, 60)
    for Ck in (0.1e-9, 0.5e-9, 1e-9, 2e-9):
        res.add_series(f"cm_{int(Ck * 1e12)}", f"C = {Ck * 1e9:g} nF", "A", (dvs / 1e6).tolist(), (Ck * dvs).tolist())
    res.add_plot("p_cm", "변위전류 C·dv/dt (edge 순간값)", [f"cm_{k}" for k in (100, 500, 1000, 2000)], x_label="dv/dt", x_unit="V/us", y_label="i", y_unit="A", kind="xy", level="A",
                 markers=[{"x": dvdt / 1e6, "y": icm, "label": f"{icm:.4g} A"}],
                 proved="절연 경계를 건너는 기생 C가 edge 동안 A 단위 전류를 만든다는 크기를 보였다.",
                 not_yet="이 전류의 경로·스펙트럼·수신측(LISN)은 EX09에서 다룬다. EMC 판정이 아니다.")
    res.add_series("gc_v", "v_GS (piecewise 전하 모델)", "V", (gc["Q"]).tolist(), gc["v"].tolist())
    res.add_plot("p_qg", "gate charge 곡선 v_GS(Q_g)", ["gc_v"], x_label="Q_g", x_unit="C", y_label="v_GS", y_unit="V", kind="xy", level="B",
                 hlines=[{"y": v["Vpl"], "label": "Miller plateau"}], markers=[{"x": v["Qgs"], "y": v["Vpl"], "label": "plateau 시작"}, {"x": v["Qgs"] + v["Qgd"], "y": v["Vpl"], "label": "Q_gd 끝"}],
                 proved="plateau 동안 gate 전하는 Q_gd만큼 drain 쪽으로 가고 v_GS는 멈춘다 — 그래서 plateau 시간은 Q_gd/I_g다.",
                 not_yet="전하 곡선은 한 V_DS·I_D 조건의 합성 곡선이다. 전력 루프·common-source 되먹임은 실험 3에서 푼다.")
    res.add_series("gc_vt", "v_GS(t)", "V", gc["t"].tolist(), gc["v"].tolist())
    res.add_series("gc_it", "i_G(t)", "A", gc["t"].tolist(), gc["ig"].tolist())
    gbands = [{"x0": 0.0, "x1": gc["t1"], "mode": "g_charge", "label": "충전"}, {"x0": gc["t1"], "x1": gc["t2"], "mode": "g_miller", "label": "Miller"}, {"x0": gc["t2"], "x1": float(gc["t"][-1]), "mode": "g_final", "label": "최종 충전"}]
    res.add_plot("p_gt", "gate 전압과 전류 (gate-charge ODE)", ["gc_vt", "gc_it"], y_label="v / i", y_unit="", bands=gbands, level="B", group="gate",
                 proved="plateau 동안 gate 전류가 일정(= (V_drv − V_pl)/R)하다는 것을 ODE로 확인했다.",
                 not_yet="gate loop inductance·driver 출력 droop·온도는 없다.")
    # protection timeline (bands) and three layers
    edges, t = [], 0.0
    tl_bands = []
    for key, lab, x in stages:
        tl_bands.append({"x0": t, "x1": t + x, "mode": key, "label": next(sh for k2, _, sh in SC_STAGES if k2 == key)})
        edges.append((t + x, len(edges) + 1))
        t += x
    t_view = max(t_sc, t_chain) * 1.25
    if t_sc > t_chain:
        tl_bands.append({"x0": t_chain, "x1": t_sc, "mode": "t_left", "label": "남는 시간 (마진 아님)"})
    xs, ys = [0.0], [0.0]
    for te, kk in edges:
        xs += [te, te]
        ys += [ys[-1], kk]
    xs.append(t_view)
    ys.append(ys[-1])
    res.add_series("sc_prog", "완료된 보호 단계 수", "", xs, ys)
    res.add_plot("p_sc", "단락 보호 타임라인 (합성 예, 소자 계층)", ["sc_prog"], y_label="완료 단계", y_unit="", bands=tl_bands, level="A", group="sc",
                 vlines=[{"x": t_sc, "label": f"가정한 조건부 SC 생존 {t_sc * 1e6:.3g} µs (실제 정격 아님)"}, {"x": t_chain, "label": f"체인 끝 {t_chain * 1e6:.3g} µs"}],
                 proved="보호 체인의 각 단계 시간과 그 합을 가정한 생존시간과 같은 축에서 비교했다.",
                 not_yet="생존시간 3 µs는 가정이다. 실제 SC 정격(시험전압·온도·gate 조건·보증)과 온도·분산 최악조건이 없으면 남는 시간은 마진이 아니다.")
    layers = [("① 소자 보호 (DESAT·soft-off)", 0.05e-6, t_chain, 1.0), ("② 고장 보고 (드라이버 → 제어기)", t_chain, t_chain + v["t_report"], 2.0), ("③ 시스템 반응 (ASC/FW·안전상태)", t_chain + v["t_report"], t_chain + v["t_report"] + v["t_sys"], 3.0)]
    for j, (lab, a, b, yv) in enumerate(layers):
        res.add_series(f"layer{j}", lab, "", [a, b], [yv, yv])
    res.add_plot("p_layers", "세 계층의 시간 규모 (로그 시간축)", [f"layer{j}" for j in range(3)], x_label="시간", x_unit="s", y_label="계층", y_unit="", kind="xy", log_x=True, level="A",
                 vlines=[{"x": t_sc, "label": "가정 SC 생존"}], hlines=[{"y": 1.0, "label": "① 소자 보호"}, {"y": 2.0, "label": "② 고장 보고"}, {"y": 3.0, "label": "③ 시스템 반응"}],
                 proved="소자 보호는 µs, 시스템 반응은 수십~수백 µs로 시간 규모가 달라 서로를 대신할 수 없다는 것을 보였다.",
                 not_yet="보고·반응 시간은 합성 가정이다. 실제 기능안전 요구·검출 경로는 시스템 사양에서 온다.")
    order = ["p_sc", "p_layers", "p_ov", "p_cm", "p_qg", "p_gt"]
    res.plots.sort(key=lambda pl: order.index(pl.key))
    res.circuit = {"diagram": _protect_circuit(v).to_json(), "intervals": tl_bands, "plot_group": "sc"}
    res.tables.append(Table("t_sc", "보호 체인 (합성 예)", ["단계", "시간 [µs]", "누적 [µs]"], [[lab, x * 1e6, sum(y for _, _, y in stages[: j + 1]) * 1e6] for j, (_, lab, x) in enumerate(stages)] + [["가정한 조건부 SC 생존시간", t_sc * 1e6, ""], ["남는 시간 (확정 마진 아님)", (t_sc - t_chain) * 1e6, ""]],
                            note="이 숫자는 Infineon 부품 정격이 아니며 실제 SC 보증이 없는 부품에 적용하지 않는다."))
    if t_chain >= t_sc:
        res.verdict("FAIL_CONSTRAINT", f"보호 체인 {t_chain * 1e6:.3g} µs가 가정한 생존시간 {t_sc * 1e6:.3g} µs 안에 끝나지 않는다 (마진 검토 이전에 이미 불만족)")
    else:
        res.verdict("SCREEN_ONLY", "1차 크기 screen: L·di/dt, C·dv/dt, Q_gd/I_g, 보호 체인 합")
    res.verdict("MISSING_INPUT", "실제 소자의 SC 생존시간·조건이 없다 — 단순 모델로 SC survival을 보장하지 않는다")
    res.assumptions += ["di/dt·dv/dt 일정 (1차 크기)", "gate-charge 곡선은 합성 piecewise 모델 (C₁ 충전 → Q_gd plateau → C₂ 충전)", "plateau 전류 = (V_drv − V_pl)/(R_drv + R_g + R_g,int); driver droop·gate loop L 없음", "SC 생존시간 3 µs는 교재의 조건부 가정"]
    res.not_valid_for += ["실제 overshoot peak·ringing (실험 3의 합성 셀, EX09)", "실제 SC 생존·보호 인증", "EMC 적합성"]
    res.interpretation = (
        f"같은 di/dt에서 loop 15 nH는 {dV:.4g} V를 만들고, 1 nF 기생 C는 30 kV/µs에서 {icm:.4g} A의 순간 전류를 만든다. Miller 구간은 gate가 Q_gd를 drain 쪽으로 "
        f"옮기는 시간이라 plateau 전류 {Ig:.3g} A로 {tM * 1e9:.3g} ns다. 보호는 소자 계층(µs)과 시스템 계층(수십~수백 µs)이 따로 있고, 체인 합 {t_chain * 1e6:.3g} µs를 "
        "뺀 나머지는 온도·분산·최악조건을 검토하기 전에는 마진이 아니다."
    )
    return res


def _protect_circuit(v: dict) -> Circuit:
    c = Circuit("gate_protect", 700, 270, title="게이트 드라이버와 소자 보호 경로 (개념도)")
    drv = c.add("block", "DRV", 110, 150, 0, "게이트 드라이버", w=110, h=46)
    rg = c.add("resistor", "Rg", 250, 150, 0, "R_g", f"{v['Rg_ext']:g} Ω", lpos=(250, 124, "middle"))
    q = c.add("nmos", "Q", 330, 150, 90, "DUT", lpos=(362, 150, "start"))
    des = c.add("block", "DESAT", 480, 70, 0, "DESAT 검출 (blanking·filter)", w=190, h=40)
    lg = c.add("block", "LOGIC", 480, 150, 0, "로직·전파", w=120, h=36)
    soft = c.add("block", "SOFT", 480, 225, 0, "soft turn-off (gate 방전)", w=170, h=36)
    sysb = c.add("block", "SYS", 640, 150, 0, "제어기 ASC/FW", w=110, h=40)
    c.wire("w_drv_rg", (165, 150), rg["a"])
    c.wire("w_rg_g", rg["b"], q["g"])
    c.wire("w_d_des", q["a"], (330, 70), (385, 70))
    c.wire("w_des_logic", (480, 90), (480, 132))
    c.wire("w_logic_soft", (480, 168), (480, 207))
    c.wire("w_soft_drv", (395, 225), (110, 225), (110, 173))
    c.wire("w_logic_sys", (540, 150), (585, 150))
    c.wire("w_q_src", q["b"], (330, 245))
    c.add("ground", "g", 330, 245)
    c.mode("t_blank", "blanking", ["DESAT", "w_d_des"], "turn-on 직후 오검출을 막기 위해 DESAT 비교를 잠시 무시한다 (이 동안 단락이면 전류가 계속 오른다)", dim=["SOFT", "SYS"])
    c.mode("t_filter", "검출·필터", ["DESAT", "w_d_des", "w_des_logic"], "V_DS 상승을 비교기·필터로 확인", dim=["SOFT", "SYS"])
    c.mode("t_prop", "전파·로직", ["LOGIC", "w_des_logic", "w_logic_soft"], "fault 신호가 드라이버 출력단까지 전달", dim=["SYS"])
    c.mode("t_soft", "gate 방전·전류 소거", ["SOFT", "w_logic_soft", "w_soft_drv", "DRV", "w_drv_rg", "Rg", "w_rg_g", "Q"], "gate를 천천히 내려 L·di/dt overshoot를 제한하며 전류를 끊는다", dim=["SYS"])
    c.mode("t_left", "남는 시간 (가정)", ["SYS", "w_logic_sys"], "이 구간은 확정 마진이 아니다. 시스템 보고·반응은 별도 계층", dim=[])
    return c


# ======================================================================================
# DPT analysis helpers (windows, deskew, referents)
# ======================================================================================


def first_cross(t, y, level, t0, t1, rising=True):
    """First time in [t0, t1] at which y crosses ``level`` (linear interpolation), or None."""
    m = (t >= t0) & (t <= t1)
    tt, yy = t[m], y[m] - level
    if tt.size < 2:
        return None
    k = np.nonzero((yy[:-1] < 0) & (yy[1:] >= 0))[0] if rising else np.nonzero((yy[:-1] > 0) & (yy[1:] <= 0))[0]
    if not k.size:
        return None
    j = int(k[0])
    return float(tt[j] + (tt[j + 1] - tt[j]) * (0.0 - yy[j]) / (yy[j + 1] - yy[j]))


def window_integral(t, y, a, b):
    """Trapezoid integral of a sampled signal over [a, b] (end points interpolated)."""
    inner = (t > a) & (t < b)
    tt = np.concatenate([[a], t[inner], [b]])
    return float(np.trapezoid(np.interp(tt, t, y), tt))


def shifted(t, y, tau):
    """Measured channel delayed by tau: y_m(t) = y(t - tau) (tau < 0: the trace appears earlier)."""
    return np.interp(t - tau, t, y)


def dpt_energy(t, v, i, event, t_cmd, V, I, skew=0.0, window="fixed", pre=5e-9, post=250e-9):
    """E = integral of v_m * i_m over the chosen window; returns (E, (a, b), definition text)."""
    im = shifted(t, i, skew)
    if window == "fixed":
        a, b = t_cmd - pre, t_cmd + post
        txt = f"고정: 명령 {-pre * 1e9:+.3g} ns ~ {post * 1e9:+.3g} ns"
    else:
        lo, hi = t_cmd - pre, t_cmd + post
        if event == "on":
            a = first_cross(t, im, 0.1 * I, lo, hi, rising=True)
            b = first_cross(t, v, 0.02 * V, a if a is not None else lo, hi, rising=False)
            txt = "문턱: i_m ≥ 10 % I_L → v ≤ 2 % V_bus"
        else:
            a = first_cross(t, v, 0.1 * V, lo, hi, rising=True)
            b = first_cross(t, im, 0.02 * I, a if a is not None else lo, hi, rising=False)
            txt = "문턱: v ≥ 10 % V_bus → i_m ≤ 2 % I_L (첫 교차)"
        if a is None or b is None or b <= a:
            return float("nan"), (float("nan"), float("nan")), txt + " (교차 없음)"
    return window_integral(t, v * im, a, b), (a, b), txt


# ======================================================================================
# Experiments 3-4: synthetic DPT cell (level D)
# ======================================================================================

T_OFF = 20e-9  # turn-off command of the first pulse (s)


def _dut_from(v: dict) -> DeviceParams:
    base = DeviceParams()
    V0 = base.Vgd0
    Cgd0 = v["Qgd"] / (2.0 * V0 * (math.sqrt(1.0 + 800.0 / V0) - 1.0))
    return replace(base, Vth=v["Vth"], gm=v["gm"], Ron=v["Ron"], Cgs=v["Cgs"], Cgd0=Cgd0, Von_ref=v["Von"])


def _dpt_spec(v: dict, dev: DeviceParams, hs_over: dict | None = None) -> CellSpec:
    topo = "individual_kelvin" if v["drv_ref"] == "kelvin" else "common_source_node"
    hs = dev
    RgH = v.get("RgH", 3.5)
    if v.get("clamp", False):
        RgH = 1.0 / (1.0 / RgH + 1.0 / v.get("R_clamp", 0.5))
    kw = dict(dut=[dev], hs=hs, Vbus=v["Vbus"], IL=v["IL"], La=v["La"], Lb=v["Lb"], Rp=v["Rp"], Rloop=2e-3, LsH=v["LsH"], Ld=[[0.0]], Ls=[v["Ls"]], Lcs=0.0,
              Lg=[v["Lg"]], Rg_on=[v["Rg_on"]], Rg_off=[v["Rg_off"]], Von=v["Von"], Voff=v["Voff"], topology=topo, LgH=v["Lg"], RgH=RgH, VoffH=v.get("VoffH", v["Voff"]), t_slew=2e-9)
    kw.update(hs_over or {})
    return CellSpec(**kw)


def _dpt_params(extra: list | None = None, rg: float = 10.0) -> list[Param]:
    ps = [
        Param("Vbus", "DC-link 전압 V_bus", "V", 800.0, "V", vmin=50, vmax=1500, source="TEXTBOOK", source_note="E03 합성 DPT와 같은 800 V", group="회로"),
        Param("IL", "DPT 부하전류 I_L (사건 동안 일정)", "A", 100.0, "A", vmin=1, vmax=400, source="TEXTBOOK", source_note="E03 합성 DPT와 같은 100 A", group="회로"),
        Param("La", "외부 loop inductance L_a (DC-link·busbar)", "H", 8e-9, "nH", vmin=1e-9, vmax=100e-9, source="ASSUMED", source_note="L_a + L_b + L_s + L_sH = 15 nH (교재 L_loop)", group="회로"),
        Param("Lb", "손실성 loop 부분 L_b (R_p와 병렬)", "H", 3e-9, "nH", vmin=0.1e-9, vmax=20e-9, source="ASSUMED", source_note="skin·근접효과 감쇠의 합성 등가", group="회로"),
        Param("Rp", "감쇠 저항 R_p (L_b와 병렬)", "Ω", 3.0, "Ω", vmin=0.05, vmax=100, source="ASSUMED", source_note="DC 손실 없이 링잉만 감쇠", group="회로"),
        Param("Ls", "DUT common-source inductance L_s", "H", 2e-9, "nH", vmin=0.05e-9, vmax=20e-9, source="ASSUMED", source_note="E03 공통 source 2 nH와 같은 크기", group="회로"),
        Param("LsH", "상측 소자 source inductance L_sH", "H", 2e-9, "nH", vmin=0.05e-9, vmax=20e-9, source="ASSUMED", group="회로"),
        Param("drv_ref", "DUT 드라이버 반환", "", "kelvin", kind="choice", choices=[("kelvin", "Kelvin source"), ("power_source", "power source 핀")], source="ASSUMED", source_note="power source 핀이면 L_s가 게이트 루프 안에 들어간다", group="게이트"),
        Param("Von", "gate on 전압", "V", 18.0, "V", vmin=5, vmax=25, source="ASSUMED", group="게이트"),
        Param("Voff", "gate off 전압", "V", -4.0, "V", vmin=-10, vmax=0, source="ASSUMED", group="게이트"),
        Param("Rg_on", "외부 R_g (turn-on 경로)", "Ω", rg, "Ω", vmin=0.1, vmax=100, source="ASSUMED", group="게이트"),
        Param("Rg_off", "외부 R_g (turn-off 경로)", "Ω", rg, "Ω", vmin=0.1, vmax=100, source="ASSUMED", group="게이트"),
        Param("Lg", "gate loop inductance L_g", "H", 10e-9, "nH", vmin=1e-9, vmax=100e-9, source="ASSUMED", group="게이트"),
        Param("Vth", "V_th", "V", 4.0, "V", vmin=1, vmax=8, source="ASSUMED", source_note="합성 SiC형 소자의 문턱전압, 고정 T_j", group="소자"),
        Param("gm", "g_m", "A/V", 20.0, "A/V", vmin=1, vmax=500, source="ASSUMED", group="소자"),
        Param("Ron", "R_DS(on) @ V_on", "Ω", 20e-3, "mΩ", vmin=1e-3, vmax=1.0, source="ASSUMED", group="소자"),
        Param("Cgs", "C_gs (일정)", "F", 4e-9, "nF", vmin=0.1e-9, vmax=100e-9, source="ASSUMED", group="소자"),
        Param("Qgd", "Q_gd (0→800 V)", "C", 20e-9, "nC", vmin=1e-9, vmax=500e-9, source="TEXTBOOK", source_note="교재 Q_gd 20 nC와 같은 크기 (합성 C_gd(v)로 환산)", group="소자"),
    ]
    return ps + (extra or [])


def _dpt_bands(t, vds, i, t_off, t_on, V, I):
    """Sequential state intervals of the DPT run (turn-off then turn-on) for plot bands and circuit modes."""
    b = []

    def add(x0, x1, mode, label):
        if x0 is not None and x1 is not None and x1 > x0:
            b.append({"x0": float(x0), "x1": float(x1), "mode": mode, "label": label})

    v10 = first_cross(t, vds, 0.1 * V, t_off, t_off + 300e-9, True)
    v90 = first_cross(t, vds, 0.9 * V, t_off, t_off + 300e-9, True)
    i10 = first_cross(t, i, 0.1 * I, v90 or t_off, t_off + 300e-9, False)
    add(0.5 * t_off, t_off, "on1", "ON")
    add(t_off, v10, "toff_delay", "지연")
    add(v10, v90, "toff_v", "v↑")
    add(v90, i10, "toff_i", "i↓")
    end_ring = (i10 or t_off) + 150e-9
    add(i10, end_ring, "ring_off", "링잉")
    add(end_ring, t_on, "freewheel", "환류")
    i10n = first_cross(t, i, 0.1 * I, t_on, t_on + 300e-9, True)
    v90n = first_cross(t, vds, 0.9 * V, i10n or t_on, t_on + 300e-9, False)
    v10n = first_cross(t, vds, 0.1 * V, v90n or t_on, t_on + 300e-9, False)
    add(t_on, i10n, "ton_delay", "지연")
    add(i10n, v90n, "ton_i", "i↑")
    add(v90n, v10n, "ton_v", "v↓")
    add(v10n, float(t[-1]), "on2", "ON (2nd)")
    return b, {"v10": v10, "v90": v90, "i10": i10, "i10n": i10n, "v90n": v90n, "v10n": v10n}


def _dpt_circuit(v: dict, focus: str = "dpt") -> Circuit:
    c = Circuit("dpt_cell", 720, 400, title="합성 DPT 셀: 하측 DUT, 상측 소자(OFF), 부하 인덕터")
    vb = c.add("vsource", "Vbus", 60, 205, 90, "V_bus", f"{v['Vbus']:g} V", lpos=(40, 250, "end"))
    ll = c.add("inductor", "Lloop", 180, 40, 0, "L_loop = L_a + (L_b‖R_p)", lpos=(180, 24, "middle"))
    hs = c.add("nmos", "HS", 330, 90, 90, "상측 (OFF)", lpos=(356, 84, "start"))
    lsh = c.add("inductor", "LsH", 330, 160, 90, "L_sH", f"{v['LsH'] * 1e9:g} nH", lpos=(344, 164, "start"))
    lo = c.add("inductor", "Lload", 470, 125, 90, "L_load", f"I_L = {v['IL']:g} A", lpos=(486, 118, "start"))
    dut = c.add("nmos", "DUT", 330, 255, 90, "DUT (하측)", lpos=(356, 249, "start"))
    ls = c.add("inductor", "Ls", 330, 320, 90, "L_s", f"{v['Ls'] * 1e9:g} nH", lpos=(344, 324, "start"))
    rg = c.add("resistor", "Rg", 238, 255, 0, "R_g", f"{v['Rg_on']:g} Ω", lpos=(238, 229, "middle"))
    drv = c.add("block", "DRV", 130, 255, 0, "드라이버", w=86, h=36)
    rgh = c.add("resistor", "RgH", 238, 90, 0, "R_gH", "", lpos=(238, 74, "middle"))
    drvh = c.add("block", "DRVH", 130, 90, 0, "V_off,H", w=86, h=34)
    c.add("ground", "gnd", 330, 380)
    c.wire("w_top_l", vb["a"], (60, 40), ll["a"])
    c.wire("w_top_r", ll["b"], (330, 40), hs["a"])
    c.wire("w_load_top", (330, 40), (470, 40), lo["a"])
    c.wire("w_hs_lsh", hs["b"], lsh["a"])
    c.wire("w_mid", lsh["b"], (330, 200), dut["a"])
    c.wire("w_load_bot", lo["b"], (470, 200), (330, 200))
    c.wire("w_ls", dut["b"], ls["a"])
    c.wire("w_bot", ls["b"], (330, 370), (60, 370), vb["b"])
    c.wire("w_drv_rg", (173, 255), rg["a"])
    c.wire("w_rg_g", rg["b"], dut["g"])
    c.wire("w_drvh_rg", (173, 90), rgh["a"])
    c.wire("w_rgh_g", rgh["b"], hs["g"])
    kel_y = 292 if v.get("drv_ref", "kelvin") == "kelvin" else 356
    src_pt = (330, 285) if v.get("drv_ref", "kelvin") == "kelvin" else (330, 356)
    c.wire("w_kelvin", src_pt, (130, src_pt[1]), (130, 273))
    c.wire("w_kelvinH", (330, 124), (130, 124), (130, 107))
    c.dot((330, 40), (330, 200), src_pt, (330, 124))
    c.text(346, 196, "M", "node")
    c.text(346, 36, "P", "node")
    c.text(180, kel_y + (10 if kel_y < 300 else -6), "Kelvin 반환" if kel_y < 300 else "power-source 반환 (L_s가 게이트 루프 안)", "")
    c.probe("pD", "id_all", 318, 212, "down", "i_D")
    c.probe("pL", "IL_all", 486, 160, "down", "I_L")
    c.probe("pH", "iH_all", 318, 145, "up", "−i_HS")
    power_on = ["Vbus", "w_top_l", "Lloop", "w_top_r", "w_load_top", "Lload", "w_load_bot", "w_mid", "DUT", "w_ls", "Ls", "w_bot"]
    fw = ["Lload", "w_load_top", "w_load_bot", "w_mid", "LsH", "w_hs_lsh", "HS", "w_top_r"]
    gate = ["DRV", "w_drv_rg", "Rg", "w_rg_g", "w_kelvin"]
    c.mode("on1", "DUT ON", power_on + gate, "V_bus → L_loop → L_load → DUT → L_s: 부하전류가 DUT 채널로 흐른다", dim=["HS"])
    c.mode("toff_delay", "turn-off 지연", power_on + gate, "gate가 V_off로 방전되는 동안 전류·전압은 아직 그대로", dim=["HS"])
    c.mode("toff_v", "v_DS 상승 (Miller)", power_on + gate + ["HS"], "gate 전류가 C_gd를 통해 drain 전압을 올린다; HS의 C_oss가 방전된다", dim=[])
    c.mode("toff_i", "i_D 하강", power_on + fw + gate, "HS diode가 전류를 넘겨받으며 L_loop·di/dt만큼 v_DS가 V_bus를 넘는다", dim=[])
    c.mode("ring_off", "링잉", fw + ["Lloop", "w_top_l", "Vbus", "w_bot", "Ls", "w_ls", "DUT"], "L_loop와 DUT C_oss가 공진 (이상 스위치 링잉이 아니라 회로 L·C가 만든 것)", dim=[])
    c.mode("freewheel", "HS diode 환류", fw, "부하전류가 L_load ↔ HS body diode로 돈다; DUT는 V_bus를 막는다", dim=["DUT"])
    c.mode("ton_delay", "turn-on 지연", fw + gate, "gate가 V_th까지 충전되는 동안", dim=["DUT"])
    c.mode("ton_i", "i_D 상승", power_on + fw + gate, "DUT가 전류를 넘겨받고, HS C_oss 충전 전류가 더해져 i_D가 I_L을 넘는다", dim=[])
    c.mode("ton_v", "v_DS 하강 (Miller)", power_on + gate + ["HS"], "HS C_oss가 충전되고 DUT C_oss는 자기 채널로 방전된다", dim=[])
    c.mode("on2", "DUT ON (2nd pulse)", power_on + gate, "두 번째 펄스: DUT가 다시 부하전류를 흘린다", dim=["HS"])
    del drv, drvh, focus
    return c


def _simulate_dpt(v: dict, rtol: float = 1e-6, t_gap: float = 1e-6, post: float = 300e-9):
    dev = _dut_from(v)
    spec = _dpt_spec(v, dev)
    t_on = T_OFF + t_gap
    cell = Cell(spec, [(T_OFF, "off"), (t_on, "on")], start="on")
    fine = [(T_OFF - 10e-9, T_OFF + post, 12000), (t_on - 10e-9, t_on + post, 12000)]
    run = cell.simulate(t_on + post + 20e-9, n_samples=800, rtol=rtol, t_fine=fine)
    return cell, run, t_on, dev


def run_dpt_cell(v: dict) -> Result:
    res = Result("FL02", "dpt_cell", "D (합성 commutation cell)")
    V, I = v["Vbus"], v["IL"]
    cell, r, t_on, dev = _simulate_dpt(v)
    t = r.t
    vds = r.x[cell.i_vds][0]
    i = r.x[cell.i_i][0]
    vgs = r.x[cell.i_vgs][0]
    o = r.out
    vdsPS = o["vdsPS"][0]
    ich = o["ich"][0]
    pre, post = v["win_pre"], v["win_post"]
    skew = v["skew"]
    # reference definitions (Kelvin v, device current, zero skew)
    E = {}
    for ev, tc in (("on", t_on), ("off", T_OFF)):
        for w in ("fixed", "threshold"):
            E[(ev, w)] = dpt_energy(t, vds, i, ev, tc, V, I, 0.0, w, pre, post)
    # the user's stored settings
    vsig = vds if v["v_ref"] == "kelvin" else vdsPS
    isig = {"device": i, "channel": ich, "inductor": np.full_like(i, I)}[v["i_ref"]]
    Eu = {ev: dpt_energy(t, vsig, isig, ev, tc, V, I, skew, v["window"], pre, post) for ev, tc in (("on", t_on), ("off", T_OFF))}
    # channel (die) energies inside the fixed windows from the solver's quadrature
    qi = cell.qnames.index("E_ch0")
    qt = cell.qnames.index("E_term0")
    qg = cell.qnames.index("Qg0")
    Ech = {}
    Eterm_q = {}
    for ev, tc in (("on", t_on), ("off", T_OFF)):
        a, b = tc - pre, tc + post
        Ech[ev] = float(np.interp(b, t, r.q[qi]) - np.interp(a, t, r.q[qi]))
        Eterm_q[ev] = float(np.interp(b, t, r.q[qt]) - np.interp(a, t, r.q[qt]))
    Eon, Eoff = E[("on", "fixed")][0], E[("off", "fixed")][0]
    res.add_metric("Eon_fix", "E_on (Kelvin v_DS × 소자 단자 i_D, 고정창)", Eon, "J", basis=E[("on", "fixed")][2])
    res.add_metric("Eoff_fix", "E_off (Kelvin v_DS × 소자 단자 i_D, 고정창)", Eoff, "J", basis=E[("off", "fixed")][2])
    res.add_metric("Eon_thr", "E_on (문턱창)", E[("on", "threshold")][0], "J", basis=E[("on", "threshold")][2])
    res.add_metric("Eoff_thr", "E_off (문턱창)", E[("off", "threshold")][0], "J", basis=E[("off", "threshold")][2])
    res.add_metric("Eon_ch", "E_on 채널 소산 (die, 모델 전용)", Ech["on"], "J", basis="같은 고정창, ∫v_DS·i_ch — 측정 불가 양")
    res.add_metric("Eoff_ch", "E_off 채널 소산 (die, 모델 전용)", Ech["off"], "J", basis="같은 고정창")
    res.add_metric("Eon_user", "E_on (저장된 설정)", Eu["on"][0], "J", basis=f"v: {v['v_ref']}, i: {v['i_ref']}, skew {skew * 1e9:+.3g} ns, {Eu['on'][2]}")
    res.add_metric("Eoff_user", "E_off (저장된 설정)", Eu["off"][0], "J", basis=f"v: {v['v_ref']}, i: {v['i_ref']}, skew {skew * 1e9:+.3g} ns, {Eu['off'][2]}")
    if abs(Eon) > 0:
        res.add_metric("Eon_err", "저장된 설정의 E_on − 기준 정의", (Eu["on"][0] - Eon) / Eon, "", basis="기준: Kelvin v · 단자 i · skew 0 · 고정창")
    # Coss energy boundary
    Dv = cell.D
    Eoss = float(Dv.E_ds(np.array([V]))[0] + Dv.E_gd(np.array([V]))[0])
    res.add_metric("Eoss", "DUT E_oss(V_bus) = ∫v·(C_ds + C_gd)dv (합성 C(v))", Eoss, "J", basis="gate-source 단락 조건의 저장에너지")
    tot_term = Eon + Eoff
    tot_ch = Ech["on"] + Ech["off"]
    res.add_metric("E_sum_term", "E_on + E_off (단자 적분)", tot_term, "J", basis="한 주기의 두 사건")
    res.add_metric("E_sum_ch", "E_on + E_off (채널 소산)", tot_ch, "J", note="합은 거의 같다: C_oss 에너지는 turn-off 단자 적분에 들어가고 turn-on 때 채널에서 소산된다")
    # waveform metrics
    m_off = (t > T_OFF) & (t < T_OFF + post)
    m_on = (t > t_on) & (t < t_on + post)
    didt = np.gradient(i, t)
    dvdt = np.gradient(vds, t)
    vpk = float(vds[m_off].max())
    didt_off = float(-didt[m_off].min())
    L_tot = cell.s.La + cell.s.Lb + cell.s.Ls[0] + cell.s.LsH
    res.add_metric("vpk", "turn-off v_DS peak (Kelvin)", vpk, "V", basis="die 전압 = Kelvin 단자")
    res.add_metric("dv_over", "overshoot v_DS,peak − V_bus", vpk - V, "V", ref=L_tot * didt_off, ref_label="screen L_loop·|di/dt|max", tol=None, note="1차 screen과의 비교는 아래 검증표")
    res.add_metric("didt_off", "turn-off 최대 |di/dt|", didt_off / 1e6, "A/us")
    res.add_metric("dvdt_off", "turn-off 최대 dv/dt", float(dvdt[m_off].max()) / 1e6, "V/us")
    ipk = float(i[m_on].max())
    res.add_metric("ipk_on", "turn-on i_D peak (단자)", ipk, "A", note=f"I_L보다 {ipk - I:.3g} A 큼: HS C_oss 충전 + DUT C_oss 방전 전류 (역회복 전하는 모델에 없음)")
    res.add_metric("didt_on", "turn-on 최대 di/dt", float(didt[m_on].max()) / 1e6, "A/us")
    res.add_metric("dvdt_on", "turn-on 최대 |dv/dt|", float(-dvdt[m_on].min()) / 1e6, "V/us")
    bands, bt = _dpt_bands(t, vds, i, T_OFF, t_on, V, I)
    # Miller plateau at turn-on vs the first-order screen, and the gate-charge identity
    checks_ok = True
    if bt["v90n"] and bt["v10n"]:
        ta, tb_ = bt["v90n"], bt["v10n"]
        vpl = float(np.interp(0.5 * (ta + tb_), t, vgs))
        Rg_tot = v["Rg_on"] + dev.Rgint
        Ig = (v["Von"] - vpl) / Rg_tot
        vgs_a, vgs_b = float(np.interp(ta, t, vgs)), float(np.interp(tb_, t, vgs))
        vds_a, vds_b = float(np.interp(ta, t, vds)), float(np.interp(tb_, t, vds))
        Qswing = float(Dv.Q_gd(np.array([vds_a - vgs_a]))[0] - Dv.Q_gd(np.array([vds_b - vgs_b]))[0])
        res.add_metric("t_miller", "turn-on v_DS 90→10 % 시간 (Miller 구간)", tb_ - ta, "s", ref=Qswing / Ig, ref_label="screen ΔQ_gd/I_g", tol=None, basis=f"plateau ≈ {vpl:.3g} V, I_g ≈ {Ig:.3g} A")
        dQ_ode = float(np.interp(tb_, t, r.q[qg]) - np.interp(ta, t, r.q[qg]))
        dQ_cf = float(Dv.Q_gate(np.array([vgs_b]), np.array([vds_b]))[0] - Dv.Q_gate(np.array([vgs_a]), np.array([vds_a]))[0])
        res.add_check(check_close("gate 전하 보존: ∫i_G dt vs Q_gate(v) 닫힌 식", dQ_ode, dQ_cf, 1e-4, "solver가 적분한 gate 전류 vs C_gs·v_gs − Q_gd(v_dg)의 해석 적분식 (Miller 구간)", True, "C"))
    # ringing frequency vs linearised LC estimate
    if bt["i10"]:
        tr0 = bt["i10"] + 5e-9
        mm = (t > tr0) & (t < tr0 + 200e-9)
        yy = vds[mm] - V
        tt = t[mm]
        zc = [tt[k] + (tt[k + 1] - tt[k]) * (-yy[k]) / (yy[k + 1] - yy[k]) for k in range(yy.size - 1) if yy[k] * yy[k + 1] < 0]
        if len(zc) >= 5:
            f_ring = (len(zc) - 1) / (2 * (zc[-1] - zc[0]))
            Coss = float(Dv.Cds(np.array([V]))[0] + Dv.Cgd(np.array([V - v["Voff"]]))[0])
            s_ = cell.s
            Lrest = s_.La + s_.Ls[0] + s_.LsH
            w = 2 * math.pi * f_ring
            Leff = Lrest + s_.Lb * s_.Rp**2 / (s_.Rp**2 + (w * s_.Lb) ** 2)
            f_est = 1.0 / (2 * math.pi * math.sqrt(Leff * Coss))
            res.add_metric("f_ring", "turn-off 링잉 주파수 (파형 영점 교차)", f_ring, "Hz", ref=f_est, ref_label="소신호 1/(2π√(L_eff·C_oss(V_bus)))", tol=0.15, basis="HS diode 도통 중 L_loop–C_oss,DUT 공진")
    # stored settings table and energy comparison table
    rows = []
    defs = [("Kelvin v_DS", "소자 단자 i_D", "fixed", vds, i), ("Kelvin v_DS", "소자 단자 i_D", "threshold", vds, i), ("power-source v_DS (L_s 포함)", "소자 단자 i_D", "fixed", vdsPS, i),
            ("Kelvin v_DS", "채널 i_ch (모델 전용)", "fixed", vds, ich), ("Kelvin v_DS", "부하 인덕터 I_L (잘못된 대용)", "fixed", vds, np.full_like(i, I))]
    for vl, il, w, vv, ii in defs:
        e_on = dpt_energy(t, vv, ii, "on", t_on, V, I, 0.0, w, pre, post)[0]
        e_off = dpt_energy(t, vv, ii, "off", T_OFF, V, I, 0.0, w, pre, post)[0]
        rows.append([vl, il, "고정" if w == "fixed" else "문턱", e_on * 1e6, e_off * 1e6, (e_on - Eon) / Eon * 100 if Eon else float("nan")])
    res.tables.append(Table("t_ref", "같은 파형, 다른 참조·창 (E는 정의의 결과다)", ["v 참조", "i 참조", "창", "E_on [µJ]", "E_off [µJ]", "E_on 차이 [%]"], rows,
                            note="부하 인덕터 전류를 소자 전류로 쓰면 C_oss 충방전 전류를 놓친다. power-source 기준 v_DS에는 L_s·di/dt가 섞인다. 채널 에너지는 측정할 수 없는 모델 전용 양이다."))
    res.tables.append(Table("t_settings", "DPT 적분 설정 (결과와 함께 저장)", ["항목", "값"], [
        ["v 참조 (referent)", {"kelvin": "Kelvin source 기준 v_DS (die 전압과 같음 — 내부 drain L 없음 가정)", "power_source": "power source 핀 기준 (L_s·di/dt 포함)"}[v["v_ref"]]],
        ["i 참조 (referent)", {"device": "소자 단자 drain 전류 (C_oss 전류 포함)", "channel": "채널 전류 (모델 전용, 측정 불가)", "inductor": "부하 인덕터 전류 I_L (C_oss 전류 누락)"}[v["i_ref"]]],
        ["deskew (전류 채널 이동)", f"{skew * 1e9:+.3g} ns (−: 전류가 앞당겨짐)"],
        ["적분창", Eu["on"][2] + f" / turn-off: {Eu['off'][2]}"],
        ["고정창 경계", f"명령 −{pre * 1e9:.3g} ns ~ +{post * 1e9:.3g} ns (turn-on·off 동일, gate 설정마다 바꾸지 않음)"],
        ["Coss 포함 경계", "단자 적분 E_off에는 DUT C_oss 충전 에너지가 들어가고, turn-on 단자 적분에는 DUT C_oss 방전 에너지가 빠진다 (채널에서 소산)"],
        ["역회복", "모델에 없음 (unipolar body diode 가정) — HS C_oss 충전 전하만 turn-on 전류에 포함"],
        ["대역폭·표본", "이상 측정 (대역 제한 없음), 표본 간격 0.026 ns (EX03에서 대역폭·gain·offset 민감도)"],
    ]))
    # terminal energy decomposition (turn-off, fixed window)
    a_, b_ = T_OFF - pre, T_OFF + post
    xa = np.array([np.interp(a_, t, r.x[k]) for k in range(cell.n_core)])
    xb = np.array([np.interp(b_, t, r.x[k]) for k in range(cell.n_core)])
    dEc_off = float(Dv.E_cap(xb[cell.i_vgs], xb[cell.i_vds])[0] - Dv.E_cap(xa[cell.i_vgs], xa[cell.i_vds])[0])
    xa2 = np.array([np.interp(t_on - pre, t, r.x[k]) for k in range(cell.n_core)])
    xb2 = np.array([np.interp(t_on + post, t, r.x[k]) for k in range(cell.n_core)])
    dEc_on = float(Dv.E_cap(xb2[cell.i_vgs], xb2[cell.i_vds])[0] - Dv.E_cap(xa2[cell.i_vgs], xa2[cell.i_vds])[0])
    res.tables.append(Table("t_decomp", "단자 적분 = 채널 소산 + DUT 용량 저장 변화 + gate 측 교환 (고정창)", ["사건", "단자 ∫v·i [µJ]", "채널 소산 [µJ]", "DUT 용량 저장 변화 [µJ]", "나머지 (gate 교환 등) [µJ]"], [
        ["turn-off", Eoff * 1e6, Ech["off"] * 1e6, dEc_off * 1e6, (Eoff - Ech["off"] - dEc_off) * 1e6],
        ["turn-on", Eon * 1e6, Ech["on"] * 1e6, dEc_on * 1e6, (Eon - Ech["on"] - dEc_on) * 1e6],
    ], note="용량 저장 변화는 C_gs·C_gd·C_ds의 해석 에너지식으로 계산했다. turn-off 단자 적분의 일부는 소산이 아니라 저장이며, 그 에너지는 turn-on 때 채널에서 소산된다 — 둘을 따로 더하면 중복이다."))
    # checks: energy ledger, tolerance, sampled vs quadrature
    led = r.energy
    res.add_check(Check("에너지 잔차 (source + drivers − load − 소산 − ΔW)", "PASS" if abs(led["normalised"]) < 1e-5 else "FAIL", led["normalised"], "rel", 1e-5, path="V_bus·∫i, 드라이버 ∫v·i_G, 부하 ∫v_PM·I_L, 저항·채널·diode 소산, ½Li² + 비선형 C 에너지식 — 상태식과 독립적으로 정의한 항", independent=True,
                        detail=f"E_src {led['E_src'] * 1e6:.5g} µJ, E_drv {led['E_drv'] * 1e6:.4g} µJ, E_load {led['E_load'] * 1e6:.5g} µJ, 소산 {led['E_diss'] * 1e6:.5g} µJ, ΔW {led['dW'] * 1e6:.3g} µJ → 잔차 {led['residual']:.3g} J"))
    res.add_check(check_close("표본 적분 vs solver 적분 (turn-on 단자 에너지)", Eon, Eterm_q["on"], 1e-4, "dense output 표본의 사다리꼴 적분 vs 적분상태 ∫v_DS·i_D", True, "J"))
    _, r2, _, _ = _simulate_dpt(v, rtol=1e-8)
    i2, vds2 = r2.x[cell.i_i][0], r2.x[cell.i_vds][0]
    Eon2 = dpt_energy(r2.t, vds2, i2, "on", t_on, V, I, 0.0, "fixed", pre, post)[0]
    Eoff2 = dpt_energy(r2.t, vds2, i2, "off", T_OFF, V, I, 0.0, "fixed", pre, post)[0]
    m2 = (r2.t > T_OFF) & (r2.t < T_OFF + post)
    dev_tol = max(abs(Eon2 - Eon) / abs(Eon), abs(Eoff2 - Eoff) / abs(Eoff), abs(float(vds2[m2].max()) - vpk) / vpk)
    res.add_check(Check("허용오차 강화 수렴 (rtol 1e-6 → 1e-8)", "PASS" if dev_tol < 1e-3 else "FAIL", dev_tol, "rel", 1e-3, path="같은 셀을 더 엄격한 허용오차로 다시 적분: E_on·E_off·v_DS peak 변화", independent=True, detail=f"E_on {Eon2 * 1e6:.6g} µJ, E_off {Eoff2 * 1e6:.6g} µJ"))
    # plots
    def ser(key, lab, unit, y, a, b):
        m = (t >= a) & (t <= b)
        res.add_series(key, lab, unit, t[m].tolist(), np.asarray(y)[m].tolist())
    for ev, tc in (("off", T_OFF), ("on", t_on)):
        a, b = tc - 10e-9, tc + 190e-9
        ser(f"vds_{ev}", "v_DS DUT (Kelvin)", "V", vds, a, b)
        ser(f"vdsH_{ev}", "v_DS 상측", "V", r.x[cell.i_vdsH], a, b)
        ser(f"id_{ev}", "i_D DUT 단자", "A", i, a, b)
        ser(f"ich_{ev}", "i_ch DUT 채널 (모델 전용)", "A", ich, a, b)
        ser(f"il_{ev}", "I_L 부하 (일정)", "A", np.full_like(i, I), a, b)
        ser(f"vgs_{ev}", "v_GS die (내부)", "V", vgs, a, b)
        ser(f"vgsK_{ev}", "v_GS Kelvin 측정", "V", o["vgsK"][0], a, b)
        ser(f"vgsPS_{ev}", "v_GS power-source 측정", "V", o["vgsPS"][0], a, b)
        ser(f"p_{ev}", "순간전력 v_DS·i_D", "W", vds * i, a, b)
    # probe series over both windows (circuit arrows)
    mprobe = ((t >= T_OFF - 10e-9) & (t <= T_OFF + 190e-9)) | ((t >= t_on - 10e-9) & (t <= t_on + 190e-9))
    res.add_series("id_all", "i_D", "A", t[mprobe].tolist(), i[mprobe].tolist())
    res.add_series("IL_all", "I_L", "A", t[mprobe].tolist(), [I] * int(mprobe.sum()))
    res.add_series("iH_all", "−i_HS", "A", t[mprobe].tolist(), (-(o["idc"] - I))[mprobe].tolist())
    b_off = [bb for bb in bands if bb["x1"] <= t_on - 1e-9 and bb["x0"] >= T_OFF - 20e-9]
    b_on = [bb for bb in bands if bb["x0"] >= t_on - 1e-9]
    hl = [{"y": V, "label": "V_bus"}]
    res.add_plot("p_off_v", "turn-off: v_DS와 overshoot", ["vds_off", "vdsH_off"], y_label="v", y_unit="V", bands=b_off, group="off", level="D", hlines=hl,
                 proved=f"v_DS가 Miller 구간에 오르고, 전류가 HS diode로 넘어가는 동안 L_loop·di/dt만큼 V_bus를 넘어 {vpk:.4g} V까지 간 뒤 L–C_oss 링잉으로 감쇠한다.",
                 not_yet="링잉 감쇠는 합성 R_p‖L_b 등가가 정한다. 실제 감쇠(skin·C_oss 손실)와 EMI는 계산하지 않았다.")
    res.add_plot("p_off_i", "turn-off: 단자 전류 vs 채널 전류", ["id_off", "ich_off", "il_off"], y_label="i", y_unit="A", bands=b_off, group="off", level="D",
                 proved="단자 전류와 채널 전류가 다르다: v_DS 상승 동안 부하전류의 일부가 채널이 아니라 C_oss 충전으로 흐른다.",
                 not_yet="채널 전류는 모델에서만 보인다. 측정으로는 단자 전류만 얻는다.")
    res.add_plot("p_off_g", "turn-off: v_GS die / Kelvin / power-source", ["vgs_off", "vgsK_off", "vgsPS_off"], y_label="v_GS", y_unit="V", bands=b_off, group="off", level="D",
                 proved="같은 gate라도 측정 기준점에 따라 파형이 다르다: power-source 기준에는 L_s·di/dt가 더해진다.",
                 not_yet="gate 산화막 스트레스 판정은 die 전압과 소자 정격이 필요하다 (합성 소자).")
    res.add_plot("p_on_v", "turn-on: v_DS 하강 (Miller)", ["vds_on", "vdsH_on"], y_label="v", y_unit="V", bands=b_on, group="on", level="D", hlines=hl,
                 proved="전류가 I_L에 도달한 뒤 v_DS가 Miller 구간 동안 떨어지고, HS의 v_DS는 V_bus로 충전된다.",
                 not_yet="역회복 전하(bipolar diode)는 모델에 없다 — SiC body diode의 역회복을 0으로 둔 경향 결과다.")
    res.add_plot("p_on_i", "turn-on: i_D peak와 C_oss 전류", ["id_on", "ich_on", "il_on"], y_label="i", y_unit="A", bands=b_on, group="on", level="D",
                 proved=f"i_D가 I_L보다 {ipk - I:.3g} A 더 흐른다: HS C_oss 충전 전류가 DUT로 들어오고, DUT 채널은 자기 C_oss 방전 전류까지 흘린다.",
                 not_yet="이 초과 전류는 이 합성 C(v)의 결과다. 실제 소자의 Q_rr·Q_c 데이터가 필요하다.")
    res.add_plot("p_on_g", "turn-on: v_GS die / Kelvin / power-source", ["vgs_on", "vgsK_on", "vgsPS_on"], y_label="v_GS", y_unit="V", bands=b_on, group="on", level="D",
                 proved="Miller plateau(die)와 측정 기준에 따른 차이(L_s·di/dt)를 같은 사건에서 보였다.", not_yet="probe 대역·CM 오차는 없다 (EX03).")
    Wa, Wb = Eu["on"][1]
    res.add_plot("p_on_p", "turn-on 순간전력과 적분창", ["p_on"], y_label="p", y_unit="W", bands=b_on, group="on", level="D", window=(Wa, Wb) if Wa == Wa else None,
                 proved="E_on은 순간전력의 적분이며 창의 시작·끝 정의에 따라 값이 바뀐다 (표 참조).", not_yet="측정 오차(skew·gain·offset·대역)의 영향은 아래 skew sweep과 EX03에서 본다.")
    skews = np.linspace(-5e-9, 5e-9, 21)
    for w, lab in (("fixed", "고정창"), ("threshold", "문턱창")):
        res.add_series(f"sk_on_{w}", f"E_on ({lab})", "J", skews.tolist(), [dpt_energy(t, vds, i, "on", t_on, V, I, float(sk), w, pre, post)[0] for sk in skews])
        res.add_series(f"sk_off_{w}", f"E_off ({lab})", "J", skews.tolist(), [dpt_energy(t, vds, i, "off", T_OFF, V, I, float(sk), w, pre, post)[0] for sk in skews], dash=True)
    res.add_plot("p_skew", "전류 채널 skew에 따른 E_on·E_off", ["sk_on_fixed", "sk_on_threshold", "sk_off_fixed", "sk_off_threshold"], x_label="current skew", x_unit="s", y_label="E", y_unit="J", kind="xy", level="D",
                 vlines=[{"x": 0.0, "label": "deskew 0"}],
                 proved="소자를 바꾸지 않아도 전류 채널 수 ns 이동만으로 E_on/E_off가 크게 바뀐다 — 측정 정렬이 손실 순위를 바꿀 수 있다.",
                 not_yet="실제 probe 지연·대역폭 분포는 측정 셋업에서 확인해야 한다 (EX03 감도 분석).")
    res.circuit = {"diagram": _dpt_circuit(v).to_json(), "intervals": bands, "plot_group": ""}
    res.extra["dpt_settings"] = {"v_ref": v["v_ref"], "i_ref": v["i_ref"], "skew_s": skew, "window": v["window"], "pre_s": pre, "post_s": post, "coss_boundary": "terminal integral includes DUT Coss exchange; channel energy excludes it"}
    res.verdict("PASS_WITHIN_MODEL", "합성 D 수준 셀: 에너지 잔차·허용오차 수렴·gate 전하 보존 통과. E 값은 이 합성 소자·회로·창 정의에서만 의미가 있다.")
    if v["i_ref"] == "inductor" or v["v_ref"] == "power_source" or abs(skew) > 0:
        res.warnings.append(f"저장된 설정이 기준 정의와 다르다: E_on이 {((Eu['on'][0] - Eon) / Eon * 100) if Eon else float('nan'):+.1f} % 달라졌다 (소자는 그대로).")
    res.assumptions += [
        "합성 SiC형 MOSFET: V_th·g_m 채널(tanh 선형영역), C_gs 일정, C_gd(v)·C_ds(v) = C₀/√(1+v/V₀), 고정 T_j, 역회복 없음",
        f"power loop: L_a {v['La'] * 1e9:g} + L_b {v['Lb'] * 1e9:g} (‖R_p {v['Rp']:g} Ω) + L_s {v['Ls'] * 1e9:g} + L_sH {v['LsH'] * 1e9:g} nH, R_loop 2 mΩ",
        "부하 인덕터는 사건 동안 일정 전류원 I_L (두 번째 펄스 전 전류 증가 무시)",
        "드라이버: 이상 전압원 + 2 ns 선형 slew, on/off 경로별 R_g, gate loop L_g; 상측 gate는 V_off,H로 유지",
        "stiff solver(LSODA) + 이벤트 구간 분할; 에너지는 적분상태로 solver와 같은 정확도로 적분",
    ]
    res.not_valid_for += ["특정 vendor 소자의 E_on/E_off·overshoot 예측", "단락(SC) 생존·보호 인증", "역회복이 큰 Si diode 조합", "EMI 스펙트럼 (EX09)"]
    res.interpretation = (
        f"turn-on: gate가 V_th를 넘으면 전류가 먼저 I_L까지 오르고(HS C_oss 충전 전류로 {ipk:.4g} A까지), 그 뒤 Miller 구간에서 v_DS가 떨어진다. turn-off는 반대로 v_DS가 먼저 오르고 "
        f"전류가 떨어지며 L_loop·di/dt 때문에 {vpk:.4g} V까지 overshoot한다. 링잉은 이상 스위치가 아니라 L_loop와 C_oss가 만든 것이다. 같은 파형에서 E_on은 창·참조·skew에 따라 "
        f"달라지고(표), 단자 적분 E_off의 일부({dEc_off * 1e6:.3g} µJ)는 소산이 아니라 C_oss 저장에너지다 — 그 에너지는 다음 turn-on 때 채널에서 소산된다."
    )
    return res


def _spike_run(v: dict, over: dict | None = None, post: float = 230e-9):
    vv = dict(v)
    vv.update(over or {})
    dev = _dut_from(vv)
    spec = _dpt_spec(vv, dev)
    t_on = T_OFF
    cell = Cell(spec, [(t_on, "on")], start="off")
    run = cell.simulate(t_on + post, n_samples=400, rtol=1e-7, t_fine=[(t_on - 10e-9, t_on + post, 9000)])
    t = run.t
    o = run.out
    ichH = o["ichH"]
    m = t >= t_on - 5e-9
    Qst = float(np.trapezoid(np.maximum(ichH[m], 0.0), t[m]))
    EstH = float(run.energy["E_chH"])
    vK = o["vgsHK"]
    vPS = o["vgsHPS"]
    vdie = run.x[cell.i_vgsH]
    H = cell.H
    Qoss = float(H.Q_ds(np.array([vv["Vbus"]]))[0] + H.Q_gd(np.array([vv["Vbus"]]))[0])
    return {"cell": cell, "run": run, "t": t, "t_on": t_on, "vK": vK, "vPS": vPS, "vdie": vdie, "ichH": ichH, "Qst": Qst, "EstH": EstH, "Qoss": Qoss,
            "maxK": float(vK[m].max()), "maxPS": float(vPS[m].max()), "maxdie": float(vdie[m].max()), "art": float((vPS - vK)[m].max()), "Vth": vv["Vth"], "v": vv}


def _classify_spike(s, real_frac=0.05, marg_frac=0.005):
    """Classify with the model's hidden truth: shoot-through charge relative to the high-side Q_oss."""
    r = s["Qst"] / s["Qoss"]
    if r >= real_frac:
        return "REAL", f"실제 Miller turn-on: 상측 채널 {s['Qst'] * 1e9:.3g} nC 도통 (Q_oss,H의 {r * 100:.2g} %, die v_GS 최대 {s['maxdie']:.3g} V > V_th {s['Vth']:g} V)"
    if r >= marg_frac:
        return "MARGINAL", f"부분 turn-on: 채널 {s['Qst'] * 1e9:.3g} nC (Q_oss,H의 {r * 100:.2g} %), die v_GS 최대 {s['maxdie']:.3g} V ≈ V_th — 여유 없음"
    if s["maxPS"] > s["Vth"]:
        return "ARTIFACT", f"측정 기준 오차: power-source 기준 {s['maxPS']:.3g} V가 V_th를 넘지만 die 최대 {s['maxdie']:.3g} V, 채널 전하 ≈ 0"
    return "NONE", f"문턱 교차 없음 (power-source 최대 {s['maxPS']:.3g} V)"


SPIKE_KO = {"REAL": "실제 Miller turn-on (shoot-through)", "MARGINAL": "부분 turn-on (여유 없음)", "ARTIFACT": "측정 기준 오차 (L_sH·di/dt)", "NONE": "문턱 교차 없음"}


def run_vgs_spike(v: dict) -> Result:
    res = Result("FL02", "vgs_spike", "D (합성 commutation cell)")
    s0 = _spike_run(v)
    cls, why = _classify_spike(s0)
    t, t_on = s0["t"], s0["t_on"]
    cell, run = s0["cell"], s0["run"]
    V, I = v["Vbus"], v["IL"]
    res.add_metric("maxPS", "상측 v_GS 최대 (power-source 기준 probe)", s0["maxPS"], "V", basis="캡처에서 보이는 값")
    res.add_metric("maxK", "상측 v_GS 최대 (Kelvin 핀 probe)", s0["maxK"], "V", basis="Kelvin 단자 = die + R_g,int·i_G")
    res.add_metric("maxdie", "상측 v_GS 최대 (die 내부, 모델 전용)", s0["maxdie"], "V", basis="채널을 실제로 여는 전압")
    res.add_metric("Vth", "상측 V_th (합성)", v["Vth"], "V")
    res.add_metric("art", "측정 기준 차이 max(v_PS − v_K) = L_sH·di_HS/dt", s0["art"], "V", basis="공통 source inductance가 만든 겉보기 전압")
    res.add_metric("Qst", "상측 채널 도통 전하 ∫i_ch,H dt (shoot-through)", s0["Qst"], "C", basis="실제 turn-on의 직접 증거 (모델 전용)")
    res.add_metric("Est", "상측 채널 소산 에너지", s0["EstH"], "J")
    res.add_metric("class", "판정 (합성 모델의 내부 상태 기준)", SPIKE_KO[cls], "", note=why)
    o = run.out
    didt_H = np.gradient(o["idc"] - I, t)
    m = t >= t_on - 5e-9
    res.add_check(check_close("겉보기 전압 = L_sH·max(di_HS/dt) (측정 정의 일관성)", s0["art"], v["LsH"] * float(didt_H[m].max()), 0.02, "출력식의 L_sH·(선형계에서 푼 di/dt) vs 표본 전류의 수치 미분", True, "V",
                              detail="probe 기준점이 바뀌면 L_sH에 걸린 유도전압이 v_GS 캡처에 그대로 더해진다"))
    led = run.energy
    res.add_check(Check("에너지 잔차 (turn-on 사건)", "PASS" if abs(led["normalised"]) < 1e-5 else "FAIL", led["normalised"], "rel", 1e-5, path="포트·소산·저장에너지 원장 (상태식과 독립 정의)", independent=True,
                        detail=f"잔차 {led['residual']:.3g} J, 상측 채널 소산 {led['E_chH'] * 1e6:.4g} µJ"))
    # discriminating tests (each is a separate simulation)
    tests = [("기준 조건", {}, s0)]
    if v["clamp"]:
        tests.append(("Miller clamp 끔", {"clamp": False}, None))
    else:
        tests.append(("Miller clamp 켬 (R_clamp)", {"clamp": True}, None))
    tests.append(("R_gH 2배 (off 유지 약화)", {"RgH": 2 * v["RgH"]}, None))
    tests.append(("L_sH 1/10 (layout·Kelvin 개선)", {"LsH": 0.1 * v["LsH"]}, None))
    rows = []
    outs = []
    for name, over, s in tests:
        s = s or _spike_run(v, over)
        c2, _ = _classify_spike(s)
        outs.append((name, s))
        rows.append([name, s["maxPS"], s["maxK"], s["maxdie"], s["Qst"] * 1e9, SPIKE_KO[c2]])
    res.tables.append(Table("t_disc", "판별 시험: 무엇을 바꾸면 무엇이 반응하나", ["조건", "v_GS 최대 PS [V]", "v_GS 최대 Kelvin [V]", "v_GS 최대 die [V]", "채널 전하 [nC]", "모델 판정"], rows,
                            note="실제 Miller turn-on은 gate hold 임피던스(clamp·R_gH)에 반응하고, L_sH 겉보기 전압은 layout(측정 기준)에 반응한다. 현장에서는 die 전압 대신 Kelvin v_GS·DC-link 전류 동시성·clamp 변화 반응으로 구분한다."))
    # plots
    a, b = t_on - 5e-9, t_on + 120e-9
    mm = (t >= a) & (t <= b)
    tt = t[mm].tolist()
    res.add_series("gH_ps", "상측 v_GS (power-source 기준 probe)", "V", tt, s0["vPS"][mm].tolist())
    res.add_series("gH_k", "상측 v_GS (Kelvin 핀)", "V", tt, s0["vK"][mm].tolist())
    res.add_series("gH_die", "상측 v_GS die (모델 전용)", "V", tt, s0["vdie"][mm].tolist(), dash=True)
    res.add_series("i_d", "i_D DUT", "A", tt, run.x[cell.i_i][0][mm].tolist())
    res.add_series("i_chH", "i_ch 상측 채널 (shoot-through)", "A", tt, s0["ichH"][mm].tolist())
    res.add_series("i_dc", "i_dc DC-link", "A", tt, o["idc"][mm].tolist(), dash=True)
    res.add_series("v_ds", "v_DS DUT", "V", tt, run.x[cell.i_vds][0][mm].tolist())
    res.add_series("v_dsH", "v_DS 상측", "V", tt, run.x[cell.i_vdsH][mm].tolist())
    res.add_series("art", "L_sH·di_HS/dt (PS − Kelvin)", "V", tt, (s0["vPS"] - s0["vK"])[mm].tolist())
    res.add_series("id_all", "i_D", "A", tt, run.x[cell.i_i][0][mm].tolist())
    res.add_series("IL_all", "I_L", "A", tt, [I] * int(mm.sum()))
    res.add_series("iH_all", "−i_HS", "A", tt, (-(o["idc"] - I))[mm].tolist())
    vds = run.x[cell.i_vds][0]
    i = run.x[cell.i_i][0]
    i10n = first_cross(t, i, 0.1 * I, t_on, t_on + 200e-9, True)
    v90n = first_cross(t, vds, 0.9 * V, i10n or t_on, t_on + 200e-9, False)
    v10n = first_cross(t, vds, 0.1 * V, v90n or t_on, t_on + 200e-9, False)
    bands = []
    for x0, x1, md, lab in ((a, t_on, "freewheel", "환류"), (t_on, i10n, "ton_delay", "지연"), (i10n, v90n, "ton_i", "i↑"), (v90n, v10n, "ton_v", "v↓"), (v10n, b, "on2", "ON")):
        if x0 is not None and x1 is not None and x1 > x0:
            bands.append({"x0": float(x0), "x1": float(x1), "mode": md, "label": lab})
    th = [{"y": v["Vth"], "label": f"V_th {v['Vth']:g} V"}]
    res.add_plot("p_g", "상측(OFF) 소자의 v_GS: 어떤 기준점에서 보았나", ["gH_ps", "gH_k", "gH_die"], y_label="v_GS", y_unit="V", bands=bands, group="sp", level="D", hlines=th,
                 proved="같은 순간 같은 gate를 power-source 기준으로 보면 L_sH·di/dt가 더해져 V_th를 넘어 보일 수 있고, Kelvin·die 전압은 다르다는 것을 보였다.",
                 not_yet="probe의 CM 오차·대역 제한은 모델에 없다. 실제 die 전압은 측정할 수 없으므로 현장 판정은 전류 증거로 한다.")
    res.add_plot("p_i", "전류 증거: 상측 채널이 실제로 열렸나", ["i_d", "i_chH", "i_dc"], y_label="i", y_unit="A", bands=bands, group="sp", level="D",
                 proved="실제 turn-on이면 상측 채널 전류(shoot-through)가 생기고 DC-link 전류가 그만큼 더 흐른다. 겉보기 스파이크만이면 채널 전류는 0이다.",
                 not_yet="이 합성 소자의 V_th·C_gd/C_gs 비가 결과를 정한다. 실제 소자는 온도에 따라 V_th가 낮아진다.")
    res.add_plot("p_v", "dv/dt와 di/dt의 원천: DUT turn-on", ["v_ds", "v_dsH"], y_label="v", y_unit="V", bands=bands, group="sp", level="D",
                 proved="상측 v_DS가 V_bus로 오르는 dv/dt가 C_gd를 통해 상측 gate로 전하를 밀어 넣는다.", not_yet="이 turn-on 사건은 포트·소산·저장에너지 원장(잔차 < 1e-5)으로 닫혔다. dv/dt 크기는 합성 die의 C_gd(v)·C_ds(v)와 구동 R_g, 고정 T_j가 정하며, 실제 소자의 커패시턴스 곡선·역회복·온도 의존과 probe 대역 제한은 포함하지 않는다.")
    res.add_plot("p_art", "겉보기 전압 L_sH·di_HS/dt", ["art"], y_label="Δv", y_unit="V", bands=bands, group="sp", level="D",
                 proved="두 probe 기준의 차이는 정확히 공통 source inductance의 유도전압이다.", not_yet="차이 전압은 출력식의 L_sH·di/dt와 표본 전류의 수치 미분이 2 % 이내로 일치해 확인했다. L_sH를 집중 인덕턴스 하나로 두었으므로 패키지 내부의 분포·상호 인덕턴스와 probe의 공통모드 오차·대역 제한은 이 겉보기 전압에 들어 있지 않다.")
    res.circuit = {"diagram": _dpt_circuit(v).to_json(), "intervals": bands, "plot_group": "sp"}
    res.add_metric("Qst_rel", "shoot-through 전하 / Q_oss,H(V_bus)", s0["Qst"] / s0["Qoss"], "", basis="판정 기준: ≥ 5 % 실제 turn-on, 0.5–5 % 부분 turn-on (학습용 기준)")
    if cls == "REAL":
        res.verdict("FAIL_CONSTRAINT", why)
    elif cls == "MARGINAL":
        res.verdict("MARGINAL", why)
    elif cls == "ARTIFACT":
        res.verdict("UNRESOLVED_ROOT_CAUSE", "power-source 기준 v_GS 캡처 하나로는 실제 shoot-through인지 판정할 수 없다 (Kelvin v_GS, DC-link 전류 동시성, clamp·R_g,off 변화 반응으로 분리)")
        res.verdict("PASS_WITHIN_MODEL", why)
    else:
        res.verdict("PASS_WITHIN_MODEL", why)
    res.assumptions += ["실험 3과 같은 합성 셀; DUT turn-on 한 사건 (상측은 body diode로 I_L 환류 중에서 시작)", "상측 gate: V_off,H를 R_gH(+R_g,int)로 유지; Miller clamp는 off 동안 늘 켜진 저임피던스 경로(R_clamp)로 단순화", "Kelvin 핀 = die source (내부 source 인덕턴스 없음)"]
    res.not_valid_for += ["실제 소자의 false turn-on 여유 (온도별 V_th, C_rss/C_iss 데이터 필요)", "probe CM rejection·대역 오차", "gate 산화막 신뢰성 판정"]
    res.interpretation = (
        f"DUT가 켜지며 상측 전류가 −I_L에서 0으로 빠르게 바뀌면 L_sH에 {s0['art']:.3g} V가 걸린다. power-source 기준 probe는 이 전압을 gate 전압처럼 보여 준다. "
        f"실제로 채널을 여는 것은 die 전압(최대 {s0['maxdie']:.3g} V)이며, 이것은 C_gd·dv/dt로 들어온 전하가 C_gs와 hold 경로(R_gH, clamp)로 나뉘는 결과다. "
        "그래서 판정은 threshold crossing이 아니라 전류 증거(상측 채널·DC-link 전류의 동시성)와 hold 임피던스 변화에 대한 반응으로 한다."
    )
    return res


# ======================================================================================
# Lab definition and learning content
# ======================================================================================

_Q_LOSS = [
    Question(
        "고객이 더 낮은 R_DS(on) 부품으로 drop-in 교체했는데 온도가 올랐습니다. 불량입니까?",
        "첫 답은 불량이 아니다. 같은 조건에서 손실 구성을 비교한다: 전도는 I_rms²R이라 줄었지만 큰 die는 Q_g·Q_oss·E_on/E_off가 커서 f_s가 높으면 스위칭손실이 더 늘 수 있다 "
        "(합성 예: 100 kHz에서 A 14 W, B 20.5 W, 교차 18.75 kHz). 다음으로 gate 파형, dead time, 열 경계, 측정 조건의 변화를 확인한다. R_g를 바꾸는 조치면 overshoot와 보호 동작도 다시 검증한다.",
        "The customer replaced a part with a lower R_DS(on) and it runs hotter. Is it defective?",
        "Not as a first answer. Conduction loss fell, but a larger die usually brings more gate charge, output charge and switching energy, so at a high switching frequency the total can rise. In the synthetic example the lower-R_DS(on) part loses 20.5 W against 14 W at 100 kHz, and the crossover is 18.75 kHz. Then I would compare gate waveforms, dead time, the thermal boundary and the measurement conditions.",
        ["전도 vs 스위칭", "f_s 의존", "교차 주파수", "gate·dead time·열 경계"],
        kind="pressure",
    ),
    Question(
        "E_on 데이터에 E_oss를 더해야 합니까?",
        "데이터 제공자의 적분 정의부터 확인한다. hard turn-on에서 C_oss 에너지는 채널에서 소산되므로 E_on 측정에 이미 들어 있는 경우가 많다. 이미 포함됐는데 E_oss를 더하면 중복이고, "
        "정의를 모르면 두 경계를 모두 계산해 순위가 바뀌는지 본다 — 바뀌면 순위를 확정하지 않는다.",
        "Should I add E_oss to the E_on figure?",
        "First check how the data provider integrated it. In hard turn-on the output-capacitance energy is dissipated in the channel, so it is often already inside the measured E_on; adding it again double-counts. If the definition is unknown, compute both boundaries and do not rank the parts if the order flips.",
        ["적분 정의", "중복 계산", "두 경계 비교"],
    ),
    Question(
        "병렬 소자 수를 늘리면 손실이 계속 줄어듭니까?",
        "아니다. 이상적 전도손실은 1/N로 줄지만 총 gate 전하·출력용량·스위칭 상호작용은 N과 함께 커진다. 최적 N은 f_s와 E의 전하 성분에 달려 있고, 정적·동적 전류분담은 따로 검증한다(EX03).",
        "Does adding parallel devices keep reducing the loss?",
        "No. Ideal conduction loss falls as one over N, but total gate charge, output capacitance and switching interaction grow with N. The best N depends on the switching frequency, and static and dynamic current sharing must be verified separately.",
        ["1/N 전도", "N 비례 전하·용량", "동적 분담 별도"],
    ),
]

_Q_GATE = [
    Question(
        "L_loop 15 nH, di/dt 5 kA/µs면 overshoot는 75 V입니까?",
        "75 V는 L·di/dt의 1차 크기다. 실제 peak은 di/dt가 일정하지 않고 C_oss와 공진하며 감쇠가 있으므로 다르다. 측정·합성 셀로 확인하고, 이 숫자를 V_DS 정격 여유 계산의 유일한 근거로 쓰지 않는다.",
        "With 15 nH and 5 kA/µs, is the overshoot 75 V?",
        "Seventy-five volts is the first-order size of L times di/dt. The real peak differs because di/dt is not constant and the loop resonates with the output capacitance, so I would confirm it with a waveform before using it for voltage margin.",
        ["1차 크기", "공진·감쇠", "파형 확인"],
        kind="calc",
    ),
    Question(
        "SC 생존시간 3 µs, 보호 체인 1.8 µs입니다. 1.2 µs 마진이 있다고 말해도 됩니까?",
        "아니다. 3 µs는 조건부 가정이고 부품 정격·시험조건(전압·온도·gate)·보증이 확인되지 않았다. 온도·분산·최악조건을 검토하기 전에는 확정 마진이 아니다. "
        "또한 소자 보호(µs)와 시스템 안전반응(수십~수백 µs)은 다른 계층이다 — DESAT로 gate를 끄는 것이 모터·DC-link의 안전상태를 정하지 않는다.",
        "The short-circuit withstand is 3 µs and the protection chain is 1.8 µs. Can I claim 1.2 µs of margin?",
        "No. The 3 µs is a conditional assumption without a confirmed rating, test voltage, temperature or gate condition, so the remainder is not a margin until temperature, spread and worst cases are reviewed. Device protection in microseconds and the system safe-state reaction in tens to hundreds of microseconds are separate layers.",
        ["조건부 가정", "최악조건 검토 전", "소자 vs 시스템 계층"],
        kind="pressure",
    ),
    Question(
        "드라이버 peak 전류가 9 A인데 Miller 시간을 Q_gd/9 A로 계산해도 됩니까?",
        "안 된다. plateau 동안의 실제 gate 전류는 (V_drv − V_plateau)/(R_drv + R_g + R_g,int)이고 공급전압 droop·패키지 경로가 더해진다. 합성 예에서는 2 A라 10 ns다.",
        "The driver is rated 9 A peak. Can I compute the Miller time as Q_gd over 9 A?",
        "No. During the plateau the gate current is the drive voltage minus the plateau voltage divided by the total gate resistance, including the driver output resistance and the internal gate resistance. In the synthetic example that is 2 A and 10 ns.",
        ["plateau 전류", "R_g,int 포함", "광고 peak ≠ 실제"],
    ),
]

_Q_DPT = [
    Question(
        "DPT에서 E_on을 비교하기 전에 무엇을 고정합니까?",
        "v 참조면(Kelvin vs power source), i 참조(소자 단자 vs 인덕터/DC-link), deskew, 대역폭, 적분창 정의, C_oss 포함 경계를 먼저 고정하고 결과와 함께 저장한다. "
        "인덕터 전류를 소자 전류로 쓰면 C_oss·diode 전류 분배를 놓친다. gate 설정마다 창을 유리하게 바꾸지 않는다.",
        "What do you fix before comparing E_on values from a double-pulse test?",
        "The voltage reference plane, the current referent, the deskew, the bandwidth, the integration window and the output-capacitance boundary, and I store them with the result. Using the inductor current as the device current misses the capacitive current, and the window is not tuned per gate setting.",
        ["v/i referent", "deskew", "창 정의", "Coss 경계", "설정 저장"],
    ),
    Question(
        "이상 스위치 모델에서 gate ringing이 보였다고 보고해도 됩니까?",
        "안 된다. 이상 스위치에는 gate·C_oss·loop L이 없어 ringing이 생길 수 없다. ringing은 L과 C를 가진 회로(이 실험의 합성 셀)에서만 생기며, 그것도 합성 소자의 경향 결과다.",
        "Can you report gate ringing from an ideal-switch model?",
        "No. An ideal switch has no gate, output capacitance or loop inductance, so it cannot ring. Ringing appears only in a circuit with those inductances and capacitances, and here it is a synthetic trend.",
        ["이상 스위치에 L·C 없음", "D 수준 합성 경향"],
    ),
]

_Q_SPIKE = [
    Question(
        "V_GS spike가 threshold를 넘은 캡처가 하나 있습니다. shoot-through입니까?",
        "캡처 하나로는 판정하지 않는다. 가능한 원인은 ① 실제 Miller turn-on ② source inductance에 의한 외부 참조 오차(L_s·di/dt) ③ probe CM artifact다. "
        "DUT gate–Kelvin source 측정, DC-link 전류와의 동시성(shoot-through면 추가 전류), clamp·R_g,off 변화에 대한 반응으로 분리한다. 전류 증거 없는 threshold crossing만으로 결론내리지 않는다.",
        "One capture shows a V_GS spike above threshold. Is it shoot-through?",
        "One capture is not enough. It could be a real Miller turn-on, a reference error from the source inductance, or a probe common-mode artifact. I would measure gate to Kelvin source, look for simultaneous extra DC-link current, and see whether the spike responds to a Miller clamp or a lower turn-off gate resistance.",
        ["세 가설", "Kelvin 측정", "전류 동시성", "clamp·R_g,off 반응"],
        kind="pressure",
    ),
]

EXPERIMENTS = [
    Experiment(
        key="loss_ab",
        title="합성 A/B 손실 비교: 낮은 R_DS(on)이 더 뜨거운 이유",
        goal=(
            "합성 소자 A(40 mΩ, 100 µJ)와 B(25 mΩ, 180 µJ)를 I_rms 10 A에서 비교해 100 kHz에서 A 14 W·B 20.5 W, 20 kHz에서 6 W·6.1 W, 교차 18.75 kHz를 계산하고, "
            "병렬 수 N의 효과를 명시적 가정과 함께 본다. 손실 포함 경계(Coss·게이트·dead time)와 실제 부품 데이터 양식(MISSING_INPUT)을 분리한다."
        ),
        params=[
            Param("RA", "소자 A R_DS(on) (운전온도)", "Ω", 0.04, "mΩ", vmin=1e-4, vmax=10.0, source="TEXTBOOK", source_note="합성 소자 A 40 mΩ", group="합성 소자"),
            Param("EA", "소자 A E_on+E_off (사건당)", "J", 100e-6, "µJ", vmin=0, vmax=1.0, source="TEXTBOOK", source_note="100 µJ", group="합성 소자"),
            Param("RB", "소자 B R_DS(on)", "Ω", 0.025, "mΩ", vmin=1e-4, vmax=10.0, source="TEXTBOOK", source_note="25 mΩ", group="합성 소자"),
            Param("EB", "소자 B E_on+E_off", "J", 180e-6, "µJ", vmin=0, vmax=1.0, source="TEXTBOOK", source_note="180 µJ", group="합성 소자"),
            Param("Irms", "스위치 RMS 전류", "A", 10.0, "A", vmin=0.01, vmax=2000, source="TEXTBOOK", source_note="10 A, 주기당 on/off 1회", group="운전점"),
            Param("fs", "스위칭 주파수 f_s", "Hz", 100e3, "kHz", vmin=100, vmax=10e6, source="TEXTBOOK", source_note="100 kHz (20 kHz 비교)", group="운전점"),
            Param("coss_boundary", "E 데이터의 Coss 포함 경계", "", "included", kind="choice", choices=[("included", "E에 이미 포함 (교재 예)"), ("excluded", "E에 없음 → E_oss·f_s 추가"), ("unknown", "정의 미확인 → 두 경계로 순위 확인")], source="ASSUMED", group="손실 경계"),
            Param("EossA", "A E_oss (합성)", "J", 20e-6, "µJ", vmin=0, vmax=1e-2, source="ASSUMED", source_note="die 면적 ∝ 1/R 가정의 합성값", group="손실 경계"),
            Param("EossB", "B E_oss (합성)", "J", 32e-6, "µJ", vmin=0, vmax=1e-2, source="ASSUMED", source_note="20 µJ × 40/25", group="손실 경계"),
            Param("QgA", "A 총 gate 전하 Q_g (합성)", "C", 100e-9, "nC", vmin=0, vmax=1e-5, source="ASSUMED", group="게이트"),
            Param("QgB", "B 총 gate 전하 Q_g (합성)", "C", 160e-9, "nC", vmin=0, vmax=1e-5, source="ASSUMED", group="게이트"),
            Param("dVg", "gate swing ΔV_g (−4 → 18 V)", "V", 22.0, "V", vmin=1, vmax=40, source="ASSUMED", group="게이트"),
            Param("k_cap", "E 중 die 수에 비례하는 전하·용량 성분 비율 k_cap", "", 0.3, "", vmin=0, vmax=1, source="ASSUMED", source_note="병렬 sweep용 합성 가정", group="병렬"),
        ],
        presets=[
            Preset("textbook", "교재 100 kHz", {}, "교재 04장 합성 A/B", ("nominal", "reference")),
            Preset("f20k", "f_s 20 kHz", {"fs": 20e3}, "교재 20 kHz 비교 (거의 같음)", ("reference", "variant")),
            Preset("coss_unknown_20k", "20 kHz, Coss 정의 미확인", {"fs": 20e3, "coss_boundary": "unknown"}, "순위를 확정할 수 없는 경우", ("reference", "failure")),
            Preset("coss_excluded", "E에 Coss 미포함 (100 kHz)", {"coss_boundary": "excluded"}, "정의가 다르면 수치가 달라짐", ("variant",)),
        ],
        run=run_loss_ab,
        model_level="A (postprocessed loss estimate)",
        suggested_change="f_s를 100 kHz → 20 kHz로 낮춘다 (그 다음 Coss 경계를 ‘미확인’으로).",
        prediction=Prediction(
            "f_s를 100 → 20 kHz로 낮추면 두 소자의 손실 차이는?",
            ["B가 여전히 훨씬 뜨겁다", "거의 같아진다", "B가 훨씬 시원해진다", "모르겠다"],
            "거의 같아진다",
            "스위칭손실만 f_s에 비례해 줄어든다: A = 4 + 2 = 6 W, B = 2.5 + 3.6 = 6.1 W. 교차 주파수 18.75 kHz 부근이라 차이가 사라진다. 이 차이는 E 데이터 정의(Coss 포함 여부)의 불확실성보다 작다.",
            ["PA", "PB", "f_cross"],
            handcalc=[{"key": "PA", "label": "A 손실 @100 kHz", "unit": "W"}, {"key": "PB", "label": "B 손실 @100 kHz", "unit": "W"}, {"key": "f_cross", "label": "교차 주파수", "unit": "Hz"}],
        ),
        suggested={"fs": 20e3},
        student=(
            "소자 손실은 두 종류다. 켜져 있는 동안 저항처럼 내는 전도손실(I_rms²R)과, 켜지고 꺼지는 순간 전압과 전류가 겹쳐 생기는 스위칭손실(사건당 에너지 × 초당 사건 수)이다. "
            "R이 작은 소자는 보통 die가 커서 전하와 용량이 크고 사건당 에너지가 커진다. 그래서 스위칭을 자주 하면 R이 작은 소자가 오히려 더 뜨겁다."
        ),
        expert=(
            "① 이 비교는 이상 전류(I_rms)에 사건 에너지를 곱한 postprocessed loss estimate다. 전력단 방정식에 결합된 손실이 아니다. "
            "② E_on/E_off는 한 운전점(V_DC·I_D·T_j·R_g·상대 소자·적분 정의)의 값이라 다른 전류·온도에 반복 적용하지 않는다(FL03 loss map). "
            "③ 포함 경계: Coss 에너지는 hard turn-on 채널 소산으로 E_on에 들어 있는 경우가 많고, 역회복은 E_on과 E_rec 중 한 곳에만 넣으며, 게이트 전력은 드라이버 쪽 손실이다. "
            "정의를 모르면 두 경계를 계산해 구간이 겹치면 UNRESOLVED_RANKING으로 둔다. ④ 병렬 N: 전도 1/N, 전하·용량 성분 ∝ N — 최적 N은 f_s와 k_cap 가정에 의존한다."
        ),
        customer_ko=(
            "낮은 R_DS(on) 부품이 이 주파수에서 더 뜨거운 것은 불량이라기보다 스위칭 에너지가 커진 결과일 가능성이 큽니다. 같은 조건(전압·전류·온도·R_g·적분 정의)의 E_on/E_off와 "
            "gate 파형을 비교해 손실 구성을 먼저 확인하시죠. 실제 부품 수치는 데이터시트 조건과 함께 별도로 정리하겠습니다."
        ),
        customer_en=(
            "A lower-R_DS(on) part running hotter at this frequency is more likely a switching-energy effect than a defect. Let's compare turn-on and turn-off energies and gate waveforms under the same voltage, current, temperature, gate resistance and integration definition, and keep the real part data in a separate, sourced dataset."
        ),
        questions=_Q_LOSS,
        circuit=None,
        textbook=[TB_04],
        reference_presets=["textbook", "f20k", "coss_unknown_20k"],
        claim_limit="합성 A/B의 postprocessed 손실 비교(A). 실제 부품·제품 순위를 주장하지 않는다.",
    ),
    Experiment(
        key="gate_protect",
        title="Gate·Miller·loop 전압과 보호 타임라인 (1차 크기)",
        goal=(
            "L_loop 15 nH·5 kA/µs → 75 V, C_par 1 nF·30 kV/µs → 30 A, Q_gd 20 nC·2 A → 10 ns를 계산하고 gate-charge ODE로 plateau를 확인한다. "
            "합성 SC 보호 체인 0.5 + 0.3 + 0.2 + 0.8 = 1.8 µs를 가정한 3 µs와 비교하되, 남는 1.2 µs가 확정 마진이 아니며 소자 보호와 시스템 반응이 다른 계층임을 분리한다."
        ),
        params=[
            Param("L_loop", "power loop inductance", "H", 15e-9, "nH", vmin=0.1e-9, vmax=1e-6, source="TEXTBOOK", source_note="15 nH", group="loop"),
            Param("didt", "di/dt", "A/us", 5000.0, "kA/us", vmin=1, vmax=1e5, source="TEXTBOOK", source_note="5 kA/µs", group="loop"),
            Param("C_par", "기생 용량 C_par", "F", 1e-9, "nF", vmin=1e-13, vmax=1e-6, source="TEXTBOOK", source_note="1 nF", group="loop"),
            Param("dvdt", "dv/dt", "V/us", 30000.0, "kV/us", vmin=1, vmax=1e6, source="TEXTBOOK", source_note="30 kV/µs", group="loop"),
            Param("Qgd", "Q_gd", "C", 20e-9, "nC", vmin=1e-10, vmax=1e-5, source="TEXTBOOK", source_note="20 nC", group="게이트"),
            Param("Vdrv", "gate on 전압 V_drv", "V", 18.0, "V", vmin=5, vmax=25, source="ASSUMED", source_note="plateau 전류 2 A(교재)가 되도록 고른 합성 조합", group="게이트"),
            Param("Vpl", "Miller plateau 전압", "V", 8.0, "V", vmin=1, vmax=20, source="ASSUMED", group="게이트"),
            Param("Voff", "gate off 전압", "V", -4.0, "V", vmin=-15, vmax=0, source="ASSUMED", group="게이트"),
            Param("R_drv", "드라이버 출력저항", "Ω", 0.5, "Ω", vmin=0, vmax=50, source="ASSUMED", group="게이트"),
            Param("Rg_ext", "외부 R_g", "Ω", 3.5, "Ω", vmin=0, vmax=100, source="ASSUMED", group="게이트"),
            Param("Rg_int", "내부 R_g,int", "Ω", 1.0, "Ω", vmin=0, vmax=50, source="ASSUMED", group="게이트"),
            Param("Ipk_drv", "드라이버 광고 peak 전류", "A", 9.0, "A", vmin=0.1, vmax=100, source="ASSUMED", source_note="비교용 — 실제 plateau 전류와 다르다", group="게이트"),
            Param("Qgs", "plateau까지의 gate 전하 (V_off → V_pl)", "C", 40e-9, "nC", vmin=1e-10, vmax=1e-5, source="ASSUMED", group="게이트"),
            Param("C2", "plateau 이후 입력용량", "F", 5e-9, "nF", vmin=1e-11, vmax=1e-6, source="ASSUMED", group="게이트"),
            Param("fs", "게이트 전력용 f_s", "Hz", 100e3, "kHz", vmin=100, vmax=10e6, source="ASSUMED", group="게이트"),
            Param("t_sc", "조건부 SC 생존시간 (가정)", "s", 3e-6, "µs", vmin=0.1e-6, vmax=1e-3, source="TEXTBOOK", source_note="교재 조건부 가정 — 실제 정격 아님", group="보호"),
            Param("t_blank", "DESAT blanking", "s", 0.5e-6, "µs", vmin=0, vmax=1e-4, source="TEXTBOOK", group="보호"),
            Param("t_filter", "검출·필터", "s", 0.3e-6, "µs", vmin=0, vmax=1e-4, source="TEXTBOOK", group="보호"),
            Param("t_prop", "전파·로직", "s", 0.2e-6, "µs", vmin=0, vmax=1e-4, source="TEXTBOOK", group="보호"),
            Param("t_soft", "gate 방전·전류 소거", "s", 0.8e-6, "µs", vmin=0, vmax=1e-4, source="TEXTBOOK", group="보호"),
            Param("t_report", "고장 보고 (드라이버 → 제어기)", "s", 10e-6, "µs", vmin=0, vmax=1e-2, source="ASSUMED", group="시스템"),
            Param("t_sys", "시스템 안전반응 시간", "s", 100e-6, "µs", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="토크 안전반응 100 µs 예", group="시스템"),
        ],
        presets=[
            Preset("textbook", "교재 수치", {}, "교재 05장", ("nominal", "reference")),
            Preset("blank_long", "blanking 2 µs (오검출 회피용으로 늘림)", {"t_blank": 2e-6}, "체인이 가정 생존시간을 넘음", ("failure", "reference")),
            Preset("low_rg", "외부 R_g 1 Ω", {"Rg_ext": 1.0}, "Miller 시간 감소 → dv/dt·overshoot 증가 방향", ("variant",)),
        ],
        run=run_gate_protect,
        model_level="A (1차 screen) + B (gate-charge ODE)",
        suggested_change="DESAT blanking을 0.5 µs → 2 µs로 늘린다 (false trip을 줄이려는 흔한 조치).",
        prediction=Prediction(
            "blanking을 2 µs로 늘리면 보호 체인은 가정한 3 µs 생존시간 안에 끝나는가?",
            ["여유 있게 끝난다", "간신히 끝난다", "끝나지 않는다", "모르겠다"],
            "끝나지 않는다",
            "2 + 0.3 + 0.2 + 0.8 = 3.3 µs > 3 µs. blanking 확대는 false trip을 줄이지만 실제 fault clearing을 늦춘다. 게다가 3 µs 자체가 조건부 가정이다.",
            ["t_chain", "t_left"],
            handcalc=[{"key": "dV", "label": "L·di/dt", "unit": "V"}, {"key": "tM", "label": "Miller 시간", "unit": "s"}, {"key": "t_chain", "label": "보호 체인 합", "unit": "s"}],
        ),
        suggested={"t_blank": 2e-6},
        student=(
            "gate는 전하를 움직이는 액추에이터다. Miller 구간에서는 gate 전류가 drain 쪽 전하(Q_gd)를 옮기느라 gate 전압이 멈춘다. 빠른 edge는 loop inductance에 전압을, 기생 용량에 전류를 만든다. "
            "단락 보호는 검출·전달·소거에 시간이 걸리고, 그 합이 소자가 버티는 시간 안에 끝나야 한다."
        ),
        expert=(
            "① I_g,Miller ≈ (V_drv − V_pl)/(R_drv + R_g + R_g,int): 광고 peak 전류가 아니다. ② L·di/dt와 C·dv/dt는 크기 screen이며 실제 peak은 공진·감쇠가 정한다(실험 3). "
            "③ Miller clamp는 꺼진 gate의 저임피던스 경로, Kelvin source는 power source inductance와 gate 참조를 분리, negative gate는 여유를 주지만 산화막 권장값 검토가 먼저다. "
            "④ 소자 보호 → 고장 보고 → 시스템 반응의 세 계층: DESAT로 gate를 끄는 것이 모터·DC-link의 안전상태를 정하지 않고, 시스템 100 µs가 µs 단위 소자 보호를 대신하지 못한다. "
            "절연 드라이버 CMTI와 시스템 절연내력은 다른 사양이다."
        ),
        customer_ko=(
            "blanking을 늘리면 false trip은 줄지만 실제 단락 제거가 늦어집니다. 지금 가정한 3 µs 생존시간은 조건부 값이라, 부품 정격 조건(전압·온도·gate)과 최악조건을 먼저 확인하고 "
            "보호 체인 각 단계의 실측 시간으로 여유를 정하시죠. 시스템 안전반응과는 따로 검토하겠습니다."
        ),
        customer_en=(
            "A longer blanking time reduces false trips but delays real fault clearing. The 3 µs withstand is a conditional assumption, so let's confirm the rated test conditions and the worst case first, and size the margin from measured stage timings. The system safe-state reaction is a separate layer."
        ),
        questions=_Q_GATE,
        circuit="gate_protect",
        textbook=[TB_05],
        reference_presets=["textbook", "blank_long"],
        claim_limit="1차 크기 screen과 합성 보호 타임라인. SC survival·EMC 적합성을 주장하지 않는다.",
    ),
    Experiment(
        key="dpt_cell",
        title="합성 DPT 셀: Miller·L·di/dt·링잉과 E_on/E_off 적분의 정의",
        goal=(
            "V_th·g_m 채널, 비선형 C_gd·C_ds, gate loop, power loop·common-source inductance를 가진 합성 셀로 800 V·100 A double-pulse를 풀어 "
            "turn-off overshoot·링잉·turn-on Miller 구간을 회로의 L·C에서 얻고, E_on/E_off를 v/i 참조·deskew·적분창·Coss 경계를 저장해 계산한다."
        ),
        params=_dpt_params(
            [
                Param("v_ref", "v 참조면", "", "kelvin", kind="choice", choices=[("kelvin", "Kelvin 기준 v_DS"), ("power_source", "power source 기준")], source="ASSUMED", source_note="power source 기준에는 L_s·di/dt가 포함된다", group="측정"),
                Param("i_ref", "i 참조", "", "device", kind="choice", choices=[("device", "소자 단자 전류"), ("channel", "채널 (모델 전용)"), ("inductor", "인덕터 I_L (대용)")], source="ASSUMED", source_note="인덕터 전류는 C_oss 전류를 놓치는 잘못된 대용", group="측정"),
                Param("skew", "전류 채널 skew", "s", 0.0, "ns", vmin=-20e-9, vmax=20e-9, source="ASSUMED", source_note="−: 전류 파형이 앞당겨짐", group="측정"),
                Param("window", "적분창 정의", "", "fixed", kind="choice", choices=[("fixed", "고정창"), ("threshold", "문턱창 10 %/2 %")], source="ASSUMED", source_note="고정창은 gate 명령 기준", group="측정"),
                Param("win_pre", "고정창 시작 (명령 전)", "s", 5e-9, "ns", vmin=0, vmax=100e-9, source="ASSUMED", group="측정"),
                Param("win_post", "고정창 끝 (명령 후)", "s", 250e-9, "ns", vmin=50e-9, vmax=290e-9, source="ASSUMED", group="측정"),
            ]
        ),
        presets=[
            Preset("nominal", "Kelvin 드라이버, R_g 10 Ω", {}, "합성 기준 셀", ("nominal", "reference")),
            Preset("fast", "R_g 3 Ω (빠른 edge)", {"Rg_on": 3.0, "Rg_off": 3.0}, "overshoot·링잉 증가", ("variant", "reference")),
            Preset("no_kelvin", "드라이버 반환 = power source", {"drv_ref": "power_source"}, "common-source 되먹임", ("variant", "reference")),
            Preset("high_L", "L_a 25 nH", {"La": 25e-9}, "overshoot 증가", ("corner",)),
            Preset("skew5", "측정 skew −5 ns", {"skew": -5e-9}, "소자는 그대로, 측정만 틀림", ("failure",)),
            Preset("inductor_ref", "i 참조 = 부하 인덕터 전류", {"i_ref": "inductor"}, "C_oss 전류를 놓침", ("failure",)),
        ],
        run=run_dpt_cell,
        model_level="D (합성 commutation cell)",
        suggested_change="드라이버 반환을 Kelvin → power source 핀으로 바꾼다 (R_g 10 Ω 유지).",
        prediction=Prediction(
            "드라이버 반환을 power source 핀으로 바꾸면(L_s가 게이트 루프 안) turn-on di/dt와 E_on은?",
            ["di/dt 증가, E_on 감소", "di/dt 감소, E_on 증가", "변화 없음", "모르겠다"],
            "di/dt 감소, E_on 증가",
            "L_s·di/dt가 gate 구동전압을 거꾸로 깎는 음의 되먹임이 된다(2 nH × 수 kA/µs = 수~십수 V). 전류 상승이 느려져 V·I overlap이 길어지므로 E_on이 커진다. 대신 overshoot는 줄 수 있다.",
            ["didt_on", "Eon_fix", "vpk"],
            handcalc=[{"key": "dv_over", "label": "turn-off overshoot (L·di/dt 추정)", "unit": "V"}],
        ),
        suggested={"drv_ref": "power_source"},
        student=(
            "켜질 때는 gate가 V_th를 넘으면서 전류가 먼저 부하전류까지 오르고, 그 다음 drain 전압이 떨어진다(Miller 구간). 꺼질 때는 반대로 전압이 먼저 오르고 전류가 떨어진다. "
            "전류가 빨리 변하면 loop inductance에 전압이 생겨 V_bus 위로 튀고, 그 에너지가 소자 용량과 주고받으며 링잉이 된다."
        ),
        expert=(
            "① D 수준 합성 셀: 채널 V_th·g_m, 비선형 C_gd(v)·C_ds(v), gate loop R_g·L_g, power loop와 L_s, 상측 소자 gate loop, 손실성 loop 등가(R_p‖L_b). 링잉은 이 L·C가 만든다. "
            "② DPT 적분: v 참조(Kelvin/power source), i 참조(단자/채널/인덕터), deskew, 창(고정/문턱), Coss 경계를 결과와 함께 저장한다. 같은 파형이라도 정의가 바뀌면 E가 바뀐다(표). "
            "③ 단자 적분 E_off의 일부는 DUT C_oss 저장이고 turn-on 때 채널에서 소산된다 — E_on + E_off 합은 단자·채널이 같다. ④ 검증: 에너지 원장, 허용오차 강화, gate 전하 보존, 링잉 주파수 소신호 추정. "
            "⑤ 역회복 없음·고정 T_j·합성 C(v): vendor 소자 예측이나 SC survival이 아니다."
        ),
        customer_ko=(
            "E_on 비교는 측정 정의가 같을 때만 의미가 있습니다. v는 Kelvin 기준, i는 소자 단자 전류로, deskew와 적분창을 고정해 설정과 함께 기록하시죠. "
            "인덕터 전류로 대신하거나 창을 설정마다 바꾸면 소자를 바꾸지 않아도 수십 % 차이가 납니다."
        ),
        customer_en=(
            "Turn-on energies are comparable only under the same measurement definition. Let's measure the drain-source voltage to the Kelvin source and the device terminal current, fix the deskew and the integration window, and record them with every result. Using the inductor current or moving the window per setting changes the number by tens of percent without changing the device."
        ),
        questions=_Q_DPT,
        circuit="dpt_cell",
        textbook=[TB_05, TB_18],
        reference_presets=["nominal", "fast", "no_kelvin"],
        runtime_hint="seconds",
        claim_limit="합성 소자·회로의 D 수준 경향. vendor 소자 E·overshoot 예측, SC survival, EMI를 주장하지 않는다.",
    ),
    Experiment(
        key="vgs_spike",
        title="블라인드 문제: V_GS spike — 실제 Miller turn-on인가, 측정 기준 오차인가",
        goal=(
            "빠른 DUT turn-on에서 꺼져 있는 상측 소자의 v_GS를 power-source 기준·Kelvin 핀·die 내부로 동시에 보고, L_sH·di/dt 겉보기 전압과 C_gd·dv/dt에 의한 실제 Miller turn-on을 "
            "채널 전류(shoot-through) 증거와 clamp·R_g,off·layout 변화 반응으로 구분한다."
        ),
        params=_dpt_params(
            [
                Param("VoffH", "상측 gate off 전압 V_off,H", "V", -4.0, "V", vmin=-10, vmax=2, source="ASSUMED", group="상측 gate"),
                Param("RgH", "상측 off 유지 R_gH (외부)", "Ω", 3.5, "Ω", vmin=0.1, vmax=100, source="ASSUMED", group="상측 gate"),
                Param("clamp", "Miller clamp", "", False, kind="bool", source="ASSUMED", group="상측 gate"),
                Param("R_clamp", "clamp 경로 저항", "Ω", 0.5, "Ω", vmin=0.01, vmax=10, source="ASSUMED", group="상측 gate"),
            ],
            rg=3.0,
        ),
        presets=[
            Preset("artifact", "빠른 turn-on, 상측 −4 V 유지", {}, "겉보기 spike만", ("nominal", "reference")),
            Preset("real", "상측 0 V·R_gH 10 Ω (unipolar, 약한 hold)", {"VoffH": 0.0, "RgH": 10.0}, "실제 Miller turn-on", ("failure", "reference")),
            Preset("clamp", "0 V·10 Ω + Miller clamp", {"VoffH": 0.0, "RgH": 10.0, "clamp": True}, "clamp 효과", ("variant", "reference")),
            Preset("slow", "DUT R_g 10 Ω", {"Rg_on": 10.0, "Rg_off": 10.0}, "dv/dt·di/dt 감소", ("variant",)),
        ],
        run=run_vgs_spike,
        model_level="D (합성 commutation cell)",
        suggested_change="상측 gate off 전압을 −4 V → 0 V, R_gH를 3.5 → 10 Ω으로 바꾼다 (unipolar·약한 hold).",
        prediction=Prediction(
            "상측을 0 V·10 Ω으로 붙잡으면 같은 DUT turn-on에서 무엇이 달라지나?",
            ["겉보기 spike만 커진다", "실제로 상측 채널이 열려 shoot-through 전류가 생긴다", "아무 변화 없다", "모르겠다"],
            "실제로 상측 채널이 열려 shoot-through 전류가 생긴다",
            "C_gd·dv/dt로 들어온 전하가 C_gs와 약한 hold 경로로 나뉘어 die v_GS가 0 V에서 약 Q_gd/C_iss만큼 올라 V_th를 넘는다. L_sH 겉보기 전압은 hold 조건과 거의 무관하다.",
            ["Qst", "maxdie", "maxPS"],
        ),
        suggested={"VoffH": 0.0, "RgH": 10.0},
        student=(
            "아래 소자가 켜지면 위 소자의 drain 전압이 빠르게 오른다. drain과 gate 사이 용량(C_gd)으로 전류가 gate에 밀려들어 위 소자의 gate 전압이 잠깐 오른다. "
            "그런데 측정 probe를 power source 핀에 대면 source inductance에 걸린 전압까지 gate 전압처럼 보인다."
        ),
        expert=(
            "① 세 기준: power-source probe = Kelvin + L_sH·di/dt, Kelvin 핀 = die + R_g,int·i_G, die 전압이 채널을 연다. Miller 주입 중에는 Kelvin 핀도 die 전압을 과소평가할 수 있다. "
            "② 실제 turn-on 증거는 전류다: 상측 채널 전하, DC-link 전류의 동시 증가. ③ 판별 시험: clamp·R_g,off를 바꾸면 실제 turn-on은 반응하고 L_sH 겉보기 전압은 거의 그대로다. "
            "layout(L_sH)을 줄이면 겉보기 전압이 준다. ④ 합성 V_th·C_gd/C_gs가 결과를 정하므로 실제 여유는 온도별 V_th와 소자 데이터로 확인한다."
        ),
        customer_ko=(
            "이 캡처만으로는 shoot-through라고 판단하지 않겠습니다. gate–Kelvin source 기준으로 다시 측정하고, 같은 timebase에서 DC-link 전류에 추가 전류가 있는지 보겠습니다. "
            "Miller clamp나 R_g,off를 바꿨을 때 spike가 반응하는지로 실제 turn-on과 측정 기준 오차를 분리하시죠."
        ),
        customer_en=(
            "I would not call this shoot-through from one capture. Let's re-measure gate to Kelvin source, check the DC-link current on the same timebase for extra current, and see whether the spike responds to a Miller clamp or a lower turn-off gate resistance. That separates a real turn-on from a reference artifact."
        ),
        questions=_Q_SPIKE,
        circuit="dpt_cell",
        textbook=[TB_05],
        reference_presets=["artifact", "real", "clamp"],
        runtime_hint="seconds",
        claim_limit="합성 셀의 판별 논리(D). 실제 소자의 false turn-on 여유·산화막 신뢰성을 주장하지 않는다.",
    ),
]

LAB = Lab(
    id="FL02",
    title="Si·SiC·GaN과 데이터시트, Gate drive·DPT·보호",
    title_en="Devices, gate drive, double-pulse test and protection",
    track="basic",
    order=2,
    path_note="14일 경로 2일차 (04 소자, 05 게이트)",
    textbook=[TB_04, TB_05],
    prerequisites=["FL01"],
    summary="합성 A/B 손실 비교 → gate·Miller·보호 타임라인 → 합성 DPT 셀(D)의 E 적분 정의 → V_GS spike 판별. 모델 수준을 섞지 않는다.",
    experiments=EXPERIMENTS,
    minimum_scope="loss 비교·gate/DPT 제한된 commutation; 조건표·E 적분·L·di/dt·가설 분리 (교재 19장 표). 14/20.5 W, 6/6.1 W, 75 V, 30 A, 10 ns, 1.8 µs",
    claim_limits=[
        "합성 A/B는 비교 논리 훈련 — 실제 부품 데이터셋은 별도(MISSING_INPUT), 제품 순위 아님",
        "loss 비교는 postprocessed loss estimate (전력단에 결합된 손실 아님)",
        "DPT 셀은 합성 D 수준 경향 — vendor 소자 모델이 아니며 역회복·온도 없음",
        "이상 스위치로 gate ringing을 만들었다고 주장하지 않음 (ringing은 셀의 L·C에서만)",
        "단순 모델로 SC survival을 보장하지 않음",
    ],
    test_paths=["tests/test_fl02.py"],
)
