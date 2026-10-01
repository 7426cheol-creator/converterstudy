"""FL12 - FAE debugging: raise the quality of the questions before the waveforms (textbook ch.15).

Eight synthetic customer cases (textbook CASE A-H).  Every case is a small multiple-hypothesis model:
each textbook hypothesis is its own synthetic 'world'.  Where a hypothesis has an unknown parameter,
that parameter is calibrated so that the world reproduces the customer's single observation; the
candidate discriminating tests are then run in every world and tabulated (rows = hypotheses,
columns = tests, cells = predicted outcome), together with the groups of hypotheses each test
separates.  The verdict is always UNRESOLVED_ROOT_CAUSE: a simulation shows what each hypothesis
predicts, it never identifies the customer's root cause.

Models, each labelled with its level:
  A  SiC false turn-on: prescribed commutation edges from gate-charge timing and the off device's gate
     loop solved in closed form (piecewise exponentials, checked against an ODE and a gate-loop energy
     ledger); probe references (power source, Kelvin), probe pickup and a dead-time overlap world.
     FL02's D-level commutation cell is cited for the same separation.
  B  low-line PFC current limit: line current (FL05 run, plus a time-domain three-phase power path),
     limiter definitions, switching ripple of a 4-wire equivalent leg, and a piecewise-linear
     saturating inductor solved per switching period exactly in flux coordinates (checked by an ODE).
  C  DAB at zero power: exact piecewise-linear waveforms (_dab) for ratio mismatch, magnetizing current,
     a bridge phase offset, a DC-biased saturating core and a phase-command limit cycle (switched engine).
  D  LLC light load: rectifier-switching periodic solutions (_resonant) regulated to the output voltage,
     a 2x2 factorial device x L_m, EX02's nonlinear-Coss charge screen and residual hard-turn-on loss.
  E  transformer start-up: full bridge + series R + blocking C + a piecewise-linear saturating
     magnetizing branch in the exact switched engine (region guards), checked by an ODE written in
     flux coordinates and a PWL energy ledger; the periodic steady state (shooting) that hides the fault.
  F  efficiency vs calorimetry: GUM propagation vs Monte Carlo (seeded), calorimeter, boundary and
     stored-energy biases against the 150 W gap.
  G  EMC peak vs R_g: exact Fourier coefficients of piecewise-linear switch-node waveforms (checked by
     FFT and the sinc closed form) through the power-loop and CM-path transfer functions, an auxiliary
     converter and a control-side line; a relative proxy only.
  H  energy after gate-off: bidirectional non-isolated stage with body diodes, output capacitor, a fault
     path R/L and the battery in the exact switched engine (diode guards, clamp state, energy ledger);
     EX10's RLC numbers reproduced independently.
Numbers of other labs are cited from actual runs (the runner and cache the UI uses); every evidence row
names lab/experiment@preset.  Open answers are recorded by the app with a rubric; nothing is auto-graded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..engine.switched import AffineMode, Guard, HybridSystem, simulate
from ..model.circuit import Circuit
from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table, check_close

TB_15 = TextbookRef("fae-디버깅-파형보다-먼저-질문의-질을-높인다-fl12", "15. FAE 디버깅 — 파형보다 먼저 질문의 질을 높인다 [FL12]")
TB_05 = TextbookRef("gate-drivedpt보호-강점을-면접-증거로-만들기-fl02", "05. Gate drive·DPT·보호 [FL02] (블라인드 문제)")
TB_08 = TextbookRef("obcpfc-전력품질을-설계-변수로-바꾸기-fl05", "08. OBC·PFC — 전력품질을 설계 변수로 바꾸기 [FL05]")
TB_10 = TextbookRef("자성체-컨버터-전문가로-가는-실제-관문-fl07", "10. 자성체 [FL07]")
TB_11 = TextbookRef("dab-식에서-파형-파형에서-설계-판단으로-fl08", "11. DAB [FL08]")
TB_12 = TextbookRef("llc-공진을-말로-설명하고-식으로-확인하기-fl09", "12. LLC [FL09]")
TB_17 = TextbookRef("면접-대비-답을-외우지-않고-질문을-방어한다", "17. 면접 대비 — 답을 외우지 않고 질문을 방어한다")
TB_E02 = TextbookRef("expert-e02.-비선형-cossdead-timezvs-에너지식-하나로-판정하지-않기-ex02", "E02. 비선형 Coss·dead time·ZVS [EX02]")
TB_E03 = TextbookRef("expert-e03.-병렬-sicgate-loopdpt-계측-사용자-강점의-전문가-확장-ex03", "E03. 병렬 SiC·gate loop·DPT 계측 [EX03]")
TB_E09 = TextbookRef("expert-e09-emi-noise-source-경로-victim을-같이-본다", "E09 · EMI: noise source, 경로, victim")
TB_E10 = TextbookRef("expert-e10-고장-gate를-꺼도-에너지는-남아-있다", "E10 · 고장: gate를 꺼도 에너지는 남아 있다")
TB_E11 = TextbookRef("expert-e11-모델을-믿을-수-있는-범위-검증식별불확도", "E11 · 검증·식별·불확도")
TB_E12 = TextbookRef("expert-e12-설계-리뷰를-통과하는-답변-세-개의-통합-사례", "E12 · 설계 리뷰 (결정 메모)")

UNRESOLVED = (
    "근본원인 미확정: 시뮬레이션은 각 가설이 관측과 판별 시험에서 무엇을 예측하는지 보여줄 뿐이다. 모든 가설이 같은 관측을 재현하므로 "
    "고객 회로의 근본원인은 실제 판별 시험의 결과로만 정한다 (선택한 가설의 시뮬레이션 결과를 고객의 원인으로 확정하지 않는다)."
)

MEMO_POINTS = ["고객 목표 한 문장", "확인한 조건 / 모르는 조건", "유력 가설과 반례", "현재 권고 (임시조치와 최종변경 구분)", "부작용·잔여 리스크", "필요한 파형·시험과 완료 판정"]
MEMO_EN_TEMPLATE = (
    "At this operating point, the first constraint appears to be ___. My current hypothesis is ___, but ___ could produce a similar symptom. "
    "I would separate them by measuring ___. If the result is ___, I recommend ___ and would recheck ___."
)


# ======================================================================================
# Shared helpers: cited runs, discrimination tables, verdicts, questions
# ======================================================================================


def _run(lab: str, exp: str, preset: str | None, values: dict | None = None) -> dict:
    """Run another lab's experiment through the runner (validated inputs, shared cache).

    SI values are passed with their explicit base unit, because a bare number is read in the
    parameter's display unit (same convention as EX12)."""
    from ..labs import get_lab  # local import: the registry imports this module
    from ..runner import run_request

    params = {p.key: p for p in get_lab(lab).experiment(exp).params}
    vals = {}
    for k, val in (values or {}).items():
        p = params[k]
        if isinstance(val, (str, bool)) or p.kind != "float":
            vals[k] = val
        else:
            vals[k] = f"{val!r} {p.unit}".strip()
    return run_request(lab, exp, preset, vals, use_cache=True)["result"]


def _metric_of(r: dict, key: str) -> dict:
    for m in r["metrics"]:
        if m["key"] == key:
            return m
    raise KeyError(key)


def _fmt(v, unit: str = "", d: int = 5) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "—"
    a = abs(v)
    if unit in ("A", "V", "W", "J", "T", "Ω", "Hz", "s", "C", "H", "F") and a != 0 and (a < 1e-2 or a >= 1e4):
        from ..model.units import fmt_si

        return fmt_si(v, unit, d)
    return f"{v:.{d}g} {unit}".strip()


class Evidence:
    """Numbers taken from actual runs of other labs; each row names its source run."""

    def __init__(self, res: Result):
        self.res = res
        self.rows: list[list[str]] = []

    def cite(self, lab: str, exp: str, preset: str | None, key: str, values: dict | None = None, note: str = "", d: int = 7):
        r = _run(lab, exp, preset, values)
        m = _metric_of(r, key)
        src = f"{lab}/{exp}@{preset or (exp + ' 기본')}" + (" (값 변경: " + ", ".join(f"{k}={_fmt(v)}" for k, v in (values or {}).items()) + ")" if values else "")
        self.rows.append([src, m["label"], _fmt(m["value"], m["unit"], d), r["status"]["code"], note])
        return m["value"]

    def finish(self, note: str = ""):
        if not self.rows:
            return
        self.res.tables.append(
            Table(
                "t_evidence",
                "다른 실습의 실제 실행에서 가져온 수치 (lab/실험@preset)",
                ["출처 실행", "양", "값", "그 실행의 상태", "이 사례에서의 의미"],
                self.rows,
                note=note or "숫자를 옮겨 적지 않고 같은 runner로 다른 실습을 실행해 가져왔다. 그 실습의 모델 수준·가정이 그대로 따라온다.",
            )
        )
        self.res.add_check(
            Check(
                "인용 수치가 실제 실행에서 왔는가",
                "PASS",
                len(self.rows),
                "행",
                None,
                path="각 행 = runner.run_request(lab, 실험, preset) 결과의 metric (입력은 preset·기록된 값)",
                independent=False,
                detail="회귀 성격의 연결 확인 (인용한 실습 자체의 검증은 그 실습의 check가 맡는다)",
            )
        )


@dataclass
class Hyp:
    key: str
    label: str  # short Korean label shown as the row name
    mechanism: str  # one sentence: what this world assumes
    unknown: str = ""  # the calibrated unknown, as text


def _change_class(old: float, new: float, eps: float = 1e-12) -> str:
    if abs(old) <= eps:
        return "0 → " + ("있음" if abs(new) > eps else "0")
    r = new / old
    if r < 0.4:
        return "사라짐/크게 감소"
    if r < 0.85:
        return "감소"
    if r <= 1.18:
        return "유지"
    return "증가"


def disc_tables(res: Result, hyps: list[Hyp], tests: list[tuple[str, str]], cells: dict, title: str, note: str = "") -> dict:
    """Discrimination table (rows = hypotheses, columns = tests) and the groups each test separates.

    ``cells[h][t] = (text, cls)``: text is the predicted outcome, cls a short category used to group the
    hypotheses that a test cannot tell apart.  Returns {test: number of groups}."""
    rows = []
    for h in hyps:
        rows.append([f"{h.key} {h.label}"] + [cells[h.key][t][0] for t, _ in tests])
    res.tables.append(Table("t_disc", title, ["가설 \\ 판별 시험"] + [lab for _, lab in tests], rows, note=note or "칸 = 그 가설의 세계에서 그 시험이 보여줄 결과 (같은 관측을 재현하도록 맞춘 합성 모델)."))
    groups_out = {}
    grow = []
    for t, lab in tests:
        groups: dict[str, list[str]] = {}
        for h in hyps:
            groups.setdefault(cells[h.key][t][1], []).append(h.key)
        groups_out[t] = len(groups)
        txt = " | ".join("{" + ", ".join(v) + "}: " + k for k, v in groups.items())
        grow.append([lab, len(groups), txt, "모든 가설을 가른다" if len(groups) == len(hyps) else ("가르지 못한다" if len(groups) == 1 else "일부만 가른다")])
    res.tables.append(Table("t_groups", "시험별로 함께 남는 가설 묶음", ["판별 시험", "묶음 수", "묶음 (예측 결과가 같은 가설)", "판별력"], grow, note="한 시험으로 모든 가설이 갈리지 않으면 시험을 조합한다. 결과 범주는 학습용 구분이다."))
    return groups_out


def unresolved(res: Result, extra: str = "") -> None:
    res.verdict("UNRESOLVED_ROOT_CAUSE", UNRESOLVED + (" " + extra if extra else ""))


def q_hyp(case: str, q_ko: str, a_ko: str, q_en: str, a_en: str, points: list[str]) -> Question:
    return Question(f"CASE {case} · Q1 가설 3개: {q_ko}", a_ko, f"Case {case}, Q1 (three hypotheses): {q_en}", a_en, points, kind="interview")


def q_wave(case: str, a_ko: str, a_en: str, points: list[str]) -> Question:
    return Question(f"CASE {case} · Q2 먼저 받을 파형 3개와 측정 조건은?", a_ko, f"Case {case}, Q2: which three waveforms do you ask for first, and under what measurement conditions?", a_en, points, kind="check")


def q_first(case: str, a_ko: str, a_en: str, points: list[str]) -> Question:
    return Question(f"CASE {case} · Q3 첫 판별 실험 1개와 가설별 예상 결과는?", a_ko, f"Case {case}, Q3: your first discriminating experiment, and the result each hypothesis predicts?", a_en, points, kind="check")


def q_memo(case: str, a_ko: str, a_en: str) -> Question:
    return Question(
        f"CASE {case} · Q4 고객에게 보낼 1쪽 메모 (교재 구조: 목표 → 확인/모르는 조건 → 가설과 반례 → 권고 → 부작용·잔여 리스크 → 필요한 시험과 완료 판정)",
        a_ko,
        f"Case {case}, Q4: write the one-page customer memo (goal, known/unknown conditions, hypotheses and counter-evidence, recommendation, side effects, required tests and completion criterion).",
        a_en,
        MEMO_POINTS + ["결과별 선택 (‘검토하겠다’로 끝내지 않기)"],
        kind="interview",
    )


def _test_param(choices: list[tuple[str, str]], default: str = "none") -> Param:
    return Param("test", "판별 시험 (그래프에 표시할 것)", "", default, kind="choice", choices=choices, source="TEXTBOOK", source_note="교재 해설의 판별 실험 (표는 모든 시험을 계산)", group="판별 시험")


# ======================================================================================
# CASE A - SiC false turn-on suspicion (+5 V on the off gate, DC-link current peak up)
# ======================================================================================

A_TESTS = [
    ("none", "시험 전: 고객 probe (power source 기준)"),
    ("kelvin", "Kelvin 기준 재측정 (짧은 접지)"),
    ("rg_off", "R_g,off 감소 (또는 Miller clamp)"),
    ("sync", "V_DS·leg 전류·gate 명령 동시 취득"),
    ("short_probe", "probe tip 단락 (같은 위치·같은 lead)"),
    ("slow_on", "turn-on 감속 (상대 소자 R_g,on ↑, 임시조치)"),
]


@dataclass
class EdgeA:
    """Prescribed turn-on of the complementary device Q_H from gate-charge timing (level A/D screen).

    t = 0 is Q_H's gate command.  Current rises on [t1, t2] (load current commutates from Q_L's
    freewheeling path to Q_H), Q_L's drain voltage rises on [t2, t3] (Miller plateau of Q_H)."""

    t_don: float
    t_ri: float
    t_fv: float
    Vdc: float
    IL: float
    Coss: float

    @property
    def t1(self) -> float:
        return self.t_don

    @property
    def t2(self) -> float:
        return self.t_don + self.t_ri

    @property
    def t3(self) -> float:
        return self.t2 + self.t_fv

    @property
    def didt(self) -> float:
        return self.IL / self.t_ri

    @property
    def dvdt(self) -> float:
        return self.Vdc / self.t_fv


def edge_a(v: dict, Rg_on: float) -> EdgeA:
    R = Rg_on + v["Rg_int"]
    Ciss = v["Cgs"] + v["Cgd0"]
    t_don = R * Ciss * math.log((v["Von"] - v["Voff"]) / (v["Von"] - v["Vth"]))
    Ig_m = (v["Von"] - 0.5 * (v["Vth"] + v["Vpl"])) / R
    Ig = (v["Von"] - v["Vpl"]) / R
    return EdgeA(t_don, v["Qgs2"] / Ig_m, v["Qgd"] / Ig, v["Vdc"], v["IL"], v["Coss"])


@dataclass
class WorldA:
    Cgd: float  # effective Miller capacitance of the off device (H1 unknown)
    Ls: float  # source-lead inductance between Kelvin and the probe's power-source reference (H2 unknown)
    Mp: float  # probe ground-loop pickup of the loop di/dt (H3 unknown)
    t_dead: float  # effective dead time: Q_L gate-off command before Q_H's command (H4 unknown)
    Rg_off: float
    Rg_on: float


def gate_a(t: np.ndarray, w: WorldA, e: EdgeA, v: dict) -> dict:
    """Off device's gate loop in closed form: (C_gs + C_gd) dv/dt = C_gd dv_DS/dt + (V_drv - v)/R.

    The driver steps V_on -> V_off at -t_dead; the Miller current C_gd V_dc/t_fv flows on [t2, t3].
    Inside each interval the target u = V_drv + R i_M is constant, so v is an exact exponential."""
    R = w.Rg_off + v["Rg_int"]
    C = v["Cgs"] + w.Cgd
    tau = R * C
    iM = w.Cgd * e.dvdt
    t_off = -w.t_dead
    knots = sorted({t_off, e.t2, e.t3})

    def drv(tt):
        return v["Von"] if tt < t_off else v["Voff"]

    def target(tt):
        return drv(tt) + (R * iM if e.t2 <= tt < e.t3 else 0.0)

    # state at the start of the grid (gate assumed at V_on in steady state before its off command)
    t0 = float(t[0])
    v0 = v["Von"] if t0 < t_off else v["Voff"] + (v["Von"] - v["Voff"]) * math.exp(-(t0 - t_off) / tau)
    pts = [t0] + [k for k in knots if k > t0]
    vals = [v0]
    for a, b in zip(pts[:-1], pts[1:]):
        u = target(0.5 * (a + b))
        vals.append(u + (vals[-1] - u) * math.exp(-(b - a) / tau))
    out = np.empty_like(t)
    du = np.empty_like(t)
    seg = np.searchsorted(np.array(pts), t, side="right") - 1
    for k in range(len(pts)):
        m = seg == k
        if not np.any(m):
            continue
        a = pts[k]
        b = pts[k + 1] if k + 1 < len(pts) else math.inf
        u = target(0.5 * (a + min(b, a + 1e-9)))
        out[m] = u + (vals[k] - u) * np.exp(-(t[m] - a) / tau)
        du[m] = (u - out[m]) / tau
    vd = np.array([drv(x) for x in t])
    ig = (vd - out) / R
    return {"v": out, "dv": du, "vpin": out + v["Rg_int"] * ig, "ig": ig, "vdrv": vd, "iM": np.where((t >= e.t2) & (t < e.t3), iM, 0.0), "tau": tau, "R": R, "C": C}


def signals_a(t: np.ndarray, w: WorldA, v: dict) -> dict:
    e = edge_a(v, w.Rg_on)
    g = gate_a(t, w, e, v)
    on = t >= e.t1
    ov = g["v"] - v["Vth"]
    i_ch = np.where(on & (ov > 0), v["gfs"] * ov, 0.0)  # what Q_L's channel can carry
    i_cap_H = np.where(on, (t - e.t1) * e.didt, 0.0)  # what Q_H's turning-on channel can add beyond the load ramp
    ist = np.minimum(i_ch, i_cap_H)
    dist = np.where(i_ch <= i_cap_H, np.where(on & (ov > 0), v["gfs"] * g["dv"], 0.0), np.where(on, e.didt, 0.0))
    dist = np.where(ist > 0, dist, 0.0)
    ramp = np.where((t >= e.t1) & (t < e.t2), e.didt, 0.0)
    didt = ramp + dist
    iH = np.clip((t - e.t1) / e.t_ri, 0.0, 1.0) * e.IL
    icap = np.where((t >= e.t2) & (t < e.t3), e.Coss * e.dvdt, 0.0)
    vds = np.clip((t - e.t2) / e.t_fv, 0.0, 1.0) * e.Vdc
    ext = (w.Ls + w.Mp) * didt
    return {
        "edge": e,
        "gate": g,
        "vdie": g["v"],
        "vK": g["vpin"],
        "vmeas": g["vpin"] + ext,
        "pick": w.Mp * didt,
        "ist": ist,
        "idc": iH + icap + ist,
        "vds": vds,
        "didt": didt,
    }


def grid_a(e_list: list[EdgeA], t_lo: float = -10e-9, t_hi: float = 90e-9) -> np.ndarray:
    base = np.linspace(t_lo, t_hi, 2001)
    extra = []
    for e in e_list:
        for k in (e.t1, e.t2, e.t3):
            extra += [k - 1e-13, k, k + 1e-13]
    g = np.unique(np.concatenate([base, np.array(extra)]))
    return g[(g >= t_lo) & (g <= t_hi)]


def _peak_after(t, y, t0=0.0):
    m = t >= t0
    k = int(np.argmax(y[m]))
    return float(y[m][k]), float(t[m][k])


def obs_a(t, s) -> tuple[float, float]:
    """What the customer reads: the largest probe value once the complementary device starts to conduct."""
    return _peak_after(t, s["vmeas"], s["edge"].t1)


def calibrate_a(v: dict, key: str, base: WorldA, target: float) -> tuple[WorldA | None, str]:
    """Set the hypothesis' unknown so that the customer's probe shows ``target`` after Q_H's command."""

    def make(x):
        d = dict(base.__dict__)
        d[key] = x
        return WorldA(**d)

    def f(x):
        w = make(x)
        t = grid_a([edge_a(v, w.Rg_on)])
        return obs_a(t, signals_a(t, w, v))[0] - target

    lo_hi = {"Cgd": (base.Cgd, 2e-9), "Ls": (0.0, 50e-9), "Mp": (0.0, 50e-9), "t_dead": (0.5e-9, base.t_dead)}[key]
    fa, fb = f(lo_hi[0]), f(lo_hi[1])
    if fa * fb > 0:
        return None, f"이 가설만으로는 관측 {target:g} V를 만들 수 없다 (범위 {lo_hi} 양 끝 {fa + target:.3g} / {fb + target:.3g} V)"
    x = brentq(f, lo_hi[0], lo_hi[1], xtol=1e-30, rtol=1e-13, maxiter=300)
    return make(x), ""


def ode_check_a(w: WorldA, v: dict) -> float:
    """Independent path for the gate loop: the same circuit integrated with DOP853 (no closed form)."""
    e = edge_a(v, w.Rg_on)
    R = w.Rg_off + v["Rg_int"]
    C = v["Cgs"] + w.Cgd
    iM = w.Cgd * e.dvdt
    t_off = -w.t_dead
    t0, t1 = -40e-9, 140e-9
    v0 = v["Von"] if t0 < t_off else v["Voff"] + (v["Von"] - v["Voff"]) * math.exp(-(t0 - t_off) / (R * C))
    knots = sorted(k for k in {t_off, e.t2, e.t3} if t0 < k < t1)
    ts = np.linspace(t0, t1, 721)
    y = v0
    out_t, out_v = [], []
    for a, b in zip([t0] + knots, knots + [t1]):
        def rhs(tt, yy, a=a, b=b):
            mid = 0.5 * (a + b)
            vd = v["Von"] if mid < t_off else v["Voff"]
            im = iM if e.t2 <= mid < e.t3 else 0.0
            return [(im + (vd - yy[0]) / R) / C]

        m = (ts >= a) & (ts <= b)
        sol = solve_ivp(rhs, (a, b), [y], method="DOP853", rtol=1e-11, atol=1e-12, dense_output=True)
        if np.any(m):
            out_t += list(ts[m])
            out_v += list(sol.sol(ts[m])[0])
        y = float(sol.y[0][-1])
    tt = np.array(out_t)
    vv = np.array(out_v)
    ref = gate_a(tt, w, e, v)["v"]
    return float(np.max(np.abs(vv - ref)))


def ledger_a(w: WorldA, v: dict) -> dict:
    """Gate-loop energy over the window: driver + Miller source = resistor loss + change of 1/2 C v^2."""
    e = edge_a(v, w.Rg_on)
    t = grid_a([e])
    g = gate_a(t, w, e, v)
    E_drv = float(np.trapezoid(g["vdrv"] * g["ig"], t))
    E_M = float(np.trapezoid(g["v"] * g["iM"], t))
    E_R = float(np.trapezoid(g["R"] * g["ig"] ** 2, t))
    dW = 0.5 * g["C"] * (g["v"][-1] ** 2 - g["v"][0] ** 2)
    res = E_drv + E_M - E_R - dW
    norm = max(abs(E_drv), abs(E_M), E_R, 1e-15)
    return {"E_drv": E_drv, "E_M": E_M, "E_R": E_R, "dW": dW, "residual": res, "normalised": res / norm}


def eon_a(v: dict, Rg_on: float) -> float:
    """Turn-on overlap energy of the prescribed edge (linear current rise, then linear voltage fall)."""
    e = edge_a(v, Rg_on)
    return 0.5 * e.Vdc * e.IL * (e.t_ri + e.t_fv)


def circuit_a(v: dict) -> Circuit:
    c = Circuit("fl12_a", 700, 350, title="Half-bridge leg: 꺼진 Q_L의 gate를 어디 기준으로 재나")
    vd = c.add("vsource", "VDC", 60, 180, 90, "V_DC", f"{v['Vdc']:g} V", lpos=(38, 176, "end"))
    qh = c.add("nmos", "QH", 300, 100, 90, "Q_H (turn-on)", lpos=(334, 96, "start"))
    ql = c.add("nmos", "QL", 300, 220, 90, "Q_L (off)", lpos=(334, 216, "start"))
    ls = c.add("inductor", "LS", 300, 290, 90, "", "")
    il = c.add("isource", "IL", 460, 230, 90, "부하 I_L", f"{v['IL']:g} A", lpos=(484, 226, "start"))
    rg = c.add("resistor", "RG", 200, 220, 0, "R_g,off", f"{v['Rg_off']:g} Ω", lpos=(200, 200, "middle"))
    drv = c.add("vsource", "DRV", 120, 262, 90, "driver", f"{v['Voff']:g} V", lpos=(98, 258, "end"))
    c.wire("w_top", vd["a"], (60, 40), (300, 40), qh["a"])
    c.wire("w_mid", qh["b"], (300, 165))
    c.wire("w_mid2", (300, 165), ql["a"])
    c.wire("w_load", (300, 165), (460, 165), il["a"])
    c.wire("w_load2", il["b"], (460, 320), (300, 320))
    c.wire("w_ks", ql["b"], ls["a"])
    c.wire("w_ls", ls["b"], (300, 320), (60, 320), vd["b"])
    c.wire("w_gate", rg["b"], ql["g"])
    c.wire("w_drv", drv["a"], (120, 220), rg["a"])
    c.wire("w_kelvin", drv["b"], (120, 302), (240, 302), (240, 255), (300, 255))
    c.dot((300, 165), (300, 255), (300, 40), (300, 320))
    c.text(370, 259, "Kelvin source", "node")
    c.text(345, 294, "L_s", "node")
    c.text(380, 337, "power source (DC−)", "node")
    c.text(590, 120, "PS 기준 probe =", "note")
    c.text(590, 136, "Kelvin v_GS + L_s·di/dt", "note")
    c.probe("pIDC", "idc_H1", 180, 28, "right", "i_DC (H1 세계)")
    load = ["w_load", "IL", "w_load2"]
    c.mode("fw", "환류: Q_L이 I_L을 흘림", ["QL", "w_mid2", "w_ks", "LS", "w_ls"] + load, "Q_H turn-on 전: 부하전류가 DC−에서 Q_L(body diode/채널)을 거쳐 부하로 환류. Q_L gate는 driver가 V_off로 잡고 있다.", dim=["QH"])
    c.mode("delay", "turn-on 지연", ["QL", "w_mid2", "w_ks", "LS", "w_ls", "RG", "DRV", "w_gate", "w_drv", "w_kelvin"] + load, "Q_H gate가 V_th까지 충전되는 동안. H4(겹침) 세계에서는 Q_L gate가 아직 방전 중이다.", dim=["QH"])
    c.mode("di", "전류 전환 (di/dt)", ["VDC", "w_top", "QH", "w_mid", "QL", "w_mid2", "w_ks", "LS", "w_ls"] + load, "I_L이 Q_L에서 Q_H로 옮겨 간다: Q_L source 전류가 줄며 L_s·di/dt와 probe loop의 M·di/dt가 이 구간에 생긴다 (H2·H3).")
    c.mode("dv", "전압 전환 (dv/dt)", ["VDC", "w_top", "QH", "w_mid", "RG", "DRV", "w_gate", "w_drv", "w_kelvin", "w_ls"] + load, "Q_L의 V_DS가 오른다: C_gd·dv/dt 전류가 R_g,off를 거쳐 driver로 흐르며 Q_L gate를 밀어 올린다 (H1 Miller).", dim=["QL"])
    c.mode("on", "Q_H ON", ["VDC", "w_top", "QH", "w_mid", "w_ls"] + load, "전환 완료. die v_GS가 V_th 위에 남아 있으면 Q_L 채널로 shoot-through 전류가 계속 흐른다 (H1·H4).", dim=["QL"])
    return c


def bands_a(e: EdgeA, t_lo: float, t_hi: float) -> list[dict]:
    out = []
    for x0, x1, md, lab in ((t_lo, 0.0, "fw", "환류"), (0.0, e.t1, "delay", "지연"), (e.t1, e.t2, "di", "di/dt"), (e.t2, e.t3, "dv", "dv/dt"), (e.t3, t_hi, "on", "Q_H ON")):
        if x1 > x0:
            out.append({"x0": x0, "x1": x1, "mode": md, "label": lab})
    return out


def run_case_a(v: dict) -> Result:
    res = Result("FL12", "case_a", "A/D 합성 screen (규정 edge + gate RC 닫힌 식) + FL02 D 수준 cell 인용")
    Vobs = v["V_obs"]
    base = WorldA(Cgd=v["Cgd0"], Ls=v["Ls0"], Mp=v["Mp0"], t_dead=v["t_dead"], Rg_off=v["Rg_off"], Rg_on=v["Rg_on"])
    e0 = edge_a(v, v["Rg_on"])
    hyps = [
        Hyp("H1", "Miller coupling (실제 turn-on)", "C_gd·dv/dt 전류가 R_g,off로 Q_L gate를 실제로 V_th 위로 올린다", "유효 C_gd"),
        Hyp("H2", "common-source inductance (측정 기준 오차)", "probe 접지가 power source 핀: L_s·di/dt가 gate 전압처럼 보인다", "L_s (probe 기준)"),
        Hyp("H3", "probe artifact (접지 loop pickup)", "긴 접지 lead loop가 M·di/dt를 줍는다 (die 전압 변화 없음)", "M (pickup)"),
        Hyp("H4", "실제 dead-time overlap", "실제 dead time이 짧아 Q_H가 켜질 때 Q_L gate가 아직 방전 중", "유효 dead time"),
    ]
    keys = {"H1": "Cgd", "H2": "Ls", "H3": "Mp", "H4": "t_dead"}
    worlds: dict[str, WorldA] = {}
    notes = {}
    for h in hyps:
        w, why = calibrate_a(v, keys[h.key], base, Vobs)
        if w is None:
            notes[h.key] = why
            continue
        worlds[h.key] = w
    if not worlds:
        res.verdict("MISSING_INPUT", "어떤 가설도 관측을 재현하지 못했다: 배경 값이 이미 관측보다 크다")
        unresolved(res)
        return res
    unit = {"Cgd": ("F", 1e12, "pF"), "Ls": ("H", 1e9, "nH"), "Mp": ("H", 1e9, "nH"), "t_dead": ("s", 1e9, "ns")}
    for h in hyps:
        if h.key in worlds:
            k = keys[h.key]
            val = getattr(worlds[h.key], k)
            res.add_metric(f"cal_{h.key}", f"{h.key} 세계가 +{Vobs:g} V를 만들려면: {h.unknown}", val, unit[k][0], basis="관측 재현을 위해 맞춘 미지수 (합성)", note=h.mechanism)
    t = grid_a([edge_a(v, w.Rg_on) for w in worlds.values()])
    sig = {k: signals_a(t, w, v) for k, w in worlds.items()}
    # --- observation and hidden truth per world
    for k, s in sig.items():
        Qst = float(np.trapezoid(s["ist"], t))
        res.add_metric(f"die_{k}", f"{k}: die 내부 v_GS 최대 (측정 불가, 모델 전용)", _peak_after(t, s["vdie"], s["edge"].t1)[0], "V", basis=f"V_th {v['Vth']:g} V (가정)와 비교, Q_H 도통 이후")
        res.add_metric(f"Qst_{k}", f"{k}: shoot-through 전하 ∫i_st dt (screen)", Qst, "C", basis=f"g_fs(v_GS,die − V_th), E ≈ V_DC·Q = {v['Vdc'] * Qst * 1e6:.3g} µJ/사건")
        res.add_metric(f"idc_{k}", f"{k}: DC-link 전류 peak", float(np.max(s["idc"])), "A", basis=f"I_L {v['IL']:g} A + C_oss dv/dt {e0.Coss * e0.dvdt:.3g} A (+ shoot-through)")
    # textbook hand screens
    tb = abs(v["Qgd"] - 20e-9) < 1e-15 and abs(v["Von"] - v["Vpl"] - 10.0) < 1e-9 and abs(v["Rg_on"] + v["Rg_int"] - 5.0) < 1e-9
    res.add_metric("t_fv", "Miller 시간 t ≈ Q_gd/I_g (Q_H turn-on)", e0.t_fv, "s", ref=10e-9 if tb else None, ref_label="교재 20 nC / 2 A = 10 ns", tol=1e-9, basis=f"I_g = (V_on − V_pl)/(R_g,on + R_g,int) = {(v['Von'] - v['Vpl']) / (v['Rg_on'] + v['Rg_int']):.3g} A")
    res.add_metric("dvdt", "dv/dt (Q_L drain)", e0.dvdt / 1e6, "V/us", basis="V_DC/t_fv")
    res.add_metric("didt", "di/dt (전류 전환)", e0.didt / 1e6, "A/us", basis="I_L/t_ri")
    res.add_metric("Ls_screen", "교재 screen: L_s·di/dt = 2 nH × 2 kA/µs", 2e-9 * 2e9, "V", ref=4.0, ref_label="교재 E03 4 V", tol=1e-12, basis="공통 source inductance 겉보기 전압의 크기 감각")
    res.add_metric("icap", "C_oss·dv/dt (용량성 전류, shoot-through 아님)", e0.Coss * e0.dvdt, "A", basis="빠른 SiC edge만으로도 DC-link 전류 peak가 커진다")
    # --- discriminating tests in every world
    w_rg = {k: WorldA(**{**w.__dict__, "Rg_off": v["Rg_off_test"]}) for k, w in worlds.items()}
    w_slow = {k: WorldA(**{**w.__dict__, "Rg_on": v["Rg_on_test"]}) for k, w in worlds.items()}
    t_rg = grid_a([edge_a(v, w.Rg_on) for w in w_rg.values()])
    t_sl = grid_a([edge_a(v, w.Rg_on) for w in w_slow.values()])
    cells: dict = {}
    summary_rows = []
    for k, s in sig.items():
        e = s["edge"]
        pk0, tp0 = obs_a(t, s)
        Q0 = float(np.trapezoid(s["ist"], t))
        pkK = _peak_after(t, s["vK"], e.t1)[0]
        s_rg = signals_a(t_rg, w_rg[k], v)
        pk_rg = obs_a(t_rg, s_rg)[0]
        Q_rg = float(np.trapezoid(s_rg["ist"], t_rg))
        s_sl = signals_a(t_sl, w_slow[k], v)
        pk_sl = obs_a(t_sl, s_sl)[0]
        Q_sl = float(np.trapezoid(s_sl["ist"], t_sl))
        pick = _peak_after(t, s["pick"], e.t1)[0]
        if tp0 < e.t2 - 1e-12:
            region = "di/dt 구간 (전류 전환)"
        elif tp0 <= e.t3 + 1e-12:
            region = "dv/dt 구간 (Miller)"
        else:
            region = "dv/dt 이후"
        vK_t1 = float(np.interp(e.t1, t, s["vK"]))
        overlap = vK_t1 > v["Vth"]
        st = "추가 전하 있음" if Q0 > 1e-9 else "추가 전하 없음"
        cells[k] = {
            "none": (f"{pk0:.2f} V", "관측"),
            "kelvin": (f"{pk0:.2f} → {pkK:.2f} V ({_change_class(pk0, pkK)})", _change_class(pk0, pkK)),
            "rg_off": (f"{pk0:.2f} → {pk_rg:.2f} V, Q_st {Q0 * 1e9:.3g} → {Q_rg * 1e9:.3g} nC ({_change_class(pk0, pk_rg)})", _change_class(pk0, pk_rg)),
            "sync": (f"spike {region}; 명령 간격 {worlds[k].t_dead * 1e9:.0f} ns, Q_H 전류 시작 때 v_GS,K {vK_t1:.1f} V; {st} {Q0 * 1e9:.3g} nC",
                     f"{region} / {'겹침' if overlap else '겹침 없음'} / {st}"),
            "short_probe": (f"{pick:.2f} V", "큼" if pick > 0.5 * Vobs else "작음"),
            "slow_on": (f"{pk0:.2f} → {pk_sl:.2f} V, Q_st {Q0 * 1e9:.3g} → {Q_sl * 1e9:.3g} nC ({_change_class(pk0, pk_sl)})", "감소" if pk_sl < 0.85 * pk0 else _change_class(pk0, pk_sl)),
        }
        summary_rows.append([k, f"{pk0:.3g}", f"{pkK:.3g}", f"{_peak_after(t, s['vdie'], e.t1)[0]:.3g}", f"{Q0 * 1e9:.3g}", f"{float(np.max(s['idc'])):.4g}"])
    for k in notes:
        cells[k] = {tk: ("관측 재현 불가", "재현 불가") for tk, _ in A_TESTS}
    active = [h for h in hyps if h.key in worlds]
    groups = disc_tables(res, active + [h for h in hyps if h.key in notes], A_TESTS, cells, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설이 예측하는 결과",
                         note=f"모든 세계는 고객 probe에 +{Vobs:g} V가 보이도록 맞췄다. ‘Kelvin 재측정’은 H2·H3를, ‘R_g,off 감소’는 H1·H4를 다른 쪽과 가르고, H1과 H4는 ‘동시 취득’의 timing으로, H2와 H3는 ‘probe 단락’으로 갈린다. turn-on 감속은 모든 가설에서 spike를 줄여 판별력이 없다.")
    res.tables.append(Table("t_world", "가설별 세계: 관측과 숨은 상태", ["가설", "고객 probe v_GS 최대 [V]", "Kelvin v_GS 최대 [V]", "die v_GS 최대 [V]", "shoot-through 전하 [nC]", "DC-link 전류 peak [A]"], summary_rows,
                            note="die 전압과 shoot-through 전하는 모델 안에서만 보인다. 현장에서는 Kelvin 기준 v_GS, 전류 증거(동시성), R_g,off/clamp 반응으로 대신 판단한다."))
    # --- slow turn-on cost (temporary fix)
    E1, E2 = eon_a(v, v["Rg_on"]), eon_a(v, v["Rg_on_test"])
    dP = (E2 - E1) * v["fs"]
    res.add_metric("Eon", f"turn-on overlap 에너지 (R_g,on {v['Rg_on']:g} Ω)", E1, "J", basis="½·V·I·(t_ri + t_fv), 규정 edge (후처리 추정)")
    res.add_metric("dP_slow", f"turn-on 감속 (R_g,on {v['Rg_on']:g} → {v['Rg_on_test']:g} Ω)의 손실 증가", dP, "W", basis=f"f_s {v['fs'] / 1e3:g} kHz, 소자 1개", note="spike는 모든 가설에서 줄지만 원인을 가리지 못한다")
    res.add_metric("dT_slow", "그 손실 증가에 따른 접합 온도 상승", dP * v["Rth"], "K", basis=f"R_th,j-coolant {v['Rth']:g} K/W (가정)")
    Rs = np.geomspace(1.0, 20.0, 25)
    res.add_series("cost_P", "turn-on 손실 P_on = E_on·f_s", "W", Rs.tolist(), [eon_a(v, r) * v["fs"] for r in Rs])
    for k, w in worlds.items():
        ys = []
        for r in Rs:
            ww = WorldA(**{**w.__dict__, "Rg_on": float(r)})
            tt = grid_a([edge_a(v, ww.Rg_on)])
            ys.append(obs_a(tt, signals_a(tt, ww, v))[0])
        res.add_series(f"slow_{k}", f"{k} 고객 probe spike", "V", Rs.tolist(), ys)
    # --- plots (time in ns for readability on a shared cursor)
    keys_w = list(worlds)
    for k in keys_w:
        s = sig[k]
        res.add_series(f"vmeas_{k}", f"{k} 고객 probe v_GS (PS 기준)", "V", t.tolist(), s["vmeas"].tolist())
        res.add_series(f"die_{k}", f"{k} die v_GS", "V", t.tolist(), s["vdie"].tolist())
        res.add_series(f"idc_{k}", f"{k} DC-link 전류", "A", t.tolist(), s["idc"].tolist())
    bands = bands_a(e0, float(t[0]), float(t[-1]))
    th = [{"y": v["Vth"], "label": f"V_th {v['Vth']:g} V (가정)"}, {"y": Vobs, "label": f"관측 +{Vobs:g} V"}]
    res.add_plot("p_obs", "고객 probe(power source 기준)가 보는 Q_L v_GS: 네 가설 모두 같은 +5 V", [f"vmeas_{k}" for k in keys_w], y_label="v_GS", y_unit="V", bands=bands, hlines=th, group="a", level="A/D",
                 proved="각 가설의 미지수를 하나씩 맞추면 네 가지 원인이 모두 Q_H 도통 이후 같은 +5 V 최대값을 만든다(보정 잔차 check). H2와 H3는 고객 probe에서 파형까지 같아 선이 겹친다 — 이 캡처 하나로는 원인을 가를 수 없다.",
                 not_yet="규정된 직선 edge와 선형 gate RC의 합성 screen이다. 실제 소자의 비선형 C_gd(v)·링잉·probe 대역은 FL02 cell(인용 표)과 실측으로 확인한다.")
    res.add_plot("p_die", "모델 안에서만 보이는 die v_GS (채널을 실제로 여는 전압)", [f"die_{k}" for k in keys_w], y_label="v_GS,die", y_unit="V", bands=bands, hlines=th[:1], group="a", level="A/D",
                 proved="H1·H4 세계에서만 die 전압이 V_th를 넘는다. H2·H3 세계의 +5 V는 측정 기준·pickup이 만든 겉보기 전압이다.",
                 not_yet="die 전압은 측정할 수 없다. V_th의 온도 의존과 실제 C_rss(V)는 MISSING_INPUT.")
    res.add_plot("p_idc", "DC-link 전류: 용량성 전류와 shoot-through", [f"idc_{k}" for k in keys_w], y_label="i_DC", y_unit="A", bands=bands, group="a", level="A/D",
                 proved="모든 세계에서 빠른 edge의 C_oss·dv/dt 때문에 전류 peak가 오른다. 실제 turn-on(H1·H4)만 dv/dt 구간 밖까지 이어지는 추가 전하를 만든다.",
                 not_yet="shoot-through 전류는 g_fs·(v_GS − V_th) screen이다 (loop 인덕턴스·포화·온도 미포함).")
    tsel = v["test"]
    if tsel == "kelvin":
        for k in keys_w:
            res.add_series(f"sel_{k}", f"{k} Kelvin v_GS", "V", t.tolist(), sig[k]["vK"].tolist())
        ttl, pr = "시험: Kelvin 기준 재측정 — 기준점 오차와 pickup이 빠진 v_GS", "H2·H3의 spike는 사라지고 H1·H4의 spike는 남는다."
    elif tsel == "rg_off":
        for k in keys_w:
            res.add_series(f"sel_{k}", f"{k} 고객 probe (R_g,off {v['Rg_off_test']:g} Ω)", "V", t_rg.tolist(), signals_a(t_rg, w_rg[k], v)["vmeas"].tolist())
        ttl, pr = f"시험: R_g,off {v['Rg_off']:g} → {v['Rg_off_test']:g} Ω", "실제 turn-on(H1·H4)은 hold 임피던스에 반응하고, 측정 오차(H2·H3)는 거의 그대로다."
    elif tsel == "short_probe":
        for k in keys_w:
            res.add_series(f"sel_{k}", f"{k} 단락 probe 출력 (pickup만)", "V", t.tolist(), sig[k]["pick"].tolist())
        ttl, pr = "시험: probe tip을 자기 접지에 단락 (같은 위치·lead 배치)", "pickup만 남으므로 H3만 큰 신호를 보인다."
    elif tsel == "slow_on":
        for k in keys_w:
            res.add_series(f"sel_{k}", f"{k} 고객 probe (R_g,on {v['Rg_on_test']:g} Ω)", "V", t_sl.tolist(), signals_a(t_sl, w_slow[k], v)["vmeas"].tolist())
        ttl, pr = f"임시조치: 상대 소자 R_g,on {v['Rg_on']:g} → {v['Rg_on_test']:g} Ω", "모든 가설에서 spike가 줄어 원인을 가리지 못하고 turn-on 손실만 는다."
    else:
        for k in keys_w:
            res.add_series(f"sel_{k}", f"{k} 고객 probe v_GS", "V", t.tolist(), sig[k]["vmeas"].tolist())
        res.add_series("sel_vds", "V_DS(Q_L)/100 (동시 취득)", "V", t.tolist(), (sig[keys_w[0]]["vds"] / 100.0).tolist(), dash=True)
        ttl, pr = ("시험: V_DS·전류·gate 명령 동시 취득 — spike가 어느 구간에 있나" if tsel == "sync" else "시험 전 관측 (+ V_DS/100 timing 참조)"), "H2·H3의 spike는 di/dt 구간, H1은 dv/dt 구간, H4는 Q_H 명령 직후(gate 방전 중)에 있다."
    res.add_plot("p_test", ttl, [s.key for s in res.series if s.key.startswith("sel_")], y_label="v", y_unit="V", bands=bands, hlines=th[:1], group="a", level="A/D", proved=pr,
                 not_yet="현장 시험은 probe 대역·deskew·CM 오차를 먼저 검증해야 같은 결론이 나온다.")
    res.add_plot("p_slow", "turn-on 감속(R_g,on)은 모든 가설의 spike를 줄인다 — 판별이 아니라 비용", [f"slow_{k}" for k in keys_w], x_label="R_g,on (상대 소자)", x_unit="Ω", y_label="고객 probe spike", y_unit="V", kind="xy", log_x=True, level="A/D",
                 hlines=th[:1], vlines=[{"x": v["Rg_on"], "label": "현재"}],
                 proved="상대 소자의 R_g,on을 키우면 dv/dt·di/dt가 함께 줄어 네 세계 모두 고객 probe의 spike가 감소한다는 것을 같은 edge·gate RC 모델로 보였다: 이 조치는 판별 시험이 아니고, 효과가 있었다는 사실로 원인을 확정할 수 없다.",
                 not_yet="spike가 V_th 아래로 내려가는 R_g,on 값은 합성 V_th·C_gd에 달려 있어 고객 소자의 여유가 아니다. 감속의 비용(오른쪽 그래프의 turn-on 손실·온도)과 dead time 여유 변화는 실측으로 확인해야 한다.")
    res.add_plot("p_cost", "그 대가: turn-on 손실 P_on = E_on·f_s", ["cost_P"], x_label="R_g,on", x_unit="Ω", y_label="P_on", y_unit="W", kind="xy", log_x=True, level="A",
                 vlines=[{"x": v["Rg_on"], "label": "현재"}, {"x": v["Rg_on_test"], "label": "감속안"}],
                 proved=f"R_g,on {v['Rg_on']:g} → {v['Rg_on_test']:g} Ω이면 소자당 {dP:.3g} W, R_th {v['Rth']:g} K/W에서 약 {dP * v['Rth']:.3g} K 상승한다.",
                 not_yet="E_on은 규정 edge의 overlap 적분(후처리 추정)이다. 역회복·C_oss 방전 손실은 별도.")
    # --- checks
    kref = "H1" if "H1" in worlds else keys_w[0]
    err = ode_check_a(worlds[kref], v)
    res.add_check(Check("gate loop: 닫힌 식 vs 독립 ODE (DOP853)", "PASS" if err < 1e-6 else "FAIL", err, "V", 1e-6, path="구간별 지수 해 vs 같은 회로를 상태식으로 적분", independent=True, detail=f"{kref} 세계, 창 전체 최대 |Δv_die|"))
    led = ledger_a(worlds[kref], v)
    res.add_check(Check("gate loop 에너지 잔차 (driver + Miller 주입 = R 손실 + ½CΔv²)", "PASS" if abs(led["normalised"]) < 2e-3 else "FAIL", led["normalised"], "rel", 2e-3, path="해의 표본을 사다리꼴 적분한 포트·손실 에너지 vs 저장에너지 변화", independent=True,
                        detail=f"E_drv {led['E_drv'] * 1e9:.4g} nJ, E_M {led['E_M'] * 1e9:.4g} nJ, E_R {led['E_R'] * 1e9:.4g} nJ, ΔW {led['dW'] * 1e9:.4g} nJ (사다리꼴 오차 수준)"))
    cal_err = max(abs(obs_a(t, sig[k])[0] - Vobs) for k in keys_w)
    res.add_check(Check("모든 세계가 관측을 재현 (보정 잔차)", "PASS" if cal_err < 1e-6 * Vobs + 1e-9 else "FAIL", cal_err, "V", 1e-6, path="각 가설의 미지수를 brentq로 맞춘 뒤 다시 계산한 probe 최대값", independent=False))
    # --- cited D-level evidence (FL02 commutation cell)
    ev = Evidence(res)
    a_ps = ev.cite("FL02", "vgs_spike", "artifact", "maxPS", note="빠른 turn-on: power-source 기준 v_GS가 V_th를 넘어 보임")
    a_die = ev.cite("FL02", "vgs_spike", "artifact", "maxdie", note="그러나 die 전압은 V_th 아래 (겉보기 spike)")
    a_q = ev.cite("FL02", "vgs_spike", "artifact", "Qst", note="채널 전하 ≈ 0")
    r_die = ev.cite("FL02", "vgs_spike", "real", "maxdie", note="0 V·약한 hold: die 전압이 V_th 위")
    r_q = ev.cite("FL02", "vgs_spike", "real", "Qst", note="실제 shoot-through 전하")
    vth_fl02 = 4.0
    ok = a_ps > vth_fl02 and a_die < vth_fl02 and a_q < 1e-12 and r_die > vth_fl02 and r_q > 1e-9
    res.add_check(Check("FL02 D 수준 cell도 같은 분리 논리를 보이나", "PASS" if ok else "FAIL", r_q, "C", None, path="FL02 vgs_spike@artifact vs @real (비선형 C(v)·loop L을 가진 별도 모델)", independent=True,
                        detail=f"artifact: PS {a_ps:.3g} V > V_th, die {a_die:.3g} V, Q {a_q:.2g} C / real: die {r_die:.3g} V, Q {r_q * 1e9:.3g} nC"))
    ev.finish()
    res.circuit = {"diagram": circuit_a(v).to_json(), "intervals": bands, "plot_group": "a"}
    unresolved(res, f"판별력 있는 첫 시험: Kelvin 재측정({groups['kelvin']}묶음)과 R_g,off 변경({groups['rg_off']}묶음), 이어서 동시 취득.")
    res.verdict("MISSING_INPUT", "실제 V_th(온도), C_rss(V), 내부 gate 저항, probe 배치·접지 위치, 실제 dead time이 없어 고객 소자의 여유는 정하지 않는다 (합성 screen은 그대로 실행)")
    res.verdict("SCREEN_ONLY", "shoot-through 전류·E_on은 규정 edge 위의 screen이다 (D 수준 cell은 FL02)")
    for k, why in notes.items():
        res.warnings.append(f"{k}: {why}")
    res.assumptions += [
        "Q_H turn-on은 gate charge로 정한 직선 전류 상승(t_ri) 뒤 직선 전압 하강(t_fv): 규정된 edge, 소자 상호작용(shoot-through가 dv/dt를 늦추는 효과) 없음",
        "Q_L gate loop는 선형 C_gs + 유효 C_gd(상수), driver는 R_g,off + R_g,int 뒤의 이상 전압원",
        "측정 오차 = (L_s + M)·di/dt, di/dt는 전류 전환 ramp와 shoot-through만 (C_oss 전류 edge의 di/dt는 미포함)",
        "shoot-through = g_fs·max(0, v_GS,die − V_th), Q_H가 도통한 뒤부터; 에너지 ≈ V_DC·∫i dt",
        "각 가설 세계는 그 가설의 미지수만 바꿔 관측을 재현하고 나머지는 같은 배경값",
    ]
    res.not_valid_for += ["실제 소자의 false turn-on 여유·산화막 신뢰성", "probe CM rejection·대역 오차의 정량", "E_on의 정밀값 (DPT 측정 정의는 FL02·EX03)", "고객 회로의 근본원인 확정"]
    res.interpretation = (
        f"고객이 본 +{Vobs:g} V는 네 가지 서로 다른 세계에서 똑같이 나온다: C_gd·dv/dt로 gate가 실제로 올라가는 Miller turn-on(H1), probe 접지가 power source 핀이라 L_s·di/dt가 더해진 측정 기준 오차(H2), "
        f"접지 lead loop의 pickup(H3), 그리고 실제 dead time이 짧아 Q_L gate가 아직 방전 중인 겹침(H4). DC-link 전류 peak 증가도 모든 세계에서 일어난다 — 빠른 SiC edge의 C_oss·dv/dt = {e0.Coss * e0.dvdt:.3g} A만으로도 오른다. "
        "그래서 판정은 threshold crossing이 아니라 ① Kelvin 기준 재측정(H2·H3 제거) ② R_g,off/clamp 반응(H1·H4는 반응) ③ V_DS·전류·gate 명령의 동시 취득(spike의 구간과 추가 전하) ④ probe 단락 시험(H3)의 조합으로 한다. "
        f"turn-on을 늦추는 임시조치는 모든 가설의 spike를 줄여 원인을 가리지 못하면서 소자당 {dP:.3g} W를 더 쓴다. "
        "1쪽 메모: 목표 → 확인/모르는 조건 → 유력 가설과 반례 → 현재 권고 → 부작용 → 필요한 파형과 완료 판정. 영어 틀: " + MEMO_EN_TEMPLATE
    )
    return res


def params_a() -> list[Param]:
    g_obs, g_c, g_d, g_g, g_m, g_t = "관측", "회로", "소자 (합성)", "게이트", "측정 배경 (나머지 가설의 기본값)", "판별 시험"
    return [
        Param("V_obs", "관측: 꺼진 gate의 spike (고객 probe)", "V", 5.0, "V", vmin=0.5, vmax=30, source="TEXTBOOK", source_note="CASE A: off gate에 +5 V spike", group=g_obs),
        Param("Vdc", "DC-link 전압", "V", 800.0, "V", vmin=50, vmax=1500, source="ASSUMED", source_note="교재 E03 DPT와 같은 800 V (CASE A는 값 없음)", group=g_c),
        Param("IL", "전환되는 부하전류 I_L", "A", 40.0, "A", vmin=1, vmax=1000, source="ASSUMED", source_note="di/dt ≈ 2 kA/µs가 되도록 (교재 E03 screen과 같은 크기)", group=g_c),
        Param("Coss", "Q_L 유효 C_oss (전환 구간)", "F", 150e-12, "pF", vmin=1e-12, vmax=50e-9, source="ASSUMED", group=g_c),
        Param("fs", "스위칭 주파수 (손실 비용 계산)", "Hz", 50e3, "kHz", vmin=1e3, vmax=1e6, source="ASSUMED", group=g_c),
        Param("Rth", "R_th,j-coolant (온도 비용 계산)", "K/W", 0.5, "K/W", vmin=0.01, vmax=20, source="ASSUMED", group=g_c),
        Param("Vth", "V_th (고온, 실제값 모름)", "V", 2.5, "V", vmin=0.5, vmax=8, source="ASSUMED", source_note="교재: 실제 threshold는 모른다 → MISSING_INPUT", group=g_d),
        Param("gfs", "g_fs (V_th 위 선형 transconductance)", "A/V", 10.0, "A/V", vmin=0.1, vmax=500, source="ASSUMED", group=g_d),
        Param("Cgs", "C_gs", "F", 1.9e-9, "nF", vmin=0.05e-9, vmax=100e-9, source="ASSUMED", group=g_d),
        Param("Cgd0", "C_gd 기본값 (H1 외 세계)", "F", 12e-12, "pF", vmin=0.1e-12, vmax=5e-9, source="ASSUMED", source_note="고전압 C_rss 수준", group=g_d),
        Param("Qgd", "Q_gd (Q_H)", "C", 20e-9, "nC", vmin=1e-9, vmax=1e-6, source="TEXTBOOK", source_note="교재 05장: Q_gd 20 nC, I_g 2 A → 10 ns", group=g_d),
        Param("Qgs2", "Q_gs2 (V_th → plateau 전하)", "C", 50e-9, "nC", vmin=1e-9, vmax=1e-6, source="ASSUMED", group=g_d),
        Param("Rg_int", "내부 gate 저항 R_g,int", "Ω", 2.5, "Ω", vmin=0.0, vmax=50, source="ASSUMED", group=g_g),
        Param("Von", "gate on 전압", "V", 18.0, "V", vmin=5, vmax=25, source="ASSUMED", group=g_g),
        Param("Voff", "gate off 전압", "V", -3.0, "V", vmin=-10, vmax=0, source="ASSUMED", group=g_g),
        Param("Vpl", "Miller plateau 전압", "V", 8.0, "V", vmin=1, vmax=15, source="ASSUMED", source_note="V_on − V_pl = 10 V → I_g = 2 A (교재 05장)", group=g_g),
        Param("Rg_on", "상대 소자 R_g,on (외부)", "Ω", 2.5, "Ω", vmin=0.1, vmax=100, source="ASSUMED", group=g_g),
        Param("Rg_off", "Q_L R_g,off (외부)", "Ω", 10.0, "Ω", vmin=0.1, vmax=100, source="ASSUMED", group=g_g),
        Param("t_dead", "설정 dead time", "s", 100e-9, "ns", vmin=5e-9, vmax=5e-6, source="ASSUMED", group=g_g),
        Param("Ls0", "probe 기준 L_s 기본값 (H2 외)", "H", 0.3e-9, "nH", vmin=0.0, vmax=20e-9, source="ASSUMED", group=g_m),
        Param("Mp0", "probe pickup M 기본값 (H3 외)", "H", 0.2e-9, "nH", vmin=0.0, vmax=20e-9, source="ASSUMED", group=g_m),
        Param("Rg_off_test", "시험: 바꾼 R_g,off", "Ω", 2.5, "Ω", vmin=0.1, vmax=100, source="ASSUMED", source_note="또는 Miller clamp의 등가 저임피던스", group=g_t),
        Param("Rg_on_test", "임시조치: 바꾼 R_g,on", "Ω", 10.0, "Ω", vmin=0.1, vmax=100, source="ASSUMED", group=g_t),
        _test_param(A_TESTS),
    ]


# ======================================================================================
# CASE C - DAB at no load: the transformer is hot (P command ~ 0, phase ~ 0, AC current remains)
# ======================================================================================

C_TESTS = [
    ("none", "시험 전: P 명령 ≈ 0, φ ≈ 0에서 권선 AC 전류"),
    ("ratio", "전압비 일치 (n·V_L = V_H)"),
    ("lm", "L_m ×2 (다른 gap/변압기, 또는 L_m 측정)"),
    ("phase", "φ 명령 sweep: I_rms 최소점과 P = 0 교차"),
    ("dc", "DC 결합 전류 probe: 주기 평균·peak 비대칭"),
    ("log", "φ 명령·포트 전류의 장시간 기록"),
]


def dab_mag_sat(lam_dc: float, V: float, fs: float, Lm0: float, lam_s: float, Lsat: float):
    """Magnetizing current with a DC flux offset in a piecewise-linear saturating core (exact PWL in time).

    lam(t) = lam_dc - lam_pk + 4 lam_pk t/T on the positive half (square-wave V), mirrored on the negative half."""
    from ..engine.pwl import PWL

    T = 1.0 / fs
    lam_pk = V * T / 4.0
    ind = PwlL(Lm0, lam_s / Lm0, Lm0 / Lsat)
    ts = [0.0, T / 2, T]
    lam_at = lambda t: lam_dc - lam_pk + (4 * lam_pk * t / T if t <= T / 2 else 2 * lam_pk - 4 * lam_pk * (t - T / 2) / T)  # noqa: E731
    for level in (lam_s, -lam_s):
        for a, b in ((0.0, T / 2), (T / 2, T)):
            la, lb = lam_at(a), lam_at(b)
            if (la - level) * (lb - level) < 0:
                ts.append(a + (level - la) / (lb - la) * (b - a))
    ts = np.array(sorted(set(ts)))
    il = np.array([float(ind.i(lam_at(t))) for t in ts])
    return PWL(ts, il[:-1], il[1:]), ind


def dab_limit_cycle(V: float, L: float, fs: float, phi: float, M: int, n_cyc: int, phi_cmd: float = 0.0):
    """Phase alternating +phi / -phi every M cycles (around phi_cmd) in the exact switched engine (ideal L, R = 0)."""
    from ..engine.switched import simulate
    from ._dab import PI as DPI
    from ._dab import SPS_TEXTBOOK_C1, DABSystem, Modulation, half_wave_periodic

    mp = Modulation(DPI, DPI, phi_cmd + phi, SPS_TEXTBOOK_C1)
    mm = Modulation(DPI, DPI, phi_cmd - phi, SPS_TEXTBOOK_C1)
    sysd = DABSystem(V, V, L, fs, mod_of_cycle=lambda k: mp if (k // M) % 2 == 0 else mm, key=f"fl12lc|{phi}|{M}|{phi_cmd}")
    s0 = DABSystem(V, V, L, fs, mod_of_cycle=lambda k: mp, key=f"fl12lc0|{phi}|{phi_cmd}")
    q0 = mp.levels(1e-9)
    x0 = half_wave_periodic(s0, q0)
    T = 1.0 / fs
    tr = simulate(sysd, q0, x0, 0.0, n_cyc * T)
    return sysd, tr


def run_case_c(v: dict) -> Result:
    from ..engine.pwl import PWL
    from ..engine.switched import simulate
    from ._common import energy_ledger, ledger_check
    from ._dab import PI as DPI
    from ._dab import SPS_TEXTBOOK_C1, DABSystem, Modulation, bands_for, dab_circuit, half_wave_periodic, pwl_waves

    res = Result("FL12", "case_c", "A + C (구간선형 정확 파형, 스위칭 엔진) + FL08 실행 인용")
    VH, n, L, fs = v["VH"], v["Np"] / v["Ns"], v["L"], v["fs"]
    V2 = n * v["VL"]
    Vm = v["V_match"]
    T = 1.0 / fs
    M = int(v["M_lc"])
    w1 = pwl_waves(VH, V2, L, fs, Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1))
    target = w1.Irms
    tb = abs(VH - 900) < 1e-9 and abs(V2 - 600) < 1e-6 and abs(L - 200e-6) < 1e-12 and abs(fs - 1e5) < 1e-6
    res.add_metric("I_H1", "H1: 전압비 불일치 φ = 0의 L 순환전류 RMS", target, "A", ref=2.165063509 if tb else None, ref_label="교재 2.165063509 A", tol=1e-8, basis=f"V_H {VH:g} V, n·V_L {V2:g} V (1차 환산)")
    res.add_metric("Ipk_H1", "H1: 순환전류 peak", w1.Ipk, "A", ref=3.75 if tb else None, ref_label="교재 3.75 A", tol=1e-9)
    res.add_metric("P_H1", "H1: 전달전력", w1.P2, "W", basis="P = 0인데 전류가 흐른다")
    # H2: magnetizing current at a matched ratio (series L on the primary: the core voltage is set by the secondary bridge)
    def h2(Lm):
        return pwl_waves(Vm, Vm, L, fs, Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1), Lm)

    Lm2 = brentq(lambda Lm: h2(Lm).im.rms() - target, 1e-6, 10.0, xtol=1e-15, rtol=1e-13)
    w2 = h2(Lm2)
    # H3: actual phase offset while the command is zero (matched ratio)
    def h3(ph):
        return pwl_waves(Vm, Vm, L, fs, Modulation(DPI, DPI, ph, SPS_TEXTBOOK_C1))

    ph3 = brentq(lambda ph: h3(ph).Irms - target, 1e-6, DPI / 2, xtol=1e-15, rtol=1e-13)
    w3 = h3(ph3)
    # H4: DC flux offset in a saturating core (matched ratio, magnetizing branch)
    lam_pk = Vm * T / 4.0
    lam_s = v["Bs_ratio"] * lam_pk
    Lsat = v["Lm_core"] / v["k_core"]

    def h4(ldc):
        p, _ = dab_mag_sat(ldc, Vm, fs, v["Lm_core"], lam_s, Lsat)
        return math.sqrt(max(p.rms() ** 2 - p.mean() ** 2, 0.0))

    ldc4 = brentq(lambda x: h4(x) - target, 0.0, lam_s + 2 * lam_pk, xtol=1e-15, rtol=1e-13)
    p4, _ = dab_mag_sat(ldc4, Vm, fs, v["Lm_core"], lam_s, Lsat)
    # H5: phase-command limit cycle (+phi / -phi every M cycles), exact engine
    def h5(ph):
        _, tr = dab_limit_cycle(Vm, L, fs, ph, M, 4 * M)
        return tr.rms(2 * M * T, 4 * M * T, "iL")

    ph5 = brentq(lambda ph: h5(ph) - target, 1e-4, DPI / 2, xtol=1e-12, rtol=1e-10)
    sys5, tr5 = dab_limit_cycle(Vm, L, fs, ph5, M, 4 * M)
    hyps = [
        Hyp("H1", "전압비 불일치", f"V_H {VH:g} V와 n·V_L {V2:g} V가 달라 φ = 0에서도 v_L = ±(V₁ − V₂′)", "V₁, V₂′ (교재값)"),
        Hyp("H2", "여자전류 (작은 L_m)", "전압비는 맞지만 L_m이 작아 여자전류가 크다", "L_m"),
        Hyp("H3", "phase offset (실제 φ ≠ 명령)", "명령은 0이지만 gate 지연 차이로 실제 φ가 어긋남", "실제 φ"),
        Hyp("H4", "자속 불균형 / DC offset", "volt-second 불균형 이력으로 코어에 DC 자속 → 한쪽 포화", "DC 자속"),
        Hyp("H5", "제어 limit cycle", f"φ 명령이 {M}주기마다 ±로 뒤집혀 평균 전력은 0", "±φ 진폭"),
    ]
    res.add_metric("cal_H2", "H2 세계가 같은 RMS를 만들려면: L_m", Lm2, "H", basis=f"V₂′T/(4L_m)/√3 = {target:.4g} A")
    res.add_metric("cal_H3", "H3 세계: 실제 φ (명령 0)", ph3, "rad", basis=f"= {ph3 * 180 / DPI:.3g}° — 이때 실제 전력 {w3.P2:.4g} W")
    res.add_metric("cal_H4", "H4 세계: DC 자속 / AC 자속 진폭", ldc4 / lam_pk, "", basis=f"포화 λ_s = {v['Bs_ratio']:g}·λ_pk, L_sat = L_m/{v['k_core']:g}")
    res.add_metric("cal_H5", "H5 세계: limit cycle φ 진폭", ph5, "rad", basis=f"{M}주기마다 ±, 평균 φ = 0")
    res.add_metric("P_H3", "H3: 실제 전달전력 (명령은 0)", w3.P2, "W", note="P ≈ 0이 측정으로 확인됐다면 H3는 이 크기로는 기각된다")
    res.add_metric("Idc_H4", "H4: DC 결합 probe의 주기 평균 전류", p4.mean(), "A", basis="AC 결합 probe에는 보이지 않는다")
    P5 = tr5.energy(2 * M * T, 4 * M * T, "p2") / (2 * M * T)
    res.add_metric("P_H5", "H5: limit cycle 한 주기 평균 전력", P5, "W", basis="평균 0, 순간 포트 전류는 ±로 뒤집힌다")
    # --- port (DC) average current per cycle for every world (actual LV-side current)
    nwin = 2 * M
    cyc_t = [(k + 0.5) * T for k in range(nwin)]
    port = {
        "H1": [w1.P2 / v["VL"]] * nwin,
        "H2": [w2.P2 / v["VL"]] * nwin,
        "H3": [w3.P2 / v["VL"]] * nwin,
        "H4": [0.0] * nwin,
        "H5": [tr5.energy((2 * M + k) * T, (2 * M + k + 1) * T, "p2") / T / v["VL"] for k in range(nwin)],
    }
    for k, ys in port.items():
        res.add_series(f"port_{k}", f"{k} LV 포트 평균전류 (주기별)", "A", cyc_t, ys, style="points" if k != "H5" else "line")
    # --- winding current waveforms over 2M periods (periodic worlds repeated)
    def rep(p: PWL, periods: int):
        xs, ys = [], []
        for kk in range(periods):
            tx, ty = p.plot_points()
            xs += [x + kk * T for x in tx]
            ys += ty
        return xs, ys

    for k, p in (("H1", w1.iL), ("H2", w2.im), ("H3", w3.iL), ("H4", p4)):
        xs, ys = rep(p, nwin)
        res.add_series(f"wind_{k}", f"{k} 권선 전류", "A", xs, ys)
    smp = tr5.sample(["iL"], 2 * M * T, 4 * M * T, per_segment=2)
    res.add_series("wind_H5", "H5 권선 전류", "A", [x - 2 * M * T for x in smp["t"]], smp["iL"])
    # --- discriminating tests
    def rms_after_ratio(k):
        if k == "H1":
            return pwl_waves(VH, VH, L, fs, Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1)).Irms
        return {"H2": target, "H3": target, "H4": target, "H5": target}[k]

    def rms_after_lm(k):
        if k == "H2":
            return h2(2 * Lm2).im.rms()
        if k == "H4":
            p, _ = dab_mag_sat(ldc4, Vm, fs, 2 * v["Lm_core"], lam_s, 2 * Lsat)
            return math.sqrt(max(p.rms() ** 2 - p.mean() ** 2, 0.0))
        return target

    phis = np.linspace(-0.5, 0.5, 21)
    sweep = {}
    for k in ("H1", "H2", "H3", "H4", "H5"):
        ys, ps = [], []
        for ph in phis:
            ph = float(ph)
            if k == "H1":
                ww = pwl_waves(VH, V2, L, fs, Modulation(DPI, DPI, ph, SPS_TEXTBOOK_C1))
                ys.append(ww.Irms)
                ps.append(ww.P2)
            elif k == "H2":
                ww = pwl_waves(Vm, Vm, L, fs, Modulation(DPI, DPI, ph, SPS_TEXTBOOK_C1), Lm2)
                ys.append(math.sqrt(ww.iL.rms() ** 2 + ww.im.rms() ** 2))
                ps.append(ww.P2)
            elif k == "H3":
                ww = h3(ph + ph3)
                ys.append(ww.Irms)
                ps.append(ww.P2)
            elif k == "H4":
                ww = pwl_waves(Vm, Vm, L, fs, Modulation(DPI, DPI, ph, SPS_TEXTBOOK_C1))
                ys.append(math.sqrt(ww.Irms**2 + target**2))
                ps.append(ww.P2)
            else:
                _, trp = dab_limit_cycle(Vm, L, fs, ph5, M, 4 * M, phi_cmd=ph)
                ys.append(trp.rms(2 * M * T, 4 * M * T, "iL"))
                ps.append(trp.energy(2 * M * T, 4 * M * T, "p2") / (2 * M * T))
        sweep[k] = (ys, ps)
        res.add_series(f"sw_{k}", f"{k} I_rms (권선)", "A", phis.tolist(), ys)
    cells = {}
    for k in ("H1", "H2", "H3", "H4", "H5"):
        r_ratio = rms_after_ratio(k)
        r_lm = rms_after_lm(k)
        ys, ps = sweep[k]
        kmin = int(np.argmin(ys))
        phmin = float(phis[kmin])
        mean_dc = p4.mean() if k == "H4" else 0.0
        asym = (p4.max() + p4.min()) if k == "H4" else 0.0
        if k == "H5":
            lg, lgc = f"φ가 {M}주기마다 ±{ph5:.3g} rad로 뒤집힘, 포트 전류 ±{max(abs(x) for x in port['H5']):.3g} A 교번 (평균 0)", "교번"
        elif k == "H3":
            lg, lgc = f"일정한 실제 φ {ph3:.3g} rad, 포트 평균 {port['H3'][0]:.3g} A (P {w3.P2:.4g} W)", "일정 (전력 있음)"
        else:
            lg, lgc = "일정, 포트 평균 0 A", "일정 (전력 0)"
        cells[k] = {
            "none": (f"{target:.4g} A RMS", "관측"),
            "ratio": (f"{target:.4g} → {r_ratio:.3g} A ({_change_class(target, r_ratio)})", _change_class(target, r_ratio)),
            "lm": (f"{target:.4g} → {r_lm:.3g} A ({_change_class(target, r_lm)})", _change_class(target, r_lm)),
            "phase": (f"최소 {min(ys):.3g} A @ φ명령 {phmin:+.2f} rad", "최소점 이동 (0 A 가능)" if abs(phmin) > 0.02 else ("최소점 0, 바닥 큼" if min(ys) > 0.5 * target else "최소점 0, 바닥 작음")),
            "dc": (f"평균 {mean_dc:+.3g} A, peak 합 {asym:+.3g} A" if k == "H4" else "평균 0, 대칭", "DC·비대칭" if abs(mean_dc) > 0.05 * target else "대칭"),
            "log": (lg, lgc),
        }
    groups = disc_tables(res, hyps, C_TESTS, cells, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설이 예측하는 결과",
                         note=f"모든 세계는 P 명령 ≈ 0·φ 명령 ≈ 0에서 권선 AC 전류 {target:.4g} A RMS가 되도록 미지수를 맞췄다. 평균 출력전류만 보면 H1·H2·H4·H5는 0 A라 ‘센서 이상’처럼 보이지만 물리적으로 맞는 값이다.")
    # --- plots
    bands = bands_for(Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1), fs, nwin)
    res.add_plot("p_obs", f"P ≈ 0인데 같은 {target:.3g} A RMS: 다섯 세계의 권선 전류", [f"wind_{k}" for k in ("H1", "H2", "H3", "H4", "H5")], y_label="i", y_unit="A", bands=bands, group="c", level="C",
                 proved="전압비 불일치·작은 L_m·실제 φ offset·DC 자속(포화)·φ limit cycle이 각자의 미지수만으로 같은 권선 RMS를 만든다(구간선형 정확 적분·스위칭 엔진). 파형 모양은 다르지만 RMS 하나로는 원인을 가를 수 없다.",
                 not_yet="이상 bridge·무손실 L(H5만 스위칭 엔진)이다. 권선·코어 손실과 온도는 계산하지 않았고, 어느 세계가 고객의 변압기인지는 아직 정하지 않는다.")
    res.add_plot("p_port", "LV 포트 평균전류 (주기별): ‘평균 0’은 센서 고장의 증거가 아니다", [f"port_{k}" for k in ("H1", "H2", "H3", "H4", "H5")], x_label="t", x_unit="s", y_label="평균 전류", y_unit="A", kind="xy", level="C",
                 hlines=[{"y": 0.0, "label": "0 A"}],
                 proved="H1·H2·H4는 전력이 0이라 포트 평균전류가 0 A이고, H5는 ±로 교번해 장시간 평균만 0이다. H3만 실제 전력을 보낸다 — 평균 전류 0을 보고 센서를 의심할 이유가 없다.",
                 not_yet="출력 커패시터·배터리 임피던스가 이 교번 전류를 어떻게 거르는지는 계산하지 않았다.")
    res.add_plot("p_sweep", "판별 시험: φ 명령 sweep의 권선 I_rms", [f"sw_{k}" for k in ("H1", "H2", "H3", "H4", "H5")], x_label="φ 명령", x_unit="rad", y_label="I_rms", y_unit="A", kind="xy", level="A+C",
                 vlines=[{"x": 0.0, "label": "명령 0"}],
                 proved="H3만 최소점이 φ 명령 0에서 벗어나 0 A까지 내려간다(실제 φ가 offset만큼 어긋남). H1·H2·H4의 바닥은 φ로 없앨 수 없는 전류이고, H5는 limit cycle이 바닥을 만든다.",
                 not_yet="H4의 φ 의존은 L 전류와 여자전류를 RMS 합으로 근사했다. 제어 루프가 닫힌 상태의 sweep 응답은 다를 수 있다.")
    tsel = v["test"]
    if tsel == "ratio":
        wr = pwl_waves(VH, VH, L, fs, Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1))
        xs, ys = rep(wr.iL, nwin)
        res.add_series("t_H1", "H1 (전압비 일치 후)", "A", xs, ys)
        for k in ("H2", "H3", "H4"):
            s = next(s for s in res.series if s.key == f"wind_{k}")
            res.add_series(f"t_{k}", f"{k} (변화 없음)", "A", s.x, s.y, dash=True)
        sel = ["t_H1", "t_H2", "t_H3", "t_H4"]
        pr = "전압비를 맞추면 H1의 L 순환전류만 사라진다 — 나머지 원인은 전압비와 무관하다."
    elif tsel == "lm":
        xs, ys = rep(h2(2 * Lm2).im, nwin)
        res.add_series("t_H2", "H2 (L_m ×2)", "A", xs, ys)
        pz, _ = dab_mag_sat(ldc4, Vm, fs, 2 * v["Lm_core"], lam_s, 2 * Lsat)
        xs, ys = rep(pz, nwin)
        res.add_series("t_H4", "H4 (L_m ×2, 같은 DC 자속)", "A", xs, ys)
        sel = ["t_H2", "t_H4"]
        pr = "L_m을 두 배로 하면 H2의 여자전류는 절반이 되고, H4는 포화 구간의 전류가 바뀐다. 전압비·φ 원인(H1·H3·H5)은 L_m과 무관하다."
    elif tsel == "dc":
        for k in ("H2", "H4"):
            s = next(s for s in res.series if s.key == f"wind_{k}")
            res.add_series(f"t_{k}", f"{k} DC 결합", "A", s.x, s.y)
        res.add_series("t_mean", "H4 주기 평균 (DC)", "A", [0.0, nwin * T], [p4.mean(), p4.mean()], dash=True)
        sel = ["t_H2", "t_H4", "t_mean"]
        pr = "DC 결합 probe에서 H4만 평균이 0이 아니고 peak가 비대칭이다. 같은 RMS의 H2는 대칭이다."
    elif tsel == "log":
        _, trl = dab_limit_cycle(Vm, L, fs, ph5, M, 8 * M)
        cyc = [(k + 0.5) * T for k in range(8 * M)]
        res.add_series("t_H5", "H5 포트 평균전류 (주기별)", "A", cyc, [trl.energy(k * T, (k + 1) * T, "p2") / T / v["VL"] for k in range(8 * M)])
        res.add_series("t_H3", "H3 포트 평균전류", "A", cyc, [port["H3"][0]] * (8 * M), dash=True)
        sel = ["t_H5", "t_H3"]
        pr = "장시간 기록에서 H5는 포트 전류가 ±로 교번하고 H3는 일정한 실제 전력을 보낸다. 다른 원인은 0 A로 일정하다."
    else:
        sel = []
        pr = ""
    if sel:
        res.add_plot("p_test", "시험: " + dict(C_TESTS)[tsel], sel, y_label="i", y_unit="A", kind="time" if tsel != "log" else "xy", group="c" if tsel in ("ratio", "lm", "dc") else "", level="C", proved=pr,
                     not_yet="실제 시험에서는 전압비·L_m을 바꾸면 운전점 전체가 바뀌므로 다른 원인의 기여도 함께 움직일 수 있다 — 한 번에 하나만 바꾼다.")
    # --- checks
    tri = abs(VH - V2) / (4 * fs * L) / math.sqrt(3)
    res.add_check(check_close("H1 RMS: 삼각파 닫힌 식 vs 구간선형 정확 적분", target, tri, 1e-12, "|V₁−V₂′|/(4f_sL)/√3 vs PWL Σ Δt(i_a²+i_ai_b+i_b²)/3", True, "A"))
    sysd = DABSystem(VH, V2, L, fs, mod_of_cycle=lambda k: Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1), key="fl12c")
    q0 = Modulation(DPI, DPI, 0.0, SPS_TEXTBOOK_C1).levels(1e-9)
    x0 = half_wave_periodic(sysd, q0)
    tr1 = simulate(sysd, q0, x0, 0.0, T)
    res.add_check(check_close("H1 RMS: 스위칭 엔진(행렬지수) vs 구간선형", tr1.rms(0, T, "iL"), target, 1e-9, "expm 전파·Kronecker 모멘트 vs PWL", True, "A"))
    res.add_check(check_close("H2 여자전류 peak: V₂′T/(4L_m) vs 구간선형", w2.im.max_abs(), Vm * T / (4 * Lm2), 1e-12, "닫힌 식 vs PWL 정점", True, "A"))
    led = energy_ledger(tr5, sys5, 2 * M * T, 4 * M * T, ["p1"], ["p2"], [], rated_power=max(abs(Vm * target), 1.0))
    res.add_check(ledger_check(led, what="H5 limit cycle 창: "))
    # cited runs (FL08)
    ev = Evidence(res)
    i_fl08 = ev.cite("FL08", "zero_power_mismatch", "mismatch", "Irms_L", note="H1의 교재 수치 (900/600 V, φ = 0)")
    ev.cite("FL08", "zero_power_mismatch", "mismatch", "Ipk_L", note="순환전류 peak")
    ev.cite("FL08", "zero_power_mismatch", "matched", "Irms_L", note="전압비를 맞추면 L 순환전류 0 (전압비 일치 시험)")
    ev.cite("FL08", "zero_power_mismatch", "matched", "Im_rms", note="FL08 기본 L_m 2 mH의 여자전류 (H2 세계는 이보다 작은 L_m)")
    if tb:
        res.add_check(check_close("FL08 실행값 = 이 사례의 H1 계산", i_fl08, target, 1e-12, "FL08 zero_power_mismatch@mismatch (별도 모듈) vs 이 사례의 pwl_waves", True, "A"))
    ev.finish()
    res.circuit = {"diagram": dab_circuit(VH, v["VL"], f"{v['Np']}:{v['Ns']}", L, None).to_json(), "intervals": bands, "plot_group": "c"}
    unresolved(res, f"전압비 일치·L_m·φ sweep·DC probe·장시간 기록({groups['ratio']}/{groups['lm']}/{groups['phase']}/{groups['dc']}/{groups['log']}묶음)을 조합해야 다섯 가설이 갈린다.")
    res.verdict("MISSING_INPUT", "실제 n·V_L 범위, 변압기 L_m·누설 배치·포화 자료, gate 지연 차이, 제어 로그가 없어 고객 원인은 정하지 않는다")
    res.assumptions += [
        "이상 full bridge 두 개(SPS), 무손실 직렬 L(1차 환산), 이상 변압기 + 자화 가지",
        "H2·H4: 직렬 L이 1차에 있어 코어 전압은 2차 bridge가 정한다 → 여자전류는 그 bridge 쪽 권선이 공급",
        f"H4: 구간선형 포화 코어 (λ_s = {v['Bs_ratio']:g}·λ_pk, L_sat = L_m/{v['k_core']:g}), DC 자속은 주어진 이력의 결과로 고정",
        f"H5: φ가 {M}주기마다 ±로 뒤집히는 limit cycle을 스위칭 엔진으로 적분 (R = 0이라 전환 때 DC offset이 남는다)",
        "각 세계의 미지수는 같은 권선 RMS를 재현하도록 맞춘 합성 값",
    ]
    res.not_valid_for += ["무부하 손실·온도의 정밀값 (권선·코어 손실 모델 없음)", "제어 루프의 실제 limit cycle 조건", "고객 변압기의 근본원인 확정"]
    res.interpretation = (
        f"전력이 0이라는 것은 평균 포트 에너지 전달이 0이라는 뜻일 뿐이다. 900/600 V의 전압비 불일치는 φ = 0에서도 L에 ±300 V를 걸어 {target:.4g} A RMS를 흘린다(H1). "
        f"같은 RMS는 작은 L_m의 여자전류(H2, L_m {Lm2 * 1e3:.3g} mH), 명령과 다른 실제 φ({ph3:.3g} rad — 이때는 {w3.P2:.4g} W가 실제로 흐른다, H3), DC 자속으로 한쪽 포화된 여자전류(H4), "
        f"±{ph5:.3g} rad를 오가는 limit cycle(H5)로도 만들어진다. 평균 출력전류만 보면 대부분 0 A라 ‘전류센서 오동작’처럼 보이지만 물리적으로 맞는 값이다. "
        "먼저 같은 전압비에서 사라지는지 보고(H1), L_m 측정(H2), φ sweep의 최소점(H3), DC 결합 probe(H4), 장시간 로그(H5)로 가른다. 1쪽 메모 영어 틀: " + MEMO_EN_TEMPLATE
    )
    return res


def params_c() -> list[Param]:
    g_obs, g_c, g_w, g_t = "관측", "회로", "가설 세계의 고정값", "판별 시험"
    return [
        Param("VH", "HV 포트 V_H", "V", 900.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="교재 11장·E12: 900 V corner", group=g_obs),
        Param("VL", "LV 포트 V_L", "V", 36.0, "V", vmin=1, vmax=1000, source="TEXTBOOK", source_note="n·V_L = 600 V", group=g_obs),
        Param("Np", "N_p", "", 50, "", vmin=1, vmax=1000, kind="int", source="TEXTBOOK", source_note="n = 50/3", group=g_c),
        Param("Ns", "N_s", "", 3, "", vmin=1, vmax=1000, kind="int", source="TEXTBOOK", group=g_c),
        Param("L", "직렬 L (1차 환산)", "H", 200e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", source_note="200 µH", group=g_c),
        Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", group=g_c),
        Param("V_match", "H2–H5 세계의 일치된 bridge 전압", "V", 800.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", source_note="nominal 800/800 V", group=g_w),
        Param("Lm_core", "H4 코어의 비포화 L_m", "H", 2e-3, "mH", vmin=1e-5, vmax=1.0, source="ASSUMED", group=g_w),
        Param("Bs_ratio", "H4 포화 자속 / AC 자속 진폭", "", 1.6, "", vmin=1.01, vmax=10, source="ASSUMED", source_note="예: 0.32 T / 0.2 T", group=g_w),
        Param("k_core", "H4 포화 후 L 감소비", "", 20.0, "", vmin=1.5, vmax=1000, source="ASSUMED", group=g_w),
        Param("M_lc", "H5 limit cycle 반주기 (스위칭 주기 수)", "", 5, "", vmin=1, vmax=200, kind="int", source="ASSUMED", group=g_w),
        _test_param(C_TESTS),
    ]


# ======================================================================================
# CASE D - LLC at light load: the low-R_DS(on) device runs hotter (L_m was raised at the same time)
# ======================================================================================

D_TESTS = [
    ("none", "시험 전: 소자·L_m 동시 변경 결과"),
    ("restore_lm", "L_m만 원래로 (새 소자 유지)"),
    ("restore_dev", "소자만 원래로 (새 L_m 유지)"),
    ("vds", "gate-on 직전 V_DS"),
    ("ir", "공진 전류의 edge 값"),
    ("sr", "SR 전류 (역전류)"),
    ("fsw", "실제 f_s·burst 로그"),
]
D_LOADS = (("light", "경부하"), ("mid", "40 %"), ("full", "전부하"))


def llc_regulated(v: dict, Lm: float, P: float):
    """Rectifier-switching LLC (exact engine) regulated to V_o by a secant on the switching frequency.

    The FHA root gives the first guess; every evaluation is a periodic solution (shooting) warm-started
    from the previous one.  Returns (F, system, periodic solution, summary)."""
    from ._resonant import PI as RPI
    from ._resonant import ResonantSystem, Tank, fha, periodic, phasor_guess, summarize

    n = v["Vin"] / v["Vo_nom"]
    RL = v["Vo"] ** 2 / P
    tank = Tank(v["Lr"], v["Cr"], Lm)
    Rac = 8.0 / RPI**2 * n * n * RL
    g_t = v["Vo"] * n / v["Vin"]
    try:
        Fg = brentq(lambda F: fha(tank, F * tank.fr1, Rac).gain - g_t, 1.0001, 4.0)
    except ValueError:
        Fg = 1.1
    state = {"x": None}

    def g(F):
        s = ResonantSystem(tank, v["Vin"], F * tank.fr1, n, "FB", "rc", Co=v["Co"], Rout=RL, key="fl12d")
        guess = phasor_guess(s, Rac) if state["x"] is None else state["x"]
        per = periodic(s, guess, pre_cycles=12 if state["x"] is None else 2, tol=1e-9)
        state["x"] = per.x0
        sm = summarize(s, per.traj, s.T)
        return sm["vo_avg"] - v["Vo"], s, per, sm

    F1, F2 = Fg, 0.97 * Fg
    r1 = g(F1)[0]
    out = g(F2)
    r2 = out[0]
    for _ in range(12):
        if abs(r2) < 1e-5 * v["Vo"] or r2 == r1:
            break
        F3 = F2 - r2 * (F2 - F1) / (r2 - r1)
        F1, r1 = F2, r2
        F2 = F3
        out = g(F2)
        r2 = out[0]
    return F2, out[1], out[2], out[3]


def dev_screen(i_edge: float, Vin: float, td: float, C0: float, V0: float):
    """Charge screen at the bridge's rising edge with EX02's fixed-rail node (SCREEN_ONLY).

    Returns (node voltage at the end of the dead time, residual V_DS, residual hard turn-on energy, node charge)."""
    from .ex02_coss_zvs import Node

    nd = Node(Vin, C0, V0)
    q = abs(i_edge) * td if i_edge < 0 else 0.0
    v_end = nd.v_of_q(q) if q < nd.Qn(Vin) else Vin
    E = nd.hard_on_loss(v_end) if v_end < Vin - 1e-9 else 0.0
    return v_end, Vin - v_end, max(E, 0.0), nd.Qn(Vin)


def vds_dead(i_edge: float, Vin: float, td: float, C0: float, V0: float, t: np.ndarray) -> list[float]:
    """V_DS of the device about to turn on, during the dead time of the charge screen; 0 after its gate-on."""
    from .ex02_coss_zvs import Node

    nd = Node(Vin, C0, V0)
    Qn = nd.Qn(Vin)
    out = []
    for tt in t:
        if tt < 0:
            out.append(Vin)
        elif tt >= td:
            out.append(0.0)
        else:
            q = abs(i_edge) * tt if i_edge < 0 else 0.0
            out.append(Vin - (nd.v_of_q(q) if q < Qn else Vin))
    return out


def run_case_d(v: dict) -> Result:
    from scipy.integrate import quad

    from ..engine.switched import simulate
    from ._common import energy_ledger, ledger_check
    from ._resonant import Tank, bands_from, edge_currents, ivp_period, resonant_circuit, scales_for
    from .ex02_coss_zvs import Node

    res = Result("FL12", "case_d", "C (정류기 포함 스위칭 주기해, V_o 조절) + SCREEN (EX02 비선형 C_oss 전하) + 후처리 손실 추정")
    loads = {"light": v["P_full"] * v["load_light"], "mid": v["P_full"] * 0.4, "full": v["P_full"]}
    Lms = {"old": v["Lm_old"], "h2": v["Lm_h2"], "new": v["Lm_new"]}
    sol = {}
    for lk, Lm in Lms.items():
        for ld, P in loads.items():
            F, s, per, sm = llc_regulated(v, Lm, P)
            ec = edge_currents(s, per.traj, s.T)
            sol[(lk, ld)] = {"F": F, "f": F * s.tank.fr1, "s": s, "per": per, "sm": sm, "i_edge": ec["i_rise"]}
    Ro, Rn, C0o, td, target = v["Ron_old"], v["Ron_new"], v["C0_old"], v["td"], v["dP_obs"]

    def loss(R, C0, lk, ld):
        o = sol[(lk, ld)]
        _, vres, E, Qn = dev_screen(o["i_edge"], v["Vin"], td, C0, v["V0"])
        Pc = R * o["sm"]["I1_rms"] ** 2 / 2.0
        return {"Pc": Pc, "Ps": E * o["f"], "P": Pc + E * o["f"], "vres": vres, "E": E, "Imin": Qn / td}

    def dP(c_old, c_new, lk, ld="light"):
        return loss(Rn, c_new, lk, ld)["P"] - loss(Ro, c_old, "old", ld)["P"]

    def solve_c(fun, lo, hi):
        flo, fhi = fun(lo), fun(hi)
        if flo * fhi > 0:
            return None
        return brentq(fun, lo, hi, xtol=1e-26, rtol=1e-13)

    # the old design completed its transition at the old L_m: largest common C0 that still does (Q_n is linear in C0)
    q_per_F = Node(v["Vin"], 1.0, v["V0"]).Qn(v["Vin"])
    c_zvs_old = abs(sol[("old", "light")]["i_edge"]) * td / q_per_F
    W = {
        "H1": {"lk": "new", "c_old": None, "c_new": None, "what": f"두 소자 C_oss 같음, L_m {Lms['old'] * 1e6:g}→{Lms['new'] * 1e6:g} µH"},
        "H2": {"lk": "h2", "c_old": C0o, "c_new": None, "what": f"L_m 증가 작음 ({Lms['old'] * 1e6:g}→{Lms['h2'] * 1e6:g} µH)"},
        "H12": {"lk": "new", "c_old": C0o, "c_new": None, "what": f"L_m {Lms['old'] * 1e6:g}→{Lms['new'] * 1e6:g} µH"},
        "H5": {"lk": "h2", "c_old": C0o, "c_new": C0o, "dQg": None, "what": f"C_oss 같음, ZVS 유지 (L_m {Lms['h2'] * 1e6:g} µH), Q_g만 증가"},
    }
    c1 = solve_c(lambda c: dP(c, c, "new") - target, 1e-3 * C0o, c_zvs_old)
    W["H1"]["c_old"] = W["H1"]["c_new"] = c1
    W["H2"]["c_new"] = solve_c(lambda c: dP(C0o, c, "h2") - target, C0o, 50 * C0o)
    W["H12"]["c_new"] = solve_c(lambda c: dP(C0o, c, "new") - target, C0o, 50 * C0o)
    # H5: the gate charge alone (ZVS kept), all of Q_g V_drv f taken as die loss (upper bound for this hypothesis)
    W["H5"]["dQg"] = (target - dP(C0o, C0o, "h2")) / (v["Vdrv"] * sol[("h2", "light")]["f"])

    def wP(w, dev, lk, ld):
        new = dev == "new"
        P = loss(Rn if new else Ro, w["c_new"] if new else w["c_old"], lk, ld)["P"]
        return P + (w.get("dQg") or 0.0) * v["Vdrv"] * sol[(lk, ld)]["f"] if new else P

    for hk, w in W.items():
        w["ok"] = w["c_new"] is not None and w["c_old"] is not None and (hk != "H5" or w["dQg"] > 0)
        if not w["ok"]:
            continue
        lk = w["lk"]
        a_oo, a_no, a_on, a_nn = wP(w, "old", "old", "light"), wP(w, "new", "old", "light"), wP(w, "old", lk, "light"), wP(w, "new", lk, "light")
        w.update(obs=loss(Rn, w["c_new"], lk, "light"), dP_obs=a_nn - a_oo, dP_rlm=a_no - a_oo, dP_rdev=a_on - a_oo,
                 dP_full=wP(w, "new", lk, "full") - wP(w, "old", "old", "full"),
                 main_dev=0.5 * ((a_no - a_oo) + (a_nn - a_on)), main_lm=0.5 * ((a_on - a_oo) + (a_nn - a_no)), inter=0.5 * ((a_nn - a_no) - (a_on - a_oo)),
                 curve={ld: wP(w, "new", lk, ld) for ld, _ in D_LOADS}, curve_rlm={ld: wP(w, "new", "old", ld) for ld, _ in D_LOADS},
                 curve_rdev={ld: wP(w, "old", lk, ld) for ld, _ in D_LOADS})
    base_curve = {ld: loss(Ro, C0o, "old", ld)["P"] for ld, _ in D_LOADS}
    Qg_req = v["Qg_old"] + (W["H5"]["dQg"] or 0.0)
    # metrics
    for hk, lab_ in (("H1", "H1 세계: 두 소자 공통 C₀"), ("H2", "H2 세계: 새 소자 C₀"), ("H12", "H12 세계: 새 소자 C₀")):
        w = W[hk]
        res.add_metric(f"cal_{hk}", f"{lab_} (관측 ΔP {target:g} W 재현)", w["c_new"] if w["ok"] else float("nan"), "F",
                       basis=w["what"] + (f" → 2Q_oss/t_d {w['obs']['Imin']:.3g} A" if w["ok"] else " → 이 입력에서 재현 불가"))
    res.add_metric("Qg_req", "H5: 관측을 gate 전하만으로 설명하려면 필요한 새 소자 Q_g", Qg_req, "C", basis=f"상한 가정: Q_g·V_drv·f 전부가 die에서 소산 (기존 {v['Qg_old'] * 1e9:g} nC)")
    for lk in Lms:
        o = sol[(lk, "light")]
        res.add_metric(f"edge_{lk}", f"경부하 상승 edge 전류 (L_m {Lms[lk] * 1e6:g} µH)", o["i_edge"], "A", basis=f"f = {o['f'] / 1e3:.4g} kHz (V_o {v['Vo']:g} V 조절), 음수 = ZVS 방향")
        res.add_metric(f"f_{lk}", f"경부하 스위칭 주파수 (L_m {Lms[lk] * 1e6:g} µH)", o["f"], "Hz")
    m_imin = res.add_metric("Imin_old", "기존 소자의 전하 screen 최소 전류 2Q_oss/t_d", 2.0 * Node(v["Vin"], C0o, v["V0"]).Qd(v["Vin"]) / td, "A", basis=f"C₀ {C0o * 1e9:g} nF, t_d {td * 1e9:g} ns (SCREEN_ONLY)")
    for hk, w in W.items():
        if w["ok"]:
            res.add_metric(f"inter_{hk}", f"{hk} 세계의 상호작용 (소자 × L_m, 경부하)", w["inter"], "W", basis=f"주효과 소자 {w['main_dev']:+.3g} W, L_m {w['main_lm']:+.3g} W")
            res.add_metric(f"dPf_{hk}", f"{hk} 세계의 전부하 손실 변화", w["dP_full"], "W", note="음수 = 개선 (관측과 일치), 양수 = 관측과 모순")
    if W["H12"]["ok"]:
        res.add_metric("vres_H12", "H12 세계: turn-on 직전 잔류 V_DS (경부하)", W["H12"]["obs"]["vres"], "V", basis="정전류 charge screen (EX02 고정 rail)")
    # world table (2x2 per world)
    rows = []
    for hk, w in W.items():
        if not w["ok"]:
            rows.append([hk, w["what"], "재현 불가", "—", "—", "—", "—", "—", "—"])
            continue
        extra = f"Q_g {Qg_req * 1e9:.0f} nC" if hk == "H5" else (f"C₀ {w['c_old'] * 1e9:.3g} / {w['c_new'] * 1e9:.3g} nF")
        rows.append([hk, w["what"] + f" ({extra})", f"{w['dP_obs']:+.3g}", f"{w['dP_rlm']:+.3g}", f"{w['dP_rdev']:+.3g}",
                     f"{w['main_dev']:+.3g}", f"{w['main_lm']:+.3g}", f"{w['inter']:+.3g}", f"{w['dP_full']:+.3g}"])
    res.tables.append(Table("t_world", "세계별 2×2 결과: 경부하 소자당 손실 변화 [W] (기준 = 기존 소자·기존 L_m, 후처리 추정)",
                            ["세계", "가정 (C₀ 기존 / 새)", "관측 조합 (새 소자 + 새 L_m)", "L_m만 되돌림", "소자만 되돌림", "주효과: 소자", "주효과: L_m", "상호작용", "전부하 변화"], rows,
                            note="모든 세계가 경부하 +ΔP 관측을 재현하도록 미지수 하나를 맞췄다. 고객은 대각선(기존·기존 → 새·새)만 봤다. 한 번에 하나만 되돌리는 두 칸이 세계를 가르고, 전부하 변화가 양수인 세계는 이미 관측과 모순이다."))
    hyps = [
        Hyp("H1", "L_m 증가 → edge 전류 감소 → 부분 ZVS", "두 소자의 C_oss가 같고 L_m 증가만으로 전환을 못 끝낸다", "공통 C₀"),
        Hyp("H2", "C_oss/Q_oss 증가 (새 소자)", "L_m 증가는 작고 새 소자의 전하가 커서 기존 L_m에서도 전환을 못 끝낸다", "새 C₀"),
        Hyp("H12", "H1 × H2 상호작용", "둘 중 하나만으로는 전하가 충분하고, 둘이 겹치면 부족하다", "새 C₀"),
        Hyp("H3", "SR 역전류", "경부하 SR 타이밍이 맞지 않아 역전류·손실 (이 모델은 다이오드 정류)", "모델 밖"),
        Hyp("H4", "burst 조건 변화", "L_m 변경으로 gain 곡선이 바뀌어 burst 진입·첫 펄스가 바뀜 (모델 없음)", "모델 밖"),
        Hyp("H5", "gate 전하 손실 (큰 die)", "Q_g·V_drv·f 증가만으로 설명 (ZVS 유지)", "Q_g"),
    ]

    def cls_restore(dp):
        r = dp / target
        return "해결" if r < 0.3 else ("그대로" if r > 0.7 else "일부 남음")

    e_old = abs(sol[("old", "light")]["i_edge"])
    cl = {}
    for hk in ("H1", "H2", "H12", "H5"):
        w = W[hk]
        if not w["ok"]:
            cl[hk] = {t: ("이 입력에서 관측 재현 불가", "재현 불가") for t, _ in D_TESTS}
            continue
        o = sol[(w["lk"], "light")]
        im = w["obs"]["Imin"]
        none = (f"경부하 {w['dP_obs']:+.3g} W, 전부하 {w['dP_full']:+.3g} W", "관측 재현" if w["dP_full"] < 0 else "전부하 악화 예측 (모순)")
        if hk == "H5":
            none = (f"Q_g {Qg_req * 1e9:.0f} nC 필요, 전부하 {w['dP_full']:+.3g} W", none[1])
        cl[hk] = {
            "none": none,
            "restore_lm": (f"{w['dP_rlm']:+.3g} W → {cls_restore(w['dP_rlm'])}", cls_restore(w["dP_rlm"])),
            "restore_dev": (f"{w['dP_rdev']:+.3g} W → {cls_restore(w['dP_rdev'])}", cls_restore(w["dP_rdev"])),
            "vds": (f"잔류 {w['obs']['vres']:.3g} V", "잔류 있음" if w["obs"]["vres"] > 1.0 else "잔류 ≈ 0"),
            "ir": (f"edge {o['i_edge']:.3g} A {'<' if abs(o['i_edge']) < im else '≥'} 필요 {im:.3g} A", "edge 크게 감소" if abs(o["i_edge"]) / e_old < 0.75 else "edge 거의 유지"),
            "sr": ("SR 역전류 없음 (다이오드 정류)", "역전류 없음"),
            "fsw": (f"연속, f {o['f'] / 1e3:.4g} kHz", "연속"),
        }
    cl["H3"] = {"none": ("모델 밖 (MISSING_INPUT)", "모델 밖"), "restore_lm": ("모델 밖: f_s·SR timing이 함께 바뀜", "모델 밖"), "restore_dev": ("모델 밖: 1차 소자와 무관할 수 있음", "모델 밖"),
                "vds": ("1차 ZVS와 무관: 잔류 0일 수 있음", "잔류 ≈ 0"), "ir": ("역전류가 edge 전류를 바꿈 (모델 밖)", "모델 밖"), "sr": ("음의 SR 전류가 보인다", "역전류 보임"), "fsw": ("연속", "연속")}
    cl["H4"] = {"none": ("모델 밖 (MISSING_INPUT)", "모델 밖"), "restore_lm": ("모델 밖: burst 문턱이 gain과 함께 바뀜", "모델 밖"), "restore_dev": ("모델 밖", "모델 밖"),
                "vds": ("burst 첫 펄스만 잔류", "첫 펄스만"), "ir": ("burst 첫 펄스의 edge 전류가 작다", "첫 펄스만"), "sr": ("SR 역전류 없음", "역전류 없음"), "fsw": ("burst 진입·패턴 변화", "burst")}
    groups = disc_tables(res, hyps, D_TESTS, cl, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설의 세계가 예측하는 결과",
                         note="H1·H2·H12는 같은 관측(경부하 ΔP·전부하 개선)을 재현하도록 미지수를 맞춘 정류기 포함 주기해 + 전하 screen이다. H3(SR)·H4(burst)는 이 모델 밖이라 그 가설을 가르는 측정만 적었고, "
                         "H5는 크기 검토(필요한 Q_g, 전부하 예측)로 다뤘다. 한 번에 하나씩 되돌리는 두 시험의 조합이 H1·H2·H12를 가른다.")
    # plots ------------------------------------------------------------------------------
    for lk in Lms:
        o = sol[(lk, "light")]
        tr = simulate(o["s"], o["per"].q0, o["per"].x0, 0.0, 2 * o["s"].T)
        smp = tr.sample(["i1"], per_segment=40)
        res.add_series(f"ir_{lk}", f"i_r (L_m {Lms[lk] * 1e6:g} µH, {o['f'] / 1e3:.4g} kHz)", "A", smp["t"], [o["s"].n_in * y for y in smp["i1"]])
        o["tr"] = tr
    req = [("기존 소자", m_imin.value)] + [(f"{hk} 새 소자", W[hk]["obs"]["Imin"]) for hk in ("H1", "H2", "H12") if W[hk]["ok"]]
    merged: list[list] = []
    for name, val in req:  # equal requirements (H1 and H12 share the calibrated C0) get one line and one label
        for m_ in merged:
            if abs(m_[1] - val) <= 0.02 * abs(val):
                m_[0] = m_[0].replace(" 새 소자", "") + "·" + name
                break
        else:
            merged.append([name, val])
    hl = [{"y": -val, "label": f"{name} −{val:.3g} A"} for name, val in merged]
    bands = bands_from(sol[("new", "light")]["tr"], 0.0, 2 * sol[("new", "light")]["s"].T)
    res.add_plot("p_ir", f"{v['load_light'] * 100:g} % 부하의 공진 전류와 세계별 필요 전류 (t = 0, T/2가 상승·하강 edge)", ["ir_old", "ir_h2", "ir_new"], y_label="i_r", y_unit="A", bands=bands, group="d", level="C",
                 hlines=hl + [{"y": 0.0, "label": ""}],
                 proved="V_o를 같은 값으로 조절한 정류기 포함 주기해에서 L_m을 키울수록 경부하 edge 전류가 줄어든다는 것과, 세 세계의 새 소자가 요구하는 전류(점선, 2Q_oss/t_d)가 모두 그 edge 전류보다 크다는 것을 보였다 "
                 "(독립 DOP853 적분·에너지 원장 check).",
                 not_yet="이상 스위치·다이오드 정류 모델이다. dead time 동안 전류가 변하는 실제 commutation(EX02 event)·SR 타이밍·burst는 계산하지 않았다. 점선의 필요 전류는 각 세계에서 맞춘 합성 C_oss이고 고객 소자 값이 아니다.")
    tds = np.linspace(20e-9, 300e-9, 57)
    vr_keys = []
    for hk in ("H1", "H2", "H12"):
        w = W[hk]
        if not w["ok"]:
            continue
        ys = [dev_screen(sol[(w["lk"], "light")]["i_edge"], v["Vin"], float(t_), w["c_new"], v["V0"])[1] for t_ in tds]
        res.add_series(f"vr_{hk}", f"{hk} 세계 (새 소자 + L_m {Lms[w['lk']] * 1e6:g} µH)", "V", tds.tolist(), ys)
        vr_keys.append(f"vr_{hk}")
    ys = [dev_screen(sol[("old", "light")]["i_edge"], v["Vin"], float(t_), C0o, v["V0"])[1] for t_ in tds]
    res.add_series("vr_base", f"기존 설계 (기존 소자 + L_m {Lms['old'] * 1e6:g} µH)", "V", tds.tolist(), ys, dash=True)
    res.add_plot("p_vres", f"{v['load_light'] * 100:g} % 부하 turn-on 직전 잔류 V_DS vs dead time: 세 세계 모두 현재 dead time에서 잔류가 남는다", vr_keys + ["vr_base"], x_label="dead time t_d", x_unit="s", y_label="잔류 V_DS", y_unit="V", kind="xy", level="SCREEN",
                 vlines=[{"x": td, "label": f"현재 {td * 1e9:g} ns"}],
                 proved="같은 관측을 재현하는 세 세계 모두 현재 dead time에서 잔류 V_DS가 남고 기존 설계는 0이라는 것을 보였다(H1과 H12는 새 소자 C₀와 L_m이 같아 곡선이 겹친다). 그래서 gate-on 직전 V_DS 측정은 ‘ZVS 손실이냐 아니냐’(H1·H2·H12 대 H3·H5)는 가르지만 H1·H2·H12 사이는 가르지 못한다. "
                 "곡선의 잔류 전압은 EX02 닫힌 식 v(q)와 독립 ODE dv/dt = I/C(v)의 일치로 확인했다.",
                 not_yet="edge 전류를 dead time 동안 일정하게 둔 charge screen(SCREEN_ONLY)이다. dead time을 늘려 잔류가 사라지는지는 실제 소자의 C_oss(V)와 전류 변화를 넣은 EX02 commutation event로 다시 확인해야 하고, 고객 회로의 원인을 정하지 않는다.")
    xs_load = [loads[ld] for ld, _ in D_LOADS]
    res.add_series("pl_base", f"기존 설계 (기존 소자 + L_m {Lms['old'] * 1e6:g} µH)", "W", xs_load, [base_curve[ld] for ld, _ in D_LOADS], dash=True)
    pl_keys = []
    for hk in ("H1", "H2", "H12"):
        w = W[hk]
        if w["ok"]:
            res.add_series(f"pl_{hk}", f"{hk} 세계: 새 소자 + L_m {Lms[w['lk']] * 1e6:g} µH", "W", xs_load, [w["curve"][ld] for ld, _ in D_LOADS])
            pl_keys.append(f"pl_{hk}")
    res.add_plot("p_loss", "소자당 손실(전도 + 잔류 turn-on) vs 부하: 세 세계가 같은 관측(전부하 개선·경부하 악화)을 만든다", pl_keys + ["pl_base"], x_label="출력", x_unit="W", y_label="소자당 손실", y_unit="W", kind="xy", log_x=True, level="C+SCREEN",
                 proved="세 세계의 ‘새 소자 + 새 L_m’ 곡선이 경부하에서 같은 점(+ΔP 관측)을 지나고 전부하에서는 모두 기존 설계보다 낮다는 것을 보였다 — 고객이 본 두 숫자만으로는 세계가 갈리지 않는다 (V_o 조절 check, 보정 잔차 check).",
                 not_yet="손실은 이상 파형에 후처리한 추정(R_DS(on) 온도 의존·turn-off 손실·코어 손실 미포함)이고 세 부하점만 계산했다. 어느 세계가 고객 회로인지는 정하지 않는다.")
    tsel = v["test"]
    if tsel in ("restore_lm", "restore_dev"):
        key = "curve_rlm" if tsel == "restore_lm" else "curve_rdev"
        tk = []
        for hk in ("H1", "H2", "H12"):
            w = W[hk]
            if not w["ok"]:
                continue
            lab_ = f"{hk}: 새 소자 + L_m {Lms['old'] * 1e6:g} µH" if tsel == "restore_lm" else f"{hk}: 기존 소자 + L_m {Lms[w['lk']] * 1e6:g} µH"
            res.add_series(f"t_{hk}", lab_, "W", xs_load, [w[key][ld] for ld, _ in D_LOADS])
            tk.append(f"t_{hk}")
        res.add_plot("p_test", "시험: " + dict(D_TESTS)[tsel] + " — 세계마다 경부하 손실이 다르게 반응한다", tk + ["pl_base"], x_label="출력", x_unit="W", y_label="소자당 손실", y_unit="W", kind="xy", log_x=True, level="C+SCREEN",
                     proved=("L_m만 되돌리면 H1·H12 세계는 기존 설계 곡선으로 돌아오고 H2 세계만 경부하 손실이 남는다는 것을 보였다. 그래서 ‘L_m을 되돌렸더니 해결’이라는 결과만으로는 H1과 H12를 가르지 못한다."
                             if tsel == "restore_lm" else
                             "소자만 되돌리면 H2·H12 세계는 해결되고 H1 세계만 경부하 손실이 남는다는 것을 보였다. 두 시험을 함께 하면 H1(L_m만 되돌려 해결), H2(소자만 되돌려 해결), H12(어느 쪽이든 해결)가 갈린다."),
                     not_yet="실제 시험에서는 소자 교체가 gate 회로·열 경계도 함께 바꿀 수 있고 L_m 변경은 f_s·burst 조건도 바꾼다 — 바꾼 것을 모두 기록한다. SR·burst 가설(H3·H4)은 이 그래프에 없다.")
    elif tsel == "vds":
        tt = np.concatenate([np.linspace(-0.25 * td, td * (1 - 1e-6), 160), [td, 1.4 * td]])
        tk = []
        for hk in ("H1", "H2", "H12"):
            w = W[hk]
            if w["ok"]:
                res.add_series(f"t_{hk}", f"{hk} 세계", "V", tt.tolist(), vds_dead(sol[(w["lk"], "light")]["i_edge"], v["Vin"], td, w["c_new"], v["V0"], tt))
                tk.append(f"t_{hk}")
        res.add_series("t_base", "기존 설계 (H3·H5 세계도 ZVS 유지)", "V", tt.tolist(), vds_dead(sol[("old", "light")]["i_edge"], v["Vin"], td, C0o, v["V0"], tt), dash=True)
        res.add_plot("p_test", "시험: gate-on 직전 V_DS — dead time 동안 들어올 소자의 V_DS (t = t_d에서 gate-on)", tk + ["t_base"], y_label="V_DS", y_unit="V", level="SCREEN",
                     vlines=[{"x": td, "label": "gate-on"}],
                     proved="H1·H2·H12 세계에서는 gate-on 순간 V_DS가 남아(hard turn-on) 수직으로 떨어지고, ZVS가 유지되는 세계(기존 설계, H3·H5)는 그 전에 0에 닿는다는 것을 보였다. 이 측정은 ZVS 손실 가설군과 나머지를 가른다.",
                     not_yet="정전류 charge screen의 V_DS 궤적이다. 실제 파형의 기울기·ringing·probe 지연은 다르고, H1·H2·H12 사이는 이 측정만으로 갈리지 않는다.")
    # checks -----------------------------------------------------------------------------
    o = sol[("new", "light")]
    xT = ivp_period(o["s"], o["per"].x0)
    dev_err = float(np.max(np.abs(xT - o["per"].x0) / scales_for(o["s"])))
    res.add_check(Check("독립 경로: 상태식 + DOP853 한 주기 (경부하, 새 L_m)", "PASS" if dev_err < 1e-7 else "FAIL", dev_err, "rel", 1e-7, path="affine 행렬·행렬지수 대신 상태식을 직접 적고 solve_ivp 이벤트로 다이오드 전환", independent=True))
    led = energy_ledger(o["tr"], o["s"], 0.0, o["s"].T, ["p_in"], ["p_load"], ["p_R"], rated_power=max(loads["light"], 1.0))
    res.add_check(ledger_check(led, what="경부하·새 L_m 1주기: "))
    vo_err = max(abs(s_["sm"]["vo_avg"] / v["Vo"] - 1) for s_ in sol.values())
    res.add_check(Check("V_o 조절 (모든 운전점)", "PASS" if vo_err < 1e-4 else "FAIL", vo_err, "rel", 1e-4, path="secant로 f_s를 맞춘 주기해의 출력 평균", independent=False))
    cal_err = max((abs(w["dP_obs"] - target) / max(abs(target), 1e-9) for w in W.values() if w["ok"]), default=0.0)
    res.add_check(Check("세 세계가 같은 관측(경부하 ΔP)을 재현", "PASS" if cal_err < 1e-6 else "FAIL", cal_err, "rel", 1e-6, path="brentq로 각 세계의 C₀를 맞춘 뒤 손실을 다시 계산", independent=False,
                        detail="재현 불가 세계: " + (", ".join(k for k, w in W.items() if not w["ok"]) or "없음")))
    for lab_, C0 in (("기존 소자", C0o), ("H12 새 소자", W["H12"]["c_new"])):
        if C0 is None:
            continue
        qn, _ = quad(lambda x, C0=C0: C0 / math.sqrt(1 + x / v["V0"]), 0, v["Vin"], epsabs=0, epsrel=1e-13)
        res.add_check(check_close(f"Q_oss({v['Vin']:g} V) {lab_}: EX02 닫힌 식 vs 수치 적분", Node(v["Vin"], C0, v["V0"]).Qd(v["Vin"]), qn, 1e-10, "2C₀V₀(√(1+V/V₀)−1) vs quad ∫C(v)dv", True, "C"))
    if W["H12"]["ok"]:
        w = W["H12"]
        nd = Node(v["Vin"], w["c_new"], v["V0"])
        i0 = abs(sol[(w["lk"], "light")]["i_edge"])
        ode = solve_ivp(lambda t, y: [i0 / nd.Cn(min(max(y[0], 0.0), v["Vin"]))], (0.0, td), [0.0], method="DOP853", rtol=1e-12, atol=1e-9)
        res.add_check(check_close("H12 잔류 V_DS: EX02 v(q) 닫힌 식 vs ODE dv/dt = I/C_n(v)", w["obs"]["vres"], v["Vin"] - float(ode.y[0][-1]), 1e-6, "brentq로 Q_n(v) = I·t_d를 푼 값 vs DOP853 적분", True, "V"))
    ev = Evidence(res)
    fr = ev.cite("FL09", "fha_gain", "textbook", "fr", note="같은 합성 tank의 공진주파수 (교재 150 kHz)")
    ev.cite("FL09", "fha_gain", "textbook", "k", note="k = L_m/L_r = 5 (기존 L_m 200 µH)")
    ev.cite("FL09", "fha_gain", "textbook", "n", note="400→48 V FB 권선비")
    res.add_check(check_close("FL09 실행의 f_r = 이 사례 tank의 f_r", fr, Tank(v["Lr"], v["Cr"], v["Lm_old"]).fr1, 1e-12, "FL09 fha_gain@textbook vs 이 사례의 Tank", True, "Hz"))
    ev.finish()
    res.circuit = {"diagram": resonant_circuit("LLC", v["Vin"], f"{v['Vin'] / v['Vo_nom']:.4g}", Tank(v["Lr"], v["Cr"], v["Lm_new"]), "FB").to_json(), "intervals": bands, "plot_group": "d"}
    unresolved(res, f"소자와 L_m을 한 번에 하나씩 되돌리는 두 시험({groups['restore_lm']}·{groups['restore_dev']}묶음)과 V_DS·SR 전류·f_s 로그가 필요하다.")
    res.verdict("SCREEN_ONLY", "ZVS는 edge 전류 × dead time의 전하 screen(합성 C_oss)으로만 판정했다 — 실제 turn-on 직전 V_DS가 아니다")
    res.verdict("MISSING_INPUT", "SR 타이밍·burst 조건(H3·H4)은 모델 밖이고 실제 소자 Q_oss(V)·R_DS(on)(T)·Q_g가 없다")
    res.assumptions += [
        f"FL09 합성 tank (L_r {v['Lr'] * 1e6:g} µH, C_r {v['Cr'] * 1e9:g} nF), 기존 L_m {v['Lm_old'] * 1e6:g} µH; 늘린 L_m은 H1·H12 {v['Lm_new'] * 1e6:g} µH, H2 {v['Lm_h2'] * 1e6:g} µH (폭은 모름: ASSUMED)",
        f"V_in {v['Vin']:g} V, V_o {v['Vo']:g} V로 조절 (f_r보다 위에서 운전하도록 ASSUMED), 저항부하, 이상 다이오드 정류",
        f"관측 = 경부하 소자당 손실 +{target:g} W (온도 상승을 손실로 환산했다고 가정), 전부하는 개선",
        "새 소자: R_DS(on) ↓ (합성), C_oss는 세계마다 관측을 재현하도록 보정 — 고객 소자 값이 아니다",
        "ZVS: 상승 edge 전류를 dead time 동안 일정하게 둔 charge screen + EX02 고정 rail 잔류 turn-on 에너지",
        "손실: R_DS(on)·I_rms²/2 + E_res·f (후처리 추정); H5는 Q_g·V_drv·f 전부가 die에서 소산된다는 상한",
    ]
    res.not_valid_for += ["실제 소자 온도·효율", "SR·burst 동작 (모델 없음)", "turn-off 손실·코어 손실", "고객 회로의 근본원인 확정"]
    w12 = W["H12"]
    res.interpretation = (
        f"전부하에서는 새 소자의 낮은 R_DS(on)이 전도손실을 줄여 개선된다. 경부하에서는 ZVS 전하를 주는 것이 여자전류인데 L_m을 키우면 그 전류가 준다(edge {sol[('old', 'light')]['i_edge']:.3g} → {sol[('new', 'light')]['i_edge']:.3g} A). "
        f"같은 관측(경부하 +{target:g} W, 전부하 개선)을 세 세계가 모두 재현한다: 두 소자의 C_oss가 같고 L_m만 원인인 세계(H1), L_m 증가는 작고 새 소자의 C_oss가 원인인 세계(H2), 둘이 겹쳐야만 부족해지는 상호작용 세계(H12"
        + (f", 상호작용 {w12['inter']:+.3g} W" if w12["ok"] else "")
        + "). 동시에 바꾼 결과만으로는 ‘L_m 때문’이나 ‘소자 때문’이라고 말할 수 없고, L_m만 되돌리는 시험과 소자만 되돌리는 시험을 둘 다 해야 갈린다. "
        + (f"gate 전하만으로 설명하려면 Q_g가 {Qg_req * 1e9:.0f} nC는 되어야 하고(모두 die 손실이라는 상한) 전부하 변화가 {W['H5']['dP_full']:+.3g} W로 예측된다"
           + (" — 전부하 개선이라는 관측과 맞지 않는다. " if W["H5"]["ok"] and W["H5"]["dP_full"] > 0 else ". ") if W["H5"]["ok"] else "")
        + "SR 역전류·burst 조건 변화는 모델 밖이므로 SR 전류와 실제 f_s 로그로 따로 확인한다. 1쪽 메모 영어 틀: "
        + MEMO_EN_TEMPLATE
    )
    return res


def params_d() -> list[Param]:
    g_t, g_o, g_d = "tank (FL09)", "운전점·관측", "소자 (합성)"
    return [
        Param("Lr", "공진 L_r", "H", 40e-6, "µH", vmin=1e-7, vmax=1e-2, source="TEXTBOOK", source_note="FL09 합성 tank", group=g_t),
        Param("Cr", "공진 C_r", "F", 28.1448e-9, "nF", vmin=1e-11, vmax=1e-4, source="TEXTBOOK", source_note="f_r = 150 kHz", group=g_t),
        Param("Lm_old", "기존 L_m", "H", 200e-6, "µH", vmin=1e-6, vmax=1.0, source="TEXTBOOK", source_note="FL09 k = 5", group=g_t),
        Param("Lm_new", "늘린 L_m (H1·H12 세계)", "H", 400e-6, "µH", vmin=1e-6, vmax=1.0, source="ASSUMED", source_note="교재: L_m도 동시에 늘렸다 (폭은 모름)", group=g_t),
        Param("Lm_h2", "늘린 L_m (H2 세계: 작은 증가)", "H", 240e-6, "µH", vmin=1e-6, vmax=1.0, source="ASSUMED", source_note="L_m 증가가 ZVS 여유를 거의 깎지 않는 세계", group=g_t),
        Param("Vin", "입력 V_in", "V", 400.0, "V", vmin=10, vmax=2000, source="TEXTBOOK", group=g_o),
        Param("Vo_nom", "f_r에서 맞춘 V_o (권선비 기준)", "V", 48.0, "V", vmin=1, vmax=2000, source="TEXTBOOK", source_note="n = 8.3333 (FB)", group=g_o),
        Param("Vo", "조절 출력 V_o", "V", 45.0, "V", vmin=1, vmax=2000, source="ASSUMED", source_note="f_r 위 운전점", group=g_o),
        Param("P_full", "전부하 출력", "W", 2500.0, "W", vmin=10, vmax=1e5, source="ASSUMED", group=g_o),
        Param("load_light", "경부하 비율", "", 0.1, "", vmin=0.01, vmax=0.39, source="TEXTBOOK", source_note="CASE D: 10 % load", group=g_o),
        Param("dP_obs", "관측: 경부하 소자당 손실 증가", "W", 1.5, "W", vmin=0.01, vmax=100, source="ASSUMED", source_note="온도 상승을 R_th로 환산했다고 가정 (교재: ‘더 뜨겁다’)", group=g_o),
        Param("Co", "출력 C_o", "F", 200e-6, "µF", vmin=1e-7, vmax=1.0, source="ASSUMED", group=g_o),
        Param("td", "dead time", "s", 100e-9, "ns", vmin=1e-9, vmax=5e-6, source="ASSUMED", source_note="FL09 screen과 같은 값", group=g_d),
        Param("V0", "합성 C_oss의 V₀", "V", 40.0, "V", vmin=0.1, vmax=1e4, source="ASSUMED", group=g_d),
        Param("C0_old", "기존 소자 C₀ (H2·H12 세계)", "F", 0.5e-9, "nF", vmin=1e-12, vmax=1e-6, source="ASSUMED", source_note="FL09 합성 C_oss", group=g_d),
        Param("Ron_old", "기존 R_DS(on)", "Ω", 40e-3, "mΩ", vmin=1e-4, vmax=10, source="ASSUMED", group=g_d),
        Param("Ron_new", "새 R_DS(on)", "Ω", 25e-3, "mΩ", vmin=1e-4, vmax=10, source="ASSUMED", group=g_d),
        Param("Qg_old", "기존 Q_g", "C", 60e-9, "nC", vmin=1e-10, vmax=1e-5, source="ASSUMED", group=g_d),
        Param("Vdrv", "gate 전압 swing", "V", 20.0, "V", vmin=1, vmax=40, source="ASSUMED", group=g_d),
        _test_param(D_TESTS),
    ]


EXP_A = Experiment(
    key="case_a",
    title="CASE A — SiC 교체 후 false turn-on 의심: 같은 +5 V를 만드는 네 가지 세계",
    goal=(
        "증상: SiC로 교체한 뒤 꺼져 있어야 할 소자(Q_L)의 gate에 +5 V spike가 보이고 DC-link 전류 peak가 커졌다. 확인된 것: 고객 probe는 power source 핀 기준이고 +5 V를 보였다. "
        "모르는 것: 실제 threshold(온도), die 내부 gate 전압, probe 접지 배치, 실제 dead time. "
        "먼저 증상만 읽고 아래 ‘면접 질문’ 카드의 Q1–Q3에 가설 3개·받을 파형 3개·첫 실험 1개를 적은 뒤 실행한다. 실행하면 Miller coupling·common-source inductance·probe artifact·dead-time overlap "
        "네 가설을 각각 같은 +5 V를 재현하도록 맞춘 세계로 만들고, Kelvin 재측정·R_g,off 변경·동시 취득·probe 단락·turn-on 감속에서 각 가설이 무엇을 예측하는지 판별표로 비교한다."
    ),
    params=params_a(),
    presets=[
        Preset("observed", "관측 그대로 (시험 전)", {}, "CASE A 증상", ("nominal", "reference")),
        Preset("kelvin", "첫 시험: Kelvin 기준 재측정", {"test": "kelvin"}, "H2·H3 제거", ("reference",)),
        Preset("rg_off", "R_g,off 10 → 2.5 Ω", {"test": "rg_off"}, "H1·H4가 반응", ("variant", "reference")),
        Preset("sync", "V_DS·전류·gate 명령 동시 취득", {"test": "sync"}, "spike의 구간", ("variant",)),
        Preset("short_probe", "probe tip 단락 시험", {"test": "short_probe"}, "H3 확인", ("variant",)),
        Preset("slow_on", "임시조치: R_g,on 2.5 → 10 Ω", {"test": "slow_on"}, "판별력 없음 + 손실 비용", ("variant",)),
        Preset("hot_vth", "고온 V_th 1.8 V", {"Vth": 1.8}, "V_th를 모르면 여유도 모른다", ("corner",)),
    ],
    run=run_case_a,
    model_level="A/D 합성 screen (+ FL02 D 수준 cell 인용)",
    suggested_change="판별 시험을 ‘Kelvin 기준 재측정’으로 바꾼다 (다음에 ‘R_g,off 감소’, ‘동시 취득’).",
    prediction=Prediction(
        "Kelvin 기준·짧은 접지로 다시 재면 네 가설의 +5 V는 각각 어떻게 되나?",
        ["모든 가설에서 그대로 +5 V", "H2(공통 source L)·H3(probe pickup)에서만 사라진다", "H1(Miller)에서만 사라진다", "모르겠다"],
        "H2(공통 source L)·H3(probe pickup)에서만 사라진다",
        "H2의 L_s·di/dt와 H3의 M·di/dt는 측정 경로가 만든 전압이라 기준점과 접지 loop를 고치면 빠진다. H1(C_gd·dv/dt)과 H4(dead-time 겹침)의 gate 전압은 실제 die 전압이라 남는다. "
        "남은 둘은 R_g,off/clamp 반응과 동시 취득의 timing으로 가른다.",
        ["die_H1", "die_H2", "Qst_H1", "Qst_H2"],
        handcalc=[{"key": "t_fv", "label": "Miller 시간 Q_gd/I_g", "unit": "s"}, {"key": "Ls_screen", "label": "L_s·di/dt (2 nH, 2 kA/µs)", "unit": "V"}, {"key": "icap", "label": "C_oss·dv/dt", "unit": "A"}],
    ),
    suggested={"test": "kelvin"},
    student=(
        "꺼진 소자의 gate 전압이 올라 보이는 이유는 넷이다. ① 상대 소자가 켜지며 이 소자의 drain 전압이 빠르게 오르면 drain–gate 용량으로 전류가 밀려들어 gate가 실제로 올라간다(Miller). "
        "② probe 접지를 power source 핀에 대면 source lead 인덕턴스에 걸린 L·di/dt가 gate 전압처럼 더해진다. ③ 긴 접지 lead loop가 주변 전류 변화를 주워 담는다. "
        "④ dead time이 실제로 짧으면 상대 소자가 켜질 때 이 소자의 gate가 아직 덜 꺼져 있다. ①과 ④만 실제로 채널을 연다."
    ),
    expert=(
        "교재 해설(CASE A): Miller coupling, common-source inductance, measurement artifact, 실제 dead-time overlap을 모두 유지한다. VGS를 Kelvin 기준으로 재측정하고 VDS·leg current·gate command를 동시 취득한다. "
        "R_g,off/clamp만 바꿨을 때 spike와 실제 current가 함께 바뀌는지 본다. turn-on 속도만 늦추는 임시조치의 switching-loss·열 비용을 계산한다. "
        "추가: ① 05장 블라인드 문제와 같이 전류 증거 없는 threshold crossing만으로 결론내리지 않는다. ② DC-link 전류 peak 증가는 빠른 edge의 C_oss·dv/dt만으로도 생기므로 shoot-through의 증거가 아니다 — "
        "dv/dt 구간을 넘어서는 추가 전하와 손실이 증거다. ③ H1과 H4는 R_g,off에 둘 다 반응하므로 gate 명령·die 문턱 교차 timing(동시 취득, deskew)으로 가른다. ④ H2와 H3는 probe tip 단락 시험으로 가른다. "
        "⑤ 이 합성 screen의 V_th·C_gd는 가정이다: 실제 여유는 온도별 V_th, C_rss(V), 내부 R_g 자료로 정한다 (MISSING_INPUT)."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: SiC 교체 후 Q_L의 +5 V spike가 실제 shoot-through인지 확정하고 대책을 정한다. 확인한 조건: power source 기준 probe에서 +5 V, DC-link 전류 peak 증가. "
        "모르는 조건: 고온 V_th, 실제 dead time, probe 접지 위치. 유력 가설과 반례: Miller turn-on과 dead-time 겹침은 실제 전류를 만들고, 공통 source L·probe pickup은 겉보기 전압만 만든다 — 전류 peak 증가는 빠른 edge의 용량성 전류만으로도 생겨 반례가 된다. "
        "현재 권고: gate–Kelvin source를 짧은 접지로 다시 재고, V_DS·leg 전류·gate 명령을 deskew된 같은 timebase로 받는다. 부작용: R_g,on을 키우는 임시조치는 원인을 가리지 못하고 소자당 수십 W 손실을 늘린다. "
        "완료 판정: Kelvin 기준 spike가 V_th 아래이고 dv/dt 구간 밖의 추가 전하가 없으면 측정 문제로 닫고, 남으면 R_g,off/clamp 변경 후 같은 측정을 반복한다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the gate margin of the off device during the complementary turn-on. My current hypothesis is a Miller-induced turn-on, but a common-source inductance in the probe reference or probe pickup could produce a similar 5 V spike, and a short real dead time could too. "
        "I would separate them by re-measuring gate to Kelvin source with a short ground and capturing drain-source voltage, leg current and gate commands on one deskewed timebase. "
        "If the Kelvin-referenced spike stays below threshold and there is no extra charge outside the dv/dt edge, I recommend treating it as a measurement artefact and would recheck the probe setup; otherwise I recommend a lower turn-off gate resistance or a Miller clamp and would recheck the switching loss."
    ),
    questions=[
        q_hyp(
            "A",
            "off gate에 +5 V spike, DC-link 전류 peak 증가. 원인 가설 3개는?",
            "Miller coupling(C_gd·dv/dt로 실제 gate 상승), common-source inductance(power source 기준 측정에 L_s·di/dt가 더해짐), 측정 artifact(probe 접지 loop pickup·CM 오차), 실제 dead-time overlap 중 3개 이상. "
            "앞의 둘(Miller, overlap)만 채널을 실제로 열고, DC-link 전류 peak 증가는 빠른 edge의 용량성 전류로도 설명된다는 반례를 함께 말한다.",
            "A 5 V spike on the off gate and a higher DC-link current peak: give three hypotheses.",
            "Miller coupling raising the real gate voltage, the common-source inductance adding L di/dt to a power-source-referenced measurement, a probe artefact from the ground loop, and a real dead-time overlap. Only Miller and overlap open the channel; the higher current peak can also come from the capacitive current of the faster edge.",
            ["Miller C_gd·dv/dt", "common-source L·di/dt", "probe artifact", "dead-time overlap", "전류 peak 반례"],
        ),
        q_wave(
            "A",
            "① gate–Kelvin source 전압(짧은 접지 spring 또는 절연 차동 probe) ② 같은 timebase의 V_DS(상대 소자 turn-on edge)와 leg/DC-link 전류 ③ 두 소자의 gate 명령(또는 driver 입력). "
            "probe deskew·대역폭·접지 위치·온도·V_DC·부하전류를 함께 기록한다.",
            "Gate to Kelvin source with a short ground or an isolated differential probe; the drain-source voltage and the leg or DC-link current on the same timebase; and both gate commands. Record deskew, bandwidth, ground point, temperature, bus voltage and load current.",
            ["Kelvin 기준 v_GS", "V_DS + 전류 동시", "gate 명령", "deskew·대역·접지 위치"],
        ),
        q_first(
            "A",
            "Kelvin 기준 재측정: H2(공통 source L)·H3(pickup)이면 spike가 사라지고 H1·H4면 남는다. 남으면 R_g,off를 줄이거나 clamp를 켜 반응을 보고(H1·H4 반응), 동시 취득으로 spike가 dv/dt 구간(H1)인지 명령 직후 gate 방전 중(H4)인지 본다. "
            "probe tip 단락 시험은 H3만 신호를 보인다.",
            "Re-measure gate to Kelvin source. If the spike disappears it was the reference inductance or pickup; if it stays, change the turn-off gate resistance or enable a clamp and use a synchronized capture to see whether the spike sits in the dv/dt edge (Miller) or right after the command while the gate is still discharging (overlap).",
            ["Kelvin 재측정", "H2·H3 사라짐", "R_g,off/clamp 반응", "timing으로 H1/H4 구분"],
        ),
        q_memo(
            "A",
            "목표: shoot-through 여부 확정. 확인: PS 기준 +5 V, 전류 peak 증가. 모름: 고온 V_th, 실제 dead time, probe 배치. 가설과 반례: Miller·겹침은 실제 전류, L_s·pickup은 겉보기 — 전류 peak는 C_oss·dv/dt만으로도 증가. "
            "권고: Kelvin 재측정 + 동시 취득 (임시로 R_g,on을 키우는 것은 원인 미확정 상태의 손실 증가). 부작용: R_g,off/clamp 변경 시 turn-off 손실·overshoot 재확인. 완료 판정: Kelvin spike < V_th이고 dv/dt 밖 추가 전하 없음.",
            MEMO_EN_TEMPLATE.replace("___", "the off-device gate margin", 1),
        ),
    ],
    circuit="fl12_a",
    textbook=[TB_15, TB_05, TB_E03, TB_17],
    reference_presets=["observed", "kelvin", "rg_off"],
    runtime_hint="seconds",
    claim_limit="합성 edge·gate RC screen으로 가설별 예측을 비교한다. 고객 소자의 false turn-on 여부·여유를 확정하지 않는다.",
)


# ======================================================================================
# CASE B - PFC over-current only at low line (11 kW at 360 V)
# ======================================================================================

SQ2 = math.sqrt(2.0)
SQ3 = math.sqrt(3.0)

B_TESTS = [
    ("none", "시험 전: 400 V 정상 / 360 V 제한"),
    ("sweep", "360 V에서 전력을 올리며 trip 지점의 RMS (기준 분석기)"),
    ("fs", "f_s +25 % (ripple −20 %)"),
    ("probe", "기준 probe vs 내부 센서 (같은 운전점)"),
    ("hf", "line peak에서 인덕터 전류 고대역 캡처 (곡률)"),
    ("flat", "flat-top 계통 전압 (grid simulator, 5차 −5 %)"),
]


def i_line(P: float, V: float, eta: float, PF: float) -> float:
    return P / (SQ3 * V * eta * PF)


def ripple_pp_b(v_ph, Vdc: float, L: float, fs: float):
    """Switching ripple (pp) of a 4-wire-equivalent leg (+-Vdc/2 against the phase voltage), linear L."""
    x = np.asarray(v_ph, dtype=float) / (Vdc / 2)
    return Vdc * (1.0 - x * x) / (4.0 * L * fs)


class PwlL:
    """Piecewise-linear flux linkage of a saturating inductor: L0 below I_sat, L0/k above (odd symmetric)."""

    def __init__(self, L0: float, Isat: float, k: float):
        self.L0, self.Isat, self.k = L0, Isat, k
        self.Ls = L0 / k
        self.lam_s = L0 * Isat

    def i(self, lam):
        lam = np.asarray(lam, dtype=float)
        a = np.abs(lam)
        return np.sign(lam) * np.where(a <= self.lam_s, a / self.L0, self.Isat + (a - self.lam_s) / self.Ls)

    def W(self, lam):
        """Co-content W(lam) = int_0^lam i dlam (even)."""
        a = np.abs(np.asarray(lam, dtype=float))
        lin = a * a / (2 * self.L0)
        sat = self.lam_s**2 / (2 * self.L0) + self.Isat * (a - self.lam_s) + (a - self.lam_s) ** 2 / (2 * self.Ls)
        return np.where(a <= self.lam_s, lin, sat)

    def L_at(self, i):
        return np.where(np.abs(i) <= self.Isat, self.L0, self.Ls)


def leg_period(i_avg: float, v_ph: float, Vdc: float, fs: float, ind: PwlL) -> dict:
    """One switching period of a 4-wire-equivalent leg (+-Vdc/2 to the grid phase) in flux coordinates.

    lam rises by dlam = (Vdc/2 - v) d T_s (d from volt-second balance) and falls back; both slopes sample
    lam uniformly, so the period average of i equals the lam-average (W(lam_p) - W(lam_v))/dlam, which
    fixes lam_v for the controlled average current.  Exact for the piecewise-linear lam(i)."""
    x = v_ph / (Vdc / 2)
    d = 0.5 * (1.0 + x)
    dlam = (Vdc / 2 - v_ph) * d / fs
    f = lambda lv: float((ind.W(lv + dlam) - ind.W(lv)) / dlam) - i_avg  # noqa: E731
    span = abs(i_avg) * ind.L0 + 10 * dlam + ind.lam_s
    lv = brentq(f, -span - 1.0, span + 1.0, xtol=1e-15, rtol=1e-13, maxiter=300)
    return {"i_v": float(ind.i(lv)), "i_p": float(ind.i(lv + dlam)), "d": d, "lam_v": lv, "dlam": dlam}


def leg_period_ode(i_v: float, v_ph: float, Vdc: float, fs: float, ind: PwlL) -> float:
    """Independent path: di/dt = v_L / L(i) over the on-time with DOP853 (event at I_sat)."""
    x = v_ph / (Vdc / 2)
    d = 0.5 * (1.0 + x)
    vL = Vdc / 2 - v_ph
    sol = solve_ivp(lambda t, y: [vL / float(ind.L_at(y[0]))], (0.0, d / fs), [i_v], method="DOP853", rtol=1e-12, atol=1e-12, max_step=d / fs / 2000)
    return float(sol.y[0][-1])


def peak_envelope(I_rms: float, V_LL: float, Vdc: float, fs: float, ind: PwlL, n: int = 91, flat: float = 0.0) -> dict:
    """Average current and switching-ripple peak over a positive half line cycle (unity PF, resistor emulation)."""
    th = np.linspace(1e-3, math.pi - 1e-3, n)
    shape = np.sin(th) - flat * np.sin(5 * th)
    shape_rms = math.sqrt(0.5 * (1 + flat * flat))
    Vpk = V_LL / SQ3 * SQ2
    iavg = I_rms / shape_rms * shape
    vph = Vpk * shape  # resistor emulation: the grid voltage has the same shape as the current
    ipk = np.empty_like(th)
    ivl = np.empty_like(th)
    for k in range(th.size):
        r = leg_period(float(iavg[k]), float(vph[k]), Vdc, fs, ind)
        ipk[k], ivl[k] = r["i_p"], r["i_v"]
    return {"th": th, "iavg": iavg, "ipk": ipk, "ivl": ivl}


@dataclass
class WorldB:
    key: str
    kind: str  # rms | peak | ocp | sensor | sat
    thr: float  # threshold of the limiter quantity
    g: float = 1.0  # sensor gain (H4)
    Isat: float = 1e9  # saturation current (H5)


def limiter_q(w: WorldB, v: dict, V_LL: float, P: float, fs: float | None = None, flat: float = 0.0) -> float:
    """The quantity this world's limiter compares with its threshold."""
    I = i_line(P, V_LL, v["eta"], v["PF"])
    fs = v["fs"] if fs is None else fs
    if w.kind == "rms":
        return I
    if w.kind == "sensor":
        return w.g * I
    if w.kind == "peak":
        shape_peak = max(abs(math.sin(t) - flat * math.sin(5 * t)) for t in np.linspace(0, math.pi, 721))
        return I / math.sqrt(0.5 * (1 + flat * flat)) * shape_peak
    if w.kind == "ocp":  # linear L: closed-form ripple, vectorised over the half cycle
        th = np.linspace(0.0, math.pi, 1441)
        shape = np.sin(th) - flat * np.sin(5 * th)
        iavg = I / math.sqrt(0.5 * (1 + flat * flat)) * shape
        return float(np.max(iavg + 0.5 * ripple_pp_b(V_LL / SQ3 * SQ2 * shape, v["Vdc"], v["L"], fs)))
    ind = PwlL(v["L"], w.Isat, v["k_sat"])
    th = np.linspace(math.pi / 4, 3 * math.pi / 4, 19)
    shape = np.sin(th) - flat * np.sin(5 * th)
    iavg = I / math.sqrt(0.5 * (1 + flat * flat)) * shape
    vph = V_LL / SQ3 * SQ2 * shape
    return max(leg_period(float(a), float(b), v["Vdc"], fs, ind)["i_p"] for a, b in zip(iavg, vph))


def trip_rms(w: WorldB, v: dict, V_LL: float, fs: float | None = None, flat: float = 0.0) -> float:
    """Line RMS current at which this world's limiter trips at V_LL (power raised until the limit)."""
    f = lambda P: limiter_q(w, v, V_LL, P, fs, flat) - w.thr  # noqa: E731
    P = brentq(f, 1e3, 40e3, xtol=1e-3, rtol=1e-10)
    return i_line(P, V_LL, v["eta"], v["PF"])


def run_case_b(v: dict) -> Result:
    res = Result("FL12", "case_b", "A (입력 경계·limiter 정의·4-wire 등가 ripple·포화 인덕터 자속좌표 해) + FL05 실행 인용")
    P, eta, PF, Vh, Vl = v["P_bat"], v["eta"], v["PF"], v["V_LL"], v["V_low"]
    Ih, Il = i_line(P, Vh, eta, PF), i_line(P, Vl, eta, PF)
    tb = abs(P - 11e3) < 1e-6 and abs(Vh - 400) < 1e-9 and abs(Vl - 360) < 1e-9 and abs(eta - 0.97) < 1e-12 and abs(PF - 0.995) < 1e-12
    res.add_metric("I_400", f"{P / 1e3:g} kW에 필요한 선전류 ({Vh:g} V)", Ih, "A", ref=16.45043 if tb else None, ref_label="교재 16.45 A (16.45043)", tol=2e-6, basis="RMS, 3상 선전류")
    res.add_metric("I_360", f"{P / 1e3:g} kW에 필요한 선전류 ({Vl:g} V)", Il, "A", ref=18.278 if tb else None, ref_label="교재 18.28 A (18.278)", tol=2e-5, basis="같은 출력 → 전압이 낮으면 RMS가 오른다")
    P16h = SQ3 * Vh * v["I_req"] * eta * PF
    P16l = SQ3 * Vl * v["I_req"] * eta * PF
    res.add_metric("P16_400", f"{v['I_req']:g} A 요구에서 가능한 출력 ({Vh:g} V)", P16h, "W", ref=10698.81 if (tb and abs(v["I_req"] - 16) < 1e-9) else None, ref_label="교재 10.69881 kW", tol=2e-6)
    res.add_metric("P16_360", f"{v['I_req']:g} A 요구에서 가능한 출력 ({Vl:g} V)", P16l, "W", basis="16 A가 실제 요구라면 이것이 올바른 derating 출력")
    # worlds: calibrate each limiter so that 400 V passes and 360 V trips (mid of the consistency interval)
    worlds: dict[str, WorldB] = {}
    intervals = {}
    lo, hi = Ih, Il
    worlds["H1"] = WorldB("H1", "rms", 0.5 * (lo + hi))
    intervals["H1"] = (lo, hi, "A (RMS 한계)")
    lo2, hi2 = SQ2 * Ih, SQ2 * Il
    worlds["H2"] = WorldB("H2", "peak", 0.5 * (lo2 + hi2))
    intervals["H2"] = (lo2, hi2, "A (ripple 없는 peak 한계)")
    w3 = WorldB("H3", "ocp", 0.0)
    lo3, hi3 = limiter_q(w3, v, Vh, P), limiter_q(w3, v, Vl, P)
    w3.thr = 0.5 * (lo3 + hi3)
    worlds["H3"] = w3
    intervals["H3"] = (lo3, hi3, "A (ripple 포함 순시 OCP)")
    glo, ghi = v["I_lim4"] / Il, v["I_lim4"] / Ih
    worlds["H4"] = WorldB("H4", "sensor", v["I_lim4"], g=0.5 * (glo + ghi))
    intervals["H4"] = (glo, ghi, f"(센서 이득, 한계 {v['I_lim4']:g} A RMS)")

    def sat_ok(Isat):
        w = WorldB("H5", "sat", v["I_ocp5"], Isat=Isat)
        return limiter_q(w, v, Vh, P) < v["I_ocp5"] < limiter_q(w, v, Vl, P)

    # bracket of I_sat values consistent with the observation (bisection on both ends)
    grid = np.linspace(0.5 * SQ2 * Ih, v["I_ocp5"], 41)
    okk = [float(x) for x in grid if sat_ok(float(x))]
    notes = {}
    if okk:
        a0, b0 = min(okk), max(okk)

        def edge(inside, outside):
            for _ in range(40):
                mid = 0.5 * (inside + outside)
                if sat_ok(mid):
                    inside = mid
                else:
                    outside = mid
            return inside
        step = grid[1] - grid[0]
        lo5 = edge(a0, max(a0 - step, 1e-3))
        hi5 = edge(b0, b0 + step)
        worlds["H5"] = WorldB("H5", "sat", v["I_ocp5"], Isat=0.5 * (lo5 + hi5))
        intervals["H5"] = (lo5, hi5, f"A (I_sat, OCP {v['I_ocp5']:g} A)")
    else:
        notes["H5"] = f"OCP {v['I_ocp5']:g} A에서는 어떤 I_sat도 ‘400 V 정상·360 V trip’을 재현하지 못한다"
    hyps = [
        Hyp("H1", "정상 RMS 증가 + 올바른 RMS 제한", "360 V에서 같은 출력이 18.28 A를 요구해 RMS 한계에 걸린다 (derating이 올바른 동작일 수 있음)", "RMS 한계값"),
        Hyp("H2", "한계 정의 불일치 (RMS 요구를 peak로 구현)", "ripple 없는 표본의 peak/√2로 비교 — 정현파에서는 RMS와 같지만 파형이 바뀌면 달라진다", "peak 한계값"),
        Hyp("H3", "ripple peak (순시 OCP)", "저전압에서 변조지수가 낮아 line peak의 switching ripple이 커져 순시 OCP에 닿는다", "OCP 값"),
        Hyp("H4", "센서 scaling 오차", "전류센서가 실제보다 크게 읽어 올바른 한계가 일찍 걸린다", "센서 이득"),
        Hyp("H5", "인덕터 포화", "line peak 전류가 커지면 L이 무너져 ripple peak가 급증해 OCP에 닿는다", "I_sat"),
    ]
    for k, (a, b, u) in intervals.items():
        w = worlds[k]
        val = {"H1": w.thr, "H2": w.thr, "H3": w.thr, "H4": w.g, "H5": w.Isat}[k]
        res.add_metric(f"cal_{k}", f"{k} 세계의 미지수 (관측과 맞는 범위의 가운데)", val, "" if k == "H4" else "A", basis=f"관측과 맞는 범위 {a:.4g} – {b:.4g} {u}")
    # utilisation vs line voltage for every world (the single observation: < 1 at 400 V, > 1 at 360 V)
    Vs = np.linspace(330.0, 440.0, 23)
    for k, w in worlds.items():
        res.add_series(f"u_{k}", f"{k} 판정량/한계", "", Vs.tolist(), [limiter_q(w, v, float(x), P) / w.thr for x in Vs])
    # current waveforms at 360 V
    ind_lin = PwlL(v["L"], 1e9, v["k_sat"])
    env = peak_envelope(Il, Vl, v["Vdc"], v["fs"], ind_lin)
    deg = (env["th"] * 180 / math.pi).tolist()
    res.add_series("w_avg", "평균 상전류 (정현, 모든 세계)", "A", deg, env["iavg"].tolist())
    res.add_series("w_rip", "H3: ripple peak (선형 L)", "A", deg, env["ipk"].tolist())
    if "H5" in worlds:
        env5 = peak_envelope(Il, Vl, v["Vdc"], v["fs"], PwlL(v["L"], worlds["H5"].Isat, v["k_sat"]))
        res.add_series("w_sat", "H5: ripple peak (포화 L)", "A", deg, env5["ipk"].tolist())
    res.add_series("w_sens", "H4: 센서가 읽는 평균", "A", deg, (worlds["H4"].g * env["iavg"]).tolist(), dash=True)
    hl = [{"y": worlds["H2"].thr, "label": f"H2 peak 한계 {worlds['H2'].thr:.3g} A"}, {"y": worlds["H3"].thr, "label": f"H3 OCP {worlds['H3'].thr:.3g} A"}]
    if "H5" in worlds:
        hl.append({"y": v["I_ocp5"], "label": f"H5 OCP {v['I_ocp5']:g} A"})
    # switching period at the line peak (H3 linear vs H5 saturating)
    Vpk_l = Vl / SQ3 * SQ2
    per = {}
    for tag, ind in (("lin", ind_lin), ("sat", PwlL(v["L"], worlds["H5"].Isat, v["k_sat"]) if "H5" in worlds else None)):
        if ind is None:
            continue
        r = leg_period(SQ2 * Il, Vpk_l, v["Vdc"], v["fs"], ind)
        d, dl = r["d"], r["dlam"]
        n_up, n_dn = 120, 120
        ts = np.concatenate([np.linspace(0, d / v["fs"], n_up), np.linspace(d / v["fs"], 1 / v["fs"], n_dn)[1:]])
        lam = np.where(ts <= d / v["fs"], r["lam_v"] + dl * ts / (d / v["fs"]), r["lam_v"] + dl * (1 - (ts - d / v["fs"]) / ((1 - d) / v["fs"])))
        per[tag] = (ts, ind.i(lam), r)
        res.add_series(f"sw_{tag}", "H3 선형 L" if tag == "lin" else "H5 포화 L", "A", ts.tolist(), ind.i(lam).tolist())
    # discriminating tests
    fs2 = v["fs"] * 1.25
    flat = 0.05
    cells: dict = {}
    trip_rows = []
    for k, w in worlds.items():
        tr0 = trip_rms(w, v, Vl)
        tr_fs = trip_rms(w, v, Vl, fs=fs2)
        tr_fl = trip_rms(w, v, Vl, flat=flat)
        reading = (w.g - 1.0) * 100
        if k == "H5":
            ts_, iw, r = per["sat"]
            sl = np.diff(iw[: 120]) / np.diff(ts_[: 120])
            curv = float(np.max(sl) / np.min(sl))
        else:
            curv = 1.0
        cells[k] = {
            "none": ("400 V 통과, 360 V trip", "관측"),
            "sweep": (f"trip @ {tr0:.3g} A RMS", "한계 근처 (설정 모르면 구분 불가)"),
            "fs": (f"trip {tr0:.3g} → {tr_fs:.3g} A ({'이동' if abs(tr_fs / tr0 - 1) > 2e-3 else '그대로'})", "이동" if abs(tr_fs / tr0 - 1) > 2e-3 else "그대로"),
            "probe": (f"센서 − 기준 = {reading:+.2f} %", "센서가 크게 읽음" if abs(reading) > 0.5 else "일치"),
            "hf": (f"on 구간 기울기 비 {curv:.2f}" + (" (곡률: 포화)" if curv > 1.05 else " (직선)"), "곡률" if curv > 1.05 else "직선"),
            "flat": (f"trip {tr0:.3g} → {tr_fl:.3g} A RMS ({'증가' if tr_fl > tr0 * 1.002 else '그대로'})", "증가" if tr_fl > tr0 * 1.002 else "그대로"),
        }
        trip_rows.append([k, f"{limiter_q(w, v, Vh, P) / w.thr:.4f}", f"{limiter_q(w, v, Vl, P) / w.thr:.4f}", f"{tr0:.4g}", f"{SQ3 * Vl * tr0 * eta * PF / 1e3:.4g}"])
    for k in notes:
        cells[k] = {t: ("관측 재현 불가", "재현 불가") for t, _ in B_TESTS}
    groups = disc_tables(res, hyps, B_TESTS, cells, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설이 예측하는 결과",
                         note="모든 세계는 ‘400 V 정상, 360 V 제한’을 재현하도록 한계값·센서 이득·I_sat을 범위의 가운데로 맞췄다. trip 지점의 RMS만으로는 설정값을 모르면 구분할 수 없다: f_s 변경(ripple 의존), 기준 probe(센서), 고대역 캡처(포화 곡률), flat-top 전압(RMS 대 peak 정의)을 조합한다.")
    res.tables.append(Table("t_world", "가설별 세계: limiter 사용률과 360 V trip 지점", ["가설", "사용률 400 V", "사용률 360 V", "360 V trip RMS [A]", "그때 출력 [kW]"], trip_rows,
                            note="사용률 = 그 세계 limiter의 판정량 / 한계값. 1을 넘으면 제한. 모든 세계가 400 V < 1 < 360 V."))
    # plots
    res.add_plot("p_util", "다섯 세계 모두 400 V에서는 통과, 360 V에서는 제한 (사용률 = 판정량/한계)", [f"u_{k}" for k in worlds], x_label="선간전압 V_LL", x_unit="V", y_label="사용률", y_unit="", kind="xy", level="A",
                 hlines=[{"y": 1.0, "label": "한계"}], vlines=[{"x": Vl, "label": f"{Vl:g} V"}, {"x": Vh, "label": f"{Vh:g} V"}],
                 proved="RMS·peak 정의·순시 OCP·센서 이득·포화라는 다섯 메커니즘이 각자의 미지수를 관측 범위 안에 두면 모두 400 V와 360 V 사이에서 한계를 넘는다 — 같은 증상이다.",
                 not_yet="η·PF를 운전점과 무관한 상수로 두었고(교재 합성 조건) 실제 limiter의 필터·hysteresis·제어 응답은 넣지 않았다. 어느 세계가 고객의 것인지는 아직 아무것도 말하지 않는다.")
    res.add_plot("p_wave", f"{Vl:g} V에서 {P / 1e3:g} kW 요청 시 한 상 전류 (반주기): 평균과 ripple peak", ["w_avg", "w_rip"] + (["w_sat"] if "H5" in worlds else []) + ["w_sens"], x_label="계통 위상 θ", x_unit="deg", y_label="i", y_unit="A", kind="xy", level="A", hlines=hl,
                 proved="평균 전류는 정현이지만 line peak에서 4-wire 등가 ripple(V_dc(1−x²)/(4Lf_s))과 포화 L의 ripple이 순시 peak를 키우고, 센서 이득 오차는 읽는 값을 키운다 — 세계마다 한계에 닿는 양이 다르다.",
                 not_yet="ripple은 한 상을 DC 중점에 연결한 4-wire 등가와 단위역률 가정(A 수준)이다. 3선식 공통모드 전압·dead time·제어 지연은 포함하지 않았다.")
    if "sat" in per:
        res.add_plot("p_sw", "line peak의 한 스위칭 주기: 선형 L은 직선, 포화 L은 on 구간 끝이 휜다", ["sw_lin", "sw_sat"], y_label="i_L", y_unit="A", level="A", group="b_sw",
                     hlines=[{"y": worlds["H5"].Isat, "label": f"I_sat {worlds['H5'].Isat:.3g} A"}],
                     proved="자속좌표 정확해(평균전류 고정)에서 I_sat을 넘는 순간 기울기가 L0/L_sat배로 바뀌어 peak가 커진다. 독립 ODE(DOP853)가 같은 peak를 준다(check).",
                     not_yet="구간선형 λ(i)는 학습용이다. 실제 코어의 연속 포화곡선·온도 의존 B_sat은 MISSING_INPUT이다.")
    tsel = v["test"]
    if tsel in ("fs", "flat"):
        for k, w in worlds.items():
            res.add_series(f"t_{k}", f"{k} ({'f_s +25 %' if tsel == 'fs' else 'flat-top'})", "", Vs.tolist(), [limiter_q(w, v, float(x), P, fs2 if tsel == "fs" else None, flat if tsel == "flat" else 0.0) / w.thr for x in Vs])
        res.add_plot("p_test", "시험: " + dict(B_TESTS)[tsel], [f"t_{k}" for k in worlds], x_label="선간전압 V_LL", x_unit="V", y_label="사용률", y_unit="", kind="xy", level="A", hlines=[{"y": 1.0, "label": "한계"}], vlines=[{"x": Vl, "label": f"{Vl:g} V"}],
                     proved=("ripple에 의존하는 H3·H5만 사용률 곡선이 내려가 trip 전압이 이동한다." if tsel == "fs" else "peak로 판정하는 H2·H3·H5는 crest factor가 줄어 같은 RMS에서 사용률이 내려가고, RMS로 판정하는 H1·H4는 그대로다."),
                     not_yet="실제 시험에서는 f_s 변경이 손실·EMI·제어 대역도 바꾸고, grid simulator 파형은 PFC 전류 제어의 추종 오차를 함께 만든다.")
    elif tsel == "probe":
        ref_I = np.linspace(10, 20, 11)
        for k, w in worlds.items():
            res.add_series(f"t_{k}", f"{k} 센서 읽음", "A", ref_I.tolist(), (w.g * ref_I).tolist())
        res.add_plot("p_test", "시험: 기준 probe(가로) vs 내부 센서(세로)", [f"t_{k}" for k in worlds], x_label="기준 분석기 RMS", x_unit="A", y_label="센서 RMS", y_unit="A", kind="xy", level="A",
                     proved="센서 scaling 오차(H4)만 기울기가 1이 아니다 — 같은 운전점에서 두 측정을 비교하면 한 번에 가려진다.", not_yet="기준 probe 자체의 교정 불확도와 대역폭을 먼저 확인해야 한다(EX11).")
    elif tsel == "sweep":
        Ps = np.linspace(8e3, 13e3, 26)
        for k, w in worlds.items():
            res.add_series(f"t_{k}", f"{k} 판정량/한계", "", (Ps / 1e3).tolist(), [limiter_q(w, v, Vl, float(p_)) / w.thr for p_ in Ps])
        res.add_plot("p_test", f"시험: {Vl:g} V에서 전력을 올리며 trip 지점 찾기", [f"t_{k}" for k in worlds], x_label="요청 출력", x_unit="kW", y_label="사용률", y_unit="", kind="xy", level="A", hlines=[{"y": 1.0, "label": "한계"}],
                     proved="모든 세계가 비슷한 전력·RMS에서 한계에 닿는다: 설정값을 모르는 한 trip 지점만으로는 원인을 가를 수 없다.", not_yet="trip 지점은 한계 설정의 공차·온도에 따라서도 움직인다.")
    res.add_metric("I_rip_400", "line peak ripple pp (400 V, 선형 L)", ripple_pp_b(Vh / SQ3 * SQ2, v["Vdc"], v["L"], v["fs"]), "A", basis="4-wire 등가 V_dc(1−x²)/(4Lf_s)")
    res.add_metric("I_rip_360", "line peak ripple pp (360 V, 선형 L)", ripple_pp_b(Vl / SQ3 * SQ2, v["Vdc"], v["L"], v["fs"]), "A", basis="저전압 → 변조지수 x↓ → ripple ↑")
    # checks: independent time-domain three-phase power path (FL05's helper), ODE for the saturating period
    from .fl05_pfc import three_phase_power_td

    for V_, I_ in ((Vh, Ih), (Vl, Il)):
        Itd = brentq(lambda I: three_phase_power_td(V_, I, PF)[0] - P / eta, 1e-3, 200.0, xtol=1e-13, rtol=1e-14)
        res.add_check(check_close(f"선전류 {V_:g} V: 시간영역 Σv·i vs √3 식", Itd, I_, 1e-9, "상별 정현 v·i를 한 주기 평균해 P/η가 되는 전류 (FL05 helper) vs P/(√3VηPF)", True, "A"))
    r_lin = leg_period(SQ2 * Il, Vpk_l, v["Vdc"], v["fs"], ind_lin)
    rip_cf = ripple_pp_b(Vpk_l, v["Vdc"], v["L"], v["fs"])
    res.add_check(check_close("ripple: 자속좌표 해(선형 L) vs 닫힌 식 V_dc(1−x²)/(4Lf_s)", r_lin["i_p"] - r_lin["i_v"], rip_cf, 1e-9, "W(λ) 평균조건 brentq vs 4-wire 등가 해석식", True, "A"))
    if "sat" in per:
        r5 = per["sat"][2]
        ind5 = PwlL(v["L"], worlds["H5"].Isat, v["k_sat"])
        ip_ode = leg_period_ode(r5["i_v"], Vpk_l, v["Vdc"], v["fs"], ind5)
        res.add_check(check_close("포화 L의 peak: 자속좌표 정확해 vs ODE di/dt = v/L(i)", ip_ode, r5["i_p"], 1e-6, "구간선형 λ(i)의 해석 역함수 vs DOP853 적분 (on 구간)", True, "A"))
    # cited run (FL05)
    ev = Evidence(res)
    i1 = ev.cite("FL05", "grid_boundary", "nominal", "I_line", note="400 V에서 11 kW에 필요한 전류")
    p1 = ev.cite("FL05", "grid_boundary", "nominal", "P_at_lim", note="16 A 요구 안에서 가능한 출력 (400 V)")
    i2 = ev.cite("FL05", "grid_boundary", "nominal", "I_low", note="360 V low line에서 같은 출력에 필요한 전류")
    if tb and abs(v["I_req"] - 16) < 1e-9:
        res.add_check(Check("FL05 실행값 = 이 사례의 계산", "PASS" if abs(i1 / Ih - 1) < 1e-12 and abs(i2 / Il - 1) < 1e-12 and abs(p1 / P16h - 1) < 1e-12 else "FAIL", max(abs(i1 / Ih - 1), abs(i2 / Il - 1), abs(p1 / P16h - 1)), "rel", 1e-12,
                            path="FL05 grid_boundary@nominal (별도 모듈) vs 이 사례의 P/(√3VηPF)", independent=True))
    ev.finish()
    res.circuit = {"diagram": circuit_b(v).to_json(), "intervals": [], "plot_group": ""}
    unresolved(res, f"f_s 변경({groups['fs']}묶음)·기준 probe({groups['probe']})·고대역 캡처({groups['hf']})·flat-top({groups['flat']})를 조합해야 다섯 가설이 갈린다.")
    res.verdict("CUSTOMER_DECISION_REQUIRED", f"16 A가 실제 계통/EVSE 요구라면 {Vh:g} V에서도 11 kW({Ih:.4g} A)는 요구 위반이고, {Vl:g} V에서 {P16l / 1e3:.4g} kW로 derating하는 것이 올바른 동작이다 — 요구를 고객과 먼저 확정한다. OCP를 올리기 전에 자성체 peak·소자 SOA·배선 한계를 확인한다.")
    res.verdict("MISSING_INPUT", "limiter의 정의(RMS/peak/순시, 필터)·설정값·센서 교정·인덕터 포화곡선(온도)이 없어 고객 원인은 정하지 않는다 (합성 세계는 그대로 실행)")
    for k, why in notes.items():
        res.warnings.append(f"{k}: {why}")
    res.assumptions += [
        "η·PF는 상수 (교재 합성 조건), 3상 평형, 단위역률 저항 에뮬레이션",
        f"ripple: 한 상을 DC 중점에 잇는 4-wire 등가 leg (±V_dc/2), L {v['L'] * 1e6:g} µH, f_s {v['fs'] / 1e3:g} kHz, V_dc {v['Vdc']:g} V (ASSUMED)",
        "포화: 구간선형 λ(i) (I_sat 위에서 L/k), 스위칭 주기 안에서 평균전류를 맞춘 정확해 (A 수준 준정적)",
        "각 세계의 한계값·이득·I_sat은 관측(400 V 통과, 360 V 제한)과 맞는 범위의 가운데로 둔 합성 값",
    ]
    res.not_valid_for += ["실제 limiter 동작·제어 응답", "3선식 PFC의 실제 ripple 분포", "소자 SOA·배선 온도 판정", "고객 회로의 근본원인 확정"]
    res.interpretation = (
        f"같은 {P / 1e3:g} kW를 {Vl:g} V에서 내려면 선전류가 {Ih:.4g} A에서 {Il:.4g} A로 오른다. 이 정상적인 증가만으로도 400 V와 360 V 사이에 있는 RMS 한계(H1)에 걸린다. "
        "그런데 한계를 peak로 구현했거나(H2), ripple 포함 순시 OCP(H3), 센서 이득 오차(H4), 인덕터 포화(H5)도 ‘400 V 정상·360 V 제한’을 똑같이 만든다. "
        "trip 지점 하나로는 가를 수 없고 f_s 변경(ripple 의존 여부), 기준 probe 비교(센서), 고대역 전류 캡처(포화 곡률), flat-top 전압(RMS 대 peak 정의)으로 가른다. "
        f"무엇보다 16 A가 실제 요구라면 400 V에서도 11 kW는 불가능하고(16 A에서 {P16h / 1e3:.5g} kW) derating이 올바른 동작이다 — CUSTOMER_DECISION_REQUIRED. "
        "1쪽 메모 구조와 영어 틀: " + MEMO_EN_TEMPLATE
    )
    return res


def circuit_b(v: dict) -> Circuit:
    c = Circuit("fl12_b", 700, 260, title="3상 계통 → L → PFC → DC-link, 그리고 전류 한계가 보는 양")
    g = c.add("vsource", "G", 70, 130, 90, "계통 3상", f"{v['V_LL']:g}/{v['V_low']:g} V LL", lpos=(48, 126, "end"))
    ll = c.add("inductor", "L", 200, 60, 0, "L (상당)", f"{v['L'] * 1e6:g} µH", lpos=(200, 34, "middle"))
    c.add("block", "PFC", 370, 130, 0, "PFC (3상)", w=120, h=140)
    cc = c.add("capacitor", "C", 520, 130, 90, "C_dc", f"{v['Vdc']:g} V", lpos=(542, 126, "start"))
    c.add("block", "CT", 200, 175, 0, "전류센서", w=80, h=34)
    c.add("block", "LIM", 200, 230, 0, "limiter: RMS? peak? OCP?", w=180, h=30)
    c.add("block", "DCDC", 640, 130, 0, "DC/DC", w=80, h=50)
    c.wire("w1", g["a"], (70, 60), ll["a"])
    c.wire("w2", ll["b"], (310, 60))
    c.wire("w3", g["b"], (70, 200), (150, 200))
    c.wire("w3b", (250, 200), (310, 200))
    c.wire("w4", (430, 80), (520, 80), cc["a"])
    c.wire("w5", (430, 180), (520, 180), cc["b"])
    c.wire("w6", (520, 80), (640, 80), (640, 105))
    c.wire("w7", (520, 180), (640, 180), (640, 155))
    c.wire("w8", (200, 192), (200, 215))
    c.probe("pI", "w_avg", 130, 48, "right", "i_line")
    return c


def params_b() -> list[Param]:
    g_obs, g_c, g_w, g_t = "관측·요구", "회로 (합성)", "가설 세계의 고정값", "판별 시험"
    return [
        Param("P_bat", "요청 배터리 출력", "W", 11e3, "kW", vmin=100, vmax=1e6, source="TEXTBOOK", source_note="CASE B: 11 kW", group=g_obs),
        Param("V_LL", "정상 계통 선간전압", "V", 400.0, "V", vmin=100, vmax=1000, source="TEXTBOOK", source_note="400 V에서는 정상", group=g_obs),
        Param("V_low", "제한이 걸리는 low line", "V", 360.0, "V", vmin=100, vmax=1000, source="TEXTBOOK", source_note="360 V에서 제한", group=g_obs),
        Param("eta", "효율 η", "", 0.97, "", vmin=0.5, vmax=1.0, source="TEXTBOOK", source_note="교재 08장 합성 조건", group=g_obs),
        Param("PF", "역률 PF", "", 0.995, "", vmin=0.5, vmax=1.0, source="TEXTBOOK", group=g_obs),
        Param("I_req", "고객이 말한 전류 한계 (요구인지 확인 필요)", "A", 16.0, "A", vmin=1, vmax=500, source="TEXTBOOK", source_note="16 A", group=g_obs),
        Param("Vdc", "DC-link 전압", "V", 800.0, "V", vmin=100, vmax=2000, source="ASSUMED", group=g_c),
        Param("L", "PFC 인덕터 L (상당, 선형 영역)", "H", 250e-6, "µH", vmin=1e-6, vmax=0.1, source="ASSUMED", group=g_c),
        Param("fs", "스위칭 주파수", "Hz", 40e3, "kHz", vmin=1e3, vmax=1e6, source="ASSUMED", group=g_c),
        Param("k_sat", "포화 후 L 감소비 L0/L_sat", "", 5.0, "", vmin=1.01, vmax=1000, source="ASSUMED", source_note="구간선형 λ(i)", group=g_c),
        Param("I_lim4", "H4 세계의 RMS 한계 (센서가 맞으면 360 V도 통과할 값)", "A", 18.5, "A", vmin=1, vmax=500, source="ASSUMED", group=g_w),
        Param("I_ocp5", "H5 세계의 순시 OCP", "A", 34.0, "A", vmin=1, vmax=1000, source="ASSUMED", group=g_w),
        _test_param(B_TESTS),
    ]


EXP_B = Experiment(
    key="case_b",
    title="CASE B — PFC 저전압 입력에서만 과전류: 정상 RMS 증가인가, 한계 정의·ripple·센서·포화인가",
    goal=(
        "증상: 400 V 계통에서는 정상인데 360 V에서 11 kW를 요청하면 전류 제한에 걸린다. 확인된 것: 출력 11 kW, η 0.97·PF 0.995(교재 합성 조건), 고객이 말한 16 A. "
        "모르는 것: 16 A가 실제 계통/EVSE 요구인지, limiter가 RMS·peak·순시 OCP 중 무엇을 보는지, 센서 교정, 인덕터 포화 특성. "
        "먼저 Q1–Q3에 답을 적고 실행한다. 실행하면 같은 출력의 RMS 증가(16.45→18.28 A)를 계산하고, 정상 RMS 제한·정의 불일치·ripple peak·센서 scaling·포화 다섯 세계를 "
        "모두 ‘400 V 통과, 360 V 제한’이 되도록 맞춘 뒤 trip 지점·f_s 변경·기준 probe·고대역 캡처·flat-top 전압 시험의 예측을 비교한다."
    ),
    params=params_b(),
    presets=[
        Preset("observed", "관측 그대로 (400 V 정상, 360 V 제한)", {}, "CASE B 증상", ("nominal", "reference")),
        Preset("fs", "f_s +25 %", {"test": "fs"}, "ripple 의존(H3·H5)만 이동", ("reference",)),
        Preset("probe", "기준 probe vs 센서", {"test": "probe"}, "H4", ("variant", "reference")),
        Preset("hf", "고대역 인덕터 전류 캡처", {"test": "hf"}, "H5 곡률", ("variant",)),
        Preset("flat", "flat-top 계통 전압", {"test": "flat"}, "RMS 대 peak 정의", ("variant",)),
        Preset("sweep", "360 V 전력 sweep", {"test": "sweep"}, "trip 지점만으로는 구분 불가", ("variant",)),
        Preset("high_ocp", "H5 OCP 40 A", {"I_ocp5": 40.0}, "포화 세계가 관측을 재현하는 범위가 바뀐다", ("corner",)),
    ],
    run=run_case_b,
    model_level="A (경계·limiter 정의·ripple·포화 자속좌표 해)",
    suggested_change="판별 시험을 ‘f_s +25 %’로 바꾼다 (다음에 ‘기준 probe’, ‘flat-top’).",
    prediction=Prediction(
        "f_s를 25 % 올리면(ripple −20 %) 360 V의 trip 지점(RMS)은 어느 가설에서 움직이나?",
        ["모든 가설에서 움직인다", "ripple을 보는 H3(순시 OCP)와 H5(포화)에서만", "RMS 한계(H1)와 센서(H4)에서만", "모르겠다"],
        "ripple을 보는 H3(순시 OCP)와 H5(포화)에서만",
        "RMS 한계·정의 불일치·센서 이득은 ripple과 무관하다. 순시 OCP와 포화는 line peak의 ripple peak에 닿아 trip하므로 ripple이 줄면 더 큰 RMS까지 버틴다.",
        ["cal_H1", "cal_H3", "cal_H5"],
        handcalc=[{"key": "I_400", "label": "11 kW @ 400 V 선전류", "unit": "A"}, {"key": "I_360", "label": "11 kW @ 360 V 선전류", "unit": "A"}, {"key": "P16_400", "label": "16 A @ 400 V 출력", "unit": "W"}],
    ),
    suggested={"test": "fs"},
    student=(
        "같은 출력을 내려면 전압이 낮을수록 전류가 더 필요하다: 11 kW는 400 V에서 16.45 A, 360 V에서 18.28 A다. 그래서 360 V에서만 한계에 걸리는 것은 고장이 아니라 정상일 수 있다. "
        "다만 그 ‘한계’가 무엇을 재는지(RMS, 순간 peak, ripple 포함 peak), 센서가 정확한지, 인덕터가 큰 전류에서 포화하지 않는지에 따라 같은 증상이 나온다."
    ),
    expert=(
        "교재 해설(CASE B): 같은 출력에 필요한 RMS 증가(16.45→18.28 A)가 정상 원인일 수 있다. input current limit의 RMS/peak 정의, ripple, sensor scaling, saturation을 확인한다. "
        "16 A 제한이 실제 요구라면 출력 derating이 올바른 동작일 수 있다. OCP threshold만 높이기 전에 magnetic peak·semiconductor SOA·배선 한계를 확인한다. "
        "추가: ① 16 A가 요구라면 400 V에서도 11 kW는 16.45 A라 이미 위반이다 — 요구부터 확정(CUSTOMER_DECISION_REQUIRED). ② 저전압에서는 변조지수가 낮아 line peak의 switching ripple이 커진다(4-wire 등가 V_dc(1−x²)/(4Lf_s)): "
        "ripple을 보는 순시 OCP는 RMS 증가보다 더 빨리 한계에 닿는다. ③ trip 지점의 RMS만으로는 설정값을 모르는 한 다섯 가설이 갈리지 않는다. ④ 포화는 on 구간 전류 기울기의 곡률로, 센서 오차는 같은 운전점의 기준 probe 비교로, "
        "RMS 대 peak 정의는 crest factor를 바꾸는 시험(flat-top 전압)으로 드러난다."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: 360 V low line에서 11 kW가 제한되는 원인을 정하고 요구와 맞춘다. 확인한 조건: 같은 출력에 필요한 선전류는 400 V 16.45 A, 360 V 18.28 A. "
        "모르는 조건: 16 A가 계통/EVSE의 실제 요구인지, limiter의 정의(RMS/peak/OCP)와 설정, 센서 교정, 인덕터 포화 특성. 유력 가설과 반례: 정상 RMS 증가로 한계에 닿는 것이 가장 단순하지만, "
        "peak 정의·ripple OCP·센서 이득·포화도 같은 증상을 만든다. 현재 권고: 16 A가 요구라면 360 V에서 약 9.63 kW로 derating하는 것이 올바르다 — OCP만 올리지 않는다. "
        "부작용: 한계를 올리면 자성체 peak·소자 SOA·배선 온도 여유가 줄어든다. 필요한 시험과 완료 판정: 같은 운전점의 기준 probe 비교, f_s 변경, line peak 고대역 전류 캡처, limiter 정의 확인 → 원인이 하나로 좁혀지면 요구에 맞춰 derating 곡선이나 한계를 확정한다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the input current: 11 kW needs 16.45 A at 400 V and 18.28 A at 360 V. My current hypothesis is a normal RMS limit, but a peak-defined limit, the switching-ripple peak on an instantaneous over-current trip, a sensor scaling error or inductor saturation could produce a similar symptom. "
        "I would separate them by comparing the internal sensor with a reference probe, changing the switching frequency and capturing the inductor current at the line peak. "
        "If 16 A is a real grid requirement, I recommend derating to about 9.6 kW at 360 V and would recheck the magnetics peak, device SOA and wiring before raising any threshold."
    ),
    questions=[
        q_hyp(
            "B",
            "400 V는 정상, 360 V에서 11 kW 요청 시 제한. 원인 가설 3개는?",
            "① 정상 RMS 증가: 같은 출력에 360 V는 18.28 A(400 V 16.45 A)라 RMS 한계에 닿음 — derating이 올바를 수 있다 ② limiter 정의 불일치(RMS 요구를 peak로 구현) ③ line peak의 ripple peak가 순시 OCP에 닿음 ④ 센서 scaling 오차 ⑤ 인덕터 포화로 peak 급증. "
            "16 A가 실제 요구라면 400 V의 11 kW도 이미 위반이라는 반례를 말한다.",
            "No problem at 400 V, but the current limit trips at 360 V for 11 kW: give three hypotheses.",
            "A normal RMS increase, 16.45 A to 18.28 A for the same output, reaching an RMS limit, where derating may be correct; a limit defined on the peak instead of the RMS; the switching-ripple peak hitting an instantaneous over-current trip; a sensor scaling error; and inductor saturation.",
            ["16.45 → 18.28 A", "RMS/peak 정의", "ripple", "sensor scaling", "포화", "derating이 정상일 수 있음"],
        ),
        q_wave(
            "B",
            "① 기준 분석기로 잰 선전류 RMS·peak와 내부 센서 값(같은 운전점, 같은 창) ② line peak 부근의 인덕터 전류 고대역 캡처(on 구간 기울기) ③ limiter가 실제로 비교하는 내부 변수와 trip 순간의 계통 전압·출력. "
            "계통 전압 파형(왜곡), 온도, 측정 창·필터를 함께 기록한다.",
            "The line RMS and peak current from a reference analyzer next to the internal sensor value at the same point; a high-bandwidth inductor-current capture near the line peak; and the limiter's internal variable at the trip with the grid voltage and output power. Record the voltage distortion, temperature and the measurement window.",
            ["기준 vs 센서", "고대역 인덕터 전류", "limiter 내부 변수", "같은 창·온도"],
        ),
        q_first(
            "B",
            "같은 운전점에서 기준 probe와 내부 센서를 비교(H4이면 수 % 차이)하고, f_s를 바꿔 trip 지점이 움직이는지 본다(ripple에 의존하는 H3·H5만 이동). 남으면 line peak 전류의 곡률(H5)과 limiter 정의(RMS 대 peak, flat-top 전압 시험)로 가른다. "
            "그와 별도로 16 A가 요구인지 고객과 확인한다.",
            "Compare the internal sensor with a reference probe at the same point, then change the switching frequency and see whether the trip point moves; only the ripple-dependent hypotheses move. Then look for curvature in the inductor current at the line peak and check whether the limit is RMS- or peak-defined. Separately, confirm whether 16 A is a requirement.",
            ["기준 probe 비교", "f_s 변경", "곡률", "요구 확인"],
        ),
        q_memo(
            "B",
            "목표: 360 V 제한의 원인과 요구 확정. 확인: 16.45→18.28 A. 모름: 16 A 요구 여부, limiter 정의, 센서 교정, 포화. 가설과 반례: 정상 RMS 증가 vs 정의·ripple·센서·포화 — 16 A가 요구면 400 V 11 kW도 위반. "
            "권고: 요구라면 360 V 9.63 kW derating, OCP만 올리지 않음. 부작용: 자성체 peak·SOA·배선 여유 감소. 완료 판정: 기준 probe·f_s·곡률 시험으로 원인 하나, 요구에 맞춘 derating/한계 확정.",
            MEMO_EN_TEMPLATE.replace("___", "the input current limit", 1),
        ),
    ],
    circuit="fl12_b",
    textbook=[TB_15, TB_08, TB_17],
    reference_presets=["observed", "fs", "probe"],
    runtime_hint="seconds",
    claim_limit="A 수준 경계·ripple·포화 screen으로 가설별 예측을 비교한다. 고객 limiter의 실제 원인과 허용 전류를 확정하지 않는다.",
)


EXP_C = Experiment(
    key="case_c",
    title="CASE C — DAB 무부하에서 transformer가 뜨겁다: P = 0인데 흐르는 전류의 다섯 가지 원인",
    goal=(
        "증상: power command ≈ 0, phase ≈ 0인데 AC 전류가 남고 변압기가 뜨겁다. 확인된 것: 명령값(전력·φ ≈ 0), AC 전류가 흐른다는 사실. "
        "모르는 것: 실제 전압비(n·V_L), L_m, 실제 φ(gate 지연), 코어의 DC 자속, 제어 로그. 먼저 Q1–Q3에 답을 적고 실행한다. "
        "실행하면 교재의 900/600 V(φ = 0에서 2.165063509 A RMS)를 재현하고, 같은 RMS를 만드는 여자전류·실제 φ offset·DC 자속·limit cycle 세계를 맞춘 뒤 "
        "전압비 일치·L_m 변경·φ sweep·DC probe·장시간 기록에서 각 가설의 예측을 비교한다. 평균 출력전류만 보고 센서 오동작이라고 판단하지 않는 이유도 본다."
    ),
    params=params_c(),
    presets=[
        Preset("observed", "관측 그대로 (900/600 V corner)", {}, "CASE C 증상", ("nominal", "reference")),
        Preset("ratio", "첫 시험: 전압비 일치", {"test": "ratio"}, "H1만 사라짐", ("reference",)),
        Preset("dc", "DC 결합 probe", {"test": "dc"}, "H4", ("variant", "reference")),
        Preset("lm", "L_m ×2", {"test": "lm"}, "H2·H4", ("variant",)),
        Preset("log", "장시간 φ·포트 전류 기록", {"test": "log"}, "H3·H5", ("variant",)),
        Preset("fast_lc", "limit cycle 반주기 1주기", {"M_lc": 1}, "빠른 limit cycle", ("corner",)),
    ],
    run=run_case_c,
    model_level="A + C (구간선형 정확 파형, 스위칭 엔진)",
    suggested_change="판별 시험을 ‘전압비 일치’로 바꾼다 (다음에 ‘DC 결합 probe’, ‘장시간 기록’).",
    prediction=Prediction(
        "같은 전압비(n·V_L = V_H)에서 다시 재면 무엇이 사라지나?",
        ["모든 원인의 전류가 사라진다", "전압비 불일치(H1)의 순환전류만 사라진다", "여자전류(H2)도 함께 사라진다", "모르겠다"],
        "전압비 불일치(H1)의 순환전류만 사라진다",
        "v_L = ±(V₁ − V₂′)이므로 V₁ = V₂′이면 H1의 전류는 0이 된다. 여자전류·DC 자속·실제 φ offset·limit cycle은 전압비와 무관하게 남는다.",
        ["I_H1", "cal_H2", "P_H3"],
        handcalc=[{"key": "I_H1", "label": "900/600 V, φ = 0의 RMS", "unit": "A"}, {"key": "Ipk_H1", "label": "그 peak", "unit": "A"}],
    ),
    suggested={"test": "ratio"},
    student=(
        "전력이 0이어도 두 bridge의 사각파 전압이 다르면 그 차이가 직렬 L에 걸려 전류가 오르내린다(H1). 전압비가 맞아도 변압기를 자화하는 여자전류는 늘 흐르고(H2), "
        "명령은 0이지만 실제 위상이 어긋나 있거나(H3), 코어에 DC 자속이 쌓여 한쪽이 포화하거나(H4), 제어가 +와 −를 오가며 평균만 0으로 맞추는(H5) 경우에도 전류가 남는다."
    ),
    expert=(
        "교재 해설(CASE C): reflected voltage mismatch면 이상적으로도 순환전류가 생긴다. 동일 전압비에서 사라지는지 먼저 확인한다. magnetizing current, phase offset, flux imbalance, control limit cycle도 구분한다. "
        "평균 output current만 보고 current sensor 오동작이라고 판단하지 않는다. 추가: ① 같은 RMS 2.165 A를 만들려면 H2는 L_m ≈ 0.53 mH, H3는 실제 φ ≈ 0.35 rad(이때 실제로 1.6 kW가 흐른다 — 전력계로 기각 가능), "
        "H4는 AC 자속 진폭만큼의 DC 자속이 필요하다. ② 직렬 L이 1차에 있으면 여자전류는 코어 전압을 정하는 2차 bridge 쪽 권선이 공급한다 — 어느 권선에서 측정했는지 확인한다. "
        "③ DC offset은 AC 결합 probe·Rogowski에 보이지 않는다. ④ limit cycle은 짧은 캡처에서 정상 SPS 파형처럼 보인다 — 장시간 로그가 필요하다."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: 무부하 변압기 발열의 원인을 정해 무부하 손실을 줄인다. 확인한 조건: 전력·φ 명령 ≈ 0에서 AC 전류가 흐른다. 모르는 조건: 실제 n·V_L과 V_H, L_m, 실제 φ, DC 자속, 제어 로그. "
        "유력 가설과 반례: 900/600 V corner의 전압비 불일치면 φ = 0에서도 2.17 A RMS가 흐르는 것이 정상 물리다 — 평균 출력전류 0 A는 센서 고장의 증거가 아니다. 다만 작은 L_m, 실제 φ offset, DC 자속, limit cycle도 같은 RMS를 만든다. "
        "현재 권고: 같은 전압비에서 전류가 사라지는지 먼저 보고, 남으면 L_m 측정·DC 결합 probe·장시간 로그를 받는다. 부작용: 전압비 운전점이나 변조를 바꾸면 ZVS 여유와 경부하 효율이 함께 바뀐다. "
        "완료 판정: 원인 하나가 남고 그 원인을 바꾼 뒤 무부하 RMS와 변압기 온도가 함께 내려가면 닫는다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the circulating current at zero power. My current hypothesis is the voltage-ratio mismatch at the 900 to 600 V corner, which drives about 2.17 A rms even at zero phase shift, "
        "but magnetizing current, a real phase offset, a DC flux offset or a control limit cycle could produce a similar symptom. I would separate them by repeating the test at a matched ratio and, if current remains, measuring L_m, using a DC-coupled current probe and logging the phase command over many cycles. "
        "If the current disappears at a matched ratio, I recommend reviewing the turns ratio or the light-load modulation and would recheck ZVS margin and no-load loss."
    ),
    questions=[
        q_hyp(
            "C",
            "DAB가 P ≈ 0, φ ≈ 0인데 AC 전류가 남고 변압기가 뜨겁다. 가설 3개는?",
            "① 반사 전압비 불일치: φ = 0에서도 v_L = ±(V₁−V₂′) → 900/600 V에서 2.165 A RMS ② 여자전류(작은 L_m, gap) ③ 실제 φ offset(명령과 다른 gate 지연) ④ 자속 불균형/DC offset(한쪽 포화) ⑤ 제어 limit cycle(±φ 교번). "
            "평균 출력전류 0 A만 보고 센서 오동작으로 판단하지 않는다.",
            "A DAB at zero power command and zero phase still carries AC current and the transformer is hot: give three hypotheses.",
            "A reflected voltage-ratio mismatch, which drives 2.17 A rms at 900 to 600 V even at zero phase; magnetizing current from a low L_m; a real phase offset; flux imbalance with a DC offset; and a control limit cycle. A zero average output current does not mean the sensor is wrong.",
            ["전압비 불일치", "여자전류", "phase offset", "DC 자속/포화", "limit cycle", "평균 0 ≠ 센서 고장"],
        ),
        q_wave(
            "C",
            "① 1차·2차 권선 전류를 DC 결합 probe로(평균·peak 비대칭 확인) ② 두 bridge 전압(v₁, v₂′)과 실제 위상차를 같은 timebase로 ③ φ 명령·포트 전류의 장시간 로그(수백 주기). "
            "V_H, V_L, 온도, 측정 권선 위치를 함께 기록한다.",
            "Both winding currents with DC-coupled probes, the two bridge voltages and their actual phase difference on one timebase, and a long log of the phase command and port currents. Record V_H, V_L, temperature and which winding was measured.",
            ["권선 전류 DC 결합", "v₁·v₂′와 실제 위상", "장시간 로그", "V_H·V_L 기록"],
        ),
        q_first(
            "C",
            "같은 전압비(n·V_L = V_H)에서 다시 잰다: H1이면 순환전류가 사라지고 나머지는 남는다. 남으면 L_m 측정(H2: 작다), DC 결합 probe(H4: 평균·비대칭), φ sweep(H3: 최소점 이동), 장시간 로그(H5: 교번)로 가른다.",
            "Repeat at a matched voltage ratio. If the current disappears it was the mismatch; if it stays, measure L_m, use a DC-coupled probe, sweep the phase command and log over many cycles.",
            ["전압비 일치", "H1 사라짐", "L_m·DC probe·sweep·로그"],
        ),
        q_memo(
            "C",
            "목표: 무부하 발열 원인. 확인: 명령 ≈ 0에서 AC 전류. 모름: 실제 전압비·L_m·실제 φ·DC 자속·로그. 가설과 반례: 전압비 불일치(2.17 A, 정상 물리) vs 여자·φ offset·DC·limit cycle; 평균 0 A는 센서 고장 증거 아님. "
            "권고: 전압비 일치 시험 → 남으면 L_m·DC probe·로그. 부작용: 권선비·변조 변경은 ZVS·경부하 효율을 바꾼다. 완료 판정: 원인 제거 후 무부하 RMS와 온도가 함께 감소.",
            MEMO_EN_TEMPLATE.replace("___", "the no-load circulating current", 1),
        ),
    ],
    circuit="dab",
    textbook=[TB_15, TB_11, TB_10, TB_E12],
    reference_presets=["observed", "ratio", "dc"],
    claim_limit="이상 bridge·구간선형 정확 파형의 가설별 예측. 무부하 손실·온도와 고객 원인은 확정하지 않는다.",
)


EXP_D = Experiment(
    key="case_d",
    title="CASE D — LLC 경부하에서 낮은 R_DS(on) 소자가 더 뜨겁다: 둘을 동시에 바꾼 결과는 원인을 말해주지 않는다",
    goal=(
        "증상: 낮은 R_DS(on) 소자로 바꾼 뒤 전부하는 좋아졌는데 10 % 부하에서 소자가 더 뜨겁다. L_m도 동시에 늘렸다. 확인된 것: 두 가지를 한 번에 바꿨다는 사실, 전부하 개선과 경부하 악화(여기서는 소자당 +1.5 W로 환산했다고 가정). "
        "모르는 것: 새 소자의 실제 Q_oss(V), L_m을 얼마나 늘렸는지, gate-on 직전 V_DS, SR 타이밍, burst 조건, 실제 f_s. 먼저 Q1–Q3에 답을 적고 실행한다. "
        "실행하면 FL09 tank에서 V_o를 조절한 정류기 포함 주기해를 L_m 세 값·부하 세 점에서 풀고, ‘L_m만’, ‘C_oss만’, ‘둘의 상호작용’, ‘gate 전하만’ 네 세계를 같은 관측을 재현하도록 맞춘 뒤 "
        "L_m만 되돌리기·소자만 되돌리기·V_DS·edge 전류·SR 전류·f_s 로그가 각 세계에서 무엇을 보일지 비교한다. SR 역전류와 burst는 모델 밖(MISSING_INPUT)이다."
    ),
    params=params_d(),
    presets=[
        Preset("observed", "관측 그대로 (소자·L_m 동시 변경)", {}, "CASE D 증상", ("nominal", "reference")),
        Preset("restore_lm", "첫 시험: L_m만 원래로", {"test": "restore_lm"}, "H1·H12 해결, H2 남음", ("reference",)),
        Preset("restore_dev", "소자만 원래로", {"test": "restore_dev"}, "H2·H12 해결, H1 남음", ("variant", "reference")),
        Preset("vds", "gate-on 직전 V_DS", {"test": "vds"}, "ZVS 손실 가설군 대 나머지", ("variant",)),
        Preset("ir", "공진 전류 edge 값", {"test": "ir"}, "실제 L_m 변경 폭", ("variant",)),
        Preset("sr", "SR 전류 캡처", {"test": "sr"}, "H3", ("variant",)),
        Preset("fsw", "실제 f_s·burst 로그", {"test": "fsw"}, "H4", ("variant",)),
        Preset("big_obs", "관측 +3 W", {"dP_obs": 3.0}, "H1 세계는 재현 불가 (기존 설계도 ZVS를 잃었어야 한다)", ("corner",)),
    ],
    run=run_case_d,
    model_level="C + SCREEN (정류기 포함 주기해, EX02 전하 screen, 후처리 손실)",
    suggested_change="판별 시험을 ‘L_m만 원래로’로 바꾼다 (다음에 ‘소자만 원래로’, ‘gate-on 직전 V_DS’).",
    prediction=Prediction(
        "새 소자를 유지하고 L_m만 원래(200 µH)로 되돌리면 경부하 손실은 어느 세계에서 해결되나?",
        ["네 세계 모두", "L_m 세계(H1)와 상호작용 세계(H12)에서", "C_oss 세계(H2)에서만", "모르겠다"],
        "L_m 세계(H1)와 상호작용 세계(H12)에서",
        "L_m을 되돌리면 경부하 edge 전류가 약 3.05 A로 돌아온다. H1·H12 세계의 새 소자는 그 전류로 전환을 끝내지만(필요 약 2.96 A), H2 세계의 새 소자는 전하가 커서(필요 약 4.03 A) 기존 L_m에서도 부족하고, "
        "gate 전하 세계(H5)는 L_m과 무관하다. 그래서 ‘L_m을 되돌리니 해결’만으로는 H1과 H12가 갈리지 않는다 — 소자만 되돌리는 시험이 하나 더 필요하다.",
        ["cal_H1", "cal_H2", "cal_H12", "edge_old", "edge_new"],
        handcalc=[{"key": "Imin_old", "label": "기존 소자 2Q_oss/t_d", "unit": "A"}, {"key": "edge_new", "label": "L_m 400 µH 경부하 edge 전류", "unit": "A"}],
    ),
    suggested={"test": "restore_lm"},
    student=(
        "LLC의 ZVS는 dead time 동안 공진 전류(경부하에서는 주로 여자전류)가 소자의 출력 용량을 충·방전해 V_DS를 0으로 만든 뒤 켜는 것이다. L_m을 키우면 여자전류가 줄어 경부하에서는 전하가 모자랄 수 있고, "
        "die가 큰 저 R_DS(on) 소자는 옮겨야 할 전하(Q_oss)가 크다. 전부하에서는 부하 전류가 충분해 ZVS가 유지되고 낮은 R_DS(on)의 전도손실 이득이 이긴다. 그런데 두 가지를 한 번에 바꿨으므로 이 결과만으로는 어느 쪽 때문인지 알 수 없다."
    ),
    expert=(
        "교재 해설(CASE D): L_m 증가에 따른 commutation current 감소와 ZVS 손실을 우선 확인하되 C_oss/Q_oss 차이, SR reverse current, burst 조건 변화도 남긴다. gate-on 직전 V_DS, resonant current, SR current, 실제 switching frequency를 취득한다. "
        "한 번에 소자와 L_m을 모두 바꾼 결과만으로 causal attribution을 하지 않는다. "
        "추가: ① 같은 관측을 ‘L_m만’(두 소자의 C_oss가 같다), ‘C_oss만’(L_m 증가가 작다), ‘상호작용’ 세계가 모두 재현한다 — 대각선 한 칸(기존·기존 → 새·새)만으로는 주효과와 상호작용을 나눌 수 없다(2×2 요인 설계). "
        "② L_m만 되돌리는 시험과 소자만 되돌리는 시험의 결과 조합이 세 세계를 가른다: H1은 L_m만 되돌려 해결, H2는 소자만 되돌려 해결, H12는 어느 쪽이든 해결. "
        "③ gate-on 직전 V_DS는 ZVS 손실 가설군(H1·H2·H12)과 SR·gate 가설을 가르지만 그 셋 사이는 가르지 못한다. ④ 공진 전류의 edge 값과 실제 f_s는 실제 L_m 변경 폭을 드러낸다. "
        "⑤ gate 전하만으로 설명하려면 Q_g가 수백 nC여야 하고 전부하도 악화되어야 한다 — 크기와 부호가 가설을 거른다. ⑥ dead time을 늘리는 임시조치는 전하를 늘리지만 유효 duty와 body diode 도통 시간을 바꾼다."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: 경부하 소자 온도 상승의 원인을 정하고 전부하 개선은 유지한다. 확인한 조건: 저 R_DS(on) 소자와 큰 L_m을 동시에 적용했고, 전부하는 개선, 10 % 부하는 악화됐다. "
        "모르는 조건: 새 소자의 실제 Q_oss(V), L_m 변경 폭, gate-on 직전 V_DS, SR 타이밍, burst 조건. 유력 가설과 반례: L_m 증가로 여자전류가 줄어 부분 ZVS가 된 것이 우선 후보지만, 새 소자의 큰 Q_oss나 두 변경의 상호작용도 같은 결과를 만든다 — "
        "두 가지를 같이 바꾼 결과는 어느 쪽의 증거도 아니다. 현재 권고: L_m만 되돌린 시료와 소자만 되돌린 시료를 같은 조건(10 % 부하, 같은 냉각)에서 비교하고 gate-on 직전 V_DS·공진 전류·SR 전류·실제 f_s를 함께 받는다. "
        "dead time을 늘리는 임시조치는 잔류 V_DS를 먼저 확인한 뒤에 한다. 부작용: L_m을 되돌리면 순환전류와 전부하 전도손실이 다시 늘고, 소자를 되돌리면 전부하 이득을 잃는다. "
        "완료 판정: 원인이 하나로 좁혀지고, 그 대책 뒤 10 % 부하의 잔류 V_DS ≈ 0과 온도 회복을 같은 측정으로 확인하면 닫는다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the charge available for zero-voltage switching at 10 % load. My current hypothesis is that the larger L_m reduced the magnetizing current at the switching edge, "
        "but the larger output charge of the new device, or the combination of both changes, could produce a similar symptom; SR reverse current and a change in burst behaviour are still open. "
        "I would separate them by testing one change at a time, the new device with the original L_m and the original device with the new L_m, and by measuring V_DS just before gate turn-on, the resonant current, the SR current and the actual switching frequency. "
        "If the residual V_DS disappears with the original L_m only, I recommend reconsidering L_m or the light-load dead time and would recheck full-load efficiency."
    ),
    questions=[
        q_hyp(
            "D",
            "전부하는 개선, 10 % 부하에서 저 R_DS(on) 소자가 더 뜨겁다. L_m도 동시에 늘렸다. 원인 가설 3개는?",
            "① L_m 증가 → edge 여자전류 감소 → 부분 ZVS(잔류 turn-on 손실) ② 새 소자의 C_oss/Q_oss 증가로 같은 전류에서 전환 미완료 ③ 두 변경의 상호작용(각각은 괜찮고 겹치면 부족) ④ 경부하 SR 역전류 ⑤ L_m 변경에 따른 burst 진입·패턴 변화. "
            "두 가지를 한 번에 바꾼 결과만으로는 원인을 하나로 돌릴 수 없다는 점을 함께 말한다.",
            "Full load improved, but at 10 % load the low-R_DS(on) device runs hotter, and L_m was raised at the same time: give three hypotheses.",
            "Lower magnetizing current at the edge after the L_m increase, giving partial ZVS; the larger output charge of the new device; an interaction of both; SR reverse current at light load; and a change of burst behaviour. Changing both at once does not allow attributing the cause to either.",
            ["L_m → edge 전류 → ZVS", "C_oss/Q_oss", "상호작용", "SR 역전류", "burst", "동시 변경은 원인 귀속 불가"],
        ),
        q_wave(
            "D",
            "① 10 % 부하에서 gate-on 직전 V_DS(gate 신호보다 먼저 0에 닿는지) ② 공진 전류(edge 순간 값)와 1차 bridge 전압 ③ SR 전류(음의 구간)와 실제 f_s·burst 패턴(장시간). "
            "부하·V_in·V_o·온도·dead time 설정과 probe deskew를 함께 기록한다.",
            "V_DS just before gate turn-on at 10 % load; the resonant current at the switching edge with the bridge voltage; and the SR current with the actual switching frequency and burst pattern over a long record. Record load, input and output voltage, temperature, dead-time setting and probe deskew.",
            ["gate-on 직전 V_DS", "공진 전류 edge 값", "SR 전류", "실제 f_s·burst"],
        ),
        q_first(
            "D",
            "한 번에 하나만 바꾼다: 새 소자 + 기존 L_m, 기존 소자 + 새 L_m을 같은 10 % 부하·냉각에서 비교한다. H1이면 L_m만 되돌렸을 때 해결되고 소자만 되돌리면 그대로, H2이면 반대, 상호작용이면 어느 쪽이든 해결된다. "
            "같은 시료에서 gate-on 직전 V_DS를 보면 ZVS 손실 가설군(잔류 V_DS 있음)인지 SR·burst 쪽인지도 갈린다.",
            "Change one thing at a time: the new device with the original L_m, and the original device with the new L_m, at the same 10 % load and cooling. If restoring L_m alone fixes it, L_m is the driver; if restoring the device alone fixes it, the device charge is; if either fixes it, it is the interaction. "
            "Measuring V_DS before turn-on on the same samples separates the ZVS family from SR and burst causes.",
            ["한 번에 하나", "L_m만 되돌림", "소자만 되돌림", "결과 조합으로 H1/H2/상호작용", "V_DS 동시 확인"],
        ),
        q_memo(
            "D",
            "목표: 경부하 발열 원인 확정, 전부하 이득 유지. 확인: 소자·L_m 동시 변경, 전부하 개선·경부하 악화. 모름: 실제 Q_oss(V), L_m 폭, V_DS, SR 타이밍, burst. "
            "가설과 반례: L_m → 부분 ZVS가 유력하지만 Q_oss·상호작용도 같은 결과 — 동시 변경은 어느 쪽의 증거도 아님; gate 전하만으로는 크기·부호가 맞지 않음. "
            "권고: 한 번에 하나씩 되돌린 두 시료 + V_DS·공진 전류·SR 전류·f_s 동시 취득, dead time 연장은 잔류 V_DS 확인 후. 부작용: L_m 원복은 순환전류·전부하 손실 증가, 소자 원복은 전부하 이득 상실. "
            "완료 판정: 원인 하나, 대책 후 10 % 부하 잔류 V_DS ≈ 0과 온도 회복.",
            MEMO_EN_TEMPLATE.replace("___", "the ZVS charge at 10 % load", 1),
        ),
    ],
    circuit="llc",
    textbook=[TB_15, TB_12, TB_E02, TB_17],
    reference_presets=["observed", "restore_lm", "restore_dev"],
    runtime_hint="seconds",
    claim_limit="정류기 포함 주기해 + 전하 screen으로 가설별 예측을 비교한다. 고객 소자의 손실·온도와 근본원인을 확정하지 않는다.",
)


# ======================================================================================
# CASE E - transformer saturation suspected at start-up (primary current grows to one side after enable)
# ======================================================================================

E_TESTS = [
    ("none", "시험 전: 고객 전류 센서"),
    ("vs", "권선 volt-second + 전류 기울기"),
    ("gate", "gate 명령 + bridge 출력 (deskew)"),
    ("cb", "C_b 전압 (기동 전·정상상태)"),
    ("probe", "두 번째 DC 결합 전류 probe"),
    ("demag", "소거(demag) 후 재기동"),
]
E_REG = {-1: "sat−", 0: "lin", 1: "sat+"}
E_MODE_LABEL = {"lin": "선형", "sat+": "포화 (+)", "sat−": "포화 (−)"}


class SatBridge(HybridSystem):
    """Full bridge (+-V) -> series R -> blocking C_b -> winding with a piecewise-linear saturating magnetizing branch.

    x = [i, v_Cb] (i = magnetizing = primary current at no load); q = (b, r): b = +1/-1 the bridge output,
    r = -1/0/+1 the region of the magnetizing curve (|i| <= I_s: L_m, beyond: L_sat), switched by guards at
    i = +-I_s.  Bridge timing: cycle k's + interval is [kT - T/4, kT + T/4 + d_k/2) and the - interval runs to
    (k+1)T - T/4; the enable is t = 0, so the first + pulse is half wide (the usual symmetric start).
    """

    state_names = ("i", "v_Cb")
    state_units = ("A", "V")

    def __init__(self, V, fs, R, Cb, Lm, Ls, Is, N, Ae, d_of_k):
        self.V, self.fs, self.R, self.Cb, self.Lm, self.Ls, self.Is, self.N, self.Ae = V, fs, R, Cb, Lm, Ls, Is, N, Ae
        self.T = 1.0 / fs
        self.d_of_k = d_of_k
        self.key = f"fl12e|{V:.12g}|{fs:.12g}|{R:.12g}|{Cb:.12g}|{Lm:.12g}|{Ls:.12g}"

    def mode(self, q):
        b, r = q
        L = self.Lm if r == 0 else self.Ls
        A = np.array([[-self.R / L, -1.0 / L], [1.0 / self.Cb, 0.0]])
        return AffineMode(f"{self.key}|{b}|{r}", A, np.array([b * self.V / L, 0.0]), label=self.describe(q))

    def guards(self, q):
        r = q[1]
        Is = self.Is
        if r == 0:
            return [Guard("i = +I_s (포화 진입)", np.array([1.0, 0.0, -Is]), +1, lambda qq: (qq[0], 1)), Guard("i = −I_s (포화 진입)", np.array([1.0, 0.0, Is]), -1, lambda qq: (qq[0], -1))]
        if r == 1:
            return [Guard("i = +I_s (선형 복귀)", np.array([1.0, 0.0, -Is]), -1, lambda qq: (qq[0], 0))]
        return [Guard("i = −I_s (선형 복귀)", np.array([1.0, 0.0, Is]), +1, lambda qq: (qq[0], 0))]

    def edges(self, k: int) -> tuple[float, float]:
        T = self.T
        return k * T - 0.25 * T, k * T + 0.25 * T + 0.5 * self.d_of_k(k)

    def gate_schedule(self, t0, t1):
        ev = []
        k0 = max(int(math.floor(t0 / self.T)) - 1, 0)
        k1 = int(math.ceil(t1 / self.T)) + 1
        for k in range(k0, k1 + 1):
            tr, tf = self.edges(k)
            if k >= 1 and t0 - 1e-15 <= tr < t1:
                ev.append((tr, lambda q: (1, q[1])))
            if t0 - 1e-15 <= tf < t1:
                ev.append((tf, lambda q: (-1, q[1])))
        return sorted(ev, key=lambda e: e[0])

    def outputs(self, q):
        b, r = q
        lam = np.array([self.Lm, 0.0, 0.0]) if r == 0 else np.array([self.Ls, 0.0, r * (self.Lm - self.Ls) * self.Is])
        return {"i": np.array([1.0, 0.0, 0.0]), "vcb": np.array([0.0, 1.0, 0.0]), "vb": np.array([0.0, 0.0, b * self.V]),
                "vw": np.array([-self.R, -1.0, b * self.V]), "lam": lam, "B": lam / (self.N * self.Ae)}

    def powers(self, q):
        b, _ = q
        Pin = np.zeros((3, 3))
        Pin[0, 2] = Pin[2, 0] = 0.5 * b * self.V
        PR = np.zeros((3, 3))
        PR[0, 0] = self.R
        return {"p_in": Pin, "p_R": PR}

    def describe(self, q):
        return ("+" if q[0] > 0 else "−") + "|" + E_REG[q[1]]

    # piecewise-linear magnetizing curve
    def lam_of_i(self, i: float) -> float:
        if abs(i) <= self.Is:
            return self.Lm * i
        return math.copysign(self.Lm * self.Is + self.Ls * (abs(i) - self.Is), i)

    def i_of_lam(self, lam: float) -> float:
        ls = self.Lm * self.Is
        if abs(lam) <= ls:
            return lam / self.Lm
        return math.copysign(self.Is + (abs(lam) - ls) / self.Ls, lam)

    def W_mag(self, i: float) -> float:
        """Magnetic energy int i dlam of the piecewise-linear curve."""
        if abs(i) <= self.Is:
            return 0.5 * self.Lm * i * i
        return 0.5 * self.Lm * self.Is**2 + 0.5 * self.Ls * (i * i - self.Is**2)

    def region(self, i: float) -> int:
        return 0 if abs(i) <= self.Is else (1 if i > 0 else -1)


@dataclass
class WorldE:
    key: str
    lam0: float = 0.0  # flux linkage at enable (residual flux)
    vcb0: float = 0.0  # blocking-capacitor voltage at enable
    d_cmd: float = 0.0  # command asymmetry of the first K pulses (+ wider than -)
    d_out: float = 0.0  # bridge-output asymmetry with symmetric commands (dead-time / delay imbalance)
    drift: float = 0.0  # sensor offset settling after the enable (A)
    ok: bool = True


def e_consts(v: dict) -> dict:
    Ae = v["Ae"] * 1e-6
    Is = v["Bs"] * v["N"] * Ae / v["Lm"]
    return {"Ae": Ae, "Is": Is, "Ls": v["Lm"] / v["k_sat"], "T": 1.0 / v["fs"], "lam_s": v["Lm"] * Is, "K": int(v["K_soft"])}


def sys_e(v: dict, w: WorldE, R: float | None = None, Cb: float | None = None) -> SatBridge:
    c = e_consts(v)
    K = c["K"]
    return SatBridge(v["V"], v["fs"], v["R"] if R is None else R, v["Cb"] if Cb is None else Cb, v["Lm"], c["Ls"], c["Is"], v["N"], c["Ae"],
                     lambda k: (w.d_cmd if k < K else 0.0) + w.d_out)


def run_e(v: dict, w: WorldE, cycles: int, demag: bool = False, discharge: bool = False):
    s = sys_e(v, w)
    lam0 = 0.0 if demag else w.lam0
    vcb0 = 0.0 if discharge else w.vcb0
    i0 = s.i_of_lam(lam0)
    return s, simulate(s, (1, s.region(i0)), [i0, vcb0], 0.0, cycles * s.T)


def drift_e(v: dict, w: WorldE, t):
    return w.drift * (1.0 - np.exp(-np.asarray(t, dtype=float) / v["tau_s"]))


def cycle_peaks(v: dict, w: WorldE, s: SatBridge, tr, cycles: int) -> dict:
    """Per-cycle + peak (sensor and true) and - peak; cycle k spans [kT - T/4, (k+1)T - T/4)."""
    T = s.T
    pk_m, pk_t, mn_t, t_pk = np.full(cycles, -np.inf), np.full(cycles, -np.inf), np.full(cycles, np.inf), np.zeros(cycles)
    for seg in tr.segments:
        k = int(math.floor((seg.t1 + 0.25 * T) / T - 1e-9))
        if not 0 <= k < cycles:
            continue
        i = float(seg.z1[0])
        im = i + float(drift_e(v, w, seg.t1))
        if im > pk_m[k]:
            pk_m[k], t_pk[k] = im, seg.t1
        pk_t[k] = max(pk_t[k], i)
        mn_t[k] = min(mn_t[k], i)
    return {"meas": pk_m, "true": pk_t, "min": mn_t, "t": t_pk}


def imax_e(v: dict, w: WorldE, cycles: int, **kw) -> float:
    s, tr = run_e(v, w, cycles, **kw)
    return float(np.max(cycle_peaks(v, w, s, tr, cycles)["meas"]))


def calibrate_e(v: dict, key: str, cycles: int) -> tuple[WorldE, str]:
    c = e_consts(v)
    target = v["I_obs"]
    T = c["T"]
    spec = {
        "H1": (lambda x: WorldE("H1", lam0=x), 0.0, c["lam_s"]),
        "H2": (lambda x: WorldE("H2", d_cmd=x), 0.0, 0.25 * T),
        "H3": (lambda x: WorldE("H3", vcb0=-x), 0.0, v["V"]),
        "H4": (lambda x: WorldE("H4", d_out=x), 0.0, 0.25 * T),
    }
    if key == "H5":
        s, tr = run_e(v, WorldE("H5"), cycles)

        def f5(x):
            w = WorldE("H5", drift=x)
            return float(np.max(cycle_peaks(v, w, s, tr, cycles)["meas"])) - target

        if f5(0.0) * f5(10 * target) > 0:
            return WorldE("H5", ok=False), "재현 불가"
        x = brentq(f5, 0.0, 10 * target, xtol=1e-14, rtol=1e-13)
        return WorldE("H5", drift=x), f"{x:.4g} A"
    make, lo, hi = spec[key]
    f = lambda x: imax_e(v, make(x), cycles) - target  # noqa: E731
    if f(lo) * f(hi) > 0:
        return WorldE(key, ok=False), "재현 불가"
    x = brentq(f, lo, hi, xtol=1e-14 * (hi - lo), rtol=1e-12)
    return make(x), ""


def ode_e(v: dict, w: WorldE, cycles: int) -> tuple[float, np.ndarray]:
    """Independent path: flux coordinates [lam, v_Cb] with the continuous PWL i(lam), DOP853 between the bridge edges."""
    s = sys_e(v, w)
    T = s.T
    t_end = cycles * T
    ev = [(t, f((0, 0))[0]) for t, f in s.gate_schedule(0.0, t_end)]
    pts = [(0.0, 1)] + ev + [(t_end, None)]
    y = np.array([w.lam0, w.vcb0], dtype=float)
    imax = -np.inf
    Cb, R, V = v["Cb"], v["R"], v["V"]
    for (ta, b), (tb, _) in zip(pts[:-1], pts[1:]):
        if tb <= ta:
            continue

        def rhs(t, yy, b=b):
            i = s.i_of_lam(yy[0])
            return [b * V - R * i - yy[1], i / Cb]

        sol = solve_ivp(rhs, (ta, tb), y, method="DOP853", rtol=1e-12, atol=[1e-14, 1e-11])
        y = sol.y[:, -1]
        if b == 1:
            imax = max(imax, s.i_of_lam(y[0]) + float(drift_e(v, w, tb)))
    return imax, np.array([s.i_of_lam(y[0]), y[1]])


def ledger_e(s: SatBridge, tr, t0: float, t1: float, i_scale: float) -> dict:
    E_in = tr.energy(t0, t1, "p_in")
    E_R = tr.energy(t0, t1, "p_R")
    za, _ = tr.state_at(t0)
    zb, _ = tr.state_at(t1)

    def W(z):
        return s.W_mag(float(z[0])) + 0.5 * s.Cb * float(z[1]) ** 2

    dW = W(zb) - W(za)
    resid = E_in - E_R - dW
    norm = max(abs(E_in), abs(E_R), s.W_mag(i_scale))
    return {"E_in": E_in, "E_out": 0.0, "E_loss": E_R, "dW": dW, "residual": resid, "normalised": resid / norm, "norm": norm}


def periodic_e(v: dict, d_out: float):
    from ..engine.periodic import shoot

    s = sys_e(v, WorldE("ss", d_out=d_out))
    T = s.T
    guess = [-v["V"] / (4.0 * v["fs"] * v["Lm"]), v["V"] * d_out * v["fs"]]
    per = shoot(s, (1, 0), guess, T, scales=[s.Is, v["V"]], tol=1e-11, t0=0.75 * T)
    return s, per


def circuit_e(v: dict) -> Circuit:
    c = Circuit("fl12_e", 700, 290, title="Full bridge → R → C_b → 변압기 (자화 가지가 포화), enable 직후")
    vb = c.add("vsource", "VB", 70, 150, 90, "bridge 출력", f"±{v['V']:g} V", lpos=(96, 146, "start"))
    rr = c.add("resistor", "R", 180, 60, 0, "R (권선+채널)", f"{v['R'] * 1e3:g} mΩ", lpos=(180, 24, "middle"))
    cb = c.add("capacitor", "Cb", 300, 60, 0, "C_b (blocking)", f"{v['Cb'] * 1e6:g} µF", lpos=(300, 24, "middle"))
    lm = c.add("inductor", "Lm", 420, 150, 90, "L_m (PWL 포화)", f"{v['Lm'] * 1e3:g} mH", lpos=(398, 146, "end"))
    tx = c.add("transformer", "TX", 520, 150, 0, "", "")
    c.add("block", "LD", 640, 150, 0, "2차 (무부하)", w=90, h=50)
    c.wire("w_a", vb["a"], (70, 60), rr["a"])
    c.wire("w_rc", rr["b"], cb["a"])
    c.wire("w_cb", cb["b"], (420, 60), lm["a"])
    c.wire("w_p1", (420, 60), (498, 60), tx["p1"])
    c.wire("w_ret", lm["b"], (420, 240), (70, 240), vb["b"])
    c.wire("w_p2", tx["p2"], (498, 240), (420, 240))
    c.wire("w_s1", tx["s1"], (595, 120))
    c.wire("w_s2", tx["s2"], (595, 180))
    c.dot((420, 60), (420, 240))
    c.text(520, 104, "이상 변압기", "node")
    c.text(452, 205, "v_w", "node")
    c.text(250, 270, "센서: i_p = i_m (무부하) — H5 세계는 센서 offset이 더해진다", "note")
    c.probe("pI", "wv_H4", 250, 240, "left", "i_p (H4 세계)")
    loop = ["VB", "w_a", "R", "w_rc", "Cb", "w_cb", "Lm", "w_ret"]
    for b in ("+", "−"):
        for r in ("lin", "sat+", "sat−"):
            sat = r != "lin"
            txt = (f"bridge {b}V: 권선 전압 {b}V − R·i − v_Cb가 자속을 {'올린다' if b == '+' else '내린다'}. "
                   + ("자속이 λ_s를 넘어 증분 인덕턴스가 L_m/k로 떨어졌다 — 같은 전압에 전류 기울기가 k배라 센서에는 ‘한쪽으로 커지는’ 전류로 보인다." if sat else "선형 구간: 전류 기울기 = 권선 전압 / L_m."))
            c.mode(f"{b}|{r}", f"v_b = {b}V, {E_MODE_LABEL[r]}", loop, txt)
    return c


def bands_e(tr, t0: float, t1: float, t_off: float = 0.0) -> list[dict]:
    out = []
    for iv in tr.mode_intervals(t0, t1):
        b, r = iv["mode"].split("|")
        out.append({"x0": iv["t0"] - t_off, "x1": iv["t1"] - t_off, "mode": iv["mode"], "label": f"{b}V · {E_MODE_LABEL[r]}"})
    return out


def run_case_e(v: dict) -> Result:
    from ._common import ledger_check

    res = Result("FL12", "case_e", "C (정확 스위칭: full bridge + R + C_b + PWL 포화 자화 가지, 영역 guard) + 측정 경로")
    c = e_consts(v)
    M, T, Is, K = int(v["M_obs"]), c["T"], c["Is"], c["K"]
    NA = v["N"] * c["Ae"]
    worlds: dict[str, WorldE] = {}
    notes = {}
    for hk in ("H1", "H2", "H3", "H4", "H5"):
        worlds[hk], notes[hk] = calibrate_e(v, hk, M)
    hyps = [
        Hyp("H1", "초기·잔류 자속", "이전 정지에서 남은 자속에서 출발한다", "B_r"),
        Hyp("H2", "비대칭 펄스 (기동 순서)", f"soft-start 처음 {K}펄스의 + 폭이 − 폭보다 길다 (명령 자체가 비대칭)", "Δt_a"),
        Hyp("H3", "C_b 초기 전압", "blocking capacitor가 충전된 채 enable된다", "V_Cb0"),
        Hyp("H4", "dead time·지연 불균형", "명령은 대칭인데 두 leg의 dead time·지연이 달라 출력 펄스 폭이 다르다 (C_b가 정상상태에서 흡수)", "Δt_d"),
        Hyp("H5", "전류 측정 offset", "실제 전류는 대칭이고 센서 offset이 enable 뒤 τ_s로 자리 잡는다", "I_os"),
    ]
    runs = {}
    for hk, w in worlds.items():
        if not w.ok:
            continue
        s, tr = run_e(v, w, M)
        runs[hk] = {"s": s, "tr": tr, "pk": cycle_peaks(v, w, s, tr, M)}
    # periodic steady state (zero-DC flux): what a periodic model would show
    _, per0 = periodic_e(v, 0.0)
    pk_ss = max(float(seg.z1[0]) for seg in per0.trajectory.segments)
    d4 = worlds["H4"].d_out if worlds["H4"].ok else 0.0
    _, per4 = periodic_e(v, d4)
    vcb_ss4 = per4.trajectory.mean(per4.trajectory.segments[0].t0, per4.trajectory.t_end, "vcb")
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(v["Lm"] * v["Cb"]))
    # metrics
    unit_of = {"H1": "T", "H2": "s", "H3": "V", "H4": "s", "H5": "A"}
    val_of = {"H1": lambda w: w.lam0 / NA, "H2": lambda w: w.d_cmd, "H3": lambda w: -w.vcb0, "H4": lambda w: w.d_out, "H5": lambda w: w.drift}
    lab_of = {"H1": "H1 세계: 잔류 자속 B_r", "H2": f"H2 세계: 처음 {K}펄스의 폭 차이 Δt_a", "H3": "H3 세계: C_b 초기 전압 (권선에 + volt-second를 주는 극성)", "H4": "H4 세계: 출력 펄스 폭 차이 Δt_d", "H5": "H5 세계: 센서 offset (정착값)"}
    for hk in worlds:
        w = worlds[hk]
        res.add_metric(f"cal_{hk}", lab_of[hk] + f" (관측 {v['I_obs']:g} A 재현)", val_of[hk](w) if w.ok else float("nan"), unit_of[hk], basis="보정값 (합성 세계)" if w.ok else "이 입력에서 재현 불가")
    res.add_metric("Is", "포화 시작 전류 I_s = B_s·N·A_e/L_m", Is, "A", basis=f"B_s {v['Bs']:g} T (가정)")
    res.add_metric("Ipk_ss", "주기해(대칭 정상상태)의 peak 자화전류", pk_ss, "A", basis="shooting: 매 주기 같은 대칭 자속 — 기동 과도가 없다")
    res.add_metric("Bpk_sym", "대칭 구동의 B_pk = V/(4 f N A_e)", v["V"] / (4.0 * v["fs"] * NA), "T", ref=0.16 if abs(v["V"] - 800) < 1e-9 and abs(v["fs"] - 1e5) < 1e-6 and abs(NA - 50 * 250e-6) < 1e-15 else None,
                   ref_label="FL07 flux_walk@textbook Bpk_sym 0.16 T", tol=1e-12)
    if worlds["H4"].ok:
        res.add_metric("walk_H4", "H4의 Δt_d가 만드는 주기당 자속 이동 V·Δt_d/(N A_e)", v["V"] * d4 / NA, "T", basis="FL07 교재 예: 100 ns → 0.0064 T")
        res.add_metric("vcb_ss_H4", "H4 정상상태 C_b 평균 전압 (shooting)", vcb_ss4, "V", ref=v["V"] * d4 * v["fs"], ref_label="DC 균형 V·Δt_d·f", tol=1e-8, basis="C_b가 비대칭 DC를 흡수해 자속은 대칭")
    res.add_metric("f0", "L_m–C_b 직렬 공진", f0, "Hz", basis=f"1/4 주기 = {0.25 / f0 * v['fs']:.1f} 스위칭 주기 (C_b 경로 세계의 peak 시점)")
    for hk, r in runs.items():
        kpk = int(np.argmax(r["pk"]["meas"]))
        res.add_metric(f"kpk_{hk}", f"{hk} 세계: 센서 peak가 나오는 주기", kpk, "", basis=f"음의 peak 최저 {np.min(r['pk']['min']):.3g} A (실제 전류)")
    # one-cycle walk in this model with R = 0 and no C_b (FL07 textbook case): independent of FL07's own system class
    sw = sys_e(v, WorldE("w100", d_out=v["dt_ref"]), R=0.0, Cb=1e6)
    lam_a = -v["V"] / (4.0 * v["fs"])
    trw = simulate(sw, (1, 0), [lam_a / v["Lm"], 0.0], 0.75 * T, 1.75 * T)
    walk100 = (sw.lam_of_i(float(trw.z_end[0])) - lam_a) / NA
    tbw = abs(v["V"] - 800) < 1e-9 and abs(v["dt_ref"] - 100e-9) < 1e-18 and abs(NA - 50 * 250e-6) < 1e-15
    res.add_metric("walk_100", f"{v['dt_ref'] * 1e9:g} ns 비대칭의 주기당 자속 이동 (이 모델, R = 0, C_b 없음)", walk100, "T", ref=0.0064 if tbw else v["V"] * v["dt_ref"] / NA, ref_label="교재 800 V × 100 ns/(50·250 mm²) = 0.0064 T (FL07)" if tbw else "V·Δt/(N A_e)", tol=1e-9)
    # calibration table
    plaus = {"H1": "재료 B_r·gap 자료로 크기 확인 (gap이 있으면 작다)", "H2": "soft-start 코드·PWM 갱신 시점 확인", "H3": "정지 시 C_b 방전 경로(bleed) 확인", "H4": "leg별 dead time·driver 지연 실측", "H5": "센서 offset·온도 drift 사양 확인"}
    rows = []
    for hk in worlds:
        w = worlds[hk]
        if not w.ok:
            rows.append([hk, "재현 불가", "—", "—", "—", plaus[hk]])
            continue
        r = runs[hk]
        val = val_of[hk](w)
        vtxt = {"H1": f"{val:.4g} T", "H2": f"{val * 1e9:.4g} ns × {K}펄스", "H3": f"{val:.4g} V", "H4": f"{val * 1e9:.4g} ns (매 주기)", "H5": f"{val:.4g} A, τ_s {v['tau_s'] * 1e6:g} µs"}[hk]
        rows.append([hk, vtxt, f"{np.max(r['pk']['meas']):.4g}", str(int(np.argmax(r["pk"]["meas"]))), f"{np.max(r['pk']['true']):.4g}", plaus[hk]])
    res.tables.append(Table("t_cal", f"세계별 보정: 처음 {M}주기 안에 센서 + peak {v['I_obs']:g} A (관측)", ["세계", "보정한 미지수", "센서 최대 [A]", "peak 주기", "실제 전류 최대 [A]", "크기 검토"], rows,
                            note="각 세계의 미지수 하나를 brentq로 맞춰 같은 관측을 재현했다. 값은 합성 세계의 것이고 고객 회로 값이 아니다 — 크기가 그럴듯한지가 첫 걸러내기다."))
    # test predictions
    pred = {}
    for hk, w in worlds.items():
        if not w.ok:
            continue
        r = runs[hk]
        s, tr = r["s"], r["tr"]
        lam_c = [s.lam_of_i(float(tr.state_at(min(max(k * T - 0.25 * T, 0.0), M * T))[0][0])) for k in range(M + 1)]
        dB = np.diff(lam_c) / NA
        sat = bool(np.max(r["pk"]["true"]) > Is)
        im_demag = imax_e(v, w, M, demag=True)
        im_dis = imax_e(v, w, M, discharge=True)
        pred[hk] = {"dB": dB, "sat": sat, "im_demag": im_demag, "im_dis": im_dis, "ref_max": float(np.max(r["pk"]["true"]))}
    cl = {}
    for hk, w in worlds.items():
        if not w.ok:
            cl[hk] = {t: ("이 입력에서 재현 불가", "재현 불가") for t, _ in E_TESTS}
            continue
        p = pred[hk]
        dB0 = float(np.mean(p["dB"][1:4])) * 1e3
        if not p["sat"]:
            vs = (f"주기당 ΔB {dB0:+.2g} mT, 기울기 꺾임 없음 (포화 없음)", "곡률 없음")
        elif hk == "H1":
            vs = (f"첫 펄스부터 기울기 꺾임, 처음 주기 ΔB {dB0:+.2g} mT (거의 균형)", "균형 + 첫 펄스 포화")
        elif hk == "H2":
            vs = (f"처음 {K}주기 ΔB {dB0:+.2g} mT/주기, 이후 균형 + 꺾임", f"처음 {K}주기 불균형")
        else:
            vs = (f"ΔB {dB0:+.2g} mT/주기로 시작해 C_b 공진으로 감소 + 꺾임", "불균형 (C_b 공진으로 감소)")
        gate = {"H2": (f"명령·출력 모두 처음 {K}펄스 +{w.d_cmd * 1e9:.3g} ns", "명령 비대칭"), "H4": (f"명령 대칭, 출력만 +{w.d_out * 1e9:.3g} ns", "출력만 비대칭")}.get(hk, ("명령·출력 대칭", "대칭"))
        if hk == "H3":
            cb = (f"기동 전 {w.vcb0:+.3g} V, 정상상태 0 V; 방전 후 재기동 {p['im_dis']:.3g} A", "기동 전 전압")
        elif hk == "H4":
            cb = (f"기동 전 0 V, 정상상태 {vcb_ss4:+.3g} V (DC 흡수)", "정상상태 DC")
        else:
            cb = ("기동 전·정상상태 ≈ 0 V", "0")
        probe = (f"기준 probe 최대 {p['ref_max']:.3g} A (센서와 다름)", "센서만 증가") if hk == "H5" else (f"기준 probe도 {p['ref_max']:.3g} A", "둘 다 증가")
        demag = (f"재기동 최대 {p['im_demag']:.3g} A", "사라짐" if p["im_demag"] < 0.6 * v["I_obs"] else "그대로")
        none = (f"센서 최대 {np.max(runs[hk]['pk']['meas']):.3g} A, peak {int(np.argmax(runs[hk]['pk']['meas']))}주기", "관측 재현")
        cl[hk] = {"none": none, "vs": vs, "gate": gate, "cb": cb, "probe": probe, "demag": demag}
    groups = disc_tables(res, hyps, E_TESTS, cl, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설의 세계가 예측하는 결과",
                         note="모든 세계가 ‘처음 수십 주기 안에 + peak 4 A’라는 같은 관측을 재현하도록 미지수 하나를 맞췄다. 권선 volt-second와 전류 기울기를 함께 보면 포화 여부(H5 분리)와 불균형의 시간 모양이 보이고, "
                         "gate 명령·bridge 출력 비교가 H2·H4를, C_b 전압이 H3·H4를, 소거 후 재기동이 H1을 가른다.")
    # plots ------------------------------------------------------------------------------
    ks = list(range(M))
    for hk, r in runs.items():
        res.add_series(f"pk_{hk}", f"{hk} {next(h.label for h in hyps if h.key == hk)}", "A", ks, r["pk"]["meas"].tolist())
    res.add_plot("p_env", f"기동 후 주기별 + peak (고객 센서): 다섯 세계가 모두 {v['I_obs']:g} A를 만든다", [f"pk_{hk}" for hk in runs], x_label="enable 후 주기", x_unit="", y_label="+ peak", y_unit="A", kind="xy", level="C",
                 hlines=[{"y": v["I_obs"], "label": f"관측 {v['I_obs']:g} A"}, {"y": Is, "label": f"포화 시작 I_s {Is:.3g} A"}, {"y": pk_ss, "label": f"주기해 peak {pk_ss:.3g} A"}],
                 proved="같은 관측(처음 수십 주기 안에 센서 + peak 4 A)을 다섯 세계가 모두 재현한다는 것과, peak가 나오는 시점이 세계마다 다르다(H1 첫 펄스, H2 soft-start 끝, H3·H4 L_m–C_b 공진 1/4 주기 — 두 곡선은 거의 겹친다, H5 창 끝)는 것을 보였다 "
                 "(독립 DOP853 자속좌표 적분·에너지 원장·보정 잔차 check).",
                 not_yet="PWL 포화 곡선·잔류자속·센서 drift는 합성 값이다. 어느 세계가 고객 회로인지, 실제 포화 여유가 얼마인지는 이 그래프로 정하지 않는다.")
    wave_keys = []
    t_off4 = None
    for hk, r in runs.items():
        k = int(np.argmax(r["pk"]["meas"]))
        tp = float(r["pk"]["t"][k])
        a, b = max(tp - 1.75 * T, 0.0), min(tp + 0.25 * T, M * T)
        smp = r["tr"].sample(["i"], a, b, per_segment=24)
        tt = np.array(smp["t"])
        res.add_series(f"wv_{hk}", f"{hk}", "A", (tt - tp).tolist(), (np.array(smp["i"]) + drift_e(v, worlds[hk], tt)).tolist())
        wave_keys.append(f"wv_{hk}")
        if hk == "H4":
            t_off4 = (tp, a, b)
    bands = bands_e(runs["H4"]["tr"], t_off4[1], t_off4[2], t_off4[0]) if t_off4 else []
    res.add_plot("p_wave", "peak 직전 2주기 (각 세계의 peak 시각 기준 정렬): 포화는 기울기의 꺾임으로 보인다", wave_keys, x_label="t − t_peak", x_unit="s", y_label="센서 전류", y_unit="A", bands=bands, group="e", level="C",
                 hlines=[{"y": Is, "label": f"I_s {Is:.3g} A"}],
                 proved="H1–H4 세계는 + 구간 끝에서 전류 기울기가 L_m/L_sat배로 꺾이는(포화) 모양이고, H5 세계는 기울기가 일정한 채 전체가 위로 밀린(offset) 모양이라는 것을 보였다. 띠는 H4 세계의 회로 상태다.",
                 not_yet="실제 B-H 곡선은 둥글게 포화하므로 꺾임이 이렇게 날카롭지 않다. 기울기 비교에는 같은 timebase의 권선 전압이 필요하고, 센서 대역이 꺾임을 뭉갤 수 있다.")
    # periodic model hides the fault
    if "H4" in runs:
        r4 = runs["H4"]
        smp = r4["tr"].sample(["B"], per_segment=8)
        res.add_series("B_H4", "H4 세계: 기동 과도 B(t)", "T", smp["t"], smp["B"])
        sp = per0.trajectory.sample(["B"], per_segment=8)
        t_ss = np.array(sp["t"]) - 0.75 * T
        reps_t, reps_b = [], []
        for k in range(M):
            reps_t += (t_ss + 0.75 * T + (k - 1) * T).tolist()
            reps_b += sp["B"]
        sel = [(t, b) for t, b in zip(reps_t, reps_b) if 0.0 <= t <= M * T]
        res.add_series("B_ss", "주기해 (대칭 정상상태)", "T", [t for t, _ in sel], [b for _, b in sel], dash=True)
        res.add_plot("p_flux", "주기해는 이 고장을 숨긴다: 매 주기 대칭 자속을 가정하면 기동 과도가 없다", ["B_H4", "B_ss"], y_label="B", y_unit="T", level="C",
                     hlines=[{"y": v["Bs"], "label": f"B_s {v['Bs']:g} T"}, {"y": -v["Bs"], "label": f"−B_s"}],
                     proved="같은 회로의 주기해(shooting, 대칭 정상상태)는 B가 ±B_pk 안에 머물러 포화가 없고, enable부터 푼 과도해만 B_s를 넘는다는 것을 보였다 — 매 주기 자속을 0으로 reset하거나 정상상태만 푸는 모델은 이 고장을 원리적으로 보지 못한다.",
                     not_yet="H4 세계 하나의 과도다. 다른 세계도 정상상태는 같은 대칭 해로 수렴한다(H4는 C_b 평균 전압만 다르다). 실제 기동 순서·제어 루프는 모델에 없다.")
    tsel = v["test"]
    if tsel == "vs":
        tk = []
        for hk in runs:
            res.add_series(f"t_{hk}", f"{hk}", "T", ks[1:], (pred[hk]["dB"][1:]).tolist())
            tk.append(f"t_{hk}")
        res.add_plot("p_test", "시험: 권선 volt-second — 주기당 자속 변화 ΔB = ∫v_w dt/(N A_e) (첫 반폭 주기 제외)", tk, x_label="enable 후 주기", x_unit="", y_label="주기당 ΔB", y_unit="T", kind="xy", level="C",
                     proved="H2는 처음 K주기 동안만, H3·H4는 시작부터 같은 크기로 불균형이 있다가 C_b 공진으로 줄어들고, H1·H5는 처음 주기의 불균형이 거의 0이라는 것을 보였다. 같은 timebase의 전류 기울기 꺾임과 함께 보면 H1(균형인데 포화)과 H5(포화 없음)도 갈린다.",
                     not_yet="H3과 H4는 권선 volt-second가 같은 모양이라 이 시험만으로는 갈리지 않는다(C_b 전압이나 출력 펄스 폭으로 가른다). 권선 전압 probe의 offset 자체가 적분 drift를 만들 수 있다(FL07).")
    elif tsel == "gate":
        tk = []
        for hk in ("H2", "H4"):
            if hk in runs:
                w = worlds[hk]
                cmd = [(w.d_cmd if k < K else 0.0) for k in ks]
                out_ = [cmd[k] + w.d_out for k in ks]
                res.add_series(f"tc_{hk}", f"{hk} 명령", "s", ks, cmd, dash=True)
                res.add_series(f"to_{hk}", f"{hk} bridge 출력", "s", ks, out_)
                tk += [f"tc_{hk}", f"to_{hk}"]
        res.add_plot("p_test", "시험: 주기별 펄스 폭 차이 (+폭 − −폭), gate 명령 vs bridge 출력", tk, x_label="enable 후 주기", x_unit="", y_label="폭 차이", y_unit="s", kind="xy", level="C",
                     proved="H2는 명령부터 비대칭이고(처음 K펄스), H4는 명령이 대칭인데 출력만 매 주기 비대칭이라는 것을 보였다. H1·H3·H5는 둘 다 0이라 이 시험에서 같이 남는다.",
                     not_yet="실제로는 deskew된 고대역 probe로 수십 ns 차이를 재야 한다. 펄스 폭 차이는 부하·온도에 따라 달라질 수 있다.")
    elif tsel == "cb":
        tk = []
        for hk, r in runs.items():
            smp = r["tr"].sample(["vcb"], per_segment=4)
            res.add_series(f"t_{hk}", f"{hk}", "V", smp["t"], smp["vcb"])
            tk.append(f"t_{hk}")
        hl = [{"y": vcb_ss4, "label": f"H4 정상상태 {vcb_ss4:.3g} V"}] if worlds["H4"].ok else []
        res.add_plot("p_test", "시험: C_b 전압 — H3은 기동 전부터, H4는 정상상태에서 DC를 가진다", tk, y_label="v_Cb", y_unit="V", level="C", hlines=hl,
                     proved="H3 세계만 enable 순간 C_b 전압이 이미 있고(방전 후 재기동하면 증상이 사라짐), H4 세계만 정상상태 평균이 V·Δt_d·f로 남는다는 것을 보였다. H1·H2도 기동 과도 중에는 DC 전류가 C_b를 충전해 전압이 움직이므로, 판별에 쓰는 것은 기동 전 값과 정상상태 평균이다.",
                     not_yet="실제 측정은 큰 공통모드 전압 위의 작은 DC 전압이라 차동 probe의 offset·CMRR을 확인해야 한다.")
    elif tsel == "probe":
        tk = []
        for hk, r in runs.items():
            res.add_series(f"t_{hk}", f"{hk} 기준 probe", "A", ks, r["pk"]["true"].tolist())
            tk.append(f"t_{hk}")
        res.add_plot("p_test", "시험: 두 번째 DC 결합 probe로 본 주기별 + peak", tk, x_label="enable 후 주기", x_unit="", y_label="+ peak (실제)", y_unit="A", kind="xy", level="C",
                     hlines=[{"y": v["I_obs"], "label": f"센서 {v['I_obs']:g} A"}],
                     proved="H5 세계에서는 기준 probe가 대칭 정상 전류만 보이고 센서만 커진다는 것, 다른 세계에서는 두 측정이 같은 증가를 보인다는 것을 보였다.",
                     not_yet="기준 probe도 degauss·zero·대역을 확인해야 한다. 두 측정이 같아도 원인은 H1–H4 중 무엇인지 남는다.")
    elif tsel == "demag":
        tk = []
        for hk, w in worlds.items():
            if not w.ok:
                continue
            s2, tr2 = run_e(v, w, M, demag=True)
            res.add_series(f"t_{hk}", f"{hk} 소거 후", "A", ks, cycle_peaks(v, w, s2, tr2, M)["meas"].tolist())
            tk.append(f"t_{hk}")
        res.add_plot("p_test", "시험: 코어를 소거(demag)한 뒤 재기동한 주기별 + peak", tk, x_label="enable 후 주기", x_unit="", y_label="+ peak (센서)", y_unit="A", kind="xy", level="C",
                     hlines=[{"y": v["I_obs"], "label": f"관측 {v['I_obs']:g} A"}],
                     proved="잔류 자속 세계(H1)만 소거 후 증상이 사라지고 나머지 세계는 그대로라는 것을 보였다.",
                     not_yet="실제 소거가 잔류 자속을 얼마나 지우는지, 정지 시점(자속 위치)에 따라 B_r이 달라지는지는 측정으로 확인해야 한다.")
    # checks -----------------------------------------------------------------------------
    hc = "H4" if "H4" in runs else next(iter(runs))
    w = worlds[hc]
    r = runs[hc]
    im_ode, end_ode = ode_e(v, w, M)
    im_eng = float(np.max(r["pk"]["meas"]))
    z_end = r["tr"].z_end[:2]
    err = max(abs(im_ode - im_eng) / max(abs(im_eng), 1e-12), abs(end_ode[0] - z_end[0]) / max(Is, 1e-12), abs(end_ode[1] - z_end[1]) / v["V"])
    res.add_check(Check(f"독립 경로: 자속좌표 [λ, v_Cb] + 연속 PWL i(λ), DOP853 ({hc} 세계, {M}주기)", "PASS" if err < 1e-6 else "FAIL", err, "rel", 1e-6,
                        path="영역 guard 없이 자속을 상태로 적분 (엔진은 전류 상태 + 포화 guard + 행렬지수)", independent=True,
                        detail=f"최대 전류 엔진 {im_eng:.9g} A vs ODE {im_ode:.9g} A"))
    led = ledger_e(r["s"], r["tr"], 0.0, M * T, max(v["I_obs"], Is))
    res.add_check(ledger_check(led, threshold=1e-6, what=f"{hc} 세계 {M}주기 (PWL 자기 에너지 ∫i dλ 포함): "))
    cal_err = max((abs(float(np.max(rr["pk"]["meas"])) - v["I_obs"]) / v["I_obs"] for rr in runs.values()), default=0.0)
    res.add_check(Check("세계들이 같은 관측(센서 최대값)을 재현", "PASS" if cal_err < 1e-6 else "FAIL", cal_err, "rel", 1e-6, path="brentq로 맞춘 미지수로 다시 시뮬레이션", independent=False,
                        detail="재현 불가: " + (", ".join(k for k, ww in worlds.items() if not ww.ok) or "없음")))
    res.add_check(Check("주기해 shooting 수렴 (대칭·H4)", "PASS" if per0.converged and per4.converged else "FAIL", max(per0.residual, per4.residual), "rel", 1e-11, path="engine.periodic.shoot (Newton, 유한차분 Jacobian)", independent=False))
    ev = Evidence(res)
    w_fl07 = ev.cite("FL07", "flux_walk", "textbook", "walk", note="교재 800 V × 100 ns 비대칭의 주기당 자속 이동 (FL07 자체 시스템 WalkCore). 그 실행의 FAIL_CONSTRAINT는 R = 0이면 자속이 B_sat을 넘는다는 FL07의 판정이다")
    b_fl07 = ev.cite("FL07", "flux_walk", "textbook", "Bpk_sym", note="대칭 구동 B_pk — 이 사례의 정상상태 자속")
    f0_fl07 = ev.cite("FL07", "flux_walk", "blocking", "f0", note="L_m–C_b 공진 — C_b 경로 세계(H3·H4)의 peak 시점")
    if tbw:
        res.add_check(check_close("주기당 자속 이동: 이 모델 (R = 0, C_b 없음) vs FL07 flux_walk@textbook", walk100, w_fl07, 1e-9, "SatBridge 정확 해 vs FL07 WalkCore 정확 해", True, "T"))
    res.add_check(check_close("B_pk(대칭): 닫힌 식 vs FL07 실행", v["V"] / (4.0 * v["fs"] * NA), b_fl07, 1e-9, "V/(4 f N A_e) vs FL07 flux_walk@textbook", True, "T"))
    if abs(v["Lm"] - 2e-3) < 1e-15 and abs(v["Cb"] - 10e-6) < 1e-18:
        res.add_check(check_close("L_m–C_b 공진: 이 사례 vs FL07 flux_walk@blocking", f0, f0_fl07, 1e-9, "1/(2π√(L_m C_b)) vs FL07 실행", True, "Hz"))
    ev.finish()
    res.circuit = {"diagram": circuit_e(v).to_json(), "intervals": bands, "plot_group": "e"}
    unresolved(res, f"권선 volt-second + 전류 기울기({groups['vs']}묶음), gate 명령·출력({groups['gate']}), C_b 전압({groups['cb']}), 두 번째 probe({groups['probe']}), 소거 후 재기동({groups['demag']})을 조합해야 한다.")
    res.verdict("MISSING_INPUT", "실제 B-H 곡선(잔류자속·포화), soft-start 순서, leg별 dead time·지연, C_b 방전 경로, 센서 사양이 없다 — 세계의 미지수는 관측을 재현하도록 맞춘 합성 값이다")
    res.assumptions += [
        f"bridge ±{v['V']:g} V, {v['fs'] / 1e3:g} kHz, 2준위 구동, 첫 펄스 반폭(대칭 기동); 무부하 (부하 전류는 대칭이라 자속에 영향 없음)",
        f"N {v['N']:g}, A_e {v['Ae']:g} mm², L_m {v['Lm'] * 1e3:g} mH, PWL 포화: B_s {v['Bs']:g} T 위에서 L_m/{v['k_sat']:g} (합성), R {v['R'] * 1e3:g} mΩ, C_b {v['Cb'] * 1e6:g} µF (FL07과 같은 값)",
        f"관측 = 처음 {M}주기 안에 센서 + peak {v['I_obs']:g} A (ASSUMED 수치화)",
        f"H2: 처음 {K}펄스만 비대칭; H5: 센서 offset이 τ_s {v['tau_s'] * 1e6:g} µs로 정착",
    ]
    res.not_valid_for += ["실제 코어 재료의 B-H 곡선·잔류자속", "제어 루프(전류 모드 자속 균형 등)", "부하가 걸린 기동", "고객 회로의 근본원인 확정"]
    res.interpretation = (
        f"enable 직후 + 전류가 커지는 같은 증상을 다섯 세계가 모두 만든다: 잔류 자속(B_r {val_of['H1'](worlds['H1']):.3g} T 필요), soft-start 비대칭, C_b 초기 전압, dead time 불균형, 센서 offset. "
        "자속이 포화에 닿으면 증분 인덕턴스가 급감해 전류가 한쪽으로만 튀고, 센서 offset은 기울기 변화 없이 전체를 밀어 올린다. "
        "C_b가 있으면 DC volt-second는 L_m–C_b 공진으로 되돌아오므로 H3·H4의 peak는 공진 1/4 주기쯤에 나온다 — 정상상태(주기해)만 푸는 모델은 이 과도를 원리적으로 보지 못한다. "
        "실제 권선 전압의 volt-second와 전류 기울기를 같은 timebase로 보고, gate 명령과 bridge 출력, C_b 전압, 두 번째 probe, 소거 후 재기동으로 가른다. 1쪽 메모 영어 틀: " + MEMO_EN_TEMPLATE
    )
    return res


def params_e() -> list[Param]:
    g_d, g_c, g_o, g_w = "구동 (FL07)", "코어·회로", "관측", "세계 가정"
    return [
        Param("V", "bridge 출력 ±V", "V", 800.0, "V", vmin=1, vmax=3000, source="TEXTBOOK", source_note="FL07 800 V", group=g_d),
        Param("fs", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=2e6, source="TEXTBOOK", source_note="FL07", group=g_d),
        Param("dt_ref", "FL07 비교용 비대칭 Δt", "s", 100e-9, "ns", vmin=0.0, vmax=5e-6, source="TEXTBOOK", source_note="FL07 100 ns → 0.0064 T", group=g_d),
        Param("N", "권선수 N", "", 50, "", vmin=1, vmax=2000, kind="int", source="TEXTBOOK", source_note="FL07", group=g_c),
        Param("Ae", "유효 단면적 A_e", "mm²", 250.0, "mm²", vmin=1.0, vmax=1e5, source="TEXTBOOK", source_note="FL07", group=g_c),
        Param("Lm", "자화 인덕턴스 L_m (선형 구간)", "H", 2e-3, "mH", vmin=1e-5, vmax=10.0, source="ASSUMED", source_note="FL07과 같은 값", group=g_c),
        Param("Bs", "포화 자속 B_s", "T", 0.35, "T", vmin=0.05, vmax=3.0, source="ASSUMED", source_note="FL07 B_sat (재료 자료 없음)", group=g_c),
        Param("k_sat", "포화 후 인덕턴스 감소비 L_m/L_sat", "", 20.0, "", vmin=1.5, vmax=1000, source="ASSUMED", source_note="PWL 합성 곡선", group=g_c),
        Param("R", "직렬 저항 R", "Ω", 0.1, "mΩ", vmin=1e-6, vmax=100.0, source="ASSUMED", source_note="FL07", group=g_c),
        Param("Cb", "blocking capacitor C_b", "F", 10e-6, "µF", vmin=1e-9, vmax=1.0, source="ASSUMED", source_note="FL07", group=g_c),
        Param("I_obs", "관측: 센서 + peak", "A", 4.0, "A", vmin=0.5, vmax=1000, source="ASSUMED", source_note="‘한쪽으로 증가’를 수치화", group=g_o),
        Param("M_obs", "관측 창 (enable 후 주기 수)", "", 40, "", vmin=5, vmax=400, kind="int", source="ASSUMED", group=g_o),
        Param("K_soft", "H2: 비대칭 기동 펄스 수", "", 10, "", vmin=1, vmax=200, kind="int", source="ASSUMED", group=g_w),
        Param("tau_s", "H5: 센서 offset 정착 시정수", "s", 100e-6, "µs", vmin=1e-6, vmax=1.0, source="ASSUMED", group=g_w),
        _test_param(E_TESTS),
    ]


EXP_E = Experiment(
    key="case_e",
    title="CASE E — 변압기 기동 포화 의심: enable 직후 1차 전류가 한쪽으로 커지는 다섯 가지 이유",
    goal=(
        "증상: 정상 동작은 괜찮은데 enable 직후 primary current가 한쪽으로 증가한다. 확인된 것: 정상상태 파형은 대칭이고 기동 직후에만 + 쪽 peak가 커진다(여기서는 처음 40주기 안에 4 A로 수치화). "
        "모르는 것: 잔류 자속, soft-start 순서, C_b의 초기 전압, leg별 dead time·지연, 센서 offset, 실제 권선 전압. 먼저 Q1–Q3에 답을 적고 실행한다. "
        "실행하면 FL07과 같은 코어·구동(800 V, 100 kHz, N 50, A_e 250 mm²)에 PWL 포화 자화 가지와 C_b를 넣은 정확 스위칭 모델로 잔류 자속·비대칭 펄스·C_b 초기 전압·dead time 불균형·센서 offset 다섯 세계를 "
        "같은 관측에 맞추고, 권선 volt-second·gate 명령·C_b 전압·두 번째 probe·소거 후 재기동이 각 세계에서 무엇을 보일지 비교한다. 주기해만 푸는 모델이 이 고장을 숨기는 이유도 본다."
    ),
    params=params_e(),
    presets=[
        Preset("observed", "관측 그대로 (enable 직후 + peak 4 A)", {}, "CASE E 증상", ("nominal", "reference")),
        Preset("vs", "첫 시험: 권선 volt-second + 기울기", {"test": "vs"}, "포화 여부와 불균형의 시간 모양", ("reference",)),
        Preset("gate", "gate 명령 vs bridge 출력", {"test": "gate"}, "H2·H4", ("variant", "reference")),
        Preset("cb", "C_b 전압", {"test": "cb"}, "H3·H4", ("variant",)),
        Preset("probe", "두 번째 DC 결합 probe", {"test": "probe"}, "H5", ("variant",)),
        Preset("demag", "소거 후 재기동", {"test": "demag"}, "H1", ("variant",)),
        Preset("soft_core", "포화가 완만한 코어 (L_m/L_sat = 4)", {"k_sat": 4.0}, "보정값이 바뀐다", ("corner",)),
    ],
    run=run_case_e,
    model_level="C (정확 스위칭 + PWL 포화, 영역 guard) + 측정 경로",
    suggested_change="판별 시험을 ‘권선 volt-second + 기울기’로 바꾼다 (다음에 ‘gate 명령 vs bridge 출력’, ‘C_b 전압’).",
    prediction=Prediction(
        "권선 전압의 volt-second와 전류 기울기를 같은 timebase로 보면 어느 가설이 바로 떨어져 나가나?",
        ["모두 같은 모양이라 아무것도 갈리지 않는다", "센서 offset(H5): 기울기 꺾임이 없다", "잔류 자속(H1)만: volt-second가 0이다", "모르겠다"],
        "센서 offset(H5): 기울기 꺾임이 없다",
        "H1–H4는 실제 자속이 포화에 닿아 + 구간 끝의 전류 기울기가 L_m/L_sat배로 꺾인다. H5는 실제 전류가 대칭이고 센서 값만 밀려 올라가므로 기울기가 그대로다. "
        "나머지는 volt-second 불균형의 시간 모양(H1 거의 0, H2 처음 K주기, H3·H4 C_b 공진으로 감소)으로 일부만 갈리고, H3·H4는 C_b 전압이나 bridge 출력 펄스 폭으로 가른다.",
        ["cal_H1", "cal_H3", "cal_H4", "Ipk_ss"],
        handcalc=[{"key": "Bpk_sym", "label": "대칭 B_pk = V/(4fNA_e)", "unit": "T"}, {"key": "walk_100", "label": "100 ns 비대칭의 주기당 ΔB", "unit": "T"}, {"key": "Is", "label": "포화 시작 전류", "unit": "A"}],
    ),
    suggested={"test": "vs"},
    student=(
        "코어 자속은 권선 전압을 시간으로 적분한 것이다. 정상상태에서는 +와 − 구간의 volt-second가 같아 자속이 대칭으로 오르내리지만, enable 직후에는 출발점(잔류 자속, C_b 전압)과 처음 펄스들의 폭이 대칭을 깰 수 있다. "
        "자속이 포화에 닿으면 인덕턴스가 급히 작아져 전류가 한쪽으로만 튄다. 그런데 센서 offset이 기동 후 변해도 화면에서는 비슷하게 보인다 — 그래서 전류만이 아니라 권선 전압의 volt-second와 전류 기울기를 함께 본다."
    ),
    expert=(
        "교재 해설(CASE E): 초기 flux/잔류자속, 비대칭 펄스, DC-blocking capacitor 초기조건, dead time 불균형, current measurement offset을 구분한다. 실제 winding voltage의 volt-second와 current slope를 동시 본다. "
        "시뮬레이션이 매 주기 자속을 0으로 reset한다면 이 고장을 숨기는 모델이다. "
        "추가: ① 포화는 기울기의 꺾임(증분 인덕턴스 급감)으로, 센서 offset은 기울기 변화 없는 평행이동으로 보인다. ② C_b가 있으면 DC volt-second는 L_m–C_b 공진으로 되돌아온다 — C_b 초기 전압 V와 dead time 불균형 V·Δt·f는 "
        "기동 직후 같은 DC 구동이라 전류 모양이 거의 같고, 정상상태 C_b 평균 전압(H4: V·Δt·f, H3: 0)이나 기동 전 C_b 전압으로 가른다. ③ 잔류 자속 가설은 필요한 B_r 크기를 재료·gap 자료와 비교한다. "
        "④ bridge 명령이 대칭이어도 출력 펄스 폭은 다를 수 있다 — gate 명령과 bridge 출력을 deskew해서 같이 본다. ⑤ FL07처럼 100 ns 비대칭은 주기당 0.0064 T를 옮긴다: 수십 주기면 포화 여유를 다 쓴다."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: enable 직후 1차 전류가 한쪽으로 커지는 원인을 정하고 기동 포화를 없앤다. 확인한 조건: 정상상태는 대칭, 기동 직후 + peak가 커진다. "
        "모르는 조건: 잔류 자속, soft-start 순서, C_b 초기 전압, leg별 dead time·지연, 센서 offset. 유력 가설과 반례: 기동 비대칭(펄스·C_b·dead time)이나 잔류 자속으로 코어가 포화에 닿는 것이 유력하지만, "
        "센서 offset도 화면에서는 같은 증상이다 — 실제 포화라면 전류 기울기가 꺾여야 한다. 현재 권고: 권선 전압(volt-second)과 전류를 같은 timebase로, gate 명령과 bridge 출력을 deskew해서, C_b 전압을 enable 전후로 받고, "
        "두 번째 DC 결합 probe로 센서를 확인한다. 부작용: 기동을 느리게 하거나 C_b를 키우는 임시조치는 기동 시간·과도 응답을 바꾼다. 완료 판정: 원인 하나를 고친 뒤 처음 100주기의 권선 volt-second가 균형이고 peak가 정상상태 수준이면 닫는다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the flux margin during the first cycles after enable. My current hypothesis is a start-up asymmetry that walks the core into saturation, "
        "but residual flux, a pre-charged blocking capacitor, a dead-time imbalance or a drifting current-sensor offset could produce a similar symptom. "
        "I would separate them by measuring the winding volt-seconds together with the current slope, the gate commands against the bridge output, the blocking-capacitor voltage before and after enable, and a second DC-coupled current probe. "
        "If the current slope does not bend, I recommend treating it as a sensor issue; otherwise I recommend fixing the start sequence or the imbalance and would recheck the first hundred cycles."
    ),
    questions=[
        q_hyp(
            "E",
            "정상 동작은 괜찮은데 enable 직후 1차 전류가 한쪽으로 커진다. 가설 3개는?",
            "① 초기·잔류 자속(이전 정지 위치에서 출발) ② 비대칭 펄스(soft-start 첫 펄스 폭·PWM 갱신) ③ DC-blocking capacitor 초기 전압 ④ dead time·driver 지연 불균형(명령은 대칭, 출력은 비대칭) ⑤ 전류 측정 offset/drift. "
            "포화라면 전류 기울기가 꺾여야 하고, 정상상태만 푸는 시뮬레이션은 이 고장을 숨긴다는 점을 말한다.",
            "Right after enable the primary current grows to one side, although normal operation is fine: give three hypotheses.",
            "Residual or initial flux, asymmetric pulses in the start sequence, an initial voltage on the blocking capacitor, a dead-time or delay imbalance with symmetric commands, and a current-sensor offset. Real saturation bends the current slope; a steady-state-only simulation hides this fault.",
            ["잔류 자속", "비대칭 펄스", "C_b 초기조건", "dead time 불균형", "측정 offset", "주기해는 고장을 숨김"],
        ),
        q_wave(
            "E",
            "① 권선 전압과 1차 전류를 같은 timebase로(volt-second 적분과 기울기 꺾임) ② 두 leg의 gate 명령과 bridge 출력 전압(deskew) ③ enable 전후의 C_b 전압과 두 번째 DC 결합 전류 probe. "
            "enable 직전 정지 상태(자속 위치), 온도, probe offset·zero를 함께 기록한다.",
            "The winding voltage and the primary current on one timebase, for the volt-seconds and the slope; both legs' gate commands with the bridge output voltage, deskewed; and the blocking-capacitor voltage before and after enable plus a second DC-coupled current probe. Record the shutdown state before enable, temperature and probe offsets.",
            ["권선 전압 + 전류 동시", "gate 명령 vs 출력", "C_b 전압", "두 번째 probe"],
        ),
        q_first(
            "E",
            "권선 volt-second와 전류 기울기를 함께 본다: 기울기가 꺾이지 않으면 H5(센서), 꺾이는데 처음 주기 volt-second가 균형이면 H1(잔류 자속), 처음 K주기만 불균형이면 H2, 시작부터 불균형이 C_b 공진으로 줄면 H3·H4다. "
            "H3·H4는 enable 전 C_b 전압(H3)과 정상상태 C_b 평균 전압·bridge 출력 펄스 폭(H4)으로 가른다.",
            "Capture the winding volt-seconds with the current slope. No bend means a sensor offset; a bend with balanced volt-seconds in the first cycle points to residual flux; imbalance only in the first K cycles points to the start sequence; imbalance from the start that decays with the L_m-C_b resonance points to the capacitor pre-charge or a dead-time imbalance, which the capacitor voltage and the bridge pulse widths then separate.",
            ["volt-second + 기울기", "꺾임 없음 → 센서", "시간 모양", "C_b 전압으로 H3/H4"],
        ),
        q_memo(
            "E",
            "목표: 기동 포화 원인 확정·제거. 확인: 정상상태 대칭, 기동 직후 + peak 증가. 모름: 잔류 자속, soft-start, C_b 초기 전압, dead time, 센서. 가설과 반례: 기동 비대칭·잔류 자속 → 포화가 유력, 그러나 센서 offset도 같은 화면 — 포화면 기울기가 꺾여야 함. "
            "권고: 권선 volt-second + 기울기, gate 명령 vs 출력, C_b 전압, 두 번째 probe. 부작용: 느린 기동·큰 C_b는 기동 시간·과도 응답 변경. 완료 판정: 처음 100주기 volt-second 균형, peak가 정상상태 수준.",
            MEMO_EN_TEMPLATE.replace("___", "the flux margin in the first cycles after enable", 1),
        ),
    ],
    circuit="fl12_e",
    textbook=[TB_15, TB_10, TB_17],
    reference_presets=["observed", "vs", "gate"],
    runtime_hint="seconds",
    claim_limit="PWL 포화 합성 코어의 정확 스위칭 해로 가설별 예측을 비교한다. 실제 코어의 포화 여유와 고객의 근본원인을 확정하지 않는다.",
)


# ======================================================================================
# CASE F - 98 % efficiency, but the calorimetric loss does not agree (200 W electrical vs 350 W)
# ======================================================================================

F_TESTS = [
    ("none", "시험 전: 전기 200 W vs 열량 350 W"),
    ("heater", "알려진 heater로 열량계 교정 (2 수준)"),
    ("aux", "보조전원·pump 전력 별도 측정"),
    ("sync", "같은 분석기 동시 취득 + ½CV² 보정"),
    ("steady", "열 정상상태(5τ) 후 재측정"),
    ("pf", "PF 1 부하 / 위상 교정 센서"),
    ("swap", "두 채널을 같은 도체에 (교차 확인)"),
]


def u_diff_f(u1: float, u2: float, rho: float) -> float:
    """GUM first-order uncertainty of a difference x1 - x2 with correlation rho."""
    return math.sqrt(max(u1 * u1 + u2 * u2 - 2.0 * rho * u1 * u2, 0.0))


def f_setup(v: dict) -> dict:
    Pin = v["P_in"]
    Pout = Pin * v["eta"]
    G = v["rho_c"] * v["cp"] * v["flow_lpm"] / 60000.0  # coolant heat-capacity rate rho cp Vdot [W/K]
    dT = v["Q_cal"] / G
    rel_q = math.sqrt(v["u_rhoc"] ** 2 + v["u_cp"] ** 2 + v["u_flow"] ** 2 + (v["u_dT"] / dT) ** 2)
    return {"Pin": Pin, "Pout": Pout, "Pe": Pin - Pout, "gap": v["Q_cal"] - (Pin - Pout), "G": G, "dT": dT, "uin": v["rel_u"] * Pin, "uout": v["rel_u"] * Pout, "uQ": rel_q * v["Q_cal"]}


def f_worlds(v: dict, c: dict) -> dict:
    """Calibrated bias of each world that turns a true state into the observed pair (200 W electrical, 350 W calorimetric)."""
    gap, Pout, Pe = c["gap"], c["Pout"], c["Pe"]
    tanphi = math.tan(math.acos(v["PF"]))
    W = {}
    W["H1"] = {"true": v["Q_cal"], "val": -gap / Pout, "unit": "", "txt": f"출력 채널 이득 {-gap / Pout * 100:+.3g} % (사양 ±{v['rel_u'] * 100:g} %의 {gap / Pout / v['rel_u']:.3g}배)"}
    W["H2"] = {"true": v["Q_cal"], "val": gap / (Pout * tanphi), "unit": "rad", "txt": f"전류 센서 위상 {math.degrees(gap / (Pout * tanphi)):.3g}° (PF {v['PF']:g}, tan φ {tanphi:.3g})"}
    W["H3"] = {"true": v["Q_cal"], "val": gap / (Pout * v["drift"]), "unit": "s", "txt": f"P_in·P_out 읽은 시각 차 {gap / (Pout * v['drift']):.3g} s (전력 drift {v['drift'] * 100:g} %/s)"}
    W["H4"] = {"true": Pe, "val": gap, "unit": "W", "txt": f"경계 밖 보조전원·pump 열 {gap:.4g} W가 냉각수로"}
    W["H5"] = {"true": Pe, "val": gap / c["G"], "unit": "K", "txt": f"ΔT 센서 offset {gap / c['G']:.3g} K (측정 ΔT {c['dT']:.3g} K 중)"}
    W["H6"] = {"true": Pe, "val": gap / Pe, "unit": "", "txt": f"유량·c_p 배율 오차 {gap / Pe * 100:+.3g} %"}
    if v["P_prev"] > v["Q_cal"]:
        tm = v["tau_th"] * math.log((v["P_prev"] - Pe) / (v["Q_cal"] - Pe))
        W["H7"] = {"true": Pe, "val": tm, "unit": "s", "txt": f"이전 시험(손실 {v['P_prev']:g} W) 후 {tm:.4g} s에 측정 (τ {v['tau_th']:g} s)"}
    else:
        W["H7"] = {"true": Pe, "val": None, "unit": "s", "txt": f"재현 불가: 이전 시험 손실 {v['P_prev']:g} W ≤ 관측 {v['Q_cal']:g} W"}
    dE = gap * v["T_w"]
    if 2.0 * dE / v["C_dc"] < v["V_dc"] ** 2:
        dV = v["V_dc"] - math.sqrt(v["V_dc"] ** 2 - 2.0 * dE / v["C_dc"])
        W["H8"] = {"true": v["Q_cal"], "val": dV, "unit": "V", "txt": f"측정 창 {v['T_w']:g} s 동안 DC-link {dV:.3g} V 하강 (½CV² {dE:.3g} J)"}
    else:
        W["H8"] = {"true": v["Q_cal"], "val": None, "unit": "V", "txt": "재현 불가: 저장에너지보다 큰 변화"}
    return W


def f_observe(v: dict, c: dict, key: str, w: dict, test: str = "none") -> tuple[float, float]:
    """(electrical loss reading, calorimetric reading) that world `key` predicts under `test`."""
    Pe, Pout = c["Pe"], c["Pout"]
    true = w["true"]
    if w["val"] is None:
        return float("nan"), float("nan")
    e_bias = 0.0  # electrical reading = true loss - e_bias (worlds whose true loss is 350 W)
    q = true
    if key == "H1":
        e_bias = -w["val"] * Pout
        if test == "swap":
            e_bias = e_bias  # the swap shows the channel mismatch; the gap itself is unchanged until corrected
    elif key == "H2":
        e_bias = Pout * math.tan(math.acos(v["PF"])) * w["val"] if test != "pf" else 0.0
    elif key == "H3":
        e_bias = Pout * v["drift"] * w["val"] if test != "sync" else 0.0
    elif key == "H8":
        e_bias = 0.5 * v["C_dc"] * (v["V_dc"] ** 2 - (v["V_dc"] - w["val"]) ** 2) / v["T_w"] if test != "sync" else 0.0
    elif key == "H4":
        q = true + (w["val"] if test != "aux" else 0.0)
    elif key == "H5":
        q = true + c["G"] * w["val"]
    elif key == "H6":
        q = true * (1.0 + w["val"])
    elif key == "H7":
        t = w["val"] if test != "steady" else 5.0 * v["tau_th"]
        q = Pe + (v["P_prev"] - Pe) * math.exp(-t / v["tau_th"])
    return true - e_bias, q


def thermal_ode_time(v: dict, Pe: float) -> float:
    """Independent path for H7: first-order thermal network C dT/dt = P - (T - T_c)/R_th, coolant heat (T - T_c)/R_th."""
    R_th = 0.05
    C_th = v["tau_th"] / R_th
    T0 = v["P_prev"] * R_th  # steady temperature rise of the previous test

    def ev(t, y):
        return y[0] / R_th - v["Q_cal"]

    ev.terminal = True
    ev.direction = -1
    sol = solve_ivp(lambda t, y: [(Pe - y[0] / R_th) / C_th], (0.0, 20.0 * v["tau_th"]), [T0], method="DOP853", rtol=1e-12, atol=1e-12, events=ev)
    return float(sol.t_events[0][0]) if sol.t_events[0].size else float("nan")


def circuit_f(v: dict) -> Circuit:
    c = Circuit("fl12_f", 700, 300, title="측정 경계: 전기(P_in − P_out)와 열량(ṁ c_p ΔT)이 같은 것을 재는가")
    c.add("block", "SRC", 60, 90, 0, "DC 입력", w=80, h=44)
    c.add("block", "MI", 170, 90, 0, "P_in 계측", w=90, h=34)
    c.add("block", "CONV", 320, 90, 0, f"컨버터 {v['P_in'] / 1e3:g} kW", w=120, h=60)
    c.add("block", "MO", 490, 90, 0, f"P_out 계측 (PF {v['PF']:g})", w=150, h=34)
    c.add("block", "LOAD", 640, 90, 0, "AC 부하", w=80, h=44)
    c.add("block", "AUX", 320, 190, 0, "보조전원·pump (경계?)", w=170, h=34)
    c.add("block", "CP", 320, 255, 0, "냉각수: T_in → T_out, 유량", w=200, h=34)
    c.add("block", "CAL", 560, 255, 0, "열량 ṁ c_p ΔT", w=130, h=34)
    c.wire("w1", (100, 90), (125, 90))
    c.wire("w2", (215, 90), (260, 90))
    c.wire("w3", (380, 90), (415, 90))
    c.wire("w4", (565, 90), (600, 90))
    c.wire("w5", (320, 120), (320, 173))
    c.wire("w6", (320, 207), (320, 238))
    c.wire("w7", (420, 255), (495, 255))
    c.text(170, 135, "전기 경계", "note")
    c.text(560, 210, "열 경계 (pump 열 포함?)", "note")
    return c


def run_case_f(v: dict) -> Result:
    res = Result("FL12", "case_f", "A (GUM 1차 전파·경계 대수) + Monte Carlo (seed) + 1차 열 모델")
    c = f_setup(v)
    Pin, Pout, Pe, gap = c["Pin"], c["Pout"], c["Pe"], c["gap"]
    rho = v["rho"]
    u0 = u_diff_f(c["uin"], c["uout"], 0.0)
    ur = u_diff_f(c["uin"], c["uout"], rho)
    u_anti = u_diff_f(c["uin"], c["uout"], -1.0)
    u_gap = math.sqrt(ur**2 + c["uQ"] ** 2)
    tb = abs(Pin - 1e4) < 1e-9 and abs(v["eta"] - 0.98) < 1e-12 and abs(v["Q_cal"] - 350) < 1e-9
    res.add_metric("P_e", "전기적 손실 P_in − P_out", Pe, "W", ref=200.0 if tb else None, ref_label="교재 200 W (10 kW, η 98 %)", tol=1e-9)
    res.add_metric("gap", "열량 − 전기 차이", gap, "W", ref=150.0 if tb else None, ref_label="교재 350 − 200 W", tol=1e-9)
    res.add_metric("u_e_ind", "u(전기 손실), 독립 ρ = 0", u0, "W", basis=f"u(P_in) {c['uin']:.4g} W, u(P_out) {c['uout']:.4g} W (각 ±{v['rel_u'] * 100:g} %를 1σ로)")
    res.add_metric("u_e_rho", f"u(전기 손실), ρ = {rho:g}", ur, "W", basis="ρ의 근거(같은 분석기·동시 취득)가 기록돼야 쓸 수 있다")
    res.add_metric("u_e_anti", "u(전기 손실), ρ = −1 (최악)", u_anti, "W")
    res.add_metric("u_Q", "u(열량)", c["uQ"], "W", basis=f"ΔT {c['dT']:.4g} K, 유량 {v['flow_lpm']:g} L/min, ρc_p 상대 불확도 합성")
    res.add_metric("z_gap", "차이 / 합성 표준불확도", gap / u_gap, "", basis=f"u_gap = √(u_e² + u_Q²) = {u_gap:.4g} W — 2를 크게 넘으면 우연 오차가 아니라 편향")
    # EX11 cross-check numbers with this lab's own formula
    u11 = u_diff_f(11.0, 10.78, 0.0)
    u11r = u_diff_f(11.0, 10.78, 0.9)
    res.add_metric("u_ex11_ind", "EX11 조건(11 kW/10.78 kW, 각 0.1 %) 독립", u11, "W", ref=15.40157, ref_label="교재 15.40157 W (EX11)", tol=1e-6)
    res.add_metric("u_ex11_rho", "EX11 조건, ρ = 0.9", u11r, "W", ref=4.87487, ref_label="교재 4.87487 W (EX11)", tol=1e-5)
    W = f_worlds(v, c)
    hyps = [
        Hyp("H1", "전력계 이득(정확도)", "출력 채널이 높게 읽는다 (range·CT 비 설정 오류 포함)", "이득 오차"),
        Hyp("H2", "위상 오차", "AC 출력 전류 센서의 위상 지연이 유효전력을 높게 읽게 한다", "δ"),
        Hyp("H3", "동시성", "P_in과 P_out을 다른 시각에 읽었고 그 사이 전력이 변했다", "Δt"),
        Hyp("H4", "보조전원·계측 경계", "경계 밖에서 공급되는 보조전원·pump의 열이 냉각수로 들어간다", "P_aux"),
        Hyp("H5", "ΔT 센서 offset", "작은 ΔT에 센서 offset이 더해진다", "δT"),
        Hyp("H6", "유량·c_p 배율", "유량계나 냉각수 물성(c_p·밀도) 값이 틀렸다", "배율"),
        Hyp("H7", "열 저장 과도", "이전의 더 큰 손실에서 쌓인 열이 아직 빠져나오는 중에 쟀다", "측정 시각"),
        Hyp("H8", "전기 저장에너지 과도", "측정 창 동안 DC-link ½CV²가 줄어 입력 계측이 덜 읽었다", "ΔV"),
    ]
    for h in hyps:
        w = W[h.key]
        res.add_metric(f"cal_{h.key}", f"{h.key} 세계: {h.unknown} (관측 재현)", w["val"] if w["val"] is not None else float("nan"), w["unit"], basis=w["txt"])
    plaus = {"H1": "사양 밖 — 설정·range·CT 비 오류라면 가능", "H2": "clamp·deskew에 따라 가능", "H3": "따로 읽었다면 가능", "H4": "pump가 열 경계 안이면 가능",
             "H5": "미교정 센서라면 가능", "H6": "물 vs glycol c_p 오적용(약 15 %)으로도 부족", "H7": "측정 시각 기록이 없으면 배제 불가", "H8": "부하 과도 중이면 가능"}
    rows = []
    for h in hyps:
        w = W[h.key]
        e, q = f_observe(v, c, h.key, w)
        rows.append([f"{h.key} {h.label}", f"{w['true']:.4g} W" if w["val"] is not None else "—", w["txt"], f"{e:.4g} / {q:.4g}" if w["val"] is not None else "재현 불가", plaus[h.key]])
    res.tables.append(Table("t_cal", "세계별 보정: 참 손실과, 관측(전기 200 W / 열량 350 W)을 만드는 편향 하나", ["세계", "참 손실", "보정한 편향", "재현 (전기 / 열량) [W]", "크기 검토"], rows,
                            note="H1·H2·H3·H8 세계에서는 열량이 맞고(참 손실 350 W, η 96.5 %), H4–H7 세계에서는 전기가 맞다(참 손실 200 W, η 98 %). 고객에게 중요한 것은 어느 숫자를 믿을지다."))
    # discrimination cells
    cl = {}
    for h in hyps:
        w = W[h.key]
        if w["val"] is None:
            cl[h.key] = {t: ("재현 불가", "재현 불가") for t, _ in F_TESTS}
            continue
        row = {}
        e0, q0 = f_observe(v, c, h.key, w)
        row["none"] = (f"전기 {e0:.4g} W / 열량 {q0:.4g} W", "관측 재현")
        # heater test: calorimeter error at two heater levels with the converter off (pump and aux running)
        errs = []
        for Ph in (v["P_heat"], 2 * v["P_heat"]):
            if h.key == "H4":
                errs.append(w["val"])
            elif h.key == "H5":
                errs.append(c["G"] * w["val"])
            elif h.key == "H6":
                errs.append(Ph * w["val"])
            else:
                errs.append(0.0)
        if abs(errs[0]) < 1e-6 * max(1.0, Pe):
            row["heater"] = ("열량계 정확 (오차 0)", "열량계 정확")
        elif abs(errs[1] - errs[0]) < 1e-6 * abs(errs[0]):
            row["heater"] = (f"heater {v['P_heat']:g}·{2 * v['P_heat']:g} W 모두 +{errs[0]:.3g} W (offset)", "offset")
        else:
            row["heater"] = (f"+{errs[0]:.3g} / +{errs[1]:.3g} W (비례)", "비례 오차")
        row["aux"] = (f"보조·pump {w['val']:.3g} W 발견 → 경계 맞춤", "보조 전력 큼") if h.key == "H4" else (f"보조 전력 ≤ {v['P_aux_known']:g} W (사양 수준)", "작음")
        for t in ("sync", "steady", "pf"):
            e1, q1 = f_observe(v, c, h.key, w, t)
            g1 = q1 - e1
            row[t] = (f"차이 {g1:.3g} W", "차이 사라짐" if abs(g1) < 0.1 * gap else "그대로")
        row["swap"] = (f"채널 불일치 {abs(w['val']) * 100:.3g} %", "불일치") if h.key == "H1" else (f"일치 (≤ {v['rel_u'] * 100:g} %)", "일치")
        cl[h.key] = row
    groups = disc_tables(res, hyps, F_TESTS, cl, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설의 세계가 예측하는 결과",
                         note="모든 세계가 같은 두 숫자(전기 200 W, 열량 350 W)를 재현하도록 편향 하나를 맞췄다. heater 교정은 열량계 쪽(H4·H5·H6)을, 보조 전력 측정은 경계(H4)를, 동시 취득은 H3·H8을, "
                         "정상상태 재측정은 H7을, PF 1 시험은 H2를, 채널 교차 확인은 H1을 가른다.")
    # Monte Carlo vs GUM (independent path) -----------------------------------------------
    rng = np.random.default_rng(int(v["seed"]))
    n = int(v["mc_n"])
    z = rng.standard_normal((n, 2))
    out = {}
    for r_ in sorted({0.0, rho}):
        L = np.linalg.cholesky(np.array([[1.0, r_], [r_, 1.0 + 1e-15]]))
        e = z @ L.T
        loss_s = (Pin + c["uin"] * e[:, 0]) - (Pout + c["uout"] * e[:, 1])
        out[r_] = loss_s
    zq = rng.standard_normal((n, 4))
    G_s = v["rho_c"] * (1 + v["u_rhoc"] * zq[:, 0]) * v["cp"] * (1 + v["u_cp"] * zq[:, 1]) * v["flow_lpm"] / 60000.0 * (1 + v["u_flow"] * zq[:, 2])
    q_s = G_s * (c["dT"] + v["u_dT"] * zq[:, 3])
    tol_mc = 4.0 / math.sqrt(2.0 * n)
    for r_, ls in out.items():
        res.add_check(check_close(f"u(전기 손실), ρ = {r_:g}: GUM vs Monte Carlo ({n}개, seed {int(v['seed'])})", float(np.std(ls, ddof=1)), u_diff_f(c["uin"], c["uout"], r_), tol_mc,
                                  "1차 전파 식 vs 상관 표본(Cholesky)의 표준편차", True, "W"))
    res.add_check(check_close(f"u(열량): GUM 상대 합성 vs Monte Carlo ({n}개)", float(np.std(q_s, ddof=1)), c["uQ"], max(tol_mc, 0.01), "곱의 상대 불확도 합성 vs 네 입력 표본의 곱 (비선형 포함, 허용 1 %)", True, "W"))
    w7 = W["H7"]
    if w7["val"] is not None:
        res.add_check(check_close("H7 측정 시각: 닫힌 식 τ·ln(...) vs 열 회로 ODE 사건", thermal_ode_time(v, Pe), w7["val"], 1e-8, "지수 닫힌 식 vs C dT/dt = P − T/R_th DOP853 + event", True, "s"))
    rep = max(abs((f_observe(v, c, k, w)[1] - f_observe(v, c, k, w)[0]) - gap) / gap for k, w in W.items() if w["val"] is not None)
    res.add_check(Check("세계들이 같은 관측(전기·열량 두 숫자)을 재현", "PASS" if rep < 1e-9 else "FAIL", rep, "rel", 1e-9, path="보정한 편향으로 두 측정값을 다시 계산", independent=False))
    ev = Evidence(res)
    e11 = ev.cite("EX11", "loss_uncertainty", "textbook", "u_loss_ind", note="11 kW·0.1 % 독립: 같은 식이 10 kW에서는 14.0 W")
    e11r = ev.cite("EX11", "loss_uncertainty", "rho09_nobasis", "u_loss_rho", note="ρ = 0.9를 근거 없이 쓰면 불확도가 작아 보인다")
    e10 = ev.cite("EX10", "rlc_fault", "textbook", "E0", note="같은 1 mF·800 V DC-link의 저장에너지 — H8 세계의 크기 기준")
    res.add_check(check_close("이 사례의 GUM 식 vs EX11 실행 (독립 ρ = 0)", u11, e11, 1e-9, "u_diff_f(11, 10.78, 0) vs EX11 loss_uncertainty@textbook", False, "W"))
    res.add_check(check_close("이 사례의 GUM 식 vs EX11 실행 (ρ = 0.9)", u11r, e11r, 1e-9, "u_diff_f(11, 10.78, 0.9) vs EX11 loss_uncertainty@rho09_nobasis", False, "W"))
    if abs(v["C_dc"] - 1e-3) < 1e-15 and abs(v["V_dc"] - 800) < 1e-9:
        res.add_check(check_close("DC-link ½CV²: 이 사례 vs EX10 실행", 0.5 * v["C_dc"] * v["V_dc"] ** 2, e10, 1e-12, "½CV² vs EX10 rlc_fault@textbook E0", False, "J"))
    ev.finish()
    # plots ------------------------------------------------------------------------------
    rr = np.linspace(-1.0, 1.0, 81)
    res.add_series("u_rho", "GUM u(P_in − P_out)", "W", rr.tolist(), [u_diff_f(c["uin"], c["uout"], float(x)) for x in rr])
    mc_r = [-0.9, 0.0, 0.5, 0.9]
    res.add_series("u_mc", f"Monte Carlo ({n}개)", "W", mc_r, [float(np.std((Pin + c["uin"] * zz[:, 0]) - (Pout + c["uout"] * zz[:, 1]), ddof=1))
                                                          for zz in [z @ np.linalg.cholesky(np.array([[1.0, r_], [r_, 1.0 + 1e-15]])).T for r_ in mc_r]], style="points")
    res.add_plot("p_rho", f"전기적 손실의 표준불확도 vs 입력·출력 오차 상관 ρ (각 ±{v['rel_u'] * 100:g} %)", ["u_rho", "u_mc"], x_label="상관 ρ", x_unit="", y_label="u(손실)", y_unit="W", kind="xy", level="A + MC",
                 hlines=[{"y": c["uQ"], "label": f"u(열량) {c['uQ']:.3g} W"}, {"y": gap / 2.0, "label": f"차이/2 = {gap / 2:.3g} W"}],
                 proved=f"10 kW급에서 각 0.1 %여도 손실의 표준불확도는 상관 가정에 따라 {u_diff_f(c['uin'], c['uout'], 0.9):.3g}–{u_anti:.3g} W로 달라지고(독립이면 {u0:.3g} W), GUM 식과 seed가 고정된 Monte Carlo가 일치한다는 것을 보였다. "
                 f"어느 ρ에서도 150 W 차이의 절반에 못 미친다.",
                 not_yet="계측기 사양(±0.1 %)을 1σ로 읽었다. 직사각 한계나 k = 2로 읽으면 값이 바뀐다(EX11). ρ는 같은 분석기·동시 취득 같은 근거가 기록될 때만 쓸 수 있다.")
    bins = np.linspace(min(Pe, v["Q_cal"]) - 6 * max(u0, c["uQ"]), max(Pe, v["Q_cal"]) + 6 * max(u0, c["uQ"]), 161)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    he, _ = np.histogram(out[0.0], bins=bins, density=True)
    hq, _ = np.histogram(q_s, bins=bins, density=True)
    res.add_series("d_e", f"전기 P_in − P_out (ρ = 0)", "1/W", ctr.tolist(), he.tolist())
    res.add_series("d_q", "열량 ṁ c_p ΔT", "1/W", ctr.tolist(), hq.tolist())
    res.add_plot("p_dist", "두 측정의 분포 (Monte Carlo): 우연 오차로는 겹치지 않는다", ["d_e", "d_q"], x_label="손실", x_unit="W", y_label="확률 밀도", y_unit="1/W", kind="xy", level="MC",
                 vlines=[{"x": Pe, "label": f"{Pe:.4g} W"}, {"x": v["Q_cal"], "label": f"{v['Q_cal']:.4g} W"}],
                 proved=f"선언한 불확도(전기 {u0:.3g} W, 열량 {c['uQ']:.3g} W)로 만든 두 분포가 겹치지 않는다는 것(차이 = {gap / u_gap:.3g} u_gap)을 보였다 — 차이는 우연 오차가 아니라 어딘가의 편향이다.",
                 not_yet="편향이 어느 쪽에 있는지는 이 그래프가 말하지 않는다. 선언하지 않은 오차원(경계, 과도, 위상)은 분포에 들어 있지 않다.")
    if w7["val"] is not None:
        tt = np.linspace(0.0, 5.0 * v["tau_th"], 201)
        res.add_series("q_t", "냉각수가 가져가는 열 (H7 세계)", "W", tt.tolist(), (Pe + (v["P_prev"] - Pe) * np.exp(-tt / v["tau_th"])).tolist())
        res.add_plot("p_thermal", "H7 세계: 이전 시험의 열이 빠지는 동안 잰 열량", ["q_t"], x_label="이전 시험 후 시간", x_unit="s", y_label="냉각수 열", y_unit="W", kind="xy", level="A",
                     vlines=[{"x": w7["val"], "label": f"측정 {w7['val']:.4g} s"}], hlines=[{"y": Pe, "label": f"참 손실 {Pe:.4g} W"}, {"y": v["Q_cal"], "label": f"관측 {v['Q_cal']:g} W"}],
                     proved=f"이전 시험 손실 {v['P_prev']:g} W, 열 시정수 {v['tau_th']:g} s라면 {w7['val']:.4g} s 뒤에 잰 열량이 350 W로 보인다는 것을 1차 열 모델(닫힌 식과 ODE event 일치)로 보였다.",
                     not_yet="1차 열 회로 가정이고 τ와 이전 시험 손실은 ASSUMED다. 실제 열 경로는 여러 시정수를 가진다 — 냉각수 온도 추세 기록이 필요하다.")
    tsel = v["test"]
    if tsel == "heater":
        Ph = np.linspace(0.0, 2.0 * v["P_heat"], 21)
        tk = []
        for key, lab_ in (("H4", "H4 (보조·pump 열)"), ("H5", "H5 (ΔT offset)"), ("H6", "H6 (배율)"), ("H1", "나머지 세계 (열량계 정확)")):
            w = W[key]
            if key == "H4":
                y = Ph + w["val"]
            elif key == "H5":
                y = Ph + c["G"] * w["val"]
            elif key == "H6":
                y = Ph * (1.0 + w["val"])
            else:
                y = Ph
            res.add_series(f"t_{key}", lab_, "W", Ph.tolist(), y.tolist(), dash=(key == "H1"))
            tk.append(f"t_{key}")
        res.add_plot("p_test", "시험: 컨버터를 끄고 알려진 heater로 열량계를 읽는다", tk, x_label="heater 전력", x_unit="W", y_label="열량계 읽음", y_unit="W", kind="xy", level="A",
                     proved="H4·H5는 heater 전력과 무관한 offset(+150 W), H6은 비례 오차(×1.75), 나머지는 정확하다는 것을 보였다. heater 두 수준이 offset과 배율을 가르고, H4와 H5는 보조 전력 측정으로 가른다.",
                     not_yet="heater 자체의 전력 측정 불확도와 heater가 냉각수에 열을 전부 주는지(배관 손실)는 따로 확인해야 한다.")
    elif tsel != "none":
        tk = []
        for h in hyps:
            w = W[h.key]
            if w["val"] is None:
                continue
            e1, q1 = f_observe(v, c, h.key, w, tsel)
            after = q1 - e1
            if tsel == "aux" and h.key == "H4":
                after = 0.0
            if tsel == "swap" and h.key == "H1":
                after = 0.0  # the channel mismatch is found and corrected
            res.add_series(f"t_{h.key}", f"{h.key} {h.label}", "W", [0.0, 1.0], [gap, after])
            tk.append(f"t_{h.key}")
        res.add_plot("p_test", "시험: " + dict(F_TESTS)[tsel] + " — 시험 전(0)과 후(1)의 열량 − 전기 차이", tk, x_label="0 = 시험 전, 1 = 시험·보정 후", x_unit="", y_label="차이", y_unit="W", kind="xy", level="A",
                     proved={
                         "aux": "보조전원·pump 전력을 따로 재서 경계를 맞추면 H4 세계의 차이만 0이 되고 나머지 일곱 세계는 150 W가 그대로라는 것을 보였다.",
                         "sync": "같은 분석기로 입력·출력을 동시에(정수 주기 창, DC-link ½CV² 보정) 읽으면 H3(비동시)과 H8(전기 저장에너지) 세계의 차이만 사라진다는 것을 보였다.",
                         "steady": "열 시정수의 5배를 기다려 다시 재면 열 저장 과도 세계(H7)의 차이만 거의 0이 되고 나머지는 그대로라는 것을 보였다.",
                         "pf": "PF 1 부하(또는 위상 교정 센서)로 바꾸면 위상 오차 세계(H2)의 차이만 사라진다는 것을 보였다.",
                         "swap": "두 채널을 같은 도체에 물려 교차 확인하면 이득 오차 세계(H1)에서만 채널 불일치가 드러나고(보정하면 차이 0), 나머지는 일치한다는 것을 보였다.",
                     }[tsel] + " 0으로 떨어지는 선이 이 시험에 반응하는 가설이다.",
                     not_yet="한 시험이 차이를 없애도 다른 편향이 함께 있을 수 있다 — 남은 차이를 선언한 불확도와 다시 비교한다. 각 시험의 실제 실행 조건(같은 부하·온도)을 맞춰야 한다.")
    res.circuit = {"diagram": circuit_f(v).to_json(), "intervals": [], "plot_group": ""}
    unresolved(res, f"heater 교정({groups['heater']}묶음), 보조 전력({groups['aux']}), 동시 취득({groups['sync']}), 정상상태({groups['steady']}), PF 1({groups['pf']}), 채널 교차({groups['swap']})를 조합해야 한다.")
    res.verdict("INFO", f"선언한 불확도로는 차이가 설명되지 않는다: 150 W = {gap / u_gap:.3g} u_gap (u_gap {u_gap:.3g} W) — 편향을 찾는 문제다")
    res.verdict("MISSING_INPUT", "계측기 교정 성적서·사양 해석(1σ/한계), 위상 사양, 읽은 시각, 보조전원 경계, 냉각수 물성·유량계 교정, 이전 시험 이력이 없다")
    res.assumptions += [
        f"P_in {Pin:g} W DC, η {v['eta']:g} → P_out {Pout:g} W (3상 AC, PF {v['PF']:g}), 각 ±{v['rel_u'] * 100:g} %를 표준불확도로 (EX11 교재 가정)",
        f"냉각수 50 % glycol 가정: 밀도 {v['rho_c']:g} kg/m³, c_p {v['cp']:g} J/(kg·K), 유량 {v['flow_lpm']:g} L/min → ṁc_p {c['G']:.4g} W/K, 열량 350 W의 ΔT {c['dT']:.4g} K",
        f"세계 가정: drift {v['drift'] * 100:g} %/s (H3), 이전 시험 손실 {v['P_prev']:g} W·τ {v['tau_th']:g} s (H7), 측정 창 {v['T_w']:g} s·C {v['C_dc'] * 1e3:g} mF·{v['V_dc']:g} V (H8)",
        f"Monte Carlo {n}개, seed {int(v['seed'])}",
    ]
    res.not_valid_for += ["계측기 인증·교정", "실제 열 경로(다중 시정수)", "고객 컨버터의 실제 효율", "고객 회로의 근본원인 확정"]
    res.interpretation = (
        f"효율 98 %면 손실은 입력의 2 %다: 큰 두 숫자의 차라서 각 0.1 % 오차도 손실에는 {u0:.3g} W(독립)–{u_anti:.3g} W(최악 상관)가 된다. 그래도 150 W 차이는 선언한 불확도의 {gap / u_gap:.3g}배라 우연이 아니다 — 어딘가에 편향이 있다. "
        "여덟 세계가 같은 두 숫자를 만든다: 네 개는 전기 쪽이 틀리고(이득, 위상, 동시성, ½CV² 과도: 참 손실 350 W), 네 개는 열량 쪽이 다른 것을 잰다(보조·pump 열, ΔT offset, 물성 배율, 열 저장 과도: 참 손실 200 W). "
        "필요한 편향의 크기가 첫 걸러내기다(배율 75 %는 어렵고, ΔT 0.31 K offset은 미교정 센서면 쉽다). heater 교정·보조 전력·동시 취득·정상상태·PF 1·채널 교차로 가른다. 1쪽 메모 영어 틀: " + MEMO_EN_TEMPLATE
    )
    return res


def params_f() -> list[Param]:
    g_e, g_q, g_w, g_m = "전기 측정", "열량 측정", "세계 가정", "Monte Carlo"
    return [
        Param("P_in", "입력 전력 P_in (DC)", "W", 10000.0, "W", vmin=100, vmax=1e7, source="TEXTBOOK", source_note="10 kW급", group=g_e),
        Param("eta", "측정 효율 η", "", 0.98, "", vmin=0.5, vmax=0.9999, source="TEXTBOOK", source_note="98 % → 손실 200 W", group=g_e),
        Param("rel_u", "P_in·P_out 상대 불확도 (1σ로 읽음)", "", 1e-3, "", vmin=0.0, vmax=0.1, source="TEXTBOOK", source_note="±0.1 %", group=g_e),
        Param("rho", "입력·출력 오차 상관 ρ", "", 0.0, "", vmin=-1.0, vmax=1.0, source="ASSUMED", source_note="근거 없으면 0", group=g_e),
        Param("PF", "AC 출력 역률", "", 0.8, "", vmin=0.05, vmax=1.0, source="ASSUMED", source_note="모터 부하 inverter 가정 (위상 오차의 영향)", group=g_e),
        Param("Q_cal", "열량 추정 손실", "W", 350.0, "W", vmin=1, vmax=1e6, source="TEXTBOOK", source_note="350 W", group=g_q),
        Param("flow_lpm", "냉각수 유량 [L/min]", "", 8.0, "", vmin=0.1, vmax=1000, source="ASSUMED", group=g_q),
        Param("rho_c", "냉각수 밀도 [kg/m³]", "", 1070.0, "", vmin=500, vmax=2000, source="ASSUMED", source_note="50 % glycol", group=g_q),
        Param("cp", "냉각수 비열 c_p [J/(kg·K)]", "", 3400.0, "", vmin=500, vmax=5000, source="ASSUMED", source_note="50 % glycol", group=g_q),
        Param("u_flow", "유량 상대 불확도", "", 0.01, "", vmin=0.0, vmax=0.5, source="ASSUMED", group=g_q),
        Param("u_cp", "c_p 상대 불확도", "", 0.01, "", vmin=0.0, vmax=0.5, source="ASSUMED", group=g_q),
        Param("u_rhoc", "밀도 상대 불확도", "", 0.005, "", vmin=0.0, vmax=0.5, source="ASSUMED", group=g_q),
        Param("u_dT", "ΔT 표준불확도", "K", 0.02, "K", vmin=0.0, vmax=10, source="ASSUMED", source_note="교정한 RTD 쌍", group=g_q),
        Param("P_heat", "heater 교정 전력 (낮은 수준)", "W", 200.0, "W", vmin=1, vmax=1e5, source="ASSUMED", group=g_q),
        Param("P_aux_known", "사양상 보조 전력", "W", 10.0, "W", vmin=0, vmax=1e4, source="ASSUMED", group=g_w),
        Param("drift", "H3: 전력 drift (상대, 초당)", "", 5e-4, "", vmin=1e-6, vmax=0.1, source="ASSUMED", source_note="0.05 %/s", group=g_w),
        Param("P_prev", "H7: 이전 시험의 손실", "W", 500.0, "W", vmin=1, vmax=1e6, source="ASSUMED", group=g_w),
        Param("tau_th", "H7: 열 시정수", "s", 600.0, "s", vmin=1, vmax=1e5, source="ASSUMED", group=g_w),
        Param("T_w", "H8: 전력 평균 창", "s", 0.2, "s", vmin=1e-3, vmax=1e3, source="ASSUMED", group=g_w),
        Param("C_dc", "H8: DC-link C", "F", 1e-3, "mF", vmin=1e-6, vmax=1.0, source="ASSUMED", source_note="EX10과 같은 1 mF", group=g_w),
        Param("V_dc", "H8: DC-link 전압", "V", 800.0, "V", vmin=1, vmax=3000, source="ASSUMED", source_note="EX10과 같은 800 V", group=g_w),
        Param("mc_n", "Monte Carlo 표본 수", "", 200000, "", vmin=1000, vmax=5_000_000, kind="int", source="ASSUMED", group=g_m),
        Param("seed", "난수 seed", "", 20261001, "", vmin=0, vmax=2**31 - 1, kind="int", source="ASSUMED", group=g_m),
        _test_param(F_TESTS),
    ]


EXP_F = Experiment(
    key="case_f",
    title="CASE F — 효율 98 %인데 열량이 맞지 않는다: 전기 200 W와 열량 350 W 중 무엇을 믿나",
    goal=(
        "증상: 입력·출력 차이로는 손실 200 W(10 kW, η 98 %)인데 calorimetric 추정은 350 W다. 확인된 것: 두 숫자와 각 전력계의 사양(±0.1 %). "
        "모르는 것: 사양의 의미(1σ/한계), 두 측정의 동시성·상관, 출력 측 위상 오차, 보조전원·pump가 어느 경계에 있는지, 냉각수 물성·유량·ΔT 교정, 측정 전 이력. 먼저 Q1–Q3에 답을 적고 실행한다. "
        "실행하면 손실 불확도를 GUM과 seed가 고정된 Monte Carlo로 따로 계산해 150 W가 우연 오차로는 설명되지 않음을 보이고, 이득·위상·동시성·보조 경계·ΔT offset·물성 배율·열 저장·½CV² 과도 여덟 세계를 "
        "같은 두 숫자에 맞춘 뒤 heater 교정·보조 전력·동시 취득·정상상태·PF 1·채널 교차 시험이 각 세계에서 무엇을 보일지 비교한다."
    ),
    params=params_f(),
    presets=[
        Preset("observed", "관측 그대로 (200 W vs 350 W)", {}, "CASE F 증상", ("nominal", "reference")),
        Preset("heater", "첫 시험: heater로 열량계 교정", {"test": "heater"}, "H4·H5 offset, H6 배율", ("reference",)),
        Preset("sync", "같은 분석기 동시 취득", {"test": "sync"}, "H3·H8", ("variant", "reference")),
        Preset("aux", "보조 전력 측정", {"test": "aux"}, "H4", ("variant",)),
        Preset("steady", "정상상태 재측정", {"test": "steady"}, "H7", ("variant",)),
        Preset("pf", "PF 1 / 위상 교정", {"test": "pf"}, "H2", ("variant",)),
        Preset("swap", "채널 교차 확인", {"test": "swap"}, "H1", ("variant",)),
        Preset("rho09", "ρ = 0.9 (근거 없음)", {"rho": 0.9}, "불확도가 작아 보일 뿐 차이는 그대로", ("corner",)),
    ],
    run=run_case_f,
    model_level="A + MC (GUM 1차 전파, seed 고정 Monte Carlo, 경계 대수, 1차 열 모델)",
    suggested_change="판별 시험을 ‘heater로 열량계 교정’으로 바꾼다 (다음에 ‘동시 취득’, ‘보조 전력 측정’).",
    prediction=Prediction(
        "각 ±0.1 % 전력계 두 대로 잰 10 kW급 손실 200 W의 표준불확도(독립)는 대략 얼마이고, 150 W 차이를 설명하나?",
        ["약 0.2 W — 무시해도 된다", "약 14 W — 150 W 차이는 설명하지 못한다", "약 150 W — 차이는 측정 불확도 안이다", "모르겠다"],
        "약 14 W — 150 W 차이는 설명하지 못한다",
        "u = √(10² + 9.8²) ≈ 14.0 W(독립). 상관 ρ에 따라 0.2 W(완전 상관)–19.8 W(반상관)까지 달라지지만 어느 경우도 150 W에 못 미친다. 열량 쪽 불확도(약 11 W)를 합쳐도 150 W는 8배가 넘는다 — 우연 오차가 아니라 편향을 찾아야 한다.",
        ["u_e_ind", "u_Q", "z_gap"],
        handcalc=[{"key": "u_e_ind", "label": "√(u_in² + u_out²)", "unit": "W"}, {"key": "u_ex11_ind", "label": "EX11 조건 15.40157 W", "unit": "W"}],
    ),
    suggested={"test": "heater"},
    student=(
        "효율이 높을수록 손실은 두 큰 숫자(입력·출력)의 작은 차이라 각 측정의 작은 오차가 손실에서는 크게 보인다. 그래도 계측기 사양으로 계산한 불확도는 수십 W 수준이라 150 W 차이를 설명하지 못한다. "
        "그렇다면 어느 한쪽이 다른 것을 재고 있다: 전기 쪽(이득, 위상, 읽은 시각, 저장에너지)이 틀렸거나 열 쪽(보조·pump 열, 온도 센서, 물성, 아직 식는 중)이 다른 경계를 잰다."
    ),
    expert=(
        "교재 해설(CASE F): power meter 정확도·위상오차·동시성, auxiliary 소비 포함 여부, 계측 경계, 저장에너지 과도, coolant 유량·ΔT 오차를 확인한다. 높은 효율에서는 큰 두 숫자의 차로 작은 손실을 구하므로 상대오차가 커진다. "
        "10 kW급의 input/output 각각 ±0.1 %라도 loss uncertainty는 독립성 가정에 따라 수십 W가 될 수 있다. 오차 전파를 선언한다. "
        "추가: ① 사양 ±0.1 %를 1σ로 읽으면 독립 14.0 W, 같은 분석기·동시 취득 근거가 있어 ρ = 0.9면 4.4 W, 최악 19.8 W — 어느 쪽이든 150 W의 원인은 아니다(EX11의 15.40157 W·4.87487 W와 같은 식). "
        "② 필요한 편향 크기로 걸러낸다: 출력 채널 이득 −1.5 %(사양 15배), AC 위상 1.2°(PF 0.8), 읽은 시각 31 s, 보조·pump 150 W, ΔT offset 0.31 K(ΔT 0.72 K 중), 물성 배율 75 %(어려움), 이전 시험 후 7분, 창 0.2 s 동안 DC-link 38 V 하강. "
        "③ 네 세계는 참 손실 350 W(전기 쪽 편향), 네 세계는 200 W(열 쪽 경계·편향)다 — 효율을 어느 쪽으로 보고할지가 결정 사항이다. ④ 판별: heater 2수준(offset vs 배율), 보조 전력, 동시 취득과 ½CV² 보정, 5τ 뒤 재측정, PF 1 또는 위상 교정, 채널 교차."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: 효율을 98 %와 96.5 % 중 어느 쪽으로 보고할지 정하고, 그 근거를 측정으로 남긴다. 확인한 조건: 전기 200 W, 열량 350 W, 전력계 각 ±0.1 %. "
        "모르는 조건: 사양 해석(1σ/한계), 두 측정의 동시성과 상관, 출력 위상 오차, 보조전원·pump의 경계, 냉각수 물성·유량·ΔT 교정, 측정 전 이력. "
        "유력 가설과 반례: 선언한 불확도(전기 약 14 W, 열량 약 11 W)로는 150 W가 설명되지 않으므로 편향이다. 보조·pump 열이 열 경계 안에 있거나 ΔT offset이면 전기 200 W가 맞고, 출력 이득·위상·동시성·½CV² 과도이면 열량 350 W가 맞다. "
        "현재 권고: 효율을 확정하기 전에 heater 2수준 교정, 보조 전력 측정, 같은 분석기 동시 취득(정수 주기 창, DC-link 전압 기록), 5τ 뒤 재측정, PF 1 확인, 채널 교차를 한다. "
        "부작용: 측정 경계를 바꾸면 이전 보고 수치와 비교할 수 없게 된다 — 경계를 문서로 고정한다. 완료 판정: 두 방법의 차이가 선언한 합성 불확도의 2배 안에 들어오면 닫는다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the measurement itself: the 150 W gap is about eight times the declared combined uncertainty, so one of the two methods carries a bias. "
        "My current hypothesis is a boundary or sensor bias on the calorimetric side, such as pump or auxiliary heat entering the coolant or a temperature-sensor offset, but an output-channel gain or phase error, non-simultaneous readings or a DC-link energy change could produce a similar symptom on the electrical side. "
        "I would separate them by calibrating the calorimeter with a known heater at two levels, measuring the auxiliary power, re-acquiring input and output simultaneously on one analyzer and repeating after thermal steady state. "
        "If the heater shows a constant offset, I recommend correcting the calorimeter and would recheck the efficiency at the same boundary."
    ),
    questions=[
        q_hyp(
            "F",
            "효율 98 %(전기 손실 200 W)인데 열량 추정은 350 W다. 가설 3개는?",
            "① 전력계 정확도·이득(출력 채널이 높게 읽음, range·CT 비 설정) ② AC 측 위상 오차(PF가 낮을수록 영향 큼) ③ 입력·출력 측정의 비동시성 ④ 보조전원·pump 열이 열 경계 안, 전기 경계 밖 ⑤ ΔT offset·유량·c_p 오차 ⑥ 열 저장·½CV² 과도. "
            "높은 효율에서는 두 큰 숫자의 차라 상대오차가 커지지만, 선언한 불확도(수십 W)로는 150 W를 설명하지 못한다는 점을 말한다.",
            "The electrical loss is 200 W at 98 % efficiency, but the calorimetric estimate is 350 W: give three hypotheses.",
            "Meter accuracy or a gain or range error, an AC-side phase error, non-simultaneous readings, auxiliary or pump heat inside the thermal but outside the electrical boundary, a coolant ΔT offset or flow and heat-capacity error, and a stored-energy transient. The declared uncertainty of a few tens of watts does not explain 150 W.",
            ["정확도·이득", "위상 오차", "동시성", "보조·경계", "ΔT·유량", "저장에너지 과도", "오차 전파 선언"],
        ),
        q_wave(
            "F",
            "① 같은 분석기에서 입력·출력 전력을 동시에, 정수 주기 창으로(각 채널 range·CT 비·위상 보정 설정 포함) ② 냉각수 입·출구 온도와 유량의 시간 추세(정상상태 확인), heater 교정 기록 ③ 보조전원·pump 전력과 DC-link 전압 추세. "
            "측정 경계 그림과 각 계측기의 사양 해석(1σ/한계)을 함께 기록한다.",
            "Input and output power acquired simultaneously on one analyzer over an integer number of periods, with range, sensor ratio and phase settings; coolant inlet and outlet temperatures and flow over time with a heater calibration; and the auxiliary and pump power with the DC-link voltage trend. Record a boundary drawing and how each specification is interpreted.",
            ["동시 취득·정수 주기", "냉각수 온도 추세·heater", "보조 전력·DC-link", "경계 그림·사양 해석"],
        ),
        q_first(
            "F",
            "컨버터를 끄고 알려진 heater 두 수준으로 열량계를 읽는다: offset(+150 W 일정)이면 H4(보조·pump 열) 또는 H5(ΔT offset), 비례면 H6, 정확하면 전기 쪽(H1·H2·H3·H8)이나 열 과도(H7)다. "
            "이어서 보조 전력 측정이 H4/H5를, 동시 취득이 H3·H8을, 5τ 뒤 재측정이 H7을, PF 1이 H2를, 채널 교차가 H1을 가른다.",
            "With the converter off, read the calorimeter with a known heater at two levels: a constant offset points to auxiliary heat or a temperature offset, a proportional error to flow or heat capacity, and a correct reading to the electrical side or a thermal transient. Then the auxiliary power, a simultaneous acquisition, a repeat after five thermal time constants, a unity power factor test and a channel cross-check separate the rest.",
            ["heater 2수준", "offset vs 비례", "보조 전력", "동시 취득", "정상상태"],
        ),
        q_memo(
            "F",
            "목표: 효율 98 % vs 96.5 % 결정과 근거. 확인: 전기 200 W, 열량 350 W, ±0.1 %. 모름: 사양 해석, 동시성·상관, 위상, 보조 경계, 냉각수 교정, 이력. 가설과 반례: 선언 불확도(약 14 W·11 W)로 150 W 설명 불가 → 편향; "
            "열 쪽(보조·ΔT·배율·열 과도)이면 200 W, 전기 쪽(이득·위상·동시성·½CV²)이면 350 W. 권고: heater 2수준, 보조 전력, 동시 취득, 5τ 재측정, PF 1, 채널 교차. 부작용: 경계 변경 시 이전 수치와 비교 불가 → 경계 문서화. "
            "완료 판정: 두 방법 차이가 합성 불확도 2배 안.",
            MEMO_EN_TEMPLATE.replace("___", "the measurement boundary and its uncertainty", 1),
        ),
    ],
    circuit="fl12_f",
    textbook=[TB_15, TB_E11, TB_17],
    reference_presets=["observed", "heater", "sync"],
    runtime_hint="seconds",
    claim_limit="GUM·Monte Carlo와 경계 대수로 가설별 예측을 비교한다. 계측기 교정 상태와 고객 컨버터의 실제 효율·근본원인을 확정하지 않는다.",
)


# ======================================================================================
# CASE G - R_g raised, but one EMC peak barely moves
# ======================================================================================

G_TESTS = [
    ("none", "시험 전: R_g 2.5 Ω 기준 spectrum"),
    ("rg", "R_g만 변경 (edge 감속)"),
    ("cmdm", "CM/DM 분리 측정"),
    ("cable", "cable·접지 용량 변경"),
    ("aux_off", "보조 컨버터 끄기"),
    ("ctrl", "제어 클록·변조 설정 변경"),
    ("snub", "전력 loop 감쇠 (RC snubber)"),
]


def pwl_fourier(ts: np.ndarray, vs: np.ndarray, T: float, f: np.ndarray) -> np.ndarray:
    """Exact Fourier-series coefficients c(f) = (1/T) int_0^T v(t) e^{-j 2 pi f t} dt of a continuous periodic PWL waveform.

    Integration by parts per linear segment; the boundary terms telescope for a continuous periodic waveform, leaving
    c = (1/T) sum_i s_i (e^{-j w t_b} - e^{-j w t_a}) / w^2 with the segment slopes s_i."""
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    out = np.zeros(w.shape, dtype=complex)
    for a in range(len(ts) - 1):
        ta, tb = ts[a], ts[a + 1]
        if tb <= ta:
            continue
        s = (vs[a + 1] - vs[a]) / (tb - ta)
        if s != 0.0:
            out += s * (np.exp(-1j * w * tb) - np.exp(-1j * w * ta)) / (w * w)
    return out / T


def trapezoid(V: float, T: float, D: float, tr: float) -> tuple[np.ndarray, np.ndarray]:
    """Switch-node trapezoid: 50 %-to-50 % width D T, equal rise and fall times tr."""
    t0 = 0.0
    ts = np.array([t0, tr, D * T, D * T + tr, T])
    vs = np.array([0.0, V, V, 0.0, 0.0])
    return ts, vs


def g_consts(v: dict) -> dict:
    tr = {k: v["Qgd"] * (v[k] + v["Rg_int"]) / v["dVg"] for k in ("Rg1", "Rg2")}
    return {"T": 1.0 / v["fsw"], "tr1": tr["Rg1"], "tr2": tr["Rg2"]}


def h_loop(f, L, C, R):
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    return 1.0 / (1.0 - w * w * L * C + 1j * w * R * C)


def y_cm(f, L, C, R):
    w = 2.0 * math.pi * np.asarray(f, dtype=float)
    return 1j * w * C / (1.0 - w * w * L * C + 1j * w * R * C)


@dataclass
class WorldG:
    key: str
    L_loop: float
    R_loop: float
    L_cab: float
    R_d: float
    s_aux: float = 0.0
    L_aux: float = 0.0
    s_ctrl: float = 0.0
    f_ctrl: float = 0.0
    ok: bool = True


def g_main(v: dict, w: WorldG, f: np.ndarray, tr: float, Cpar: float | None = None, R_loop: float | None = None) -> np.ndarray:
    """CM measurement proxy of the main stage at the harmonic frequencies f: |c(f)| |H_loop| |Y_cm| R_m (one-sided amplitude)."""
    c = g_consts(v)
    ts, vs = trapezoid(v["V"], c["T"], v["D"], tr)
    cf = pwl_fourier(ts, vs, c["T"], f)
    Cp = v["Cpar"] if Cpar is None else Cpar
    Rl = w.R_loop if R_loop is None else R_loop
    return 2.0 * np.abs(cf) * np.abs(h_loop(f, w.L_loop, v["Coss"], Rl)) * np.abs(y_cm(f, w.L_cab, Cp, v["Rm"] + w.R_d)) * v["Rm"]


def g_aux(v: dict, w: WorldG, f: np.ndarray) -> np.ndarray:
    """Auxiliary converter lines (its own trapezoid and leakage ringing), scaled by the calibrated coupling s_aux."""
    Ta = 1.0 / v["f_aux"]
    ts, vs = trapezoid(v["V_aux"], Ta, 0.5, v["tr_aux"])
    cf = pwl_fourier(ts, vs, Ta, f)
    Raux = math.sqrt(w.L_aux / v["C_aux"]) / v["Q_aux"] if w.L_aux > 0 else 0.0
    return w.s_aux * 2.0 * np.abs(cf) * np.abs(h_loop(f, w.L_aux, v["C_aux"], Raux))


def g_lines(v: dict, w: WorldG, tr: float, fmax: float, test: str = "none") -> dict:
    """All lines of world w: {'f': frequencies, 'a': amplitudes, 'src': source tags} (main CM, aux DM, control CM)."""
    kmax = int(fmax / v["fsw"])
    fk = v["fsw"] * np.arange(1, kmax + 1, 2)
    Cp = v["Cpar"] * (v["k_cable"] if test == "cable" else 1.0)
    Rl = w.R_loop * (v["k_snub"] if test == "snub" else 1.0)
    a_main = g_main(v, w, fk, tr, Cpar=Cp, R_loop=Rl)
    F, A, S = [fk], [a_main], [np.array(["main"] * fk.size)]
    if w.s_aux > 0 and test != "aux_off":
        fa = v["f_aux"] * np.arange(1, int(fmax / v["f_aux"]) + 1, 2)
        F.append(fa)
        A.append(g_aux(v, w, fa))
        S.append(np.array(["aux"] * fa.size))
    if w.s_ctrl > 0:
        fc = w.f_ctrl * (v["k_ctrl"] if test == "ctrl" else 1.0)
        F.append(np.array([fc]))
        A.append(np.array([w.s_ctrl]))
        S.append(np.array(["ctrl"]))
    return {"f": np.concatenate(F), "a": np.concatenate(A), "src": np.concatenate(S)}


def g_read(lines: dict, f0: float, rbw: float, drop: tuple = ()) -> float:
    """Receiver-like reading at f0: power sum of the lines inside the resolution bandwidth (sources in `drop` removed)."""
    m = np.abs(lines["f"] - f0) <= 0.5 * rbw
    if drop:
        m &= ~np.isin(lines["src"], list(drop))
    return float(math.sqrt(np.sum(lines["a"][m] ** 2))) if np.any(m) else 0.0


def g_peak(lines: dict, f_lo: float, f_hi: float, drop: tuple = ()) -> tuple[float, float]:
    m = (lines["f"] >= f_lo) & (lines["f"] <= f_hi)
    if drop:
        m &= ~np.isin(lines["src"], list(drop))
    i = int(np.argmax(np.where(m, lines["a"], -1.0)))
    return float(lines["f"][i]), float(lines["a"][i])


def g_worlds(v: dict) -> dict:
    c = g_consts(v)
    f0 = v["f_obs"]
    w0 = 2.0 * math.pi * f0
    base = WorldG("base", v["L_loop"], v["R_loop"], v["L_cab"], 0.0)
    fmax, rbw = v["f_max"], v["rbw"]
    a_base = g_read(g_lines(v, base, c["tr1"], fmax), f0, rbw)
    target = 10.0 ** (v["prom_obs"] / 20.0)
    W = {"base": base}
    # H1: power-loop ringing at f_obs (L_loop from the frequency, damping R_loop from the prominence)
    L1 = 1.0 / (w0 * w0 * v["Coss"])

    def prom1(R):
        return g_read(g_lines(v, WorldG("H1", L1, R, v["L_cab"], 0.0), c["tr1"], fmax), f0, rbw) / a_base - target

    W["H1"] = WorldG("H1", L1, brentq(prom1, 1e-4, 1e4, xtol=1e-14, rtol=1e-13), v["L_cab"], 0.0) if prom1(1e-4) * prom1(1e4) < 0 else WorldG("H1", L1, v["R_loop"], v["L_cab"], 0.0, ok=False)
    # H2: CM path (cable / ground) resonance at f_obs; extra path damping R_d from the prominence
    L2 = 1.0 / (w0 * w0 * v["Cpar"])

    def prom2(Rd):
        return g_read(g_lines(v, WorldG("H2", v["L_loop"], v["R_loop"], L2, Rd), c["tr1"], fmax), f0, rbw) / a_base - target

    W["H2"] = WorldG("H2", v["L_loop"], v["R_loop"], L2, brentq(prom2, 0.0, 1e5, xtol=1e-12, rtol=1e-13)) if prom2(0.0) * prom2(1e5) < 0 else WorldG("H2", v["L_loop"], v["R_loop"], L2, 0.0, ok=False)
    # H3: auxiliary converter ringing at f_obs (L_aux from the frequency), coupling from the prominence (power sum)
    L3 = 1.0 / (w0 * w0 * v["C_aux"])
    w3 = WorldG("H3", v["L_loop"], v["R_loop"], v["L_cab"], 0.0, s_aux=1.0, L_aux=L3)
    a_aux1 = g_read({"f": np.array([f0]), "a": g_aux(v, w3, np.array([f0])), "src": np.array(["aux"])}, f0, rbw)
    on_grid = abs(round(f0 / v["f_aux"]) * v["f_aux"] - f0) < 1e-6 * f0 and int(round(f0 / v["f_aux"])) % 2 == 1
    w3.s_aux = a_base * math.sqrt(target**2 - 1.0) / a_aux1 if on_grid and a_aux1 > 0 else 0.0
    w3.ok = on_grid
    W["H3"] = w3
    # H4: control clock / modulation line at f_obs (control side, not the power-stage edge)
    W["H4"] = WorldG("H4", v["L_loop"], v["R_loop"], v["L_cab"], 0.0, s_ctrl=a_base * math.sqrt(target**2 - 1.0), f_ctrl=f0)
    return W


def circuit_g(v: dict) -> Circuit:
    c = Circuit("fl12_g", 700, 290, title="noise source → 경로 → 측정 (relative proxy): edge, loop ringing, CM 경로, 보조 컨버터, 제어")
    sw = c.add("vsource", "SW", 70, 140, 90, "switch node", f"{v['V']:g} V, t_r(R_g)", lpos=(96, 136, "start"))
    ll = c.add("inductor", "LL", 190, 50, 0, "L_loop", f"{v['L_loop'] * 1e9:g} nH", lpos=(190, 26, "middle"))
    co = c.add("capacitor", "CO", 280, 140, 90, "C_oss", f"{v['Coss'] * 1e9:g} nF", lpos=(302, 136, "start"))
    cp = c.add("capacitor", "CP", 400, 50, 0, "C_par", f"{v['Cpar'] * 1e12:g} pF", lpos=(400, 14, "middle"))
    lc = c.add("inductor", "LC", 500, 50, 0, "L_cab", f"{v['L_cab'] * 1e9:g} nH", lpos=(500, 26, "middle"))
    rm = c.add("resistor", "RM", 600, 140, 90, "R_m (측정 proxy)", f"{v['Rm']:g} Ω", lpos=(578, 136, "end"))
    c.add("ground", "G", 340, 240)
    c.wire("w1", sw["a"], (70, 50), ll["a"])
    c.wire("w2", ll["b"], (280, 50), co["a"])
    c.wire("w3", (280, 50), cp["a"])
    c.wire("w4", cp["b"], lc["a"])
    c.wire("w5", lc["b"], (600, 50), rm["a"])
    c.wire("w6", rm["b"], (600, 240), (70, 240), sw["b"])
    c.wire("w7", co["b"], (280, 240))
    c.dot((280, 50), (280, 240))
    c.add("block", "AUX", 450, 190, 0, "보조 컨버터 (DM)", w=130, h=30)
    c.add("block", "CTL", 450, 270, 0, "제어 클록·변조", w=130, h=26)
    c.text(400, 92, "CM 경로 (chassis)", "note")
    return c


def run_case_g(v: dict) -> Result:
    res = Result("FL12", "case_g", "A (PWL 정확 Fourier 계수 × 선형 경로 전달함수) — relative proxy")
    c = g_consts(v)
    f0, fmax, rbw = v["f_obs"], v["f_max"], v["rbw"]
    W = g_worlds(v)
    hyps = [
        Hyp("H1", "gate edge가 여기하는 전력 loop ringing", "L_loop–C_oss 공진이 f_obs에 있다 (edge가 source, loop가 공진)", "L_loop, R_loop"),
        Hyp("H2", "cable·접지 LC 공진 (CM 경로)", "C_par–L_cab 경로 공진이 f_obs에 있다", "L_cab, R_d"),
        Hyp("H3", "보조 컨버터", "보조 컨버터의 누설 ringing 선이 f_obs에 있다 (주 전력단 R_g와 무관)", "L_aux, 결합"),
        Hyp("H4", "제어 클록·변조", "제어 쪽 협대역 성분이 f_obs에 선을 만든다", "결합"),
    ]
    tr1, tr2 = c["tr1"], c["tr2"]
    base_lines = g_lines(v, W["base"], tr1, fmax)
    a_base = g_read(base_lines, f0, rbw)
    read0 = {}
    for h in hyps:
        w = W[h.key]
        read0[h.key] = g_read(g_lines(v, w, tr1, fmax), f0, rbw) if w.ok else float("nan")
    dvdt1, dvdt2 = v["V"] / tr1, v["V"] / tr2
    res.add_metric("tr1", f"t_r @ R_g {v['Rg1']:g} Ω = Q_gd(R_g + R_g,int)/ΔV_g", tr1, "s", basis=f"dv/dt {dvdt1 * 1e-9:.3g} V/ns")
    res.add_metric("tr2", f"t_r @ R_g {v['Rg2']:g} Ω", tr2, "s", basis=f"dv/dt {dvdt2 * 1e-9:.3g} V/ns")
    res.add_metric("icm1", "C_par·dv/dt (R_g 기준)", v["Cpar"] * dvdt1, "A", basis="EX09 screen 식 — peak 변위전류, spectrum 아님")
    res.add_metric("d_dvdt", "고객 기대: C·dv/dt 비 20·log10(t_r1/t_r2)", 20 * math.log10(tr1 / tr2), "dB")
    sinc = lambda x: math.sin(x) / x if x != 0 else 1.0  # noqa: E731
    d_edge = 20 * math.log10(abs(sinc(math.pi * f0 * tr2)) / abs(sinc(math.pi * f0 * tr1)))
    res.add_metric("d_edge", f"{f0 / 1e6:.4g} MHz에서 실제 edge 영향 20·log10|sinc(πf t_r2)/sinc(πf t_r1)|", d_edge, "dB", basis="trapezoid 고조파 비 (닫힌 식) — 1/(πt_r) 아래에서는 거의 0")
    res.add_metric("f2_1", "두 번째 corner 1/(π t_r1)", 1.0 / (math.pi * tr1), "Hz")
    res.add_metric("f2_2", "두 번째 corner 1/(π t_r2)", 1.0 / (math.pi * tr2), "Hz")
    cal_txt = {}
    for h in hyps:
        w = W[h.key]
        if not w.ok:
            cal_txt[h.key] = "재현 불가"
            continue
        if h.key == "H1":
            cal_txt[h.key] = f"L_loop {w.L_loop * 1e9:.4g} nH (C_oss {v['Coss'] * 1e9:g} nF), R_loop {w.R_loop:.3g} Ω (Q {math.sqrt(w.L_loop / v['Coss']) / w.R_loop:.3g})"
            res.add_metric("cal_H1", "H1 세계: 전력 loop L (f_obs 공진)", w.L_loop, "H", basis=cal_txt[h.key])
        elif h.key == "H2":
            cal_txt[h.key] = f"L_cab {w.L_cab * 1e6:.4g} µH (C_par {v['Cpar'] * 1e12:g} pF), 추가 감쇠 R_d {w.R_d:.3g} Ω (Q {math.sqrt(w.L_cab / v['Cpar']) / (v['Rm'] + w.R_d):.3g})"
            res.add_metric("cal_H2", "H2 세계: CM 경로 L_cab (f_obs 공진)", w.L_cab, "H", basis=cal_txt[h.key])
        elif h.key == "H3":
            cal_txt[h.key] = f"보조 {v['f_aux'] / 1e3:g} kHz의 {int(round(f0 / v['f_aux']))}차, 누설 L {w.L_aux * 1e6:.4g} µH (C {v['C_aux'] * 1e12:g} pF, Q {v['Q_aux']:g})"
            res.add_metric("cal_H3", "H3 세계: 보조 컨버터 누설 L (f_obs ringing)", w.L_aux, "H", basis=cal_txt[h.key])
        else:
            cal_txt[h.key] = f"f_obs {f0 / 1e6:.4g} MHz 협대역 선 (주변보다 {v['prom_obs']:g} dB)"
            res.add_metric("cal_H4", "H4 세계: 제어 성분 주파수", w.f_ctrl, "Hz", basis=cal_txt[h.key])
    res.add_metric("f_loop0", "기본 전력 loop 공진 1/(2π√(L_loop C_oss))", 1.0 / (2 * math.pi * math.sqrt(v["L_loop"] * v["Coss"])), "Hz",
                   ref=50.329e6 if abs(v["L_loop"] - 10e-9) < 1e-18 and abs(v["Coss"] - 1e-9) < 1e-18 else None, ref_label="교재 E09 10 nH·1 nF → 50.329 MHz", tol=1e-5)
    res.add_metric("icm_ex09", "EX09 조건 C_par·dv/dt (100 pF × 50 kV/µs)", v["Cpar"] * 5e10, "A", ref=5.0 if abs(v["Cpar"] - 100e-12) < 1e-18 else None, ref_label="교재 E09 5 A", tol=1e-12)
    # predictions under each test -----------------------------------------------------
    pred = {}
    for h in hyps:
        w = W[h.key]
        if not w.ok:
            continue
        r0 = read0[h.key]
        out = {}
        out["rg"] = 20 * math.log10(g_read(g_lines(v, w, tr2, fmax), f0, rbw) / r0)
        out["cm"] = 20 * math.log10(max(g_read(g_lines(v, w, tr1, fmax), f0, rbw, drop=("aux",)), 1e-30) / r0)
        lc = g_lines(v, w, tr1, fmax, "cable")
        fpk, apk = g_peak(lc, 0.5 * f0, min(fmax, 1.5 * f0))
        out["cable"] = (20 * math.log10(g_read(lc, f0, rbw) / r0), fpk, 20 * math.log10(apk / r0))
        out["aux_off"] = 20 * math.log10(g_read(g_lines(v, w, tr1, fmax, "aux_off"), f0, rbw) / r0)
        out["ctrl"] = 20 * math.log10(g_read(g_lines(v, w, tr1, fmax, "ctrl"), f0, rbw) / r0)
        out["snub"] = 20 * math.log10(g_read(g_lines(v, w, tr1, fmax, "snub"), f0, rbw) / r0)
        pred[h.key] = out
    cmdm = {"H1": ("CM·DM 모두 (loop ringing이 C_par로도 결합)", "CM+DM"), "H2": ("CM에만", "CM"), "H3": ("DM에만 (보조 입력 전류)", "DM"), "H4": ("CM에만 (제어 접지 결합)", "CM")}
    cl = {}
    for h in hyps:
        if h.key not in pred:
            cl[h.key] = {t: ("재현 불가", "재현 불가") for t, _ in G_TESTS}
            continue
        p = pred[h.key]
        dc, fpk, _ = p["cable"]
        moved = abs(fpk - f0) > 0.05 * f0
        cl[h.key] = {
            "none": (f"{f0 / 1e6:.4g} MHz peak, 기본보다 +{v['prom_obs']:g} dB", "관측 재현"),
            "rg": (f"{p['rg']:+.2f} dB", "약 −2 dB (edge 영향)" if p["rg"] < -1.0 else "≈ 0 dB (edge 무관)"),
            "cmdm": cmdm[h.key],
            "cable": (f"peak {f0 / 1e6:.4g} → {fpk / 1e6:.4g} MHz" if moved else f"f 그대로, {dc:+.2f} dB", "주파수 이동" if moved else ("진폭만" if abs(dc) > 1.0 else "그대로")),
            "aux_off": (f"{p['aux_off']:+.2f} dB", "사라짐" if p["aux_off"] < -3.0 else "그대로"),
            "ctrl": (f"{p['ctrl']:+.2f} dB (선 이동)" if p["ctrl"] < -3.0 else f"{p['ctrl']:+.2f} dB", "선 이동" if p["ctrl"] < -3.0 else "그대로"),
            "snub": (f"{p['snub']:+.2f} dB", "감쇠" if p["snub"] < -3.0 else "그대로"),
        }
    groups = disc_tables(res, hyps, G_TESTS, cl, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설의 세계가 예측하는 f_obs 변화",
                         note=f"모든 세계가 {f0 / 1e6:.4g} MHz에 주변보다 {v['prom_obs']:g} dB 높은 peak를 만들고 R_g를 {v['Rg1']:g} → {v['Rg2']:g} Ω으로 바꿔도 거의 그대로다(관측). "
                         "R_g 시험은 edge가 source인 H1·H2(약 −2 dB)와 무관한 H3·H4(≈ 0 dB)만 가른다. cable 용량은 H2의 주파수를, snubber는 H1을, 보조 끄기는 H3을, 제어 변경은 H4를 움직인다.")
    rows = []
    for h in hyps:
        w = W[h.key]
        rows.append([f"{h.key} {h.label}", cal_txt[h.key], f"{pred[h.key]['rg']:+.2f}" if h.key in pred else "—", cmdm[h.key][1]])
    res.tables.append(Table("t_cal", f"세계별 보정: {f0 / 1e6:.4g} MHz peak가 주변(기본 전력단)보다 {v['prom_obs']:g} dB 높게", ["세계", "보정한 미지수", f"R_g {v['Rg1']:g}→{v['Rg2']:g} Ω [dB]", "CM/DM"], rows,
                            note="값은 같은 관측을 재현하도록 맞춘 합성 경로·결합이다. relative proxy의 dB 차이만 의미가 있다 — 규격 LISN·receiver 읽음이 아니다."))
    fr = [0.5e6, 2e6, 5e6, f0, 25e6, fmax]
    rows = []
    for fq in fr:
        de = 20 * math.log10(abs(sinc(math.pi * fq * tr2)) / abs(sinc(math.pi * fq * tr1)))
        rows.append([f"{fq / 1e6:.4g} MHz", f"{20 * math.log10(tr1 / tr2):+.2f}", f"{de:+.2f}"])
    res.tables.append(Table("t_dvdt", "R_g 변경 효과: C·dv/dt 한 식의 기대 vs trapezoid spectrum의 실제 (주 전력단 source)", ["주파수", "C·dv/dt 기대 [dB]", "trapezoid 실제 [dB]"], rows,
                            note=f"edge 시간은 1/(π t_r) = {1 / (math.pi * tr1) / 1e6:.3g}·{1 / (math.pi * tr2) / 1e6:.3g} MHz 위에서만 spectrum을 바꾼다. 그 아래 peak는 R_g로 거의 움직이지 않는다."))
    # plots ------------------------------------------------------------------------------
    kmax = int(fmax / v["fsw"])
    fk = v["fsw"] * np.arange(1, kmax + 1, 2)
    ts1, vs1 = trapezoid(v["V"], c["T"], v["D"], tr1)
    ts2, vs2 = trapezoid(v["V"], c["T"], v["D"], tr2)
    c1 = np.abs(pwl_fourier(ts1, vs1, c["T"], fk))
    c2 = np.abs(pwl_fourier(ts2, vs2, c["T"], fk))
    m = fk >= 150e3
    res.add_series("d_edge", "trapezoid 고조파 변화 (홀수 차수)", "dB", fk[m].tolist(), (20 * np.log10(c2[m] / c1[m])).tolist())
    res.add_plot("p_edge", f"R_g {v['Rg1']:g} → {v['Rg2']:g} Ω (t_r {tr1 * 1e9:.3g} → {tr2 * 1e9:.3g} ns): spectrum은 1/(πt_r) 위에서만 바뀐다", ["d_edge"], x_label="주파수", x_unit="Hz", y_label="변화", y_unit="dB", kind="xy", log_x=True, level="A",
                 hlines=[{"y": 20 * math.log10(tr1 / tr2), "label": f"C·dv/dt 기대 {20 * math.log10(tr1 / tr2):+.1f} dB"}], vlines=[{"x": f0, "label": "f_obs"}],
                 proved=f"정확 Fourier 계수(FFT·sinc 닫힌 식과 일치)로 edge를 늦추면 {f0 / 1e6:.3g} MHz에서는 {d_edge:+.2f} dB만 바뀌고, C·dv/dt 비({20 * math.log10(tr1 / tr2):+.1f} dB)에 가까워지는 것은 1/(πt_r)보다 훨씬 위라는 것을 보였다.",
                 not_yet="이상 trapezoid edge다. 실제 edge의 곡률·overshoot·dv/dt가 전류에 따라 바뀌는 것(EX09)은 넣지 않았고, 이것은 source 쪽만의 비교다 — 경로와 측정망이 spectrum을 다시 바꾼다.")
    sel = (fk >= 5e6) & (fk <= fmax)
    keys = []
    for h in hyps:
        w = W[h.key]
        if not w.ok:
            continue
        r0 = read0[h.key]
        if h.key in ("H1", "H2"):
            a = g_main(v, w, fk[sel], tr1)
            res.add_series(f"s_{h.key}", f"{h.key} 세계", "dB", fk[sel].tolist(), (20 * np.log10(a / r0)).tolist())
            keys.append(f"s_{h.key}")
    a = g_main(v, W["base"], fk[sel], tr1)
    res.add_series("s_base", "H3·H4 세계의 주 전력단 (공진 없음)", "dB", fk[sel].tolist(), (20 * np.log10(a / read0.get("H3", a_base * 10 ** (v["prom_obs"] / 20)))).tolist(), dash=True)
    keys.append("s_base")
    if W["H3"].ok:
        fa = v["f_aux"] * np.arange(1, int(fmax / v["f_aux"]) + 1, 2)
        sa = (fa >= 5e6) & (fa <= fmax)
        res.add_series("s_aux", "H3 세계: 보조 컨버터 선", "dB", fa[sa].tolist(), (20 * np.log10(g_aux(v, W["H3"], fa[sa]) / read0["H3"])).tolist())
        keys.append("s_aux")
    res.add_series("s_ctrl", "H4 세계: 제어 선", "dB", [f0], [20 * math.log10(W["H4"].s_ctrl / read0["H4"])], style="points")
    keys.append("s_ctrl")
    res.add_plot("p_spec", f"네 세계의 spectrum (relative proxy, 각 세계의 {f0 / 1e6:.3g} MHz 읽음 = 0 dB): 같은 peak, 다른 원인", keys, x_label="주파수", x_unit="Hz", y_label="상대 레벨", y_unit="dB", kind="xy", log_x=True, level="A",
                 vlines=[{"x": f0, "label": "f_obs"}],
                 proved=f"전력 loop ringing(H1), CM 경로 공진(H2), 보조 컨버터(H3), 제어 성분(H4)이 모두 {f0 / 1e6:.3g} MHz에 주변보다 {v['prom_obs']:g} dB 높은 peak를 만든다는 것을 같은 계산(정확 Fourier 계수 × 경로 전달함수)으로 보였다. peak 하나의 읽음으로는 원인을 가를 수 없다.",
                 not_yet="relative proxy다: 규격 LISN·검출기(peak/QP/AV)·RBW 조건이 없고 dB 절대값은 의미가 없다. 각 세계의 경로·결합 값은 관측을 재현하도록 맞춘 합성 값이다. EMI 적합성은 판정하지 않는다.")
    tsel = v["test"]
    if tsel != "none":
        tk = []
        for h in hyps:
            w = W[h.key]
            if not w.ok:
                continue
            r0 = read0[h.key]
            trx = tr2 if tsel == "rg" else tr1
            ln = g_lines(v, w, trx, fmax, tsel)
            drop = ("aux",) if tsel == "cmdm" else ()
            mm = (ln["f"] >= 5e6) & (ln["f"] <= fmax)
            if drop:
                mm &= ~np.isin(ln["src"], list(drop))
            main = mm & (ln["src"] == "main")
            extra = mm & (ln["src"] != "main")
            if h.key in ("H1", "H2"):
                res.add_series(f"t_{h.key}", f"{h.key} 세계 (시험 후)", "dB", ln["f"][main].tolist(), (20 * np.log10(np.maximum(ln["a"][main], 1e-30) / r0)).tolist())
                tk.append(f"t_{h.key}")
            elif np.any(extra):
                lab_x = "보조 컨버터 선" if h.key == "H3" else "제어 선"
                res.add_series(f"t_{h.key}", f"{h.key} 세계: {lab_x} (시험 후)", "dB", ln["f"][extra].tolist(), (20 * np.log10(np.maximum(ln["a"][extra], 1e-30) / r0)).tolist(), style="points")
                tk.append(f"t_{h.key}")
        wb = W["base"]
        lb = g_lines(v, wb, tr2 if tsel == "rg" else tr1, fmax, tsel)
        mb = (lb["f"] >= 5e6) & (lb["f"] <= fmax)
        rb = read0.get("H3", a_base * 10 ** (v["prom_obs"] / 20))
        res.add_series("t_base", "H3·H4 세계의 주 전력단 (시험 후)", "dB", lb["f"][mb].tolist(), (20 * np.log10(lb["a"][mb] / rb)).tolist(), dash=True)
        tk.append("t_base")
        res.add_plot("p_test", "시험: " + dict(G_TESTS)[tsel] + " — 각 세계의 시험 후 spectrum (시험 전 f_obs 읽음 = 0 dB)", tk, x_label="주파수", x_unit="Hz", y_label="상대 레벨", y_unit="dB", kind="xy", log_x=True, level="A",
                     vlines=[{"x": f0, "label": "f_obs"}],
                     proved={
                         "rg": "R_g만 바꾸면 edge가 여기하는 H1·H2 세계의 f_obs peak는 약 2 dB 내려가고 H3·H4는 그대로라는 것을 보였다 — 이 시험은 edge 관련 여부만 가르고 공진 경로는 가르지 못한다.",
                         "cmdm": "CM 출력만 보면 DM인 보조 컨버터 선(H3)은 사라지고 H1·H2·H4의 peak는 남는다는 것을 보였다. H1은 DM 쪽에도 나타나야 한다(표).",
                         "cable": f"cable·접지 용량을 {v['k_cable']:g}배로 하면 H2의 peak는 더 낮은 주파수로 옮겨 가고, H1은 주파수는 그대로 CM 결합만 커지며, H3·H4는 거의 그대로라는 것을 보였다.",
                         "aux_off": "보조 컨버터를 끄면 H3 세계의 peak만 주 전력단 수준으로 내려가고 나머지 세계는 그대로라는 것을 보였다.",
                         "ctrl": f"제어 클록·변조 주파수를 {v['k_ctrl']:g}배로 하면 H4의 선만 다른 주파수로 옮겨 가고 나머지 세계의 f_obs 읽음은 그대로라는 것을 보였다.",
                         "snub": f"loop 감쇠 R을 {v['k_snub']:g}배로 키우면 전력 loop ringing 세계(H1)의 peak만 크게 내려가고 나머지는 그대로라는 것을 보였다.",
                     }[tsel] + " 표의 dB 값은 f_obs의 receiver식 읽음(RBW 안 선의 전력 합)이다.",
                     not_yet="시험 하나가 한 경로만 바꾼다고 가정했다. 실제로 cable을 바꾸면 CM·DM 경로가 함께 바뀌고 snubber는 손실·dv/dt도 바꾼다. probe 위치·LISN·검출기 조건을 고정한 비교만 의미가 있다.")
    # checks -----------------------------------------------------------------------------
    N = 2**16
    tt = np.arange(N) * c["T"] / N
    vv = np.interp(tt, ts1, vs1)
    ff = np.fft.rfft(vv) / N
    kk = np.arange(1, kmax + 1, 2)
    c_fft = np.abs(ff[kk])
    c_ex = np.abs(pwl_fourier(ts1, vs1, c["T"], kk * v["fsw"]))
    err_fft = float(np.max(np.abs(c_fft - c_ex) / c_ex))
    res.add_check(Check(f"독립 경로: 정확 Fourier 계수 vs FFT ({N}점/주기, 홀수 차수 ≤ {kmax})", "PASS" if err_fft < 2e-4 else "FAIL", err_fft, "rel", 2e-4, path="구간별 부분적분 닫힌 식 vs 표본 FFT (aliasing만 차이)", independent=True))
    c_sinc = np.array([v["V"] / (math.pi * k) * abs(sinc(math.pi * k * tr1 / c["T"])) for k in kk]) if abs(v["D"] - 0.5) < 1e-12 else None
    if c_sinc is not None:
        err_s = float(np.max(np.abs(c_sinc - c_ex) / c_ex))
        res.add_check(Check("독립 경로: D = 0.5 trapezoid의 홀수 고조파 (V/πk)|sinc(πk t_r/T)|", "PASS" if err_s < 1e-9 else "FAIL", err_s, "rel", 1e-9, path="sinc 닫힌 식 vs 구간 적분", independent=True))
    rep = max((abs(20 * math.log10(read0[k] / a_base) - v["prom_obs"]) for k in read0 if W[k].ok), default=0.0)
    res.add_check(Check("세계들이 같은 관측(f_obs peak, 주변 + prom dB)을 재현", "PASS" if rep < 1e-6 else "FAIL", rep, "dB", 1e-6, path="brentq·닫힌 식으로 맞춘 경로·결합으로 다시 계산", independent=False,
                        detail="재현 불가: " + (", ".join(h.key for h in hyps if not W[h.key].ok) or "없음")))
    if W["H2"].ok:
        f_num = brentq(lambda fq: float(np.imag(1.0 / y_cm(fq, W["H2"].L_cab, v["Cpar"] * v["k_cable"], v["Rm"] + W["H2"].R_d))), 0.3 * f0, 3 * f0, xtol=1e-6, rtol=1e-14)
        f_cl = 1.0 / (2 * math.pi * math.sqrt(W["H2"].L_cab * v["Cpar"] * v["k_cable"]))
        res.add_check(check_close("cable 시험의 새 공진: Im Z(f) = 0 수치 근 vs 1/(2π√(L C)) 닫힌 식", f_num, f_cl, 1e-9, "경로 임피던스 허수부 근 vs 닫힌 식", True, "Hz"))
    ev = Evidence(res)
    e_i = ev.cite("EX09", "hand_screens", "textbook", "i_pk", note="C_par·dv/dt screen (100 pF × 50 kV/µs) — 이 사례의 R_g 기준 edge는 80 kV/µs")
    e_f = ev.cite("EX09", "hand_screens", "textbook", "f_r", note="10 nH·1 nF loop ringing — 이 사례 기본 loop (H1 세계는 L_loop가 더 크다)")
    if abs(v["Cpar"] - 100e-12) < 1e-18:
        res.add_check(check_close("C·dv/dt (100 pF × 50 kV/µs): 이 사례 vs EX09 실행", v["Cpar"] * 5e10, e_i, 1e-12, "C·dv/dt vs EX09 hand_screens@textbook", False, "A"))
    if abs(v["L_loop"] - 10e-9) < 1e-18 and abs(v["Coss"] - 1e-9) < 1e-18:
        res.add_check(check_close("기본 loop 공진: 이 사례 vs EX09 실행", 1.0 / (2 * math.pi * math.sqrt(v["L_loop"] * v["Coss"])), e_f, 1e-9, "1/(2π√(LC)) vs EX09 hand_screens@textbook", False, "Hz"))
    ev.finish()
    res.circuit = {"diagram": circuit_g(v).to_json(), "intervals": [], "plot_group": ""}
    unresolved(res, f"R_g 시험({groups['rg']}묶음)은 edge 관련 여부만 가른다. CM/DM 분리({groups['cmdm']}), cable 용량({groups['cable']}), 보조 끄기({groups['aux_off']}), 제어 변경({groups['ctrl']}), snubber({groups['snub']})를 조합해야 한다.")
    res.verdict("NOT_EVALUABLE", "EMI 규격 적합성: relative proxy만 계산했다 (규격 LISN·receiver·검출기·limit 없음)")
    res.verdict("MISSING_INPUT", "실제 경로 기생값(C_par, cable·접지 L), 측정 setup(LISN·probe 위치·검출기), 보조 컨버터·제어 사양이 없다")
    res.assumptions += [
        f"switch node: {v['V']:g} V trapezoid, {v['fsw'] / 1e3:g} kHz, D {v['D']:g}; t_r = Q_gd(R_g + R_g,int)/ΔV_g, Q_gd {v['Qgd'] * 1e9:g} nC, R_g,int {v['Rg_int']:g} Ω, ΔV_g {v['dVg']:g} V",
        f"CM proxy = |c_k|·|H_loop|·|Y_CM|·R_m (R_m {v['Rm']:g} Ω), 기본 경로 L_loop {v['L_loop'] * 1e9:g} nH·C_oss {v['Coss'] * 1e9:g} nF, C_par {v['Cpar'] * 1e12:g} pF·L_cab {v['L_cab'] * 1e9:g} nH",
        f"관측 = {f0 / 1e6:.4g} MHz peak가 주변보다 {v['prom_obs']:g} dB, R_g {v['Rg1']:g} → {v['Rg2']:g} Ω에 거의 그대로; receiver식 읽음 = RBW {v['rbw'] / 1e3:g} kHz 안 선의 전력 합",
        f"보조 컨버터 {v['f_aux'] / 1e3:g} kHz·{v['V_aux']:g} V·t_r {v['tr_aux'] * 1e9:g} ns (H3), 제어 성분은 협대역 선 하나 (H4) — 합성",
    ]
    res.not_valid_for += ["EMI 규격 적합성·limit 여유", "방사 emission", "실제 경로 기생값", "고객 회로의 근본원인 확정"]
    res.interpretation = (
        f"R_g를 {v['Rg1']:g} → {v['Rg2']:g} Ω으로 바꾸면 t_r이 {tr1 * 1e9:.3g} → {tr2 * 1e9:.3g} ns가 되어 C·dv/dt로는 {20 * math.log10(tr1 / tr2):+.1f} dB를 기대하지만, trapezoid spectrum은 1/(πt_r) 위에서만 바뀌므로 {f0 / 1e6:.3g} MHz에서는 {d_edge:+.2f} dB뿐이다. "
        "그 peak가 edge로 여기되는 공진(전력 loop H1, CM 경로 H2)이면 R_g로 약 2 dB, 보조 컨버터(H3)나 제어 성분(H4)이면 0 dB 움직인다 — 어느 쪽이든 ‘거의 그대로’라는 관측과 맞는다. "
        "CM/DM 분리, cable·접지 용량 변경(공진 주파수가 옮겨 가는가), snubber(loop ringing인가), 보조 끄기, 제어 설정 변경으로 가르고, probe 위치·LISN·검출기 조건은 고정한다. 1쪽 메모 영어 틀: " + MEMO_EN_TEMPLATE
    )
    return res


def params_g() -> list[Param]:
    g_s, g_p, g_o, g_w = "source (주 전력단)", "경로", "관측·receiver", "세계 가정"
    return [
        Param("V", "switch node 전압", "V", 800.0, "V", vmin=1, vmax=3000, source="ASSUMED", source_note="EX09와 같은 800 V bus", group=g_s),
        Param("fsw", "스위칭 주파수", "Hz", 100e3, "kHz", vmin=1e3, vmax=5e6, source="ASSUMED", group=g_s),
        Param("D", "duty (50 % 기준 폭)", "", 0.5, "", vmin=0.05, vmax=0.95, source="ASSUMED", group=g_s),
        Param("Qgd", "Q_gd", "C", 20e-9, "nC", vmin=1e-10, vmax=1e-6, source="ASSUMED", source_note="CASE A와 같은 합성 소자", group=g_s),
        Param("Rg_int", "내부 R_g", "Ω", 2.5, "Ω", vmin=0.0, vmax=100, source="ASSUMED", group=g_s),
        Param("dVg", "Miller 구간 gate 구동 전압차 V_drv − V_pl", "V", 10.0, "V", vmin=0.1, vmax=40, source="ASSUMED", group=g_s),
        Param("Rg1", "R_g (기준)", "Ω", 2.5, "Ω", vmin=0.0, vmax=1000, source="ASSUMED", source_note="t_r 10 ns", group=g_s),
        Param("Rg2", "R_g (늘림)", "Ω", 10.0, "Ω", vmin=0.0, vmax=1000, source="ASSUMED", source_note="t_r 25 ns", group=g_s),
        Param("L_loop", "전력 loop L (기본)", "H", 10e-9, "nH", vmin=1e-12, vmax=1e-3, source="TEXTBOOK", source_note="E09 10 nH", group=g_p),
        Param("Coss", "switch node C (C_oss)", "F", 1e-9, "nF", vmin=1e-13, vmax=1e-5, source="TEXTBOOK", source_note="E09 1 nF → 50.329 MHz", group=g_p),
        Param("R_loop", "loop 감쇠 R (기본)", "Ω", 1.0, "Ω", vmin=1e-4, vmax=1e3, source="ASSUMED", group=g_p),
        Param("Cpar", "node → chassis C_par", "F", 100e-12, "pF", vmin=1e-14, vmax=1e-6, source="TEXTBOOK", source_note="E09 100 pF", group=g_p),
        Param("L_cab", "CM 경로 L (기본: 공진이 대역 밖)", "H", 100e-9, "nH", vmin=1e-10, vmax=1e-2, source="ASSUMED", group=g_p),
        Param("Rm", "측정 proxy 저항 (CM)", "Ω", 25.0, "Ω", vmin=0.1, vmax=1e4, source="ASSUMED", source_note="50 Ω 두 개 병렬 같은 proxy — 규격 LISN 아님", group=g_p),
        Param("f_obs", "관측 peak 주파수", "Hz", 15.9e6, "MHz", vmin=0.15e6, vmax=30e6, source="ASSUMED", source_note="‘특정 주파수 peak’", group=g_o),
        Param("prom_obs", "관측 peak의 돌출 (주변 대비)", "", 10.0, "", vmin=1.0, vmax=40.0, source="ASSUMED", source_note="dB", group=g_o),
        Param("f_max", "spectrum 상한", "Hz", 30e6, "MHz", vmin=1e6, vmax=100e6, source="ASSUMED", source_note="전도 대역 상한", group=g_o),
        Param("rbw", "receiver RBW", "Hz", 9e3, "kHz", vmin=10, vmax=1e6, source="ASSUMED", source_note="CISPR band B 9 kHz", group=g_o),
        Param("f_aux", "H3: 보조 컨버터 주파수", "Hz", 300e3, "kHz", vmin=1e3, vmax=5e6, source="ASSUMED", group=g_w),
        Param("V_aux", "H3: 보조 컨버터 전압", "V", 400.0, "V", vmin=1, vmax=3000, source="ASSUMED", group=g_w),
        Param("tr_aux", "H3: 보조 컨버터 edge", "s", 20e-9, "ns", vmin=1e-10, vmax=1e-5, source="ASSUMED", group=g_w),
        Param("C_aux", "H3: 보조 switch node C", "F", 100e-12, "pF", vmin=1e-14, vmax=1e-6, source="ASSUMED", group=g_w),
        Param("Q_aux", "H3: 보조 ringing Q", "", 10.0, "", vmin=0.5, vmax=1000, source="ASSUMED", group=g_w),
        Param("k_cable", "시험: cable·접지 용량 배율", "", 1.5, "", vmin=0.1, vmax=10, source="ASSUMED", group="판별 시험"),
        Param("k_snub", "시험: snubber로 loop 감쇠 R 배율", "", 4.0, "", vmin=1.0, vmax=100, source="ASSUMED", group="판별 시험"),
        Param("k_ctrl", "시험: 제어 클록·변조 주파수 배율", "", 1.05, "", vmin=0.5, vmax=2.0, source="ASSUMED", group="판별 시험"),
        _test_param(G_TESTS),
    ]


EXP_G = Experiment(
    key="case_g",
    title="CASE G — R_g를 늘렸는데 EMC peak가 거의 그대로다: C·dv/dt 한 식으로는 spectrum을 예측하지 못한다",
    goal=(
        "증상: R_g를 늘려 edge를 늦췄는데 특정 주파수 peak가 거의 유지된다. 확인된 것: R_g 2.5 → 10 Ω(t_r 10 → 25 ns), 그 peak가 몇 dB밖에 줄지 않았다(여기서는 15.9 MHz, 주변보다 10 dB로 수치화). "
        "모르는 것: 그 성분이 CM인지 DM인지, source가 gate edge인지 cable·접지 LC 공진·보조 컨버터·제어 변조인지, probe 위치·LISN·검출기 조건. 먼저 Q1–Q3에 답을 적고 실행한다. "
        "실행하면 trapezoid의 정확 Fourier 계수로 edge 감속이 그 주파수에서 실제로 몇 dB인지 보이고(C·dv/dt 기대 −8 dB 대 실제 −2 dB), 전력 loop ringing·CM 경로 공진·보조 컨버터·제어 성분 네 세계를 "
        "같은 peak에 맞춘 뒤 CM/DM 분리·cable 용량·보조 끄기·제어 변경·snubber가 각 세계에서 무엇을 보일지 비교한다. 모든 레벨은 relative proxy다."
    ),
    params=params_g(),
    presets=[
        Preset("observed", "관측 그대로 (R_g 기준 spectrum)", {}, "CASE G 증상", ("nominal", "reference")),
        Preset("rg", "R_g만 변경", {"test": "rg"}, "edge 관련(H1·H2) 대 무관(H3·H4)", ("reference",)),
        Preset("cable", "첫 시험: cable·접지 용량 변경", {"test": "cable"}, "H2 주파수 이동", ("variant", "reference")),
        Preset("cmdm", "CM/DM 분리", {"test": "cmdm"}, "H3 DM", ("variant",)),
        Preset("aux_off", "보조 컨버터 끄기", {"test": "aux_off"}, "H3", ("variant",)),
        Preset("ctrl", "제어 설정 변경", {"test": "ctrl"}, "H4", ("variant",)),
        Preset("snub", "RC snubber", {"test": "snub"}, "H1", ("variant",)),
        Preset("low_peak", "관측 peak 1.5 MHz", {"f_obs": 1.5e6}, "edge 영향이 0에 가까운 저주파", ("corner",)),
    ],
    run=run_case_g,
    model_level="A (정확 Fourier 계수 × 경로 전달함수, relative proxy)",
    suggested_change="판별 시험을 ‘cable·접지 용량 변경’으로 바꾼다 (다음에 ‘CM/DM 분리’, ‘RC snubber’).",
    prediction=Prediction(
        "R_g 2.5 → 10 Ω(t_r 10 → 25 ns)이면 15.9 MHz의 edge 성분은 몇 dB 줄어드나?",
        ["약 −8 dB (dv/dt 비)", "약 −2 dB (1/(πt_r) 아래라 sinc가 거의 1)", "약 −20 dB", "모르겠다"],
        "약 −2 dB (1/(πt_r) 아래라 sinc가 거의 1)",
        "trapezoid 고조파는 |sinc(πf t_r)|에 비례한다. 15.9 MHz에서 t_r 10 ns는 0.96, 25 ns는 0.76이라 −2.0 dB다. dv/dt 비(−8 dB)는 f ≫ 1/(πt_r)에서만 맞는다. "
        "peak가 그 아래에 있거나 edge와 무관한 source(보조 컨버터, 제어)라면 R_g로는 거의 움직이지 않는다.",
        ["d_dvdt", "d_edge", "f2_2"],
        handcalc=[{"key": "tr1", "label": "t_r = Q_gd(R_g + R_g,int)/ΔV_g", "unit": "s"}, {"key": "icm1", "label": "C_par·dv/dt", "unit": "A"}],
    ),
    suggested={"test": "cable"},
    student=(
        "스위칭 파형의 spectrum은 낮은 주파수에서는 펄스 폭과 주파수가, 아주 높은 주파수(1/(π·t_r) 위)에서만 edge 속도가 정한다. 그래서 R_g로 edge를 늦춰도 그 아래에 있는 peak는 거의 줄지 않는다. "
        "게다가 peak는 대개 경로의 공진(전력 loop, cable·접지)에서 커지고, 아예 다른 source(보조 컨버터, 제어 클록)일 수도 있다. 그래서 ‘C·dv/dt를 줄이면 EMI가 준다’는 한 식으로 spectrum 전체를 예측하지 않는다."
    ),
    expert=(
        "교재 해설(CASE G): 해당 성분이 CM/DM인지, source가 gate edge인지 cable/LC resonance·aux converter·control modulation인지 분리한다. probe 위치·LISN·검출기 조건을 고정한다. "
        "ringing frequency가 고정되고 amplitude만 변하는지, cable/ground capacitance 변경에 반응하는지 비교한다. C·dv/dt 한 식으로 전체 EMC spectrum을 예측하지 않는다. "
        "추가: ① trapezoid 고조파는 1/(πT_on)과 1/(πt_r) 두 corner를 가진다 — t_r 10 → 25 ns면 두 번째 corner가 31.8 → 12.7 MHz로 내려오고, 15.9 MHz에서는 −2 dB, 그보다 훨씬 위에서만 −8 dB에 가까워진다. "
        "② 공진 peak의 주파수는 경로가 정하고(전력 loop: L_loop·C_oss, CM: L_cab·C_par), edge는 그 주파수에서의 여기 크기만 바꾼다 — R_g를 바꿔 주파수는 그대로 진폭만 조금 바뀌면 edge 여기 공진이다. "
        "③ cable·접지 용량을 바꿨을 때 peak 주파수가 옮겨 가면 CM 경로 공진, snubber에 반응하면 전력 loop ringing, 보조를 끄면 사라지면 보조 컨버터, 제어 설정을 바꾸면 선이 옮겨 가면 제어 성분이다. "
        "④ EX09의 C·dv/dt(100 pF × 50 kV/µs = 5 A)는 edge 동안의 peak 변위전류 screen이지 receiver 읽음이 아니다."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: 15.9 MHz 부근 peak의 원인을 찾아 R_g 증가(스위칭 손실 증가) 없이 줄인다. 확인한 조건: R_g 2.5 → 10 Ω에서 peak가 약 2 dB 이하로만 줄었다. "
        "모르는 조건: CM/DM, source(전력 loop·cable 공진·보조 컨버터·제어), 측정 setup. 유력 가설과 반례: edge가 여기하는 공진이 가장 흔하지만 그 주파수는 1/(πt_r) 아래라 R_g의 효과가 원래 작다 — "
        "R_g가 효과 없다는 것이 edge 무관의 증거는 아니고, 보조 컨버터·제어 성분도 같은 그림을 만든다. 현재 권고: probe 위치·LISN·검출기를 고정하고 CM/DM 분리, cable·접지 용량 변경, 보조 컨버터 끄기, 제어 설정 변경, snubber 시험을 한 번에 하나씩 한다. "
        "부작용: R_g를 더 키우면 스위칭 손실·dead time 여유가 나빠진다. 완료 판정: 원인 경로 하나를 바꿨을 때 peak가 움직이고, 그 대책 후 같은 setup에서 여유가 확보되면 닫는다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the 15.9 MHz peak, which lies below the edge corner frequency, so a slower edge can only change it by about 2 dB. "
        "My current hypothesis is a resonance excited by the switching edge, in the power loop or the common-mode cable and ground path, but an auxiliary converter or a control clock or modulation could produce a similar symptom. "
        "I would separate them by fixing the probe position, LISN and detector, then splitting common and differential mode, changing the cable or ground capacitance, switching the auxiliary converter off, changing the control setting and adding a snubber. "
        "If the peak frequency moves with the cable capacitance, I recommend damping that common-mode path rather than raising the gate resistance, and would recheck the switching loss."
    ),
    questions=[
        q_hyp(
            "G",
            "R_g를 늘려 edge를 늦췄는데 특정 주파수 EMC peak가 거의 그대로다. 가설 3개는?",
            "① 그 주파수가 1/(πt_r) 아래라 edge 영향이 원래 작다(spectrum의 corner) ② edge가 여기하는 공진: 전력 loop ringing 또는 cable·접지 LC(CM 경로) — 주파수는 경로가 정한다 ③ 보조 컨버터 ④ 제어 변조·클록. "
            "CM인지 DM인지부터 나누고, C·dv/dt 한 식으로 spectrum 전체를 예측하지 않는다는 점을 말한다.",
            "The gate resistance was raised but one EMC peak barely moved: give three hypotheses.",
            "The frequency lies below the edge corner 1/(pi t_r), so the edge rate has little effect there; a resonance excited by the edge in the power loop or in the common-mode cable and ground path, whose frequency the path sets; an auxiliary converter; and a control clock or modulation. Separate common and differential mode first, and do not predict the spectrum from C dv/dt alone.",
            ["CM/DM", "corner 1/(πt_r)", "경로 공진", "보조 컨버터", "제어 변조", "C·dv/dt 한계"],
        ),
        q_wave(
            "G",
            "① 같은 setup(probe 위치·LISN·검출기·RBW 고정)의 spectrum을 CM/DM 분리해서 ② switch node 전압과 CM 전류(chassis 귀환)를 고대역으로 동시에 — ringing 주파수와 감쇠 ③ 보조 컨버터·제어 클록의 동작 상태와 주파수 기록(보조 on/off, 제어 설정별). "
            "케이블 배치·접지 위치·온도를 함께 기록한다.",
            "The spectrum split into common and differential mode with a fixed setup (probe position, LISN, detector, bandwidth); the switch-node voltage with the common-mode current on a high-bandwidth timebase, for the ringing frequency and damping; and the state and frequency of the auxiliary converter and the control clock. Record cable routing and ground points.",
            ["CM/DM 분리 spectrum", "switch node + CM 전류", "보조·제어 상태", "setup 고정"],
        ),
        q_first(
            "G",
            "probe 위치·LISN·검출기를 고정하고 cable·접지 용량을 바꾼다: peak 주파수가 옮겨 가면 CM 경로 공진(H2), 주파수는 그대로 진폭만 바뀌면 전력 loop ringing(H1)이나 다른 source다. "
            "이어서 snubber(H1 감쇠), 보조 끄기(H3 사라짐), 제어 설정 변경(H4 선 이동), CM/DM 분리(H3은 DM)로 가른다. R_g만 바꾸는 시험은 edge 관련(약 −2 dB)과 무관(0 dB)만 가른다.",
            "Fix the probe, LISN and detector, then change the cable or ground capacitance: a moving peak frequency means a common-mode path resonance; a fixed frequency with a changed amplitude points to the power-loop ringing or another source. Then a snubber, switching the auxiliary converter off, changing the control setting and a CM/DM split separate the rest.",
            ["setup 고정", "cable 용량 → 주파수 이동", "snubber", "보조 끄기", "제어 변경"],
        ),
        q_memo(
            "G",
            "목표: peak 원인 경로 찾기, R_g 증가 없이. 확인: R_g 2.5 → 10 Ω에서 약 2 dB 이하. 모름: CM/DM, source, setup. 가설과 반례: edge 여기 공진(loop·CM 경로)이 흔하지만 1/(πt_r) 아래라 R_g 효과가 원래 작음 — R_g 무효가 edge 무관의 증거는 아님; 보조·제어도 같은 그림. "
            "권고: setup 고정 후 CM/DM, cable 용량, 보조 끄기, 제어 변경, snubber를 하나씩. 부작용: R_g 증가는 스위칭 손실·dead time 여유 악화. 완료 판정: 원인 경로 변경 시 peak가 움직이고 대책 후 같은 setup에서 여유 확보.",
            MEMO_EN_TEMPLATE.replace("___", "the 15.9 MHz peak below the edge corner", 1),
        ),
    ],
    circuit="fl12_g",
    textbook=[TB_15, TB_E09, TB_17],
    reference_presets=["observed", "rg", "cable"],
    runtime_hint="seconds",
    claim_limit="정확 Fourier 계수와 선형 경로의 relative proxy로 가설별 예측을 비교한다. EMI 규격 적합성과 고객의 근본원인을 판정하지 않는다.",
)


# ======================================================================================
# CASE H - gate off, but energy keeps flowing to the output (bidirectional stage after a fault)
# ======================================================================================

H_TESTS = [
    ("none", "시험 전: 단락 전류 (출력 단자)"),
    ("ibat", "battery 전류 측정"),
    ("vbus", "bus(출력 C) 전압"),
    ("vin", "battery 쪽 C_in 전압·방출 에너지"),
    ("path", "단락 경로 R/L 측정 (LCR)"),
    ("contactor", "battery contactor 열기"),
]
H_MODE = {"DH": "D_H 도통 (i_L → bus)", "DL": "D_L 도통 (i_L < 0)", "OFF": "diode 모두 차단", "CL": "clamp: D_L·D_H 모두 도통"}


class Backfeed(HybridSystem):
    """Bidirectional non-isolated stage after gate-off (body diodes only), output short through R_f + L_f.

    battery V_b (R_bat, contactor k) -> C_in (v_in) -> L (R_L) -> switch node; D_H: node -> bus+, D_L: ground -> node;
    bus: C (v_C) || R_load || fault path R_f + L_f (i_f).  x = [i_L, v_in, v_C, i_f]; q in DH, DL, OFF, CL.
    CL is the clamp: both diodes conduct, the node sits at -V_f and the bus at -2 V_f while the fault inductance freewheels.
    """

    state_names = ("i_L", "v_in", "v_C", "i_f")
    state_units = ("A", "V", "V", "A")

    def __init__(self, p: dict):
        self.p = dict(p)
        self.k = 1.0 if p["contactor"] else 0.0
        self.key = "fl12h|" + "|".join(f"{k}={p[k]!r}" for k in sorted(p))

    def mode(self, q):
        p, k = self.p, self.k
        L, Cin, C, Lf, Vf = p["L"], p["Cin"], p["C"], p["Lf"], p["Vf"]
        A = np.zeros((4, 4))
        b = np.zeros(4)
        A[1, 1] = -k / (p["Rbat"] * Cin)
        b[1] = k * p["Vb"] / (p["Rbat"] * Cin)
        A[1, 0] = -1.0 / Cin
        if q == "DH":
            A[0, 0], A[0, 1], A[0, 2], b[0] = -p["RL"] / L, 1.0 / L, -1.0 / L, -Vf / L
        elif q in ("DL", "CL"):
            A[0, 0], A[0, 1], b[0] = -p["RL"] / L, 1.0 / L, Vf / L
        if q != "CL":
            A[2, 2] = -1.0 / (p["Rload"] * C)
            A[2, 3] = -1.0 / C
            if q == "DH":
                A[2, 0] = 1.0 / C
            A[3, 2], A[3, 3] = 1.0 / Lf, -p["Rf"] / Lf
        else:
            A[3, 3], b[3] = -p["Rf"] / Lf, -2.0 * Vf / Lf
        return AffineMode(f"{self.key}|{q}", A, b, label=q)

    def guards(self, q):
        Vf, Rl = self.p["Vf"], self.p["Rload"]

        def e(*c):
            return np.array(c, dtype=float)

        def pin_vc(z):
            z = z.copy()
            z[2] = -2.0 * Vf
            return z

        def zero_il(z):
            z = z.copy()
            z[0] = 0.0
            return z

        if q == "DH":
            return [Guard("i_L → 0 (D_H 차단)", e(1, 0, 0, 0, 0), -1, lambda qq: "OFF", zero_il), Guard("v_C → −2V_f (clamp)", e(0, 0, 1, 0, 2 * Vf), -1, lambda qq: "CL", pin_vc)]
        if q == "OFF":
            return [Guard("v_in − v_C − V_f > 0 (D_H 도통)", e(0, 1, -1, 0, -Vf), +1, lambda qq: "DH"), Guard("v_C → −2V_f (clamp)", e(0, 0, 1, 0, 2 * Vf), -1, lambda qq: "CL", pin_vc),
                    Guard("v_in < −V_f (D_L 도통)", e(0, -1, 0, 0, -Vf), +1, lambda qq: "DL")]
        if q == "DL":
            return [Guard("i_L → 0 (D_L 차단)", e(1, 0, 0, 0, 0), +1, lambda qq: "OFF", zero_il), Guard("v_C → −2V_f (clamp)", e(0, 0, 1, 0, 2 * Vf), -1, lambda qq: "CL", pin_vc)]
        return [Guard("i_DL → 0 (clamp 해제)", e(-1, 0, 1.0 / Rl, 1, 0), -1, lambda qq: "DH")]

    def outputs(self, q):
        p, k = self.p, self.k
        o = {"iL": np.array([1.0, 0, 0, 0, 0]), "vin": np.array([0, 1.0, 0, 0, 0]), "vC": np.array([0, 0, 1.0, 0, 0]), "if": np.array([0, 0, 0, 1.0, 0]),
             "ibat": np.array([0, -k / p["Rbat"], 0, 0, k * p["Vb"] / p["Rbat"]])}
        return o

    def powers(self, q):
        p, k, Vf = self.p, self.k, self.p["Vf"]
        Pb = np.zeros((5, 5))
        Pb[1, 4] = Pb[4, 1] = -0.5 * k * p["Vb"] / p["Rbat"]
        Pb[4, 4] = k * p["Vb"] ** 2 / p["Rbat"]
        PR = np.zeros((5, 5))
        PR[0, 0] = p["RL"]
        PR[3, 3] = p["Rf"]
        PR[2, 2] = 1.0 / p["Rload"]
        # battery internal resistance k (Vb - vin)^2 / Rbat
        PR[1, 1] += k / p["Rbat"]
        PR[1, 4] += -k * p["Vb"] / p["Rbat"]
        PR[4, 1] += -k * p["Vb"] / p["Rbat"]
        PR[4, 4] += k * p["Vb"] ** 2 / p["Rbat"]
        lin = np.zeros(5)  # diode conduction V_f (i_DH + i_DL), linear in z
        if q == "DH":
            lin[0] = Vf
        elif q == "DL":
            lin[0] = -Vf
        elif q == "CL":
            lin[3], lin[2], lin[0] = 2 * Vf, 2 * Vf / p["Rload"], -Vf
        PR[:, 4] += 0.5 * lin
        PR[4, :] += 0.5 * lin
        return {"p_bat": Pb, "p_R": PR}

    def stored_energy(self):
        p = self.p
        return np.diag([p["L"], p["Cin"], p["C"], p["Lf"], 0.0])

    def describe(self, q):
        return q


def h_params(v: dict, **over) -> dict:
    p = {"Vb": v["Vb"], "Rbat": v["Rbat"], "Cin": v["Cin"], "L": v["L"], "RL": v["RL"], "C": v["C"], "Rload": v["Rload"], "Rf": v["Rf"], "Lf": v["Lf"], "Vf": v["Vf"], "contactor": True}
    p.update(over)
    return p


def h_run(p: dict, iL0: float, vC0: float, t_end: float, vin0: float | None = None):
    s = Backfeed(p)
    x0 = [iL0, p["Vb"] if vin0 is None else vin0, vC0, 0.0]
    q0 = "DH" if iL0 > 0 else "OFF"
    return s, simulate(s, q0, x0, 0.0, t_end)


def h_if_at(tr, t: float) -> float:
    z, _ = tr.state_at(t)
    return float(z[3])


H_WORLD = {
    "H1": {"label": "인덕터·입력 C 저장에너지", "unk": "C_in"},
    "H2": {"label": "battery backfeed (body diode)", "unk": "R_bat (battery 경로 R)"},
    "H3": {"label": "출력 C 방전 (저항성 단락)", "unk": "R_f"},
    "H4": {"label": "clamp 경로 (단락 경로 L의 환류)", "unk": "L_f"},
}


def h_world_params(v: dict, key: str, x: float) -> tuple[dict, float]:
    """(circuit parameters, inductor current at the fault) of world `key` with its unknown x."""
    if key == "H1":
        return h_params(v, contactor=False, Cin=x), v["IL0"]
    if key == "H2":
        return h_params(v, Rbat=x), v["IL0"]
    if key == "H3":
        return h_params(v, contactor=False, Cin=v["Cin_small"], Rf=x), 0.0
    return h_params(v, contactor=False, Cin=v["Cin_small"], Lf=x), 0.0


def h_calibrate(v: dict, key: str) -> tuple[float | None, str]:
    lo, hi = {"H1": (1e-6, 0.1), "H2": (1e-3, 100.0), "H3": (2.0 * math.sqrt(v["Lf"] / v["C"]) * (1 + 1e-9), 100.0), "H4": (1e-6, 0.05)}[key]

    def f(x):
        p, il0 = h_world_params(v, key, x)
        _, tr = h_run(p, il0, v["VC0"], v["t_obs"])
        return h_if_at(tr, v["t_obs"]) - v["I_obs"]

    xs = np.geomspace(lo, hi, 15)
    fs = [f(float(x)) for x in xs]
    for a in range(len(xs) - 1):
        if fs[a] == 0.0:
            return float(xs[a]), ""
        if fs[a] * fs[a + 1] < 0:
            return brentq(f, float(xs[a]), float(xs[a + 1]), xtol=1e-14 * float(xs[a]), rtol=1e-12), ""
    return None, "재현 불가"


def circuit_h(v: dict) -> Circuit:
    c = Circuit("fl12_h", 720, 320, title="gate off 뒤의 경로: battery → L → body diode → bus C ∥ 단락 경로")
    vb = c.add("battery", "VB", 60, 170, 90, "battery", f"{v['Vb']:g} V", lpos=(84, 166, "start"))
    k = c.add("switch", "K", 120, 60, 0, "contactor", "", lpos=(120, 36, "middle"))
    cin = c.add("capacitor", "CIN", 190, 170, 90, "C_in", f"{v['Cin'] * 1e6:g} µF", lpos=(212, 166, "start"))
    ll = c.add("inductor", "L", 290, 60, 0, "L", f"{v['L'] * 1e6:g} µH", lpos=(290, 36, "middle"))
    dh = c.add("diode", "DH", 400, 60, 0, "D_H (Q_H body)", "", lpos=(400, 36, "middle"))
    dl = c.add("diode", "DL", 360, 170, 270, "D_L (Q_L body)", "", lpos=(370, 232, "start"))
    cc = c.add("capacitor", "C", 480, 170, 90, "C (bus)", f"{v['C'] * 1e3:g} mF · {v['VC0']:g} V", lpos=(502, 166, "start"))
    rf = c.add("resistor", "RF", 620, 110, 90, "R_f", f"{v['Rf'] * 1e3:g} mΩ", lpos=(642, 106, "start"))
    lf = c.add("inductor", "LF", 620, 220, 90, "L_f", f"{v['Lf'] * 1e6:g} µH", lpos=(642, 216, "start"))
    c.wire("w_b", vb["a"], (60, 60), k["a"])
    c.wire("w_k", k["b"], (190, 60), cin["a"])
    c.wire("w_l", (190, 60), ll["a"])
    c.wire("w_sn", ll["b"], (360, 60), dh["a"])
    c.wire("w_dl", dl["b"], (360, 60))
    c.wire("w_bus", dh["b"], (480, 60), cc["a"])
    c.wire("w_f", (480, 60), (620, 60), rf["a"])
    c.wire("w_f2", rf["b"], lf["a"])
    c.wire("w_g", lf["b"], (620, 290), (60, 290), vb["b"])
    c.wire("w_gc", cin["b"], (190, 290))
    c.wire("w_gl", dl["a"], (360, 290))
    c.wire("w_gb", cc["b"], (480, 290))
    c.dot((190, 60), (360, 60), (480, 60), (190, 290), (360, 290), (480, 290))
    c.text(676, 168, "단락 경로", "note")
    c.text(330, 312, "gate off: Q_H·Q_L 채널은 꺼져도 body diode는 남는다", "note")
    c.probe("pIF", "if_H2", 560, 48, "right", "i_out (H2)")
    loop_dh = ["VB", "K", "w_b", "w_k", "w_l", "L", "w_sn", "DH", "w_bus", "C", "w_f", "RF", "w_f2", "LF", "w_g", "w_gb"]
    c.mode("DH", H_MODE["DH"], loop_dh + ["CIN", "w_gc"], "인덕터 전류가 D_H로 bus와 단락 경로에 흐른다. battery가 연결돼 있고 bus가 V_b보다 낮으면 battery가 계속 밀어 넣는다(backfeed).")
    c.mode("OFF", H_MODE["OFF"], ["C", "w_f", "RF", "w_f2", "LF", "w_g", "w_gb"], "인덕터 전류 0, diode 차단: 출력 C만 단락 경로로 방전한다 (EX10의 RLC 방전).")
    c.mode("CL", H_MODE["CL"], ["DL", "w_dl", "w_gl", "DH", "w_bus", "w_f", "RF", "w_f2", "LF", "w_g"] + ["L", "w_sn"], "bus 전압이 −2V_f에서 잡히고 단락 경로 L_f의 전류가 D_L → D_H로 환류한다 (clamp 경로).")
    c.mode("DL", H_MODE["DL"], ["DL", "w_dl", "w_gl", "L", "w_sn", "CIN", "w_l", "w_gc"], "인덕터 전류가 음일 때 D_L로 환류한다.")
    return c


def run_case_h(v: dict) -> Result:
    from ._common import energy_ledger, ledger_check

    res = Result("FL12", "case_h", "C (정확 스위칭: body diode guard·clamp 상태, 에너지 원장) + EX10 RLC 재현")
    t_obs, I_obs, t_end = v["t_obs"], v["I_obs"], v["t_end"]
    # EX10 reproduction: bus C into the fault path, no battery, no inductor current, no load
    from ..engine.switched import propagator

    p10 = h_params(v, contactor=False, Cin=v["Cin_small"], Rload=1e12)
    s10, tr10 = h_run(p10, 0.0, v["VC0"], min(t_end, 400e-6), vin0=0.0)
    _, hi = tr10.extrema(0.0, min(t_end, 400e-6), "if")
    t_pk = float("nan")
    for seg in tr10.segments:  # first maximum of the fault current: di/dt changes sign from + to - inside a segment
        cF = s10.outputs(seg.q)["if"] @ seg.mode.F

        def didt(t, seg=seg, cF=cF):
            return float(cF @ (propagator(seg.mode, t - seg.t0) @ seg.z0))

        if didt(seg.t0 + 1e-15) > 0 >= didt(seg.t1):
            t_pk = brentq(didt, seg.t0 + 1e-15, seg.t1, xtol=1e-18, rtol=1e-15)
            z_pk = propagator(seg.mode, t_pk - seg.t0) @ seg.z0
            hi = float(z_pk[3])
            break
    tb = all(abs(v[k_] - r_) <= 1e-12 * abs(r_) for k_, r_ in (("VC0", 800.0), ("C", 1e-3), ("Lf", 10e-6), ("Rf", 0.1)))
    res.add_metric("E0", "출력 C 저장에너지 ½CV₀²", 0.5 * v["C"] * v["VC0"] ** 2, "J", ref=320.0 if tb else None, ref_label="교재 E10 320 J", tol=1e-12)
    res.add_metric("i_pk10", "battery 없이 C → 단락 경로 peak (정확 엔진)", hi, "A", ref=4370.344 if tb else None, ref_label="EX10·지침 4370.344 A", tol=1e-7, basis="diode가 도통하기 전 구간은 순수 RLC")
    res.add_metric("t_pk10", "그 peak 시각 (엔진: di/dt = 0 근)", t_pk, "s", ref=120.919958e-6 if tb else None, ref_label="EX10·지침 120.919958 µs", tol=1e-8)
    # worlds
    cal = {}
    runs = {}
    for hk in H_WORLD:
        x, _ = h_calibrate(v, hk)
        cal[hk] = x
        if x is None:
            continue
        p, il0 = h_world_params(v, hk, x)
        s, tr = h_run(p, il0, v["VC0"], t_end)
        runs[hk] = {"s": s, "tr": tr, "p": p, "il0": il0}
    unit = {"H1": "F", "H2": "Ω", "H3": "Ω", "H4": "H"}
    for hk, meta in H_WORLD.items():
        res.add_metric(f"cal_{hk}", f"{hk} 세계: {meta['unk']} (관측 재현)", cal[hk] if cal[hk] is not None else float("nan"), unit[hk],
                       basis=meta["label"] + ("" if cal[hk] is not None else " — 이 입력에서 재현 불가"))
    # backfeed steady state (closed form) and its time constant
    pb = h_params(v)
    Rsum = (pb["Rbat"] + pb["RL"]) * (1 + pb["Rf"] / pb["Rload"]) + pb["Rf"]
    i_bf = (pb["Vb"] - pb["Vf"]) / Rsum
    tau_bf = (pb["L"] + pb["Lf"]) / (pb["Rbat"] + pb["RL"] + pb["Rf"])
    res.add_metric("i_bf", "battery backfeed 정상 전류 (기본 단락 경로, 닫힌 식)", i_bf, "A", basis=f"(V_b − V_f)/(R_bat + R_L + R_f ...); 시정수 (L + L_f)/R ≈ {tau_bf * 1e3:.3g} ms — gate off로는 멈추지 않는다")
    res.add_metric("tau_bf", "backfeed 시정수 (L + L_f)/(R_bat + R_L + R_f)", tau_bf, "s")
    # ideal short screen: what a 0-ohm, tiny-L model would claim
    Lmin, Rmin = 10e-9, 1e-3
    zeta = Rmin / 2.0 * math.sqrt(v["C"] / Lmin)
    i_ideal = v["VC0"] / math.sqrt(Lmin / v["C"]) * math.exp(-zeta * math.atan(math.sqrt(1 - zeta**2) / zeta) / math.sqrt(1 - zeta**2)) if zeta < 1 else v["VC0"] / Rmin
    res.add_metric("i_ideal", f"‘이상 단락’({Rmin * 1e3:g} mΩ, {Lmin * 1e9:g} nH)으로 넣으면 나오는 peak", i_ideal, "A", basis="물리적으로 불가능한 수준 — 실제 경로의 R/L와 fuse·contactor 동작을 정의해야 한다")
    # energy table up to t_obs
    rows = []
    for hk, r in runs.items():
        tr, p = r["tr"], r["p"]
        z0, _ = tr.state_at(0.0)
        z1, _ = tr.state_at(t_obs)
        dWC = 0.5 * p["C"] * (z0[2] ** 2 - z1[2] ** 2)
        dWin = 0.5 * p["Cin"] * (z0[1] ** 2 - z1[1] ** 2)
        dWL = 0.5 * p["L"] * (z0[0] ** 2 - z1[0] ** 2)
        WLf = 0.5 * p["Lf"] * z1[3] ** 2
        Eb = tr.energy(0.0, t_obs, "p_bat")
        Eloss = tr.energy(0.0, t_obs, "p_R")
        rows.append([f"{hk} {H_WORLD[hk]['label']}", f"{dWC:.4g}", f"{dWin:.4g}", f"{dWL:.4g}", f"{Eb:.4g}", f"{Eloss:.4g}", f"{WLf:.4g}"])
    res.tables.append(Table("t_energy", f"t = 0 → {t_obs * 1e3:g} ms 에너지 출처 [J] (정확 적분)", ["세계", "출력 C 방출 ΔW_C", "C_in 방출", "L 방출", "battery 공급", "손실 (R·diode)", f"{t_obs * 1e3:g} ms의 L_f 저장"], rows,
                            note="각 세계에서 단락 경로로 간 에너지가 어디서 왔는지 보인다: 원장 E_bat + ΔW 방출 = 손실 + L_f 저장. gate off는 이 어느 것도 끊지 않는다."))
    # discrimination cells
    hyps = [Hyp(hk, meta["label"], {"H1": "battery contactor는 열렸고, 입력 C·인덕터의 저장에너지가 D_H로 단락에 흐른다", "H2": "gate만 껐고 battery 경로(contactor·precharge 경로)가 닫혀 있어 body diode(D_H)로 계속 공급한다",
                                    "H3": "단락 경로가 저항성이라 출력 C가 천천히 방전한다", "H4": "단락 경로 L이 커서 C 방전 뒤 clamp(D_L·D_H)로 오래 환류한다"}[hk], meta["unk"]) for hk, meta in H_WORLD.items()]
    cl = {}
    for hk in H_WORLD:
        if hk not in runs:
            cl[hk] = {t: ("재현 불가", "재현 불가") for t, _ in H_TESTS}
            continue
        r = runs[hk]
        tr, p = r["tr"], r["p"]
        z, q = tr.state_at(t_obs)
        ib = (p["Vb"] - z[1]) / p["Rbat"] if p["contactor"] else 0.0
        if p["contactor"]:
            _, tr2 = h_run(dict(p, contactor=False), r["il0"], v["VC0"], t_end)
            i_open = h_if_at(tr2, t_end)
        else:
            i_open = h_if_at(tr, t_end)
        i_keep = h_if_at(tr, t_end)
        dWin = 0.5 * p["Cin"] * (p["Vb"] ** 2 - z[1] ** 2)
        big_r, big_l = p["Rf"] > 2.0 * v["Rf"], p["Lf"] > 2.0 * v["Lf"]
        cl[hk] = {
            "none": (f"i_out({t_obs * 1e3:g} ms) = {z[3]:.4g} A", "관측 재현"),
            "ibat": (f"{ib:.4g} A", "battery 전류 큼" if ib > 0.2 * I_obs else "≈ 0"),
            "vbus": (f"v_C {z[2]:+.4g} V ({H_MODE[q]})", "clamp (−2V_f)" if q == "CL" else ("양의 전압이 남음" if z[2] > 0.05 * v["VC0"] else "0 근처")),
            "vin": (f"v_in {z[1]:.4g} V, C_in 방출 {dWin:.3g} J", "C_in이 에너지를 냄" if dWin > 0.01 * 0.5 * v["C"] * v["VC0"] ** 2 else "무시할 만함"),
            "path": (f"R_f {p['Rf'] * 1e3:.4g} mΩ, L_f {p['Lf'] * 1e6:.4g} µH", "R_f 큼" if big_r else ("L_f 큼" if big_l else "기본 경로")),
            "contactor": (f"{t_end * 1e3:g} ms에 {i_open:.4g} A (열지 않으면 {i_keep:.4g} A)" if p["contactor"] else "이미 열림 — 변화 없음", "줄어듦 (backfeed 제거)" if p["contactor"] and i_open < 0.5 * max(i_keep, 1e-9) else ("이미 열림" if not p["contactor"] else "그대로")),
        }
    groups = disc_tables(res, hyps, H_TESTS, cl, "판별표: 가설(행) × 판별 시험(열) — 칸 = 그 가설의 세계가 t_obs에 보일 값",
                         note=f"모든 세계가 gate off {t_obs * 1e3:g} ms 뒤 출력 전류 {I_obs:g} A를 재현하도록 미지수(C_in, R_bat, R_f, L_f) 하나를 맞췄다. battery 전류와 contactor 시험은 backfeed(H2)를, bus 전압은 clamp(H4)와 저항성 방전(H3)을, "
                         "C_in 방출 에너지는 저장에너지(H1)를, 단락 경로 R/L 측정은 H3·H4를 가른다.")
    # plots ------------------------------------------------------------------------------
    keys = []
    for hk, r in runs.items():
        smp = r["tr"].sample(["if"], per_segment=60)
        res.add_series(f"if_{hk}", f"{hk} {H_WORLD[hk]['label']}", "A", smp["t"], smp["if"])
        keys.append(f"if_{hk}")
    smp10 = tr10.sample(["if"], per_segment=60)
    res.add_series("if_10", "EX10 조건 (battery 없음, 기본 단락 경로)", "A", smp10["t"], smp10["if"], dash=True)
    bands = []
    if "H2" in runs:
        for iv in runs["H2"]["tr"].mode_intervals(0.0, t_end):
            bands.append({"x0": iv["t0"], "x1": iv["t1"], "mode": iv["mode"], "label": H_MODE[iv["mode"]]})
    res.add_plot("p_if", f"gate off 뒤 출력(단락) 전류: 네 세계 모두 {t_obs * 1e3:g} ms에 {I_obs:g} A", keys + ["if_10"], y_label="i_out", y_unit="A", bands=bands, group="h", level="C",
                 hlines=[{"y": I_obs, "label": f"관측 {I_obs:g} A"}], vlines=[{"x": t_obs, "label": f"t_obs {t_obs * 1e3:g} ms"}],
                 proved=f"battery backfeed(H2), 저장에너지(H1), 저항성 단락의 C 방전(H3), clamp 환류(H4)가 모두 gate off {t_obs * 1e3:g} ms 뒤 {I_obs:g} A를 만든다는 것과, 그 뒤 모양(H2는 battery가 계속 공급해 남고, H1은 L–C_in 진동, H3·H4는 감소)이 다르다는 것을 정확 엔진으로 보였다 "
                 f"(에너지 원장 check, EX10 4370.344 A @ 120.92 µs 재현). 띠는 H2 세계의 diode 상태다.",
                 not_yet="이상 diode(V_f 일정)·집중 R/L 모델이다. 실제 단락은 아크·fuse·contactor 동작·소자 파손으로 경로가 바뀌고, 미지수는 관측을 재현하도록 맞춘 합성 값이다.")
    tsel = v["test"]
    test_out = {"ibat": ("ibat", "battery 전류", "A"), "vbus": ("vC", "bus 전압 v_C", "V"), "vin": ("vin", "battery 쪽 v_in", "V")}
    if tsel in test_out:
        name, ylab, yu = test_out[tsel]
        tk = []
        for hk, r in runs.items():
            smp = r["tr"].sample([name], per_segment=40)
            res.add_series(f"t_{hk}", f"{hk}", yu, smp["t"], smp[name])
            tk.append(f"t_{hk}")
        txt = {
            "ibat": "battery 전류는 backfeed 세계(H2)에서만 출력 전류와 같은 크기로 흐르고, contactor가 열린 세계(H1·H3·H4)에서는 0이라는 것을 보였다.",
            "vbus": "bus 전압은 clamp 세계(H4)에서 −2V_f에 잡히고, 저항성 단락(H3)에서는 수백 V가 천천히 줄며, 전류원이 받치는 H1·H2에서는 0 근처라는 것을 보였다.",
            "vin": "battery 쪽 전압은 H1에서 큰 C_in이 L을 거쳐 에너지를 내주며 크게 흔들리고(역전까지), H2에서는 battery 경로 R의 전압 강하만큼 내려가며, H3·H4의 작은 C_in은 에너지가 거의 없다는 것을 보였다.",
        }[tsel]
        res.add_plot("p_test", "시험: " + dict(H_TESTS)[tsel], tk, y_label=ylab, y_unit=yu, group="h", level="C", vlines=[{"x": t_obs, "label": "t_obs"}],
                     proved=txt, not_yet="실제 측정은 큰 di/dt·공통모드 아래에서 하므로 probe 대역·포화를 확인해야 한다. 이 모델에는 fuse·contactor 아크와 소자 온도가 없다.")
    elif tsel == "path":
        fz = np.geomspace(100.0, 1e6, 121)
        tk = []
        for hk, r in runs.items():
            p = r["p"]
            res.add_series(f"t_{hk}", f"{hk}: R_f {p['Rf'] * 1e3:.3g} mΩ, L_f {p['Lf'] * 1e6:.3g} µH", "Ω", fz.tolist(), np.abs(p["Rf"] + 2j * np.pi * fz * p["Lf"]).tolist())
            tk.append(f"t_{hk}")
        res.add_plot("p_test", "시험: 단락 경로 임피던스 |R_f + jωL_f| (LCR 측정)", tk, x_label="주파수", x_unit="Hz", y_label="|Z|", y_unit="Ω", kind="xy", log_x=True, log_y=True, level="A",
                     proved="저항성 단락 세계(H3)는 저주파 |Z|가 높고, clamp 세계(H4)는 인덕턴스 기울기 구간이 낮은 주파수로 내려오며, H1·H2는 기본 경로(0.1 Ω, 10 µH)라는 것을 보였다 — 단락 경로를 재면 H3·H4가 갈린다.",
                     not_yet="사고 뒤의 경로(아크, 녹은 도체)는 사고 중과 다를 수 있다. 측정은 소신호 LCR이고 큰 전류의 비선형(아크 전압)은 넣지 않았다.")
    elif tsel == "contactor":
        tk = []
        for hk, r in runs.items():
            p = r["p"]
            _, tr2 = h_run(dict(p, contactor=False), r["il0"], v["VC0"], t_end)
            smp = tr2.sample(["if"], per_segment=40)
            res.add_series(f"t_{hk}", f"{hk} (contactor 열림)", "A", smp["t"], smp["if"])
            tk.append(f"t_{hk}")
        res.add_plot("p_test", "시험: battery contactor를 단락과 동시에 연다", tk, y_label="i_out", y_unit="A", group="h", level="C", vlines=[{"x": t_obs, "label": "t_obs"}],
                     proved="contactor를 열면 backfeed 세계(H2)의 출력 전류가 남은 저장에너지만큼만 흐르고 사라지며, 이미 열린 세계(H1·H3·H4)는 그대로라는 것을 보였다 — gate off가 아니라 contactor·fuse가 battery 경로를 끊는다.",
                     not_yet="contactor가 수 kA DC를 차단할 수 있는지(아크, 정격), 여는 데 걸리는 시간은 모델에 없다.")
    # checks -----------------------------------------------------------------------------
    if tb:
        alpha = v["Rf"] / (2 * v["Lf"])
        wd = math.sqrt(1.0 / (v["Lf"] * v["C"]) - alpha**2)
        t_cf = math.atan(wd / alpha) / wd
        i_cf = v["VC0"] / (v["Lf"] * wd) * math.exp(-alpha * t_cf) * math.sin(wd * t_cf)
        res.add_check(check_close("EX10 peak: 정확 엔진 vs RLC 닫힌 식", hi, i_cf, 1e-9, "행렬지수·guard 엔진 vs V₀/(Lω_d)·e^{−αt}·sin ω_d t", True, "A"))
        res.add_check(check_close("EX10 peak 시각: 엔진 di/dt = 0 근 vs atan(ω_d/α)/ω_d", t_pk, t_cf, 1e-9, "엔진 출력의 미분 근 vs 닫힌 식", True, "s"))
    hc = "H2" if "H2" in runs else next(iter(runs))
    led = energy_ledger(runs[hc]["tr"], runs[hc]["s"], 0.0, t_end, ["p_bat"], [], ["p_R"], rated_power=v["Vb"] * v["I_obs"])
    res.add_check(ledger_check(led, threshold=1e-6, what=f"{hc} 세계 0–{t_end * 1e3:g} ms (battery·R·diode·C_in·L·C·L_f): "))
    if "H4" in runs:
        r4 = runs["H4"]
        led4 = energy_ledger(r4["tr"], r4["s"], 0.0, t_end, ["p_bat"], [], ["p_R"], rated_power=v["VC0"] * v["I_obs"])
        res.add_check(ledger_check(led4, threshold=1e-6, what=f"H4 세계 (clamp 포함): "))
    # backfeed steady state: exact engine at long time vs closed form
    _, trb = h_run(pb, v["IL0"], v["VC0"], 25.0 * tau_bf)
    res.add_check(check_close("backfeed 정상 전류: 정확 엔진(25τ) vs 닫힌 식", h_if_at(trb, 25.0 * tau_bf), i_bf, 1e-6, "시간 적분 끝값 vs DH 상태 평형 (V_b − V_f)/R_합", True, "A"))
    rep = max((abs(h_if_at(r["tr"], t_obs) - I_obs) / I_obs for r in runs.values()), default=0.0)
    res.add_check(Check("세계들이 같은 관측(t_obs의 출력 전류)을 재현", "PASS" if rep < 1e-6 else "FAIL", rep, "rel", 1e-6, path="scan + brentq로 맞춘 미지수로 다시 시뮬레이션", independent=False,
                        detail="재현 불가: " + (", ".join(k for k in H_WORLD if k not in runs) or "없음")))
    ev = Evidence(res)
    e_E = ev.cite("EX10", "rlc_fault", "textbook", "E0", note="출력 C 저장에너지 — gate off와 무관하게 남는다")
    e_i = ev.cite("EX10", "rlc_fault", "textbook", "i_pk", note="같은 C·단락 경로의 수동 RLC peak")
    e_t = ev.cite("EX10", "rlc_fault", "textbook", "t_pk", note="그 peak 시각")
    if tb:
        res.add_check(check_close("C → 단락 경로 peak: 이 사례 엔진 vs EX10 실행", hi, e_i, 1e-9, "Backfeed 시스템(battery 없음) vs EX10 rlc_fault@textbook", True, "A"))
        res.add_check(check_close("그 peak 시각: 이 사례 vs EX10 실행", t_pk, e_t, 1e-8, "di/dt = 0 근 vs EX10 실행", True, "s"))
        res.add_check(check_close("½CV²: 이 사례 vs EX10 실행", 0.5 * v["C"] * v["VC0"] ** 2, e_E, 1e-12, "½CV² vs EX10 E0", False, "J"))
    ev.finish()
    res.circuit = {"diagram": circuit_h(v).to_json(), "intervals": bands, "plot_group": "h"}
    unresolved(res, f"battery 전류({groups['ibat']}묶음), bus 전압({groups['vbus']}), C_in 전압·에너지({groups['vin']}), 단락 경로 R/L({groups['path']}), contactor 시험({groups['contactor']})이 필요하다.")
    res.verdict("MISSING_INPUT", "실제 단락 경로의 R/L, fuse·contactor 동작(정격·시간), diode·소자의 서지 한계가 없다 — 이상 0 Ω 단락은 불가능한 전류를 만든다")
    if hi > v["I_phys"]:
        res.verdict("OUT_OF_VALIDITY", f"이 단락 경로(R_f {v['Rf'] * 1e3:.3g} mΩ, L_f {v['Lf'] * 1e9:.3g} nH)는 C 방전만으로 {hi / 1e3:.3g} kA를 계산한다 — 도체·소자·fuse가 먼저 반응하는 영역이라 숫자를 쓰지 않는다. 실제 경로 R/L와 보호 동작을 정의한다")
    res.assumptions += [
        f"battery {v['Vb']:g} V (R_bat {v['Rbat'] * 1e3:g} mΩ), C_in {v['Cin'] * 1e6:g} µF, L {v['L'] * 1e6:g} µH (R_L {v['RL'] * 1e3:g} mΩ), bus C {v['C'] * 1e3:g} mF @ {v['VC0']:g} V, 부하 {v['Rload']:g} Ω",
        f"단락 경로 기본값 R_f {v['Rf'] * 1e3:g} mΩ, L_f {v['Lf'] * 1e6:g} µH (EX10과 같은 값), 단락 순간 gate off, 그때 인덕터 전류 {v['IL0']:g} A",
        f"body diode = 이상 diode + V_f {v['Vf']:g} V; clamp 상태에서 bus는 −2V_f",
        f"관측 = gate off {t_obs * 1e3:g} ms 뒤 출력 단자 전류 {I_obs:g} A (ASSUMED 수치화)",
    ]
    res.not_valid_for += ["fuse·contactor 차단 성능", "소자 서지·파손", "아크", "고객 회로의 근본원인 확정"]
    res.interpretation = (
        "gate off는 채널만 끈다: body diode, 인덕터·커패시터의 저장에너지, battery는 그대로 남는다. 단락으로 bus가 battery보다 낮아지면 battery가 L과 D_H를 거쳐 계속 공급하고(backfeed: 기본 단락 경로에서 정상 "
        f"{i_bf:.4g} A, 시정수 {tau_bf * 1e3:.3g} ms), 출력 C의 {0.5 * v['C'] * v['VC0'] ** 2:.4g} J은 단락 경로로 방전하며, 단락 경로 L의 전류는 bus가 −2V_f에 잡힌 뒤 diode로 환류한다. "
        f"네 세계가 모두 {t_obs * 1e3:g} ms 뒤 {I_obs:g} A를 만들지만 battery 전류·bus 전압·인덕터 전류·C_in 전압이 다르다. 단락을 이상 0 Ω으로 넣으면 수십만 A가 나오므로 실제 경로 R/L와 fuse·contactor 동작을 정의한다. 1쪽 메모 영어 틀: "
        + MEMO_EN_TEMPLATE
    )
    return res


def params_h() -> list[Param]:
    g_b, g_c, g_f, g_o, g_w = "battery 쪽", "컨버터·bus", "단락 경로 (EX10)", "관측", "세계 가정"
    return [
        Param("Vb", "battery 전압", "V", 400.0, "V", vmin=1, vmax=2000, source="ASSUMED", group=g_b),
        Param("Rbat", "battery 내부 R", "Ω", 20e-3, "mΩ", vmin=1e-5, vmax=10, source="ASSUMED", group=g_b),
        Param("Cin", "battery 쪽 C_in", "F", 100e-6, "µF", vmin=1e-8, vmax=1.0, source="ASSUMED", group=g_b),
        Param("L", "컨버터 인덕터 L", "H", 200e-6, "µH", vmin=1e-7, vmax=1.0, source="ASSUMED", group=g_c),
        Param("RL", "인덕터 R_L", "Ω", 20e-3, "mΩ", vmin=1e-5, vmax=10, source="ASSUMED", group=g_c),
        Param("IL0", "단락 순간 인덕터 전류", "A", 25.0, "A", vmin=0.0, vmax=5000, source="ASSUMED", source_note="10 kW / 400 V", group=g_c),
        Param("C", "bus(출력) C", "F", 1e-3, "mF", vmin=1e-7, vmax=1.0, source="TEXTBOOK", source_note="E10 1 mF", group=g_c),
        Param("VC0", "bus 초기 전압", "V", 800.0, "V", vmin=1, vmax=3000, source="TEXTBOOK", source_note="E10 800 V → 320 J", group=g_c),
        Param("Rload", "정상 부하 R", "Ω", 64.0, "Ω", vmin=0.1, vmax=1e9, source="ASSUMED", source_note="800 V에서 10 kW", group=g_c),
        Param("Vf", "body diode V_f", "V", 1.5, "V", vmin=0.0, vmax=10, source="ASSUMED", group=g_c),
        Param("Rf", "단락 경로 R_f", "Ω", 0.1, "Ω", vmin=1e-4, vmax=100, source="TEXTBOOK", source_note="E10 0.1 Ω", group=g_f),
        Param("Lf", "단락 경로 L_f", "H", 10e-6, "µH", vmin=1e-9, vmax=0.1, source="TEXTBOOK", source_note="E10 10 µH", group=g_f),
        Param("t_obs", "관측 시각 (gate off 후)", "s", 1e-3, "ms", vmin=1e-5, vmax=0.1, source="ASSUMED", group=g_o),
        Param("I_obs", "관측 출력 전류", "A", 250.0, "A", vmin=1, vmax=1e5, source="ASSUMED", source_note="‘즉시 0이 되지 않음’을 수치화", group=g_o),
        Param("t_end", "표시 구간", "s", 4e-3, "ms", vmin=1e-4, vmax=0.2, source="ASSUMED", group=g_o),
        Param("Cin_small", "H3·H4: battery 쪽 C (작음)", "F", 1e-6, "µF", vmin=1e-9, vmax=1e-3, source="ASSUMED", group=g_w),
        Param("I_phys", "물리적으로 쓸 수 있는 단락 전류 상한 (screen)", "A", 50e3, "kA", vmin=1, vmax=1e7, source="ASSUMED", source_note="이보다 크면 경로 모델이 비현실적", group=g_w),
        _test_param(H_TESTS),
    ]


EXP_H = Experiment(
    key="case_h",
    title="CASE H — gate를 껐는데 출력에 에너지가 계속 간다: gate off는 절연 스위치가 아니다",
    goal=(
        "증상: 양방향 컨버터에서 출력 단락 fault 후 PWM을 껐는데 출력 전류가 즉시 0이 되지 않는다(여기서는 1 ms 뒤에도 250 A로 수치화). 확인된 것: gate는 꺼졌다. "
        "모르는 것: battery contactor 상태, 실제 단락 경로의 R/L, 단락 순간 인덕터 전류, battery 쪽 C, clamp 경로. 먼저 Q1–Q3에 답을 적고 실행한다. "
        "실행하면 battery → C_in → L → body diode → bus C ∥ 단락 경로 회로를 정확 엔진(diode guard, clamp 상태, 에너지 원장)으로 풀어 EX10의 320 J·4370.344 A @ 120.919958 µs를 재현하고, "
        "저장에너지·battery backfeed·출력 C 방전·clamp 환류 네 세계를 같은 관측에 맞춘 뒤 battery 전류·bus 전압·인덕터 전류·C_in 전압·contactor 시험의 예측을 비교한다."
    ),
    params=params_h(),
    presets=[
        Preset("observed", "관측 그대로 (1 ms에 250 A)", {}, "CASE H 증상", ("nominal", "reference")),
        Preset("ibat", "첫 시험: battery 전류", {"test": "ibat"}, "H2 backfeed", ("reference",)),
        Preset("vbus", "bus 전압", {"test": "vbus"}, "H3·H4", ("variant", "reference")),
        Preset("vin", "C_in 전압·에너지", {"test": "vin"}, "H1", ("variant",)),
        Preset("path", "단락 경로 R/L 측정", {"test": "path"}, "H3·H4", ("variant",)),
        Preset("contactor", "contactor 열기", {"test": "contactor"}, "backfeed는 contactor가 끊는다", ("variant",)),
        Preset("ideal_short", "이상 단락 (1 mΩ, 10 nH)", {"Rf": 1e-3, "Lf": 1e-8}, "불가능한 전류 — 실제 경로 R/L 필요", ("failure",)),
    ],
    run=run_case_h,
    model_level="C (정확 스위칭: body diode guard·clamp, 에너지 원장)",
    suggested_change="판별 시험을 ‘battery 전류’로 바꾼다 (다음에 ‘bus 전압’, ‘C_in 전압’).",
    prediction=Prediction(
        "출력 단락 후 gate만 끄고 battery contactor는 닫혀 있다면, 출력 전류는 어떻게 되나?",
        ["gate off 즉시 0", "출력 C가 비워지면 0", "battery가 body diode로 계속 공급해 줄지 않고 커진다", "모르겠다"],
        "battery가 body diode로 계속 공급해 줄지 않고 커진다",
        "bus가 battery 전압보다 낮아지면 L과 Q_H의 body diode가 battery → 단락 경로를 잇는 비제어 정류 경로가 된다. 기본 단락 경로(0.1 Ω, 10 µH)면 정상 전류가 약 2.8 kA, 시정수 1.5 ms다. "
        "gate off는 이 경로를 끊지 못한다 — contactor·fuse가 끊는다.",
        ["i_bf", "tau_bf", "cal_H2"],
        handcalc=[{"key": "E0", "label": "½CV₀² (E10)", "unit": "J"}, {"key": "i_pk10", "label": "C → 단락 RLC peak", "unit": "A"}, {"key": "i_bf", "label": "(V_b − V_f)/R_합", "unit": "A"}],
    ),
    suggested={"test": "ibat"},
    student=(
        "MOSFET의 gate를 끄면 채널만 꺼지고 body diode는 그대로 남는다. 그래서 인덕터에 흐르던 전류는 diode로 계속 흐르고, 출력 커패시터에 남은 에너지(800 V·1 mF면 320 J)는 단락 경로로 방전하며, "
        "단락으로 출력 쪽 전압이 battery보다 낮아지면 battery가 diode를 통해 계속 전류를 밀어 넣는다. 단락 경로 자체의 인덕턴스도 전류를 이어가게 한다. gate off는 회로를 끊는 절연 스위치가 아니다."
    ),
    expert=(
        "교재 해설(CASE H): inductor/transformer 저장에너지, body-diode 경로, battery backfeed, output capacitor와 clamp 경로를 회로로 추적한다. gate off는 galvanic isolation 스위치가 아니다. "
        "출력 short를 이상적인 0 Ω으로만 넣으면 불가능한 전류가 나올 수 있으므로 실제 경로의 R/L와 보호 동작을 정의한다. "
        "추가: ① battery 없이 출력 C만 방전하면 EX10과 같은 수동 RLC다(320 J, 4370.344 A @ 120.919958 µs) — diode가 도통하기 전까지는 회로가 같다. ② bus가 −2V_f에 잡히면 단락 경로 L_f의 전류가 D_L → D_H로 환류한다(clamp). "
        "③ battery가 연결돼 있으면 정상 전류 (V_b − V_f)/R_합, 시정수 (L + L_f)/R_합으로 커진다 — 끊는 것은 contactor·fuse다. ④ 판별: battery 전류(backfeed), bus 전압(clamp −2V_f인지, 수백 V가 남는지), C_in 전압·방출 에너지(저장에너지), 단락 경로 R/L 측정(저항성 단락인지, 큰 L인지). "
        "⑤ 단락 경로를 1 mΩ·10 nH로 넣으면 수십만 A가 계산된다 — 그 숫자로 보호를 설계하지 않는다."
    ),
    customer_ko=(
        "[1쪽 메모] 목표: fault 후 PWM off에도 출력 전류가 남는 원인을 정하고 보호 순서(gate off, contactor, fuse)를 확정한다. 확인한 조건: gate는 꺼졌다. "
        "모르는 조건: contactor 상태와 여는 시점, 실제 단락 경로 R/L, 단락 순간 인덕터 전류, battery 쪽 C, clamp 경로. 유력 가설과 반례: battery가 연결된 채라면 body diode로 backfeed가 이어진다 — 그러나 출력 C 방전(저항성 단락)이나 단락 경로 L의 clamp 환류, "
        "입력 C·인덕터 저장에너지도 같은 시점에 같은 전류를 만든다. 현재 권고: 같은 사건에서 battery 전류, bus 전압, 인덕터 전류, C_in 전압을 동시에 기록하고, 단락 경로의 R/L를 측정한다. "
        "부작용: contactor를 부하 전류에서 여는 보호는 아크·정격 문제를 만든다 — fuse와의 협조가 필요하다. 완료 판정: 각 에너지원이 보호 동작 뒤 정해진 시간 안에 0이 되는 것을 같은 측정으로 확인하면 닫는다."
    ),
    customer_en=(
        "At this operating point, the first constraint appears to be the energy left in the circuit after gate-off: the body diodes, the stored energy in the capacitors and inductors, and the battery are still connected. "
        "My current hypothesis is battery backfeed through the high-side body diode, but the output-capacitor discharge into a resistive short, the clamp path of the short-circuit inductance or the stored energy of the input capacitor and inductor could produce a similar symptom. "
        "I would separate them by recording the battery current, the bus voltage, the inductor current and the input-capacitor voltage during the same event, and by measuring the real short-circuit path. "
        "If the battery current follows the output current, I recommend opening the battery path with a contactor or fuse rather than relying on gate-off, and would recheck the interruption rating."
    ),
    questions=[
        q_hyp(
            "H",
            "양방향 컨버터에서 fault 후 PWM을 껐는데 출력 전류가 즉시 0이 되지 않는다. 가설 3개는?",
            "① 인덕터·변압기·입력 C의 저장에너지가 body diode로 흐름 ② battery backfeed: bus가 battery보다 낮아지면 body diode가 비제어 경로 ③ 출력 커패시터 방전(800 V·1 mF = 320 J) ④ clamp 경로: 단락 경로 L의 전류가 D_L·D_H로 환류. "
            "gate off는 galvanic isolation이 아니고, 이상 0 Ω 단락 모델은 불가능한 전류를 낸다는 점을 말한다.",
            "After a fault the PWM was switched off in a bidirectional converter, but the output current does not drop to zero at once: give three hypotheses.",
            "Stored energy in the inductor, transformer or input capacitor flowing through the body diodes; battery backfeed once the bus falls below the battery; the output-capacitor discharge; and the clamp path in which the short-circuit inductance freewheels through both diodes. Gate-off is not galvanic isolation, and an ideal zero-ohm short gives impossible currents.",
            ["저장에너지", "body diode 경로", "battery backfeed", "출력 C 방전", "clamp 경로", "gate off ≠ 절연"],
        ),
        q_wave(
            "H",
            "① battery 전류와 출력 단자 전류를 같은 timebase로(backfeed 여부) ② bus(출력 C) 전압과 battery 쪽 C_in 전압 ③ 인덕터 전류와 gate·contactor·fuse 동작 시각. 큰 di/dt에 맞는 probe(Rogowski 대역·포화), 단락 경로 위치·길이를 함께 기록한다.",
            "The battery current with the output-terminal current on one timebase; the bus voltage and the battery-side capacitor voltage; and the inductor current with the gate, contactor and fuse timing. Record probe bandwidth and saturation and the location and length of the short-circuit path.",
            ["battery 전류", "bus·C_in 전압", "인덕터 전류", "보호 동작 시각"],
        ),
        q_first(
            "H",
            "같은 사건에서 battery 전류를 본다: 출력 전류와 같이 흐르면 backfeed(H2) — contactor·fuse가 끊어야 한다. 0이면 bus 전압을 본다: −2V_f에 잡혀 있으면 clamp 환류(H4), 수백 V가 천천히 줄면 저항성 단락의 C 방전(H3), "
            "0 근처이고 battery 쪽 C_in 전압이 크게 흔들리며 에너지를 내주면 저장에너지(H1)다. 사고 뒤 단락 경로의 R/L를 재면 H3·H4를 확인한다.",
            "Look at the battery current during the same event: if it follows the output current, it is backfeed, which only a contactor or fuse interrupts. If it is zero, look at the bus voltage: clamped at minus two diode drops means the clamp path, hundreds of volts decaying slowly means a resistive short discharging the capacitor, and a value near zero with a large swing of the battery-side capacitor voltage means stored energy. Measuring the short-circuit path afterwards confirms the resistive or inductive case.",
            ["battery 전류 먼저", "bus 전압", "C_in 전압·에너지", "단락 경로 R/L", "contactor·fuse"],
        ),
        q_memo(
            "H",
            "목표: PWM off 후 남는 출력 전류의 원인과 보호 순서 확정. 확인: gate off. 모름: contactor 상태·시점, 단락 경로 R/L, I_L0, C_in, clamp. 가설과 반례: battery 연결 시 backfeed 유력, 그러나 C 방전·clamp 환류·저장에너지도 같은 전류. "
            "권고: battery 전류·bus 전압·인덕터 전류·C_in 전압 동시 기록, 단락 경로 R/L 측정. 부작용: 부하 중 contactor 개방의 아크·정격 → fuse 협조. 완료 판정: 보호 동작 뒤 정한 시간 안에 모든 에너지원이 0.",
            MEMO_EN_TEMPLATE.replace("___", "the energy left in the circuit after gate-off", 1),
        ),
    ],
    circuit="fl12_h",
    textbook=[TB_15, TB_E10, TB_17],
    reference_presets=["observed", "ibat", "vbus"],
    runtime_hint="seconds",
    claim_limit="이상 diode·집중 R/L의 정확 스위칭 해로 가설별 예측을 비교한다. fuse·contactor 차단 성능과 고객의 근본원인을 확정하지 않는다.",
)


EXPERIMENTS = [EXP_A, EXP_B, EXP_C, EXP_D, EXP_E, EXP_F, EXP_G, EXP_H]

LAB = Lab(
    id="FL12",
    title="FAE 디버깅 — 파형보다 먼저 질문의 질을 높인다",
    title_en="FAE debugging: better questions before more waveforms",
    track="basic",
    order=12,
    path_note="14일 경로 12일차 (15 진단·제품 지도)",
    textbook=[TB_15, TB_17],
    prerequisites=["FL02", "FL05", "FL07", "FL08", "FL09", "EX02", "EX09", "EX10", "EX11"],
    summary="교재 CASE A–H: 증상만 보고 가설·파형·첫 실험을 적고, 같은 관측을 재현하는 가설별 합성 세계와 판별표로 시험의 판별력을 본다. 근본원인은 확정하지 않는다.",
    experiments=EXPERIMENTS,
    minimum_scope="교재 8개 case; 원인이 둘 이상 가능한 파형은 multiple-hypothesis diagnosis; 시뮬레이션 결과로 고객 root cause를 확정하지 않음; 설명형 답은 rubric + 사용자 독립 답변 기록",
    claim_limits=[
        "모든 case의 판정은 UNRESOLVED_ROOT_CAUSE: 시뮬레이션은 가설별 예측을 보일 뿐 근본원인을 확정하지 않는다",
        "가설별 세계의 미지수는 관측을 재현하도록 맞춘 합성 값이다 (고객 회로 값이 아니다)",
        "설명형 답은 자동 채점하지 않는다: rubric(must_include)과 사용자가 저장한 답만",
    ],
    test_paths=["tests/test_fl12.py"],
)
