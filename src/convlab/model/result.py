"""Result structure shared by every lab (serialised to JSON for the UI and exports)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .status import status_info, worst

__all__ = ["Metric", "Series", "Plot", "Check", "Table", "Result", "rel_err", "check_close"]


def _num(x):
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def rel_err(value: float, ref: float, abs_scale: float | None = None) -> float:
    """Relative error; with ``abs_scale`` the denominator is max(|ref|, abs_scale) (rated-scale guard)."""
    den = abs(ref)
    if abs_scale is not None:
        den = max(den, abs(abs_scale))
    if den == 0:
        return abs(value - ref)
    return abs(value - ref) / den


@dataclass
class Metric:
    key: str
    label: str
    value: Any
    unit: str = ""
    ref: float | None = None
    ref_label: str = ""  # where the reference comes from (hand calculation / textbook / contract)
    tol: float | None = None  # relative tolerance for the check
    check: str | None = None  # PASS | FAIL | INFO (filled automatically when ref and tol are given)
    basis: str = ""  # e.g. "per module", "total", "phase peak", "RMS", "DC average", "window: 1 period"
    note: str = ""
    abs_scale: float | None = None

    def __post_init__(self) -> None:
        if self.check is None and self.ref is not None and self.tol is not None and isinstance(self.value, (int, float)):
            self.check = "PASS" if rel_err(float(self.value), float(self.ref), self.abs_scale) <= self.tol else "FAIL"

    def to_json(self) -> dict:
        err = None
        if self.ref is not None and isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            err = rel_err(float(self.value), float(self.ref), self.abs_scale)
        return {
            "key": self.key,
            "label": self.label,
            "value": _num(self.value) if not isinstance(self.value, str) else self.value,
            "unit": self.unit,
            "ref": _num(self.ref),
            "ref_label": self.ref_label,
            "tol": self.tol,
            "rel_err": _num(err),
            "check": self.check,
            "basis": self.basis,
            "note": self.note,
        }


@dataclass
class Series:
    key: str
    label: str
    unit: str
    x: list[float]
    y: list[float]
    style: str = "line"  # line | step | points | area
    color: str | None = None
    dash: bool = False

    def to_json(self) -> dict:
        x = [(_num(v)) for v in self.x]
        y = [(_num(v)) for v in self.y]
        return {"key": self.key, "label": self.label, "unit": self.unit, "x": x, "y": y, "style": self.style, "color": self.color, "dash": self.dash}


@dataclass
class Plot:
    key: str
    title: str
    series: list[str]
    x_label: str = "t"
    x_unit: str = "s"
    y_label: str = ""
    y_unit: str = ""
    kind: str = "time"  # time | xy | bode | bar | map
    log_x: bool = False
    log_y: bool = False
    bands: list[dict] = field(default_factory=list)  # [{x0, x1, label, mode}] state intervals linked to the circuit
    markers: list[dict] = field(default_factory=list)  # [{x, y, label}]
    hlines: list[dict] = field(default_factory=list)  # [{y, label}]
    vlines: list[dict] = field(default_factory=list)  # [{x, label}]
    window: tuple[float, float] | None = None  # analysis window used for the metrics
    proved: str = ""  # 한국어: 무엇을 입증했는가
    not_yet: str = ""  # 한국어: 무엇은 아직인가
    level: str = ""  # model level of the curves in this plot
    group: str = ""  # plots in the same group share the x cursor

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "series": self.series,
            "x_label": self.x_label,
            "x_unit": self.x_unit,
            "y_label": self.y_label,
            "y_unit": self.y_unit,
            "kind": self.kind,
            "log_x": self.log_x,
            "log_y": self.log_y,
            "bands": self.bands,
            "markers": self.markers,
            "hlines": self.hlines,
            "vlines": self.vlines,
            "window": list(self.window) if self.window else None,
            "proved": self.proved,
            "not_yet": self.not_yet,
            "level": self.level,
            "group": self.group,
        }


@dataclass
class Check:
    """A verification check (numerical, conservation, independence, convergence)."""

    name: str
    status: str  # PASS | FAIL | INFO | NOT_RUN
    value: Any = None
    unit: str = ""
    threshold: Any = None
    path: str = ""  # what was compared with what (e.g. "switched simulation vs closed form")
    independent: bool = False  # True when the two sides share no formula
    detail: str = ""

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "value": _num(self.value) if not isinstance(self.value, str) else self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "path": self.path,
            "independent": self.independent,
            "detail": self.detail,
        }


@dataclass
class Table:
    key: str
    title: str
    columns: list[str]
    rows: list[list[Any]]
    note: str = ""

    def to_json(self) -> dict:
        return {"key": self.key, "title": self.title, "columns": self.columns, "rows": [[_num(c) if not isinstance(c, str) else c for c in r] for r in self.rows], "note": self.note}


def check_close(name: str, value: float, ref: float, tol: float, path: str, independent: bool, unit: str = "", abs_scale=None, detail="") -> Check:
    e = rel_err(value, ref, abs_scale)
    return Check(
        name=name,
        status="PASS" if e <= tol else "FAIL",
        value=e,
        unit="rel",
        threshold=tol,
        path=path,
        independent=independent,
        detail=detail or f"값 {value:.9g} {unit} / 기준 {ref:.9g} {unit}",
    )


@dataclass
class Result:
    lab: str
    experiment: str
    model_level: str  # A | B | C | D or combination such as "A+C"
    verdicts: list[tuple[str, str]] = field(default_factory=list)  # (status code, why)
    metrics: list[Metric] = field(default_factory=list)
    series: list[Series] = field(default_factory=list)
    plots: list[Plot] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    not_valid_for: list[str] = field(default_factory=list)
    interpretation: str = ""  # 한국어 해석(왜 그런가)
    circuit: dict | None = None  # {"id": ..., "states": [...], "annotations": {...}}
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    # convenience -------------------------------------------------------------------
    def add_metric(self, *a, **k) -> Metric:
        m = Metric(*a, **k)
        self.metrics.append(m)
        return m

    def add_series(self, *a, **k) -> Series:
        s = Series(*a, **k)
        self.series.append(s)
        return s

    def add_plot(self, *a, **k) -> Plot:
        p = Plot(*a, **k)
        self.plots.append(p)
        return p

    def add_check(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    def verdict(self, code: str, why: str) -> None:
        status_info(code)  # validate
        self.verdicts.append((code, why))

    @property
    def status(self) -> str:
        return worst([c for c, _ in self.verdicts])

    @property
    def checks_ok(self) -> bool:
        return all(c.status in ("PASS", "INFO", "NOT_RUN") for c in self.checks) and all(m.check != "FAIL" for m in self.metrics)

    def to_json(self) -> dict:
        st = status_info(self.status)
        return {
            "lab": self.lab,
            "experiment": self.experiment,
            "model_level": self.model_level,
            "status": {"code": st.code, "ko": st.ko, "tone": st.tone, "meaning": st.meaning},
            "verdicts": [{"code": c, "ko": status_info(c).ko, "tone": status_info(c).tone, "why": w} for c, w in self.verdicts],
            "metrics": [m.to_json() for m in self.metrics],
            "series": [s.to_json() for s in self.series],
            "plots": [p.to_json() for p in self.plots],
            "checks": [c.to_json() for c in self.checks],
            "checks_ok": self.checks_ok,
            "tables": [t.to_json() for t in self.tables],
            "assumptions": self.assumptions,
            "not_valid_for": self.not_valid_for,
            "interpretation": self.interpretation,
            "circuit": self.circuit,
            "warnings": self.warnings,
            "extra": self.extra,
        }
