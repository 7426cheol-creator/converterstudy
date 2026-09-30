"""Declarative circuit diagrams (drawn as SVG by web/circuit.js).

Two-terminal elements are placed by centre and axis angle ``rot`` (0: a->b left to
right, 90: top to bottom, 180, 270).  Terminal ``a`` is the reference-positive end:
anode of a diode, drain of a MOSFET, + of a source/capacitor, dot end of an inductor.
Each circuit also lists its discrete modes: which elements and wires carry current
and a one-line explanation, so the waveform cursor can light up the conducting path.
Probes attach a series (e.g. i_L) to an arrow on the diagram; the arrow flips when
the value at the cursor is negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Circuit"]

HALF = 30  # half length of a two-terminal element in px


@dataclass
class Circuit:
    id: str
    width: int = 640
    height: int = 320
    title: str = ""
    elements: list[dict] = field(default_factory=list)
    wires: list[dict] = field(default_factory=list)
    dots: list[list[float]] = field(default_factory=list)
    labels: list[dict] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    modes: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def add(self, kind: str, eid: str, x: float, y: float, rot: int = 0, label: str = "", value: str = "", **extra) -> dict:
        """Place an element; returns {'a': (x, y), 'b': (x, y), ...} terminal coordinates."""
        el = {"id": eid, "type": kind, "x": x, "y": y, "rot": rot, "label": label, "value": value, **extra}
        self.elements.append(el)
        return self.terminals(el)

    @staticmethod
    def terminals(el: dict) -> dict:
        x, y, rot = el["x"], el["y"], el.get("rot", 0)
        if el["type"] == "transformer":
            dx = 22
            return {
                "p1": (x - dx, y - HALF),
                "p2": (x - dx, y + HALF),
                "s1": (x + dx, y - HALF),
                "s2": (x + dx, y + HALF),
            }
        if el["type"] in ("ground", "text", "block"):
            return {"a": (x, y), "b": (x, y)}
        u = (math.cos(math.radians(rot)), math.sin(math.radians(rot)))
        a = (round(x - HALF * u[0], 3), round(y - HALF * u[1], 3))
        b = (round(x + HALF * u[0], 3), round(y + HALF * u[1], 3))
        t = {"a": a, "b": b}
        if el["type"] == "nmos":
            # gate on the perpendicular (-u_y, u_x): left side for a drain-top vertical device
            t["g"] = (round(x - 26 * u[1], 3), round(y + 26 * u[0], 3))
        return t

    def wire(self, wid: str, *pts) -> None:
        self.wires.append({"id": wid, "pts": [[float(p[0]), float(p[1])] for p in pts]})

    def dot(self, *pts) -> None:
        for p in pts:
            self.dots.append([float(p[0]), float(p[1])])

    def text(self, x: float, y: float, s: str, cls: str = "") -> None:
        self.labels.append({"x": x, "y": y, "text": s, "cls": cls})

    def probe(self, pid: str, series: str, x: float, y: float, direction: str, label: str) -> None:
        """Current arrow: ``direction`` is the positive reference (right/left/up/down)."""
        self.probes.append({"id": pid, "series": series, "x": x, "y": y, "dir": direction, "label": label})

    def mode(self, key: str, label: str, active: list[str], text: str = "", dim: list[str] | None = None) -> None:
        self.modes[key] = {"label": label, "active": active, "text": text, "dim": dim or []}

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "title": self.title,
            "elements": self.elements,
            "wires": self.wires,
            "dots": self.dots,
            "labels": self.labels,
            "probes": self.probes,
            "modes": self.modes,
            "notes": self.notes,
        }
