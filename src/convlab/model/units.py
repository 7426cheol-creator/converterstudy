"""SI units: parsing user input with prefixes, formatting for display.

Inputs are stored in SI base units.  A field shows its preferred display unit
(for example uH for an inductance); the user may type a bare number (interpreted in
the display unit) or a number with an explicit unit.  A unit whose dimension does
not match the field is an error, never silently converted or clamped.
"""

from __future__ import annotations

import math
import re

__all__ = ["UnitError", "parse_quantity", "prefix_scale", "fmt_si", "base_unit"]

_PREFIX = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "μ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

# base units the app uses; compound units are listed explicitly
_BASE = [
    "V/us",
    "kV/us",
    "A/us",
    "kA/us",
    "V/A",
    "V/(A·s)",
    "V·s",
    "A/mm²",
    "mm²",
    "K/W",
    "1/K",
    "W/K",
    "J/K",
    "rpm",
    "rad",
    "deg",
    "°C",
    "Ohm",
    "Ω",
    "Wb",
    "Hz",
    "Nm",
    "H",
    "F",
    "V",
    "A",
    "W",
    "J",
    "C",
    "s",
    "T",
    "K",
    "%",
    "",
]

_ALIASES = {"ohm": "Ω", "Ohm": "Ω", "ohms": "Ω", "degC": "°C", "sec": "s"}


class UnitError(ValueError):
    pass


def base_unit(unit: str) -> str:
    return _ALIASES.get(unit, unit)


def prefix_scale(display_unit: str, base: str) -> float:
    """Scale factor from display unit to base unit, e.g. ('uH', 'H') -> 1e-6."""
    base = base_unit(base)
    display_unit = base_unit(display_unit)
    if display_unit == base:
        return 1.0
    if base and display_unit.endswith(base):
        p = display_unit[: -len(base)]
        if p in _PREFIX:
            return _PREFIX[p]
    # special compound displays
    special = {("kV/us", "V/us"): 1e3, ("kA/us", "A/us"): 1e3, ("mΩ", "Ω"): 1e-3, ("kW", "W"): 1e3}
    if (display_unit, base) in special:
        return special[(display_unit, base)]
    raise UnitError(f"표시 단위 '{display_unit}'를 기본 단위 '{base}'로 환산할 수 없습니다")


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_quantity(text: str, base: str, display_unit: str | None = None) -> float:
    """Parse '100', '100u', '100 µH', '0.1 mH', '1e-4 H' into SI base units.

    Bare numbers are read in ``display_unit`` (or base).  Raises UnitError with a
    Korean message on malformed input or dimension mismatch.
    """
    base = base_unit(base)
    s = str(text).strip().replace(",", "")
    if not s:
        raise UnitError("값이 비어 있습니다")
    m = re.fullmatch(rf"({_NUM})\s*(.*)", s)
    if not m:
        raise UnitError(f"숫자로 읽을 수 없습니다: '{text}'")
    val = float(m.group(1))
    suffix = base_unit(m.group(2).strip())
    if not math.isfinite(val):
        raise UnitError("유한한 숫자가 아닙니다")
    if suffix == "":
        scale = prefix_scale(display_unit, base) if display_unit else 1.0
        return val * scale
    # explicit unit: prefix + base
    if suffix == base:
        return val
    if base and suffix.endswith(base) and suffix[: -len(base)] in _PREFIX:
        return val * _PREFIX[suffix[: -len(base)]]
    if suffix in _PREFIX and base not in ("", "%"):
        # bare prefix like '100u' means micro-base
        return val * _PREFIX[suffix]
    try:
        return val * prefix_scale(suffix, base)
    except UnitError:
        pass
    raise UnitError(f"단위 불일치: 이 입력의 단위는 {base or '무차원'}인데 '{suffix}'가 입력되었습니다")


_ENG = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]
_NO_PREFIX = {"%", "", "rad", "deg", "°C", "K", "1/K", "K/W", "rpm", "Nm", "mm²", "A/mm²"}


def fmt_si(value: float, unit: str, digits: int = 4) -> str:
    """Engineering formatting with an SI prefix, e.g. 1.0e-4 H -> '100 µH'."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f"— {unit}".strip()
    if unit in _NO_PREFIX or value == 0:
        return f"{value:.{digits}g} {unit}".strip()
    a = abs(value)
    for scale, p in _ENG:
        if a >= scale * 0.9999999:
            return f"{value / scale:.{digits}g} {p}{unit}"
    return f"{value:.{digits}g} {unit}"
