"""Parameter schema with units, provenance and validity.

Every input carries its source status:
  TEXTBOOK   value stated by textbook v4.0 / instructions v4.0 (a synthetic learning spec)
  ASSUMED    chosen by this lab where the textbook gives no value (flagged in the UI)
  DERIVED    computed from other inputs (e.g. referred values)
  DATASHEET  needs exact part/revision/conditions/URL; none are used in this package
  MISSING_INPUT  a real value is required for a precise claim and is not available

Validity limits are hard: a value outside [vmin, vmax] is rejected with the reason,
never clamped.  ``soft`` limits only warn (the model still applies but the case is
unusual or outside the range the lab was checked for).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .units import UnitError, fmt_si, parse_quantity, prefix_scale

__all__ = ["Param", "Preset", "ParamError", "resolve_params"]

SOURCES = ("TEXTBOOK", "ASSUMED", "DERIVED", "DATASHEET", "MISSING_INPUT")


class ParamError(ValueError):
    """A rejected input; ``errors`` maps parameter key -> Korean reason."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass
class Param:
    key: str
    label: str
    unit: str
    default: Any
    display_unit: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    soft_min: float | None = None
    soft_max: float | None = None
    source: str = "ASSUMED"
    source_note: str = ""
    description: str = ""
    kind: str = "float"  # float | int | choice | bool
    choices: list[tuple[Any, str]] | None = None
    group: str = "회로"
    validity_note: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"param {self.key}: bad source {self.source}")
        if self.display_unit is None:
            self.display_unit = self.unit
        if self.choices is not None:
            # accept plain values as (value, value); a bare string would otherwise be split into characters
            self.choices = [c if isinstance(c, (tuple, list)) else (c, str(c)) for c in self.choices]
        if self.kind == "choice" and self.default not in [c[0] for c in (self.choices or [])]:
            raise ValueError(f"param {self.key}: default {self.default!r} is not one of its choices")

    @property
    def scale(self) -> float:
        if self.kind != "float":
            return 1.0
        return prefix_scale(self.display_unit or self.unit, self.unit)

    def coerce(self, raw: Any) -> Any:
        """Convert a raw UI/CLI value to the SI value, raising UnitError/ValueError."""
        if self.kind == "choice":
            allowed = [c[0] for c in (self.choices or [])]
            if raw not in allowed:
                raise UnitError(f"허용된 선택지가 아닙니다: {raw!r} (가능: {allowed})")
            return raw
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            if str(raw).lower() in ("1", "true", "yes", "on"):
                return True
            if str(raw).lower() in ("0", "false", "no", "off"):
                return False
            raise UnitError(f"참/거짓 값이 아닙니다: {raw!r}")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            val = float(raw)
        else:
            val = parse_quantity(str(raw), self.unit, self.display_unit)
        if self.kind == "int":
            if abs(val - round(val)) > 1e-9:
                raise UnitError("정수만 허용됩니다")
            val = int(round(val))
        return val

    def check(self, val: Any) -> tuple[str | None, str | None]:
        """(error, warning) for an SI value."""
        if self.kind in ("choice", "bool"):
            return None, None
        if self.vmin is not None and val < self.vmin:
            return (
                f"{self.label} = {fmt_si(val, self.unit)} 은(는) 모델 적용범위 하한 {fmt_si(self.vmin, self.unit)} 미만입니다."
                + (f" ({self.validity_note})" if self.validity_note else ""),
                None,
            )
        if self.vmax is not None and val > self.vmax:
            return (
                f"{self.label} = {fmt_si(val, self.unit)} 은(는) 모델 적용범위 상한 {fmt_si(self.vmax, self.unit)} 초과입니다."
                + (f" ({self.validity_note})" if self.validity_note else ""),
                None,
            )
        warn = None
        if self.soft_min is not None and val < self.soft_min:
            warn = f"{self.label}: 검토된 범위({fmt_si(self.soft_min, self.unit)} 이상) 밖입니다. 결과 해석에 주의."
        if self.soft_max is not None and val > self.soft_max:
            warn = f"{self.label}: 검토된 범위({fmt_si(self.soft_max, self.unit)} 이하) 밖입니다. 결과 해석에 주의."
        return None, warn

    def to_json(self) -> dict:
        d = asdict(self)
        d["scale"] = self.scale
        if self.choices is not None:
            d["choices"] = [{"value": c[0], "label": c[1]} for c in self.choices]
        return d


@dataclass
class Preset:
    key: str
    label: str
    values: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()  # nominal | corner | failure | reference | variant

    def to_json(self) -> dict:
        return {"key": self.key, "label": self.label, "values": self.values, "description": self.description, "tags": list(self.tags)}


def resolve_params(params: list[Param], presets: list[Preset], preset_key: str | None, overrides: dict | None):
    """Defaults <- preset <- user overrides; returns (values, warnings, changed_keys).

    Raises ParamError listing every rejected input (unknown key, unit mismatch,
    out-of-validity) so the UI can show all reasons at once.
    """
    by_key = {p.key: p for p in params}
    values = {p.key: p.default for p in params}
    if preset_key:
        pr = next((p for p in presets if p.key == preset_key), None)
        if pr is None:
            raise ParamError({"preset": f"알 수 없는 preset: {preset_key}"})
        for k, v in pr.values.items():
            if k not in by_key:
                raise ParamError({k: f"preset {preset_key}에 정의되지 않은 파라미터"})
            values[k] = v
    changed: list[str] = []
    errors: dict[str, str] = {}
    for k, raw in (overrides or {}).items():
        if k not in by_key:
            errors[k] = "알 수 없는 파라미터입니다"
            continue
        try:
            v = by_key[k].coerce(raw)
        except (UnitError, ValueError) as exc:
            errors[k] = str(exc)
            continue
        if v != values[k]:
            changed.append(k)
        values[k] = v
    warnings: list[str] = []
    for k, p in by_key.items():
        if k in errors:
            continue
        err, warn = p.check(values[k])
        if err:
            errors[k] = err
        if warn:
            warnings.append(warn)
    if errors:
        raise ParamError(errors)
    return values, warnings, changed
