"""Lab registry: every module named flNN_* or exNN_* in this package that defines LAB.

Discovery keeps each lab self-contained (physics, content, circuit, presets in its
own module), so labs can be added without editing a central list.
"""

from __future__ import annotations

import importlib
import pkgutil
import re

from ..model.labspec import Lab

__all__ = ["all_labs", "get_lab", "OFFICIAL_IDS"]

OFFICIAL_IDS = [f"FL{k:02d}" for k in range(1, 13)] + [f"EX{k:02d}" for k in range(1, 13)]
_LABS: dict[str, Lab] | None = None


def _discover() -> dict[str, Lab]:
    labs: dict[str, Lab] = {}
    for mod in pkgutil.iter_modules(__path__):
        if not re.fullmatch(r"(fl|ex)\d\d_\w+", mod.name):
            continue
        m = importlib.import_module(f"{__name__}.{mod.name}")
        lab = getattr(m, "LAB", None)
        if lab is None:
            continue
        if lab.id not in OFFICIAL_IDS:
            raise RuntimeError(f"{mod.name}: {lab.id} is not an official lab id (FL01..FL12, EX01..EX12)")
        if lab.id in labs:
            raise RuntimeError(f"duplicate lab id {lab.id}")
        lab.code_path = lab.code_path or f"src/convlab/labs/{mod.name}.py"
        labs[lab.id] = lab
    return labs


def all_labs() -> dict[str, Lab]:
    global _LABS
    if _LABS is None:
        _LABS = _discover()
    return _LABS


def get_lab(lab_id: str) -> Lab:
    labs = all_labs()
    if lab_id not in labs:
        raise KeyError(f"unknown or not yet implemented lab {lab_id}")
    return labs[lab_id]
