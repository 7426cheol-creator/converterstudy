"""Run an experiment with validated inputs and package the response."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
from collections import OrderedDict

import numpy as np
import scipy

from . import CONTRACT_VERSION, __version__
from .engine.switched import SimulationError
from .labs import get_lab
from .model.params import ParamError, resolve_params
from .model.units import fmt_si

__all__ = ["run_request", "RunError"]


class RunError(Exception):
    def __init__(self, status: int, payload: dict):
        super().__init__(payload.get("message", ""))
        self.status = status
        self.payload = payload


_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 64


def environment() -> dict:
    return {
        "app_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def run_request(lab_id: str, exp_key: str, preset: str | None, values: dict | None, use_cache: bool = True) -> dict:
    lab = get_lab(lab_id)
    exp = lab.experiment(exp_key)
    try:
        resolved, warnings, changed = resolve_params(exp.params, exp.presets, preset, values or {})
    except ParamError as exc:
        raise RunError(400, {"message": "입력값을 확인하세요 (값을 자동으로 잘라내지 않았습니다)", "errors": exc.errors}) from exc
    key = hashlib.sha256(json.dumps([lab_id, exp_key, resolved], sort_keys=True, default=str).encode()).hexdigest()
    if use_cache and key in _CACHE:
        _CACHE.move_to_end(key)
        out = dict(_CACHE[key])
        out["cached"] = True
        return out
    t0 = time.perf_counter()
    try:
        res = exp.run(dict(resolved))
    except SimulationError as exc:
        raise RunError(422, {"message": f"수치 해법 실패 (SOLVER_FAILED): {exc}", "status": "SOLVER_FAILED"}) from exc
    except Exception as exc:  # surfaced, never hidden
        raise RunError(500, {"message": f"계산 중 오류: {exc}", "status": "SOLVER_FAILED", "trace": traceback.format_exc(limit=6)}) from exc
    dt = time.perf_counter() - t0
    res.warnings = list(warnings) + list(res.warnings)
    by_key = {p.key: p for p in exp.params}
    inputs = []
    for k, v in resolved.items():
        p = by_key[k]
        inputs.append(
            {
                "key": k,
                "label": p.label,
                "value": v,
                "unit": p.unit,
                "display": fmt_si(v, p.unit) if p.kind == "float" else str(v),
                "source": p.source,
                "source_note": p.source_note,
                "changed": k in changed,
                "group": p.group,
            }
        )
    out = {
        "lab": lab_id,
        "experiment": exp_key,
        "preset": preset,
        "inputs": inputs,
        "result": res.to_json(),
        "runtime_s": dt,
        "provenance": {
            "code_path": lab.code_path,
            "function": getattr(exp.run, "__name__", ""),
            "request_hash": key[:16],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **environment(),
        },
        "cached": False,
    }
    _CACHE[key] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out
