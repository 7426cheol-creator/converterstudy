"""The learner's own record, kept locally (no login, no cloud).

Three things are always separate:
  * what the person predicted / answered (their own words and numbers),
  * what the simulation reported (result status of a run),
  * whether the person marks the experiment as learned (set only by the person).
Nothing in the app sets "learned" or "interview ready" automatically.
Do not enter confidential company or customer data here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

__all__ = ["UserStore", "default_store_path"]


def default_store_path() -> Path:
    env = os.environ.get("CONVLAB_USERDATA")
    if env:
        return Path(env)
    return Path.home() / ".convlab" / "progress.json"


class UserStore:
    VERSION = 1

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_store_path()
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (OSError, ValueError):
            return {"version": self.VERSION, "experiments": {}, "sessions": []}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def get(self) -> dict:
        with self._lock:
            return self._load()

    def _slot(self, data: dict, lab: str, exp: str) -> dict:
        key = f"{lab}/{exp}"
        slot = data["experiments"].setdefault(
            key,
            {"predictions": [], "answers": {}, "handcalc": [], "self_assessment": {}, "learned": False, "learned_at": None, "last_sim_status": None, "notes": ""},
        )
        return slot

    def record(self, lab: str, exp: str, kind: str, payload: dict) -> dict:
        """kind: prediction | answer | handcalc | self_assessment | learned | sim_status | notes."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            data = self._load()
            slot = self._slot(data, lab, exp)
            if kind == "prediction":
                slot["predictions"].append({**payload, "at": now})
            elif kind == "handcalc":
                slot["handcalc"].append({**payload, "at": now})
            elif kind == "answer":
                qid = str(payload.get("qid"))
                slot["answers"][qid] = {"text": payload.get("text", ""), "at": now}
            elif kind == "self_assessment":
                slot["self_assessment"] = {**payload.get("scores", {}), "at": now}
            elif kind == "learned":
                slot["learned"] = bool(payload.get("value"))
                slot["learned_at"] = now if slot["learned"] else None
            elif kind == "sim_status":
                slot["last_sim_status"] = {"code": payload.get("code"), "preset": payload.get("preset"), "at": now}
            elif kind == "notes":
                slot["notes"] = str(payload.get("text", ""))[:20000]
            else:
                raise ValueError(f"unknown record kind {kind}")
            self._save(data)
            return slot
