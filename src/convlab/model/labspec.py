"""Lab and experiment specifications: physics entry point + the learning flow.

The learning flow the UI enforces for every experiment:
  이번에 배우는 것 (goal) -> 바꿀 파라미터 (suggested change) -> 먼저 예측 (prediction)
  -> 실행 -> 파형에서 확인 -> 왜 그런가 (student / expert, folded)
  -> 고객에게 어떻게 말할까 (KO, English folded) -> 면접 질문 (answers folded).

User-entered answers, the simulation verdict and "learning complete" are separate
fields in the user store; nothing here marks a person as interview-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .params import Param, Preset
from .result import Result

__all__ = ["Question", "Prediction", "Experiment", "Lab", "TextbookRef"]


@dataclass
class TextbookRef:
    anchor: str  # section id in Infineon_FAE_Expert_Integrated.html v4.0
    title: str


@dataclass
class Question:
    q_ko: str
    answer_ko: str
    q_en: str = ""
    answer_en: str = ""
    must_include: list[str] = field(default_factory=list)  # rubric points
    kind: str = "interview"  # interview | check | pressure | calc

    def to_json(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Prediction:
    question: str
    options: list[str]
    expected: str  # expected option for the suggested change (shown only after the run)
    why: str  # shown after the run
    metric_keys: list[str] = field(default_factory=list)  # metrics to compare before/after
    handcalc: list[dict] = field(default_factory=list)  # [{key, label, unit}] numbers to hand-calculate before the first run

    def to_json(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Experiment:
    key: str
    title: str
    goal: str
    params: list[Param]
    presets: list[Preset]
    run: Callable[[dict], Result]
    model_level: str
    suggested_change: str
    prediction: Prediction
    student: str
    expert: str
    customer_ko: str
    customer_en: str
    questions: list[Question] = field(default_factory=list)
    circuit: str | None = None
    textbook: list[TextbookRef] = field(default_factory=list)
    reference_presets: list[str] = field(default_factory=list)  # presets run by run-all/verification
    suggested: dict = field(default_factory=dict)  # machine-readable suggested change {param: SI value}
    runtime_hint: str = "fast"  # fast | seconds | slow
    claim_limit: str = ""

    def meta(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "goal": self.goal,
            "model_level": self.model_level,
            "suggested_change": self.suggested_change,
            "prediction": self.prediction.to_json(),
            "student": self.student,
            "expert": self.expert,
            "customer_ko": self.customer_ko,
            "customer_en": self.customer_en,
            "questions": [q.to_json() for q in self.questions],
            "circuit": self.circuit,
            "textbook": [t.__dict__ for t in self.textbook],
            "params": [p.to_json() for p in self.params],
            "presets": [p.to_json() for p in self.presets],
            "reference_presets": self.reference_presets,
            "suggested": self.suggested,
            "runtime_hint": self.runtime_hint,
            "claim_limit": self.claim_limit,
        }


@dataclass
class Lab:
    id: str  # FL01 .. FL12, EX01 .. EX12 (the only official IDs)
    title: str
    title_en: str
    track: str  # basic | expert
    order: int  # position in the learning path of its track
    path_note: str  # e.g. "14일 경로 1일차" / "E13 1회전"
    textbook: list[TextbookRef]
    prerequisites: list[str]
    summary: str
    experiments: list[Experiment]
    minimum_scope: str  # from the contract (reconstructed)
    claim_limits: list[str]
    implementation: str = "COMPLETE"  # COMPLETE | PARTIAL (+ reason)
    implementation_note: str = ""
    code_path: str = ""
    test_paths: list[str] = field(default_factory=list)
    extends: list[str] = field(default_factory=list)  # FL labs an EX lab reuses

    def experiment(self, key: str) -> Experiment:
        for e in self.experiments:
            if e.key == key:
                return e
        raise KeyError(f"{self.id}: unknown experiment {key}")

    def meta(self, full: bool = False) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "title_en": self.title_en,
            "track": self.track,
            "order": self.order,
            "path_note": self.path_note,
            "textbook": [t.__dict__ for t in self.textbook],
            "prerequisites": self.prerequisites,
            "summary": self.summary,
            "minimum_scope": self.minimum_scope,
            "claim_limits": self.claim_limits,
            "implementation": self.implementation,
            "implementation_note": self.implementation_note,
            "code_path": self.code_path,
            "test_paths": self.test_paths,
            "extends": self.extends,
            "experiments": [e.meta() if full else {"key": e.key, "title": e.title, "model_level": e.model_level} for e in self.experiments],
        }
        return d
