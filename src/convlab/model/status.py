"""Result status vocabulary.

Different failures are different facts and are never merged into one red flag:
a design that does not meet a constraint (FAIL_CONSTRAINT), an operating point with
no steady solution (NO_SOLUTION), a solution that exists but is unstable (UNSTABLE),
a numerical failure (SOLVER_FAILED), missing real data (MISSING_INPUT), an input
outside the model's validity (OUT_OF_VALIDITY), a question the model cannot answer
(NOT_EVALUABLE) and an external tool that was not available (NOT_RUN_ENVIRONMENT).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Status", "STATUS", "status_info", "worst"]


@dataclass(frozen=True)
class Status:
    code: str
    ko: str
    meaning: str
    tone: str  # ok | info | warn | fail | blocked | neutral
    rank: int  # higher = more important to surface as the headline


_ALL = [
    Status("PASS_WITHIN_MODEL", "모델 범위 내 통과", "계산이 완료되었고 기준을 만족한다. 모델 가정 밖(실측·양산)의 통과를 뜻하지 않는다.", "ok", 10),
    Status("INFO", "정보", "판정 대상이 아닌 계산 결과.", "neutral", 5),
    Status("SCREEN_ONLY", "스크리닝만", "필요조건 성격의 선별 계산(예: 전하·에너지 screen). 실제 통과 판정이 아니다.", "info", 30),
    Status("CANDIDATE_FHA_ONLY", "FHA 후보", "FHA 정적 해만 존재. 스위칭·ZVS·역방향·기동 검증 전의 후보.", "info", 35),
    Status("NOT_EVALUABLE", "평가 불가", "이 모델 수준으로는 답할 수 없는 질문(예: 이상 스위치 모델의 ZVS).", "blocked", 40),
    Status("MISSING_INPUT", "입력 없음", "정밀 주장을 하려면 실제 데이터(재료·소자·측정)가 필요하다. 합성 학습 모델은 따로 실행된다.", "blocked", 45),
    Status("NOT_RUN_ENVIRONMENT", "환경 없음·미실행", "외부 도구(MATLAB/Simulink/PSIM)가 이 환경에 없어 실행하지 않았다.", "blocked", 42),
    Status("UNRESOLVED_RANKING", "순위 미확정", "성능 차이가 불확도보다 작아 순위를 확정할 수 없다.", "warn", 50),
    Status("CUSTOMER_DECISION_REQUIRED", "고객 결정 필요", "요구 자체(입력전류 한계, derating 등)를 고객과 합의해야 한다.", "warn", 52),
    Status("UNRESOLVED_ROOT_CAUSE", "근본원인 미확정", "관측만으로 가설을 분리할 수 없다. 판별 시험이 필요하다.", "warn", 53),
    Status("OUT_OF_VALIDITY", "모델 적용범위 밖", "입력이 모델 적용범위를 벗어났다. 값을 조용히 잘라내지 않고 계산을 거부했다.", "warn", 55),
    Status("MARGINAL", "경계 운전점", "수학적으로는 가능하지만 여유가 거의 없다(예: 전압여유 0.24 V).", "warn", 58),
    Status("FAIL_CONSTRAINT", "제약 불만족", "계산은 정상이며, 설계가 요구 제약을 만족하지 못한다.", "fail", 70),
    Status("NO_SOLUTION", "해 없음", "지정 범위에 정상상태 해가 존재하지 않는다.", "fail", 72),
    Status("UNSTABLE", "불안정", "평형점/주기해는 존재하지만 국소적으로 불안정하다.", "fail", 74),
    Status("NO_STABLE_FIXED_POINT", "안정 고정점 없음", "전열 등 반복 계산의 고정점이 없거나 발산한다(단순 모델의 결과; 실물 파손 예측 아님).", "fail", 75),
    Status("SOLVER_FAILED", "수치 실패", "수치 해법이 실패했다. 물리적 불안정과 구분한다.", "fail", 80),
]

STATUS: dict[str, Status] = {s.code: s for s in _ALL}


def status_info(code: str) -> Status:
    try:
        return STATUS[code]
    except KeyError as exc:  # pragma: no cover - programming error
        raise KeyError(f"unknown status code {code!r}") from exc


def worst(codes) -> str:
    codes = [c for c in codes if c]
    if not codes:
        return "INFO"
    return max(codes, key=lambda c: status_info(c).rank)
