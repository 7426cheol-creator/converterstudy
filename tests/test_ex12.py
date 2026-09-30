"""EX12 design-review memos: numbers come from actual runs, required memo parts, decision statuses."""

import functools

import pytest

from convlab.labs import get_lab
from convlab.model.params import resolve_params
from convlab.runner import run_request

REQUIRED = ["요구/경계", "지배 제약", "후보", "측정 불확도", "권고 조건", "다음 분별 시험"]


@functools.lru_cache(maxsize=None)
def run(exp, preset="textbook"):
    e = get_lab("EX12").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset, {})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)["value"]


def table(js, key):
    return next(t for t in js["tables"] if t["key"] == key)


def decisions(js):
    return {row[1] for row in table(js, "t_decision")["rows"]}


@pytest.mark.parametrize("exp", ["capstone_a", "capstone_b", "capstone_c"])
def test_memo_has_every_required_part_and_cited_evidence(exp):
    js = run(exp)
    titles = " ".join(r[0] for r in table(js, "t_memo")["rows"])
    for part in REQUIRED:
        assert part in titles, (exp, part)
    ev = table(js, "t_evidence")["rows"]
    assert len(ev) >= 6
    for row in ev:
        lab_exp, preset = row[0].split("@")
        lab, e = lab_exp.split("/")
        assert get_lab(lab).experiment(e) is not None and preset


def test_capstone_a_numbers_and_decisions():
    js = run("capstone_a")
    assert metric(js, "P_av") == pytest.approx(10698.8, rel=1e-5)  # textbook 10.6988 kW
    assert metric(js, "shortfall") == pytest.approx(301.2, abs=0.5)  # textbook "about 301 W"
    assert {"FAIL_CONSTRAINT", "CANDIDATE_FHA_ONLY", "CUSTOMER_DECISION_REQUIRED"} <= decisions(js)


def test_capstone_a_evidence_equals_a_direct_run_of_the_source_lab():
    js = run("capstone_a")
    direct = run_request("FL05", "grid_boundary", "nominal", {}, use_cache=False)["result"]
    p = next(m for m in direct["metrics"] if m["key"] == "P_at_lim")["value"]
    assert metric(js, "P_av") == pytest.approx(p, rel=1e-12)


def test_capstone_a_input_limit_agreed_keeps_the_cllc_failure():
    js = run("capstone_a", "limit20")
    d = {row[0]: row[1] for row in table(js, "t_decision")["rows"]}
    assert d["seed CLLC"] == "FAIL_CONSTRAINT"
    assert any(k.startswith("입력 요구") and v == "PASS_WITHIN_MODEL" for k, v in d.items())


def test_capstone_b_rms_is_not_efficiency():
    js = run("capstone_b")
    assert metric(js, "i0") == pytest.approx(2.1651, abs=5e-5)
    assert metric(js, "cond_red") == pytest.approx(0.1366, abs=5e-4)
    assert {"MISSING_INPUT", "UNRESOLVED_RANKING"} <= decisions(js)


def test_capstone_c_root_cause_stays_unresolved():
    js = run("capstone_c")
    assert js["status"]["code"] == "UNRESOLVED_ROOT_CAUSE"
    assert metric(js, "I1") == pytest.approx(110.553, abs=5e-4)
    assert metric(js, "I4") == pytest.approx(90.452, abs=5e-4)
