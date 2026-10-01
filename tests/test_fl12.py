"""FL12 FAE debugging cases A-H: textbook numbers, independent checks, multiple-hypothesis tables, no root cause."""

import functools

import pytest

from convlab.labs import get_lab
from convlab.labs.fl12_cases import MEMO_POINTS
from convlab.model.params import ParamError, resolve_params

CASES = [f"case_{c}" for c in "abcdefgh"]


@functools.lru_cache(maxsize=None)
def run(exp, preset=None):
    e = get_lab("FL12").experiment(exp)
    vals, _, _ = resolve_params(e.params, e.presets, preset or e.reference_presets[0], {})
    return e.run(vals).to_json()


def metric(js, key):
    return next(m for m in js["metrics"] if m["key"] == key)


def value(js, key):
    return metric(js, key)["value"]


def table(js, key):
    return next(t for t in js["tables"] if t["key"] == key)


def groups(js, test_label_start):
    """The 'groups' cell of t_groups for the test whose label starts with the given text."""
    return next(r[2] for r in table(js, "t_groups")["rows"] if r[0].startswith(test_label_start))


def test_lab_metadata():
    lab = get_lab("FL12")
    assert lab.id == "FL12" and lab.track == "basic" and lab.order == 12
    assert lab.textbook[0].anchor == "fae-디버깅-파형보다-먼저-질문의-질을-높인다-fl12"
    assert lab.test_paths == ["tests/test_fl12.py"]
    assert [e.key for e in lab.experiments] == CASES
    assert {"FL02", "FL05", "FL07", "FL08", "FL09", "EX02", "EX09", "EX10", "EX11"} <= set(lab.prerequisites)


# ---------------------------------------------------------------- textbook numbers (literal)


def test_case_a_textbook_screens():
    js = run("case_a")
    assert value(js, "t_fv") == pytest.approx(10e-9, rel=1e-9)  # Q_gd / I_g Miller time, 10 ns
    assert value(js, "Ls_screen") == pytest.approx(4.0, rel=1e-9)  # 2 nH x 2 kA/us = 4 V


def test_case_b_textbook_line_currents():
    js = run("case_b")
    assert value(js, "I_400") == pytest.approx(16.45043, rel=2e-6)
    assert value(js, "I_360") == pytest.approx(18.278, rel=2e-5)
    assert value(js, "P16_400") == pytest.approx(10698.81, rel=2e-6)
    assert "CUSTOMER_DECISION_REQUIRED" in {v["code"] for v in js["verdicts"]}


def test_case_c_textbook_mismatch_current():
    js = run("case_c")
    assert value(js, "I_H1") == pytest.approx(2.165063509, rel=1e-9)
    assert value(js, "Ipk_H1") == pytest.approx(3.75, rel=1e-12)


def test_case_e_flux_walk_numbers_from_fl07():
    js = run("case_e")
    assert value(js, "walk_100") == pytest.approx(0.0064, rel=1e-9)
    assert value(js, "Bpk_sym") == pytest.approx(0.16, rel=1e-12)


def test_case_f_loss_uncertainty():
    js = run("case_f")
    assert value(js, "P_e") == pytest.approx(200.0, rel=1e-12)
    assert value(js, "gap") == pytest.approx(150.0, rel=1e-12)
    assert value(js, "u_ex11_ind") == pytest.approx(15.40157, rel=1e-6)
    assert value(js, "u_ex11_rho") == pytest.approx(4.87487, rel=1e-5)
    assert value(js, "u_e_ind") == pytest.approx(14.0014, rel=1e-5)  # sqrt(10^2 + 9.8^2)
    assert value(js, "z_gap") > 2.0  # the 150 W gap is not explained by the declared uncertainty


def test_case_g_edge_rate_is_not_the_spectrum():
    js = run("case_g")
    assert value(js, "icm_ex09") == pytest.approx(5.0, rel=1e-12)  # 100 pF x 50 kV/us
    assert value(js, "f_loop0") == pytest.approx(50.329e6, rel=1e-5)  # 10 nH, 1 nF
    assert value(js, "d_dvdt") == pytest.approx(-7.9588, abs=1e-3)
    assert value(js, "d_edge") == pytest.approx(-2.0238, abs=1e-3)


def test_case_h_reproduces_ex10():
    js = run("case_h")
    assert value(js, "E0") == pytest.approx(320.0, rel=1e-12)
    assert value(js, "i_pk10") == pytest.approx(4370.344, rel=1e-7)
    assert value(js, "t_pk10") == pytest.approx(120.919958e-6, rel=1e-8)


# ---------------------------------------------------------------- multiple-hypothesis contract


@pytest.mark.parametrize("exp", CASES)
def test_root_cause_stays_unresolved_and_tables_cover_three_hypotheses(exp):
    js = run(exp)
    assert js["status"]["code"] == "UNRESOLVED_ROOT_CAUSE"
    assert "UNRESOLVED_ROOT_CAUSE" in {v["code"] for v in js["verdicts"]}
    disc = table(js, "t_disc")
    assert len(disc["rows"]) >= 3
    assert len(disc["columns"]) >= 4  # hypothesis column + 'before test' + at least two tests
    assert len(table(js, "t_groups")["rows"]) == len(disc["columns"]) - 1


@pytest.mark.parametrize("exp", CASES)
def test_reference_presets_checks_pass_and_plots_are_explained(exp):
    e = get_lab("FL12").experiment(exp)
    for pk in e.reference_presets:
        js = run(exp, pk)
        assert js["status"]["code"] == "UNRESOLVED_ROOT_CAUSE", (exp, pk)
        for c in js["checks"]:
            assert c["status"] in ("PASS", "INFO", "NOT_RUN"), (exp, pk, c)
        for m in js["metrics"]:
            assert m["check"] in (None, "PASS", "INFO"), (exp, pk, m)
        assert any(c["independent"] and c["status"] == "PASS" for c in js["checks"]), (exp, pk)
        assert js["plots"], (exp, pk)
        for p in js["plots"]:
            assert p["proved"].strip() and p["not_yet"].strip(), (exp, pk, p["key"])


@pytest.mark.parametrize("exp", CASES)
def test_cited_numbers_come_from_runs_of_real_labs(exp):
    js = run(exp)
    ev = table(js, "t_evidence")["rows"]
    assert ev
    for row in ev:
        lab_exp, preset = row[0].split(" ")[0].split("@")
        lab, e = lab_exp.split("/")
        assert get_lab(lab).experiment(e) is not None and preset


@pytest.mark.parametrize("exp", CASES)
def test_questions_are_rubrics_not_graded(exp):
    e = get_lab("FL12").experiment(exp)
    assert len(e.questions) == 4
    for q in e.questions:
        assert q.kind in ("interview", "check")
        assert len(q.must_include) >= 3
    memo = e.questions[3]
    for point in MEMO_POINTS:
        assert point in memo.must_include
    assert e.expert.startswith("교재 해설")
    assert "[1쪽 메모]" in e.customer_ko and e.customer_en.startswith("At this operating point")


# ---------------------------------------------------------------- what the discriminating tests show


def test_case_a_kelvin_removes_only_measurement_worlds():
    js = run("case_a", "kelvin")
    assert "{H1, H4}: 감소 | {H2, H3}: 사라짐" in groups(js, "Kelvin")


def test_case_c_matched_ratio_removes_only_the_mismatch():
    js = run("case_c", "ratio")
    assert groups(js, "전압비 일치").startswith("{H1}: 사라짐")


def test_case_d_one_change_at_a_time_separates_the_worlds():
    js = run("case_d", "restore_lm")
    assert "{H1, H12}: 해결" in groups(js, "L_m만 원래로")
    assert "{H1}: 그대로" in groups(js, "소자만 원래로")
    # the same observation in every modelled world, and gate charge contradicts the full-load improvement
    assert value(js, "dPf_H1") < 0 and value(js, "dPf_H2") < 0 and value(js, "dPf_H12") < 0
    assert value(js, "dPf_H5") > 0


def test_case_d_large_observation_rules_out_the_lm_only_world():
    js = run("case_d", "big_obs")
    assert groups(js, "시험 전").startswith("{H1}: 재현 불가")


def test_case_e_probe_and_demag():
    js = run("case_e", "vs")
    assert "{H5}: 센서만 증가" in groups(js, "두 번째 DC 결합")
    assert "{H1}: 사라짐" in groups(js, "소거")
    assert value(js, "Ipk_ss") == pytest.approx(1.0, rel=1e-3)  # the periodic model shows the healthy 1 A peak


def test_case_g_rg_only_separates_edge_related_sources():
    js = run("case_g", "rg")
    assert "{H1, H2}: 약 −2 dB (edge 영향) | {H3, H4}: ≈ 0 dB (edge 무관)" in groups(js, "R_g만")
    assert "{H2}: 주파수 이동" in groups(js, "cable")


def test_case_h_battery_current_and_contactor_show_backfeed():
    js = run("case_h", "ibat")
    assert "{H2}: battery 전류 큼" in groups(js, "battery 전류")
    assert "{H2}: 줄어듦" in groups(js, "battery contactor")
    assert value(js, "i_bf") == pytest.approx(2845.158, rel=1e-6)


def test_case_h_ideal_short_is_out_of_validity():
    js = run("case_h", "ideal_short")
    assert js["status"]["code"] == "OUT_OF_VALIDITY"
    assert value(js, "i_pk10") > 5e4


def test_inputs_are_rejected_not_clamped():
    e = get_lab("FL12").experiment("case_f")
    with pytest.raises(ParamError) as err:
        resolve_params(e.params, e.presets, "observed", {"eta": "1.5", "rel_u": "-1"})
    assert {"eta", "rel_u"} <= set(err.value.errors)
    e = get_lab("FL12").experiment("case_d")
    with pytest.raises(ParamError):
        resolve_params(e.params, e.presets, "observed", {"load_light": "0.95"})
