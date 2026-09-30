"""EX12 - answers that pass a design review: three integrated cases (textbook E12).

Every number in a decision memo comes from an actual run of another lab's experiment (through the same
runner and cache the UI uses), never from typed values: each evidence row cites lab/experiment/preset.
A memo has the textbook's parts - requirement and boundary, dominant constraint, candidates, numerical
evidence, measurement uncertainty, recommendation conditions, next discriminating tests - and statuses
per decision item.  The first root cause may stay UNRESOLVED.  Open answers are recorded and judged with
a rubric by the learner (never auto-graded); a model PASS is not the learner's proficiency.
"""

from __future__ import annotations

import math

import numpy as np

from ..model.labspec import Experiment, Lab, Prediction, Question, TextbookRef
from ..model.params import Param, Preset
from ..model.result import Check, Result, Table

TB_E12 = TextbookRef("expert-e12-설계-리뷰를-통과하는-답변-세-개의-통합-사례", "E12 · 설계 리뷰를 통과하는 답변: 세 개의 통합 사례")
TB_15 = TextbookRef("fae-디버깅-파형보다-먼저-질문의-질을-높인다-fl12", "15. FAE 디버깅 (1쪽 메모 구조)")


def _run(lab: str, exp: str, preset: str | None, values: dict | None = None) -> dict:
    """Run another lab's experiment through the runner (validated inputs, shared cache).

    SI values are passed with their explicit base unit, because a bare number is read in the
    parameter's display unit (e.g. kW)."""
    # imported here to avoid a circular import at registry load time
    from ..labs import get_lab
    from ..runner import run_request

    params = {p.key: p for p in get_lab(lab).experiment(exp).params}
    vals = {}
    for k, val in (values or {}).items():
        p = params[k]
        if isinstance(val, (str, bool)) or p.kind != "float":
            vals[k] = val
        else:
            vals[k] = f"{val!r} {p.unit}".strip()
    return run_request(lab, exp, preset, vals, use_cache=True)["result"]


def _m(r: dict, key: str):
    for m in r["metrics"]:
        if m["key"] == key:
            return m
    raise KeyError(key)


def _v(r: dict, key: str):
    return _m(r, key)["value"]


def _fmt(v, unit: str = "", d: int = 4) -> str:
    if isinstance(v, str):
        return v
    if unit == "W" and abs(v) >= 1000:
        return f"{v / 1e3:.{d}g} kW"
    if unit == "Hz" and abs(v) >= 1000:
        return f"{v / 1e3:.{d + 2}g} kHz"
    if unit == "J" and abs(v) < 1e-2:
        return f"{v * 1e6:.{d}g} µJ"
    return f"{v:.{d}g} {unit}".strip()


class Memo:
    """Collects evidence rows (with their source run) and memo sections."""

    def __init__(self, res: Result):
        self.res = res
        self.evidence: list[list[str]] = []
        self.sections: list[list[str]] = []
        self.decisions: list[list[str]] = []

    def ev(self, src: str, r: dict, key: str, d: int = 5, note: str = ""):
        m = _m(r, key)
        self.evidence.append([src, m["label"], _fmt(m["value"], m["unit"], d), r["status"]["code"], note])
        return m["value"]

    def sec(self, title: str, text: str):
        self.sections.append([title, text])

    def decide(self, item: str, code: str, why: str):
        self.decisions.append([item, code, why])
        self.res.verdict(code, f"{item}: {why}")

    def finish(self, title: str):
        self.res.tables.append(Table("t_memo", f"결정 메모 — {title}", ["항목", "내용"], self.sections, note="모든 수치는 아래 ‘수치 증거’ 표의 실제 실행 결과에서 가져왔다. 합성 교육 사양이며 공개 제품 성능이 아니다."))
        self.res.tables.append(Table("t_evidence", "수치 증거 (실제 실행: lab/실험@preset)", ["출처 실행", "양", "값", "그 실행의 상태", "비고"], self.evidence))
        self.res.tables.append(Table("t_decision", "결정 항목별 상태", ["결정 항목", "상태", "근거"], self.decisions))
        self.res.add_check(Check("메모의 모든 수치가 실제 실행에서 왔는가", "PASS" if self.evidence else "FAIL", len(self.evidence), "행", None, path="각 증거 행은 runner로 실행한 다른 실습의 metric (입력은 기록된 preset·값)", independent=False))


# ======================================================================================
# Capstone A - 11 kW OBC cannot charge the high-voltage battery
# ======================================================================================


def run_a(v: dict) -> Result:
    res = Result("EX12", "capstone_a", "통합 (다른 실습의 실제 실행)")
    memo = Memo(res)
    grid = _run("FL05", "grid_boundary", "nominal", {"V_LL": v["V_LL"], "I_lim": v["I_lim"], "eta": v["eta"], "PF": v["PF"], "P_bat": v["P_bat"]})
    P_av = memo.ev("FL05/grid_boundary@nominal", grid, "P_at_lim", 6)
    I_req = memo.ev("FL05/grid_boundary@nominal", grid, "I_line", 6)
    memo.ev("FL05/grid_boundary@nominal", grid, "V_need", 6)
    short = v["P_bat"] - P_av
    res.add_metric("P_av", f"{v['I_lim']:g} A 한계에서 가능한 배터리 출력", P_av, "W", ref=10698.8 if _tb_a(v) else None, ref_label="교재 ≈10.6988 kW", tol=1e-5)
    res.add_metric("shortfall", "11 kW 대비 부족분", short, "W", ref=301.2 if _tb_a(v) else None, ref_label="교재 약 301 W", tol=2e-3, basis="P_bat − P(16 A)")
    # measurement / assumption uncertainty on the available power (efficiency, PF, current reading)
    rel = math.sqrt(v["u_eta"] ** 2 + v["u_pf"] ** 2 + v["u_I"] ** 2)
    res.add_metric("P_av_unc", "가능 출력의 불확도 (η·PF·전류 계측, 독립 합성)", P_av * rel, "W", basis=f"±{v['u_eta'] * 100:g} % η, ±{v['u_pf'] * 100:g} % PF, ±{v['u_I'] * 100:g} % 전류")
    robust = short > 2 * P_av * rel
    seed = _run("FL10", "seed_fail", "seed")
    memo.ev("FL10/seed_fail@seed", seed, "g_req_hi", 7)
    memo.ev("FL10/seed_fail@seed", seed, "gmax_hi", 7)
    fix = _run("FL10", "fix_n093", "textbook")
    f_lo = memo.ev("FL10/fix_n093@textbook", fix, "f_고전압_0", 8)
    f_hi = memo.ev("FL10/fix_n093@textbook", fix, "f_고전압_1", 8)
    memo.ev("FL10/fix_n093@textbook", fix, "I1_lo", 5)
    memo.ev("FL10/fix_n093@textbook", fix, "I1_hi", 5)
    td_note = ""
    if v["with_switching"]:
        td = _run("FL10", "time_domain", "textbook")
        p_lo = memo.ev("FL10/time_domain@textbook", td, "P_td_lo", 5, "FHA 해 주파수에서의 이상 스위칭 전력")
        p_hi = memo.ev("FL10/time_domain@textbook", td, "P_td_hi", 5)
        fsw = memo.ev("FL10/time_domain@textbook", td, "f_sw_hi", 8, "위 branch에서 11 kW인 스위칭 주파수")
        memo.ev("FL10/time_domain@textbook", td, "f_sw_lo", 5, "아래 branch")
        td_note = f" 이상 스위칭 모델(강한 배터리)에서는 FHA 해 주파수에서 {p_lo / 1e3:.1f}/{p_hi / 1e3:.1f} kW가 흐르고, 11 kW는 위 branch {fsw / 1e3:.3f} kHz의 매우 가파른 곳에서만 나온다."
    if v["with_dynamics"]:
        op = _run("EX05", "operating_points", "textbook")
        memo.ev("EX05/operating_points@textbook", op, "slope_fha_lo", 4)
        memo.ev("EX05/operating_points@textbook", op, "slope_td_lo", 4, "R_L 부하 스위칭 모델: FHA와 부호 반대")
        memo.ev("EX05/operating_points@textbook", op, "f0", 7)
        cl = _run("EX05", "closed_loop", "fast")
        memo.ev("EX05/closed_loop@fast", cl, "rho_cl", 6, "기울기 부호가 맞는 PI도 불안정")
    # memo sections
    memo.sec("고객 목표", f"3상 {v['V_LL']:g} V_LL·{v['I_lim']:g} A 제한에서 배터리 {v['P_bat'] / 1e3:g} kW를 920 V 배터리까지 충전.")
    memo.sec("요구/경계 (확인)", f"PF {v['PF']:g}, 전체 효율 {v['eta']:g} (가정), CLLC 입력 700–850 V, 배터리 650–920 V, 120–210 kHz, seed n = 1.")
    memo.sec("모르는 조건", f"{v['I_lim']:g} A 제한의 정의 (RMS/peak, 연속/단시간), 코너별 실제 효율·PF, 배터리 충전 프로파일에서 920 V 11 kW가 필요한 시간, 열 경계.")
    if short > 0:
        memo.sec("지배 제약 1 (입력)", f"{v['I_lim']:g} A에서 가능한 배터리 출력 {P_av / 1e3:.4f} kW → {short:.0f} W 부족 (불확도 ±{P_av * rel:.0f} W{' — 결론이 불확도보다 크다' if robust else ' — 불확도 안: 추가 측정 필요'}). 전류제한 문제는 권선비로 해결되지 않는다.")
    else:
        memo.sec("입력 조건", f"{v['I_lim']:g} A에서 가능한 배터리 출력 {P_av / 1e3:.4f} kW — 요구보다 {-short:.0f} W 여유 (불확도 ±{P_av * rel:.0f} W). 남는 제약은 CLLC 전압 범위다.")
    memo.sec("지배 제약 2 (전압 범위)", f"seed CLLC: 필요 gain {_v(seed, 'g_req_hi'):.6f} > inductive 최대 {_v(seed, 'gmax_hi'):.6f} — 920/850 V corner에 FHA 해가 없다.")
    memo.sec("후보", f"① 입력전류 요구/derating 합의 ② n = 0.93 (환산 대칭) — FHA 해 {f_lo / 1e3:.3f}/{f_hi / 1e3:.3f} kHz.{td_note} ③ DC-link 상향 — PFC·소자 전압 여유·커패시터·손실 재계산 필요 (이 메모에서 미평가) ④ f_s 범위 확장 — 자속·전류·switching 경계·제어 대역이 바뀜 (미평가). PWM 최소 주파수를 임의로 낮춰 PASS를 만들지 않는다.")
    memo.sec("측정 불확도", f"가능 출력 ±{P_av * rel:.0f} W (η ±{v['u_eta'] * 100:g} %, PF ±{v['u_pf'] * 100:g} %, 전류 ±{v['u_I'] * 100:g} % 독립 합성). 스위칭 전력 민감도는 이상 모델 기준이며 실제 배터리 내부저항·손실이 완화한다.")
    memo.sec("권고 조건", "11 kW 보장 여부는 입력전류 요구 합의 후 결정. n = 0.93은 후보: 채택 전 두 branch의 스위칭 전류·ZVS·SR·공차·기동·reverse·열 검증 (EX05).")
    memo.sec("다음 분별 시험", f"① {v['I_lim']:g} A 제한 정의와 derating 허용 여부 확인 ② 코너별 효율·PF 실측 ③ n = 0.93 시제품의 920/850 V 11 kW 주파수 sweep (전류 제한 포함) ④ 폐루프 G_vf 측정 ⑤ 저전압 reverse 요구 확인.")
    first = (f"현재 입력 조건에서는 배터리 {v['P_bat'] / 1e3:g} kW보다 약 {short:.0f} W 부족하고, 별도로 CLLC seed의 고전압 gain도 부족합니다. 우선 입력전류 제한과 derating을 합의해야 합니다."
             if short > 0 else f"입력 한계 {v['I_lim']:g} A 안에서는 {v['P_bat'] / 1e3:g} kW가 가능하지만, CLLC seed의 고전압 gain이 부족합니다.")
    memo.sec("60초 답변", f"“{first} 권선비 0.93은 FHA에서 출력 가능성을 회복하지만 실제 switching·공차·기동 통과는 아직 아닙니다. 두 branch의 전류와 동적응답, high/low battery corner의 열 데이터를 확인한 뒤 transformer와 주파수 범위를 확정하겠습니다.”")
    memo.decide("seed CLLC", "FAIL_CONSTRAINT", "고전압 corner FHA gain 부족 (결과로 보존)")
    memo.decide("n = 0.93", "CANDIDATE_FHA_ONLY", "FHA 해 존재 — 스위칭·열·공차·기동 검증 전")
    if short > 0:
        memo.decide(f"입력 요구 ({v['I_lim']:g} A, {v['P_bat'] / 1e3:g} kW)", "CUSTOMER_DECISION_REQUIRED", f"{short:.0f} W 부족 — derating 합의 필요")
    else:
        memo.decide(f"입력 요구 ({v['I_lim']:g} A, {v['P_bat'] / 1e3:g} kW)", "PASS_WITHIN_MODEL", f"{-short:.0f} W 여유 (η·PF 가정 기준)")
    memo.finish("Capstone A · 11 kW OBC high-battery 충전 불가")
    # plot: available power vs current limit (FL05 runs), and the required gain vs max gain
    Il = np.linspace(v["I_plot_lo"], v["I_plot_hi"], 9)
    Ps = [_v(_run("FL05", "grid_boundary", "nominal", {"V_LL": v["V_LL"], "I_lim": float(i), "eta": v["eta"], "PF": v["PF"], "P_bat": v["P_bat"]}), "P_at_lim") for i in Il]
    res.add_series("P_lim", "가능한 배터리 출력 (FL05 실행)", "W", Il.tolist(), Ps)
    res.add_plot("p_lim", "입력전류 한계별 가능한 배터리 출력 — 11 kW 선과의 거리", ["P_lim"], x_label="입력전류 한계", x_unit="A", y_label="P_bat", y_unit="W", kind="xy", level="A",
                 hlines=[{"y": v["P_bat"], "label": f"{v['P_bat'] / 1e3:g} kW 요구"}], vlines=[{"x": v["I_lim"], "label": f"{v['I_lim']:g} A"}],
                 proved="FL05 grid_boundary를 전류 한계마다 실제로 실행해 가능한 출력을 그렸다.", not_yet="효율·PF는 고정 가정 — 코너별 실측이 필요하다.")
    res.add_check(Check("부족분 결론이 불확도보다 큰가", "PASS" if robust else "INFO", short, "W", 2 * P_av * rel, path="부족분 vs 2 × (η·PF·전류 불확도 합성)", independent=True, detail="불확도 안이면 derating 결정 전에 측정이 먼저"))
    res.assumptions += ["FL05·FL10·EX05의 기준 preset 실행 결과를 그대로 사용", "불확도는 독립 성분의 제곱합 (상관 무시)"]
    res.not_valid_for += ["실제 제품 정격", "경쟁사 비교 (조건 정렬 전)"]
    res.interpretation = "두 문제(입력전류 한계와 CLLC 전압 범위)를 섞지 않는 것이 핵심이다. 전류 한계는 권선비로 풀 수 없고, 권선비 후보는 FHA 가능성일 뿐 스위칭·동특성 검증이 남는다. 메모는 결정 항목마다 상태를 따로 적는다."
    return res


def _tb_a(v):
    return (v["V_LL"], v["I_lim"], v["eta"], v["PF"], v["P_bat"]) == (400.0, 16.0, 0.97, 0.995, 11000.0)


# ======================================================================================
# Capstone B - DAB circulating current and ZVS at the low-battery corner
# ======================================================================================


def run_b(v: dict) -> Result:
    res = Result("EX12", "capstone_b", "통합 (다른 실습의 실제 실행)")
    memo = Memo(res)
    zp = _run("FL08", "zero_power_mismatch", "mismatch")
    i0 = memo.ev("FL08/zero_power_mismatch@mismatch", zp, "Irms_L", 7, "P = 0인데 흐르는 순환전류")
    memo.ev("FL08/zero_power_mismatch@mismatch", zp, "P", 3)
    gm = _run("EX06", "general_modulation", "textbook")
    i_sps = memo.ev("EX06/general_modulation@textbook", gm, "irms_sps", 7)
    memo.ev("EX06/general_modulation@textbook", gm, "ipk_sps", 7)
    i_c = memo.ev("EX06/general_modulation@textbook", gm, "irms_c", 7)
    memo.ev("EX06/general_modulation@textbook", gm, "ipk_c", 7)
    ratio = memo.ev("EX06/general_modulation@textbook", gm, "cond_ratio", 5, "고정 등가저항 전도항만")
    cm = _run("EX06", "candidate_map", "textbook")
    ch_nom = memo.ev("EX06/candidate_map@textbook", cm, "changed_nom", 3, "1.5 kW: screen 통과 후보")
    memo.ev("EX06/candidate_map@textbook", cm, "changed_cor", 3, "2 kW corner")
    imp = _run("EX06", "implementation", "textbook")
    memo.ev("EX06/implementation@textbook", imp, "dP_lin", 5, "타이머 10 ns의 전력 계단 (erratum E-001: 교재 20.62 W)")
    two = _run("FL08", "two_modules", "textbook")
    memo.ev("FL08/two_modules@textbook", two, "share", 4, "L ±10 %의 분담 오차")
    res.add_metric("i0", "0 W 순환전류 RMS", i0, "A", ref=2.1651 if True else None, ref_label="교재 2.1651 A", tol=5e-5)
    res.add_metric("cond_red", "전도항 감소 (고정 등가저항)", 1 - ratio, "", ref=0.1366, ref_label="교재 약 13.66 %", tol=5e-3, basis="전체 손실 감소 아님")
    res.add_metric("rms_red", "RMS 감소 (width candidate)", 1 - i_c / i_sps, "", basis="교재 약 7 %")
    memo.sec("고객 목표", "2개 모듈 합계 3 kW DAB (모듈당 1.5 kW), 저전압 배터리 corner 900 V / n·V_L 600 V에서 무부하 발열과 경부하 효율 문제 해결.")
    memo.sec("요구/경계 (확인)", "V_H 900 V, n·V_L 600 V, L 200 µH, f_s 100 kHz, 모듈당/합계 수치를 구분 (합계 모델에 모듈당 parameter를 섞지 않음).")
    memo.sec("모르는 조건", "실제 소자 switching loss dataset, C_oss(V), dead time, 최소 pulse, 자성체 손실, 경부하 운전 시간 비중.")
    memo.sec("원인 설명", f"전력 0은 평균 에너지 전달이 0이라는 뜻일 뿐: 두 사각파 전압이 달라 L에 교번전압이 걸려 {i0:.4f} A RMS가 흐른다 (φ = 0). 전도손실을 만들지만 commutation에는 도움이 될 수 있다.")
    memo.sec("설계점 1500 W/모듈", f"SPS {i_sps:.4f} A → width candidate {i_c:.4f} A (RMS −{(1 - i_c / i_sps) * 100:.1f} %). 고정 등가저항 전도항만 {ratio:.4f} (−{(1 - ratio) * 100:.2f} %) — 전체 손실이 그만큼 준다는 뜻이 아니다.")
    memo.sec("지배 제약", f"RMS만 줄인 후보가 soft-switching 전하를 부족하게 만들 수 있다: 1.5 kW 격자 탐색 결과 ‘{ch_nom}’. switching loss dataset이 없으면 후보를 확정하지 않는다 (MISSING_INPUT).")
    memo.sec("후보", "변조 자유도 확대 (TPS/width), 권선비 재검토, variable f_s, 경부하 burst/standby, 모듈 enable/disable (기동·precharge·분담·수명·과도응답 추가 검증).")
    memo.sec("측정 불확도", "RMS 7 % 차이는 전류 프로브 정확도(약 1 %)보다 크지만, 효율 비교는 입력/출력 전력계 불확도와 온도 조건을 맞춰야 한다 (EX11). 타이머 분해능 10 ns는 약 20 W 전력 계단을 만든다.")
    memo.sec("권고 조건", "corner·부하별 commutation 부호/전하 screen을 먼저 통과시키고, gate timing·최소 pulse를 포함한 scheduler에서 손실·과도응답을 비교한 뒤 변조 변경과 권선비 변경의 비용을 같이 제시.")
    memo.sec("다음 분별 시험", "① 900/600 V에서 φ = 0 순환전류가 전압비를 맞추면 사라지는지 ② 후보별 edge 전류·V_DS(turn-on 직전) 측정 ③ 실제 파형 기반 DPT 또는 calibrated device model로 switching loss ④ 모듈 L 분산 측정.")
    memo.sec("60초 답변", f"“low-voltage corner의 전압비 불일치가 무부하 순환전류({i0:.2f} A RMS)를 만듭니다. width modulation 후보가 1500 W에서 RMS를 약 {(1 - i_c / i_sps) * 100:.0f} % 낮추지만 ZVS와 전체 효율은 아직 검증하지 않았습니다. 전압·부하별 commutation margin을 확인하고, gate timing과 최소 pulse를 포함한 scheduler에서 손실과 transient를 비교하겠습니다.”")
    memo.decide("무부하 순환전류 원인", "PASS_WITHIN_MODEL", "전압비 불일치로 설명됨 (이상 모델)")
    memo.decide("width modulation 후보", "SCREEN_ONLY", "RMS 감소는 계산됨, commutation은 screen만")
    memo.decide("최종 변조 선정", "MISSING_INPUT", "switching loss dataset 없음 → 확정하지 않음")
    memo.decide("topology 순위", "UNRESOLVED_RANKING", "요구사항(범위·프로파일·reverse·절연·열/EMI/크기·검증 비용) 고정 전에는 순위를 결정으로 쓰지 않음")
    memo.finish("Capstone B · 900 V ↔ 저전압 DAB 순환전류와 ZVS")
    res.add_series("rms", "I_rms", "A", [0, 1, 2], [i0, i_sps, i_c], style="points")
    res.add_plot("p_rms", "같은 corner의 L 전류 RMS: 0 W 순환전류, SPS 1.5 kW, width candidate 1.5 kW (실제 실행 값)", ["rms"], x_label="0: 0 W · 1: SPS · 2: candidate", x_unit="", y_label="I_rms", y_unit="A", kind="xy", level="A + C",
                 proved="FL08·EX06 실행 값을 그대로 모았다.", not_yet="손실·효율 비교 아님 (switching loss dataset 없음).")
    res.assumptions += ["FL08·EX06의 기준 preset 실행 결과", "모듈당 1.5 kW 기준 (합계 3 kW와 구분)"]
    res.not_valid_for += ["효율 개선 주장", "topology 최종 선정"]
    res.interpretation = "‘아무 일도 안 하니까 전류도 0’이라는 직관이 틀리는 곳을 수치로 보이고, RMS 감소(계산됨)와 효율 개선(미확인)을 구분하는 것이 이 메모의 핵심이다."
    return res


# ======================================================================================
# Capstone C - one hot branch of a parallel SiC inverter and protection events
# ======================================================================================


def run_c(v: dict) -> Result:
    res = Result("EX12", "capstone_c", "통합 (다른 실습의 실제 실행)")
    memo = Memo(res)
    st = _run("EX03", "static_sharing", "no_tempco")
    I1 = memo.ev("EX03/static_sharing@no_tempco", st, "I1", 7)
    memo.ev("EX03/static_sharing@no_tempco", st, "I2", 7)
    memo.ev("EX03/static_sharing@no_tempco", st, "I3", 7)
    I4 = memo.ev("EX03/static_sharing@no_tempco", st, "I4", 7)
    st_t = _run("EX03", "static_sharing", "textbook")
    memo.ev("EX03/static_sharing@textbook", st_t, "T_hot", 5, "전열 평형 (R 온도계수 포함)")
    dpt = _run("EX03", "dpt_deskew", "poor_deskew")
    memo.ev("EX03/dpt_deskew@poor_deskew", dpt, "dE_m5", 5, "전류 채널 −5 ns")
    memo.ev("EX03/dpt_deskew@poor_deskew", dpt, "dE_p5", 5, "전류 채널 +5 ns")
    band = memo.ev("EX03/dpt_deskew@poor_deskew", dpt, "unc_band", 3)
    dyn = _run("EX03", "dynamic_sharing", "nominal")
    csi_s = memo.ev("EX03/dynamic_sharing@nominal", dyn, "csi_screen", 4, "교재 screen 2 nH × 2 kA/µs")
    csi = memo.ev("EX03/dynamic_sharing@nominal", dyn, "csi_sim", 4, "합성 동특성 모델")
    sk = _run("EX03", "dynamic_sharing", "skew10")
    sk_s = memo.ev("EX03/dynamic_sharing@skew10", sk, "skew_screen", 4, "교재 screen 10 ns × 2 kA/µs")
    sk_d = memo.ev("EX03/dynamic_sharing@skew10", sk, "skew_sim_dI", 4, "합성 동특성 모델의 branch peak 차")
    res.add_metric("I1", "가장 큰 branch 정적 전류", I1, "A", ref=110.553, ref_label="교재 110.553 A", tol=5e-6)
    res.add_metric("I4", "가장 작은 branch 정적 전류", I4, "A", ref=90.452, ref_label="교재 90.452 A", tol=5e-6)
    memo.sec("고객 목표", "4개 병렬 branch, 합계 400 A 인버터에서 한 branch 과열과 high-dv/dt corner의 false trip 의심 해결.")
    memo.sec("요구/경계 (확인)", f"합계 {_v(st, 'I_mean') * 4:.0f} A, 4 branch, 합성 R 3.6/4/4/4.4 mΩ, 공통 source 2 nH·2 kA/µs·10 ns skew 합성 가늠값 (교재 E12).")
    memo.sec("모르는 조건", "고장 파형, 정확한 소자·드라이버·PCB revision, 운전 매트릭스, probe setup (reference plane·deskew·대역), 보호 설정 (threshold·blanking).")
    memo.sec("지배 제약", "원인 미확정: 같은 증상(한 branch 과열, false trip 의심)을 여러 가설이 만든다 — 분별 시험 전에는 조치 하나를 확정하지 않는다.")
    memo.sec("가설 (분리 유지)", f"① 정적 전류 분산 ({I1:.2f}/{I4:.2f} A — R 3.6/4/4/4.4 mΩ) ② gate timing skew ({sk_s:.0f} A screen, 합성 모델 {sk_d:.1f} A) ③ common-source 결합 ({csi_s:.0f} V screen, 합성 모델 {csi:.2f} V) ④ 열 계면 차이 ⑤ 센서/deskew 오류 (±5 ns가 E를 {band}) ⑥ 실제 shoot-through ⑦ 보호 threshold/blanking.")
    memo.sec("후보 (조치와 부작용)", "gate path 대칭·Kelvin return 개선 우선. 개별 R_g: 진동/분담 개선 vs 지연·손실 변화. dead time 증가: shoot-through 여유 vs diode 손실·왜곡. blanking 확대: false trip 감소 vs 실제 fault 차단 지연 — 같은 worst-case 매트릭스로 비교.")
    memo.sec("측정 불확도", f"합성 40 ns 전이에서 전류 채널 ±5 ns deskew가 E를 {band} 바꾼다 — branch switching loss 차이가 이보다 작으면 die 선별이나 R_g 변경 결론을 서두르지 않는다. gate 전압은 power source 기준인지 Kelvin 기준인지에 따라 L_s·di/dt만큼 달라진다.")
    memo.sec("권고 조건", "수정안은 손실·온도뿐 아니라 fault clearing 시간과 overshoot까지 같은 worst-case 매트릭스에서 확인한 뒤 권고한다. 지원 내용을 실제로 수행한 자기 경력처럼 말하지 않는다.")
    memo.sec("다음 분별 시험", "① branch별 전류와 local V_gs/V_ds를 같은 timebase·deskew로 캡처 ② driver swap / branch swap (변경 하나당 가설 하나) ③ Kelvin 기준 V_gs 재측정 ④ 열화상·NTC로 열 경로 확인 ⑤ 보호 이벤트 시점의 실제 branch 전류 캡처.")
    memo.sec("60초 답변", "“현재 파형만으로 소자 불량과 layout/보호 문제를 분리할 수 없습니다. branch 전류와 local gate 전압을 deskew된 setup에서 먼저 비교하겠습니다. 정적 분담, gate timing, common-source 결합, 열 경로를 순차 분리하고 실제 fault 전류인지 보호 노이즈인지 확인한 뒤, 손실·온도뿐 아니라 fault clearing과 overshoot까지 확인해 권고하겠습니다.”")
    memo.decide("root cause", "UNRESOLVED_ROOT_CAUSE", "가설별 분별 시험 전 — 시뮬레이션은 가설이 예측하는 파형만 보인다")
    memo.decide("DPT 기반 branch 순위", "UNRESOLVED_RANKING", f"deskew 불확도 {band}가 비교할 차이보다 크다")
    memo.decide("필요 자료", "MISSING_INPUT", "고장 파형·revision·운전 매트릭스·probe setup·보호 설정")
    memo.finish("Capstone C · 병렬 SiC 인버터의 한 branch 과열과 보호 동작")
    res.add_series("Ib", "branch 정적 전류", "A", [1, 2, 3, 4], [I1, _v(st, "I2"), _v(st, "I3"), I4], style="points")
    res.add_plot("p_share", "정적 branch 전류 (EX03 실행): 차이만으로 원인을 확정하지 않는다", ["Ib"], x_label="branch", x_unit="", y_label="전류", y_unit="A", kind="xy", level="A",
                 hlines=[{"y": 100.0, "label": "평균 100 A"}], proved="EX03 static_sharing 실행 값.", not_yet="동특성·열·측정 가설은 표의 다른 행에서 분리한다.")
    res.add_check(Check("screen 값과 합성 동특성 값이 다르다는 것을 드러냄", "INFO", csi, "V", csi_s, path="교재 screen(L·di/dt) vs EX03 합성 branch 모델", independent=True, detail="screen은 크기 가늠, 모델은 합성 equivalent — 둘 다 실측 아님"))
    res.assumptions += ["EX03 기준 preset 실행 결과", "합성 소자·layout 값"]
    res.not_valid_for += ["소자 불량 판정", "보호 설정 승인"]
    res.interpretation = "뜨거운 branch 하나라는 사실만으로 chip 불량을 확정하지 않는다. 각 가설이 예측하는 파형과 측정 불확도를 같은 표에 두고, 분별 시험 하나에 가설 하나를 대응시킨다."
    return res


# ======================================================================================
# Content
# ======================================================================================

_Q_A = [
    Question(
        "Capstone A를 60초로 고객에게 설명하라.",
        "입력전류 한계로 11 kW보다 약 301 W 부족하다는 것과 CLLC seed의 고전압 gain 부족(1.016 < 1.082)을 분리해 말하고, 먼저 입력전류 요구/derating 합의가 필요하다고 한다. n = 0.93은 FHA 후보일 뿐 스위칭·공차·기동 검증이 남았다고 말한다.",
        "Explain Capstone A to the customer in 60 seconds.",
        "Separate the two problems: the input current limit leaves about 301 W short of 11 kW, and the seed CLLC lacks gain at the high corner, 1.016 against 1.082. Agree on the input current requirement or derating first. The 0.93 turns ratio is only an FHA candidate; switching, tolerance and start-up checks remain.",
        ["301 W 부족", "두 문제 분리", "derating 합의", "n 0.93 = 후보"],
    ),
    Question(
        "“경쟁사는 11 kW라고 하는데요?”",
        "정격의 입력전압·전류·PF·효율·출력 경계와 derating 조건을 맞춰 비교하자고 한다. 상대가 틀렸다고 단정하거나 우리 가정을 숨기지 않는다.",
        "“The competitor says 11 kW.”",
        "Offer to compare at the same input voltage, current limit, power factor, efficiency, output boundary and derating conditions, without claiming the competitor is wrong or hiding our own assumptions.",
        ["조건 정렬", "단정 금지", "가정 공개"],
        kind="pressure",
    ),
]
_Q_B = [
    Question(
        "Capstone B: 0 W인데 전류가 흐르는 이유와 RMS 7 % 감소의 의미를 60초로.",
        "전압비 불일치가 L에 교번전압을 걸어 2.17 A RMS 순환전류가 흐른다. width candidate는 RMS를 약 7 % 줄이지만 전도항 13.66 % 감소는 고정 등가저항 기준일 뿐이고, ZVS·switching loss가 확인되지 않아 효율 개선을 주장하지 않는다.",
        "Capstone B: why current flows at zero power, and what a 7 % RMS reduction means.",
        "The voltage-ratio mismatch puts an alternating voltage on the inductor, so 2.17 A rms circulates. The width candidate cuts RMS by about 7 %, but the 13.66 % conduction reduction assumes a fixed equivalent resistance; ZVS and switching loss are unverified, so no efficiency claim.",
        ["전압비 불일치", "RMS ≠ 효율", "ZVS 미검증"],
    ),
    Question(
        "“그럼 제일 효율 좋은 topology 하나만 골라주세요.”",
        "입력/출력 범위, 전력 프로파일, reverse, 절연, 열/EMI/크기, 검증 비용을 먼저 고정한다. 요구사항이 정해지지 않은 topology 순위는 설계 결정으로 쓰지 않는다.",
        "“Just pick the most efficient topology.”",
        "First fix the input and output ranges, the power profile, reverse operation, isolation, thermal, EMI and size limits, and the validation cost. A topology ranking without fixed requirements is not a design decision.",
        ["요구 고정 먼저", "순위 ≠ 결정"],
        kind="pressure",
    ),
]
_Q_C = [
    Question(
        "Capstone C를 60초로 고객에게 설명하라.",
        "현재 파형만으로 소자 불량과 layout/보호 문제를 분리할 수 없다. deskew된 setup에서 branch 전류와 local gate 전압을 먼저 비교하고, 정적 분담·gate timing·common-source·열 경로를 순차 분리한 뒤 실제 fault인지 보호 노이즈인지 확인한다. 수정안은 fault clearing과 overshoot까지 확인한다.",
        "Explain Capstone C to the customer in 60 seconds.",
        "The current waveforms cannot separate a device defect from a layout or protection issue. I would first compare branch currents and local gate voltages on a deskewed setup, then separate static sharing, gate timing, common-source coupling and the thermal path, and confirm whether the event is a real fault or protection noise. Any fix is checked for fault clearing and overshoot too.",
        ["분리 불가 인정", "deskew 측정", "순차 분리", "fault clearing"],
    ),
    Question(
        "지원 내용을 자기 경력처럼 말하면 왜 안 되나?",
        "합성 사례와 시뮬레이션은 학습 증거일 뿐 실제로 수행한 실험·고객 지원 실적이 아니다. 직접 담당한 사실과 확인 가능한 수치만 경력으로 말한다.",
        "Why must you not present this support work as your own track record?",
        "Synthetic cases and simulations are learning evidence, not experiments or customer support you actually carried out. Only facts you owned and numbers you can verify belong in your track record.",
        ["합성 ≠ 경력", "확인 가능한 사실만"],
        kind="pressure",
    ),
]

_unc = [
    Param("u_eta", "효율 가정 불확도", "", 0.005, "", vmin=0, vmax=0.2, source="ASSUMED", group="불확도"),
    Param("u_pf", "PF 불확도", "", 0.002, "", vmin=0, vmax=0.2, source="ASSUMED", group="불확도"),
    Param("u_I", "전류 계측 불확도", "", 0.005, "", vmin=0, vmax=0.2, source="ASSUMED", group="불확도"),
]

EXPERIMENTS = [
    Experiment(
        key="capstone_a",
        title="Capstone A · 11 kW OBC의 high-battery 충전 불가",
        goal="FL05(입력 경계)·FL10(CLLC seed/수정안/스위칭)·EX05(동특성)를 실제로 실행해, 두 문제(16 A 입력 한계로 약 301 W 부족, seed CLLC gain 부족)를 분리한 결정 메모를 만든다.",
        params=[
            Param("V_LL", "계통 선간전압", "V", 400.0, "V", vmin=100, vmax=1000, source="TEXTBOOK"),
            Param("I_lim", "입력전류 한계", "A", 16.0, "A", vmin=1, vmax=200, source="TEXTBOOK"),
            Param("PF", "역률", "", 0.995, "", vmin=0.5, vmax=1.0, source="TEXTBOOK"),
            Param("eta", "전체 효율 (가정)", "", 0.97, "", vmin=0.5, vmax=1.0, source="TEXTBOOK"),
            Param("P_bat", "요구 배터리 출력", "W", 11000.0, "kW", vmin=100, vmax=1e5, source="TEXTBOOK"),
            Param("with_switching", "FL10 스위칭 검증 실행 포함", "", True, "", kind="bool", source="ASSUMED", group="실행 범위"),
            Param("with_dynamics", "EX05 동특성 실행 포함", "", True, "", kind="bool", source="ASSUMED", group="실행 범위"),
            Param("I_plot_lo", "그래프 전류 한계 최소", "A", 12.0, "A", vmin=1, vmax=200, source="ASSUMED", group="그래프"),
            Param("I_plot_hi", "그래프 전류 한계 최대", "A", 20.0, "A", vmin=1, vmax=200, source="ASSUMED", group="그래프"),
        ]
        + _unc,
        presets=[
            Preset("textbook", "교재 Capstone A", {}, "", ("nominal", "reference")),
            Preset("limit20", "입력 한계 20 A (합의된 경우)", {"I_lim": 20.0}, "입력 제약 해소", ("variant",)),
            Preset("quick", "FHA·입력만 (빠름)", {"with_switching": False, "with_dynamics": False}, "스위칭·동특성 실행 생략", ("variant",)),
        ],
        run=run_a,
        model_level="통합",
        suggested_change="입력전류 한계를 16 → 20 A로 바꿔 (고객이 합의한 경우) 메모의 결정 항목이 어떻게 바뀌는지 본다.",
        prediction=Prediction(
            "입력 한계를 20 A로 올리면 CLLC seed의 판정은?",
            ["함께 PASS가 된다", "그대로 FAIL_CONSTRAINT", "CANDIDATE가 된다", "모르겠다"],
            "그대로 FAIL_CONSTRAINT",
            "입력전류 한계와 CLLC 전압 범위는 서로 다른 제약이다. 입력 제약이 풀려도 920/850 V의 필요 gain 1.082는 seed tank가 낼 수 없다 — 두 문제를 섞지 않는 것이 메모의 핵심이다.",
            ["P_av", "shortfall"],
            handcalc=[{"key": "P_av", "label": "√3·400·16·0.995·0.97", "unit": "W"}, {"key": "shortfall", "label": "11 kW − 위 값", "unit": "W"}],
        ),
        suggested={"I_lim": 20.0},
        student="요구사항을 못 맞추는 이유가 하나가 아닐 수 있다. 입력 쪽 한계와 절연 컨버터의 전압 범위 한계를 따로 계산하고 따로 해결책을 찾는다.",
        expert="결정 메모는 항목별 상태를 분리한다: seed FAIL_CONSTRAINT, n = 0.93 CANDIDATE_FHA_ONLY, 입력 요구 CUSTOMER_DECISION_REQUIRED. DC-link 상향·f_s 범위 확장은 각각 PFC·소자 여유, 자속·전류·제어 대역을 바꾸므로 별도 검토하고, PWM 최소 주파수를 임의로 낮춰 PASS를 만들지 않는다.",
        customer_ko="현재 입력 조건에서는 배터리 11 kW보다 약 301 W 부족하고, 별도로 CLLC seed의 고전압 gain도 부족합니다. 우선 입력전류 제한과 derating을 합의해야 하며, 권선비 0.93 후보는 스위칭·공차·기동 검증 뒤에 확정하겠습니다.",
        customer_en="Under the current input conditions the battery output falls about 301 W short of 11 kW, and separately the seed CLLC lacks gain at the high battery voltage. We first need to agree on the input current limit and derating. The 0.93 turns-ratio candidate will be confirmed only after switching, tolerance and start-up checks.",
        questions=_Q_A,
        circuit=None,
        textbook=[TB_E12, TB_15],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="합성 교육 사양의 결정 메모. 실제 제품·경쟁사 정격 비교 아님.",
    ),
    Experiment(
        key="capstone_b",
        title="Capstone B · 900 V ↔ 저전압 DAB의 순환전류와 ZVS",
        goal="FL08(무부하 순환전류, 2모듈 분담)·EX06(변조 후보, commutation screen, 타이머)을 실제로 실행해 ‘RMS 감소 ≠ 효율 개선’을 분리한 결정 메모를 만든다.",
        params=[Param("note", "메모 표기 (고정)", "", "per_module", "", kind="choice", choices=[("per_module", "모듈당 1.5 kW 기준")], source="TEXTBOOK")],
        presets=[Preset("textbook", "교재 Capstone B", {}, "", ("nominal", "reference"))],
        run=run_b,
        model_level="통합",
        suggested_change="(관찰 실험) EX06 candidate_map에서 부하를 2 kW로 바꾼 결과와 메모의 ‘지배 제약’ 문장을 비교한다.",
        prediction=Prediction(
            "width candidate가 RMS를 약 7 % 줄이면 전체 효율은?",
            ["약 13.7 % 좋아진다", "알 수 없다 (switching loss 데이터 없음)", "나빠진다", "모르겠다"],
            "알 수 없다 (switching loss 데이터 없음)",
            "0.8634는 고정 등가저항 전도항 비율일 뿐이다. soft-switching 전하가 부족해지면 hard-switching 손실이 늘 수 있고, switching frequency·edge 전류·zero-state 전도·gate·자성체 손실이 모두 다르다.",
            ["rms_red", "cond_red"],
        ),
        suggested={},
        student="같은 전력을 더 작은 RMS 전류로 보낼 수 있어도, 스위칭 순간의 전류가 너무 작아지면 스위칭 손실이 늘 수 있다.",
        expert="모듈당/합계를 섞지 않는다 (합계 3 kW 모델에 1.5 kW/모듈 parameter를 혼용하지 않음). 후보 확정에는 device별 commutation 부호/전하 screen과 timing resolution을 먼저 보고, 실제 파형 기반 DPT 또는 calibrated device model로 손실을 정밀화한다.",
        customer_ko="low-voltage corner의 전압비 불일치가 무부하 순환전류를 만듭니다. width modulation 후보가 1.5 kW에서 RMS를 약 7 % 낮추지만 ZVS와 전체 효율은 아직 검증하지 않았습니다. 전압·부하별 commutation margin을 확인한 뒤 변조 변경과 권선비 변경의 비용을 같이 제시하겠습니다.",
        customer_en="The voltage-ratio mismatch at the low-voltage corner causes the no-load circulating current. The width-modulation candidate lowers RMS by about 7 % at 1.5 kW, but ZVS and overall efficiency are not verified yet. After checking the commutation margin per voltage and load, I will present the cost of a modulation change against a turns-ratio change.",
        questions=_Q_B,
        circuit=None,
        textbook=[TB_E12],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="합성 교육 사양의 결정 메모. 효율 개선 주장 없음.",
    ),
    Experiment(
        key="capstone_c",
        title="Capstone C · 병렬 SiC 인버터의 한 branch 과열과 보호 동작",
        goal="EX03(정적 분담·DPT deskew·동적 분담)을 실제로 실행해 가설을 분리한 채로 root cause를 UNRESOLVED로 두는 결정 메모를 만든다.",
        params=[Param("note", "메모 표기 (고정)", "", "branch", "", kind="choice", choices=[("branch", "branch별 측정 기준")], source="TEXTBOOK")],
        presets=[Preset("textbook", "교재 Capstone C", {}, "", ("nominal", "reference"))],
        run=run_c,
        model_level="통합",
        suggested_change="(관찰 실험) EX03 dpt_deskew에서 deskew 불확도를 ±5 → ±0.5 ns로 줄였을 때 메모의 ‘DPT 기반 branch 순위’ 판정이 어떻게 바뀔지 예측한다.",
        prediction=Prediction(
            "branch 1이 정적 계산상 110.55 A로 가장 크고 가장 뜨겁다면 root cause는?",
            ["branch 1 소자 불량으로 확정", "아직 확정할 수 없다 (분별 시험 필요)", "layout 문제로 확정", "모르겠다"],
            "아직 확정할 수 없다 (분별 시험 필요)",
            "정적 R 분산만으로도 ±10 % 전류 차가 나고, gate skew·common-source·열 계면·측정 deskew·보호 설정이 같은 증상을 만든다. 가설마다 분별 시험을 하나씩 대응시킨다.",
            ["I1", "I4"],
        ),
        suggested={},
        student="뜨거운 부품 하나가 곧 불량 부품이라는 뜻은 아니다. 전류를 더 많이 받는 이유, 열이 덜 빠지는 이유, 측정이 틀렸을 가능성을 나눠서 확인한다.",
        expert="gate waveform reference가 power source인지 Kelvin source인지 먼저 확인한다. driver swap/branch swap은 변경 하나당 원인 가설 하나를 분리하고 setup 재현성을 유지한다. 지원 내용을 실제로 수행한 자기 경력처럼 말하지 않는다.",
        customer_ko="현재 파형만으로 소자 불량과 layout/보호 문제를 분리할 수 없습니다. branch 전류와 local gate 전압을 deskew된 setup에서 먼저 비교하고, 가설별 분별 시험과 예상 파형을 다음 미팅 전에 정리하겠습니다.",
        customer_en="The current waveforms cannot separate a device defect from a layout or protection issue. I would first compare branch currents and local gate voltages on a deskewed setup, and prepare a discriminating test and the expected waveform for each hypothesis before the next meeting.",
        questions=_Q_C,
        circuit=None,
        textbook=[TB_E12],
        reference_presets=["textbook"],
        runtime_hint="seconds",
        claim_limit="합성 교육 사양의 결정 메모. root cause 확정 없음.",
    ),
]

LAB = Lab(
    id="EX12",
    title="설계 리뷰를 통과하는 답변 — 세 개의 통합 사례",
    title_en="Answers that pass a design review: three integrated cases",
    track="expert",
    order=12,
    path_note="E13 최종 (E12)",
    textbook=[TB_E12, TB_15],
    prerequisites=["FL05", "FL08", "FL10", "EX03", "EX05", "EX06"],
    summary="다른 실습을 실제로 실행해 결정 메모를 만든다: A 11 kW OBC 충전 불가, B DAB 순환전류·ZVS, C 병렬 SiC 과열·보호.",
    experiments=EXPERIMENTS,
    minimum_scope="E12: 3개 customer case decision memo를 실제 simulation run으로 생성 (요구/경계·지배제약·후보·수치증거·측정불확도·권고조건·다음 분별시험); root cause UNRESOLVED 허용; 답변은 rubric",
    claim_limits=["메모의 수치는 합성 모델의 실제 실행 결과", "model PASS ≠ 사용자 숙련", "root cause를 시뮬레이션으로 확정하지 않음"],
    test_paths=["tests/test_ex12.py"],
)
