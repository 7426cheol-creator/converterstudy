#!/usr/bin/env python3
"""Generate the evidence documents from the lab registry and a real `convlab run-all` output.

    python -m convlab run-all --out results
    python tools/gen_docs.py --results results

Writes (all derived from code and actual runs, never typed by hand):
  docs/TRACEABILITY.md                      24-row table: textbook -> lab -> code -> tests -> run result -> status
  docs/MODEL_CARDS.md                       per lab / experiment: model level, assumptions, not valid for, verification
  docs/TEST_INDEPENDENCE.md                 independent checks vs regression checks per lab
  docs/run_manifest.json                    copy of the run-all manifest (evidence snapshot)
  docs/contract/simulation_contract_v4_reconstructed.json   ids, scope, presets, statuses (the original was not provided)
  docs/contract/reference_results_reconstructed.json        every metric with a reference value, as computed
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convlab import CONTRACT_VERSION, __version__  # noqa: E402
from convlab.labs import OFFICIAL_IDS, all_labs  # noqa: E402

HW = "미수행 (하드웨어·실측 없음)"


def load_runs(results: Path) -> tuple[dict, dict]:
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    runs = {}
    for rec in manifest["runs"]:
        f = results / rec["lab"] / f"{rec['lab']}_{rec['experiment']}_{rec['preset']}.json"
        if f.exists():
            runs[(rec["lab"], rec["experiment"], rec["preset"])] = json.loads(f.read_text(encoding="utf-8"))
    return manifest, runs


def count_tests(path: Path) -> int:
    if not path.exists():
        return 0
    return len(re.findall(r"^def test_", path.read_text(encoding="utf-8"), flags=re.M))


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results)
    manifest, runs = load_runs(results)
    labs = all_labs()
    docs = ROOT / "docs"
    (docs / "contract").mkdir(parents=True, exist_ok=True)
    now = manifest.get("generated", time.strftime("%Y-%m-%dT%H:%M:%S"))

    # ---------------- traceability ----------------
    # one row per lab (the 24 official ids), then one row per experiment x reference preset
    rows = []
    detail = []
    trace_json = []
    for lid in OFFICIAL_IDS:
        lab = labs.get(lid)
        if lab is None:
            rows.append(f"| {lid} | - | 미구현 | - | - | - | - | - | - | - | {HW} |")
            trace_json.append({"lab": lid, "implemented": False, "hardware_validation": "NOT_DONE"})
            continue
        my = [r for r in manifest["runs"] if r["lab"] == lid]
        n_fail = sum(1 for r in my if r.get("checks_failed") or r["status"] == "SOLVER_FAILED")
        statuses = sorted({r["status"] for r in my})
        n_ind = n_reg = n_ref = n_ref_pass = 0
        nvf: list[str] = []
        exp_json = []
        for r in my:
            js = runs.get((r["lab"], r["experiment"], r["preset"]))
            if not js:
                continue
            res = js["result"]
            ind = sum(1 for c in res["checks"] if c.get("independent") and c["status"] == "PASS")
            reg = sum(1 for c in res["checks"] if not c.get("independent"))
            refs_here = [m for m in res["metrics"] if m.get("ref") is not None and m.get("check")]
            rp = sum(m["check"] == "PASS" for m in refs_here)
            n_ind += ind
            n_reg += reg
            n_ref += len(refs_here)
            n_ref_pass += rp
            for x in res["not_valid_for"]:
                if x not in nvf:
                    nvf.append(x)
            exp = lab.experiment(r["experiment"])
            files = r.get("files", [])
            detail.append(
                f"| {lid} | {md_escape(r['experiment'])} | {md_escape(r['preset'])} | {md_escape(exp.model_level)} | `{r['status']}` | {md_escape(' · '.join(r.get('verdicts', [])))} | "
                f"{rp}/{len(refs_here)} | {ind} / {reg} | {'실패: ' + md_escape(', '.join(r['checks_failed'])) if r.get('checks_failed') else '없음'} | "
                f"{('`results/' + files[0].rsplit('.', 1)[0] + '.{json,csv,html}`') if files else '-'} | {md_escape('; '.join(res['not_valid_for'][:3]))} |"
            )
            exp_json.append({"experiment": r["experiment"], "preset": r["preset"], "model_level": exp.model_level, "status": r["status"], "verdicts": r.get("verdicts", []), "reference_metrics": [rp, len(refs_here)], "independent_checks_passed": ind, "regression_checks": reg, "checks_failed": r.get("checks_failed", []), "result_files": ["results/" + f for f in files], "not_valid_for": res["not_valid_for"]})
        tb = "; ".join(t.title for t in lab.textbook[:2])
        test_files = lab.test_paths or [f"tests/test_{lid.lower()}.py"]
        n_tests = sum(count_tests(ROOT / p) for p in test_files)
        exps = ", ".join(e.key for e in lab.experiments)
        code = f"src/convlab/labs/{Path(lab.code_path).name}" if lab.code_path else "-"
        rows.append(
            f"| {lid} | {md_escape(tb)} | {md_escape(exps)} | `{code}` | {', '.join('`' + p + '`' for p in test_files)} ({n_tests}) | `results/{lid}/` | "
            f"{len(my)} runs · 실패 {n_fail} · {' / '.join(statuses)} | 기준값 {n_ref_pass}/{n_ref} · 독립 {n_ind} · 회귀 {n_reg} | {md_escape('; '.join(nvf[:3]))} | {md_escape('; '.join(lab.claim_limits[:2]))} | {HW} |"
        )
        trace_json.append({"lab": lid, "implemented": True, "textbook": [t.title for t in lab.textbook], "experiments": [e.key for e in lab.experiments], "code": code, "tests": test_files, "n_tests": n_tests, "runs": len(my), "runs_failed": n_fail, "statuses": statuses, "reference_metrics": [n_ref_pass, n_ref], "independent_checks_passed": n_ind, "regression_checks": n_reg, "result_dir": f"results/{lid}/", "not_valid_for": nvf, "claim_limits": lab.claim_limits, "hardware_validation": "NOT_DONE", "runs_detail": exp_json})
    n_impl = sum(1 for x in trace_json if x.get("implemented"))
    head = (
        f"# 추적표 (Traceability) — 교재 v4.0 ↔ 실습 ↔ 코드 ↔ 테스트 ↔ 실행 결과\n\n"
        f"생성: `tools/gen_docs.py` · run-all {now} · 앱 {__version__} · 계약 {CONTRACT_VERSION} · 구현 {n_impl}/24\n\n"
        "열: **교재 절**(학습 내용) · **시나리오**(실험 key) · **코드** · **테스트**(test 함수 수) · **결과**(`convlab run-all --out results`가 쓰는 폴더; 저장소에는 "
        "run manifest만 커밋) · **실제 실행**(기준 preset 실행 수, 실패 수, 나온 상태) · **검증**(기준값 metric 통과/전체, 독립 경로 check PASS 수, 회귀 check 수) · "
        "**not_valid_for**(적용 불가, 앞 3개) · **주장 한계** · **하드웨어 검증**(모두 미수행). "
        "상태는 합성 모델 안의 판정이며 학습 완료나 면접 준비 완료를 뜻하지 않는다. 의도된 FAIL(예: FL10 seed)은 실패 수에 들어가지 않는다 — 실패 수는 검증 check 실패와 수치 실패만 센다.\n\n"
        "| Lab | 교재 절 | 시나리오 | 코드 | 테스트 | 결과 | 실제 실행 | 검증 | not_valid_for | 주장 한계 | 하드웨어 검증 |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    dhead = (
        "\n\n## 실험 × 기준 preset 상세\n\n"
        "| Lab | 실험 | preset | 모델 수준 | 헤드라인 | 판정 전체 | 기준값 | 독립 / 회귀 | 검증 실패 | 결과 파일 | not_valid_for (앞 3개) |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    (docs / "TRACEABILITY.md").write_text(head + "\n".join(rows) + dhead + "\n".join(detail) + "\n", encoding="utf-8")
    (docs / "traceability.json").write_text(json.dumps({"generated": now, "implemented": n_impl, "labs": trace_json}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- errata (rendered from docs/errata.json, which is written by hand) ----------------
    ej = docs / "errata.json"
    if ej.exists():
        er = json.loads(ej.read_text(encoding="utf-8"))
        out = [f"# Errata — 교재·지침 수치와 계산의 차이\n\n`docs/errata.json`에서 생성 (`tools/gen_docs.py`). {er.get('note', '')}\n"]
        out.append("\n## 정정\n")
        for it in er.get("items", []):
            out.append(f"\n### {it['id']} · {it['where']}\n\n- **원문:** {it['original']}\n- **정정:** {it['corrected']}\n- **근거:** {it['evidence']}\n- **영향:** {it['impact']}\n")
        if er.get("findings"):
            out.append("\n## 발견 (정정 아님: 교재 수치는 맞고, 더 높은 수준의 모델에서 결론이 달라진다)\n")
            for it in er["findings"]:
                out.append(f"\n### {it['id']} · {it['where']}\n\n- **교재:** {it['textbook']}\n- **모델 결과:** {it['model_result']}\n- **근거:** {it['evidence']}\n- **영향:** {it['impact']}\n")
        (docs / "ERRATA.md").write_text("".join(out), encoding="utf-8")

    # ---------------- model cards and independence map ----------------
    mc = [f"# 모델 카드 (Model cards)\n\n생성: `tools/gen_docs.py` · run-all {now}\n\n모델 수준: A 해석/FHA · B 평균 동역학 · C 이상 스위칭 · D 비이상 commutation(합성). 가정·적용 불가 범위는 각 실험의 기준 preset 실행 결과에서 그대로 가져왔다.\n"]
    ti = [f"# 테스트 독립성 지도 (Test independence map)\n\n생성: `tools/gen_docs.py` · run-all {now}\n\n**독립** = 서로 다른 식/적분기/표현으로 계산한 두 경로의 비교 (`Check.independent = True`). **회귀** = 같은 코드 경로의 재현성 확인. 교재 기준값은 테스트 파일에 숫자로 직접 적혀 있다 (`reference/` 모듈을 다시 부르지 않는다).\n"]
    contract = {"contract_version": CONTRACT_VERSION, "note": "원본 simulation_contract_v4.json이 제공되지 않아 구현된 실습과 실제 실행에서 재구성했다 (docs/ERRATA.md E-000).", "generated": now, "labs": [], "expert_labs": []}
    refs = []
    for lid in OFFICIAL_IDS:
        lab = labs.get(lid)
        if lab is None:
            continue
        mc.append(f"\n## {lid} · {lab.title}\n\n범위: {lab.minimum_scope}\n\n주장 한계: {'; '.join(lab.claim_limits)}\n")
        ti.append(f"\n## {lid} · {lab.title}\n")
        entry = {"id": lid, "title": lab.title, "title_en": lab.title_en, "minimum_scope": lab.minimum_scope, "claim_limits": lab.claim_limits, "experiments": []}
        for e in lab.experiments:
            pk = (e.reference_presets or [e.presets[0].key])[0]
            js = runs.get((lid, e.key, pk))
            res = js["result"] if js else None
            mc.append(f"\n### {e.key} — {e.title}\n\n- 모델 수준: **{e.model_level}**" + (f" · 기준 실행 상태: `{res['status']['code']}`" if res else ""))
            if res:
                if res["assumptions"]:
                    mc.append("- 가정: " + "; ".join(res["assumptions"]))
                if res["not_valid_for"]:
                    mc.append("- 적용 불가: " + "; ".join(res["not_valid_for"]))
                mc.append("- 주장 한계: " + (e.claim_limit or "-"))
            ind = []
            reg = []
            for p in e.reference_presets or [e.presets[0].key]:
                r2 = runs.get((lid, e.key, p))
                if not r2:
                    continue
                for c in r2["result"]["checks"]:
                    (ind if c.get("independent") else reg).append((p, c))
                for m in r2["result"]["metrics"]:
                    if m.get("ref") is not None:
                        refs.append({"lab": lid, "experiment": e.key, "preset": p, "key": m["key"], "label": m["label"], "value": m["value"], "unit": m["unit"], "ref": m["ref"], "ref_label": m.get("ref_label", ""), "tol": m.get("tol"), "check": m.get("check")})
            ti.append(f"\n**{e.key}** — 독립 {len(ind)} · 회귀 {len(reg)}\n")
            for p, c in ind:
                ti.append(f"- [{c['status']}] ({p}) {md_escape(c['name'])} — {md_escape(c.get('path', ''))}")
            for p, c in reg:
                ti.append(f"- [회귀 {c['status']}] ({p}) {md_escape(c['name'])}")
            entry["experiments"].append({"key": e.key, "title": e.title, "model_level": e.model_level, "presets": [{"key": pr.key, "label": pr.label, "values": pr.values, "tags": list(pr.tags)} for pr in e.presets], "reference_presets": e.reference_presets, "status_of_reference": res["status"]["code"] if res else None, "claim_limit": e.claim_limit})
        (contract["labs"] if lid.startswith("FL") else contract["expert_labs"]).append(entry)
    (docs / "MODEL_CARDS.md").write_text("\n".join(mc) + "\n", encoding="utf-8")
    (docs / "TEST_INDEPENDENCE.md").write_text("\n".join(ti) + "\n", encoding="utf-8")
    (docs / "contract" / "simulation_contract_v4_reconstructed.json").write_text(json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")
    (docs / "contract" / "reference_results_reconstructed.json").write_text(json.dumps({"generated": now, "metrics": refs}, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(results / "run_manifest.json", docs / "run_manifest.json")
    print(f"wrote docs for {sum(1 for x in trace_json if x.get('implemented'))} labs, {len(refs)} reference metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
