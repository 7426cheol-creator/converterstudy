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
    rows = []
    trace_json = []
    for lid in OFFICIAL_IDS:
        lab = labs.get(lid)
        if lab is None:
            rows.append(f"| {lid} | - | 미구현 | - | - | - | - | - | - | {HW} |")
            trace_json.append({"lab": lid, "implemented": False})
            continue
        my = [r for r in manifest["runs"] if r["lab"] == lid]
        n_fail = sum(1 for r in my if r.get("checks_failed") or r["status"] == "SOLVER_FAILED")
        statuses = sorted({r["status"] for r in my})
        n_ind = 0
        n_ref = 0
        n_ref_pass = 0
        for r in my:
            js = runs.get((r["lab"], r["experiment"], r["preset"]))
            if not js:
                continue
            res = js["result"]
            n_ind += sum(1 for c in res["checks"] if c.get("independent") and c["status"] == "PASS")
            for m in res["metrics"]:
                if m.get("ref") is not None and m.get("check"):
                    n_ref += 1
                    n_ref_pass += m["check"] == "PASS"
        tb = "; ".join(t.title for t in lab.textbook[:2])
        test_files = lab.test_paths or [f"tests/test_{lid.lower()}.py"]
        n_tests = sum(count_tests(ROOT / p) for p in test_files)
        exps = ", ".join(e.key for e in lab.experiments)
        code = f"src/convlab/labs/{Path(lab.code_path).name}" if lab.code_path else "-"
        rows.append(
            f"| {lid} | {md_escape(tb)} | {md_escape(lab.title)} | {md_escape(exps)} | `{code}` | {', '.join('`' + p + '`' for p in test_files)} ({n_tests}) | "
            f"{len(my)} runs, 실패 {n_fail} · {' / '.join(statuses)} | 기준값 {n_ref_pass}/{n_ref} · 독립 검산 {n_ind} | {md_escape('; '.join(lab.claim_limits[:2]))} | {HW} |"
        )
        trace_json.append({"lab": lid, "implemented": True, "experiments": [e.key for e in lab.experiments], "code": code, "tests": test_files, "n_tests": n_tests, "runs": len(my), "runs_failed": n_fail, "statuses": statuses, "reference_metrics": [n_ref_pass, n_ref], "independent_checks_passed": n_ind, "hardware_validation": "NOT_DONE"})
    head = (
        f"# 추적표 (Traceability) — 교재 v4.0 ↔ 실습 ↔ 코드 ↔ 테스트 ↔ 실행 결과\n\n"
        f"생성: `tools/gen_docs.py` · run-all {now} · 앱 {__version__} · 계약 {CONTRACT_VERSION}\n\n"
        "열 설명: **학습 내용**(교재 절) · **구현**(실험 key) · **코드/테스트** · **실제 실행**(`convlab run-all`의 기준 preset 실행 수, 실패 수, 나온 상태) · "
        "**검증**(기준값 metric 통과 수 / 독립 경로 check 수) · **범위 한계** · **하드웨어 검증**(모두 미수행). 상태는 모델 안에서의 판정이며 학습 완료나 면접 준비 완료를 뜻하지 않는다.\n\n"
        "| Lab | 교재 | 제목 | 실험 | 코드 | 테스트 (개수) | 실제 실행 | 검증 | 범위 한계 | 하드웨어 검증 |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    (docs / "TRACEABILITY.md").write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    (docs / "traceability.json").write_text(json.dumps({"generated": now, "labs": trace_json}, ensure_ascii=False, indent=1), encoding="utf-8")

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
