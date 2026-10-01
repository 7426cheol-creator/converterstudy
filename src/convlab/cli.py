"""Command line: serve the app, list/run labs, run everything, export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .labs import OFFICIAL_IDS, all_labs, get_lab
from .runner import RunError, environment, run_request

__all__ = ["main"]


def _parse_set(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_list(_args) -> int:
    labs = all_labs()
    for lid in OFFICIAL_IDS:
        lab = labs.get(lid)
        if lab is None:
            print(f"{lid}  (not implemented)")
            continue
        print(f"{lid}  {lab.title}  [{lab.implementation}]")
        for e in lab.experiments:
            presets = ", ".join(p.key for p in e.presets)
            print(f"      {e.key:22s} level {e.model_level:26s} presets: {presets}")
    return 0


def cmd_run(args) -> int:
    try:
        out = run_request(args.lab, args.experiment, args.preset, _parse_set(args.set), use_cache=False)
    except RunError as exc:
        print(json.dumps(exc.payload, ensure_ascii=False, indent=1))
        return 2
    res = out["result"]
    print(f"{out['lab']} / {out['experiment']} / {out['preset']}  status={res['status']['code']} ({res['status']['ko']})  level={res['model_level']}  {out['runtime_s']:.2f}s")
    for m in res["metrics"]:
        ref = "" if m["ref"] is None else f"  ref {m['ref']:.6g} ({m['ref_label']})  err {m['rel_err']:.2e} {m['check'] or ''}"
        val = m["value"] if isinstance(m["value"], str) else f"{m['value']:.6g}"
        print(f"   {m['label']:40s} {val} {m['unit']}{ref}")
    for c in res["checks"]:
        print(f"   [{c['status']}] {c['name']}: {c['detail'][:140]}")
    for v in res["verdicts"]:
        print(f"   verdict {v['code']}: {v['why']}")
    if args.out:
        from .export import to_csv, to_html, to_json

        base = Path(args.out)
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{out['lab']}_{out['experiment']}_{out['preset'] or 'custom'}"
        (base / f"{stem}.json").write_text(to_json(out), encoding="utf-8")
        (base / f"{stem}.csv").write_text("﻿" + to_csv(out), encoding="utf-8")
        lab = get_lab(out["lab"])
        (base / f"{stem}.html").write_text(to_html(out, {"lab": lab.meta(), "experiment": lab.experiment(out["experiment"]).meta()}), encoding="utf-8")
        print(f"   wrote {base / stem}.[json|csv|html]")
    return 0


def cmd_run_all(args) -> int:
    """Run every experiment's reference presets; write results and the run manifest."""
    from .export import to_csv, to_html, to_json

    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "environment": environment(), "runs": [], "not_implemented": []}
    labs = all_labs()
    only = set(args.only.split(",")) if args.only else None
    t_all = time.perf_counter()
    for lid in OFFICIAL_IDS:
        if only and lid not in only:
            continue
        lab = labs.get(lid)
        if lab is None:
            manifest["not_implemented"].append(lid)
            continue
        for e in lab.experiments:
            presets = [pr.key for pr in e.presets] if getattr(args, "all_presets", False) else (e.reference_presets or [e.presets[0].key])
            for pk in presets:
                rec = {"lab": lid, "experiment": e.key, "preset": pk}
                t0 = time.perf_counter()
                try:
                    out = run_request(lid, e.key, pk, {}, use_cache=False)
                    res = out["result"]
                    stem = f"{lid}_{e.key}_{pk}"
                    (base / lid).mkdir(exist_ok=True)
                    (base / lid / f"{stem}.json").write_text(to_json(out), encoding="utf-8")
                    (base / lid / f"{stem}.csv").write_text("﻿" + to_csv(out), encoding="utf-8")
                    (base / lid / f"{stem}.html").write_text(to_html(out, {"lab": lab.meta(), "experiment": e.meta()}), encoding="utf-8")
                    issues = contract_issues(res)
                    if issues:
                        rec["contract_issues"] = issues
                    rec.update(
                        {
                            "status": res["status"]["code"],
                            "verdicts": [v["code"] for v in res["verdicts"]],
                            "model_level": res["model_level"],
                            "checks_total": len(res["checks"]) + sum(1 for m in res["metrics"] if m["check"]),
                            "checks_failed": [c["name"] for c in res["checks"] if c["status"] == "FAIL"] + [m["key"] for m in res["metrics"] if m["check"] == "FAIL"],
                            "files": [f"{lid}/{stem}.{x}" for x in ("json", "csv", "html")],
                            "sha256": {x: hashlib.sha256((base / lid / f"{stem}.{x}").read_bytes()).hexdigest() for x in ("json", "csv", "html")},
                            "runtime_s": round(time.perf_counter() - t0, 3),
                        }
                    )
                except RunError as exc:
                    rec.update({"status": "SOLVER_FAILED", "error": exc.payload.get("message"), "runtime_s": round(time.perf_counter() - t0, 3)})
                manifest["runs"].append(rec)
                flag = "OK " if not rec.get("checks_failed") and rec["status"] != "SOLVER_FAILED" and not rec.get("contract_issues") else "!! "
                print(f"{flag}{lid} {e.key:22s} {pk:14s} {rec['status']:22s} {rec['runtime_s']:.2f}s {rec.get('checks_failed') or ''} {rec.get('contract_issues') or ''}")
    manifest["total_runtime_s"] = round(time.perf_counter() - t_all, 2)
    (base / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [r for r in manifest["runs"] if r.get("checks_failed") or r["status"] == "SOLVER_FAILED"]
    contract = [r for r in manifest["runs"] if r.get("contract_issues")]
    print(f"{len(manifest['runs'])} runs, {len(bad)} with failed checks, {len(contract)} with contract issues, {manifest['total_runtime_s']} s → {base / 'run_manifest.json'}")
    return 1 if bad or contract else 0


def contract_issues(res: dict) -> list[str]:
    """Presentation rules every result must meet (instruction §13): each plot says, in a paragraph, what it proved
    and what it has not proved yet. Returned as messages; run-all fails on any."""
    out = []
    for pl in res.get("plots", []):
        missing = [k for k in ("proved", "not_yet") if not str(pl.get(k) or "").strip()]
        if missing:
            out.append(f"plot {pl.get('key')}: {'/'.join(missing)} 없음")
    return out


def cmd_serve(args) -> int:
    from .server import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="convlab", description="Converter FAE Lab (textbook v4.0, FL01-FL12 / EX01-EX12)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("serve", help="로컬 학습 화면 실행 (http://127.0.0.1:8765)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("list", help="실습·실험·preset 목록")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("run", help="실험 하나 실행")
    p.add_argument("lab")
    p.add_argument("experiment")
    p.add_argument("--preset")
    p.add_argument("--set", action="append", help="key=value (예: L=50u, fs=50k)")
    p.add_argument("--out", help="json/csv/html 저장 폴더")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("run-all", help="모든 실험의 기준 preset 실행 + run manifest")
    p.add_argument("--out", default="results")
    p.add_argument("--only", help="FL01,EX02 처럼 일부만")
    p.add_argument("--all-presets", action="store_true", help="기준 preset만이 아니라 모든 preset 실행 (오류·계약 점검용)")
    p.set_defaults(func=cmd_run_all)
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        args = ap.parse_args(["serve"])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
