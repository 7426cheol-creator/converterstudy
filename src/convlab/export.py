"""Exports: CSV (long format), JSON, and a self-contained HTML report.

The HTML report inlines the same plotting and circuit code the app uses, so the
exported page is interactive offline and shows exactly what the app showed:
values with units and basis, reference comparisons, model level, assumptions,
not-valid-for, verification checks and the provenance of the run.
"""

from __future__ import annotations

import csv
import html
import io
import json
from importlib import resources

__all__ = ["to_csv", "to_json", "to_html"]


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1)


def to_csv(payload: dict) -> str:
    res = payload["result"]
    buf = io.StringIO()
    w = csv.writer(buf)
    prov = payload.get("provenance", {})
    w.writerow(["# Converter FAE Lab export", payload["lab"], payload["experiment"], payload.get("preset") or "", prov.get("timestamp", ""), f"app {prov.get('app_version', '')}", f"contract {prov.get('contract_version', '')}"])
    w.writerow(["# model_level", res["model_level"], "status", res["status"]["code"]])
    w.writerow([])
    w.writerow(["section", "key", "label", "value", "unit", "source/basis", "ref", "ref_label", "rel_err", "check"])
    for i in payload["inputs"]:
        w.writerow(["input", i["key"], i["label"], i["value"], i["unit"], f"{i['source']} {i['source_note']}".strip(), "", "", "", ""])
    for m in res["metrics"]:
        w.writerow(["metric", m["key"], m["label"], m["value"], m["unit"], m["basis"], m["ref"], m["ref_label"], m["rel_err"], m["check"] or ""])
    for c in res["checks"]:
        w.writerow(["check", c["name"], c["path"], c["value"], c["unit"], c["detail"], c["threshold"], "independent" if c["independent"] else "", "", c["status"]])
    for v in res["verdicts"]:
        w.writerow(["verdict", v["code"], v["ko"], "", "", v["why"], "", "", "", ""])
    w.writerow([])
    w.writerow(["series", "key", "label", "x", "y", "unit"])
    for s in res["series"]:
        for x, y in zip(s["x"], s["y"]):
            w.writerow(["series", s["key"], s["label"], x, y, s["unit"]])
    for t in res["tables"]:
        w.writerow([])
        w.writerow(["table", t["key"], t["title"]])
        w.writerow(t["columns"])
        for r in t["rows"]:
            w.writerow(r)
    return buf.getvalue()


def _asset(name: str) -> str:
    return resources.files("convlab").joinpath("web", name).read_text(encoding="utf-8")


def to_html(payload: dict, meta: dict | None = None) -> str:
    """Standalone report page (no network)."""
    res = payload["result"]
    title = f"{payload['lab']} · {payload['experiment']} · {res['status']['ko']}"
    data = json.dumps({"payload": payload, "meta": meta or {}}, ensure_ascii=False).replace("</", "<\\/")
    css = _asset("style.css")
    js = "\n".join(_asset(n).replace("export ", "") for n in ("fmt.js", "plot.js", "circuit.js", "report.js"))
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}</style></head>
<body class="report"><main id="report"></main>
<script>const REPORT_DATA = {data};</script>
<script>{js}
renderReport(document.getElementById('report'), REPORT_DATA);</script>
</body></html>"""
