"""Local web server (stdlib only).  Binds to 127.0.0.1; no login, no cloud, no LLM."""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from .export import to_csv, to_html, to_json
from .labs import OFFICIAL_IDS, all_labs, get_lab
from .runner import RunError, environment, run_request
from .userstore import UserStore

__all__ = ["serve", "make_handler"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _web_root():
    return resources.files("convlab").joinpath("web")


def app_meta() -> dict:
    labs = all_labs()
    docs = {}
    for name in ("traceability.json", "run_manifest.json", "errata.json"):
        p = REPO_ROOT / "docs" / name
        if p.exists():
            try:
                docs[name] = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                docs[name] = None
    tb = _textbook_path()
    return {
        "env": environment(),
        "official_ids": OFFICIAL_IDS,
        "implemented": sorted(labs),
        "textbook_available": tb is not None,
        "docs": docs,
        "external_tools": {
            "matlab_simulink": "NOT_RUN_ENVIRONMENT",
            "psim_12_0_2": "NOT_RUN_ENVIRONMENT",
        },
    }


def _textbook_path() -> Path | None:
    for base in (REPO_ROOT / "textbook", Path.cwd() / "textbook"):
        if base.is_dir():
            for p in sorted(base.glob("*.html")):
                return p
    return None


def make_handler(store: UserStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ConverterFAELab/4.0"

        def log_message(self, fmt, *args):  # quiet by default
            pass

        # ---------------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 5_000_000:
                raise RunError(413, {"message": "요청이 너무 큽니다"})
            raw = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except ValueError as exc:
                raise RunError(400, {"message": "JSON 형식이 아닙니다"}) from exc

        def _check_origin(self) -> None:
            # reject cross-site POSTs from other origins (the server is local only)
            origin = self.headers.get("Origin")
            if origin and not (origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost")):
                raise RunError(403, {"message": "다른 출처의 요청은 허용하지 않습니다"})

        # ---------------------------------------------------------------
        def do_GET(self):  # noqa: N802
            try:
                self._route_get()
            except RunError as exc:
                self._json(exc.status, exc.payload)
            except Exception as exc:  # pragma: no cover
                self._json(500, {"message": str(exc), "trace": traceback.format_exc(limit=5)})

        def do_POST(self):  # noqa: N802
            try:
                self._check_origin()
                self._route_post()
            except RunError as exc:
                self._json(exc.status, exc.payload)
            except Exception as exc:  # pragma: no cover
                self._json(500, {"message": str(exc), "trace": traceback.format_exc(limit=5)})

        def _route_get(self):
            u = urllib.parse.urlparse(self.path)
            path = u.path
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path == "/api/meta":
                return self._json(200, app_meta())
            if path == "/api/labs":
                return self._json(200, [lab.meta() for lab in sorted(all_labs().values(), key=lambda lb: (lb.track, lb.order))])
            if path.startswith("/api/labs/"):
                lab_id = path.split("/")[3]
                try:
                    return self._json(200, get_lab(lab_id).meta(full=True))
                except KeyError:
                    raise RunError(404, {"message": f"{lab_id}는 아직 구현되지 않았거나 공식 ID가 아닙니다"})
            if path == "/api/progress":
                return self._json(200, store.get())
            if path == "/textbook":
                tb = _textbook_path()
                if tb is None:
                    return self._send(404, "교재 HTML이 없습니다. textbook/ 폴더에 Infineon_FAE_Expert_Integrated.html을 두면 여기서 열립니다.".encode(), "text/plain; charset=utf-8")
                return self._send(200, tb.read_bytes(), "text/html; charset=utf-8")
            raise RunError(404, {"message": "없는 경로"})

        def _static(self, rel: str):
            rel = urllib.parse.unquote(rel)
            if ".." in rel or rel.startswith("/"):
                raise RunError(400, {"message": "bad path"})
            f = _web_root().joinpath(rel)
            if not f.is_file():
                raise RunError(404, {"message": f"{rel} 없음"})
            ctype = mimetypes.guess_type(rel)[0] or "application/octet-stream"
            if rel.endswith(".js"):
                ctype = "text/javascript; charset=utf-8"
            elif rel.endswith(".css"):
                ctype = "text/css; charset=utf-8"
            elif rel.endswith(".html"):
                ctype = "text/html; charset=utf-8"
            return self._send(200, f.read_bytes(), ctype)

        def _route_post(self):
            path = urllib.parse.urlparse(self.path).path
            body = self._body()
            if path == "/api/run":
                out = run_request(body.get("lab", ""), body.get("experiment", ""), body.get("preset"), body.get("values") or {})
                return self._json(200, out)
            if path == "/api/export":
                fmt = body.get("format", "json")
                out = run_request(body.get("lab", ""), body.get("experiment", ""), body.get("preset"), body.get("values") or {})
                name = f"{out['lab']}_{out['experiment']}_{(body.get('preset') or 'custom')}"
                if fmt == "csv":
                    return self._send(200, ("﻿" + to_csv(out)).encode("utf-8"), "text/csv; charset=utf-8", {"Content-Disposition": f'attachment; filename="{name}.csv"'})
                if fmt == "html":
                    lab = get_lab(out["lab"])
                    meta = {"lab": lab.meta(), "experiment": lab.experiment(out["experiment"]).meta()}
                    return self._send(200, to_html(out, meta).encode("utf-8"), "text/html; charset=utf-8", {"Content-Disposition": f'attachment; filename="{name}.html"'})
                return self._send(200, to_json(out).encode("utf-8"), "application/json; charset=utf-8", {"Content-Disposition": f'attachment; filename="{name}.json"'})
            if path == "/api/progress":
                slot = store.record(body.get("lab", ""), body.get("experiment", ""), body.get("kind", ""), body.get("payload") or {})
                return self._json(200, slot)
            raise RunError(404, {"message": "없는 경로"})

    return Handler


def serve(port: int = 8765, open_browser: bool = True, store: UserStore | None = None) -> None:
    store = store or UserStore()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"Converter FAE Lab: {url}  (Ctrl+C로 종료, 학습 기록: {store.path})")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("종료합니다.")
    finally:
        httpd.server_close()
