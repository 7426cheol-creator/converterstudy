// Rendering of one run: verdicts, circuit linked to the waveform cursor, metrics
// against hand calculations, plots, verification checks, tables, model card.
import { esc, fmtNum, fmtPct, fmtSI, h, valueAt, TONE_CLASS, SOURCE_KO } from "./fmt.js";
import { CursorBus, renderPlot } from "./plot.js";
import { renderCircuit } from "./circuit.js";

const CHECK_CLASS = { PASS: "tone-ok", FAIL: "tone-fail", INFO: "tone-neutral", NOT_RUN: "tone-blocked" };

export function statusChip(st) {
  return h("span", { class: "chip status " + (TONE_CLASS[st.tone] || ""), title: st.meaning || st.why || "" }, `${st.ko} · ${st.code}`);
}

export function renderResult(host, payload, opts = {}) {
  const res = payload.result;
  const bus = new CursorBus();
  const seriesByKey = {};
  const unitOf = {};
  for (const s of res.series) {
    seriesByKey[s.key] = s;
    unitOf[s.key] = s.unit;
  }
  const root = h("div", { class: "result" });
  host.append(root);

  // ---- verdicts ------------------------------------------------------------
  const verd = h("section", { class: "card verdicts" });
  verd.append(h("div", { class: "row wrap gap" }, h("span", { class: "k" }, "결과 판정"), statusChip(res.status), h("span", { class: "chip level" }, "모델 수준 " + res.model_level), h("span", { class: "chip " + (res.checks_ok ? "tone-ok" : "tone-fail") }, res.checks_ok ? "검증 항목 모두 통과" : "검증 실패 항목 있음"), h("span", { class: "chip tone-blocked", title: "하드웨어 측정으로 검증한 것이 아님" }, "HW 검증 없음")));
  const ul = h("ul", { class: "verdict-list" });
  for (const v of res.verdicts) ul.append(h("li", {}, h("span", { class: "chip small " + (TONE_CLASS[v.tone] || "") }, v.ko), " ", v.why));
  verd.append(ul);
  for (const w of res.warnings || []) verd.append(h("div", { class: "warn-line" }, "⚠ " + w));
  root.append(verd);

  // ---- circuit + interpretation ---------------------------------------------------
  let circuitCtl = null;
  if (res.circuit && res.circuit.diagram) {
    const cc = h("section", { class: "card circuit-card" });
    const dg = res.circuit.diagram;
    const intervals = res.circuit.intervals || [];
    cc.append(h("h3", {}, "회로와 도통 경로 ", h("span", { class: "muted small" }, intervals.length ? "(파형 커서와 연결)" : "(이 실험에는 시간 파형 구간이 없어 고정 그림)")));
    if (dg.title) cc.append(h("p", { class: "muted small circuit-title" }, dg.title));
    circuitCtl = renderCircuit(cc, dg, { linked: intervals.length > 0 });
    for (const n of dg.notes || []) cc.append(h("p", { class: "muted small" }, "※ " + n));
    root.append(cc);
    const probeSeries = (res.circuit.diagram.probes || []).map((p) => p.series);
    bus.on((group, x) => {
      if (res.circuit.plot_group && group !== res.circuit.plot_group) return;
      const iv = intervals.find((b) => x >= b.x0 && x <= b.x1);
      circuitCtl.setMode(iv ? iv.mode : null);
      const vals = {};
      for (const k of probeSeries) if (seriesByKey[k]) vals[k] = valueAt(seriesByKey[k].x, seriesByKey[k].y, x);
      circuitCtl.setProbes(vals, unitOf);
    });
  }
  if (res.interpretation) root.append(h("section", { class: "card interp" }, h("h3", {}, "왜 그런가 — 이번 결과의 해석"), h("p", {}, res.interpretation)));

  // ---- metrics ------------------------------------------------------------
  const prev = opts.previous ? Object.fromEntries(opts.previous.result.metrics.map((m) => [m.key, m])) : null;
  const hi = new Set(opts.highlight || []);
  const mt = h("table", { class: "metrics" });
  const head = ["항목", "값", "기준·범위", "손계산/기준값", "상대오차", "판정"];
  if (prev) head.push("이전 실행", "변화");
  mt.append(h("thead", {}, h("tr", {}, ...head.map((c) => h("th", {}, c)))));
  const tb = h("tbody");
  for (const m of res.metrics) {
    const row = h("tr", { class: hi.has(m.key) ? "hl" : "" });
    row.append(h("td", {}, m.label, m.note ? h("div", { class: "muted small" }, m.note) : null));
    // text values (e.g. "flux @ 고전압 경부하", a sentence about an unconfirmed datum) wrap; numbers stay on one line
    row.append(h("td", { class: typeof m.value === "string" ? "text-val" : "num" }, typeof m.value === "string" ? m.value : fmtSI(m.value, m.unit, 6)));
    row.append(h("td", { class: "muted small" }, m.basis || ""));
    row.append(h("td", { class: "num" }, m.ref === null ? "" : fmtSI(m.ref, m.unit, 6), m.ref_label ? h("div", { class: "muted small" }, m.ref_label) : null));
    row.append(h("td", { class: "num" }, m.rel_err === null ? "" : fmtPct(m.rel_err, 3), m.tol ? h("div", { class: "muted small" }, "허용 " + fmtPct(m.tol, 2)) : null));
    row.append(h("td", {}, m.check ? h("span", { class: "chip small " + (CHECK_CLASS[m.check] || "") }, m.check) : ""));
    if (prev) {
      const pm = prev[m.key];
      row.append(h("td", { class: "num muted" }, pm ? (typeof pm.value === "string" ? pm.value : fmtSI(pm.value, pm.unit, 5)) : ""));
      let ch = "";
      if (pm && typeof pm.value === "number" && typeof m.value === "number") {
        const d = m.value - pm.value;
        const r = pm.value !== 0 ? d / Math.abs(pm.value) : null;
        ch = Math.abs(d) < 1e-12 * Math.max(1, Math.abs(pm.value)) ? "=" : `${d > 0 ? "▲" : "▼"} ${r === null ? fmtSI(d, m.unit, 3) : fmtPct(r, 1)}`;
      } else if (pm && pm.value !== m.value) ch = `${pm.value} → ${m.value}`;
      row.append(h("td", { class: "num" }, ch));
    }
    tb.append(row);
  }
  mt.append(tb);
  root.append(h("section", { class: "card" }, h("h3", {}, "핵심 수치 (값 · 단위 · 기준)"), h("div", { class: "tablewrap" }, mt)));

  // ---- plots --------------------------------------------------------------
  const plotsCard = h("section", { class: "card plots" }, h("h3", {}, "파형과 그래프"));
  const grid = h("div", { class: "plot-grid" });
  plotsCard.append(grid);
  root.append(plotsCard);
  const plots = [];
  for (const p of res.plots) plots.push(renderPlot(grid, p, seriesByKey, { bus }));

  // ---- checks -------------------------------------------------------------
  if (res.checks.length) {
    const ct = h("table", { class: "checks" });
    ct.append(h("thead", {}, h("tr", {}, ...["검증 항목", "비교 경로", "독립", "값", "기준", "판정"].map((c) => h("th", {}, c)))));
    const cb = h("tbody");
    for (const c of res.checks) {
      cb.append(
        h(
          "tr",
          {},
          h("td", {}, c.name, h("div", { class: "muted small" }, c.detail || "")),
          h("td", { class: "small" }, c.path || ""),
          h("td", {}, c.independent ? "독립" : "회귀"),
          h("td", { class: typeof c.value === "number" ? "num" : "text-val" }, typeof c.value === "number" ? (c.unit === "rel" ? c.value.toExponential(2) : fmtSI(c.value, c.unit, 5)) : c.value ?? ""),
          h("td", { class: "num" }, c.threshold === null || c.threshold === undefined ? "" : typeof c.threshold === "number" ? c.threshold.toExponential(1) : String(c.threshold)),
          h("td", {}, h("span", { class: "chip small " + (CHECK_CLASS[c.status] || "") }, c.status))
        )
      );
    }
    ct.append(cb);
    root.append(h("section", { class: "card" }, h("h3", {}, "수치 검증 (verification — 하드웨어 validation 아님)"), h("div", { class: "tablewrap" }, ct)));
  }

  // ---- tables -------------------------------------------------------------
  for (const t of res.tables) {
    const tt = h("table", { class: "data" });
    tt.append(h("thead", {}, h("tr", {}, ...t.columns.map((c) => h("th", {}, c)))));
    const bb = h("tbody");
    for (const r of t.rows) bb.append(h("tr", {}, ...r.map((c) => h("td", { class: typeof c === "number" ? "num" : "" }, typeof c === "number" ? fmtNum(c, 6) : c ?? ""))));
    tt.append(bb);
    root.append(h("section", { class: "card" }, h("h3", {}, t.title), h("div", { class: "tablewrap" }, tt), t.note ? h("p", { class: "muted small" }, t.note) : null));
  }

  // ---- model card ------------------------------------------------------------
  const mc = h("section", { class: "card modelcard" }, h("h3", {}, "모델 카드: 가정과 적용 범위"));
  mc.append(h("div", { class: "cols2" }, h("div", {}, h("h4", {}, "가정 (assumptions)"), h("ul", {}, ...(res.assumptions || []).map((a) => h("li", {}, a)))), h("div", {}, h("h4", {}, "이 결과로 말하면 안 되는 것 (not_valid_for)"), h("ul", {}, ...(res.not_valid_for || []).map((a) => h("li", {}, a))))));
  if (payload.inputs) {
    const it = h("table", { class: "inputs" });
    it.append(h("thead", {}, h("tr", {}, ...["입력", "값", "출처"].map((c) => h("th", {}, c)))));
    const ib = h("tbody");
    for (const i of payload.inputs) ib.append(h("tr", { class: i.changed ? "hl" : "" }, h("td", {}, i.label), h("td", { class: "num" }, i.display), h("td", { class: "small" }, h("span", { class: "chip small src-" + i.source }, SOURCE_KO[i.source] || i.source), " ", i.source_note || "")));
    it.append(ib);
    mc.append(h("h4", {}, "입력값과 출처 (기본은 ASSUMED 합성 사양 · 실제 부품 데이터 아님)"), h("div", { class: "tablewrap" }, it));
  }
  if (payload.provenance) {
    const p = payload.provenance;
    mc.append(h("p", { class: "muted small" }, `코드 ${p.code_path} · ${p.function} · 요청 ${p.request_hash} · ${p.timestamp} · app ${p.app_version} · contract ${p.contract_version} · Python ${p.python} / numpy ${p.numpy} / scipy ${p.scipy} · ${fmtNum(payload.runtime_s, 3)} s${payload.cached ? " (캐시)" : ""}`));
  }
  root.append(mc);
  // prime the cursor so the circuit shows the first state
  const firstBand = res.plots.find((p) => (p.bands || []).length);
  if (firstBand && circuitCtl) {
    const b = firstBand.bands[0];
    bus.emit(firstBand.group || firstBand.key, (b.x0 + b.x1) / 2, null);
  }
  return { bus, plots, root };
}
