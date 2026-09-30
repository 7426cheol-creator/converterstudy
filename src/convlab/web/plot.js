// SVG plots with a shared cursor, switching-state bands, zoom and PNG export.
import { esc, fmtNum, fmtSI, h, logTicks, niceTicks, prefixFor, svgEl, tickLabel, valueAt, download } from "./fmt.js";

export class CursorBus {
  constructor() {
    this.subs = new Set();
    this.zoomSubs = new Set();
  }
  on(fn) {
    this.subs.add(fn);
    return () => this.subs.delete(fn);
  }
  emit(group, x, src) {
    for (const fn of this.subs) fn(group, x, src);
  }
  onZoom(fn) {
    this.zoomSubs.add(fn);
  }
  emitZoom(group, dom, src) {
    for (const fn of this.zoomSubs) fn(group, dom, src);
  }
}

const PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)", "var(--c8)"];

function hashMode(s) {
  let x = 0;
  for (const c of String(s)) x = (x * 31 + c.charCodeAt(0)) >>> 0;
  return x % 4;
}

const VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"];

function lerpColor(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  const c = (k) => Math.round(parseInt(a.slice(k, k + 2), 16) * (1 - f) + parseInt(b.slice(k, k + 2), 16) * f);
  return `rgb(${c(1)},${c(3)},${c(5)})`;
}

function edges(centres) {
  const n = centres.length;
  if (n === 1) return [centres[0] - 0.5, centres[0] + 0.5];
  const e = [centres[0] - (centres[1] - centres[0]) / 2];
  for (let i = 0; i < n - 1; i++) e.push((centres[i] + centres[i + 1]) / 2);
  e.push(centres[n - 1] + (centres[n - 1] - centres[n - 2]) / 2);
  return e;
}

// Marker labels: every marker circle is an obstacle; each label takes the first offset that stays inside the
// frame and overlaps neither a circle nor an earlier label (width estimated per glyph: Hangul/symbols wider).
function placeLabels(parent, items, frame, r, circleClass, textClass, wAscii, wWide) {
  const boxes = items.map((it) => ({ x: it.cx - r - 1, y: it.cy - r - 1, w: 2 * r + 2, h: 2 * r + 2 }));
  const textW = (s) => [...s].reduce((a, ch) => a + (ch.codePointAt(0) > 0x2000 ? wWide : wAscii), 0);
  for (const it of items) {
    const w = textW(it.text), hgt = 13;
    const tries = [[r + 3, -r - 2, "start"], [r + 3, r + 11, "start"], [-r - 3, -r - 2, "end"], [-r - 3, r + 11, "end"], [r + 3, -r - 16, "start"], [-r - 3, -r - 16, "end"], [r + 3, r + 25, "start"], [-r - 3, r + 25, "end"], [0, -r - 6, "middle"], [0, r + 16, "middle"]];
    const boxOf = (tr) => ({ x: tr[2] === "end" ? it.cx + tr[0] - w : tr[2] === "middle" ? it.cx - w / 2 : it.cx + tr[0], y: it.cy + tr[1] - 10, w, h: hgt });
    const ok = (b) => b.x >= frame.l - 2 && b.x + b.w <= frame.l + frame.w + 2 && b.y >= frame.t - 2 && b.y + b.h <= frame.t + frame.h + 2 &&
      !boxes.some((o) => b.x < o.x + o.w && b.x + b.w > o.x && b.y < o.y + o.h && b.y + b.h > o.y);
    const pick = tries.find((tr) => ok(boxOf(tr))) || tries[0];
    boxes.push(boxOf(pick));
    parent.append(svgEl("circle", { cx: it.cx, cy: it.cy, r, class: circleClass }));
    const t = svgEl("text", { x: it.cx + pick[0], y: it.cy + pick[1], "text-anchor": pick[2], class: textClass });
    t.textContent = it.text;
    parent.append(t);
  }
}

// Markers with a short in-plot tag are spelled out under the plot, with their coordinates.
function markerLegend(markers, spec = {}) {
  const lg = h("div", { class: "plot-legend map-legend" });
  for (const mk of markers) {
    const coord = spec.kind === "map" ? `(${fmtNum(mk.x, 3)}, ${fmtNum(mk.y, 3)})` : "";
    lg.append(h("span", { class: "lg-item" }, h("b", {}, mk.short || mk.label), mk.short ? ` ${mk.label} ` : " ", h("span", { class: "muted" }, coord)));
  }
  return lg;
}

// Heatmap: one series with z[iy][ix]; null cells are infeasible and show their note.
export function renderMap(host, spec, seriesByKey) {
  const s = seriesByKey[spec.series[0]];
  const wrap = h("figure", { class: "plot map", "data-plot": spec.key });
  const tools = h("div", { class: "plot-tools" });
  if (spec.level) tools.append(h("span", { class: "chip level" }, "모델 " + spec.level));
  const btnPng = h("button", { class: "mini" }, "PNG");
  tools.append(btnPng);
  wrap.append(h("div", { class: "plot-head" }, h("figcaption", { class: "plot-title" }, spec.title), tools));
  const svgHost = h("div", { class: "plot-svg" });
  const readout = h("div", { class: "plot-readout" }, "셀 위에 마우스를 올리면 값과 판정 이유가 표시됩니다");
  wrap.append(svgHost);
  if ((spec.markers || []).some((mk) => mk.short)) wrap.append(markerLegend(spec.markers, spec));
  wrap.append(readout);
  if (spec.proved || spec.not_yet) {
    const note = h("div", { class: "plot-note" });
    if (spec.proved) note.append(h("p", {}, h("b", {}, "입증한 것: "), spec.proved));
    if (spec.not_yet) note.append(h("p", {}, h("b", {}, "아직 아닌 것: "), spec.not_yet));
    wrap.append(note);
  }
  host.append(wrap);
  if (!s || !s.z) {
    svgHost.textContent = "데이터 없음";
    return { el: wrap };
  }
  let zmin = Infinity, zmax = -Infinity;
  for (const row of s.z) for (const v of row) if (v !== null && isFinite(v)) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); }
  if (!isFinite(zmin)) { zmin = 0; zmax = 1; }
  if (zmax === zmin) zmax = zmin + Math.abs(zmin || 1) * 1e-6;
  const ex = edges(s.x), ey = edges(s.y);
  function draw() {
    svgHost.innerHTML = "";
    const W = Math.max(320, svgHost.clientWidth || 560);
    const H = opts_h(W);
    const m = { l: 62, r: 76, t: 10, b: 40 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const X = (v) => m.l + ((v - ex[0]) / (ex[ex.length - 1] - ex[0])) * iw;
    const Y = (v) => m.t + ih - ((v - ey[0]) / (ey[ey.length - 1] - ey[0])) * ih;
    const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "plot-canvas", role: "img", "aria-label": spec.title });
    const defs = svgEl("defs");
    const pat = svgEl("pattern", { id: "hatch-" + spec.key, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pat.append(svgEl("rect", { width: 6, height: 6, class: "hatch-bg" }), svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "hatch-line" }));
    defs.append(pat);
    svg.append(defs);
    for (let iy = 0; iy < s.y.length; iy++) for (let ix = 0; ix < s.x.length; ix++) {
      const v = s.z[iy][ix];
      const x0 = X(ex[ix]), x1 = X(ex[ix + 1]), y0 = Y(ey[iy + 1]), y1 = Y(ey[iy]);
      const r = svgEl("rect", { x: x0, y: y0, width: Math.max(0, x1 - x0 + 0.3), height: Math.max(0, y1 - y0 + 0.3), fill: v === null || !isFinite(v) ? `url(#hatch-${spec.key})` : lerpColor((v - zmin) / (zmax - zmin)), class: "cell" });
      r.addEventListener("mousemove", () => {
        const note = s.notes && s.notes[iy] ? s.notes[iy][ix] : "";
        readout.textContent = `${spec.x_label} = ${fmtSI(s.x[ix], spec.x_unit)} · ${spec.y_label} = ${fmtSI(s.y[iy], spec.y_unit)} · ${s.label}: ${v === null ? "해 없음/제외" : fmtSI(v, s.unit)}${note ? " · " + note : ""}`;
      });
      svg.append(r);
    }
    const gx = svgEl("g", { class: "grid" });
    for (const v of niceTicks(ex[0], ex[ex.length - 1], 6)) {
      const t = svgEl("text", { x: X(v), y: m.t + ih + 15, "text-anchor": "middle", class: "tick" });
      t.textContent = fmtNum(v, 3);
      if (X(v) >= m.l - 1 && X(v) <= m.l + iw + 1) gx.append(t);
    }
    for (const v of niceTicks(ey[0], ey[ey.length - 1], 6)) {
      const t = svgEl("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "tick" });
      t.textContent = fmtNum(v, 3);
      if (Y(v) >= m.t - 1 && Y(v) <= m.t + ih + 1) gx.append(t);
    }
    svg.append(gx);
    svg.append(svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "frame" }));
    const xl = svgEl("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "axis-label" });
    xl.textContent = `${spec.x_label}${spec.x_unit ? ` [${spec.x_unit}]` : ""}`;
    const yl = svgEl("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", class: "axis-label", transform: `rotate(-90 12 ${m.t + ih / 2})` });
    yl.textContent = `${spec.y_label}${spec.y_unit ? ` [${spec.y_unit}]` : ""}`;
    svg.append(xl, yl);
    placeLabels(svg, (spec.markers || []).map((mk) => ({ cx: X(mk.x), cy: Y(mk.y), text: mk.short || mk.label || "" })), { l: m.l, t: m.t, w: iw, h: ih }, 6, "map-marker", "map-label", 6.8, 11.2);
    // colour bar
    const cbx = m.l + iw + 16, cbw = 12;
    for (let k = 0; k < 40; k++) svg.append(svgEl("rect", { x: cbx, y: m.t + (ih * k) / 40, width: cbw, height: ih / 40 + 0.5, fill: lerpColor(1 - k / 39) }));
    const [cs, cp] = prefixFor(Math.max(Math.abs(zmin), Math.abs(zmax)), s.unit);
    for (const [v, y] of [[zmax, m.t + 8], [zmin, m.t + ih]]) {
      const t = svgEl("text", { x: cbx + cbw + 3, y, class: "tick" });
      t.textContent = fmtNum(v / cs, 4);
      svg.append(t);
    }
    const ul = svgEl("text", { x: cbx, y: m.t + ih + 15, class: "tick" });
    ul.textContent = `[${cp}${s.unit}]`;
    svg.append(ul);
    svgHost.append(svg);
  }
  function opts_h(W) {
    return Math.min(360, Math.max(240, W * 0.55));
  }
  btnPng.addEventListener("click", () => exportPng(svgHost.querySelector("svg"), spec.key + ".png"));
  let rt;
  new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(draw, 60); }).observe(svgHost);
  draw();
  return { el: wrap };
}

export function renderPlot(host, spec, seriesByKey, opts = {}) {
  if (spec.kind === "map") return renderMap(host, spec, seriesByKey, opts);
  const bus = opts.bus;
  const group = spec.group || spec.key;
  const series = spec.series.map((k) => seriesByKey[k]).filter(Boolean);
  const hidden = new Set();
  const wrap = h("figure", { class: "plot", "data-plot": spec.key });
  const head = h("div", { class: "plot-head" });
  const title = h("figcaption", { class: "plot-title" }, spec.title);
  const tools = h("div", { class: "plot-tools" });
  if (spec.level) tools.append(h("span", { class: "chip level", title: "모델 수준" }, "모델 " + spec.level));
  const btnReset = h("button", { class: "mini", title: "확대 초기화 (더블클릭도 가능)" }, "전체");
  const btnPng = h("button", { class: "mini", title: "PNG로 저장" }, "PNG");
  const btnCsv = h("button", { class: "mini", title: "이 그래프의 데이터를 CSV로" }, "CSV");
  tools.append(btnReset, btnPng, btnCsv);
  head.append(title, tools);
  const svgHost = h("div", { class: "plot-svg" });
  const legend = h("div", { class: "plot-legend" });
  const readout = h("div", { class: "plot-readout" }, " ");
  wrap.append(head, svgHost, legend);
  if ((spec.markers || []).some((mk) => mk.short)) wrap.append(markerLegend(spec.markers));
  wrap.append(readout);
  if (spec.proved || spec.not_yet) {
    const note = h("div", { class: "plot-note" });
    if (spec.proved) note.append(h("p", {}, h("b", {}, "입증한 것: "), spec.proved));
    if (spec.not_yet) note.append(h("p", {}, h("b", {}, "아직 아닌 것: "), spec.not_yet));
    wrap.append(note);
  }
  host.append(wrap);

  // ---- data ranges --------------------------------------------------------
  const isTime = spec.kind === "time";
  let xmin = Infinity, xmax = -Infinity;
  for (const s of series) for (const x of s.x) if (x !== null && isFinite(x)) { if (x < xmin) xmin = x; if (x > xmax) xmax = x; }
  for (const v of spec.vlines || []) if (isFinite(v.x) && !spec.log_x) { xmin = Math.min(xmin, v.x); xmax = Math.max(xmax, v.x); }
  if (!isFinite(xmin)) { xmin = 0; xmax = 1; }
  const full = [xmin, xmax];
  let dom = [xmin, xmax];
  let cursorX = null;

  const xUnit = spec.x_unit || "";
  const yUnit = spec.y_unit || (series[0] && series[0].unit) || "";

  function yRange() {
    let lo = Infinity, hi = -Infinity;
    for (const s of series) {
      if (hidden.has(s.key)) continue;
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], y = s.y[i];
        if (y === null || !isFinite(y)) continue;
        if (x < dom[0] || x > dom[1]) continue;
        if (spec.log_y && y <= 0) continue;
        if (y < lo) lo = y;
        if (y > hi) hi = y;
      }
    }
    for (const l of spec.hlines || []) if (isFinite(l.y) && (!spec.log_y || l.y > 0)) { lo = Math.min(lo, l.y); hi = Math.max(hi, l.y); }
    if (!isFinite(lo)) { lo = spec.log_y ? 1e-6 : -1; hi = spec.log_y ? 1 : 1; }
    if (spec.log_y) return [lo / 1.5, hi * 1.5];
    if (hi === lo) { const d = Math.abs(hi) * 0.05 || 1; return [lo - d, hi + d]; }
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  }

  function draw() {
    svgHost.innerHTML = "";
    const W = Math.max(320, svgHost.clientWidth || host.clientWidth || 640);
    const H = opts.height || 230;
    const m = { l: 62, r: 14, t: 10, b: 38 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "plot-canvas", role: "img", "aria-label": spec.title });
    const [ylo, yhi] = yRange();
    const lx = (v) => (spec.log_x ? Math.log10(v) : v);
    const ly = (v) => (spec.log_y ? Math.log10(v) : v);
    const X = (v) => m.l + ((lx(v) - lx(dom[0])) / (lx(dom[1]) - lx(dom[0]) || 1)) * iw;
    const Y = (v) => m.t + ih - ((ly(v) - ly(ylo)) / (ly(yhi) - ly(ylo) || 1)) * ih;
    const invX = (px) => {
      const t = lx(dom[0]) + ((px - m.l) / iw) * (lx(dom[1]) - lx(dom[0]));
      return spec.log_x ? Math.pow(10, t) : t;
    };
    const [xs, xp] = spec.log_x ? [1, ""] : prefixFor(Math.max(Math.abs(dom[0]), Math.abs(dom[1])), xUnit);
    const [ys, yp] = spec.log_y ? [1, ""] : prefixFor(Math.max(Math.abs(ylo), Math.abs(yhi)), yUnit);
    const clipId = "clip-" + spec.key + "-" + Math.random().toString(36).slice(2, 7);
    const defs = svgEl("defs");
    const cp = svgEl("clipPath", { id: clipId });
    cp.append(svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih }));
    defs.append(cp);
    svg.append(defs);
    // bands (switching states)
    const gb = svgEl("g", { "clip-path": `url(#${clipId})`, class: "bands" });
    for (const b of spec.bands || []) {
      if (b.x1 < dom[0] || b.x0 > dom[1]) continue;
      const x0 = X(Math.max(b.x0, dom[0])), x1 = X(Math.min(b.x1, dom[1]));
      const r = svgEl("rect", { x: x0, y: m.t, width: Math.max(0, x1 - x0), height: ih, class: "band band-" + hashMode(b.mode) });
      r.append(svgEl("title"));
      r.lastChild.textContent = b.label;
      gb.append(r);
      if (x1 - x0 > 44) {
        const t = svgEl("text", { x: (x0 + x1) / 2, y: m.t + 11, class: "band-label", "text-anchor": "middle" });
        t.textContent = b.label;
        gb.append(t);
      }
    }
    svg.append(gb);
    // grid and axes
    const gx = svgEl("g", { class: "grid" });
    const xt = spec.log_x ? logTicks(dom[0], dom[1]) : niceTicks(dom[0], dom[1], Math.max(3, Math.floor(iw / 90)));
    for (const v of xt) {
      const px = X(v);
      if (px < m.l - 0.5 || px > m.l + iw + 0.5) continue;
      gx.append(svgEl("line", { x1: px, x2: px, y1: m.t, y2: m.t + ih }));
      const t = svgEl("text", { x: px, y: m.t + ih + 15, "text-anchor": "middle", class: "tick" });
      t.textContent = spec.log_x ? fmtSI(v, "", 3).replace(/ /g, "") : tickLabel(v, xt.length > 1 ? xt[1] - xt[0] : v, xs);
      gx.append(t);
    }
    const yt = spec.log_y ? logTicks(ylo, yhi) : niceTicks(ylo, yhi, Math.max(3, Math.floor(ih / 40)));
    for (const v of yt) {
      const py = Y(v);
      if (py < m.t - 0.5 || py > m.t + ih + 0.5) continue;
      gx.append(svgEl("line", { x1: m.l, x2: m.l + iw, y1: py, y2: py }));
      const t = svgEl("text", { x: m.l - 6, y: py + 4, "text-anchor": "end", class: "tick" });
      t.textContent = spec.log_y ? v.toExponential(0) : tickLabel(v, yt.length > 1 ? yt[1] - yt[0] : v, ys);
      gx.append(t);
    }
    svg.append(gx);
    svg.append(svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "frame" }));
    const xl = svgEl("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "axis-label" });
    xl.textContent = `${spec.x_label || "t"}${xUnit || xp ? ` [${xp}${xUnit}]` : ""}`;
    svg.append(xl);
    const yl = svgEl("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", class: "axis-label", transform: `rotate(-90 12 ${m.t + ih / 2})` });
    yl.textContent = `${spec.y_label || ""}${yUnit || yp ? ` [${yp}${yUnit}]` : ""}`;
    svg.append(yl);
    // reference lines
    const gl = svgEl("g", { "clip-path": `url(#${clipId})`, class: "reflines" });
    for (const l of spec.hlines || []) {
      if (!isFinite(l.y) || (spec.log_y && l.y <= 0)) continue;
      const py = Y(l.y);
      gl.append(svgEl("line", { x1: m.l, x2: m.l + iw, y1: py, y2: py, class: "hline" }));
      const t = svgEl("text", { x: m.l + iw - 4, y: py - 3, "text-anchor": "end", class: "ref-label" });
      t.textContent = l.label;
      gl.append(t);
    }
    for (const l of spec.vlines || []) {
      if (!isFinite(l.x) || l.x < dom[0] || l.x > dom[1]) continue;
      const px = X(l.x);
      gl.append(svgEl("line", { x1: px, x2: px, y1: m.t, y2: m.t + ih, class: "vline" }));
      const t = svgEl("text", { x: px + 3, y: m.t + ih - 5, class: "ref-label" });
      t.textContent = l.label;
      gl.append(t);
    }
    if (spec.window) {
      const [a, b] = spec.window;
      if (b >= dom[0] && a <= dom[1]) {
        const x0 = X(Math.max(a, dom[0])), x1 = X(Math.min(b, dom[1]));
        gl.append(svgEl("line", { x1: x0, x2: x1, y1: m.t + ih - 1, y2: m.t + ih - 1, class: "window-bar" }));
        const t = svgEl("text", { x: x0 + 2, y: m.t + ih - 5, class: "ref-label" });
        t.textContent = "해석 구간";
        gl.append(t);
      }
    }
    svg.append(gl);
    // series
    const gs = svgEl("g", { "clip-path": `url(#${clipId})` });
    series.forEach((s, i) => {
      if (hidden.has(s.key)) return;
      const color = s.color || PALETTE[i % PALETTE.length];
      if (s.style === "points") {
        for (let k = 0; k < s.x.length; k++) {
          const x = s.x[k], y = s.y[k];
          if (y === null || x < dom[0] || x > dom[1] || (spec.log_y && y <= 0) || (spec.log_x && x <= 0)) continue;
          gs.append(svgEl("circle", { cx: X(x), cy: Y(y), r: 2.6, fill: color, class: "pt" }));
        }
        return;
      }
      let d = "";
      let pen = false;
      for (let k = 0; k < s.x.length; k++) {
        const x = s.x[k], y = s.y[k];
        if (y === null || !isFinite(y) || (spec.log_y && y <= 0) || (spec.log_x && x <= 0)) { pen = false; continue; }
        const px = X(x), py = Y(y);
        d += (pen ? "L" : "M") + px.toFixed(2) + "," + py.toFixed(2);
        pen = true;
      }
      gs.append(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": s.dash ? 1.6 : 1.7, "stroke-dasharray": s.dash ? "6 4" : null, "vector-effect": "non-scaling-stroke", class: "series" }));
    });
    // markers: circles are obstacles; each label (its short tag if given) takes the first free offset
    const vis = (spec.markers || []).filter((mk) => isFinite(mk.x) && isFinite(mk.y) && mk.x >= dom[0] && mk.x <= dom[1]);
    placeLabels(gs, vis.map((mk) => ({ cx: X(mk.x), cy: Y(mk.y), text: mk.short || mk.label || "" })), { l: m.l, t: m.t, w: iw, h: ih }, 4.5, "marker", "ref-label", 6.2, 10.2);
    svg.append(gs);
    // cursor + zoom interaction
    const cur = svgEl("line", { x1: 0, x2: 0, y1: m.t, y2: m.t + ih, class: "cursor", visibility: "hidden" });
    const sel = svgEl("rect", { x: 0, y: m.t, width: 0, height: ih, class: "zoomsel", visibility: "hidden" });
    const hit = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "hit" });
    svg.append(cur, sel, hit);
    let dragging = null;
    hit.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) * W) / r.width;
      const x = invX(px);
      if (dragging !== null) {
        const a = Math.min(dragging, px), b = Math.max(dragging, px);
        sel.setAttribute("x", a);
        sel.setAttribute("width", b - a);
        sel.setAttribute("visibility", "visible");
      }
      if (bus) bus.emit(group, x, wrap);
      else setCursor(x);
    });
    hit.addEventListener("mouseleave", () => {
      if (dragging === null && !bus) setCursor(null);
    });
    hit.addEventListener("mousedown", (ev) => {
      const r = svg.getBoundingClientRect();
      dragging = ((ev.clientX - r.left) * W) / r.width;
    });
    window.addEventListener("mouseup", (ev) => {
      if (dragging === null) return;
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) * W) / r.width;
      const a = Math.min(dragging, px), b = Math.max(dragging, px);
      dragging = null;
      sel.setAttribute("visibility", "hidden");
      if (b - a > 6) {
        const nd = [invX(a), invX(b)];
        if (bus && isTime) bus.emitZoom(group, nd, wrap);
        else setDomain(nd);
      }
    }, { once: false });
    hit.addEventListener("dblclick", () => {
      if (bus && isTime) bus.emitZoom(group, full, wrap);
      else setDomain(full);
    });
    svg._cursor = { cur, X, dom: () => dom, m, ih };
    svgHost.append(svg);
    placeCursor();
  }

  function placeCursor() {
    const svg = svgHost.querySelector("svg");
    if (!svg || !svg._cursor) return;
    const { cur, X } = svg._cursor;
    if (cursorX === null || cursorX < dom[0] || cursorX > dom[1]) {
      cur.setAttribute("visibility", "hidden");
    } else {
      const px = X(cursorX);
      cur.setAttribute("x1", px);
      cur.setAttribute("x2", px);
      cur.setAttribute("visibility", "visible");
    }
    updateReadout();
  }

  function updateReadout() {
    if (cursorX === null) {
      readout.textContent = "그래프 위에 마우스를 올리면 값이 표시됩니다 · 드래그=확대, 더블클릭=전체";
      return;
    }
    const parts = [];
    parts.push(`${spec.x_label || "x"} = ${spec.log_x ? fmtSI(cursorX, xUnit) : fmtSI(cursorX, xUnit)}`);
    for (const s of series) {
      if (hidden.has(s.key)) continue;
      const v = s.style === "points" ? nearestPoint(s, cursorX) : valueAt(s.x, s.y, cursorX);
      parts.push(`${s.label}: ${v === null ? "—" : fmtSI(v, s.unit)}`);
    }
    const band = (spec.bands || []).find((b) => cursorX >= b.x0 && cursorX <= b.x1);
    readout.innerHTML = (band ? `<span class="chip mode">${esc(band.label)}</span> ` : "") + parts.map(esc).join(" · ");
  }

  function nearestPoint(s, x) {
    let best = null, bd = Infinity;
    for (let i = 0; i < s.x.length; i++) {
      const d = Math.abs(s.x[i] - x);
      if (d < bd) { bd = d; best = s.y[i]; }
    }
    const span = dom[1] - dom[0];
    return bd < span * 0.03 ? best : null;
  }

  function setCursor(x) {
    cursorX = x;
    placeCursor();
  }
  function setDomain(d) {
    dom = [Math.min(d[0], d[1]), Math.max(d[0], d[1])];
    draw();
  }

  // legend
  series.forEach((s, i) => {
    const color = s.color || PALETTE[i % PALETTE.length];
    const item = h("button", { class: "legend-item", title: "표시/숨김" }, h("span", { class: "swatch" + (s.dash ? " dash" : "") + (s.style === "points" ? " dot" : ""), style: `--sw:${color}` }), s.label + (s.unit ? ` [${s.unit}]` : ""));
    item.addEventListener("click", () => {
      if (hidden.has(s.key)) hidden.delete(s.key);
      else hidden.add(s.key);
      item.classList.toggle("off", hidden.has(s.key));
      draw();
    });
    legend.append(item);
  });

  btnReset.addEventListener("click", () => (bus && isTime ? bus.emitZoom(group, full, wrap) : setDomain(full)));
  btnPng.addEventListener("click", () => exportPng(svgHost.querySelector("svg"), spec.key + ".png"));
  btnCsv.addEventListener("click", () => {
    let txt = "series,label,unit,x,y\n";
    for (const s of series) for (let i = 0; i < s.x.length; i++) txt += `${s.key},"${s.label}",${s.unit},${s.x[i]},${s.y[i]}\n`;
    download(spec.key + ".csv", "﻿" + txt, "text/csv");
  });

  if (bus) {
    bus.on((g, x) => {
      if (g === group) setCursor(x);
    });
    bus.onZoom((g, d) => {
      if (g === group && isTime) setDomain(d);
    });
  }
  let rt;
  const ro = new ResizeObserver(() => {
    clearTimeout(rt);
    rt = setTimeout(draw, 60);
  });
  ro.observe(svgHost);
  draw();
  return { setCursor, setDomain, redraw: draw, el: wrap, group };
}

export function exportPng(svg, name) {
  if (!svg) return;
  const clone = svg.cloneNode(true);
  const cs = getComputedStyle(document.body);
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  const vars = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8", "--fg", "--muted", "--grid", "--bg"].map((v) => `${v}:${cs.getPropertyValue(v)};`).join("");
  style.textContent = `svg{${vars}font-family:${cs.fontFamily};} .grid line{stroke:${cs.getPropertyValue("--grid")};} .tick,.axis-label,.ref-label,.band-label{fill:${cs.getPropertyValue("--fg")};font-size:11px;} .frame{fill:none;stroke:${cs.getPropertyValue("--muted")};} .hit,.cursor,.zoomsel{display:none} .band{opacity:.12} .band-0{fill:#3b82f6}.band-1{fill:#f59e0b}.band-2{fill:#10b981}.band-3{fill:#a855f7} .hline,.vline{stroke:${cs.getPropertyValue("--muted")};stroke-dasharray:4 3}`;
  clone.prepend(style);
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("width", "100%");
  bg.setAttribute("height", "100%");
  bg.setAttribute("fill", cs.getPropertyValue("--bg") || "#fff");
  clone.insertBefore(bg, clone.firstChild.nextSibling);
  const data = new XMLSerializer().serializeToString(clone);
  const img = new Image();
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = W * 2;
    c.height = H * 2;
    const ctx = c.getContext("2d");
    ctx.scale(2, 2);
    ctx.drawImage(img, 0, 0);
    c.toBlob((b) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = name;
      a.click();
    });
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(data);
}
