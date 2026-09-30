// Formatting and small DOM helpers (shared by the app and exported reports).

export const NO_PREFIX = new Set(["%", "", "rad", "deg", "°C", "K", "1/K", "K/W", "rpm", "Nm", "mm²", "A/mm²", "dB", "rel", "주기", "cycles"]);
const PREFIXES = [[1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"], [1e-15, "f"]];

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function prefixFor(maxAbs, unit) {
  if (NO_PREFIX.has(unit) || !isFinite(maxAbs) || maxAbs === 0) return [1, ""];
  for (const [s, p] of PREFIXES) if (maxAbs >= s * 0.9999999) return [s, p];
  return [1e-15, "f"];
}

export function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "string") return v;
  if (!isFinite(v)) return v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a >= 1e7)) return v.toExponential(Math.max(1, digits - 1)).replace("e", "e");
  return Number(v.toPrecision(digits)).toString();
}

export function fmtSI(v, unit, digits = 4) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v + (unit ? " " + unit : "");
  if (typeof v === "boolean") return v ? "예" : "아니오";
  if (!isFinite(v)) return "—";
  if (NO_PREFIX.has(unit) || v === 0) return `${fmtNum(v, digits)}${unit ? " " + unit : ""}`;
  const [s, p] = prefixFor(Math.abs(v), unit);
  return `${fmtNum(v / s, digits)} ${p}${unit}`;
}

export function fmtPct(v, digits = 2) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const p = v * 100;
  if (Math.abs(p) < 1e-3 && p !== 0) return p.toExponential(1) + " %";
  return p.toFixed(digits) + " %";
}

export function niceTicks(lo, hi, n = 6) {
  if (!(hi > lo)) {
    const d = Math.abs(lo) || 1;
    lo -= d * 0.5;
    hi += d * 0.5;
  }
  const span = hi - lo;
  const raw = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const start = Math.ceil(lo / step - 1e-9) * step;
  const out = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function logTicks(lo, hi) {
  const out = [];
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const decades = b - a;
  const mults = decades <= 3 ? [1, 2, 5] : [1];
  const every = decades > 12 ? 3 : decades > 7 ? 2 : 1;
  for (let e = a; e <= b; e++) {
    if (decades > 7 && (e - a) % every) continue;
    for (const m of mults) {
      const v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out;
}

// Tick label with just enough decimals for the tick step (after SI scaling).
export function tickLabel(v, step, scale = 1) {
  const x = v / scale;
  const st = Math.abs(step / scale);
  if (!isFinite(st) || st === 0) return fmtNum(x, 4);
  const dec = Math.max(0, Math.min(8, -Math.floor(Math.log10(st) + 1e-9)));
  if (Math.abs(x) >= 1e6 || (Math.abs(x) < 1e-4 && x !== 0)) return x.toExponential(2);
  return x.toFixed(dec);
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  return el;
}

// Value of a sampled series at x: linear interpolation, right-continuous at jumps (duplicate x).
export function valueAt(xs, ys, x) {
  const n = xs.length;
  if (!n) return null;
  if (x < xs[0] || x > xs[n - 1]) return null;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] <= x) lo = m;
    else hi = m;
  }
  while (lo + 1 < n && xs[lo + 1] === x) lo++;
  const x0 = xs[lo], y0 = ys[lo];
  if (lo + 1 >= n) return y0;
  const x1 = xs[lo + 1], y1 = ys[lo + 1];
  if (y0 === null || y1 === null) return null;
  if (x1 === x0) return y1;
  return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
}

export const TONE_CLASS = { ok: "tone-ok", info: "tone-info", warn: "tone-warn", fail: "tone-fail", blocked: "tone-blocked", neutral: "tone-neutral" };

export const SOURCE_KO = {
  TEXTBOOK: "교재 값",
  ASSUMED: "가정 (교재 값 없음)",
  DERIVED: "유도값",
  DATASHEET: "데이터시트",
  MISSING_INPUT: "입력 없음",
};

export function download(name, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 500);
}
