// Circuit diagrams drawn from the lab's declarative description.
// Terminal a of every two-terminal element is its reference-positive end
// (anode, drain, +, dot).  Modes light up the conducting path; probes show the
// sign and value of a current at the waveform cursor.
import { esc, fmtSI, svgEl } from "./fmt.js";

const HALF = 30;

function sym(g, type, el) {
  const P = (d, cls = "sym") => g.append(svgEl("path", { d, class: cls }));
  switch (type) {
    case "resistor":
      P("M-30,0 H-15 l2.5,-6 l5,12 l5,-12 l5,12 l5,-12 l5,12 l2.5,-6 H30");
      break;
    case "inductor":
      P("M-30,0 H-20 a5,5 0 0 1 10,0 a5,5 0 0 1 10,0 a5,5 0 0 1 10,0 a5,5 0 0 1 10,0 H30");
      break;
    case "capacitor":
      P("M-30,0 H-4 M4,0 H30");
      P("M-4,-12 V12 M4,-12 V12", "sym plate");
      g.append(Object.assign(svgEl("text", { x: -14, y: -9, class: "pol" }), { textContent: "+" }));
      break;
    case "diode":
      P("M-30,0 H-9 M9,0 H30");
      P("M-9,-9 L9,0 L-9,9 Z", "sym solid");
      P("M9,-9 V9", "sym plate");
      break;
    case "vsource":
      P("M-30,0 H-14 M14,0 H30");
      g.append(svgEl("circle", { cx: 0, cy: 0, r: 14, class: "sym" }));
      g.append(Object.assign(svgEl("text", { x: -8, y: 4, class: "pol", "text-anchor": "middle" }), { textContent: "+" }));
      g.append(Object.assign(svgEl("text", { x: 8, y: 4, class: "pol", "text-anchor": "middle" }), { textContent: "−" }));
      break;
    case "battery":
      P("M-30,0 H-4 M4,0 H30");
      P("M-4,-14 V14", "sym plate");
      P("M4,-8 V8", "sym plate thick");
      g.append(Object.assign(svgEl("text", { x: -12, y: -10, class: "pol" }), { textContent: "+" }));
      break;
    case "isource":
      P("M-30,0 H-14 M14,0 H30");
      g.append(svgEl("circle", { cx: 0, cy: 0, r: 14, class: "sym" }));
      P("M-8,0 H6 M2,-4 L7,0 L2,4", "sym");
      break;
    case "nmos":
      // drain at -30, source at +30 along the axis, gate towards +y (local)
      P("M-30,0 H-10 V7 M30,0 H10 V7 M-14,7 H14");
      P("M-10,13 H10 M0,13 V26", "sym gate");
      P("M4,7 l-4,-3 v6 z", "sym solid");
      // body diode (source -> drain)
      P("M-10,0 V-10 H-3 M3,-10 H10 V0", "sym thin");
      P("M3,-15 V-5 M3,-10 L-3,-15 L-3,-5 Z", "sym thin solid");
      break;
    case "switch":
      P("M-30,0 H-10 M10,0 H30");
      P("M-10,0 L9,-9", "sym");
      g.append(svgEl("circle", { cx: -10, cy: 0, r: 2, class: "sym solid" }));
      g.append(svgEl("circle", { cx: 10, cy: 0, r: 2, class: "sym solid" }));
      break;
    case "block":
      g.append(svgEl("rect", { x: -(el.w || 60) / 2, y: -(el.h || 40) / 2, width: el.w || 60, height: el.h || 40, rx: 4, class: "sym blockbox" }));
      break;
    default:
      P("M-30,0 H30");
  }
}

export function renderCircuit(host, diagram) {
  const W = diagram.width || 640, H = diagram.height || 320;
  const wrap = document.createElement("div");
  wrap.className = "circuit";
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "circuit-svg", role: "img", "aria-label": diagram.title || diagram.id });
  const nodes = {};
  // wires
  for (const w of diagram.wires || []) {
    const d = w.pts.map((p, i) => (i ? "L" : "M") + p[0] + "," + p[1]).join("");
    const path = svgEl("path", { d, class: "wire", "data-id": w.id });
    svg.append(path);
    nodes[w.id] = path;
  }
  // elements
  for (const el of diagram.elements || []) {
    if (el.type === "ground") {
      const g = svgEl("g", { transform: `translate(${el.x},${el.y})`, class: "el ground" });
      g.append(svgEl("path", { d: "M0,0 V6 M-10,6 H10 M-6,10 H6 M-2,14 H2", class: "sym" }));
      svg.append(g);
      continue;
    }
    if (el.type === "transformer") {
      const g = svgEl("g", { transform: `translate(${el.x},${el.y})`, class: "el", "data-id": el.id });
      const coil = (x, dir) => `M${x},-30 v8 ` + [0, 1, 2, 3].map(() => `a5,5 0 0 ${dir} 0,10`).join(" ") + " v12";
      g.append(svgEl("path", { d: coil(-22, 1), class: "sym" }));
      g.append(svgEl("path", { d: coil(22, 0), class: "sym" }));
      g.append(svgEl("path", { d: "M-4,-22 V22 M4,-22 V22", class: "sym plate" }));
      g.append(svgEl("circle", { cx: -30, cy: -24, r: 2.4, class: "sym solid" }));
      g.append(svgEl("circle", { cx: 14, cy: -24, r: 2.4, class: "sym solid" }));
      svg.append(g);
      nodes[el.id] = g;
      const t = svgEl("text", { x: el.x, y: el.y + 48, "text-anchor": "middle", class: "el-label" });
      t.textContent = el.label || "";
      svg.append(t);
      if (el.value) {
        const t2 = svgEl("text", { x: el.x, y: el.y + 62, "text-anchor": "middle", class: "el-value" });
        t2.textContent = el.value;
        svg.append(t2);
      }
      continue;
    }
    const rot = el.rot || 0;
    const g = svgEl("g", { transform: `translate(${el.x},${el.y}) rotate(${rot})`, class: "el el-" + el.type, "data-id": el.id });
    sym(g, el.type, el);
    svg.append(g);
    nodes[el.id] = g;
    // labels in global frame, on the side opposite to the gate
    const rad = (rot * Math.PI) / 180;
    const ux = Math.cos(rad), uy = Math.sin(rad);
    let nx = uy, ny = -ux; // perpendicular (away from the nmos gate side)
    if (Math.abs(ny) > 0.5 && ny > 0) { nx = -nx; ny = -ny; }
    const off = el.type === "block" ? 0 : 18;
    const lxp = el.x + nx * off, lyp = el.y + ny * off;
    let anchor = Math.abs(nx) > 0.5 ? (nx > 0 ? "start" : "end") : "middle";
    let tx = el.type === "block" ? el.x : lxp + (anchor === "start" ? 2 : anchor === "end" ? -2 : 0);
    let ty = el.type === "block" ? el.y + 4 : lyp + (Math.abs(ny) > 0.5 ? -2 : 4);
    if (el.type === "block") anchor = "middle";
    if (el.lpos) { tx = el.lpos[0]; ty = el.lpos[1]; anchor = el.lpos[2] || "middle"; }
    const t = svgEl("text", { x: tx, y: ty, "text-anchor": anchor, class: "el-label" });
    t.textContent = el.label || "";
    svg.append(t);
    if (el.value) {
      const t2 = svgEl("text", { x: +t.getAttribute("x"), y: +t.getAttribute("y") + 13, "text-anchor": t.getAttribute("text-anchor"), class: "el-value" });
      t2.textContent = el.value;
      svg.append(t2);
    }
  }
  for (const d of diagram.dots || []) svg.append(svgEl("circle", { cx: d[0], cy: d[1], r: 3, class: "dot" }));
  for (const l of diagram.labels || []) {
    const t = svgEl("text", { x: l.x, y: l.y, "text-anchor": "middle", class: "c-label " + (l.cls || "") });
    t.textContent = l.text;
    svg.append(t);
  }
  // probes
  const probes = [];
  for (const p of diagram.probes || []) {
    const g = svgEl("g", { class: "probe", transform: `translate(${p.x},${p.y})` });
    const arrow = svgEl("path", { d: "M-9,0 H6 M2,-4 L8,0 L2,4", class: "probe-arrow" });
    const ang = { right: 0, down: 90, left: 180, up: 270 }[p.dir] || 0;
    arrow.setAttribute("transform", `rotate(${ang})`);
    const lab = svgEl("text", { x: 0, y: p.dir === "up" || p.dir === "down" ? 4 : -8, "text-anchor": p.dir === "up" || p.dir === "down" ? "start" : "middle", dx: p.dir === "up" || p.dir === "down" ? 10 : 0, class: "probe-label" });
    lab.textContent = p.label;
    g.append(arrow, lab);
    svg.append(g);
    probes.push({ spec: p, arrow, lab, ang });
  }
  const caption = document.createElement("div");
  caption.className = "circuit-caption";
  const probeLine = document.createElement("div");
  probeLine.className = "probe-line";
  wrap.append(svg, caption, probeLine);
  host.append(wrap);

  function setMode(key) {
    for (const el of svg.querySelectorAll(".on,.dim")) el.classList.remove("on", "dim");
    const md = (diagram.modes || {})[key];
    if (!md) {
      caption.innerHTML = key ? `<span class="chip mode">${esc(key)}</span>` : "파형 위에 커서를 올리면 그 순간의 도통 경로가 표시됩니다.";
      return;
    }
    for (const id of md.active || []) if (nodes[id]) nodes[id].classList.add("on");
    for (const id of md.dim || []) if (nodes[id]) nodes[id].classList.add("dim");
    caption.innerHTML = `<span class="chip mode">${esc(md.label)}</span> ${esc(md.text || "")}`;
  }

  function setProbes(values, units) {
    const parts = [];
    for (const pr of probes) {
      const v = values ? values[pr.spec.series] : undefined;
      pr.lab.textContent = pr.spec.label;
      if (v === undefined || v === null) {
        pr.arrow.setAttribute("transform", `rotate(${pr.ang})`);
        pr.arrow.classList.remove("neg", "zero");
        continue;
      }
      const zero = Math.abs(v) < 1e-9;
      pr.arrow.classList.toggle("zero", zero);
      pr.arrow.classList.toggle("neg", v < 0);
      pr.arrow.setAttribute("transform", `rotate(${pr.ang + (v < 0 ? 180 : 0)})`);
      parts.push(`<span class="pv${v < 0 ? " neg" : ""}">${esc(pr.spec.label)} = ${esc(fmtSI(v, (units && units[pr.spec.series]) || "A", 4))}</span>`);
    }
    probeLine.innerHTML = parts.length ? "커서 순간값 (화살표 = 기준 방향, 음수면 반대로 뒤집힘): " + parts.join(" · ") : "";
  }

  setMode(null);
  return { setMode, setProbes, el: wrap };
}
