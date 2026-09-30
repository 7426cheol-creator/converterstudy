// Converter FAE Lab — single page app (no build step, no external libraries).
import { esc, fmtNum, fmtPct, fmtSI, h, download, SOURCE_KO, TONE_CLASS } from "./fmt.js";
import { renderResult, statusChip } from "./results.js";

const OFFICIAL = [...Array(12)].map((_, i) => `FL${String(i + 1).padStart(2, "0")}`).concat([...Array(12)].map((_, i) => `EX${String(i + 1).padStart(2, "0")}`));

// 14-day basic path (textbook ch.01) and E13 expert rotations
const BASIC_PATH = [
  ["1일차", ["FL01"]], ["2일차", ["FL02"]], ["3일차", ["FL03", "FL04"]], ["4일차", ["FL05"]], ["5일차", ["FL06"]], ["6일차", ["FL07"]],
  ["7–8일차", ["FL08"]], ["9일차", ["FL09"]], ["10일차", ["FL10"]], ["11일차", ["FL11"]], ["12일차", ["FL12"]],
];
const EXPERT_PATH = [
  ["1회전 · 소자·측정", ["EX01", "EX02", "EX03"]], ["2회전 · 자성체·CLLC", ["EX04", "EX05"]], ["3회전 · DAB 변조", ["EX06"]],
  ["4회전 · 제어·안정성", ["EX07"]], ["5회전 · 전열·EMI·고장·불확도", ["EX08", "EX09", "EX10", "EX11"]], ["6회전 · 설계 리뷰", ["EX12"]],
];
const TITLES = {
  FL01: "Buck/Boost", FL02: "소자·gate·DPT", FL03: "손실·열", FL04: "인버터", FL05: "OBC·PFC", FL06: "제어", FL07: "자성체", FL08: "DAB", FL09: "LLC", FL10: "CLLC", FL11: "PSFB·HV-LV", FL12: "FAE 디버깅",
  EX01: "설계영역·손실지도", EX02: "비선형 Coss·ZVS", EX03: "병렬 SiC·DPT", EX04: "transformer closure", EX05: "CLLC 동역학", EX06: "DAB 변조", EX07: "시스템 안정성", EX08: "전열 mission", EX09: "EMI 경로", EX10: "고장 에너지", EX11: "검증·불확도", EX12: "통합 design review",
};
const RUBRIC = [["physics", "물리"], ["quant", "정량 추론"], ["validation", "검증"], ["tradeoff", "시스템 tradeoff"], ["customer", "고객 소통"]];

const S = {
  meta: null,
  labs: {},
  progress: { experiments: {} },
  track: safeGet("track") || "basic",
  forms: {},
  presets: {},
  lastRun: {},
  lastValues: {},
};

function safeGet(k) {
  try { return localStorage.getItem(k); } catch { return null; }
}
function safeSet(k, v) {
  try { localStorage.setItem(k, v); } catch { /* private mode */ }
}

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  let data = null;
  try { data = await r.json(); } catch { data = { message: `HTTP ${r.status}` }; }
  if (!r.ok) { const e = new Error(data.message || `HTTP ${r.status}`); e.data = data; e.status = r.status; throw e; }
  return data;
}

async function record(lab, experiment, kind, payload) {
  try {
    const slot = await api("/api/progress", { lab, experiment, kind, payload });
    S.progress.experiments[`${lab}/${experiment}`] = slot;
    return slot;
  } catch (e) {
    toast("기록 저장 실패: " + e.message, "fail");
  }
}

function toast(msg, tone = "info") {
  const t = h("div", { class: "toast " + (TONE_CLASS[tone] || tone) }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 4200);
}

// ---------------------------------------------------------------------------------
async function boot() {
  const [meta, labs, progress] = await Promise.all([api("/api/meta"), api("/api/labs"), api("/api/progress")]);
  S.meta = meta;
  for (const l of labs) S.labs[l.id] = l;
  S.progress = progress || { experiments: {} };
  renderShell();
  window.addEventListener("hashchange", route);
  route();
}

function renderShell() {
  document.body.innerHTML = "";
  const top = h(
    "header",
    { class: "topbar" },
    h("a", { class: "brand", href: "#/" }, h("span", { class: "logo" }, "⏚"), " Converter FAE Lab", h("span", { class: "ver" }, `v${S.meta.env.app_version} · 교재 v4.0`)),
    h(
      "nav",
      { class: "topnav" },
      h("a", { href: "#/" }, "시작"),
      h("a", { href: "#/trace" }, "추적표"),
      h("a", { href: "#/verify" }, "검증·환경"),
      h("a", { href: "#/interview" }, "면접 연습"),
      h("a", { href: "#/progress" }, "내 기록"),
      h("a", { href: "#/about" }, "원칙")
    )
  );
  const side = h("aside", { class: "side" });
  const main = h("main", { id: "main" });
  document.body.append(top, h("div", { class: "layout" }, side, main));
  renderSide();
}

function labState(id) {
  const lab = S.labs[id];
  const exps = Object.entries(S.progress.experiments || {}).filter(([k]) => k.startsWith(id + "/"));
  const learned = exps.some(([, v]) => v.learned);
  const last = exps.map(([, v]) => v.last_sim_status).filter(Boolean).sort((a, b) => (a.at < b.at ? 1 : -1))[0];
  return { lab, learned, last };
}

function renderSide() {
  const side = document.querySelector(".side");
  side.innerHTML = "";
  const tog = h(
    "div",
    { class: "track-toggle", role: "tablist" },
    h("button", { class: S.track === "basic" ? "on" : "", onclick: () => setTrack("basic") }, "기초 FL (14일)"),
    h("button", { class: S.track === "expert" ? "on" : "", onclick: () => setTrack("expert") }, "전문가 EX (E13)")
  );
  side.append(tog);
  const path = S.track === "basic" ? BASIC_PATH : EXPERT_PATH;
  for (const [step, ids] of path) {
    side.append(h("div", { class: "step-label" }, step));
    for (const id of ids) {
      const st = labState(id);
      const impl = !!st.lab;
      const cur = location.hash.includes(`/lab/${id}`);
      side.append(
        h(
          "a",
          { class: "lab-link" + (impl ? "" : " todo") + (cur ? " cur" : ""), href: `#/lab/${id}` },
          h("span", { class: "lid" }, id),
          h("span", { class: "lt" }, impl ? st.lab.title : TITLES[id]),
          h("span", { class: "dots" }, st.last ? h("span", { class: "dot-s " + ((TONE_CLASS[toneOf(st.last.code)]) || ""), title: "최근 시뮬레이션: " + st.last.code }) : null, st.learned ? h("span", { class: "dot-l", title: "내가 학습 완료로 표시" }, "✓") : null, impl ? null : h("span", { class: "muted small" }, "구현 예정"))
        )
      );
    }
  }
  side.append(h("p", { class: "muted small side-note" }, S.track === "basic" ? "13–14일차: 면접·영어·오답 재시험 (면접 연습 메뉴)" : "심화는 해당 FL 모델을 확장해 사용한다. 공식 추가 ID는 EX01–EX12뿐이다."));
}

function toneOf(code) {
  const map = { PASS_WITHIN_MODEL: "ok", INFO: "neutral", SCREEN_ONLY: "info", CANDIDATE_FHA_ONLY: "info", NOT_EVALUABLE: "blocked", MISSING_INPUT: "blocked", NOT_RUN_ENVIRONMENT: "blocked", UNRESOLVED_RANKING: "warn", CUSTOMER_DECISION_REQUIRED: "warn", UNRESOLVED_ROOT_CAUSE: "warn", OUT_OF_VALIDITY: "warn", MARGINAL: "warn", FAIL_CONSTRAINT: "fail", NO_SOLUTION: "fail", UNSTABLE: "fail", NO_STABLE_FIXED_POINT: "fail", SOLVER_FAILED: "fail" };
  return map[code] || "neutral";
}

function setTrack(t) {
  S.track = t;
  safeSet("track", t);
  renderSide();
}

function route() {
  const main = document.getElementById("main");
  main.innerHTML = "";
  window.scrollTo(0, 0);
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "lab" && parts[1]) {
    if (parts[1].startsWith("EX") && S.track !== "expert") setTrack("expert");
    if (parts[1].startsWith("FL") && S.track !== "basic") setTrack("basic");
    renderSide();
    return labPage(main, parts[1], parts[2]);
  }
  renderSide();
  if (parts[0] === "trace") return tracePage(main);
  if (parts[0] === "verify") return verifyPage(main);
  if (parts[0] === "progress") return progressPage(main);
  if (parts[0] === "interview") return interviewPage(main);
  if (parts[0] === "about") return aboutPage(main);
  return homePage(main);
}

// ---------------------------------------------------------------------------------
function homePage(main) {
  main.append(
    h(
      "section",
      { class: "hero" },
      h("h1", {}, "예측하고, 실행하고, 파형으로 확인하고, 고객 언어로 말한다"),
      h("p", {}, "교재 v4.0의 기초 12개(FL01–FL12)와 심화 12개(EX01–EX12) 실습을 같은 ID로 연결한 로컬 학습 시뮬레이터입니다. 모든 값은 단위·출처·모델 수준과 함께 표시되고, 시뮬레이션 통과와 나의 학습 완료는 따로 기록됩니다.")
    )
  );
  const ex02 = !!S.labs["EX02"];
  main.append(
    h(
      "div",
      { class: "start-grid" },
      h(
        "article",
        { class: "card start" },
        h("div", { class: "chip tone-info" }, "처음이라면"),
        h("h2", {}, "FL01 Buck 48→12 V부터"),
        h(
          "ol",
          {},
          h("li", {}, "기준 preset(48→12 V, 5 A)을 열고 ", h("b", {}, "D·ΔI·peak·RMS를 먼저 손으로"), " 적은 뒤 실행한다."),
          h("li", {}, "파형 위에서 커서를 움직여 회로의 ON/OFF 도통 경로와 전류 방향을 확인한다."),
          h("li", {}, h("b", {}, "L 100 → 50 µH"), "로 바꾸기 전에 리플·평균·peak가 어떻게 될지 예측하고 실행한다."),
          h("li", {}, "‘경부하’ 실험에서 I_o를 0.3 A로 내려 diode-emulation DCM과 강제 동기 음전류를 비교한다.")
        ),
        h("a", { class: "btn primary", href: "#/lab/FL01/buck_ccm" }, "FL01 시작 →")
      ),
      h(
        "article",
        { class: "card start" },
        h("div", { class: "chip tone-ok" }, "기초를 통과했다면"),
        h("h2", {}, "EX02 비선형 Coss와 dead time"),
        h(
          "ol",
          {},
          h("li", {}, "C(v) = 2 nF/√(1+v/40 V), 800 V에서 Qoss·Eoss를 먼저 적분하고, 4 A 정전류 전환시간(286.6 ns)을 Coss(800 V) 한 점 계산(174.6 ns)과 비교한다."),
          h("li", {}, h("b", {}, "dead time 200 → 300 ns"), "로 바꾸기 전에 완전/부분 전환 여부를 예측한다."),
          h("li", {}, "전류 부호를 뒤집어 전하가 원하는 노드로 가지 않는 경우를 본다.")
        ),
        ex02 ? h("a", { class: "btn primary", href: "#/lab/EX02" }, "EX02 시작 →") : h("span", { class: "chip tone-blocked" }, "EX02 구현 중")
      ),
      h(
        "article",
        { class: "card start" },
        h("div", { class: "chip tone-warn" }, "오늘 30분"),
        h("h2", {}, "세 가지 변경 실험"),
        h(
          "ul",
          {},
          h("li", {}, h("a", { href: "#/lab/EX02/hb_constant_current" }, h("b", {}, "EX02")), " dead time 200 → 300 ns: 부분 전환이 완전 전환으로 바뀌는가? (‘제안된 변경 적용’)"),
          h("li", {}, h("a", { href: "#/lab/EX06/general_modulation" }, h("b", {}, "EX06")), " 900/600 V, 1.5 kW에서 SPS와 width candidate(w₁/π = 0.7)의 RMS·peak 비교"),
          h("li", {}, h("a", { href: "#/lab/EX07/cpl_exact" }, h("b", {}, "EX07")), " CPL 입력필터 C 100 µF → 1 mF: 극점이 좌반면으로 가는가?")
        ),
        h("p", { class: "muted small" }, "각 실험은 ‘예측 → 실행 → 파형 → 왜 → 고객 문장’ 순서로 진행됩니다.")
      )
    )
  );
  // status overview
  const tb = h("tbody");
  for (const id of OFFICIAL) {
    const st = labState(id);
    tb.append(
      h(
        "tr",
        {},
        h("td", {}, h("a", { href: `#/lab/${id}` }, id)),
        h("td", {}, st.lab ? st.lab.title : TITLES[id]),
        h("td", {}, st.lab ? h("span", { class: "chip small tone-ok" }, st.lab.implementation) : h("span", { class: "chip small tone-blocked" }, "구현 예정")),
        h("td", {}, st.last ? h("span", { class: "chip small " + TONE_CLASS[toneOf(st.last.code)] }, st.last.code) : h("span", { class: "muted small" }, "아직 실행 안 함")),
        h("td", {}, st.learned ? h("span", { class: "chip small tone-ok" }, "내가 완료 표시") : h("span", { class: "muted small" }, "—")),
        h("td", {}, h("span", { class: "chip small tone-blocked" }, "HW 검증 없음"))
      )
    );
  }
  main.append(
    h(
      "section",
      { class: "card" },
      h("h3", {}, "24개 실습 상태 (구현 · 최근 시뮬레이션 · 나의 학습 · 하드웨어 검증은 모두 별도)"),
      h("div", { class: "tablewrap" }, h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["ID", "주제", "구현", "최근 시뮬레이션", "나의 학습", "HW"].map((c) => h("th", {}, c)))), tb))
    )
  );
}

// ---------------------------------------------------------------------------------
async function labPage(main, id, expKey) {
  if (!S.labs[id]) {
    main.append(h("section", { class: "card" }, h("h2", {}, `${id} · ${TITLES[id] || ""}`), h("p", {}, "이 실습은 아직 구현되지 않았습니다. 진행 상태는 추적표와 docs/progress.json에 기록됩니다.")));
    return;
  }
  let lab;
  try {
    lab = await api(`/api/labs/${id}`);
  } catch (e) {
    main.append(h("p", { class: "err" }, e.message));
    return;
  }
  const exp = lab.experiments.find((e) => e.key === expKey) || lab.experiments[0];
  const tb = S.meta.textbook_available;
  main.append(
    h(
      "section",
      { class: "labhead" },
      h("div", { class: "row wrap gap" }, h("span", { class: "lab-id" }, lab.id), h("h1", {}, lab.title), h("span", { class: "muted" }, lab.title_en)),
      h(
        "div",
        { class: "row wrap gap small" },
        h("span", { class: "chip" }, lab.path_note),
        ...lab.textbook.map((t) => (tb ? h("a", { class: "chip link", href: `/textbook#${t.anchor}`, target: "_blank" }, "📖 " + t.title) : h("span", { class: "chip", title: "textbook/ 폴더에 교재 HTML을 두면 링크가 열립니다" }, "📖 " + t.title))),
        lab.prerequisites.length ? h("span", { class: "muted" }, "선행: " + lab.prerequisites.join(", ")) : null
      ),
      h("p", {}, lab.summary),
      h("details", { class: "scope" }, h("summary", {}, "계약상 최소 범위와 주장 한계"), h("p", {}, h("b", {}, "최소 범위: "), lab.minimum_scope), h("ul", {}, ...lab.claim_limits.map((c) => h("li", {}, c))), lab.implementation_note ? h("p", { class: "muted" }, lab.implementation_note) : null)
    )
  );
  const tabs = h("nav", { class: "exp-tabs" });
  for (const e of lab.experiments) tabs.append(h("a", { class: e.key === exp.key ? "on" : "", href: `#/lab/${id}/${e.key}` }, e.title, h("span", { class: "chip small level" }, e.model_level.split(" ")[0])));
  main.append(tabs);
  experimentView(main, lab, exp);
}

function formKey(lab, exp) {
  return `${lab.id}/${exp.key}`;
}

function displayValue(p, v) {
  if (p.kind === "choice" || p.kind === "bool") return v;
  if (p.kind === "int") return String(v);
  const x = v / (p.scale || 1);
  return String(Number(x.toPrecision(7)));
}

function experimentView(main, lab, exp) {
  const fk = formKey(lab, exp);
  if (!S.forms[fk]) {
    S.presets[fk] = exp.presets[0] ? exp.presets[0].key : null;
    S.forms[fk] = presetValues(exp, S.presets[fk]);
  }
  const wrap = h("div", { class: "exp" });
  main.append(wrap);
  // step 1
  wrap.append(stepCard(1, "이번에 배우는 것", h("p", {}, exp.goal), h("p", { class: "muted small" }, "모델 수준: ", h("b", {}, exp.model_level), exp.claim_limit ? " · 주장 한계: " + exp.claim_limit : "")));
  // step 2: parameters
  const errBox = h("div", { class: "form-errors" });
  const form = h("div", { class: "params" });
  const presetSel = h("select", { "aria-label": "preset" });
  for (const p of exp.presets) presetSel.append(h("option", { value: p.key, selected: p.key === S.presets[fk] }, `${p.label}${p.tags.length ? " · " + p.tags.join("/") : ""}`));
  presetSel.addEventListener("change", () => {
    S.presets[fk] = presetSel.value;
    S.forms[fk] = presetValues(exp, presetSel.value);
    rerender();
  });
  const btnReset = h("button", { class: "btn", onclick: () => { S.forms[fk] = presetValues(exp, S.presets[fk]); rerender(); } }, "preset 값으로 되돌리기");
  const sugg = Object.keys(exp.suggested || {}).length
    ? h("button", { class: "btn accent", onclick: () => { for (const [k, v] of Object.entries(exp.suggested)) { const p = exp.params.find((q) => q.key === k); if (p) S.forms[fk][k] = displayValue(p, v); } rerender(); toast("제안된 변경을 적용했습니다. 실행 전에 결과를 예측하세요."); } }, "제안된 변경 적용")
    : null;
  const groups = {};
  for (const p of exp.params) (groups[p.group] = groups[p.group] || []).push(p);
  const base = presetValues(exp, S.presets[fk]);
  for (const [g, ps] of Object.entries(groups)) {
    const fs = h("fieldset", {}, h("legend", {}, g));
    for (const p of ps) {
      const val = S.forms[fk][p.key];
      let input;
      if (p.kind === "choice") {
        input = h("select", { "data-key": p.key });
        for (const c of p.choices) input.append(h("option", { value: c.value, selected: c.value === val }, c.label));
      } else if (p.kind === "bool") {
        input = h("input", { type: "checkbox", "data-key": p.key, checked: !!val });
      } else {
        input = h("input", { type: "text", inputmode: "decimal", value: val, "data-key": p.key, size: 9, spellcheck: "false" });
      }
      input.addEventListener("change", () => {
        S.forms[fk][p.key] = p.kind === "bool" ? input.checked : input.value;
        row.classList.toggle("changed", String(S.forms[fk][p.key]) !== String(base[p.key]));
      });
      const range = p.kind === "float" || p.kind === "int" ? rangeText(p) : "";
      const row = h(
        "label",
        { class: "prow" + (String(val) !== String(base[p.key]) ? " changed" : ""), "data-row": p.key },
        h("span", { class: "plabel" }, p.label),
        h("span", { class: "pinput" }, input, p.kind === "float" || p.kind === "int" ? h("span", { class: "unit" }, p.display_unit || p.unit) : null),
        h("span", { class: "chip small src-" + p.source, title: p.source_note || "" }, SOURCE_KO[p.source] || p.source),
        range ? h("span", { class: "muted tiny" }, range) : null,
        h("span", { class: "perr", "data-err": p.key })
      );
      fs.append(row);
    }
    form.append(fs);
  }
  wrap.append(
    stepCard(
      2,
      "바꿀 파라미터",
      h("div", { class: "row wrap gap" }, h("label", { class: "row gap" }, "preset ", presetSel), btnReset, sugg),
      exp.suggested_change ? h("p", { class: "suggest" }, "제안: ", h("b", {}, exp.suggested_change)) : null,
      h("p", { class: "muted small" }, "숫자만 쓰면 옆에 표시된 단위로 읽습니다. ‘0.1 mH’, ‘50k’처럼 단위를 붙여도 됩니다. 범위를 벗어나면 값을 잘라내지 않고 이유와 함께 거부합니다."),
      form,
      errBox
    )
  );
  // step 3/4: predict and run
  const predictBox = h("div", { class: "predict" });
  const runBtn = h("button", { class: "btn primary big" }, "실행");
  const runNote = h("span", { class: "muted small" }, "");
  wrap.append(stepCard(3, "먼저 예측 → 실행", predictBox, h("div", { class: "row gap" }, runBtn, runNote)));
  const resultBox = h("div", { class: "result-box" });
  wrap.append(resultBox);
  if (S.lastRun[fk]) {
    renderResultInto(resultBox, S.lastRun[fk], null, exp);
  }
  // steps 6-7: explanations, customer, questions
  wrap.append(
    stepCard(
      6,
      "왜 그런가 (펼쳐서 읽기)",
      h("details", {}, h("summary", {}, "학생 설명"), h("p", {}, exp.student)),
      h("details", {}, h("summary", {}, "전문가 확장"), h("p", {}, exp.expert))
    )
  );
  wrap.append(stepCard(7, "고객에게 어떻게 말할까", h("p", {}, exp.customer_ko), h("details", {}, h("summary", {}, "English"), h("p", { lang: "en" }, exp.customer_en))));
  wrap.append(questionsCard(lab, exp));
  wrap.append(selfAssessCard(lab, exp));

  function rerender() {
    const m = document.getElementById("main");
    m.innerHTML = "";
    labPage(m, lab.id, exp.key);
  }

  function collectValues() {
    const out = {};
    for (const p of exp.params) {
      const v = S.forms[fk][p.key];
      out[p.key] = p.kind === "choice" || p.kind === "bool" ? v : String(v).trim();
    }
    return out;
  }

  function changedKeys(a, b) {
    if (!a || !b) return [];
    return Object.keys(a).filter((k) => String(a[k]) !== String(b[k]));
  }

  const predState = { choice: null, reason: "", sig: null };
  function refreshPredict() {
    const values = collectValues();
    const prevValues = S.lastValues[fk];
    const ch = changedKeys(values, prevValues);
    const sig = JSON.stringify([!!prevValues, ch.map((k) => [k, values[k]])]);
    if (sig === predState.sig) return; // nothing new: keep the user's choice and text
    predState.sig = sig;
    predictBox.innerHTML = "";
    if (!prevValues) {
      predictBox.append(h("p", {}, h("b", {}, "기준 실행입니다."), " 실행 전에 아래 값을 손으로 계산해 적어 보세요 (건너뛰어도 됩니다)."));
      const hc = exp.prediction.handcalc || [];
      const inputs = {};
      if (hc.length) {
        const grid = h("div", { class: "handcalc" });
        for (const x of hc) {
          const inp = h("input", { type: "text", size: 9, placeholder: "내 계산", "data-hc": x.key });
          inputs[x.key] = inp;
          grid.append(h("label", { class: "row gap" }, h("span", {}, x.label), inp, h("span", { class: "unit" }, x.unit)));
        }
        predictBox.append(grid);
      }
      runBtn.textContent = "기준 실행";
      runBtn.onclick = () => doRun({ kind: "handcalc", inputs, hc });
      runNote.textContent = "";
    } else if (!ch.length) {
      predictBox.append(h("p", { class: "muted" }, "파라미터가 이전 실행과 같습니다. 무엇을 바꿀지 정한 뒤 예측하세요. ", exp.suggested_change ? "제안: " + exp.suggested_change : ""));
      runBtn.textContent = "다시 실행";
      runBtn.onclick = () => doRun(null);
    } else {
      const diffTxt = ch.map((k) => { const p = exp.params.find((q) => q.key === k); return `${p ? p.label : k}: ${prevValues[k]} → ${values[k]} ${p && p.display_unit ? p.display_unit : ""}`; }).join(", ");
      predictBox.append(h("p", {}, h("b", {}, "바뀐 것: "), diffTxt));
      predictBox.append(h("p", { class: "q" }, exp.prediction.question));
      const opts = h("div", { class: "opts" });
      for (const o of exp.prediction.options) {
        const b = h("button", { class: "opt" + (predState.choice === o ? " on" : ""), type: "button" }, o);
        b.addEventListener("click", () => { predState.choice = o; for (const x of opts.children) x.classList.toggle("on", x === b); });
        opts.append(b);
      }
      const reason = h("textarea", { rows: 2, placeholder: "이유를 한두 문장으로 (예: ΔI = (V_in−V_o)D/(L f_s)이므로…)" });
      reason.value = predState.reason;
      reason.addEventListener("input", () => { predState.reason = reason.value; });
      predictBox.append(opts, reason);
      runBtn.textContent = "예측하고 실행";
      runBtn.onclick = () => {
        const cur = changedKeys(collectValues(), S.lastValues[fk]);
        if (!predState.choice) { toast("먼저 예측을 고르세요 (모르겠다도 가능)", "warn"); return; }
        doRun({ kind: "prediction", choice: predState.choice, reason: predState.reason, changed: cur.length ? cur : ch, diffTxt });
      };
      const skip = h("a", { href: "javascript:void 0", class: "muted small" }, "예측 없이 결과 보기 (건너뜀으로 기록)");
      skip.addEventListener("click", () => doRun({ kind: "prediction", choice: null, skipped: true, changed: ch, diffTxt }));
      runNote.innerHTML = "";
      runNote.append(skip);
    }
  }
  for (const el of form.querySelectorAll("input,select")) {
    el.addEventListener("change", refreshPredict);
    el.addEventListener("input", () => { const k = el.getAttribute("data-key"); const p = exp.params.find((q) => q.key === k); if (p && p.kind !== "bool") S.forms[fk][k] = el.value; refreshPredict(); });
  }
  refreshPredict();

  async function doRun(pred) {
    for (const e of form.querySelectorAll(".perr")) e.textContent = "";
    errBox.innerHTML = "";
    runBtn.disabled = true;
    const spin = h("span", { class: "spinner" }, exp.runtime_hint === "fast" ? "계산 중…" : "계산 중… (수 초 걸릴 수 있음)");
    runNote.prepend(spin);
    const values = collectValues();
    let payload;
    try {
      payload = await api("/api/run", { lab: lab.id, experiment: exp.key, preset: S.presets[fk], values });
    } catch (e) {
      runBtn.disabled = false;
      spin.remove();
      if (e.data && e.data.errors) {
        for (const [k, msg] of Object.entries(e.data.errors)) {
          const slot = form.querySelector(`[data-err="${k}"]`);
          if (slot) slot.textContent = msg;
          else errBox.append(h("div", { class: "err" }, `${k}: ${msg}`));
        }
        errBox.prepend(h("div", { class: "err" }, e.message));
      } else {
        errBox.append(h("div", { class: "err" }, "계산 실패: " + e.message), e.data && e.data.trace ? h("pre", { class: "trace" }, e.data.trace) : null);
      }
      return;
    }
    runBtn.disabled = false;
    spin.remove();
    const previous = S.lastRun[fk] || null;
    let reveal = null;
    if (pred && pred.kind === "handcalc") {
      const rows = [];
      const mt = Object.fromEntries(payload.result.metrics.map((m) => [m.key, m]));
      for (const x of pred.hc || []) {
        const raw = pred.inputs[x.key] ? pred.inputs[x.key].value.trim() : "";
        const m = mt[x.key];
        if (!raw || !m) continue;
        const mine = parseFloat(raw.replace(",", ""));
        const err = m && typeof m.value === "number" && isFinite(mine) ? Math.abs(mine - m.value) / Math.max(Math.abs(m.value), 1e-30) : null;
        rows.push({ key: x.key, label: x.label, mine: raw, unit: x.unit, model: m.value, err });
      }
      if (rows.length) record(lab.id, exp.key, "handcalc", { rows, preset: S.presets[fk] });
      reveal = { kind: "handcalc", rows };
    } else if (pred && pred.kind === "prediction") {
      const sugg = exp.suggested || {};
      const matchesSuggested = pred.changed.length && pred.changed.every((k) => k in sugg);
      record(lab.id, exp.key, "prediction", { question: exp.prediction.question, choice: pred.choice, reason: pred.reason || "", skipped: !!pred.skipped, change: pred.diffTxt, matches_suggested: !!matchesSuggested, expected: matchesSuggested ? exp.prediction.expected : null });
      reveal = { kind: "prediction", ...pred, matchesSuggested };
    }
    record(lab.id, exp.key, "sim_status", { code: payload.result.status.code, preset: S.presets[fk] });
    S.lastRun[fk] = payload;
    S.lastValues[fk] = values;
    renderResultInto(resultBox, payload, previous, exp, reveal);
    predState.choice = null;
    predState.reason = "";
    predState.sig = null;
    refreshPredict();
    renderSide();
    resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderResultInto(box, payload, previous, exp2, reveal) {
    box.innerHTML = "";
    const head = h("div", { class: "row wrap gap result-head" }, h("h2", {}, "④ 실행 결과 → ⑤ 파형에서 확인"));
    const ex = h("div", { class: "row gap" });
    for (const f of ["csv", "html", "json"]) ex.append(h("button", { class: "btn small", onclick: () => exportRun(lab, exp2, f) }, f.toUpperCase() + " 내보내기"));
    head.append(ex);
    box.append(head);
    if (reveal) box.append(revealCard(exp2, reveal, payload));
    renderResult(box, payload, { previous, highlight: exp2.prediction.metric_keys });
  }

  async function exportRun(lab2, exp2, fmt) {
    const values = S.lastValues[fk] || collectValues();
    const r = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lab: lab2.id, experiment: exp2.key, preset: S.presets[fk], values, format: fmt }) });
    if (!r.ok) { toast("내보내기 실패", "fail"); return; }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${lab2.id}_${exp2.key}.${fmt}`;
    a.click();
  }
}

function revealCard(exp, rv, payload) {
  const card = h("section", { class: "card reveal" });
  if (rv.kind === "handcalc") {
    card.append(h("h3", {}, "내 손계산 vs 모델"));
    if (!rv.rows.length) card.append(h("p", { class: "muted" }, "손계산을 적지 않았습니다. 다음 변경 실행 전에는 예측을 먼저 적으세요."));
    else {
      const t = h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["항목", "내 계산", "모델", "차이"].map((c) => h("th", {}, c)))));
      const b = h("tbody");
      for (const r of rv.rows) b.append(h("tr", {}, h("td", {}, r.label), h("td", { class: "num" }, `${r.mine} ${r.unit}`), h("td", { class: "num" }, fmtSI(r.model, r.unit, 6)), h("td", { class: "num" }, r.err === null ? "—" : fmtPct(r.err, 2))));
      t.append(b);
      card.append(t, h("p", { class: "muted small" }, "단위는 표시된 단위 그대로 읽습니다(접두어 없이 기본 단위). 차이가 크면 식·단위·기준(평균/peak/RMS)을 다시 확인하세요."));
    }
    return card;
  }
  card.append(h("h3", {}, "내 예측과 결과"));
  card.append(h("p", {}, h("b", {}, "질문: "), exp.prediction.question));
  card.append(h("p", {}, h("b", {}, "내 예측: "), rv.skipped ? "건너뜀" : rv.choice, rv.reason ? ` — “${rv.reason}”` : ""));
  if (rv.matchesSuggested) {
    const ok = rv.choice === exp.prediction.expected;
    card.append(h("p", {}, h("b", {}, "제안된 변경의 기대 답: "), h("span", { class: "chip " + (ok ? "tone-ok" : "tone-warn") }, exp.prediction.expected), ok ? " ✓ 일치" : " — 다름: 파형과 수치를 보고 이유를 다시 써 보세요"));
  } else {
    card.append(h("p", { class: "muted" }, "제안된 변경과 다른 변경이라 자동 비교는 하지 않습니다. 아래 ‘이전 실행’ 대비 변화 열로 직접 확인하세요."));
  }
  card.append(h("details", { open: !!rv.matchesSuggested }, h("summary", {}, "왜 그런가"), h("p", {}, exp.prediction.why)));
  return card;
}

function stepCard(n, title, ...children) {
  return h("section", { class: "card step" }, h("h3", {}, h("span", { class: "stepn" }, String(n)), " ", title), ...children);
}

function rangeText(p) {
  const f = (v) => fmtSI(v, p.unit, 4);
  if (p.vmin !== null && p.vmax !== null) return `유효 ${f(p.vmin)} ~ ${f(p.vmax)}`;
  if (p.vmin !== null) return `유효 ≥ ${f(p.vmin)}`;
  if (p.vmax !== null) return `유효 ≤ ${f(p.vmax)}`;
  return "";
}

function presetValues(exp, presetKey) {
  const out = {};
  const pr = exp.presets.find((p) => p.key === presetKey);
  for (const p of exp.params) {
    const v = pr && p.key in pr.values ? pr.values[p.key] : p.default;
    out[p.key] = displayValue(p, v);
  }
  return out;
}

function questionsCard(lab, exp) {
  const slot = S.progress.experiments[`${lab.id}/${exp.key}`] || {};
  const card = stepCard(8, "면접 질문 — 먼저 내 답을 쓰고 모범 답을 연다");
  exp.questions.forEach((q, i) => {
    const saved = (slot.answers || {})[String(i)];
    const ta = h("textarea", { rows: 3, placeholder: "내 답 (60~90초 분량). 저장하면 내 기록에 남습니다." });
    ta.value = saved ? saved.text : "";
    const btn = h("button", { class: "btn small" }, "내 답 저장");
    btn.addEventListener("click", async () => {
      await record(lab.id, exp.key, "answer", { qid: i, text: ta.value });
      toast("저장했습니다 (시뮬레이션 결과와 별도 필드)", "ok");
    });
    card.append(
      h(
        "div",
        { class: "qa" },
        h("p", { class: "q" }, h("span", { class: "chip small" }, q.kind), " ", q.q_ko, q.q_en ? h("span", { class: "muted small", lang: "en" }, "  / " + q.q_en) : null),
        ta,
        h("div", { class: "row gap" }, btn, saved ? h("span", { class: "muted small" }, "저장됨 " + saved.at) : null),
        h("details", {}, h("summary", {}, "모범 답 열기"), h("p", {}, q.answer_ko), q.must_include.length ? h("p", { class: "small" }, h("b", {}, "빠지면 안 되는 내용: "), q.must_include.join(" · ")) : null, q.answer_en ? h("details", {}, h("summary", {}, "English answer"), h("p", { lang: "en" }, q.answer_en)) : null)
      )
    );
  });
  return card;
}

function selfAssessCard(lab, exp) {
  const key = `${lab.id}/${exp.key}`;
  const slot = S.progress.experiments[key] || {};
  const sa = slot.self_assessment || {};
  const card = stepCard(9, "나의 학습 기록 (시뮬레이션 통과와 별개)");
  card.append(h("p", { class: "muted small" }, "0=단어만 기억 · 1=식을 보고 설명 · 2=손계산·단위 맞음 · 3=파형·corner·tradeoff 독립 설명 · 4=반례·불확도·추가검증까지 방어 (교재 E13 rubric, 자체평가)"));
  const scores = {};
  for (const [k, label] of RUBRIC) {
    const sel = h("select", { "data-axis": k });
    sel.append(h("option", { value: "" }, "—"));
    for (let v = 0; v <= 4; v++) sel.append(h("option", { value: String(v), selected: String(sa[k]) === String(v) }, String(v)));
    scores[k] = sel;
    card.append(h("label", { class: "row gap" }, h("span", { class: "plabel" }, label), sel));
  }
  const save = h("button", { class: "btn small" }, "자체평가 저장");
  save.addEventListener("click", async () => {
    const out = {};
    for (const [k, sel] of Object.entries(scores)) if (sel.value !== "") out[k] = +sel.value;
    await record(lab.id, exp.key, "self_assessment", { scores: out });
    toast("저장했습니다", "ok");
  });
  const learned = h("input", { type: "checkbox", checked: !!slot.learned });
  learned.addEventListener("change", async () => {
    await record(lab.id, exp.key, "learned", { value: learned.checked });
    renderSide();
  });
  card.append(h("div", { class: "row gap" }, save), h("label", { class: "row gap learned" }, learned, "이 실험을 ‘학습 완료’로 표시 (내가 판단 — 자동으로 켜지지 않음)"));
  const preds = (slot.predictions || []).slice(-3).reverse();
  if (preds.length) {
    card.append(h("h4", {}, "최근 예측"));
    for (const p of preds) card.append(h("p", { class: "small" }, `${p.at} · ${p.change || ""} · 예측: ${p.skipped ? "건너뜀" : p.choice}${p.expected ? ` (기대: ${p.expected})` : ""}`));
  }
  return card;
}

// ---------------------------------------------------------------------------------
function tracePage(main) {
  const tr = S.meta.docs && S.meta.docs["traceability.json"];
  main.append(h("h1", {}, "추적표 (textbook ↔ Lab ↔ code ↔ test ↔ result)"));
  main.append(h("p", { class: "muted" }, "학습 내용 · 구현 · 실제 실행 · 수치 검증 · 하드웨어 validation 상태를 따로 적습니다. 원본 simulation_contract_v4.json이 제공되지 않아 계약은 교재·지침에서 재구성했습니다(docs/ERRATA.md)."));
  const rows = tr && tr.rows ? tr.rows : OFFICIAL.map((id) => {
    const l = S.labs[id];
    return { textbook: l ? l.textbook.map((t) => t.title).join("; ") : "", id, scenario: l ? l.experiments.map((e) => e.key).join(", ") : "", code: l ? l.code_path : "", reference: "", test: l ? l.test_paths.join(", ") : "", result: "", implementation: l ? l.implementation : "NOT_IMPLEMENTED", run: "", verification: "", hardware: "NOT_VALIDATED_HARDWARE", not_valid_for: l ? l.claim_limits.join("; ") : "" };
  });
  const cols = [["id", "ID"], ["textbook", "교재"], ["scenario", "시나리오"], ["code", "코드"], ["reference", "기준"], ["test", "테스트"], ["result", "결과 파일"], ["implementation", "구현"], ["run", "실행"], ["verification", "검증"], ["hardware", "HW"], ["not_valid_for", "not_valid_for"]];
  const t = h("table", { class: "data trace" }, h("thead", {}, h("tr", {}, ...cols.map((c) => h("th", {}, c[1])))));
  const b = h("tbody");
  for (const r of rows) b.append(h("tr", {}, ...cols.map(([k]) => h("td", { class: "small" }, String(r[k] ?? "")))));
  t.append(b);
  main.append(h("section", { class: "card" }, h("div", { class: "tablewrap" }, t)));
}

function verifyPage(main) {
  const m = S.meta;
  main.append(h("h1", {}, "검증 · 실행 환경"));
  const env = m.env;
  main.append(h("section", { class: "card" }, h("h3", {}, "이 실행 환경"), h("p", {}, `app ${env.app_version} · contract ${env.contract_version} · Python ${env.python} · numpy ${env.numpy} · scipy ${env.scipy} · ${env.platform}`)));
  main.append(
    h(
      "section",
      { class: "card" },
      h("h3", {}, "외부 도구 (조건부 독립 대조)"),
      h("ul", {}, h("li", {}, "MATLAB/Simulink/Simscape: ", h("span", { class: "chip tone-blocked" }, m.external_tools.matlab_simulink), " — builder 스크립트와 실행 절차는 matlab/에 있고 실행하지 않았다."), h("li", {}, "PSIM 12.0.2: ", h("span", { class: "chip tone-blocked" }, m.external_tools.psim_12_0_2), " — 수동 재구성용 build sheet는 psim/에 있다."), h("li", {}, "GNU Octave: 교재 식으로 다시 쓴 독립 검산 (matlab/xc_*.m, verification/run_octave_crosscheck.sh, 결과 matlab/results/octave_crosscheck.json). Octave 8.4에서 실행했다; MATLAB에서는 실행하지 않았다."))
    )
  );
  const man = m.docs && m.docs["run_manifest.json"];
  if (man) {
    const t = h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["Lab", "실험", "preset", "상태", "실패 검증", "시간 s"].map((c) => h("th", {}, c)))));
    const b = h("tbody");
    for (const r of man.runs) b.append(h("tr", {}, h("td", {}, r.lab), h("td", {}, r.experiment), h("td", {}, r.preset), h("td", {}, h("span", { class: "chip small " + TONE_CLASS[toneOf(r.status)] }, r.status)), h("td", { class: "small" }, (r.checks_failed || []).join(", ")), h("td", { class: "num" }, fmtNum(r.runtime_s, 3))));
    t.append(b);
    main.append(h("section", { class: "card" }, h("h3", {}, `run manifest (${man.generated})`), h("div", { class: "tablewrap" }, t)));
  } else main.append(h("p", { class: "muted" }, "docs/run_manifest.json이 아직 없습니다: `python -m convlab run-all --out results` 후 docs에 복사됩니다."));
  const er = m.docs && m.docs["errata.json"];
  if (er) {
    const t = h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["ID", "위치", "원문", "정정", "근거", "영향"].map((c) => h("th", {}, c)))));
    const b = h("tbody");
    for (const r of er.items || []) b.append(h("tr", {}, h("td", {}, r.id), h("td", { class: "small" }, r.where), h("td", { class: "small" }, r.original), h("td", { class: "small" }, r.corrected), h("td", { class: "small" }, r.evidence), h("td", { class: "small" }, r.impact)));
    t.append(b);
    main.append(h("section", { class: "card" }, h("h3", {}, "Errata (교재·지침 수치 정정 기록)"), h("div", { class: "tablewrap" }, t)));
  }
}

function progressPage(main) {
  main.append(h("h1", {}, "내 기록"));
  main.append(h("p", { class: "muted" }, "예측·손계산·답변·자체평가·학습 완료 표시는 이 컴퓨터의 ~/.convlab/progress.json에만 저장됩니다. 회사·고객 자료는 입력하지 마세요."));
  const exps = Object.entries(S.progress.experiments || {});
  if (!exps.length) main.append(h("p", {}, "아직 기록이 없습니다."));
  for (const [k, v] of exps) {
    main.append(
      h(
        "section",
        { class: "card" },
        h("h3", {}, h("a", { href: `#/lab/${k}` }, k)),
        h("p", { class: "small" }, `예측 ${v.predictions.length}건 · 손계산 ${v.handcalc.length}건 · 답변 ${Object.keys(v.answers).length}건 · 최근 시뮬레이션 ${v.last_sim_status ? v.last_sim_status.code : "—"} · 학습 완료 표시 ${v.learned ? "예 (" + v.learned_at + ")" : "아니오"}`),
        ...v.predictions.slice(-5).map((p) => h("p", { class: "small muted" }, `${p.at} · ${p.change} · ${p.skipped ? "건너뜀" : p.choice}${p.expected ? " / 기대 " + p.expected : ""} ${p.reason ? "— " + p.reason : ""}`))
      )
    );
  }
  main.append(h("button", { class: "btn", onclick: () => download("convlab_progress.json", JSON.stringify(S.progress, null, 1), "application/json") }, "JSON으로 내보내기"));
}

async function interviewPage(main) {
  main.append(h("h1", {}, "면접 연습"));
  main.append(h("p", {}, "구현된 모든 실습의 질문을 모았습니다. 60~90초 답을 먼저 말하거나 적고, 모범 답을 연 뒤 E13 rubric으로 자체평가합니다. 이 점수는 readiness로 자동 저장되지 않습니다."));
  const all = [];
  for (const id of OFFICIAL) {
    if (!S.labs[id]) continue;
    const lab = await api(`/api/labs/${id}`);
    for (const e of lab.experiments) e.questions.forEach((q, i) => all.push({ lab: id, exp: e.key, i, q }));
  }
  const box = h("div");
  const timer = h("span", { class: "chip" }, "90 s");
  let t0 = null, iv = null;
  const next = h("button", { class: "btn primary" }, "무작위 질문");
  next.addEventListener("click", () => {
    const pick = all[Math.floor(Math.random() * all.length)];
    box.innerHTML = "";
    box.append(h("section", { class: "card" }, h("p", { class: "muted small" }, `${pick.lab} / ${pick.exp}`), h("h2", {}, pick.q.q_ko), pick.q.q_en ? h("p", { lang: "en", class: "muted" }, pick.q.q_en) : null, h("details", {}, h("summary", {}, "모범 답"), h("p", {}, pick.q.answer_ko), h("p", { class: "small" }, "빠지면 안 되는 내용: " + pick.q.must_include.join(" · ")), pick.q.answer_en ? h("p", { lang: "en" }, pick.q.answer_en) : null)));
    clearInterval(iv);
    t0 = Date.now();
    iv = setInterval(() => { const s = 90 - Math.floor((Date.now() - t0) / 1000); timer.textContent = s >= 0 ? `${s} s` : `+${-s} s`; }, 250);
  });
  main.append(h("div", { class: "row gap" }, next, timer, h("span", { class: "muted small" }, `${all.length}문항`)), box);
}

function aboutPage(main) {
  main.append(
    h("h1", {}, "이 앱이 하지 않는 것"),
    h(
      "section",
      { class: "card" },
      h(
        "ul",
        {},
        h("li", {}, "합격을 보장하지 않고, AI가 만든 모델 결과를 사용자의 실무 경력·숙련으로 기록하지 않습니다."),
        h("li", {}, "기본 입력은 합성(ASSUMED/교재) 값입니다. 실제 Infineon 제품을 합성 수치로 성능 비교하지 않습니다. 회사·고객 회로/데이터를 쓰지 않습니다."),
        h("li", {}, "모델 수준 A(해석/FHA)·B(평균)·C(이상 스위칭)·D(비이상 commutation, 합성)를 결과마다 표시하고, A/B 결과로 D 수준 주장을 하지 않습니다."),
        h("li", {}, "이상 스위치 모델의 ZVS는 NOT_EVALUABLE, 전하 screen만 있으면 SCREEN_ONLY입니다. EMI·수명·안전 적합 PASS를 만들지 않습니다."),
        h("li", {}, "숫자를 기준값에 맞추기 위한 비물리적 보정을 하지 않습니다. 원본과 다르면 errata에 근거를 남깁니다."),
        h("li", {}, "시뮬레이션 통과, 나의 답, 학습 완료 표시는 서로 다른 필드입니다. ‘면접 준비 완료’는 자동으로 표시되지 않습니다.")
      )
    )
  );
}

boot().catch((e) => {
  document.body.innerHTML = `<pre class="err">앱을 시작하지 못했습니다: ${esc(e.message)}</pre>`;
});
