// Standalone exported report (inlined into the HTML export; no network access).
import { esc, h } from "./fmt.js";
import { renderResult } from "./results.js";

export function renderReport(host, data) {
  const { payload, meta } = data;
  const lab = (meta && meta.lab) || {};
  const exp = (meta && meta.experiment) || {};
  host.append(
    h(
      "header",
      { class: "report-head" },
      h("div", { class: "muted small" }, "Converter FAE Lab · 교재 v4.0 정합 학습 시뮬레이터 · 내보낸 보고서 (합성 학습 모델, 실측 아님)"),
      h("h1", {}, `${payload.lab} · ${lab.title || ""}`),
      h("h2", {}, exp.title || payload.experiment),
      exp.goal ? h("p", {}, h("b", {}, "이번에 배우는 것: "), exp.goal) : null,
      h("p", { class: "muted small" }, `preset: ${payload.preset || "사용자 변경"} · ${payload.provenance ? payload.provenance.timestamp : ""}`)
    )
  );
  const body = h("div", {});
  host.append(body);
  renderResult(body, payload, {});
  if (exp.student || exp.expert) {
    host.append(
      h(
        "section",
        { class: "card" },
        h("h3", {}, "해설"),
        exp.student ? h("details", { open: true }, h("summary", {}, "학생 설명"), h("p", {}, exp.student)) : null,
        exp.expert ? h("details", {}, h("summary", {}, "전문가 확장"), h("p", {}, exp.expert)) : null,
        exp.customer_ko ? h("details", {}, h("summary", {}, "고객에게 어떻게 말할까"), h("p", {}, exp.customer_ko), exp.customer_en ? h("p", { lang: "en" }, exp.customer_en) : null) : null
      )
    );
  }
  host.append(h("footer", { class: "muted small", html: esc("이 보고서는 합성 입력으로 만든 학습용 시뮬레이션 결과다. 실제 부품·하드웨어 검증·면접 준비 완료를 뜻하지 않는다.") }));
}
