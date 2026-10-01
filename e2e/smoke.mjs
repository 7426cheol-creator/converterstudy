// End-to-end smoke test of the Converter FAE Lab UI in a real (headless) Chromium.
//
// For every implemented experiment: open it, run the baseline preset, apply the suggested change, pick a
// prediction and run again. Then check the result area (status, metrics, plots or tables), and collect every
// page error and console error. It also checks the answer-save flow, the CSV/HTML/JSON exports and the
// progress page. Exit code 1 on any failure.
//
//   node smoke.mjs [--spawn] [--quick] [--only=FL01,EX05] [--screenshots=dir] [--url=http://127.0.0.1:8765]
//
// --spawn starts `python -m convlab serve` itself (python from $PYTHON, default "python") with a throw-away
// progress file; otherwise the server must already be running at --url.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const arg = (name, def = null) => {
  const a = process.argv.find((x) => x === `--${name}` || x.startsWith(`--${name}=`));
  if (!a) return def;
  return a.includes("=") ? a.slice(a.indexOf("=") + 1) : true;
};
const PORT = 8765 + Math.floor(Math.random() * 1000);
let BASE = arg("url", null);
const quick = !!arg("quick");
const only = (arg("only", "") || "").split(",").filter(Boolean);
const shotDir = arg("screenshots", null);
const exe = process.env.PW_CHROMIUM || (fs.existsSync("/opt/pw-browsers/chromium-1194/chrome-linux/chrome") ? "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" : undefined);

let server = null;
async function waitFor(url, ms = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("server did not come up: " + url);
}

if (arg("spawn")) {
  BASE = `http://127.0.0.1:${PORT}`;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "convlab-e2e-"));
  server = spawn(process.env.PYTHON || "python", ["-m", "convlab", "serve", "--port", String(PORT), "--no-browser"], {
    env: { ...process.env, CONVLAB_USERDATA: path.join(tmp, "progress.json"), OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1" },
    stdio: ["ignore", "inherit", "inherit"],
  });
  await waitFor(BASE + "/api/meta");
}
BASE = BASE || "http://127.0.0.1:8765";

const failures = [];
const fail = (where, msg) => { failures.push(`${where}: ${msg}`); console.log(`  FAIL ${where}: ${msg}`); };

const browser = await chromium.launch(exe ? { executablePath: exe } : {});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
let pageErrors = [];
page.on("pageerror", (e) => pageErrors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") pageErrors.push("console: " + m.text()); });

const labs = await (await fetch(BASE + "/api/labs")).json();
let runs = 0;
const t0 = Date.now();
for (const lab of labs) {
  if (only.length && !only.includes(lab.id)) continue;
  const detail = await (await fetch(`${BASE}/api/labs/${lab.id}`)).json();
  const exps = quick ? detail.experiments.slice(0, 1) : detail.experiments;
  for (const exp of exps) {
    const where = `${lab.id}/${exp.key}`;
    pageErrors = [];
    const ts = Date.now();
    try {
      await page.goto(`${BASE}/#/lab/${lab.id}/${exp.key}`, { waitUntil: "networkidle" });
      await page.waitForSelector("button.btn.primary.big", { timeout: 15000 });
      await page.click("button.btn.primary.big");
      await page.waitForSelector(".result .verdicts", { timeout: 120000 });
      const status = (await page.locator(".verdicts .chip.status").first().textContent()) || "";
      if (!status.trim()) fail(where, "empty status chip");
      const nPlots = await page.locator(".result figure.plot").count();
      const nTables = await page.locator(".result table").count();
      const nMetrics = await page.locator(".result .metric, .result tr").count();
      if (nPlots + nTables === 0) fail(where, "no plot and no table in the result");
      if (nMetrics === 0) fail(where, "no metrics");
      // layout: a long text value must not squeeze the metric name column (seen once: 27 px wide, 6000 px tall card)
      const squeezed = await page.evaluate(() => {
        const t = document.querySelector(".result table.metrics");
        if (!t) return null;
        const first = t.querySelector("tbody tr td");
        return first && first.getBoundingClientRect().width < 80 ? Math.round(first.getBoundingClientRect().width) : null;
      });
      if (squeezed !== null) fail(where, `metric name column squeezed to ${squeezed} px`);
      const err = await page.locator(".err").count();
      if (err) fail(where, "error box: " + (await page.locator(".err").first().textContent()));
      // prediction flow with the suggested change
      if (await page.locator("button.btn.accent").count()) {
        await page.click("button.btn.accent");
        await page.waitForTimeout(150);
        if (await page.locator(".opts .opt").count()) {
          await page.locator(".opts .opt").first().click();
          await page.click("button.btn.primary.big");
          await page.waitForFunction(() => !document.querySelector("button.btn.primary.big")?.disabled, null, { timeout: 120000 });
          await page.waitForSelector(".result .verdicts", { timeout: 120000 });
          if (await page.locator(".err").count()) fail(where, "error after the suggested change: " + (await page.locator(".err").first().textContent()));
        } else {
          fail(where, "suggested change applied but no prediction options appeared");
        }
      }
      if (shotDir) {
        fs.mkdirSync(shotDir, { recursive: true });
        await page.locator(".result-box").screenshot({ path: path.join(shotDir, `${lab.id}_${exp.key}.png`) }).catch(() => {});
      }
      runs++;
      console.log(`ok  ${where.padEnd(34)} ${status.trim().padEnd(28)} ${(Date.now() - ts) / 1000}s`);
    } catch (e) {
      fail(where, e.message.split("\n")[0]);
    }
    for (const pe of pageErrors) fail(where, pe);
  }
}

// answer save, exports and the progress page on the first lab
try {
  const first = labs.find((l) => !only.length || only.includes(l.id)) || labs[0];
  const d = await (await fetch(`${BASE}/api/labs/${first.id}`)).json();
  const exp = d.experiments.find((e) => (e.questions || []).length) || d.experiments[0];
  await page.goto(`${BASE}/#/lab/${first.id}/${exp.key}`, { waitUntil: "networkidle" });
  if ((exp.questions || []).length) {
    const ta = page.locator("textarea[placeholder^='내 답']").first();
    await ta.fill("e2e 답변: 가설·측정·판단 순서로 설명");
    await page.locator("button:has-text('내 답 저장')").first().click();
    await page.waitForTimeout(400);
    const prog = await (await fetch(BASE + "/api/progress")).json();
    const txt = JSON.stringify(prog);
    if (!txt.includes("e2e 답변")) fail("answers", "saved answer not found in /api/progress");
  }
  for (const format of ["csv", "html", "json"]) {
    const r = await fetch(BASE + "/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: BASE },
      body: JSON.stringify({ lab: first.id, experiment: exp.key, preset: exp.presets[0].key, values: {}, format }),
    });
    const body = await r.text();
    if (!r.ok || body.length < 100) fail("export", `${format}: HTTP ${r.status}, ${body.length} bytes`);
  }
  pageErrors = [];
  await page.goto(`${BASE}/#/progress`, { waitUntil: "networkidle" });
  await page.waitForTimeout(300);
  for (const pe of pageErrors) fail("progress page", pe);
} catch (e) {
  fail("flows", e.message.split("\n")[0]);
}

await browser.close();
if (server) server.kill();
console.log(`\n${runs} experiments ran in ${((Date.now() - t0) / 1000).toFixed(0)} s, ${failures.length} failure(s)`);
if (failures.length) {
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
}
