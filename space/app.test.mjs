import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { JSDOM } from "jsdom";
import { payloadFixture, textContext, pdfContext } from "./test/fixtures.js";

const html = await readFile(new URL("./index.html", import.meta.url), "utf8");
let loadCount = 0;
async function loadApp(t, payload = payloadFixture()) {
  const dom = new JSDOM(html, { url: "http://localhost/" });
  dom.window.matchMedia = () => ({ matches: false });
  t.mock.method(globalThis, "fetch", async (path) =>
    Response.json(
      path === "data/results.json"
        ? payload
        : {
            benchmarks: [textContext, pdfContext].map((context) => ({
              ...context,
              routes: [],
              dimensions: [],
              queries: [],
              max_depth: 50,
            })),
          },
    ),
  );
  for (const key of ["window", "document", "Option"])
    globalThis[key] = dom.window[key];
  t.after(async () => {
    await new Promise((resolve) => setImmediate(resolve));
    dom.window.close();
    for (const key of ["window", "document", "Option"]) delete globalThis[key];
  });
  await import(`./app.js?test=${loadCount++}`);
  await new Promise((resolve) => setImmediate(resolve));
  const query = (selector) => dom.window.document.querySelector(selector);
  const all = (selector) => [...dom.window.document.querySelectorAll(selector)];
  const change = (selector, value) => {
    query(selector).value = value;
    query(selector).dispatchEvent(new dom.window.Event("change"));
  };
  const check = (selector, checked) => {
    query(selector).checked = checked;
    query(selector).dispatchEvent(new dom.window.Event("change"));
  };
  if (query("#task-selector").options.length) {
    change("#task-selector", "modern-ir-ranked-retrieval");
  }
  return { query, all, change, check };
}

test("leaderboard, tags, picker, ranks and exclusion link follow overall/en/zh", async (t) => {
  const { query, all, change, check } = await loadApp(t);
  const names = () =>
    all("#leaderboard-table .solution-name").map((node) => node.textContent);
  const pickers = () =>
    all("[data-profile-solution]").map((node) => node.dataset.profileSolution);
  assert.deepEqual(names(), [
    "Bilingual One",
    "Bilingual Two",
    "Measured Zero",
  ]);
  assert.deepEqual(pickers(), ["b1", "b2", "zero"]);
  assert.match(query("#coverage-note-text").textContent, /^2 English-only/);
  assert.equal(query('[data-solution-tag="english-only"]'), null);
  query("#show-english").click();
  assert.equal(query("#breakdown-selector").value, "language");
  assert.equal(query("#breakdown-value-selector").value, "en");
  assert.equal(query("#coverage-note").hidden, true);
  assert.equal(names()[0], "English OneEnglish only");
  assert.equal(all("#leaderboard-table .language-badge").length, 2);
  assert.deepEqual(pickers(), ["e1", "b2", "b1", "e2", "slice", "zero"]);
  assert.equal(
    query('[data-solution-tag="english-only"]').parentElement.querySelector(
      ".tag-option-count",
    ).textContent,
    "2",
  );
  check('[data-solution-tag="sparse"]', true);
  check('[data-solution-tag="lexical"]', true);
  assert.deepEqual(names(), ["English OneEnglish only", "Bilingual Two"]);
  assert.deepEqual(pickers(), ["e1", "b2"]);
  assert.deepEqual(
    all(".rank").map((node) => node.textContent),
    ["1", "2"],
  );
  query('[data-sort-key="recall@10"]').click();
  query('[data-sort-key="recall@10"]').click();
  assert.deepEqual(names(), ["Bilingual Two", "English OneEnglish only"]);
  assert.deepEqual(
    all(".rank").map((node) => node.textContent),
    ["2", "1"],
  );
  change("#breakdown-value-selector", "zh");
  assert.equal(query('[data-solution-tag="sparse"]'), null);
  assert.deepEqual(pickers(), ["b2"]);
  assert.equal(query("#coverage-note").hidden, false);
  query("#clear-solution-tags").click();
  assert.deepEqual(names(), [
    "Bilingual One",
    "Bilingual Two",
    "Measured Zero",
  ]);
  assert.deepEqual(pickers(), ["b1", "b2", "zero"]);
  change("#task-selector", "modern-ir-pdf-retrieval");
  assert.equal(query("#breakdown-selector").value, "");
  assert.equal(query("#coverage-note").hidden, false);
  query("#show-english").click();
  assert.equal(names().length, 6);
  assert.equal(query("#error-notice").hidden, true);
});

test("English-only selections stay available but never draw a fabricated Chinese bar", async (t) => {
  const { query, all, check } = await loadApp(t);
  query("#show-english").click();
  check('[data-profile-solution="e1"]', true);
  assert.match(
    query("#profile-incomplete-note").textContent,
    /English One \(no Chinese score\)/,
  );
  assert.equal(query("#profile-incomplete-note").hidden, false);
  assert.doesNotMatch(query("#profile-chart").innerHTML, /English One/);
  assert.doesNotMatch(query("#profile-legend").textContent, /English One/);
  check('[data-solution-tag="english-only"]', true);
  assert.equal(all("#profile-chart svg").length, 0);
  assert.match(
    query("#profile-chart").textContent,
    /No selected solution has scores/,
  );
  assert.equal(all("[data-profile-solution]").length, 2);
});

test("missing secondary scores remove table rows and picker candidates without placeholders", async (t) => {
  const payload = payloadFixture();
  payload.results.find(
    (row) => row.solution_id === "b1" && !row.primary,
  ).value = null;
  const { query, all } = await loadApp(t, payload);
  assert.equal(all("#leaderboard-table tbody tr").length, 2);
  assert.equal(query('[data-profile-solution="b1"]'), null);
  assert.doesNotMatch(
    query("#leaderboard-table").textContent,
    /Bilingual One|NaN|—/,
  );
  assert.match(query("#leaderboard-table").textContent, /0\.000/);
  assert.equal(query("#incomplete-note").hidden, false);
});

test("a missing radar axis omits only that series and explains incomparability", async (t) => {
  const payload = payloadFixture();
  const groups = payload.breakdowns.find(
    (item) =>
      item.task_id === textContext.task_id &&
      item.id === "relevant_document_domain",
  );
  groups.values[2].results.find(
    (row) => row.solution_id === "b1" && row.primary,
  ).value = null;
  const { query, all } = await loadApp(t, payload);
  assert.equal(all("#profile-chart polygon.chart-series").length, 2);
  assert.doesNotMatch(query("#profile-chart").innerHTML, /Bilingual One/);
  assert.match(
    query("#profile-incomplete-note").textContent,
    /Bilingual One \(no Group C score\)/,
  );
});

test("stale payload fails closed before rendering old overall scores", async (t) => {
  const payload = payloadFixture();
  delete payload.coverage;
  t.mock.method(console, "error", () => {});
  const { query, all } = await loadApp(t, payload);
  assert.equal(query("#error-notice").hidden, false);
  assert.match(
    query("#error-notice").textContent,
    /Regenerate data\/results.json/,
  );
  assert.equal(all("#leaderboard-table tbody tr").length, 0);
});

test("profile never clips or zero-fills a score outside its declared range", async (t) => {
  const payload = payloadFixture();
  const groups = payload.breakdowns.find(
    (item) =>
      item.task_id === textContext.task_id &&
      item.id === "relevant_document_domain",
  );
  groups.values[0].results.find(
    (row) => row.solution_id === "b1" && row.primary,
  ).value = 1.2;
  const { query } = await loadApp(t, payload);
  assert.equal(query("#profile-chart svg"), null);
  assert.match(query("#profile-chart").textContent, /declared chart range/);
  assert.equal(query("#profile-legend").textContent, "");
});

test("an empty scope clears stale picker options and chart series", async (t) => {
  const payload = payloadFixture();
  const language = payload.breakdowns.find(
    (item) => item.task_id === textContext.task_id && item.id === "language",
  );
  language.values
    .find((value) => value.id === "zh")
    .results.forEach((row) => {
      row.value = null;
    });
  const { query, all, change } = await loadApp(t, payload);
  query("#show-english").click();
  assert.equal(all("[data-profile-solution]").length, 6);
  change("#breakdown-value-selector", "zh");
  assert.equal(all("#leaderboard-table tbody tr").length, 0);
  assert.equal(query("#profile-section").hidden, true);
  assert.equal(all("[data-profile-solution]").length, 0);
  assert.equal(query("#profile-chart").textContent, "");
});
