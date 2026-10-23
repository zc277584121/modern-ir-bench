import assert from "node:assert/strict";
import test from "node:test";
import {
  comparableProfile,
  evaluateScope,
  isEnglishOnly,
  sortEntries,
  validatePayload,
} from "./scope.js";
import { payloadFixture, textContext, pdfContext } from "./test/fixtures.js";

const ids = (scope) => scope.entries.map(({ solution }) => solution.id);
const english = { breakdownId: "language", valueId: "en" };
const chinese = { breakdownId: "language", valueId: "zh" };

for (const context of [textContext, pdfContext]) {
  test(`${context.task_id}: overall/en/zh have independent eligible rankings`, () => {
    const payload = validatePayload(payloadFixture());
    const overall = evaluateScope(payload, context);
    const en = evaluateScope(payload, context, english);
    const zh = evaluateScope(payload, context, chinese);
    assert.deepEqual(ids(overall), ["b1", "b2", "zero"]);
    assert.deepEqual(ids(en), ["e1", "b2", "b1", "e2", "slice", "zero"]);
    assert.deepEqual(ids(zh), ["b1", "b2", "zero"]);
    assert.equal(overall.ranks.get("b1"), 1);
    assert.equal(en.ranks.get("b1"), 3);
    assert.equal(zh.ranks.get("b1"), 1);
    assert.deepEqual(
      [
        overall.excludedEnglishCount,
        en.excludedEnglishCount,
        zh.excludedEnglishCount,
      ],
      [2, 0, 2],
    );
    assert.equal(en.entries.at(-1).primaryRecord.value, 0);
  });
}

test("tag counts and multi-select OR filtering use comparable scope candidates", () => {
  const payload = payloadFixture();
  assert.equal(
    evaluateScope(payload, textContext).tagCounts.has("english-only"),
    false,
  );
  assert.equal(
    evaluateScope(payload, textContext, english).tagCounts.get("english-only"),
    2,
  );
  const selected = evaluateScope(payload, textContext, {
    ...english,
    tags: new Set(["sparse", "lexical"]),
  });
  assert.deepEqual(ids(selected), ["e1", "b2"]);
  assert.deepEqual(
    [...selected.ranks],
    [
      ["e1", 1],
      ["b2", 2],
    ],
  );
  assert.deepEqual(
    ids(
      evaluateScope(payload, textContext, {
        ...chinese,
        tags: new Set(["english-only"]),
      }),
    ),
    [],
  );
});

test("sorting uses actual scores and leaves scope primary ranks intact", () => {
  const scope = evaluateScope(payloadFixture(), textContext, english);
  assert.equal(
    sortEntries(scope.entries, { key: "recall@10", direction: "asc" })[0]
      .solution.id,
    "zero",
  );
  assert.equal(
    sortEntries(scope.entries, { key: "solution", direction: "asc" })[0]
      .solution.id,
    "b1",
  );
  assert.equal(scope.ranks.get("e1"), 1);
});

for (const value of [null, undefined, NaN, Infinity, "0.8"]) {
  test(`missing/non-numeric score ${String(value)} is never a zero or a ranked row`, () => {
    const payload = payloadFixture();
    payload.results.find((row) => row.solution_id === "b1").value = value;
    const scope = evaluateScope(payload, textContext);
    assert.deepEqual(ids(scope), ["b2", "zero"]);
    assert.deepEqual(scope.incomplete, ["b1"]);
    assert.equal(scope.tagCounts.get("dense"), 1);
  });
}

test("a missing secondary score excludes the row even when filtering by tags", () => {
  const payload = payloadFixture();
  payload.results = payload.results.filter(
    (row) =>
      !(
        row.task_id === textContext.task_id &&
        row.solution_id === "b1" &&
        !row.primary
      ),
  );
  assert.deepEqual(
    ids(evaluateScope(payload, textContext, { tags: new Set(["dense"]) })),
    ["zero"],
  );
});

test("separate single-language evidence cannot imply a complete overall evaluation", () => {
  const payload = payloadFixture();
  const scope = payload.coverage[0];
  const slice = scope.solutions.find((item) => item.solution_id === "slice");
  slice.evidence.push({ ...slice.evidence[0], languages: ["zh"] });
  assert.equal(
    ids(evaluateScope(payload, textContext)).includes("slice"),
    false,
  );
});

test("profile only includes series with a real score on every displayed axis", () => {
  const payload = payloadFixture();
  const solutions = new Map(
    payload.solutions.map((solution) => [solution.id, solution]),
  );
  const values = payload.breakdowns[0].values;
  const profile = comparableProfile(
    values,
    ["b1", "e1", "zero"],
    "ndcg@10",
    solutions,
  );
  assert.deepEqual(profile.comparable, ["b1", "zero"]);
  assert.deepEqual(profile.incomplete, [{ id: "e1", missing: ["Chinese"] }]);
  values[0].results.find(
    (row) => row.solution_id === "b1" && row.primary,
  ).value = null;
  assert.deepEqual(
    comparableProfile(values, ["b1", "zero"], "ndcg@10", solutions).comparable,
    ["zero"],
  );
});

test("language badges depend on explicit coverage, not legacy tags or model names", () => {
  assert.equal(isEnglishOnly({ languages: ["en"], tags: [] }), true);
  assert.equal(
    isEnglishOnly({ languages: ["en", "zh"], tags: ["english-only"] }),
    false,
  );
});

test("payload rejects absent coverage rather than trusting old bilingual aggregates", () => {
  const payload = payloadFixture();
  delete payload.coverage;
  assert.throws(
    () => validatePayload(payload),
    /Regenerate data\/results.json/,
  );
});

for (const [label, mutate, message] of [
  [
    "unknown context",
    (p) => {
      p.results[0].dataset_version = "other";
    },
    /no coverage scope/,
  ],
  [
    "duplicate metric",
    (p) => {
      p.results.push(p.results[0]);
    },
    /duplicate metric/,
  ],
  [
    "unsupported evidence",
    (p) => {
      p.coverage[0].solutions[3].evidence[0].languages = ["en", "zh"];
    },
    /exceeds Solution language coverage/,
  ],
  [
    "missing languages",
    (p) => {
      delete p.solutions[0].languages;
    },
    /missing language coverage/,
  ],
  [
    "bad checksum",
    (p) => {
      p.coverage[0].solutions[0].evidence[0].rankings_sha256 = "wrong";
    },
    /checksum/,
  ],
  [
    "missing scope languages",
    (p) => {
      delete p.breakdowns[0].values[0].required_languages;
    },
    /invalid scope languages/,
  ],
  [
    "invalid overall metric",
    (p) => {
      p.results.push({ ...p.results[0], solution_id: "e1" });
    },
    /metric exceeds its Solution scope/,
  ],
  [
    "invalid Chinese metric",
    (p) => {
      p.breakdowns[0].values[1].results.push({
        ...p.results[0],
        solution_id: "e1",
      });
    },
    /metric exceeds its Solution scope/,
  ],
]) {
  test(`coverage contract rejects ${label}`, () => {
    const payload = payloadFixture();
    mutate(payload);
    assert.throws(() => validatePayload(payload), message);
  });
}
