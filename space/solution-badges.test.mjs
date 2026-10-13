import assert from "node:assert/strict";
import test from "node:test";

import { solutionBadges } from "./solution-badges.js";

const badgeIds = (tags) => solutionBadges({ tags }).map((badge) => badge.id);

test("marks a single dense route as atomic", () => {
  assert.deepEqual(badgeIds(["dense", "managed-api", "chunked"]), [
    "atomic",
    "dense",
  ]);
});

test("shows the retrieval signals and algorithm for fusion", () => {
  assert.deepEqual(
    badgeIds(["lexical", "learned-sparse", "hybrid", "rrf"]),
    ["fusion", "sparse", "lexical", "rrf"],
  );
});

test("distinguishes reranked fusion from one-stage fusion", () => {
  assert.deepEqual(badgeIds(["hybrid", "two-stage", "reranker"]), [
    "reranked",
    "fusion",
  ]);
});

test("describes OCR and visual PDF retrieval without operational tags", () => {
  assert.deepEqual(
    badgeIds(["ocr", "visual", "dense", "lexical", "hybrid", "managed-api", "page-level", "weighted"]),
    ["fusion", "lexical", "visual", "ocr", "weighted"],
  );
});
