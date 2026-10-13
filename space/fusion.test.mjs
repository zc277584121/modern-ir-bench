import assert from "node:assert/strict";
import test from "node:test";

import { atomicRanking, evaluateRankings, fuseRankings } from "./fusion.js";

const documents = ["a", "b", "c", "d"];

test("RRF combines weighted atomic rankings", () => {
  const ranking = fuseRankings(
    [
      { ranking: [0, 3, 1, 2, 2, 1], weight: 1, scoreDirection: "higher" },
      { ranking: [1, 3, 3, 2, 0, 1], weight: 1, scoreDirection: "higher" },
    ],
    { method: "rrf", depth: 3, rankConstant: 0, documents, topK: 3 },
  );
  assert.deepEqual(ranking, [1, 0, 3]);
});

test("weighted fusion respects lower-is-better scores", () => {
  const ranking = fuseRankings(
    [
      { ranking: [0, -10, 1, -5, 2, -1], weight: 0.75, scoreDirection: "lower" },
      { ranking: [1, 1, 2, 0.5, 0, 0], weight: 0.25, scoreDirection: "higher" },
    ],
    { method: "weighted", depth: 3, rankConstant: 60, documents, topK: 3 },
  );
  assert.deepEqual(ranking, [0, 1, 2]);
});

test("metric calculation matches binary ranked retrieval semantics", () => {
  const queries = [
    { relevant: [0, 2] },
    { relevant: [3] },
  ];
  const result = evaluateRankings(queries, [[0, 1], [1, 3]]);
  assert.equal(result.count, 2);
  assert.equal(result.metrics["recall@1"], 0.25);
  assert.equal(result.metrics["recall@5"], 0.75);
  assert.equal(result.metrics["mrr@10"], 0.75);
  assert.ok(Math.abs(result.metrics["ndcg@10"] - 0.62203847) < 1e-8);
});

test("atomic rankings unpack the requested depth", () => {
  assert.deepEqual(atomicRanking([3, 0.9, 2, 0.8, 1, 0.7], 2), [3, 2]);
});
