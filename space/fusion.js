export const metricCatalog = [
  { id: "ndcg@10", label: "NDCG@10" },
  { id: "recall@1", label: "Recall@1" },
  { id: "recall@5", label: "Recall@5" },
  { id: "recall@10", label: "Recall@10" },
  { id: "mrr@10", label: "MRR@10" },
];

export function atomicRanking(packed, depth = 10) {
  const documents = [];
  for (let index = 0; index < Math.min(packed.length, depth * 2); index += 2) {
    documents.push(packed[index]);
  }
  return documents;
}

export function fuseRankings(routeInputs, {
  method,
  depth,
  rankConstant,
  documents,
  topK = 10,
}) {
  const scores = new Map();
  routeInputs.forEach(({ ranking, weight, scoreDirection }) => {
    const hits = [];
    for (let index = 0; index < Math.min(ranking.length, depth * 2); index += 2) {
      hits.push({ document: ranking[index], score: ranking[index + 1] });
    }
    if (method === "rrf") {
      hits.forEach((hit, index) => {
        scores.set(
          hit.document,
          (scores.get(hit.document) || 0) + weight / (rankConstant + index + 1),
        );
      });
      return;
    }

    const values = hits.map((hit) => hit.score);
    const low = Math.min(...values);
    const high = Math.max(...values);
    hits.forEach((hit) => {
      const normalized = high === low
        ? 0
        : scoreDirection === "lower"
          ? (high - hit.score) / (high - low)
          : (hit.score - low) / (high - low);
      scores.set(hit.document, (scores.get(hit.document) || 0) + weight * normalized);
    });
  });
  return [...scores.entries()]
    .sort((left, right) => right[1] - left[1] || documents[left[0]].localeCompare(documents[right[0]]))
    .slice(0, topK)
    .map(([document]) => document);
}

export function evaluateRankings(queries, rankings, selectedIndexes = null) {
  const totals = Object.fromEntries(metricCatalog.map((metric) => [metric.id, 0]));
  const indexes = selectedIndexes || queries.map((_, index) => index);
  if (indexes.length === 0) throw new Error("The selected query slice is empty.");
  indexes.forEach((queryIndex) => {
    const relevant = new Set(queries[queryIndex].relevant);
    const ranking = rankings[queryIndex].slice(0, 10);
    [1, 5, 10].forEach((cutoff) => {
      const hits = new Set(ranking.slice(0, cutoff).filter((document) => relevant.has(document)));
      totals[`recall@${cutoff}`] += hits.size / relevant.size;
    });
    let dcg = 0;
    let reciprocalRank = 0;
    ranking.forEach((document, index) => {
      if (!relevant.has(document)) return;
      dcg += 1 / Math.log2(index + 2);
      if (!reciprocalRank) reciprocalRank = 1 / (index + 1);
    });
    let ideal = 0;
    for (let rank = 1; rank <= Math.min(relevant.size, 10); rank += 1) {
      ideal += 1 / Math.log2(rank + 1);
    }
    totals["ndcg@10"] += dcg / ideal;
    totals["mrr@10"] += reciprocalRank;
  });
  Object.keys(totals).forEach((metricId) => {
    totals[metricId] /= indexes.length;
  });
  return { count: indexes.length, metrics: totals };
}
