const badgeDefinitions = {
  atomic: {
    label: "Atomic",
    description: "Uses one retrieval route",
  },
  fusion: {
    label: "Fusion",
    description: "Combines multiple retrieval routes",
  },
  reranked: {
    label: "Reranked",
    description: "Retrieves candidates and reranks them in a second stage",
  },
  dense: {
    label: "Dense",
    description: "Dense embedding retrieval",
  },
  sparse: {
    label: "Sparse",
    description: "Learned sparse retrieval",
  },
  lexical: {
    label: "Lexical",
    description: "Lexical retrieval such as BM25",
  },
  visual: {
    label: "Visual",
    description: "Retrieval from page images or multimodal representations",
  },
  ocr: {
    label: "OCR",
    description: "Extracts document text with OCR before retrieval",
  },
  rrf: {
    label: "RRF",
    description: "Reciprocal rank fusion",
  },
  weighted: {
    label: "Weighted",
    description: "Weighted score fusion",
  },
};

function badge(id) {
  return { id, ...badgeDefinitions[id] };
}

export function solutionBadges(solution) {
  const tags = new Set(solution.tags || []);
  const badges = [];

  if (tags.has("reranker")) {
    badges.push(badge("reranked"));
  } else if (tags.has("hybrid")) {
    badges.push(badge("fusion"));
  } else {
    badges.push(badge("atomic"));
  }

  if (tags.has("reranker") && tags.has("hybrid")) badges.push(badge("fusion"));
  if (tags.has("dense") && !tags.has("visual")) badges.push(badge("dense"));
  if (tags.has("learned-sparse")) badges.push(badge("sparse"));
  if (tags.has("lexical")) badges.push(badge("lexical"));
  if (tags.has("visual")) badges.push(badge("visual"));
  if (tags.has("ocr")) badges.push(badge("ocr"));

  if (tags.has("rrf")) badges.push(badge("rrf"));
  if (tags.has("weighted")) badges.push(badge("weighted"));

  return badges;
}
