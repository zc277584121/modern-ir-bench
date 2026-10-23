export const textContext = {
  task_id: "modern-ir-ranked-retrieval-v2.2",
  dataset_id: "modern-ir-bench-v2.2",
  dataset_version: "revision",
};
export const pdfContext = {
  task_id: "modern-ir-pdf-retrieval-v1",
  dataset_id: "modern-ir-bench-pdf-v1",
  dataset_version: "pdf-revision",
};

export function payloadFixture() {
  const solutions = [
    {
      id: "b1",
      title: "Bilingual One",
      languages: ["en", "zh"],
      tags: ["dense"],
    },
    {
      id: "b2",
      title: "Bilingual Two",
      languages: ["en", "zh"],
      tags: ["lexical"],
    },
    {
      id: "zero",
      title: "Measured Zero",
      languages: ["en", "zh"],
      tags: ["dense"],
    },
    {
      id: "e1",
      title: "English One",
      languages: ["en"],
      tags: ["english-only", "sparse"],
    },
    {
      id: "e2",
      title: "English Two",
      languages: ["en"],
      tags: ["english-only", "visual"],
    },
    {
      id: "slice",
      title: "Bilingual Slice",
      languages: ["en", "zh"],
      tags: ["dense"],
    },
  ];
  const rows = (context, scores) =>
    Object.entries(scores).flatMap(([solution_id, value]) => [
      {
        ...context,
        solution_id,
        value,
        metric_id: "ndcg@10",
        metric_label: "NDCG@10",
        primary: true,
      },
      {
        ...context,
        solution_id,
        value: Math.min(1, value + 0.05),
        metric_id: "recall@10",
        metric_label: "Recall@10",
        primary: false,
      },
    ]);
  const exclusions = (languages, included) =>
    solutions
      .filter((solution) => !included.includes(solution.id))
      .map((solution) => ({
        solution_id: solution.id,
        reason: languages.every((language) =>
          solution.languages.includes(language),
        )
          ? "missing-scope-evidence"
          : "unsupported-language",
      }));
  const payload = {
    schema_version: 1,
    tasks: [],
    solutions,
    coverage: [],
    results: [],
    breakdowns: [],
  };
  for (const context of [textContext, pdfContext]) {
    payload.tasks.push({
      id: context.task_id,
      title: context === textContext ? "Text v2.2" : "PDF v1",
      version: "1",
      primary_metric: "ndcg@10",
    });
    payload.coverage.push({
      ...context,
      required_languages: ["en", "zh"],
      solutions: solutions.map((solution) => ({
        solution_id: solution.id,
        evidence: [
          {
            release: context.dataset_id,
            languages: ["e1", "e2", "slice"].includes(solution.id)
              ? ["en"]
              : ["en", "zh"],
            queries: 500,
            rankings_sha256: "a".repeat(64),
            source: {},
            result_url: "https://example.com/evidence",
          },
        ],
      })),
      excluded_solutions: exclusions(["en", "zh"], ["b1", "b2", "zero"]),
    });
    payload.results.push(...rows(context, { b1: 0.8, b2: 0.7, zero: 0 }));
    const value = (id, label, languages, scores) => ({
      id,
      label,
      count: 500,
      required_languages: languages,
      excluded_solutions: exclusions(languages, Object.keys(scores)),
      results: rows(context, scores),
    });
    payload.breakdowns.push({
      ...context,
      id: "language",
      label: "Language",
      category: "query",
      description: "Query language.",
      primary_metric: "ndcg@10",
      values: [
        value("en", "English", ["en"], {
          b1: 0.8,
          b2: 0.9,
          zero: 0,
          e1: 0.95,
          e2: 0.75,
          slice: 0.6,
        }),
        value("zh", "Chinese", ["zh"], { b1: 0.8, b2: 0.5, zero: 0 }),
      ],
    });
    payload.breakdowns.push({
      ...context,
      id:
        context === textContext
          ? "relevant_document_domain"
          : "relevant_document_transform_tier",
      label: "Groups",
      category: "relevant_documents",
      description: "Document groups.",
      primary_metric: "ndcg@10",
      values: [
        value("a", "Group A", ["en", "zh"], { b1: 0.8, b2: 0.7, zero: 0 }),
        value("b", "Group B", ["en", "zh"], { b1: 0.75, b2: 0.65, zero: 0 }),
        value("c", "Group C", ["en", "zh"], { b1: 0.7, b2: 0.6, zero: 0 }),
      ],
    });
  }
  return payload;
}
