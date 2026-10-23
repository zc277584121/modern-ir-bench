const contextKey = (value) =>
  JSON.stringify([value.task_id, value.dataset_id, value.dataset_version]);

const includesLanguages = (languages, required) =>
  required.every((language) => languages.includes(language));

export const isEnglishOnly = (solution) =>
  solution.languages.length === 1 && solution.languages[0] === "en";

function languageList(value) {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    new Set(value).size === value.length &&
    value.every((language) => language === "en" || language === "zh")
  );
}

export function validatePayload(payload) {
  const check = (condition, message) => {
    if (!condition) throw new Error(`Invalid benchmark data: ${message}`);
  };
  check(payload.schema_version === 1, "unsupported payload schema.");
  for (const field of ["tasks", "solutions", "results", "breakdowns"]) {
    check(Array.isArray(payload[field]), `missing ${field}.`);
  }
  check(
    Array.isArray(payload.coverage) && payload.coverage.length > 0,
    "coverage is missing. Regenerate data/results.json with the current publisher.",
  );
  const solutions = new Map(
    payload.solutions.map((solution) => [solution.id, solution]),
  );
  check(
    solutions.size === payload.solutions.length,
    "duplicate Solution metadata.",
  );
  for (const solution of solutions.values()) {
    check(
      languageList(solution.languages),
      `missing language coverage for ${solution.id}.`,
    );
  }
  const tasks = new Map(payload.tasks.map((task) => [task.id, task]));
  const contexts = new Map(
    payload.coverage.map((scope) => [contextKey(scope), scope]),
  );
  check(contexts.size === payload.coverage.length, "duplicate coverage scope.");
  const validateExclusions = (scope, ids) => {
    check(languageList(scope.required_languages), "invalid scope languages.");
    check(Array.isArray(scope.excluded_solutions), "missing scope exclusions.");
    const excluded = new Set();
    for (const item of scope.excluded_solutions) {
      check(
        ids.has(item.solution_id) && !excluded.has(item.solution_id),
        "invalid scope exclusion.",
      );
      check(
        ["unsupported-language", "missing-scope-evidence"].includes(
          item.reason,
        ),
        "invalid exclusion reason.",
      );
      excluded.add(item.solution_id);
    }
  };
  for (const scope of contexts.values()) {
    check(tasks.has(scope.task_id), "unknown coverage Task.");
    const ids = new Set(scope.solutions.map((item) => item.solution_id));
    check(ids.size === scope.solutions.length, "duplicate Solution coverage.");
    validateExclusions(scope, ids);
    for (const item of scope.solutions) {
      const solution = solutions.get(item.solution_id);
      check(solution && item.evidence.length > 0, "missing Solution evidence.");
      for (const evidence of item.evidence) {
        check(
          languageList(evidence.languages) &&
            includesLanguages(solution.languages, evidence.languages),
          "evidence exceeds Solution language coverage.",
        );
        check(
          Number.isInteger(evidence.queries) &&
            evidence.queries > 0 &&
            /^[a-f0-9]{64}$/.test(evidence.rankings_sha256),
          "invalid evidence count or checksum.",
        );
      }
    }
  }
  const validateRows = (rows, scope, context) => {
    const ids = new Set(context.solutions.map((item) => item.solution_id));
    const seen = new Set();
    for (const row of rows) {
      const key = JSON.stringify([row.solution_id, row.metric_id]);
      check(
        ids.has(row.solution_id) && !seen.has(key),
        "unknown Solution or duplicate metric in scope.",
      );
      seen.add(key);
      // Ineligible rows are also filtered at selection time. Reject them here so a
      // stale or malformed publication cannot silently leak into any visualization.
      check(
        eligibleSolution(solutions.get(row.solution_id), context, scope),
        "metric exceeds its Solution scope.",
      );
    }
  };
  const overall = new Map();
  for (const row of payload.results) {
    const key = contextKey(row);
    check(contexts.has(key), "metric has no coverage scope.");
    if (!overall.has(key)) overall.set(key, []);
    overall.get(key).push(row);
  }
  for (const [key, rows] of overall)
    validateRows(rows, contexts.get(key), contexts.get(key));
  const breakdowns = new Set();
  for (const breakdown of payload.breakdowns) {
    const context = contexts.get(contextKey(breakdown));
    const key = JSON.stringify([contextKey(breakdown), breakdown.id]);
    check(
      context && !breakdowns.has(key),
      "unknown or duplicate breakdown scope.",
    );
    breakdowns.add(key);
    const ids = new Set(context.solutions.map((item) => item.solution_id));
    const values = new Set();
    for (const value of breakdown.values) {
      check(!values.has(value.id), "duplicate breakdown value.");
      values.add(value.id);
      validateExclusions(value, ids);
      validateRows(value.results, value, context);
    }
  }
  return payload;
}

function eligibleSolution(solution, coverage, scope) {
  return (
    includesLanguages(solution.languages, scope.required_languages) &&
    !scope.excluded_solutions.some(
      (item) => item.solution_id === solution.id,
    ) &&
    coverage.solutions
      .find((item) => item.solution_id === solution.id)
      ?.evidence.some((evidence) =>
        includesLanguages(evidence.languages, scope.required_languages),
      )
  );
}

export function matchesTags(solution, tags) {
  return tags.size === 0 || (solution.tags || []).some((tag) => tags.has(tag));
}

export function evaluateScope(
  payload,
  context,
  { breakdownId, valueId, tags = new Set() } = {},
) {
  const key = contextKey(context);
  const coverage = payload.coverage.find((item) => contextKey(item) === key);
  if (!coverage)
    throw new Error("No coverage for the selected Task and Dataset.");
  const solutions = new Map(
    payload.solutions.map((solution) => [solution.id, solution]),
  );
  const task = payload.tasks.find((item) => item.id === context.task_id);
  const breakdown = breakdownId
    ? payload.breakdowns.find(
        (item) => contextKey(item) === key && item.id === breakdownId,
      )
    : null;
  const breakdownValue = breakdown?.values.find(
    (value) => value.id === valueId,
  );
  if (breakdownId && !breakdownValue)
    throw new Error("Unknown breakdown scope.");
  const scope = breakdownValue || coverage;
  const primaryMetric = breakdown?.primary_metric || task.primary_metric;
  const selected = (
    breakdownValue?.results ||
    payload.results.filter((row) => contextKey(row) === key)
  ).filter((row) =>
    eligibleSolution(solutions.get(row.solution_id), coverage, scope),
  );
  const labels = new Map(
    selected.map((row) => [row.metric_id, row.metric_label]),
  );
  if (!labels.has(primaryMetric)) labels.set(primaryMetric, primaryMetric);
  const metrics = [...labels]
    .map(([id, label]) => ({ id, label }))
    .sort(
      (left, right) =>
        Number(right.id === primaryMetric) - Number(left.id === primaryMetric),
    );
  const grouped = new Map();
  for (const row of selected) {
    if (!grouped.has(row.solution_id)) grouped.set(row.solution_id, new Map());
    grouped.get(row.solution_id).set(row.metric_id, row);
  }
  const comparable = [];
  const incomplete = [];
  for (const [id, records] of grouped) {
    const solution = solutions.get(id);
    if (
      metrics.some((metric) => !Number.isFinite(records.get(metric.id)?.value))
    ) {
      incomplete.push(id);
      continue;
    }
    comparable.push({
      solution,
      records,
      primaryRecord: records.get(primaryMetric),
    });
  }
  comparable.sort(
    (left, right) =>
      right.primaryRecord.value - left.primaryRecord.value ||
      left.solution.id.localeCompare(right.solution.id),
  );
  const tagCounts = new Map();
  for (const { solution } of comparable) {
    for (const tag of new Set(solution.tags || []))
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
  }
  const entries = comparable.filter(({ solution }) =>
    matchesTags(solution, tags),
  );
  return {
    breakdown,
    breakdownValue,
    primaryMetric,
    metrics,
    entries,
    tagCounts,
    incomplete,
    ranks: new Map(
      entries.map(({ solution }, index) => [solution.id, index + 1]),
    ),
    excludedEnglishCount: coverage.solutions.filter(
      ({ solution_id: id }) =>
        isEnglishOnly(solutions.get(id)) &&
        !includesLanguages(
          solutions.get(id).languages,
          scope.required_languages,
        ),
    ).length,
  };
}

export function sortEntries(entries, { key, direction }) {
  return [...entries].sort((left, right) => {
    const comparison =
      key === "solution"
        ? left.solution.title.localeCompare(right.solution.title)
        : left.records.get(key).value - right.records.get(key).value;
    return (
      (direction === "asc" ? comparison : -comparison) ||
      left.solution.id.localeCompare(right.solution.id)
    );
  });
}

export function comparableProfile(values, solutionIds, metricId, solutions) {
  const comparable = [];
  const incomplete = [];
  for (const id of solutionIds) {
    const missing = values.filter(
      (value) =>
        !includesLanguages(
          solutions.get(id).languages,
          value.required_languages,
        ) ||
        value.excluded_solutions.some((item) => item.solution_id === id) ||
        !Number.isFinite(
          value.results.find(
            (row) => row.solution_id === id && row.metric_id === metricId,
          )?.value,
        ),
    );
    if (missing.length)
      incomplete.push({ id, missing: missing.map((value) => value.label) });
    else comparable.push(id);
  }
  return { comparable, incomplete };
}
