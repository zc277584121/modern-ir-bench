import { createProfileController } from "./profile.js?v=2";

const state = {
  payload: null,
  tasks: new Map(),
  solutions: new Map(),
  taskSort: { key: null, direction: "desc" },
  solutionSort: { key: "task", direction: "asc" },
  profile: null,
};

const breakdownDisplayByTask = new Map([
  [
    "modern-ir-ranked-retrieval-v2.2",
    {
      ids: new Set([
        "language",
        "relevant_document_domain",
        "relevant_document_form",
        "relevant_document_length",
      ]),
      defaultProfile: "relevant_document_domain",
    },
  ],
]);

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const formatScore = (value) => Number(value).toFixed(3);
const formatInteger = (value) => Number(value).toLocaleString("en-US");

function datasetLabel(datasetId) {
  const release = datasetId.match(/^modern-ir-bench-v(\d+(?:\.\d+)?)/);
  if (release) return `Modern IR Bench v${release[1]}`;
  return datasetId.replace(/-\d{8}$/, "");
}

function solutionCodeLink(solution, fallbackRecord) {
  const url = solution.code_url || fallbackRecord?.source_url;
  if (!url) return "";
  return `<a class="solution-code-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer" aria-label="Open code for ${escapeHtml(solution.title)}" title="Open the code for this Solution">
    <svg aria-hidden="true" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9.5 2.5h4v4" />
      <path d="m7 9 6.5-6.5" />
      <path d="M13 9.5v3a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h3" />
    </svg>
  </a>`;
}

function solutionName(solution, record) {
  return `<span class="solution-name"><button class="solution-detail-link" type="button" data-solution-id="${escapeHtml(solution.id)}" title="${escapeHtml(solution.description)}" aria-label="View results for ${escapeHtml(solution.title)}">${escapeHtml(solution.title)}</button>${solutionCodeLink(solution, record)}</span>`;
}

function renderTable(headers, rows, tableName, sortState) {
  const head = headers
    .map((header) => {
      if (!header.key) return `<th>${escapeHtml(header.label)}</th>`;
      const active = sortState?.key === header.key;
      const arrow = active ? (sortState.direction === "asc" ? "▲" : "▼") : "↕";
      return `<th><button class="sort-button${active ? " active" : ""}" data-table="${tableName}" data-sort-key="${escapeHtml(header.key)}">${escapeHtml(header.label)} <span class="sort-arrow">${arrow}</span></button></th>`;
    })
    .join("");
  const body = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function compareValues(left, right, direction) {
  let comparison;
  if (typeof left === "number" && typeof right === "number") {
    comparison = left - right;
  } else {
    comparison = String(left).localeCompare(String(right));
  }
  return direction === "asc" ? comparison : -comparison;
}

function toggleSort(sortState, key, numeric) {
  if (sortState.key === key) {
    sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
  } else {
    sortState.key = key;
    sortState.direction = numeric ? "desc" : "asc";
  }
}

function installSortButtons(containerSelector, tableName, handler) {
  document
    .querySelector(containerSelector)
    .querySelectorAll(`.sort-button[data-table="${tableName}"]`)
    .forEach((button) => {
      button.addEventListener("click", () => handler(button.dataset.sortKey));
    });
}

function primaryRecords() {
  return state.payload.results.filter((record) => record.primary);
}

function taskDatasets(taskId) {
  const seen = new Set();
  return state.payload.results.flatMap((record) => {
    if (record.task_id !== taskId) return [];
    const value = `${record.dataset_id}|${record.dataset_version}`;
    if (seen.has(value)) return [];
    seen.add(value);
    return [{ value, label: datasetLabel(record.dataset_id) }];
  });
}

function applicableBreakdowns(taskId, datasetId, datasetVersion) {
  const display = breakdownDisplayByTask.get(taskId);
  if (!display) return [];
  return (state.payload.breakdowns || []).filter(
    (breakdown) =>
      breakdown.task_id === taskId &&
      breakdown.dataset_id === datasetId &&
      breakdown.dataset_version === datasetVersion &&
      display.ids.has(breakdown.id),
  );
}

function breakdownCategoryLabel(category) {
  if (!category || category === "query") return "Query";
  if (category === "relevant_documents") return "Relevant docs";
  return category
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function populateBreakdownSelector() {
  const taskId = document.querySelector("#task-selector").value;
  const [datasetId, datasetVersion] = document
    .querySelector("#dataset-selector")
    .value.split("|");
  const selector = document.querySelector("#breakdown-selector");
  selector.replaceChildren(new Option("Overall", ""));
  applicableBreakdowns(taskId, datasetId, datasetVersion).forEach((breakdown) => {
    const category = breakdown.category || "query";
    const prefix = breakdownCategoryLabel(category);
    selector.add(new Option(`${prefix} · ${breakdown.label}`, breakdown.id));
  });
  populateBreakdownValueSelector();
}

function selectedBreakdown() {
  const breakdownId = document.querySelector("#breakdown-selector").value;
  if (!breakdownId) return null;
  const taskId = document.querySelector("#task-selector").value;
  const [datasetId, datasetVersion] = document
    .querySelector("#dataset-selector")
    .value.split("|");
  return applicableBreakdowns(taskId, datasetId, datasetVersion).find(
    (breakdown) => breakdown.id === breakdownId,
  );
}

function populateBreakdownValueSelector() {
  const breakdown = selectedBreakdown();
  const control = document.querySelector("#breakdown-value-control");
  const selector = document.querySelector("#breakdown-value-selector");
  const label = document.querySelector("#breakdown-value-label");
  selector.replaceChildren();
  control.hidden = !breakdown;
  if (!breakdown) return;
  label.textContent = breakdown.label;
  breakdown.values.forEach((value) => {
    selector.add(new Option(value.label, value.id));
  });
}

function selectedBreakdownValue() {
  const breakdown = selectedBreakdown();
  if (!breakdown) return null;
  const valueId = document.querySelector("#breakdown-value-selector").value;
  return breakdown.values.find((value) => value.id === valueId) || null;
}

function populateDatasetSelector() {
  const selector = document.querySelector("#dataset-selector");
  selector.replaceChildren();
  taskDatasets(document.querySelector("#task-selector").value).forEach((option) =>
    selector.add(new Option(option.label, option.value)),
  );
}

function currentTaskDataset() {
  const taskId = document.querySelector("#task-selector").value;
  const [datasetId, datasetVersion] = document
    .querySelector("#dataset-selector")
    .value.split("|");
  return { taskId, datasetId, datasetVersion };
}

function currentOverallPrimaryRecords() {
  const { taskId, datasetId, datasetVersion } = currentTaskDataset();
  return primaryRecords()
    .filter(
      (record) =>
        record.task_id === taskId &&
        record.dataset_id === datasetId &&
        record.dataset_version === datasetVersion,
    )
    .sort(
      (left, right) =>
        right.value - left.value || left.solution_id.localeCompare(right.solution_id),
    );
}

function renderTask() {
  const taskId = document.querySelector("#task-selector").value;
  const [datasetId, datasetVersion] = document
    .querySelector("#dataset-selector")
    .value.split("|");
  const task = state.tasks.get(taskId);
  const overallResults = state.payload.results.filter(
    (record) =>
      record.task_id === taskId &&
      record.dataset_id === datasetId &&
      record.dataset_version === datasetVersion,
  );
  const breakdown = selectedBreakdown();
  const breakdownValue = selectedBreakdownValue();
  const selected = breakdownValue?.results || overallResults;
  const primaryMetric = breakdown?.primary_metric || task.primary_metric;
  const metricOrder = [...new Set(selected.map((record) => record.metric_id))].sort(
    (left, right) => Number(right === primaryMetric) - Number(left === primaryMetric),
  );
  const labels = new Map(selected.map((record) => [record.metric_id, record.metric_label]));
  const grouped = new Map();
  selected.forEach((record) => {
    if (!grouped.has(record.solution_id)) grouped.set(record.solution_id, new Map());
    grouped.get(record.solution_id).set(record.metric_id, record);
  });
  if (!state.taskSort.key || (!metricOrder.includes(state.taskSort.key) && state.taskSort.key !== "solution")) {
    state.taskSort = { key: primaryMetric, direction: "desc" };
  }
  const valueForSolution = (solutionId) => {
    if (state.taskSort.key === "solution") return state.solutions.get(solutionId).title;
    return grouped.get(solutionId).get(state.taskSort.key)?.value ?? Number.NEGATIVE_INFINITY;
  };
  const matching = [...grouped.keys()].sort((left, right) => {
    const comparison = compareValues(
      valueForSolution(left),
      valueForSolution(right),
      state.taskSort.direction,
    );
    return comparison || left.localeCompare(right);
  });
  const primaryRanks = new Map(
    [...grouped.keys()]
      .sort((left, right) => {
        const leftValue =
          grouped.get(left).get(primaryMetric)?.value ?? Number.NEGATIVE_INFINITY;
        const rightValue =
          grouped.get(right).get(primaryMetric)?.value ?? Number.NEGATIVE_INFINITY;
        return rightValue - leftValue || left.localeCompare(right);
      })
      .map((solutionId, index) => [solutionId, index + 1]),
  );
  const rows = matching.map((solutionId) => {
    const solution = state.solutions.get(solutionId);
    const records = grouped.get(solutionId);
    const rank = primaryRanks.get(solutionId);
    const metricCells = metricOrder.map((metricId) => {
      const record = records.get(metricId);
      if (!record) return '<span class="muted">—</span>';
      const primaryClass = metricId === primaryMetric ? " primary" : "";
      return `<span class="score${primaryClass}">${formatScore(record.value)}</span>`;
    });
    return [
      `<span class="rank">${rank}</span>`,
      solutionName(solution, records.get(primaryMetric) || records.values().next().value),
      ...metricCells,
    ];
  });

  document.querySelector("#task-context").innerHTML = `
    <div class="task-note-header">
      <div>
        <h2>${escapeHtml(task.title)}</h2>
        <p>${escapeHtml(task.description)}</p>
      </div>
    </div>
    <div class="meta-list">
      <span class="meta-chip">Primary metric: ${escapeHtml(labels.get(primaryMetric))}</span>
      <span class="meta-chip">${escapeHtml(datasetLabel(datasetId))}</span>
      ${breakdownValue ? `<span class="meta-chip">${escapeHtml(breakdown.category === "relevant_documents" ? "Relevant docs" : "Query")} · ${escapeHtml(breakdown.label)}: ${escapeHtml(breakdownValue.label)} · ${formatInteger(breakdownValue.count)} queries${breakdownValue.target_count ? ` · ${formatInteger(breakdownValue.target_count)} relevant pairs` : ""}</span>` : ""}
    </div>`;
  const breakdownNote = document.querySelector("#breakdown-note");
  breakdownNote.hidden = !breakdownValue;
  breakdownNote.textContent = breakdownValue
    ? `${breakdown.description}${breakdownValue.count < 30 ? " This group is small, so its ranking may be unstable." : ""}`
    : "";
  document.querySelector("#task-table").innerHTML = renderTable(
    [
      { label: "Rank" },
      { label: "Solution", key: "solution" },
      ...metricOrder.map((id) => ({ label: labels.get(id), key: id })),
    ],
    rows,
    "task",
    state.taskSort,
  );
  installSortButtons("#task-table", "task", (key) => {
    toggleSort(state.taskSort, key, metricOrder.includes(key));
    renderTask();
  });
  installSolutionDetailLinks("#task-table");
}

function rankForResult(record) {
  const peers = primaryRecords()
    .filter(
      (item) =>
        item.task_id === record.task_id &&
        item.dataset_id === record.dataset_id &&
        item.dataset_version === record.dataset_version,
    )
    .sort(
      (left, right) =>
        right.value - left.value || left.solution_id.localeCompare(right.solution_id),
    );
  return peers.findIndex((item) => item.solution_id === record.solution_id) + 1;
}

function renderSolution() {
  const solutionId = document.querySelector("#solution-selector").value;
  const solution = state.solutions.get(solutionId);
  const matching = primaryRecords()
    .filter((record) => record.solution_id === solutionId)
    .sort((left, right) => {
      const value = (record) => {
        if (state.solutionSort.key === "task") return state.tasks.get(record.task_id).title;
        if (state.solutionSort.key === "metric") return record.metric_label;
        if (state.solutionSort.key === "score") return record.value;
        if (state.solutionSort.key === "rank") return rankForResult(record);
        return `${record.dataset_id} ${record.dataset_version}`;
      };
      return compareValues(value(left), value(right), state.solutionSort.direction);
    });
  const rows = matching.map((record) => {
    const peers = primaryRecords()
      .filter(
        (item) =>
          item.task_id === record.task_id &&
          item.dataset_id === record.dataset_id &&
          item.dataset_version === record.dataset_version,
      )
      .sort((left, right) => right.value - left.value);
    const rank = rankForResult(record);
    return [
      `<strong>${escapeHtml(state.tasks.get(record.task_id).title)}</strong>`,
      escapeHtml(record.metric_label),
      `<span class="score primary">${formatScore(record.value)}</span>`,
      `<span class="rank">${rank} / ${peers.length}</span>`,
      escapeHtml(datasetLabel(record.dataset_id)),
    ];
  });

  const sourceRecord = matching[0];
  document.querySelector("#solution-context").innerHTML = `
    <div class="task-note-header">
      <div>
        <div class="solution-heading"><h2>${escapeHtml(solution.title)}</h2>${solutionCodeLink(solution, sourceRecord)}</div>
        <p>${escapeHtml(solution.description)}</p>
      </div>
    </div>`;
  document.querySelector("#solution-table").innerHTML = renderTable(
    [
      { label: "Task", key: "task" },
      { label: "Primary metric", key: "metric" },
      { label: "Score", key: "score" },
      { label: "Rank", key: "rank" },
      { label: "Dataset release", key: "dataset" },
    ],
    rows,
    "solution",
    state.solutionSort,
  );
  installSortButtons("#solution-table", "solution", (key) => {
    toggleSort(state.solutionSort, key, ["score", "rank"].includes(key));
    renderSolution();
  });
}

function renderCoverage() {
  const taskIds = [...state.tasks.keys()].sort();
  const primary = primaryRecords();
  const rows = [...state.solutions.entries()].map(([solutionId, solution]) => {
    const sourceRecord = primary.find((item) => item.solution_id === solutionId);
    const cells = [
      solutionName(solution, sourceRecord),
    ];
    taskIds.forEach((taskId) => {
      const record = primary.find(
        (item) => item.solution_id === solutionId && item.task_id === taskId,
      );
      if (!record) {
        cells.push('<span class="muted">Not evaluated</span>');
        return;
      }
      const peers = primary
        .filter((item) => item.task_id === taskId)
        .sort((left, right) => right.value - left.value);
      const rank = peers.findIndex((item) => item.solution_id === solutionId) + 1;
      cells.push(
        `<span class="score">${formatScore(record.value)}</span> ` +
          `<span class="rank">#${rank}</span><br>` +
          `<span class="muted">${escapeHtml(record.metric_label)}</span>`,
      );
    });
    return cells;
  });
  document.querySelector("#coverage-table").innerHTML = renderTable(
    [
      { label: "Solution" },
      ...taskIds.map((taskId) => ({ label: state.tasks.get(taskId).title })),
    ],
    rows,
  );
  installSolutionDetailLinks("#coverage-table");
}

function activatePanel(panelId) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.panel === panelId);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });
}

function installTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activatePanel(button.dataset.panel));
  });
}

function installSolutionDetailLinks(containerSelector) {
  document
    .querySelector(containerSelector)
    .querySelectorAll(".solution-detail-link")
    .forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelector("#solution-selector").value = button.dataset.solutionId;
        renderSolution();
        activatePanel("solution-panel");
      });
    });
}

function installControls() {
  const taskSelector = document.querySelector("#task-selector");
  [...state.tasks.values()].forEach((task) => taskSelector.add(new Option(task.title, task.id)));
  taskSelector.addEventListener("change", () => {
    populateDatasetSelector();
    populateBreakdownSelector();
    state.profile.refresh(true);
    updateControlVisibility();
    state.taskSort = { key: state.tasks.get(taskSelector.value).primary_metric, direction: "desc" };
    renderTask();
  });
  document.querySelector("#dataset-selector").addEventListener("change", () => {
    populateBreakdownSelector();
    state.profile.refresh(true);
    renderTask();
  });

  document.querySelector("#breakdown-selector").addEventListener("change", () => {
    populateBreakdownValueSelector();
    renderTask();
  });
  document.querySelector("#breakdown-value-selector").addEventListener("change", renderTask);

  const solutionSelector = document.querySelector("#solution-selector");
  [...state.solutions.values()].forEach((solution) =>
    solutionSelector.add(new Option(solution.title, solution.id)),
  );
  solutionSelector.addEventListener("change", renderSolution);
}

function updateControlVisibility() {
  document.querySelector("#task-control").hidden = state.tasks.size <= 1;
  document.querySelector("#dataset-control").hidden =
    document.querySelector("#dataset-selector").options.length <= 1;
  document.querySelector("#task-controls").hidden =
    document.querySelector("#task-control").hidden &&
    document.querySelector("#dataset-control").hidden;
  document.querySelector("#coverage-tab").hidden = state.tasks.size <= 1;
}

function renderSummary() {
  const counts = state.payload.benchmark.counts || {};
  const items = [];
  if (counts.documents) items.push(`${formatInteger(counts.documents)} documents`);
  if (counts.queries) items.push(`${formatInteger(counts.queries)} queries`);
  items.push(`${formatInteger(state.solutions.size)} solutions`);
  if (state.tasks.size > 1) items.push(`${formatInteger(state.tasks.size)} tasks`);
  if (state.payload.benchmark.languages?.length) {
    items.push(state.payload.benchmark.languages.join(" + "));
  }
  document.querySelector("#summary").innerHTML = items
    .map((item) => `<span class="summary-item">${escapeHtml(item)}</span>`)
    .join("");
}

async function main() {
  const response = await fetch("data/results.json");
  if (!response.ok) throw new Error(`Unable to load benchmark data: ${response.status}`);
  state.payload = await response.json();
  state.tasks = new Map(state.payload.tasks.map((task) => [task.id, task]));
  state.solutions = new Map(
    state.payload.solutions.map((solution) => [solution.id, solution]),
  );
  state.profile = createProfileController({
    solutions: state.solutions,
    getBreakdowns: () => {
      const { taskId, datasetId, datasetVersion } = currentTaskDataset();
      return applicableBreakdowns(taskId, datasetId, datasetVersion);
    },
    getOverallRecords: currentOverallPrimaryRecords,
    getDefaultBreakdownId: () =>
      breakdownDisplayByTask.get(currentTaskDataset().taskId)?.defaultProfile,
    categoryLabel: breakdownCategoryLabel,
    escapeHtml,
    formatScore,
    formatInteger,
  });

  installTabs();
  installControls();
  state.profile.install();
  populateDatasetSelector();
  populateBreakdownSelector();
  state.profile.refresh(true);
  updateControlVisibility();
  renderSummary();
  renderTask();
  renderSolution();
  if (state.tasks.size > 1) renderCoverage();
}

main().catch((error) => {
  const notice = document.querySelector("#error-notice");
  notice.hidden = false;
  notice.textContent = error.message;
  console.error(error);
});
