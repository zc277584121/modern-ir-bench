import { createProfileController } from "./profile.js";

const state = {
  payload: null,
  tasks: new Map(),
  solutions: new Map(),
  leaderboardSort: { key: null, direction: "desc" },
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
  [
    "modern-ir-pdf-retrieval-v1",
    {
      ids: new Set([
        "language",
        "relevant_document_transform_tier",
      ]),
      defaultProfile: "relevant_document_transform_tier",
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
  const pdfRelease = datasetId.match(/^modern-ir-bench-pdf-v(\d+(?:\.\d+)?)/);
  if (pdfRelease) return `Modern IR PDF v${pdfRelease[1]}`;
  const release = datasetId.match(/^modern-ir-bench-v(\d+(?:\.\d+)?)/);
  if (release) return `Modern IR Bench v${release[1]}`;
  return datasetId.replace(/-\d{8}$/, "");
}

function solutionCodeLink(solution) {
  const url = solution.code_url;
  if (!url) return "";
  return `<a class="solution-code-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer" aria-label="Open maintained code for ${escapeHtml(solution.title)}" title="Open maintained Solution code">
    <svg aria-hidden="true" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9.5 2.5h4v4" />
      <path d="m7 9 6.5-6.5" />
      <path d="M13 9.5v3a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h3" />
    </svg>
  </a>`;
}

function solutionReadmeLink(solution) {
  if (!solution.readme_url) return `<strong>${escapeHtml(solution.title)}</strong>`;
  return `<a class="solution-readme-link" href="${escapeHtml(solution.readme_url)}" target="_blank" rel="noreferrer" data-solution-id="${escapeHtml(solution.id)}" aria-describedby="solution-preview">${escapeHtml(solution.title)}</a>`;
}

function solutionName(solution) {
  return `<span class="solution-name">${solutionReadmeLink(solution)}${solutionCodeLink(solution)}</span>`;
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
  return `<p class="table-scroll-hint">Swipe horizontally to see all columns →</p>
    <div class="table-wrap"><table class="result-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function orderedTasks() {
  const primaryCount = (taskId) =>
    primaryRecords().filter((record) => record.task_id === taskId).length;
  return [...state.tasks.values()].sort(
    (left, right) =>
      primaryCount(right.id) - primaryCount(left.id) || left.title.localeCompare(right.title),
  );
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

function selectedSolutionTag() {
  return document.querySelector("#solution-tag-selector")?.value || "";
}

function solutionHasTag(solutionId, tag = selectedSolutionTag()) {
  if (!tag) return true;
  return (state.solutions.get(solutionId)?.tags || []).includes(tag);
}

function solutionTagLabel(tag) {
  const labels = {
    ocr: "OCR",
    "ocr-free": "OCR-free",
    rrf: "RRF",
  };
  return labels[tag] || tag
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function populateSolutionTagSelector() {
  const selector = document.querySelector("#solution-tag-selector");
  const previous = selector.value;
  const { taskId, datasetId, datasetVersion } = currentTaskDataset();
  const solutionIds = new Set(
    primaryRecords()
      .filter(
        (record) =>
          record.task_id === taskId &&
          record.dataset_id === datasetId &&
          record.dataset_version === datasetVersion,
      )
      .map((record) => record.solution_id),
  );
  const counts = new Map();
  solutionIds.forEach((solutionId) => {
    (state.solutions.get(solutionId)?.tags || []).forEach((tag) => {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    });
  });

  selector.replaceChildren(new Option(`All solutions (${solutionIds.size})`, ""));
  [...counts.entries()]
    .sort(([left], [right]) => solutionTagLabel(left).localeCompare(solutionTagLabel(right)))
    .forEach(([tag, count]) => {
      selector.add(new Option(`${solutionTagLabel(tag)} (${count})`, tag));
    });
  selector.value = counts.has(previous) ? previous : "";
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
        record.dataset_version === datasetVersion &&
        solutionHasTag(record.solution_id),
    )
    .sort(
      (left, right) =>
        right.value - left.value || left.solution_id.localeCompare(right.solution_id),
    );
}

function currentEvaluation() {
  const { taskId, datasetId, datasetVersion } = currentTaskDataset();
  const task = state.tasks.get(taskId);
  const overallResults = state.payload.results.filter(
    (record) =>
      record.task_id === taskId &&
      record.dataset_id === datasetId &&
      record.dataset_version === datasetVersion &&
      solutionHasTag(record.solution_id),
  );
  const breakdown = selectedBreakdown();
  const breakdownValue = selectedBreakdownValue();
  const primaryMetric = breakdown?.primary_metric || task.primary_metric;
  return {
    breakdown,
    breakdownValue,
    results: breakdownValue
      ? breakdownValue.results.filter((record) => solutionHasTag(record.solution_id))
      : overallResults,
    primaryMetric,
  };
}

function metricCatalog(results, primaryMetric) {
  const labels = new Map(results.map((record) => [record.metric_id, record.metric_label]));
  const order = [...labels.keys()].sort(
    (left, right) => Number(right === primaryMetric) - Number(left === primaryMetric),
  );
  return { labels, order };
}

function renderLeaderboard() {
  const evaluation = currentEvaluation();
  const {
    breakdown,
    breakdownValue,
    primaryMetric,
    results: selected,
  } = evaluation;
  const { labels, order: metricOrder } = metricCatalog(selected, primaryMetric);
  const grouped = new Map();
  selected.forEach((record) => {
    if (!grouped.has(record.solution_id)) grouped.set(record.solution_id, new Map());
    grouped.get(record.solution_id).set(record.metric_id, record);
  });
  if (!state.leaderboardSort.key || (!metricOrder.includes(state.leaderboardSort.key) && state.leaderboardSort.key !== "solution")) {
    state.leaderboardSort = { key: primaryMetric, direction: "desc" };
  }
  const valueForSolution = (solutionId) => {
    if (state.leaderboardSort.key === "solution") {
      return state.solutions.get(solutionId).title;
    }
    return (
      grouped.get(solutionId).get(state.leaderboardSort.key)?.value ??
      Number.NEGATIVE_INFINITY
    );
  };
  const matching = [...grouped.keys()].sort((left, right) => {
    const comparison = compareValues(
      valueForSolution(left),
      valueForSolution(right),
      state.leaderboardSort.direction,
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
      solutionName(solution),
      ...metricCells,
    ];
  });

  const breakdownNote = document.querySelector("#breakdown-note");
  breakdownNote.hidden = !breakdownValue;
  breakdownNote.textContent = breakdownValue
    ? `${breakdown.description}${breakdownValue.count < 30 ? " This group is small, so its ranking may be unstable." : ""}`
    : "";
  document.querySelector("#leaderboard-table").innerHTML = renderTable(
    [
      { label: "Rank" },
      { label: "Solution", key: "solution" },
      ...metricOrder.map((id) => ({ label: labels.get(id), key: id })),
    ],
    rows,
    "leaderboard",
    state.leaderboardSort,
  );
  installSortButtons("#leaderboard-table", "leaderboard", (key) => {
    toggleSort(state.leaderboardSort, key, metricOrder.includes(key));
    renderLeaderboard();
  });
}

function activatePanel(panelId) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.panel === panelId);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });
  document.querySelector("#evaluation-context").hidden = panelId === "about-panel";
}

function installTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activatePanel(button.dataset.panel));
  });
}

function installSolutionPreviews() {
  const preview = document.querySelector("#solution-preview");
  let activeTrigger = null;

  const hide = () => {
    activeTrigger = null;
    preview.hidden = true;
  };

  const show = (trigger) => {
    const solution = state.solutions.get(trigger.dataset.solutionId);
    if (!solution?.readme_preview) return;
    activeTrigger = trigger;
    preview.innerHTML = `
      <span class="solution-preview-label">README preview</span>
      <strong>${escapeHtml(solution.readme_title)}</strong>
      <span>${escapeHtml(solution.readme_preview)}</span>
      ${(solution.tags || []).length ? `<span class="solution-preview-tags">${solution.tags.map(solutionTagLabel).map(escapeHtml).join(" · ")}</span>` : ""}
      <span class="solution-preview-hint">Click to open method notes ↗</span>`;
    preview.hidden = false;

    const triggerRect = trigger.getBoundingClientRect();
    const margin = 12;
    const gap = 8;
    const left = Math.min(
      Math.max(margin, triggerRect.left),
      window.innerWidth - preview.offsetWidth - margin,
    );
    let top = triggerRect.bottom + gap;
    if (top + preview.offsetHeight > window.innerHeight - margin) {
      top = Math.max(margin, triggerRect.top - preview.offsetHeight - gap);
    }
    preview.style.left = `${left}px`;
    preview.style.top = `${top}px`;
  };

  document.addEventListener("pointerover", (event) => {
    const trigger = event.target.closest?.(".solution-readme-link");
    if (trigger && trigger !== activeTrigger) show(trigger);
  });
  document.addEventListener("pointerout", (event) => {
    const trigger = event.target.closest?.(".solution-readme-link");
    if (trigger && !trigger.contains(event.relatedTarget)) hide();
  });
  document.addEventListener("focusin", (event) => {
    const trigger = event.target.closest?.(".solution-readme-link");
    if (trigger) show(trigger);
  });
  document.addEventListener("focusout", (event) => {
    if (event.target.closest?.(".solution-readme-link")) hide();
  });
  window.addEventListener("scroll", hide, true);
  window.addEventListener("resize", hide);
}

function installControls() {
  const taskSelector = document.querySelector("#task-selector");
  orderedTasks().forEach((task) => taskSelector.add(new Option(task.title, task.id)));
  taskSelector.addEventListener("change", () => {
    refreshDatasetContext(true);
  });
  document.querySelector("#dataset-selector").addEventListener("change", () => {
    refreshDatasetContext(false);
  });

  document.querySelector("#breakdown-selector").addEventListener("change", () => {
    populateBreakdownValueSelector();
    refreshBreakdownContext();
  });
  document
    .querySelector("#breakdown-value-selector")
    .addEventListener("change", refreshBreakdownContext);
  document.querySelector("#solution-tag-selector").addEventListener("change", () => {
    resetResultSorts();
    state.profile.refresh(true);
    renderLeaderboard();
  });
}

function resetResultSorts() {
  state.leaderboardSort = { key: null, direction: "desc" };
}

function refreshDatasetContext(rebuildDatasets) {
  if (rebuildDatasets) populateDatasetSelector();
  populateBreakdownSelector();
  populateSolutionTagSelector();
  state.profile.refresh(true);
  resetResultSorts();
  renderLeaderboard();
}

function refreshBreakdownContext() {
  resetResultSorts();
  state.profile.sync();
  renderLeaderboard();
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
    getActiveBreakdownId: () => selectedBreakdown()?.id,
    getActiveBreakdownValueId: () => selectedBreakdownValue()?.id,
    categoryLabel: breakdownCategoryLabel,
    formatSolutionTag: solutionTagLabel,
    escapeHtml,
    formatScore,
    formatInteger,
  });

  installSolutionPreviews();
  installTabs();
  installControls();
  state.profile.install();
  populateDatasetSelector();
  populateBreakdownSelector();
  populateSolutionTagSelector();
  state.profile.refresh(true);
  document.querySelector("#task-control").hidden = state.tasks.size <= 1;
  renderLeaderboard();
}

main().catch((error) => {
  const notice = document.querySelector("#error-notice");
  notice.hidden = false;
  notice.textContent = error.message;
  console.error(error);
});
