const state = {
  payload: null,
  tasks: new Map(),
  solutions: new Map(),
  taskSort: { key: null, direction: "desc" },
  solutionSort: { key: "task", direction: "asc" },
};

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
  if (datasetId.startsWith("modern-ir-bench-v2.1")) return "Modern IR Bench v2.1";
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
  return `<span class="solution-name"><strong title="${escapeHtml(solution.description)}">${escapeHtml(solution.title)}</strong>${solutionCodeLink(solution, record)}</span>`;
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

function populateDatasetSelector() {
  const selector = document.querySelector("#dataset-selector");
  selector.replaceChildren();
  taskDatasets(document.querySelector("#task-selector").value).forEach((option) =>
    selector.add(new Option(option.label, option.value)),
  );
}

function renderTask() {
  const taskId = document.querySelector("#task-selector").value;
  const [datasetId, datasetVersion] = document
    .querySelector("#dataset-selector")
    .value.split("|");
  const task = state.tasks.get(taskId);
  const selected = state.payload.results.filter(
    (record) =>
      record.task_id === taskId &&
      record.dataset_id === datasetId &&
      record.dataset_version === datasetVersion,
  );
  const metricOrder = [...new Set(selected.map((record) => record.metric_id))].sort(
    (left, right) => Number(right === task.primary_metric) - Number(left === task.primary_metric),
  );
  const labels = new Map(selected.map((record) => [record.metric_id, record.metric_label]));
  const grouped = new Map();
  selected.forEach((record) => {
    if (!grouped.has(record.solution_id)) grouped.set(record.solution_id, new Map());
    grouped.get(record.solution_id).set(record.metric_id, record);
  });
  if (!state.taskSort.key || (!metricOrder.includes(state.taskSort.key) && state.taskSort.key !== "solution")) {
    state.taskSort = { key: task.primary_metric, direction: "desc" };
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
  const rows = matching.map((solutionId) => {
    const solution = state.solutions.get(solutionId);
    const records = grouped.get(solutionId);
    const rank = matching.indexOf(solutionId) + 1;
    const metricCells = metricOrder.map((metricId) => {
      const record = records.get(metricId);
      if (!record) return '<span class="muted">—</span>';
      const primaryClass = metricId === task.primary_metric ? " primary" : "";
      return `<span class="score${primaryClass}">${formatScore(record.value)}</span>`;
    });
    return [
      `<span class="rank">${rank}</span>`,
      solutionName(solution, records.get(task.primary_metric) || records.values().next().value),
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
      <span class="meta-chip">Primary metric: ${escapeHtml(labels.get(task.primary_metric))}</span>
      <span class="meta-chip">${escapeHtml(datasetLabel(datasetId))}</span>
    </div>`;
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
        if (state.solutionSort.key === "rank") {
          const peers = primaryRecords()
            .filter((item) => item.task_id === record.task_id)
            .sort((a, b) => b.value - a.value);
          return peers.findIndex((item) => item.solution_id === solutionId) + 1;
        }
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
    const rank = peers.findIndex((item) => item.solution_id === solutionId) + 1;
    return [
      `<strong>${escapeHtml(state.tasks.get(record.task_id).title)}</strong><br><span class="muted">${escapeHtml(record.task_id)}</span>`,
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
}

function installTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.querySelector(`#${button.dataset.panel}`).classList.add("active");
    });
  });
}

function installControls() {
  const taskSelector = document.querySelector("#task-selector");
  [...state.tasks.values()].forEach((task) => taskSelector.add(new Option(task.title, task.id)));
  taskSelector.addEventListener("change", () => {
    populateDatasetSelector();
    updateControlVisibility();
    state.taskSort = { key: state.tasks.get(taskSelector.value).primary_metric, direction: "desc" };
    renderTask();
  });
  document.querySelector("#dataset-selector").addEventListener("change", renderTask);

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

  installTabs();
  installControls();
  populateDatasetSelector();
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
