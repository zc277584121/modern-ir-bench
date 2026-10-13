import { atomicRanking, evaluateRankings, fuseRankings, metricCatalog } from "./fusion.js";

async function loadCompressedJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load fusion evidence: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot open the compressed fusion evidence.");
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).json();
}

export function createFusionPlayground({
  getContext,
  escapeHtml,
  formatScore,
  formatInteger,
}) {
  let payloadPromise;
  let benchmark;
  let installed = false;

  const element = (selector) => document.querySelector(selector);

  function selectedRoutes() {
    return benchmark.routes
      .filter((route) => element(`[data-fusion-route="${route.id}"]`)?.checked)
      .map((route) => ({
        ...route,
        weight: Math.max(0.1, Number(element(`[data-route-weight="${route.id}"]`)?.value || 1)),
      }));
  }

  function renderRoutes() {
    const defaultKinds = new Set();
    const selectedByDefault = new Set(
      benchmark.routes
        .filter((route) => {
          if (defaultKinds.has(route.kind)) return false;
          defaultKinds.add(route.kind);
          return true;
        })
        .map((route) => route.id),
    );
    element("#fusion-routes").innerHTML = benchmark.routes
      .map((route) => `<label class="fusion-route">
        <input type="checkbox" data-fusion-route="${escapeHtml(route.id)}"${selectedByDefault.has(route.id) ? " checked" : ""} />
        <span class="fusion-route-copy">
          <strong>${escapeHtml(route.label)}</strong>
          <span>${escapeHtml(route.kind)}</span>
        </span>
        <span class="fusion-weight-field">
          <span>Weight</span>
          <input type="number" min="0.1" max="10" step="0.1" value="1" data-route-weight="${escapeHtml(route.id)}" aria-label="Weight for ${escapeHtml(route.label)}" />
        </span>
      </label>`)
      .join("");
    element("#fusion-routes").querySelectorAll("input").forEach((input) => {
      input.addEventListener("change", renderResults);
      input.addEventListener("input", renderResults);
    });
  }

  function populateSliceDimensions() {
    const selector = element("#fusion-slice-dimension");
    selector.replaceChildren(new Option("All queries", ""));
    benchmark.dimensions.forEach((dimension, index) => {
      selector.add(new Option(dimension.label, String(index)));
    });
    populateSliceValues();
  }

  function populateSliceValues() {
    const dimensionIndex = element("#fusion-slice-dimension").value;
    const control = element("#fusion-slice-value-control");
    const selector = element("#fusion-slice-value");
    selector.replaceChildren();
    control.hidden = dimensionIndex === "";
    if (dimensionIndex === "") return;
    benchmark.dimensions[Number(dimensionIndex)].values.forEach((value) => {
      selector.add(new Option(`${value.label} · ${formatInteger(value.count)}`, value.id));
    });
  }

  function selectedQueryIndexes() {
    const dimensionIndex = element("#fusion-slice-dimension").value;
    if (dimensionIndex === "") return benchmark.queries.map((_, index) => index);
    const index = Number(dimensionIndex);
    const value = element("#fusion-slice-value").value;
    return benchmark.queries
      .map((query, queryIndex) => query.labels[index] === value ? queryIndex : -1)
      .filter((queryIndex) => queryIndex >= 0);
  }

  function resultRow(label, result, bestAtomic, isFusion = false) {
    const cells = metricCatalog.map((metric) => {
      const score = result.metrics[metric.id];
      const delta = isFusion ? score - bestAtomic[metric.id] : null;
      const displayedDelta = delta !== null && Math.abs(delta) < 0.0005 ? 0 : delta;
      const deltaText = displayedDelta === null
        ? ""
        : `<span class="metric-delta ${displayedDelta > 0 ? "positive" : displayedDelta < 0 ? "negative" : "neutral"}">${displayedDelta > 0 ? "+" : ""}${formatScore(displayedDelta)}</span>`;
      return `<td><span class="score${metric.id === "ndcg@10" ? " primary" : ""}">${formatScore(score)}</span>${deltaText}</td>`;
    }).join("");
    return `<tr${isFusion ? ' class="fusion-result-row"' : ""}><td><strong>${escapeHtml(label)}</strong>${isFusion ? '<span class="experimental-label">Live</span>' : ""}</td>${cells}</tr>`;
  }

  function renderResults() {
    if (!benchmark) return;
    const routes = selectedRoutes();
    const result = element("#fusion-results");
    const summary = element("#fusion-summary");
    const rrfControl = element("#rank-constant-control");
    const method = element("#fusion-method").value;
    rrfControl.hidden = method !== "rrf";
    if (routes.length < 2) {
      summary.textContent = "Choose at least two retrieval routes.";
      result.innerHTML = '<p class="empty-state">Select two or more routes to compare a live fusion.</p>';
      return;
    }

    const depth = Number(element("#fusion-depth").value);
    const rankConstant = Number(element("#rank-constant").value);
    const queryIndexes = selectedQueryIndexes();
    const atomicResults = routes.map((route) => {
      const rankings = route.rankings.map((ranking) => atomicRanking(ranking, depth));
      return { route, result: evaluateRankings(benchmark.queries, rankings, queryIndexes) };
    });
    const fusedRankings = benchmark.queries.map((_, queryIndex) => fuseRankings(
      routes.map((route) => ({
        ranking: route.rankings[queryIndex],
        weight: route.weight,
        scoreDirection: route.score_direction,
      })),
      {
        method,
        depth,
        rankConstant,
        documents: benchmark.documents,
      },
    ));
    const fused = evaluateRankings(benchmark.queries, fusedRankings, queryIndexes);
    const bestAtomic = Object.fromEntries(metricCatalog.map((metric) => [
      metric.id,
      Math.max(...atomicResults.map(({ result: atomic }) => atomic.metrics[metric.id])),
    ]));
    const rawNdcgDelta = fused.metrics["ndcg@10"] - bestAtomic["ndcg@10"];
    const ndcgDelta = Math.abs(rawNdcgDelta) < 0.0005 ? 0 : rawNdcgDelta;
    const methodLabel = method === "rrf" ? "RRF" : "Weighted scores";
    summary.textContent = `${methodLabel} · ${formatInteger(fused.count)} queries · ${ndcgDelta > 0 ? "+" : ""}${formatScore(ndcgDelta)} NDCG@10 vs best selected route`;
    result.innerHTML = `<div class="table-wrap"><table class="result-table fusion-result-table">
      <thead><tr><th>Configuration</th>${metricCatalog.map((metric) => `<th>${escapeHtml(metric.label)}</th>`).join("")}</tr></thead>
      <tbody>
        ${resultRow(`${methodLabel} fusion`, fused, bestAtomic, true)}
        ${atomicResults.map(({ route, result: atomic }) => resultRow(route.label, atomic, bestAtomic)).join("")}
      </tbody>
    </table></div>`;
  }

  function renderBenchmark(nextBenchmark) {
    benchmark = nextBenchmark;
    renderRoutes();
    populateSliceDimensions();
    element("#fusion-depth").max = String(benchmark.max_depth);
    element("#fusion-depth").value = String(Math.min(50, benchmark.max_depth));
    renderResults();
  }

  async function refresh() {
    if (element("#fusion-panel").hidden) return;
    element("#fusion-loading").hidden = false;
    element("#fusion-workspace").hidden = true;
    try {
      payloadPromise ||= loadCompressedJson("data/fusion-playground.json.gz");
      const payload = await payloadPromise;
      const context = getContext();
      const nextBenchmark = payload.benchmarks.find(
        (candidate) =>
          candidate.task_id === context.taskId &&
          candidate.dataset_id === context.datasetId &&
          candidate.dataset_version === context.datasetVersion,
      );
      if (!nextBenchmark) throw new Error("No reusable atomic rankings are available for this release.");
      renderBenchmark(nextBenchmark);
      element("#fusion-loading").hidden = true;
      element("#fusion-workspace").hidden = false;
    } catch (error) {
      element("#fusion-loading").textContent = error.message;
      console.error(error);
    }
  }

  function install() {
    if (installed) return;
    installed = true;
    element("#fusion-method").addEventListener("change", renderResults);
    element("#fusion-depth").addEventListener("change", renderResults);
    element("#rank-constant").addEventListener("input", renderResults);
    element("#fusion-slice-dimension").addEventListener("change", () => {
      populateSliceValues();
      renderResults();
    });
    element("#fusion-slice-value").addEventListener("change", renderResults);
  }

  return { install, refresh };
}
