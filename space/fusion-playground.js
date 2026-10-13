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
  let manifestPromise;
  let benchmark;
  let installed = false;
  const routePromises = new Map();

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
    element("#fusion-routes").innerHTML = benchmark.routes
      .map((route) => `<label class="fusion-route">
        <input type="checkbox" data-fusion-route="${escapeHtml(route.id)}" />
        <span class="fusion-route-copy">
          <strong>${escapeHtml(route.label)}</strong>
          <span data-route-status="${escapeHtml(route.id)}">${escapeHtml(route.kind)}</span>
        </span>
        <span class="fusion-weight-field">
          <span>Weight</span>
          <input type="number" min="0.1" max="10" step="0.1" value="1" data-route-weight="${escapeHtml(route.id)}" aria-label="Weight for ${escapeHtml(route.label)}" disabled />
        </span>
      </label>`)
      .join("");
    element("#fusion-routes").querySelectorAll("[data-fusion-route]").forEach((input) => {
      input.addEventListener("change", async () => {
        const route = benchmark.routes.find((candidate) => candidate.id === input.dataset.fusionRoute);
        const weight = element(`[data-route-weight="${route.id}"]`);
        weight.disabled = !input.checked;
        if (!input.checked) {
          renderResults();
          return;
        }
        setRouteStatus(route, "Loading…");
        renderResults();
        try {
          await loadRoute(route);
          if (!benchmark.routes.includes(route)) return;
          setRouteStatus(route, route.kind);
          renderResults();
        } catch (error) {
          if (!benchmark.routes.includes(route)) return;
          input.checked = false;
          weight.disabled = true;
          setRouteStatus(route, "Unable to load");
          renderResults();
          console.error(error);
        }
      });
    });
    element("#fusion-routes").querySelectorAll("[data-route-weight]").forEach((input) => {
      input.addEventListener("input", renderResults);
    });
  }

  function setRouteStatus(route, status) {
    const target = element(`[data-route-status="${route.id}"]`);
    if (target) target.textContent = status;
  }

  async function loadRoute(route) {
    if (route.rankings) return;
    if (!routePromises.has(route.path)) {
      const expectedQueryCount = benchmark.queries.length;
      routePromises.set(
        route.path,
        loadCompressedJson(`data/fusion/${route.path}`).then((payload) => {
          if (payload.route_id !== route.id || payload.rankings.length !== expectedQueryCount) {
            throw new Error(`Invalid fusion evidence for ${route.label}.`);
          }
          return payload.rankings;
        }),
      );
    }
    route.rankings = await routePromises.get(route.path);
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

  function resultRow(label, result, bestRoute, isFusion = false) {
    const cells = metricCatalog.map((metric) => {
      const score = result.metrics[metric.id];
      const delta = isFusion ? score - bestRoute[metric.id] : null;
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
    if (routes.some((route) => !route.rankings)) {
      summary.textContent = "Loading selected rankings…";
      result.innerHTML = '<p class="empty-state">The selected route evidence is loading.</p>';
      return;
    }

    const depth = Number(element("#fusion-depth").value);
    const rankConstant = Number(element("#rank-constant").value);
    const queryIndexes = selectedQueryIndexes();
    const routeResults = routes.map((route) => {
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
    const bestRoute = Object.fromEntries(metricCatalog.map((metric) => [
      metric.id,
      Math.max(...routeResults.map(({ result }) => result.metrics[metric.id])),
    ]));
    const rawNdcgDelta = fused.metrics["ndcg@10"] - bestRoute["ndcg@10"];
    const ndcgDelta = Math.abs(rawNdcgDelta) < 0.0005 ? 0 : rawNdcgDelta;
    const methodLabel = method === "rrf" ? "RRF" : "Weighted scores";
    summary.textContent = `${methodLabel} · ${formatInteger(fused.count)} queries · ${ndcgDelta > 0 ? "+" : ""}${formatScore(ndcgDelta)} NDCG@10 vs best selected route`;
    result.innerHTML = `<div class="table-wrap"><table class="result-table fusion-result-table">
      <thead><tr><th>Configuration</th>${metricCatalog.map((metric) => `<th>${escapeHtml(metric.label)}</th>`).join("")}</tr></thead>
      <tbody>
        ${resultRow(`${methodLabel} fusion`, fused, bestRoute, true)}
        ${routeResults.map(({ route, result }) => resultRow(route.label, result, bestRoute)).join("")}
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
    element("#fusion-loading").hidden = false;
    element("#fusion-workspace").hidden = true;
    try {
      manifestPromise ||= loadCompressedJson("data/fusion/manifest.json.gz");
      const payload = await manifestPromise;
      const context = getContext();
      const nextBenchmark = payload.benchmarks.find(
        (candidate) =>
          candidate.task_id === context.taskId &&
          candidate.dataset_id === context.datasetId &&
          candidate.dataset_version === context.datasetVersion,
      );
      if (!nextBenchmark) throw new Error("No reusable route evidence is available for this release.");
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
