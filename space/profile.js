const colors = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
];

export function createProfileController({
  solutions,
  getBreakdowns,
  getOverallRecords,
  categoryLabel,
  escapeHtml,
  formatScore,
  formatInteger,
}) {
  let selectedSolutionIds = new Set();
  let resizeTimer;

  function metricRecord(value, solutionId, metricId) {
    return value.results.find(
      (record) => record.solution_id === solutionId && record.metric_id === metricId,
    );
  }

  function splitLabel(label) {
    if (label.length <= 20) return [label];
    const parts = label.split(" / ");
    if (parts.length === 2) return parts;
    const words = label.split(" ");
    if (words.length === 1) {
      const midpoint = Math.ceil(label.length / 2);
      return [label.slice(0, midpoint), label.slice(midpoint)];
    }
    const midpoint = Math.ceil(words.length / 2);
    return [words.slice(0, midpoint).join(" "), words.slice(midpoint).join(" ")];
  }

  function polarPoint(centerX, centerY, radius, index, total) {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / total;
    return {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
      angle,
    };
  }

  function renderRadar(values, solutionIds, metricId, metricLabel) {
    const width = 760;
    const height = 480;
    const centerX = width / 2;
    const centerY = 225;
    const radius = 145;
    const grid = [1, 2, 3, 4, 5]
      .map((level) => {
        const levelRadius = (radius * level) / 5;
        const points = values
          .map((_, index) => polarPoint(centerX, centerY, levelRadius, index, values.length))
          .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
          .join(" ");
        return `<polygon class="chart-grid" points="${points}" />
          <text class="chart-grid-label" x="${centerX + 5}" y="${(centerY - levelRadius + 4).toFixed(1)}">${(level / 5).toFixed(1)}</text>`;
      })
      .join("");
    const axes = values
      .map((value, index) => {
        const end = polarPoint(centerX, centerY, radius, index, values.length);
        const labelPoint = polarPoint(centerX, centerY, radius + 42, index, values.length);
        const anchor = Math.cos(labelPoint.angle) > 0.25
          ? "start"
          : Math.cos(labelPoint.angle) < -0.25
            ? "end"
            : "middle";
        const tspans = splitLabel(value.label)
          .map(
            (line, lineIndex) =>
              `<tspan x="${labelPoint.x.toFixed(1)}" dy="${lineIndex === 0 ? 0 : 14}">${escapeHtml(line)}</tspan>`,
          )
          .join("");
        const sample = value.target_count || value.count;
        return `<line class="chart-axis" x1="${centerX}" y1="${centerY}" x2="${end.x.toFixed(1)}" y2="${end.y.toFixed(1)}" />
          <text class="chart-axis-label" x="${labelPoint.x.toFixed(1)}" y="${labelPoint.y.toFixed(1)}" text-anchor="${anchor}"><title>${escapeHtml(value.label)} · ${formatInteger(sample)} samples</title>${tspans}</text>`;
      })
      .join("");
    const series = solutionIds
      .map((solutionId, solutionIndex) => {
        const solution = solutions.get(solutionId);
        const color = colors[solutionIndex];
        const scoredPoints = values.map((value, valueIndex) => {
          const score = metricRecord(value, solutionId, metricId)?.value ?? 0;
          return {
            ...polarPoint(centerX, centerY, radius * score, valueIndex, values.length),
            score,
            value,
          };
        });
        const points = scoredPoints
          .map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`)
          .join(" ");
        const markers = scoredPoints
          .map(
            (point) =>
              `<circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="3.5" fill="${color}"><title>${escapeHtml(solution.title)} · ${escapeHtml(point.value.label)} · ${formatScore(point.score)}</title></circle>`,
          )
          .join("");
        return `<polygon class="chart-series" points="${points}" fill="${color}" fill-opacity="0.08" stroke="${color}"><title>${escapeHtml(solution.title)}</title></polygon>${markers}`;
      })
      .join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metricLabel)} profile across ${values.length} groups">
      <title>${escapeHtml(metricLabel)} dimension profile</title>
      <desc>Scores use a fixed zero-to-one scale. Larger shapes indicate higher scores across the displayed groups.</desc>
      ${grid}${axes}${series}
    </svg>`;
  }

  function renderBars(values, solutionIds, metricId, metricLabel) {
    const width = 760;
    const height = 330;
    const plotTop = 24;
    const plotBottom = 265;
    const plotLeft = 64;
    const plotRight = 730;
    const plotHeight = plotBottom - plotTop;
    const groupWidth = (plotRight - plotLeft) / values.length;
    const barWidth = Math.min(42, (groupWidth - 48) / solutionIds.length);
    const grid = [0, 0.2, 0.4, 0.6, 0.8, 1]
      .map((tick) => {
        const y = plotBottom - tick * plotHeight;
        return `<line class="chart-grid" x1="${plotLeft}" y1="${y.toFixed(1)}" x2="${plotRight}" y2="${y.toFixed(1)}" />
          <text class="chart-grid-label" x="${plotLeft - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end">${tick.toFixed(1)}</text>`;
      })
      .join("");
    const groups = values
      .map((value, valueIndex) => {
        const groupCenter = plotLeft + groupWidth * (valueIndex + 0.5);
        const bars = solutionIds
          .map((solutionId, solutionIndex) => {
            const solution = solutions.get(solutionId);
            const score = metricRecord(value, solutionId, metricId)?.value ?? 0;
            const barHeight = score * plotHeight;
            const x =
              groupCenter + (solutionIndex - (solutionIds.length - 1) / 2) * barWidth;
            const y = plotBottom - barHeight;
            return `<rect x="${(x - barWidth * 0.42).toFixed(1)}" y="${y.toFixed(1)}" width="${(barWidth * 0.84).toFixed(1)}" height="${barHeight.toFixed(1)}" fill="${colors[solutionIndex]}"><title>${escapeHtml(solution.title)} · ${escapeHtml(value.label)} · ${formatScore(score)}</title></rect>
              <text class="chart-bar-value" x="${x.toFixed(1)}" y="${Math.max(plotTop + 10, y - 6).toFixed(1)}">${formatScore(score)}</text>`;
          })
          .join("");
        return `${bars}<text class="chart-axis-label" x="${groupCenter.toFixed(1)}" y="${plotBottom + 28}" text-anchor="middle">${escapeHtml(value.label)}</text>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metricLabel)} comparison across ${values.length} groups">
      <title>${escapeHtml(metricLabel)} grouped comparison</title>
      <desc>Scores use a fixed zero-to-one scale.</desc>
      ${grid}${groups}
    </svg>`;
  }

  function renderSolutionOptions() {
    const records = getOverallRecords();
    const container = document.querySelector("#profile-solution-options");
    const atLimit = selectedSolutionIds.size >= colors.length;
    container.innerHTML = records
      .map((record) => {
        const solution = solutions.get(record.solution_id);
        const checked = selectedSolutionIds.has(record.solution_id);
        return `<label class="solution-option">
          <input type="checkbox" data-profile-solution="${escapeHtml(record.solution_id)}"${checked ? " checked" : ""}${atLimit && !checked ? " disabled" : ""} />
          <span>${escapeHtml(solution.title)}</span>
        </label>`;
      })
      .join("");
    document.querySelector("#profile-solution-summary").textContent =
      `${selectedSolutionIds.size} solutions selected`;

    container.querySelectorAll("[data-profile-solution]").forEach((input) => {
      input.addEventListener("change", () => {
        const solutionId = input.dataset.profileSolution;
        if (input.checked) {
          if (selectedSolutionIds.size >= colors.length) {
            input.checked = false;
            return;
          }
          selectedSolutionIds.add(solutionId);
        } else if (selectedSolutionIds.size === 1) {
          input.checked = true;
          return;
        } else {
          selectedSolutionIds.delete(solutionId);
        }
        renderSolutionOptions();
        render();
      });
    });
  }

  function render() {
    const breakdownId = document.querySelector("#profile-breakdown-selector").value;
    const breakdown = getBreakdowns().find((item) => item.id === breakdownId);
    if (!breakdown) return;

    const solutionOrder = getOverallRecords()
      .map((record) => record.solution_id)
      .filter((solutionId) => selectedSolutionIds.has(solutionId));
    const allValues = [...breakdown.values];
    const groupLimit = window.matchMedia("(max-width: 760px)").matches ? 6 : 8;
    const values = allValues.length > groupLimit
      ? allValues
          .sort(
            (left, right) =>
              (right.target_count || right.count) - (left.target_count || left.count),
          )
          .slice(0, groupLimit)
      : allValues;
    const exampleRecord = values
      .flatMap((value) => value.results)
      .find((record) => record.metric_id === breakdown.primary_metric);
    const metricLabel = exampleRecord?.metric_label || breakdown.primary_metric;
    const scores = values.flatMap((value) =>
      solutionOrder.map(
        (solutionId) => metricRecord(value, solutionId, breakdown.primary_metric)?.value,
      ),
    );
    const supportsFixedScale = scores.every(
      (score) => Number.isFinite(score) && score >= 0 && score <= 1,
    );

    document.querySelector("#profile-description").textContent =
      `${categoryLabel(breakdown.category)} · ${breakdown.label}, measured with ${metricLabel}.`;
    document.querySelector("#profile-note").textContent = allValues.length > values.length
      ? `Showing the ${values.length} largest groups by ${breakdown.category === "relevant_documents" ? "relevant-pair count" : "query count"} out of ${allValues.length}. Scores are not normalized per group.`
      : "Scores use the same fixed 0–1 scale and are not normalized per group.";

    const chart = document.querySelector("#profile-chart");
    if (!supportsFixedScale) {
      chart.innerHTML =
        '<p class="muted">This metric needs a declared chart range before it can be profiled.</p>';
    } else {
      chart.innerHTML = values.length >= 3
        ? renderRadar(values, solutionOrder, breakdown.primary_metric, metricLabel)
        : renderBars(values, solutionOrder, breakdown.primary_metric, metricLabel);
    }
    document.querySelector("#profile-legend").innerHTML = solutionOrder
      .map((solutionId, index) => {
        const solution = solutions.get(solutionId);
        return `<span class="profile-legend-item"><span class="profile-legend-swatch" style="background:${colors[index]}"></span>${escapeHtml(solution.title)}</span>`;
      })
      .join("");
  }

  function refresh(resetSolutions = false) {
    const breakdowns = getBreakdowns().filter((breakdown) => breakdown.values.length >= 2);
    const section = document.querySelector("#profile-section");
    const selector = document.querySelector("#profile-breakdown-selector");
    const previous = selector.value;
    selector.replaceChildren();
    section.hidden = breakdowns.length === 0;
    if (!breakdowns.length) return;

    breakdowns.forEach((breakdown) => {
      selector.add(
        new Option(
          `${categoryLabel(breakdown.category)} · ${breakdown.label}`,
          breakdown.id,
        ),
      );
    });
    const preferred = breakdowns.find((breakdown) => breakdown.id === "query_intent");
    selector.value = breakdowns.some((breakdown) => breakdown.id === previous)
      ? previous
      : (preferred?.id ?? breakdowns[0].id);

    const availableSolutionIds = new Set(
      getOverallRecords().map((record) => record.solution_id),
    );
    selectedSolutionIds = new Set(
      [...selectedSolutionIds].filter((solutionId) => availableSolutionIds.has(solutionId)),
    );
    if (resetSolutions || selectedSolutionIds.size === 0) {
      selectedSolutionIds = new Set(
        getOverallRecords()
          .slice(0, 3)
          .map((record) => record.solution_id),
      );
    }
    renderSolutionOptions();
    render();
  }

  function install() {
    document.querySelector("#profile-breakdown-selector").addEventListener("change", render);
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(render, 120);
    });
  }

  return { install, refresh };
}
