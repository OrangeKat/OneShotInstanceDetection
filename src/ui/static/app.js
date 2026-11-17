// Pair inspector: draws the reference crop and query side by side, links every
// pre-RANSAC correspondence, and annotates each query point with the reference
// keypoint index it matched.
const SVG_NS = "http://www.w3.org/2000/svg";
const STATE = {
  models: [],
  pairs: [],
  index: null,
  model: null,
  dump: null,
  view: { scale: 1, tx: 0, ty: 0 },
  occlusion: null,
  lineWidth: Number(localStorage.getItem("lineWidth") || 2),
};

const ABLATION = {
  runs: [],
  jobs: [],
  factors: [],
  filterModel: null,
  filterDataset: null,
  timer: null,
};

const DEFAULTS = { max_keypoints: 2048, ransac_threshold: 3, max_side: 640, ratio: 0.75 };

const el = (id) => document.getElementById(id);

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) url.searchParams.set(key, value);
  }
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function formatNumber(value, digits = 2) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

async function loadModels() {
  const payload = await api("/api/models");
  STATE.models = payload.models;
  const select = el("model");
  select.innerHTML = "";
  for (const model of STATE.models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.append(option);
  }
  STATE.model = STATE.models[0] ?? null;
  select.value = STATE.model;
}

async function loadPairs() {
  const payload = await api("/api/pairs", {
    search: el("search").value,
    hard: el("hard-only").checked ? 1 : 0,
    positives: 1,
    limit: 600,
  });
  STATE.pairs = payload.pairs;
  el("counts").textContent =
    `${payload.pairs.length} shown · ${payload.total} pairs · ${payload.hard_total} hard`;
  renderPairs();
}

function renderPairs() {
  const list = el("pair-list");
  list.innerHTML = "";
  for (const pair of STATE.pairs) {
    const item = document.createElement("li");
    if (pair.index === STATE.index) item.classList.add("active");
    const label = document.createElement("span");
    label.textContent = pair.instance_id;
    const right = document.createElement("span");
    right.className = "query";
    right.textContent = pair.hard ? "HARD" : pair.query;
    if (pair.hard) right.classList.add("hard");
    item.append(label, right);
    item.addEventListener("click", () => selectPair(pair.index));
    list.append(item);
  }
}

async function selectPair(index) {
  STATE.index = index;
  STATE.occlusion = null;
  renderPairs();
  await loadDump();
}

async function loadDump(refresh = false) {
  if (STATE.index === null || !STATE.model) return;
  const meta = el("meta");
  meta.textContent = "running matcher…";
  try {
    const dump = await api("/api/pair", {
      index: STATE.index,
      model: STATE.model,
      refresh: refresh ? 1 : 0,
    });
    STATE.dump = dump;
    renderMeta();
    draw();
  } catch (error) {
    meta.textContent = `error: ${error.message}`;
  }
}

function renderMeta() {
  const dump = STATE.dump;
  const inlierShare = dump.match_count
    ? ((dump.inlier_count / dump.match_count) * 100).toFixed(1)
    : "0.0";
  const residualValues = dump.matches
    .map((match) => match.residual)
    .filter((value) => value !== null);
  const medianResidual = residualValues.length
    ? residualValues.sort((a, b) => a - b)[Math.floor(residualValues.length / 2)]
    : null;
  el("meta").innerHTML = "";
  const parts = [
    ["pair", `${dump.instance_id} / ${dump.query}`],
    ["model", dump.model],
    ["matches (pre-RANSAC)", dump.match_count],
    ["inliers", `${dump.inlier_count} (${inlierShare}%)`],
    ["median residual", `${formatNumber(medianResidual)} px`],
    ["scores", dump.has_scores ? "available" : "not exposed"],
  ];
  for (const [label, value] of parts) {
    const span = document.createElement("span");
    span.innerHTML = `${label}: <b>${value}</b>`;
    el("meta").append(span);
  }
}

function draw() {
  const dump = STATE.dump;
  const viewer = el("viewer");
  if (!dump) return;
  viewer.innerHTML = "";

  const box = viewer.getBoundingClientRect();
  const gap = 48;
  const panelWidth = Math.max(240, (box.width - gap - 24) / 2);
  const maxHeight = Math.max(240, box.height - 28);
  const refScale = Math.min(panelWidth / dump.reference_size[0], maxHeight / dump.reference_size[1]);
  const queryScale = Math.min(panelWidth / dump.query_size[0], maxHeight / dump.query_size[1]);
  const refW = dump.reference_size[0] * refScale;
  const refH = dump.reference_size[1] * refScale;
  const queryW = dump.query_size[0] * queryScale;
  const queryH = dump.query_size[1] * queryScale;
  const queryX = refW + gap;
  const offsetY = (maxHeight - Math.max(refH, queryH)) / 2 + 14;

  const svg = svgEl("svg", { viewBox: `0 0 ${refW + gap + queryW} ${maxHeight + 28}` });
  const root = svgEl("g");
  svg.append(root);

  const image = (href, x, y, width, height) => {
    const node = svgEl("image", {
      href,
      x,
      y,
      width,
      height,
      preserveAspectRatio: "none",
    });
    root.append(node);
  };
  image(`/img/reference?index=${dump.index}`, 0, offsetY, refW, refH);
  image(`/img/query?index=${dump.index}`, queryX, offsetY, queryW, queryH);

  const toRef = (match) => [match.x0 * refScale, match.y0 * refScale + offsetY];
  const toQuery = (match) => [
    queryX + match.x1 * queryScale,
    match.y1 * queryScale + offsetY,
  ];

  if (el("density").checked) {
    drawDensity(root, dump.reference_density, 0, offsetY, refW, refH);
    drawDensity(root, dump.query_density, queryX, offsetY, queryW, queryH);
  }
  if (STATE.occlusion) {
    drawOcclusion(root, STATE.occlusion, 0, offsetY, refW, refH);
  }

  const linkLayer = svgEl("g", { opacity: 0.45 });
  const pointLayer = svgEl("g");
  const residualLayer = svgEl("g", { "stroke-dasharray": "3 3" });
  root.append(linkLayer, residualLayer, pointLayer);

  const labelEvery = Math.max(1, Math.ceil(dump.matches.length / 80));
  dump.matches.forEach((match, position) => {
    const [x0, y0] = toRef(match);
    const [x1, y1] = toQuery(match);
    if (el("links").checked) {
      linkLayer.append(
        svgEl("line", {
          x1: x0,
          y1: y0,
          x2: x1,
          y2: y1,
          stroke: match.inlier ? "#35c07a" : "#ef6b6b",
          "stroke-width": STATE.lineWidth,
        }),
      );
    }
    if (el("residuals").checked && dump.homography) {
      const projected = projectPoint(match, dump.homography, queryScale, queryX, offsetY);
      if (projected) {
        residualLayer.append(
          svgEl("line", {
            x1,
            y1,
            x2: projected[0],
            y2: projected[1],
            stroke: "#f0a94a",
            "stroke-width": STATE.lineWidth,
          }),
        );
      }
    }
    const dot = svgEl("circle", {
      cx: x1,
      cy: y1,
      r: match.inlier ? 3.2 : 2.4,
      fill: match.inlier ? "#35c07a" : "#ef6b6b",
      stroke: "#0b0f14",
      "stroke-width": 0.6,
    });
    dot.addEventListener("mouseenter", (event) => showTooltip(event, match));
    dot.addEventListener("mousemove", (event) => moveTooltip(event));
    dot.addEventListener("mouseleave", hideTooltip);
    dot.addEventListener("click", () => highlight(match, position));
    pointLayer.append(dot);
    if (position % labelEvery === 0) {
      const label = svgEl("text", {
        x: x1 + 5,
        y: y1 - 4,
        fill: "#cfe0f5",
        "font-size": 9,
      });
      label.textContent = `#${match.ref}`;
      pointLayer.append(label);
    }
  });

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML =
    '<span><i style="background:#35c07a"></i>inlier</span>' +
    '<span><i style="background:#ef6b6b"></i>outlier</span>' +
    '<span><i style="background:#f0a94a"></i>residual to homography</span>';
  viewer.append(legend);

  viewer.append(svg);
  attachPanZoom(svg, root);
}

function projectPoint(match, homography, scale, offsetX, offsetY) {
  const h = homography;
  const denominator = h[2][0] * match.x0 + h[2][1] * match.y0 + h[2][2];
  if (!denominator) return null;
  const x = (h[0][0] * match.x0 + h[0][1] * match.y0 + h[0][2]) / denominator;
  const y = (h[1][0] * match.x0 + h[1][1] * match.y0 + h[1][2]) / denominator;
  return [offsetX + x * scale, offsetY + y * scale];
}

function drawDensity(root, grid, x, y, width, height) {
  const rows = grid.length;
  const columns = grid[0]?.length ?? 0;
  if (!rows || !columns) return;
  const layer = svgEl("g", { opacity: 0.45 });
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = grid[row][column];
      if (value <= 0.02) continue;
      layer.append(
        svgEl("rect", {
          x: x + (column / columns) * width,
          y: y + (row / rows) * height,
          width: width / columns,
          height: height / rows,
          fill: `rgba(77,163,255,${Math.min(0.9, value)})`,
        }),
      );
    }
  }
  root.append(layer);
}

function drawOcclusion(root, heat, x, y, width, height) {
  const cells = heat.cells ?? [];
  if (!cells.length) return;
  const rows = cells.length;
  const columns = cells[0].length;
  const layer = svgEl("g", { opacity: 0.55 });
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const survival = cells[row][column];
      layer.append(
        svgEl("rect", {
          x: x + (column / columns) * width,
          y: y + (row / rows) * height,
          width: width / columns,
          height: height / rows,
          fill: `rgba(240,169,74,${Math.min(0.85, survival)})`,
        }),
      );
    }
  }
  root.append(layer);
}

function highlight(match, position) {
  const text = `#${match.ref}↔#${match.query} · (${formatNumber(match.x0, 1)}, ${formatNumber(
    match.y0,
    1,
  )}) → (${formatNumber(match.x1, 1)}, ${formatNumber(match.y1, 1)}) · residual ${formatNumber(
    match.residual,
  )} px`;
  el("meta").dataset.pinned = text;
}

function showTooltip(event, match) {
  const tooltip = el("tooltip");
  tooltip.hidden = false;
  tooltip.innerHTML =
    `<b>#${match.ref}</b> → query #${match.query}<br>` +
    `ref (${formatNumber(match.x0, 1)}, ${formatNumber(match.y0, 1)})<br>` +
    `query (${formatNumber(match.x1, 1)}, ${formatNumber(match.y1, 1)})<br>` +
    `score ${formatNumber(match.score, 3)} · residual ${formatNumber(match.residual)} px<br>` +
    `<span style="color:${match.inlier ? "#35c07a" : "#ef6b6b"}">${
      match.inlier ? "RANSAC inlier" : "RANSAC outlier"
    }</span>`;
  moveTooltip(event);
}

function moveTooltip(event) {
  const tooltip = el("tooltip");
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
}

function hideTooltip() {
  el("tooltip").hidden = true;
}

function attachPanZoom(svg, root) {
  const state = STATE.view;
  const apply = () => {
    root.setAttribute(
      "transform",
      `translate(${state.tx} ${state.ty}) scale(${state.scale})`,
    );
  };
  let dragging = false;
  let origin = null;
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    state.scale = Math.min(8, Math.max(0.5, state.scale * factor));
    apply();
  });
  svg.addEventListener("mousedown", (event) => {
    dragging = true;
    svg.classList.add("dragging");
    origin = [event.clientX - state.tx, event.clientY - state.ty];
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    svg.classList.remove("dragging");
  });
  svg.addEventListener("mousemove", (event) => {
    if (!dragging || !origin) return;
    state.tx = event.clientX - origin[0];
    state.ty = event.clientY - origin[1];
    apply();
  });
  apply();
}

async function runOcclusion() {
  if (STATE.index === null) return;
  const button = el("occlusion");
  button.disabled = true;
  button.textContent = "running…";
  try {
    STATE.occlusion = await api("/api/occlusion", {
      index: STATE.index,
      model: STATE.model,
      grid: 6,
    });
    draw();
  } catch (error) {
    el("meta").textContent = `occlusion failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Occlusion";
  }
}

function wireEvents() {
  el("model").addEventListener("change", (event) => {
    STATE.model = event.target.value;
    loadDump();
  });
  el("search").addEventListener("input", () => loadPairs());
  el("hard-only").addEventListener("change", () => loadPairs());
  for (const id of ["links", "residuals", "density"]) {
    el(id).addEventListener("change", () => draw());
  }
  el("occlusion").addEventListener("click", () => runOcclusion());
  el("tab-inspector").addEventListener("click", () => showView("inspector"));
  el("tab-ablations").addEventListener("click", () => showView("ablations"));
  el("ablation-form").addEventListener("submit", submitAblation);
  el("refresh-runs").addEventListener("click", () => loadAblations());
  el("run-model").addEventListener("change", () => renderRuns());
  el("run-dataset").addEventListener("change", () => renderRuns());
  wireHelp();
  const width = el("line-width");
  width.value = String(STATE.lineWidth);
  width.addEventListener("input", (event) => {
    STATE.lineWidth = Number(event.target.value);
    localStorage.setItem("lineWidth", String(STATE.lineWidth));
    draw();
  });
  window.addEventListener("resize", () => draw());
}

function wireHelp() {
  const wrap = document.querySelector(".help");
  const button = el("help-button");
  const hints = {
    model: "Switch matcher without changing the pair.",
    search: "Filter pairs by instance or query file name.",
    "hard-only": "Show only pairs that no model solves.",
    links: "Draw the pre-RANSAC correspondences.",
    residuals: "Show each match's error against the homography.",
    density: "Overlay where matches concentrate on each image.",
    "line-width": "Thickness of links and residual arrows.",
    occlusion: "Find which reference regions the matches depend on.",
  };
  for (const [id, text] of Object.entries(hints)) {
    const node = el(id);
    if (node) node.title = text;
  }
  button.addEventListener("click", () => {
    const open = wrap.classList.toggle("open");
    button.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      wrap.classList.remove("open");
      button.setAttribute("aria-expanded", "false");
    }
  });
}

function showView(view) {
  const inspector = view === "inspector";
  el("view-inspector").hidden = !inspector;
  el("view-ablations").hidden = inspector;
  el("tab-inspector").classList.toggle("active", inspector);
  el("tab-ablations").classList.toggle("active", !inspector);
  if (!inspector) loadAblations();
}

function metric(run, model) {
  return run.metrics?.[model] ?? null;
}

function baselineRun(rows) {
  return (
    rows.find((run) =>
      Object.entries(DEFAULTS).every(([key, value]) => Number(run.config[key]) === Number(value))
    ) ?? rows[rows.length - 1]
  );
}

async function loadAblations() {
  const data = await api("/api/ablations");
  ABLATION.runs = data.runs;
  ABLATION.jobs = data.jobs;
  ABLATION.factors = data.factors;
  fillSelect(el("ablation-factor"), data.factors, el("ablation-factor").value);
  fillSelect(el("ablation-model"), data.models, el("ablation-model").value || STATE.model);
  const datasets = [...new Set(data.runs.map((run) => run.dataset))].sort();
  fillSelect(el("run-dataset"), datasets, ABLATION.filterDataset || datasets[0]);
  const models = [...new Set(data.runs.flatMap((run) => run.models))].sort();
  fillSelect(el("run-model"), models, ABLATION.filterModel || STATE.model || models[0]);
  renderJobs();
  renderRuns();
  schedulePoll();
}

function fillSelect(select, values, current) {
  const previous = current || select.value;
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  if (previous && values.includes(previous)) select.value = previous;
}

function renderJobs() {
  const body = el("job-table").querySelector("tbody");
  body.innerHTML = "";
  const active = ABLATION.jobs.filter((job) => ["queued", "running"].includes(job.status));
  el("job-hint").textContent = active.length
    ? `${active.length} pending · one run at a time`
    : ABLATION.jobs.length
      ? "idle"
      : "no jobs yet";
  for (const job of [...ABLATION.jobs].reverse()) {
    const row = document.createElement("tr");
    const status = document.createElement("span");
    status.className = `status ${job.status}`;
    status.textContent = job.status;
    row.innerHTML = `<td>${job.label}</td><td></td><td class="mono">${job.output}</td><td></td>`;
    row.children[1].append(status);
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "log";
    const pre = document.createElement("pre");
    pre.textContent = job.log || "(empty)";
    details.append(summary, pre);
    row.children[3].append(details);
    body.append(row);
  }
}

function renderRuns() {
  const model = el("run-model").value;
  const dataset = el("run-dataset").value;
  ABLATION.filterModel = model;
  ABLATION.filterDataset = dataset;
  const rows = ABLATION.runs.filter(
    (run) => run.models.includes(model) && run.dataset === dataset
  );
  const body = el("run-table").querySelector("tbody");
  body.innerHTML = "";
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="12" class="muted">no runs recorded for this selection</td></tr>';
    return;
  }
  const base = baselineRun(rows);
  const baseMetrics = metric(base, model);
  for (const run of rows) {
    const values = metric(run, model);
    if (!values) continue;
    const hit = Number(values.hit_rate ?? NaN);
    const ap = Number(values.ap50 ?? NaN);
    const deltaHit = hit - Number(baseMetrics?.hit_rate ?? NaN);
    const deltaAp = ap - Number(baseMetrics?.ap50 ?? NaN);
    const tr = document.createElement("tr");
    if (run === base) tr.classList.add("baseline");
    tr.innerHTML = `
      <td class="mono">${run.run_id}${run === base ? " · baseline" : ""}</td>
      <td>${run.config.max_keypoints ?? "—"}</td>
      <td>${run.config.ransac_threshold ?? "—"}</td>
      <td>${run.config.max_side ?? "—"}</td>
      <td>${run.config.ratio ?? "—"}</td>
      <td>${run.config.samples ?? "—"}</td>
      <td>${formatNumber(hit, 3)}</td>
      <td>${formatNumber(ap, 3)}</td>
      <td>${formatNumber(values.mean_iou, 3)}</td>
      <td>${formatNumber(values.p50, 0)} ms</td>
      <td class="${deltaClass(deltaHit)}">${signed(deltaHit)}</td>
      <td class="${deltaClass(deltaAp)}">${signed(deltaAp)}</td>`;
    body.append(tr);
  }
}

function deltaClass(value) {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-6) return "muted";
  return value > 0 ? "up" : "down";
}

function signed(value) {
  if (!Number.isFinite(value)) return "—";
  const text = value.toFixed(3);
  return value > 0 ? `+${text}` : text;
}

function schedulePoll() {
  clearTimeout(ABLATION.timer);
  const active = ABLATION.jobs.some((job) => ["queued", "running"].includes(job.status));
  if (!active) return;
  ABLATION.timer = setTimeout(() => loadAblations(), 2500);
}

async function submitAblation(event) {
  event.preventDefault();
  const status = el("ablation-status");
  status.textContent = "queueing…";
  try {
    const payload = {
      factor: el("ablation-factor").value,
      values: el("ablation-values").value,
      models: [el("ablation-model").value],
      max_samples: el("ablation-samples").value || null,
    };
    const response = await fetch("/api/ablation/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    status.textContent = `${body.jobs.length} run(s) queued`;
    await loadAblations();
  } catch (error) {
    status.textContent = `failed: ${error.message}`;
  }
  return false;
}

async function boot() {
  try {
    await loadModels();
    await loadPairs();
    wireEvents();
    if (STATE.pairs.length) {
      await selectPair(STATE.pairs[0].index);
    }
  } catch (error) {
    el("meta").textContent = `startup failed: ${error.message}`;
  }
}

boot();
