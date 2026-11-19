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
  response: null,
  cam: null,
  lineWidth: Number(localStorage.getItem("lineWidth") || 2),
  dataset: null,
  datasets: [],
  labels: localStorage.getItem("ui:labels") !== "0",
};

const REFS = { instances: [], current: null, candidates: [], chosen: null, statuses: [] };

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
  const saved = localStorage.getItem("model");
  STATE.model = STATE.models.includes(saved) ? saved : STATE.models[0] ?? null;
  select.value = STATE.model;
}

async function loadPairs() {
  const payload = await api("/api/pairs", {
    dataset: STATE.dataset,
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
    label.className = "name";
    label.textContent = pair.instance_id;
    const right = document.createElement("span");
    right.className = "meta-right";
    if (pair.hard) {
      const badge = document.createElement("span");
      badge.className = "badge hard";
      badge.textContent = "hard";
      right.append(badge);
    }
    const query = document.createElement("span");
    query.className = "query";
    query.textContent = pair.query;
    right.append(query);
    item.append(label, right);
    item.addEventListener("click", () => selectPair(pair.index));
    list.append(item);
  }
}

async function selectPair(index) {
  STATE.index = index;
  STATE.response = null;
  renderPairs();
  await loadDump();
}

async function loadDump(refresh = false) {
  if (STATE.index === null || !STATE.model) return;
  const meta = el("meta");
  meta.textContent = "running matcher…";
  el("viewer").classList.add("busy");
  try {
    const dump = await api("/api/pair", {
      dataset: STATE.dataset,
      index: STATE.index,
      model: STATE.model,
      refresh: refresh ? 1 : 0,
    });
    STATE.dump = dump;
    setKeypointsAvailable(dump.reference_keypoint_count > 0);
    renderMeta();
    el("viewer").classList.remove("busy");
    draw();
  } catch (error) {
    meta.textContent = `error: ${error.message}`;
  }
}

function setKeypointsAvailable(available) {
  const chip = document.querySelector("label.chip:has(#keypoints)");
  const topk = el("topk");
  if (!chip) return;
  chip.classList.toggle("disabled", !available);
  chip.title = available ? "" : "this matcher detects no keypoints";
  el("keypoints").disabled = !available;
  topk.disabled = !available;
  if (!available) el("keypoints").checked = false;
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
  const keypointInfo = dump.reference_keypoint_count
    ? `${dump.reference_keypoint_count} / ${dump.query_keypoint_count}`
    : "none";
  const parts = [
    ["pair", `${dump.instance_id} / ${dump.query}`],
    ["model", dump.model],
    ["matches (pre-RANSAC)", dump.match_count],
    ["inliers", `${dump.inlier_count} (${inlierShare}%)`],
    ["median residual", `${formatNumber(medianResidual)} px`],
    ["scores", dump.has_scores ? "available" : "not exposed"],
    ["keypoints", keypointInfo],
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
  const dataset = encodeURIComponent(STATE.dataset ?? "");
  const revision = dump.revision ?? 0;
  image(`/img/reference?index=${dump.index}&dataset=${dataset}&rev=${revision}`, 0, offsetY, refW, refH);
  image(`/img/query?index=${dump.index}&dataset=${dataset}&rev=${revision}`, queryX, offsetY, queryW, queryH);

  const toRef = (match) => [match.x0 * refScale, match.y0 * refScale + offsetY];
  const toQuery = (match) => [
    queryX + match.x1 * queryScale,
    match.y1 * queryScale + offsetY,
  ];

  if (el("keypoints").checked) {
    drawKeypoints(root, dump.reference_keypoints, 0, offsetY, refW, refH, "ref");
    drawKeypoints(root, dump.query_keypoints, queryX, offsetY, queryW, queryH, "query");
  }
  if (el("density").checked) {
    drawDensity(root, dump.reference_density, 0, offsetY, refW, refH);
    drawDensity(root, dump.query_density, queryX, offsetY, queryW, queryH);
  }
  if (STATE.cam) {
    drawCam(root, STATE.cam.reference, 0, offsetY, refW, refH);
    drawCam(root, STATE.cam.query, queryX, offsetY, queryW, queryH);
  }
  if (STATE.response) {
    drawResponse(root, STATE.response.reference, 0, offsetY, refW, refH);
    drawResponse(root, STATE.response.query, queryX, offsetY, queryW, queryH);
  }

  const linkOpacity = Math.min(0.45, Math.max(0.1, 30 / Math.max(1, dump.matches.length)));
  const linkLayer = svgEl("g", { opacity: linkOpacity });
  const pointLayer = svgEl("g");
  const residualLayer = svgEl("g", { "stroke-dasharray": "3 3" });
  root.append(linkLayer, residualLayer, pointLayer);

  const labelBudget = 60;
  const placed = [];
  const labelled = new Set();
  const byScore = [...dump.matches.keys()].sort(
    (a, b) => (dump.matches[b].score ?? 0) - (dump.matches[a].score ?? 0),
  );
  for (const position of byScore) {
    if (labelled.size >= labelBudget) break;
    const [, y] = toQuery(dump.matches[position]);
    const [, x] = toQuery(dump.matches[position]);
    const crowded = placed.some(
      ([px, py]) => Math.abs(px - x) < 34 && Math.abs(py - y) < 15,
    );
    if (crowded) continue;
    placed.push([x, y]);
    labelled.add(position);
  }
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
    dot.addEventListener("click", () => focusMatch(match));
    pointLayer.append(dot);
    if (STATE.labels && labelled.has(position)) {
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
    '<span><i style="background:#f0a94a"></i>residual to homography</span>' +
    (STATE.response ? '<span><i style="background:#eb463c"></i>response (red = depended on)</span>' : '');
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

function drawKeypoints(root, points, x, y, width, height, side) {
  if (!points?.length) return;
  const limit = Number(el("topk").value) || points.length;
  const shown = points.slice(0, limit);
  const scores = shown.map((point) => point[2]);
  const peak = Math.max(...scores, 1e-6);
  const layer = svgEl("g", { opacity: 0.85 });
  for (const [px, py, score] of shown) {
    const relative = Math.max(0.15, score / peak);
    layer.append(
      svgEl("circle", {
        cx: x + (px / (side === "ref" ? STATE.dump.reference_size[0] : STATE.dump.query_size[0])) * width,
        cy: y + (py / (side === "ref" ? STATE.dump.reference_size[1] : STATE.dump.query_size[1])) * height,
        r: 1.4 + 2.6 * relative,
        fill: side === "ref" ? "#9fd0ff" : "#b9f0d0",
        "fill-opacity": 0.45 + 0.45 * relative,
        stroke: "none",
      }),
    );
  }
  root.append(layer);
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

function drawCam(root, grid, x, y, width, height) {
  const rows = grid?.length ?? 0;
  if (!rows) return;
  const columns = grid[0].length;
  const layer = svgEl("g", { opacity: 0.55 });
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = Math.max(0, Math.min(1, grid[row][column]));
      if (value < 0.06) continue;
      layer.append(
        svgEl("rect", {
          x: x + (column / columns) * width,
          y: y + (row / rows) * height,
          width: width / columns + 0.5,
          height: height / rows + 0.5,
          fill: camColor(value),
        }),
      );
    }
  }
  root.append(layer);
}

// Turbo-ish ramp: blue → cyan → green → yellow → red.
function camColor(value) {
  const stops = [
    [0.0, [40, 60, 190]],
    [0.25, [0, 170, 200]],
    [0.5, [60, 200, 90]],
    [0.75, [250, 210, 60]],
    [1.0, [235, 55, 45]],
  ];
  let lower = stops[0];
  let upper = stops[stops.length - 1];
  for (let index = 0; index < stops.length - 1; index += 1) {
    if (value >= stops[index][0] && value <= stops[index + 1][0]) {
      lower = stops[index];
      upper = stops[index + 1];
      break;
    }
  }
  const span = Math.max(upper[0] - lower[0], 1e-6);
  const t = (value - lower[0]) / span;
  const colour = lower[1].map((channel, index) =>
    Math.round(channel + (upper[1][index] - channel) * t),
  );
  return `rgba(${colour[0]},${colour[1]},${colour[2]},${0.35 + 0.55 * value})`;
}

function drawResponse(root, heat, x, y, width, height) {
  const cells = heat?.cells ?? [];
  if (!cells.length) return;
  const rows = cells.length;
  const columns = cells[0].length;
  const peak = Math.max(...cells.flat(), 0);
  if (peak <= 0) return;
  const layer = svgEl("g");
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = cells[row][column] / peak;
      if (value < 0.08) continue;
      layer.append(
        svgEl("rect", {
          x: x + (column / columns) * width,
          y: y + (row / rows) * height,
          width: width / columns,
          height: height / rows,
          fill: responseColor(value),
        }),
      );
    }
  }
  root.append(layer);
}

// Amber (weak) to red (the match set depends on this cell). Values are
// normalized per image, so the map shows where, not how much overall.
function responseColor(value) {
  const alpha = 0.16 + 0.62 * value;
  const red = 255;
  const green = Math.round(196 - 130 * value);
  const blue = Math.round(60 - 40 * value);
  return `rgba(${red},${green},${blue},${alpha.toFixed(3)})`;
}

// Clicking a point pins its numbers into the meta bar so they can be read
// without holding the pointer still over a moving element.
function focusMatch(match) {
  const pinned = el("meta").querySelector(".pinned");
  if (pinned) pinned.remove();
  const chip = document.createElement("span");
  chip.className = "pinned";
  chip.innerHTML =
    `pinned <b>#${match.ref} → #${match.query}</b>` +
    ` · (${formatNumber(match.x0, 1)}, ${formatNumber(match.y0, 1)}) → ` +
    `(${formatNumber(match.x1, 1)}, ${formatNumber(match.y1, 1)})` +
    ` · residual ${formatNumber(match.residual)} px`;
  el("meta").append(chip);
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

async function loadLayers() {
  const select = el("layer");
  const button = el("gradcam");
  select.innerHTML = "";
  button.disabled = true;
  try {
    const payload = await api("/api/layers", { model: STATE.model });
    if (!payload.supported) {
      const option = document.createElement("option");
      option.textContent = "not available";
      select.append(option);
      button.title = "Grad-CAM needs a convolution backbone";
      return;
    }
    for (const layer of payload.layers) {
      const option = document.createElement("option");
      option.value = layer.name;
      option.textContent = `${layer.name} · ${layer.kind}`;
      select.append(option);
    }
    select.value = payload.layers[payload.layers.length - 1].name;
    button.disabled = false;
    button.title = "Saliency of the selected layer";
  } catch (error) {
    button.title = `layers unavailable: ${error.message}`;
  }
}

async function runGradcam() {
  if (STATE.index === null) return;
  const button = el("gradcam");
  button.disabled = true;
  const previous = button.textContent;
  button.textContent = "running…";
  el("meta").textContent = "computing Grad-CAM…";
  try {
    STATE.cam = await api("/api/gradcam", {
      dataset: STATE.dataset,
      index: STATE.index,
      model: STATE.model,
      layer: el("layer").value,
    });
    if (!STATE.cam.supported) {
      el("meta").textContent = `${STATE.model}: Grad-CAM not available`;
      STATE.cam = null;
    } else {
      renderMeta();
    }
    draw();
  } catch (error) {
    el("meta").textContent = `grad-cam failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = previous;
  }
}

async function runResponse() {
  if (STATE.index === null) return;
  const button = el("response");
  button.disabled = true;
  const previous = button.textContent;
  button.textContent = "running…";
  el("meta").textContent = "running response sweep…";
  try {
    STATE.response = await api("/api/response", {
      dataset: STATE.dataset,
      index: STATE.index,
      model: STATE.model,
      grid: 6,
    });
    renderMeta();
    draw();
  } catch (error) {
    el("meta").textContent = `response failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = previous;
  }
}

function persist(id, value) {
  localStorage.setItem(`ui:${id}`, value);
}

function restorePersisted() {
  for (const id of [
    "links",
    "residuals",
    "density",
    "labels",
    "keypoints",
    "hard-only",
    "line-width",
  ]) {
    const saved = localStorage.getItem(`ui:${id}`);
    const node = el(id);
    if (!node || saved === null) continue;
    if (node.type === "checkbox") node.checked = saved === "1";
    else node.value = saved;
  }
  STATE.lineWidth = Number(el("line-width").value);
  STATE.labels = el("labels").checked;
  el("keypoints").addEventListener("change", () => draw());
  el("topk").addEventListener("change", () => draw());
}

function wireEvents() {
  restorePersisted();
  el("model").addEventListener("change", (event) => {
    STATE.model = event.target.value;
    persist("model", STATE.model);
    STATE.cam = null;
    STATE.response = null;
    loadLayers();
    loadDump();
  });
  el("search").addEventListener("input", () => loadPairs());
  el("hard-only").addEventListener("change", () => {
    persist("hard-only", el("hard-only").checked ? "1" : "0");
    loadPairs();
  });
  for (const id of ["links", "residuals", "density", "labels", "keypoints"]) {
    el(id).addEventListener("change", () => {
      persist(id, el(id).checked ? "1" : "0");
      if (id === "labels") STATE.labels = el(id).checked;
      draw();
    });
  }
  el("response").addEventListener("click", () => runResponse());
  el("gradcam").addEventListener("click", () => runGradcam());
  el("layer").addEventListener("change", () => {
    if (STATE.cam) runGradcam();
  });
  el("tab-inspector").addEventListener("click", () => showView("inspector"));
  el("tab-references").addEventListener("click", () => showView("references"));
  el("tab-ablations").addEventListener("click", () => showView("ablations"));
  el("dataset").addEventListener("change", async (event) => {
    STATE.dataset = event.target.value;
    persist("dataset", STATE.dataset);
    STATE.index = null;
    STATE.dump = null;
    await loadPairs();
    if (STATE.pairs.length) await selectPair(STATE.pairs[0].index);
  });
  el("ref-save").addEventListener("click", saveReference);
  el("reload").addEventListener("click", reloadData);
  el("ablation-form").addEventListener("submit", submitAblation);
  el("refresh-runs").addEventListener("click", () => loadAblations());
  el("run-model").addEventListener("change", () => renderRuns());
  el("run-dataset").addEventListener("change", () => renderRuns());
  wireHelp();
  const width = el("line-width");
  width.value = String(STATE.lineWidth);
  width.addEventListener("input", (event) => {
    STATE.lineWidth = Number(event.target.value);
    persist("line-width", event.target.value);
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
    labels: "Tag the strongest query points with their reference index.",
    residuals: "Show each match's error against the homography.",
    density: "Overlay where matches concentrate on each image.",
    keypoints: "Show detected keypoints, sized by detector response.",
    topk: "How many keypoints per image to show.",
    "line-width": "Thickness of links and residual arrows.",
    response: "Show which image regions the matches depend on.",
    layer: "Grad-CAM layer to visualise.",
    gradcam: "Gradient saliency of the selected layer.",
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
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(
      document.activeElement?.tagName ?? ""
    );
    if (event.key === "Escape") {
      wrap.classList.remove("open");
      button.setAttribute("aria-expanded", "false");
      return;
    }
    if (typing) return;
    if (event.key === "?") {
      wrap.classList.toggle("open");
      button.setAttribute("aria-expanded", String(wrap.classList.contains("open")));
      return;
    }
    const layers = {
      l: "links",
      r: "residuals",
      d: "density",
      n: "labels",
      k: "keypoints",
    };
    const layer = layers[event.key.toLowerCase()];
    if (layer) {
      const node = el(layer);
      node.checked = !node.checked;
      persist(layer, node.checked ? "1" : "0");
      draw();
    }
  });
}

function initialRoute() {
  const params = new URLSearchParams(window.location.search);
  const hash = (window.location.hash || "").replace("#", "");
  const views = ["inspector", "references", "ablations"];
  return {
    view: views.includes(hash) ? hash : "inspector",
    help: params.get("help") === "1",
    dataset: params.get("dataset"),
    model: params.get("model"),
    pair: params.get("pair"),
    instance: params.get("instance"),
  };
}

function showView(view) {
  const views = {
    inspector: el("view-inspector"),
    references: el("view-references"),
    ablations: el("view-ablations"),
  };
  for (const [name, node] of Object.entries(views)) {
    node.hidden = name !== view;
    el(`tab-${name}`).classList.toggle("active", name === view);
  }
  if (window.location.hash !== `#${view}`) {
    history.replaceState(null, "", `${window.location.pathname}${window.location.search}#${view}`);
  }
  if (view === "ablations") loadAblations();
  if (view === "references") loadReferences();
}

function metric(run, model) {
  return run.metrics?.[model] ?? null;
}

function baselineRun(rows) {
  const defaults = rows.filter((run) =>
    Object.entries(DEFAULTS).every(([key, value]) => Number(run.config[key]) === Number(value))
  );
  const pool = defaults.length ? defaults : rows;
  return [...pool].sort(
    (a, b) => Number(b.config.samples ?? 0) - Number(a.config.samples ?? 0)
  )[0];
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

async function reloadData() {
  const button = el("reload");
  button.disabled = true;
  button.textContent = "reloading…";
  try {
    const response = await fetch("/api/reload", { method: "POST" });
    if (!response.ok) throw new Error(response.statusText);
    STATE.dump = null;
    STATE.response = null;
    STATE.cam = null;
    await loadDatasets();
    await loadPairs();
    if (STATE.index !== null) await loadDump(true);
    if (!el("view-references").hidden) await loadReferences();
    el("reload").textContent = "Reload data";
  } catch (error) {
    el("reload").textContent = "Reload data";
    el("meta").textContent = `reload failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function loadDatasets() {
  const payload = await api("/api/datasets");
  STATE.datasets = payload.datasets;
  const select = el("dataset");
  select.innerHTML = "";
  for (const dataset of payload.datasets) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent =
      `${dataset.name} · ${dataset.pairs} pairs` +
      (dataset.selectable ? "" : " · single reference");
    select.append(option);
  }
  const saved = localStorage.getItem("ui:dataset");
  const names = payload.datasets.map((dataset) => dataset.name);
  STATE.dataset = names.includes(saved) ? saved : names[0] ?? null;
  select.value = STATE.dataset;
}

async function loadReferences() {
  if (!STATE.dataset) return;
  const payload = await api("/api/references", { dataset: STATE.dataset });
  REFS.instances = payload.instances;
  REFS.statuses = payload.statuses;
  REFS.selectable = payload.selectable;
  document
    .querySelector(".references")
    ?.classList.toggle("single", !payload.selectable);
  if (!payload.selectable) {
    REFS.current = null;
    el("ref-counts").textContent = "single reference per instance";
    el("ref-status").textContent = "";
    el("ref-title").textContent = `${payload.dataset}: nothing to choose`;
    el("ref-subtitle").textContent =
      "every instance has exactly one reference, so it is not shown here";
    el("cand-grid").innerHTML =
      '<p class="empty">This dataset has one reference per instance, so there are no alternatives to pick from. Use the Inspector tab to look at them.</p>';
    el("ref-form").hidden = true;
    el("ref-help").hidden = true;
    el("ref-list").innerHTML = "";
    return;
  }
  el("ref-form").hidden = false;
  el("ref-help").hidden = false;
  el("ref-counts").textContent = `${payload.instances.length} instances`;
  const flagged = payload.instances.filter((item) => item.status !== "ok").length;
  el("ref-status").textContent = flagged ? `${flagged} flagged` : "all ok";
  fillSelect(el("ref-flag"), payload.statuses, el("ref-flag").value || "ok");
  renderRefList();
}

function renderRefList() {
  const list = el("ref-list");
  list.innerHTML = "";
  for (const item of REFS.instances) {
    const row = document.createElement("li");
    if (item.instance === REFS.current) row.classList.add("active");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.instance;
    const right = document.createElement("span");
    right.className = "meta-right";
    if (item.status !== "ok") {
      const badge = document.createElement("span");
      badge.className = "badge hard";
      badge.textContent = item.status;
      right.append(badge);
    }
    const count = document.createElement("span");
    count.className = "query";
    count.textContent = `${item.pairs}`;
    right.append(count);
    row.append(name, right);
    row.addEventListener("click", () => selectInstance(item.instance));
    list.append(row);
  }
}

async function selectInstance(instance) {
  REFS.current = instance;
  const info = REFS.instances.find((item) => item.instance === instance);
  REFS.chosen = info?.selected_image
    ? { image: info.selected_image }
    : { image: info?.image };
  el("ref-title").textContent = instance;
  el("ref-flag").value = info?.status ?? "ok";
  el("ref-note").value = info?.note ?? "";
  renderRefList();
  const payload = await api("/api/reference/candidates", {
    dataset: STATE.dataset,
    instance,
  });
  REFS.candidates = payload.candidates;
  renderCandidates();
}

function renderCandidates() {
  const grid = el("cand-grid");
  grid.innerHTML = "";
  if (!REFS.candidates.length) {
    grid.innerHTML = '<p class="empty">No candidate crops for this instance.</p>';
    return;
  }
  el("ref-subtitle").textContent =
    `${REFS.candidates.length} candidate crop(s) · click to choose`;
  for (const candidate of REFS.candidates) {
    const card = document.createElement("button");
    card.className = "cand";
    card.type = "button";
    const active = REFS.chosen?.image === candidate.image;
    if (active) card.classList.add("active");
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `${candidate.url}&dataset=${encodeURIComponent(STATE.dataset)}`;
    img.alt = `${candidate.image} crop`;
    const meta = document.createElement("span");
    meta.className = "cand-meta";
    meta.textContent = `${candidate.width}×${candidate.height}`;
    if (candidate.current) {
      const badge = document.createElement("span");
      badge.className = "badge quiet";
      badge.textContent = "in use";
      meta.append(" ", badge);
    } else if (REFS.chosen?.image === candidate.image) {
      const badge = document.createElement("span");
      badge.className = "badge hard";
      badge.textContent = "saved";
      meta.append(" ", badge);
    }
    card.append(img, meta);
    card.addEventListener("click", () => {
      REFS.chosen = { image: candidate.image, box: candidate.box };
      renderCandidates();
    });
    grid.append(card);
  }
}

async function saveReference() {
  if (!REFS.current) return;
  const button = el("ref-save");
  button.disabled = true;
  try {
    const response = await fetch("/api/references", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: STATE.dataset,
        instance: REFS.current,
        image: REFS.chosen?.image,
        box: REFS.chosen?.box,
        status: el("ref-flag").value,
        note: el("ref-note").value,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    el("ref-saved").textContent = "saved";
    await loadReferences();
    renderRefList();
  } catch (error) {
    el("ref-saved").textContent = `failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function boot() {
  try {
    restorePersisted();
    const route = initialRoute();
    await loadModels();
    await loadDatasets();
    if (route.dataset && STATE.datasets.some((item) => item.name === route.dataset)) {
      STATE.dataset = route.dataset;
      el("dataset").value = STATE.dataset;
    }
    if (route.model && STATE.models.includes(route.model)) {
      STATE.model = route.model;
      el("model").value = STATE.model;
    }
    await loadLayers();
    wireEvents();
    showView(route.view);
    if (route.help) {
      document.querySelector(".help")?.classList.add("open");
      el("help-button")?.setAttribute("aria-expanded", "true");
    }
    if (route.view === "inspector") {
      await loadPairs();
      if (STATE.pairs.length) {
        const match = route.pair
          ? STATE.pairs.find((pair) => String(pair.index) === route.pair)
          : null;
        await selectPair(match ? match.index : STATE.pairs[0].index);
      }
      return;
    }
    if (route.view === "references") {
      await loadReferences();
      const instance = route.instance;
      if (instance && REFS.instances.some((item) => item.instance === instance)) {
        await selectInstance(instance);
      } else if (REFS.instances.length) {
        await selectInstance(REFS.instances[0].instance);
      }
      return;
    }
    await loadAblations();
  } catch (error) {
    el("meta").textContent = `startup failed: ${error.message}`;
  }
}

boot();
