/* LeafGuard frontend.
 *
 * One script for all three pages; it looks at which elements exist and wires up only
 * those. The API address is configurable: nginx proxies /api to the backend, and setting
 * window.LEAFGUARD_API before this script loads points the page somewhere else instead.
 * The browser never loads the model - it only ever calls the REST API.
 */

const API = window.LEAFGUARD_API || "/api/v1";
const MAX_MB = 5;
const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
const HISTORY_KEY = "leafguard.history";

/* ------------------------------------------------------------------ helpers */

const $ = (id) => document.getElementById(id);

/* Text from the API is escaped before it goes into innerHTML, so an odd class name or
   error message can never break the page or inject markup. */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

async function callApi(path, options) {
  const reply = await fetch(`${API}${path}`, options);
  let body = null;
  try {
    body = await reply.json();
  } catch (err) {
    body = null;
  }
  if (!reply.ok) {
    const message = body && body.error ? body.error.message : "The service is unavailable.";
    throw new Error(message);
  }
  return body;
}

function readableSize(bytes) {
  return bytes < 1024 * 1024
    ? `${Math.round(bytes / 1024)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function shortTime(text) {
  if (!text) return "-";
  const when = new Date(text);
  return Number.isNaN(when.getTime()) ? text : when.toLocaleString();
}

/* localStorage can be unavailable or full, and a broken history must never stop the
   page working, so every access is wrapped. */
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch (err) {
    return [];
  }
}

function saveHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 5)));
  } catch (err) {
    /* nothing to do: history is a convenience, not a requirement */
  }
}

/* ------------------------------------------------------------------ diagnose page */

function setupDiagnose() {
  const dropzone = $("dropzone");
  if (!dropzone) return;

  const fileInput = $("file-input");
  const previewArea = $("preview-area");
  const preview = $("preview");
  const diagnoseBtn = $("diagnose-btn");
  const uploadError = $("upload-error");
  const statusLine = $("status-line");
  const result = $("result");

  let chosenFile = null;
  let lastRequestId = null;

  function showError(message) {
    uploadError.textContent = message;
    uploadError.classList.remove("hidden");
  }

  function clearError() {
    uploadError.classList.add("hidden");
  }

  function acceptFile(file) {
    clearError();
    if (!file) return;

    if (!ALLOWED.includes(file.type)) {
      showError("That file type is not supported. Please choose a JPG, PNG or WEBP photo.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      showError(`That photo is ${readableSize(file.size)}. Please choose one under ${MAX_MB} MB.`);
      return;
    }

    chosenFile = file;
    preview.src = URL.createObjectURL(file);
    $("file-name").textContent = file.name;
    $("file-size").textContent = readableSize(file.size);
    dropzone.classList.add("hidden");
    previewArea.classList.remove("hidden");
    diagnoseBtn.disabled = false;
    result.classList.add("hidden");
  }

  function resetChoice() {
    chosenFile = null;
    fileInput.value = "";
    previewArea.classList.add("hidden");
    dropzone.classList.remove("hidden");
    diagnoseBtn.disabled = true;
    result.classList.add("hidden");
    clearError();
  }

  /* The upload button opens the file picker. */
  $("gallery-btn").addEventListener("click", () => fileInput.click());
  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("over");
    })
  );
  dropzone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));
  fileInput.addEventListener("change", (event) => acceptFile(event.target.files[0]));
  $("change-btn").addEventListener("click", resetChoice);

  function showResult(data) {
    lastRequestId = data.request_id;
    const uncertain = data.status === "uncertain";
    const healthy = data.predicted_class === "healthy";
    const percent = Math.round(data.confidence * 100);

    /* One coloured label: green = healthy, red = disease found, yellow = not sure. */
    const state = uncertain ? "unsure" : healthy ? "good" : "sick";
    const label = $("verdict-label");
    label.className = `label ${state}`;
    label.textContent = { good: "Healthy", sick: "Disease found", unsure: "Not sure" }[state];
    result.className = `card result ${state}`;

    $("result-name").textContent = data.display_name;
    $("result-description").textContent = uncertain
      ? "The photo was not clear enough to be sure. Please take a new photo."
      : "";

    $("confidence-text").textContent = `${percent}%`;
    const bar = $("confidence-bar");
    bar.style.width = `${percent}%`;
    bar.className = uncertain ? "fill low" : "fill";

    $("advice-box").classList.toggle("hidden", uncertain || !data.advice);
    $("advice-text").textContent = data.advice || "";
    $("hint-box").classList.toggle("hidden", !uncertain);
    $("hint-text").textContent = data.hint || "";

    $("top-list").innerHTML = data.top_k
      .map((item) => `<li><span>${esc(item.display_name)}</span>` +
                     `<span>${Math.round(item.confidence * 100)}%</span></li>`)
      .join("");

    $("disclaimer").textContent = data.disclaimer;
    $("feedback-done").classList.add("hidden");
    $("correction").classList.add("hidden");
    ["yes-btn", "no-btn"].forEach((id) => $(id).setAttribute("aria-pressed", "false"));
    result.classList.remove("hidden");
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });

    const items = loadHistory();
    items.unshift({ name: data.display_name, percent, status: data.status,
                    at: new Date().toISOString() });
    saveHistory(items);
    renderHistory();
  }

  diagnoseBtn.addEventListener("click", async () => {
    if (!chosenFile) return;
    clearError();
    diagnoseBtn.disabled = true;
    $("diagnose-label").textContent = "Checking...";
    statusLine.textContent = "This usually takes under a second.";
    statusLine.classList.remove("hidden");

    const form = new FormData();
    form.append("file", chosenFile);
    try {
      showResult(await callApi("/predict", { method: "POST", body: form }));
    } catch (err) {
      showError(err.message);
    } finally {
      diagnoseBtn.disabled = false;
      $("diagnose-label").textContent = "Check this leaf";
      statusLine.classList.add("hidden");
    }
  });

  async function sendFeedback(correct, correctClass) {
    try {
      await callApi("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: lastRequestId, correct,
                               correct_class: correctClass || null }),
      });
      $("feedback-done").textContent = "Thank you. This helps improve LeafGuard.";
    } catch (err) {
      $("feedback-done").textContent = err.message;
    }
    $("feedback-done").classList.remove("hidden");
    $("correction").classList.add("hidden");
  }

  $("yes-btn").addEventListener("click", () => {
    $("yes-btn").setAttribute("aria-pressed", "true");
    $("no-btn").setAttribute("aria-pressed", "false");
    sendFeedback(true, null);
  });

  $("no-btn").addEventListener("click", () => {
    $("no-btn").setAttribute("aria-pressed", "true");
    $("yes-btn").setAttribute("aria-pressed", "false");
    $("correction").classList.remove("hidden");
    $("feedback-done").classList.add("hidden");
  });

  $("send-feedback").addEventListener("click", () =>
    sendFeedback(false, $("correct-class").value)
  );

  function renderHistory() {
    const items = loadHistory();
    $("history-card").classList.toggle("hidden", items.length === 0);
    $("history").innerHTML = items
      .map((item) => `<li><span>${esc(item.name)}` +
                     `${item.status === "uncertain" ? " (uncertain)" : ""}</span>` +
                     `<span>${item.percent}%</span></li>`)
      .join("");
  }

  async function loadReference() {
    try {
      const classes = await callApi("/classes");
      $("correct-class").innerHTML = classes
        .map((item) => `<option value="${esc(item.name)}">${esc(item.display_name)}</option>`)
        .join("");
    } catch (err) {
      $("correct-class").innerHTML = "<option>Could not load the class list</option>";
    }
    try {
      const info = await callApi("/model/info");
      $("model-line").textContent = `Model ${info.name} version ${info.version} (champion)`;
    } catch (err) {
      $("model-line").textContent = "The diagnosis service is still starting up.";
    }
  }

  renderHistory();
  loadReference();
}

/* ------------------------------------------------------------------ pipeline console */

function statusTag(status) {
  if (status === "success") return '<span class="tag ok">success</span>';
  if (status === "failed") return '<span class="tag bad">failed</span>';
  return `<span class="tag idle">${esc(status || "unknown")}</span>`;
}

function factList(pairs) {
  return `<ul class="facts">${pairs
    .map(([label, value]) => `<li><span>${esc(label)}</span><span>${value}</span></li>`)
    .join("")}</ul>`;
}

function setupPipeline() {
  const stageTable = $("stage-table");
  if (!stageTable) return;

  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false, theme: "neutral" });
  }

  /* Point each tool link at the same host the page was loaded from, on the port that
     config.js (written from docker-compose settings) says the tool listens on. */
  const ports = Object.assign(
    { mlflow: "5000", grafana: "3000", prometheus: "9090", api_docs: "8000" },
    window.LEAFGUARD_TOOLS || {}
  );
  document.querySelectorAll("a[data-tool]").forEach((link) => {
    const port = ports[link.dataset.tool];
    link.href = `${location.protocol}//${location.hostname}:${port}${link.dataset.path || ""}`;
  });

  async function refresh() {
    let summary;
    try {
      summary = await callApi("/pipeline/summary");
    } catch (err) {
      stageTable.querySelector("tbody").innerHTML =
        `<tr><td colspan="7" class="wrap">${esc(err.message)}</td></tr>`;
      return;
    }

    stageTable.querySelector("tbody").innerHTML = summary.stages.length
      ? summary.stages
          .map((run) => `<tr>
              <td><strong>${esc(run.stage)}</strong></td>
              <td>${statusTag(run.status)}</td>
              <td>${run.duration_s.toFixed(1)} s</td>
              <td>${run.items}</td>
              <td>${run.items_per_s ? run.items_per_s.toFixed(1) : "-"}</td>
              <td>${esc(shortTime(run.started_at))}</td>
              <td class="wrap">${esc(run.error || "-")}</td>
            </tr>`)
          .join("")
      : '<tr><td colspan="7">No runs recorded yet. Run `dvc repro` to populate this.</td></tr>';

    const gate = summary.gate;
    $("gate-box").innerHTML = gate
      ? factList([
          ["Decision", gate.promoted
            ? '<span class="tag ok">promoted</span>'
            : '<span class="tag bad">not promoted</span>'],
          ["Candidate version", esc(gate.candidate_version || "-")],
          ["Now serving version", esc(gate.serving_version || "-")],
          ["Candidate accuracy", esc(gate.candidate_accuracy ?? "-")],
          ["Previous champion accuracy", esc(gate.previous_champion_accuracy ?? "none")],
          ["p95 latency", gate.candidate_latency_p95_ms
            ? `${gate.candidate_latency_p95_ms} ms` : "-"],
        ]) +
        `<ul class="reasons">${(gate.reasons || [])
          .map((line) => `<li>${esc(line)}</li>`).join("")}</ul>`
      : '<p class="muted">No decision recorded yet.</p>';

    const quality = summary.data_quality;
    $("quality-box").innerHTML = quality
      ? factList([
          ["Images read", quality.total_images],
          ["Valid", quality.valid_images],
          ["Rejected", quality.rejected_images],
          ["Reasons", esc(Object.entries(quality.rejected_by_reason || {})
            .map(([name, count]) => `${name}: ${count}`).join(", ") || "none")],
          ["Validation speed", `${quality.images_per_second} images/sec`],
        ])
      : '<p class="muted">Run the pipeline to see this.</p>';

    const throughput = summary.throughput;
    const metrics = summary.metrics;
    $("throughput-box").innerHTML = throughput
      ? factList([
          ["Images processed", throughput.processed_images],
          ["Resize speed", `${throughput.images_per_second} images/sec`],
          ["Spark master", esc(throughput.spark_master)],
          ["Partitions", throughput.partitions],
          ["Test accuracy", metrics ? metrics.accuracy : "-"],
          ["Test macro F1", metrics ? metrics.macro_f1 : "-"],
        ])
      : '<p class="muted">Run the pipeline to see this.</p>';

    const versions = await callApi("/models").catch(() => []);
    const mlflowBase = `${location.protocol}//${location.hostname}:${ports.mlflow}`;
    $("model-table").querySelector("tbody").innerHTML = versions.length
      ? versions
          .map((row) => `<tr>
              <td><a href="${mlflowBase}/#/models/leafguard/versions/${
                encodeURIComponent(row.version)}" target="_blank" rel="noopener"
                ><strong>v${esc(row.version)}</strong></a></td>
              <td>${row.aliases.length
                ? row.aliases.map((name) => `<span class="tag ok">${esc(name)}</span>`).join(" ")
                : "-"}</td>
              <td>${row.gate === "passed"
                ? '<span class="tag ok">passed</span>'
                : row.gate === "failed" ? '<span class="tag bad">failed</span>' : "-"}</td>
              <td>${esc(row.accuracy ?? "-")}</td>
              <td>${esc(row.git_commit || "-")}</td>
              <td>${esc(shortTime(row.created_at))}</td>
            </tr>`)
          .join("")
      : '<tr><td colspan="6">No model registered yet. Run `dvc repro` to train one.</td></tr>';

    await refreshRuns();
  }

  /* Run history honours the stage and status filters; the counts line doubles as the
     console's tally of successful and failed runs. */
  async function refreshRuns() {
    const stage = $("stage-filter").value;
    const wanted = $("status-filter").value;
    const query = stage ? `&stage=${encodeURIComponent(stage)}` : "";
    const all = await callApi(`/pipeline/runs?limit=200${query}`).catch(() => []);
    const ok = all.filter((run) => run.status === "success").length;
    const failed = all.filter((run) => run.status === "failed").length;
    $("run-counts").textContent = `${all.length} runs: ${ok} succeeded, ${failed} failed`;
    const runs = (wanted ? all.filter((run) => run.status === wanted) : all).slice(0, 60);
    $("run-table").querySelector("tbody").innerHTML = runs.length
      ? runs
          .map((run) => `<tr>
              <td>${esc(shortTime(run.started_at))}</td>
              <td>${esc(run.stage)}</td>
              <td>${statusTag(run.status)}</td>
              <td>${run.duration_s.toFixed(1)} s</td>
              <td>${run.items}</td>
              <td>${esc(run.git_commit || "-")}</td>
              <td class="wrap">${esc(run.error || "-")}</td>
            </tr>`)
          .join("")
      : '<tr><td colspan="7">No runs recorded yet.</td></tr>';
  }

  async function drawGraph() {
    try {
      const reply = await fetch(`${API}/pipeline/dag`);
      const text = await reply.text();
      if (window.mermaid) {
        const drawn = await window.mermaid.render("dagGraph", text);
        $("dag").innerHTML = drawn.svg;
      } else {
        $("dag").innerHTML = `<pre>${esc(text)}</pre>`;
      }
    } catch (err) {
      $("dag").innerHTML = '<p class="muted">The pipeline graph is not available.</p>';
    }
  }

  $("stage-filter").addEventListener("change", refreshRuns);
  $("status-filter").addEventListener("change", refreshRuns);
  refresh();
  drawGraph();
  setInterval(refresh, 10000);
}

/* ------------------------------------------------------------------ help page */

async function setupHelp() {
  const box = $("class-list");
  if (!box) return;
  try {
    const classes = await callApi("/classes");
    box.innerHTML = classes
      .map((item) => `<p><strong>${esc(item.display_name)}.</strong> ${esc(item.description)}</p>`)
      .join("");
  } catch (err) {
    box.innerHTML = '<p class="muted">The class list could not be loaded right now.</p>';
  }
}

setupDiagnose();
setupPipeline();
setupHelp();
