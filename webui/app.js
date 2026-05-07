const form = document.getElementById("form");
const fileInput = document.getElementById("file");
const attackSelect = document.getElementById("attack");
const pgdWrap = document.getElementById("pgd_wrap");
const alphaWrap = document.getElementById("alpha_wrap");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const errorEl = document.getElementById("error");
const results = document.getElementById("results");
const hint = document.getElementById("hint");
const submitBtn = document.getElementById("submit");

const presetLabel = document.getElementById("preset-label");
const galleryHoldout = document.getElementById("gallery-holdout");
const galleryImages = document.getElementById("gallery-images");
const galleryNote = document.getElementById("gallery-note");
const clearPresetBtn = document.getElementById("clear-preset");

let selectedPreset = null;

function togglePgdFields() {
  const isPgd = attackSelect.value === "pgd";
  pgdWrap.classList.toggle("hidden", !isPgd);
  alphaWrap.classList.toggle("hidden", !isPgd);
}

attackSelect.addEventListener("change", togglePgdFields);
togglePgdFields();

function setPreset(item) {
  selectedPreset = item;
  fileInput.value = "";
  presetLabel.textContent = item
    ? `Using sample: ${item.source}/${item.name}`
    : "";
  presetLabel.classList.toggle("hidden", !item);
  document.querySelectorAll(".gallery-grid button").forEach((b) => {
    b.classList.toggle("selected", item && b.dataset.name === item.name && b.dataset.source === item.source);
  });
}

clearPresetBtn.addEventListener("click", () => setPreset(null));

async function loadGallery() {
  try {
    const r = await fetch("/api/gallery");
    const data = await r.json();
    galleryNote.textContent = data.note || "";

    function fill(grid, items) {
      grid.innerHTML = "";
      items.forEach((it) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "gallery-thumb";
        btn.dataset.name = it.name;
        btn.dataset.source = it.source;
        const img = document.createElement("img");
        img.src = it.url;
        img.alt = it.name;
        img.loading = "lazy";
        btn.appendChild(img);
        btn.appendChild(document.createElement("span")).textContent = it.name;
        btn.addEventListener("click", () => setPreset({ name: it.name, source: it.source }));
        grid.appendChild(btn);
      });
    }

    fill(galleryHoldout, data.holdout || []);
    fill(galleryImages, data.images || []);
  } catch {
    galleryNote.textContent = "Could not load gallery API.";
  }
}

async function fetchHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    hint.textContent = j.clip_loaded
      ? "CLIP is loaded on the server."
      : "CLIP will load on first attack (may take a minute).";
  } catch {
    hint.textContent = "";
  }
}

fetchHealth();
loadGallery();

function showGlobalError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function hideGlobalError() {
  errorEl.classList.add("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideGlobalError();
  results.classList.add("hidden");

  if (!selectedPreset && (!fileInput.files || !fileInput.files[0])) {
    showGlobalError("Choose a sample image or upload a file.");
    return;
  }

  loadingText.textContent = "Loading CLIP / running attack…";
  loading.classList.remove("hidden");
  submitBtn.disabled = true;

  const fd = new FormData();
  if (selectedPreset) {
    fd.append("preset_name", selectedPreset.name);
    fd.append("preset_source", selectedPreset.source);
  } else {
    fd.append("image", fileInput.files[0]);
  }
  fd.append("attack", attackSelect.value);
  fd.append("target_text", document.getElementById("target_text").value.trim());

  const eps = document.getElementById("epsilon").value;
  if (eps !== "") fd.append("epsilon", eps);

  if (attackSelect.value === "pgd") {
    const steps = document.getElementById("pgd_steps").value;
    const alpha = document.getElementById("pgd_alpha").value;
    if (steps !== "") fd.append("pgd_steps", steps);
    if (alpha !== "") fd.append("pgd_alpha", alpha);
  }

  try {
    const res = await fetch("/api/attack", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      let msg = res.statusText || "Request failed";
      if (Array.isArray(data.detail)) {
        msg = data.detail.map((d) => (d.msg ? `${d.loc?.join?.(".")}: ${d.msg}` : JSON.stringify(d))).join("; ");
      } else if (typeof data.detail === "string") {
        msg = data.detail;
      }
      throw new Error(msg);
    }
    showClipResults(data);
  } catch (err) {
    showGlobalError(err.message || String(err));
  } finally {
    loading.classList.add("hidden");
    submitBtn.disabled = false;
  }
});

function showClipResults(data) {
  const m = document.getElementById("metrics");
  const succ = data.attack_succeeded;
  const preset = data.preset_used
    ? `<div class="metric"><strong>Sample</strong><span>${escapeHtml(data.preset_used.source + "/" + data.preset_used.name)}</span></div>`
    : "";
  m.innerHTML = `
    ${preset}
    <div class="metric"><strong>Attack</strong><span>${escapeHtml(data.attack)}</span></div>
    <div class="metric"><strong>Device</strong><span>${escapeHtml(data.device)}</span></div>
    <div class="metric"><strong>Similarity (original)</strong><span>${data.similarity_original.toFixed(4)}</span></div>
    <div class="metric"><strong>Similarity (adversarial)</strong><span>${data.similarity_adversarial.toFixed(4)}</span></div>
    <div class="metric"><strong>Confidence shift</strong><span>${data.confidence_shift.toFixed(4)}</span></div>
    <div class="metric"><strong>Attack succeeded</strong><span class="${succ ? "ok" : "bad"}">${succ ? "Yes" : "No"}</span></div>
    <div class="metric"><strong>Epsilon</strong><span>${data.epsilon}</span></div>
    ${data.pgd_steps != null ? `<div class="metric"><strong>PGD steps</strong><span>${data.pgd_steps}</span></div>` : ""}
    ${data.pgd_alpha != null ? `<div class="metric"><strong>PGD α</strong><span>${data.pgd_alpha}</span></div>` : ""}
  `;
  document.getElementById("img_orig").src = data.image_original;
  document.getElementById("img_adv").src = data.image_adversarial;
  results.classList.remove("hidden");
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

/* Tabs */
const tabs = document.querySelectorAll(".tabs .tab");
const panelClip = document.getElementById("panel-clip");
const panelPaper = document.getElementById("panel-paper");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const id = tab.dataset.tab;
    tabs.forEach((t) => {
      const on = t.dataset.tab === id;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    panelClip.classList.toggle("hidden", id !== "clip");
    panelPaper.classList.toggle("hidden", id !== "paper");
    if (id === "paper") loadPaperCheckpoints();
  });
});

/* Paper CNN */
const paperForm = document.getElementById("paper-form");
const paperCheckpoint = document.getElementById("paper_checkpoint");
const paperAttack = document.getElementById("paper_attack");
const paperSubmit = document.getElementById("paper-submit");
const paperHint = document.getElementById("paper-hint");
const paperResults = document.getElementById("paper-results");
const paperMetrics = document.getElementById("paper-metrics");
const paperCwWrap = document.getElementById("paper_cw_wrap");

function togglePaperAttackOpts() {
  const a = paperAttack.value;
  paperCwWrap.classList.toggle("hidden", a !== "cw");
  document.getElementById("paper_pgd_steps_wrap").classList.toggle("hidden", a !== "pgd");
  document.getElementById("paper_pgd_alpha_wrap").classList.toggle("hidden", a !== "pgd");
}

paperAttack.addEventListener("change", togglePaperAttackOpts);
togglePaperAttackOpts();

async function loadPaperCheckpoints() {
  paperHint.textContent = "Loading checkpoints…";
  try {
    const r = await fetch("/api/paper/checkpoints");
    const data = await r.json();
    paperCheckpoint.innerHTML = "";
    const opts = data.checkpoints || [];
    if (opts.length === 0) {
      paperHint.textContent =
        data.note ||
        "No checkpoints in paper_checkpoints/. Run train or fast-benchmark.";
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "(none)";
      paperCheckpoint.appendChild(o);
      return;
    }
    opts.forEach((c) => {
      const o = document.createElement("option");
      o.value = c.filename;
      o.textContent = `${c.filename} (${c.size_mb} MB)`;
      paperCheckpoint.appendChild(o);
    });
    const sug = data.suggested || [];
    const preferred = sug.find((x) => opts.some((o) => o.filename === x)) || opts[0].filename;
    paperCheckpoint.value = preferred;
    paperHint.textContent = `Directory: ${data.checkpoint_dir}`;
    if (data.note) paperHint.textContent += " — " + data.note;
  } catch (e) {
    paperHint.textContent = e.message || String(e);
  }
}

function numOrNull(id) {
  const el = document.getElementById(id);
  const v = el.value.trim();
  return v === "" ? null : Number(v);
}

paperForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideGlobalError();
  paperResults.classList.add("hidden");

  const checkpoint = document.getElementById("paper_checkpoint").value;
  if (!checkpoint) {
    showGlobalError("Select a checkpoint or generate one first.");
    return;
  }

  loadingText.textContent = "Running CNN robustness evaluation…";
  loading.classList.remove("hidden");
  paperSubmit.disabled = true;

  const body = {
    dataset: document.getElementById("paper_dataset").value,
    checkpoint,
    attack: document.getElementById("paper_attack").value,
    limit_batches: numOrNull("paper_limit"),
    epsilon: numOrNull("paper_epsilon"),
    pgd_steps: numOrNull("paper_pgd_steps"),
    pgd_alpha: numOrNull("paper_pgd_alpha"),
    cw_steps: numOrNull("paper_cw_steps"),
    cw_lr: numOrNull("paper_cw_lr"),
    cw_c: numOrNull("paper_cw_c"),
  };

  try {
    const res = await fetch("/api/paper/eval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      let msg = res.statusText || "Request failed";
      if (Array.isArray(data.detail)) {
        msg = data.detail.map((d) => (d.msg ? d.msg : JSON.stringify(d))).join("; ");
      } else if (typeof data.detail === "string") {
        msg = data.detail;
      }
      throw new Error(msg);
    }

    paperMetrics.innerHTML = `
      <div class="metric"><strong>Dataset</strong><span>${escapeHtml(data.dataset)}</span></div>
      <div class="metric"><strong>Checkpoint</strong><span>${escapeHtml(data.checkpoint)}</span></div>
      <div class="metric"><strong>Attack</strong><span>${escapeHtml(data.attack)}</span></div>
      <div class="metric"><strong>Device</strong><span>${escapeHtml(data.device)}</span></div>
      <div class="metric"><strong>Clean test accuracy</strong><span>${data.clean_accuracy}</span></div>
      <div class="metric"><strong>Evaluated samples</strong><span>${data.evaluated_samples}</span></div>
      <div class="metric"><strong>ASR (%)</strong><span>${data.asr_percent}</span></div>
      <div class="metric"><strong>Robustness R = 1 − ASR</strong><span>${data.robustness_score}</span></div>
      <div class="metric"><strong>ε used</strong><span>${data.epsilon_used}</span></div>
      <div class="metric"><strong>Batches cap</strong><span>${data.limit_batches ?? "none (full loader)"}</span></div>
      ${data.pgd_steps_used != null ? `<div class="metric"><strong>PGD steps</strong><span>${data.pgd_steps_used}</span></div>` : ""}
      ${data.pgd_alpha_used != null ? `<div class="metric"><strong>PGD α</strong><span>${data.pgd_alpha_used}</span></div>` : ""}
      ${data.cw_steps_used != null ? `<div class="metric"><strong>CW steps</strong><span>${data.cw_steps_used}</span></div>` : ""}
    `;
    paperResults.classList.remove("hidden");
  } catch (err) {
    showGlobalError(err.message || String(err));
  } finally {
    loading.classList.add("hidden");
    paperSubmit.disabled = false;
  }
});
