const datasetSelect = document.getElementById("dataset-select");
const splitSelect = document.getElementById("split-select");
const sampleSelect = document.getElementById("sample-select");
const pulseTableBody = document.getElementById("pulse-table-body");
const pulseMeta = document.getElementById("pulse-meta");
const pdwTableBody = document.querySelector("#pdw-detail-table tbody");
const liveControls = document.getElementById("live-controls");
const liveEmittersWrap = document.getElementById("live-emitters-wrap");
const liveSeed = document.getElementById("live-seed");
const liveEmitters = document.getElementById("live-emitters");
const refreshBtn = document.getElementById("refresh-btn");

let selectedPulse = 0;
let iqChart = null;
let phaseChart = null;
let ifChart = null;

async function loadDatasets() {
  const res = await fetch("/api/mock-data/datasets");
  const datasets = await res.json();
  datasetSelect.innerHTML = datasets
    .map((d) => {
      const label =
        d.id === "live"
          ? "실시간 생성 (Live)"
          : `${d.id} (train:${d.train_samples}, test:${d.test_samples})`;
      return `<option value="${d.id}">${label}</option>`;
    })
    .join("");
}

async function loadSamples() {
  const datasetId = datasetSelect.value;
  const split = splitSelect.value;
  const isLive = datasetId === "live";

  splitSelect.disabled = isLive;
  liveControls.hidden = !isLive;
  liveEmittersWrap.hidden = !isLive;

  const res = await fetch(
    `/api/mock-data/samples?dataset_id=${encodeURIComponent(datasetId)}&split=${split}`
  );
  const samples = await res.json();
  sampleSelect.innerHTML = samples
    .map((s) => `<option value="${s.index}">샘플 ${s.index} (${s.num_pulses} pulses)</option>`)
    .join("");

  await loadSequence();
}

async function loadSequence() {
  const datasetId = datasetSelect.value;
  const split = splitSelect.value;
  const sampleIndex = parseInt(sampleSelect.value, 10);

  let url = `/api/mock-data/sequence?dataset_id=${encodeURIComponent(datasetId)}&split=${split}&sample_index=${sampleIndex}`;
  if (datasetId === "live") {
    url += `&seed=${liveSeed.value}&emitters=${liveEmitters.value}`;
  }

  const res = await fetch(url);
  if (!res.ok) {
    pulseTableBody.innerHTML = `<tr><td colspan="6">데이터 없음</td></tr>`;
    return;
  }
  const data = await res.json();

  pulseTableBody.innerHTML = data.pulses
    .map(
      (p) => `
    <tr class="pulse-row${p.pulse_index === selectedPulse ? " selected" : ""}" data-pulse="${p.pulse_index}">
      <td>${p.pulse_index}</td>
      <td>${p.emitter}</td>
      <td>${p.modulation}</td>
      <td>${p.cf.toFixed(4)}</td>
      <td>${p.pw.toFixed(4)}</td>
      <td>${p.pa.toFixed(4)}</td>
    </tr>`
    )
    .join("");

  document.querySelectorAll(".pulse-row").forEach((row) => {
    row.addEventListener("click", () => {
      selectedPulse = parseInt(row.dataset.pulse, 10);
      loadPulseDetail();
      document.querySelectorAll(".pulse-row").forEach((r) => r.classList.remove("selected"));
      row.classList.add("selected");
    });
  });

  if (selectedPulse >= data.num_pulses) selectedPulse = 0;
  await loadPulseDetail();
}

async function loadPulseDetail() {
  const datasetId = datasetSelect.value;
  const split = splitSelect.value;
  const sampleIndex = parseInt(sampleSelect.value, 10);

  let url = `/api/mock-data/pulse?dataset_id=${encodeURIComponent(datasetId)}&split=${split}&sample_index=${sampleIndex}&pulse_index=${selectedPulse}`;
  if (datasetId === "live") {
    url += `&seed=${liveSeed.value}&emitters=${liveEmitters.value}`;
  }

  const res = await fetch(url);
  if (!res.ok) return;
  const data = await res.json();

  pulseMeta.textContent = `펄스 #${data.pulse_index} · 방사원 ${data.emitter_label} · ${data.modulation_type}`;

  pdwTableBody.innerHTML = data.pdw.columns
    .map(
      (col, i) =>
        `<tr><td>${col}</td><td>${Number(data.pdw.values[i]).toFixed(6)}</td></tr>`
    )
    .join("");

  renderIqChart(data.iq);
  renderSpectrum(data.spectrum.values);
}

function renderIqChart(iq) {
  const labels = iq.i.map((_, i) => i);
  const ctx = document.getElementById("chart-iq");
  if (iqChart) iqChart.destroy();
  iqChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "I", data: iq.i, borderColor: "#3b82f6", borderWidth: 1, pointRadius: 0 },
        { label: "Q", data: iq.q, borderColor: "#f59e0b", borderWidth: 1, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#e7ecf3" } } },
      scales: {
        x: { ticks: { color: "#8b9cb3", maxTicksLimit: 8 }, grid: { color: "#2d3a4f" } },
        y: { ticks: { color: "#8b9cb3" }, grid: { color: "#2d3a4f" } },
      },
    },
  });

  const instWrap = document.getElementById("inst-charts");
  if (iq.inst) {
    instWrap.hidden = false;
    const instLabels = iq.inst.phase.map((_, i) => i);

    if (phaseChart) phaseChart.destroy();
    phaseChart = new Chart(document.getElementById("chart-phase"), {
      type: "line",
      data: {
        labels: instLabels,
        datasets: [{ label: "Phase φ(t)", data: iq.inst.phase, borderColor: "#a78bfa", borderWidth: 1, pointRadius: 0 }],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e7ecf3" } } },
        scales: {
          x: { display: false },
          y: { ticks: { color: "#8b9cb3" }, grid: { color: "#2d3a4f" } },
        },
      },
    });

    if (ifChart) ifChart.destroy();
    ifChart = new Chart(document.getElementById("chart-if"), {
      type: "line",
      data: {
        labels: instLabels,
        datasets: [{ label: "Inst. Freq", data: iq.inst.inst_freq, borderColor: "#22c55e", borderWidth: 1, pointRadius: 0 }],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e7ecf3" } } },
        scales: {
          x: { display: false },
          y: { ticks: { color: "#8b9cb3" }, grid: { color: "#2d3a4f" } },
        },
      },
    });
  } else {
    instWrap.hidden = true;
  }
}

function renderSpectrum(matrix) {
  const canvas = document.getElementById("chart-spectrum");
  const h = matrix.length;
  const w = matrix[0].length;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  canvas.style.maxHeight = "320px";

  let min = Infinity;
  let max = -Infinity;
  for (const row of matrix) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;

  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const t = (matrix[y][x] - min) / range;
      const idx = (y * w + x) * 4;
      img.data[idx] = Math.floor(t * 80);
      img.data[idx + 1] = Math.floor(80 + t * 120);
      img.data[idx + 2] = Math.floor(180 + t * 75);
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.hidden = true;
      p.classList.remove("active");
    });
    btn.classList.add("active");
    const panel = document.getElementById(`tab-${btn.dataset.tab}`);
    panel.hidden = false;
    panel.classList.add("active");
  });
});

datasetSelect.addEventListener("change", () => {
  selectedPulse = 0;
  loadSamples();
});
splitSelect.addEventListener("change", () => {
  selectedPulse = 0;
  loadSamples();
});
sampleSelect.addEventListener("change", () => {
  selectedPulse = 0;
  loadSequence();
});
liveSeed.addEventListener("change", () => loadSequence());
liveEmitters.addEventListener("change", () => loadSequence());
refreshBtn.addEventListener("click", () => loadSequence());

loadDatasets().then(loadSamples);
