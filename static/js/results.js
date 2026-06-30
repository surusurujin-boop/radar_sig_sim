let lossChart = null;
let ablationChart = null;

async function loadResults() {
  const loading = document.getElementById("loading");
  const content = document.getElementById("results-content");

  try {
    const [jobRes, predRes] = await Promise.all([
      fetch(`/api/jobs/${JOB_ID}`),
      fetch(`/api/jobs/${JOB_ID}/predictions`),
    ]);

    const job = await jobRes.json();
    const predictions = await predRes.json();

    if (job.error) {
      loading.textContent = job.error;
      return;
    }

    document.getElementById("job-title").textContent = `#${job.id}`;
    document.getElementById("job-meta").textContent =
      `${job.name} | ${job.modality_set} | ${job.status} | Epochs ${job.epochs}`;

    renderEvalTable(job.evaluations || []);
    renderCharts(job.epoch_logs || [], job.evaluations || []);
    renderPredictions(predictions);

    loading.hidden = true;
    content.hidden = false;
  } catch (e) {
    loading.textContent = "불러오기 실패: " + e.message;
  }
}

function renderEvalTable(evals) {
  const body = document.getElementById("eval-body");
  if (!evals.length) {
    body.innerHTML = `<tr><td colspan="6">평가 결과 없음</td></tr>`;
    return;
  }
  body.innerHTML = evals
    .map(
      (e) => `
    <tr>
      <td>${e.modality_label}</td>
      <td>${e.ari.toFixed(4)}</td>
      <td>${e.nmi.toFixed(4)}</td>
      <td>${e.purity.toFixed(4)}</td>
      <td>${e.pairwise_f1.toFixed(4)}</td>
      <td>${e.cluster_count_error}</td>
    </tr>`
    )
    .join("");
}

function renderCharts(epochLogs, evals) {
  const epochs = epochLogs.map((e) => e.epoch);
  const losses = epochLogs.map((e) => e.loss);
  const aris = epochLogs.map((e) => e.ari || 0);

  const ctx1 = document.getElementById("chart-loss");
  if (lossChart) lossChart.destroy();
  lossChart = new Chart(ctx1, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Loss",
          data: losses,
          borderColor: "#ef4444",
          tension: 0.3,
          yAxisID: "y",
        },
        {
          label: "ARI",
          data: aris,
          borderColor: "#3b82f6",
          tension: 0.3,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { position: "left", title: { display: true, text: "Loss" } },
        y1: {
          position: "right",
          min: 0,
          max: 1,
          grid: { drawOnChartArea: false },
          title: { display: true, text: "ARI" },
        },
      },
    },
  });

  if (evals.length > 1) {
    const ctx2 = document.getElementById("chart-ablation");
    if (ablationChart) ablationChart.destroy();
    ablationChart = new Chart(ctx2, {
      type: "bar",
      data: {
        labels: evals.map((e) => e.modality_label),
        datasets: [
          {
            label: "ARI",
            data: evals.map((e) => e.ari),
            backgroundColor: ["#6366f1", "#3b82f6", "#22c55e"],
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { min: 0, max: 1 } },
        plugins: { legend: { display: false } },
      },
    });
  }
}

function renderPredictions(rows) {
  const body = document.getElementById("pred-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="9">예측 데이터 없음</td></tr>`;
    return;
  }
  body.innerHTML = rows
    .slice(0, 100)
    .map(
      (r) => `
    <tr>
      <td>${r.pulse_index}</td>
      <td>${r.true_label}</td>
      <td>${r.pred_label}</td>
      <td>${r.cf_norm.toFixed(4)}</td>
      <td>${r.pw_log.toFixed(4)}</td>
      <td>${r.pa.toFixed(4)}</td>
      <td>${r.doa_norm.toFixed(4)}</td>
      <td>${r.toa_norm.toFixed(4)}</td>
      <td class="${r.match ? "match-yes" : "match-no"}">${r.match ? "O" : "X"}</td>
    </tr>`
    )
    .join("");
}

loadResults();
