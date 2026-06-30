const STATUS_LABEL = {
  pending: "대기",
  running: "학습 중",
  completed: "완료",
  failed: "실패",
};

function formatDate(iso) {
  if (!iso) return "-";
  return new Date(iso).toLocaleString("ko-KR");
}

function bestAri(evaluations) {
  if (!evaluations || !evaluations.length) return "-";
  return Math.max(...evaluations.map((e) => e.ari)).toFixed(4);
}

async function loadJobs() {
  const loading = document.getElementById("jobs-loading");
  const table = document.getElementById("jobs-table");
  const empty = document.getElementById("jobs-empty");
  const body = document.getElementById("jobs-body");

  try {
    const res = await fetch("/api/jobs");
    const jobs = await res.json();
    loading.hidden = true;

    if (!jobs.length) {
      empty.hidden = false;
      return;
    }

    table.hidden = false;
    body.innerHTML = jobs
      .map(
        (j) => `
      <tr>
        <td>${j.id}</td>
        <td>${j.name}</td>
        <td>${j.modality_set}</td>
        <td>${j.fusion_mode || "pulse"}</td>
        <td><span class="status status-${j.status}">${STATUS_LABEL[j.status] || j.status}</span></td>
        <td>${j.current_epoch}/${j.epochs}</td>
        <td>${bestAri(j.evaluations)}</td>
        <td><a href="/results/${j.id}" class="btn btn-secondary">결과</a></td>
      </tr>`
      )
      .join("");
  } catch (e) {
    loading.textContent = "불러오기 실패: " + e.message;
  }
}

async function loadScenarios() {
  const grid = document.getElementById("scenarios-grid");
  try {
    const res = await fetch("/api/scenarios");
    const scenarios = await res.json();
    grid.innerHTML = scenarios
      .slice(0, 12)
      .map(
        (s) => `
      <div class="scenario-item">
        <strong>${s.scenario_id} — ${s.name}</strong>
        <span>${s.description}</span>
      </div>`
      )
      .join("");
  } catch (e) {
    grid.innerHTML = `<p class="empty">${e.message}</p>`;
  }
}

loadJobs();
loadScenarios();
setInterval(loadJobs, 5000);
