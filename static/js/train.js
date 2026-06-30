const form = document.getElementById("train-form");
const submitBtn = document.getElementById("submit-btn");
const progressArea = document.getElementById("progress-area");
const progressPlaceholder = document.getElementById("progress-placeholder");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const jobIdEl = document.getElementById("job-id");
const epochLog = document.getElementById("epoch-log");
const dataPathEl = document.getElementById("data-path");
const resultLink = document.getElementById("result-link");
const phasePipeline = document.getElementById("phase-pipeline");
const phaseDetail = document.getElementById("phase-detail");

let pollTimer = null;
let trainingPhases = [];
let trainingEnabled = true;

async function loadRuntime() {
  try {
    const res = await fetch("/api/runtime");
    const info = await res.json();
    trainingEnabled = info.training_enabled;
    if (!trainingEnabled) {
      submitBtn.disabled = true;
      submitBtn.textContent = "학습 불가 (Demo 모드)";
      const notice = document.getElementById("train-disabled-notice");
      if (notice) notice.hidden = false;
    }
  } catch (_) {
    /* ignore */
  }
}

async function loadPhases() {
  const res = await fetch("/api/training-phases");
  trainingPhases = await res.json();
}

function renderPhasePipeline(job) {
  if (!phasePipeline || !trainingPhases.length) return;

  const current = job.current_phase || "queued";
  const currentIdx = trainingPhases.findIndex((p) => p.id === current);

  phasePipeline.innerHTML = trainingPhases
    .map((p, i) => {
      let state = "pending";
      if (job.status === "completed" || current === "done") state = "done";
      else if (p.id === current) state = "active";
      else if (currentIdx >= 0 && i < currentIdx) state = "done";
      return `<div class="phase-step ${state}" data-phase="${p.id}">
        <div class="phase-dot">${i + 1}</div>
        <div class="phase-label">${p.label}</div>
      </div>`;
    })
    .join("");

  if (phaseDetail) {
    phaseDetail.textContent = job.phase_message || "";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "시작 중...";

  const data = {
    name: form.name.value,
    modality_set: form.modality_set.value,
    fusion_mode: form.fusion_mode.value,
    embed_dim: parseInt(form.embed_dim.value, 10),
    aux_lambda: parseFloat(form.aux_lambda.value),
    epochs: parseInt(form.epochs.value, 10),
    batch_size: parseInt(form.batch_size.value, 10),
    lr: parseFloat(form.lr.value),
    train_samples: parseInt(form.train_samples.value, 10),
    test_samples: parseInt(form.test_samples.value, 10),
    num_emitters: parseInt(form.num_emitters.value, 10),
  };

  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await res.json();
    if (!res.ok) {
      throw new Error(result.message || result.error || "학습 시작 실패");
    }
    progressPlaceholder.hidden = true;
    progressArea.hidden = false;
    jobIdEl.textContent = result.job_id;
    if (dataPathEl) dataPathEl.textContent = `DATA/job_${result.job_id}/`;
    resultLink.href = `/results/${result.job_id}`;
    resultLink.hidden = true;
    pollJob(result.job_id);
  } catch (err) {
    alert("학습 시작 실패: " + err.message);
    submitBtn.disabled = !trainingEnabled;
    submitBtn.textContent = trainingEnabled ? "학습 시작" : "학습 불가 (Demo 모드)";
  }
});

async function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  const update = async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();

    if (job.training_phases) trainingPhases = job.training_phases;
    renderPhasePipeline(job);

    const pct = job.epochs > 0 ? (job.current_epoch / job.epochs) * 100 : 0;
    progressBar.style.width = `${pct}%`;

    const phaseLabel =
      trainingPhases.find((p) => p.id === job.current_phase)?.label || job.current_phase;
    progressText.textContent = `${job.status} — ${phaseLabel} (Epoch ${job.current_epoch}/${job.epochs})`;

    if (job.epoch_logs && job.epoch_logs.length) {
      epochLog.innerHTML = job.epoch_logs
        .slice(-8)
        .map(
          (e) =>
            `<div>Epoch ${e.epoch}: loss=${e.loss.toFixed(4)}, ARI=${(e.ari || 0).toFixed(4)}</div>`
        )
        .join("");
    }

    if (job.status === "completed") {
      progressText.textContent = "학습 완료!";
      renderPhasePipeline({ ...job, current_phase: "done", status: "completed" });
      resultLink.hidden = false;
      submitBtn.disabled = false;
      submitBtn.textContent = "학습 시작";
      clearInterval(pollTimer);
    } else if (job.status === "failed") {
      progressText.textContent = "실패: " + (job.error_message || "unknown");
      submitBtn.disabled = false;
      submitBtn.textContent = "학습 시작";
      clearInterval(pollTimer);
    }
  };

  await update();
  pollTimer = setInterval(update, 2000);
}

loadPhases();
loadRuntime();
