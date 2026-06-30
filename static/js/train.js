const form = document.getElementById("train-form");
const submitBtn = document.getElementById("submit-btn");
const progressArea = document.getElementById("progress-area");
const progressPlaceholder = document.getElementById("progress-placeholder");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const jobIdEl = document.getElementById("job-id");
const epochLog = document.getElementById("epoch-log");
const dataPathEl = document.getElementById("data-path");

let pollTimer = null;

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
    progressPlaceholder.hidden = true;
    progressArea.hidden = false;
    jobIdEl.textContent = result.job_id;
    if (dataPathEl) dataPathEl.textContent = `DATA/job_${result.job_id}/`;
    resultLink.href = `/results/${result.job_id}`;
    pollJob(result.job_id);
  } catch (err) {
    alert("학습 시작 실패: " + err.message);
    submitBtn.disabled = false;
    submitBtn.textContent = "학습 시작";
  }
});

async function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  const update = async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();

    const pct = job.epochs > 0 ? (job.current_epoch / job.epochs) * 100 : 0;
    progressBar.style.width = `${pct}%`;
    progressText.textContent = `${job.status} — Epoch ${job.current_epoch}/${job.epochs}`;

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
