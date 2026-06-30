"""모델 학습·평가 서비스 (웹 job용)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.clustering import HDBSCANClusterer
from src.data.export import DATA_ROOT, save_train_test_pair
from src.data.scenarios import SimulationScenario
from src.data.synthetic import ScenarioDataset, collate_pulse_batch
from src.db.models import EpochLog, EvaluationResult, PulsePrediction, SessionLocal, TrainingJob
from src.evaluation.metrics import compute_clustering_metrics
from src.losses.combined import CombinedClusteringLoss
from src.models.multimodal_transformer import FusionMode, ModalitySet, MultimodalPulseClusteringModel
from src.services.training_phases import TRAINING_PHASES

MODALITY_LABELS: dict[ModalitySet, str] = {
    "pdw": "PDW only",
    "pdw_iq": "PDW + IQ",
    "full": "PDW + IQ + Spectrum",
}

_job_lock = threading.Lock()
_running_jobs: set[int] = set()


def _set_phase(job_id: int, phase_id: str, message: str | None = None) -> None:
    desc = message
    if desc is None:
        for p in TRAINING_PHASES:
            if p["id"] == phase_id:
                desc = p["desc"]
                break
    sess = SessionLocal()
    try:
        job = sess.get(TrainingJob, job_id)
        if job:
            job.current_phase = phase_id
            job.phase_message = desc
            sess.commit()
    finally:
        sess.close()


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_contrastive_loss(embeddings, labels, mask, criterion):
    valid = mask.view(-1)
    emb = embeddings.reshape(-1, embeddings.shape[-1])[valid]
    lab = labels.reshape(-1)[valid]
    return criterion(emb, lab)


@torch.no_grad()
def _evaluate_model(model, test_scenario, device):
    model.eval()
    dataset = ScenarioDataset(test_scenario)
    clusterer = HDBSCANClusterer(min_cluster_size=5)

    all_true, all_pred = [], []
    sample = dataset.samples[0]
    pdw = sample["pdw"]
    true_labels = sample["labels"]

    for seq in dataset.samples:
        pdw_t = torch.from_numpy(seq["pdw"]).unsqueeze(0).float().to(device)
        iq_t = torch.from_numpy(seq["iq"]).unsqueeze(0).float().to(device)
        spec_t = torch.from_numpy(seq["spectrum"]).unsqueeze(0).float().to(device)
        inst_t = torch.from_numpy(seq["iq_inst"]).unsqueeze(0).float().to(device)
        tf_t = torch.from_numpy(seq["iq_tf"]).unsqueeze(0).float().to(device)
        emb = model.get_embedding(pdw_t, iq_t, spec_t, inst_t, tf_t)[0].cpu().numpy()
        pred = clusterer.fit_predict(emb).labels
        all_true.append(seq["labels"])
        all_pred.append(pred)

    metrics = compute_clustering_metrics(
        np.concatenate(all_true), np.concatenate(all_pred), exclude_noise=True
    )

    pdw_t = torch.from_numpy(sample["pdw"]).unsqueeze(0).float().to(device)
    iq_t = torch.from_numpy(sample["iq"]).unsqueeze(0).float().to(device)
    spec_t = torch.from_numpy(sample["spectrum"]).unsqueeze(0).float().to(device)
    inst_t = torch.from_numpy(sample["iq_inst"]).unsqueeze(0).float().to(device)
    tf_t = torch.from_numpy(sample["iq_tf"]).unsqueeze(0).float().to(device)
    emb = model.get_embedding(pdw_t, iq_t, spec_t, inst_t, tf_t)[0].cpu().numpy()
    sample_pred = clusterer.fit_predict(emb).labels

    return metrics, true_labels, sample_pred, pdw


def _train_single_modality(job_id, modality_set, job, device, session_factory, train_scenario, test_scenario):
    dataset = ScenarioDataset(train_scenario)
    loader = DataLoader(
        dataset,
        batch_size=job.batch_size,
        shuffle=True,
        collate_fn=collate_pulse_batch,
        drop_last=True,
    )

    fusion_mode: FusionMode = getattr(job, "fusion_mode", "pulse") or "pulse"
    aux_lambda = float(getattr(job, "aux_lambda", 0.2) or 0.2)

    model = MultimodalPulseClusteringModel(
        embed_dim=job.embed_dim,
        modality_set=modality_set,
        fusion_mode=fusion_mode,
    ).to(device)

    _set_phase(job_id, "model_init", f"{MODALITY_LABELS[modality_set]} Multimodal Transformer 가중치 초기화")

    criterion = CombinedClusteringLoss(temperature=0.1, aux_lambda=aux_lambda)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job.lr, weight_decay=1e-4)

    for epoch in range(1, job.epochs + 1):
        _set_phase(
            job_id,
            "train_epoch",
            f"Epoch {epoch}/{job.epochs} — {MODALITY_LABELS[modality_set]} Contrastive + Aux 학습",
        )
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            pdw = batch["pdw"].to(device)
            iq = batch["iq"].to(device)
            iq_inst = batch["iq_inst"].to(device)
            iq_tf = batch["iq_tf"].to(device)
            spectrum = batch["spectrum"].to(device)
            labels = batch["labels"].to(device)
            mod_labels = batch["mod_labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            if modality_set in ("pdw_iq", "full"):
                embeddings, mod_logits = model(
                    pdw, iq, spectrum, iq_inst, iq_tf, return_aux=True
                )
            else:
                embeddings = model(pdw, iq, spectrum, iq_inst, iq_tf)
                mod_logits = None

            loss, _ = criterion(embeddings, labels, mod_logits, mod_labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        _set_phase(job_id, "eval", f"Epoch {epoch} — HDBSCAN 클러스터링 및 ARI/NMI 계산")
        metrics, _, _, _ = _evaluate_model(model, test_scenario, device)

        sess = session_factory()
        try:
            sess.add(
                EpochLog(
                    job_id=job_id,
                    epoch=epoch,
                    loss=avg_loss,
                    ari=metrics.ari,
                    nmi=metrics.nmi,
                )
            )
            db_job = sess.get(TrainingJob, job_id)
            if db_job:
                db_job.current_epoch = epoch
            sess.commit()
        finally:
            sess.close()

    _set_phase(job_id, "checkpoint", f"{MODALITY_LABELS[modality_set]} 체크포인트 저장")
    ckpt_dir = Path("checkpoints/web") / str(job_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{modality_set}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "embed_dim": job.embed_dim,
                "modality_set": modality_set,
                "fusion_mode": fusion_mode,
            },
        },
        ckpt_path,
    )

    metrics, true_labels, pred_labels, pdw = _evaluate_model(model, test_scenario, device)
    return {
        "modality_set": modality_set,
        "label": MODALITY_LABELS[modality_set],
        "metrics": metrics,
        "checkpoint_path": str(ckpt_path),
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "pdw": pdw,
    }


def _run_job(job_id: int) -> None:
    session = SessionLocal()
    try:
        job = session.get(TrainingJob, job_id)
        if not job:
            return

        job.status = "running"
        job.current_phase = "queued"
        job.phase_message = "학습 파이프라인 시작"
        session.commit()
        device = _get_device()

        _set_phase(job_id, "data_gen", f"합성 데이터 생성 (train {job.train_samples}, test {job.test_samples})")
        train_scenario = SimulationScenario(
            scenario_id="train",
            name="Train",
            description="Training split",
            num_emitters=job.num_emitters,
            num_samples=job.train_samples,
            seed=42,
        )
        test_scenario = SimulationScenario(
            scenario_id="test",
            name="Test",
            description="Test split",
            num_emitters=job.num_emitters,
            num_samples=job.test_samples,
            seed=999,
        )
        _set_phase(job_id, "data_save", f"DATA/job_{job_id}/ 에 npz·manifest 저장")
        save_train_test_pair(
            train_scenario,
            test_scenario,
            DATA_ROOT / f"job_{job_id}",
            extra_meta={
                "job_id": job_id,
                "job_name": job.name,
                "modality_set": job.modality_set,
            },
        )

        if job.modality_set == "ablation":
            modalities: list[ModalitySet] = ["pdw", "pdw_iq", "full"]
        else:
            modalities = [job.modality_set]  # type: ignore

        primary_ckpt = None
        for mod in modalities:
            result = _train_single_modality(
                job_id, mod, job, device, SessionLocal, train_scenario, test_scenario
            )

            sess = SessionLocal()
            try:
                m = result["metrics"]
                sess.add(
                    EvaluationResult(
                        job_id=job_id,
                        modality_label=result["label"],
                        ari=m.ari,
                        nmi=m.nmi,
                        purity=m.purity,
                        v_measure=m.v_measure,
                        pairwise_f1=m.pairwise_f1,
                        n_clusters_pred=m.n_clusters_pred,
                        n_clusters_true=m.n_clusters_true,
                        cluster_count_error=m.cluster_count_error,
                        noise_ratio=m.noise_ratio,
                    )
                )
                if mod == modalities[-1]:
                    for i, (t, p) in enumerate(
                        zip(result["true_labels"], result["pred_labels"], strict=False)
                    ):
                        sess.add(
                            PulsePrediction(
                                job_id=job_id,
                                sequence_index=0,
                                pulse_index=i,
                                true_label=int(t),
                                pred_label=int(p),
                                cf_norm=float(result["pdw"][i, 0]),
                                pw_log=float(result["pdw"][i, 1]),
                                pa=float(result["pdw"][i, 2]),
                                doa_norm=float(result["pdw"][i, 3]),
                                toa_norm=float(result["pdw"][i, 4]),
                            )
                        )
                if primary_ckpt is None or mod == "full":
                    primary_ckpt = result["checkpoint_path"]
                sess.commit()
            finally:
                sess.close()

        sess = SessionLocal()
        try:
            db_job = sess.get(TrainingJob, job_id)
            if db_job:
                db_job.status = "completed"
                db_job.current_phase = "done"
                db_job.phase_message = "학습·평가·저장 완료"
                db_job.checkpoint_path = primary_ckpt
                db_job.completed_at = datetime.now(timezone.utc)
                db_job.current_epoch = db_job.epochs
                sess.commit()
        finally:
            sess.close()

    except Exception as e:
        sess = SessionLocal()
        try:
            db_job = sess.get(TrainingJob, job_id)
            if db_job:
                db_job.status = "failed"
                db_job.current_phase = "failed"
                db_job.phase_message = str(e)
                db_job.error_message = str(e)
                db_job.completed_at = datetime.now(timezone.utc)
                sess.commit()
        finally:
            sess.close()
    finally:
        session.close()
        with _job_lock:
            _running_jobs.discard(job_id)


def start_training_job(job_id: int) -> None:
    with _job_lock:
        if job_id in _running_jobs:
            return
        _running_jobs.add(job_id)
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()


def create_and_start_job(
    name: str,
    modality_set: str,
    epochs: int,
    batch_size: int,
    lr: float,
    train_samples: int,
    test_samples: int,
    num_emitters: int,
    embed_dim: int = 256,
    fusion_mode: str = "pulse",
    aux_lambda: float = 0.2,
) -> int:
    session = SessionLocal()
    try:
        job = TrainingJob(
            name=name,
            modality_set=modality_set,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            train_samples=train_samples,
            test_samples=test_samples,
            num_emitters=num_emitters,
            embed_dim=embed_dim,
            fusion_mode=fusion_mode,
            aux_lambda=aux_lambda,
            status="pending",
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    finally:
        session.close()

    start_training_job(job_id)
    return job_id
