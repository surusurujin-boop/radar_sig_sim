"""레이다 펄스 클러스터링 웹 애플리케이션."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from init_db import seed_scenarios
from src.db.models import (
    EvaluationResult,
    EpochLog,
    PulsePrediction,
    ScenarioMeta,
    SessionLocal,
    TrainingJob,
    init_db,
    job_to_dict,
)
from src.services.training_service import create_and_start_job

app = Flask(__name__)


@app.before_request
def ensure_db():
    if not hasattr(app, "_db_ready"):
        init_db()
        seed_scenarios()
        app._db_ready = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/design")
def design_page():
    return render_template("design.html")


@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/results/<int:job_id>")
def results_page(job_id: int):
    return render_template("results.html", job_id=job_id)


@app.route("/api/scenarios")
def api_scenarios():
    session = SessionLocal()
    try:
        rows = session.query(ScenarioMeta).all()
        return jsonify(
            [
                {
                    "scenario_id": r.scenario_id,
                    "name": r.name,
                    "description": r.description,
                }
                for r in rows
            ]
        )
    finally:
        session.close()


@app.route("/api/jobs", methods=["GET"])
def api_list_jobs():
    session = SessionLocal()
    try:
        jobs = session.query(TrainingJob).order_by(TrainingJob.id.desc()).all()
        result = []
        for job in jobs:
            d = job_to_dict(job)
            evals = session.query(EvaluationResult).filter_by(job_id=job.id).all()
            d["evaluations"] = [
                {
                    "modality_label": e.modality_label,
                    "ari": e.ari,
                    "nmi": e.nmi,
                    "purity": e.purity,
                    "pairwise_f1": e.pairwise_f1,
                }
                for e in evals
            ]
            result.append(d)
        return jsonify(result)
    finally:
        session.close()


@app.route("/api/jobs", methods=["POST"])
def api_create_job():
    data = request.get_json() or {}
    job_id = create_and_start_job(
        name=data.get("name", "Training Run"),
        modality_set=data.get("modality_set", "full"),
        epochs=int(data.get("epochs", 10)),
        batch_size=int(data.get("batch_size", 8)),
        lr=float(data.get("lr", 1e-4)),
        train_samples=int(data.get("train_samples", 60)),
        test_samples=int(data.get("test_samples", 20)),
        num_emitters=int(data.get("num_emitters", 3)),
        embed_dim=int(data.get("embed_dim", 256)),
        fusion_mode=data.get("fusion_mode", "pulse"),
        aux_lambda=float(data.get("aux_lambda", 0.2)),
    )
    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/api/jobs/<int:job_id>")
def api_job_detail(job_id: int):
    session = SessionLocal()
    try:
        job = session.get(TrainingJob, job_id)
        if not job:
            return jsonify({"error": "Not found"}), 404

        d = job_to_dict(job)
        d["epoch_logs"] = [
            {"epoch": e.epoch, "loss": e.loss, "ari": e.ari, "nmi": e.nmi}
            for e in session.query(EpochLog)
            .filter_by(job_id=job_id)
            .order_by(EpochLog.epoch)
            .all()
        ]
        d["evaluations"] = [
            {
                "modality_label": e.modality_label,
                "ari": e.ari,
                "nmi": e.nmi,
                "purity": e.purity,
                "v_measure": e.v_measure,
                "pairwise_f1": e.pairwise_f1,
                "n_clusters_pred": e.n_clusters_pred,
                "n_clusters_true": e.n_clusters_true,
                "cluster_count_error": e.cluster_count_error,
                "noise_ratio": e.noise_ratio,
            }
            for e in session.query(EvaluationResult).filter_by(job_id=job_id).all()
        ]
        return jsonify(d)
    finally:
        session.close()


@app.route("/api/jobs/<int:job_id>/predictions")
def api_predictions(job_id: int):
    session = SessionLocal()
    try:
        rows = (
            session.query(PulsePrediction)
            .filter_by(job_id=job_id)
            .order_by(PulsePrediction.pulse_index)
            .limit(200)
            .all()
        )
        return jsonify(
            [
                {
                    "pulse_index": r.pulse_index,
                    "true_label": r.true_label,
                    "pred_label": r.pred_label,
                    "cf_norm": r.cf_norm,
                    "pw_log": r.pw_log,
                    "pa": r.pa,
                    "doa_norm": r.doa_norm,
                    "toa_norm": r.toa_norm,
                    "match": r.true_label == r.pred_label,
                }
                for r in rows
            ]
        )
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    seed_scenarios()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
