"""레이다 펄스 클러스터링 웹 앱 DB 모델."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from src.runtime import get_database_url, get_db_path, is_vercel

DATABASE_URL = get_database_url()


class Base(DeclarativeBase):
    pass


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    modality_set = Column(String(32), nullable=False)  # pdw | pdw_iq | full | ablation
    status = Column(String(32), default="pending")  # pending | running | completed | failed
    epochs = Column(Integer, default=15)
    batch_size = Column(Integer, default=8)
    lr = Column(Float, default=1e-4)
    train_samples = Column(Integer, default=80)
    test_samples = Column(Integer, default=30)
    num_emitters = Column(Integer, default=3)
    embed_dim = Column(Integer, default=256)
    fusion_mode = Column(String(16), default="pulse")
    aux_lambda = Column(Float, default=0.2)
    current_epoch = Column(Integer, default=0)
    current_phase = Column(String(32), default="queued")
    phase_message = Column(Text, nullable=True)
    checkpoint_path = Column(String(512), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    epoch_logs = relationship("EpochLog", back_populates="job", cascade="all, delete-orphan")
    evaluations = relationship("EvaluationResult", back_populates="job", cascade="all, delete-orphan")
    predictions = relationship("PulsePrediction", back_populates="job", cascade="all, delete-orphan")


class EpochLog(Base):
    __tablename__ = "epoch_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    ari = Column(Float, nullable=True)
    nmi = Column(Float, nullable=True)

    job = relationship("TrainingJob", back_populates="epoch_logs")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    modality_label = Column(String(64), nullable=False)
    ari = Column(Float, nullable=False)
    nmi = Column(Float, nullable=False)
    purity = Column(Float, nullable=False)
    v_measure = Column(Float, nullable=False)
    pairwise_f1 = Column(Float, nullable=False)
    n_clusters_pred = Column(Integer, nullable=False)
    n_clusters_true = Column(Integer, nullable=False)
    cluster_count_error = Column(Integer, nullable=False)
    noise_ratio = Column(Float, nullable=False)

    job = relationship("TrainingJob", back_populates="evaluations")


class PulsePrediction(Base):
    __tablename__ = "pulse_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)
    sequence_index = Column(Integer, default=0)
    pulse_index = Column(Integer, nullable=False)
    true_label = Column(Integer, nullable=False)
    pred_label = Column(Integer, nullable=False)
    cf_norm = Column(Float, nullable=False)
    pw_log = Column(Float, nullable=False)
    pa = Column(Float, nullable=False)
    doa_norm = Column(Float, nullable=False)
    toa_norm = Column(Float, nullable=False)

    job = relationship("TrainingJob", back_populates="predictions")


class ScenarioMeta(Base):
    __tablename__ = "scenario_meta"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(String(16), unique=True, nullable=False)
    name = Column(String(64), nullable=False)
    description = Column(Text, nullable=False)
    config_json = Column(Text, nullable=False)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def job_to_dict(job: TrainingJob) -> dict:
    return {
        "id": job.id,
        "name": job.name,
        "modality_set": job.modality_set,
        "status": job.status,
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "lr": job.lr,
        "train_samples": job.train_samples,
        "test_samples": job.test_samples,
        "num_emitters": job.num_emitters,
        "embed_dim": job.embed_dim,
        "fusion_mode": getattr(job, "fusion_mode", "pulse") or "pulse",
        "aux_lambda": getattr(job, "aux_lambda", 0.2) or 0.2,
        "current_epoch": job.current_epoch,
        "current_phase": getattr(job, "current_phase", "queued") or "queued",
        "phase_message": getattr(job, "phase_message", None),
        "checkpoint_path": job.checkpoint_path,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
