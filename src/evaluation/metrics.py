"""클러스터링 평가 지표."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    v_measure_score,
)


@dataclass
class ClusteringMetrics:
    """군집화 평가 결과."""

    ari: float
    nmi: float
    purity: float
    v_measure: float
    pairwise_f1: float
    n_clusters_pred: int
    n_clusters_true: int
    cluster_count_error: int
    noise_ratio: float  # HDBSCAN -1 비율

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ari": self.ari,
            "nmi": self.nmi,
            "purity": self.purity,
            "v_measure": self.v_measure,
            "pairwise_f1": self.pairwise_f1,
            "n_clusters_pred": self.n_clusters_pred,
            "n_clusters_true": self.n_clusters_true,
            "cluster_count_error": self.cluster_count_error,
            "noise_ratio": self.noise_ratio,
        }


def _purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """클러스터 순도: 각 클러스터 내 최다 클래스 비율의 가중 평균."""
    if len(true_labels) == 0:
        return 0.0
    correct = 0
    for cluster in np.unique(pred_labels):
        mask = pred_labels == cluster
        if not mask.any():
            continue
        _, counts = np.unique(true_labels[mask], return_counts=True)
        correct += counts.max()
    return correct / len(true_labels)


def _pairwise_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """동일 방사원 페어 분류 F1 (micro)."""
    n = len(true_labels)
    if n < 2:
        return 0.0
    true_pairs = true_labels[:, None] == true_labels[None, :]
    pred_pairs = pred_labels[:, None] == pred_labels[None, :]
    upper = np.triu_indices(n, k=1)
    y_true = true_pairs[upper].astype(int)
    y_pred = pred_pairs[upper].astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_clustering_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    exclude_noise: bool = True,
) -> ClusteringMetrics:
    """
    군집화 성능 지표 계산.

    Args:
        true_labels: 정답 방사원 ID (-1 = 잡음 펄스)
        pred_labels: 예측 클러스터 ID (-1 = HDBSCAN noise)
        exclude_noise: True이면 true=-1 또는 pred=-1인 샘플 제외 후 평가
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    noise_ratio = float((pred_labels == -1).mean()) if len(pred_labels) > 0 else 0.0

    if exclude_noise:
        valid = (true_labels >= 0) & (pred_labels >= 0)
    else:
        valid = np.ones(len(true_labels), dtype=bool)

    n_clusters_true = len(set(true_labels[true_labels >= 0])) if exclude_noise else len(set(true_labels))
    unique_pred = set(pred_labels[pred_labels >= 0])
    n_clusters_pred = len(unique_pred)

    if valid.sum() == 0:
        return ClusteringMetrics(
            ari=0.0,
            nmi=0.0,
            purity=0.0,
            v_measure=0.0,
            pairwise_f1=0.0,
            n_clusters_pred=n_clusters_pred,
            n_clusters_true=n_clusters_true,
            cluster_count_error=abs(n_clusters_pred - n_clusters_true),
            noise_ratio=noise_ratio,
        )

    y_true = true_labels[valid]
    y_pred = pred_labels[valid]

    return ClusteringMetrics(
        ari=float(adjusted_rand_score(y_true, y_pred)),
        nmi=float(normalized_mutual_info_score(y_true, y_pred)),
        purity=_purity(y_true, y_pred),
        v_measure=float(v_measure_score(y_true, y_pred)),
        pairwise_f1=_pairwise_f1(y_true, y_pred),
        n_clusters_pred=n_clusters_pred,
        n_clusters_true=n_clusters_true,
        cluster_count_error=abs(n_clusters_pred - n_clusters_true),
        noise_ratio=noise_ratio,
    )
