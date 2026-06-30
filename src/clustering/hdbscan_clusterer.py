"""학습된 임베딩 공간에 HDBSCAN 클러스터링 적용."""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    n_clusters: int
    n_noise: int


class HDBSCANClusterer:
    """밀도 기반 클러스터링 — 클러스터 수 사전 지정 불필요."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

    def fit_predict(self, embeddings: np.ndarray) -> ClusterResult:
        """
        Args:
            embeddings: (M, D) numpy array
        Returns:
            ClusterResult
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embeddings)
        probabilities = clusterer.probabilities_
        if probabilities is None:
            probabilities = np.ones(len(labels))

        unique = set(labels)
        unique.discard(-1)
        n_noise = int((labels == -1).sum())

        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            n_clusters=len(unique),
            n_noise=n_noise,
        )
