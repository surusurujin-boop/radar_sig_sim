"""PDW 파라미터만 사용하는 클러스터링 baseline."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.clustering import HDBSCANClusterer


class PDWHDBSCANBaseline:
    """PDW(CF, PW, PA, DOA) 특징 HDBSCAN — TOA/DTOA 미사용."""

    name = "PDW-HDBSCAN"

    def __init__(self, min_cluster_size: int = 5) -> None:
        self.clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)

    def fit_predict_batch(
        self,
        pdw: np.ndarray,
        iq: np.ndarray | None = None,
        spectrum: np.ndarray | None = None,
        true_num_emitters: int | None = None,
    ) -> np.ndarray:
        features = pdw[:, :4].astype(np.float32)
        if len(features) == 0:
            return np.array([], dtype=np.int64)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        result = self.clusterer.fit_predict(features)
        return result.labels


class PDWKMeansBaseline:
    """Oracle K KMeans — 방사원 수를 알고 있다고 가정하는 상한 baseline."""

    name = "PDW-KMeans (oracle K)"

    def fit_predict_batch(
        self,
        pdw: np.ndarray,
        iq: np.ndarray | None = None,
        spectrum: np.ndarray | None = None,
        true_num_emitters: int | None = None,
    ) -> np.ndarray:
        features = pdw[:, :4].astype(np.float32)
        if len(features) == 0:
            return np.array([], dtype=np.int64)
        k = true_num_emitters or max(2, min(8, len(features) // 5))
        k = min(k, len(features))
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        return km.fit_predict(features)
