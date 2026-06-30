"""DTOA-PRI 기반 전통 deinterleaving baseline.

TOA 정렬 → DTOA 계산 → PDW + DTOA 통계 특징 → HDBSCAN 클러스터링.
PRI 주기성을 직접 추정하지 않고, DTOA 패턴을 보조 특징으로 활용한다.
"""

from __future__ import annotations

import numpy as np

from src.clustering import HDBSCANClusterer


class DTOAPriBaseline:
    """DTOA + PDW 특징 기반 HDBSCAN deinterleaving."""

    name = "DTOA-PRI"

    def __init__(self, min_cluster_size: int = 5) -> None:
        self.clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)

    def _extract_features(self, pdw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        PDW + DTOA 통계 특징 추출.

        pdw columns: [CF, PW, PA, DOA, TOA] (정규화됨, TOA 순서 보존)
        """
        n = pdw.shape[0]
        if n == 0:
            return np.zeros((0, 8), dtype=np.float32)

        # TOA 순 정렬
        order = np.argsort(pdw[:, 4])
        sorted_pdw = pdw[order]

        dtoa = np.zeros(n, dtype=np.float32)
        dtoa[1:] = sorted_pdw[1:, 4] - sorted_pdw[:-1, 4]

        # local DTOA 통계 (window=3)
        local_mean = np.zeros(n, dtype=np.float32)
        local_std = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lo = max(0, i - 3)
            hi = min(n, i + 4)
            window = dtoa[lo:hi]
            local_mean[i] = window.mean()
            local_std[i] = window.std()

        features = np.column_stack(
            [
                sorted_pdw[:, 0],  # CF
                sorted_pdw[:, 1],  # PW
                sorted_pdw[:, 2],  # PA
                sorted_pdw[:, 3],  # DOA
                dtoa,
                local_mean,
                local_std,
                sorted_pdw[:, 4],  # TOA (보조)
            ]
        ).astype(np.float32)

        # 정규화
        std = features.std(axis=0) + 1e-8
        features = (features - features.mean(axis=0)) / std
        return features, order

    def fit_predict_batch(
        self,
        pdw: np.ndarray,
        iq: np.ndarray | None = None,
        spectrum: np.ndarray | None = None,
        true_num_emitters: int | None = None,
    ) -> np.ndarray:
        features, order = self._extract_features(pdw)
        if len(features) == 0:
            return np.array([], dtype=np.int64)

        result = self.clusterer.fit_predict(features)
        pred_sorted = result.labels

        # 원래 순서로 복원
        pred = np.empty(len(pdw), dtype=np.int64)
        pred[order] = pred_sorted
        return pred
