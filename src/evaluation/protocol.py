"""평가 프로토콜 정의 및 실행."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.scenarios import SimulationScenario, get_scenarios
from src.data.synthetic import NOISE_EMITTER_ID, ScenarioDataset, collate_pulse_batch
from src.evaluation.metrics import ClusteringMetrics, compute_clustering_metrics


class ClusteringMethod(Protocol):
    """클러스터링 방법 인터페이스."""

    name: str

    def fit_predict_batch(
        self,
        pdw: np.ndarray,
        iq: np.ndarray | None,
        spectrum: np.ndarray | None,
        true_num_emitters: int | None = None,
    ) -> np.ndarray:
        """
        단일 시퀀스에 대한 클러스터 레이블 반환.

        Args:
            pdw: (N, 5) 정규화된 PDW
            iq: (N, 2, L) 또는 None
            spectrum: (N, 1, H, W) 또는 None
            true_num_emitters: Oracle baseline용 실제 방사원 수
        Returns:
            pred_labels: (N,)
        """
        ...


@dataclass
class EvaluationResult:
    """단일 (시나리오, 방법) 평가 결과."""

    scenario_id: str
    method_name: str
    metrics: ClusteringMetrics
    metrics_per_sequence: list[ClusteringMetrics] = field(default_factory=list)

    @property
    def mean_ari(self) -> float:
        return self.metrics.ari

    @property
    def mean_nmi(self) -> float:
        return self.metrics.nmi


@dataclass
class EvaluationProtocol:
    """
    논문 실험 프로토콜.

    - Train/Val/Test: 시나리오별 독립 생성 (seed 분리)
    - 평가 단위: 펄스 단위 (모든 시퀀스의 펄스를 pool하여 macro 평균 + 시퀀스별 micro)
    - 잡음 펄스(label=-1)는 ARI/NMI 계산 시 제외
    - 클러스터 수: 사전 미지정 (HDBSCAN 계열), KMeans baseline만 oracle K 사용
    """

    train_scenario: SimulationScenario | None = None
    test_scenarios: list[SimulationScenario] = field(default_factory=list)
    batch_size: int = 8
    exclude_noise_in_metrics: bool = True

    @classmethod
    def default(cls, scenario_group: str = "quick") -> EvaluationProtocol:
        """기본 프로토콜: S0에서 학습, 지정 그룹 시나리오에서 테스트."""
        from src.data.scenarios import SCENARIO_S0_BASELINE

        return cls(
            train_scenario=SCENARIO_S0_BASELINE,
            test_scenarios=get_scenarios(scenario_group),
        )


def _collect_sequences(dataset: ScenarioDataset) -> list[dict[str, np.ndarray]]:
    sequences = []
    for sample in dataset.samples:
        sequences.append(sample)
    return sequences


def evaluate_method_on_scenario(
    method: ClusteringMethod,
    scenario: SimulationScenario,
    exclude_noise: bool = True,
) -> EvaluationResult:
    """단일 방법 × 단일 시나리오 평가."""
    dataset = ScenarioDataset(scenario)
    sequences = _collect_sequences(dataset)

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_seq_metrics: list[ClusteringMetrics] = []

    for seq in sequences:
        pdw = seq["pdw"]
        iq = seq["iq"]
        spectrum = seq["spectrum"]
        true_labels = seq["labels"]

        true_emitters = len(set(true_labels[true_labels >= 0]))
        pred_labels = method.fit_predict_batch(
            pdw, iq, spectrum, true_num_emitters=true_emitters
        )

        per_seq_metrics.append(
            compute_clustering_metrics(true_labels, pred_labels, exclude_noise=exclude_noise)
        )
        all_true.append(true_labels)
        all_pred.append(pred_labels)

    pooled_true = np.concatenate(all_true)
    pooled_pred = np.concatenate(all_pred)
    pooled_metrics = compute_clustering_metrics(
        pooled_true, pooled_pred, exclude_noise=exclude_noise
    )

    return EvaluationResult(
        scenario_id=scenario.scenario_id,
        method_name=method.name,
        metrics=pooled_metrics,
        metrics_per_sequence=per_seq_metrics,
    )


def evaluate_all(
    methods: list[ClusteringMethod],
    scenarios: list[SimulationScenario],
    exclude_noise: bool = True,
) -> list[EvaluationResult]:
    """전체 (방법 × 시나리오) 매트릭스 평가."""
    results: list[EvaluationResult] = []
    for scenario in scenarios:
        for method in methods:
            results.append(
                evaluate_method_on_scenario(method, scenario, exclude_noise=exclude_noise)
            )
    return results


def format_results_table(results: list[EvaluationResult]) -> str:
    """결과를 ASCII 테이블로 포맷."""
    if not results:
        return "No results."

    methods = sorted({r.method_name for r in results})
    scenarios = sorted({r.scenario_id for r in results}, key=lambda x: x)

    header = f"{'Scenario':<8}" + "".join(f"{m[:14]:>16}" for m in methods)
    lines = [header, "-" * len(header)]

    for sid in scenarios:
        row = f"{sid:<8}"
        for m in methods:
            match = [r for r in results if r.scenario_id == sid and r.method_name == m]
            if match:
                row += f"{match[0].metrics.ari:>16.4f}"
            else:
                row += f"{'N/A':>16}"
        lines.append(row)

    lines.append("")
    lines.append("Metric: ARI (pooled, noise excluded)")
    return "\n".join(lines)


@torch.no_grad()
def extract_model_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """딥러닝 모델 임베딩 추출."""
    model.eval()
    all_emb: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        pdw = batch["pdw"].to(device)
        iq = batch["iq"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["labels"]
        mask = batch["mask"]

        emb = model.get_embedding(pdw, iq, spectrum)
        for i in range(emb.shape[0]):
            valid = mask[i]
            all_emb.append(emb[i, valid].cpu().numpy())
            all_labels.append(labels[i, valid].numpy())

    return all_emb, all_labels
