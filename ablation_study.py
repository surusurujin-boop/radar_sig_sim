"""모달리티 ablation 학습 및 정량 비교."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.clustering import HDBSCANClusterer
from src.data.export import DATA_ROOT, save_train_test_pair
from src.data.scenarios import SimulationScenario
from src.data.synthetic import ScenarioDataset, collate_pulse_batch
from src.evaluation.metrics import ClusteringMetrics, compute_clustering_metrics
from src.losses import SupervisedContrastiveLoss
from src.models.multimodal_transformer import MODALITY_CONFIG, ModalitySet, MultimodalPulseClusteringModel

MODALITY_LABELS: dict[ModalitySet, str] = {
    "pdw": "PDW only",
    "pdw_iq": "PDW + IQ",
    "full": "PDW + IQ + Spectrum",
}


@dataclass
class AblationResult:
    modality_set: ModalitySet
    label: str
    metrics: ClusteringMetrics
    sample_true: np.ndarray
    sample_pred: np.ndarray
    sample_pdw: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modality ablation study")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-samples", type=int, default=120)
    parser.add_argument("--test-samples", type=int, default=40)
    parser.add_argument("--num-emitters", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ablation")
    parser.add_argument("--results", type=str, default="results/ablation.json")
    parser.add_argument("--sample-index", type=int, default=0, help="상세 출력할 테스트 샘플 인덱스")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def masked_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    criterion: SupervisedContrastiveLoss,
) -> torch.Tensor:
    valid = mask.view(-1)
    emb = embeddings.reshape(-1, embeddings.shape[-1])[valid]
    lab = labels.reshape(-1)[valid]
    return criterion(emb, lab)


def train_variant(
    modality_set: ModalitySet,
    train_scenario: SimulationScenario,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    embed_dim: int,
) -> MultimodalPulseClusteringModel:
    train_scenario = SimulationScenario(
        scenario_id=train_scenario.scenario_id,
        name=train_scenario.name,
        description=train_scenario.description,
        num_emitters=train_scenario.num_emitters,
        pulses_per_emitter=train_scenario.pulses_per_emitter,
        pri_modulation=train_scenario.pri_modulation,
        drop_rate=train_scenario.drop_rate,
        snr_db=train_scenario.snr_db,
        noise_pulse_rate=train_scenario.noise_pulse_rate,
        num_samples=train_scenario.num_samples,
        seed=train_scenario.seed,
        tags=train_scenario.tags,
    )
    dataset = ScenarioDataset(train_scenario)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pulse_batch,
        drop_last=True,
    )

    model = MultimodalPulseClusteringModel(
        embed_dim=embed_dim,
        modality_set=modality_set,
    ).to(device)

    criterion = SupervisedContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            pdw = batch["pdw"].to(device)
            iq = batch["iq"].to(device)
            spectrum = batch["spectrum"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            embeddings = model(pdw, iq, spectrum)
            loss = masked_contrastive_loss(embeddings, labels, mask, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        if epoch == epochs or epoch % max(1, epochs // 4) == 0:
            avg = total_loss / max(steps, 1)
            print(f"  [{MODALITY_LABELS[modality_set]}] epoch {epoch}/{epochs} loss={avg:.4f}")

    return model


@torch.no_grad()
def evaluate_variant(
    model: MultimodalPulseClusteringModel,
    test_scenario: SimulationScenario,
    device: torch.device,
    sample_index: int = 0,
    min_cluster_size: int = 5,
) -> AblationResult:
    model.eval()
    dataset = ScenarioDataset(test_scenario)
    clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    sample_true = np.array([])
    sample_pred = np.array([])
    sample_pdw = np.array([])

    for idx, seq in enumerate(dataset.samples):
        pdw = seq["pdw"]
        iq = seq["iq"]
        spectrum = seq["spectrum"]
        true_labels = seq["labels"]

        pdw_t = torch.from_numpy(pdw).unsqueeze(0).float().to(device)
        iq_t = torch.from_numpy(iq).unsqueeze(0).float().to(device)
        spec_t = torch.from_numpy(spectrum).unsqueeze(0).float().to(device)

        emb = model.get_embedding(pdw_t, iq_t, spec_t)[0].cpu().numpy()
        pred = clusterer.fit_predict(emb).labels

        all_true.append(true_labels)
        all_pred.append(pred)

        if idx == sample_index:
            sample_true = true_labels
            sample_pred = pred
            sample_pdw = pdw

    metrics = compute_clustering_metrics(
        np.concatenate(all_true),
        np.concatenate(all_pred),
        exclude_noise=True,
    )

    return AblationResult(
        modality_set=model.modality_set,
        label=MODALITY_LABELS[model.modality_set],
        metrics=metrics,
        sample_true=sample_true,
        sample_pred=sample_pred,
        sample_pdw=sample_pdw,
    )


def print_sample_detail(result: AblationResult, max_rows: int = 25) -> None:
    """단일 샘플 펄스별 식별 결과 출력."""
    true = result.sample_true
    pred = result.sample_pred
    pdw = result.sample_pdw
    n = min(len(true), max_rows)

    print(f"\n--- 샘플 펄스별 클러스터 식별 [{result.label}] (상위 {n}/{len(true)}개) ---")
    print(f"{'#':>4} {'True EM':>8} {'Pred CL':>8} {'CF(norm)':>10} {'PW(log)':>10} {'PA':>8} {'DOA':>8} {'TOA':>8} {'Match':>6}")
    print("-" * 82)
    for i in range(n):
        match = "O" if true[i] >= 0 and pred[i] >= 0 else "-"
        print(
            f"{i:>4} {true[i]:>8} {pred[i]:>8} "
            f"{pdw[i, 0]:>10.4f} {pdw[i, 1]:>10.4f} {pdw[i, 2]:>8.4f} "
            f"{pdw[i, 3]:>8.4f} {pdw[i, 4]:>8.4f} {match:>6}"
        )

    n_true = len(set(true[true >= 0]))
    n_pred = len(set(pred[pred >= 0]))
    print(f"\n  실제 방사원 수: {n_true} | 예측 클러스터 수: {n_pred} | 펄스 수: {len(true)}")


def print_comparison_table(results: list[AblationResult]) -> None:
    print("\n" + "=" * 78)
    print("모달리티 Ablation 정량 비교 (테스트 세트 pooled)")
    print("=" * 78)
    print(f"{'Configuration':<22} {'ARI':>8} {'NMI':>8} {'Purity':>8} {'F1':>8} {'|dK|':>6} {'Noise%':>8}")
    print("-" * 78)
    for r in results:
        m = r.metrics
        print(
            f"{r.label:<22} {m.ari:>8.4f} {m.nmi:>8.4f} {m.purity:>8.4f} "
            f"{m.pairwise_f1:>8.4f} {m.cluster_count_error:>6} {m.noise_ratio * 100:>7.1f}%"
        )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_scenario = SimulationScenario(
        scenario_id="train",
        name="Ablation Train",
        description="S0 baseline train split",
        num_emitters=args.num_emitters,
        num_samples=args.train_samples,
        seed=42,
    )
    test_scenario = SimulationScenario(
        scenario_id="test",
        name="Ablation Test",
        description="S0 baseline test split",
        num_emitters=args.num_emitters,
        num_samples=args.test_samples,
        seed=999,
    )

    data_path = save_train_test_pair(
        train_scenario,
        test_scenario,
        DATA_ROOT / "ablation",
        extra_meta={"source": "ablation_study.py"},
    )
    print(f"Training data saved to {data_path}")

    print("=" * 78)
    print("Modality Ablation Study")
    print("=" * 78)
    print(f"Train: {args.train_samples} seq | Test: {args.test_samples} seq | Emitters: {args.num_emitters}")
    print(f"Modalities: {list(MODALITY_CONFIG.keys())}")
    print(f"Device: {device}")
    print()

    results: list[AblationResult] = []
    for modality_set in ("pdw", "pdw_iq", "full"):
        print(f"Training: {MODALITY_LABELS[modality_set]} ...")
        model = train_variant(
            modality_set=modality_set,
            train_scenario=train_scenario,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embed_dim=args.embed_dim,
        )

        ckpt_path = output_dir / f"model_{modality_set}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "embed_dim": args.embed_dim,
                    "modality_set": modality_set,
                    "active_modalities": MODALITY_CONFIG[modality_set],
                },
            },
            ckpt_path,
        )
        print(f"  Saved: {ckpt_path}")

        result = evaluate_variant(
            model, test_scenario, device, sample_index=args.sample_index
        )
        results.append(result)
        print_sample_detail(result)

    print_comparison_table(results)

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_samples": args.train_samples,
                "test_samples": args.test_samples,
                "num_emitters": args.num_emitters,
                "epochs": args.epochs,
                "results": [
                    {
                        "modality_set": r.modality_set,
                        "label": r.label,
                        "active_modalities": MODALITY_CONFIG[r.modality_set],
                        **r.metrics.to_dict(),
                        "sample_true_labels": r.sample_true.tolist(),
                        "sample_pred_labels": r.sample_pred.tolist(),
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
