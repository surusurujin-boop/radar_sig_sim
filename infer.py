"""학습된 모델로 펄스 클러스터링 추론."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.clustering import HDBSCANClusterer
from src.data import RadarPulseDataset, collate_pulse_batch
from src.models import MultimodalPulseClusteringModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer radar pulse clustering")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-emitters", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="results/clusters.npz")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> MultimodalPulseClusteringModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = MultimodalPulseClusteringModel(
        embed_dim=config.get("embed_dim", 128),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model: MultimodalPulseClusteringModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    all_emb: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    sample_indices: list[int] = []

    offset = 0
    for batch in loader:
        pdw = batch["pdw"].to(device)
        iq = batch["iq"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["labels"]
        mask = batch["mask"]

        emb = model.get_embedding(pdw, iq, spectrum)
        for i in range(emb.shape[0]):
            valid = mask[i]
            n = int(valid.sum())
            all_emb.append(emb[i, valid].cpu())
            all_labels.append(labels[i, valid])
            sample_indices.extend([offset + i] * n)

    embeddings = torch.cat(all_emb, dim=0).numpy()
    true_labels = torch.cat(all_labels, dim=0).numpy()
    return embeddings, true_labels, sample_indices


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.checkpoint, device)

    dataset = RadarPulseDataset(
        num_samples=args.num_samples,
        num_emitters=args.num_emitters,
        seed=123,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_pulse_batch)

    embeddings, true_labels, _ = extract_embeddings(model, loader, device)

    clusterer = HDBSCANClusterer(min_cluster_size=args.min_cluster_size)
    result = clusterer.fit_predict(embeddings)

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    valid = result.labels >= 0
    if valid.sum() > 0:
        ari = adjusted_rand_score(true_labels[valid], result.labels[valid])
        nmi = normalized_mutual_info_score(true_labels[valid], result.labels[valid])
    else:
        ari, nmi = 0.0, 0.0

    print(f"Clusters found: {result.n_clusters}")
    print(f"Noise points: {result.n_noise}")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        true_labels=true_labels,
        pred_labels=result.labels,
        probabilities=result.probabilities,
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
