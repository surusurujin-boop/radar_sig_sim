"""멀티모달 Transformer 펄스 클러스터링 학습 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.clustering import HDBSCANClusterer
from src.data import RadarPulseDataset, collate_pulse_batch
from src.data.export import DATA_ROOT, save_samples
from src.data.scenarios import SimulationScenario
from src.losses import SupervisedContrastiveLoss
from src.models import MultimodalPulseClusteringModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal radar pulse clustering model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-emitters", type=int, default=3)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="checkpoints/model.pt")
    parser.add_argument("--data-dir", type=str, default="DATA/cli", help="학습 데이터 저장 경로")
    return parser.parse_args()


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


@torch.no_grad()
def evaluate_clustering(
    model: MultimodalPulseClusteringModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_emb: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        pdw = batch["pdw"].to(device)
        iq = batch["iq"].to(device)
        spectrum = batch["spectrum"].to(device)
        labels = batch["labels"]
        mask = batch["mask"]

        emb = model.get_embedding(pdw, iq, spectrum)
        for i in range(emb.shape[0]):
            valid = mask[i]
            all_emb.append(emb[i, valid].cpu())
            all_labels.append(labels[i, valid])

    if not all_emb:
        return {"ari": 0.0, "nmi": 0.0, "n_clusters": 0}

    embeddings = torch.cat(all_emb, dim=0).numpy()
    true_labels = torch.cat(all_labels, dim=0).numpy()

    clusterer = HDBSCANClusterer(min_cluster_size=5)
    result = clusterer.fit_predict(embeddings)

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    valid = result.labels >= 0
    if valid.sum() == 0:
        return {"ari": 0.0, "nmi": 0.0, "n_clusters": 0}

    ari = adjusted_rand_score(true_labels[valid], result.labels[valid])
    nmi = normalized_mutual_info_score(true_labels[valid], result.labels[valid])

    return {"ari": ari, "nmi": nmi, "n_clusters": result.n_clusters}


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = RadarPulseDataset(
        num_samples=args.num_samples,
        num_emitters=args.num_emitters,
        drop_rate=args.drop_rate,
        snr_db=args.snr_db,
    )

    data_dir = Path(args.data_dir)
    scenario = SimulationScenario(
        scenario_id="cli",
        name="CLI Train",
        description="train.py 학습 데이터",
        num_emitters=args.num_emitters,
        drop_rate=args.drop_rate,
        snr_db=args.snr_db,
        num_samples=args.num_samples,
        seed=42,
    )
    save_samples(dataset.samples, data_dir / "train", "train", scenario)
    print(f"Training data saved to {data_dir / 'train'}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pulse_batch,
        drop_last=True,
    )

    model = MultimodalPulseClusteringModel(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
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

        scheduler.step()
        avg_loss = total_loss / max(steps, 1)
        metrics = evaluate_clustering(model, loader, device)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} | "
            f"ARI={metrics['ari']:.4f} NMI={metrics['nmi']:.4f} "
            f"clusters={metrics['n_clusters']}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
            },
        },
        output_path,
    )
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
