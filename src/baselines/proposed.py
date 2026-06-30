"""제안 멀티모달 Transformer + HDBSCAN baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.clustering import HDBSCANClusterer
from src.models import MultimodalPulseClusteringModel


class ProposedModelBaseline:
    """학습된 MultimodalPulseClusteringModel + HDBSCAN."""

    name = "Proposed (Ours)"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        min_cluster_size: int = 5,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})
        self.model = MultimodalPulseClusteringModel(
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            modality_set=config.get("modality_set", "full"),
            fusion_mode=config.get("fusion_mode", "pulse"),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)

    @torch.no_grad()
    def fit_predict_batch(
        self,
        pdw: np.ndarray,
        iq: np.ndarray | None = None,
        spectrum: np.ndarray | None = None,
        true_num_emitters: int | None = None,
    ) -> np.ndarray:
        if iq is None or spectrum is None:
            raise ValueError("Proposed model requires IQ and spectrum inputs")

        pdw_t = torch.from_numpy(pdw).unsqueeze(0).float().to(self.device)
        iq_t = torch.from_numpy(iq).unsqueeze(0).float().to(self.device)
        spec_t = torch.from_numpy(spectrum).unsqueeze(0).float().to(self.device)
        emb = self.model.get_embedding(pdw_t, iq_t, spec_t)
        embeddings = emb[0].cpu().numpy()

        result = self.clusterer.fit_predict(embeddings)
        return result.labels
