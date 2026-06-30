"""멀티모달 Transformer 융합 및 펄스 클러스터링 모델 (설계서 2~3.7)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import IQEncoder, PDWEncoder, SpectrumEncoder, TOATemporalEmbedding

ModalityName = Literal["pdw", "iq", "spectrum"]
ModalitySet = Literal["pdw", "pdw_iq", "full"]
FusionMode = Literal["pulse", "token"]

MODALITY_CONFIG: dict[ModalitySet, list[ModalityName]] = {
    "pdw": ["pdw"],
    "pdw_iq": ["pdw", "iq"],
    "full": ["pdw", "iq", "spectrum"],
}


class MultimodalPulseClusteringModel(nn.Module):
    """
    Option B (pulse): 펄스별 모달리티 concat → projection → N 토큰 Transformer
    Option A (token): 3N 토큰 Transformer → mean pool
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pdw_dim: int = 5,
        modality_set: ModalitySet = "full",
        fusion_mode: FusionMode = "pulse",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_set = modality_set
        self.fusion_mode = fusion_mode
        self.active_modalities = MODALITY_CONFIG[modality_set]
        self.num_modalities = len(self.active_modalities)

        self.pdw_encoder = PDWEncoder(pdw_dim=pdw_dim, out_dim=embed_dim, dropout=dropout)
        self.iq_encoder = IQEncoder(out_dim=embed_dim, dropout=dropout)
        self.spectrum_encoder = SpectrumEncoder(out_dim=embed_dim, dropout=dropout)

        self.modality_embeddings = nn.Parameter(
            torch.randn(self.num_modalities, embed_dim) * 0.02
        )
        self.temporal_embedding = TOATemporalEmbedding(embed_dim)

        if fusion_mode == "pulse":
            self.pulse_fusion = nn.Sequential(
                nn.Linear(embed_dim * self.num_modalities, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _encode_iq(
        self, iq: torch.Tensor, iq_inst: torch.Tensor | None, iq_tf: torch.Tensor | None, return_aux: bool
    ):
        return self.iq_encoder(iq, iq_inst, iq_tf, return_aux=return_aux)

    def _encode_single(
        self,
        name: ModalityName,
        pdw: torch.Tensor,
        iq: torch.Tensor,
        spectrum: torch.Tensor,
        iq_inst: torch.Tensor | None,
        iq_tf: torch.Tensor | None,
        return_aux: bool,
    ):
        if name == "pdw":
            return self.pdw_encoder(pdw), None
        if name == "iq":
            out = self._encode_iq(iq, iq_inst, iq_tf, return_aux)
            if return_aux:
                return out[0], out[1]
            return out, None
        return self.spectrum_encoder(spectrum), None

    def encode_tokens(
        self,
        pdw: torch.Tensor,
        iq: torch.Tensor,
        spectrum: torch.Tensor,
        iq_inst: torch.Tensor | None = None,
        iq_tf: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        feats: list[torch.Tensor] = []
        mod_logits = None

        for idx, m in enumerate(self.active_modalities):
            feat, aux = self._encode_single(
                m, pdw, iq, spectrum, iq_inst, iq_tf, return_aux and m == "iq"
            )
            feat = feat + self.modality_embeddings[idx]
            feats.append(feat)
            if aux is not None:
                mod_logits = aux

        b, n, d = feats[0].shape
        toa_emb = self.temporal_embedding(pdw[:, :, 4])

        if self.fusion_mode == "pulse":
            concat = torch.cat(feats, dim=-1)
            tokens = self.pulse_fusion(concat) + toa_emb
            return tokens, mod_logits

        stacked = torch.stack(feats, dim=2).reshape(b, n * self.num_modalities, d)
        toa_rep = toa_emb.repeat_interleave(self.num_modalities, dim=1)
        return stacked + toa_rep, mod_logits

    def forward(
        self,
        pdw: torch.Tensor,
        iq: torch.Tensor,
        spectrum: torch.Tensor,
        iq_inst: torch.Tensor | None = None,
        iq_tf: torch.Tensor | None = None,
        return_aux: bool = False,
        return_tokens: bool = False,
    ):
        tokens, mod_logits = self.encode_tokens(
            pdw, iq, spectrum, iq_inst, iq_tf, return_aux=return_aux
        )
        fused = self.transformer(tokens)

        if self.fusion_mode == "token":
            b, _, d = fused.shape
            n = fused.shape[1] // self.num_modalities
            fused = fused.reshape(b, n, self.num_modalities, d)
            pulse_embeddings = fused.mean(dim=2)
        else:
            pulse_embeddings = fused

        pulse_embeddings = self.output_proj(pulse_embeddings)

        if return_aux and mod_logits is not None:
            if return_tokens:
                return pulse_embeddings, mod_logits, tokens
            return pulse_embeddings, mod_logits
        if return_tokens:
            return pulse_embeddings, tokens
        return pulse_embeddings

    def get_embedding(self, pdw, iq, spectrum, iq_inst=None, iq_tf=None) -> torch.Tensor:
        emb = self.forward(pdw, iq, spectrum, iq_inst, iq_tf)
        return F.normalize(emb, p=2, dim=-1)
