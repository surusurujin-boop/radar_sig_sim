"""Contrastive + 변조 유형 보조 손실 (설계서 3.2.4, 5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.contrastive import SupervisedContrastiveLoss


class CombinedClusteringLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, aux_lambda: float = 0.2) -> None:
        super().__init__()
        self.contrastive = SupervisedContrastiveLoss(temperature=temperature)
        self.aux_lambda = aux_lambda
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        emitter_labels: torch.Tensor,
        mod_logits: torch.Tensor | None = None,
        mod_labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if mask is not None:
            valid = mask.view(-1)
            emb = embeddings.reshape(-1, embeddings.shape[-1])[valid]
            elab = emitter_labels.reshape(-1)[valid]
        else:
            emb = embeddings.reshape(-1, embeddings.shape[-1])
            elab = emitter_labels.reshape(-1)

        loss_con = self.contrastive(emb, elab)
        total = loss_con
        metrics = {"loss_contrastive": float(loss_con.item())}

        if mod_logits is not None and mod_labels is not None:
            if mask is not None:
                mlog = mod_logits.reshape(-1, mod_logits.shape[-1])[valid]
                mlab = mod_labels.reshape(-1)[valid]
            else:
                mlog = mod_logits.reshape(-1, mod_logits.shape[-1])
                mlab = mod_labels.reshape(-1)

            valid_mod = mlab >= 0
            if valid_mod.any():
                loss_aux = self.ce(mlog[valid_mod], mlab[valid_mod])
                total = total + self.aux_lambda * loss_aux
                metrics["loss_aux"] = float(loss_aux.item())

        metrics["loss_total"] = float(total.item())
        return total, metrics
