"""Supervised Contrastive Loss (Khosla et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    동일 방사원(레이블) 펄스 쌍을 가깝게, 다른 방사원 펄스 쌍을 멀게 학습.

    Args:
        temperature: softmax temperature
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N, D) 또는 (M, D) — L2 정규화 권장
            labels: (B, N) 또는 (M,) — 방사원 ID
        Returns:
            scalar loss
        """
        if embeddings.dim() == 3:
            b, n, d = embeddings.shape
            embeddings = embeddings.reshape(b * n, d)
            labels = labels.reshape(b * n)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        device = embeddings.device
        batch_size = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(torch.matmul(embeddings, embeddings.T), self.temperature)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        logits_max, _ = torch.max(logits * logits_mask + logits_mask * (-1e9), dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mask_pos_pairs = mask.sum(dim=1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-8, torch.ones_like(mask_pos_pairs), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs
        loss = -mean_log_prob_pos.mean()
        return loss
