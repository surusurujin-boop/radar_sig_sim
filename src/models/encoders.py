"""모달리티별 특징 추출 인코더 (설계서 3.1~3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.modulation_features import NUM_MODULATION_CLASSES


class PDWEncoder(nn.Module):
    """PDW(CF, PW, PA, DOA, TOA) MLP 인코더 — 보조 문맥."""

    def __init__(
        self,
        pdw_dim: int = 5,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = pdw_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, pdw: torch.Tensor) -> torch.Tensor:
        return self.net(pdw)


class IQEncoder(nn.Module):
    """
    IQ 3-branch 인코더 (설계서 3.2.3).
    raw IQ | 순시 특징(φ, IF, A) | STFT/WVD-like TF
    """

    def __init__(self, out_dim: int = 256, dropout: float = 0.1, tf_channels: int = 1) -> None:
        super().__init__()
        half = out_dim // 2
        quarter = out_dim // 4

        self.raw_branch = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.raw_proj = nn.Linear(128, half)

        self.inst_branch = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.inst_proj = nn.Linear(64, quarter)

        self.tf_branch = nn.Sequential(
            nn.Conv2d(tf_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tf_proj = nn.Linear(64, quarter)

        self.fusion = nn.Sequential(
            nn.Linear(half + quarter + quarter, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

        self.mod_aux_head = nn.Linear(out_dim, NUM_MODULATION_CLASSES)

    def _compute_inst_tf_torch(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """배치 내 IQ에서 순시/STFT 특징 on-the-fly (fallback)."""
        b, n, _, length = iq.shape
        flat = iq.reshape(b * n, 2, length)
        i, q = flat[:, 0], flat[:, 1]
        phase = torch.atan2(q, i)
        inst_freq = F.pad(phase[:, 1:] - phase[:, :-1], (1, 0)) / (2 * 3.14159)
        amp = torch.sqrt(i * i + q * q + 1e-8)
        inst = torch.stack([phase, inst_freq, amp], dim=1)

        spec = torch.abs(torch.fft.fftshift(torch.fft.fft(flat.to(torch.complex64), dim=-1), dim=-1))
        spec = spec / (spec.amax(dim=-1, keepdim=True).values + 1e-8)
        tf = spec.unsqueeze(1)
        if tf.shape[-1] < 8:
            tf = F.interpolate(tf, size=(8, 8), mode="bilinear", align_corners=False)
        else:
            tf = F.interpolate(tf, size=(64, 64), mode="bilinear", align_corners=False)
        return inst, tf

    def forward(
        self,
        iq: torch.Tensor,
        iq_inst: torch.Tensor | None = None,
        iq_tf: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            iq: (B, N, 2, L)
            iq_inst: (B, N, 3, L) optional
            iq_tf: (B, N, 1, H, W) optional
        """
        b, n, _, length = iq.shape
        if iq_inst is None or iq_tf is None:
            inst_flat, tf_flat = self._compute_inst_tf_torch(iq)
            iq_inst = inst_flat.reshape(b, n, 3, length)
            iq_tf = tf_flat.reshape(b, n, 1, tf_flat.shape[-2], tf_flat.shape[-1])

        x_raw = iq.reshape(b * n, 2, length)
        feat_raw = self.raw_proj(self.raw_branch(x_raw).squeeze(-1))

        x_inst = iq_inst.reshape(b * n, 3, length)
        feat_inst = self.inst_proj(self.inst_branch(x_inst).squeeze(-1))

        x_tf = iq_tf.reshape(b * n, *iq_tf.shape[2:])
        if x_tf.dim() == 3:
            x_tf = x_tf.unsqueeze(1)
        feat_tf = self.tf_proj(self.tf_branch(x_tf).flatten(1))

        token = self.fusion(torch.cat([feat_raw, feat_inst, feat_tf], dim=-1))
        token = token.reshape(b, n, -1)

        if return_aux:
            return token, self.mod_aux_head(token)
        return token


class SpectrumEncoder(nn.Module):
    """스펙트로그램 2D CNN 인코더."""

    def __init__(self, in_channels: int = 1, out_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Linear(128, out_dim), nn.LayerNorm(out_dim), nn.Dropout(dropout))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.dim() == 4:
            spectrum = spectrum.unsqueeze(2)
        b, n, c, h, w = spectrum.shape
        x = spectrum.reshape(b * n, c, h, w)
        x = self.proj(self.conv(x).flatten(1))
        return x.reshape(b, n, -1)


class TOATemporalEmbedding(nn.Module):
    """TOA 기반 연속값 temporal embedding (설계서 3.4)."""

    def __init__(self, embed_dim: int, num_frequencies: int = 32) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.proj = nn.Linear(num_frequencies * 2, embed_dim)

    def forward(self, toa: torch.Tensor) -> torch.Tensor:
        """toa: (B, N) normalized TOA."""
        freqs = torch.arange(self.num_frequencies, device=toa.device, dtype=toa.dtype)
        freqs = (2 ** freqs) * 3.14159
        scaled = toa.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(emb)
