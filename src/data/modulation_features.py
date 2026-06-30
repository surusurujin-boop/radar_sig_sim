"""IQ 신호로부터 변조 특징(순시 위상/주파수/진폭, STFT) 산출."""

from __future__ import annotations

import numpy as np

MODULATION_TYPES = ("cw", "lfm", "nlfm", "fsk", "psk")
MODULATION_TO_ID: dict[str, int] = {m: i for i, m in enumerate(MODULATION_TYPES)}
NUM_MODULATION_CLASSES = len(MODULATION_TYPES)


def compute_inst_features(i: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    순시 위상, 순시 주파수, 순시 진폭 (L, 3).
    """
    z = i + 1j * q
    phase = np.unwrap(np.angle(z)).astype(np.float32)
    dphase = np.diff(phase, prepend=phase[0])
    inst_freq = (dphase / (2 * np.pi)).astype(np.float32)
    amp = np.sqrt(i * i + q * q).astype(np.float32)

    stack = np.stack([phase, inst_freq, amp], axis=-1)
    mean = stack.mean(axis=0, keepdims=True)
    std = stack.std(axis=0, keepdims=True) + 1e-8
    return ((stack - mean) / std).astype(np.float32)


def compute_stft(i: np.ndarray, q: np.ndarray, height: int = 64, width: int = 64) -> np.ndarray:
    """STFT 기반 시간-주파수 표현 (1, H, W)."""
    z = i + 1j * q
    n_fft = min(256, len(z))
    hop = max(1, n_fft // 8)
    frames = []
    for start in range(0, len(z) - n_fft + 1, hop):
        frame = z[start : start + n_fft]
        frames.append(np.fft.fftshift(np.abs(np.fft.fft(frame))))
    if not frames:
        spec = np.abs(np.fft.fftshift(np.fft.fft(z)))
        frames = [spec]

    spec = np.stack(frames, axis=0).astype(np.float32)
    spec = spec / (spec.max() + 1e-8)

    out = np.zeros((height, width), dtype=np.float32)
    sh, sw = spec.shape
    out[: min(height, sh), : min(width, sw)] = spec[: min(height, sh), : min(width, sw)]
    return out[np.newaxis, ...]


def enrich_pulse_iq(iq: np.ndarray, spec_height: int = 64, spec_width: int = 64) -> dict[str, np.ndarray]:
    """
    Args:
        iq: (2, L)
    Returns:
        iq_inst: (3, L), iq_tf: (1, H, W)
    """
    i, q = iq[0], iq[1]
    inst = compute_inst_features(i, q).T  # (3, L)
    tf = compute_stft(i, q, spec_height, spec_width)
    return {"iq_inst": inst.astype(np.float32), "iq_tf": tf}


def mod_type_to_id(mod_type: str) -> int:
    key = mod_type.lower()
    if key == "bpsk":
        key = "psk"
    return MODULATION_TO_ID.get(key, 0)
