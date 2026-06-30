"""모델·알고리즘 설명 (HTML API용)."""

MODEL_INFO = {
    "title": "멀티모달 Transformer 펄스 클러스터링",
    "encoders": [
        {"name": "PDW Encoder", "type": "MLP (3-layer)", "input": "CF, PW, PA, DOA, TOA", "role": "운용·시간 배열 보조 문맥"},
        {"name": "IQ Encoder", "type": "3-Branch 1D/2D CNN", "input": "Raw IQ + φ/IF/A + STFT", "role": "변조 특성 (핵심)"},
        {"name": "Spectrum Encoder", "type": "2D CNN", "input": "스펙트로그램", "role": "주파수 대역 특성 (핵심)"},
    ],
    "fusion": [
        {"name": "Option B (pulse)", "desc": "펄스별 모달리티 concat → Linear → N 토큰"},
        {"name": "Option A (token)", "desc": "3N 토큰 self-attention (ablation)"},
        {"name": "TOA Embedding", "desc": "Sinusoidal temporal positional encoding"},
    ],
    "transformer": {
        "type": "Pre-LN Transformer Encoder",
        "attention": "Multi-Head Self-Attention",
        "default_layers": 4,
        "default_heads": 8,
        "default_dim": 256,
    },
    "training": [
        {"name": "Supervised Contrastive Loss", "desc": "동일 방사원 펄스 임베딩 거리 최소화 (τ=0.1)"},
        {"name": "Modulation Aux Loss", "desc": "CW/LFM/NLFM/FSK/PSK 보조 분류 Cross-Entropy (λ=0.2)"},
        {"name": "Optimizer", "desc": "AdamW (lr=1e-4, weight_decay=1e-4)"},
    ],
    "inference": [
        {"name": "Embedding", "desc": "L2 정규화 펄스별 D차원 벡터"},
        {"name": "HDBSCAN", "desc": "밀도 기반 클러스터링 (군집 수 사전 불필요)"},
        {"name": "Metrics", "desc": "ARI, NMI, Purity, Pairwise F1"},
    ],
}
