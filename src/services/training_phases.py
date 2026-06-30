"""학습 Job 단계 정의."""

from __future__ import annotations

TRAINING_PHASES = [
    {"id": "queued", "label": "대기열 등록", "desc": "학습 Job이 생성되고 백그라운드 스레드에 등록됩니다."},
    {"id": "data_gen", "label": "합성 데이터 생성", "desc": "PRI/변조/SNR 조건에 따라 PDW·IQ·스펙트럼 펄스열을 생성합니다."},
    {"id": "data_save", "label": "DATA 저장", "desc": "train/test npz 및 manifest.json을 DATA 폴더에 저장합니다."},
    {"id": "model_init", "label": "모델 초기화", "desc": "Multimodal Transformer 인코더·Fusion·Projection 가중치를 초기화합니다."},
    {"id": "train_epoch", "label": "Epoch 학습", "desc": "Supervised Contrastive Loss + 변조유형 Aux Loss로 역전파 학습합니다."},
    {"id": "eval", "label": "클러스터링 검증", "desc": "임베딩 추출 후 HDBSCAN 적용, ARI/NMI를 계산합니다."},
    {"id": "checkpoint", "label": "체크포인트 저장", "desc": "학습된 가중치를 checkpoints/web/ 에 저장합니다."},
    {"id": "done", "label": "완료", "desc": "평가 결과와 펄스 예측을 DB에 기록합니다."},
]

PHASE_ORDER = [p["id"] for p in TRAINING_PHASES]


def phase_index(phase_id: str) -> int:
    try:
        return PHASE_ORDER.index(phase_id)
    except ValueError:
        return 0
