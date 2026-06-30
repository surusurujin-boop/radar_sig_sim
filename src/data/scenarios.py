"""시뮬레이션 시나리오 정의.

논문 실험 매트릭스에 대응하는 시나리오 세트.
각 시나리오는 PRI 변조, 펄스 누락, SNR, 잡음 펄스, 방사원 수 등을 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PriModulation(str, Enum):
    STABLE = "stable"
    JITTER = "jitter"
    STAGGER = "stagger"
    SLIDING = "sliding"
    GROUP = "group"


@dataclass(frozen=True)
class SimulationScenario:
    """단일 시뮬레이션 시나리오."""

    scenario_id: str
    name: str
    description: str
    num_emitters: int = 3
    pulses_per_emitter: int = 30
    pri_modulation: PriModulation = PriModulation.STABLE
    drop_rate: float = 0.0
    snr_db: float = 15.0
    noise_pulse_rate: float = 0.0
    num_samples: int = 50
    seed: int = 42
    tags: tuple[str, ...] = field(default_factory=tuple)


# ── 표준 시나리오 세트 ──────────────────────────────────────────────

SCENARIO_S0_BASELINE = SimulationScenario(
    scenario_id="S0",
    name="Baseline",
    description="안정 PRI, SNR 15 dB, 펄스 누락·잡음 없음",
    pri_modulation=PriModulation.STABLE,
    drop_rate=0.0,
    snr_db=15.0,
    noise_pulse_rate=0.0,
    tags=("baseline",),
)

SCENARIO_S1_JITTER = SimulationScenario(
    scenario_id="S1",
    name="PRI Jitter",
    description="PRI ±50% 지터 변조",
    pri_modulation=PriModulation.JITTER,
    tags=("pri_modulation",),
)

SCENARIO_S2_STAGGER = SimulationScenario(
    scenario_id="S2",
    name="PRI Stagger",
    description="2-값 교대 PRI (stagger)",
    pri_modulation=PriModulation.STAGGER,
    tags=("pri_modulation",),
)

SCENARIO_S3_SLIDING = SimulationScenario(
    scenario_id="S3",
    name="PRI Sliding",
    description="PRI 슬라이딩(wobulation) 변조",
    pri_modulation=PriModulation.SLIDING,
    tags=("pri_modulation",),
)

SCENARIO_S4_DROP_10 = SimulationScenario(
    scenario_id="S4a",
    name="Drop 10%",
    description="펄스 누락률 10%",
    drop_rate=0.10,
    tags=("dropout",),
)

SCENARIO_S4_DROP_20 = SimulationScenario(
    scenario_id="S4b",
    name="Drop 20%",
    description="펄스 누락률 20%",
    drop_rate=0.20,
    tags=("dropout",),
)

SCENARIO_S4_DROP_30 = SimulationScenario(
    scenario_id="S4c",
    name="Drop 30%",
    description="펄스 누락률 30%",
    drop_rate=0.30,
    tags=("dropout",),
)

SCENARIO_S5_SNR_5 = SimulationScenario(
    scenario_id="S5a",
    name="SNR 5 dB",
    description="저 SNR 5 dB",
    snr_db=5.0,
    tags=("snr",),
)

SCENARIO_S5_SNR_10 = SimulationScenario(
    scenario_id="S5b",
    name="SNR 10 dB",
    description="중 SNR 10 dB",
    snr_db=10.0,
    tags=("snr",),
)

SCENARIO_S5_SNR_15 = SimulationScenario(
    scenario_id="S5c",
    name="SNR 15 dB",
    description="고 SNR 15 dB",
    snr_db=15.0,
    tags=("snr",),
)

SCENARIO_S6_NOISE_5 = SimulationScenario(
    scenario_id="S6a",
    name="Noise 5%",
    description="잡음 펄스 5% 유입",
    noise_pulse_rate=0.05,
    tags=("noise",),
)

SCENARIO_S6_NOISE_10 = SimulationScenario(
    scenario_id="S6b",
    name="Noise 10%",
    description="잡음 펄스 10% 유입",
    noise_pulse_rate=0.10,
    tags=("noise",),
)

SCENARIO_S7_MULTI_5 = SimulationScenario(
    scenario_id="S7a",
    name="5 Emitters",
    description="5개 방사원 중첩",
    num_emitters=5,
    tags=("multi_emitter",),
)

SCENARIO_S7_MULTI_8 = SimulationScenario(
    scenario_id="S7b",
    name="8 Emitters",
    description="8개 방사원 중첩",
    num_emitters=8,
    pulses_per_emitter=25,
    tags=("multi_emitter",),
)

# ── 시나리오 그룹 ──────────────────────────────────────────────────

ALL_SCENARIOS: list[SimulationScenario] = [
    SCENARIO_S0_BASELINE,
    SCENARIO_S1_JITTER,
    SCENARIO_S2_STAGGER,
    SCENARIO_S3_SLIDING,
    SCENARIO_S4_DROP_10,
    SCENARIO_S4_DROP_20,
    SCENARIO_S4_DROP_30,
    SCENARIO_S5_SNR_5,
    SCENARIO_S5_SNR_10,
    SCENARIO_S5_SNR_15,
    SCENARIO_S6_NOISE_5,
    SCENARIO_S6_NOISE_10,
    SCENARIO_S7_MULTI_5,
    SCENARIO_S7_MULTI_8,
]

QUICK_SCENARIOS: list[SimulationScenario] = [
    SCENARIO_S0_BASELINE,
    SCENARIO_S1_JITTER,
    SCENARIO_S4_DROP_20,
    SCENARIO_S5_SNR_10,
    SCENARIO_S6_NOISE_5,
]

SCENARIO_BY_ID: dict[str, SimulationScenario] = {s.scenario_id: s for s in ALL_SCENARIOS}


def get_scenarios(group: str = "all") -> list[SimulationScenario]:
    """시나리오 그룹 반환. group: 'all' | 'quick'."""
    if group == "quick":
        return list(QUICK_SCENARIOS)
    return list(ALL_SCENARIOS)
