"""합성 레이다 펄스 데이터 생성기 (학습·검증용)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .pulse_generator import (
    NOISE_EMITTER_ID,
    EmitterProfile,
    SyntheticRadarPulseGenerator,
    normalize_pdw,
)
from .scenarios import SimulationScenario

__all__ = [
    "NOISE_EMITTER_ID",
    "EmitterProfile",
    "SyntheticRadarPulseGenerator",
    "normalize_pdw",
    "ScenarioDataset",
    "RadarPulseDataset",
    "collate_pulse_batch",
]


def _sample_to_tensors(sample: dict) -> dict[str, torch.Tensor]:
    return {
        "pdw": torch.from_numpy(sample["pdw"]),
        "iq": torch.from_numpy(sample["iq"]),
        "spectrum": torch.from_numpy(sample["spectrum"]),
        "iq_inst": torch.from_numpy(sample["iq_inst"]),
        "iq_tf": torch.from_numpy(sample["iq_tf"]),
        "mod_labels": torch.from_numpy(sample["mod_labels"]),
        "labels": torch.from_numpy(sample["labels"]),
    }


class ScenarioDataset(Dataset):
    def __init__(
        self,
        scenario: SimulationScenario,
        iq_length: int = 256,
        spec_height: int = 64,
        spec_width: int = 64,
    ) -> None:
        self.scenario = scenario
        self.samples: list[dict] = []
        gen = SyntheticRadarPulseGenerator(
            iq_length=iq_length,
            spec_height=spec_height,
            spec_width=spec_width,
            snr_db=scenario.snr_db,
            pri_modulation=scenario.pri_modulation,
            seed=scenario.seed,
        )
        for _ in range(scenario.num_samples):
            sample = gen.generate_interleaved_sequence(
                num_emitters=scenario.num_emitters,
                pulses_per_emitter=scenario.pulses_per_emitter,
                drop_rate=scenario.drop_rate,
                noise_pulse_rate=scenario.noise_pulse_rate,
            )
            sample["pdw"] = normalize_pdw(sample["pdw"])
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _sample_to_tensors(self.samples[idx])


class RadarPulseDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 200,
        num_emitters: int = 3,
        pulses_per_emitter: int = 20,
        drop_rate: float = 0.1,
        snr_db: float = 15.0,
        iq_length: int = 256,
        spec_height: int = 64,
        spec_width: int = 64,
        seed: int = 42,
    ) -> None:
        scenario = SimulationScenario(
            scenario_id="custom",
            name="Custom",
            description="RadarPulseDataset",
            num_emitters=num_emitters,
            pulses_per_emitter=pulses_per_emitter,
            drop_rate=drop_rate,
            snr_db=snr_db,
            num_samples=num_samples,
            seed=seed,
        )
        self.samples = ScenarioDataset(scenario, iq_length, spec_height, spec_width).samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _sample_to_tensors(self.samples[idx])


def collate_pulse_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_n = max(item["pdw"].shape[0] for item in batch)
    pdw_dim = batch[0]["pdw"].shape[1]
    iq_shape = batch[0]["iq"].shape[1:]
    inst_shape = batch[0]["iq_inst"].shape[1:]
    tf_shape = batch[0]["iq_tf"].shape[1:]
    spec_shape = batch[0]["spectrum"].shape[1:]

    b = len(batch)
    pdw = torch.zeros(b, max_n, pdw_dim)
    iq = torch.zeros(b, max_n, *iq_shape)
    iq_inst = torch.zeros(b, max_n, *inst_shape)
    iq_tf = torch.zeros(b, max_n, *tf_shape)
    spectrum = torch.zeros(b, max_n, *spec_shape)
    labels = torch.full((b, max_n), -1, dtype=torch.long)
    mod_labels = torch.full((b, max_n), -1, dtype=torch.long)
    mask = torch.zeros(b, max_n, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["pdw"].shape[0]
        pdw[i, :n] = item["pdw"]
        iq[i, :n] = item["iq"]
        iq_inst[i, :n] = item["iq_inst"]
        iq_tf[i, :n] = item["iq_tf"]
        spectrum[i, :n] = item["spectrum"]
        labels[i, :n] = item["labels"]
        mod_labels[i, :n] = item["mod_labels"]
        mask[i, :n] = True

    return {
        "pdw": pdw,
        "iq": iq,
        "iq_inst": iq_inst,
        "iq_tf": iq_tf,
        "spectrum": spectrum,
        "labels": labels,
        "mod_labels": mod_labels,
        "mask": mask,
    }
