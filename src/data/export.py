"""학습 데이터 DATA 폴더 저장·로드."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .scenarios import SimulationScenario
from .synthetic import ScenarioDataset

DATA_ROOT = Path("DATA")

PDW_COLUMNS = ["cf_norm", "pw_log", "pa", "doa_norm", "toa_norm"]


def _scenario_to_dict(scenario: SimulationScenario) -> dict:
    return {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "description": scenario.description,
        "num_emitters": scenario.num_emitters,
        "pulses_per_emitter": scenario.pulses_per_emitter,
        "pri_modulation": scenario.pri_modulation.value,
        "drop_rate": scenario.drop_rate,
        "snr_db": scenario.snr_db,
        "noise_pulse_rate": scenario.noise_pulse_rate,
        "num_samples": scenario.num_samples,
        "seed": scenario.seed,
    }


def save_samples(
    samples: list[dict[str, np.ndarray]],
    split_dir: Path,
    split_name: str,
    scenario: SimulationScenario | None = None,
) -> Path:
    """펄스 시퀀스 샘플 목록을 npz 파일로 저장."""
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest_samples = []
    for idx, sample in enumerate(samples):
        fname = f"sample_{idx:04d}.npz"
        fpath = split_dir / fname
        np.savez_compressed(
            fpath,
            pdw=sample["pdw"],
            iq=sample["iq"],
            iq_inst=sample["iq_inst"],
            iq_tf=sample["iq_tf"],
            spectrum=sample["spectrum"],
            mod_labels=sample["mod_labels"],
            labels=sample["labels"],
        )
        manifest_samples.append(
            {
                "file": fname,
                "num_pulses": int(sample["pdw"].shape[0]),
                "pdw_shape": list(sample["pdw"].shape),
                "iq_shape": list(sample["iq"].shape),
                "spectrum_shape": list(sample["spectrum"].shape),
            }
        )

    manifest = {
        "split": split_name,
        "num_samples": len(samples),
        "pdw_columns": PDW_COLUMNS,
        "pdw_columns_desc": {
            "cf_norm": "Carrier Frequency (normalized)",
            "pw_log": "Pulse Width log10(us)",
            "pa": "Pulse Amplitude",
            "doa_norm": "Direction of Arrival (normalized)",
            "toa_norm": "Time of Arrival (normalized, order preserved)",
        },
        "samples": manifest_samples,
    }
    if scenario is not None:
        manifest["scenario"] = _scenario_to_dict(scenario)

    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return split_dir


def save_dataset(dataset: ScenarioDataset, output_dir: Path, split: str) -> Path:
    """ScenarioDataset을 DATA 폴더에 저장."""
    return save_samples(dataset.samples, output_dir / split, split, dataset.scenario)


def save_train_test_pair(
    train_scenario: SimulationScenario,
    test_scenario: SimulationScenario,
    output_dir: Path,
    iq_length: int = 256,
    spec_height: int = 64,
    spec_width: int = 64,
    extra_meta: dict | None = None,
) -> Path:
    """train/test 시나리오 생성 후 DATA 폴더에 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ScenarioDataset(train_scenario, iq_length, spec_height, spec_width)
    test_ds = ScenarioDataset(test_scenario, iq_length, spec_height, spec_width)

    save_samples(train_ds.samples, output_dir / "train", "train", train_scenario)
    save_samples(test_ds.samples, output_dir / "test", "test", test_scenario)

    root_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "splits": ["train", "test"],
        "modalities": ["pdw", "iq", "iq_inst", "iq_tf", "spectrum"],
        "train_scenario": _scenario_to_dict(train_scenario),
        "test_scenario": _scenario_to_dict(test_scenario),
    }
    if extra_meta:
        root_manifest.update(extra_meta)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(root_manifest, f, indent=2, ensure_ascii=False)

    return output_dir


def load_sample(npz_path: Path | str) -> dict[str, np.ndarray]:
    """단일 npz 샘플 로드."""
    with np.load(npz_path) as data:
        return {
            "pdw": data["pdw"],
            "iq": data["iq"],
            "spectrum": data["spectrum"],
            "labels": data["labels"],
        }


def load_split(split_dir: Path | str) -> list[dict[str, np.ndarray]]:
    """split 폴더 내 모든 샘플 로드."""
    split_dir = Path(split_dir)
    samples = []
    for npz_path in sorted(split_dir.glob("sample_*.npz")):
        samples.append(load_sample(npz_path))
    return samples
