"""모의 데이터 탐색 API 서비스."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.export import DATA_ROOT, load_sample
from src.data.modulation_features import MODULATION_TYPES
from src.data.scenarios import SimulationScenario
from src.data.synthetic import ScenarioDataset

PDW_LABELS = ["CF (norm)", "PW (log µs)", "PA", "DOA (norm)", "TOA (norm)"]


def _data_roots() -> list[Path]:
    roots = []
    for name in ("DATA", "data"):
        p = Path(name)
        if p.is_dir():
            roots.append(p)
    return roots or [DATA_ROOT]


def list_datasets() -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for root in _data_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in seen:
                continue
            manifest = child / "manifest.json"
            if (child / "train").is_dir() or manifest.exists():
                seen.add(child.name)
                meta = {}
                if manifest.exists():
                    with open(manifest, encoding="utf-8") as f:
                        meta = json.load(f)
                train_n = len(list((child / "train").glob("sample_*.npz"))) if (child / "train").exists() else 0
                test_n = len(list((child / "test").glob("sample_*.npz"))) if (child / "test").exists() else 0
                items.append(
                    {
                        "id": child.name,
                        "path": str(child),
                        "train_samples": train_n,
                        "test_samples": test_n,
                        "meta": meta,
                    }
                )
    items.append({"id": "live", "path": "live", "train_samples": 0, "test_samples": 0, "meta": {}})
    return items


def list_samples(dataset_id: str, split: str) -> list[dict]:
    if dataset_id == "live":
        sample = generate_live_sample()
        return [{"index": 0, "num_pulses": int(sample["pdw"].shape[0]), "source": "live"}]

    for root in _data_roots():
        split_dir = root / dataset_id / split
        if not split_dir.exists():
            continue
        manifest_path = split_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

        samples = []
        for i, entry in enumerate(manifest.get("samples", [])):
            samples.append({"index": i, "num_pulses": entry.get("num_pulses", 0), "file": entry.get("file")})

        if not samples:
            files = sorted(split_dir.glob("sample_*.npz"))
            for i, f in enumerate(files):
                with np.load(f) as data:
                    n = int(data["pdw"].shape[0])
                samples.append({"index": i, "num_pulses": n, "file": f.name})
        return samples
    return []


def _load_sample_from_disk(dataset_id: str, split: str, sample_index: int) -> dict[str, np.ndarray]:
    for root in _data_roots():
        split_dir = root / dataset_id / split
        fpath = split_dir / f"sample_{sample_index:04d}.npz"
        if fpath.exists():
            return load_sample(fpath)
    raise FileNotFoundError(f"Sample not found: {dataset_id}/{split}/{sample_index}")


def generate_live_sample(seed: int = 42, num_emitters: int = 3) -> dict[str, np.ndarray]:
    scenario = SimulationScenario(
        scenario_id="live",
        name="Live",
        description="실시간 생성",
        num_emitters=num_emitters,
        num_samples=1,
        seed=seed,
    )
    ds = ScenarioDataset(scenario)
    return ds.samples[0]


def get_pulse_detail(
    dataset_id: str,
    split: str,
    sample_index: int,
    pulse_index: int,
    live_seed: int = 42,
    live_emitters: int = 3,
) -> dict:
    if dataset_id == "live":
        sample = generate_live_sample(live_seed, live_emitters)
    else:
        sample = _load_sample_from_disk(dataset_id, split, sample_index)

    n = sample["pdw"].shape[0]
    if pulse_index < 0 or pulse_index >= n:
        raise IndexError(f"Pulse index {pulse_index} out of range 0..{n-1}")

    pdw = sample["pdw"][pulse_index]
    iq = sample["iq"][pulse_index]
    spec = sample["spectrum"][pulse_index]
    label = int(sample["labels"][pulse_index])
    mod_id = int(sample.get("mod_labels", np.array([-1]))[pulse_index])
    mod_name = MODULATION_TYPES[mod_id] if 0 <= mod_id < len(MODULATION_TYPES) else "unknown"

    i_sig = iq[0].tolist()
    q_sig = iq[1].tolist()
    spec_2d = spec[0] if spec.ndim == 3 else spec
    inst = sample.get("iq_inst")
    inst_data = None
    if inst is not None:
        inst_data = {
            "phase": inst[pulse_index, 0].tolist(),
            "inst_freq": inst[pulse_index, 1].tolist(),
            "amplitude": inst[pulse_index, 2].tolist(),
        }

    return {
        "sample_index": sample_index,
        "pulse_index": pulse_index,
        "num_pulses": n,
        "emitter_label": label,
        "modulation_type": mod_name,
        "modulation_id": mod_id,
        "pdw": {
            "columns": PDW_LABELS,
            "values": pdw.tolist(),
            "raw_desc": "정규화된 PDW (CF, PW, PA, DOA, TOA)",
        },
        "iq": {
            "length": len(i_sig),
            "i": i_sig,
            "q": q_sig,
            "inst": inst_data,
        },
        "spectrum": {
            "height": int(spec_2d.shape[0]),
            "width": int(spec_2d.shape[1]),
            "values": spec_2d.tolist(),
        },
    }


def get_sequence_summary(
    dataset_id: str,
    split: str,
    sample_index: int,
    live_seed: int = 42,
    live_emitters: int = 3,
) -> dict:
    if dataset_id == "live":
        sample = generate_live_sample(live_seed, live_emitters)
    else:
        sample = _load_sample_from_disk(dataset_id, split, sample_index)

    pulses = []
    for i in range(sample["pdw"].shape[0]):
        mod_id = int(sample.get("mod_labels", np.full(sample["pdw"].shape[0], -1))[i])
        mod_name = MODULATION_TYPES[mod_id] if 0 <= mod_id < len(MODULATION_TYPES) else "-"
        pulses.append(
            {
                "pulse_index": i,
                "emitter": int(sample["labels"][i]),
                "modulation": mod_name,
                "cf": float(sample["pdw"][i, 0]),
                "pw": float(sample["pdw"][i, 1]),
                "pa": float(sample["pdw"][i, 2]),
            }
        )
    return {"sample_index": sample_index, "num_pulses": len(pulses), "pulses": pulses}
