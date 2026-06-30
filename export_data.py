"""기본 학습 데이터를 DATA 폴더에 생성·저장."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.data.export import DATA_ROOT, save_train_test_pair
from src.data.scenarios import SCENARIO_S0_BASELINE, SimulationScenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export training data to DATA folder")
    parser.add_argument("--train-samples", type=int, default=120)
    parser.add_argument("--test-samples", type=int, default=40)
    parser.add_argument("--num-emitters", type=int, default=3)
    parser.add_argument("--output", type=str, default=None, help="출력 경로 (기본: DATA/default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output) if args.output else DATA_ROOT / "default"

    train_scenario = SimulationScenario(
        scenario_id="S0",
        name=SCENARIO_S0_BASELINE.name,
        description=SCENARIO_S0_BASELINE.description,
        num_emitters=args.num_emitters,
        pulses_per_emitter=SCENARIO_S0_BASELINE.pulses_per_emitter,
        pri_modulation=SCENARIO_S0_BASELINE.pri_modulation,
        drop_rate=SCENARIO_S0_BASELINE.drop_rate,
        snr_db=SCENARIO_S0_BASELINE.snr_db,
        noise_pulse_rate=SCENARIO_S0_BASELINE.noise_pulse_rate,
        num_samples=args.train_samples,
        seed=42,
    )
    test_scenario = SimulationScenario(
        scenario_id="S0",
        name=SCENARIO_S0_BASELINE.name,
        description="S0 baseline test split",
        num_emitters=args.num_emitters,
        pulses_per_emitter=SCENARIO_S0_BASELINE.pulses_per_emitter,
        pri_modulation=SCENARIO_S0_BASELINE.pri_modulation,
        drop_rate=SCENARIO_S0_BASELINE.drop_rate,
        snr_db=SCENARIO_S0_BASELINE.snr_db,
        noise_pulse_rate=SCENARIO_S0_BASELINE.noise_pulse_rate,
        num_samples=args.test_samples,
        seed=999,
    )

    path = save_train_test_pair(
        train_scenario,
        test_scenario,
        output_dir,
        extra_meta={
            "source": "export_data.py",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    train_count = len(list((path / "train").glob("sample_*.npz")))
    test_count = len(list((path / "test").glob("sample_*.npz")))
    print(f"Saved to {path}")
    print(f"  train: {train_count} samples")
    print(f"  test:  {test_count} samples")


if __name__ == "__main__":
    main()
