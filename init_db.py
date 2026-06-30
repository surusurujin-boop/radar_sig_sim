"""DB 초기화 및 시드 데이터."""

from __future__ import annotations

import json

from pathlib import Path

from src.data.scenarios import ALL_SCENARIOS
from src.db.models import ScenarioMeta, SessionLocal, init_db


def seed_scenarios() -> None:
    session = SessionLocal()
    try:
        existing = {s.scenario_id for s in session.query(ScenarioMeta).all()}
        for sc in ALL_SCENARIOS:
            if sc.scenario_id in existing:
                continue
            session.add(
                ScenarioMeta(
                    scenario_id=sc.scenario_id,
                    name=sc.name,
                    description=sc.description,
                    config_json=json.dumps(
                        {
                            "num_emitters": sc.num_emitters,
                            "pulses_per_emitter": sc.pulses_per_emitter,
                            "pri_modulation": sc.pri_modulation.value,
                            "drop_rate": sc.drop_rate,
                            "snr_db": sc.snr_db,
                            "noise_pulse_rate": sc.noise_pulse_rate,
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        session.commit()
    finally:
        session.close()


def main() -> None:
    init_db()
    _migrate_db()
    seed_scenarios()
    print("Database initialized: data/radar.db")
    print(f"Seeded {len(ALL_SCENARIOS)} simulation scenarios.")


def _migrate_db() -> None:
    """기존 DB에 신규 컬럼 추가."""
    import sqlite3

    db_path = Path("data/radar.db")
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(training_jobs)")
    cols = {row[1] for row in cur.fetchall()}
    if "fusion_mode" not in cols:
        cur.execute("ALTER TABLE training_jobs ADD COLUMN fusion_mode VARCHAR(16) DEFAULT 'pulse'")
    if "aux_lambda" not in cols:
        cur.execute("ALTER TABLE training_jobs ADD COLUMN aux_lambda FLOAT DEFAULT 0.2")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
