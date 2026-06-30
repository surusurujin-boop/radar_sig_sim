"""로컬 / Vercel 등 실행 환경 설정."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def is_vercel() -> bool:
    return os.environ.get("VERCEL") == "1"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_db_path() -> Path:
    if is_vercel():
        # Vercel(Linux): /tmp  |  Windows 로컬 VERCEL 테스트: TEMP
        if Path("/tmp").is_dir():
            return Path("/tmp") / "radar.db"
        return Path(tempfile.gettempdir()) / "radar.db"
    return get_project_root() / "data" / "radar.db"


def get_database_url() -> str:
    path = get_db_path().resolve()
    return f"sqlite:///{path.as_posix()}"


def get_data_root() -> Path:
    """학습 데이터 manifest 디렉터리 (DATA 또는 data)."""
    root = get_project_root()
    for name in ("DATA", "data"):
        p = root / name
        if p.is_dir():
            return p
    return root / "data"


def training_enabled() -> bool:
    """PyTorch 학습 Job — Vercel 서버리스에서는 비활성."""
    if is_vercel():
        return False
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def runtime_info() -> dict:
    return {
        "platform": "vercel" if is_vercel() else "local",
        "training_enabled": training_enabled(),
        "data_root": str(get_data_root()),
        "db_path": str(get_db_path()),
    }
