from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    """Load .env.example defaults, then non-empty .env values."""
    load_dotenv(ROOT / ".env.example", override=False)
    env_values = dotenv_values(ROOT / ".env")
    for key, value in env_values.items():
        if value:
            os.environ[key] = value
