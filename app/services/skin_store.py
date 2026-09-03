"""Persistence for generated skins: the PNG plus the data behind it.

The pixel grids and palette are saved next to the PNG so a later edit does not
need the photo, the vision call, or a re-render. Because the whole skin is
drawn from one palette with fixed role slots, recoloring is a substitution on
this stored data — cheap enough to feel instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import settings


def _dir() -> Path:
    path = Path(settings.skins_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def png_path(skin_id: str) -> Path:
    return _dir() / f"{skin_id}.png"


def data_path(skin_id: str) -> Path:
    return _dir() / f"{skin_id}.json"


def companion_path(skin_id: str) -> Path:
    """The 64x64 copy that vanilla Java will actually accept."""
    return _dir() / f"{skin_id}_64.png"


def save_companion(skin_id: str, image: Image.Image) -> None:
    image.save(str(companion_path(skin_id)), "PNG")


def save(skin_id: str, image: Image.Image, record: dict[str, Any]) -> None:
    image.save(str(png_path(skin_id)), "PNG")
    data_path(skin_id).write_text(json.dumps(record), encoding="utf-8")


def load(skin_id: str) -> dict[str, Any] | None:
    path = data_path(skin_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
