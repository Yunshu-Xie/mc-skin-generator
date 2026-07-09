"""Persistence for per-skin generation state, used by conversational color edits.

Generation only paints 7-11 faces pixel-by-pixel (see procedural.py for
why); the raw palette-index grids and full palette are saved next to the
PNG so a later edit can re-decode with a changed palette without a new
Vision call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


def _state_path(skin_id: str) -> Path:
    return Path(settings.skins_dir) / f"{skin_id}.json"


def save_skin_state(
    skin_id: str,
    model: str,
    ai_model: str,
    palette: list[str],
    pixel_grids: dict[str, list[list[int]]],
    description: str,
    hair_style: str,
) -> None:
    state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": pixel_grids,
        "description": description,
        "hair_style": hair_style,
    }
    _state_path(skin_id).write_text(json.dumps(state))


def load_skin_state(skin_id: str) -> dict[str, Any] | None:
    path = _state_path(skin_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())
