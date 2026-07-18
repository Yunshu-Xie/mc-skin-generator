"""Persistence for per-skin generation state.

Only the 6 photo-derived front faces (head + torso + 4 limbs, see
photo_render.py / procedural.py for why the rest is deterministic
fill) have raw index grids; those grids and the full palette are saved
next to the PNG. This was originally the data backing a conversational
color-edit feature, which has since been retired (it depended on the
fixed-role palette that no longer exists). Nothing in production
currently calls `load_skin_state` — the persistence is kept as-is
because it's still exactly the data a possible future edit redesign
would need, and `save_skin_state` costs little to keep calling on
every generation.
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
