"""Gemini AI pipeline: single Vision call → head/body_front (+ optional detail
faces) pixels + a fixed-role color palette.

Uses the Gemini API's OpenAI-compatible endpoint. Callers pick between
gemini-2.5-flash and gemini-2.5-flash-lite per request via `ai_model`.

Only head (6 faces), body_front, and any faces the model opportunistically
chooses to draw are generated pixel-by-pixel. Everything else is filled in
procedurally by app.services.procedural, reading named colors out of the
same palette the AI-drawn pixels reference. See
docs/superpowers/specs/2026-07-09-skin-fidelity-and-color-edit-design.md
for the full design.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Literal

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings
from app.services.procedural import AI_GENERATED_KEYS, generate_procedural_regions
from app.services.skin_map import (
    MAX_PALETTE_SIZE,
    MIN_PALETTE_SIZE,
    PALETTE_ROLES,
    PIXEL_KEY_MAP,
    ModelType,
    get_all_regions,
)

logger = logging.getLogger(__name__)

AIModel = Literal["flash", "flash-lite"]

# Face groups eligible for the AI's optional "detail faces" — the base
# (non-overlay) layer only; overlay (hat/jacket) groups are unused by this app.
BASE_GROUPS = {"head", "body", "right_arm", "left_arm", "right_leg", "left_leg"}

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _is_valid_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value))


def _resolve_model_name(ai_model: AIModel) -> str:
    return (
        settings.gemini_model_flash_lite
        if ai_model == "flash-lite"
        else settings.gemini_model_flash
    )


def _build_prompt(model: ModelType) -> str:
    arm_w = 4 if model == "classic" else 3
    extra_face_keys = ", ".join(
        key
        for key in PIXEL_KEY_MAP
        if key not in AI_GENERATED_KEYS and PIXEL_KEY_MAP[key][0] in BASE_GROUPS
    )

    return f"""\
You are a Minecraft skin designer. Look at this photo and design a Minecraft \
skin based on the character/person's appearance.

Output ONLY a JSON object (no markdown, no explanation) with this structure:

{{
  "description": "Brief description of what you see",
  "hair_style": "short|long|bald|hat|helmet",

  "palette": [
    "#HEX index 0 = skin_tone",
    "#HEX index 1 = hair_color",
    "#HEX index 2 = eye_color",
    "#HEX index 3 = shirt_main",
    "#HEX index 4 = shirt_shadow",
    "#HEX index 5 = arm_main",
    "#HEX index 6 = arm_shadow",
    "#HEX index 7 = pants_main",
    "#HEX index 8 = pants_shadow",
    "#HEX index 9 = shoe_color",
    "... optionally 0-6 more freeform colors (index 10-15) for logos, \
patterns, or accessories you want to draw"
  ],

  "head_front":  [[palette index per pixel] × 8 cols] × 8 rows,
  "head_back":   [8×8 palette indices],
  "head_top":    [8×8 palette indices],
  "head_bottom": [8×8 palette indices],
  "head_left":   [8×8 palette indices],
  "head_right":  [8×8 palette indices],
  "body_front":  [12×8 palette indices]
}}

PIXEL ART RULES (row-major, [row][col], every cell is an integer palette index):
- The first 10 palette entries MUST be in exactly this order: skin_tone, \
hair_color, eye_color, shirt_main, shirt_shadow, arm_main, arm_shadow, \
pants_main, pants_shadow, shoe_color. Index 10+ are your choice, up to 16 total.
- head_front: row 0-1 = hair/forehead, row 2-3 = eyes, row 4 = nose, \
row 5-6 = mouth/chin, row 7 = neck
- Eyes and mouth in head_front MUST use a palette index other than \
skin_tone (0) and hair_color (1) — reuse eye_color (2) or another index, so \
facial features are visible against the skin
- body_front: torso/shirt front, {arm_w}-px-wide-arm model
- Most characters do NOT need anything beyond the 7 faces above. ONLY if \
the photo shows a genuinely distinctive design element (a back logo, a \
sleeve pattern, a belt, a cape) that would look wrong as a flat color, add \
up to 4 more faces as extra top-level keys using the same [row][col] \
palette-index format. Valid extra keys: {extra_face_keys}
- shirt_main/shirt_shadow, arm_main/arm_shadow, pants_main/pants_shadow, \
and shoe_color are also used to procedurally color any face you don't draw \
yourself — pick them to match the photo's clothing
- All palette hex values are exactly 7 chars: "#RRGGBB\""""


def _prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Downscale to max_image_dimension and re-encode as JPEG to cut vision tokens."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    max_dim = settings.max_image_dimension
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _get_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible async client for the Gemini API."""
    return AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)  # type: ignore[no-any-return]


def decode_indexed_grid(grid: list[list[Any]], palette: list[str]) -> list[list[str]]:
    """Map a grid of palette indices to hex colors, clamping bad indices to palette[0]."""
    return [
        [
            palette[cell] if isinstance(cell, int) and 0 <= cell < len(palette) else palette[0]
            for cell in row
        ]
        for row in grid
    ]


def _try_decode_face(
    data: dict[str, Any],
    key: str,
    group: str,
    face: str,
    regions: dict[str, dict[str, Any]],
    palette: list[str],
) -> tuple[list[list[int]], list[list[str]]] | None:
    rect = regions[group][face]
    grid = data.get(key)
    if not (
        isinstance(grid, list)
        and len(grid) == rect.h
        and all(
            isinstance(row, list)
            and len(row) == rect.w
            and all(isinstance(cell, int) for cell in row)
            for row in grid
        )
    ):
        return None
    return grid, decode_indexed_grid(grid, palette)


def _validate_and_decode(
    data: dict[str, Any], model: ModelType
) -> tuple[
    dict[str, list[list[str]]], dict[str, list[list[int]]], dict[str, str], dict[str, Any], bool
]:
    """Validate the model response and decode its indexed pixel grids to hex.

    Returns (pixel_data_hex, raw_index_grids, named_colors, metadata, is_complete).
    """
    regions = get_all_regions(model)
    palette = data.get("palette")
    has_valid_palette = (
        isinstance(palette, list)
        and MIN_PALETTE_SIZE <= len(palette) <= MAX_PALETTE_SIZE
        and all(_is_valid_hex(c) for c in palette)
    )

    pixel_data: dict[str, list[list[str]]] = {}
    raw_grids: dict[str, list[list[int]]] = {}

    if has_valid_palette:
        for key in AI_GENERATED_KEYS:
            group, face = PIXEL_KEY_MAP[key]
            decoded = _try_decode_face(data, key, group, face, regions, palette)
            if decoded is None:
                logger.warning("Invalid/missing grid for %s", key)
                continue
            raw_grids[key], pixel_data[key] = decoded

        for key, (group, face) in PIXEL_KEY_MAP.items():
            if key in AI_GENERATED_KEYS or group not in BASE_GROUPS:
                continue
            decoded = _try_decode_face(data, key, group, face, regions, palette)
            if decoded is not None:
                raw_grids[key], pixel_data[key] = decoded

    colors = (
        {name: palette[idx] for name, idx in PALETTE_ROLES.items()} if has_valid_palette else {}
    )

    mandatory_ok = all(key in pixel_data for key in AI_GENERATED_KEYS)

    metadata = {
        "description": data.get("description", ""),
        "skin_tone": colors.get("skin_tone", ""),
        "hair_color": colors.get("hair_color", ""),
        "regions_generated": len(pixel_data),
    }

    is_complete = has_valid_palette and mandatory_ok
    return pixel_data, raw_grids, colors, metadata, is_complete


async def _call_vision(
    image_bytes: bytes,
    media_type: str,
    model: ModelType,
    style_notes: str,
    ai_model: AIModel,
) -> dict[str, Any]:
    client = _get_client()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    user_content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        },
        {"type": "text", "text": _build_prompt(model)},
    ]
    if style_notes:
        user_content.append(
            {"type": "text", "text": f"\nAdditional style notes: {style_notes}"}
        )

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=4000,
        temperature=0.3,
        # Gemini 2.5 defaults to "thinking" mode, which eats into max_tokens
        # before any visible output — without this the response gets cut off.
        reasoning_effort="none",
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.choices[0].message.content or ""
    return _extract_json(text)


async def generate_skin_data(
    image_bytes: bytes,
    media_type: str,
    model: ModelType = "classic",
    style_notes: str = "",
    ai_model: AIModel = "flash",
    max_retries: int = 2,
) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]:
    """Full pipeline: photo → one Vision call → AI pixels + procedural fill.

    Returns:
        (pixel_data covering all base regions, metadata, persist_state).
        `persist_state` is ready to pass to skin_store.save_skin_state(skin_id, **persist_state).
    """
    prepped_bytes, prepped_media_type = _prepare_image(image_bytes)

    pixel_data: dict[str, list[list[str]]] = {}
    raw_grids: dict[str, list[list[int]]] = {}
    colors: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "description": "",
        "skin_tone": "",
        "hair_color": "",
        "regions_generated": 0,
    }
    raw: dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        raw = await _call_vision(prepped_bytes, prepped_media_type, model, style_notes, ai_model)
        pixel_data, raw_grids, colors, metadata, is_complete = _validate_and_decode(raw, model)
        if is_complete:
            break
        logger.warning(
            "Attempt %d: %d/%d mandatory AI regions decoded",
            attempt + 1,
            sum(1 for k in AI_GENERATED_KEYS if k in pixel_data),
            len(AI_GENERATED_KEYS),
        )

    logger.info("Analysis complete: %s", metadata.get("description", ""))

    pixel_data.update(
        generate_procedural_regions(colors, model, exclude_keys=frozenset(pixel_data.keys()))
    )
    metadata["ai_model"] = ai_model

    palette = raw.get("palette") if isinstance(raw.get("palette"), list) else []
    persist_state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": raw_grids,
        "description": metadata.get("description", ""),
        "hair_style": raw.get("hair_style", "") if isinstance(raw.get("hair_style"), str) else "",
    }

    return pixel_data, metadata, persist_state


EDIT_MAX_TOKENS = 300


async def interpret_color_edit(
    current_roles: dict[str, str], instruction: str, ai_model: AIModel
) -> dict[str, str]:
    """Text-only call: map a free-text instruction to changed named color roles.

    Returns a dict of role name -> new hex value, containing only the roles
    that should change. Returns {} if nothing recognized or parsing fails.
    """
    client = _get_client()
    roles_json = json.dumps(current_roles, indent=2)
    prompt = f"""You are editing the colors of an existing Minecraft skin design.

Current named colors:
{roles_json}

User's instruction: "{instruction}"

Output ONLY a JSON object containing the roles that should change and their \
new color as "#RRGGBB". Only use these exact role names: \
{", ".join(current_roles.keys())}. If the instruction doesn't clearly map to \
any of these roles, output {{}}."""

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=EDIT_MAX_TOKENS,
        temperature=0.2,
        reasoning_effort="none",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    try:
        changes = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(changes, dict):
        return {}
    return {
        role: value
        for role, value in changes.items()
        if role in current_roles and _is_valid_hex(value)
    }


async def apply_color_edit(
    state: dict[str, Any], instruction: str
) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]:
    """Apply a conversational color edit to a persisted skin state.

    Returns (pixel_data, metadata, persist_state) — same shapes as
    generate_skin_data, ready to reassemble the PNG and re-save the state.
    """
    model: ModelType = state["model"]
    ai_model: AIModel = state["ai_model"]
    palette = list(state["palette"])

    current_roles = {name: palette[idx] for name, idx in PALETTE_ROLES.items()}
    changes = await interpret_color_edit(current_roles, instruction, ai_model)
    for role, hex_value in changes.items():
        palette[PALETTE_ROLES[role]] = hex_value

    pixel_data: dict[str, list[list[str]]] = {
        key: decode_indexed_grid(grid, palette) for key, grid in state["pixel_grids"].items()
    }
    colors = {name: palette[idx] for name, idx in PALETTE_ROLES.items()}
    pixel_data.update(
        generate_procedural_regions(colors, model, exclude_keys=frozenset(pixel_data.keys()))
    )

    metadata = {
        "description": state.get("description", ""),
        "skin_tone": colors["skin_tone"],
        "hair_color": colors["hair_color"],
        "regions_generated": len(pixel_data),
        "ai_model": ai_model,
        "changed_roles": list(changes.keys()),
    }

    persist_state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": state["pixel_grids"],
        "description": state.get("description", ""),
        "hair_style": state.get("hair_style", ""),
    }

    return pixel_data, metadata, persist_state
