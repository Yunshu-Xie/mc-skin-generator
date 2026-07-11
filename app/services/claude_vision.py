"""Gemini AI pipeline: single Vision call → region bounding boxes + a
couple of categorical fields, then deterministic Pillow rendering (see
app.services.photo_render) produces the actual pixels.

Uses the Gemini API's OpenAI-compatible endpoint. Callers pick between
gemini-2.5-flash and gemini-2.5-flash-lite per request via `ai_model`.

The model is asked only to locate 6 body-part regions (head, torso,
right/left arm, right/left leg) as bounding boxes, plus classify
hair_style and eye_shape — a much smaller, more reliable ask than
directly authoring per-pixel palette indices. See
docs/superpowers/specs/2026-07-10-photo-derived-pixel-rendering-design.md
for the full design and rationale.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings
from app.services.photo_render import (
    Bbox,
    build_head_front,
    crop_region,
    dominant_hex,
    downsample_dominant,
    extract_face_colors,
    quantize_shared,
)
from app.services.procedural import AI_GENERATED_KEYS, generate_procedural_regions
from app.services.skin_map import ModelType, get_all_regions

logger = logging.getLogger(__name__)

AIModel = Literal["flash", "flash-lite"]

REGION_KEYS = ("head", "torso", "right_arm", "left_arm", "right_leg", "left_leg")
HAIR_STYLES = ("short", "long", "bald", "hat", "helmet")
EYE_SHAPES = ("narrow", "round")

# Maps a "regions" key to the skin_map part name + the AI_GENERATED_KEYS
# front face it produces, for every region except "head" (handled
# separately since it's colors-plus-template, not crop-and-quantize).
_PHOTO_PARTS = (
    ("torso", "body", "body_front"),
    ("right_arm", "right_arm", "right_arm_front"),
    ("left_arm", "left_arm", "left_arm_front"),
    ("right_leg", "right_leg", "right_leg_front"),
    ("left_leg", "left_leg", "left_leg_front"),
)

# Same defaults as app.services.procedural.DEFAULT_COLORS, used when a front
# face has no photo-derived color at all (region not visible, no sibling
# limb visible either) so its fallback raw grid matches procedural fill.
_FRONT_FALLBACK_COLOR = {
    "body_front": "#3B5998",
    "right_arm_front": "#C4A882",
    "left_arm_front": "#C4A882",
    "right_leg_front": "#1A1A3E",
    "left_leg_front": "#1A1A3E",
}


def _resolve_model_name(ai_model: AIModel) -> str:
    return (
        settings.gemini_model_flash_lite
        if ai_model == "flash-lite"
        else settings.gemini_model_flash
    )


def _build_prompt() -> str:
    return """\
You are looking at a photo to prepare it for turning into a Minecraft skin. \
Do NOT draw or describe any pixels — you only need to locate a few regions \
and classify two features.

Output ONLY a JSON object (no markdown, no explanation) with this structure:

{
  "hair_style": "short|long|bald|hat|helmet",
  "eye_shape": "narrow|round",
  "regions": {
    "head":      {"visible": true, "bbox": [x0, y0, x1, y1]},
    "torso":     {"visible": true, "bbox": [x0, y0, x1, y1]},
    "right_arm": {"visible": true, "bbox": [x0, y0, x1, y1]},
    "left_arm":  {"visible": true, "bbox": [x0, y0, x1, y1]},
    "right_leg": {"visible": true, "bbox": [x0, y0, x1, y1]},
    "left_leg":  {"visible": true, "bbox": [x0, y0, x1, y1]}
  }
}

RULES:
- bbox is [x0, y0, x1, y1], each a fraction from 0 to 1 of the image's \
width/height, top-left origin, so x1 > x0 and y1 > y0.
- "head" should tightly bound just the face (forehead to chin), not the \
whole head/hair.
- "torso" should bound the visible chest/shirt area.
- "right_arm"/"left_arm" are the character's own right/left (mirrored from \
the viewer's perspective) — bound the upper arm/forearm area, not the hand.
- "right_leg"/"left_leg" similarly bound the visible thigh/shin area.
- If a region isn't visible in the photo at all (e.g. a headshot has no \
visible legs), set "visible": false and omit "bbox" for that region — this \
is expected and normal, not an error.
- eye_shape: "narrow" if the eyes read as small/narrow in the photo, \
"round" if they read as large/round.
- hair_style: pick the closest category based on what's visible."""


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


def _valid_bbox(value: Any) -> Bbox | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    if not all(isinstance(v, (int, float)) for v in value):
        return None
    x0, y0, x1, y1 = (float(v) for v in value)
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _validate_regions_response(
    data: dict[str, Any],
) -> tuple[str, str, dict[str, Bbox | None], bool]:
    """Validate the model's region-localization response.

    A malformed individual region degrades to "not visible" rather than
    failing the whole response; only a missing/invalid hair_style,
    eye_shape, or a structurally-malformed `regions` dict fails it.

    Returns (hair_style, eye_shape, {region_key: bbox_or_None}, is_complete).
    """
    hair_style = data.get("hair_style")
    eye_shape = data.get("eye_shape")
    regions_raw = data.get("regions")

    structurally_ok = (
        hair_style in HAIR_STYLES
        and eye_shape in EYE_SHAPES
        and isinstance(regions_raw, dict)
        and set(regions_raw.keys()) == set(REGION_KEYS)
    )
    if not structurally_ok:
        return "", "", {key: None for key in REGION_KEYS}, False

    regions: dict[str, Bbox | None] = {}
    for key in REGION_KEYS:
        entry = regions_raw[key]
        if isinstance(entry, dict) and entry.get("visible") is True:
            regions[key] = _valid_bbox(entry.get("bbox"))
        else:
            regions[key] = None

    return hair_style, eye_shape, regions, True


async def _call_vision(
    image_bytes: bytes,
    media_type: str,
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
        {"type": "text", "text": _build_prompt()},
    ]
    if style_notes:
        user_content.append(
            {"type": "text", "text": f"\nAdditional style notes: {style_notes}"}
        )

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=800,
        temperature=0.2,
        # Gemini 2.5 defaults to "thinking" mode, which eats into max_tokens
        # before any visible output — without this the response gets cut off.
        reasoning_effort="none",
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.choices[0].message.content or ""
    return _extract_json(text)


def _render_photo(
    photo: Image.Image,
    regions: dict[str, Bbox | None],
    hair_style: str,
    eye_shape: str,
    model: ModelType,
) -> tuple[dict[str, list[list[str]]], dict[str, list[list[int]]], list[str], dict[str, str]]:
    """Build pixel_data/raw index grids/palette/named-colors from a located photo.

    Returns (pixel_data_hex, raw_index_grids, palette, named_colors) where
    named_colors has whichever of shirt_main/arm_main/pants_main/
    head_fill_color could be derived from a visible region (missing keys
    fall back to generate_procedural_regions's own defaults).
    """
    face_regions = get_all_regions(model)
    rgb_grids: dict[str, list[list[tuple[int, int, int]]]] = {}
    shapes: dict[str, tuple[int, int]] = {}

    for region_key, part, front_key in _PHOTO_PARTS:
        bbox = regions[region_key]
        if bbox is None:
            continue
        crop = crop_region(photo, bbox)
        if crop is None:
            continue
        rect = face_regions[part]["front"]
        rgb_grids[front_key] = downsample_dominant(crop, rect.h, rect.w)
        shapes[front_key] = (rect.h, rect.w)

    shared_palette, shared_indices = quantize_shared(rgb_grids)

    named_colors: dict[str, str] = {}
    pixel_data: dict[str, list[list[str]]] = {}
    raw_grids: dict[str, list[list[int]]] = {}
    color_key_by_front = {
        "body_front": "shirt_main",
        "right_arm_front": "arm_main",
        "left_arm_front": "arm_main",
        "right_leg_front": "pants_main",
        "left_leg_front": "pants_main",
    }
    for front_key, index_grid in shared_indices.items():
        pixel_data[front_key] = decode_indexed_grid(index_grid, shared_palette)
        raw_grids[front_key] = index_grid
        color_key = color_key_by_front[front_key]
        if color_key not in named_colors:
            named_colors[color_key] = dominant_hex(pixel_data[front_key])

    # A region the model marked not-visible (or whose crop degenerated to
    # zero size) still needs a raw index grid and hex grid of its own —
    # persist_state["pixel_grids"] must always cover all 6 mandatory front
    # faces so a future palette-based edit feature has something to retint
    # for every base region, not just the ones visible in this particular
    # photo. Falls back to the same named color a sibling limb already
    # supplied, or otherwise the same default generate_procedural_regions
    # would have used, so the flat fill this produces is visually identical
    # to what procedural fill would have produced anyway.
    for region_key, part, front_key in _PHOTO_PARTS:
        if front_key in pixel_data:
            continue
        rect = face_regions[part]["front"]
        color_key = color_key_by_front[front_key]
        hexval = named_colors.get(color_key, _FRONT_FALLBACK_COLOR[front_key])
        if hexval in shared_palette:
            idx = shared_palette.index(hexval)
        else:
            idx = len(shared_palette)
            shared_palette.append(hexval)
        raw_grids[front_key] = [[idx] * rect.w for _ in range(rect.h)]
        pixel_data[front_key] = [[hexval] * rect.w for _ in range(rect.h)]
        if color_key not in named_colors:
            named_colors[color_key] = hexval

    head_bbox = regions["head"]
    head_crop = crop_region(photo, head_bbox) if head_bbox is not None else None
    if head_crop is not None:
        face_colors = extract_face_colors(head_crop)
    else:
        face_colors = {
            "skin_tone": "#C4A882",
            "hair_color": "#5B3A1A",
            "eye_color": "#3B5998",
            "mouth_color": "#AA5555",
        }

    named_colors["skin_tone"] = face_colors["skin_tone"]
    named_colors["hair_color"] = face_colors["hair_color"]
    named_colors["head_fill_color"] = (
        face_colors["skin_tone"] if hair_style == "bald" else face_colors["hair_color"]
    )

    head_hex_grid = build_head_front(face_colors, eye_shape)
    head_local_palette = [
        face_colors["skin_tone"],
        face_colors["hair_color"],
        face_colors["eye_color"],
        face_colors["mouth_color"],
    ]
    offset = len(shared_palette)
    head_index_grid = [
        [offset + head_local_palette.index(hexval) for hexval in row] for row in head_hex_grid
    ]
    pixel_data["head_front"] = head_hex_grid
    raw_grids["head_front"] = head_index_grid

    full_palette = shared_palette + head_local_palette
    return pixel_data, raw_grids, full_palette, named_colors


async def generate_skin_data(
    image_bytes: bytes,
    media_type: str,
    model: ModelType = "classic",
    style_notes: str = "",
    ai_model: AIModel = "flash",
    max_retries: int = 2,
) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]:
    """Full pipeline: photo → one Vision call (region localization) → photo_render pixels.

    Returns:
        (pixel_data covering all base regions, metadata, persist_state).
        `persist_state` is ready to pass to skin_store.save_skin_state(skin_id, **persist_state).
    """
    prepped_bytes, prepped_media_type = _prepare_image(image_bytes)
    photo = Image.open(io.BytesIO(prepped_bytes)).convert("RGB")

    hair_style, eye_shape, regions, is_complete = "", "", {}, False
    raw: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        raw = await _call_vision(prepped_bytes, prepped_media_type, style_notes, ai_model)
        hair_style, eye_shape, regions, is_complete = _validate_regions_response(raw)
        if is_complete:
            break
        logger.warning("Attempt %d: region-localization response incomplete", attempt + 1)

    if not is_complete:
        hair_style, eye_shape = "short", "round"
        regions = {key: None for key in REGION_KEYS}

    pixel_data, raw_grids, palette, named_colors = _render_photo(
        photo, regions, hair_style, eye_shape, model
    )

    pixel_data.update(
        generate_procedural_regions(
            pixel_data, named_colors, model, exclude_keys=frozenset(pixel_data.keys())
        )
    )

    metadata = {
        "description": "",
        "skin_tone": named_colors["skin_tone"],
        "hair_color": named_colors["hair_color"],
        "regions_generated": len(AI_GENERATED_KEYS),
        "ai_model": ai_model,
    }

    persist_state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": raw_grids,
        "description": "",
        "hair_style": hair_style,
    }

    return pixel_data, metadata, persist_state
