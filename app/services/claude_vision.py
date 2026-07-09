"""Gemini AI pipeline: single Vision call → head/body_front pixels + palette colors.

Uses the Gemini API's OpenAI-compatible endpoint. Callers pick between
gemini-2.5-flash and gemini-2.5-flash-lite per request via `ai_model`, to
compare quality/cost while both are free-tier.

Only head (6 faces) and body_front are generated pixel-by-pixel by the model —
that's the part that actually needs to look like the uploaded photo. Everything
else (rest of the body, both arms, both legs) is filled in procedurally by
app.services.procedural from a handful of top-level color fields the model
also returns. This keeps the model's output small enough to reliably fit in
one call, instead of the previous two-call, ~4096-pixel design.
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
from app.services.procedural import AI_GENERATED_KEYS, generate_procedural_regions
from app.services.skin_map import PIXEL_KEY_MAP, ModelType, get_all_regions

logger = logging.getLogger(__name__)

AIModel = Literal["flash", "flash-lite"]


def _resolve_model_name(ai_model: AIModel) -> str:
    return (
        settings.gemini_model_flash_lite
        if ai_model == "flash-lite"
        else settings.gemini_model_flash
    )


COLOR_FIELDS = (
    "skin_tone",
    "hair_color",
    "eye_color",
    "shirt_main",
    "shirt_shadow",
    "arm_main",
    "arm_shadow",
    "pants_main",
    "pants_shadow",
    "shoe_color",
)
REQUIRED_COLOR_FIELDS = ("skin_tone", "hair_color", "eye_color")


def _build_prompt(model: ModelType) -> str:
    arm_w = 4 if model == "classic" else 3

    return f"""\
You are a Minecraft skin designer. Look at this photo and design a Minecraft \
skin based on the character/person's appearance.

Output ONLY a JSON object (no markdown, no explanation) with this structure:

{{
  "description": "Brief description of what you see",
  "skin_tone": "#HEX", "hair_color": "#HEX", "hair_style": "short|long|bald|hat|helmet",
  "eye_color": "#HEX",
  "shirt_main": "#HEX", "shirt_shadow": "#HEX",
  "arm_main": "#HEX", "arm_shadow": "#HEX",
  "pants_main": "#HEX", "pants_shadow": "#HEX",
  "shoe_color": "#HEX",

  "palette": ["#HEX", "#HEX", "... up to 8 colors total, index 0-7"],

  "head_front":  [[palette index per pixel] × 8 cols] × 8 rows,
  "head_back":   [8×8 palette indices],
  "head_top":    [8×8 palette indices],
  "head_bottom": [8×8 palette indices],
  "head_left":   [8×8 palette indices],
  "head_right":  [8×8 palette indices],
  "body_front":  [12×8 palette indices]
}}

PIXEL ART RULES for the palette-index grids above (row-major, [row][col]):
- head_front: row 0-1 = hair/forehead, row 2-3 = eyes, row 4 = nose, \
row 5-6 = mouth/chin, row 7 = neck
- body_front: torso/shirt front, {arm_w}-wide-arm model — leave room for \
a visible chest design if the photo shows one (logo, pattern, zipper, etc.)
- Every grid cell is an integer index into "palette" (e.g. 0, 1, 2...), NOT a hex string
- Use 2-4 of the palette colors per face for shading (highlight/base/shadow)
- shirt_main/shirt_shadow, arm_main/arm_shadow, pants_main/pants_shadow, and \
shoe_color are used to procedurally color the rest of the body (back, arms, \
legs) — pick them to match the photo's clothing even though you don't draw \
those faces yourself
- All hex values are exactly 7 chars: "#RRGGBB\""""


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


def _decode_indexed_grid(
    grid: list[list[Any]], palette: list[str]
) -> list[list[str]]:
    """Map a grid of palette indices to hex colors, clamping bad indices to palette[0]."""
    return [
        [
            palette[cell] if isinstance(cell, int) and 0 <= cell < len(palette) else palette[0]
            for cell in row
        ]
        for row in grid
    ]


def _validate_and_decode(
    data: dict[str, Any], model: ModelType
) -> tuple[dict[str, list[list[str]]], dict[str, str], dict[str, Any], bool]:
    """Validate the model response and decode its indexed pixel grids to hex.

    Returns (ai_pixel_data, procedural_colors, metadata, is_complete).
    """
    regions = get_all_regions(model)
    palette = data.get("palette")
    has_palette = isinstance(palette, list) and len(palette) > 0

    pixel_data: dict[str, list[list[str]]] = {}
    if has_palette:
        for key in AI_GENERATED_KEYS:
            group, face = PIXEL_KEY_MAP[key]
            rect = regions[group][face]
            grid = data.get(key)
            if (
                isinstance(grid, list)
                and len(grid) == rect.h
                and all(isinstance(row, list) and len(row) == rect.w for row in grid)
            ):
                pixel_data[key] = _decode_indexed_grid(grid, palette)
            else:
                logger.warning(
                    "Invalid/missing grid for %s: expected %dx%d", key, rect.h, rect.w
                )

    colors = {k: data.get(k, "") for k in COLOR_FIELDS}
    colors_ok = all(colors[k] for k in REQUIRED_COLOR_FIELDS)

    metadata = {
        "description": data.get("description", ""),
        "skin_tone": colors.get("skin_tone", ""),
        "hair_color": colors.get("hair_color", ""),
        "regions_generated": len(pixel_data),
    }

    is_complete = has_palette and colors_ok and len(pixel_data) == len(AI_GENERATED_KEYS)
    return pixel_data, colors, metadata, is_complete


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
        max_tokens=3000,
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
) -> tuple[dict[str, list[list[str]]], dict[str, Any]]:
    """Full pipeline: photo → one Vision call → AI pixels + procedural fill.

    Returns:
        Tuple of (pixel_data dict covering all 36 base regions, metadata dict).
    """
    prepped_bytes, prepped_media_type = _prepare_image(image_bytes)

    pixel_data: dict[str, list[list[str]]] = {}
    colors: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "description": "",
        "skin_tone": "",
        "hair_color": "",
        "regions_generated": 0,
    }

    for attempt in range(max_retries + 1):
        raw = await _call_vision(
            prepped_bytes, prepped_media_type, model, style_notes, ai_model
        )
        pixel_data, colors, metadata, is_complete = _validate_and_decode(raw, model)
        if is_complete:
            break
        logger.warning(
            "Attempt %d: incomplete response (%d/%d AI regions)",
            attempt + 1,
            len(pixel_data),
            len(AI_GENERATED_KEYS),
        )

    logger.info("Analysis complete: %s", metadata.get("description", ""))

    pixel_data.update(generate_procedural_regions(colors, model))
    metadata["ai_model"] = ai_model

    return pixel_data, metadata
