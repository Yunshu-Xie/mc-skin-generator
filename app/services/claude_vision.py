"""CodeBuddy AI pipeline: image analysis → pixel grid generation.

Uses OpenAI-compatible API via Tencent CodeBuddy (Coding Plan).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
import httpx
from openai import AsyncOpenAI

from app.config import settings
from app.services.skin_map import ModelType, get_all_regions

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """\
You are a Minecraft skin designer. Analyze this image and design a Minecraft skin \
based on the character/person's appearance.

Output ONLY a JSON object (no markdown, no explanation) with this structure:
{
  "description": "Brief description of what you see",
  "skin_tone": "#HEX",
  "hair_color": "#HEX",
  "hair_style": "short|long|bald|hat|helmet",
  "eye_color": "#HEX",
  "head": {
    "front": {"description": "...", "palette": {"skin": "#HEX", "hair": "#HEX", "eyes": "#HEX", "mouth": "#HEX"}},
    "back": {"description": "...", "palette": {"main": "#HEX"}},
    "top": {"description": "...", "palette": {"main": "#HEX"}},
    "bottom": {"description": "...", "palette": {"main": "#HEX"}},
    "left": {"description": "...", "palette": {"skin": "#HEX", "hair": "#HEX"}},
    "right": {"description": "...", "palette": {"skin": "#HEX", "hair": "#HEX"}}
  },
  "body": {
    "front": {"description": "...", "palette": {"main": "#HEX", "accent": "#HEX", "shadow": "#HEX"}},
    "back": {"description": "...", "palette": {"main": "#HEX"}},
    "top": {"description": "...", "palette": {"main": "#HEX"}},
    "bottom": {"description": "...", "palette": {"main": "#HEX"}},
    "left": {"description": "...", "palette": {"main": "#HEX", "shadow": "#HEX"}},
    "right": {"description": "...", "palette": {"main": "#HEX", "shadow": "#HEX"}}
  },
  "right_arm": {
    "front": {"description": "...", "palette": {"main": "#HEX"}},
    "back": {"description": "...", "palette": {"main": "#HEX"}},
    "top": {"description": "...", "palette": {"main": "#HEX"}},
    "bottom": {"description": "...", "palette": {"main": "#HEX"}},
    "left": {"description": "...", "palette": {"main": "#HEX"}},
    "right": {"description": "...", "palette": {"main": "#HEX"}}
  },
  "left_arm": { "...same as right_arm..." : "..." },
  "right_leg": {
    "front": {"description": "...", "palette": {"main": "#HEX"}},
    "back": {"description": "...", "palette": {"main": "#HEX"}},
    "top": {"description": "...", "palette": {"main": "#HEX"}},
    "bottom": {"description": "...", "palette": {"main": "#HEX"}},
    "left": {"description": "...", "palette": {"main": "#HEX"}},
    "right": {"description": "...", "palette": {"main": "#HEX"}}
  },
  "left_leg": { "...same as right_leg..." : "..." },
  "has_overlay": false
}
"""


def _get_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible async client for CodeBuddy.

    CodeBuddy uses X-Api-Key header for authentication instead of the
    standard Bearer token. We set a dummy api_key to satisfy the SDK
    and inject the real key via default_headers.
    """
    return AsyncOpenAI(
        api_key="placeholder",
        base_url=settings.codebuddy_base_url,
        default_headers={
            "X-Api-Key": settings.codebuddy_api_key,
        },
    )


def _build_pixel_prompt(analysis: dict[str, Any], model: ModelType) -> str:
    """Build the prompt for pixel grid generation from analysis results."""
    arm_w = 4 if model == "classic" else 3
    arm_side_w = 4

    return f"""\
You are a pixel artist creating a Minecraft skin. Based on this design plan, \
generate the exact pixel colors for each body part face.

Design plan:
{json.dumps(analysis, indent=2)}

Model type: {model} ({arm_w}px wide arm front/back)

For each region below, output a 2D array of hex color strings ("#RRGGBB").
Use "#00000000" for transparent pixels.

Output ONLY a JSON object (no markdown, no explanation) with these keys. \
Every value is a row-major 2D array [rows][cols]:

{{
  "head_front":   [8 rows × 8 cols],
  "head_back":    [8 rows × 8 cols],
  "head_top":     [8 rows × 8 cols],
  "head_bottom":  [8 rows × 8 cols],
  "head_left":    [8 rows × 8 cols],
  "head_right":   [8 rows × 8 cols],

  "body_front":   [12 rows × 8 cols],
  "body_back":    [12 rows × 8 cols],
  "body_top":     [4 rows × 8 cols],
  "body_bottom":  [4 rows × 8 cols],
  "body_left":    [12 rows × 4 cols],
  "body_right":   [12 rows × 4 cols],

  "right_arm_front":  [12 rows × {arm_w} cols],
  "right_arm_back":   [12 rows × {arm_w} cols],
  "right_arm_top":    [4 rows × {arm_w} cols],
  "right_arm_bottom": [4 rows × {arm_w} cols],
  "right_arm_left":   [12 rows × {arm_side_w} cols],
  "right_arm_right":  [12 rows × {arm_side_w} cols],

  "left_arm_front":   [12 rows × {arm_w} cols],
  "left_arm_back":    [12 rows × {arm_w} cols],
  "left_arm_top":     [4 rows × {arm_w} cols],
  "left_arm_bottom":  [4 rows × {arm_w} cols],
  "left_arm_left":    [12 rows × {arm_side_w} cols],
  "left_arm_right":   [12 rows × {arm_side_w} cols],

  "right_leg_front":  [12 rows × 4 cols],
  "right_leg_back":   [12 rows × 4 cols],
  "right_leg_top":    [4 rows × 4 cols],
  "right_leg_bottom": [4 rows × 4 cols],
  "right_leg_left":   [12 rows × 4 cols],
  "right_leg_right":  [12 rows × 4 cols],

  "left_leg_front":   [12 rows × 4 cols],
  "left_leg_back":    [12 rows × 4 cols],
  "left_leg_top":     [4 rows × 4 cols],
  "left_leg_bottom":  [4 rows × 4 cols],
  "left_leg_left":    [12 rows × 4 cols],
  "left_leg_right":   [12 rows × 4 cols]
}}

PIXEL ART RULES:
- head_front: row 0-1 = hair/forehead, row 2-3 = eyes area, row 4 = nose, \
row 5-6 = mouth/chin, row 7 = neck/bottom
- Use 2-4 shading levels per color for depth (highlight, base, shadow)
- Arms/legs inner sides should be slightly darker
- Clothing seams/edges use a 1-shade-darker variant
- Every hex must be exactly 7 chars (#RRGGBB) or 9 chars (#RRGGBBAA)
- Make the pixel art look good at this tiny resolution — bold shapes, clear features
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)  # type: ignore[no-any-return]


def _validate_pixel_data(
    data: dict[str, Any], model: ModelType
) -> dict[str, list[list[str]]]:
    """Validate and extract pixel grids, keeping only correctly-sized ones."""
    regions = get_all_regions(model)
    from app.services.skin_map import PIXEL_KEY_MAP

    valid: dict[str, list[list[str]]] = {}

    for key, grid in data.items():
        if key not in PIXEL_KEY_MAP:
            continue
        group, face = PIXEL_KEY_MAP[key]
        if group not in regions or face not in regions[group]:
            continue

        rect = regions[group][face]
        if (
            isinstance(grid, list)
            and len(grid) == rect.h
            and all(isinstance(row, list) and len(row) == rect.w for row in grid)
        ):
            valid[key] = grid
        else:
            logger.warning(
                "Invalid grid for %s: expected %dx%d, got %s",
                key,
                rect.h,
                rect.w,
                f"{len(grid)}x{len(grid[0]) if grid else 0}" if isinstance(grid, list) else type(grid),
            )

    return valid


async def analyze_image(
    image_bytes: bytes, media_type: str, model: ModelType, style_notes: str = ""
) -> dict[str, Any]:
    """Step 1: Analyze the uploaded image with CodeBuddy Vision."""
    client = _get_client()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Build user message with image (OpenAI Vision format)
    user_content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_b64}",
            },
        },
        {"type": "text", "text": ANALYSIS_PROMPT},
    ]

    if style_notes:
        user_content.append(
            {"type": "text", "text": f"\nAdditional style notes: {style_notes}"}
        )

    response = await client.chat.completions.create(
        model=settings.codebuddy_vision_model,
        max_tokens=2500,
        temperature=0.3,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.choices[0].message.content or ""
    return _extract_json(text)


async def generate_pixels(
    analysis: dict[str, Any], model: ModelType
) -> dict[str, list[list[str]]]:
    """Step 2: Generate pixel grids from the analysis plan."""
    client = _get_client()

    prompt = _build_pixel_prompt(analysis, model)

    response = await client.chat.completions.create(
        model=settings.codebuddy_text_model,
        max_tokens=10000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content or ""
    raw_data = _extract_json(text)
    return _validate_pixel_data(raw_data, model)


async def generate_skin_data(
    image_bytes: bytes,
    media_type: str,
    model: ModelType = "classic",
    style_notes: str = "",
    max_retries: int = 2,
) -> tuple[dict[str, list[list[str]]], dict[str, Any]]:
    """Full pipeline: image → analysis → pixel data.

    Returns:
        Tuple of (pixel_data dict, analysis metadata dict).
    """
    analysis = await analyze_image(image_bytes, media_type, model, style_notes)
    logger.info("Analysis complete: %s", analysis.get("description", ""))

    pixel_data = await generate_pixels(analysis, model)

    # Check coverage — we expect at least 36 base layer regions (6 parts × 6 faces)
    expected_base = 36
    if len(pixel_data) < expected_base:
        logger.warning(
            "Only got %d/%d regions, retrying pixel generation...",
            len(pixel_data),
            expected_base,
        )
        for attempt in range(max_retries):
            retry_data = await generate_pixels(analysis, model)
            # Merge — keep existing valid data, fill gaps
            for k, v in retry_data.items():
                if k not in pixel_data:
                    pixel_data[k] = v
            if len(pixel_data) >= expected_base:
                break
            logger.warning(
                "Retry %d: %d/%d regions", attempt + 1, len(pixel_data), expected_base
            )

    metadata = {
        "description": analysis.get("description", ""),
        "skin_tone": analysis.get("skin_tone", ""),
        "hair_color": analysis.get("hair_color", ""),
        "regions_generated": len(pixel_data),
    }

    return pixel_data, metadata
