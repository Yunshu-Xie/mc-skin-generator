"""Procedural (non-AI) generation for skin faces that don't need per-pixel detail.

Only head (6 faces), body_front, and any AI-chosen "detail faces" are
generated pixel-by-pixel. Everything else is a shaded flat fill built
directly from a handful of named colors (see PALETTE_ROLES in skin_map.py).
Shading is derived purely from one `main` hex per part via HSL lightness
adjustment — no separate shadow color needed here, so left/right pairs and
every face just reuse the same main color with a per-orientation bias.
"""

from __future__ import annotations

import colorsys

from app.services.skin_map import ModelType, get_all_regions

# Faces the AI generates directly; everything else in get_all_regions()
# (excluding overlay groups, which are unused) is procedural unless also
# excluded via the `exclude_keys` param (AI-chosen detail faces).
AI_GENERATED_KEYS = {
    "head_front",
    "head_back",
    "head_top",
    "head_bottom",
    "head_left",
    "head_right",
    "body_front",
}

DEFAULT_COLORS: dict[str, str] = {
    "shirt_main": "#3B5998",
    "arm_main": "#C4A882",
    "pants_main": "#1A1A3E",
    "shoe_color": "#2B2B2B",
}

# Lightness delta applied to `main` per face orientation, so a flat-colored
# part still reads as a 3D form instead of identical tiles on every side.
ORIENTATION_BIAS: dict[str, float] = {
    "top": 0.12,
    "front": 0.0,
    "right": 0.0,
    "left": -0.08,
    "back": -0.15,
    "bottom": -0.22,
}
EDGE_STEP = 0.12  # extra darkening for the outermost 1px ring


def _adjust_lightness(hex_color: str, delta: float) -> str:
    """Shift a hex color's HSL lightness by `delta`, clamped to [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = min(1.0, max(0.0, lightness + delta))
    r2, g2, b2 = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02X}{:02X}{:02X}".format(round(r2 * 255), round(g2 * 255), round(b2 * 255))


def shade_face(h: int, w: int, main: str, face: str) -> list[list[str]]:
    """A shaded grid: interior tone biased by face orientation, 1px darker edge ring."""
    bias = ORIENTATION_BIAS.get(face, 0.0)
    interior = _adjust_lightness(main, bias)
    edge = _adjust_lightness(main, bias - EDGE_STEP)
    return [
        [edge if (row in (0, h - 1) or col in (0, w - 1)) else interior for col in range(w)]
        for row in range(h)
    ]


def generate_procedural_regions(
    colors: dict[str, str],
    model: ModelType,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, list[list[str]]]:
    """Build hex pixel grids for every face not covered by AI generation."""
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    skip = AI_GENERATED_KEYS | exclude_keys
    out: dict[str, list[list[str]]] = {}

    for face, rect in regions["body"].items():
        key = f"body_{face}"
        if key in skip:
            continue
        out[key] = shade_face(rect.h, rect.w, c["shirt_main"], face)

    for part in ("right_arm", "left_arm"):
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in skip:
                continue
            out[key] = shade_face(rect.h, rect.w, c["arm_main"], face)

    for part in ("right_leg", "left_leg"):
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in skip:
                continue
            if face == "bottom":
                out[key] = [[c["shoe_color"]] * rect.w for _ in range(rect.h)]
            else:
                out[key] = shade_face(rect.h, rect.w, c["pants_main"], face)

    return out
