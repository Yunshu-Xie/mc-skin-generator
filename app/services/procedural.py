"""Procedural (non-AI) generation for skin faces that don't need per-pixel detail.

Head (6 faces), body_front, and the four limb-front faces are generated
pixel-by-pixel by the AI (see PALETTE_ROLES in skin_map.py for the shared
palette). Every other face is derived directly from its part's front face:
back/left/right copy the front face's per-row color (so a clothing boundary
visible on the front — e.g. a sleeve ending partway down the arm — stays
consistent all the way around the limb), while top/bottom (the small
end-cap faces) are a flat fill from a handful of named colors. No synthetic
shading is added anywhere — Minecraft's own in-game lighting already shades
the 3D model.
"""

from __future__ import annotations

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
    "right_arm_front",
    "left_arm_front",
    "right_leg_front",
    "left_leg_front",
}

DEFAULT_COLORS: dict[str, str] = {
    "shirt_main": "#3B5998",
    "arm_main": "#C4A882",
    "pants_main": "#1A1A3E",
    "shoe_color": "#2B2B2B",
}

# (part group name, that part's mandatory front-face key, named color used
# for its top/bottom caps and as a fallback if the front grid is missing).
_PARTS = (
    ("body", "body_front", "shirt_main"),
    ("right_arm", "right_arm_front", "arm_main"),
    ("left_arm", "left_arm_front", "arm_main"),
    ("right_leg", "right_leg_front", "pants_main"),
    ("left_leg", "left_leg_front", "pants_main"),
)


def propagate_front_row(front_grid: list[list[str]], target_width: int) -> list[list[str]]:
    """Build a same-height grid where row N is a flat fill of front_grid[N]'s middle column.

    Used for the "wrap" faces (back/left/right) of a body part, so a clothing
    boundary visible on the front (e.g. a sleeve ending partway down the arm)
    stays consistent all the way around the limb, without any fabricated shading.
    """
    return [[row[len(row) // 2]] * target_width for row in front_grid]


def generate_procedural_regions(
    pixel_data: dict[str, list[list[str]]],
    colors: dict[str, str],
    model: ModelType,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, list[list[str]]]:
    """Build hex pixel grids for every face not covered by AI generation.

    `pixel_data` is the AI-decoded hex grids collected so far (must contain
    each part's front face for row propagation to apply; falls back to a
    flat named-color fill for a part whose front face isn't present — e.g. a
    skin persisted before this mandatory-front-faces change shipped).
    """
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    skip = AI_GENERATED_KEYS | exclude_keys
    out: dict[str, list[list[str]]] = {}

    for part, front_key, color_key in _PARTS:
        front_grid = pixel_data.get(front_key)
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in skip:
                continue
            if face == "bottom" and part in ("right_leg", "left_leg"):
                out[key] = [[c["shoe_color"]] * rect.w for _ in range(rect.h)]
            elif face in ("top", "bottom"):
                out[key] = [[c[color_key]] * rect.w for _ in range(rect.h)]
            elif front_grid is not None:
                out[key] = propagate_front_row(front_grid, rect.w)
            else:
                out[key] = [[c[color_key]] * rect.w for _ in range(rect.h)]

    return out
