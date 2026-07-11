"""Procedural (non-AI) generation for skin faces that don't need per-pixel detail.

Only head_front, body_front, and the four limb-front faces are photo/AI-derived
(see PALETTE_ROLES in skin_map.py for the shared palette) — a front-facing
photo never shows the back/top/sides of someone's head anyway, so those five
head faces are a flat fill from `head_fill_color` just like the other parts'
wrap/cap faces. Every other face is derived directly from its part's front
face: back/left/right copy the front face's per-row color (so a clothing
boundary visible on the front — e.g. a sleeve ending partway down the arm —
stays consistent all the way around the limb), while top/bottom (the small
end-cap faces) are a flat fill from a handful of named colors. No synthetic
shading is added anywhere — Minecraft's own in-game lighting already shades
the 3D model.
"""

from __future__ import annotations

from app.services.skin_map import ModelType, get_all_regions

# Faces that are photo/AI-derived; everything else in get_all_regions()
# (excluding overlay groups, which are unused) is procedural unless already
# present in the caller-supplied `exclude_keys` (i.e. real data exists for it).
AI_GENERATED_KEYS = {
    "head_front",
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
    "head_fill_color": "#5B3A1A",
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
    """Build hex pixel grids for every face not already present in `pixel_data`.

    A face (including a front face) is skipped exactly when its key is in
    `exclude_keys` — callers pass `exclude_keys=frozenset(pixel_data.keys())`
    so "already has real data" and "skip" always mean the same thing,
    whether that data came from the photo pipeline or not at all (a body
    part invisible in the source photo has no front data, isn't excluded,
    and falls through to the same flat-color fallback its wrap faces use —
    it does NOT get silently omitted, which would otherwise leave a
    transparent hole in the assembled skin).
    """
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    out: dict[str, list[list[str]]] = {}

    for face, rect in regions["head"].items():
        key = f"head_{face}"
        if key in exclude_keys:
            continue
        out[key] = [[c["head_fill_color"]] * rect.w for _ in range(rect.h)]

    for part, front_key, color_key in _PARTS:
        front_grid = pixel_data.get(front_key)
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in exclude_keys:
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
