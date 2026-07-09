"""Procedural (non-AI) generation for skin faces that don't need per-pixel detail.

Only head (6 faces) and body_front are generated pixel-by-pixel by the AI.
Everything else — the rest of the body, both arms, both legs — is a flat
color fill with a 1px darker border, built directly from a handful of
top-level color fields. Left/right pairs reuse the same colors since the
border-fill pattern is already symmetric; no pixel-level mirroring is needed.
"""

from __future__ import annotations

from app.services.skin_map import ModelType, get_all_regions

# Faces the AI generates directly; everything else in get_all_regions()
# (excluding overlay groups, which are unused) is procedural.
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
    "shirt_shadow": "#28406E",
    "arm_main": "#C4A882",
    "arm_shadow": "#9C8266",
    "pants_main": "#1A1A3E",
    "pants_shadow": "#101028",
    "shoe_color": "#2B2B2B",
}


def flat_fill_with_border(
    h: int, w: int, main: str, shadow: str
) -> list[list[str]]:
    """A solid-color grid with a 1px border of `shadow`, `main` everywhere else."""
    return [
        [
            shadow if (row in (0, h - 1) or col in (0, w - 1)) else main
            for col in range(w)
        ]
        for row in range(h)
    ]


def generate_procedural_regions(
    colors: dict[str, str], model: ModelType
) -> dict[str, list[list[str]]]:
    """Build hex pixel grids for every face not covered by AI generation."""
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    out: dict[str, list[list[str]]] = {}

    for face, rect in regions["body"].items():
        if f"body_{face}" in AI_GENERATED_KEYS:
            continue
        out[f"body_{face}"] = flat_fill_with_border(
            rect.h, rect.w, c["shirt_main"], c["shirt_shadow"]
        )

    for part in ("right_arm", "left_arm"):
        for face, rect in regions[part].items():
            out[f"{part}_{face}"] = flat_fill_with_border(
                rect.h, rect.w, c["arm_main"], c["arm_shadow"]
            )

    for part in ("right_leg", "left_leg"):
        for face, rect in regions[part].items():
            if face == "bottom":
                out[f"{part}_{face}"] = [
                    [c["shoe_color"]] * rect.w for _ in range(rect.h)
                ]
            else:
                out[f"{part}_{face}"] = flat_fill_with_border(
                    rect.h, rect.w, c["pants_main"], c["pants_shadow"]
                )

    return out
