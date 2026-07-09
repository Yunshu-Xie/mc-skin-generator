"""Tests for procedural — flat-fill generation for non-AI-painted faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    flat_fill_with_border,
    generate_procedural_regions,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_flat_fill_with_border_shape():
    grid = flat_fill_with_border(4, 4, "#FF0000", "#000000")
    assert len(grid) == 4
    assert all(len(row) == 4 for row in grid)


def test_flat_fill_with_border_colors():
    grid = flat_fill_with_border(4, 4, "#FF0000", "#000000")
    # Corners and edges are the shadow color
    assert grid[0][0] == "#000000"
    assert grid[0][3] == "#000000"
    assert grid[3][0] == "#000000"
    assert grid[3][3] == "#000000"
    # Interior is the main color
    assert grid[1][1] == "#FF0000"
    assert grid[2][2] == "#FF0000"


def test_flat_fill_with_border_narrow_width():
    """3px-wide grid (slim arm) still has a valid interior column."""
    grid = flat_fill_with_border(12, 3, "#0000FF", "#000011")
    assert grid[5][0] == "#000011"
    assert grid[5][1] == "#0000FF"
    assert grid[5][2] == "#000011"


def _colors(**overrides: str) -> dict[str, str]:
    base = {
        "shirt_main": "#111111",
        "shirt_shadow": "#101010",
        "arm_main": "#222222",
        "arm_shadow": "#202020",
        "pants_main": "#333333",
        "pants_shadow": "#303030",
        "shoe_color": "#444444",
    }
    base.update(overrides)
    return base


def test_generate_procedural_regions_covers_all_non_ai_faces():
    regions = get_all_regions("classic")
    expected_keys = {
        key
        for key, (group, _face) in PIXEL_KEY_MAP.items()
        if group in ("body", "right_arm", "left_arm", "right_leg", "left_leg")
        and key not in AI_GENERATED_KEYS
    }

    out = generate_procedural_regions(_colors(), "classic")

    assert set(out.keys()) == expected_keys
    for key, grid in out.items():
        group, face = PIXEL_KEY_MAP[key]
        rect = regions[group][face]
        assert len(grid) == rect.h
        assert all(len(row) == rect.w for row in grid)


def test_generate_procedural_regions_never_touches_ai_keys():
    out = generate_procedural_regions(_colors(), "classic")
    assert AI_GENERATED_KEYS.isdisjoint(out.keys())


def test_generate_procedural_regions_slim_arm_width():
    out = generate_procedural_regions(_colors(), "slim")
    assert len(out["right_arm_front"][0]) == 3
    assert len(out["left_arm_front"][0]) == 3


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    out = generate_procedural_regions(_colors(shoe_color="#ABCDEF"), "classic")
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_falls_back_to_defaults_on_missing_colors():
    """Missing/blank color fields shouldn't crash — defaults fill in."""
    out = generate_procedural_regions({}, "classic")
    assert len(out) > 0
    assert out["body_back"][1][1]  # some non-empty hex string
