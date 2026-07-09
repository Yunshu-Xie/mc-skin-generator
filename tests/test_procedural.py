"""Tests for procedural — shaded flat-fill generation for non-AI-painted faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    generate_procedural_regions,
    shade_face,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_shade_face_shape():
    grid = shade_face(4, 4, "#FF0000", "front")
    assert len(grid) == 4
    assert all(len(row) == 4 for row in grid)


def test_shade_face_edge_darker_than_interior():
    import colorsys

    grid = shade_face(6, 6, "#8080A0", "front")
    edge_l = colorsys.rgb_to_hls(
        *[int(grid[0][0].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    interior_l = colorsys.rgb_to_hls(
        *[int(grid[2][2].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    assert edge_l < interior_l


def test_shade_face_top_lighter_than_bottom():
    import colorsys

    top = shade_face(4, 4, "#606060", "top")
    bottom = shade_face(4, 4, "#606060", "bottom")
    top_l = colorsys.rgb_to_hls(
        *[int(top[1][1].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    bottom_l = colorsys.rgb_to_hls(
        *[int(bottom[1][1].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    assert top_l > bottom_l


def test_shade_face_narrow_width():
    """3px-wide grid (slim arm) still has a valid interior column."""
    grid = shade_face(12, 3, "#0000FF", "front")
    assert len(grid[5]) == 3


def _colors(**overrides: str) -> dict[str, str]:
    base = {
        "shirt_main": "#3355AA",
        "arm_main": "#C4A882",
        "pants_main": "#223355",
        "shoe_color": "#2B2B2B",
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


def test_generate_procedural_regions_respects_exclude_keys():
    out = generate_procedural_regions(
        _colors(), "classic", exclude_keys=frozenset({"right_arm_front"})
    )
    assert "right_arm_front" not in out
    assert "right_arm_back" in out  # sibling face still procedural


def test_generate_procedural_regions_slim_arm_width():
    out = generate_procedural_regions(_colors(), "slim")
    assert len(out["right_arm_front"][0]) == 3
    assert len(out["left_arm_front"][0]) == 3


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    out = generate_procedural_regions(_colors(shoe_color="#ABCDEF"), "classic")
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_falls_back_to_defaults_on_missing_colors():
    out = generate_procedural_regions({}, "classic")
    assert len(out) > 0
    assert out["body_back"][1][1]  # some non-empty hex string
