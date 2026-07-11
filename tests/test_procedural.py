"""Tests for procedural — front-row propagation and flat-fill for non-photo-derived faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    generate_procedural_regions,
    propagate_front_row,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_ai_generated_keys_is_head_front_plus_five_photo_derived_fronts():
    assert AI_GENERATED_KEYS == {
        "head_front",
        "body_front",
        "right_arm_front",
        "left_arm_front",
        "right_leg_front",
        "left_leg_front",
    }


def test_propagate_front_row_uses_middle_column_per_row():
    front = [
        ["#111111", "#222222", "#333333"],
        ["#AAAAAA", "#BBBBBB", "#CCCCCC"],
    ]
    out = propagate_front_row(front, target_width=4)
    assert out[0] == ["#222222"] * 4
    assert out[1] == ["#BBBBBB"] * 4


def test_propagate_front_row_preserves_height():
    front = [["#111111"], ["#222222"], ["#333333"]]
    out = propagate_front_row(front, target_width=2)
    assert len(out) == 3


def test_propagate_front_row_target_width_independent_of_front_width():
    front = [["#111111", "#222222", "#333333", "#444444"]]  # 4 cols wide
    out = propagate_front_row(front, target_width=8)
    assert len(out[0]) == 8


def _pixel_data_with_fronts(**overrides: list[list[str]]) -> dict[str, list[list[str]]]:
    data = {
        "head_front": [["#C4A882"] * 8 for _ in range(8)],
        "body_front": [["#3355AA"] * 8 for _ in range(12)],
        "right_arm_front": [["#C4A882"] * 4 for _ in range(12)],
        "left_arm_front": [["#C4A882"] * 4 for _ in range(12)],
        "right_leg_front": [["#223355"] * 4 for _ in range(12)],
        "left_leg_front": [["#223355"] * 4 for _ in range(12)],
    }
    data.update(overrides)
    return data


def _colors(**overrides: str) -> dict[str, str]:
    base = {
        "shirt_main": "#3355AA",
        "arm_main": "#C4A882",
        "pants_main": "#223355",
        "shoe_color": "#2B2B2B",
        "head_fill_color": "#5B3A1A",
    }
    base.update(overrides)
    return base


def test_generate_procedural_regions_covers_all_non_provided_faces():
    """With every front provided (and excluded), procedural fill covers exactly
    the remaining 30 of the 36 base regions (36 total - 6 excluded fronts)."""
    pixel_data = _pixel_data_with_fronts()
    regions = get_all_regions("classic")
    all_base_keys = {
        key
        for key, (group, _face) in PIXEL_KEY_MAP.items()
        if group in ("head", "body", "right_arm", "left_arm", "right_leg", "left_leg")
    }
    expected_keys = all_base_keys - set(pixel_data.keys())

    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )

    assert set(out.keys()) == expected_keys
    for key, grid in out.items():
        group, face = PIXEL_KEY_MAP[key]
        rect = regions[group][face]
        assert len(grid) == rect.h
        assert all(len(row) == rect.w for row in grid)


def test_generate_procedural_regions_never_overwrites_excluded_faces():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    assert set(pixel_data.keys()).isdisjoint(out.keys())


def test_generate_procedural_regions_wrap_faces_match_front_rows():
    front = [[f"#{i:02X}{i:02X}{i:02X}"] * 8 for i in range(12)]  # row i -> shade #i,i,i
    pixel_data = _pixel_data_with_fronts(body_front=front)
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    for row_idx, row in enumerate(front):
        row_color = row[len(row) // 2]
        assert out["body_back"][row_idx] == [row_color] * 8
        assert out["body_left"][row_idx] == [row_color] * 4
        assert out["body_right"][row_idx] == [row_color] * 4


def test_generate_procedural_regions_top_bottom_are_flat_named_color():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    assert all(cell == "#3355AA" for row in out["body_top"] for cell in row)
    assert all(cell == "#3355AA" for row in out["body_bottom"] for cell in row)
    assert all(cell == "#C4A882" for row in out["right_arm_top"] for cell in row)
    assert all(cell == "#223355" for row in out["right_leg_top"] for cell in row)


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data,
        _colors(shoe_color="#ABCDEF"),
        "classic",
        exclude_keys=frozenset(pixel_data.keys()),
    )
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_covers_head_wrap_faces():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    for face in ("top", "back", "left", "right", "bottom"):
        assert f"head_{face}" in out


def test_generate_procedural_regions_head_faces_are_flat_head_fill_color():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data,
        _colors(head_fill_color="#ABCDEF"),
        "classic",
        exclude_keys=frozenset(pixel_data.keys()),
    )
    for face in ("top", "back", "left", "right", "bottom"):
        grid = out[f"head_{face}"]
        assert all(cell == "#ABCDEF" for row in grid for cell in row)


def test_generate_procedural_regions_missing_front_gets_flat_fill_not_omitted():
    """A body part not visible in the photo (no front data, and thus not in
    exclude_keys) must still get its front face flat-filled -- not left
    entirely absent, which would render as a transparent hole."""
    pixel_data = _pixel_data_with_fronts()
    del pixel_data["right_leg_front"]  # simulates: legs weren't visible in the photo

    out = generate_procedural_regions(
        pixel_data,
        _colors(pants_main="#654321"),
        "classic",
        exclude_keys=frozenset(pixel_data.keys()),
    )

    assert "right_leg_front" in out
    assert all(cell == "#654321" for row in out["right_leg_front"] for cell in row)
    # Its wrap faces fall back to the same flat color too, since there's no
    # real front data to propagate from.
    assert all(cell == "#654321" for row in out["right_leg_back"] for cell in row)


def test_generate_procedural_regions_slim_arm_width():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "slim", exclude_keys=frozenset(pixel_data.keys())
    )
    assert len(out["right_arm_top"][0]) == 3
    assert len(out["left_arm_top"][0]) == 3


def test_generate_procedural_regions_falls_back_to_defaults_when_nothing_provided():
    """Nothing provided at all (e.g. a totally failed generation) -> every
    face, including all 6 fronts, gets a flat default-color fill rather
    than a transparent skin."""
    out = generate_procedural_regions({}, {}, "classic")
    assert len(out) == 36
    assert out["body_front"][0][0]  # non-empty hex string, DEFAULT_COLORS fallback applied
    assert out["head_front"][0][0]
