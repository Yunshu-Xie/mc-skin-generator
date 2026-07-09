"""Tests for procedural — front-row propagation for non-AI-painted wrap faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    generate_procedural_regions,
    propagate_front_row,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_ai_generated_keys_includes_all_four_limb_fronts():
    assert AI_GENERATED_KEYS == {
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

    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "classic")

    assert set(out.keys()) == expected_keys
    for key, grid in out.items():
        group, face = PIXEL_KEY_MAP[key]
        rect = regions[group][face]
        assert len(grid) == rect.h
        assert all(len(row) == rect.w for row in grid)


def test_generate_procedural_regions_never_touches_ai_keys():
    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "classic")
    assert AI_GENERATED_KEYS.isdisjoint(out.keys())


def test_generate_procedural_regions_wrap_faces_match_front_rows():
    front = [[f"#{i:02X}{i:02X}{i:02X}"] * 8 for i in range(12)]  # row i -> shade #i,i,i
    out = generate_procedural_regions(
        _pixel_data_with_fronts(body_front=front), _colors(), "classic"
    )
    for row_idx, row in enumerate(front):
        row_color = row[len(row) // 2]
        assert out["body_back"][row_idx] == [row_color] * 8
        assert out["body_left"][row_idx] == [row_color] * 4
        assert out["body_right"][row_idx] == [row_color] * 4


def test_generate_procedural_regions_top_bottom_are_flat_named_color():
    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "classic")
    assert all(cell == "#3355AA" for row in out["body_top"] for cell in row)
    assert all(cell == "#3355AA" for row in out["body_bottom"] for cell in row)
    assert all(cell == "#C4A882" for row in out["right_arm_top"] for cell in row)
    assert all(cell == "#223355" for row in out["right_leg_top"] for cell in row)


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    out = generate_procedural_regions(
        _pixel_data_with_fronts(), _colors(shoe_color="#ABCDEF"), "classic"
    )
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_falls_back_to_flat_when_front_missing():
    """Old persisted skins (pre-migration) may lack the new mandatory front grids."""
    out = generate_procedural_regions({}, _colors(), "classic")
    assert all(cell == "#C4A882" for row in out["right_arm_back"] for cell in row)
    assert all(cell == "#223355" for row in out["left_leg_left"] for cell in row)


def test_generate_procedural_regions_respects_exclude_keys():
    out = generate_procedural_regions(
        _pixel_data_with_fronts(),
        _colors(),
        "classic",
        exclude_keys=frozenset({"right_arm_back"}),
    )
    assert "right_arm_back" not in out
    assert "right_arm_left" in out  # sibling face still procedural


def test_generate_procedural_regions_slim_arm_width():
    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "slim")
    assert len(out["right_arm_top"][0]) == 3
    assert len(out["left_arm_top"][0]) == 3


def test_generate_procedural_regions_falls_back_to_defaults_on_missing_colors():
    out = generate_procedural_regions(_pixel_data_with_fronts(), {}, "classic")
    assert len(out) > 0
    assert out["body_top"][0][0]  # some non-empty hex string
