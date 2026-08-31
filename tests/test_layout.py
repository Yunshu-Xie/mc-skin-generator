from __future__ import annotations

from app.services.layout import (
    ANCHORS,
    REQUIRED_ROLES,
    Bbox,
    default_layout,
    parse_layout,
)


def test_parses_a_complete_response():
    layout, complete = parse_layout(
        {
            "description": "a person in a blue shirt",
            "boxes": {name: [0.1, 0.1, 0.9, 0.9] for name in ANCHORS},
            "roles": {"skin_tone": "#c4a882", "hair_color": "#3a2a1e"},
        }
    )
    assert complete
    assert set(layout.boxes) == set(ANCHORS)
    assert layout.roles["skin_tone"] == "#C4A882"  # normalized to upper case


def test_missing_face_box_is_incomplete():
    _, complete = parse_layout(
        {
            "boxes": {"torso": [0.1, 0.1, 0.9, 0.9]},
            "roles": dict.fromkeys(REQUIRED_ROLES, "#000000"),
        }
    )
    assert not complete


def test_degenerate_and_malformed_boxes_are_dropped():
    layout, _ = parse_layout(
        {
            "boxes": {
                "face": [0.5, 0.5, 0.505, 0.9],  # 0.5% wide — unusable
                "torso": "not a box",
                "legs": [0.1, 0.1, 0.9],  # wrong arity
            }
        }
    )
    assert layout.boxes == {}


def test_reversed_coordinates_are_repaired():
    layout, _ = parse_layout({"boxes": {"face": [0.9, 0.8, 0.2, 0.1]}})
    assert layout.boxes["face"] == Bbox(0.2, 0.1, 0.9, 0.8)


def test_invalid_hex_is_ignored_and_falls_back():
    layout, _ = parse_layout({"roles": {"skin_tone": "reddish", "hair_color": "#GGGGGG"}})
    assert layout.roles == {}
    assert layout.role("skin_tone") == "#C4A882"  # default


def test_missing_box_falls_back_to_the_default_framing():
    layout, _ = parse_layout({})
    assert layout.box("torso") == Bbox(0.22, 0.44, 0.78, 0.78)


def test_bbox_to_pixels_clamps_and_never_degenerates():
    assert Bbox(0.0, 0.0, 1.0, 1.0).to_pixels(64, 32) == (0, 0, 64, 32)
    assert Bbox(0.999, 0.999, 1.0, 1.0).to_pixels(10, 10) == (9, 9, 10, 10)


def test_default_layout_is_marked_as_a_fallback():
    layout = default_layout()
    assert layout.source == "fallback"
    assert set(layout.boxes) == set(ANCHORS)
