"""Garment templates. Structure is asserted here; color comes from the palette."""

from __future__ import annotations

import numpy as np

from app.imaging.color import hex_to_linear
from app.services.clothing import (
    ACCENT,
    INNER,
    PANTS,
    SHOE,
    SHOE_ROWS,
    SKIN,
    TOP,
    back_of,
    build_arm,
    build_leg,
    build_torso,
    colors_from_palette,
    render_clothing,
    solid,
)

COLORS = colors_from_palette(
    shirt=hex_to_linear("#232A3D"),
    inner=hex_to_linear("#F2F2F0"),
    accent=hex_to_linear("#B32433"),
    skin=hex_to_linear("#E8B48C"),
    pants=hex_to_linear("#1A1A3E"),
    shoe=hex_to_linear("#171717"),
)


def test_templates_match_the_requested_shape():
    assert len(build_torso("suit", 12, 8)) == 12
    assert all(len(row) == 8 for row in build_torso("suit", 12, 8))
    assert all(len(row) == 3 for row in build_arm("suit", 12, 3))  # slim arms


def test_a_suit_has_an_inner_layer_and_a_tie():
    torso = build_torso("suit", 12, 8)
    flat = [c for row in torso for c in row]
    assert INNER in flat and ACCENT in flat
    # the tie sits between the two halves of the inner layer
    accent_cols = {c for row in torso for c, code in enumerate(row) if code == ACCENT}
    assert accent_cols == {3, 4}


def test_a_plain_top_has_no_detail():
    assert {c for row in build_torso("plain", 12, 8) for c in row} == {TOP}


def test_a_tshirt_has_short_sleeves_and_a_suit_has_long_ones():
    short = build_arm("tshirt", 12, 4)
    long = build_arm("suit", 12, 4)
    assert short.count([SKIN] * 4) > long.count([SKIN] * 4)
    assert long[-2] == [INNER] * 4  # a cuff
    assert long[-1] == [SKIN] * 4  # the hand


def test_a_dress_leaves_the_arms_bare():
    assert {c for row in build_arm("dress", 12, 4) for c in row} == {SKIN}


def test_shoes_reach_every_side_of_the_leg():
    """The old code painted only leg_bottom — the one face nobody ever sees."""
    leg = build_leg("pants", 12, 4)
    for row in leg[-SHOE_ROWS:]:
        assert set(row) == {SHOE}
    assert PANTS in {c for row in leg for c in row}


def test_shorts_show_more_leg_than_trousers():
    def skin_cells(style: str) -> int:
        return sum(row.count(SKIN) for row in build_leg(style, 12, 4))

    assert skin_cells("skirt") > skin_cells("shorts") > skin_cells("pants")


def test_the_back_of_a_garment_keeps_the_boundaries_but_drops_the_detail():
    front = build_torso("suit", 12, 8)
    back = back_of(front)
    flat = [c for row in back for c in row]
    assert ACCENT not in flat and INNER not in flat
    assert len(back) == len(front) and len(back[0]) == len(front[0])


def test_arm_and_leg_boundaries_line_up_across_faces():
    """A seam shows if the sleeve ends on a different row front and side."""
    front = build_arm("suit", 12, 3)
    side = build_arm("suit", 12, 4)
    assert [row[0] for row in front] == [row[0] for row in side]


def test_rendering_paints_every_region_from_the_palette():
    template = build_torso("suit", 12, 8)
    painted = render_clothing(template, COLORS, modulation=0.0)
    assert painted.shape == (12, 8, 3)
    assert np.allclose(painted[0, 0], COLORS[TOP])
    assert np.allclose(painted[2, 3], COLORS[ACCENT])


def test_the_accent_is_exempt_from_photo_modulation():
    template = build_torso("suit", 12, 8)
    rng = np.random.default_rng(0)
    photo = rng.random((12, 8, 3)).astype(np.float32)

    flat = render_clothing(template, COLORS, photo, modulation=0.0)
    lit = render_clothing(template, COLORS, photo, modulation=0.8)
    for r in range(12):
        for c in range(8):
            if template[r][c] == ACCENT:
                assert np.allclose(flat[r, c], lit[r, c])
    assert not np.allclose(flat, lit)  # everything else did move


def test_solid_is_a_single_region():
    assert solid(4, 3, TOP) == [[TOP] * 3] * 4
