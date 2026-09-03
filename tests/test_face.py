"""The drawn head. Structure is asserted here; color comes from the palette."""

from __future__ import annotations

import numpy as np

from app.imaging.color import hex_to_linear, linear_to_oklab
from app.services.face import (
    EYE,
    HAIR,
    MOUTH,
    SKIN,
    build_template,
    eye_row_from_box,
    render_face,
)

SKIN_C = hex_to_linear("#C4A882")
HAIR_C = hex_to_linear("#3A2A1E")
EYE_C = hex_to_linear("#3B2B20")


def _lightness(grid: np.ndarray) -> np.ndarray:
    return linear_to_oklab(np.asarray(grid, dtype=np.float32))[..., 0]


def test_template_is_eight_by_eight_for_every_style():
    for style in ("short", "long", "fringe", "bald", "hat", "helmet", "nonsense"):
        grid = build_template(style)
        assert len(grid) == 8
        assert all(len(row) == 8 for row in grid)


def test_hair_rows_differ_by_style():
    assert build_template("short")[2][3] == SKIN
    assert build_template("fringe")[2][3] == HAIR
    assert HAIR not in build_template("bald")[0]
    assert build_template("long")[6][0] == HAIR  # hair down the sides
    assert build_template("short")[6][0] == SKIN


def test_every_template_has_exactly_two_eyes_and_a_mouth():
    for style in ("short", "long", "fringe", "bald", "hat"):
        flat = [c for row in build_template(style) for c in row]
        assert flat.count(EYE) == 2
        assert flat.count(MOUTH) == 2


def test_eyes_are_never_hidden_under_the_hair():
    for style, eye_row in (("hat", 0), ("fringe", 1), ("long", 2)):
        grid = build_template(style, eye_row)
        rows = {r for r, row in enumerate(grid) for c in row if c == EYE}
        hair_rows = {r for r, row in enumerate(grid) if all(c == HAIR for c in row)}
        assert rows and not (rows & hair_rows)


def test_eye_row_maps_from_the_reported_band():
    # eye band in the upper third of the head → an upper row
    assert eye_row_from_box(0.0, 0.8, 0.20, 0.30) < 4
    # band low in the head → a lower row
    assert eye_row_from_box(0.0, 0.8, 0.55, 0.65) > 4
    assert eye_row_from_box(0.5, 0.5, 0.1, 0.2) == 3  # degenerate box


def test_eyes_are_the_darkest_thing_in_the_face():
    """The one local extreme that makes an 8x8 face readable."""
    face = render_face("short", 3, SKIN_C, HAIR_C, EYE_C)
    lightness = _lightness(face)
    template = build_template("short", 3)
    eyes = [lightness[r][c] for r in range(8) for c in range(8) if template[r][c] == EYE]
    others = [
        lightness[r][c] for r in range(8) for c in range(8) if template[r][c] != EYE
    ]
    assert max(eyes) < min(others)


def test_regions_are_flat_without_modulation():
    face = render_face("short", 3, SKIN_C, HAIR_C, EYE_C, modulation=0.0)
    assert np.allclose(face[0, 0], face[0, 7])  # the whole hair row is one color
    assert np.allclose(face[7, 0], face[7, 7])


def test_photo_modulation_adds_variation_but_not_structure():
    rng = np.random.default_rng(0)
    photo = rng.random((8, 8, 3)).astype(np.float32)

    flat = render_face("short", 3, SKIN_C, HAIR_C, EYE_C, photo=photo, modulation=0.0)
    lit = render_face("short", 3, SKIN_C, HAIR_C, EYE_C, photo=photo, modulation=0.8)

    assert _lightness(lit)[0].std() > _lightness(flat)[0].std()
    # eyes stay exempt: they must not be lightened back toward the photo
    template = build_template("short", 3)
    for r in range(8):
        for c in range(8):
            if template[r][c] == EYE:
                assert np.allclose(flat[r, c], lit[r, c])


def test_a_mismatched_photo_grid_is_ignored():
    face = render_face(
        "short", 3, SKIN_C, HAIR_C, EYE_C, photo=np.zeros((4, 4, 3), np.float32)
    )
    assert face.shape == (8, 8, 3)
