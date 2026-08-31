from __future__ import annotations

import numpy as np

from app.imaging.color import hex_to_linear, linear_to_oklab
from app.services.shading import ORIENTATION_LIGHT, shaded_face, shift_lightness, solid_face


def _lightness(lin: np.ndarray) -> float:
    return float(linear_to_oklab(np.asarray(lin).reshape(-1, 3))[:, 0].mean())


def test_shift_lightness_moves_only_lightness():
    base = hex_to_linear("#3B5998")
    lighter = shift_lightness(base, 0.1)
    assert _lightness(lighter) > _lightness(base)
    lab_a, lab_b = linear_to_oklab(base.reshape(1, 3)), linear_to_oklab(lighter.reshape(1, 3))
    assert np.allclose(lab_a[0, 1:], lab_b[0, 1:], atol=0.02)  # hue/chroma preserved


def test_faces_are_lit_by_orientation():
    base = hex_to_linear("#3B5998")
    order = ["top", "front", "left", "back", "bottom"]
    values = [_lightness(shaded_face(4, 4, base, o)) for o in order]
    assert values == sorted(values, reverse=True)


def test_no_hard_border_is_drawn():
    """The old flat_fill_with_border made every limb read as an outlined box."""
    face = shaded_face(12, 4, hex_to_linear("#3B5998"), "front")
    assert np.allclose(face[5, 0], face[5, 2], atol=1e-4)  # no darker edge column


def test_vertical_gradient_is_soft_and_downward():
    face = shaded_face(12, 4, hex_to_linear("#C4A882"), "front")
    top, bottom = _lightness(face[0]), _lightness(face[-1])
    assert top > bottom
    assert (top - bottom) < 0.08  # a cue, not a stripe


def test_solid_face_shape():
    assert solid_face(3, 5, hex_to_linear("#101010")).shape == (3, 5, 3)


def test_orientation_table_covers_every_face():
    assert set(ORIENTATION_LIGHT) == {"top", "bottom", "left", "right", "front", "back"}
