"""Shading removal. A photo is material × light; a skin texture wants material."""

from __future__ import annotations

import numpy as np

from app.imaging.albedo import remove_shading
from app.imaging.color import hex_to_linear, linear_to_oklab

WHITE = hex_to_linear("#F0EDEA")


def _lightness(lin: np.ndarray) -> np.ndarray:
    return linear_to_oklab(np.asarray(lin, dtype=np.float32))[:, :, 0]


def _shaded_garment(width: int = 64) -> np.ndarray:
    """A flat garment with a smooth left-to-right illumination ramp."""
    flat = np.tile(WHITE.reshape(1, 1, 3), (width, width, 1))
    ramp = np.linspace(1.0, 0.45, width, dtype=np.float32)[None, :, None]
    return (flat * ramp).astype(np.float32)


def test_a_smooth_gradient_is_flattened():
    shaded = _shaded_garment()
    before = np.ptp(_lightness(shaded))
    after = np.ptp(_lightness(remove_shading(shaded, 1.0)))
    assert after < before / 2


def test_strength_zero_is_a_no_op():
    shaded = _shaded_garment()
    assert np.allclose(remove_shading(shaded, 0.0), shaded)


def test_strength_scales_the_effect():
    shaded = _shaded_garment()
    spans = [np.ptp(_lightness(remove_shading(shaded, s))) for s in (0.25, 0.6, 1.0)]
    assert spans == sorted(spans, reverse=True)


def test_a_hard_material_boundary_survives():
    """Only the low-frequency component is illumination; an edge is material."""
    img = np.tile(WHITE.reshape(1, 1, 3), (64, 64, 1))
    img[:, 32:] = hex_to_linear("#C03030")
    flattened = remove_shading(img, 1.0)

    left = flattened[32, 8]
    right = flattened[32, 56]
    assert np.linalg.norm(left - right) > 0.15


def test_chromaticity_is_left_alone():
    shaded = _shaded_garment()
    lab_before = linear_to_oklab(shaded)
    lab_after = linear_to_oklab(remove_shading(shaded, 1.0))
    assert np.allclose(lab_before[:, :, 1:], lab_after[:, :, 1:], atol=0.02)


def test_non_image_input_is_returned_unchanged():
    flat = np.zeros((8, 8), np.float32)
    assert remove_shading(flat, 1.0) is flat
