"""Color science. These are the tests that would have caught the original bug."""

from __future__ import annotations

import numpy as np

from app.imaging.color import (
    hex_to_linear,
    linear_to_hex,
    linear_to_oklab,
    linear_to_srgb,
    linear_to_u8,
    oklab_to_linear,
    srgb_to_linear,
    u8_to_linear,
)


def test_srgb_round_trip():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    assert np.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-5)


def test_srgb_anchors():
    assert srgb_to_linear(np.float32(0.0)) == 0.0
    assert np.isclose(srgb_to_linear(np.float32(1.0)), 1.0, atol=1e-6)
    # Mid-grey in sRGB is only ~21% of the light, not 50%. This is the whole
    # reason downscaling must not happen on encoded values.
    assert 0.20 < float(srgb_to_linear(np.float32(0.5))) < 0.22


def test_averaging_in_srgb_is_wrong():
    """Black + white: the correct mid-tone is sRGB ~188, not 128."""
    naive = np.uint8(round((0 + 255) / 2))
    correct = linear_to_u8(u8_to_linear(np.array([0.0, 255.0])).mean())
    assert int(naive) == 128
    assert 185 <= int(np.atleast_1d(correct)[0]) <= 190


def test_oklab_round_trip():
    rng = np.random.default_rng(0)
    x = rng.random((64, 3), dtype=np.float32)
    assert np.allclose(oklab_to_linear(linear_to_oklab(x)), x, atol=2e-3)


def test_oklab_lightness_is_monotonic():
    greys = np.stack([np.full(3, v) for v in np.linspace(0.02, 1.0, 12)]).astype(np.float32)
    lightness = linear_to_oklab(greys)[:, 0]
    assert np.all(np.diff(lightness) > 0)


def test_hex_round_trip():
    for value in ("#000000", "#FFFFFF", "#3B5998", "#C4A882"):
        assert linear_to_hex(hex_to_linear(value)) == value
