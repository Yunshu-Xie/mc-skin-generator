from __future__ import annotations

import numpy as np
import pytest

from app.imaging.downscale import downscale, fit_crop, gaussian_blur, unsharp


def test_fit_crop_matches_target_aspect():
    wide = np.zeros((100, 400, 3), np.float32)
    cropped = fit_crop(wide, 8, 8)
    assert cropped.shape[0] == cropped.shape[1] == 100

    tall = np.zeros((400, 100, 3), np.float32)
    cropped = fit_crop(tall, 12, 8)
    assert cropped.shape[1] == 100
    assert abs(cropped.shape[0] / cropped.shape[1] - 12 / 8) < 0.02


@pytest.mark.parametrize("method", ["box", "dpid", "dominant"])
def test_output_shape_and_flat_input(method):
    flat = np.full((64, 48, 3), 0.37, np.float32)
    out = downscale(flat, 8, 6, method=method)
    assert out.shape == (8, 6, 3)
    assert np.allclose(out, 0.37, atol=1e-5)


def test_box_is_an_exact_area_average():
    src = np.zeros((4, 4, 3), np.float32)
    src[:2, :2] = 1.0
    out = downscale(src, 2, 2, method="box")
    assert np.isclose(out[0, 0, 0], 1.0)
    assert np.isclose(out[1, 1, 0], 0.0)


def test_dpid_keeps_an_outlier_that_box_averages_away():
    """One dark pixel in a bright tile: box dilutes it, DPID pulls toward it."""
    src = np.full((8, 8, 3), 0.9, np.float32)
    src[3:5, 3:5] = 0.05
    box = downscale(src, 1, 1, method="box")[0, 0, 0]
    dpid = downscale(src, 1, 1, method="dpid", dpid_lambda=1.4)[0, 0, 0]
    assert dpid < box


def test_dominant_returns_the_modal_color():
    src = np.full((8, 8, 3), 0.8, np.float32)
    src[:2, :2] = 0.1  # a minority region
    out = downscale(src, 1, 1, method="dominant")[0, 0, 0]
    assert out > 0.7


def test_unsharp_increases_local_contrast():
    rng = np.random.default_rng(1)
    src = rng.random((32, 32, 3)).astype(np.float32)
    assert unsharp(src, 0.8).std() > src.std()
    assert np.allclose(unsharp(src, 0.0), src)


def test_gaussian_blur_reduces_variance():
    rng = np.random.default_rng(2)
    src = rng.random((32, 32, 3)).astype(np.float32)
    assert gaussian_blur(src, 2.0).std() < src.std()


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        downscale(np.zeros((8, 8), np.float32), 4, 4)
