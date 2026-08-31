from __future__ import annotations

import numpy as np

from app.imaging.metrics import compare, ssim, upsample_nearest


def test_ssim_is_one_for_identical_images():
    rng = np.random.default_rng(0)
    x = rng.random((32, 32)).astype(np.float32)
    assert ssim(x, x) > 0.999


def test_ssim_is_low_for_unrelated_images():
    rng = np.random.default_rng(0)
    x = rng.random((32, 32)).astype(np.float32)
    assert ssim(x, np.full_like(x, 0.5)) < 0.2


def test_upsample_nearest_block_replicates():
    src = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], np.float32)
    up = upsample_nearest(src, 2, 4)
    assert up.shape == (2, 4, 3)
    assert np.allclose(up[0, 0], 0.0) and np.allclose(up[0, 3], 1.0)


def test_compare_reports_perfect_scores_on_a_flat_region():
    flat = np.full((16, 16, 3), 0.4, np.float32)
    scores = compare(flat, np.full((4, 4, 3), 0.4, np.float32))
    assert scores["ssim"] > 0.99
    assert scores["delta_e_mean"] < 1e-4


def test_detail_drops_when_structure_is_averaged_away():
    src = np.zeros((16, 16, 3), np.float32)
    src[:, :8] = 1.0
    averaged = np.full((1, 1, 3), 0.5, np.float32)
    preserved = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], np.float32)
    assert compare(src, averaged)["detail"] < compare(src, preserved)["detail"]
