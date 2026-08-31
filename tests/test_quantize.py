from __future__ import annotations

import numpy as np

from app.imaging.color import hex_to_linear, linear_to_oklab
from app.imaging.quantize import Palette, assign, build_palette, kmeans_oklab


def test_anchors_occupy_the_front_slots_unchanged():
    anchors = {"skin_tone": hex_to_linear("#C4A882"), "hair_color": hex_to_linear("#3A2A1E")}
    rng = np.random.default_rng(0)
    palette = build_palette(rng.random((500, 3)).astype(np.float32), 8, anchors=anchors)

    assert palette.size == 8
    assert palette.roles == {"skin_tone": 0, "hair_color": 1}
    assert palette.hex_list()[0] == "#C4A882"
    assert palette.hex_list()[1] == "#3A2A1E"


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(3)
    a = rng.normal(0.2, 0.01, (200, 3))
    b = rng.normal(0.8, 0.01, (200, 3))
    lab = linear_to_oklab(np.concatenate([a, b]).astype(np.float32))
    centers = kmeans_oklab(lab, 2, seed=0)
    assert centers.shape == (2, 3)
    assert abs(centers[0, 0] - centers[1, 0]) > 0.2


def test_weights_pull_the_palette_toward_important_samples():
    rare = np.tile(np.array([[0.9, 0.1, 0.1]], np.float32), (10, 1))
    common = np.tile(np.array([[0.2, 0.2, 0.2]], np.float32), (400, 1))
    samples = np.concatenate([rare, common])

    unweighted = build_palette(samples, 2, seed=0)
    weights = np.concatenate([np.full(10, 200.0, np.float32), np.ones(400, np.float32)])
    weighted = build_palette(samples, 2, weights=weights, seed=0)

    def redness(p: Palette) -> float:
        return float(max(c[0] - c[1] for c in p.linear))

    assert redness(weighted) >= redness(unweighted)


def test_assign_picks_the_nearest_palette_entry():
    palette = build_palette(
        np.zeros((0, 3), np.float32),
        2,
        anchors={"a": hex_to_linear("#000000"), "b": hex_to_linear("#FFFFFF")},
    )
    grid = np.array([[[0.02, 0.02, 0.02], [0.95, 0.95, 0.95]]], np.float32)
    assert assign(grid, palette).tolist() == [[0, 1]]


def test_role_lookup_and_fallback():
    palette = build_palette(
        np.zeros((0, 3), np.float32), 1, anchors={"skin_tone": hex_to_linear("#C4A882")}
    )
    assert palette.index_of("skin_tone") == 0
    assert palette.index_of("nope") is None
    fallback = np.array([0.5, 0.5, 0.5], np.float32)
    assert np.allclose(palette.color_of("nope", fallback), fallback)
