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


def test_lightness_weight_groups_a_material_with_its_own_shadow():
    """Lit red + shaded red + blue, into two slots.

    With the full lightness metric the split happens along L, merging the dark
    red with the blue. Shrinking the L axis splits by material instead.
    """
    rng = np.random.default_rng(0)

    def material(hex_color: str, mul: float = 1.0, n: int = 300) -> np.ndarray:
        base = np.tile(hex_to_linear(hex_color), (n, 1)) * mul
        return (base + rng.normal(0, 0.006, (n, 3))).astype(np.float32)

    samples = np.concatenate([material("#C03030"), material("#C03030", 0.40), material("#2A3A80")])

    def hues(weight: float) -> list[float]:
        palette = build_palette(samples, 2, lightness_weight=weight, seed=0)
        lab = linear_to_oklab(palette.linear)
        return sorted(float(np.arctan2(b, a)) for _, a, b in lab)

    # One red-ish and one blue-ish center when lightness is downweighted; the
    # unweighted run produces two centers of muddled hue instead.
    weighted = hues(0.7)
    unweighted = hues(1.0)
    assert abs(weighted[0] - weighted[1]) > abs(unweighted[0] - unweighted[1])


def test_albedo_percentile_reports_the_lit_end_of_a_cluster():
    rng = np.random.default_rng(1)
    lit = np.tile(hex_to_linear("#F0EDEA"), (200, 1))
    shade = lit * 0.45
    samples = (np.concatenate([lit, shade]) + rng.normal(0, 0.004, (400, 3))).astype(np.float32)

    mean_based = build_palette(samples, 1, lightness_weight=0.3, seed=0)
    lit_based = build_palette(samples, 1, lightness_weight=0.3, albedo_percentile=85.0, seed=0)
    assert linear_to_oklab(lit_based.linear)[0, 0] > linear_to_oklab(mean_based.linear)[0, 0]


def test_assignment_uses_the_palette_own_metric():
    palette = build_palette(
        np.zeros((0, 3), np.float32),
        2,
        anchors={"a": hex_to_linear("#C03030"), "b": hex_to_linear("#303030")},
        lightness_weight=0.25,
    )
    # A very dark red: nearer the grey by lightness, nearer the red by hue.
    dark_red = (hex_to_linear("#C03030") * 0.25).reshape(1, 1, 3)
    assert assign(dark_red, palette).tolist() == [[0]]
