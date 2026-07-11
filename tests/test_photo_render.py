"""Tests for photo_render — deterministic crop/downsample/quantize pixel-art rendering."""

from PIL import Image

from app.services.photo_render import (
    build_head_front,
    crop_region,
    dominant_hex,
    downsample_dominant,
    extract_face_colors,
    quantize_shared,
)


def _solid(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_crop_region_converts_fractional_bbox_to_pixels():
    img = _solid(100, 200, (255, 0, 0))
    crop = crop_region(img, (0.1, 0.2, 0.5, 0.6))
    assert crop is not None
    assert crop.size == (40, 80)  # (0.5-0.1)*100, (0.6-0.2)*200


def test_crop_region_clamps_out_of_bounds_bbox():
    img = _solid(100, 100, (0, 255, 0))
    crop = crop_region(img, (-0.5, -0.5, 1.5, 1.5))
    assert crop is not None
    assert crop.size == (100, 100)


def test_crop_region_returns_none_for_degenerate_bbox():
    img = _solid(100, 100, (0, 0, 255))
    assert crop_region(img, (0.5, 0.5, 0.5, 0.9)) is None  # zero width
    assert crop_region(img, (0.2, 0.5, 0.8, 0.5)) is None  # zero height
    assert crop_region(img, (0.9, 0.5, 0.2, 0.9)) is None  # x1 < x0


def test_downsample_dominant_picks_majority_color_per_cell():
    # Left half red, right half blue, downsample to 1x2 -> one red cell, one blue cell
    img = Image.new("RGB", (10, 4), (255, 0, 0))
    for x in range(5, 10):
        for y in range(4):
            img.putpixel((x, y), (0, 0, 255))

    out = downsample_dominant(img, target_h=1, target_w=2)

    assert out[0][0] == (255, 0, 0)
    assert out[0][1] == (0, 0, 255)


def test_downsample_dominant_output_shape():
    img = _solid(20, 30, (10, 20, 30))
    out = downsample_dominant(img, target_h=12, target_w=4)
    assert len(out) == 12
    assert all(len(row) == 4 for row in out)


def test_quantize_shared_produces_consistent_indices_and_capped_palette():
    grids = {
        "a": [[(255, 0, 0), (255, 0, 0)]],
        "b": [[(0, 0, 255)]],
    }
    palette, indices = quantize_shared(grids, max_colors=24)

    assert len(palette) <= 24
    assert len(indices["a"]) == 1 and len(indices["a"][0]) == 2
    assert len(indices["b"]) == 1 and len(indices["b"][0]) == 1
    # Same input color -> same index within a grid
    assert indices["a"][0][0] == indices["a"][0][1]
    # Different colors -> different indices
    assert indices["a"][0][0] != indices["b"][0][0]
    for hexval in palette:
        assert hexval.startswith("#") and len(hexval) == 7


def test_quantize_shared_preserves_grid_shapes():
    grids = {
        "wide": [[(1, 1, 1)] * 8 for _ in range(12)],
        "narrow": [[(2, 2, 2)] * 4 for _ in range(12)],
    }
    _palette, indices = quantize_shared(grids)
    assert len(indices["wide"]) == 12 and len(indices["wide"][0]) == 8
    assert len(indices["narrow"]) == 12 and len(indices["narrow"][0]) == 4


def _face_image(skin=(200, 160, 120), hair=(40, 20, 10), eye=(20, 20, 20), mouth=(180, 60, 60)):
    """8-row-band face image: rows 0-1 hair, rows 2-3 mostly skin with an eye dot,
    row 4 skin, rows 5-6 mostly skin with a mouth dot, row 7 skin (scaled up 10x)."""
    img = Image.new("RGB", (80, 80), skin)
    for y in range(0, 20):
        for x in range(80):
            img.putpixel((x, y), hair)
    for y in range(20, 40):
        for x in range(30, 34):
            img.putpixel((x, y), eye)
    for y in range(50, 70):
        for x in range(30, 50):
            img.putpixel((x, y), mouth)
    return img


def test_extract_face_colors_measures_all_four_bands():
    colors = extract_face_colors(_face_image())
    assert colors["skin_tone"] == "#C8A078"
    assert colors["hair_color"] == "#28140A"
    assert colors["eye_color"] == "#141414"
    assert colors["mouth_color"] == "#B43C3C"


def test_dominant_hex_returns_most_common_color():
    grid = [["#111111", "#111111", "#222222"], ["#111111", "#333333", "#111111"]]
    assert dominant_hex(grid) == "#111111"


def test_build_head_front_shape_and_bands():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    grid = build_head_front(colors, eye_shape="round")

    assert len(grid) == 8 and all(len(row) == 8 for row in grid)
    assert grid[0] == ["#28140A"] * 8  # hair row
    assert grid[7] == ["#C8A078"] * 8  # neck row is skin
    assert "#141414" in grid[2]  # eye color present in eye row
    assert "#B43C3C" in grid[5]  # mouth color present in mouth row


def test_build_head_front_eye_width_depends_on_eye_shape():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    narrow = build_head_front(colors, eye_shape="narrow")
    round_ = build_head_front(colors, eye_shape="round")

    narrow_eye_count = sum(cell == "#141414" for row in narrow[2:4] for cell in row)
    round_eye_count = sum(cell == "#141414" for row in round_[2:4] for cell in row)
    assert round_eye_count > narrow_eye_count
