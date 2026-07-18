"""Tests for photo_render — deterministic crop/downsample/quantize pixel-art rendering."""

from PIL import Image

from app.services.photo_render import (
    build_head_front,
    consolidate_shadows,
    crop_region,
    dominant_hex,
    downsample_dominant,
    extract_face_colors,
    quantize_shared,
)


def _distinct(out: dict) -> set:
    return {c for grid in out.values() for row in grid for c in row}


def test_consolidate_shadows_collapses_shaded_materials_to_intrinsic_colors():
    grids = {
        "a": [[(220, 60, 60), (150, 40, 40)]],  # lit-red + shadow-red (same hue)
        "b": [[(60, 60, 220), (40, 40, 150)]],  # lit-blue + shadow-blue
    }
    out = consolidate_shadows(grids)
    distinct = _distinct(out)
    assert len(distinct) == 2  # one red material, one blue material
    assert (150, 40, 40) not in distinct  # dark shadow-red collapsed away
    assert (40, 40, 150) not in distinct  # dark shadow-blue collapsed away


def test_consolidate_shadows_keeps_black_gray_white_distinct():
    # Grayscale-safety: near-grays must NOT collapse across value bands.
    grids = {"g": [[(20, 20, 20), (128, 128, 128), (235, 235, 235)]]}
    out = consolidate_shadows(grids)
    assert len(_distinct(out)) == 3


def test_consolidate_shadows_collapses_single_material_gradient_to_one():
    grids = {"g": [[(r, r // 4, r // 4) for r in range(120, 240, 10)]]}
    out = consolidate_shadows(grids)
    assert len(_distinct(out)) == 1


def test_consolidate_shadows_representative_is_bright_not_dark():
    grids = {"g": [[(210, 55, 55), (110, 28, 28)]]}  # bright + dark red
    out = consolidate_shadows(grids)
    (rep,) = _distinct(out)
    # highlight tone: brighter than the dark shadow input's max channel (110)
    assert max(rep) > 110


def test_consolidate_shadows_preserves_grid_shapes():
    grids = {
        "a": [[(10, 10, 10)] * 8 for _ in range(12)],
        "b": [[(200, 50, 50)] * 4 for _ in range(12)],
    }
    out = consolidate_shadows(grids)
    assert len(out["a"]) == 12 and len(out["a"][0]) == 8
    assert len(out["b"]) == 12 and len(out["b"][0]) == 4


def test_consolidate_shadows_empty_is_safe():
    assert consolidate_shadows({}) == {}
    assert consolidate_shadows({"a": []}) == {"a": []}


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


def test_extract_face_colors_hair_beats_flat_background_texture():
    """A textured hair region (many near values) plus a flat background strip
    (one exact value) must extract the hair color, not the background — the raw
    most-common-exact-pixel bug that made blonde hair render near-white."""
    import random

    img = Image.new("RGB", (80, 80), (210, 170, 120))  # skin base
    rng = random.Random(0)
    for y in range(0, 20):  # hair band (rows 0-1)
        for x in range(80):
            if x < 12:  # flat near-white background strip (minority area, one exact color)
                img.putpixel((x, y), (245, 247, 249))
            else:  # textured golden hair (many near values)
                d = rng.randint(-6, 6)
                img.putpixel((x, y), (220 + d, 150 + d, 60 + d))

    hair = extract_face_colors(img)["hair_color"]
    r, b = int(hair[1:3], 16), int(hair[5:7], 16)
    assert r > b + 30, f"hair {hair} read as neutral/background, not warm golden"


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


def test_build_head_front_has_two_symmetric_eyes():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    for shape in ("narrow", "round"):
        grid = build_head_front(colors, eye_shape=shape)
        assert grid[0] == ["#28140A"] * 8 and grid[1] == ["#28140A"] * 8  # both hair rows
        for eye_row in (2, 3):  # eyes span rows 2-3
            assert "#141414" in grid[eye_row][0:4], f"{shape} row{eye_row}: no eye left half"
            assert "#141414" in grid[eye_row][4:8], f"{shape} row{eye_row}: no eye right half"


def test_build_head_front_mouth_is_single_row_not_rectangle():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    grid = build_head_front(colors, eye_shape="narrow", mouth_width="small")
    assert "#B43C3C" in grid[5]  # mouth on row 5
    assert "#B43C3C" not in grid[6]  # not a 2-row rectangle
    assert grid[6] == ["#C8A078"] * 8  # row 6 is skin/chin


def test_build_head_front_mouth_width():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    small = build_head_front(colors, eye_shape="narrow", mouth_width="small")
    wide = build_head_front(colors, eye_shape="narrow", mouth_width="wide")
    assert sum(c == "#B43C3C" for c in small[5]) == 2
    assert sum(c == "#B43C3C" for c in wide[5]) == 4
