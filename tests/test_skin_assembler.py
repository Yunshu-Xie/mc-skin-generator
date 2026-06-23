"""Tests for skin_assembler — PNG generation from pixel data."""

from PIL import Image

from app.services.skin_assembler import assemble_skin, hex_to_rgba, validate_pixel_grid


def test_hex_to_rgba_6_chars():
    assert hex_to_rgba("#FF0000") == (255, 0, 0, 255)
    assert hex_to_rgba("#00FF00") == (0, 255, 0, 255)
    assert hex_to_rgba("#000000") == (0, 0, 0, 255)


def test_hex_to_rgba_8_chars():
    assert hex_to_rgba("#FF000080") == (255, 0, 0, 128)
    assert hex_to_rgba("#00000000") == (0, 0, 0, 0)


def test_validate_pixel_grid():
    grid = [["#FF0000"] * 8 for _ in range(8)]
    assert validate_pixel_grid(grid, 8, 8) is True
    assert validate_pixel_grid(grid, 8, 4) is False
    assert validate_pixel_grid(grid, 4, 8) is False


def _make_solid_grid(rows: int, cols: int, color: str = "#FF0000") -> list[list[str]]:
    return [[color] * cols for _ in range(rows)]


def test_assemble_skin_produces_64x64():
    """Assembling with minimal data should produce a 64×64 RGBA image."""
    pixel_data = {
        "head_front": _make_solid_grid(8, 8, "#C4A882"),
        "head_back": _make_solid_grid(8, 8, "#5B3A1A"),
        "head_top": _make_solid_grid(8, 8, "#5B3A1A"),
        "head_bottom": _make_solid_grid(8, 8, "#C4A882"),
        "head_left": _make_solid_grid(8, 8, "#C4A882"),
        "head_right": _make_solid_grid(8, 8, "#C4A882"),
    }

    img = assemble_skin(pixel_data, "classic")
    assert img.size == (64, 64)
    assert img.mode == "RGBA"


def test_assemble_skin_places_pixels_correctly():
    """Head front at (8,8) should have the color we painted."""
    pixel_data = {
        "head_front": _make_solid_grid(8, 8, "#AABBCC"),
    }
    img = assemble_skin(pixel_data, "classic")
    # Head front starts at (8, 8)
    pixel = img.getpixel((8, 8))
    assert pixel == (0xAA, 0xBB, 0xCC, 255)


def test_assemble_skin_skips_invalid_grid():
    """An incorrectly-sized grid should be skipped, not crash."""
    pixel_data = {
        "head_front": _make_solid_grid(4, 4, "#FF0000"),  # Wrong size
    }
    img = assemble_skin(pixel_data, "classic")
    # Should be transparent since the grid was skipped
    pixel = img.getpixel((8, 8))
    assert pixel == (0, 0, 0, 0)


def test_assemble_skin_slim():
    """Slim model arm front should be 3px wide."""
    pixel_data = {
        "right_arm_front": _make_solid_grid(12, 3, "#0000FF"),
    }
    img = assemble_skin(pixel_data, "slim")
    assert img.size == (64, 64)
    # Right arm front for slim starts at (44, 20), width 3
    assert img.getpixel((44, 20)) == (0, 0, 255, 255)
    assert img.getpixel((46, 20)) == (0, 0, 255, 255)


def test_full_classic_skin_assembly():
    """Assemble a full skin with all 36 base regions."""
    pixel_data = {}
    parts = {
        "head": (8, 8),
        "body": None,  # custom sizes
        "right_arm": None,
        "left_arm": None,
        "right_leg": None,
        "left_leg": None,
    }

    # Head: all 8×8
    for face in ["front", "back", "top", "bottom", "left", "right"]:
        pixel_data[f"head_{face}"] = _make_solid_grid(8, 8, "#C4A882")

    # Body: front/back 12×8, top/bottom 4×8, left/right 12×4
    for face in ["front", "back"]:
        pixel_data[f"body_{face}"] = _make_solid_grid(12, 8, "#3B5998")
    for face in ["top", "bottom"]:
        pixel_data[f"body_{face}"] = _make_solid_grid(4, 8, "#3B5998")
    for face in ["left", "right"]:
        pixel_data[f"body_{face}"] = _make_solid_grid(12, 4, "#3B5998")

    # Arms (classic, 4px): front/back 12×4, top/bottom 4×4, left/right 12×4
    for arm in ["right_arm", "left_arm"]:
        for face in ["front", "back", "left", "right"]:
            pixel_data[f"{arm}_{face}"] = _make_solid_grid(12, 4, "#C4A882")
        for face in ["top", "bottom"]:
            pixel_data[f"{arm}_{face}"] = _make_solid_grid(4, 4, "#C4A882")

    # Legs: same as classic arms
    for leg in ["right_leg", "left_leg"]:
        for face in ["front", "back", "left", "right"]:
            pixel_data[f"{leg}_{face}"] = _make_solid_grid(12, 4, "#1A1A3E")
        for face in ["top", "bottom"]:
            pixel_data[f"{leg}_{face}"] = _make_solid_grid(4, 4, "#1A1A3E")

    img = assemble_skin(pixel_data, "classic")
    assert img.size == (64, 64)
    # Spot check a few key locations
    assert img.getpixel((8, 8))[:3] == (0xC4, 0xA8, 0x82)   # head front
    assert img.getpixel((20, 20))[:3] == (0x3B, 0x59, 0x98)  # body front
    assert img.getpixel((4, 20))[:3] == (0x1A, 0x1A, 0x3E)   # right leg front
