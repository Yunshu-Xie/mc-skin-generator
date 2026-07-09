"""Tests for claude_vision — indexed-grid decoding, response validation, image prep."""

import io

from PIL import Image

from app.config import settings
from app.services.claude_vision import (
    _decode_indexed_grid,
    _prepare_image,
    _resolve_model_name,
    _validate_and_decode,
)
from app.services.procedural import AI_GENERATED_KEYS


def test_resolve_model_name_flash():
    assert _resolve_model_name("flash") == settings.gemini_model_flash


def test_resolve_model_name_flash_lite():
    assert _resolve_model_name("flash-lite") == settings.gemini_model_flash_lite


def test_decode_indexed_grid_maps_to_palette():
    palette = ["#111111", "#222222", "#333333"]
    grid = [[0, 1], [2, 0]]
    assert _decode_indexed_grid(grid, palette) == [
        ["#111111", "#222222"],
        ["#333333", "#111111"],
    ]


def test_decode_indexed_grid_clamps_bad_index():
    palette = ["#111111", "#222222"]
    grid = [[0, 99], [-1, 1]]
    decoded = _decode_indexed_grid(grid, palette)
    assert decoded[0][1] == "#111111"  # out-of-range clamps to palette[0]
    assert decoded[1][0] == "#111111"  # negative clamps to palette[0]
    assert decoded[1][1] == "#222222"


def _valid_response(model: str = "classic") -> dict:
    data = {
        "description": "A test character",
        "skin_tone": "#C4A882",
        "hair_color": "#5B3A1A",
        "hair_style": "short",
        "eye_color": "#3B5998",
        "shirt_main": "#3B5998",
        "shirt_shadow": "#28406E",
        "arm_main": "#C4A882",
        "arm_shadow": "#9C8266",
        "pants_main": "#1A1A3E",
        "pants_shadow": "#101028",
        "shoe_color": "#2B2B2B",
        "palette": ["#C4A882", "#5B3A1A", "#3B5998"],
    }
    for key in AI_GENERATED_KEYS:
        h = 12 if key == "body_front" else 8
        data[key] = [[0] * 8 for _ in range(h)]
    return data


def test_validate_and_decode_complete_response():
    pixel_data, colors, metadata, ok = _validate_and_decode(_valid_response(), "classic")
    assert ok is True
    assert set(pixel_data.keys()) == AI_GENERATED_KEYS
    assert metadata["description"] == "A test character"
    assert colors["shirt_main"] == "#3B5998"
    # decoded grids use hex, not raw indices
    assert pixel_data["head_front"][0][0] == "#C4A882"


def test_validate_and_decode_missing_palette_is_incomplete():
    data = _valid_response()
    del data["palette"]
    pixel_data, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert pixel_data == {}


def test_validate_and_decode_wrong_grid_size_is_incomplete():
    data = _valid_response()
    data["head_front"] = [[0] * 4 for _ in range(4)]  # wrong size
    pixel_data, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert "head_front" not in pixel_data
    # other grids still decoded
    assert "body_front" in pixel_data


def test_validate_and_decode_missing_required_color_is_incomplete():
    data = _valid_response()
    data["skin_tone"] = ""
    _pixel_data, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False


def test_prepare_image_downscales_large_image():
    img = Image.new("RGB", (2000, 1000), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")

    out_bytes, media_type = _prepare_image(buf.getvalue())

    assert media_type == "image/jpeg"
    out_img = Image.open(io.BytesIO(out_bytes))
    assert max(out_img.size) <= 768


def test_prepare_image_leaves_small_image_dimensions_alone():
    img = Image.new("RGB", (100, 50), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")

    out_bytes, _media_type = _prepare_image(buf.getvalue())

    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.size == (100, 50)
