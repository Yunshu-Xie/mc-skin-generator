"""Tests for claude_vision — indexed-grid decoding, response validation, image prep."""

import io

from PIL import Image

from app.config import settings
from app.services.claude_vision import (
    _is_valid_hex,
    _prepare_image,
    _resolve_model_name,
    _validate_and_decode,
    decode_indexed_grid,
)
from app.services.procedural import AI_GENERATED_KEYS
from app.services.skin_map import MAX_PALETTE_SIZE, MIN_PALETTE_SIZE, PALETTE_ROLES


def test_resolve_model_name_flash():
    assert _resolve_model_name("flash") == settings.gemini_model_flash


def test_resolve_model_name_flash_lite():
    assert _resolve_model_name("flash-lite") == settings.gemini_model_flash_lite


def test_is_valid_hex():
    assert _is_valid_hex("#AABBCC") is True
    assert _is_valid_hex("#aabbcc") is True
    assert _is_valid_hex("not-a-color") is False
    assert _is_valid_hex("#AABBCCDD") is False  # 8-char not accepted for palette entries
    assert _is_valid_hex(123) is False


def test_decode_indexed_grid_maps_to_palette():
    palette = ["#111111", "#222222", "#333333"]
    grid = [[0, 1], [2, 0]]
    assert decode_indexed_grid(grid, palette) == [
        ["#111111", "#222222"],
        ["#333333", "#111111"],
    ]


def test_decode_indexed_grid_clamps_bad_index():
    palette = ["#111111", "#222222"]
    grid = [[0, 99], [-1, 1]]
    decoded = decode_indexed_grid(grid, palette)
    assert decoded[0][1] == "#111111"  # out-of-range clamps to palette[0]
    assert decoded[1][0] == "#111111"  # negative clamps to palette[0]
    assert decoded[1][1] == "#222222"


def _fixed_palette(n_freeform: int = 0) -> list[str]:
    base = [
        "#C4A882",  # skin_tone
        "#5B3A1A",  # hair_color
        "#3B5998",  # eye_color
        "#AA3355",  # shirt_main
        "#882244",  # shirt_shadow
        "#C4A882",  # arm_main
        "#9C8266",  # arm_shadow
        "#1A1A3E",  # pants_main
        "#101028",  # pants_shadow
        "#2B2B2B",  # shoe_color
    ]
    return base + [f"#{i:02X}{i:02X}{i:02X}" for i in range(10, 10 + n_freeform)]


def _valid_response(n_freeform: int = 0, extra_faces: dict | None = None) -> dict:
    data = {
        "description": "A test character",
        "hair_style": "short",
        "palette": _fixed_palette(n_freeform),
    }
    for key in AI_GENERATED_KEYS:
        h = 12 if key == "body_front" else 8
        data[key] = [[0] * 8 for _ in range(h)]
    if extra_faces:
        data.update(extra_faces)
    return data


def test_validate_and_decode_complete_response():
    pixel_data, raw_grids, colors, metadata, ok = _validate_and_decode(
        _valid_response(), "classic"
    )
    assert ok is True
    assert set(pixel_data.keys()) == AI_GENERATED_KEYS
    assert set(raw_grids.keys()) == AI_GENERATED_KEYS
    assert metadata["description"] == "A test character"
    assert colors["shirt_main"] == "#AA3355"
    assert pixel_data["head_front"][0][0] == "#C4A882"  # decoded index 0 -> skin_tone
    assert raw_grids["head_front"][0][0] == 0  # raw index preserved


def test_validate_and_decode_named_colors_match_palette_roles():
    _pixel_data, _raw, colors, _metadata, _ok = _validate_and_decode(
        _valid_response(), "classic"
    )
    palette = _fixed_palette()
    for name, idx in PALETTE_ROLES.items():
        assert colors[name] == palette[idx]


def test_validate_and_decode_decodes_optional_detail_face():
    extra = {"body_back": [[3] * 8 for _ in range(12)]}
    pixel_data, raw_grids, _colors, metadata, ok = _validate_and_decode(
        _valid_response(extra_faces=extra), "classic"
    )
    assert ok is True
    assert "body_back" in pixel_data
    assert "body_back" in raw_grids
    assert pixel_data["body_back"][0][0] == "#AA3355"  # index 3 -> shirt_main
    assert metadata["regions_generated"] == len(AI_GENERATED_KEYS) + 1


def test_validate_and_decode_ignores_malformed_detail_face():
    extra = {"body_back": [[3] * 4 for _ in range(4)]}  # wrong dims for body_back
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(
        _valid_response(extra_faces=extra), "classic"
    )
    assert ok is True  # mandatory faces still fine
    assert "body_back" not in pixel_data


def test_validate_and_decode_palette_too_short_is_incomplete():
    data = _valid_response()
    data["palette"] = data["palette"][:5]
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert pixel_data == {}


def test_validate_and_decode_palette_too_long_is_incomplete():
    data = _valid_response(n_freeform=MAX_PALETTE_SIZE - MIN_PALETTE_SIZE + 1)
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert pixel_data == {}


def test_validate_and_decode_invalid_hex_in_palette_is_incomplete():
    data = _valid_response()
    data["palette"][0] = "not-a-color"
    _pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False


def test_validate_and_decode_wrong_grid_size_is_incomplete():
    data = _valid_response()
    data["head_front"] = [[0] * 4 for _ in range(4)]  # wrong size
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert "head_front" not in pixel_data
    assert "body_front" in pixel_data  # other mandatory grids still decoded


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
