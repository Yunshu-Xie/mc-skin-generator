"""Tests for claude_vision — indexed-grid decoding, response validation, image prep."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from app.config import settings
from app.services.claude_vision import (
    _build_prompt,
    _is_valid_hex,
    _prepare_image,
    _resolve_model_name,
    _validate_and_decode,
    apply_color_edit,
    decode_indexed_grid,
    interpret_color_edit,
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


_MANDATORY_FACE_DIMS = {
    "head_front": (8, 8),
    "head_back": (8, 8),
    "head_top": (8, 8),
    "head_bottom": (8, 8),
    "head_left": (8, 8),
    "head_right": (8, 8),
    "body_front": (12, 8),
    "right_arm_front": (12, 4),
    "left_arm_front": (12, 4),
    "right_leg_front": (12, 4),
    "left_leg_front": (12, 4),
}


def _valid_response(n_freeform: int = 0, extra_faces: dict | None = None) -> dict:
    data = {
        "description": "A test character",
        "hair_style": "short",
        "palette": _fixed_palette(n_freeform),
    }
    for key, (h, w) in _MANDATORY_FACE_DIMS.items():
        data[key] = [[0] * w for _ in range(h)]
    if extra_faces:
        data.update(extra_faces)
    return data


def test_validate_and_decode_complete_response():
    pixel_data, raw_grids, colors, metadata, ok = _validate_and_decode(
        _valid_response(), "classic"
    )
    assert ok is True
    # pixel_data legitimately contains AI_GENERATED_KEYS's 6 mandatory keys
    # PLUS the 5 non-front head faces, which the fixture still supplies and
    # which still get decoded via the "optional detail face" loop.
    assert AI_GENERATED_KEYS <= set(pixel_data.keys())
    assert AI_GENERATED_KEYS <= set(raw_grids.keys())
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
    # regions_generated counts however many keys actually got decoded: the
    # fixture's full set of (still-11) mandatory-per-fixture faces + 1 detail
    # face, not len(AI_GENERATED_KEYS) which is now smaller than what the
    # fixture actually supplies.
    assert metadata["regions_generated"] == len(_MANDATORY_FACE_DIMS) + 1


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


def test_validate_and_decode_missing_arm_front_is_incomplete():
    data = _valid_response()
    del data["right_arm_front"]
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert "right_arm_front" not in pixel_data
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


def _mock_chat_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


@patch("app.services.claude_vision._get_client")
async def test_interpret_color_edit_maps_instruction_to_role(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response('{"shirt_main": "#0000FF"}')
    )
    mock_get_client.return_value = mock_client

    changes = await interpret_color_edit(
        {"shirt_main": "#AA3355", "hair_color": "#5B3A1A"},
        "把衬衫改成蓝色",
        "flash-lite",
    )

    assert changes == {"shirt_main": "#0000FF"}


@patch("app.services.claude_vision._get_client")
async def test_interpret_color_edit_ignores_unknown_roles_and_bad_hex(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response(
            '{"shirt_main": "#0000FF", "made_up_role": "#FFFFFF", "hair_color": "blue"}'
        )
    )
    mock_get_client.return_value = mock_client

    changes = await interpret_color_edit(
        {"shirt_main": "#AA3355", "hair_color": "#5B3A1A"}, "make it blue", "flash"
    )

    assert changes == {"shirt_main": "#0000FF"}


@patch("app.services.claude_vision._get_client")
async def test_interpret_color_edit_returns_empty_on_bad_json(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response("not json at all")
    )
    mock_get_client.return_value = mock_client

    changes = await interpret_color_edit({"shirt_main": "#AA3355"}, "anything", "flash")

    assert changes == {}


@patch(
    "app.services.claude_vision.interpret_color_edit",
    new_callable=AsyncMock,
    return_value={"shirt_main": "#0000FF"},
)
async def test_apply_color_edit_retints_palette_and_reassembles(mock_interpret):
    state = {
        "model": "classic",
        "ai_model": "flash",
        "palette": _fixed_palette(),
        "pixel_grids": {"body_front": [[3] * 8 for _ in range(12)]},
        "description": "a test character",
        "hair_style": "short",
    }

    pixel_data, metadata, persist_state = await apply_color_edit(state, "把衬衫改成蓝色")

    assert pixel_data["body_front"][0][0] == "#0000FF"  # index 3 now retinted
    assert persist_state["palette"][3] == "#0000FF"
    assert persist_state["pixel_grids"] == state["pixel_grids"]  # raw indices untouched
    assert metadata["changed_roles"] == ["shirt_main"]
    assert "body_back" in pixel_data  # procedural fill still runs for the rest
    mock_interpret.assert_awaited_once_with(
        {name: _fixed_palette()[idx] for name, idx in PALETTE_ROLES.items()},
        "把衬衫改成蓝色",
        "flash",
    )


@patch(
    "app.services.claude_vision.interpret_color_edit",
    new_callable=AsyncMock,
    return_value={"shirt_main": "#0000FF"},
)
async def test_apply_color_edit_short_palette_is_noop(mock_interpret):
    """A persisted skin whose palette is shorter than MIN_PALETTE_SIZE (e.g. from an
    incomplete generation that still got persisted) must not raise IndexError —
    apply_color_edit should return the skin unchanged instead."""
    state = {
        "model": "classic",
        "ai_model": "flash",
        "palette": ["#C4A882", "#5B3A1A", "#3B5998"],  # only 3 entries, < MIN_PALETTE_SIZE
        "pixel_grids": {"body_front": [[0] * 8 for _ in range(12)]},
        "description": "a test character",
        "hair_style": "short",
    }

    pixel_data, metadata, persist_state = await apply_color_edit(state, "make it blue")

    assert metadata["changed_roles"] == []
    assert persist_state["palette"] == state["palette"]
    assert "body_front" in pixel_data
    mock_interpret.assert_not_awaited()


def test_build_prompt_includes_face_features_guidance():
    prompt = _build_prompt("classic")
    assert "face_features" in prompt
    assert "eye_shape" in prompt
    assert "eyebrow_color" in prompt
    assert "mouth_color" in prompt


def test_build_prompt_includes_mandatory_limb_fronts():
    prompt = _build_prompt("classic")
    for key in (
        "right_arm_front",
        "left_arm_front",
        "right_leg_front",
        "left_leg_front",
    ):
        assert key in prompt
