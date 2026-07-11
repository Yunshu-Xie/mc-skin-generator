"""Tests for claude_vision — region-localization validation, photo-render orchestration."""

import io

from PIL import Image

from app.config import settings
from app.services.claude_vision import (
    _build_prompt,
    _prepare_image,
    _resolve_model_name,
    _validate_regions_response,
)

REGION_KEYS = ("head", "torso", "right_arm", "left_arm", "right_leg", "left_leg")


def test_resolve_model_name_flash():
    assert _resolve_model_name("flash") == settings.gemini_model_flash


def test_resolve_model_name_flash_lite():
    assert _resolve_model_name("flash-lite") == settings.gemini_model_flash_lite


def _valid_regions_response(**overrides) -> dict:
    data = {
        "hair_style": "short",
        "eye_shape": "round",
        "regions": {
            "head": {"visible": True, "bbox": [0.30, 0.05, 0.68, 0.35]},
            "torso": {"visible": True, "bbox": [0.20, 0.30, 0.75, 0.70]},
            "right_arm": {"visible": True, "bbox": [0.05, 0.30, 0.25, 0.65]},
            "left_arm": {"visible": True, "bbox": [0.70, 0.30, 0.95, 0.65]},
            "right_leg": {"visible": False},
            "left_leg": {"visible": False},
        },
    }
    data.update(overrides)
    return data


def test_validate_regions_response_complete():
    hair_style, eye_shape, regions, ok = _validate_regions_response(_valid_regions_response())
    assert ok is True
    assert hair_style == "short"
    assert eye_shape == "round"
    assert regions["head"] == (0.30, 0.05, 0.68, 0.35)
    assert regions["right_leg"] is None
    assert regions["left_leg"] is None


def test_validate_regions_response_missing_hair_style_is_incomplete():
    data = _valid_regions_response()
    del data["hair_style"]
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False


def test_validate_regions_response_invalid_eye_shape_is_incomplete():
    data = _valid_regions_response(eye_shape="squinty")
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False


def test_validate_regions_response_missing_regions_key_is_incomplete():
    data = _valid_regions_response()
    del data["regions"]
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False


def test_validate_regions_response_missing_one_region_entry_is_incomplete():
    data = _valid_regions_response()
    del data["regions"]["left_leg"]
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False  # regions dict must have all 6 keys to be structurally valid


def test_validate_regions_response_malformed_single_bbox_degrades_to_not_visible():
    data = _valid_regions_response()
    data["regions"]["torso"] = {"visible": True, "bbox": [0.9, 0.5, 0.1, 0.9]}  # x1 < x0
    _hs, _es, regions, ok = _validate_regions_response(data)
    assert ok is True  # whole response still valid
    assert regions["torso"] is None  # this one region degrades gracefully


def test_validate_regions_response_out_of_range_bbox_degrades_to_not_visible():
    data = _valid_regions_response()
    data["regions"]["right_arm"] = {"visible": True, "bbox": [0.1, 0.1, 1.5, 0.5]}
    _hs, _es, regions, ok = _validate_regions_response(data)
    assert ok is True
    assert regions["right_arm"] is None


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


def test_build_prompt_mentions_all_six_regions_and_categorical_fields():
    prompt = _build_prompt()
    for key in REGION_KEYS:
        assert key in prompt
    assert "hair_style" in prompt
    assert "eye_shape" in prompt
    assert "bbox" in prompt
    assert "visible" in prompt


def _portrait_photo_bytes() -> bytes:
    """A synthetic 'portrait' with distinct colors per body region, so the
    orchestration test can assert real photo-render output, not just shapes."""
    img = Image.new("RGB", (200, 400), (240, 240, 240))
    for y in range(20, 140):  # head band: hair top, skin below
        for x in range(60, 140):
            img.putpixel((x, y), (40, 20, 10) if y < 60 else (200, 160, 120))
    for y in range(140, 280):  # torso: red
        for x in range(40, 160):
            img.putpixel((x, y), (200, 30, 30))
    for y in range(140, 280):  # right arm: green (image-left = character's right)
        for x in range(10, 40):
            img.putpixel((x, y), (30, 200, 30))
    for y in range(140, 280):  # left arm: blue
        for x in range(160, 190):
            img.putpixel((x, y), (30, 30, 200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


async def test_generate_skin_data_end_to_end_with_mocked_vision_call(monkeypatch):
    async def fake_call_vision(*_args, **_kwargs):
        return {
            "hair_style": "short",
            "eye_shape": "round",
            "regions": {
                "head": {"visible": True, "bbox": [0.30, 0.05, 0.70, 0.35]},
                "torso": {"visible": True, "bbox": [0.20, 0.35, 0.80, 0.70]},
                "right_arm": {"visible": True, "bbox": [0.05, 0.35, 0.20, 0.70]},
                "left_arm": {"visible": True, "bbox": [0.80, 0.35, 0.95, 0.70]},
                "right_leg": {"visible": False},
                "left_leg": {"visible": False},
            },
        }

    import app.services.claude_vision as claude_vision

    monkeypatch.setattr(claude_vision, "_call_vision", fake_call_vision)

    pixel_data, metadata, persist_state = await claude_vision.generate_skin_data(
        image_bytes=_portrait_photo_bytes(),
        media_type="image/png",
        model="classic",
    )

    # Photo-derived mandatory faces are present with correct shapes.
    assert len(pixel_data["head_front"]) == 8 and len(pixel_data["head_front"][0]) == 8
    assert len(pixel_data["body_front"]) == 12 and len(pixel_data["body_front"][0]) == 8
    assert len(pixel_data["right_arm_front"]) == 12 and len(pixel_data["right_arm_front"][0]) == 4

    # Legs weren't visible in the mocked regions -> procedural flat fill, not crashed.
    assert len(pixel_data["right_leg_front"]) == 12

    # Every base region (36 total for classic) ends up populated.
    assert len(pixel_data) == 36

    assert metadata["skin_tone"]
    assert metadata["hair_color"]
    assert metadata["ai_model"] == "flash"

    assert persist_state["model"] == "classic"
    assert isinstance(persist_state["palette"], list) and len(persist_state["palette"]) > 0
    assert set(persist_state["pixel_grids"].keys()) == {
        "head_front",
        "body_front",
        "right_arm_front",
        "left_arm_front",
        "right_leg_front",
        "left_leg_front",
    }


async def test_generate_skin_data_retries_on_incomplete_response(monkeypatch):
    calls = {"n": 0}

    async def flaky_call_vision(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"hair_style": "short"}  # missing eye_shape/regions -> incomplete
        return {
            "hair_style": "short",
            "eye_shape": "narrow",
            "regions": {
                "head": {"visible": True, "bbox": [0.30, 0.05, 0.70, 0.35]},
                "torso": {"visible": False},
                "right_arm": {"visible": False},
                "left_arm": {"visible": False},
                "right_leg": {"visible": False},
                "left_leg": {"visible": False},
            },
        }

    import app.services.claude_vision as claude_vision

    monkeypatch.setattr(claude_vision, "_call_vision", flaky_call_vision)

    _pixel_data, metadata, _persist_state = await claude_vision.generate_skin_data(
        image_bytes=_portrait_photo_bytes(),
        media_type="image/png",
        model="classic",
    )

    assert calls["n"] == 2
    assert metadata["skin_tone"]
