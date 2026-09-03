"""API surface. The vision call is stubbed — these tests cover routing,
validation and persistence, not model behaviour.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.services import layout as layout_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_skins_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "skins_dir", str(tmp_path))
    yield


@pytest.fixture(autouse=True)
def stub_vision(monkeypatch):
    async def fake_analyze(*_args, **kwargs):
        result = layout_module.default_layout("stubbed portrait")
        result.ai_model = kwargs.get("ai_model", "flash")
        return result

    monkeypatch.setattr("app.routers.skin.analyze_photo", fake_analyze)


def _upload(portrait_bytes: bytes, **form):
    return client.post(
        "/api/generate",
        files={"image": ("portrait.png", portrait_bytes, "image/png")},
        data={"model": "classic", **form},
    )


def test_generate_returns_a_skin_with_palette_and_metrics(portrait_bytes):
    response = _upload(portrait_bytes)
    assert response.status_code == 200
    body = response.json()

    assert body["skin_url"] == f"/api/skin/{body['skin_id']}.png"
    assert len(body["palette"]) == settings.palette_size
    assert body["roles"]["skin_tone"] == 0
    assert "head_front" in body["metrics"]
    assert body["metadata"]["description"] == "stubbed portrait"


def test_the_primary_texture_matches_the_requested_scale(portrait_bytes):
    body = _upload(portrait_bytes, scale="2").json()
    assert body["scale"] == 2

    img = Image.open(io.BytesIO(client.get(body["skin_url"]).content))
    assert img.size == (128, 128)
    assert img.mode == "RGBA"


def test_a_64x64_companion_ships_with_every_larger_texture(portrait_bytes):
    """Vanilla Java rejects anything but 64x64, so it always gets one."""
    body = _upload(portrait_bytes, scale="2").json()
    assert body["vanilla_url"]

    img = Image.open(io.BytesIO(client.get(body["vanilla_url"]).content))
    assert img.size == (64, 64)


def test_scale_one_produces_no_companion(portrait_bytes):
    body = _upload(portrait_bytes, scale="1").json()
    assert body["scale"] == 1 and body["vanilla_url"] == ""

    img = Image.open(io.BytesIO(client.get(body["skin_url"]).content))
    assert img.size == (64, 64)


def test_an_unsupported_scale_is_rejected(portrait_bytes):
    assert _upload(portrait_bytes, scale="4").status_code == 400


def test_recolor_produces_a_new_skin_without_another_upload(portrait_bytes):
    original = _upload(portrait_bytes).json()
    old = original["palette"][3]

    response = client.post(
        f"/api/skin/{original['skin_id']}/recolor",
        json={"old_color": old, "new_color": "#FF00FF"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["skin_id"] != original["skin_id"]
    assert "#FF00FF" in body["palette"]
    assert client.get(body["skin_url"]).status_code == 200


def test_recolor_of_an_unknown_skin_is_404():
    response = client.post(
        "/api/skin/deadbeef/recolor", json={"old_color": "#000000", "new_color": "#FFFFFF"}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "form,expected",
    [({"model": "chunky"}, 400), ({"ai_model": "gpt"}, 400)],
)
def test_invalid_form_values_are_rejected(portrait_bytes, form, expected):
    assert _upload(portrait_bytes, **form).status_code == expected


def test_unsupported_content_type_is_rejected():
    response = client.post(
        "/api/generate",
        files={"image": ("notes.txt", b"hello", "text/plain")},
        data={"model": "classic"},
    )
    assert response.status_code == 400


def test_empty_upload_is_rejected():
    response = client.post(
        "/api/generate",
        files={"image": ("empty.png", b"", "image/png")},
        data={"model": "classic"},
    )
    assert response.status_code == 400


def test_path_traversal_in_skin_id_is_rejected():
    assert client.get("/api/skin/..%2F..%2Fetc%2Fpasswd.png").status_code in (400, 404)


def test_missing_skin_is_404():
    assert client.get("/api/skin/abc12345.png").status_code == 404
