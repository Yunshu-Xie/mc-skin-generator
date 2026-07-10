"""Integration tests for the API endpoints."""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def dummy_image_bytes() -> bytes:
    """Create a small test image."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _make_solid_grid(rows: int, cols: int, color: str = "#FF0000") -> list[list[str]]:
    return [[color] * cols for _ in range(rows)]


def _make_dummy_pixel_data() -> dict:
    """Create a complete set of pixel data for classic model."""
    data = {}
    for face in ["front", "back", "top", "bottom", "left", "right"]:
        data[f"head_{face}"] = _make_solid_grid(8, 8, "#C4A882")

    for face in ["front", "back"]:
        data[f"body_{face}"] = _make_solid_grid(12, 8, "#3B5998")
    for face in ["top", "bottom"]:
        data[f"body_{face}"] = _make_solid_grid(4, 8, "#3B5998")
    for face in ["left", "right"]:
        data[f"body_{face}"] = _make_solid_grid(12, 4, "#3B5998")

    for part in ["right_arm", "left_arm", "right_leg", "left_leg"]:
        for face in ["front", "back", "left", "right"]:
            data[f"{part}_{face}"] = _make_solid_grid(12, 4, "#C4A882")
        for face in ["top", "bottom"]:
            data[f"{part}_{face}"] = _make_solid_grid(4, 4, "#C4A882")

    return data


@patch("app.routers.skin.generate_skin_data")
def test_generate_skin_success(mock_generate, client, dummy_image_bytes):
    """POST /api/generate should return skin info on success."""
    mock_generate.return_value = (
        _make_dummy_pixel_data(),
        {"description": "Test skin", "skin_tone": "#C4A882", "regions_generated": 36},
        {
            "model": "classic",
            "ai_model": "flash",
            "palette": ["#C4A882"] * 10,
            "pixel_grids": {"head_front": [[0] * 8 for _ in range(8)]},
            "description": "Test skin",
            "hair_style": "short",
        },
    )

    response = client.post(
        "/api/generate",
        files={"image": ("test.png", dummy_image_bytes, "image/png")},
        data={"model": "classic", "style_notes": ""},
    )

    assert response.status_code == 200
    data = response.json()
    assert "skin_id" in data
    assert "skin_url" in data
    assert data["model"] == "classic"


@patch("app.routers.skin.generate_skin_data")
def test_generate_skin_creates_png(mock_generate, client, dummy_image_bytes):
    """Generated skin should be downloadable as PNG."""
    mock_generate.return_value = (
        _make_dummy_pixel_data(),
        {"description": "Test", "regions_generated": 36},
        {
            "model": "classic",
            "ai_model": "flash",
            "palette": ["#C4A882"] * 10,
            "pixel_grids": {"head_front": [[0] * 8 for _ in range(8)]},
            "description": "Test",
            "hair_style": "short",
        },
    )

    response = client.post(
        "/api/generate",
        files={"image": ("test.png", dummy_image_bytes, "image/png")},
        data={"model": "classic"},
    )

    skin_url = response.json()["skin_url"]
    skin_response = client.get(skin_url)
    assert skin_response.status_code == 200
    assert skin_response.headers["content-type"] == "image/png"

    # Verify it's a valid 64×64 PNG
    img = Image.open(io.BytesIO(skin_response.content))
    assert img.size == (64, 64)


def test_generate_skin_invalid_model(client, dummy_image_bytes):
    """Invalid model type should return 400."""
    response = client.post(
        "/api/generate",
        files={"image": ("test.png", dummy_image_bytes, "image/png")},
        data={"model": "invalid"},
    )
    assert response.status_code == 400


def test_generate_skin_invalid_ai_model(client, dummy_image_bytes):
    """Invalid ai_model choice should return 400."""
    response = client.post(
        "/api/generate",
        files={"image": ("test.png", dummy_image_bytes, "image/png")},
        data={"model": "classic", "ai_model": "gpt-4"},
    )
    assert response.status_code == 400


@patch("app.routers.skin.generate_skin_data")
def test_generate_skin_passes_ai_model_choice(mock_generate, client, dummy_image_bytes):
    """ai_model from the request should be forwarded to generate_skin_data."""
    mock_generate.return_value = (
        _make_dummy_pixel_data(),
        {"description": "Test", "regions_generated": 36, "ai_model": "flash-lite"},
        {
            "model": "classic",
            "ai_model": "flash-lite",
            "palette": ["#C4A882"] * 10,
            "pixel_grids": {"head_front": [[0] * 8 for _ in range(8)]},
            "description": "Test",
            "hair_style": "short",
        },
    )

    response = client.post(
        "/api/generate",
        files={"image": ("test.png", dummy_image_bytes, "image/png")},
        data={"model": "classic", "ai_model": "flash-lite"},
    )

    assert response.status_code == 200
    assert mock_generate.call_args.kwargs["ai_model"] == "flash-lite"
    assert response.json()["metadata"]["ai_model"] == "flash-lite"


def test_generate_skin_no_file(client):
    """Missing image should return 422."""
    response = client.post("/api/generate", data={"model": "classic"})
    assert response.status_code == 422


def test_get_skin_not_found(client):
    """Non-existent skin should return 404."""
    response = client.get("/api/skin/nonexistent.png")
    assert response.status_code == 404


def test_get_skin_invalid_id(client):
    """Path traversal attempt should return 400."""
    response = client.get("/api/skin/../etc/passwd.png")
    # FastAPI/Starlette will either 400 or 404 this
    assert response.status_code in (400, 404, 422)
