"""Tests for skin_store — persisting per-skin generation state for later edits."""

from app.config import settings
from app.services.skin_store import load_skin_state, save_skin_state


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "skins_dir", str(tmp_path))

    save_skin_state(
        "abc123",
        model="classic",
        ai_model="flash",
        palette=["#111111", "#222222"],
        pixel_grids={"head_front": [[0, 1], [1, 0]]},
        description="a test character",
        hair_style="short",
    )

    state = load_skin_state("abc123")

    assert state == {
        "model": "classic",
        "ai_model": "flash",
        "palette": ["#111111", "#222222"],
        "pixel_grids": {"head_front": [[0, 1], [1, 0]]},
        "description": "a test character",
        "hair_style": "short",
    }


def test_load_missing_skin_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "skins_dir", str(tmp_path))
    assert load_skin_state("does-not-exist") is None
