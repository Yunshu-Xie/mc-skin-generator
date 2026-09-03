from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Vision (semantic layout only — never pixels) ──────────────────
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model_flash: str = "gemini-2.5-flash"
    gemini_model_flash_lite: str = "gemini-2.5-flash-lite"
    gemini_default_model: str = "flash"  # "flash" or "flash-lite"
    max_image_dimension: int = 768  # longest side sent to the vision API

    # ── Rendering (app.imaging / app.services.renderer) ───────────────
    palette_size: int = 16
    face_method: str = "dpid"  # box | dpid | dominant
    material_method: str = "dominant"
    presharpen: float = 0.45
    dpid_lambda: float = 1.4
    # Vision models report face boxes, not head boxes; grow one into the other.
    head_top_margin: float = 0.45
    head_side_margin: float = 0.12
    # "template" draws the head front from a face template coloured by the
    # photo; "photo" resamples the photo directly (the old, unreadable path).
    head_mode: str = "template"
    face_modulation: float = 0.6

    # ── Storage ───────────────────────────────────────────────────────
    skins_dir: str = "skins"
    max_upload_size: int = 5 * 1024 * 1024  # 5MB

    model_config = {"env_file": ".env"}


settings = Settings()
