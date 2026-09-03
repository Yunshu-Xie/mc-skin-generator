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
    # Texture scale: 1 = 64x64, 2 = 128x128. Vanilla Java only accepts 64x64,
    # so a 64x64 companion is always exported alongside a larger primary.
    skin_scale: int = 2
    head_mode: str = "auto"
    # "template" draws clothes from garment templates; "photo" resamples them
    # (keeps chest logos, loses garment structure).
    body_mode: str = "auto"
    face_modulation: float = 0.6
    brows: bool = True
    overlay_hair: bool = True
    # Albedo: strip the photo's lighting so the texture is the material itself.
    flatten_shading: float = 1.0
    palette_lightness_weight: float = 0.7
    albedo_percentile: float = 80.0
    bake_orientation_shading: float = 0.0

    # ── Storage ───────────────────────────────────────────────────────
    skins_dir: str = "skins"
    max_upload_size: int = 5 * 1024 * 1024  # 5MB

    model_config = {"env_file": ".env"}


settings = Settings()
