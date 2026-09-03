"""Skin generation API.

Thin by design: validate input, run the two stages (semantic layout, then
rendering), persist, respond. All the interesting decisions live in
``app.imaging`` and ``app.services.renderer``.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import RecolorRequest, SkinGenerateResponse
from app.services import skin_store
from app.services.layout import analyze_photo
from app.services.renderer import RenderConfig, recolor, render
from app.services.skin_assembler import assemble_skin
from app.services.skin_map import ModelType

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _config() -> RenderConfig:
    return RenderConfig(
        palette_size=settings.palette_size,
        face_method=settings.face_method,  # type: ignore[arg-type]
        material_method=settings.material_method,  # type: ignore[arg-type]
        presharpen=settings.presharpen,
        dpid_lambda=settings.dpid_lambda,
        head_top_margin=settings.head_top_margin,
        head_side_margin=settings.head_side_margin,
        head_mode=settings.head_mode,
        face_modulation=settings.face_modulation,
    )


def _valid_id(skin_id: str) -> bool:
    return skin_id.isalnum() and len(skin_id) <= 32


@router.post("/generate", response_model=SkinGenerateResponse)
async def generate_skin(
    image: UploadFile = File(...),
    model: str = Form("classic"),
    style_notes: str = Form(""),
    ai_model: str = Form(None),
) -> SkinGenerateResponse:
    """Upload a photo and render a Minecraft skin from it."""
    if model not in ("classic", "slim"):
        raise HTTPException(400, "model must be 'classic' or 'slim'")

    ai_model = ai_model or settings.gemini_default_model
    if ai_model not in ("flash", "flash-lite"):
        raise HTTPException(400, "ai_model must be 'flash' or 'flash-lite'")

    content_type = image.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported image type: {content_type}")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file")
    if len(image_bytes) > settings.max_upload_size:
        raise HTTPException(400, "Image too large (max 5MB)")

    model_type: ModelType = "classic" if model == "classic" else "slim"

    # Stage 1 — what is in the photo, and where. Never raises.
    layout = await analyze_photo(
        image_bytes,
        media_type=content_type,
        style_notes=style_notes,
        ai_model=ai_model,  # type: ignore[arg-type]
    )

    # Stage 2 — the pixels, derived from the photo itself.
    try:
        result = render(image_bytes, layout, model_type, _config())
    except Exception as e:
        raise HTTPException(500, f"Skin rendering failed: {e}") from e

    skin_id = uuid.uuid4().hex[:8]
    metadata = {
        "description": layout.description,
        "layout_source": layout.source,
        "ai_model": layout.ai_model or ai_model,
        "roles": layout.roles,
    }
    skin_store.save(
        skin_id,
        assemble_skin(result.pixel_data, model_type),
        {
            "model": model,
            "pixel_data": result.pixel_data,
            "palette": result.palette,
            "roles": result.roles,
            "metrics": result.metrics,
            "metadata": metadata,
        },
    )

    return SkinGenerateResponse(
        skin_id=skin_id,
        skin_url=f"/api/skin/{skin_id}.png",
        model=model,
        palette=result.palette,
        roles=result.roles,
        metrics=result.metrics,
        metadata=metadata,
    )


@router.post("/skin/{skin_id}/recolor", response_model=SkinGenerateResponse)
async def recolor_skin(skin_id: str, body: RecolorRequest) -> SkinGenerateResponse:
    """Replace one palette color throughout an existing skin.

    Free: no vision call, no re-render — just a substitution over the stored
    pixel grids, which is only possible because every face draws from one
    shared palette.
    """
    if not _valid_id(skin_id):
        raise HTTPException(400, "Invalid skin ID")
    record = skin_store.load(skin_id)
    if record is None:
        raise HTTPException(404, "Skin not found")

    try:
        pixel_data = recolor(record["pixel_data"], body.old_color, body.new_color)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    model = record.get("model", "classic")
    model_type: ModelType = "classic" if model == "classic" else "slim"
    palette = [
        body.new_color.upper() if c.upper() == body.old_color.upper() else c
        for c in record.get("palette", [])
    ]

    new_id = uuid.uuid4().hex[:8]
    skin_store.save(
        new_id,
        assemble_skin(pixel_data, model_type),
        {**record, "pixel_data": pixel_data, "palette": palette},
    )

    return SkinGenerateResponse(
        skin_id=new_id,
        skin_url=f"/api/skin/{new_id}.png",
        model=model,
        palette=palette,
        roles=record.get("roles", {}),
        metrics=record.get("metrics", {}),
        metadata=record.get("metadata", {}),
    )


@router.get("/skin/{skin_id}.png")
async def get_skin(skin_id: str) -> FileResponse:
    """Download a generated skin PNG."""
    if not _valid_id(skin_id):
        raise HTTPException(400, "Invalid skin ID")

    path = skin_store.png_path(skin_id)
    if not path.exists():
        raise HTTPException(404, "Skin not found")

    return FileResponse(str(path), media_type="image/png", filename=f"minecraft_skin_{skin_id}.png")
