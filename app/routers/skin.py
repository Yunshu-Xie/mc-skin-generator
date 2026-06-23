"""Skin generation API endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import SkinGenerateResponse
from app.services.claude_vision import generate_skin_data
from app.services.skin_assembler import assemble_skin
from app.services.skin_map import ModelType

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@router.post("/generate", response_model=SkinGenerateResponse)
async def generate_skin(
    image: UploadFile = File(...),
    model: str = Form("classic"),
    style_notes: str = Form(""),
) -> SkinGenerateResponse:
    """Upload an image and generate a Minecraft skin."""
    # Validate model type
    if model not in ("classic", "slim"):
        raise HTTPException(400, "model must be 'classic' or 'slim'")

    # Validate file type
    content_type = image.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported image type: {content_type}")

    # Read and validate size
    image_bytes = await image.read()
    if len(image_bytes) > settings.max_upload_size:
        raise HTTPException(400, "Image too large (max 5MB)")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    # Generate skin via Claude pipeline
    model_type: ModelType = "classic" if model == "classic" else "slim"

    try:
        pixel_data, metadata = await generate_skin_data(
            image_bytes=image_bytes,
            media_type=content_type,
            model=model_type,
            style_notes=style_notes,
        )
    except Exception as e:
        raise HTTPException(500, f"Skin generation failed: {e}") from e

    # Assemble the 64×64 PNG
    skin_image = assemble_skin(pixel_data, model_type)

    # Save to disk
    skin_id = uuid.uuid4().hex[:8]
    skin_path = Path(settings.skins_dir) / f"{skin_id}.png"
    skin_image.save(str(skin_path), "PNG")

    return SkinGenerateResponse(
        skin_id=skin_id,
        skin_url=f"/api/skin/{skin_id}.png",
        model=model,
        metadata=metadata,
    )


@router.get("/skin/{skin_id}.png")
async def get_skin(skin_id: str) -> FileResponse:
    """Download a generated skin PNG."""
    # Sanitize skin_id to prevent path traversal
    if not skin_id.isalnum():
        raise HTTPException(400, "Invalid skin ID")

    skin_path = Path(settings.skins_dir) / f"{skin_id}.png"
    if not skin_path.exists():
        raise HTTPException(404, "Skin not found")

    return FileResponse(
        str(skin_path),
        media_type="image/png",
        filename=f"minecraft_skin_{skin_id}.png",
    )
