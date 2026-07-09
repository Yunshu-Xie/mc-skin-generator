"""Pydantic models for API request/response."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class SkinGenerateResponse(BaseModel):
    skin_id: str
    skin_url: str
    model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EditSkinRequest(BaseModel):
    instruction: str


class ErrorResponse(BaseModel):
    detail: str
