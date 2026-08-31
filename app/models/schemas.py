"""Pydantic models for API request/response."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SkinGenerateResponse(BaseModel):
    skin_id: str
    skin_url: str
    model: str
    palette: List[str] = Field(default_factory=list)
    roles: Dict[str, int] = Field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecolorRequest(BaseModel):
    """Swap one palette color for another. No AI call, no re-render."""

    old_color: str = Field(description="Existing palette hex, e.g. '#3B5998'")
    new_color: str = Field(description="Replacement hex, e.g. '#B03030'")


class ErrorResponse(BaseModel):
    detail: str
