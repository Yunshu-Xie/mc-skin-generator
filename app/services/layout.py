"""The AI layer — semantics only, never pixels.

This is the single biggest change from the previous design. The old pipeline
asked Gemini to author an 8×8 grid of palette indices for the face. A language
model has no pixel-level spatial precision; it cannot reliably place an eye in
row 2, column 2. No amount of prompt engineering fixes that, because it is not
a knowledge problem. That is why faces came out as flesh-colored blobs.

So the model is now asked only for what it is genuinely good at:

* **where things are** — a normalized bounding box for the face, the torso,
  each arm, the legs, in the *uploaded photo*;
* **what role each material plays** — which color is skin, which is hair,
  which is the shirt.

Every actual pixel is then derived from the photo itself by
``app.services.renderer``. Output is ~20 numbers instead of ~500 grid cells,
so the call is cheaper, faster and far more reliable — the retry loop rarely
fires now.

If no API key is configured the module falls back to a fixed portrait layout,
which keeps the whole pipeline runnable (and testable) offline.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

AIModel = Literal["flash", "flash-lite"]

# Anchors the vision model is asked to locate. Names are deliberately about
# the *photo*, not about skin UV faces — mapping one to the other is the
# renderer's job.
ANCHORS = ("face", "eyes", "hair_top", "torso", "right_arm", "left_arm", "legs")
REQUIRED_ANCHORS = ("face",)

# Semantic color roles. These become anchored palette slots, so swapping one
# hex retints everywhere that material is used.
ROLES = (
    "skin_tone",
    "hair_color",
    "eye_color",
    "shirt_main",
    "arm_main",
    "pants_main",
    "shoe_color",
)
REQUIRED_ROLES = ("skin_tone", "hair_color")

# Used when the model omits an anchor, and for the offline fallback. Values
# are (x0, y0, x1, y1) as fractions of the image — a generic head-and-torso
# portrait framing.
DEFAULT_BOXES: dict[str, tuple[float, float, float, float]] = {
    "face": (0.30, 0.05, 0.70, 0.42),
    "eyes": (0.36, 0.19, 0.64, 0.26),
    "hair_top": (0.30, 0.02, 0.70, 0.14),
    "torso": (0.22, 0.44, 0.78, 0.78),
    "right_arm": (0.08, 0.46, 0.26, 0.78),
    "left_arm": (0.74, 0.46, 0.92, 0.78),
    "legs": (0.30, 0.78, 0.70, 1.00),
}

DEFAULT_ROLES: dict[str, str] = {
    "skin_tone": "#C4A882",
    "hair_color": "#3A2A1E",
    "eye_color": "#3B2B20",
    "shirt_main": "#3B5998",
    "arm_main": "#C4A882",
    "pants_main": "#1A1A3E",
    "shoe_color": "#2B2B2B",
}


@dataclass(frozen=True)
class Bbox:
    """A rectangle in normalized image coordinates, always valid once built."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        left = max(0, min(width - 1, int(round(self.x0 * width))))
        top = max(0, min(height - 1, int(round(self.y0 * height))))
        right = max(left + 1, min(width, int(round(self.x1 * width))))
        bottom = max(top + 1, min(height, int(round(self.y1 * height))))
        return left, top, right, bottom


@dataclass
class Layout:
    """What the vision model understood about the photo."""

    description: str = ""
    boxes: dict[str, Bbox] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    ai_model: str = ""
    source: Literal["vision", "fallback"] = "vision"

    def box(self, name: str) -> Bbox:
        if name in self.boxes:
            return self.boxes[name]
        return Bbox(*DEFAULT_BOXES.get(name, DEFAULT_BOXES["face"]))

    def role(self, name: str) -> str:
        return self.roles.get(name) or DEFAULT_ROLES.get(name, "#808080")


def default_layout(description: str = "") -> Layout:
    """A generic portrait layout — used offline and as the last retry fallback."""
    return Layout(
        description=description,
        boxes={k: Bbox(*v) for k, v in DEFAULT_BOXES.items()},
        roles=dict(DEFAULT_ROLES),
        source="fallback",
    )


# ── prompt ────────────────────────────────────────────────────────────

_PROMPT = """\
Look at this photo of a person or character. Return ONLY a JSON object (no \
markdown fence, no commentary) describing WHERE things are and WHAT COLOR \
they are. Do not draw anything.

{
  "description": "one short sentence about the subject",
  "boxes": {
    "face":      [x0, y0, x1, y1],
    "eyes":      [x0, y0, x1, y1],
    "hair_top":  [x0, y0, x1, y1],
    "torso":     [x0, y0, x1, y1],
    "right_arm": [x0, y0, x1, y1],
    "left_arm":  [x0, y0, x1, y1],
    "legs":      [x0, y0, x1, y1]
  },
  "roles": {
    "skin_tone": "#RRGGBB", "hair_color": "#RRGGBB", "eye_color": "#RRGGBB",
    "shirt_main": "#RRGGBB", "arm_main": "#RRGGBB",
    "pants_main": "#RRGGBB", "shoe_color": "#RRGGBB"
  }
}

Rules:
- Coordinates are fractions of the image, 0.0-1.0, origin at top-left, \
x0 < x1 and y0 < y1.
- "face" must frame the WHOLE HEAD, not the face: the top edge goes ABOVE the \
topmost hair, not at the forehead or eyebrows, and the bottom edge at the \
chin. A box that starts at the forehead produces a bald character.
- "eyes" is a thin band covering both eyes (and eyebrows if visible). At 8x8 \
the eyes are one or two pixels and would otherwise be averaged away, so this \
box is what makes the face readable — get it right even if you skip others.
- "right_arm"/"left_arm" are the subject's own right/left (mirrored on screen).
- Omit any box you cannot see in the photo rather than guessing.
- Colors are the dominant color of that material as it appears in the photo, \
in normal lighting — not in shadow, not blown out.
- Every hex value is exactly 7 characters."""


def _prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Shrink and re-encode for the vision call, to cut image tokens."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = settings.max_image_dimension
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _resolve_model_name(ai_model: AIModel) -> str:
    return (
        settings.gemini_model_flash_lite
        if ai_model == "flash-lite"
        else settings.gemini_model_flash
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.strip().startswith("```"))
    return json.loads(text)


# ── parsing ───────────────────────────────────────────────────────────


def _parse_bbox(raw: Any) -> Bbox | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:  # degenerate — treat as not reported
        return None
    return Bbox(x0, y0, x1, y1)


def _parse_hex(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            int(value[1:], 16)
        except ValueError:
            return None
        return value.upper()
    return None


def parse_layout(data: dict[str, Any]) -> tuple[Layout, bool]:
    """Turn a raw model response into a Layout. Returns (layout, is_complete)."""
    raw_boxes = data.get("boxes") if isinstance(data.get("boxes"), dict) else {}
    raw_roles = data.get("roles") if isinstance(data.get("roles"), dict) else {}

    boxes = {}
    for name in ANCHORS:
        parsed = _parse_bbox(raw_boxes.get(name))
        if parsed is not None:
            boxes[name] = parsed

    roles = {}
    for name in ROLES:
        parsed_hex = _parse_hex(raw_roles.get(name))
        if parsed_hex is not None:
            roles[name] = parsed_hex

    layout = Layout(
        description=str(data.get("description", ""))[:280],
        boxes=boxes,
        roles=roles,
    )
    complete = all(a in boxes for a in REQUIRED_ANCHORS) and all(r in roles for r in REQUIRED_ROLES)
    return layout, complete


# ── the call ──────────────────────────────────────────────────────────


async def _call_vision(
    image_bytes: bytes, media_type: str, style_notes: str, ai_model: AIModel
) -> dict[str, Any]:
    client = AsyncOpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        },
        {"type": "text", "text": _PROMPT},
    ]
    if style_notes:
        content.append({"type": "text", "text": f"\nStyle notes: {style_notes}"})

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=800,  # the whole answer is ~20 numbers now
        temperature=0.1,
        # Gemini 2.5 "thinking" spends max_tokens before emitting anything.
        reasoning_effort="none",
        messages=[{"role": "user", "content": content}],
    )
    return _extract_json(response.choices[0].message.content or "")


async def analyze_photo(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    style_notes: str = "",
    ai_model: AIModel = "flash",
    max_retries: int = 2,
) -> Layout:
    """Photo → semantic layout. Never raises; degrades to a default layout."""
    if not settings.gemini_api_key:
        logger.info("No GEMINI_API_KEY set — using the offline default layout")
        return default_layout("offline: default portrait layout")

    prepped, prepped_type = _prepare_image(image_bytes)
    last: Layout | None = None

    for attempt in range(max_retries + 1):
        try:
            raw = await _call_vision(prepped, prepped_type, style_notes, ai_model)
        except Exception:  # network, quota, malformed JSON — all recoverable
            logger.exception("Vision call failed (attempt %d)", attempt + 1)
            continue
        layout, complete = parse_layout(raw)
        layout.ai_model = ai_model
        last = layout
        if complete:
            return layout
        logger.warning(
            "Attempt %d: incomplete layout (boxes=%s roles=%s)",
            attempt + 1,
            sorted(layout.boxes),
            sorted(layout.roles),
        )

    if last is not None:
        last.source = "fallback"
        return last
    fallback = default_layout("vision unavailable")
    fallback.ai_model = ai_model
    return fallback
