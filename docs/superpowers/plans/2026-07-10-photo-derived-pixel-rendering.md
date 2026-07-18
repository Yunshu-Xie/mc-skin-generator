# Photo-Derived Pixel Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace "AI hand-places every pixel via JSON" with "AI only locates body-part regions; deterministic Pillow image processing (crop → dominant-color-per-cell downsample → shared color quantization) produces the actual pixels" — per `docs/superpowers/specs/2026-07-10-photo-derived-pixel-rendering-design.md`. Retires the conversational color-edit feature and the fixed `PALETTE_ROLES` structure it depended on.

**Architecture:** A new `app/services/photo_render.py` module is the deterministic image-processing toolbox (crop, downsample, shared quantization, face-band color measurement, head-template rendering). `app/services/claude_vision.py`'s Vision call is rewritten to ask only for 6 region bounding boxes + `hair_style`/`eye_shape` — a much smaller, more reliable ask — and its orchestration wires the real photo through `photo_render.py` to build the same `palette + {face: index grid}` shape the rest of the system already expects. `app/services/procedural.py` shrinks its AI-mandatory face set from 11 to 6 (the head's non-front faces become a flat procedural fill, since a front photo never shows them anyway) and gains a `head_fill_color` default. The conversational edit feature (route, schema, frontend UI, `PALETTE_ROLES`) is retired since it depended on the fixed-role palette this design removes.

**Tech Stack:** Python 3.11, FastAPI, Pillow (crop/resize/`getcolors`/`quantize` — no new dependency), `openai` SDK against Gemini's OpenAI-compatible endpoint, pytest.

## Global Constraints

- All Python files use `from __future__ import annotations` (existing codebase convention).
- No new pip dependencies — everything is achieved with Pillow, already a dependency.
- Every new/changed function needs a test; run `.venv/bin/python -m pytest -q` (full suite) after each task and confirm it's all green before committing — a task that leaves the suite red is not done.
- `.venv/bin/ruff check <changed files>` must be clean before committing each task.
- `app/services/skin_map.py`'s `FaceRect`/`get_all_regions`/`PIXEL_KEY_MAP`, `app/services/skin_assembler.py`, and `app/services/skin_store.py` are **not touched** by this plan except Task 5's narrow removal of `PALETTE_ROLES`/`MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE` from `skin_map.py`.
- `app/services/procedural.py`'s `propagate_front_row` function itself is unchanged; Task 3 fixes the surrounding skip logic (see Task 3's intro) so a missing front face is flat-filled rather than silently omitted — this is a bug fix, not a behavior change for any currently-working case.
- `tests/test_skin_store.py` is not touched.

---

### Task 1: Retire the conversational edit feature's route, schema, and frontend UI

This removes the *outer* layers of the edit feature (HTTP route, request schema, frontend controls) first, since they're leaf dependents that nothing else in this plan needs. `interpret_color_edit`/`apply_color_edit`/`EDIT_MAX_TOKENS` themselves (in `claude_vision.py`) and `PALETTE_ROLES` (in `skin_map.py`) are removed later (Tasks 4 and 5) once nothing calls them.

**Files:**
- Modify: `app/routers/skin.py` (remove the edit route + its imports)
- Modify: `app/models/schemas.py` (remove `EditSkinRequest`)
- Modify: `app/static/index.html` (remove the edit-controls markup)
- Modify: `app/static/app.js` (remove the edit DOM refs, handler, and reset-handler references to them)
- Modify: `app/static/style.css` (remove the `.edit-controls` rules)
- Modify: `tests/test_api.py` (remove the three edit-endpoint tests)

**Interfaces:** None — this task only removes code; nothing later in this plan depends on anything it touches.

- [ ] **Step 1: Remove the edit endpoint tests (this makes the suite's expectations match the post-removal router)**

In `tests/test_api.py`, delete these three tests entirely (including their `@patch` decorators):
`test_edit_skin_success`, `test_edit_skin_not_found`, `test_edit_skin_invalid_id`.

- [ ] **Step 2: Remove the route from `app/routers/skin.py`**

Replace the whole file with:

```python
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
from app.services.skin_store import save_skin_state

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@router.post("/generate", response_model=SkinGenerateResponse)
async def generate_skin(
    image: UploadFile = File(...),
    model: str = Form("classic"),
    style_notes: str = Form(""),
    ai_model: str = Form(None),
) -> SkinGenerateResponse:
    """Upload an image and generate a Minecraft skin."""
    # Validate model type
    if model not in ("classic", "slim"):
        raise HTTPException(400, "model must be 'classic' or 'slim'")

    # Validate AI model choice (which Gemini variant to use for this generation)
    ai_model = ai_model or settings.gemini_default_model
    if ai_model not in ("flash", "flash-lite"):
        raise HTTPException(400, "ai_model must be 'flash' or 'flash-lite'")

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

    # Generate skin via the Gemini pipeline
    model_type: ModelType = "classic" if model == "classic" else "slim"

    try:
        pixel_data, metadata, persist_state = await generate_skin_data(
            image_bytes=image_bytes,
            media_type=content_type,
            model=model_type,
            style_notes=style_notes,
            ai_model=ai_model,
        )
    except Exception as e:
        raise HTTPException(500, f"Skin generation failed: {e}") from e

    # Assemble the 64×64 PNG
    skin_image = assemble_skin(pixel_data, model_type)

    # Save PNG + generation state
    skin_id = uuid.uuid4().hex[:8]
    skin_path = Path(settings.skins_dir) / f"{skin_id}.png"
    skin_image.save(str(skin_path), "PNG")
    save_skin_state(skin_id, **persist_state)

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
```

- [ ] **Step 3: Remove `EditSkinRequest` from `app/models/schemas.py`**

Replace the whole file with:

```python
"""Pydantic models for API request/response."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class SkinGenerateResponse(BaseModel):
    skin_id: str
    skin_url: str
    model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
```

- [ ] **Step 4: Remove the edit-controls markup from `app/static/index.html`**

Replace:

```html
                <div class="viewer-controls">
                    <button id="downloadBtn" class="btn-primary">⬇ 下载 .png</button>
                    <button id="resetBtn" class="btn-secondary">🔄 重新生成</button>
                </div>

                <div class="edit-controls">
                    <input type="text" id="editInstruction" placeholder="比如：把衬衫改成蓝色">
                    <button id="editBtn" class="btn-secondary">✏️ 应用修改</button>
                </div>
```

with:

```html
                <div class="viewer-controls">
                    <button id="downloadBtn" class="btn-primary">⬇ 下载 .png</button>
                    <button id="resetBtn" class="btn-secondary">🔄 重新生成</button>
                </div>
```

- [ ] **Step 5: Remove the edit DOM refs/handler from `app/static/app.js`**

Replace:

```js
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");
const editInstruction = document.getElementById("editInstruction");
const editBtn = document.getElementById("editBtn");

let viewer = null;
let currentSkinUrl = null;
let currentSkinId = null;
```

with:

```js
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");

let viewer = null;
let currentSkinUrl = null;
let currentSkinId = null;
```

Remove the entire `// ── Conversational Color Edit ──` block:

```js
// ── Conversational Color Edit ──
editBtn.addEventListener("click", async () => {
    const instruction = editInstruction.value.trim();
    if (!instruction || !currentSkinId) return;

    editBtn.disabled = true;
    errorDiv.hidden = true;

    try {
        const response = await fetch(`/api/skin/${currentSkinId}/edit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ instruction }),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "修改失败");
        }

        const bustedUrl = `${data.skin_url}?t=${Date.now()}`;
        showViewer(data.skin_id, bustedUrl, data.model, data.metadata);
        editInstruction.value = "";
    } catch (err) {
        errorDiv.textContent = `❌ ${err.message}`;
        errorDiv.hidden = false;
    } finally {
        editBtn.disabled = false;
    }
});

// ── Reset ──
```

with just:

```js
// ── Reset ──
```

In the same reset handler, replace:

```js
    currentSkinUrl = null;
    currentSkinId = null;
    editInstruction.value = "";
    preview.hidden = true;
```

with:

```js
    currentSkinUrl = null;
    currentSkinId = null;
    preview.hidden = true;
```

(`currentSkinId` is still used — `showViewer` still tracks it even though nothing submits edits anymore; keeping it avoids a wider refactor of `showViewer`'s signature, which is out of scope here.)

- [ ] **Step 6: Remove the `.edit-controls` rules from `app/static/style.css`**

Replace:

```css
.viewer-controls button {
    flex: 1;
    margin-top: 0;
}

.edit-controls {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
}

.edit-controls input[type="text"] {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid #333;
    border-radius: 8px;
    background: #16213e;
    color: #e0e0e0;
    font-family: inherit;
    font-size: 0.9rem;
}

.edit-controls input[type="text"]::placeholder {
    color: #555;
}

.edit-controls button {
    width: auto;
    padding: 0.75rem 1.25rem;
    margin-top: 0;
    white-space: nowrap;
}

/* Responsive */
```

with:

```css
.viewer-controls button {
    flex: 1;
    margin-top: 0;
}

/* Responsive */
```

- [ ] **Step 7: Run the full suite to verify it's green**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS (the three deleted edit-endpoint tests are simply gone; nothing else references the removed route/schema/UI yet — `claude_vision.py`'s `interpret_color_edit`/`apply_color_edit` still exist and are still tested, just unreferenced by the router now)

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff check app/routers/skin.py app/models/schemas.py tests/test_api.py
git add app/routers/skin.py app/models/schemas.py app/static/index.html app/static/app.js app/static/style.css tests/test_api.py
git commit -m "feat: retire conversational color-edit route, schema, and frontend UI"
```

---

### Task 2: Build `app/services/photo_render.py` — the deterministic image-processing toolbox

A new, standalone module. Nothing existing imports it yet (Task 4 wires it in) — this task is purely additive.

**Files:**
- Create: `app/services/photo_render.py`
- Test: `tests/test_photo_render.py`

**Interfaces:**
- Produces: `Bbox = tuple[float, float, float, float]`, `crop_region(image: Image.Image, bbox: Bbox) -> Image.Image | None`, `downsample_dominant(region: Image.Image, target_h: int, target_w: int) -> list[list[tuple[int, int, int]]]`, `quantize_shared(rgb_grids: dict[str, list[list[tuple[int, int, int]]]], max_colors: int = 24) -> tuple[list[str], dict[str, list[list[int]]]]`, `extract_face_colors(face_crop: Image.Image) -> dict[str, str]`, `dominant_hex(grid: list[list[str]]) -> str`, `build_head_front(colors: dict[str, str], eye_shape: str) -> list[list[str]]` — all consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_photo_render.py`:

```python
"""Tests for photo_render — deterministic crop/downsample/quantize pixel-art rendering."""

from PIL import Image

from app.services.photo_render import (
    build_head_front,
    crop_region,
    dominant_hex,
    downsample_dominant,
    extract_face_colors,
    quantize_shared,
)


def _solid(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_crop_region_converts_fractional_bbox_to_pixels():
    img = _solid(100, 200, (255, 0, 0))
    crop = crop_region(img, (0.1, 0.2, 0.5, 0.6))
    assert crop is not None
    assert crop.size == (40, 80)  # (0.5-0.1)*100, (0.6-0.2)*200


def test_crop_region_clamps_out_of_bounds_bbox():
    img = _solid(100, 100, (0, 255, 0))
    crop = crop_region(img, (-0.5, -0.5, 1.5, 1.5))
    assert crop is not None
    assert crop.size == (100, 100)


def test_crop_region_returns_none_for_degenerate_bbox():
    img = _solid(100, 100, (0, 0, 255))
    assert crop_region(img, (0.5, 0.5, 0.5, 0.9)) is None  # zero width
    assert crop_region(img, (0.2, 0.5, 0.8, 0.5)) is None  # zero height
    assert crop_region(img, (0.9, 0.5, 0.2, 0.9)) is None  # x1 < x0


def test_downsample_dominant_picks_majority_color_per_cell():
    # Left half red, right half blue, downsample to 1x2 -> one red cell, one blue cell
    img = Image.new("RGB", (10, 4), (255, 0, 0))
    for x in range(5, 10):
        for y in range(4):
            img.putpixel((x, y), (0, 0, 255))

    out = downsample_dominant(img, target_h=1, target_w=2)

    assert out[0][0] == (255, 0, 0)
    assert out[0][1] == (0, 0, 255)


def test_downsample_dominant_output_shape():
    img = _solid(20, 30, (10, 20, 30))
    out = downsample_dominant(img, target_h=12, target_w=4)
    assert len(out) == 12
    assert all(len(row) == 4 for row in out)


def test_quantize_shared_produces_consistent_indices_and_capped_palette():
    grids = {
        "a": [[(255, 0, 0), (255, 0, 0)]],
        "b": [[(0, 0, 255)]],
    }
    palette, indices = quantize_shared(grids, max_colors=24)

    assert len(palette) <= 24
    assert len(indices["a"]) == 1 and len(indices["a"][0]) == 2
    assert len(indices["b"]) == 1 and len(indices["b"][0]) == 1
    # Same input color -> same index within a grid
    assert indices["a"][0][0] == indices["a"][0][1]
    # Different colors -> different indices
    assert indices["a"][0][0] != indices["b"][0][0]
    for hexval in palette:
        assert hexval.startswith("#") and len(hexval) == 7


def test_quantize_shared_preserves_grid_shapes():
    grids = {
        "wide": [[(1, 1, 1)] * 8 for _ in range(12)],
        "narrow": [[(2, 2, 2)] * 4 for _ in range(12)],
    }
    _palette, indices = quantize_shared(grids)
    assert len(indices["wide"]) == 12 and len(indices["wide"][0]) == 8
    assert len(indices["narrow"]) == 12 and len(indices["narrow"][0]) == 4


def _face_image(skin=(200, 160, 120), hair=(40, 20, 10), eye=(20, 20, 20), mouth=(180, 60, 60)):
    """8-row-band face image: rows 0-1 hair, rows 2-3 mostly skin with an eye dot,
    row 4 skin, rows 5-6 mostly skin with a mouth dot, row 7 skin (scaled up 10x)."""
    img = Image.new("RGB", (80, 80), skin)
    for y in range(0, 20):
        for x in range(80):
            img.putpixel((x, y), hair)
    for y in range(20, 40):
        for x in range(30, 34):
            img.putpixel((x, y), eye)
    for y in range(50, 70):
        for x in range(30, 50):
            img.putpixel((x, y), mouth)
    return img


def test_extract_face_colors_measures_all_four_bands():
    colors = extract_face_colors(_face_image())
    assert colors["skin_tone"] == "#C8A078"
    assert colors["hair_color"] == "#28140A"
    assert colors["eye_color"] == "#141414"
    assert colors["mouth_color"] == "#B43C3C"


def test_extract_face_colors_falls_back_to_skin_tone_for_degenerate_hair_band():
    # A face crop only 1px tall: every band collapses to the same single row.
    img = Image.new("RGB", (10, 1), (200, 160, 120))
    colors = extract_face_colors(img)
    assert colors["hair_color"] == colors["skin_tone"]


def test_dominant_hex_returns_most_common_color():
    grid = [["#111111", "#111111", "#222222"], ["#111111", "#333333", "#111111"]]
    assert dominant_hex(grid) == "#111111"


def test_build_head_front_shape_and_bands():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    grid = build_head_front(colors, eye_shape="round")

    assert len(grid) == 8 and all(len(row) == 8 for row in grid)
    assert grid[0] == ["#28140A"] * 8  # hair row
    assert grid[7] == ["#C8A078"] * 8  # neck row is skin
    assert "#141414" in grid[2]  # eye color present in eye row
    assert "#B43C3C" in grid[5]  # mouth color present in mouth row


def test_build_head_front_eye_width_depends_on_eye_shape():
    colors = {
        "skin_tone": "#C8A078",
        "hair_color": "#28140A",
        "eye_color": "#141414",
        "mouth_color": "#B43C3C",
    }
    narrow = build_head_front(colors, eye_shape="narrow")
    round_ = build_head_front(colors, eye_shape="round")

    narrow_eye_count = sum(cell == "#141414" for row in narrow[2:4] for cell in row)
    round_eye_count = sum(cell == "#141414" for row in round_[2:4] for cell in row)
    assert round_eye_count > narrow_eye_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_photo_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.photo_render'`

- [ ] **Step 3: Implement**

Create `app/services/photo_render.py`:

```python
"""Deterministic photo → pixel-art rendering: crop, downsample, quantize.

Replaces asking the Vision model to author per-pixel palette indices
directly (unreliable for precise, multi-region color detail) with real
image processing on the actual uploaded photo, guided only by AI-provided
region bounding boxes (see app.services.claude_vision). See
docs/superpowers/specs/2026-07-10-photo-derived-pixel-rendering-design.md
for the full design.
"""

from __future__ import annotations

from collections import Counter

from PIL import Image

Bbox = tuple[float, float, float, float]  # (x0, y0, x1, y1), fractions of image size

_HEAD_BAND_ROWS = 8  # head_front's row template is 8 rows tall


def crop_region(image: Image.Image, bbox: Bbox) -> Image.Image | None:
    """Crop `image` to `bbox` (fractional coords), clamped to bounds.

    Returns None if the resulting crop has zero width or height.
    """
    w, h = image.size
    x0, y0, x1, y1 = bbox
    left = max(0, min(w, round(x0 * w)))
    top = max(0, min(h, round(y0 * h)))
    right = max(0, min(w, round(x1 * w)))
    bottom = max(0, min(h, round(y1 * h)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _dominant_color(region: Image.Image) -> tuple[int, int, int]:
    """Most common RGB color in `region`, via a full-resolution histogram."""
    rgb = region.convert("RGB")
    colors = rgb.getcolors(maxcolors=rgb.width * rgb.height)
    return max(colors, key=lambda c: c[0])[1]


def downsample_dominant(
    region: Image.Image, target_h: int, target_w: int
) -> list[list[tuple[int, int, int]]]:
    """Downsample `region` to target_h x target_w, one dominant color per cell.

    Uses the most common color within each output cell's source tile rather
    than a blurred average, so a contrasting boundary (e.g. a red trim
    against a white base) stays crisp instead of blending into a muddy
    in-between color.
    """
    w, h = region.size
    out: list[list[tuple[int, int, int]]] = []
    for row in range(target_h):
        top = h * row // target_h
        bottom = max(top + 1, h * (row + 1) // target_h)
        out_row: list[tuple[int, int, int]] = []
        for col in range(target_w):
            left = w * col // target_w
            right = max(left + 1, w * (col + 1) // target_w)
            tile = region.crop((left, top, right, bottom))
            out_row.append(_dominant_color(tile))
        out.append(out_row)
    return out


def quantize_shared(
    rgb_grids: dict[str, list[list[tuple[int, int, int]]]], max_colors: int = 24
) -> tuple[list[str], dict[str, list[list[int]]]]:
    """Quantize all pixels across `rgb_grids` to one shared palette.

    A single shared quantization pass (rather than one per region) keeps
    e.g. skin tone consistent between an arm and a leg instead of each
    drifting to a slightly different quantized shade.

    Returns (palette hex list, {key: index grid}) — palette[i] is the hex
    for index i, and each returned index grid has the same shape as the
    input grid of the same key.
    """
    shapes: dict[str, tuple[int, int]] = {}
    all_pixels: list[tuple[int, int, int]] = []
    for key, grid in rgb_grids.items():
        shapes[key] = (len(grid), len(grid[0]) if grid else 0)
        for row in grid:
            all_pixels.extend(row)

    if not all_pixels:
        return [], {key: [] for key in rgb_grids}

    flat_img = Image.new("RGB", (len(all_pixels), 1))
    flat_img.putdata(all_pixels)
    quantized = flat_img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)

    palette_rgb = quantized.getpalette()[: max_colors * 3]
    palette = [
        "#{:02X}{:02X}{:02X}".format(*palette_rgb[i : i + 3])
        for i in range(0, len(palette_rgb), 3)
    ]
    flat_indices = list(quantized.getdata())

    out: dict[str, list[list[int]]] = {}
    pos = 0
    for key, (h, w) in shapes.items():
        out[key] = [flat_indices[pos + row * w : pos + (row + 1) * w] for row in range(h)]
        pos += h * w
    return palette, out


def _band(face_crop: Image.Image, row_start: int, row_end: int) -> Image.Image:
    """Crop the horizontal band spanning rows [row_start, row_end) of an 8-row template."""
    w, h = face_crop.size
    top = h * row_start // _HEAD_BAND_ROWS
    bottom = max(top + 1, h * row_end // _HEAD_BAND_ROWS)
    return face_crop.crop((0, top, w, bottom))


def _minority_color(band: Image.Image, fallback: str) -> str:
    """Quantize `band` to 2 colors and return the less-prevalent one as hex.

    The deciding criterion is prevalence, not darkness: a feature like an
    eye or mouth occupies less area than the surrounding skin within its
    band, so the minority cluster is the feature.
    """
    rgb = band.convert("RGB")
    quantized = rgb.quantize(colors=2, method=Image.Quantize.MEDIANCUT)
    counts = Counter(quantized.getdata())
    if len(counts) < 2:
        return fallback
    minority_idx = min(counts, key=lambda idx: counts[idx])
    palette = quantized.getpalette()
    r, g, b = palette[minority_idx * 3 : minority_idx * 3 + 3]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def extract_face_colors(face_crop: Image.Image) -> dict[str, str]:
    """Measure skin/hair/eye/mouth colors from a cropped face region.

    Uses head_front's existing row template (rows 0-1 hair, row 4 skin,
    rows 2-3 eyes, rows 5-6 mouth, out of 8 total rows) to sample real
    photo pixels, rather than asking the model to guess hex values.
    """
    skin_tone = "#{:02X}{:02X}{:02X}".format(*_dominant_color(_band(face_crop, 4, 5)))

    hair_pixels = _band(face_crop, 0, 2)
    if hair_pixels.size[0] * hair_pixels.size[1] == 0:
        hair_color = skin_tone
    else:
        hair_color = "#{:02X}{:02X}{:02X}".format(*_dominant_color(hair_pixels))

    eye_color = _minority_color(_band(face_crop, 2, 4), fallback=skin_tone)
    mouth_color = _minority_color(_band(face_crop, 5, 7), fallback=skin_tone)

    return {
        "skin_tone": skin_tone,
        "hair_color": hair_color,
        "eye_color": eye_color,
        "mouth_color": mouth_color,
    }


def dominant_hex(grid: list[list[str]]) -> str:
    """Most common hex color in an already-decoded hex grid."""
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0]


def build_head_front(colors: dict[str, str], eye_shape: str) -> list[list[str]]:
    """Build the 8x8 head_front hex grid from measured colors + the fixed template.

    Layout: row 0-1 hair/forehead, row 2-3 eyes, row 4 nose, row 5-6
    mouth/chin, row 7 neck. Eyes are 1px wide (centered at column 3) if
    eye_shape == "narrow", 2px wide (columns 3-4) if "round".
    """
    skin = colors["skin_tone"]
    hair = colors["hair_color"]
    eye = colors["eye_color"]
    mouth = colors["mouth_color"]

    grid = [[skin] * 8 for _ in range(8)]
    grid[0] = [hair] * 8
    grid[1] = [hair] * 8

    eye_cols = (3, 4) if eye_shape == "round" else (3,)
    for row in (2, 3):
        for col in eye_cols:
            grid[row][col] = eye

    for row in (5, 6):
        for col in (2, 3, 4, 5):
            grid[row][col] = mouth

    return grid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_photo_render.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/photo_render.py tests/test_photo_render.py
git add app/services/photo_render.py tests/test_photo_render.py
git commit -m "feat: add photo_render — deterministic crop/downsample/quantize pixel rendering"
```

---

### Task 3: `procedural.py` — shrink `AI_GENERATED_KEYS` to 6, add head flat-fill, fix the "invisible region" hole

Only `head_front` is genuinely photo-derived now; `head_top`/`head_back`/`head_left`/`head_right`/`head_bottom` become a flat procedural fill (a front photo never shows the back/top/sides of someone's head anyway).

This task also fixes a real gap that Task 4 would otherwise introduce: `generate_procedural_regions` currently computes `skip = AI_GENERATED_KEYS | exclude_keys`, which **unconditionally** excludes every front face (head_front + the 5 part fronts) from procedural generation, regardless of whether real data for it actually exists. That's harmless today because a front face is *always* present (the old mandatory-AI-emission design guaranteed it) — but Task 4 introduces `visible: false` regions (e.g. a photo that doesn't show the legs), where a front face's data is legitimately absent. With the unconditional exclusion, that front face would never be produced by *anything* — not by the photo pipeline (nothing to crop) and not by the procedural fallback (unconditionally skipped) — leaving a permanent transparent hole in the assembled PNG where that body part's front should be.

The fix: `skip` becomes just `exclude_keys` — callers are responsible for passing `exclude_keys=frozenset(pixel_data.keys())` (which `generate_skin_data`/`apply_color_edit` already do), so a face is skipped exactly when real data for it already exists, front or not. A front face with no real data falls through to the same flat-color fallback back/left/right already use. `AI_GENERATED_KEYS` itself is unchanged in purpose (still describes which faces are meant to be photo/AI-derived, still consumed by `claude_vision.py`) — it's just no longer baked into this function's own skip logic.

**Files:**
- Modify: `app/services/procedural.py`
- Modify: `tests/test_procedural.py` (every existing call to `generate_procedural_regions` that expects front faces to be skipped now passes `exclude_keys` explicitly, matching real caller behavior — see Step 1)

**Interfaces:**
- Produces: `AI_GENERATED_KEYS` now `{"head_front", "body_front", "right_arm_front", "left_arm_front", "right_leg_front", "left_leg_front"}` (6 entries) — consumed by Task 4. `generate_procedural_regions`'s signature is unchanged; a face (including any front face) is skipped if and only if its key is in the caller-supplied `exclude_keys` — it no longer consults `AI_GENERATED_KEYS` internally. It flat-fills `head_top`/`head_back`/`head_left`/`head_right`/`head_bottom` (and `head_front` itself, if not excluded) from `colors["head_fill_color"]`, falling back to `DEFAULT_COLORS["head_fill_color"]` if absent.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_procedural.py` entirely with:

```python
"""Tests for procedural — front-row propagation and flat-fill for non-photo-derived faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    generate_procedural_regions,
    propagate_front_row,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_ai_generated_keys_is_head_front_plus_five_photo_derived_fronts():
    assert AI_GENERATED_KEYS == {
        "head_front",
        "body_front",
        "right_arm_front",
        "left_arm_front",
        "right_leg_front",
        "left_leg_front",
    }


def test_propagate_front_row_uses_middle_column_per_row():
    front = [
        ["#111111", "#222222", "#333333"],
        ["#AAAAAA", "#BBBBBB", "#CCCCCC"],
    ]
    out = propagate_front_row(front, target_width=4)
    assert out[0] == ["#222222"] * 4
    assert out[1] == ["#BBBBBB"] * 4


def test_propagate_front_row_preserves_height():
    front = [["#111111"], ["#222222"], ["#333333"]]
    out = propagate_front_row(front, target_width=2)
    assert len(out) == 3


def test_propagate_front_row_target_width_independent_of_front_width():
    front = [["#111111", "#222222", "#333333", "#444444"]]  # 4 cols wide
    out = propagate_front_row(front, target_width=8)
    assert len(out[0]) == 8


def _pixel_data_with_fronts(**overrides: list[list[str]]) -> dict[str, list[list[str]]]:
    data = {
        "head_front": [["#C4A882"] * 8 for _ in range(8)],
        "body_front": [["#3355AA"] * 8 for _ in range(12)],
        "right_arm_front": [["#C4A882"] * 4 for _ in range(12)],
        "left_arm_front": [["#C4A882"] * 4 for _ in range(12)],
        "right_leg_front": [["#223355"] * 4 for _ in range(12)],
        "left_leg_front": [["#223355"] * 4 for _ in range(12)],
    }
    data.update(overrides)
    return data


def _colors(**overrides: str) -> dict[str, str]:
    base = {
        "shirt_main": "#3355AA",
        "arm_main": "#C4A882",
        "pants_main": "#223355",
        "shoe_color": "#2B2B2B",
        "head_fill_color": "#5B3A1A",
    }
    base.update(overrides)
    return base


def test_generate_procedural_regions_covers_all_non_provided_faces():
    """With every front provided (and excluded), procedural fill covers exactly
    the remaining 30 of the 36 base regions (36 total - 6 excluded fronts)."""
    pixel_data = _pixel_data_with_fronts()
    regions = get_all_regions("classic")
    all_base_keys = {
        key
        for key, (group, _face) in PIXEL_KEY_MAP.items()
        if group in ("head", "body", "right_arm", "left_arm", "right_leg", "left_leg")
    }
    expected_keys = all_base_keys - set(pixel_data.keys())

    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )

    assert set(out.keys()) == expected_keys
    for key, grid in out.items():
        group, face = PIXEL_KEY_MAP[key]
        rect = regions[group][face]
        assert len(grid) == rect.h
        assert all(len(row) == rect.w for row in grid)


def test_generate_procedural_regions_never_overwrites_excluded_faces():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    assert set(pixel_data.keys()).isdisjoint(out.keys())


def test_generate_procedural_regions_wrap_faces_match_front_rows():
    front = [[f"#{i:02X}{i:02X}{i:02X}"] * 8 for i in range(12)]  # row i -> shade #i,i,i
    pixel_data = _pixel_data_with_fronts(body_front=front)
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    for row_idx, row in enumerate(front):
        row_color = row[len(row) // 2]
        assert out["body_back"][row_idx] == [row_color] * 8
        assert out["body_left"][row_idx] == [row_color] * 4
        assert out["body_right"][row_idx] == [row_color] * 4


def test_generate_procedural_regions_top_bottom_are_flat_named_color():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    assert all(cell == "#3355AA" for row in out["body_top"] for cell in row)
    assert all(cell == "#3355AA" for row in out["body_bottom"] for cell in row)
    assert all(cell == "#C4A882" for row in out["right_arm_top"] for cell in row)
    assert all(cell == "#223355" for row in out["right_leg_top"] for cell in row)


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data,
        _colors(shoe_color="#ABCDEF"),
        "classic",
        exclude_keys=frozenset(pixel_data.keys()),
    )
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_covers_head_wrap_faces():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "classic", exclude_keys=frozenset(pixel_data.keys())
    )
    for face in ("top", "back", "left", "right", "bottom"):
        assert f"head_{face}" in out


def test_generate_procedural_regions_head_faces_are_flat_head_fill_color():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data,
        _colors(head_fill_color="#ABCDEF"),
        "classic",
        exclude_keys=frozenset(pixel_data.keys()),
    )
    for face in ("top", "back", "left", "right", "bottom"):
        grid = out[f"head_{face}"]
        assert all(cell == "#ABCDEF" for row in grid for cell in row)


def test_generate_procedural_regions_missing_front_gets_flat_fill_not_omitted():
    """A body part not visible in the photo (no front data, and thus not in
    exclude_keys) must still get its front face flat-filled -- not left
    entirely absent, which would render as a transparent hole."""
    pixel_data = _pixel_data_with_fronts()
    del pixel_data["right_leg_front"]  # simulates: legs weren't visible in the photo

    out = generate_procedural_regions(
        pixel_data,
        _colors(pants_main="#654321"),
        "classic",
        exclude_keys=frozenset(pixel_data.keys()),
    )

    assert "right_leg_front" in out
    assert all(cell == "#654321" for row in out["right_leg_front"] for cell in row)
    # Its wrap faces fall back to the same flat color too, since there's no
    # real front data to propagate from.
    assert all(cell == "#654321" for row in out["right_leg_back"] for cell in row)


def test_generate_procedural_regions_slim_arm_width():
    pixel_data = _pixel_data_with_fronts()
    out = generate_procedural_regions(
        pixel_data, _colors(), "slim", exclude_keys=frozenset(pixel_data.keys())
    )
    assert len(out["right_arm_top"][0]) == 3
    assert len(out["left_arm_top"][0]) == 3


def test_generate_procedural_regions_falls_back_to_defaults_when_nothing_provided():
    """Nothing provided at all (e.g. a totally failed generation) -> every
    face, including all 6 fronts, gets a flat default-color fill rather
    than a transparent skin."""
    out = generate_procedural_regions({}, {}, "classic")
    assert len(out) == 36
    assert out["body_front"][0][0]  # non-empty hex string, DEFAULT_COLORS fallback applied
    assert out["head_front"][0][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_procedural.py -v`
Expected: FAIL — `test_ai_generated_keys_is_head_front_plus_five_photo_derived_fronts` fails (`AI_GENERATED_KEYS` still has 11 entries); `test_generate_procedural_regions_missing_front_gets_flat_fill_not_omitted` and the head-wrap tests fail (`KeyError`/missing keys); several others fail because passing explicit `exclude_keys` against the *current* `skip = AI_GENERATED_KEYS | exclude_keys` logic still works today by coincidence (AI_GENERATED_KEYS already covers the same fronts) but will only be a real behavioral guarantee — not a coincidence — once Step 3 lands.

- [ ] **Step 3: Implement**

In `app/services/procedural.py`, replace:

```python
AI_GENERATED_KEYS = {
    "head_front",
    "head_back",
    "head_top",
    "head_bottom",
    "head_left",
    "head_right",
    "body_front",
    "right_arm_front",
    "left_arm_front",
    "right_leg_front",
    "left_leg_front",
}

DEFAULT_COLORS: dict[str, str] = {
    "shirt_main": "#3B5998",
    "arm_main": "#C4A882",
    "pants_main": "#1A1A3E",
    "shoe_color": "#2B2B2B",
}

# (part group name, that part's mandatory front-face key, named color used
# for its top/bottom caps and as a fallback if the front grid is missing).
_PARTS = (
    ("body", "body_front", "shirt_main"),
    ("right_arm", "right_arm_front", "arm_main"),
    ("left_arm", "left_arm_front", "arm_main"),
    ("right_leg", "right_leg_front", "pants_main"),
    ("left_leg", "left_leg_front", "pants_main"),
)


def propagate_front_row(front_grid: list[list[str]], target_width: int) -> list[list[str]]:
    """Build a same-height grid where row N is a flat fill of front_grid[N]'s middle column.

    Used for the "wrap" faces (back/left/right) of a body part, so a clothing
    boundary visible on the front (e.g. a sleeve ending partway down the arm)
    stays consistent all the way around the limb, without any fabricated shading.
    """
    return [[row[len(row) // 2]] * target_width for row in front_grid]


def generate_procedural_regions(
    pixel_data: dict[str, list[list[str]]],
    colors: dict[str, str],
    model: ModelType,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, list[list[str]]]:
    """Build hex pixel grids for every face not covered by AI generation.

    `pixel_data` is the AI-decoded hex grids collected so far (must contain
    each part's front face for row propagation to apply; falls back to a
    flat named-color fill for a part whose front face isn't present — e.g. a
    skin persisted before this mandatory-front-faces change shipped).
    """
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    skip = AI_GENERATED_KEYS | exclude_keys
    out: dict[str, list[list[str]]] = {}

    for part, front_key, color_key in _PARTS:
        front_grid = pixel_data.get(front_key)
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in skip:
                continue
            if face == "bottom" and part in ("right_leg", "left_leg"):
                out[key] = [[c["shoe_color"]] * rect.w for _ in range(rect.h)]
            elif face in ("top", "bottom"):
                out[key] = [[c[color_key]] * rect.w for _ in range(rect.h)]
            elif front_grid is not None:
                out[key] = propagate_front_row(front_grid, rect.w)
            else:
                out[key] = [[c[color_key]] * rect.w for _ in range(rect.h)]

    return out
```

with:

```python
AI_GENERATED_KEYS = {
    "head_front",
    "body_front",
    "right_arm_front",
    "left_arm_front",
    "right_leg_front",
    "left_leg_front",
}

DEFAULT_COLORS: dict[str, str] = {
    "shirt_main": "#3B5998",
    "arm_main": "#C4A882",
    "pants_main": "#1A1A3E",
    "shoe_color": "#2B2B2B",
    "head_fill_color": "#5B3A1A",
}

# (part group name, that part's mandatory front-face key, named color used
# for its top/bottom caps and as a fallback if the front grid is missing).
_PARTS = (
    ("body", "body_front", "shirt_main"),
    ("right_arm", "right_arm_front", "arm_main"),
    ("left_arm", "left_arm_front", "arm_main"),
    ("right_leg", "right_leg_front", "pants_main"),
    ("left_leg", "left_leg_front", "pants_main"),
)


def propagate_front_row(front_grid: list[list[str]], target_width: int) -> list[list[str]]:
    """Build a same-height grid where row N is a flat fill of front_grid[N]'s middle column.

    Used for the "wrap" faces (back/left/right) of a body part, so a clothing
    boundary visible on the front (e.g. a sleeve ending partway down the arm)
    stays consistent all the way around the limb, without any fabricated shading.
    """
    return [[row[len(row) // 2]] * target_width for row in front_grid]


def generate_procedural_regions(
    pixel_data: dict[str, list[list[str]]],
    colors: dict[str, str],
    model: ModelType,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, list[list[str]]]:
    """Build hex pixel grids for every face not already present in `pixel_data`.

    A face (including a front face) is skipped exactly when its key is in
    `exclude_keys` — callers pass `exclude_keys=frozenset(pixel_data.keys())`
    so "already has real data" and "skip" always mean the same thing,
    whether that data came from the photo pipeline or not at all (a body
    part invisible in the source photo has no front data, isn't excluded,
    and falls through to the same flat-color fallback its wrap faces use —
    it does NOT get silently omitted, which would otherwise leave a
    transparent hole in the assembled skin).
    """
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    out: dict[str, list[list[str]]] = {}

    for face, rect in regions["head"].items():
        key = f"head_{face}"
        if key in exclude_keys:
            continue
        out[key] = [[c["head_fill_color"]] * rect.w for _ in range(rect.h)]

    for part, front_key, color_key in _PARTS:
        front_grid = pixel_data.get(front_key)
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in exclude_keys:
                continue
            if face == "bottom" and part in ("right_leg", "left_leg"):
                out[key] = [[c["shoe_color"]] * rect.w for _ in range(rect.h)]
            elif face in ("top", "bottom"):
                out[key] = [[c[color_key]] * rect.w for _ in range(rect.h)]
            elif front_grid is not None:
                out[key] = propagate_front_row(front_grid, rect.w)
            else:
                out[key] = [[c[color_key]] * rect.w for _ in range(rect.h)]

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_procedural.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full suite to verify it's still green**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS. The current (pre-Task-4) `claude_vision.py` always passes `exclude_keys=frozenset(pixel_data.keys())`, and its `pixel_data` always already contains every front face plus all 6 head faces (decoded either as mandatory or, for the 5 non-front head faces, via the existing optional-detail-face path) — so `exclude_keys` alone reproduces the exact same skip set the old `AI_GENERATED_KEYS | exclude_keys` produced, and nothing behaves differently yet. The fix only changes behavior once Task 4 introduces responses where a front face's data is genuinely absent.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check app/services/procedural.py tests/test_procedural.py
git add app/services/procedural.py tests/test_procedural.py
git commit -m "feat: shrink AI_GENERATED_KEYS to 6, flat-fill head faces, fix invisible-region hole"
```

---

### Task 4: Rewrite `claude_vision.py` — region-localization prompt + photo_render orchestration

Replaces the old "AI emits pixel indices" pipeline with "AI locates 6 regions, `photo_render.py` produces the pixels." Removes `interpret_color_edit`, `apply_color_edit`, `EDIT_MAX_TOKENS`, and the `PALETTE_ROLES`/`MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE` imports (nothing else in this file needs them once the fixed-role palette is gone). `generate_skin_data` keeps its external signature and return shape (`tuple[pixel_data, metadata, persist_state]`).

**Files:**
- Modify: `app/services/claude_vision.py` (full rewrite)
- Modify: `tests/test_claude_vision.py` (full rewrite)

**Interfaces:**
- Consumes: `AI_GENERATED_KEYS`, `generate_procedural_regions` (Task 3, `app.services.procedural`); `crop_region`, `downsample_dominant`, `quantize_shared`, `extract_face_colors`, `dominant_hex`, `build_head_front` (Task 2, `app.services.photo_render`); `decode_indexed_grid` (kept, unchanged in shape, still in this file).
- Produces: `generate_skin_data(image_bytes, media_type, model="classic", style_notes="", ai_model="flash", max_retries=2) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]` (unchanged signature/shape) — consumed by `app/routers/skin.py` (already wired in Task 1, no further change needed there).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_claude_vision.py` entirely with:

```python
"""Tests for claude_vision — region-localization validation, photo-render orchestration."""

import io

from PIL import Image

from app.config import settings
from app.services.claude_vision import (
    _build_prompt,
    _prepare_image,
    _resolve_model_name,
    _validate_regions_response,
    generate_skin_data,
)

REGION_KEYS = ("head", "torso", "right_arm", "left_arm", "right_leg", "left_leg")


def test_resolve_model_name_flash():
    assert _resolve_model_name("flash") == settings.gemini_model_flash


def test_resolve_model_name_flash_lite():
    assert _resolve_model_name("flash-lite") == settings.gemini_model_flash_lite


def _valid_regions_response(**overrides) -> dict:
    data = {
        "hair_style": "short",
        "eye_shape": "round",
        "regions": {
            "head": {"visible": True, "bbox": [0.30, 0.05, 0.68, 0.35]},
            "torso": {"visible": True, "bbox": [0.20, 0.30, 0.75, 0.70]},
            "right_arm": {"visible": True, "bbox": [0.05, 0.30, 0.25, 0.65]},
            "left_arm": {"visible": True, "bbox": [0.70, 0.30, 0.95, 0.65]},
            "right_leg": {"visible": False},
            "left_leg": {"visible": False},
        },
    }
    data.update(overrides)
    return data


def test_validate_regions_response_complete():
    hair_style, eye_shape, regions, ok = _validate_regions_response(_valid_regions_response())
    assert ok is True
    assert hair_style == "short"
    assert eye_shape == "round"
    assert regions["head"] == (0.30, 0.05, 0.68, 0.35)
    assert regions["right_leg"] is None
    assert regions["left_leg"] is None


def test_validate_regions_response_missing_hair_style_is_incomplete():
    data = _valid_regions_response()
    del data["hair_style"]
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False


def test_validate_regions_response_invalid_eye_shape_is_incomplete():
    data = _valid_regions_response(eye_shape="squinty")
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False


def test_validate_regions_response_missing_regions_key_is_incomplete():
    data = _valid_regions_response()
    del data["regions"]
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False


def test_validate_regions_response_missing_one_region_entry_is_incomplete():
    data = _valid_regions_response()
    del data["regions"]["left_leg"]
    _hs, _es, _regions, ok = _validate_regions_response(data)
    assert ok is False  # regions dict must have all 6 keys to be structurally valid


def test_validate_regions_response_malformed_single_bbox_degrades_to_not_visible():
    data = _valid_regions_response()
    data["regions"]["torso"] = {"visible": True, "bbox": [0.9, 0.5, 0.1, 0.9]}  # x1 < x0
    _hs, _es, regions, ok = _validate_regions_response(data)
    assert ok is True  # whole response still valid
    assert regions["torso"] is None  # this one region degrades gracefully


def test_validate_regions_response_out_of_range_bbox_degrades_to_not_visible():
    data = _valid_regions_response()
    data["regions"]["right_arm"] = {"visible": True, "bbox": [0.1, 0.1, 1.5, 0.5]}
    _hs, _es, regions, ok = _validate_regions_response(data)
    assert ok is True
    assert regions["right_arm"] is None


def test_prepare_image_downscales_large_image():
    img = Image.new("RGB", (2000, 1000), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")

    out_bytes, media_type = _prepare_image(buf.getvalue())

    assert media_type == "image/jpeg"
    out_img = Image.open(io.BytesIO(out_bytes))
    assert max(out_img.size) <= 768


def test_prepare_image_leaves_small_image_dimensions_alone():
    img = Image.new("RGB", (100, 50), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")

    out_bytes, _media_type = _prepare_image(buf.getvalue())

    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.size == (100, 50)


def test_build_prompt_mentions_all_six_regions_and_categorical_fields():
    prompt = _build_prompt()
    for key in REGION_KEYS:
        assert key in prompt
    assert "hair_style" in prompt
    assert "eye_shape" in prompt
    assert "bbox" in prompt
    assert "visible" in prompt


def _portrait_photo_bytes() -> bytes:
    """A synthetic 'portrait' with distinct colors per body region, so the
    orchestration test can assert real photo-render output, not just shapes."""
    img = Image.new("RGB", (200, 400), (240, 240, 240))
    for y in range(20, 140):  # head band: hair top, skin below
        for x in range(60, 140):
            img.putpixel((x, y), (40, 20, 10) if y < 60 else (200, 160, 120))
    for y in range(140, 280):  # torso: red
        for x in range(40, 160):
            img.putpixel((x, y), (200, 30, 30))
    for y in range(140, 280):  # right arm: green (image-left = character's right)
        for x in range(10, 40):
            img.putpixel((x, y), (30, 200, 30))
    for y in range(140, 280):  # left arm: blue
        for x in range(160, 190):
            img.putpixel((x, y), (30, 30, 200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


async def test_generate_skin_data_end_to_end_with_mocked_vision_call(monkeypatch):
    async def fake_call_vision(*_args, **_kwargs):
        return {
            "hair_style": "short",
            "eye_shape": "round",
            "regions": {
                "head": {"visible": True, "bbox": [0.30, 0.05, 0.70, 0.35]},
                "torso": {"visible": True, "bbox": [0.20, 0.35, 0.80, 0.70]},
                "right_arm": {"visible": True, "bbox": [0.05, 0.35, 0.20, 0.70]},
                "left_arm": {"visible": True, "bbox": [0.80, 0.35, 0.95, 0.70]},
                "right_leg": {"visible": False},
                "left_leg": {"visible": False},
            },
        }

    import app.services.claude_vision as claude_vision

    monkeypatch.setattr(claude_vision, "_call_vision", fake_call_vision)

    pixel_data, metadata, persist_state = await claude_vision.generate_skin_data(
        image_bytes=_portrait_photo_bytes(),
        media_type="image/png",
        model="classic",
    )

    # Photo-derived mandatory faces are present with correct shapes.
    assert len(pixel_data["head_front"]) == 8 and len(pixel_data["head_front"][0]) == 8
    assert len(pixel_data["body_front"]) == 12 and len(pixel_data["body_front"][0]) == 8
    assert len(pixel_data["right_arm_front"]) == 12 and len(pixel_data["right_arm_front"][0]) == 4

    # Legs weren't visible in the mocked regions -> procedural flat fill, not crashed.
    assert len(pixel_data["right_leg_front"]) == 12

    # Every base region (36 total for classic) ends up populated.
    assert len(pixel_data) == 36

    assert metadata["skin_tone"]
    assert metadata["hair_color"]
    assert metadata["ai_model"] == "flash"

    assert persist_state["model"] == "classic"
    assert isinstance(persist_state["palette"], list) and len(persist_state["palette"]) > 0
    assert set(persist_state["pixel_grids"].keys()) == {
        "head_front",
        "body_front",
        "right_arm_front",
        "left_arm_front",
        "right_leg_front",
        "left_leg_front",
    }


async def test_generate_skin_data_retries_on_incomplete_response(monkeypatch):
    calls = {"n": 0}

    async def flaky_call_vision(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"hair_style": "short"}  # missing eye_shape/regions -> incomplete
        return {
            "hair_style": "short",
            "eye_shape": "narrow",
            "regions": {
                "head": {"visible": True, "bbox": [0.30, 0.05, 0.70, 0.35]},
                "torso": {"visible": False},
                "right_arm": {"visible": False},
                "left_arm": {"visible": False},
                "right_leg": {"visible": False},
                "left_leg": {"visible": False},
            },
        }

    import app.services.claude_vision as claude_vision

    monkeypatch.setattr(claude_vision, "_call_vision", flaky_call_vision)

    _pixel_data, metadata, _persist_state = await claude_vision.generate_skin_data(
        image_bytes=_portrait_photo_bytes(),
        media_type="image/png",
        model="classic",
    )

    assert calls["n"] == 2
    assert metadata["skin_tone"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v`
Expected: FAIL — `ImportError: cannot import name '_validate_regions_response'` (and `_build_prompt()` currently requires a `model` argument, doesn't yet accept zero args)

- [ ] **Step 3: Implement**

Replace `app/services/claude_vision.py` entirely with:

```python
"""Gemini AI pipeline: single Vision call → region bounding boxes + a
couple of categorical fields, then deterministic Pillow rendering (see
app.services.photo_render) produces the actual pixels.

Uses the Gemini API's OpenAI-compatible endpoint. Callers pick between
gemini-2.5-flash and gemini-2.5-flash-lite per request via `ai_model`.

The model is asked only to locate 6 body-part regions (head, torso,
right/left arm, right/left leg) as bounding boxes, plus classify
hair_style and eye_shape — a much smaller, more reliable ask than
directly authoring per-pixel palette indices. See
docs/superpowers/specs/2026-07-10-photo-derived-pixel-rendering-design.md
for the full design and rationale.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings
from app.services.procedural import AI_GENERATED_KEYS, generate_procedural_regions
from app.services.photo_render import (
    Bbox,
    build_head_front,
    crop_region,
    dominant_hex,
    downsample_dominant,
    extract_face_colors,
    quantize_shared,
)
from app.services.skin_map import ModelType, get_all_regions

logger = logging.getLogger(__name__)

AIModel = Literal["flash", "flash-lite"]

REGION_KEYS = ("head", "torso", "right_arm", "left_arm", "right_leg", "left_leg")
HAIR_STYLES = ("short", "long", "bald", "hat", "helmet")
EYE_SHAPES = ("narrow", "round")

# Maps a "regions" key to the skin_map part name + the AI_GENERATED_KEYS
# front face it produces, for every region except "head" (handled
# separately since it's colors-plus-template, not crop-and-quantize).
_PHOTO_PARTS = (
    ("torso", "body", "body_front"),
    ("right_arm", "right_arm", "right_arm_front"),
    ("left_arm", "left_arm", "left_arm_front"),
    ("right_leg", "right_leg", "right_leg_front"),
    ("left_leg", "left_leg", "left_leg_front"),
)


def _resolve_model_name(ai_model: AIModel) -> str:
    return (
        settings.gemini_model_flash_lite
        if ai_model == "flash-lite"
        else settings.gemini_model_flash
    )


def _build_prompt() -> str:
    return """\
You are looking at a photo to prepare it for turning into a Minecraft skin. \
Do NOT draw or describe any pixels — you only need to locate a few regions \
and classify two features.

Output ONLY a JSON object (no markdown, no explanation) with this structure:

{
  "hair_style": "short|long|bald|hat|helmet",
  "eye_shape": "narrow|round",
  "regions": {
    "head":      {"visible": true, "bbox": [x0, y0, x1, y1]},
    "torso":     {"visible": true, "bbox": [x0, y0, x1, y1]},
    "right_arm": {"visible": true, "bbox": [x0, y0, x1, y1]},
    "left_arm":  {"visible": true, "bbox": [x0, y0, x1, y1]},
    "right_leg": {"visible": true, "bbox": [x0, y0, x1, y1]},
    "left_leg":  {"visible": true, "bbox": [x0, y0, x1, y1]}
  }
}

RULES:
- bbox is [x0, y0, x1, y1], each a fraction from 0 to 1 of the image's \
width/height, top-left origin, so x1 > x0 and y1 > y0.
- "head" should tightly bound just the face (forehead to chin), not the \
whole head/hair.
- "torso" should bound the visible chest/shirt area.
- "right_arm"/"left_arm" are the character's own right/left (mirrored from \
the viewer's perspective) — bound the upper arm/forearm area, not the hand.
- "right_leg"/"left_leg" similarly bound the visible thigh/shin area.
- If a region isn't visible in the photo at all (e.g. a headshot has no \
visible legs), set "visible": false and omit "bbox" for that region — this \
is expected and normal, not an error.
- eye_shape: "narrow" if the eyes read as small/narrow in the photo, \
"round" if they read as large/round.
- hair_style: pick the closest category based on what's visible."""


def _prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Downscale to max_image_dimension and re-encode as JPEG to cut vision tokens."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    max_dim = settings.max_image_dimension
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _get_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible async client for the Gemini API."""
    return AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)  # type: ignore[no-any-return]


def decode_indexed_grid(grid: list[list[Any]], palette: list[str]) -> list[list[str]]:
    """Map a grid of palette indices to hex colors, clamping bad indices to palette[0]."""
    return [
        [
            palette[cell] if isinstance(cell, int) and 0 <= cell < len(palette) else palette[0]
            for cell in row
        ]
        for row in grid
    ]


def _valid_bbox(value: Any) -> Bbox | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    if not all(isinstance(v, (int, float)) for v in value):
        return None
    x0, y0, x1, y1 = (float(v) for v in value)
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _validate_regions_response(
    data: dict[str, Any],
) -> tuple[str, str, dict[str, Bbox | None], bool]:
    """Validate the model's region-localization response.

    A malformed individual region degrades to "not visible" rather than
    failing the whole response; only a missing/invalid hair_style,
    eye_shape, or a structurally-malformed `regions` dict fails it.

    Returns (hair_style, eye_shape, {region_key: bbox_or_None}, is_complete).
    """
    hair_style = data.get("hair_style")
    eye_shape = data.get("eye_shape")
    regions_raw = data.get("regions")

    structurally_ok = (
        hair_style in HAIR_STYLES
        and eye_shape in EYE_SHAPES
        and isinstance(regions_raw, dict)
        and set(regions_raw.keys()) == set(REGION_KEYS)
    )
    if not structurally_ok:
        return "", "", {key: None for key in REGION_KEYS}, False

    regions: dict[str, Bbox | None] = {}
    for key in REGION_KEYS:
        entry = regions_raw[key]
        if isinstance(entry, dict) and entry.get("visible") is True:
            regions[key] = _valid_bbox(entry.get("bbox"))
        else:
            regions[key] = None

    return hair_style, eye_shape, regions, True


async def _call_vision(
    image_bytes: bytes,
    media_type: str,
    style_notes: str,
    ai_model: AIModel,
) -> dict[str, Any]:
    client = _get_client()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    user_content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        },
        {"type": "text", "text": _build_prompt()},
    ]
    if style_notes:
        user_content.append(
            {"type": "text", "text": f"\nAdditional style notes: {style_notes}"}
        )

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=800,
        temperature=0.2,
        # Gemini 2.5 defaults to "thinking" mode, which eats into max_tokens
        # before any visible output — without this the response gets cut off.
        reasoning_effort="none",
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.choices[0].message.content or ""
    return _extract_json(text)


def _render_photo(
    photo: Image.Image,
    regions: dict[str, Bbox | None],
    hair_style: str,
    eye_shape: str,
    model: ModelType,
) -> tuple[dict[str, list[list[str]]], dict[str, list[list[int]]], list[str], dict[str, str]]:
    """Build pixel_data/raw index grids/palette/named-colors from a located photo.

    Returns (pixel_data_hex, raw_index_grids, palette, named_colors) where
    named_colors has whichever of shirt_main/arm_main/pants_main/
    head_fill_color could be derived from a visible region (missing keys
    fall back to generate_procedural_regions's own defaults).
    """
    face_regions = get_all_regions(model)
    rgb_grids: dict[str, list[list[tuple[int, int, int]]]] = {}
    shapes: dict[str, tuple[int, int]] = {}

    for region_key, part, front_key in _PHOTO_PARTS:
        bbox = regions[region_key]
        if bbox is None:
            continue
        crop = crop_region(photo, bbox)
        if crop is None:
            continue
        rect = face_regions[part]["front"]
        rgb_grids[front_key] = downsample_dominant(crop, rect.h, rect.w)
        shapes[front_key] = (rect.h, rect.w)

    shared_palette, shared_indices = quantize_shared(rgb_grids)

    named_colors: dict[str, str] = {}
    pixel_data: dict[str, list[list[str]]] = {}
    raw_grids: dict[str, list[list[int]]] = {}
    color_key_by_front = {
        "body_front": "shirt_main",
        "right_arm_front": "arm_main",
        "left_arm_front": "arm_main",
        "right_leg_front": "pants_main",
        "left_leg_front": "pants_main",
    }
    for front_key, index_grid in shared_indices.items():
        pixel_data[front_key] = decode_indexed_grid(index_grid, shared_palette)
        raw_grids[front_key] = index_grid
        color_key = color_key_by_front[front_key]
        if color_key not in named_colors:
            named_colors[color_key] = dominant_hex(pixel_data[front_key])

    head_bbox = regions["head"]
    head_crop = crop_region(photo, head_bbox) if head_bbox is not None else None
    if head_crop is not None:
        face_colors = extract_face_colors(head_crop)
    else:
        face_colors = {
            "skin_tone": "#C4A882",
            "hair_color": "#5B3A1A",
            "eye_color": "#3B5998",
            "mouth_color": "#AA5555",
        }

    named_colors["skin_tone"] = face_colors["skin_tone"]
    named_colors["hair_color"] = face_colors["hair_color"]
    named_colors["head_fill_color"] = (
        face_colors["skin_tone"] if hair_style == "bald" else face_colors["hair_color"]
    )

    head_hex_grid = build_head_front(face_colors, eye_shape)
    head_local_palette = [
        face_colors["skin_tone"],
        face_colors["hair_color"],
        face_colors["eye_color"],
        face_colors["mouth_color"],
    ]
    offset = len(shared_palette)
    head_index_grid = [
        [offset + head_local_palette.index(hexval) for hexval in row] for row in head_hex_grid
    ]
    pixel_data["head_front"] = head_hex_grid
    raw_grids["head_front"] = head_index_grid

    full_palette = shared_palette + head_local_palette
    return pixel_data, raw_grids, full_palette, named_colors


async def generate_skin_data(
    image_bytes: bytes,
    media_type: str,
    model: ModelType = "classic",
    style_notes: str = "",
    ai_model: AIModel = "flash",
    max_retries: int = 2,
) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]:
    """Full pipeline: photo → one Vision call (region localization) → photo_render pixels.

    Returns:
        (pixel_data covering all base regions, metadata, persist_state).
        `persist_state` is ready to pass to skin_store.save_skin_state(skin_id, **persist_state).
    """
    prepped_bytes, prepped_media_type = _prepare_image(image_bytes)
    photo = Image.open(io.BytesIO(prepped_bytes)).convert("RGB")

    hair_style, eye_shape, regions, is_complete = "", "", {}, False
    raw: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        raw = await _call_vision(prepped_bytes, prepped_media_type, style_notes, ai_model)
        hair_style, eye_shape, regions, is_complete = _validate_regions_response(raw)
        if is_complete:
            break
        logger.warning("Attempt %d: region-localization response incomplete", attempt + 1)

    if not is_complete:
        hair_style, eye_shape = "short", "round"
        regions = {key: None for key in REGION_KEYS}

    pixel_data, raw_grids, palette, named_colors = _render_photo(
        photo, regions, hair_style, eye_shape, model
    )

    pixel_data.update(
        generate_procedural_regions(
            pixel_data, named_colors, model, exclude_keys=frozenset(pixel_data.keys())
        )
    )

    metadata = {
        "description": "",
        "skin_tone": named_colors["skin_tone"],
        "hair_color": named_colors["hair_color"],
        "regions_generated": len(AI_GENERATED_KEYS),
        "ai_model": ai_model,
    }

    persist_state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": raw_grids,
        "description": "",
        "hair_style": hair_style,
    }

    return pixel_data, metadata, persist_state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v`
Expected: all PASS

Then run the whole suite:

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/claude_vision.py tests/test_claude_vision.py
git add app/services/claude_vision.py tests/test_claude_vision.py
git commit -m "feat: rewrite generation pipeline to region-localization + photo_render rendering"
```

---

### Task 5: Retire `PALETTE_ROLES`/`MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE` from `skin_map.py`

Nothing imports these anymore after Task 4 (confirm with a repo-wide grep before deleting — see Step 1). Final cleanup task.

**Files:**
- Modify: `app/services/skin_map.py`
- Modify: `tests/test_skin_map.py`

**Interfaces:** None — these constants are removed, and nothing in this plan or the existing codebase (post-Task-4) references them.

- [ ] **Step 1: Confirm nothing still references these constants**

Run: `grep -rn "PALETTE_ROLES\|MIN_PALETTE_SIZE\|MAX_PALETTE_SIZE" app/ tests/ --include="*.py"`
Expected: only hits inside `app/services/skin_map.py` itself and `tests/test_skin_map.py` (the tests this task is about to delete). If anything else shows up, stop — a prior task missed a reference; do not proceed with deletion until that's resolved.

- [ ] **Step 2: Remove the tests**

In `tests/test_skin_map.py`, delete these three tests entirely: `test_palette_roles_cover_ten_fixed_slots`, `test_palette_roles_expected_names`, `test_palette_size_bounds`.

- [ ] **Step 3: Run to verify the suite is still green with the tests gone**

Run: `.venv/bin/python -m pytest tests/test_skin_map.py -v`
Expected: all remaining tests PASS (no test currently imports `PALETTE_ROLES` etc. anymore since Step 2 removed the only ones that did)

- [ ] **Step 4: Remove the constants from `app/services/skin_map.py`**

Delete this whole block from the end of the file:

```python

# ── Palette roles (used by claude_vision.py and procedural.py) ───

# Fixed positions in the AI's `palette` output. Indices 10+ are freeform,
# chosen by the model for logos/patterns/accessories on optional detail
# faces. Keeping these positions fixed (rather than letting the model
# choose) is what lets a conversational color edit retint by swapping one
# array entry — see docs/superpowers/specs/2026-07-09-skin-fidelity-and-color-edit-design.md.
PALETTE_ROLES: dict[str, int] = {
    "skin_tone": 0,
    "hair_color": 1,
    "eye_color": 2,
    "shirt_main": 3,
    "shirt_shadow": 4,
    "arm_main": 5,
    "arm_shadow": 6,
    "pants_main": 7,
    "pants_shadow": 8,
    "shoe_color": 9,
}
MIN_PALETTE_SIZE = 10
MAX_PALETTE_SIZE = 16
```

- [ ] **Step 5: Run the full suite to verify it's green**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check app/services/skin_map.py tests/test_skin_map.py
git add app/services/skin_map.py tests/test_skin_map.py
git commit -m "chore: retire PALETTE_ROLES and its size bounds, unused after the photo-render rewrite"
```

---

## Self-Review Notes

- **Spec coverage:** Region-localization prompt/schema with per-region graceful degradation (Task 4) ✓. `photo_render.py`'s crop/dominant-downsample/shared-quantize/face-band-extraction/head-template (Task 2) ✓. `AI_GENERATED_KEYS` shrink to 6 + head flat-fill with `head_fill_color` (Task 3) ✓. What stays unchanged (`skin_map.py`'s UV coords, `skin_assembler.py`, `skin_store.py`, `propagate_front_row`) — untouched by every task, confirmed by each task's Files list ✓. What's retired — edit route/schema/frontend (Task 1), `interpret_color_edit`/`apply_color_edit`/`EDIT_MAX_TOKENS` (removed by Task 4's full-file replacement), `PALETTE_ROLES`/`MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE` (Task 5) ✓.
- **Bug caught during self-review, fixed before finalizing:** the spec's "missing front falls back to flat fill" behavior (inherited from the previous plan) was only ever exercised by a front face that's *always present in practice* (old mandatory AI emission guaranteed it) — under this plan's `visible: false` regions, a front face can be legitimately and commonly absent (e.g. no legs in a headshot). The original Task 3 draft kept the old `skip = AI_GENERATED_KEYS | exclude_keys` unconditional-exclusion logic, which would have silently omitted that front face entirely (a transparent hole in the PNG) instead of flat-filling it. Fixed by making `skip` purely `exclude_keys` — Task 3 now includes this fix plus a dedicated regression test (`test_generate_procedural_regions_missing_front_gets_flat_fill_not_omitted`) and Task 4's integration test's `assert len(pixel_data) == 36` depends on it.
- **Full-suite-green-per-commit checked:** Task 3's fix is backward-compatible with the pre-Task-4 `claude_vision.py` by coincidence-turned-guarantee (its `pixel_data` always already contains every front face before Task 3 lands, so `exclude_keys` alone reproduces the same skip set `AI_GENERATED_KEYS | exclude_keys` did) — reasoned through in the task's intro and re-verified by its Step 5 full-suite run — rather than needing to be bundled with Task 4, unlike the previous plan's Task 1. Task 5's Step 1 grep gate ensures the retired constants really are dead before deleting them.
- **Type consistency checked:** `generate_procedural_regions(pixel_data, colors, model, exclude_keys=...)` signature (defined before this plan, unchanged) is called identically in Task 4's `generate_skin_data`, always passing `exclude_keys=frozenset(pixel_data.keys())` — required for Task 3's fixed skip semantics to behave correctly. `photo_render.py`'s function names/signatures (Task 2) match exactly how Task 4 imports and calls them (`crop_region`, `downsample_dominant`, `quantize_shared`, `extract_face_colors`, `dominant_hex`, `build_head_front`, the `Bbox` type alias). `AI_GENERATED_KEYS` (Task 3) matches the 6 keys `_render_photo`/`generate_skin_data` (Task 4) actually produce (`head_front` + the 5 `_PHOTO_PARTS` front keys).
- **No placeholders:** every step has complete, runnable code.
