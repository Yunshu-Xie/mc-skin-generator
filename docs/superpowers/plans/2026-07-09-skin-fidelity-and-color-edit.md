# Skin Fidelity Fix + Conversational Color Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two rendering-fidelity complaints from manual testing (no facial features, body reads as a flat outlined rectangle) and add a conversational color-editing endpoint, by unifying the AI's color palette and the procedurally-filled body colors into one array with fixed semantic role slots.

**Architecture:** The Vision call's `palette` array gets 10 fixed-role slots (skin/hair/eye/shirt/arm/pants/shoe) followed by up to 6 freeform slots the model uses for logos/patterns. Both the AI-drawn pixel grids (indices into `palette`) and the procedural fill (`generate_procedural_regions`, reading named roles) draw from this one array, so changing a palette entry retints everywhere it's used — the mechanism a new `POST /api/skin/{id}/edit` endpoint uses to apply cheap, image-free color edits. Procedural shading moves from a flat 2-tone fill to a code-derived multi-tone shade per face orientation (`colorsys`-based lightness adjustment), at zero added token cost. The model may also opportunistically draw up to 4 extra "detail faces" beyond the mandatory 7 when a photo has a distinctive design element.

**Tech Stack:** Python 3.11, FastAPI, Pillow, `openai` SDK against Gemini's OpenAI-compatible endpoint, pytest, vanilla JS frontend.

## Global Constraints

- All Python files use `from __future__ import annotations` (existing codebase convention).
- No new pip dependencies — `colorsys`, `re`, `json` are stdlib.
- Every new/changed function needs a test; run `pytest -q` after each task and confirm the full suite passes before committing.
- `ruff check <changed files>` must be clean before committing each task.
- Existing `AI_GENERATED_KEYS`, `PIXEL_KEY_MAP`, `get_all_regions`, `assemble_skin`, `hex_to_rgba` are unchanged — don't touch `app/services/skin_assembler.py`.

---

### Task 1: Palette role constants in `skin_map.py`

**Files:**
- Modify: `app/services/skin_map.py` (append to end of file, after `PIXEL_KEY_MAP` construction)
- Test: `tests/test_skin_map.py` (append)

**Interfaces:**
- Produces: `PALETTE_ROLES: dict[str, int]`, `MIN_PALETTE_SIZE: int`, `MAX_PALETTE_SIZE: int` — consumed by Tasks 4, 5, 6.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_skin_map.py`:

```python
def test_palette_roles_cover_ten_fixed_slots():
    from app.services.skin_map import PALETTE_ROLES

    assert len(PALETTE_ROLES) == 10
    assert set(PALETTE_ROLES.values()) == set(range(10))


def test_palette_roles_expected_names():
    from app.services.skin_map import PALETTE_ROLES

    assert PALETTE_ROLES["skin_tone"] == 0
    assert PALETTE_ROLES["hair_color"] == 1
    assert PALETTE_ROLES["eye_color"] == 2
    assert PALETTE_ROLES["shirt_main"] == 3
    assert PALETTE_ROLES["shirt_shadow"] == 4
    assert PALETTE_ROLES["arm_main"] == 5
    assert PALETTE_ROLES["arm_shadow"] == 6
    assert PALETTE_ROLES["pants_main"] == 7
    assert PALETTE_ROLES["pants_shadow"] == 8
    assert PALETTE_ROLES["shoe_color"] == 9


def test_palette_size_bounds():
    from app.services.skin_map import MAX_PALETTE_SIZE, MIN_PALETTE_SIZE

    assert MIN_PALETTE_SIZE == 10
    assert MAX_PALETTE_SIZE == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_skin_map.py -v -k palette`
Expected: FAIL with `ImportError: cannot import name 'PALETTE_ROLES'`

- [ ] **Step 3: Implement**

Append to `app/services/skin_map.py`:

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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_skin_map.py -v`
Expected: all PASS (existing tests + 3 new ones)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/skin_map.py tests/test_skin_map.py
git add app/services/skin_map.py tests/test_skin_map.py
git commit -m "feat: add fixed palette role constants to skin_map"
```

---

### Task 2: Multi-tone procedural shading (`shade_face`)

Replaces `flat_fill_with_border` with a version that derives highlight/shadow tones from one `main` color via HSL lightness adjustment, biased by face orientation, instead of requiring a separately-authored shadow color. `generate_procedural_regions` gains an `exclude_keys` parameter so faces the AI opportunistically draws (Task 4) aren't overwritten.

**Files:**
- Modify: `app/services/procedural.py` (full rewrite)
- Modify: `tests/test_procedural.py` (full rewrite)

**Interfaces:**
- Consumes: `ModelType`, `get_all_regions` from `app.services.skin_map` (unchanged).
- Produces: `shade_face(h: int, w: int, main: str, face: str) -> list[list[str]]`, `generate_procedural_regions(colors: dict[str, str], model: ModelType, exclude_keys: frozenset[str] = frozenset()) -> dict[str, list[list[str]]]` — consumed by Tasks 4 and 5. `AI_GENERATED_KEYS: set[str]` unchanged (still the 7 mandatory keys), still consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_procedural.py` entirely with:

```python
"""Tests for procedural — shaded flat-fill generation for non-AI-painted faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    generate_procedural_regions,
    shade_face,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_shade_face_shape():
    grid = shade_face(4, 4, "#FF0000", "front")
    assert len(grid) == 4
    assert all(len(row) == 4 for row in grid)


def test_shade_face_edge_darker_than_interior():
    import colorsys

    grid = shade_face(6, 6, "#8080A0", "front")
    edge_l = colorsys.rgb_to_hls(
        *[int(grid[0][0].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    interior_l = colorsys.rgb_to_hls(
        *[int(grid[2][2].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    assert edge_l < interior_l


def test_shade_face_top_lighter_than_bottom():
    import colorsys

    top = shade_face(4, 4, "#606060", "top")
    bottom = shade_face(4, 4, "#606060", "bottom")
    top_l = colorsys.rgb_to_hls(
        *[int(top[1][1].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    bottom_l = colorsys.rgb_to_hls(
        *[int(bottom[1][1].lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )[1]
    assert top_l > bottom_l


def test_shade_face_narrow_width():
    """3px-wide grid (slim arm) still has a valid interior column."""
    grid = shade_face(12, 3, "#0000FF", "front")
    assert len(grid[5]) == 3


def _colors(**overrides: str) -> dict[str, str]:
    base = {
        "shirt_main": "#3355AA",
        "arm_main": "#C4A882",
        "pants_main": "#223355",
        "shoe_color": "#2B2B2B",
    }
    base.update(overrides)
    return base


def test_generate_procedural_regions_covers_all_non_ai_faces():
    regions = get_all_regions("classic")
    expected_keys = {
        key
        for key, (group, _face) in PIXEL_KEY_MAP.items()
        if group in ("body", "right_arm", "left_arm", "right_leg", "left_leg")
        and key not in AI_GENERATED_KEYS
    }

    out = generate_procedural_regions(_colors(), "classic")

    assert set(out.keys()) == expected_keys
    for key, grid in out.items():
        group, face = PIXEL_KEY_MAP[key]
        rect = regions[group][face]
        assert len(grid) == rect.h
        assert all(len(row) == rect.w for row in grid)


def test_generate_procedural_regions_never_touches_ai_keys():
    out = generate_procedural_regions(_colors(), "classic")
    assert AI_GENERATED_KEYS.isdisjoint(out.keys())


def test_generate_procedural_regions_respects_exclude_keys():
    out = generate_procedural_regions(
        _colors(), "classic", exclude_keys=frozenset({"right_arm_front"})
    )
    assert "right_arm_front" not in out
    assert "right_arm_back" in out  # sibling face still procedural


def test_generate_procedural_regions_slim_arm_width():
    out = generate_procedural_regions(_colors(), "slim")
    assert len(out["right_arm_front"][0]) == 3
    assert len(out["left_arm_front"][0]) == 3


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    out = generate_procedural_regions(_colors(shoe_color="#ABCDEF"), "classic")
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_falls_back_to_defaults_on_missing_colors():
    out = generate_procedural_regions({}, "classic")
    assert len(out) > 0
    assert out["body_back"][1][1]  # some non-empty hex string
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_procedural.py -v`
Expected: FAIL — `ImportError: cannot import name 'shade_face'`

- [ ] **Step 3: Implement**

Replace `app/services/procedural.py` entirely with:

```python
"""Procedural (non-AI) generation for skin faces that don't need per-pixel detail.

Only head (6 faces), body_front, and any AI-chosen "detail faces" are
generated pixel-by-pixel. Everything else is a shaded flat fill built
directly from a handful of named colors (see PALETTE_ROLES in skin_map.py).
Shading is derived purely from one `main` hex per part via HSL lightness
adjustment — no separate shadow color needed here, so left/right pairs and
every face just reuse the same main color with a per-orientation bias.
"""

from __future__ import annotations

import colorsys

from app.services.skin_map import ModelType, get_all_regions

# Faces the AI generates directly; everything else in get_all_regions()
# (excluding overlay groups, which are unused) is procedural unless also
# excluded via the `exclude_keys` param (AI-chosen detail faces).
AI_GENERATED_KEYS = {
    "head_front",
    "head_back",
    "head_top",
    "head_bottom",
    "head_left",
    "head_right",
    "body_front",
}

DEFAULT_COLORS: dict[str, str] = {
    "shirt_main": "#3B5998",
    "arm_main": "#C4A882",
    "pants_main": "#1A1A3E",
    "shoe_color": "#2B2B2B",
}

# Lightness delta applied to `main` per face orientation, so a flat-colored
# part still reads as a 3D form instead of identical tiles on every side.
ORIENTATION_BIAS: dict[str, float] = {
    "top": 0.12,
    "front": 0.0,
    "right": 0.0,
    "left": -0.08,
    "back": -0.15,
    "bottom": -0.22,
}
EDGE_STEP = 0.12  # extra darkening for the outermost 1px ring


def _adjust_lightness(hex_color: str, delta: float) -> str:
    """Shift a hex color's HSL lightness by `delta`, clamped to [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = min(1.0, max(0.0, lightness + delta))
    r2, g2, b2 = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02X}{:02X}{:02X}".format(round(r2 * 255), round(g2 * 255), round(b2 * 255))


def shade_face(h: int, w: int, main: str, face: str) -> list[list[str]]:
    """A shaded grid: interior tone biased by face orientation, 1px darker edge ring."""
    bias = ORIENTATION_BIAS.get(face, 0.0)
    interior = _adjust_lightness(main, bias)
    edge = _adjust_lightness(main, bias - EDGE_STEP)
    return [
        [edge if (row in (0, h - 1) or col in (0, w - 1)) else interior for col in range(w)]
        for row in range(h)
    ]


def generate_procedural_regions(
    colors: dict[str, str],
    model: ModelType,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, list[list[str]]]:
    """Build hex pixel grids for every face not covered by AI generation."""
    c = {**DEFAULT_COLORS, **{k: v for k, v in colors.items() if v}}
    regions = get_all_regions(model)
    skip = AI_GENERATED_KEYS | exclude_keys
    out: dict[str, list[list[str]]] = {}

    for face, rect in regions["body"].items():
        key = f"body_{face}"
        if key in skip:
            continue
        out[key] = shade_face(rect.h, rect.w, c["shirt_main"], face)

    for part in ("right_arm", "left_arm"):
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in skip:
                continue
            out[key] = shade_face(rect.h, rect.w, c["arm_main"], face)

    for part in ("right_leg", "left_leg"):
        for face, rect in regions[part].items():
            key = f"{part}_{face}"
            if key in skip:
                continue
            if face == "bottom":
                out[key] = [[c["shoe_color"]] * rect.w for _ in range(rect.h)]
            else:
                out[key] = shade_face(rect.h, rect.w, c["pants_main"], face)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_procedural.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/procedural.py tests/test_procedural.py
git add app/services/procedural.py tests/test_procedural.py
git commit -m "feat: replace flat-fill border shading with multi-tone shade_face"
```

---

### Task 3: Skin state persistence (`skin_store.py`)

New module: saves the raw palette-index grids and full palette next to the generated PNG, so a later color edit can re-decode with a changed palette without a new Vision call.

**Files:**
- Create: `app/services/skin_store.py`
- Test: `tests/test_skin_store.py`

**Interfaces:**
- Produces: `save_skin_state(skin_id: str, model: str, ai_model: str, palette: list[str], pixel_grids: dict[str, list[list[int]]], description: str, hair_style: str) -> None`, `load_skin_state(skin_id: str) -> dict[str, Any] | None` — consumed by Task 6 (router).

- [ ] **Step 1: Write the failing test**

Create `tests/test_skin_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_skin_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.skin_store'`

- [ ] **Step 3: Implement**

Create `app/services/skin_store.py`:

```python
"""Persistence for per-skin generation state, used by conversational color edits.

Generation only paints 7-11 faces pixel-by-pixel (see procedural.py for
why); the raw palette-index grids and full palette are saved next to the
PNG so a later edit can re-decode with a changed palette without a new
Vision call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


def _state_path(skin_id: str) -> Path:
    return Path(settings.skins_dir) / f"{skin_id}.json"


def save_skin_state(
    skin_id: str,
    model: str,
    ai_model: str,
    palette: list[str],
    pixel_grids: dict[str, list[list[int]]],
    description: str,
    hair_style: str,
) -> None:
    state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": pixel_grids,
        "description": description,
        "hair_style": hair_style,
    }
    _state_path(skin_id).write_text(json.dumps(state))


def load_skin_state(skin_id: str) -> dict[str, Any] | None:
    path = _state_path(skin_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_skin_store.py -v`
Expected: both PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/skin_store.py tests/test_skin_store.py
git add app/services/skin_store.py tests/test_skin_store.py
git commit -m "feat: add skin_store for persisting generation state"
```

---

### Task 4: Rewrite the generate path in `claude_vision.py`

New prompt/schema (fixed-role palette, up to 16 colors, optional detail faces, explicit eye/mouth color rule), new validation that decodes both the mandatory 7 faces and any AI-drawn detail faces, and `generate_skin_data` now also returns a `persist_state` dict ready for `skin_store.save_skin_state`. `_decode_indexed_grid` is renamed to `decode_indexed_grid` (dropped leading underscore) since Task 6 needs to call it from `routers/skin.py`.

**Files:**
- Modify: `app/services/claude_vision.py` (full rewrite of everything except `_prepare_image`, `_get_client`, `_extract_json`, which are unchanged)
- Modify: `tests/test_claude_vision.py` (full rewrite)

**Interfaces:**
- Consumes: `PALETTE_ROLES`, `MIN_PALETTE_SIZE`, `MAX_PALETTE_SIZE`, `PIXEL_KEY_MAP`, `ModelType`, `get_all_regions` from `app.services.skin_map` (Task 1); `AI_GENERATED_KEYS`, `generate_procedural_regions` from `app.services.procedural` (Task 2).
- Produces: `decode_indexed_grid(grid: list[list[Any]], palette: list[str]) -> list[list[str]]` (renamed, now public), `_is_valid_hex(value: Any) -> bool`, `generate_skin_data(...) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]` (now 3-tuple — third element is `persist_state`) — consumed by Task 5 (`decode_indexed_grid`) and Task 6 (router).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_claude_vision.py` entirely with:

```python
"""Tests for claude_vision — indexed-grid decoding, response validation, image prep."""

import io

from PIL import Image

from app.config import settings
from app.services.claude_vision import (
    _is_valid_hex,
    _prepare_image,
    _resolve_model_name,
    _validate_and_decode,
    decode_indexed_grid,
)
from app.services.procedural import AI_GENERATED_KEYS
from app.services.skin_map import MAX_PALETTE_SIZE, MIN_PALETTE_SIZE, PALETTE_ROLES


def test_resolve_model_name_flash():
    assert _resolve_model_name("flash") == settings.gemini_model_flash


def test_resolve_model_name_flash_lite():
    assert _resolve_model_name("flash-lite") == settings.gemini_model_flash_lite


def test_is_valid_hex():
    assert _is_valid_hex("#AABBCC") is True
    assert _is_valid_hex("#aabbcc") is True
    assert _is_valid_hex("not-a-color") is False
    assert _is_valid_hex("#AABBCCDD") is False  # 8-char not accepted for palette entries
    assert _is_valid_hex(123) is False


def test_decode_indexed_grid_maps_to_palette():
    palette = ["#111111", "#222222", "#333333"]
    grid = [[0, 1], [2, 0]]
    assert decode_indexed_grid(grid, palette) == [
        ["#111111", "#222222"],
        ["#333333", "#111111"],
    ]


def test_decode_indexed_grid_clamps_bad_index():
    palette = ["#111111", "#222222"]
    grid = [[0, 99], [-1, 1]]
    decoded = decode_indexed_grid(grid, palette)
    assert decoded[0][1] == "#111111"  # out-of-range clamps to palette[0]
    assert decoded[1][0] == "#111111"  # negative clamps to palette[0]
    assert decoded[1][1] == "#222222"


def _fixed_palette(n_freeform: int = 0) -> list[str]:
    base = [
        "#C4A882",  # skin_tone
        "#5B3A1A",  # hair_color
        "#3B5998",  # eye_color
        "#AA3355",  # shirt_main
        "#882244",  # shirt_shadow
        "#C4A882",  # arm_main
        "#9C8266",  # arm_shadow
        "#1A1A3E",  # pants_main
        "#101028",  # pants_shadow
        "#2B2B2B",  # shoe_color
    ]
    return base + [f"#{i:02X}{i:02X}{i:02X}" for i in range(10, 10 + n_freeform)]


def _valid_response(n_freeform: int = 0, extra_faces: dict | None = None) -> dict:
    data = {
        "description": "A test character",
        "hair_style": "short",
        "palette": _fixed_palette(n_freeform),
    }
    for key in AI_GENERATED_KEYS:
        h = 12 if key == "body_front" else 8
        data[key] = [[0] * 8 for _ in range(h)]
    if extra_faces:
        data.update(extra_faces)
    return data


def test_validate_and_decode_complete_response():
    pixel_data, raw_grids, colors, metadata, ok = _validate_and_decode(
        _valid_response(), "classic"
    )
    assert ok is True
    assert set(pixel_data.keys()) == AI_GENERATED_KEYS
    assert set(raw_grids.keys()) == AI_GENERATED_KEYS
    assert metadata["description"] == "A test character"
    assert colors["shirt_main"] == "#AA3355"
    assert pixel_data["head_front"][0][0] == "#C4A882"  # decoded index 0 -> skin_tone
    assert raw_grids["head_front"][0][0] == 0  # raw index preserved


def test_validate_and_decode_named_colors_match_palette_roles():
    _pixel_data, _raw, colors, _metadata, _ok = _validate_and_decode(
        _valid_response(), "classic"
    )
    palette = _fixed_palette()
    for name, idx in PALETTE_ROLES.items():
        assert colors[name] == palette[idx]


def test_validate_and_decode_decodes_optional_detail_face():
    extra = {"body_back": [[3] * 8 for _ in range(12)]}
    pixel_data, raw_grids, _colors, metadata, ok = _validate_and_decode(
        _valid_response(extra_faces=extra), "classic"
    )
    assert ok is True
    assert "body_back" in pixel_data
    assert "body_back" in raw_grids
    assert pixel_data["body_back"][0][0] == "#AA3355"  # index 3 -> shirt_main
    assert metadata["regions_generated"] == len(AI_GENERATED_KEYS) + 1


def test_validate_and_decode_ignores_malformed_detail_face():
    extra = {"body_back": [[3] * 4 for _ in range(4)]}  # wrong dims for body_back
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(
        _valid_response(extra_faces=extra), "classic"
    )
    assert ok is True  # mandatory faces still fine
    assert "body_back" not in pixel_data


def test_validate_and_decode_palette_too_short_is_incomplete():
    data = _valid_response()
    data["palette"] = data["palette"][:5]
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert pixel_data == {}


def test_validate_and_decode_palette_too_long_is_incomplete():
    data = _valid_response(n_freeform=MAX_PALETTE_SIZE - MIN_PALETTE_SIZE + 1)
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert pixel_data == {}


def test_validate_and_decode_invalid_hex_in_palette_is_incomplete():
    data = _valid_response()
    data["palette"][0] = "not-a-color"
    _pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False


def test_validate_and_decode_wrong_grid_size_is_incomplete():
    data = _valid_response()
    data["head_front"] = [[0] * 4 for _ in range(4)]  # wrong size
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert "head_front" not in pixel_data
    assert "body_front" in pixel_data  # other mandatory grids still decoded


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v`
Expected: FAIL — `ImportError: cannot import name 'decode_indexed_grid'` (and others, since `_validate_and_decode` still has the old 4-tuple return / old signature)

- [ ] **Step 3: Implement**

Replace `app/services/claude_vision.py` entirely with:

```python
"""Gemini AI pipeline: single Vision call → head/body_front (+ optional detail
faces) pixels + a fixed-role color palette.

Uses the Gemini API's OpenAI-compatible endpoint. Callers pick between
gemini-2.5-flash and gemini-2.5-flash-lite per request via `ai_model`.

Only head (6 faces), body_front, and any faces the model opportunistically
chooses to draw are generated pixel-by-pixel. Everything else is filled in
procedurally by app.services.procedural, reading named colors out of the
same palette the AI-drawn pixels reference. See
docs/superpowers/specs/2026-07-09-skin-fidelity-and-color-edit-design.md
for the full design.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Literal

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings
from app.services.procedural import AI_GENERATED_KEYS, generate_procedural_regions
from app.services.skin_map import (
    MAX_PALETTE_SIZE,
    MIN_PALETTE_SIZE,
    PALETTE_ROLES,
    PIXEL_KEY_MAP,
    ModelType,
    get_all_regions,
)

logger = logging.getLogger(__name__)

AIModel = Literal["flash", "flash-lite"]

# Face groups eligible for the AI's optional "detail faces" — the base
# (non-overlay) layer only; overlay (hat/jacket) groups are unused by this app.
BASE_GROUPS = {"head", "body", "right_arm", "left_arm", "right_leg", "left_leg"}

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _is_valid_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value))


def _resolve_model_name(ai_model: AIModel) -> str:
    return (
        settings.gemini_model_flash_lite
        if ai_model == "flash-lite"
        else settings.gemini_model_flash
    )


def _build_prompt(model: ModelType) -> str:
    arm_w = 4 if model == "classic" else 3
    extra_face_keys = ", ".join(
        key for key in PIXEL_KEY_MAP if key not in AI_GENERATED_KEYS and PIXEL_KEY_MAP[key][0] in BASE_GROUPS
    )

    return f"""\
You are a Minecraft skin designer. Look at this photo and design a Minecraft \
skin based on the character/person's appearance.

Output ONLY a JSON object (no markdown, no explanation) with this structure:

{{
  "description": "Brief description of what you see",
  "hair_style": "short|long|bald|hat|helmet",

  "palette": [
    "#HEX index 0 = skin_tone",
    "#HEX index 1 = hair_color",
    "#HEX index 2 = eye_color",
    "#HEX index 3 = shirt_main",
    "#HEX index 4 = shirt_shadow",
    "#HEX index 5 = arm_main",
    "#HEX index 6 = arm_shadow",
    "#HEX index 7 = pants_main",
    "#HEX index 8 = pants_shadow",
    "#HEX index 9 = shoe_color",
    "... optionally 0-6 more freeform colors (index 10-15) for logos, patterns, or accessories you want to draw"
  ],

  "head_front":  [[palette index per pixel] × 8 cols] × 8 rows,
  "head_back":   [8×8 palette indices],
  "head_top":    [8×8 palette indices],
  "head_bottom": [8×8 palette indices],
  "head_left":   [8×8 palette indices],
  "head_right":  [8×8 palette indices],
  "body_front":  [12×8 palette indices]
}}

PIXEL ART RULES (row-major, [row][col], every cell is an integer palette index):
- The first 10 palette entries MUST be in exactly this order: skin_tone, \
hair_color, eye_color, shirt_main, shirt_shadow, arm_main, arm_shadow, \
pants_main, pants_shadow, shoe_color. Index 10+ are your choice, up to 16 total.
- head_front: row 0-1 = hair/forehead, row 2-3 = eyes, row 4 = nose, \
row 5-6 = mouth/chin, row 7 = neck
- Eyes and mouth in head_front MUST use a palette index other than \
skin_tone (0) and hair_color (1) — reuse eye_color (2) or another index, so \
facial features are visible against the skin
- body_front: torso/shirt front, {arm_w}-px-wide-arm model
- Most characters do NOT need anything beyond the 7 faces above. ONLY if \
the photo shows a genuinely distinctive design element (a back logo, a \
sleeve pattern, a belt, a cape) that would look wrong as a flat color, add \
up to 4 more faces as extra top-level keys using the same [row][col] \
palette-index format. Valid extra keys: {extra_face_keys}
- shirt_main/shirt_shadow, arm_main/arm_shadow, pants_main/pants_shadow, \
and shoe_color are also used to procedurally color any face you don't draw \
yourself — pick them to match the photo's clothing
- All palette hex values are exactly 7 chars: "#RRGGBB\""""


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


def _try_decode_face(
    data: dict[str, Any],
    key: str,
    group: str,
    face: str,
    regions: dict[str, dict[str, Any]],
    palette: list[str],
) -> tuple[list[list[int]], list[list[str]]] | None:
    rect = regions[group][face]
    grid = data.get(key)
    if not (
        isinstance(grid, list)
        and len(grid) == rect.h
        and all(
            isinstance(row, list)
            and len(row) == rect.w
            and all(isinstance(cell, int) for cell in row)
            for row in grid
        )
    ):
        return None
    return grid, decode_indexed_grid(grid, palette)


def _validate_and_decode(
    data: dict[str, Any], model: ModelType
) -> tuple[
    dict[str, list[list[str]]], dict[str, list[list[int]]], dict[str, str], dict[str, Any], bool
]:
    """Validate the model response and decode its indexed pixel grids to hex.

    Returns (pixel_data_hex, raw_index_grids, named_colors, metadata, is_complete).
    """
    regions = get_all_regions(model)
    palette = data.get("palette")
    has_valid_palette = (
        isinstance(palette, list)
        and MIN_PALETTE_SIZE <= len(palette) <= MAX_PALETTE_SIZE
        and all(_is_valid_hex(c) for c in palette)
    )

    pixel_data: dict[str, list[list[str]]] = {}
    raw_grids: dict[str, list[list[int]]] = {}

    if has_valid_palette:
        for key in AI_GENERATED_KEYS:
            group, face = PIXEL_KEY_MAP[key]
            decoded = _try_decode_face(data, key, group, face, regions, palette)
            if decoded is None:
                logger.warning("Invalid/missing grid for %s", key)
                continue
            raw_grids[key], pixel_data[key] = decoded

        for key, (group, face) in PIXEL_KEY_MAP.items():
            if key in AI_GENERATED_KEYS or group not in BASE_GROUPS:
                continue
            decoded = _try_decode_face(data, key, group, face, regions, palette)
            if decoded is not None:
                raw_grids[key], pixel_data[key] = decoded

    colors = (
        {name: palette[idx] for name, idx in PALETTE_ROLES.items()} if has_valid_palette else {}
    )

    mandatory_ok = all(key in pixel_data for key in AI_GENERATED_KEYS)

    metadata = {
        "description": data.get("description", ""),
        "skin_tone": colors.get("skin_tone", ""),
        "hair_color": colors.get("hair_color", ""),
        "regions_generated": len(pixel_data),
    }

    is_complete = has_valid_palette and mandatory_ok
    return pixel_data, raw_grids, colors, metadata, is_complete


async def _call_vision(
    image_bytes: bytes,
    media_type: str,
    model: ModelType,
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
        {"type": "text", "text": _build_prompt(model)},
    ]
    if style_notes:
        user_content.append(
            {"type": "text", "text": f"\nAdditional style notes: {style_notes}"}
        )

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=4000,
        temperature=0.3,
        # Gemini 2.5 defaults to "thinking" mode, which eats into max_tokens
        # before any visible output — without this the response gets cut off.
        reasoning_effort="none",
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.choices[0].message.content or ""
    return _extract_json(text)


async def generate_skin_data(
    image_bytes: bytes,
    media_type: str,
    model: ModelType = "classic",
    style_notes: str = "",
    ai_model: AIModel = "flash",
    max_retries: int = 2,
) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]:
    """Full pipeline: photo → one Vision call → AI pixels + procedural fill.

    Returns:
        (pixel_data covering all base regions, metadata, persist_state).
        `persist_state` is ready to pass to skin_store.save_skin_state(skin_id, **persist_state).
    """
    prepped_bytes, prepped_media_type = _prepare_image(image_bytes)

    pixel_data: dict[str, list[list[str]]] = {}
    raw_grids: dict[str, list[list[int]]] = {}
    colors: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "description": "",
        "skin_tone": "",
        "hair_color": "",
        "regions_generated": 0,
    }
    raw: dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        raw = await _call_vision(prepped_bytes, prepped_media_type, model, style_notes, ai_model)
        pixel_data, raw_grids, colors, metadata, is_complete = _validate_and_decode(raw, model)
        if is_complete:
            break
        logger.warning(
            "Attempt %d: %d/%d mandatory AI regions decoded",
            attempt + 1,
            sum(1 for k in AI_GENERATED_KEYS if k in pixel_data),
            len(AI_GENERATED_KEYS),
        )

    logger.info("Analysis complete: %s", metadata.get("description", ""))

    pixel_data.update(
        generate_procedural_regions(colors, model, exclude_keys=frozenset(pixel_data.keys()))
    )
    metadata["ai_model"] = ai_model

    palette = raw.get("palette") if isinstance(raw.get("palette"), list) else []
    persist_state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": raw_grids,
        "description": metadata.get("description", ""),
        "hair_style": raw.get("hair_style", "") if isinstance(raw.get("hair_style"), str) else "",
    }

    return pixel_data, metadata, persist_state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/claude_vision.py tests/test_claude_vision.py
git add app/services/claude_vision.py tests/test_claude_vision.py
git commit -m "feat: fixed-role palette, detail faces, and persist_state in generate path"
```

---

### Task 5: Conversational color edit (`interpret_color_edit` + `apply_color_edit`)

Adds a cheap, text-only (no image) call that maps a free-text instruction to changed named-role colors, plus an orchestration function that applies those changes to a persisted skin state and re-renders — mirroring what `generate_skin_data` returns, so the router (Task 6) can treat both endpoints the same way.

**Files:**
- Modify: `app/services/claude_vision.py` (append)
- Modify: `tests/test_claude_vision.py` (append)
- Modify: `pyproject.toml` (add `pytest-asyncio`, enable `asyncio_mode = "auto"`)

**Interfaces:**
- Consumes: `PALETTE_ROLES` (Task 1), `generate_procedural_regions` (Task 2), `decode_indexed_grid` (Task 4, same file).
- Produces: `interpret_color_edit(current_roles: dict[str, str], instruction: str, ai_model: AIModel) -> dict[str, str]`, `apply_color_edit(state: dict[str, Any], instruction: str) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]` — consumed by Task 6 (router).

This task's tests are the first `async def test_...` functions in this project — `pytest-asyncio` is not currently installed or configured, so Step 1 below sets that up first.

- [ ] **Step 0: Install and configure `pytest-asyncio`**

```bash
.venv/bin/pip install pytest-asyncio
```

In `pyproject.toml`, change:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "ruff>=0.6",
    "mypy>=1.11",
]
```

to:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
    "ruff>=0.6",
    "mypy>=1.11",
]
```

And add `asyncio_mode = "auto"` to the existing pytest section so `async def test_...` functions run without a per-test `@pytest.mark.asyncio` decorator:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

Verify the plugin is picked up:

```bash
.venv/bin/python -m pytest --version
```

Expected: output includes `plugins: ... asyncio-...` in the plugin list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_claude_vision.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.claude_vision import apply_color_edit, interpret_color_edit


def _mock_chat_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


@patch("app.services.claude_vision._get_client")
async def test_interpret_color_edit_maps_instruction_to_role(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response('{"shirt_main": "#0000FF"}')
    )
    mock_get_client.return_value = mock_client

    changes = await interpret_color_edit(
        {"shirt_main": "#AA3355", "hair_color": "#5B3A1A"},
        "把衬衫改成蓝色",
        "flash-lite",
    )

    assert changes == {"shirt_main": "#0000FF"}


@patch("app.services.claude_vision._get_client")
async def test_interpret_color_edit_ignores_unknown_roles_and_bad_hex(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response(
            '{"shirt_main": "#0000FF", "made_up_role": "#FFFFFF", "hair_color": "blue"}'
        )
    )
    mock_get_client.return_value = mock_client

    changes = await interpret_color_edit(
        {"shirt_main": "#AA3355", "hair_color": "#5B3A1A"}, "make it blue", "flash"
    )

    assert changes == {"shirt_main": "#0000FF"}


@patch("app.services.claude_vision._get_client")
async def test_interpret_color_edit_returns_empty_on_bad_json(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_chat_response("not json at all")
    )
    mock_get_client.return_value = mock_client

    changes = await interpret_color_edit({"shirt_main": "#AA3355"}, "anything", "flash")

    assert changes == {}


@patch(
    "app.services.claude_vision.interpret_color_edit",
    new_callable=AsyncMock,
    return_value={"shirt_main": "#0000FF"},
)
async def test_apply_color_edit_retints_palette_and_reassembles(mock_interpret):
    state = {
        "model": "classic",
        "ai_model": "flash",
        "palette": _fixed_palette(),
        "pixel_grids": {"body_front": [[3] * 8 for _ in range(12)]},
        "description": "a test character",
        "hair_style": "short",
    }

    pixel_data, metadata, persist_state = await apply_color_edit(state, "把衬衫改成蓝色")

    assert pixel_data["body_front"][0][0] == "#0000FF"  # index 3 now retinted
    assert persist_state["palette"][3] == "#0000FF"
    assert persist_state["pixel_grids"] == state["pixel_grids"]  # raw indices untouched
    assert metadata["changed_roles"] == ["shirt_main"]
    assert "body_back" in pixel_data  # procedural fill still runs for the rest
    mock_interpret.assert_awaited_once_with(
        {name: _fixed_palette()[idx] for name, idx in PALETTE_ROLES.items()},
        "把衬衫改成蓝色",
        "flash",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v -k "edit"`
Expected: FAIL — `ImportError: cannot import name 'apply_color_edit'`

- [ ] **Step 3: Implement**

Append to `app/services/claude_vision.py`:

```python
EDIT_MAX_TOKENS = 300


async def interpret_color_edit(
    current_roles: dict[str, str], instruction: str, ai_model: AIModel
) -> dict[str, str]:
    """Text-only call: map a free-text instruction to changed named color roles.

    Returns a dict of role name -> new hex value, containing only the roles
    that should change. Returns {} if nothing recognized or parsing fails.
    """
    client = _get_client()
    roles_json = json.dumps(current_roles, indent=2)
    prompt = f"""You are editing the colors of an existing Minecraft skin design.

Current named colors:
{roles_json}

User's instruction: "{instruction}"

Output ONLY a JSON object containing the roles that should change and their \
new color as "#RRGGBB". Only use these exact role names: \
{", ".join(current_roles.keys())}. If the instruction doesn't clearly map to \
any of these roles, output {{}}."""

    response = await client.chat.completions.create(
        model=_resolve_model_name(ai_model),
        max_tokens=EDIT_MAX_TOKENS,
        temperature=0.2,
        reasoning_effort="none",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    try:
        changes = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(changes, dict):
        return {}
    return {
        role: value
        for role, value in changes.items()
        if role in current_roles and _is_valid_hex(value)
    }


async def apply_color_edit(
    state: dict[str, Any], instruction: str
) -> tuple[dict[str, list[list[str]]], dict[str, Any], dict[str, Any]]:
    """Apply a conversational color edit to a persisted skin state.

    Returns (pixel_data, metadata, persist_state) — same shapes as
    generate_skin_data, ready to reassemble the PNG and re-save the state.
    """
    model: ModelType = state["model"]
    ai_model: AIModel = state["ai_model"]
    palette = list(state["palette"])

    current_roles = {name: palette[idx] for name, idx in PALETTE_ROLES.items()}
    changes = await interpret_color_edit(current_roles, instruction, ai_model)
    for role, hex_value in changes.items():
        palette[PALETTE_ROLES[role]] = hex_value

    pixel_data: dict[str, list[list[str]]] = {
        key: decode_indexed_grid(grid, palette) for key, grid in state["pixel_grids"].items()
    }
    colors = {name: palette[idx] for name, idx in PALETTE_ROLES.items()}
    pixel_data.update(
        generate_procedural_regions(colors, model, exclude_keys=frozenset(pixel_data.keys()))
    )

    metadata = {
        "description": state.get("description", ""),
        "skin_tone": colors["skin_tone"],
        "hair_color": colors["hair_color"],
        "regions_generated": len(pixel_data),
        "ai_model": ai_model,
        "changed_roles": list(changes.keys()),
    }

    persist_state = {
        "model": model,
        "ai_model": ai_model,
        "palette": palette,
        "pixel_grids": state["pixel_grids"],
        "description": state.get("description", ""),
        "hair_style": state.get("hair_style", ""),
    }

    return pixel_data, metadata, persist_state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v`
Expected: all PASS (full file)

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/claude_vision.py tests/test_claude_vision.py
git add app/services/claude_vision.py tests/test_claude_vision.py pyproject.toml
git commit -m "feat: add interpret_color_edit and apply_color_edit for conversational recoloring"
```

---

### Task 6: Wire persistence into `/api/generate` and add `POST /api/skin/{id}/edit`

**Files:**
- Modify: `app/models/schemas.py` (add `EditSkinRequest`)
- Modify: `app/routers/skin.py` (persist state on generate; new edit endpoint)
- Modify: `tests/test_api.py` (update mocks for the 3-tuple return; add edit endpoint tests)

**Interfaces:**
- Consumes: `generate_skin_data` (3-tuple, Task 4), `apply_color_edit` (Task 5), `save_skin_state` / `load_skin_state` (Task 3).
- Produces: `POST /api/skin/{skin_id}/edit` HTTP endpoint — consumed by Task 7 (frontend).

- [ ] **Step 1: Write the failing tests**

Add to `app/models/schemas.py`, after `SkinGenerateResponse`:

```python
class EditSkinRequest(BaseModel):
    instruction: str
```

In `tests/test_api.py`, three existing tests mock `generate_skin_data`'s return value as a 2-tuple; each needs a third `persist_state` element since Task 4 changed `generate_skin_data` to return a 3-tuple. Apply these three exact replacements:

In `test_generate_skin_success`, replace:

```python
    mock_generate.return_value = (
        _make_dummy_pixel_data(),
        {"description": "Test skin", "skin_tone": "#C4A882", "regions_generated": 36},
    )
```

with:

```python
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
```

In `test_generate_skin_creates_png`, replace:

```python
    mock_generate.return_value = (
        _make_dummy_pixel_data(),
        {"description": "Test", "regions_generated": 36},
    )
```

with:

```python
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
```

In `test_generate_skin_passes_ai_model_choice`, replace:

```python
    mock_generate.return_value = (
        _make_dummy_pixel_data(),
        {"description": "Test", "regions_generated": 36, "ai_model": "flash-lite"},
    )
```

with:

```python
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
```

Then append new tests to `tests/test_api.py`:

```python
from unittest.mock import AsyncMock


@patch("app.routers.skin.apply_color_edit", new_callable=AsyncMock)
@patch("app.routers.skin.load_skin_state")
def test_edit_skin_success(mock_load, mock_apply, client):
    mock_load.return_value = {
        "model": "classic",
        "ai_model": "flash",
        "palette": ["#C4A882"] * 10,
        "pixel_grids": {"head_front": [[0] * 8 for _ in range(8)]},
        "description": "a test character",
        "hair_style": "short",
    }
    mock_apply.return_value = (
        _make_dummy_pixel_data(),
        {
            "description": "a test character",
            "skin_tone": "#C4A882",
            "hair_color": "#C4A882",
            "regions_generated": 36,
            "ai_model": "flash",
            "changed_roles": ["shirt_main"],
        },
        {
            "model": "classic",
            "ai_model": "flash",
            "palette": ["#0000FF"] + ["#C4A882"] * 9,
            "pixel_grids": {"head_front": [[0] * 8 for _ in range(8)]},
            "description": "a test character",
            "hair_style": "short",
        },
    )

    response = client.post(
        "/api/skin/deadbeef/edit",
        json={"instruction": "把衬衫改成蓝色"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["skin_id"] == "deadbeef"
    assert data["metadata"]["changed_roles"] == ["shirt_main"]


@patch("app.routers.skin.load_skin_state", return_value=None)
def test_edit_skin_not_found(mock_load, client):
    response = client.post(
        "/api/skin/doesnotexist/edit",
        json={"instruction": "make it blue"},
    )
    assert response.status_code == 404


def test_edit_skin_invalid_id(client):
    response = client.post(
        "/api/skin/../etc/edit",
        json={"instruction": "make it blue"},
    )
    assert response.status_code in (400, 404, 422)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL — existing tests fail unpacking a 2-tuple where 3 are now expected (once Task 4's `generate_skin_data` is live), and new edit tests fail with 404 (no route yet) or import errors for `EditSkinRequest`

- [ ] **Step 3: Implement**

Replace `app/routers/skin.py` entirely with:

```python
"""Skin generation API endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import EditSkinRequest, SkinGenerateResponse
from app.services.claude_vision import apply_color_edit, generate_skin_data
from app.services.skin_assembler import assemble_skin
from app.services.skin_map import ModelType
from app.services.skin_store import load_skin_state, save_skin_state

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

    # Save PNG + generation state (state enables later conversational edits)
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


@router.post("/skin/{skin_id}/edit", response_model=SkinGenerateResponse)
async def edit_skin(skin_id: str, body: EditSkinRequest) -> SkinGenerateResponse:
    """Apply a conversational color edit to a previously generated skin."""
    if not skin_id.isalnum():
        raise HTTPException(400, "Invalid skin ID")

    state = load_skin_state(skin_id)
    if state is None:
        raise HTTPException(404, "Skin not found")

    try:
        pixel_data, metadata, persist_state = await apply_color_edit(state, body.instruction)
    except Exception as e:
        raise HTTPException(500, f"Skin edit failed: {e}") from e

    skin_image = assemble_skin(pixel_data, persist_state["model"])
    skin_path = Path(settings.skins_dir) / f"{skin_id}.png"
    skin_image.save(str(skin_path), "PNG")
    save_skin_state(skin_id, **persist_state)

    return SkinGenerateResponse(
        skin_id=skin_id,
        skin_url=f"/api/skin/{skin_id}.png",
        model=persist_state["model"],
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: all PASS

Then run the whole suite to catch anything else that broke:

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/models/schemas.py app/routers/skin.py tests/test_api.py
git add app/models/schemas.py app/routers/skin.py tests/test_api.py
git commit -m "feat: persist skin state on generate, add POST /api/skin/{id}/edit"
```

---

### Task 7: Frontend — conversational edit UI

Adds a text input + button under the 3D viewer. On submit it calls the new edit endpoint and reloads the viewer's texture with a cache-busted URL (same filename, new PNG bytes).

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/app.js`
- Modify: `app/static/style.css`

No automated test — this task's deliverable is verified manually (Step 4).

- [ ] **Step 1: Add the edit form to `index.html`**

In `app/static/index.html`, find:

```html
                <div class="viewer-controls">
                    <button id="downloadBtn" class="btn-primary">⬇ 下载 .png</button>
                    <button id="resetBtn" class="btn-secondary">🔄 重新生成</button>
                </div>
```

Replace with:

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

- [ ] **Step 2: Add edit-controls CSS to `style.css`**

Find the `.viewer-controls button` rule:

```css
.viewer-controls button {
    flex: 1;
    margin-top: 0;
}
```

Add immediately after it:

```css
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
```

- [ ] **Step 3: Wire the edit flow in `app.js`**

In `app/static/app.js`, find the DOM element declarations at the top:

```js
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");

let viewer = null;
let currentSkinUrl = null;
```

Replace with:

```js
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");
const editInstruction = document.getElementById("editInstruction");
const editBtn = document.getElementById("editBtn");

let viewer = null;
let currentSkinUrl = null;
let currentSkinId = null;
```

Find the form-submit success call:

```js
        statusDiv.hidden = true;
        showViewer(data.skin_url, data.model, data.metadata);
```

Replace with:

```js
        statusDiv.hidden = true;
        showViewer(data.skin_id, data.skin_url, data.model, data.metadata);
```

Find the `showViewer` function signature and first line:

```js
function showViewer(skinUrl, model, metadata) {
    viewerSection.hidden = false;
    currentSkinUrl = skinUrl;
```

Replace with:

```js
function showViewer(skinId, skinUrl, model, metadata) {
    viewerSection.hidden = false;
    currentSkinId = skinId;
    currentSkinUrl = skinUrl;
```

Find the `// ── Reset ──` block:

```js
// ── Reset ──
resetBtn.addEventListener("click", () => {
    viewerSection.hidden = true;
    if (viewer) {
        viewer.dispose();
        viewer = null;
    }
    currentSkinUrl = null;
    preview.hidden = true;
    preview.src = "";
    dropPrompt.hidden = false;
    imageInput.value = "";
    generateBtn.disabled = true;
    errorDiv.hidden = true;
    document.getElementById("styleNotes").value = "";
});
```

Add immediately after it:

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
```

Also update `resetBtn`'s handler to clear the new state — find:

```js
    currentSkinUrl = null;
    preview.hidden = true;
```

Replace with:

```js
    currentSkinUrl = null;
    currentSkinId = null;
    editInstruction.value = "";
    preview.hidden = true;
```

- [ ] **Step 4: Manually verify**

```bash
pkill -f "uvicorn app.main:app" 2>/dev/null
cd "/Users/yunshuxie/belong to xys/claude_workspace/mc-skin-generator"
nohup .venv/bin/uvicorn app.main:app --reload --port 8000 > /tmp/uvicorn_dev.log 2>&1 &
disown
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
```

Expected: `200`. Then open `http://localhost:8000` in a browser, generate a skin, type an instruction like "把衬衫改成蓝色" into the new input, click "应用修改", and confirm the 3D viewer updates without a page reload and the shirt color visibly changes. Check the browser console for errors.

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html app/static/app.js app/static/style.css
git commit -m "feat: add conversational color-edit UI to the frontend"
```

---

### Task 8: Update docs

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `README.md`**

In the "Features" list, add a line after the existing "One AI call, mostly procedural" bullet:

```markdown
- **Conversational color edits** — after generating, describe a color change in plain language ("make the shirt blue") and it's applied in place, almost for free (no image re-upload, no Vision call)
```

In "Why mostly-procedural instead of a full AI-painted skin?", after the existing numbered list, add:

```markdown
The AI's palette uses 10 fixed-role slots (skin/hair/eye/shirt×2/arm×2/pants×2/shoe) followed by up to 6 freeform slots for logos or patterns on faces the model opportunistically decides to draw (up to 4 extra, beyond the mandatory 7). Both the AI-drawn pixels and the procedural fill read from this same array, which is also what makes conversational color edits cheap — changing one palette entry retints everywhere it's used, with no new Vision call.
```

Add a new `##` section after "## API" documenting the edit endpoint:

```markdown
| `POST` | `/api/skin/{id}/edit` | JSON: `{ "instruction": "..." }` | Same shape as `/api/generate`; `metadata.changed_roles` lists which named colors were updated |
```

(add this row to the existing API table rather than a new table)

In "## Architecture" → services list, add:

```markdown
│   ├── skin_store.py           # persists palette + raw pixel-index grids per skin, for later edits
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the "AI Pipeline" section, after the existing bullet points, add:

```markdown
- 调色板结构固定：`palette[0..9]` 是命名角色（`PALETTE_ROLES` in `skin_map.py`：skin_tone/hair_color/eye_color/shirt_main/shirt_shadow/arm_main/arm_shadow/pants_main/pants_shadow/shoe_color），`palette[10..15]`（最多 6 个）是 AI 自由选择的细节色，用于可选的"特色面"
- 除了固定的 7 个面，AI 可以自主追加最多 4 个"特色面"（比如背后徽标、袖子图案），用同样的索引格式，判断标准写在 prompt 里；未被追加的面照常走 `procedural.py`
```

Add a new subsection after "### Procedural Fill":

```markdown
### Skin State Persistence (`app/services/skin_store.py`)
生成时把完整调色板 + 每个 AI 画的面的原始索引网格（不是解码后的 hex）存成 `skins/<id>.json`，和 PNG 放在一起。这是对话式改色的基础——改色只需要换调色板里一个槽位的值，重新解码已经存好的索引网格即可，不需要重新调用 Vision。

### Conversational Color Edit
`POST /api/skin/{id}/edit`，body 是 `{"instruction": "..."}`。`claude_vision.interpret_color_edit` 用一次纯文本（不带图）小调用把指令映射到 `PALETTE_ROLES` 里的具名角色 + 新 hex 值；`claude_vision.apply_color_edit` 把改动写回调色板对应槽位，重新解码 AI 面 + 重新程序化生成 + 重新组装，原地覆盖同一个 `skin_id`。
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document fixed-role palette, detail faces, and conversational edit endpoint"
```

---

## Self-Review Notes

- **Spec coverage:** Fixed-role palette (Task 1, 4) ✓. Detail faces (Task 4) ✓. `shade_face` orientation shading (Task 2) ✓. Persistence (Task 3) ✓. Edit endpoint (Task 5, 6) ✓. Frontend (Task 7) ✓. Docs (Task 8) ✓.
- **Type consistency checked:** `generate_skin_data` returns a 3-tuple everywhere it's called (Task 4 definition, Task 6 router, Task 6 test mocks). `decode_indexed_grid` (public, no underscore) is the name used consistently from Task 4 onward — Task 6's router does not need to import it directly since `apply_color_edit` (Task 5) wraps it. `generate_procedural_regions(colors, model, exclude_keys=...)` signature matches between Task 2's definition and every call site in Tasks 4 and 5.
- **No placeholders:** every step has complete, runnable code — verified by re-reading each task's Step 3 before finalizing this plan.
