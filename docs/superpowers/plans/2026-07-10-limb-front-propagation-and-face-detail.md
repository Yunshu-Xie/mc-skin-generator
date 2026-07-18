# Limb Front-Propagation + Face Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix limbs/torso still reading as flat outlined boxes by replacing synthetic shading (`shade_face`) with front-row color propagation, and improve `head_front` facial-feature fidelity via richer prompt guidance — per `docs/superpowers/specs/2026-07-10-limb-front-propagation-and-face-detail-design.md`.

**Architecture:** `right_arm_front`/`left_arm_front`/`right_leg_front`/`left_leg_front` become mandatory AI-drawn faces (like `head_front`/`body_front`), growing `AI_GENERATED_KEYS` from 7 to 11. `procedural.py`'s `shade_face` (HSL lightness shading) is deleted entirely and replaced with `propagate_front_row`: each part's back/left/right faces copy their front face's per-row middle-column color, so a clothing boundary visible on the front (sleeve length, pant length) stays consistent around the limb — no fabricated shading anywhere, since Minecraft's own in-game lighting already shades the model. Top/bottom cap faces stay flat single-tone fills. `claude_vision.py`'s prompt gains a `face_features` object (`eye_shape`/`eyebrow_color`/`mouth_color`) that the model fills in before choosing pixels, purely to improve reasoning — never parsed, validated, or surfaced by code.

**Tech Stack:** Python 3.11, FastAPI, Pillow, `openai` SDK against Gemini's OpenAI-compatible endpoint, pytest.

## Global Constraints

- All Python files use `from __future__ import annotations` (existing codebase convention).
- No new pip dependencies.
- Every new/changed function needs a test; run `.venv/bin/python -m pytest -q` (full suite) after each task and confirm it's all green before committing — a task that leaves the suite red is not done, regardless of whether its own new tests pass.
- `.venv/bin/ruff check <changed files>` must be clean before committing each task.
- Don't touch `app/services/skin_assembler.py`, `app/services/skin_store.py`, `app/routers/skin.py`, `app/models/schemas.py`, `app/static/*`, or `PALETTE_ROLES`/`MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE` in `skin_map.py` — this plan is scoped to `procedural.py` and `claude_vision.py` (+ their tests) only.
- `tests/test_api.py` needs no changes — its `_make_dummy_pixel_data()` fixture already fills all 36 base regions uniformly regardless of the AI/procedural split.

---

### Task 1: Mandatory limb-front faces + front-row propagation

This is one task, not two, because the pieces are load-bearing for each other: growing `AI_GENERATED_KEYS` to include the 4 limb-front keys immediately makes every existing test that builds a "complete" mock response (`_valid_response()` in `tests/test_claude_vision.py`) invalid until its fixture is updated to the new dimensions, and `generate_procedural_regions`'s new signature immediately breaks both call sites in `claude_vision.py` (`generate_skin_data`, `apply_color_edit`) until they're updated to match. Splitting these across two commits would leave the suite red in between.

**Files:**
- Modify: `app/services/procedural.py` (full rewrite)
- Modify: `tests/test_procedural.py` (full rewrite)
- Modify: `app/services/claude_vision.py` (`_build_prompt`'s mandatory-face list, and the `generate_procedural_regions` call sites in `generate_skin_data`/`apply_color_edit`)
- Modify: `tests/test_claude_vision.py` (`_valid_response` fixture dimensions, one new test)

**Interfaces:**
- Produces: `AI_GENERATED_KEYS: set[str]` (now 11 entries), `propagate_front_row(front_grid: list[list[str]], target_width: int) -> list[list[str]]`, `generate_procedural_regions(pixel_data: dict[str, list[list[str]]], colors: dict[str, str], model: ModelType, exclude_keys: frozenset[str] = frozenset()) -> dict[str, list[list[str]]]` — consumed by Task 2 (unchanged further).

- [ ] **Step 1: Write the failing tests for `procedural.py`**

Replace `tests/test_procedural.py` entirely with:

```python
"""Tests for procedural — front-row propagation for non-AI-painted wrap faces."""

from app.services.procedural import (
    AI_GENERATED_KEYS,
    generate_procedural_regions,
    propagate_front_row,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions


def test_ai_generated_keys_includes_all_four_limb_fronts():
    assert AI_GENERATED_KEYS == {
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

    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "classic")

    assert set(out.keys()) == expected_keys
    for key, grid in out.items():
        group, face = PIXEL_KEY_MAP[key]
        rect = regions[group][face]
        assert len(grid) == rect.h
        assert all(len(row) == rect.w for row in grid)


def test_generate_procedural_regions_never_touches_ai_keys():
    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "classic")
    assert AI_GENERATED_KEYS.isdisjoint(out.keys())


def test_generate_procedural_regions_wrap_faces_match_front_rows():
    front = [[f"#{i:02X}{i:02X}{i:02X}"] * 8 for i in range(12)]  # row i -> shade #i,i,i
    out = generate_procedural_regions(
        _pixel_data_with_fronts(body_front=front), _colors(), "classic"
    )
    for row_idx, row in enumerate(front):
        row_color = row[len(row) // 2]
        assert out["body_back"][row_idx] == [row_color] * 8
        assert out["body_left"][row_idx] == [row_color] * 4
        assert out["body_right"][row_idx] == [row_color] * 4


def test_generate_procedural_regions_top_bottom_are_flat_named_color():
    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "classic")
    assert all(cell == "#3355AA" for row in out["body_top"] for cell in row)
    assert all(cell == "#3355AA" for row in out["body_bottom"] for cell in row)
    assert all(cell == "#C4A882" for row in out["right_arm_top"] for cell in row)
    assert all(cell == "#223355" for row in out["right_leg_top"] for cell in row)


def test_generate_procedural_regions_leg_bottom_uses_shoe_color():
    out = generate_procedural_regions(
        _pixel_data_with_fronts(), _colors(shoe_color="#ABCDEF"), "classic"
    )
    assert all(cell == "#ABCDEF" for row in out["right_leg_bottom"] for cell in row)
    assert all(cell == "#ABCDEF" for row in out["left_leg_bottom"] for cell in row)


def test_generate_procedural_regions_falls_back_to_flat_when_front_missing():
    """Old persisted skins (pre-migration) may lack the new mandatory front grids."""
    out = generate_procedural_regions({}, _colors(), "classic")
    assert all(cell == "#C4A882" for row in out["right_arm_back"] for cell in row)
    assert all(cell == "#223355" for row in out["left_leg_left"] for cell in row)


def test_generate_procedural_regions_respects_exclude_keys():
    out = generate_procedural_regions(
        _pixel_data_with_fronts(),
        _colors(),
        "classic",
        exclude_keys=frozenset({"right_arm_back"}),
    )
    assert "right_arm_back" not in out
    assert "right_arm_left" in out  # sibling face still procedural


def test_generate_procedural_regions_slim_arm_width():
    out = generate_procedural_regions(_pixel_data_with_fronts(), _colors(), "slim")
    assert len(out["right_arm_top"][0]) == 3
    assert len(out["left_arm_top"][0]) == 3


def test_generate_procedural_regions_falls_back_to_defaults_on_missing_colors():
    out = generate_procedural_regions(_pixel_data_with_fronts(), {}, "classic")
    assert len(out) > 0
    assert out["body_top"][0][0]  # some non-empty hex string
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_procedural.py -v`
Expected: FAIL — `ImportError: cannot import name 'propagate_front_row'`

- [ ] **Step 3: Implement `procedural.py`**

Replace `app/services/procedural.py` entirely with:

```python
"""Procedural (non-AI) generation for skin faces that don't need per-pixel detail.

Head (6 faces), body_front, and the four limb-front faces are generated
pixel-by-pixel by the AI (see PALETTE_ROLES in skin_map.py for the shared
palette). Every other face is derived directly from its part's front face:
back/left/right copy the front face's per-row color (so a clothing boundary
visible on the front — e.g. a sleeve ending partway down the arm — stays
consistent all the way around the limb), while top/bottom (the small
end-cap faces) are a flat fill from a handful of named colors. No synthetic
shading is added anywhere — Minecraft's own in-game lighting already shades
the 3D model.
"""

from __future__ import annotations

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

- [ ] **Step 4: Run `test_procedural.py` to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_procedural.py -v`
Expected: all PASS

- [ ] **Step 5: Update `test_claude_vision.py`'s fixture and add a mandatory-face test**

The full suite is still red at this point (`AI_GENERATED_KEYS` now has 11 entries, but `_valid_response()` still only sizes grids as 8×8 except for `body_front`, and `generate_procedural_regions`'s call sites in `claude_vision.py` still use the old 2-positional-arg form) — this step and the next two fix that.

In `tests/test_claude_vision.py`, replace:

```python
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
```

with:

```python
_MANDATORY_FACE_DIMS = {
    "head_front": (8, 8),
    "head_back": (8, 8),
    "head_top": (8, 8),
    "head_bottom": (8, 8),
    "head_left": (8, 8),
    "head_right": (8, 8),
    "body_front": (12, 8),
    "right_arm_front": (12, 4),
    "left_arm_front": (12, 4),
    "right_leg_front": (12, 4),
    "left_leg_front": (12, 4),
}


def _valid_response(n_freeform: int = 0, extra_faces: dict | None = None) -> dict:
    data = {
        "description": "A test character",
        "hair_style": "short",
        "palette": _fixed_palette(n_freeform),
    }
    for key, (h, w) in _MANDATORY_FACE_DIMS.items():
        data[key] = [[0] * w for _ in range(h)]
    if extra_faces:
        data.update(extra_faces)
    return data
```

Then append this new test directly after `test_validate_and_decode_wrong_grid_size_is_incomplete`:

```python
def test_validate_and_decode_missing_arm_front_is_incomplete():
    data = _valid_response()
    del data["right_arm_front"]
    pixel_data, _raw, _colors, _metadata, ok = _validate_and_decode(data, "classic")
    assert ok is False
    assert "right_arm_front" not in pixel_data
    assert "body_front" in pixel_data  # other mandatory grids still decoded
```

- [ ] **Step 6: Fix the two `generate_procedural_regions` call sites in `claude_vision.py`**

In `app/services/claude_vision.py`, in `generate_skin_data`, replace:

```python
    pixel_data.update(
        generate_procedural_regions(colors, model, exclude_keys=frozenset(pixel_data.keys()))
    )
```

with:

```python
    pixel_data.update(
        generate_procedural_regions(
            pixel_data, colors, model, exclude_keys=frozenset(pixel_data.keys())
        )
    )
```

In `apply_color_edit`, replace the identical block the same way:

```python
    pixel_data.update(
        generate_procedural_regions(
            pixel_data, colors, model, exclude_keys=frozenset(pixel_data.keys())
        )
    )
```

Also update `_build_prompt`'s JSON structure so real generation actually requests the 4 new mandatory grids (`face_features` and the richer facial rules come in Task 2 — this step only adds the new mandatory keys to keep the prompt and `AI_GENERATED_KEYS` in sync). Replace the structure block:

```python
  "head_front":  [[palette index per pixel] × 8 cols] × 8 rows,
  "head_back":   [8×8 palette indices],
  "head_top":    [8×8 palette indices],
  "head_bottom": [8×8 palette indices],
  "head_left":   [8×8 palette indices],
  "head_right":  [8×8 palette indices],
  "body_front":  [12×8 palette indices]
}}
```

with:

```python
  "head_front":      [[palette index per pixel] × 8 cols] × 8 rows,
  "head_back":       [8×8 palette indices],
  "head_top":        [8×8 palette indices],
  "head_bottom":     [8×8 palette indices],
  "head_left":       [8×8 palette indices],
  "head_right":      [8×8 palette indices],
  "body_front":      [12×8 palette indices],
  "right_arm_front": [12×{arm_w} palette indices],
  "left_arm_front":  [12×{arm_w} palette indices],
  "right_leg_front": [12×4 palette indices],
  "left_leg_front":  [12×4 palette indices]
}}
```

And update the two prose rules that reference the old mandatory-face count/list. Replace:

```python
- body_front: torso/shirt front, {arm_w}-px-wide-arm model
- Most characters do NOT need anything beyond the 7 faces above. ONLY if \
the photo shows a genuinely distinctive design element (a back logo, a \
sleeve pattern, a belt, a cape) that would look wrong as a flat color, add \
up to 4 more faces as extra top-level keys using the same [row][col] \
palette-index format. Valid extra keys: {extra_face_keys}
- shirt_main/shirt_shadow, arm_main/arm_shadow, pants_main/pants_shadow, \
and shoe_color are also used to procedurally color any face you don't draw \
yourself — pick them to match the photo's clothing
```

with:

```python
- body_front, right_arm_front, left_arm_front, right_leg_front, \
left_leg_front are the front-facing torso/limb pixels visible in most \
photos — paint them to match the photo's actual clothing and skin as \
closely as you can (e.g. where a sleeve ends and skin begins)
- Most characters do NOT need anything beyond the 11 faces above. ONLY if \
the photo shows a genuinely distinctive design element (a back logo, a \
sleeve pattern, a belt, a cape) that would look wrong as a flat color, add \
up to 4 more faces as extra top-level keys using the same [row][col] \
palette-index format. Valid extra keys: {extra_face_keys}
- shirt_main, arm_main, and pants_main also color the back/side faces of \
the torso/arms/legs (derived from the front faces you paint) and any \
top/bottom cap faces you don't draw yourself; shoe_color fills the sole
```

(`extra_face_keys` already excludes the 4 new mandatory keys automatically — it's computed as `key not in AI_GENERATED_KEYS`, and `AI_GENERATED_KEYS` already grew to 11 in Step 3.)

- [ ] **Step 7: Run the full suite to verify it's green**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS, including `tests/test_api.py` (needs no changes) and `test_apply_color_edit_retints_palette_and_reassembles` (its `state["pixel_grids"]` only has `body_front`, which now exercises the fallback-to-flat-fill path in `generate_procedural_regions` for the arm/leg faces it lacks, instead of crashing)

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff check app/services/procedural.py app/services/claude_vision.py tests/test_procedural.py tests/test_claude_vision.py
git add app/services/procedural.py app/services/claude_vision.py tests/test_procedural.py tests/test_claude_vision.py
git commit -m "feat: make limb-front faces mandatory, propagate front-row colors to wrap faces"
```

---

### Task 2: `face_features` prompt guidance for richer facial detail

**Files:**
- Modify: `app/services/claude_vision.py` (`_build_prompt`, `_call_vision`'s `max_tokens`)
- Modify: `tests/test_claude_vision.py`

**Interfaces:**
- No new public functions or signature changes — this task only changes prompt text and one constant (`max_tokens`). `face_features` is intentionally never parsed, validated, or returned by any function.

- [ ] **Step 1: Write a failing test asserting the new prompt content**

Append to `tests/test_claude_vision.py`:

```python
from app.services.claude_vision import _build_prompt


def test_build_prompt_includes_face_features_guidance():
    prompt = _build_prompt("classic")
    assert "face_features" in prompt
    assert "eye_shape" in prompt
    assert "eyebrow_color" in prompt
    assert "mouth_color" in prompt


def test_build_prompt_includes_mandatory_limb_fronts():
    prompt = _build_prompt("classic")
    for key in (
        "right_arm_front",
        "left_arm_front",
        "right_leg_front",
        "left_leg_front",
    ):
        assert key in prompt
```

- [ ] **Step 2: Run the tests to verify the first one fails**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v -k build_prompt`
Expected: `test_build_prompt_includes_face_features_guidance` FAILS (no `face_features` in the prompt yet); `test_build_prompt_includes_mandatory_limb_fronts` already PASSES (Task 1 added these keys to the prompt already) — that's expected, it's a regression guard for Task 1's change, not new-in-this-task behavior.

- [ ] **Step 3: Implement — add `face_features` and richer facial rules to `_build_prompt`, bump `max_tokens`**

In `app/services/claude_vision.py`, in `_build_prompt`, insert the `face_features` block into the JSON structure, immediately after `"hair_style": "short|long|bald|hat|helmet",` and before the blank line preceding `"palette": [`:

```python
  "description": "Brief description of what you see",
  "hair_style": "short|long|bald|hat|helmet",

  "face_features": {{
    "eye_shape": "narrow or round, based on the photo's eyes",
    "eyebrow_color": "#HEX, or 'none' if no distinct eyebrows are visible",
    "mouth_color": "#HEX of the person's lip/mouth color"
  }},

  "palette": [
```

Also update the freeform-palette-slots line to mention faces can use them for facial detail colors too. Replace:

```python
    "... optionally 0-6 more freeform colors (index 10-15) for logos, \
patterns, or accessories you want to draw"
```

with:

```python
    "... optionally 0-6 more freeform colors (index 10-15) for logos, \
patterns, accessories, or distinct eyebrow/mouth colors"
```

Then replace the eyes/mouth rule:

```python
- Eyes and mouth in head_front MUST use a palette index other than \
skin_tone (0) and hair_color (1) — reuse eye_color (2) or another index, so \
facial features are visible against the skin
```

with:

```python
- Fill in "face_features" first, based on careful observation of the \
photo — it guides the pixel choices below but is not itself validated or \
rendered.
- Eyes and mouth in head_front MUST use a palette index other than \
skin_tone (0) and hair_color (1) — reuse eye_color (2) by default. If \
face_features.eye_shape is "narrow", render each eye 1 pixel wide; if \
"round", render each eye 2 pixels wide. If face_features.mouth_color is \
meaningfully different from eye_color, use a freeform slot (10+) for the \
mouth instead of reusing eye_color. If face_features.eyebrow_color isn't \
"none", add a row of eyebrow-colored pixels at row 1.
```

In `_call_vision`, change `max_tokens=4000` to `max_tokens=5000` (headroom for the 4 extra mandatory grids from Task 1 plus the new `face_features` object).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_claude_vision.py -v -k build_prompt`
Expected: both PASS

Then run the whole suite:

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check app/services/claude_vision.py tests/test_claude_vision.py
git add app/services/claude_vision.py tests/test_claude_vision.py
git commit -m "feat: add face_features prompt guidance for eye/eyebrow/mouth detail"
```

---

## Self-Review Notes

- **Spec coverage:** `AI_GENERATED_KEYS` 7→11 (Task 1) ✓. `shade_face` deletion + `propagate_front_row` (Task 1) ✓. `generate_procedural_regions` new signature + top/bottom flat fill + shoe-bottom special case (Task 1) ✓. Fallback for missing front data, needed for old persisted skins (Task 1) ✓. `max_tokens` bump (Task 2) ✓. `face_features` prompt guidance, purely advisory (Task 2) ✓. `tests/test_api.py` needs no changes, confirmed and documented ✓.
- **Full-suite-green-per-commit checked:** Task 1 bundles the `procedural.py` rewrite with the `claude_vision.py` call-site fixes and the `test_claude_vision.py` fixture fix specifically because splitting them would leave the suite red between commits — noted explicitly in Task 1's intro.
- **Type consistency checked:** `generate_procedural_regions(pixel_data, colors, model, exclude_keys=...)` signature matches between its Task 1 definition and both call sites (also fixed in Task 1). `propagate_front_row(front_grid, target_width)` name and signature used consistently. `AI_GENERATED_KEYS` referenced identically everywhere it's imported.
- **No placeholders:** every step has complete, runnable code.
