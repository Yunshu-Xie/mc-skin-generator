# Photo-Derived Pixel Rendering — Design

## Problem

Manual testing of the shipped pipeline (single Vision call → the model directly emits palette-index pixel grids for 11 "mandatory" faces: head ×6, `body_front`, and the four limb-front faces) against a real test image (an anime-style portrait with a red-trimmed collar, a small red diamond logo on the chest, white sun-sleeve gloves distinct from bare skin, and a light gray outer layer over a white base) showed severe fidelity loss:

- The red collar trim and chest logo did not appear at all — the torso rendered as a flat, undifferentiated white.
- The eyes were rendered oddly.
- The arm sleeves/gloves weren't distinguished from bare skin at all.
- The clothing had no visible layering/depth despite the source having several distinct, contrasting color regions.

**Root cause:** asking an LLM to directly author a JSON grid of per-pixel palette indices requires precise spatial reasoning about where multiple small, contrasting color regions sit relative to each other — a task LLMs are unreliable at, especially at Minecraft's tiny per-face resolutions (as small as 4×4). Prompt-engineering iterations on this same mechanism (fixed-role palette, mandatory limb fronts, `face_features` guidance) improved things incrementally but did not fix the underlying reliability problem, because the model is still the one placing every pixel.

Reference research (see Approach discussion below) confirms: real "photo → pixel art" tools do this with deterministic image processing (crop, resize with dominant-color-per-cell, then color quantization), not by asking a model to hand-place pixels. No existing open-source project solves the harder sub-problem of *automatically locating* body-part regions in an arbitrary uploaded photo (the one closely-related project found, `paulknewton/minecraft-skin-generator`, requires the user to manually supply per-part coordinate offsets) — for that, this design still relies on the Vision LLM, but only for coarse region localization, which is a task it's much better suited to than pixel-level authorship.

## Design

### Priority decision

Fidelity is the priority for this redesign. The conversational color-edit feature (`POST /api/skin/{id}/edit`, `interpret_color_edit`, `apply_color_edit`) depends on the fixed 10-slot `PALETTE_ROLES` structure, which this design removes in favor of a photo-derived adaptive palette. **The edit feature is retired in this pass** (route, functions, and its frontend UI removed) rather than half-adapted — re-enabling conversational editing against an adaptive palette is a separate future redesign, explicitly out of scope here. `skin_store.py`'s persistence (palette + raw index grids per skin) is kept as-is: it's still exactly the data a future edit redesign would need, and generation still benefits from persisting its result even without an edit endpoint consuming it yet.

### Step 1 — Region localization (single Vision call, replaces today's pixel-emission call)

The prompt no longer asks for any pixel grids or palette. It asks for:

```json
{
  "hair_style": "short|long|bald|hat|helmet",
  "eye_shape": "narrow|round",
  "regions": {
    "head":      {"visible": true, "bbox": [0.30, 0.05, 0.68, 0.35]},
    "torso":     {"visible": true, "bbox": [0.20, 0.30, 0.75, 0.70]},
    "right_arm": {"visible": true, "bbox": [0.05, 0.30, 0.25, 0.65]},
    "left_arm":  {"visible": true, "bbox": [0.70, 0.30, 0.95, 0.65]},
    "right_leg": {"visible": false},
    "left_leg":  {"visible": false}
  }
}
```

`bbox` is `[x0, y0, x1, y1]` as fractions (0–1) of the source image's width/height, top-left origin. A region with `"visible": false` omits `bbox` entirely (a photo that only shows the upper body legitimately has no leg region — this is normal, not an error).

Validation is more forgiving than the previous pixel-grid validation, because a bad *single region* now degrades gracefully (that one part falls back to a flat-color fill) instead of corrupting the whole face:
- Retry the whole call (same 2-retry budget as before) only if `hair_style`/`eye_shape` are missing/invalid, or if `regions` itself isn't a dict containing all 6 expected keys.
- Within a valid `regions` dict, a malformed individual entry (bad bbox shape, `x1 <= x0` or `y1 <= y0`, out-of-`[0,1]`-range values) is treated as `visible: false` for that region rather than failing the response — this is the same "graceful degradation over strict failure" philosophy `generate_procedural_regions`'s existing missing-front-data fallback already uses.

### Step 2 — Torso/limbs: crop → dominant-color-per-cell downsample → shared color quantization (new module, e.g. `app/services/photo_render.py`, Pillow only, no new dependency)

For each of `torso`, `right_arm`, `left_arm`, `right_leg`, `left_leg` with `visible: true`:

1. **Crop** the source photo to the region's bbox (converted to pixel coordinates, clamped to image bounds).
2. **Downsample to the target face's exact pixel dimensions** (e.g. 12×8 for `body_front`, 12×4/12×3 for an arm front) using **dominant-color-per-cell**, not blur-averaging: for each output cell, take the corresponding source sub-rectangle and pick its single most common color via Pillow's `Image.getcolors()` histogram on that sub-rectangle (call it with `maxcolors=` the tile's pixel count, not the 256 default — `getcolors()` silently returns `None` if the tile has more distinct colors than `maxcolors`, which would otherwise be a real footgun on a busy source region). This is the standard technique real pixel-art-from-photo tools use specifically to avoid a contrasting boundary (e.g. red trim against a white base) blurring into a muddy in-between color.
3. Once all `visible` regions are downsampled, **combine every pixel from all of them into one shared color-quantization pass** using Pillow's built-in `Image.quantize()` (median-cut), producing one adaptive palette capped at 24 colors — a shared pass (rather than one quantization per region) keeps e.g. skin tone consistent between an arm and a leg instead of each independently drifting to a slightly different quantized shade.
4. Map each region's per-cell dominant color to its nearest quantized palette index. The result is exactly the same shape the rest of the system already consumes: a `palette: list[str]` of hex colors, and one `{part}_front: list[list[int]]` index grid per visible region.

A region with `visible: false` (or a degenerate bbox) simply produces no front grid. `generate_procedural_regions`'s existing fallback path — added in the previous plan specifically for a front face that isn't present — already flat-fills that whole part with a default named color, so **no new fallback logic is needed for the not-visible-in-photo case.**

### Step 3 — Head/face: real colors, existing template layout (hybrid)

An 8×8 grid is too small for a literal photographic downsample of a face — naive resizing risks a mushy, uniformly-skin-toned blob that loses eye position entirely, which would be a regression, not an improvement. Instead:

- Crop the `head` bbox, then split it into the same row-bands the current hand-authored template already defines (row 0–1 hair/forehead, row 2–3 eyes, row 4 nose, row 5–6 mouth/chin, row 7 neck).
- `skin_tone`: dominant color of the nose/cheek band (row 4), via `getcolors()`.
- `hair_color`: dominant color of the top band (rows 0–1); falls back to `skin_tone` if that band is degenerate (e.g. bbox too short).
- `eye_color` / `mouth_color`: within the eye-row band and mouth-row band respectively, run `Image.quantize(colors=2)` on just that band and take the cluster with the **smaller pixel count** of the two as the feature color (the deciding criterion is prevalence, not darkness — eyes/mouth occupy less area than the surrounding skin within their band, so the minority cluster is the feature; it typically is also darker, but that's a consequence, not a second condition to check).
- Eye pixel width (1px vs 2px) is still driven by the AI's `eye_shape` field from Step 1, same rule as before.
- The pixels are then placed into the **existing hand-authored 8×8 template** (unchanged row assignments), just using these measured colors instead of AI-guessed hex values.

`head_top`, `head_back`, `head_left`, `head_right`, `head_bottom` are **no longer derived from the photo at all** — a front photo never shows the back/top/sides of someone's head anyway, so asking the model to hallucinate believable pixel content there (as the current design does) was always a guess. These 5 faces become a flat single-tone procedural fill from one resolved color: `hair_color`, or `skin_tone` if `hair_style == "bald"` (this resolution happens in `claude_vision.py`, which passes the single resolved value into `generate_procedural_regions`'s `colors` dict under a new key, e.g. `head_fill_color`).

### What stays unchanged

- `app/services/skin_map.py`'s `FaceRect`/`get_all_regions`/`PIXEL_KEY_MAP` (UV coordinates) — untouched.
- `app/services/skin_assembler.py` — untouched; it only ever consumed `{face_key: hex grid}`, which this design still produces.
- `app/services/skin_store.py` — untouched; still persists palette + raw index grids per skin.
- The previous plan's `propagate_front_row` mechanism in `app/services/procedural.py` (back/left/right of torso/arms/legs copying their front face's per-row color) — unchanged; it only reads whatever hex ended up in `pixel_data[f"{part}_front"]`, indifferent to whether that came from the AI or from `photo_render.py`.
- `generate_procedural_regions`'s existing fallback-to-flat-fill when a front grid is absent — unchanged, now also the mechanism for "photo didn't show this body part."

### What changes in `procedural.py`

`AI_GENERATED_KEYS` shrinks from 11 to 6: `head_front`, `body_front`, `right_arm_front`, `left_arm_front`, `right_leg_front`, `left_leg_front` (the 5 non-front head faces are no longer AI/photo-derived, so they move to procedural). `generate_procedural_regions` gains a small new branch: for the `head` group, `top`/`back`/`left`/`right`/`bottom` are all flat-filled from `colors["head_fill_color"]` (no front-row propagation — hair is reasonably uniform across the whole head, unlike a sleeve boundary on an arm, so a simple flat fill is the right level of effort here, not a new propagation mechanism).

### What's retired

- `app/services/skin_map.py`: `PALETTE_ROLES`, `MIN_PALETTE_SIZE`, `MAX_PALETTE_SIZE` — only ever consumed by the fixed-role palette logic and the edit feature, both retired.
- `app/services/claude_vision.py`: `interpret_color_edit`, `apply_color_edit`, `EDIT_MAX_TOKENS`, and all `face_features`-as-AI-guessed-hex prompt logic (colors are now measured, not guessed). `decode_indexed_grid` is **kept** — it's still exactly how an index grid becomes a hex grid, just fed by `photo_render.py`'s quantization output instead of AI JSON.
- `app/models/schemas.py`: `EditSkinRequest`.
- `app/routers/skin.py`: the `POST /skin/{id}/edit` route and its `apply_color_edit`/`EditSkinRequest` imports. `POST /api/generate` and `GET /api/skin/{id}.png` are unchanged.
- `app/static/*`: the edit-controls UI (text input + button + JS handler) added by the previous plan's Task 7 — since the endpoint is gone, leaving that UI in place would 404 on submit.

## Testing

- New `tests/test_photo_render.py`: dominant-color-per-cell downsampling (given a small synthetic image with a known majority color per cell, assert the output matches), the shared-quantization + index-mapping step (given known RGB inputs across multiple regions, assert consistent/expected index assignment and that palette size is capped), and face-band color extraction (given a synthetic face-like test image with known skin/hair/eye/mouth colored regions, assert each is extracted correctly, including the not-enough-contrast/degenerate-band fallback to `skin_tone`).
- `tests/test_claude_vision.py`: rewritten for the new region-localization prompt/schema — validation tests for the retry-vs-graceful-degrade split (whole-response retry only for missing `hair_style`/`eye_shape`/malformed `regions` dict; a single bad region degrades to `visible: false`), and an integration-style test feeding a mocked region-localization response through the real `photo_render` functions against a real small test image, asserting the final `pixel_data` shape matches what `skin_assembler` expects. All `interpret_color_edit`/`apply_color_edit` tests are deleted.
- `tests/test_procedural.py`: updated for the smaller `AI_GENERATED_KEYS` (6 entries) and the new head flat-fill branch (bald vs not, using `head_fill_color`).
- `tests/test_api.py`: the `POST /skin/{id}/edit` tests are deleted; `POST /api/generate` tests are updated for the new `generate_skin_data` internals (still mocked at the same boundary, so most of these tests need only their mock's shape to change, not their assertions).
- `tests/test_skin_map.py`: `PALETTE_ROLES`/`MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE` tests are deleted.

## Out of scope

- Re-enabling conversational color editing against the new adaptive palette (e.g. tagging palette entries with free-text semantic labels after generation) — a separate future redesign.
- Any change to `Classic`/`Slim` model handling, UV coordinates, or the PNG assembly/download/3D-preview flow.
- Any face/pose detection library (OpenCV, mediapipe, dlib) — deliberately not used, since those are trained on real human photos and unreliable on illustration/anime-style art, which this app must also support; region localization stays with the Vision LLM itself.
