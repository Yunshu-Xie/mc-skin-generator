# Limb Front-Propagation + Face Detail — Design

## Problem

The previous fidelity pass (see `docs/superpowers/specs/2026-07-09-skin-fidelity-and-color-edit-design.md`, implemented) replaced `flat_fill_with_border` with `shade_face`: a per-orientation HSL lightness bias plus a darker 1px edge ring. Manual testing after shipping that change showed it didn't actually fix the complaint — every procedural face is still exactly two flat tones (interior + edge ring), so arms/legs/torso still read as "描边长方体" (outlined rectangles), just recolored per orientation instead of uniformly.

Discussion with the user surfaced two separate, more targeted problems:

1. **Procedural faces have no relationship to the photo at all.** They're filled from a handful of named colors (`shirt_main`, `arm_main`, `pants_main`, …) with no concept of clothing coverage — e.g. if a shirt has a sleeve ending partway down the arm (visible in the photo), that boundary only exists in `body_front`'s AI-drawn pixels; the arm faces (100% procedural) have no way to reflect it. Adding synthetic shading (gradients, banding) doesn't fix this — it's decorating a wrong-shaped hole. The user also noted Minecraft's own in-game lighting already shades the 3D model; baking fake shadows into the texture on top of that is redundant.
2. **`head_front`'s facial features are underspecified.** The fixed 10-slot palette reserves exactly one color (`eye_color`) that's guaranteed distinct from `skin_tone`/`hair_color`. Eyes and mouth currently share that single color, which reads as flat/generic rather than reflecting the photo's actual face.

## Design

### Part 1 — Front-to-wrap color propagation replaces synthetic shading

**Make the four remaining visible-in-a-front-photo faces mandatory AI-drawn pixels**, matching how `head_front`/`body_front` already work:

- `AI_GENERATED_KEYS` (in `app/services/procedural.py`) grows from 7 to 11: add `right_arm_front`, `left_arm_front`, `right_leg_front`, `left_leg_front`.
- `_build_prompt` in `app/services/claude_vision.py` requires indexed grids for these 4 faces alongside the existing 7, with dimensions from `get_all_regions(model)` (12×4 classic / 12×3 slim for arms, 12×4 for legs — arms/legs are already narrower on Slim, `get_all_regions` already encodes this).
- These 4 keys are removed from the optional "detail face" candidate list automatically, since `_build_prompt`'s `extra_face_keys` is already derived as `key not in AI_GENERATED_KEYS`. No separate code path needed — `_validate_and_decode`'s existing `for key in AI_GENERATED_KEYS: decode(...)` loop and the retry logic in `generate_skin_data` pick these up for free by just growing the set.
- `max_tokens` in `_call_vision` increases from 4000 — start at 5000 and confirm against real usage during implementation (current usage was ~2,700–3,000 before this change per the prior spec; 4 more ≤12×4 grids adds a bounded, modest amount of output).
- This removes the "no data to propagate" fallback case entirely: body/right_arm/left_arm/right_leg/left_leg now *always* have a real AI-drawn front face to read from.

**Replace `shade_face` with row-propagation from the front face.** Delete `shade_face`, `ORIENTATION_BIAS`, `EDGE_STEP`, `_adjust_lightness`, and the `colorsys` import from `procedural.py` — no synthetic shading of any kind.

New function:

```python
def propagate_front_row(front_grid: list[list[str]], target_width: int) -> list[list[str]]:
    """Build a same-height grid where row N is a flat fill of front_grid[N]'s middle column.

    Used for the "wrap" faces (back/left/right) of a body part, so a clothing
    boundary visible on the front (e.g. a sleeve ending partway down the arm)
    stays consistent all the way around the limb, without any fabricated shading.
    """
```

For each of `body`, `right_arm`, `left_arm`, `right_leg`, `left_leg`: the `back`, `left`, and `right` faces are built by calling `propagate_front_row(pixel_data[f"{part}_front"], target_rect.w)`. This is always valid because Minecraft's UV layout gives every face of the same body part the same height as its front face (verified: 12 rows for body/arms/legs; only `top`/`bottom` differ in height and are handled separately below) — width can differ (e.g. body left/right are 4px vs front's 8px) and doesn't matter since we're flat-filling each row with one representative color.

`top`/`bottom` faces (the small end-caps — shoulder top, palm/sole, collar, waist) are **not** part of this propagation (they're a different axis — a cross-section, not a wrap-around). They keep a plain single-tone fill from the existing named colors: `shirt_main` for `body_top`/`body_bottom`, `arm_main` for arm tops/bottoms, `pants_main` for leg tops/bottoms. `right_leg_bottom`/`left_leg_bottom` keep the existing special case (flat `shoe_color`). No border darkening, no bias — literally the flat color, since the user explicitly doesn't want synthetic shading anywhere.

**`generate_procedural_regions` signature changes:**

```python
def generate_procedural_regions(
    pixel_data: dict[str, list[list[str]]],
    colors: dict[str, str],
    model: ModelType,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, list[list[str]]]:
```

It now needs the AI-decoded pixel data (to read `{part}_front` for row propagation) in addition to the named colors (still used for the flat top/bottom fills and as a fallback default dict via `DEFAULT_COLORS`). `exclude_keys` keeps its existing purpose: skipping any face the AI opportunistically drew as one of the (still-optional, still capped at 4) extra detail faces beyond the now-11 mandatory ones — e.g. a back logo.

Call sites in `app/services/claude_vision.py` (`generate_skin_data` and `apply_color_edit`) pass `pixel_data` through to `generate_procedural_regions` instead of just `colors`.

### Part 2 — Richer facial feature guidance (prompt-only, no schema change)

Add a `face_features` object to the JSON the model outputs, positioned **before** `palette` in the prompt's structure example so the model commits to describing the face before choosing colors/pixels for it (a lightweight forced-reasoning step, useful since `reasoning_effort="none"` disables actual thinking tokens):

```
"face_features": {
  "eye_shape": "narrow" | "round",
  "eyebrow_color": "#HEX or 'none' if no visible eyebrows",
  "mouth_color": "#HEX"
},
```

This field is **purely advisory** — the model fills it in to reason more carefully, but no code ever reads it. `_validate_and_decode` doesn't validate its shape or require its presence; it's not surfaced in `metadata` or the API response. If it's missing, malformed, or the model ignores it, nothing breaks — it's dropped along with the rest of the raw response after parsing.

`PIXEL ART RULES` prose gains concrete, actionable instructions tying `face_features` to actual pixel decisions:
- `eye_shape: "narrow"` → render each eye 1px wide; `"round"` → 2px wide.
- If `eyebrow_color` is set (not `"none"`) and visibly distinct from `hair_color`, add a row of eyebrow-colored pixels at row 1.
- If `mouth_color` is meaningfully different from `eye_color`, use a freeform palette slot (index 10+) for it instead of reusing `eye_color` — relaxes the current hard rule that eyes and mouth must share one index.

No changes to `PALETTE_ROLES`, `MIN_PALETTE_SIZE`/`MAX_PALETTE_SIZE`, or any validation code — this is entirely prompt text plus the one new (unvalidated, unused) JSON key.

## Testing

- `tests/test_procedural.py`: full rewrite. Remove all `shade_face`-related tests. Add tests for `propagate_front_row` (correct height, each row uses the front grid's middle-column color, works for narrower/wider target widths) and updated `generate_procedural_regions` (back/left/right rows match the front's row colors for body/arms/legs; top/bottom are flat single-tone; leg-bottom shoe-color special case still passes; `exclude_keys` still skips AI-drawn detail faces).
- `tests/test_claude_vision.py`: `AI_GENERATED_KEYS`-based fixtures grow to cover 11 mandatory faces instead of 7 (`_valid_response()` helper and friends). Add/update a test confirming a response missing one of the 4 new mandatory arm/leg-front grids is treated as incomplete (triggers retry), mirroring existing coverage for `head_front`/`body_front`. Update the `_validate_and_decode` call sites that construct `generate_procedural_regions` calls to pass `pixel_data`. `face_features` needs no dedicated validation test beyond confirming its presence/absence in the raw response doesn't affect parsing (already implied by existing "extra unknown keys are ignored" behavior, but add one explicit test since it's a new, intentionally-ignored field).
- `tests/test_api.py`: no change needed — `_make_dummy_pixel_data()` already fills all 36 base regions uniformly (it mocks the *combined* post-assembly pixel data, not the AI/procedural split), so it already covers the 4 newly-mandatory arm/leg-front keys.

## Out of scope

- No changes to `PALETTE_ROLES`, `MIN_PALETTE_SIZE`, `MAX_PALETTE_SIZE`, `skin_store.py`, the edit endpoint, or the frontend — this is purely a generation-quality change to `procedural.py` and `claude_vision.py`'s prompt/mandatory-key-set.
- No new dedicated `mouth_color`/`eyebrow_color` fixed palette slots — still 10 fixed + up to 6 freeform, per the earlier decision to keep the palette schema stable.
- No structured consumption of `face_features` anywhere in code, now or planned — it is intentionally a throwaway reasoning aid.
