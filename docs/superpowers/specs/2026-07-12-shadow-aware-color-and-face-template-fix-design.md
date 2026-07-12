# Shadow-Aware Color Extraction + Face Template Fix — Design

## Problem

User testing of the photo-derived pixel rendering pipeline (`app/services/photo_render.py` +
`app/services/claude_vision.py`, per `docs/superpowers/specs/2026-07-10-photo-derived-pixel-rendering-design.md`)
surfaced four distinct defects. Each root cause was confirmed by reading the current code and, for the
shadow issue, by running the actual functions on synthetic inputs (not guessed).

### Defect 1 — Shadows produce color chaos

**Symptom:** "未能正确识别原图中阴影，导致颜色混乱."

**Root cause:** `downsample_dominant` faithfully preserves every lighting variation as a distinct color
(a soft-gradient shirt region downsampled to `body_front`'s 12×8 target already yields ~25 distinct
colors — verified empirically). `quantize_shared` then has only a `max_colors=24` shared budget across
ALL five photo regions (torso + 4 limbs), so shading gradients of one garment consume the budget and
force genuinely-different materials (and different body parts) to collapse into shared palette slots —
chaotic, wrong colors.

The earlier instinct to *raise* `max_colors` is the wrong direction: more palette room would reproduce
shading gradients more faithfully, the opposite of the goal. Minecraft's own engine re-lights the 3D
model, so the texture should carry each material's flat *固有色* (intrinsic color), with the photo's
baked-in shadows removed.

### Defect 2 — Eyes render as a single eye

**Symptom:** "眼睛变成了一只."

**Root cause:** `build_head_front` places eye pixels at a single central column range (`(3,)` narrow or
`(3, 4)` round) for both eye rows. There is only one eye position in the code — no left/right symmetry.

### Defect 3 — Mouth renders as a big rectangle

**Symptom:** "嘴巴变成了一条大长方形."

**Root cause:** `build_head_front`'s mouth is `for row in (5, 6): for col in (2, 3, 4, 5)` — a solid
4-wide × 2-tall block, i.e. literally a filled rectangle, with no shaping and no width variation.

### Defect 4 — Hair not restored at all

**Symptom:** "头发完全未还原."

**Root cause:** A direct contradiction between the prompt and the extraction code. `_build_prompt` tells
the model the `head` bbox should "tightly bound just the face (forehead to chin), not the whole
head/hair." But `extract_face_colors` samples `hair_color` from rows 0–2 of that same crop, assuming it
contains hair. When the model obeys the prompt (face-only crop), rows 0–2 are just forehead skin — so
"hair_color" is actually forehead skin, and the head reads as having no hair.

## Design

### Fix 1 — Shadow-aware color consolidation (new step in `photo_render.py`)

**User decisions (settled):** collapse each shaded material to its **highlight / brighter** tone; apply
to **torso + all four limbs**.

**Mechanism — cluster in HSV, collapse the value (lightness) axis per material:**

Shadow's defining property is that it preserves hue and (roughly) saturation while lowering value. So
colors that share a hue+saturation but differ in value are "the same material under different lighting"
and should collapse to one representative.

A new function `consolidate_shadows(rgb_grids, ...) -> dict[str, list[list[tuple[int,int,int]]]]` runs
**before** `quantize_shared`, rewriting each region's per-cell RGB grid so shadow variants of one
material already share one color. Pipeline becomes:
`downsample_dominant` (per region) → **`consolidate_shadows` (across all regions)** → `quantize_shared`.

Algorithm:
1. Gather all distinct colors across all regions' downsampled grids, convert each to HSV
   (`colorsys.rgb_to_hsv`, stdlib, no new dependency).
2. Partition colors into clusters. Two colors join the same cluster when they are "the same material":
   - **Saturated colors** (S ≥ `SAT_FLOOR`): same material iff hue difference ≤ `HUE_TOL` AND saturation
     difference ≤ `SAT_TOL`. Value (lightness) is deliberately ignored here — that's the shadow axis.
   - **Near-gray colors** (S < `SAT_FLOOR`): hue is numerically unstable and perceptually meaningless for
     grays, so clustering by hue would wrongly merge, e.g., black pants and a mid-gray shirt. Instead
     near-grays cluster by **coarse value band** (value quantized into `GRAY_VALUE_BANDS` bands): two
     near-grays are the same material iff they fall in the same value band. This keeps black / gray /
     white distinct while still merging minor shadow/JPEG noise within one gray material.
3. For each cluster, choose the representative color = the member at a **high value percentile**
   (`HIGHLIGHT_PERCENTILE`, e.g. 85th) — the "highlight/brighter" tone the user chose, but a percentile
   rather than the absolute max so a single blown-out specular pixel doesn't hijack the whole material.
   The representative's hue+saturation is the cluster's dominant (most-common) hue+saturation, recombined
   with that high-percentile value, then converted back to RGB.
4. Rewrite every cell to its cluster's representative RGB.

`quantize_shared` still runs afterward as a safety net / final palette cap, but now operates on an
already-consolidated set, so it no longer causes cross-material chaos. Its `max_colors` stays 24 (now
comfortably sufficient, since shadow variants are already gone — this can be revisited if real photos
prove otherwise, but is not raised speculatively).

**Tunable constants** (defined at module top, covered by tests): `SAT_FLOOR`, `HUE_TOL`, `SAT_TOL`,
`GRAY_VALUE_BANDS`, `HIGHLIGHT_PERCENTILE`. Initial values are a starting point to be validated against a
real photo during implementation, not hard guarantees.

**Head is unaffected** by this step — the face uses measured band colors + a template (Fix 2/3/4), not
crop-and-quantize, so it never enters `consolidate_shadows`.

### Fix 2 & 3 — Redesigned face template with two eyes and a shaped, width-variable mouth

`build_head_front(colors, eye_shape, mouth_width)` gains a `mouth_width` parameter (like the existing
`eye_shape`). The 8×8 layout (columns 0–7, rows 0–7):

```
row 0  H H H H H H H H     rows 0-1: hair
row 1  H H H H H H H H
row 2  . . E . . E . .     rows 2-3: two symmetric eyes (narrow: 1px each)
row 3  . . E . . E . .                (round: 2px each -> cols 1-2 and 5-6)
row 4  . . . . . . . .     row 4: nose/cheek (skin)
row 5  . . M M M M . .     row 5: mouth (1px tall). wide: cols 2-5 (4px)
row 6  . . . . . . . .     row 6: chin (skin).       small: cols 3-4 (2px)
row 7  . . . . . . . .     row 7: neck (skin)
```

- **Eyes** (rows 2–3, 2px tall): two eyes, symmetric about the vertical centerline.
  - `eye_shape == "narrow"` → 1px wide each: left eye col 2, right eye col 5.
  - `eye_shape == "round"` → 2px wide each: left eye cols 1–2, right eye cols 5–6.
- **Mouth** (row 5 only, 1px tall — not a 2-row block): centered horizontal segment.
  - `mouth_width == "small"` → 2px: cols 3–4.
  - `mouth_width == "wide"` → 4px: cols 2–5.
  - Row 6 is chin (skin), no longer mouth.

`mouth_width` is a new categorical field the model returns (see Fix 4's prompt change), classified from
the photo's actual mouth width — mirroring how `eye_shape` is already classified. It is advisory-only
(never validated beyond membership in `{"small", "wide"}`, defaulting to `"small"` if absent/invalid).

### Fix 4 — Reconcile the head bbox contract so hair is actually sampled

The head crop must contain hair for row 0–1 hair sampling to work. Change `_build_prompt`'s `head` rule
from "tightly bound just the face (forehead to chin), not the whole head/hair" to bound the **whole head
including hair — from the top of the hair down to the chin**. The existing 8-row template already
reserves rows 0–1 for hair and rows 2–7 for the face, so a whole-head crop maps correctly: hair lands in
the top band, face features in the lower bands. `extract_face_colors`'s row-band assignments are
unchanged — they were always written for a hair-inclusive crop; only the prompt was inconsistent with
them.

For a genuinely bald/hairless subject, rows 0–1 sample the top of the head (skin), so `hair_color` ≈
`skin_tone` — which is correct (a bald head's "hair" region is skin). The `hair_style == "bald"` path in
`_render_photo` (which already sets `head_fill_color = skin_tone`) stays as-is.

## Validation / Testing

- New `consolidate_shadows` unit tests (`tests/test_photo_render.py`): a synthetic region with two flat
  materials each in two shades (e.g. lit-red/shadow-red + lit-blue/shadow-blue) collapses to exactly two
  colors, each the brighter shade; a black + mid-gray + white input stays three distinct colors (the
  grayscale-safety case); a single-material shaded gradient collapses to one color.
- `build_head_front` tests: assert two distinct eye positions symmetric about center (left-half and
  right-half both contain an eye pixel) for both `eye_shape` values; assert mouth occupies exactly one
  row with the correct width for each `mouth_width`; assert hair fills rows 0–1.
- `extract_face_colors` unchanged, existing tests still pass (its band math didn't change).
- `_build_prompt` test asserts the head rule now says to include hair and that `mouth_width` is
  requested; `_validate_regions_response` (or wherever categorical fields are parsed) accepts/round-trips
  `mouth_width`.
- End-to-end `generate_skin_data` test updated for the new `_render_photo` → `consolidate_shadows` wiring
  and the `mouth_width` field threading through to `build_head_front`.
- Manual verification against the user's real test image before claiming done: confirm hair color
  appears, two eyes, a non-rectangular mouth, and that a shaded garment reads as one flat intrinsic color
  rather than a noisy gradient.

## Out of scope

- Any change to `skin_map.py` UV coordinates, `skin_assembler.py`, `skin_store.py`, `procedural.py`, or
  the API/router/frontend.
- Re-introducing the retired conversational color-edit feature.
- Perceptual color-difference (CIE Lab / ΔE) clustering — HSV is sufficient and dependency-free for this
  scale; a Lab upgrade can be a future refinement if HSV tuning proves inadequate on real photos.
