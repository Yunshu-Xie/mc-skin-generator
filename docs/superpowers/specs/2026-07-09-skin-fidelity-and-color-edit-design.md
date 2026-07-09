# Skin fidelity fix + conversational color editing

## Problem

Manual testing of the mostly-procedural pipeline (see `app/services/procedural.py`, `app/services/claude_vision.py`) surfaced two quality issues:

1. **No facial features.** `head_front` only used 3 of the ≤8 shared palette colors (hair, skin, one reused "eye" tone). No mouth/nose distinction. The ≤8-color palette is shared across all 7 AI-drawn regions (head ×6 + body_front), leaving too little color budget for facial detail.
2. **Body/limbs read as flat outlined rectangles.** `flat_fill_with_border` fills every procedural face with exactly 2 tones (main + 1px shadow border). At Minecraft's small face sizes (4–12px) this reads as a plain box, not clothing — and there's no way to add a distinctive design element (logo, pattern) to a procedural face at all.

Separately, the user wants to tweak colors on a generated skin conversationally ("make the shirt blue") without re-uploading the photo or burning a full Vision call.

## Root cause behind both

Top-level color fields (`skin_tone`, `shirt_main`, …) and the `palette` used for indexed pixel grids are currently **two disconnected sources of truth**. The top-level fields feed `procedural.py`; the palette feeds the AI-drawn regions. Editing a top-level field never touches already-decoded pixels, and there's no way to say "recolor the shirt" and have it apply everywhere the shirt appears.

## Design

### 1. Palette with fixed semantic roles

Collapse the two sources into one. The model's `palette` array gets 10 fixed-role slots (indices are a code constant, not model-chosen) followed by up to 6 freeform slots:

```
palette[0] skin_tone      palette[5] arm_main
palette[1] hair_color     palette[6] arm_shadow
palette[2] eye_color      palette[7] pants_main
palette[3] shirt_main     palette[8] pants_shadow
palette[4] shirt_shadow   palette[9] shoe_color
palette[10..15]           freeform — AI's choice, for logos/patterns/accessories
```

`MIN_PALETTE_SIZE = 10`, `MAX_PALETTE_SIZE = 16`. `PALETTE_ROLES: dict[str, int]` is a new constant (lives in `app/services/skin_map.py` or a new small module — implementer's call) mapping role name → fixed index.

Both AI-drawn pixel grids (indices into `palette`) and `procedural.generate_procedural_regions` (reading named roles) draw from this single array. Top-level `skin_tone` / `hair_color` / etc. fields are removed from the prompt schema — the model only outputs `palette` (with the first 10 slots meaning what the fixed roles say) plus `description` and `hair_style`.

This is also the mechanism that makes conversational editing cheap: changing `palette[3]` retints every pixel that referenced index 3, in both the AI-drawn regions and procedural fill, with no re-render logic beyond re-decoding.

`metadata.skin_tone` / `metadata.hair_color` in the API response are now derived (`palette[PALETTE_ROLES["skin_tone"]]`, etc.) rather than read from separate model output. `metadata.regions_generated` becomes `7 + len(detail faces the model actually drew)` instead of a fixed 7.

### 2. AI-chosen "detail faces" beyond the mandatory 7

Prompt allows the model to optionally emit indexed grids for **up to 4 extra faces** (any valid `PIXEL_KEY_MAP` key beyond the mandatory `head_front/back/top/bottom/left/right` + `body_front`), using palette slots 10–15 for pattern-specific colors. Prompt language: only add these if there's a genuinely distinctive design element (back logo, sleeve pattern, belt, etc.) that would look wrong as a flat fill — most characters won't need any.

Validation accepts whatever valid extra face keys are present (no hard cap enforced in code — the cap is prompt guidance to bound the typical-case token cost; a model that ignores it just costs more, it doesn't break anything). `procedural.generate_procedural_regions` gets a new `exclude_keys: set[str]` param so it skips any face the AI already drew.

### 3. Procedural shading: `shade_face` replaces `flat_fill_with_border`

Pure code change, zero added tokens. From one `main` hex, derive `highlight` / `shadow` / `deep_shadow` via HSL lightness adjustment (`colorsys`). Apply a per-face-orientation lightness bias so faces read as a 3D form instead of identical flat tiles:

```
top: +0.12   front: 0.0   right: 0.0   left: -0.08   back: -0.15   bottom: -0.22
```

Interior = main adjusted by the face's bias. Outermost 1px ring = interior color adjusted one more step darker (seam definition). `right_leg_bottom` / `left_leg_bottom` keep the existing special case: flat `shoe_color`, no shading (reads as a sole).

### 4. Prompt changes

- Palette instructions rewritten to describe the 10 fixed roles by position and the freeform tail.
- Explicit rule: eye and mouth pixels in `head_front` must use a palette index distinct from both `skin_tone` (0) and `hair_color` (1) — reuse an existing distinct role (e.g. `eye_color`, index 2) rather than inventing new colors for these, to keep the palette small.
- `max_tokens` likely needs to increase from 3000 to ~4000 to cover the larger palette + optional detail faces headroom; confirm against real usage during implementation (current usage sits around 2,700/3,000).

### 5. Persistence for editing

New file per skin: `skins/<id>.json`, written alongside `skins/<id>.png` at generation time:

```json
{
  "model": "classic",
  "ai_model": "flash",
  "palette": ["#...", "...16 entries..."],
  "pixel_grids": {
    "head_front": [[0,1,2,...]],
    "...": "... all AI-drawn regions, mandatory + any detail faces ..."
  },
  "description": "...",
  "hair_style": "short"
}
```

### 6. Edit endpoint

`POST /api/skin/{skin_id}/edit`, JSON body `{"instruction": "把衬衫改成蓝色"}`.

Flow:
1. Load `skins/{id}.json`; 404 if missing.
2. `interpret_color_edit(current_roles: dict[str, str], instruction: str, ai_model: AIModel) -> dict[str, str]` (new function in `claude_vision.py`) — text-only Gemini call, no image. Prompt: the 10 named roles + their current hex values + the user's instruction; ask for a JSON object containing only the roles that should change and their new hex value. Uses the skin's stored `ai_model`.
3. Validate returned hex values; for each, write into the correct fixed `palette` index.
4. Re-decode all stored `pixel_grids` with the updated palette.
5. Re-run `generate_procedural_regions` (same `exclude_keys` as original generation).
6. Reassemble and **overwrite** `skins/{id}.png` in place (editing, not forking a new skin).
7. Save the updated `skins/{id}.json`.
8. Response: same shape as `/api/generate` (`skin_id, skin_url, model, metadata`); `skin_id`/`skin_url` unchanged since it's an in-place edit.

Out of scope for this pass: edits that touch geometry/pixel content (add glasses, change hairstyle) — the user explicitly scoped this to named-role color changes only. Unrecognized instructions (edit maps to no known role) return the skin unchanged with a message in the response saying nothing matched — implementer's call on exact wording.

### 7. Frontend

Viewer section gets a text input + "应用修改" button, visible once a skin exists. On submit: `POST` to the edit endpoint, then reload the skinview3d texture with a cache-busted URL (`skin_url + '?t=' + Date.now()`) since the PNG bytes changed at the same path.

## Testing

- `procedural.py`: tests for `shade_face` (orientation bias directions, edge-vs-interior tone), `exclude_keys` skipping AI-drawn detail faces, existing shoe-bottom special case still passes.
- `claude_vision.py`: palette role extraction/validation, `interpret_color_edit` mapping (mocked model response → correct palette index writes), persistence read/write round-trip.
- `routers/skin.py`: new edit endpoint — 404 on unknown skin_id, successful edit updates the PNG and JSON, malformed instruction handled gracefully.
- Update existing tests wherever the old `flat_fill_with_border` / top-level-colors-dict signatures are referenced.

## Sequencing

Two phases, second depends on the first:

1. Palette-role restructure + detail faces + `shade_face` (fixes the visual quality complaints, no new endpoints).
2. Persistence + edit endpoint + frontend (adds conversational editing, built on the fixed-role palette from phase 1).
