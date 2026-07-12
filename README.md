# mc-skin-generator

A local Minecraft skin generator: upload a photo or image, let Gemini AI design a character from it, and get a ready-to-use 64×64 Minecraft skin PNG with a live 3D preview and one-click download.

Built with FastAPI, vanilla HTML/JS (no build step), Pillow for image assembly, and an OpenAI-compatible client pointed at the Gemini API's OpenAI-compatible endpoint for the AI pipeline. Each request can pick between `gemini-2.5-flash` and `gemini-2.5-flash-lite`, both free-tier, to compare quality/cost.

## Features

- **Image → skin** — drop in any JPEG / PNG / WebP / GIF (≤ 5 MB) and get a valid Minecraft skin texture
- **Photo-derived pixels, not AI-guessed ones** — a single Vision call only locates 6 rough body-part regions (head, torso, right/left arm, right/left leg) as bounding boxes and classifies hair style + eye shape; the actual pixels for those regions are produced deterministically from the real uploaded photo (crop → downsample → color-quantize), so fine clothing detail and skin tone come straight from the source image instead of being reconstructed by the model
- **Classic & Slim** — pick Steve (4 px arms) or Alex (3 px arms); the only difference is arm UV widths, handled by the coordinate map
- **3D preview** — generated skins render in-browser with a walking animation via [skinview3d](https://github.com/bs-community/skinview3d)
- **Download** — grab the raw 64×64 PNG, drop it straight into Minecraft

## Why photo-derived pixels instead of AI-painted ones?

Earlier versions asked the Vision model to author pixels directly — either all ~4096 skin pixels in one shot, or (a later iteration) a smaller set of "mandatory" faces via per-pixel palette indices. Both were expensive and, worse, unreliable: manual testing showed the model losing fine clothing detail and drawing odd-looking faces even when the token budget was fine. Models are good at *locating* things in a photo; they're much worse at *reproducing* precise per-pixel color detail from one.

So the pipeline now splits those two jobs:

1. **One Vision call** (`app/services/claude_vision.py`, see `_build_prompt()`) — the model looks at the photo and returns only bounding boxes for 6 regions (head, torso, right/left arm, right/left leg, each optionally "not visible") plus two categorical fields, `hair_style` and `eye_shape`. No colors, no pixels.
2. **Deterministic photo rendering** (`app/services/photo_render.py`) — for torso and the four limbs, each region is cropped from the actual photo, downsampled to the target face's pixel grid one dominant color per cell, then all five regions are quantized together into one shared palette (so e.g. skin tone stays consistent between an arm and a leg). For the head, a fixed row-template (hair / eyes / nose / mouth / neck bands) is filled in with colors measured directly from the corresponding photo region — this produces `head_front` plus 5 "torso+limb front" faces, 6 faces total.
3. **Procedural fill** (`app/services/procedural.py`) — everything else (the other 5 head faces, and each limb/torso's back/left/right/top/bottom) is derived without AI: back/left/right copy their part's front face per-row color via `propagate_front_row` (so a clothing boundary visible on the front — e.g. a sleeve ending partway down the arm — stays consistent all the way around the limb), while the small top/bottom end-caps and the head's non-front faces are a flat fill from measured/derived colors. No synthetic shading is added anywhere — Minecraft's own in-game lighting already shades the 3D model.

If the Vision response is structurally incomplete (missing/invalid `hair_style`, `eye_shape`, or a malformed `regions` object), the whole call retries up to 2 times; a single region's bad bounding box just degrades that one region to "not visible" without triggering a retry.

## Quick start

```bash
git clone https://github.com/Yunshu-Xie/mc-skin-generator.git
cd mc-skin-generator

python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

cp .env.example .env
# Edit .env and paste your Gemini API key (from https://aistudio.google.com/apikey)

.venv/bin/uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Configuration

`.env` reads these variables (see `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | Google AI Studio API key |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Gemini's OpenAI-compatible endpoint |
| `GEMINI_MODEL_FLASH` | `gemini-2.5-flash` | Model used when a request sends `ai_model=flash` |
| `GEMINI_MODEL_FLASH_LITE` | `gemini-2.5-flash-lite` | Model used when a request sends `ai_model=flash-lite` |
| `GEMINI_DEFAULT_MODEL` | `flash` | Which of the two above to use when a request omits `ai_model` |
| `MAX_IMAGE_DIMENSION` | `768` | Uploaded photo is downscaled to this many px (longest side) and re-encoded as JPEG before sending, to cut vision input tokens |

`app/config.py` reads `.env` via pydantic-settings. Generated skins are saved to `skins/` (max upload size 5 MB).

## Architecture

```
app/
├── config.py                   # pydantic-settings, reads .env
├── main.py                     # FastAPI + CORS + static mount + skins dir
├── models/schemas.py           # Pydantic request/response shapes
├── routers/
│   └── skin.py                 # POST /api/generate, GET /api/skin/{id}.png
├── services/
│   ├── claude_vision.py        # Gemini (OpenAI-compatible) client → single call, region bounding boxes + hair_style/eye_shape only
│   ├── photo_render.py         # crop/downsample/quantize the real photo into 6 front-face pixel grids (torso, 4 limbs, head)
│   ├── procedural.py           # front-row propagation (back/left/right copy their part's front) + flat-fill caps for the rest
│   ├── skin_store.py           # persists palette + raw pixel-index grids per skin, kept for a possible future edit feature
│   ├── skin_map.py             # 64×64 UV coordinates (FaceRect) for Classic/Slim
│   └── skin_assembler.py       # pixel data → 64×64 RGBA PNG via Pillow
└── static/                     # vanilla HTML/CSS/JS, no build step
```

### Generation flow

1. Front-end `POST /api/generate` with the image (multipart) + model type (`classic` / `slim`)
2. `claude_vision.generate_skin_data` resizes the photo, makes the single Vision call for region boxes + `hair_style`/`eye_shape`, calls `photo_render` to turn those regions into the 6 photo-derived front faces, and merges in `procedural.generate_procedural_regions` for everything else
3. `skin_assembler.assemble_skin` draws each face at its correct UV rectangle into a 64×64 RGBA image
4. The PNG is saved to `skins/<id>.png` and the front-end renders it with skinview3d

### UV map (`skin_map.py`)

Every Minecraft skin face is a `FaceRect(x, y, w, h)` on the 64×64 texture, covering both base and overlay (hat / jacket / sleeves / pants) layers. `PIXEL_KEY_MAP` maps model output keys (e.g. `"head_front"`, `"hat_top"`) to a region group + face name, and `get_all_regions(model)` returns the right coordinate set for Classic vs Slim.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/generate` | multipart: `image`, `model` (`classic`/`slim`), `style_notes`, `ai_model` (`flash`/`flash-lite`) | `{ skin_id, skin_url, model, metadata }` (`metadata.ai_model` echoes which model ran) |
| `GET` | `/api/skin/{id}.png` | — | the generated PNG (404 if missing) |

`skin_id` is validated as alphanumeric to prevent path traversal.

## Development

```bash
pytest          # all mocked — does not call Gemini
ruff check .
ruff format .
mypy app
```

Tests cover the UV map, the skin assembler, and the API layer with the AI pipeline mocked, so no network calls are made.

## Key design decisions

- **OpenAI-compatible SDK** — uses the `openai` package with `base_url` pointed at Gemini's OpenAI-compatible endpoint, so swapping in another compatible provider is trivial
- **Per-request model choice** — `ai_model=flash` vs `flash-lite` lets you A/B the two free-tier Gemini models without restarting the server; `GEMINI_DEFAULT_MODEL` picks the default when omitted
- **No database** — generated skins are plain PNG files named by a short UUID in `skins/`
- **AI locates, code renders** — the model only returns 6 region bounding boxes + 2 categorical fields; `app/services/photo_render.py` deterministically crops/downsamples/quantizes the real photo into the 6 photo-derived front faces (head + torso + 4 limbs), and `app/services/procedural.py` derives every other face from those via `propagate_front_row` and flat fills — no pixel is ever guessed by the model
- **Retry** — a structurally incomplete Vision response (missing/invalid `hair_style`, `eye_shape`, or malformed `regions`) triggers up to 2 retries of the single call; a single region's bad bounding box just degrades that region to "not visible"
- **Classic vs Slim** — the sole difference is arm front/back width (4 → 3 px), resolved purely in `skin_map.py`
- **Conversational color editing retired** — an earlier design let you retint a generated skin by describing the change in plain language, backed by a fixed 10-slot color palette. That palette was replaced by this photo-derived, unstructured palette, so the edit feature was deliberately retired rather than half-adapted; `skin_store.py`'s persistence is kept in case a future redesign re-adds editing

## License

MIT
