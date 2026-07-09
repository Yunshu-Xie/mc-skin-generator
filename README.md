# mc-skin-generator

A local Minecraft skin generator: upload a photo or image, let Gemini AI design a character from it, and get a ready-to-use 64×64 Minecraft skin PNG with a live 3D preview and one-click download.

Built with FastAPI, vanilla HTML/JS (no build step), Pillow for image assembly, and an OpenAI-compatible client pointed at the Gemini API's OpenAI-compatible endpoint for the AI pipeline. Each request can pick between `gemini-2.5-flash` and `gemini-2.5-flash-lite`, both free-tier, to compare quality/cost.

## Features

- **Image → skin** — drop in any JPEG / PNG / WebP / GIF (≤ 5 MB) and get a valid Minecraft skin texture
- **One AI call, mostly procedural** — a single Vision call draws only the head (6 faces) and body_front pixel-by-pixel (the parts that actually need to look like the photo) plus a handful of top-level clothing colors; everything else (rest of the body, both arms, both legs) is filled in by code
- **Classic & Slim** — pick Steve (4 px arms) or Alex (3 px arms); the only difference is arm UV widths, handled by the coordinate map
- **3D preview** — generated skins render in-browser with a walking animation via [skinview3d](https://github.com/bs-community/skinview3d)
- **Download** — grab the raw 64×64 PNG, drop it straight into Minecraft

## Why mostly-procedural instead of a full AI-painted skin?

Asking a model to emit all ~4096 skin pixels is expensive and unreliable — earlier versions used a two-step pipeline (analyze, then paint everything) that could burn 13k+ tokens per run and still hit output-length limits. Since the arms/legs/back of a Minecraft skin are usually flat clothing colors anyway, only the head and shirt front carry the character's actual likeness:

1. **One Vision call** — the model looks at the photo and returns a small palette (≤8 colors) + indexed pixel grids for head (6 faces) and body_front (480 pixels total), plus ~10 top-level hex colors for skin/hair/eyes/shirt/arms/pants/shoes.
2. **Procedural fill** (`app/services/procedural.py`) — every other face (body back/top/bottom/sides, both arms, both legs) is a flat color fill with a 1px darker border, built directly from those top-level colors. No AI call needed for ~29 of the 36 base regions.

If the AI response is incomplete (bad palette index, missing grid, missing required color), it retries the single call up to 2 times.

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
│   ├── claude_vision.py        # Gemini (OpenAI-compatible) client → single call, head+body_front pixels + colors
│   ├── procedural.py           # flat-fill + border shading for the rest of the body/arms/legs
│   ├── skin_map.py             # 64×64 UV coordinates (FaceRect) for Classic/Slim
│   └── skin_assembler.py       # pixel data → 64×64 RGBA PNG via Pillow
└── static/                     # vanilla HTML/CSS/JS, no build step
```

### Generation flow

1. Front-end `POST /api/generate` with the image (multipart) + model type (`classic` / `slim`)
2. `claude_vision.generate_skin_data` resizes the photo, makes the single Vision call, decodes the indexed head/body_front grids, and merges in `procedural.generate_procedural_regions` for everything else
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
- **Mostly procedural** — only head + body_front are AI-painted pixel-by-pixel; the rest is a deterministic flat-fill from top-level colors (`app/services/procedural.py`), which is most of the token savings
- **Retry** — an incomplete AI response (bad palette index, missing grid, missing required color) triggers up to 2 retries of the single call
- **Classic vs Slim** — the sole difference is arm front/back width (4 → 3 px), resolved purely in `skin_map.py`

## License

MIT
