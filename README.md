# mc-skin-generator

A local Minecraft skin generator: upload a photo or image, let CodeBuddy (Tencent) AI design a character from it, and get a ready-to-use 64×64 Minecraft skin PNG with a live 3D preview and one-click download.

Built with FastAPI, vanilla HTML/JS (no build step), Pillow for image assembly, and an OpenAI-compatible client pointed at Tencent CodeBuddy for the AI pipeline.

## Features

- **Image → skin** — drop in any JPEG / PNG / WebP / GIF (≤ 5 MB) and get a valid Minecraft skin texture
- **Two-step AI pipeline** — the model first *analyzes* the image (per-face color palettes + descriptions), then *paints* the 2D pixel grids; splitting "understand" from "draw" makes the ~4096-pixel output reliable
- **Classic & Slim** — pick Steve (4 px arms) or Alex (3 px arms); the only difference is arm UV widths, handled by the coordinate map
- **3D preview** — generated skins render in-browser with a walking animation via [skinview3d](https://github.com/bs-community/skinview3d)
- **Download** — grab the raw 64×64 PNG, drop it straight into Minecraft

## Why a two-step AI pipeline?

Asking a model to emit all ~4096 skin pixels in one shot is unreliable. So the pipeline separates concerns:

1. **Image analysis** — the vision model looks at the photo and outputs structured JSON: a color palette and short description for every body-part face.
2. **Pixel generation** — a second (text-only) call turns that analysis into 2D hex-color arrays for each face.

If the returned pixel data is incomplete (fewer than the 36 base regions), it auto-retries up to 2 times and merges results.

## Quick start

```bash
git clone https://github.com/Yunshu-Xie/mc-skin-generator.git
cd mc-skin-generator

python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

cp .env.example .env
# Edit .env and paste your CodeBuddy API key (sk-sp-xxxx)

.venv/bin/uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Configuration

`.env` reads these variables (see `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `CODEBUDDY_API_KEY` | _(required)_ | CodeBuddy Coding Plan API key (`sk-sp-xxxx`) |
| `CODEBUDDY_BASE_URL` | `https://api.lkeap.cloud.tencent.com/coding/v3` | API endpoint |
| `CODEBUDDY_VISION_MODEL` | `hunyuan-2.0-instruct` | Model used for image analysis |
| `CODEBUDDY_TEXT_MODEL` | `hunyuan-2.0-instruct` | Model used for pixel generation |

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
│   ├── claude_vision.py        # OpenAI-compatible client → analysis + pixel generation
│   ├── skin_map.py             # 64×64 UV coordinates (FaceRect) for Classic/Slim
│   └── skin_assembler.py       # pixel data → 64×64 RGBA PNG via Pillow
└── static/                     # vanilla HTML/CSS/JS, no build step
```

### Generation flow

1. Front-end `POST /api/generate` with the image (multipart) + model type (`classic` / `slim`)
2. `claude_vision.generate_skin_data` runs the two-step pipeline and returns per-face pixel grids + metadata
3. `skin_assembler.assemble_skin` draws each face at its correct UV rectangle into a 64×64 RGBA image
4. The PNG is saved to `skins/<id>.png` and the front-end renders it with skinview3d

### UV map (`skin_map.py`)

Every Minecraft skin face is a `FaceRect(x, y, w, h)` on the 64×64 texture, covering both base and overlay (hat / jacket / sleeves / pants) layers. `PIXEL_KEY_MAP` maps model output keys (e.g. `"head_front"`, `"hat_top"`) to a region group + face name, and `get_all_regions(model)` returns the right coordinate set for Classic vs Slim.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/api/generate` | multipart: `image`, `model` (`classic`/`slim`), `style_notes` | `{ skin_id, skin_url, model, metadata }` |
| `GET` | `/api/skin/{id}.png` | — | the generated PNG (404 if missing) |

`skin_id` is validated as alphanumeric to prevent path traversal.

## Development

```bash
pytest          # 22 tests, all mocked — does not call CodeBuddy
ruff check .
ruff format .
mypy app
```

Tests cover the UV map, the skin assembler, and the API layer with the AI pipeline mocked, so no network calls are made.

## Key design decisions

- **OpenAI-compatible SDK** — uses the `openai` package with `base_url` pointed at CodeBuddy, so swapping in another compatible provider is trivial
- **No database** — generated skins are plain PNG files named by a short UUID in `skins/`
- **Retry + merge** — incomplete pixel data triggers up to 2 retries with merged results
- **Classic vs Slim** — the sole difference is arm front/back width (4 → 3 px), resolved purely in `skin_map.py`

## License

MIT
