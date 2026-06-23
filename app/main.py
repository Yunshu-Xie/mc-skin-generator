from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import skin

app = FastAPI(title="MC Skin Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure skins directory exists
Path(settings.skins_dir).mkdir(exist_ok=True)

# Mount routers
app.include_router(skin.router, prefix="/api")

# Static files at /static/
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the frontend page."""
    html_path = static_dir / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
