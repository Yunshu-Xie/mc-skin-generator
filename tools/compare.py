#!/usr/bin/env python3
"""Render one photo through every downscaling strategy and print the scores.

    python3 tools/compare.py photo.jpg --ai
    python3 tools/compare.py photo.jpg --face 0.38,0.05,0.66,0.27 --eyes 0.43,0.16,0.65,0.19
    python3 tools/compare.py --demo

Writes a contact sheet: the source, the 8×8 head rendered by each method
(blown up so you can actually see it), and the assembled skin. Look at the
sheet *and* the table — at this scale the numbers alone are misleading, for
the reason spelled out in app/imaging/metrics.py.

``--ai`` makes the real vision call. Without it the generic default framing is
used, which is fine for comparing methods against each other but will crop
badly on any particular photo; ``--face``/``--eyes`` let you supply the two
boxes that matter by hand and tune the renderer without spending API calls.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.imaging.color import hex_to_linear, linear_to_u8  # noqa: E402
from app.services.layout import Bbox, analyze_photo, default_layout  # noqa: E402
from app.services.renderer import RenderConfig, render  # noqa: E402
from app.services.skin_assembler import assemble_skin  # noqa: E402

METHODS = ("box", "dpid", "dominant")
ZOOM = 32


def _demo_photo() -> bytes:
    """The synthetic portrait from the test fixtures — every feature at a
    known coordinate, so the sheet is reproducible and ships no one's art."""
    img = np.full((512, 384, 3), 230, dtype=np.uint8)
    img[220:400, 60:324] = (60, 90, 170)
    img[220:400, 150:234] = (200, 40, 40)
    img[30:220, 120:264] = (215, 175, 140)
    img[30:80, 120:264] = (45, 33, 24)
    img[110:126, 148:172] = (40, 30, 25)
    img[110:126, 212:236] = (40, 30, 25)
    img[160:172, 176:208] = (170, 90, 90)
    img[400:512, 140:244] = (25, 25, 60)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


def _box(value: str) -> Bbox:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    return Bbox(*parts)


def _grid_image(grid: list[list[str]], zoom: int = ZOOM) -> Image.Image:
    arr = np.stack([np.stack([hex_to_linear(c) for c in row]) for row in grid]).astype(np.float32)
    img = Image.fromarray(linear_to_u8(arr))
    return img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)


def _contact_sheet(tiles: list[Image.Image], path: Path) -> None:
    width = sum(t.width + 8 for t in tiles) + 8
    height = max(t.height for t in tiles) + 16
    sheet = Image.new("RGB", (width, height), (24, 26, 25))
    x = 8
    for tile in tiles:
        sheet.paste(tile, (x, 8))
        x += tile.width + 8
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path, nargs="?")
    parser.add_argument("--demo", action="store_true", help="use the synthetic portrait")
    parser.add_argument("--ai", action="store_true", help="call the vision API")
    parser.add_argument("--face", type=_box, help="x0,y0,x1,y1 of the head")
    parser.add_argument("--eyes", type=_box, help="x0,y0,x1,y1 of the eye band")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2])
    parser.add_argument("--out", type=Path, default=Path("comparison.png"))
    args = parser.parse_args()

    if not args.photo and not args.demo:
        parser.error("pass a photo path or --demo")
    image_bytes = _demo_photo() if args.demo else args.photo.read_bytes()

    if args.ai:
        layout = asyncio.run(analyze_photo(image_bytes, media_type="image/jpeg"))
    else:
        layout = default_layout()
    if args.face:
        layout.boxes["face"] = args.face
    if args.eyes:
        layout.boxes["eyes"] = args.eyes

    print(f"layout: {layout.source}  {layout.description}")
    print(f"{'face method':<12}{'ssim':>8}{'ΔE mean':>10}{'ΔE p95':>9}{'detail':>9}")

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    source.thumbnail((8 * ZOOM, 8 * ZOOM))
    tiles = [source]

    for method in METHODS:
        result = render(
            image_bytes,
            layout,
            "classic",
            RenderConfig(face_method=method, scale=args.scale, head_mode="photo"),
        )
        scores = result.metrics["head_front"]
        print(
            f"{method:<12}{scores['ssim']:>8.3f}{scores['delta_e_mean']:>10.3f}"
            f"{scores['delta_e_p95']:>9.3f}{scores['detail']:>9.3f}"
        )
        tiles.append(_grid_image(result.pixel_data["head_front"], ZOOM // args.scale))
        if method == "dpid":
            skin = assemble_skin(result.pixel_data, "classic", result.scale)
            tiles.append(skin.convert("RGB").resize((256, 256), Image.NEAREST))

    _contact_sheet(tiles, args.out)
    print(f"\nsource | box | dpid | full skin | dominant\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
