#!/usr/bin/env python3
"""Render a photo end-to-end and lay the result out as a front-facing figure.

    python3 tools/preview.py photo.jpg [--ai] [--face x0,y0,x1,y1] [--eyes ...]

Writes <out> containing: the source photo, the assembled 64x64 texture, and a
front view of the character built from the front faces — which is what the
skin actually looks like on a player, and the only honest way to judge it.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.imaging.color import hex_to_linear, linear_to_u8  # noqa: E402
from app.services.layout import Bbox, analyze_photo, default_layout  # noqa: E402
from app.services.renderer import RenderConfig, render  # noqa: E402
from app.services.skin_assembler import assemble_skin  # noqa: E402

# Where each front face sits on a 16x32 character, in character pixels.
FIGURE = {
    "head_front": (4, 0),
    "body_front": (4, 8),
    "right_arm_front": (0, 8),
    "left_arm_front": (12, 8),
    "right_leg_front": (4, 20),
    "left_leg_front": (8, 20),
}


def _grid(pixels: list[list[str]]) -> np.ndarray:
    return np.stack([np.stack([hex_to_linear(c) for c in row]) for row in pixels]).astype(
        np.float32
    )


def front_view(pixel_data: dict[str, list[list[str]]], scale: int = 12) -> Image.Image:
    canvas = np.zeros((32, 16, 3), dtype=np.float32)
    alpha = np.zeros((32, 16), dtype=bool)
    for key, (x, y) in FIGURE.items():
        grid = _grid(pixel_data[key])
        h, w = grid.shape[:2]
        if key == "left_arm_front":  # slim arms are 3px; keep them right-aligned
            x = 16 - w
        canvas[y : y + h, x : x + w] = grid
        alpha[y : y + h, x : x + w] = True

    rgb = linear_to_u8(canvas)
    rgb[~alpha] = (32, 34, 33)
    img = Image.fromarray(rgb)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def _box(value: str) -> Bbox:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    return Bbox(*parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    parser.add_argument("--ai", action="store_true")
    parser.add_argument("--face", type=_box)
    parser.add_argument("--eyes", type=_box)
    parser.add_argument("--model", default="classic", choices=["classic", "slim"])
    parser.add_argument("--out", type=Path, default=Path("preview.png"))
    args = parser.parse_args()

    image_bytes = args.photo.read_bytes()
    layout = (
        asyncio.run(analyze_photo(image_bytes, media_type="image/jpeg"))
        if args.ai
        else default_layout()
    )
    if args.face:
        layout.boxes["face"] = args.face
    if args.eyes:
        layout.boxes["eyes"] = args.eyes

    result = render(image_bytes, layout, args.model, RenderConfig())

    print(f"layout   : {layout.source}  {layout.description}")
    for name in sorted(layout.boxes):
        box = layout.boxes[name]
        print(
            f"  {name:<10} {box.x0:.3f},{box.y0:.3f},{box.x1:.3f},{box.y1:.3f}"
            f"   ({box.x1 - box.x0:.2f} x {box.y1 - box.y0:.2f})"
        )
    print(f"roles    : {layout.roles or '(defaults)'}")
    print(f"palette  : {' '.join(result.palette)}")
    for name, scores in result.metrics.items():
        print(
            f"{name:11}: ssim {scores['ssim']:.3f}  ΔE {scores['delta_e_mean']:.3f}"
            f"  detail {scores['detail']:.3f}"
        )

    figure = front_view(result.pixel_data)
    texture = assemble_skin(result.pixel_data, args.model)
    texture_big = texture.convert("RGB").resize((384, 384), Image.NEAREST)
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    source.thumbnail((384, 384))

    tiles = [source, figure, texture_big]
    width = sum(t.width + 10 for t in tiles) + 10
    height = max(t.height for t in tiles) + 20
    sheet = Image.new("RGB", (width, height), (24, 26, 25))
    x = 10
    for tile in tiles:
        sheet.paste(tile, (x, 10))
        x += tile.width + 10
    sheet.save(args.out)

    texture.save(args.out.with_name(args.out.stem + "_skin.png"))

    # Persist the layout: a bad render is almost always a bad box, and without
    # this the only way to find out is to spend another API call.
    layout_path = args.out.with_name(args.out.stem + "_layout.json")
    layout_path.write_text(
        json.dumps(
            {
                "source": layout.source,
                "description": layout.description,
                "boxes": {k: [v.x0, v.y0, v.x1, v.y1] for k, v in layout.boxes.items()},
                "roles": layout.roles,
                "palette": result.palette,
                "metrics": result.metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}, {args.out.with_name(args.out.stem + '_skin.png')}, {layout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
