"""Assemble a 64×64 Minecraft skin PNG from pixel color data."""

from __future__ import annotations

from PIL import Image

from app.services.skin_map import PIXEL_KEY_MAP, FaceRect, ModelType, get_all_regions


def hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """Convert '#RRGGBB' or '#RRGGBBAA' to (R, G, B, A) tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    if len(h) == 8:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
    raise ValueError(f"Invalid hex color: {hex_color}")


def paint_face(img: Image.Image, rect: FaceRect, pixels: list[list[str]]) -> None:
    """Paint a 2D grid of hex colors onto the image at the given rectangle."""
    for row_idx, row in enumerate(pixels):
        for col_idx, hex_color in enumerate(row):
            if row_idx >= rect.h or col_idx >= rect.w:
                continue
            rgba = hex_to_rgba(hex_color)
            img.putpixel((rect.x + col_idx, rect.y + row_idx), rgba)


def validate_pixel_grid(grid: list[list[str]], expected_h: int, expected_w: int) -> bool:
    """Check that a pixel grid has the correct dimensions."""
    if len(grid) != expected_h:
        return False
    return all(len(row) == expected_w for row in grid)


def assemble_skin(
    pixel_data: dict[str, list[list[str]]],
    model: ModelType = "classic",
) -> Image.Image:
    """Assemble a 64×64 Minecraft skin from Claude's pixel data.

    Args:
        pixel_data: Dict mapping keys like "head_front", "body_back", etc.
            to 2D lists of hex color strings (row-major).
        model: "classic" (4px arms) or "slim" (3px arms).

    Returns:
        A 64×64 RGBA PIL Image.
    """
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    regions = get_all_regions(model)

    for key, pixel_grid in pixel_data.items():
        if key not in PIXEL_KEY_MAP:
            continue
        group, face = PIXEL_KEY_MAP[key]
        if group not in regions or face not in regions[group]:
            continue
        rect = regions[group][face]

        if not validate_pixel_grid(pixel_grid, rect.h, rect.w):
            # Skip malformed grids rather than crash
            continue

        paint_face(img, rect, pixel_grid)

    return img
