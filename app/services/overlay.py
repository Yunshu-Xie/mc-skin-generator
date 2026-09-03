"""The second layer — half the texture, and it was empty.

A Minecraft model has two shells: the base cube and an outer "overlay" layer
rendered fractionally outside it, transparent wherever it is unused. That
doubles the usable pixel budget without changing the file format: on the test
skin, 1632 base cells were painted and 1632 overlay cells sat unwritten.

It is not extra spatial resolution — a face is still 8x8 either way. What it
buys is *silhouette and depth*, which at this size matters more: hair that
overhangs the forehead instead of sitting flush with it, glasses that float in
front of the eyes, a hat with a brim. Marketplace skins lean on this layer
heavily, and its absence is a large part of why a generated skin reads as
flat next to a hand-drawn one.

Everything here emits hex strings rather than linear arrays, because the
overlay's defining feature is transparency and the quantization path has no
alpha channel. Colors come from the already-built palette, so the overlay
cannot introduce a shade the base layer does not also use.
"""

from __future__ import annotations

from app.services.face import HAIR_ROWS

__all__ = ["TRANSPARENT", "build_head_overlay"]

TRANSPARENT = "#00000000"

# How far the hair overhangs the base layer's hairline, by style. One row is
# enough to read as a fringe; two starts to look like a helmet.
OVERHANG = {"short": 1, "fringe": 1, "long": 1, "hat": 1, "helmet": 1, "bald": 0}

# Styles whose hair also runs down the sides of the head on the overlay.
SIDE_STYLES = {"long", "hat", "helmet"}


def _blank(height: int, width: int) -> list[list[str]]:
    return [[TRANSPARENT] * width for _ in range(height)]


def build_head_overlay(
    style: str,
    hair: str,
    eye_row: int = 3,
    glasses: str | None = None,
) -> dict[str, list[list[str]]]:
    """Hair silhouette (and optionally glasses) on the head's overlay layer.

    Returns the ``hat_*`` grids for :func:`app.services.skin_assembler.assemble_skin`,
    with :data:`TRANSPARENT` everywhere the layer should not be drawn.
    """
    hair_rows = HAIR_ROWS(style)
    rows = min(8, hair_rows + OVERHANG.get(style, 1))
    sides = style in SIDE_STYLES

    out = {
        f"hat_{face}": _blank(8, 8) for face in ("front", "back", "left", "right", "top", "bottom")
    }

    if hair_rows == 0 and glasses is None:
        return out  # bald and unbespectacled: nothing to add

    if hair_rows > 0:
        for face in ("front", "back", "left", "right"):
            grid = out[f"hat_{face}"]
            for row in range(rows):
                grid[row] = [hair] * 8
            if sides:
                for row in range(rows, 8):
                    grid[row][0] = grid[row][7] = hair
        out["hat_top"] = [[hair] * 8 for _ in range(8)]

    if glasses is not None:
        row = max(0, min(7, eye_row))
        front = out["hat_front"]
        # Two lenses and a bridge; the temples continue onto the side faces.
        for col in (1, 2, 5, 6):
            front[row][col] = glasses
        front[row][3] = glasses
        front[row][4] = glasses
        for face in ("left", "right"):
            out[f"hat_{face}"][row][3] = glasses
            out[f"hat_{face}"][row][4] = glasses

    return out
