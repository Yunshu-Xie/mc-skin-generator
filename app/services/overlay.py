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
    size: int = 8,
) -> dict[str, list[list[str]]]:
    """Hair silhouette (and optionally glasses) on the head's overlay layer.

    Returns the ``hat_*`` grids for :func:`app.services.skin_assembler.assemble_skin`,
    with :data:`TRANSPARENT` everywhere the layer should not be drawn.
    """
    unit = max(1, size // 8)
    hair_rows = HAIR_ROWS(style) * unit
    # The overhang must stop above the eyes. A "fringe" style already puts
    # three rows of hair on the base layer and its eye row sits directly under
    # them, so an unclamped extra row draws hair straight over both eyes.
    rows = min(size, hair_rows + OVERHANG.get(style, 1) * unit, max(0, eye_row))
    sides = style in SIDE_STYLES

    out = {
        f"hat_{face}": _blank(size, size)
        for face in ("front", "back", "left", "right", "top", "bottom")
    }

    if hair_rows == 0 and glasses is None:
        return out  # bald and unbespectacled: nothing to add

    if hair_rows > 0:
        for face in ("front", "back", "left", "right"):
            grid = out[f"hat_{face}"]
            for row in range(rows):
                grid[row] = [hair] * size
            if sides:
                for row in range(rows, size):
                    for col in list(range(unit)) + list(range(size - unit, size)):
                        grid[row][col] = hair
        out["hat_top"] = [[hair] * size for _ in range(size)]

    if glasses is not None:
        row = max(0, min(size - unit, eye_row))
        front = out["hat_front"]
        # Frame only. The lens cells (columns 2 and 5 in eighths) are left
        # transparent so the eyes painted on the base layer show through them —
        # a solid band across the eye row reads as a blindfold, not spectacles.
        for eighth in (1, 3, 4, 6):
            for r in range(row, min(size, row + unit)):
                for c in range(eighth * unit, (eighth + 1) * unit):
                    front[r][c] = glasses
        for face in ("left", "right"):  # temples, running back over the ears
            for r in range(row, min(size, row + unit)):
                for c in range(3 * unit, 5 * unit):
                    out[f"hat_{face}"][r][c] = glasses

    return out
