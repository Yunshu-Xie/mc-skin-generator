"""Clothes are drawn too.

The Trump test made the case: given the same seven colors, the pipeline's
photo-sampled torso rendered as a navy rectangle, while the same colors laid
over a garment template read instantly as a suit with a red tie. Structure,
not color, is what was missing — the identical conclusion the head reached in
:mod:`app.services.face`, arrived at again one body part later.

So the torso, arms and legs are drawn from garment templates: a coarse class
from the vision layer picks the shape, the palette supplies the colors, and
the photo is allowed back in only to restore lightness variation within each
region (:mod:`app.services.templating`).

The templates are built programmatically rather than written out as literal
grids, because arm width differs between the classic and slim models and a
literal grid would have to exist twice.

What this costs: a logo or a printed pattern on a chest no longer survives,
because the torso is no longer a photograph of that chest. ``body_mode="photo"``
keeps the old path for exactly that case.
"""

from __future__ import annotations

import numpy as np

from app.services.shading import shift_lightness
from app.services.templating import modulate_by_photo, paint

__all__ = [
    "TOP_STYLES",
    "BOTTOM_STYLES",
    "solid",
    "build_torso",
    "build_arm",
    "build_leg",
    "render_clothing",
]

# Region codes.
TOP, INNER, ACCENT, SKIN, PANTS, SHOE = "T", "I", "A", "S", "P", "H"

TOP_STYLES = ("plain", "tshirt", "shirt", "jacket", "suit", "hoodie", "dress")
BOTTOM_STYLES = ("pants", "shorts", "skirt")
DEFAULT_TOP, DEFAULT_BOTTOM = "tshirt", "pants"

# Tops that close over an inner layer, so the torso shows a strip of it.
LAYERED = {"jacket", "suit", "hoodie"}
# Tops with sleeves down to the wrist.
LONG_SLEEVED = {"shirt", "jacket", "suit", "hoodie"}

SHOE_ROWS = 2  # how far up the leg the shoe reaches, on every side


def solid(height: int, width: int, code: str) -> list[list[str]]:
    """A single-region template — every face that carries no garment detail."""
    return [[code] * width for _ in range(height)]


def build_torso(style: str, height: int, width: int) -> list[list[str]]:
    """Front of the torso: the only face where a garment has real structure."""
    grid = solid(height, width, TOP)
    if width < 4 or height < 4:
        return grid

    mid = width // 2
    if style in LAYERED:
        # An open jacket: a column of the inner layer down the middle, closing
        # about two thirds of the way down where the garment buttons.
        close = max(2, int(height * 0.45))
        for row in range(height):
            for col in range(mid - 2, mid + 2):
                if 0 <= col < width and row < close:
                    grid[row][col] = INNER
        if style == "suit":
            for row in range(1, close + 2):
                for col in (mid - 1, mid):
                    if 0 <= col < width:
                        grid[row][col] = ACCENT
        if style == "hoodie":  # a pouch pocket low on the front
            for row in range(int(height * 0.6), int(height * 0.8)):
                for col in range(1, width - 1):
                    grid[row][col] = ACCENT
    elif style == "tshirt":
        for col in range(mid - 1, mid + 1):  # a small neckline
            if 0 <= col < width:
                grid[0][col] = SKIN
    return grid


def build_arm(style: str, height: int, width: int, bare: bool = False) -> list[list[str]]:
    """Arms: sleeve down to some row, skin below it, cuff where it ends."""
    if bare or style == "dress":
        return solid(height, width, SKIN)

    sleeve = height - 2 if style in LONG_SLEEVED else int(height * 0.45)
    grid = solid(height, width, SKIN)
    for row in range(min(sleeve, height)):
        grid[row] = [TOP] * width
    if style in LONG_SLEEVED and sleeve < height:
        grid[sleeve] = [INNER] * width  # a cuff
    return grid


def build_leg(style: str, height: int, width: int) -> list[list[str]]:
    """Legs: trouser down to some row, skin below, shoe at the bottom."""
    grid = solid(height, width, SKIN)
    if style == "shorts":
        covered = int(height * 0.45)
    elif style == "skirt":
        covered = int(height * 0.3)
    else:
        covered = height - SHOE_ROWS

    for row in range(min(covered, height)):
        grid[row] = [PANTS] * width
    # Shoes on every side, not just the sole — the old code painted only
    # leg_bottom, which is the one face a player never sees.
    for row in range(max(0, height - SHOE_ROWS), height):
        grid[row] = [SHOE] * width
    return grid


def render_clothing(
    template: list[list[str]],
    colors: dict[str, np.ndarray],
    photo: np.ndarray | None = None,
    modulation: float = 0.6,
) -> np.ndarray:
    """Paint one garment template and fold the photo's shading back in."""
    return modulate_by_photo(paint(template, colors), template, photo, modulation, exempt=(ACCENT,))


def colors_from_palette(
    shirt: np.ndarray,
    inner: np.ndarray,
    accent: np.ndarray,
    skin: np.ndarray,
    pants: np.ndarray,
    shoe: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        TOP: shirt,
        INNER: inner,
        ACCENT: accent,
        SKIN: skin,
        PANTS: pants,
        SHOE: shoe,
    }


def back_of(template: list[list[str]]) -> list[list[str]]:
    """The back of a garment: the same vertical structure, no front detail.

    Accent and inner-layer cells are front-only — a tie has no back — but the
    sleeve/trouser/shoe boundaries must line up exactly or the seam shows.
    """
    return [[TOP if code in (INNER, ACCENT) else code for code in row] for row in template]


def shade_cap(color: np.ndarray, up: bool) -> np.ndarray:
    """A shoulder or sole cap, nudged so the two are distinguishable."""
    return shift_lightness(color, 0.03 if up else -0.03).reshape(3)
