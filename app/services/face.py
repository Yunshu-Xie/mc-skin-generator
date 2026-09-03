"""The head is drawn, not sampled.

Everywhere else in this project, pixels are derived from the photograph. The
head front is the one place that cannot work, and the reason is not a tuning
failure — it is arithmetic. A Minecraft head front is 8x8. A face crop reduced
to it is a ~16x reduction, at which **every facial feature is smaller than one
output cell**. Weighted averaging, however cleverly weighted, can only pull a
cell's color *toward* a feature; it can never render the feature. That is the
ceiling `stamp_eyes` hit: the best it could do was darken one cell.

Measured on a real run (see docs/ARCHITECTURE.md section 5b), the resampled
head and a hand-authored glyph carried almost the same hair-to-skin lightness
separation — 0.082 vs 0.034 in OKLab L, the glyph actually *lower*. The glyph
still read instantly as a face. Legibility at this size comes from two things
the resampler cannot produce:

1. **contiguous regions with hard boundaries** — hair is one block, skin is one
   block, instead of 64 cells each deciding independently and producing noise;
2. **one local extreme in the right place** — eyes pushed far darker than
   anything in the photograph actually is.

So structure comes from a template here, and color comes from the photo. The
templates are deliberately few and blunt; what makes a generated head look
like its subject at this size is the palette and the eye placement, not the
number of hairstyles on offer.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from app.imaging.color import linear_to_oklab, oklab_to_linear
from app.services.shading import shift_lightness

__all__ = [
    "HairStyle",
    "STYLES",
    "HAIR_ROWS",
    "build_template",
    "render_face",
    "eye_row_from_box",
]

HairStyle = Literal["short", "long", "fringe", "bald", "hat"]

# Region codes used inside a template grid.
HAIR, SKIN, EYE, SCLERA, MOUTH, BROW = "H", "S", "E", "W", "M", "B"

# style -> (rows of hair across the top, whether hair runs down both sides)
STYLES: dict[str, tuple[int, bool]] = {
    "short": (2, False),
    "fringe": (3, False),
    "long": (2, True),
    "hat": (3, True),
    "helmet": (3, True),
    "bald": (0, False),
}
DEFAULT_STYLE = "short"


def HAIR_ROWS(style: str) -> int:
    """How many full rows of hair this style puts across the top of the head."""
    return STYLES.get(style, STYLES[DEFAULT_STYLE])[0]


# How far below the reported eye color the eye cells are pushed, in OKLab L.
# Large on purpose: this single local extreme is most of what makes the glyph
# readable, and photographic eye colors are never this dark.
EYE_DARKEN = 0.40
SCLERA_LIGHTEN = 0.07
MOUTH_DARKEN = 0.13
MOUTH_WARMTH = 0.02  # nudge along OKLab's a axis, toward red
# Brows are hair, but almost always read darker than the hair on the head.
BROW_DARKEN = 0.10


def build_template(style: str, eye_row: int = 3, brows: bool = True) -> list[list[str]]:
    """An 8x8 grid of region codes for the given hairstyle.

    ``brows`` costs two cells and buys a surprising amount of identity — a face
    without them reads as blank in a way that is hard to place until you add
    them back. They are skipped when the eye row sits directly under the hair,
    where there is no room for them.
    """
    hair_rows, side_hair = STYLES.get(style, STYLES[DEFAULT_STYLE])
    grid = [[SKIN] * 8 for _ in range(8)]

    for row in range(hair_rows):
        grid[row] = [HAIR] * 8
    if side_hair:
        for row in range(hair_rows, 8):
            grid[row][0] = grid[row][7] = HAIR

    row = max(hair_rows, min(5, eye_row))
    grid[row][1] = SCLERA
    grid[row][2] = EYE
    grid[row][5] = EYE
    grid[row][6] = SCLERA

    if brows and row - 1 >= hair_rows:
        grid[row - 1][2] = BROW
        grid[row - 1][5] = BROW

    mouth = min(7, row + 2)
    grid[mouth][3] = MOUTH
    grid[mouth][4] = MOUTH
    return grid


def eye_row_from_box(face_y0: float, face_y1: float, eyes_y0: float, eyes_y1: float) -> int:
    """Which of the 8 rows the reported eye band falls in."""
    span = face_y1 - face_y0
    if span <= 0:
        return 3
    center = ((eyes_y0 + eyes_y1) / 2.0 - face_y0) / span
    return int(max(0, min(7, round(center * 8 - 0.5))))


def _warm(color: np.ndarray, amount: float) -> np.ndarray:
    lab = linear_to_oklab(np.asarray(color, dtype=np.float32).reshape(1, 3))
    lab[0, 1] += amount
    return oklab_to_linear(lab).reshape(3)


def render_face(
    style: str,
    eye_row: int,
    skin: np.ndarray,
    hair: np.ndarray,
    eye: np.ndarray,
    photo: np.ndarray | None = None,
    modulation: float = 0.6,
    brows: bool = True,
) -> np.ndarray:
    """Draw an 8x8 head front: structure from the template, color from the photo.

    ``photo`` is the downscaled 8x8 crop of the actual head. It is not used for
    structure — only to put back the lightness variation within each region, so
    a cheek in shadow stays darker than one in light and the result does not
    look like flat vector art. ``modulation`` is how much of that variation
    survives; 0 gives the pure template.

    Eye and mouth cells are exempt from modulation on purpose. Their whole job
    is to be the extreme values in the grid, and letting the photo lighten them
    is exactly the averaging this module exists to avoid.
    """
    template = build_template(style, eye_row, brows)
    palette = {
        HAIR: np.asarray(hair, dtype=np.float32).reshape(3),
        SKIN: np.asarray(skin, dtype=np.float32).reshape(3),
        EYE: shift_lightness(eye, -EYE_DARKEN).reshape(3),
        SCLERA: shift_lightness(skin, SCLERA_LIGHTEN).reshape(3),
        MOUTH: _warm(shift_lightness(skin, -MOUTH_DARKEN), MOUTH_WARMTH),
        BROW: shift_lightness(hair, -BROW_DARKEN).reshape(3),
    }

    out = np.stack([np.stack([palette[code] for code in row]) for row in template]).astype(
        np.float32
    )

    if photo is None or modulation <= 0 or photo.shape[:2] != (8, 8):
        return out

    lightness = linear_to_oklab(photo)[:, :, 0]
    for code in (HAIR, SKIN, SCLERA):
        cells = [(r, c) for r in range(8) for c in range(8) if template[r][c] == code]
        if not cells:
            continue
        mean = float(np.mean([lightness[r, c] for r, c in cells]))
        for r, c in cells:
            delta = (lightness[r, c] - mean) * modulation
            out[r, c] = shift_lightness(out[r, c], float(delta)).reshape(3)
    return out
