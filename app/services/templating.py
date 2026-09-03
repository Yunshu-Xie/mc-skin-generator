"""Shared machinery for the template-drawn parts of a skin.

Both the head (:mod:`app.services.face`) and the clothing
(:mod:`app.services.clothing`) work the same way: a grid of region codes gives
the structure, the palette gives each region its color, and the photograph is
allowed back in only to restore the lightness variation *within* a region. It
never moves a boundary.

The exemption list matters more than it looks. A region of two cells — eyes,
brows, a mouth — must not be modulated: the photo splits the pair into two
near-identical shades, quantization then merges one of them into the
surrounding skin, and the feature comes out lopsided.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.imaging.color import linear_to_oklab
from app.services.shading import shift_lightness

__all__ = ["paint", "modulate_by_photo"]


def paint(template: list[list[str]], palette: dict[str, np.ndarray]) -> np.ndarray:
    """Turn a grid of region codes into an (H, W, 3) linear-light array."""
    return np.stack(
        [
            np.stack([np.asarray(palette[code], dtype=np.float32).reshape(3) for code in row])
            for row in template
        ]
    ).astype(np.float32)


def modulate_by_photo(
    painted: np.ndarray,
    template: list[list[str]],
    photo: np.ndarray | None,
    modulation: float,
    exempt: Iterable[str] = (),
) -> np.ndarray:
    """Add the photo's within-region lightness variation back to a flat paint.

    ``photo`` must already be downscaled to the same shape. Cells whose region
    code is in ``exempt`` are left exactly as painted.
    """
    if photo is None or modulation <= 0 or photo.shape[:2] != painted.shape[:2]:
        return painted

    height, width = painted.shape[:2]
    lightness = linear_to_oklab(photo)[:, :, 0]
    exempt = set(exempt)
    out = painted.copy()

    codes = {template[r][c] for r in range(height) for c in range(width)}
    for code in codes - exempt:
        cells = [(r, c) for r in range(height) for c in range(width) if template[r][c] == code]
        mean = float(np.mean([lightness[r, c] for r, c in cells]))
        for r, c in cells:
            out[r, c] = shift_lightness(
                out[r, c], float((lightness[r, c] - mean) * modulation)
            ).reshape(3)
    return out
