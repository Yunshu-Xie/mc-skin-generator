"""Procedural faces for surfaces the photo cannot show.

A single front-facing photo has no information about the back of a leg. Those
faces are generated, but generated *from the same palette the photo produced*,
so a trouser back is the same blue as the trouser front rather than a second,
slightly different blue.

Two deliberate departures from the previous implementation:

* Lightness is adjusted in **OKLab**, not HSV. HSV's "value" is not lightness —
  darkening a saturated blue and a pale yellow by the same HSV amount produces
  two visibly different perceptual steps, which is why procedurally shaded
  parts used to look like they belonged to a different skin.
* The old ``flat_fill_with_border`` drew a 1px darker outline around every
  face. In-game that reads as a drawn rectangle, not as a limb. It is replaced
  by a soft vertical lightness gradient plus a per-orientation offset — the
  cue the eye actually uses to read volume.
"""

from __future__ import annotations

import numpy as np

from app.imaging.color import linear_to_oklab, oklab_to_linear

__all__ = ["shift_lightness", "solid_face", "shaded_face", "ORIENTATION_LIGHT"]

# How much lighter/darker a face is than the material's base color, by which
# way it points. Roughly a single overhead light source.
ORIENTATION_LIGHT: dict[str, float] = {
    "top": 0.085,
    "front": 0.0,
    "left": -0.045,
    "right": -0.045,
    "back": -0.070,
    "bottom": -0.130,
}

# Top-to-bottom falloff within one face, in OKLab L units.
VERTICAL_GRADIENT = 0.045


def shift_lightness(lin: np.ndarray, delta: float) -> np.ndarray:
    """Shift a linear-light color's OKLab lightness by ``delta``."""
    lab = linear_to_oklab(np.asarray(lin, dtype=np.float32).reshape(-1, 3))
    lab[:, 0] = np.clip(lab[:, 0] + delta, 0.0, 1.0)
    return oklab_to_linear(lab).reshape(np.asarray(lin).shape)


def solid_face(height: int, width: int, color: np.ndarray) -> np.ndarray:
    """A flat (H, W, 3) linear-light face."""
    return np.tile(np.asarray(color, dtype=np.float32).reshape(1, 1, 3), (height, width, 1))


def shaded_face(
    height: int,
    width: int,
    base: np.ndarray,
    orientation: str,
    gradient: float = VERTICAL_GRADIENT,
) -> np.ndarray:
    """A face lit for its orientation, with a soft top-to-bottom falloff."""
    offset = ORIENTATION_LIGHT.get(orientation, 0.0)
    if height > 1:
        ramp = np.linspace(gradient / 2.0, -gradient / 2.0, height, dtype=np.float32)
    else:
        ramp = np.zeros(1, dtype=np.float32)

    lab = linear_to_oklab(np.asarray(base, dtype=np.float32).reshape(3))
    out = np.empty((height, width, 3), dtype=np.float32)
    for row in range(height):
        row_lab = lab.copy()
        row_lab[0] = float(np.clip(row_lab[0] + offset + ramp[row], 0.0, 1.0))
        out[row, :, :] = oklab_to_linear(row_lab.reshape(1, 3)).reshape(3)
    return out
