"""Separating a material's own color from the light falling on it.

A photograph records albedo × illumination. A Minecraft skin wants albedo
alone: the game lights the model itself, so a shadow baked into the texture is
shading applied twice, and it also wastes palette slots — a white dress in
sun and the same dress in shade quantize to two different "materials".

Full intrinsic image decomposition is an open research problem. This is the
cheap, old and surprisingly serviceable approximation behind Retinex:

    illumination is low-frequency, albedo is what is left

So blur the OKLab lightness channel hard enough that only the illumination
gradient survives, subtract it, and keep the mean lightness so the result does
not drift. Chromaticity is left alone — shadows in a photograph are usually
darker and slightly bluer, but the lightness term dominates by far, and
touching hue here does more harm than good.

What this does and does not do:

* a garment lit from one side comes out evenly lit — the thing we want;
* a hard cast shadow with a crisp edge is *not* removed, because at that
  frequency it is indistinguishable from a real material boundary;
* it cannot recover detail that the shadow destroyed.
"""

from __future__ import annotations

import numpy as np

from app.imaging.color import linear_to_oklab, oklab_to_linear
from app.imaging.downscale import gaussian_blur

__all__ = ["remove_shading"]

# Blur radius as a fraction of the region's smaller side. Large on purpose:
# anything sharper starts erasing the material boundaries we want to keep.
SIGMA_FRACTION = 0.28
MIN_SIGMA = 2.0


def remove_shading(
    lin: np.ndarray, strength: float = 0.8, sigma_fraction: float = SIGMA_FRACTION
) -> np.ndarray:
    """Flatten the illumination gradient across a linear-light region.

    ``strength`` 0 leaves the image alone, 1 removes the whole estimated
    gradient. Values slightly below 1 keep a hint of form, which reads better
    than a completely flat garment.
    """
    if strength <= 0 or lin.ndim != 3:
        return lin

    lab = linear_to_oklab(lin)
    lightness = lab[:, :, 0]

    sigma = max(MIN_SIGMA, sigma_fraction * min(lin.shape[0], lin.shape[1]))
    illumination = gaussian_blur(lightness[:, :, None].repeat(3, axis=2), sigma)[:, :, 0]

    lab[:, :, 0] = np.clip(lightness - strength * (illumination - illumination.mean()), 0.0, 1.0)
    return oklab_to_linear(lab)
