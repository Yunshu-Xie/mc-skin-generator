"""Pure image science: no AI, no web framework, no Minecraft.

Everything here takes and returns numpy arrays in linear light or OKLab and
can be tested numerically. The Minecraft-specific layers (app.services) build
on top of it. Keeping the split sharp is what makes the fidelity work
reviewable: if a color came out wrong, it is either a bug in here (provable
with a unit test) or a bad decision up there (visible in a rendering).
"""

from app.imaging.albedo import remove_shading
from app.imaging.color import (
    delta_e_ok,
    hex_to_linear,
    linear_to_hex,
    linear_to_oklab,
    linear_to_srgb,
    linear_to_u8,
    oklab_to_linear,
    srgb_to_linear,
    u8_to_linear,
)
from app.imaging.downscale import downscale, fit_crop, unsharp
from app.imaging.metrics import compare, ssim, upsample_nearest
from app.imaging.quantize import Palette, assign, build_palette, kmeans_oklab

__all__ = [
    "Palette",
    "remove_shading",
    "assign",
    "build_palette",
    "compare",
    "delta_e_ok",
    "downscale",
    "fit_crop",
    "hex_to_linear",
    "kmeans_oklab",
    "linear_to_hex",
    "linear_to_oklab",
    "linear_to_srgb",
    "linear_to_u8",
    "oklab_to_linear",
    "srgb_to_linear",
    "ssim",
    "u8_to_linear",
    "unsharp",
    "upsample_nearest",
]
