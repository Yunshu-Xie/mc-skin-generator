"""Color science primitives: sRGB ↔ linear light ↔ OKLab.

Every module downstream of this one works in **linear light** (physical
radiance, float32 in [0,1]) or in **OKLab** (perceptually uniform), never in
raw sRGB values.

Why this file exists at all: sRGB stores roughly the 1/2.2 power of physical
light, so averaging sRGB values — which is what every resize, blur and blend
does — is a mathematically meaningless operation. It darkens edges, greys out
fine texture and eats highlights. Decode → operate → encode is not an
optimization here, it is the difference between a correct and an incorrect
pipeline. See docs/ARCHITECTURE.md §1.

OKLab matrices are Björn Ottosson's (https://bottosson.github.io/posts/oklab/).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "u8_to_linear",
    "linear_to_u8",
    "linear_to_oklab",
    "oklab_to_linear",
    "hex_to_linear",
    "linear_to_hex",
    "delta_e_ok",
]

# ── sRGB transfer function ────────────────────────────────────────────


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0,1] → linear light [0,1]."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear light [0,1] → sRGB-encoded [0,1]."""
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


def u8_to_linear(arr: np.ndarray) -> np.ndarray:
    """uint8 sRGB array (…, 3) → float32 linear light."""
    return srgb_to_linear(np.asarray(arr, dtype=np.float32) / 255.0)


def linear_to_u8(arr: np.ndarray) -> np.ndarray:
    """float32 linear light (…, 3) → uint8 sRGB, with correct rounding."""
    return np.clip(np.rint(linear_to_srgb(arr) * 255.0), 0, 255).astype(np.uint8)


# ── OKLab ─────────────────────────────────────────────────────────────

_LIN_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)

_LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)

_LAB_TO_LMS = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float32,
)

_LMS_TO_LIN = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float32,
)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Linear-light RGB (…, 3) → OKLab (…, 3) as (L, a, b)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    lms = rgb @ _LIN_TO_LMS.T
    lms_ = np.cbrt(lms)
    return (lms_ @ _LMS_TO_LAB.T).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    """OKLab (…, 3) → linear-light RGB (…, 3), clipped to gamut."""
    lab = np.asarray(lab, dtype=np.float32)
    lms_ = lab @ _LAB_TO_LMS.T
    lms = lms_**3
    return np.clip(lms @ _LMS_TO_LIN.T, 0.0, 1.0).astype(np.float32)


def delta_e_ok(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perceptual distance between two OKLab arrays (Euclidean in OKLab).

    Roughly: 0.01 ≈ a just-noticeable difference, 0.05 ≈ obviously different.
    """
    return np.linalg.norm(
        np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32), axis=-1
    )


# ── hex helpers (the boundary with the JSON/API world) ────────────────


def hex_to_linear(value: str) -> np.ndarray:
    """'#RRGGBB' → float32 linear-light array of shape (3,)."""
    h = value.strip().lstrip("#")
    if len(h) not in (6, 8):
        raise ValueError(f"Invalid hex color: {value!r}")
    rgb = np.array([int(h[i : i + 2], 16) for i in range(0, 6, 2)], dtype=np.float32)
    return u8_to_linear(rgb)


def linear_to_hex(lin: np.ndarray) -> str:
    """float32 linear-light (3,) → '#RRGGBB'."""
    r, g, b = linear_to_u8(np.asarray(lin, dtype=np.float32).reshape(3))
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
