"""Quality metrics — so "the skin looks better now" can be a number.

Without a metric, every tuning change is a coin flip. Three notes on the
choice made here:

* **L2 / MSE is deliberately not offered as an objective.** The minimizer of
  L2 is the average of all plausible answers, which is exactly "blurry". It
  causes the failure mode we are trying to fix; it cannot detect it.
* **SSIM** (Wang et al., IEEE TIP 2004) compares local luminance, contrast and
  structure. It is cheap and tracks human judgement far better than L2.
* It is computed on the **OKLab L channel**, not a naive 0.299R+0.587G+0.114B
  grey, so "brightness" means perceptual lightness.

Comparison happens at *source* resolution with the skin nearest-neighbour
upsampled, because that is how a player actually sees it: big blocks, not a
64-pixel thumbnail.
"""

from __future__ import annotations

import numpy as np

from app.imaging.color import delta_e_ok, linear_to_oklab

__all__ = ["upsample_nearest", "ssim", "compare"]

_C1 = (0.01) ** 2  # dynamic range of the L channel is ~1
_C2 = (0.03) ** 2


def upsample_nearest(lin: np.ndarray, height: int, width: int) -> np.ndarray:
    """Block-replicate a small linear-light image up to (height, width)."""
    rows = (np.arange(height) * lin.shape[0] // height).clip(0, lin.shape[0] - 1)
    cols = (np.arange(width) * lin.shape[1] // width).clip(0, lin.shape[1] - 1)
    return lin[rows][:, cols]


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    """Uniform (box) filter via an integral image, edge-padded.

    Hand-rolled rather than pulled from scipy: it is eight lines, exact, and
    keeps the dependency list at numpy + Pillow.
    """
    h, w = x.shape
    k = 2 * radius + 1
    padded = np.pad(np.asarray(x, dtype=np.float64), radius, mode="edge")
    ii = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    total = ii[k : k + h, k : k + w] - ii[0:h, k : k + w] - ii[k : k + h, 0:w] + ii[0:h, 0:w]
    return (total / (k * k)).astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray, radius: int = 3) -> float:
    """Mean SSIM between two single-channel float arrays of equal shape."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    radius = max(1, min(radius, (min(a.shape) - 1) // 2))

    mu_a, mu_b = _box_mean(a, radius), _box_mean(b, radius)
    saa = _box_mean(a * a, radius) - mu_a * mu_a
    sbb = _box_mean(b * b, radius) - mu_b * mu_b
    sab = _box_mean(a * b, radius) - mu_a * mu_b

    num = (2 * mu_a * mu_b + _C1) * (2 * sab + _C2)
    den = (mu_a**2 + mu_b**2 + _C1) * (saa + sbb + _C2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def compare(source_lin: np.ndarray, rendered_lin: np.ndarray) -> dict[str, float]:
    """Score a rendered region against the source crop it came from.

    Returns ``ssim`` (1.0 = identical structure), mean and 95th-percentile
    OKLab ΔE (lower is better; ~0.02 is around the threshold of noticing), and
    ``detail``.

    ``detail`` deserves a word, because SSIM alone is actively misleading at
    16×. A face crop is mostly flat cheek; an eye is two pixels out of 64. A
    method that averages the eye away and nails the cheek scores *better* on
    SSIM than one that keeps the eye and slightly misplaces the cheek — the
    metric is dominated by area, and the thing a human looks at has almost
    none. ``detail`` is the ratio of the rendered image's lightness spread to
    the source's: ≈1.0 means the tonal range survived the reduction, well
    under 1.0 means it was averaged into mush. Read the two together.
    """
    h, w = source_lin.shape[:2]
    up = upsample_nearest(rendered_lin, h, w)
    src_lab, up_lab = linear_to_oklab(source_lin), linear_to_oklab(up)
    de = delta_e_ok(src_lab, up_lab)
    src_spread = float(src_lab[:, :, 0].std())
    return {
        "ssim": round(ssim(src_lab[:, :, 0], up_lab[:, :, 0]), 4),
        "delta_e_mean": round(float(de.mean()), 4),
        "delta_e_p95": round(float(np.percentile(de, 95)), 4),
        "detail": round(float(up_lab[:, :, 0].std() / src_spread), 4) if src_spread > 1e-6 else 1.0,
    }
