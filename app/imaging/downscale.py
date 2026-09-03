"""Structure-preserving downscaling, done in linear light.

A Minecraft skin asks for ~16× reduction: a whole face has to survive in an
8×8 tile where an eye is one or two pixels. At that ratio a fixed convolution
kernel (bilinear / bicubic / Lanczos) is guaranteed to fail — it averages the
eye into the cheek and the result reads as a flesh-colored blob.

Three strategies are provided, all operating on float32 linear-light arrays:

``box``      exact area average. The *correct* baseline (this is what
             everyone thinks they are doing when they call ``resize``), and
             the right choice for flat regions like a shirt.

``dpid``     Weber et al., *Rapid, Detail-Preserving Image Downscaling*
             (SIGGRAPH Asia 2016). Counter-intuitive and very effective: the
             further an input pixel is from its tile's mean, the *more* it is
             weighted. Outliers at 16× are usually the detail you wanted to
             keep, not noise to suppress. Distances are measured in OKLab so
             "far" means perceptually far, not numerically far.

``dominant`` modal color of the tile. Keeps hard boundaries (a red trim on a
             white shirt) perfectly crisp at the cost of all gradients.

Default is ``dpid``: it keeps facial features legible without inventing the
posterized look ``dominant`` gives.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from app.imaging.color import linear_to_oklab

Method = Literal["box", "dpid", "dominant"]

__all__ = ["fit_crop", "gaussian_blur", "unsharp", "downscale", "Method"]


def _tile_bounds(src: int, dst: int) -> list[tuple[int, int]]:
    """Split ``src`` rows/cols into ``dst`` contiguous, non-empty tiles."""
    out: list[tuple[int, int]] = []
    for i in range(dst):
        lo = src * i // dst
        hi = max(lo + 1, src * (i + 1) // dst)
        out.append((min(lo, src - 1), min(hi, src)))
    return out


def fit_crop(lin: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop ``lin`` (H, W, 3) to the aspect ratio of target_h × target_w.

    Cropping rather than squashing: a face squeezed into an 8×8 square looks
    wrong in a way no amount of downscaling quality can repair.
    """
    h, w = lin.shape[:2]
    want = target_w / target_h
    have = w / h
    if have > want:  # too wide → trim left/right
        new_w = max(1, int(round(h * want)))
        x0 = (w - new_w) // 2
        return lin[:, x0 : x0 + new_w]
    new_h = max(1, int(round(w / want)))
    y0 = (h - new_h) // 2
    return lin[y0 : y0 + new_h, :]


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(lin: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a linear-light array, edge-padded."""
    if sigma <= 0:
        return lin
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    h, w = lin.shape[:2]

    pad = np.pad(lin, ((0, 0), (r, r), (0, 0)), mode="edge")
    tmp = np.zeros_like(lin)
    for i, weight in enumerate(k):
        tmp += weight * pad[:, i : i + w]

    pad = np.pad(tmp, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(lin)
    for i, weight in enumerate(k):
        out += weight * pad[i : i + h]
    return out.astype(np.float32)


def unsharp(lin: np.ndarray, amount: float, sigma: float = 1.2) -> np.ndarray:
    """Pre-sharpen in linear light to compensate for downscaling's MTF loss.

    Applied *before* the reduction, not after: sharpening 8×8 output would
    just make blocky artifacts crunchier, whereas boosting the source's high
    frequencies gives the reduction something to preserve.
    """
    if amount <= 0:
        return lin
    high = lin - gaussian_blur(lin, sigma)
    return np.clip(lin + amount * high, 0.0, 1.0).astype(np.float32)


def _box(lin: np.ndarray, rows: list, cols: list) -> np.ndarray:
    out = np.empty((len(rows), len(cols), 3), dtype=np.float32)
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(cols):
            out[r, c] = lin[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return out


def _dpid(lin: np.ndarray, rows: list, cols: list, lam: float) -> np.ndarray:
    lab = linear_to_oklab(lin)
    out = np.empty((len(rows), len(cols), 3), dtype=np.float32)
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(cols):
            tile = lin[y0:y1, x0:x1].reshape(-1, 3)
            tile_lab = lab[y0:y1, x0:x1].reshape(-1, 3)
            mean_lab = tile_lab.mean(axis=0)
            dist = np.linalg.norm(tile_lab - mean_lab, axis=1)
            w = np.power(dist, lam)
            total = w.sum()
            if total <= 1e-8:  # perfectly flat tile — box average is exact
                out[r, c] = tile.mean(axis=0)
            else:
                out[r, c] = (tile * w[:, None]).sum(axis=0) / total
    return out


def _dominant(lin: np.ndarray, rows: list, cols: list, bin_size: float) -> np.ndarray:
    lab = linear_to_oklab(lin)
    out = np.empty((len(rows), len(cols), 3), dtype=np.float32)
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(cols):
            tile = lin[y0:y1, x0:x1].reshape(-1, 3)
            keys = np.floor(lab[y0:y1, x0:x1].reshape(-1, 3) / bin_size).astype(np.int64)
            packed = (keys[:, 0] << 42) ^ (keys[:, 1] << 21) ^ keys[:, 2]
            uniq, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
            winner = int(np.argmax(counts))
            out[r, c] = tile[inverse == winner].mean(axis=0)
    return out


def downscale(
    lin: np.ndarray,
    target_h: int,
    target_w: int,
    method: Method = "dpid",
    presharpen: float = 0.0,
    dpid_lambda: float = 1.0,
    dominant_bin: float = 0.05,
) -> np.ndarray:
    """Reduce a linear-light image to exactly (target_h, target_w, 3).

    The caller is expected to have cropped to the right aspect already
    (:func:`fit_crop`); this function does not preserve aspect ratio.
    """
    if lin.ndim != 3 or lin.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) linear array, got {lin.shape}")
    lin = unsharp(lin, presharpen) if presharpen else lin
    rows = _tile_bounds(lin.shape[0], target_h)
    cols = _tile_bounds(lin.shape[1], target_w)

    if method == "box":
        return _box(lin, rows, cols)
    if method == "dominant":
        return _dominant(lin, rows, cols, dominant_bin)
    if method == "dpid":
        return _dpid(lin, rows, cols, dpid_lambda)
    raise ValueError(f"unknown downscale method: {method!r}")
