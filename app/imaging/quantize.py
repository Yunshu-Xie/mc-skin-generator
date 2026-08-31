"""Palette construction and assignment — weighted k-means in OKLab.

Two decisions in here matter more than the algorithm itself.

**Clustering happens in OKLab, not RGB.** Euclidean distance in RGB has
almost no relationship to "how different do these look". Run k-means in RGB on
a portrait and it will happily merge two skin tones a human reads as clearly
different while spending three palette slots separating shadows nobody can
tell apart. That is what "the face came out muddy" actually is.

**One palette for the whole skin, not one per region.** Quantizing each face
separately lets the same forearm drift to a different shade on the arm and on
the hand. A single weighted pass over every region keeps materials consistent.
Weights let the face carry more influence than a trouser leg.

Some palette slots are *anchored*: the semantic colors the vision model
reports (skin, hair, eyes, shirt…) occupy fixed indices and are never moved by
the optimizer. That keeps a stable handle for later recoloring — swap the hex
at one index and everything drawn with it retints — and guarantees the palette
always spans the roles the procedural faces need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.imaging.color import linear_to_hex, linear_to_oklab, oklab_to_linear

__all__ = ["Palette", "kmeans_oklab", "build_palette", "assign"]


@dataclass
class Palette:
    """A fixed set of colors, some of which carry a semantic role."""

    linear: np.ndarray  # (k, 3) linear-light RGB
    roles: dict[str, int] = field(default_factory=dict)

    @property
    def lab(self) -> np.ndarray:
        return linear_to_oklab(self.linear)

    @property
    def size(self) -> int:
        return int(self.linear.shape[0])

    def hex_list(self) -> list[str]:
        return [linear_to_hex(c) for c in self.linear]

    def index_of(self, role: str) -> int | None:
        return self.roles.get(role)

    def color_of(self, role: str, fallback: np.ndarray | None = None) -> np.ndarray:
        idx = self.roles.get(role)
        if idx is None:
            if fallback is None:
                raise KeyError(f"palette has no role {role!r} and no fallback given")
            return fallback
        return self.linear[idx]


def _kmeanspp_init(
    lab: np.ndarray, k: int, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding, weighted by sample importance."""
    n = lab.shape[0]
    first = int(rng.choice(n, p=weights / weights.sum()))
    centers = [lab[first]]
    d2 = np.sum((lab - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        p = d2 * weights
        total = p.sum()
        if total <= 1e-12:  # every remaining sample coincides with a center
            centers.append(lab[int(rng.integers(n))])
        else:
            centers.append(lab[int(rng.choice(n, p=p / total))])
        d2 = np.minimum(d2, np.sum((lab - centers[-1]) ** 2, axis=1))
    return np.asarray(centers, dtype=np.float32)


def kmeans_oklab(
    lab: np.ndarray,
    k: int,
    weights: np.ndarray | None = None,
    fixed: np.ndarray | None = None,
    iters: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Weighted Lloyd's algorithm in OKLab.

    ``fixed`` centers participate in assignment but are never moved, so
    anchored roles stay exactly on the color the vision model reported.
    Returns the free centers only, shape (k, 3).
    """
    if k <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    lab = np.asarray(lab, dtype=np.float32).reshape(-1, 3)
    if weights is None:
        weights = np.ones(lab.shape[0], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    fixed = (
        np.zeros((0, 3), dtype=np.float32)
        if fixed is None
        else np.asarray(fixed, dtype=np.float32).reshape(-1, 3)
    )

    k = min(k, lab.shape[0])
    rng = np.random.default_rng(seed)
    free = _kmeanspp_init(lab, k, weights, rng)

    for _ in range(iters):
        centers = np.concatenate([fixed, free], axis=0)
        d2 = np.sum((lab[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        moved = 0.0
        for j in range(k):
            member = labels == (len(fixed) + j)
            if not member.any():
                # Empty cluster: re-seed it on the worst-represented sample.
                worst = int(np.argmax(d2[np.arange(len(lab)), labels] * weights))
                new = lab[worst]
            else:
                w = weights[member][:, None]
                new = (lab[member] * w).sum(axis=0) / max(w.sum(), 1e-8)
            moved = max(moved, float(np.linalg.norm(new - free[j])))
            free[j] = new
        if moved < 1e-4:
            break
    return free.astype(np.float32)


def build_palette(
    samples_lin: np.ndarray,
    size: int,
    weights: np.ndarray | None = None,
    anchors: dict[str, np.ndarray] | None = None,
    seed: int = 0,
) -> Palette:
    """Build one palette of ``size`` colors covering all given samples.

    ``anchors`` maps a role name to a linear-light color; each takes a fixed
    slot at the front of the palette, in insertion order.
    """
    anchors = anchors or {}
    roles = {name: i for i, name in enumerate(anchors)}
    anchor_lin = (
        np.stack(list(anchors.values())).astype(np.float32)
        if anchors
        else np.zeros((0, 3), dtype=np.float32)
    )

    samples_lin = np.asarray(samples_lin, dtype=np.float32).reshape(-1, 3)
    free_count = max(0, size - len(anchors))
    if samples_lin.size == 0 or free_count == 0:
        return Palette(linear=anchor_lin, roles=roles)

    free_lab = kmeans_oklab(
        linear_to_oklab(samples_lin),
        free_count,
        weights=weights,
        fixed=linear_to_oklab(anchor_lin) if len(anchor_lin) else None,
        seed=seed,
    )
    return Palette(
        linear=np.concatenate([anchor_lin, oklab_to_linear(free_lab)], axis=0),
        roles=roles,
    )


def assign(grid_lin: np.ndarray, palette: Palette) -> np.ndarray:
    """Map an (H, W, 3) linear grid to (H, W) palette indices, nearest in OKLab."""
    h, w = grid_lin.shape[:2]
    lab = linear_to_oklab(grid_lin).reshape(-1, 3)
    d2 = np.sum((lab[:, None, :] - palette.lab[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).reshape(h, w).astype(np.int32)
