"""Deterministic photo → pixel-art rendering: crop, downsample, quantize.

Replaces asking the Vision model to author per-pixel palette indices
directly (unreliable for precise, multi-region color detail) with real
image processing on the actual uploaded photo, guided only by AI-provided
region bounding boxes (see app.services.claude_vision). See
docs/superpowers/specs/2026-07-10-photo-derived-pixel-rendering-design.md
for the full design.
"""

from __future__ import annotations

import colorsys
from collections import Counter

from PIL import Image

Bbox = tuple[float, float, float, float]  # (x0, y0, x1, y1), fractions of image size

_HEAD_BAND_ROWS = 8  # head_front's row template is 8 rows tall

# HSV shadow-consolidation tuning (see the 2026-07-12 design spec). These are
# a starting point, tuned visually against the real test images.
SAT_FLOOR = 0.25  # below this saturation a color is treated as near-gray
HUE_TOL = 0.06  # hue distance (fraction of the 0-1 wheel) to be "same material"
SAT_TOL = 0.30  # saturation distance to be "same material" (saturated colors)
GRAY_VALUE_BANDS = 5  # near-grays split into this many value bands
HIGHLIGHT_PERCENTILE = 80  # representative value = this percentile of a cluster's values


def crop_region(image: Image.Image, bbox: Bbox) -> Image.Image | None:
    """Crop `image` to `bbox` (fractional coords), clamped to bounds.

    Returns None if the resulting crop has zero width or height.
    """
    w, h = image.size
    x0, y0, x1, y1 = bbox
    left = max(0, min(w, round(x0 * w)))
    top = max(0, min(h, round(y0 * h)))
    right = max(0, min(w, round(x1 * w)))
    bottom = max(0, min(h, round(y1 * h)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _dominant_color(region: Image.Image) -> tuple[int, int, int]:
    """Most common RGB color in `region`, via a full-resolution histogram."""
    rgb = region.convert("RGB")
    colors = rgb.getcolors(maxcolors=rgb.width * rgb.height)
    return max(colors, key=lambda c: c[0])[1]


def downsample_dominant(
    region: Image.Image, target_h: int, target_w: int
) -> list[list[tuple[int, int, int]]]:
    """Downsample `region` to target_h x target_w, one dominant color per cell.

    Uses the most common color within each output cell's source tile rather
    than a blurred average, so a contrasting boundary (e.g. a red trim
    against a white base) stays crisp instead of blending into a muddy
    in-between color.
    """
    w, h = region.size
    out: list[list[tuple[int, int, int]]] = []
    for row in range(target_h):
        top = h * row // target_h
        bottom = max(top + 1, h * (row + 1) // target_h)
        out_row: list[tuple[int, int, int]] = []
        for col in range(target_w):
            left = w * col // target_w
            right = max(left + 1, w * (col + 1) // target_w)
            tile = region.crop((left, top, right, bottom))
            out_row.append(_dominant_color(tile))
        out.append(out_row)
    return out


def quantize_shared(
    rgb_grids: dict[str, list[list[tuple[int, int, int]]]], max_colors: int = 24
) -> tuple[list[str], dict[str, list[list[int]]]]:
    """Quantize all pixels across `rgb_grids` to one shared palette.

    A single shared quantization pass (rather than one per region) keeps
    e.g. skin tone consistent between an arm and a leg instead of each
    drifting to a slightly different quantized shade.

    Returns (palette hex list, {key: index grid}) — palette[i] is the hex
    for index i, and each returned index grid has the same shape as the
    input grid of the same key.
    """
    shapes: dict[str, tuple[int, int]] = {}
    all_pixels: list[tuple[int, int, int]] = []
    for key, grid in rgb_grids.items():
        shapes[key] = (len(grid), len(grid[0]) if grid else 0)
        for row in grid:
            all_pixels.extend(row)

    if not all_pixels:
        return [], {key: [] for key in rgb_grids}

    flat_img = Image.new("RGB", (len(all_pixels), 1))
    flat_img.putdata(all_pixels)
    quantized = flat_img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)

    palette_rgb = quantized.getpalette()[: max_colors * 3]
    palette = [
        "#{:02X}{:02X}{:02X}".format(*palette_rgb[i : i + 3])
        for i in range(0, len(palette_rgb), 3)
    ]
    flat_indices = list(quantized.getdata())

    out: dict[str, list[list[int]]] = {}
    pos = 0
    for key, (h, w) in shapes.items():
        out[key] = [flat_indices[pos + row * w : pos + (row + 1) * w] for row in range(h)]
        pos += h * w
    return palette, out


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (c / 255 for c in rgb)
    return colorsys.rgb_to_hsv(r, g, b)


def _hue_dist(h1: float, h2: float) -> float:
    """Circular distance on the 0-1 hue wheel."""
    d = abs(h1 - h2)
    return min(d, 1.0 - d)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _same_material(
    a_hsv: tuple[float, float, float], b_hsv: tuple[float, float, float]
) -> bool:
    """Whether two colors are the same material under different lighting.

    Saturated colors: same hue + saturation (value, the shadow axis, ignored).
    Near-gray colors: hue is meaningless, so cluster by value band instead —
    this keeps black / gray / white distinct rather than collapsing them.
    """
    ha, sa, va = a_hsv
    hb, sb, vb = b_hsv
    a_gray = sa < SAT_FLOOR
    b_gray = sb < SAT_FLOOR
    if a_gray != b_gray:
        return False
    if a_gray:
        band_a = min(int(va * GRAY_VALUE_BANDS), GRAY_VALUE_BANDS - 1)
        band_b = min(int(vb * GRAY_VALUE_BANDS), GRAY_VALUE_BANDS - 1)
        return band_a == band_b
    return _hue_dist(ha, hb) <= HUE_TOL and abs(sa - sb) <= SAT_TOL


def consolidate_shadows(
    rgb_grids: dict[str, list[list[tuple[int, int, int]]]],
) -> dict[str, list[list[tuple[int, int, int]]]]:
    """Collapse shadow variants of one material to a single highlight-tone color.

    Shadow preserves hue+saturation and only lowers value, so colors sharing
    hue+saturation but differing in value are the same material under different
    lighting; they merge to one representative at a high value percentile (the
    brighter/intrinsic tone). Near-gray colors cluster by value band instead
    (see _same_material). Runs before quantize_shared so the palette carries
    flat intrinsic colors rather than the photo's baked-in shading — Minecraft
    re-lights the 3D model itself.
    """
    counts: Counter[tuple[int, int, int]] = Counter()
    for grid in rgb_grids.values():
        for row in grid:
            for cell in row:
                counts[cell] += 1
    if not counts:
        return {k: [row[:] for row in grid] for k, grid in rgb_grids.items()}

    hsv = {c: _rgb_to_hsv(c) for c in counts}

    # Greedy clustering: each color joins the first cluster whose seed it matches.
    clusters: list[list[tuple[int, int, int]]] = []
    for color in counts:
        for cluster in clusters:
            if _same_material(hsv[color], hsv[cluster[0]]):
                cluster.append(color)
                break
        else:
            clusters.append([color])

    rep: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for cluster in clusters:
        seed = max(cluster, key=lambda c: counts[c])  # most-common member sets hue/sat
        sh, ss, _sv = hsv[seed]
        values = sorted(v for c in cluster for v in [hsv[c][2]] * counts[c])
        rep_v = _percentile(values, HIGHLIGHT_PERCENTILE)
        rr, rg, rb = colorsys.hsv_to_rgb(sh, ss, rep_v)
        rep_rgb = (round(rr * 255), round(rg * 255), round(rb * 255))
        for c in cluster:
            rep[c] = rep_rgb

    return {
        key: [[rep[cell] for cell in row] for row in grid]
        for key, grid in rgb_grids.items()
    }


def _band(face_crop: Image.Image, row_start: int, row_end: int) -> Image.Image:
    """Crop the horizontal band spanning rows [row_start, row_end) of an 8-row template."""
    w, h = face_crop.size
    top = h * row_start // _HEAD_BAND_ROWS
    bottom = max(top + 1, h * row_end // _HEAD_BAND_ROWS)
    return face_crop.crop((0, top, w, bottom))


def _minority_color(band: Image.Image, fallback: str) -> str:
    """Quantize `band` to 2 colors and return the less-prevalent one as hex.

    The deciding criterion is prevalence, not darkness: a feature like an
    eye or mouth occupies less area than the surrounding skin within its
    band, so the minority cluster is the feature.
    """
    rgb = band.convert("RGB")
    quantized = rgb.quantize(colors=2, method=Image.Quantize.MEDIANCUT)
    counts = Counter(quantized.getdata())
    if len(counts) < 2:
        return fallback
    minority_idx = min(counts, key=lambda idx: counts[idx])
    palette = quantized.getpalette()
    r, g, b = palette[minority_idx * 3 : minority_idx * 3 + 3]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def _dominant_quantized(band: Image.Image, n_colors: int = 5) -> str:
    """Most common color of `band` after quantizing to n_colors, as hex.

    More robust than the single most-common exact pixel for textured photo
    regions: a textured material (e.g. hair) spreads across many near-identical
    exact values, which lets a flat background of ONE exact value win the raw
    histogram even when it covers less area. Quantizing first collapses the
    material into one bucket that then correctly dominates.
    """
    rgb = band.convert("RGB")
    quantized = rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    idx = Counter(quantized.getdata()).most_common(1)[0][0]
    r, g, b = quantized.getpalette()[idx * 3 : idx * 3 + 3]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def extract_face_colors(face_crop: Image.Image) -> dict[str, str]:
    """Measure skin/hair/eye/mouth colors from a cropped face region.

    Uses head_front's existing row template (rows 0-1 hair, row 4 skin,
    rows 2-3 eyes, rows 5-6 mouth, out of 8 total rows) to sample real
    photo pixels, rather than asking the model to guess hex values. Hair and
    skin use a quantized dominant (robust to texture + background bleed);
    eyes/mouth use the minority cluster of a 2-color split of their band.
    """
    skin_tone = _dominant_quantized(_band(face_crop, 4, 5))
    hair_color = _dominant_quantized(_band(face_crop, 0, 2))
    eye_color = _minority_color(_band(face_crop, 2, 4), fallback=skin_tone)
    mouth_color = _minority_color(_band(face_crop, 5, 7), fallback=skin_tone)

    return {
        "skin_tone": skin_tone,
        "hair_color": hair_color,
        "eye_color": eye_color,
        "mouth_color": mouth_color,
    }


def dominant_hex(grid: list[list[str]]) -> str:
    """Most common hex color in an already-decoded hex grid."""
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0]


def build_head_front(
    colors: dict[str, str], eye_shape: str, mouth_width: str = "small"
) -> list[list[str]]:
    """Build the 8x8 head_front hex grid from measured colors + the fixed template.

    Layout (columns 0-7):
      rows 0-1  hair
      rows 2-3  two symmetric eyes  (narrow: 1px each at cols 2 & 5;
                                     round: 2px each at cols 1-2 & 5-6)
      row  4    nose/cheek (skin)
      row  5    mouth, 1px tall, centered (small: cols 3-4; wide: cols 2-5)
      rows 6-7  chin/neck (skin)
    """
    skin = colors["skin_tone"]
    hair = colors["hair_color"]
    eye = colors["eye_color"]
    mouth = colors["mouth_color"]

    grid = [[skin] * 8 for _ in range(8)]
    grid[0] = [hair] * 8
    grid[1] = [hair] * 8

    # Two symmetric eyes (about the vertical centerline, col 3.5).
    if eye_shape == "round":
        eye_cols = (1, 2, 5, 6)
    else:  # narrow
        eye_cols = (2, 5)
    for row in (2, 3):
        for col in eye_cols:
            grid[row][col] = eye

    # Centered mouth on row 5 only — a horizontal segment, not a block.
    mouth_cols = (2, 3, 4, 5) if mouth_width == "wide" else (3, 4)
    for col in mouth_cols:
        grid[5][col] = mouth

    return grid
