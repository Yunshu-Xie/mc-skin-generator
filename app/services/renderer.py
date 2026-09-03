"""Photo → 64×64 skin. This is where the actual pixels are decided.

The pipeline, in order — the order is the point:

1. **Decode to linear light.** Everything after this averages, weights and
   blends; none of that is valid on sRGB values.
2. **Crop each region** to the box the vision model reported, then trim to the
   destination face's aspect ratio (crop, never squash).
3. **Pre-sharpen**, because a 16× reduction has a brutal MTF rolloff and the
   only chance to protect an eye that will be one pixel wide is to boost its
   contrast beforehand.
4. **Structure-preserving downscale** (DPID by default) to the exact face size.
5. **Build one palette for the whole skin** by weighted k-means in OKLab, with
   the vision model's semantic colors anchored at fixed indices. The face gets
   a heavier weight than a trouser leg, so palette slots are spent where a
   human looks.
6. **Derive the faces the photo cannot see** — back of the head, insides of
   limbs — from that same palette (``app.services.shading``).
7. **Assign** every region to the shared palette and emit hex grids for
   ``app.services.skin_assembler``, whose interface is unchanged.
8. **Score** the result against the source crops (SSIM + OKLab ΔE), so a
   tuning change can be judged by a number rather than by vibes.

Step 5 being a single global pass, rather than one quantization per region, is
what keeps skin tone identical between a cheek and a forearm.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.imaging.albedo import remove_shading
from app.imaging.color import (
    hex_to_linear,
    linear_to_hex,
    linear_to_oklab,
    u8_to_linear,
)
from app.imaging.downscale import Method, downscale, fit_crop
from app.imaging.metrics import compare
from app.imaging.quantize import Palette, assign, build_palette
from app.services.face import eye_row_from_box, render_face
from app.services.layout import Bbox, Layout
from app.services.shading import (
    ORIENTATION_LIGHT,
    shaded_face,
    shift_lightness,
    solid_face,
)
from app.services.skin_map import ModelType, get_all_regions

logger = logging.getLogger(__name__)

# Skin faces rendered directly from the photograph, and the layout anchor each
# one is cut from.
PHOTO_FACES: dict[str, str] = {
    "head_front": "face",
    "head_top": "hair_top",
    "body_front": "torso",
    "right_arm_front": "right_arm",
    "left_arm_front": "left_arm",
    "right_leg_front": "legs",
    "left_leg_front": "legs",
}

# Which palette role clothes each body part's unphotographed faces.
GROUP_ROLE: dict[str, str] = {
    "head": "hair_color",
    "body": "shirt_main",
    "right_arm": "arm_main",
    "left_arm": "arm_main",
    "right_leg": "pants_main",
    "left_leg": "pants_main",
}

# Faces where per-pixel structure matters more than flat-color accuracy.
DETAIL_FACES = frozenset({"head_front", "head_top"})

# How strongly the eye band is pulled toward the reported eye color. 1.0 would
# paint flat eyes; this keeps some of the photo's own variation.
EYE_STRENGTH = 0.85

# Within the eye band, how dark a cell must be (relative to the band's own
# mean-to-darkest range) before it is treated as an eye rather than as the skin
# around one — and the largest share of the band that may be called "eye", so
# a mis-sized band cannot paint a stripe across the face.
EYE_MASK_THRESHOLD = 0.5
EYE_MASK_MAX_COVERAGE = 0.5

# Regions whose fidelity is worth measuring and reporting.
SCORED_FACES = ("head_front", "body_front")

# Palette slots 0..6, always these materials in always this order.
ANCHORED_ROLES = (
    "skin_tone",
    "hair_color",
    "eye_color",
    "shirt_main",
    "arm_main",
    "pants_main",
    "shoe_color",
)

# The face is what a player looks at, so it gets a disproportionate say in
# which colors make it into the palette.
FACE_SAMPLE_WEIGHT = 5.0


@dataclass
class RenderConfig:
    """Rendering knobs. Defaults chosen by sweeping against test portraits.

    Two methods, chosen by content rather than one global setting:

    * The **face** uses ``dpid`` with a high λ (1.4). Lower λ — and both other
      methods — score *better* on SSIM while dropping the eyes entirely,
      because SSIM is dominated by the large flat areas of a face crop. Here
      feature survival matters more than average color accuracy.
    * **Clothing and limbs** use ``dominant``, which keeps a red trim against a
      white shirt perfectly crisp instead of blending it into pink, and gives
      the clean flat blocks pixel art wants.

    Watch the ``detail`` metric, not just ``ssim``, when retuning.
    """

    palette_size: int = 16
    face_method: Method = "dpid"
    material_method: Method = "dominant"
    presharpen: float = 0.45
    dpid_lambda: float = 1.4
    eye_strength: float = EYE_STRENGTH
    head_top_margin: float = 0.45
    head_side_margin: float = 0.12
    head_mode: str = "template"  # "template" draws the face; "photo" resamples it
    face_modulation: float = 0.6

    # ── albedo: a photo is material × light, a skin texture wants material ──
    # How much of each region's illumination gradient to flatten before
    # quantizing (0 keeps the photo's shading). 1.0, not something gentler:
    # measured on a white dress with a red trim, partial flattening (0.85) was
    # the *worst* of both — a residual gradient still ate a palette slot while
    # the trim's lightness got pulled toward the mean, and the accent color
    # vanished entirely. Removing the shading outright frees that slot for a
    # real material, and the trim came back.
    flatten_shading: float = 1.0
    # How much the OKLab L axis counts when clustering. Below 1, a garment in
    # sun and in shade fall into one palette slot instead of two. Only bites on
    # materials that have chroma to cluster on: for a near-white garment there
    # is no hue signal and flatten_shading does all the work.
    palette_lightness_weight: float = 0.7
    # Which percentile of a cluster's lightness becomes its color: the material
    # as it looks lit, not an average dragged dark by its own shadow.
    albedo_percentile: float = 80.0
    # How much orientation shading to bake into procedurally generated faces.
    # 0 by default — the game does its own lighting.
    bake_orientation_shading: float = 0.0

    def method_for(self, key: str) -> Method:
        return self.face_method if key in DETAIL_FACES else self.material_method


@dataclass
class RenderResult:
    pixel_data: dict[str, list[list[str]]]
    palette: list[str] = field(default_factory=list)
    roles: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def _split_box(box: Bbox, half: str) -> Bbox:
    """Left or right half of a box — the ``legs`` anchor covers both legs."""
    mid = (box.x0 + box.x1) / 2.0
    return (
        Bbox(box.x0, box.y0, mid, box.y1) if half == "left" else Bbox(mid, box.y0, box.x1, box.y1)
    )


def expand_head_box(box: Bbox, top: float, side: float) -> Bbox:
    """Grow a reported face box outward to cover the whole head.

    Vision models report a *face* box — roughly eyebrows to chin — because
    that is what "face" means to a detector, and no amount of asking for "the
    whole head, hairline to chin" reliably changes it. Feeding that box
    straight into an 8x8 head texture produces a bald mannequin: measured on a
    real run, the eyes landed in row 2 of 8 and not a single cell of hair made
    it into the texture. The head sides and back are derived from this face,
    so the hair then disappears from the entire head.

    Anatomically a face box spans roughly the lower 60% of a head, so growing
    the top edge by ~0.45 of the box height recovers the crown; the sides get
    a smaller margin because hair is only a little wider than the face.
    Clamped to the image, and skipped entirely when both margins are 0.
    """
    if top <= 0 and side <= 0:
        return box
    height, width = box.y1 - box.y0, box.x1 - box.x0
    return Bbox(
        max(0.0, box.x0 - width * side),
        max(0.0, box.y0 - height * top),
        min(1.0, box.x1 + width * side),
        box.y1,
    )


def expand_to_aspect(box: Bbox, target_h: int, target_w: int) -> Bbox:
    """Grow a box to the destination face's aspect ratio, never shrink it.

    The alternative — center-cropping the box down to the right aspect — is
    quietly destructive: a head box is taller than the 8×8 face it feeds, so
    trimming it to square throws away the hairline and the chin and leaves an
    8×8 texture of somebody's cheekbones. Expanding instead pulls in a little
    surrounding image, which costs almost nothing and keeps the subject whole.
    Expansion is clamped to the image, so a subject already touching an edge
    falls back to :func:`app.imaging.downscale.fit_crop`.
    """
    want = target_w / target_h
    width, height = box.x1 - box.x0, box.y1 - box.y0
    if height <= 0 or width <= 0:
        return box

    if width / height < want:
        grow = (height * want - width) / 2.0
        x0, x1, y0, y1 = box.x0 - grow, box.x1 + grow, box.y0, box.y1
    else:
        grow = (width / want - height) / 2.0
        x0, x1, y0, y1 = box.x0, box.x1, box.y0 - grow, box.y1 + grow

    # Slide back inside the image rather than losing the growth we just added.
    if x0 < 0.0:
        x1, x0 = min(1.0, x1 - x0), 0.0
    if x1 > 1.0:
        x0, x1 = max(0.0, x0 - (x1 - 1.0)), 1.0
    if y0 < 0.0:
        y1, y0 = min(1.0, y1 - y0), 0.0
    if y1 > 1.0:
        y0, y1 = max(0.0, y0 - (y1 - 1.0)), 1.0
    return Bbox(x0, y0, x1, y1)


def _crop(lin: np.ndarray, box: Bbox) -> np.ndarray:
    h, w = lin.shape[:2]
    left, top, right, bottom = box.to_pixels(w, h)
    return lin[top:bottom, left:right]


def _tile_column(grid: np.ndarray, col: int, width: int) -> np.ndarray:
    """Repeat one column of a face across ``width`` — a cheap side view.

    The left and right of a head are not in a frontal photo, but the outermost
    column of the front face already carries the right vertical structure
    (hair on top, skin below, neck at the bottom), so stretching it sideways
    lands much closer than any flat fill.
    """
    return np.repeat(grid[:, col : col + 1, :], width, axis=1)


def _render_photo_faces(
    lin: np.ndarray,
    layout: Layout,
    regions: dict[str, dict],
    config: RenderConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Bbox]]:
    """Downscale each photo-derived face.

    Returns (faces, source crops, the crop box actually used per face).
    """
    faces: dict[str, np.ndarray] = {}
    sources: dict[str, np.ndarray] = {}
    used: dict[str, Bbox] = {}

    for key, anchor in PHOTO_FACES.items():
        group, face = key.rsplit("_", 1)
        rect = regions[group][face]
        box = layout.box(anchor)
        if anchor == "face":
            box = expand_head_box(box, config.head_top_margin, config.head_side_margin)
        if anchor == "legs":
            box = _split_box(box, "left" if group == "right_leg" else "right")

        box = expand_to_aspect(box, rect.h, rect.w)
        crop = _crop(lin, box)
        if crop.size == 0:
            continue
        used[key] = box
        crop = fit_crop(crop, rect.h, rect.w)  # no-op unless expansion hit an edge
        crop = remove_shading(crop, config.flatten_shading)
        faces[key] = downscale(
            crop,
            rect.h,
            rect.w,
            method=config.method_for(key),
            presharpen=config.presharpen,
            dpid_lambda=config.dpid_lambda,
        )
        if key in SCORED_FACES:
            sources[key] = crop
    return faces, sources, used


def _derive_head_faces(
    faces: dict[str, np.ndarray],
    regions: dict[str, dict],
    palette_hair: np.ndarray,
    palette_skin: np.ndarray,
    bake: float = 0.0,
) -> None:
    """Fill in head sides, back and underside from the rendered front face."""
    front = faces.get("head_front")
    head = regions["head"]

    for face, col in (("right", 0), ("left", -1)):
        rect = head[face]
        if front is not None and front.shape[0] == rect.h:
            side = _tile_column(front, col % front.shape[1], rect.w)
        else:
            side = solid_face(rect.h, rect.w, palette_hair)
        faces[f"head_{face}"] = _shade_grid(side, face, bake)

    rect = head["back"]
    back = solid_face(rect.h, rect.w, palette_hair)
    if front is not None and rect.h > 1:
        back[-1, :, :] = front[-1, :, :]  # keep the neck continuous
    faces["head_back"] = _shade_grid(back, "back", bake)

    rect = head["bottom"]
    faces["head_bottom"] = shaded_face(rect.h, rect.w, palette_skin, "bottom", strength=bake)


def _shade_grid(grid: np.ndarray, orientation: str, strength: float = 1.0) -> np.ndarray:
    """Apply a face's orientation lighting to an already-textured grid."""
    if strength <= 0:
        return grid
    return shift_lightness(grid, ORIENTATION_LIGHT.get(orientation, 0.0) * strength)


def stamp_eyes(
    grid: np.ndarray,
    face_box: Bbox,
    eyes_box: Bbox,
    eye_color: np.ndarray,
    strength: float = EYE_STRENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Sharpen the eye band toward the reported eye color, weighted by darkness.

    This is the one place the pipeline overrides what pure downsampling
    produced, and it is here because of a measured failure, not a hunch: at
    8x8 an eye covers a few percent of a face crop, so every reduction method
    — box, DPID and dominant alike — averages it into the cheek, and SSIM
    *rewards* them for doing so, because the metric is dominated by the large
    flat areas (see app/imaging/metrics.py). The eye is the feature a human
    recognises a face by, so it gets a prior instead of a vote.

    The prior is deliberately soft. Painting the reported band flat would draw
    a stripe across the face, since a band wide enough to contain both eyes
    also contains the nose bridge and the temples. Instead the band is used as
    a *saliency mask*: within it, each cell is pulled toward the eye color in
    proportion to how much darker it already is than the band's mean. Cells
    that were dark (the eyes) go dark; cells that were pale (skin between and
    beside them) are left alone. Local contrast enhancement, sited by
    semantics rather than by guessing coordinates.

    Returns the adjusted grid and a boolean mask of the cells judged to be
    eyes. The caller snaps those cells to the eye_color palette slot after
    quantization: a blended near-eye color can otherwise land nearer some
    neutral in the palette than to the eye color itself, which is how you get
    grey eyes on a brown-eyed subject.

    A cheap stand-in for the saliency-weighted joint optimization in
    docs/ARCHITECTURE.md section 6, which would derive this rather than assert
    it.
    """
    empty = np.zeros(grid.shape[:2], dtype=bool)
    height, width = grid.shape[:2]
    span_x, span_y = face_box.x1 - face_box.x0, face_box.y1 - face_box.y0
    if span_x <= 0 or span_y <= 0 or strength <= 0:
        return grid, empty

    col0 = int(np.floor((eyes_box.x0 - face_box.x0) / span_x * width))
    col1 = int(np.ceil((eyes_box.x1 - face_box.x0) / span_x * width))
    row0 = int(np.floor((eyes_box.y0 - face_box.y0) / span_y * height))
    row1 = int(np.ceil((eyes_box.y1 - face_box.y0) / span_y * height))

    col0, col1 = max(0, col0), min(width, max(col0 + 1, col1))
    row0, row1 = max(0, row0), min(height, max(row0 + 1, row1))
    if col0 >= width or row0 >= height:
        return grid, empty  # reported band falls outside the crop — leave it

    band = grid[row0:row1, col0:col1]
    lightness = linear_to_oklab(band)[:, :, 0]
    mean, floor = float(lightness.mean()), float(lightness.min())
    if mean - floor < 1e-4:  # a uniform band carries no eye to find
        return grid, empty

    # Normalized against the band's *mean-to-darkest* range, not its full
    # range: a single bright highlight in the band would otherwise stretch the
    # denominator and push the actual eyes below any sensible threshold.
    darkness = np.clip((mean - lightness) / (mean - floor), 0.0, 1.0)
    weight = (strength * darkness)[:, :, None]

    out = grid.copy()
    out[row0:row1, col0:col1] = (1.0 - weight) * band + weight * eye_color

    selected = darkness >= EYE_MASK_THRESHOLD
    if selected.mean() > EYE_MASK_MAX_COVERAGE:
        # A band that is mostly dark is hair or shadow, not two eyes. Keep only
        # the darkest cells rather than painting a stripe across the face.
        cutoff = np.quantile(darkness, 1.0 - EYE_MASK_MAX_COVERAGE)
        selected = darkness > cutoff

    mask = empty.copy()
    mask[row0:row1, col0:col1] = selected
    return out, mask


def _render_procedural_faces(
    faces: dict[str, np.ndarray],
    regions: dict[str, dict],
    palette: Palette,
    bake: float = 0.0,
) -> None:
    """Everything still missing gets a shaded fill from the shared palette."""
    shoe = palette.color_of("shoe_color", palette.color_of("pants_main"))

    for group, role in GROUP_ROLE.items():
        if group == "head":
            continue
        base = palette.color_of(role)
        for face, rect in regions[group].items():
            key = f"{group}_{face}"
            if key in faces:
                continue
            if group.endswith("_leg") and face == "bottom":
                faces[key] = solid_face(rect.h, rect.w, shoe)
            else:
                faces[key] = shaded_face(rect.h, rect.w, base, face, strength=bake)


def render(
    image_bytes: bytes,
    layout: Layout,
    model: ModelType = "classic",
    config: RenderConfig | None = None,
) -> RenderResult:
    """Render a full set of hex pixel grids for every base region of the skin."""
    config = config or RenderConfig()
    regions = get_all_regions(model)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    lin = u8_to_linear(np.asarray(image))

    faces, sources, used_boxes = _render_photo_faces(lin, layout, regions, config)

    # ── the head front: drawn, not resampled (see app/services/face.py) ──
    eye_mask: np.ndarray | None = None
    if config.head_mode == "template":
        head_box = used_boxes.get("head_front")
        eyes = layout.boxes.get("eyes")
        eye_row = (
            eye_row_from_box(head_box.y0, head_box.y1, eyes.y0, eyes.y1)
            if head_box is not None and eyes is not None
            else 3
        )
        faces["head_front"] = render_face(
            style=layout.hair_style,
            eye_row=eye_row,
            skin=hex_to_linear(layout.role("skin_tone")),
            hair=hex_to_linear(layout.role("hair_color")),
            eye=hex_to_linear(layout.role("eye_color")),
            photo=faces.get("head_front"),
            modulation=config.face_modulation,
        )
    elif "eyes" in layout.boxes and "head_front" in faces:
        faces["head_front"], eye_mask = stamp_eyes(
            faces["head_front"],
            used_boxes["head_front"],
            layout.boxes["eyes"],
            hex_to_linear(layout.role("eye_color")),
            config.eye_strength,
        )

    # ── one palette for the whole skin ────────────────────────────────
    # Anchored slots are fixed in count and order, so index N always means the
    # same material — that is what makes `recolor` a one-line substitution.
    anchors = {role: hex_to_linear(layout.role(role)) for role in ANCHORED_ROLES}

    samples: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for key, grid in faces.items():
        flat = grid.reshape(-1, 3)
        samples.append(flat)
        weight = FACE_SAMPLE_WEIGHT if key == "head_front" else 1.0
        weights.append(np.full(flat.shape[0], weight, dtype=np.float32))

    if samples:
        palette = build_palette(
            np.concatenate(samples),
            config.palette_size,
            weights=np.concatenate(weights),
            anchors=anchors,
        )
    else:
        palette = build_palette(np.zeros((0, 3), np.float32), config.palette_size, anchors=anchors)

    # ── faces the photo cannot show ───────────────────────────────────
    _derive_head_faces(
        faces,
        regions,
        palette.color_of("hair_color"),
        palette.color_of("skin_tone"),
        config.bake_orientation_shading,
    )
    _render_procedural_faces(faces, regions, palette, config.bake_orientation_shading)

    # ── quantize everything to the shared palette ─────────────────────
    palette_hex = palette.hex_list()
    pixel_data: dict[str, list[list[str]]] = {}
    quantized: dict[str, np.ndarray] = {}
    eye_index = palette.index_of("eye_color")
    for key, grid in faces.items():
        indices = assign(grid, palette)
        if key == "head_front" and eye_mask is not None and eye_index is not None:
            indices[eye_mask] = eye_index
        quantized[key] = palette.linear[indices]
        pixel_data[key] = [[palette_hex[i] for i in row] for row in indices]

    metrics = {
        key: compare(sources[key], quantized[key])
        for key in SCORED_FACES
        if key in sources and key in quantized
    }

    return RenderResult(
        pixel_data=pixel_data,
        palette=palette_hex,
        roles=palette.roles,
        metrics=metrics,
    )


def recolor(
    pixel_data: dict[str, list[list[str]]], old_hex: str, new_hex: str
) -> dict[str, list[list[str]]]:
    """Swap every occurrence of one palette color — the cheap edit path.

    Because the whole skin is drawn from one palette, changing a shirt color
    costs no API call and no re-render: it is a string substitution.
    """
    old = old_hex.upper()
    new = linear_to_hex(hex_to_linear(new_hex))
    return {
        key: [[new if c.upper() == old else c for c in row] for row in grid]
        for key, grid in pixel_data.items()
    }
