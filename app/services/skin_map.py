"""Minecraft skin UV coordinate map for 64×64 textures.

Every body part face is defined as (x, y, width, height) on the texture.
Supports both Classic (Steve, 4px arms) and Slim (Alex, 3px arms) models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelType = Literal["classic", "slim"]


@dataclass(frozen=True)
class FaceRect:
    """A rectangular region on the 64×64 skin texture."""

    x: int
    y: int
    w: int
    h: int


# ── Base Layer ────────────────────────────────────────────

HEAD: dict[str, FaceRect] = {
    "top": FaceRect(8, 0, 8, 8),
    "bottom": FaceRect(16, 0, 8, 8),
    "right": FaceRect(0, 8, 8, 8),
    "front": FaceRect(8, 8, 8, 8),
    "left": FaceRect(16, 8, 8, 8),
    "back": FaceRect(24, 8, 8, 8),
}

BODY: dict[str, FaceRect] = {
    "top": FaceRect(20, 16, 8, 4),
    "bottom": FaceRect(28, 16, 8, 4),
    "right": FaceRect(16, 20, 4, 12),
    "front": FaceRect(20, 20, 8, 12),
    "left": FaceRect(28, 20, 4, 12),
    "back": FaceRect(32, 20, 8, 12),
}

RIGHT_ARM_CLASSIC: dict[str, FaceRect] = {
    "top": FaceRect(44, 16, 4, 4),
    "bottom": FaceRect(48, 16, 4, 4),
    "right": FaceRect(40, 20, 4, 12),
    "front": FaceRect(44, 20, 4, 12),
    "left": FaceRect(48, 20, 4, 12),
    "back": FaceRect(52, 20, 4, 12),
}

RIGHT_ARM_SLIM: dict[str, FaceRect] = {
    "top": FaceRect(44, 16, 3, 4),
    "bottom": FaceRect(47, 16, 3, 4),
    "right": FaceRect(40, 20, 4, 12),
    "front": FaceRect(44, 20, 3, 12),
    "left": FaceRect(47, 20, 4, 12),
    "back": FaceRect(51, 20, 3, 12),
}

LEFT_ARM_CLASSIC: dict[str, FaceRect] = {
    "top": FaceRect(36, 48, 4, 4),
    "bottom": FaceRect(40, 48, 4, 4),
    "right": FaceRect(32, 52, 4, 12),
    "front": FaceRect(36, 52, 4, 12),
    "left": FaceRect(40, 52, 4, 12),
    "back": FaceRect(44, 52, 4, 12),
}

LEFT_ARM_SLIM: dict[str, FaceRect] = {
    "top": FaceRect(36, 48, 3, 4),
    "bottom": FaceRect(39, 48, 3, 4),
    "right": FaceRect(32, 52, 4, 12),
    "front": FaceRect(36, 52, 3, 12),
    "left": FaceRect(39, 52, 4, 12),
    "back": FaceRect(43, 52, 3, 12),
}

RIGHT_LEG: dict[str, FaceRect] = {
    "top": FaceRect(4, 16, 4, 4),
    "bottom": FaceRect(8, 16, 4, 4),
    "right": FaceRect(0, 20, 4, 12),
    "front": FaceRect(4, 20, 4, 12),
    "left": FaceRect(8, 20, 4, 12),
    "back": FaceRect(12, 20, 4, 12),
}

LEFT_LEG: dict[str, FaceRect] = {
    "top": FaceRect(20, 48, 4, 4),
    "bottom": FaceRect(24, 48, 4, 4),
    "right": FaceRect(16, 52, 4, 12),
    "front": FaceRect(20, 52, 4, 12),
    "left": FaceRect(24, 52, 4, 12),
    "back": FaceRect(28, 52, 4, 12),
}

# ── Overlay Layer (Hat / Jacket / Sleeves / Pants) ────────

HEAD_OVERLAY: dict[str, FaceRect] = {
    "top": FaceRect(40, 0, 8, 8),
    "bottom": FaceRect(48, 0, 8, 8),
    "right": FaceRect(32, 8, 8, 8),
    "front": FaceRect(40, 8, 8, 8),
    "left": FaceRect(48, 8, 8, 8),
    "back": FaceRect(56, 8, 8, 8),
}

BODY_OVERLAY: dict[str, FaceRect] = {
    "top": FaceRect(20, 32, 8, 4),
    "bottom": FaceRect(28, 32, 8, 4),
    "right": FaceRect(16, 36, 4, 12),
    "front": FaceRect(20, 36, 8, 12),
    "left": FaceRect(28, 36, 4, 12),
    "back": FaceRect(32, 36, 8, 12),
}

RIGHT_ARM_OVERLAY_CLASSIC: dict[str, FaceRect] = {
    "top": FaceRect(44, 32, 4, 4),
    "bottom": FaceRect(48, 32, 4, 4),
    "right": FaceRect(40, 36, 4, 12),
    "front": FaceRect(44, 36, 4, 12),
    "left": FaceRect(48, 36, 4, 12),
    "back": FaceRect(52, 36, 4, 12),
}

RIGHT_ARM_OVERLAY_SLIM: dict[str, FaceRect] = {
    "top": FaceRect(44, 32, 3, 4),
    "bottom": FaceRect(47, 32, 3, 4),
    "right": FaceRect(40, 36, 4, 12),
    "front": FaceRect(44, 36, 3, 12),
    "left": FaceRect(47, 36, 4, 12),
    "back": FaceRect(51, 36, 3, 12),
}

LEFT_ARM_OVERLAY_CLASSIC: dict[str, FaceRect] = {
    "top": FaceRect(52, 48, 4, 4),
    "bottom": FaceRect(56, 48, 4, 4),
    "right": FaceRect(48, 52, 4, 12),
    "front": FaceRect(52, 52, 4, 12),
    "left": FaceRect(56, 52, 4, 12),
    "back": FaceRect(60, 52, 4, 12),
}

LEFT_ARM_OVERLAY_SLIM: dict[str, FaceRect] = {
    "top": FaceRect(52, 48, 3, 4),
    "bottom": FaceRect(55, 48, 3, 4),
    "right": FaceRect(48, 52, 4, 12),
    "front": FaceRect(52, 52, 3, 12),
    "left": FaceRect(55, 52, 4, 12),
    "back": FaceRect(59, 52, 3, 12),
}

RIGHT_LEG_OVERLAY: dict[str, FaceRect] = {
    "top": FaceRect(4, 32, 4, 4),
    "bottom": FaceRect(8, 32, 4, 4),
    "right": FaceRect(0, 36, 4, 12),
    "front": FaceRect(4, 36, 4, 12),
    "left": FaceRect(8, 36, 4, 12),
    "back": FaceRect(12, 36, 4, 12),
}

LEFT_LEG_OVERLAY: dict[str, FaceRect] = {
    "top": FaceRect(4, 48, 4, 4),
    "bottom": FaceRect(8, 48, 4, 4),
    "right": FaceRect(0, 52, 4, 12),
    "front": FaceRect(4, 52, 4, 12),
    "left": FaceRect(8, 52, 4, 12),
    "back": FaceRect(12, 52, 4, 12),
}


def get_all_regions(model: ModelType) -> dict[str, dict[str, FaceRect]]:
    """Return all UV regions for the given model type."""
    if model == "classic":
        right_arm = RIGHT_ARM_CLASSIC
        left_arm = LEFT_ARM_CLASSIC
        right_arm_overlay = RIGHT_ARM_OVERLAY_CLASSIC
        left_arm_overlay = LEFT_ARM_OVERLAY_CLASSIC
    else:
        right_arm = RIGHT_ARM_SLIM
        left_arm = LEFT_ARM_SLIM
        right_arm_overlay = RIGHT_ARM_OVERLAY_SLIM
        left_arm_overlay = LEFT_ARM_OVERLAY_SLIM

    return {
        "head": HEAD,
        "body": BODY,
        "right_arm": right_arm,
        "left_arm": left_arm,
        "right_leg": RIGHT_LEG,
        "left_leg": LEFT_LEG,
        "head_overlay": HEAD_OVERLAY,
        "body_overlay": BODY_OVERLAY,
        "right_arm_overlay": right_arm_overlay,
        "left_arm_overlay": left_arm_overlay,
        "right_leg_overlay": RIGHT_LEG_OVERLAY,
        "left_leg_overlay": LEFT_LEG_OVERLAY,
    }


# Mapping from Claude output keys to (region_group, face_name)
PIXEL_KEY_MAP: dict[str, tuple[str, str]] = {}

for _part in ["head", "body", "right_arm", "left_arm", "right_leg", "left_leg"]:
    for _face in ["front", "back", "top", "bottom", "left", "right"]:
        PIXEL_KEY_MAP[f"{_part}_{_face}"] = (_part, _face)

# Overlay keys use "hat_" prefix for head, "{part}_overlay_" for others
for _face in ["front", "back", "top", "bottom", "left", "right"]:
    PIXEL_KEY_MAP[f"hat_{_face}"] = ("head_overlay", _face)

for _part in ["body", "right_arm", "left_arm", "right_leg", "left_leg"]:
    for _face in ["front", "back", "top", "bottom", "left", "right"]:
        PIXEL_KEY_MAP[f"{_part}_overlay_{_face}"] = (f"{_part}_overlay", _face)


# ── Palette roles (used by claude_vision.py and procedural.py) ───

# Fixed positions in the AI's `palette` output. Indices 10+ are freeform,
# chosen by the model for logos/patterns/accessories on optional detail
# faces. Keeping these positions fixed (rather than letting the model
# choose) is what lets a conversational color edit retint by swapping one
# array entry — see docs/superpowers/specs/2026-07-09-skin-fidelity-and-color-edit-design.md.
PALETTE_ROLES: dict[str, int] = {
    "skin_tone": 0,
    "hair_color": 1,
    "eye_color": 2,
    "shirt_main": 3,
    "shirt_shadow": 4,
    "arm_main": 5,
    "arm_shadow": 6,
    "pants_main": 7,
    "pants_shadow": 8,
    "shoe_color": 9,
}
MIN_PALETTE_SIZE = 10
MAX_PALETTE_SIZE = 16
