from __future__ import annotations

from app.imaging.color import hex_to_linear, linear_to_oklab
from app.services.layout import Bbox, default_layout
from app.services.renderer import (
    ANCHORED_ROLES,
    RenderConfig,
    expand_to_aspect,
    recolor,
    render,
)
from app.services.skin_map import PIXEL_KEY_MAP, get_all_regions

BASE_GROUPS = ("head", "body", "right_arm", "left_arm", "right_leg", "left_leg")


def _lightness(hex_color: str) -> float:
    return float(linear_to_oklab(hex_to_linear(hex_color).reshape(1, 3))[0, 0])


def test_every_base_region_is_rendered_at_the_right_size(portrait_bytes):
    result = render(portrait_bytes, default_layout(), "classic")
    regions = get_all_regions("classic")

    for group in BASE_GROUPS:
        for face, rect in regions[group].items():
            grid = result.pixel_data[f"{group}_{face}"]
            assert len(grid) == rect.h
            assert all(len(row) == rect.w for row in grid)

    assert len(result.pixel_data) == 36
    assert all(key in PIXEL_KEY_MAP for key in result.pixel_data)


def test_one_shared_palette_covers_the_whole_skin(portrait_bytes):
    """The invariant that makes recoloring a substitution instead of a re-render."""
    result = render(portrait_bytes, default_layout(), "classic")
    palette = set(result.palette)
    used = {c for grid in result.pixel_data.values() for row in grid for c in row}
    assert used <= palette


def test_semantic_roles_hold_fixed_palette_slots(portrait_bytes):
    result = render(portrait_bytes, default_layout(), "classic")
    assert result.roles == {role: i for i, role in enumerate(ANCHORED_ROLES)}
    assert len(result.palette) == RenderConfig().palette_size


def test_the_face_is_derived_from_the_photo_not_invented(portrait_bytes):
    """Hair band on top, skin below — structure that only the photo can supply."""
    head = render(portrait_bytes, default_layout(), "classic").pixel_data["head_front"]
    top_row = sum(_lightness(c) for c in head[0]) / len(head[0])
    mid_row = sum(_lightness(c) for c in head[4]) / len(head[4])
    assert top_row < mid_row - 0.15


def test_a_hard_edge_survives_instead_of_blending(portrait_bytes):
    """A red logo on a blue shirt must not average into purple."""
    body = render(portrait_bytes, default_layout(), "classic").pixel_data["body_front"]
    center = body[6][4]
    r, g, b = (int(center[i : i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and g < 90 and b < 90


def test_metrics_are_reported_for_the_scored_faces(portrait_bytes):
    metrics = render(portrait_bytes, default_layout(), "classic").metrics
    assert set(metrics) == {"head_front", "body_front"}
    for scores in metrics.values():
        assert 0.0 <= scores["ssim"] <= 1.0
        assert scores["delta_e_mean"] >= 0.0
        assert scores["detail"] > 0.0


def test_slim_model_narrows_the_arms(portrait_bytes):
    result = render(portrait_bytes, default_layout(), "slim")
    assert len(result.pixel_data["right_arm_front"][0]) == 3
    assert len(result.pixel_data["right_arm_right"][0]) == 4


def test_rendering_is_deterministic(portrait_bytes):
    layout = default_layout()
    a = render(portrait_bytes, layout, "classic")
    b = render(portrait_bytes, layout, "classic")
    assert a.pixel_data == b.pixel_data
    assert a.palette == b.palette


def test_recolor_swaps_one_color_everywhere(portrait_bytes):
    result = render(portrait_bytes, default_layout(), "classic")
    target = result.pixel_data["body_front"][0][0]
    swapped = recolor(result.pixel_data, target, "#FF00FF")

    flat_before = [c for g in result.pixel_data.values() for row in g for c in row]
    flat_after = [c for g in swapped.values() for row in g for c in row]
    assert flat_after.count("#FF00FF") == flat_before.count(target)
    assert target not in flat_after


def test_downscale_method_is_chosen_per_face():
    config = RenderConfig()
    assert config.method_for("head_front") == config.face_method
    assert config.method_for("body_front") == config.material_method


def test_eye_band_lands_on_the_eyes_and_uses_the_eye_color(portrait_bytes):
    """Ground truth: the fixture's eyes are at known coordinates."""
    layout = default_layout()
    layout.boxes["eyes"] = Bbox(0.385, 0.210, 0.615, 0.250)
    layout.roles["eye_color"] = "#101820"

    result = render(portrait_bytes, layout, "classic")
    head = result.pixel_data["head_front"]
    eye_hex = result.palette[result.roles["eye_color"]]

    marked = {(r, c) for r, row in enumerate(head) for c, v in enumerate(row) if v == eye_hex}
    assert marked, "the eye band produced no eye cells at all"
    # Every marked cell sits inside the reported band, and the band is not
    # painted solid — some of it stays skin.
    assert all(2 <= r <= 5 for r, _ in marked)
    assert len(marked) < 8 * 8 // 4


def test_eye_band_is_skipped_when_no_eyes_box_is_reported(portrait_bytes):
    layout = default_layout()
    layout.boxes.pop("eyes", None)
    layout.roles["eye_color"] = "#101820"

    head = render(portrait_bytes, layout, "classic").pixel_data["head_front"]
    assert not any("#101820" == v for row in head for v in row)


def test_eye_strength_zero_disables_the_override(portrait_bytes):
    layout = default_layout()
    layout.boxes["eyes"] = Bbox(0.385, 0.210, 0.615, 0.250)
    with_eyes = render(portrait_bytes, layout, "classic", RenderConfig())
    without = render(portrait_bytes, layout, "classic", RenderConfig(eye_strength=0.0))
    assert with_eyes.pixel_data["head_front"] != without.pixel_data["head_front"]


def test_expand_to_aspect_grows_rather_than_trims():
    tall = Bbox(0.4, 0.1, 0.6, 0.5)  # 0.2 × 0.4, needs to become square
    grown = expand_to_aspect(tall, 8, 8)
    assert grown.y0 == tall.y0 and grown.y1 == tall.y1  # height untouched
    assert grown.x1 - grown.x0 > tall.x1 - tall.x0


def test_expand_to_aspect_stays_inside_the_image():
    edge = Bbox(0.0, 0.1, 0.15, 0.6)
    grown = expand_to_aspect(edge, 8, 8)
    assert grown.x0 >= 0.0 and grown.x1 <= 1.0
