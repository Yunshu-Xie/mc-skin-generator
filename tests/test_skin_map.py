"""Tests for skin_map UV coordinates — no overlaps, correct bounds."""

from app.services.skin_map import FaceRect, get_all_regions


def _rects_overlap(a: FaceRect, b: FaceRect) -> bool:
    """Check if two rectangles overlap."""
    if a.x >= b.x + b.w or b.x >= a.x + a.w:
        return False
    if a.y >= b.y + b.h or b.y >= a.y + a.h:
        return False
    return True


def _collect_all_rects(model: str) -> list[tuple[str, FaceRect]]:
    """Flatten all regions into a list of (label, rect) pairs."""
    regions = get_all_regions(model)  # type: ignore[arg-type]
    result = []
    for group_name, faces in regions.items():
        for face_name, rect in faces.items():
            result.append((f"{group_name}.{face_name}", rect))
    return result


def test_no_overlaps_classic():
    """No two faces should occupy the same pixels (classic model)."""
    rects = _collect_all_rects("classic")
    for i, (label_a, rect_a) in enumerate(rects):
        for label_b, rect_b in rects[i + 1 :]:
            assert not _rects_overlap(rect_a, rect_b), (
                f"Overlap between {label_a} and {label_b}: {rect_a} vs {rect_b}"
            )


def test_no_overlaps_slim():
    """No two faces should occupy the same pixels (slim model)."""
    rects = _collect_all_rects("slim")
    for i, (label_a, rect_a) in enumerate(rects):
        for label_b, rect_b in rects[i + 1 :]:
            assert not _rects_overlap(rect_a, rect_b), (
                f"Overlap between {label_a} and {label_b}: {rect_a} vs {rect_b}"
            )


def test_all_within_bounds():
    """Every face rect must fit within 64×64."""
    for model in ("classic", "slim"):
        rects = _collect_all_rects(model)
        for label, rect in rects:
            assert rect.x >= 0 and rect.y >= 0, f"{label}: negative coords {rect}"
            assert rect.x + rect.w <= 64, f"{label}: exceeds width {rect}"
            assert rect.y + rect.h <= 64, f"{label}: exceeds height {rect}"


def test_head_dimensions():
    """Head faces should all be 8×8."""
    regions = get_all_regions("classic")
    for face_name, rect in regions["head"].items():
        assert rect.w == 8 and rect.h == 8, f"head.{face_name}: expected 8x8, got {rect}"


def test_body_front_dimensions():
    """Body front should be 8×12."""
    regions = get_all_regions("classic")
    body_front = regions["body"]["front"]
    assert body_front.w == 8 and body_front.h == 12


def test_classic_arm_width():
    """Classic arm front/back should be 4px wide."""
    regions = get_all_regions("classic")
    assert regions["right_arm"]["front"].w == 4
    assert regions["left_arm"]["front"].w == 4


def test_slim_arm_width():
    """Slim arm front/back should be 3px wide."""
    regions = get_all_regions("slim")
    assert regions["right_arm"]["front"].w == 3
    assert regions["left_arm"]["front"].w == 3


def test_region_count():
    """Should have 12 region groups (6 base + 6 overlay)."""
    regions = get_all_regions("classic")
    assert len(regions) == 12
    for group in regions.values():
        assert len(group) == 6  # 6 faces per body part
