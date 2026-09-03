"""The second layer: silhouette and transparency."""

from __future__ import annotations

from app.services.overlay import TRANSPARENT, build_head_overlay

HAIR = "#3A2A1E"
GLASSES = "#101820"


def _cells(grid: list[list[str]]) -> list[str]:
    return [c for row in grid for c in row]


def test_every_head_overlay_face_is_produced_at_full_size():
    overlay = build_head_overlay("short", HAIR)
    assert set(overlay) == {
        "hat_front",
        "hat_back",
        "hat_left",
        "hat_right",
        "hat_top",
        "hat_bottom",
    }
    for grid in overlay.values():
        assert len(grid) == 8 and all(len(row) == 8 for row in grid)


def test_unused_cells_are_transparent_not_black():
    overlay = build_head_overlay("short", HAIR)
    assert TRANSPARENT in _cells(overlay["hat_front"])
    assert set(_cells(overlay["hat_bottom"])) == {TRANSPARENT}


def test_the_overlay_overhangs_the_base_hairline():
    """Two base rows of hair, three on the overlay — that is the fringe."""
    front = build_head_overlay("short", HAIR)["hat_front"]
    hair_rows = {r for r, row in enumerate(front) if HAIR in row}
    assert hair_rows == {0, 1, 2}


def test_long_hair_runs_down_the_sides():
    front = build_head_overlay("long", HAIR)["hat_front"]
    assert front[7][0] == HAIR and front[7][7] == HAIR
    assert front[7][3] == TRANSPARENT


def test_bald_writes_nothing():
    overlay = build_head_overlay("bald", HAIR)
    assert all(set(_cells(g)) == {TRANSPARENT} for g in overlay.values())


def test_glasses_sit_on_the_eye_row_and_wrap_to_the_temples():
    overlay = build_head_overlay("bald", HAIR, eye_row=4, glasses=GLASSES)
    assert overlay["hat_front"][4][2] == GLASSES
    assert overlay["hat_front"][3][2] == TRANSPARENT
    assert GLASSES in overlay["hat_left"][4]


def test_glasses_survive_on_a_haired_head():
    overlay = build_head_overlay("long", HAIR, eye_row=5, glasses=GLASSES)
    assert overlay["hat_front"][5][2] == GLASSES
