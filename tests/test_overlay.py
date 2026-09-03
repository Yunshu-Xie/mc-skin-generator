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


def test_glasses_are_a_frame_with_see_through_lenses():
    """Filling the lens cells too would read as a blindfold, not spectacles."""
    overlay = build_head_overlay("bald", HAIR, eye_row=4, glasses=GLASSES)
    front = overlay["hat_front"]
    assert front[4][1] == GLASSES and front[4][6] == GLASSES  # frame
    assert front[4][3] == GLASSES and front[4][4] == GLASSES  # bridge
    assert front[4][2] == TRANSPARENT and front[4][5] == TRANSPARENT  # lenses
    assert front[3][2] == TRANSPARENT  # nothing on the neighbouring row
    assert GLASSES in overlay["hat_left"][4]  # temples


def test_glasses_survive_on_a_haired_head():
    overlay = build_head_overlay("long", HAIR, eye_row=5, glasses=GLASSES)
    assert overlay["hat_front"][5][1] == GLASSES


def test_the_overhang_never_covers_the_eyes():
    """A fringe plus an overhang used to draw hair straight over both eyes."""
    for style in ("short", "fringe", "long", "hat"):
        for eye_row in range(2, 6):
            front = build_head_overlay(style, HAIR, eye_row=eye_row)["hat_front"]
            hair_rows = {r for r, row in enumerate(front) if HAIR in row[2:6]}
            assert all(r < eye_row for r in hair_rows), (style, eye_row, hair_rows)


def test_glasses_are_not_buried_under_the_overhang():
    overlay = build_head_overlay("fringe", HAIR, eye_row=3, glasses=GLASSES)
    assert overlay["hat_front"][3][1] == GLASSES
    assert HAIR not in overlay["hat_front"][3]
