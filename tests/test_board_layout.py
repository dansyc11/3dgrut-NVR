"""Board layout invariants, no renders (boards.py geometry helpers only).

Checks:
  1. aprilgrid is first in BOARD_SPECS (material id 0 = default-quad board)
     and names are unique, so build_materials assigns dense ids 0..n-1.
  2. The default scene and the all-boards scene stay clear of each other
     face-on at LAYOUT_SQUARE_CM (15 cm) squares, with the promised
     VERTICAL_GAP_FRACTION gap between neighbours.
  3. The same layouts are clear at the 5.173 cm spawn size.
  4. Regression: the OLD angular layout (-22/0/+22 deg) does overlap at
     15 cm squares - the reason the layout changed.

Run:  python tests/test_board_layout.py
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from threedgrut_playground.utils.boards import (
    BOARD_SPECS, DEFAULT_SCENE, LAYOUT_SQUARE_CM, VERTICAL_GAP_FRACTION,
    board_aabbs, board_size_m, check_layout_no_overlap, layout_positions,
    vertical_step)

TAG_CM = 5.173
NEAR = 0.20 * TAG_CM
SPAN = 0.6
ALL_BOARDS = [s[0] for s in BOARD_SPECS]


def overlapping_pairs(aabbs):
    pairs = []
    for i in range(len(aabbs)):
        for j in range(i + 1, len(aabbs)):
            (na, amin, amax), (nb, bmin, bmax) = aabbs[i], aabbs[j]
            if all(amin[k] < bmax[k] and bmin[k] < amax[k] for k in range(2)):
                pairs.append((na, nb))
    return pairs


def main():
    n_pass = 0

    # 1. spec invariants
    assert BOARD_SPECS[0][0] == "aprilgrid", \
        f"aprilgrid must be first, got {BOARD_SPECS[0][0]}"
    names = [s[0] for s in BOARD_SPECS]
    assert len(set(names)) == len(names), "duplicate board names"
    print(f"PASS  aprilgrid first, {len(names)} unique specs -> dense ids")
    n_pass += 1

    # 2. new layout clear at 15 cm, with the promised gap
    for scene in (DEFAULT_SCENE, ALL_BOARDS):
        aabbs = check_layout_no_overlap(scene, tag_cm=TAG_CM,
                                        square_cm=LAYOUT_SQUARE_CM)
        ups = sorted((a[1][1], a[2][1], a[0]) for a in aabbs)
        min_gap = min(ups[i + 1][0] - ups[i][1] for i in range(len(ups) - 1))
        tallest = max(board_size_m(nm, LAYOUT_SQUARE_CM)[1] for nm in scene)
        want_gap = VERTICAL_GAP_FRACTION * tallest
        assert min_gap >= want_gap - 1e-9, \
            f"gap {min_gap:.3f} < promised {want_gap:.3f}"
        print(f"PASS  {len(scene)} boards clear at {LAYOUT_SQUARE_CM:.0f} cm "
              f"squares, min gap {min_gap:.3f} m (>= {want_gap:.3f})")
        n_pass += 1

    # 3. also clear at the spawn size
    for scene in (DEFAULT_SCENE, ALL_BOARDS):
        check_layout_no_overlap(scene, tag_cm=TAG_CM, square_cm=TAG_CM)
    print(f"PASS  both scenes clear at the {TAG_CM} cm spawn size")
    n_pass += 1

    # 4. the old angular layout overlaps at 15 cm (why it was replaced)
    n = len(DEFAULT_SCENE)
    old = []
    for i, name in enumerate(DEFAULT_SCENE):
        ang = -22.0 + 44.0 * i / (n - 1)
        dist = NEAR * (1.0 + SPAN * i / (n - 1))
        old.append((name, [0.0, dist * math.tan(math.radians(ang)), -dist]))
    bad = overlapping_pairs(board_aabbs(old, LAYOUT_SQUARE_CM))
    assert bad, "expected the old angular layout to overlap at 15 cm"
    print(f"PASS  old angular layout overlaps at 15 cm as expected: {bad}")
    n_pass += 1

    step = vertical_step(DEFAULT_SCENE)
    print(f"\nvertical step for the default scene: {step:.3f} m "
          f"(tallest board {max(board_size_m(nm, LAYOUT_SQUARE_CM)[1] for nm in DEFAULT_SCENE):.3f} m"
          f" x {1 + VERTICAL_GAP_FRACTION:.1f})")
    print("positions at defaults:")
    for name, pos in layout_positions(DEFAULT_SCENE, TAG_CM, NEAR, SPAN):
        print(f"  {name:<12} ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    print(f"\n{n_pass} checks passed")


if __name__ == "__main__":
    main()
