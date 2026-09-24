"""Combined-orbit framing checks, no renders (pure geometry only).

Rebuilds the default three-board scene from boards.layout_positions at
15 cm squares (the resized-in-GUI case the layout is spaced for), then:

  1. FIT_HALF_FOV_X/Y must sit below the device file's KB4 half fields
     (atan(half_sensor / f) per camera, DP180IP-30020104.json - read, not
     guessed; the fisheye only widens the true field).
  2. From EVERY eye eye_positions() produces with the min_fit_radius clamp,
     every corner of every board must fall inside the device half fields at
     zero aim (per-axis angles in the camera frame).
  3. Single-board scenes must be untouched: build_orbit_trajectory only
     passes a min_radius for multi-board scenes, and with min_radius=None
     eye_positions must place the rings at exactly width * DISTANCE_FACTORS
     (the old behaviour; the old near ring deliberately lets corners spill
     for close-up coverage, so no fit clamp may sneak in).

Run:  python tests/test_orbit_framing.py
"""
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# The orbit module reads these at call time; the test must see defaults.
for var in ("ORBIT_DIST", "ARM_REACH", "ORBIT_TARGET", "ORBIT_FLIP",
            "ORBIT_CAMS"):
    os.environ.pop(var, None)

from threedgrut_playground.utils.boards import (
    DEFAULT_SCENE, board_half_extents, layout_positions)
from threedgrut_playground.utils.orbit_trajectory import (
    DISTANCE_FACTORS, FIT_HALF_FOV_X_DEG, FIT_HALF_FOV_Y_DEG, eye_positions,
    frame_from, min_fit_radius)

CALIB = os.path.join(REPO, "calibration_files", "DP180IP-30020104.json")
TAG_CM = 5.173
SQ_CM = 15.0


def kb4_half_fovs():
    """Per-axis half fields of the device's KB4 cameras, degrees."""
    with open(CALIB) as f:
        data = json.load(f)
    half_h, half_v = [], []
    for _sock, cam in data["cameraData"]:
        if cam["cameraType"] != 1:
            continue
        K = cam["intrinsicMatrix"]
        half_h.append(math.degrees(math.atan(cam["width"] / 2.0 / K[0][0])))
        half_v.append(math.degrees(math.atan(cam["height"] / 2.0 / K[1][1])))
    return min(half_h), min(half_v)


def scene_corners(square_cm):
    """Corner cloud of the default scene's boards at the given square size."""
    placed = layout_positions(DEFAULT_SCENE, TAG_CM, 0.20 * TAG_CM, 0.6)
    pts = []
    for name, pos in placed:
        hx, hy = board_half_extents(name, square_cm)
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append([pos[0] + sx * hx, pos[1] + sy * hy, pos[2]])
    return np.array(pts)


def main():
    dev_h, dev_v = kb4_half_fovs()
    assert FIT_HALF_FOV_X_DEG < dev_h and FIT_HALF_FOV_Y_DEG < dev_v, \
        f"fit bounds ({FIT_HALF_FOV_X_DEG}, {FIT_HALF_FOV_Y_DEG}) not " \
        f"below device KB4 half fields ({dev_h:.1f}, {dev_v:.1f})"
    print(f"PASS  fit bounds ({FIT_HALF_FOV_X_DEG:.0f}, "
          f"{FIT_HALF_FOV_Y_DEG:.0f}) deg below device KB4 half fields "
          f"({dev_h:.1f}, {dev_v:.1f}) deg")

    corners = scene_corners(SQ_CM)
    mn, mx = corners.min(axis=0), corners.max(axis=0)
    center = (mn + mx) / 2.0
    width = float((mx - mn).max())
    normal = np.array([0.0, 0.0, 1.0])  # flat boards facing the rig side

    r_fit = min_fit_radius(center, normal, corners,
                           start=width * min(DISTANCE_FACTORS))
    eyes = eye_positions(center, normal, width, r_fit)
    assert len(eyes) == len(DISTANCE_FACTORS) * 2 * 3

    worst_h = worst_v = 0.0
    for eye in eyes:
        view = center - eye
        view = view / np.linalg.norm(view)
        right, up = frame_from(view)
        rel = corners - eye
        fwd = rel @ view
        assert (fwd > 0).all(), "board corner behind an eye"
        worst_h = max(worst_h, np.degrees(
            np.arctan2(np.abs(rel @ right), fwd)).max())
        worst_v = max(worst_v, np.degrees(
            np.arctan2(np.abs(rel @ up), fwd)).max())
    assert worst_h <= dev_h and worst_v <= dev_v, \
        f"corners leave the KB4 view: worst ({worst_h:.1f}, {worst_v:.1f})" \
        f" vs device ({dev_h:.1f}, {dev_v:.1f})"
    print(f"PASS  all {len(corners)} corners x {len(eyes)} eyes inside the "
          f"KB4 view: worst ({worst_h:.1f}, {worst_v:.1f}) deg, fit radius "
          f"{r_fit:.2f} m")

    # 3. single-board path: min_radius=None must give exactly the old rings
    for sq in (TAG_CM, SQ_CM):
        hx, hy = board_half_extents("aprilgrid", sq)
        w1 = 2.0 * max(hx, hy)
        eyes1 = eye_positions(np.zeros(3), normal, w1, None)
        radii = sorted({round(float(np.linalg.norm(e)), 6) for e in eyes1})
        want = sorted(round(w1 * f, 6) for f in DISTANCE_FACTORS)
        assert radii == want, f"single-board rings moved: {radii} != {want}"
    print(f"PASS  single 4x7 board rings untouched at {TAG_CM} and "
          f"{SQ_CM:.0f} cm squares (width x {DISTANCE_FACTORS})")

    print("\n3 checks passed")


if __name__ == "__main__":
    main()
