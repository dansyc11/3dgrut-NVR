"""VIO trajectory generator checks, no renders (pure geometry only).

Scenes: (a) one aprilgrid board at 30 cm squares (the doc's recommended
VIO render), (b) the default respaced three-board scene at 15 cm squares.
For each:
  1. verify_path passes: boards inside the reference camera's image on
     every frame (exact DS projection from DP180IP-30020104.json), all
     four quadrants + both distance bands covered on the driver's KB4
     cameras (camB, camC; camA points away and is not consumed), and the
     per-frame pixel displacement (formula AND measured) under
     TRACKABLE_PX at the default 20 fps.
  2. The stationary lead is truly stationary: every frame before
     STATIONARY_LEAD_S is bit-identical to frame 0.
  3. C2 continuity: finite-difference jerk (translation) and angular
     jerk, sampled at 100 Hz, stay bounded, and the window around the
     stationary->motion boundary is no worse than the rest of the path.
  4. The ground-truth npz has ate_compare's layout (t_ns, p, q_xyzw),
     int64 monotonic stamps matching the mcap stamp formula, unit
     quaternions - and loads through ate_compare.load_npz itself when
     ~/vio_offline/ate_compare.py is present.

Run:  python tests/test_vio_trajectory.py
"""
import math
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

for var in ("VIO_TRAJ", "STATIONARY_LEAD_S", "VIO_MOTION_S", "VIO_FPS",
            "TRACKABLE_PX", "VIO_TRAJ_NPZ", "ORBIT_DIST", "ORBIT_TARGET",
            "ORBIT_FLIP", "ARM_REACH"):
    os.environ.pop(var, None)

from threedgrut_playground.utils.boards import (
    DEFAULT_SCENE, board_half_extents, layout_positions)
from threedgrut_playground.utils import vio_trajectory as vt

CALIB = os.path.join(REPO, "calibration_files", "DP180IP-30020104.json")


def quad_corners(center, hx, hy):
    return np.array([[center[0] + sx * hx, center[1] + sy * hy, center[2]]
                     for sx in (-1, 1) for sy in (-1, 1)])


def scene_single():
    """One 4x7 board, 30 cm squares, drawn at z = -2.75 facing +z."""
    hx, hy = board_half_extents("aprilgrid", 30.0)
    cloud = quad_corners([0.0, 0.0, -2.75], hx, hy)
    return cloud, np.array([0.0, 0.0, -2.75]), np.array([0.0, 0.0, 1.0])


def scene_default():
    """The respaced three-board scene at 15 cm squares (drawn +1.25 in z)."""
    placed = layout_positions(DEFAULT_SCENE, 5.173, 0.2 * 5.173, 0.6)
    pts = []
    for name, pos in placed:
        hx, hy = board_half_extents(name, 15.0)
        pts.append(quad_corners([pos[0], pos[1], pos[2] + 1.25], hx, hy))
    cloud = np.concatenate(pts)
    mn, mx = cloud.min(0), cloud.max(0)
    return cloud, (mn + mx) / 2.0, np.array([0.0, 0.0, 1.0])


def run_scene(tag, cloud, center, normal, rig, ref):
    cfg = vt.VioConfig()
    geom = vt.PathGeometry(cloud, center, normal, rig[ref])
    print(f"\n=== {tag}: bands {geom.d_near:.2f}..{geom.d_far:.2f} m, "
          f"aim budget ({geom.bud_x:.0f}, {geom.bud_y:.0f}) deg")
    times, eyes, R_wc = vt.sample_path(cfg, geom)
    metrics = vt.verify_path(cfg, geom, rig, ref, times, eyes, R_wc)
    print(f"PASS  {tag}: verify_path (in-view, quadrants, bands, "
          f"trackable px {metrics['peak']['px_formula']:.1f}/"
          f"{metrics['peak']['px_measured']:.1f} <= {cfg.trackable_px:.0f})")

    # 2. stationary lead truly stationary
    lead_n = int(math.floor(cfg.lead_s * cfg.fps)) + 1
    assert np.array_equal(eyes[:lead_n], np.repeat(eyes[:1], lead_n, 0)), \
        "positions move during the stationary lead"
    assert np.array_equal(R_wc[:lead_n], np.repeat(R_wc[:1], lead_n, 0)), \
        "orientation moves during the stationary lead"
    assert not np.allclose(eyes[lead_n + int(5 * cfg.fps)], eyes[0]), \
        "path never leaves the initial pose"
    print(f"PASS  {tag}: {lead_n} lead frames bit-identical to frame 0")

    # 3. C2 continuity at 100 Hz: jerk bounded, boundary not an outlier
    hz = 100.0
    tt = np.arange(int(cfg.total_s * hz)) / hz
    _, e2, R2 = vt.sample_path(cfg, geom, times=tt)
    dtf = 1.0 / hz
    acc = np.diff(e2, 2, axis=0) / dtf ** 2
    jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1) / dtf
    wv = np.array([vt._rotvec(R2[i], R2[i + 1]) / dtf
                   for i in range(len(R2) - 1)])
    aacc = np.diff(wv, axis=0) / dtf
    ajerk = np.linalg.norm(np.diff(aacc, axis=0), axis=1) / dtf
    b = int(cfg.lead_s * hz)
    win = slice(max(b - 10, 0), b + 10)
    rest = np.r_[jerk[:max(b - 10, 0)], jerk[b + 10:]]
    arest = np.r_[ajerk[:max(b - 10, 0)], ajerk[b + 10:]]
    assert jerk.max() < 2.0, f"translational jerk {jerk.max():.2f} m/s^3"
    assert ajerk.max() < 2.0, f"angular jerk {ajerk.max():.2f} rad/s^3"
    assert jerk[win].max() <= max(rest.max(), 1e-6) * 1.2 + 1e-6, \
        f"jerk spike at the boundary: {jerk[win].max():.3f} vs {rest.max():.3f}"
    assert ajerk[win].max() <= max(arest.max(), 1e-6) * 1.2 + 1e-6, \
        f"angular jerk spike at the boundary: {ajerk[win].max():.3f}"
    print(f"PASS  {tag}: C2 - jerk max {jerk.max():.3f} m/s^3 "
          f"(boundary {jerk[win].max():.3f}), angular {ajerk.max():.3f} "
          f"rad/s^3 (boundary {ajerk[win].max():.3f})")

    # 4. npz layout
    with tempfile.TemporaryDirectory() as d:
        npz = os.path.join(d, "truth.npz")
        t_ns = vt.export_npz(npz, cfg, eyes, R_wc)
        data = np.load(npz, allow_pickle=True)
        assert set(("t_ns", "p", "q_xyzw")) <= set(data.keys())
        assert data["t_ns"].dtype == np.int64
        assert np.all(np.diff(data["t_ns"]) == cfg.interval_ns)
        assert data["t_ns"][0] == vt.START_TIME_NS
        assert np.allclose(np.linalg.norm(data["q_xyzw"], axis=1), 1.0)
        assert len(data["t_ns"]) == cfg.n_frames
        ate = os.path.expanduser("~/vio_offline/ate_compare.py")
        if os.path.exists(ate):
            sys.path.insert(0, os.path.dirname(ate))
            import ate_compare
            t, p, q = ate_compare.load_npz(npz)
            assert len(t) == cfg.n_frames and p.shape == (cfg.n_frames, 3)
            print(f"PASS  {tag}: npz loads through ate_compare.load_npz")
        else:
            print(f"PASS  {tag}: npz layout ok (ate_compare.py not present)")
        assert np.array_equal(t_ns, data["t_ns"])
    return metrics


def main():
    rig, ref = vt.load_rig(CALIB)
    assert rig[ref]["model"] == "ds", "reference camera should be DS here"
    run_scene("single-board 30cm", *scene_single(), rig, ref)
    run_scene("default 3-board 15cm", *scene_default(), rig, ref)
    print("\nall vio trajectory checks passed")


if __name__ == "__main__":
    main()
