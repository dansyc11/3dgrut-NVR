"""Verify the loader stores camera view matrices in the pipeline convention.

Loads calibration_files/DP180IP-30020104.json headlessly, snapshots each
camera's stored view_matrix, calls move_rig_to_view(identity, origin), and
snapshots again. Since commit e02ce14 the loader stores F inv(E) F — exactly
what move_rig_to_view produces for an identity origin pose — so the two
snapshots must match per camera. Before that commit the loaded matrix was the
raw calibration extrinsic and differed by F inv(.) F, which the candidate
table below makes visible.

No polyscope, no rendering, no GPU. Do not run playground.py for this.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from threedgrut_playground.utils.novel_view_renderer import VilotaDevice

CALIB = os.path.join(REPO_ROOT, "calibration_files", "DP180IP-30020104.json")

np.set_printoptions(precision=5, suppress=True, linewidth=140)


def snapshot(dev, tag):
    mats = {}
    for i, cam in dev.cameras.items():
        m = cam.view_matrix()[0].detach().cpu().numpy()
        mats[i] = m
        print(f"\n[{tag}] cam {i} view_matrix:\n{m}")
    return mats


def main():
    dev = VilotaDevice.load_from_path(CALIB)
    assert dev.loaded, f"device failed to load from {CALIB}"

    origin = dev.get_origin_camera_index()
    print(f"\norigin camera index: {origin}")
    print(f"camera count: {dev.get_camera_count()}")

    before = snapshot(dev, "loaded")
    dev.move_rig_to_view(np.eye(4), origin)
    after = snapshot(dev, "after move_rig_to_view(I, origin)")

    F = np.diag([1.0, -1.0, -1.0, 1.0])
    print("\n=== Frobenius norms of (after - candidate(loaded)) per camera ===")
    print(f"{'cam':>3} {'identity':>12} {'F M F':>12} {'inv(M)':>12} {'F inv(M) F':>12}")
    for i in sorted(before):
        M, A = before[i], after[i]
        cands = [M, F @ M @ F, np.linalg.inv(M), F @ np.linalg.inv(M) @ F]
        errs = [np.linalg.norm(A - c) for c in cands]
        print(f"{i:>3} " + " ".join(f"{e:12.6f}" for e in errs))

    for i in sorted(before):
        assert np.allclose(after[i], before[i], atol=1e-9), (
            f"cam {i}: loaded != post-move, "
            f"|diff| = {np.linalg.norm(after[i] - before[i])}")
    print("\nASSERT PASS: loaded view_matrix == post-move view_matrix "
          "for all cameras")


if __name__ == "__main__":
    main()
