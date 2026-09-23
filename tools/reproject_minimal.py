"""Minimal, self-contained reprojection check: one camera, the 4x7 aprilgrid.
Tag detections from a tags mcap vs the true 3D corners (scene JSON) seen
from the trajectory CSV poses through the device calibration (KB4 or Double
Sphere): prints the per-frame and overall median |detected - projected| px.
Teaching copy of tools/reproject_far.py, which adds multi-board scenes,
rotated boards, swaps and cross-camera checks.

--fitted-calib drops the half-pixel shift applied to every projection. Leave
it off for the device file that drove the render: kaolin's rays pass through
pixel i + 0.5 and the detector counts from i, so the true model's projections
sit half a pixel off the detections. Turn it on for vk_calibrate results
(tools/basalt_to_device.py) and any calibration fitted from detections: the
fit already absorbed that half pixel, and the shift would count it twice.
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.append("/opt/vilota/messages")
import capnp; capnp.add_import_hook()
import tagdetection_capnp as T
from mcap.reader import make_reader
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQUARE, GAP = 800, 240        # texture: 800 px squares, 240 px gaps and border
ROWS, COLS = 4, 7             # the aprilgrid board, tag ids 0..27
QUAD_Z_OFF = 1.25             # engine draws the quad at local z = 2.5 * sz(0.5)
APRILGRID_HASH = 7341007258749575218   # gridId of the 4x7 base layout
CAM_NAMES = ["cama", "camb", "camc", "camd"]   # rig index -> topic name
F = np.diag([1.0, -1.0, -1.0, 1.0])            # OpenCV <-> OpenGL axis flip

def board_corners(scene_path):
    """{tag id: (4,3) world corners, texture order TL,TR,BR,BL}."""
    b = [e for e in json.load(open(scene_path))["boards"] if e["material"] == "aprilgrid"][0]
    assert b.get("rot", [0, 0, 0]) == [0, 0, 0], "rotated boards: use reproject_far.py"
    s = b["tag_cm"] / 100.0                      # tag side in metres
    sx = (COLS + 0.3 * (COLS - 1) + 0.6) * s / 2 # board half extents: gaps and
    sy = (ROWS + 0.3 * (ROWS - 1) + 0.6) * s / 2 # border are 0.3 of a square
    W, H = SQUARE * COLS + GAP * (COLS + 1), SQUARE * ROWS + GAP * (ROWS + 1)
    pos = np.array(b["pos"], float)
    out = {}
    for tr in range(ROWS):                       # texture row 0 is the TOP row,
        for c in range(COLS):                    # which holds the LAST id row:
            tid = (ROWS - 1 - tr) * COLS + c     # ids run bottom row first
            x0, y0 = GAP + c * (SQUARE + GAP), GAP + tr * (SQUARE + GAP)
            px = np.array([[x0, y0], [x0 + SQUARE, y0], [x0 + SQUARE, y0 + SQUARE], [x0, y0 + SQUARE]], float)
            out[tid] = np.stack([(2 * px[:, 0] / W - 1) * sx, (2 * px[:, 1] / H - 1) * sy,
                                 np.full(4, QUAD_Z_OFF)], 1) + pos
    return out

def load_cam(calib_path, cam):
    """(project fn, T_D_i) for one camera. Translation is stored in cm."""
    c = dict(json.load(open(calib_path))["cameraData"])[CAM_NAMES.index(cam)]
    T_D_i = np.eye(4)
    rot = c["extrinsics"].get("rotationMatrix") or []   # empty for the reference cam
    if len(rot):
        T_D_i[:3, :3] = np.array(rot, float)
    T_D_i[:3, 3] = np.array([c["extrinsics"]["translation"][k] for k in "xyz"]) / 100.0
    dc = c["distortionCoeff"]
    if c["cameraType"] == 1:                            # KB4 (equidistant fisheye)
        fx, fy = c["intrinsicMatrix"][0][0], c["intrinsicMatrix"][1][1]
        cx, cy = c["intrinsicMatrix"][0][2], c["intrinsicMatrix"][1][2]
        k1, k2, k3, k4 = dc[0:4]
        def project(X):
            x, y, z = X.T
            th = np.arctan2(np.hypot(x, y), z)
            t2 = th * th
            d = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            s = d / np.maximum(np.hypot(x, y), 1e-12)
            return np.stack([fx * s * x + cx, fy * s * y + cy], 1)
    else:                                               # Double Sphere
        fx, fy, cx, cy, xi, alpha = dc[5:11]
        def project(X):
            x, y, z = X.T
            d1 = np.sqrt(x * x + y * y + z * z)
            z1 = xi * d1 + z
            d2 = np.sqrt(x * x + y * y + z1 * z1)
            den = alpha * d2 + (1 - alpha) * z1
            return np.stack([fx * x / den + cx, fy * y / den + cy], 1)
    return project, T_D_i

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", default=os.path.expanduser("~/vilota_results/far500_tags.mcap"))
    ap.add_argument("--traj", default=os.path.expanduser("~/vilota_results/far500_traj.csv"))
    ap.add_argument("--scene", default=os.path.join(REPO, "far_500m.json"))
    ap.add_argument("--calib", default=os.path.join(REPO, "calibration_files/DP180IP-30020104.json"))
    ap.add_argument("--cam", default="camd")
    ap.add_argument("--fitted-calib", action="store_true",
                    help="--calib was fitted from detections (vk_calibrate): no half-pixel shift")
    a = ap.parse_args()
    corners = board_corners(a.scene)
    project, T_D_i = load_cam(a.calib, a.cam)
    poses = [[float(r[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")] for r in csv.DictReader(open(a.traj))]
    per_frame, errs = {}, []
    with open(a.tags, "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages():
            if not ch.topic.endswith(f"/{a.cam}/tags"):
                continue
            with T.TagDetections.from_bytes(msg.data) as m:
                if m.header.seq >= len(poses):
                    continue
                row = poses[m.header.seq]                # CSV row index = header.seq
                Tbw = np.eye(4)                          # body (=CamD) pose in world
                Tbw[:3, :3] = Rot.from_euler("xyz", row[3:6], degrees=True).as_matrix()
                Tbw[:3, 3] = row[:3]
                view = np.linalg.inv(Tbw)                # CamD view, OpenGL axes
                if a.cam != "camd":                      # move_rig_to_view: conjugate
                    view = F @ np.linalg.inv(np.linalg.inv(F @ view @ F) @ T_D_i) @ F
                fe = []
                for t in m.tags:
                    if int(t.gridId) != APRILGRID_HASH or not 0 <= int(t.id) < 28:
                        continue
                    meas = np.array(t.pointsPolygon, float)[:8].reshape(4, 2) \
                        * [m.image.width, m.image.height]   # corners come normalised
                    X = corners[int(t.id)] @ view[:3, :3].T + view[:3, 3]
                    uv = project(X * [1.0, -1.0, -1.0])  # OpenGL -> OpenCV camera axes
                    if not a.fitted_calib:
                        uv -= 0.5   # kaolin rays go through pixel centre i+0.5; the
                                    # detector reports integer-centred pixels.
                    # The detector's corner order is the texture order REVERSED:
                    fe.extend(np.linalg.norm(meas - uv[[3, 2, 1, 0]], axis=1))
                if fe:
                    per_frame[m.header.seq] = np.median(fe)
                    errs.extend(fe)
    for seq in sorted(per_frame):
        print(f"frame {seq:4d}  median {per_frame[seq]:.3f} px")
    print(f"\n{a.cam}: {len(errs)} corners in {len(per_frame)} frames, overall median {np.median(errs):.3f} px")

if __name__ == "__main__":
    main()
