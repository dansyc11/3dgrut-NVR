"""Corner-level reprojection check of a playground render, per camera and board.

    python tools/reproject_far.py [--tags office_v9_tags.mcap] [--traj ...csv]
        [--scene office_scene_v9.json] [--calib calibration_files/....json]
        [--stride 1] [--cams cama,camb,camc,camd]

Every tag's four black-square corners are placed in 3D from the texture
layout (boards.py / create_aprilgrid.py: square 800 px, gap 240 px, padding
240 px on all sides, texture row 0 = top = last solver row) on the engine's
quad (pos + R(rot) (+-sx, +-sy, 2.5 * 0.5), see viz_rerun.board_outline), then
projected per camera exactly as the renderer casts its rays:

    CamD view  = inv(CSV body->world)              polyscope/OpenGL axes
    cam_i view = F inv( inv(F view F) T_D_i ) F     move_rig_to_view
    X_cv       = diag(1,-1,-1) (R_i X + t_i)       raygen axis flip
    KB4 for cameraType 1 (coeff[5] == 0), DS from coeff[5:11] otherwise

Residual = detected corner - projected corner [px], matched by tag id with a
per-board cyclic corner-order shift (detector canonical order vs texture
TL,TR,BR,BL) chosen by minimum median. The CSV row index is header.seq.
CPU only.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/opt/vilota/messages")
sys.path.insert(0, REPO)
import capnp  # noqa: E402
capnp.add_import_hook()
import tagdetection_capnp as T  # noqa: E402
from mcap.reader import make_reader  # noqa: E402
from threedgrut_playground.utils.boards import BOARD_SPECS, row_major_to_texture_order  # noqa: E402

GRID_NAMES = {10534446874174779866: "grid_3x1", 2537504867360144137: "grid_2x2",
              7341007258749575218: "aprilgrid", 4722288866674018535: "grid_off14",
              9771855239151313725: "grid_off15", 12633434617871127365: "grid_off16"}
CAM_NAMES = ["cama", "camb", "camc", "camd"]     # rig index -> topic name
F = np.diag([1.0, -1.0, -1.0, 1.0])
SQUARE, GAP = 800, 240          # boards.py build_materials(square=800), gap 0.3
QUAD_Z_OFF = 2.5 * 0.5          # mesh_io MZ * autoscale sz

_spec = importlib.util.spec_from_file_location("viz_rerun", os.path.join(REPO, "viz_rerun.py"))
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)


# ---------------------------------------------------------------- cameras
class KB4:
    def __init__(self, K, dc, w, h):
        self.fx, self.fy, self.cx, self.cy = K[0][0], K[1][1], K[0][2], K[1][2]
        self.k = dc[0:4]
        self.w, self.h = w, h

    def project(self, X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        rho = np.hypot(x, y)
        th = np.arctan2(rho, z)
        t2 = th * th
        k1, k2, k3, k4 = self.k
        d = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        s = np.where(rho > 0, rho, 1.0)
        u = self.fx * d * x / s + self.cx
        v = self.fy * d * y / s + self.cy
        ok = (z > 0) & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        return np.stack([u, v], 1), ok


class DS:
    def __init__(self, dc, w, h):
        self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha = dc[5:11]
        self.w, self.h = w, h

    def project(self, X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        d1 = np.sqrt(x * x + y * y + z * z)
        z1 = self.xi * d1 + z
        d2 = np.sqrt(x * x + y * y + z1 * z1)
        den = self.alpha * d2 + (1 - self.alpha) * z1
        u = self.fx * x / den + self.cx
        v = self.fy * y / den + self.cy
        wv = self.alpha / (1 - self.alpha) if self.alpha <= 0.5 else (1 - self.alpha) / self.alpha
        w2 = (wv + self.xi) / np.sqrt(2 * wv * self.xi + self.xi * self.xi + 1)
        ok = (z > -w2 * d1) & (den > 0) & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        return np.stack([u, v], 1), ok


def load_cams(path):
    """-> {rig index: (model, T_D_i)} with T_D_i cam_i -> reference, t in m."""
    data = json.load(open(path))
    cams = {}
    for idx, c in data["cameraData"]:
        e = c["extrinsics"]
        rot = e.get("rotationMatrix") or []
        R = np.array(rot, float) if len(rot) else np.eye(3)
        t = np.array([e["translation"][k] for k in "xyz"]) / 100.0
        T_D_i = np.eye(4)
        T_D_i[:3, :3], T_D_i[:3, 3] = R, t
        dc = c["distortionCoeff"]
        model = (KB4(c["intrinsicMatrix"], dc, c["width"], c["height"]) if c["cameraType"] == 1
                 else DS(dc, c["width"], c["height"]))
        cams[int(idx)] = (model, T_D_i)
    return cams


def camd_view(row):
    Tbw = np.eye(4)
    Tbw[:3, :3] = Rot.from_euler("xyz", row[3:6], degrees=True).as_matrix()
    Tbw[:3, 3] = row[:3]
    return np.linalg.inv(Tbw)


def cam_view(view_d, T_D_i):
    """novel_view_renderer.move_rig_to_view, lines 490-498."""
    world_to_cam0 = F @ view_d @ F
    cam0_to_world = np.linalg.inv(world_to_cam0)
    cami_to_world = cam0_to_world @ T_D_i
    return F @ np.linalg.inv(cami_to_world) @ F


def project(model, view_i, X):
    Xgl = X @ view_i[:3, :3].T + view_i[:3, 3]
    Xcv = Xgl * np.array([1.0, -1.0, -1.0])
    return model.project(Xcv)


# ----------------------------------------------------------------- boards
def board_corners_3d(entry):
    """{tag id: (4,3) world corners in texture order TL, TR, BR, BL}."""
    name = entry["material"]
    rows, cols, ids = [(r, c, i) for n, r, c, i in BOARD_SPECS if n == name][0]
    if "sx" in entry:
        sx, sy = float(entry["sx"]), float(entry["sy"])
    else:
        sx, sy = viz.board_half_extents(rows, cols, entry.get("tag_cm", 15.0))
    W = SQUARE * cols + GAP * (cols - 1) + 2 * GAP
    H = SQUARE * rows + GAP * (rows - 1) + 2 * GAP
    order = row_major_to_texture_order(ids, rows, cols)   # texture walks top row first
    R = viz.rot_matrix(entry.get("rot", [0, 0, 0]))
    pos = np.array(entry["pos"], float)
    out = {}
    for tr in range(rows):
        for c in range(cols):
            x0 = GAP + c * (SQUARE + GAP)
            y0 = GAP + tr * (SQUARE + GAP)
            px = np.array([[x0, y0], [x0 + SQUARE, y0], [x0 + SQUARE, y0 + SQUARE], [x0, y0 + SQUARE]], float)
            local = np.stack([(2 * px[:, 0] / W - 1) * sx, (2 * px[:, 1] / H - 1) * sy,
                              np.full(4, QUAD_Z_OFF)], 1)
            out[order[tr * cols + c]] = local @ R.T + pos
    return out


def read_detections(path, stride, n_poses, boards, wanted):
    """cam -> board -> [(seq, tag id, (4,2) px)]"""
    det = defaultdict(lambda: defaultdict(list))
    with open(path, "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages():
            if not ch.topic.endswith("/tags"):
                continue
            cam = ch.topic.split("/")[1]
            if cam not in wanted:
                continue
            with T.TagDetections.from_bytes(msg.data) as m:
                seq = m.header.seq
                if seq % stride or seq >= n_poses:
                    continue
                w, h = m.image.width, m.image.height
                for t in m.tags:
                    name = GRID_NAMES.get(int(t.gridId))
                    if name in boards and int(t.id) in boards[name]:
                        det[cam][name].append((seq, int(t.id),
                                               np.array(t.pointsPolygon, float)[:8].reshape(4, 2) * [w, h]))
    return det


def match_order(meas, proj):
    """Cyclic shift / reversal of the texture corner order that best matches
    the detector's canonical order. Returns (residual (N,4,2), (shift, rev))."""
    best = None
    for s in range(4):
        for rev in (1, -1):
            order = np.roll(np.arange(4)[::rev], s)
            r = meas - proj[:, order]
            med = np.median(np.linalg.norm(r, axis=2))
            if best is None or med < best[0]:
                best = (med, r, (s, rev))
    return best[1], best[2]


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", default=os.path.expanduser("~/vilota_results/office_v9_tags.mcap"))
    ap.add_argument("--traj", default=os.path.expanduser("~/vilota_results/office_v9_traj.csv"))
    ap.add_argument("--scene", default=os.path.join(REPO, "office_scene_v9.json"))
    ap.add_argument("--calib", default=os.path.join(REPO, "calibration_files/DP180IP-30020104.json"))
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--cams", default="cama,camb,camc,camd")
    args = ap.parse_args()
    wanted = [c.strip() for c in args.cams.split(",")]

    cams = load_cams(args.calib)
    rows = [[float(r[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
            for r in csv.DictReader(open(args.traj))]
    boards = {b["material"]: board_corners_3d(b) for b in json.load(open(args.scene))["boards"]
              if b["material"] in GRID_NAMES.values()}
    print(f"{len(rows)} poses, boards {list(boards)}, cameras {wanted}, calib {os.path.basename(args.calib)}")
    det = read_detections(args.tags, args.stride, len(rows), boards, wanted)

    results = {}
    for ci in sorted(cams):
        cam = CAM_NAMES[ci]
        if cam not in wanted:
            continue
        model, T_D_i = cams[ci]
        for name, corners3d in boards.items():
            items = det[cam].get(name, [])
            if not items:
                continue
            by_seq = defaultdict(list)
            for seq, tid, px in items:
                by_seq[seq].append((tid, px))
            proj, meas = [], []
            for seq, lst in by_seq.items():
                view_i = cam_view(camd_view(rows[seq]), T_D_i)
                X = np.vstack([corners3d[tid] for tid, _ in lst])
                uv, ok = project(model, view_i, X)
                uv, ok = uv.reshape(-1, 4, 2), ok.reshape(-1, 4).all(1)
                for k, (tid, px) in enumerate(lst):
                    if ok[k]:
                        proj.append(uv[k])
                        meas.append(px)
            proj, meas = np.array(proj), np.array(meas)
            r, shift = match_order(meas, proj)
            e = np.linalg.norm(r, axis=2)
            bias = r.reshape(-1, 2).mean(0)
            e_nobias = np.linalg.norm(r - bias, axis=2)
            results[(cam, name)] = dict(n=e.size, tags=len(e), frames=len(by_seq), med=np.median(e),
                                        p95=np.percentile(e, 95), mean=e.mean(), bias=bias,
                                        med_nobias=np.median(e_nobias), shift=shift,
                                        per_corner=np.median(e, axis=0), e=e)

    print("\ncorner-level residual |detected - projected| [px]  (shift = detector index of texture TL, reversed?)")
    hdr = (f"{'cam':5} {'board':11} {'frames':>6} {'tags':>6} {'corners':>7} | {'median':>7} {'p95':>7} {'mean':>7} | "
           f"{'bias du':>8} {'dv':>7} | {'median-bias':>11} | {'per-corner medians':>28} | shift")
    print(hdr)
    print("-" * len(hdr))
    for (cam, name), R in results.items():
        print(f"{cam:5} {name:11} {R['frames']:6d} {R['tags']:6d} {R['n']:7d} | {R['med']:7.3f} {R['p95']:7.3f} {R['mean']:7.3f} | "
              f"{R['bias'][0]:+8.3f} {R['bias'][1]:+7.3f} | {R['med_nobias']:11.3f} | "
              f"{' '.join(f'{v:6.3f}' for v in R['per_corner'])} | {R['shift']}")
    print("\nper camera, all boards pooled:")
    for cam in wanted:
        es = [R["e"] for (c, _), R in results.items() if c == cam]
        if es:
            e = np.concatenate([x.ravel() for x in es])
            print(f"  {cam}: {e.size} corners  median {np.median(e):.3f}  p95 {np.percentile(e, 95):.3f}")


if __name__ == "__main__":
    main()
