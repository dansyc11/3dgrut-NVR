"""Corner-level reprojection check of a playground render, per camera and board.

    python tools/reproject_far.py [--tags office_v9_tags.mcap] [--traj ...csv]
        [--scene office_scene_v9.json] [--calib calibration_files/....json]
        [--stride 1] [--cams cama,camb,camc,camd]
        [--swap camb,camc [--remove-rotation] [--remove-translation] [--plot out.png]]

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
TL,TR,BR,BL) chosen by minimum median. The projection includes the half-pixel
convention difference between kaolin's ray centres (i + 0.5) and the
detector's integer-centred coordinates. The CSV row index is header.seq.

--swap camX,camY projects each of the two cameras' detections through the
OTHER camera's calibration entry (intrinsics and extrinsics), i.e. the error
a mislabelled calibration would leave on real detections. --remove-rotation
/ --remove-translation keep the camera's own extrinsic rotation / translation
(only the rest is swapped); with both, only the intrinsics are swapped, the
counterpart of swap_error_map.py --same-pose. Swap runs report the constant
offset (mean residual vector) and the residual after removing it, and with
--plot overlay the measured error vs field angle on the model prediction at
the detected corners and on the uniform-field --same-pose prediction.
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


def cam_points(view_i, X):
    """World points -> OpenCV camera frame of the camera with view_i."""
    Xgl = X @ view_i[:3, :3].T + view_i[:3, 3]
    return Xgl * np.array([1.0, -1.0, -1.0])


def project(model, view_i, X):
    uv, ok = model.project(cam_points(view_i, X))
    # kaolin casts pixel i's ray through i + 0.5; the detector reports
    # integer-centred pixel coordinates, so the render is offset by half a pixel.
    return uv - 0.5, ok


def field_angle_deg(Xcv):
    return np.degrees(np.arctan2(np.hypot(Xcv[:, 0], Xcv[:, 1]), Xcv[:, 2]))


def uniform_swap_prediction(model_own, model_use, step_deg=1.0):
    """--same-pose prediction on a uniform direction grid in the camera's own
    frame: pi_own - pi_use of the same ray, both models at one pose.
    Returns (field angle deg, error vector (N,2)) over the jointly valid field."""
    az = np.radians(np.arange(-110, 110 + 1e-9, step_deg))
    el = np.radians(np.arange(-80, 80 + 1e-9, step_deg))
    A, E = np.meshgrid(az, el, indexing="xy")
    a, e = A.ravel(), E.ravel()
    dirs = np.stack([np.cos(e) * np.sin(a), -np.sin(e), np.cos(e) * np.cos(a)], 1)
    uo, ok_o = model_own.project(dirs)
    uu, ok_u = model_use.project(dirs)
    m = ok_o & ok_u
    return field_angle_deg(dirs[m]), (uo - uu)[m]


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
    ap.add_argument("--swap", default="",
                    help="camX,camY: project each one's detections through the other's calibration entry")
    ap.add_argument("--remove-rotation", action="store_true",
                    help="with --swap: keep each camera's own extrinsic rotation")
    ap.add_argument("--remove-translation", action="store_true",
                    help="with --swap: keep each camera's own extrinsic translation")
    ap.add_argument("--plot", default="", help="with --swap: error-vs-field-angle overlay PNG")
    args = ap.parse_args()
    wanted = [c.strip() for c in args.cams.split(",")]

    cams = load_cams(args.calib)
    used = dict(cams)
    swapped = []
    if args.swap:
        a, b = [CAM_NAMES.index(c.strip()) for c in args.swap.split(",")]
        for me, other in ((a, b), (b, a)):
            model_o, T_o = cams[other]
            T = T_o.copy()
            if args.remove_rotation:
                T[:3, :3] = cams[me][1][:3, :3]
            if args.remove_translation:
                T[:3, 3] = cams[me][1][:3, 3]
            used[me] = (model_o, T)
        swapped = [CAM_NAMES[a], CAM_NAMES[b]]
        print(f"--swap: {swapped[0]} <-> {swapped[1]} calibration entries"
              + (" (own rotation kept)" if args.remove_rotation else "")
              + (" (own translation kept)" if args.remove_translation else ""))
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
        model_u, T_u = used[ci]
        is_swap = cam in swapped
        for name, corners3d in boards.items():
            items = det[cam].get(name, [])
            if not items:
                continue
            by_seq = defaultdict(list)
            for seq, tid, px in items:
                by_seq[seq].append((tid, px))
            proj, proj_u, meas, ang = [], [], [], []
            for seq, lst in by_seq.items():
                view_d = camd_view(rows[seq])
                view_i = cam_view(view_d, T_D_i)
                X = np.vstack([corners3d[tid] for tid, _ in lst])
                uv, ok = project(model, view_i, X)
                th = field_angle_deg(cam_points(view_i, X)).reshape(-1, 4)
                if is_swap:
                    uv_u, ok_u = project(model_u, cam_view(view_d, T_u), X)
                    ok = ok & ok_u
                    uv_u = uv_u.reshape(-1, 4, 2)
                uv, ok = uv.reshape(-1, 4, 2), ok.reshape(-1, 4).all(1)
                for k, (tid, px) in enumerate(lst):
                    if ok[k]:
                        proj.append(uv[k])
                        meas.append(px)
                        ang.append(th[k])
                        if is_swap:
                            proj_u.append(uv_u[k])
            proj, meas, ang = np.array(proj), np.array(meas), np.array(ang)
            # the corner order is decided on the true projection (the floor)
            r_floor, shift = match_order(meas, proj)
            e = np.linalg.norm(r_floor, axis=2)
            R = dict(n=e.size, tags=len(e), frames=len(by_seq), med=np.median(e),
                     p95=np.percentile(e, 95), mean=e.mean(), shift=shift,
                     per_corner=np.median(e, axis=0), e=e, ang=ang)
            if is_swap:
                s_, rev = shift
                order = np.roll(np.arange(4)[::rev], s_)
                proj_u = np.array(proj_u)[:, order]
                r = meas - proj_u                     # measured swap residual
                expct = proj[:, order] - proj_u       # model prediction at these corners
                R.update(r=r, expct=expct, e_swap=np.linalg.norm(r, axis=2),
                         med_swap=np.median(np.linalg.norm(r, axis=2)))
            results[(cam, name)] = R

    print("\ncorner-level residual |detected - projected| [px]  (shift = detector index of texture TL, reversed?)")
    hdr = (f"{'cam':5} {'board':11} {'frames':>6} {'tags':>6} {'corners':>7} | {'median':>7} {'p95':>7} {'mean':>7} | "
           f"{'per-corner medians':>28} | shift")
    print(hdr)
    print("-" * len(hdr))
    for (cam, name), R in results.items():
        print(f"{cam:5} {name:11} {R['frames']:6d} {R['tags']:6d} {R['n']:7d} | {R['med']:7.3f} {R['p95']:7.3f} {R['mean']:7.3f} | "
              f"{' '.join(f'{v:6.3f}' for v in R['per_corner'])} | {R['shift']}")
    print("\nper camera, all boards pooled (true calibration = the floor):")
    for cam in wanted:
        es = [R["e"] for (c, _), R in results.items() if c == cam]
        if es:
            e = np.concatenate([x.ravel() for x in es])
            print(f"  {cam}: {e.size} corners  median {np.median(e):.3f}  p95 {np.percentile(e, 95):.3f}")

    if not swapped:
        return
    report_swap(results, swapped, cams, used, args)


def _stats(v):
    e = np.linalg.norm(v, axis=-1).ravel()
    return np.median(e), np.percentile(e, 95)


def _binned(ang, err, bins):
    xs, ys = [], []
    idx = np.digitize(ang, bins)
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() >= 8:
            xs.append(0.5 * (bins[b - 1] + bins[b]))
            ys.append(np.median(err[sel]))
    return np.array(xs), np.array(ys)


def report_swap(results, swapped, cams, used, args):
    """Swap residual per board and per camera: constant offset (mean residual
    vector), residual after removing it, the model prediction at the same
    corners, and the uniform-field --same-pose prediction."""
    print(f"\nswap residual |detected - projected through the swapped entry| [px]")
    hdr = (f"{'cam':5} {'board':11} {'corners':>7} | {'median':>8} {'p95':>8} | {'offset du':>9} {'dv':>8} "
           f"| {'after offset':>12} {'p95':>8} | {'model at corners':>16} {'p95':>8}")
    print(hdr)
    print("-" * len(hdr))
    pooled = {}
    for cam in swapped:
        rs = [(R["r"], R["expct"], R["ang"], name) for (c, name), R in results.items() if c == cam and "r" in R]
        if not rs:
            continue          # the other half of the pair was not in --cams
        for r, ex, ang, name in rs:
            off = r.reshape(-1, 2).mean(0)
            m, p = _stats(r)
            m2, p2 = _stats(r - off)
            m3, p3 = _stats(ex - ex.reshape(-1, 2).mean(0))
            print(f"{cam:5} {name:11} {r.shape[0] * 4:7d} | {m:8.3f} {p:8.3f} | {off[0]:+9.3f} {off[1]:+8.3f} "
                  f"| {m2:12.3f} {p2:8.3f} | {m3:16.3f} {p3:8.3f}")
        r = np.concatenate([x[0] for x in rs]).reshape(-1, 2)
        ex = np.concatenate([x[1] for x in rs]).reshape(-1, 2)
        ang = np.concatenate([x[2] for x in rs]).ravel()
        off, off_ex = r.mean(0), ex.mean(0)
        m, p = _stats(r)
        m2, p2 = _stats(r - off)
        m3, p3 = _stats(ex - off_ex)
        ci = CAM_NAMES.index(cam)
        ang_u, d_u = uniform_swap_prediction(cams[ci][0], used[ci][0])
        off_u = d_u.mean(0)
        m4, p4 = _stats(d_u - off_u)
        print(f"{cam:5} {'ALL':11} {len(r):7d} | {m:8.3f} {p:8.3f} | {off[0]:+9.3f} {off[1]:+8.3f} "
              f"| {m2:12.3f} {p2:8.3f} | {m3:16.3f} {p3:8.3f}")
        print(f"      model offset at corners ({off_ex[0]:+.3f}, {off_ex[1]:+.3f}); uniform-field --same-pose "
              f"prediction: offset ({off_u[0]:+.3f}, {off_u[1]:+.3f}), after offset median {m4:.3f} p95 {p4:.3f}; "
              f"corners span field angle {ang.min():.1f}..{ang.max():.1f} deg")
        pooled[cam] = (ang, np.linalg.norm(r - off, axis=1), np.linalg.norm(ex - off_ex, axis=1),
                       ang_u, np.linalg.norm(d_u - off_u, axis=1), (m2, p2, m3, p3, m4, p4))

    bins = np.arange(0, 90.1, 2.0)
    print("\nerror vs field angle from the camera's own optical axis, offset removed (2 deg bins, median px):")
    for cam, (ang, e_m, e_x, ang_u, e_u, _) in pooled.items():
        xm, ym = _binned(ang, e_m, bins)
        xx, yx = _binned(ang, e_x, bins)
        xu, yu = _binned(ang_u, e_u, bins)
        print(f"  {cam}:  {'angle':>6} {'measured':>9} {'model@corners':>14} {'same-pose':>10}")
        for x, m_ in zip(xm, ym):
            a = yx[np.isclose(xx, x)]
            u = yu[np.isclose(xu, x)]
            print(f"         {x:6.0f} {m_:9.3f} {a[0] if a.size else float('nan'):14.3f} {u[0] if u.size else float('nan'):10.3f}")

    if not args.plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"measured": "#2a78d6", "model at corners": "#eb6834", "same-pose prediction": "#1baf7a"}
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e6e5e1",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    fig, axs = plt.subplots(1, len(pooled), figsize=(5.5 * len(pooled), 4.2), squeeze=False)
    for ax, (cam, (ang, e_m, e_x, ang_u, e_u, st)) in zip(axs[0], pooled.items()):
        for (lab, col), (a_, e_) in zip(COL.items(), ((ang, e_m), (ang, e_x), (ang_u, e_u))):
            x, y = _binned(a_, e_, bins)
            ax.plot(x, y, "-", color=col, lw=2, label=lab)
        ax.set_xlabel(f"field angle from {cam}'s optical axis [deg]")
        ax.set_ylabel("median error after constant offset [px]")
        ax.set_title(f"{cam} with the {[c for c in swapped if c != cam][0]} entry"
                     + (" (own R)" if args.remove_rotation else "") + (" (own t)" if args.remove_translation else "")
                     + f"\nmedian/p95: measured {st[0]:.2f}/{st[1]:.2f}, model@corners {st[2]:.2f}/{st[3]:.2f}, "
                     f"same-pose {st[4]:.2f}/{st[5]:.2f}", loc="left", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=140)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
