"""Mirrored-pair swap error map from a device calibration file. CPU only.

    python tools/swap_error_map.py [calibration_files/DP180IP-30020104.json]
        [--out swap_error_map.png] [--distances 1 3 10 100 1000]
        [--equalize-to mid|camb|camc] [--step-deg 1.0] [--same-pose]

The swap pair is found from the extrinsics: the unique same-resolution KB4
pair whose optical axes are 40-80 deg apart (the mirrored front mounts;
CamB-CamC 60.5 deg on DP180IP, camb-camc 55.1 deg on NV180V2 - the
resolution test is what excludes NV180V2's camb-camd at 75.6 deg). If the
rule does not give exactly one pair, rig indices 1,2 are used as before.

For a grid of world points X (world = the reference camera's frame, CamD),
projects X through CamB and CamC (KB4 models + device extrinsics, float64) and
reports the swap error

    e(X) = | pi_C(T_C^-1 X) - pi_B(T_B^-1 X) |   [px]

as median / p95 per distance, restricted to the field both cameras see at
every distance. The error is split into a translation term and a
rotation+intrinsics term by re-projecting with the two camera centres moved
onto one point (--equalize-to): Delta_eq is the rotation+intrinsics term,
Delta - Delta_eq is the translation (parallax) term.

--same-pose models a physical module swap on the mirrored mounts: both
cameras sit at CamC's full pose (R_C, t_C) and the error is

    e(X) = | pi_B(T_C^-1 X) - pi_C(T_C^-1 X) |   [px]

i.e. CamB's intrinsics on CamC's mount vs what CamC produced there. Reported
per distance as median / p95, its constant-offset component (the mean pixel
shift over the field, dominated by the principal-point difference) and the
residual once that offset is removed.

Device-file conventions, DP180 format ("cameraData"; verified against the
calib_v9 vk_calibrate solve to 0.1-0.5 mm / 0.02-0.05 deg): extrinsics are
cam_i -> reference, X_ref = R_i X_i + t_i, translation in CENTIMETRES; the
reference camera has an empty rotationMatrix (identity) and zero translation.
KB4 (cameraType 1): fx fy cx cy from intrinsicMatrix,
k1..k4 = distortionCoeff[0:4],
d(theta) = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8) - the same
polynomial kaolin_future/fisheye.py:279 inverts for the render.

NV180 format ("calibration_data" wrapper, e.g. NV180V2 80.120.0101): two
blocks, calibration_data_kb4 (all cameras KB4) and calibration_data_ds (the
DS-slot camera, index 0, as Double Sphere). T_imu_cam is cam_i -> reference
as px..pz METRES + qx qy qz qw, but the reference DIFFERS per block: the kb4
block is IMU-referenced (T_imu_cam[0] non-trivial) while the ds block is
camera-gauge (T_imu_cam[0] identity, reference = cam 0). Verified: rebasing
the kb4 block onto cam 0 reproduces the ds block to 0.007 deg / 0.11 mm, and
|t_i - t_j| matches the stored baselines to <1 um in both blocks. This tool
reads the kb4 block (the only one with KB4 intrinsics for every camera) and
rebases it onto cam 0 so the world is a camera frame, as in the DP180 path.
"""

import argparse
import json
import os

import numpy as np

CAM_NAMES = ["CamA", "CamB", "CamC", "CamD"]  # rig index -> name (DP180IP)

# dataviz reference palette: categorical slots 1-3 (validated all-pairs) and
# the sequential blue ramp used as an ordinal series scale / heatmap.
SERIES = {"total": "#2a78d6", "translation": "#eb6834", "rot+intr": "#1baf7a"}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


class KB4:
    """Kannala-Brandt 4-coefficient fisheye, OpenCV axes (x right, y down,
    z forward)."""

    def __init__(self, fx, fy, cx, cy, k, width, height):
        self.fx, self.fy, self.cx, self.cy = map(float, (fx, fy, cx, cy))
        self.k1, self.k2, self.k3, self.k4 = map(float, k)
        self.width, self.height = int(width), int(height)
        # Largest theta up to which d(theta) is increasing (bisected on a
        # fine grid); beyond it the model folds over and is not invertible.
        th = np.linspace(0.0, np.pi, 20001)
        dd = 1 + 3 * self.k1 * th**2 + 5 * self.k2 * th**4 + 7 * self.k3 * th**6 + 9 * self.k4 * th**8
        bad = np.nonzero(dd <= 0)[0]
        self.theta_max = float(th[bad[0] - 1]) if bad.size else float(np.pi)

    def d(self, theta):
        t2 = theta * theta
        return theta * (1 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def project(self, X):
        """X: (N,3) camera-frame points -> (uv (N,2), valid (N,) bool)."""
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        safe = np.where(rho > 0, rho, 1.0)
        r = self.d(theta)
        u = self.fx * r * x / safe + self.cx
        v = self.fy * r * y / safe + self.cy
        on_axis = rho == 0
        u = np.where(on_axis, self.cx, u)
        v = np.where(on_axis, self.cy, v)
        valid = (theta < self.theta_max) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return np.stack([u, v], axis=1), valid


def quat_to_R(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_device(path):
    """-> (serial, dict rig_index -> (KB4|None, R (3,3), t (3,) metres,
    type label), dict rig_index -> camera name). Extrinsics are cam -> world
    with the world a camera frame (DP180: the reference camera; NV180: cam 0,
    the kb4 block rebased from the IMU onto it)."""
    with open(path) as fh:
        data = json.load(fh)
    cams, names = {}, {}

    if "calibration_data" in data:  # NV180 format
        wrap = data["calibration_data"]
        block = wrap["calibration_data_kb4"]
        for i, cname in enumerate(block["cam_names"]):
            names[i] = cname.split("/")[-1]
            T = block["T_imu_cam"][i]
            R = quat_to_R(T["qx"], T["qy"], T["qz"], T["qw"])
            t = np.array([T["px"], T["py"], T["pz"]], dtype=np.float64)
            k = block["intrinsics"][i]
            w, h = block["resolution"][i]
            model = None
            if k["camera_type"] == "kb4":
                v = k["intrinsics"]
                model = KB4(v["fx"], v["fy"], v["cx"], v["cy"], (v["k1"], v["k2"], v["k3"], v["k4"]), w, h)
            cams[i] = (model, R, t, k["camera_type"].upper())
        R0, t0 = cams[0][1], cams[0][2]  # rebase IMU -> cam 0 gauge
        cams = {i: (m, R0.T @ R, R0.T @ (t - t0), ct) for i, (m, R, t, ct) in cams.items()}
        return wrap.get("product_id", block.get("serial_number", "?")), cams, names

    for idx, c in data["cameraData"]:  # DP180 format
        e = c["extrinsics"]
        rot = e.get("rotationMatrix") or []
        R = np.array(rot, dtype=np.float64) if len(rot) else np.eye(3)
        tr = e["translation"]
        t = np.array([tr["x"], tr["y"], tr["z"]], dtype=np.float64) / 100.0
        K = c["intrinsicMatrix"]
        model = None
        if c["cameraType"] == 1:
            model = KB4(K[0][0], K[1][1], K[0][2], K[1][2], c["distortionCoeff"][0:4], c["width"], c["height"])
        cams[int(idx)] = (model, R, t, {0: "DS", 1: "KB4"}.get(c["cameraType"], f"type{c['cameraType']}"))
        names[int(idx)] = CAM_NAMES[int(idx)]
    return data.get("deviceName", "?"), cams, names


def find_mirror_pair(cams):
    """The unique same-resolution KB4 pair with optical axes 40-80 deg apart
    (a module swap needs interchangeable sensors), else (1, 2)."""
    kb4 = [i for i in sorted(cams) if cams[i][0] is not None]
    pairs = []
    for a in kb4:
        for b in kb4:
            if a < b:
                ma, mb = cams[a][0], cams[b][0]
                if (ma.width, ma.height) != (mb.width, mb.height):
                    continue
                ang = np.degrees(np.arccos(np.clip(cams[a][1][:, 2] @ cams[b][1][:, 2], -1, 1)))
                if 40.0 <= ang <= 80.0:
                    pairs.append((a, b, ang))
    if len(pairs) == 1:
        return pairs[0]
    print(f"  (mirror-pair rule matched {len(pairs)} pairs, " f"falling back to rig indices 1,2)")
    ang = np.degrees(np.arccos(np.clip(cams[1][1][:, 2] @ cams[2][1][:, 2], -1, 1)))
    return 1, 2, ang


def to_cam(R, t, Xw):
    """world -> cam_i for X_w = R X_i + t."""
    return (Xw - t) @ R  # == R.T @ (Xw - t) row-wise


def direction_grid(step_deg, az_max=110.0, el_max=80.0):
    """Unit directions in the world frame (a camera frame, see load_device)
    over an az/el grid.
    az about the world y axis (positive towards +x, right), el about x
    (positive towards -y, up). Returns dirs (N,3), az (N,), el (N,) in deg."""
    az = np.arange(-az_max, az_max + 1e-9, step_deg)
    el = np.arange(-el_max, el_max + 1e-9, step_deg)
    AZ, EL = np.meshgrid(az, el, indexing="xy")
    a, e = np.radians(AZ.ravel()), np.radians(EL.ravel())
    dirs = np.stack([np.cos(e) * np.sin(a), -np.sin(e), np.cos(e) * np.cos(a)], axis=1)
    return dirs, AZ.ravel(), EL.ravel(), (len(el), len(az))


def pair_uv(camB, camC, dirs, dist, poseB, poseC):
    """(uvB, uvC, joint-validity mask) for X = dist * dirs, each camera at
    its own (R, t) pose."""
    X = dirs * dist
    uvB, vB = camB[0].project(to_cam(poseB[0], poseB[1], X))
    uvC, vC = camC[0].project(to_cam(poseC[0], poseC[1], X))
    return uvB, uvC, vB & vC


def swap_delta(camB, camC, dirs, dist, tB, tC):
    """pi_C - pi_B for X = dist * dirs, and the joint-validity mask."""
    uvB, uvC, m = pair_uv(camB, camC, dirs, dist, (camB[1], tB), (camC[1], tC))
    return uvC - uvB, m


def stats(err, mask):
    e = err[mask]
    return (float(np.median(e)), float(np.percentile(e, 95)), float(e.min()), float(e.max()))


def make_figure(
    out, dists, rows, series, field, az, el, shape, maps, title, note, field_label="field angle from world +z [deg]"
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "legend.frameon": False,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")

    # (a) error vs distance: median solid, p95 dotted, per term
    ax = axs[0, 0]
    for key, col in series.items():
        med = [rows[d][key][0] for d in dists]
        p95 = [rows[d][key][1] for d in dists]
        ax.plot(dists, med, "-", color=col, lw=2, marker="o", ms=4)
        ax.plot(dists, p95, ":", color=col, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance from the world origin [m]")
    ax.set_ylabel("swap error [px]")
    ax.set_title("Error vs distance (solid median, dotted p95)", loc="left", fontsize=10)
    ax.legend(
        [plt.Line2D([], [], color=c, lw=2) for c in series.values()], [f"{k}" for k in series], loc="best", fontsize=8
    )

    # (b) total error vs field angle, one ordinal series per distance
    ax = axs[0, 1]
    bins = np.arange(0, 90.1, 2.0)
    for d, col in zip(dists, BLUE_RAMP):
        ang, err, m = field[d]
        idx = np.digitize(ang[m], bins)
        xs, ys = [], []
        for b in range(1, len(bins)):
            sel = idx == b
            if sel.sum() >= 5:
                xs.append(0.5 * (bins[b - 1] + bins[b]))
                ys.append(np.median(err[m][sel]))
        ax.plot(xs, ys, "-", color=col, lw=2, label=f"{d:g} m")
    ax.set_xlabel(field_label)
    ax.set_ylabel("median total swap error [px]")
    ax.set_title("Error vs field angle (2 deg bins)", loc="left", fontsize=10)
    ax.legend(title="distance", fontsize=8, title_fontsize=8)

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_RAMP)
    # crop the maps to the shared field plus a margin; the rest is empty
    seen = ~np.isnan(maps[0][1])
    az_ok = az.reshape(shape)[seen]
    el_ok = el.reshape(shape)[seen]
    pad = 5.0
    for ax, (title, grid, unit) in zip(axs[1], maps):
        im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal", extent=[az.min(), az.max(), el.min(), el.max()])
        ax.set_xlim(az_ok.min() - pad, az_ok.max() + pad)
        ax.set_ylim(el_ok.min() - pad, el_ok.max() + pad)
        ax.grid(False)
        ax.set_xlabel("azimuth in the world frame [deg]  (+ = right)")
        ax.set_ylabel("elevation [deg]  (+ = up)")
        ax.set_title(title, loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(unit, color=INK2)
        cb.outline.set_visible(False)
    fig.text(0.02, 0.005, note, color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib", nargs="?", default="calibration_files/DP180IP-30020104.json")
    ap.add_argument("--distances", type=float, nargs="+", default=[1.0, 3.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--step-deg", type=float, default=1.0, help="direction grid step (default 1 deg)")
    ap.add_argument(
        "--equalize-to",
        choices=["mid", "camb", "camc"],
        default="mid",
        help="where both camera centres go for the " "rotation+intrinsics term (default midpoint)",
    )
    ap.add_argument(
        "--same-pose",
        action="store_true",
        help="module swap: both cameras at CamC's pose, error = "
        "|pi_B - pi_C| from CamB's intrinsics on CamC's mount",
    )
    ap.add_argument("--out", default="swap_error_map.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    serial, cams, names = load_device(args.calib)
    print(f"device {serial}  ({args.calib})")
    for i in sorted(cams):
        m, R, t, kind = cams[i]
        fwd = R[:, 2]
        print(
            f"  {names[i]} idx {i} {kind:3s} centre {t * 100} cm  "
            f"forward {fwd}  yaw {np.degrees(np.arctan2(fwd[0], fwd[2])):+.2f} deg"
            + (f"  theta_max {np.degrees(m.theta_max):.1f} deg" if m else "")
        )
    iB, iC, pair_ang = find_mirror_pair(cams)
    nameB, nameC = names[iB], names[iC]
    print(f"  mirrored pair: {nameB} <-> {nameC} " f"(optical axes {pair_ang:.2f} deg apart)")
    camB, camC = cams[iB], cams[iC]
    for name, cam in ((nameB, camB), (nameC, camC)):
        if cam[0] is None:
            raise SystemExit(f"{name} is not KB4 ({cam[3]})")
        print(
            f"  {name} fx {cam[0].fx:.3f} fy {cam[0].fy:.3f} cx {cam[0].cx:.3f} "
            f"cy {cam[0].cy:.3f} k {cam[0].k1:+.5f} {cam[0].k2:+.5f} "
            f"{cam[0].k3:+.5f} {cam[0].k4:+.5f}  {cam[0].width}x{cam[0].height}"
        )
    tB, tC = camB[2], camC[2]
    baseline = np.linalg.norm(tC - tB)
    rel = camB[1].T @ camC[1]
    ang_bc = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    print(f"  baseline {nameB}-{nameC} {baseline * 100:.3f} cm, " f"full relative rotation {ang_bc:.2f} deg")

    eq = {"mid": 0.5 * (tB + tC), "camb": tB, "camc": tC}[args.equalize_to]
    eq_label = {"mid": f"the {nameB}-{nameC} midpoint", "camb": f"{nameB}'s centre", "camc": f"{nameC}'s centre"}[
        args.equalize_to
    ]

    dirs, az, el, shape = direction_grid(args.step_deg)
    field_angle = np.degrees(np.arccos(np.clip(dirs[:, 2], -1, 1)))
    dists = list(args.distances)

    if args.same_pose:
        same_pose_mode(args, camB, camC, nameB, nameC, dirs, az, el, shape, field_angle, dists)
        return

    # One common mask: directions inside both images at EVERY distance, so
    # the per-distance statistics compare the same set of rays.
    deltas, masks = {}, {}
    common = np.ones(len(dirs), dtype=bool)
    for d in dists:
        dl, m = swap_delta(camB, camC, dirs, d, tB, tC)
        dl_eq, m_eq = swap_delta(camB, camC, dirs, d, eq, eq)
        deltas[d] = (dl, dl_eq)
        masks[d] = m & m_eq
        common &= masks[d]
    n = int(common.sum())
    print(
        f"\nshared field: {n} directions of {len(dirs)} "
        f"({args.step_deg:g} deg grid), az {az[common].min():+.0f}..{az[common].max():+.0f} "
        f"el {el[common].min():+.0f}..{el[common].max():+.0f} deg, "
        f"field angle up to {field_angle[common].max():.1f} deg"
    )
    for d in dists:
        extra = int((masks[d] & ~common).sum())
        if extra:
            print(f"  ({extra} more directions are jointly visible at {d:g} m only)")

    print(f"\nswap error |pi_C - pi_B| [px], translations equalized to {eq_label}")
    hdr = f"{'dist [m]':>9} | {'total med':>9} {'p95':>8} | {'trans med':>9} {'p95':>8} | {'rot+intr med':>12} {'p95':>8} | {'trans med*d':>11}"
    print(hdr)
    print("-" * len(hdr))
    rows, field = {}, {}
    for d in dists:
        dl, dl_eq = deltas[d]
        e_tot = np.linalg.norm(dl, axis=1)
        e_eq = np.linalg.norm(dl_eq, axis=1)
        e_tr = np.linalg.norm(dl - dl_eq, axis=1)
        rows[d] = {"total": stats(e_tot, common), "translation": stats(e_tr, common), "rot+intr": stats(e_eq, common)}
        field[d] = (field_angle, e_tot, common)
        r = rows[d]
        print(
            f"{d:9g} | {r['total'][0]:9.2f} {r['total'][1]:8.2f} | "
            f"{r['translation'][0]:9.3f} {r['translation'][1]:8.3f} | "
            f"{r['rot+intr'][0]:12.2f} {r['rot+intr'][1]:8.2f} | "
            f"{r['translation'][0] * d:11.2f}"
        )

    # Far asymptote: pure directions, no translation at all (X at infinity).
    dl_inf, m_inf = swap_delta(camB, camC, dirs, 1.0, np.zeros(3), np.zeros(3))
    e_inf = np.linalg.norm(dl_inf, axis=1)
    med, p95, lo, hi = stats(e_inf, common)
    far = dists[-1]
    print(
        f"\nfar-distance asymptote (X at infinity, translation dropped): "
        f"median {med:.3f} px, p95 {p95:.3f} px, min {lo:.3f}, max {hi:.3f}"
    )
    print(
        f"  at {far:g} m the total is median {rows[far]['total'][0]:.3f} / "
        f"p95 {rows[far]['total'][1]:.3f} px, i.e. within "
        f"{abs(rows[far]['total'][0] - med):.4f} / "
        f"{abs(rows[far]['total'][1] - p95):.4f} px of the asymptote; the "
        f"translation term there is median {rows[far]['translation'][0]:.4f} px"
    )
    # Where the asymptote lands: forward ray and the field-angle trend.
    fwd = np.argmin(field_angle + np.where(common, 0, 1e9))
    print(
        f"  forward ray (az {az[fwd]:+.0f}, el {el[fwd]:+.0f}): "
        f"|pi_C - pi_B| = {e_inf[fwd]:.2f} px, u_C - u_B = {dl_inf[fwd][0]:+.1f} px, "
        f"v_C - v_B = {dl_inf[fwd][1]:+.1f} px"
    )
    for lim in (10, 20, 30, 40):
        sel = common & (field_angle <= lim)
        if sel.any():
            print(
                f"  field angle <= {lim:2d} deg: asymptote median "
                f"{np.median(e_inf[sel]):8.2f} px, p95 {np.percentile(e_inf[sel], 95):8.2f}"
            )

    if args.no_plot:
        return
    grid_far = np.full(len(dirs), np.nan)
    grid_far[common] = e_inf[common]
    grid_tr = np.full(len(dirs), np.nan)
    e_tr1 = np.linalg.norm(deltas[dists[0]][0] - deltas[dists[0]][1], axis=1)
    grid_tr[common] = e_tr1[common]
    maps = [
        ("Rotation+intrinsics term (asymptote, X at infinity)", grid_far.reshape(shape), "px"),
        (f"Translation term at {dists[0]:g} m", grid_tr.reshape(shape), "px"),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_figure(
        args.out,
        dists,
        rows,
        SERIES,
        field,
        az,
        el,
        shape,
        maps,
        f"{nameB} <-> {nameC} swap error  |pi_C(X) - pi_B(X)|",
        f"translations equalized to {eq_label}; white = outside the "
        f"field shared by {nameB} and {nameC} at all distances",
    )


def same_pose_mode(args, camB, camC, nameB, nameC, dirs, az, el, shape, field_angle, dists):
    """Both cameras at the C-side pose: the error is the B module's
    intrinsics on the C mount, |pi_B(T_C^-1 X) - pi_C(T_C^-1 X)|,
    direction-only by construction (the two centres coincide), so it does
    not fall off with distance."""
    pose = (camC[1], camC[2])
    kB, kC = camB[0], camC[0]
    # field angle from the C mount's own optical axis, the natural coordinate
    field_angle = np.degrees(np.arccos(np.clip(dirs @ camC[1][:, 2], -1, 1)))
    print(f"\n--same-pose: both cameras at {nameC}'s pose, error = |pi_B - pi_C|")
    print(
        f"  intrinsic deltas {nameB}-{nameC}: cx {kB.cx - kC.cx:+.3f} px, "
        f"cy {kB.cy - kC.cy:+.3f} px, "
        f"fx ratio {kB.fx / kC.fx:.5f}, fy ratio {kB.fy / kC.fy:.5f}, "
        f"k1..k4 {kB.k1 - kC.k1:+.5f} {kB.k2 - kC.k2:+.5f} "
        f"{kB.k3 - kC.k3:+.5f} {kB.k4 - kC.k4:+.5f}"
    )

    deltas, masks = {}, {}
    common = np.ones(len(dirs), dtype=bool)
    for d in dists:
        uvB, uvC, m = pair_uv(camB, camC, dirs, d, pose, pose)
        deltas[d] = uvB - uvC
        masks[d] = m
        common &= m
    n = int(common.sum())
    print(
        f"shared field (both models valid from {nameC}'s mount): {n} directions "
        f"of {len(dirs)} ({args.step_deg:g} deg grid), az {az[common].min():+.0f}.."
        f"{az[common].max():+.0f} el {el[common].min():+.0f}..{el[common].max():+.0f} deg "
        f"in the world frame, up to {field_angle[common].max():.1f} deg off {nameC}'s axis"
    )
    for d in dists:
        extra = int((masks[d] & ~common).sum())
        if extra:
            print(f"  ({extra} more directions are jointly visible at {d:g} m only)")

    print(f"\nmodule-swap error |pi_B - pi_C| [px] on {nameC}'s mount")
    hdr = (
        f"{'dist [m]':>9} | {'total med':>9} {'p95':>8} | "
        f"{'const offset du':>15} {'dv':>8} {'|off|':>7} | {'residual med':>12} {'p95':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows, field = {}, {}
    for d in dists:
        dl = deltas[d]
        off = dl[common].mean(axis=0)
        e_tot = np.linalg.norm(dl, axis=1)
        e_res = np.linalg.norm(dl - off, axis=1)
        e_off = np.full(len(dl), np.linalg.norm(off))
        rows[d] = {
            "intrinsics": stats(e_tot, common),
            "constant offset": stats(e_off, common),
            "offset removed": stats(e_res, common),
        }
        field[d] = (field_angle, e_tot, common)
        r = rows[d]
        print(
            f"{d:9g} | {r['intrinsics'][0]:9.3f} {r['intrinsics'][1]:8.3f} | "
            f"{off[0]:15.3f} {off[1]:8.3f} {np.linalg.norm(off):7.3f} | "
            f"{r['offset removed'][0]:12.3f} {r['offset removed'][1]:8.3f}"
        )

    # Direction-only limit (X at infinity): the centres coincide anyway, so
    # this differs from the table only through the grid's parallax to CamC.
    uvB, uvC, _ = pair_uv(camB, camC, dirs, 1.0, (camC[1], np.zeros(3)), (camC[1], np.zeros(3)))
    dl_inf = uvB - uvC
    off_inf = dl_inf[common].mean(axis=0)
    e_inf = np.linalg.norm(dl_inf, axis=1)
    e_res_inf = np.linalg.norm(dl_inf - off_inf, axis=1)
    med, p95, lo, hi = stats(e_inf, common)
    rmed, rp95, rlo, rhi = stats(e_res_inf, common)
    print(
        f"\nfar-distance asymptote (X at infinity): median {med:.3f} px, p95 {p95:.3f}, " f"min {lo:.3f}, max {hi:.3f}"
    )
    print(
        f"  constant offset (mean over the field): du {off_inf[0]:+.3f} px, "
        f"dv {off_inf[1]:+.3f} px, |offset| {np.linalg.norm(off_inf):.3f} px "
        f"(principal-point delta alone: {kB.cx - kC.cx:+.3f}, {kB.cy - kC.cy:+.3f})"
    )
    print(
        f"  after removing the offset: median {rmed:.3f} px, p95 {rp95:.3f}, "
        f"min {rlo:.3f}, max {rhi:.3f}  (focal + distortion difference)"
    )
    axis = np.argmin(np.linalg.norm(uvC - [kC.cx, kC.cy], axis=1) + np.where(common, 0, 1e9))
    print(
        f"  on {nameC}'s optical axis (az {az[axis]:+.0f}, el {el[axis]:+.0f}): "
        f"du {dl_inf[axis][0]:+.3f} dv {dl_inf[axis][1]:+.3f} px"
    )
    rad = np.linalg.norm(uvC - [kC.cx, kC.cy], axis=1)
    for lim in (100, 200, 300, 400, 500, 600):
        sel = common & (rad <= lim)
        if sel.any():
            print(
                f"  within {lim:3d} px of {nameC}'s principal point: "
                f"median {np.median(e_inf[sel]):7.3f} px, p95 {np.percentile(e_inf[sel], 95):7.3f}, "
                f"residual median {np.median(e_res_inf[sel]):7.3f}"
            )

    if args.no_plot:
        return
    grid_tot = np.full(len(dirs), np.nan)
    grid_tot[common] = e_inf[common]
    grid_res = np.full(len(dirs), np.nan)
    grid_res[common] = e_res_inf[common]
    maps = [
        ("Module-swap error |pi_B - pi_C| (X at infinity)", grid_tot.reshape(shape), "px"),
        ("After removing the constant offset", grid_res.reshape(shape), "px"),
    ]
    series = {
        "intrinsics": SERIES["total"],
        "constant offset": SERIES["translation"],
        "offset removed": SERIES["rot+intr"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_figure(
        args.out,
        dists,
        rows,
        series,
        field,
        az,
        el,
        shape,
        maps,
        f"{nameB} module on {nameC}'s mount  |pi_B(X) - pi_C(X)|",
        f"both cameras at {nameC}'s pose; white = outside the field " "both models see from that mount",
        field_label=f"field angle from {nameC}'s optical axis [deg]",
    )


if __name__ == "__main__":
    main()
