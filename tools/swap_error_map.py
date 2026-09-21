"""CamB <-> CamC swap error map from a device calibration file. CPU only.

    python tools/swap_error_map.py [calibration_files/DP180IP-30020104.json]
        [--out swap_error_map.png] [--distances 1 3 10 100 1000]
        [--equalize-to mid|camb|camc] [--step-deg 1.0]

For a grid of world points X (world = the reference camera's frame, CamD),
projects X through CamB and CamC (KB4 models + device extrinsics, float64) and
reports the swap error

    e(X) = | pi_C(T_C^-1 X) - pi_B(T_B^-1 X) |   [px]

as median / p95 per distance, restricted to the field both cameras see at
every distance. The error is split into a translation term and a
rotation+intrinsics term by re-projecting with the two camera centres moved
onto one point (--equalize-to): Delta_eq is the rotation+intrinsics term,
Delta - Delta_eq is the translation (parallax) term.

Device-file conventions (verified against the calib_v9 vk_calibrate solve to
0.1-0.5 mm / 0.02-0.05 deg): extrinsics are cam_i -> reference,
X_ref = R_i X_i + t_i, translation in CENTIMETRES; the reference camera has an
empty rotationMatrix (identity) and zero translation. KB4 (cameraType 1):
fx fy cx cy from intrinsicMatrix, k1..k4 = distortionCoeff[0:4],
d(theta) = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8) - the same
polynomial kaolin_future/fisheye.py:279 inverts for the render.
"""
import argparse
import json
import os

import numpy as np

CAM_NAMES = ["CamA", "CamB", "CamC", "CamD"]   # rig index -> name (DP180IP)

# dataviz reference palette: categorical slots 1-3 (validated all-pairs) and
# the sequential blue ramp used as an ordinal series scale / heatmap.
SERIES = {"total": "#2a78d6", "translation": "#eb6834",
          "rot+intr": "#1baf7a"}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"]
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
        dd = (1 + 3 * self.k1 * th**2 + 5 * self.k2 * th**4
              + 7 * self.k3 * th**6 + 9 * self.k4 * th**8)
        bad = np.nonzero(dd <= 0)[0]
        self.theta_max = float(th[bad[0] - 1]) if bad.size else float(np.pi)

    def d(self, theta):
        t2 = theta * theta
        return theta * (1 + t2 * (self.k1 + t2 * (self.k2 + t2 * (
            self.k3 + t2 * self.k4))))

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
        valid = ((theta < self.theta_max) & (u >= 0) & (u < self.width)
                 & (v >= 0) & (v < self.height))
        return np.stack([u, v], axis=1), valid


def load_device(path):
    """-> dict rig_index -> (KB4|None, R (3,3), t (3,) metres, cameraType)."""
    with open(path) as fh:
        data = json.load(fh)
    cams = {}
    for idx, c in data["cameraData"]:
        e = c["extrinsics"]
        rot = e.get("rotationMatrix") or []
        R = np.array(rot, dtype=np.float64) if len(rot) else np.eye(3)
        tr = e["translation"]
        t = np.array([tr["x"], tr["y"], tr["z"]], dtype=np.float64) / 100.0
        K = c["intrinsicMatrix"]
        model = None
        if c["cameraType"] == 1:
            model = KB4(K[0][0], K[1][1], K[0][2], K[1][2],
                        c["distortionCoeff"][0:4], c["width"], c["height"])
        cams[int(idx)] = (model, R, t, c["cameraType"])
    return data.get("deviceName", "?"), cams


def to_cam(R, t, Xw):
    """world -> cam_i for X_w = R X_i + t."""
    return (Xw - t) @ R          # == R.T @ (Xw - t) row-wise


def direction_grid(step_deg, az_max=110.0, el_max=80.0):
    """Unit directions in the world (CamD) frame over an az/el grid.
    az about the world y axis (positive towards +x, right), el about x
    (positive towards -y, up). Returns dirs (N,3), az (N,), el (N,) in deg."""
    az = np.arange(-az_max, az_max + 1e-9, step_deg)
    el = np.arange(-el_max, el_max + 1e-9, step_deg)
    AZ, EL = np.meshgrid(az, el, indexing="xy")
    a, e = np.radians(AZ.ravel()), np.radians(EL.ravel())
    dirs = np.stack([np.cos(e) * np.sin(a), -np.sin(e), np.cos(e) * np.cos(a)],
                    axis=1)
    return dirs, AZ.ravel(), EL.ravel(), (len(el), len(az))


def swap_delta(camB, camC, dirs, dist, tB, tC):
    """pi_C - pi_B for X = dist * dirs, and the joint-validity mask."""
    X = dirs * dist
    uvB, vB = camB[0].project(to_cam(camB[1], tB, X))
    uvC, vC = camC[0].project(to_cam(camC[1], tC, X))
    return uvC - uvB, vB & vC


def stats(err, mask):
    e = err[mask]
    return (float(np.median(e)), float(np.percentile(e, 95)),
            float(e.min()), float(e.max()))


def make_figure(out, dists, rows, field, az, el, shape, maps, eq_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "legend.frameon": False,
    })
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle("CamB <-> CamC swap error  |pi_C(X) - pi_B(X)|", color=INK,
                 fontsize=12, x=0.02, ha="left")

    # (a) error vs distance: median solid, p95 dotted, per term
    ax = axs[0, 0]
    for key, col in SERIES.items():
        med = [rows[d][key][0] for d in dists]
        p95 = [rows[d][key][1] for d in dists]
        ax.plot(dists, med, "-", color=col, lw=2, marker="o", ms=4)
        ax.plot(dists, p95, ":", color=col, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance from CamD [m]")
    ax.set_ylabel("swap error [px]")
    ax.set_title("Error vs distance (solid median, dotted p95)", loc="left",
                 fontsize=10)
    ax.legend([plt.Line2D([], [], color=c, lw=2) for c in SERIES.values()],
              [f"{k}" for k in SERIES], loc="lower left", fontsize=8)

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
    ax.set_xlabel("field angle from CamD +z [deg]")
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
        im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal",
                       extent=[az.min(), az.max(), el.min(), el.max()])
        ax.set_xlim(az_ok.min() - pad, az_ok.max() + pad)
        ax.set_ylim(el_ok.min() - pad, el_ok.max() + pad)
        ax.grid(False)
        ax.set_xlabel("azimuth in CamD frame [deg]  (+ = right)")
        ax.set_ylabel("elevation [deg]  (+ = up)")
        ax.set_title(title, loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(unit, color=INK2)
        cb.outline.set_visible(False)
    fig.text(0.02, 0.005, f"translations equalized to {eq_label}; white = "
             "outside the field shared by CamB and CamC at all distances",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib", nargs="?",
                    default="calibration_files/DP180IP-30020104.json")
    ap.add_argument("--distances", type=float, nargs="+",
                    default=[1.0, 3.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--step-deg", type=float, default=1.0,
                    help="direction grid step (default 1 deg)")
    ap.add_argument("--equalize-to", choices=["mid", "camb", "camc"],
                    default="mid",
                    help="where both camera centres go for the "
                         "rotation+intrinsics term (default midpoint)")
    ap.add_argument("--out", default="swap_error_map.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    serial, cams = load_device(args.calib)
    print(f"device {serial}  ({args.calib})")
    for i in sorted(cams):
        m, R, t, ct = cams[i]
        kind = "KB4" if ct == 1 else ("DS" if ct == 0 else f"type{ct}")
        fwd = R[:, 2]
        print(f"  {CAM_NAMES[i]} idx {i} {kind:3s} centre {t * 100} cm  "
              f"forward {fwd}  yaw {np.degrees(np.arctan2(fwd[0], fwd[2])):+.2f} deg"
              + (f"  theta_max {np.degrees(m.theta_max):.1f} deg" if m else ""))
    camB, camC = cams[1], cams[2]
    for name, cam in (("CamB", camB), ("CamC", camC)):
        if cam[0] is None:
            raise SystemExit(f"{name} is not KB4 (cameraType {cam[3]})")
        print(f"  {name} fx {cam[0].fx:.3f} fy {cam[0].fy:.3f} cx {cam[0].cx:.3f} "
              f"cy {cam[0].cy:.3f} k {cam[0].k1:+.5f} {cam[0].k2:+.5f} "
              f"{cam[0].k3:+.5f} {cam[0].k4:+.5f}  {cam[0].width}x{cam[0].height}")
    tB, tC = camB[2], camC[2]
    baseline = np.linalg.norm(tC - tB)
    rel = camB[1].T @ camC[1]
    ang_bc = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    print(f"  baseline B-C {baseline * 100:.3f} cm, optical axes {ang_bc:.2f} deg apart")

    eq = {"mid": 0.5 * (tB + tC), "camb": tB, "camc": tC}[args.equalize_to]
    eq_label = {"mid": "the B-C midpoint", "camb": "CamB's centre",
                "camc": "CamC's centre"}[args.equalize_to]

    dirs, az, el, shape = direction_grid(args.step_deg)
    field_angle = np.degrees(np.arccos(np.clip(dirs[:, 2], -1, 1)))
    dists = list(args.distances)

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
    print(f"\nshared field: {n} directions of {len(dirs)} "
          f"({args.step_deg:g} deg grid), az {az[common].min():+.0f}..{az[common].max():+.0f} "
          f"el {el[common].min():+.0f}..{el[common].max():+.0f} deg, "
          f"field angle up to {field_angle[common].max():.1f} deg")
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
        rows[d] = {"total": stats(e_tot, common),
                   "translation": stats(e_tr, common),
                   "rot+intr": stats(e_eq, common)}
        field[d] = (field_angle, e_tot, common)
        r = rows[d]
        print(f"{d:9g} | {r['total'][0]:9.2f} {r['total'][1]:8.2f} | "
              f"{r['translation'][0]:9.3f} {r['translation'][1]:8.3f} | "
              f"{r['rot+intr'][0]:12.2f} {r['rot+intr'][1]:8.2f} | "
              f"{r['translation'][0] * d:11.2f}")

    # Far asymptote: pure directions, no translation at all (X at infinity).
    dl_inf, m_inf = swap_delta(camB, camC, dirs, 1.0, np.zeros(3), np.zeros(3))
    e_inf = np.linalg.norm(dl_inf, axis=1)
    med, p95, lo, hi = stats(e_inf, common)
    far = dists[-1]
    print(f"\nfar-distance asymptote (X at infinity, translation dropped): "
          f"median {med:.3f} px, p95 {p95:.3f} px, min {lo:.3f}, max {hi:.3f}")
    print(f"  at {far:g} m the total is median {rows[far]['total'][0]:.3f} / "
          f"p95 {rows[far]['total'][1]:.3f} px, i.e. within "
          f"{abs(rows[far]['total'][0] - med):.4f} / "
          f"{abs(rows[far]['total'][1] - p95):.4f} px of the asymptote; the "
          f"translation term there is median {rows[far]['translation'][0]:.4f} px")
    # Where the asymptote lands: forward ray and the field-angle trend.
    fwd = np.argmin(field_angle + np.where(common, 0, 1e9))
    print(f"  forward ray (az {az[fwd]:+.0f}, el {el[fwd]:+.0f}): "
          f"|pi_C - pi_B| = {e_inf[fwd]:.2f} px, u_C - u_B = {dl_inf[fwd][0]:+.1f} px, "
          f"v_C - v_B = {dl_inf[fwd][1]:+.1f} px")
    for lim in (10, 20, 30, 40):
        sel = common & (field_angle <= lim)
        if sel.any():
            print(f"  field angle <= {lim:2d} deg: asymptote median "
                  f"{np.median(e_inf[sel]):8.2f} px, p95 {np.percentile(e_inf[sel], 95):8.2f}")

    if args.no_plot:
        return
    grid_far = np.full(len(dirs), np.nan)
    grid_far[common] = e_inf[common]
    grid_tr = np.full(len(dirs), np.nan)
    e_tr1 = np.linalg.norm(deltas[dists[0]][0] - deltas[dists[0]][1], axis=1)
    grid_tr[common] = e_tr1[common]
    maps = [("Rotation+intrinsics term (asymptote, X at infinity)",
             grid_far.reshape(shape), "px"),
            (f"Translation term at {dists[0]:g} m", grid_tr.reshape(shape), "px")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    make_figure(args.out, dists, rows, field, az, el, shape, maps, eq_label)


if __name__ == "__main__":
    main()
