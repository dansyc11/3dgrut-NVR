"""Recover the metric scale + Sim(3) of a Gaussian-splat scene from its LiDAR scan.

Method (CPU only, reads only):
  1. Detect the splat's up axis (FIORD meetingroom is -Y up, main_campus is +Y up
     -- verified from data: the ground carpet must sit BELOW the cameras).
  2. Level both clouds on their RANSAC ground planes (floor = LOWEST substantial
     horizontal z-peak below the cameras, NOT the largest plane: in the meeting
     room the conference table out-votes the floor).
  3. Global coarse search over (yaw, scale): translation per candidate from 2D
     FFT cross-correlation of above-ground occupancy grids; candidates rescored
     by symmetric 3D trimmed-NN inlier fractions (fwd * rev; the reverse
     direction punishes scale collapse).
  4. Re-crop the LiDAR to the matched footprint, REFIT its ground locally
     (the campus ground is sloped; a global plane misleads), re-run a fine
     (yaw, scale) grid, then a trimmed-ICP polish of (scale, yaw, tx, ty, tz).
  5. Extract vertical planes as 2D lines (wall-band points -> z-coverage cells
     -> sequential line RANSAC) in both clouds, match them through the
     registration, and report scale = gap ratio for every matched parallel
     pair. Matching MUST be correspondence-driven: in the meeting room the
     splat reconstructs whiteboards/blinds ~0.2 m inside featureless walls, so
     naively pairing outermost peaks gives 0.548 where the truth is 0.50.
  6. Validate by NN residuals of transformed splat points against the LiDAR,
     and write <dataset>/lidar_alignment.json with the full Sim(3).

Usage:
    python tools/lidar_scale_align.py meetingroom
    python tools/lidar_scale_align.py campus
(Use the 3dgrut venv python: /home/vilota/niel_gs/3dgrut/.venv/bin/python)
"""
import argparse
import datetime
import json
import struct
import sys

import numpy as np
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)

PRESETS = {
    "meetingroom": dict(
        splat="/home/vilota/datasets/meetingroom/splat_points.ply",
        lidar="/home/vilota/datasets/meetingroom/lidar_sub.npy",
        images_bin="/home/vilota/datasets/meetingroom/model/images.bin",
        out="/home/vilota/datasets/meetingroom/lidar_alignment.json",
        scale_range=(0.30, 0.80),
        path_crop=6.0,          # splat units around the camera path
        band=(0.10, 1.95),      # metres above ground used for registration
        facade_band=(0.45, 1.90),  # metres above ground used for wall lines
        cell=0.12,              # occupancy cell, metres
        coarse_cell=0.12,       # occupancy cell for the global sweep
        line_thresh=0.06,       # line RANSAC threshold, metres (lidar side)
        line_thresh_splat=0.10, # line RANSAC threshold, splat units
        icp_trims=(1.0, 0.6, 0.4, 0.3, 0.22, 0.16),
        nn_eps=0.15,            # rescore inlier radius, metres
        eps_tight=0.10,         # basin adjudication radius, metres
        struct_lo=0.30,         # metres above ground: "structure" (no floor)
        z_min_u=None,           # optional splat structure floor, UNITS
        sfm_points="/home/vilota/datasets/meetingroom/model/points3D.bin",
        register_with="splat",
        splat_peak_frac=0.25,
        off_tol=0.30,
        expected_scale=0.50,    # known-answer control
    ),
    "campus": dict(
        splat="/home/vilota/datasets/main_campus/splat_points.npy",
        lidar="/home/vilota/datasets/main_campus/lidar_sub.npy",
        images_bin="/home/vilota/datasets/main_campus/colmap/model/images.bin",
        out="/home/vilota/datasets/main_campus/lidar_alignment.json",
        # cameras sit 0.30u above the splat ground; a walking rig is 0.9-2.2 m
        # above ground, so scale is physically confined to ~[3.0, 7.4]
        scale_range=(2.0, 8.0),
        path_crop=6.0,
        band=(0.8, 12.0),
        facade_band=(1.0, 9.0),
        cell=0.40,
        coarse_cell=0.80,
        line_thresh=0.25,
        line_thresh_splat=0.10,
        icp_trims=(4.0, 2.5, 1.5, 1.0, 0.7, 0.5),
        nn_eps=0.60,
        eps_tight=0.40,
        struct_lo=1.0,
        # structure = above CAMERA height (0.30u): the ground carpet cannot
        # reach there at any scale, so large-scale candidates cannot flood
        # the registration band with carpet points
        z_min_u=0.40,
        # register on COLMAP sparse points (same frame as the splat): the
        # splat is a ground carpet + wall fuzz whose peaks smear over ~1 m,
        # while the SfM cloud shows crisp facade lines on both street sides
        sfm_points="/home/vilota/datasets/main_campus/colmap/model/points3D.bin",
        register_with="sfm",
        # looser peak/match thresholds: SfM facade peaks are weaker relative
        # to the strongest one than in the indoor scene, and the street-width
        # pair (the best scale evidence) sits just under the indoor cutoffs
        splat_peak_frac=0.15,
        off_tol=0.35,
        expected_scale=None,
    ),
}

A_MINUS_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # up = -Y -> +Z
A_PLUS_Y = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # up = +Y -> +Z


# ---------------------------------------------------------------- data loading

def read_cam_centers(path):
    """COLMAP images.bin -> Nx3 camera centres C = -R^T t."""
    centers = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = np.array(struct.unpack("<dddd", f.read(32)))
            t = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)
            while f.read(1) != b"\x00":
                pass
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * npts, 1)
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            centers.append(-R.T @ t)
    return np.stack(centers)


def load_points(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    from plyfile import PlyData
    v = PlyData.read(path).elements[0]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                     np.asarray(v["z"])], axis=1).astype(np.float64)


def read_points3d_bin(path, max_err=2.0, min_track=3):
    """COLMAP points3D.bin -> Nx3, filtered by reprojection error and track
    length. Same world frame as the splat; bundle-adjusted feature points are
    far crisper registration material than fuzzy splat centers."""
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            data = f.read(51)
            xyz = struct.unpack("<ddd", data[8:32])
            err = struct.unpack("<d", data[35:43])[0]
            tlen = struct.unpack("<Q", data[43:51])[0]
            f.seek(8 * tlen, 1)
            if err < max_err and tlen >= min_track:
                pts.append(xyz)
    return np.array(pts)


# ------------------------------------------------------------ plane primitives

def plane_ransac(pts, thresh, iters=800, normal_prior=None, max_tilt_deg=15.0):
    """RANSAC plane n.p = d; optional cone constraint on the normal."""
    best = None
    idx = rng.integers(0, len(pts), size=(iters, 3))
    for i0, i1, i2 in idx:
        n = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n = n / L
        if normal_prior is not None and abs(n @ normal_prior) < np.cos(np.radians(max_tilt_deg)):
            continue
        d = n @ pts[i0]
        nin = int((np.abs(pts @ n - d) < thresh).sum())
        if best is None or nin > best[0]:
            best = (nin, n, d)
    if best is None:
        raise RuntimeError("plane RANSAC found no candidate")
    _, n, d = best
    for _ in range(3):
        m = np.abs(pts @ n - d) < thresh
        P = pts[m]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[2]
        if normal_prior is not None and n @ normal_prior < 0:
            n = -n
        d = n @ c
    m = np.abs(pts @ n - d) < thresh
    return n, d, int(m.sum())


def leveling(n_up, d):
    """Rotation R with R@n_up = +z; leveled points = pts@R.T with z -= d."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n_up, z)
    s = np.linalg.norm(v)
    c = float(n_up @ z)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def lowest_floor_peak(z_below, nbins=120):
    """Lowest substantial peak of a z histogram = the floor, not the table."""
    lo, hi = np.percentile(z_below, [0.2, 99.8])
    binw = max((hi - lo) / nbins, 1e-4)
    h, e = np.histogram(z_below, bins=np.arange(lo, hi + binw, binw))
    cands = [((e[i] + e[i + 1]) / 2) for i in range(len(h)) if h[i] > 0.25 * h.max()]
    return min(cands), binw


def fit_floor(pts, cam_med_z, thresh=None):
    below = pts[pts[:, 2] < cam_med_z]
    z0, binw = lowest_floor_peak(below[:, 2])
    thresh = 3.0 * binw if thresh is None else thresh
    seed = pts[np.abs(pts[:, 2] - z0) < 3.5 * thresh]
    n, d, nin = plane_ransac(seed, thresh=thresh,
                             normal_prior=np.array([0, 0, 1.0]), max_tilt_deg=15)
    return n, d, nin, thresh


# --------------------------------------------------------------- registration

def occupancy(xy, x0, y0, nx, ny, cell):
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[nx, ny],
                             range=[[x0, x0 + nx * cell], [y0, y0 + ny * cell]])
    return np.sqrt(H)


def coarse_search(spl, lidar_band, scale_range, cell, yaw_step=2.0,
                  scale_step=1.035, band=(0.1, 2.0), n_keep=60, sp_max=30000,
                  yaw_range=(0.0, 360.0), z_min_u=None):
    """(yaw, s) grid; per-candidate translation by FFT xcorr. Returns candidates."""
    x0 = lidar_band[:, 0].min() - 2 * cell
    y0 = lidar_band[:, 1].min() - 2 * cell
    nx = int(np.ceil((lidar_band[:, 0].max() - x0) / cell)) + 4
    ny = int(np.ceil((lidar_band[:, 1].max() - y0) / cell)) + 4
    Hl = occupancy(lidar_band[:, :2], x0, y0, nx, ny, cell)
    Fl_cache = {}
    if len(spl) > sp_max:
        spl = spl[rng.choice(len(spl), sp_max, replace=False)]
    scales = []
    s = scale_range[0]
    while s <= scale_range[1] * 1.0001:
        scales.append(s)
        s *= scale_step
    cands = []
    for s in scales:
        P = spl[splat_band_mask(spl[:, 2], s, band, z_min_u)]
        if len(P) < 200:
            continue
        for yaw in np.arange(yaw_range[0], yaw_range[1], yaw_step):
            r = np.radians(yaw)
            R2 = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            xy = P[:, :2] @ R2.T * s
            # splat histogram on its OWN origin-anchored grid: the shift is
            # recovered from the correlation PLUS the grid-origin difference,
            # so nothing depends on the clouds overlapping a priori
            sx0 = xy[:, 0].min() - 2 * cell
            sy0 = xy[:, 1].min() - 2 * cell
            nsx = int(np.ceil((xy[:, 0].max() - sx0) / cell)) + 4
            nsy = int(np.ceil((xy[:, 1].max() - sy0) / cell)) + 4
            NX = 1 << int(np.ceil(np.log2(nx + nsx)))
            NY = 1 << int(np.ceil(np.log2(ny + nsy)))
            if (NX, NY) not in Fl_cache:
                Fl_cache[(NX, NY)] = np.fft.rfft2(Hl, s=(NX, NY))
            Fl = Fl_cache[(NX, NY)]
            Hs = occupancy(xy, sx0, sy0, nsx, nsy, cell)
            cc = np.fft.irfft2(Fl * np.conj(np.fft.rfft2(Hs, s=(NX, NY))),
                               s=(NX, NY))
            k = np.unravel_index(np.argmax(cc), cc.shape)
            mx = k[0] if k[0] < NX // 2 else k[0] - NX
            my = k[1] if k[1] < NY // 2 else k[1] - NY
            cands.append((float(cc[k]), float(s), float(yaw),
                          (x0 - sx0) + mx * cell, (y0 - sy0) + my * cell))
    cands.sort(key=lambda t: -t[0])
    return cands[:n_keep]


def voxel_down(pts, cell):
    """One point per (cell)^3 voxel. Terrestrial-scanner subsamples carry
    ~1/r^2 density around each station; without this, NN scores measure scan
    density instead of structure, and a splat parked on a dense clump wins."""
    ij = np.floor(pts / cell).astype(np.int64)
    _, idx = np.unique(ij, axis=0, return_index=True)
    return pts[idx]


def splat_band_mask(z, s, band, z_min_u=None):
    """Splat structure slab for scale hypothesis s: metric band, with an
    optional scale-INdependent floor in units (keeps the ground carpet out of
    the band even when a large s would pull the metric threshold below it)."""
    lo = band[0] / s
    if z_min_u is not None:
        lo = max(lo, z_min_u)
    return (z > lo) & (z * s < band[1])


def transform_pts(P, s, yaw_deg, t):
    r = np.radians(yaw_deg)
    R2 = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    Q = np.empty_like(P)
    Q[:, :2] = P[:, :2] @ R2.T * s + t[:2]
    Q[:, 2] = P[:, 2] * s + t[2]
    return Q


def rescore(cands, spl, lidar_band, ltree, band, eps, sp_max=12000, li_max=20000,
            z_min_u=None):
    """Symmetric trimmed-NN score: fwd (splat->lidar) * rev (lidar->splat).

    rev is measured against a FIXED sample of the WHOLE lidar band, never a
    region cropped to the candidate: a collapsed-scale candidate then explains
    almost none of the lidar and scores near zero, while restricting to the
    candidate's own bbox would let it 'fully explain' a tiny patch."""
    Lfix = lidar_band if len(lidar_band) <= li_max else \
        lidar_band[rng.choice(len(lidar_band), li_max, replace=False)]
    out = []
    for _, s, yaw, tx, ty in cands:
        P = spl[splat_band_mask(spl[:, 2], s, band, z_min_u)]
        if len(P) < 200:
            continue
        if len(P) > sp_max:
            P = P[rng.choice(len(P), sp_max, replace=False)]
        Q = transform_pts(P, s, yaw, np.array([tx, ty, 0.0]))
        d, _ = ltree.query(Q, k=1)
        fwd = float((d < eps).mean())
        dr, _ = cKDTree(Q).query(Lfix, k=1)
        rev = float((dr < eps).mean())
        out.append(dict(score=fwd * rev, fwd=fwd, rev=rev, s=s, yaw=yaw, tx=tx, ty=ty))
    out.sort(key=lambda r: -r["score"])
    return out


def icp_refine(spl_slab, ltree, s, yaw, t, trims, fix_scale=False):
    """Trimmed ICP over (s, yaw, tx, ty, tz): 2D Umeyama + median tz per round."""
    t = np.array(t, dtype=float)
    P0 = spl_slab
    if len(P0) > 25000:
        P0 = P0[rng.choice(len(P0), 25000, replace=False)]
    stats = {}
    for trim in trims:
        Q = transform_pts(P0, s, yaw, t)
        d, j = ltree.query(Q, k=1)
        keep = d < trim
        if keep.sum() < 100:
            break
        P = P0[keep]
        X = ltree.data[j[keep]]
        # 2D similarity (Umeyama): leveled-splat xy -> lidar xy
        p = P[:, :2]
        x = X[:, :2]
        mp, mx = p.mean(0), x.mean(0)
        pc, xc = p - mp, x - mx
        cov = xc.T @ pc / len(p)
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(2)
        if np.linalg.det(U @ Vt) < 0:
            S[1, 1] = -1
        R2 = U @ S @ Vt
        var_p = (pc ** 2).sum() / len(p)
        s_new = s if fix_scale else float(np.trace(np.diag(D) @ S) / var_p)
        t2 = mx - s_new * (R2 @ mp)
        yaw_new = float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0])))
        tz = float(np.median(X[:, 2] - s_new * P[:, 2]))
        s, yaw, t = s_new, yaw_new, np.array([t2[0], t2[1], tz])
        stats = dict(trim_m=float(trim), matched=int(keep.sum()), of=int(len(P0)),
                     rms_m=float(np.sqrt((d[keep] ** 2).mean())))
    return s, yaw % 360.0, t, stats


# -------------------------------------------------------------- wall planes

def wall_cells(pts, cell, zmin, zmax, min_cover=0.5, min_pts=4, nz=8):
    """Centres of 2D cells whose z occupancy covers >= min_cover of [zmin,zmax]."""
    band = pts[(pts[:, 2] > zmin) & (pts[:, 2] < zmax)]
    if len(band) == 0:
        return np.empty((0, 2))
    ij = np.floor(band[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 1000003 + ij[:, 1]
    order = np.argsort(key)
    key_s, z_s, xy_s = key[order], band[order, 2], band[order, :2]
    uniq, start = np.unique(key_s, return_index=True)
    out = []
    for k in range(len(uniq)):
        a = start[k]
        b = start[k + 1] if k + 1 < len(uniq) else len(key_s)
        if b - a < min_pts:
            continue
        zz = z_s[a:b]
        bins = np.floor((zz - zmin) / (zmax - zmin) * nz).clip(0, nz - 1).astype(int)
        if len(np.unique(bins)) / nz >= min_cover:
            out.append(xy_s[a:b].mean(0))
    return np.array(out) if out else np.empty((0, 2))


def seq_line_ransac(xy, thresh, n_lines=8, iters=1500, min_in=20):
    """Sequential 2D line RANSAC; returns [{theta, d, nin, seg}] (normal form)."""
    pts = xy.copy()
    out = []
    while len(out) < n_lines and len(pts) >= max(min_in, 8):
        best = None
        for _ in range(iters):
            i, j = rng.choice(len(pts), 2, replace=False)
            dv = pts[j] - pts[i]
            L = np.hypot(*dv)
            if L < 1e-6:
                continue
            nv = np.array([-dv[1], dv[0]]) / L
            d = nv @ pts[i]
            nin = int((np.abs(pts @ nv - d) < thresh).sum())
            if best is None or nin > best[0]:
                best = (nin, nv, d)
        if best is None or best[0] < min_in:
            break
        _, nv, d = best
        for _ in range(2):
            m = np.abs(pts @ nv - d) < thresh
            P = pts[m]
            c = P.mean(0)
            _, _, vt = np.linalg.svd(P - c, full_matrices=False)
            direc = vt[0]
            nv = np.array([-direc[1], direc[0]])
            d = float(nv @ c)
        m = np.abs(pts @ nv - d) < thresh
        P = pts[m]
        direc = np.array([nv[1], -nv[0]])
        tp = (P - P.mean(0)) @ direc
        seg = float(np.percentile(tp, 98) - np.percentile(tp, 2))
        out.append(dict(theta=float(np.degrees(np.arctan2(nv[1], nv[0])) % 180.0),
                        n=nv.copy(), d=d, nin=int(m.sum()), seg=seg))
        pts = pts[~m]
    return out


def ang_diff(a, b):
    """Distance between undirected line angles, degrees in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def refine_angle(xy, th0, half=2.5, step=0.1, binw=0.05):
    """Angle near th0 (deg) whose 1D projection histogram is sharpest."""
    best = None
    if len(xy) > 60000:
        xy = xy[rng.choice(len(xy), 60000, replace=False)]
    for th in np.arange(th0 - half, th0 + half + 1e-9, step):
        nv = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
        v = xy @ nv
        h, _ = np.histogram(v, bins=np.arange(v.min(), v.max() + binw, binw))
        sc = float(((h / max(1, len(v))) ** 2).sum())
        if best is None or sc > best[0]:
            best = (sc, th)
    return float(best[1])


def profile_peaks(pts, nvec, binw, zmin, zmax, min_sep, min_frac=0.10,
                  min_count=30, min_cover=0.40, nz=8, max_peaks=6):
    """Vertical-plane candidates along one family normal.

    Peaks of the 1D offset histogram of band points, each required to have
    real vertical extent (z octile coverage of [zmin,zmax] in its slab) so
    table edges and furniture rows do not masquerade as walls."""
    v = pts[:, :2] @ nvec
    z = pts[:, 2]
    e = np.arange(v.min() - binw, v.max() + 2 * binw, binw)
    h, e = np.histogram(v, bins=e)
    order = np.argsort(h)[::-1]
    halfw = max(binw, min_sep / 2)
    peaks = []
    for i in order:
        if h[i] < max(min_count, min_frac * h.max()):
            break
        c = (e[i] + e[i + 1]) / 2
        if any(abs(c - p["offset"]) < min_sep for p in peaks):
            continue
        m = np.abs(v - c) < halfw
        off = float(v[m].mean())
        if any(abs(off - p["offset"]) < min_sep for p in peaks):
            continue
        zz = z[m]
        octs = np.floor((zz - zmin) / (zmax - zmin) * nz).clip(0, nz - 1).astype(int)
        cover = len(np.unique(octs)) / nz
        peaks.append(dict(offset=off, count=int(m.sum()), cover=float(cover),
                          is_vertical=bool(cover >= min_cover)))
        if len(peaks) >= max_peaks:
            break
    return sorted(peaks, key=lambda p: p["offset"])


# ------------------------------------------------------------------- pipeline

def T4(R=None, t=None, s=1.0):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, :3] *= s
    if t is not None:
        M[:3, 3] = t
    return M


def run(name, cfg, args):
    print(f"=== {name} ===")
    sp_raw = load_points(cfg["splat"])
    li_raw = load_points(cfg["lidar"])
    cams = read_cam_centers(cfg["images_bin"])
    reg_src = args.register_with or cfg["register_with"]
    reg_raw = read_points3d_bin(cfg["sfm_points"]) if reg_src == "sfm" else sp_raw
    print(f"splat {len(sp_raw)} pts, lidar {len(li_raw)} pts, cams {len(cams)}; "
          f"registering with {reg_src} cloud ({len(reg_raw)} pts)")

    # --- 1. up-axis detection. Two candidates (+-Y up); for each, fit the
    # lowest substantial horizontal plane below the cameras with a COMMON
    # threshold, then score = inliers penalized by the mass hanging BELOW the
    # plane: a true floor has only sub-floor floaters beneath it, while a
    # mirrored ceiling (the wrong sign) has the entire room beneath it.
    up_data = {}
    for tag, A in (("-y", A_MINUS_Y), ("+y", A_PLUS_Y)):
        q = reg_raw @ A.T
        qc = cams @ A.T
        tree = cKDTree(qc[:, :2])
        d, _ = tree.query(q[:, :2], k=1)
        near = q[d < cfg["path_crop"]]
        below = near[near[:, 2] < np.median(qc[:, 2])]
        z0, binw = lowest_floor_peak(below[:, 2])
        up_data[tag] = dict(near=near, qc=qc, z0=z0, binw=binw)
    thr = 3.0 * min(up_data[t]["binw"] for t in up_data)
    up_scores = {}
    for tag, ud in up_data.items():
        seed = ud["near"][np.abs(ud["near"][:, 2] - ud["z0"]) < 3.5 * thr]
        try:
            n, dpl, nin = plane_ransac(seed, thresh=thr,
                                       normal_prior=np.array([0, 0, 1.0]),
                                       max_tilt_deg=15)
        except RuntimeError:
            up_scores[tag] = dict(inliers=0, below=0, score=0.0)
            continue
        resid = ud["near"] @ n - dpl
        nin_all = int((np.abs(resid) < thr).sum())
        n_below = int((resid < -3 * thr).sum())
        up_scores[tag] = dict(
            inliers=nin_all, below_plane=n_below,
            score=float(nin_all / (1.0 + 5.0 * n_below)),
            cam_height=float(np.median(ud["qc"][:, 2]) - dpl))
    chosen = args.up_axis
    if chosen == "auto":
        chosen = max(up_scores, key=lambda k: up_scores[k]["score"])
    print(f"up-axis: {chosen}  (common thr {thr:.3f}u, scores {up_scores})")
    A = A_MINUS_Y if chosen == "-y" else A_PLUS_Y

    spu = reg_raw @ A.T
    cams_u = cams @ A.T
    ctree = cKDTree(cams_u[:, :2])
    dpath, _ = ctree.query(spu[:, :2], k=1)
    spn = spu[dpath < cfg["path_crop"]]
    dnear = dpath[dpath < cfg["path_crop"]]
    print(f"near-path {reg_src} points ({cfg['path_crop']}u): {len(spn)}")

    # --- 2. level splat on its floor
    gn_s, gd_s, gin_s, gthr_s = fit_floor(spn, np.median(cams_u[:, 2]))
    R_s = leveling(gn_s, gd_s)
    spl = spn @ R_s.T
    spl[:, 2] -= gd_s
    cam_h = float(np.median(cams_u @ R_s.T, axis=0)[2] - gd_s)
    print(f"splat floor: n={gn_s.round(4)} d={gd_s:.3f} inl={gin_s} "
          f"(thr {gthr_s:.3f}u); cameras {cam_h:.2f}u above floor")

    # --- 3. level lidar on its (global) ground
    low = li_raw[li_raw[:, 2] < np.percentile(li_raw[:, 2], 25)]
    if len(low) > 60000:
        low = low[rng.choice(len(low), 60000, replace=False)]
    gn_l, gd_l, gin_l, _ = fit_floor(low, cam_med_z=low[:, 2].max() + 1.0,
                                     thresh=cfg["line_thresh"])
    R_l0 = leveling(gn_l, gd_l)
    lil = li_raw @ R_l0.T
    lil[:, 2] -= gd_l
    print(f"lidar global ground: n={gn_l.round(4)} d={gd_l:.3f} inl={gin_l}")

    # --- 4. coarse (yaw, scale) search (lidar density-normalized by voxels)
    band = cfg["band"]
    lb = lil[(lil[:, 2] > band[0]) & (lil[:, 2] < band[1])]
    lb = voxel_down(lb, cfg["cell"])
    lsub = lb if len(lb) <= 150000 else lb[rng.choice(len(lb), 150000, replace=False)]
    ltree = cKDTree(lsub)
    print(f"coarse search: scale {cfg['scale_range']}, lidar band pts "
          f"{len(lb)} (voxel {cfg['cell']} m)")
    cands = coarse_search(spl, lb, cfg["scale_range"], cfg["coarse_cell"],
                          band=band, z_min_u=cfg["z_min_u"], n_keep=150)
    scored = rescore(cands, spl, lb, ltree, band, cfg["nn_eps"],
                     z_min_u=cfg["z_min_u"])
    best = scored[0]
    print("top coarse candidates (score=fwd*rev):")
    for r in scored[:5]:
        print(f"  s={r['s']:.3f} yaw={r['yaw']:6.1f} t=({r['tx']:+7.2f},{r['ty']:+7.2f}) "
              f"fwd={r['fwd']:.3f} rev={r['rev']:.3f} score={r['score']:.4f}")

    # distinct candidate basins: rev only VETOES collapsed candidates (a
    # shrunken splat explains almost none of the lidar), then rank by fwd
    # physical path test: the transformed CAMERA PATH must lie on scanned,
    # walkable lidar ground and not inside obstacles. Without this the search
    # happily parks the walk in unscanned void next to some matching edge
    # structure (campus: every early attractor failed this test at <=0.41).
    camsl = cams_u @ R_s.T
    camsl[:, 2] -= gd_s
    GCELL = 1.0
    gx0, gy0 = lil[:, 0].min() - GCELL, lil[:, 1].min() - GCELL
    gnx = int((lil[:, 0].max() - gx0) / GCELL) + 2
    gny = int((lil[:, 1].max() - gy0) / GCELL) + 2
    gm = lil[(lil[:, 2] > -1.8) & (lil[:, 2] < 0.8)]
    Hg, _, _ = np.histogram2d(gm[:, 0], gm[:, 1], bins=[gnx, gny],
                              range=[[gx0, gx0 + gnx * GCELL],
                                     [gy0, gy0 + gny * GCELL]])
    om = lil[(lil[:, 2] > 0.4) & (lil[:, 2] < 2.0)]
    Ho, _, _ = np.histogram2d(om[:, 0], om[:, 1], bins=[gnx, gny],
                              range=[[gx0, gx0 + gnx * GCELL],
                                     [gy0, gy0 + gny * GCELL]])

    def path_stats(s, yaw, tx, ty):
        r = np.radians(yaw)
        R2 = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        pc = camsl[:, :2] @ R2.T * s + [tx, ty]
        ii = ((pc[:, 0] - gx0) / GCELL).astype(int)
        jj = ((pc[:, 1] - gy0) / GCELL).astype(int)
        inb = (ii >= 0) & (ii < gnx) & (jj >= 0) & (jj < gny)
        ground = np.zeros(len(pc), bool)
        ground[inb] = Hg[ii[inb], jj[inb]] > 0
        obst = np.zeros(len(pc), bool)
        obst[inb] = Ho[ii[inb], jj[inb]] > 3
        return float(ground.mean()), float(obst.mean())

    for r_ in scored:
        g, o = path_stats(r_["s"], r_["yaw"], r_["tx"], r_["ty"])
        r_["path_ground"], r_["path_obst"] = g, o
    ok = [r for r in scored if r["path_ground"] >= 0.6 and r["path_obst"] <= 0.35]
    if len(ok) < 2:
        print("WARNING: <2 candidates keep the camera path on scanned ground "
              ">=0.6; relaxing to 0.4")
        ok = [r for r in scored if r["path_ground"] >= 0.4 and
              r["path_obst"] <= 0.45]
    if not ok:
        print("WARNING: NO candidate keeps the camera path on scanned ground; "
              "proceeding unfiltered -- treat the result as untrusted")
        ok = scored
    rev_max = max(r["rev"] for r in ok)
    survivors = [r for r in ok if r["rev"] >= 0.5 * rev_max]
    survivors.sort(key=lambda r: -r["fwd"])
    basins = []
    for r in survivors:
        if any(abs(np.log(r["s"] / b["s"])) < 0.06 and
               min(abs(r["yaw"] - b["yaw"]), 360 - abs(r["yaw"] - b["yaw"])) < 6
               for b in basins):
            continue
        basins.append(r)
        if len(basins) >= 5:
            break
    print(f"{len(basins)} distinct basins kept "
          f"({[(round(b['s'], 3), b['yaw']) for b in basins]})")

    # --- 5. crop lidar to the UNION of the basins' footprints, refit ground
    # LOCALLY there (the campus ground is sloped; a global plane misleads)
    lo = np.array([np.inf, np.inf])
    hi = -lo.copy()
    for b in basins:
        zin = splat_band_mask(spl[:, 2], b["s"], band, cfg["z_min_u"])
        Q = transform_pts(spl[zin], b["s"], b["yaw"],
                          np.array([b["tx"], b["ty"], 0.0]))
        m = 0.15 * max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1])) + 4 * cfg["cell"]
        lo = np.minimum(lo, Q[:, :2].min(0) - m)
        hi = np.maximum(hi, Q[:, :2].max(0) + m)
    crop = (lil[:, 0] > lo[0]) & (lil[:, 0] < hi[0]) & \
           (lil[:, 1] > lo[1]) & (lil[:, 1] < hi[1])
    li_crop_raw = li_raw[crop]
    print(f"lidar crop to footprint+{m:.1f}m margin: {len(li_crop_raw)} pts")
    lowc = li_crop_raw[li_crop_raw[:, 2] <
                       np.percentile(li_crop_raw[:, 2], 30)]
    gn_lc, gd_lc, gin_lc, _ = fit_floor(lowc, cam_med_z=lowc[:, 2].max() + 1.0,
                                        thresh=cfg["line_thresh"])
    R_l = leveling(gn_lc, gd_lc)
    lic = li_crop_raw @ R_l.T
    lic[:, 2] -= gd_lc
    tilt = np.degrees(np.arccos(np.clip(gn_l @ gn_lc, -1, 1)))
    print(f"lidar local ground: n={gn_lc.round(4)} d={gd_lc:.3f} inl={gin_lc} "
          f"(tilt vs global: {tilt:.2f} deg)")

    # --- 6. multi-basin refinement in the locally-leveled crop. The coarse
    # score (fwd*rev) can prefer a stretched registration whose fuzzy splat
    # walls drape over the outer lidar walls (meeting room: a 0.53 basin vs
    # the true 0.50). So: take the DISTINCT top basins (rev only vetoes
    # collapsed candidates), converge ICP in each, and adjudicate by the
    # tight-eps forward inlier fraction on structure points -- precision of
    # matched structure, which the wrong basin cannot fake.
    lbc = voxel_down(lic[(lic[:, 2] > band[0]) & (lic[:, 2] < band[1])],
                     cfg["cell"])
    ltree_band = cKDTree(lbc if len(lbc) <= 150000
                         else lbc[rng.choice(len(lbc), 150000, replace=False)])
    lslab = voxel_down(lic[(lic[:, 2] > -1.0) & (lic[:, 2] < band[1] + 1.0)],
                       cfg["cell"] / 2)
    ltree_full = cKDTree(lslab if len(lslab) <= 250000
                         else lslab[rng.choice(len(lslab), 250000, replace=False)])
    lstruct = voxel_down(lic[(lic[:, 2] > cfg["struct_lo"]) &
                             (lic[:, 2] < band[1])], cfg["cell"] / 2)
    if len(lstruct) > 150000:
        lstruct = lstruct[rng.choice(len(lstruct), 150000, replace=False)]
    ltree_struct = cKDTree(lstruct)
    lstruct_fix = lstruct if len(lstruct) <= 15000 else \
        lstruct[rng.choice(len(lstruct), 15000, replace=False)]
    results = []
    for b in basins:
        fine = coarse_search(spl, lbc, (b["s"] * 0.93, b["s"] * 1.07),
                             cell=cfg["cell"] / 2, yaw_step=0.75,
                             scale_step=1.01, band=band, n_keep=20,
                             yaw_range=(b["yaw"] - 4.0, b["yaw"] + 4.0),
                             z_min_u=cfg["z_min_u"])
        fine_scored = rescore(fine, spl, lbc, ltree_band, band, cfg["nn_eps"],
                              z_min_u=cfg["z_min_u"])
        if not fine_scored:
            continue
        fb = max(fine_scored, key=lambda r: r["fwd"])
        slab = spl[(spl[:, 2] * fb["s"] > -0.5) & (spl[:, 2] * fb["s"] < band[1])]
        s_i, yaw_i, t_i, st_i = icp_refine(
            slab, ltree_full, fb["s"], fb["yaw"],
            np.array([fb["tx"], fb["ty"], 0.0]), cfg["icp_trims"])
        if not st_i:  # ICP never matched anything: dead candidate
            print(f"basin s0={b['s']:.3f} yaw0={b['yaw']:.1f} -> ICP found no "
                  f"matches, dropped")
            continue
        # adjudication: tight-eps precision on structure points
        P = spl[splat_band_mask(spl[:, 2], s_i, (cfg["struct_lo"], band[1]),
                                cfg["z_min_u"])]
        if len(P) > 20000:
            P = P[rng.choice(len(P), 20000, replace=False)]
        if len(P) < 200:
            continue
        Q = transform_pts(P, s_i, yaw_i, t_i)
        d, _ = ltree_struct.query(Q, k=1)
        tight_fwd = float((d < cfg["eps_tight"]).mean())
        dr, _ = cKDTree(Q).query(lstruct_fix, k=1)
        tight_rev = float((dr < cfg["eps_tight"]).mean())
        # symmetric product: fwd alone cannot punish a COLLAPSED basin (a
        # compressed cloud parks on dense lidar), rev alone rewards a
        # STRETCHED one (fuzz draped over far structure); their product
        # picks the true basin in the meeting-room control. The converged
        # path-on-ground fraction folds in the walkability constraint.
        g_i, o_i = path_stats(s_i, yaw_i, t_i[0], t_i[1])
        tight = tight_fwd * tight_rev * g_i
        results.append(dict(start_s=b["s"], start_yaw=b["yaw"], s=s_i,
                            yaw=yaw_i, t=t_i, icp_stats=st_i, tight_score=tight,
                            tight_fwd=tight_fwd, tight_rev=tight_rev,
                            path_ground=g_i, path_obst=o_i))
        print(f"basin s0={b['s']:.3f} yaw0={b['yaw']:.1f} -> ICP s={s_i:.4f} "
              f"yaw={yaw_i:.2f} t={t_i.round(3)} tight({cfg['eps_tight']}m) "
              f"fwd={tight_fwd:.4f} rev={tight_rev:.4f} ground={g_i:.2f} "
              f"score={tight:.4f}")
    if not results:
        raise RuntimeError("no basin survived refinement")
    win = max(results, key=lambda r: r["tight_score"])
    s_icp, yaw_icp, t_icp, icp_stats = win["s"], win["yaw"], win["t"], win["icp_stats"]
    slab = spl[(spl[:, 2] * s_icp > -0.5) & (spl[:, 2] * s_icp < band[1])]
    print(f"ICP winner: s={s_icp:.4f} yaw={yaw_icp:.2f} t={t_icp.round(3)} {icp_stats}")
    if abs(np.log(s_icp / win["start_s"])) > 0.15:
        print("WARNING: ICP moved scale >15% from grid optimum -- inspect!")

    # --- 7. vertical planes. Lidar wall-cell lines give the family
    # directions; wall OFFSETS in both clouds come from 1D profile peaks along
    # each family normal. The registration pins yaw to a fraction of a degree,
    # so the splat is projected at exactly the right angle -- its fuzzy walls
    # defeat independent line fitting (they yield one line and miss the rest).
    fband = cfg["facade_band"]
    wc_l = wall_cells(lic, cell=cfg["cell"], zmin=fband[0], zmax=fband[1],
                      min_cover=args.min_cover)
    lines_l = seq_line_ransac(wc_l, thresh=cfg["line_thresh"],
                              min_in=args.min_line_cells, n_lines=10, iters=2500)
    print(f"lidar wall cells {len(wc_l)} -> {len(lines_l)} lines")
    for l in lines_l:
        print(f"  lidar line: th={l['theta']:7.2f} d={l['d']:8.3f}m "
              f"nin={l['nin']:4d} seg={l['seg']:.2f}m")
    fams = []
    for l in sorted(lines_l, key=lambda x: -x["nin"]):
        for f in fams:
            if ang_diff(l["theta"], f["theta"]) < 8.0:
                f["nlines"] += 1
                break
        else:
            fams.append(dict(theta=l["theta"], nlines=1))
    fams = fams[:4]

    lbf = voxel_down(lic[(lic[:, 2] > fband[0]) & (lic[:, 2] < fband[1])],
                     cfg["cell"] / 2)
    zs = splat_band_mask(spl[:, 2], s_icp, fband, cfg["z_min_u"])
    spl_fac = spl[zs & (dnear < args.facade_crop)]
    print(f"facade band: lidar {len(lbf)} pts, splat {len(spl_fac)} pts")
    binw_l = cfg["line_thresh"]
    binw_s = cfg["line_thresh_splat"]
    families = []
    pairs = []
    for f in fams:
        th_l = refine_angle(lbf[:, :2], f["theta"], binw=binw_l)
        nv_l = np.array([np.cos(np.radians(th_l)), np.sin(np.radians(th_l))])
        # lidar side: LOW threshold + fine separation, so interior surfaces
        # (whiteboards, blinds -- what the splat actually reconstructs in
        # front of featureless walls) are present, and wall/furniture doubles
        # ~0.12 m apart stay resolved
        pk_l = profile_peaks(lbf, nv_l, binw=binw_l, zmin=fband[0],
                             zmax=fband[1], min_sep=2 * binw_l, min_frac=0.03,
                             min_count=max(30, int(0.001 * len(lbf))),
                             min_cover=args.min_cover, max_peaks=10)
        th_s = th_l - yaw_icp
        nv_s = np.array([np.cos(np.radians(th_s)), np.sin(np.radians(th_s))])
        # splat side: HIGH threshold, so the fuzz shoulders of a thick splat
        # wall (sub-peaks at ~20% of the main one) do not spawn fake planes
        pk_s = profile_peaks(spl_fac, nv_s, binw=binw_s,
                             zmin=fband[0] / s_icp, zmax=fband[1] / s_icp,
                             min_sep=3 * binw_s, min_frac=args.splat_peak_frac,
                             min_cover=args.min_cover)
        c = float(t_icp[:2] @ nv_l)
        cand = []
        for si, ps in enumerate(pk_s):
            if not ps["is_vertical"]:
                continue
            mapped = s_icp * ps["offset"] + c
            for li, pl in enumerate(pk_l):
                if not pl["is_vertical"]:
                    continue
                err = abs(mapped - pl["offset"])
                if err < args.off_tol:
                    cand.append((err, si, li, mapped))
        cand.sort()
        matches, used_s, used_l = [], set(), set()
        for err, si, li, mapped in cand:  # greedy mutually-exclusive nearest
            if si in used_s or li in used_l:
                continue
            used_s.add(si)
            used_l.add(li)
            matches.append(dict(splat_offset_units=pk_s[si]["offset"],
                                mapped_m=float(mapped),
                                lidar_offset_m=pk_l[li]["offset"],
                                err_m=float(err),
                                splat_count=pk_s[si]["count"],
                                lidar_count=pk_l[li]["count"]))
        for a in range(len(matches)):
            for b in range(a + 1, len(matches)):
                ma, mb = matches[a], matches[b]
                gap_l = ma["lidar_offset_m"] - mb["lidar_offset_m"]
                gap_s = ma["splat_offset_units"] - mb["splat_offset_units"]
                if abs(gap_l) < args.min_gap or abs(gap_s) * s_icp < args.min_gap:
                    continue
                if gap_l / gap_s <= 0:  # ordering flip = bad correspondence
                    continue
                pairs.append(dict(
                    family_theta_lidar_deg=float(th_l),
                    lidar_offsets_m=[ma["lidar_offset_m"], mb["lidar_offset_m"]],
                    splat_offsets_units=[ma["splat_offset_units"],
                                         mb["splat_offset_units"]],
                    gap_lidar_m=float(abs(gap_l)),
                    gap_splat_units=float(abs(gap_s)),
                    scale=float(gap_l / gap_s)))
        families.append(dict(theta_lidar_deg=float(th_l),
                             theta_splat_deg=float(th_s % 180.0),
                             lidar_peaks=pk_l, splat_peaks=pk_s,
                             matches=matches))
        print(f"family th_l={th_l:.2f} th_s={th_s % 180:.2f}:")
        print("  lidar peaks:", [(round(p['offset'], 3), p['count'],
                                  'V' if p['is_vertical'] else 'h')
                                 for p in pk_l])
        print("  splat peaks:", [(round(p['offset'], 3), p['count'],
                                  'V' if p['is_vertical'] else 'h')
                                 for p in pk_s])
        print(f"  matches: {[(round(m['splat_offset_units'], 2), round(m['lidar_offset_m'], 2), round(m['err_m'], 2)) for m in matches]}")
    print(f"{len(pairs)} parallel plane pairs:")
    for p in pairs:
        print(f"  fam {p['family_theta_lidar_deg']:6.1f} deg: lidar gap "
              f"{p['gap_lidar_m']:.3f} m / splat gap {p['gap_splat_units']:.3f} u "
              f"-> scale {p['scale']:.4f}")

    # --- 8. final scale. Pairs are weighted by their lidar gap: a 0.1 m peak
    # error is 8% on a 1.3 m gap but 0.7% on a 14 m one, so wide pairs carry
    # the estimate. Consistency is judged on the wide subset when it exists.
    pair_vals = np.array([p["scale"] for p in pairs])
    pair_w = np.array([p["gap_lidar_m"] for p in pairs])
    s_final, method = float(s_icp), "registration_icp"
    if len(pair_vals) >= 2:
        order = np.argsort(pair_vals)
        cw = np.cumsum(pair_w[order])
        wmed = float(pair_vals[order][min(len(order) - 1,
                                          np.searchsorted(cw, cw[-1] / 2))])
        big = pair_vals[pair_w >= 3 * args.min_gap]
        subset = big if len(big) >= 2 else pair_vals
        if (subset.max() / subset.min() < 1.12 and
                abs(np.log(wmed / s_icp)) < 0.10):
            s_final, method = wmed, "plane_pairs_gap_weighted_median"
    if method == "registration_icp" and len(pair_vals) >= 1 and \
            abs(np.log(pair_vals[np.argmax(pair_w)] / s_icp)) < 0.08:
        s_final = float(np.mean([pair_vals[np.argmax(pair_w)], s_icp]))
        method = "widest_pair_and_icp_mean"
    # re-polish yaw/translation at the FIXED final scale so the Sim(3) is
    # self-consistent (scale itself is not re-estimated here)
    s_report_repolish = float(s_icp)
    if abs(np.log(s_final / s_icp)) > 1e-6:
        _, yaw_icp, t_icp, _ = icp_refine(
            slab, ltree_full, s_final, yaw_icp, t_icp, cfg["icp_trims"][-3:],
            fix_scale=True)
    print(f"FINAL scale = {s_final:.4f} m/unit ({method}); "
          f"icp={s_icp:.4f}, pairs={pair_vals.round(4).tolist()}")

    # --- 9. compose Sim(3): raw splat -> lidar metric
    r = np.radians(yaw_icp)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1.0]])
    L_s = T4(t=[0, 0, -gd_s]) @ T4(R=R_s) @ T4(R=A)
    S = T4(t=t_icp) @ T4(R=Rz, s=s_final)
    L_l = T4(t=[0, 0, -gd_lc]) @ T4(R=R_l)
    M = np.linalg.inv(L_l) @ S @ L_s
    # numeric check of the composition against the step-by-step pipeline
    chk_raw = sp_raw[rng.choice(len(sp_raw), 200, replace=False)]
    lev = chk_raw @ A.T @ R_s.T
    lev[:, 2] -= gd_s
    direct = transform_pts(lev, s_final, yaw_icp, t_icp)
    direct = (direct + [0, 0, gd_lc]) @ R_l  # inverse leveling: R_l^T @ x
    viaM = (M[:3, :3] @ chk_raw.T).T + M[:3, 3]
    assert np.abs(direct - viaM).max() < 1e-6, "Sim(3) composition mismatch"

    up_in_splat = A.T @ R_s.T @ np.array([0, 0, 1.0])

    # --- 10. validate: NN residuals of transformed splat vs lidar crop
    def residuals(P):
        n = min(len(P), 20000)
        X = P[rng.choice(len(P), n, replace=False)]
        Q = (M[:3, :3] @ X.T).T + M[:3, 3]
        tree = cKDTree(li_crop_raw if len(li_crop_raw) <= 400000 else
                       li_crop_raw[rng.choice(len(li_crop_raw), 400000, replace=False)])
        d, _ = tree.query(Q, k=1)
        cov = d < 2.0  # points landing where the scan has ANY coverage
        return dict(n=int(n), median_m=float(np.median(d)),
                    p90_m=float(np.percentile(d, 90)),
                    frac_lt_0p25m=float((d < 0.25).mean()),
                    frac_lt_0p50m=float((d < 0.50).mean()),
                    frac_in_scan_coverage=float(cov.mean()),
                    median_covered_m=float(np.median(d[cov])) if cov.any()
                    else None)
    spu_s = sp_raw @ A.T
    dpath_s, _ = ctree.query(spu_s[:, :2], k=1)
    spn_s_raw = sp_raw[dpath_s < cfg["path_crop"]]
    spl_s = (spn_s_raw @ A.T) @ R_s.T
    spl_s[:, 2] -= gd_s
    sp_band_raw = spn_s_raw[
        (spl_s[:, 2] * s_final > -0.5) & (spl_s[:, 2] * s_final < band[1])]
    res_band = residuals(sp_band_raw)
    res_all = residuals(spn_s_raw)
    print(f"splat residuals (band): {res_band}")
    print(f"splat residuals (all near-path, incl. floaters): {res_all}")

    ctrl = None
    if cfg["expected_scale"] is not None:
        ok = abs(s_final - cfg["expected_scale"]) <= 0.02
        ctrl = dict(expected_scale=cfg["expected_scale"], recovered=s_final,
                    passed=bool(ok))
        print(f"CONTROL: expected {cfg['expected_scale']} recovered {s_final:.4f} "
              f"-> {'PASS' if ok else 'FAIL'}")

    out = dict(
        dataset=name,
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        inputs=dict(splat=cfg["splat"], lidar=cfg["lidar"],
                    images_bin=cfg["images_bin"],
                    registration_cloud=reg_src,
                    sfm_points=cfg["sfm_points"] if reg_src == "sfm" else None),
        up_axis=dict(chosen=chosen, scores=up_scores,
                     note="splat axis whose ground plane lies below the cameras"),
        ground=dict(
            splat=dict(normal_zup_frame=gn_s.tolist(), d_units=float(gd_s),
                       inliers=int(gin_s),
                       cameras_above_floor_units=cam_h),
            lidar_global=dict(normal=gn_l.tolist(), d_m=float(gd_l),
                              inliers=int(gin_l)),
            lidar_local_crop=dict(normal=gn_lc.tolist(), d_m=float(gd_lc),
                                  inliers=int(gin_lc),
                                  tilt_vs_global_deg=float(tilt))),
        registration=dict(
            coarse=dict(scale=best["s"], yaw_deg=best["yaw"],
                        t_xy_m=[best["tx"], best["ty"]],
                        fwd_inlier_frac=best["fwd"], rev_inlier_frac=best["rev"]),
            basins=[dict(start_scale=r["start_s"], start_yaw_deg=r["start_yaw"],
                         icp_scale=float(r["s"]), icp_yaw_deg=float(r["yaw"]),
                         icp_t_m=[float(v) for v in r["t"]],
                         tight_fwd=r["tight_fwd"], tight_rev=r["tight_rev"],
                         path_ground=r["path_ground"], path_obst=r["path_obst"],
                         tight_score=r["tight_score"]) for r in results],
            adjudication=f"winner = max fwd*rev inlier product at "
                         f"{cfg['eps_tight']} m on structure points, times the "
                         f"fraction of the camera path on scanned ground",
            icp=dict(scale=float(s_icp), yaw_deg=float(yaw_icp),
                     t_m=t_icp.tolist(), **icp_stats)),
        planes=dict(
            lidar_lines_m=[dict(theta_deg=l["theta"], offset=l["d"],
                                cells=l["nin"], seg=l["seg"])
                           for l in lines_l],
            families=families),
        scale_pairs=pairs,
        scale=dict(final_m_per_unit=s_final, method=method,
                   icp_scale=float(s_icp),
                   icp_repolish_at_final=float(s_report_repolish),
                   pair_estimates=pair_vals.tolist(),
                   pair_spread=(float(pair_vals.max() - pair_vals.min())
                                if len(pair_vals) else None)),
        residuals=dict(band=res_band, all_near_path=res_all,
                       note="NN distance (m) of transformed splat points to the "
                            "lidar crop; 'band' excludes floaters outside the "
                            "registration slab"),
        sim3=dict(matrix_splat_to_lidar_metric=[[float(v) for v in row]
                                                for row in M],
                  scale=s_final,
                  gravity_up_in_splat_coords=up_in_splat.tolist(),
                  note="x_lidar = M[:3,:3] @ x_splat + M[:3,3]; lidar frame is "
                       "the raw georeferenced metric frame of lidar_sub.npy"),
        control=ctrl,
    )
    with open(cfg["out"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {cfg['out']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=sorted(PRESETS))
    ap.add_argument("--up-axis", choices=["auto", "-y", "+y"], default="auto")
    ap.add_argument("--register-with", choices=["splat", "sfm"], default=None,
                    help="override the preset's registration cloud")
    ap.add_argument("--facade-crop", type=float, default=None,
                    help="splat units around camera path for facade candidates")
    ap.add_argument("--min-cover", type=float, default=0.45,
                    help="min z-coverage fraction for a wall cell")
    ap.add_argument("--min-line-cells", type=int, default=14)
    ap.add_argument("--off-tol", type=float, default=None,
                    help="plane match offset tolerance, metres")
    ap.add_argument("--splat-peak-frac", type=float, default=None,
                    help="splat profile peak threshold, fraction of max peak")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="min plane-pair gap (m) used for a scale estimate")
    args = ap.parse_args()
    cfg = PRESETS[args.dataset]
    if args.facade_crop is None:
        args.facade_crop = cfg["path_crop"]
    if args.off_tol is None:
        args.off_tol = cfg["off_tol"]
    if args.splat_peak_frac is None:
        args.splat_peak_frac = cfg["splat_peak_frac"]
    run(args.dataset, cfg, args)


if __name__ == "__main__":
    main()
