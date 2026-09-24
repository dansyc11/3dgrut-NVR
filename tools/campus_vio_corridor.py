"""Campus VIO corridor builder + checker (CPU only, no GUI, no renders).

The main_campus capture is NOT a ring: it is one L-shaped walkway,
walked end to end over two days (end A near (16.9, -3.6) m, elbow near
(6.5, -10.4) m, end B near (-9.7, 13.1) m in metric scene axes). The
"closed(ish) loop" is therefore a stadium circuit: out along the
corridor centreline offset half a lane to one side, a half-turn cap at
end B, back along the other half-lane, cap at end A. That keeps the
whole circuit inside the seat tube and gives the periodic waypoint
spline of vio_trajectory.py a genuinely closed circuit.

Speed control uses the spline's own contract: waypoints are uniform in
spline phase (vio_trajectory._PeriodicSpline, uniform knots), so the
rendered speed between knots is STRESS * spacing / h. `build` therefore
places knots by integrating ds/dt = v_target(s) along the circuit and
sampling every h seconds: the waypoint FILE encodes the speed profile,
STRESS scales it, VIO_MOTION_S must equal the printed value so one
phase period is exactly one circuit lap.

Waypoint rows are the 7-float format vio_trajectory._load_waypoints
accepts: x,y,z [metres, scene axes = the SCENE_ALIGNMENT design frame],
then qx,qy,qz,qw = camD-optical camera-to-world (optical z = direction
of travel, optical y = -gravity_up).

Subcommands:
  build  --images-bin ... --alignment ... --out-dir campus_vio/
         writes campus_baseline_waypoints.csv, campus_stress_waypoints.csv,
         campus_overhead_board_units.csv and prints the knob values +
         dry-run command lines.
  check  --npz mcap_outputs/campus_*.npz --images-bin ... --alignment ...
         tube distance to nearest seat (assert < 2 m), height above the
         seat plane / ground, speed / gyro / accel / |specific force|
         stats recomputed from the truth npz.
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threedgrut_playground.utils import scene_alignment  # noqa: E402

RAMP_S = 4.0            # keep equal to vio_trajectory.RAMP_S
G = 9.81

# ---- circuit design constants (metres, seconds) --------------------------
LANE_OFF = 0.8          # half-lane offset of each leg from the centreline
TRIM_END = 1.5          # trim off each corridor end before the caps
CELL = 1.8              # binning cell for the centreline skeleton
SNAP_R = 2.2            # recentre radius: local seat centroid
CHAIN_MAX_STEP = 4.0    # greedy chain gives up beyond this
RESAMPLE = 0.5          # centreline resample step
HEIGHT_R = 2.5          # local seat-height median radius
CAM_ABOVE_FLOOR = None  # read from the alignment json (units)

# BASELINE profile: constant walk, slow through the caps.
BASE_V_STRAIGHT = 1.4
BASE_V_CAP = 0.7
BASE_H = 1.4            # seconds of phase per knot
BASE_STRESS = 1.0
BASE_FPS = 20

# STRESS profile: sprints to 4 m/s, two stop-and-go events, fast caps,
# two head-turn windows, vertical bob. Targets hold at STRESS_STRESS.
STRESS_H = 0.7
STRESS_STRESS = 1.25
STRESS_FPS = 45
STRESS_V_CAP = 2.0   # cap turn rate ~ v / (0.7 x LANE_OFF spline radius)
STRESS_BOB_AMP = 0.10   # vertical bob amplitude on the straights
STRESS_BOB_WAVE = 6.5   # metres per bob cycle
HEAD_TURN_DEG = 45.0
HEAD_TURN_HALF = 2.0    # half-width (m) of a head-turn window

STATIONARY_LEAD_S = 3.0

# EXTREME profile: 20 kph (5.6 m/s) held on the long straight past the
# elbow, two hard stop-and-go events shaped for ~8 m/s^2 peaks (the band
# the IMU spline chain reproduces to <2%), cap turns kept at the STRESS
# calibration's 2.0 m/s, a one-sided
# lane weave on the return leg that reaches 0.9 m past the centreline so
# the figure crosses the outbound lane several times WITHOUT leaving the
# seat tube, and a full-lap vertical sweep across the scene-valid band.
EXT_H = 0.35            # knot interval [s]: ~2.0 m at peak speed, so a
                        #   3.2 m decel event still gets several knots
EXT_STRESS = 1.0        # the waypoint FILE encodes the speeds; no scaling
EXT_FPS = 45
EXT_V_PEAK = 5.6        # 20 kph
EXT_V_BASE = 3.0
EXT_V_CAP = 2.0
EXT_V_STOP = 0.25
EXT_WEAVE_AMP = 1.7     # one-sided swing off the return lane (own lane at
EXT_WEAVE_WAVE = 10.0   #   -0.8 m -> reaches +0.9 m past the centreline)
EXT_WEAVE_EDGE = 3.0    # envelope ramp [m] at the weave window edges
EXT_BAND_M = (0.3, 1.7)  # height above GROUND swept once per lap;
                        #   verified by eye in the playground 2026-09-18
                        #   (coherent to 1.7 m; breakdown ~3.6 m off the
                        #   0.64 m capture plane)

# Aim stand-in for the dry-run --cloud: a small board far OVERHEAD, so
# _pose_waypoint_channels' aim distance |geom.center - eye| stays 25+ m
# (stable orientation spline) and the px-formula's d_min term stays small.
OVERHEAD_Y_M = 25.0


# ---------------------------------------------------------------------------
# COLMAP + frames

def read_images_bin(path):
    """images.bin -> list of (name, C_scene_units(3,)). Points2D skipped."""
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)  # image_id
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            t = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)  # camera_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.seek(npts * 24, 1)
            w, x, y, z = qw, qx, qy, qz
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            out.append((name.decode(), -R.T @ t))
    return out


def load_seats(images_bin, align):
    """All seat positions in METRES (design frame) + gravity-up unit."""
    seats = np.array([c for _, c in read_images_bin(images_bin)])
    return seats * align.scale, align.gravity_up_scene


def horiz(p, gup):
    """Component of p orthogonal to gravity-up (works on (N,3) or (3,))."""
    p = np.asarray(p, float)
    return p - np.outer(p @ gup, gup).reshape(p.shape)


# ---------------------------------------------------------------------------
# Centreline: skeleton chain -> snap to seat centroids -> resample

def centreline(seats, gup):
    ph = horiz(seats, gup)
    # bin on two horizontal axes
    e1 = np.cross(gup, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(gup, e1)
    uv = np.stack([ph @ e1, ph @ e2], 1)
    cells = {}
    for i, (a, b) in enumerate(uv):
        cells.setdefault((int(a // CELL), int(b // CELL)), []).append(i)
    cents = np.array([ph[idx].mean(0) for idx in cells.values()
                      if len(idx) >= 3])
    # end A = the seat furthest along +e1 (the (16.9, -3.6) end)
    start = cents[np.argmin(np.linalg.norm(
        cents - ph[np.argmax(uv[:, 0])], axis=1))]
    chain = [start]
    left = [c for c in cents if np.linalg.norm(c - start) > 1e-9]
    while left:
        d = [np.linalg.norm(c - chain[-1]) for c in left]
        j = int(np.argmin(d))
        if d[j] > CHAIN_MAX_STEP:
            break
        chain.append(left.pop(j))
    chain = np.array(chain)

    def smooth(p, w=3):
        out = p.copy()
        for i in range(len(p)):
            lo, hi = max(0, i - w // 2), min(len(p), i + w // 2 + 1)
            out[i] = p[lo:hi].mean(0)
        return out

    def snap(p):
        out = p.copy()
        for i in range(len(p)):
            d = np.linalg.norm(ph - p[i], axis=1)
            m = d < SNAP_R
            if m.sum() >= 5:
                out[i] = ph[m].mean(0)
        return out

    chain = smooth(snap(smooth(chain)))
    # resample to uniform steps
    seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(s[-1] / RESAMPLE)
    su = np.linspace(0, s[-1], n)
    line = np.stack([np.interp(su, s, chain[:, k]) for k in range(3)], 1)
    # trim the ends
    keep = (su >= TRIM_END) & (su <= s[-1] - TRIM_END)
    line = smooth(line[keep], w=9)
    # heights: local median of seat elevation along gravity, smoothed
    hs = seats @ gup
    hline = np.empty(len(line))
    for i in range(len(line)):
        d = np.linalg.norm(ph - line[i], axis=1)
        m = d < HEIGHT_R
        hline[i] = np.median(hs[m]) if m.sum() >= 5 else np.nan
    ok = ~np.isnan(hline)
    hline = np.interp(np.arange(len(line)), np.arange(len(line))[ok], hline[ok])
    for _ in range(3):
        hline = np.convolve(np.pad(hline, 4, mode="edge"),
                            np.ones(9) / 9, mode="valid")
    return line + np.outer(hline, gup)


# ---------------------------------------------------------------------------
# Stadium circuit + arc-length parametrization

def stadium(line, gup):
    """Centreline (with heights) -> closed circuit polyline + leg map.

    Returns (pts (M,3), s (M,), L, spans) with spans naming the four
    pieces as (kind, s_start, s_end), kind in {legA, capB, legB, capA}.
    """
    t = np.gradient(horiz(line, gup), axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.cross(gup, t)              # left of travel
    out_leg = line + LANE_OFF * nrm
    back_leg = (line - LANE_OFF * nrm)[::-1]

    def cap(centre, frm, outward, k=12):
        """Semicircle frm -> mirror of frm about centre, bulging along
        `outward` (the corridor-end tangent), at centre height."""
        ah = horiz(frm - centre, gup)
        r = np.linalg.norm(ah)
        ah /= r
        side = np.cross(gup, ah)
        if side @ outward < 0:
            side = -side
        pts = []
        for th in np.linspace(0, math.pi, k + 2)[1:-1]:
            pts.append(centre + r * (math.cos(th) * ah
                                     + math.sin(th) * side))
        return np.array(pts)

    capB = cap(line[-1], out_leg[-1], t[-1])
    capA = cap(line[0], back_leg[-1], -t[0])
    pieces = [("legA", out_leg), ("capB", capB),
              ("legB", back_leg), ("capA", capA)]
    pts, spans, s0 = [], [], 0.0
    for kind, p in pieces:
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        pts.append(p)
        spans.append((kind, s0, s0 + seg.sum()))
        s0 += seg.sum() + np.linalg.norm(
            (pieces + pieces[:1])[len(spans)][1][0] - p[-1])
    pts = np.concatenate(pts)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    return pts, s, s[-1] + seg[-1], spans


def circuit_point(pts, s_arr, L, s):
    s = s % L
    i = int(np.searchsorted(s_arr, s, side="right")) - 1
    j = (i + 1) % len(pts)
    s1 = s_arr[i]
    s2 = s_arr[i] + np.linalg.norm(pts[j] - pts[i])
    u = 0.0 if s2 <= s1 else (s - s1) / (s2 - s1)
    return pts[i] * (1 - u) + pts[j] * u


# ---------------------------------------------------------------------------
# Speed plans (real m/s at the shipped STRESS value)

def plan_speed(spans, L, profile):
    """-> v(s) callable and event windows dict."""
    (_, a0, a1), (_, b0, b1), (_, c0, c1), (_, d0, d1) = spans
    keys = []           # (s, v)
    ev = {"stops": [], "turns": []}

    def leg(lo, hi, sprint_frac, stop_frac, turn_frac, base, v_cap):
        Ll = hi - lo
        keys.append((lo, v_cap))
        keys.append((lo + 0.10 * Ll, base))
        for f in sprint_frac:
            keys.append((lo + (f - 0.10) * Ll, base))
            keys.append((lo + f * Ll, profile["v_peak"]))
            keys.append((lo + (f + 0.10) * Ll, base))
        for f in stop_frac:
            sm = lo + f * Ll
            keys.append((sm - 1.6, base))
            keys.append((sm - 0.20, profile["v_stop"]))
            keys.append((sm + 0.20, profile["v_stop"]))
            keys.append((sm + 1.6, base))
            ev["stops"].append(sm)
        for f in turn_frac:
            ev["turns"].append(lo + f * Ll)
        keys.append((hi - 0.08 * Ll, base))
        keys.append((hi, v_cap))

    if profile["kind"] == "baseline":
        leg(a0, a1, [], [], [], BASE_V_STRAIGHT, BASE_V_CAP)
        keys.append((0.5 * (b0 + b1), BASE_V_CAP))
        leg(c0, c1, [], [], [], BASE_V_STRAIGHT, BASE_V_CAP)
        keys.append((0.5 * (d0 + d1), BASE_V_CAP))
    elif profile["kind"] == "extreme":
        # legA carries the sprints: base speed through the elbow zone
        # (~0.32 of the leg), then ramp / HOLD / stop 1 / reaccel / hold /
        # decel into capB, with fractions sized so each 5.6 <-> crawl
        # transition spans ~3.2 m (~8 m/s^2 with the C2 spline's roughly
        # sinusoidal accel shape). legB carries the weave and stop 2.
        vp, vs = profile["v_peak"], profile["v_stop"]
        La, Lc = a1 - a0, c1 - c0
        keys.append((a0, EXT_V_CAP))
        keys.append((a0 + 0.06 * La, EXT_V_BASE))
        keys.append((a0 + 0.38 * La, EXT_V_BASE))   # elbow at base speed
        keys.append((a0 + 0.46 * La, vp))           # ramp over ~2.6 m
        keys.append((a0 + 0.68 * La, vp))           # HOLD ~7.2 m (~1.3 s)
        sm1 = a0 + 0.78 * La                        # stop 1
        keys.append((sm1 - 0.20, vs))
        keys.append((sm1 + 0.20, vs))
        ev["stops"].append(sm1)
        keys.append((a0 + 0.88 * La, vp))           # reaccel over ~3.1 m
        keys.append((a0 + 0.90 * La, vp))           # brief second peak
        keys.append((a1, EXT_V_CAP))                # decel into capB
        keys.append((0.5 * (b0 + b1), EXT_V_CAP))
        keys.append((c0, EXT_V_CAP))
        keys.append((c0 + 0.08 * Lc, EXT_V_BASE))
        sm2 = c0 + 0.50 * Lc                        # stop 2, in the weave
        keys.append((sm2 - 1.4, EXT_V_BASE))
        keys.append((sm2 - 0.20, vs))
        keys.append((sm2 + 0.20, vs))
        keys.append((sm2 + 1.4, EXT_V_BASE))
        ev["stops"].append(sm2)
        keys.append((c1 - 0.08 * Lc, EXT_V_BASE))
        keys.append((c1, EXT_V_CAP))
        keys.append((0.5 * (d0 + d1), EXT_V_CAP))
        ev["weave"] = (c0 + 2.0, c1 - 2.0)
    else:
        leg(a0, a1, [0.30, 0.75], [0.52], [0.20], 3.0, STRESS_V_CAP)
        keys.append((0.5 * (b0 + b1), STRESS_V_CAP))
        leg(c0, c1, [0.62], [0.40], [0.82], 3.0, STRESS_V_CAP)
        keys.append((0.5 * (d0 + d1), STRESS_V_CAP))
    keys.sort()
    ks = np.array([k[0] for k in keys])
    kv = np.array([k[1] for k in keys])

    def v(s):
        s = s % L
        return float(np.interp(s, ks, kv, left=kv[0], right=kv[-1]))
    return v, ev


def march_knots(pts, s_arr, L, v, h_real):
    """Integrate ds/dt = v(s); knots every h_real seconds of real time.

    Returns (knot arc positions, k, lap time T_real). k is chosen so that
    k * h == T_real exactly (h re-derived), keeping the circuit closed with
    no time hiccup at the wrap.
    """
    dt = 0.004
    s, t = 0.0, 0.0
    while True:
        s += v(s) * dt
        t += dt
        if s >= L:
            break
    T_real = t * (1.0 - (s - L) / max(v(L), 1e-9) / t)  # trim overshoot
    k = max(8, int(round(T_real / h_real)))
    h = T_real / k
    knots, s, t_next, t = [0.0], 0.0, h, 0.0
    while len(knots) < k:
        s += v(s) * dt
        t += dt
        if t >= t_next - 0.5 * dt:
            knots.append(min(s, L - 1e-6))
            t_next += h
    return np.array(knots), k, T_real


# ---------------------------------------------------------------------------
# Poses

def rot_to_quat_xyzw(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def knot_poses(pts, s_arr, L, knots, gup, ev, kind):
    """Knot positions + camD-optical camera-to-world quats.

    optical z = travel tangent (yawed inside head-turn windows),
    optical y = -gravity_up, optical x = y cross z.
    """
    rows = []
    eps = 0.35
    for s in knots:
        p = circuit_point(pts, s_arr, L, s)
        t = circuit_point(pts, s_arr, L, s + eps) - \
            circuit_point(pts, s_arr, L, s - eps)
        t = horiz(t, gup)
        t /= np.linalg.norm(t)
        yaw = 0.0
        if kind == "stress":
            for st in ev["turns"]:
                d = s - st
                if abs(d) < HEAD_TURN_HALF:
                    yaw = math.radians(HEAD_TURN_DEG) * \
                        (1.0 - abs(d) / HEAD_TURN_HALF)
            p = p + gup * (STRESS_BOB_AMP
                           * math.sin(2 * math.pi * s / STRESS_BOB_WAVE))
        if kind == "extreme":
            # One-sided lane weave on the return leg. Travel there runs
            # against the centreline direction, so the traveller's right,
            # -cross(gup, t), points from the return lane (-0.8 m) across
            # the centreline toward the outbound lane (+0.8 m); the
            # 0..EXT_WEAVE_AMP swing tops out 0.9 m past the centreline,
            # crossing the outbound path twice per cycle while staying
            # well inside the 2 m seat tube.
            wlo, whi = ev["weave"]
            if wlo < s < whi:
                env = min(s - wlo, whi - s, EXT_WEAVE_EDGE) / EXT_WEAVE_EDGE
                off = 0.5 * EXT_WEAVE_AMP * (1.0 - math.cos(
                    2 * math.pi * (s - wlo) / EXT_WEAVE_WAVE))
                p = p - np.cross(gup, t) * (env * off)
            # Full-lap vertical sweep across the scene-valid band
            # (heights relative to the local seat plane, precomputed in
            # cmd_build from the alignment's camera-above-floor height).
            lo_rel, hi_rel = ev["h_band"]
            p = p + gup * (lo_rel + (hi_rel - lo_rel)
                           * 0.5 * (1.0 - math.cos(2 * math.pi * s / L)))
        if yaw:
            c, sn = math.cos(yaw), math.sin(yaw)
            t = c * t + sn * np.cross(gup, t)
        z = t
        y = -gup
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        R_wc = np.stack([x, y, z], axis=1)
        rows.append(np.concatenate([p, rot_to_quat_xyzw(R_wc)]))
    return np.array(rows)


# ---------------------------------------------------------------------------
# build

def cmd_build(args):
    align = scene_alignment.load(args.alignment)
    seats, gup = load_seats(args.images_bin, align)
    line = centreline(seats, gup)
    L1 = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
    pts, s_arr, L, spans = stadium(line, gup)
    print(f"centreline one-way {L1:.1f} m, stadium circuit {L:.1f} m "
          f"({len(line)} samples, lane offset {LANE_OFF} m)")
    for kind, a, b in spans:
        print(f"  {kind}: s {a:6.1f}..{b:6.1f} m")

    os.makedirs(args.out_dir, exist_ok=True)
    board = []
    centre_h = horiz(seats, gup).mean(0) + (np.median(seats @ gup)
                                            + OVERHEAD_Y_M) * gup
    for dx in (-1.41, 0.0, 1.41):
        for dz in (-0.825, 0.0, 0.825):
            e1 = np.cross(gup, [0.0, 0.0, 1.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(gup, e1)
            board.append(centre_h + dx * e1 + dz * e2)
    board = np.array(board) / align.scale       # file is in SCENE UNITS
    bpath = os.path.join(args.out_dir, "campus_overhead_board_units.csv")
    np.savetxt(bpath, board, delimiter=",",
               header="stand-in aim board, SCENE UNITS, %.1f m overhead; "
                      "pass as --cloud WITH --alignment" % OVERHEAD_Y_M)
    print(f"wrote {bpath}")

    with open(args.alignment) as f:
        raw = json.load(f)
    cam_above = raw["ground"]["splat"]["cameras_above_floor_units"] \
        * align.scale

    for kind, h_real, stress, fps, name, prof in (
            ("baseline", BASE_H, BASE_STRESS, BASE_FPS,
             "campus_baseline_waypoints.csv",
             {"kind": "baseline", "v_peak": 4.0, "v_stop": 0.22}),
            ("stress", STRESS_H, STRESS_STRESS, STRESS_FPS,
             "campus_stress_waypoints.csv",
             {"kind": "stress", "v_peak": 4.0, "v_stop": 0.22}),
            ("extreme", EXT_H, EXT_STRESS, EXT_FPS,
             "campus_extreme_waypoints.csv",
             {"kind": "extreme", "v_peak": EXT_V_PEAK,
              "v_stop": EXT_V_STOP})):
        v, ev = plan_speed(spans, L, prof)
        if kind == "extreme":
            # EXT_BAND_M is above GROUND; the circuit line rides the local
            # seat plane, which sits cam_above above the ground.
            ev["h_band"] = (EXT_BAND_M[0] - cam_above,
                            EXT_BAND_M[1] - cam_above)
        knots, k, T_real = march_knots(pts, s_arr, L, v, h_real)
        rows = knot_poses(pts, s_arr, L, knots, gup, ev, kind)
        motion_s = stress * T_real + RAMP_S / 2.0
        path = os.path.join(args.out_dir, name)
        np.savetxt(
            path, rows, delimiter=",", fmt="%.6f",
            header=(f"{kind}: x,y,z [m, scene axes] + qx,qy,qz,qw "
                    f"(camD-optical cam-to-world)\n"
                    f"run with STRESS={stress:g} VIO_MOTION_S={motion_s:.1f} "
                    f"VIO_FPS={fps} (knots every {T_real / k:.3f} s of lap "
                    f"time, {k} knots, lap {T_real:.1f} s, circuit {L:.1f} m)"))
        n_frames = int(round((STATIONARY_LEAD_S + motion_s) * fps))
        print(f"wrote {path}: {k} knots, lap {T_real:.1f} s, "
              f"STRESS={stress:g} -> VIO_MOTION_S={motion_s:.1f}, "
              f"VIO_FPS={fps}, ~{n_frames} frames "
              f"(+1 tail), laps {stress:g}")
        print(f"  dry-run: STRESS={stress:g} VIO_MOTION_S={motion_s:.1f} "
              f"VIO_FPS={fps} python -m "
              f"threedgrut_playground.utils.vio_trajectory --dry-run "
              f"--waypoints {path} --cloud {bpath} "
              f"--alignment {args.alignment} "
              f"--npz mcap_outputs/campus_{kind}_truth.npz "
              f"--rerun mcap_outputs/campus_{kind}.rrd")


# ---------------------------------------------------------------------------
# check

def cmd_check(args):
    align = scene_alignment.load(args.alignment)
    seats, gup = load_seats(args.images_bin, align)
    with open(args.alignment) as f:
        raw = json.load(f)
    cam_above_floor = raw["ground"]["splat"]["cameras_above_floor_units"] \
        * align.scale

    d = np.load(args.npz, allow_pickle=True)
    p = np.asarray(d["p"], float)
    q = np.asarray(d["q_xyzw_camd"], float)
    t = np.asarray(d["t_ns"], np.int64) * 1e-9
    meta = json.loads(str(d["meta"]))
    fps = float(meta["fps"])
    lead = float(meta["stationary_lead_s"])
    dur = t[-1] - t[0] + 1.0 / fps
    motion = t - t[0] >= lead

    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    length = step[motion[:-1]].sum()
    vel = np.gradient(p, 1.0 / fps, axis=0)
    acc = np.gradient(vel, 1.0 / fps, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    a_mag = np.linalg.norm(acc, axis=1)
    f_spec = acc + G * gup                      # a_world - g, g = -G*gup
    f_mag = np.linalg.norm(f_spec, axis=1)

    w = np.empty(len(p) - 1)
    for i in range(len(p) - 1):
        R = quat_to_rot(q[i]).T @ quat_to_rot(q[i + 1])
        w[i] = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * fps

    # tube: nearest seat, full 3-D, chunked
    dmin = np.empty(len(p))
    for i in range(0, len(p), 256):
        blk = p[i:i + 256]
        dd = np.linalg.norm(blk[:, None, :] - seats[None], axis=2)
        dmin[i:i + 256] = dd.min(1)

    # height above the local seat plane
    ph = horiz(p, gup)
    sh = horiz(seats, gup)
    selev = seats @ gup
    habove = np.empty(len(p))
    for i in range(0, len(p), 256):
        dd = np.linalg.norm(ph[i:i + 256, None, :] - sh[None], axis=2)
        for j in range(dd.shape[0]):
            m = dd[j] < 3.0
            ref = np.median(selev[m]) if m.sum() >= 5 else np.median(selev)
            habove[i + j] = (p[i + j] @ gup) - ref

    m = motion
    mv = m.copy()
    print(f"npz {args.npz}")
    print(f"  duration {dur:.1f} s = {lead:g} s lead + {dur - lead:.1f} s "
          f"motion, {len(p)} frames @ {fps:g} fps")
    print(f"  path length (motion) {length:.1f} m")
    print(f"  speed  m/s   peak {speed[mv].max():.2f}  p95 "
          f"{np.percentile(speed[mv], 95):.2f}  mean {speed[mv].mean():.2f}")
    print(f"  accel  m/s^2 peak {a_mag[mv].max():.2f}  p95 "
          f"{np.percentile(a_mag[mv], 95):.2f}")
    print(f"  |f|    m/s^2 min {f_mag[mv].min():.2f}  max "
          f"{f_mag[mv].max():.2f}   (parked: {G:.2f})")
    print(f"  gyro  rad/s  peak {w[mv[:-1]].max():.2f}  p95 "
          f"{np.percentile(w[mv[:-1]], 95):.2f}")
    print(f"  nearest seat m   max {dmin.max():.2f}  median "
          f"{np.median(dmin):.2f}")
    print(f"  height above seat plane m  min {habove.min():.2f}  max "
          f"{habove.max():.2f}  (seats sit {cam_above_floor:.2f} m above "
          f"the splat floor -> above ground "
          f"{habove.min() + cam_above_floor:.2f}.."
          f"{habove.max() + cam_above_floor:.2f})")
    ok = dmin.max() < 2.0
    print(f"  tube constraint (<2 m): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    # Pass the dataset paths, or set VILOTA_DATASETS to the folder that
    # holds main_campus/ and they default to the files under it.
    root = os.environ.get("VILOTA_DATASETS")
    campus = os.path.join(root, "main_campus") if root else None
    for name in ("build", "check"):
        s = sub.add_parser(name)
        s.add_argument("--images-bin", required=campus is None,
                       default=campus and os.path.join(campus, "colmap", "model", "images.bin"),
                       help="COLMAP images.bin of the campus capture "
                            "(default $VILOTA_DATASETS/main_campus/colmap/model/images.bin)")
        s.add_argument("--alignment", required=campus is None,
                       default=campus and os.path.join(campus, "lidar_alignment.json"),
                       help="campus lidar_alignment.json "
                            "(default $VILOTA_DATASETS/main_campus/lidar_alignment.json)")
        if name == "build":
            s.add_argument("--out-dir", default="campus_vio")
        else:
            s.add_argument("--npz", required=True)
    args = ap.parse_args()
    return cmd_build(args) or 0 if args.cmd == "build" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
