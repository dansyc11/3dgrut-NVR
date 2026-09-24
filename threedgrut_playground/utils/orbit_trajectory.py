"""Build an orbit trajectory that serves every camera on the rig.

Poses are grouped by camera. Each group is built in world space around the
target, then stored through add_pose_to_trajectory with that camera's index,
which converts the view into the rig frame. The named camera therefore sees
the target exactly as aimed, and every camera gets direct coverage instead of
whatever falls out of serving CamD alone.

By default the target is ALL boards together: the combined bounding box of
every board primitive in the scene, orbited about its centre at distances
derived from the combined extent, pushed out (min_fit_radius) until every
board corner stays inside a conservative KB4 field of view from every eye at
zero aim. Set ORBIT_TARGET to orbit a single board the old way.

Aim grids are per camera type. The KB4 cameras are 1280x800 with about a 58
degree half field, so wide aims would only point them at nothing. The Double
Sphere camera keeps the wide rectangular grid.
"""
import math
import os
import numpy as np
import polyscope as ps

# Camera indices on the rig and which aim grid each uses.
# ORBIT_CAMS: restrict the orbit to some cameras, e.g. ORBIT_CAMS=3 for
# CamD only. Unset = all four. The render still writes all four topics; this
# only chooses which cameras get poses aimed for them.
_only = os.environ.get("ORBIT_CAMS")
_sel = [int(x) for x in _only.split(",")] if _only else [0, 1, 2, 3]
KB4_CAMS = [c for c in (0, 1, 2) if c in _sel]
DS_CAMS = [c for c in (3,) if c in _sel]

# Viewpoints, shared by all cameras.
ELEVATIONS = [-20.0, 20.0]
AZIMUTH_STEPS = 3
AZIMUTH_SPAN = 90.0
DISTANCE_FACTORS = [0.5, 0.9]

# Aim grids, degrees off the optical axis. Yaw is horizontal.
AIM_KB4 = ([-45.0, -22.0, 0.0, 22.0, 45.0],
           [-28.0, 0.0, 28.0])
AIM_DS = ([-78.0, -58.0, -38.0, -19.0, 0.0, 19.0, 38.0, 58.0, 78.0],
          [-49.0, -33.0, -16.0, 0.0, 16.0, 33.0, 49.0])
MAX_AIM_ANGLE = 95.0

# The scene up axis, from the Up setting in the Render widget (neg_y_up).
UP_AXIS = np.array([0.0, 1.0, 0.0])

# Combined-framing bounds, degrees. Conservative against the KB4 cameras of
# DP180IP-30020104.json, where atan(half_sensor / f) gives half fields of
# 58.0-58.5 (horizontal) and 45.1-45.6 (vertical) degrees - and the fisheye
# only widens that. tests/test_orbit_framing.py re-derives the device bounds
# from the file and asserts these sit below them.
FIT_HALF_FOV_X_DEG = 55.0
FIT_HALF_FOV_Y_DEG = 43.0


def board_frame(gui, name_hint="Quad"):
    """Return the center, normal, width and key of the target primitive.

    The single-board scene names its quad "Quad N". Scene-file boards are
    named after their material. Match order: the caller's hint, then
    ORBIT_TARGET from the environment, then grid_off14 (the mid board),
    then aprilgrid. The old scene matches on the first try, so nothing
    changes there.
    """
    import os
    objs = gui.primitives.objects

    def match(hint):
        return next((k for k in objs if hint.lower() in k.lower()), None)

    hints = [name_hint, os.environ.get("ORBIT_TARGET") or "",
             "grid_off14", "aprilgrid"]
    key = None
    for hint in hints:
        if hint:
            key = match(hint)
            if key is not None:
                if hint != name_hint:
                    print(f"[orbit] no '{name_hint}', aiming at '{key}'")
                break
    if key is None:
        raise ValueError(f"No primitive matching any of {hints}. "
                         f"Have: {list(objs)}")

    prim = objs[key].apply_transform()
    verts = prim.vertices.detach().cpu().numpy()
    center = verts.mean(axis=0)

    extent = verts.max(axis=0) - verts.min(axis=0)
    width = float(np.sort(extent)[-1])

    if prim.vertex_normals is not None:
        normal = prim.vertex_normals.detach().cpu().numpy().mean(axis=0)
    else:
        tris = prim.triangles.detach().cpu().numpy()
        a, b, c = verts[tris[0][0]], verts[tris[0][1]], verts[tris[0][2]]
        normal = np.cross(b - a, c - a)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    return center, normal, width, key


def combined_board_frame(gui, name_hint="Quad"):
    """Frame ALL boards: centre/normal/width of their combined bounding box.

    Gathers every primitive named after a board material (or matching the
    bare-quad hint), skipping the logo. Returns (center, normal, width,
    label, corners, n_boards) with corners the 8 combined-AABB corners.
    Falls back to board_frame when nothing matches.
    """
    from threedgrut_playground.utils.boards import BOARD_NAMES
    objs = gui.primitives.objects
    wanted = [n.lower() for n in BOARD_NAMES if n != "vilota_logo"]
    wanted.append(name_hint.lower())
    keys = [k for k in objs if any(k.lower().startswith(w) for w in wanted)]
    if not keys:
        center, normal, width, key = board_frame(gui, name_hint)
        return center, normal, width, key, None, 1

    all_verts, normals = [], []
    for k in keys:
        prim = objs[k].apply_transform()
        all_verts.append(prim.vertices.detach().cpu().numpy())
        if prim.vertex_normals is not None:
            normals.append(
                prim.vertex_normals.detach().cpu().numpy().mean(axis=0))
    pts = np.concatenate(all_verts, axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    center = (mn + mx) / 2.0
    width = float((mx - mn).max())
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    normal = np.sum(normals, axis=0) if normals else np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(normal)
    normal = normal / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return center, normal, width, f"{len(keys)} boards {keys}", corners, len(keys)


def min_fit_radius(center, normal, corners, start, step=0.05, max_r=50.0):
    """Smallest eye-ring radius from which every corner fits the FOV bounds.

    Walks radii outward and, at each, rebuilds the actual elevation/azimuth
    eye ring and checks every corner against FIT_HALF_FOV_X/Y at zero aim.
    An on-axis formula is not enough: the +-20 degree elevation eyes see the
    outer boards at larger vertical angles than an eye on the axis does.
    """
    tx = math.tan(math.radians(FIT_HALF_FOV_X_DEG))
    ty = math.tan(math.radians(FIT_HALF_FOV_Y_DEG))
    r = max(float(start), step)
    while r < max_r:
        if _ring_fits(center, normal, corners, r, tx, ty):
            return r
        r += step
    return r


def _ring_fits(center, normal, corners, r, tx, ty):
    for eye in _eyes_for_radii(center, normal, [r]):
        view = center - eye
        d = np.linalg.norm(view)
        if d < 1e-9:
            return False
        view = view / d
        right, up = frame_from(view)
        rel = corners - eye
        fwd = rel @ view
        if (fwd <= 1e-6).any():
            return False
        if (np.abs(rel @ right) > fwd * tx).any() \
                or (np.abs(rel @ up) > fwd * ty).any():
            return False
    return True


def frame_from(forward):
    """Return a right and up axis perpendicular to forward."""
    world_up = UP_AXIS
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up


def aim_angles(yaws, pitches):
    """Rectangular aim grid in serpentine order."""
    pairs = []
    flip = False
    for pitch in pitches:
        row = list(yaws)
        if flip:
            row.reverse()
        for yaw in row:
            if math.hypot(yaw, pitch) <= MAX_AIM_ANGLE:
                pairs.append((yaw, pitch))
        flip = not flip
    return pairs


def _eyes_for_radii(center, normal, radii):
    """The serpentine elevation/azimuth eye ring at each radius, in order."""
    fwd = normal
    right, up = frame_from(fwd)
    eyes = []
    flip = False
    for r in radii:
        for el in ELEVATIONS:
            el_rad = math.radians(el)
            steps = list(range(AZIMUTH_STEPS))
            if flip:
                steps.reverse()
            for k in steps:
                frac = k / max(AZIMUTH_STEPS - 1, 1)
                az = math.radians(-AZIMUTH_SPAN / 2.0 + AZIMUTH_SPAN * frac)
                offset = (
                    fwd * (r * math.cos(el_rad) * math.cos(az))
                    + right * (r * math.cos(el_rad) * math.sin(az))
                    + up * (r * math.sin(el_rad))
                )
                eyes.append(center + offset)
            flip = not flip
    return eyes


def eye_positions(center, normal, width, min_radius=None):
    """Camera positions on a serpentine arc in front of the board(s)."""
    # ORBIT_DIST: mean eye distance from the board in metres. Rescales the
    # distance factors so their mean lands on that distance. Unset = old
    # behaviour, which is a fraction of board width.
    factors = list(DISTANCE_FACTORS)
    dist_env = os.environ.get("ORBIT_DIST")
    if dist_env:
        d = float(dist_env)
        mean_f = sum(DISTANCE_FACTORS) / len(DISTANCE_FACTORS)
        factors = [f * d / (width * mean_f) for f in DISTANCE_FACTORS]
        print(f"[orbit] ORBIT_DIST {d:.2f} m -> eye radii "
              f"{[round(width * f, 2) for f in factors]} m")
    radii = [width * f for f in factors]
    if min_radius is not None:
        if dist_env:
            if min(radii) < min_radius:
                print(f"[orbit] WARNING ORBIT_DIST rings "
                      f"{[round(r, 2) for r in radii]} m sit inside the "
                      f"{min_radius:.2f} m all-boards fit radius; outer "
                      "boards will leave the KB4 field of view")
        else:
            pushed = [max(r, min_radius) for r in radii]
            if pushed != radii:
                print(f"[orbit] rings {[round(r, 2) for r in radii]} m -> "
                      f"{[round(r, 2) for r in pushed]} m so every board "
                      "fits the KB4 view from every eye")
            radii = pushed
    eyes = _eyes_for_radii(center, normal, radii)

    # ARM_REACH: the arm moves +-R metres in each cardinal direction from its
    # start. Clamp every viewpoint into that box, centred on the sweep's own
    # centroid, and leave the aims alone. Set ARM_REACH=0.5 to switch on.
    reach = os.environ.get("ARM_REACH")
    if reach:
        r = float(reach)
        arr = np.asarray(eyes)
        home = arr.mean(axis=0)
        clamped = np.clip(arr, home - r, home + r)
        moved = np.linalg.norm(clamped - arr, axis=1)
        print(f"[orbit] ARM_REACH {r:.2f} m box at "
              f"({home[0]:+.2f}, {home[1]:+.2f}, {home[2]:+.2f})")
        print(f"[orbit] {int((moved > 1e-6).sum())} of {len(arr)} viewpoints "
              f"pulled in, max move {moved.max():.2f} m")
        eyes = [row for row in clamped]
    return eyes


def make_pose(center, eye, yaw, pitch):
    """Return an aim point, or None when look_at would be degenerate."""
    view = center - eye
    dist = float(np.linalg.norm(view))
    if dist < 1e-6:
        return None
    view = view / dist
    cam_right, cam_up = frame_from(view)
    aim = (
        center
        + cam_right * (dist * math.tan(math.radians(yaw)))
        + cam_up * (dist * math.tan(math.radians(pitch)))
    )
    look = aim - eye
    n = np.linalg.norm(look)
    if n < 1e-6:
        return None
    if abs(float(np.dot(look / n, UP_AXIS))) > 0.97:
        return None
    return aim


def build_orbit_trajectory(gui, cam_index=None, name_hint="Quad", flip=False):
    """Fill the trajectory, serving every camera on the rig in turn."""
    # ORBIT_TARGET keeps the old single-board orbit. The default frames the
    # combined bounding box of every board in the scene.
    corners, n_boards = None, 1
    if os.environ.get("ORBIT_TARGET"):
        center, normal, width, key = board_frame(gui, name_hint)
    else:
        center, normal, width, key, corners, n_boards = \
            combined_board_frame(gui, name_hint)
    # The centre comes from the DRAWN vertices (apply_transform): the quad's
    # local vertices sit at z=2.5, so a board spawned at tz is drawn at
    # tz + sz*2.5. Do not replace it with the transform.
    # ORBIT_FLIP puts the camera BEHIND the board. The quad normal is local
    # -z and the tracer does not cull back faces, so the tags render
    # mirror-reversed and tag16h5 cannot decode them. Leave it unset.
    if flip or os.environ.get("ORBIT_FLIP"):
        print("[orbit] WARNING ORBIT_FLIP set: camera will be behind the "
              "boards and tags will be mirrored")
        normal = -normal

    min_radius = None
    if corners is not None and n_boards > 1:
        min_radius = min_fit_radius(center, normal, corners,
                                    start=width * min(DISTANCE_FACTORS))
        print(f"[orbit] framing {n_boards} boards, fit radius "
              f"{min_radius:.2f} m")

    nvr = gui.novel_view_renderer
    nvr.create_new_trajectory()

    eyes = eye_positions(center, normal, width, min_radius)
    ez = [float(e[2]) for e in eyes]
    print(f"[orbit] eye z {min(ez):+.2f} .. {max(ez):+.2f}   board z {center[2]:+.2f}")
    # AIM_SCALE: multiply every yaw and pitch. The stock grids assume an eye
    # 0.5 m from the board. At 3 m the same swing throws the board to the
    # fisheye edge where tags shrink below the detector floor. 0.5 is a good
    # start at 3 m. Unset = 1.0 = old behaviour.
    aim_scale = float(os.environ.get("AIM_SCALE", "1.0"))
    # AIM_SCALE_KB4: the same multiplier for the KB4 cameras only. The KB4
    # grid (45/28 deg) is narrower than the DS grid (78/49 deg), so the two
    # camera types need different values. Falls back to AIM_SCALE.
    aim_scale_kb4 = float(os.environ.get("AIM_SCALE_KB4", aim_scale))
    def scaled(grid, s):
        yaws, pitches = grid
        return [y * s for y in yaws], [pt * s for pt in pitches]
    if aim_scale != 1.0 or aim_scale_kb4 != 1.0:
        print(f"[orbit] AIM_SCALE {aim_scale:.2f}, AIM_SCALE_KB4 {aim_scale_kb4:.2f}: "
              f"DS yaw/pitch max {max(abs(a) for a in AIM_DS[0]) * aim_scale:.0f}/"
              f"{max(abs(a) for a in AIM_DS[1]) * aim_scale:.0f} deg, "
              f"KB4 yaw/pitch max {max(abs(a) for a in AIM_KB4[0]) * aim_scale_kb4:.0f}/"
              f"{max(abs(a) for a in AIM_KB4[1]) * aim_scale_kb4:.0f} deg")
    plan = [(c, aim_angles(*scaled(AIM_KB4, aim_scale_kb4))) for c in KB4_CAMS]
    plan += [(c, aim_angles(*scaled(AIM_DS, aim_scale))) for c in DS_CAMS]

    poses = None
    per_cam = {}
    first_pose = {}  # cam -> index of its first pose in the trajectory
    for cam, angles in plan:
        first_pose[cam] = sum(per_cam.values())
        n_added = 0
        flip_seq = False
        for eye in eyes:
            seq = list(reversed(angles)) if flip_seq else angles
            flip_seq = not flip_seq
            for yaw, pitch in seq:
                aim = make_pose(center, eye, yaw, pitch)
                if aim is None:
                    continue
                ps.look_at(eye, aim)
                view_mat = ps.get_view_camera_parameters().get_view_mat()
                poses = nvr.add_pose_to_trajectory(view_mat, cam)
                n_added += 1
        per_cam[cam] = n_added

    total = sum(per_cam.values())
    # Lets the GUI park the rig at the first pose aimed for a given camera.
    gui.orbit_first_pose = first_pose
    print(f"[orbit] target '{key}' center={center} width={width:.3f}m")
    print(f"[orbit] {len(eyes)} viewpoints, per-camera poses: {per_cam}, first pose {first_pose}")
    print(f"[orbit] added {total} poses")
    return poses
