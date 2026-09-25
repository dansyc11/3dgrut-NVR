#!/usr/bin/env python3
"""Show the render, the tag detections and the board scene in one Rerun window.

Reads a tags bag for detections, an image bag for the rendered frames, a
trajectory CSV for the eye path, and the board scene file for board geometry.
Logs everything on one frame timeline, then prints detection statistics.

--cam takes one camera (unchanged single-cam layout under cam/...) or a comma
list (cama,camb,camc,camd): each camera then logs under its own subtree
(cams/<name>/image, cams/<name>/image/tags, cams/<name>/stats/...) in ONE
recording, keyed by header.seq on the shared "frame" timeline plus a "stamp"
timeline from header.stampMonotonic, so all cameras tile in one viewer and
scrub together. The 3D scene (boards, eye path) is logged once, shared.
"""

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rerun as rr

sys.path.append("/opt/vilota/messages")
import capnp  # noqa: E402

capnp.add_import_hook()
import image_capnp as VKI  # noqa: E402
import tagdetection_capnp as T  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

PALETTE = [
    (230, 80, 60),
    (60, 180, 230),
    (250, 200, 40),
    (120, 220, 120),
    (220, 120, 220),
    (255, 150, 50),
    (150, 150, 255),
]


def set_frame(i, stamp_ns=None):
    """Set the frame index (and optionally the header stamp timeline).
    The API changed name across Rerun versions."""
    rr.set_time("frame", sequence=i)
    if stamp_ns is not None:
        try:
            rr.set_time("stamp", timestamp=stamp_ns / 1e9)
        except TypeError:
            rr.set_time_nanos("stamp", stamp_ns)


def log_scalar(path, value):
    try:
        rr.log(path, rr.Scalars(value))
    except AttributeError:
        rr.log(path, rr.Scalar(value))


def decode_image(img):
    """Return a numpy array for one vkc.Image, or None."""
    enc = str(img.encoding)
    w, h, step = img.width, img.height, img.step
    if not img.data:
        return None
    buf = np.frombuffer(img.data, dtype=np.uint8)
    if step == 0:
        step = w
    try:
        if enc.endswith("mono8"):
            return buf[: h * step].reshape(h, step)[:, :w]
        if enc.endswith("bgr8"):
            a = buf[: h * step].reshape(h, step)[:, : w * 3].reshape(h, w, 3)
            return a[:, :, ::-1]
        if enc.endswith("yuv420") or enc.endswith("nv12"):
            return buf[: h * step].reshape(-1, step)[:h, :w]
        if enc.endswith("jpeg") or enc.endswith("png"):
            from PIL import Image as PILImage

            return np.asarray(PILImage.open(io.BytesIO(img.data)))
    except ValueError:
        return None
    return None


def rot_matrix(deg):
    """Rotation the engine applies for a primitive's rx, ry, rz (degrees).

    Mirrors kaolin_future/transform.py: Rz @ Ry @ Rx, where Rx and Ry are
    the TRANSPOSES of the right-hand-rule matrices (transform.py:108-131)
    while Rz is the usual one (transform.py:134-146). A textbook Rz Ry Rx
    only agrees when rx = ry = 0, so tilted boards decoded mirrored.
    """
    rx, ry, rz = [np.radians(a) for a in deg]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# The engine draws a board as its Quad primitive: local vertices
# (+-1, +-1, +2.5) scaled by (sx, sy, sz), rotated, then translated
# (mesh_io.py:183-188, transform.py model_matrix = T R S). add_primitive's
# autoscale leaves sz = 0.5 in any scene wider than 5 units
# (engine.py:243-249) and boards.py never writes sz, so a board spawned at
# pos is drawn 2.5 * sz = 1.25 units away along its rotated local +z.
QUAD_Z = 2.5
DEFAULT_SZ = 0.5


def board_half_extents(rows, cols, tag_cm, spacing=0.3):
    """Half width / height of a board with its border, as boards.py:150-155."""
    s = tag_cm / 100.0
    return (
        (cols + spacing * (cols - 1) + 2 * spacing) * s / 2.0,
        (rows + spacing * (rows - 1) + 2 * spacing) * s / 2.0,
    )


def board_outline(entry, specs, sz=DEFAULT_SZ):
    """World corners (5 points, closed) of one scene-file board, where the
    engine actually draws it: pos + R(rot) (+-sx, +-sy, QUAD_Z * sz).

    Scene files size a board either by tag_cm or by hand with sx/sy (the
    quad's half extents, what spawn_from_file writes to the transform).
    """
    mat = entry.get("material", "aprilgrid")
    rows, cols = specs.get(mat, (4, 7))
    if "sx" in entry:
        sx, sy = float(entry["sx"]), float(entry["sy"])
    else:
        sx, sy = board_half_extents(rows, cols, entry.get("tag_cm", 15.0), entry.get("spacing", 0.3))
    z = QUAD_Z * sz
    local = np.array([[-sx, -sy, z], [sx, -sy, z], [sx, sy, z], [-sx, sy, z], [-sx, -sy, z]])
    return local @ rot_matrix(entry.get("rot", [0, 0, 0])).T + np.array(entry["pos"], dtype=float)


def read_board_specs(path):
    """Parse rows and cols per material name out of boards.py BOARD_SPECS."""
    out = {}
    if not Path(path).exists():
        return out
    text = Path(path).read_text()
    for name, rows, cols in re.findall(r'\(\s*"([a-z0-9_]+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*,', text):
        out[name] = (int(rows), int(cols))
    out.setdefault("aprilgrid", (4, 7))
    return out


def log_scene(scene_file, boards_py, traj_csv, sz=DEFAULT_SZ):
    specs = read_board_specs(boards_py)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    if Path(scene_file).exists():
        boards = json.loads(Path(scene_file).read_text()).get("boards", [])
        for i, b in enumerate(boards):
            mat = b.get("material", "aprilgrid")
            rows, cols = specs.get(mat, (4, 7))
            pts = board_outline(b, specs, sz)
            w = float(np.linalg.norm(pts[1] - pts[0]))
            h = float(np.linalg.norm(pts[2] - pts[1]))
            rr.log(
                "world/boards/" + mat,
                rr.LineStrips3D(
                    [pts], colors=[PALETTE[i % len(PALETTE)]], labels=[mat + " " + str(rows) + "x" + str(cols)]
                ),
                static=True,
            )
            print("  board " + mat + "  " + str(round(w, 3)) + " x " + str(round(h, 3)) + " m  at " + str(b["pos"]))

    if Path(traj_csv).exists():
        with open(traj_csv) as fh:
            rows = list(csv.DictReader(fh))
        eye = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
        rr.log("world/eye_path", rr.LineStrips3D([eye], colors=[(180, 180, 180)]), static=True)
        print("  eye path " + str(len(eye)) + " poses")
        return eye
    return None


def process_cam(args, cam, multi, grid_order):
    """Read one camera's images and tags, log them, print its statistics.

    Single-cam mode (multi False) keeps the original layout and keying: entity
    cam/..., stats under stats/..., timeline = per-topic message index. Multi
    mode logs under cams/<cam>/... keyed by header.seq (identical across
    cameras of one frame) and also sets the "stamp" timeline from
    header.stampMonotonic. grid_order is shared across cameras so a grid keeps
    one colour everywhere.
    """
    tag_topic = "S1/" + cam + "/tags"
    img_topic = "S1/" + cam
    prefix = ("cams/" + cam) if multi else "cam"
    stats_prefix = ("cams/" + cam + "/stats") if multi else "stats"

    images = {}
    if args.images and not args.stats_only:
        print("reading images from " + args.images + " (every " + str(args.stride) + ")")
        with open(args.images, "rb") as fh:
            n = 0
            for _, ch, msg in make_reader(fh).iter_messages(topics=[img_topic]):
                if multi:
                    with VKI.Image.from_bytes(msg.data) as m:
                        key = int(m.header.seq)
                        if key % args.stride == 0:
                            arr = decode_image(m)
                            if arr is not None:
                                images[key] = arr
                elif n % args.stride == 0:
                    with VKI.Image.from_bytes(msg.data) as m:
                        arr = decode_image(m)
                        if arr is not None:
                            images[n] = arr
                n += 1
        print("  decoded " + str(len(images)) + " frames")

    grid_hits = Counter()
    per_frame = []
    spans = []
    meta_printed = False
    norm_mode = None

    with open(args.tags, "rb") as fh:
        n = 0
        for _, ch, msg in make_reader(fh).iter_messages(topics=[tag_topic]):
            with T.TagDetections.from_bytes(msg.data) as m:
                if not meta_printed:
                    im = m.image
                    print(
                        "camera "
                        + cam
                        + ": "
                        + str(im.width)
                        + "x"
                        + str(im.height)
                        + "  encoding "
                        + str(im.encoding)
                        + "  exposure "
                        + str(im.exposureUSec)
                        + " us"
                        + "  gain "
                        + str(im.gain)
                    )
                    print("declared grids: " + str(len(m.grids)))
                    for g in m.grids:
                        print(
                            "  gridId "
                            + str(g.gridId)
                            + "  "
                            + str(g.tagRows)
                            + "x"
                            + str(g.tagCols)
                            + "  size "
                            + str(round(g.tagSize, 5))
                        )
                    meta_printed = True

                key = int(m.header.seq) if multi else n
                tags = list(m.tags)
                per_frame.append(len(tags))
                strips, colors, labels = [], [], []
                for t in tags:
                    grid_hits[int(t.gridId)] += 1
                    if int(t.gridId) not in grid_order:
                        grid_order.append(int(t.gridId))
                    pts = np.array(t.pointsPolygon, dtype=np.float32)
                    if pts.size < 8:
                        continue
                    xy = pts[:8].reshape(4, 2)
                    if norm_mode is None:
                        norm_mode = bool(np.max(xy) <= 2.0)
                        print("pointsPolygon is " + ("normalised 0..1" if norm_mode else "pixels"))
                    if norm_mode:
                        xy = xy * np.array([m.image.width, m.image.height])
                    spans.append(float(max(xy.max(0) - xy.min(0))))
                    if not args.stats_only and key in images:
                        strips.append(np.vstack([xy, xy[0]]))
                        gi = grid_order.index(int(t.gridId))
                        colors.append(PALETTE[gi % len(PALETTE)])
                        labels.append(str(t.id))

                if not args.stats_only and key in images:
                    set_frame(key, int(m.header.stampMonotonic) if multi else None)
                    rr.log(prefix + "/image", rr.Image(images[key]))
                    if strips:
                        rr.log(prefix + "/image/tags", rr.LineStrips2D(strips, colors=colors, labels=labels))
                    else:
                        rr.log(prefix + "/image/tags", rr.Clear(recursive=False))
                    log_scalar(stats_prefix + "/tags_per_frame", len(tags))
                    log_scalar(stats_prefix + "/grids_per_frame", len({int(t.gridId) for t in tags}))
            n += 1

    arr = np.array(per_frame)
    print("")
    print("frames " + str(len(arr)) + "   total detections " + str(int(arr.sum())))
    print("empty frames " + str(int((arr == 0).sum())) + " (" + str(round(100.0 * (arr == 0).mean(), 1)) + " percent)")
    print("frames with 4 or more tags " + str(int((arr >= 4).sum())))
    print("per grid:")
    for gid, c in grid_hits.most_common():
        print("  " + str(gid) + "  " + str(c))
    if spans:
        s = np.array(spans)
        print(
            "tag span px: p5 "
            + str(round(float(np.percentile(s, 5)), 1))
            + "  median "
            + str(round(float(np.median(s)), 1))
            + "  p95 "
            + str(round(float(np.percentile(s, 95)), 1))
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", required=True)
    p.add_argument("--images", default=None)
    p.add_argument("--traj", default="")
    p.add_argument("--scene", default="office_scene.json")
    p.add_argument("--boards-py", default="threedgrut_playground/utils/boards.py")
    p.add_argument(
        "--sz",
        type=float,
        default=DEFAULT_SZ,
        help="quad z scale the boards were rendered with " "(engine autoscale: 0.5 in scenes wider than 5 units)",
    )
    p.add_argument(
        "--cam",
        default="camd",
        help="one camera, or a comma list (cama,camb,camc,camd) to " "tile all of them in one recording",
    )
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--save", default="", help="write an .rrd instead of opening a window")
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()

    cams = [c.strip() for c in args.cam.split(",") if c.strip()]
    multi = len(cams) > 1

    if not args.stats_only:
        rr.init("vilota_boards_" + "_".join(cams), spawn=not args.save)
        if args.save:
            rr.save(args.save)
        print("scene:")
        log_scene(args.scene, args.boards_py, args.traj, args.sz)

    grid_order = []  # shared first-seen grid order -> stable colours
    for cam in cams:
        if multi:
            print("\n=== " + cam + " ===")
        process_cam(args, cam, multi, grid_order)

    if args.save:
        print("wrote " + args.save)


if __name__ == "__main__":
    main()
