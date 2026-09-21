#!/usr/bin/env python3
"""Show the render, the tag detections and the board scene in one Rerun window.

Reads a tags bag for detections, an image bag for the rendered frames, a
trajectory CSV for the eye path, and the board scene file for board geometry.
Logs everything on one frame timeline, then prints detection statistics.
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
import tagdetection_capnp as T  # noqa: E402
import image_capnp as VKI  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

PALETTE = [(230, 80, 60), (60, 180, 230), (250, 200, 40), (120, 220, 120),
           (220, 120, 220), (255, 150, 50), (150, 150, 255)]


def set_frame(i):
    """Set the frame index. The API changed name across Rerun versions."""
    rr.set_time("frame", sequence=i)


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
            return buf[:h * step].reshape(h, step)[:, :w]
        if enc.endswith("bgr8"):
            a = buf[:h * step].reshape(h, step)[:, :w * 3].reshape(h, w, 3)
            return a[:, :, ::-1]
        if enc.endswith("yuv420") or enc.endswith("nv12"):
            return buf[:h * step].reshape(-1, step)[:h, :w]
        if enc.endswith("jpeg") or enc.endswith("png"):
            from PIL import Image as PILImage
            return np.asarray(PILImage.open(io.BytesIO(img.data)))
    except ValueError:
        return None
    return None


def rot_matrix(deg):
    rx, ry, rz = [np.radians(a) for a in deg]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
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
    return ((cols + spacing * (cols - 1) + 2 * spacing) * s / 2.0,
            (rows + spacing * (rows - 1) + 2 * spacing) * s / 2.0)


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
        sx, sy = board_half_extents(rows, cols, entry.get("tag_cm", 15.0),
                                    entry.get("spacing", 0.3))
    z = QUAD_Z * sz
    local = np.array([[-sx, -sy, z], [sx, -sy, z], [sx, sy, z],
                      [-sx, sy, z], [-sx, -sy, z]])
    return local @ rot_matrix(entry.get("rot", [0, 0, 0])).T \
        + np.array(entry["pos"], dtype=float)


def read_board_specs(path):
    """Parse rows and cols per material name out of boards.py BOARD_SPECS."""
    out = {}
    if not Path(path).exists():
        return out
    text = Path(path).read_text()
    for name, rows, cols in re.findall(
            r'\(\s*"([a-z0-9_]+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*,', text):
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
            rr.log("world/boards/" + mat,
                   rr.LineStrips3D([pts], colors=[PALETTE[i % len(PALETTE)]],
                                   labels=[mat + " " + str(rows) + "x" + str(cols)]),
                   static=True)
            print("  board " + mat + "  " + str(round(w, 3)) + " x "
                  + str(round(h, 3)) + " m  at " + str(b["pos"]))

    if Path(traj_csv).exists():
        with open(traj_csv) as fh:
            rows = list(csv.DictReader(fh))
        eye = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
        rr.log("world/eye_path", rr.LineStrips3D([eye], colors=[(180, 180, 180)]),
               static=True)
        print("  eye path " + str(len(eye)) + " poses")
        return eye
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", required=True)
    p.add_argument("--images", default=None)
    p.add_argument("--traj", default="")
    p.add_argument("--scene", default="office_scene.json")
    p.add_argument("--boards-py", default="threedgrut_playground/utils/boards.py")
    p.add_argument("--sz", type=float, default=DEFAULT_SZ,
                   help="quad z scale the boards were rendered with "
                        "(engine autoscale: 0.5 in scenes wider than 5 units)")
    p.add_argument("--cam", default="camd")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--save", default="", help="write an .rrd instead of opening a window")
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()

    tag_topic = "S1/" + args.cam + "/tags"
    img_topic = "S1/" + args.cam

    if not args.stats_only:
        rr.init("vilota_boards_" + args.cam, spawn=not args.save)
        if args.save:
            rr.save(args.save)
        print("scene:")
        log_scene(args.scene, args.boards_py, args.traj, args.sz)

    images = {}
    if args.images and not args.stats_only:
        print("reading images from " + args.images + " (every " + str(args.stride) + ")")
        with open(args.images, "rb") as fh:
            n = 0
            for _, ch, msg in make_reader(fh).iter_messages(topics=[img_topic]):
                if n % args.stride == 0:
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
                    print("camera " + args.cam + ": " + str(im.width) + "x"
                          + str(im.height) + "  encoding " + str(im.encoding)
                          + "  exposure " + str(im.exposureUSec) + " us"
                          + "  gain " + str(im.gain))
                    print("declared grids: " + str(len(m.grids)))
                    for g in m.grids:
                        print("  gridId " + str(g.gridId) + "  " + str(g.tagRows)
                              + "x" + str(g.tagCols) + "  size " + str(round(g.tagSize, 5)))
                    meta_printed = True

                tags = list(m.tags)
                per_frame.append(len(tags))
                strips, colors, labels = [], [], []
                for t in tags:
                    grid_hits[int(t.gridId)] += 1
                    pts = np.array(t.pointsPolygon, dtype=np.float32)
                    if pts.size < 8:
                        continue
                    xy = pts[:8].reshape(4, 2)
                    if norm_mode is None:
                        norm_mode = bool(np.max(xy) <= 2.0)
                        print("pointsPolygon is "
                              + ("normalised 0..1" if norm_mode else "pixels"))
                    if norm_mode:
                        xy = xy * np.array([m.image.width, m.image.height])
                    spans.append(float(max(xy.max(0) - xy.min(0))))
                    if not args.stats_only and n in images:
                        strips.append(np.vstack([xy, xy[0]]))
                        gi = list(grid_hits).index(int(t.gridId))
                        colors.append(PALETTE[gi % len(PALETTE)])
                        labels.append(str(t.id))

                if not args.stats_only and n in images:
                    set_frame(n)
                    rr.log("cam/image", rr.Image(images[n]))
                    if strips:
                        rr.log("cam/image/tags",
                               rr.LineStrips2D(strips, colors=colors, labels=labels))
                    else:
                        rr.log("cam/image/tags", rr.Clear(recursive=False))
                    log_scalar("stats/tags_per_frame", len(tags))
                    log_scalar("stats/grids_per_frame",
                               len({int(t.gridId) for t in tags}))
            n += 1

    arr = np.array(per_frame)
    print("")
    print("frames " + str(len(arr)) + "   total detections " + str(int(arr.sum())))
    print("empty frames " + str(int((arr == 0).sum())) + " ("
          + str(round(100.0 * (arr == 0).mean(), 1)) + " percent)")
    print("frames with 4 or more tags " + str(int((arr >= 4).sum())))
    print("per grid:")
    for gid, c in grid_hits.most_common():
        print("  " + str(gid) + "  " + str(c))
    if spans:
        s = np.array(spans)
        print("tag span px: p5 " + str(round(float(np.percentile(s, 5)), 1))
              + "  median " + str(round(float(np.median(s)), 1))
              + "  p95 " + str(round(float(np.percentile(s, 95)), 1)))
    if args.save:
        print("wrote " + args.save)


if __name__ == "__main__":
    main()
