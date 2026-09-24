#!/usr/bin/env python3
"""Write three MP4 files from a render bag and its tags bag.

  <out>_camera.mp4      the rendered frames only
  <out>_detections.mp4  tag outlines on black, nothing else
  <out>_overlay.mp4     rendered frames with tag outlines and a stats line

Frames pair with detections by header timestamp. If the stamps do not
match, the script falls back to pairing by index and says so.
"""

import argparse
import sys
import numpy as np
import cv2

sys.path.append("/opt/vilota/messages")
import capnp  # noqa: E402
capnp.add_import_hook()
import tagdetection_capnp as T  # noqa: E402
import image_capnp as VKI  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

PALETTE = [(60, 80, 230), (230, 180, 60), (40, 200, 250),
           (120, 220, 120), (220, 120, 220), (50, 150, 255)]


def flat_ints(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in flat_ints(v, prefix + k + "."):
                yield kk, vv
        elif isinstance(v, int):
            yield prefix + k, v


def stamp_of(header):
    best = None
    for k, v in flat_ints(header.to_dict()):
        if "stamp" in k.lower() and v > 1_000_000:
            if best is None or len(k) < len(best[0]):
                best = (k, v)
    return best[1] if best else None


def decode_image(img):
    enc = str(img.encoding)
    w, h, step = img.width, img.height, img.step or img.width
    if not img.data:
        return None
    buf = np.frombuffer(img.data, dtype=np.uint8)
    if enc.endswith("mono8"):
        g = buf[:h * step].reshape(h, step)[:, :w]
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if enc.endswith("bgr8"):
        return buf[:h * step].reshape(h, step)[:, :w * 3].reshape(h, w, 3).copy()
    if enc.endswith("jpeg") or enc.endswith("png"):
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if enc.endswith("yuv420") or enc.endswith("nv12"):
        g = buf[:h * step].reshape(-1, step)[:h, :w]
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return None


def draw_tags(canvas, tags, w, h, colors, thick):
    for t in tags:
        pts = np.array(t.pointsPolygon, dtype=np.float32)
        if pts.size < 8:
            continue
        xy = pts[:8].reshape(4, 2)
        if xy.max() <= 2.0:
            xy = xy * np.array([w, h])
        gid = int(t.gridId)
        if gid not in colors:
            colors[gid] = PALETTE[len(colors) % len(PALETTE)]
        poly = xy.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, colors[gid], thick, cv2.LINE_AA)
        c = xy.mean(0).astype(int)
        cv2.putText(canvas, str(int(t.id)), (int(c[0]) - 8, int(c[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[gid], 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--cam", default="camd")
    p.add_argument("--out", default="office_v5")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args()

    by_stamp, by_index = {}, []
    with open(a.tags, "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages(topics=["S1/" + a.cam + "/tags"]):
            with T.TagDetections.from_bytes(msg.data) as m:
                tags = list(m.tags)
                s = stamp_of(m.header)
            by_index.append(tags)
            if s is not None:
                by_stamp[s] = tags
    stamps = np.array(sorted(by_stamp)) if by_stamp else None
    print("detections:", len(by_index), "messages,", len(by_stamp), "with stamps")

    writers = None
    colors = {}
    n = matched = 0
    W = H = 0
    with open(a.images, "rb") as fh:
        for _, ch, msg in make_reader(fh).iter_messages(topics=["S1/" + a.cam]):
            with VKI.Image.from_bytes(msg.data) as m:
                frame = decode_image(m)
                s = stamp_of(m.header)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            tags = None
            if stamps is not None and s is not None:
                i = np.searchsorted(stamps, s)
                for j in (i - 1, i):
                    if 0 <= j < len(stamps) and abs(int(stamps[j]) - s) < 2_000_000:
                        tags = by_stamp[int(stamps[j])]
                        matched += 1
                        break
            if tags is None:
                tags = by_index[n] if n < len(by_index) else []

            if writers is None:
                W, H = int(w * a.scale), int(h * a.scale)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers = {k: cv2.VideoWriter(a.out + "_" + k + ".mp4", fourcc, a.fps, (W, H))
                           for k in ("camera", "detections", "overlay")}
                print("writing", W, "x", H, "at", a.fps, "fps")

            thick = max(1, int(round(2 * a.scale)))
            det = np.zeros_like(frame)
            draw_tags(det, tags, w, h, colors, thick + 1)
            ovl = frame.copy()
            draw_tags(ovl, tags, w, h, colors, thick)
            grids = len({int(t.gridId) for t in tags})
            cv2.putText(ovl, "frame %d   tags %d   grids %d" % (n, len(tags), grids),
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            for k, img in (("camera", frame), ("detections", det), ("overlay", ovl)):
                if a.scale != 1.0:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                writers[k].write(img)
            n += 1
            if n % 200 == 0:
                print("  ", n, "frames")

    for wr in writers.values():
        wr.release()
    print("frames written:", n, "  matched by stamp:", matched,
          "  by index:", n - matched)
    print("grid colors:", colors)
    for k in ("camera", "detections", "overlay"):
        print("  ", a.out + "_" + k + ".mp4")


if __name__ == "__main__":
    main()
