#!/usr/bin/env python3
"""Check a tags MCAP against vk_calibrate's dataset criteria.

vk_calibrate bins detected corners into a 4x4 grid over the image and applies
two count tests before it will calibrate a camera:

    MIN_BUCKET_CORNERS_COUNT = 100    every bucket needs at least 100 corners
    BUCKET_CORNERS_WORST_RATIO = 10   min * 10 must be at least the average

This runs the same tests in seconds, so a trajectory can be judged without a
full calibration run. It also reports the closest approach to each image
corner, which vk_calibrate checks separately.

Usage:
    python dataset_check.py tags_rect.mcap
    python dataset_check.py tags_rect.mcap --topic S1/camd/tags
"""

import argparse
import sys
from collections import defaultdict

import capnp

sys.path.append("/opt/vilota/messages")
capnp.add_import_hook()

import tagdetection_capnp as T  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

MIN_BUCKET_CORNERS_COUNT = 100
BUCKET_CORNERS_WORST_RATIO = 10
CORNER_NORMALISED_THRESHOLD = 0.05
N_CLOSEST = 7

LABELS = "ABCDEFGHIJKLMNOP"
CORNERS = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
CORNER_LABELS = ["A", "M", "P", "D"]


def collect(path, wanted):
    buckets = defaultdict(lambda: [0] * 16)
    near = defaultdict(lambda: [[] for _ in range(4)])

    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            if wanted and channel.topic != wanted:
                continue
            with T.TagDetections.from_bytes(message.data) as m:
                for tag in m.tags:
                    pts = list(tag.pointsPolygon)
                    for i in range(0, len(pts) - 1, 2):
                        x, y = pts[i], pts[i + 1]
                        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
                            continue
                        buckets[channel.topic][int(y * 4) * 4 + int(x * 4)] += 1
                        for c, (cx, cy) in enumerate(CORNERS):
                            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                            near[channel.topic][c].append(d)
    return buckets, near


def report(topic, counts, distances):
    total = sum(counts)
    if total == 0:
        print(f"\n{topic}: no corners")
        return False

    print(f"\n{topic}   {total} corners")
    for r in range(4):
        row = "  ".join(f"{LABELS[r*4+c]} {counts[r*4+c]:6d}" for c in range(4))
        print("  " + row)

    low = min(counts)
    avg = total // 16
    ok = True

    if low < MIN_BUCKET_CORNERS_COUNT:
        print(f"  FAIL  weakest bucket {low} is under {MIN_BUCKET_CORNERS_COUNT}")
        ok = False
    if low * BUCKET_CORNERS_WORST_RATIO < avg:
        need = (avg + BUCKET_CORNERS_WORST_RATIO - 1) // BUCKET_CORNERS_WORST_RATIO
        print(f"  FAIL  weakest bucket {low} against average {avg}, "
              f"needs {need} or more")
        ok = False

    d_norm = []
    for c in range(4):
        ds = sorted(distances[c])[:N_CLOSEST]
        if not ds:
            d_norm.append(9.9)
            continue
        d_norm.append((ds[0] + ds[-1]) / 2)
    print("  closest approach to corners: " + ", ".join(
        f"{CORNER_LABELS[c]} {d_norm[c]:.3f}" for c in range(4)))
    print(f"  (vk_calibrate compares these against {CORNER_NORMALISED_THRESHOLD} "
          f"on a diagonal basis)")

    print("  PASS  count tests" if ok else "  dataset check would fail")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--topic", default=None)
    args = p.parse_args()

    buckets, near = collect(args.input, args.topic)
    if not buckets:
        print("no detections found")
        return
    passed = sum(report(t, buckets[t], near[t]) for t in sorted(buckets))
    print(f"\n{passed}/{len(buckets)} cameras would pass the count tests")


if __name__ == "__main__":
    main()
