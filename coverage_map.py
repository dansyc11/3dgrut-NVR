#!/usr/bin/env python3
"""Print where tag detections actually landed in the image.

The detector stores tag corners normalised 0 to 1, so this bins tag centres
into a grid and draws an ASCII heatmap per camera.

The ramp is logarithmic, and any cell with at least one tag renders as a
visible character. A blank means genuinely zero. A linear ramp hides low but
non-zero cells, which is misleading on data with a strong central peak.

Usage:
    python coverage_map.py tags_all.mcap
    python coverage_map.py tags_all.mcap --bins 16 --topic S1/camd/tags
"""

import argparse
import math
import sys
from collections import defaultdict

import capnp

sys.path.append("/opt/vilota/messages")
capnp.add_import_hook()

import tagdetection_capnp as T  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

RAMP = ".:-=+*#%@"   # index 0 is the lowest non-zero level


def collect(path, wanted, bins):
    grids = defaultdict(lambda: [[0] * bins for _ in range(bins)])
    totals = defaultdict(int)
    outside = defaultdict(int)

    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            if wanted and channel.topic != wanted:
                continue
            with T.TagDetections.from_bytes(message.data) as m:
                for tag in m.tags:
                    pts = list(tag.pointsPolygon)
                    if len(pts) < 8:
                        continue
                    n = len(pts) // 2
                    cx = sum(pts[0::2]) / n
                    cy = sum(pts[1::2]) / n
                    totals[channel.topic] += 1
                    if not (0.0 <= cx < 1.0 and 0.0 <= cy < 1.0):
                        outside[channel.topic] += 1
                        continue
                    grids[channel.topic][min(bins - 1, int(cy * bins))][
                        min(bins - 1, int(cx * bins))] += 1
    return grids, totals, outside


def draw(topic, grid, total, out_of_range, bins):
    flat = [c for row in grid for c in row]
    peak = max(flat) or 1
    zero = sum(1 for c in flat if c == 0)
    thin = sum(1 for c in flat if 0 < c < peak * 0.02)

    print(f"\n{topic}   {total} tags, peak cell {peak}")
    print(f"  empty {zero}/{bins*bins},  under 2% of peak {thin}/{bins*bins}")
    if out_of_range:
        print(f"  {out_of_range} tags fell outside 0..1")

    print("  +" + "-" * (bins * 2) + "+")
    for row in grid:
        out = ""
        for c in row:
            if c == 0:
                out += "  "
            else:
                # Log scale, so low but non-zero cells stay visible.
                frac = math.log1p(c) / math.log1p(peak)
                out += RAMP[min(len(RAMP) - 1, int(frac * len(RAMP)))] * 2
        print("  |" + out + "|")
    print("  +" + "-" * (bins * 2) + "+")

    # Quadrant and edge summary, which the picture alone does not give.
    h = bins // 2
    q = {
        "top-left": sum(grid[r][c] for r in range(h) for c in range(h)),
        "top-right": sum(grid[r][c] for r in range(h) for c in range(h, bins)),
        "bot-left": sum(grid[r][c] for r in range(h, bins) for c in range(h)),
        "bot-right": sum(grid[r][c] for r in range(h, bins) for c in range(h, bins)),
    }
    print("  quadrants: " + ", ".join(f"{k} {v}" for k, v in q.items()))

    border = [grid[r][c] for r in range(bins) for c in range(bins)
              if r in (0, bins - 1) or c in (0, bins - 1)]
    print(f"  outer ring holds {sum(border)} tags "
          f"({round(100*sum(border)/max(total,1),1)}% of all)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--topic", default=None)
    args = p.parse_args()

    grids, totals, outside = collect(args.input, args.topic, args.bins)
    if not grids:
        print("no detections found")
        return
    for topic in sorted(grids):
        draw(topic, grids[topic], totals[topic], outside[topic], args.bins)
    print(f"\nramp: blank = zero, '{RAMP[0]}' lowest, '{RAMP[-1]}' peak (log scale)")


if __name__ == "__main__":
    main()
