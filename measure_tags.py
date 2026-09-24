#!/usr/bin/env python3
"""Measure whether rendered AprilTags come out square.

AprilTags are physically square. A tag viewed head-on must project to a square
in a correct render. This finds the black tag squares by contour detection and
reports the width to height ratio of each, without needing a tag detector.

A ratio near 1.000 on the most face-on tags means the render is geometrically
sound. A consistent departure from 1.000 means the image is stretched.

Usage:
    python measure_tags.py ecal_pub_sub/CamD/00500.png
    python measure_tags.py ecal_pub_sub/CamD/00500.png --show out.png
"""

import argparse
import math

import cv2
import numpy as np


def find_tags(gray, min_side=25):
    """Return min-area rectangles of dark four-sided blobs."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_side * min_side:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if min(w, h) < min_side:
            continue
        # A tag is a filled quad, so the contour should fill its own box.
        if area < 0.7 * w * h:
            continue
        rects.append((rect, approx.reshape(4, 2).astype(float)))
    return rects


def side_ratio(quad):
    """Return width/height from the mean of opposite side lengths."""
    s = [math.dist(quad[(i + 1) % 4], quad[i]) for i in range(4)]
    return (s[0] + s[2]) / 2, (s[1] + s[3]) / 2


def skew(quad):
    """Return 1.0 when the two diagonals match, lower when the view is oblique."""
    d1 = math.dist(quad[0], quad[2])
    d2 = math.dist(quad[1], quad[3])
    return min(d1, d2) / max(d1, d2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--show", default=None, help="write an annotated copy here")
    p.add_argument("--min-side", type=int, default=25)
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"cannot read {args.image}")
        return
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = find_tags(gray, args.min_side)
    if not found:
        print(f"{args.image}: {w}x{h}, no tags found. "
              f"Try a frame where the board is larger, or lower --min-side.")
        return

    rows = []
    for rect, quad in found:
        wid, hei = side_ratio(quad)
        rows.append((skew(quad), wid / hei, wid, hei, quad))
    rows.sort(reverse=True)

    print(f"{args.image}: {w}x{h}, {len(rows)} tag-like quads")
    print("  most face-on first")
    for sk, ratio, wid, hei, _ in rows[:10]:
        print(f"    diagonals {sk:.3f}   w/h = {ratio:.3f}   "
              f"({wid:6.1f} x {hei:6.1f} px)")

    top = [r[1] for r in rows[:10]]
    print(f"\n  mean w/h over the 10 most face-on: {sum(top)/len(top):.3f}")
    print("  a correct render gives 1.000 for a face-on tag")

    if args.show:
        for _, _, _, _, quad in rows[:20]:
            cv2.polylines(img, [quad.astype(int)], True, (0, 0, 255), 2)
        cv2.imwrite(args.show, img)
        print(f"\n  annotated copy written to {args.show}")


if __name__ == "__main__":
    main()
