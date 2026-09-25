"""Census of dark/opaque Gaussians vs distance to the training cameras.

Re-check after a FIORD retrain that dead-corner supervision is gone:
    python tools/fiord_gaussian_census.py runs/<exp>/<run>/export_last.ply \
        <meetingroom dataset>/model/images.bin

Baseline (unmasked run meetingroom-1409_132541): far-field dark&opaque
(lum<0.1 & opacity>0.5) = 7454 gaussians, 5.6%. A masked retrain should
cut this sharply; the near-camera bucket was clean even unmasked.
CPU only.
"""

import struct
import sys

import numpy as np
from plyfile import PlyData


def read_images_bin_centers(path):
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
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )
            centers.append(-R.T @ t)
    return np.stack(centers)


def main(ply_path, images_bin):
    centers = read_images_bin_centers(images_bin)
    v = PlyData.read(ply_path).elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * np.stack([np.asarray(v[f"f_dc_{i}"]) for i in range(3)], axis=1)
    lum = rgb.clip(0, 1).mean(axis=1)
    N = xyz.shape[0]
    mind = np.empty(N, dtype=np.float32)
    for i in range(0, N, 200000):
        d = np.linalg.norm(xyz[i : i + 200000, None, :] - centers[None], axis=2)
        mind[i : i + 200000] = d.min(axis=1)
    print(f"{N} gaussians, {len(centers)} cameras")
    for tag, m in (("near (<0.3)", mind < 0.3), ("far (rest)", mind >= 0.3)):
        dk = (lum[m] < 0.1) & (opacity[m] > 0.5)
        print(
            f"{tag}: n={m.sum()} ({100 * m.mean():.1f}%) "
            f"opacity_med={np.median(opacity[m]):.3f} lum_med={np.median(lum[m]):.3f} "
            f"dark&opaque={dk.sum()} ({100 * dk.mean():.1f}%)"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
