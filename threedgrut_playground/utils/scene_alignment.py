"""Scene alignment loader: metric scale + gravity from lidar_alignment.json.

Per scene, tools/lidar_scale_align.py writes a lidar_alignment.json whose
sim3 block holds x_metric = M[:3,:3] @ x_scene + M[:3,3], the Sim(3) taking
splat/scene coordinates into the georeferenced metric lidar frame (z-up),
with M[:3,:3] = scale * R, plus the gravity-up unit direction expressed in
scene coordinates. Nothing here assumes which scene axis is up: meetingroom
is -y up, main_campus is +y up, and both load identically.

Two frames matter downstream:
  - the DESIGN frame: scene axes, metric units. Trajectories are designed
    here in metres; metric_to_scene_units / scene_units_to_metric convert
    positions across the render boundary by the scalar scale alone, so
    orientations never change and no axis convention is assumed.
  - the full metric lidar frame, reachable via T_scene_to_metric and its
    inverse, for comparing against georeferenced data.

Gravity: gravity_up_scene is the unit up direction in scene coordinates
(sim3.gravity_up_in_splat_coords, normalized). gravity_w(magnitude) returns
the DOWN vector, -magnitude * up, in scene/design axes, which is exactly
what imu_gen --gravity expects for trajectories designed in that frame.

Loading validates the file: the two scale fields must agree, M[:3,:3]/scale
must be a proper rotation, the up vector must be unit length, and rotating
it into the metric frame must land near +z (the lidar frame's up). A file
that fails any of these raises rather than propagating a bad convention.
"""

import json

import numpy as np

# Rotating gravity-up into the metric z-up frame must land this close to +z.
# The splat ground plane is fit independently of the lidar's, so a residual
# tilt of the order the JSON reports (0.5 to 1.5 deg) is expected; 3 deg
# catches sign and axis-permutation errors without tripping on that.
_UP_TOL_RAD = np.radians(3.0)


class SceneAlignment:
    """Parsed, validated lidar_alignment.json. Build via load()."""

    def __init__(self, path, scale, T_scene_to_metric, gravity_up_scene):
        self.path = path
        self.scale = float(scale)  # metres per scene unit
        self.T_scene_to_metric = T_scene_to_metric  # 4x4 Sim(3)
        self.T_metric_to_scene = np.linalg.inv(T_scene_to_metric)
        self.R_scene_to_metric = T_scene_to_metric[:3, :3] / self.scale
        self.gravity_up_scene = gravity_up_scene  # unit, scene axes
        self.gravity_down_scene = -gravity_up_scene

    # ---- design frame (scene axes, metric units) <-> scene units --------
    def metric_to_scene_units(self, p):
        """Design-frame positions [m] -> scene units (render boundary)."""
        return np.asarray(p, float) / self.scale

    def scene_units_to_metric(self, p):
        """Scene-unit positions -> design frame [m]."""
        return np.asarray(p, float) * self.scale

    # ---- full Sim(3) to and from the georeferenced metric lidar frame ---
    def points_scene_to_metric(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        return p @ self.T_scene_to_metric[:3, :3].T + self.T_scene_to_metric[:3, 3]

    def points_metric_to_scene(self, p):
        p = np.atleast_2d(np.asarray(p, float))
        return p @ self.T_metric_to_scene[:3, :3].T + self.T_metric_to_scene[:3, 3]

    # ---- gravity ---------------------------------------------------------
    def gravity_w(self, magnitude=9.81):
        """World(scene-axes)-frame gravity vector, pointing DOWN [m/s^2]."""
        return self.gravity_down_scene * float(magnitude)

    def meta_dict(self):
        """Provenance stamp for npz metadata."""
        return {"path": self.path, "scale_m_per_unit": self.scale, "gravity_up_scene": self.gravity_up_scene.tolist()}


def load(path):
    """lidar_alignment.json -> SceneAlignment, or raise on inconsistency."""
    with open(path) as f:
        data = json.load(f)
    try:
        sim3 = data["sim3"]
        M = np.asarray(sim3["matrix_splat_to_lidar_metric"], float)
        s = float(sim3["scale"])
        s_final = float(data["scale"]["final_m_per_unit"])
        up = np.asarray(sim3["gravity_up_in_splat_coords"], float)
    except KeyError as e:
        raise ValueError(f"{path}: missing key {e}") from None

    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"{path}: scale {s} not a positive number")
    if abs(s - s_final) > 1e-3 * s:
        raise ValueError(f"{path}: sim3.scale {s} disagrees with " f"scale.final_m_per_unit {s_final}")
    if M.shape != (4, 4) or not np.allclose(M[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"{path}: sim3 matrix is not a 4x4 with [0,0,0,1] " f"bottom row (shape {M.shape})")
    R = M[:3, :3] / s
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        raise ValueError(
            f"{path}: M[:3,:3]/scale is not orthogonal " f"(max dev {np.abs(R @ R.T - np.eye(3)).max():.2e})"
        )
    if abs(np.linalg.det(R) - 1.0) > 1e-5:
        raise ValueError(f"{path}: M[:3,:3]/scale has det " f"{np.linalg.det(R):.6f}, want +1 (proper rotation)")
    if up.shape != (3,) or abs(np.linalg.norm(up) - 1.0) > 1e-3:
        raise ValueError(f"{path}: gravity_up_in_splat_coords norm " f"{np.linalg.norm(up):.6f}, want a unit vector")
    up = up / np.linalg.norm(up)
    up_metric = R @ up
    tilt = np.arccos(np.clip(up_metric[2], -1.0, 1.0))
    if tilt > _UP_TOL_RAD:
        raise ValueError(
            f"{path}: gravity-up rotated into the metric frame is "
            f"{np.degrees(tilt):.2f} deg from +z (maps to {up_metric}); "
            f"the sim3 and gravity fields disagree on the up direction"
        )
    return SceneAlignment(path, s, M, up)


if __name__ == "__main__":
    import sys

    a = load(sys.argv[1])
    print(f"{a.path}")
    print(f"  scale            {a.scale:.6f} m/unit " f"(1 m = {1.0 / a.scale:.6f} units)")
    print(f"  gravity up scene {np.array2string(a.gravity_up_scene, precision=5)}")
    print(f"  gravity_w(9.81)  {np.array2string(a.gravity_w(), precision=4)}")
    print(f"  up in metric     " f"{np.array2string(a.R_scene_to_metric @ a.gravity_up_scene, precision=5)}")
