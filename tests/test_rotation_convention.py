"""The board rotation reproject_far assumes is the rotation the engine renders.

viz_rerun.rot_matrix (used by tools/reproject_far.py to place tag corners) copies
the rotation of the ObjectTransform the playground engine draws boards with. If the
two ever differ, every board with a nonzero rx or ry renders mirrored relative to
the corners reproject_far computes, and the reprojection error jumps for that board
only - easy to mistake for a calibration problem.

Upstream 3dgrut 6f8489d ("Fix Playground ObjectTransform rotations to right handed")
changed threedgrut_playground/utils/transform.py, a copy the engine does not import;
the engine uses utils/kaolin_future/transform.py, which kept the original signs. This
test follows whatever module engine.py imports ObjectTransform from, so an upstream
change to the engine's rotations fails here.

Checks:
  1. every board of every scene file: rot_matrix(rot) == the engine's rotation for rot
  2. grid_off15 and vilota_logo of office_scene_v9.json, the tilted boards of the
     calibration scene, including their board normals R @ (0, 0, -1)
  3. a sweep of single-axis and combined angles, so a sign change fails even if no
     scene file has a tilted board

Run:  python tests/test_rotation_convention.py
"""

import ast
import glob
import importlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import viz_rerun  # noqa: E402

TOL = 1e-6


def engine_object_transform():
    """The ObjectTransform class engine.py imports, without importing the engine."""
    tree = ast.parse(open(os.path.join(REPO, "threedgrut_playground", "engine.py")).read())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == "ObjectTransform" for a in node.names):
            return node.module, importlib.import_module(node.module).ObjectTransform
    raise AssertionError("engine.py no longer imports ObjectTransform; update this test")


def engine_rotation(cls, rot):
    t = cls(device="cpu")
    t.rx, t.ry, t.rz = rot
    return t.rotation_matrix()[:3, :3].double().numpy()


def max_diff(cls, rot):
    return float(np.abs(engine_rotation(cls, rot) - np.asarray(viz_rerun.rot_matrix(rot), float)).max())


def main():
    module, cls = engine_object_transform()
    print(f"engine ObjectTransform: {module}")
    n_pass = 0

    # 1. every board of every scene file
    scenes = sorted(glob.glob(os.path.join(REPO, "*.json")))
    boards = []
    for path in scenes:
        data = json.load(open(path))
        if isinstance(data, dict) and isinstance(data.get("boards"), list):
            boards += [(os.path.basename(path), b["material"], b.get("rot", [0, 0, 0])) for b in data["boards"]]
    assert boards, "no scene files with boards found"
    worst = max(max_diff(cls, rot) for _, _, rot in boards)
    assert worst < TOL, f"rot_matrix differs from the engine by {worst:.2e} on a scene board"
    print(f"PASS  {len(boards)} boards in {len({s for s, _, _ in boards})} scene files: max |diff| {worst:.2e}")
    n_pass += 1

    # 2. the tilted boards of the calibration scene, with their normals
    v9 = json.load(open(os.path.join(REPO, "office_scene_v9.json")))
    for b in v9["boards"]:
        if b["material"] not in ("grid_off15", "vilota_logo"):
            continue
        rot = b.get("rot", [0, 0, 0])
        d = max_diff(cls, rot)
        n_eng = engine_rotation(cls, rot) @ np.array([0.0, 0.0, -1.0])
        n_viz = np.asarray(viz_rerun.rot_matrix(rot), float) @ np.array([0.0, 0.0, -1.0])
        assert d < TOL, f"{b['material']}: rot_matrix differs from the engine by {d:.2e}"
        print(
            f"PASS  {b['material']:<12} rot {rot}: max |diff| {d:.2e}, normal {np.round(n_eng, 4)} == {np.round(n_viz, 4)}"
        )
        n_pass += 1

    # 3. angle sweep: each axis alone, then combined
    rots = [[a, 0, 0] for a in (-60, -25, 30)] + [[0, a, 0] for a in (-45, 20)] + [[0, 0, a] for a in (-90, 45)]
    rots += [[-25, 40, 10], [150.612, 0, -180], [30, -60, 120]]
    worst = max(max_diff(cls, rot) for rot in rots)
    assert worst < TOL, f"rot_matrix differs from the engine by {worst:.2e} in the angle sweep"
    print(f"PASS  angle sweep, {len(rots)} rotations: max |diff| {worst:.2e}")
    n_pass += 1

    print(f"\n{n_pass} checks passed")


if __name__ == "__main__":
    main()
