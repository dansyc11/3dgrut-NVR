"""Headless driver for the four-camera KB4 re-render.

Calls exactly the same underlying methods the GUI buttons call, in the same
order a human would press them:

  Load Calibration            -> novel_view_renderer.load_device()
  Primitives > Quad 1 >
    Reset to 15cm square size -> transform.reset(); sx=1.410; sy=0.825
  Render widget > Camera      -> engine.camera_type = 'KB4'
  Frames Between = 1          -> video_recorder.frames_between_cameras = 1
  Build Orbit Trajectory      -> build_orbit_trajectory(gui)
  Render Device Trajectory
    MCAP                      -> gui.render_mcap_trajectory(poses)

render_mcap_trajectory ends in sys.exit(), which is the normal exit path.
"""
import os
import sys

# Run from the repo root (this file's folder): every path below is repo-relative.
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)

from threedgrut_playground.ps_gui import Playground
from threedgrut_playground.utils.orbit_trajectory import build_orbit_trajectory


def log(msg):
    print(f"[drive] {msg}", flush=True)


pg = Playground(
    gs_object='ply_files/room.ply',
    mesh_assets_folder='threedgrut_playground/assets',
    default_config='apps/colmap_3dgrt.yaml',
    buffer_mode='device2device',
)
log("playground constructed")

# --- Load Calibration -------------------------------------------------- #
pg.novel_view_renderer.calibration_filename = 'vk180.json'
pg.novel_view_renderer.calibration_fullpath = './calibration_files/vk180.json'
pg.novel_view_renderer.load_device()
pg.calibration_loaded = True
pg.distortions = pg.novel_view_renderer.get_cam_distortions()
n_cams = pg.novel_view_renderer.get_camera_count()
log(f"calibration loaded: {n_cams} cameras from ./calibration_files/vk180.json")
log(f"device: {pg.novel_view_renderer.get_device_name_and_serial_no()}")
for i in range(n_cams):
    cam = pg.novel_view_renderer.get_camera_at_index(i)
    fx, fy, cx, cy = cam.get_camera_intrinsics()
    log(f"  cam{i} {pg.novel_view_renderer.get_cam_name_at_index(i)} "
        f"{cam.width}x{cam.height} fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

# --- Quad: Reset to 15cm square size ----------------------------------- #
objs = pg.primitives.objects
key = next(k for k in objs if 'quad' in k.lower())
tr = objs[key].transform
tr.reset()
tr.sx = 1.410
tr.sy = 0.825
pg.primitives.rebuild_bvh_if_needed(force=True, rebuild=False)
pg.is_force_canvas_dirty = True
log(f"quad '{key}' reset: sx={tr.sx} sy={tr.sy} (aspect {tr.sx/tr.sy:.4f}, "
    f"texture aspect 752/440={752/440:.4f})")

# verify by geometry, not by the field
import numpy as np
prim = objs[key].apply_transform()
v = prim.vertices.detach().cpu().numpy()
ext = v.max(axis=0) - v.min(axis=0)
nz = np.sort(ext)[::-1][:2]
log(f"quad world extent {ext} -> {nz[0]:.4f} x {nz[1]:.4f} m, "
    f"aspect {nz[0]/nz[1]:.4f} {'WIDER THAN TALL - OK' if nz[0] > nz[1] else 'SQUARE/TALL - ABORT'}")
assert nz[0] / nz[1] > 1.5, "quad is not wider than tall; refusing to render (GUI trap 1)"

# --- Render widget: camera model --------------------------------------- #
pg.engine.camera_type = 'KB4'
log(f"engine.camera_type = {pg.engine.camera_type!r}")

# --- Frames Between ----------------------------------------------------- #
pg.video_recorder.frames_between_cameras = 1
log(f"frames_between_cameras = {pg.video_recorder.frames_between_cameras}")

# --- Build Orbit Trajectory --------------------------------------------- #
poses = build_orbit_trajectory(pg, flip=False)
pg.trajectory_loaded = True
log(f"orbit built: {len(poses)} poses")
log(f"frames per camera = {pg.video_recorder.get_num_frames(len(poses))}")

# --- confirm the override is really gone before spending an hour -------- #
import inspect
from threedgrut_playground.utils.kaolin_future import fisheye
src = inspect.getsource(fisheye.generate_rays_kb4)
assert 'theta_star = ru\n' not in src, "bug 1 override still present"
assert 'k1 * theta**3' in inspect.getsource(fisheye.estimate_theta_star), "bug 4 typo still present"
log("fisheye.py checks pass: override removed, theta**3 in place")

import threedgrut_playground.ps_gui as gui_mod
oc = inspect.getsource(gui_mod.Playground.render_mcap_trajectory)
assert 'only_cams = None' in oc, "only_cams not None"
log("only_cams = None (all four cameras)")

# --- pre-flight: every camera must carry ITS OWN intrinsics + distortion - #
import json
calib = json.load(open('calibration_files/vk180.json'))
by_id = {cid: c for cid, c in calib['cameraData']}
names = ["CamA", "CamB", "CamC", "CamD"]
for idx, nm in enumerate(names):
    pg.video_recorder.reset_for_new_cam_path_export()
    pg.populate_vid_trajectory(poses[:2], idx=idx)
    c = pg.video_recorder.trajectory[0]
    fx, fy, cx, cy = c.get_camera_intrinsics()
    d = c.distortion_coefficients
    d = [float(x) for x in (d.tolist() if hasattr(d, 'tolist') else d)]
    ref = by_id[idx]
    rK = ref['intrinsicMatrix']
    rd = ref['distortionCoeff']
    ok_i = abs(fx - rK[0][0]) < 1e-3 and abs(fy - rK[1][1]) < 1e-3
    ok_d = all(abs(a - b) < 1e-9 for a, b in zip(d[:6], rd[:6]))
    log(f"  {nm} idx{idx}: fx={fx:.3f}(file {rK[0][0]:.3f}) "
        f"k[0:4]={[round(x,6) for x in d[:4]]} "
        f"(file {[round(x,6) for x in rd[:4]]}) "
        f"d[5]={d[5]:.4f} intr={'OK' if ok_i else 'MISMATCH'} dist={'OK' if ok_d else 'MISMATCH'}")
    assert ok_i and ok_d, f"{nm} carries the wrong camera's parameters"
pg.video_recorder.reset_for_new_cam_path_export()
log("pre-flight pass: all four cameras carry their own intrinsics and distortion")

# --- Render Device Trajectory MCAP -------------------------------------- #
log("starting render")
pg.render_mcap_trajectory(poses)
