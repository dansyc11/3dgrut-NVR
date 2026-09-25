# 3dgrut-NVR at Vilota

Vilota's fork of NVIDIA's 3DGRUT. `README.md` is NVIDIA's and covers training and the upstream tools. This file covers the renderer's part of Vilota's two simulation processes. The full pipeline guide, both processes end to end, is the README of [vio-sim-offline](https://github.com/vilota-dev/vio-sim-offline) (Vilota access).

Run every command from the repo root, with the environment activated.

# Quick start

The renderer's part of each process:

* **Calibration process** — render AprilGrid boards through a device calibration you already know; the rest of the pipeline detects, solves and compares. Reference (three-board scene, sample device 30.02.0104): focal within 0.08 %, rotation within 0.05°, position within 0.5 mm of the device file.
* **VIO process** — design a trajectory, render along it and write its exact ground truth, then convert the images for vk_camera_driver. Reference (metric meeting room, 24 Sep 2026): aligned position RMS 8.2–16.7 mm over six replays, median 11.0 mm.

Also here: the corner-level reprojection check, metric scale from a LiDAR scan, and fixes for splat training.

## Hardware requirements

You need **an NVIDIA GPU on native Linux** to render and to train.

* **WSL2 and macOS do not work.**
* 12 GB+ VRAM recommended. Reference machine is an RTX 5060 Ti 16 GB, driver 595.

No GPU? The CPU environment runs the tests, the trajectory dry run, the MCAP convertor, the reprojection check, the LiDAR tool and the calibration helper scripts, on the `samples-v1` release or on MCAPs someone else rendered.

## What you need to have ready

| # | What | Process | Repo |
|----|----|----|----|
| 1 | System dependencies | both | apt + NVIDIA driver |
| 2 | `vk-system` — message schemas, detector, playback | both | <https://github.com/vilota-dev/vk-system> |
| 3 | This repo and its Python environment | both | this repo, branch `refactor-v-device` |
| 4 | Scene, calibration file, samples | both | releases of <https://github.com/vilota-dev/simulation-calibration> |

Clone this repo as in [The four repos](#the-four-repos) and export `NVR` there.

### 1. System dependencies

```bash
sudo apt install -y build-essential git wget curl libgl1-mesa-dev libx11-6
```

NVIDIA driver 570 or newer, gcc 14 or older, and an X display for the GUI. No system CUDA is needed, and none is used: step 3 installs CUDA 12.8.1 inside the venv.

Verify:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
gcc --version | head -1
```

Expect your GPU with a driver of 570 or newer, and gcc 14 or older.

### 2. vk-system

The playground loads vk-system's capnp schemas from `/opt/vilota/messages` at startup, for the MCAP export, and every MCAP tool here reads them too. Build vk-system from `main` as in the full pipeline guide (vio-sim-offline README, step 2), then put its tools on `PATH`:

```bash
echo 'export PATH=/opt/vilota/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
ls /opt/vilota/messages/image.capnp /opt/vilota/messages/tagdetection.capnp
```

Expect both files.

### 3. This repo and its Python environment

#### GPU environment: render and train

Needs Linux, an NVIDIA driver 570 or newer, gcc 14 or older and an X display. Upstream's scripts build it: `create_venv_cuda.sh` puts CUDA 12.8.1 inside the venv, because torch is built for cu128, and `install_env_uv.sh` installs torch 2.8.0, kaolin 0.18.0, the tiny-cuda-nn bindings, PPISP, fused-ssim and the slang compiler 2026.5.2. The last line adds what Vilota's tools need on top.

```bash
sudo apt install -y build-essential git wget curl libgl1-mesa-dev libx11-6
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd 3dgrut-NVR          # cloned as in "The four repos" below
FORCE_LOCAL_CUDA=1 CUDA_VERSION=12.8.1 ./scripts/create_venv_cuda.sh 3dgrut-nvr
source .venv/bin/activate
./install_env_uv.sh 3dgrut-nvr
uv pip install mcap pycapnp rerun-sdk==0.36.3 scipy matplotlib
```

> ⚠️ Keep `FORCE_LOCAL_CUDA=1` even with a system CUDA installed. Without it the install uses `/usr/local/cuda`, and it supports only CUDA 11.8, 12.4, 12.6, 12.8 and 13.0: a 13.2 or 13.3 toolkit stops it.

`create_venv_cuda.sh` downloads the 5.4 GB CUDA runfile to `/tmp/cuda_12.8.1_linux.run` and reuses it on a second run. `install_env_uv.sh` initialises the submodules, builds tiny-cuda-nn (several minutes) and ends with its own check.

Verify, without rendering:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
command -v slangc && slangc -version
python -c "import kaolin, polyscope, fused_ssim, tinycudann, ppisp; print('gpu deps ok')"
python -c "import threedgrut_playground.ps_gui, threedgrut.trainer; print('imports ok')"
```

Expect `2.8.0+cu128 12.8`, `.venv/bin/slangc` with `2026.5.2`, `gpu deps ok` and `imports ok`.

* The first playground launch compiles the CUDA extensions into the torch extension cache, which takes a few minutes.
* For the VIO process, add imu-sim's packages: `uv pip install -r $IMUSIM/requirements.txt`.

#### CPU environment: tools and tests

For everything except rendering and training:

```bash
python3 -m venv .venv-cpu && . .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cpu.html \
    "torch==2.8.0+cpu" "torchvision==0.23.0+cpu" "kaolin==0.18.0" \
    numpy scipy matplotlib "polyscope==2.6.1" mcap pycapnp opencv-python-headless "rerun-sdk==0.36.3" plyfile
```

Verify:

```bash
python tests/test_board_layout.py
```

Expect `5 checks passed`. For the VIO process, add imu-sim's packages: `pip install -r $IMUSIM/requirements.txt`.

### 4. Scene, calibration file and samples

* `room.ply` (258 MB), the sample Gaussian-splat scene:

  ```bash
  gh release download v0.1 -R vilota-dev/simulation-calibration -p room.ply -D ply_files --skip-existing
  ```

* `calibration_files/DP180IP-30020104.json` — a sample calibration from a real VK180 unit, serial 30.02.0104. It ships in this repo, next to `vk180.json`, `dp180_1.json`, `dp180_2.json` and `vkl.json`.
* The `samples-v1` release — download as in [Sample data](#sample-data); it sets `SAMPLES`.

`gh` must be logged in with Vilota access: `gh auth login`.

Verify:

```bash
ls -la ply_files/room.ply $SAMPLES/far10p4_tags.mcap
```

Expect both, `room.ply` at 269678220 bytes.

### Check the whole setup

```bash
source .venv/bin/activate          # or .venv-cpu, without the render check
python -c "import threedgrut_playground.ps_gui" && echo "render deps ok"
python -c "import threedgrut_playground.utils.vio_trajectory" && echo "tools ok"
ls /opt/vilota/messages/image.capnp > /dev/null && echo "schemas ok"
ls ply_files/room.ply $SAMPLES/far10p4_tags.mcap > /dev/null && echo "data ok"
python tests/test_estimate_theta_star.py > /dev/null && echo "fisheye ok"
```

Expect every `ok` line.

## Tests

All five run on the CPU in either environment:

```bash
python tests/test_estimate_theta_star.py   # KB4 inverse: fails if the round trip exceeds 1e-5 rad / 0.01 px
python tests/test_board_layout.py          # board materials and layouts
python tests/test_orbit_framing.py         # orbit framing against the device fields of view
python tests/test_vio_trajectory.py        # VIO path smoothness, stamps, projection
python tests/test_rotation_convention.py   # reproject_far's board rotation == the engine's
```

Expect exit 0 from each, and `5 checks passed`, `3 checks passed`, `all vio trajectory checks passed` and `4 checks passed` from the last four. The rotation test guards an upstream quirk: 3dgrut 6f8489d made `utils/transform.py` right-handed, but the engine imports `utils/kaolin_future/transform.py`, which kept the original signs. If upstream ever changes the engine's rotations, this test fails before any tilted board renders mirrored.

---

# The calibration render

The three-board scene `office_scene_v9.json`: three AprilGrids at different depths, which Double Sphere needs to separate focal length from xi. Detection, solve and comparison follow in the full pipeline guide.

## Step 1 — launch

```bash
PLAYGROUND_BOARDS=1 BOARD_SCENE=office_scene_v9.json ORBIT_TARGET=grid_off14 \
ORBIT_DIST=2.5 ARM_REACH=0.5 AIM_SCALE=0.9 AIM_SCALE_KB4=1.3 \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python playground.py --gs_object ply_files/room.ply 2>&1 | tee /tmp/pg.log
```

The two `__NV_*` variables force the NVIDIA GPU on hybrid-graphics machines. Harmless otherwise.

Verify: the startup log.

Expect `[playground] scene file office_scene_v9.json, 4 board(s)`, then one line each for `aprilgrid`, `grid_off14`, `grid_off15` and `vilota_logo` with its `pos`, `rot`, `sx`, `sy`.

## Step 2 — GUI, in this exact order

1. **Novel View from Vilota Calibration file** → pick `DP180IP-30020104.json` → **Load Calibration**.
2. **Select CamD**. Its label must show xi and alpha. `Not Fisheye` next to **Select CamA** is normal.
3. **Render** → **Camera** → **Double Sphere Fisheye**, the default. CamA/B/C still render KB4.
4. **Record Trajectory Video** → Ctrl+click **Frames Between** → `1`.
5. **Save/Load Video trajectory** → **Trajectory type** → **Orbit** → **Build Trajectory**.
6. **Render Device Trajectory MCAP**. The app exits when done.

> ⚠️ Do **not** press **Reset to 30cm square size** in this mode: the scene file sizes the boards, and the reset doubles them.

Verify:

```bash
mcap info mcap_outputs/long_final_path.mcap
```

Expect `S1/cama..camd`, 1295 msgs each at 30 Hz. Rename it and copy `video_trajectories/test_1.csv` (the poses, for the reprojection check) before the next build.

## Scene files

`BOARD_SCENE` places each board: `material` (`aprilgrid`, `grid_off14`, `grid_off15`, `grid_off16`, `grid_2x2`, `grid_3x1`, `grid_3x3`, `vilota_logo`), `pos` in metres, optional `rot` in degrees, and `tag_cm` or half extents `sx`, `sy`. `PLAYGROUND_BOARDS=1` without a scene file spawns the default three boards. The orbit knobs (`ORBIT_TARGET`, `ORBIT_DIST`, `ARM_REACH`, `AIM_SCALE`, `AIM_SCALE_KB4`, `ORBIT_CAMS`) are listed in `docs/simulation-calibration-set-up.md`.

---

# The VIO render

## Step 1 — check the trajectory on the CPU

```bash
python -m threedgrut_playground.utils.vio_trajectory --dry-run \
    --alignment $SAMPLES/meetingroom/lidar_alignment.json --npz /tmp/<run>_design.npz
```

The dry run builds the same path the GUI builds, checks it and writes the ground truth. `--alignment` designs it in metres.

Expect a summary table, `verdict: PASS` and an `npz:` line naming the file.

## Step 2 — build the trajectory and render

```bash
SCENE_ALIGNMENT=$SAMPLES/meetingroom/lidar_alignment.json \
PLAYGROUND_BOARDS=1 PLAYGROUND_CALIB=calibration_files/DP180IP-30020104.json \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python playground.py --gs_object <meeting-room splat>.ply
```

The meeting-room splat is internal: ask for it. With `room.ply`, leave out `SCENE_ALIGNMENT`; the path is then in scene units.

1. **Novel View from Vilota Calibration file** → pick `DP180IP-30020104.json` → **Load Calibration**.
2. **Save/Load Video trajectory** → **Trajectory type** → **VIO trajectory** → **Build Trajectory**. The console prints the summary and writes `mcap_outputs/vio_truth.npz` before any GPU work.
3. **Record Trajectory Video** → **Frames Between** → `1`.
4. **Render Device Trajectory MCAP**.

```bash
mv mcap_outputs/long_final_path.mcap mcap_outputs/<run>_img.mcap
cp mcap_outputs/vio_truth.npz mcap_outputs/<run>_truth.npz
```

Verify:

```bash
mcap info mcap_outputs/<run>_img.mcap
```

Expect `S1/cama..camd`, 860 msgs each at 20 Hz, a ~43 s span starting at 1 s.

## Step 3 — convert for vk_camera_driver

The driver reads the calibration from inside the image messages. The converter embeds it, stamps the frames in real time and keeps the three cameras VIO uses.

```bash
python threedgrut_playground/utils/mcap_convertor.py --calib calibration_files/DP180IP-30020104.json \
    --from-mcap mcap_outputs/<run>_img.mcap --cams CamB CamC CamD --fps 20 --output <run>_fixed.mcap
python verify_mcap_2b.py --mcap <run>_fixed.mcap --calib calibration_files/DP180IP-30020104.json --fps 20
```

> ⚠️ `--fps` must match the render: 20 unless you set `VIO_FPS`.

Expect `ALL SECTION-2b CHECKS PASSED`. The IMU, merge, replay and scoring steps are in the full pipeline guide.

## Scene profiles

`campus_vio/` (BASELINE, STRESS, EXTREME) and `meetingroom_vio/` hold ready-made profiles as env files. `source` one with `VILOTA_DATASETS` set; each lists its own dry-run command:

```bash
export VILOTA_DATASETS=$SAMPLES && source meetingroom_vio/meetingroom_aggressive_env.sh
python -m threedgrut_playground.utils.vio_trajectory --dry-run --alignment $SCENE_ALIGNMENT --npz $VIO_TRAJ_NPZ
```

Expect `verdict: PASS` for the meeting room. The campus profiles end with `verdict: 6 check(s) failed` and exit 1 by design: all six are board-visibility checks, and the corridor has no boards. `python tools/campus_vio_corridor.py build --out-dir campus_vio` regenerates the campus waypoint files; it needs the internal campus dataset under `VILOTA_DATASETS`.

---

# The reprojection check

`tools/reproject_far.py` places every tag corner of a render in 3D, projects it through a calibration, and compares with the detections, per camera and board. It needs the tags MCAP, the trajectory CSV that drove the render and the scene JSON; `--calib` defaults to `calibration_files/DP180IP-30020104.json`.

```bash
python tools/reproject_far.py --tags $SAMPLES/far10p4_tags.mcap --traj $SAMPLES/far10p4_traj.csv --scene far_10m.json
```

Expect, per camera: cama 0.388, camb 0.399, camc 0.398, camd 0.392 px median. That is the floor: detection noise with the true calibration.

Variants, on the same inputs:

```bash
python tools/reproject_far.py <inputs> --calib fitted.json --fitted-calib        # score a vk_calibrate result
python tools/reproject_far.py <inputs> --swap camb,camc [--remove-rotation] [--remove-translation] [--plot swap.png]
python tools/reproject_far.py <inputs> --cross camb,camc                          # cross-camera projection
python tools/reproject_minimal.py <inputs> --cam camd                             # short teaching copy, one camera
python tools/swap_error_map.py --no-plot                                          # swap error from the calibration alone
python viz_rerun.py <inputs> --cam camd,camb --save view.rrd                      # detections and boards in Rerun
```

`<inputs>` is `--tags … --traj … --scene …`. `--fitted-calib` is for calibrations fitted from detections: the render's rays pass through pixel i + 0.5 and the detector counts from i, and a fitted calibration has already absorbed that half pixel. `tools/basalt_to_device.py` turns a vk_calibrate result into a device file for `--calib`.

---

# Metric scale from a LiDAR scan

A COLMAP splat has no real scale. `tools/lidar_scale_align.py` aligns the splat with the scene's LiDAR scan and writes `lidar_alignment.json`: the Sim(3) from splat units to metres and the gravity direction.

```bash
python tools/lidar_scale_align.py meetingroom --datasets <datasets>     # or campus
python -m threedgrut_playground.utils.scene_alignment $SAMPLES/meetingroom/lidar_alignment.json
```

`<datasets>` holds `meetingroom/` or `main_campus/` with the splat points, the LiDAR cloud and the COLMAP model (internal). The second line loads and checks an alignment.

> ⚠️ The first line overwrites `<datasets>/<scene>/lidar_alignment.json`. Truth npz files record the alignment they were built with, so keep a copy of the old file first.

Expect `scale            0.505297 m/unit (1 m = 1.979035 units)` for the meeting room.

---

# Splat training

Train as in `README.md`, for example:

```bash
python train.py --config-name apps/colmap_3dgut.yaml path=<colmap dataset> out_dir=runs \
    experiment_name=<name> dataset.downsample_factor=4 export_ply.enabled=true
```

* **Image folder layout.** The loader opens `images_<N>/<name>` with the name exactly as `images.bin` stores it, subfolders included; the meeting room and the main campus store `cam1/IMG_…_fisheye1.png` and `cam2/IMG_…_fisheye2.png`. A flat `images_4/` gives `Image … not found` for every frame. Both datasets keep their flat files and add `cam1/` and `cam2/` folders of relative symlinks to them, masks included, so older checkouts, which look up the basename, load them too. For another dataset, mirror `images.bin`'s subfolders the same way; for these two cameras:

  ```bash
  cd <dataset>/images_4 && mkdir cam1 cam2
  for f in *fisheye1*; do ln -s "../$f" "cam1/$f"; done
  for f in *fisheye2*; do ln -s "../$f" "cam2/$f"; done
  ```

* **Masks.** Put `<image stem>_mask.png` next to each image, in the same subfolder (single channel, 255 = valid). They zero the loss outside the lens's image circle, and validation PSNR counts valid pixels only.
* **Fisheye cull cone.** `FISHEYE_MAX_ANGLE_DEG=105` clamps the fisheye max angle for lenses whose frame corners lie outside the image circle.
* **Lower-degree SH.** PLY files with SH degree below 3 load (zero-padded). Start the camera at a pose for large scenes: `--initial_pose EX EY EZ TX TY TZ UX UY UZ`.
* **Diagnostics.** `tools/fiord_gaussian_census.py <export.ply> <images.bin>` (CPU) and `tools/render_pose_diff.py --run_dir <run>` (GPU).

---

# Troubleshooting

| What you see | Why | Fix |
|----|----|----|
| `ModuleNotFoundError: No module named 'image_capnp'` | vk-system schemas missing | install vk-system; check `/opt/vilota/messages/image.capnp` |
| `ERROR: Unsupported CUDA version: 13.3` from `install_env_uv.sh` | it picked the system CUDA | run `create_venv_cuda.sh` with `FORCE_LOCAL_CUDA=1` first, as in step 3 |
| `exec: -title: not found` while CUDA extracts | the runfile reopens itself in an xterm when `DISPLAY` is set and there is no terminal | run step 3 in a terminal, or `unset DISPLAY` first |
| Extensions build against the wrong CUDA | venv not activated, so `CUDA_HOME` is a system CUDA | `source .venv/bin/activate` |
| `python -m …vio_trajectory` fails with `No module named 'threedgrut_playground'` | run by file path, or not from the repo root | `python -m threedgrut_playground.utils.vio_trajectory` from the repo root |
| boards render at double size | reset button pressed in scene-JSON mode | relaunch; the scene file sizes the boards |
| cameras missing from the render | `only_cams` in `ps_gui.py` set to a subset | set it to `None` |
| verifier fails `frame period == 1/fps` | `--fps` doesn't match the render | reconvert with the render's rate |
| `reproject_far.py`: `the following arguments are required` | `--tags`, `--traj`, `--scene` are required | pass all three |
| fitted calibration reads 0.2–0.3 px worse than it should | half pixel counted twice | add `--fitted-calib` |
| `lidar_scale_align.py`: `the following arguments are required: --datasets` | no datasets folder given | `--datasets <dir>` or `VILOTA_DATASETS` |

---

# Using a different device

Put the device file in `calibration_files/`, load it in the GUI and pass it to every `--calib` and to `PLAYGROUND_CALIB`. Each camera renders from its `cameraType`: 1 = KB4 (`intrinsicMatrix`, `distortionCoeff[0:4]`), 0 = Double Sphere (`distortionCoeff[5:11]`). The full pipeline guide lists the solver flags that change with it.

### Known limits

> ⚠️ **The loader knows four rig names, not files.** `novel_view_renderer.py` picks the camera layout from the file's device name (`DP180-`, `DP180IP`) or product name (`VK180…`, `VKL-`). Another product needs one more case and its camera-name map.

> ⚠️ **Only 4-camera rigs have been tested.**

> ⚠️ **Metric only with an alignment.** Without `SCENE_ALIGNMENT` / `--alignment`, trajectories are in scene units.

> ⚠️ **The render is cleaner than reality** — no motion blur, no vibration above the pose band. Use the results for comparisons, not as field-accuracy promises.

# Notes

* Rename every render output immediately. The next render overwrites `mcap_outputs/long_final_path.mcap`, and the next build overwrites `mcap_outputs/vio_truth.npz` and `video_trajectories/test_1.csv`.
* Image MCAPs are big (1–4 GB) and regenerable. The tags MCAP, the trajectory CSV and the truth npz carry the evidence.
* One calibration file per run, everywhere.

---

## The four repos

| Repo | What it holds |
|---|---|
| [3dgrut-NVR](https://github.com/vilota-dev/3dgrut-NVR), branch `refactor-v-device` | The renderer. Its playground renders synthetic captures of a Vilota device through the device's real calibration. It also holds splat training, the calibration setup guide, the reprojection tools, the VIO trajectory generator and the image-MCAP convertor. |
| [simulation-calibration](https://github.com/vilota-dev/simulation-calibration) | Release data: the sample scene `room.ply` (v0.1) and sample inputs for the CPU workflows (`samples-v1`). Also older copies of the calibration helper scripts. |
| [imu-sim](https://github.com/vilota-dev/imu-sim) | Synthetic IMU data from a pose trajectory: spline fit, IMU samples, vk-system IMU MCAP. |
| [vio-sim-offline](https://github.com/vilota-dev/vio-sim-offline) | The VIO loop after the render: simulate the IMU with imu-sim, merge it with the image MCAP, replay through vk_camera_driver and vk_vio, score against the ground truth. |

They serve two loops:

- **Calibration:** render AprilGrid boards (3dgrut-NVR), detect the tags (vk-system), solve (vk_calibrate), and compare the result with the device file (3dgrut-NVR `tools/basalt_to_device.py` and `tools/reproject_far.py`).
- **VIO:** design and render a trajectory (3dgrut-NVR writes the ground-truth npz and the image MCAP), simulate the IMU and merge (vio-sim-offline with imu-sim), replay through vk_vio, and score (vio-sim-offline `ate_compare.py`).

Not in these repos:

- vk-system (`.deb`): `vk_camera_driver`, `vk_record`, `vk_playback`, `vk_mcap_to_rrd`, and the capnp schemas in `/opt/vilota/messages` that every MCAP tool reads.
- vk_calibrate, the calibration solver.
- vk-vio (`.deb`), which installs `/opt/vilota/bin/vk_vio`.
- eCAL 6.0, which the vk-system tools talk over.
- An NVIDIA GPU, for rendering and training only.

vk-system installs its tools in `/opt/vilota/bin`, which is not on `PATH` by default.
The scripts call them by name, so add it (for example in `~/.bashrc`):

```bash
export PATH=/opt/vilota/bin:$PATH
```

Clone the four side by side. The commands in these READMEs use the three variables below.

```bash
git clone -b refactor-v-device https://github.com/vilota-dev/3dgrut-NVR.git
git clone https://github.com/vilota-dev/simulation-calibration.git
git clone https://github.com/vilota-dev/imu-sim.git
git clone https://github.com/vilota-dev/vio-sim-offline.git
export NVR=$PWD/3dgrut-NVR IMUSIM=$PWD/imu-sim VSO=$PWD/vio-sim-offline
```

### Sample data

The `samples-v1` release of simulation-calibration holds rendered tag detections with
their trajectories, and the scene alignment files of the meeting room and the main
campus. With them the CPU workflows run without a GPU render or the internal datasets.
Download it once (the GitHub CLI must be logged in with Vilota access), into a folder
laid out like a datasets folder:

```bash
gh release download samples-v1 -R vilota-dev/simulation-calibration -D samples
mkdir -p samples/meetingroom samples/main_campus
mv samples/meetingroom_lidar_alignment.json samples/meetingroom/lidar_alignment.json
mv samples/main_campus_lidar_alignment.json samples/main_campus/lidar_alignment.json
export SAMPLES=$PWD/samples
```

The sample commands in these READMEs use `$SAMPLES`. It also works as `VILOTA_DATASETS`
for the scene profiles.
