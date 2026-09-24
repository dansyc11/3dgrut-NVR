# 3dgrut-NVR at Vilota

This is Vilota's fork of NVIDIA's 3DGRUT. `README.md` is NVIDIA's and covers training
and the upstream tools. This file covers what Vilota added, on branch `refactor-v-device`:

- **Playground device rendering.** The playground renders synthetic captures of a Vilota
  device through its real calibration (KB4 and Double Sphere cameras, from a device
  JSON in `calibration_files/`). It adds AprilGrid board scenes, an orbit trajectory
  for calibration renders, and a VIO trajectory with exact ground truth.
- **Calibration checks.** Tools compare a render or a fitted calibration with the
  device file at corner level.
- **Image-MCAP convertor.** Rendered frames become MCAPs that vk_camera_driver can
  consume, for the VIO loop in vio-sim-offline.
- **Splat training fixes.** Lower-degree SH files load, validation PSNR is
  mask-aware for fisheye datasets, and `--initial_pose` sets the launch camera.

Run every command from the repo root, with the environment activated.

## Install

vk-system (`.deb`) is needed for both environments below. It installs the capnp
schemas in `/opt/vilota/messages`, which every MCAP tool loads, and so does the
playground at startup.

### GPU environment: render and train

Needs Linux, an NVIDIA driver 570 or newer, gcc 14 or older and an X display. The
steps install CUDA 12.8.1 inside the venv, because torch is built for cu128. They
also install the slang compiler 2025.13.2: newer slang releases reject the
playground's kernels.

```bash
sudo apt install -y build-essential git wget curl libgl1-mesa-dev libx11-6
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd 3dgrut-NVR          # cloned as in "The four repos" below
git submodule update --init --recursive thirdparty/tiny-cuda-nn threedgrt_tracer/dependencies/optix-dev
uv venv .venv --python 3.11 --prompt 3dgrut-nvr
wget -O /tmp/cuda_12.8.1_linux.run https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh /tmp/cuda_12.8.1_linux.run --toolkit --toolkitpath="$PWD/.venv/cuda-12.8.1" --silent --no-man-page --override
cat >> .venv/bin/activate <<'EOT'
export CUDA_HOME="$VIRTUAL_ENV/cuda-12.8.1"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
EOT
source .venv/bin/activate
wget -O /tmp/slang-2025.13.2.tgz https://github.com/shader-slang/slang/releases/download/v2025.13.2/slang-2025.13.2-linux-x86_64.tar.gz
tar -xzf /tmp/slang-2025.13.2.tgz -C .venv
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
uv pip install setuptools==78.1.1 slangtorch==1.3.22 plyfile torchmetrics tensorboard fire omegaconf hydra-core \
    scikit-learn wandb polyscope==2.6.1 addict rich kornia opencv-python einops imageio msgpack dataclasses_json \
    tqdm libigl pygltflib matplotlib scipy mcap pycapnp rerun-sdk==0.36.3
uv pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"
```

Check it without rendering:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"     # 2.8.0+cu128 12.8
command -v slangc && slangc -version                                       # .venv/bin/slangc, 2025.13.2
python -c "import kaolin, polyscope, slangtorch, fused_ssim; print('gpu deps ok')"
python -c "import threedgrut_playground.ps_gui, threedgrut.trainer; print('imports ok')"
```

- The first playground launch compiles the CUDA extensions into the torch
  extension cache, which takes a few minutes.
- The `.so` files committed under `threedgrut_playground/` are old CUDA 11
  builds. The playground rebuilds them when they cannot load.
- On hybrid-graphics laptops, prefix GUI launches with
  `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`.

### CPU environment: tools and tests

For everything except rendering and training:

```bash
python3 -m venv .venv-cpu && . .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cpu.html \
    "torch==2.8.0+cpu" "torchvision==0.23.0+cpu" "kaolin==0.18.0" \
    numpy scipy matplotlib "polyscope==2.6.1" mcap pycapnp opencv-python-headless "rerun-sdk==0.36.3" plyfile
```

To run the vio-sim-offline and imu-sim steps from the same environment, add
`pip install -r $IMUSIM/requirements.txt`.

## Tests

All four run on the CPU in either environment and print their checks:

```bash
python tests/test_estimate_theta_star.py   # KB4 inverse (fisheye ray generation)
python tests/test_board_layout.py          # board materials and layouts
python tests/test_orbit_framing.py         # orbit framing against the device fields of view
python tests/test_vio_trajectory.py        # VIO path smoothness, stamps, projection
```

## Workflows

### Calibration loop: render boards, detect, solve, compare

`docs/simulation-calibration-set-up.md` walks through it, including the GUI clicks and
the multi-board scene. The short form, for the sample device (serial 30.02.0104):

```bash
# GPU: in the GUI, load calibration_files/DP180IP-30020104.json,
#      then Trajectory type Orbit -> Build Trajectory -> Render Device Trajectory MCAP
python playground.py --gs_object ply_files/room.ply
mv mcap_outputs/long_final_path.mcap mcap_outputs/orbit_myrun.mcap
cp video_trajectories/test_1.csv mcap_outputs/orbit_myrun_traj.csv   # the poses, for reproject_far
# CPU from here
python fix_mcap_labels.py mcap_outputs/orbit_myrun.mcap myrun_img.mcap \
    --topics S1/cama S1/camb S1/camc S1/camd --serial 30.02.0104
python tools/retime_for_detector.py myrun_img.mcap myrun_img_rt.mcap
CONFIG=offline_tags_all.json \
TOPIC="S1/cama/tags:queued S1/camb/tags:queued S1/camc/tags:queued S1/camd/tags:queued" \
./run_offline_tags.sh myrun_img_rt.mcap myrun_tags.mcap
python dataset_check.py myrun_tags.mcap
vk_calibrate --vbag-path myrun_tags.mcap --cam-types kb4 kb4 kb4 ds \
    --focal-lengths -1 -1 -1 550 --serial-number 30.02.0104 --tag-sizes 0.30 --focal-ratio-prior
python tools/basalt_to_device.py calibration-kb4-kb4-kb4-ds.json \
    --template calibration_files/DP180IP-30020104.json --output fitted.json \
    --compare calibration_files/DP180IP-30020104.json
```

- `retime_for_detector.py` is needed because the export stamps frames 33 ms apart,
  and vk_camera_driver detects tags at most once per ~250 ms of stamp time.
- Multi-board scene: launch with
  `PLAYGROUND_BOARDS=1 BOARD_SCENE=office_scene_v9.json ORBIT_TARGET=grid_off14 ORBIT_DIST=2.5 ARM_REACH=0.5 AIM_SCALE=0.9 AIM_SCALE_KB4=1.3`,
  and give one `--tag-sizes` entry per grid, in the order of `offline_tags_all.json`.

Sample, without a render: the calibration-loop checks on the multi-board detections.

```bash
python dataset_check.py $SAMPLES/office_v9_tags.mcap      # 4/4 cameras would pass the count tests
python coverage_map.py $SAMPLES/office_v9_tags.mcap
```

### Corner-level reprojection check

Given a tags MCAP, the trajectory CSV that drove the render and the scene JSON, this
places every tag corner in 3D, projects it through a calibration, and reports
detected minus projected per camera and board. The default calibration is
`calibration_files/DP180IP-30020104.json`, a sample device calibration.

```bash
python tools/reproject_far.py --tags myrun_tags.mcap --traj mcap_outputs/orbit_myrun_traj.csv --scene office_scene_v9.json
python tools/reproject_far.py ... --calib fitted.json --fitted-calib   # score a vk_calibrate result
python tools/reproject_far.py ... --swap camb,camc [--remove-rotation] [--remove-translation] [--plot swap.png]
python tools/reproject_far.py ... --cross camb,camc                    # cross-camera projection check
python tools/reproject_minimal.py --tags <far500 tags.mcap> --traj <far500 traj.csv> --scene far_500m.json --cam camd
python tools/swap_error_map.py --no-plot                               # CamB/CamC swap error from the calibration alone
python viz_rerun.py --tags myrun_tags.mcap --traj ... --scene office_scene_v9.json --cam camd,camb --save view.rrd
```

- Use `--fitted-calib` for any calibration fitted from detections (a vk_calibrate
  result converted by `basalt_to_device.py`). Leave it off for the device file
  that drove the render: that file describes rays through pixel centres i + 0.5,
  while the detector counts from i.
- `far_10m.json` and `far_500m.json` are single-board far scenes. `reproject_minimal.py` is a short
  teaching copy of the check, for one camera and one unrotated 4x7 board.
- `viz_rerun.py` writes a Rerun recording of the detections and the board outlines;
  without `--save` it opens a viewer.

Sample:

```bash
python tools/reproject_far.py --tags $SAMPLES/far10p4_tags.mcap --traj $SAMPLES/far10p4_traj.csv --scene far_10m.json
# per camera, all boards pooled: median cama 0.388, camb 0.399, camc 0.398, camd 0.392 px
```

### VIO trajectory and image MCAP

```bash
# Design and check a path without the GUI (CPU): writes the ground-truth npz
python -m threedgrut_playground.utils.vio_trajectory --dry-run --npz mcap_outputs/vio_truth.npz \
    [--waypoints campus_vio/campus_baseline_waypoints.csv --cloud campus_vio/campus_overhead_board_units.csv] \
    [--alignment <lidar_alignment.json>]
# Render it (GPU): Trajectory type "VIO trajectory" -> Build Trajectory -> Render Device Trajectory MCAP
PLAYGROUND_CALIB=calibration_files/DP180IP-30020104.json VIO_TRAJ=1 python playground.py --gs_object <scene.ply>
# Convert for vk_camera_driver (CPU), then check every field
python threedgrut_playground/utils/mcap_convertor.py --calib calibration_files/DP180IP-30020104.json \
    --from-mcap mcap_outputs/vio_run_img.mcap --cams CamB CamC CamD --fps 20 --output vio_run_fixed.mcap
python verify_mcap_2b.py --mcap vio_run_fixed.mcap --calib calibration_files/DP180IP-30020104.json --fps 20
```

- `campus_vio/` and `meetingroom_vio/` hold scene profiles: `source` an env file
  before the dry run or the launch, with `VILOTA_DATASETS` set to the folder that
  holds the scene datasets. Each file lists its own dry-run command.
- `python tools/campus_vio_corridor.py build --out-dir campus_vio` regenerates the
  campus waypoint files, and `check --npz <truth.npz>` audits a result.
- The IMU simulation, the replay through vk_vio and the scoring are in vio-sim-offline.

Sample, the meeting-room profile with the sample alignment (all checks pass):

```bash
export VILOTA_DATASETS=$SAMPLES && source meetingroom_vio/meetingroom_aggressive_env.sh
python -m threedgrut_playground.utils.vio_trajectory --dry-run --alignment $SCENE_ALIGNMENT --npz $VIO_TRAJ_NPZ
```

### Metric scale and gravity from a LiDAR scan

```bash
python tools/lidar_scale_align.py meetingroom --datasets <datasets>   # or campus; writes <scene>/lidar_alignment.json
python -m threedgrut_playground.utils.scene_alignment <datasets>/meetingroom/lidar_alignment.json
```

`--datasets` (or `VILOTA_DATASETS`) is the folder holding `meetingroom/` and
`main_campus/`, each with its splat points, LiDAR cloud and COLMAP model. The
trajectory, IMU and conversion steps use the resulting `lidar_alignment.json`
through `SCENE_ALIGNMENT` or `--alignment`.

Sample, loading and checking an alignment file:

```bash
python -m threedgrut_playground.utils.scene_alignment $SAMPLES/meetingroom/lidar_alignment.json
```

### Splat training

Train as in `README.md`, for example:

```bash
python train.py --config-name apps/colmap_3dgut.yaml path=<colmap dataset> out_dir=runs \
    experiment_name=<name> dataset.downsample_factor=4 export_ply.enabled=true
```

- **Masks.** Put `<image stem>_mask.png` next to each image (single channel, 255 =
  valid) and they are used automatically. They zero the loss outside the lens's
  image circle, and validation PSNR is then computed on valid pixels only.
- **Fisheye cull cone.** `FISHEYE_MAX_ANGLE_DEG=105` clamps the fisheye max angle
  (the 3DGUT cull cone) for lenses whose frame corners lie outside the image circle.
- **Lower-degree SH.** PLY files with SH degree below 3 load (they are zero-padded).
  To view a large scene, start the camera at an explicit pose:
  `python playground.py --gs_object <scene.ply> --initial_pose EX EY EZ TX TY TZ UX UY UZ`.
- **Diagnostics.** `tools/fiord_gaussian_census.py` counts dark, opaque Gaussians by
  distance to the training cameras (CPU). `tools/render_pose_diff.py --run_dir <run>`
  compares one validation frame through the playground and the trainer (GPU).

## Inputs and where they come from

| Input | Source |
|---|---|
| Sample scene `room.ply` | v0.1 release of [simulation-calibration](https://github.com/vilota-dev/simulation-calibration), into `ply_files/` |
| Device calibration JSON | `calibration_files/`: `DP180IP-30020104.json` (a sample calibration of a real unit, serial 30.02.0104), `vk180.json`, `dp180_1.json`, `dp180_2.json`, `vkl.json` |
| Board scenes | `office_scene.json`, `office_scene_v9.json`, `far_10m.json`, `far_500m.json` |
| Tags MCAP | `run_offline_tags.sh` (vk-system) on a relabelled, retimed render; samples: `far10p4_tags.mcap`, `office_v9_tags.mcap` in `samples-v1` |
| Trajectory CSV | `video_trajectories/test_1.csv`, written by Build Trajectory; copy it before the next build. Samples: `far10p4_traj.csv`, `office_v9_traj.csv` in `samples-v1` |
| `lidar_alignment.json` | `tools/lidar_scale_align.py`; samples for the meeting room and the main campus in `samples-v1` |
| vk_calibrate result | `calibration-<types>.json` in the folder vk_calibrate ran in |
| Training datasets and scene datasets (COLMAP model, splat points, LiDAR cloud) | your own capture; the meetingroom and main_campus datasets are internal, so ask for them |

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
