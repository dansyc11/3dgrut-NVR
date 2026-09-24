# Simulation-Calibration Set Up

# Quick start

Render a camera whose parameters you already know, detect AprilTags in the render, calibrate, and check you get your numbers back. 

## Hardware requirements

You need **an NVIDIA GPU on native Linux**. Not a preference, a hard requirement:

* **WSL2 and MacOS does not work.** 
* 12 GB+ VRAM recommended. Reference machine is an RTX 5060 Ti 16 GB.

No GPU? You can still do stages 2–4 (relabel, detect, solve) on a tag MCAP someone else rendered then skip to step 4.

## What you need to have ready

| # | What | Repo |
|----|----|----|
| 1 | System dependencies | apt + CUDA |
| 2 | `vk-system` — detector, recorder, viewer | <https://github.com/vilota-dev/vk-system> |
| 3 | `vk_calibrate` — the solver (**needs vk-system installed first**) | <https://github.com/vilota-dev/vk_calibrate> |
| 4 | `3dgrut-NVR` — the renderer | <https://github.com/vilota-dev/3dgrut-NVR> |
| 5 | Scene + calibration file | calibration: `calibration_files/` in 3dgrut-NVR; scene: v0.1 release of <https://github.com/vilota-dev/simulation-calibration> |

### 1. System dependencies

```bash
sudo apt update

sudo apt install -y build-essential cmake autoconf git curl

sudo apt install -y libopencv-dev libprotobuf-dev libtbb-dev libboost-serialization-dev
```

eCAL 6.0 (the message transport the VK tools use):

```bash

sudo add-apt-repository -y ppa:ecal/ecal-6.0

sudo apt-get update

sudo apt-get install -y ecal
```

Rust, for the vk-system build:

```bash

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source ~/.bashrc

rustup install 1.79.0

rustup default 1.79.0
```

NVIDIA driver + CUDA. On RTX 50-series (Blackwell, sm_120) you need **CUDA 12.8 or newer**:

```bash

nvidia-smi          # driver present, note the CUDA version

nvcc --version      # toolkit present
```

### 2. vk-system

You need `vk_camera_driver`, `vk_record` and `vk_mcap_to_rrd`.

**Fast path:** vk-system publishes `.deb` packages on its Releases tab. If there's a release matching your Ubuntu version, install that and skip the build.

**From source:**

```bash

git clone https://github.com/vilota-dev/vk-system.git

cd vk-system

git submodule update --recursive --init
./pre_build.py

mkdir build && cd build

cmake .. -DVK-SYSTEM_BUILD=ON

make -j$(nproc)          # lower the number on a weaker machine

cd ..
./build_tools.py -j$(nproc) --package
```

If compilation fails on a fresh checkout, it's almost always a missing dependency from step 1 — install and re-run.

To produce `.deb` packages, use `make package` from the build folder rather than calling `cpack` directly (it resolves the CMake version variables properly):

```bash

cd build && make package
```

The package installs its tools in `/opt/vilota/bin`, which is not on `PATH` by default. Add it, for example in `~/.bashrc`, so the commands below and `run_offline_tags.sh` find them:

```bash
export PATH=/opt/vilota/bin:$PATH
```

Verify:

```bash

which vk_camera_driver vk_record vk_mcap_to_rrd

ls /opt/vilota/messages/image.capnp
```

All four must resolve. `fix_mcap_labels.py` reads that schema path directly.

> You do **not** need vk-manager (the web dashboard). That's for configuring a physical DP180IP.

### 3. vk_calibrate

> Build vk-system **first**. This project depends on components vk-system installs.

Clone recursively — it has submodules:

```bash

git clone --recurse-submodules https://github.com/vilota-dev/vk_calibrate.git

cd vk_calibrate
```

Extra dependencies on top of step 1:

```bash

sudo apt-get update

sudo apt install -y libeigen3-dev libzstd-dev liblz4-dev
```

Build and install:

```bash

mkdir build && cd build

cmake ..
make -j$(nproc)          # lower the number on a weaker machine

make package

sudo dpkg -i *.deb
```

Verify:

```bash

vk_calibrate --help
```

Help lists `eucm, ds, kb4, pinhole`. It also accepts `pinhole-radtan8`, which is real but undocumented. Anything else crashes with a bare `std::runtime_error`.

#### Optional: the Streamlit UI

A web frontend for the same solver. It reproduces the terminal results digit for digit, so it's a convenience, not a requirement.

```bash

python -m venv venv

source venv/bin/activate

sudo apt install -y python3-pybind11

pip3 install sophuspy --no-binary sophuspy --force-reinstall

pip install -r requirements.txt

cd python

streamlit run visualise_calibration_plot.py
```

> ⚠️ The UI's presets pick their own flags and may ignore the focal prior you type. Always read the `running command:` line it prints — that's what actually ran.

### 4. 3dgrut-NVR (the renderer)

```bash
git clone https://github.com/vilota-dev/3dgrut-NVR.git
cd 3dgrut-NVR
git checkout refactor-v-device
```

Follow that repo's README for the build. It creates its own venv, and the exact commands change with CUDA versions so a copy here would go stale. Two things it doesn't say loudly enough:

* On **Blackwell (RTX 50-series, sm_120)** use `CUDA_VERSION=12.8.1` and the `WITH_GCC11` flag. Older notes saying 50-series is unsupported are out of date.
* Note where the venv lands. Every command below needs it active.

Verify:

```bash

source /path/to/3dgrut/.venv/bin/activate

python -c "import kaolin, polyscope; print('render deps ok')"
```

If kaolin says `no kernel image is available`, it was built for the wrong GPU architecture. Rebuild with the CUDA version above.

### 5. Scene and calibration file

* `DP180IP-30020104.json` — a sample calibration from a real VK180 unit. It ships in `calibration_files/`, next to `vk180.json`.
* `room.ply` (258 MB) — the Gaussian splat scene the examples use. Download it from the v0.1 release of <https://github.com/vilota-dev/simulation-calibration> (Vilota access) into `<3dgrut-NVR>/ply_files/`, or use any 3DGRUT-trained splat.

**The examples below use** `DP180IP-30020104.json`, but nothing in the pipeline is specific to it. Any device calibration file the loader supports will work — see [Using a different device](#using-a-different-device). Pick one file and keep it consistent: the same file drives the render *and* is what you compare the result against.

### 6. Helper scripts

The relabel, detection and check scripts used below (`fix_mcap_labels.py`, `run_offline_tags.sh`, `offline_tags_*.json`, `dataset_check.py`, `coverage_map.py`) ship in the 3dgrut-NVR root, and every command runs from there. The simulation-calibration repo keeps older copies: do not copy them over these, because its `offline_tags_all.json` declares only one grid.

### Check the whole setup

Before taking a while (15-20 mins) on a render:

```bash

source /path/to/3dgrut/.venv/bin/activate

python -c "import kaolin, polyscope; print('render deps ok')"
which vk_camera_driver vk_record && echo "vk tools ok"
which vk_calibrate && echo "solver ok"
ls ply_files/room.ply calibration_files/*.json && echo "data ok"
```

4 "ok" lines and you're ready.

## Step 1 — pick which cameras to render

In `threedgrut_playground/ps_gui.py`, around line 1467:

```python

only_cams = ["CamD"]   # just CamD, ~12 min

only_cams = None       # all four, ~20 min
```

Edit before launching — the GUI reads it once at startup.

> ⚠️ Don't touch the `cam_names` list above it. The code uses `cam_names.index()` to pick each camera's extrinsics. Trim the list and you render the wrong camera under the right name.

## Step 2 — launch

```bash

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python playground.py --gs_object ply_files/room.ply 2>&1 | tee /tmp/pg.log
```

The two env vars force the NVIDIA GPU on hybrid-graphics machines. Harmless otherwise.

## Step 3 — GUI, in this exact order

Screenshots for the 📷 notes in this guide are available on request.


1. Expand **Novel View from Vilota Calibration file**. The path field is pre-filled with `vk180.json` — replace it with your filename (the examples here use `DP180IP-30020104.json`) and click **Load Calibration**.

   It should report `Reloaded 4 cameras from ./calibration_files/<yours>` plus a **Device loaded** block with the device name and product name. The button then becomes **Reload Calibration** — same button, use it if you change the file.

> 📷 **PIC 1 — right after loading.** The Device loaded block, the four Select buttons, and `Select CamA — (Selected), Not Fisheye`.


2. **Select CamD**. On a VK180-family rig CamD is the fisheye, and the label must change to `xi: -0.3, alpha: 0.55` (values differ per device — you just need to see xi and alpha at all). The 3D view will visibly snap to a barrel-distorted fisheye projection, which is the clearest confirmation it worked.

   `Not Fisheye` next to **Select CamA** is normal: CamA is selected by default and is not a fisheye. It's a statement about the selected camera, not an error.

> 📷 **PIC 2 — side panel** with the xi/alpha label visible and the fisheye view behind it.


3. **Primitives → Quad 1 → Transform → Reset to 30cm square size**. Do this every launch.

   The board does **not** start square — it starts slightly too narrow, showing 6 tag columns where the texture has 7. That's subtle enough to miss, which is exactly why this step is mandatory. After pressing, count the columns: **7 across, 4 down**. If you see 6, the button didn't take.

   (Older checkouts label this button "15cm". Same button, and the board it makes was always 30 cm — the label was the lie, not the board. The button only changes the size: translation and rotation stay where you put them, so it's safe on a board you've already placed. The `Set Square Size: 15.00` field and `Square size (cm)` button next to it work now — type a tag size in cm and press the button to size any board to it — but the walkthrough here assumes the 30 cm reset.)


> 📷 **PIC 3 — the board after reset**, wider than tall.


4. Render widget dropdown → **Double Sphere Fisheye**. CamA/B/C still render KB4 automatically; the dropdown only decides CamD.
5. Expand **Record Trajectory Video** and Ctrl+click **Frames Between** → `1`. Higher values interpolate extra poses and multiply render time for no extra coverage. It will say `There are 0 cameras in the trajectory` — expected, you haven't built one yet.

> 📷 **PIC 4 — Render widget**, DS dropdown + Frames Between 1 in one shot.


6. Scroll back down to **Save/Load Video trajectory** (a different section from Record Trajectory Video, further down under the calibration block) → **Trajectory type** Orbit → **Build Trajectory**. Terminal prints `per-camera poses: {0: 180, 1: 180, 2: 180, 3: 756}`.
7. Last check: dropdown says Double Sphere, board is wider than tall. These two have wasted more renders than everything else combined.
8. **Render Device Trajectory MCAP**. One progress bar per camera. The app exits when done and writes `mcap_outputs/long_final_path.mcap`.

## Step 4 — relabel, detect, solve

Rename first — the render always writes the same filename. Set `SERIAL` to match the calibration file you loaded — `30.02.0104` for the DP180IP file, `DP180IP-2404-0004` for `vk180.json`. A mismatch makes the solver reject the dataset.

**Single camera (CamD, Double Sphere):**

```bash

RUN=myrun; SERIAL=30.02.0104   # <- your device's serial

mv mcap_outputs/long_final_path.mcap mcap_outputs/orbit_${RUN}.mcap

python fix_mcap_labels.py mcap_outputs/orbit_${RUN}.mcap ${RUN}_img.mcap \
  --topics S1/camd --serial ${SERIAL}
python tools/retime_for_detector.py ${RUN}_img.mcap ${RUN}_img_rt.mcap
TOPIC="S1/camd/tags:queued" ./run_offline_tags.sh ${RUN}_img_rt.mcap ${RUN}_tags.mcap

vk_calibrate \
  --vbag-path ${RUN}_tags.mcap \
  --cam-types ds --focal-lengths 431 --tag-size 0.30
```

**All four cameras:**

```bash

RUN=myrun; SERIAL=30.02.0104   # <- your device's serial

mv mcap_outputs/long_final_path.mcap mcap_outputs/orbit_${RUN}.mcap

python fix_mcap_labels.py mcap_outputs/orbit_${RUN}.mcap ${RUN}_img.mcap \
  --topics S1/cama S1/camb S1/camc S1/camd --serial ${SERIAL}
python tools/retime_for_detector.py ${RUN}_img.mcap ${RUN}_img_rt.mcap
CONFIG=offline_tags_all.json \
TOPIC="S1/cama/tags:queued S1/camb/tags:queued S1/camc/tags:queued S1/camd/tags:queued" \
./run_offline_tags.sh ${RUN}_img_rt.mcap ${RUN}_tags.mcap

vk_calibrate \
  --vbag-path ${RUN}_tags.mcap \
  --cam-types kb4 kb4 kb4 ds --focal-lengths -1 -1 -1 550 \
  --serial-number ${SERIAL} --tag-sizes 0.30 --focal-ratio-prior
```

`retime_for_detector.py` spaces the frame stamps 300 ms apart. The export stamps frames in real time (33 ms apart at 30 fps), and vk_camera_driver runs tag detection at most once per ~250 ms of stamp time, so without this step only every 8th frame gets detected.

### Flag notes

`--cam-types` must list one model per camera, in rig order, and must match what the renderer actually produced. The renderer picks the model *from the calibration data*, not from the GUI dropdown: a camera whose Double Sphere slot is zero renders KB4, a camera with that slot filled renders Double Sphere. On a VK180-family rig that's `kb4 kb4 kb4 ds`. Check your file before assuming — see [Using a different device](#using-a-different-device).

`--focal-lengths` are starting guesses, not answers. `-1` means auto-estimate. Guidance from the vk_calibrate README: KB4 works anywhere from 550 to 614; Double Sphere wants a *smaller* prior, around 550. On some modules (VKL-M7) DS should not be used at all. For this rig, DS converges from 431 and KB4 from 550. `-1` is the safe default if you don't know.

`--tag-size` is 0.30 because that is the board you actually rendered. This one bites people, so to be explicit:

|    | Board | Tag size flag |
|----|----|----|
| This pipeline (simulated) | the quad from the GUI reset button — **30 cm squares** | `0.30` |
| A real capture from the DP180IP device | its physical printed board, 0.05173 m | `0.058` |

They are two different physical objects, not two settings for the same run. If you render a board at a different size, change the flag to match.

Why it matters, measured: solving with `0.15` returns **identical intrinsics** (they are dimensionless) but every translation comes out at exactly half — CamB's baseline reads 3.4 cm instead of the true 6.9 cm. The result looks plausible and is wrong. Any dataset solved with `0.15` in the past has correct intrinsics and half-scale poses; re-solving with `0.30` fixes it, no re-render needed.

## Step 5 — read the result

```
S1/camd has type ds, intrinsics = fx fy cx cy xi alpha

mean_reprojection_error S1/camd: 0.41px

calibrated fov = 202 >= 186

success = true
```

> 📷 **PIC 5 — the end of the solver output**, intrinsics line through `success = true`.

`success = false` does not mean the fit failed. It's an AND over per-camera gates, and the strictest one is the field-of-view check — a test of whether the board reached the frame corners, not of whether the optimisation converged. Read these two lines together:

```
success_map [3] = false

success = false; success_without_calibrated_fov_check = true
```

That says: camera 3 didn't get enough corner coverage, everything else is healthy. The intrinsics on such a run are usually still good. A genuinely bad fit shows up as a high `mean_reprojection_error` or a non-converging optimisation, not as this flag.

Compare against the file you loaded:

```bash

CALIB=calibration_files/DP180IP-30020104.json

python3 -c "
import json, sys

d = json.load(open(sys.argv[1]))
for i, c in sorted(d['cameraData'], key=lambda x: x[0]):
    v = c['distortionCoeff']
    if v[5]:
        print(f'cam{i}  DS  fx fy cx cy xi alpha :', [round(x,4) for x in v[5:11]])
    else:
        k = c['intrinsicMatrix']
        print(f'cam{i}  KB4 k1..k4 :', [round(x,4) for x in v[0:4]])
" $CALIB
```

That also tells you which model each camera uses: a filled DS slot means Double Sphere, an empty one means KB4.

Healthy: focal within 0.15%, principal point within 0.5 px (a known pixel-convention offset, harmless), reprojection \~0.41 px, `success = true`. Extrinsics are trustworthy too, verified 2026-08-27: rotations within 0.04° of the device file and translations within \~0.5% — **provided the tag size is 0.30**.

## Step 6 (optional) — look at the detections

The detector embeds the source image in every detection message, so the tags file alone is enough. Note the viewer shows the detected tag quads over a low-resolution intensity layer, not the full rendered photo:

```bash

pip3 install rerun-sdk==0.20.1

vk_mcap_to_rrd ${RUN}_tags.mcap viewer_demo.json -o /tmp/${RUN}.rrd -s 0 -e 15

rerun /tmp/${RUN}.rrd
```

> 📷 **PIC 6 — Rerun**, four synchronised camera streams with detections. Scrub to a busy part of the timeline (around +20s) so several cameras have the board in view — each camera only sees it during its own trajectory segment.

Don't merge the image and tag MCAPs for viewing. Images use a zero-based clock, tags use wall-clock — they never line up on one timeline.

## Troubleshooting

### Setup

| What you see | Why | Fix |
|----|----|----|
| `python: command not found` | venv not active | `source /path/to/3dgrut/.venv/bin/activate` |
| vk-system fails on a fresh checkout | missing apt dependency | re-run step 1, then rebuild |
| `pre_build.py` fails on submodules | submodules not initialised | `git submodule update --recursive --init` |
| eCAL topics never appear | eCAL not installed or not configured | check `ecal_monitor` lists topics |
| `cpack` produces a bad version | called directly | use `make package` from the build folder |
| kaolin `no kernel image is available` | built for the wrong GPU arch | rebuild with `CUDA_VERSION=12.8.1` (+ `WITH_GCC11` on Blackwell) |
| `cudaErrorOperatingSystem` / error 304 | running under WSL2 | native Linux only |
| `vk_camera_driver: not found` | VK tools not installed | VK System Installation Guide |
| `fix_mcap_labels.py` can't find the schema | capnp schemas missing | check `/opt/vilota/messages/image.capnp` |
| `camera model does not exist` | model name not in the binary | `ds`, `kb4`, `eucm`, `pinhole`, `pinhole-radtan8` |

### Running

| What you see | Why | Fix |
|----|----|----|
| "Not Fisheye" next to Select CamA | normal — CamA isn't a fisheye | not an error; click Select CamD and read that row instead |
| No xi/alpha after Select CamD | file didn't load, or this rig has no DS camera | check for `Reloaded 4 cameras` in the panel |
| Board shows 6 tag columns, not 7 | reset button not pressed | press **Reset to 30cm square size**, recount |
| `success = false` but numbers look fine | a per-camera FOV gate failed | check `success_without_calibrated_fov_check`; the fit is probably fine |
| Focal \~446 with `ds` | equidistant render — wrong branch or build | confirm you're on the fixes branch, redo GUI steps 2 and 4 |
| Focal \~615 with `ds`, expected 431 | dropdown was on KB4 for CamD | set Double Sphere, re-render |
| Render takes hours | Frames Between > 1 | set it to 1 |
| `mv: cannot stat ...` | no render yet, or already renamed | `ls mcap_outputs/` |
| Solver rejects the dataset | wrong serial stamped | re-run `fix_mcap_labels.py` with the right `--serial` |
| Solver finds 0 corners | topic list wrong | topics are `S1/cama..camd`, tag topics end `:queued` |
| Cameras missing from output | `only_cams` still a subset | set to `None`, re-render |
| Baselines exactly half of truth | solved with `--tag-size 0.15` | re-solve with `0.30` — no re-render needed |
| Disk full | image MCAPs are 1–11 GB each | delete old ones, keep the tag MCAPs |
| Garbage result, fy/fx ≈ 1.76 | board not reset — 6 columns instead of 7 | GUI step 3, then re-render |
| Only two grids detected out of three | one board is outside the sweep, or behind the eye box | Fly to after Build Trajectory and look. Move the board so every eye is in front of it and it sits inside the camera's vertical field. |
| Two detections in the whole bag, frames look clean, boards visible | **mirrored render**: the camera is behind the boards and sees the tags through them, reversed. tag16h5 cannot decode a mirror image. | never set `ORBIT_FLIP`. Check `eye z` in the orbit log is on the same side of the boards as the origin. |
| Frames are a beige smear, `std` under 40 | the eye is inside the scene geometry | change `ORBIT_DIST` until Fly to shows the room |
| Boards at the fisheye edge, few detections | `AIM_SCALE` too large for the distance | lower it. 0.9 for DS at 2.5 m is verified. |
| Gate fails: `need increase data points diagonal` | aims do not reach the corners | raise `AIM_SCALE` (DS) or `AIM_SCALE_KB4` (KB4). The four `d_normalised_corners` must all drop under 0.05 for KB4, 0.06 for DS. |
| Gate fails: `minimum bucket count ... 10 times lesser` | one image region never sees tags | a board is missing from that camera's view. For CamA check the near board is inside its 47° vertical field. |
| CamA detects almost nothing on the near board | half the eyes are behind that board's plane | move it deeper than the eye box: `pos z -4.85` in v9 |
| Board renders at double size | reset button pressed in three-grid mode | relaunch. The scene file sizes the boards. |
| `Orbit failed: No primitive matching` | wrong `ORBIT_TARGET` or no boards spawned | check the `[playground]` startup lines |
| Fly to shows the origin, not the boards | orbit not built yet | Build Trajectory (type Orbit) first; it parks the rig at the first pose |
| Room upside down in Fly to | expected on older checkouts; fixed | the current branch negates the up vector in the preview. The render was never affected. |
| Empty tags bag, 432 bytes | detector aborted at startup | a declared grid shares adjacency with another (a 3×3 with ids 0..8 collides with `aprilgrid`). Run the detector unfiltered and read its first lines. |
| Focal 3.4 % off with `ds` | one board plane | three-grid mode, or fit KB4 |
| Baselines off by a constant factor | wrong `--tag-sizes` | one value overrides every grid. Pass six, in grid order. Intrinsics are unaffected, so no re-render. |

## How the detector sees tags

Measured on this pipeline, CamD, tag16h5 with border 2, by shrinking frames from a working bag until detection failed:

| tag side in the image | detection rate |
|---|---|
| 16 px | 50 % |
| 25 px | 90 % |
| 30 px | 95 %, plateau |

Plan for **30 px**. The tag size needed at a given distance is $s = 30\,d / f$:

| distance | CamD, $f = 615$ (1920 px) | CamA/B/C, $f = 393$ (1280 px) |
|---|---|---|
| 1 m | 4.9 cm | 7.6 cm |
| 3 m | 14.6 cm | 22.8 cm |
| 5 m | 24.4 cm | 38.0 cm |
| 7 m | 34.1 cm | 53.2 cm |

The detector runs on level 1 of an image pyramid (`apriltag.use_mipmap = 1`, input `camX_pyr`), so these are full-resolution numbers for a half-resolution detector. Turning the pyramid off did not change the result on a failing bag; the floor is about legibility, not resolution.

This table is also the rig's requirement. A real monitor at 7 m needs tags of about 34 cm for CamD and 53 cm for the KB4 cameras, or those cameras never see it.

## Using a different device

Nothing above is specific to the DP180IP. The pipeline takes whatever calibration file you give it and renders those cameras. What you change:

| What | Where | How to find it |
|----|----|----|
| Calibration file | GUI step 3.1 | your file in `calibration_files/` |
| `SERIAL` | `fix_mcap_labels.py`, `--serial-number` | the device serial the file belongs to |
| `--cam-types` | solver | one per camera, from the DS-slot rule below |
| `--focal-lengths` | solver | rough priors, or `-1` for auto |
| Which camera is the fisheye | GUI step 3.2 | the one whose label shows xi and alpha |
| `--tag-size` | solver | the board *you rendered*, not the device's physical board |

**How the renderer picks a model.** Per camera, from the data:

* `distortionCoeff[5] == 0` → **KB4** path, reads focal from `intrinsicMatrix` and k1..k4 from `distortionCoeff[0:4]`
* `distortionCoeff[5] != 0` → **Double Sphere** path, reads fx, fy, cx, cy, xi, alpha from `distortionCoeff[5:11]`

Run the truth-print snippet from step 5 against your file and it prints the model per camera. Whatever it says is what `--cam-types` must match.

**Which solver models exist.** `ds`, `kb4`, `eucm`, `pinhole`, `pinhole-radtan8`. The renderer only produces KB4 and Double Sphere, but you can *fit* any of them to the same detections — useful for comparing models on identical data. Worth knowing: on a 190° lens, `pinhole-radtan8` gets good reprojection but fails the FOV gate at 134°, which is the model's limit, not a data problem.

### Known limits

> ⚠️ **The loader recognises two rig types, not two files.** `novel_view_renderer.py` reads the product name *inside* the JSON and looks up that rig's camera layout. Only `DP180IP` and `VK180*` cases exist. Any number of individual device files work as long as their product name is one of those two — a file from a different product won't load whatever you name it. Adding a third product means one more case plus that rig's camera-name map.

> ⚠️ **Only 4-camera rigs have been tested.** The dispatch and export code loop over cameras generically, so other counts may work, but nobody has tried.

> ⚠️ **The walkthrough above uses the single AprilGrid quad (30 cm squares), and the step-5 numbers were measured on it.** The real device rig uses six boards at different depths with distinct `start_ids` — the playground can spawn those now, see [Multi-board walkthrough](#multi-board-walkthrough), with matching `april_grids` entries in the detector config.

## Multi-board walkthrough

The real rig is six grids sharing one tag pool, told apart by layout, not by tag id. The playground can spawn them as separate quads, one material per board, in two modes: the default respaced layout, or a scene JSON that places every board explicitly. The verified 31 Aug 2026 run below used the scene-JSON mode and predates the respacing — its positions belong to that mode only, not to the default layout.

### The default layout

```bash

PLAYGROUND_BOARDS=1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
python playground.py --gs_object ply_files/room.ply
```

`PLAYGROUND_BOARDS=1` spawns the default three-board scene. A comma-separated list of board names picks others; `all` spawns every board plus the logo.

| Board | Grid | Tag ids | Default position (x, y, z) |
|----|----|----|----|
| `aprilgrid` | 4x7 base | 0..27 | (0, −0.990, −1.035) |
| `grid_off14` | 4x7 interleaved | 0 14 1 15 … 13 27 | (0, 0.000, −1.345) |
| `grid_off15` | 4x7 interleaved | 0 15 1 16 … 13 28 | (0, +0.990, −1.655) |

The boards stack vertically 0.99 m apart — the height of a 4x7 board at 15 cm squares plus a 20% gap — so you can resize every board up to 15 cm squares in the GUI and they stay fully visible instead of the near board hiding the far ones. Setting `BOARD_ANGLES` restores the old angular layout, which does overlap at that size.

**Build Trajectory** with type Orbit frames all boards together in this mode: it orbits the centre of their combined bounding box and pushes the near eye ring out until every board fits the KB4 field of view from every viewpoint. Set `ORBIT_TARGET=<board name>` to orbit a single board the old way; `ORBIT_DIST` and `ORBIT_FLIP` still work as before.

### Scene-JSON mode

The scene is a JSON file. `office_scene_v9.json` is the verified layout:

```json
{
  "boards": [
    {"material": "aprilgrid",  "tag_cm": 15.0, "pos": [0.0, -1.20, -4.85], "rot": [0, 0, 0]},
    {"material": "grid_off14", "tag_cm": 20.0, "pos": [0.0, -0.16, -4.00], "rot": [0, 0, 0]},
    {"material": "grid_off15", "sx": 1.41, "sy": 0.825, "pos": [0.0, 2.53, -3.00], "rot": [-25, 0, 0]},
    {"material": "vilota_logo", "sx": 0.35, "sy": 0.35, "pos": [-1.77, -5.43, -4.0], "rot": [150.6, 0, -180]}
  ]
}
```

Each entry:

| key | required | meaning |
|---|---|---|
| `material` | yes | one of `aprilgrid`, `grid_off14`, `grid_off15`, `grid_off16`, `grid_2x2`, `grid_3x1`, `grid_3x3`, `vilota_logo` |
| `pos` | yes | `[x, y, z]` in metres, Polyscope world frame |
| `rot` | no | `[rx, ry, rz]` in degrees, default `[0, 0, 0]` |
| `tag_cm` | one of | tag side in cm. The board's half extents follow from it. |
| `sx`, `sy` | one of | half extents in metres, for anything that is not a board. A 4×7 board with 30 cm tags is `sx 1.41, sy 0.825`. |
| `_note` | no | ignored |

Two facts about the geometry:

* **The four 4×7 materials carry the same 28 tag ids, 0 to 27, in different orders.** `aprilgrid` is row-major. `grid_off14` interleaves as `[0, 14, 1, 15, ...]`, `grid_off15` as `[0, 15, 1, 16, ...]`, `grid_off16` as `[0, 16, 1, 17, ...]`. The detector tells boards apart by **tag adjacency**, not by id range, so all four fit inside tag16h5's 30 ids. Do not declare a 3×3 with ids 0 to 8: its adjacency pairs collide with `aprilgrid` and the detector aborts at startup with an empty bag.
* **A board is drawn 1.25 m from its `pos` along its local z.** The quad's local vertices sit at z = 2.5, scaled by `sz = 0.5`. A board at `pos z = -4.0` is drawn at `z = -2.75`. The orbit generator uses the drawn position. Place boards by looking, not by arithmetic on `pos`.

The verified 31 Aug 2026 launch command — scene-JSON mode, all four cameras:

```bash
cd <3dgrut-NVR>
source .venv/bin/activate
PLAYGROUND_BOARDS=1 BOARD_SCENE=office_scene_v9.json ORBIT_TARGET=grid_off14 \
ORBIT_DIST=2.5 ARM_REACH=0.5 AIM_SCALE=0.9 AIM_SCALE_KB4=1.3 \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  python playground.py --gs_object ply_files/room.ply 2>&1 | tee /tmp/pg.log
```

Startup prints one `[playground]` line per board with its `pos`, `rot`, `sx`, `sy`. Check them against the scene file.

For CamD only, add `ORBIT_CAMS=3`. The orbit then aims every pose for CamD (756 poses instead of 1296), and the render takes about six minutes instead of ten. All four image topics are still written. Solve with `--cam-types ds` on a CamD-only bag ([step 4](#step-4--relabel-detect-solve)).

### The orbit controls

All are environment variables. Unset means the original behaviour.

| variable | default | what it does |
|---|---|---|
| `PLAYGROUND_BOARDS` | unset | `1` spawns the board materials and enables `BOARD_SCENE`. |
| `BOARD_SCENE` | unset | path to the scene JSON. |
| `ORBIT_TARGET` | `grid_off14`, then `aprilgrid`, then `Quad` | which primitive the orbit aims at. Its drawn centre is the aim point for every pose. |
| `ORBIT_DIST` | `0.5×` and `0.9×` board width | mean eye distance from the target board in metres. `2.5` gives an eye about 1.9 m in front. |
| `ARM_REACH` | unset, no clamp | clamps every viewpoint into a box of ±R m around their centroid. `0.5` models the arm's reach. |
| `AIM_SCALE` | `1.0` | multiplies every yaw and pitch in the CamD (Double Sphere) aim grid. |
| `AIM_SCALE_KB4` | `= AIM_SCALE` | same for CamA, B, C. The KB4 grid is narrower, so it needs a larger value to reach the corners. |
| `ORBIT_CAMS` | `0,1,2,3` | which cameras get aimed poses. `3` is CamD only. |
| `ORBIT_FLIP` | unset | **do not set.** It puts the camera behind the boards. See troubleshooting. |

Why the defaults are not right for three boards: the stock orbit was written to hug one board at half a metre, with swings of ±78° for CamD so the board reaches the image corners. At three metres the same swing throws the boards to the fisheye edge where the tags fall below the detector floor. `ORBIT_DIST` moves the eye back, `AIM_SCALE` narrows the swing to match, and `ARM_REACH` keeps the eye inside what an arm can do.

One default has moved since that table was written: with `ORBIT_TARGET` unset the orbit now frames all boards together, per [the default layout](#the-default-layout). Setting `ORBIT_TARGET` restores the single-board aim the table describes.

### Verified result, 31 Aug 2026

Scene `office_scene_v9.json`, the four-camera launch command above, one render, one solve. Device file `DP180IP-30020104.json`.

| cam | model | $f_x$ file → fitted | error | $k_1$ or $\xi$ file → fitted | error | $\alpha$ error | reprojection |
|---|---|---|---|---|---|---|---|
| A | KB4 | 392.654 → 392.396 | 0.066 % | 0.372462 → 0.373208 | 0.20 % | — | 0.447 px |
| B | KB4 | 397.357 → 397.050 | 0.077 % | 0.355063 → 0.355198 | 0.04 % | — | 0.452 px |
| C | KB4 | 399.369 → 399.057 | 0.078 % | 0.356578 → 0.357312 | 0.21 % | — | 0.450 px |
| D | DS | 432.070 → 432.205 | 0.031 % | −0.3015 → −0.300464 | 0.34 % | 0.002 % | 0.436 px |

Principal points within 0.9 px on every camera. FOV gate passed on all four. 588 932 corners, 0 of 8 218 poses rejected, 16 of 16 iterations converged.

For CamD the whole projection curve agrees to within 0.26 px: at every angle off the optical axis, the recovered model and the true model place the ray within a quarter pixel of each other, and within 0.03 px inside 50°. That number is smaller than the solver's own reprojection error, so the disagreement is below what the measurement can resolve.

## Rendering for VIO runs

Feeding vk_vio (through vk_camera_driver) takes raw images, not tag detections, and the recipe differs from a calibration render:

* **One board suffices.** VIO tracks corners frame to frame; it doesn't solve intrinsics, so the multi-board scene buys you nothing here.
* **Start stationary.** Hold the first pose for 2–3 seconds so the estimator can initialise before anything moves.
* **Translate and rotate.** Pure rotation gives the backend no parallax. The motion should do both.
* **15–30 fps** is the right frame rate range. Higher wastes render time, lower starves the tracker.
* **Set `PLAYGROUND_CALIB=calibration_files/<your file>.json` before launching.** The MCAP export then embeds each camera's intrinsics and extrinsics in every image message, which the driver requires — without it the driver silently emits empty flow and VIO sees nothing.
* The export still writes `mcap_outputs/long_final_path.mcap` and the next render overwrites it — rename immediately, same as step 4.

## Notes

* Image MCAPs are big and regenerable. Tag MCAPs are \~1 MB and carry the evidence — back those up, delete the images.
* Rename every render output immediately. The next render overwrites it.
* Every solver run prints its full command line. When in doubt about what a run did, read the log rather than your memory.