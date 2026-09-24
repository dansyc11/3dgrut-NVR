# meetingroom AGGRESSIVE VIO profile - source this before the playground
# launch (or before a headless dry-run). The meeting room cannot host the
# 20 kph extreme profile (usable camera-path footprint only ~3.6 x 5.3 m),
# so this is the
# scaled-down aggressive variant: the stock analytic board path with
# STRESS=3.5, which scales every oscillation frequency and hence all
# rates by 3.5 while amplitudes and coverage stay fixed
# (vio_trajectory.py:221, :315). Dry-run verified 2026-09-18: all nine
# checks PASS, peak speed 1.58 m/s (base path is 0.32 m/s), peak angular
# rate 143.7 deg/s, peak per-frame px 43.2 with VIO_FPS=40 (ceiling 50;
# 30 fps FAILs at 59.3 px, so 40 is the floor for this STRESS).
#
# Knob read sites: VioConfig vio_trajectory.py:207-231. Duration stays
# the default 3 s lead + 40 s motion -> 1720 frames at 40 fps.
#
# Set VILOTA_DATASETS to the folder that holds meetingroom/ before sourcing.
export SCENE_ALIGNMENT="${VILOTA_DATASETS:?set VILOTA_DATASETS to the folder that holds meetingroom/}/meetingroom/lidar_alignment.json"
export PLAYGROUND_CALIB=calibration_files/DP180IP-30020104.json
export VIO_TRAJ=1
export STRESS=3.5
export STATIONARY_LEAD_S=3
export VIO_FPS=40
export VIO_TRAJ_NPZ=mcap_outputs/meetingroom_aggressive_truth.npz

# Headless dry-run (CPU only, all checks must PASS):
#   python -m threedgrut_playground.utils.vio_trajectory --dry-run \
#     --alignment $SCENE_ALIGNMENT \
#     --npz $VIO_TRAJ_NPZ --rerun mcap_outputs/meetingroom_aggressive.rrd
