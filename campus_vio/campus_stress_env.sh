# campus STRESS VIO profile - source this, then run the dry-run below.
# Same corridor circuit as BASELINE, aggressive motion baked into the
# waypoint spacing (knots are uniform in spline phase, so rendered speed =
# STRESS * spacing / knot interval): sprints to 4 m/s, two stop-and-go
# events (~0.2 m/s crawls), 2.0 m/s U-turn caps, +-45 deg head-turn
# windows, 0.10 m vertical bob. STRESS=1.25 laps the circuit 1.25x faster
# (vio_trajectory.py :792) and is part of the calibration: measured peaks
# 3.93 m/s / 3.13 rad/s / 5.53 m/s^2, |f| 8.71..11.42 (room band was
# 9.66..9.96). Raising STRESS scales rates ~linearly, accel ~quadratically.
#
# Knob read sites: identical to campus_baseline_env.sh (VioConfig
# vio_trajectory.py:201-214, STRESS applied :302/:792, PLAYGROUND_CALIB
# mcap_convertor.py:237). VIO_FPS=45 keeps the per-frame displacement
# formula under TRACKABLE_PX=50 at 3.1 rad/s (px check :539-556; measured
# peak 30.7 px): per the machinery's own rule, raise fps, never the speed.
# VIO_MOTION_S MUST stay 41.7 = STRESS * 31.8 s lap + RAMP_S/2; changing
# it rescales every speed.
export SCENE_ALIGNMENT=/home/vilota/datasets/main_campus/lidar_alignment.json
export PLAYGROUND_CALIB=calibration_files/DP180IP-30020104.json
export VIO_TRAJ=1
export STRESS=1.25
export STATIONARY_LEAD_S=3
export VIO_MOTION_S=41.7
export VIO_FPS=45
export VIO_TRAJ_NPZ=mcap_outputs/campus_stress_truth.npz

# Headless dry-run (CPU only; 6 board-visibility FAILs expected, no boards
# on the corridor - kinematics/px/C2 all PASS):
#   python -m threedgrut_playground.utils.vio_trajectory --dry-run \
#     --waypoints campus_vio/campus_stress_waypoints.csv \
#     --cloud campus_vio/campus_overhead_board_units.csv \
#     --alignment $SCENE_ALIGNMENT \
#     --npz $VIO_TRAJ_NPZ --rerun mcap_outputs/campus_stress.rrd
# Tube / kinematics audit:
#   python tools/campus_vio_corridor.py check --npz $VIO_TRAJ_NPZ
#
# RENDERING NOT WIRED YET: see campus_baseline_env.sh - build_vio_trajectory
# has no waypoint input; the circuit currently exists only as dry-run
# npz/rrd.
