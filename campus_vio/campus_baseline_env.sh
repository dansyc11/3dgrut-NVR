# campus BASELINE VIO profile - source this, then run the dry-run below.
# Walking pace on the campus corridor stadium circuit (72.7 m, one lap):
# 1.4 m/s straights, 0.7 m/s through the U-turn caps, measured peaks
# 1.45 m/s / 0.96 rad/s / 0.96 m/s^2, |f| 9.81..9.84.
#
# Where each knob is read (threedgrut_playground/utils/vio_trajectory.py):
#   STATIONARY_LEAD_S  VioConfig, :201   parked lead before motion
#   VIO_MOTION_S       VioConfig, :202   MUST stay 60.8: spline period =
#                      motion - RAMP_S/2 (:785 via _tau) and the waypoint
#                      file was placed for exactly one 58.8 s lap; changing
#                      it rescales every speed by (58.8 / (motion-2)).
#   VIO_FPS            VioConfig, :207   also becomes PLAYGROUND_FPS for
#                      the mcap export (build_vio_trajectory :706)
#   STRESS             VioConfig, :212   phase-clock multiplier: pose_at
#                      :302, spline circuits :792 ("laps the circuit
#                      faster"); rates scale ~STRESS, accel ~STRESS^2
#   SCENE_ALIGNMENT    VioConfig, :213   metric design frame; cloud->metres
#                      at :670-677, poses->scene units at the render
#                      boundary :727-728, rig extrinsics compose in scene
#                      units via nvr.v_device.metres_per_unit :714
#   VIO_TRAJ_NPZ       VioConfig, :214   ground-truth output
#   VIO_TRAJ=1         ps_gui.py :698    presets the GUI Trajectory-type
#                      combo to "VIO trajectory"
#   PLAYGROUND_CALIB   mcap_convertor.py :237  calibration embedded in the
#                      exported mcap (the real Aug 17 unit, serial
#                      30.02.0104)
export SCENE_ALIGNMENT=/home/vilota/datasets/main_campus/lidar_alignment.json
export PLAYGROUND_CALIB=calibration_files/DP180IP-30020104.json
export VIO_TRAJ=1
export STRESS=1
export STATIONARY_LEAD_S=3
export VIO_MOTION_S=60.8
export VIO_FPS=20
export VIO_TRAJ_NPZ=mcap_outputs/campus_baseline_truth.npz

# Headless dry-run (CPU only; the 6 board-visibility FAILs are expected,
# there are no boards on the corridor - kinematics/px/C2 all PASS):
#   python -m threedgrut_playground.utils.vio_trajectory --dry-run \
#     --waypoints campus_vio/campus_baseline_waypoints.csv \
#     --cloud campus_vio/campus_overhead_board_units.csv \
#     --alignment $SCENE_ALIGNMENT \
#     --npz $VIO_TRAJ_NPZ --rerun mcap_outputs/campus_baseline.rrd
# Tube / kinematics audit:
#   python tools/campus_vio_corridor.py check --npz $VIO_TRAJ_NPZ
#
# RENDERING NOT WIRED YET: build_vio_trajectory (vio_trajectory.py:662,
# reached from the GUI combo and drive_render.py via
# build_orbit_trajectory(vio=True), orbit_trajectory.py:306-311) samples
# only the analytic pose_at (:693, :726) - it has no --waypoints/env
# input. Until spline_pose_fn is plumbed in there, this circuit renders
# nowhere; only the dry-run npz/rrd above exist.
