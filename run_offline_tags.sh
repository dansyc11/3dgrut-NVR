#!/usr/bin/env bash
# Run offline AprilTag detection on a converted MCAP.
# Input:  an MCAP of capnp::Image messages.
# Output: an MCAP of TagDetections for the calibration tool.

set -euo pipefail

IN_MCAP="${1:?usage: run_offline_tags.sh input.mcap output.mcap}"
OUT_MCAP="${2:?usage: run_offline_tags.sh input.mcap output.mcap}"
CONFIG="${CONFIG:-offline_tags_camd.json}"
RATE="${RATE:-0.5}"
TOPIC="${TOPIC:-S1/camd/tags}"

cleanup() {
    kill "${DRIVER_PID:-}" "${RECORD_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/3] starting detector with ${CONFIG}"
vk_camera_driver "${CONFIG}" &
DRIVER_PID=$!
sleep 3

echo "[2/3] starting recorder to ${OUT_MCAP}"
vk_record "${OUT_MCAP}" -t ${TOPIC} &
RECORD_PID=$!
sleep 2

echo "[3/3] playing ${IN_MCAP} at rate ${RATE}"
vk_playback "${IN_MCAP}" -r "${RATE}"

echo "playback done, waiting for the detector to drain"
sleep 20

echo "stopping detector"
kill -INT "${DRIVER_PID}" 2>/dev/null || true
sleep 10

echo "stopping recorder, this flushes the backlog"
kill -INT "${RECORD_PID}" 2>/dev/null || true
wait "${RECORD_PID}" 2>/dev/null || true

ls -lh "${OUT_MCAP}"
