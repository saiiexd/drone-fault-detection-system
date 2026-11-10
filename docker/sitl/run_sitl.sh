#!/usr/bin/env bash
# Start PX4 SITL with a Gazebo vehicle, headless. Runs PX4 as a daemon so that
# commands can be sent from outside with `docker exec <c> px4-<module> ...`.
#
# Env: PX4_SIM_MODEL (default gz_x500), PX4_GZ_WORLD (default "default").
set -euo pipefail

export PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}"
export HEADLESS=1

BUILD=/px4/build/px4_sitl_default
cd "$BUILD/rootfs"
# -d: daemon mode, no interactive shell (stdin is not a terminal under docker)
exec "$BUILD/bin/px4" -d "$BUILD/etc"
