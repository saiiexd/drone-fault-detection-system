#!/usr/bin/env bash
# Start the aeropulse dashboard (backend + frontend) in one command.
#
#   ./run.sh         replay a recorded motor-failure flight (no Docker needed)
#   ./run.sh live    fly one simulated drone in Docker (3D view in a window) and detect
#                    its motor failure live on the dashboard
#
# Works from any folder. Ctrl+C stops everything.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

URL=http://127.0.0.1:8000
mode="${1:-replay}"
case "$mode" in
  replay) args=(--replay data/raw/sitl6/p6_motor_off_5/telemetry.tlog) ;;
  live)   args=() ;;
  *) echo "usage: ./run.sh [replay|live]" >&2; exit 2 ;;
esac

# A dashboard left running from an earlier run holds port 8000 (and UDP 14550),
# and the browser would show its stale data. Stop it first.
OLD="\.venv/bin/python[0-9.]* -m aeropulse\.stream\.service"   # the server process only
if pgrep -f "$OLD" >/dev/null; then
  echo "Stopping a dashboard that is still running from an earlier run."
  pkill -f "$OLD" || true
  while pgrep -f "$OLD" >/dev/null; do sleep 0.2; done
fi

uv run python -m aeropulse.stream.service "${args[@]}" &
svc=$!
# A background job in a script ignores Ctrl+C, so stop it explicitly.
cleanup() { kill $svc 2>/dev/null || true; wait $svc 2>/dev/null || true
            docker rm -f aeropulse-sitl aeropulse-gui >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM

# Wait until the server answers; stop if it died (the error is printed above).
until curl -sf -o /dev/null "$URL/api/state"; do
  kill -0 $svc 2>/dev/null || { echo "The dashboard failed to start; see the error above." >&2; exit 1; }
  sleep 0.5
done
xdg-open "$URL" >/dev/null 2>&1 || echo "Open $URL in your browser."

if [ "$mode" = live ]; then
  uv run python -m aeropulse.sim.scenarios --out data/raw/demo --set phase6_live --only live_motor_off_1 --no-tlog --gui --world town
  echo "Flight finished. Dashboard still running at $URL; Ctrl+C to stop."
fi
wait $svc
