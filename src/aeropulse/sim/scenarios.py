"""Fly PX4 SITL (Gazebo x500 quad, Docker) with injected faults and collect ULogs.

Each run starts a fresh container, so every flight begins from the same
simulator state. The fault is injected over MAVLink (MAV_CMD_INJECT_FAILURE,
the same command a ground station would send). PX4 logs that command in its
`vehicle_command` topic, on the same clock as every sensor, so the label start
time comes from the log itself, exact to the microsecond.

Each flight can also record the ground-station MAVLink stream (UDP 14550)
as a .tlog next to the ULog. Phase 6 trains its live detector on those tlogs,
so it learns from the same stream it later sees live.

Usage:
    uv run python -m aeropulse.sim.scenarios --out data/raw/sitl            # phase 5 set
    uv run python -m aeropulse.sim.scenarios --out data/raw/sitl6 --set phase6
    uv run python -m aeropulse.sim.scenarios --out data/raw/sitl --only motor_off_1
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mavsdk import ComponentType, Configuration, Mavsdk
from mavsdk.plugins.action import Action
from mavsdk.plugins.failure import Failure, FailureType, FailureUnit
from mavsdk.plugins.param import Param
from mavsdk.plugins.telemetry import Telemetry
from pyulog import ULog

from aeropulse.stream.mavlink_io import Recorder

IMAGE = "aeropulse/px4-sitl:v1.17.0"
CONTAINER = "aeropulse-sitl"
VIEWER = "aeropulse-gui"   # optional 3D view (--gui)
VIEWER_CONFIG = "/opt/aeropulse/viewer.gui.config"   # baked into the image
# Gazebo worlds: "default" (PX4's empty world; all recorded flights) or "town"
# (the same world with buildings and trees, docker/sitl/worlds/town.sdf).
WORLDS = ("default", "town")
LOG_IN_CONTAINER = "/px4/build/px4_sitl_default/rootfs/log"
MAV_CMD_INJECT_FAILURE = 420
TAKEOFF_ALT_M = 10.0

# Fault component -> UAV-SEAD class, so SITL logs share the label vocabulary.
CLASS_OF_UNIT = {"motor": "Mechanical", "gps": "Global Position", "baro": "Altitude"}
UNITS = {"motor": FailureUnit.SYSTEM_MOTOR, "gps": FailureUnit.SENSOR_GPS,
         "baro": FailureUnit.SENSOR_BARO}
TYPES = {"off": FailureType.OFF, "stuck": FailureType.STUCK, "wrong": FailureType.WRONG}

PARAMS_INT = {
    "SYS_FAILURE_EN": 1,    # allow failure injection
    "CA_FAILURE_MODE": 1,   # control allocator removes a motor reported as failed
    "COM_RC_IN_MODE": 4,    # no RC in simulation: stick input disabled
    "NAV_DLL_ACT": 0,       # no failsafe on datalink loss (scripted flight)
}


@dataclass(frozen=True)
class Scenario:
    name: str
    unit: str | None = None       # None = nominal flight
    ftype: str = "off"
    instance: int = 0             # 0 = all instances; motors are 1-based
    inject_after_s: float = 15.0  # after reaching takeoff altitude
    after_fault_s: float = 15.0
    seed: int = 0
    takeoff_alt_m: float = TAKEOFF_ALT_M
    cruise_mps: float | None = None   # MPC_XY_CRUISE; None = PX4 default (5 m/s)
    alt_jitter_m: tuple[float, float] = (-2.0, 3.0)
    # 1: PX4's control allocator stops and removes a motor its failure detector
    # reports. In v1.17 + Gazebo this is what makes "motor off" physical: the
    # injector only flags the motor, and the allocator stops it. 0: the flag is
    # ignored and the motor keeps running (verified 2026-11-09: command stays
    # ~0.74 after injection), so the injection has no physical effect.
    ca_failure_mode: int = 1


# Only physical motor-off faults are used. In PX4 v1.17.0 with Gazebo (gz):
# - sensor failures (gps/baro/...) are handled only by simulator_mavlink
#   (Gazebo Classic), not by gz_bridge: no ACK, no effect (verified in logs);
# - motor "stuck"/"wrong" only falsify ESC telemetry; the motor keeps thrusting.
# (The current docs describe newer behaviour from PX4 main.)
SCENARIOS = [
    *[Scenario(f"nominal_{i}", seed=i, inject_after_s=25.0, after_fault_s=0.0) for i in range(1, 5)],
    *[Scenario(f"motor_off_{m}", "motor", "off", m, seed=10 + m) for m in range(1, 5)],
    *[Scenario(f"nominal_{i}", seed=i, inject_after_s=35.0, after_fault_s=0.0) for i in range(5, 9)],
    # second batch: same motors, different flight paths and injection times
    *[Scenario(f"motor_off_{m + 4}", "motor", "off", m, seed=40 + m,
               inject_after_s=8.0 + 5.0 * m) for m in range(1, 5)],
]


def phase6_scenarios() -> list[Scenario]:
    """Varied quad flights for training and testing the live detector.

    Faults: every motor, takeoff altitude 8-30 m (time to impact varies),
    injection 6-25 s into the pattern, cruise 3-10 m/s. Nominal flights
    include fast cruising and large climbs/descents, so "aggressive but
    healthy" is in the data and the detector cannot equate motion with failure.
    Seeds are fixed, so the set is reproducible.
    """
    rng = random.Random(6)
    out = []
    for i in range(1, 33):
        out.append(Scenario(
            f"p6_motor_off_{i}", "motor", "off", instance=(i - 1) % 4 + 1, seed=600 + i,
            inject_after_s=round(rng.uniform(6, 25), 1), after_fault_s=8.0,
            takeoff_alt_m=round(rng.uniform(8, 30), 1), cruise_mps=round(rng.uniform(3, 10), 1)))
    for i in range(1, 21):
        aggressive = i % 2 == 0
        out.append(Scenario(
            f"p6_nominal_{i}", seed=700 + i, inject_after_s=round(rng.uniform(35, 60), 1),
            after_fault_s=0.0, takeoff_alt_m=round(rng.uniform(8, 30), 1),
            cruise_mps=round(rng.uniform(7, 10) if aggressive else rng.uniform(3, 6), 1),
            alt_jitter_m=(-6.0, 8.0) if aggressive else (-2.0, 3.0)))
    return out


def phase6_live_scenarios() -> list[Scenario]:
    """Held-out flights flown with the live service attached (new seeds).

    The last four use CA_FAILURE_MODE=0. These were meant to test a failure
    the controller is unaware of; in v1.17 + Gazebo the motor then does not
    stop at all (see Scenario.ca_failure_mode), so they serve as nominal
    flights that contain an accepted injection command.
    """
    rng = random.Random(66)
    out = []
    for i in range(1, 13):
        out.append(Scenario(
            f"live_motor_off_{i}", "motor", "off", instance=(i - 1) % 4 + 1, seed=900 + i,
            inject_after_s=round(rng.uniform(6, 20), 1), after_fault_s=8.0,
            takeoff_alt_m=round(rng.uniform(8, 30), 1), cruise_mps=round(rng.uniform(3, 10), 1),
            ca_failure_mode=0 if i > 8 else 1))
    for i in range(1, 5):
        aggressive = i % 2 == 0
        out.append(Scenario(
            f"live_nominal_{i}", seed=950 + i, inject_after_s=round(rng.uniform(30, 45), 1),
            after_fault_s=0.0, takeoff_alt_m=round(rng.uniform(8, 30), 1),
            cruise_mps=round(rng.uniform(7, 10) if aggressive else rng.uniform(3, 6), 1),
            alt_jitter_m=(-6.0, 8.0) if aggressive else (-2.0, 3.0)))
    return out


SCENARIO_SETS = {"phase5": lambda: SCENARIOS, "phase6": phase6_scenarios,
                 "phase6_live": phase6_live_scenarios}


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


def start_container(log_dir: Path, world: str = "default") -> None:
    docker("rm", "-f", CONTAINER, check=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    docker("run", "-d", "--name", CONTAINER, "--network", "host",
           "--user", f"{_uid()}:{_gid()}", "-v", f"{log_dir.resolve()}:{LOG_IN_CONTAINER}",
           "-e", f"PX4_GZ_WORLD={world}", IMAGE)


def start_viewer() -> None:
    """Open Gazebo's 3D view of the running simulation in a window.

    The simulator runs headless; this is a second container with only the
    Gazebo GUI client, which joins the server over gz-transport on loopback.
    Both containers must run as the same user: gz-transport only connects
    processes in the same partition, which defaults to hostname:username.
    The window is drawn through the host's X server (Xwayland on Wayland).
    """
    import grp
    import os
    import shutil
    display = os.environ.get("DISPLAY")
    if not display:
        print("  no DISPLAY: cannot open the 3D view", flush=True)
        return
    if shutil.which("xhost"):   # let this user's containers draw on the X server
        subprocess.run(["xhost", f"+si:localuser:{os.environ.get('USER', '')}"], capture_output=True)
    groups = []
    for g in ("video", "render"):   # GPU device access
        try:
            groups += ["--group-add", str(grp.getgrnam(g).gr_gid)]
        except KeyError:
            pass
    docker("rm", "-f", VIEWER, check=False)
    docker("run", "-d", "--name", VIEWER, "--network", "host", "--user", f"{_uid()}:{_gid()}",
           "-e", f"DISPLAY={display}", "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
           "--device", "/dev/dri", *groups, "--entrypoint", "gz", IMAGE,
           "sim", "-g", "--gui-config", VIEWER_CONFIG)
    # Once the GUI listens on /gui/track, make the camera follow the drone from
    # 4 m behind and 2 m above, always looking at it. pgain 0.3 keeps up with a
    # falling drone; Gazebo's default (0.01) lags far behind. Runs in the
    # background, so the flight does not wait for the window.
    follow = ("for i in $(seq 60); do gz topic -i -t /gui/track | grep -q tcp && break; sleep 1; done; "
              "gz topic -t /gui/track -m gz.msgs.CameraTrack -p 'track_mode: FOLLOW_LOOK_AT, "
              "follow_target: {name: \"x500_0\"}, track_target: {name: \"x500_0\"}, "
              "follow_offset: {x: -4, y: 0, z: 2}, follow_pgain: 0.3, track_pgain: 0.3'")
    subprocess.Popen(["docker", "exec", VIEWER, "bash", "-c", follow],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_container() -> str:
    docker("rm", "-f", VIEWER, check=False)
    logs = docker("logs", CONTAINER, check=False).stdout
    docker("stop", "-t", "10", CONTAINER, check=False)
    docker("rm", "-f", CONTAINER, check=False)
    return logs


def _uid() -> int:
    import os
    return os.getuid()


def _gid() -> int:
    import os
    return os.getgid()


def wait_for(cond, timeout_s: float, what: str, poll_s: float = 0.5):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if cond():
                return
        except Exception:  # noqa: BLE001 - telemetry not ready yet
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"timed out after {timeout_s:.0f} s waiting for {what}")


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = north_m / 6_378_137.0
    dlon = east_m / (6_378_137.0 * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def fly(sc: Scenario, log_dir: Path, log=print, tlog: Path | None = None,
        gui: bool = False, world: str = "default") -> dict:
    """Fly one scenario. Returns run metadata (wall-clock times, notes)."""
    rng = random.Random(sc.seed)
    start_container(log_dir, world)
    meta = {"scenario": sc.name, "unit": sc.unit, "ftype": sc.ftype, "instance": sc.instance,
            "takeoff_alt_m": sc.takeoff_alt_m, "cruise_mps": sc.cruise_mps,
            "ca_failure_mode": sc.ca_failure_mode}
    recorder = Recorder(tlog) if tlog else None
    if recorder:
        recorder.start()
    try:
        mav = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
        mav.add_any_connection("udpin://0.0.0.0:14540")
        drone = mav.first_autopilot(120.0)
        if drone is None:
            raise RuntimeError("no autopilot found on udp 14540")
        if gui:   # PX4 is up, so the Gazebo world is ready to be viewed
            start_viewer()
        action, tel = Action(drone), Telemetry(drone)
        param, failure = Param(drone), Failure(drone)

        wait_for(tel.health_all_ok, 120, "health_all_ok")
        for name, value in {**PARAMS_INT, "CA_FAILURE_MODE": sc.ca_failure_mode}.items():
            param.set_param_int(name, value)
        if sc.cruise_mps:
            param.set_param_float("MPC_XY_CRUISE", sc.cruise_mps)
        action.set_takeoff_altitude(sc.takeoff_alt_m)
        # health_all_ok can be true before PX4 has registered the ground-station
        # heartbeat and the estimator has converged, so arming is retried until
        # PX4's own preflight checks accept it.
        t_arm = time.time()
        while True:
            try:
                action.arm()
                break
            except Exception:  # noqa: BLE001 - preflight checks not passed yet
                if time.time() - t_arm > 120:
                    raise
                time.sleep(2)
        meta["arm_wait_s"] = round(time.time() - t_arm, 1)
        action.takeoff()
        wait_for(lambda: tel.position().relative_altitude_m > sc.takeoff_alt_m - 1.0, 90,
                 "takeoff altitude")
        meta["t_hover_wall"] = time.time()

        # Random waypoints so flights are not identical: fly while the fault hits.
        pos = tel.position()
        t_end_pattern = time.time() + sc.inject_after_s
        while time.time() < t_end_pattern:
            lat, lon = offset_latlon(pos.latitude_deg, pos.longitude_deg,
                                     rng.uniform(-25, 25), rng.uniform(-25, 25))
            alt = max(pos.absolute_altitude_m - sc.takeoff_alt_m + 4.0,
                      pos.absolute_altitude_m + rng.uniform(*sc.alt_jitter_m))
            action.goto_location(lat, lon, alt, rng.uniform(-180, 180))
            time.sleep(min(rng.uniform(5, 9), max(0.0, t_end_pattern - time.time())))

        if sc.unit:
            # PX4 v1.17 executes sensor failures without always sending a
            # COMMAND_ACK, so MAVSDK reports TIMEOUT even though the command
            # arrived. Record the ack status and keep flying; whether the fault
            # really took effect is checked from the log (fault_effect()).
            try:
                failure.inject(UNITS[sc.unit], TYPES[sc.ftype], sc.instance)
                meta["inject_ack"] = "accepted"
            except Exception as e:  # noqa: BLE001
                meta["inject_ack"] = f"{type(e).__name__}: {e}"
            meta["t_inject_wall"] = time.time()
            log(f"  injected {sc.unit} {sc.ftype} instance {sc.instance}")
            time.sleep(sc.after_fault_s)

        try:
            action.land()
        except Exception as e:  # noqa: BLE001 - e.g. vehicle already crashed
            meta["land_error"] = str(e)
        try:
            wait_for(lambda: not tel.armed(), 60, "disarm after landing")
        except TimeoutError:
            action.kill()          # crashed/flipped vehicles may never auto-disarm
            meta["killed"] = True
            wait_for(lambda: not tel.armed(), 10, "disarm after kill")
        time.sleep(3)              # let the logger close the file
    finally:
        meta["container_log_tail"] = stop_container()[-2000:]
        if recorder:
            meta["tlog_messages"] = recorder.stop()
    return meta


def injection_time_us(path: Path) -> int | None:
    ulog = ULog(str(path), message_name_filter_list=["vehicle_command"])
    for d in ulog.data_list:
        hits = np.flatnonzero(d.data["command"] == MAV_CMD_INJECT_FAILURE)
        if len(hits):
            return int(d.data["timestamp"][hits[0]])
    return None


def fault_effect(path: Path, unit: str, instance: int, t_inj: int) -> dict:
    """Measure from the log whether the injected fault actually happened.

    motor: the failed motor's command after injection (0 or NaN = stopped)
    gps:   GPS message rate before vs after (Hz)
    baro:  barometer altitude std before vs after (stuck -> ~0)
    """
    ulog = ULog(str(path))
    def topic(*names):
        for n in names:
            try:
                return ulog.get_dataset(n, 0).data
            except (KeyError, IndexError, ValueError):
                continue
        return None
    win = 5e6
    if unit == "motor":
        d = topic("actuator_motors")
        t = d["timestamp"]
        after = d[f"control[{instance - 1}]"][(t > t_inj + 0.5e6) & (t < t_inj + win)]
        before = d[f"control[{instance - 1}]"][(t > t_inj - win) & (t < t_inj)]
        return {"motor_cmd_before": float(np.nanmean(before)),
                "motor_cmd_after": float(np.nan_to_num(np.nanmax(after), nan=0.0))
                if np.isfinite(after).any() else 0.0}
    if unit == "gps":
        d = topic("sensor_gps", "vehicle_gps_position")
        t = d["timestamp"]
        n_before = int(((t > t_inj - win) & (t < t_inj)).sum())
        n_after = int(((t > t_inj + 0.5e6) & (t < t_inj + 0.5e6 + win)).sum())
        return {"gps_hz_before": n_before / (win / 1e6), "gps_hz_after": n_after / (win / 1e6)}
    if unit == "baro":
        d = topic("vehicle_air_data")
        t, alt = d["timestamp"], d["baro_alt_meter"]
        return {"baro_std_before": float(np.std(alt[(t > t_inj - win) & (t < t_inj)])),
                "baro_std_after": float(np.std(alt[(t > t_inj + 0.5e6) & (t < t_inj + win)]))}
    return {}


def build_manifest(out: Path, runs: list[dict]) -> dict:
    files = []
    for run in runs:
        logs = sorted((out / run["scenario"] / "log").rglob("*.ulg"))
        if not logs:
            run["error"] = run.get("error") or "no ulog written"
            continue
        path = logs[-1]
        data = path.read_bytes()
        cls, annotations = "Normal", [{"class": "Normal", "note": "", "ranges": []}]
        if run["unit"]:
            t_inj = injection_time_us(path)
            if t_inj is None:
                run["error"] = "injection command not found in log"
                continue
            end = ULog(str(path), message_name_filter_list=["vehicle_status"]).last_timestamp
            cls = CLASS_OF_UNIT[run["unit"]]
            signal = f"injected {run['unit']} {run['ftype']} instance {run['instance']}"
            annotations = [{"class": cls, "note": signal, "ranges": [[signal, [[t_inj, int(end)]]]]}]
            run["fault_effect"] = fault_effect(path, run["unit"], run["instance"], t_inj)
        files.append({"key": run["scenario"], "path": str(path.relative_to(out)), "class": cls,
                      "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "annotations": annotations, "fault_effect": run.get("fault_effect"),
                      "tlog": (str((out / run["scenario"] / "telemetry.tlog").relative_to(out))
                               if (out / run["scenario"] / "telemetry.tlog").exists() else None),
                      "takeoff_alt_m": run.get("takeoff_alt_m"), "cruise_mps": run.get("cruise_mps"),
                      "ca_failure_mode": run.get("ca_failure_mode", 1),
                      "inject_ack": run.get("inject_ack")})
    (out / "runs.json").write_text(json.dumps(runs, indent=2, default=str))
    manifest = {"source": "px4_sitl", "revision": f"{IMAGE} gz_x500", "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", choices=sorted(SCENARIO_SETS), default="phase5")
    ap.add_argument("--no-tlog", action="store_true",
                    help="don't record UDP 14550 (e.g. while the live service uses it)")
    ap.add_argument("--skip-done", action="store_true", help="skip scenarios already in runs.json")
    ap.add_argument("--gui", action="store_true", help="open Gazebo's 3D view of each flight")
    ap.add_argument("--world", choices=WORLDS, default="default",
                    help="Gazebo world; recorded flights use 'default', 'town' adds scenery")
    args = ap.parse_args()
    todo = [s for s in SCENARIO_SETS[args.set]() if not args.only or s.name in args.only]

    runs_path = args.out / "runs.json"
    runs = {r["scenario"]: r for r in json.loads(runs_path.read_text())} if runs_path.exists() else {}
    for sc in todo:
        if args.skip_done and sc.name in runs and not runs[sc.name].get("error"):
            continue
        t0 = time.time()
        print(f"{sc.name}: flying", flush=True)
        try:
            tlog = None if args.no_tlog else args.out / sc.name / "telemetry.tlog"
            run = fly(sc, args.out / sc.name / "log", tlog=tlog, gui=args.gui, world=args.world)
        except Exception as e:  # noqa: BLE001 - record and continue with the next run
            stop_container()
            run = {"scenario": sc.name, "unit": sc.unit, "ftype": sc.ftype,
                   "instance": sc.instance, "error": f"{type(e).__name__}: {e}"}
        run["wall_s"] = round(time.time() - t0, 1)
        runs[sc.name] = run
        print(f"{sc.name}: done in {run['wall_s']} s {run.get('error') or ''}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        runs_path.write_text(json.dumps(list(runs.values()), indent=2, default=str))

    manifest = build_manifest(args.out, list(runs.values()))
    print(f"\n{len(manifest['files'])} labelled logs -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
