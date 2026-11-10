"""Live detection service: MAVLink in, alarms and a dashboard out.

    PX4 --UDP 14550--> reader thread --> LiveDetector --> asyncio queue --> WebSocket --> browser
                           |                                   |
                       session.tlog                     decisions.jsonl

Design choices:
- The reader and detector run in one plain thread. A blocking UDP read
  plus ~1 ms of model inference per frame would stall an asyncio event
  loop, and keeping detection on its own thread means a slow browser can
  never delay an alarm. Only finished decisions cross into asyncio, via
  loop.call_soon_threadsafe.
- No message broker (Kafka/Redpanda). There is one vehicle, one consumer
  and one host; a broker would add a network hop and an operations burden
  for no benefit. The transport is behind LiveLink/read_tlog, so a broker
  consumer could replace it if many vehicles or consumers appeared.
- The browser gets 10 Hz updates (every 5th frame) plus every alarm onset
  immediately. Charts don't need 50 Hz; alarms must not wait.
- Every decision is appended to decisions.jsonl with the autopilot time
  (t_us) and host times, so latency can be measured afterwards without
  comparing two different clocks (see stream.live_eval).

Usage:
    uv run python -m aeropulse.stream.service                      # live SITL on UDP 14550
    uv run python -m aeropulse.stream.service --replay data/raw/sitl6/p6_motor_off_2/telemetry.tlog
Then open http://127.0.0.1:8000
"""

import argparse
import asyncio
import json
import math
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from aeropulse.stream.detector import LiveDetector
from aeropulse.stream.features import MotorScale
from aeropulse.stream.mavlink_io import GCS_URL, LiveLink, decimate, read_tlog

STATIC = Path(__file__).parent / "static"
UI_EVERY = 5
HISTORY = 200          # UI records kept for late joiners: 20 s at 10 Hz


class Hub:
    """Shared state between the detector thread and the web side."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.history: deque = deque(maxlen=HISTORY)
        self.state = {"frames": 0, "alarms": [], "session": 0, "source": None,
                      "threshold": None, "last": None, "proc_us_max": 0.0}


def run_detector(hub: Hub, args, stop: threading.Event) -> None:
    det = LiveDetector(args.model, MotorScale(args.motor_lo, args.motor_hi, args.n_motors))
    hub.state["threshold"] = det.threshold
    args.session_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.session_dir / "decisions.jsonl", "a", buffering=1)
    if args.replay:
        source, link = replay_paced(args.replay, args.speed, stop), None
        hub.state["source"] = f"replay {args.replay.name} x{args.speed}"
    else:
        link = LiveLink(args.url, args.session_dir / "session.tlog")
        source = link.messages(stop)
        hub.state["source"] = f"live {args.url}"
    if det.radio:
        # A radio-trained model on a 50 Hz source (SITL, a recording): thin
        # the input to what a telemetry radio delivers, as in training.
        source = decimate(source)
        hub.state["source"] += " (thinned to radio rates)"
    try:
        for msg, rx_ns in source:
            d = det.update(msg, rx_ns)
            if d is None:
                continue
            hub.state["frames"] += 1
            hub.state["session"] = d.session
            hub.state["proc_us_max"] = max(hub.state["proc_us_max"], d.proc_us)
            rec = asdict(d)
            log.write(json.dumps(rec) + "\n")
            if d.onset:
                hub.state["alarms"].append({"session": d.session, "t_us": d.t_us,
                                            "score": d.score, "wall": time.time()})
            if d.onset or hub.state["frames"] % UI_EVERY == 0:
                rec = {**rec, "push_ns": time.monotonic_ns()}
                hub.state["last"] = rec
                if hub.history and hub.history[-1]["session"] != d.session:
                    hub.history.clear()
                hub.history.append(rec)
                if hub.loop and hub.queue is not None:
                    hub.loop.call_soon_threadsafe(_put_latest, hub.queue, rec)
    finally:
        log.close()
        if link:
            link.close()


def browser_safe(x):
    """NaN/inf -> None, recursively. Strict JSON has no NaN: Python's json.dumps
    would write a bare NaN that the browser's JSON.parse rejects (seen for the
    score before a window fills and for signals a radio link does not carry)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: browser_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [browser_safe(v) for v in x]
    return x


def _put_latest(q: asyncio.Queue, item) -> None:
    if q.full():                       # never let a slow client build a backlog
        q.get_nowait()
    q.put_nowait(item)


def replay_paced(path: Path, speed: float, stop: threading.Event):
    """Replay a tlog at its recorded pace (speed 0 = as fast as possible).
    rx_ns is replaced by the replay-time monotonic clock, like a live link."""
    t_first, w_first = None, None
    for msg, rec_ns in read_tlog(path):
        if stop.is_set():
            return
        if speed > 0:
            if t_first is None:
                t_first, w_first = rec_ns, time.monotonic_ns()
            due = w_first + (rec_ns - t_first) / speed
            delay = (due - time.monotonic_ns()) / 1e9
            if delay > 0:
                time.sleep(delay)
        yield msg, time.monotonic_ns()


def create_app(args) -> FastAPI:
    hub = Hub()
    stop = threading.Event()

    async def broadcaster():
        while True:
            rec = await hub.queue.get()
            dead = []
            msg = json.dumps(browser_safe(rec), allow_nan=False)
            for ws in list(hub.clients):
                try:
                    await ws.send_text(msg)
                except Exception:  # noqa: BLE001 - client went away
                    dead.append(ws)
            for ws in dead:
                hub.clients.discard(ws)

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue(maxsize=50)
        task = asyncio.create_task(broadcaster())
        th = threading.Thread(target=run_detector, args=(hub, args, stop), daemon=True)
        th.start()
        yield
        stop.set()
        th.join(timeout=3)
        task.cancel()

    app = FastAPI(title="aeropulse live", lifespan=lifespan)

    @app.get("/")
    async def index():
        # The recent history is inlined, so a browser that opens mid-flight
        # draws the last 20 s at once instead of waiting for new frames.
        boot = {"threshold": hub.state["threshold"], "source": hub.state["source"],
                "history": list(hub.history)}
        html = (STATIC / "dashboard.html").read_text()
        return HTMLResponse(html.replace("/*BOOT*/null", json.dumps(browser_safe(boot), allow_nan=False)))

    @app.get("/api/state")
    async def state():
        return JSONResponse(browser_safe(hub.state))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.add(websocket)
        await websocket.send_text(json.dumps(browser_safe({"hello": True, "threshold": hub.state["threshold"],
                                                           "source": hub.state["source"]}), allow_nan=False))
        try:
            while True:
                await websocket.receive_text()   # keeps the connection open
        except WebSocketDisconnect:
            hub.clients.discard(websocket)

    return app


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("data/models/stream_gbt.joblib"))
    ap.add_argument("--url", default=GCS_URL)
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--session-dir", type=Path,
                    default=Path("data/stream/sessions") / time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--motor-lo", type=float, default=0.0)
    ap.add_argument("--motor-hi", type=float, default=1000.0)
    ap.add_argument("--n-motors", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args(argv)


def main() -> None:
    import uvicorn
    args = parse_args()
    # Fail before starting anything: otherwise uvicorn runs the detector,
    # then fails to bind, and a browser on this port sees some other server.
    import socket
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # as uvicorn does
        try:
            s.bind((args.host, args.port))
        except OSError:
            raise SystemExit(f"error: port {args.port} is already in use, probably by another dashboard. "
                             f"Stop it with Ctrl+C in its terminal, or use --port {args.port + 1}.") from None
    source = f"replaying {args.replay}" if args.replay else f"waiting for MAVLink on {args.url}"
    print(f"aeropulse dashboard: http://{args.host}:{args.port}  ({source}; Ctrl+C to stop)", flush=True)
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
