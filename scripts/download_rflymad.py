"""Download selected RflyMAD flights (real and hardware-in-the-loop) from Kaggle.

RflyMAD: Le, Quan et al., "RflyMAD: A Dataset for Multicopter Fault Detection
and Health Assessment", arXiv:2311.11340 (IJRR 2024). Beihang University.
License: non-commercial research use (the official page asks commercial users
to contact the authors); the data is never committed or redistributed here.

Kaggle stores the dataset unzipped (one entry per file), and its public API
lists and serves single files without an account (checked 2026-11-09). Only
the files phase 7 needs are fetched: per flight case the PX4 ULog (.ulg),
the ground-station telemetry (.tlog) and the test-info sheet (.xlsx). The
per-topic CSVs and ROS bags are duplicates of the ULog and are skipped.

Bandwidth here is ~1.5 MB/s, so `--per-group N` keeps N flight cases per
folder (e.g. Real-Motor/hover), evenly spaced through the sorted list, for
the ULogs. Tlogs are small and always fetched for every case.

Kaggle gives no checksums, so integrity is checked by exact byte size against
the listing. Re-running skips files that are already complete.

Standard library plus numpy.

Usage:
    uv run python scripts/download_rflymad.py --out data/raw/rflymad --set real --per-group 12
    uv run python scripts/download_rflymad.py --out data/raw/rflymad --set hil --per-group 12
"""

import argparse
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

API = "https://www.kaggle.com/api/v1/datasets"
OWNER = "xianglile"
SETS = {
    # dataset slug, case-directory prefixes, file types
    "real": ("rflymad", ("Real-Motor/", "Real-No_Fault/"), (".ulg", ".tlog", ".xlsx")),
    # HIL: the real Pixhawk running PX4, with simulated physics. Same fault
    # model as the real flights, so HIL -> real isolates the sim-to-real gap.
    "hil": ("rflymad-hil", ("HIL_Motor_1/hover-motor-", "HIL_Motor_2/waypoint-motor-1/",
                            "HIL-NoFault/"), (".ulg", ".tlog")),
}
CASE_SUBDIRS = {"Log", "TLog", "TrueData"}


def case_of(name: str) -> str:
    """Flight-case directory: the parent of Log/TLog/TrueData (HIL layout),
    else the first three path components (real-flight layout)."""
    parts = name.split("/")
    for i, part in enumerate(parts[:-1]):
        if part in CASE_SUBDIRS:
            return "/".join(parts[:i])
    return "/".join(parts[:3])


def select(files: list[tuple[str, int]], prefixes, exts, per_group: int | None):
    files = [(n, b) for n, b in files if n.startswith(prefixes) and n.lower().endswith(exts)]
    cases = sorted({case_of(n) for n, _ in files})
    if per_group:
        groups: dict[str, list[str]] = {}
        for c in cases:
            groups.setdefault(c.rsplit("/", 1)[0], []).append(c)
        keep = set()
        for cs in groups.values():
            idx = np.linspace(0, len(cs) - 1, min(per_group, len(cs))).round().astype(int)
            keep |= {cs[i] for i in idx}
    else:
        keep = set(cases)
    return [(n, b) for n, b in files if n.lower().endswith(".tlog") or case_of(n) in keep]


def get_json(url: str) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_files(slug: str, cache: Path) -> list[tuple[str, int]]:
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    files, token = [], None
    while True:
        url = f"{API}/list/{OWNER}/{slug}?pageSize=1000"
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        d = get_json(url)
        files += [(f["name"], int(f["totalBytes"])) for f in d["datasetFiles"]]
        token = d.get("nextPageTokenNullable")
        if not token:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(files))
    return files


def download(slug: str, name: str, size: int, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size == size:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/download/{OWNER}/{slug}?fileName=" + urllib.parse.quote(name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            # Kaggle serves larger files as a one-member ZIP archive (seen for
            # ULogs of ~10 MB and up); small files come raw. An .xlsx is itself
            # a ZIP, so only a single member named like the target is unpacked.
            if tmp.stat().st_size != size and zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as z:
                    members = z.infolist()
                    if len(members) == 1 and Path(members[0].filename).name == dest.name:
                        unz = dest.with_suffix(dest.suffix + ".unz")
                        with z.open(members[0]) as src, open(unz, "wb") as out:
                            shutil.copyfileobj(src, out, 1 << 20)
                        tmp.unlink()
                        tmp = unz
            if tmp.stat().st_size != size:
                raise OSError(f"size {tmp.stat().st_size} != {size}")
            tmp.replace(dest)
            return "ok"
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-group", type=int, default=None)
    args = ap.parse_args()
    slug, prefixes, exts = SETS[args.set]

    files = list_files(slug, args.out / f"index_{slug}.json")
    # tlogs first: small, and the live detector needs them
    todo = sorted(select(files, prefixes, exts, args.per_group),
                  key=lambda nb: (not nb[0].endswith(".tlog"), nb[0]))
    total = sum(b for _, b in todo)
    print(f"{len(todo)} files, {total / 1e9:.2f} GB", flush=True)

    done, failed = 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(download, slug, n, b, args.out / n): n for n, b in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                done += 1
            except Exception as e:  # noqa: BLE001 - report and continue
                failed.append((futs[fut], str(e)))
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)

    cases: dict[str, dict] = {}
    for n, _size in todo:
        if (args.out / n).exists():
            case = case_of(n)
            kind = Path(n).suffix.lower().lstrip(".")
            cases.setdefault(case, {})[kind] = n
    (args.out / f"cases_{args.set}.json").write_text(json.dumps(cases, indent=1))
    print(f"{done} ok, {len(failed)} failed, {len(cases)} cases")
    for n, e in failed[:20]:
        print("  FAILED", n, e)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
