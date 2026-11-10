"""Download a small, class-balanced subset of UAV-SEAD ULog files.

Source: https://huggingface.co/datasets/aykutkabaoglu/uav-flight-anomaly-dataset
(the repo cited in arXiv:2602.13900). License: CC BY 4.0.

Standard library only, so it runs before the project venv exists.
The revision is pinned so the subset is reproducible.

Usage:
    uv run python scripts/download_uavsead.py --out data/raw/uav_sead
    uv run python scripts/download_uavsead.py --out data/raw/uav_sead --per-class Normal=4 Mechanical=2
    uv run python scripts/download_uavsead.py --out data/raw/uav_sead --all   # all annotated, ~11.4 GB
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "aykutkabaoglu/uav-flight-anomaly-dataset"
REVISION = "774a336a1d1d06b248927a35c2ea1f6869e9e1a4"  # main as of 2026-11-08
API = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"

DEFAULT_PER_CLASS = {
    "Normal": 4,
    "Mechanical": 2,
    "External Position": 2,
    "Altitude": 1,
    "Global Position": 1,
}
MIN_BYTES, MAX_BYTES = 1_000_000, 15_000_000  # skip tiny/aborted logs and big ones


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def flight_class(entry: dict) -> str:
    return " + ".join(sorted({a["class"] for a in entry["annotations"]}))


def has_ranges(entry: dict) -> bool:
    return any(a["ranges"] for a in entry["annotations"])


def select(mapping: dict, sizes: dict, per_class: dict) -> list[str]:
    """Smallest files in the size window, one per recording date, per class.
    Anomalous flights must have labelled time ranges."""
    picked = []
    for cls, n in per_class.items():
        cands = sorted(
            (k for k in mapping
             if k in sizes
             and flight_class(mapping[k]) == cls
             and MIN_BYTES < sizes[k] < MAX_BYTES
             and (cls == "Normal" or has_ranges(mapping[k]))),
            key=lambda k: sizes[k],
        )
        dates, chosen = set(), []
        for k in cands:
            date = k.split("/")[0]
            if date in dates:
                continue
            dates.add(date)
            chosen.append(k)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            print(f"warning: only {len(chosen)}/{n} candidates for {cls!r}", file=sys.stderr)
        picked += chosen
    return picked


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(out: Path, rel: str, blob: dict, retries: int = 3) -> tuple[int, str]:
    """Download (if needed) and verify one file. Returns (bytes, sha256)."""
    dest = out / rel
    expected_sha = (blob.get("lfs") or {}).get("sha256")
    for attempt in range(retries):
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            try:
                with urllib.request.urlopen(RESOLVE + urllib.request.quote(rel),
                                            timeout=120) as r, tmp.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                tmp.rename(dest)
            except OSError:
                tmp.unlink(missing_ok=True)
                time.sleep(2 ** attempt)
                continue
        size, sha = dest.stat().st_size, sha256_file(dest)
        if size == blob.get("size") and (expected_sha is None or sha == expected_sha):
            return size, sha
        dest.unlink()  # corrupt or partial: fetch again
    raise RuntimeError(f"failed after {retries} attempts: {rel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", nargs="*", default=None,
                    help="e.g. Normal=4 Mechanical=2 (overrides defaults)")
    ap.add_argument("--all", action="store_true",
                    help="download every annotated flight (ignores --per-class)")
    ap.add_argument("--exclude-class", nargs="*", default=["Uncategorized"],
                    help="classes skipped with --all (default: Uncategorized)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    per_class = DEFAULT_PER_CLASS
    if args.per_class:
        per_class = {k: int(v) for k, v in (s.split("=", 1) for s in args.per_class)}

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(fetch(API))
    blobs = {s["rfilename"]: s for s in meta["siblings"]}

    mapping_bytes = fetch(RESOLVE + "mapping.json")
    (out / "mapping.json").write_bytes(mapping_bytes)
    mapping = json.loads(mapping_bytes)

    # mapping keys are paths under ulg_files/ without the .ulg suffix
    sizes = {f[len("ulg_files/"):-len(".ulg")]: s.get("size", 0)
             for f, s in blobs.items()
             if f.startswith("ulg_files/") and f.endswith(".ulg")}

    if args.all:
        excluded = set(args.exclude_class)
        keys = sorted(k for k in mapping
                      if k in sizes and flight_class(mapping[k]) not in excluded)
    else:
        keys = select(mapping, sizes, per_class)
    missing = [k for k in mapping if k not in sizes]
    if missing:
        print(f"note: {len(missing)} annotated key(s) have no file: {missing}", file=sys.stderr)

    total = sum(sizes[k] for k in keys)
    print(f"{len(keys)} files, {total / 1e9:.2f} GB, {args.workers} workers")

    manifest, failed = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download_one, out, f"ulg_files/{k}.ulg",
                               blobs[f"ulg_files/{k}.ulg"]): k for k in keys}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                size, sha = fut.result()
            except RuntimeError as e:
                failed.append(key)
                print(f"FAIL {e}", file=sys.stderr)
                continue
            manifest.append({
                "key": key,
                "path": f"ulg_files/{key}.ulg",
                "class": flight_class(mapping[key]),
                "bytes": size,
                "sha256": sha,
                "annotations": mapping[key]["annotations"],
            })
            if i % 50 == 0 or i == len(keys):
                print(f"{i}/{len(keys)} done", flush=True)

    manifest.sort(key=lambda e: e["key"])
    (out / "manifest.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "files": manifest}, indent=2))
    print(f"\n{len(manifest)} files verified, {len(failed)} failed, "
          f"{sum(m['bytes'] for m in manifest) / 1e9:.2f} GB -> {out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
