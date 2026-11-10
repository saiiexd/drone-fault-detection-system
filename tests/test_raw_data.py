"""Smoke tests for the downloaded UAV-SEAD subset (phase 1).

Skipped when the data has not been downloaded, so CI without data still passes.
"""

import hashlib
import json
from pathlib import Path

import pytest
from pyulog import ULog

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "uav_sead"
MANIFEST = RAW / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="UAV-SEAD subset not downloaded")


SAMPLE_SIZE = 10


def manifest_files():
    """A deterministic, evenly spread sample of the manifest (keeps tests fast)."""
    if not MANIFEST.exists():
        return []
    # Only files present on disk: the .ulg files may be deleted to save space
    # once the lake is built (the manifest stays as the record of what was used).
    files = [f for f in json.loads(MANIFEST.read_text())["files"] if (RAW / f["path"]).exists()]
    step = max(1, len(files) // SAMPLE_SIZE)
    return files[::step][:SAMPLE_SIZE]


@pytest.mark.parametrize("entry", manifest_files(), ids=lambda e: e["key"])
def test_file_matches_manifest(entry):
    data = (RAW / entry["path"]).read_bytes()
    assert len(data) == entry["bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("entry", manifest_files(), ids=lambda e: e["key"])
def test_load_topics_contract(entry):
    from aeropulse.ingest.ulog_reader import load_topics

    topics = load_topics(RAW / entry["path"], ["actuator_outputs", "no_such_topic"])
    # Instance 0 always exists; the octorotor logs a second instance, the quad not.
    # The missing topic is skipped, not raised.
    assert "actuator_outputs_0" in topics
    assert set(topics) <= {"actuator_outputs_0", "actuator_outputs_1"}
    for df in topics.values():
        assert df["timestamp"].dtype == "int64"
        assert df["timestamp"].is_monotonic_increasing


@pytest.mark.parametrize("entry", manifest_files(), ids=lambda e: e["key"])
def test_pyulog_opens_file(entry):
    ulog = ULog(str(RAW / entry["path"]))
    assert ulog.data_list, "log contains no data messages"
    assert ulog.last_timestamp > ulog.start_timestamp
