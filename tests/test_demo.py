"""The portable demonstration runner over a synthetic folder, end to end."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_demo  # noqa: E402
from fl3_helpers import make_folder


def test_verify_demo_passes_on_a_synthetic_folder(tmp_path):
    media = make_folder(tmp_path, videos=3, images=2, sidecars=False)
    report = verify_demo.run(media, tmp_path / "state", tmp_path / "report.json")
    assert report["status"] == "PASS", report["failures"]
    assert report["folder_files"] == {"total": 5, "videos": 3, "images": 2} and report["synthetic_sidecars_written"] == 5
    assert report["server"]["loopback_only"] is True and report["server_stopped"] is True
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["status"] == "PASS" and all(v["evidence"].startswith("synthetic") for v in written["synthetic_checks"].values())
    assert set(Path(p).name for p in (tmp_path / "state").iterdir()) >= {"source.sqlite", "ledger.sqlite", "tuner.sqlite", "spaces"}
    assert json.loads((media / "sample-00.mp4.json").read_text())["note"].startswith("synthetic")
    # Search results carry the fixture's own segment starts, never an invented 0.0 or None.
    search = written["search"]
    results = search["results"]
    positions = [r["best_t"] for r in results]
    assert results and all(isinstance(t, float) and math.isfinite(t) and t > 0.0 for t in positions), positions
    assert 0.0 not in positions and None not in positions
    assert all(r["best_t"] == r["expected_best_t"] for r in results), results
    # independent oracle: the sidecars the runner wrote, read back here
    query = search["query"]
    sidecar_starts = {}
    for sidecar in media.glob("*.mp4.json"):
        payload = json.loads(sidecar.read_text())
        sidecar_starts[payload["title"]] = {tag: segment["start_s"] for segment in payload["segments"] for tag in segment["tags"]}
    supplied = {start for starts in sidecar_starts.values() for start in starts.values()}
    assert supplied == {0.0, 30.0, 60.0} and set(positions) <= supplied, (positions, supplied)
    carrying = [r for r in results if query in sidecar_starts[r["title"]]]
    assert carrying and all(r["best_t"] == sidecar_starts[r["title"]][query] > 0.0 for r in carrying), (query, carrying)


def test_verify_demo_fails_honestly_without_a_video(tmp_path):
    media = make_folder(tmp_path, videos=0, images=2, sidecars=False)
    report = verify_demo.run(media, tmp_path / "state", tmp_path / "report.json")
    assert report["status"] == "FAIL" and "at_least_one_video" in report["failures"]
