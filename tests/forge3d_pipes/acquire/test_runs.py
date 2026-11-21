from dataclasses import dataclass
from datetime import datetime

from eo_art.forge3d_pipes.acquire import runs


def test_new_run_dir_is_timestamped_and_never_reused(tmp_path):
    run_id = runs.new_run_id("sr", datetime(2026, 11, 22, 10, 15, 0))
    assert run_id == "20261122-101500-sr"

    first = runs.new_run_dir(tmp_path, "sr", run_id)
    second = runs.new_run_dir(tmp_path, "sr", run_id)

    assert first == tmp_path / "runs" / "20261122-101500-sr"
    assert second == tmp_path / "runs" / "20261122-101500-sr-2"
    assert first.is_dir() and second.is_dir()


def test_latest_link_follows_the_newest_finished_run(tmp_path):
    assert runs.latest_run(tmp_path, "acquire") is None
    old = runs.new_run_dir(tmp_path, "acquire", "20260101-000000-acquire")
    new = runs.new_run_dir(tmp_path, "acquire", "20260102-000000-acquire")

    runs.link_latest(tmp_path, "acquire", old)
    runs.link_latest(tmp_path, "acquire", new)

    link = tmp_path / "runs" / "latest-acquire"
    assert link.is_symlink() and not link.readlink().is_absolute()
    assert runs.latest_run(tmp_path, "acquire") == new
    assert runs.list_runs(tmp_path) == [new, old]


def test_write_run_config(tmp_path):
    @dataclass
    class Cfg:
        aoi_path: str = "aoi/a.gpkg"

    path = runs.write_run_config(tmp_path, Cfg())

    assert path.read_text() == "aoi_path: aoi/a.gpkg\n"
