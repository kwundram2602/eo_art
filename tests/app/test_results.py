import os

from eo_art.app.projects import create_project
from eo_art.app.results import (
    is_timeseries_folder,
    media_files,
    project_tifs,
    run_logs,
    timeseries_dirs,
    zone_dirs,
    zone_mosaic,
)
from eo_art.app.runner import PipelineKind


def _touch(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_media_files_newest_first_and_skip_internal_dirs(tmp_path):
    project = create_project(tmp_path, "alps")
    old = _touch(project.out_dir / "a.png", 1_000)
    new = _touch(project.out_dir / "look" / "b.mp4", 2_000)
    _touch(project.out_dir / "_prep" / "c.png")
    _touch(project.out_dir / "look" / "frames" / "0001.png")
    _touch(project.out_dir / "dtm.tif")

    assert media_files(project) == [new, old]


def test_project_tifs_order_sr_then_timeseries_then_rest(tmp_path):
    project = create_project(tmp_path, "alps")
    out = project.out_dir
    dtm = _touch(out / "dtm.tif", 3_000)
    sr = _touch(out / "alps_sr.tif", 1_000)
    feb = _touch(out / "alps_sr_timeseries" / "alps_sr_2023-02-01.tif")
    jan = _touch(out / "alps_sr_timeseries" / "alps_sr_2023-01-01.tif")

    assert project_tifs(project) == [sr, jan, feb, dtm]
    assert timeseries_dirs(project) == [out / "alps_sr_timeseries"]


def test_run_logs_filter_by_kind_newest_first(tmp_path):
    project = create_project(tmp_path, "alps")
    logs = project.logs_dir
    a1 = _touch(logs / "20260101-100000-acquire.log")
    sr = _touch(logs / "20260102-100000-sr.log")
    render = _touch(logs / "20260103-100000-render.log")
    both = _touch(logs / "20260104-100000-acquire_render.log")
    _touch(logs / "notes.log")

    assert run_logs(project, (PipelineKind.acquire, PipelineKind.sr)) == [sr, a1]
    assert run_logs(project, (PipelineKind.render, PipelineKind.acquire_render)) == [
        both,
        render,
    ]


def test_zone_folders_tiles_and_mosaic(tmp_path):
    project = create_project(tmp_path, "alps")
    out = project.out_dir
    dtm = _touch(out / "dtm.tif", 3_000)
    mosaic = _touch(out / "alps_sr_mosaic.tif", 1_000)
    tile_b = _touch(out / "alps_sr_zones" / "alps_sr_z00_01.tif")
    tile_a = _touch(out / "alps_sr_zones" / "alps_sr_z00_00.tif")

    assert project_tifs(project) == [mosaic, tile_a, tile_b, dtm]
    assert zone_dirs(project) == [out / "alps_sr_zones"]
    assert zone_mosaic(out / "alps_sr_zones") == mosaic


def test_timeseries_per_zone(tmp_path):
    project = create_project(tmp_path, "alps")
    out = project.out_dir
    ts = out / "alps_sr_timeseries"
    step = _touch(ts / "z00_00" / "alps_sr_2023-01-01.tif")

    assert timeseries_dirs(project) == [ts, ts / "z00_00"]
    assert step in project_tifs(project)
    assert is_timeseries_folder(ts / "z00_00")
    assert is_timeseries_folder(ts)
    assert not is_timeseries_folder(out / "alps_sr_zones")


def test_results_are_grouped_per_run(tmp_path):
    from eo_art.app.results import ResultGroup, result_groups
    from eo_art.forge3d_pipes.acquire.runs import link_latest

    project = create_project(tmp_path, "alps")
    out = project.out_dir
    runs = out / "runs"
    old = _touch(runs / "20260101-000000-sr" / "alps_sr.tif")
    dtm = _touch(runs / "20260102-000000-acquire" / "dtm.tif")
    tile = _touch(
        runs / "20260102-000000-acquire" / "alps_sr_zones" / "alps_sr_z00_00.tif"
    )
    step = _touch(
        runs
        / "20260103-000000-sr"
        / "alps_sr_timeseries"
        / "z00_00"
        / "alps_sr_2024-01-03.tif"
    )
    legacy = _touch(out / "optical_aligned.tif")
    log = _touch(project.logs_dir / "20260102-000000-acquire.log")
    link_latest(out, "acquire", dtm.parent)

    groups = result_groups(project)

    assert [g.root.name for g in groups] == [
        "20260103-000000-sr",
        "20260102-000000-acquire",
        "20260101-000000-sr",
        "out",
    ]
    acquire = groups[1]
    assert isinstance(acquire, ResultGroup)
    assert (acquire.kind, acquire.latest, acquire.log) == ("acquire", True, log)
    assert acquire.rasters == [dtm]
    assert acquire.zone_dirs == [tile.parent]
    assert groups[0].timeseries_dirs == [step.parent.parent, step.parent]
    assert groups[2].rasters == [old] and groups[2].latest is False
    assert groups[3].rasters == [legacy] and groups[3].kind is None
    # The stable latest-acquire paths come first for config pickers.
    tifs = project_tifs(project)
    latest = runs / "latest-acquire"
    assert tifs[:2] == [
        latest / "alps_sr_zones" / "alps_sr_z00_00.tif",
        latest / "dtm.tif",
    ]
    assert {old, dtm, tile, step, legacy} <= set(tifs)
