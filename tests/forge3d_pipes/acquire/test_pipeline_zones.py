import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from eo_art.forge3d_pipes.acquire import pipeline
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig, SrConfig
from eo_art.forge3d_pipes.acquire.sentinel_bridge import (
    MANIFEST_NAME,
    Scene,
    read_candidates,
)
from eo_art.forge3d_pipes.acquire.zones import Zone, ZoneGrid

SENTINEL = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "model_path": "/models/LDSRS2-SEN2SR",
    "max_items": 3,
}

GRID = ZoneGrid(
    epsg=32632,
    tile_m=1280.0,
    zones=[
        Zone("z00_00", 0, 0, 47.01, 11.01, (0.0, 1280.0, 1280.0, 2560.0)),
        Zone("z00_01", 0, 1, 47.01, 11.03, (1280.0, 1280.0, 2560.0, 2560.0)),
        Zone("z01_00", 1, 0, 47.00, 11.01, (0.0, 0.0, 1280.0, 1280.0)),
    ],
    bounds=(0.0, 0.0, 2560.0, 2560.0),
)


def _sr_cfg(**sentinel):
    raw = OmegaConf.merge(
        OmegaConf.structured(SrConfig),
        {
            "aoi_path": "aoi/valley.gpkg",
            "sentinel_sr_dir": "../sentinel_sr",
            "sentinel": {**SENTINEL, "zones": {"enabled": True}},
        },
        {"sentinel": sentinel},
    )
    return OmegaConf.to_object(raw)


def _acquire_cfg(**overrides):
    raw = OmegaConf.merge(
        OmegaConf.structured(AcquireConfig),
        {
            "aoi_path": "aoi/valley.gpkg",
            "ee_project": "ee-test",
            "sentinel_sr_dir": "../sentinel_sr",
            "sentinel": {**SENTINEL, "zones": {"enabled": True}},
            "dtm": {"mode": "native"},
        },
        overrides,
    )
    return OmegaConf.to_object(raw)


def _wire(monkeypatch, *, empty=(), grid=GRID):
    """sentinel_sr stand-in writing one scene (plus manifest) per zone, except
    for the zones at the (lat, lon) of ``empty``."""
    calls = []

    def _run_sentinel_sr(cfg, sentinel_sr_dir, lat, lon, out_dir, **kw):
        calls.append(("run_sentinel_sr", lat, lon, cfg.max_items, out_dir, kw))
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = []
        if (lat, lon) not in empty:
            path = out_dir / "superres_a.tif"
            path.write_bytes(f"{lat},{lon}".encode())
            scenes.append(Scene(path, 1.0, "2023-05-01"))
        (out_dir / MANIFEST_NAME).write_text(
            '{"scenes": ['
            + ",".join(
                f'{{"path": "{s.path}", "cloud_cover": 1.0, "datetime": "2023-05-01"}}'
                for s in scenes
            )
            + '], "skipped_bins": []}'
        )
        return scenes

    def _build_vrt(tiles, dst):
        calls.append(("build_vrt", [Path(t).name for t in tiles], dst))
        Path(dst).write_text("vrt")
        return Path(dst)

    def _export_mosaic(vrt, dst):
        calls.append(("export_mosaic", vrt, dst))
        Path(dst).write_bytes(b"mosaic")
        return Path(dst)

    def _fetch_terrain_dem_bounds(bounds, crs, out_dir, **kw):
        calls.append(("fetch_terrain_dem_bounds", bounds, crs, kw))
        path = Path(out_dir) / "dem.tif"
        path.write_bytes(b"dem")
        return path

    def _fetch_local_dem_bounds(bounds, crs, out_dir, **kw):
        calls.append(("fetch_local_dem_bounds", bounds, crs, kw))
        path = Path(out_dir) / "dem.tif"
        path.write_bytes(b"localdem")
        return path

    def _erosion(dem, out, **kw):
        calls.append(("simulate_thermal_erosion", dem, out))
        Path(out).write_bytes(b"eroded")
        return out

    def _forbidden(name):
        def _fail(*args, **kwargs):
            raise AssertionError(f"{name} must not run for zones")

        return _fail

    monkeypatch.setattr(pipeline, "plan_zones", lambda *a, **kw: grid)
    monkeypatch.setattr(pipeline, "run_sentinel_sr", _run_sentinel_sr)
    monkeypatch.setattr(pipeline, "build_vrt", _build_vrt)
    monkeypatch.setattr(pipeline, "export_mosaic", _export_mosaic)
    monkeypatch.setattr(pipeline, "fetch_terrain_dem_bounds", _fetch_terrain_dem_bounds)
    monkeypatch.setattr(pipeline, "fetch_local_dem_bounds", _fetch_local_dem_bounds)
    monkeypatch.setattr(pipeline, "simulate_thermal_erosion", _erosion)
    for name in (
        "read_aoi_center",
        "fetch_terrain_dem",
        "fetch_local_dem",
        "super_resolve_dtm",
        "align_raster_grid",
    ):
        monkeypatch.setattr(pipeline, name, _forbidden(name))
    return calls


def _sr_calls(calls):
    return [c for c in calls if c[0] == "run_sentinel_sr"]


def test_run_sr_zones_runs_sentinel_sr_once_per_zone(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)

    result = pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")

    sr = _sr_calls(calls)
    assert [(c[1], c[2]) for c in sr] == [(z.lat, z.lon) for z in GRID.zones]
    # max_items applies per zone, each zone in its own cache dir.
    assert {c[3] for c in sr} == {3}
    assert len({c[4] for c in sr}) == 3
    assert result.out_dir == tmp_path / "out" / "valley_sr_zones"
    assert [p.name for p in result.tiles] == [
        "valley_sr_z00_00.tif",
        "valley_sr_z00_01.tif",
        "valley_sr_z01_00.tif",
    ]
    assert result.tiles[1].read_bytes() == b"47.01,11.03"
    assert result.vrt == result.out_dir / "valley_sr_zones.vrt"
    assert result.mosaic is None
    assert result.empty_zones == []


def test_run_sr_zones_writes_the_mosaic_on_request(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)

    result = pipeline.run_sr_zones(_sr_cfg(zones={"mosaic": True}), tmp_path / "out")

    assert result.mosaic == tmp_path / "out" / "valley_sr_mosaic.tif"
    assert [c for c in calls if c[0] == "export_mosaic"][0][1] == result.vrt


def test_run_sr_zones_resumes_with_the_open_zones(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)
    pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")
    calls.clear()

    pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")
    assert _sr_calls(calls) == []

    pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out", use_cache=False)
    assert len(_sr_calls(calls)) == 3
    assert all(c[5] == {"overwrite": True} for c in _sr_calls(calls))


def test_empty_zones_are_reported_not_fatal(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, empty={(47.01, 11.03)})

    result = pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")

    assert result.empty_zones == ["z00_01"]
    assert [p.name for p in result.tiles] == [
        "valley_sr_z00_00.tif",
        "valley_sr_z01_00.tif",
    ]
    vrt_call = [c for c in calls if c[0] == "build_vrt"][0]
    assert vrt_call[1] == ["valley_sr_z00_00.tif", "valley_sr_z01_00.tif"]

    # The empty zone's finished manifest is reused, not searched again.
    calls.clear()
    pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")
    assert _sr_calls(calls) == []


def test_all_zones_empty_raises(monkeypatch, tmp_path):
    _wire(monkeypatch, empty={(z.lat, z.lon) for z in GRID.zones})

    with pytest.raises(RuntimeError, match="any of the 3 zone"):
        pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")


def test_stale_tiles_of_an_earlier_grid_are_removed(monkeypatch, tmp_path):
    _wire(monkeypatch)
    stale = tmp_path / "out" / "valley_sr_zones" / "valley_sr_z05_05.tif"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")

    assert not stale.exists()


def test_run_sr_rejects_zones_and_run_sr_zones_requires_them(tmp_path):
    with pytest.raises(ValueError, match="run_sr_zones"):
        pipeline.run_sr(_sr_cfg(), tmp_path)
    with pytest.raises(ValueError, match="run_sr"):
        pipeline.run_sr_zones(_sr_cfg(zones={"enabled": False}), tmp_path)
    with pytest.raises(ValueError, match="run_sr_timeseries_zones"):
        pipeline.run_sr_timeseries(_sr_cfg(timeseries={"enabled": True}), tmp_path)


def test_disabled_zones_keep_the_cache_key():
    plain = _sr_cfg(zones={"enabled": False})
    tweaked = _sr_cfg(zones={"enabled": False, "mosaic": True})

    assert pipeline._cache_key(plain) == pipeline._cache_key(tweaked)
    assert pipeline._cache_key(plain) != pipeline._cache_key(_sr_cfg())


# Zone (lat, lon) -> its candidates as (date, cloud, bin).
TS_CANDIDATES = {
    (47.01, 11.01): [
        ("2024-01-03", 2.0, "2024-01-01"),
        ("2024-01-28", 1.0, "2024-01-01"),
        ("2024-02-10", 5.0, "2024-02-01"),
    ],
    (47.01, 11.03): [
        ("2024-01-03", 3.0, "2024-01-01"),
        ("2024-02-10", 4.0, "2024-02-01"),
    ],
    (47.00, 11.01): [
        ("2024-01-03", 9.0, "2024-01-01"),
        ("2024-01-28", 0.0, "2024-01-01"),
    ],
}
TS_BINS = ["2024-01-01", "2024-02-01", "2024-03-01"]


def _wire_timeseries(monkeypatch):
    calls = _wire(monkeypatch)

    def _list_candidates(cfg, sentinel_sr_dir, lat, lon, out_dir):
        calls.append(("list_candidates", lat, lon))
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = [
            {"id": f"S2_{d}", "datetime": d, "cloud_cover": c, "bin": b}
            for d, c, b in TS_CANDIDATES[(lat, lon)]
        ]
        (out_dir / "candidates.json").write_text(
            json.dumps({"scenes": scenes, "bins": TS_BINS})
        )
        return read_candidates(out_dir)

    def _run_sentinel_sr(cfg, sentinel_sr_dir, lat, lon, out_dir, **kw):
        calls.append(("run_sentinel_sr", lat, lon, kw.get("dates")))
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = []
        for date in kw["dates"]:
            path = out_dir / f"superres_{date}_x.tif"
            path.write_bytes(f"{lat},{lon},{date}".encode())
            scenes.append(Scene(path, 1.0, date))
        return scenes

    monkeypatch.setattr(pipeline, "list_sentinel_candidates", _list_candidates)
    monkeypatch.setattr(pipeline, "run_sentinel_sr", _run_sentinel_sr)
    return calls


def _ts(**timeseries):
    return _sr_cfg(timeseries={"enabled": True, **timeseries})


def test_zone_timeseries_shares_one_window_per_interval(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)

    result = pipeline.run_sr_timeseries_zones(
        _ts(max_zone_offset_days=14), tmp_path / "out"
    )

    ts = tmp_path / "out" / "valley_sr_timeseries"
    january, february = result.steps
    # 01-28 window: mean (1 + 0) / 2 beats 01-03 (2 + 3 + 9) / 3 only if it
    # covered as many zones -- it doesn't, so all three share 01-03.
    assert {p.date for p in january.step.picks.values()} == {"2024-01-03"}
    assert january.vrt == ts / "valley_sr_2024-01-03.vrt"
    assert january.tiles["z01_00"] == ts / "z01_00" / "valley_sr_2024-01-03.tif"
    assert february.step.empty_zones == ["z01_00"]
    assert result.skipped_bins == ["2024-03-01"]
    # Pass 2 asks each zone for exactly its chosen dates.
    sr = {(c[1], c[2]): c[3] for c in calls if c[0] == "run_sentinel_sr"}
    assert sr[(47.01, 11.01)] == ["2024-01-03", "2024-02-10"]
    assert sr[(47.00, 11.01)] == ["2024-01-03"]
    steps = json.loads((ts / "steps.json").read_text())
    assert steps["max_zone_offset_days"] == 14
    assert steps["steps"][0]["spread_days"] == 0
    assert steps["steps"][1]["empty_zones"] == ["z01_00"]


def test_zone_timeseries_without_offset_lets_zones_choose(monkeypatch, tmp_path):
    _wire_timeseries(monkeypatch)

    result = pipeline.run_sr_timeseries_zones(_ts(), tmp_path / "out")

    january = result.steps[0].step
    assert january.picks["z00_00"].date == "2024-01-28"
    assert january.picks["z00_01"].date == "2024-01-03"
    assert january.spread_days == 25
    assert result.steps[0].vrt.name == "valley_sr_2024-01-03.vrt"


def test_zone_timeseries_mosaic_per_step(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)

    result = pipeline.run_sr_timeseries_zones(
        _sr_cfg(timeseries={"enabled": True}, zones={"enabled": True, "mosaic": True}),
        tmp_path / "out",
    )

    ts = tmp_path / "out" / "valley_sr_timeseries"
    assert [s.mosaic for s in result.steps] == [
        ts / "valley_sr_2024-01-03.tif",
        ts / "valley_sr_2024-02-10.tif",
    ]
    # The mosaics make the top folder a regular timeseries.
    assert list(pipeline.timeseries_frames("aoi/valley.gpkg", ts)) == [
        "2024-01-03",
        "2024-02-10",
    ]
    assert len([c for c in calls if c[0] == "export_mosaic"]) == 2


def test_zone_timeseries_reuses_candidates_across_offsets(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)
    pipeline.run_sr_timeseries_zones(_ts(), tmp_path / "out")
    calls.clear()

    pipeline.run_sr_timeseries_zones(_ts(max_zone_offset_days=3), tmp_path / "out")

    assert [c for c in calls if c[0] == "list_candidates"] == []


def test_zone_timeseries_reset_and_fill_gaps(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)
    out = tmp_path / "out"
    ts = out / "valley_sr_timeseries"
    pipeline.run_sr_timeseries_zones(_ts(), out)
    stale = ts / "z00_00" / "valley_sr_2023-06-01.tif"
    stale.write_bytes(b"old")

    # Without fill_gaps the folder is rebuilt: stale steps go.
    pipeline.run_sr_timeseries_zones(_ts(), out)
    assert not stale.exists()

    # With fill_gaps finished intervals are kept and not chosen again.
    calls.clear()
    result = pipeline.run_sr_timeseries_zones(_ts(fill_gaps=True), out)
    assert result.steps == []
    assert len(json.loads((ts / "steps.json").read_text())["steps"]) == 2
    assert all(c[3] is None for c in calls if c[0] == "run_sentinel_sr")


def test_zone_timeseries_offset_keeps_the_cache_key():
    assert pipeline._cache_key(_ts()) == pipeline._cache_key(
        _ts(max_zone_offset_days=14)
    )


def test_acquire_zones_fetches_one_dem_and_aligns_nothing(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)

    result = pipeline.run_acquire(_acquire_cfg(), tmp_path / "out")

    dem_calls = [c for c in calls if c[0] == "fetch_terrain_dem_bounds"]
    assert len(dem_calls) == 1
    assert dem_calls[0][1:3] == (GRID.bounds, "EPSG:32632")
    assert result.dtm_path == tmp_path / "out" / "dtm.tif"
    assert result.dtm_path.read_bytes() == b"dem"
    assert result.optical_path is None
    assert result.zones is not None
    assert len(result.zones.tiles) == 3


def test_acquire_zones_local_dem_and_erosion(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)
    cfg = _acquire_cfg(
        ee_project=None,
        dem={"source": "local", "path": "/dem/dgm1.tif"},
        dtm={"mode": "native", "apply_erosion": True},
    )

    result = pipeline.run_acquire(cfg, tmp_path / "out")

    local = [c for c in calls if c[0] == "fetch_local_dem_bounds"]
    assert local[0][3]["path"] == "/dem/dgm1.tif"
    assert result.dtm_path.read_bytes() == b"eroded"


def test_acquire_zones_shares_the_sr_cache_with_run_sr_zones(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)
    pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")
    calls.clear()

    pipeline.run_acquire(_acquire_cfg(), tmp_path / "out")

    assert _sr_calls(calls) == []
    # A DEM change reuses the zone SR, only the DEM is fetched again.
    pipeline.run_acquire(_acquire_cfg(dem={"scale": 90.0}), tmp_path / "out")
    assert _sr_calls(calls) == []
    assert len([c for c in calls if c[0] == "fetch_terrain_dem_bounds"]) == 2


def test_export_zones_mosaic_from_an_earlier_run(monkeypatch, tmp_path):
    calls = _wire(monkeypatch)
    with pytest.raises(FileNotFoundError, match="sentinel.zones"):
        pipeline.export_zones_mosaic("aoi/valley.gpkg", tmp_path / "out")

    result = pipeline.run_sr_zones(_sr_cfg(), tmp_path / "out")
    mosaic = pipeline.export_zones_mosaic("aoi/valley.gpkg", tmp_path / "out")

    assert mosaic == tmp_path / "out" / "valley_sr_mosaic.tif"
    assert [c for c in calls if c[0] == "export_mosaic"][0][1] == result.vrt


def test_acquire_zones_ignores_guided_upsample(monkeypatch, tmp_path, capsys):
    calls = _wire(monkeypatch)
    cfg = _acquire_cfg(dtm={"mode": "guided_upsample"})

    result = pipeline.run_acquire(cfg, tmp_path / "out")

    # super_resolve_dtm / align_raster_grid would raise via _forbidden.
    assert result.dtm_path.read_bytes() == b"dem"
    assert len([c for c in calls if c[0] == "fetch_terrain_dem_bounds"]) == 1
    assert "dtm.mode='guided_upsample' ignored" in capsys.readouterr().out
