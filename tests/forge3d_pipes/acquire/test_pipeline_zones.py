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

    def _build_vrt(tiles, dst, bounds=None):
        calls.append(("build_vrt", [Path(t).name for t in tiles], dst, bounds))
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


def _wire_timeseries(monkeypatch, *, drop=()):
    """Stand-ins for the zoned timeseries: candidates, 10 m fetch (with SCL),
    co-registration (dropping the dates in ``drop``) and whole-raster SR."""
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
        calls.append(("run_sentinel_sr", lat, lon, kw.get("dates"), kw.get("lr_only")))
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = []
        for date in kw["dates"]:
            path = out_dir / f"lr_{date}_x.tif"
            path.write_bytes(f"{lat},{lon},{date}".encode())
            scl = out_dir / f"lr_{date}_x_scl.tif"
            scl.write_bytes(b"scl")
            scenes.append(Scene(path, 1.0, date, scl_path=scl))
        return scenes

    def _run_sentinel_coreg(cfg, sentinel_sr_dir, jobs, jobs_path):
        calls.append(("coreg", [(j["date"], Path(j["src"]).name) for j in jobs]))
        report = []
        for j in jobs:
            if j["date"] in drop:
                report.append(
                    {**j, "dst": None, "shift_x_px": None, "shift_y_px": None}
                )
                continue
            Path(j["dst"]).write_bytes(Path(j["src"]).read_bytes() + b"+cr")
            report.append({**j, "shift_x_px": 0.5, "shift_y_px": -0.25})
        return report

    def _run_sentinel_superres(cfg, sentinel_sr_dir, jobs, jobs_path, overwrite=False):
        calls.append(
            ("superres", [(Path(j["src"]).name, Path(j["dst"]).name) for j in jobs])
        )
        for j in jobs:
            Path(j["dst"]).parent.mkdir(parents=True, exist_ok=True)
            Path(j["dst"]).write_bytes(b"sr:" + Path(j["src"]).read_bytes())

    def _merge_first_valid(parts, dst):
        calls.append(("merge", [Path(p).name for p in parts]))
        Path(dst).write_bytes(b"merged")
        return Path(dst)

    monkeypatch.setattr(pipeline, "list_sentinel_candidates", _list_candidates)
    monkeypatch.setattr(pipeline, "run_sentinel_sr", _run_sentinel_sr)
    monkeypatch.setattr(pipeline, "run_sentinel_coreg", _run_sentinel_coreg)
    monkeypatch.setattr(pipeline, "run_sentinel_superres", _run_sentinel_superres)
    monkeypatch.setattr(pipeline, "merge_first_valid", _merge_first_valid)
    return calls


def _ts(**timeseries):
    return _sr_cfg(timeseries={"enabled": True, **timeseries})


def test_zone_timeseries_fetches_10m_mosaics_coregisters_then_srs(
    monkeypatch, tmp_path
):
    calls = _wire_timeseries(monkeypatch)

    result = pipeline.run_sr_timeseries_zones(
        _ts(max_zone_offset_days=14), tmp_path / "out"
    )

    ts = tmp_path / "out" / "valley_sr_timeseries"
    january, february = result.steps
    # 01-28 window: mean (1 + 0) / 2 beats 01-03 (2 + 3 + 9) / 3 only if it
    # covered as many zones -- it doesn't, so all three share 01-03.
    assert {p.date for p in january.step.picks.values()} == {"2024-01-03"}
    assert january.tiles["z01_00"] == ts / "lr" / "z01_00" / "valley_lr_2024-01-03.tif"
    assert february.step.empty_zones == ["z01_00"]
    assert result.skipped_bins == ["2024-03-01"]
    # Pass 2 fetches each zone's chosen dates at 10 m, no SR per zone.
    fetch = {(c[1], c[2]): c[3:] for c in calls if c[0] == "run_sentinel_sr"}
    assert fetch[(47.01, 11.01)] == (["2024-01-03", "2024-02-10"], True)
    assert fetch[(47.00, 11.01)] == (["2024-01-03"], True)
    # Mosaics on the full zone grid, SCL alongside.
    vrts = [c for c in calls if c[0] == "build_vrt"]
    assert [Path(c[2]).name for c in vrts] == [
        "valley_lr_2024-01-03.vrt",
        "valley_lr_2024-01-03_scl.vrt",
        "valley_lr_2024-02-10.vrt",
        "valley_lr_2024-02-10_scl.vrt",
    ]
    assert all(c[3] == GRID.bounds for c in vrts)
    # One co-registration over the whole series, before SR.
    names = [c[0] for c in calls]
    assert names.index("coreg") < names.index("superres")
    [coreg] = [c for c in calls if c[0] == "coreg"]
    assert coreg[1] == [
        ("2024-01-03", "valley_lr_2024-01-03.tif"),
        ("2024-02-10", "valley_lr_2024-02-10.tif"),
    ]
    [superres] = [c for c in calls if c[0] == "superres"]
    assert [src for src, _ in superres[1]] == [
        "valley_lr_2024-01-03_cr.tif",
        "valley_lr_2024-02-10_cr.tif",
    ]
    assert january.shifts == {"2024-01-03": (0.5, -0.25)}
    assert january.sr == ts / "valley_sr_2024-01-03.tif"
    assert january.sr.read_bytes() == b"sr:mosaic+cr"
    # The SR mosaics make the folder a regular timeseries.
    assert list(pipeline.timeseries_frames("aoi/valley.gpkg", ts)) == [
        "2024-01-03",
        "2024-02-10",
    ]
    steps = json.loads((ts / "steps.json").read_text())
    assert steps["max_zone_offset_days"] == 14
    assert steps["coreg"]["anchor"] == "auto"
    assert steps["steps"][0]["spread_days"] == 0
    assert steps["steps"][0]["coreg"] == {"2024-01-03": [0.5, -0.25]}
    assert steps["steps"][0]["lr_mosaic"] == "lr/valley_lr_2024-01-03_cr.tif"
    assert steps["steps"][1]["empty_zones"] == ["z01_00"]


def test_zone_timeseries_without_offset_merges_date_parts(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)

    result = pipeline.run_sr_timeseries_zones(_ts(), tmp_path / "out")

    january = result.steps[0]
    assert january.step.picks["z00_00"].date == "2024-01-28"
    assert january.step.picks["z00_01"].date == "2024-01-03"
    assert january.step.spread_days == 25
    # Each acquisition date is its own 10 m part with its own shift, merged
    # after co-registration.
    [coreg] = [c for c in calls if c[0] == "coreg"]
    assert coreg[1][:2] == [
        ("2024-01-03", "valley_lr_2024-01-03_part_2024-01-03.tif"),
        ("2024-01-28", "valley_lr_2024-01-03_part_2024-01-28.tif"),
    ]
    merges = [c for c in calls if c[0] == "merge"]
    assert merges[0][1] == [
        "valley_lr_2024-01-03_part_2024-01-03_cr.tif",
        "valley_lr_2024-01-03_part_2024-01-28_cr.tif",
    ]
    assert set(january.shifts) == {"2024-01-03", "2024-01-28"}
    assert january.sr is not None and january.sr.name == "valley_sr_2024-01-03.tif"


def test_zone_timeseries_single_date(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)

    result = pipeline.run_sr_timeseries_zones(_ts(single_date=True), tmp_path / "out")

    # January: 01-03 is the only day all three zones share; February has no
    # scene for z01_00, so no step.
    [january] = result.steps
    assert {p.date for p in january.step.picks.values()} == {"2024-01-03"}
    assert result.skipped_bins == ["2024-02-01", "2024-03-01"]
    # A single step can't be co-registered against anything.
    assert [c for c in calls if c[0] == "coreg"] == []
    assert january.shifts == {}
    assert january.lr_mosaic is not None
    assert january.lr_mosaic.name == "valley_lr_2024-01-03.tif"
    steps = json.loads(
        (tmp_path / "out" / "valley_sr_timeseries" / "steps.json").read_text()
    )
    assert steps["single_date"] is True


def test_zone_timeseries_coreg_disabled(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)

    pipeline.run_sr_timeseries_zones(
        _sr_cfg(
            timeseries={"enabled": True, "max_zone_offset_days": 14},
            coreg={"enabled": False},
        ),
        tmp_path / "out",
    )

    assert [c for c in calls if c[0] == "coreg"] == []
    [superres] = [c for c in calls if c[0] == "superres"]
    assert [src for src, _ in superres[1]] == [
        "valley_lr_2024-01-03.tif",
        "valley_lr_2024-02-10.tif",
    ]


def test_zone_timeseries_step_dropped_by_coreg_gets_no_sr(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch, drop={"2024-02-10"})

    result = pipeline.run_sr_timeseries_zones(
        _ts(max_zone_offset_days=14), tmp_path / "out"
    )

    january, february = result.steps
    assert february.sr is None and february.lr_mosaic is None
    assert february.shifts == {"2024-02-10": None}
    [superres] = [c for c in calls if c[0] == "superres"]
    assert len(superres[1]) == 1
    steps = json.loads(
        (tmp_path / "out" / "valley_sr_timeseries" / "steps.json").read_text()
    )
    assert steps["steps"][1]["coreg"] == {"2024-02-10": None}
    assert steps["steps"][1]["sr"] is None


def test_zone_timeseries_sr_is_cached_by_input(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)
    pipeline.run_sr_timeseries_zones(_ts(max_zone_offset_days=14), tmp_path / "out")
    [first] = [c for c in calls if c[0] == "superres"]
    calls.clear()

    pipeline.run_sr_timeseries_zones(_ts(max_zone_offset_days=14), tmp_path / "out")

    # Same mosaics -> same cache paths; sentinel_sr keeps existing outputs.
    [second] = [c for c in calls if c[0] == "superres"]
    assert second[1] == first[1]


def test_zone_timeseries_reuses_candidates_across_offsets(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)
    pipeline.run_sr_timeseries_zones(_ts(), tmp_path / "out")
    calls.clear()

    pipeline.run_sr_timeseries_zones(_ts(max_zone_offset_days=3), tmp_path / "out")
    pipeline.run_sr_timeseries_zones(_ts(single_date=True), tmp_path / "out")

    assert [c for c in calls if c[0] == "list_candidates"] == []


def test_zone_timeseries_reset_and_fill_gaps(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch)
    out = tmp_path / "out"
    ts = out / "valley_sr_timeseries"
    pipeline.run_sr_timeseries_zones(_ts(max_zone_offset_days=14), out)
    stale = ts / "valley_sr_2023-06-01.tif"
    stale.write_bytes(b"old")
    stale_lr = ts / "lr" / "valley_lr_2023-06-01.tif"
    stale_lr.write_bytes(b"old")

    # Without fill_gaps the folder is rebuilt: stale steps go.
    pipeline.run_sr_timeseries_zones(_ts(max_zone_offset_days=14), out)
    assert not stale.exists() and not stale_lr.exists()

    # With fill_gaps finished intervals are kept and not chosen again, but
    # their 10 m mosaics are co-registered with the series again.
    calls.clear()
    result = pipeline.run_sr_timeseries_zones(
        _ts(max_zone_offset_days=14, fill_gaps=True), out
    )
    assert result.steps == []
    assert all(c[3] is None for c in calls if c[0] == "run_sentinel_sr")
    [coreg] = [c for c in calls if c[0] == "coreg"]
    assert [d for d, _ in coreg[1]] == ["2024-01-03", "2024-02-10"]
    assert len(json.loads((ts / "steps.json").read_text())["steps"]) == 2


def test_zone_timeseries_offset_keeps_the_cache_key():
    assert pipeline._cache_key(_ts()) == pipeline._cache_key(
        _ts(max_zone_offset_days=14)
    )
    assert pipeline._cache_key(_ts()) == pipeline._cache_key(_ts(single_date=True))
    assert pipeline._cache_key(_ts()) == pipeline._cache_key(
        _sr_cfg(timeseries={"enabled": True}, coreg={"anchor": "first"})
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
