import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from eo_art.forge3d_pipes.acquire import pipeline
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig, SrConfig
from eo_art.forge3d_pipes.acquire.sentinel_bridge import MANIFEST_NAME, Scene


def _cfg(**overrides):
    base = OmegaConf.merge(
        OmegaConf.structured(AcquireConfig),
        {
            "aoi_path": "aoi.gpkg",
            "ee_project": "ee-test",
            "sentinel_sr_dir": "../sentinel_sr",
            "sentinel": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "model_path": "/models/LDSRS2-SEN2SR",
            },
        },
        overrides,
    )
    return OmegaConf.to_object(base)


def _wire(monkeypatch, tmp_path, *, scenes=(("scene0.tif", None),)):
    calls = []

    def _read_aoi_center(aoi_path):
        calls.append(("read_aoi_center", aoi_path))
        return 39.49, -0.43

    def _run_sentinel_sr(
        sentinel_cfg, sentinel_sr_dir, lat, lon, out_dir, overwrite=False
    ):
        calls.append(("run_sentinel_sr", sentinel_sr_dir, lat, lon))
        out_dir = tmp_path / "sentinel_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, cloud_cover in scenes:
            path = out_dir / name
            path.write_bytes(b"scene")
            written.append(Scene(path=path, cloud_cover=cloud_cover))
        return written

    def _fetch_terrain_dem(reference_tif, out_dir, *, source, ee_project, scale):
        calls.append(("fetch_terrain_dem", reference_tif, source, ee_project, scale))
        dem_path = tmp_path / "dem.tif"
        dem_path.write_bytes(b"dem")
        return dem_path

    def _fetch_local_dem(reference_tif, out_dir, *, path):
        calls.append(("fetch_local_dem", reference_tif, out_dir, path))
        dem_path = tmp_path / "local_dem.tif"
        dem_path.write_bytes(b"localdem")
        return dem_path

    def _super_resolve_dtm(dem, optical, out, **kwargs):
        calls.append(("super_resolve_dtm", dem, optical, out, kwargs))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"dtm")
        return out

    def _simulate_thermal_erosion(dem, out, **kwargs):
        calls.append(("simulate_thermal_erosion", dem, out, kwargs))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"eroded")
        return out

    def _align_raster_grid(reference, target, out, *, resampling):
        calls.append(("align_raster_grid", reference, target, out, resampling))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"aligned")
        return out

    monkeypatch.setattr(pipeline, "read_aoi_center", _read_aoi_center)
    monkeypatch.setattr(pipeline, "run_sentinel_sr", _run_sentinel_sr)
    monkeypatch.setattr(pipeline, "fetch_terrain_dem", _fetch_terrain_dem)
    monkeypatch.setattr(pipeline, "fetch_local_dem", _fetch_local_dem)
    monkeypatch.setattr(pipeline, "super_resolve_dtm", _super_resolve_dtm)
    monkeypatch.setattr(pipeline, "simulate_thermal_erosion", _simulate_thermal_erosion)
    monkeypatch.setattr(pipeline, "align_raster_grid", _align_raster_grid)
    return calls


def test_run_acquire_chains_steps_in_order(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg()

    result = pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    assert [c[0] for c in calls] == [
        "read_aoi_center",
        "run_sentinel_sr",
        "fetch_terrain_dem",
        "super_resolve_dtm",
        "align_raster_grid",
    ]
    assert result.dtm_path.exists()
    assert result.optical_path.exists()
    # the returned paths are stable, directly under out_dir -- not buried in
    # the internal _acquire/<hash> cache directory.
    assert result.dtm_path == tmp_path / "out" / "dtm.tif"
    assert result.optical_path == tmp_path / "out" / "optical_aligned.tif"

    # the DTM must be super-resolved from the raw DEM + reference scene...
    _, dem, optical, dtm_out, _ = calls[3]
    assert dem.name == "dem.tif"
    assert optical.name == "scene0.tif"
    assert dtm_out.name == result.dtm_path.name
    assert "_acquire" in str(dtm_out)  # written to the cache, then copied out
    # ...and the optical scene aligned onto *that* DTM's grid, not the raw DEM.
    _, reference, target, optical_out, resampling = calls[4]
    assert reference == dtm_out
    assert target.name == "scene0.tif"
    assert optical_out.name == result.optical_path.name
    assert resampling == "bilinear"


def test_reference_index_selects_scene(monkeypatch, tmp_path):
    calls = _wire(
        monkeypatch,
        tmp_path,
        scenes=(("scene0.tif", 5.0), ("scene1.tif", 50.0)),
    )
    cfg = _cfg(sentinel={"reference_index": 1})

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    _, _, optical, _, _ = calls[3]
    assert optical.name == "scene1.tif"


def test_out_of_range_reference_index_raises(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, scenes=(("scene0.tif", None),))
    cfg = _cfg(sentinel={"reference_index": 5})

    try:
        pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "reference_index" in str(exc)


def test_default_selects_scene_with_lowest_cloud_cover(monkeypatch, tmp_path):
    calls = _wire(
        monkeypatch,
        tmp_path,
        scenes=(("cloudy.tif", 80.0), ("clear.tif", 3.0), ("medium.tif", 40.0)),
    )
    cfg = _cfg()

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    _, _, optical, _, _ = calls[3]
    assert optical.name == "clear.tif"


def test_default_treats_missing_cloud_cover_as_worst(monkeypatch, tmp_path):
    calls = _wire(
        monkeypatch,
        tmp_path,
        scenes=(("unknown.tif", None), ("clear.tif", 10.0)),
    )
    cfg = _cfg()

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    _, _, optical, _, _ = calls[3]
    assert optical.name == "clear.tif"


def test_no_scenes_written_raises(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)

    def _empty(sentinel_cfg, sentinel_sr_dir, lat, lon, out_dir, overwrite=False):
        return []

    monkeypatch.setattr(pipeline, "run_sentinel_sr", _empty)
    cfg = _cfg()

    try:
        pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "no scenes" in str(exc)


def test_cache_hit_skips_all_steps(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg()
    out_dir = tmp_path / "out"

    first = pipeline.run_acquire(cfg, out_dir, use_cache=True)
    assert len(calls) == 5

    second = pipeline.run_acquire(cfg, out_dir, use_cache=True)
    assert len(calls) == 5  # no new calls
    assert second == first
    assert second.dtm_path.exists()
    assert second.optical_path.exists()


def test_no_cache_reruns_even_with_prior_result(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg()
    out_dir = tmp_path / "out"

    pipeline.run_acquire(cfg, out_dir, use_cache=True)
    assert len(calls) == 5

    pipeline.run_acquire(cfg, out_dir, use_cache=False)
    assert len(calls) == 10


def test_different_config_gets_a_different_cache_key(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    pipeline.run_acquire(_cfg(), out_dir, use_cache=True)
    pipeline.run_acquire(_cfg(dem={"scale": 10}), out_dir, use_cache=True)

    assert len(calls) == 10


def test_out_dir_is_not_part_of_the_cache_key(monkeypatch, tmp_path):
    """A config that only differs in out_dir must produce the same cache key,
    or moving --out (or just adding the out_dir field to the schema, as
    happened once) silently forces an expensive GPU+network recompute."""
    cfg_a = _cfg(out_dir="out_a")
    cfg_b = _cfg(out_dir="out_b")
    assert pipeline._cache_key(cfg_a) == pipeline._cache_key(cfg_b)


def test_dtm_mode_is_part_of_the_cache_key(monkeypatch, tmp_path):
    cfg_a = _cfg(dtm={"mode": "guided_upsample"})
    cfg_b = _cfg(dtm={"mode": "native"})
    assert pipeline._cache_key(cfg_a) != pipeline._cache_key(cfg_b)


def test_local_dem_source_is_part_of_the_cache_key(monkeypatch, tmp_path):
    cfg_a = _cfg(dem={"source": "fabdem"})
    cfg_b = _cfg(dem={"source": "local", "path": "dgm.tif"})
    assert pipeline._cache_key(cfg_a) != pipeline._cache_key(cfg_b)


def test_local_dem_source_calls_fetch_local_dem_not_fetch_terrain_dem(
    monkeypatch, tmp_path
):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg(dem={"source": "local", "path": "some_local_dgm.tif"})

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    names = [c[0] for c in calls]
    assert "fetch_local_dem" in names
    assert "fetch_terrain_dem" not in names
    _, reference_tif, out_dir, path = next(
        c for c in calls if c[0] == "fetch_local_dem"
    )
    assert reference_tif.name == "scene0.tif"
    assert path == "some_local_dgm.tif"


def test_native_mode_skips_super_resolve_and_aligns_onto_dem_grid(
    monkeypatch, tmp_path
):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg(dtm={"mode": "native"})

    result = pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    assert [c[0] for c in calls] == [
        "read_aoi_center",
        "run_sentinel_sr",
        "fetch_terrain_dem",
        "align_raster_grid",
    ]
    # the DTM is the raw DEM's own content, untouched by any upsampling.
    assert result.dtm_path.read_bytes() == b"dem"

    _, reference, target, optical_out, resampling = calls[-1]
    assert reference.name == result.dtm_path.name
    assert target.name == "scene0.tif"
    assert optical_out.name == result.optical_path.name
    assert resampling == "bilinear"


def test_guided_upsample_mode_is_the_default_and_still_calls_super_resolve_dtm(
    monkeypatch, tmp_path
):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg()

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    assert [c[0] for c in calls] == [
        "read_aoi_center",
        "run_sentinel_sr",
        "fetch_terrain_dem",
        "super_resolve_dtm",
        "align_raster_grid",
    ]


def test_native_mode_without_erosion_does_not_call_simulate_thermal_erosion(
    monkeypatch, tmp_path
):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg(dtm={"mode": "native", "apply_erosion": False})

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    assert "simulate_thermal_erosion" not in [c[0] for c in calls]


def test_native_mode_with_erosion_calls_simulate_thermal_erosion(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg(
        dtm={
            "mode": "native",
            "apply_erosion": True,
            "erosion_kwargs": {"iterations": 5},
        }
    )

    result = pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    names = [c[0] for c in calls]
    assert "simulate_thermal_erosion" in names
    assert "super_resolve_dtm" not in names
    _, dem_in, dem_out, kwargs = next(
        c for c in calls if c[0] == "simulate_thermal_erosion"
    )
    assert dem_in.name == "dem.tif"
    assert kwargs == {"iterations": 5}
    assert result.dtm_path.read_bytes() == b"eroded"


def test_guided_upsample_mode_with_erosion_still_chains_through_super_resolve_dtm(
    monkeypatch, tmp_path
):
    """Regression guard: erosion in guided_upsample mode is forwarded into
    super_resolve_dtm exactly as before, not routed through the native-mode
    simulate_thermal_erosion call."""
    calls = _wire(monkeypatch, tmp_path)
    cfg = _cfg(dtm={"apply_erosion": True, "erosion_kwargs": {"iterations": 3}})

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    names = [c[0] for c in calls]
    assert "super_resolve_dtm" in names
    assert "simulate_thermal_erosion" not in names
    _, _, _, _, kwargs = next(c for c in calls if c[0] == "super_resolve_dtm")
    assert kwargs["apply_erosion"] is True
    assert kwargs["erosion_kwargs"] == {"iterations": 3}


def _sr_cfg(**overrides):
    base = OmegaConf.merge(
        OmegaConf.structured(SrConfig),
        {
            "aoi_path": "aoi.gpkg",
            "sentinel_sr_dir": "../sentinel_sr",
            "sentinel": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "model_path": "/models/LDSRS2-SEN2SR",
            },
        },
        overrides,
    )
    return OmegaConf.to_object(base)


def test_run_sr_copies_lowest_cloud_scene_without_dem(monkeypatch, tmp_path):
    calls = _wire(
        monkeypatch,
        tmp_path,
        scenes=(("cloudy.tif", 40.0), ("clear.tif", 2.0)),
    )
    result = pipeline.run_sr(_sr_cfg(), tmp_path / "out", use_cache=False)

    assert [c[0] for c in calls] == ["read_aoi_center", "run_sentinel_sr"]
    assert result == tmp_path / "out" / "aoi_sr.tif"
    assert result.read_bytes() == b"scene"


def test_run_sr_names_output_after_aoi(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)

    valley = pipeline.run_sr(
        _sr_cfg(aoi_path="aoi/valley.gpkg"), tmp_path / "out", use_cache=False
    )
    ridge = pipeline.run_sr(
        _sr_cfg(aoi_path="aoi/ridge.gpkg"), tmp_path / "out", use_cache=False
    )

    assert valley == tmp_path / "out" / "valley_sr.tif"
    assert ridge == tmp_path / "out" / "ridge_sr.tif"
    # Two AOIs sharing one out_dir no longer overwrite each other.
    assert valley.exists() and ridge.exists()


def test_run_sr_cache_hit_skips_sentinel(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _sr_cfg()
    pipeline.run_sr(cfg, tmp_path / "out")
    calls.clear()

    result = pipeline.run_sr(cfg, tmp_path / "out")

    assert calls == []
    assert result.exists()


def test_run_sr_no_scenes_written_raises(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, scenes=())

    with pytest.raises(RuntimeError, match="no scenes"):
        pipeline.run_sr(_sr_cfg(), tmp_path / "out", use_cache=False)


def test_run_sr_reuses_cache_written_before_per_aoi_naming(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path)
    cfg = _sr_cfg(aoi_path="aoi/valley.gpkg")
    cache_dir = tmp_path / "out" / pipeline.CACHE_DIRNAME / pipeline._cache_key(cfg)
    cache_dir.mkdir(parents=True)
    (cache_dir / "optical_sr.tif").write_bytes(b"old")

    result = pipeline.run_sr(cfg, tmp_path / "out")

    assert calls == []
    assert result == tmp_path / "out" / "valley_sr.tif"
    assert result.read_bytes() == b"old"


def test_acquire_ignores_timeseries(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    seen = []
    inner = pipeline.run_sentinel_sr

    def _capture(sentinel_cfg, *args, **kwargs):
        seen.append(sentinel_cfg)
        return inner(sentinel_cfg, *args, **kwargs)

    monkeypatch.setattr(pipeline, "run_sentinel_sr", _capture)
    cfg = _cfg(sentinel={"timeseries": {"enabled": True, "interval": "week"}})

    pipeline.run_acquire(cfg, tmp_path / "out", use_cache=False)

    assert seen[0].timeseries.enabled is False
    assert pipeline._cache_key(cfg) == pipeline._cache_key(_cfg())


def test_disabled_timeseries_keeps_the_sr_cache_key():
    plain = _sr_cfg()
    disabled = _sr_cfg(sentinel={"timeseries": {"interval": "week", "every": 3}})
    enabled = _sr_cfg(sentinel={"timeseries": {"enabled": True}})

    assert pipeline._cache_key(plain) == pipeline._cache_key(disabled)
    assert pipeline._cache_key(plain) != pipeline._cache_key(enabled)


def test_no_cache_asks_sentinel_sr_to_overwrite(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    seen = []
    inner = pipeline.run_sentinel_sr

    def _capture(*args, overwrite=False):
        seen.append(overwrite)
        return inner(*args, overwrite=overwrite)

    monkeypatch.setattr(pipeline, "run_sentinel_sr", _capture)

    pipeline.run_sr(_sr_cfg(), tmp_path / "out", use_cache=True)
    pipeline.run_sr(_sr_cfg(), tmp_path / "out", use_cache=False)

    assert seen == [False, True]


def _ts_cfg(**timeseries):
    return _sr_cfg(
        aoi_path="aoi/valley.gpkg",
        sentinel={"timeseries": {"enabled": True, **timeseries}},
    )


def _wire_timeseries(monkeypatch, scenes, skipped_bins=()):
    """sentinel_sr stand-in that writes ``scenes`` ((date, cloud), ...) plus
    its manifest into the directory it's given."""
    calls = []

    def _run_sentinel_sr(sentinel_cfg, sentinel_sr_dir, lat, lon, out_dir, **kw):
        calls.append((sentinel_cfg, kw))
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for date, cloud in scenes:
            path = out_dir / f"superres_{date}_id.tif"
            path.write_bytes(date.encode())
            entries.append({"path": str(path), "cloud_cover": cloud, "datetime": date})
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps({"scenes": entries, "skipped_bins": list(skipped_bins)})
        )
        return [
            Scene(Path(e["path"]), e["cloud_cover"], e["datetime"]) for e in entries
        ]

    monkeypatch.setattr(pipeline, "read_aoi_center", lambda _: (1.0, 2.0))
    monkeypatch.setattr(pipeline, "run_sentinel_sr", _run_sentinel_sr)
    return calls


def test_run_sr_timeseries_writes_one_dated_file_per_scene(monkeypatch, tmp_path):
    calls = _wire_timeseries(
        monkeypatch,
        [("2023-03-02", 4.0), ("2023-01-05", 1.0)],
        skipped_bins=["2023-02-01"],
    )

    result = pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    assert result.out_dir == tmp_path / "out" / "valley_sr_timeseries"
    assert [p.name for p in result.paths] == [
        "valley_sr_2023-01-05.tif",
        "valley_sr_2023-03-02.tif",
    ]
    assert result.paths[0].read_bytes() == b"2023-01-05"
    assert result.skipped_bins == ["2023-02-01"]
    assert calls[0][0].timeseries.enabled is True
    assert calls[0][1] == {"overwrite": False, "skip_dates": ()}


def test_run_sr_timeseries_cache_hit_skips_sentinel(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch, [("2023-01-05", 1.0)])
    pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    result = pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    assert len(calls) == 1
    assert [p.name for p in result.paths] == ["valley_sr_2023-01-05.tif"]


def test_run_sr_timeseries_resumes_without_manifest(monkeypatch, tmp_path):
    """A run that died halfway left scenes but no manifest: rerun sentinel_sr
    (without overwrite, so it keeps what it already wrote)."""
    calls = _wire_timeseries(monkeypatch, [("2023-01-05", 1.0)])
    sentinel_dir = (
        tmp_path / "out" / pipeline.CACHE_DIRNAME / pipeline._cache_key(_ts_cfg())
    ) / "sentinel"
    sentinel_dir.mkdir(parents=True)
    (sentinel_dir / "superres_2023-01-05_id.tif").write_bytes(b"partial")

    pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    assert calls[0][1] == {"overwrite": False, "skip_dates": ()}


def test_run_sr_timeseries_no_cache_overwrites(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch, [("2023-01-05", 1.0)])
    pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out", use_cache=False)

    assert [kw for _, kw in calls] == [
        {"overwrite": False, "skip_dates": ()},
        {"overwrite": True, "skip_dates": ()},
    ]


def test_run_sr_timeseries_drops_stale_dates(monkeypatch, tmp_path):
    _wire_timeseries(monkeypatch, [("2023-01-05", 1.0)])
    target = tmp_path / "out" / "valley_sr_timeseries"
    target.mkdir(parents=True)
    (target / "valley_sr_2022-12-01.tif").write_bytes(b"old")
    (target / "notes.txt").write_text("keep")

    pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    assert sorted(p.name for p in target.iterdir()) == [
        "notes.txt",
        "valley_sr_2023-01-05.tif",
    ]


def test_run_sr_timeseries_no_scenes_raises(monkeypatch, tmp_path):
    _wire_timeseries(monkeypatch, [], skipped_bins=["2023-01-01"])

    with pytest.raises(RuntimeError, match="no scenes"):
        pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")


def test_run_sr_and_run_sr_timeseries_reject_the_other_mode(tmp_path):
    with pytest.raises(ValueError, match="run_sr_timeseries"):
        pipeline.run_sr(_ts_cfg(), tmp_path)
    with pytest.raises(ValueError, match="use run_sr"):
        pipeline.run_sr_timeseries(_sr_cfg(), tmp_path)


def test_max_items_only_counts_for_the_cache_key_without_timeseries():
    assert pipeline._cache_key(
        _sr_cfg(sentinel={"max_items": 3})
    ) != pipeline._cache_key(_sr_cfg(sentinel={"max_items": 5}))
    assert pipeline._cache_key(_ts_cfg()) == pipeline._cache_key(
        _sr_cfg(
            aoi_path="aoi/valley.gpkg",
            sentinel={"max_items": 5, "timeseries": {"enabled": True}},
        )
    )


def test_reference_prefers_aoi_cloud_over_tile_cloud(tmp_path):
    hazy_tile_clear_aoi = Scene(tmp_path / "a.tif", cloud_cover=40.0, aoi_cloud=0.0)
    clear_tile_cloudy_aoi = Scene(tmp_path / "b.tif", cloud_cover=1.0, aoi_cloud=8.0)

    reference = pipeline._select_reference(
        [clear_tile_cloudy_aoi, hazy_tile_clear_aoi], None
    )

    assert reference == tmp_path / "a.tif"


def test_fill_gaps_keeps_existing_timesteps_and_skips_their_intervals(
    monkeypatch, tmp_path
):
    calls = _wire_timeseries(monkeypatch, [("2023-03-02", 1.0)])
    target = tmp_path / "out" / "valley_sr_timeseries"
    target.mkdir(parents=True)
    (target / "valley_sr_2023-01-05.tif").write_bytes(b"jan")
    (target / "valley_sr_2023-02-14.tif").write_bytes(b"feb")
    (target / "valley_sr_2023-02-14_indices.tif").write_bytes(b"not a frame")

    result = pipeline.run_sr_timeseries(_ts_cfg(fill_gaps=True), tmp_path / "out")

    assert calls[0][1] == {
        "overwrite": False,
        "skip_dates": ["2023-01-05", "2023-02-14"],
    }
    assert [p.name for p in result.added] == ["valley_sr_2023-03-02.tif"]
    assert [p.name for p in result.paths] == [
        "valley_sr_2023-01-05.tif",
        "valley_sr_2023-02-14.tif",
        "valley_sr_2023-03-02.tif",
    ]
    assert (target / "valley_sr_2023-01-05.tif").read_bytes() == b"jan"


def test_fill_gaps_searches_again_despite_a_cached_manifest(monkeypatch, tmp_path):
    calls = _wire_timeseries(monkeypatch, [])
    target = tmp_path / "out" / "valley_sr_timeseries"
    target.mkdir(parents=True)
    (target / "valley_sr_2023-01-05.tif").write_bytes(b"jan")

    first = pipeline.run_sr_timeseries(_ts_cfg(fill_gaps=True), tmp_path / "out")
    second = pipeline.run_sr_timeseries(_ts_cfg(fill_gaps=True), tmp_path / "out")

    # Nothing new is fine when there already are timesteps.
    assert len(calls) == 2
    assert first.added == second.added == []
    assert [p.name for p in second.paths] == ["valley_sr_2023-01-05.tif"]


def test_without_fill_gaps_only_frames_are_reset(monkeypatch, tmp_path):
    _wire_timeseries(monkeypatch, [("2023-03-02", 1.0)])
    target = tmp_path / "out" / "valley_sr_timeseries"
    target.mkdir(parents=True)
    (target / "valley_sr_2023-01-05.tif").write_bytes(b"old")
    (target / "valley_sr_2023-01-05_indices.tif").write_bytes(b"keep")

    result = pipeline.run_sr_timeseries(_ts_cfg(), tmp_path / "out")

    assert sorted(p.name for p in target.iterdir()) == [
        "valley_sr_2023-01-05_indices.tif",
        "valley_sr_2023-03-02.tif",
    ]
    assert result.added == result.paths
