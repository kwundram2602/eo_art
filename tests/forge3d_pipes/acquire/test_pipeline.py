from omegaconf import OmegaConf

from eo_art.forge3d_pipes.acquire import pipeline
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig
from eo_art.forge3d_pipes.acquire.sentinel_bridge import Scene


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

    def _run_sentinel_sr(sentinel_cfg, sentinel_sr_dir, lat, lon, out_dir):
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

    def _super_resolve_dtm(dem, optical, out, **kwargs):
        calls.append(("super_resolve_dtm", dem, optical, out, kwargs))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"dtm")
        return out

    def _align_raster_grid(reference, target, out, *, resampling):
        calls.append(("align_raster_grid", reference, target, out, resampling))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"aligned")
        return out

    monkeypatch.setattr(pipeline, "read_aoi_center", _read_aoi_center)
    monkeypatch.setattr(pipeline, "run_sentinel_sr", _run_sentinel_sr)
    monkeypatch.setattr(pipeline, "fetch_terrain_dem", _fetch_terrain_dem)
    monkeypatch.setattr(pipeline, "super_resolve_dtm", _super_resolve_dtm)
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

    def _empty(sentinel_cfg, sentinel_sr_dir, lat, lon, out_dir):
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
