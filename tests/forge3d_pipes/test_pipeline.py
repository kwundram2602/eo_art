import pytest
import yaml

from eo_art.forge3d_pipes import pipeline
from eo_art.forge3d_pipes.render.runner import RenderResult


@pytest.fixture
def config_file(tmp_path, synthetic_dem):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "run": {"name": "test", "out_dir": str(tmp_path / "out")},
                "prepare": [{"op": "reproject", "crs": "EPSG:32610"}],
            }
        )
    )
    return path


@pytest.fixture
def fake_render(monkeypatch):
    calls = []

    def _render(cfg, terrain_path, out_dir, overlays=()):
        calls.append((cfg, terrain_path, out_dir, overlays))
        snapshot = out_dir / cfg.render.snapshot_name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"png")
        return RenderResult(snapshot=snapshot)

    monkeypatch.setattr(pipeline, "render", _render)
    return calls


def test_single_variant_runs_end_to_end(config_file, fake_render, tmp_path):
    results = pipeline.run([config_file])
    assert len(results) == 1
    assert results[0].ok
    assert results[0].name == "default"
    assert results[0].snapshot.exists()


def test_resolved_config_is_written_per_variant(config_file, fake_render, tmp_path):
    results = pipeline.run([config_file])
    resolved = results[0].out_dir / "resolved.yaml"
    assert resolved.exists()
    assert yaml.safe_load(resolved.read_text())["render"]["width"] == 1200


def test_prep_runs_before_render_and_render_gets_prepared_path(
    config_file, fake_render, synthetic_dem
):
    pipeline.run([config_file])
    _, terrain_path, _, _ = fake_render[0]
    assert terrain_path != synthetic_dem
    assert "_prep" in str(terrain_path)


def test_sweep_produces_one_directory_per_variant(config_file, fake_render, tmp_path):
    results = pipeline.run(
        [config_file],
        overrides=[
            "sweep.mode=product",
            "sweep.params={render.pbr.exposure: [1.0, 2.0]}",
        ],
    )
    assert len(results) == 2
    assert {r.name for r in results} == {"exposure=1.0", "exposure=2.0"}
    assert all(r.out_dir.exists() for r in results)


def test_prep_is_cached_across_sweep_variants(
    config_file, fake_render, monkeypatch, tmp_path
):
    calls = []
    original = pipeline.run_prep_chain

    def _counting(src, entries, cache_dir, use_cache=True):
        calls.append(src)
        return original(src, entries, cache_dir, use_cache)

    monkeypatch.setattr(pipeline, "run_prep_chain", _counting)
    pipeline.run(
        [config_file],
        overrides=["sweep.params={render.pbr.exposure: [1.0, 2.0, 3.0]}"],
    )
    # Called once per variant, but the underlying reprojection is cached,
    # so only one output file exists.
    assert len(calls) == 3
    cache = tmp_path / "out" / "test" / "_prep"
    assert len(list(cache.glob("*.tif"))) == 1


def test_failing_variant_is_recorded_and_others_continue(
    config_file, monkeypatch, tmp_path
):
    def _render(cfg, terrain_path, out_dir, overlays=()):
        if cfg.render.pbr.exposure == 2.0:
            raise RuntimeError("gpu exploded")
        snapshot = out_dir / "snapshot.png"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"png")
        return RenderResult(snapshot=snapshot)

    monkeypatch.setattr(pipeline, "render", _render)
    results = pipeline.run(
        [config_file],
        overrides=["sweep.params={render.pbr.exposure: [1.0, 2.0, 3.0]}"],
    )
    assert [r.ok for r in results] == [True, False, True]
    assert "gpu exploded" in results[1].error


def test_fail_fast_aborts_on_first_error(config_file, monkeypatch):
    def _render(cfg, terrain_path, out_dir, overlays=()):
        raise RuntimeError("gpu exploded")

    monkeypatch.setattr(pipeline, "render", _render)
    with pytest.raises(RuntimeError, match="gpu exploded"):
        pipeline.run(
            [config_file],
            overrides=["sweep.params={render.pbr.exposure: [1.0, 2.0]}"],
            fail_fast=True,
        )


def test_missing_input_file_fails_before_rendering(tmp_path, monkeypatch):
    def _render(cfg, terrain_path, out_dir, overlays=()):
        raise AssertionError("render must not be reached")

    monkeypatch.setattr(pipeline, "render", _render)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"input": {"path": str(tmp_path / "nope.tif")}}))
    with pytest.raises(FileNotFoundError, match="nope.tif"):
        pipeline.run([path])


def test_bad_config_fails_before_any_variant_runs(tmp_path, synthetic_dem, monkeypatch):
    def _render(cfg, terrain_path, out_dir, overlays=()):
        raise AssertionError("render must not be reached")

    monkeypatch.setattr(pipeline, "render", _render)
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "sweep": {"params": {"render.camera.fov": [60.0, 300.0]}},
            }
        )
    )
    with pytest.raises(ValueError, match="camera.fov"):
        pipeline.run([path])


def test_run_cache_false_in_config_disables_caching(
    config_file, fake_render, monkeypatch, tmp_path
):
    """`run.cache: false` in YAML must actually reach run_prep_chain."""
    seen = []
    original = pipeline.run_prep_chain

    def _spy(src, entries, cache_dir, use_cache=True):
        seen.append(use_cache)
        return original(src, entries, cache_dir, use_cache)

    monkeypatch.setattr(pipeline, "run_prep_chain", _spy)

    pipeline.run([config_file])
    assert seen == [True]  # schema default

    seen.clear()
    pipeline.run([config_file], overrides=["run.cache=false"])
    assert seen == [False]


def test_explicit_use_cache_argument_overrides_config(
    config_file, fake_render, monkeypatch
):
    seen = []
    original = pipeline.run_prep_chain

    def _spy(src, entries, cache_dir, use_cache=True):
        seen.append(use_cache)
        return original(src, entries, cache_dir, use_cache)

    monkeypatch.setattr(pipeline, "run_prep_chain", _spy)
    pipeline.run([config_file], overrides=["run.cache=false"], use_cache=True)
    assert seen == [True]


def test_cli_sweep_composes_with_config_sweep(config_file, fake_render, tmp_path):
    """--sweep adds a parameter to a config-defined sweep, it does not replace it."""
    results = pipeline.run(
        [config_file],
        overrides=[
            "sweep.params={render.camera.phi: [10, 20]}",
            "sweep.params={render.pbr.exposure: [1.0, 2.0]}",
        ],
    )
    assert len(results) == 4
    assert all(r.ok for r in results)
    assert {r.name for r in results} == {
        "phi=10__exposure=1.0",
        "phi=10__exposure=2.0",
        "phi=20__exposure=1.0",
        "phi=20__exposure=2.0",
    }


def test_video_export_is_wired_end_to_end(config_file, monkeypatch, tmp_path):
    """The frames_dir -> encode_video handoff and video filename construction."""
    import imageio.v3 as iio
    import numpy as np

    def _render(cfg, terrain_path, out_dir, overlays=()):
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            iio.imwrite(
                frames_dir / f"frame_{index:04d}.png",
                np.full((8, 8, 3), index * 80, dtype=np.uint8),
            )
        return RenderResult(frames_dir=frames_dir)

    monkeypatch.setattr(pipeline, "render", _render)

    results = pipeline.run(
        [config_file],
        overrides=[
            "animation.kind=orbit",
            "animation.fps=3",
            "animation.orbit.duration=1.0",
            "export.video.enabled=true",
            "export.video.format=gif",
            "export.video.fps=3",
        ],
    )
    assert len(results) == 1 and results[0].ok
    assert results[0].video is not None
    assert results[0].video.name == "video.gif"
    assert results[0].video.exists() and results[0].video.stat().st_size > 0


def test_overlay_is_prepped_and_resolved_and_passed_to_render(
    tmp_path, synthetic_dem, synthetic_overlay, fake_render
):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "run": {"name": "test", "out_dir": str(tmp_path / "out")},
                "overlays": [
                    {"name": "ndvi", "path": str(synthetic_overlay), "opacity": 0.5}
                ],
            }
        )
    )
    pipeline.run([path])
    _, _, _, overlays = fake_render[0]
    assert len(overlays) == 1
    resolved = overlays[0]
    assert resolved.name == "ndvi"
    assert resolved.path == synthetic_overlay  # no prepare chain -> unchanged path
    assert resolved.opacity == 0.5
    assert resolved.extent == pytest.approx((0.5, 0.0, 1.0, 0.5))


def test_overlay_manual_extent_bypasses_auto_compute(
    tmp_path, synthetic_dem, synthetic_overlay, fake_render, monkeypatch
):
    def _boom(terrain_path, overlay_path):
        raise AssertionError("compute_normalized_extent must not be called")

    monkeypatch.setattr(pipeline, "compute_normalized_extent", _boom)

    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "run": {"name": "test", "out_dir": str(tmp_path / "out")},
                "overlays": [
                    {
                        "name": "ndvi",
                        "path": str(synthetic_overlay),
                        "extent": [0.1, 0.1, 0.9, 0.9],
                    }
                ],
            }
        )
    )
    pipeline.run([path])
    _, _, _, overlays = fake_render[0]
    assert overlays[0].extent == (0.1, 0.1, 0.9, 0.9)


def test_multiple_overlays_all_resolved_in_order(
    tmp_path, synthetic_dem, synthetic_overlay, fake_render
):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "run": {"name": "test", "out_dir": str(tmp_path / "out")},
                "overlays": [
                    {"name": "a", "path": str(synthetic_overlay)},
                    {"name": "b", "path": str(synthetic_dem)},
                ],
            }
        )
    )
    pipeline.run([path])
    _, _, _, overlays = fake_render[0]
    assert [o.name for o in overlays] == ["a", "b"]


def test_bad_overlay_prep_op_fails_before_any_variant_runs(
    tmp_path, synthetic_dem, synthetic_overlay, monkeypatch
):
    def _render(cfg, terrain_path, out_dir, overlays=()):
        raise AssertionError("render must not be reached")

    monkeypatch.setattr(pipeline, "render", _render)
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "overlays": [
                    {
                        "name": "ndvi",
                        "path": str(synthetic_overlay),
                        "prepare": [{"op": "nope"}],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match=r"overlays\[0\]\.prepare"):
        pipeline.run([path])
