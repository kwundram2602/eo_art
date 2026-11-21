import sys
from pathlib import Path

import pytest

from eo_art.app.runner import (
    PipelineKind,
    PipelineSpec,
    build_command,
    start,
    stop,
    tail,
)

BIN = Path("/venv/bin")


def test_acquire_command():
    spec = PipelineSpec(
        PipelineKind.acquire, acquire_config="configs/acquire.yaml", no_cache=True
    )
    assert build_command(spec, BIN) == [
        "/venv/bin/eo-art-f3d",
        "acquire",
        "configs/acquire.yaml",
        "--no-cache",
    ]


def test_sr_command_passes_overrides():
    spec = PipelineSpec(
        PipelineKind.sr,
        acquire_config="configs/acquire.yaml",
        overrides=["aoi_path=aoi/b.gpkg"],
    )
    assert build_command(spec, BIN) == [
        "/venv/bin/eo-art-sr",
        "configs/acquire.yaml",
        "--set",
        "aoi_path=aoi/b.gpkg",
    ]


def test_render_command_keeps_layer_order_and_flags():
    spec = PipelineSpec(
        PipelineKind.render,
        render_layers=["configs/render.yaml", "configs/look.yaml"],
        overrides=["render.width=640"],
        sweep="render.pbr.exposure=1.0,1.8",
        fail_fast=True,
    )
    assert build_command(spec, BIN) == [
        "/venv/bin/eo-art-f3d",
        "run",
        "configs/render.yaml",
        "configs/look.yaml",
        "--set",
        "render.width=640",
        "--sweep",
        "render.pbr.exposure=1.0,1.8",
        "--fail-fast",
    ]


def test_acquire_render_command():
    spec = PipelineSpec(
        PipelineKind.acquire_render,
        acquire_config="configs/acquire.yaml",
        render_layers=["configs/render.yaml"],
        acquire_overlay="optical",
        no_cache=True,
    )
    assert build_command(spec, BIN) == [
        "/venv/bin/eo-art-f3d",
        "run",
        "configs/render.yaml",
        "--no-cache",
        "--acquire",
        "configs/acquire.yaml",
        "--acquire-overlay",
        "optical",
        "--acquire-no-cache",
    ]


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        (PipelineKind.acquire, {}),
        (PipelineKind.sr, {}),
        (PipelineKind.render, {}),
        (PipelineKind.acquire_render, {"render_layers": ["r.yaml"]}),
        (PipelineKind.acquire_render, {"acquire_config": "a.yaml"}),
    ],
)
def test_spec_rejects_missing_inputs(kind, kwargs):
    with pytest.raises(ValueError):
        PipelineSpec(kind, **kwargs)


def test_start_writes_command_and_output_to_log(tmp_path):
    log = tmp_path / "logs" / "run.log"
    handle = start([sys.executable, "-c", "print('hello')"], tmp_path, log)

    assert handle.process.wait(timeout=30) == 0
    assert not handle.running
    text = tail(log)
    assert "-c" in text.splitlines()[1]
    assert text.splitlines()[-1] == "hello"


def test_stop_terminates_running_process(tmp_path):
    handle = start(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        tmp_path / "run.log",
    )
    assert handle.running

    stop(handle)

    assert not handle.running
    assert handle.returncode != 0


def test_tail_of_missing_file_is_empty(tmp_path):
    assert tail(tmp_path / "nope.log") == ""
