import os
import re
import subprocess
import sys

import pytest

from eo_art.forge3d_pipes import gpu
from eo_art.forge3d_pipes.config.schema import GpuPreference

_ENV_VARS = ("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "__NV_PRIME_RENDER_OFFLOAD")


@pytest.fixture(autouse=True)
def clean_vulkan_env(monkeypatch):
    """Start every test from a clean slate.

    ``select_gpu`` mutates ``os.environ`` directly (real Vulkan loaders only
    read real env vars), not through ``monkeypatch``. Calling
    ``monkeypatch.delenv`` here first still gets the restoration right:
    ``monkeypatch`` records each key's pre-test value once, at this first
    touch, and unconditionally restores that value at teardown regardless
    of what ``select_gpu`` does to the key afterwards.
    """
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_enumerate(monkeypatch):
    """Stub ``forge3d.enumerate_adapters`` so tests never touch a real GPU."""
    import forge3d

    def _fake():
        return [{"status": "ok", "name": "Stub Adapter", "device_type": "Cpu"}]

    monkeypatch.setattr(forge3d, "enumerate_adapters", _fake)


def _write_nvidia_icd(directory, filename="nvidia_icd.json"):
    icd = directory / filename
    icd.write_text("{}")
    return icd


def test_off_touches_no_environment_and_returns_none(tmp_path):
    result = gpu.select_gpu(GpuPreference.off, search_dirs=[tmp_path])
    assert result is None
    for name in _ENV_VARS:
        assert name not in os.environ


def test_off_never_imports_or_calls_forge3d(tmp_path, monkeypatch):
    import forge3d

    def _boom():
        raise AssertionError("enumerate_adapters must not be called for 'off'")

    monkeypatch.setattr(forge3d, "enumerate_adapters", _boom)
    assert gpu.select_gpu(GpuPreference.off, search_dirs=[tmp_path]) is None


def test_discrete_sets_all_three_vars_to_found_icd(tmp_path, fake_enumerate):
    icd = _write_nvidia_icd(tmp_path)
    result = gpu.select_gpu(GpuPreference.discrete, search_dirs=[tmp_path])
    assert os.environ["VK_DRIVER_FILES"] == str(icd)
    assert os.environ["VK_ICD_FILENAMES"] == str(icd)
    assert os.environ["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    assert result == "Stub Adapter"


def test_discrete_search_is_case_insensitive(tmp_path, fake_enumerate):
    icd = _write_nvidia_icd(tmp_path, filename="NVIDIA_Icd.json")
    gpu.select_gpu(GpuPreference.discrete, search_dirs=[tmp_path])
    assert os.environ["VK_ICD_FILENAMES"] == str(icd)


def test_discrete_ignores_non_nvidia_files(tmp_path, fake_enumerate):
    (tmp_path / "intel_icd.json").write_text("{}")
    (tmp_path / "nvidia_readme.txt").write_text("not json")
    icd = _write_nvidia_icd(tmp_path)
    gpu.select_gpu(GpuPreference.discrete, search_dirs=[tmp_path])
    assert os.environ["VK_ICD_FILENAMES"] == str(icd)


def test_discrete_searches_every_directory_given(tmp_path, fake_enumerate):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    icd_a = _write_nvidia_icd(dir_a)
    icd_b = _write_nvidia_icd(dir_b)
    gpu.select_gpu(GpuPreference.discrete, search_dirs=[dir_a, dir_b])
    assert os.environ["VK_ICD_FILENAMES"] == os.pathsep.join([str(icd_a), str(icd_b)])


def test_missing_search_directory_is_skipped_not_an_error(tmp_path, fake_enumerate):
    icd = _write_nvidia_icd(tmp_path)
    missing = tmp_path / "does-not-exist"
    gpu.select_gpu(GpuPreference.discrete, search_dirs=[missing, tmp_path])
    assert os.environ["VK_ICD_FILENAMES"] == str(icd)


def test_discrete_raises_naming_the_searched_directories_when_nothing_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match=re.escape(str(empty))):
        gpu.select_gpu(GpuPreference.discrete, search_dirs=[empty])
    for name in _ENV_VARS:
        assert name not in os.environ


def test_auto_falls_back_silently_when_no_nvidia_icd_found(tmp_path, fake_enumerate):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = gpu.select_gpu(GpuPreference.auto, search_dirs=[empty])
    assert result == "Stub Adapter"
    for name in _ENV_VARS:
        assert name not in os.environ


def test_auto_sets_vars_when_nvidia_icd_is_found(tmp_path, fake_enumerate):
    icd = _write_nvidia_icd(tmp_path)
    result = gpu.select_gpu(GpuPreference.auto, search_dirs=[tmp_path])
    assert os.environ["VK_ICD_FILENAMES"] == str(icd)
    assert result == "Stub Adapter"


def test_integrated_never_touches_environment_even_if_nvidia_icd_present(
    tmp_path, fake_enumerate
):
    _write_nvidia_icd(tmp_path)
    result = gpu.select_gpu(GpuPreference.integrated, search_dirs=[tmp_path])
    for name in _ENV_VARS:
        assert name not in os.environ
    assert result == "Stub Adapter"


def test_pre_existing_vk_driver_files_is_not_overwritten(
    tmp_path, fake_enumerate, monkeypatch
):
    monkeypatch.setenv("VK_DRIVER_FILES", "/custom/path.json")
    _write_nvidia_icd(tmp_path)
    result = gpu.select_gpu(GpuPreference.discrete, search_dirs=[tmp_path])
    assert os.environ["VK_DRIVER_FILES"] == "/custom/path.json"
    assert "VK_ICD_FILENAMES" not in os.environ
    assert "__NV_PRIME_RENDER_OFFLOAD" not in os.environ
    assert result == "Stub Adapter"


def test_pre_existing_vk_icd_filenames_is_not_overwritten(
    tmp_path, fake_enumerate, monkeypatch
):
    monkeypatch.setenv("VK_ICD_FILENAMES", "/custom/other.json")
    _write_nvidia_icd(tmp_path)
    gpu.select_gpu(GpuPreference.auto, search_dirs=[tmp_path])
    assert os.environ["VK_ICD_FILENAMES"] == "/custom/other.json"
    assert "VK_DRIVER_FILES" not in os.environ


def test_pre_existing_override_does_not_raise_for_discrete_even_without_a_search_hit(
    tmp_path, fake_enumerate, monkeypatch
):
    """An explicit override wins outright -- discrete must not fall back to
    its own search-and-raise behaviour once the caller has already decided."""
    monkeypatch.setenv("VK_DRIVER_FILES", "/custom/path.json")
    empty = tmp_path / "empty"
    empty.mkdir()
    result = gpu.select_gpu(GpuPreference.discrete, search_dirs=[empty])
    assert result == "Stub Adapter"


@pytest.mark.gpu
def test_off_returns_none_against_real_forge3d():
    assert gpu.select_gpu(GpuPreference.off) is None


@pytest.mark.gpu
def test_integrated_returns_the_real_active_adapter_name():
    name = gpu.select_gpu(GpuPreference.integrated)
    assert isinstance(name, str) and name


@pytest.mark.gpu
def test_discrete_selects_the_real_nvidia_adapter_when_present():
    """Opt-in: only meaningful on a machine with an NVIDIA Vulkan ICD.

    Runs in a subprocess. Vulkan ICD env vars are only read once, at loader
    init inside forge3d's native library, the first time anything in the
    process touches it -- so if an earlier ``@pytest.mark.gpu`` test in the
    same pytest run already triggered that init under a different ICD
    context (e.g. the offscreen/viewer smoke tests, which open a Session
    without calling ``select_gpu`` first), this assertion would silently
    pass or fail based on test *order* rather than on what ``select_gpu``
    actually configured. A fresh subprocess guarantees a fresh Vulkan
    instance every time, matching what actually happens in production
    (``pipeline.run`` calls ``select_gpu`` before anything else touches
    forge3d).
    """
    script = (
        "from eo_art.forge3d_pipes import gpu\n"
        "from eo_art.forge3d_pipes.config.schema import GpuPreference\n"
        "name = gpu.select_gpu(GpuPreference.discrete)\n"
        "assert name is not None and 'nvidia' in name.lower(), name\n"
    )
    env = dict(os.environ)
    for name in _ENV_VARS:
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
