"""Selects which GPU forge3d renders on.

forge3d exposes no adapter-selection API of its own; the only lever
available is the Vulkan loader's ICD environment variables, which steer
which physical device the loader exposes to forge3d's native library.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from eo_art.forge3d_pipes.config.schema import GpuPreference

logger = logging.getLogger(__name__)

# Kept as a module-level constant (rather than hardcoding a single path) so
# tests can point discovery at a tmp_path instead of the real system, and so
# a future Vulkan installation layout doesn't require code changes.
DEFAULT_ICD_SEARCH_DIRS: tuple[Path, ...] = (
    Path("/usr/share/vulkan/icd.d"),
    Path("/etc/vulkan/icd.d"),
)

_OVERRIDE_VARS = ("VK_DRIVER_FILES", "VK_ICD_FILENAMES")


def _find_nvidia_icds(search_dirs: Sequence[Path | str]) -> list[Path]:
    found = []
    for directory in search_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.json")):
            if "nvidia" in candidate.name.lower():
                found.append(candidate)
    return found


def _active_adapter_name() -> str | None:
    # Imported here, never at module top level: the ICD environment must be
    # set before forge3d's native library initialises the Vulkan instance,
    # and any module-level import would run at package-import time, before
    # `select_gpu` has had a chance to run.
    import forge3d

    adapters = forge3d.enumerate_adapters()
    return adapters[0]["name"] if adapters else None


def select_gpu(
    preference: GpuPreference,
    search_dirs: Sequence[Path | str] = DEFAULT_ICD_SEARCH_DIRS,
) -> str | None:
    """Point the Vulkan loader at the requested GPU.

    Returns the selected adapter's name (from ``forge3d.enumerate_adapters``)
    so the caller can log it, or ``None`` for ``GpuPreference.off``.
    """
    if preference is GpuPreference.off:
        return None

    if preference in (GpuPreference.discrete, GpuPreference.auto):
        pre_existing = [name for name in _OVERRIDE_VARS if os.environ.get(name)]
        if pre_existing:
            logger.info(
                "gpu selection skipped: %s already set in the environment; "
                "respecting the explicit override",
                ", ".join(pre_existing),
            )
        else:
            icds = _find_nvidia_icds(search_dirs)
            if icds:
                joined = os.pathsep.join(str(path) for path in icds)
                os.environ["VK_DRIVER_FILES"] = joined
                os.environ["VK_ICD_FILENAMES"] = joined
                os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
            elif preference is GpuPreference.discrete:
                searched = ", ".join(str(d) for d in search_dirs)
                raise RuntimeError(
                    "gpu: discrete GPU requested but no NVIDIA Vulkan ICD "
                    f"(*.json containing 'nvidia') was found; searched: {searched}"
                )
            # else: preference is auto and none was found -- fall back to
            # the loader's default silently, no environment changes.

    # `integrated` leaves the environment untouched; `discrete`/`auto` have
    # either set the ICD vars above or fallen back -- either way, report
    # whichever adapter forge3d ends up handing out.
    return _active_adapter_name()
