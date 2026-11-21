"""Band composites, spectral indices and stretches for super-resolved S2 rasters.

The SR step writes float32 reflectance (~0-1) with one GeoTIFF band per
Sentinel-2 band; band descriptions are STAC common names (``blue``,
``rededge_B05``, ``nir`` ...), see ``sentinel_sr.rgbn.band_descriptions``.
Channels below are named by band id (``B08``) or index name (``NDVI``).
"""

import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio import CRS
from rasterio.io import MemoryFile

from eo_art.forge3d_pipes.acquire.schema import Sentinel
from eo_art.forge3d_pipes.prep.ops import _percentile_bounds

DEFAULT_BANDS: tuple[str, ...] = Sentinel.bands

COMMON_TO_BAND = {
    "coastal": "B01",
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "nir08": "B8A",
    "nir09": "B09",
    "cirrus": "B10",
    "swir16": "B11",
    "swir22": "B12",
}
_BAND_ID_RE = re.compile(r"\bB(\d{1,2})(A?)\b", re.IGNORECASE)

# Stats (percentiles, equalization, histograms) only need a sample; this caps
# the cost on large rasters without visibly changing the bounds.
_MAX_SAMPLE = 1_000_000


def _band_id(description: str) -> str | None:
    """``rededge_B05`` / ``B5`` -> ``B05``, ``nir`` -> ``B08``, else None."""
    match = _BAND_ID_RE.search(description.replace("_", " "))
    if match:
        number, suffix = int(match.group(1)), match.group(2).upper()
        # Sentinel-2 pads B1..B9 to B01..B09 but writes the narrow NIR band as B8A.
        return f"B{number}A" if suffix else f"B{number:02d}"
    return COMMON_TO_BAND.get(description.strip().lower())


def resolve_bands(descriptions: Sequence[str | None], count: int) -> dict[str, int]:
    """Band id -> 0-based band position.

    Descriptions win; a raster without (or with unrecognised) descriptions
    falls back to the SR step's default band order if the count matches.
    """
    ids = [_band_id(d) if d else None for d in descriptions]
    if all(ids) and len(set(ids)) == len(ids) == count:
        return {band: i for i, band in enumerate(ids) if band}
    if count == len(DEFAULT_BANDS):
        return {band: i for i, band in enumerate(DEFAULT_BANDS)}
    return {band: i for i, band in enumerate(ids) if band and band not in ids[:i]}


def short(band: str) -> str:
    """``B08`` -> ``B8`` (the notation used in composite names)."""
    return re.sub(r"^B0", "B", band)


@dataclass(frozen=True, eq=False)
class Raster:
    data: np.ndarray  # float32 (bands, h, w), invalid pixels NaN
    bands: dict[str, int]
    crs: CRS | None
    transform: Affine

    def band(self, band_id: str) -> np.ndarray:
        return self.data[self.bands[band_id]]


def load_raster(source: Path | bytes) -> Raster:
    if isinstance(source, bytes):
        with MemoryFile(source) as memfile, memfile.open() as dataset:
            return _read(dataset)
    with rasterio.open(source) as dataset:
        return _read(dataset)


def _read(dataset) -> Raster:
    data = dataset.read(masked=True).astype("float32").filled(np.nan)
    data[~np.isfinite(data)] = np.nan
    return Raster(
        data=data,
        bands=resolve_bands(dataset.descriptions, dataset.count),
        crs=dataset.crs,
        transform=dataset.transform,
    )


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


@dataclass(frozen=True)
class Index:
    name: str
    bands: tuple[str, ...]
    formula: Callable[..., np.ndarray]
    info: str


def _savi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.5 * (nir - red) / (nir + red + 0.5)


def _evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    out[~np.isfinite(out)] = np.nan
    return out


INDICES: dict[str, Index] = {
    i.name: i
    for i in (
        Index("NDVI", ("B08", "B04"), _nd, "Vegetation vigour (NIR vs red)."),
        Index("NDWI", ("B03", "B08"), _nd, "Open water (green vs NIR)."),
        Index("MNDWI", ("B03", "B11"), _nd, "Water, robust against built-up land."),
        Index("NDBI", ("B11", "B08"), _nd, "Built-up areas and bare soil."),
        Index("NDMI", ("B08", "B11"), _nd, "Vegetation water content."),
        Index("NBR", ("B08", "B12"), _nd, "Burn severity (low = burnt)."),
        Index("NDRE", ("B08", "B05"), _nd, "Red-edge chlorophyll / plant stress."),
        Index("SAVI", ("B08", "B04"), _savi, "NDVI corrected for bright soil."),
        Index(
            "EVI",
            ("B08", "B04", "B02"),
            _evi,
            "NDVI without saturation in dense canopy.",
        ),
    )
}


def is_index(name: str) -> bool:
    return name in INDICES


def required_bands(names: Sequence[str]) -> list[str]:
    bands = []
    for name in names:
        for band in INDICES[name].bands if is_index(name) else (name,):
            if band not in bands:
                bands.append(band)
    return bands


def missing_bands(raster: Raster, names: Sequence[str]) -> list[str]:
    return [b for b in required_bands(names) if b not in raster.bands]


def compute_index(raster: Raster, name: str) -> np.ndarray:
    index = INDICES[name]
    return index.formula(*(raster.band(b) for b in index.bands)).astype("float32")


def channel(raster: Raster, name: str) -> np.ndarray:
    return compute_index(raster, name) if is_index(name) else raster.band(name)


@dataclass(frozen=True)
class Preset:
    group: str
    label: str
    channels: tuple[str, str, str]
    info: str

    @property
    def notation(self) -> str:
        return "-".join(short(c) for c in self.channels)


PRESETS: tuple[Preset, ...] = (
    Preset(
        "Natural",
        "True color",
        ("B04", "B03", "B02"),
        "What the eye would see.",
    ),
    Preset(
        "Vegetation",
        "Color infrared",
        ("B08", "B04", "B03"),
        "The classic: vegetation glows red, cities cyan-grey, water dark blue to "
        "black. Very graphic over farmland with many field patterns.",
    ),
    Preset(
        "Vegetation",
        "NIR-SWIR-blue",
        ("B08", "B11", "B02"),
        "Vegetation in strong orange-red, open soil in blue tones. High contrast.",
    ),
    Preset(
        "Vegetation",
        "Agriculture",
        ("B11", "B08", "B02"),
        "Vegetation lush green, fallow land and soil magenta-brown. Looks "
        "natural, but oversaturated.",
    ),
    Preset(
        "Vegetation",
        "Red edge",
        ("B8A", "B06", "B05"),
        "Fine gradations within vegetation, e.g. conifer vs. broadleaf or "
        "stress. Less garish, rather pastel.",
    ),
    Preset(
        "Vegetation",
        "Red edge (alt.)",
        ("B07", "B05", "B04"),
        "Fine gradations within vegetation, pastel tones.",
    ),
    Preset(
        "Water / coast",
        "Bathymetric",
        ("B04", "B03", "B01"),
        "Shallow water and sediment plumes turn turquoise to light blue, the "
        "seabed shines through. Spectacular in lagoons, reefs and estuaries.",
    ),
    Preset(
        "Water / coast",
        "Land-water contrast",
        ("B08", "B11", "B04"),
        "Water turns deep black to dark blue and stands out sharply from land. "
        "Good for lake districts, deltas and meandering rivers.",
    ),
    Preset(
        "Water / coast",
        "Stylised map",
        ("B12", "B8A", "B04"),
        "Water very dark, vegetation green, settlements violet. Looks almost "
        "like a stylised map.",
    ),
    Preset(
        "Water / coast",
        "Algae / chlorophyll",
        ("B05", "B04", "B03"),
        "Algal blooms and chlorophyll-rich water stand out through B5 (705 nm). "
        "A strong stretch on water surfaces brings out beautiful streaks.",
    ),
    Preset(
        "Urban / soil / geology",
        "Urban false color",
        ("B12", "B11", "B04"),
        "Cities in violet and magenta tones, vegetation green. Very aesthetic "
        "over megacities.",
    ),
    Preset(
        "Urban / soil / geology",
        "Geology",
        ("B12", "B11", "B02"),
        "Rock differences in deserts and mountains appear in ochre, pink and "
        "blue. Arid regions are often the most beautiful here.",
    ),
    Preset(
        "Urban / soil / geology",
        "Atmospheric penetration",
        ("B12", "B11", "B8A"),
        "Largely cuts through haze; cool, technical colors. Good for hazy scenes.",
    ),
    Preset(
        "Snow / fire",
        "Snow and ice",
        ("B12", "B11", "B02"),
        "Snow and ice glow bright blue, clouds white, rock brown. Ideal for the "
        "Alps and glaciers.",
    ),
    Preset(
        "Snow / fire",
        "Surreal snow",
        ("B02", "B11", "B12"),
        "Snow turns red instead of blue, clouds stay white. Surreal effect.",
    ),
    Preset(
        "Snow / fire",
        "Burn scars",
        ("B12", "B8A", "B04"),
        "Burn scars dark red to brown, active fires glow orange-yellow in SWIR.",
    ),
    Preset(
        "Snow / fire",
        "Burn scars (SWIR)",
        ("B12", "B11", "B8A"),
        "Burn scars dark red to brown, active fires glow orange-yellow in SWIR.",
    ),
    Preset(
        "Indices",
        "SWIR-NDVI-MNDWI",
        ("B11", "NDVI", "MNDWI"),
        "A composition that doesn't exist spectrally, but separates water and "
        "green as far as possible.",
    ),
)


class Method(StrEnum):
    PERCENTILE = "percentile"
    MINMAX = "minmax"
    EQUALIZE = "equalize"


@dataclass(frozen=True)
class Stretch:
    method: Method = Method.PERCENTILE
    lower: float = 2.0
    upper: float = 98.0
    # MINMAX: one (lo, hi) per channel, or a single one when shared.
    manual: tuple[tuple[float, float], ...] = ()
    shared: bool = True
    gamma: float = 1.0


def sample(values: np.ndarray) -> np.ndarray:
    """Finite values, strided down to at most ``_MAX_SAMPLE``."""
    flat = values.ravel()
    flat = flat[:: max(1, flat.size // _MAX_SAMPLE)]
    return flat[np.isfinite(flat)]


def value_range(values: np.ndarray) -> tuple[float, float]:
    valid = sample(values)
    if valid.size == 0:
        return 0.0, 1.0
    return float(valid.min()), float(valid.max())


def percentile_bounds(
    channels: Sequence[np.ndarray], lower: float, upper: float, shared: bool
) -> list[tuple[float, float]]:
    if shared:
        pooled = np.concatenate([sample(c) for c in channels])
        return [_percentile_bounds(pooled, lower, upper)] * len(channels)
    return [_percentile_bounds(sample(c), lower, upper) for c in channels]


def pooled_reference(
    frames: Sequence[Sequence[np.ndarray]],
) -> list[np.ndarray]:
    """Per channel, finite samples pooled over every frame of a timeseries.

    Passed as ``reference`` to ``normalize`` it makes all frames share one
    set of stretch statistics, so brightness differences between dates
    survive instead of every frame being auto-levelled on its own.
    """
    if not frames:
        raise ValueError("no frames")
    per_frame = max(1, _MAX_SAMPLE // len(frames))
    pooled = []
    for channel in zip(*frames, strict=True):
        parts = []
        for values in channel:
            flat = values.ravel()
            flat = flat[:: max(1, flat.size // per_frame)]
            parts.append(flat[np.isfinite(flat)])
        pooled.append(np.concatenate(parts))
    return pooled


def bounds(
    channels: Sequence[np.ndarray],
    stretch: Stretch,
    reference: Sequence[np.ndarray] | None = None,
) -> list[tuple[float, float]] | None:
    """Linear (lo, hi) per channel; None for histogram equalization.

    Statistics come from ``reference`` (per-channel samples, e.g.
    ``pooled_reference``) when given, else from the channels themselves.
    """
    stats = channels if reference is None else reference
    match stretch.method:
        case Method.PERCENTILE:
            return percentile_bounds(
                stats, stretch.lower, stretch.upper, stretch.shared
            )
        case Method.MINMAX:
            manual = list(stretch.manual)
            if len(manual) == 1:
                manual *= len(channels)
            if len(manual) != len(channels):
                raise ValueError(
                    f"minmax stretch needs 1 or {len(channels)} ranges, got {manual}"
                )
            return [(lo, hi if hi > lo else lo + 1e-9) for lo, hi in manual]
        case Method.EQUALIZE:
            return None


def _equalize(channel: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.size == 0:
        return np.zeros_like(channel)
    levels = np.linspace(0.0, 1.0, 1025)
    quantiles = np.quantile(reference, levels)
    return np.interp(channel, quantiles, levels)


def normalize(
    channels: Sequence[np.ndarray],
    stretch: Stretch,
    reference: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Each channel mapped to float [0, 1] (NaN stays NaN), gamma applied.

    See ``bounds`` for ``reference``.
    """
    linear = bounds(channels, stretch, reference)
    if linear is None:
        stats = channels if reference is None else reference
        pooled = np.concatenate([sample(c) for c in stats])
        refs = [pooled if stretch.shared else sample(c) for c in stats]
        out = [_equalize(c, r) for c, r in zip(channels, refs, strict=True)]
    else:
        out = [
            np.clip((c - lo) / (hi - lo), 0.0, 1.0)
            for c, (lo, hi) in zip(channels, linear, strict=True)
        ]
    if stretch.gamma != 1.0:
        out = [np.power(c, 1.0 / stretch.gamma) for c in out]
    return out


def _to_uint8(unit: np.ndarray) -> np.ndarray:
    return (np.nan_to_num(unit, nan=0.0) * 255.0 + 0.5).astype("uint8")


def to_rgb8(
    channels: Sequence[np.ndarray],
    stretch: Stretch,
    reference: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    if len(channels) != 3:
        raise ValueError(f"an RGB composite needs 3 channels, got {len(channels)}")
    return np.dstack([_to_uint8(c) for c in normalize(channels, stretch, reference)])


def to_colormap8(
    channel: np.ndarray,
    stretch: Stretch,
    cmap: str,
    reference: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    (unit,) = normalize([channel], stretch, reference)
    rgba = matplotlib.colormaps[cmap](np.nan_to_num(unit, nan=0.0), bytes=True)
    return np.ascontiguousarray(rgba[..., :3])


def encode_png(rgb8: np.ndarray) -> bytes:
    return iio.imwrite("<bytes>", rgb8, extension=".png")


_FRAME_DATE_RE = re.compile(r"_sr_(\d{4}-\d{2}-\d{2})$")


def frame_date(path: Path) -> str | None:
    """``valley_sr_2023-02-14.tif`` -> ``2023-02-14`` (SR timeseries naming)."""
    match = _FRAME_DATE_RE.search(Path(path).stem)
    return match.group(1) if match else None


def encode_gif(frames: Sequence[np.ndarray], seconds: float, loop: bool) -> bytes:
    """Animated GIF from uint8 RGB frames of equal size.

    All frames are quantized against one palette built from a sample of every
    frame; a palette per frame (Pillow's default) makes colours jump from one
    date to the next.
    """
    if not frames:
        raise ValueError("no frames")
    # A few rows from each frame, stacked: small, but covers every date's colours.
    rows = max(1, 256 // len(frames))
    swatch = np.concatenate([f[:: max(1, f.shape[0] // rows)] for f in frames], axis=0)
    palette = Image.fromarray(swatch).quantize(
        colors=256, method=Image.Quantize.MEDIANCUT
    )
    images = [Image.fromarray(f).quantize(palette=palette) for f in frames]
    buffer = io.BytesIO()
    options = {"loop": 0} if loop else {}
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=round(seconds * 1000),
        optimize=False,
        **options,
    )
    return buffer.getvalue()


def indices_geotiff(raster: Raster, names: Sequence[str]) -> bytes:
    """Float32 GeoTIFF with one band per index, on the source raster's grid."""
    if not names:
        raise ValueError("no indices selected")
    _, height, width = raster.data.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(names),
        "height": height,
        "width": width,
        "crs": raster.crs,
        "transform": raster.transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            for i, name in enumerate(names, start=1):
                dataset.write(compute_index(raster, name), i)
                dataset.set_band_description(i, name)
        return memfile.read()
