# eo-art-sentinel-sr

Sentinel-2 RGBN super-resolution for `eo_art`'s acquire pipeline, kept as its
own nested project with its own `.venv` (`torch`, `cubo`, `mlstac`,
`opensr-model`, `sen2sr`, `stac2cube` pin a heavy, GPU-specific dependency
stack that would otherwise conflict with the rest of `eo_art`).

`eo_art.forge3d_pipes.acquire.sentinel_bridge` invokes this as a subprocess
(`uv run --project sentinel_sr eo-art-sentinel-sr ...`) and reads back the
`manifest.json` it writes.

## Usage

```
uv run eo-art-sentinel-sr \
  --lat 39.49 --lon -0.43 \
  --start-date 2023-01-01 --end-date 2023-12-31 \
  --model-path /path/to/LDSRS2-SEN2SR \
  --out /path/to/out
```
