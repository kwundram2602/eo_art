# eo_art — Design Spec

**Date:** 2026-05-01  
**Status:** Approved

---

## Ziel

Ein Python-Paket das Earth-Observation-Daten (Multispektral, DEM, Zeitreihen, Vektoren) in künstlerische Visualisierungen verwandelt — 2D-Bilder, Animationen, und 3D-Szenen. Inspiriert von rayshader (R), aber breiter als reine Terrain-Visualisierung.

---

## Paketstruktur

```text
eo_art/
├── core/
│   ├── data.py          # EOData: xarray-zentrierte Hauptklasse
│   ├── io.py            # GeoTIFF, STAC, xarray loaders
│   └── catalog.py       # Sentinel/Landsat/SRTM/Copernicus DEM
├── render2d/
│   ├── hillshade.py     # xarray-spatial / whitebox unter der Haube
│   ├── composite.py     # RGB, NDVI, false-color
│   └── style.py         # Paletten & Texturen (Imhof, etc.)
├── render3d/
│   ├── backend_pyvista.py   # primary 3D backend
│   ├── backend_plotoptix.py # optional, GPU-Raytracing, später
│   └── scene.py             # Backend-Abstraktion
├── export/
│   ├── mesh.py          # OBJ/PLY/glTF via trimesh
│   └── blender.py       # bpy, optional dependency
├── art/
│   ├── generative.py    # abstrakte/generative Modi
│   └── presets.py       # programmatische "Looks"
└── styles/              # JSON/YAML Style-Definitionen
```

---

## Architektur

Drei klare Schichten, die nur die darunterliegende kennen:

1. **`core/`** — Datenschicht. Lädt, normalisiert, hält Spatial-Kontext.
2. **`render2d/` / `render3d/`** — Verarbeitungsschicht. Hillshade, Composite, 3D-Mesh.
3. **`export/`** — Ausgabeschicht. Schreibt Dateien, wählt Format.

`art/` sitzt quer dazu: nutzt die Render-Schicht für höherwertige, kreative Outputs.

---

## Kernklasse: `EOData`

```python
@dataclass
class EOData:
    ds: xr.Dataset
    crs: str                  # immer gesetzt, z.B. "EPSG:32632"
    resolution: float         # in CRS-Einheiten
    kind: Literal["raster", "timeseries", "dem", "vector"]
```

`kind` wird beim Laden automatisch erkannt:

- hat `time`-Dimension → `timeseries`
- single-band float ohne `time` → `dem`
- GeoDataFrame → `vector`
- sonst → `raster`

---

## Public API

### Einstiegspunkte

```python
from eo_art import EOData

eo = EOData.from_file("dem.tif")
eo = EOData.from_stac("sentinel-2-l2a", bbox=..., datetime="2024-06")
eo = EOData.from_xarray(ds, crs="EPSG:4326")
```

### 2D Pipeline

```python
# Wenn EOData ein DEM-Band enthält, nutzt .hillshade() es automatisch.
# Ansonsten: dem= explizit übergeben.
eo.composite.rgb(bands=["B04", "B03", "B02"])
  .hillshade(azimuth=315, altitude=45)          # DEM aus eo.ds["dem"] oder dem=...
  .style(palette="imhof", blend="overlay")
  .render("map.png")

# Oder: DEM und Multispektral getrennt laden, dann kombinieren
dem = EOData.from_file("dem.tif")
rgb = EOData.from_file("sentinel.tif").composite.rgb()
dem.hillshade().blend(rgb).style(palette="imhof").render("map.png")
```

### 3D Pipeline

```python
eo.scene3d(dem=dem_layer)
  .drape(texture=rgb_composite)
  .render("scene.glb")
```

### Zeitreihe → Animation

```python
eo.composite.ndvi()
  .style(palette="rdylgn")
  .animate("timelapse.gif", fps=4)
```

### Chaining-Mechanismus

Jede Methode gibt ein neues unveränderliches Objekt zurück. Pipelines sind reproducierbar:

```python
base = eo.composite.rgb()
light = base.style(palette="pastel")
dark  = base.style(palette="dark_terrain")
# base bleibt unverändert
```

Der Renderer wird automatisch anhand des `output`-Formats gewählt:

- `.png` / `.svg` → Matplotlib / datashader
- `.gif` / `.mp4` → Animationsrenderer
- `.glb` / `.obj` / `.ply` → PyVista

---

## Layer-Typen

| Typ            | Beispiel              | Primäre Verwendung   |
| -------------- | --------------------- | -------------------- |
| `raster`       | Sentinel-Kachel       | composite, hillshade |
| `dem`          | SRTM, Copernicus DEM  | hillshade, 3D-Mesh   |
| `timeseries`   | NDVI-Stapel           | animate              |
| `vector`       | Flüsse, Grenzen       | overlay auf Raster   |

---

## Style-System

**`styles/` (deklarativ)** — YAML/JSON-Dateien für fertige "Looks":

```yaml
# styles/imhof_alpine.yaml
name: imhof_alpine
palette: imhof
hillshade:
  azimuth: 315
  altitude: 45
  blend: overlay
  opacity: 0.6
saturation_boost: 1.2
```

Abrufbar via `eo.style(preset="imhof_alpine")`.

**`art/presets.py` (programmatisch)** — für Looks die Logik brauchen (z.B. automatische Farbpalette aus dem Bild extrahieren, generative Überblendungen).

---

## Abhängigkeiten

### Kern (immer installiert)

- `xarray`, `rioxarray`, `rasterio`
- `geopandas`
- `xarray-spatial`, `whitebox`
- `matplotlib`
- `numpy`

### Optional

```toml
[project.optional-dependencies]
stac    = ["pystac-client", "planetary-computer", "eodag"]
dem     = ["py3dep", "elevation"]
3d      = ["pyvista", "trimesh"]
gpu     = ["plotoptix"]
blender = []  # bpy kommt aus Blenders eigener Python-Umgebung
```

---

## Fehlerbehandlung

Validierung nur an Systemgrenzen:

- **`io.py` / `catalog.py`** — `EODataLoadError`, `CRSMissingError` bei fehlenden/ungültigen Daten
- **Optionale Dependencies** — beim Import sofort mit klarer Meldung scheitern, nicht erst beim Aufruf
- **Pipeline-Inneres** — keine defensive Absicherung; xarray/numpy werfen selbst sinnvolle Fehler

---

## Testing-Strategie

- Kleine GeoTIFF-Fixtures (~10×10px) als echte Testdaten — kein Mocking von xarray/rasterio
- Unit-Tests für Shader-Funktionen mit bekannten Input/Output-Werten
- Integration-Tests für Lade- und Export-Pfade
- 3D/GPU-Tests nur wenn PyVista verfügbar: `pytest.importorskip("pyvista")`

---

## Offene Entscheidungen

| Thema                 | Optionen                      | Wann entscheiden                 |
| --------------------- | ----------------------------- | -------------------------------- |
| Blender-Connector     | bpy-Script vs. glTF-Export    | wenn 3D-Basis steht              |
| PlotOptiX-Integration | optionales Extra              | wenn PyVista-Backend fertig      |
| Catalog-Abstraktion   | eodag vs. pystac-client       | beim ersten `catalog.py`-Sprint  |
| Stil-Format           | YAML vs. TOML vs. Python-Dict | beim ersten Preset               |

---

## Nicht in Scope (v0.1)

- Blender-Connector (kommt später)
- PlotOptiX / GPU-Raytracing
- Web-App / interaktive Widgets
- CLI-Tool
