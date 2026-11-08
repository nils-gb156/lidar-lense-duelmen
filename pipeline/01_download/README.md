# 01 Download: LAZ and DOP tiles for Dülmen

This step downloads the airborne laser scanning point clouds (3D-Messdaten NW, LAZ) and digital orthophotos (DOP10 RGBI, JPEG2000) for the municipality of Dülmen from [OpenGeodata NRW](https://www.opengeodata.nrw.de/produkte/geobasis/).

| File | Purpose |
|---|---|
| `make_tile_list.py` | Builds `tiles_duelmen.csv` from the Dülmen boundary and the official file indexes |
| `tiles_duelmen.csv` | List of all tiles with exact LAZ and DOP file names and sizes (committed) |
| `download_tiles.py` | Downloads the files in the tile list to `data/raw/laz/` and `data/raw/dop/` |

## Quick start

```powershell
conda activate pcsearch
python pipeline/01_download/download_tiles.py all            # LAZ + DOP, ~23 GB
python pipeline/01_download/download_tiles.py all --check    # verify afterwards
```

The tile list is already in the repo, so `make_tile_list.py` only needs to run again if the study area changes or OpenGeodata NRW publishes new data (for example new DOP flights).

> **Windows / PowerShell:** if `conda` is not found, run `& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" init powershell` once and open a new terminal.
> Always run the scripts in the activated `pcsearch` environment (or via `conda run -n pcsearch ...`). Calling `envs\pcsearch\python.exe` directly skips the PROJ fix and causes `proj.db` errors.

## Step 1: tile list (`make_tile_list.py`)

```powershell
python pipeline/01_download/make_tile_list.py
```

What it does:

1. Fetches the Dülmen municipal boundary from OpenStreetMap (Nominatim via osmnx, query `Dülmen, Kreis Coesfeld, Nordrhein-Westfalen, Deutschland`) and saves it to `data/raw/boundary/duelmen_boundary.gpkg` in EPSG:25832.
2. Lists every 1 km × 1 km tile of the NRW tile grid that intersects the boundary.
3. Reads the official file indexes and looks up the exact file name and size of each tile:
   - LAZ: `https://www.opengeodata.nrw.de/produkte/geobasis/hm/3dm_l_las/3dm_l_las/index.json`
   - DOP: `https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/index.json`
4. Writes `tiles_duelmen.csv`.

Options: `--place` (another Nominatim query), `--out`, `--boundary-out`.

### Why the file names come from the index

The tiles are named after the lower-left corner of the tile in km (EPSG:25832, UTM zone 32N):

| Dataset | Pattern | Example |
|---|---|---|
| LAZ | `3dm_32_<E>_<N>_1_nw.laz` | `3dm_32_377_5741_1_nw.laz` |
| DOP | `dop10rgbi_32_<E>_<N>_1_nw_<year>.jp2` | `dop10rgbi_32_377_5741_1_nw_2026.jp2` |

The DOP name contains the flight year, which differs between tiles. It cannot be built from the coordinates, so it is taken from the index. If a tile appears more than once in the DOP index, the most recent year is used.

### Tile list format

One row per 1 km tile:

```csv
tile_id,easting_km,northing_km,laz_file,laz_size,dop_file,dop_size,dop_year
32_369_5744,369,5744,3dm_32_369_5744_1_nw.laz,62787344,dop10rgbi_32_369_5744_1_nw_2026.jp2,32823966,2026
```

| Column | Meaning |
|---|---|
| `tile_id` | `32_<E>_<N>`, UTM zone and lower-left corner in km |
| `easting_km`, `northing_km` | lower-left corner in km (EPSG:25832) |
| `laz_file`, `laz_size` | LAZ file name and size in bytes from the index |
| `dop_file`, `dop_size` | DOP file name and size in bytes from the index |
| `dop_year` | DOP flight year from the file name |

Download URL = folder of the index URL + file name. The sizes are used to check that downloads are complete.

### Current result (index state of 2026-11-08)

| | Value |
|---|---|
| Boundary area | 184.7 km² |
| Tiles intersecting the boundary | 229 |
| LAZ total | 15.8 GB |
| DOP total | 7.5 GB |
| DOP flight years | 2026: 224 tiles, 2024: 5 tiles |
| Tiles missing in an index | none |

## Step 2: download (`download_tiles.py`)

```powershell
python pipeline/01_download/download_tiles.py laz              # only LAZ  -> data/raw/laz/
python pipeline/01_download/download_tiles.py dop              # only DOP  -> data/raw/dop/
python pipeline/01_download/download_tiles.py all              # both
python pipeline/01_download/download_tiles.py all --workers 4  # more parallel downloads (default 3)
python pipeline/01_download/download_tiles.py laz --limit 1    # quick test with the first tile
python pipeline/01_download/download_tiles.py all --check      # only verify, no download
```

Behaviour:

- **Re-runnable:** files that exist with the expected size are skipped. An interrupted download can simply be started again.
- **No broken files:** each file is written to `<name>.part` and renamed only after its size matches the index.
- **Retries:** each file is tried 3 times. Files that still fail are written to `data/raw/<laz|dop>/download_failures.log` (time, file, URL, error), and the script exits with code 1. Run it again to retry only the missing files.
- **Check mode:** `--check` lists missing files and files with the wrong size and exits with code 1 if any are found.

At about 5 to 8 MB/s the full download takes roughly 1 hour.

## Output

```
data/raw/
  boundary/duelmen_boundary.gpkg   Dülmen boundary (OSM, EPSG:25832)
  laz/3dm_32_*_nw.laz              229 point cloud tiles
  dop/dop10rgbi_32_*_nw_*.jp2      229 orthophoto tiles
data/cache/osmnx/                  osmnx HTTP cache
```

Verified on test tiles:

| Dataset | Properties |
|---|---|
| LAZ | LAS 1.2, point format 1, EPSG:25832, about 10 million points per tile |
| DOP | 4 bands (R, G, B, NIR), 0.1 m resolution, 10000 × 10000 px, uint8, EPSG:25832 |

Reading the DOP `.jp2` files with rasterio/GDAL needs the conda-forge plugin `libgdal-jp2openjpeg`, which is included in `environment.yml`.

## Data licence

Both datasets come from Geobasis NRW / OpenGeodata NRW under [Datenlizenz Deutschland – Zero – Version 2.0](https://www.govdata.de/dl-de/zero-2-0) (checked on Open.NRW, 2026-11-08). Acquisition dates per tile are in `3dm_meta.zip` and `dop_meta.zip` next to the index files; a summary is in the main README.
