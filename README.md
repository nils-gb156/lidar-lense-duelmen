# LiDAR Lens Dülmen

**Semantic text search in airborne LiDAR point clouds using CLIP-aligned embeddings and a 3D web viewer.**

> Status: early development. The pipeline, model and web app are being built. See the [project board](../../projects) and [issues](../../issues) for the current state.

## Idea

Type *"wind turbine"* and see where in Dülmen the airborne laser scan contains one, then jump into the 3D point cloud of each hit.

The project divides the airborne LiDAR point cloud of Dülmen (NRW, Germany) into 50 m × 50 m patches. A pretrained point cloud encoder is fine-tuned so that each patch is embedded into the same vector space as CLIP embeddings of the matching aerial image. Because CLIP also embeds text into this space, the point cloud becomes searchable with natural language, without manually labelled training data.

The project is part of the M.Sc. course *Foundation Models & Embeddings for EO Analytics* at the Institute for Geoinformatics (ifgi), University of Münster.

## How it works

```
Offline (once)
  LAZ tiles ──► COPC ──► height above ground ──► 50 m patches ──┐
  Orthophotos ──────────────────────────────► image crops ──────┤
                                                                 ▼
             fine-tune point cloud encoder ◄── CLIP image embeddings (frozen)
                                                                 ▼
             patch embeddings ──► GeoParquet + FAISS index

Online (web app)
  text query ──► CLIP text encoder ──► FAISS search ──► heat map + hit list
                                                    └──► 3D view of the patch (from COPC)
```

## Research question

Can a point cloud encoder aligned to CLIP enable semantic text search in airborne LiDAR, and where does 3D information help compared to searching aerial images alone?

Planned comparison:

| Variant | Input |
|---|---|
| Baseline A | CLIP on aerial images only |
| Baseline B | hand-crafted 3D features (height percentiles, density) |
| Baseline C | pretrained point cloud encoder without fine-tuning |
| Main model | fine-tuned point cloud encoder |
| Optional | fusion of point cloud and image |

Evaluation uses OpenStreetMap objects (e.g. wind turbines, solar parks, buildings, forest) as ground truth and measures precision@k / recall@k on a spatially separated test area.

## Tech stack

| Component | Technology |
|---|---|
| Point cloud processing | PDAL, laspy, untwine (COPC) |
| Raster processing | rasterio |
| Machine learning | PyTorch, open_clip, pretrained point cloud encoder (ULIP or OpenShape, to be decided) |
| Embedding storage and search | GeoParquet, FAISS |
| Backend | FastAPI |
| Web app | Open Pioneer Trails (React, OpenLayers), deck.gl for the 3D view |
| Ground truth | OpenStreetMap via osmnx |

## Repository structure

```
pipeline/          data processing, fine-tuning, embeddings, evaluation (numbered steps)
backend/           FastAPI search API
webapp/            Open Pioneer Trails web app
data/              local data (not in git)
docs/              results and figures
```

## Setup

> The setup will be completed as the project progresses.

**Requirements**
- [Miniforge](https://github.com/conda-forge/miniforge) (conda)
- NVIDIA GPU with CUDA for local inference (fine-tuning runs on a remote GPU)
- Node.js LTS and pnpm for the web app

**Python environment**
```bash
conda env create -f environment.yml
conda activate pcsearch
```
`environment.yml` already installs PyTorch with CUDA 13.2 from the PyTorch index. For a different CUDA version, change the `--extra-index-url` in the pip section (see https://pytorch.org/get-started/locally).

On systems with a local PostGIS installation, clear the PROJ variables inside the environment:
```bash
conda env config vars set PROJ_LIB= PROJ_DATA=
```

**Data**

All data is stored locally under `data/` and is not part of the repository. Download the point clouds and orthophotos for Dülmen (229 tiles, about 23 GB):
```bash
python pipeline/01_download/download_tiles.py all
```
Details on the tile list, file naming and options: [pipeline/01_download/README.md](pipeline/01_download/README.md).

## Data sources

| Data | Source | Licence |
|---|---|---|
| Airborne laser scanning point clouds (3D-Messdaten NW) | [Geobasis NRW / Open.NRW](https://ckan.open.nrw.de/dataset/ef0b51eb-31ea-49dd-a1c3-42e565c2b1a1) | [Datenlizenz Deutschland – Zero – Version 2.0](https://www.govdata.de/dl-de/zero-2-0) |
| Digital orthophotos (Digitale Orthophotos NW) | [Geobasis NRW / Open.NRW](https://ckan.open.nrw.de/dataset/56fb584b-10cf-4009-a405-0bef06bb3e00) | [Datenlizenz Deutschland – Zero – Version 2.0](https://www.govdata.de/dl-de/zero-2-0) |
| Reference objects | [OpenStreetMap](https://www.openstreetmap.org) | ODbL, © OpenStreetMap contributors |

Known data details:

| | LAZ (3D-Messdaten NW) | DOP (DOP10 RGBI) |
|---|---|---|
| Tile size | 1 km × 1 km | 1 km × 1 km |
| CRS | EPSG:25832 (ETRS89 / UTM 32N) | EPSG:25832 |
| Format | LAZ, LAS 1.2, point format 1 | JPEG2000, 4 bands (RGB + NIR), 0.1 m |
| Tiles for Dülmen | 229 (15.8 GB) | 229 (7.5 GB) |
| Height reference | DHHN2016 (normal heights) | – |
| Acquisition date | 2024: 203 tiles 9–17 March, 26 tiles 1 December | 2026: 189 tiles 7 April (flight 1463/26), 35 tiles 5 March (1464/26); 2024: 5 tiles 26 June |
| Accuracy (official) | 0.30 m horizontal, 0.15 m vertical, 4 pts/m² nominal | 0.10 m ground pixel size |

Source of the acquisition dates: per-tile metadata `3dm_meta.zip` and `dop_meta.zip` next to the `index.json` files on OpenGeodata NRW (state 2026-10-29 / 2026-10-21).

**Time gap:** the point clouds are from 2024 and most orthophotos are from 2026. Changes in between (new buildings, solar parks, felled forest) make image and point cloud disagree for some patches. This matters for fine-tuning, since the CLIP image embeddings are the training target.

## Related work

- Hess et al. (2024): *LidarCLIP or: How I Learned to Talk to Point Clouds*, WACV 2024: aligns automotive LiDAR with CLIP.
- ULIP and OpenShape: align object point clouds with CLIP image and text embeddings.
- Goel & Jung: *ALMA: Airborne LiDAR Masked Autoencoding* (self-supervised learning on airborne LiDAR).

## License

Code: MIT, see [LICENSE](LICENSE).
Data and pretrained models are subject to their own licences (see Data sources).

## Author

Nils Große Beikel – M.Sc. Geoinformatics and Spatial Data Science, University of Münster