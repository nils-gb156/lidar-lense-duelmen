# CLAUDE.md – LiDAR Lens Dülmen

## Project
Semantic text search in airborne LiDAR point clouds of Dülmen (NRW, Germany).
A pretrained point cloud encoder is fine-tuned so that 50 m × 50 m point cloud patches are embedded into the same space as CLIP embeddings of the matching orthophoto. A text query (e.g. "wind turbine") is encoded with the CLIP text encoder and compared with all patch embeddings. Results are shown in an Open Pioneer Trails web app with a heat map, a hit list and a 3D point cloud viewer.

Context: M.Sc. course "Foundation Models & Embeddings for EO Analytics", ifgi, Universität Münster (WS 26/27). One-person project.

## Repository structure
```
pipeline/          Python pipeline (numbered steps)
  01_download/     LAZ, DOP and OSM download scripts, tile lists
  02_preprocess/   COPC conversion, height normalisation, patch grid, patch/image extraction, split
  03_finetune/     encoder fine-tuning (dataset, training, checkpoints)
  04_embed/        patch embeddings → GeoParquet, FAISS index
  05_evaluate/     query set, metrics, comparison of variants
backend/           FastAPI search API
webapp/            Open Pioneer Trails app (React, OpenLayers)
data/              all data (NOT in git)
docs/              results, figures, report material
.github/           issue templates, issue contents
```

## Data (all under data/, never committed)
| Path | Content |
|---|---|
| `data/raw/laz/` | 3D-Messdaten NW (airborne laser scanning, LAZ) from OpenGeodata NRW |
| `data/raw/dop/` | Digitale Orthophotos NW (JPEG2000) from OpenGeodata NRW |
| `data/copc/` | point cloud tiles converted to COPC |
| `data/processed/` | tiles with height above ground (HAG) |
| `data/patches/` | `patch_index.parquet` (patch_id, geometry, tile, n_points, split), point arrays, image crops |
| `data/groundtruth/` | OSM reference data and query set |
| `data/embeddings/` | `duelmen_patches.parquet` (patch embeddings), `duelmen.faiss` |
| `data/models/` | checkpoints (fine-tuned encoder) |

Known facts about the source data (verified): point density on average 4–10 points/m², LAZ (LAS 1.2, PDRF 1), classified point cloud, orthophotos DOP10 RGBI in JPEG2000.
Still to be documented in README once checked: tile size, CRS/EPSG, height reference, acquisition years, exact licence.

## Environment
- Conda environment `pcsearch` (see `environment.yml`, conda-forge): PDAL, python-pdal, untwine, geopandas, pyarrow, rasterio, osmnx, faiss-cpu, fastapi, uvicorn, laspy[lazrs], open_clip_torch, jupyterlab
- PyTorch with CUDA installed via pip from the PyTorch index (`--extra-index-url https://download.pytorch.org/whl/cu132`, pinned in the pip section of `environment.yml` so open_clip_torch does not pull a CPU-only torch). Verified: Python 3.11, torch 2.15.0+cu132, CUDA available on the MX550.
- Activate: `conda activate pcsearch`
- Encoder repos pin old stacks (ULIP: Python 3.7, torch 1.10; OpenShape: Python 3.9, torch 1.12, MinkowskiEngine/DGL). `pcsearch` stays on Python 3.11. Encoder code runs either in this environment with relaxed pins (pure-PyTorch backbones) or in a separate environment on Colab. Decide in the issue "Choose the encoder".
- PROJ conflict with a local PostGIS installation: `PROJ_LIB` and `PROJ_DATA` must be empty in the environment (`conda env config vars set PROJ_LIB= PROJ_DATA=`). If rasterio/PDAL report an incompatible `proj.db`, check this first.
- Local GPU: NVIDIA GeForce MX550 with 2 GB VRAM. Enough for inference and small tests, NOT for fine-tuning. Fine-tuning runs on Colab or a university GPU.
- Web app: Node.js LTS and pnpm (Open Pioneer Trails conventions).
- OS: Windows, PowerShell.

## Running
- Backend: `uvicorn backend.main:app --reload`
- Web app: see `webapp/README.md` (OPT dev server)
- Pipeline scripts are run step by step in numbered order; every script must be re-runnable and skip existing outputs.

## Key decisions (update when decided)
| Topic | Decision |
|---|---|
| Study area | Dülmen municipal boundary |
| Patch size | 50 m × 50 m (provisional) |
| Points per patch | 4096 (provisional, depends on encoder) |
| Point cloud encoder | open: ULIP or OpenShape (issue "Choose the encoder") |
| CLIP model | open: e.g. RemoteCLIP or open_clip model (issue "Choose the CLIP model") |
| Embedding storage | GeoParquet (one row per patch) + FAISS index |
| Point cloud serving | COPC, read per patch bounding box |
| 3D viewer | deck.gl PointCloudLayer for patches; Potree/CesiumJS optional for full city |
| Evaluation | OSM ground truth, precision@k / recall@k on the geographic test split |

## Rules
1. **CLIP consistency:** the CLIP text encoder used for search must be exactly the CLIP model the point cloud encoder was aligned/fine-tuned to. Never mix models; store the model name in embedding metadata and check it when loading.
2. **No data in git:** never commit LAZ, COPC, JPEG2000, parquet, FAISS, checkpoints or other large files. Respect `.gitignore`.
3. **Geographic split:** evaluation only on the spatially separated test split. Never use test patches for training or model selection.
4. **Metadata:** every embedding file and checkpoint stores model name, model version, CLIP model, patch size, data source and acquisition year.
5. **CRS:** keep all processing in the point cloud CRS; transform to EPSG:4326/3857 only for the web app.
6. **Complete code:** when changing a script, deliver the complete file, not snippets or diffs.
7. **No invented facts:** if a fact about data, models or licences is not verified, mark it as open or ask instead of assuming.
8. **Licences:** check licences of pretrained weights before redistributing anything derived from them.

## Workflow
- Planning in GitHub issues (milestones M0–M8, labels for type and area, MoSCoW priority: must/should/could). Reference issues in commits (`#12`).
- Critical path: data → preprocessing → encoder choice → fine-tuning → embeddings → FAISS → backend → web app search. Encoder choice and fine-tuning are the main risks; check feasibility early.
- Document results in `docs/results.md` and decisions in this file.