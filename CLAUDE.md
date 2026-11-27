# CLAUDE.md – LiDAR Lens Dülmen

## Project
Semantic text search in airborne LiDAR point clouds of Dülmen (NRW, Germany).
A point cloud encoder is trained so that 50 m × 50 m point cloud patches are embedded into the CLIP space. A text query (e.g. "solar park") is encoded with the CLIP text encoder and compared with all patch embeddings. Results are shown in an Open Pioneer Trails web app with a heat map, a hit list and a 3D point cloud viewer.

**Core concept (LidarCLIP-style distillation):**
- **Teacher:** a frozen CLIP image encoder (GeoRSCLIP ViT-H/14) turns the orthophoto (DOP 2024) crop of each patch into a target vector. Computed once and stored. CLIP is never trained.
- **Student (the model that is trained):** a point cloud encoder learns to output the same vector from the point cloud of the patch. Loss: MSE on matching pairs (in LidarCLIP, Hess et al. 2023, Tab. 4, MSE gave clearly better retrieval than cosine), no labels needed.
- **Research question:** does combining point cloud and orthophoto embeddings improve text search compared to orthophoto embeddings alone? (LidarCLIP: P@10 0.94 combined vs. 0.86 image / 0.81 lidar, Tab. 5, car LiDAR.)
- **Search:** point cloud embeddings, orthophoto embeddings or their combination (mean of both vectors), always with the text encoder of the same CLIP model. The DOP is not needed at search time (only as map layer / preview).
- **Limit:** the encoder can only learn what CLIP recognises in the orthophotos. CLIP on DOP works well for flat concepts (forest, solar park, roundabout, parking lot) and poorly for tall vertical structures (church, wind turbine), see `docs/results.md`.
- **Validation plan:** first a Dülmen prototype with a geographic split inside Dülmen; then train on Münster and test on all of Dülmen (unseen area). Details in `docs/plan.md`.

Context: M.Sc. course "Foundation Models & Embeddings for EO Analytics", ifgi, Universität Münster (WS 26/27). One-person project.

## Project documents (read at the start of every session)
| File | Content |
|---|---|
| `CLAUDE.md` | rules, data, environment, decisions (this file) |
| `docs/plan.md` | use case, hypotheses, target values, method, evaluation design, status, timetable, next steps |
| `docs/results.md` | results with dates (feasibility, model comparison, baselines, metrics) |
| `docs/literature.md` | notes on the papers (core idea, what is used, key facts with page/figure) |
| `docs/encoder_comparison.md` | comparison of pretrained point cloud encoders and the reasons for the choice |
| GitHub issues | concrete tasks (milestones M0–M8) |

## Repository structure
```
pipeline/          Python pipeline (numbered steps)
  01_download/     tile list (make_tile_list.py), download (download_tiles.py), tiles_duelmen.csv
  02_preprocess/   patch grid (make_patch_grid.py), DOP crops (extract_dop_patches.py), geographic split (make_split.py), LAZ analysis (analyse_laz.py), HAG + point arrays (make_patch_points.py)
  03_finetune/     encoder training: PyTorch dataset (dataset.py), fine-tuning (train.py); runs in data/models/<run>/
  04_embed/        CLIP DOP embeddings per backbone, backbone comparison, DOP search
  05_evaluate/     evaluation of the LiDAR model – not started
backend/           FastAPI search API – not started
webapp/            Open Pioneer Trails app (React, OpenLayers) – not started
data/              all data (NOT in git)
docs/              plan, results, literature notes, figures
```

## Data (all under data/, never committed)
| Path | Content |
|---|---|
| `data/raw/laz/` | 3D-Messdaten NW (LAZ), 229 tiles of 1 km × 1 km |
| `data/raw/dop/` | DOP10 RGBI (JPEG2000), flight year 2024 (`*_2024.jp2`, from the historical DOP archive); old `*_2026.jp2` files may still be there but are not used |
| `data/raw/boundary/` | `duelmen_boundary.gpkg` (OSM, EPSG:25832) |
| `data/patches/` | `patch_index.parquet` (patch_id, tile, n_points, dop_file, dop_year, geometry), `images/` (DOP crops), `thumbs/` |
| `data/embeddings/` | `dop_image_embeddings_<backbone>.npz` + `.meta.json` (DOP 2024: GeoRSCLIP, OpenCLIP ViT-H/14, SkyCLIP), `text_cache/`, `compare/`; `2026/` = old embeddings on DOP 2026 |
| `data/processed/patch_points/` | point arrays per tile (`<tile>_xyz.npy` float16 x, y rel. to patch centre + HAG; `_intensity.npy`, `_class.npy`), `index.parquet` (patch_id, start, count, HAG range, split), `meta.json`; from `make_patch_points.py` |
| `data/models/` | one folder per training run (`train.py`): `config.json`, `log.jsonl`, `best.pt` (weights + metadata, best val recall@1), `last.pt` (for `--resume`) |
| `data/analysis/laz/` | LAZ analysis (#7): `tiles/*.json` cache, `tile_stats.csv`, `patch_stats.parquet`, `summary.md` |

Verified facts about the source data:
- Tiles: 1 km × 1 km, EPSG:25832 (UTM 32N), 229 tiles intersect the Dülmen boundary (184.7 km²), 73,898 patches of 50 m.
- LAZ (all 229 tiles checked, #7): LAS 1.2, point format 1, scale 0.01 m, EPSG:25832 + DHHN2016 height (EPSG:7837), adjusted standard GPS time. 8.1–19.4 points/m² (median 10.5), 5.4–9.5 first returns/m² (median 8.4), 8–19 million points per tile. Flight dates from GPS time: 2024-03-09/10/16/17 (203 tiles) and 2024-12-01 (26 tiles).
- 50 m patches: median 22,838 points (without noise), p1 16,904; only 59 of 73,898 patches have fewer than 4,096 points. Height range up to 116 m (wind turbines in 378_5749).
- LAZ classes (Nutzerinformationen 3D-Messdaten NRW, Stand 02/2020): 1 unclassified, 2 ground, 17 bridge, 18 noise, 20 last return not ground (buildings, cars, …), 24 basement, 26 synthetic ground. **There is no building class 6**; buildings are in class 20. Classes 9 and 14 also occur but are not documented (meaning open). Classes 9 and 26 carry GPS times of 2011-09-14 (older fill data).
- DOP: 4 bands (R, G, B, NIR), 0.1 m, 10,000 × 10,000 px per tile, uint8. 2024 tiles from `https://www.opengeodata.nrw.de/produkte/geobasis/lusat/hist/hist_dop/hist_dop_jp2_f10/hist_dop_2024/epsg_25832/`, all 229 Dülmen tiles available.
- Licence of LAZ and DOP: Datenlizenz Deutschland – Zero – 2.0.
- Open: exact DOP 2024 flight dates (in `dop_meta.zip` on the server).

## Environment
- Conda environment `pcsearch` (see `environment.yml`, conda-forge): PDAL, python-pdal, untwine, geopandas, pyarrow, rasterio, osmnx, faiss-cpu, fastapi, uvicorn, laspy[lazrs], open_clip_torch, jupyterlab, libgdal-jp2openjpeg
- PyTorch with CUDA via pip from the PyTorch index (`--extra-index-url https://download.pytorch.org/whl/cu132`, pinned in `environment.yml`). Verified: Python 3.11, torch 2.15.0+cu132, CUDA on the MX550.
- Activate: `conda activate pcsearch`. Run scripts in the activated env or via `conda run -n pcsearch ...`.
- PROJ conflict with a local PostGIS installation: `PROJ_LIB` and `PROJ_DATA` must be empty in the env (`conda env config vars set PROJ_LIB= PROJ_DATA=`). Calling `anaconda3\envs\pcsearch\python.exe` directly brings the `proj.db` error back.
- JPEG2000 needs `GDAL_DRIVER_PATH`, which `conda activate` sets. VS Code kernels may start without it ("plugin gdal_JP2OpenJPEG.dll is not available"): start VS Code from the activated env (`conda activate pcsearch; code .`, VS Code fully closed before) or use `jupyter lab` from the activated prompt.
- DOFA-CLIP needs a fork of open_clip in a separate env; its text embeddings are cached in `data/embeddings/text_cache/` (see `pipeline/04_embed/embed_dofaclip.py`).
- Encoder repos pin old stacks (ULIP: Python 3.7, torch 1.10; OpenShape: Python 3.9, torch 1.12, MinkowskiEngine/DGL). `pcsearch` stays on Python 3.11; encoder code runs here with relaxed pins (pure-PyTorch backbones) or separately on Colab.
- Local GPU: NVIDIA GeForce MX550, 2 GB VRAM: enough for CLIP inference and tests, NOT for training. Training runs on Colab or a university GPU.
- Web app: Node.js LTS and pnpm (Open Pioneer Trails conventions).
- OS: Windows, PowerShell.

## Running
- Pipeline scripts run step by step in numbered order; every script is re-runnable and skips existing outputs.
- Data: `make_tile_list.py` (default `--dop-year 2024`) → `download_tiles.py laz|dop|all` → `make_patch_grid.py` → `extract_dop_patches.py` → `make_split.py` → `04_embed/embed_<family>.py` → `compare_backbones.py` / `search_dop.py`.
- Backend (later): `uvicorn backend.main:app --reload`. Web app (later): see `webapp/README.md`.

## Key decisions (update when decided)
| Topic | Decision |
|---|---|
| Study area | Dülmen municipal boundary (test area); Münster as training area (planned) |
| DOP year | 2024, same year as the LiDAR; 2026 DOP dropped as training data because of changes between the flights (e.g. Solarpark Hausdülmen built 2025) |
| Patch size | 50 m × 50 m (provisional; larger patches open for tall objects) |
| Points per patch | 10,000 (OpenShape training setting), drawn at training time from all stored points (random, shuffled order: OpenShape's ball query takes the first points in list order); x, y, HAG divided by 25 m; RGB = 0.4. Patches with < 1,000 points skipped (proposal, 39 patches) |
| CLIP teacher | **GeoRSCLIP ViT-H/14** (`RS5M_ViT-H-14.pt`, HF `Zilun/GeoRSCLIP`, 1024-dim), chosen 2026-11 by visual comparison of the top 10 with OpenCLIP ViT-H/14 and SkyCLIP ViT-L/14 on DOP 2024 (see `docs/results.md`). All later steps (training targets, search) use only this model. Licence of the weights still to be checked |
| Point cloud encoder | **OpenShape PointBERT** (variant aligned to OpenCLIP ViT-bigG/14, HF `OpenShape/openshape-pointbert-vitg14-rgb`), fine-tuned with a new head LayerNorm(512) + Linear(512, 1024), linear layer zero-initialised (pretrained 1280-dim head dropped); smoke test on 6 Dülmen patches passed 2026-11-27 (one-off test, code not kept; results in `docs/results.md`). Input: 10,000 points, xyz with fixed scale (not unit ball), RGB filled with a constant (no DOP colours). Fallback: Uni3D-B. Comparison: `docs/encoder_comparison.md` |
| Training loss | MSE to the L2-normalised GeoRSCLIP vector, summed over the 1024 dims (better than cosine in LidarCLIP, Tab. 4); contrastive loss not required. Model selection: matching recall@1 on val; reference = mean-vector prediction (val loss 0.164) |
| Combined search | mean of DOP and LiDAR vector, then cosine similarity with the text vector (best joint method in LidarCLIP, Tab. 5) |
| Embedding storage | per backbone `.npz` + `.meta.json` now; GeoParquet + FAISS for the search index later |
| Point cloud serving | COPC, read per patch bounding box |
| 3D viewer | deck.gl PointCloudLayer for patches; Potree/CesiumJS optional |
| Evaluation | manual check of the top 10 per query on the test area (no OSM ground truth); same queries for DOP baseline, LiDAR model and combination |

## Rules
1. **CLIP consistency:** the CLIP text encoder used for search must be exactly the CLIP model the point cloud encoder was trained against. Never mix models; store the model name in the metadata and check it when loading.
2. **No data in git:** never commit LAZ, COPC, JPEG2000, npz, parquet, FAISS, checkpoints, PDFs or other large files. Respect `.gitignore`.
3. **Geographic split:** evaluate only on spatially separated test areas. Never use test patches for training or model selection.
4. **Metadata:** every embedding file and checkpoint stores model name, model version, CLIP model, patch size, data source and acquisition year.
5. **CRS:** keep all processing in EPSG:25832; transform to EPSG:4326/3857 only for the web app.
6. **Complete code:** when changing a script, deliver the complete file, not snippets or diffs.
7. **No invented facts:** if a fact about data, models or licences is not verified, mark it as open or ask instead of assuming.
8. **Licences:** check licences of pretrained weights before redistributing anything derived from them.
9. **Documentation:** results go into `docs/results.md`, decisions into this file, status and next steps into `docs/plan.md`, before a session ends.

## Workflow
- One chat / session per work package. Start: read `CLAUDE.md` and `docs/plan.md`, name the issue. End: update `docs/plan.md`, `docs/results.md` and the decisions here, then commit.
- Planning in GitHub issues (milestones M0–M8, MoSCoW priority). Reference issues in commits (`#12`).
- Code work in Claude Code (VS Code); presentations, report and literature in Cowork with the repo folder (and the course folder if needed) connected.
- Critical path: DOP 2024 embeddings → baseline A → encoder choice → Dülmen prototype training → Münster data + training → evaluation on Dülmen → search index → backend → web app.
