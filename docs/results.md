# Results – LiDAR Lens Dülmen

Newest results at the top of each section. All results so far are for the **teacher side** (CLIP on DOP); no point cloud encoder has been trained yet.

## 2026-11-27 – Point cloud analysis, all 229 LAZ tiles (#7)
Script: `pipeline/02_preprocess/analyse_laz.py` (every point of every tile read with laspy). Outputs: `data/analysis/laz/` (`tile_stats.csv`, `patch_stats.parquet`, `summary.md`), figures `docs/figures/laz_*.png`.

**Format and CRS (all tiles identical):** LAS 1.2, point format 1, scale 0.01 m, EPSG:25832 + vertical EPSG:7837 (DHHN2016 height), adjusted standard GPS time. Header creation dates 2024-07-15 to 2026-05-11 are processing dates, not flight dates.

**Points and density:**
| | min | median | max |
|---|---|---|---|
| points per tile | 8.1 M | 10.5 M | 19.4 M |
| all points per m² | 8.1 | 10.5 | 19.4 |
| first returns per m² (≈ pulse density) | 5.4 | 8.4 | 9.5 |

Total 2.56 billion points. Median 31.6 % of the points come from multi-return pulses (vegetation). The densest tiles (17–19 pts/m²) have many multiple returns, not more pulses. The December tiles have the lowest pulse density (5.4–7.0 /m²).

**Classes:**
| Class | Meaning (NRW user information 02/2020) | Share |
|---|---|---|
| 2 | ground | 67.7 % |
| 1 | unclassified (vegetation, among others) | 26.2 % |
| 20 | last return not ground (buildings, cars, …) | 5.8 % |
| 26 | synthetic ground | 0.22 % |
| 18 | noise | 0.02 % |
| 9 | **not documented** | 0.02 % |
| 14 | **not documented** | 0.01 % |
| 17 | bridge | 0.01 % |
| 24 | basement | 0.01 % |

- Class 9 and class 26 carry GPS times of **2011-09-14** (older data filled in); for class 9 the 2011 count equals the class 9 count in every tile. Class 14 has 2024 GPS times. Meaning of classes 9 and 14 open (in ASPRS LAS: 9 water, 14 wire conductor; not verified for NRW).

**Flight dates** (GPS time, UTC, without class 9 / 26): 2024-03-09 (134 tiles), 03-10 (15), 03-16 (37), 03-17 (17), 2024-12-01 (26, south-west of the town). 103 tiles contain points from more than one day (strip overlaps). This matches the per-tile metadata (203 tiles 9–17 March, 26 tiles 1 December). Both periods are leaf-off.

**50 m patches inside the boundary (73,898), points without noise:**
- min 1,809, p1 16,904, p5 18,464, median 22,838, p95 54,198, max 110,991
- fewer than 4,096 points: 59 patches (0.08 %), fewer than 8,192: 111 (0.15 %), fewer than 10,000: 125 (0.17 %), mostly in a few tiles (e.g. `386_5747`, `374_5746`; probably water, not checked)
- patches without a ground point: 24
- height range (max z − mean ground z): median 11.2 m, p95 32.2 m, p99 37.4 m, max 116.2 m. 20,976 patches > 20 m, 6,344 > 30 m, 24 > 50 m (wind turbines in `377_5749`/`378_5749`)

**Visualisation** (`laz_tile_32_378_5749.png`, `laz_patch_32_378_5749_05_06.png`): wind turbine patch with 23,178 points, of which only 98 lie more than 10 m above ground (90 of them above 50 m: nacelle and rotor at 100–115 m), almost no points on the tower. A random sample of 4,096 keeps on average about 17 of them, a random sample of 10,000 about 42. Rotor points spread over several neighbouring 50 m patches.

**Consequences for preprocessing:**
- Points per patch: 10,000 (OpenShape PointBERT, provisional encoder choice) means downsampling by about 2.3× (median); 4,096 would be about 5×. Only 125 patches (0.17 %) have fewer than 10,000 points and need padding/duplication.
- Random downsampling drops thin tall structures (wind turbines, masts): use farthest point sampling or keep all points above ground, then sample the rest.
- Use height above ground (HAG), not DHHN2016 heights: absolute heights range from 43 m to 180 m over Dülmen. Normalising z by the patch size (50 m) clips the 24 patches > 50 m; a fixed z-scale (e.g. divide by 50 m without clipping) keeps them.
- Drop classes 18 (noise); keep 9 and 26 out of the training points or flag them, since they are not from 2024.
- 50 m patch size: well sampled (≈ 23k points). Tall objects (turbines, church spires) are cut across patches, as already seen in the feasibility test; this supports testing larger or overlapping patches later.

## 2026-11-24 – Teacher choice: three CLIP models on DOP 2024 (visual)
Setup: 73,898 DOP 2024 crops, prompt "an aerial photo of {query}", top 10 per model and query (`data/embeddings/compare/`). Counts = estimated correct hits out of 10, by eye.

| Query | GeoRSCLIP H/14 | OpenCLIP H/14 | SkyCLIP L/14 |
|---|---|---|---|
| forest, farmland, lake, parking lot, railway tracks, industrial building, playground, soccer pitch | ≈ 10 | ≈ 10 | ≈ 10 |
| solar park | ~6 (rooftop PV mixed in) | 10 | 9 |
| cemetery / graveyard | ~9 / ~9 | ~8 / ~8 | ~6 / ~6 |
| roundabout | ~10 | ~9 | ~6 |
| tennis court | 10 | ~7 | 9 |
| football field | ~6 | ~8 | 10 |
| swimming pool | ~7 | ~9 | ~6 |
| greenhouse | ~9 | ~8 | ~8 |
| allotment gardens | ~6 | ~3 | ~6 |
| construction site | ~8 | ~7 | ~8 |
| power line pylon | ~8 | ~3 | ~7 |
| wind turbine | ~5 | ~2 | ~6 |
| large flat roof | ~9 | 10 | ~3 |
| biogas plant | ~8 | 10 | ~8 |
| church | fails | fails | fails |

- **Decision: GeoRSCLIP ViT-H/14** as teacher: most clean top-10 lists for objects, most consistent between synonyms, best of the three for pylons and wind turbines. OpenCLIP ViT-H/14 is a close second.
- General: flat and shape-based concepts work very well; tall structures stay weak (church fails for all). Similarities lie close together (e.g. 0.27–0.30), so only the ranking is meaningful.
- Expectation for the LiDAR model: shape-based concepts (halls, roundabouts, greenhouses, forest, sports fields) should transfer better than colour-based ones (tennis clay, solar panels).

## 2026-11 – Time gap between LiDAR and DOP 2026 → switch to DOP 2024
- Solarpark Hausdülmen (tile `380_5740`): fully visible in the DOP 2026, but the LiDAR patch contains only ground points (heights within ±0.3 m above ground). GPS time of the LiDAR: 17 March 2024. OSM `start_date` of the solar park: 2025.
- A comparison solar field (tile `382_5745`, "Solarfeld Dülmen") shows panels at about 3 m height in the LiDAR.
- Consequence: pairs of LiDAR 2024 and DOP 2026 can be wrong training pairs (teacher says "solar park", input is a bare field). Decision: DOP 2024 from the historical DOP archive (all 229 tiles available). A pair filter is still planned for the remaining differences.

## 2026-11-14 – Six CLIP models on all Dülmen patches (DOP 2026, qualitative)
Setup: 73,898 DOP crops (50 m, 224 px), prompt "an aerial photo of {query}", top 10 per model and query (`data/embeddings/compare/`, script `pipeline/04_embed/compare_backbones.py`).

| Model | Architecture | Pretraining data | Dim |
|---|---|---|---|
| OpenCLIP | ViT-B/32 (`laion2b_s34b_b79k`) | LAION-2B | 512 |
| OpenCLIP | ViT-H/14 (`laion2b_s32b_b79k`) | LAION-2B | 1024 |
| RemoteCLIP | ViT-L/14 | remote sensing image–text | 768 |
| GeoRSCLIP | ViT-H/14 | RS5M | 1024 |
| SkyCLIP | ViT-L/14 (top30pct) | SkyScript | 768 |
| DOFA-CLIP | ViT-B/16 (SigLIP) | Earth observation | 768 |

Observations (visual inspection, not yet measured):
- **Solar park:** five models return ground-mounted solar parks; RemoteCLIP returns mostly rooftop PV, plus pylons.
- **Wind turbine:** most models find turbines, but mainly through their long shadows; confusion with power pylons and silos.
- **Church:** fails for all six models (large houses with steep roofs).
- Similarity values are only comparable within one model.
- No final teacher choice yet; a quantitative comparison (baseline A with ground truth) is needed.

Note: computed on DOP 2026; to be recomputed on DOP 2024.

## 2026-11-14 – OpenCLIP ViT-B/32 search on all Dülmen patches (DOP 2026, qualitative)
Top 12 per query (`data/embeddings/search/`, script `pipeline/04_embed/search_dop.py`).

| Query | Assessment |
|---|---|
| forest, parking space, roundabout, solar panels | very good (almost all correct) |
| lake, swimming pool | good |
| train station | good for railway tracks |
| houses | plausible |
| terraced house | partial |
| wind turbine | partial (3 turbines and one shadow in the top 9, otherwise pylons, silos) |
| church, graveyard | bad |
| corn maze | unclear (fields with tractor tracks) |

- Similarity values lie close together (about 0.30–0.39): only the ranking is meaningful, no fixed threshold.
- Most "solar panels" hits come from Solarpark Hausdülmen, which is not in the LiDAR 2024 (see above).

## 2026-11 – Feasibility test: CLIP on orthophotos (3 tiles)
Setup: tiles `381_5743` (town centre with church), `378_5749` (wind turbines), `380_5740` (solar park), 1,200 patches of 50 m, DOP 2026 at 224 px, prompt "an aerial photo of {query}". Models: OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) and RemoteCLIP ViT-B/32. The feasibility notebook and figures are no longer in the repo.

| Query | OpenCLIP B/32 | RemoteCLIP B/32 |
|---|---|---|
| solar park | good | partial |
| forest | good | partial |
| farmland | good | bad |
| flat roof buildings | good | partial |
| terraced houses | partial | bad |
| church | bad | bad |
| wind turbine | bad (shadows only) | bad |

precision@10 against OSM:

| Query | OpenCLIP B/32 | RemoteCLIP B/32 | base rate | OSM-positive patches |
|---|---|---|---|---|
| wind turbine | 0.0 | 0.0 | 0.002 | 3 |
| solar park | 1.0 | 0.4 | 0.045 | 54 |
| church | 0.0 | 0.0 | 0.008 | 10 |

- In the LAZ tiles, the three wind turbines (`378_5749`, up to about 116 m above ground) and the church spire of St. Viktor (`381_5743`, up to about 74 m) are clearly visible.
- Conclusion: CLIP on aerial images works for flat, area-covering concepts and fails for tall vertical structures; the point cloud contains these structures.
- Caveats: only 3 tiles, OSM incomplete, a 50 m patch is smaller than a turbine rotor, 224 px loses detail.
