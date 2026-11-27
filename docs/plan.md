# Project plan – LiDAR Lens Dülmen

Last updated: 2026-11-24 (teacher chosen)

## Course framework
- Course: Foundation Models & Embeddings for EO Analytics, ifgi Münster, WS 26/27, 10 ECTS (presentation "Usage and Tooling" held on 5 Nov 2026 + this project)
- Status presentation held in November 2026 (idea, CLIP, data, DOP embeddings with six CLIP models)
- Requested for the next presentation: explicit use case, timetable, hypotheses, target values, methodology
- Open: submission deadline, final presentation date, grading of the project (report? code? demo?)

## Use cases (draft, to be confirmed)

### Problem
Airborne LiDAR covers all of NRW, but a point cloud contains no words: you cannot ask it "where are the solar parks?". Today, finding objects relies on maps like OSM or on fixed classes:
- **OSM** contains only what someone has mapped and tagged. Objects can be missing (e.g. new or small structures), and many properties are rarely or never tagged.
- **Fixed classes** (e.g. the NRW LiDAR classification: ground, bridge, noise, "last return not ground") do not distinguish a hall from a church or a solar park from a field. Every new search term would need new labels and retraining.

### Main use case: make objects searchable that maps do not contain
A user (e.g. municipality, planning office, OSM mapper) types a term such as "solar park", "greenhouse" or "large flat-roof hall" and gets a map of matching places in the point cloud. Comparing the hits with OSM gives a list of candidates that are **not mapped yet**, which can then be checked and added.

### Further use cases
- **Concepts without a tag or class:** search for descriptions instead of categories, e.g. "flat roof industrial building", "farm with silos", "allotment gardens", even if no matching OSM tag or LiDAR class exists.
- **Find similar:** select one patch (e.g. a biogas plant) and find places that look alike, without any text and without a vocabulary limit.
- **Exploring large point clouds:** quickly find examples of a structure in many GB of LiDAR, e.g. to build a labelled training set for another model (LidarCLIP uses retrieval for data curation in the same way).
- **No retraining for new questions:** the search terms are free text; the model is trained once.

### Limits (to state honestly)
- The encoder learns only what CLIP recognises in the orthophotos: it should work for things visible from above (solar parks, halls, forest, roundabouts) and poorly for tall vertical structures (church towers, wind turbines), see `docs/results.md`.
- Hits are 50 m patches, not exact objects; small objects in a patch may be missed.
- OSM is used only as a comparison / ground truth, not as input of the model.

## Goal
Text search in the Dülmen LiDAR point cloud, using a point cloud encoder that was trained only with CLIP embeddings of orthophotos as teacher (no labels).

## Research question
Does combining point cloud and orthophoto embeddings improve text search compared to orthophoto embeddings alone?

Evidence from LidarCLIP (Hess et al. 2023, Tab. 5, car-mounted LiDAR + camera, ONCE val, mean over all prompts): P@10 image only 0.856, lidar only 0.813, combined (mean of features) 0.944. Open: whether this transfers to airborne LiDAR + orthophotos with ≈ 74 k instead of ≈ 7 M pairs.

## Hypotheses
- **H1:** A point cloud encoder trained only with DOP-CLIP targets enables text search in the point cloud that is comparable to CLIP search on the DOP.
- **H2:** An encoder trained on Münster works in Dülmen, an area it has never seen.
- **H3:** DOP and LiDAR combined beat each source alone, especially for tall structures (church, wind turbine).

## Target values (proposal, not yet confirmed)
- precision@10 ≥ 0.7 on average for flat classes (forest, solar park, roundabout, parking lot)
- LiDAR search at most 0.1 below baseline A (DOP-CLIP) on average
- combined search better than both single sources for church and wind turbine

## Method
1. Data: LAZ 2024 + DOP 2024 (same flight year), 50 m patch grid (Dülmen: 73,898 patches)
2. Teacher: frozen CLIP image encoder → one target vector per DOP crop (computed once)
3. Filter: remove patches where DOP and point cloud do not match (check CLIP similarity vs. point cloud height statistics)
4. Student: point cloud encoder (OpenShape PointBERT bigG + new 1024-dim head), input 10,000 points per patch (x, y, HAG / 25 m, RGB = 0.4), trained with MSE loss to reproduce the target vector (MSE better than cosine in LidarCLIP, Tab. 4)
5. Embeddings: point cloud embeddings for all test patches → search index
6. Search: text → CLIP text encoder (same CLIP model) → cosine similarity → ranking
7. Evaluation (below), then backend and web app

## Evaluation design
- **Stage 1 – Dülmen prototype:** geographic split inside Dülmen in contiguous blocks (e.g. 70 % train, 10 % validation, 20 % test). Goal: pipeline works end to end, the encoder learns.
- **Stage 2 – generalisation:** train on Münster (check that LiDAR and DOP years match there first), test on all of Dülmen.
- **Check 1 (training works):** for each test patch, is its point vector closest to the CLIP vector of its own DOP among all test patches? → matching recall@1. No ground truth needed.
- **Check 2 (search works):** fixed query set (the list in `pipeline/04_embed/compare_backbones.py`); top 10 patches per query in the test area, checked by eye; share of correct hits = precision@10.
- **Ground truth:** manual check of the top 10 (decided). OSM is not used for evaluation, because it is incomplete and its tags do not match the queries reliably.
- **Variants on the same queries and patches:**
  - Baseline A: CLIP on the DOP (no training, measurable now)
  - Baseline B: hand-crafted height features (no deep learning)
  - Baseline C: point cloud encoder before training
  - LiDAR model (Dülmen prototype / Münster-trained)
  - Combination: mean of DOP and LiDAR vector, then cosine similarity with the text vector (best joint method in LidarCLIP, Tab. 5)

## Status
| Step | Status |
|---|---|
| Repo, conda env, CLAUDE.md, issues | done |
| Tile list + download LAZ (229 tiles) | done |
| DOP 2024 download + patch grid + DOP crops | done (commit 17cc465) |
| CLIP DOP embeddings, six models (on DOP 2026) | done, qualitative comparison in `docs/results.md` |
| CLIP DOP embeddings on DOP 2024 (GeoRSCLIP, OpenCLIP H/14, SkyCLIP) | done |
| Teacher choice | done: GeoRSCLIP ViT-H/14 (visual comparison) |
| Geographic split (`make_split.py`, 2 km blocks, 70/10/20) | script written, to run |
| Encoder choice | done (provisional): OpenShape PointBERT ViT-bigG/14, see `docs/encoder_comparison.md` |
| Point cloud analysis (#7: CRS, density, classes, dates, points per patch) | done 2026-11-27, see `docs/results.md` |
| OpenShape PointBERT smoke test (6 patches) | done 2026-11-27: runs, fixed 25 m scale chosen |
| Point cloud patches (#10 HAG, #12 cut): `make_patch_points.py` | done 2026-11-27: all 73,898 patches, all points stored (float16, 18 GB in `data/processed/patch_points/`); sampling to 10,000 at training time |
| PyTorch dataset (#15): `pipeline/03_finetune/dataset.py` | done 2026-11-27: points (6, 10,000) + GeoRSCLIP target, train 50,020 / val 8,284 / test 15,555 patches (39 with < 1,000 points skipped); rotation/jitter optional, off by default |
| Dülmen prototype training | open |
| Münster data + training, evaluation on Dülmen | open |
| Search index, backend, web app | open |
| Report, final presentation | open |

## Open decisions
- Patch size (50 m vs. larger for tall objects; #7: 50 m is well sampled, wind turbine rotors span several patches)
- Sampling to 10,000 points (random vs. farthest point sampling / keep all points above ground, #7); handling of undocumented classes 9 and 14 and of class 26 (2011 data)
- Münster: LiDAR and DOP years, data volume
- Rotation augmentation: the DOP target is north-up and is not rotated with the points; test none vs. 90° steps in #22 (option: embed rotated DOP crops as matching targets)

## Timetable
Open until the submission deadline and the presentation dates are known. Planned order: DOP 2024 embeddings → baseline A → encoder + point cloud patches → Dülmen prototype → Münster training + evaluation → search index, backend, web app → report and presentation.

## Next 3 steps
1. Baseline A (#18): DOP search with GeoRSCLIP on the test area, top 10 per query counted by hand (reference for all later variants)
2. Training PC: copy `data/processed/patch_points/`, `data/patches/patch_index.parquet` and `data/embeddings/dop_image_embeddings_georsclip.*`; set up the environment (#21)
3. Fine-tuning script (#22) on top of `pipeline/03_finetune/dataset.py`: MSE to the GeoRSCLIP DOP vector, train split only, val for model selection (matching recall@1), compare rotation augmentation none / 90° on val
