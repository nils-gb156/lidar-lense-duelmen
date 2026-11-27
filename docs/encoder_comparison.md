# Point cloud encoder comparison – LiDAR Lens Dülmen

Researched 2026-11-27. Goal: a pretrained point cloud encoder that is fine-tuned with the LidarCLIP method (MSE to the frozen GeoRSCLIP ViT-H/14 image vectors of the DOP 2024, 1024-dim) on 50 m × 50 m ALS patches. Facts are taken from the project READMEs / papers listed below; "not stated" means the sources do not say it.

**Decision: OpenShape PointBERT, variant aligned to OpenCLIP ViT-bigG/14** (provisional, until a first patch has gone through the model).

## Comparison

| Encoder | Pretrained on | Aligned to | Input | Size | Setup | Code licence | Weights |
|---|---|---|---|---|---|---|---|
| **OpenShape PointBERT** (variants for CLIP ViT-B/32, ViT-L/14, OpenCLIP ViT-bigG/14) | ≈ 1 M 3D models (Objaverse, ShapeNet, 3D-FUTURE, ABO) | OpenCLIP ViT-bigG/14 (main model) | 10,000 points, XYZ-RGB, centred + unit ball; gravity axis z for bigG, y for B/32 and L/14 | not stated | recent PyTorch + DGL via the `openshape-demo-support` loader; MinkowskiEngine only needed for the SparseConv models (per README) | Apache-2.0 | Hugging Face (`OpenShape/openshape-pointbert-vitg14-rgb` …), licence not stated |
| ULIP-2 PointBERT | Objaverse + ShapeNet triplets (image, BLIP-2 caption, point cloud) | OpenCLIP ViT-G/14 | 10k XYZ-RGB | 22.1 M | Python 3.7, PyTorch 1.10.1, CUDA 11; repo archived 06/2026 | BSD-3-Clause | Hugging Face (`SFXX/ulip`), licence not stated |
| Uni3D (B / L / g) | same ensemble as OpenShape | EVA-CLIP-E | 10k coloured points; encoder initialised from a 2D ViT | 88 M / 307 M / 1 B | Python 3.8, PyTorch 2.0.1, compiled CUDA op `pointnet2_ops` | MIT | Hugging Face (`BAAI/Uni3D`), licence not stated |
| LidarCLIP SST | ONCE driving dataset, ≈ 1 M scenes (car LiDAR + camera) | OpenAI CLIP ViT-L/14 or B/32 | voxels in a fixed range around the car, camera frustum | ≈ 8.5 M | PyTorch 1.9, mmcv-full 1.3.8, mmdet 2.14, SST build (Docker) | not verified | Google Drive links in the README, licence not stated |
| ALS pre-training (BEV-MAE, Xiu et al. 2025) | 17,000 km² USGS 3DEP airborne LiDAR | none (self-supervised) | voxels, xyz | not stated | PyTorch 2.3.1, OpenPCDet, spconv | not stated | **not released yet** ("Coming soon") |
| ForPT (LitePT-S) | forest ALS / ULS / MLS | none (self-supervised) | per-point model, 20 m chunks | 12.4 M | not checked | not checked | Hugging Face (`prs-eth/ForPT`), licence not verified; forest only |

## Why OpenShape PointBERT
- Pretrained for alignment with a CLIP space, i.e. the closest pretraining task to this project.
- Easiest to run on current PyTorch / Colab (loader library on Hugging Face, no old CUDA stack).
- Same PointBERT architecture as ULIP-2, so ULIP-2 weights could be tried later without changing the model code.
- The bigG variant uses z as vertical axis, like the ALS data.
- A 50 m patch with a few thousand points is structurally similar to an "object" point cloud (one fixed-size, locally normalised point set).

## Why not the others
- **ULIP-2:** same architecture, but old software stack and archived repo.
- **Uni3D:** modern stack, but larger and needs the `pointnet2_ops` build; fallback if OpenShape fails.
- **LidarCLIP SST:** side-view driving scenes in a voxel grid around the car, aligned to another CLIP model, oldest and most fragile stack (mmcv/mmdet 2021). The domain advantage (outdoor LiDAR) is smaller than it looks, because perspective, sensor pattern and input format differ.
- **ALS-specific models:** the only general ALS model has no released weights yet; ForPT is forest-only and per-point.

## Adaptations for this project (apply to every option)
- **New projection head:** none of the encoders is aligned to GeoRSCLIP; replace the head with a 1024-dim one. Only the backbone features transfer.
- **No DOP colours as input** (research question: LiDAR vs. DOP). Fill RGB with a constant (OpenShape suggests 0.4) or use intensity as grey value.
- **Normalisation:** the models normalise each object to a unit ball. Here absolute height matters, so use one fixed scale for all patches (e.g. divide by 25 m) and let fine-tuning adapt.
- **Points:** 10,000 per patch (OpenShape training setting); store more per patch and sample at training time.
- **Weight licences** are not stated for any model: check before publishing derived results or weights (CLAUDE.md rule 8).

## Domain gap: what "CAD data" means here
The CLIP-aligned encoders were pretrained on synthetic 3D models, not on real scans:
- **ShapeNet:** CAD models of everyday objects in about 55 categories (chairs, cars, planes, lamps …).
- **Objaverse:** hundreds of thousands of user-made 3D models from Sketchfab (furniture, vehicles, characters, some buildings and scenes).
- **3D-FUTURE / ABO:** furniture and product models.
- The point clouds are sampled from the mesh surfaces: complete from all sides, evenly dense, no noise, one centred object.

ALS patches are different: seen only from above (roofs and canopies, hardly any walls), 4–10 points/m², noise and multiple returns, ground plus many objects in one patch. What transfers are generic geometric features (planes, edges, cylinders, curvature, height structure), not the semantics. Fine-tuning on Dülmen (and Münster) has to close the rest of the gap.

## Sources
- ULIP: https://github.com/salesforce/ULIP ; ULIP-2 paper: https://arxiv.org/abs/2305.08275
- Uni3D: https://github.com/baaivision/uni3d ; paper: https://arxiv.org/abs/2310.06773
- OpenShape: https://github.com/Colin97/OpenShape_code ; models: https://huggingface.co/OpenShape ; loader: https://huggingface.co/OpenShape/openshape-demo-support ; paper: https://arxiv.org/abs/2305.10764
- LidarCLIP: https://github.com/atonderski/lidarclip ; paper: Hess et al. (2023), arXiv:2212.06858, Sec. 4 and App. B
- ALS pre-training: https://arxiv.org/abs/2501.05095 ; https://github.com/martianxiu/ALS_pretraining
- ForPT: https://huggingface.co/prs-eth/ForPT ; https://arxiv.org/abs/2609.24787
