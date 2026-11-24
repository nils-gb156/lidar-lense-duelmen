# Literature notes – LiDAR Lens Dülmen

PDFs are kept in the course folder, not in this repo. One section per source: core idea, what the project uses, key facts with page/figure.

## Radford et al. (2021) – CLIP
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. arXiv:2103.00020.

- **Core idea:** an image encoder and a text encoder are trained jointly on 400 million (image, text) pairs from the internet. In each batch the matching pairs (diagonal of the N × N similarity matrix) must be more similar than all other pairs (contrastive pre-training, Fig. 1).
- **Zero-shot:** class names are written into a prompt ("A photo of a {object}."), encoded by the text encoder; the closest text wins (Fig. 1, parts 2 and 3).
- **Used in the project:** frozen CLIP image encoder as teacher for the DOP crops, CLIP text encoder for search queries. The project uses open_clip re-implementations (e.g. ViT-B/32 trained on LAION-2B), not the original OpenAI weights.
- **Figure for slides:** Fig. 1 (p. 2).

## Hess et al. (2023) – LidarCLIP
Hess, G., Tonderski, A., Petersson, C., Åström, K., & Svensson, L. (2023). LidarCLIP or: How I learned to talk to point clouds. arXiv:2212.06858.

- **Core idea:** there are no text–lidar datasets, but millions of image–lidar pairs. A lidar encoder is trained to reproduce the embedding of a frozen CLIP image encoder for the matching image; images act as intermediary between text and point clouds (Sec. 3, Fig. 2).
- **Loss:** MSE or cosine similarity on matching pairs only (Eq. 3, 4). Advantage over contrastive loss: no large batch sizes needed; the goal is to map into an existing space, not to learn a new one (Sec. 3).
- **Setup:** automotive dataset ONCE, about 1 million scenes, about 7 million image–lidar pairs; CLIP ViT-L/14 (d = 768); lidar encoder SST (Sec. 4). Point clouds are cropped to the camera field of view.
- **Results:** lidar-based retrieval is generally on par with image-based retrieval, with complementary strengths; combining image and lidar scores improves both (abstract, Sec. 3.1).
- **Noise:** preprocessing errors and visibility differences between sensors are mitigated by the large number of training pairs (Sec. 3) – the project has far fewer pairs (≈ 74 k in Dülmen), hence pair filtering and Münster as additional training area.
- **Used in the project:** the training concept (teacher–student distillation, MSE/cosine loss), joint retrieval as variant "combination". The published weights do not fit (automotive data, different CLIP model).
- **Figure for slides:** Fig. 2 (p. 3).

## Xue et al. (2023) – ULIP
Xue, L., Gao, M., Xing, C., Martín-Martín, R., Wu, J., Xiong, C., Xu, R., Niebles, J. C., & Savarese, S. (2023). ULIP: Learning a unified representation of language, images, and point clouds for 3D understanding. arXiv:2212.05171.

- **Core idea:** a 3D encoder is aligned with the pre-aligned image–text space of a frozen vision-language model, using automatically synthesised triplets (text, rendered image, point cloud) of CAD models (Fig. 1, Sec. 3).
- **Data:** ShapeNet55, about 52.5 k 3D objects with 55 categories; text from the category metadata with prompt templates.
- **Training:** contrastive loss between 3D and image / text features; CLIP frozen.
- **Backbone-agnostic:** PointNet++, PointMLP, PointBERT; improves 3D classification and zero-shot classification (ModelNet40, ScanObjectNN).
- **Used in the project:** candidate for pretrained encoder weights (ULIP / ULIP-2) as starting point. Limits: single objects instead of aerial scenes (domain gap), aligned to a different CLIP model than the project's teacher, old software stack.
- **Figure for slides:** Fig. 1 (p. 1).

## Data documentation
- Bezirksregierung Köln (2020). Nutzerinformationen für die 3D-Messdaten aus dem Laserscanning für NRW. Stand 02/2020. – Point classes (1, 2, 9, 17, 18, 20, 21, 24, 26); class 20 = last return not ground, used for building models; no dedicated building class.
- Geobasis NRW. 3D-Messdaten NW; Digitale Orthophotos NW (current and historical, DOP10 since 2014 with 10 cm GSD). OpenGeodata NRW, dl-de/zero-2.0.

## Models used, papers not read yet
Weights used for the DOP embeddings; the corresponding papers have not been read and summarised yet:
- RemoteCLIP (weights: Hugging Face `chendelong/RemoteCLIP`)
- GeoRSCLIP, RS5M dataset (weights: Hugging Face `Zilun/GeoRSCLIP`)
- SkyCLIP, SkyScript dataset (weights: SkyScript release, `SkyCLIP_ViT_L14_top30pct`)
- DOFA-CLIP (weights: Hugging Face `earthflow/GeoLB-ViT-B-16-SigLIP-All-EO`)
