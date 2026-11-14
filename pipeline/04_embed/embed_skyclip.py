"""CLIP image embeddings of all DOP patches: SkyCLIP (continual pretraining on SkyScript, OSM-labelled aerial images).

Weights: SkyScript S3 bucket, zip with one epoch_20.pt (downloaded and unpacked to
data/cache/weights/skyclip/ on the first run, the zip is deleted afterwards).
Variants (--variant):
    vitl14_top30pct   ViT-L-14, 768 dim, 4.3 GB download (default)
    vitl14_top50pct   ViT-L-14, 768 dim, 4.3 GB download
    vitb32_top50pct   ViT-B-32, 512 dim, 1.4 GB download

The checkpoints contain the full training state (5.1 GB / 1.8 GB unpacked), which needs
that much free RAM while loading. The ViT-L-14 variants use QuickGELU, as in the SkyScript notebooks.

Output: data/embeddings/dop_image_embeddings_skyclip_<variant>.npz (+ .meta.json)
See embed_common.py for input, output and resume behaviour.

Usage:
    python pipeline/04_embed/embed_skyclip.py --limit 256 --variant vitb32_top50pct   # quick test
    python pipeline/04_embed/embed_skyclip.py
"""

from embed_common import run

if __name__ == "__main__":
    run("skyclip", "vitl14_top30pct", __doc__)
