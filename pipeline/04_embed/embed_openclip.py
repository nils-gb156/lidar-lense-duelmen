"""CLIP image embeddings of all DOP patches: OpenCLIP (LAION-2B weights), the general-purpose baseline.

Variants (--variant):
    vitb32   ViT-B-32, laion2b_s34b_b79k, 512 dim   (default, the first try)
    vitl14   ViT-L-14, laion2b_s32b_b82k, 768 dim
    vith14   ViT-H-14, laion2b_s32b_b79k, 1024 dim  (same architecture as GeoRSCLIP ViT-H/14)

Output: data/embeddings/dop_image_embeddings_openclip_<variant>.npz (+ .meta.json)
See embed_common.py for input, output and resume behaviour.

Usage:
    python pipeline/04_embed/embed_openclip.py --limit 256     # quick test, writes *_test files
    python pipeline/04_embed/embed_openclip.py
    python pipeline/04_embed/embed_openclip.py --variant vith14
"""

from embed_common import run

if __name__ == "__main__":
    run("openclip", "vitb32", __doc__)
