"""CLIP image embeddings of all DOP patches: GeoRSCLIP (RS5M, remote-sensing fine-tuned CLIP).

Weights: huggingface.co/Zilun/GeoRSCLIP, ckpt/RS5M_<variant>.pt, loaded into an open_clip model.
Variants (--variant):
    vith14   ViT-H-14, 1024 dim, 3.9 GB download (default)
    vitb32   ViT-B-32, 512 dim, 0.6 GB download

ViT-H/14 is about 40x the compute of ViT-B/32 per image. The 2 GB local GPU can hold the
image tower in fp16 but will be slow; Colab is a better place for it. The run is resumable.

Output: data/embeddings/dop_image_embeddings_georsclip_<variant>.npz (+ .meta.json)
See embed_common.py for input, output and resume behaviour.

Usage:
    python pipeline/04_embed/embed_georsclip.py --limit 256 --variant vitb32   # quick test
    python pipeline/04_embed/embed_georsclip.py
    python pipeline/04_embed/embed_georsclip.py --variant vitb32
"""

from embed_common import run

if __name__ == "__main__":
    run("georsclip", "vith14", __doc__)
