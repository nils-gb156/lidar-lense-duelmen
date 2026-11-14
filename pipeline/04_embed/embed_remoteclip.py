"""CLIP image embeddings of all DOP patches: RemoteCLIP (CLIP fine-tuned on remote-sensing image-text data).

Weights: huggingface.co/chendelong/RemoteCLIP, RemoteCLIP-<arch>.pt, loaded into an open_clip model.
Variants (--variant):
    vitl14   ViT-L-14, 768 dim, 1.7 GB download (default)
    vitb32   ViT-B-32, 512 dim, 0.6 GB download

Output: data/embeddings/dop_image_embeddings_remoteclip_<variant>.npz (+ .meta.json)
See embed_common.py for input, output and resume behaviour.

Usage:
    python pipeline/04_embed/embed_remoteclip.py --limit 256 --variant vitb32   # quick test
    python pipeline/04_embed/embed_remoteclip.py
"""

from embed_common import run

if __name__ == "__main__":
    run("remoteclip", "vitl14", __doc__)
