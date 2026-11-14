"""CLIP image embeddings of all DOP patches: DOFA-CLIP ViT-B/16 (GeoLangBind, SigLIP-based, wavelength-aware).

Weights: huggingface.co/earthflow/GeoLB-ViT-B-16-SigLIP-All-EO (the official model, not the
BiliSakura port, whose config maps this SigLIP model onto the Hugging Face CLIP classes).
The image tower takes the band wavelengths as input; the RGB crops use R 0.665, G 0.560, B 0.490 um
as in the DOFA-CLIP demo. Embedding dimension 768.

This model needs the DOFA-CLIP fork of open_clip, which replaces the normal open_clip.
Do NOT install it into `pcsearch`. Use a separate environment (or Colab):

    conda create -n dofaclip --clone pcsearch      # or any environment with torch, geopandas, pillow
    conda activate dofaclip
    git clone https://github.com/xiong-zhitong/DOFA-CLIP
    pip install -e DOFA-CLIP/open_clip transformers sentencepiece

Output: data/embeddings/dop_image_embeddings_dofaclip_vitb16.npz (+ .meta.json)
See embed_common.py for input, output and resume behaviour.

Usage:
    python pipeline/04_embed/embed_dofaclip.py --limit 256     # quick test, writes *_test files
    python pipeline/04_embed/embed_dofaclip.py
"""

from embed_common import run

if __name__ == "__main__":
    run("dofaclip", "vitb16", __doc__)
