"""Try out text search on the DOP image embeddings of one backbone.

Encodes a text query with the same backbone the image embeddings were made with
(family and variant are read from the embedding metadata), ranks all patches by
cosine similarity and saves a contact sheet of the top hits.

Input:  data/embeddings/dop_image_embeddings_<backbone>.npz (+ .meta.json)  from embed_<family>.py
        data/patches/patch_index.parquet, data/patches/thumbs/
Output: data/embeddings/search/<backbone>/<query>.png    contact sheet of the top-k thumbnails

Usage:
    python pipeline/04_embed/search_dop.py "wind turbine"
    python pipeline/04_embed/search_dop.py "solar panels" "forest" -k 24 --backbone georsclip_vith14
    python pipeline/04_embed/search_dop.py "church" --emb data/embeddings/dop_image_embeddings_openclip_vitb32_test.npz

dofaclip embeddings can only be searched in the separate DOFA-CLIP environment (see embed_dofaclip.py).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import clip_backbones

import geopandas as gpd
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = clip_backbones.ROOT
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
THUMB_DIR = ROOT / "data" / "patches" / "thumbs"
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = EMB_DIR / "search"

PROMPT_TEMPLATE = "an aerial photo of {}"


def contact_sheet(rows, cols: int, thumb: int, path: Path) -> None:
    label_h = 14
    n_rows = -(-len(rows) // cols)
    sheet = Image.new("RGB", (cols * thumb, n_rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (tile, pid, score) in enumerate(rows):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
        with Image.open(THUMB_DIR / tile / f"{pid}.jpg") as im:
            sheet.paste(im.convert("RGB").resize((thumb, thumb)), (x, y + label_h))
        draw.text((x + 2, y + 1), f"{score:.3f} {pid[-11:]}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="+", help="text queries, e.g. 'wind turbine'")
    ap.add_argument("-k", type=int, default=12, help="number of hits (default 12)")
    ap.add_argument("--backbone", default="openclip_vitb32",
                    help="embedding set <family>_<variant> (default openclip_vitb32)")
    ap.add_argument("--emb", type=Path, help="embeddings file, overrides --backbone")
    ap.add_argument("--no-template", action="store_true", help=f"use the query as is, not '{PROMPT_TEMPLATE}'")
    args = ap.parse_args()

    emb_path = args.emb or EMB_DIR / f"dop_image_embeddings_{args.backbone}.npz"
    meta_path = emb_path.with_suffix(".meta.json")
    if not emb_path.exists() or not meta_path.exists():
        sys.exit(f"{emb_path} not found, run the matching embed_<family>.py first")
    meta = json.loads(meta_path.read_text())
    data = np.load(emb_path)
    ids, emb = data["ids"], torch.from_numpy(data["emb"])

    patches = gpd.read_parquet(PATCH_INDEX).set_index("patch_id").loc[ids]
    tiles = patches["tile"].to_numpy()
    centers = patches.geometry.centroid

    # text and image embeddings must come from exactly the same model
    bb = clip_backbones.load(meta["family"], meta["variant"], "text", "cpu")
    if bb.info["clip_pretrained"] != meta["clip_pretrained"]:
        sys.exit(f"embeddings were made with {meta['clip_pretrained']}, text model is {bb.info['clip_pretrained']}")

    texts = [q if args.no_template else PROMPT_TEMPLATE.format(q) for q in args.queries]
    # cosine similarity, both sides are L2-normalised; done in torch because numpy's MKL and
    # torch's OpenMP runtime clash on this conda environment
    scores = (bb.encode_text(texts) @ emb.T).numpy()

    for q, text, s in zip(args.queries, texts, scores):
        top = np.argsort(-s)[: args.k]
        print(f"\n{q!r}  (prompt: {text!r}, backbone: {meta['backbone']})")
        for rank, i in enumerate(top, 1):
            c = centers.iloc[i]
            print(f"{rank:3d}  {s[i]:.3f}  {ids[i]}  E {c.x:.0f} N {c.y:.0f}")
        slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
        out = OUT_DIR / meta["backbone"] / f"{slug}.png"
        contact_sheet([(tiles[i], ids[i], s[i]) for i in top], cols=4, thumb=128, path=out)
        print(f"contact sheet: {out}")


if __name__ == "__main__":
    main()
