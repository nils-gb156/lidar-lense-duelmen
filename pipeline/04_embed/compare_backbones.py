"""Compare the DOP image embeddings of all backbones on a list of text queries.

For every query one picture is written: one row per backbone, each row shows the top-10
patches (most similar first) with the cosine similarity below every image. The backbone
name is on the left of its row.

Every backbone encodes the query with its own text encoder (family and variant come from
the embedding metadata). The text embeddings are cached in
data/embeddings/text_cache/<backbone>.json, so the text models are only needed once.

Input:  data/embeddings/dop_image_embeddings_<backbone>.npz (+ .meta.json)   from embed_<family>.py
        data/patches/patch_index.parquet, data/patches/thumbs/ (or data/patches/images/)
Output: data/embeddings/compare/<nn>_<query>.png

Usage:
    python pipeline/04_embed/compare_backbones.py
    python pipeline/04_embed/compare_backbones.py --queries "church" "forest"
    python pipeline/04_embed/compare_backbones.py --only openclip_vitb32 georsclip_vith14
    python pipeline/04_embed/compare_backbones.py --text-cache-only --only dofaclip_vitb16

dofaclip needs the DOFA-CLIP fork of open_clip (see embed_dofaclip.py). Run it once in the
dofaclip environment with --text-cache-only --only dofaclip_vitb16; afterwards the normal
run in pcsearch reads its cached text embeddings and needs no fork.

Text models that are not cached yet are downloaded or loaded on the first run (several GB for
the large models); run it on the machine that already has the weights, then the small
text_cache files can be copied anywhere.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import clip_backbones

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = clip_backbones.ROOT
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
THUMB_DIR = ROOT / "data" / "patches" / "thumbs"
IMG_DIR = ROOT / "data" / "patches" / "images"
EMB_DIR = ROOT / "data" / "embeddings"
CACHE_DIR = EMB_DIR / "text_cache"
OUT_DIR = EMB_DIR / "compare"

PROMPT_TEMPLATE = "an aerial photo of {}"
QUERIES = [
    # area-covering, visible from above
    "forest",
    "farmland",
    "lake",
    "parking lot",
    "solar park",
    "cemetery",
    "allotment gardens",
    "construction site",
    # objects with a clear shape from above
    "roundabout",
    "tennis court",
    "football field",
    "swimming pool",
    "playground",
    "greenhouse",
    "railway tracks",
    "industrial building",
    # defined by height: weak in the DOP, relevant for the lidar model later
    "power line pylon",
    "wind turbine",
    "church",
    # descriptions without a fixed class
    "farm with silos",
    "house with solar panels on the roof",
    "large flat roof",
    "biogas plant",
    # synonyms of queries above: is the ranking robust to the wording?
    "graveyard",
    "soccer pitch",
]

TOP_K = 10
CELL = 160
LABEL_W = 210
TITLE_H = 32
SCORE_H = 16


def find_backbones(only: list[str] | None) -> list[tuple[str, dict, Path]]:
    found = []
    for meta_path in sorted(EMB_DIR.glob("dop_image_embeddings_*.meta.json")):
        if meta_path.name.endswith("_test.meta.json"):
            continue
        npz = meta_path.with_name(meta_path.name.replace(".meta.json", ".npz"))
        if not npz.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if only and meta["backbone"] not in only:
            continue
        found.append((meta["backbone"], meta, npz))
    if only:
        unknown = set(only) - {k for k, _, _ in found}
        if unknown:
            sys.exit(f"no embeddings found for: {sorted(unknown)}")
    return found


def text_embeddings(key: str, meta: dict, texts: list[str]) -> torch.Tensor:
    cache_path = CACHE_DIR / f"{key}.json"
    cache = {"clip_pretrained": meta["clip_pretrained"], "texts": {}}
    if cache_path.exists():
        old = json.loads(cache_path.read_text())
        if old.get("clip_pretrained") == meta["clip_pretrained"]:
            cache = old
    missing = [t for t in texts if t not in cache["texts"]]
    if missing:
        # text and image embeddings must come from exactly the same model
        bb = clip_backbones.load(meta["family"], meta["variant"], "text", "cpu")
        if bb.info["clip_pretrained"] != meta["clip_pretrained"]:
            raise RuntimeError(f"embeddings made with {meta['clip_pretrained']}, text model is {bb.info['clip_pretrained']}")
        for t, v in zip(missing, bb.encode_text(missing)):
            cache["texts"][t] = v.tolist()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return torch.tensor([cache["texts"][t] for t in texts], dtype=torch.float32)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def cell_image(tile: str, patch_id: str) -> Image.Image:
    for folder in (THUMB_DIR, IMG_DIR):
        path = folder / tile / f"{patch_id}.jpg"
        if path.exists():
            with Image.open(path) as im:
                return im.convert("RGB").resize((CELL, CELL))
    return Image.new("RGB", (CELL, CELL), (200, 200, 200))


def draw_query(path: Path, query: str, prompt: str, results: dict, tiles: pd.Series) -> None:
    rows = list(results.items())
    width = LABEL_W + TOP_K * CELL
    height = TITLE_H + len(rows) * (CELL + SCORE_H)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    f_title, f_label, f_score = font(18), font(14), font(12)
    draw.text((8, 6), f'"{query}"   (prompt: "{prompt}")', fill="black", font=f_title)
    for r, (key, (ids, scores)) in enumerate(rows):
        y = TITLE_H + r * (CELL + SCORE_H)
        draw.text((8, y + CELL // 2 - 8), key, fill="black", font=f_label)
        for c, (pid, score) in enumerate(zip(ids, scores)):
            x = LABEL_W + c * CELL
            sheet.paste(cell_image(tiles[pid], pid), (x, y))
            draw.text((x + 4, y + CELL + 1), f"{score:.3f}", fill="black", font=f_score)
        draw.line([(0, y + CELL + SCORE_H - 1), (width, y + CELL + SCORE_H - 1)], fill=(180, 180, 180))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", default=QUERIES, help="text queries (default: the list in this script)")
    ap.add_argument("--only", nargs="+", help="only these backbones, e.g. openclip_vitb32 georsclip_vith14")
    ap.add_argument("--no-template", action="store_true", help=f"use the queries as is, not '{PROMPT_TEMPLATE}'")
    ap.add_argument("--text-cache-only", action="store_true", help="only fill the text embedding cache, no pictures")
    args = ap.parse_args()

    backbones = find_backbones(args.only)
    if not backbones:
        sys.exit(f"no embeddings found in {EMB_DIR}")
    prompts = [q if args.no_template else PROMPT_TEMPLATE.format(q) for q in args.queries]

    results: dict[str, dict] = {}  # backbone -> {"ids": (n_queries, K), "scores": (n_queries, K)}
    cached, skipped = [], []
    ids_md5 = None
    for key, meta, npz in backbones:
        print(f"{key}: encoding {len(prompts)} queries")
        try:
            txt = text_embeddings(key, meta, prompts)
        except Exception as e:
            print(f"  SKIPPED {key}: {e}")
            skipped.append(key)
            continue
        cached.append(key)
        if args.text_cache_only:
            continue
        data = np.load(npz)
        md5 = hashlib.md5("\n".join(data["ids"]).encode()).hexdigest()
        if ids_md5 not in (None, md5):
            sys.exit(f"{key} was made for a different patch index than the other backbones")
        ids_md5 = md5
        emb = torch.from_numpy(data["emb"])
        if emb.shape[1] != txt.shape[1]:
            print(f"  SKIPPED {key}: text dim {txt.shape[1]} != image dim {emb.shape[1]}")
            continue
        # cosine similarity (both sides L2-normalised), in torch because numpy's MKL and
        # torch's OpenMP runtime clash on this conda environment
        top = (txt @ emb.T).topk(TOP_K, dim=1)
        results[key] = (data["ids"][top.indices.numpy()], top.values.numpy())
        del emb, data

    if args.text_cache_only:
        print(f"text embeddings cached for {cached or 'no backbone'} in {CACHE_DIR}")
        if skipped:
            sys.exit(f"failed: {skipped}")
        return
    if not results:
        sys.exit("no backbone could be evaluated")

    tiles = pd.read_parquet(PATCH_INDEX, columns=["patch_id", "tile"]).set_index("patch_id")["tile"]
    for i, (query, prompt) in enumerate(zip(args.queries, prompts)):
        slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
        out = OUT_DIR / f"{i:02d}_{slug}.png"
        per_query = {key: (ids[i], scores[i]) for key, (ids, scores) in results.items()}
        draw_query(out, query, prompt, per_query, tiles)
        print(f"wrote {out}")
    print(f"{len(results)} backbones: {', '.join(results)}")


if __name__ == "__main__":
    main()
