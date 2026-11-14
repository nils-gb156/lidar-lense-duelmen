"""Crop the DOP orthophoto (RGB only) for every patch in the patch index.

Each DOP tile is read once and cut with exact pixel windows, so a crop is
aligned with the patch geometry (the 50 m patch is 500 x 500 px at 0.1 m).
Every crop is resized to the CLIP input size and saved together with a small
thumbnail for the web app.

Input:  data/patches/patch_index.parquet   (from make_patch_grid.py)
        data/raw/dop/*.jp2                  (bands R, G, B, NIR; NIR is dropped)
Output: data/patches/images/<tile>/<patch_id>.jpg   CLIP input crops
        data/patches/thumbs/<tile>/<patch_id>.jpg   web app thumbnails
        data/patches/images/meta.json               settings used

Re-runnable: existing crops are skipped. Changing --size or --thumb-size
requires --force, which rewrites all files.

Usage:
    python pipeline/02_preprocess/extract_dop_patches.py --limit 1   # quick test
    python pipeline/02_preprocess/extract_dop_patches.py
    python pipeline/02_preprocess/extract_dop_patches.py --size 224 --workers 3
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
DOP_DIR = ROOT / "data" / "raw" / "dop"
IMG_DIR = ROOT / "data" / "patches" / "images"
THUMB_DIR = ROOT / "data" / "patches" / "thumbs"
META = IMG_DIR / "meta.json"

PIXEL_TOL = 1e-6

# CLIP model the crops are made for (ViT-B-32 takes 224 x 224 px input)
CLIP_META = {
    "clip_library": "open_clip", # The Python library that loads the model (open re-implementation of CLIP)
    "clip_arch": "ViT-B-32", # Model architecture: Vision Transformer, “Base” size, the image is cut into 32 × 32 px pieces
    "clip_pretrained": "laion2b_s34b_b79k", # Which trained weights: trained on LAION-2B (2 billion image-text pairs), 34 billion samples seen, batch size 79k
    "embedding_dim": 512, # Length of each vector that CLIP outputs (image and text)
    "prompt_template": "an aerial photo of {}", # Sentence the search word is put into before the text encoder; {} is replaced by e.g. “solar park”
}


def save_jpeg(img: Image.Image, path: Path, quality: int) -> None:
    tmp = path.with_name(path.name + ".part")
    img.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, path)


def process_tile(job: dict) -> tuple[str, int, int, list[str]]:
    """Cut all missing patches of one DOP tile. Returns (tile, written, skipped, errors)."""
    tile = job["tile"]
    img_dir = IMG_DIR / tile
    thumb_dir = THUMB_DIR / tile
    force = job["force"]

    todo = []
    for pid, bounds in job["patches"]:
        img_path, thumb_path = img_dir / f"{pid}.jpg", thumb_dir / f"{pid}.jpg"
        if force or not (img_path.exists() and thumb_path.exists()):
            todo.append((pid, bounds, img_path, thumb_path))
    skipped = len(job["patches"]) - len(todo)
    if not todo:
        return tile, 0, skipped, []

    img_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    written = 0
    with rasterio.open(DOP_DIR / job["dop_file"]) as src:
        if src.count < 3:
            return tile, 0, skipped, [f"{job['dop_file']}: only {src.count} band(s)"]
        rgb = src.read([1, 2, 3])  # (3, H, W) uint8; band 4 is NIR
        t = src.transform
        height, width = src.height, src.width

        for pid, (minx, miny, maxx, maxy), img_path, thumb_path in todo:
            col_f = (minx - t.c) / t.a
            row_f = (t.f - maxy) / -t.e
            w_f = (maxx - minx) / t.a
            h_f = (maxy - miny) / -t.e
            col, row, w, h = round(col_f), round(row_f), round(w_f), round(h_f)
            if max(abs(col_f - col), abs(row_f - row), abs(w_f - w), abs(h_f - h)) > PIXEL_TOL:
                errors.append(f"{pid}: patch is not aligned with the DOP pixel grid")
                continue
            if col < 0 or row < 0 or col + w > width or row + h > height:
                errors.append(f"{pid}: patch lies outside {job['dop_file']}")
                continue

            crop = np.ascontiguousarray(rgb[:, row : row + h, col : col + w].transpose(1, 2, 0))
            img = Image.fromarray(crop, mode="RGB").resize((job["size"], job["size"]), Image.BICUBIC)
            thumb = img.resize((job["thumb_size"], job["thumb_size"]), Image.BICUBIC)
            save_jpeg(img, img_path, job["quality"])
            save_jpeg(thumb, thumb_path, job["quality"])
            written += 1
    return tile, written, skipped, errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=224, help="crop size in px = CLIP input size (default 224)")
    ap.add_argument("--thumb-size", type=int, default=128, help="thumbnail size in px (default 128)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--workers", type=int, default=2, help="parallel tiles (default 2, ~0.3 GB RAM each)")
    ap.add_argument("--limit", type=int, help="only process the first N tiles")
    ap.add_argument("--force", action="store_true", help="rewrite existing files")
    args = ap.parse_args()

    if not PATCH_INDEX.exists():
        sys.exit(f"{PATCH_INDEX} not found, run make_patch_grid.py first")
    patches = gpd.read_parquet(PATCH_INDEX)

    meta = {
        "size": args.size,
        "thumb_size": args.thumb_size,
        "jpeg_quality": args.quality,
        "bands": "R,G,B from DOP10 RGBI (NIR dropped)",
        "resampling": "bicubic",
        "source": "DOP10 RGBI, Geobasis NRW / OpenGeodata NRW",
        "dop_years": sorted(int(y) for y in patches["dop_year"].unique()),
        "crs": str(patches.crs),
        "layout": "<tile>/<patch_id>.jpg",
        **CLIP_META,
    }
    if META.exists() and not args.force:
        old = json.loads(META.read_text())
        if (old["size"], old["thumb_size"], old["jpeg_quality"]) != (args.size, args.thumb_size, args.quality):
            sys.exit(f"{META} was written with other settings, use --force to rewrite all crops")

    jobs = []
    for (tile, dop_file), group in patches.groupby(["tile", "dop_file"], sort=True):
        jobs.append(
            {
                "tile": tile,
                "dop_file": dop_file,
                "patches": [(pid, geom.bounds) for pid, geom in zip(group["patch_id"], group.geometry)],
                "size": args.size,
                "thumb_size": args.thumb_size,
                "quality": args.quality,
                "force": args.force,
            }
        )
    if args.limit:
        jobs = jobs[: args.limit]

    missing = [j["dop_file"] for j in jobs if not (DOP_DIR / j["dop_file"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} DOP file(s) missing, e.g. {missing[0]}; run download_tiles.py dop")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=2))

    total_written = total_skipped = 0
    all_errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_tile, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            tile, written, skipped, errors = fut.result()
            total_written += written
            total_skipped += skipped
            all_errors += errors
            print(f"[{i}/{len(jobs)}] {tile}: {written} written, {skipped} skipped, {len(errors)} errors")

    print(f"done: {total_written} written, {total_skipped} skipped, {len(all_errors)} errors")
    for e in all_errors[:20]:
        print("  ERROR", e)
    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
