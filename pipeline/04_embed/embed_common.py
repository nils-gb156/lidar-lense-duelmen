"""Shared embedding run for the per-backbone scripts (embed_openclip.py, embed_georsclip.py, ...).

Encodes every crop in data/patches/images/ with the frozen image encoder of one
backbone and stores the L2-normalised embeddings in the order of patch_index.parquet.

Input:  data/patches/patch_index.parquet
        data/patches/images/<tile>/<patch_id>.jpg
        data/patches/images/meta.json
Output: data/embeddings/dop_image_embeddings_<backbone>.npz       ids (patch_id), emb (N x D, float32)
        data/embeddings/dop_image_embeddings_<backbone>.meta.json model, weights, patch size, data source, ...

Re-runnable and resumable: finished blocks of 4096 patches are kept in
data/embeddings/.partial_<backbone>/ so an interrupted run continues where it stopped.
An existing final output is skipped unless --force is given.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import clip_backbones

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = clip_backbones.ROOT
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
IMG_DIR = ROOT / "data" / "patches" / "images"
IMG_META = IMG_DIR / "meta.json"
OUT_DIR = ROOT / "data" / "embeddings"

PATCH_SIZE_M = 50
BLOCK = 4096


class CropDataset(Dataset):
    def __init__(self, tiles, patch_ids, preprocess):
        self.paths = [IMG_DIR / t / f"{p}.jpg" for t, p in zip(tiles, patch_ids)]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.preprocess(img.convert("RGB"))


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def run(family: str, default_variant: str, description: str) -> None:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=default_variant, choices=clip_backbones.variants(family),
                    help=f"model variant (default {default_variant})")
    ap.add_argument("--batch-size", type=int, help="images per batch (default depends on the model size)")
    ap.add_argument("--workers", type=int, default=4, help="image loading processes (default 4)")
    ap.add_argument("--limit", type=int, help="only embed the first N patches (writes *_test files)")
    ap.add_argument("--force", action="store_true", help="overwrite existing embeddings")
    args = ap.parse_args()

    if not PATCH_INDEX.exists() or not IMG_META.exists():
        sys.exit("patch index or crops missing, run make_patch_grid.py and extract_dop_patches.py first")
    img_meta = json.loads(IMG_META.read_text())

    key = f"{family}_{args.variant}"
    suffix = "_test" if args.limit else ""
    out_npz = OUT_DIR / f"dop_image_embeddings_{key}{suffix}.npz"
    out_meta = OUT_DIR / f"dop_image_embeddings_{key}{suffix}.meta.json"
    part_dir = OUT_DIR / f".partial_{key}{suffix}"
    if out_npz.exists() and not args.force:
        print(f"{out_npz} exists, skipping (use --force to rebuild)")
        return

    patches = gpd.read_parquet(PATCH_INDEX)
    if args.limit:
        patches = patches.iloc[: args.limit]
    ids = patches["patch_id"].to_numpy().astype(str)
    n = len(ids)
    ids_md5 = hashlib.md5("\n".join(ids).encode()).hexdigest()

    missing = [p for p in (IMG_DIR / t / f"{i}.jpg" for t, i in zip(patches["tile"], ids)) if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} crop(s) missing, e.g. {missing[0]}; run extract_dop_patches.py")

    state = {"ids_md5": ids_md5, "n": n, "block": BLOCK, "backbone": key}
    state_file = part_dir / "state.json"
    if part_dir.exists() and (args.force or not state_file.exists() or json.loads(state_file.read_text()) != state):
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))

    n_blocks = -(-n // BLOCK)
    bounds = [(b * BLOCK, min((b + 1) * BLOCK, n)) for b in range(n_blocks)]
    todo = [b for b in range(n_blocks) if not (part_dir / f"block_{b:04d}.npy").exists()]
    print(f"{key}: {n} patches, {n_blocks - len(todo)}/{n_blocks} blocks already done")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if todo:
        bb = clip_backbones.load(family, args.variant, "image", device)
        print(f"{bb.info['clip_arch']} / {bb.info['clip_pretrained']} on {device}")
        batch_size = args.batch_size or clip_backbones.default_batch_size(args.variant)
        indices = [i for b in todo for i in range(*bounds[b])]
        loader = DataLoader(
            Subset(CropDataset(patches["tile"], ids, bb.preprocess), indices),
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=device == "cuda",
            persistent_workers=args.workers > 0,
        )

        buf, buf_n, pending, processed = [], 0, iter(todo), 0
        cur = next(pending, None)
        for k, batch in enumerate(loader, 1):
            feats = bb.encode_image(batch).numpy()
            buf.append(feats)
            buf_n += len(feats)
            processed += len(feats)
            while cur is not None and buf_n >= bounds[cur][1] - bounds[cur][0]:
                size = bounds[cur][1] - bounds[cur][0]
                arr = np.concatenate(buf)
                _save_npy(part_dir / f"block_{cur:04d}.npy", arr[:size])
                buf, buf_n = ([arr[size:]] if len(arr) > size else []), len(arr) - size
                cur = next(pending, None)
            if k % 20 == 0 or processed == len(indices):
                print(f"{processed}/{len(indices)} ({processed / (time.time() - t0):.1f} img/s)")
        info = bb.info
    else:
        # all blocks exist already; only the metadata is needed
        info = clip_backbones.load(family, args.variant, "text", "cpu").info

    emb = np.concatenate([np.load(part_dir / f"block_{b:04d}.npy") for b in range(n_blocks)]).astype(np.float32)
    if len(emb) != n:
        sys.exit(f"expected {n} embeddings, got {len(emb)}; rerun with --force")
    meta = {
        "backbone": key,
        "family": family,
        "variant": args.variant,
        **info,
        "embedding_dim": int(emb.shape[1]),
        "normalized": "L2",
        "n_patches": int(n),
        "patch_size_m": PATCH_SIZE_M,
        "image_size_px": img_meta["size"],
        "data_source": img_meta["source"],
        "dop_years": img_meta["dop_years"],
        "crs": img_meta["crs"],
        "ids_md5": ids_md5,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(out_npz.stem + ".part.npz")
    np.savez(tmp, ids=ids, emb=emb)
    os.replace(tmp, out_npz)
    out_meta.write_text(json.dumps(meta, indent=2))
    shutil.rmtree(part_dir)
    print(f"saved {out_npz} {emb.shape}, {time.time() - t0:.0f} s")
