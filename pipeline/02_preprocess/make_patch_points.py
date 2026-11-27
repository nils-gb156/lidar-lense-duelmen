"""Height above ground (HAG) and point arrays for all 50 m patches (issues #10 and #12).

One pass per LAZ tile:
1. read the whole tile and compute HAG with PDAL filters.hag_nn (ground = class 2)
2. drop noise (class 18) and the 2011 fill data (classes 9 and 26, see docs/results.md #7)
3. assign the points to the 50 m patches of the patch index (only patches inside the boundary)
4. store ALL points of each patch, in random order (OpenShape's ball query takes the
   first points in list order, so the order must not follow the flight lines)

No sampling and no scaling here: drawing 10,000 points and dividing by 25 m happen at
training time (see docs/results.md, OpenShape smoke test).

Per tile three .npy files (memory-mappable, rows grouped by patch):
    <tile>_xyz.npy        float16 (N, 3): x, y relative to the patch centre [m], HAG [m]
    <tile>_intensity.npy  uint16  (N,)
    <tile>_class.npy      uint8   (N,)
plus one index for all patches:
    index.parquet         patch_id, tile, split, start, count, hag_min, hag_max, n_hag_below_minus_0_5
float16 resolution: about 1–3 cm for x, y (|v| <= 25 m), up to 6 cm for HAG above 64 m.

HAG at tile edges uses only ground points of the same tile (no buffer from the
neighbour tile); with ground points every few dm this has little effect.

Input:  data/raw/laz/*.laz
        data/patches/patch_index.parquet
        pipeline/01_download/tiles_duelmen.csv
Output: data/processed/patch_points/  (tile arrays, index.parquet, meta.json)

Usage:
    python pipeline/02_preprocess/make_patch_points.py
    python pipeline/02_preprocess/make_patch_points.py --tiles 32_378_5749 32_381_5743
    python pipeline/02_preprocess/make_patch_points.py --workers 3 --limit 10
    python pipeline/02_preprocess/make_patch_points.py --force
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TILES_CSV = ROOT / "pipeline" / "01_download" / "tiles_duelmen.csv"
LAZ_DIR = ROOT / "data" / "raw" / "laz"
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
OUT_DIR = ROOT / "data" / "processed" / "patch_points"

PATCH_SIZE = 50
TILE_SIZE = 1000
DROP_CLASSES = (18, 9, 26)
SEED = 0


def tile_paths(tile_id: str) -> dict[str, Path]:
    return {
        "xyz": OUT_DIR / f"{tile_id}_xyz.npy",
        "intensity": OUT_DIR / f"{tile_id}_intensity.npy",
        "class": OUT_DIR / f"{tile_id}_class.npy",
        "index": OUT_DIR / "tiles" / f"{tile_id}.parquet",
    }


def process_tile(tile_id: str, laz_path: str, easting_km: int, northing_km: int, patch_ids: list[str]) -> dict:
    import pdal  # imported in the worker process

    t0 = time.time()
    pipe = pdal.Pipeline(
        json.dumps([{"type": "readers.las", "filename": laz_path}, {"type": "filters.hag_nn"}])
    )
    pipe.execute()
    a = pipe.arrays[0]
    n_read = len(a)

    cls = a["Classification"].astype(np.uint8)
    a = a[~np.isin(cls, DROP_CLASSES)]

    x0, y0 = easting_km * 1000, northing_km * 1000
    npatch = TILE_SIZE // PATCH_SIZE
    col = np.clip(np.floor((a["X"] - x0) / PATCH_SIZE).astype(np.int64), 0, npatch - 1)
    row = np.clip(np.floor((a["Y"] - y0) / PATCH_SIZE).astype(np.int64), 0, npatch - 1)
    cell = row * npatch + col

    # cells of the patches inside the boundary (patch_id = <tile>_<col>_<row>)
    wanted = np.zeros(npatch * npatch, dtype=bool)
    cell_of = {}
    for pid in patch_ids:
        c, r = (int(v) for v in pid.rsplit("_", 2)[1:])
        wanted[r * npatch + c] = True
        cell_of[r * npatch + c] = pid
    keep = wanted[cell]
    a, cell, col, row = a[keep], cell[keep], col[keep], row[keep]

    # group by patch, random order inside each patch
    rng = np.random.default_rng([SEED, easting_km, northing_km])
    order = np.lexsort((rng.random(len(a)), cell))
    a, cell, col, row = a[order], cell[order], col[order], row[order]

    cx = x0 + (col + 0.5) * PATCH_SIZE
    cy = y0 + (row + 0.5) * PATCH_SIZE
    hag = a["HeightAboveGround"]
    xyz = np.stack([a["X"] - cx, a["Y"] - cy, hag], axis=1).astype(np.float16)

    cells, start, count = np.unique(cell, return_index=True, return_counts=True)
    hag_min = np.minimum.reduceat(hag, start) if len(start) else np.array([])
    hag_max = np.maximum.reduceat(hag, start) if len(start) else np.array([])
    neg = np.add.reduceat((hag < -0.5).astype(np.int64), start) if len(start) else np.array([])
    idx = pd.DataFrame(
        {
            "patch_id": [cell_of[c] for c in cells],
            "tile": tile_id,
            "start": start.astype(np.int64),
            "count": count.astype(np.int64),
            "hag_min": np.round(hag_min, 2),
            "hag_max": np.round(hag_max, 2),
            "n_hag_below_minus_0_5": neg.astype(np.int64),
        }
    )
    missing = sorted(set(patch_ids) - set(idx["patch_id"]))
    if missing:  # patches without any point keep an entry with count 0
        idx = pd.concat(
            [idx, pd.DataFrame({"patch_id": missing, "tile": tile_id, "start": len(a), "count": 0})],
            ignore_index=True,
        )

    p = tile_paths(tile_id)
    p["index"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["xyz"], xyz)
    np.save(p["intensity"], a["Intensity"].astype(np.uint16))
    np.save(p["class"], a["Classification"].astype(np.uint8))
    idx.to_parquet(p["index"], index=False)  # written last: marks the tile as done
    return {
        "tile": tile_id,
        "n_read": int(n_read),
        "n_kept": int(len(a)),
        "n_patches": int((idx["count"] > 0).sum()),
        "n_empty": len(missing),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3, help="parallel tiles (default 3, about 2–3 GB RAM each)")
    ap.add_argument("--tiles", nargs="+", default=None, help="only these tile ids")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tiles")
    ap.add_argument("--force", action="store_true", help="recompute existing tiles")
    args = ap.parse_args()

    tiles = pd.read_csv(TILES_CSV)
    patches = pd.read_parquet(PATCH_INDEX, columns=["patch_id", "tile", "split"])
    if args.tiles:
        tiles = tiles[tiles["tile_id"].isin(args.tiles)]
    if args.limit:
        tiles = tiles.head(args.limit)
    by_tile = patches.groupby("tile")["patch_id"].apply(list).to_dict()

    todo = []
    for t in tiles.itertuples(index=False):
        if t.tile_id not in by_tile:
            continue
        if tile_paths(t.tile_id)["index"].exists() and not args.force:
            continue
        laz = LAZ_DIR / t.laz_file
        if not laz.exists():
            print(f"missing {laz.name}, skipped")
            continue
        todo.append((t.tile_id, str(laz), int(t.easting_km), int(t.northing_km), by_tile[t.tile_id]))
    print(f"{len(tiles)} tiles selected, {len(todo)} to process with {args.workers} workers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_tile, *a): a[0] for a in todo}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    r = f.result()
                except Exception as e:
                    print(f"[{i}/{len(todo)}] {futs[f]}: ERROR {e!r}", flush=True)
                    continue
                el = time.time() - t_start
                print(
                    f"[{i}/{len(todo)}] {r['tile']}: {r['n_kept']:,} of {r['n_read']:,} points, "
                    f"{r['n_patches']} patches ({r['n_empty']} empty), {r['seconds']} s "
                    f"| eta {el / i * (len(todo) - i) / 60:.0f} min",
                    flush=True,
                )

    # global index over all finished tiles
    parts = sorted((OUT_DIR / "tiles").glob("*.parquet"))
    if not parts:
        sys.exit("no tiles processed")
    index = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    index = index.merge(patches[["patch_id", "split"]], on="patch_id", how="left")
    index.to_parquet(OUT_DIR / "index.parquet", index=False)

    meta = {
        "created": date.today().isoformat(),
        "script": "pipeline/02_preprocess/make_patch_points.py",
        "data_source": "3D-Messdaten NW (Geobasis NRW), LAZ, dl-de/zero-2-0",
        "acquisition_year": 2024,
        "crs": "EPSG:25832 + DHHN2016 (EPSG:7837)",
        "patch_size_m": PATCH_SIZE,
        "hag": "PDAL filters.hag_nn on the whole tile, ground = class 2",
        "dropped_classes": list(DROP_CLASSES),
        "xyz": "float16; x, y relative to the patch centre in m; z = HAG in m; not scaled",
        "point_order": f"random within each patch (seed {SEED})",
        "sampling": "none (all points); sample to 10,000 and scale by 25 m at training time",
        "files": "<tile>_xyz.npy, <tile>_intensity.npy, <tile>_class.npy; rows of a patch = [start, start + count)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = index[index["count"] > 0]
    print(f"\nindex: {len(index):,} patches in {len(parts)} tiles ({(index['count'] == 0).sum()} empty) -> {OUT_DIR / 'index.parquet'}")
    print(f"points per patch: median {done['count'].median():,.0f}, min {done['count'].min():,}, < 10,000: {(done['count'] < 10_000).sum()}")
    print(f"HAG: min {done['hag_min'].min():.1f} m, max {done['hag_max'].max():.1f} m, patches with points < -0.5 m: {(done['n_hag_below_minus_0_5'] > 0).sum()}")


if __name__ == "__main__":
    main()
