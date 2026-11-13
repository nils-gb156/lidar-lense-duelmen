"""Build the 50 m x 50 m patch grid for Duelmen and save it as GeoParquet.

The grid is anchored at the lower-left corner of each 1 km tile of the NRW tile
grid, so every patch lies completely inside exactly one tile (and one DOP tile).
Only patches whose area lies at least --min-overlap inside the municipal boundary
are kept.

The point count (n_points) is not known yet because the LAZ files are processed
later; the column is created as null and filled by a later step. Patches with too
few points are dropped in that step.

Input:  pipeline/01_download/tiles_duelmen.csv
        data/raw/boundary/duelmen_boundary.gpkg   (EPSG:25832)
Output: data/patches/patch_index.parquet          (EPSG:25832)

Usage:
    python pipeline/02_preprocess/make_patch_grid.py
    python pipeline/02_preprocess/make_patch_grid.py --force
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box

ROOT = Path(__file__).resolve().parents[2]
TILES_CSV = ROOT / "pipeline" / "01_download" / "tiles_duelmen.csv"
BOUNDARY = ROOT / "data" / "raw" / "boundary" / "duelmen_boundary.gpkg"
OUT = ROOT / "data" / "patches" / "patch_index.parquet"

CRS = "EPSG:25832"
TILE_SIZE_M = 1000


def build_grid(tiles: pd.DataFrame, patch_size: int) -> gpd.GeoDataFrame:
    if TILE_SIZE_M % patch_size != 0:
        sys.exit(f"patch size {patch_size} m does not divide the {TILE_SIZE_M} m tile size")
    n = TILE_SIZE_M // patch_size
    idx = np.arange(n)
    col, row = np.meshgrid(idx, idx)
    col, row = col.ravel(), row.ravel()

    records = []
    for t in tiles.itertuples(index=False):
        records.append(
            pd.DataFrame(
                {
                    "patch_id": [f"{t.tile_id}_{c:02d}_{r:02d}" for c, r in zip(col, row)],
                    "tile": t.tile_id,
                    "dop_file": t.dop_file,
                    "dop_year": t.dop_year,
                    "minx": t.easting_km * 1000 + col * patch_size,
                    "miny": t.northing_km * 1000 + row * patch_size,
                }
            )
        )
    df = pd.concat(records, ignore_index=True)
    geometry = box(df["minx"], df["miny"], df["minx"] + patch_size, df["miny"] + patch_size)
    return gpd.GeoDataFrame(df, geometry=geometry, crs=CRS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-size", type=int, default=50, help="patch edge length in m (default 50)")
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=0.5,
        help="minimum fraction of a patch inside the boundary (default 0.5)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing patch index")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT} exists, skipping (use --force to rebuild)")
        return

    tiles = pd.read_csv(TILES_CSV)
    boundary = gpd.read_file(BOUNDARY)
    if boundary.crs is None or boundary.crs.to_epsg() != 25832:
        boundary = boundary.to_crs(CRS)
    boundary_geom = boundary.geometry.union_all()

    grid = build_grid(tiles, args.patch_size)
    print(f"candidate patches: {len(grid)} ({len(tiles)} tiles)")

    candidates = grid[grid.intersects(boundary_geom)].copy()
    overlap = candidates.geometry.intersection(boundary_geom).area / candidates.geometry.area
    patches = candidates[overlap >= args.min_overlap].copy()
    print(f"patches with >= {args.min_overlap:.0%} inside the boundary: {len(patches)}")

    patches["n_points"] = pd.array([pd.NA] * len(patches), dtype="Int64")
    patches = patches[["patch_id", "tile", "n_points", "dop_file", "dop_year", "geometry"]]
    patches = patches.sort_values("patch_id").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    patches.to_parquet(OUT)
    per_tile = patches.groupby("tile").size()
    print(f"saved {OUT}")
    print(f"patches per tile: min {per_tile.min()}, max {per_tile.max()}")


if __name__ == "__main__":
    main()
