"""Build the tile list for the Dülmen study area.

Steps:
1. Fetch the Dülmen municipal boundary from OSM (Nominatim via osmnx) and
   save it to data/raw/boundary/duelmen_boundary.gpkg (EPSG:25832).
2. List all 1 km x 1 km tiles (EPSG:25832 grid) intersecting the boundary.
3. Read the OpenGeodata NRW index.json files for LAZ (3D-Messdaten) and
   DOP (DOP10 RGBI, JPEG2000) and look up exact file names and sizes.
   DOP file names contain the acquisition year, which differs per tile,
   so names must come from the index, not from the coordinates.
4. Write pipeline/01_download/tiles_duelmen.csv.

Run (inside the activated env, otherwise PROJ breaks):
    conda run -n pcsearch python pipeline/01_download/make_tile_list.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[2]

LAZ_BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/3dm_l_las/3dm_l_las/"
DOP_BASE = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/"

DEFAULT_PLACE = "Dülmen, Kreis Coesfeld, Nordrhein-Westfalen, Deutschland"
CRS = 25832
TILE_SIZE_M = 1000


def read_index(base_url: str) -> dict[str, int]:
    """Return {file_name: size_in_bytes} from an OpenGeodata NRW index.json."""
    with urllib.request.urlopen(base_url + "index.json", timeout=120) as r:
        data = json.load(r)
    return {
        f["name"]: int(f["size"])
        for ds in data["datasets"]
        for f in ds["files"]
    }


def tile_key(name: str) -> tuple[int, int]:
    """'3dm_32_377_5741_1_nw.laz' / 'dop10rgbi_32_377_5741_1_nw_2026.jp2' -> (377, 5741)."""
    parts = name.split("_")
    return int(parts[2]), int(parts[3])


def dop_year(name: str) -> int:
    """'dop10rgbi_32_377_5741_1_nw_2026.jp2' -> 2026."""
    return int(Path(name).stem.rsplit("_", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", default=DEFAULT_PLACE, help="Nominatim query for the boundary")
    ap.add_argument("--out", type=Path, default=REPO / "pipeline/01_download/tiles_duelmen.csv")
    ap.add_argument("--boundary-out", type=Path, default=REPO / "data/raw/boundary/duelmen_boundary.gpkg")
    args = ap.parse_args()

    # 1. Boundary
    ox.settings.cache_folder = str(REPO / "data/cache/osmnx")  # keep the HTTP cache out of the repo root
    print(f"Fetching boundary: {args.place}")
    boundary = ox.geocode_to_gdf(args.place).to_crs(CRS)
    if len(boundary) != 1:
        print(f"ERROR: expected 1 boundary, got {len(boundary)}", file=sys.stderr)
        return 1
    poly = boundary.geometry.iloc[0]
    print(f"  area: {poly.area / 1e6:.1f} km²")
    args.boundary_out.parent.mkdir(parents=True, exist_ok=True)
    boundary[["display_name", "geometry"]].to_file(args.boundary_out, driver="GPKG")
    print(f"  saved: {args.boundary_out.relative_to(REPO)}")

    # 2. Tiles intersecting the boundary
    minx, miny, maxx, maxy = poly.bounds
    tiles = [
        (e, n)
        for e in range(math.floor(minx / TILE_SIZE_M), math.ceil(maxx / TILE_SIZE_M))
        for n in range(math.floor(miny / TILE_SIZE_M), math.ceil(maxy / TILE_SIZE_M))
        if box(e * TILE_SIZE_M, n * TILE_SIZE_M, (e + 1) * TILE_SIZE_M, (n + 1) * TILE_SIZE_M).intersects(poly)
    ]
    print(f"  tiles intersecting boundary: {len(tiles)}")

    # 3. Indexes
    print("Reading LAZ index ...")
    laz = read_index(LAZ_BASE)
    laz_by_tile = {tile_key(n): n for n in laz if n.endswith(".laz")}
    print("Reading DOP index ...")
    dop = read_index(DOP_BASE)
    dop_by_tile: dict[tuple[int, int], str] = {}
    for n in dop:
        if not n.endswith(".jp2"):
            continue
        k = tile_key(n)
        # keep the most recent year if a tile appears more than once
        if k not in dop_by_tile or dop_year(n) > dop_year(dop_by_tile[k]):
            dop_by_tile[k] = n

    # 4. Table
    rows, missing = [], []
    for e, n in sorted(tiles):
        laz_name = laz_by_tile.get((e, n))
        dop_name = dop_by_tile.get((e, n))
        if laz_name is None:
            missing.append(f"LAZ {e}_{n}")
        if dop_name is None:
            missing.append(f"DOP {e}_{n}")
        rows.append({
            "tile_id": f"32_{e}_{n}",
            "easting_km": e,
            "northing_km": n,
            "laz_file": laz_name or "",
            "laz_size": laz.get(laz_name, "") if laz_name else "",
            "dop_file": dop_name or "",
            "dop_size": dop.get(dop_name, "") if dop_name else "",
            "dop_year": dop_year(dop_name) if dop_name else "",
        })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")

    laz_gb = pd.to_numeric(df["laz_size"], errors="coerce").sum() / 1e9
    dop_gb = pd.to_numeric(df["dop_size"], errors="coerce").sum() / 1e9
    print(f"Wrote {args.out.relative_to(REPO)}: {len(df)} tiles")
    print(f"  LAZ: {laz_gb:.1f} GB, DOP: {dop_gb:.1f} GB")
    print(f"  DOP years: {df['dop_year'].value_counts().sort_index().to_dict()}")
    if missing:
        print(f"WARNING: {len(missing)} files not in index: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
