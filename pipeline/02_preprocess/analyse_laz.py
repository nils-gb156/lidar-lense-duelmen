"""Analyse the LAZ tiles of Duelmen: CRS, height reference, point counts, density,
classes, returns and acquisition dates, per tile and per 50 m patch (issue #7).

Every tile is read once completely (laspy + lazrs). The result of each tile is
cached as JSON, so the script can be re-run and only processes new tiles.

Per tile:   LAS version, point format, CRS (horizontal + vertical EPSG), bounds,
            point count, density (all points and first returns = pulse density),
            class counts, return counts, z range, GPS time range and flight dates.
Per patch:  50 m x 50 m cells of the patch grid (same ids as make_patch_grid.py):
            point count (all, without noise class 18, ground class 2, class 20),
            mean ground height, max height, height range (max z - mean ground z).

GPS time: the tiles use adjusted standard GPS time (global encoding bit 0 = 1),
i.e. GPS seconds since 1980-01-06 minus 1e9. Dates are converted to UTC with
18 leap seconds (valid since 2017).

Input:  data/raw/laz/*.laz
        pipeline/01_download/tiles_duelmen.csv
        data/patches/patch_index.parquet   (to mark patches inside the boundary)
Output: data/analysis/laz/tiles/<tile_id>.json   (cache per tile)
        data/analysis/laz/tile_stats.csv
        data/analysis/laz/patch_stats.parquet
        data/analysis/laz/summary.md
        docs/figures/laz_*.png

Usage:
    python pipeline/02_preprocess/analyse_laz.py
    python pipeline/02_preprocess/analyse_laz.py --limit 5          # first 5 tiles only
    python pipeline/02_preprocess/analyse_laz.py --workers 4 --force
    python pipeline/02_preprocess/analyse_laz.py --plot-tile 32_380_5740 --plot-patch 32_380_5740_10_10
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TILES_CSV = ROOT / "pipeline" / "01_download" / "tiles_duelmen.csv"
LAZ_DIR = ROOT / "data" / "raw" / "laz"
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
OUT_DIR = ROOT / "data" / "analysis" / "laz"
TILE_CACHE = OUT_DIR / "tiles"
FIG_DIR = ROOT / "docs" / "figures"

TILE_SIZE_M = 1000
GPS_EPOCH_UNIX = 315964800  # 1980-01-06T00:00:00Z
LEAP_SECONDS = 18
ADJUSTED_GPS_OFFSET = 1_000_000_000

# Nutzerinformationen 3D-Messdaten NRW (Stand 02/2020)
CLASS_NAMES = {
    0: "never classified",
    1: "unclassified",
    2: "ground",
    17: "bridge",
    18: "noise",
    20: "last return not ground",
    24: "basement",
    26: "synthetic ground",
}
NOISE = 18
GROUND = 2
NOT_GROUND = 20
SYNTH_GROUND = 26


def gps_to_unix(t: np.ndarray) -> np.ndarray:
    return t + ADJUSTED_GPS_OFFSET + GPS_EPOCH_UNIX - LEAP_SECONDS


def analyse_tile(tile_id: str, laz_path: str, easting_km: int, northing_km: int, patch_size: int) -> dict:
    las = laspy.read(laz_path)
    h = las.header
    x0, y0 = easting_km * 1000, northing_km * 1000

    # GeoTIFF keys: 3072 ProjectedCSTypeGeoKey, 4096 VerticalCSTypeGeoKey
    # (laspy's parse_crs() only returns the horizontal CRS)
    epsg_h = epsg_v = None
    for vlr in h.vlrs:
        for key in getattr(vlr, "geo_keys", []) or []:
            if key.id == 3072:
                epsg_h = int(key.value_offset)
            elif key.id == 4096:
                epsg_v = int(key.value_offset)
    crs = h.parse_crs()
    crs_name = crs.name if crs is not None else None
    if epsg_h is None and crs is not None:
        epsg_h = crs.to_epsg()

    cls = np.asarray(las.classification, dtype=np.int64)
    rn = np.asarray(las.return_number, dtype=np.int64)
    nr = np.asarray(las.number_of_returns, dtype=np.int64)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    n = len(cls)

    class_counts = {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))}
    rn_counts = {int(k): int(v) for k, v in zip(*np.unique(rn, return_counts=True))}
    nr_counts = {int(k): int(v) for k, v in zip(*np.unique(nr, return_counts=True))}

    gps = {}
    if "gps_time" in las.point_format.dimension_names:
        t = np.asarray(las.gps_time)

        def day_counts(tt: np.ndarray) -> dict:
            days = np.floor(gps_to_unix(tt) / 86400).astype(np.int64)
            d, c = np.unique(days, return_counts=True)
            return {
                datetime.fromtimestamp(int(dd) * 86400, tz=timezone.utc).strftime("%Y-%m-%d"): int(cc)
                for dd, cc in zip(d, c)
            }

        # synthetic ground (26) carries GPS times of older data, so it is kept apart
        synth = cls == SYNTH_GROUND
        gps = {
            "global_encoding_adjusted": bool(int(h.global_encoding.gps_time_type) == 1),
            "gps_min": float(t[~synth].min()) if (~synth).any() else None,
            "gps_max": float(t[~synth].max()) if (~synth).any() else None,
            "dates": day_counts(t[~synth]),
            "dates_synthetic_ground": day_counts(t[synth]),
        }

    # covered area: occupied 1 m cells (robust against gaps / water / partial tiles)
    cx = np.clip(np.floor(x - x0).astype(np.int64), 0, TILE_SIZE_M - 1)
    cy = np.clip(np.floor(y - y0).astype(np.int64), 0, TILE_SIZE_M - 1)
    occupied_1m = int(np.unique(cy * TILE_SIZE_M + cx).size)

    # patch statistics on the 50 m grid (col, row from the lower-left tile corner)
    npatch = TILE_SIZE_M // patch_size
    pc = np.clip(np.floor((x - x0) / patch_size).astype(np.int64), 0, npatch - 1)
    pr = np.clip(np.floor((y - y0) / patch_size).astype(np.int64), 0, npatch - 1)
    pidx = pr * npatch + pc
    m = npatch * npatch
    valid = cls != NOISE
    ground = cls == GROUND
    n_all = np.bincount(pidx, minlength=m)
    n_valid = np.bincount(pidx[valid], minlength=m)
    n_ground = np.bincount(pidx[ground], minlength=m)
    n_c20 = np.bincount(pidx[cls == NOT_GROUND], minlength=m)
    n_first = np.bincount(pidx[valid & (rn == 1)], minlength=m)
    zg_sum = np.bincount(pidx[ground], weights=z[ground], minlength=m)
    with np.errstate(invalid="ignore", divide="ignore"):
        zg_mean = np.where(n_ground > 0, zg_sum / n_ground, np.nan)
    z_max = np.full(m, -np.inf)
    np.maximum.at(z_max, pidx[valid], z[valid])
    z_max[~np.isfinite(z_max)] = np.nan

    rows, cols = np.divmod(np.arange(m), npatch)
    patches = {
        "patch_id": [f"{tile_id}_{c:02d}_{r:02d}" for c, r in zip(cols, rows)],
        "n_all": n_all.tolist(),
        "n_valid": n_valid.tolist(),
        "n_ground": n_ground.tolist(),
        "n_class20": n_c20.tolist(),
        "n_first": n_first.tolist(),
        "z_ground_mean": [None if np.isnan(v) else round(float(v), 2) for v in zg_mean],
        "z_max": [None if np.isnan(v) else round(float(v), 2) for v in z_max],
    }

    zv = z[valid]
    return {
        "tile_id": tile_id,
        "file": Path(laz_path).name,
        "las_version": f"{h.version.major}.{h.version.minor}",
        "point_format": int(h.point_format.id),
        "crs_name": crs_name,
        "epsg_horizontal": epsg_h,
        "epsg_vertical": epsg_v,
        "creation_date": str(h.creation_date) if h.creation_date else None,
        "software": h.generating_software.strip("\x00 "),
        "scale": [float(s) for s in h.scales],
        "bounds": [float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1])],
        "n_points": int(n),
        "n_points_header": int(h.point_count),
        "occupied_1m_cells": occupied_1m,
        "z_min": float(zv.min()) if zv.size else None,
        "z_max": float(zv.max()) if zv.size else None,
        "z_p01": float(np.percentile(zv, 1)) if zv.size else None,
        "z_p99": float(np.percentile(zv, 99)) if zv.size else None,
        "class_counts": class_counts,
        "return_number_counts": rn_counts,
        "number_of_returns_counts": nr_counts,
        "gps": gps,
        "patches": patches,
    }


def run_tiles(tiles: pd.DataFrame, workers: int, patch_size: int, force: bool) -> None:
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in tiles.itertuples(index=False):
        cache = TILE_CACHE / f"{t.tile_id}.json"
        laz = LAZ_DIR / t.laz_file
        if cache.exists() and not force:
            continue
        if not laz.exists():
            print(f"missing {laz.name}, skipped")
            continue
        todo.append((t.tile_id, str(laz), int(t.easting_km), int(t.northing_km)))
    print(f"{len(tiles) - len(todo)} tiles cached, {len(todo)} to analyse with {workers} workers")
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyse_tile, *a, patch_size): a[0] for a in todo}
        for i, f in enumerate(as_completed(futs), 1):
            tid = futs[f]
            try:
                res = f.result()
            except Exception as e:  # keep going, report at the end
                print(f"[{i}/{len(todo)}] {tid}: ERROR {e}")
                continue
            (TILE_CACHE / f"{tid}.json").write_text(json.dumps(res))
            print(f"[{i}/{len(todo)}] {tid}: {res['n_points']:,} points", flush=True)


def load_results(tiles: pd.DataFrame) -> list[dict]:
    out = []
    for tid in tiles["tile_id"]:
        p = TILE_CACHE / f"{tid}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def build_tables(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, patch_frames = [], []
    for r in results:
        cc = {int(k): v for k, v in r["class_counts"].items()}
        rn = {int(k): v for k, v in r["return_number_counts"].items()}
        nr = {int(k): v for k, v in r["number_of_returns_counts"].items()}
        n = r["n_points"]
        dates = r["gps"].get("dates", {})
        main_date = max(dates, key=dates.get) if dates else None
        area_km2 = r["occupied_1m_cells"] / 1e6
        row = {
            "tile_id": r["tile_id"],
            "las_version": r["las_version"],
            "point_format": r["point_format"],
            "epsg_horizontal": r["epsg_horizontal"],
            "epsg_vertical": r["epsg_vertical"],
            "crs_name": r["crs_name"],
            "creation_date": r["creation_date"],
            "n_points": n,
            "covered_km2": area_km2,
            "density_all": n / (area_km2 * 1e6) if area_km2 else np.nan,
            "density_first": rn.get(1, 0) / (area_km2 * 1e6) if area_km2 else np.nan,
            "share_multi_return": sum(v for k, v in nr.items() if k > 1) / n if n else np.nan,
            "z_min": r["z_min"],
            "z_max": r["z_max"],
            "z_p01": r["z_p01"],
            "z_p99": r["z_p99"],
            "main_flight_date": main_date,
            "flight_dates": ";".join(sorted(dates)),
            "n_flight_dates": len(dates),
        }
        for k in sorted(set(CLASS_NAMES) | set(cc)):
            row[f"class_{k}"] = cc.get(k, 0)
        rows.append(row)
        pf = pd.DataFrame(r["patches"])
        pf["tile"] = r["tile_id"]
        patch_frames.append(pf)
    tile_df = pd.DataFrame(rows)
    patch_df = pd.concat(patch_frames, ignore_index=True)
    patch_df["height_range"] = patch_df["z_max"] - patch_df["z_ground_mean"]
    return tile_df, patch_df


def q(s: pd.Series, p: float) -> float:
    return float(np.nanpercentile(s.astype(float), p))


def write_summary(tile_df: pd.DataFrame, patch_df: pd.DataFrame, results: list[dict], patch_size: int) -> str:
    n_tiles = len(tile_df)
    total = int(tile_df["n_points"].sum())
    class_cols = [c for c in tile_df.columns if c.startswith("class_")]
    class_tot = tile_df[class_cols].sum()

    dates_tot: dict[str, int] = {}
    synth_tot: dict[str, int] = {}
    for r in results:
        for d, c in r["gps"].get("dates", {}).items():
            dates_tot[d] = dates_tot.get(d, 0) + c
        for d, c in r["gps"].get("dates_synthetic_ground", {}).items():
            synth_tot[d] = synth_tot.get(d, 0) + c

    def uniq(col: str) -> list:
        return sorted({(int(v) if isinstance(v, (int, np.integer, float)) and not pd.isna(v) else v) for v in tile_df[col].dropna()})

    inside = patch_df[patch_df["in_boundary"]]
    nv = inside["n_valid"]
    lines = []
    lines.append(f"# LAZ analysis Duelmen ({n_tiles} tiles)\n")
    lines.append("## Format and CRS")
    lines.append(f"- LAS versions: {uniq('las_version')}, point formats: {uniq('point_format')}")
    lines.append(f"- CRS (laspy): {uniq('crs_name')}")
    lines.append(f"- EPSG horizontal: {uniq('epsg_horizontal')}, vertical: {uniq('epsg_vertical')} (GeoTIFF keys)")
    lines.append(f"- Creation dates (header, processing date): {tile_df['creation_date'].min()} to {tile_df['creation_date'].max()}")
    lines.append("")
    lines.append("## Points and density")
    lines.append(f"- Total points: {total:,}")
    lines.append(
        f"- Points per tile: min {tile_df['n_points'].min():,}, median {int(tile_df['n_points'].median()):,}, max {tile_df['n_points'].max():,}"
    )
    lines.append(
        f"- Density all points (pts/m², covered area): min {tile_df['density_all'].min():.1f}, "
        f"median {tile_df['density_all'].median():.1f}, max {tile_df['density_all'].max():.1f}"
    )
    lines.append(
        f"- First returns (≈ pulse density, pts/m²): min {tile_df['density_first'].min():.1f}, "
        f"median {tile_df['density_first'].median():.1f}, max {tile_df['density_first'].max():.1f}"
    )
    lines.append(f"- Share of points from multi-return pulses: median {tile_df['share_multi_return'].median():.1%}")
    lines.append(f"- Height (DHHN2016, without noise), 1st–99th percentile over tiles: {tile_df['z_p01'].min():.1f} m to {tile_df['z_p99'].max():.1f} m")
    lines.append("")
    lines.append("## Classes (all tiles)")
    lines.append("| Class | Name | Points | Share |")
    lines.append("|---|---|---|---|")
    for c in class_cols:
        k = int(c.split("_")[1])
        v = int(class_tot[c])
        if v == 0:
            continue
        lines.append(f"| {k} | {CLASS_NAMES.get(k, 'not documented')} | {v:,} | {v / total:.2%} |")
    lines.append("")
    lines.append("## Acquisition dates (from GPS time, UTC)")
    adjusted = {r["gps"].get("global_encoding_adjusted") for r in results}
    lines.append(f"- Adjusted standard GPS time in all tiles: {adjusted == {True}}")
    lines.append(f"- Flight days (without class 26): {len(dates_tot)}, from {min(dates_tot)} to {max(dates_tot)}")
    if synth_tot:
        lines.append(
            f"- Class 26 (synthetic ground) GPS dates: {len(synth_tot)} days, from {min(synth_tot)} to {max(synth_tot)} "
            "(older data, not the 2024 flight)"
        )
    lines.append(f"- Tiles with more than one flight day: {(tile_df['n_flight_dates'] > 1).sum()}")
    lines.append("| Date | Points | Share | Tiles (main date) |")
    lines.append("|---|---|---|---|")
    main_counts = tile_df["main_flight_date"].value_counts()
    for d in sorted(dates_tot):
        lines.append(f"| {d} | {dates_tot[d]:,} | {dates_tot[d] / total:.1%} | {int(main_counts.get(d, 0))} |")
    lines.append("")
    lines.append(f"## {patch_size} m patches inside the boundary ({len(inside):,} patches)")
    lines.append("Points without noise (class 18):")
    lines.append(
        f"- min {int(nv.min()):,}, p1 {q(nv, 1):,.0f}, p5 {q(nv, 5):,.0f}, median {q(nv, 50):,.0f}, "
        f"p95 {q(nv, 95):,.0f}, max {int(nv.max()):,}"
    )
    for thr in (1024, 2048, 4096, 8192, 16384):
        lines.append(f"- patches with fewer than {thr:,} points: {(nv < thr).sum():,} ({(nv < thr).mean():.2%})")
    lines.append(f"- patches without any ground point: {(inside['n_ground'] == 0).sum():,}")
    hr = inside["height_range"]
    lines.append(
        f"- height range (max z − mean ground z): median {q(hr, 50):.1f} m, p95 {q(hr, 95):.1f} m, "
        f"p99 {q(hr, 99):.1f} m, max {np.nanmax(hr):.1f} m"
    )
    for thr in (20, 30, 50):
        lines.append(f"- patches with height range > {thr} m: {(hr > thr).sum():,}")
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    return text


def plot_overview(tile_df: pd.DataFrame, patch_df: pd.DataFrame, tiles: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    inside = patch_df[patch_df["in_boundary"]]

    # points per patch histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    v = inside["n_valid"].clip(upper=np.nanpercentile(inside["n_valid"], 99.5))
    ax.hist(v, bins=80, color="#4477aa")
    for thr, ls in ((4096, "--"), (8192, ":")):
        ax.axvline(thr, color="k", ls=ls, lw=1, label=f"{thr} points")
    ax.set_xlabel("points per 50 m patch (without noise)")
    ax.set_ylabel("patches")
    ax.set_title(f"Points per patch, {len(inside):,} patches inside the boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "laz_points_per_patch.png", dpi=150)
    plt.close(fig)

    # density + flight date maps per tile
    t = tiles.merge(tile_df, on="tile_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    sc = axes[0].scatter(t["easting_km"] + 0.5, t["northing_km"] + 0.5, c=t["density_first"], marker="s", s=70, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label="first returns per m²")
    axes[0].set_title("Pulse density per tile")
    dates = sorted(t["main_flight_date"].dropna().unique())
    code = t["main_flight_date"].map({d: i for i, d in enumerate(dates)})
    cmap = plt.get_cmap("tab10" if len(dates) <= 10 else "tab20", max(len(dates), 1))
    axes[1].scatter(t["easting_km"] + 0.5, t["northing_km"] + 0.5, c=code, marker="s", s=70, cmap=cmap, vmin=-0.5, vmax=len(dates) - 0.5)
    for i, d in enumerate(dates):
        axes[1].scatter([], [], c=[cmap(i)], marker="s", label=d)
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    axes[1].set_title("Main flight date per tile")
    for a in axes:
        a.set_aspect("equal")
        a.set_xlabel("easting [km], EPSG:25832")
        a.set_ylabel("northing [km]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "laz_tiles_density_date.png", dpi=150)
    plt.close(fig)

    # height range per patch
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(inside["height_range"].dropna().clip(-5, 80), bins=85, color="#228833")
    ax.set_yscale("log")
    ax.set_xlabel("height range per patch: max z − mean ground z [m] (clipped at 80)")
    ax.set_ylabel("patches (log)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "laz_height_range_per_patch.png", dpi=150)
    plt.close(fig)


CLASS_COLORS = {1: "#bbbbbb", 2: "#a6761d", 17: "#e7298a", 18: "#ff0000", 20: "#1b9e77", 24: "#7570b3", 26: "#e6ab02"}


def plot_tile(tile_id: str, tiles: pd.DataFrame, patch_id: str | None, patch_size: int, n_sample: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    row = tiles[tiles["tile_id"] == tile_id]
    if row.empty:
        sys.exit(f"unknown tile {tile_id}")
    row = row.iloc[0]
    las = laspy.read(LAZ_DIR / row.laz_file)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    cls = np.asarray(las.classification)
    x0, y0 = row.easting_km * 1000, row.northing_km * 1000
    valid = cls != NOISE

    # 1 m rasters: max height and class of the highest point
    cx = np.clip(np.floor(x - x0).astype(int), 0, 999)
    cy = np.clip(np.floor(y - y0).astype(int), 0, 999)
    idx = cy * 1000 + cx
    order = np.argsort(z, kind="stable")
    order = order[valid[order]]
    zmax = np.full(10**6, np.nan)
    cmax = np.zeros(10**6, dtype=int)
    zmax[idx[order]] = z[order]  # last write = highest point
    cmax[idx[order]] = cls[order]
    zmax = zmax.reshape(1000, 1000)
    cmax = cmax.reshape(1000, 1000)

    present = sorted(int(c) for c in np.unique(cmax) if c != 0)
    lut = {c: i for i, c in enumerate(present)}
    cimg = np.full(cmax.shape, np.nan)
    for c, i in lut.items():
        cimg[cmax == c] = i
    cm = ListedColormap([CLASS_COLORS.get(c, "#000000") for c in present])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ext = (x0, x0 + 1000, y0, y0 + 1000)
    im = axes[0].imshow(zmax, origin="lower", extent=ext, cmap="terrain", vmin=np.nanpercentile(zmax, 1), vmax=np.nanpercentile(zmax, 99.5))
    fig.colorbar(im, ax=axes[0], label="max height [m, DHHN2016]")
    axes[0].set_title(f"{tile_id}: highest point per 1 m cell")
    axes[1].imshow(cimg, origin="lower", extent=ext, cmap=cm, vmin=-0.5, vmax=len(present) - 0.5, interpolation="nearest")
    for c in present:
        axes[1].scatter([], [], c=CLASS_COLORS.get(c, "#000000"), marker="s", label=f"{c} {CLASS_NAMES.get(c, '?')}")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title("class of the highest point per 1 m cell")
    for a in axes:
        a.set_xticks(np.arange(x0, x0 + 1001, 250))
        a.set_yticks(np.arange(y0, y0 + 1001, 250))
        a.grid(True, color="w", lw=0.3)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"laz_tile_{tile_id}.png", dpi=150)
    plt.close(fig)
    print(f"saved {FIG_DIR / f'laz_tile_{tile_id}.png'}")

    if patch_id:
        c, r = (int(s) for s in patch_id.split("_")[-2:])
        px0, py0 = x0 + c * patch_size, y0 + r * patch_size
        m = valid & (x >= px0) & (x < px0 + patch_size) & (y >= py0) & (y < py0 + patch_size)
        sel = np.flatnonzero(m)
        rng = np.random.default_rng(0)
        if sel.size > n_sample:
            sel = rng.choice(sel, n_sample, replace=False)
        gz = z[m & (cls == GROUND)]
        zref = np.median(gz) if gz.size else z[m].min()
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        cols = [CLASS_COLORS.get(int(k), "#000000") for k in cls[sel]]
        ax.scatter(x[sel] - px0, y[sel] - py0, z[sel] - zref, c=cols, s=1)
        ax.set_title(f"{patch_id}: {sel.size:,} of {int(m.sum()):,} points (random sample)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z − median ground [m]")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(x[sel] - px0, z[sel] - zref, c=cols, s=1)
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("z − median ground [m]")
        ax2.set_title("side view (all y)")
        ax2.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"laz_patch_{patch_id}.png", dpi=150)
        plt.close(fig)
        print(f"saved {FIG_DIR / f'laz_patch_{patch_id}.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6, help="parallel processes (default 6, ~1.5 GB RAM each)")
    ap.add_argument("--patch-size", type=int, default=50, help="patch edge length in m (default 50)")
    ap.add_argument("--limit", type=int, default=None, help="analyse only the first N tiles")
    ap.add_argument("--force", action="store_true", help="re-analyse cached tiles")
    ap.add_argument("--no-figures", action="store_true", help="skip the overview figures")
    ap.add_argument("--plot-tile", default=None, help="tile_id to visualise (top view, classes)")
    ap.add_argument("--plot-patch", default=None, help="patch_id inside --plot-tile to show in 3D")
    ap.add_argument("--sample", type=int, default=4096, help="points shown in the patch plot (default 4096)")
    args = ap.parse_args()

    tiles = pd.read_csv(TILES_CSV)
    if args.plot_tile:
        plot_tile(args.plot_tile, tiles, args.plot_patch, args.patch_size, args.sample)
        return
    if args.limit:
        tiles = tiles.head(args.limit)
    if args.force:
        print("--force: cached tiles are re-analysed")

    run_tiles(tiles, args.workers, args.patch_size, args.force)
    results = load_results(tiles)
    if not results:
        sys.exit("no tile results")
    tile_df, patch_df = build_tables(results)

    if PATCH_INDEX.exists():
        ids = set(pd.read_parquet(PATCH_INDEX, columns=["patch_id"])["patch_id"])
        patch_df["in_boundary"] = patch_df["patch_id"].isin(ids)
    else:
        print(f"{PATCH_INDEX} not found, all patches treated as inside the boundary")
        patch_df["in_boundary"] = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_df.to_csv(OUT_DIR / "tile_stats.csv", index=False)
    patch_df.to_parquet(OUT_DIR / "patch_stats.parquet", index=False)
    print(f"saved {OUT_DIR / 'tile_stats.csv'} and {OUT_DIR / 'patch_stats.parquet'}")

    print()
    print(write_summary(tile_df, patch_df, results, args.patch_size))
    if not args.no_figures:
        plot_overview(tile_df, patch_df, tiles)
        print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
