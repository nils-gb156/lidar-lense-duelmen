"""Geographic train / val / test split of the Duelmen patches (CLAUDE.md rule 3).

Neighbouring patches look alike, so a random split per patch would leak information
from training into the test. Instead, whole blocks of tiles are assigned to one split:
the 1 km tiles are grouped into blocks of --block x --block km, the blocks are shuffled
with a fixed seed and assigned greedily so that the patch counts come close to the
target shares (default 70 / 10 / 20).

Use:
    train  training of the point cloud encoder (Duelmen prototype)
    val    every choice: CLIP teacher, encoder, hyperparameters, thresholds
    test   only the final evaluation, never for decisions

The split is written into the column "split" of the patch index. An existing split is
kept unless --force is given, so results stay comparable.

Input:  data/patches/patch_index.parquet
Output: data/patches/patch_index.parquet   (column split added)
        data/patches/split_blocks.csv        (block -> split, for documentation)

Usage:
    python pipeline/02_preprocess/make_split.py
    python pipeline/02_preprocess/make_split.py --block 2 --shares 0.7 0.1 0.2 --seed 42 --force
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PATCH_INDEX = ROOT / "data" / "patches" / "patch_index.parquet"
BLOCKS_CSV = ROOT / "data" / "patches" / "split_blocks.csv"
SPLITS = ("train", "val", "test")


def tile_xy(tile: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'32_377_5741' -> (377, 5741)."""
    parts = tile.str.split("_", expand=True)
    return parts[1].astype(int), parts[2].astype(int)


def assign_blocks(counts: pd.Series, shares: list[float], seed: int) -> dict:
    """Greedy: each shuffled block goes to the split that is furthest below its target."""
    rng = np.random.default_rng(seed)
    order = counts.index.to_numpy()
    rng.shuffle(order)
    total = counts.sum()
    target = {s: share * total for s, share in zip(SPLITS, shares)}
    filled = {s: 0 for s in SPLITS}
    out = {}
    # the smaller splits (val, test) first get one block each, so none stays empty
    for s, block in zip(("test", "val"), order[:2]):
        out[block] = s
        filled[s] += counts[block]
    for block in order[2:]:
        s = max(SPLITS, key=lambda k: (target[k] - filled[k]) / target[k])
        out[block] = s
        filled[s] += counts[block]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, default=2, help="block edge length in tiles (km), default 2")
    ap.add_argument("--shares", type=float, nargs=3, default=[0.7, 0.1, 0.2], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    if abs(sum(args.shares) - 1) > 1e-6:
        sys.exit(f"shares must sum to 1, got {sum(args.shares)}")
    if not PATCH_INDEX.exists():
        sys.exit(f"{PATCH_INDEX} not found, run make_patch_grid.py first")

    patches = gpd.read_parquet(PATCH_INDEX)
    if "split" in patches.columns and patches["split"].notna().all() and not args.force:
        print(f"split exists, keeping it (use --force to rebuild):")
        print(patches["split"].value_counts().reindex(SPLITS).to_string())
        return

    tx, ty = tile_xy(patches["tile"])
    patches["block"] = (tx // args.block).astype(str) + "_" + (ty // args.block).astype(str)
    counts = patches.groupby("block").size()
    if len(counts) < 3:
        sys.exit(f"only {len(counts)} blocks, use a smaller --block")

    mapping = assign_blocks(counts, args.shares, args.seed)
    patches["split"] = patches["block"].map(mapping)

    BLOCKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"block": counts.index, "n_patches": counts.values,
                  "split": [mapping[b] for b in counts.index]}).to_csv(BLOCKS_CSV, index=False)
    patches = patches.drop(columns="block")
    patches.to_parquet(PATCH_INDEX)

    n = patches["split"].value_counts().reindex(SPLITS)
    print(f"{len(counts)} blocks of {args.block} x {args.block} km, seed {args.seed}")
    for s in SPLITS:
        print(f"  {s:5s} {n[s]:6d} patches ({n[s] / len(patches):.1%}), "
              f"{sum(v == s for v in mapping.values())} blocks")
    print(f"saved {PATCH_INDEX} and {BLOCKS_CSV}")


if __name__ == "__main__":
    main()
