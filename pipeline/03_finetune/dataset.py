"""PyTorch dataset: point cloud patch -> GeoRSCLIP DOP embedding (issue #15).

One item = one 50 m patch:
    points   float32 (6, N)  x, y, HAG divided by 25 m, then RGB = 0.4 constant
                             (OpenShape PointBERT input layout (B, 6, N), z up)
    target   float32 (1024,) GeoRSCLIP ViT-H/14 embedding of the DOP 2024 crop (L2-normalised)
    patch_id str
    n_points int             number of stored points of the patch (before sampling)

Input settings (see CLAUDE.md "Points per patch" and docs/results.md, OpenShape smoke test):
- N = 10,000 points drawn at random from all stored points; the result is in random order
  (OpenShape's ball query takes the first points in list order)
- patches with fewer than N points: all points plus random repeats up to N
- patches with fewer than min_points points (default 1,000) are skipped
- no DOP colours: they would leak image information into the LiDAR branch

Sampling:
- train: new random points every epoch (call set_epoch(e) at the start of each epoch);
  reproducible from (seed, epoch, patch)
- val / test: fixed points per patch (same every call), so validation scores are comparable

Augmentation (train only, off by default; compare in #22):
- rotate: "none" | "90" (k * 90 deg around z plus random mirror) | "any" (any angle around z)
  Note: the target is the embedding of the north-up DOP crop and is NOT rotated with the points,
  so rotation teaches the encoder to ignore orientation. Whether this helps is open.
- jitter_m: Gaussian noise on x, y, z in metres (e.g. 0.02), clipped to 3 sigma

The image itself is not loaded: the CLIP teacher is frozen, so its embeddings are computed once
(pipeline/04_embed/embed_georsclip.py) and only the vectors are needed for training.

Input:  data/processed/patch_points/  (index.parquet, <tile>_xyz.npy, meta.json)  from make_patch_points.py
        data/embeddings/dop_image_embeddings_georsclip_vith14.npz (+ .meta.json)  from embed_georsclip.py

Quick check (shapes, timing, GPU transfer):
    python pipeline/03_finetune/dataset.py
    python pipeline/03_finetune/dataset.py --batch-size 16 --workers 4 --batches 20
    python pipeline/03_finetune/dataset.py --points-dir D:/data/patch_points --emb D:/data/emb.npz
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
POINTS_DIR = ROOT / "data" / "processed" / "patch_points"
EMB_FILE = ROOT / "data" / "embeddings" / "dop_image_embeddings_georsclip_vith14.npz"

EXPECTED_CLIP = {"clip_arch": "ViT-H-14", "clip_pretrained": "RS5M_ViT-H-14.pt"}  # CLAUDE.md rule 1
N_POINTS = 10_000
SCALE_M = 25.0
RGB_VALUE = 0.4
MIN_POINTS = 1_000
SPLITS = ("train", "val", "test")
ROTATIONS = ("none", "90", "any")


class PatchPointDataset(Dataset):
    def __init__(
        self,
        split: str,
        points_dir: Path | str = POINTS_DIR,
        emb_file: Path | str = EMB_FILE,
        n_points: int = N_POINTS,
        scale_m: float = SCALE_M,
        rgb_value: float = RGB_VALUE,
        min_points: int = MIN_POINTS,
        rotate: str = "none",
        jitter_m: float = 0.0,
        seed: int = 0,
        limit: int | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if rotate not in ROTATIONS:
            raise ValueError(f"rotate must be one of {ROTATIONS}, got {rotate!r}")
        self.split = split
        self.points_dir = Path(points_dir)
        self.n_points = n_points
        self.scale_m = scale_m
        self.rgb_value = rgb_value
        self.train = split == "train"
        self.rotate = rotate if self.train else "none"
        self.jitter_m = jitter_m if self.train else 0.0
        self.seed = seed
        self.epoch = 0
        self._mmaps: dict[str, np.ndarray] = {}  # opened lazily, per DataLoader worker

        # targets + metadata check
        emb_file = Path(emb_file)
        self.emb_meta = json.loads(emb_file.with_suffix(".meta.json").read_text(encoding="utf-8"))
        for k, v in EXPECTED_CLIP.items():
            if self.emb_meta.get(k) != v:
                raise ValueError(f"{emb_file.name}: {k} = {self.emb_meta.get(k)!r}, expected {v!r} (CLIP consistency)")
        emb = np.load(emb_file)
        row_of = {pid: i for i, pid in enumerate(emb["ids"])}

        # patches of this split with enough points and a target
        index = pd.read_parquet(self.points_dir / "index.parquet")
        self.points_meta = json.loads((self.points_dir / "meta.json").read_text(encoding="utf-8"))
        sel = index[index["split"] == split]
        n_split = len(sel)
        sel = sel[sel["count"] >= min_points]
        n_few = n_split - len(sel)
        sel = sel[sel["patch_id"].isin(row_of)]
        n_no_target = n_split - n_few - len(sel)
        sel = sel.sort_values("patch_id").reset_index(drop=True)
        if limit:
            sel = sel.head(limit)

        self.patch_ids = sel["patch_id"].to_numpy()
        self.tiles = sel["tile"].to_numpy()
        self.starts = sel["start"].to_numpy(np.int64)
        self.counts = sel["count"].to_numpy(np.int64)
        self.targets = torch.from_numpy(emb["emb"][[row_of[p] for p in self.patch_ids]].astype(np.float32))
        self.stats = {"split": split, "patches_in_split": n_split, "skipped_few_points": n_few,
                      "skipped_no_target": n_no_target, "used": len(sel)}

    def __len__(self) -> int:
        return len(self.patch_ids)

    def set_epoch(self, epoch: int) -> None:
        """New random points per epoch (train). With persistent_workers=True this must be set before
        the workers are created, so use persistent_workers=False (the default) when calling it."""
        self.epoch = epoch

    def _xyz(self, tile: str) -> np.ndarray:
        if tile not in self._mmaps:
            self._mmaps[tile] = np.load(self.points_dir / f"{tile}_xyz.npy", mmap_mode="r")
        return self._mmaps[tile]

    def _rng(self, i: int) -> np.random.Generator:
        if self.train:
            return np.random.default_rng([self.seed, self.epoch, i])
        return np.random.default_rng([self.seed, i])  # fixed per patch for val/test

    def __getitem__(self, i: int) -> dict:
        rng = self._rng(i)
        start, count = int(self.starts[i]), int(self.counts[i])
        if count >= self.n_points:
            pick = np.sort(rng.choice(count, self.n_points, replace=False))  # sorted = faster mmap read
        else:
            pick = np.concatenate([np.arange(count), rng.integers(0, count, self.n_points - count)])
        xyz = np.asarray(self._xyz(self.tiles[i])[start + pick], dtype=np.float32)
        xyz = xyz[rng.permutation(self.n_points)]  # random order for the ball query

        if self.rotate == "90":
            k = int(rng.integers(4))
            for _ in range(k):
                xyz[:, [0, 1]] = np.stack([-xyz[:, 1], xyz[:, 0]], axis=1)
            if rng.random() < 0.5:
                xyz[:, 0] = -xyz[:, 0]
        elif self.rotate == "any":
            a = rng.uniform(0, 2 * math.pi)
            c, s = math.cos(a), math.sin(a)
            x, y = xyz[:, 0].copy(), xyz[:, 1].copy()
            xyz[:, 0], xyz[:, 1] = c * x - s * y, s * x + c * y
        if self.jitter_m > 0:
            xyz += np.clip(rng.normal(0, self.jitter_m, xyz.shape), -3 * self.jitter_m, 3 * self.jitter_m).astype(np.float32)

        xyz /= self.scale_m
        rgb = np.full_like(xyz, self.rgb_value)
        points = torch.from_numpy(np.concatenate([xyz, rgb], axis=1).T.copy())  # (6, N)
        return {"points": points, "target": self.targets[i], "patch_id": self.patch_ids[i], "n_points": count}

    def metadata(self) -> dict:
        """Settings to store in checkpoints and embedding files (CLAUDE.md rule 4)."""
        return {
            "clip_arch": self.emb_meta.get("clip_arch"),
            "clip_pretrained": self.emb_meta.get("clip_pretrained"),
            "clip_weights_source": self.emb_meta.get("weights_source"),
            "target_backbone": self.emb_meta.get("backbone"),
            "target_dim": int(self.targets.shape[1]),
            "patch_size_m": self.points_meta.get("patch_size_m"),
            "point_data_source": self.points_meta.get("data_source"),
            "point_acquisition_year": self.points_meta.get("acquisition_year"),
            "dop_years": self.emb_meta.get("dop_years"),
            "dropped_classes": self.points_meta.get("dropped_classes"),
            "n_points": self.n_points,
            "scale_m": self.scale_m,
            "rgb_value": self.rgb_value,
            "rotate": self.rotate,
            "jitter_m": self.jitter_m,
            "seed": self.seed,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points-dir", default=str(POINTS_DIR))
    ap.add_argument("--emb", default=str(EMB_FILE))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batches", type=int, default=10, help="batches to time per split")
    ap.add_argument("--rotate", default="90", choices=ROTATIONS, help="augmentation used for the train check")
    ap.add_argument("--jitter-m", type=float, default=0.02)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for split in SPLITS:
        ds = PatchPointDataset(split, args.points_dir, args.emb, rotate=args.rotate, jitter_m=args.jitter_m)
        print(f"\n{split}: {ds.stats}")
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=ds.train, num_workers=args.workers,
                        pin_memory=device == "cuda", drop_last=ds.train)
        t0 = time.time()
        for b, batch in enumerate(dl):
            pts = batch["points"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            if b == 0:
                t_first = time.time() - t0
                print(f"  points {tuple(pts.shape)} {pts.dtype} on {pts.device}, target {tuple(tgt.shape)} {tgt.dtype}")
                xyz = pts[:, :3]
                print(f"  x, y range {xyz[:, :2].min():.2f} .. {xyz[:, :2].max():.2f}, "
                      f"z range {xyz[:, 2].min():.2f} .. {xyz[:, 2].max():.2f}, rgb {pts[:, 3:].unique().tolist()}")
                print(f"  target norm {tgt.norm(dim=1).mean():.4f}, ids {list(batch['patch_id'][:3])}")
            if b + 1 >= args.batches:
                break
        el = time.time() - t0
        print(f"  {b + 1} batches in {el:.1f} s (first {t_first:.1f} s incl. worker start), "
              f"{(el - t_first) / max(b, 1) / args.batch_size * 1000:.1f} ms per patch after that")

    # determinism checks
    val = PatchPointDataset("val", args.points_dir, args.emb, limit=5)
    assert torch.equal(val[0]["points"], val[0]["points"]), "val sampling must be fixed"
    tr = PatchPointDataset("train", args.points_dir, args.emb, limit=5)
    a = tr[0]["points"]
    tr.set_epoch(1)
    assert not torch.equal(a, tr[0]["points"]), "train sampling must change per epoch"
    tr.set_epoch(0)
    assert torch.equal(a, tr[0]["points"]), "train sampling must be reproducible"
    print("\ndeterminism checks passed (val fixed, train changes per epoch and is reproducible)")
    print(json.dumps(tr.metadata(), indent=2))


if __name__ == "__main__":
    main()
