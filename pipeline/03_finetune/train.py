"""Fine-tune OpenShape PointBERT (bigG) so that point cloud patches map to GeoRSCLIP DOP embeddings (issue #22).

LidarCLIP-style distillation:
- teacher: GeoRSCLIP ViT-H/14 embedding of the DOP 2024 crop of each patch (precomputed, frozen)
- student: OpenShape PointBERT ViT-bigG/14 backbone (pretrained) + NEW head LayerNorm(512) -> Linear(512, 1024)
  (the pretrained 512 -> 1280 head maps into OpenCLIP bigG space, which is not the GeoRSCLIP space).
  The backbone features have a norm of about 190; the LayerNorm brings them to a fixed scale and the
  linear layer starts at zero (first prediction = zero vector), so the new head does not send large
  random gradients into the pretrained backbone.
- loss: MSE between student output and the L2-normalised target vector (LidarCLIP, Tab. 4), summed over
  the 1024 dimensions (= MSE * 1024). For unit vectors this is |y - t|^2 = 2 - 2 cos: predicting the zero
  vector gives 1.0, a perfect prediction 0.
- data: pipeline/03_finetune/dataset.py (10,000 points, x, y, HAG / 25 m, RGB 0.4)
- splits: train for training, val for model selection; the test split is never loaded here

Validation after every epoch, on all val patches (fixed points per patch):
- matching recall@k: rank of the patch's own DOP vector among all val DOP vectors
  (cosine similarity of the normalised student output); chance level for recall@1 = 1 / n_val
- mean cosine similarity to the own target, MSE
The checkpoint with the best val recall@1 is kept as best.pt.

Output (one folder per run, default data/models/<run-name>/):
    config.json   all arguments + data/model metadata
    log.jsonl     one line per epoch (train loss, val metrics, learning rate, time)
    best.pt       model weights + metadata of the best epoch (val recall@1)
    last.pt       model, optimizer, scheduler, scaler, epoch -> resume with --resume

Requirements (besides torch): torch.redstone, einops, huggingface_hub (see environment.yml).
The OpenShape loader (https://huggingface.co/OpenShape/openshape-demo-support, MIT) is cloned to
~/.cache/openshape-demo-support if missing. dgl is replaced by a PyTorch farthest point sampler.
Weights: huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb (no licence stated on the model page: open,
check before redistributing checkpoints, CLAUDE.md rule 8).

Hyperparameter defaults are starting values, not tuned.

Usage:
    # quick test on the laptop (MX550): few patches, small batch
    python pipeline/03_finetune/train.py --run-name smoke --limit-train 64 --limit-val 32 --batch-size 2 --epochs 2 --workers 0
    # training PC
    python pipeline/03_finetune/train.py --run-name dl_v1 --batch-size 32 --workers 8
    python pipeline/03_finetune/train.py --run-name dl_v1 --resume      # continue from last.pt
    python pipeline/03_finetune/train.py --points-dir D:/data/patch_points --emb D:/data/dop_image_embeddings_georsclip_vith14.npz
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import EMB_FILE, POINTS_DIR, ROTATIONS, PatchPointDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "data" / "models"
OPENSHAPE_DIR = Path.home() / ".cache" / "openshape-demo-support"
OPENSHAPE_REPO = "https://huggingface.co/OpenShape/openshape-demo-support"
OPENSHAPE_MODEL = "openshape-pointbert-vitg14-rgb"
BACKBONE_DIM = 512


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
def farthest_point_sampler(pos: torch.Tensor, npoints: int, start_idx: int | None = 0) -> torch.Tensor:
    """Same interface as dgl.geometry.farthest_point_sampler: pos (B, N, C) -> indices (B, npoints).

    Starts at point 0. The dataset shuffles the points, so point 0 is a random point in training
    and a fixed point for val/test (deterministic validation)."""
    B, N, _ = pos.shape
    idx = torch.zeros(B, npoints, dtype=torch.long, device=pos.device)
    dist = torch.full((B, N), float("inf"), device=pos.device)
    if start_idx is None:
        far = torch.randint(0, N, (B,), device=pos.device)
    else:
        far = torch.full((B,), int(start_idx), dtype=torch.long, device=pos.device)
    bidx = torch.arange(B, device=pos.device)
    for i in range(npoints):
        idx[:, i] = far
        c = pos[bidx, far].unsqueeze(1)
        dist = torch.minimum(dist, ((pos - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


def import_openshape(loader_dir: Path):
    if not loader_dir.is_dir():
        print(f"cloning the OpenShape loader to {loader_dir}")
        subprocess.run(["git", "clone", "-q", OPENSHAPE_REPO, str(loader_dir)], check=True)
    sys.path.insert(0, str(loader_dir))
    try:
        import dgl.geometry  # noqa: F401
    except Exception:
        dgl = types.ModuleType("dgl")
        geom = types.ModuleType("dgl.geometry")
        geom.farthest_point_sampler = farthest_point_sampler
        dgl.geometry = geom
        sys.modules["dgl"] = dgl
        sys.modules["dgl.geometry"] = geom
    import openshape

    return openshape


def fp32_forward(module: nn.Module) -> None:
    """Run a module in float32 under autocast. The ball query compares squared distances
    (|a|^2 + |b|^2 - 2ab) with radius^2 = 0.04; in float16 that is too coarse."""
    orig = module.forward

    def forward(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            args = [a.float() if torch.is_tensor(a) else a for a in args]
            return orig(*args, **kwargs)

    module.forward = forward


def build_model(loader_dir: Path, out_dim: int) -> nn.Module:
    openshape = import_openshape(loader_dir)
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})  # trusted file from OpenShape on HF
    try:
        model = openshape.load_pc_encoder(OPENSHAPE_MODEL)
    finally:
        torch.load = _load
    head = nn.Sequential(nn.LayerNorm(BACKBONE_DIM), nn.Linear(BACKBONE_DIM, out_dim))  # new head into the GeoRSCLIP space
    nn.init.zeros_(head[1].weight)
    nn.init.zeros_(head[1].bias)
    model.proj = head
    fp32_forward(model.ppat.sa)
    return model.cpu()


# ---------------------------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------------------------
def make_scheduler(opt, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE summed over the embedding dimensions, mean over the batch (= MSE * dim)."""
    return ((pred.float() - target) ** 2).sum(1).mean()


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.ppat.parameters():
        p.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> dict:
    model.eval()
    preds, targets = [], []
    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(pts)
        preds.append(out.float().cpu())
        targets.append(batch["target"])
    pred, tgt = torch.cat(preds), torch.cat(targets)
    mse = loss_fn(pred, tgt).item()
    pn, tn = F.normalize(pred, dim=1), F.normalize(tgt, dim=1)
    own = (pn * tn).sum(1)
    # rank of the own target among all targets, in chunks (n x n similarity)
    ranks = []
    for s in range(0, len(pn), 1024):
        sim = pn[s : s + 1024] @ tn.T
        own_s = own[s : s + 1024].unsqueeze(1)
        ranks.append((sim > own_s).sum(1))
    ranks = torch.cat(ranks)
    n = len(pn)
    return {
        "val_mse": mse,
        "val_cos_own": own.mean().item(),
        "val_r1": (ranks < 1).float().mean().item(),
        "val_r5": (ranks < 5).float().mean().item(),
        "val_r10": (ranks < 10).float().mean().item(),
        "val_median_rank": int(ranks.median().item()) + 1,
        "val_n": n,
        "val_r1_chance": 1 / n,
        "pred_norm_mean": pred.norm(dim=1).mean().item(),
    }


def save(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None, help="output folder name (default: timestamp)")
    ap.add_argument("--out-dir", default=str(MODELS_DIR), help="parent folder of the run folders")
    ap.add_argument("--points-dir", default=str(POINTS_DIR))
    ap.add_argument("--emb", default=str(EMB_FILE))
    ap.add_argument("--openshape-dir", default=str(OPENSHAPE_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5, help="learning rate of the backbone")
    ap.add_argument("--head-lr", type=float, default=1e-4, help="learning rate of the new head")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0, help="linear LR warmup, then cosine decay")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1, help="train only the new head first")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--rotate", default="none", choices=ROTATIONS, help="train augmentation (see dataset.py)")
    ap.add_argument("--jitter-m", type=float, default=0.0)
    ap.add_argument("--n-points", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=None, help="only the first N train patches (tests)")
    ap.add_argument("--limit-val", type=int, default=None, help="only the first N val patches (tests)")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp == "auto":
        amp = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type == "cuda" else "off")
    else:
        amp = args.amp
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[amp]

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_name
    if run_dir.exists() and (run_dir / "last.pt").exists() and not args.resume:
        sys.exit(f"{run_dir} already has a checkpoint; use --resume or another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    # data
    ds_kw = dict(points_dir=args.points_dir, emb_file=args.emb, n_points=args.n_points, seed=args.seed)
    train_ds = PatchPointDataset("train", rotate=args.rotate, jitter_m=args.jitter_m, limit=args.limit_train, **ds_kw)
    val_ds = PatchPointDataset("val", limit=args.limit_val, **ds_kw)
    print(f"train: {train_ds.stats}\nval:   {val_ds.stats}")
    # reference: predict the mean train target for every patch (DOP embeddings share a large common
    # direction, so this already gives a high cosine; the model must beat it to have learned anything)
    mean_t = train_ds.targets.mean(0, keepdim=True)
    mean_baseline = {
        "val_mse_mean_vector": loss_fn(mean_t.expand_as(val_ds.targets), val_ds.targets).item(),
        "val_cos_mean_vector": F.cosine_similarity(mean_t, val_ds.targets).mean().item(),
    }
    print(f"mean-vector reference on val: loss {mean_baseline['val_mse_mean_vector']:.4f}, "
          f"cos {mean_baseline['val_cos_mean_vector']:.3f}, r@1 = chance")
    pin = device.type == "cuda"
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    # model
    model = build_model(Path(args.openshape_dir), out_dim=train_ds.targets.shape[1]).to(device)
    head_params = list(model.proj.parameters())
    backbone_params = list(model.ppat.parameters())
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_ds) // args.batch_size
    if steps_per_epoch == 0:
        sys.exit("fewer train patches than one batch")
    sched = make_scheduler(opt, int(args.warmup_epochs * steps_per_epoch), args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    n_params = sum(p.numel() for p in model.parameters())

    meta = {
        "model": "OpenShape PointBERT (PointPatchTransformer) + new linear head",
        "model_version": f"{OPENSHAPE_MODEL} (huggingface.co/OpenShape/{OPENSHAPE_MODEL}), loader openshape-demo-support",
        "head": f"LayerNorm({BACKBONE_DIM}) + Linear({BACKBONE_DIM}, {train_ds.targets.shape[1]}), new, linear zero-initialised",
        "n_params": n_params,
        **train_ds.metadata(),
        "loss": "MSE to the L2-normalised target, summed over dimensions (= MSE * dim = |y - t|^2)",
        "train_patches": len(train_ds),
        "val_patches": len(val_ds),
        "split_source": "data/patches/patch_index.parquet (make_split.py)",
        "amp": amp,
        **mean_baseline,
    }
    config = {"args": vars(args), "meta": meta, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "torch": torch.__version__, "device": torch.cuda.get_device_name() if pin else "cpu"}

    start_epoch, best = 0, {"val_r1": -1.0, "epoch": None}
    if args.resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']}, best val_r1 {best['val_r1']:.4f} (epoch {best['epoch']})")
    else:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"device {config['device']}, amp {amp}, {n_params / 1e6:.1f} M parameters, "
          f"{steps_per_epoch} steps per epoch, run folder {run_dir}")

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.freeze_backbone_epochs
        set_backbone_trainable(model, not frozen)
        train_ds.set_epoch(epoch)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        model.train()
        t0, losses, n_seen = time.time(), [], 0
        for step, batch in enumerate(train_dl):
            pts = batch["points"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(pts)
            loss = loss_fn(out, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            n_seen += len(pts)
            if step % 50 == 0 or step == steps_per_epoch - 1:
                el = time.time() - t0
                print(f"  epoch {epoch} step {step + 1}/{steps_per_epoch} loss {np.mean(losses[-50:]):.4f} "
                      f"| {n_seen / el:.1f} patches/s", flush=True)
        t_train = time.time() - t0

        t1 = time.time()
        val = evaluate(model, val_dl, device, amp_dtype)
        rec = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "train_mse": float(np.mean(losses)),
            **val,
            "lr_backbone": opt.param_groups[0]["lr"],
            "lr_head": opt.param_groups[1]["lr"],
            "train_seconds": round(t_train, 1),
            "val_seconds": round(time.time() - t1, 1),
        }
        with open(run_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch}: train loss {rec['train_mse']:.4f} | val loss {val['val_mse']:.4f} "
              f"cos {val['val_cos_own']:.3f} r@1 {val['val_r1']:.3f} r@5 {val['val_r5']:.3f} "
              f"r@10 {val['val_r10']:.3f} median rank {val['val_median_rank']}/{val['val_n']} "
              f"| {t_train / 60:.1f} + {rec['val_seconds'] / 60:.1f} min", flush=True)

        if val["val_r1"] > best["val_r1"]:
            best = {"val_r1": val["val_r1"], "epoch": epoch}
            save(run_dir / "best.pt", {"model": model.state_dict(), "meta": meta, "epoch": epoch, "val": val})
            print(f"  new best -> {run_dir / 'best.pt'}")
        save(run_dir / "last.pt", {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                   "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                                   "epoch": epoch, "best": best, "meta": meta})

    print(f"done. best val r@1 {best['val_r1']:.4f} in epoch {best['epoch']} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
