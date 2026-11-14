"""Loaders for the CLIP-like backbones that are compared on the DOP patches.

Every backbone has a family (code + weights source) and one or more variants
(model size / checkpoint). A backbone provides:

    preprocess(PIL image) -> tensor      image transform of that model
    encode_image(batch)   -> (B, D)      L2-normalised float32 on the CPU  (mode "image")
    encode_text(list[str])-> (n, D)      L2-normalised float32 on the CPU  (mode "text")

Image and text embeddings of one search must come from the same backbone, so the
embedding files store family and variant and search_dop.py reloads exactly those.

Family      Code                          Weights
openclip    open_clip                     LAION-2B pretrained tags (open_clip downloads them)
georsclip   open_clip                     Zilun/GeoRSCLIP (HF), RS5M checkpoints
remoteclip  open_clip                     chendelong/RemoteCLIP (HF)
skyclip     open_clip                     SkyScript S3 bucket (zip, downloaded to data/cache/weights/)
dofaclip    DOFA-CLIP fork of open_clip   earthflow/GeoLB-ViT-B-16-SigLIP-All-EO (HF)

The licences of the pretrained weights have NOT been checked yet (CLAUDE.md rule 8).
"""

import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "hf"))
WEIGHTS_DIR = ROOT / "data" / "cache" / "weights"

import open_clip
import torch
import torch.nn.functional as F

# variant -> (open_clip architecture, pretrained tag)
OPENCLIP = {
    "vitb32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "vitl14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "vith14": ("ViT-H-14", "laion2b_s32b_b79k"),
}
# variant -> (open_clip architecture, checkpoint name RS5M_<name>.pt). The ViT-B-32 checkpoint
# was fine-tuned from the OpenAI weights, which use QuickGELU.
GEORSCLIP = {
    "vith14": ("ViT-H-14", "ViT-H-14"),
    "vitb32": ("ViT-B-32-quickgelu", "ViT-B-32"),
}
# variant -> open_clip architecture (checkpoint RemoteCLIP-<arch>.pt)
REMOTECLIP = {
    "vitl14": "ViT-L-14",
    "vitb32": "ViT-B-32",
}
# variant -> (open_clip architecture, zip name on the SkyScript bucket, force QuickGELU).
# The ViT-L-14 models were initialised from the OpenAI weights (see the SkyScript notebooks).
SKYCLIP_URL = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/{name}.zip"
SKYCLIP = {
    "vitl14_top30pct": ("ViT-L-14", "SkyCLIP_ViT_L14_top30pct", True),
    "vitl14_top50pct": ("ViT-L-14", "SkyCLIP_ViT_L14_top50pct", True),
    "vitb32_top50pct": ("ViT-B-32", "SkyCLIP_ViT_B32_top50pct", False),
}
# variant -> Hugging Face repo with an open_clip config
DOFACLIP = {"vitb16": "hf-hub:earthflow/GeoLB-ViT-B-16-SigLIP-All-EO"}
DOFA_WAVELENGTHS_UM = [0.665, 0.560, 0.490]  # R, G, B centre wavelengths as in the DOFA-CLIP demo


@dataclass
class Backbone:
    key: str
    family: str
    variant: str
    info: dict = field(default_factory=dict)
    preprocess: Optional[Callable] = None
    encode_image: Optional[Callable] = None
    encode_text: Optional[Callable] = None


def _base_info() -> dict:
    return {"clip_library": "open_clip", "open_clip_version": open_clip.__version__}


def _image_fn(forward: Callable, device: str, dtype: torch.dtype) -> Callable:
    @torch.inference_mode()
    def encode(x: torch.Tensor) -> torch.Tensor:
        feats = forward(x.to(device=device, dtype=dtype, non_blocking=True))
        return F.normalize(feats.float(), dim=-1).cpu()

    return encode


def _text_fn(encode_text: Callable, tokenizer: Callable) -> Callable:
    @torch.inference_mode()
    def encode(texts: list[str]) -> torch.Tensor:
        return F.normalize(encode_text(tokenizer(texts)).float(), dim=-1)

    return encode


def _finish(family, variant, model, preprocess, tokenizer, info, mode, device) -> Backbone:
    """Wrap a standard open_clip CLIP model. Image mode only moves the image tower to the
    device, in fp16 on CUDA, so that ViT-H/14 fits into 2 GB of GPU memory."""
    bb = Backbone(f"{family}_{variant}", family, variant, info, preprocess)
    if mode == "image":
        dtype = torch.float16 if device == "cuda" else torch.float32
        visual = model.visual.to(device=device, dtype=dtype).eval()
        bb.encode_image = _image_fn(visual, device, dtype)
    else:
        model.eval()
        bb.encode_text = _text_fn(model.encode_text, tokenizer)
    return bb


def _numpy_safe_globals() -> list:
    """Numpy types that appear in training checkpoints (e.g. SkyCLIP's saved metrics)."""
    import numpy as np

    core = getattr(np, "_core", None) or np.core
    return [(core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType]


def _load_state_dict(model: torch.nn.Module, path: str, strict_unexpected: bool) -> None:
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or (strict_unexpected and unexpected):
        raise RuntimeError(
            f"{path} does not fit the model: {len(missing)} missing keys {missing[:5]}, "
            f"{len(unexpected)} unexpected keys {unexpected[:5]}"
        )
    if unexpected:
        print(f"note: ignored {len(unexpected)} unexpected checkpoint keys, e.g. {unexpected[:3]}")


def _hf_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def load_openclip(variant: str, mode: str, device: str) -> Backbone:
    arch, tag = OPENCLIP[variant]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
    info = {**_base_info(), "clip_arch": arch, "clip_pretrained": tag,
            "weights_source": f"open_clip pretrained tag {tag}"}
    return _finish("openclip", variant, model, preprocess, open_clip.get_tokenizer(arch), info, mode, device)


def load_georsclip(variant: str, mode: str, device: str) -> Backbone:
    arch, name = GEORSCLIP[variant]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None)
    path = _hf_file("Zilun/GeoRSCLIP", f"ckpt/RS5M_{name}.pt")
    _load_state_dict(model, path, strict_unexpected=False)
    info = {**_base_info(), "clip_arch": arch, "clip_pretrained": f"RS5M_{name}.pt",
            "weights_source": "huggingface.co/Zilun/GeoRSCLIP"}
    return _finish("georsclip", variant, model, preprocess, open_clip.get_tokenizer(arch), info, mode, device)


def load_remoteclip(variant: str, mode: str, device: str) -> Backbone:
    arch = REMOTECLIP[variant]
    model, _, preprocess = open_clip.create_model_and_transforms(arch)
    path = _hf_file("chendelong/RemoteCLIP", f"RemoteCLIP-{arch}.pt")
    _load_state_dict(model, path, strict_unexpected=True)
    info = {**_base_info(), "clip_arch": arch, "clip_pretrained": f"RemoteCLIP-{arch}.pt",
            "weights_source": "huggingface.co/chendelong/RemoteCLIP"}
    return _finish("remoteclip", variant, model, preprocess, open_clip.get_tokenizer(arch), info, mode, device)


def _skyclip_checkpoint(name: str) -> Path:
    out_dir = WEIGHTS_DIR / "skyclip" / name
    found = sorted(out_dir.rglob("*.pt"))
    if not found:
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir.with_suffix(".zip")
        part = zip_path.with_name(zip_path.name + ".part")
        url = SKYCLIP_URL.format(name=name)
        print(f"downloading {url} (several GB)")

        def progress(blocks, block_size, total):
            if blocks % 2000 == 0:
                print(f"  {blocks * block_size / 1e9:.1f} / {total / 1e9:.1f} GB")

        urllib.request.urlretrieve(url, part, progress)
        os.replace(part, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir.parent)
        zip_path.unlink()
        found = sorted(out_dir.rglob("*.pt"))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one .pt file in {out_dir}, found {found}")
    return found[0]


def load_skyclip(variant: str, mode: str, device: str) -> Backbone:
    arch, name, quick_gelu = SKYCLIP[variant]
    path = _skyclip_checkpoint(name)
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None, force_quick_gelu=quick_gelu)
    _load_state_dict(model, str(path), strict_unexpected=True)
    info = {**_base_info(), "clip_arch": arch, "clip_pretrained": f"{name}/{path.name}",
            "weights_source": SKYCLIP_URL.format(name=name)}
    return _finish("skyclip", variant, model, preprocess, open_clip.get_tokenizer(arch), info, mode, device)


def load_dofaclip(variant: str, mode: str, device: str) -> Backbone:
    repo = DOFACLIP[variant]
    try:
        model, preprocess = open_clip.create_model_from_pretrained(repo)
    except Exception as e:
        if "dofa" in str(e).lower():
            raise RuntimeError(
                "DOFA-CLIP needs the DOFA-CLIP fork of open_clip, which replaces the normal open_clip. "
                "Use a separate environment:  git clone https://github.com/xiong-zhitong/DOFA-CLIP  and  "
                "pip install -e DOFA-CLIP/open_clip transformers sentencepiece"
            ) from e
        raise
    tokenizer = open_clip.get_tokenizer(repo)
    info = {**_base_info(), "clip_arch": "dofa_vit_base_patch16_siglip_224", "clip_pretrained": repo,
            "weights_source": "huggingface.co/earthflow/GeoLB-ViT-B-16-SigLIP-All-EO",
            "wavelengths_um": DOFA_WAVELENGTHS_UM, "note": "DOFA-CLIP fork of open_clip"}
    bb = Backbone(f"dofaclip_{variant}", "dofaclip", variant, info, preprocess)
    if mode == "image":
        trunk = model.visual.trunk.to(device).eval()
        wvs = torch.tensor(DOFA_WAVELENGTHS_UM, device=device)
        bb.encode_image = _image_fn(lambda x: trunk(x, wvs)[0], device, torch.float32)
    else:
        model.eval()
        bb.encode_text = _text_fn(
            model.encode_text, lambda texts: tokenizer(texts, context_length=model.context_length)
        )
    return bb


FAMILIES = {
    "openclip": (load_openclip, OPENCLIP),
    "georsclip": (load_georsclip, GEORSCLIP),
    "remoteclip": (load_remoteclip, REMOTECLIP),
    "skyclip": (load_skyclip, SKYCLIP),
    "dofaclip": (load_dofaclip, DOFACLIP),
}


def variants(family: str) -> list[str]:
    return list(FAMILIES[family][1])


def default_batch_size(variant: str) -> int:
    return 32 if variant.startswith(("vitl14", "vith14")) else 128


def load(family: str, variant: str, mode: str, device: str) -> Backbone:
    """mode 'image' prepares encode_image on `device`; mode 'text' prepares encode_text on the CPU."""
    if family not in FAMILIES or variant not in FAMILIES[family][1]:
        raise ValueError(f"unknown backbone {family}/{variant}")
    if mode not in ("image", "text"):
        raise ValueError("mode must be 'image' or 'text'")
    return FAMILIES[family][0](variant, mode, device)
